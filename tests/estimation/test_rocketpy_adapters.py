from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from sim.estimation.adapters import (
    GravityAlignmentFlightPhasePolicy,
    RocketFlightPhase,
    RocketFlightPhaseDetector,
    RocketPyReplayConfig,
    estimate_sea_level_pressure_pa,
    find_latest_matching_log_pair,
    find_latest_telemetry_log,
    geodetic_to_local_enu,
    pressure_to_altitude_m,
    run_rocketpy_replay,
)
from sim.sitl.session import build_session_manifest
from sim.estimation.stacks import LayeredNavigationState


class RocketFlightPhasePolicyTests(unittest.TestCase):
    def test_detector_transitions_through_powered_coast_and_descent(self) -> None:
        detector = RocketFlightPhaseDetector()

        self.assertEqual(
            detector.update(
                accelerometer_magnitude_mps2=9.80665,
                vertical_velocity_mps=0.0,
            ),
            RocketFlightPhase.ON_PAD,
        )
        self.assertEqual(
            detector.update(
                accelerometer_magnitude_mps2=35.0,
                vertical_velocity_mps=20.0,
            ),
            RocketFlightPhase.POWERED_ASCENT,
        )
        self.assertEqual(
            detector.update(
                accelerometer_magnitude_mps2=9.9,
                vertical_velocity_mps=20.0,
            ),
            RocketFlightPhase.COAST,
        )
        self.assertEqual(
            detector.update(
                accelerometer_magnitude_mps2=9.9,
                vertical_velocity_mps=-4.0,
            ),
            RocketFlightPhase.DESCENT,
        )

    def test_policy_skips_gravity_updates_only_during_powered_ascent(self) -> None:
        policy = GravityAlignmentFlightPhasePolicy()
        gravity_measurement = np.array([0.0, 0.0, -9.80665], dtype=float)

        on_pad = policy.evaluate(
            accelerometer_mps2=gravity_measurement,
            vertical_velocity_mps=0.0,
        )
        powered = policy.evaluate(
            accelerometer_mps2=np.array([0.0, 0.0, -30.0], dtype=float),
            vertical_velocity_mps=30.0,
        )
        coast = policy.evaluate(
            accelerometer_mps2=gravity_measurement,
            vertical_velocity_mps=10.0,
        )

        self.assertTrue(on_pad.submit_update)
        self.assertEqual(on_pad.phase, RocketFlightPhase.ON_PAD)
        self.assertFalse(powered.submit_update)
        self.assertEqual(powered.phase, RocketFlightPhase.POWERED_ASCENT)
        self.assertTrue(coast.submit_update)
        self.assertEqual(coast.phase, RocketFlightPhase.COAST)


class RocketPyReplayAdapterTests(unittest.TestCase):
    def test_pressure_and_geodetic_conversions_are_consistent(self) -> None:
        sea_level_pressure_pa = 101325.0
        altitude_m = 1250.0
        pressure_pa = sea_level_pressure_pa * (1.0 - altitude_m / 44330.0) ** 5.255

        recovered_altitude_m = pressure_to_altitude_m(pressure_pa, sea_level_pressure_pa)
        recovered_sea_level_pressure_pa = estimate_sea_level_pressure_pa(pressure_pa, altitude_m)

        self.assertAlmostEqual(recovered_altitude_m, altitude_m, places=6)
        self.assertAlmostEqual(recovered_sea_level_pressure_pa, sea_level_pressure_pa, places=6)

        origin = (35.0, -97.0, 400.0)
        sample = (35.0005, -96.9990, 412.0)
        local_enu = geodetic_to_local_enu(
            latitude_deg=sample[0],
            longitude_deg=sample[1],
            altitude_m=sample[2],
            origin_latitude_deg=origin[0],
            origin_longitude_deg=origin[1],
            origin_altitude_m=origin[2],
        )

        expected_east_m = (
            np.deg2rad(sample[1] - origin[1])
            * np.cos(np.deg2rad(0.5 * (sample[0] + origin[0])))
            * 6378137.0
        )
        expected_north_m = np.deg2rad(sample[0] - origin[0]) * 6378137.0

        self.assertAlmostEqual(float(local_enu[0]), expected_east_m, places=8)
        self.assertAlmostEqual(float(local_enu[1]), expected_north_m, places=8)
        self.assertAlmostEqual(float(local_enu[2]), 12.0, places=9)

    def test_replay_reports_phase_and_measurement_updates(self) -> None:
        telemetry = pd.DataFrame(
            {
                "time_s": [0.0, 0.1, 0.2],
                "accelerometer_x": [0.0, 0.0, 0.0],
                "accelerometer_y": [0.0, 0.0, 0.0],
                "accelerometer_z": [-9.80665, -30.0, -9.80665],
                "gyroscope_x": [0.0, 0.0, 0.0],
                "gyroscope_y": [0.0, 0.0, 0.0],
                "gyroscope_z": [0.0, 0.0, 0.0],
                "gnss_x": [0.0, 1.0, 2.0],
                "gnss_y": [0.0, 0.0, 0.0],
                "gnss_z": [0.0, 0.0, 0.0],
                "barometer_v1": [101325.0, 101315.0, 101305.0],
            }
        )
        config = RocketPyReplayConfig(
            gnss_is_geodetic=False,
            sea_level_pressure_pa=101325.0,
            derive_parameters_from_telemetry=False,
            initial_state=LayeredNavigationState(),
        )

        result = run_rocketpy_replay(telemetry, config=config)
        estimates = result.estimates

        self.assertEqual(estimates["flight_phase"].tolist(), ["ON_PAD", "POWERED_ASCENT", "COAST"])
        self.assertEqual(estimates["gravity_alignment_submitted"].tolist(), [True, False, True])
        self.assertEqual(estimates["gravity_alignment_update"].tolist(), [True, False, True])
        self.assertEqual(estimates["gnss_position_update"].tolist(), [True, True, True])
        self.assertEqual(estimates["gnss_velocity_update"].tolist(), [False, True, True])
        self.assertEqual(estimates["baro_update"].tolist(), [True, True, True])
        self.assertEqual(estimates["navigation_update_label"].tolist(), ["gps_position", "gps_velocity", "gps_velocity"])
        self.assertEqual(estimates["navigation_update_status"].tolist(), ["accepted", "accepted", "accepted"])

    def test_replay_tracks_constant_velocity_gnss_trace(self) -> None:
        """New adapter replay converges toward a constant-velocity GNSS trace.

        This replaces the old legacy-parity test.  With the legacy monolithic
        estimator removed, the test validates that ``run_rocketpy_replay``
        produces position estimates that track the GNSS input to within a
        reasonable tolerance.
        """
        telemetry = pd.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0, 3.0],
                "accelerometer_x": [0.0, 0.0, 0.0, 0.0],
                "accelerometer_y": [0.0, 0.0, 0.0, 0.0],
                "accelerometer_z": [0.0, 0.0, 0.0, 0.0],
                "gyroscope_x": [0.0, 0.0, 0.0, 0.0],
                "gyroscope_y": [0.0, 0.0, 0.0, 0.0],
                "gyroscope_z": [0.0, 0.0, 0.0, 0.0],
                "gnss_x": [0.0, 10.0, 20.0, 30.0],
                "gnss_y": [0.0, 0.0, 0.0, 0.0],
                "gnss_z": [0.0, 0.0, 0.0, 0.0],
            }
        )

        result = run_rocketpy_replay(
            telemetry,
            config=RocketPyReplayConfig(
                barometer_column=None,
                gnss_is_geodetic=False,
                derive_parameters_from_telemetry=False,
                initial_state=LayeredNavigationState(
                    quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                    position_m=np.zeros(3, dtype=float),
                ),
            ),
        )

        est_x = result.estimates["est_x_m"].to_numpy(dtype=float)
        # The filter should track the constant-velocity GNSS input reasonably.
        # Final estimate should be within 5 m of the GNSS position at t=3.
        self.assertAlmostEqual(est_x[-1], 30.0, delta=5.0)
        # Monotonically increasing x estimates after the first step.
        for i in range(1, len(est_x)):
            self.assertGreater(est_x[i], est_x[i - 1])

    def test_replay_accepts_session_directory_for_offline_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = _write_test_session(Path(temp_dir))
            result = run_rocketpy_replay(
                session_dir,
                config=RocketPyReplayConfig(
                    derive_parameters_from_telemetry=False,
                    initial_state=LayeredNavigationState(),
                ),
            )

        self.assertEqual(result.telemetry_path, session_dir / "manifest.json")
        self.assertEqual(result.estimates["time_s"].round(6).tolist(), [0.0, 0.003333, 0.006667, 0.01, 0.1])
        self.assertEqual(result.estimates["baro_update"].tolist(), [True, False, False, True, False])
        self.assertEqual(result.estimates["gnss_position_update"].tolist(), [True, False, False, False, True])

    def test_legacy_log_discovery_functions_warn_while_still_working(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            sensor_log = logs_dir / "virtual_sensors_full_rate_260313_190556.csv"
            kinematics_log = logs_dir / "flight_kinematics_260313_190556.csv"
            pd.DataFrame({"time_s": [0.0]}).to_csv(sensor_log, index=False)
            pd.DataFrame({"time_s": [0.0]}).to_csv(kinematics_log, index=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                latest_sensor_log = find_latest_telemetry_log(logs_dir)
                matched_sensor_log, matched_kinematics_log = find_latest_matching_log_pair(logs_dir)

        self.assertEqual(latest_sensor_log, sensor_log)
        self.assertEqual(matched_sensor_log, sensor_log)
        self.assertEqual(matched_kinematics_log, kinematics_log)
        self.assertGreaterEqual(len(caught), 2)
        self.assertTrue(all(issubclass(item.category, DeprecationWarning) for item in caught))

    def test_legacy_log_discovery_materializes_session_compatibility_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            session_dir = _write_test_session(logs_dir)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                latest_sensor_log = find_latest_telemetry_log(logs_dir)
                matched_sensor_log, matched_kinematics_log = find_latest_matching_log_pair(logs_dir)

            self.assertEqual(latest_sensor_log, matched_sensor_log)
            self.assertEqual(latest_sensor_log.name, "virtual_sensors_full_rate_260313_190556.csv")
            self.assertEqual(matched_kinematics_log, logs_dir / "flight_kinematics_260313_190556.csv")
            self.assertTrue(latest_sensor_log.exists())
            self.assertTrue(matched_kinematics_log.exists())
            self.assertEqual(
                pd.read_csv(latest_sensor_log).columns.tolist(),
                [
                    "time_s",
                    "accelerometer_x",
                    "accelerometer_y",
                    "accelerometer_z",
                    "gyroscope_x",
                    "gyroscope_y",
                    "gyroscope_z",
                    "barometer_v1",
                    "gnss_x",
                    "gnss_y",
                    "gnss_z",
                ],
            )
            self.assertEqual(
                pd.read_csv(matched_kinematics_log).columns.tolist(),
                [
                    "time_s",
                    "x_m",
                    "y_m",
                    "z_m",
                    "vx_mps",
                    "vy_mps",
                    "vz_mps",
                    "e0",
                    "e1",
                    "e2",
                    "e3",
                    "w1_radps",
                    "w2_radps",
                    "w3_radps",
                ],
            )
            self.assertEqual(session_dir.name, "session_260313_190556")
            self.assertGreaterEqual(len(caught), 2)
            self.assertTrue(all(issubclass(item.category, DeprecationWarning) for item in caught))


def _write_test_session(logs_dir: Path) -> Path:
    session_dir = logs_dir / "session_260313_190556"
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_session_manifest(
        session_id="260313_190556",
        vehicle_name="Itzamna",
        generated_at_utc="2026-03-27T18:45:00Z",
        reference_latitude_deg=33.4986251,
        reference_longitude_deg=-99.3376125,
        reference_altitude_m=417.0,
        sea_level_pressure_pa=101325.0,
    )
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    pd.DataFrame(
        {
            "time_s": [0.0, 0.002, 0.004],
            "x_m": [0.0, 0.1, 0.2],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [417.0, 417.1, 417.2],
            "vx_mps": [10.0, 10.0, 10.0],
            "vy_mps": [0.0, 0.0, 0.0],
            "vz_mps": [20.0, 20.0, 20.0],
            "e0": [1.0, 1.0, 1.0],
            "e1": [0.0, 0.0, 0.0],
            "e2": [0.0, 0.0, 0.0],
            "e3": [0.0, 0.0, 0.0],
            "w1_radps": [0.1, 0.1, 0.1],
            "w2_radps": [0.2, 0.2, 0.2],
            "w3_radps": [0.3, 0.3, 0.3],
        }
    ).to_csv(session_dir / "truth.csv", index=False)

    pd.DataFrame(
        {
            "time_s": [0.0, 1 / 300, 2 / 300],
            "accelerometer_x": [0.0, 0.0, 0.0],
            "accelerometer_y": [0.0, 0.0, 0.0],
            "accelerometer_z": [-9.80665, -9.80665, -9.80665],
            "gyroscope_x": [0.0, 0.0, 0.0],
            "gyroscope_y": [0.0, 0.0, 0.0],
            "gyroscope_z": [0.0, 0.0, 0.0],
        }
    ).to_csv(session_dir / "imu.csv", index=False)

    pd.DataFrame(
        {
            "time_s": [0.0, 0.01],
            "barometer_v1": [96453.7, 96453.9],
        }
    ).to_csv(session_dir / "baro.csv", index=False)

    pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "gnss_x": [33.4986538, 33.4987538],
            "gnss_y": [-99.3375871, -99.3374871],
            "gnss_z": [413.9, 414.1],
        }
    ).to_csv(session_dir / "gps.csv", index=False)

    return session_dir


if __name__ == "__main__":
    unittest.main()
