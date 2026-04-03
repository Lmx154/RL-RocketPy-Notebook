from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from sim.simulation import CsvReplayController
from sim.sitl.replay_session import ReplaySessionScheduler
from sim.sitl.session import (
    build_session_manifest,
    load_replay_session,
    merge_replay_session_sensors,
)


class ReplaySessionSchedulerTests(unittest.TestCase):
    def test_scheduler_advances_sensor_streams_only_when_due(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = load_replay_session(_write_test_session(Path(temp_dir)))

        scheduler = ReplaySessionScheduler(session)

        state0 = scheduler.current_state()
        self.assertEqual(state0["time"], 0.0)
        self.assertEqual(state0["sensors"]["accelerometer_x"], 0.1)
        self.assertTrue(state0["sensor_freshness"]["accelerometer_x"])
        self.assertTrue(state0["sensor_freshness"]["barometer_v1"])
        self.assertTrue(state0["sensor_freshness"]["gnss_x"])
        self.assertAlmostEqual(state0["position"]["altitude"], 413.9)

        state1 = scheduler.advance_one_tick()
        self.assertEqual(state1["time"], 0.002)
        self.assertEqual(state1["sensors"]["accelerometer_x"], 0.1)
        self.assertFalse(state1["sensor_freshness"]["accelerometer_x"])
        self.assertFalse(state1["sensor_freshness"]["barometer_v1"])
        self.assertFalse(state1["sensor_freshness"]["gnss_x"])
        self.assertAlmostEqual(state1["position"]["altitude"], 413.9)

        state2 = scheduler.advance_one_tick()
        self.assertEqual(state2["time"], 0.004)
        self.assertAlmostEqual(state2["sensors"]["accelerometer_x"], 0.11)
        self.assertTrue(state2["sensor_freshness"]["accelerometer_x"])
        self.assertFalse(state2["sensor_freshness"]["barometer_v1"])
        self.assertFalse(state2["sensor_freshness"]["gnss_x"])

    def test_scheduler_seek_rebuilds_stream_cursors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = load_replay_session(_write_test_session(Path(temp_dir)))

        scheduler = ReplaySessionScheduler(session)
        scheduler.advance_one_tick()
        scheduler.advance_one_tick()

        rewind = scheduler.seek_truth_index(0)
        self.assertEqual(rewind["step_index"], 0)
        self.assertEqual(rewind["sensors"]["accelerometer_x"], 0.1)
        self.assertTrue(rewind["sensor_freshness"]["accelerometer_x"])
        self.assertTrue(rewind["sensor_freshness"]["barometer_v1"])
        self.assertTrue(rewind["sensor_freshness"]["gnss_x"])

        jump = scheduler.seek_truth_index(2)
        self.assertEqual(jump["step_index"], 2)
        self.assertAlmostEqual(jump["sensors"]["accelerometer_x"], 0.11)
        self.assertTrue(jump["sensor_freshness"]["accelerometer_x"])
        self.assertFalse(jump["sensor_freshness"]["barometer_v1"])
        self.assertFalse(jump["sensor_freshness"]["gnss_x"])


class CsvReplayControllerTests(unittest.TestCase):
    def test_controller_uses_multi_rate_scheduler_with_session_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = load_replay_session(_write_test_session(Path(temp_dir)))

        controller = CsvReplayController(session, update_rate=120.0)

        state = controller.get_state_at_index(2)
        self.assertEqual(controller.total_steps, 3)
        self.assertEqual(controller.index, 2)
        self.assertAlmostEqual(state["time"], 0.004)
        self.assertAlmostEqual(state["sensors"]["accelerometer_x"], 0.11)
        self.assertTrue(state["sensor_freshness"]["accelerometer_x"])
        self.assertFalse(state["sensor_freshness"]["barometer_v1"])

        controller.seek(1, is_progress=False)
        self.assertEqual(controller.index, 1)
        self.assertAlmostEqual(controller.get_time_info()["current_time"], 0.002)

    def test_controller_interpolates_position_from_truth_velocity(self) -> None:
        truth = pd.DataFrame(
            {
                "time_s": [0.0, 1.0],
                "x_m": [0.0, 2.0],
                "y_m": [0.0, 0.0],
                "z_m": [417.0, 417.0],
                "vx_mps": [0.0, 4.0],
                "vy_mps": [0.0, 0.0],
                "vz_mps": [0.0, 0.0],
                "e0": [1.0, 1.0],
                "e1": [0.0, 0.0],
                "e2": [0.0, 0.0],
                "e3": [0.0, 0.0],
                "w1_radps": [0.0, 0.0],
                "w2_radps": [0.0, 0.0],
                "w3_radps": [0.0, 0.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            session = load_replay_session(
                _write_custom_session(Path(temp_dir), truth=truth)
            )

        controller = CsvReplayController(session, update_rate=120.0)
        state = controller.get_state_at_time(0.5)

        self.assertAlmostEqual(state["time"], 0.5)
        self.assertAlmostEqual(state["position"]["x"], 0.5, places=6)
        self.assertAlmostEqual(state["velocity"]["vx"], 2.0, places=6)

    def test_controller_interpolates_attitude_between_truth_rows(self) -> None:
        quarter_turn = 0.5 ** 0.5
        truth = pd.DataFrame(
            {
                "time_s": [0.0, 1.0],
                "x_m": [0.0, 0.0],
                "y_m": [0.0, 0.0],
                "z_m": [417.0, 417.0],
                "vx_mps": [0.0, 0.0],
                "vy_mps": [0.0, 0.0],
                "vz_mps": [0.0, 0.0],
                "e0": [1.0, quarter_turn],
                "e1": [0.0, 0.0],
                "e2": [0.0, 0.0],
                "e3": [0.0, quarter_turn],
                "w1_radps": [0.0, 0.0],
                "w2_radps": [0.0, 0.0],
                "w3_radps": [np.pi / 2.0, np.pi / 2.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            session = load_replay_session(
                _write_custom_session(Path(temp_dir), truth=truth)
            )

        controller = CsvReplayController(session, update_rate=120.0)
        state = controller.get_state_at_time(0.5)

        self.assertAlmostEqual(state["quaternion"]["e0"], np.cos(np.pi / 8.0), places=6)
        self.assertAlmostEqual(state["quaternion"]["e3"], np.sin(np.pi / 8.0), places=6)
        self.assertAlmostEqual(state["angular_velocity"]["w3"], np.pi / 2.0, places=6)

    def test_session_merge_helper_builds_offline_compatibility_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = load_replay_session(_write_test_session(Path(temp_dir)))

        sensors = merge_replay_session_sensors(session)

        self.assertEqual(
            sensors.columns.tolist(),
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
        self.assertEqual(sensors["time_s"].round(6).tolist(), [0.0, 0.003333, 0.006667, 0.01, 0.1])


def _write_test_session(logs_dir: Path) -> Path:
    return _write_custom_session(logs_dir, truth=_default_truth_frame())


def _default_truth_frame() -> pd.DataFrame:
    return pd.DataFrame(
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
    )


def _write_custom_session(
    logs_dir: Path,
    *,
    truth: pd.DataFrame,
) -> Path:
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

    truth.to_csv(session_dir / "truth.csv", index=False)

    pd.DataFrame(
        {
            "time_s": [0.0, 1 / 300, 2 / 300],
            "accelerometer_x": [0.1, 0.11, 0.12],
            "accelerometer_y": [0.2, 0.21, 0.22],
            "accelerometer_z": [-9.7, -9.6, -9.5],
            "gyroscope_x": [0.01, 0.011, 0.012],
            "gyroscope_y": [0.02, 0.021, 0.022],
            "gyroscope_z": [0.03, 0.031, 0.032],
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
