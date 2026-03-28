from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import pandas as pd

from sim.sitl.session import (
    BARO_COLUMNS,
    CANONICAL_STREAM_RATES_HZ,
    DEVICE_EVENT_COLUMNS,
    ESTIMATOR_FEEDBACK_COLUMNS,
    GPS_COLUMNS,
    IMU_COLUMNS,
    MAG_COLUMNS,
    OPTIONAL_STREAM_KEYS,
    REQUIRED_STREAM_KEYS,
    ReplaySession,
    SESSION_SCHEMA_VERSION,
    SESSION_STREAM_SPECS,
    TRUTH_COLUMNS,
    build_session_manifest,
    find_latest_session_manifest,
    load_replay_session,
    load_manifest_schema,
    manifest_schema_path,
    merge_replay_session_sensors,
    session_spec_path,
    validate_session_manifest,
)


class SitlSessionContractTests(unittest.TestCase):
    def test_stream_specs_freeze_required_and_optional_contract(self) -> None:
        self.assertEqual(
            tuple(SESSION_STREAM_SPECS.keys()),
            (
                "truth",
                "imu",
                "baro",
                "gps",
                "mag",
                "estimator_feedback",
                "device_events",
            ),
        )
        self.assertEqual(REQUIRED_STREAM_KEYS, ("truth", "imu", "baro", "gps"))
        self.assertEqual(
            OPTIONAL_STREAM_KEYS,
            ("mag", "estimator_feedback", "device_events"),
        )

        self.assertEqual(SESSION_STREAM_SPECS["truth"].filename, "truth.csv")
        self.assertEqual(SESSION_STREAM_SPECS["truth"].rate_hz, 500)
        self.assertEqual(SESSION_STREAM_SPECS["truth"].columns, TRUTH_COLUMNS)

        self.assertEqual(SESSION_STREAM_SPECS["imu"].filename, "imu.csv")
        self.assertEqual(SESSION_STREAM_SPECS["imu"].rate_hz, 300)
        self.assertEqual(SESSION_STREAM_SPECS["imu"].columns, IMU_COLUMNS)

        self.assertEqual(SESSION_STREAM_SPECS["baro"].filename, "baro.csv")
        self.assertEqual(SESSION_STREAM_SPECS["baro"].rate_hz, 100)
        self.assertEqual(SESSION_STREAM_SPECS["baro"].columns, BARO_COLUMNS)

        self.assertEqual(SESSION_STREAM_SPECS["gps"].filename, "gps.csv")
        self.assertEqual(SESSION_STREAM_SPECS["gps"].rate_hz, 10)
        self.assertEqual(SESSION_STREAM_SPECS["gps"].columns, GPS_COLUMNS)

        self.assertEqual(SESSION_STREAM_SPECS["mag"].filename, "mag.csv")
        self.assertEqual(SESSION_STREAM_SPECS["mag"].rate_hz, 50)
        self.assertEqual(SESSION_STREAM_SPECS["mag"].columns, MAG_COLUMNS)

        self.assertEqual(
            SESSION_STREAM_SPECS["estimator_feedback"].columns,
            ESTIMATOR_FEEDBACK_COLUMNS,
        )
        self.assertIsNone(SESSION_STREAM_SPECS["estimator_feedback"].rate_hz)
        self.assertEqual(
            SESSION_STREAM_SPECS["device_events"].columns,
            DEVICE_EVENT_COLUMNS,
        )
        self.assertIsNone(SESSION_STREAM_SPECS["device_events"].rate_hz)

        self.assertEqual(
            CANONICAL_STREAM_RATES_HZ,
            {
                "truth": 500,
                "imu": 300,
                "baro": 100,
                "gps": 10,
                "mag": 50,
            },
        )

    def test_build_session_manifest_uses_canonical_filenames_and_rates(self) -> None:
        manifest = build_session_manifest(
            session_id="260313_190556",
            vehicle_name="Itzamna",
            generated_at_utc="2026-03-27T18:45:00Z",
            reference_latitude_deg=33.4986251,
            reference_longitude_deg=-99.3376125,
            reference_altitude_m=417.0,
            sea_level_pressure_pa=101325.0,
            include_optional_streams=("mag",),
        )

        self.assertEqual(manifest["schema_version"], SESSION_SCHEMA_VERSION)
        self.assertEqual(
            manifest["streams"],
            {
                "truth": "truth.csv",
                "imu": "imu.csv",
                "baro": "baro.csv",
                "gps": "gps.csv",
                "mag": "mag.csv",
            },
        )
        self.assertEqual(
            manifest["rates_hz"],
            {
                "truth": 500,
                "imu": 300,
                "baro": 100,
                "gps": 10,
                "mag": 50,
            },
        )

    def test_validate_session_manifest_accepts_required_only_manifest(self) -> None:
        manifest = build_session_manifest(
            session_id="260313_190556",
            vehicle_name="Itzamna",
            generated_at_utc="2026-03-27T18:45:00Z",
            reference_latitude_deg=33.4986251,
            reference_longitude_deg=-99.3376125,
            reference_altitude_m=417.0,
            sea_level_pressure_pa=101325.0,
        )

        validate_session_manifest(manifest)

    def test_validate_session_manifest_rejects_unknown_stream(self) -> None:
        manifest = build_session_manifest(
            session_id="260313_190556",
            vehicle_name="Itzamna",
            generated_at_utc="2026-03-27T18:45:00Z",
            reference_latitude_deg=33.4986251,
            reference_longitude_deg=-99.3376125,
            reference_altitude_m=417.0,
            sea_level_pressure_pa=101325.0,
        )
        manifest["streams"]["mystery"] = "mystery.csv"

        with self.assertRaisesRegex(ValueError, "unsupported streams"):
            validate_session_manifest(manifest)

    def test_validate_session_manifest_rejects_noncanonical_filename_or_rate(self) -> None:
        manifest = build_session_manifest(
            session_id="260313_190556",
            vehicle_name="Itzamna",
            generated_at_utc="2026-03-27T18:45:00Z",
            reference_latitude_deg=33.4986251,
            reference_longitude_deg=-99.3376125,
            reference_altitude_m=417.0,
            sea_level_pressure_pa=101325.0,
        )

        manifest["streams"]["truth"] = "subdir/truth.csv"
        with self.assertRaisesRegex(ValueError, "canonical filename"):
            validate_session_manifest(manifest)

        manifest["streams"]["truth"] = "truth.csv"
        manifest["rates_hz"]["imu"] = 200
        with self.assertRaisesRegex(ValueError, "must be 300 Hz"):
            validate_session_manifest(manifest)

    def test_contract_artifacts_exist_and_pin_schema_constants(self) -> None:
        self.assertTrue(manifest_schema_path().exists())
        self.assertTrue(session_spec_path().exists())

        schema = load_manifest_schema()
        self.assertEqual(schema["properties"]["schema_version"]["const"], SESSION_SCHEMA_VERSION)
        self.assertEqual(
            schema["properties"]["streams"]["required"],
            ["truth", "imu", "baro", "gps"],
        )
        self.assertEqual(schema["properties"]["streams"]["properties"]["truth"]["const"], "truth.csv")
        self.assertEqual(schema["properties"]["rates_hz"]["properties"]["truth"]["const"], 500)

    def test_load_replay_session_accepts_manifest_or_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = _write_test_session(Path(temp_dir))

            from_manifest = load_replay_session(session_dir / "manifest.json")
            from_directory = load_replay_session(session_dir)

        self.assertIsInstance(from_manifest, ReplaySession)
        self.assertEqual(from_manifest.session_dir.name, "session_260313_190556")
        self.assertEqual(from_manifest.manifest_path.name, "manifest.json")
        self.assertEqual(from_manifest.truth.columns.tolist(), list(TRUTH_COLUMNS))
        self.assertEqual(from_manifest.imu.columns.tolist(), list(IMU_COLUMNS))
        self.assertEqual(from_manifest.baro.columns.tolist(), list(BARO_COLUMNS))
        self.assertEqual(from_manifest.gps.columns.tolist(), list(GPS_COLUMNS))
        self.assertEqual(from_directory.manifest["session_id"], "260313_190556")

    def test_merge_replay_session_sensors_builds_offline_compatibility_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = _write_test_session(Path(temp_dir))
            replay_session = load_replay_session(session_dir)

        merged = merge_replay_session_sensors(replay_session)

        self.assertEqual(
            merged.columns.tolist(),
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
        self.assertEqual(merged["time_s"].round(6).tolist(), [0.0, 0.003333, 0.006667, 0.01, 0.1])

    def test_find_latest_session_manifest_and_default_loader_use_latest_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            _write_test_session(logs_dir, session_id="260313_190556")
            latest_dir = _write_test_session(logs_dir, session_id="260313_190557")

            manifest_path = find_latest_session_manifest(logs_dir)
            replay_session = load_replay_session(logs_directory=logs_dir)

        self.assertEqual(manifest_path.parent, latest_dir)
        self.assertEqual(replay_session.session_dir, latest_dir)
        self.assertEqual(replay_session.manifest["session_id"], "260313_190557")

    def test_load_replay_session_rejects_missing_stream_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = _write_test_session(Path(temp_dir))
            (session_dir / "gps.csv").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "stream 'gps'"):
                load_replay_session(session_dir)

    def test_load_replay_session_rejects_non_monotonic_stream_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = _write_test_session(
                Path(temp_dir),
                imu_times=[0.0, 0.004, 0.003],
            )

            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_replay_session(session_dir)


def _write_test_session(
    logs_dir: Path,
    *,
    session_id: str = "260313_190556",
    imu_times: list[float] | None = None,
) -> Path:
    session_dir = logs_dir / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_session_manifest(
        session_id=session_id,
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
            "time_s": imu_times or [0.0, 1 / 300, 2 / 300],
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
