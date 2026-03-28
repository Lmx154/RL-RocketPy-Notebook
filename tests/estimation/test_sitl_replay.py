from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import pandas as pd

from sim.sitl.replay import ReplayClock, load_replay_telemetry
from sim.sitl.session import build_session_manifest


class SitlReplayTests(unittest.TestCase):
    def test_load_replay_telemetry_accepts_session_directory_and_latest_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            _write_test_session(logs_dir, session_id="260313_190556")
            latest_dir = _write_test_session(logs_dir, session_id="260313_190557")

            from_session_dir, session_path = load_replay_telemetry(
                latest_dir,
                logs_directory=logs_dir,
            )
            from_latest, latest_path = load_replay_telemetry(
                None,
                logs_directory=logs_dir,
            )

        self.assertEqual(session_path, latest_dir / "manifest.json")
        self.assertEqual(latest_path, latest_dir / "manifest.json")
        self.assertEqual(from_session_dir["time_s"].round(6).tolist(), [0.0, 0.003333, 0.006667, 0.01, 0.1])
        self.assertEqual(from_latest["time_s"].round(6).tolist(), [0.0, 0.003333, 0.006667, 0.01, 0.1])

    def test_replay_clock_tracks_session_derived_rows_and_normalizes_sparse_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = _write_test_session(Path(temp_dir))
            telemetry, telemetry_path = load_replay_telemetry(
                session_dir,
                logs_directory=Path(temp_dir),
            )

        clock = ReplayClock(telemetry)

        self.assertEqual(telemetry_path, session_dir / "manifest.json")
        self.assertEqual(clock.current_time_s(), 0.0)
        self.assertAlmostEqual(clock.dt_to_next_s(), 1 / 300, places=6)

        clock.sync_to_time(0.0034)
        self.assertEqual(clock.index, 2)
        self.assertAlmostEqual(clock.current_time_s(), 2 / 300, places=6)

        clock.seek_index(1)
        sample = clock.current_sample()
        json_row = sample.json_row()

        self.assertEqual(sample.index, 1)
        self.assertAlmostEqual(sample.time_s, 1 / 300, places=6)
        self.assertIsNone(json_row["barometer_v1"])
        self.assertIsNone(json_row["gnss_x"])

        clock.step(100)
        self.assertTrue(clock.at_end)
        self.assertEqual(clock.snapshot()["total_samples"], 5)

        clock.reset()
        self.assertEqual(clock.index, 0)


def _write_test_session(
    logs_dir: Path,
    *,
    session_id: str = "260313_190556",
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
