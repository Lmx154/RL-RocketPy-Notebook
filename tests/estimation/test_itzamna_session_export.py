from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

import pandas as pd


def _load_module(module_name: str, relative_path: str):
    repository_root = Path(__file__).resolve().parents[2]
    module_path = repository_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ItzamnaSessionExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.telemetry_logger = _load_module(
            "itzamna_telemetry_logger",
            "notebooks/Itzamna/telemetry_logger.py",
        )
        cls.rocket_config = _load_module(
            "itzamna_rocket_config",
            "notebooks/Itzamna/rocket_config.py",
        )

    def test_rocket_config_uses_phase2_sampling_rates(self) -> None:
        self.assertEqual(self.rocket_config.TRUTH_SAMPLING_RATE, 500)
        self.assertEqual(self.rocket_config.IMU_SAMPLING_RATE, 300)
        self.assertEqual(self.rocket_config.BARO_SAMPLING_RATE, 100)
        self.assertEqual(self.rocket_config.GNSS_SAMPLING_RATE, 10)

    def test_export_telemetry_writes_session_directory_and_manifest(self) -> None:
        flight = _FakeFlight(include_mag=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = self.telemetry_logger.export_telemetry(flight, logs_dir=temp_dir)
            session_path = Path(session_dir)
            logs_path = Path(temp_dir)

            self.assertTrue((session_path / "manifest.json").exists())
            self.assertTrue((session_path / "truth.csv").exists())
            self.assertTrue((session_path / "imu.csv").exists())
            self.assertTrue((session_path / "baro.csv").exists())
            self.assertTrue((session_path / "gps.csv").exists())
            self.assertFalse((session_path / "mag.csv").exists())
            self.assertTrue((logs_path / "virtual_sensors_full_rate_260313_190556.csv").exists())
            self.assertTrue((logs_path / "flight_kinematics_260313_190556.csv").exists())

            manifest = json.loads((session_path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["session_id"], "260313_190556")
            self.assertEqual(manifest["vehicle_name"], "Itzamna")
            self.assertEqual(
                manifest["streams"],
                {
                    "truth": "truth.csv",
                    "imu": "imu.csv",
                    "baro": "baro.csv",
                    "gps": "gps.csv",
                },
            )
            self.assertEqual(
                manifest["rates_hz"],
                {
                    "truth": 500,
                    "imu": 300,
                    "baro": 100,
                    "gps": 10,
                },
            )
            self.assertAlmostEqual(manifest["reference_latitude_deg"], 33.4986251)
            self.assertAlmostEqual(manifest["reference_longitude_deg"], -99.3376125)
            self.assertAlmostEqual(manifest["reference_altitude_m"], 417.0)
            self.assertGreater(manifest["sea_level_pressure_pa"], 100000.0)

            truth = pd.read_csv(session_path / "truth.csv")
            self.assertEqual(
                truth.columns.tolist(),
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
            self.assertEqual(truth["time_s"].round(6).tolist(), [0.0, 0.002, 0.004])

            imu = pd.read_csv(session_path / "imu.csv")
            self.assertEqual(
                imu.columns.tolist(),
                [
                    "time_s",
                    "accelerometer_x",
                    "accelerometer_y",
                    "accelerometer_z",
                    "gyroscope_x",
                    "gyroscope_y",
                    "gyroscope_z",
                ],
            )
            self.assertEqual(imu["time_s"].round(6).tolist(), [0.0, 0.003333, 0.006667])

            legacy_sensor = pd.read_csv(logs_path / "virtual_sensors_full_rate_260313_190556.csv")
            self.assertEqual(
                legacy_sensor.columns.tolist(),
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

            legacy_truth = pd.read_csv(logs_path / "flight_kinematics_260313_190556.csv")
            self.assertEqual(legacy_truth.columns.tolist(), truth.columns.tolist())
            self.assertEqual(legacy_truth["time_s"].round(6).tolist(), [0.0, 0.002, 0.004])

    def test_export_telemetry_writes_optional_mag_stream_when_present(self) -> None:
        flight = _FakeFlight(include_mag=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = self.telemetry_logger.export_telemetry(flight, logs_dir=temp_dir)
            session_path = Path(session_dir)
            logs_path = Path(temp_dir)

            self.assertTrue((session_path / "mag.csv").exists())
            manifest = json.loads((session_path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["streams"]["mag"], "mag.csv")
            self.assertEqual(manifest["rates_hz"]["mag"], 50)

            mag = pd.read_csv(session_path / "mag.csv")
            self.assertEqual(
                mag.columns.tolist(),
                ["time_s", "magnetometer_x", "magnetometer_y", "magnetometer_z"],
            )

            legacy_sensor = pd.read_csv(logs_path / "virtual_sensors_full_rate_260313_190556.csv")
            self.assertEqual(
                legacy_sensor.columns.tolist(),
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
                    "magnetometer_x",
                    "magnetometer_y",
                    "magnetometer_z",
                ],
            )

    def test_export_telemetry_collapses_duplicate_sensor_timestamps(self) -> None:
        flight = _FakeFlight(include_mag=False, include_duplicate_imu_timestamps=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = self.telemetry_logger.export_telemetry(flight, logs_dir=temp_dir)
            session_path = Path(session_dir)

            imu = pd.read_csv(session_path / "imu.csv")

            self.assertEqual(imu["time_s"].round(6).tolist(), [0.0, 0.003333, 0.006667])
            self.assertEqual(len(imu), 3)
            duplicate_row = imu.iloc[1]
            self.assertAlmostEqual(duplicate_row["accelerometer_x"], 9.99)
            self.assertAlmostEqual(duplicate_row["gyroscope_x"], 0.91)


class _FakeEnvironment:
    date = (2026, 3, 13, 19, 5, 56)
    latitude = 33.4986251
    longitude = -99.3376125
    elevation = 417.0


class _FakeRocket:
    name = "Itzamna"


def _make_sensor(class_name: str, name: str, measured_data: list[tuple[float, ...]]):
    sensor_type = type(class_name, (), {})
    sensor = sensor_type()
    sensor.name = name
    sensor.measured_data = measured_data
    return sensor


class _FakeFlight:
    def __init__(
        self,
        *,
        include_mag: bool,
        include_duplicate_imu_timestamps: bool = False,
    ) -> None:
        self.environment = _FakeEnvironment()
        self.rocket = _FakeRocket()
        accelerometer_data = [
            (0.0, 0.1, 0.2, -9.7),
            (1 / 300, 0.11, 0.21, -9.6),
            (2 / 300, 0.12, 0.22, -9.5),
        ]
        gyroscope_data = [
            (0.0, 0.01, 0.02, 0.03),
            (1 / 300, 0.011, 0.021, 0.031),
            (2 / 300, 0.012, 0.022, 0.032),
        ]
        if include_duplicate_imu_timestamps:
            accelerometer_data.insert(2, (1 / 300, 9.99, 8.88, 7.77))
            gyroscope_data.insert(2, (1 / 300, 0.91, 0.81, 0.71))

        self.t_final = 0.004
        self.sensors = [
            _make_sensor(
                "Accelerometer",
                "Itzamna Accelerometer",
                accelerometer_data,
            ),
            _make_sensor(
                "Gyroscope",
                "Itzamna Gyroscope",
                gyroscope_data,
            ),
            _make_sensor(
                "Barometer",
                "Itzamna Barometer",
                [
                    (0.0, 96453.7),
                    (0.01, 96453.9),
                ],
            ),
            _make_sensor(
                "GnssReceiver",
                "Itzamna GNSS",
                [
                    (0.0, 33.4986538, -99.3375871, 413.9),
                    (0.1, 33.4987538, -99.3374871, 414.1),
                ],
            ),
        ]
        if include_mag:
            self.sensors.append(
                _make_sensor(
                    "Magnetometer",
                    "Itzamna Magnetometer",
                    [
                        (0.0, 0.4, 0.1, -0.2),
                        (0.02, 0.41, 0.11, -0.21),
                    ],
                )
            )

    def x(self, time_s: float) -> float:
        return 10.0 * time_s

    def y(self, time_s: float) -> float:
        return -5.0 * time_s

    def z(self, time_s: float) -> float:
        return 417.0 + (20.0 * time_s)

    def vx(self, time_s: float) -> float:
        return 10.0

    def vy(self, time_s: float) -> float:
        return -5.0

    def vz(self, time_s: float) -> float:
        return 20.0

    def e0(self, time_s: float) -> float:
        return 1.0

    def e1(self, time_s: float) -> float:
        return 0.0

    def e2(self, time_s: float) -> float:
        return 0.0

    def e3(self, time_s: float) -> float:
        return 0.0

    def w1(self, time_s: float) -> float:
        return 0.1 + time_s

    def w2(self, time_s: float) -> float:
        return 0.2 + time_s

    def w3(self, time_s: float) -> float:
        return 0.3 + time_s

    def latitude(self, time_s: float) -> float:
        return 33.4986251

    def longitude(self, time_s: float) -> float:
        return -99.3376125


if __name__ == "__main__":
    unittest.main()
