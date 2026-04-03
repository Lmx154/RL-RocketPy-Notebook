from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

import numpy as np
import pandas as pd
from sim.sitl.session import TRUTH_COLUMNS, load_replay_session


def _load_module(module_name: str, relative_path: str):
    repository_root = Path(__file__).resolve().parents[1]
    module_path = repository_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class LonestarTelemetryToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module("lonestar_telemetry", "tools/lonestar_telemetry.py")

    def test_normalize_featherweight_converts_units_and_events(self) -> None:
        csv_text = textwrap.dedent(
            """\
            Flight_Time_(s),Temperature_(F),Baro_Press_(atm),Baro_Altitude_ASL_(feet),Baro_Altitude_AGL_(feet),Batt_Volts,Velocity_Up,Velocity_DR,Velocity_CR,Inertial_Altitude,Inertial_DR_Position,Inertial_CR_position,Tilt_Angle_(deg),Future_Angle_(deg),Roll_Angle_(deg),Liftoff,Apogee,Press_Increasing,Burnout_Coast,Apo_fired,Main_fired,3rd_fired,4th_fired,Normal_Ascent,Accel_Vel_LE_0,ECI_Vvel_le_0,Tilt Exceeded 90deg
            0.00,68.0,1.0,1000.0,100.0,4.1,10.0,5.0,-2.0,900.0,40.0,-10.0,3.0,4.0,5.0,1,0,1,0,0,0,0,0,1,0,1,0
            0.02,69.8,0.99,1001.0,101.0,4.0,11.0,6.0,-1.0,901.0,41.0,-11.0,6.0,7.0,8.0,1,1,0,1,1,0,0,0,0,1,0,1
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "featherweight.csv"
            source_path.write_text(csv_text, encoding="utf-8")

            normalized = self.module.normalize_featherweight_csv(source_path)

            self.assertEqual(normalized.source_name, "featherweight")
            baro = normalized.streams["baro"]
            navigation = normalized.streams["navigation"]
            events = normalized.streams["events"]

            self.assertAlmostEqual(baro.loc[0, "pressure_pa"], 101325.0)
            self.assertAlmostEqual(baro.loc[0, "altitude_agl_m"], 30.48)
            self.assertAlmostEqual(baro.loc[0, "temperature_c"], 20.0)
            self.assertAlmostEqual(navigation.loc[1, "vertical_velocity_mps"], 11.0 * 0.3048)
            self.assertEqual(events["liftoff"].tolist(), [True, True])
            self.assertEqual(events["apogee"].tolist(), [False, True])
            self.assertEqual(events["tilt_exceeded_90deg"].tolist(), [False, True])

    def test_normalize_marv_collapses_duplicate_baro_samples(self) -> None:
        csv_text = textwrap.dedent(
            """\
            log_us,sink_state,imu_state,imu_sample_us,imu_lagged,imu_ax_mps2,imu_ay_mps2,imu_az_mps2,imu_gx_rad_s,imu_gy_rad_s,imu_gz_rad_s,aux_imu_state,aux_imu_sample_us,aux_imu_lagged,aux_imu_ax_mps2,aux_imu_ay_mps2,aux_imu_az_mps2,aux_imu_gx_rad_s,aux_imu_gy_rad_s,aux_imu_gz_rad_s,baro_state,baro_sample_us,baro_lagged,baro_pressure_pa,baro_temp_c
            1000000,healthy,fresh,999000,0,1.0,2.0,3.0,0.1,0.2,0.3,fresh,999500,0,4.0,5.0,6.0,0.4,0.5,0.6,missing,,0,,
            1010000,healthy,fresh,1009000,0,1.1,2.1,3.1,0.11,0.21,0.31,fresh,1009500,0,4.1,5.1,6.1,0.41,0.51,0.61,fresh,1008000,0,97000,20.0
            1020000,healthy,fresh,1019000,0,1.2,2.2,3.2,0.12,0.22,0.32,fresh,1019500,0,4.2,5.2,6.2,0.42,0.52,0.62,stale,1008000,0,97000,20.0
            1030000,healthy,fresh,1029000,1,1.3,2.3,3.3,0.13,0.23,0.33,fresh,1029500,0,4.3,5.3,6.3,0.43,0.53,0.63,fresh,1028000,0,96980,21.0
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "marv.csv"
            source_path.write_text(csv_text, encoding="utf-8")

            normalized = self.module.normalize_marv_csv(source_path)

            self.assertEqual(normalized.source_name, "marv")
            self.assertEqual(normalized.metadata["common_origin_us"], 999000)

            primary_imu = normalized.streams["primary_imu"]
            aux_imu = normalized.streams["aux_imu"]
            baro = normalized.streams["baro"]

            self.assertEqual(len(primary_imu), 4)
            self.assertEqual(len(aux_imu), 4)
            self.assertEqual(len(baro), 2)
            self.assertEqual(baro["state"].tolist(), ["fresh", "fresh"])
            self.assertAlmostEqual(primary_imu.loc[0, "time_s"], 0.0)
            self.assertAlmostEqual(baro.loc[0, "time_s"], 0.009)
            self.assertEqual(primary_imu["lagged"].tolist(), [False, False, False, True])

    def test_normalize_featherweight_gps_csv_converts_units(self) -> None:
        csv_text = textwrap.dedent(
            """\
            UTCTIME,UNIXTIME,ALT,LAT,LON,#SATS,FIX,HORZV,VERTV,HEAD,FLAGS,>40,>32,>24,RSSI,BATT,Altitude AGL,Launch detection,Apogee detection,Landing detection,Distance (feet)
            Mar 30 2026 13:47:10.400 UTC,1774896430.4,1000.0,33.0,-99.0,12,3,100.0,200.0,90.0,1,2,3,4,-55,4.1,10.0,FALSE,FALSE,FALSE,0.0
            Mar 30 2026 13:47:10.500 UTC,1774896430.5,1010.0,33.1,-99.1,13,3,110.0,210.0,95.0,1,3,4,5,-54,4.0,20.0,TRUE,FALSE,FALSE,100.0
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "gps.csv"
            source_path.write_text(csv_text, encoding="utf-8")

            normalized = self.module.normalize_featherweight_gps_csv(source_path)

            self.assertEqual(normalized.source_name, "featherweight_gps")
            navigation = normalized.streams["navigation"]
            events = normalized.streams["events"]

            self.assertAlmostEqual(navigation.loc[0, "time_s"], 0.0)
            self.assertAlmostEqual(navigation.loc[1, "time_s"], 0.1, places=6)
            self.assertAlmostEqual(navigation.loc[0, "gps_altitude_asl_m"], 1000.0 * 0.3048)
            self.assertAlmostEqual(navigation.loc[1, "gps_altitude_agl_m"], 20.0 * 0.3048)
            self.assertAlmostEqual(navigation.loc[0, "horizontal_speed_mps"], 100.0 * 0.3048)
            self.assertEqual(events["launch_detected"].tolist(), [False, True])

    def test_isolate_marv_flight_window_finds_active_segment(self) -> None:
        baro_time = np.arange(0.0, 120.0, 0.05)
        pressure = np.full_like(baro_time, 97100.0)
        active_mask = (baro_time >= 20.0) & (baro_time <= 80.0)
        active_time = baro_time[active_mask]
        active_profile = np.interp(
            active_time,
            [20.0, 35.0, 50.0, 65.0, 80.0],
            [97100.0, 86000.0, 70000.0, 84000.0, 97100.0],
        )
        pressure[active_mask] = active_profile
        baro = pd.DataFrame(
            {
                "time_s": baro_time,
                "log_time_s": baro_time,
                "sample_time_us": (baro_time * 1_000_000).astype(np.int64),
                "log_time_us": (baro_time * 1_000_000).astype(np.int64),
                "state": "fresh",
                "lagged": False,
                "pressure_pa": pressure,
                "temperature_c": 25.0,
            }
        )

        imu_time = np.arange(0.0, 120.0, 0.01)
        ax = np.zeros_like(imu_time)
        ay = np.zeros_like(imu_time)
        az = np.full_like(imu_time, 9.81)
        imu_active_mask = (imu_time >= 20.0) & (imu_time <= 80.0)
        az[imu_active_mask] = 25.0 + 12.0 * np.sin((imu_time[imu_active_mask] - 20.0) * 0.2)
        ax[imu_active_mask] = 8.0 * np.sin((imu_time[imu_active_mask] - 20.0) * 0.7)

        primary_imu = pd.DataFrame(
            {
                "time_s": imu_time,
                "log_time_s": imu_time,
                "sample_time_us": (imu_time * 1_000_000).astype(np.int64),
                "log_time_us": (imu_time * 1_000_000).astype(np.int64),
                "state": "fresh",
                "lagged": False,
                "accelerometer_x_mps2": ax,
                "accelerometer_y_mps2": ay,
                "accelerometer_z_mps2": az,
                "gyroscope_x_rad_s": 0.0,
                "gyroscope_y_rad_s": 0.0,
                "gyroscope_z_rad_s": 0.0,
            }
        )

        source = self.module.NormalizedSource(
            source_name="marv",
            source_path=Path("synthetic_marv.csv"),
            metadata={"flight_window_isolated": False},
            streams={
                "primary_imu": primary_imu,
                "aux_imu": primary_imu.copy(),
                "baro": baro,
            },
        )

        isolated, report = self.module.isolate_marv_flight_window(
            source,
            margin_s=2.0,
            min_segment_duration_s=10.0,
        )

        self.assertFalse(report["used_full_range"])
        self.assertGreaterEqual(report["start_time_s"], 17.0)
        self.assertLessEqual(report["start_time_s"], 22.5)
        self.assertGreaterEqual(report["end_time_s"], 77.5)
        self.assertLessEqual(report["end_time_s"], 83.5)
        self.assertAlmostEqual(
            isolated.metadata["flight_window_start_time_s"],
            report["start_time_s"],
        )
        self.assertGreater(len(isolated.streams["baro"]), 100)

    def test_align_baro_sources_recovers_scaled_offset(self) -> None:
        marv_time = np.arange(0.0, 140.0, 0.05)
        marv_pressure = np.interp(
            marv_time,
            [0.0, 10.0, 20.0, 35.0, 50.0, 70.0, 90.0, 110.0, 140.0],
            [97100.0, 96900.0, 94000.0, 79000.0, 69000.0, 72000.0, 81000.0, 90000.0, 97200.0],
        )
        marv_baro = pd.DataFrame(
            {
                "time_s": marv_time,
                "log_time_s": marv_time,
                "sample_time_us": (marv_time * 1_000_000).astype(np.int64),
                "log_time_us": (marv_time * 1_000_000).astype(np.int64),
                "state": "fresh",
                "lagged": False,
                "pressure_pa": marv_pressure,
                "temperature_c": 24.0,
            }
        )

        featherweight_time = np.arange(0.0, 45.0, 0.05)
        known_offset_s = 32.4
        marv_window_pressure = np.interp(
            featherweight_time + known_offset_s,
            marv_time,
            marv_pressure,
        )
        featherweight_pressure = 85000.0 + 0.22 * (marv_window_pressure - marv_window_pressure[0])
        featherweight_pressure = np.round(featherweight_pressure / 20.0) * 20.0
        featherweight_baro = pd.DataFrame(
            {
                "time_s": featherweight_time,
                "pressure_pa": featherweight_pressure,
                "pressure_atm": featherweight_pressure / 101325.0,
                "altitude_asl_m": 0.0,
                "altitude_agl_m": 0.0,
                "temperature_c": 20.0,
                "temperature_f": 68.0,
                "battery_v": 4.0,
            }
        )

        featherweight_source = self.module.NormalizedSource(
            source_name="featherweight",
            source_path=Path("synthetic_fw.csv"),
            metadata={},
            streams={
                "baro": featherweight_baro,
                "navigation": featherweight_baro[["time_s"]].copy(),
                "events": featherweight_baro[["time_s"]].copy(),
            },
        )
        marv_source = self.module.NormalizedSource(
            source_name="marv",
            source_path=Path("synthetic_marv.csv"),
            metadata={},
            streams={
                "baro": marv_baro,
                "primary_imu": marv_baro[["time_s"]].copy(),
                "aux_imu": marv_baro[["time_s"]].copy(),
            },
        )

        report, featherweight_debug, marv_debug = self.module.align_baro_sources(
            featherweight_source,
            marv_source,
            coarse_step_s=0.1,
            refine_step_s=0.01,
            edge_guard_s=0.5,
            featherweight_smoothing_window_s=0.5,
            marv_smoothing_window_s=0.8,
        )

        self.assertAlmostEqual(report["offset_s"], known_offset_s, delta=0.15)
        self.assertGreater(report["derivative_correlation"], 0.9)
        self.assertIn("alignment_weight", featherweight_debug.columns)
        self.assertIn("dpdt_z", marv_debug.columns)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["can_merge"])

    def test_featherweight_baro_quality_detects_duplicate_runs(self) -> None:
        time_s = np.arange(0.0, 8.0, 0.1)
        block = np.linspace(100500.0, 98500.0, 25)
        altitude_block = np.linspace(120.0, 40.0, 25)
        pressure = np.concatenate(
            [
                np.full(3, 120000.0),
                block,
                block,
                np.linspace(98700.0, 99000.0, len(time_s) - 53),
            ]
        )
        altitude_agl_m = np.concatenate(
            [
                np.full(3, 500.0),
                altitude_block,
                altitude_block,
                np.linspace(39.0, 30.0, len(time_s) - 53),
            ]
        )
        frame = pd.DataFrame(
            {
                "time_s": time_s,
                "pressure_pa": pressure,
                "altitude_agl_m": altitude_agl_m,
                "temperature_c": 20.0,
                "battery_v": 4.0,
            }
        )

        quality = self.module.analyze_featherweight_baro_alignment_quality(
            frame,
            duplicate_lag_min_s=2.0,
            duplicate_lag_max_s=3.0,
            min_duplicate_run_s=2.0,
        )

        self.assertEqual(quality["status"], "ambiguous_repeated_segments")
        self.assertGreaterEqual(quality["leading_outlier_rows"], 3)
        self.assertGreater(quality["duplicate_sample_fraction"], 0.5)
        self.assertGreater(len(quality["top_duplicate_lags"]), 0)
        self.assertAlmostEqual(quality["top_duplicate_lags"][0]["lag_s"], 2.5, delta=0.11)

    def test_merge_aligned_sources_resamples_streams_on_shared_timebase(self) -> None:
        featherweight_baro = pd.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0],
                "pressure_pa": [10.0, 20.0, 30.0],
                "temperature_c": [1.0, 2.0, 3.0],
            }
        )
        featherweight_navigation = pd.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0],
                "vertical_velocity_mps": [0.0, 1.0, 2.0],
            }
        )
        featherweight_events = pd.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0],
                "main_fired": [False, True, True],
            }
        )
        featherweight_source = self.module.NormalizedSource(
            source_name="featherweight",
            source_path=Path("synthetic_fw.csv"),
            metadata={},
            streams={
                "baro": featherweight_baro,
                "navigation": featherweight_navigation,
                "events": featherweight_events,
            },
        )

        marv_baro = pd.DataFrame(
            {
                "time_s": [1.0, 2.0, 3.0],
                "pressure_pa": [100.0, 200.0, 300.0],
                "temperature_c": [10.0, 11.0, 12.0],
            }
        )
        marv_primary_imu = pd.DataFrame(
            {
                "time_s": [1.0, 2.0, 3.0],
                "accelerometer_x_mps2": [1.0, 2.0, 3.0],
                "lagged": [False, False, True],
            }
        )
        marv_aux_imu = pd.DataFrame(
            {
                "time_s": [1.0, 2.0, 3.0],
                "accelerometer_x_mps2": [4.0, 5.0, 6.0],
                "lagged": [False, True, True],
            }
        )
        marv_source = self.module.NormalizedSource(
            source_name="marv",
            source_path=Path("synthetic_marv.csv"),
            metadata={},
            streams={
                "baro": marv_baro,
                "primary_imu": marv_primary_imu,
                "aux_imu": marv_aux_imu,
            },
        )

        merged, summary = self.module.merge_aligned_sources(
            featherweight_source,
            marv_source,
            offset_s=1.0,
            time_step_s=1.0,
            timebase_source="shared_uniform",
        )

        self.assertEqual(merged["time_s"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(merged["featherweight_source_time_s"].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(merged["featherweight_baro_pressure_pa"].tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(merged["marv_baro_pressure_pa"].tolist(), [100.0, 200.0, 300.0])
        self.assertEqual(merged["featherweight_events_main_fired"].astype(bool).tolist(), [False, True, True])
        self.assertEqual(merged["marv_primary_imu_lagged"].astype(bool).tolist(), [False, False, True])
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["time_start_s"], 1.0)
        self.assertEqual(summary["time_end_s"], 3.0)
        self.assertEqual(summary["timebase"]["source"], "shared_uniform")

    def test_merge_aligned_sources_can_preserve_marv_primary_imu_timeline(self) -> None:
        gps_navigation = pd.DataFrame(
            {
                "time_s": [0.0, 0.2, 0.4],
                "gps_altitude_agl_m": [0.0, 20.0, 40.0],
                "latitude_deg": [33.0, 33.0001, 33.0002],
                "longitude_deg": [-99.0, -99.0001, -99.0002],
            }
        )
        gps_events = pd.DataFrame(
            {
                "time_s": [0.0, 0.2, 0.4],
                "launch_detected": [False, True, True],
            }
        )
        gps_source = self.module.NormalizedSource(
            source_name="featherweight_gps",
            source_path=Path("synthetic_gps.csv"),
            metadata={},
            streams={
                "navigation": gps_navigation,
                "events": gps_events,
            },
        )

        marv_primary_imu = pd.DataFrame(
            {
                "time_s": [1.05, 1.17, 1.31],
                "accelerometer_x_mps2": [1.0, 2.0, 3.0],
                "lagged": [False, False, True],
            }
        )
        marv_baro = pd.DataFrame(
            {
                "time_s": [1.0, 1.2, 1.4],
                "pressure_pa": [100.0, 90.0, 80.0],
            }
        )
        marv_source = self.module.NormalizedSource(
            source_name="marv",
            source_path=Path("synthetic_marv.csv"),
            metadata={},
            streams={
                "baro": marv_baro,
                "primary_imu": marv_primary_imu,
                "aux_imu": marv_primary_imu.copy(),
            },
        )

        merged, summary = self.module.merge_aligned_sources(
            gps_source,
            marv_source,
            offset_s=1.0,
            timebase_source="marv_primary_imu",
        )

        self.assertEqual(merged["time_s"].tolist(), [1.05, 1.17, 1.31])
        self.assertEqual(summary["timebase"]["source"], "marv_primary_imu")
        self.assertEqual(summary["timebase"]["stream"], "primary_imu")
        self.assertTrue(np.allclose(merged["featherweight_source_time_s"], [0.05, 0.17, 0.31]))
        self.assertTrue(
            np.allclose(
                merged["featherweight_navigation_gps_altitude_agl_m"],
                [5.0, 17.0, 31.0],
            )
        )
        self.assertEqual(
            merged["featherweight_events_launch_detected"].astype(bool).tolist(),
            [False, True, True],
        )

    def test_align_gps_altitude_to_marv_baro_recovers_offset(self) -> None:
        gps_time = np.arange(0.0, 32.0, 0.1)
        gps_altitude = np.interp(
            gps_time,
            [0.0, 2.0, 4.0, 8.0, 15.0, 22.0, 28.0, 31.9],
            [0.0, 0.0, 20.0, 400.0, 2200.0, 1800.0, 200.0, 0.0],
        )
        gps_navigation = pd.DataFrame(
            {
                "time_s": gps_time,
                "unix_time_s": 1_700_000_000.0 + gps_time,
                "gps_altitude_asl_m": 100.0 + gps_altitude,
                "gps_altitude_agl_m": gps_altitude,
                "latitude_deg": 33.0,
                "longitude_deg": -99.0,
                "horizontal_speed_mps": np.gradient(gps_altitude, gps_time),
                "vertical_speed_mps": np.gradient(gps_altitude, gps_time),
                "course_heading_deg": 120.0,
                "battery_v": 4.0,
            }
        )
        gps_events = pd.DataFrame(
            {
                "time_s": gps_time,
                "launch_detected": gps_time >= 3.0,
                "apogee_detected": gps_time >= 15.0,
                "landing_detected": gps_time >= 30.0,
            }
        )
        gps_source = self.module.NormalizedSource(
            source_name="featherweight_gps",
            source_path=Path("synthetic_gps.csv"),
            metadata={},
            streams={
                "navigation": gps_navigation,
                "events": gps_events,
            },
        )

        marv_time = np.arange(0.0, 30.0, 0.05)
        known_offset_s = -1.35
        scale = 0.96
        bias_m = 45.0
        marv_altitude = scale * np.interp(
            marv_time - known_offset_s,
            gps_time,
            gps_altitude,
            left=gps_altitude[0],
            right=gps_altitude[-1],
        ) + bias_m
        p0 = 97_100.0
        marv_pressure = p0 * np.power(1.0 - (marv_altitude / 44_330.0), 1.0 / 0.190294957)
        marv_baro = pd.DataFrame(
            {
                "time_s": marv_time,
                "log_time_s": marv_time,
                "sample_time_us": (marv_time * 1_000_000).astype(np.int64),
                "log_time_us": (marv_time * 1_000_000).astype(np.int64),
                "state": "fresh",
                "lagged": False,
                "pressure_pa": marv_pressure,
                "temperature_c": 24.0,
            }
        )
        marv_source = self.module.NormalizedSource(
            source_name="marv",
            source_path=Path("synthetic_marv.csv"),
            metadata={},
            streams={
                "baro": marv_baro,
                "primary_imu": marv_baro[["time_s"]].copy(),
                "aux_imu": marv_baro[["time_s"]].copy(),
            },
        )

        report, gps_debug, marv_debug = self.module.align_gps_altitude_to_marv_baro(
            gps_source,
            marv_source,
            coarse_step_s=0.05,
            refine_step_s=0.005,
            gps_smoothing_window_s=0.4,
            marv_smoothing_window_s=0.4,
        )

        self.assertAlmostEqual(report["offset_s"], known_offset_s, delta=0.15)
        self.assertGreater(report["altitude_correlation"], 0.98)
        self.assertGreater(report["derivative_correlation"], 0.9)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["can_merge"])
        self.assertIn("smooth_altitude_m", gps_debug.columns)
        self.assertIn("baro_altitude_rel_m", marv_debug.columns)

    def test_resolve_merge_alignment_rejects_low_confidence_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "baro_alignment_report.json"
            report_path.write_text(
                textwrap.dedent(
                    """\
                    {
                      "alignment": {
                        "offset_s": 12.5,
                        "confidence": 0.05,
                        "status": "ambiguous_input",
                        "warnings": ["ambiguous"]
                      }
                    }
                    """
                ),
                encoding="utf-8",
            )
            args = self.module.argparse.Namespace(
                offset_s=None,
                alignment_report=str(report_path),
                allow_low_confidence=False,
                min_confidence=0.25,
            )

            with self.assertRaises(ValueError):
                self.module._resolve_merge_alignment(args)

    def test_export_aligned_replay_session_writes_loadable_session(self) -> None:
        gps_time = np.arange(0.0, 6.0, 0.1)
        gps_altitude_agl_m = np.interp(
            gps_time,
            [0.0, 1.0, 2.0, 3.5, 5.0, 5.9],
            [0.0, 0.0, 25.0, 120.0, 30.0, 0.0],
        )
        latitude_deg = 33.5 + (gps_time * 1.0e-5)
        longitude_deg = -99.3 + (gps_time * 1.5e-5)
        gps_navigation = pd.DataFrame(
            {
                "time_s": gps_time,
                "unix_time_s": 1_774_896_430.0 + gps_time,
                "gps_altitude_asl_m": 415.0 + gps_altitude_agl_m,
                "gps_altitude_agl_m": gps_altitude_agl_m,
                "latitude_deg": latitude_deg,
                "longitude_deg": longitude_deg,
                "satellite_count": 12.0,
                "fix_type": 3.0,
                "horizontal_speed_mps": 8.0,
                "vertical_speed_mps": np.gradient(gps_altitude_agl_m, gps_time),
                "course_heading_deg": 90.0,
                "battery_v": 4.0,
                "distance_ft": gps_time * 10.0,
                "distance_m": gps_time * 10.0 * 0.3048,
                "flags": 1.0,
                "rssi_dbm": -55.0,
                "count_gt_40": 2.0,
                "count_gt_32": 4.0,
                "count_gt_24": 6.0,
            }
        )
        gps_events = pd.DataFrame(
            {
                "time_s": gps_time,
                "launch_detected": gps_time >= 1.0,
                "apogee_detected": gps_time >= 3.5,
                "landing_detected": gps_time >= 5.8,
            }
        )
        gps_source = self.module.NormalizedSource(
            source_name="featherweight_gps",
            source_path=Path("synthetic_gps.csv"),
            metadata={},
            streams={
                "navigation": gps_navigation,
                "events": gps_events,
            },
        )

        marv_time = np.arange(0.0, 6.0, 0.02)
        marv_altitude_m = np.interp(marv_time, gps_time, gps_altitude_agl_m, left=0.0, right=0.0)
        sea_level_pressure_pa = 101325.0
        ground_altitude_m = 415.0
        marv_pressure_pa = sea_level_pressure_pa * np.power(
            1.0 - (ground_altitude_m + marv_altitude_m) / 44330.0,
            5.255,
        )
        primary_imu = pd.DataFrame(
            {
                "time_s": marv_time,
                "log_time_s": marv_time,
                "sample_time_us": (marv_time * 1_000_000).astype(np.int64),
                "log_time_us": (marv_time * 1_000_000).astype(np.int64),
                "state": "fresh",
                "lagged": False,
                "accelerometer_x_mps2": 0.0,
                "accelerometer_y_mps2": 0.0,
                "accelerometer_z_mps2": -9.80665,
                "gyroscope_x_rad_s": 0.0,
                "gyroscope_y_rad_s": 0.0,
                "gyroscope_z_rad_s": 0.0,
            }
        )
        baro = pd.DataFrame(
            {
                "time_s": marv_time,
                "log_time_s": marv_time,
                "sample_time_us": (marv_time * 1_000_000).astype(np.int64),
                "log_time_us": (marv_time * 1_000_000).astype(np.int64),
                "state": "fresh",
                "lagged": False,
                "pressure_pa": marv_pressure_pa,
                "temperature_c": 24.0,
            }
        )
        marv_source = self.module.NormalizedSource(
            source_name="marv",
            source_path=Path("synthetic_marv.csv"),
            metadata={},
            streams={
                "primary_imu": primary_imu,
                "aux_imu": primary_imu.copy(),
                "baro": baro,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = self.module.export_aligned_replay_session(
                gps_source,
                marv_source,
                offset_s=0.0,
                output_dir=temp_dir,
                session_id="synthetic_replay",
                derive_parameters_from_telemetry=False,
            )

            self.assertTrue((Path(temp_dir) / "manifest.json").exists())
            self.assertTrue((Path(temp_dir) / "truth.csv").exists())
            self.assertEqual(summary["truth_source"], "layered_navigation_replay_estimate")

            session = load_replay_session(temp_dir)
            self.assertEqual(session.truth.columns.tolist(), list(TRUTH_COLUMNS))
            self.assertGreater(len(session.truth), 10)
            self.assertGreater(session.truth["z_m"].max(), 10.0)


if __name__ == "__main__":
    unittest.main()
