from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from sim.informatics import InformaticsContext, InformaticsPlaybackSource, RocketGeometryComponent
from sim.gui.data_informatics_panel import DataInformaticsPanel
from sim.sitl.session import ReplaySession


class Motor:
    grain_outer_radius = 0.075
    grain_initial_inner_radius = 0.02
    grain_initial_height = 1.2
    nozzle_radius = 0.03
    throat_radius = 0.01
    nozzle_position = -2.1


class NoseCone:
    length = 0.55
    base_radius = 0.0762
    kind = "Von Karman"
    rocket_radius = 0.0762


class TrapezoidalFins:
    n = 4
    root_chord = 0.32
    tip_chord = 0.14
    span = 0.17
    sweep_length = 0.11
    rocket_radius = 0.0762


class Tail:
    top_radius = 0.0762
    bottom_radius = 0.05
    length = 0.19
    rocket_radius = 0.0762


class DummyRocket:
    def __init__(self) -> None:
        self.mass = 23.5
        self.radius = 0.0762
        self.motor = Motor()
        self.motor_position = -2.55
        self.coordinate_system_orientation = "tail_to_nose"
        self.aerodynamic_surfaces = [
            (NoseCone(), 0.0),
            (TrapezoidalFins(), -2.15),
            (Tail(), -2.6),
        ]


def _make_truth_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.5],
            "z_m": [0.0, 10.0],
            "vx_mps": [0.0, 10.0],
            "vy_mps": [0.0, 5.0],
            "vz_mps": [0.0, 100.0],
            "e0": [1.0, 1.0],
            "e1": [0.0, 0.0],
            "e2": [0.0, 0.0],
            "e3": [0.0, 0.0],
            "w1_radps": [0.0, 0.1],
            "w2_radps": [0.0, 0.2],
            "w3_radps": [0.0, 0.3],
        }
    )


class InformaticsPanelModelTests(unittest.TestCase):
    def test_geometry_component_is_independent_from_playback_sources(self) -> None:
        component = RocketGeometryComponent.from_rocket(
            DummyRocket(),
            source_name="Itzamna rocket_config.py",
            source_path="notebooks/Itzamna/rocket_config.py",
        )

        self.assertEqual(component.radius_m, 0.0762)
        self.assertEqual(component.mass_kg, 23.5)
        self.assertEqual(component.coordinate_system, "tail_to_nose")
        self.assertEqual(
            component.default_component_ids(),
            ("motor", "nosecone", "body", "fins", "tail"),
        )
        self.assertIn(
            "Config path: notebooks/Itzamna/rocket_config.py",
            component.summary_lines(),
        )

    def test_playback_source_can_be_defined_from_arbitrary_frames(self) -> None:
        truth = _make_truth_frame()
        imu = pd.DataFrame(
            {
                "time_s": [0.0, 0.01],
                "accelerometer_x": [0.0, 0.2],
                "accelerometer_y": [0.0, 0.1],
                "accelerometer_z": [9.81, 10.1],
            }
        )
        gps = pd.DataFrame(
            {
                "time_s": [0.0],
                "latitude_deg": [26.2],
                "longitude_deg": [-98.1],
                "altitude_m": [15.0],
            }
        )

        source = InformaticsPlaybackSource.from_data_frames(
            display_name="Postflight merge",
            kinematics_frame=truth,
            sensor_frames={"imu": imu, "gps": gps},
            source_kind="mixed",
            stream_rates_hz={"kinematics": 500.0, "imu": 300.0, "gps": 10.0},
            stream_origins={
                "kinematics": "derived",
                "imu": "virtual_sensor",
                "gps": "real_sensor",
            },
        )

        self.assertEqual(source.display_name, "Postflight merge")
        self.assertIs(source.kinematics_frame, truth)
        self.assertIs(source.sensor_frames["gps"], gps)
        self.assertEqual(
            [stream.origin for stream in source.sensor_streams],
            ["virtual_sensor", "real_sensor"],
        )
        self.assertTrue(any("real_sensor" in line for line in source.summary_lines()))
        self.assertTrue(any("Duration:" in line for line in source.stats_lines()))
        self.assertTrue(any("Max altitude:" in line for line in source.stats_lines()))

    def test_context_accepts_replay_session_as_separate_data_source(self) -> None:
        truth = _make_truth_frame()
        imu = pd.DataFrame(
            {
                "time_s": [0.0],
                "accelerometer_x": [0.0],
                "accelerometer_y": [0.0],
                "accelerometer_z": [9.81],
                "gyroscope_x": [0.0],
                "gyroscope_y": [0.0],
                "gyroscope_z": [0.0],
            }
        )
        baro = pd.DataFrame({"time_s": [0.0], "barometer_v1": [101325.0]})
        gps = pd.DataFrame(
            {"time_s": [0.0], "gnss_x": [26.2], "gnss_y": [-98.1], "gnss_z": [15.0]}
        )

        session = ReplaySession(
            session_dir=Path("logs/session_demo"),
            manifest_path=Path("logs/session_demo/manifest.json"),
            manifest={
                "vehicle_name": "Itzamna",
                "session_id": "session_demo",
                "reference_latitude_deg": 26.2,
                "reference_longitude_deg": -98.1,
                "rates_hz": {"truth": 500, "imu": 300, "baro": 100, "gps": 10},
            },
            stream_paths={
                "truth": Path("truth.csv"),
                "imu": Path("imu.csv"),
                "baro": Path("baro.csv"),
                "gps": Path("gps.csv"),
            },
            stream_frames={"truth": truth, "imu": imu, "baro": baro, "gps": gps},
            truth=truth,
            imu=imu,
            baro=baro,
            gps=gps,
            mag=None,
        )

        context = InformaticsContext(
            RocketGeometryComponent.from_rocket(DummyRocket())
        ).with_data_source(InformaticsPlaybackSource.from_replay_session(session))

        self.assertIsNotNone(context.data_source)
        assert context.data_source is not None
        self.assertEqual(context.data_source.source_kind, "replay_session")
        self.assertEqual(context.data_source.kinematics_stream.origin, "simulation")
        self.assertEqual(
            {stream.key for stream in context.data_source.sensor_streams},
            {"imu", "baro", "gps"},
        )

    def test_acceleration_columns_are_reserved_for_bottom_plot(self) -> None:
        columns = [
            "time_s",
            "accelerometer_x",
            "accelerometer_y",
            "accelerometer_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z",
        ]

        self.assertEqual(
            DataInformaticsPanel._acceleration_columns(columns),
            ["accelerometer_x", "accelerometer_y", "accelerometer_z"],
        )
        self.assertEqual(
            DataInformaticsPanel._generic_plot_columns(columns),
            ["gyroscope_x", "gyroscope_y", "gyroscope_z"],
        )


if __name__ == "__main__":
    unittest.main()
