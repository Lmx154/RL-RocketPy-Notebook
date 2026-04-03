"""Informatics-facing models for geometry and playback data sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from sim.geometry.extraction import RocketAssemblyGeometry
from sim.sitl.session import ReplaySession


@dataclass(frozen=True, slots=True)
class SceneComponentDefinition:
    """Declarative description of one independently selectable rocket component."""

    component_id: str
    display_name: str
    available: bool
    mesh_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RocketGeometryComponent:
    """Rocket geometry component derived from the shared rocket configuration."""

    rocket: Any
    geometry: RocketAssemblyGeometry
    source_name: str
    source_path: Path | None
    components: tuple[SceneComponentDefinition, ...]

    @classmethod
    def from_rocket(
        cls,
        rocket: Any,
        *,
        source_name: str = "RocketPy configuration",
        source_path: str | Path | None = None,
    ) -> "RocketGeometryComponent":
        geometry = RocketAssemblyGeometry.from_rocketpy_rocket(rocket)
        components = (
            SceneComponentDefinition("motor", "Motor", geometry.motor is not None, ("motor_casing", "motor_nozzle", "motor_closure")),
            SceneComponentDefinition("nosecone", "Nose Cone", geometry.nosecone is not None, ("nosecone",)),
            SceneComponentDefinition("body", "Body Tube", True, ("body",)),
            SceneComponentDefinition("fins", "Fins", geometry.fins is not None, ("fin_1", "fin_2", "fin_3", "fin_4")),
            SceneComponentDefinition("tail", "Tail", geometry.tail is not None, ("tail",)),
        )
        return cls(
            rocket=rocket,
            geometry=geometry,
            source_name=source_name,
            source_path=Path(source_path) if source_path else None,
            components=components,
        )

    @property
    def mass_kg(self) -> float:
        return float(getattr(self.rocket, "mass", 0.0))

    @property
    def radius_m(self) -> float:
        return float(self.geometry.radius)

    @property
    def total_length_m(self) -> float:
        return float(self.geometry.total_length or 0.0)

    @property
    def coordinate_system(self) -> str:
        return str(self.geometry.coordinate_system)

    def available_components(self) -> tuple[SceneComponentDefinition, ...]:
        return tuple(component for component in self.components if component.available)

    def default_component_ids(self) -> tuple[str, ...]:
        return tuple(component.component_id for component in self.available_components())

    def summary_lines(self) -> list[str]:
        component_names = ", ".join(component.display_name for component in self.available_components())
        lines = [
            f"Source: {self.source_name}",
            f"Radius: {self.radius_m:.4f} m",
            f"Length: {self.total_length_m:.3f} m",
            f"Mass: {self.mass_kg:.2f} kg",
            f"Coordinate system: {self.coordinate_system}",
            f"Components: {component_names or 'none'}",
        ]
        if self.source_path is not None:
            lines.append(f"Config path: {self.source_path}")
        return lines

    def build_renderer(self, *, cache: bool = True):
        """Create a renderer from the geometry component on demand."""
        from sim.rendering.renderer import RocketRenderer

        return RocketRenderer(self.rocket, cache=cache)


@dataclass(frozen=True, slots=True)
class InformaticsStream:
    """One kinematics or sensor stream participating in visualization playback."""

    key: str
    display_name: str
    kind: str
    origin: str
    row_count: int
    columns: tuple[str, ...]
    sample_rate_hz: float | None = None
    path: Path | None = None

    @classmethod
    def from_frame(
        cls,
        *,
        key: str,
        display_name: str,
        frame: pd.DataFrame,
        kind: str,
        origin: str,
        sample_rate_hz: float | None = None,
        path: str | Path | None = None,
    ) -> "InformaticsStream":
        return cls(
            key=key,
            display_name=display_name,
            kind=kind,
            origin=origin,
            row_count=int(len(frame)),
            columns=tuple(str(column) for column in frame.columns),
            sample_rate_hz=None if sample_rate_hz is None else float(sample_rate_hz),
            path=Path(path) if path else None,
        )

    def summary_line(self) -> str:
        rate = "event-driven" if self.sample_rate_hz is None else f"{self.sample_rate_hz:g} Hz"
        return f"{self.display_name}: {self.origin}, {self.row_count:,} rows @ {rate}"


@dataclass(frozen=True, slots=True)
class InformaticsPlaybackSource:
    """Data source driving replay, independent of rocket geometry definition."""

    display_name: str
    source_kind: str
    kinematics_frame: pd.DataFrame
    sensor_frames: Mapping[str, pd.DataFrame]
    kinematics_stream: InformaticsStream
    sensor_streams: tuple[InformaticsStream, ...]
    metadata: Mapping[str, Any]
    session_dir: Path | None = None
    manifest_path: Path | None = None

    @classmethod
    def from_data_frames(
        cls,
        *,
        display_name: str,
        kinematics_frame: pd.DataFrame,
        sensor_frames: Mapping[str, pd.DataFrame] | None = None,
        source_kind: str = "mixed",
        metadata: Mapping[str, Any] | None = None,
        stream_rates_hz: Mapping[str, float | None] | None = None,
        stream_paths: Mapping[str, str | Path] | None = None,
        stream_origins: Mapping[str, str] | None = None,
    ) -> "InformaticsPlaybackSource":
        sensor_frames = sensor_frames or {}
        stream_rates_hz = stream_rates_hz or {}
        stream_paths = stream_paths or {}
        stream_origins = stream_origins or {}

        kinematics_stream = InformaticsStream.from_frame(
            key="kinematics",
            display_name="Kinematics",
            frame=kinematics_frame,
            kind="kinematics",
            origin=stream_origins.get("kinematics", "derived"),
            sample_rate_hz=stream_rates_hz.get("kinematics"),
            path=stream_paths.get("kinematics"),
        )

        sensor_streams = []
        for key, frame in sensor_frames.items():
            sensor_streams.append(
                InformaticsStream.from_frame(
                    key=key,
                    display_name=key.replace("_", " ").title(),
                    frame=frame,
                    kind="sensor",
                    origin=stream_origins.get(key, "unknown"),
                    sample_rate_hz=stream_rates_hz.get(key),
                    path=stream_paths.get(key),
                )
            )

        return cls(
            display_name=display_name,
            source_kind=source_kind,
            kinematics_frame=kinematics_frame,
            sensor_frames=dict(sensor_frames),
            kinematics_stream=kinematics_stream,
            sensor_streams=tuple(sensor_streams),
            metadata=metadata or {},
        )

    @classmethod
    def from_replay_session(cls, replay_session: ReplaySession) -> "InformaticsPlaybackSource":
        rates_hz = replay_session.manifest.get("rates_hz", {})
        stream_paths = replay_session.stream_paths
        sensor_frames = {
            key: replay_session.stream_frames[key]
            for key in ("imu", "baro", "gps", "mag")
            if key in replay_session.stream_frames
        }
        stream_origins = {
            "kinematics": "simulation",
            **{key: "virtual_sensor" for key in sensor_frames},
        }
        metadata = {
            "vehicle_name": replay_session.manifest.get("vehicle_name"),
            "session_id": replay_session.manifest.get("session_id"),
            "reference_latitude_deg": replay_session.manifest.get("reference_latitude_deg"),
            "reference_longitude_deg": replay_session.manifest.get("reference_longitude_deg"),
        }
        return cls(
            display_name=replay_session.session_dir.name,
            source_kind="replay_session",
            kinematics_frame=replay_session.truth,
            sensor_frames=dict(sensor_frames),
            kinematics_stream=InformaticsStream.from_frame(
                key="truth",
                display_name="Kinematics",
                frame=replay_session.truth,
                kind="kinematics",
                origin=stream_origins["kinematics"],
                sample_rate_hz=rates_hz.get("truth"),
                path=stream_paths.get("truth"),
            ),
            sensor_streams=tuple(
                InformaticsStream.from_frame(
                    key=key,
                    display_name=key.upper() if key != "baro" else "Barometer",
                    frame=frame,
                    kind="sensor",
                    origin=stream_origins.get(key, "virtual_sensor"),
                    sample_rate_hz=rates_hz.get(key),
                    path=stream_paths.get(key),
                )
                for key, frame in sensor_frames.items()
            ),
            metadata=metadata,
            session_dir=replay_session.session_dir,
            manifest_path=replay_session.manifest_path,
        )

    @property
    def stream_count(self) -> int:
        return 1 + len(self.sensor_streams)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Data source: {self.display_name}",
            f"Mode: {self.source_kind}",
            f"Kinematics: {self.kinematics_stream.summary_line()}",
        ]
        if self.sensor_streams:
            for stream in self.sensor_streams:
                lines.append(f"Sensor: {stream.summary_line()}")
        else:
            lines.append("Sensor: none loaded")
        if self.session_dir is not None:
            lines.append(f"Session dir: {self.session_dir}")
        if self.manifest_path is not None:
            lines.append(f"Manifest: {self.manifest_path.name}")
        return lines

    def stats_lines(self) -> list[str]:
        """Return concise analysis-oriented statistics for the loaded source."""
        lines = [
            f"Stream count: {self.stream_count}",
            f"Sensor stream count: {len(self.sensor_streams)}",
        ]

        frame = self.kinematics_frame
        if not frame.empty and "time_s" in frame.columns:
            duration_s = float(frame["time_s"].iloc[-1] - frame["time_s"].iloc[0])
            lines.append(f"Duration: {duration_s:.3f} s")

        if "z_m" in frame.columns and not frame["z_m"].empty:
            lines.append(f"Max altitude: {float(frame['z_m'].max()):.3f} m")

        velocity_columns = {"vx_mps", "vy_mps", "vz_mps"}
        if velocity_columns.issubset(frame.columns):
            velocity = frame[["vx_mps", "vy_mps", "vz_mps"]].to_numpy(dtype=float)
            speed = ((velocity ** 2).sum(axis=1)) ** 0.5 if len(velocity) else []
            max_speed = float(speed.max()) if len(velocity) else 0.0
            lines.append(f"Max speed: {max_speed:.3f} m/s")

        if self.sensor_streams:
            origins = ", ".join(sorted({stream.origin for stream in self.sensor_streams}))
            lines.append(f"Sensor origins: {origins}")

        if self.metadata:
            vehicle_name = self.metadata.get("vehicle_name")
            if vehicle_name:
                lines.append(f"Vehicle: {vehicle_name}")
            session_id = self.metadata.get("session_id")
            if session_id:
                lines.append(f"Session ID: {session_id}")

        return lines


@dataclass(frozen=True, slots=True)
class InformaticsContext:
    """Top-level GUI context composed from independent geometry and data modules."""

    geometry_component: RocketGeometryComponent
    data_source: InformaticsPlaybackSource | None = None

    def with_data_source(self, data_source: InformaticsPlaybackSource | None) -> "InformaticsContext":
        return type(self)(geometry_component=self.geometry_component, data_source=data_source)
