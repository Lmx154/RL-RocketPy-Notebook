"""Truth-tick replay scheduler for manifest-based SITL sessions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .session import ReplaySession

_SENSOR_STREAM_COLUMNS: dict[str, tuple[str, ...]] = {
    "imu": (
        "accelerometer_x",
        "accelerometer_y",
        "accelerometer_z",
        "gyroscope_x",
        "gyroscope_y",
        "gyroscope_z",
    ),
    "baro": ("barometer_v1",),
    "gps": ("gnss_x", "gnss_y", "gnss_z"),
    "mag": ("magnetometer_x", "magnetometer_y", "magnetometer_z"),
}

SCHEDULER_SENSOR_COLUMNS = tuple(
    column
    for stream_columns in _SENSOR_STREAM_COLUMNS.values()
    for column in stream_columns
)


@dataclass(slots=True)
class _StreamCursor:
    key: str
    frame: pd.DataFrame
    columns: tuple[str, ...]
    times_s: np.ndarray
    cursor: int = 0
    latest_row_index: int | None = None
    fresh: bool = False


class ReplaySessionScheduler:
    """Advance replay state by truth timestamps while honoring sensor stream rates."""

    def __init__(self, replay_session: ReplaySession):
        if replay_session.truth.empty:
            raise ValueError("Replay session truth stream is empty")

        self.replay_session = replay_session
        self.truth = replay_session.truth.sort_values("time_s").reset_index(drop=True)
        self.truth_times_s = self.truth["time_s"].to_numpy(dtype=float)
        self._stream_cursors = {
            key: _build_stream_cursor(key, replay_session.stream_frames.get(key))
            for key in _SENSOR_STREAM_COLUMNS
        }
        self.truth_index = 0
        self.reset()

    @property
    def total_steps(self) -> int:
        return int(len(self.truth))

    @property
    def at_end(self) -> bool:
        return self.truth_index >= self.total_steps - 1

    @property
    def t_final_s(self) -> float:
        return float(self.truth_times_s[-1])

    def clamp_index(self, index: int) -> int:
        return max(0, min(int(index), self.total_steps - 1))

    def current_time_s(self) -> float:
        return float(self.truth_times_s[self.truth_index])

    def reset(self) -> dict[str, Any]:
        return self.seek_truth_index(0)

    def seek_truth_index(self, index: int) -> dict[str, Any]:
        target = self.clamp_index(index)
        self.truth_index = target

        target_time_s = float(self.truth_times_s[target])
        previous_time_s = float(self.truth_times_s[target - 1]) if target > 0 else None
        for stream in self._stream_cursors.values():
            _seek_stream_cursor(
                stream,
                target_time_s=target_time_s,
                previous_time_s=previous_time_s,
            )
        return self.current_state()

    def advance_one_tick(self) -> dict[str, Any]:
        if self.at_end:
            return self.current_state()

        previous_time_s = float(self.truth_times_s[self.truth_index])
        self.truth_index += 1
        current_time_s = float(self.truth_times_s[self.truth_index])

        for stream in self._stream_cursors.values():
            stream.fresh = False
            while stream.cursor < len(stream.times_s) and stream.times_s[stream.cursor] <= current_time_s:
                if stream.times_s[stream.cursor] > previous_time_s:
                    stream.fresh = True
                stream.latest_row_index = stream.cursor
                stream.cursor += 1

        return self.current_state()

    def current_state(self) -> dict[str, Any]:
        row = self.truth.iloc[self.truth_index]

        e0 = float(row.get("e0", 1.0))
        e1 = float(row.get("e1", 0.0))
        e2 = float(row.get("e2", 0.0))
        e3 = float(row.get("e3", 0.0))

        sensors: dict[str, float | None] = {
            column: None for column in SCHEDULER_SENSOR_COLUMNS
        }
        freshness: dict[str, bool] = {
            column: False for column in SCHEDULER_SENSOR_COLUMNS
        }

        for stream in self._stream_cursors.values():
            latest_row = None
            if stream.latest_row_index is not None:
                latest_row = stream.frame.iloc[stream.latest_row_index]

            for column in stream.columns:
                value = None if latest_row is None else latest_row.get(column)
                sensors[column] = float(value) if pd.notna(value) else None
                freshness[column] = bool(stream.fresh)

        gnss_altitude = sensors.get("gnss_z")
        altitude = float(gnss_altitude) if gnss_altitude is not None else float(row.get("z_m", 0.0))

        vx = float(row.get("vx_mps", 0.0))
        vy = float(row.get("vy_mps", 0.0))
        vz = float(row.get("vz_mps", 0.0))

        phi, theta, psi = _quaternion_to_euler(e0, e1, e2, e3)

        return {
            "time": float(row["time_s"]),
            "step_index": self.truth_index,
            "total_steps": self.total_steps,
            "position": {
                "x": float(row.get("x_m", 0.0)),
                "y": float(row.get("y_m", 0.0)),
                "z": float(row.get("z_m", 0.0)),
                "altitude": altitude,
            },
            "gps": {
                "latitude": float(sensors["gnss_x"]) if sensors["gnss_x"] is not None else 0.0,
                "longitude": float(sensors["gnss_y"]) if sensors["gnss_y"] is not None else 0.0,
            },
            "quaternion": {"e0": e0, "e1": e1, "e2": e2, "e3": e3},
            "euler": {"phi": float(phi), "theta": float(theta), "psi": float(psi)},
            "velocity": {
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "speed": float(np.sqrt(vx * vx + vy * vy + vz * vz)),
                "horizontal_speed": float(np.sqrt(vx * vx + vy * vy)),
            },
            "angular_velocity": {
                "w1": float(row.get("w1_radps", 0.0)),
                "w2": float(row.get("w2_radps", 0.0)),
                "w3": float(row.get("w3_radps", 0.0)),
            },
            "mach_number": 0.0,
            "dynamic_pressure": 0.0,
            "phase": "REPLAY",
            "t_final": self.t_final_s,
            "apogee": float(self.truth["z_m"].max()),
            "sensors": sensors,
            "sensor_freshness": freshness,
        }
def _build_stream_cursor(stream_key: str, frame: pd.DataFrame | None) -> _StreamCursor:
    columns = _SENSOR_STREAM_COLUMNS[stream_key]
    if frame is None:
        frame = pd.DataFrame(columns=["time_s", *columns])
    else:
        frame = frame.sort_values("time_s").reset_index(drop=True)
    return _StreamCursor(
        key=stream_key,
        frame=frame,
        columns=columns,
        times_s=frame["time_s"].to_numpy(dtype=float) if "time_s" in frame.columns else np.array([], dtype=float),
    )


def _seek_stream_cursor(
    stream: _StreamCursor,
    *,
    target_time_s: float,
    previous_time_s: float | None,
) -> None:
    if len(stream.times_s) == 0:
        stream.cursor = 0
        stream.latest_row_index = None
        stream.fresh = False
        return

    cursor = int(np.searchsorted(stream.times_s, target_time_s, side="right"))
    if previous_time_s is None:
        previous_cursor = 0
    else:
        previous_cursor = int(np.searchsorted(stream.times_s, previous_time_s, side="right"))

    stream.cursor = cursor
    stream.latest_row_index = cursor - 1 if cursor > 0 else None
    stream.fresh = cursor > previous_cursor


def _quaternion_to_euler(e0: float, e1: float, e2: float, e3: float) -> tuple[float, float, float]:
    """Convert quaternion to Euler angles without importing the simulation package."""

    sinr_cosp = 2.0 * (e0 * e1 + e2 * e3)
    cosr_cosp = 1.0 - 2.0 * (e1 * e1 + e2 * e2)
    phi = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (e0 * e2 - e3 * e1)
    if abs(sinp) >= 1.0:
        theta = math.degrees(math.copysign(math.pi / 2.0, sinp))
    else:
        theta = math.degrees(math.asin(sinp))

    siny_cosp = 2.0 * (e0 * e3 + e1 * e2)
    cosy_cosp = 1.0 - 2.0 * (e2 * e2 + e3 * e3)
    psi = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    return phi, theta, psi


__all__ = [
    "ReplaySessionScheduler",
    "SCHEDULER_SENSOR_COLUMNS",
]
