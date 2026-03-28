"""Replay primitives for session-derived telemetry and offline compatibility views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .session import (
    ReplaySession,
    find_latest_session_manifest,
    load_replay_session,
    merge_replay_session_sensors,
)


@dataclass(slots=True)
class ReplaySample:
    """Single replay sample pulled from a telemetry CSV row."""

    index: int
    time_s: float
    row: dict[str, Any]

    def json_row(self) -> dict[str, Any]:
        return {column: json_scalar(value) for column, value in self.row.items()}


@dataclass(slots=True)
class ReplayClock:
    """Mutable replay cursor and time synchronization helper."""

    telemetry: pd.DataFrame
    time_column: str = "time_s"
    index: int = 0
    playing: bool = False
    replay_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.time_column not in self.telemetry.columns:
            raise ValueError(f"Missing time column '{self.time_column}' in telemetry frame")
        if self.telemetry.empty:
            raise ValueError("Telemetry frame is empty")
        self.telemetry = self.telemetry.sort_values(self.time_column).reset_index(drop=True)

    @property
    def times_s(self) -> np.ndarray:
        return self.telemetry[self.time_column].to_numpy(dtype=float)

    @property
    def at_end(self) -> bool:
        return self.index >= len(self.telemetry) - 1

    def clamp_index(self, index: int) -> int:
        return int(max(0, min(index, len(self.telemetry) - 1)))

    def reset(self) -> None:
        self.index = 0

    def sync_to_time(self, target_time_s: float) -> int:
        position = int(np.searchsorted(self.times_s, float(target_time_s), side="left"))
        self.index = self.clamp_index(position)
        return self.index

    def step(self, count: int = 1) -> int:
        self.index = self.clamp_index(self.index + max(int(count), 0))
        return self.index

    def seek_index(self, index: int) -> int:
        self.index = self.clamp_index(index)
        return self.index

    def current_sample(self) -> ReplaySample:
        row = self.telemetry.iloc[self.index].to_dict()
        return ReplaySample(
            index=self.index,
            time_s=self.current_time_s(),
            row=row,
        )

    def current_time_s(self) -> float:
        return float(self.times_s[self.index])

    def dt_to_next_s(self) -> float:
        if self.at_end:
            return 0.0
        current = self.current_time_s()
        nxt = float(self.times_s[self.index + 1])
        return max(0.0, nxt - current)

    def snapshot(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "time_s": self.current_time_s(),
            "at_end": self.at_end,
            "playing": self.playing,
            "replay_rate": self.replay_rate,
            "total_samples": int(len(self.telemetry)),
        }


def load_replay_telemetry(
    telemetry: ReplaySession | str | Path | pd.DataFrame | None,
    *,
    logs_directory: str | Path,
    time_column: str = "time_s",
) -> tuple[pd.DataFrame, Path | None]:
    """Load replay telemetry from a session, DataFrame, explicit path, or latest session."""

    telemetry_path: Path | None = None
    if telemetry is None:
        replay_session = load_replay_session(find_latest_session_manifest(logs_directory))
        telemetry_path = replay_session.manifest_path
        frame = merge_replay_session_sensors(replay_session)
    elif isinstance(telemetry, ReplaySession):
        telemetry_path = telemetry.manifest_path
        frame = merge_replay_session_sensors(telemetry)
    elif isinstance(telemetry, pd.DataFrame):
        frame = telemetry.copy()
    else:
        telemetry_path = Path(telemetry)
        if telemetry_path.is_dir() or telemetry_path.name == "manifest.json":
            replay_session = load_replay_session(telemetry_path)
            telemetry_path = replay_session.manifest_path
            frame = merge_replay_session_sensors(replay_session)
        else:
            frame = pd.read_csv(telemetry_path)

    if time_column not in frame.columns:
        raise ValueError(f"Telemetry is missing required '{time_column}' column")

    frame = frame.sort_values(time_column).reset_index(drop=True)
    return frame, telemetry_path


def json_scalar(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value
