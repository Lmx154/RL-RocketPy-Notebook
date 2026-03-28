"""Frozen session contract for per-flight SITL replay directories."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping

import pandas as pd

SESSION_SCHEMA_VERSION = "1.0"
SESSION_MANIFEST_FILENAME = "manifest.json"
SESSION_MANIFEST_SCHEMA_FILENAME = "manifest.schema.json"
SESSION_SPEC_FILENAME = "session_spec.md"
TIME_COLUMN = "time_s"

TRUTH_COLUMNS = (
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
)

IMU_COLUMNS = (
    "time_s",
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
)

BARO_COLUMNS = (
    "time_s",
    "barometer_v1",
)

GPS_COLUMNS = (
    "time_s",
    "gnss_x",
    "gnss_y",
    "gnss_z",
)

MAG_COLUMNS = (
    "time_s",
    "magnetometer_x",
    "magnetometer_y",
    "magnetometer_z",
)

ESTIMATOR_FEEDBACK_COLUMNS = (
    "time_s",
    "feedback_type",
    "payload_json",
)

DEVICE_EVENT_COLUMNS = (
    "time_s",
    "event_type",
    "event_name",
    "payload_json",
)


@dataclass(frozen=True, slots=True)
class SessionStreamSpec:
    """Frozen specification for one session CSV stream."""

    key: str
    filename: str
    required: bool
    columns: tuple[str, ...]
    rate_hz: int | None
    description: str


@dataclass(frozen=True, slots=True)
class ReplaySession:
    """Loaded replay session with canonical per-stream frames."""

    session_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    stream_paths: dict[str, Path]
    stream_frames: dict[str, pd.DataFrame]
    truth: pd.DataFrame
    imu: pd.DataFrame
    baro: pd.DataFrame
    gps: pd.DataFrame
    mag: pd.DataFrame | None


SESSION_STREAM_SPECS: dict[str, SessionStreamSpec] = {
    "truth": SessionStreamSpec(
        key="truth",
        filename="truth.csv",
        required=True,
        columns=TRUTH_COLUMNS,
        rate_hz=500,
        description="Uniform truth timeline sampled from the RocketPy flight state.",
    ),
    "imu": SessionStreamSpec(
        key="imu",
        filename="imu.csv",
        required=True,
        columns=IMU_COLUMNS,
        rate_hz=300,
        description="Combined accelerometer and gyroscope samples.",
    ),
    "baro": SessionStreamSpec(
        key="baro",
        filename="baro.csv",
        required=True,
        columns=BARO_COLUMNS,
        rate_hz=100,
        description="Barometric pressure samples.",
    ),
    "gps": SessionStreamSpec(
        key="gps",
        filename="gps.csv",
        required=True,
        columns=GPS_COLUMNS,
        rate_hz=10,
        description="GNSS samples in the existing RocketPy replay field naming.",
    ),
    "mag": SessionStreamSpec(
        key="mag",
        filename="mag.csv",
        required=False,
        columns=MAG_COLUMNS,
        rate_hz=50,
        description="Optional magnetometer samples.",
    ),
    "estimator_feedback": SessionStreamSpec(
        key="estimator_feedback",
        filename="estimator_feedback.csv",
        required=False,
        columns=ESTIMATOR_FEEDBACK_COLUMNS,
        rate_hz=None,
        description="Future typed estimator/device feedback stream.",
    ),
    "device_events": SessionStreamSpec(
        key="device_events",
        filename="device_events.csv",
        required=False,
        columns=DEVICE_EVENT_COLUMNS,
        rate_hz=None,
        description="Future event-driven device state stream.",
    ),
}

REQUIRED_STREAM_KEYS = tuple(
    key for key, spec in SESSION_STREAM_SPECS.items() if spec.required
)
OPTIONAL_STREAM_KEYS = tuple(
    key for key, spec in SESSION_STREAM_SPECS.items() if not spec.required
)
CANONICAL_STREAM_RATES_HZ = {
    key: spec.rate_hz
    for key, spec in SESSION_STREAM_SPECS.items()
    if spec.rate_hz is not None
}
REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "session_id",
    "vehicle_name",
    "generated_at_utc",
    "reference_latitude_deg",
    "reference_longitude_deg",
    "reference_altitude_m",
    "sea_level_pressure_pa",
    "streams",
    "rates_hz",
)
OPTIONAL_MANIFEST_KEYS = ("notes",)


def session_contract_directory() -> Path:
    """Return the directory containing the frozen session contract artifacts."""

    return Path(__file__).resolve().parent


def manifest_schema_path() -> Path:
    """Return the on-disk JSON schema path for ``manifest.json``."""

    return session_contract_directory() / SESSION_MANIFEST_SCHEMA_FILENAME


def session_spec_path() -> Path:
    """Return the on-disk markdown session spec path."""

    return session_contract_directory() / SESSION_SPEC_FILENAME


def load_manifest_schema() -> dict[str, Any]:
    """Load the frozen JSON schema for ``manifest.json``."""

    return json.loads(manifest_schema_path().read_text(encoding="utf-8"))


def default_stream_filenames(*, include_optional: bool = False) -> dict[str, str]:
    """Return canonical filenames for the frozen session contract."""

    stream_keys = REQUIRED_STREAM_KEYS
    if include_optional:
        stream_keys = tuple(SESSION_STREAM_SPECS.keys())
    return {key: SESSION_STREAM_SPECS[key].filename for key in stream_keys}


def build_session_manifest(
    *,
    session_id: str,
    vehicle_name: str,
    generated_at_utc: str,
    reference_latitude_deg: float,
    reference_longitude_deg: float,
    reference_altitude_m: float,
    sea_level_pressure_pa: float,
    notes: str | None = None,
    include_optional_streams: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a manifest dictionary using canonical filenames and rates."""

    streams = default_stream_filenames()
    rates_hz = {
        key: CANONICAL_STREAM_RATES_HZ[key]
        for key in REQUIRED_STREAM_KEYS
        if key in CANONICAL_STREAM_RATES_HZ
    }

    for stream_key in include_optional_streams:
        spec = _require_known_stream_key(stream_key)
        if spec.required:
            continue
        streams[stream_key] = spec.filename
        if spec.rate_hz is not None:
            rates_hz[stream_key] = spec.rate_hz

    manifest = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "session_id": session_id,
        "vehicle_name": vehicle_name,
        "generated_at_utc": generated_at_utc,
        "reference_latitude_deg": float(reference_latitude_deg),
        "reference_longitude_deg": float(reference_longitude_deg),
        "reference_altitude_m": float(reference_altitude_m),
        "sea_level_pressure_pa": float(sea_level_pressure_pa),
        "streams": streams,
        "rates_hz": rates_hz,
    }
    if notes:
        manifest["notes"] = notes

    validate_session_manifest(manifest)
    return manifest


def validate_session_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate that a manifest matches the frozen phase-1 session contract."""

    if not isinstance(manifest, Mapping):
        raise TypeError("Session manifest must be a mapping")

    allowed_top_level = set(REQUIRED_MANIFEST_KEYS) | set(OPTIONAL_MANIFEST_KEYS)
    unexpected_keys = sorted(set(manifest.keys()) - allowed_top_level)
    if unexpected_keys:
        raise ValueError(f"Session manifest has unsupported keys: {unexpected_keys}")

    missing_top_level = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing_top_level:
        raise ValueError(f"Session manifest missing required keys: {missing_top_level}")

    schema_version = _require_non_empty_string(manifest, "schema_version")
    if schema_version != SESSION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported session schema_version '{schema_version}', expected '{SESSION_SCHEMA_VERSION}'"
        )

    _require_non_empty_string(manifest, "session_id")
    _require_non_empty_string(manifest, "vehicle_name")
    _require_non_empty_string(manifest, "generated_at_utc")
    _require_finite_number(
        manifest,
        "reference_latitude_deg",
        minimum=-90.0,
        maximum=90.0,
    )
    _require_finite_number(
        manifest,
        "reference_longitude_deg",
        minimum=-180.0,
        maximum=180.0,
    )
    _require_finite_number(manifest, "reference_altitude_m")
    _require_finite_number(
        manifest,
        "sea_level_pressure_pa",
        minimum=1.0,
    )

    if "notes" in manifest and not isinstance(manifest["notes"], str):
        raise ValueError("Session manifest notes must be a string when provided")

    streams = manifest["streams"]
    if not isinstance(streams, Mapping):
        raise ValueError("Session manifest streams must be a mapping")

    missing_required_streams = [key for key in REQUIRED_STREAM_KEYS if key not in streams]
    if missing_required_streams:
        raise ValueError(
            f"Session manifest missing required streams: {missing_required_streams}"
        )

    unknown_streams = sorted(set(streams.keys()) - set(SESSION_STREAM_SPECS.keys()))
    if unknown_streams:
        raise ValueError(f"Session manifest has unsupported streams: {unknown_streams}")

    for stream_key, filename in streams.items():
        spec = _require_known_stream_key(stream_key)
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"Session stream '{stream_key}' filename must be a non-empty string")
        if filename != spec.filename:
            raise ValueError(
                f"Session stream '{stream_key}' must use canonical filename '{spec.filename}'"
            )
        if Path(filename).name != filename:
            raise ValueError(
                f"Session stream '{stream_key}' must be a direct filename, got '{filename}'"
            )

    rates_hz = manifest["rates_hz"]
    if not isinstance(rates_hz, Mapping):
        raise ValueError("Session manifest rates_hz must be a mapping")

    allowed_rate_keys = set(CANONICAL_STREAM_RATES_HZ.keys())
    unknown_rate_keys = sorted(set(rates_hz.keys()) - allowed_rate_keys)
    if unknown_rate_keys:
        raise ValueError(f"Session manifest has unsupported rate keys: {unknown_rate_keys}")

    missing_rate_keys = [
        key
        for key in streams
        if SESSION_STREAM_SPECS[key].rate_hz is not None and key not in rates_hz
    ]
    if missing_rate_keys:
        raise ValueError(f"Session manifest missing required rate keys: {missing_rate_keys}")

    for stream_key, rate_hz in rates_hz.items():
        spec = _require_known_stream_key(stream_key)
        if spec.rate_hz is None:
            raise ValueError(
                f"Session stream '{stream_key}' does not define a canonical fixed rate"
            )
        if stream_key not in streams:
            raise ValueError(
                f"Session manifest rate '{stream_key}' requires matching stream entry"
            )
        if not isinstance(rate_hz, int):
            raise ValueError(f"Session rate '{stream_key}' must be an integer")
        if rate_hz != spec.rate_hz:
            raise ValueError(
                f"Session rate '{stream_key}' must be {spec.rate_hz} Hz, got {rate_hz}"
            )


def find_latest_session_manifest(logs_directory: str | Path = "logs") -> Path:
    """Return the latest session manifest beneath the logs directory."""

    logs_path = Path(logs_directory)
    session_manifests = sorted(logs_path.glob("session_*/manifest.json"))
    if not session_manifests:
        session_manifests = sorted(logs_path.rglob("manifest.json"))
    if not session_manifests:
        raise FileNotFoundError(f"Could not find any replay session manifest under {logs_path}")
    return session_manifests[-1]


def resolve_session_manifest_path(
    session_path: str | Path | None,
    *,
    logs_directory: str | Path = "logs",
) -> Path:
    """Resolve a replay session folder or manifest path to ``manifest.json``."""

    if session_path is None:
        return find_latest_session_manifest(logs_directory)

    candidate = Path(session_path)
    if candidate.is_dir():
        manifest_path = candidate / SESSION_MANIFEST_FILENAME
    else:
        manifest_path = candidate

    if manifest_path.name != SESSION_MANIFEST_FILENAME:
        raise ValueError(
            "Replay session path must be a session folder or a manifest.json file"
        )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Replay session manifest not found: {manifest_path}")
    return manifest_path


def load_replay_session(
    session_path: str | Path | None = None,
    *,
    logs_directory: str | Path = "logs",
) -> ReplaySession:
    """Load one replay session from a folder or manifest path."""

    manifest_path = resolve_session_manifest_path(session_path, logs_directory=logs_directory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_session_manifest(manifest)

    session_dir = manifest_path.parent
    stream_paths: dict[str, Path] = {}
    stream_frames: dict[str, pd.DataFrame] = {}
    for stream_key, filename in manifest["streams"].items():
        spec = _require_known_stream_key(stream_key)
        stream_path = session_dir / filename
        if not stream_path.exists():
            raise FileNotFoundError(
                f"Replay session stream '{stream_key}' not found: {stream_path}"
            )
        stream_paths[stream_key] = stream_path
        stream_frames[stream_key] = _load_session_stream_frame(
            stream_path,
            stream_key=stream_key,
            spec=spec,
        )

    return ReplaySession(
        session_dir=session_dir,
        manifest_path=manifest_path,
        manifest=dict(manifest),
        stream_paths=stream_paths,
        stream_frames=stream_frames,
        truth=stream_frames["truth"],
        imu=stream_frames["imu"],
        baro=stream_frames["baro"],
        gps=stream_frames["gps"],
        mag=stream_frames.get("mag"),
    )


def _require_known_stream_key(stream_key: str) -> SessionStreamSpec:
    try:
        return SESSION_STREAM_SPECS[stream_key]
    except KeyError as exc:
        raise ValueError(f"Unknown session stream '{stream_key}'") from exc


def _require_non_empty_string(manifest: Mapping[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Session manifest field '{key}' must be a non-empty string")
    return value


def _require_finite_number(
    manifest: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = manifest.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Session manifest field '{key}' must be a finite number")
    numeric_value = float(value)
    if minimum is not None and numeric_value < minimum:
        raise ValueError(f"Session manifest field '{key}' must be >= {minimum}")
    if maximum is not None and numeric_value > maximum:
        raise ValueError(f"Session manifest field '{key}' must be <= {maximum}")
    return numeric_value


def _load_session_stream_frame(
    stream_path: Path,
    *,
    stream_key: str,
    spec: SessionStreamSpec,
) -> pd.DataFrame:
    frame = pd.read_csv(stream_path)
    expected_columns = list(spec.columns)
    actual_columns = frame.columns.tolist()
    if actual_columns != expected_columns:
        raise ValueError(
            f"Replay session stream '{stream_key}' columns must be {expected_columns}, got {actual_columns}"
        )
    if frame.empty:
        raise ValueError(f"Replay session stream '{stream_key}' is empty")

    frame = frame.copy()
    frame[TIME_COLUMN] = pd.to_numeric(frame[TIME_COLUMN], errors="coerce")
    if frame[TIME_COLUMN].isna().any():
        raise ValueError(
            f"Replay session stream '{stream_key}' has non-numeric timestamps in {stream_path}"
        )

    time_values = frame[TIME_COLUMN].to_numpy(dtype=float)
    if (time_values < 0.0).any():
        raise ValueError(
            f"Replay session stream '{stream_key}' has negative timestamps in {stream_path}"
        )
    if len(time_values) > 1 and (pd.Series(time_values).diff().iloc[1:] <= 0.0).any():
        raise ValueError(
            f"Replay session stream '{stream_key}' timestamps must be strictly increasing"
        )

    if stream_key in {"truth", "imu", "baro", "gps", "mag"}:
        for column in spec.columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.reset_index(drop=True)


def merge_replay_session_sensors(
    replay_session: ReplaySession,
) -> pd.DataFrame:
    """Build a merged compatibility frame for offline analysis tools."""

    return _merge_sensor_stream_frames(replay_session.stream_frames)


def _merge_sensor_stream_frames(stream_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for stream_key in ("imu", "baro", "gps", "mag"):
        frame = stream_frames.get(stream_key)
        if frame is None or frame.empty:
            continue
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(frame, on=TIME_COLUMN, how="outer")

    if merged is None:
        raise ValueError("Replay session did not contain any sensor streams")
    return merged.sort_values(TIME_COLUMN).reset_index(drop=True)


__all__ = [
    "BARO_COLUMNS",
    "CANONICAL_STREAM_RATES_HZ",
    "DEVICE_EVENT_COLUMNS",
    "ESTIMATOR_FEEDBACK_COLUMNS",
    "GPS_COLUMNS",
    "IMU_COLUMNS",
    "MAG_COLUMNS",
    "OPTIONAL_MANIFEST_KEYS",
    "OPTIONAL_STREAM_KEYS",
    "REQUIRED_MANIFEST_KEYS",
    "REQUIRED_STREAM_KEYS",
    "ReplaySession",
    "SESSION_MANIFEST_FILENAME",
    "SESSION_MANIFEST_SCHEMA_FILENAME",
    "SESSION_SCHEMA_VERSION",
    "SESSION_SPEC_FILENAME",
    "SESSION_STREAM_SPECS",
    "TIME_COLUMN",
    "TRUTH_COLUMNS",
    "SessionStreamSpec",
    "build_session_manifest",
    "default_stream_filenames",
    "find_latest_session_manifest",
    "load_replay_session",
    "load_manifest_schema",
    "manifest_schema_path",
    "merge_replay_session_sensors",
    "resolve_session_manifest_path",
    "session_contract_directory",
    "session_spec_path",
    "validate_session_manifest",
]
