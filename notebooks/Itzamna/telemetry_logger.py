import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sim.sitl.session import (
    BARO_COLUMNS,
    GPS_COLUMNS,
    IMU_COLUMNS,
    MAG_COLUMNS,
    TRUTH_COLUMNS,
    build_session_manifest,
)

TRUTH_SAMPLING_RATE_HZ = 500


def _sensor_key(sensor):
    """Map RocketPy sensor object to a stable filename key."""
    class_name = sensor.__class__.__name__.lower()
    sensor_name = str(getattr(sensor, "name", "")).lower()

    if "accelerometer" in class_name or "accelerometer" in sensor_name:
        return "accelerometer"
    if "gyroscope" in class_name or "gyroscope" in sensor_name:
        return "gyroscope"
    if "barometer" in class_name or "barometer" in sensor_name:
        return "barometer"
    if "gnss" in class_name or "gnss" in sensor_name or "gps" in sensor_name:
        return "gnss"
    if "magnetometer" in class_name or "magnetometer" in sensor_name or sensor_name == "mag":
        return "mag"
    return None


def _to_dataframe_from_measured_data(sensor):
    """Build DataFrame from RocketPy measured_data list: [(t, v1, v2,...), ...]."""
    rows = list(getattr(sensor, "measured_data", []) or [])
    if not rows:
        return pd.DataFrame()

    first = rows[0]
    width = len(first) if isinstance(first, (list, tuple)) else 1

    if width == 2 and isinstance(first[1], (list, tuple)):
        # Format: (time, (x, y, z))
        dim = len(first[1])
        columns = ["time_s"] + [f"v{i}" for i in range(1, dim + 1)]
        data = [[r[0], *list(r[1])] for r in rows]
        return pd.DataFrame(data, columns=columns)

    # Format: (time, x, y, z, ...)
    columns = ["time_s"] + [f"v{i}" for i in range(1, width)]
    return pd.DataFrame(rows, columns=columns)


def _rename_sensor_axes(sensor_name, df):
    """Rename vector columns to canonical per-stream axis names."""
    if df.empty or "time_s" not in df.columns:
        return df

    canonical_sensor_name = "magnetometer" if sensor_name == "mag" else sensor_name
    non_time_cols = [c for c in df.columns if c != "time_s"]
    if canonical_sensor_name in {"accelerometer", "gyroscope", "gnss", "magnetometer"} and len(non_time_cols) >= 3:
        axis_map = {
            non_time_cols[0]: f"{canonical_sensor_name}_x",
            non_time_cols[1]: f"{canonical_sensor_name}_y",
            non_time_cols[2]: f"{canonical_sensor_name}_z",
        }
        # Keep extra dimensions if present (e.g., additional GNSS outputs).
        for idx, col in enumerate(non_time_cols[3:], start=4):
            axis_map[col] = f"{canonical_sensor_name}_v{idx}"
        return df.rename(columns=axis_map)

    # Generic naming for scalar or unknown sensor outputs.
    generic_map = {col: f"{canonical_sensor_name}_{col}" for col in non_time_cols}
    return df.rename(columns=generic_map)


def _dedupe_frame_on_time(df):
    """Collapse duplicate timestamps to the last sample for replay compatibility."""
    if df.empty or "time_s" not in df.columns:
        return df

    deduped = (
        df.sort_values("time_s", kind="stable")
        .drop_duplicates(subset=["time_s"], keep="last")
        .reset_index(drop=True)
    )
    return deduped.loc[:, df.columns.tolist()]


def _collect_sensors(flight):
    """Collect sensor objects from flight.sensors or flight.rocket.sensors."""
    sensors = {}
    containers = []

    if hasattr(flight, "sensors"):
        containers.append(flight.sensors)
    if hasattr(flight, "rocket") and hasattr(flight.rocket, "sensors"):
        containers.append(flight.rocket.sensors)

    for container in containers:
        try:
            items = list(container)
        except Exception:
            continue

        for item in items:
            sensor = None
            try:
                sensor = item[0]  # component_tuple format
            except Exception:
                sensor = item

            if not hasattr(sensor, "measured_data"):
                continue

            key = _sensor_key(sensor)
            if key and key not in sensors:
                sensors[key] = sensor

    return sensors


def _extract_simulation_datetime(flight):
    """Best-effort extraction of simulation datetime from Flight/Environment."""
    # Most robust fallback is current local time when export is generated.
    now = datetime.now()

    env = getattr(flight, "environment", None)
    if env is None:
        return now

    # Try common RocketPy environment datetime attribute names.
    for attr_name in ("datetime_date", "date", "datetime"):
        value = getattr(env, attr_name, None)
        if value is None:
            continue

        if isinstance(value, datetime):
            return value

        # Accept tuple/list like (year, month, day, hour)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                year = int(value[0])
                month = int(value[1])
                day = int(value[2])
                hour = int(value[3]) if len(value) >= 4 else 0
                minute = int(value[4]) if len(value) >= 5 else 0
                second = int(value[5]) if len(value) >= 6 else 0
                return datetime(year, month, day, hour, minute, second)
            except Exception:
                continue

    return now


def _simulation_timestamp(flight):
    """Short timestamp for filenames: YYMMDD_HHMMSS."""
    return _extract_simulation_datetime(flight).strftime("%y%m%d_%H%M%S")


def _utc_timestamp() -> str:
    """Return the export creation timestamp in UTC ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_session_directory(logs_dir, session_id):
    """Return the canonical per-session export directory path."""
    return Path(logs_dir) / f"session_{session_id}"


def _call_flight_series(flight, attribute_name, times_s):
    """Evaluate one RocketPy flight state callable over the target timeline."""
    accessor = getattr(flight, attribute_name)
    return np.array([float(accessor(float(time_s))) for time_s in times_s], dtype=float)


def _truth_timebase(flight, rate_hz=TRUTH_SAMPLING_RATE_HZ):
    """Build the canonical uniform truth timeline up to ``flight.t_final``."""
    dt_s = 1.0 / float(rate_hz)
    t_final = max(float(getattr(flight, "t_final", 0.0)), 0.0)
    if t_final <= 0.0:
        return np.array([0.0], dtype=float)

    sample_count = int(np.floor(t_final / dt_s)) + 1
    times_s = np.arange(sample_count, dtype=float) * dt_s
    times_s = np.clip(times_s, 0.0, t_final)
    return np.unique(times_s)


def _build_truth_frame(flight):
    """Sample the RocketPy flight object onto the canonical truth timeline."""
    times_s = _truth_timebase(flight)
    truth = pd.DataFrame(
        {
            "time_s": times_s,
            "x_m": _call_flight_series(flight, "x", times_s),
            "y_m": _call_flight_series(flight, "y", times_s),
            "z_m": _call_flight_series(flight, "z", times_s),
            "vx_mps": _call_flight_series(flight, "vx", times_s),
            "vy_mps": _call_flight_series(flight, "vy", times_s),
            "vz_mps": _call_flight_series(flight, "vz", times_s),
            "e0": _call_flight_series(flight, "e0", times_s),
            "e1": _call_flight_series(flight, "e1", times_s),
            "e2": _call_flight_series(flight, "e2", times_s),
            "e3": _call_flight_series(flight, "e3", times_s),
            "w1_radps": _call_flight_series(flight, "w1", times_s),
            "w2_radps": _call_flight_series(flight, "w2", times_s),
            "w3_radps": _call_flight_series(flight, "w3", times_s),
        }
    )
    return truth.loc[:, list(TRUTH_COLUMNS)]


def _measurement_frame(sensor, sensor_name):
    """Convert one RocketPy sensor measured_data payload into canonical columns."""
    df = _to_dataframe_from_measured_data(sensor)
    if df.empty or "time_s" not in df.columns:
        return pd.DataFrame()
    renamed = _rename_sensor_axes(sensor_name, df)
    return _dedupe_frame_on_time(renamed)


def _merge_frames_on_time(frames, columns):
    """Outer-join sensor subframes while preserving canonical column order."""
    merged = None
    for frame in frames:
        if frame.empty:
            continue
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(frame, on="time_s", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame(columns=columns)

    merged = _dedupe_frame_on_time(merged)
    for column in columns:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged.loc[:, list(columns)]


def _first_available_numeric(frame, column):
    """Return the first finite value from a column, or ``None`` if unavailable."""
    if column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def _resolve_reference_latitude_deg(flight, gps_frame):
    """Resolve session reference latitude from environment or first GPS sample."""
    env = getattr(flight, "environment", None)
    latitude = getattr(env, "latitude", None)
    if latitude is not None:
        return float(latitude)

    gps_latitude = _first_available_numeric(gps_frame, "gnss_x")
    if gps_latitude is not None:
        return gps_latitude
    return float(getattr(flight, "latitude")(0.0))


def _resolve_reference_longitude_deg(flight, gps_frame):
    """Resolve session reference longitude from environment or first GPS sample."""
    env = getattr(flight, "environment", None)
    longitude = getattr(env, "longitude", None)
    if longitude is not None:
        return float(longitude)

    gps_longitude = _first_available_numeric(gps_frame, "gnss_y")
    if gps_longitude is not None:
        return gps_longitude
    return float(getattr(flight, "longitude")(0.0))


def _resolve_reference_altitude_m(flight, truth_frame):
    """Resolve session reference altitude from environment or truth start state."""
    env = getattr(flight, "environment", None)
    elevation = getattr(env, "elevation", None)
    if elevation is not None:
        return float(elevation)

    truth_altitude = _first_available_numeric(truth_frame, "z_m")
    if truth_altitude is not None:
        return truth_altitude
    return float(getattr(flight, "z")(0.0))


def _estimate_sea_level_pressure_pa(reference_pressure_pa, reference_altitude_m):
    """Infer sea-level pressure from one pressure-altitude sample."""
    altitude_factor = max(1.0 - float(reference_altitude_m) / 44330.0, 1e-6)
    return float(reference_pressure_pa) / altitude_factor ** 5.255


def _resolve_sea_level_pressure_pa(flight, baro_frame, reference_altitude_m):
    """Resolve sea-level pressure metadata from barometer or environment state."""
    reference_pressure_pa = _first_available_numeric(baro_frame, "barometer_v1")
    if reference_pressure_pa is not None:
        return _estimate_sea_level_pressure_pa(reference_pressure_pa, reference_altitude_m)

    env = getattr(flight, "environment", None)
    pressure = getattr(env, "pressure", None)
    if pressure is not None:
        return float(pressure)
    return 101325.0


def _write_stream_csv(frame, path, columns):
    """Persist one canonical stream CSV."""
    frame.loc[:, list(columns)].to_csv(path, index=False)


def _build_legacy_sensor_frame(imu_frame, baro_frame, gps_frame, mag_frame):
    """Build the merged sensor CSV expected by older replay consumers."""
    frames = [imu_frame, baro_frame, gps_frame]
    columns = [*IMU_COLUMNS, *BARO_COLUMNS[1:], *GPS_COLUMNS[1:]]
    if not mag_frame.empty:
        frames.append(mag_frame)
        columns.extend(MAG_COLUMNS[1:])
    return _merge_frames_on_time(frames, columns)


def _write_legacy_compatibility_logs(
    *,
    logs_path,
    session_id,
    truth_frame,
    imu_frame,
    baro_frame,
    gps_frame,
    mag_frame,
):
    """Persist compatibility CSVs for stale notebooks and legacy replay helpers."""
    sensor_frame = _build_legacy_sensor_frame(imu_frame, baro_frame, gps_frame, mag_frame)
    _write_stream_csv(
        sensor_frame,
        logs_path / f"virtual_sensors_full_rate_{session_id}.csv",
        sensor_frame.columns,
    )
    _write_stream_csv(
        truth_frame,
        logs_path / f"flight_kinematics_{session_id}.csv",
        TRUTH_COLUMNS,
    )


def export_telemetry(flight, logs_dir="../../logs"):
    """Export one self-contained replay session directory."""
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    print(f"Exporting logs to: {logs_path.resolve()} ...\\n")

    session_id = _simulation_timestamp(flight)
    session_dir = _build_session_directory(logs_path, session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    sensors = _collect_sensors(flight)
    if not sensors:
        raise ValueError("No sensors found in flight/rocket; cannot build a replay session.")
    missing_required = [
        sensor_key
        for sensor_key in ("accelerometer", "gyroscope", "barometer", "gnss")
        if sensor_key not in sensors
    ]
    if missing_required:
        raise ValueError(
            f"Missing required sensors for session export: {', '.join(missing_required)}"
        )

    truth_frame = _build_truth_frame(flight)
    imu_frame = _merge_frames_on_time(
        [
            _measurement_frame(sensors["accelerometer"], "accelerometer"),
            _measurement_frame(sensors["gyroscope"], "gyroscope"),
        ],
        IMU_COLUMNS,
    )
    baro_frame = _measurement_frame(sensors["barometer"], "barometer").loc[:, list(BARO_COLUMNS)]
    gps_frame = _measurement_frame(sensors["gnss"], "gnss").loc[:, list(GPS_COLUMNS)]

    optional_streams = []
    mag_frame = pd.DataFrame(columns=MAG_COLUMNS)
    if "mag" in sensors:
        mag_frame = _measurement_frame(sensors["mag"], "mag").loc[:, list(MAG_COLUMNS)]
        if not mag_frame.empty:
            optional_streams.append("mag")

    reference_latitude_deg = _resolve_reference_latitude_deg(flight, gps_frame)
    reference_longitude_deg = _resolve_reference_longitude_deg(flight, gps_frame)
    reference_altitude_m = _resolve_reference_altitude_m(flight, truth_frame)
    sea_level_pressure_pa = _resolve_sea_level_pressure_pa(
        flight,
        baro_frame,
        reference_altitude_m,
    )

    manifest = build_session_manifest(
        session_id=session_id,
        vehicle_name=str(getattr(getattr(flight, "rocket", None), "name", "") or "Rocket"),
        generated_at_utc=_utc_timestamp(),
        reference_latitude_deg=reference_latitude_deg,
        reference_longitude_deg=reference_longitude_deg,
        reference_altitude_m=reference_altitude_m,
        sea_level_pressure_pa=sea_level_pressure_pa,
        include_optional_streams=tuple(optional_streams),
    )

    _write_stream_csv(truth_frame, session_dir / "truth.csv", TRUTH_COLUMNS)
    _write_stream_csv(imu_frame, session_dir / "imu.csv", IMU_COLUMNS)
    _write_stream_csv(baro_frame, session_dir / "baro.csv", BARO_COLUMNS)
    _write_stream_csv(gps_frame, session_dir / "gps.csv", GPS_COLUMNS)
    if optional_streams:
        _write_stream_csv(mag_frame, session_dir / "mag.csv", MAG_COLUMNS)
    _write_legacy_compatibility_logs(
        logs_path=logs_path,
        session_id=session_id,
        truth_frame=truth_frame,
        imu_frame=imu_frame,
        baro_frame=baro_frame,
        gps_frame=gps_frame,
        mag_frame=mag_frame,
    )

    (session_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"  Saved session manifest -> {session_dir / 'manifest.json'}")
    print(f"  Saved truth stream -> {session_dir / 'truth.csv'}")
    print(f"  Saved IMU stream -> {session_dir / 'imu.csv'}")
    print(f"  Saved barometer stream -> {session_dir / 'baro.csv'}")
    print(f"  Saved GPS stream -> {session_dir / 'gps.csv'}")
    if optional_streams:
        print(f"  Saved magnetometer stream -> {session_dir / 'mag.csv'}")
    print(f"  Saved legacy merged sensors -> {logs_path / f'virtual_sensors_full_rate_{session_id}.csv'}")
    print(f"  Saved legacy kinematics -> {logs_path / f'flight_kinematics_{session_id}.csv'}")
    print("\\nTelemetry export complete.")
    return session_dir
