import os
from datetime import datetime
import pandas as pd


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
    """Rename vector columns to x/y/z for selected sensors in merged export."""
    if df.empty or "time_s" not in df.columns:
        return df

    non_time_cols = [c for c in df.columns if c != "time_s"]
    if sensor_name in {"accelerometer", "gyroscope", "gnss"} and len(non_time_cols) >= 3:
        axis_map = {
            non_time_cols[0]: f"{sensor_name}_x",
            non_time_cols[1]: f"{sensor_name}_y",
            non_time_cols[2]: f"{sensor_name}_z",
        }
        # Keep extra dimensions if present (e.g., additional GNSS outputs).
        for idx, col in enumerate(non_time_cols[3:], start=4):
            axis_map[col] = f"{sensor_name}_v{idx}"
        return df.rename(columns=axis_map)

    # Generic naming for scalar or unknown sensor outputs.
    generic_map = {col: f"{sensor_name}_{col}" for col in non_time_cols}
    return df.rename(columns=generic_map)


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


def export_telemetry(flight, logs_dir="../../logs"):
    """Export exactly two files: source-of-truth kinematics and full-rate merged sensors."""
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Exporting logs to: {os.path.abspath(logs_dir)} ...\\n")

    timestamp = _simulation_timestamp(flight)
    kinematics_filename = f"flight_kinematics_{timestamp}.csv"
    merged_filename = f"virtual_sensors_full_rate_{timestamp}.csv"
    kinematics_path = os.path.join(logs_dir, kinematics_filename)
    merged_path = os.path.join(logs_dir, merged_filename)

    # Remove legacy per-sensor exports so only two files remain from this logger flow.
    for filename in os.listdir(logs_dir):
        if filename.startswith("sensor_") and filename.endswith(".csv"):
            try:
                os.remove(os.path.join(logs_dir, filename))
            except OSError:
                pass

    # 1) RocketPy built-in flight kinematics export.
    try:
        flight.export_data(kinematics_path)
        print(f"  Saved source-of-truth flight kinematics -> {kinematics_path}")
    except Exception as e:
        print(f"  Warning: could not export flight kinematics: {e}")

    # 2) Build merged full-rate sensor table from measured_data.
    sensors = _collect_sensors(flight)
    if not sensors:
        print("  Warning: no sensors found in flight/rocket.")
        return

    sensor_frames = {}

    for name, sensor in sensors.items():
        df = _to_dataframe_from_measured_data(sensor)
        if not df.empty and "time_s" in df.columns:
            sensor_frames[name] = _rename_sensor_axes(name, df)

    # 3) Merge at full resolution: outer join on time union.
    merged = None
    for _, frame in sensor_frames.items():
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="time_s", how="outer")

    if merged is not None and not merged.empty:
        merged = merged.sort_values("time_s").reset_index(drop=True)
        merged.to_csv(merged_path, index=False)
        print(f"  Saved merged full-rate sensors -> {merged_path}")
    else:
        print("  Warning: merged full-rate sensor CSV was not created (no sensor samples).")

    print("\\nTelemetry export complete.")
