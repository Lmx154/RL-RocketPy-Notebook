"""Normalize Lonestar flight telemetry exports into canonical analysis streams."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ATM_TO_PA = 101_325.0
FEET_TO_METERS = 0.3048
FEET_PER_SECOND_TO_MPS = 0.3048
DEFAULT_BARO_ALIGN_COARSE_STEP_S = 0.05
DEFAULT_BARO_ALIGN_REFINE_STEP_S = 0.005
DEFAULT_MERGED_TIME_STEP_S = 0.02
DEFAULT_MERGE_MIN_CONFIDENCE = 0.25
DEFAULT_GPS_ALIGN_COARSE_STEP_S = 0.05
DEFAULT_GPS_ALIGN_REFINE_STEP_S = 0.005
DEFAULT_REPLAY_VEHICLE_NAME = "Itzamna"
EARTH_RADIUS_M = 6_378_137.0
DEFAULT_MERGE_TIMEBASE_SOURCE = "marv_primary_imu"
MERGE_TIMEBASE_CHOICES = (
    "shared_uniform",
    "marv_primary_imu",
    "marv_baro",
    "marv_aux_imu",
)

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

FEATHERWEIGHT_REQUIRED_COLUMNS = (
    "Flight_Time_(s)",
    "Temperature_(F)",
    "Baro_Press_(atm)",
    "Baro_Altitude_ASL_(feet)",
    "Baro_Altitude_AGL_(feet)",
    "Batt_Volts",
    "Velocity_Up",
    "Velocity_DR",
    "Velocity_CR",
    "Inertial_Altitude",
    "Inertial_DR_Position",
    "Inertial_CR_position",
    "Tilt_Angle_(deg)",
    "Future_Angle_(deg)",
    "Roll_Angle_(deg)",
    "Liftoff",
    "Apogee",
    "Press_Increasing",
    "Burnout_Coast",
    "Apo_fired",
    "Main_fired",
    "3rd_fired",
    "4th_fired",
    "Normal_Ascent",
    "Accel_Vel_LE_0",
    "ECI_Vvel_le_0",
    "Tilt Exceeded 90deg",
)

FEATHERWEIGHT_EVENT_COLUMN_MAP = {
    "Liftoff": "liftoff",
    "Apogee": "apogee",
    "Press_Increasing": "press_increasing",
    "Burnout_Coast": "burnout_coast",
    "Apo_fired": "apo_fired",
    "Main_fired": "main_fired",
    "3rd_fired": "third_fired",
    "4th_fired": "fourth_fired",
    "Normal_Ascent": "normal_ascent",
    "Accel_Vel_LE_0": "accel_vel_le_zero",
    "ECI_Vvel_le_0": "eci_vvel_le_zero",
    "Tilt Exceeded 90deg": "tilt_exceeded_90deg",
}

FEATHERWEIGHT_GPS_REQUIRED_COLUMNS = (
    "UTCTIME",
    "UNIXTIME",
    "ALT",
    "LAT",
    "LON",
    "#SATS",
    "FIX",
    "HORZV",
    "VERTV",
    "HEAD",
    "FLAGS",
    ">40",
    ">32",
    ">24",
    "RSSI",
    "BATT",
    "Altitude AGL",
    "Launch detection",
    "Apogee detection",
    "Landing detection",
    "Distance (feet)",
)

MARV_REQUIRED_COLUMNS = (
    "log_us",
    "imu_state",
    "imu_sample_us",
    "imu_lagged",
    "imu_ax_mps2",
    "imu_ay_mps2",
    "imu_az_mps2",
    "imu_gx_rad_s",
    "imu_gy_rad_s",
    "imu_gz_rad_s",
    "aux_imu_state",
    "aux_imu_sample_us",
    "aux_imu_lagged",
    "aux_imu_ax_mps2",
    "aux_imu_ay_mps2",
    "aux_imu_az_mps2",
    "aux_imu_gx_rad_s",
    "aux_imu_gy_rad_s",
    "aux_imu_gz_rad_s",
    "baro_state",
    "baro_sample_us",
    "baro_lagged",
    "baro_pressure_pa",
    "baro_temp_c",
)

MARV_PRIMARY_IMU_RENAME = {
    "imu_ax_mps2": "accelerometer_x_mps2",
    "imu_ay_mps2": "accelerometer_y_mps2",
    "imu_az_mps2": "accelerometer_z_mps2",
    "imu_gx_rad_s": "gyroscope_x_rad_s",
    "imu_gy_rad_s": "gyroscope_y_rad_s",
    "imu_gz_rad_s": "gyroscope_z_rad_s",
}

MARV_AUX_IMU_RENAME = {
    "aux_imu_ax_mps2": "accelerometer_x_mps2",
    "aux_imu_ay_mps2": "accelerometer_y_mps2",
    "aux_imu_az_mps2": "accelerometer_z_mps2",
    "aux_imu_gx_rad_s": "gyroscope_x_rad_s",
    "aux_imu_gy_rad_s": "gyroscope_y_rad_s",
    "aux_imu_gz_rad_s": "gyroscope_z_rad_s",
}


@dataclass(slots=True)
class NormalizedSource:
    source_name: str
    source_path: Path
    metadata: dict[str, Any]
    streams: dict[str, pd.DataFrame]

    def summary(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_path": str(self.source_path),
            "metadata": self.metadata,
            "streams": {
                stream_name: _frame_summary(frame)
                for stream_name, frame in self.streams.items()
            },
        }


def normalize_featherweight_csv(path: str | Path) -> NormalizedSource:
    source_path = _resolve_input_path(path)
    raw = pd.read_csv(source_path)
    _require_columns(raw, FEATHERWEIGHT_REQUIRED_COLUMNS, source_path)

    numeric_columns = tuple(
        column for column in FEATHERWEIGHT_REQUIRED_COLUMNS if column != "Flight_Time_(s)"
    ) + ("Flight_Time_(s)",)
    raw = raw.copy()
    _coerce_numeric_columns(raw, numeric_columns)

    if raw["Flight_Time_(s)"].isna().any():
        raise ValueError(f"Featherweight file has non-numeric Flight_Time_(s): {source_path}")

    time_s = raw["Flight_Time_(s)"].to_numpy(dtype=float)
    _validate_monotonic_time(time_s, source_path, "Flight_Time_(s)")

    baro = pd.DataFrame(
        {
            "time_s": time_s,
            "pressure_pa": raw["Baro_Press_(atm)"] * ATM_TO_PA,
            "pressure_atm": raw["Baro_Press_(atm)"],
            "altitude_asl_m": raw["Baro_Altitude_ASL_(feet)"] * FEET_TO_METERS,
            "altitude_agl_m": raw["Baro_Altitude_AGL_(feet)"] * FEET_TO_METERS,
            "temperature_c": _fahrenheit_to_c(raw["Temperature_(F)"]),
            "temperature_f": raw["Temperature_(F)"],
            "battery_v": raw["Batt_Volts"],
        }
    )

    navigation = pd.DataFrame(
        {
            "time_s": time_s,
            "vertical_velocity_mps": raw["Velocity_Up"] * FEET_PER_SECOND_TO_MPS,
            "downrange_velocity_mps": raw["Velocity_DR"] * FEET_PER_SECOND_TO_MPS,
            "crossrange_velocity_mps": raw["Velocity_CR"] * FEET_PER_SECOND_TO_MPS,
            "inertial_altitude_m": raw["Inertial_Altitude"] * FEET_TO_METERS,
            "inertial_downrange_m": raw["Inertial_DR_Position"] * FEET_TO_METERS,
            "inertial_crossrange_m": raw["Inertial_CR_position"] * FEET_TO_METERS,
            "tilt_deg": raw["Tilt_Angle_(deg)"],
            "future_angle_deg": raw["Future_Angle_(deg)"],
            "roll_angle_deg": raw["Roll_Angle_(deg)"],
        }
    )

    events = pd.DataFrame({"time_s": time_s})
    for raw_column, canonical_column in FEATHERWEIGHT_EVENT_COLUMN_MAP.items():
        values = pd.to_numeric(raw[raw_column], errors="coerce").fillna(0).astype(int)
        events[canonical_column] = values.astype(bool)

    metadata = {
        "clock_reference": "Flight_Time_(s)",
        "clock_reference_units": "seconds",
        "wall_clock_time_reliable": False,
        "pressure_alignment_ready": True,
        "gps_present": False,
    }

    return NormalizedSource(
        source_name="featherweight",
        source_path=source_path,
        metadata=metadata,
        streams={
            "baro": baro,
            "navigation": navigation,
            "events": events,
        },
    )


def normalize_featherweight_gps_csv(path: str | Path) -> NormalizedSource:
    source_path = _resolve_input_path(path)
    raw = pd.read_csv(source_path)
    _require_columns(raw, FEATHERWEIGHT_GPS_REQUIRED_COLUMNS, source_path)

    raw = raw.copy()
    numeric_columns = (
        "UNIXTIME",
        "ALT",
        "LAT",
        "LON",
        "#SATS",
        "FIX",
        "HORZV",
        "VERTV",
        "HEAD",
        "FLAGS",
        ">40",
        ">32",
        ">24",
        "RSSI",
        "BATT",
        "Altitude AGL",
        "Distance (feet)",
    )
    _coerce_numeric_columns(raw, numeric_columns)

    if raw["UNIXTIME"].isna().any():
        raise ValueError(f"GPS file has non-numeric UNIXTIME: {source_path}")

    unix_time_s = raw["UNIXTIME"].to_numpy(dtype=float)
    _validate_monotonic_time(unix_time_s, source_path, "UNIXTIME")
    time_s = unix_time_s - float(unix_time_s[0])

    navigation = pd.DataFrame(
        {
            "time_s": time_s,
            "unix_time_s": unix_time_s,
            "utc_time": raw["UTCTIME"].astype(str),
            "gps_altitude_asl_m": raw["ALT"] * FEET_TO_METERS,
            "gps_altitude_agl_m": raw["Altitude AGL"] * FEET_TO_METERS,
            "latitude_deg": raw["LAT"],
            "longitude_deg": raw["LON"],
            "satellite_count": raw["#SATS"],
            "fix_type": raw["FIX"],
            "horizontal_speed_mps": raw["HORZV"] * FEET_PER_SECOND_TO_MPS,
            "vertical_speed_mps": raw["VERTV"] * FEET_PER_SECOND_TO_MPS,
            "course_heading_deg": raw["HEAD"],
            "battery_v": raw["BATT"],
            "distance_ft": raw["Distance (feet)"],
            "distance_m": raw["Distance (feet)"] * FEET_TO_METERS,
            "flags": raw["FLAGS"],
            "rssi_dbm": raw["RSSI"],
            "count_gt_40": raw[">40"],
            "count_gt_32": raw[">32"],
            "count_gt_24": raw[">24"],
        }
    )

    events = pd.DataFrame(
        {
            "time_s": time_s,
            "launch_detected": _coerce_bool_series(raw["Launch detection"]),
            "apogee_detected": _coerce_bool_series(raw["Apogee detection"]),
            "landing_detected": _coerce_bool_series(raw["Landing detection"]),
        }
    )

    metadata = {
        "clock_reference": "UNIXTIME",
        "clock_reference_units": "seconds",
        "wall_clock_time_reliable": True,
        "pressure_alignment_ready": False,
        "gps_present": True,
        "course_heading_present": True,
    }

    return NormalizedSource(
        source_name="featherweight_gps",
        source_path=source_path,
        metadata=metadata,
        streams={
            "navigation": navigation,
            "events": events,
        },
    )


def normalize_marv_csv(path: str | Path) -> NormalizedSource:
    source_path = _resolve_input_path(path)
    raw = pd.read_csv(source_path)
    _require_columns(raw, MARV_REQUIRED_COLUMNS, source_path)

    numeric_columns = tuple(column for column in MARV_REQUIRED_COLUMNS if column not in {
        "imu_state",
        "aux_imu_state",
        "baro_state",
    })
    raw = raw.copy()
    _coerce_numeric_columns(raw, numeric_columns)

    origin_us = _resolve_marv_origin_us(raw)
    primary_imu = _build_marv_imu_stream(
        raw,
        origin_us=origin_us,
        state_column="imu_state",
        sample_time_column="imu_sample_us",
        lagged_column="imu_lagged",
        rename_map=MARV_PRIMARY_IMU_RENAME,
    )
    aux_imu = _build_marv_imu_stream(
        raw,
        origin_us=origin_us,
        state_column="aux_imu_state",
        sample_time_column="aux_imu_sample_us",
        lagged_column="aux_imu_lagged",
        rename_map=MARV_AUX_IMU_RENAME,
    )
    baro = _build_marv_baro_stream(raw, origin_us=origin_us)

    metadata = {
        "clock_reference": "sensor sample timestamps",
        "clock_reference_units": "microseconds",
        "common_origin_us": int(origin_us),
        "flight_window_isolated": False,
        "gps_present": False,
    }

    return NormalizedSource(
        source_name="marv",
        source_path=source_path,
        metadata=metadata,
        streams={
            "primary_imu": primary_imu,
            "aux_imu": aux_imu,
            "baro": baro,
        },
    )


def write_normalized_sources(
    sources: list[NormalizedSource],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {"sources": {}}
    for source in sources:
        source_summary = source.summary()
        stream_files: dict[str, str] = {}
        for stream_name, frame in source.streams.items():
            filename = f"{source.source_name}_{stream_name}.csv"
            frame.to_csv(output_path / filename, index=False)
            stream_files[stream_name] = filename
        source_summary["stream_files"] = stream_files
        summary["sources"][source.source_name] = source_summary

    summary_path = output_path / "normalization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def isolate_marv_flight_window(
    marv_source: NormalizedSource,
    *,
    pressure_window_s: float = 1.0,
    accel_window_s: float = 0.35,
    activity_threshold: float = 1.5,
    gap_merge_s: float = 1.0,
    margin_s: float = 2.0,
    min_segment_duration_s: float = 5.0,
) -> tuple[NormalizedSource, dict[str, Any]]:
    """Trim MARV streams to the most flight-like contiguous activity window."""

    if marv_source.source_name != "marv":
        raise ValueError("Flight-window isolation currently expects a MARV normalized source")

    baro = marv_source.streams.get("baro")
    primary_imu = marv_source.streams.get("primary_imu")
    if baro is None or baro.empty:
        raise ValueError("MARV source is missing a usable baro stream")
    if primary_imu is None or primary_imu.empty:
        raise ValueError("MARV source is missing a usable primary_imu stream")

    baro_t = baro["time_s"].to_numpy(dtype=float)
    baro_pressure = baro["pressure_pa"].to_numpy(dtype=float)
    imu_t = primary_imu["time_s"].to_numpy(dtype=float)
    accelerometer = primary_imu.loc[
        :,
        [
            "accelerometer_x_mps2",
            "accelerometer_y_mps2",
            "accelerometer_z_mps2",
        ],
    ].to_numpy(dtype=float)

    baro_smooth = _smooth_time_series(baro_t, baro_pressure, pressure_window_s)
    baro_dpdt = np.gradient(baro_smooth, baro_t)
    baro_activity = np.abs(_robust_zscore(baro_dpdt))
    pressure_excursion = np.abs(_robust_zscore(baro_smooth))
    baro_activity = _suppress_edges(baro_activity, baro_t, guard_s=pressure_window_s)
    pressure_excursion = _suppress_edges(
        pressure_excursion,
        baro_t,
        guard_s=0.5 * pressure_window_s,
    )

    accel_norm = np.linalg.norm(accelerometer, axis=1)
    accel_smooth = _smooth_time_series(imu_t, accel_norm, accel_window_s)
    accel_rate = np.gradient(accel_smooth, imu_t)
    imu_activity = np.maximum(
        np.abs(_robust_zscore(accel_rate)),
        0.5 * np.abs(_robust_zscore(accel_smooth)),
    )
    imu_activity = _suppress_edges(imu_activity, imu_t, guard_s=accel_window_s)
    imu_activity_on_baro = np.interp(
        baro_t,
        imu_t,
        imu_activity,
        left=float(imu_activity[0]),
        right=float(imu_activity[-1]),
    )

    combined_activity = (
        0.5 * np.clip(baro_activity, 0.0, 10.0)
        + 0.2 * np.clip(pressure_excursion, 0.0, 10.0)
        + 0.3 * np.clip(imu_activity_on_baro, 0.0, 10.0)
    )
    active_mask = combined_activity >= float(activity_threshold)
    active_mask = _fill_short_false_gaps(
        active_mask,
        max_gap_samples=_duration_to_samples(baro_t, gap_merge_s),
    )

    segments = _extract_mask_segments(active_mask, baro_t, combined_activity)
    eligible_segments = [
        segment
        for segment in segments
        if segment["duration_s"] >= float(min_segment_duration_s)
    ]

    full_start = float(baro_t[0])
    full_end = float(baro_t[-1])
    full_duration = full_end - full_start

    used_full_range = False
    if not eligible_segments:
        window_start = full_start
        window_end = full_end
        segment_count = 0
        peak_activity = float(np.nanmax(combined_activity)) if combined_activity.size else 0.0
        used_full_range = True
    else:
        segment_count = len(eligible_segments)
        peak_activity = float(max(segment["peak_activity"] for segment in eligible_segments))
        window_start = max(
            full_start,
            float(eligible_segments[0]["start_time_s"]) - margin_s,
        )
        window_end = min(
            full_end,
            float(eligible_segments[-1]["end_time_s"]) + margin_s,
        )
        if (window_end - window_start) >= 0.85 * full_duration:
            window_start = full_start
            window_end = full_end
            used_full_range = True

    trimmed_source = _slice_source_by_time(
        marv_source,
        start_time_s=window_start,
        end_time_s=window_end,
    )
    trimmed_metadata = dict(trimmed_source.metadata)
    trimmed_metadata.update(
        {
            "flight_window_isolated": True,
            "flight_window_start_time_s": float(window_start),
            "flight_window_end_time_s": float(window_end),
            "flight_window_used_full_range": bool(used_full_range),
        }
    )
    trimmed_source = NormalizedSource(
        source_name=trimmed_source.source_name,
        source_path=trimmed_source.source_path,
        metadata=trimmed_metadata,
        streams=trimmed_source.streams,
    )

    report = {
        "start_time_s": float(window_start),
        "end_time_s": float(window_end),
        "duration_s": float(window_end - window_start),
        "full_span_start_time_s": full_start,
        "full_span_end_time_s": full_end,
        "full_span_duration_s": full_duration,
        "used_full_range": bool(used_full_range),
        "activity_threshold": float(activity_threshold),
        "segment_count": int(segment_count),
        "peak_activity": peak_activity,
    }
    return trimmed_source, report


def align_baro_sources(
    featherweight_source: NormalizedSource,
    marv_source: NormalizedSource,
    *,
    coarse_step_s: float = DEFAULT_BARO_ALIGN_COARSE_STEP_S,
    refine_step_s: float = DEFAULT_BARO_ALIGN_REFINE_STEP_S,
    featherweight_smoothing_window_s: float = 0.75,
    marv_smoothing_window_s: float = 1.0,
    edge_guard_s: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Estimate the Featherweight-to-MARV time offset from barometric trends."""

    featherweight_baro = featherweight_source.streams.get("baro")
    marv_baro = marv_source.streams.get("baro")
    if featherweight_baro is None or featherweight_baro.empty:
        raise ValueError("Featherweight source is missing a usable baro stream")
    if marv_baro is None or marv_baro.empty:
        raise ValueError("MARV source is missing a usable baro stream")

    featherweight_features = _build_baro_alignment_features(
        featherweight_baro,
        smoothing_window_s=featherweight_smoothing_window_s,
        edge_guard_s=edge_guard_s,
    )
    marv_features = _build_baro_alignment_features(
        marv_baro,
        smoothing_window_s=marv_smoothing_window_s,
        edge_guard_s=0.0,
    )

    fw_t = featherweight_features["time_s"].to_numpy(dtype=float)
    marv_t = marv_features["time_s"].to_numpy(dtype=float)
    offset_min = float(marv_t[0] - fw_t[0])
    offset_max = float(marv_t[-1] - fw_t[-1])
    if offset_max < offset_min:
        raise ValueError("MARV baro window is shorter than the Featherweight baro interval")

    coarse_offsets = np.arange(
        offset_min,
        offset_max + 0.5 * float(coarse_step_s),
        float(coarse_step_s),
    )
    coarse_candidates = _score_alignment_offsets(
        featherweight_features,
        marv_features,
        coarse_offsets,
    )
    if not coarse_candidates:
        raise ValueError("Could not generate any valid baro alignment candidates")

    best_coarse = coarse_candidates[0]
    if refine_step_s > 0.0 and refine_step_s < coarse_step_s:
        refine_start = max(offset_min, float(best_coarse["offset_s"]) - float(coarse_step_s))
        refine_end = min(offset_max, float(best_coarse["offset_s"]) + float(coarse_step_s))
        refine_offsets = np.arange(
            refine_start,
            refine_end + 0.5 * float(refine_step_s),
            float(refine_step_s),
        )
        refined_candidates = _score_alignment_offsets(
            featherweight_features,
            marv_features,
            refine_offsets,
        )
        candidates = refined_candidates or coarse_candidates
    else:
        candidates = coarse_candidates

    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    score_gap = float(best["score"] - runner_up["score"]) if runner_up is not None else float(best["score"])
    confidence = _alignment_confidence(best["score"], score_gap)
    featherweight_quality = analyze_featherweight_baro_alignment_quality(featherweight_baro)
    strong_match = (
        float(best["derivative_correlation"]) >= 0.80
        and float(best["pressure_correlation"]) >= 0.50
    )

    warnings = list(featherweight_quality["warnings"])
    if confidence < DEFAULT_MERGE_MIN_CONFIDENCE and not strong_match:
        warnings.append(
            "Global baro alignment remains low confidence because the top offset candidates are nearly tied."
        )

    if not featherweight_quality["ready_for_deterministic_alignment"]:
        status = "ambiguous_input"
    elif confidence < DEFAULT_MERGE_MIN_CONFIDENCE and not strong_match:
        status = "low_confidence"
    else:
        status = "ok"

    report = {
        "offset_s": float(best["offset_s"]),
        "score": float(best["score"]),
        "derivative_correlation": float(best["derivative_correlation"]),
        "pressure_correlation": float(best["pressure_correlation"]),
        "score_gap_to_runner_up": score_gap,
        "confidence": confidence,
        "strong_match": strong_match,
        "status": status,
        "can_merge": status == "ok",
        "warnings": warnings,
        "featherweight_interval": {
            "start_time_s": float(fw_t[0]),
            "end_time_s": float(fw_t[-1]),
            "duration_s": float(fw_t[-1] - fw_t[0]),
        },
        "marv_overlap_interval": {
            "start_time_s": float(fw_t[0] + best["offset_s"]),
            "end_time_s": float(fw_t[-1] + best["offset_s"]),
            "duration_s": float(fw_t[-1] - fw_t[0]),
        },
        "top_candidates": [
            {
                "offset_s": float(candidate["offset_s"]),
                "score": float(candidate["score"]),
                "derivative_correlation": float(candidate["derivative_correlation"]),
                "pressure_correlation": float(candidate["pressure_correlation"]),
            }
            for candidate in candidates[:5]
        ],
        "input_quality": {
            "featherweight_baro": featherweight_quality,
        },
        "search": {
            "offset_min_s": offset_min,
            "offset_max_s": offset_max,
            "coarse_step_s": float(coarse_step_s),
            "refine_step_s": float(refine_step_s),
            "edge_guard_s": float(edge_guard_s),
            "featherweight_smoothing_window_s": float(featherweight_smoothing_window_s),
            "marv_smoothing_window_s": float(marv_smoothing_window_s),
        },
    }
    return report, featherweight_features, marv_features


def analyze_featherweight_baro_alignment_quality(
    frame: pd.DataFrame,
    *,
    duplicate_lag_min_s: float = 1.0,
    duplicate_lag_max_s: float = 6.0,
    min_duplicate_run_s: float = 2.0,
) -> dict[str, Any]:
    """Assess whether the Featherweight baro trace is suitable for deterministic alignment."""

    if frame.empty or "time_s" not in frame.columns or "pressure_pa" not in frame.columns:
        return {
            "status": "missing_baro",
            "ready_for_deterministic_alignment": False,
            "leading_outlier_rows": 0,
            "duplicate_run_count": 0,
            "duplicate_sample_fraction": 0.0,
            "top_duplicate_lags": [],
            "top_duplicate_runs": [],
            "warnings": ["Featherweight baro trace is missing or empty."],
        }

    times = frame["time_s"].to_numpy(dtype=float)
    pressure = frame["pressure_pa"].to_numpy(dtype=float)
    leading_outlier_rows = _count_leading_pressure_outliers(times, pressure)

    duplicate_columns = [
        column
        for column in ("pressure_pa", "altitude_agl_m", "temperature_c", "battery_v")
        if column in frame.columns
    ]
    duplicate_runs = _detect_exact_duplicate_runs(
        frame,
        columns=duplicate_columns,
        lag_min_s=duplicate_lag_min_s,
        lag_max_s=duplicate_lag_max_s,
        min_run_s=min_duplicate_run_s,
    )

    duplicate_mask = np.zeros(len(frame), dtype=bool)
    lag_summary: dict[int, dict[str, Any]] = {}
    for run in duplicate_runs:
        start_index = int(run["start_index"])
        end_index = int(run["end_index"])
        lag_samples = int(run["lag_samples"])
        duplicate_mask[start_index : end_index + 1] = True
        duplicate_mask[start_index + lag_samples : end_index + lag_samples + 1] = True

        lag_entry = lag_summary.setdefault(
            lag_samples,
            {
                "lag_samples": lag_samples,
                "lag_s": float(run["lag_s"]),
                "run_count": 0,
                "max_run_samples": 0,
                "duplicate_sample_count": 0,
            },
        )
        lag_entry["run_count"] += 1
        lag_entry["max_run_samples"] = max(
            int(lag_entry["max_run_samples"]),
            int(run["run_samples"]),
        )
        lag_entry["duplicate_sample_count"] += 2 * int(run["run_samples"])

    duplicate_sample_fraction = (
        float(np.count_nonzero(duplicate_mask)) / float(len(frame))
        if len(frame) > 0
        else 0.0
    )
    top_duplicate_lags = sorted(
        lag_summary.values(),
        key=lambda item: (
            -int(item["duplicate_sample_count"]),
            -int(item["max_run_samples"]),
            float(item["lag_s"]),
        ),
    )[:5]
    top_duplicate_runs = [
        {
            "lag_samples": int(run["lag_samples"]),
            "lag_s": float(run["lag_s"]),
            "run_samples": int(run["run_samples"]),
            "run_duration_s": float(run["run_duration_s"]),
            "start_time_s": float(run["start_time_s"]),
            "repeat_time_s": float(run["repeat_time_s"]),
        }
        for run in duplicate_runs[:5]
    ]

    warnings: list[str] = []
    if leading_outlier_rows > 0:
        warnings.append(
            f"Featherweight baro trace starts with {leading_outlier_rows} pressure outlier rows before it settles into the main band."
        )
    if duplicate_sample_fraction >= 0.25:
        dominant_lag = top_duplicate_lags[0] if top_duplicate_lags else None
        if dominant_lag is None:
            warnings.append(
                "Featherweight baro trace contains repeated exact sample runs across a large fraction of the dataset."
            )
        else:
            warnings.append(
                "Featherweight baro trace contains repeated exact sample runs across "
                f"{duplicate_sample_fraction:.1%} of samples, dominated by a lag of "
                f"{dominant_lag['lag_s']:.3f} s."
            )

    if duplicate_sample_fraction >= 0.25:
        status = "ambiguous_repeated_segments"
    elif leading_outlier_rows > 0:
        status = "leading_outliers"
    else:
        status = "ok"

    ready = leading_outlier_rows == 0 and duplicate_sample_fraction < 0.10
    return {
        "status": status,
        "ready_for_deterministic_alignment": ready,
        "leading_outlier_rows": int(leading_outlier_rows),
        "duplicate_run_count": int(len(duplicate_runs)),
        "duplicate_sample_fraction": duplicate_sample_fraction,
        "top_duplicate_lags": top_duplicate_lags,
        "top_duplicate_runs": top_duplicate_runs,
        "warnings": warnings,
    }


def write_baro_alignment_artifacts(
    *,
    featherweight_source: NormalizedSource,
    marv_source: NormalizedSource,
    isolated_marv_source: NormalizedSource,
    window_report: dict[str, Any],
    alignment_report: dict[str, Any],
    featherweight_baro_debug: pd.DataFrame,
    marv_baro_debug: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write step-3/4 artifacts to disk for inspection."""

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    for stream_name, frame in isolated_marv_source.streams.items():
        frame.to_csv(output_path / f"marv_window_{stream_name}.csv", index=False)

    featherweight_baro_debug.to_csv(
        output_path / "featherweight_baro_alignment_debug.csv",
        index=False,
    )
    marv_baro_debug.to_csv(
        output_path / "marv_baro_alignment_debug.csv",
        index=False,
    )

    report = {
        "featherweight": featherweight_source.summary(),
        "marv_full": marv_source.summary(),
        "marv_window": isolated_marv_source.summary(),
        "flight_window": window_report,
        "alignment": alignment_report,
    }
    report_path = output_path / "baro_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def align_gps_altitude_to_marv_baro(
    gps_source: NormalizedSource,
    marv_source: NormalizedSource,
    *,
    coarse_step_s: float = DEFAULT_GPS_ALIGN_COARSE_STEP_S,
    refine_step_s: float = DEFAULT_GPS_ALIGN_REFINE_STEP_S,
    gps_smoothing_window_s: float = 0.5,
    marv_smoothing_window_s: float = 0.5,
    prelaunch_margin_s: float = 1.0,
    postlanding_margin_s: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Estimate GPS-to-MARV offset from GPS altitude versus MARV baro-derived altitude."""

    if gps_source.source_name != "featherweight_gps":
        raise ValueError("GPS altitude alignment expects a normalized Featherweight GPS source")

    gps_navigation = gps_source.streams.get("navigation")
    gps_events = gps_source.streams.get("events")
    marv_baro = marv_source.streams.get("baro")
    if gps_navigation is None or gps_navigation.empty:
        raise ValueError("GPS source is missing a usable navigation stream")
    if marv_baro is None or marv_baro.empty:
        raise ValueError("MARV source is missing a usable baro stream")

    gps_window = _select_gps_flight_window(
        gps_navigation,
        gps_events,
        prelaunch_margin_s=prelaunch_margin_s,
        postlanding_margin_s=postlanding_margin_s,
    )
    marv_altitude = _build_marv_baro_altitude_frame(marv_baro)

    gps_features = _build_altitude_alignment_features(
        gps_window,
        altitude_column="gps_altitude_agl_m",
        smoothing_window_s=gps_smoothing_window_s,
        edge_guard_s=0.5,
    )
    marv_features = _build_altitude_alignment_features(
        marv_altitude,
        altitude_column="baro_altitude_rel_m",
        smoothing_window_s=marv_smoothing_window_s,
        edge_guard_s=0.0,
    )

    gps_t = gps_features["time_s"].to_numpy(dtype=float)
    marv_t = marv_features["time_s"].to_numpy(dtype=float)
    offset_min = float(marv_t[0] - gps_t[0])
    offset_max = float(marv_t[-1] - gps_t[-1])
    if offset_max < offset_min:
        raise ValueError("MARV baro interval is shorter than the selected GPS flight window")

    coarse_offsets = np.arange(
        offset_min,
        offset_max + 0.5 * float(coarse_step_s),
        float(coarse_step_s),
    )
    coarse_candidates = _score_altitude_alignment_offsets(
        gps_features,
        marv_features,
        coarse_offsets,
    )
    if not coarse_candidates:
        raise ValueError("Could not generate any valid GPS/MARV alignment candidates")

    best_coarse = coarse_candidates[0]
    if refine_step_s > 0.0 and refine_step_s < coarse_step_s:
        refine_start = max(offset_min, float(best_coarse["offset_s"]) - float(coarse_step_s))
        refine_end = min(offset_max, float(best_coarse["offset_s"]) + float(coarse_step_s))
        refine_offsets = np.arange(
            refine_start,
            refine_end + 0.5 * float(refine_step_s),
            float(refine_step_s),
        )
        refined_candidates = _score_altitude_alignment_offsets(
            gps_features,
            marv_features,
            refine_offsets,
        )
        candidates = refined_candidates or coarse_candidates
    else:
        candidates = coarse_candidates

    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    score_gap = float(best["score"] - runner_up["score"]) if runner_up is not None else float(best["score"])
    confidence = _alignment_confidence(best["score"], score_gap)
    strong_match = (
        float(best["altitude_correlation"]) >= 0.98
        and float(best["derivative_correlation"]) >= 0.95
        and float(best["rmse_m"]) <= 75.0
    )
    status = "ok" if strong_match or confidence >= DEFAULT_MERGE_MIN_CONFIDENCE else "low_confidence"
    warnings: list[str] = []
    if not strong_match and confidence < DEFAULT_MERGE_MIN_CONFIDENCE:
        warnings.append(
            "GPS/MARV altitude alignment has a strong visual match but the score surface is broad, so treat the offset as best-effort."
        )

    report = {
        "offset_s": float(best["offset_s"]),
        "score": float(best["score"]),
        "confidence": confidence,
        "strong_match": strong_match,
        "status": status,
        "can_merge": status == "ok",
        "warnings": warnings,
        "altitude_correlation": float(best["altitude_correlation"]),
        "derivative_correlation": float(best["derivative_correlation"]),
        "rmse_m": float(best["rmse_m"]),
        "scale": float(best["scale"]),
        "bias_m": float(best["bias_m"]),
        "gps_window": {
            "start_time_s": float(gps_t[0]),
            "end_time_s": float(gps_t[-1]),
            "duration_s": float(gps_t[-1] - gps_t[0]),
        },
        "marv_overlap_interval": {
            "start_time_s": float(gps_t[0] + best["offset_s"]),
            "end_time_s": float(gps_t[-1] + best["offset_s"]),
            "duration_s": float(gps_t[-1] - gps_t[0]),
        },
        "top_candidates": [
            {
                "offset_s": float(candidate["offset_s"]),
                "score": float(candidate["score"]),
                "altitude_correlation": float(candidate["altitude_correlation"]),
                "derivative_correlation": float(candidate["derivative_correlation"]),
                "rmse_m": float(candidate["rmse_m"]),
                "scale": float(candidate["scale"]),
                "bias_m": float(candidate["bias_m"]),
            }
            for candidate in candidates[:5]
        ],
        "search": {
            "offset_min_s": offset_min,
            "offset_max_s": offset_max,
            "coarse_step_s": float(coarse_step_s),
            "refine_step_s": float(refine_step_s),
            "gps_smoothing_window_s": float(gps_smoothing_window_s),
            "marv_smoothing_window_s": float(marv_smoothing_window_s),
            "prelaunch_margin_s": float(prelaunch_margin_s),
            "postlanding_margin_s": float(postlanding_margin_s),
        },
    }
    return report, gps_features, marv_features


def write_gps_baro_alignment_artifacts(
    *,
    gps_source: NormalizedSource,
    marv_source: NormalizedSource,
    alignment_report: dict[str, Any],
    gps_debug: pd.DataFrame,
    marv_debug: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    gps_debug.to_csv(output_path / "gps_altitude_alignment_debug.csv", index=False)
    marv_debug.to_csv(output_path / "marv_baro_altitude_alignment_debug.csv", index=False)

    report = {
        "gps": gps_source.summary(),
        "marv": marv_source.summary(),
        "alignment": alignment_report,
    }
    report_path = output_path / "gps_baro_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def merge_aligned_sources(
    featherweight_source: NormalizedSource,
    marv_source: NormalizedSource,
    *,
    offset_s: float,
    time_step_s: float = DEFAULT_MERGED_TIME_STEP_S,
    timebase_source: str = "shared_uniform",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resample Featherweight onto a shared or MARV-native aligned time base."""

    if not np.isfinite(offset_s):
        raise ValueError("Alignment offset must be finite")
    if time_step_s <= 0.0:
        raise ValueError("Merged time step must be positive")

    shifted_featherweight = _shift_source_time(featherweight_source, offset_s=offset_s)
    overlap_start_s, overlap_end_s = _resolve_common_overlap_bounds(
        [shifted_featherweight, marv_source]
    )
    if overlap_end_s <= overlap_start_s:
        raise ValueError("Aligned Featherweight and MARV streams do not overlap in time")

    time_s, timebase_summary = _build_merge_timebase(
        marv_source,
        overlap_start_s=overlap_start_s,
        overlap_end_s=overlap_end_s,
        time_step_s=float(time_step_s),
        timebase_source=timebase_source,
    )
    merged = pd.DataFrame(
        {
            "time_s": time_s,
            "featherweight_source_time_s": time_s - float(offset_s),
            "marv_source_time_s": time_s,
        }
    )

    stream_specs = [
        ("featherweight", "baro", shifted_featherweight.streams.get("baro")),
        ("featherweight", "navigation", shifted_featherweight.streams.get("navigation")),
        ("featherweight", "events", shifted_featherweight.streams.get("events")),
        ("marv", "baro", marv_source.streams.get("baro")),
        ("marv", "primary_imu", marv_source.streams.get("primary_imu")),
        ("marv", "aux_imu", marv_source.streams.get("aux_imu")),
    ]
    for source_name, stream_name, frame in stream_specs:
        if frame is None or frame.empty:
            continue
        prefix = f"{source_name}_{stream_name}_"
        merged = pd.concat(
            [merged, _resample_stream_to_timebase(frame, time_s=time_s, prefix=prefix)],
            axis=1,
        )

    summary = {
        "offset_s": float(offset_s),
        "time_step_s": float(time_step_s),
        "rows": int(len(merged)),
        "time_start_s": float(time_s[0]),
        "time_end_s": float(time_s[-1]),
        "columns": merged.columns.tolist(),
        "timebase": timebase_summary,
    }
    return merged, summary


def _build_merge_timebase(
    marv_source: NormalizedSource,
    *,
    overlap_start_s: float,
    overlap_end_s: float,
    time_step_s: float,
    timebase_source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if timebase_source == "shared_uniform":
        time_s = np.arange(
            overlap_start_s,
            overlap_end_s + 0.5 * float(time_step_s),
            float(time_step_s),
        )
        return time_s, {
            "source": "shared_uniform",
            "stream": None,
            "rows": int(len(time_s)),
            "approx_rate_hz": _estimate_rate_hz(time_s),
        }

    if timebase_source not in MERGE_TIMEBASE_CHOICES:
        raise ValueError(
            f"Unsupported merge timebase source {timebase_source!r}. "
            f"Expected one of {MERGE_TIMEBASE_CHOICES}."
        )

    marv_stream_name = timebase_source.removeprefix("marv_")
    frame = marv_source.streams.get(marv_stream_name)
    if frame is None or frame.empty or "time_s" not in frame.columns:
        raise ValueError(
            f"MARV timebase stream {marv_stream_name!r} is missing or has no time_s column"
        )

    source_time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    finite_mask = (
        np.isfinite(source_time_s)
        & (source_time_s >= float(overlap_start_s))
        & (source_time_s <= float(overlap_end_s))
    )
    time_s = source_time_s[finite_mask]
    if time_s.size == 0:
        raise ValueError(
            "No MARV timestamps remain inside the aligned overlap window for "
            f"timebase source {timebase_source!r}"
        )

    return time_s, {
        "source": timebase_source,
        "stream": marv_stream_name,
        "rows": int(len(time_s)),
        "approx_rate_hz": _estimate_rate_hz(time_s),
    }


def _build_marv_imu_stream(
    raw: pd.DataFrame,
    *,
    origin_us: float,
    state_column: str,
    sample_time_column: str,
    lagged_column: str,
    rename_map: dict[str, str],
) -> pd.DataFrame:
    measurement_columns = list(rename_map)
    stream = raw.loc[:, ["log_us", state_column, sample_time_column, lagged_column, *measurement_columns]].copy()
    stream = stream.dropna(subset=[sample_time_column, *measurement_columns])
    stream = stream.sort_values(["log_us", sample_time_column]).drop_duplicates(
        subset=[sample_time_column],
        keep="first",
    )

    sample_time_us = stream[sample_time_column].to_numpy(dtype=float)
    _validate_monotonic_time(sample_time_us, Path(state_column), sample_time_column)

    renamed = stream.rename(columns=rename_map).reset_index(drop=True)
    lagged = pd.to_numeric(renamed[lagged_column], errors="coerce").fillna(0).astype(int)
    normalized = pd.DataFrame(
        {
            "time_s": (renamed[sample_time_column] - origin_us) / 1_000_000.0,
            "log_time_s": (renamed["log_us"] - origin_us) / 1_000_000.0,
            "sample_time_us": renamed[sample_time_column].astype(np.int64),
            "log_time_us": renamed["log_us"].astype(np.int64),
            "state": renamed[state_column].astype(str),
            "lagged": lagged.astype(bool),
        }
    )
    for column in rename_map.values():
        normalized[column] = renamed[column]
    return normalized


def _build_marv_baro_stream(raw: pd.DataFrame, *, origin_us: float) -> pd.DataFrame:
    stream = raw.loc[
        :,
        ["log_us", "baro_state", "baro_sample_us", "baro_lagged", "baro_pressure_pa", "baro_temp_c"],
    ].copy()
    stream = stream.dropna(subset=["baro_sample_us", "baro_pressure_pa"])
    stream = stream.sort_values(["log_us", "baro_sample_us"]).drop_duplicates(
        subset=["baro_sample_us"],
        keep="first",
    )

    sample_time_us = stream["baro_sample_us"].to_numpy(dtype=float)
    _validate_monotonic_time(sample_time_us, Path("baro_state"), "baro_sample_us")

    lagged = pd.to_numeric(stream["baro_lagged"], errors="coerce").fillna(0).astype(int)
    return pd.DataFrame(
        {
            "time_s": (stream["baro_sample_us"] - origin_us) / 1_000_000.0,
            "log_time_s": (stream["log_us"] - origin_us) / 1_000_000.0,
            "sample_time_us": stream["baro_sample_us"].astype(np.int64),
            "log_time_us": stream["log_us"].astype(np.int64),
            "state": stream["baro_state"].astype(str),
            "lagged": lagged.astype(bool),
            "pressure_pa": stream["baro_pressure_pa"],
            "temperature_c": stream["baro_temp_c"],
        }
    ).reset_index(drop=True)


def _build_baro_alignment_features(
    frame: pd.DataFrame,
    *,
    smoothing_window_s: float,
    edge_guard_s: float,
) -> pd.DataFrame:
    times = frame["time_s"].to_numpy(dtype=float)
    pressure = frame["pressure_pa"].to_numpy(dtype=float)
    smooth_pressure = _smooth_time_series(times, pressure, smoothing_window_s)
    pressure_z = _robust_zscore(smooth_pressure)
    dpdt = np.gradient(smooth_pressure, times)
    dpdt_z = _robust_zscore(dpdt)
    weights = 0.25 + np.clip(np.abs(dpdt_z), 0.0, 4.0)
    weights = _suppress_edges(weights, times, guard_s=edge_guard_s)
    if np.all(weights <= 0.0):
        weights = np.ones_like(weights, dtype=float)

    features = frame.copy()
    features["smooth_pressure_pa"] = smooth_pressure
    features["pressure_z"] = pressure_z
    features["dpdt_pa_s"] = dpdt
    features["dpdt_z"] = dpdt_z
    features["alignment_weight"] = weights
    return features


def _build_marv_baro_altitude_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("MARV baro stream is empty")
    p0 = float(frame["pressure_pa"].iloc[0])
    altitude_rel_m = 44_330.0 * (1.0 - (frame["pressure_pa"] / p0) ** 0.190294957)
    derived = frame.copy()
    derived["baro_altitude_rel_m"] = altitude_rel_m
    return derived


def _build_altitude_alignment_features(
    frame: pd.DataFrame,
    *,
    altitude_column: str,
    smoothing_window_s: float,
    edge_guard_s: float,
) -> pd.DataFrame:
    times = frame["time_s"].to_numpy(dtype=float)
    altitude_m = frame[altitude_column].to_numpy(dtype=float)
    smooth_altitude = _smooth_time_series(times, altitude_m, smoothing_window_s)
    dalt_dt = np.gradient(smooth_altitude, times)
    derivative_z = _robust_zscore(dalt_dt)
    weights = 0.25 + np.clip(np.abs(derivative_z), 0.0, 4.0)
    weights = _suppress_edges(weights, times, guard_s=edge_guard_s)
    if np.all(weights <= 0.0):
        weights = np.ones_like(weights, dtype=float)

    features = frame.copy()
    features["smooth_altitude_m"] = smooth_altitude
    features["dalt_dt_mps"] = dalt_dt
    features["dalt_dt_z"] = derivative_z
    features["alignment_weight"] = weights
    return features


def _detect_exact_duplicate_runs(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    lag_min_s: float,
    lag_max_s: float,
    min_run_s: float,
) -> list[dict[str, float]]:
    if frame.empty or "time_s" not in frame.columns or not columns:
        return []

    times = frame["time_s"].to_numpy(dtype=float)
    lag_min_samples = _duration_to_samples(times, lag_min_s)
    lag_max_samples = _duration_to_samples(times, lag_max_s)
    if lag_max_samples < lag_min_samples:
        return []

    min_run_samples = max(3, _duration_to_samples(times, min_run_s))
    rounded_values = [np.round(frame[column].to_numpy(dtype=float), decimals=6) for column in columns]
    row_keys = list(zip(*rounded_values))

    runs: list[dict[str, float]] = []
    for lag_samples in range(lag_min_samples, lag_max_samples + 1):
        index = 0
        while index + lag_samples < len(row_keys):
            run_samples = 0
            while (
                index + lag_samples + run_samples < len(row_keys)
                and row_keys[index + run_samples] == row_keys[index + lag_samples + run_samples]
            ):
                run_samples += 1

            if run_samples >= min_run_samples:
                end_index = index + run_samples - 1
                runs.append(
                    {
                        "lag_samples": float(lag_samples),
                        "lag_s": float(times[index + lag_samples] - times[index]),
                        "start_index": float(index),
                        "end_index": float(end_index),
                        "run_samples": float(run_samples),
                        "run_duration_s": float(times[end_index] - times[index]),
                        "start_time_s": float(times[index]),
                        "repeat_time_s": float(times[index + lag_samples]),
                    }
                )
                index += run_samples
            else:
                index += max(run_samples, 1)

    runs.sort(
        key=lambda run: (
            -float(run["run_samples"]),
            float(run["lag_s"]),
            float(run["start_time_s"]),
        )
    )
    return runs


def _score_alignment_offsets(
    featherweight_features: pd.DataFrame,
    marv_features: pd.DataFrame,
    offsets_s: np.ndarray,
) -> list[dict[str, float]]:
    fw_t = featherweight_features["time_s"].to_numpy(dtype=float)
    fw_pressure = featherweight_features["pressure_z"].to_numpy(dtype=float)
    fw_derivative = featherweight_features["dpdt_z"].to_numpy(dtype=float)
    weights = featherweight_features["alignment_weight"].to_numpy(dtype=float)
    marv_t = marv_features["time_s"].to_numpy(dtype=float)
    marv_pressure = marv_features["pressure_z"].to_numpy(dtype=float)
    marv_derivative = marv_features["dpdt_z"].to_numpy(dtype=float)

    candidates: list[dict[str, float]] = []
    for offset_s in offsets_s:
        sample_times = fw_t + float(offset_s)
        if sample_times[0] < marv_t[0] or sample_times[-1] > marv_t[-1]:
            continue

        marv_pressure_interp = np.interp(sample_times, marv_t, marv_pressure)
        marv_derivative_interp = np.interp(sample_times, marv_t, marv_derivative)

        derivative_corr = _weighted_correlation(
            fw_derivative,
            marv_derivative_interp,
            weights,
        )
        pressure_corr = _weighted_correlation(
            fw_pressure,
            marv_pressure_interp,
            weights,
        )
        score = 0.75 * derivative_corr + 0.25 * pressure_corr
        candidates.append(
            {
                "offset_s": float(offset_s),
                "score": float(score),
                "derivative_correlation": float(derivative_corr),
                "pressure_correlation": float(pressure_corr),
            }
        )

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return candidates


def _score_altitude_alignment_offsets(
    gps_features: pd.DataFrame,
    marv_features: pd.DataFrame,
    offsets_s: np.ndarray,
) -> list[dict[str, float]]:
    gps_t = gps_features["time_s"].to_numpy(dtype=float)
    gps_altitude = gps_features["smooth_altitude_m"].to_numpy(dtype=float)
    gps_derivative = gps_features["dalt_dt_mps"].to_numpy(dtype=float)
    weights = gps_features["alignment_weight"].to_numpy(dtype=float)
    marv_t = marv_features["time_s"].to_numpy(dtype=float)
    marv_altitude = marv_features["smooth_altitude_m"].to_numpy(dtype=float)
    marv_derivative = marv_features["dalt_dt_mps"].to_numpy(dtype=float)
    altitude_span = max(1.0, float(np.nanmax(marv_altitude) - np.nanmin(marv_altitude)))

    candidates: list[dict[str, float]] = []
    for offset_s in offsets_s:
        sample_times = gps_t + float(offset_s)
        if sample_times[0] < marv_t[0] or sample_times[-1] > marv_t[-1]:
            continue

        marv_altitude_interp = np.interp(sample_times, marv_t, marv_altitude)
        marv_derivative_interp = np.interp(sample_times, marv_t, marv_derivative)

        scale, bias = _weighted_affine_fit(
            source_values=gps_altitude,
            target_values=marv_altitude_interp,
            weights=weights,
        )
        fitted_altitude = scale * gps_altitude + bias
        fitted_derivative = scale * gps_derivative
        altitude_corr = _weighted_correlation(
            fitted_altitude,
            marv_altitude_interp,
            weights,
        )
        derivative_corr = _weighted_correlation(
            fitted_derivative,
            marv_derivative_interp,
            weights,
        )
        rmse_m = _weighted_rmse(
            fitted_altitude,
            marv_altitude_interp,
            weights,
        )
        score = (0.55 * derivative_corr + 0.45 * altitude_corr) - (rmse_m / altitude_span)
        candidates.append(
            {
                "offset_s": float(offset_s),
                "score": float(score),
                "altitude_correlation": float(altitude_corr),
                "derivative_correlation": float(derivative_corr),
                "rmse_m": float(rmse_m),
                "scale": float(scale),
                "bias_m": float(bias),
            }
        )

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return candidates


def _weighted_correlation(
    values_a: np.ndarray,
    values_b: np.ndarray,
    weights: np.ndarray,
) -> float:
    valid = (
        np.isfinite(values_a)
        & np.isfinite(values_b)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if int(np.count_nonzero(valid)) < 3:
        return 0.0

    a = values_a[valid]
    b = values_b[valid]
    w = weights[valid]
    w = w / np.sum(w)

    mean_a = float(np.sum(w * a))
    mean_b = float(np.sum(w * b))
    centered_a = a - mean_a
    centered_b = b - mean_b
    variance_a = float(np.sum(w * centered_a * centered_a))
    variance_b = float(np.sum(w * centered_b * centered_b))
    if variance_a <= 1e-12 or variance_b <= 1e-12:
        return 0.0

    covariance = float(np.sum(w * centered_a * centered_b))
    return float(covariance / np.sqrt(variance_a * variance_b))


def _weighted_affine_fit(
    *,
    source_values: np.ndarray,
    target_values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    valid = (
        np.isfinite(source_values)
        & np.isfinite(target_values)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if int(np.count_nonzero(valid)) < 3:
        return 1.0, 0.0

    x = source_values[valid]
    y = target_values[valid]
    w = weights[valid]
    w = w / np.sum(w)

    design = np.column_stack([x, np.ones_like(x)])
    design_w = design * np.sqrt(w)[:, None]
    target_w = y * np.sqrt(w)
    coefficients, *_ = np.linalg.lstsq(design_w, target_w, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _weighted_rmse(
    values_a: np.ndarray,
    values_b: np.ndarray,
    weights: np.ndarray,
) -> float:
    valid = (
        np.isfinite(values_a)
        & np.isfinite(values_b)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if int(np.count_nonzero(valid)) < 3:
        return float("inf")
    residual = values_a[valid] - values_b[valid]
    w = weights[valid]
    w = w / np.sum(w)
    return float(np.sqrt(np.sum(w * residual * residual)))


def _alignment_confidence(score: float, score_gap: float) -> float:
    score_term = np.clip((float(score) + 1.0) * 0.5, 0.0, 1.0)
    gap_term = np.clip(float(score_gap) / 0.15, 0.0, 1.0)
    return float(score_term * gap_term)


def _smooth_time_series(times: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    width = _duration_to_samples(times, window_s)
    width = max(3, width)
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=float) / float(width)
    pad = width // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return (values - median) / scale


def _suppress_edges(values: np.ndarray, times: np.ndarray, *, guard_s: float) -> np.ndarray:
    if guard_s <= 0.0 or values.size == 0:
        return values.copy()

    result = values.copy()
    start = float(times[0])
    end = float(times[-1])
    edge_mask = (times <= (start + guard_s)) | (times >= (end - guard_s))
    result[edge_mask] = 0.0
    return result


def _duration_to_samples(times: np.ndarray, duration_s: float) -> int:
    if len(times) < 2 or duration_s <= 0.0:
        return 1
    dt = np.diff(times)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return 1
    return max(1, int(round(float(duration_s) / float(np.median(dt)))))


def _fill_short_false_gaps(mask: np.ndarray, *, max_gap_samples: int) -> np.ndarray:
    if max_gap_samples <= 0 or mask.size == 0:
        return mask.copy()

    filled = mask.copy()
    index = 0
    size = filled.size
    while index < size:
        if filled[index]:
            index += 1
            continue

        gap_start = index
        while index < size and not filled[index]:
            index += 1
        gap_end = index
        if gap_start == 0 or gap_end == size:
            continue
        if (gap_end - gap_start) <= max_gap_samples:
            filled[gap_start:gap_end] = True
    return filled


def _extract_mask_segments(
    mask: np.ndarray,
    times: np.ndarray,
    activity: np.ndarray,
) -> list[dict[str, float]]:
    segments: list[dict[str, float]] = []
    if mask.size == 0:
        return segments

    index = 0
    size = mask.size
    while index < size:
        if not mask[index]:
            index += 1
            continue

        start_index = index
        while index < size and mask[index]:
            index += 1
        end_index = index - 1
        segment_activity = activity[start_index : end_index + 1]
        segments.append(
            {
                "start_index": float(start_index),
                "end_index": float(end_index),
                "start_time_s": float(times[start_index]),
                "end_time_s": float(times[end_index]),
                "duration_s": float(times[end_index] - times[start_index]),
                "activity_integral": float(np.sum(segment_activity)),
                "peak_activity": float(np.max(segment_activity)),
            }
        )
    return segments


def _count_leading_pressure_outliers(
    times: np.ndarray,
    pressure: np.ndarray,
    *,
    baseline_window_s: float = 1.0,
    z_threshold: float = 8.0,
) -> int:
    if len(pressure) < 5:
        return 0

    baseline_start = min(len(pressure) - 1, max(5, _duration_to_samples(times, baseline_window_s)))
    baseline = pressure[baseline_start:]
    center = float(np.nanmedian(baseline))
    scale = float(np.nanmedian(np.abs(baseline - center))) * 1.4826
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.nanstd(baseline))
    if not np.isfinite(scale) or scale <= 1e-9:
        return 0

    count = 0
    for value in pressure[:baseline_start]:
        if abs(float(value) - center) > z_threshold * scale:
            count += 1
        else:
            break
    return count


def _select_gps_flight_window(
    navigation: pd.DataFrame,
    events: pd.DataFrame | None,
    *,
    prelaunch_margin_s: float,
    postlanding_margin_s: float,
) -> pd.DataFrame:
    if events is None or events.empty:
        return navigation.copy().reset_index(drop=True)

    launch_times = events.loc[events["launch_detected"], "time_s"]
    landing_times = events.loc[events["landing_detected"], "time_s"]
    if launch_times.empty:
        start_time_s = float(navigation["time_s"].iloc[0])
    else:
        start_time_s = max(
            float(navigation["time_s"].iloc[0]),
            float(launch_times.iloc[0]) - float(prelaunch_margin_s),
        )

    if landing_times.empty:
        end_time_s = float(navigation["time_s"].iloc[-1])
    else:
        end_time_s = min(
            float(navigation["time_s"].iloc[-1]),
            float(landing_times.iloc[0]) + float(postlanding_margin_s),
        )

    window = navigation.loc[
        (navigation["time_s"] >= start_time_s)
        & (navigation["time_s"] <= end_time_s)
    ].copy()
    return window.reset_index(drop=True)


def _slice_source_by_time(
    source: NormalizedSource,
    *,
    start_time_s: float,
    end_time_s: float,
) -> NormalizedSource:
    streams: dict[str, pd.DataFrame] = {}
    for stream_name, frame in source.streams.items():
        if "time_s" not in frame.columns:
            streams[stream_name] = frame.copy()
            continue
        trimmed = frame.loc[
            (frame["time_s"] >= float(start_time_s))
            & (frame["time_s"] <= float(end_time_s))
        ].copy()
        streams[stream_name] = trimmed.reset_index(drop=True)
    return NormalizedSource(
        source_name=source.source_name,
        source_path=source.source_path,
        metadata=dict(source.metadata),
        streams=streams,
    )


def _shift_source_time(source: NormalizedSource, *, offset_s: float) -> NormalizedSource:
    shifted_streams: dict[str, pd.DataFrame] = {}
    for stream_name, frame in source.streams.items():
        shifted = frame.copy()
        if "time_s" in shifted.columns:
            shifted["time_s"] = shifted["time_s"] + float(offset_s)
        if "log_time_s" in shifted.columns and source.source_name == "featherweight":
            shifted["log_time_s"] = shifted["log_time_s"] + float(offset_s)
        shifted_streams[stream_name] = shifted
    shifted_metadata = dict(source.metadata)
    shifted_metadata["time_offset_s"] = float(offset_s)
    return NormalizedSource(
        source_name=source.source_name,
        source_path=source.source_path,
        metadata=shifted_metadata,
        streams=shifted_streams,
    )


def _resolve_common_overlap_bounds(sources: list[NormalizedSource]) -> tuple[float, float]:
    starts: list[float] = []
    ends: list[float] = []
    for source in sources:
        for frame in source.streams.values():
            if "time_s" not in frame.columns or frame.empty:
                continue
            starts.append(float(frame["time_s"].iloc[0]))
            ends.append(float(frame["time_s"].iloc[-1]))
    if not starts or not ends:
        raise ValueError("Could not find time-based streams to merge")
    return max(starts), min(ends)


def _resample_stream_to_timebase(
    frame: pd.DataFrame,
    *,
    time_s: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return pd.DataFrame(index=np.arange(len(time_s)))

    source_time_s = frame["time_s"].to_numpy(dtype=float)
    result = pd.DataFrame(index=np.arange(len(time_s)))
    for column in frame.columns:
        if column == "time_s":
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or _is_boolean_like_numeric(series):
            sampled = _nearest_sample(
                target_time_s=time_s,
                source_time_s=source_time_s,
                source_values=pd.to_numeric(series, errors="coerce").to_numpy(dtype=float),
            )
            result[prefix + column] = pd.Series(sampled >= 0.5, dtype="boolean")
            continue
        if not pd.api.types.is_numeric_dtype(series):
            continue

        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        sampled = np.interp(time_s, source_time_s, values)
        result[prefix + column] = sampled
    return result


def _nearest_sample(
    *,
    target_time_s: np.ndarray,
    source_time_s: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    insertion_points = np.searchsorted(source_time_s, target_time_s, side="left")
    right_indices = np.clip(insertion_points, 0, len(source_time_s) - 1)
    left_indices = np.clip(insertion_points - 1, 0, len(source_time_s) - 1)
    choose_right = np.abs(source_time_s[right_indices] - target_time_s) < np.abs(
        source_time_s[left_indices] - target_time_s
    )
    indices = np.where(choose_right, right_indices, left_indices)
    return source_values[indices]


def _is_boolean_like_numeric(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return False
    unique_values = np.unique(values)
    return bool(np.all(np.isin(unique_values, [0.0, 1.0])))


def _resolve_marv_origin_us(raw: pd.DataFrame) -> float:
    candidates = []
    for column in ("log_us", "imu_sample_us", "aux_imu_sample_us", "baro_sample_us"):
        values = pd.to_numeric(raw[column], errors="coerce").dropna()
        if not values.empty:
            candidates.append(float(values.min()))
    if not candidates:
        raise ValueError("MARV file did not contain any usable timestamps")
    return min(candidates)


def _fahrenheit_to_c(values: pd.Series) -> pd.Series:
    return (values - 32.0) * (5.0 / 9.0)


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    time_start_s = None
    time_end_s = None
    approx_rate_hz = None
    if "time_s" in frame.columns and not frame.empty:
        times = frame["time_s"].to_numpy(dtype=float)
        time_start_s = float(times[0])
        time_end_s = float(times[-1])
        approx_rate_hz = _estimate_rate_hz(times)
    return {
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "time_start_s": time_start_s,
        "time_end_s": time_end_s,
        "approx_rate_hz": approx_rate_hz,
    }


def _estimate_rate_hz(times: np.ndarray) -> float | None:
    if len(times) < 2:
        return None
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return None
    return float(1.0 / np.median(diffs))


def _coerce_numeric_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _require_columns(frame: pd.DataFrame, required_columns: tuple[str, ...], path: Path) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def _validate_monotonic_time(time_values: np.ndarray, path: Path, label: str) -> None:
    if np.isnan(time_values).any():
        raise ValueError(f"Non-numeric timestamps in {path} column {label}")
    if (time_values < 0.0).any():
        raise ValueError(f"Negative timestamps in {path} column {label}")
    if len(time_values) > 1 and np.any(np.diff(time_values) <= 0.0):
        raise ValueError(f"Timestamps must be strictly increasing in {path} column {label}")


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    true_values = {"1", "TRUE", "T", "YES", "Y"}
    return normalized.isin(true_values)


def _resolve_input_path(path: str | Path) -> Path:
    raw_path = Path(path).expanduser()
    candidate = raw_path.resolve()
    if candidate.exists():
        return candidate

    if not raw_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = (repo_root / raw_path).resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Input file not found: {raw_path}")


def export_aligned_replay_session(
    gps_source: NormalizedSource,
    marv_source: NormalizedSource,
    *,
    offset_s: float,
    output_dir: str | Path,
    vehicle_name: str = DEFAULT_REPLAY_VEHICLE_NAME,
    session_id: str | None = None,
    time_step_s: float = DEFAULT_MERGED_TIME_STEP_S,
    derive_parameters_from_telemetry: bool = True,
) -> dict[str, Any]:
    from sim.estimation.adapters import (
        RocketPyReplayConfig,
        run_rocketpy_replay,
    )
    from sim.sitl.session import build_session_manifest

    if gps_source.source_name != "featherweight_gps":
        raise ValueError("Replay export currently expects a Featherweight GPS normalized source")
    if marv_source.source_name != "marv":
        raise ValueError("Replay export currently expects a MARV normalized source")

    shifted_gps = _shift_source_time(gps_source, offset_s=float(offset_s))
    overlap_start_s, overlap_end_s = _resolve_common_overlap_bounds([shifted_gps, marv_source])

    gps_overlap = _slice_source_by_time(
        shifted_gps,
        start_time_s=overlap_start_s,
        end_time_s=overlap_end_s,
    )
    marv_overlap = _slice_source_by_time(
        marv_source,
        start_time_s=overlap_start_s,
        end_time_s=overlap_end_s,
    )

    zero_shift_s = -float(overlap_start_s)
    gps_session = _shift_source_time(gps_overlap, offset_s=zero_shift_s)
    marv_session = _shift_source_time(marv_overlap, offset_s=zero_shift_s)

    navigation = gps_session.streams["navigation"]
    reference_latitude_deg, reference_longitude_deg = _resolve_replay_reference_lat_lon(navigation)
    reference_altitude_m = _resolve_replay_reference_altitude_m(navigation)
    sea_level_pressure_pa = _resolve_replay_sea_level_pressure_pa(
        marv_session.streams["baro"],
        reference_altitude_m=reference_altitude_m,
    )

    imu_frame = _build_replay_imu_frame(marv_session.streams["primary_imu"])
    baro_frame = _build_replay_baro_frame(marv_session.streams["baro"])
    gps_frame = _build_replay_gps_frame(
        navigation,
        reference_altitude_m=reference_altitude_m,
    )

    merged_frame, merge_summary = merge_aligned_sources(
        gps_source,
        marv_source,
        offset_s=float(offset_s),
        time_step_s=float(time_step_s),
        timebase_source="shared_uniform",
    )
    estimator_frame = _build_replay_estimator_frame(
        merged_frame,
        time_shift_s=float(overlap_start_s),
        reference_altitude_m=reference_altitude_m,
    )

    replay_result = run_rocketpy_replay(
        estimator_frame,
        config=RocketPyReplayConfig(
            gnss_is_geodetic=True,
            reference_latitude_deg=float(reference_latitude_deg),
            reference_longitude_deg=float(reference_longitude_deg),
            reference_altitude_m=float(reference_altitude_m),
            sea_level_pressure_pa=float(sea_level_pressure_pa),
            derive_parameters_from_telemetry=bool(derive_parameters_from_telemetry),
        ),
    )
    truth_frame = _build_replay_truth_frame(
        replay_result.estimates,
        estimator_frame,
    )

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    truth_path = output_path / "truth.csv"
    imu_path = output_path / "imu.csv"
    baro_path = output_path / "baro.csv"
    gps_path = output_path / "gps.csv"
    if session_id is None:
        session_id = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")

    manifest = build_session_manifest(
        session_id=session_id,
        vehicle_name=vehicle_name,
        generated_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        reference_latitude_deg=float(reference_latitude_deg),
        reference_longitude_deg=float(reference_longitude_deg),
        reference_altitude_m=float(reference_altitude_m),
        sea_level_pressure_pa=float(sea_level_pressure_pa),
        notes=(
            "Best-effort Lonestar replay session exported from aligned Featherweight GPS "
            "and MARV telemetry. truth.csv is derived from the layered navigation replay estimator."
        ),
    )

    truth_frame.to_csv(truth_path, index=False)
    imu_frame.to_csv(imu_path, index=False)
    baro_frame.to_csv(baro_path, index=False)
    gps_frame.to_csv(gps_path, index=False)
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "session_dir": str(output_path),
        "manifest_path": str(manifest_path),
        "session_id": session_id,
        "vehicle_name": vehicle_name,
        "offset_s": float(offset_s),
        "time_step_s": float(time_step_s),
        "overlap_start_s": float(overlap_start_s),
        "overlap_end_s": float(overlap_end_s),
        "reference_latitude_deg": float(reference_latitude_deg),
        "reference_longitude_deg": float(reference_longitude_deg),
        "reference_altitude_m": float(reference_altitude_m),
        "sea_level_pressure_pa": float(sea_level_pressure_pa),
        "truth_rows": int(len(truth_frame)),
        "imu_rows": int(len(imu_frame)),
        "baro_rows": int(len(baro_frame)),
        "gps_rows": int(len(gps_frame)),
        "truth_source": "layered_navigation_replay_estimate",
        "merge_summary": merge_summary,
        "stream_files": {
            "truth": str(truth_path),
            "imu": str(imu_path),
            "baro": str(baro_path),
            "gps": str(gps_path),
        },
    }
    summary_path = output_path / "session_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _resolve_replay_reference_lat_lon(navigation: pd.DataFrame) -> tuple[float, float]:
    latitude = pd.to_numeric(navigation["latitude_deg"], errors="coerce")
    longitude = pd.to_numeric(navigation["longitude_deg"], errors="coerce")
    finite_mask = latitude.notna() & longitude.notna()
    if not finite_mask.any():
        raise ValueError("Replay export requires finite GPS latitude/longitude samples")

    latitude_samples = latitude.loc[finite_mask].iloc[: min(10, int(finite_mask.sum()))]
    longitude_samples = longitude.loc[finite_mask].iloc[: min(10, int(finite_mask.sum()))]
    return float(latitude_samples.median()), float(longitude_samples.median())


def _resolve_replay_reference_altitude_m(navigation: pd.DataFrame) -> float:
    altitude_asl_m = pd.to_numeric(navigation["gps_altitude_asl_m"], errors="coerce")
    altitude_agl_m = pd.to_numeric(navigation["gps_altitude_agl_m"], errors="coerce")
    ground_asl_m = altitude_asl_m - altitude_agl_m

    candidate_mask = ground_asl_m.notna()
    if "distance_m" in navigation.columns:
        distance_m = pd.to_numeric(navigation["distance_m"], errors="coerce")
        candidate_mask &= distance_m.fillna(0.0) <= 10.0

    candidates = ground_asl_m.loc[candidate_mask]
    if candidates.empty:
        candidates = ground_asl_m.dropna()
    if candidates.empty:
        raise ValueError("Replay export requires GPS altitude samples to resolve reference altitude")
    return float(candidates.iloc[: min(20, len(candidates))].median())


def _resolve_replay_sea_level_pressure_pa(
    baro: pd.DataFrame,
    *,
    reference_altitude_m: float,
) -> float:
    from sim.estimation.adapters.rocketpy_replay import estimate_sea_level_pressure_pa

    pressure_pa = pd.to_numeric(baro["pressure_pa"], errors="coerce").dropna()
    if pressure_pa.empty:
        raise ValueError("Replay export requires a finite MARV baro pressure sample")
    reference_pressure_pa = float(pressure_pa.iloc[: min(20, len(pressure_pa))].median())
    return float(
        estimate_sea_level_pressure_pa(
            reference_pressure_pa=reference_pressure_pa,
            reference_altitude_m=float(reference_altitude_m),
        )
    )


def _build_replay_imu_frame(primary_imu: pd.DataFrame) -> pd.DataFrame:
    imu_frame = pd.DataFrame(
        {
            "time_s": pd.to_numeric(primary_imu["time_s"], errors="coerce"),
            "accelerometer_x": pd.to_numeric(primary_imu["accelerometer_x_mps2"], errors="coerce"),
            "accelerometer_y": pd.to_numeric(primary_imu["accelerometer_y_mps2"], errors="coerce"),
            "accelerometer_z": pd.to_numeric(primary_imu["accelerometer_z_mps2"], errors="coerce"),
            "gyroscope_x": pd.to_numeric(primary_imu["gyroscope_x_rad_s"], errors="coerce"),
            "gyroscope_y": pd.to_numeric(primary_imu["gyroscope_y_rad_s"], errors="coerce"),
            "gyroscope_z": pd.to_numeric(primary_imu["gyroscope_z_rad_s"], errors="coerce"),
        }
    )
    return imu_frame.loc[:, list(IMU_COLUMNS)].reset_index(drop=True)


def _build_replay_baro_frame(baro: pd.DataFrame) -> pd.DataFrame:
    baro_frame = pd.DataFrame(
        {
            "time_s": pd.to_numeric(baro["time_s"], errors="coerce"),
            "barometer_v1": pd.to_numeric(baro["pressure_pa"], errors="coerce"),
        }
    )
    return baro_frame.loc[:, list(BARO_COLUMNS)].reset_index(drop=True)


def _build_replay_gps_frame(
    navigation: pd.DataFrame,
    *,
    reference_altitude_m: float,
) -> pd.DataFrame:
    altitude_agl_m = pd.to_numeric(navigation["gps_altitude_agl_m"], errors="coerce")
    gps_frame = pd.DataFrame(
        {
            "time_s": pd.to_numeric(navigation["time_s"], errors="coerce"),
            "gnss_x": pd.to_numeric(navigation["latitude_deg"], errors="coerce"),
            "gnss_y": pd.to_numeric(navigation["longitude_deg"], errors="coerce"),
            "gnss_z": altitude_agl_m + float(reference_altitude_m),
        }
    )
    return gps_frame.loc[:, list(GPS_COLUMNS)].reset_index(drop=True)


def _build_replay_estimator_frame(
    merged_frame: pd.DataFrame,
    *,
    time_shift_s: float,
    reference_altitude_m: float,
) -> pd.DataFrame:
    altitude_agl_m = pd.to_numeric(
        merged_frame["featherweight_navigation_gps_altitude_agl_m"],
        errors="coerce",
    )
    replay_frame = pd.DataFrame(
        {
            "time_s": pd.to_numeric(merged_frame["time_s"], errors="coerce") - float(time_shift_s),
            "accelerometer_x": pd.to_numeric(
                merged_frame["marv_primary_imu_accelerometer_x_mps2"],
                errors="coerce",
            ),
            "accelerometer_y": pd.to_numeric(
                merged_frame["marv_primary_imu_accelerometer_y_mps2"],
                errors="coerce",
            ),
            "accelerometer_z": pd.to_numeric(
                merged_frame["marv_primary_imu_accelerometer_z_mps2"],
                errors="coerce",
            ),
            "gyroscope_x": pd.to_numeric(
                merged_frame["marv_primary_imu_gyroscope_x_rad_s"],
                errors="coerce",
            ),
            "gyroscope_y": pd.to_numeric(
                merged_frame["marv_primary_imu_gyroscope_y_rad_s"],
                errors="coerce",
            ),
            "gyroscope_z": pd.to_numeric(
                merged_frame["marv_primary_imu_gyroscope_z_rad_s"],
                errors="coerce",
            ),
            "barometer_v1": pd.to_numeric(merged_frame["marv_baro_pressure_pa"], errors="coerce"),
            "gnss_x": pd.to_numeric(
                merged_frame["featherweight_navigation_latitude_deg"],
                errors="coerce",
            ),
            "gnss_y": pd.to_numeric(
                merged_frame["featherweight_navigation_longitude_deg"],
                errors="coerce",
            ),
            "gnss_z": altitude_agl_m + float(reference_altitude_m),
        }
    )
    return replay_frame.sort_values("time_s").reset_index(drop=True)


def _build_replay_truth_frame(
    estimates: pd.DataFrame,
    estimator_frame: pd.DataFrame,
) -> pd.DataFrame:
    if len(estimates) != len(estimator_frame):
        raise ValueError("Replay truth construction requires estimator output and telemetry to share a timeline")

    gyroscope_x = pd.to_numeric(estimator_frame["gyroscope_x"], errors="coerce")
    gyroscope_y = pd.to_numeric(estimator_frame["gyroscope_y"], errors="coerce")
    gyroscope_z = pd.to_numeric(estimator_frame["gyroscope_z"], errors="coerce")

    truth = pd.DataFrame(
        {
            "time_s": pd.to_numeric(estimates["time_s"], errors="coerce"),
            "x_m": pd.to_numeric(estimates["est_x_m"], errors="coerce"),
            "y_m": pd.to_numeric(estimates["est_y_m"], errors="coerce"),
            "z_m": pd.to_numeric(estimates["est_z_m"], errors="coerce"),
            "vx_mps": pd.to_numeric(estimates["est_vx_mps"], errors="coerce"),
            "vy_mps": pd.to_numeric(estimates["est_vy_mps"], errors="coerce"),
            "vz_mps": pd.to_numeric(estimates["est_vz_mps"], errors="coerce"),
            "e0": pd.to_numeric(estimates["est_qw"], errors="coerce"),
            "e1": pd.to_numeric(estimates["est_qx"], errors="coerce"),
            "e2": pd.to_numeric(estimates["est_qy"], errors="coerce"),
            "e3": pd.to_numeric(estimates["est_qz"], errors="coerce"),
            "w1_radps": gyroscope_x - pd.to_numeric(estimates["est_bgx_rps"], errors="coerce"),
            "w2_radps": gyroscope_y - pd.to_numeric(estimates["est_bgy_rps"], errors="coerce"),
            "w3_radps": gyroscope_z - pd.to_numeric(estimates["est_bgz_rps"], errors="coerce"),
        }
    )
    return truth.loc[:, list(TRUTH_COLUMNS)].reset_index(drop=True)


def _inspect_command(args: argparse.Namespace) -> int:
    sources = _load_requested_sources(args)
    report = {
        "sources": {
            source.source_name: source.summary()
            for source in sources
        }
    }
    print(json.dumps(report, indent=2))
    return 0


def _normalize_command(args: argparse.Namespace) -> int:
    sources = _load_requested_sources(args)
    summary = write_normalized_sources(sources, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


def _align_baro_command(args: argparse.Namespace) -> int:
    featherweight_source = normalize_featherweight_csv(args.featherweight)
    marv_source = normalize_marv_csv(args.marv)
    isolated_marv_source, window_report = isolate_marv_flight_window(
        marv_source,
        margin_s=float(args.window_margin_s),
    )
    featherweight_duration_s = float(
        featherweight_source.streams["baro"]["time_s"].iloc[-1]
        - featherweight_source.streams["baro"]["time_s"].iloc[0]
    )
    isolated_marv_duration_s = float(
        isolated_marv_source.streams["baro"]["time_s"].iloc[-1]
        - isolated_marv_source.streams["baro"]["time_s"].iloc[0]
    )
    if isolated_marv_duration_s < featherweight_duration_s:
        isolated_marv_source = marv_source
        window_report = dict(window_report)
        window_report["fallback_to_full_marv"] = True
        window_report["fallback_reason"] = (
            "Detected MARV activity window was shorter than the Featherweight interval"
        )
    else:
        window_report = dict(window_report)
        window_report["fallback_to_full_marv"] = False

    alignment_report, featherweight_baro_debug, marv_baro_debug = align_baro_sources(
        featherweight_source,
        isolated_marv_source,
        coarse_step_s=float(args.coarse_step_s),
        refine_step_s=float(args.refine_step_s),
        featherweight_smoothing_window_s=float(args.featherweight_smoothing_window_s),
        marv_smoothing_window_s=float(args.marv_smoothing_window_s),
        edge_guard_s=float(args.edge_guard_s),
    )
    report = write_baro_alignment_artifacts(
        featherweight_source=featherweight_source,
        marv_source=marv_source,
        isolated_marv_source=isolated_marv_source,
        window_report=window_report,
        alignment_report=alignment_report,
        featherweight_baro_debug=featherweight_baro_debug,
        marv_baro_debug=marv_baro_debug,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))
    return 0


def _align_gps_baro_command(args: argparse.Namespace) -> int:
    gps_source = normalize_featherweight_gps_csv(args.gps)
    marv_source = normalize_marv_csv(args.marv)
    alignment_report, gps_debug, marv_debug = align_gps_altitude_to_marv_baro(
        gps_source,
        marv_source,
        coarse_step_s=float(args.coarse_step_s),
        refine_step_s=float(args.refine_step_s),
        gps_smoothing_window_s=float(args.gps_smoothing_window_s),
        marv_smoothing_window_s=float(args.marv_smoothing_window_s),
        prelaunch_margin_s=float(args.prelaunch_margin_s),
        postlanding_margin_s=float(args.postlanding_margin_s),
    )
    report = write_gps_baro_alignment_artifacts(
        gps_source=gps_source,
        marv_source=marv_source,
        alignment_report=alignment_report,
        gps_debug=gps_debug,
        marv_debug=marv_debug,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))
    return 0


def _merge_aligned_command(args: argparse.Namespace) -> int:
    if args.gps:
        aligned_source = normalize_featherweight_gps_csv(args.gps)
    elif args.featherweight:
        aligned_source = normalize_featherweight_csv(args.featherweight)
    else:
        raise ValueError("merge-aligned requires either --featherweight or --gps")
    marv_source = normalize_marv_csv(args.marv)
    offset_s, merge_context = _resolve_merge_alignment(args)

    merged, summary = merge_aligned_sources(
        aligned_source,
        marv_source,
        offset_s=float(offset_s),
        time_step_s=float(args.time_step_s),
        timebase_source=args.timebase_source,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    summary = {
        **summary,
        "output_csv": str(output_path),
        "alignment_context": merge_context,
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _export_replay_session_command(args: argparse.Namespace) -> int:
    gps_source = normalize_featherweight_gps_csv(args.gps)
    marv_source = normalize_marv_csv(args.marv)
    offset_s, merge_context = _resolve_merge_alignment(args)

    summary = export_aligned_replay_session(
        gps_source,
        marv_source,
        offset_s=float(offset_s),
        output_dir=args.output_dir,
        vehicle_name=args.vehicle_name,
        session_id=args.session_id,
        time_step_s=float(args.time_step_s),
    )
    summary["alignment_context"] = merge_context
    summary_path = Path(summary["summary_path"])
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _load_requested_sources(args: argparse.Namespace) -> list[NormalizedSource]:
    sources: list[NormalizedSource] = []
    if args.featherweight:
        sources.append(normalize_featherweight_csv(args.featherweight))
    if getattr(args, "gps", None):
        sources.append(normalize_featherweight_gps_csv(args.gps))
    if args.marv:
        sources.append(normalize_marv_csv(args.marv))
    if not sources:
        raise ValueError("Provide at least one of --featherweight, --gps, or --marv")
    return sources


def _resolve_merge_alignment(args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    if args.offset_s is not None:
        return float(args.offset_s), {"source": "manual_offset"}

    if not args.alignment_report:
        raise ValueError("Provide either --offset-s or --alignment-report for merge-aligned")

    report_path = _resolve_input_path(args.alignment_report)
    raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    alignment = raw_report.get("alignment", raw_report)
    if "offset_s" not in alignment:
        raise ValueError(f"Alignment report is missing offset_s: {report_path}")

    confidence = alignment.get("confidence")
    status = alignment.get("status")
    strong_match = bool(alignment.get("strong_match", False))
    if not args.allow_low_confidence:
        if (
            confidence is not None
            and float(confidence) < float(args.min_confidence)
            and status not in ("ok",)
            and not strong_match
        ):
            raise ValueError(
                "Alignment confidence is too low for merge-aligned. "
                "Pass --allow-low-confidence or supply a manual --offset-s after review."
            )
        if status not in (None, "ok") and not strong_match:
            raise ValueError(
                f"Alignment report status is {status!r}, so merge-aligned requires "
                "--allow-low-confidence or a manual --offset-s."
            )

    return float(alignment["offset_s"]), {
        "source": "alignment_report",
        "report_path": str(report_path),
        "confidence": confidence,
        "status": status,
        "strong_match": strong_match,
        "warnings": alignment.get("warnings", []),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize Lonestar telemetry exports into canonical analysis streams.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print normalized stream summaries.")
    inspect_parser.add_argument("--featherweight", help="Path to Featherweight CSV export.")
    inspect_parser.add_argument("--gps", help="Path to Featherweight GPS CSV export.")
    inspect_parser.add_argument("--marv", help="Path to MARV CSV export.")
    inspect_parser.set_defaults(func=_inspect_command)

    normalize_parser = subparsers.add_parser("normalize", help="Write normalized CSV streams.")
    normalize_parser.add_argument("--featherweight", help="Path to Featherweight CSV export.")
    normalize_parser.add_argument("--gps", help="Path to Featherweight GPS CSV export.")
    normalize_parser.add_argument("--marv", help="Path to MARV CSV export.")
    normalize_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where normalized streams and normalization_summary.json are written.",
    )
    normalize_parser.set_defaults(func=_normalize_command)

    align_parser = subparsers.add_parser(
        "align-baro",
        help="Isolate the MARV flight window and solve the Featherweight baro offset.",
    )
    align_parser.add_argument(
        "--featherweight",
        required=True,
        help="Path to Featherweight CSV export.",
    )
    align_parser.add_argument(
        "--marv",
        required=True,
        help="Path to MARV CSV export.",
    )
    align_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the windowed MARV streams and alignment report are written.",
    )
    align_parser.add_argument(
        "--coarse-step-s",
        type=float,
        default=DEFAULT_BARO_ALIGN_COARSE_STEP_S,
        help="Coarse offset search step in seconds.",
    )
    align_parser.add_argument(
        "--refine-step-s",
        type=float,
        default=DEFAULT_BARO_ALIGN_REFINE_STEP_S,
        help="Refinement offset search step in seconds.",
    )
    align_parser.add_argument(
        "--window-margin-s",
        type=float,
        default=2.0,
        help="Extra margin added around the detected MARV activity window.",
    )
    align_parser.add_argument(
        "--edge-guard-s",
        type=float,
        default=1.0,
        help="Amount of Featherweight edge time to downweight during baro alignment.",
    )
    align_parser.add_argument(
        "--featherweight-smoothing-window-s",
        type=float,
        default=0.75,
        help="Smoothing window applied to the Featherweight pressure trace.",
    )
    align_parser.add_argument(
        "--marv-smoothing-window-s",
        type=float,
        default=1.0,
        help="Smoothing window applied to the MARV pressure trace.",
    )
    align_parser.set_defaults(func=_align_baro_command)

    align_gps_parser = subparsers.add_parser(
        "align-gps-baro",
        help="Align Featherweight GPS altitude against MARV baro-derived altitude.",
    )
    align_gps_parser.add_argument(
        "--gps",
        required=True,
        help="Path to Featherweight GPS CSV export.",
    )
    align_gps_parser.add_argument(
        "--marv",
        required=True,
        help="Path to MARV CSV export.",
    )
    align_gps_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where GPS/MARV alignment debug files and report are written.",
    )
    align_gps_parser.add_argument(
        "--coarse-step-s",
        type=float,
        default=DEFAULT_GPS_ALIGN_COARSE_STEP_S,
        help="Coarse offset search step in seconds.",
    )
    align_gps_parser.add_argument(
        "--refine-step-s",
        type=float,
        default=DEFAULT_GPS_ALIGN_REFINE_STEP_S,
        help="Refinement offset search step in seconds.",
    )
    align_gps_parser.add_argument(
        "--gps-smoothing-window-s",
        type=float,
        default=0.5,
        help="Smoothing window applied to the GPS altitude trace.",
    )
    align_gps_parser.add_argument(
        "--marv-smoothing-window-s",
        type=float,
        default=0.5,
        help="Smoothing window applied to the MARV baro-derived altitude trace.",
    )
    align_gps_parser.add_argument(
        "--prelaunch-margin-s",
        type=float,
        default=1.0,
        help="Amount of GPS prelaunch time retained before launch detection for alignment.",
    )
    align_gps_parser.add_argument(
        "--postlanding-margin-s",
        type=float,
        default=1.0,
        help="Amount of GPS post-landing time retained after landing detection for alignment.",
    )
    align_gps_parser.set_defaults(func=_align_gps_baro_command)

    merge_parser = subparsers.add_parser(
        "merge-aligned",
        help="Write a merged Featherweight+MARV CSV on one aligned time base.",
    )
    merge_parser.add_argument(
        "--featherweight",
        help="Path to Featherweight CSV export.",
    )
    merge_parser.add_argument(
        "--gps",
        help="Path to Featherweight GPS CSV export.",
    )
    merge_parser.add_argument(
        "--marv",
        required=True,
        help="Path to MARV CSV export.",
    )
    merge_parser.add_argument(
        "--alignment-report",
        help="Path to baro_alignment_report.json produced by align-baro.",
    )
    merge_parser.add_argument(
        "--offset-s",
        type=float,
        help="Manual Featherweight-to-MARV time offset in seconds.",
    )
    merge_parser.add_argument(
        "--allow-low-confidence",
        action="store_true",
        help="Allow merge-aligned to proceed even when the alignment report is flagged as ambiguous.",
    )
    merge_parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MERGE_MIN_CONFIDENCE,
        help="Minimum report confidence required when using --alignment-report.",
    )
    merge_parser.add_argument(
        "--time-step-s",
        type=float,
        default=DEFAULT_MERGED_TIME_STEP_S,
        help="Time step used only when --timebase-source=shared_uniform.",
    )
    merge_parser.add_argument(
        "--timebase-source",
        choices=MERGE_TIMEBASE_CHOICES,
        default=DEFAULT_MERGE_TIMEBASE_SOURCE,
        help=(
            "Target timeline for the merged CSV. "
            "Use a MARV-native stream to preserve MARV sampling and interpolate Featherweight onto it."
        ),
    )
    merge_parser.add_argument(
        "--output",
        required=True,
        help="CSV path where the merged aligned telemetry is written.",
    )
    merge_parser.set_defaults(func=_merge_aligned_command)

    replay_parser = subparsers.add_parser(
        "export-replay-session",
        help="Write a manifest-based replay session for the 3D viewer from aligned GPS+MARV telemetry.",
    )
    replay_parser.add_argument(
        "--gps",
        required=True,
        help="Path to Featherweight GPS CSV export.",
    )
    replay_parser.add_argument(
        "--marv",
        required=True,
        help="Path to MARV CSV export.",
    )
    replay_parser.add_argument(
        "--alignment-report",
        help="Path to gps_baro_alignment_report.json produced by align-gps-baro.",
    )
    replay_parser.add_argument(
        "--offset-s",
        type=float,
        help="Manual GPS-to-MARV time offset in seconds.",
    )
    replay_parser.add_argument(
        "--allow-low-confidence",
        action="store_true",
        help="Allow replay export to proceed even when the alignment report is flagged as ambiguous.",
    )
    replay_parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MERGE_MIN_CONFIDENCE,
        help="Minimum report confidence required when using --alignment-report.",
    )
    replay_parser.add_argument(
        "--time-step-s",
        type=float,
        default=DEFAULT_MERGED_TIME_STEP_S,
        help="Merged telemetry time-step used for the estimator-backed truth stream.",
    )
    replay_parser.add_argument(
        "--vehicle-name",
        default=DEFAULT_REPLAY_VEHICLE_NAME,
        help="Vehicle name written into the replay manifest.",
    )
    replay_parser.add_argument(
        "--session-id",
        help="Optional explicit replay session id. Defaults to the current UTC timestamp.",
    )
    replay_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where manifest.json, truth.csv, imu.csv, baro.csv, and gps.csv are written.",
    )
    replay_parser.set_defaults(func=_export_replay_session_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to normalize telemetry: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
