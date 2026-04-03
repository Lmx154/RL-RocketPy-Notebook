"""Generate quick-look post-flight plots from a merged Lonestar telemetry log."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import os
import sys
from typing import Any

import matplotlib

if "ipykernel" not in sys.modules and "IPython" not in sys.modules:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_378_137.0
STANDARD_GRAVITY_MPS2 = 9.80665
DEFAULT_ROCKET_CONFIG = Path("notebooks/Itzamna/rocket_config.py")
DEFAULT_BOOST_WINDOW_S = 5.0


@dataclass(slots=True)
class MotorOverlay:
    dry_mass_kg: float
    burn_start_s: float
    burn_end_s: float
    thrust_n: Any
    total_mass_kg: Any
    source_label: str


def load_merged_log(path: str | Path) -> pd.DataFrame:
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Merged log not found: {source_path}")
    return pd.read_csv(source_path)


def build_trajectory_frame(
    merged: pd.DataFrame,
    *,
    reference_latitude_deg: float | None,
    reference_longitude_deg: float | None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    latitude_deg = pd.to_numeric(
        merged["featherweight_navigation_latitude_deg"],
        errors="coerce",
    )
    longitude_deg = pd.to_numeric(
        merged["featherweight_navigation_longitude_deg"],
        errors="coerce",
    )
    altitude_agl_m = pd.to_numeric(
        merged["featherweight_navigation_gps_altitude_agl_m"],
        errors="coerce",
    )
    valid_mask = latitude_deg.notna() & longitude_deg.notna() & altitude_agl_m.notna()
    if not valid_mask.any():
        raise ValueError("Merged log is missing finite GPS latitude/longitude/altitude samples")

    latitude_valid = latitude_deg.loc[valid_mask].reset_index(drop=True)
    longitude_valid = longitude_deg.loc[valid_mask].reset_index(drop=True)
    altitude_valid = altitude_agl_m.loc[valid_mask].reset_index(drop=True)
    time_valid = pd.to_numeric(merged.loc[valid_mask, "time_s"], errors="coerce").reset_index(drop=True)

    if reference_latitude_deg is None:
        reference_latitude_deg = float(latitude_valid.iloc[: min(10, len(latitude_valid))].median())
    if reference_longitude_deg is None:
        reference_longitude_deg = float(longitude_valid.iloc[: min(10, len(longitude_valid))].median())

    reference_latitude_rad = np.deg2rad(reference_latitude_deg)
    east_m = (
        np.deg2rad(longitude_valid - float(reference_longitude_deg))
        * EARTH_RADIUS_M
        * np.cos(reference_latitude_rad)
    )
    north_m = np.deg2rad(latitude_valid - float(reference_latitude_deg)) * EARTH_RADIUS_M

    trajectory = pd.DataFrame(
        {
            "time_s": time_valid,
            "east_m": east_m,
            "north_m": north_m,
            "up_m": altitude_valid,
        }
    )
    return trajectory, {
        "reference_latitude_deg": float(reference_latitude_deg),
        "reference_longitude_deg": float(reference_longitude_deg),
    }


def estimate_apparent_thrust_curve(
    merged: pd.DataFrame,
    *,
    dry_mass_kg: float | None,
    motor_overlay: MotorOverlay | None,
    boost_window_s: float,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    time_s = pd.to_numeric(merged["time_s"], errors="coerce")
    ax = pd.to_numeric(merged["marv_primary_imu_accelerometer_x_mps2"], errors="coerce")
    ay = pd.to_numeric(merged["marv_primary_imu_accelerometer_y_mps2"], errors="coerce")
    az = pd.to_numeric(merged["marv_primary_imu_accelerometer_z_mps2"], errors="coerce")
    accel_norm_mps2 = np.sqrt(ax * ax + ay * ay + az * az)

    time_values = time_s.to_numpy(dtype=float)
    accel_values = accel_norm_mps2.to_numpy(dtype=float)
    launch_time_s = _resolve_launch_time_s(merged, time_values, accel_values)
    finite_mask = np.isfinite(time_values) & np.isfinite(accel_values)
    if not np.any(finite_mask):
        raise ValueError("Merged log does not contain finite time/acceleration samples")

    first_time_s = float(time_values[finite_mask][0])
    baseline_window_end_s = min(float(launch_time_s) - 0.15, first_time_s + 0.5)
    baseline_mask = finite_mask & (time_values <= baseline_window_end_s)
    if int(np.count_nonzero(baseline_mask)) < 3:
        baseline_mask = np.zeros(len(time_values), dtype=bool)
        finite_indices = np.flatnonzero(finite_mask)
        baseline_mask[finite_indices[: min(50, len(finite_indices))]] = True
    baseline_specific_force_mps2 = float(np.nanmedian(accel_values[baseline_mask]))
    excess_specific_force_mps2 = np.clip(
        accel_values - baseline_specific_force_mps2,
        0.0,
        None,
    )

    full_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "time_since_launch_s": time_s - float(launch_time_s),
            "accelerometer_x_mps2": ax,
            "accelerometer_y_mps2": ay,
            "accelerometer_z_mps2": az,
            "accelerometer_norm_mps2": accel_norm_mps2,
            "accelerometer_norm_g": accel_norm_mps2 / STANDARD_GRAVITY_MPS2,
            "excess_specific_force_mps2": excess_specific_force_mps2,
        }
    )

    nominal_thrust_n = None
    if motor_overlay is not None:
        motor_time_s = time_s.to_numpy(dtype=float) - float(launch_time_s)
        nominal_thrust_n = np.array(
            [
                _sample_motor_thrust_n(motor_overlay, motor_time)
                for motor_time in motor_time_s
            ],
            dtype=float,
        )
        full_frame["nominal_motor_thrust_n"] = nominal_thrust_n

    boost_mask = (
        np.isfinite(full_frame["time_since_launch_s"].to_numpy(dtype=float))
        & (full_frame["time_since_launch_s"].to_numpy(dtype=float) >= 0.0)
        & (full_frame["time_since_launch_s"].to_numpy(dtype=float) <= float(boost_window_s))
    )
    result = full_frame.loc[boost_mask].reset_index(drop=True)
    if result.empty:
        raise ValueError(
            "No samples fall inside the requested boost window. "
            f"launch_time_s={launch_time_s:.3f}, boost_window_s={boost_window_s:.3f}"
        )

    summary = {
        "launch_time_s": float(launch_time_s),
        "boost_window_s": float(boost_window_s),
        "window_sample_count": int(len(result)),
        "baseline_specific_force_mps2": baseline_specific_force_mps2,
        "peak_accelerometer_x_mps2": float(np.nanmax(np.abs(result["accelerometer_x_mps2"]))),
        "peak_accelerometer_y_mps2": float(np.nanmax(np.abs(result["accelerometer_y_mps2"]))),
        "peak_accelerometer_z_mps2": float(np.nanmax(np.abs(result["accelerometer_z_mps2"]))),
        "peak_accelerometer_norm_mps2": float(np.nanmax(result["accelerometer_norm_mps2"])),
        "peak_nominal_motor_thrust_n": (
            float(np.nanmax(nominal_thrust_n))
            if nominal_thrust_n is not None
            else None
        ),
    }
    return result, summary


def build_altitude_frame(merged: pd.DataFrame) -> pd.DataFrame:
    time_s = pd.to_numeric(merged["time_s"], errors="coerce")
    gps_altitude_source = (
        merged["featherweight_navigation_gps_altitude_agl_m"]
        if "featherweight_navigation_gps_altitude_agl_m" in merged.columns
        else pd.Series(np.nan, index=merged.index, dtype=float)
    )
    altitude_frame = pd.DataFrame(
        {
            "time_s": time_s,
            "gps_altitude_agl_m": pd.to_numeric(gps_altitude_source, errors="coerce"),
        }
    )

    baro_pressure_source = (
        merged["marv_baro_pressure_pa"]
        if "marv_baro_pressure_pa" in merged.columns
        else pd.Series(np.nan, index=merged.index, dtype=float)
    )
    baro_pressure_pa = pd.to_numeric(baro_pressure_source, errors="coerce")
    if baro_pressure_pa.notna().any():
        p0 = float(baro_pressure_pa.dropna().iloc[0])
        altitude_frame["marv_baro_altitude_rel_m"] = 44_330.0 * (
            1.0 - np.power(baro_pressure_pa / p0, 0.190294957)
        )
    return altitude_frame


def write_quicklook_plots(
    merged: pd.DataFrame,
    *,
    output_dir: str | Path,
    reference_latitude_deg: float | None,
    reference_longitude_deg: float | None,
    dry_mass_kg: float | None,
    motor_overlay: MotorOverlay | None,
    boost_window_s: float,
    title: str,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    trajectory, reference_summary = build_trajectory_frame(
        merged,
        reference_latitude_deg=reference_latitude_deg,
        reference_longitude_deg=reference_longitude_deg,
    )
    thrust_frame, thrust_summary = estimate_apparent_thrust_curve(
        merged,
        dry_mass_kg=dry_mass_kg,
        motor_overlay=motor_overlay,
        boost_window_s=float(boost_window_s),
    )
    altitude_frame = build_altitude_frame(merged)

    trajectory_path = output_path / "trajectory_3d.png"
    altitude_path = output_path / "altitude_profile.png"
    thrust_path = output_path / "thrust_curve.png"

    _plot_trajectory_3d(trajectory, title=title, output_path=trajectory_path)
    _plot_altitude_profile(altitude_frame, title=title, output_path=altitude_path)
    _plot_thrust_curve(
        thrust_frame,
        title=title,
        output_path=thrust_path,
        show_nominal=motor_overlay is not None,
        boost_window_s=float(boost_window_s),
    )

    apogee_index = int(trajectory["up_m"].idxmax())
    summary = {
        "output_dir": str(output_path),
        "files": {
            "trajectory_3d": str(trajectory_path),
            "altitude_profile": str(altitude_path),
            "thrust_curve": str(thrust_path),
        },
        "reference": reference_summary,
        "trajectory": {
            "samples": int(len(trajectory)),
            "max_altitude_agl_m": float(trajectory["up_m"].max()),
            "downrange_extent_m": float(np.sqrt(trajectory["east_m"] ** 2 + trajectory["north_m"] ** 2).max()),
            "apogee_time_s": float(trajectory.loc[apogee_index, "time_s"]),
        },
        "thrust": thrust_summary,
        "motor_overlay": (
            {
                "source_label": motor_overlay.source_label,
                "burn_start_s": float(motor_overlay.burn_start_s),
                "burn_end_s": float(motor_overlay.burn_end_s),
            }
            if motor_overlay is not None
            else None
        ),
    }
    summary_path = output_path / "postflight_plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _resolve_launch_time_s(
    merged: pd.DataFrame,
    time_s: np.ndarray,
    accel_norm_mps2: np.ndarray,
) -> float:
    finite_mask = np.isfinite(time_s) & np.isfinite(accel_norm_mps2)
    if not np.any(finite_mask):
        raise ValueError("Merged log does not contain finite time/acceleration samples")

    finite_indices = np.flatnonzero(finite_mask)
    seed_indices = finite_indices[: min(50, len(finite_indices))]
    baseline_seed_mps2 = float(np.nanmedian(accel_norm_mps2[seed_indices]))
    accel_threshold_mps2 = max(baseline_seed_mps2 + 5.0, baseline_seed_mps2 * 1.35)
    accel_launch_candidates = time_s[finite_mask & (accel_norm_mps2 >= accel_threshold_mps2)]

    launch_series = merged.get("featherweight_events_launch_detected")
    event_launch_time_s = None
    if launch_series is not None:
        launch_flags = launch_series.astype(str).str.lower().isin({"true", "1"})
        launch_times = pd.to_numeric(merged.loc[launch_flags, "time_s"], errors="coerce").dropna()
        if not launch_times.empty:
            event_launch_time_s = float(launch_times.iloc[0])

    if accel_launch_candidates.size > 0 and event_launch_time_s is not None:
        return float(min(event_launch_time_s, float(accel_launch_candidates[0])))
    if accel_launch_candidates.size > 0:
        return float(accel_launch_candidates[0])
    if event_launch_time_s is not None:
        return event_launch_time_s
    return float(time_s[finite_mask][0])


def _sample_motor_thrust_n(motor_overlay: MotorOverlay, motor_time_s: float) -> float:
    if motor_time_s < float(motor_overlay.burn_start_s) or motor_time_s > float(motor_overlay.burn_end_s):
        return 0.0
    return float(motor_overlay.thrust_n(float(motor_time_s)))


def _sample_motor_mass_kg(motor_overlay: MotorOverlay, motor_time_s: float) -> float:
    if motor_time_s < float(motor_overlay.burn_start_s):
        sample_time_s = float(motor_overlay.burn_start_s)
    elif motor_time_s > float(motor_overlay.burn_end_s):
        sample_time_s = float(motor_overlay.burn_end_s)
    else:
        sample_time_s = float(motor_time_s)
    return float(motor_overlay.total_mass_kg(sample_time_s))


def _plot_trajectory_3d(
    trajectory: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    axes.plot(
        trajectory["east_m"],
        trajectory["north_m"],
        trajectory["up_m"],
        linewidth=2.0,
        color="#0b6e4f",
    )
    axes.scatter(
        trajectory["east_m"].iloc[0],
        trajectory["north_m"].iloc[0],
        trajectory["up_m"].iloc[0],
        color="#d1495b",
        s=40,
        label="Launch",
    )
    apogee_index = int(trajectory["up_m"].idxmax())
    axes.scatter(
        trajectory["east_m"].iloc[apogee_index],
        trajectory["north_m"].iloc[apogee_index],
        trajectory["up_m"].iloc[apogee_index],
        color="#edae49",
        s=50,
        label="Apogee",
    )
    axes.set_title(f"{title}\n3D GPS Trajectory")
    axes.set_xlabel("East (m)")
    axes.set_ylabel("North (m)")
    axes.set_zlabel("Altitude AGL (m)")
    axes.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_altitude_profile(
    altitude_frame: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(
        altitude_frame["time_s"],
        altitude_frame["gps_altitude_agl_m"],
        label="GPS altitude AGL",
        color="#00798c",
        linewidth=2.0,
    )
    if "marv_baro_altitude_rel_m" in altitude_frame:
        axes.plot(
            altitude_frame["time_s"],
            altitude_frame["marv_baro_altitude_rel_m"],
            label="MARV baro altitude (relative)",
            color="#d1495b",
            linewidth=1.5,
            alpha=0.9,
        )
    axes.set_title(f"{title}\nAltitude Profile")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Altitude (m)")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_thrust_curve(
    thrust_frame: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    show_nominal: bool,
    boost_window_s: float,
) -> None:
    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(
        thrust_frame["time_since_launch_s"],
        thrust_frame["accelerometer_x_mps2"],
        label="Accel X",
        color="#1d3557",
        linewidth=1.7,
    )
    axes.plot(
        thrust_frame["time_since_launch_s"],
        thrust_frame["accelerometer_y_mps2"],
        label="Accel Y",
        color="#2a9d8f",
        linewidth=1.7,
    )
    axes.plot(
        thrust_frame["time_since_launch_s"],
        thrust_frame["accelerometer_z_mps2"],
        label="Accel Z",
        color="#e76f51",
        linewidth=1.9,
    )
    axes.plot(
        thrust_frame["time_since_launch_s"],
        thrust_frame["accelerometer_norm_mps2"],
        label="Accel norm",
        color="#6d597a",
        linewidth=1.2,
        linestyle=":",
        alpha=0.8,
    )
    axes.set_ylabel("Acceleration (m/s^2)")

    if show_nominal and "nominal_motor_thrust_n" in thrust_frame:
        thrust_axes = axes.twinx()
        thrust_axes.plot(
            thrust_frame["time_since_launch_s"],
            thrust_frame["nominal_motor_thrust_n"],
            label="Nominal thrust",
            color="#264653",
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
        )
        thrust_axes.set_ylabel("Nominal thrust (N)")
        thrust_handles, thrust_labels = thrust_axes.get_legend_handles_labels()
    else:
        thrust_handles, thrust_labels = [], []

    axes.set_title(f"{title}\nMARV Boost Acceleration Components (0-{boost_window_s:.1f} s)")
    axes.set_xlabel("Time Since Launch (s)")
    axes.set_xlim(0.0, float(boost_window_s))
    axes.grid(True, alpha=0.3)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles + thrust_handles, labels + thrust_labels, loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _load_motor_overlay(rocket_config_path: str | Path | None) -> tuple[float | None, MotorOverlay | None]:
    if rocket_config_path is None:
        return None, None

    config_path = Path(rocket_config_path).expanduser().resolve()
    if not config_path.exists():
        return None, None

    module = _load_module(config_path, module_name="plot_lonestar_rocket_config")
    dry_mass_kg = getattr(module, "ROCKET_DRY_MASS", None)
    if dry_mass_kg is not None:
        dry_mass_kg = float(dry_mass_kg)

    create_motor = getattr(module, "create_motor", None)
    if create_motor is None:
        return dry_mass_kg, None

    motor = create_motor()
    burn_time = getattr(motor, "burn_time", (0.0, getattr(motor, "burn_out_time", 0.0)))
    overlay = MotorOverlay(
        dry_mass_kg=float(getattr(module, "ROCKET_DRY_MASS", dry_mass_kg or 0.0)),
        burn_start_s=float(burn_time[0]),
        burn_end_s=float(burn_time[1]),
        thrust_n=motor.thrust,
        total_mass_kg=motor.total_mass,
        source_label=str(config_path),
    )
    return dry_mass_kg, overlay


def _load_module(path: Path, *, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate quick-look plots from a merged Lonestar telemetry CSV.",
    )
    parser.add_argument(
        "--merged",
        required=True,
        help="Path to merged telemetry CSV produced by tools/lonestar_telemetry.py merge-aligned.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where plot PNGs and a summary JSON will be written.",
    )
    parser.add_argument(
        "--title",
        default="Lonestar Post-Flight Quicklook",
        help="Title prefix used on the generated plots.",
    )
    parser.add_argument(
        "--reference-latitude-deg",
        type=float,
        help="Optional local tangent-plane reference latitude. Defaults to the median of the first GPS samples.",
    )
    parser.add_argument(
        "--reference-longitude-deg",
        type=float,
        help="Optional local tangent-plane reference longitude. Defaults to the median of the first GPS samples.",
    )
    parser.add_argument(
        "--dry-mass-kg",
        type=float,
        help="Optional dry vehicle mass used to convert acceleration into an apparent force curve.",
    )
    parser.add_argument(
        "--rocket-config",
        default=str(DEFAULT_ROCKET_CONFIG),
        help="Optional RocketPy config module used to load dry mass and nominal motor thrust overlay.",
    )
    parser.add_argument(
        "--boost-window-s",
        type=float,
        default=DEFAULT_BOOST_WINDOW_S,
        help="Launch-relative duration shown in the MARV acceleration plot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    merged = load_merged_log(args.merged)
    config_dry_mass_kg, motor_overlay = _load_motor_overlay(args.rocket_config)
    dry_mass_kg = float(args.dry_mass_kg) if args.dry_mass_kg is not None else config_dry_mass_kg
    summary = write_quicklook_plots(
        merged,
        output_dir=args.output_dir,
        reference_latitude_deg=args.reference_latitude_deg,
        reference_longitude_deg=args.reference_longitude_deg,
        dry_mass_kg=dry_mass_kg,
        motor_overlay=motor_overlay,
        boost_window_s=float(args.boost_window_s),
        title=args.title,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
