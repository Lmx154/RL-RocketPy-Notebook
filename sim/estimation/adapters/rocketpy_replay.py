"""RocketPy telemetry replay adapter built on the layered navigation stack."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd

from sim.sitl.session import (
    ReplaySession,
    find_latest_session_manifest,
    load_replay_session,
    merge_replay_session_sensors,
)
from ..core import MeasurementUpdateResult, MeasurementUpdateStatus
from ..measurements import (
    BarometricAltitudeConfig,
    BarometricAltitudeMeasurementModel,
    GpsPositionConfig,
    GpsPositionMeasurementModel,
    GpsVelocityConfig,
    GpsVelocityMeasurementModel,
    GravityAlignmentConfig,
    GravityAlignmentMeasurementModel,
)
from ..models import (
    AttitudeErrorStateProcessModel,
    AttitudeProcessModelConfig,
    NavigationProcessModel,
    NavigationProcessModelConfig,
)
from ..stacks import (
    LayeredNavigationCovariance,
    LayeredNavigationStack,
    LayeredNavigationState,
)
from .rocket_flight_phase import GravityAlignmentFlightPhasePolicy


EARTH_RADIUS_M = 6378137.0


def pressure_to_altitude_m(pressure_pa: float, sea_level_pressure_pa: float) -> float:
    """Convert barometric pressure into altitude using the ISA troposphere model."""

    pressure_ratio = max(float(pressure_pa), 1.0) / max(float(sea_level_pressure_pa), 1.0)
    return 44330.0 * (1.0 - pressure_ratio ** 0.19029495718363465)


def estimate_sea_level_pressure_pa(
    reference_pressure_pa: float,
    reference_altitude_m: float,
) -> float:
    """Infer sea-level pressure from one reference pressure-altitude sample."""

    altitude_factor = max(1.0 - float(reference_altitude_m) / 44330.0, 1e-6)
    return float(reference_pressure_pa) / altitude_factor ** 5.255


def geodetic_to_local_enu(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    origin_altitude_m: float,
) -> np.ndarray:
    """Convert a geodetic sample into a local ENU frame using a local tangent approximation."""

    latitude_rad = np.deg2rad(latitude_deg)
    longitude_rad = np.deg2rad(longitude_deg)
    origin_latitude_rad = np.deg2rad(origin_latitude_deg)
    origin_longitude_rad = np.deg2rad(origin_longitude_deg)

    east_m = (
        (longitude_rad - origin_longitude_rad)
        * np.cos(0.5 * (latitude_rad + origin_latitude_rad))
        * EARTH_RADIUS_M
    )
    north_m = (latitude_rad - origin_latitude_rad) * EARTH_RADIUS_M
    up_m = float(altitude_m) - float(origin_altitude_m)
    return np.array([east_m, north_m, up_m], dtype=float)


@dataclass(slots=True)
class RocketPyReplayConfig:
    """Configuration for replaying RocketPy telemetry through the layered stack."""

    time_column: str = "time_s"
    accelerometer_columns: tuple[str, str, str] = (
        "accelerometer_x",
        "accelerometer_y",
        "accelerometer_z",
    )
    gyroscope_columns: tuple[str, str, str] = (
        "gyroscope_x",
        "gyroscope_y",
        "gyroscope_z",
    )
    barometer_column: Optional[str] = "barometer_v1"
    gnss_columns: Optional[tuple[str, str, str]] = ("gnss_x", "gnss_y", "gnss_z")
    gnss_is_geodetic: bool = True
    reference_latitude_deg: Optional[float] = None
    reference_longitude_deg: Optional[float] = None
    reference_altitude_m: Optional[float] = None
    sea_level_pressure_pa: Optional[float] = None
    initial_state: Optional[LayeredNavigationState] = None
    initial_covariance: Optional[LayeredNavigationCovariance] = None
    initial_quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    )
    attitude_process_model_config: AttitudeProcessModelConfig = field(
        default_factory=AttitudeProcessModelConfig
    )
    navigation_process_model_config: NavigationProcessModelConfig = field(
        default_factory=NavigationProcessModelConfig
    )
    gravity_alignment_std_mps2: float = 0.75
    gnss_position_std_m: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 3.0, 5.0], dtype=float)
    )
    gnss_velocity_std_mps: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 3.0, 4.0], dtype=float)
    )
    baro_altitude_std_m: float = 2.0
    derive_gnss_velocity: bool = True
    derive_parameters_from_telemetry: bool = True
    submit_gravity_alignment_updates: bool = True
    gravity_alignment_policy: GravityAlignmentFlightPhasePolicy | None = field(
        default_factory=GravityAlignmentFlightPhasePolicy
    )
    logs_directory: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "logs"
    )


@dataclass(slots=True)
class RocketPyReplayResult:
    """Outputs produced by replaying RocketPy telemetry through the layered stack."""

    estimates: pd.DataFrame
    stack: LayeredNavigationStack
    local_origin_lla: Optional[tuple[float, float, float]]
    sea_level_pressure_pa: Optional[float]
    telemetry_path: Optional[Path]
    parameter_source: str


def run_rocketpy_replay(
    telemetry: ReplaySession | str | Path | pd.DataFrame | None = None,
    config: Optional[RocketPyReplayConfig] = None,
) -> RocketPyReplayResult:
    """Replay RocketPy telemetry through the phase-6 layered navigation stack."""

    replay_config = copy.deepcopy(config) if config is not None else RocketPyReplayConfig()

    telemetry_source_path: Optional[Path] = None
    if telemetry is None:
        telemetry = _load_default_replay_source(replay_config.logs_directory)
        if isinstance(telemetry, ReplaySession):
            telemetry_source_path = telemetry.manifest_path
        else:
            telemetry_source_path = Path(telemetry)
    elif isinstance(telemetry, ReplaySession):
        telemetry_source_path = telemetry.manifest_path
    elif isinstance(telemetry, (str, Path)):
        telemetry_path = Path(telemetry)
        if telemetry_path.is_dir() or telemetry_path.name == "manifest.json":
            replay_session = load_replay_session(telemetry_path)
            telemetry = replay_session
            telemetry_source_path = replay_session.manifest_path
        else:
            telemetry_source_path = telemetry_path

    telemetry_frame = _load_telemetry_frame(telemetry, replay_config.time_column)

    if replay_config.derive_parameters_from_telemetry:
        _derive_stack_parameters_from_telemetry(telemetry_frame, replay_config)

    local_origin = _resolve_local_origin(telemetry_frame, replay_config)
    sea_level_pressure_pa = _resolve_sea_level_pressure(
        telemetry_frame,
        replay_config,
        local_origin,
    )
    initial_state = _build_initial_state(telemetry_frame, replay_config, local_origin)
    altitude_reference_m = _resolve_altitude_reference(replay_config, local_origin)

    stack = _build_layered_navigation_stack(
        replay_config,
        initial_state=initial_state,
    )

    results = []
    previous_time_s: Optional[float] = None
    previous_gnss_position: Optional[np.ndarray] = None
    previous_gnss_time_s: Optional[float] = None

    for row in telemetry_frame.itertuples(index=False):
        row_data = row._asdict()
        current_time_s = float(row_data[replay_config.time_column])
        accelerometer = _extract_vector(row_data, replay_config.accelerometer_columns)
        gyroscope = _extract_vector(row_data, replay_config.gyroscope_columns)

        gravity_decision = None
        if replay_config.gravity_alignment_policy is not None and np.all(np.isfinite(accelerometer)):
            gravity_decision = replay_config.gravity_alignment_policy.evaluate(
                accelerometer_mps2=accelerometer,
                vertical_velocity_mps=float(stack.state.velocity_mps[2]),
            )

        if (
            previous_time_s is not None
            and np.all(np.isfinite(accelerometer))
            and np.all(np.isfinite(gyroscope))
        ):
            stack.predict(
                accelerometer_mps2=accelerometer,
                gyroscope_rps=gyroscope,
                dt=current_time_s - previous_time_s,
                timestamp_s=current_time_s,
            )
        else:
            stack.last_timestamp_s = current_time_s
            stack.last_prediction_dt_s = None

        gravity_result: MeasurementUpdateResult[np.ndarray] | None = None
        gravity_submitted = False
        if replay_config.submit_gravity_alignment_updates and np.all(np.isfinite(accelerometer)):
            if gravity_decision is None or gravity_decision.submit_update:
                gravity_result = stack.update_gravity_alignment(accelerometer_mps2=accelerometer)
                gravity_submitted = True

        baro_result: MeasurementUpdateResult[float] | None = None
        if (
            replay_config.barometer_column
            and sea_level_pressure_pa is not None
            and replay_config.barometer_column in row_data
        ):
            pressure_pa = row_data.get(replay_config.barometer_column)
            if pd.notna(pressure_pa):
                baro_altitude_m = pressure_to_altitude_m(pressure_pa, sea_level_pressure_pa)
                baro_altitude_m -= altitude_reference_m
                baro_result = stack.update_barometric_altitude(altitude_m=baro_altitude_m)

        position_result: MeasurementUpdateResult[np.ndarray] | None = None
        velocity_result: MeasurementUpdateResult[np.ndarray] | None = None
        gnss_position = _extract_gnss_position(row_data, replay_config, local_origin)
        if gnss_position is not None:
            position_result = stack.update_position(position_m=gnss_position)

            if (
                replay_config.derive_gnss_velocity
                and previous_gnss_position is not None
                and previous_gnss_time_s is not None
                and current_time_s > previous_gnss_time_s
            ):
                gnss_velocity = (gnss_position - previous_gnss_position) / (
                    current_time_s - previous_gnss_time_s
                )
                velocity_result = stack.update_velocity(velocity_mps=gnss_velocity)

            previous_gnss_position = gnss_position
            previous_gnss_time_s = current_time_s

        results.append(
            _build_result_row(
                stack=stack,
                timestamp_s=current_time_s,
                flight_phase=(
                    None
                    if gravity_decision is None
                    else gravity_decision.phase.name
                ),
                gravity_alignment_submitted=gravity_submitted,
                gravity_result=gravity_result,
                baro_result=baro_result,
                position_result=position_result,
                velocity_result=velocity_result,
            )
        )
        previous_time_s = current_time_s

    return RocketPyReplayResult(
        estimates=pd.DataFrame(results),
        stack=stack,
        local_origin_lla=local_origin,
        sea_level_pressure_pa=sea_level_pressure_pa,
        telemetry_path=telemetry_source_path,
        parameter_source=(
            "telemetry_statistics"
            if replay_config.derive_parameters_from_telemetry
            else "config_values"
        ),
    )


def find_latest_telemetry_log(logs_directory: str | Path) -> Path:
    """Return the newest merged sensor log by timestamped filename."""

    warnings.warn(
        "find_latest_telemetry_log() is deprecated; prefer manifest-based replay sessions "
        "and merge_replay_session_sensors() for offline analysis.",
        DeprecationWarning,
        stacklevel=2,
    )
    logs_path = Path(logs_directory)
    candidates = sorted(logs_path.glob("virtual_sensors_full_rate_*.csv"))
    if not candidates:
        replay_session = _load_latest_session_for_legacy_compat(logs_path)
        return _materialize_legacy_sensor_log(replay_session, logs_path)
    return candidates[-1]


def find_latest_matching_log_pair(logs_directory: str | Path) -> tuple[Path, Optional[Path]]:
    """Return newest merged sensor log and same-timestamp kinematics log if present."""

    warnings.warn(
        "find_latest_matching_log_pair() is deprecated; prefer session manifests and "
        "stream-specific replay files.",
        DeprecationWarning,
        stacklevel=2,
    )
    logs_path = Path(logs_directory)
    sensor_log = find_latest_telemetry_log(logs_path)
    suffix = sensor_log.stem.removeprefix("virtual_sensors_full_rate_")
    kinematics_log = sensor_log.with_name(f"flight_kinematics_{suffix}.csv")
    if kinematics_log.exists():
        return sensor_log, kinematics_log

    replay_session = _load_latest_session_for_legacy_compat(logs_path)
    if str(replay_session.manifest.get("session_id", "")) != suffix:
        return sensor_log, None
    return sensor_log, _materialize_legacy_kinematics_log(replay_session, logs_path)


def _load_latest_session_for_legacy_compat(logs_directory: str | Path) -> ReplaySession:
    logs_path = Path(logs_directory)
    try:
        return load_replay_session(find_latest_session_manifest(logs_path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No telemetry logs found in {logs_path}. Expected files named "
            "virtual_sensors_full_rate_<timestamp>.csv or a manifest-based "
            "session directory under logs/session_*/manifest.json"
        ) from exc


def _materialize_legacy_sensor_log(
    replay_session: ReplaySession,
    logs_directory: str | Path,
) -> Path:
    logs_path = Path(logs_directory)
    session_id = str(replay_session.manifest["session_id"])
    legacy_path = logs_path / f"virtual_sensors_full_rate_{session_id}.csv"
    if not legacy_path.exists():
        merge_replay_session_sensors(replay_session).to_csv(legacy_path, index=False)
    return legacy_path


def _materialize_legacy_kinematics_log(
    replay_session: ReplaySession,
    logs_directory: str | Path,
) -> Path:
    logs_path = Path(logs_directory)
    session_id = str(replay_session.manifest["session_id"])
    legacy_path = logs_path / f"flight_kinematics_{session_id}.csv"
    if not legacy_path.exists():
        replay_session.truth.to_csv(legacy_path, index=False)
    return legacy_path


def _build_layered_navigation_stack(
    config: RocketPyReplayConfig,
    *,
    initial_state: LayeredNavigationState,
) -> LayeredNavigationStack:
    return LayeredNavigationStack(
        initial_state=initial_state,
        initial_covariance=config.initial_covariance,
        attitude_process_model=AttitudeErrorStateProcessModel(config.attitude_process_model_config),
        navigation_process_model=NavigationProcessModel(config.navigation_process_model_config),
        gravity_alignment_model=GravityAlignmentMeasurementModel(
            GravityAlignmentConfig(
                gravity_vector=np.asarray(
                    config.navigation_process_model_config.gravity_vector,
                    dtype=float,
                ),
                measurement_std_mps2=float(config.gravity_alignment_std_mps2),
            )
        ),
        position_measurement_model=GpsPositionMeasurementModel(
            GpsPositionConfig(measurement_std_m=np.asarray(config.gnss_position_std_m, dtype=float))
        ),
        velocity_measurement_model=GpsVelocityMeasurementModel(
            GpsVelocityConfig(measurement_std_mps=np.asarray(config.gnss_velocity_std_mps, dtype=float))
        ),
        barometric_altitude_model=BarometricAltitudeMeasurementModel(
            BarometricAltitudeConfig(measurement_std_m=float(config.baro_altitude_std_m))
        ),
    )


def _load_telemetry_frame(
    telemetry: ReplaySession | str | Path | pd.DataFrame,
    time_column: str,
) -> pd.DataFrame:
    if isinstance(telemetry, ReplaySession):
        telemetry_frame = merge_replay_session_sensors(telemetry)
    elif isinstance(telemetry, pd.DataFrame):
        telemetry_frame = telemetry.copy()
    else:
        telemetry_path = Path(telemetry)
        if telemetry_path.is_dir() or telemetry_path.name == "manifest.json":
            telemetry_frame = merge_replay_session_sensors(load_replay_session(telemetry_path))
        else:
            telemetry_frame = pd.read_csv(telemetry_path)
    return telemetry_frame.sort_values(time_column).reset_index(drop=True)


def _load_default_replay_source(logs_directory: str | Path) -> ReplaySession | Path:
    try:
        return load_replay_session(find_latest_session_manifest(logs_directory))
    except FileNotFoundError:
        return find_latest_telemetry_log(logs_directory)


def _derive_stack_parameters_from_telemetry(
    telemetry_frame: pd.DataFrame,
    config: RocketPyReplayConfig,
) -> None:
    """Estimate replay noise parameters directly from telemetry statistics."""

    dt_s = _estimate_nominal_dt_s(telemetry_frame, config.time_column)

    accelerometer = _extract_matrix(telemetry_frame, config.accelerometer_columns)
    gyroscope = _extract_matrix(telemetry_frame, config.gyroscope_columns)

    accel_noise_std = _vector_noise_from_differences(accelerometer)
    gyro_noise_std = _vector_noise_from_differences(gyroscope)
    accel_bias_rw_std = _vector_random_walk_from_smoothed_differences(accelerometer, dt_s)
    gyro_bias_rw_std = _vector_random_walk_from_smoothed_differences(gyroscope, dt_s)

    config.navigation_process_model_config.process_noise.accelerometer_noise_std = float(
        np.clip(np.nanmean(accel_noise_std), 1e-4, 10.0)
    )
    config.attitude_process_model_config.process_noise.gyroscope_noise_std = float(
        np.clip(np.nanmean(gyro_noise_std), 1e-6, 1.0)
    )
    config.navigation_process_model_config.process_noise.accel_bias_random_walk_std = float(
        np.clip(np.nanmean(accel_bias_rw_std), 1e-6, 5.0)
    )
    config.attitude_process_model_config.process_noise.gyro_bias_random_walk_std = float(
        np.clip(np.nanmean(gyro_bias_rw_std), 1e-8, 1.0)
    )
    config.gravity_alignment_std_mps2 = float(
        np.clip(np.nanmean(accel_noise_std) * 3.0, 0.1, 20.0)
    )

    if config.barometer_column and config.barometer_column in telemetry_frame.columns:
        baro_series = telemetry_frame[config.barometer_column].to_numpy(dtype=float)
        baro_noise_pa = _scalar_noise_from_differences(baro_series)
        finite_baro = baro_series[np.isfinite(baro_series)]
        if finite_baro.size >= 1:
            reference_pressure = float(finite_baro[0])
            local_origin_altitude_m = float(config.reference_altitude_m or 0.0)
            sea_level_pa = estimate_sea_level_pressure_pa(reference_pressure, local_origin_altitude_m)
            h_plus = pressure_to_altitude_m(reference_pressure + baro_noise_pa, sea_level_pa)
            h_nominal = pressure_to_altitude_m(reference_pressure, sea_level_pa)
            config.baro_altitude_std_m = float(np.clip(abs(h_plus - h_nominal), 0.25, 50.0))

    gnss_samples = _extract_gnss_samples(telemetry_frame, config)
    if gnss_samples is not None and gnss_samples.shape[0] >= 3:
        gnss_pos_noise_std = _vector_noise_from_differences(gnss_samples)
        config.gnss_position_std_m = np.clip(gnss_pos_noise_std, 0.5, 50.0)

        gnss_raw = telemetry_frame.loc[:, list(config.gnss_columns)].to_numpy(dtype=float)
        gnss_rows = np.all(np.isfinite(gnss_raw), axis=1)
        gnss_times = telemetry_frame[config.time_column].to_numpy(dtype=float)[gnss_rows]
        if gnss_times.size == gnss_samples.shape[0]:
            delta_t = np.diff(gnss_times)
            valid_dt = np.isfinite(delta_t) & (delta_t > 1e-6)
            if np.any(valid_dt):
                velocity = np.diff(gnss_samples, axis=0)[valid_dt] / delta_t[valid_dt, None]
                gnss_vel_noise_std = _vector_noise_from_differences(velocity)
                config.gnss_velocity_std_mps = np.clip(gnss_vel_noise_std, 0.5, 100.0)


def _estimate_nominal_dt_s(telemetry_frame: pd.DataFrame, time_column: str) -> float:
    time_values = telemetry_frame[time_column].to_numpy(dtype=float)
    delta = np.diff(time_values)
    delta = delta[np.isfinite(delta) & (delta > 0.0)]
    if delta.size == 0:
        return 0.01
    return float(np.median(delta))


def _extract_matrix(
    telemetry_frame: pd.DataFrame,
    columns: tuple[str, str, str],
) -> np.ndarray:
    return telemetry_frame.loc[:, list(columns)].to_numpy(dtype=float)


def _vector_noise_from_differences(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return np.ones(3, dtype=float)

    diffs = np.diff(samples, axis=0)
    noise_std = np.full(3, np.nan, dtype=float)
    for axis in range(min(3, diffs.shape[1])):
        axis_diff = diffs[:, axis]
        axis_diff = axis_diff[np.isfinite(axis_diff)]
        if axis_diff.size >= 2:
            noise_std[axis] = np.std(axis_diff, ddof=1) / np.sqrt(2.0)
    fallback = np.nanmedian(noise_std)
    if not np.isfinite(fallback):
        fallback = 1.0
    return np.where(np.isfinite(noise_std), noise_std, fallback)


def _vector_random_walk_from_smoothed_differences(samples: np.ndarray, dt_s: float) -> np.ndarray:
    if samples.size == 0:
        return np.ones(3, dtype=float) * 1e-3

    window = int(max(3, round(1.0 / max(dt_s, 1e-3))))
    frame = pd.DataFrame(samples)
    smoothed = frame.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    diffs = np.diff(smoothed, axis=0)
    rw_std = np.full(3, np.nan, dtype=float)
    for axis in range(min(3, diffs.shape[1])):
        axis_diff = diffs[:, axis]
        axis_diff = axis_diff[np.isfinite(axis_diff)]
        if axis_diff.size >= 2:
            rw_std[axis] = np.std(axis_diff, ddof=1) / np.sqrt(max(dt_s, 1e-6))
    fallback = np.nanmedian(rw_std)
    if not np.isfinite(fallback):
        fallback = 1e-3
    return np.where(np.isfinite(rw_std), rw_std, fallback)


def _scalar_noise_from_differences(series: np.ndarray) -> float:
    finite = series[np.isfinite(series)]
    if finite.size < 3:
        return 1.0
    diffs = np.diff(finite)
    if diffs.size < 2:
        return 1.0
    return float(np.std(diffs, ddof=1) / np.sqrt(2.0))


def _extract_gnss_samples(
    telemetry_frame: pd.DataFrame,
    config: RocketPyReplayConfig,
) -> Optional[np.ndarray]:
    if not config.gnss_columns:
        return None

    gnss = telemetry_frame.loc[:, list(config.gnss_columns)].to_numpy(dtype=float)
    finite_rows = np.all(np.isfinite(gnss), axis=1)
    gnss = gnss[finite_rows]
    if gnss.size == 0:
        return None

    if not config.gnss_is_geodetic:
        out = gnss.copy()
        if config.reference_altitude_m is not None:
            out[:, 2] -= float(config.reference_altitude_m)
        return out

    if (
        config.reference_latitude_deg is not None
        and config.reference_longitude_deg is not None
        and config.reference_altitude_m is not None
    ):
        origin = (
            float(config.reference_latitude_deg),
            float(config.reference_longitude_deg),
            float(config.reference_altitude_m),
        )
    else:
        origin = (float(gnss[0, 0]), float(gnss[0, 1]), float(gnss[0, 2]))

    return np.vstack(
        [
            geodetic_to_local_enu(
                latitude_deg=sample[0],
                longitude_deg=sample[1],
                altitude_m=sample[2],
                origin_latitude_deg=origin[0],
                origin_longitude_deg=origin[1],
                origin_altitude_m=origin[2],
            )
            for sample in gnss
        ]
    )


def _resolve_local_origin(
    telemetry_frame: pd.DataFrame,
    config: RocketPyReplayConfig,
) -> Optional[tuple[float, float, float]]:
    if not config.gnss_columns or not config.gnss_is_geodetic:
        return None

    if (
        config.reference_latitude_deg is not None
        and config.reference_longitude_deg is not None
        and config.reference_altitude_m is not None
    ):
        return (
            float(config.reference_latitude_deg),
            float(config.reference_longitude_deg),
            float(config.reference_altitude_m),
        )

    gnss_frame = telemetry_frame.loc[:, list(config.gnss_columns)].dropna()
    if gnss_frame.empty:
        return None

    first_row = gnss_frame.iloc[0]
    return (float(first_row.iloc[0]), float(first_row.iloc[1]), float(first_row.iloc[2]))


def _resolve_sea_level_pressure(
    telemetry_frame: pd.DataFrame,
    config: RocketPyReplayConfig,
    local_origin: Optional[tuple[float, float, float]],
) -> Optional[float]:
    if config.barometer_column is None or config.barometer_column not in telemetry_frame.columns:
        return None
    if config.sea_level_pressure_pa is not None:
        return float(config.sea_level_pressure_pa)

    baro_series = telemetry_frame[config.barometer_column].dropna()
    if baro_series.empty:
        return None

    reference_altitude_m = 0.0 if local_origin is None else local_origin[2]
    if config.reference_altitude_m is not None:
        reference_altitude_m = float(config.reference_altitude_m)

    return estimate_sea_level_pressure_pa(baro_series.iloc[0], reference_altitude_m)


def _resolve_altitude_reference(
    config: RocketPyReplayConfig,
    local_origin: Optional[tuple[float, float, float]],
) -> float:
    if config.reference_altitude_m is not None:
        return float(config.reference_altitude_m)
    if local_origin is not None:
        return float(local_origin[2])
    return 0.0


def _build_initial_state(
    telemetry_frame: pd.DataFrame,
    config: RocketPyReplayConfig,
    local_origin: Optional[tuple[float, float, float]],
) -> LayeredNavigationState:
    if config.initial_state is not None:
        return config.initial_state.copy()

    initial_position = np.zeros(3, dtype=float)
    if config.gnss_columns:
        for row in telemetry_frame.itertuples(index=False):
            gnss_position = _extract_gnss_position(row._asdict(), config, local_origin)
            if gnss_position is not None:
                initial_position = gnss_position
                break

    return LayeredNavigationState(
        quaternion=np.asarray(config.initial_quaternion, dtype=float).copy(),
        position_m=initial_position,
    )


def _extract_vector(row_data: dict, columns: tuple[str, str, str]) -> np.ndarray:
    return np.array([row_data.get(column, np.nan) for column in columns], dtype=float)


def _extract_gnss_position(
    row_data: dict,
    config: RocketPyReplayConfig,
    local_origin: Optional[tuple[float, float, float]],
) -> Optional[np.ndarray]:
    if not config.gnss_columns:
        return None

    gnss_measurement = _extract_vector(row_data, config.gnss_columns)
    if not np.all(np.isfinite(gnss_measurement)):
        return None

    if config.gnss_is_geodetic:
        if local_origin is None:
            return None
        return geodetic_to_local_enu(
            latitude_deg=gnss_measurement[0],
            longitude_deg=gnss_measurement[1],
            altitude_m=gnss_measurement[2],
            origin_latitude_deg=local_origin[0],
            origin_longitude_deg=local_origin[1],
            origin_altitude_m=local_origin[2],
        )

    gnss_position = gnss_measurement.copy()
    if config.reference_altitude_m is not None:
        gnss_position[2] -= float(config.reference_altitude_m)
    return gnss_position


def _build_result_row(
    *,
    stack: LayeredNavigationStack,
    timestamp_s: float,
    flight_phase: str | None,
    gravity_alignment_submitted: bool,
    gravity_result: MeasurementUpdateResult[np.ndarray] | None,
    baro_result: MeasurementUpdateResult[float] | None,
    position_result: MeasurementUpdateResult[np.ndarray] | None,
    velocity_result: MeasurementUpdateResult[np.ndarray] | None,
) -> dict[str, float | str | bool | None]:
    state = stack.state
    covariance = stack.covariance
    attitude_covariance = covariance.attitude
    navigation_covariance = covariance.navigation

    navigation_result = velocity_result or position_result or baro_result

    return {
        "time_s": timestamp_s,
        "est_x_m": float(state.position_m[0]),
        "est_y_m": float(state.position_m[1]),
        "est_z_m": float(state.position_m[2]),
        "est_vx_mps": float(state.velocity_mps[0]),
        "est_vy_mps": float(state.velocity_mps[1]),
        "est_vz_mps": float(state.velocity_mps[2]),
        "est_qw": float(state.quaternion[0]),
        "est_qx": float(state.quaternion[1]),
        "est_qy": float(state.quaternion[2]),
        "est_qz": float(state.quaternion[3]),
        "est_bgx_rps": float(state.gyro_bias_rps[0]),
        "est_bgy_rps": float(state.gyro_bias_rps[1]),
        "est_bgz_rps": float(state.gyro_bias_rps[2]),
        "est_bax_mps2": float(state.accel_bias_mps2[0]),
        "est_bay_mps2": float(state.accel_bias_mps2[1]),
        "est_baz_mps2": float(state.accel_bias_mps2[2]),
        "attitude_sigma_rad": float(np.sqrt(np.mean(np.diag(attitude_covariance)[0:3]))),
        "gyro_bias_sigma_rps": float(np.sqrt(np.mean(np.diag(attitude_covariance)[3:6]))),
        "position_sigma_m": float(np.sqrt(np.mean(np.diag(navigation_covariance)[0:3]))),
        "velocity_sigma_mps": float(np.sqrt(np.mean(np.diag(navigation_covariance)[3:6]))),
        "accel_bias_sigma_mps2": float(np.sqrt(np.mean(np.diag(navigation_covariance)[6:9]))),
        "trace_p": float(np.trace(attitude_covariance) + np.trace(navigation_covariance)),
        "flight_phase": flight_phase,
        "gravity_alignment_submitted": gravity_alignment_submitted,
        "gravity_alignment_update": _measurement_accepted(gravity_result),
        "baro_update": _measurement_accepted(baro_result),
        "gnss_position_update": _measurement_accepted(position_result),
        "gnss_velocity_update": _measurement_accepted(velocity_result),
        "attitude_update_label": None if gravity_result is None else gravity_result.label,
        "attitude_update_status": None if gravity_result is None else gravity_result.status.value,
        "navigation_update_label": None if navigation_result is None else navigation_result.label,
        "navigation_update_status": None if navigation_result is None else navigation_result.status.value,
    }


def _measurement_accepted(result: MeasurementUpdateResult[object] | None) -> bool:
    if result is None:
        return False
    return result.status is MeasurementUpdateStatus.ACCEPTED
