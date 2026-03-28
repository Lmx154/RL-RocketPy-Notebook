"""Shared MAVLink HIL encoding helpers for SITL replay paths."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pymavlink.dialects.v20 import common as mavlink2

from sim.estimation.adapters.rocketpy_replay import (
    estimate_sea_level_pressure_pa,
    pressure_to_altitude_m,
)
from .session import ReplaySession

# See: https://mavlink.io/en/messages/common.html#HIL_SENSOR_UPDATED_FLAGS
IMU_UPDATED_FIELDS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
BARO_UPDATED_FIELDS = (1 << 9) | (1 << 11)
ALL_STANDARD_HIL_SENSOR_FIELDS = IMU_UPDATED_FIELDS | BARO_UPDATED_FIELDS
EARTH_RADIUS_M = 6_378_137.0


@dataclass(slots=True)
class MavlinkHilCodec:
    """Encode standard MAVLink HIL messages with shared replay semantics."""

    reference_altitude_m: float
    sea_level_pressure_pa: float
    system_id: int = 1
    component_id: int = 1
    unix_epoch_base_usec: int = 0
    _buffer: io.BytesIO = field(init=False, repr=False)
    _mav: Any = field(init=False, repr=False)
    _previous_gnss: tuple[float, float, float, float] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._buffer = io.BytesIO()
        self._mav = mavlink2.MAVLink(self._buffer)
        self._mav.srcSystem = int(self.system_id)
        self._mav.srcComponent = int(self.component_id)

    @classmethod
    def from_telemetry(
        cls,
        telemetry: pd.DataFrame,
        *,
        system_id: int = 1,
        component_id: int = 1,
        unix_epoch_base_usec: int = 0,
    ) -> "MavlinkHilCodec":
        reference_altitude_m = resolve_reference_altitude_m(telemetry)
        sea_level_pressure_pa = resolve_sea_level_pressure_pa(
            telemetry,
            reference_altitude_m=reference_altitude_m,
        )
        return cls(
            reference_altitude_m=reference_altitude_m,
            sea_level_pressure_pa=sea_level_pressure_pa,
            system_id=system_id,
            component_id=component_id,
            unix_epoch_base_usec=unix_epoch_base_usec,
        )

    @classmethod
    def from_replay_session(
        cls,
        replay_session: ReplaySession,
        *,
        system_id: int = 1,
        component_id: int = 1,
        unix_epoch_base_usec: int = 0,
    ) -> "MavlinkHilCodec":
        reference_altitude_m = resolve_reference_altitude_from_session(replay_session)
        sea_level_pressure_pa = resolve_sea_level_pressure_from_session(
            replay_session,
            reference_altitude_m=reference_altitude_m,
        )
        return cls(
            reference_altitude_m=reference_altitude_m,
            sea_level_pressure_pa=sea_level_pressure_pa,
            system_id=system_id,
            component_id=component_id,
            unix_epoch_base_usec=unix_epoch_base_usec,
        )

    def reset_runtime_state(self) -> None:
        self._previous_gnss = None
        self._buffer.seek(0)
        self._buffer.truncate(0)

    def pack(self, message: Any) -> bytes:
        self._buffer.seek(0)
        self._buffer.truncate(0)
        self._mav.send(message)
        payload = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate(0)
        return payload

    def system_time_message(self, time_s: float) -> Any:
        return mavlink2.MAVLink_system_time_message(
            time_unix_usec=self.time_usec(time_s),
            time_boot_ms=int(round(float(time_s) * 1_000.0)),
        )

    def hil_sensor_message(
        self,
        time_s: float,
        sensors: dict[str, float | None],
        *,
        fields_updated: int,
    ) -> Any:
        pressure_pa = finite_float(
            sensors.get("barometer_v1"),
            default=self.sea_level_pressure_pa,
        )
        pressure_alt_m = pressure_to_altitude_m(pressure_pa, self.sea_level_pressure_pa)
        pressure_alt_m -= self.reference_altitude_m

        return mavlink2.MAVLink_hil_sensor_message(
            time_usec=self.time_usec(time_s),
            xacc=finite_float(sensors.get("accelerometer_x")),
            yacc=finite_float(sensors.get("accelerometer_y")),
            zacc=finite_float(sensors.get("accelerometer_z")),
            xgyro=finite_float(sensors.get("gyroscope_x")),
            ygyro=finite_float(sensors.get("gyroscope_y")),
            zgyro=finite_float(sensors.get("gyroscope_z")),
            xmag=0.0,
            ymag=0.0,
            zmag=0.0,
            abs_pressure=float(pressure_pa / 100.0),
            diff_pressure=0.0,
            pressure_alt=float(pressure_alt_m),
            temperature=0.0,
            fields_updated=int(fields_updated),
        )

    def hil_gps_message(
        self,
        time_s: float,
        sensors: dict[str, float | None],
    ) -> Any | None:
        latitude_deg = optional_finite_float(sensors.get("gnss_x"))
        longitude_deg = optional_finite_float(sensors.get("gnss_y"))
        altitude_m = optional_finite_float(sensors.get("gnss_z"))
        if latitude_deg is None or longitude_deg is None or altitude_m is None:
            return None

        vn_cms, ve_cms, vd_cms, speed_cms, cog_cdeg = self._derive_gps_velocity(
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            altitude_m=altitude_m,
            time_s=float(time_s),
        )
        self._previous_gnss = (float(time_s), latitude_deg, longitude_deg, altitude_m)

        return mavlink2.MAVLink_hil_gps_message(
            time_usec=self.time_usec(time_s),
            fix_type=3,
            lat=int(round(latitude_deg * 1e7)),
            lon=int(round(longitude_deg * 1e7)),
            alt=int(round(altitude_m * 1_000.0)),
            eph=100,
            epv=100,
            vel=int(round(speed_cms)),
            vn=int(round(vn_cms)),
            ve=int(round(ve_cms)),
            vd=int(round(vd_cms)),
            cog=int(round(cog_cdeg)),
            satellites_visible=10,
            id=0,
            yaw=0,
        )

    def time_usec(self, time_s: float) -> int:
        return int(self.unix_epoch_base_usec) + int(round(float(time_s) * 1_000_000.0))

    def _derive_gps_velocity(
        self,
        *,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float,
        time_s: float,
    ) -> tuple[float, float, float, float, float]:
        if self._previous_gnss is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        previous_time_s, previous_latitude_deg, previous_longitude_deg, previous_altitude_m = (
            self._previous_gnss
        )
        dt_s = time_s - previous_time_s
        if dt_s <= 1e-9:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        latitude_rad = math.radians(latitude_deg)
        previous_latitude_rad = math.radians(previous_latitude_deg)
        north_m = (latitude_rad - previous_latitude_rad) * EARTH_RADIUS_M
        east_m = (
            (math.radians(longitude_deg) - math.radians(previous_longitude_deg))
            * math.cos(0.5 * (latitude_rad + previous_latitude_rad))
            * EARTH_RADIUS_M
        )
        down_m = -(altitude_m - previous_altitude_m)

        vn_mps = north_m / dt_s
        ve_mps = east_m / dt_s
        vd_mps = down_m / dt_s
        speed_mps = math.sqrt(vn_mps * vn_mps + ve_mps * ve_mps + vd_mps * vd_mps)
        cog_deg = (math.degrees(math.atan2(ve_mps, vn_mps)) + 360.0) % 360.0

        return (
            vn_mps * 100.0,
            ve_mps * 100.0,
            vd_mps * 100.0,
            speed_mps * 100.0,
            cog_deg * 100.0,
        )


def resolve_reference_altitude_m(telemetry: pd.DataFrame) -> float:
    if "gnss_z" not in telemetry.columns:
        return 0.0

    valid_altitude = telemetry["gnss_z"].dropna()
    if valid_altitude.empty:
        return 0.0
    return float(valid_altitude.iloc[0])


def resolve_sea_level_pressure_pa(
    telemetry: pd.DataFrame,
    *,
    reference_altitude_m: float,
) -> float:
    if "barometer_v1" not in telemetry.columns:
        return 101_325.0

    valid_pressure = telemetry["barometer_v1"].dropna()
    if valid_pressure.empty:
        return 101_325.0

    return estimate_sea_level_pressure_pa(
        reference_pressure_pa=float(valid_pressure.iloc[0]),
        reference_altitude_m=reference_altitude_m,
    )


def resolve_reference_altitude_from_session(replay_session: ReplaySession) -> float:
    return resolve_reference_altitude_m(replay_session.gps)


def resolve_sea_level_pressure_from_session(
    replay_session: ReplaySession,
    *,
    reference_altitude_m: float,
) -> float:
    return resolve_sea_level_pressure_pa(
        replay_session.baro,
        reference_altitude_m=reference_altitude_m,
    )


def optional_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def finite_float(value: Any, *, default: float = 0.0) -> float:
    maybe_value = optional_finite_float(value)
    if maybe_value is None:
        return float(default)
    return maybe_value


__all__ = [
    "ALL_STANDARD_HIL_SENSOR_FIELDS",
    "BARO_UPDATED_FIELDS",
    "EARTH_RADIUS_M",
    "IMU_UPDATED_FIELDS",
    "MavlinkHilCodec",
    "finite_float",
    "optional_finite_float",
    "resolve_reference_altitude_from_session",
    "resolve_reference_altitude_m",
    "resolve_sea_level_pressure_from_session",
    "resolve_sea_level_pressure_pa",
]
