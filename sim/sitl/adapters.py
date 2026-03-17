"""Protocol adapters that map replay samples onto transport payloads."""

from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pymavlink.dialects.v20 import common as mavlink2

from sim.estimation.adapters.rocketpy_replay import (
    estimate_sea_level_pressure_pa,
    pressure_to_altitude_m,
)

from .replay import ReplayClock, ReplaySample


_HIL_SENSOR_FIELDS_UPDATED = (
    (1 << 0)
    | (1 << 1)
    | (1 << 2)
    | (1 << 3)
    | (1 << 4)
    | (1 << 5)
    | (1 << 9)
    | (1 << 11)
)
_EARTH_RADIUS_M = 6378137.0


@dataclass(slots=True)
class EncodedPacket:
    """Single encoded payload produced by an adapter."""

    payload: bytes
    message_type: str
    content_type: str


class ProtocolAdapter(ABC):
    """Maps replay events to one or more transport payloads."""

    name: str

    @abstractmethod
    def encode_event(
        self,
        *,
        event: str,
        clock: ReplayClock,
        sample: ReplaySample,
    ) -> list[EncodedPacket]:
        raise NotImplementedError


class JsonUdpAdapter(ProtocolAdapter):
    """Simple JSON datagram adapter for debugging and custom firmware glue."""

    name = "json"

    def encode_event(
        self,
        *,
        event: str,
        clock: ReplayClock,
        sample: ReplaySample,
    ) -> list[EncodedPacket]:
        payload = {
            "type": "replay_sample",
            "protocol": self.name,
            "event": event,
            "state": clock.snapshot(),
            "sample": {
                "index": sample.index,
                "time_s": sample.time_s,
                "row": sample.json_row(),
            },
        }
        return [
            EncodedPacket(
                payload=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                message_type="json.replay_sample",
                content_type="application/json",
            )
        ]


class MavlinkCommonAdapter(ProtocolAdapter):
    """MAVLink adapter backed by the standard common.xml dialect."""

    name = "mavlink-common"

    def __init__(
        self,
        telemetry: pd.DataFrame,
        *,
        system_id: int = 1,
        component_id: int = 1,
        unix_epoch_base_usec: int = 0,
    ) -> None:
        self._buffer = io.BytesIO()
        self._mav = mavlink2.MAVLink(self._buffer)
        self._mav.srcSystem = int(system_id)
        self._mav.srcComponent = int(component_id)
        self._unix_epoch_base_usec = int(unix_epoch_base_usec)
        self._reference_altitude_m = _resolve_reference_altitude_m(telemetry)
        self._sea_level_pressure_pa = _resolve_sea_level_pressure_pa(
            telemetry,
            reference_altitude_m=self._reference_altitude_m,
        )
        self._previous_gnss: tuple[float, float, float, float] | None = None

    def encode_event(
        self,
        *,
        event: str,
        clock: ReplayClock,
        sample: ReplaySample,
    ) -> list[EncodedPacket]:
        del event, clock

        packets = [
            EncodedPacket(
                payload=self._pack_message(self._build_system_time_message(sample)),
                message_type="SYSTEM_TIME",
                content_type="application/mavlink",
            ),
            EncodedPacket(
                payload=self._pack_message(self._build_hil_sensor_message(sample)),
                message_type="HIL_SENSOR",
                content_type="application/mavlink",
            ),
        ]

        gps_message = self._build_hil_gps_message(sample)
        if gps_message is not None:
            packets.append(
                EncodedPacket(
                    payload=self._pack_message(gps_message),
                    message_type="HIL_GPS",
                    content_type="application/mavlink",
                )
            )

        return packets

    def _pack_message(self, message: Any) -> bytes:
        self._buffer.seek(0)
        self._buffer.truncate(0)
        self._mav.send(message)
        payload = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate(0)
        return payload

    def _build_system_time_message(self, sample: ReplaySample) -> Any:
        time_usec = self._time_usec(sample.time_s)
        return mavlink2.MAVLink_system_time_message(
            time_unix_usec=time_usec,
            time_boot_ms=int(round(sample.time_s * 1000.0)),
        )

    def _build_hil_sensor_message(self, sample: ReplaySample) -> Any:
        row = sample.row
        pressure_pa = _finite_float(row.get("barometer_v1"), default=self._sea_level_pressure_pa)
        pressure_alt_m = pressure_to_altitude_m(pressure_pa, self._sea_level_pressure_pa)
        pressure_alt_m -= self._reference_altitude_m

        return mavlink2.MAVLink_hil_sensor_message(
            time_usec=self._time_usec(sample.time_s),
            xacc=_finite_float(row.get("accelerometer_x")),
            yacc=_finite_float(row.get("accelerometer_y")),
            zacc=_finite_float(row.get("accelerometer_z")),
            xgyro=_finite_float(row.get("gyroscope_x")),
            ygyro=_finite_float(row.get("gyroscope_y")),
            zgyro=_finite_float(row.get("gyroscope_z")),
            xmag=0.0,
            ymag=0.0,
            zmag=0.0,
            abs_pressure=float(pressure_pa / 100.0),
            diff_pressure=0.0,
            pressure_alt=float(pressure_alt_m),
            temperature=0.0,
            fields_updated=_HIL_SENSOR_FIELDS_UPDATED,
        )

    def _build_hil_gps_message(self, sample: ReplaySample) -> Any | None:
        row = sample.row
        latitude_deg = _maybe_finite_float(row.get("gnss_x"))
        longitude_deg = _maybe_finite_float(row.get("gnss_y"))
        altitude_m = _maybe_finite_float(row.get("gnss_z"))
        if latitude_deg is None or longitude_deg is None or altitude_m is None:
            return None

        vn_cms, ve_cms, vd_cms, speed_cms, cog_cdeg = self._derive_gps_velocity(
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            altitude_m=altitude_m,
            time_s=sample.time_s,
        )
        self._previous_gnss = (sample.time_s, latitude_deg, longitude_deg, altitude_m)

        return mavlink2.MAVLink_hil_gps_message(
            time_usec=self._time_usec(sample.time_s),
            fix_type=3,
            lat=int(round(latitude_deg * 1e7)),
            lon=int(round(longitude_deg * 1e7)),
            alt=int(round(altitude_m * 1000.0)),
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

        latitude_rad = np.deg2rad(latitude_deg)
        previous_latitude_rad = np.deg2rad(previous_latitude_deg)
        longitude_rad = np.deg2rad(longitude_deg)
        previous_longitude_rad = np.deg2rad(previous_longitude_deg)

        north_m = (latitude_rad - previous_latitude_rad) * _EARTH_RADIUS_M
        east_m = (
            (longitude_rad - previous_longitude_rad)
            * np.cos(0.5 * (latitude_rad + previous_latitude_rad))
            * _EARTH_RADIUS_M
        )
        down_m = -(altitude_m - previous_altitude_m)

        vn_mps = north_m / dt_s
        ve_mps = east_m / dt_s
        vd_mps = down_m / dt_s
        speed_mps = float(np.sqrt(vn_mps**2 + ve_mps**2 + vd_mps**2))
        cog_deg = float((np.degrees(np.arctan2(ve_mps, vn_mps)) + 360.0) % 360.0)

        return (
            vn_mps * 100.0,
            ve_mps * 100.0,
            vd_mps * 100.0,
            speed_mps * 100.0,
            cog_deg * 100.0,
        )

    def _time_usec(self, time_s: float) -> int:
        return self._unix_epoch_base_usec + int(round(time_s * 1_000_000.0))


def _resolve_reference_altitude_m(telemetry: pd.DataFrame) -> float:
    if "gnss_z" not in telemetry.columns:
        return 0.0

    valid_altitude = telemetry["gnss_z"].dropna()
    if valid_altitude.empty:
        return 0.0
    return float(valid_altitude.iloc[0])


def _resolve_sea_level_pressure_pa(
    telemetry: pd.DataFrame,
    *,
    reference_altitude_m: float,
) -> float:
    if "barometer_v1" not in telemetry.columns:
        return 101325.0

    valid_pressure = telemetry["barometer_v1"].dropna()
    if valid_pressure.empty:
        return 101325.0

    return estimate_sea_level_pressure_pa(
        reference_pressure_pa=float(valid_pressure.iloc[0]),
        reference_altitude_m=reference_altitude_m,
    )


def _maybe_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    maybe_value = _maybe_finite_float(value)
    if maybe_value is None:
        return float(default)
    return maybe_value