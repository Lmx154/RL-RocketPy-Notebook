"""Rate-aware MAVLink SITL service for GUI step-through integration.

Emits MAVLink HIL_SENSOR / HIL_GPS packets over UDP, gated by per-sensor
freshness flags so each sensor only fires at its own data rate — mirroring
real hardware behaviour (IMU at ~800 Hz, baro at ~50 Hz, GPS at ~5 Hz).

Usage::

    service = SitlMavlinkService(sensors_df)
    service.start()            # defaults to 127.0.0.1:14560
    # ... per GUI step:
    service.emit_state(state)  # state dict from CsvReplayController
    service.stop()
"""

from __future__ import annotations

import io
import socket
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pymavlink.dialects.v20 import common as mavlink2

from sim.estimation.adapters.rocketpy_replay import (
    estimate_sea_level_pressure_pa,
    pressure_to_altitude_m,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MAVLink HIL_SENSOR fields_updated bitmask constants
# See: https://mavlink.io/en/messages/common.html#HIL_SENSOR_UPDATED_FLAGS
# ---------------------------------------------------------------------------
_IMU_FIELDS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
_BARO_FIELDS = (1 << 9) | (1 << 11)  # abs_pressure + pressure_alt
_EARTH_RADIUS_M = 6_378_137.0


@dataclass
class SitlMavlinkService:
    """Rate-aware MAVLink SITL emitter for GUI step-through replay.

    Parameters
    ----------
    sensors_df:
        The loaded sensor DataFrame (same one passed to CsvReplayController).
        Used only to derive reference altitude and sea-level pressure.
    host:
        UDP destination host (default: loopback, for local PX4 / ArduPilot SITL).
    port:
        UDP destination port (PX4 HIL default: 14560).
    system_id / component_id:
        MAVLink source IDs embedded into every packet.
    unix_epoch_base_usec:
        Offset added to flight time_s to form the SYSTEM_TIME timestamp.
        Leave at 0 to emit simulation-relative times.
    """

    sensors_df: pd.DataFrame
    host: str = "127.0.0.1"
    port: int = 14560
    system_id: int = 1
    component_id: int = 1
    unix_epoch_base_usec: int = 0
    # Optional callback invoked after each emit_state() call with a human-readable
    # summary string. Set by the GUI to feed the MAVLink output window.
    on_emit: Callable[[str], None] | None = field(default=None, repr=False)

    # private state — not part of the public API
    _active: bool = field(default=False, init=False, repr=False)
    _socket: socket.socket | None = field(default=None, init=False, repr=False)
    _mav: Any = field(default=None, init=False, repr=False)
    _buf: io.BytesIO = field(default_factory=io.BytesIO, init=False, repr=False)
    _reference_altitude_m: float = field(default=0.0, init=False, repr=False)
    _sea_level_pressure_pa: float = field(default=101_325.0, init=False, repr=False)
    _previous_gnss: tuple[float, float, float, float] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._reference_altitude_m = _resolve_reference_altitude(self.sensors_df)
        self._sea_level_pressure_pa = _resolve_sea_level_pressure(
            self.sensors_df, self._reference_altitude_m
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, host: str | None = None, port: int | None = None) -> None:
        """Open the UDP socket and begin accepting emit_state() calls."""
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._buf = io.BytesIO()
        self._mav = mavlink2.MAVLink(self._buf)
        self._mav.srcSystem = int(self.system_id)
        self._mav.srcComponent = int(self.component_id)
        self._previous_gnss = None
        self._active = True
        log.info("SitlMavlinkService started → udp://%s:%d", self.host, self.port)

    def stop(self) -> None:
        """Disable emission and close the socket."""
        self._active = False
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        log.info("SitlMavlinkService stopped")

    @property
    def active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Per-step emission
    # ------------------------------------------------------------------

    def emit_state(self, state: dict[str, Any]) -> None:
        """Emit MAVLink packets for the given simulation state.

        Uses sensor_freshness to gate each sensor type so only sensors
        with new data at this step generate packets — matching real hardware
        sample rates naturally.
        """
        if not self._active or self._socket is None:
            return

        sensors: dict[str, float | None] = state.get("sensors", {})
        freshness: dict[str, bool] = state.get("sensor_freshness", {})
        time_s: float = float(state.get("time", 0.0))

        # Always emit SYSTEM_TIME so the FC tracks sim wall-clock
        self._send(self._pack(self._system_time_msg(time_s)))

        # Determine which sensor groups have fresh data this step
        imu_fresh = bool(freshness.get("accelerometer_x", False))
        baro_fresh = bool(freshness.get("barometer_v1", False))
        fields_updated = (
            (_IMU_FIELDS if imu_fresh else 0)
            | (_BARO_FIELDS if baro_fresh else 0)
        )

        emitted_sensor = False
        if fields_updated:
            self._send(self._pack(self._hil_sensor_msg(time_s, sensors, fields_updated)))
            emitted_sensor = True

        gps_fresh = bool(freshness.get("gnss_x", False))
        emitted_gps = False
        if gps_fresh:
            gps_msg = self._hil_gps_msg(time_s, sensors)
            if gps_msg is not None:
                self._send(self._pack(gps_msg))
                emitted_gps = True

        if self.on_emit is not None:
            self.on_emit(self._format_emit_line(
                time_s, sensors, freshness,
                imu_fresh, baro_fresh, emitted_sensor, fields_updated,
                emitted_gps,
            ))

    def _format_emit_line(
        self,
        time_s: float,
        sensors: dict[str, float | None],
        freshness: dict[str, bool],
        imu_fresh: bool,
        baro_fresh: bool,
        emitted_sensor: bool,
        fields_updated: int,
        emitted_gps: bool,
    ) -> str:
        parts = [f"t={time_s:.4f}s  SYSTEM_TIME"]

        if emitted_sensor:
            flags = []
            if imu_fresh:
                ax = _safe_float(sensors.get("accelerometer_x"))
                ay = _safe_float(sensors.get("accelerometer_y"))
                az = _safe_float(sensors.get("accelerometer_z"))
                gx = _safe_float(sensors.get("gyroscope_x"))
                gy = _safe_float(sensors.get("gyroscope_y"))
                gz = _safe_float(sensors.get("gyroscope_z"))
                flags.append(
                    f"IMU acc=({ax:.3f},{ay:.3f},{az:.3f}) "
                    f"gyro=({gx:.3f},{gy:.3f},{gz:.3f}) m/s²,rad/s"
                )
            if baro_fresh:
                pressure_pa = _safe_float(sensors.get("barometer_v1"), self._sea_level_pressure_pa)
                flags.append(f"BARO {pressure_pa:.1f} Pa")
            parts.append("HIL_SENSOR [" + "  ".join(flags) + f"]  fields=0x{fields_updated:04x}")

        if emitted_gps:
            lat = _safe_float(sensors.get("gnss_x"))
            lon = _safe_float(sensors.get("gnss_y"))
            alt = _safe_float(sensors.get("gnss_z"))
            parts.append(f"HIL_GPS lat={lat:.6f}° lon={lon:.6f}° alt={alt:.1f}m")

        return "  |  ".join(parts)

    # ------------------------------------------------------------------
    # MAVLink message builders
    # ------------------------------------------------------------------

    def _system_time_msg(self, time_s: float) -> Any:
        time_usec = self.unix_epoch_base_usec + int(round(time_s * 1_000_000.0))
        return mavlink2.MAVLink_system_time_message(
            time_unix_usec=time_usec,
            time_boot_ms=int(round(time_s * 1_000.0)),
        )

    def _hil_sensor_msg(
        self, time_s: float, sensors: dict[str, float | None], fields_updated: int
    ) -> Any:
        pressure_pa = _safe_float(sensors.get("barometer_v1"), self._sea_level_pressure_pa)
        pressure_alt_m = pressure_to_altitude_m(pressure_pa, self._sea_level_pressure_pa)
        pressure_alt_m -= self._reference_altitude_m

        return mavlink2.MAVLink_hil_sensor_message(
            time_usec=self.unix_epoch_base_usec + int(round(time_s * 1_000_000.0)),
            xacc=_safe_float(sensors.get("accelerometer_x")),
            yacc=_safe_float(sensors.get("accelerometer_y")),
            zacc=_safe_float(sensors.get("accelerometer_z")),
            xgyro=_safe_float(sensors.get("gyroscope_x")),
            ygyro=_safe_float(sensors.get("gyroscope_y")),
            zgyro=_safe_float(sensors.get("gyroscope_z")),
            xmag=0.0,
            ymag=0.0,
            zmag=0.0,
            abs_pressure=float(pressure_pa / 100.0),  # Pa → hPa (mbar)
            diff_pressure=0.0,
            pressure_alt=float(pressure_alt_m),
            temperature=0.0,
            fields_updated=fields_updated,
        )

    def _hil_gps_msg(self, time_s: float, sensors: dict[str, float | None]) -> Any | None:
        lat = _safe_optional_float(sensors.get("gnss_x"))
        lon = _safe_optional_float(sensors.get("gnss_y"))
        alt = _safe_optional_float(sensors.get("gnss_z"))
        if lat is None or lon is None or alt is None:
            return None

        vn, ve, vd, speed, cog = self._derive_gps_velocity(lat, lon, alt, time_s)
        self._previous_gnss = (time_s, lat, lon, alt)

        return mavlink2.MAVLink_hil_gps_message(
            time_usec=self.unix_epoch_base_usec + int(round(time_s * 1_000_000.0)),
            fix_type=3,
            lat=int(round(lat * 1e7)),
            lon=int(round(lon * 1e7)),
            alt=int(round(alt * 1_000.0)),
            eph=100,
            epv=100,
            vel=int(round(speed)),
            vn=int(round(vn)),
            ve=int(round(ve)),
            vd=int(round(vd)),
            cog=int(round(cog)),
            satellites_visible=10,
            id=0,
            yaw=0,
        )

    def _derive_gps_velocity(
        self, lat: float, lon: float, alt: float, time_s: float
    ) -> tuple[float, float, float, float, float]:
        if self._previous_gnss is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        prev_t, prev_lat, prev_lon, prev_alt = self._previous_gnss
        dt = time_s - prev_t
        if dt <= 1e-9:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        lat_r = math.radians(lat)
        prev_lat_r = math.radians(prev_lat)
        north_m = (lat_r - prev_lat_r) * _EARTH_RADIUS_M
        east_m = (
            (math.radians(lon) - math.radians(prev_lon))
            * math.cos(0.5 * (lat_r + prev_lat_r))
            * _EARTH_RADIUS_M
        )
        down_m = -(alt - prev_alt)

        vn = north_m / dt
        ve = east_m / dt
        vd = down_m / dt
        speed = math.sqrt(vn * vn + ve * ve + vd * vd)
        cog = (math.degrees(math.atan2(ve, vn)) + 360.0) % 360.0

        # Return in cm/s and centidegrees
        return vn * 100.0, ve * 100.0, vd * 100.0, speed * 100.0, cog * 100.0

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------

    def _pack(self, message: Any) -> bytes:
        self._buf.seek(0)
        self._buf.truncate(0)
        self._mav.send(message)
        data = self._buf.getvalue()
        self._buf.seek(0)
        self._buf.truncate(0)
        return data

    def _send(self, payload: bytes) -> None:
        try:
            assert self._socket is not None
            self._socket.sendto(payload, (self.host, self.port))
        except OSError as exc:
            log.warning("SitlMavlinkService UDP send failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers (same logic as MavlinkCommonAdapter)
# ---------------------------------------------------------------------------

def _resolve_reference_altitude(sensors_df: pd.DataFrame) -> float:
    if "gnss_z" not in sensors_df.columns:
        return 0.0
    valid = sensors_df["gnss_z"].dropna()
    return float(valid.iloc[0]) if not valid.empty else 0.0


def _resolve_sea_level_pressure(sensors_df: pd.DataFrame, ref_alt_m: float) -> float:
    if "barometer_v1" not in sensors_df.columns:
        return 101_325.0
    valid = sensors_df["barometer_v1"].dropna()
    if valid.empty:
        return 101_325.0
    return estimate_sea_level_pressure_pa(
        reference_pressure_pa=float(valid.iloc[0]),
        reference_altitude_m=ref_alt_m,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _safe_optional_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None
