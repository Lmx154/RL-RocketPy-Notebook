"""Rate-aware MAVLink SITL service for GUI step-through integration.

Emits MAVLink HIL_SENSOR / HIL_GPS packets over either UDP or USB serial,
gated by per-sensor freshness flags so each sensor only fires at its own data
rate. This mirrors real hardware behaviour more closely for both SITL and HIL.

Usage::

    service = SitlMavlinkService(sensors_df)
    service.start()  # defaults to udp://127.0.0.1:14560
    # ... per GUI step:
    service.emit_state(state)
    service.stop()
"""

from __future__ import annotations

import io
import logging
import math
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pymavlink.dialects.v20 import common as mavlink2

from sim.estimation.adapters.rocketpy_replay import (
    estimate_sea_level_pressure_pa,
    pressure_to_altitude_m,
)

try:
    import serial as _pyserial
    from serial.tools import list_ports as _serial_list_ports
except ImportError:  # pragma: no cover - exercised through explicit fallback tests
    _pyserial = None
    _serial_list_ports = None

log = logging.getLogger(__name__)

TransportMode = Literal["udp", "serial"]

# ---------------------------------------------------------------------------
# MAVLink HIL_SENSOR fields_updated bitmask constants
# See: https://mavlink.io/en/messages/common.html#HIL_SENSOR_UPDATED_FLAGS
# ---------------------------------------------------------------------------
_IMU_FIELDS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
_BARO_FIELDS = (1 << 9) | (1 << 11)  # abs_pressure + pressure_alt
_EARTH_RADIUS_M = 6_378_137.0
_SERIAL_READ_IDLE_SLEEP_S = 0.01
_SERIAL_RX_PREVIEW_BYTES = 24


@dataclass(frozen=True, slots=True)
class SerialPortInfo:
    """Small, GUI-friendly serial port descriptor."""

    device: str
    description: str = ""


def serial_support_available() -> bool:
    """Return whether pyserial is importable in the current environment."""
    return _pyserial is not None


def list_serial_ports() -> list[SerialPortInfo]:
    """Enumerate available serial ports for the GUI port picker."""
    if _serial_list_ports is None:
        return []

    ports: list[SerialPortInfo] = []
    for port in _serial_list_ports.comports():
        device = str(getattr(port, "device", "") or "").strip()
        if not device:
            continue
        description = str(getattr(port, "description", "") or "").strip()
        ports.append(SerialPortInfo(device=device, description=description))
    return sorted(ports, key=lambda item: item.device)


class _UdpTransport:
    def __init__(self, host: str, port: int) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._target = (host, port)

    def send(self, payload: bytes) -> None:
        self._socket.sendto(payload, self._target)

    def close(self) -> None:
        self._socket.close()


class _SerialTransport:
    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        bytesize: int,
        parity: str,
        stopbits: float,
        timeout_s: float,
        log_line: Callable[[str], None] | None,
        message_handler: Callable[[Any], None] | None,
    ) -> None:
        if _pyserial is None:
            raise RuntimeError(
                "USB serial MAVLink requires pyserial. Add the dependency and sync the environment."
            )

        device = port.strip()
        if not device:
            raise ValueError("Select a USB serial port before enabling MAVLink.")

        self._timeout_s = max(0.0, float(timeout_s))
        self._log_line = log_line
        self._message_handler = message_handler
        self._stop_event = threading.Event()
        self._parser = mavlink2.MAVLink(io.BytesIO())
        self._serial = _pyserial.Serial(
            port=device,
            baudrate=int(baudrate),
            bytesize=int(bytesize),
            parity=str(parity).upper(),
            stopbits=float(stopbits),
            timeout=self._timeout_s,
            write_timeout=max(self._timeout_s, 0.1),
        )
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name="SitlMavlinkSerialReader",
            daemon=True,
        )
        self._reader_thread.start()

    def send(self, payload: bytes) -> None:
        self._serial.write(payload)
        flush = getattr(self._serial, "flush", None)
        if callable(flush):
            flush()

    def close(self) -> None:
        self._stop_event.set()
        close = getattr(self._serial, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - defensive close path
                log.debug("Ignoring serial close failure", exc_info=True)
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=max(0.2, (2.0 * self._timeout_s) + 0.1))

    def _read_loop(self) -> None:
        serial_exception = _get_serial_exception_type()
        while not self._stop_event.is_set():
            try:
                waiting = int(getattr(self._serial, "in_waiting", 0) or 0)
                chunk = self._serial.read(waiting or 1)
            except serial_exception as exc:
                if not self._stop_event.is_set() and self._log_line is not None:
                    self._log_line(f"SERIAL RX ERROR {exc}")
                return
            except Exception as exc:  # pragma: no cover - defensive path
                if not self._stop_event.is_set() and self._log_line is not None:
                    self._log_line(f"SERIAL RX ERROR {exc}")
                return

            if chunk:
                if self._log_line is not None:
                    self._log_line(_format_serial_rx_line(bytes(chunk)))
                self._handle_incoming_chunk(bytes(chunk))
            elif waiting == 0 and self._timeout_s <= 0.0:
                time.sleep(_SERIAL_READ_IDLE_SLEEP_S)

    def _handle_incoming_chunk(self, payload: bytes) -> None:
        if self._message_handler is None:
            return
        for byte in payload:
            try:
                message = self._parser.parse_char(bytes([byte]))
            except Exception:  # pragma: no cover - defensive parser fallback
                log.debug("Ignoring MAVLink RX parse failure", exc_info=True)
                return
            if message is not None:
                self._message_handler(message)


@dataclass
class SitlMavlinkService:
    """Rate-aware MAVLink SITL emitter for GUI step-through replay."""

    sensors_df: pd.DataFrame
    transport: TransportMode = "udp"
    host: str = "127.0.0.1"
    port: int = 14560
    serial_port: str = ""
    serial_baudrate: int = 115200
    serial_bytesize: int = 8
    serial_parity: str = "N"
    serial_stopbits: float = 1.0
    serial_timeout_s: float = 0.02
    system_id: int = 1
    component_id: int = 1
    unix_epoch_base_usec: int = 0
    # Optional callback invoked after each emit_state() call with a human-readable
    # summary string. The GUI uses this for the existing green terminal output.
    on_emit: Callable[[str], None] | None = field(default=None, repr=False)

    # private state — not part of the public API
    _active: bool = field(default=False, init=False, repr=False)
    _transport_handle: _UdpTransport | _SerialTransport | None = field(
        default=None, init=False, repr=False
    )
    _mav: Any = field(default=None, init=False, repr=False)
    _buf: io.BytesIO = field(default_factory=io.BytesIO, init=False, repr=False)
    _reference_altitude_m: float = field(default=0.0, init=False, repr=False)
    _sea_level_pressure_pa: float = field(default=101_325.0, init=False, repr=False)
    _previous_gnss: tuple[float, float, float, float] | None = field(
        default=None, init=False, repr=False
    )
    _pending_log_lines: deque[str] = field(default_factory=deque, init=False, repr=False)
    _pending_incoming_messages: deque[Any] = field(default_factory=deque, init=False, repr=False)
    _log_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _incoming_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._reference_altitude_m = _resolve_reference_altitude(self.sensors_df)
        self._sea_level_pressure_pa = _resolve_sea_level_pressure(
            self.sensors_df, self._reference_altitude_m
        )

    @property
    def active(self) -> bool:
        return self._active

    @property
    def endpoint_description(self) -> str:
        if self.transport == "serial":
            return _format_serial_endpoint(
                port=self.serial_port,
                baudrate=self.serial_baudrate,
                bytesize=self.serial_bytesize,
                parity=self.serial_parity,
                stopbits=self.serial_stopbits,
            )
        return f"udp://{self.host}:{int(self.port)}"

    def configure_udp(self, *, host: str, port: int) -> None:
        self.transport = "udp"
        self.host = host.strip() or "127.0.0.1"
        self.port = int(port)

    def configure_serial(
        self,
        *,
        port: str,
        baudrate: int,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: float = 1.0,
        timeout_s: float = 0.02,
    ) -> None:
        self.transport = "serial"
        self.serial_port = port.strip()
        self.serial_baudrate = int(baudrate)
        self.serial_bytesize = int(bytesize)
        self.serial_parity = str(parity).upper()
        self.serial_stopbits = float(stopbits)
        self.serial_timeout_s = float(timeout_s)

    def start(
        self,
        host: str | None = None,
        port: int | None = None,
        *,
        transport: TransportMode | None = None,
        serial_port: str | None = None,
        serial_baudrate: int | None = None,
        serial_bytesize: int | None = None,
        serial_parity: str | None = None,
        serial_stopbits: float | None = None,
        serial_timeout_s: float | None = None,
    ) -> None:
        """Open the selected transport and begin accepting emit_state() calls."""
        if self._active:
            self.stop()

        if transport is not None:
            self.transport = transport
        if host is not None:
            self.host = host
        if port is not None:
            self.port = int(port)
        if serial_port is not None:
            self.serial_port = serial_port
        if serial_baudrate is not None:
            self.serial_baudrate = int(serial_baudrate)
        if serial_bytesize is not None:
            self.serial_bytesize = int(serial_bytesize)
        if serial_parity is not None:
            self.serial_parity = str(serial_parity).upper()
        if serial_stopbits is not None:
            self.serial_stopbits = float(serial_stopbits)
        if serial_timeout_s is not None:
            self.serial_timeout_s = float(serial_timeout_s)

        self._buf = io.BytesIO()
        self._mav = mavlink2.MAVLink(self._buf)
        self._mav.srcSystem = int(self.system_id)
        self._mav.srcComponent = int(self.component_id)
        self._previous_gnss = None

        if self.transport == "udp":
            self._transport_handle = _UdpTransport(self.host, int(self.port))
        elif self.transport == "serial":
            self._transport_handle = _SerialTransport(
                port=self.serial_port,
                baudrate=self.serial_baudrate,
                bytesize=self.serial_bytesize,
                parity=self.serial_parity,
                stopbits=self.serial_stopbits,
                timeout_s=self.serial_timeout_s,
                log_line=self._queue_log_line,
                message_handler=self._queue_incoming_message,
            )
        else:  # pragma: no cover - guarded by GUI controls and type hints
            raise ValueError(f"Unsupported MAVLink transport: {self.transport!r}")

        self._active = True
        self._queue_log_line(f"OPEN {self.endpoint_description}")
        log.info("SitlMavlinkService started → %s", self.endpoint_description)

    def stop(self) -> None:
        """Disable emission and close the active transport."""
        if not self._active and self._transport_handle is None:
            return

        description = self.endpoint_description
        self._active = False
        transport = self._transport_handle
        self._transport_handle = None
        if transport is not None:
            transport.close()
        self._queue_log_line(f"CLOSE {description}")
        log.info("SitlMavlinkService stopped")

    def drain_pending_log_lines(self) -> list[str]:
        """Return queued transport/system log lines generated off the GUI thread."""
        with self._log_lock:
            lines = list(self._pending_log_lines)
            self._pending_log_lines.clear()
        return lines

    def drain_pending_incoming_messages(self) -> list[Any]:
        """Return decoded inbound MAVLink messages queued off the GUI thread."""
        with self._incoming_lock:
            messages = list(self._pending_incoming_messages)
            self._pending_incoming_messages.clear()
        return messages

    def emit_state(self, state: dict[str, Any]) -> None:
        """Emit MAVLink packets for the given simulation state.

        Uses sensor_freshness to gate each sensor type so only sensors
        with new data at this step generate packets — matching real hardware
        sample rates naturally.
        """
        if not self._active or self._transport_handle is None:
            return

        sensors: dict[str, float | None] = state.get("sensors", {})
        freshness: dict[str, bool] = state.get("sensor_freshness", {})
        time_s: float = float(state.get("time", 0.0))

        # Always emit SYSTEM_TIME so the FC tracks sim wall-clock
        self._send(self._pack(self._system_time_msg(time_s)))

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
            self.on_emit(
                self._format_emit_line(
                    time_s,
                    sensors,
                    imu_fresh,
                    baro_fresh,
                    emitted_sensor,
                    fields_updated,
                    emitted_gps,
                )
            )

    def _format_emit_line(
        self,
        time_s: float,
        sensors: dict[str, float | None],
        imu_fresh: bool,
        baro_fresh: bool,
        emitted_sensor: bool,
        fields_updated: int,
        emitted_gps: bool,
    ) -> str:
        parts = [f"t={time_s:.4f}s  {self.endpoint_description}  SYSTEM_TIME"]

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
                    f"gyro=({gx:.3f},{gy:.3f},{gz:.3f}) m/s^2,rad/s"
                )
            if baro_fresh:
                pressure_pa = _safe_float(
                    sensors.get("barometer_v1"),
                    self._sea_level_pressure_pa,
                )
                flags.append(f"BARO {pressure_pa:.1f} Pa")
            parts.append(
                "HIL_SENSOR ["
                + "  ".join(flags)
                + f"]  fields=0x{fields_updated:04x}"
            )

        if emitted_gps:
            lat = _safe_float(sensors.get("gnss_x"))
            lon = _safe_float(sensors.get("gnss_y"))
            alt = _safe_float(sensors.get("gnss_z"))
            parts.append(f"HIL_GPS lat={lat:.6f} deg lon={lon:.6f} deg alt={alt:.1f}m")

        return "  |  ".join(parts)

    def _system_time_msg(self, time_s: float) -> Any:
        time_usec = self.unix_epoch_base_usec + int(round(time_s * 1_000_000.0))
        return mavlink2.MAVLink_system_time_message(
            time_unix_usec=time_usec,
            time_boot_ms=int(round(time_s * 1_000.0)),
        )

    def _hil_sensor_msg(
        self,
        time_s: float,
        sensors: dict[str, float | None],
        fields_updated: int,
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
            abs_pressure=float(pressure_pa / 100.0),  # Pa -> hPa (mbar)
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
        self,
        lat: float,
        lon: float,
        alt: float,
        time_s: float,
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
            assert self._transport_handle is not None
            self._transport_handle.send(payload)
        except Exception as exc:
            log.warning("SitlMavlinkService send failed (%s): %s", self.endpoint_description, exc)
            self._queue_log_line(f"TX ERROR {self.endpoint_description}: {exc}")

    def _queue_log_line(self, line: str) -> None:
        with self._log_lock:
            self._pending_log_lines.append(line)

    def _queue_incoming_message(self, message: Any) -> None:
        with self._incoming_lock:
            self._pending_incoming_messages.append(message)
        self._queue_log_line(_format_incoming_mavlink_line(message))


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


def _format_serial_endpoint(
    *,
    port: str,
    baudrate: int,
    bytesize: int,
    parity: str,
    stopbits: float,
) -> str:
    port_label = port.strip() or "<select-port>"
    stopbits_label = _format_stopbits(stopbits)
    return (
        f"serial://{port_label} @ {int(baudrate)} "
        f"{int(bytesize)}{str(parity).upper()}{stopbits_label}"
    )


def _format_stopbits(stopbits: float) -> str:
    rounded = float(stopbits)
    if math.isclose(rounded, round(rounded)):
        return str(int(round(rounded)))
    return f"{rounded:g}"


def _format_serial_rx_line(data: bytes) -> str:
    preview = " ".join(f"{byte:02x}" for byte in data[:_SERIAL_RX_PREVIEW_BYTES])
    if len(data) > _SERIAL_RX_PREVIEW_BYTES:
        preview += " ..."
    return f"RX {len(data)}B {preview}"


def _format_incoming_mavlink_line(message: Any) -> str:
    msg_type = str(message.get_type())
    if msg_type in {"COMMAND_LONG", "COMMAND_INT"}:
        command_id = int(getattr(message, "command", -1))
        return f"RX MAVLINK {msg_type} command={command_id}"
    if msg_type == "STATUSTEXT":
        text = str(getattr(message, "text", "") or "").strip("\x00 ")
        severity = int(getattr(message, "severity", 0))
        return f"RX MAVLINK STATUSTEXT severity={severity} text={text}"
    return f"RX MAVLINK {msg_type}"


def _get_serial_exception_type() -> type[Exception]:
    if _pyserial is None:
        return OSError
    return getattr(_pyserial, "SerialException", OSError)


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
