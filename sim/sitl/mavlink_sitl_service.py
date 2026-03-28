"""Rate-aware MAVLink SITL service for GUI step-through integration.

Emits MAVLink HIL_SENSOR / HIL_GPS packets over USB serial, gated by
per-sensor freshness flags so each sensor only fires at its own data rate.

Usage::

    service = SitlMavlinkService(replay_session)
    service.configure_serial(port="/dev/ttyUSB0", baudrate=115200)
    service.start()
    # ... per GUI step:
    service.emit_state(state)
    service.stop()
"""

from __future__ import annotations

import io
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pymavlink.dialects.v20 import common as mavlink2

from .estimator_feedback import (
    DEFAULT_COMMAND_EVENT_DEFINITIONS,
    DeviceCommandDefinition,
    MavlinkFeedback,
    decode_mavlink_feedback,
    format_feedback_log_line,
)
from .mavlink_codec import (
    BARO_UPDATED_FIELDS,
    IMU_UPDATED_FIELDS,
    MavlinkHilCodec,
    finite_float,
)
from .session import ReplaySession

try:
    import serial as _pyserial
    from serial.tools import list_ports as _serial_list_ports
except ImportError:  # pragma: no cover - exercised through explicit fallback tests
    _pyserial = None
    _serial_list_ports = None

log = logging.getLogger(__name__)

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

    telemetry_source: ReplaySession | pd.DataFrame
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
    feedback_command_definitions: dict[int, DeviceCommandDefinition] = field(
        default_factory=lambda: dict(DEFAULT_COMMAND_EVENT_DEFINITIONS),
        repr=False,
    )

    # private state — not part of the public API
    _active: bool = field(default=False, init=False, repr=False)
    _transport_handle: _SerialTransport | None = field(
        default=None, init=False, repr=False
    )
    _codec: MavlinkHilCodec | None = field(default=None, init=False, repr=False)
    _pending_log_lines: deque[str] = field(default_factory=deque, init=False, repr=False)
    _pending_feedback: deque[MavlinkFeedback] = field(
        default_factory=deque,
        init=False,
        repr=False,
    )
    _log_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _feedback_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._codec = self._build_codec()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def endpoint_description(self) -> str:
        return _format_serial_endpoint(
            port=self.serial_port,
            baudrate=self.serial_baudrate,
            bytesize=self.serial_bytesize,
            parity=self.serial_parity,
            stopbits=self.serial_stopbits,
        )

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
        self.serial_port = port.strip()
        self.serial_baudrate = int(baudrate)
        self.serial_bytesize = int(bytesize)
        self.serial_parity = str(parity).upper()
        self.serial_stopbits = float(stopbits)
        self.serial_timeout_s = float(timeout_s)

    def start(
        self,
        *,
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

        self._codec = self._build_codec()
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

    def drain_pending_feedback(self) -> list[MavlinkFeedback]:
        """Return typed inbound device feedback queued off the GUI thread."""
        with self._feedback_lock:
            feedback = list(self._pending_feedback)
            self._pending_feedback.clear()
        return feedback

    def drain_pending_incoming_messages(self) -> list[MavlinkFeedback]:
        """Compatibility alias for callers not yet renamed to feedback terminology."""
        return self.drain_pending_feedback()

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
        assert self._codec is not None

        # Always emit SYSTEM_TIME so the FC tracks sim wall-clock
        self._send(self._codec.pack(self._codec.system_time_message(time_s)))

        imu_fresh = bool(freshness.get("accelerometer_x", False))
        baro_fresh = bool(freshness.get("barometer_v1", False))
        fields_updated = (
            (IMU_UPDATED_FIELDS if imu_fresh else 0)
            | (BARO_UPDATED_FIELDS if baro_fresh else 0)
        )

        emitted_sensor = False
        if fields_updated:
            self._send(
                self._codec.pack(
                    self._codec.hil_sensor_message(
                        time_s,
                        sensors,
                        fields_updated=fields_updated,
                    )
                )
            )
            emitted_sensor = True

        gps_fresh = bool(freshness.get("gnss_x", False))
        emitted_gps = False
        if gps_fresh:
            gps_msg = self._codec.hil_gps_message(time_s, sensors)
            if gps_msg is not None:
                self._send(self._codec.pack(gps_msg))
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
                ax = finite_float(sensors.get("accelerometer_x"))
                ay = finite_float(sensors.get("accelerometer_y"))
                az = finite_float(sensors.get("accelerometer_z"))
                gx = finite_float(sensors.get("gyroscope_x"))
                gy = finite_float(sensors.get("gyroscope_y"))
                gz = finite_float(sensors.get("gyroscope_z"))
                flags.append(
                    f"IMU acc=({ax:.3f},{ay:.3f},{az:.3f}) "
                    f"gyro=({gx:.3f},{gy:.3f},{gz:.3f}) m/s^2,rad/s"
                )
            if baro_fresh:
                assert self._codec is not None
                pressure_pa = finite_float(
                    sensors.get("barometer_v1"),
                    default=self._codec.sea_level_pressure_pa,
                )
                flags.append(f"BARO {pressure_pa:.1f} Pa")
            parts.append(
                "HIL_SENSOR ["
                + "  ".join(flags)
                + f"]  fields=0x{fields_updated:04x}"
            )

        if emitted_gps:
            lat = finite_float(sensors.get("gnss_x"))
            lon = finite_float(sensors.get("gnss_y"))
            alt = finite_float(sensors.get("gnss_z"))
            parts.append(f"HIL_GPS lat={lat:.6f} deg lon={lon:.6f} deg alt={alt:.1f}m")

        return "  |  ".join(parts)

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
        decoded_feedback = decode_mavlink_feedback(
            message,
            command_definitions=self.feedback_command_definitions,
        )
        if not decoded_feedback:
            self._queue_log_line(_format_incoming_mavlink_line(message))
            return

        with self._feedback_lock:
            self._pending_feedback.extend(decoded_feedback)
        for feedback in decoded_feedback:
            self._queue_log_line(format_feedback_log_line(feedback))

    def _build_codec(self) -> MavlinkHilCodec:
        if isinstance(self.telemetry_source, ReplaySession):
            return MavlinkHilCodec.from_replay_session(
                self.telemetry_source,
                system_id=self.system_id,
                component_id=self.component_id,
                unix_epoch_base_usec=self.unix_epoch_base_usec,
            )
        return MavlinkHilCodec.from_telemetry(
            self.telemetry_source,
            system_id=self.system_id,
            component_id=self.component_id,
            unix_epoch_base_usec=self.unix_epoch_base_usec,
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
