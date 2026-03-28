from __future__ import annotations

import io
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from pymavlink.dialects.v20 import common as mavlink2

import sim.sitl.mavlink_sitl_service as mavlink_service_module
from sim.sitl.estimator_feedback import DeviceLogEvent, DeviceStateEvent
from sim.sitl.mavlink_sitl_service import SitlMavlinkService


class SitlMavlinkServiceTests(unittest.TestCase):
    def test_serial_service_emits_hil_packets_with_existing_logging(self) -> None:
        fake_serial_api = SimpleNamespace(
            Serial=_FakeSerial,
            SerialException=_FakeSerialException,
        )
        _FakeSerial.instances.clear()

        with patch.object(mavlink_service_module, "_pyserial", fake_serial_api):
            lines: list[str] = []
            service = SitlMavlinkService(_sample_sensors_frame())
            service.configure_serial(port="/dev/ttyUSB0", baudrate=115200)
            service.on_emit = lines.append

            service.start()
            service.emit_state(_sample_state())
            service.stop()

        messages = _decode_mavlink_packets(_FakeSerial.instances[0].written)

        self.assertEqual(
            [message.get_type() for message in messages],
            ["SYSTEM_TIME", "HIL_SENSOR", "HIL_GPS"],
        )
        self.assertTrue(lines)
        self.assertIn("serial:///dev/ttyUSB0 @ 115200 8N1", lines[0])
        self.assertIn("HIL_SENSOR", lines[0])
        self.assertIn("HIL_GPS", lines[0])
        queued_lines = service.drain_pending_log_lines()
        self.assertTrue(any(line.startswith("RX ") for line in queued_lines))
        self.assertTrue(any(line == "OPEN serial:///dev/ttyUSB0 @ 115200 8N1" for line in queued_lines))
        self.assertTrue(any(line == "CLOSE serial:///dev/ttyUSB0 @ 115200 8N1" for line in queued_lines))

    def test_serial_service_writes_packets_and_logs_rx_activity(self) -> None:
        fake_serial_api = SimpleNamespace(
            Serial=_FakeSerial,
            SerialException=_FakeSerialException,
        )
        _FakeSerial.instances.clear()

        with patch.object(mavlink_service_module, "_pyserial", fake_serial_api):
            lines: list[str] = []
            service = SitlMavlinkService(_sample_sensors_frame())
            service.configure_serial(
                port="/dev/ttyUSB0",
                baudrate=921600,
                bytesize=8,
                parity="N",
                stopbits=1.0,
                timeout_s=0.01,
            )
            service.on_emit = lines.append

            service.start()
            service.emit_state(_sample_state())

            queued_lines = _wait_for_pending_logs(service, expected_prefix="RX ", timeout_s=0.25)
            service.stop()
            queued_lines.extend(service.drain_pending_log_lines())

        self.assertTrue(_FakeSerial.instances)
        written_packets = _FakeSerial.instances[0].written
        messages = _decode_mavlink_packets(written_packets)

        self.assertEqual(
            [message.get_type() for message in messages],
            ["SYSTEM_TIME", "HIL_SENSOR", "HIL_GPS"],
        )
        self.assertTrue(lines)
        self.assertIn("serial:///dev/ttyUSB0 @ 921600 8N1", lines[0])
        self.assertTrue(any(line == "OPEN serial:///dev/ttyUSB0 @ 921600 8N1" for line in queued_lines))
        self.assertTrue(any(line.startswith("RX 3B fe 01 02") for line in queued_lines))
        self.assertTrue(any(line == "CLOSE serial:///dev/ttyUSB0 @ 921600 8N1" for line in queued_lines))

    def test_serial_start_fails_cleanly_when_pyserial_is_missing(self) -> None:
        with patch.object(mavlink_service_module, "_pyserial", None):
            service = SitlMavlinkService(_sample_sensors_frame())
            service.configure_serial(port="/dev/ttyUSB0", baudrate=115200)

            with self.assertRaisesRegex(RuntimeError, "pyserial"):
                service.start()

    def test_emit_state_only_sends_system_time_when_no_sensor_stream_is_fresh(self) -> None:
        fake_serial_api = SimpleNamespace(
            Serial=_FakeSerial,
            SerialException=_FakeSerialException,
        )
        _FakeSerial.instances.clear()

        with patch.object(mavlink_service_module, "_pyserial", fake_serial_api):
            service = SitlMavlinkService(_sample_sensors_frame())
            service.configure_serial(port="/dev/ttyUSB0", baudrate=115200)

            stale_state = _sample_state()
            stale_state["sensor_freshness"] = {
                "accelerometer_x": False,
                "barometer_v1": False,
                "gnss_x": False,
            }

            service.start()
            service.emit_state(stale_state)
            service.stop()

        messages = _decode_mavlink_packets(_FakeSerial.instances[0].written)
        self.assertEqual([message.get_type() for message in messages], ["SYSTEM_TIME"])

    def test_serial_service_queues_incoming_command_long_messages(self) -> None:
        fake_serial_api = SimpleNamespace(
            Serial=_FakeSerial,
            SerialException=_FakeSerialException,
        )
        _FakeSerial.instances.clear()
        _FakeSerial.next_reads = [_build_command_long_payload(command=31_000)]

        with patch.object(mavlink_service_module, "_pyserial", fake_serial_api):
            service = SitlMavlinkService(_sample_sensors_frame())
            service.configure_serial(
                port="/dev/ttyUSB0",
                baudrate=115200,
                timeout_s=0.01,
            )

            service.start()
            incoming = _wait_for_pending_feedback(
                service,
                expected_feedback_type=DeviceStateEvent,
                timeout_s=0.25,
            )
            queued_lines = service.drain_pending_log_lines()
            service.stop()

        self.assertTrue(incoming)
        self.assertIsInstance(incoming[0], DeviceStateEvent)
        self.assertEqual(incoming[0].source_message_type, "COMMAND_LONG")
        self.assertEqual(int(incoming[0].command_id or -1), 31_000)
        overlay_event = incoming[0].overlay_event()
        self.assertIsNotNone(overlay_event)
        assert overlay_event is not None
        self.assertEqual(overlay_event.category, "Payload")
        self.assertEqual(overlay_event.text, "5000FT altitude servo test")
        self.assertTrue(
            any(line == "RX MAVLINK COMMAND_LONG command=31000" for line in queued_lines),
        )

    def test_serial_service_queues_incoming_statustext_feedback(self) -> None:
        fake_serial_api = SimpleNamespace(
            Serial=_FakeSerial,
            SerialException=_FakeSerialException,
        )
        _FakeSerial.instances.clear()
        _FakeSerial.next_reads = [_build_statustext_payload(severity=4, text="sd logger armed")]

        with patch.object(mavlink_service_module, "_pyserial", fake_serial_api):
            service = SitlMavlinkService(_sample_sensors_frame())
            service.configure_serial(
                port="/dev/ttyUSB0",
                baudrate=115200,
                timeout_s=0.01,
            )

            service.start()
            incoming = _wait_for_pending_feedback(
                service,
                expected_feedback_type=DeviceLogEvent,
                timeout_s=0.25,
            )
            queued_lines = service.drain_pending_log_lines()
            service.stop()

        self.assertTrue(incoming)
        self.assertIsInstance(incoming[0], DeviceLogEvent)
        self.assertEqual(incoming[0].event_type, "statustext")
        self.assertEqual(incoming[0].text, "sd logger armed")
        overlay_event = incoming[0].overlay_event()
        self.assertIsNotNone(overlay_event)
        assert overlay_event is not None
        self.assertEqual(overlay_event.text, "STATUSTEXT[4] sd logger armed")
        self.assertTrue(
            any(line == "RX MAVLINK STATUSTEXT severity=4 text=sd logger armed" for line in queued_lines),
        )


class _FakeSerialException(Exception):
    pass


class _FakeSerial:
    instances: list["_FakeSerial"] = []
    next_reads: list[bytes] | None = None

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        bytesize: int,
        parity: str,
        stopbits: float,
        timeout: float,
        write_timeout: float,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.written: list[bytes] = []
        self._reads = list(self.__class__.next_reads or [b"\xfe\x01\x02"])
        self.__class__.next_reads = None
        self.is_open = True
        self.__class__.instances.append(self)

    @property
    def in_waiting(self) -> int:
        return len(self._reads[0]) if self._reads else 0

    def read(self, size: int = 1) -> bytes:
        del size
        if self._reads:
            return self._reads.pop(0)
        time.sleep(0.01)
        return b""

    def write(self, payload: bytes) -> int:
        self.written.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.is_open = False


def _sample_sensors_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0],
            "accelerometer_x": [0.1],
            "accelerometer_y": [0.2],
            "accelerometer_z": [-9.7],
            "gyroscope_x": [0.01],
            "gyroscope_y": [0.02],
            "gyroscope_z": [0.03],
            "barometer_v1": [96453.7],
            "gnss_x": [33.4986538],
            "gnss_y": [-99.3375871],
            "gnss_z": [413.9],
        }
    )


def _sample_state() -> dict[str, object]:
    return {
        "time": 0.25,
        "sensors": {
            "accelerometer_x": 0.1,
            "accelerometer_y": 0.2,
            "accelerometer_z": -9.7,
            "gyroscope_x": 0.01,
            "gyroscope_y": 0.02,
            "gyroscope_z": 0.03,
            "barometer_v1": 96453.7,
            "gnss_x": 33.4986538,
            "gnss_y": -99.3375871,
            "gnss_z": 413.9,
        },
        "sensor_freshness": {
            "accelerometer_x": True,
            "barometer_v1": True,
            "gnss_x": True,
        },
    }


def _wait_for_pending_logs(
    service: SitlMavlinkService,
    *,
    expected_prefix: str,
    timeout_s: float,
) -> list[str]:
    deadline = time.time() + timeout_s
    queued_lines: list[str] = []
    while time.time() < deadline:
        new_lines = service.drain_pending_log_lines()
        queued_lines.extend(new_lines)
        if any(line.startswith(expected_prefix) for line in queued_lines):
            return queued_lines
        time.sleep(0.01)
    return queued_lines


def _wait_for_pending_feedback(
    service: SitlMavlinkService,
    *,
    expected_feedback_type: type,
    timeout_s: float,
) -> list:
    deadline = time.time() + timeout_s
    queued_feedback = []
    while time.time() < deadline:
        new_feedback = service.drain_pending_feedback()
        queued_feedback.extend(new_feedback)
        if any(
            isinstance(feedback, expected_feedback_type)
            for feedback in queued_feedback
        ):
            return queued_feedback
        time.sleep(0.01)
    return queued_feedback


def _decode_mavlink_packets(payloads: list[bytes]) -> list:
    parser = mavlink2.MAVLink(io.BytesIO())
    messages = []
    for payload in payloads:
        for byte in payload:
            message = parser.parse_char(bytes([byte]))
            if message is not None:
                messages.append(message)
    return messages


def _build_command_long_payload(command: int) -> bytes:
    buffer = io.BytesIO()
    mav = mavlink2.MAVLink(buffer)
    mav.srcSystem = 42
    mav.srcComponent = 7
    mav.send(
        mavlink2.MAVLink_command_long_message(
            target_system=1,
            target_component=1,
            command=int(command),
            confirmation=0,
            param1=0.0,
            param2=0.0,
            param3=0.0,
            param4=0.0,
            param5=0.0,
            param6=0.0,
            param7=0.0,
        )
    )
    return buffer.getvalue()


def _build_statustext_payload(*, severity: int, text: str) -> bytes:
    buffer = io.BytesIO()
    mav = mavlink2.MAVLink(buffer)
    mav.srcSystem = 42
    mav.srcComponent = 7
    mav.send(
        mavlink2.MAVLink_statustext_message(
            severity=int(severity),
            text=text.encode("utf-8"),
            id=0,
            chunk_seq=0,
        )
    )
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
