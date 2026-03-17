from __future__ import annotations

import io
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from pymavlink.dialects.v20 import common as mavlink2

import sim.sitl.mavlink_sitl_service as mavlink_service_module
from sim.sitl.mavlink_sitl_service import SitlMavlinkService


class SitlMavlinkServiceTests(unittest.TestCase):
    def test_udp_service_keeps_existing_emit_behavior(self) -> None:
        fake_socket = _FakeSocket()
        with patch.object(mavlink_service_module.socket, "socket", return_value=fake_socket):
            lines: list[str] = []
            service = SitlMavlinkService(_sample_sensors_frame())
            service.on_emit = lines.append

            service.start()
            service.emit_state(_sample_state())
            service.stop()

        messages = _decode_mavlink_packets([payload for payload, _target in fake_socket.sent])

        self.assertEqual(
            [message.get_type() for message in messages],
            ["SYSTEM_TIME", "HIL_SENSOR", "HIL_GPS"],
        )
        self.assertEqual(fake_socket.sent[0][1], ("127.0.0.1", 14560))
        self.assertTrue(lines)
        self.assertIn("udp://127.0.0.1:14560", lines[0])
        self.assertIn("HIL_SENSOR", lines[0])
        self.assertIn("HIL_GPS", lines[0])
        self.assertEqual(
            service.drain_pending_log_lines(),
            ["OPEN udp://127.0.0.1:14560", "CLOSE udp://127.0.0.1:14560"],
        )

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


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def sendto(self, payload: bytes, target: tuple[str, int]) -> None:
        self.sent.append((payload, target))

    def close(self) -> None:
        self.closed = True


class _FakeSerialException(Exception):
    pass


class _FakeSerial:
    instances: list["_FakeSerial"] = []

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
        self._reads = [b"\xfe\x01\x02"]
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


def _decode_mavlink_packets(payloads: list[bytes]) -> list:
    parser = mavlink2.MAVLink(io.BytesIO())
    messages = []
    for payload in payloads:
        for byte in payload:
            message = parser.parse_char(bytes([byte]))
            if message is not None:
                messages.append(message)
    return messages


if __name__ == "__main__":
    unittest.main()
