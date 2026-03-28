from __future__ import annotations

import io
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pymavlink.dialects.v20 import common as mavlink2

import sim.sitl.mavlink_sitl_service as mavlink_service_module
from sim.sitl.mavlink_codec import ALL_STANDARD_HIL_SENSOR_FIELDS, MavlinkHilCodec
from sim.sitl.mavlink_sitl_service import SitlMavlinkService
from sim.sitl.session import ReplaySession


class MavlinkHilCodecTests(unittest.TestCase):
    def test_codec_builds_standard_hil_messages(self) -> None:
        frame = _sample_frame(time_s=0.25)
        codec = MavlinkHilCodec.from_telemetry(frame, unix_epoch_base_usec=1_000_000)
        row = frame.iloc[0].to_dict()

        payloads = [
            codec.pack(codec.system_time_message(0.25)),
            codec.pack(
                codec.hil_sensor_message(
                    0.25,
                    row,
                    fields_updated=ALL_STANDARD_HIL_SENSOR_FIELDS,
                )
            ),
        ]
        gps_message = codec.hil_gps_message(0.25, row)
        assert gps_message is not None
        payloads.append(codec.pack(gps_message))

        messages = _decode_mavlink_packets(payloads)

        self.assertEqual(
            [message.get_type() for message in messages],
            ["SYSTEM_TIME", "HIL_SENSOR", "HIL_GPS"],
        )
        self.assertEqual(messages[0].time_unix_usec, 1_250_000)
        self.assertEqual(messages[0].time_boot_ms, 250)
        self.assertAlmostEqual(messages[1].xacc, 0.1, places=6)
        self.assertAlmostEqual(messages[1].zgyro, 0.03, places=6)
        self.assertEqual(messages[1].fields_updated, ALL_STANDARD_HIL_SENSOR_FIELDS)
        self.assertEqual(messages[2].fix_type, 3)
        self.assertEqual(messages[2].lat, int(round(33.4986538 * 1e7)))

    def test_codec_reset_runtime_state_clears_previous_gps_velocity(self) -> None:
        frame = pd.DataFrame(
            {
                "time_s": [0.0, 1.0],
                "barometer_v1": [96453.7, 96453.7],
                "gnss_x": [33.4986538, 33.4987538],
                "gnss_y": [-99.3375871, -99.3374871],
                "gnss_z": [413.9, 414.9],
            }
        )
        codec = MavlinkHilCodec.from_telemetry(frame)

        first = codec.hil_gps_message(0.0, frame.iloc[0].to_dict())
        second = codec.hil_gps_message(1.0, frame.iloc[1].to_dict())
        codec.reset_runtime_state()
        after_reset = codec.hil_gps_message(1.0, frame.iloc[1].to_dict())

        assert first is not None and second is not None and after_reset is not None
        self.assertEqual(first.vel, 0)
        self.assertGreater(second.vel, 0)
        self.assertEqual(after_reset.vel, 0)

    def test_codec_can_initialize_directly_from_replay_session(self) -> None:
        frame = _sample_frame(time_s=0.25)
        session = _sample_replay_session(frame)

        codec = MavlinkHilCodec.from_replay_session(
            session,
            unix_epoch_base_usec=1_000_000,
        )

        row = frame.iloc[0].to_dict()
        gps_message = codec.hil_gps_message(0.25, row)
        assert gps_message is not None
        payloads = [
            codec.pack(codec.system_time_message(0.25)),
            codec.pack(
                codec.hil_sensor_message(
                    0.25,
                    row,
                    fields_updated=ALL_STANDARD_HIL_SENSOR_FIELDS,
                )
            ),
            codec.pack(gps_message),
        ]
        messages = _decode_mavlink_packets(payloads)

        self.assertEqual(messages[0].time_unix_usec, 1_250_000)
        self.assertAlmostEqual(messages[1].abs_pressure, 964.537, places=3)
        self.assertEqual(messages[2].lat, int(round(33.4986538 * 1e7)))

    def test_service_and_codec_share_identical_hil_encoding(self) -> None:
        frame = _sample_frame(time_s=0.25)
        fake_serial_api = type(
            "FakeSerialApi",
            (),
            {"Serial": _FakeSerial, "SerialException": _FakeSerialException},
        )
        _FakeSerial.instances.clear()

        with patch.object(mavlink_service_module, "_pyserial", fake_serial_api):
            service = SitlMavlinkService(frame)
            service.configure_serial(port="/dev/ttyUSB0", baudrate=115200)
            service.start()
            service.emit_state(_sample_state(time_s=0.25))
            service.stop()

        codec = MavlinkHilCodec.from_telemetry(frame)
        row = frame.iloc[0].to_dict()
        gps_message = codec.hil_gps_message(0.25, row)
        assert gps_message is not None
        expected_payloads = [
            codec.pack(codec.system_time_message(0.25)),
            codec.pack(
                codec.hil_sensor_message(
                    0.25,
                    row,
                    fields_updated=ALL_STANDARD_HIL_SENSOR_FIELDS,
                )
            ),
            codec.pack(gps_message),
        ]

        self.assertEqual(
            _FakeSerial.instances[0].written,
            expected_payloads,
        )

    def test_service_accepts_replay_session_telemetry_source(self) -> None:
        frame = _sample_frame(time_s=0.25)
        session = _sample_replay_session(frame)
        fake_serial_api = type(
            "FakeSerialApi",
            (),
            {"Serial": _FakeSerial, "SerialException": _FakeSerialException},
        )
        _FakeSerial.instances.clear()

        with patch.object(mavlink_service_module, "_pyserial", fake_serial_api):
            service = SitlMavlinkService(session)
            service.configure_serial(port="/dev/ttyUSB0", baudrate=115200)
            service.start()
            service.emit_state(_sample_state(time_s=0.25))
            service.stop()

        codec = MavlinkHilCodec.from_replay_session(session)
        row = frame.iloc[0].to_dict()
        gps_message = codec.hil_gps_message(0.25, row)
        assert gps_message is not None
        expected_payloads = [
            codec.pack(codec.system_time_message(0.25)),
            codec.pack(
                codec.hil_sensor_message(
                    0.25,
                    row,
                    fields_updated=ALL_STANDARD_HIL_SENSOR_FIELDS,
                )
            ),
            codec.pack(gps_message),
        ]

        self.assertEqual(_FakeSerial.instances[0].written, expected_payloads)


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
        self.is_open = True
        self.__class__.instances.append(self)

    @property
    def in_waiting(self) -> int:
        return 0

    def read(self, size: int = 1) -> bytes:
        del size
        time.sleep(0.01)
        return b""

    def write(self, payload: bytes) -> int:
        self.written.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _sample_frame(*, time_s: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [time_s],
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


def _sample_state(*, time_s: float) -> dict[str, object]:
    return {
        "time": time_s,
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


def _sample_replay_session(frame: pd.DataFrame) -> ReplaySession:
    truth = pd.DataFrame(
        {
            "time_s": [float(frame.iloc[0]["time_s"])],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [417.0],
            "vx_mps": [10.0],
            "vy_mps": [0.0],
            "vz_mps": [20.0],
            "e0": [1.0],
            "e1": [0.0],
            "e2": [0.0],
            "e3": [0.0],
            "w1_radps": [0.1],
            "w2_radps": [0.2],
            "w3_radps": [0.3],
        }
    )
    imu = frame[
        [
            "time_s",
            "accelerometer_x",
            "accelerometer_y",
            "accelerometer_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z",
        ]
    ].copy()
    baro = frame[["time_s", "barometer_v1"]].copy()
    gps = frame[["time_s", "gnss_x", "gnss_y", "gnss_z"]].copy()
    stream_frames = {
        "truth": truth,
        "imu": imu,
        "baro": baro,
        "gps": gps,
    }
    return ReplaySession(
        session_dir=Path("."),
        manifest_path=Path("manifest.json"),
        manifest={},
        stream_paths={},
        stream_frames=stream_frames,
        truth=truth,
        imu=imu,
        baro=baro,
        gps=gps,
        mag=None,
    )


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
