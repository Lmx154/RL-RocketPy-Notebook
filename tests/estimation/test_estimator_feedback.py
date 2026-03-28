from __future__ import annotations

import io
import json
import unittest

from pymavlink.dialects.v20 import common as mavlink2

from sim.sitl.estimator_feedback import (
    DEFAULT_PAYLOAD_SERVO_TEST_COMMAND_ID,
    DeviceLogEvent,
    DeviceStateEvent,
    EstimatorValueFeedback,
    decode_mavlink_feedback,
    format_feedback_log_line,
)


class EstimatorFeedbackTests(unittest.TestCase):
    def test_decode_command_long_uses_typed_device_event_definition(self) -> None:
        message = _decode_single_message(
            _build_command_long_payload(command=DEFAULT_PAYLOAD_SERVO_TEST_COMMAND_ID)
        )

        feedback = decode_mavlink_feedback(message)

        self.assertEqual(len(feedback), 1)
        self.assertIsInstance(feedback[0], DeviceStateEvent)
        self.assertTrue(feedback[0].recognized)
        self.assertEqual(feedback[0].event_name, "payload_servo_test")
        overlay_event = feedback[0].overlay_event()
        self.assertIsNotNone(overlay_event)
        assert overlay_event is not None
        self.assertEqual(overlay_event.category, "Payload")
        self.assertEqual(overlay_event.text, "5000FT altitude servo test")
        self.assertEqual(
            format_feedback_log_line(feedback[0]),
            "RX MAVLINK COMMAND_LONG command=31000",
        )

        device_row = feedback[0].to_device_event_row(time_s=12.5)
        self.assertEqual(device_row["time_s"], 12.5)
        self.assertEqual(device_row["event_type"], "mavlink_command")
        self.assertEqual(device_row["event_name"], "payload_servo_test")
        payload = json.loads(device_row["payload_json"])
        self.assertEqual(payload["command_id"], DEFAULT_PAYLOAD_SERVO_TEST_COMMAND_ID)

    def test_decode_unknown_command_keeps_structured_event_without_overlay_mapping(self) -> None:
        message = _decode_single_message(_build_command_long_payload(command=42_001))

        feedback = decode_mavlink_feedback(message)

        self.assertEqual(len(feedback), 1)
        self.assertIsInstance(feedback[0], DeviceStateEvent)
        self.assertFalse(feedback[0].recognized)
        self.assertIsNone(feedback[0].overlay_event())
        self.assertEqual(feedback[0].event_name, "mav_cmd_42001")

    def test_decode_statustext_returns_device_log_event(self) -> None:
        message = _decode_single_message(
            _build_statustext_payload(severity=4, text="sd logger armed")
        )

        feedback = decode_mavlink_feedback(message)

        self.assertEqual(len(feedback), 1)
        self.assertIsInstance(feedback[0], DeviceLogEvent)
        overlay_event = feedback[0].overlay_event()
        self.assertIsNotNone(overlay_event)
        assert overlay_event is not None
        self.assertEqual(overlay_event.category, "Avionics")
        self.assertEqual(overlay_event.text, "STATUSTEXT[4] sd logger armed")
        self.assertEqual(
            format_feedback_log_line(feedback[0]),
            "RX MAVLINK STATUSTEXT severity=4 text=sd logger armed",
        )

        device_row = feedback[0].to_device_event_row(time_s=1.0)
        self.assertEqual(device_row["event_type"], "statustext")
        payload = json.loads(device_row["payload_json"])
        self.assertEqual(payload["severity"], 4)
        self.assertEqual(payload["text"], "sd logger armed")

    def test_decode_named_value_float_returns_typed_estimator_feedback(self) -> None:
        message = _decode_single_message(
            _build_named_value_float_payload(
                time_boot_ms=250,
                name="eskf_pos_z",
                value=412.25,
            )
        )

        feedback = decode_mavlink_feedback(message)

        self.assertEqual(len(feedback), 1)
        self.assertIsInstance(feedback[0], EstimatorValueFeedback)
        self.assertEqual(feedback[0].feedback_type, "named_value_float")
        self.assertEqual(feedback[0].metric_name, "eskf_pos_z")
        self.assertEqual(feedback[0].time_boot_ms, 250)
        self.assertAlmostEqual(float(feedback[0].value), 412.25)
        self.assertEqual(
            format_feedback_log_line(feedback[0]),
            "RX MAVLINK NAMED_VALUE_FLOAT name=eskf_pos_z value=412.25",
        )

        estimator_row = feedback[0].to_estimator_feedback_row(time_s=2.5)
        self.assertEqual(estimator_row["time_s"], 2.5)
        self.assertEqual(estimator_row["feedback_type"], "named_value_float")
        payload = json.loads(estimator_row["payload_json"])
        self.assertEqual(payload["metric_name"], "eskf_pos_z")
        self.assertAlmostEqual(payload["value"], 412.25)


def _decode_single_message(payload: bytes):
    parser = mavlink2.MAVLink(io.BytesIO())
    decoded = None
    for byte in payload:
        decoded = parser.parse_char(bytes([byte]))
        if decoded is not None:
            return decoded
    raise AssertionError("Expected one MAVLink message")


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


def _build_named_value_float_payload(*, time_boot_ms: int, name: str, value: float) -> bytes:
    buffer = io.BytesIO()
    mav = mavlink2.MAVLink(buffer)
    mav.srcSystem = 42
    mav.srcComponent = 7
    mav.send(
        mavlink2.MAVLink_named_value_float_message(
            time_boot_ms=int(time_boot_ms),
            name=name.encode("utf-8"),
            value=float(value),
        )
    )
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
