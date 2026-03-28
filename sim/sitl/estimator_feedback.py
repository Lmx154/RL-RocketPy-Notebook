"""Typed inbound MAVLink feedback for serial SITL integrations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from typing import Mapping

DEFAULT_PAYLOAD_SERVO_TEST_COMMAND_ID = 31_000


@dataclass(frozen=True, slots=True)
class FeedbackOverlayEvent:
    """GUI-facing event derived from typed feedback."""

    category: str
    text: str
    source: str


@dataclass(frozen=True, slots=True)
class DeviceCommandDefinition:
    """Static mapping from a MAVLink command ID into a typed device event."""

    command_id: int
    event_type: str
    event_name: str
    category: str
    text: str


DEFAULT_COMMAND_EVENT_DEFINITIONS: dict[int, DeviceCommandDefinition] = {
    DEFAULT_PAYLOAD_SERVO_TEST_COMMAND_ID: DeviceCommandDefinition(
        command_id=DEFAULT_PAYLOAD_SERVO_TEST_COMMAND_ID,
        event_type="mavlink_command",
        event_name="payload_servo_test",
        category="Payload",
        text="5000FT altitude servo test",
    ),
}


@dataclass(frozen=True, slots=True)
class MavlinkFeedback:
    """Base type for structured inbound device feedback."""

    source_message_type: str

    def overlay_event(self) -> FeedbackOverlayEvent | None:
        return None


@dataclass(frozen=True, slots=True)
class EstimatorValueFeedback(MavlinkFeedback):
    """Named estimator value emitted by the device."""

    feedback_type: str
    metric_name: str
    value: float | int
    time_boot_ms: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_estimator_feedback_row(self, *, time_s: float | None = None) -> dict[str, Any]:
        row_time = None if time_s is None else float(time_s)
        return {
            "time_s": row_time,
            "feedback_type": self.feedback_type,
            "payload_json": json.dumps(self.payload, sort_keys=True),
        }


@dataclass(frozen=True, slots=True)
class DeviceStateEvent(MavlinkFeedback):
    """Discrete device event such as an inbound MAVLink command."""

    event_type: str
    event_name: str
    command_id: int | None = None
    category: str = "Avionics"
    text: str | None = None
    recognized: bool = False
    payload: dict[str, Any] = field(default_factory=dict)

    def overlay_event(self) -> FeedbackOverlayEvent | None:
        if not self.recognized or not self.text:
            return None
        return FeedbackOverlayEvent(
            category=self.category,
            text=self.text,
            source=self.source_message_type,
        )

    def to_device_event_row(self, *, time_s: float | None = None) -> dict[str, Any]:
        row_time = None if time_s is None else float(time_s)
        return {
            "time_s": row_time,
            "event_type": self.event_type,
            "event_name": self.event_name,
            "payload_json": json.dumps(self.payload, sort_keys=True),
        }


@dataclass(frozen=True, slots=True)
class DeviceLogEvent(MavlinkFeedback):
    """Structured device or SD/logging event."""

    event_type: str
    event_name: str
    severity: int
    text: str
    category: str = "Avionics"
    payload: dict[str, Any] = field(default_factory=dict)

    def overlay_event(self) -> FeedbackOverlayEvent | None:
        return FeedbackOverlayEvent(
            category=self.category,
            text=f"STATUSTEXT[{self.severity}] {self.text}",
            source=self.source_message_type,
        )

    def to_device_event_row(self, *, time_s: float | None = None) -> dict[str, Any]:
        row_time = None if time_s is None else float(time_s)
        return {
            "time_s": row_time,
            "event_type": self.event_type,
            "event_name": self.event_name,
            "payload_json": json.dumps(self.payload, sort_keys=True),
        }


def decode_mavlink_feedback(
    message: Any,
    *,
    command_definitions: Mapping[int, DeviceCommandDefinition] | None = None,
) -> list[MavlinkFeedback]:
    """Decode one raw MAVLink message into typed SITL feedback objects."""

    if not hasattr(message, "get_type"):
        return []

    msg_type = str(message.get_type())
    command_map = (
        dict(DEFAULT_COMMAND_EVENT_DEFINITIONS)
        if command_definitions is None
        else dict(command_definitions)
    )

    if msg_type in {"COMMAND_LONG", "COMMAND_INT"}:
        return [_decode_command_feedback(message, command_map)]
    if msg_type == "STATUSTEXT":
        return [_decode_statustext_feedback(message)]
    if msg_type in {"NAMED_VALUE_FLOAT", "NAMED_VALUE_INT"}:
        return [_decode_named_value_feedback(message)]
    return []


def format_feedback_log_line(feedback: MavlinkFeedback) -> str:
    """Format one typed feedback object for the transport log pane."""

    if isinstance(feedback, DeviceStateEvent):
        command_id = -1 if feedback.command_id is None else int(feedback.command_id)
        return (
            f"RX MAVLINK {feedback.source_message_type} "
            f"command={command_id}"
        )
    if isinstance(feedback, DeviceLogEvent):
        return (
            f"RX MAVLINK {feedback.source_message_type} "
            f"severity={feedback.severity} text={feedback.text}"
        )
    if isinstance(feedback, EstimatorValueFeedback):
        return (
            f"RX MAVLINK {feedback.source_message_type} "
            f"name={feedback.metric_name} value={feedback.value}"
        )
    return f"RX MAVLINK {feedback.source_message_type}"


def _decode_command_feedback(
    message: Any,
    command_definitions: Mapping[int, DeviceCommandDefinition],
) -> DeviceStateEvent:
    command_id = int(getattr(message, "command", -1))
    definition = command_definitions.get(command_id)
    event_name = f"mav_cmd_{command_id}" if definition is None else definition.event_name
    payload = _message_payload(message)
    payload["command_id"] = command_id
    return DeviceStateEvent(
        source_message_type=str(message.get_type()),
        event_type="mavlink_command" if definition is None else definition.event_type,
        event_name=event_name,
        command_id=command_id,
        category="Avionics" if definition is None else definition.category,
        text=None if definition is None else definition.text,
        recognized=definition is not None,
        payload=payload,
    )


def _decode_statustext_feedback(message: Any) -> DeviceLogEvent:
    text = _strip_mavlink_string(getattr(message, "text", ""))
    severity = int(getattr(message, "severity", 0))
    payload = _message_payload(message)
    payload["text"] = text
    return DeviceLogEvent(
        source_message_type="STATUSTEXT",
        event_type="statustext",
        event_name="STATUSTEXT",
        severity=severity,
        text=text,
        payload=payload,
    )


def _decode_named_value_feedback(message: Any) -> EstimatorValueFeedback:
    msg_type = str(message.get_type())
    metric_name = _strip_mavlink_string(getattr(message, "name", ""))
    value = getattr(message, "value", 0.0)
    if msg_type == "NAMED_VALUE_INT":
        value = int(value)
    else:
        value = float(value)
    payload = _message_payload(message)
    payload["metric_name"] = metric_name
    payload["value"] = value
    return EstimatorValueFeedback(
        source_message_type=msg_type,
        feedback_type=msg_type.lower(),
        metric_name=metric_name,
        value=value,
        time_boot_ms=_optional_int(getattr(message, "time_boot_ms", None)),
        payload=payload,
    )


def _message_payload(message: Any) -> dict[str, Any]:
    to_dict = getattr(message, "to_dict", None)
    if callable(to_dict):
        raw_payload = dict(to_dict())
    else:
        raw_payload = {
            key: getattr(message, key)
            for key in dir(message)
            if not key.startswith("_") and not callable(getattr(message, key))
        }

    raw_payload.pop("mavpackettype", None)
    return {
        str(key): _normalize_payload_value(value)
        for key, value in raw_payload.items()
        if value is not None
    }


def _normalize_payload_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip("\x00 ")
    if isinstance(value, str):
        return value.strip("\x00 ")
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_payload_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_payload_value(item)
            for key, item in value.items()
        }
    return str(value)


def _strip_mavlink_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip("\x00 ")
    return str(value or "").strip("\x00 ")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "DEFAULT_COMMAND_EVENT_DEFINITIONS",
    "DEFAULT_PAYLOAD_SERVO_TEST_COMMAND_ID",
    "DeviceCommandDefinition",
    "DeviceLogEvent",
    "DeviceStateEvent",
    "EstimatorValueFeedback",
    "FeedbackOverlayEvent",
    "MavlinkFeedback",
    "decode_mavlink_feedback",
    "format_feedback_log_line",
]
