"""Bottom-right HIL events overlay for the 3D replay viewer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGroupBox, QLabel, QVBoxLayout

HIL_EVENT_CATEGORIES = ("Payload", "Avionics", "Recovery")
_MAX_EVENTS_PER_CATEGORY = 6


@dataclass(frozen=True, slots=True)
class HilEventRecord:
    """Simple UI-facing HIL event record."""

    category: str
    text: str
    time_s: float | None = None
    source: str = "HIL"

    def format_line(self) -> str:
        prefix = f"t={self.time_s:.2f}s" if self.time_s is not None else self.source
        return f"{prefix}  {self.text}"


class _HilEventCategoryPanel(QGroupBox):
    """Single category panel showing the most recent HIL events."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(title, parent)
        self._events: deque[HilEventRecord] = deque(maxlen=_MAX_EVENTS_PER_CATEGORY)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(4)

        self._empty_label = QLabel("Awaiting HIL events")
        self._empty_label.setStyleSheet("color: #9ea7b3; font-style: italic;")
        layout.addWidget(self._empty_label)

        self._event_labels: list[QLabel] = []
        for _ in range(_MAX_EVENTS_PER_CATEGORY):
            label = QLabel("")
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            label.setStyleSheet("color: #f3f7fb; font-family: monospace; font-size: 10pt;")
            label.hide()
            layout.addWidget(label)
            self._event_labels.append(label)

    def clear_events(self) -> None:
        self._events.clear()
        self._refresh()

    def add_event(self, event: HilEventRecord) -> None:
        self._events.appendleft(event)
        self._refresh()

    def _refresh(self) -> None:
        has_events = bool(self._events)
        self._empty_label.setVisible(not has_events)
        for label, event in zip(self._event_labels, self._events, strict=False):
            label.setText(event.format_line())
            label.show()
        for label in self._event_labels[len(self._events):]:
            label.hide()
            label.setText("")


class HilEventOverlay(QFrame):
    """Compact overlay with fixed category panels for HIL events."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("hilEventOverlay")
        self.setStyleSheet(
            "#hilEventOverlay {"
            "background-color: rgba(12, 18, 28, 232);"
            "border: 1px solid #d7e3f4;"
            "border-radius: 10px;"
            "}"
            "#hilEventOverlay QLabel { color: #f3f7fb; }"
            "#hilEventOverlay QGroupBox {"
            "color: #dbe7f7;"
            "border: 1px solid #35506d;"
            "border-radius: 8px;"
            "margin-top: 10px;"
            "font-weight: bold;"
            "}"
            "#hilEventOverlay QGroupBox::title {"
            "subcontrol-origin: margin;"
            "left: 8px;"
            "padding: 0 4px;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        header = QLabel("HIL Events")
        header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(header)

        self._panels = {
            category: _HilEventCategoryPanel(category, self)
            for category in HIL_EVENT_CATEGORIES
        }
        for panel in self._panels.values():
            layout.addWidget(panel)

    def clear_events(self) -> None:
        for panel in self._panels.values():
            panel.clear_events()

    def add_event(
        self,
        category: str,
        text: str,
        *,
        time_s: float | None = None,
        source: str = "HIL",
    ) -> None:
        if category not in self._panels:
            raise ValueError(f"Unsupported HIL event category: {category!r}")
        self._panels[category].add_event(
            HilEventRecord(category=category, text=text, time_s=time_s, source=source)
        )
