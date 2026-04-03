"""Small reusable widgets for geometry and playback informatics summaries."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout

from sim.informatics import InformaticsPlaybackSource, RocketGeometryComponent


class _SummaryGroup(QGroupBox):
    """Shared summary group with a mono-spaced multi-line label."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        self.summary_label = QLabel()
        self.summary_label.setTextFormat(Qt.PlainText)
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-family: monospace; font-size: 9pt;")
        self.summary_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.summary_label)

    def _set_lines(self, lines: list[str]) -> None:
        self.summary_label.setText("\n".join(lines))


class GeometrySummaryWidget(_SummaryGroup):
    """Displays the currently loaded rocket geometry component."""

    def __init__(self, parent=None):
        super().__init__("Rocket Geometry Component", parent=parent)
        self.update_component(None)

    def update_component(self, component: RocketGeometryComponent | None) -> None:
        if component is None:
            self._set_lines(["No geometry component loaded."])
            return
        self._set_lines(component.summary_lines())


class PlaybackSourceSummaryWidget(_SummaryGroup):
    """Displays the currently loaded replay or sensor playback source."""

    def __init__(self, parent=None):
        super().__init__("Playback Data Source", parent=parent)
        self.update_source(None)

    def update_source(self, source: InformaticsPlaybackSource | None) -> None:
        if source is None:
            self._set_lines(
                [
                    "No playback source loaded.",
                    "Load kinematics plus virtual or real sensors to drive the viewer.",
                ]
            )
            return
        self._set_lines(source.summary_lines())
