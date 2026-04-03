"""Data-centric informatics panel with source loading, stats, and plots."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from sim.informatics import InformaticsPlaybackSource

from .informatics_summary import PlaybackSourceSummaryWidget

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except Exception:
    FigureCanvas = None
    Figure = None
    _MATPLOTLIB_AVAILABLE = False


class DataInformaticsPanel(QWidget):
    """Primary data workspace for loading sources and inspecting plots/stats."""

    load_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_source: InformaticsPlaybackSource | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Use this workspace to load a playback source and inspect kinematics, "
            "virtual sensors, or real sensor streams with summary statistics and plots."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        load_group = QGroupBox("Data Source")
        load_layout = QVBoxLayout(load_group)
        self.load_button = QPushButton("Load Playback Source")
        self.load_button.setMinimumHeight(42)
        self.load_button.clicked.connect(self.load_requested.emit)
        load_layout.addWidget(self.load_button)

        self.source_hint_label = QLabel(
            "Current loader expects a replay session manifest or session folder."
        )
        self.source_hint_label.setWordWrap(True)
        self.source_hint_label.setStyleSheet("color: #666666;")
        load_layout.addWidget(self.source_hint_label)
        layout.addWidget(load_group)

        self.source_summary_widget = PlaybackSourceSummaryWidget()
        layout.addWidget(self.source_summary_widget)

        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_display = QPlainTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMaximumBlockCount(100)
        self.stats_display.setStyleSheet("font-family: monospace; font-size: 9pt;")
        self.stats_display.setMinimumHeight(140)
        stats_layout.addWidget(self.stats_display)
        layout.addWidget(stats_group)

        plots_group = QGroupBox("Plots")
        plots_layout = QVBoxLayout(plots_group)
        if _MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(10, 12))
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            scroll = QScrollArea()
            scroll.setWidgetResizable(False)
            scroll.setWidget(self.canvas)
            scroll.setMinimumHeight(520)
            self.plots_scroll_area = scroll
            plots_layout.addWidget(scroll)
        else:
            self.figure = None
            self.canvas = None
            self.plots_scroll_area = None
            missing = QLabel(
                "Matplotlib is not available in the current environment. "
                "Install project dependencies to render plots in this tab."
            )
            missing.setWordWrap(True)
            missing.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            plots_layout.addWidget(missing)
        layout.addWidget(plots_group, stretch=1)

        self.update_source(None)

    def update_source(self, source: InformaticsPlaybackSource | None) -> None:
        self._current_source = source
        self.source_summary_widget.update_source(source)

        if source is None:
            self.stats_display.setPlainText(
                "No source loaded.\n"
                "Load a replay session to inspect the kinematics and sensor streams."
            )
            self._clear_plots()
            return

        self.stats_display.setPlainText("\n".join(source.stats_lines()))
        self._render_plots(source)

    def _clear_plots(self) -> None:
        if self.figure is None or self.canvas is None:
            return
        self.figure.clear()
        self.canvas.draw_idle()

    def _render_plots(self, source: InformaticsPlaybackSource) -> None:
        if self.figure is None or self.canvas is None:
            return

        plot_specs = self._build_plot_specs(source)
        self.figure.clear()
        if not plot_specs:
            self._resize_canvas_for_plot_count(0)
            self.canvas.draw_idle()
            return

        axes = self.figure.subplots(len(plot_specs), 1, squeeze=False)
        for axis, (title, frame, time_col, y_cols) in zip(axes.flatten(), plot_specs):
            time_values = frame[time_col].to_numpy(dtype=float)
            for column in y_cols:
                axis.plot(time_values, frame[column].to_numpy(dtype=float), label=column)
            axis.set_title(title)
            axis.set_xlabel(time_col)
            axis.grid(True, alpha=0.3)
            if len(y_cols) > 1:
                axis.legend(loc="best", fontsize=8)

        self.figure.tight_layout()
        self._resize_canvas_for_plot_count(len(plot_specs))
        self.canvas.draw_idle()

    def _resize_canvas_for_plot_count(self, plot_count: int) -> None:
        """Resize the matplotlib canvas so plots stack vertically and scroll."""
        if self.figure is None or self.canvas is None:
            return

        effective_count = max(1, plot_count)
        figure_height = max(12.0, 3.6 * effective_count)
        self.figure.set_size_inches(10.0, figure_height, forward=True)
        dpi = float(self.figure.dpi)
        self.canvas.setMinimumHeight(int(figure_height * dpi))
        self.canvas.resize(self.canvas.sizeHint().width(), int(figure_height * dpi))

    def _build_plot_specs(
        self,
        source: InformaticsPlaybackSource,
    ) -> list[tuple[str, pd.DataFrame, str, list[str]]]:
        specs: list[tuple[str, pd.DataFrame, str, list[str]]] = []
        acceleration_plot: tuple[str, pd.DataFrame, str, list[str]] | None = None
        truth = source.kinematics_frame

        if "time_s" in truth.columns:
            if "z_m" in truth.columns:
                specs.append(("Altitude vs Time", truth, "time_s", ["z_m"]))
            velocity_columns = [
                column
                for column in ("vx_mps", "vy_mps", "vz_mps")
                if column in truth.columns
            ]
            if velocity_columns:
                specs.append(("Velocity Components", truth, "time_s", velocity_columns))

        for stream in source.sensor_streams:
            frame = source.sensor_frames.get(stream.key)
            if frame is None or frame.empty or "time_s" not in frame.columns:
                continue
            acceleration_columns = self._acceleration_columns(frame.columns)
            if acceleration_columns and acceleration_plot is None:
                acceleration_plot = (
                    "Acceleration Across All Axes",
                    frame,
                    "time_s",
                    acceleration_columns,
                )

            y_cols = self._generic_plot_columns(frame.columns)
            if y_cols:
                specs.append((f"{stream.display_name} Signals", frame, "time_s", y_cols[:3]))
            if len(specs) >= 4:
                break

        if acceleration_plot is not None:
            specs.append(acceleration_plot)

        return specs[:5]

    @staticmethod
    def _acceleration_columns(columns: Iterable[str]) -> list[str]:
        column_list = list(columns)
        candidates = ["accelerometer_x", "accelerometer_y", "accelerometer_z"]
        if all(column in column_list for column in candidates):
            return candidates
        return []

    @classmethod
    def _generic_plot_columns(cls, columns: Iterable[str]) -> list[str]:
        excluded = {"time_s"}
        acceleration_columns = set(cls._acceleration_columns(columns))
        preferred = [
            column
            for column in columns
            if column not in excluded and column not in acceleration_columns
        ]
        return preferred
