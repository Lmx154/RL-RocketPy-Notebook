"""
Rocket 3D Viewer GUI Application

A dedicated cross-platform GUI for visualizing RocketPy rockets in 3D.
Built with PySide6 and PyVista for maximum compatibility.

Features:
- Interactive 3D rendering with PyVista
- Component selection for visualization
- **NEW: Live flight simulation mode with Flight class data**
- Export models to STL/OBJ formats
- Mesh statistics display
- Real-time telemetry and attitude visualization

Usage:
    from sim.gui import RocketViewerApp
    
    app = RocketViewerApp(rocket=my_rocket)
    app.run()
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QCheckBox, QTextEdit, QSplitter,
    QFileDialog, QMessageBox, QFrame, QRadioButton, QSlider,
    QScrollArea
)
from PySide6.QtCore import Qt, Slot, QEvent
from pyvistaqt import QtInteractor
import pyvista as pv

from ..rendering.renderer import RocketRenderer
from ..simulation import CsvReplayController, load_replay_pair, quaternion_to_matrix
from ..simulation.quaternion_utils import interpolate_quaternion
from .telemetry_display import TelemetryDisplay

# Suppress VTK warnings that appear during PyVista shutdown
# These are harmless cleanup warnings from VTK internal objects
pv.set_error_output_file('NUL' if sys.platform == 'win32' else '/dev/null')
logging.getLogger('vtkmodules').setLevel(logging.CRITICAL)


class RocketViewerApp(QMainWindow):
    """
    Main GUI application for 3D rocket visualization.
    
    Provides an interactive interface for:
    - Selecting rocket components to render
    - Viewing 3D models in embedded PyVista viewer
    - Exporting models to file
    - Viewing mesh statistics
    """
    
    def __init__(self, rocket):
        """
        Initialize the Rocket Viewer application.
        
        Args:
            rocket: RocketPy Rocket object to load
        """
        super().__init__()
        
        self.rocket = rocket
        self.rocket_renderer = RocketRenderer(self.rocket)
        
        # Component selection state
        self.component_checkboxes = {}
        self.selected_components = set()
        
        # Simulation state
        self.mode = 'static'  # 'static' or 'simulation'
        self.simulation_controller = None
        self.replay_sensor_path = None
        self.replay_kinematics_path = None
        self.rail_length = 5.1816  # Default, will be updated from config
        
        # Original rocket mesh (cached for static mode)
        self.original_rocket_mesh = None
        
        # Simulation rendering state
        self.rocket_actors = {}  # Store PyVista actors for updating
        self.trail_actor = None
        self.trail_polydata = None  # Store the trajectory polydata for incremental updates
        self.ground_actor = None
        self.sky_grid_actor = None
        
        # Trajectory trail - store as numpy array for efficiency
        self.trajectory_points = []
        self.last_trajectory_point = None  # Track last point to draw segments from
        
        # Interpolation state for smooth rendering
        self.last_state = None
        self.current_display_state = None
        self.interpolation_alpha = 0.0
        
        # Camera tracking control
        self.auto_tracking_enabled = True
        self.last_camera_position = None
        self.user_is_interacting = False
        self.replay_overlay_visible = True
        
        # Setup UI
        self.init_ui()
        
        # Render initial view
        self.render_selected_components()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Rocket 3D Viewer')
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel: Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: 3D Viewer
        right_panel = self.create_viewer_panel()
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (30% controls, 70% viewer)
        splitter.setSizes([420, 980])
        
        # Status bar
        self.statusBar().showMessage('Ready')
    
    def create_control_panel(self) -> QWidget:
        """Create the left control panel with simulation and component controls."""
        # Make panel scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMaximumWidth(500)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel('<h2>Rocket 3D Viewer</h2>')
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Mode selection
        mode_group = self.create_mode_selection()
        layout.addWidget(mode_group)
        
        # Simulation controls (only shown in simulation mode)
        self.sim_controls_group = self.create_simulation_controls()
        layout.addWidget(self.sim_controls_group)
        self.sim_controls_group.setVisible(False)
        
        # Object info section
        info_group = QGroupBox('Rocket Information')
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel(f'<b>Radius:</b> {self.rocket.radius:.4f} m'))
        info_layout.addWidget(QLabel(f'<b>Mass:</b> {self.rocket.mass:.2f} kg'))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Component selection
        component_group = self.create_component_selection()
        layout.addWidget(component_group)
        
        # Action buttons (static mode)
        self.static_button_group = QGroupBox('Actions')
        button_layout = QVBoxLayout()
        
        # Render button
        render_btn = QPushButton('🔄 Render Selected')
        render_btn.setMinimumHeight(40)
        render_btn.clicked.connect(self.on_render_clicked)
        button_layout.addWidget(render_btn)
        
        # Select All / None buttons
        select_layout = QHBoxLayout()
        select_all_btn = QPushButton('Select All')
        select_all_btn.clicked.connect(self.select_all_components)
        select_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton('Select None')
        select_none_btn.clicked.connect(self.select_no_components)
        select_layout.addWidget(select_none_btn)
        button_layout.addLayout(select_layout)
        
        # Export button
        export_btn = QPushButton('💾 Export to STL')
        export_btn.clicked.connect(self.on_export_clicked)
        button_layout.addWidget(export_btn)
        
        # Clear button
        clear_btn = QPushButton('🗑️ Clear Viewer')
        clear_btn.clicked.connect(self.clear_viewer)
        button_layout.addWidget(clear_btn)
        
        self.static_button_group.setLayout(button_layout)
        layout.addWidget(self.static_button_group)
        
        # Telemetry display (only in simulation mode)
        self.telemetry_widget = TelemetryDisplay()
        layout.addWidget(self.telemetry_widget)
        self.telemetry_widget.setVisible(False)
        
        # Mesh info display (static mode)
        self.mesh_info_group = QGroupBox('Mesh Information')
        info_display_layout = QVBoxLayout()
        
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setMaximumHeight(150)
        self.info_display.setStyleSheet('font-family: monospace; font-size: 9pt;')
        info_display_layout.addWidget(self.info_display)
        self.mesh_info_group.setLayout(info_display_layout)
        layout.addWidget(self.mesh_info_group)
        
        # Spacer
        layout.addStretch()
        
        # Help text
        help_text = QLabel(
            '<small><b>Controls:</b><br>'
            '• Left click + drag: Rotate<br>'
            '• Right click + drag: Pan<br>'
            '• Scroll: Zoom<br>'
            '• R: Reset camera</small>'
        )
        help_text.setWordWrap(True)
        help_text.setFrameStyle(QFrame.Box | QFrame.Plain)
        help_text.setMargin(10)
        layout.addWidget(help_text)
        
        scroll.setWidget(panel)
        return scroll
    
    def create_mode_selection(self) -> QGroupBox:
        """Create mode selection radio buttons."""
        group = QGroupBox('Visualization Mode')
        layout = QVBoxLayout()
        
        self.static_mode_radio = QRadioButton('Static Model')
        self.static_mode_radio.setChecked(True)
        self.static_mode_radio.toggled.connect(self.on_mode_changed)
        layout.addWidget(self.static_mode_radio)
        
        self.sim_mode_radio = QRadioButton('3D Simulation')
        self.sim_mode_radio.toggled.connect(self.on_mode_changed)
        layout.addWidget(self.sim_mode_radio)
        
        group.setLayout(layout)
        return group
    
    def create_simulation_controls(self) -> QGroupBox:
        """Create simulation control buttons and controls."""
        group = QGroupBox('Simulation Controls')
        layout = QVBoxLayout()
        
        # Replay loading
        self.load_replay_btn = QPushButton('📂 Load Replay CSV Pair')
        self.load_replay_btn.setMinimumHeight(45)
        self.load_replay_btn.setStyleSheet('font-weight: bold; font-size: 11pt;')
        self.load_replay_btn.clicked.connect(self.on_load_replay_clicked)
        layout.addWidget(self.load_replay_btn)

        self.replay_source_label = QLabel('Replay source: not loaded')
        self.replay_source_label.setWordWrap(True)
        layout.addWidget(self.replay_source_label)
        
        # Auto-tracking toggle button
        self.auto_tracking_btn = QPushButton('📹 Auto-Track: ON')
        self.auto_tracking_btn.setCheckable(True)
        self.auto_tracking_btn.setChecked(True)
        self.auto_tracking_btn.clicked.connect(self.on_auto_tracking_toggled)
        self.auto_tracking_btn.setStyleSheet('font-weight: bold;')
        layout.addWidget(self.auto_tracking_btn)

        self.overlay_toggle_btn = QPushButton('Hide Replay Overlay')
        self.overlay_toggle_btn.clicked.connect(self.on_overlay_toggle_clicked)
        layout.addWidget(self.overlay_toggle_btn)
        
        group.setLayout(layout)
        return group
    
    def create_component_selection(self) -> QGroupBox:
        """Create component selection checkboxes for rocket."""
        group = QGroupBox('Select Components')
        layout = QVBoxLayout()
        
        # Get available components
        geometry = self.rocket_renderer.extract_geometry()
        
        # Add checkboxes for each component
        components = [
            ('Motor', 'motor', geometry.motor is not None),
            ('Nose Cone', 'nosecone', geometry.nosecone is not None),
            ('Body Tube', 'body', True),  # Always has body
            ('Fins', 'fins', geometry.fins is not None),
            ('Tail', 'tail', geometry.tail is not None),
        ]
        
        for display_name, component_id, available in components:
            if available:
                checkbox = QCheckBox(display_name)
                checkbox.setChecked(True)  # Default: all selected
                # Store component_id as property to avoid lambda closure issues
                checkbox.setProperty('component_id', component_id)
                checkbox.stateChanged.connect(self.on_component_checkbox_changed)
                self.component_checkboxes[component_id] = checkbox
                self.selected_components.add(component_id)
                layout.addWidget(checkbox)
        
        group.setLayout(layout)
        return group
    
    def create_viewer_panel(self) -> QWidget:
        """Create the right panel with 3D viewer."""
        panel = QWidget()
        self.viewer_panel = panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create PyVista QtInteractor (embedded viewer)
        multi_samples = int(os.environ.get('ROCKET_VIEWER_MULTI_SAMPLES', '0'))
        self.plotter = QtInteractor(panel, multi_samples=multi_samples)
        self.plotter.set_background('white')
        layout.addWidget(self.plotter.interactor)
        
        # Add axes and camera widget
        self.plotter.add_axes(
            xlabel='X (m)',
            ylabel='Y (m)',
            zlabel='Z (m)',
        )
        self.plotter.add_camera_orientation_widget()
        
        # Connect interaction events to detect user camera control
        self.plotter.iren.add_observer('StartInteractionEvent', self._on_user_interaction_start)
        self.plotter.iren.add_observer('EndInteractionEvent', self._on_user_interaction_end)

        self._create_replay_overlay(panel)
        panel.installEventFilter(self)
        self._position_replay_overlay_widgets()
        
        return panel

    def _create_replay_overlay(self, parent: QWidget) -> None:
        """Create semi-transparent replay controls overlay on top of the 3D viewer."""
        self.replay_overlay = QFrame(parent)
        self.replay_overlay.setObjectName('replayOverlay')
        self.replay_overlay.setStyleSheet(
            '#replayOverlay {'
            'background-color: rgba(24, 24, 24, 165);'
            'border: 1px solid rgba(230, 230, 230, 90);'
            'border-radius: 10px;'
            '}'
            '#replayOverlay QLabel { color: #f0f0f0; }'
            '#replayOverlay QPushButton { background-color: rgba(255, 255, 255, 40); color: #f6f6f6; }'
            '#replayOverlay QPushButton:disabled { color: rgba(246, 246, 246, 110); }'
        )

        overlay_layout = QVBoxLayout(self.replay_overlay)
        overlay_layout.setContentsMargins(10, 8, 10, 8)
        overlay_layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_label = QLabel('Replay Controls')
        header_label.setStyleSheet('font-weight: bold; color: #f0f0f0;')
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        self.hide_overlay_btn = QPushButton('Hide')
        self.hide_overlay_btn.clicked.connect(lambda: self.set_replay_overlay_visibility(False))
        header_layout.addWidget(self.hide_overlay_btn)
        overlay_layout.addLayout(header_layout)

        playback_layout = QHBoxLayout()
        self.prev_step_btn = QPushButton('◀ Prev')
        self.prev_step_btn.clicked.connect(self.on_prev_step_clicked)
        self.prev_step_btn.setEnabled(False)
        playback_layout.addWidget(self.prev_step_btn)

        self.start_btn = QPushButton('▶ Start')
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.start_btn.setEnabled(False)
        playback_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton('⏸ Pause')
        self.pause_btn.clicked.connect(self.on_pause_clicked)
        self.pause_btn.setEnabled(False)
        playback_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton('⏹ Stop')
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.stop_btn.setEnabled(False)
        playback_layout.addWidget(self.stop_btn)

        self.next_step_btn = QPushButton('Next ▶')
        self.next_step_btn.clicked.connect(self.on_next_step_clicked)
        self.next_step_btn.setEnabled(False)
        playback_layout.addWidget(self.next_step_btn)
        overlay_layout.addLayout(playback_layout)

        speed_layout = QHBoxLayout()
        speed_caption = QLabel('Speed')
        speed_caption.setMinimumWidth(48)
        speed_layout.addWidget(speed_caption)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)  # 0.1x
        self.speed_slider.setMaximum(100)  # 10x
        self.speed_slider.setValue(10)  # 1.0x
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_slider)

        self.speed_label = QLabel('1.0x')
        self.speed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.speed_label.setMinimumWidth(48)
        speed_layout.addWidget(self.speed_label)
        overlay_layout.addLayout(speed_layout)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setSingleStep(1)
        self.timeline_slider.setPageStep(100)
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderPressed.connect(self.on_timeline_pressed)
        self.timeline_slider.sliderReleased.connect(self.on_timeline_released)
        self.timeline_slider.valueChanged.connect(self.on_timeline_changed)
        self.timeline_slider.setEnabled(False)
        overlay_layout.addWidget(self.timeline_slider)

        self.timeline_label = QLabel('Step: 0 / 0 | Time: 0.0 s / 0.0 s')
        self.timeline_label.setAlignment(Qt.AlignCenter)
        overlay_layout.addWidget(self.timeline_label)

        self.show_overlay_btn = QPushButton('Show Replay Overlay', parent)
        self.show_overlay_btn.clicked.connect(lambda: self.set_replay_overlay_visibility(True))
        self.show_overlay_btn.setVisible(False)
        self.show_overlay_btn.setStyleSheet('background-color: rgba(24, 24, 24, 165); color: #f0f0f0;')

        self.set_replay_overlay_visibility(True)

    def _position_replay_overlay_widgets(self) -> None:
        """Position overlay widgets relative to current 3D panel size."""
        if not hasattr(self, 'viewer_panel') or not hasattr(self, 'replay_overlay'):
            return

        panel_width = max(0, self.viewer_panel.width())
        margin = 14

        overlay_width = max(430, min(760, panel_width - (2 * margin)))
        self.replay_overlay.setFixedWidth(overlay_width)
        self.replay_overlay.adjustSize()
        self.replay_overlay.move(margin, margin)

        self.show_overlay_btn.adjustSize()
        self.show_overlay_btn.move(
            max(margin, panel_width - self.show_overlay_btn.width() - margin),
            margin,
        )

    def set_replay_overlay_visibility(self, visible: bool) -> None:
        """Set replay overlay visibility while preserving simulation-mode gating."""
        self.replay_overlay_visible = bool(visible)
        is_sim_mode = self.mode == 'simulation'
        self.replay_overlay.setVisible(is_sim_mode and self.replay_overlay_visible)
        self.show_overlay_btn.setVisible(is_sim_mode and (not self.replay_overlay_visible))
        if hasattr(self, 'overlay_toggle_btn'):
            self.overlay_toggle_btn.setText('Hide Replay Overlay' if self.replay_overlay_visible else 'Show Replay Overlay')
        self._position_replay_overlay_widgets()

    def on_overlay_toggle_clicked(self) -> None:
        """Toggle replay overlay visibility from the left-side controls panel."""
        self.set_replay_overlay_visibility(not self.replay_overlay_visible)

    def eventFilter(self, watched, event):
        if watched is getattr(self, 'viewer_panel', None) and event.type() == QEvent.Resize:
            self._position_replay_overlay_widgets()
        return super().eventFilter(watched, event)
    
    def on_component_checkbox_changed(self, state: int):
        """Handle component checkbox toggle."""
        checkbox = self.sender()
        if checkbox is None:
            return
        
        component_id = checkbox.property('component_id')
        if component_id is None:
            return
        
        if checkbox.isChecked():
            self.selected_components.add(component_id)
        else:
            self.selected_components.discard(component_id)
    
    def select_all_components(self):
        """Select all available components."""
        for checkbox in self.component_checkboxes.values():
            checkbox.setChecked(True)
    
    def select_no_components(self):
        """Deselect all components."""
        for checkbox in self.component_checkboxes.values():
            checkbox.setChecked(False)
    
    def _on_user_interaction_start(self, obj, event):
        """Called when user starts interacting with the camera (mouse drag, scroll, etc.)."""
        self.user_is_interacting = True
        # DON'T disable auto-tracking - let it keep running
        # This way camera keeps following rocket even while you zoom/pan
    
    def _on_user_interaction_end(self, obj, event):
        """Called when user stops interacting with the camera."""
        self.user_is_interacting = False
    
    def toggle_auto_tracking(self):
        """Toggle automatic camera tracking on/off."""
        self.auto_tracking_enabled = not self.auto_tracking_enabled
        if self.auto_tracking_enabled:
            self.statusBar().showMessage('Auto-tracking enabled')
        else:
            self.statusBar().showMessage('Auto-tracking disabled - manual camera control')
    
    def on_auto_tracking_toggled(self):
        """Handle auto-tracking button toggle."""
        self.auto_tracking_enabled = self.auto_tracking_btn.isChecked()
        self.user_is_interacting = False  # Clear interaction flag when manually toggling
        
        if self.auto_tracking_enabled:
            self.auto_tracking_btn.setText('📹 Auto-Track: ON')
            self.auto_tracking_btn.setStyleSheet('font-weight: bold; background-color: lightgreen;')
            self.statusBar().showMessage('Auto-tracking enabled - camera follows rocket')
            
            # Immediately update camera if we have a current state
            if self.last_state and self.rocket_actors:
                pos = self.last_state.get('position', {})
                position = np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
                self._update_camera_tracking(position, self.last_state.get('time', 0))
        else:
            self.auto_tracking_btn.setText('📹 Auto-Track: OFF')
            self.auto_tracking_btn.setStyleSheet('font-weight: bold; background-color: lightcoral;')
            self.statusBar().showMessage('Auto-tracking disabled - manual camera control')
    
    def on_mode_changed(self):
        """Handle mode selection change."""
        if self.static_mode_radio.isChecked():
            self.mode = 'static'
            self.sim_controls_group.setVisible(False)
            self.static_button_group.setVisible(True)
            self.telemetry_widget.setVisible(False)
            self.mesh_info_group.setVisible(True)
            self.set_replay_overlay_visibility(self.replay_overlay_visible)
            
            # Stop simulation if running
            if self.simulation_controller and self.simulation_controller.is_playing:
                self.simulation_controller.stop()
            
            # Reset interpolation state
            self.last_state = None
            self.current_display_state = None
            
            # Re-render static view
            self.render_selected_components()
        else:
            self.mode = 'simulation'
            self.sim_controls_group.setVisible(True)
            self.static_button_group.setVisible(False)
            self.telemetry_widget.setVisible(True)
            self.mesh_info_group.setVisible(False)
            self.set_replay_overlay_visibility(self.replay_overlay_visible)
            
            # Clear actors and reset state
            self.rocket_actors = {}
            # Reset trajectory tracking
            self.trail_actor = None
            self.trail_polydata = None
            self.trajectory_points = []
            self.last_trajectory_point = None
            self.last_state = None
            self.current_display_state = None
            
            # Clear static view
            self.clear_viewer()

            if self.simulation_controller is None:
                self.load_replay_data()
            else:
                self.statusBar().showMessage('Replay loaded. Use the overlay controls for playback and stepping.')

    def _default_logs_directory(self) -> Path:
        return Path(__file__).resolve().parents[2] / 'logs'

    def on_load_replay_clicked(self):
        """Prompt for kinematics CSV, then load matching virtual sensors CSV."""
        logs_dir = self._default_logs_directory()
        kinematics_path, _ = QFileDialog.getOpenFileName(
            self,
            'Select flight kinematics CSV',
            str(logs_dir),
            'CSV Files (*.csv)',
        )
        if not kinematics_path:
            return

        kinematics_file = Path(kinematics_path)
        suffix = kinematics_file.stem.removeprefix('flight_kinematics_')
        sensor_candidate = kinematics_file.with_name(f'virtual_sensors_full_rate_{suffix}.csv')

        if sensor_candidate.exists():
            sensor_path = str(sensor_candidate)
        else:
            sensor_path, _ = QFileDialog.getOpenFileName(
                self,
                'Select virtual sensors CSV',
                str(logs_dir),
                'CSV Files (*.csv)',
            )
            if not sensor_path:
                return

        self.load_replay_data(sensor_csv_path=sensor_path, kinematics_csv_path=str(kinematics_file))

    def load_replay_data(self, sensor_csv_path: str | None = None, kinematics_csv_path: str | None = None):
        """Load and wire synchronized kinematics/sensor replay data."""
        try:
            if self.simulation_controller and self.simulation_controller.is_playing:
                self.simulation_controller.pause()

            kinematics_df, sensors_df, sensor_path, kinematics_path = load_replay_pair(
                logs_directory=self._default_logs_directory(),
                sensor_csv_path=sensor_csv_path,
                kinematics_csv_path=kinematics_csv_path,
            )
            self.replay_sensor_path = sensor_path
            self.replay_kinematics_path = kinematics_path

            self.simulation_controller = CsvReplayController(kinematics_df, sensors_df, update_rate=120.0)
            self.simulation_controller.state_updated.connect(self.on_simulation_state_updated)
            self.simulation_controller.progress_changed.connect(self.on_progress_changed)

            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.prev_step_btn.setEnabled(True)
            self.next_step_btn.setEnabled(True)
            self.timeline_slider.setEnabled(True)
            self.timeline_slider.setMinimum(0)
            self.timeline_slider.setMaximum(self.simulation_controller.total_steps - 1)
            self.timeline_slider.setValue(0)

            self.replay_source_label.setText(
                f'Replay source:\n- {kinematics_path.name}\n- {sensor_path.name}'
            )

            self.trajectory_points = []
            self.rocket_actors = {}
            self.trail_actor = None
            self.trail_polydata = None
            self.last_trajectory_point = None
            self.last_state = None
            self.current_display_state = None

            self.render_simulation_frame(self.simulation_controller.get_state_at_index(0))
            self.on_progress_changed(0.0)
            self.statusBar().showMessage(
                f'Loaded replay with {self.simulation_controller.total_steps:,} kinematics steps '
                f'and {len(sensors_df):,} sensor rows'
            )
        except Exception as exc:
            QMessageBox.critical(self, 'Replay Load Error', f'Failed to load replay CSVs:\n{exc}')
    
    def on_start_clicked(self):
        """Handle start/resume button click."""
        if self.simulation_controller:
            self.simulation_controller.start()
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.statusBar().showMessage('Simulation playing...')
    
    def on_pause_clicked(self):
        """Handle pause button click."""
        if self.simulation_controller:
            self.simulation_controller.pause()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.statusBar().showMessage('Simulation paused')
    
    def on_stop_clicked(self):
        """Handle stop button click."""
        if self.simulation_controller:
            self.simulation_controller.stop()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.trajectory_points = []
            self.rocket_actors = {}
            self.trail_actor = None
            self.trail_polydata = None
            self.last_trajectory_point = None
            self.last_state = None
            self.current_display_state = None
            self.auto_tracking_enabled = True
            self.user_is_interacting = False
            if hasattr(self, 'auto_tracking_btn'):
                self.auto_tracking_btn.setChecked(True)
                self.auto_tracking_btn.setText('📹 Auto-Track: ON')
                self.auto_tracking_btn.setStyleSheet('font-weight: bold; background-color: lightgreen;')
            self.statusBar().showMessage('Simulation stopped')
            
            # Clear view and re-initialize
            self.plotter.clear()
            self.plotter.add_axes(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
            self.plotter.add_camera_orientation_widget()
            
            # Re-initialize at t=0
            self.render_simulation_frame(self.simulation_controller.get_state_at_index(0))
    
    def on_timeline_pressed(self):
        """Handle timeline slider press (pause playback)."""
        if self.simulation_controller and self.simulation_controller.is_playing:
            self.simulation_controller.pause()
    
    def on_timeline_released(self):
        """Handle timeline slider release."""
        # Update handled by valueChanged
        pass
    
    def on_timeline_changed(self, value):
        """Handle timeline slider value change."""
        if self.simulation_controller and not self.simulation_controller.is_playing:
            self.simulation_controller.seek(int(value), is_progress=False)

    def on_prev_step_clicked(self):
        """Move replay cursor to previous kinematics step."""
        if not self.simulation_controller:
            return
        if self.simulation_controller.is_playing:
            self.simulation_controller.pause()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
        self.simulation_controller.seek(self.simulation_controller.index - 1, is_progress=False)

    def on_next_step_clicked(self):
        """Move replay cursor to next kinematics step."""
        if not self.simulation_controller:
            return
        if self.simulation_controller.is_playing:
            self.simulation_controller.pause()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
        self.simulation_controller.seek(self.simulation_controller.index + 1, is_progress=False)
    
    def on_speed_changed(self, value):
        """Handle playback speed slider change."""
        speed = value / 10.0  # 1-100 -> 0.1x-10x
        self.speed_label.setText(f'Speed: {speed:.1f}x')
        
        if self.simulation_controller:
            self.simulation_controller.set_speed(speed)
    
    @Slot(float)
    def on_progress_changed(self, progress):
        """Handle simulation progress update."""
        del progress
        if not self.simulation_controller:
            return

        time_info = self.simulation_controller.get_time_info()
        index = self.simulation_controller.index
        total = self.simulation_controller.total_steps

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(index)
        self.timeline_slider.blockSignals(False)

        self.timeline_label.setText(
            f'Step: {index + 1:,} / {total:,} | '
            f'Time: {time_info["current_time"]:.3f} s / {time_info["t_final"]:.3f} s'
        )
    
    @Slot(dict)
    def on_simulation_state_updated(self, state):
        """Handle simulation state update from controller."""
        # Update telemetry
        self.telemetry_widget.update_telemetry(state)
        
        # Render 3D frame
        self.render_simulation_frame(state)
    
    def initialize_simulation_view(self):
        """Initialize the 3D view for simulation mode with static meshes and proper bounds."""
        # Clear any previous render
        self.plotter.clear()
        self.plotter.add_axes(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
        self.plotter.add_camera_orientation_widget()
        
        # Get rocket mesh
        full_mesh = self.rocket_renderer.generate_mesh()
        components_list = list(self.selected_components)
        filtered_mesh = self.rocket_renderer._filter_components(full_mesh, components_list)
        
        # Color scheme
        colors = {
            'motor_casing': 'silver',
            'motor_nozzle': 'dimgray',
            'motor_closure': 'silver',
            'nosecone': 'red',
            'body': 'white',
            'tail': 'lightgray',
        }
        fin_colors = ['blue', 'blue', 'blue', 'blue']
        
        # Add all rocket components and store their actors
        self.rocket_actors = {}
        for key in filtered_mesh.keys():
            if key.startswith('fin_'):
                fin_idx = int(key.split('_')[1]) - 1
                color = fin_colors[fin_idx % len(fin_colors)]
            else:
                color = colors.get(key, 'gray')
            
            # Add mesh and store the actor
            actor = self.plotter.add_mesh(
                filtered_mesh[key],
                color=color,
                show_edges=True,
                smooth_shading=True,
                name=key
            )
            self.rocket_actors[key] = {
                'actor': actor,
                'mesh': filtered_mesh[key].copy()
            }
        
        # Calculate bounding box based on loaded replay trajectory
        if self.simulation_controller and self.simulation_controller.total_steps > 0:
            kinematics = self.simulation_controller.kinematics
            x_vals = kinematics['x_m'].to_numpy(dtype=float)
            y_vals = kinematics['y_m'].to_numpy(dtype=float)
            z_vals = kinematics['z_m'].to_numpy(dtype=float)

            ground_elevation = float(np.nanmin(z_vals))
            rocket_length = 2.9

            x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
            y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
            z_min = ground_elevation - rocket_length - 5.0
            z_max = float(np.max(z_vals))

            margin = 0.2
            x_margin = max(5.0, (x_max - x_min) * margin)
            y_margin = max(5.0, (y_max - y_min) * margin)
            z_margin = 50.0

            x_min -= x_margin
            x_max += x_margin
            y_min -= y_margin
            y_max += y_margin
            z_max += z_margin

            center = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0]
        else:
            # Default bounds
            ground_elevation = 0.0
            rocket_length = 2.9
            x_min, x_max = -500, 500
            y_min, y_max = -500, 500
            z_min = ground_elevation - rocket_length - 5
            z_max = ground_elevation + 5000
            center = [0, 0, (z_min + z_max) / 2]
        
        # Create wireframe bounding box cube
        # The actual ground where rocket sits (rocket tail level when on pad)
        # RocketPy z=ground_elevation means nose at that height
        # Rocket extends ~2.9m down from nose, so tail touches at ground_elevation - 2.9
        rocket_length = 2.9  # meters
        actual_ground_z = ground_elevation - rocket_length
        
        # Bottom face (ground level where rocket tail sits)
        ground = pv.Plane(
            center=[center[0], center[1], actual_ground_z],
            direction=[0, 0, 1],
            i_size=x_max - x_min,
            j_size=y_max - y_min,
            i_resolution=20,
            j_resolution=20
        )
        self.ground_actor = self.plotter.add_mesh(
            ground,
            color='green',
            style='wireframe',
            line_width=2,
            name='ground',
            opacity=0.4
        )
        
        # Top face (sky ceiling)
        sky = pv.Plane(
            center=[center[0], center[1], z_max],
            direction=[0, 0, 1],
            i_size=x_max - x_min,
            j_size=y_max - y_min,
            i_resolution=20,
            j_resolution=20
        )
        self.sky_grid_actor = self.plotter.add_mesh(
            sky,
            color='lightblue',
            style='wireframe',
            line_width=1,
            name='sky',
            opacity=0.2
        )
        
        # Boundary frame edges (just the outline, no wall grids)
        # Create wireframe box edges for visual reference without cluttering the view
        box_edges = pv.Box(
            bounds=[x_min, x_max, y_min, y_max, z_min, z_max]
        ).extract_all_edges()
        
        self.plotter.add_mesh(
            box_edges,
            color='gray',
            line_width=1,
            opacity=0.3,
            name='boundary_frame'
        )
        
        # Add visible bounding box edges for clarity
        # Create lines connecting the 8 corners of the cube
        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ])
        
        # Bottom edges
        for i in range(4):
            line = pv.Line(corners[i], corners[(i + 1) % 4])
            self.plotter.add_mesh(line, color='red', line_width=4, name=f'bound_bottom_{i}')
        
        # Top edges
        for i in range(4):
            line = pv.Line(corners[i + 4], corners[(i + 1) % 4 + 4])
            self.plotter.add_mesh(line, color='red', line_width=4, name=f'bound_top_{i}')
        
        # Vertical edges
        for i in range(4):
            line = pv.Line(corners[i], corners[i + 4])
            self.plotter.add_mesh(line, color='red', line_width=4, name=f'bound_vert_{i}')
        
        # Add launch rail visualization
        # Rail starts at actual ground level (where rocket tail sits)
        # and extends up so the rocket nose reaches ground_elevation + rail_length
        rail_bottom = np.array([0, 0, actual_ground_z])
        rail_top = np.array([0, 0, actual_ground_z + self.rail_length])
        
        # Main rail (thick yellow cylinder)
        rail = pv.Cylinder(
            center=(rail_bottom + rail_top) / 2,
            direction=[0, 0, 1],
            radius=0.025,  # ~1 inch diameter rail
            height=self.rail_length
        )
        self.plotter.add_mesh(
            rail,
            color='yellow',
            show_edges=True,
            name='launch_rail',
            opacity=0.8
        )
        
        # Rail base (platform) - sits on actual ground
        base_size = 0.3  # 30cm square base
        rail_base = pv.Box(
            bounds=[
                -base_size/2, base_size/2,
                -base_size/2, base_size/2,
                actual_ground_z - 0.05, actual_ground_z
            ]
        )
        self.plotter.add_mesh(
            rail_base,
            color='darkgray',
            show_edges=True,
            name='rail_base'
        )
        
        # Add rail button positions as markers (where rocket attaches)
        # These are the actual rail button positions from rocket_config
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'Itzamna'))
            from rocket_config import UPPER_BUTTON_POSITION, LOWER_BUTTON_POSITION
            
            # Buttons are in tail_to_nose coordinates (negative = toward tail)
            # When rocket nose is at (actual_ground_z + rail_length), buttons are:
            upper_button_z = actual_ground_z + self.rail_length + UPPER_BUTTON_POSITION
            lower_button_z = actual_ground_z + self.rail_length + LOWER_BUTTON_POSITION
            
            # Add visual markers for rail buttons
            upper_marker = pv.Sphere(radius=0.04, center=[0, 0, upper_button_z])
            lower_marker = pv.Sphere(radius=0.04, center=[0, 0, lower_button_z])
            
            self.plotter.add_mesh(upper_marker, color='orange', name='upper_button')
            self.plotter.add_mesh(lower_marker, color='orange', name='lower_button')
        except:
            # If config not available, skip button markers
            pass
        
        # Initialize empty trail
        self.trail_actor = None
        self.trajectory_points = []
        
        # Set camera to look at rocket on rail from a good starting angle
        # Rocket nose will be at (actual_ground_z + rail_length) when on pad
        rocket_nose_on_pad = actual_ground_z + self.rail_length
        rocket_center_on_pad = rocket_nose_on_pad - rocket_length / 2
        
        # Position camera closer for better initial view
        # Camera at 20m away at 45° angle, looking at rocket center
        self.plotter.camera.position = (15, -15, rocket_center_on_pad + 5)
        self.plotter.camera.focal_point = (0, 0, rocket_center_on_pad)
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.camera.view_angle = 45.0  # Wider field of view
        
        # Reset camera bounds to ensure proper zoom
        self.plotter.reset_camera()
        
        # Re-apply our desired position after reset
        self.plotter.camera.position = (15, -15, rocket_center_on_pad + 5)
        self.plotter.camera.focal_point = (0, 0, rocket_center_on_pad)
    
    def render_simulation_frame(self, state):
        """Update rocket position and orientation efficiently with smooth interpolation."""
        if not state:
            return
        
        # First frame - initialize the view
        if not self.rocket_actors:
            self.initialize_simulation_view()
            self.last_state = state
            # Initialize trajectory tracking with starting position
            pos = state.get('position', {})
            self.last_trajectory_point = np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
        
        # Use current state directly (no interpolation for now)
        # Get position and quaternion
        pos = state.get('position', {})
        quat = state.get('quaternion', {})
        
        position = np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
        quaternion = (quat.get('e0', 1), quat.get('e1', 0), 
                     quat.get('e2', 0), quat.get('e3', 0))
        
        # Add to trajectory
        self.trajectory_points.append(position.copy())
        
        # Convert quaternion to rotation matrix
        R = quaternion_to_matrix(*quaternion)
        
        # Create transformation matrix (4x4)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = position
        
        # Create VTK transform
        import vtk
        vtk_transform = vtk.vtkTransform()
        vtk_transform.SetMatrix(transform_matrix.flatten())
        
        # Update each rocket component's transformation using SetUserTransform
        for key, data in self.rocket_actors.items():
            actor = data['actor']
            actor.SetUserTransform(vtk_transform)
        
        # Update camera position if auto-tracking is on
        self._update_camera_tracking(position, state.get('time', 0))
        
        # Update trajectory trail incrementally (efficient - no recreation)
        if self.last_trajectory_point is not None:
            # Create a line segment from last point to current point
            segment = pv.Line(self.last_trajectory_point, position)
            
            # Add the segment to the scene
            self.plotter.add_mesh(
                segment, 
                color='cyan', 
                line_width=3, 
                name=f'trail_segment_{len(self.trajectory_points)}'
            )
        
        # Update last trajectory point for next segment
        self.last_trajectory_point = position.copy()
        
        # Store state
        self.last_state = state
    
    def _interpolate_states(self, state1, state2, alpha):
        """
        Interpolate between two flight states for smooth rendering.
        
        Args:
            state1: Previous state
            state2: New state
            alpha: Interpolation factor (0.0 = state1, 1.0 = state2, 0.3 = smooth blend)
        
        Returns:
            Interpolated state dictionary
        """
        if state1 is None or state2 is None:
            return state2
        
        # Linear interpolation for position
        pos1 = state1.get('position', {})
        pos2 = state2.get('position', {})
        
        interp_pos = {
            'x': (1 - alpha) * pos1.get('x', 0) + alpha * pos2.get('x', 0),
            'y': (1 - alpha) * pos1.get('y', 0) + alpha * pos2.get('y', 0),
            'z': (1 - alpha) * pos1.get('z', 0) + alpha * pos2.get('z', 0),
            'altitude': (1 - alpha) * pos1.get('altitude', 0) + alpha * pos2.get('altitude', 0),
        }
        
        # SLERP interpolation for quaternion (smooth rotation)
        quat1 = state1.get('quaternion', {})
        quat2 = state2.get('quaternion', {})
        
        q1 = (quat1.get('e0', 1), quat1.get('e1', 0), quat1.get('e2', 0), quat1.get('e3', 0))
        q2 = (quat2.get('e0', 1), quat2.get('e1', 0), quat2.get('e2', 0), quat2.get('e3', 0))
        
        # Use SLERP for smooth quaternion interpolation
        interp_q = interpolate_quaternion(q1, q2, alpha)
        
        interp_quat = {
            'e0': interp_q[0],
            'e1': interp_q[1],
            'e2': interp_q[2],
            'e3': interp_q[3],
        }
        
        # Create interpolated state (copy other fields from state2)
        interp_state = state2.copy()
        interp_state['position'] = interp_pos
        interp_state['quaternion'] = interp_quat
        
        return interp_state
    
    def _update_camera_tracking(self, rocket_position, current_time):
        """
        Update camera to smoothly follow the rocket.
        Maintains a fixed relative position to the rocket while allowing user zoom control.
        
        Args:
            rocket_position: Current rocket position [x, y, z]
            current_time: Current simulation time
        """
        # Only update if auto-tracking is enabled
        if not self.auto_tracking_enabled:
            return
        
        camera = self.plotter.camera
        
        # Get current camera distance from focal point
        current_pos = np.array(camera.position)
        current_focal = np.array(camera.focal_point)
        current_offset = current_pos - current_focal
        current_distance = np.linalg.norm(current_offset)
        
        # If this is the first update, set a reasonable default distance
        if current_distance < 1.0:
            current_distance = 50.0
            current_offset = np.array([1, -1, 0.6]) * current_distance / np.linalg.norm([1, -1, 0.6])
        
        # Update focal point to rocket position
        camera.focal_point = tuple(rocket_position)
        
        # Maintain the same relative offset and distance from the rocket
        # This keeps the camera "attached" to the rocket at the user's chosen distance
        camera.position = tuple(rocket_position + current_offset)
        
        # Keep up direction consistent
        camera.up = (0, 0, 1)
    
    def on_render_clicked(self):
        """Handle render button click."""
        self.render_selected_components()
    
    def render_selected_components(self):
        """Render selected rocket components in the viewer."""
        if not self.selected_components:
            QMessageBox.warning(self, 'No Selection', 'Please select at least one component to render.')
            return
        
        self.statusBar().showMessage('Rendering components...')
        
        try:
            # Clear previous render
            self.plotter.clear()
            
            # Re-add axes and camera widget
            self.plotter.add_axes(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
            self.plotter.add_camera_orientation_widget()
            
            # Get full mesh and filter to selected components
            full_mesh = self.rocket_renderer.generate_mesh()
            components_list = list(self.selected_components)
            filtered_mesh = self.rocket_renderer._filter_components(full_mesh, components_list)
            
            # Color scheme
            colors = {
                'motor_casing': 'silver',
                'motor_nozzle': 'dimgray',
                'motor_closure': 'silver',
                'nosecone': 'red',
                'body': 'white',
                'tail': 'lightgray',
            }
            fin_colors = ['blue', 'blue', 'blue', 'blue']
            
            # Add components to plotter
            for key in filtered_mesh.keys():
                if key.startswith('fin_'):
                    fin_idx = int(key.split('_')[1]) - 1
                    color = fin_colors[fin_idx % len(fin_colors)]
                    label = f'Fin {fin_idx + 1}'
                else:
                    color = colors.get(key, 'gray')
                    label = key.replace('_', ' ').title()
                
                self.plotter.add_mesh(
                    filtered_mesh[key],
                    color=color,
                    show_edges=True,
                    label=label,
                    smooth_shading=True
                )
            
            # Add legend
            self.plotter.add_legend(bcolor='white')
            
            # Reset camera
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            
            # Update info display
            info = self.rocket_renderer.get_mesh_info(components=components_list)
            self.update_info_display(info)
            
            self.statusBar().showMessage(f'Rendered {len(components_list)} component(s) successfully')
            
        except Exception as e:
            QMessageBox.critical(self, 'Render Error', f'Failed to render components:\n{str(e)}')
            self.statusBar().showMessage('Render failed')
    
    def clear_viewer(self):
        """Clear the 3D viewer."""
        self.plotter.clear()
        self.info_display.clear()
        self.statusBar().showMessage('Viewer cleared')
    
    def update_info_display(self, info: dict):
        """Update the mesh information display."""
        text = '<b>Mesh Statistics:</b><br><br>'
        text += f'Components: {info["n_components"]}<br>'
        text += f'Total Vertices: {info["total_points"]:,}<br>'
        text += f'Total Polygons: {info["total_cells"]:,}<br><br>'
        
        text += '<b>Component Breakdown:</b><br>'
        for comp in info['components']:
            points = info.get(f'{comp}_points', 0)
            cells = info.get(f'{comp}_cells', 0)
            text += f'  • {comp}: {points:,} vertices, {cells:,} polygons<br>'
        
        self.info_display.setHtml(text)
    
    def on_export_clicked(self):
        """Handle export button click."""
        if not self.rocket_renderer:
            QMessageBox.warning(self, 'No Object', 'No rocket loaded.')
            return
        
        # Open file dialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            'Export Model',
            '',
            'STL Files (*.stl);;OBJ Files (*.obj);;PLY Files (*.ply);;VTK Files (*.vtk)'
        )
        
        if not filepath:
            return
        
        try:
            self.statusBar().showMessage('Exporting model...')
            components_list = list(self.selected_components) if self.selected_components else 'all'
            self.rocket_renderer.save_model(filepath, components=components_list)
            QMessageBox.information(self, 'Export Successful', f'Model exported to:\n{filepath}')
            self.statusBar().showMessage('Export successful')
        except Exception as e:
            QMessageBox.critical(self, 'Export Error', f'Failed to export model:\n{str(e)}')
            self.statusBar().showMessage('Export failed')
    
    def run(self):
        """Show the application window."""
        self.show()
    
    def closeEvent(self, event):
        """Handle window close event with proper cleanup."""
        try:
            # Close PyVista plotter gracefully to prevent VTK errors
            if hasattr(self, 'plotter') and self.plotter is not None:
                self.plotter.close()
        except Exception:
            # Ignore any errors during cleanup
            pass
        
        # Accept the close event
        event.accept()


def launch_gui(rocket):
    """
    Launch the Rocket 3D Viewer application.
    
    Args:
        rocket: RocketPy Rocket object to visualize
    Returns:
        Application exit code
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        print(f'Qt backend in use: {app.platformName()}')
    except Exception:
        pass
    
    viewer = RocketViewerApp(rocket=rocket)
    viewer.show()
    
    return app.exec_()
