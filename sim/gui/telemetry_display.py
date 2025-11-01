"""
Telemetry display widget for flight simulation visualization.

Shows real-time flight data in a formatted panel:
- Time and flight phase
- Altitude and vertical speed
- Total speed and Mach number
- GPS coordinates (lat/lon)
- Attitude angles (phi, theta, psi)
- Dynamic pressure (for air brake control)
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QFrame
from PySide6.QtCore import Qt, Slot
from typing import Dict, Any


class TelemetryDisplay(QWidget):
    """
    Widget displaying real-time flight telemetry data.
    
    Updates from SimulationController via state_updated signal.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize UI
        self.init_ui()
        
        # Current state
        self.current_state = {}
    
    def init_ui(self):
        """Initialize the telemetry display UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Title
        title = QLabel('<h3>Telemetry</h3>')
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Time and Phase
        time_group = self._create_group('Time & Phase')
        self.time_label = QLabel('--:--')
        self.phase_label = QLabel('---')
        time_group.layout().addWidget(self._create_row('Time:', self.time_label))
        time_group.layout().addWidget(self._create_row('Phase:', self.phase_label))
        layout.addWidget(time_group)
        
        # Altitude and Speed
        alt_group = self._create_group('Altitude & Velocity')
        self.altitude_label = QLabel('--- m')
        self.vz_label = QLabel('--- m/s')
        self.speed_label = QLabel('--- m/s')
        self.mach_label = QLabel('---')
        alt_group.layout().addWidget(self._create_row('Altitude:', self.altitude_label))
        alt_group.layout().addWidget(self._create_row('Vert. Speed:', self.vz_label))
        alt_group.layout().addWidget(self._create_row('Total Speed:', self.speed_label))
        alt_group.layout().addWidget(self._create_row('Mach:', self.mach_label))
        layout.addWidget(alt_group)
        
        # GPS Coordinates
        gps_group = self._create_group('GPS Coordinates')
        self.lat_label = QLabel('---° N')
        self.lon_label = QLabel('---° W')
        gps_group.layout().addWidget(self._create_row('Latitude:', self.lat_label))
        gps_group.layout().addWidget(self._create_row('Longitude:', self.lon_label))
        layout.addWidget(gps_group)
        
        # Attitude (Euler Angles)
        att_group = self._create_group('Attitude (Euler)')
        self.phi_label = QLabel('---°')
        self.theta_label = QLabel('---°')
        self.psi_label = QLabel('---°')
        att_group.layout().addWidget(self._create_row('Phi (Spin):', self.phi_label))
        att_group.layout().addWidget(self._create_row('Theta (Nutation):', self.theta_label))
        att_group.layout().addWidget(self._create_row('Psi (Precession):', self.psi_label))
        layout.addWidget(att_group)
        
        # Additional Info
        extra_group = self._create_group('Additional')
        self.q_label = QLabel('--- Pa')
        self.apogee_label = QLabel('--- m')
        extra_group.layout().addWidget(self._create_row('Dyn. Pressure:', self.q_label))
        extra_group.layout().addWidget(self._create_row('Apogee:', self.apogee_label))
        layout.addWidget(extra_group)
        
        # Spacer
        layout.addStretch()
    
    def _create_group(self, title: str) -> QGroupBox:
        """Create a group box with vertical layout."""
        group = QGroupBox(title)
        group.setLayout(QVBoxLayout())
        group.layout().setContentsMargins(5, 5, 5, 5)
        group.layout().setSpacing(3)
        return group
    
    def _create_row(self, label_text: str, value_widget: QLabel) -> QWidget:
        """Create a row with label and value."""
        from PySide6.QtWidgets import QHBoxLayout
        
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(5)
        
        label = QLabel(label_text)
        label.setMinimumWidth(100)
        
        value_widget.setAlignment(Qt.AlignRight)
        value_widget.setStyleSheet('font-family: monospace; font-size: 10pt;')
        
        row_layout.addWidget(label)
        row_layout.addWidget(value_widget)
        
        return row
    
    @Slot(dict)
    def update_telemetry(self, state: Dict[str, Any]):
        """
        Update telemetry display with new state.
        
        Args:
            state: Flight state dictionary from SimulationController
        """
        self.current_state = state
        
        if not state:
            self._clear_display()
            return
        
        # Time and Phase
        time_val = state.get('time', 0.0)
        t_final = state.get('t_final', 0.0)
        self.time_label.setText(f'{time_val:.2f} s / {t_final:.1f} s')
        self.phase_label.setText(state.get('phase', '---'))
        
        # Altitude and Velocity
        position = state.get('position', {})
        velocity = state.get('velocity', {})
        
        altitude = position.get('altitude', 0.0)
        vz = velocity.get('vz', 0.0)
        speed = velocity.get('speed', 0.0)
        mach = state.get('mach_number', 0.0)
        
        self.altitude_label.setText(f'{altitude:,.0f} m')
        self.vz_label.setText(f'{vz:+.1f} m/s')
        self.speed_label.setText(f'{speed:.1f} m/s')
        self.mach_label.setText(f'{mach:.2f}')
        
        # GPS Coordinates
        gps = state.get('gps', {})
        lat = gps.get('latitude', 0.0)
        lon = gps.get('longitude', 0.0)
        
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        
        self.lat_label.setText(f'{abs(lat):.6f}° {lat_dir}')
        self.lon_label.setText(f'{abs(lon):.6f}° {lon_dir}')
        
        # Attitude (Euler Angles)
        euler = state.get('euler', {})
        phi = euler.get('phi', 0.0)
        theta = euler.get('theta', 0.0)
        psi = euler.get('psi', 0.0)
        
        self.phi_label.setText(f'{phi:.1f}°')
        self.theta_label.setText(f'{theta:.1f}°')
        self.psi_label.setText(f'{psi:.1f}°')
        
        # Additional Info
        q = state.get('dynamic_pressure', 0.0)
        apogee = state.get('apogee', 0.0)
        
        self.q_label.setText(f'{q:.0f} Pa')
        self.apogee_label.setText(f'{apogee:,.0f} m')
    
    def _clear_display(self):
        """Clear all telemetry values."""
        self.time_label.setText('--:--')
        self.phase_label.setText('---')
        self.altitude_label.setText('--- m')
        self.vz_label.setText('--- m/s')
        self.speed_label.setText('--- m/s')
        self.mach_label.setText('---')
        self.lat_label.setText('---° N')
        self.lon_label.setText('---° W')
        self.phi_label.setText('---°')
        self.theta_label.setText('---°')
        self.psi_label.setText('---°')
        self.q_label.setText('--- Pa')
        self.apogee_label.setText('--- m')
