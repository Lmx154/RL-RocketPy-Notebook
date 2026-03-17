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
        self._fresh_color = '#2EAD4A'
        self._stale_color = '#E6C229'
        
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
        self.step_label = QLabel('0 / 0')
        self.phase_label = QLabel('---')
        time_group.layout().addWidget(self._create_row('Time:', self.time_label))
        time_group.layout().addWidget(self._create_row('Step:', self.step_label))
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

        # Raw virtual sensor values aligned to current kinematics step
        sensor_group = self._create_group('Virtual Sensors')
        self.accel_label = QLabel('---, ---, --- m/s²')
        self.gyro_label = QLabel('---, ---, --- rad/s')
        self.baro_label = QLabel('--- Pa')
        self.sensor_gnss_label = QLabel('---, ---, ---')
        sensor_group.layout().addWidget(self._create_row('Accel XYZ:', self.accel_label))
        sensor_group.layout().addWidget(self._create_row('Gyro XYZ:', self.gyro_label))
        sensor_group.layout().addWidget(self._create_row('Barometer:', self.baro_label))
        sensor_group.layout().addWidget(self._create_row('GNSS LLA:', self.sensor_gnss_label))
        layout.addWidget(sensor_group)
        
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
        step_index = int(state.get('step_index', 0))
        total_steps = int(state.get('total_steps', 0))
        self.time_label.setText(f'{time_val:.2f} s / {t_final:.1f} s')
        self.step_label.setText(f'{step_index + 1:,} / {total_steps:,}')
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

        sensors = state.get('sensors', {})
        freshness = state.get('sensor_freshness', {})

        ax = sensors.get('accelerometer_x')
        ay = sensors.get('accelerometer_y')
        az = sensors.get('accelerometer_z')
        gx = sensors.get('gyroscope_x')
        gy = sensors.get('gyroscope_y')
        gz = sensors.get('gyroscope_z')
        bp = sensors.get('barometer_v1')
        glat = sensors.get('gnss_x')
        glon = sensors.get('gnss_y')
        galt = sensors.get('gnss_z')

        self.accel_label.setText(
            f'{self._fmt_signed(ax, 3)}, {self._fmt_signed(ay, 3)}, {self._fmt_signed(az, 3)}'
        )
        self.gyro_label.setText(
            f'{self._fmt_signed(gx, 4)}, {self._fmt_signed(gy, 4)}, {self._fmt_signed(gz, 4)}'
        )
        self.baro_label.setText(self._fmt_plain(bp, 1, unit=' Pa'))
        self.sensor_gnss_label.setText(
            f'{self._fmt_plain(glat, 6)}, {self._fmt_plain(glon, 6)}, {self._fmt_plain(galt, 2)}'
        )

        accel_fresh = bool(freshness.get('accelerometer_x', False))
        gyro_fresh = bool(freshness.get('gyroscope_x', False))
        baro_fresh = bool(freshness.get('barometer_v1', False))
        gnss_fresh = bool(freshness.get('gnss_x', False))

        self._set_sensor_freshness_style(self.accel_label, accel_fresh)
        self._set_sensor_freshness_style(self.gyro_label, gyro_fresh)
        self._set_sensor_freshness_style(self.baro_label, baro_fresh)
        self._set_sensor_freshness_style(self.sensor_gnss_label, gnss_fresh)

    def _set_sensor_freshness_style(self, label: QLabel, is_fresh: bool) -> None:
        color = self._fresh_color if is_fresh else self._stale_color
        label.setStyleSheet(f'font-family: monospace; font-size: 10pt; color: {color};')

    @staticmethod
    def _fmt_signed(value: Any, decimals: int) -> str:
        if value is None:
            return '---'
        return f'{float(value):+.{decimals}f}'

    @staticmethod
    def _fmt_plain(value: Any, decimals: int, unit: str = '') -> str:
        if value is None:
            return f'---{unit}'
        return f'{float(value):.{decimals}f}{unit}'
    
    def _clear_display(self):
        """Clear all telemetry values."""
        self.time_label.setText('--:--')
        self.step_label.setText('0 / 0')
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
        self.accel_label.setText('---, ---, --- m/s²')
        self.gyro_label.setText('---, ---, --- rad/s')
        self.baro_label.setText('--- Pa')
        self.sensor_gnss_label.setText('---, ---, ---')
        self._set_sensor_freshness_style(self.accel_label, False)
        self._set_sensor_freshness_style(self.gyro_label, False)
        self._set_sensor_freshness_style(self.baro_label, False)
        self._set_sensor_freshness_style(self.sensor_gnss_label, False)
