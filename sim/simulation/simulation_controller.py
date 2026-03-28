"""
Simulation Controller for RocketPy Flight Playback.

This controller manages playback of pre-computed RocketPy Flight trajectories
at configurable rates (default 100 Hz). It provides a clean interface between
the Flight physics data and the GUI visualization.

Architecture:
- Flight simulation is pre-computed once (1-5 seconds)
- Playback queries Flight data at regular intervals (100 Hz default)
- Qt signals emit state updates for thread-safe GUI updates
- Supports pause/resume, speed control, and timeline scrubbing

Usage:
    controller = SimulationController(flight)
    controller.state_updated.connect(gui_update_function)
    controller.start()
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from rocketpy import Flight

from sim.sitl.replay_session import ReplaySessionScheduler
from sim.sitl.session import ReplaySession, load_replay_session as load_manifest_replay_session

class SimulationController(QObject):
    """
    Controls playback of a pre-computed RocketPy Flight simulation.
    
    Signals:
        state_updated: Emitted at update_rate with current flight state
        simulation_started: Emitted when playback starts
        simulation_paused: Emitted when playback pauses
        simulation_stopped: Emitted when playback stops or completes
        progress_changed: Emitted with progress percentage (0-100)
    """
    
    # Qt Signals (thread-safe)
    state_updated = Signal(dict)  # Current flight state
    simulation_started = Signal()
    simulation_paused = Signal()
    simulation_stopped = Signal()
    progress_changed = Signal(float)  # Progress 0.0 to 1.0
    
    def __init__(self, flight: Optional[Flight] = None, update_rate: float = 100.0):
        """
        Initialize simulation controller.
        
        Args:
            flight: Pre-computed RocketPy Flight object (can be set later)
            update_rate: Update frequency in Hz (default 100 Hz)
        """
        super().__init__()
        
        self.flight = flight
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate  # Time step in seconds
        
        # Playback state
        self.current_time = 0.0
        self.is_playing = False
        self.playback_speed = 1.0  # 1.0 = real-time, 2.0 = 2x speed, etc.
        
        # Qt Timer for periodic updates
        self.timer = QTimer()
        self.timer.setInterval(int(self.dt * 1000))  # Convert to milliseconds
        self.timer.timeout.connect(self._update_loop)
        
        # Performance tracking
        self.last_update_time = None
    
    def set_flight(self, flight: Flight):
        """
        Set or update the Flight object.
        
        Args:
            flight: Pre-computed RocketPy Flight object
        """
        was_playing = self.is_playing
        if was_playing:
            self.pause()
        
        self.flight = flight
        self.current_time = 0.0
        
        if was_playing:
            self.start()
    
    def get_state_at_time(self, t: float) -> Dict[str, Any]:
        """
        Extract complete flight state at given time.
        
        Args:
            t: Time in seconds since launch
        
        Returns:
            Dictionary containing all flight data at time t
        """
        if self.flight is None:
            return {}
        
        # Clamp time to valid range
        t = np.clip(t, 0.0, self.flight.t_final)
        
        # Extract position (inertial frame - ENU)
        position = {
            'x': float(self.flight.x(t)),
            'y': float(self.flight.y(t)),
            'z': float(self.flight.z(t)),
            'altitude': float(self.flight.altitude(t)),
        }
        
        # Extract GPS coordinates
        gps = {
            'latitude': float(self.flight.latitude(t)),
            'longitude': float(self.flight.longitude(t)),
        }
        
        # Extract attitude (quaternion - PREFERRED)
        quaternion = {
            'e0': float(self.flight.e0(t)),
            'e1': float(self.flight.e1(t)),
            'e2': float(self.flight.e2(t)),
            'e3': float(self.flight.e3(t)),
        }
        
        # Extract Euler angles (for display only)
        euler = {
            'phi': float(self.flight.phi(t)),
            'theta': float(self.flight.theta(t)),
            'psi': float(self.flight.psi(t)),
        }
        
        # Extract velocity (inertial frame)
        velocity = {
            'vx': float(self.flight.vx(t)),
            'vy': float(self.flight.vy(t)),
            'vz': float(self.flight.vz(t)),
            'speed': float(self.flight.speed(t)),
            'horizontal_speed': float(self.flight.horizontal_speed(t)),
        }
        
        # Extract angular velocity (body frame)
        angular_velocity = {
            'w1': float(self.flight.w1(t)),
            'w2': float(self.flight.w2(t)),
            'w3': float(self.flight.w3(t)),
        }
        
        # Extract aerodynamics
        try:
            mach = float(self.flight.mach_number(t))
            dynamic_pressure = float(self.flight.dynamic_pressure(t))
        except:
            mach = 0.0
            dynamic_pressure = 0.0
        
        # Determine flight phase
        phase = self._get_flight_phase(t)
        
        # Compile complete state
        state = {
            'time': t,
            'position': position,
            'gps': gps,
            'quaternion': quaternion,
            'euler': euler,
            'velocity': velocity,
            'angular_velocity': angular_velocity,
            'mach_number': mach,
            'dynamic_pressure': dynamic_pressure,
            'phase': phase,
            't_final': float(self.flight.t_final),
            'apogee': float(self.flight.apogee),
        }
        
        return state
    
    def _get_flight_phase(self, t: float) -> str:
        """
        Determine current flight phase based on time and state.
        
        Args:
            t: Current time in seconds
        
        Returns:
            Flight phase string
        """
        if self.flight is None:
            return "NO_FLIGHT"
        
        if t <= 0.0:
            return "PRE_LAUNCH"
        
        # Check if motor is still burning (approximate)
        # Motor burns for ~3.9 seconds for AeroTech M2500T
        motor_burn_time = 3.9  # Could extract from motor object
        
        if t < motor_burn_time:
            return "POWERED_ASCENT"
        elif t < self.flight.apogee_time:
            return "COAST_ASCENT"
        elif abs(t - self.flight.apogee_time) < 0.5:
            return "APOGEE"
        elif t < self.flight.t_final * 0.95:
            return "DESCENT"
        else:
            return "LANDING"
    
    @Slot()
    def _update_loop(self):
        """
        Main update loop called by Qt timer.
        
        Advances simulation time and emits state updates.
        """
        if not self.is_playing or self.flight is None:
            return
        
        # Calculate actual elapsed time since last update
        current_real_time = time.time()
        if self.last_update_time is not None:
            actual_dt = current_real_time - self.last_update_time
        else:
            actual_dt = self.dt
        self.last_update_time = current_real_time
        
        # Advance simulation time (with speed multiplier)
        time_step = actual_dt * self.playback_speed
        self.current_time += time_step
        
        # Check if simulation has ended
        if self.current_time >= self.flight.t_final:
            self.current_time = self.flight.t_final
            self.stop()
            return
        
        # Get state at current time
        state = self.get_state_at_time(self.current_time)
        
        # Emit signals
        self.state_updated.emit(state)
        
        progress = self.current_time / self.flight.t_final
        self.progress_changed.emit(progress)
    
    def start(self):
        """Start or resume playback."""
        if self.flight is None:
            print("Warning: No flight loaded. Call set_flight() first.")
            return
        
        if self.is_playing:
            return  # Already playing
        
        self.is_playing = True
        self.last_update_time = time.time()
        self.timer.start()
        self.simulation_started.emit()
    
    def pause(self):
        """Pause playback."""
        if not self.is_playing:
            return  # Already paused
        
        self.is_playing = False
        self.timer.stop()
        self.last_update_time = None
        self.simulation_paused.emit()
    
    def stop(self):
        """Stop playback and reset to beginning."""
        was_playing = self.is_playing
        
        self.is_playing = False
        self.timer.stop()
        self.current_time = 0.0
        self.last_update_time = None
        
        if was_playing:
            self.simulation_stopped.emit()
    
    def reset(self):
        """Reset playback to beginning without stopping."""
        self.current_time = 0.0
        if self.flight is not None:
            state = self.get_state_at_time(0.0)
            self.state_updated.emit(state)
            self.progress_changed.emit(0.0)
    
    def seek(self, time_or_progress: float, is_progress: bool = False):
        """
        Jump to a specific time or progress point.
        
        Args:
            time_or_progress: Time in seconds or progress (0.0 to 1.0)
            is_progress: If True, interpret as progress; if False, as time
        """
        if self.flight is None:
            return
        
        if is_progress:
            # Convert progress to time
            progress = np.clip(time_or_progress, 0.0, 1.0)
            self.current_time = progress * self.flight.t_final
        else:
            # Direct time
            self.current_time = np.clip(time_or_progress, 0.0, self.flight.t_final)
        
        # Emit state at new time
        state = self.get_state_at_time(self.current_time)
        self.state_updated.emit(state)
        
        progress = self.current_time / self.flight.t_final
        self.progress_changed.emit(progress)
    
    def set_speed(self, speed: float):
        """
        Set playback speed multiplier.
        
        Args:
            speed: Playback speed (1.0 = real-time, 2.0 = 2x, 0.5 = slow-mo)
        """
        self.playback_speed = max(0.1, min(10.0, speed))  # Clamp to [0.1, 10.0]
    
    def get_progress(self) -> float:
        """
        Get current playback progress.
        
        Returns:
            Progress from 0.0 to 1.0
        """
        if self.flight is None or self.flight.t_final <= 0:
            return 0.0
        return self.current_time / self.flight.t_final
    
    def get_time_info(self) -> Dict[str, float]:
        """
        Get timing information.
        
        Returns:
            Dictionary with time info
        """
        if self.flight is None:
            return {
                'current_time': 0.0,
                't_final': 0.0,
                'progress': 0.0,
                'remaining_time': 0.0,
            }
        
        return {
            'current_time': self.current_time,
            't_final': self.flight.t_final,
            'progress': self.get_progress(),
            'remaining_time': self.flight.t_final - self.current_time,
        }


KINEMATICS_COLUMNS_14 = [
    'time_s',
    'x_m',
    'y_m',
    'z_m',
    'vx_mps',
    'vy_mps',
    'vz_mps',
    'e0',
    'e1',
    'e2',
    'e3',
    'w1_radps',
    'w2_radps',
    'w3_radps',
]

def _extract_float_tokens(text: str) -> list[float]:
    pattern = r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?'
    return [float(token) for token in re.findall(pattern, text)]


def _infer_row_width(tokens: list[float]) -> int:
    if not tokens:
        raise ValueError('Kinematics file did not contain numeric values')

    best_width = None
    best_neg_steps = None
    for width in (14, 11, 12, 13):
        rows = len(tokens) // width
        if rows < 2:
            continue
        arr = np.asarray(tokens[: rows * width], dtype=float).reshape(rows, width)
        t = arr[:, 0]
        neg_steps = int(np.sum(np.diff(t) < 0.0))
        if best_neg_steps is None or neg_steps < best_neg_steps:
            best_neg_steps = neg_steps
            best_width = width
        if neg_steps == 0:
            return width

    if best_width is None:
        raise ValueError('Unable to infer kinematics row width')
    return best_width


def load_kinematics_csv(path: str | Path) -> pd.DataFrame:
    """Load kinematics CSV, including wrapped-row exports that need token reconstruction."""
    file_path = Path(path)
    raw_text = file_path.read_text(encoding='utf-8', errors='ignore')
    lines = raw_text.splitlines()
    if len(lines) < 2:
        raise ValueError(f'Kinematics file {file_path} is empty')

    tokens = _extract_float_tokens('\n'.join(lines[1:]))
    width = _infer_row_width(tokens)
    rows = len(tokens) // width
    data = np.asarray(tokens[: rows * width], dtype=float).reshape(rows, width)

    if width == 14:
        frame = pd.DataFrame(data, columns=KINEMATICS_COLUMNS_14)
    elif width == 11:
        frame = pd.DataFrame(
            data,
            columns=[
                'time_s',
                'x_m',
                'y_m',
                'z_m',
                'e0',
                'e1',
                'e2',
                'e3',
                'w1_radps',
                'w2_radps',
                'w3_radps',
            ],
        )
        frame['vx_mps'] = 0.0
        frame['vy_mps'] = 0.0
        frame['vz_mps'] = 0.0
        frame = frame[KINEMATICS_COLUMNS_14]
    else:
        raise ValueError(f'Unsupported reconstructed kinematics row width: {width}')

    return frame.sort_values('time_s').reset_index(drop=True)


def load_replay_session(
    *,
    logs_directory: str | Path = 'logs',
    session_path: str | Path | None = None,
) -> ReplaySession:
    """Load and return one manifest-based replay session."""
    return load_manifest_replay_session(
        session_path=session_path,
        logs_directory=logs_directory,
    )


class CsvReplayController(QObject):
    """Step-based replay controller for manifest-based multi-rate sessions."""

    state_updated = Signal(dict)
    simulation_started = Signal()
    simulation_paused = Signal()
    simulation_stopped = Signal()
    progress_changed = Signal(float)

    def __init__(
        self,
        replay_session: ReplaySession,
        update_rate: float = 120.0,
    ):
        super().__init__()

        self.replay_session = replay_session
        self.kinematics = replay_session.truth
        self.scheduler = ReplaySessionScheduler(replay_session)

        self.index = 0
        self.is_playing = False
        self.playback_speed = 1.0

        self.update_rate = float(update_rate)
        self.dt = 1.0 / self.update_rate
        self.timer = QTimer()
        self.timer.setInterval(int(self.dt * 1000.0))
        self.timer.timeout.connect(self._update_loop)
        self.last_update_time: float | None = None
        self.accumulated_time_s = 0.0

        if len(self.scheduler.truth) > 1:
            dt_values = np.diff(self.scheduler.truth_times_s)
            self.nominal_dt_s = float(np.median(dt_values[dt_values > 0.0])) if np.any(dt_values > 0.0) else 0.01
        else:
            self.nominal_dt_s = 0.01

    @property
    def total_steps(self) -> int:
        return self.scheduler.total_steps

    @property
    def at_end(self) -> bool:
        return self.scheduler.at_end

    def _clamp_index(self, index: int) -> int:
        return self.scheduler.clamp_index(index)

    def get_state_at_index(self, index: int) -> Dict[str, Any]:
        idx = self._clamp_index(index)
        self.index = idx
        return self.scheduler.seek_truth_index(idx)

    def get_state_at_time(self, t: float) -> Dict[str, Any]:
        times = self.scheduler.truth_times_s
        idx = int(np.searchsorted(times, float(t), side='left'))
        return self.get_state_at_index(self._clamp_index(idx))

    def _emit_current(self) -> None:
        state = self.get_state_at_index(self.index)
        self.state_updated.emit(state)
        self.progress_changed.emit(self.get_progress())

    @Slot()
    def _update_loop(self):
        if not self.is_playing or self.at_end:
            return

        now = time.time()
        if self.last_update_time is None:
            elapsed_s = self.nominal_dt_s
        else:
            elapsed_s = max(0.0, now - self.last_update_time)
        self.last_update_time = now

        self.accumulated_time_s += elapsed_s * self.playback_speed
        advanced = False
        times = self.kinematics['time_s'].to_numpy(dtype=float)

        while not self.at_end:
            dt_next = float(times[self.index + 1] - times[self.index])
            if self.accumulated_time_s + 1e-12 < dt_next:
                break
            self.accumulated_time_s -= max(dt_next, 0.0)
            self.index += 1
            self.scheduler.advance_one_tick()
            advanced = True

        if advanced:
            self._emit_current()

        if self.at_end:
            self.pause()

    def start(self):
        if self.is_playing:
            return
        if self.at_end:
            self.index = 0
            self.scheduler.reset()
            self._emit_current()
        self.is_playing = True
        self.last_update_time = time.time()
        self.accumulated_time_s = 0.0
        self.timer.start()
        self.simulation_started.emit()

    def pause(self):
        if not self.is_playing:
            return
        self.is_playing = False
        self.timer.stop()
        self.last_update_time = None
        self.simulation_paused.emit()

    def stop(self):
        was_playing = self.is_playing
        self.is_playing = False
        self.timer.stop()
        self.index = 0
        self.scheduler.reset()
        self.last_update_time = None
        self.accumulated_time_s = 0.0
        self._emit_current()
        if was_playing:
            self.simulation_stopped.emit()

    def reset(self):
        self.index = 0
        self.scheduler.reset()
        self.accumulated_time_s = 0.0
        self._emit_current()

    def seek(self, time_or_progress: float, is_progress: bool = False):
        if is_progress:
            progress = float(np.clip(time_or_progress, 0.0, 1.0))
            target = int(round(progress * (self.total_steps - 1)))
        else:
            target = int(time_or_progress)
        self.index = self._clamp_index(target)
        self.scheduler.seek_truth_index(self.index)
        self.accumulated_time_s = 0.0
        self._emit_current()

    def set_speed(self, speed: float):
        self.playback_speed = float(np.clip(speed, 0.1, 20.0))

    def get_progress(self) -> float:
        if self.total_steps <= 1:
            return 0.0
        return float(self.index / (self.total_steps - 1))

    def get_time_info(self) -> Dict[str, float]:
        t_final = self.scheduler.t_final_s
        current_time = self.scheduler.current_time_s()
        return {
            'current_time': current_time,
            't_final': t_final,
            'progress': self.get_progress(),
            'remaining_time': max(0.0, t_final - current_time),
        }
