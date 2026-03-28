"""
Simulation module for RocketPy flight playback.

Provides playback control for pre-computed RocketPy Flight trajectories.
"""

from sim.sitl.session import ReplaySession

from .simulation_controller import (
    CsvReplayController,
    SimulationController,
    load_kinematics_csv,
    load_replay_session,
)
from .quaternion_utils import (
    quaternion_to_matrix,
    quaternion_to_euler,
    normalize_quaternion,
    interpolate_quaternion
)

__all__ = [
    'SimulationController',
    'CsvReplayController',
    'ReplaySession',
    'load_kinematics_csv',
    'load_replay_session',
    'quaternion_to_matrix',
    'quaternion_to_euler',
    'normalize_quaternion',
    'interpolate_quaternion',
]
