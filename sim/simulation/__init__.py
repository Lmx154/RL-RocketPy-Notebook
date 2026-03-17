"""
Simulation module for RocketPy flight playback.

Provides playback control for pre-computed RocketPy Flight trajectories.
"""

from .simulation_controller import (
    CsvReplayController,
    SimulationController,
    load_kinematics_csv,
    load_replay_pair,
    load_sensor_csv,
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
    'load_kinematics_csv',
    'load_sensor_csv',
    'load_replay_pair',
    'quaternion_to_matrix',
    'quaternion_to_euler',
    'normalize_quaternion',
    'interpolate_quaternion',
]
