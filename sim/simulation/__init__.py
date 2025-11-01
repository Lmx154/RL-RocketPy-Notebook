"""
Simulation module for RocketPy flight playback.

Provides playback control for pre-computed RocketPy Flight trajectories.
"""

from .simulation_controller import SimulationController
from .quaternion_utils import (
    quaternion_to_matrix,
    quaternion_to_euler,
    normalize_quaternion,
    interpolate_quaternion
)

__all__ = [
    'SimulationController',
    'quaternion_to_matrix',
    'quaternion_to_euler',
    'normalize_quaternion',
    'interpolate_quaternion',
]
