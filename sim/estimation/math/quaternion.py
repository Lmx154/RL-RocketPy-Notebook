"""Quaternion helpers shared by the additive estimator rewrite."""

from __future__ import annotations

import numpy as np

from ..core.types import Matrix, Vector


def normalize_quaternion(quaternion: Vector, eps: float = 1e-12) -> Vector:
    """Return a unit quaternion, falling back to identity near zero norm."""

    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quaternion / norm


def quaternion_multiply(lhs: Vector, rhs: Vector) -> Vector:
    """Return the Hamilton product ``lhs ⊗ rhs``."""

    w1, x1, y1, z1 = np.asarray(lhs, dtype=float)
    w2, x2, y2, z2 = np.asarray(rhs, dtype=float)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_inverse(quaternion: Vector, eps: float = 1e-12) -> Vector:
    """Return the multiplicative quaternion inverse."""

    quaternion = np.asarray(quaternion, dtype=float)
    norm_sq = float(np.dot(quaternion, quaternion))
    if norm_sq < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    conjugate = quaternion.copy()
    conjugate[1:] *= -1.0
    return conjugate / norm_sq


def rotation_vector_to_quaternion(rotation_vector: Vector, eps: float = 1e-12) -> Vector:
    """Map a rotation vector in ``R^3`` to a unit quaternion on ``SO(3)``."""

    rotation_vector = np.asarray(rotation_vector, dtype=float)
    angle = float(np.linalg.norm(rotation_vector))
    if angle < eps:
        return normalize_quaternion(np.array([1.0, *(0.5 * rotation_vector)], dtype=float))

    axis = rotation_vector / angle
    half_angle = 0.5 * angle
    return np.array([np.cos(half_angle), *(axis * np.sin(half_angle))], dtype=float)


def quaternion_to_rotation_matrix(quaternion: Vector) -> Matrix:
    """Convert a quaternion to a 3x3 rotation matrix."""

    w, x, y, z = normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
