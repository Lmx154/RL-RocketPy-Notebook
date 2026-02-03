"""
Quaternion mathematics utilities for attitude representation.

Provides functions for:
- Converting quaternions to rotation matrices
- Converting quaternions to Euler angles
- Normalizing quaternions
- Interpolating between quaternions (SLERP)

Quaternion Convention:
    q = e0 + e1*i + e2*j + e3*k
    where e0 is the scalar (real) part
    Constraint: e0² + e1² + e2² + e3² = 1 (unit quaternion)
"""

import numpy as np
from typing import Tuple


def normalize_quaternion(e0: float, e1: float, e2: float, e3: float) -> Tuple[float, float, float, float]:
    """
    Normalize a quaternion to unit length.
    
    Args:
        e0, e1, e2, e3: Quaternion components (scalar, i, j, k)
    
    Returns:
        Tuple of normalized quaternion components (e0, e1, e2, e3)
    """
    magnitude = np.sqrt(e0**2 + e1**2 + e2**2 + e3**2)
    
    if magnitude < 1e-10:
        # Return identity quaternion if magnitude is too small
        return (1.0, 0.0, 0.0, 0.0)
    
    return (e0 / magnitude, e1 / magnitude, e2 / magnitude, e3 / magnitude)


def quaternion_to_matrix(e0: float, e1: float, e2: float, e3: float) -> np.ndarray:
    """
    Convert unit quaternion to 3x3 rotation matrix.
    
    The rotation matrix R converts vectors from body frame to inertial frame:
        v_inertial = R @ v_body
    
    Args:
        e0, e1, e2, e3: Unit quaternion components (scalar, i, j, k)
    
    Returns:
        3x3 rotation matrix as numpy array
    
    Note:
        Assumes quaternion is already normalized. If not sure, use normalize_quaternion first.
    """
    # Standard quaternion to rotation matrix formula
    R = np.array([
        [1 - 2*(e2**2 + e3**2),   2*(e1*e2 - e0*e3),     2*(e1*e3 + e0*e2)],
        [2*(e1*e2 + e0*e3),       1 - 2*(e1**2 + e3**2), 2*(e2*e3 - e0*e1)],
        [2*(e1*e3 - e0*e2),       2*(e2*e3 + e0*e1),     1 - 2*(e1**2 + e2**2)]
    ])
    
    return R


def quaternion_to_euler(e0: float, e1: float, e2: float, e3: float) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (phi, theta, psi) in degrees.
    
    Uses 3-2-3 (Z-Y-Z) rotation sequence (NASA standard for rockets).
    
    Args:
        e0, e1, e2, e3: Unit quaternion components (scalar, i, j, k)
    
    Returns:
        Tuple of Euler angles (phi, theta, psi) in degrees:
        - phi: Spin angle (rotation about Z)
        - theta: Nutation angle (rotation about Y')
        - psi: Precession angle (rotation about Z'')
    
    Warning:
        Euler angles suffer from gimbal lock. Use quaternions for computation,
        Euler angles only for human-readable display.
    """
    # Convert to rotation matrix first
    R = quaternion_to_matrix(e0, e1, e2, e3)
    
    # Extract 3-2-3 Euler angles from rotation matrix
    # theta: nutation angle
    theta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
    
    if np.abs(np.sin(theta)) < 1e-10:
        # Gimbal lock case
        phi = 0.0
        psi = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Normal case
        phi = np.arctan2(R[2, 1], R[2, 0])
        psi = np.arctan2(R[1, 2], -R[0, 2])
    
    # Convert to degrees
    phi_deg = np.degrees(phi)
    theta_deg = np.degrees(theta)
    psi_deg = np.degrees(psi)
    
    return (phi_deg, theta_deg, psi_deg)


def interpolate_quaternion(
    q1: Tuple[float, float, float, float],
    q2: Tuple[float, float, float, float],
    t: float
) -> Tuple[float, float, float, float]:
    """
    Spherical linear interpolation (SLERP) between two quaternions.
    
    Provides smooth interpolation along the shortest path on the quaternion
    unit sphere. Useful for smooth animation between attitude states.
    
    Args:
        q1: First quaternion (e0, e1, e2, e3)
        q2: Second quaternion (e0, e1, e2, e3)
        t: Interpolation parameter in [0, 1]
            t=0 returns q1, t=1 returns q2
    
    Returns:
        Interpolated quaternion (e0, e1, e2, e3)
    """
    # Ensure t is in [0, 1]
    t = np.clip(t, 0.0, 1.0)
    
    # Normalize inputs
    q1 = normalize_quaternion(*q1)
    q2 = normalize_quaternion(*q2)
    
    # Compute dot product
    dot = sum(a * b for a, b in zip(q1, q2))
    
    # If dot product is negative, negate q2 to take shorter path
    if dot < 0.0:
        q2 = tuple(-x for x in q2)
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = tuple(
            (1.0 - t) * a + t * b
            for a, b in zip(q1, q2)
        )
        return normalize_quaternion(*result)
    
    # Spherical interpolation
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    
    scale1 = np.sin((1.0 - t) * theta) / sin_theta
    scale2 = np.sin(t * theta) / sin_theta
    
    result = tuple(
        scale1 * a + scale2 * b
        for a, b in zip(q1, q2)
    )
    
    return normalize_quaternion(*result)


def quaternion_inverse(e0: float, e1: float, e2: float, e3: float) -> Tuple[float, float, float, float]:
    """
    Compute the inverse (conjugate) of a unit quaternion.
    
    For unit quaternions, inverse = conjugate.
    The inverse quaternion represents the opposite rotation.
    
    Args:
        e0, e1, e2, e3: Unit quaternion components (scalar, i, j, k)
    
    Returns:
        Inverse quaternion (e0, -e1, -e2, -e3)
    """
    return (e0, -e1, -e2, -e3)


def rotate_vector(v: np.ndarray, e0: float, e1: float, e2: float, e3: float) -> np.ndarray:
    """
    Rotate a 3D vector using a quaternion.
    
    Equivalent to: R @ v where R is the rotation matrix.
    
    Args:
        v: 3D vector to rotate (numpy array or list of 3 elements)
        e0, e1, e2, e3: Unit quaternion components (scalar, i, j, k)
    
    Returns:
        Rotated vector as numpy array
    """
    v = np.asarray(v)
    R = quaternion_to_matrix(e0, e1, e2, e3)
    return R @ v
