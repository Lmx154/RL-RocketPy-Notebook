"""Finite-difference Jacobian helpers for estimator math tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.types import Matrix, Vector


def finite_difference_jacobian(
    function: Callable[[Vector], Vector],
    point: Vector,
    step: float = 1e-6,
) -> Matrix:
    """Approximate ``df/dx`` with a central finite difference stencil."""

    point = np.asarray(point, dtype=float)
    baseline = np.asarray(function(point), dtype=float)
    jacobian = np.zeros((baseline.size, point.size), dtype=float)

    for column in range(point.size):
        offset = np.zeros_like(point)
        offset[column] = step
        forward = np.asarray(function(point + offset), dtype=float)
        backward = np.asarray(function(point - offset), dtype=float)
        jacobian[:, column] = (forward - backward) / (2.0 * step)

    return jacobian
