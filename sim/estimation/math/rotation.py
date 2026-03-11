"""Rotation-space helpers shared by the additive estimator rewrite."""

from __future__ import annotations

import numpy as np

from ..core.types import Matrix, Vector


def skew_symmetric(vector: Vector) -> Matrix:
    """Return the skew-symmetric matrix ``[v]_x`` for a 3-vector."""

    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
