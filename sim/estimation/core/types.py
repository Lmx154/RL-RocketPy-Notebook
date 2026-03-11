"""Shared type aliases for the additive estimator rewrite."""

from __future__ import annotations

from typing import TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

Vector: TypeAlias = npt.NDArray[np.float64]
Matrix: TypeAlias = npt.NDArray[np.float64]

StateT = TypeVar("StateT")
ControlT = TypeVar("ControlT")
MeasurementT = TypeVar("MeasurementT")
