"""GPS position measurement model for the phase-5 navigation slice."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.types import Matrix, Vector
from ..models.navigation import NavigationState


@dataclass(slots=True)
class GpsPositionConfig:
    """Configuration for position measurements expressed in meters."""

    measurement_std_m: Vector = field(
        default_factory=lambda: np.array([3.0, 3.0, 5.0], dtype=float)
    )


class GpsPositionMeasurementModel:
    """Nonlinear measurement-model wrapper for position observations."""

    label = "gps_position"

    def __init__(self, config: GpsPositionConfig | None = None) -> None:
        self.config = config or GpsPositionConfig()

    def predict_measurement(self, nominal_state: NavigationState) -> Vector:
        return np.asarray(nominal_state.position_m, dtype=float).copy()

    def innovation(
        self,
        measurement: Vector,
        predicted_measurement: Vector,
    ) -> Vector:
        return np.asarray(measurement, dtype=float) - np.asarray(predicted_measurement, dtype=float)

    def measurement_jacobian(
        self,
        measurement: Vector,
        nominal_state: NavigationState,
    ) -> Matrix:
        del measurement, nominal_state

        measurement_jacobian = np.zeros((3, 9), dtype=float)
        measurement_jacobian[:, 0:3] = np.eye(3, dtype=float)
        return measurement_jacobian

    def measurement_covariance(
        self,
        measurement: Vector,
        nominal_state: NavigationState,
    ) -> Matrix:
        del measurement, nominal_state

        measurement_std = np.broadcast_to(
            np.asarray(self.config.measurement_std_m, dtype=float),
            (3,),
        )
        return np.diag(measurement_std ** 2)
