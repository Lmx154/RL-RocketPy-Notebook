"""GPS velocity measurement model for the phase-5 navigation slice."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.types import Matrix, Vector
from ..models.navigation import NavigationState


@dataclass(slots=True)
class GpsVelocityConfig:
    """Configuration for velocity measurements expressed in meters per second."""

    measurement_std_mps: Vector = field(
        default_factory=lambda: np.array([3.0, 3.0, 4.0], dtype=float)
    )


class GpsVelocityMeasurementModel:
    """Nonlinear measurement-model wrapper for velocity observations."""

    label = "gps_velocity"

    def __init__(self, config: GpsVelocityConfig | None = None) -> None:
        self.config = config or GpsVelocityConfig()

    def predict_measurement(self, nominal_state: NavigationState) -> Vector:
        return np.asarray(nominal_state.velocity_mps, dtype=float).copy()

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
        measurement_jacobian[:, 3:6] = np.eye(3, dtype=float)
        return measurement_jacobian

    def measurement_covariance(
        self,
        measurement: Vector,
        nominal_state: NavigationState,
    ) -> Matrix:
        del measurement, nominal_state

        measurement_std = np.broadcast_to(
            np.asarray(self.config.measurement_std_mps, dtype=float),
            (3,),
        )
        return np.diag(measurement_std ** 2)
