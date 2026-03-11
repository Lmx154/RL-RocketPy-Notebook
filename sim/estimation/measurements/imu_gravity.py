"""Gravity-alignment accelerometer measurement model for the phase-4 slice."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.types import Matrix, Vector
from ..math import quaternion_to_rotation_matrix, skew_symmetric
from ..models.attitude import AttitudeState


@dataclass(slots=True)
class GravityAlignmentConfig:
    """Configuration for gravity-alignment attitude updates."""

    gravity_vector: Vector = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.80665], dtype=float)
    )
    measurement_std_mps2: float = 0.5


class GravityAlignmentMeasurementModel:
    """Predict body-frame gravity from the current attitude estimate."""

    label = "gravity_alignment"

    def __init__(self, config: GravityAlignmentConfig | None = None) -> None:
        self.config = config or GravityAlignmentConfig()

    def predict_measurement(self, nominal_state: AttitudeState) -> Vector:
        rotation_body_to_inertial = quaternion_to_rotation_matrix(nominal_state.quaternion)
        return rotation_body_to_inertial.T @ np.asarray(self.config.gravity_vector, dtype=float)

    def innovation(
        self,
        measurement: Vector,
        predicted_measurement: Vector,
    ) -> Vector:
        return np.asarray(measurement, dtype=float) - np.asarray(predicted_measurement, dtype=float)

    def measurement_jacobian(
        self,
        measurement: Vector,
        nominal_state: AttitudeState,
    ) -> Matrix:
        del measurement

        predicted_measurement = self.predict_measurement(nominal_state)
        measurement_jacobian = np.zeros((3, 6), dtype=float)
        # With the model's right-multiplicative attitude error convention,
        # finite differences give +[zhat]_x in the attitude block.
        measurement_jacobian[:, 0:3] = skew_symmetric(predicted_measurement)
        return measurement_jacobian

    def measurement_covariance(
        self,
        measurement: Vector,
        nominal_state: AttitudeState,
    ) -> Matrix:
        del measurement, nominal_state

        return np.eye(3, dtype=float) * (self.config.measurement_std_mps2 ** 2)
