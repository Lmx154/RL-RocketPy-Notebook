"""Barometric altitude measurement model for the phase-5 navigation slice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import Matrix, Vector
from ..models.navigation import NavigationState


@dataclass(slots=True)
class BarometricAltitudeConfig:
    """Configuration for barometric altitude updates."""

    measurement_std_m: float = 2.0
    axis: int = 2


class BarometricAltitudeMeasurementModel:
    """Predict scalar altitude from the navigation position state."""

    label = "baro_altitude"

    def __init__(self, config: BarometricAltitudeConfig | None = None) -> None:
        self.config = config or BarometricAltitudeConfig()

    def predict_measurement(self, nominal_state: NavigationState) -> Vector:
        return np.array([float(nominal_state.position_m[self.config.axis])], dtype=float)

    def innovation(
        self,
        measurement: float,
        predicted_measurement: Vector,
    ) -> Vector:
        return np.array(
            [float(measurement) - float(np.asarray(predicted_measurement, dtype=float)[0])],
            dtype=float,
        )

    def measurement_jacobian(
        self,
        measurement: float,
        nominal_state: NavigationState,
    ) -> Matrix:
        del measurement, nominal_state

        measurement_jacobian = np.zeros((1, 9), dtype=float)
        measurement_jacobian[0, self.config.axis] = 1.0
        return measurement_jacobian

    def measurement_covariance(
        self,
        measurement: float,
        nominal_state: NavigationState,
    ) -> Matrix:
        del measurement, nominal_state

        measurement_variance = self.config.measurement_std_m ** 2
        return np.array([[measurement_variance]], dtype=float)
