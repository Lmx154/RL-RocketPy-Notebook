"""Attitude process model for the phase-4 vertical slice."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.types import Matrix, Vector
from ..math import (
    normalize_quaternion,
    quaternion_multiply,
    rotation_vector_to_quaternion,
    skew_symmetric,
)


@dataclass(slots=True)
class AttitudeProcessNoise:
    """Discrete-time gyro and gyro-bias noise parameters."""

    gyroscope_noise_std: float = 2.0e-3
    gyro_bias_random_walk_std: float = 2.0e-4


@dataclass(slots=True)
class AttitudeProcessModelConfig:
    """Configuration for the attitude ESKF process model."""

    process_noise: AttitudeProcessNoise = field(default_factory=AttitudeProcessNoise)


@dataclass(slots=True)
class AttitudeInput:
    """Gyroscope sample submitted to the attitude process model."""

    gyroscope_rps: Vector

    def copy(self) -> "AttitudeInput":
        return AttitudeInput(gyroscope_rps=np.asarray(self.gyroscope_rps, dtype=float).copy())


@dataclass(slots=True)
class AttitudeState:
    """Nominal attitude state used by the phase-4 ESKF slice."""

    quaternion: Vector = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    )
    gyro_bias_rps: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))

    def copy(self) -> "AttitudeState":
        return AttitudeState(
            quaternion=np.asarray(self.quaternion, dtype=float).copy(),
            gyro_bias_rps=np.asarray(self.gyro_bias_rps, dtype=float).copy(),
        )


class AttitudeErrorStateProcessModel:
    """Quaternion attitude propagation with gyro-bias error-state dynamics.

    The model uses right-multiplicative quaternion injection:
    ``q <- q ⊗ Exp(dtheta)``.
    """

    state_dimension = 6
    noise_dimension = 6

    def __init__(self, config: AttitudeProcessModelConfig | None = None) -> None:
        self.config = config or AttitudeProcessModelConfig()

    def predict(
        self,
        nominal_state: AttitudeState,
        control: AttitudeInput,
        dt: float,
    ) -> AttitudeState:
        dt = float(dt)
        if dt <= 0.0:
            return nominal_state.copy()

        corrected_rate = self.corrected_angular_rate(nominal_state, control)
        predicted_state = nominal_state.copy()
        predicted_state.quaternion = normalize_quaternion(
            quaternion_multiply(
                predicted_state.quaternion,
                rotation_vector_to_quaternion(corrected_rate * dt),
            )
        )
        return predicted_state

    def corrected_angular_rate(
        self,
        nominal_state: AttitudeState,
        control: AttitudeInput,
    ) -> Vector:
        return np.asarray(control.gyroscope_rps, dtype=float) - np.asarray(
            nominal_state.gyro_bias_rps,
            dtype=float,
        )

    def continuous_error_dynamics_jacobian(
        self,
        nominal_state: AttitudeState,
        control: AttitudeInput,
        dt: float,
    ) -> Matrix:
        del dt

        corrected_rate = self.corrected_angular_rate(nominal_state, control)
        system_matrix = np.zeros((self.state_dimension, self.state_dimension), dtype=float)
        system_matrix[0:3, 0:3] = -skew_symmetric(corrected_rate)
        system_matrix[0:3, 3:6] = -np.eye(3, dtype=float)
        return system_matrix

    def error_state_jacobian(
        self,
        nominal_state: AttitudeState,
        control: AttitudeInput,
        dt: float,
    ) -> Matrix:
        dt = float(dt)
        return np.eye(self.state_dimension, dtype=float) + (
            self.continuous_error_dynamics_jacobian(nominal_state, control, dt) * dt
        )

    def process_noise_jacobian(
        self,
        nominal_state: AttitudeState,
        control: AttitudeInput,
        dt: float,
    ) -> Matrix:
        del nominal_state, control, dt

        noise_jacobian = np.zeros((self.state_dimension, self.noise_dimension), dtype=float)
        noise_jacobian[0:3, 0:3] = -np.eye(3, dtype=float)
        noise_jacobian[3:6, 3:6] = np.eye(3, dtype=float)
        return noise_jacobian

    def process_noise_covariance(
        self,
        nominal_state: AttitudeState,
        control: AttitudeInput,
        dt: float,
    ) -> Matrix:
        del nominal_state, control

        dt = max(float(dt), 0.0)
        process_noise = self.config.process_noise
        return np.diag(
            np.array(
                [
                    *([process_noise.gyroscope_noise_std ** 2 * dt] * 3),
                    *([process_noise.gyro_bias_random_walk_std ** 2 * dt] * 3),
                ],
                dtype=float,
            )
        )

    def inject(self, nominal_state: AttitudeState, error_state: Vector) -> AttitudeState:
        error_state = np.asarray(error_state, dtype=float)
        injected_state = nominal_state.copy()
        injected_state.quaternion = normalize_quaternion(
            quaternion_multiply(
                injected_state.quaternion,
                rotation_vector_to_quaternion(error_state[0:3]),
            )
        )
        injected_state.gyro_bias_rps = injected_state.gyro_bias_rps + error_state[3:6]
        return injected_state

    def reset_jacobian(self, injected_error_state: Vector) -> Matrix:
        injected_error_state = np.asarray(injected_error_state, dtype=float)
        reset = np.eye(self.state_dimension, dtype=float)
        reset[0:3, 0:3] -= 0.5 * skew_symmetric(injected_error_state[0:3])
        return reset
