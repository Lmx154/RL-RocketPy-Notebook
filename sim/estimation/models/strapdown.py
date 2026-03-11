"""Strapdown inertial propagation model for the additive estimator rewrite."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.types import Matrix, Vector
from ..math import (
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_vector_to_quaternion,
    skew_symmetric,
)


@dataclass(slots=True)
class StrapdownProcessNoise:
    """Discrete-time IMU noise parameters used by the strapdown model."""

    accelerometer_noise_std: float = 8.0e-2
    gyroscope_noise_std: float = 2.0e-3


@dataclass(slots=True)
class StrapdownConfig:
    """Configuration for nominal strapdown inertial propagation."""

    gravity_vector: Vector = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.80665], dtype=float)
    )
    accelerometer_includes_gravity: bool = True
    process_noise: StrapdownProcessNoise = field(default_factory=StrapdownProcessNoise)


@dataclass(slots=True)
class StrapdownInput:
    """IMU sample plus externally supplied bias corrections."""

    accelerometer_mps2: Vector
    gyroscope_rps: Vector
    gyro_bias_rps: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    accel_bias_mps2: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))

    def copy(self) -> "StrapdownInput":
        return StrapdownInput(
            accelerometer_mps2=np.asarray(self.accelerometer_mps2, dtype=float).copy(),
            gyroscope_rps=np.asarray(self.gyroscope_rps, dtype=float).copy(),
            gyro_bias_rps=np.asarray(self.gyro_bias_rps, dtype=float).copy(),
            accel_bias_mps2=np.asarray(self.accel_bias_mps2, dtype=float).copy(),
        )


@dataclass(slots=True)
class StrapdownState:
    """Nominal inertial state propagated by the strapdown model."""

    position_m: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity_mps: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    quaternion: Vector = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    )

    def copy(self) -> "StrapdownState":
        return StrapdownState(
            position_m=np.asarray(self.position_m, dtype=float).copy(),
            velocity_mps=np.asarray(self.velocity_mps, dtype=float).copy(),
            quaternion=np.asarray(self.quaternion, dtype=float).copy(),
        )


class StrapdownInertialProcessModel:
    """Nominal strapdown propagation plus first-order error dynamics.

    The nominal state is ``[p, v, q]`` while bias corrections arrive through
    the control input so later attitude and navigation models can decide how
    bias states are estimated without hard-coding them into this propagator.
    """

    state_dimension = 9
    noise_dimension = 6

    def __init__(self, config: StrapdownConfig | None = None) -> None:
        self.config = config or StrapdownConfig()

    def predict(
        self,
        nominal_state: StrapdownState,
        control: StrapdownInput,
        dt: float,
    ) -> StrapdownState:
        dt = float(dt)
        if dt <= 0.0:
            return nominal_state.copy()

        inertial_acceleration = self.inertial_acceleration(nominal_state, control)
        corrected_rate = self.corrected_angular_rate(control)
        delta_quaternion = rotation_vector_to_quaternion(corrected_rate * dt)

        predicted_state = nominal_state.copy()
        predicted_state.position_m = (
            predicted_state.position_m
            + predicted_state.velocity_mps * dt
            + 0.5 * inertial_acceleration * dt * dt
        )
        predicted_state.velocity_mps = predicted_state.velocity_mps + inertial_acceleration * dt
        predicted_state.quaternion = normalize_quaternion(
            quaternion_multiply(predicted_state.quaternion, delta_quaternion)
        )
        return predicted_state

    def corrected_angular_rate(self, control: StrapdownInput) -> Vector:
        return np.asarray(control.gyroscope_rps, dtype=float) - np.asarray(
            control.gyro_bias_rps,
            dtype=float,
        )

    def corrected_acceleration_body(self, control: StrapdownInput) -> Vector:
        return np.asarray(control.accelerometer_mps2, dtype=float) - np.asarray(
            control.accel_bias_mps2,
            dtype=float,
        )

    def rotation_body_to_inertial(self, nominal_state: StrapdownState) -> Matrix:
        return quaternion_to_rotation_matrix(nominal_state.quaternion)

    def inertial_acceleration(
        self,
        nominal_state: StrapdownState,
        control: StrapdownInput,
    ) -> Vector:
        corrected_acceleration = self.corrected_acceleration_body(control)
        inertial_acceleration = self.rotation_body_to_inertial(nominal_state) @ corrected_acceleration
        if self.config.accelerometer_includes_gravity:
            return inertial_acceleration
        return inertial_acceleration + np.asarray(self.config.gravity_vector, dtype=float)

    def continuous_error_dynamics_jacobian(
        self,
        nominal_state: StrapdownState,
        control: StrapdownInput,
        dt: float,
    ) -> Matrix:
        del dt

        corrected_rate = self.corrected_angular_rate(control)
        corrected_acceleration = self.corrected_acceleration_body(control)
        rotation_body_to_inertial = self.rotation_body_to_inertial(nominal_state)

        system_matrix = np.zeros((self.state_dimension, self.state_dimension), dtype=float)
        system_matrix[0:3, 3:6] = np.eye(3, dtype=float)
        system_matrix[3:6, 6:9] = -rotation_body_to_inertial @ skew_symmetric(corrected_acceleration)
        system_matrix[6:9, 6:9] = -skew_symmetric(corrected_rate)
        return system_matrix

    def error_state_jacobian(
        self,
        nominal_state: StrapdownState,
        control: StrapdownInput,
        dt: float,
    ) -> Matrix:
        dt = float(dt)
        return np.eye(self.state_dimension, dtype=float) + (
            self.continuous_error_dynamics_jacobian(nominal_state, control, dt) * dt
        )

    def process_noise_jacobian(
        self,
        nominal_state: StrapdownState,
        control: StrapdownInput,
        dt: float,
    ) -> Matrix:
        del control, dt

        rotation_body_to_inertial = self.rotation_body_to_inertial(nominal_state)
        noise_jacobian = np.zeros((self.state_dimension, self.noise_dimension), dtype=float)
        noise_jacobian[3:6, 0:3] = rotation_body_to_inertial
        noise_jacobian[6:9, 3:6] = -np.eye(3, dtype=float)
        return noise_jacobian

    def process_noise_covariance(
        self,
        nominal_state: StrapdownState,
        control: StrapdownInput,
        dt: float,
    ) -> Matrix:
        del nominal_state, control

        dt = max(float(dt), 0.0)
        process_noise = self.config.process_noise
        return np.diag(
            np.array(
                [
                    *([process_noise.accelerometer_noise_std ** 2 * dt] * 3),
                    *([process_noise.gyroscope_noise_std ** 2 * dt] * 3),
                ],
                dtype=float,
            )
        )

    def inject(self, nominal_state: StrapdownState, error_state: Vector) -> StrapdownState:
        error_state = np.asarray(error_state, dtype=float)
        injected_state = nominal_state.copy()
        injected_state.position_m = injected_state.position_m + error_state[0:3]
        injected_state.velocity_mps = injected_state.velocity_mps + error_state[3:6]
        injected_state.quaternion = normalize_quaternion(
            quaternion_multiply(
                injected_state.quaternion,
                rotation_vector_to_quaternion(error_state[6:9]),
            )
        )
        return injected_state

    def reset_jacobian(self, injected_error_state: Vector) -> Matrix:
        injected_error_state = np.asarray(injected_error_state, dtype=float)
        reset = np.eye(self.state_dimension, dtype=float)
        reset[6:9, 6:9] -= 0.5 * skew_symmetric(injected_error_state[6:9])
        return reset
