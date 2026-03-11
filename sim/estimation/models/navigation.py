"""Navigation process model for the phase-5 vertical slice."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.types import Matrix, Vector


@dataclass(slots=True)
class NavigationProcessNoise:
    """Accelerometer and accelerometer-bias noise parameters."""

    accelerometer_noise_std: float = 8.0e-2
    accel_bias_random_walk_std: float = 2.0e-2


@dataclass(slots=True)
class NavigationProcessModelConfig:
    """Configuration for the navigation EKF process model."""

    gravity_vector: Vector = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.80665], dtype=float)
    )
    accelerometer_includes_gravity: bool = True
    process_noise: NavigationProcessNoise = field(default_factory=NavigationProcessNoise)


@dataclass(slots=True)
class NavigationInput:
    """Accelerometer sample plus externally supplied attitude rotation."""

    accelerometer_mps2: Vector
    rotation_body_to_inertial: Matrix

    def copy(self) -> "NavigationInput":
        return NavigationInput(
            accelerometer_mps2=np.asarray(self.accelerometer_mps2, dtype=float).copy(),
            rotation_body_to_inertial=np.asarray(
                self.rotation_body_to_inertial,
                dtype=float,
            ).copy(),
        )


@dataclass(slots=True)
class NavigationState:
    """Euclidean navigation state for the phase-5 EKF slice."""

    position_m: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity_mps: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    accel_bias_mps2: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))

    def copy(self) -> "NavigationState":
        return NavigationState(
            position_m=np.asarray(self.position_m, dtype=float).copy(),
            velocity_mps=np.asarray(self.velocity_mps, dtype=float).copy(),
            accel_bias_mps2=np.asarray(self.accel_bias_mps2, dtype=float).copy(),
        )

    def plus(self, delta: Vector) -> "NavigationState":
        delta = np.asarray(delta, dtype=float)
        return NavigationState(
            position_m=np.asarray(self.position_m, dtype=float) + delta[0:3],
            velocity_mps=np.asarray(self.velocity_mps, dtype=float) + delta[3:6],
            accel_bias_mps2=np.asarray(self.accel_bias_mps2, dtype=float) + delta[6:9],
        )


class NavigationProcessModel:
    """Strapdown-driven navigation process model over ``[p, v, b_a]``.

    The model consumes an externally supplied body-to-inertial rotation so the
    navigation layer stays decoupled from the attitude filter implementation.
    """

    state_dimension = 9
    noise_dimension = 9

    def __init__(self, config: NavigationProcessModelConfig | None = None) -> None:
        self.config = config or NavigationProcessModelConfig()

    def predict(
        self,
        nominal_state: NavigationState,
        control: NavigationInput,
        dt: float,
    ) -> NavigationState:
        dt = float(dt)
        if dt <= 0.0:
            return nominal_state.copy()

        inertial_acceleration = self.inertial_acceleration(nominal_state, control)
        predicted_state = nominal_state.copy()
        predicted_state.position_m = (
            predicted_state.position_m
            + predicted_state.velocity_mps * dt
            + 0.5 * inertial_acceleration * dt * dt
        )
        predicted_state.velocity_mps = predicted_state.velocity_mps + inertial_acceleration * dt
        return predicted_state

    def corrected_acceleration_body(
        self,
        nominal_state: NavigationState,
        control: NavigationInput,
    ) -> Vector:
        return np.asarray(control.accelerometer_mps2, dtype=float) - np.asarray(
            nominal_state.accel_bias_mps2,
            dtype=float,
        )

    def inertial_acceleration(
        self,
        nominal_state: NavigationState,
        control: NavigationInput,
    ) -> Vector:
        corrected_acceleration = self.corrected_acceleration_body(nominal_state, control)
        inertial_acceleration = self.rotation_body_to_inertial(control) @ corrected_acceleration
        if self.config.accelerometer_includes_gravity:
            return inertial_acceleration
        return inertial_acceleration + np.asarray(self.config.gravity_vector, dtype=float)

    def rotation_body_to_inertial(self, control: NavigationInput) -> Matrix:
        return np.asarray(control.rotation_body_to_inertial, dtype=float)

    def state_jacobian(
        self,
        nominal_state: NavigationState,
        control: NavigationInput,
        dt: float,
    ) -> Matrix:
        del nominal_state

        dt = float(dt)
        rotation_body_to_inertial = self.rotation_body_to_inertial(control)

        transition = np.eye(self.state_dimension, dtype=float)
        transition[0:3, 3:6] = np.eye(3, dtype=float) * dt
        transition[0:3, 6:9] = -0.5 * rotation_body_to_inertial * dt * dt
        transition[3:6, 6:9] = -rotation_body_to_inertial * dt
        return transition

    def process_noise_jacobian(
        self,
        nominal_state: NavigationState,
        control: NavigationInput,
        dt: float,
    ) -> Matrix:
        del nominal_state, dt

        rotation_body_to_inertial = self.rotation_body_to_inertial(control)
        noise_jacobian = np.zeros((self.state_dimension, self.noise_dimension), dtype=float)
        noise_jacobian[0:3, 0:3] = rotation_body_to_inertial
        noise_jacobian[3:6, 3:6] = rotation_body_to_inertial
        noise_jacobian[6:9, 6:9] = np.eye(3, dtype=float)
        return noise_jacobian

    def process_noise_covariance(
        self,
        nominal_state: NavigationState,
        control: NavigationInput,
        dt: float,
    ) -> Matrix:
        del nominal_state, control

        dt = max(float(dt), 0.0)
        process_noise = self.config.process_noise
        accelerometer_variance = process_noise.accelerometer_noise_std ** 2
        bias_variance = process_noise.accel_bias_random_walk_std ** 2
        identity = np.eye(3, dtype=float)

        process_noise_covariance = np.zeros((self.noise_dimension, self.noise_dimension), dtype=float)
        process_noise_covariance[0:3, 0:3] = identity * (accelerometer_variance * dt ** 3 / 3.0)
        process_noise_covariance[0:3, 3:6] = identity * (accelerometer_variance * dt ** 2 / 2.0)
        process_noise_covariance[3:6, 0:3] = identity * (accelerometer_variance * dt ** 2 / 2.0)
        process_noise_covariance[3:6, 3:6] = identity * (accelerometer_variance * dt)
        process_noise_covariance[6:9, 6:9] = identity * (bias_variance * dt)
        return process_noise_covariance
