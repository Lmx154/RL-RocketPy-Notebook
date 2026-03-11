from __future__ import annotations

import unittest

import numpy as np

from sim.estimation.math import quaternion_inverse, quaternion_multiply, rotation_vector_to_quaternion
from sim.estimation.models.strapdown import (
    StrapdownConfig,
    StrapdownInertialProcessModel,
    StrapdownInput,
    StrapdownState,
)


def quaternion_small_angle_error(
    reference_quaternion: np.ndarray,
    perturbed_quaternion: np.ndarray,
) -> np.ndarray:
    delta_quaternion = quaternion_multiply(
        quaternion_inverse(reference_quaternion),
        perturbed_quaternion,
    )
    if delta_quaternion[0] < 0.0:
        delta_quaternion = -delta_quaternion
    return 2.0 * np.asarray(delta_quaternion[1:], dtype=float)


def state_error_vector(reference: StrapdownState, perturbed: StrapdownState) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(perturbed.position_m, dtype=float) - np.asarray(reference.position_m, dtype=float),
            np.asarray(perturbed.velocity_mps, dtype=float) - np.asarray(reference.velocity_mps, dtype=float),
            quaternion_small_angle_error(reference.quaternion, perturbed.quaternion),
        ]
    )


class StrapdownModelTests(unittest.TestCase):
    def test_constant_angular_rate_matches_closed_form_quaternion(self) -> None:
        model = StrapdownInertialProcessModel(
            StrapdownConfig(accelerometer_includes_gravity=True)
        )
        initial_state = StrapdownState()
        corrected_rate = np.array([0.2, -0.1, 0.3], dtype=float)
        control = StrapdownInput(
            accelerometer_mps2=np.zeros(3, dtype=float),
            gyroscope_rps=corrected_rate + np.array([0.01, -0.02, 0.015], dtype=float),
            gyro_bias_rps=np.array([0.01, -0.02, 0.015], dtype=float),
        )
        dt = 0.4

        predicted = model.predict(initial_state, control, dt)
        expected_quaternion = rotation_vector_to_quaternion(corrected_rate * dt)

        np.testing.assert_allclose(predicted.quaternion, expected_quaternion, atol=1e-12)
        np.testing.assert_allclose(predicted.position_m, np.zeros(3, dtype=float), atol=1e-12)
        np.testing.assert_allclose(predicted.velocity_mps, np.zeros(3, dtype=float), atol=1e-12)

    def test_constant_acceleration_matches_closed_form_kinematics(self) -> None:
        model = StrapdownInertialProcessModel(
            StrapdownConfig(accelerometer_includes_gravity=True)
        )
        initial_state = StrapdownState()
        inertial_acceleration = np.array([3.0, -2.0, 1.5], dtype=float)
        control = StrapdownInput(
            accelerometer_mps2=inertial_acceleration + np.array([0.1, -0.2, 0.05], dtype=float),
            gyroscope_rps=np.zeros(3, dtype=float),
            accel_bias_mps2=np.array([0.1, -0.2, 0.05], dtype=float),
        )
        dt = 0.6

        predicted = model.predict(initial_state, control, dt)

        np.testing.assert_allclose(
            predicted.velocity_mps,
            inertial_acceleration * dt,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            predicted.position_m,
            0.5 * inertial_acceleration * dt * dt,
            atol=1e-12,
        )

    def test_specific_force_convention_recovers_inertial_acceleration(self) -> None:
        model = StrapdownInertialProcessModel(
            StrapdownConfig(
                accelerometer_includes_gravity=False,
                gravity_vector=np.array([0.0, 0.0, -9.80665], dtype=float),
            )
        )
        initial_state = StrapdownState()
        inertial_acceleration = np.array([0.7, -1.2, 0.3], dtype=float)
        specific_force = inertial_acceleration - model.config.gravity_vector
        control = StrapdownInput(
            accelerometer_mps2=specific_force + np.array([0.05, -0.03, 0.02], dtype=float),
            gyroscope_rps=np.zeros(3, dtype=float),
            accel_bias_mps2=np.array([0.05, -0.03, 0.02], dtype=float),
        )

        recovered = model.inertial_acceleration(initial_state, control)
        np.testing.assert_allclose(recovered, inertial_acceleration, atol=1e-12)

    def test_discrete_error_transition_matches_first_order_and_finite_difference(self) -> None:
        model = StrapdownInertialProcessModel(
            StrapdownConfig(accelerometer_includes_gravity=False)
        )
        nominal_state = StrapdownState(
            position_m=np.array([10.0, -3.0, 5.0], dtype=float),
            velocity_mps=np.array([2.5, -1.0, 0.3], dtype=float),
            quaternion=rotation_vector_to_quaternion(np.array([0.15, -0.1, 0.05], dtype=float)),
        )
        control = StrapdownInput(
            accelerometer_mps2=np.array([0.6, -0.2, 9.9], dtype=float),
            gyroscope_rps=np.array([0.2, -0.1, 0.15], dtype=float),
            gyro_bias_rps=np.array([0.01, -0.02, 0.005], dtype=float),
            accel_bias_mps2=np.array([0.03, -0.04, 0.02], dtype=float),
        )
        dt = 1.0e-4
        epsilon = 1.0e-7

        continuous_jacobian = model.continuous_error_dynamics_jacobian(nominal_state, control, dt)
        discrete_jacobian = model.error_state_jacobian(nominal_state, control, dt)
        np.testing.assert_allclose(
            discrete_jacobian,
            np.eye(model.state_dimension, dtype=float) + continuous_jacobian * dt,
            atol=1e-15,
        )

        nominal_prediction = model.predict(nominal_state, control, dt)
        finite_difference_jacobian = np.zeros_like(discrete_jacobian)
        for column in range(model.state_dimension):
            perturbation = np.zeros(model.state_dimension, dtype=float)
            perturbation[column] = epsilon
            perturbed_state = model.inject(nominal_state, perturbation)
            perturbed_prediction = model.predict(perturbed_state, control, dt)
            finite_difference_jacobian[:, column] = (
                state_error_vector(nominal_prediction, perturbed_prediction) / epsilon
            )

        np.testing.assert_allclose(
            finite_difference_jacobian,
            discrete_jacobian,
            atol=5.0e-5,
            rtol=5.0e-4,
        )


if __name__ == "__main__":
    unittest.main()
