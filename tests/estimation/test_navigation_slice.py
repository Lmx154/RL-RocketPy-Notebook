from __future__ import annotations

import unittest

import numpy as np

from sim.estimation.core import CovarianceConvention, GenericEKF, MeasurementUpdateStatus
from sim.estimation.math import (
    finite_difference_jacobian,
    quaternion_to_rotation_matrix,
    rotation_vector_to_quaternion,
)
from sim.estimation.measurements import (
    BarometricAltitudeConfig,
    BarometricAltitudeMeasurementModel,
    GpsPositionConfig,
    GpsPositionMeasurementModel,
    GpsVelocityConfig,
    GpsVelocityMeasurementModel,
)
from sim.estimation.models import (
    NavigationInput,
    NavigationProcessModel,
    NavigationProcessModelConfig,
    NavigationProcessNoise,
    NavigationState,
)


def state_vector(state: NavigationState) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(state.position_m, dtype=float),
            np.asarray(state.velocity_mps, dtype=float),
            np.asarray(state.accel_bias_mps2, dtype=float),
        ]
    )


class NavigationSliceTests(unittest.TestCase):
    def test_constant_velocity_propagation_matches_closed_form_kinematics(self) -> None:
        process_model = NavigationProcessModel(
            NavigationProcessModelConfig(accelerometer_includes_gravity=True)
        )
        initial_state = NavigationState(
            position_m=np.array([10.0, -3.0, 5.0], dtype=float),
            velocity_mps=np.array([2.5, -1.0, 0.3], dtype=float),
            accel_bias_mps2=np.array([0.12, -0.08, 0.05], dtype=float),
        )
        estimator = GenericEKF(
            process_model=process_model,
            initial_state=initial_state,
            initial_covariance=np.diag(np.full(9, 1.0e-6, dtype=float)),
            covariance_convention=CovarianceConvention(min_variance=1.0e-12),
        )
        control = NavigationInput(
            accelerometer_mps2=initial_state.accel_bias_mps2.copy(),
            rotation_body_to_inertial=np.eye(3, dtype=float),
        )
        dt = 0.4

        estimator.predict(control=control, dt=dt)

        np.testing.assert_allclose(
            estimator.state.position_m,
            initial_state.position_m + initial_state.velocity_mps * dt,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            estimator.state.velocity_mps,
            initial_state.velocity_mps,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            estimator.state.accel_bias_mps2,
            initial_state.accel_bias_mps2,
            atol=1.0e-12,
        )

    def test_constant_acceleration_propagation_uses_supplied_rotation(self) -> None:
        process_model = NavigationProcessModel(
            NavigationProcessModelConfig(accelerometer_includes_gravity=True)
        )
        initial_state = NavigationState(
            position_m=np.array([1.0, 2.0, -4.0], dtype=float),
            velocity_mps=np.array([3.0, -2.0, 0.5], dtype=float),
            accel_bias_mps2=np.array([0.2, -0.1, 0.05], dtype=float),
        )
        corrected_acceleration_body = np.array([1.2, -0.4, 0.8], dtype=float)
        rotation_body_to_inertial = quaternion_to_rotation_matrix(
            rotation_vector_to_quaternion(np.array([0.25, -0.15, 0.3], dtype=float))
        )
        control = NavigationInput(
            accelerometer_mps2=corrected_acceleration_body + initial_state.accel_bias_mps2,
            rotation_body_to_inertial=rotation_body_to_inertial,
        )
        dt = 0.6

        predicted = process_model.predict(initial_state, control, dt)
        inertial_acceleration = rotation_body_to_inertial @ corrected_acceleration_body

        np.testing.assert_allclose(
            predicted.velocity_mps,
            initial_state.velocity_mps + inertial_acceleration * dt,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            predicted.position_m,
            initial_state.position_m
            + initial_state.velocity_mps * dt
            + 0.5 * inertial_acceleration * dt * dt,
            atol=1.0e-12,
        )

    def test_navigation_state_jacobian_matches_finite_difference(self) -> None:
        process_model = NavigationProcessModel(
            NavigationProcessModelConfig(accelerometer_includes_gravity=True)
        )
        nominal_state = NavigationState(
            position_m=np.array([2.0, -1.0, 5.0], dtype=float),
            velocity_mps=np.array([0.8, -0.6, 1.2], dtype=float),
            accel_bias_mps2=np.array([0.05, -0.03, 0.02], dtype=float),
        )
        control = NavigationInput(
            accelerometer_mps2=np.array([0.8, -0.4, 0.3], dtype=float),
            rotation_body_to_inertial=quaternion_to_rotation_matrix(
                rotation_vector_to_quaternion(np.array([0.1, -0.2, 0.15], dtype=float))
            ),
        )
        dt = 0.2

        analytic = process_model.state_jacobian(nominal_state, control, dt)
        numeric = finite_difference_jacobian(
            lambda delta: state_vector(process_model.predict(nominal_state.plus(delta), control, dt)),
            np.zeros(9, dtype=float),
        )

        np.testing.assert_allclose(analytic, numeric, atol=1.0e-9, rtol=1.0e-9)

    def test_navigation_measurement_jacobians_match_finite_differences(self) -> None:
        nominal_state = NavigationState(
            position_m=np.array([12.0, -4.0, 30.0], dtype=float),
            velocity_mps=np.array([4.0, -1.5, 0.25], dtype=float),
            accel_bias_mps2=np.array([0.1, -0.2, 0.05], dtype=float),
        )
        position_model = GpsPositionMeasurementModel(
            GpsPositionConfig(measurement_std_m=np.array([2.0, 2.0, 3.0], dtype=float))
        )
        velocity_model = GpsVelocityMeasurementModel(
            GpsVelocityConfig(measurement_std_mps=np.array([0.5, 0.5, 0.75], dtype=float))
        )
        altitude_model = BarometricAltitudeMeasurementModel(
            BarometricAltitudeConfig(measurement_std_m=1.5, axis=1)
        )

        analytic_position = position_model.measurement_jacobian(
            np.zeros(3, dtype=float),
            nominal_state,
        )
        numeric_position = finite_difference_jacobian(
            lambda delta: position_model.predict_measurement(nominal_state.plus(delta)),
            np.zeros(9, dtype=float),
        )
        np.testing.assert_allclose(analytic_position, numeric_position, atol=1.0e-9, rtol=1.0e-9)

        analytic_velocity = velocity_model.measurement_jacobian(
            np.zeros(3, dtype=float),
            nominal_state,
        )
        numeric_velocity = finite_difference_jacobian(
            lambda delta: velocity_model.predict_measurement(nominal_state.plus(delta)),
            np.zeros(9, dtype=float),
        )
        np.testing.assert_allclose(analytic_velocity, numeric_velocity, atol=1.0e-9, rtol=1.0e-9)

        analytic_altitude = altitude_model.measurement_jacobian(0.0, nominal_state)
        numeric_altitude = finite_difference_jacobian(
            lambda delta: altitude_model.predict_measurement(nominal_state.plus(delta)),
            np.zeros(9, dtype=float),
        )
        np.testing.assert_allclose(analytic_altitude, numeric_altitude, atol=1.0e-9, rtol=1.0e-9)

    def test_recovery_from_noisy_position_and_velocity_measurements(self) -> None:
        rng = np.random.default_rng(7)
        process_model = NavigationProcessModel(
            NavigationProcessModelConfig(
                accelerometer_includes_gravity=True,
                process_noise=NavigationProcessNoise(
                    accelerometer_noise_std=5.0e-2,
                    accel_bias_random_walk_std=1.0e-4,
                ),
            )
        )
        position_model = GpsPositionMeasurementModel(
            GpsPositionConfig(measurement_std_m=np.array([0.75, 0.75, 0.75], dtype=float))
        )
        velocity_model = GpsVelocityMeasurementModel(
            GpsVelocityConfig(measurement_std_mps=np.array([0.2, 0.2, 0.2], dtype=float))
        )
        true_state = NavigationState(
            velocity_mps=np.array([1.8, -0.7, 0.25], dtype=float),
            accel_bias_mps2=np.array([0.12, -0.08, 0.05], dtype=float),
        )
        estimator = GenericEKF(
            process_model=process_model,
            initial_state=NavigationState(
                position_m=np.array([8.0, -4.0, 3.0], dtype=float),
                velocity_mps=np.array([-1.0, 1.5, -0.5], dtype=float),
                accel_bias_mps2=np.zeros(3, dtype=float),
            ),
            initial_covariance=np.diag(
                np.array([25.0, 25.0, 25.0, 9.0, 9.0, 9.0, 0.5, 0.5, 0.5], dtype=float)
            ),
            covariance_convention=CovarianceConvention(min_variance=1.0e-12),
        )
        control = NavigationInput(
            accelerometer_mps2=true_state.accel_bias_mps2.copy(),
            rotation_body_to_inertial=np.eye(3, dtype=float),
        )

        for _ in range(120):
            true_state = process_model.predict(true_state, control, dt=0.1)
            estimator.predict(control=control, dt=0.1)
            estimator.update(
                measurement_model=position_model,
                measurement=true_state.position_m
                + rng.normal(0.0, position_model.config.measurement_std_m, size=3),
            )
            estimator.update(
                measurement_model=velocity_model,
                measurement=true_state.velocity_mps
                + rng.normal(0.0, velocity_model.config.measurement_std_mps, size=3),
            )

        np.testing.assert_allclose(
            estimator.state.position_m,
            true_state.position_m,
            atol=0.8,
        )
        np.testing.assert_allclose(
            estimator.state.velocity_mps,
            true_state.velocity_mps,
            atol=0.15,
        )
        np.testing.assert_allclose(
            estimator.state.accel_bias_mps2,
            true_state.accel_bias_mps2,
            atol=0.08,
        )

    def test_barometric_altitude_update_matches_scalar_reference(self) -> None:
        process_model = NavigationProcessModel(
            NavigationProcessModelConfig(accelerometer_includes_gravity=True)
        )
        measurement_model = BarometricAltitudeMeasurementModel(
            BarometricAltitudeConfig(measurement_std_m=2.0, axis=2)
        )
        estimator = GenericEKF(
            process_model=process_model,
            initial_state=NavigationState(
                position_m=np.array([4.0, -2.0, 40.0], dtype=float),
                velocity_mps=np.array([1.0, 0.5, -0.25], dtype=float),
                accel_bias_mps2=np.array([0.1, -0.05, 0.02], dtype=float),
            ),
            initial_covariance=np.diag(
                np.array([1.0, 1.0, 25.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2], dtype=float)
            ),
            covariance_convention=CovarianceConvention(min_variance=1.0e-12),
        )

        update = estimator.update(measurement_model=measurement_model, measurement=30.0)

        prior_altitude = 40.0
        prior_variance = 25.0
        measurement_variance = 4.0
        innovation = 30.0 - prior_altitude
        kalman_gain = prior_variance / (prior_variance + measurement_variance)
        expected_altitude = prior_altitude + kalman_gain * innovation
        expected_variance = (
            (1.0 - kalman_gain) ** 2 * prior_variance
            + kalman_gain ** 2 * measurement_variance
        )

        self.assertEqual(update.status, MeasurementUpdateStatus.ACCEPTED)
        self.assertEqual(update.label, "baro_altitude")
        self.assertAlmostEqual(float(estimator.state.position_m[2]), expected_altitude)
        self.assertAlmostEqual(float(estimator.covariance[2, 2]), expected_variance)
        np.testing.assert_allclose(
            estimator.state.position_m[0:2],
            np.array([4.0, -2.0], dtype=float),
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
