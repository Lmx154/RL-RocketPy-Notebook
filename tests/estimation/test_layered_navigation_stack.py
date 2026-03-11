from __future__ import annotations

import unittest

import numpy as np

from sim.estimation.math import quaternion_to_rotation_matrix
from sim.estimation.models import (
    AttitudeErrorStateProcessModel,
    AttitudeInput,
    AttitudeState,
    NavigationInput,
    NavigationProcessModel,
    NavigationState,
)
from sim.estimation.stacks import LayeredNavigationStack, LayeredNavigationState


def assert_measurement_results_equal(
    test_case: unittest.TestCase,
    left,
    right,
) -> None:
    test_case.assertEqual(left.status, right.status)
    test_case.assertEqual(left.label, right.label)
    test_case.assertEqual(left.measurement_dim, right.measurement_dim)
    test_case.assertEqual(left.gate_reason, right.gate_reason)
    if left.mahalanobis_distance is None or right.mahalanobis_distance is None:
        test_case.assertIs(left.mahalanobis_distance, right.mahalanobis_distance)
    else:
        test_case.assertAlmostEqual(left.mahalanobis_distance, right.mahalanobis_distance)
    np.testing.assert_allclose(left.innovation, right.innovation, atol=0.0, rtol=0.0)
    if left.innovation_covariance is not None or right.innovation_covariance is not None:
        np.testing.assert_allclose(
            left.innovation_covariance,
            right.innovation_covariance,
            atol=0.0,
            rtol=0.0,
        )


class LayeredNavigationStackTests(unittest.TestCase):
    def test_composition_smoke_with_synthetic_imu_and_position_measurements(self) -> None:
        attitude_model = AttitudeErrorStateProcessModel()
        navigation_model = NavigationProcessModel()
        true_attitude = AttitudeState()
        true_navigation = NavigationState()
        estimator = LayeredNavigationStack(initial_state=LayeredNavigationState())

        accelerometer_mps2 = np.array([0.8, -0.1, 0.2], dtype=float)
        gyroscope_rps = np.array([0.0, 0.0, 0.25], dtype=float)
        dt = 0.05

        for step in range(60):
            true_attitude = attitude_model.predict(
                true_attitude,
                AttitudeInput(gyroscope_rps=gyroscope_rps),
                dt,
            )
            true_navigation = navigation_model.predict(
                true_navigation,
                NavigationInput(
                    accelerometer_mps2=accelerometer_mps2,
                    rotation_body_to_inertial=quaternion_to_rotation_matrix(true_attitude.quaternion),
                ),
                dt,
            )

            prediction = estimator.predict(
                accelerometer_mps2=accelerometer_mps2,
                gyroscope_rps=gyroscope_rps,
                dt=dt,
                timestamp_s=(step + 1) * dt,
            )
            update = estimator.update_position(position_m=true_navigation.position_m)

        snapshot = estimator.snapshot()

        np.testing.assert_allclose(snapshot.state.quaternion, true_attitude.quaternion, atol=1.0e-12)
        np.testing.assert_allclose(snapshot.state.position_m, true_navigation.position_m, atol=1.0e-10)
        np.testing.assert_allclose(snapshot.state.velocity_mps, true_navigation.velocity_mps, atol=1.0e-10)
        np.testing.assert_allclose(
            snapshot.inertial_acceleration_mps2,
            quaternion_to_rotation_matrix(true_attitude.quaternion) @ accelerometer_mps2,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(float(snapshot.timestamp_s), 60 * dt)
        self.assertAlmostEqual(float(snapshot.prediction_dt_s), dt)
        self.assertIsNotNone(prediction.attitude_prediction)
        self.assertIsNotNone(prediction.navigation_prediction)
        self.assertEqual(update.label, "gps_position")
        self.assertIsNone(snapshot.diagnostics.attitude_update)
        self.assertEqual(snapshot.diagnostics.navigation_update.label, "gps_position")

    def test_composition_is_deterministic_for_repeated_sequences(self) -> None:
        initial_state = LayeredNavigationState(
            quaternion=np.array([0.99680171, 0.05993602, -0.04994668, 0.0], dtype=float),
            position_m=np.array([4.0, -2.0, 1.5], dtype=float),
            velocity_mps=np.array([0.5, -0.25, 0.1], dtype=float),
        )
        stack_a = LayeredNavigationStack(initial_state=initial_state)
        stack_b = LayeredNavigationStack(initial_state=initial_state)

        gravity_measurement = np.array([0.0, 0.0, -9.80665], dtype=float)
        accelerometer_mps2 = np.zeros(3, dtype=float)
        gyroscope_rps = np.zeros(3, dtype=float)
        dt = 0.1

        for step in range(25):
            timestamp_s = (step + 1) * dt
            for estimator in (stack_a, stack_b):
                estimator.predict(
                    accelerometer_mps2=accelerometer_mps2,
                    gyroscope_rps=gyroscope_rps,
                    dt=dt,
                    timestamp_s=timestamp_s,
                )
                estimator.update_gravity_alignment(accelerometer_mps2=gravity_measurement)
                estimator.update_position(position_m=np.zeros(3, dtype=float))
                estimator.update_velocity(velocity_mps=np.zeros(3, dtype=float))
                estimator.update_barometric_altitude(altitude_m=0.0)

        snapshot_a = stack_a.snapshot()
        snapshot_b = stack_b.snapshot()

        np.testing.assert_allclose(snapshot_a.state.quaternion, snapshot_b.state.quaternion, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(snapshot_a.state.gyro_bias_rps, snapshot_b.state.gyro_bias_rps, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(snapshot_a.state.position_m, snapshot_b.state.position_m, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(snapshot_a.state.velocity_mps, snapshot_b.state.velocity_mps, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            snapshot_a.state.accel_bias_mps2,
            snapshot_b.state.accel_bias_mps2,
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            snapshot_a.covariance.attitude,
            snapshot_b.covariance.attitude,
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            snapshot_a.covariance.navigation,
            snapshot_b.covariance.navigation,
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            snapshot_a.inertial_acceleration_mps2,
            snapshot_b.inertial_acceleration_mps2,
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(snapshot_a.timestamp_s, snapshot_b.timestamp_s)
        self.assertEqual(snapshot_a.prediction_dt_s, snapshot_b.prediction_dt_s)

        self.assertIsNotNone(snapshot_a.diagnostics.attitude_prediction)
        self.assertIsNotNone(snapshot_b.diagnostics.attitude_prediction)
        np.testing.assert_allclose(
            snapshot_a.diagnostics.attitude_prediction.transition_jacobian,
            snapshot_b.diagnostics.attitude_prediction.transition_jacobian,
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            snapshot_a.diagnostics.navigation_prediction.transition_jacobian,
            snapshot_b.diagnostics.navigation_prediction.transition_jacobian,
            atol=0.0,
            rtol=0.0,
        )
        assert_measurement_results_equal(
            self,
            snapshot_a.diagnostics.attitude_update,
            snapshot_b.diagnostics.attitude_update,
        )
        assert_measurement_results_equal(
            self,
            snapshot_a.diagnostics.navigation_update,
            snapshot_b.diagnostics.navigation_update,
        )


if __name__ == "__main__":
    unittest.main()
