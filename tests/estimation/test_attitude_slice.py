from __future__ import annotations

import unittest

import numpy as np

from sim.estimation.core import CovarianceConvention, GenericESKF
from sim.estimation.math import (
    finite_difference_jacobian,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    rotation_vector_to_quaternion,
)
from sim.estimation.measurements import GravityAlignmentMeasurementModel
from sim.estimation.models import (
    AttitudeErrorStateProcessModel,
    AttitudeInput,
    AttitudeProcessModelConfig,
    AttitudeState,
)


def quaternion_angle_error(reference_quaternion: np.ndarray, estimated_quaternion: np.ndarray) -> float:
    delta_quaternion = quaternion_multiply(
        quaternion_inverse(reference_quaternion),
        estimated_quaternion,
    )
    delta_quaternion = normalize_quaternion(delta_quaternion)
    return 2.0 * np.arctan2(
        float(np.linalg.norm(delta_quaternion[1:])),
        abs(float(delta_quaternion[0])),
    )


class AttitudeSliceTests(unittest.TestCase):
    def test_static_gravity_alignment_converges_from_small_misalignment(self) -> None:
        process_model = AttitudeErrorStateProcessModel(AttitudeProcessModelConfig())
        measurement_model = GravityAlignmentMeasurementModel()
        estimator = GenericESKF(
            process_model=process_model,
            initial_state=AttitudeState(
                quaternion=rotation_vector_to_quaternion(np.array([0.18, -0.12, 0.0], dtype=float)),
            ),
            initial_covariance=np.diag(
                np.array([0.5, 0.5, 0.5, 0.05, 0.05, 0.05], dtype=float)
            ),
            covariance_convention=CovarianceConvention(min_variance=1e-12),
        )
        static_measurement = np.array([0.0, 0.0, -9.80665], dtype=float)
        control = AttitudeInput(gyroscope_rps=np.zeros(3, dtype=float))

        for _ in range(40):
            estimator.predict(control=control, dt=0.05)
            estimator.update(
                measurement_model=measurement_model,
                measurement=static_measurement,
            )

        angle_error = quaternion_angle_error(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            estimator.state.quaternion,
        )
        predicted_gravity = measurement_model.predict_measurement(estimator.state)

        self.assertLess(angle_error, 1.0e-2)
        np.testing.assert_allclose(predicted_gravity, static_measurement, atol=1.0e-2)

    def test_quaternion_injection_matches_small_angle_correction(self) -> None:
        process_model = AttitudeErrorStateProcessModel(AttitudeProcessModelConfig())
        nominal_state = AttitudeState(
            quaternion=rotation_vector_to_quaternion(np.array([0.1, -0.05, 0.02], dtype=float)),
            gyro_bias_rps=np.array([0.01, -0.02, 0.005], dtype=float),
        )
        error_state = np.array([0.01, -0.015, 0.02, 0.002, -0.001, 0.003], dtype=float)

        injected = process_model.inject(nominal_state, error_state)
        expected_quaternion = normalize_quaternion(
            quaternion_multiply(
                nominal_state.quaternion,
                rotation_vector_to_quaternion(error_state[0:3]),
            )
        )

        np.testing.assert_allclose(injected.quaternion, expected_quaternion, atol=1e-12)
        np.testing.assert_allclose(
            injected.gyro_bias_rps,
            nominal_state.gyro_bias_rps + error_state[3:6],
            atol=1e-12,
        )

    def test_analytic_gravity_jacobian_matches_finite_difference(self) -> None:
        process_model = AttitudeErrorStateProcessModel(AttitudeProcessModelConfig())
        measurement_model = GravityAlignmentMeasurementModel()
        nominal_state = AttitudeState(
            quaternion=rotation_vector_to_quaternion(np.array([0.2, -0.15, 0.0], dtype=float)),
            gyro_bias_rps=np.array([0.01, -0.02, 0.015], dtype=float),
        )

        analytic = measurement_model.measurement_jacobian(
            np.zeros(3, dtype=float),
            nominal_state,
        )
        numeric = finite_difference_jacobian(
            lambda error_state: measurement_model.predict_measurement(
                process_model.inject(nominal_state, error_state)
            ),
            np.zeros(6, dtype=float),
        )

        np.testing.assert_allclose(analytic, numeric, atol=1.0e-6, rtol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
