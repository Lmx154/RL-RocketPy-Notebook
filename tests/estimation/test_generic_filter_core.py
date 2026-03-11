from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from sim.estimation.core import (
    CovarianceConvention,
    GenericEKF,
    GenericESKF,
    MeasurementUpdateStatus,
)
from sim.estimation.policies import GateDecision, GateStatus


@dataclass(slots=True)
class ScalarState:
    value: float

    def copy(self) -> "ScalarState":
        return ScalarState(value=float(self.value))

    def plus(self, delta: np.ndarray) -> "ScalarState":
        return ScalarState(value=float(self.value + np.asarray(delta, dtype=float)[0]))


class ScalarProcessModel:
    def __init__(self, process_variance: float) -> None:
        self.process_variance = float(process_variance)

    def predict(self, nominal_state: ScalarState, control: float, dt: float) -> ScalarState:
        return ScalarState(value=float(nominal_state.value + control * dt))

    def state_jacobian(self, nominal_state: ScalarState, control: float, dt: float) -> np.ndarray:
        return np.array([[1.0]], dtype=float)

    def process_noise_jacobian(
        self,
        nominal_state: ScalarState,
        control: float,
        dt: float,
    ) -> np.ndarray:
        return np.array([[1.0]], dtype=float)

    def process_noise_covariance(
        self,
        nominal_state: ScalarState,
        control: float,
        dt: float,
    ) -> np.ndarray:
        return np.array([[self.process_variance]], dtype=float)


class SquareMeasurementModel:
    label = "square_measurement"

    def __init__(self, measurement_variance: float) -> None:
        self.measurement_variance = float(measurement_variance)

    def predict_measurement(self, nominal_state: ScalarState) -> np.ndarray:
        return np.array([nominal_state.value ** 2], dtype=float)

    def innovation(
        self,
        measurement: np.ndarray,
        predicted_measurement: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(measurement, dtype=float) - np.asarray(predicted_measurement, dtype=float)

    def measurement_jacobian(
        self,
        measurement: np.ndarray,
        nominal_state: ScalarState,
    ) -> np.ndarray:
        return np.array([[2.0 * nominal_state.value]], dtype=float)

    def measurement_covariance(
        self,
        measurement: np.ndarray,
        nominal_state: ScalarState,
    ) -> np.ndarray:
        return np.array([[self.measurement_variance]], dtype=float)


class MahalanobisRejectGate:
    def __init__(self, max_distance: float) -> None:
        self.max_distance = float(max_distance)

    def evaluate(
        self,
        *,
        measurement_model,
        measurement,
        nominal_state,
        innovation: np.ndarray,
        innovation_covariance: np.ndarray,
    ) -> GateDecision:
        distance = float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))
        if distance > self.max_distance:
            return GateDecision(
                status=GateStatus.REJECT,
                reason="distance_limit",
                mahalanobis_distance=distance,
            )
        return GateDecision(status=GateStatus.ACCEPT, mahalanobis_distance=distance)


def wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(slots=True)
class WrappedAngleState:
    angle_rad: float

    def copy(self) -> "WrappedAngleState":
        return WrappedAngleState(angle_rad=float(self.angle_rad))


class WrappedAngleProcessModel:
    def __init__(self, process_variance: float) -> None:
        self.process_variance = float(process_variance)

    def predict(
        self,
        nominal_state: WrappedAngleState,
        control: float,
        dt: float,
    ) -> WrappedAngleState:
        return WrappedAngleState(angle_rad=wrap_angle(nominal_state.angle_rad + control * dt))

    def error_state_jacobian(
        self,
        nominal_state: WrappedAngleState,
        control: float,
        dt: float,
    ) -> np.ndarray:
        return np.array([[1.0]], dtype=float)

    def process_noise_jacobian(
        self,
        nominal_state: WrappedAngleState,
        control: float,
        dt: float,
    ) -> np.ndarray:
        return np.array([[1.0]], dtype=float)

    def process_noise_covariance(
        self,
        nominal_state: WrappedAngleState,
        control: float,
        dt: float,
    ) -> np.ndarray:
        return np.array([[self.process_variance]], dtype=float)

    def inject(
        self,
        nominal_state: WrappedAngleState,
        error_state: np.ndarray,
    ) -> WrappedAngleState:
        return WrappedAngleState(
            angle_rad=wrap_angle(nominal_state.angle_rad + np.asarray(error_state, dtype=float)[0])
        )

    def reset_jacobian(self, injected_error_state: np.ndarray) -> np.ndarray:
        return np.array([[1.0]], dtype=float)


class WrappedAngleMeasurementModel:
    label = "wrapped_angle"

    def __init__(self, measurement_variance: float) -> None:
        self.measurement_variance = float(measurement_variance)

    def predict_measurement(self, nominal_state: WrappedAngleState) -> np.ndarray:
        return np.array([nominal_state.angle_rad], dtype=float)

    def innovation(
        self,
        measurement: np.ndarray,
        predicted_measurement: np.ndarray,
    ) -> np.ndarray:
        return np.array(
            [
                wrap_angle(
                    np.asarray(measurement, dtype=float)[0]
                    - np.asarray(predicted_measurement, dtype=float)[0]
                )
            ],
            dtype=float,
        )

    def measurement_jacobian(
        self,
        measurement: np.ndarray,
        nominal_state: WrappedAngleState,
    ) -> np.ndarray:
        return np.array([[1.0]], dtype=float)

    def measurement_covariance(
        self,
        measurement: np.ndarray,
        nominal_state: WrappedAngleState,
    ) -> np.ndarray:
        return np.array([[self.measurement_variance]], dtype=float)


class GenericGaussianFilterTests(unittest.TestCase):
    def assert_covariance_is_psd(self, covariance: np.ndarray, tolerance: float = 1e-12) -> None:
        np.testing.assert_allclose(covariance, covariance.T, atol=tolerance)
        eigenvalues = np.linalg.eigvalsh(covariance)
        self.assertGreaterEqual(float(np.min(eigenvalues)), -tolerance)

    def test_generic_ekf_matches_scalar_nonlinear_reference(self) -> None:
        process_model = ScalarProcessModel(process_variance=0.25)
        measurement_model = SquareMeasurementModel(measurement_variance=9.0)
        ekf = GenericEKF(
            process_model=process_model,
            initial_state=ScalarState(value=1.0),
            initial_covariance=np.array([[4.0]], dtype=float),
            covariance_convention=CovarianceConvention(min_variance=1e-12),
        )

        prediction = ekf.predict(control=2.0, dt=0.5)
        self.assertAlmostEqual(ekf.state.value, 2.0)
        np.testing.assert_allclose(ekf.covariance, np.array([[4.25]], dtype=float))
        np.testing.assert_allclose(prediction.transition_jacobian, np.array([[1.0]], dtype=float))

        update = ekf.update(
            measurement_model=measurement_model,
            measurement=np.array([5.0], dtype=float),
        )

        prior_state = 2.0
        prior_covariance = 4.25
        innovation = 5.0 - prior_state ** 2
        measurement_jacobian = 2.0 * prior_state
        innovation_covariance = (
            measurement_jacobian * prior_covariance * measurement_jacobian + 9.0
        )
        kalman_gain = prior_covariance * measurement_jacobian / innovation_covariance
        expected_state = prior_state + kalman_gain * innovation
        expected_covariance = (
            (1.0 - kalman_gain * measurement_jacobian) ** 2 * prior_covariance
            + kalman_gain ** 2 * 9.0
        )

        self.assertEqual(update.status, MeasurementUpdateStatus.ACCEPTED)
        self.assertAlmostEqual(ekf.state.value, expected_state)
        np.testing.assert_allclose(ekf.covariance, np.array([[expected_covariance]], dtype=float))
        np.testing.assert_allclose(update.diagnostics.predicted_measurement, np.array([4.0], dtype=float))
        np.testing.assert_allclose(update.innovation, np.array([innovation], dtype=float))
        self.assert_covariance_is_psd(ekf.covariance)

    def test_generic_ekf_supports_external_gating_hook(self) -> None:
        process_model = ScalarProcessModel(process_variance=0.0)
        measurement_model = SquareMeasurementModel(measurement_variance=0.1)
        ekf = GenericEKF(
            process_model=process_model,
            initial_state=ScalarState(value=1.0),
            initial_covariance=np.array([[1.0]], dtype=float),
            measurement_gate=MahalanobisRejectGate(max_distance=1.0),
        )

        previous_state = ekf.state.copy()
        previous_covariance = ekf.covariance.copy()
        update = ekf.update(
            measurement_model=measurement_model,
            measurement=np.array([25.0], dtype=float),
        )

        self.assertEqual(update.status, MeasurementUpdateStatus.REJECTED)
        self.assertEqual(update.gate_reason, "distance_limit")
        self.assertAlmostEqual(ekf.state.value, previous_state.value)
        np.testing.assert_allclose(ekf.covariance, previous_covariance)
        self.assert_covariance_is_psd(ekf.covariance)

    def test_generic_eskf_toy_manifold_update_uses_injection_and_reset(self) -> None:
        process_model = WrappedAngleProcessModel(process_variance=0.04)
        measurement_model = WrappedAngleMeasurementModel(measurement_variance=0.04)
        eskf = GenericESKF(
            process_model=process_model,
            initial_state=WrappedAngleState(angle_rad=3.12),
            initial_covariance=np.array([[0.25]], dtype=float),
            covariance_convention=CovarianceConvention(min_variance=1e-12),
        )

        prediction = eskf.predict(control=0.0, dt=1.0)
        np.testing.assert_allclose(prediction.transition_jacobian, np.array([[1.0]], dtype=float))
        np.testing.assert_allclose(eskf.covariance, np.array([[0.29]], dtype=float))

        update = eskf.update(
            measurement_model=measurement_model,
            measurement=np.array([-3.13], dtype=float),
        )

        innovation = wrap_angle(-3.13 - 3.12)
        kalman_gain = 0.29 / (0.29 + 0.04)
        expected_error_state = kalman_gain * innovation
        expected_angle = wrap_angle(3.12 + expected_error_state)
        expected_covariance = (1.0 - kalman_gain) ** 2 * 0.29 + kalman_gain ** 2 * 0.04

        self.assertEqual(update.status, MeasurementUpdateStatus.ACCEPTED)
        self.assertAlmostEqual(float(update.state_correction[0]), expected_error_state)
        self.assertAlmostEqual(eskf.state.angle_rad, expected_angle)
        np.testing.assert_allclose(
            eskf.covariance,
            np.array([[expected_covariance]], dtype=float),
        )
        self.assertLess(eskf.state.angle_rad, 0.0)
        self.assert_covariance_is_psd(eskf.covariance)


if __name__ == "__main__":
    unittest.main()
