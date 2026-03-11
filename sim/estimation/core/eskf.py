"""Generic manifold-state ESKF engine."""

from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np

from ..policies.gating import GateDecision, GateStatus, MeasurementGate
from .base import ErrorStateProcessModel, MeasurementModel, NominalStateProtocol
from .gaussian import (
    CovarianceConvention,
    MeasurementDiagnostics,
    MeasurementUpdateResult,
    MeasurementUpdateStatus,
    PredictionDiagnostics,
    apply_minimum_variance,
    compute_innovation_covariance,
    compute_kalman_gain,
    joseph_covariance_update,
    mahalanobis_distance,
    symmetrize_covariance,
)
from .types import ControlT, Matrix, MeasurementT

NominalStateT = TypeVar("NominalStateT", bound=NominalStateProtocol)


class GenericESKF(Generic[NominalStateT, ControlT, MeasurementT]):
    """Reusable ESKF engine over manifold-valued nominal states."""

    def __init__(
        self,
        *,
        process_model: ErrorStateProcessModel[NominalStateT, ControlT],
        initial_state: NominalStateT,
        initial_covariance: Matrix,
        covariance_convention: CovarianceConvention | None = None,
        measurement_gate: MeasurementGate[NominalStateT, MeasurementT] | None = None,
    ) -> None:
        self.process_model = process_model
        self.state = initial_state.copy()
        self.covariance_convention = covariance_convention or CovarianceConvention()
        self.covariance = self._regularize_covariance(
            np.asarray(initial_covariance, dtype=float),
            after_update=False,
        )
        self.measurement_gate = measurement_gate
        self.last_prediction: PredictionDiagnostics | None = None
        self.last_update: MeasurementUpdateResult[MeasurementT] | None = None

    def predict(self, control: ControlT, dt: float) -> PredictionDiagnostics:
        dt = float(dt)
        dimension = self.covariance.shape[0]
        if dt <= 0.0:
            diagnostics = PredictionDiagnostics(
                transition_jacobian=np.eye(dimension, dtype=float),
                process_noise_jacobian=np.zeros((dimension, dimension), dtype=float),
                process_noise_covariance=np.zeros((dimension, dimension), dtype=float),
            )
            self.last_prediction = diagnostics
            return diagnostics

        prior_state = self.state.copy()
        transition_jacobian = np.asarray(
            self.process_model.error_state_jacobian(prior_state, control, dt),
            dtype=float,
        )
        process_noise_jacobian = np.asarray(
            self.process_model.process_noise_jacobian(prior_state, control, dt),
            dtype=float,
        )
        process_noise_covariance = np.asarray(
            self.process_model.process_noise_covariance(prior_state, control, dt),
            dtype=float,
        )

        predicted_covariance = (
            transition_jacobian @ self.covariance @ transition_jacobian.T
            + process_noise_jacobian @ process_noise_covariance @ process_noise_jacobian.T
        )
        self.state = self.process_model.predict(prior_state, control, dt).copy()
        self.covariance = self._regularize_covariance(predicted_covariance, after_update=False)

        diagnostics = PredictionDiagnostics(
            transition_jacobian=transition_jacobian,
            process_noise_jacobian=process_noise_jacobian,
            process_noise_covariance=process_noise_covariance,
        )
        self.last_prediction = diagnostics
        return diagnostics

    def update(
        self,
        *,
        measurement_model: MeasurementModel[NominalStateT, MeasurementT],
        measurement: MeasurementT,
        measurement_gate: MeasurementGate[NominalStateT, MeasurementT] | None = None,
    ) -> MeasurementUpdateResult[MeasurementT]:
        predicted_measurement = measurement_model.predict_measurement(self.state)
        innovation = np.asarray(
            measurement_model.innovation(measurement, predicted_measurement),
            dtype=float,
        )
        measurement_jacobian = np.asarray(
            measurement_model.measurement_jacobian(measurement, self.state),
            dtype=float,
        )
        measurement_covariance = np.asarray(
            measurement_model.measurement_covariance(measurement, self.state),
            dtype=float,
        )
        innovation_covariance = compute_innovation_covariance(
            self.covariance,
            measurement_jacobian,
            measurement_covariance,
            self.covariance_convention.min_variance,
        )
        distance = mahalanobis_distance(innovation, innovation_covariance)
        diagnostics = MeasurementDiagnostics(
            predicted_measurement=predicted_measurement,
            innovation=innovation.copy(),
            measurement_jacobian=measurement_jacobian,
            measurement_covariance=measurement_covariance,
            innovation_covariance=innovation_covariance,
        )

        gate = measurement_gate or self.measurement_gate
        if gate is not None:
            decision = gate.evaluate(
                measurement_model=measurement_model,
                measurement=measurement,
                nominal_state=self.state,
                innovation=innovation,
                innovation_covariance=innovation_covariance,
            )
        else:
            decision = GateDecision(status=GateStatus.ACCEPT, mahalanobis_distance=distance)

        if decision.status is not GateStatus.ACCEPT:
            result = MeasurementUpdateResult(
                status=self._map_gate_status(decision.status),
                innovation=innovation.copy(),
                measurement_dim=innovation.size,
                label=measurement_model.label,
                mahalanobis_distance=(
                    distance if decision.mahalanobis_distance is None else decision.mahalanobis_distance
                ),
                innovation_covariance=innovation_covariance,
                gate_reason=decision.reason,
                diagnostics=diagnostics,
            )
            self.last_update = result
            return result

        kalman_gain = compute_kalman_gain(
            self.covariance,
            measurement_jacobian,
            innovation_covariance,
        )
        error_state_correction = kalman_gain @ innovation
        updated_covariance = joseph_covariance_update(
            self.covariance,
            measurement_jacobian,
            measurement_covariance,
            kalman_gain,
        )
        self.state = self.process_model.inject(
            self.state,
            error_state_correction,
        ).copy()
        reset_jacobian = np.asarray(
            self.process_model.reset_jacobian(error_state_correction),
            dtype=float,
        )
        self.covariance = self._regularize_covariance(
            reset_jacobian @ updated_covariance @ reset_jacobian.T,
            after_update=True,
        )

        result = MeasurementUpdateResult(
            status=MeasurementUpdateStatus.ACCEPTED,
            innovation=innovation.copy(),
            measurement_dim=innovation.size,
            label=measurement_model.label,
            mahalanobis_distance=distance,
            innovation_covariance=innovation_covariance,
            state_correction=error_state_correction,
            diagnostics=diagnostics,
        )
        self.last_update = result
        return result

    def _regularize_covariance(self, covariance: Matrix, *, after_update: bool) -> Matrix:
        covariance = np.asarray(covariance, dtype=float)
        should_symmetrize = (
            self.covariance_convention.symmetrize_after_update
            if after_update
            else self.covariance_convention.symmetrize_after_predict
        )
        if should_symmetrize:
            covariance = symmetrize_covariance(covariance)
        if self.covariance_convention.min_variance > 0.0:
            covariance = apply_minimum_variance(
                covariance,
                self.covariance_convention.min_variance,
            )
        return covariance

    @staticmethod
    def _map_gate_status(status: GateStatus) -> MeasurementUpdateStatus:
        if status is GateStatus.REJECT:
            return MeasurementUpdateStatus.REJECTED
        return MeasurementUpdateStatus.SKIPPED
