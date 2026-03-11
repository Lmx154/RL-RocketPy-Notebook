"""Core estimator contracts defined in phase 0.

These interfaces freeze boundaries for the additive rewrite without
introducing filter behavior yet.
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from .types import ControlT, Matrix, MeasurementT, StateT, Vector


@runtime_checkable
class NominalStateProtocol(Protocol):
    """Minimal contract shared by Euclidean and manifold nominal states."""

    def copy(self) -> Self:
        """Return an independent copy of the nominal state container."""


@runtime_checkable
class EuclideanStateProtocol(NominalStateProtocol, Protocol):
    """Nominal state whose correction operator is direct vector addition."""

    def plus(self, delta: Vector) -> Self:
        """Return the additively corrected state without mutating the original."""


@runtime_checkable
class ProcessModel(Protocol[StateT, ControlT]):
    """Shared process-model boundary for EKF and ESKF implementations."""

    def predict(self, nominal_state: StateT, control: ControlT, dt: float) -> StateT:
        """Propagate the nominal state over one time step."""

    def process_noise_jacobian(
        self,
        nominal_state: StateT,
        control: ControlT,
        dt: float,
    ) -> Matrix:
        """Return the process noise Jacobian G for the supplied step."""

    def process_noise_covariance(
        self,
        nominal_state: StateT,
        control: ControlT,
        dt: float,
    ) -> Matrix:
        """Return the process noise covariance Q for the supplied step."""


@runtime_checkable
class EuclideanProcessModel(ProcessModel[StateT, ControlT], Protocol):
    """Process model contract for standard EKF state propagation."""

    def state_jacobian(self, nominal_state: StateT, control: ControlT, dt: float) -> Matrix:
        """Return the linearized state Jacobian F for the supplied step."""


@runtime_checkable
class ErrorStateProcessModel(ProcessModel[StateT, ControlT], Protocol):
    """Process model contract for manifold-valued ESKF propagation."""

    def error_state_jacobian(
        self,
        nominal_state: StateT,
        control: ControlT,
        dt: float,
    ) -> Matrix:
        """Return the error-state Jacobian F for the supplied step."""

    def inject(self, nominal_state: StateT, error_state: Vector) -> StateT:
        """Apply an error-state correction to the nominal state."""

    def reset_jacobian(self, injected_error_state: Vector) -> Matrix:
        """Return the covariance reset Jacobian after error injection."""


@runtime_checkable
class MeasurementModel(Protocol[StateT, MeasurementT]):
    """Nonlinear measurement-model boundary shared by EKF and ESKF engines."""

    label: str

    def predict_measurement(self, nominal_state: StateT) -> MeasurementT:
        """Return the predicted measurement h(x) for the nominal state."""

    def innovation(
        self,
        measurement: MeasurementT,
        predicted_measurement: MeasurementT,
    ) -> Vector:
        """Return the innovation y, defaulting conceptually to z - h(x)."""

    def measurement_jacobian(
        self,
        measurement: MeasurementT,
        nominal_state: StateT,
    ) -> Matrix:
        """Return the measurement Jacobian H evaluated for the measurement step."""

    def measurement_covariance(
        self,
        measurement: MeasurementT,
        nominal_state: StateT,
    ) -> Matrix:
        """Return the measurement covariance R for the submitted measurement."""
