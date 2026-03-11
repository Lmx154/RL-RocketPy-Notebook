"""Gaussian filtering conventions and utilities shared by the estimator stack."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic

import numpy as np

from .types import Matrix, MeasurementT, Vector


class CovarianceUpdateForm(str, Enum):
    """Supported covariance update conventions for future filter engines."""

    JOSEPH = "joseph"


class MeasurementUpdateStatus(str, Enum):
    """Outcome of a measurement submission after gating and update handling."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class CovarianceConvention:
    """Shared covariance-handling policy for both EKF and ESKF engines."""

    min_variance: float = 1e-9
    symmetrize_after_predict: bool = True
    symmetrize_after_update: bool = True
    update_form: CovarianceUpdateForm = CovarianceUpdateForm.JOSEPH


@dataclass(frozen=True, slots=True)
class PredictionDiagnostics:
    """Reusable diagnostics for a Gaussian predict step."""

    transition_jacobian: Matrix
    process_noise_jacobian: Matrix
    process_noise_covariance: Matrix


@dataclass(frozen=True, slots=True)
class MeasurementDiagnostics(Generic[MeasurementT]):
    """Reusable diagnostics for a Gaussian measurement step."""

    predicted_measurement: MeasurementT
    innovation: Vector
    measurement_jacobian: Matrix
    measurement_covariance: Matrix
    innovation_covariance: Matrix


@dataclass(frozen=True, slots=True)
class MeasurementUpdateResult(Generic[MeasurementT]):
    """Reusable diagnostics shape for EKF and ESKF measurement results."""

    status: MeasurementUpdateStatus
    innovation: Vector
    measurement_dim: int
    label: str
    mahalanobis_distance: float | None = None
    innovation_covariance: Matrix | None = None
    state_correction: Vector | None = None
    gate_reason: str = ""
    diagnostics: MeasurementDiagnostics[MeasurementT] | None = None


def symmetrize_covariance(covariance: Matrix) -> Matrix:
    """Return the symmetric part of a covariance matrix."""

    covariance = np.asarray(covariance, dtype=float)
    return 0.5 * (covariance + covariance.T)


def apply_minimum_variance(covariance: Matrix, min_variance: float) -> Matrix:
    """Clamp covariance diagonal entries to a minimum variance floor."""

    covariance = np.asarray(covariance, dtype=float).copy()
    covariance[np.diag_indices_from(covariance)] = np.clip(
        np.diag(covariance),
        float(min_variance),
        None,
    )
    return covariance


def regularize_covariance(covariance: Matrix, min_variance: float) -> Matrix:
    """Symmetrize a covariance matrix and enforce a minimum diagonal floor."""

    return apply_minimum_variance(symmetrize_covariance(covariance), min_variance)


def compute_innovation_covariance(
    prior_covariance: Matrix,
    measurement_jacobian: Matrix,
    measurement_covariance: Matrix,
    min_variance: float = 0.0,
) -> Matrix:
    """Return the innovation covariance ``S = HPH^T + R`` with optional regularization."""

    innovation_covariance = (
        np.asarray(measurement_jacobian, dtype=float)
        @ np.asarray(prior_covariance, dtype=float)
        @ np.asarray(measurement_jacobian, dtype=float).T
        + np.asarray(measurement_covariance, dtype=float)
    )
    return regularize_covariance(innovation_covariance, min_variance)


def mahalanobis_distance(innovation: Vector, innovation_covariance: Matrix) -> float:
    """Return the squared Mahalanobis distance for an innovation vector."""

    innovation = np.asarray(innovation, dtype=float)
    innovation_covariance = np.asarray(innovation_covariance, dtype=float)
    return float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))


def compute_kalman_gain(
    prior_covariance: Matrix,
    measurement_jacobian: Matrix,
    innovation_covariance: Matrix,
) -> Matrix:
    """Return the Kalman gain ``K = PH^T S^{-1}`` without forming ``S^{-1}`` explicitly."""

    prior_covariance = np.asarray(prior_covariance, dtype=float)
    measurement_jacobian = np.asarray(measurement_jacobian, dtype=float)
    innovation_covariance = np.asarray(innovation_covariance, dtype=float)
    return np.linalg.solve(
        innovation_covariance.T,
        measurement_jacobian @ prior_covariance.T,
    ).T


def joseph_covariance_update(
    prior_covariance: Matrix,
    measurement_jacobian: Matrix,
    measurement_covariance: Matrix,
    kalman_gain: Matrix,
) -> Matrix:
    """Return the Joseph-form covariance update."""

    prior_covariance = np.asarray(prior_covariance, dtype=float)
    measurement_jacobian = np.asarray(measurement_jacobian, dtype=float)
    measurement_covariance = np.asarray(measurement_covariance, dtype=float)
    kalman_gain = np.asarray(kalman_gain, dtype=float)

    identity = np.eye(prior_covariance.shape[0], dtype=float)
    joseph_left = identity - kalman_gain @ measurement_jacobian
    return (
        joseph_left @ prior_covariance @ joseph_left.T
        + kalman_gain @ measurement_covariance @ kalman_gain.T
    )
