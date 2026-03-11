"""External gating contracts for the additive estimator rewrite."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..core.types import Matrix, MeasurementT, StateT, Vector

if TYPE_CHECKING:
    from ..core.base import MeasurementModel


class GateStatus(str, Enum):
    """Allowed outcomes from a measurement-gating policy."""

    ACCEPT = "accept"
    REJECT = "reject"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Decision returned by an external gating or scheduling policy."""

    status: GateStatus
    reason: str = ""
    mahalanobis_distance: float | None = None


@runtime_checkable
class MeasurementGate(Protocol[StateT, MeasurementT]):
    """Policy wrapper for gating that stays outside measurement-model math."""

    def evaluate(
        self,
        *,
        measurement_model: "MeasurementModel[StateT, MeasurementT]",
        measurement: MeasurementT,
        nominal_state: StateT,
        innovation: Vector,
        innovation_covariance: Matrix,
    ) -> GateDecision:
        """Return an external accept/reject/skip decision for a measurement."""
