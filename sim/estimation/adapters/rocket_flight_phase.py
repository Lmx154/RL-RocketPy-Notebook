"""Rocket-specific flight-phase detection and gravity-update scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from ..core.types import Vector


class RocketFlightPhase(Enum):
    """Detected rocket flight regime used by adapter-layer policies."""

    ON_PAD = auto()
    POWERED_ASCENT = auto()
    COAST = auto()
    DESCENT = auto()


@dataclass(slots=True)
class RocketFlightPhaseConfig:
    """Thresholds for detecting rocket flight phase from estimator signals."""

    powered_accel_excess_mps2: float = 15.0
    burnout_accel_hysteresis_mps2: float = 5.0
    descent_velocity_threshold_mps: float = 2.0


class RocketFlightPhaseDetector:
    """Finite-state flight-phase detector kept outside the generic estimator."""

    def __init__(
        self,
        config: RocketFlightPhaseConfig | None = None,
        gravity_magnitude_mps2: float = 9.80665,
    ) -> None:
        self.config = config or RocketFlightPhaseConfig()
        self.gravity_magnitude_mps2 = float(gravity_magnitude_mps2)
        self.phase = RocketFlightPhase.ON_PAD
        self._launched = False

    def update(
        self,
        *,
        accelerometer_magnitude_mps2: float,
        vertical_velocity_mps: float,
    ) -> RocketFlightPhase:
        """Advance the detector state using current IMU magnitude and vertical velocity."""

        excess_acceleration = abs(
            float(accelerometer_magnitude_mps2) - self.gravity_magnitude_mps2
        )

        if not self._launched:
            if excess_acceleration > self.config.powered_accel_excess_mps2:
                self._launched = True
                self.phase = RocketFlightPhase.POWERED_ASCENT
            return self.phase

        if self.phase is RocketFlightPhase.POWERED_ASCENT:
            if excess_acceleration < self.config.burnout_accel_hysteresis_mps2:
                self.phase = RocketFlightPhase.COAST
            return self.phase

        if self.phase is RocketFlightPhase.COAST:
            if excess_acceleration > self.config.powered_accel_excess_mps2:
                self.phase = RocketFlightPhase.POWERED_ASCENT
            elif float(vertical_velocity_mps) < -self.config.descent_velocity_threshold_mps:
                self.phase = RocketFlightPhase.DESCENT

        return self.phase


@dataclass(frozen=True, slots=True)
class GravityAlignmentDecision:
    """Adapter-layer decision for whether to submit a gravity-alignment update."""

    phase: RocketFlightPhase
    submit_update: bool
    reason: str = ""


class GravityAlignmentFlightPhasePolicy:
    """Disable gravity-alignment updates during powered ascent by default."""

    def __init__(
        self,
        detector: RocketFlightPhaseDetector | None = None,
        *,
        disabled_phases: tuple[RocketFlightPhase, ...] = (RocketFlightPhase.POWERED_ASCENT,),
    ) -> None:
        self.detector = detector or RocketFlightPhaseDetector()
        self.disabled_phases = tuple(disabled_phases)

    @property
    def phase(self) -> RocketFlightPhase:
        return self.detector.phase

    def evaluate(
        self,
        *,
        accelerometer_mps2: Vector,
        vertical_velocity_mps: float,
    ) -> GravityAlignmentDecision:
        """Return whether the replay adapter should submit a gravity update."""

        accelerometer = np.asarray(accelerometer_mps2, dtype=float)
        phase = self.detector.update(
            accelerometer_magnitude_mps2=float(np.linalg.norm(accelerometer)),
            vertical_velocity_mps=float(vertical_velocity_mps),
        )
        submit_update = phase not in self.disabled_phases
        reason = "" if submit_update else f"gravity update disabled during {phase.name.lower()}"
        return GravityAlignmentDecision(
            phase=phase,
            submit_update=submit_update,
            reason=reason,
        )
