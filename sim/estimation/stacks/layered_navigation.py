"""Layer composition for the phase-6 layered navigation estimator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core import (
    CovarianceConvention,
    GenericEKF,
    GenericESKF,
    MeasurementUpdateResult,
    PredictionDiagnostics,
)
from ..core.types import Matrix, Vector
from ..math import quaternion_to_rotation_matrix
from ..measurements import (
    BarometricAltitudeMeasurementModel,
    GpsPositionMeasurementModel,
    GpsVelocityMeasurementModel,
    GravityAlignmentMeasurementModel,
)
from ..models import (
    AttitudeErrorStateProcessModel,
    AttitudeInput,
    AttitudeState,
    NavigationInput,
    NavigationProcessModel,
    NavigationState,
)


def _default_attitude_initial_covariance() -> Matrix:
    return np.diag(
        np.array(
            [
                *([np.deg2rad(10.0) ** 2] * 3),
                *([np.deg2rad(1.0 / 60.0) ** 2] * 3),
            ],
            dtype=float,
        )
    )


def _default_navigation_initial_covariance() -> Matrix:
    return np.diag(
        np.array(
            [
                *([100.0] * 3),
                *([25.0] * 3),
                *([0.5 ** 2] * 3),
            ],
            dtype=float,
        )
    )


@dataclass(slots=True)
class LayeredNavigationState:
    """Unified nominal state exposed by the layered composition module."""

    quaternion: Vector = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    )
    gyro_bias_rps: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    position_m: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity_mps: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))
    accel_bias_mps2: Vector = field(default_factory=lambda: np.zeros(3, dtype=float))

    def copy(self) -> "LayeredNavigationState":
        return LayeredNavigationState(
            quaternion=np.asarray(self.quaternion, dtype=float).copy(),
            gyro_bias_rps=np.asarray(self.gyro_bias_rps, dtype=float).copy(),
            position_m=np.asarray(self.position_m, dtype=float).copy(),
            velocity_mps=np.asarray(self.velocity_mps, dtype=float).copy(),
            accel_bias_mps2=np.asarray(self.accel_bias_mps2, dtype=float).copy(),
        )

    def to_attitude_state(self) -> AttitudeState:
        return AttitudeState(
            quaternion=np.asarray(self.quaternion, dtype=float).copy(),
            gyro_bias_rps=np.asarray(self.gyro_bias_rps, dtype=float).copy(),
        )

    def to_navigation_state(self) -> NavigationState:
        return NavigationState(
            position_m=np.asarray(self.position_m, dtype=float).copy(),
            velocity_mps=np.asarray(self.velocity_mps, dtype=float).copy(),
            accel_bias_mps2=np.asarray(self.accel_bias_mps2, dtype=float).copy(),
        )


@dataclass(slots=True)
class LayeredNavigationCovariance:
    """Separate covariance blocks owned by the two composed filters."""

    attitude: Matrix = field(default_factory=_default_attitude_initial_covariance)
    navigation: Matrix = field(default_factory=_default_navigation_initial_covariance)

    def copy(self) -> "LayeredNavigationCovariance":
        return LayeredNavigationCovariance(
            attitude=np.asarray(self.attitude, dtype=float).copy(),
            navigation=np.asarray(self.navigation, dtype=float).copy(),
        )


@dataclass(frozen=True, slots=True)
class LayeredNavigationDiagnostics:
    """Last diagnostics exposed by the composition layer."""

    attitude_prediction: PredictionDiagnostics | None
    navigation_prediction: PredictionDiagnostics | None
    attitude_update: MeasurementUpdateResult[Any] | None
    navigation_update: MeasurementUpdateResult[Any] | None


@dataclass(frozen=True, slots=True)
class LayeredNavigationSnapshot:
    """Unified estimator snapshot exposed by the composition layer."""

    timestamp_s: float | None
    prediction_dt_s: float | None
    state: LayeredNavigationState
    covariance: LayeredNavigationCovariance
    inertial_acceleration_mps2: Vector
    diagnostics: LayeredNavigationDiagnostics


class LayeredNavigationStack:
    """Compose the attitude ESKF and navigation EKF without domain logic."""

    def __init__(
        self,
        *,
        initial_state: LayeredNavigationState | None = None,
        initial_covariance: LayeredNavigationCovariance | None = None,
        attitude_process_model: AttitudeErrorStateProcessModel | None = None,
        navigation_process_model: NavigationProcessModel | None = None,
        gravity_alignment_model: GravityAlignmentMeasurementModel | None = None,
        position_measurement_model: GpsPositionMeasurementModel | None = None,
        velocity_measurement_model: GpsVelocityMeasurementModel | None = None,
        barometric_altitude_model: BarometricAltitudeMeasurementModel | None = None,
        attitude_covariance_convention: CovarianceConvention | None = None,
        navigation_covariance_convention: CovarianceConvention | None = None,
    ) -> None:
        nominal_state = initial_state.copy() if initial_state is not None else LayeredNavigationState()
        covariance = initial_covariance.copy() if initial_covariance is not None else LayeredNavigationCovariance()

        self.attitude_process_model = attitude_process_model or AttitudeErrorStateProcessModel()
        self.navigation_process_model = navigation_process_model or NavigationProcessModel()
        self.gravity_alignment_model = gravity_alignment_model or GravityAlignmentMeasurementModel()
        self.position_measurement_model = position_measurement_model or GpsPositionMeasurementModel()
        self.velocity_measurement_model = velocity_measurement_model or GpsVelocityMeasurementModel()
        self.barometric_altitude_model = (
            barometric_altitude_model or BarometricAltitudeMeasurementModel()
        )

        self.attitude = GenericESKF(
            process_model=self.attitude_process_model,
            initial_state=nominal_state.to_attitude_state(),
            initial_covariance=covariance.attitude,
            covariance_convention=attitude_covariance_convention or CovarianceConvention(),
        )
        self.navigation = GenericEKF(
            process_model=self.navigation_process_model,
            initial_state=nominal_state.to_navigation_state(),
            initial_covariance=covariance.navigation,
            covariance_convention=navigation_covariance_convention or CovarianceConvention(),
        )

        self.last_timestamp_s: float | None = None
        self.last_prediction_dt_s: float | None = None
        self.last_inertial_acceleration_mps2 = np.zeros(3, dtype=float)

    @property
    def state(self) -> LayeredNavigationState:
        return LayeredNavigationState(
            quaternion=np.asarray(self.attitude.state.quaternion, dtype=float).copy(),
            gyro_bias_rps=np.asarray(self.attitude.state.gyro_bias_rps, dtype=float).copy(),
            position_m=np.asarray(self.navigation.state.position_m, dtype=float).copy(),
            velocity_mps=np.asarray(self.navigation.state.velocity_mps, dtype=float).copy(),
            accel_bias_mps2=np.asarray(self.navigation.state.accel_bias_mps2, dtype=float).copy(),
        )

    @property
    def covariance(self) -> LayeredNavigationCovariance:
        return LayeredNavigationCovariance(
            attitude=np.asarray(self.attitude.covariance, dtype=float).copy(),
            navigation=np.asarray(self.navigation.covariance, dtype=float).copy(),
        )

    @property
    def diagnostics(self) -> LayeredNavigationDiagnostics:
        return LayeredNavigationDiagnostics(
            attitude_prediction=self.attitude.last_prediction,
            navigation_prediction=self.navigation.last_prediction,
            attitude_update=self.attitude.last_update,
            navigation_update=self.navigation.last_update,
        )

    def predict(
        self,
        *,
        accelerometer_mps2: Vector,
        gyroscope_rps: Vector,
        dt: float,
        timestamp_s: float | None = None,
    ) -> LayeredNavigationDiagnostics:
        attitude_prediction = self.attitude.predict(
            control=AttitudeInput(gyroscope_rps=np.asarray(gyroscope_rps, dtype=float)),
            dt=dt,
        )
        rotation_body_to_inertial = quaternion_to_rotation_matrix(self.attitude.state.quaternion)
        navigation_control = NavigationInput(
            accelerometer_mps2=np.asarray(accelerometer_mps2, dtype=float),
            rotation_body_to_inertial=rotation_body_to_inertial,
        )
        self.last_inertial_acceleration_mps2 = self.navigation_process_model.inertial_acceleration(
            self.navigation.state,
            navigation_control,
        ).copy()
        navigation_prediction = self.navigation.predict(control=navigation_control, dt=dt)

        self.last_prediction_dt_s = float(dt)
        self.last_timestamp_s = timestamp_s
        return LayeredNavigationDiagnostics(
            attitude_prediction=attitude_prediction,
            navigation_prediction=navigation_prediction,
            attitude_update=self.attitude.last_update,
            navigation_update=self.navigation.last_update,
        )

    def update_gravity_alignment(
        self,
        *,
        accelerometer_mps2: Vector,
        measurement_gate=None,
    ) -> MeasurementUpdateResult[Vector]:
        return self.attitude.update(
            measurement_model=self.gravity_alignment_model,
            measurement=np.asarray(accelerometer_mps2, dtype=float),
            measurement_gate=measurement_gate,
        )

    def update_position(
        self,
        *,
        position_m: Vector,
        measurement_gate=None,
    ) -> MeasurementUpdateResult[Vector]:
        return self.navigation.update(
            measurement_model=self.position_measurement_model,
            measurement=np.asarray(position_m, dtype=float),
            measurement_gate=measurement_gate,
        )

    def update_velocity(
        self,
        *,
        velocity_mps: Vector,
        measurement_gate=None,
    ) -> MeasurementUpdateResult[Vector]:
        return self.navigation.update(
            measurement_model=self.velocity_measurement_model,
            measurement=np.asarray(velocity_mps, dtype=float),
            measurement_gate=measurement_gate,
        )

    def update_barometric_altitude(
        self,
        *,
        altitude_m: float,
        measurement_gate=None,
    ) -> MeasurementUpdateResult[float]:
        return self.navigation.update(
            measurement_model=self.barometric_altitude_model,
            measurement=float(altitude_m),
            measurement_gate=measurement_gate,
        )

    def snapshot(self, *, timestamp_s: float | None = None) -> LayeredNavigationSnapshot:
        return LayeredNavigationSnapshot(
            timestamp_s=self.last_timestamp_s if timestamp_s is None else float(timestamp_s),
            prediction_dt_s=self.last_prediction_dt_s,
            state=self.state,
            covariance=self.covariance,
            inertial_acceleration_mps2=self.last_inertial_acceleration_mps2.copy(),
            diagnostics=self.diagnostics,
        )
