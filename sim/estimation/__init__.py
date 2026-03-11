"""State estimation framework — layered architecture (phase 8).

Public API
----------
The generic estimation framework is organized into sub-packages:

* ``core``          – generic EKF / ESKF engines and Gaussian utilities
* ``math``          – reusable quaternion and rotation helpers
* ``models``        – strapdown, attitude, and navigation process models
* ``measurements``  – nonlinear sensor measurement models
* ``policies``      – optional gating / scheduling policies
* ``stacks``        – layer composition (``LayeredNavigationStack``)
* ``adapters``      – **RocketPy-specific** replay, flight-phase, conversions

Top-level convenience re-exports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The symbols below are re-exported so that ``from sim.estimation import ...``
continues to work for common items.  Rocket-specific adapters are intentionally
*not* re-exported here; import them from ``sim.estimation.adapters`` directly.
"""

# --- generic core -----------------------------------------------------------
from .core import (
    CovarianceConvention,
    CovarianceUpdateForm,
    GenericEKF,
    GenericESKF,
    MeasurementUpdateResult,
    MeasurementUpdateStatus,
    regularize_covariance,
    symmetrize_covariance,
)

# --- math --------------------------------------------------------------------
from .math import (
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_vector_to_quaternion,
    skew_symmetric,
)

# --- models ------------------------------------------------------------------
from .models import (
    AttitudeErrorStateProcessModel,
    AttitudeProcessModelConfig,
    AttitudeProcessNoise,
    AttitudeState,
    NavigationProcessModel,
    NavigationProcessModelConfig,
    NavigationProcessNoise,
    NavigationState,
    StrapdownInertialProcessModel,
)

# --- measurements ------------------------------------------------------------
from .measurements import (
    BarometricAltitudeConfig,
    BarometricAltitudeMeasurementModel,
    GpsPositionConfig,
    GpsPositionMeasurementModel,
    GpsVelocityConfig,
    GpsVelocityMeasurementModel,
    GravityAlignmentConfig,
    GravityAlignmentMeasurementModel,
)

# --- policies ----------------------------------------------------------------
from .policies import GateDecision, GateStatus, MeasurementGate

# --- stacks ------------------------------------------------------------------
from .stacks import (
    LayeredNavigationCovariance,
    LayeredNavigationDiagnostics,
    LayeredNavigationSnapshot,
    LayeredNavigationStack,
    LayeredNavigationState,
)

__all__ = [
    # core
    "CovarianceConvention",
    "CovarianceUpdateForm",
    "GenericEKF",
    "GenericESKF",
    "MeasurementUpdateResult",
    "MeasurementUpdateStatus",
    "regularize_covariance",
    "symmetrize_covariance",
    # math
    "normalize_quaternion",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "rotation_vector_to_quaternion",
    "skew_symmetric",
    # models
    "AttitudeErrorStateProcessModel",
    "AttitudeProcessModelConfig",
    "AttitudeProcessNoise",
    "AttitudeState",
    "NavigationProcessModel",
    "NavigationProcessModelConfig",
    "NavigationProcessNoise",
    "NavigationState",
    "StrapdownInertialProcessModel",
    # measurements
    "BarometricAltitudeConfig",
    "BarometricAltitudeMeasurementModel",
    "GpsPositionConfig",
    "GpsPositionMeasurementModel",
    "GpsVelocityConfig",
    "GpsVelocityMeasurementModel",
    "GravityAlignmentConfig",
    "GravityAlignmentMeasurementModel",
    # policies
    "GateDecision",
    "GateStatus",
    "MeasurementGate",
    # stacks
    "LayeredNavigationCovariance",
    "LayeredNavigationDiagnostics",
    "LayeredNavigationSnapshot",
    "LayeredNavigationStack",
    "LayeredNavigationState",
]
