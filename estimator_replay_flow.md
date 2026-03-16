# Estimator Replay Flow (ESKF + EKF)

This document explains how `run_rocketpy_replay(...)` processes telemetry, where ESKF "error" comes from, and how the notebook plots (`Estimator Error Norms`, `Covariance Health`) are produced.

## System Summary

The replay stack is layered:

- Attitude layer: `GenericESKF` (state: quaternion + gyro bias)
- Navigation layer: `GenericEKF` (state: position + velocity + accel bias)
- Adapter: `rocketpy_replay.py` reads CSV telemetry and feeds both layers

The key point is that the ESKF does **not** read truth to get error. It estimates error-state corrections from measurement innovation (`z - h(x)`).

## Mermaid Diagram

```mermaid
flowchart TD
    A[Telemetry CSV\nvirtual_sensors_full_rate_*.csv] --> B[run_rocketpy_replay]

    B --> C[Extract per row:\ntime, accel, gyro, baro, GNSS]

    C --> D{Has previous sample\nand finite IMU?}
    D -- yes --> E[Predict step]
    D -- no --> F[Skip predict for first/invalid row]

    E --> E1[Attitude ESKF predict\ncontrol: gyroscope]
    E1 --> E2[Compute R_b_to_i from estimated quaternion]
    E2 --> E3[Navigation EKF predict\ncontrol: accelerometer + R_b_to_i]

    C --> G{Gravity update allowed?}
    G -- yes --> G1[Attitude ESKF update\nmeasurement: accelerometer]

    C --> H{Barometer valid?}
    H -- yes --> H1[Pressure -> altitude\nNavigation EKF update]

    C --> I{GNSS valid?}
    I -- yes --> I1[Navigation EKF position update]
    I1 --> I2{Can derive GNSS velocity?}
    I2 -- yes --> I3[Navigation EKF velocity update]

    E3 --> J[LayeredNavigationStack snapshot]
    G1 --> J
    H1 --> J
    I1 --> J
    I3 --> J

    J --> K[_build_result_row]
    K --> L[estimates DataFrame]

    L --> M[Notebook comparison with truth\n(position/velocity/altitude errors)]
    L --> N[Notebook covariance health\nposition_sigma, velocity_sigma, attitude_sigma, trace_p]
```

## Detailed Step-by-Step

## 1) Telemetry ingestion

`run_rocketpy_replay(...)`:

- Loads and sorts telemetry by `time_column`
- Extracts vectors each row:
  - accelerometer: `accelerometer_columns`
  - gyroscope: `gyroscope_columns`
  - optional barometer and GNSS

## 2) Prediction path

When `dt > 0` and IMU is finite, replay calls:

- `stack.predict(accelerometer_mps2=..., gyroscope_rps=..., dt=...)`

Inside `LayeredNavigationStack.predict(...)`:

1. Attitude ESKF predicts using gyro.
2. The updated attitude quaternion is converted to `R_b_to_i`.
3. Navigation EKF predicts using accelerometer and that rotation matrix.

So the navigation transition matrix depends on attitude output each step.

## 3) Update path

Per row, replay may submit:

- Attitude gravity-alignment update (accelerometer-based)
- Navigation barometric altitude update
- Navigation GNSS position update
- Navigation GNSS velocity update (derived by finite difference)

## 4) Where ESKF "gets the error"

In ESKF update:

- Predicted measurement: `h(x)` from current nominal state
- Innovation: `y = z - h(x)`
- Correction: `delta_x = K y`
- Injection: nominal state is corrected by `delta_x`
- Covariance reset uses the reset Jacobian

For attitude gravity alignment specifically:

- `z` is measured accelerometer vector
- `h(x)` is predicted gravity vector in body frame from current quaternion
- Innovation is the difference between those two vectors

This is an **internal residual-based error estimate**, not truth-minus-estimate.

## 5) How notebook plots are formed

### Estimator Error Norms

These are computed in notebook post-processing by comparing replay estimates with RocketPy kinematic truth:

- `position_error_m = ||est_position - truth_position||`
- `velocity_error_mps = ||est_velocity - truth_velocity||`
- `altitude_error_m = est_z - truth_z`

Therefore, this plot is an external performance metric, not directly the filter innovation/error-state.

### Covariance Health

These values come from `_build_result_row(...)` and are internal filter uncertainty summaries:

- `attitude_sigma_rad`: from ESKF covariance attitude block
- `position_sigma_m`: from EKF covariance position block
- `velocity_sigma_mps`: from EKF covariance velocity block
- `trace_p`: trace(attitude_covariance) + trace(navigation_covariance)

So this plot is mixed-source: one curve from ESKF, two from EKF, and one combined total.

## Quick Source Map

- Replay adapter: `sim/estimation/adapters/rocketpy_replay.py`
- Layer composition: `sim/estimation/stacks/layered_navigation.py`
- Generic filters:
  - `sim/estimation/core/eskf.py`
  - `sim/estimation/core/ekf.py`
- Gravity measurement model: `sim/estimation/measurements/imu_gravity.py`
- Notebook plotting/comparison: `notebooks/Itzamna/Itzamna_flight_sim.ipynb`

## Practical Interpretation of Your Current Plots

- Tight truth-vs-estimate altitude tracking means the overall layered estimator is behaving consistently for this run.
- Spikes in error norms can happen near initialization, mode transitions, or GNSS/baro update timing.
- A fast initial drop in `trace_p` usually indicates rapid uncertainty collapse once updates start arriving.
- The `Covariance Health` panel should be interpreted by channel:
  - attitude sigma: orientation uncertainty behavior (ESKF)
  - position/velocity sigma: translational uncertainty behavior (EKF)
  - `trace_p`: total confidence trend across both layers
