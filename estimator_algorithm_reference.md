# State Estimation Framework — Algorithm Reference

This document describes the mathematical models, filter algorithms, and software
architecture of the layered inertial navigation estimator implemented in
`sim/estimation/`.  It is intended as a self-contained reference for anyone who
needs to understand what the code is doing, tune noise parameters, or extend the
estimator with new sensor models.

---

## 1. System Overview

The estimator reconstructs a rocket's full 15-degree-of-freedom kinematic state
(orientation, gyro bias, position, velocity, accelerometer bias) from noisy
sensor measurements.  It is organized as two **decoupled** Kalman filters that
run in series at every timestep:

```
 Raw IMU (accel + gyro) + dt
              │
              ▼
    ┌─────────────────────┐
    │  Attitude ESKF      │    6 DoF error-state filter on SO(3)
    │  q, b_g             │    corrected by gravity alignment
    └────────┬────────────┘
             │  extract R_b→i (body-to-inertial rotation)
             ▼
    ┌─────────────────────┐
    │  Navigation EKF     │    9 DoF Euclidean filter
    │  p, v, b_a          │    corrected by GNSS + barometer
    └────────┬────────────┘
             │
             ▼
       Final state + covariance + diagnostics
```

**Key design rule:**  The generic filter engines (`GenericEKF`, `GenericESKF`)
know nothing about rockets, flight phases, or sensor hardware.  All
domain-specific logic lives in the adapter layer (`sim/estimation/adapters/`).

---

## 2. State Vectors

### 2.1 Attitude Layer (ESKF)

The attitude filter tracks orientation and gyroscope bias.

| Component | Symbol | Dimension | Units |
|-----------|--------|-----------|-------|
| Quaternion (scalar-first) | $\mathbf{q} = [q_0,\, q_1,\, q_2,\, q_3]^T$ | 4 | — |
| Gyro bias | $\mathbf{b}_g$ | 3 | rad/s |

Because quaternions live on the $S^3$ manifold, the filter operates on a
**6-dimensional error state** rather than the 7-dimensional nominal state:

$$
\delta \mathbf{x}_a
= \begin{bmatrix} \delta \boldsymbol{\theta} \\ \delta \mathbf{b}_g \end{bmatrix}
\in \mathbb{R}^6
$$

where $\delta\boldsymbol{\theta}$ is a 3D rotation vector (small-angle
perturbation).  This is the standard error-state Kalman filter (ESKF) approach
for attitude estimation — covariance propagation and Kalman updates happen in
the tangent space, then the correction is **injected** back onto the manifold
via the exponential map.

### 2.2 Navigation Layer (EKF)

The navigation filter is a conventional Euclidean EKF:

| Component | Symbol | Dimension | Units |
|-----------|--------|-----------|-------|
| Position | $\mathbf{p}$ | 3 | m |
| Velocity | $\mathbf{v}$ | 3 | m/s |
| Accel bias | $\mathbf{b}_a$ | 3 | m/s² |

$$
\mathbf{x}_n
= \begin{bmatrix} \mathbf{p} \\ \mathbf{v} \\ \mathbf{b}_a \end{bmatrix}
\in \mathbb{R}^9
$$

This layer receives the rotation matrix $R_{b \to i}$ from the attitude layer to transform accelerometer readings into the inertial frame;
it does **not** re-estimate orientation.

---

## 3. Prediction (Process Model)

At every IMU sample, two prediction steps execute in sequence.

### 3.1 Attitude Prediction

Given gyroscope measurement $\boldsymbol{\omega}_m$ and current bias estimate
$\mathbf{b}_g$:

$$
\boldsymbol{\omega}_{corr} = \boldsymbol{\omega}_m - \mathbf{b}_g
$$

**Quaternion propagation** (right-multiplicative):

$$
\mathbf{q}_{k+1}
= \mathbf{q}_k \otimes \mathrm{Exp}\!\bigl(\boldsymbol{\omega}_{corr}\,\Delta t\bigr)
$$

where $\mathrm{Exp}(\cdot)$ is the rotation-vector-to-quaternion exponential map.

**Bias propagation** (random walk):

$$
\mathbf{b}_{g,k+1} = \mathbf{b}_{g,k}
$$

**Error dynamics Jacobian** (continuous-time, then discretized $F = I + F_c\,\Delta t$):

$$
F_c = \begin{bmatrix}
-[\boldsymbol{\omega}_{corr}]_\times & -I_3 \\
0_{3 \times 3} & 0_{3 \times 3}
\end{bmatrix}
$$

where $[\cdot]_\times$ denotes the skew-symmetric (cross-product) matrix.

**Process noise Jacobian:**

$$
G = \begin{bmatrix} -I_3 & 0_3 \\ 0_3 & I_3 \end{bmatrix}
$$

**Process noise covariance** (continuous-time white noise scaled by $\Delta t$):

$$
Q_a = \operatorname{diag}\!\bigl(
  \sigma_g^2,\, \sigma_g^2,\, \sigma_g^2,\,
  \sigma_{b_g}^2,\, \sigma_{b_g}^2,\, \sigma_{b_g}^2
\bigr) \cdot \Delta t
$$

| Parameter | Symbol | Default | Units |
|-----------|--------|---------|-------|
| Gyro noise density | $\sigma_g$ | $2 \times 10^{-3}$ | rad/s |
| Gyro bias random walk | $\sigma_{b_g}$ | $2 \times 10^{-4}$ | rad/s |

**Covariance propagation:**

$$
\bar{P}_a = F\, P_a\, F^T + G\, Q_a\, G^T
$$

### 3.2 Navigation Prediction

Given accelerometer measurement $\mathbf{a}_m$, current accel bias $\mathbf{b}_a$,
and the rotation matrix $R = R_{b \to i}$ from the attitude layer:

**Inertial acceleration:**

$$
\mathbf{a}_{inertial} = R\,(\mathbf{a}_m - \mathbf{b}_a) + \mathbf{g}
$$

where $\mathbf{g} = [0,\, 0,\, -9.80665]^T$ m/s² (local gravity, configurable).

**Kinematic integration:**

$$
\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k\,\Delta t + \tfrac{1}{2}\,\mathbf{a}_{inertial}\,\Delta t^2
$$

$$
\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{a}_{inertial}\,\Delta t
$$

$$
\mathbf{b}_{a,k+1} = \mathbf{b}_{a,k}
$$

**State transition Jacobian:**

$$
F = \begin{bmatrix}
I_3 & I_3\,\Delta t & 0_3 \\
0_3 & I_3 & -R\,\Delta t \\
0_3 & 0_3 & I_3
\end{bmatrix}
$$

**Process noise covariance** (continuous-time noise model integrating position-velocity correlation):

$$
Q_n = \begin{bmatrix}
\frac{1}{3}\sigma_a^2\,\Delta t^3\, I_3 & \frac{1}{2}\sigma_a^2\,\Delta t^2\, I_3 & 0_3 \\
\frac{1}{2}\sigma_a^2\,\Delta t^2\, I_3 & \sigma_a^2\,\Delta t\, I_3 & 0_3 \\
0_3 & 0_3 & \sigma_{b_a}^2\,\Delta t\, I_3
\end{bmatrix}
$$

| Parameter | Symbol | Default | Units |
|-----------|--------|---------|-------|
| Accel noise density | $\sigma_a$ | $8 \times 10^{-2}$ | m/s² |
| Accel bias random walk | $\sigma_{b_a}$ | $2 \times 10^{-2}$ | m/s² |

---

## 4. Measurement Update (Kalman Correction)

All measurement updates share the same generic update equations.  The difference
between the EKF and ESKF is what happens *after* the Kalman gain is applied.

### 4.1 Generic Update Equations

For a measurement $\mathbf{z}$ with predicted measurement $h(\hat{\mathbf{x}})$,
measurement Jacobian $H$, and measurement noise covariance $R$:

**Innovation:**

$$
\mathbf{y} = \mathbf{z} - h(\hat{\mathbf{x}})
$$

**Innovation covariance:**

$$
S = H\,P\,H^T + R
$$

**Kalman gain** (computed via linear solve for numerical stability):

$$
K = P\,H^T\,S^{-1}
$$

**State correction:**

$$
\hat{\mathbf{x}}^+ = \hat{\mathbf{x}}^- + K\,\mathbf{y}
\quad\text{(EKF)}
$$

or in the ESKF case, $\delta\hat{\mathbf{x}} = K\,\mathbf{y}$ is computed in
error space and then injected (see §4.3).

**Covariance update (Joseph form):**

$$
P^+ = (I - K\,H)\,P^-\,(I - K\,H)^T + K\,R\,K^T
$$

The Joseph form is algebraically equivalent to the textbook $(I - KH)P$ but is
**numerically superior** — it preserves positive-definiteness even when
accumulated floating-point roundoff would otherwise cause the standard form to
produce a non-positive-definite covariance.

### 4.2 Measurement Gating

Before applying any update, the filter computes the **Mahalanobis distance**:

$$
d^2 = \mathbf{y}^T\, S^{-1}\, \mathbf{y}
$$

If $d^2$ exceeds a configurable threshold, the measurement is **rejected** as an
outlier and the state/covariance are left unchanged.  This protects against
GNSS glitches, barometer faults, or other sensor anomalies.

### 4.3 ESKF Injection and Reset

After the attitude ESKF computes an error-state correction
$\delta\hat{\mathbf{x}} = [\delta\boldsymbol{\theta},\, \delta\mathbf{b}_g]^T$,
the correction is **injected** onto the nominal state:

$$
\mathbf{q}^+ = \mathbf{q}^- \otimes \mathrm{Exp}(\delta\boldsymbol{\theta})
$$

$$
\mathbf{b}_g^+ = \mathbf{b}_g^- + \delta\mathbf{b}_g
$$

The covariance is then **reset** to account for the manifold geometry after injection:

$$
P^+ \leftarrow J_r\, P^+\, J_r^T
$$

where the reset Jacobian is:

$$
J_r = \begin{bmatrix}
I_3 - \tfrac{1}{2}[\delta\boldsymbol{\theta}]_\times & 0_3 \\
0_3 & I_3
\end{bmatrix}
$$

This accounts for the first-order change of coordinates on $SO(3)$ after
applying the correction.

---

## 5. Measurement Models

### 5.1 Gravity Alignment (Attitude Layer)

Uses the accelerometer to observe the direction of gravity in the body frame.

**Predicted measurement:**

$$
\hat{\mathbf{z}} = R_{i \to b}\, \mathbf{g}
= R(\mathbf{q})^T\, \mathbf{g}
$$

**Innovation:**

$$
\mathbf{y} = \mathbf{a}_{measured} - \hat{\mathbf{z}}
$$

**Measurement Jacobian** (3 × 6):

$$
H = \begin{bmatrix} [\hat{\mathbf{z}}]_\times & 0_{3 \times 3} \end{bmatrix}
$$

The skew-symmetric matrix $[\hat{\mathbf{z}}]_\times$ encodes how small
attitude errors rotate the predicted gravity vector.

**Noise:** $R = \sigma_m^2\, I_3$, with $\sigma_m = 0.75$ m/s².

**Important:** This measurement is only valid when the accelerometer reads
primarily gravity.  During powered flight (motor burning), thrust dominates the
reading and the measurement is **suppressed** by the flight-phase policy.

### 5.2 GNSS Position (Navigation Layer)

**Predicted measurement:** $\hat{\mathbf{z}} = \mathbf{p}$ (direct observation of position).

**Measurement Jacobian** (3 × 9):

$$
H = \begin{bmatrix} I_3 & 0_3 & 0_3 \end{bmatrix}
$$

**Noise:** $R = \operatorname{diag}(3.0^2,\, 3.0^2,\, 5.0^2)$ m².

Geodetic GNSS coordinates (lat, lon, alt) are converted to a local
East-North-Up (ENU) frame anchored at the first GNSS fix before being passed
to the filter.

### 5.3 GNSS Velocity (Navigation Layer)

GNSS velocity is derived by finite-differencing consecutive GNSS position
fixes.  This is an approximation, but provides useful velocity observability
between direct velocity measurements.

**Predicted measurement:** $\hat{\mathbf{z}} = \mathbf{v}$.

**Measurement Jacobian** (3 × 9):

$$
H = \begin{bmatrix} 0_3 & I_3 & 0_3 \end{bmatrix}
$$

**Noise:** $R = \operatorname{diag}(3.0^2,\, 3.0^2,\, 4.0^2)$ (m/s)².

### 5.4 Barometric Altitude (Navigation Layer)

Barometric pressure is converted to altitude using the ISA troposphere model:

$$
h_{baro} = 44330 \left(1 - \left(\frac{p}{p_0}\right)^{0.19029}\right)
$$

where $p_0$ is the sea-level reference pressure (estimated from the first
barometer reading and the known launch-site elevation).

**Predicted measurement:** $\hat{z} = p_z$ (the z-component of position).

**Measurement Jacobian** (1 × 9):

$$
H = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}
$$

**Noise:** $R = 2.0^2 = 4.0$ m².

---

## 6. Flight Phase Policy

The flight-phase detector is an adapter-layer finite state machine that runs
outside the generic estimator.  Its only effect is gating which measurements
are submitted to the attitude filter.

### State Machine

```
  ON_PAD ──────────────────────► POWERED_ASCENT
            |a| − g > 15 m/s²        │
                                      │  |a| − g < 5 m/s²
                                      ▼       (hysteresis)
                                    COAST ◄──────────────┐
                                      │     |a|−g > 15   │
                                      │   (re-ignition)  │
                                      │                   │
                                      │  vz < −2 m/s
                                      ▼
                                   DESCENT  (terminal)
```

| Transition | Condition | Rationale |
|------------|-----------|-----------|
| PAD → POWERED | Accel excess > 15 m/s² | Conservative; sounding rockets produce >50 m/s² excess |
| POWERED → COAST | Accel excess < 5 m/s² | Hysteresis prevents chatter near burnout |
| COAST → DESCENT | Downward velocity > 2 m/s | Confirms rocket is past apogee |

### Gravity Alignment Gating

| Phase | Submit gravity update? | Reason |
|-------|----------------------|--------|
| ON_PAD | Yes | Accelerometer reads pure gravity |
| POWERED_ASCENT | **No** | Thrust dominates; would corrupt attitude |
| COAST | Yes | Only gravity + drag (aero loads small vs. gravity) |
| DESCENT | Yes | Gravity-dominated again |

---

## 7. Layer Composition

The `LayeredNavigationStack` in `sim/estimation/stacks/layered_navigation.py`
wires the two filters together.  The layers are **block-diagonal** — there is
no cross-covariance between attitude and navigation.  The only coupling is the
rotation matrix passed from attitude to navigation at prediction time.

### Covariance Structure

$$
P = \begin{bmatrix}
P_{att} & 0 \\
0 & P_{nav}
\end{bmatrix}
$$

where $P_{att} \in \mathbb{R}^{6 \times 6}$ and $P_{nav} \in \mathbb{R}^{9 \times 9}$.

### Default Initial Uncertainties

| State | Initial $1\sigma$ |
|-------|-------------------|
| Attitude | $10°$ ($\approx 0.17$ rad) |
| Gyro bias | $1°/\text{min}$ ($\approx 2.9 \times 10^{-4}$ rad/s) |
| Position | $100$ m |
| Velocity | $25$ m/s |
| Accel bias | $0.5$ m/s² |

These are intentionally conservative.  The filter is designed to converge from
wide initial uncertainty rather than requiring precise initialization.

### Per-Timestep Sequence

```python
# 1. Attitude predict: integrate gyroscope
attitude_eskf.predict(gyroscope, dt)

# 2. Extract rotation for navigation
R_b2i = quaternion_to_rotation_matrix(attitude_state.quaternion)

# 3. Navigation predict: strapdown integration with corrected accel
navigation_ekf.predict(accelerometer, R_b2i, dt)

# 4. Attitude update: gravity alignment (if flight phase allows)
if policy.submit_update:
    attitude_eskf.update(gravity_alignment_model, accelerometer)

# 5. Navigation updates: barometer, GNSS position, GNSS velocity
navigation_ekf.update(baro_model, baro_altitude)
navigation_ekf.update(gps_position_model, gnss_position)
navigation_ekf.update(gps_velocity_model, gnss_velocity)
```

---

## 8. Software Architecture

The implementation is organized into a strict layered file structure where
dependencies only flow **inward** (adapters → stacks → models/measurements → core → math):

```
sim/estimation/
    core/              Generic filter engines (EKF, ESKF, Gaussian utilities)
        ekf.py         GenericEKF — predict/update for Euclidean states
        eskf.py        GenericESKF — predict/inject/reset for manifold states
        gaussian.py    Joseph update, Mahalanobis gating, covariance utilities
        base.py        Protocol definitions (ProcessModel, MeasurementModel)
        types.py       Type aliases (StateT, Vector, Matrix)

    math/              Rotation and manifold helpers
        quaternion.py  Multiply, normalize, inverse, exp map, q→R conversion
        rotation.py    Skew-symmetric matrix
        jacobians.py   Finite-difference Jacobian (test utility only)

    models/            Process models (state propagation + Jacobians)
        strapdown.py   Combined 9-DoF strapdown inertial model
        attitude.py    6-DoF attitude ESKF model (quaternion + gyro bias)
        navigation.py  9-DoF navigation EKF model (position + velocity + accel bias)

    measurements/      Sensor measurement models (h(x), H, R)
        imu_gravity.py           Gravity alignment from accelerometer
        gps_position.py          GNSS 3D position observation
        gps_velocity.py          GNSS 3D velocity observation
        baro_altitude.py         Barometric altitude from pressure sensor

    policies/          Optional external gating and scheduling
        gating.py      Mahalanobis-distance measurement gate

    stacks/            Composition layers
        layered_navigation.py    Wires attitude ESKF + navigation EKF

    adapters/          RocketPy-specific (not imported by core)
        rocketpy_replay.py       Telemetry CSV replay, geodetic conversion,
                                 barometer conversion, auto-tuning
        rocket_flight_phase.py   Flight-phase FSM, gravity-update policy
```

**Boundary rule:** The `core/`, `math/`, `models/`, `measurements/`, `policies/`,
and `stacks/` packages must be importable and usable **without** loading anything
from `adapters/`.  This is enforced by a test that imports the core stack in a
subprocess and asserts that no adapter module appears in `sys.modules`.

---

## 9. Tuning Guide

### Increasing / Decreasing Filter Responsiveness

| Want to... | Adjust | Direction |
|------------|--------|-----------|
| Track GNSS more tightly | `gnss_position_std_m` | Decrease |
| Trust IMU integration more | `accel_noise_density` / `gyro_noise_density` | Decrease |
| Allow faster bias convergence | `accel_bias_random_walk` / `gyro_bias_random_walk` | Increase |
| Smooth through GNSS dropouts | `gnss_position_std_m` | Increase |
| Reduce attitude drift | `gyro_bias_random_walk` | Increase (allows faster bias tracking) |
| Trust barometer more for altitude | `baro_altitude_std_m` | Decrease |

### Flight Phase Thresholds

| Parameter | Default | Effect |
|-----------|---------|--------|
| `powered_accel_excess_mps2` | 15 m/s² | Higher → delays POWERED detection (more conservative) |
| `burnout_accel_hysteresis_mps2` | 5 m/s² | Lower → faster COAST detection after burnout |
| `descent_velocity_threshold_mps` | 2 m/s | Higher → delays DESCENT detection |

---

## 10. Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\mathbf{q}$ | Unit quaternion (scalar-first: $[q_0, q_1, q_2, q_3]$) |
| $\otimes$ | Quaternion multiplication |
| $\mathrm{Exp}(\boldsymbol{\theta})$ | Rotation vector → quaternion via exponential map |
| $R(\mathbf{q})$ | Rotation matrix (body-to-inertial) from quaternion |
| $[\mathbf{v}]_\times$ | $3 \times 3$ skew-symmetric matrix of vector $\mathbf{v}$ |
| $\delta\boldsymbol{\theta}$ | Small-angle rotation error vector (tangent space of $SO(3)$) |
| $P$ | State error covariance matrix |
| $K$ | Kalman gain |
| $S$ | Innovation covariance |
| $H$ | Measurement Jacobian ($\partial h / \partial x$) |
| $F$ | State transition Jacobian (discrete) |
| $G$ | Process noise input Jacobian |
| $Q$ | Process noise covariance |
| $R$ | Measurement noise covariance |
