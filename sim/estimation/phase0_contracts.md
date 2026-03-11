# Phase 0 Contracts

This package now contains a parallel contract layer for the estimator rewrite.
The current implementation in `sim/estimation/eskf.py` remains the active
runtime path. Phase 0 only freezes interfaces and responsibilities so later
phases can peel behavior away from the monolith without redesigning contracts
mid-migration.

## Files introduced in phase 0

- `sim/estimation/core/types.py`
- `sim/estimation/core/gaussian.py`
- `sim/estimation/core/base.py`
- `sim/estimation/policies/gating.py`

## Decisions frozen in phase 0

1. Nominal state protocol
   Manifold and Euclidean nominal states both implement `copy()`.
   Euclidean states additionally implement `plus(delta)` for additive EKF
   correction in state coordinates.

2. Process-model contract for `F`, `G`, and `Q`
   Every process model exposes:
   - `predict(state, control, dt) -> state`
   - `process_noise_jacobian(state, control, dt) -> G`
   - `process_noise_covariance(state, control, dt) -> Q`

   EKF models additionally expose `state_jacobian(state, control, dt) -> F`.
   ESKF models additionally expose `error_state_jacobian(state, control, dt) -> F`.

3. Measurement-model contract for `h(x)`, `H(x)`, and `R`
   Every measurement model exposes:
   - `predict_measurement(state) -> h(x)`
   - `measurement_jacobian(measurement, state) -> H`
   - `measurement_covariance(measurement, state) -> R`

   Measurement models also expose `innovation(measurement, prediction) -> y`
   so manifold or wrapped measurement spaces are supported without teaching
   the filter core sensor-specific residual math.

4. ESKF injection and reset
   Every ESKF process model exposes:
   - `inject(state, error_state) -> state`
   - `reset_jacobian(injected_error_state) -> J_r`

5. Gating location
   Gating is an external policy concern. Measurement models stay focused on
   sensor prediction and covariance math. Policies under
   `sim/estimation/policies/` decide whether a measurement is accepted,
   rejected, or skipped.

## Frozen engine responsibilities

`EKF core`
- Consume Euclidean state/process/measurement contracts.
- Propagate covariance.
- Compute innovations and Kalman updates.
- Apply additive state correction through `plus(delta)`.
- Use Joseph-form covariance updates and shared covariance conventions.

`ESKF core`
- Consume manifold nominal states and error-state process models.
- Propagate covariance in error coordinates.
- Compute innovations and Kalman updates.
- Apply correction through model-level `inject(...)`.
- Apply covariance reset through `reset_jacobian(...)`.

## Explicit non-responsibilities

The future filter core does not own:
- rocket flight-phase detection
- powered-ascent trust scheduling
- telemetry schema interpretation
- RocketPy replay assumptions
- sensor-specific gating heuristics hidden inside the filter engine
