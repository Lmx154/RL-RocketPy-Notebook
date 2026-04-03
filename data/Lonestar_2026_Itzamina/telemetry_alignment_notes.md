# Lonestar 2026 Itzamina Telemetry Notes

## Purpose

This folder contains two telemetry exports from the same flight stack position:

- `BlRv_SN2487 LR_03-29-2026_08_47_13.csv`: Featherweight Blue Raven export
- `FLGT0023_TRIMMED.CSV`: MARV flight-computer export

The immediate goal is to normalize both files into canonical analysis streams so they can be aligned and merged later.

## Current Alignment Decision

Use barometric pressure as the primary alignment signal.

Reasons:

- Both datasets contain barometric pressure.
- Pressure is scalar, so axis orientation does not matter.
- Featherweight wall-clock timestamps appear inconsistent in the sampled export, while `Flight_Time_(s)` is monotonic and stable.
- MARV logs raw sample timestamps in microseconds, which are suitable for constructing a clean relative time base.

Do not use derived altitude as the primary alignment signal unless pressure alignment fails. Derived altitude can differ because of device-specific filtering, reference pressure, and internal estimation choices.

## Source Characteristics

### Featherweight Blue Raven

Observed file:

- `3196` data rows
- `Flight_Time_(s)` runs from `0.00` to `63.90`
- fixed `0.02 s` step, about `50 Hz`

Useful channels:

- `Baro_Press_(atm)`
- `Baro_Altitude_ASL_(feet)`
- `Baro_Altitude_AGL_(feet)`
- `Velocity_Up`
- `Velocity_DR`
- `Velocity_CR`
- `Inertial_Altitude`
- `Inertial_DR_Position`
- `Inertial_CR_position`
- `Tilt_Angle_(deg)`
- `Future_Angle_(deg)`
- `Roll_Angle_(deg)`
- flight-state bits such as `Liftoff`, `Apogee`, `Burnout_Coast`, `Apo_fired`, `Main_fired`

Clock note:

- Treat `Flight_Time_(s)` as authoritative for this export.
- Sampled wall-clock `Time` values are inconsistent and should not be trusted for synchronization.

GPS note:

- The current Featherweight CSV in this folder does not expose raw GPS latitude/longitude/altitude fields.
- That means GPS backfill into MARV is not currently possible from this specific export alone.

### MARV

Observed file:

- `14837` data rows
- about `159.39 s` of data in the current checked file
- sample spacing about `10.7` to `11.1 ms`, about `90` to `94 Hz`

Useful channels:

- primary IMU: `imu_ax_mps2`, `imu_ay_mps2`, `imu_az_mps2`, `imu_gx_rad_s`, `imu_gy_rad_s`, `imu_gz_rad_s`
- auxiliary IMU: `aux_imu_ax_mps2`, `aux_imu_ay_mps2`, `aux_imu_az_mps2`, `aux_imu_gx_rad_s`, `aux_imu_gy_rad_s`, `aux_imu_gz_rad_s`
- barometer: `baro_pressure_pa`, `baro_temp_c`
- per-stream sample clocks: `imu_sample_us`, `aux_imu_sample_us`, `baro_sample_us`
- stream freshness markers: `imu_state`, `aux_imu_state`, `baro_state`

Trim note:

- The current `FLGT0023_TRIMMED.CSV` appears to be reduced to a flight-sized window.
- Later phases should still verify the actual overlap window from pressure rather than assuming the trim is exact.

## Overlap Between Devices

Direct overlap today:

- barometric pressure
- barometric temperature
- flight timing progression
- motion-related derived behavior during launch, coast, and descent

Indirect overlap:

- Featherweight derived velocities and inertial positions
- MARV raw IMU signals

These indirect channels may help with second-pass refinement, but they should not be the first alignment anchor.

## Current Real-Data Blocking Issue

The current Featherweight export is not just weakly aligned. Its baro stream appears to contain repeated exact sample runs:

- a short corrupt leading stub appears before the main pressure band
- later sections repeat exact baro segments at about `3.92 s` and `4.00 s` lags
- local baro snippets can match MARV strongly, but they match in multiple places

That means the real step-4 ambiguity is driven partly by source-data repetition, not only by solver tuning. The alignment tool should report this explicitly and should not automatically feed a merged CSV unless the input quality and offset confidence are both acceptable.

## Canonical Normalized Streams

The normalization tool should emit the following stream-level CSVs:

### Featherweight

- `featherweight_baro.csv`
- `featherweight_navigation.csv`
- `featherweight_events.csv`

Canonical conventions:

- `time_s` from `Flight_Time_(s)`
- pressure in pascals
- altitude in meters
- velocity in meters per second
- temperature in both Celsius and Fahrenheit when useful

### MARV

- `marv_primary_imu.csv`
- `marv_aux_imu.csv`
- `marv_baro.csv`

Canonical conventions:

- `time_s` derived from sensor sample timestamps relative to one common dataset origin
- `log_time_s` retained separately for traceability
- repeated sensor samples collapsed by sample timestamp
- pressure in pascals
- IMU values kept in SI units as logged

## Immediate Deliverables

Phase 1:

- document the formats and normalization rules
- normalize both source files into clean typed streams

Phase 2:

- isolate the MARV flight window
- align both datasets with barometric pressure and pressure derivative

Phase 3:

- emit an aligned merged CSV
- evaluate optional motion-based refinement and, later, estimator-based fusion
