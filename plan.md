**Integration Plan**

**Phase 1: Freeze the Session Contract**
Define one session as one directory plus one manifest. This becomes the only runtime input model.

- Add a session spec document and a `manifest.json` schema.
- Define required files: `truth.csv`, `imu.csv`, `baro.csv`, `gps.csv`.
- Define optional files: `mag.csv`, future `estimator_feedback.csv`, `device_events.csv`.
- Define canonical rates for first pass: truth `500 Hz`, IMU `300 Hz`, baro `100 Hz`, mag `50 Hz`, GPS `10 Hz`.
- Define canonical columns for each CSV and required metadata in the manifest.
- Proposed new module: [sim/sitl/session.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/sitl/session.py).

Exit criteria: session schema is fixed and all later phases target it.

**Phase 2: Update Exporters to Produce Sessions**
Replace the current two-file export flow with per-session multi-stream export.

- Update [notebooks/Itzamna/rocket_config.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/notebooks/Itzamna/rocket_config.py) to use `300 / 100 / 10` and define the truth rate `500 Hz`.
- Update [notebooks/Itzamna/telemetry_logger.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/notebooks/Itzamna/telemetry_logger.py) to export a session directory instead of `virtual_sensors_full_rate_*.csv`.
- Generate `truth.csv` on a fixed 500 Hz timeline by evaluating the flight state uniformly.
- Export `imu.csv`, `baro.csv`, and `gps.csv` directly from RocketPy `measured_data`.
- Treat `mag.csv` as optional and only export it when a real source exists.
- Write `manifest.json` with session metadata and file references.

Exit criteria: one export run creates one self-contained session folder.

**Phase 3: Build a Session Loader**
Replace suffix-based pairing with manifest-based loading.

- Remove the current `load_replay_pair(...)` dependency in [sim/simulation/simulation_controller.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/simulation/simulation_controller.py#L482).
- Add `load_replay_session(...)` in [sim/sitl/session.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/sitl/session.py) or a neighboring loader module.
- Validate manifest completeness, file existence, column correctness, monotonic timestamps, and session consistency.
- Update [sim/gui/rocket_viewer_app.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/gui/rocket_viewer_app.py#L1050) so the user opens a session folder or manifest, not separate CSVs.

Exit criteria: the runtime opens exactly one session artifact and never guesses file pairings.

**Phase 4: Replace Row-Aligned Replay with a Multi-Rate Scheduler**
Stop aligning all sensors onto kinematics rows.

- Remove stale-hold alignment logic in [sim/simulation/simulation_controller.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/simulation/simulation_controller.py#L518).
- Introduce a replay core with one truth cursor and one cursor per sensor stream.
- At each truth tick, emit only the sensor samples whose own timestamps are due.
- Keep simulator ownership of `play`, `pause`, `seek`, `step`, and replay speed.
- Preserve a unified state object for the GUI, but derive it from the session scheduler rather than merged sensor rows.
- Proposed new module: [sim/sitl/replay_session.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/sitl/replay_session.py).

Exit criteria: truth and sensors advance by timestamp, not by shared row index.

**Phase 5: Unify MAVLink HIL Encoding**
Remove duplicated packet-building logic and make serial HIL the only transport path.

- Extract MAVLink encode helpers from [sim/sitl/mavlink_sitl_service.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/sitl/mavlink_sitl_service.py) and [sim/sitl/adapters.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/sitl/adapters.py) into one shared codec module.
- Proposed new module: [sim/sitl/mavlink_codec.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/sitl/mavlink_codec.py).
- Encode `SYSTEM_TIME`, `HIL_SENSOR`, and `HIL_GPS` from scheduled stream updates.
- Decide how magnetometer is represented in `HIL_SENSOR` when available; until then keep mag fields zeroed or absent by policy.

Exit criteria: only one implementation exists for MAVLink HIL packet construction.

**Phase 6: Remove UDP Completely**
Cut out all UDP transport, UDP control, and UDP-oriented adapters.

- Delete [sim/sitl/udp_bridge.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/sitl/udp_bridge.py).
- Remove UDP branches from [sim/sitl/mavlink_sitl_service.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/sitl/mavlink_sitl_service.py).
- Remove JSON replay adapter and any UDP-only abstractions from [sim/sitl/adapters.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/sitl/adapters.py), or delete the file if fully obsolete.
- Remove UDP mentions from [sim/sitl/__init__.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/sitl/__init__.py), [README.md](/home/luis/Projects/Python/RL-RocketPy-Notebook/README.md), and GUI labels/tooltips.

Exit criteria: simulator HIL is serial-only.

**Phase 7: Add Inbound Feedback as a Real Module**
Replace GUI-only raw MAVLink command mapping with typed simulator feedback.

- Add [sim/sitl/estimator_feedback.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/sitl/estimator_feedback.py).
- Define typed feedback domains: estimator output, device state-machine events, SD/logging events.
- Parse inbound MAVLink messages into those typed objects.
- Let the GUI consume feedback objects rather than decoding raw messages directly.
- Keep the existing overlay as a presentation layer only.

Exit criteria: inbound device output is structured and reusable outside the GUI.

**Phase 8: Rewire the GUI Around Sessions and Typed Feedback**
Make the viewer a client of the new session and feedback layers.

- Update [sim/gui/rocket_viewer_app.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/gui/rocket_viewer_app.py) to load sessions, configure serial only, and render typed feedback.
- Simplify transport controls to serial port, baud, framing, and timeout.
- Keep playback controls, but drive them through the shared session scheduler.
- Remove assumptions about merged sensor tables and suffix-matched filenames.

Exit criteria: the GUI no longer contains protocol logic or replay pairing logic.

**Phase 9: Compatibility and Migration**
Keep offline workflows working while removing runtime legacy paths.

- Add a compatibility utility that can derive a merged table from a session only for offline analysis if needed.
- Do not let runtime HIL depend on `virtual_sensors_full_rate_*.csv`.
- Deprecate legacy log discovery in [sim/estimation/adapters/rocketpy_replay.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/sim/estimation/adapters/rocketpy_replay.py) in a controlled way.

Exit criteria: HIL runtime uses sessions only; analysis tools can migrate gradually.

**Phase 10: Test and Documentation Sweep**
Lock the refactor down after behavior stabilizes.

- Remove [tests/estimation/test_sitl_udp_bridge.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/tests/estimation/test_sitl_udp_bridge.py).
- Expand [tests/estimation/test_mavlink_sitl_service.py](/home/luis/Projects/Python/RL-RocketPy-Notebook/tests/estimation/test_mavlink_sitl_service.py) for serial-only scheduling and feedback.
- Add tests for session loading, manifest validation, truth resampling, multi-rate scheduling, seek/reset, and codec behavior.
- Update [README.md](/home/luis/Projects/Python/RL-RocketPy-Notebook/README.md) and notebook guidance to reflect session-based serial HIL.

Exit criteria: tests cover the new architecture and docs match the code.

**Recommended Execution Order**
1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6
7. Phase 7
8. Phase 8
9. Phase 9
10. Phase 10

**Risk Control**
Do this behind a temporary compatibility boundary until Phase 8 is complete. The highest-risk transitions are exporter format changes, scheduler replacement, and UDP removal, so those should land with dedicated tests before the GUI is fully rewired.

If you want, the next step is for me to convert this into a file-by-file work breakdown with exact module additions, deletions, and replacement points.