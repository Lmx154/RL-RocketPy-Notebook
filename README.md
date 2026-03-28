# RL's RocketPy sim

## What is this?
This is an extensive notebook that houses all of the simulations of our rockets.

## Features

- 🚀 **RocketPy Integration**: Full simulation capabilities for rocket design and flight
- 📊 **Jupyter Notebooks**: Interactive development and visualization
- 🎨 **3D Rocket Renderer**: Dedicated GUI application for visualizing rockets in 3D
- 📦 **OpenRocket Import**: Convert .ork files to RocketPy format

## How do we use this?

Start by running

```bash
uv sync
```

## 3D Rocket Viewer

We now have a dedicated GUI application for viewing your rockets in 3D!

### Quick Start

```bash
uv run python launch_viewer.py
```

The viewer is kinematics-driven in simulation mode and no longer loads environment/forecast data.

In simulation mode, the 3D viewer now also shows a bottom-right `HIL Events` overlay with
three panels:
- `Payload`
- `Avionics`
- `Recovery`

The current example seed is:
- `Payload`: `5000FT altitude servo test`

If the viewer crashes on Linux with a Qt/Wayland window error (`BadWindow`),
force the backend explicitly:

```bash
ROCKET_VIEWER_QPA_PLATFORM=xcb uv run python launch_viewer.py
```

This launches a user-friendly application where you can:
- Select individual components to view (motor, nose cone, fins, etc.)
- Rotate, pan, and zoom with your mouse
- Export models to STL for 3D printing
- View mesh statistics

## State Estimation

The repository now includes a reusable PX4-style state estimation framework under [sim/estimation](sim/estimation).

It uses a strapdown inertial navigation system for the nominal state and a 15-state error-state Kalman filter for small-angle attitude, velocity, position, gyro bias, and accelerometer bias correction.

You can replay a manifest-based session directly for offline estimation:

```python
from sim.estimation.adapters import run_rocketpy_replay

result = run_rocketpy_replay("logs/session_260313_190556")
estimates = result.estimates
print(estimates[["time_s", "est_z_m", "est_vz_mps"]].tail())
```

If you need the old merged-table shape for analysis tooling, derive it explicitly from a session:

```python
from sim.sitl.session import load_replay_session, merge_replay_session_sensors

session = load_replay_session("logs/session_260313_190556")
telemetry = merge_replay_session_sensors(session)
```

Legacy `virtual_sensors_full_rate_*.csv` discovery still works for now, but it is deprecated.

The replay adapter consumes the same sensor fields:
- IMU prediction from `accelerometer_*` and `gyroscope_*`
- Barometric altitude updates from `barometer_v1`
- GNSS position updates from `gnss_x`, `gnss_y`, and `gnss_z`

For the current RocketPy exports, the GNSS columns are interpreted as latitude, longitude, and altitude, then converted into a local ENU frame relative to the first valid GNSS fix.

## SITL Serial HIL

The runtime SITL path is now serial-only.

Current layers:
- Session replay: loads one manifest-based replay session and advances truth and sensors by timestamp.
- MAVLink codec: emits standard `common.xml` `SYSTEM_TIME`, `HIL_SENSOR`, and `HIL_GPS` messages.
- Serial transport: writes outbound HIL packets and decodes inbound MAVLink messages from the connected device.
- Typed feedback: maps inbound commands, logging, and estimator values into reusable feedback objects.

In the viewer, load a replay session, configure the USB serial port in the MAVLink panel, and enable `MAVLink SITL`.

### Wiring Custom MAVLink HIL Events Into The Viewer

The viewer now consumes typed feedback objects from the serial SITL layer instead of
decoding raw MAVLink messages directly.

If you want a custom firmware command to appear in the bottom-right HIL events overlay:

1. Pick your custom MAVLink command ID in firmware.
2. Add or edit the mapping in `DEFAULT_COMMAND_EVENT_DEFINITIONS` in `sim/sitl/estimator_feedback.py`.
3. When the firmware sends `COMMAND_LONG` or `COMMAND_INT` with that command ID, the serial
   service will decode it into a typed `DeviceStateEvent`.
4. The viewer will render the resulting overlay event in the configured category panel.

Current example mapping:
- `COMMAND_LONG.command == 31000` -> `Payload` -> `5000FT altitude servo test`

Relevant integration points:
- `sim/sitl/estimator_feedback.py`
  - `decode_mavlink_feedback(...)` parses inbound MAVLink into typed feedback domains.
  - `DEFAULT_COMMAND_EVENT_DEFINITIONS` holds the current command-to-event mapping.
- `sim/sitl/mavlink_sitl_service.py`
  - `_SerialTransport._handle_incoming_chunk(...)` decodes inbound MAVLink bytes.
  - `SitlMavlinkService.drain_pending_feedback()` exposes typed feedback to the GUI.
- `sim/gui/rocket_viewer_app.py`
  - `_handle_feedback(...)` renders typed feedback into the HIL overlay.

Important detail: inbound custom-command handling is implemented on the serial/HIL path, which is
where firmware should send MAVLink commands back to the viewer.


## If you're going to work on the Itzamna notebook

To launch the notebook, type

```bash
jupyter lab
```
After opening Jupyter Lab in your web browser, navigate to notebooks on the left side bar of the user interface.

Open notebooks folder and select the Itzamna folder, then open the Itzamna_flight_sim file.

To clear things out and start anew, open the Kernel tab, select the Restart Kernel and Clear Outputs of All Cells... option

Once you have made the changes you desire, and you wish to run the simulation, open the Run tab and select the Run All Cells option


I **HIGHLY** suggest you refrain from importing from OpenRocket, and instead work with this notebook to create your iterations of the rocket. 

But if you must..... you are going to be converting an Open Rocket sim/rocket to RocketPy, and you must download the .jar Open Rocket installer.  
Note: You **MUST** get the .jar version that aligns with the Open Rocket that you made the .ork file in.  
<https://openrocket.info/downloads.html>

If you're going to convert an ork file, use

```bash
cd serializer
```

and open `INSTRUCTIONS.md` to continue with the conversion.

### **NAH** I won't be adding the .jar file in here bc it's too big, so **DEAL WITH IT** 
