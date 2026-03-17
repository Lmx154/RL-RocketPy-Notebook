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

You can replay the merged sensor CSV exports directly:

```python
from sim.estimation import run_telemetry_replay

result = run_telemetry_replay("logs/virtual_sensors_full_rate_260310_113255.csv")
estimates = result.estimates
print(estimates[["time_s", "est_z_m", "est_vz_mps"]].tail())
```

The replay adapter is built around the current log format:
- IMU prediction from `accelerometer_*` and `gyroscope_*`
- Barometric altitude updates from `barometer_v1`
- GNSS position updates from `gnss_x`, `gnss_y`, and `gnss_z`

For the current RocketPy exports, the GNSS columns are interpreted as latitude, longitude, and altitude, then converted into a local ENU frame relative to the first valid GNSS fix.

## SITL UDP Replay Bridge

The SITL bridge is layered so the CSV replay core is transport-agnostic, UDP is the command/output transport, and protocol adapters can be swapped independently.

Current layers:
- Replay core: reads `virtual_sensors_full_rate_*.csv` and maintains replay time/index state.
- UDP transport: listens for replay control commands and emits datagrams to firmware/SITL targets.
- Protocol adapters:
	- JSON UDP adapter for debugging or custom glue code.
	- MAVLink adapter based on the standard `common.xml` dialect via `pymavlink.dialects.v20.common`.

Start the UDP bridge with MAVLink output enabled by default:

```bash
uv run python -m sim.sitl.udp_bridge
```

Optional explicit telemetry file and JSON debug output:

```bash
uv run python -m sim.sitl.udp_bridge \
	--telemetry logs/virtual_sensors_full_rate_260313_190556.csv \
	--target-host 127.0.0.1 \
	--mavlink-target-port 14550 \
	--json-target-port 14610
```

UDP control commands go to `udp://0.0.0.0:14600` by default as JSON datagrams:
- `{"op": "status"}`
- `{"op": "sync", "time_s": 2.54}`
- `{"op": "step", "count": 1}`
- `{"op": "play", "rate": 1.0}`
- `{"op": "pause"}`
- `{"op": "reset"}`
- `{"op": "seek_index", "index": 150}`

The MAVLink adapter emits standard `common.xml` messages over UDP:
- `SYSTEM_TIME` for replay clock sync
- `HIL_SENSOR` for IMU and barometer values
- `HIL_GPS` for GNSS fixes derived from the CSV

CSV `NaN` values are normalized to `null` in the JSON adapter. MAVLink packets omit GNSS output when a row does not contain a valid fix.


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
