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

## SITL WebSocket Replay Bridge

Use the SITL bridge when your firmware in software-in-the-loop needs synchronized replay time and virtual sensors from the same CSV files used by the estimator.

Start the server (defaults to latest `logs/virtual_sensors_full_rate_*.csv`):

```bash
uv run python -m sim.sitl.websocket_bridge
```

Optional explicit telemetry file:

```bash
uv run python -m sim.sitl.websocket_bridge --telemetry logs/virtual_sensors_full_rate_260313_190556.csv
```

WebSocket endpoints:
- `ws://127.0.0.1:8765/clock`
- `ws://127.0.0.1:8765/sensors`

`/clock` command messages (JSON):
- `{"op": "status"}`
- `{"op": "sync", "time_s": 2.54}`
- `{"op": "step", "count": 1}`
- `{"op": "play", "rate": 1.0}`
- `{"op": "pause"}`
- `{"op": "reset"}`
- `{"op": "seek_index", "index": 150}`

`/clock` emits clock state updates and `/sensors` emits the replayed CSV row for the current replay time. `NaN` values in CSV rows are normalized to `null` for robust JSON decoding in firmware SITL clients.


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
