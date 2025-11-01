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

## 3D Rocket Viewer (NEW!)

We now have a dedicated GUI application for viewing your rockets in 3D!

### Quick Start

```bash
uv run python launch_viewer.py
```

This launches a user-friendly application where you can:
- Select individual components to view (motor, nose cone, fins, etc.)
- Rotate, pan, and zoom with your mouse
- Export models to STL for 3D printing
- View mesh statistics

### Installation

```bash
uv pip install PySide6 pyvistaqt
```

### Features

- ✅ Single unified window (no more popup spam!)
- ✅ Component selection checkboxes
- ✅ Embedded 3D viewer
- ✅ Export to STL/OBJ/PLY/VTK
- ✅ Cross-platform (Windows, macOS, Linux)

### Usage from Code

```python
from sim.gui import launch_viewer
from rocketpy import Rocket

# Create your rocket
rocket = Rocket(...)
# ... configure rocket ...

# Launch the 3D viewer
launch_viewer(rocket=rocket)
```

See [GUI_DOCUMENTATION.md](GUI_DOCUMENTATION.md) for complete details.


## If you're going to work on the V-10 notebook

To launch the notebook, type

```bash
jupyter lab
```
After opening Jupyter Lab in your web browser, navigate to notebooks on the left side bar of the user interface.

Open notebooks folder and select the v-10 folder, then open the v-10_flight_sim file.

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