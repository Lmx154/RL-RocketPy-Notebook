"""
Rocket 3D Viewer Launcher

Launch the Rocket 3D Viewer GUI application with the V-10 rocket.

This launcher uses the shared rocket configuration from the notebook
to ensure the 3D visualization matches the simulated rocket exactly.
"""

import sys
import os

# Add notebooks/v-10 to path to import rocket_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks', 'v-10'))

from rocket_config import create_rocket
from sim import launch_gui

print("=" * 60)
print("Rocket 3D Viewer - V-10 Rocket")
print("=" * 60)
print()

# ==================== CREATE ROCKET FROM SHARED CONFIG ====================
print("Creating V-10 rocket from notebook configuration...")
print("This ensures the 3D viewer shows the exact rocket from the simulation.")
print()

# Create rocket using shared configuration (without parachutes for viewer)
rocket = create_rocket(include_parachutes=False, drag_data_path='data')

print("✓ Rocket assembled from shared configuration")
print(f"  - Dry mass: {rocket.mass:.3f} kg")
print(f"  - Radius: {rocket.radius:.4f} m")
print(f"  - Surfaces: {len(rocket.aerodynamic_surfaces)}")
print()
print("Launching Rocket 3D Viewer...")
print()

# ==================== LAUNCH GUI ====================
# Launch the viewer with the rocket
sys.exit(launch_gui(rocket))
