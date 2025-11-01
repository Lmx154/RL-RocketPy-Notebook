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

from rocket_config import (
    create_rocket, 
    create_environment,
    LAUNCH_LATITUDE,
    LAUNCH_LONGITUDE
)
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

# ==================== CREATE ENVIRONMENT FOR SIMULATION ====================
print("Creating environment for flight simulation...")
print("Using GFS forecast data for realistic wind conditions...")
environment = create_environment(use_forecast=True)
print(f"✓ Environment configured for {LAUNCH_LATITUDE:.6f}°N, {LAUNCH_LONGITUDE:.6f}°W")
print("  (includes wind from weather forecast)")
print()

print("Launching Rocket 3D Viewer...")
print("  - Static Mode: View rocket geometry")
print("  - 3D Simulation Mode: Compute and visualize flight")
print()

# ==================== LAUNCH GUI ====================
# Launch the viewer with the rocket and environment
sys.exit(launch_gui(rocket, environment))
