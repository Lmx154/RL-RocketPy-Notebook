"""
Rocket 3D Viewer Launcher

Launch the Rocket 3D Viewer GUI application with the V-10 rocket.

This is a standalone launcher that creates the rocket and opens the GUI.
"""

import sys
import datetime
from rocketpy import Environment, SolidMotor, Rocket, TrapezoidalFins, NoseCone, Tail
import numpy as np
from sim import launch_gui

print("=" * 60)
print("Rocket 3D Viewer - V-10 Rocket")
print("=" * 60)
print()

# ==================== CREATE ENVIRONMENT ====================
print("Creating environment...")
env = Environment()
env.set_location(latitude=33.49862509998744, longitude=-99.3376124767802)
env.set_elevation(417.0)
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
env.set_atmospheric_model(type='Forecast', file='GFS')
print("✓ Environment created")

# ==================== CREATE MOTOR ====================
print("Creating AeroTech M2500T motor...")
m2500t = SolidMotor(
    thrust_source='data/AeroTech_M2500T.eng',
    dry_mass=3.353,
    center_of_dry_mass_position=0,
    dry_inertia=[0, 0, 0],
    grains_center_of_mass_position=0,
    grain_number=1,
    grain_density=1137.5,
    grain_outer_radius=0.049,
    grain_initial_inner_radius=0.0245,
    grain_initial_height=0.732,
    grain_separation=0,
    nozzle_radius=0.036750000000000005,
    nozzle_position=0.351,
    throat_radius=0.0245,
    reshape_thrust_curve=False,
    interpolation_method='linear',
    coordinate_system_orientation='combustion_chamber_to_nozzle',
)
print("✓ Motor created")

# ==================== CREATE ROCKET COMPONENTS ====================
print("Creating rocket components...")

nosecone = NoseCone(
    length=0.381,
    kind='Von Karman',
    base_radius=0.0777875,
    rocket_radius=0.0777875,
    name='Nose Cone',
)

fins = TrapezoidalFins(
    n=4,
    root_chord=0.3048,
    tip_chord=0.01905,
    span=0.1524,
    sweep_length=0.254,
    rocket_radius=0.0777875,
    name="CF Trapezoidal",
)

tail = Tail(
    top_radius=0.0777875,
    bottom_radius=0.0635,
    length=0.0508,
    rocket_radius=0.0777875,
    name='Boat Tail',
)

print("✓ Components created")

# ==================== CREATE ROCKET ASSEMBLY ====================
print("Assembling rocket...")

cd_power_off = np.array([
    [0.00, 0.45], [0.50, 0.50], [0.90, 0.55],
    [1.00, 0.75], [1.20, 0.65], [2.00, 0.50], [3.00, 0.45],
])
cd_power_on = np.array([
    [0.00, 0.40], [0.50, 0.45], [0.90, 0.50],
    [1.00, 0.70], [1.20, 0.60], [2.00, 0.45], [3.00, 0.40],
])

rocket = Rocket(
    radius=0.0777875,
    mass=17.732,
    inertia=[0.115, 0.115, 21.424],
    power_off_drag=cd_power_off,
    power_on_drag=cd_power_on,
    center_of_mass_without_motor=0.0,  # Origin at CoM
    coordinate_system_orientation='tail_to_nose',  # ✅ CORRECT: Z+ points UP
)

# Component positions for 'tail_to_nose' coordinate system
# Z+ points from tail to nose (UP in vertical flight)
# Nose at origin (Z=0), components toward tail have NEGATIVE Z
rocket.add_surfaces(
    surfaces=[nosecone, fins, tail],
    positions=[0.0, -2.592, -2.8964]  # Negative = toward tail
)
rocket.add_motor(m2500t, position=-2.562)  # Negative = toward tail

print("✓ Rocket assembled")
print()
print("Launching Rocket 3D Viewer...")
print()

# ==================== LAUNCH GUI ====================
# Launch the viewer with the rocket
sys.exit(launch_gui(rocket))
