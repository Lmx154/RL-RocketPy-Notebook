"""
V-10 Rocket Configuration Module

This module contains all configuration parameters for the V-10 rocket.
It serves as a single source of truth for both the flight simulation notebook
and the 3D viewer application, ensuring consistency between visualization
and simulation.

Usage:
    from rocket_config import create_rocket, create_environment

Author: The Rocket Launchers - UTRGV
Last Updated: 2025-11-01
"""

import datetime
import numpy as np
import csv
import os
from rocketpy import Environment, SolidMotor, Rocket, TrapezoidalFins, NoseCone, Tail, Parachute
from rocketpy.sensors import Accelerometer, Gyroscope, Barometer, GnssReceiver


# ==================== LAUNCH SITE CONFIGURATIONS ====================
# V-10 Mission Launch Sites

# Launch Site 1: Rocket Ranch - Seymour, Texas (Primary Testing Location)
ROCKET_RANCH_LATITUDE = 33.49862509998744   # degrees North
ROCKET_RANCH_LONGITUDE = -99.3376124767802  # degrees West
ROCKET_RANCH_ELEVATION = 417.0  # meters above sea level

# Launch Site 2: Spaceport America - Las Cruces, New Mexico (IREC Competition)
SPACEPORT_AMERICA_LATITUDE = 32.990254   # degrees North
SPACEPORT_AMERICA_LONGITUDE = -106.974998  # degrees West
SPACEPORT_AMERICA_ELEVATION = 1400.0  # meters above sea level

# Active launch site (change this to switch between sites)
LAUNCH_LATITUDE = ROCKET_RANCH_LATITUDE
LAUNCH_LONGITUDE = ROCKET_RANCH_LONGITUDE
LAUNCH_ELEVATION = ROCKET_RANCH_ELEVATION


# ==================== MOTOR PARAMETERS ====================
MOTOR_DRY_MASS = 3.353  # kg
MOTOR_GRAIN_DENSITY = 1137.5  # kg/m³
MOTOR_GRAIN_OUTER_RADIUS = 0.049  # m
MOTOR_GRAIN_INNER_RADIUS = 0.0245  # m
MOTOR_GRAIN_HEIGHT = 0.732  # m
MOTOR_NOZZLE_RADIUS = 0.036750000000000005  # m
MOTOR_THROAT_RADIUS = 0.0245  # m
MOTOR_NOZZLE_POSITION = 0.351  # m


# ==================== ROCKET DIMENSIONS ====================
ROCKET_BODY_RADIUS = 0.0777875  # m

# Nose Cone
NOSECONE_LENGTH = 0.381  # m
NOSECONE_BASE_RADIUS = 0.0777875  # m

# Fins
FIN_COUNT = 4
FIN_ROOT_CHORD = 0.3048  # m
FIN_TIP_CHORD = 0.01905  # m
FIN_SPAN = 0.1524  # m
FIN_SWEEP_LENGTH = 0.254  # m

# Boat Tail
BOATTAIL_TOP_RADIUS = 0.0777875  # m
BOATTAIL_BOTTOM_RADIUS = 0.0635  # m
BOATTAIL_LENGTH = 0.0508  # m


# ==================== ROCKET MASS PROPERTIES ====================
ROCKET_DRY_MASS = 17.732  # kg
ROCKET_INERTIA_I = 0.115  # kg⋅m²
ROCKET_INERTIA_Z = 21.424  # kg⋅m²
ROCKET_COM_WITHOUT_MOTOR = 0.0  # m


# ==================== COMPONENT POSITIONS ====================
# All positions in 'tail_to_nose' coordinate system
# Z+ points UP (from tail toward nose)
# Nose at Z=0, components toward tail have NEGATIVE positions

NOSECONE_POSITION = 0.0  # m
FINS_POSITION = -2.592  # m
BOATTAIL_POSITION = -2.8964  # m
MOTOR_POSITION = -2.562  # m


# ==================== RAIL BUTTON POSITIONS ====================
UPPER_BUTTON_POSITION = -1.7778  # m
LOWER_BUTTON_POSITION = -1.3906  # m
BUTTON_ANGULAR_POSITION = 60.0  # degrees


# ==================== PARACHUTE PARAMETERS ====================
# Main Parachute
MAIN_CHUTE_CD = 2.2
MAIN_CHUTE_DIAMETER = 2.7432  # m
MAIN_DEPLOY_ALTITUDE = 396.24  # m

# Drogue Parachute
DROGUE_CHUTE_CD = 2.2
DROGUE_CHUTE_DIAMETER = 0.6096  # m


# ==================== FLIGHT PARAMETERS ====================
RAIL_LENGTH = 5.1816  # m
LAUNCH_INCLINATION = 90.0  # degrees
LAUNCH_HEADING = 90.0  # degrees
MAX_SIMULATION_TIME = 600  # seconds


# ==================== SENSOR PARAMETERS ====================
# Position of sensor suite (from nose tip, in tail_to_nose coordinate system)
SENSOR_POSITION = -1.5  # m (negative = toward tail, mounted near center of rocket)

# Sensor orientation: (0, 0, 0) means aligned with rocket body axes
# X-axis: perpendicular to rocket longitudinal axis
# Y-axis: perpendicular to rocket longitudinal axis  
# Z-axis: along rocket longitudinal axis (tail to nose)
SENSOR_ORIENTATION = (0, 0, 0)  # degrees (roll, pitch, yaw) - aligned with rocket

# IMU (Accelerometer + Gyroscope) Configuration
IMU_SAMPLING_RATE = 100  # Hz - typical for flight computers
ACCEL_MEASUREMENT_RANGE = 160  # m/s² (~16g)
ACCEL_NOISE_DENSITY = 0.0003  # typical for MEMS accelerometers
ACCEL_NOISE_VARIANCE = 1
ACCEL_RANDOM_WALK_DENSITY = 0.00001
ACCEL_RANDOM_WALK_VARIANCE = 1
ACCEL_CONSTANT_BIAS = 0.0
ACCEL_CONSIDER_GRAVITY = True  # Include gravity in measurements

GYRO_MEASUREMENT_RANGE = 8.727  # rad/s (~500 deg/s)
GYRO_NOISE_DENSITY = 0.0001  # typical for MEMS gyroscopes
GYRO_NOISE_VARIANCE = 1
GYRO_RANDOM_WALK_DENSITY = 0.000001
GYRO_RANDOM_WALK_VARIANCE = 1
GYRO_CONSTANT_BIAS = 0.0

# Barometer Configuration
BARO_SAMPLING_RATE = 50  # Hz
BARO_MEASUREMENT_RANGE = 120000  # Pa (covers sea level to ~10km altitude)
BARO_NOISE_DENSITY = 0.1  # Pa
BARO_NOISE_VARIANCE = 1
BARO_RANDOM_WALK_DENSITY = 0.01
BARO_RANDOM_WALK_VARIANCE = 1
BARO_CONSTANT_BIAS = 0.0

# GNSS Configuration
GNSS_SAMPLING_RATE = 10  # Hz - typical for GPS modules
GNSS_POSITION_ACCURACY = 3.0  # meters - horizontal accuracy
GNSS_ALTITUDE_ACCURACY = 5.0  # meters - vertical accuracy


def load_drag_coefficients(base_path='../../data'):
    """
    Load drag coefficients from CSV file.
    
    Args:
        base_path: Path to the data directory relative to this file
        
    Returns:
        tuple: (cd_power_off, cd_power_on) as numpy arrays
    """
    cd_clean_path = os.path.join(base_path, 'drag_curve_clean.csv')
    
    # Try relative path first, then absolute path for launcher
    if not os.path.exists(cd_clean_path):
        # Try from project root for launch_viewer.py
        cd_clean_path = os.path.join('data', 'drag_curve_clean.csv')
    
    if not os.path.exists(cd_clean_path):
        # Fallback to default values if file not found
        print(f"Warning: Could not find drag_curve_clean.csv, using default values")
        M_arr = np.array([0.0, 0.5, 0.9, 1.0, 1.2, 2.0, 3.0])
        Cd_arr = np.array([0.45, 0.50, 0.55, 0.75, 0.65, 0.50, 0.45])
    else:
        M_list, Cd_list = [], []
        with open(cd_clean_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    M = float(row.get('Mach') or row.get('mach') or row[list(row.keys())[0]])
                    Cd = float(row.get('Cd_power_off') or row.get('cd') or row[list(row.keys())[1]])
                except Exception:
                    continue
                M_list.append(M)
                Cd_list.append(Cd)
        
        M_arr = np.array(M_list, dtype=float)
        Cd_arr = np.array(Cd_list, dtype=float)
        order = np.argsort(M_arr)
        M_arr, Cd_arr = M_arr[order], Cd_arr[order]
    
    cd_power_off = np.column_stack([M_arr, Cd_arr])
    cd_power_on = np.column_stack([M_arr, np.maximum(0.0, Cd_arr - 0.03)])
    
    return cd_power_off, cd_power_on


def create_environment(use_forecast=True):
    """
    Create and configure the Environment object.
    
    Args:
        use_forecast: Whether to use GFS forecast data (default: True)
        
    Returns:
        Environment: Configured environment object
    """
    env = Environment()
    env.set_location(latitude=LAUNCH_LATITUDE, longitude=LAUNCH_LONGITUDE)
    env.set_elevation(LAUNCH_ELEVATION)
    
    if use_forecast:
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
        env.set_atmospheric_model(type='Forecast', file='GFS')
    
    return env


def create_motor(thrust_source='../../data/AeroTech_M2500T_smoothed.eng'):
    """
    Create the AeroTech M2500T motor.
    
    Args:
        thrust_source: Path to the thrust curve .eng file
        
    Returns:
        SolidMotor: Configured motor object
    """
    # Try relative path first, then absolute path for launcher
    if not os.path.exists(thrust_source):
        thrust_source = 'data/AeroTech_M2500T_smoothed.eng'
    
    if not os.path.exists(thrust_source):
        # Fallback to non-smoothed version
        thrust_source = 'data/AeroTech_M2500T.eng'
    
    motor = SolidMotor(
        thrust_source=thrust_source,
        dry_mass=MOTOR_DRY_MASS,
        center_of_dry_mass_position=0,
        dry_inertia=[0, 0, 0],
        grains_center_of_mass_position=0,
        grain_number=1,
        grain_density=MOTOR_GRAIN_DENSITY,
        grain_outer_radius=MOTOR_GRAIN_OUTER_RADIUS,
        grain_initial_inner_radius=MOTOR_GRAIN_INNER_RADIUS,
        grain_initial_height=MOTOR_GRAIN_HEIGHT,
        grain_separation=0,
        nozzle_radius=MOTOR_NOZZLE_RADIUS,
        nozzle_position=MOTOR_NOZZLE_POSITION,
        throat_radius=MOTOR_THROAT_RADIUS,
        reshape_thrust_curve=False,
        interpolation_method='linear',
        coordinate_system_orientation='combustion_chamber_to_nozzle',
    )
    
    return motor


def create_nosecone():
    """Create the nose cone component."""
    return NoseCone(
        length=NOSECONE_LENGTH,
        kind='Von Karman',
        base_radius=NOSECONE_BASE_RADIUS,
        rocket_radius=ROCKET_BODY_RADIUS,
        name='Nose Cone',
    )


def create_fins():
    """Create the fin set component."""
    return TrapezoidalFins(
        n=FIN_COUNT,
        root_chord=FIN_ROOT_CHORD,
        tip_chord=FIN_TIP_CHORD,
        span=FIN_SPAN,
        sweep_length=FIN_SWEEP_LENGTH,
        rocket_radius=ROCKET_BODY_RADIUS,
        name="CF Trapezoidal (approx from V-10)",
    )


def create_boattail():
    """Create the boat tail component."""
    return Tail(
        top_radius=BOATTAIL_TOP_RADIUS,
        bottom_radius=BOATTAIL_BOTTOM_RADIUS,
        length=BOATTAIL_LENGTH,
        rocket_radius=ROCKET_BODY_RADIUS,
        name='Boat Tail',
    )


def create_parachutes():
    """
    Create the parachute recovery system.
    
    Returns:
        list: List of Parachute objects [main, drogue]
    """
    main_chute_area = 3.14159 * (MAIN_CHUTE_DIAMETER / 2) ** 2
    main_chute_cd_s = MAIN_CHUTE_CD * main_chute_area
    
    drogue_chute_area = 3.14159 * (DROGUE_CHUTE_DIAMETER / 2) ** 2
    drogue_chute_cd_s = DROGUE_CHUTE_CD * drogue_chute_area
    
    main_parachute = Parachute(
        name='Main Parachute + Bag',
        cd_s=main_chute_cd_s,
        trigger=MAIN_DEPLOY_ALTITUDE,
        sampling_rate=100,
    )
    
    drogue_parachute = Parachute(
        name='Drogue',
        cd_s=drogue_chute_cd_s,
        trigger='apogee',
        sampling_rate=100,
    )
    
    return [main_parachute, drogue_parachute]


def create_accelerometer():
    """
    Create an accelerometer sensor.
    
    Returns:
        Accelerometer: Configured accelerometer sensor aligned with rocket axes
    """
    accelerometer = Accelerometer(
        sampling_rate=IMU_SAMPLING_RATE,
        orientation=SENSOR_ORIENTATION,
        measurement_range=ACCEL_MEASUREMENT_RANGE,
        noise_density=ACCEL_NOISE_DENSITY,
        noise_variance=ACCEL_NOISE_VARIANCE,
        random_walk_density=ACCEL_RANDOM_WALK_DENSITY,
        random_walk_variance=ACCEL_RANDOM_WALK_VARIANCE,
        constant_bias=ACCEL_CONSTANT_BIAS,
        consider_gravity=ACCEL_CONSIDER_GRAVITY,
        name='V-10 Accelerometer',
    )
    return accelerometer


def create_gyroscope():
    """
    Create a gyroscope sensor.
    
    Returns:
        Gyroscope: Configured gyroscope sensor aligned with rocket axes
    """
    gyroscope = Gyroscope(
        sampling_rate=IMU_SAMPLING_RATE,
        orientation=SENSOR_ORIENTATION,
        measurement_range=GYRO_MEASUREMENT_RANGE,
        noise_density=GYRO_NOISE_DENSITY,
        noise_variance=GYRO_NOISE_VARIANCE,
        random_walk_density=GYRO_RANDOM_WALK_DENSITY,
        random_walk_variance=GYRO_RANDOM_WALK_VARIANCE,
        constant_bias=GYRO_CONSTANT_BIAS,
        name='V-10 Gyroscope',
    )
    return gyroscope


def create_barometer():
    """
    Create a barometer sensor.
    
    Returns:
        Barometer: Configured barometer sensor
    """
    barometer = Barometer(
        sampling_rate=BARO_SAMPLING_RATE,
        measurement_range=BARO_MEASUREMENT_RANGE,
        noise_density=BARO_NOISE_DENSITY,
        noise_variance=BARO_NOISE_VARIANCE,
        random_walk_density=BARO_RANDOM_WALK_DENSITY,
        random_walk_variance=BARO_RANDOM_WALK_VARIANCE,
        constant_bias=BARO_CONSTANT_BIAS,
        name='V-10 Barometer',
    )
    return barometer


def create_gnss():
    """
    Create a GNSS (GPS) receiver sensor.
    
    Returns:
        GnssReceiver: Configured GNSS receiver sensor
    """
    gnss = GnssReceiver(
        sampling_rate=GNSS_SAMPLING_RATE,
        position_accuracy=GNSS_POSITION_ACCURACY,
        altitude_accuracy=GNSS_ALTITUDE_ACCURACY,
        name='V-10 GNSS',
    )
    return gnss


def create_sensors():
    """
    Create all sensors for the V-10 rocket.
    
    Returns:
        dict: Dictionary containing all sensors with keys:
              'accelerometer', 'gyroscope', 'barometer', 'gnss'
    """
    return {
        'accelerometer': create_accelerometer(),
        'gyroscope': create_gyroscope(),
        'barometer': create_barometer(),
        'gnss': create_gnss(),
    }


def create_rocket(include_parachutes=True, include_sensors=True, drag_data_path='../../data'):
    """
    Create the complete V-10 rocket assembly.
    
    Args:
        include_parachutes: Whether to add parachutes (default: True)
        include_sensors: Whether to add sensors (default: True)
        drag_data_path: Path to drag coefficient data
        
    Returns:
        Rocket: Fully assembled rocket object
    """
    # Load drag coefficients
    cd_power_off, cd_power_on = load_drag_coefficients(drag_data_path)
    
    # Create rocket
    rocket = Rocket(
        radius=ROCKET_BODY_RADIUS,
        mass=ROCKET_DRY_MASS,
        inertia=[ROCKET_INERTIA_I, ROCKET_INERTIA_I, ROCKET_INERTIA_Z],
        power_off_drag=cd_power_off,
        power_on_drag=cd_power_on,
        center_of_mass_without_motor=ROCKET_COM_WITHOUT_MOTOR,
        coordinate_system_orientation='tail_to_nose',
    )
    
    # Create components
    nosecone = create_nosecone()
    fins = create_fins()
    boattail = create_boattail()
    motor = create_motor()
    
    # Add surfaces
    rocket.add_surfaces(
        surfaces=[nosecone, fins, boattail],
        positions=[NOSECONE_POSITION, FINS_POSITION, BOATTAIL_POSITION]
    )
    
    # Add motor
    rocket.add_motor(motor, position=MOTOR_POSITION)
    
    # Add parachutes if requested
    if include_parachutes:
        rocket.parachutes = create_parachutes()
    
    # Add rail buttons
    rocket.set_rail_buttons(
        upper_button_position=UPPER_BUTTON_POSITION,
        lower_button_position=LOWER_BUTTON_POSITION,
        angular_position=BUTTON_ANGULAR_POSITION,
    )
    
    # Add sensors if requested
    if include_sensors:
        sensors = create_sensors()
        for sensor_name, sensor in sensors.items():
            rocket.add_sensor(sensor, SENSOR_POSITION)
    
    return rocket


if __name__ == "__main__":
    # Test configuration by creating rocket
    print("Testing V-10 rocket configuration...")
    rocket = create_rocket()
    print("✓ Rocket created successfully")
    print(f"  - Dry mass: {rocket.mass:.3f} kg")
    print(f"  - Radius: {rocket.radius:.4f} m")
    print(f"  - Surfaces: {len(rocket.aerodynamic_surfaces)}")
    print(f"  - Sensors: {len(rocket.sensors)}")
    print("\nSensor Suite (all sensors aligned with rocket body axes):")
    for sensor_tuple in rocket.sensors:
        sensor_obj = sensor_tuple[0]  # sensor is stored as (sensor, position) tuple
        sensor_pos = sensor_tuple[1]
        print(f"  - {sensor_obj.name} at position {sensor_pos}")
