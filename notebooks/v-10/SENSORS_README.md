# V-10 Virtual Sensor Suite

## Overview

The V-10 rocket simulation now includes a comprehensive virtual sensor suite that models realistic flight instrumentation. All sensors are configured in `rocket_config.py` and can be easily added to the rocket in both the flight simulation notebook and 3D viewer.

## Sensor Components

### 1. Accelerometer
- **Purpose**: Measures linear acceleration in 3 axes
- **Measurement Range**: ±16g (±156.96 m/s²)
- **Sampling Rate**: 100 Hz
- **Special Features**: Includes gravity in measurements for realistic attitude estimation
- **Use Cases**: 
  - Velocity and position estimation (through integration)
  - G-force monitoring
  - Launch detection
  - Apogee detection

### 2. Gyroscope
- **Purpose**: Measures angular velocity (rotation rate) in 3 axes
- **Measurement Range**: ±500 deg/s (±8.727 rad/s)
- **Sampling Rate**: 100 Hz
- **Use Cases**:
  - Attitude determination
  - Rotation tracking
  - Stabilization control
  - Air brake orientation

### 3. Barometer
- **Purpose**: Measures atmospheric pressure for altitude estimation
- **Measurement Range**: 0-120,000 Pa (sea level to ~10km altitude)
- **Sampling Rate**: 50 Hz
- **Use Cases**:
  - Altitude estimation
  - Apogee detection
  - Parachute deployment triggering
  - Backup altitude data (independent of GNSS)

### 4. GNSS (GPS) Receiver
- **Purpose**: Measures position and altitude
- **Sampling Rate**: 10 Hz
- **Horizontal Accuracy**: ±3 meters
- **Vertical Accuracy**: ±5 meters
- **Use Cases**:
  - Landing site prediction
  - Recovery operations
  - Flight path reconstruction
  - Absolute position tracking

## Sensor Configuration

### Coordinate System Alignment

All sensors are **perfectly aligned** with the rocket's body coordinate system:

- **Orientation**: `(0, 0, 0)` degrees (roll, pitch, yaw)
- **Coordinate System**: `tail_to_nose` (Z+ points from tail toward nose)
- **Axes**:
  - X-axis: Perpendicular to rocket longitudinal axis
  - Y-axis: Perpendicular to rocket longitudinal axis
  - Z-axis: Along rocket longitudinal axis (tail to nose)

This alignment ensures that:
- All sensors measure in the same reference frame
- Integration between sensors is straightforward
- Sensor fusion algorithms have consistent inputs
- Control algorithms receive aligned measurements

### Mounting Position

All sensors are co-located at:
- **Position**: -1.5 m from nose (in tail_to_nose coordinate system)
- **Physical Location**: Near the center of the rocket body
- **Rationale**: Minimizes rotational effects on accelerometer readings

## Noise Modeling

Each sensor includes realistic noise characteristics to simulate real MEMS sensors:

### Noise Parameters

1. **Noise Density**: High-frequency random noise
   - Simulates electrical noise and quantization effects
   - Typical values: 0.0001-0.0003 for gyros/accelerometers

2. **Random Walk**: Low-frequency drift over time
   - Simulates sensor bias drift
   - Accumulates over time (integration of white noise)

3. **Constant Bias**: Fixed offset in measurements
   - Can be calibrated out in real systems
   - Currently set to 0.0 (assumes perfect calibration)

4. **Noise Variance**: Scaling factor for noise magnitude
   - Set to 1.0 for standard behavior

## Usage

### Quick Start (Using Factory Functions)

The easiest way to add sensors is using the factory function in `create_rocket()`:

```python
# Import configuration
import rocket_config

# Create rocket with sensors (default behavior)
rocket = rocket_config.create_rocket(include_sensors=True)

# Rocket now has all 4 sensors attached!
print(f"Sensors: {len(rocket.sensors)}")  # Output: Sensors: 4
```

### Manual Sensor Creation (Educational)

For learning or custom configurations, create sensors individually:

```python
from rocketpy.sensors import Accelerometer, Gyroscope, Barometer, GnssReceiver
import rocket_config

# Create individual sensors
accelerometer = rocket_config.create_accelerometer()
gyroscope = rocket_config.create_gyroscope()
barometer = rocket_config.create_barometer()
gnss = rocket_config.create_gnss()

# Add to rocket
rocket.add_sensor(accelerometer, rocket_config.SENSOR_POSITION)
rocket.add_sensor(gyroscope, rocket_config.SENSOR_POSITION)
rocket.add_sensor(barometer, rocket_config.SENSOR_POSITION)
rocket.add_sensor(gnss, rocket_config.SENSOR_POSITION)
```

### Accessing Sensor Data

After running a flight simulation:

```python
# Run simulation
flight = Flight(
    rocket=rocket,
    environment=env,
    rail_length=5.1816,
    inclination=90,
    heading=90,
)

# Access sensor data (check RocketPy documentation for exact attribute names)
# Sensor data is stored as time-series arrays with noise included
```

## Configuration Parameters

All sensor parameters are defined in `rocket_config.py`:

### Position & Orientation
```python
SENSOR_POSITION = -1.5  # m (from nose, in tail_to_nose coordinates)
SENSOR_ORIENTATION = (0, 0, 0)  # degrees (aligned with rocket)
```

### IMU Settings
```python
IMU_SAMPLING_RATE = 100  # Hz
ACCEL_MEASUREMENT_RANGE = 160  # m/s² (~16g)
GYRO_MEASUREMENT_RANGE = 8.727  # rad/s (~500 deg/s)
```

### Barometer Settings
```python
BARO_SAMPLING_RATE = 50  # Hz
BARO_MEASUREMENT_RANGE = 120000  # Pa
```

### GNSS Settings
```python
GNSS_SAMPLING_RATE = 10  # Hz
GNSS_POSITION_ACCURACY = 3.0  # meters (horizontal)
GNSS_ALTITUDE_ACCURACY = 5.0  # meters (vertical)
```

## Applications

### Hardware-in-the-Loop (HIL) Testing

The virtual sensors are ideal for HIL testing:

1. **Run simulation** with sensors enabled
2. **Extract sensor data** from flight object
3. **Feed data to flight computer** algorithms
4. **Test control systems** (air brakes, stability, parachute deployment)
5. **Validate algorithms** before flight

### Sensor Fusion

Test sensor fusion algorithms:
- Combine accelerometer + gyroscope for attitude estimation (complementary filter)
- Fuse barometer + GNSS for altitude estimation (Kalman filter)
- Integrate accelerometer for velocity/position (with drift correction from GNSS)

### Air Brake Control

Use sensor data for air brake algorithm development:
- **Barometer**: Primary altitude feedback for control loop
- **Accelerometer**: Detect apogee and deceleration events
- **Gyroscope**: Ensure rocket stability during air brake deployment
- **GNSS**: Validate altitude estimates

## Technical Notes

### Experimental Feature Warning

RocketPy sensors are experimental and may change in future versions. The current implementation provides:
- Realistic noise modeling
- Configurable sampling rates
- Multiple sensor types
- Integration with flight simulation

### Sensor Alignment Benefits

Having all sensors aligned with the rocket body:
- **Simplifies integration**: No rotation transformations needed
- **Reduces computational load**: Direct use of measurements
- **Mirrors real hardware**: Most flight computers mount sensors aligned with body
- **Easier debugging**: Measurements directly correspond to rocket motion

### Future Enhancements

Potential improvements:
- Add magnetometer for absolute heading reference
- Include temperature effects on sensors
- Model sensor saturation during high-g events
- Add cross-axis sensitivity errors
- Implement more sophisticated bias models

## Files Modified

1. **`notebooks/v-10/rocket_config.py`**
   - Added sensor imports
   - Added sensor configuration constants
   - Added sensor factory functions (`create_accelerometer()`, `create_gyroscope()`, `create_barometer()`, `create_gnss()`)
   - Added `create_sensors()` convenience function
   - Updated `create_rocket()` to accept `include_sensors` parameter
   - Added sensor attachment logic in `create_rocket()`

2. **`notebooks/v-10/v-10_flight_sim.ipynb`**
   - Added "Virtual Sensors" markdown cell explaining sensor suite
   - Added code cell to create and attach sensors
   - Added "Accessing Sensor Data" markdown cell with usage examples

## References

- [RocketPy Sensor Documentation](https://docs.rocketpy.org/) (check for sensor module updates)
- MEMS IMU datasheets for realistic noise parameters
- GPS/GNSS receiver specifications for accuracy values
- Barometric altimeter specifications for range and resolution

## Contact

For questions about the sensor implementation or configuration:
- **Flight Dynamics Team**: Sensor configuration and noise parameters
- **Avionics Team**: Sensor integration and data acquisition
- **The Rocket Launchers**: UTRGV student organization

---

*Last Updated: 2025-11-01*
*V-10 Mission: Lonestar Cup 2026 & IREC 2026*
