# SITL Session Contract

Phase 1 freezes one replay session as one directory containing one `manifest.json`
plus stream CSVs with canonical filenames, columns, and rates.

## Directory Layout

- `manifest.json`
- `truth.csv`
- `imu.csv`
- `baro.csv`
- `gps.csv`
- `mag.csv` (optional)
- `estimator_feedback.csv` (optional, future)
- `device_events.csv` (optional, future)

All file references in `manifest.json` must be direct filenames in the same
directory as the manifest.

## Canonical Streams

| Stream key | Filename | Required | Rate | Canonical columns |
| --- | --- | --- | --- | --- |
| `truth` | `truth.csv` | yes | `500 Hz` | `time_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,e0,e1,e2,e3,w1_radps,w2_radps,w3_radps` |
| `imu` | `imu.csv` | yes | `300 Hz` | `time_s,accelerometer_x,accelerometer_y,accelerometer_z,gyroscope_x,gyroscope_y,gyroscope_z` |
| `baro` | `baro.csv` | yes | `100 Hz` | `time_s,barometer_v1` |
| `gps` | `gps.csv` | yes | `10 Hz` | `time_s,gnss_x,gnss_y,gnss_z` |
| `mag` | `mag.csv` | no | `50 Hz` | `time_s,magnetometer_x,magnetometer_y,magnetometer_z` |
| `estimator_feedback` | `estimator_feedback.csv` | no | event-driven | `time_s,feedback_type,payload_json` |
| `device_events` | `device_events.csv` | no | event-driven | `time_s,event_type,event_name,payload_json` |

## Required Manifest Fields

- `schema_version`: fixed to `"1.0"`
- `session_id`: stable identifier for the session directory
- `vehicle_name`: rocket or vehicle name
- `generated_at_utc`: UTC timestamp string for export creation
- `reference_latitude_deg`
- `reference_longitude_deg`
- `reference_altitude_m`
- `sea_level_pressure_pa`
- `streams`: mapping of stream keys to canonical filenames
- `rates_hz`: mapping of fixed-rate stream keys to canonical rates

Optional manifest fields:

- `notes`

## Manifest Example

```json
{
  "schema_version": "1.0",
  "session_id": "260313_190556",
  "vehicle_name": "Itzamna",
  "generated_at_utc": "2026-03-27T18:45:00Z",
  "reference_latitude_deg": 33.4986251,
  "reference_longitude_deg": -99.3376125,
  "reference_altitude_m": 417.0,
  "sea_level_pressure_pa": 101325.0,
  "streams": {
    "truth": "truth.csv",
    "imu": "imu.csv",
    "baro": "baro.csv",
    "gps": "gps.csv"
  },
  "rates_hz": {
    "truth": 500,
    "imu": 300,
    "baro": 100,
    "gps": 10
  }
}
```
