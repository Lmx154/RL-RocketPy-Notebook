### Rocket Overview
- **Name**: Jose's Rocket
- **Designer**: The Rocket Launchers
- **Comment**: Its Joever
- **Reference Type**: Maximum
- **Stages**: Single stage (Sustainer)

### Structural Components
The rocket consists of a nose cone, multiple body tubes, a coupler, an inner tube for the motor mount, fins, launch lugs, a transition (boat tail), and various internal components. All lengths, radii, and thicknesses are in meters; masses are in kilograms; densities are in kg/m³ unless otherwise specified.

#### Nose Cone
- **Type**: Haack (shape parameter: 0.0)
- **Material**: Fiber Glass in house (density: 1650.0)
- **Finish**: Smooth
- **Length**: 0.381
- **Thickness**: 0.0015875
- **Aft Radius**: 0.0777875 (automatic)
- **Aft Shoulder Radius**: 0.0762
- **Aft Shoulder Length**: 0.1016
- **Aft Shoulder Thickness**: 0.003175
- **Aft Shoulder Capped**: False
- **Subcomponents**:
  - Nosecone Tip & Mount: Mass = 0.907, Position (top) = 0.00762, Packed Length = 0.0762, Packed Radius = 0.0125

#### Body Tube 1 (Upper Body Tube)
- **Material**: Fiber Glass in house (density: 1650.0)
- **Finish**: Smooth
- **Length**: 1.0
- **Thickness**: 0.0015875
- **Radius**: 0.0777875
- **Subcomponents**:
  - Payload Bulkhead: Material = Aluminum (density: 2700.0), Length = 0.0127, Outer Radius = 0.0762, Position (top) = 0.44958, Override Mass = 0.989
  - Payload: Mass = 2.835, Position (top) = 0.0381, Packed Length = 0.4064, Packed Radius = 0.0762, Type = Payload
  - U-Bolt: Mass = 0.188, Position (top) = 0.4572, Packed Length = 0.0635, Packed Radius = 0.00635
  - Shock Cord (40 ft including linkages): Material = Kevlar 12-strand (13 mm, 1/2 in) (line density: 0.11607678 kg/m), Cord Length = 12.192, Position (top) = 0.5207, Override Mass = 0.866, Packed Length = 0.0762, Packed Radius = 0.07366
  - Main Parachute + Bag: Material = Ripstop nylon (surface density: 0.067 kg/m²), CD = 2.2, Diameter = 2.7432, Line Count = 4, Line Length = 2.032, Line Material = Nylon flat webbing lg. (14 mm, 9/16 in) (line density: 0.02723097 kg/m), Deploy Event = Altitude, Deploy Altitude = 396.24 m, Deploy Delay = 0.0 s, Position (top) = 0.5969, Override Mass = 0.694, Packed Length = 0.2032, Packed Radius = 0.0508
  - Bulkhead: Material = ABS - 100% infill (density: 1050.0), Length = 0.0127, Outer Radius = Automatic, Position (top) = 0.0254

#### Body Tube 2 (Coupler Shoulder)
- **Material**: Fiber Glass in house (density: 1650.0)
- **Finish**: Smooth
- **Length**: 0.0762
- **Thickness**: 0.0015875
- **Radius**: 0.0777875
- **Subcomponents**:
  - Tube Coupler: Material = Fiber Glass in house (density: 1650.0), Length = 0.4572, Outer Radius = 0.0762, Thickness = 0.0015875, Position (middle) = 2.7755575615628914E-17
    - Subcomponents:
      - Telemetry: Mass = 0.680, Position (bottom) = -0.1992, Packed Length = 0.0508, Packed Radius = 0.0777875, Type = Flight Computer, Override Mass = 0.680
      - U-Bolt: Mass = 0.125, Position (top) = -0.0508, Packed Length = 0.0635, Packed Radius = 0.00635
      - U-Bolt Aft: Mass = 0.125, Position (top) = 0.4318, Packed Length = 0.0635, Packed Radius = 0.00635
      - Recovery Hardware: Mass = 0.181, Position (middle) = -0.00635, Packed Length = 0.18288, Packed Radius = 0.04445, Override Mass = 0.181, Type = Recovery Hardware
      - Recovery Disk: Mass = 0.2, Position (middle) = -0.00635, Packed Length = 0.18288, Packed Radius = 0.04445, Override Mass = 0.2, Type = Recovery Hardware
      - Coupler Bulkhead FWD With Charge Wells: Material = Aluminum (density: 2700.0), Length = 0.00889, Outer Radius = 0.0762, Position (bottom) = -0.45593, Override Mass = 0.562
      - Coupler Bulkhead AFT With Charge Wells: Material = Aluminum (density: 2700.0), Length = 0.00889, Outer Radius = 0.0762, Position (middle) = 0.23368, Override Mass = 0.562

#### Body Tube 3 (Lower Body Tube)
- **Material**: Fiber Glass in house (density: 1650.0)
- **Finish**: Smooth
- **Length**: 1.4392
- **Thickness**: 0.0015875
- **Radius**: 0.0777875
- **Subcomponents**:
  - Drogue Parachute: Material = Ripstop nylon (surface density: 0.067 kg/m²), CD = 2.2, Diameter = 0.6096, Line Count = 4, Line Length = 0.6096, Line Material = Nylon flat webbing lg. (14 mm, 9/16 in) (line density: 0.02723097 kg/m), Deploy Event = Apogee, Deploy Altitude = 200.0 m, Deploy Delay = 0.0 s, Position (top) = 0.4572, Override Mass = 0.135, Packed Length = 0.1016, Packed Radius = 0.0762
  - Shock Cord (35 ft including links): Material = Kevlar 12-strand (13 mm, 1/2 in) (line density: 0.11607678 kg/m), Cord Length = 10.668, Position (top) = 0.5842, Override Mass = 0.767, Packed Length = 0.0762, Packed Radius = 0.07366
  - Motor Casing Eye Bolt: Mass = 0.078, Position (bottom) = -0.70104, Packed Length = 0.025, Packed Radius = 0.0125
  - Inner Tube (Motor Mount): Material = Fiberglass (density: 1850.0), Length = 0.762, Outer Radius = 0.052832, Thickness = 0.003175, Position (bottom) = 0.0, Override Mass = 0.608, Cluster Configuration = Single
    - Motor Mount Details: Ignition Event = Automatic, Ignition Delay = 0.0 s, Overhang = 0.0508
    - Motors:
      - Configuration (cd82868e-2b30-451f-b30a-1f54c3c20b70): Cesaroni Technology 9994M3400-P (Reload), Diameter = 0.098, Length = 0.702, Delay = 0.0 s
      - Configuration (5a86e108-c396-4fd6-8d07-dceb59683e14, default): AeroTech M2500T (Reload), Diameter = 0.098, Length = 0.751, Delay = 0.0 s
      - Configuration (bd92ddcd-54cd-4b88-b577-a901372bd23e): Not specified
    - Subcomponents:
      - Centering Ring Forward: Material = Pine (density: 530.0), Length = 0.0127, Outer Radius = 0.0762, Inner Radius = 0.048895, Position (top) = 0.01016, Override Mass = 0.247
      - Centering Ring (Mid): Material = Pine (density: 530.0), Length = 0.0127, Outer Radius = 0.0762, Inner Radius = 0.049149, Position (bottom) = -0.2794, Override Mass = 0.290
      - Centering Ring (Aft): Material = Pine (density: 530.0), Length = 0.0127, Outer Radius = 0.0762, Inner Radius = 0.049149, Position (bottom) = -0.0127, Override Mass = 0.290
  - Launch Lug Forward: Material = Delrin (density: 1420.0), Radius = 0.0080518, Length = 0.0381, Thickness = 0.0080518, Position (top) = 0.3396742, Radial Direction = 60°, Instance Count = 1
  - Launch Lug Aft: Material = Delrin (density: 1420.0), Radius = 0.0080518, Length = 0.0381, Thickness = 0.0080518, Position (bottom) = -0.0856742, Radial Direction = 60°, Instance Count = 1
  - Freeform Fin Set: Fin Count = 4, Material = Carbon fiber (density: 1780.0), Thickness = 0.003175, Cross Section = Airfoil, Cant = 0.0°, Tab Height = 0.0254, Tab Length = 0.2921, Tab Position (relative to front/top) = 0.0, Fillet Radius = 0.0254, Fillet Material = Aluminum (density: 2700.0), Position (bottom) = -0.000002419, Override Mass = 0.907 (per fin set?)
    - Fin Points (x, y in meters): (0.0, 0.0), (0.254, 0.1524), (0.27305, 0.1524), (0.3048, 0.0)
  - Mass Component: Mass = 0.0, Position (top) = 0.0, Packed Length = 0.025, Packed Radius = 0.0125

#### Transition (Boat Tail)
- **Material**: Polycarbonate (Lexan) (density: 1200.0)
- **Finish**: Normal
- **Length**: 0.0508
- **Thickness**: Filled
- **Shape**: Conical
- **Fore Radius**: 0.0777875
- **Aft Radius**: 0.0635
- **Fore Shoulder Radius**: 0.0762
- **Fore Shoulder Length**: 0.0127
- **Fore Shoulder Thickness**: 0.024638
- **Fore Shoulder Capped**: False
- **Aft Shoulder Radius**: 0.0
- **Aft Shoulder Length**: 0.0
- **Aft Shoulder Thickness**: 0.0
- **Aft Shoulder Capped**: False
- **Override Mass**: 0.907

### Motor Configurations
- **Default Configuration ID**: 5a86e108-c396-4fd6-8d07-dceb59683e14 (AeroTech M2500T)
- Active stage: 0 (Sustainer) in all configurations.

This extracted data provides the necessary parameters for defining a RocketPy simulation, including rocket radius (primarily 0.0777875 m), component lengths for total length calculation (~3.0 m excluding overlaps), masses for inertia and CG, aerodynamic shapes (nose, fins), motor thrust curves (requires loading .eng or .rse files for specified motors), parachutes for recovery, and rail buttons (launch lugs) for launch rail simulation. For precise RocketPy implementation, convert fin points to TrapezoidalFins or RailFins, and ensure material densities are used for automatic mass calculations where overrides are not applied.