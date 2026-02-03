"""
Extract geometric parameters from RocketPy objects.

This module provides geometry extraction for all rocket components:
- MotorGeometry: SolidMotor dimensions
- NoseConeGeometry: NoseCone shape and dimensions
- FinGeometry: Fin dimensions and configuration
- TailGeometry: Tail/transition dimensions
- RocketAssemblyGeometry: Complete rocket assembly with positions

Important: We do not assume a fixed coordinate orientation here. We faithfully
carry RocketPy's coordinate system through ("nose_to_tail" or "tail_to_nose")
so downstream mesh builders can render accordingly without re-defining axes.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class MotorGeometry:
    """
    Container for motor geometric parameters extracted from RocketPy SolidMotor.
    
    This class stores all dimensional data needed for 3D rendering,
    with minimal assumptions for missing data.
    """
    
    # Grain geometry (from RocketPy)
    grain_outer_radius: float
    grain_inner_radius: float
    grain_height: float
    
    # Nozzle geometry (from RocketPy)
    nozzle_radius: float
    throat_radius: float
    nozzle_position: float
    
    # Inferred casing dimensions
    casing_thickness: float = 0.005  # 5mm typical motor casing
    casing_outer_radius: float = None
    casing_length: float = None
    
    # Inferred nozzle geometry (simplified)
    nozzle_convergence_length: float = 0.05  # 50mm typical
    nozzle_divergence_length: float = 0.10  # 100mm typical
    
    def __post_init__(self):
        """Calculate derived dimensions after initialization."""
        # Calculate casing outer radius if not provided
        if self.casing_outer_radius is None:
            self.casing_outer_radius = self.grain_outer_radius + self.casing_thickness
        
        # Calculate casing length if not provided
        if self.casing_length is None:
            self.casing_length = self.grain_height + 0.1  # Extend 100mm beyond grain
    
    @classmethod
    def from_rocketpy_motor(cls, motor) -> 'MotorGeometry':
        """
        Create MotorGeometry from a RocketPy SolidMotor object.
        
        Args:
            motor: RocketPy SolidMotor instance
            
        Returns:
            MotorGeometry instance with extracted parameters
        """
        return cls(
            grain_outer_radius=motor.grain_outer_radius,
            grain_inner_radius=motor.grain_initial_inner_radius,
            grain_height=motor.grain_initial_height,
            nozzle_radius=motor.nozzle_radius,
            throat_radius=motor.throat_radius,
            nozzle_position=motor.nozzle_position,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export geometry as dictionary for caching.
        
        Returns:
            Dictionary of all geometric parameters
        """
        return asdict(self)
    
    def __repr__(self) -> str:
        """Pretty string representation."""
        return (
            f"MotorGeometry(\n"
            f"  grain: outer_r={self.grain_outer_radius:.4f}m, "
            f"inner_r={self.grain_inner_radius:.4f}m, h={self.grain_height:.4f}m\n"
            f"  casing: outer_r={self.casing_outer_radius:.4f}m, "
            f"length={self.casing_length:.4f}m\n"
            f"  nozzle: exit_r={self.nozzle_radius:.4f}m, "
            f"throat_r={self.throat_radius:.4f}m\n"
            f")"
        )


@dataclass
class NoseConeGeometry:
    """
    Container for nose cone geometric parameters extracted from RocketPy NoseCone.
    
    Stores dimensional data needed for 3D rendering of nose cone shapes.
    Supports various nose cone kinds (Von Karman, ogive, conical, etc.)
    """
    
    length: float  # Axial length from tip to base (m)
    base_radius: float  # Radius at base (m)
    kind: str  # Shape type: 'Von Karman', 'conical', 'ogive', etc.
    rocket_radius: float  # Body tube radius for proper connection (m)
    
    @classmethod
    def from_rocketpy_nosecone(cls, nosecone) -> 'NoseConeGeometry':
        """
        Create NoseConeGeometry from a RocketPy NoseCone object.
        
        Args:
            nosecone: RocketPy NoseCone instance
            
        Returns:
            NoseConeGeometry instance with extracted parameters
        """
        return cls(
            length=nosecone.length,
            base_radius=nosecone.base_radius,
            kind=nosecone.kind,
            rocket_radius=nosecone.rocket_radius,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export geometry as dictionary for caching."""
        return asdict(self)
    
    def __repr__(self) -> str:
        """Pretty string representation."""
        return (
            f"NoseConeGeometry(\n"
            f"  length={self.length:.4f}m, base_radius={self.base_radius:.4f}m\n"
            f"  kind='{self.kind}', rocket_radius={self.rocket_radius:.4f}m\n"
            f")"
        )


@dataclass
class FinGeometry:
    """
    Container for fin geometric parameters extracted from RocketPy TrapezoidalFins.
    
    Stores dimensional data needed for 3D rendering of fin sets.
    """
    
    n: int  # Number of fins
    root_chord: float  # Length at root (attached to body) (m)
    tip_chord: float  # Length at tip (m)
    span: float  # Height perpendicular to body (m)
    sweep_length: float  # Horizontal distance from root to tip leading edge (m)
    rocket_radius: float  # Body tube radius (m)
    
    @classmethod
    def from_rocketpy_fins(cls, fins) -> 'FinGeometry':
        """
        Create FinGeometry from a RocketPy TrapezoidalFins object.
        
        Args:
            fins: RocketPy TrapezoidalFins instance
            
        Returns:
            FinGeometry instance with extracted parameters
        """
        return cls(
            n=fins.n,
            root_chord=fins.root_chord,
            tip_chord=fins.tip_chord,
            span=fins.span,
            sweep_length=fins.sweep_length,
            rocket_radius=fins.rocket_radius,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export geometry as dictionary for caching."""
        return asdict(self)
    
    def __repr__(self) -> str:
        """Pretty string representation."""
        return (
            f"FinGeometry(\n"
            f"  n={self.n}, root_chord={self.root_chord:.4f}m, "
            f"tip_chord={self.tip_chord:.4f}m\n"
            f"  span={self.span:.4f}m, sweep={self.sweep_length:.4f}m, "
            f"rocket_radius={self.rocket_radius:.4f}m\n"
            f")"
        )


@dataclass
class TailGeometry:
    """
    Container for tail/transition geometric parameters extracted from RocketPy Tail.
    
    Stores dimensional data for boat tail transitions (frustum geometry).
    """
    
    top_radius: float  # Radius at forward end (m)
    bottom_radius: float  # Radius at aft end (m)
    length: float  # Axial length (m)
    rocket_radius: float  # Body tube radius (m)
    
    @classmethod
    def from_rocketpy_tail(cls, tail) -> 'TailGeometry':
        """
        Create TailGeometry from a RocketPy Tail object.
        
        Args:
            tail: RocketPy Tail instance
            
        Returns:
            TailGeometry instance with extracted parameters
        """
        return cls(
            top_radius=tail.top_radius,
            bottom_radius=tail.bottom_radius,
            length=tail.length,
            rocket_radius=tail.rocket_radius,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export geometry as dictionary for caching."""
        return asdict(self)
    
    def __repr__(self) -> str:
        """Pretty string representation."""
        return (
            f"TailGeometry(\n"
            f"  top_radius={self.top_radius:.4f}m, "
            f"bottom_radius={self.bottom_radius:.4f}m\n"
            f"  length={self.length:.4f}m, "
            f"rocket_radius={self.rocket_radius:.4f}m\n"
            f")"
        )


@dataclass
class RocketAssemblyGeometry:
    """
    Container for complete rocket assembly geometry extracted from RocketPy Rocket.
    
    Stores all component geometries and their positions for full assembly rendering.
    No fixed axis assumption is made; the coordinate system is preserved from the
    RocketPy object via `coordinate_system`.
    """
    
    # Body tube
    radius: float  # Body tube radius (m)
    # Coordinate system from RocketPy: 'nose_to_tail' or 'tail_to_nose'
    coordinate_system: str
    
    # Components with positions
    motor: MotorGeometry
    motor_position: float  # Position of motor nozzle base (attachment to casing) from nose tip (m)
    
    nosecone: NoseConeGeometry
    nosecone_position: float  # Position from nose tip (m) - typically 0.0
    
    fins: FinGeometry
    fins_position: float  # Position from nose tip (m)
    
    tail: TailGeometry
    tail_position: float  # Position from nose tip (m)
    
    # Derived dimensions
    total_length: float = None  # Total rocket length (m)
    
    def __post_init__(self):
        """Calculate derived dimensions after initialization.

        Compute overall axial span using min/max across component extents so it
        works for both coordinate orientations.
        """
        if self.total_length is None:
            extents = []
            # Nose cone extends from tip to base
            if self.nosecone is not None:
                tip = float(self.nosecone_position)
                base = tip + (self.nosecone.length if self.coordinate_system == 'nose_to_tail' else -self.nosecone.length)
                extents.extend([tip, base])
            # Fins extent along root chord from leading to trailing edge
            if self.fins is not None:
                le = float(self.fins_position)
                te = le + (self.fins.root_chord if self.coordinate_system == 'nose_to_tail' else -self.fins.root_chord)
                extents.extend([le, te])
            # Tail from top (forward) to bottom (aft)
            if self.tail is not None:
                top = float(self.tail_position)
                bottom = top + (self.tail.length if self.coordinate_system == 'nose_to_tail' else -self.tail.length)
                extents.extend([top, bottom])
            # Motor length from nozzle (aft) toward forward
            if self.motor is not None:
                noz = float(self.motor_position)
                motor_front = noz - (self.motor.casing_length if self.coordinate_system == 'nose_to_tail' else -self.motor.casing_length)
                extents.extend([noz, motor_front])

            if extents:
                self.total_length = max(extents) - min(extents)
            else:
                self.total_length = 0.0
    
    @classmethod
    def from_rocketpy_rocket(cls, rocket) -> 'RocketAssemblyGeometry':
        """
        Create RocketAssemblyGeometry from a RocketPy Rocket object.
        
        Args:
            rocket: RocketPy Rocket instance with all components added
            
        Returns:
            RocketAssemblyGeometry instance with all extracted parameters
        """
        # Extract motor
        motor_geometry = MotorGeometry.from_rocketpy_motor(rocket.motor)
        motor_position = rocket.motor_position
        
        # Convert motor_position to float if it's a Vector-like (has .z)
        if hasattr(motor_position, 'z'):
            motor_position = float(motor_position.z)
        else:
            motor_position = float(motor_position)
        
        # Extract aerodynamic surfaces and their axial positions
        nosecone = None
        nosecone_position = 0.0
        fins = None
        fins_position = 0.0
        tail = None
        tail_position = 0.0
        
        for surface in rocket.aerodynamic_surfaces:
            surface_obj = surface[0]
            position = surface[1]
            
            # Convert position to float if it's a Vector-like
            if hasattr(position, 'z'):
                position = float(position.z)
            else:
                position = float(position)
            
            surface_type = type(surface_obj).__name__
            if surface_type == 'NoseCone':
                nosecone = NoseConeGeometry.from_rocketpy_nosecone(surface_obj)
                nosecone_position = position
            elif surface_type == 'TrapezoidalFins':
                fins = FinGeometry.from_rocketpy_fins(surface_obj)
                fins_position = position
            elif surface_type == 'Tail':
                tail = TailGeometry.from_rocketpy_tail(surface_obj)
                tail_position = position
        
        coord = getattr(rocket, 'coordinate_system_orientation', 'nose_to_tail')
        return cls(
            radius=rocket.radius,
            coordinate_system=coord,
            motor=motor_geometry,
            motor_position=motor_position,
            nosecone=nosecone,
            nosecone_position=nosecone_position,
            fins=fins,
            fins_position=fins_position,
            tail=tail,
            tail_position=tail_position,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export geometry as dictionary for caching."""
        data = asdict(self)
        return data
    
    def __repr__(self) -> str:
        """Pretty string representation."""
        return (
            f"RocketAssemblyGeometry(\n"
            f"  radius={self.radius:.4f}m, total_length={self.total_length:.4f}m\n"
            f"  motor @ {self.motor_position:.4f}m\n"
            f"  nosecone @ {self.nosecone_position:.4f}m\n"
            f"  fins @ {self.fins_position:.4f}m\n"
            f"  tail @ {self.tail_position:.4f}m\n"
            f")"
        )
