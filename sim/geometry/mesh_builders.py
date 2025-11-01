"""
Mesh generation functions for rocket components.

This module creates 3D meshes using PyVista primitives for:
- Motor components (casing, nozzle, closure)
- Nose cone (various shapes)
- Fins (trapezoidal)
- Body tube (cylinder)
- Tail (frustum/boat tail)
- Complete rocket assembly

Focus on minimal polygon count for maximum performance.

Coordinate fidelity: We do not impose a coordinate orientation. Instead, we
respect the RocketPy rocket's coordinate system (nose_to_tail or tail_to_nose)
and position/flip component meshes accordingly so the visualization matches
the RocketPy model.
"""

import pyvista as pv
import numpy as np
from .extraction import (
    MotorGeometry,
    NoseConeGeometry,
    FinGeometry,
    TailGeometry,
    RocketAssemblyGeometry
)


def create_motor_casing(geometry: MotorGeometry) -> pv.PolyData:
    """
    Create motor casing as simple solid cylinder.
    
    Minimal approach:
    - Single solid cylinder (outer surface only)
    - Low resolution for performance (24 radial divisions)
    - No internal geometry (not visible)
    
    Args:
        geometry: MotorGeometry instance with dimensional data
        
    Returns:
        PyVista PolyData mesh of motor casing
    """
    casing = pv.Cylinder(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        radius=geometry.casing_outer_radius,
        height=geometry.casing_length,
        resolution=24,  # Low poly for performance
        capping=True
    )
    
    return casing


def create_nozzle(geometry: MotorGeometry) -> pv.PolyData:
    """
    Create simple conical nozzle.
    
    Minimal approach:
    - Single cone from casing radius to nozzle exit radius
    - No complex convergent-divergent profile
    - Represents visual appearance only
    
    The nozzle is created with BASE at Z=0 (attaches to casing) and 
    APEX (pointy end) at Z=-divergence_length (extending downward/aft).
    
    Args:
        geometry: MotorGeometry instance with dimensional data
        
    Returns:
        PyVista PolyData mesh of nozzle
    """
    # Create cone with apex (pointy end) pointing downward in -Z direction
    # PyVista Cone: center is at the CENTER of the cone, direction is from apex to base
    # We want: base at Z=0, apex at Z=-divergence_length
    # So direction should be (0, 0, -1) to point apex downward
    nozzle = pv.Cone(
        center=(0, 0, -geometry.nozzle_divergence_length / 2),
        direction=(0, 0, -1),  # Points apex downward (from apex to base = negative Z to positive Z)
        height=geometry.nozzle_divergence_length,
        radius=geometry.nozzle_radius,  # Use actual nozzle exit radius, not casing radius
        resolution=24,  # Low poly
        capping=True
    )
    
    return nozzle


def create_forward_closure(geometry: MotorGeometry) -> pv.PolyData:
    """
    Create simple front end cap.
    
    Args:
        geometry: MotorGeometry instance with dimensional data
        
    Returns:
        PyVista PolyData mesh of forward closure
    """
    # Simple solid disc
    disc = pv.Disc(
        center=(0, 0, geometry.casing_length / 2),
        inner=0.0,  # Solid disc (no hole)
        outer=geometry.casing_outer_radius,
        normal=(0, 0, 1),
        r_res=1,  # Minimal radial resolution
        c_res=24  # Circumferential resolution
    )
    
    return disc


def assemble_motor_model(geometry: MotorGeometry) -> pv.MultiBlock:
    """
    Assemble all motor components into single multi-block mesh.
    
    Generates minimal geometry (outer surfaces only):
    - Motor casing (cylinder)
    - Nozzle (cone)
    - Forward closure (disc)
    
    Convention: Motor is positioned with nozzle BASE at Z=0 (where it connects to casing),
    nozzle extending DOWNWARD to Z=-divergence_length, and casing extending UPWARD to Z=+casing_length.
    
    This is the STANDARD orientation (before any directional flipping).
    
    Args:
        geometry: MotorGeometry instance with dimensional data
        
    Returns:
        PyVista MultiBlock containing all motor components
    """
    L = geometry.casing_length
    nozzle_len = geometry.nozzle_divergence_length
    
    # Casing: cylinder from Z=0 (nozzle attachment) to Z=+L (forward closure)
    casing = create_motor_casing(geometry)
    casing = casing.translate((0, 0, L / 2), inplace=False)
    
    # Nozzle: cone base at Z=0, apex extending to Z=-nozzle_len (downward/aft)
    # The nozzle mesh is already built this way
    nozzle = create_nozzle(geometry)
    
    # Closure: disc at Z=L (forward end of casing)
    closure = create_forward_closure(geometry)
    closure = closure.translate((0, 0, L / 2), inplace=False)
    
    # Combine into multi-block mesh
    motor = pv.MultiBlock()
    motor['casing'] = casing
    motor['nozzle'] = nozzle
    motor['closure'] = closure
    
    return motor


def create_nosecone(geometry: NoseConeGeometry) -> pv.PolyData:
    """
    Create nose cone mesh based on shape type.
    
    Supports various nose cone shapes:
    - Von Karman (Haack series)
    - Conical
    - Ogive
    - And more
    
    Args:
        geometry: NoseConeGeometry instance with dimensional data
        
    Returns:
        PyVista PolyData mesh of nose cone
    """
    # Number of points along the profile
    n_points = 50
    # Number of radial divisions
    resolution = 24
    
    # Generate nose cone profile based on kind
    if geometry.kind.lower() in ['von karman', 'vonkarman', 'von_karman']:
        # Von Karman (Haack series) profile
        # This is an optimal shape for minimum wave drag
        z = np.linspace(0, geometry.length, n_points)
        
        # Von Karman shape parameter (C = 0 for minimum drag)
        theta = np.arccos(1 - 2 * z / geometry.length)
        r = (geometry.base_radius / np.sqrt(np.pi)) * np.sqrt(
            theta - np.sin(2 * theta) / 2
        )
        
    elif geometry.kind.lower() in ['conical', 'cone']:
        # Simple conical profile
        z = np.linspace(0, geometry.length, n_points)
        r = geometry.base_radius * (z / geometry.length)
        
    elif geometry.kind.lower() in ['ogive', 'tangent ogive']:
        # Tangent ogive profile
        z = np.linspace(0, geometry.length, n_points)
        L = geometry.length
        R = geometry.base_radius
        
        # Ogive radius of curvature
        rho = (R**2 + L**2) / (2 * R)
        
        # Ogive profile
        y = np.sqrt(rho**2 - (L - z)**2)
        r = R - (rho - y)
        
    elif geometry.kind.lower() in ['parabolic']:
        # Parabolic profile
        z = np.linspace(0, geometry.length, n_points)
        K = 0.5  # Shape parameter
        r = geometry.base_radius * ((z / geometry.length) ** (1 - K))
        
    elif geometry.kind.lower() in ['elliptical']:
        # Elliptical profile
        z = np.linspace(0, geometry.length, n_points)
        r = geometry.base_radius * np.sqrt(1 - (z / geometry.length - 1)**2)
        
    else:
        # Default to conical if unknown
        print(f"Warning: Unknown nose cone kind '{geometry.kind}', using conical")
        z = np.linspace(0, geometry.length, n_points)
        r = geometry.base_radius * (z / geometry.length)
    
    # Create surface of revolution
    # Generate points in cylindrical coordinates, then convert to Cartesian
    points = []
    for i in range(resolution):
        angle = 2 * np.pi * i / resolution
        for j, (z_val, r_val) in enumerate(zip(z, r)):
            x = r_val * np.cos(angle)
            y = r_val * np.sin(angle)
            points.append([x, y, z_val])
    
    points = np.array(points)
    
    # Create faces (quads connecting adjacent points)
    faces = []
    for i in range(resolution):
        i_next = (i + 1) % resolution
        for j in range(n_points - 1):
            # Quad indices
            p1 = i * n_points + j
            p2 = i_next * n_points + j
            p3 = i_next * n_points + j + 1
            p4 = i * n_points + j + 1
            
            # Add two triangles to form quad
            faces.extend([3, p1, p2, p3])
            faces.extend([3, p1, p3, p4])
    
    # Create PolyData mesh
    nosecone = pv.PolyData(points, faces=faces)
    
    # Add base cap (disc at z = length)
    base_disc = pv.Disc(
        center=(0, 0, geometry.length),
        inner=0.0,
        outer=geometry.base_radius,
        normal=(0, 0, 1),
        r_res=1,
        c_res=resolution
    )
    
    # Combine nose and base
    nosecone = nosecone + base_disc
    
    return nosecone


def create_single_fin(geometry: FinGeometry) -> pv.PolyData:
    """
    Create a single trapezoidal fin.
    
    The fin is an extruded trapezoid attached to the body tube.
    
    Args:
        geometry: FinGeometry instance with dimensional data
        
    Returns:
        PyVista PolyData mesh of single fin
    """
    # Fin thickness (typical for carbon fiber fins)
    thickness = 0.003  # 3mm
    
    # Define fin profile points (trapezoid in Y-Z plane)
    # Root at body radius, extends outward in +Y direction
    # Z is along rocket axis
    
    # Fin vertices (before extrusion)
    # Root leading edge at origin
    root_le = [geometry.rocket_radius, 0, 0]
    # Root trailing edge
    root_te = [geometry.rocket_radius, 0, geometry.root_chord]
    # Tip trailing edge (swept back)
    tip_te = [geometry.rocket_radius + geometry.span, 0, geometry.sweep_length + geometry.tip_chord]
    # Tip leading edge
    tip_le = [geometry.rocket_radius + geometry.span, 0, geometry.sweep_length]
    
    # Create trapezoid points for both sides (offset by thickness/2)
    half_thick = thickness / 2
    
    points = [
        # Front face (y = -half_thick)
        [root_le[0], -half_thick, root_le[2]],
        [root_te[0], -half_thick, root_te[2]],
        [tip_te[0], -half_thick, tip_te[2]],
        [tip_le[0], -half_thick, tip_le[2]],
        # Back face (y = +half_thick)
        [root_le[0], half_thick, root_le[2]],
        [root_te[0], half_thick, root_te[2]],
        [tip_te[0], half_thick, tip_te[2]],
        [tip_le[0], half_thick, tip_le[2]],
    ]
    
    # Define faces (each face is a list of point indices)
    faces = [
        # Front face (quad)
        4, 0, 1, 2, 3,
        # Back face (quad)
        4, 4, 5, 6, 7,
        # Top edge (leading edge to tip)
        4, 0, 3, 7, 4,
        # Bottom edge (root)
        4, 0, 1, 5, 4,
        # Outer edge (tip)
        4, 3, 2, 6, 7,
        # Trailing edge
        4, 1, 2, 6, 5,
    ]
    
    fin = pv.PolyData(points, faces=faces)
    
    return fin


def create_fin_set(geometry: FinGeometry) -> pv.MultiBlock:
    """
    Create complete fin set with radial symmetry.
    
    Positions n fins equally spaced around body circumference.
    
    Args:
        geometry: FinGeometry instance with dimensional data
        
    Returns:
        PyVista MultiBlock containing all fins
    """
    # Create single fin
    single_fin = create_single_fin(geometry)
    
    # Create MultiBlock for fin set
    fin_set = pv.MultiBlock()
    
    # Angular spacing between fins
    angle_step = 360.0 / geometry.n
    
    # Position fins around circumference
    for i in range(geometry.n):
        angle = i * angle_step
        
        # Rotate fin around Z-axis
        rotated_fin = single_fin.rotate_z(angle, inplace=False)
        
        fin_set[f'fin_{i+1}'] = rotated_fin
    
    return fin_set


def create_body_tube(radius: float, length: float) -> pv.PolyData:
    """
    Create body tube as simple cylinder.
    
    Cylinder is created centered at origin, spanning from Z=-length/2 to Z=+length/2.
    
    Args:
        radius: Body tube radius (m)
        length: Body tube length (m)
        
    Returns:
        PyVista PolyData mesh of body tube
    """
    body = pv.Cylinder(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        radius=radius,
        height=length,
        resolution=24,
        capping=False  # Open ends (connects to nose and tail)
    )
    
    return body


def create_tail(geometry: TailGeometry) -> pv.PolyData:
    """
    Create tail/boat tail as truncated cone (frustum).
    
    The tail transitions from larger diameter (top) to smaller diameter (bottom).
    
    Args:
        geometry: TailGeometry instance with dimensional data
        
    Returns:
        PyVista PolyData mesh of tail
    """
    resolution = 24
    
    # Create frustum using cone with truncation
    # PyVista doesn't have direct frustum, so we build it manually
    
    # Generate points for frustum
    points = []
    
    # Top circle (larger radius)
    for i in range(resolution):
        angle = 2 * np.pi * i / resolution
        x = geometry.top_radius * np.cos(angle)
        y = geometry.top_radius * np.sin(angle)
        z = 0.0
        points.append([x, y, z])
    
    # Bottom circle (smaller radius)
    for i in range(resolution):
        angle = 2 * np.pi * i / resolution
        x = geometry.bottom_radius * np.cos(angle)
        y = geometry.bottom_radius * np.sin(angle)
        z = geometry.length
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Create faces connecting top and bottom circles
    faces = []
    for i in range(resolution):
        i_next = (i + 1) % resolution
        
        # Top indices
        p1 = i
        p2 = i_next
        # Bottom indices
        p3 = resolution + i_next
        p4 = resolution + i
        
        # Create quad (as two triangles)
        faces.extend([3, p1, p2, p3])
        faces.extend([3, p1, p3, p4])
    
    # Create top cap
    top_face = [resolution] + list(range(resolution))
    faces.extend(top_face)
    
    # Create bottom cap
    bottom_face = [resolution] + list(range(resolution, 2 * resolution))
    faces.extend(bottom_face)
    
    tail = pv.PolyData(points, faces=faces)
    
    return tail


def assemble_rocket_model(geometry: RocketAssemblyGeometry) -> pv.MultiBlock:
    """
    Assemble complete rocket with all components positioned correctly.
    
    CRITICAL: This is a FAITHFUL 1:1 representation of RocketPy's coordinate system.
    
    RocketPy Coordinate System (tail_to_nose):
    - Z+ points UPWARD (from tail toward nose) - like the rocket in vertical flight
    - Nose tip is at Z=0 (reference point)
    - Components toward tail have NEGATIVE Z values
    - ALL components point in +Z direction (upward/forward)
    
    Component Standard Meshes (before positioning):
    - Nosecone: tip at 0, base at +length (points in +Z)
    - Fins: LE at 0, TE at +root_chord (extend in +Z)
    - Tail: top at 0, bottom at +length (extends in +Z)
    - Motor: nozzle at 0, closure at +length (extends in +Z)
    
    Args:
        geometry: RocketAssemblyGeometry instance with all component data
        
    Returns:
        PyVista MultiBlock containing complete rocket assembly
    """
    assembly = pv.MultiBlock()
    
    # Get coordinate system
    coord_sys = getattr(geometry, 'coordinate_system', 'tail_to_nose')
    
    # Track extents for body tube calculation
    nose_base_z = None
    tail_top_z = None
    
    # 1. NOSE CONE
    # RocketPy position: nose TIP location (should be at Z=0)
    # Need: TIP at nosecone_position, BASE extending DOWNWARD (toward tail, -Z direction)
    if geometry.nosecone is not None:
        nosecone_mesh = create_nosecone(geometry.nosecone)
        
        # Flip nosecone to point downward (tip at 0, base at -length)
        nosecone_mesh = nosecone_mesh.rotate_x(180, inplace=False)
        
        # Now translate so tip is at nosecone_position
        tip_position = geometry.nosecone_position
        nosecone_mesh = nosecone_mesh.translate((0, 0, tip_position), inplace=False)
        assembly['nosecone'] = nosecone_mesh
        
        # Base is now at tip_position - length (more negative)
        nose_base_z = tip_position - geometry.nosecone.length
    
    # 2. TAIL (Boat Tail)
    # RocketPy position: TOP (forward end) of tail
    # Need: TOP at tail_position, extending DOWNWARD (toward aft, -Z direction)
    if geometry.tail is not None:
        tail_mesh = create_tail(geometry.tail)
        
        # Flip tail to extend downward (top at 0, bottom at -length)
        tail_mesh = tail_mesh.rotate_x(180, inplace=False)
        
        # Translate so top is at tail_position
        top_position = geometry.tail_position
        tail_mesh = tail_mesh.translate((0, 0, top_position), inplace=False)
        assembly['tail'] = tail_mesh
        
        # Store tail top for body tube
        tail_top_z = top_position
    
    # 3. FINS
    # RocketPy position: root LEADING EDGE location
    # Need: LE at fins_position, extending DOWNWARD (toward tail, -Z direction)
    if geometry.fins is not None:
        fin_set_mesh = create_fin_set(geometry.fins)
        
        # Flip fins to extend downward (LE at 0, TE at -chord)
        # This makes the trailing edge (wider part) toward the tail
        for key in fin_set_mesh.keys():
            fin_set_mesh[key] = fin_set_mesh[key].rotate_x(180, inplace=False)
        
        # Translate to place LE at fins_position
        le_position = geometry.fins_position
        for key in fin_set_mesh.keys():
            fin_set_mesh[key] = fin_set_mesh[key].translate((0, 0, le_position), inplace=False)
            assembly[key] = fin_set_mesh[key]
    
    # 4. BODY TUBE
    # Span from nose base to tail top
    if nose_base_z is not None and tail_top_z is not None:
        # nose_base_z is more negative (toward tail)
        # tail_top_z is less negative or closer to 0 (toward nose)
        body_length = abs(tail_top_z - nose_base_z)
        body_center_z = (nose_base_z + tail_top_z) / 2
        
        if body_length > 0.01:  # At least 1cm
            body_tube_mesh = create_body_tube(geometry.radius, body_length)
            body_tube_mesh = body_tube_mesh.translate((0, 0, body_center_z), inplace=False)
            assembly['body'] = body_tube_mesh
    
    # 5. MOTOR
    # CRITICAL: Faithfully position the motor using ONLY RocketPy data.
    #
    # Known from RocketPy objects we receive via extraction:
    # - geometry.motor.nozzle_position: distance from motor origin (combustion
    #   chamber/forward end) to the nozzle base measured along the motor +Z
    #   direction ("combustion_chamber_to_nozzle").
    # - geometry.motor_position: motor origin position in the rocket axis.
    # - geometry.coordinate_system: 'tail_to_nose' or 'nose_to_tail' for the rocket.
    #
    # Our standard motor mesh is built with:
    # - Nozzle base at Z = 0 (apex extends to negative Z),
    # - Casing from Z = 0 to Z = +casing_length,
    # - Forward closure at Z = +casing_length.
    #
    # Therefore, to place the motor faithfully we align the nozzle BASE using
    # RocketPy's semantics, never inferring from a made-up casing length:
    #
    # If rocket coordinate is 'tail_to_nose' (Z+ forward):
    #   motor_local +Z (toward nozzle) maps to rocket -Z, so
    #   nozzle_base_z = motor_position - nozzle_position
    # If rocket coordinate is 'nose_to_tail' (Z+ aft):
    #   motor_local +Z maps to rocket +Z, so
    #   nozzle_base_z = motor_position + nozzle_position
    #
    # Then translate the whole motor so its nozzle base (currently at Z=0 in
    # our mesh) sits at nozzle_base_z. This avoids any assumptions about
    # casing length when positioning.
    if geometry.motor is not None:
        motor_mesh = assemble_motor_model(geometry.motor)
        
        # Compute nozzle base global Z from RocketPy semantics
        if coord_sys == 'tail_to_nose':
            nozzle_base_z = geometry.motor_position - geometry.motor.nozzle_position
        else:  # 'nose_to_tail'
            nozzle_base_z = geometry.motor_position + geometry.motor.nozzle_position
        
        # Translate motor so its nozzle base (mesh Z=0) is at nozzle_base_z
        offset = nozzle_base_z
        
        for key in motor_mesh.keys():
            motor_mesh[key] = motor_mesh[key].translate((0, 0, offset), inplace=False)
        
        assembly['motor_casing'] = motor_mesh['casing']
        assembly['motor_nozzle'] = motor_mesh['nozzle']
        assembly['motor_closure'] = motor_mesh['closure']
    
    return assembly