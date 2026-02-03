"""
Geometry extraction and mesh generation for rocket components.
"""

from .extraction import (
    MotorGeometry,
    NoseConeGeometry,
    FinGeometry,
    TailGeometry,
    RocketAssemblyGeometry
)
from .mesh_builders import (
    assemble_motor_model,
    assemble_rocket_model,
    create_nosecone,
    create_fin_set,
    create_body_tube,
    create_tail
)

__all__ = [
    # Geometry classes
    'MotorGeometry',
    'NoseConeGeometry',
    'FinGeometry',
    'TailGeometry',
    'RocketAssemblyGeometry',
    # Mesh builders
    'assemble_motor_model',
    'assemble_rocket_model',
    'create_nosecone',
    'create_fin_set',
    'create_body_tube',
    'create_tail',
]
