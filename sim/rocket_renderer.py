"""
Rocket 3D Renderer for RocketPy

Provides simple API for rendering RocketPy Rocket objects in 3D.

Usage:
    from sim.rocket_renderer import render_rocket
    
    # Render full assembly
    render_rocket(rocket)
    
    # Render specific components
    render_rocket(rocket, components=['nosecone', 'fins'])
    
    # Render motor only
    render_rocket(rocket, components='motor')

For the GUI application:
    from sim import launch_gui
    
    # Launch GUI with rocket
    launch_gui(rocket)
"""

from .geometry.extraction import (
    MotorGeometry,
    NoseConeGeometry,
    FinGeometry,
    TailGeometry,
    RocketAssemblyGeometry
)
from .geometry.mesh_builders import (
    assemble_motor_model,
    assemble_rocket_model
)
from .rendering.renderer import RocketRenderer
from .cache.mesh_cache import MeshCache


def render_rocket(rocket, components='all', **kwargs):
    """
    Render a RocketPy Rocket in 3D with optional component selection.
    
    Simple convenience function for quick visualization.
    
    Args:
        rocket: RocketPy Rocket object (with all components added)
        components: Components to render (default 'all'):
            - 'all': Complete rocket assembly
            - 'motor': Motor only
            - 'nosecone': Nose cone only
            - 'body': Body tube only
            - 'fins': Fin set only
            - 'tail': Boat tail only
            - List: e.g., ['nosecone', 'fins', 'motor']
        **kwargs: Rendering options passed to RocketRenderer.render()
            - show_edges (bool): Show mesh edges. Default True
            - background (str): Background color. Default 'white'
            - window_size (tuple): (width, height) in pixels. Default (1000, 800)
            - jupyter_backend (str): 'static', 'trame', or 'none'. Default 'static'
    
    Examples:
        >>> from rocketpy import Rocket
        >>> from sim.rocket_renderer import render_rocket
        >>> 
        >>> # Render full assembly
        >>> render_rocket(rocket)
        >>> 
        >>> # Render specific components
        >>> render_rocket(rocket, components=['nosecone', 'fins'])
        >>> 
        >>> # Render motor only
        >>> render_rocket(rocket, components='motor')
    """
    renderer = RocketRenderer(rocket)
    return renderer.render(components=components, **kwargs)


# Export main classes and functions
__all__ = [
    # Renderer
    'RocketRenderer',
    # Convenience function
    'render_rocket',
    # Geometry classes
    'MotorGeometry',
    'NoseConeGeometry',
    'FinGeometry',
    'TailGeometry',
    'RocketAssemblyGeometry',
    # Mesh builders
    'assemble_motor_model',
    'assemble_rocket_model',
    # Cache
    'MeshCache',
]
