"""
3D renderer for RocketPy Rocket objects.

Provides RocketRenderer for complete rocket assembly with component selection.
Motor components are rendered as part of the rocket assembly.

COORDINATE SYSTEM:
-----------------
This renderer faithfully represents RocketPy's coordinate system:
    - 'tail_to_nose' (DEFAULT): Z+ points from tail to nose (UP in vertical flight)
    - 'nose_to_tail': Z+ points from nose to tail (DOWN in vertical flight)

For flight simulation, ALWAYS use 'tail_to_nose' in RocketPy so that:
    +Z axis: Points UP (skyward) - nose cone direction in vertical flight
    +X axis: Points to the right (starboard)
    +Y axis: Points forward

The renderer does NOT modify or compensate for coordinate system orientation.
It is a faithful 3D representation of the RocketPy model's geometry.
"""

import pyvista as pv
from typing import Tuple, Optional, List, Union
from ..geometry.extraction import (
    RocketAssemblyGeometry
)
from ..geometry.mesh_builders import (
    assemble_rocket_model
)


class RocketRenderer:
    """
    3D renderer for RocketPy Rocket objects with component selection.
    
    Features:
    - Extracts geometry from RocketPy rocket (all components)
    - Generates 3D mesh for complete assembly (cached for performance)
    - Supports selective component rendering
    - Provides interactive visualization
    
    Component Selection:
    - 'all': Complete rocket assembly
    - 'motor': Motor only
    - 'nosecone': Nose cone only
    - 'body': Body tube only
    - 'fins': Fin set only
    - 'tail': Boat tail only
    - List of components: e.g., ['nosecone', 'body', 'fins']
    
    Example:
        >>> from rocketpy import Rocket
        >>> from sim.rendering import RocketRenderer
        >>> 
        >>> rocket = Rocket(...)
        >>> renderer = RocketRenderer(rocket)
        >>> 
        >>> # Render full assembly
        >>> renderer.render(components='all')
        >>> 
        >>> # Render specific components
        >>> renderer.render(components=['nosecone', 'fins'])
        >>> 
        >>> # Render motor only
        >>> renderer.render(components='motor')
    """
    
    def __init__(self, rocket, cache: bool = True):
        """
        Initialize renderer with a RocketPy rocket.
        
        Args:
            rocket: RocketPy Rocket object (with all components added)
            cache: If True, cache generated mesh for reuse
        """
        self.rocket = rocket
        self.cache_enabled = cache
        self._cached_mesh: Optional[pv.MultiBlock] = None
        self._geometry: Optional[RocketAssemblyGeometry] = None
        
        # Detect coordinate system orientation for labeling
        self.coord_system = getattr(rocket, 'coordinate_system_orientation', 'tail_to_nose')
        
        # Warn if using inverted coordinate system (bad for flight simulation)
        if self.coord_system == 'nose_to_tail':
            print(f"⚠️  WARNING: Rocket uses 'nose_to_tail' coordinate system")
            print(f"   Z+ points from nose to tail (DOWN in vertical flight)")
            print(f"   This is INCORRECT for flight simulation!")
            print(f"   Recommendation: Change to 'tail_to_nose' in RocketPy")
            print(f"   See COORDINATE_SYSTEM.md for details")

    
    def extract_geometry(self) -> RocketAssemblyGeometry:
        """
        Extract geometric parameters from RocketPy rocket.
        
        Returns:
            RocketAssemblyGeometry instance with all component dimensions
        """
        if self._geometry is None:
            self._geometry = RocketAssemblyGeometry.from_rocketpy_rocket(self.rocket)
        return self._geometry
    
    def generate_mesh(self, force_rebuild: bool = False) -> pv.MultiBlock:
        """
        Generate 3D mesh from rocket geometry.
        
        Faithful representation of RocketPy's coordinate system (no transformations applied).
        
        Args:
            force_rebuild: If True, ignore cache and rebuild mesh
            
        Returns:
            PyVista MultiBlock containing all rocket components as defined in RocketPy
        """
        # Check cache
        if not force_rebuild and self._cached_mesh is not None:
            return self._cached_mesh
        
        # Extract geometry
        geometry = self.extract_geometry()
        
        # Generate mesh (faithful to RocketPy's coordinate system)
        mesh = assemble_rocket_model(geometry)
        
        # Cache if enabled
        if self.cache_enabled:
            self._cached_mesh = mesh
        
        return mesh
    
    def _filter_components(
        self, 
        mesh: pv.MultiBlock, 
        components: Union[str, List[str]]
    ) -> pv.MultiBlock:
        """
        Filter mesh to include only selected components.
        
        Args:
            mesh: Full rocket assembly mesh
            components: Component selection ('all', 'motor', etc., or list)
            
        Returns:
            Filtered MultiBlock with selected components only
        """
        # Handle 'all' case
        if components == 'all':
            return mesh
        
        # Convert single string to list
        if isinstance(components, str):
            components = [components]
        
        # Create new MultiBlock with selected components
        filtered = pv.MultiBlock()
        
        # Component name mapping
        component_map = {
            'motor': ['motor_casing', 'motor_nozzle', 'motor_closure'],
            'nosecone': ['nosecone'],
            'body': ['body'],
            'fins': [k for k in mesh.keys() if k.startswith('fin_')],
            'tail': ['tail'],
        }
        
        # Collect all keys to include
        keys_to_include = []
        for comp in components:
            comp_lower = comp.lower()
            if comp_lower in component_map:
                keys_to_include.extend(component_map[comp_lower])
            else:
                # Try exact key match
                if comp in mesh.keys():
                    keys_to_include.append(comp)
        
        # Add selected components to filtered mesh
        for key in keys_to_include:
            if key in mesh.keys():
                filtered[key] = mesh[key]
        
        return filtered
    
    def render(self, 
               components: Union[str, List[str]] = 'all',
               show_edges: bool = True,
               background: str = 'white',
               window_size: Tuple[int, int] = (1000, 800),
               jupyter_backend: str = 'static',
               return_plotter: bool = False,
               show: bool = True,
               **kwargs):
        """
        Render rocket in interactive 3D viewer.
        
        Args:
            components: Components to render:
                - 'all': Complete assembly (default)
                - 'motor': Motor only
                - 'nosecone': Nose cone only
                - 'body': Body tube only
                - 'fins': Fin set only
                - 'tail': Boat tail only
                - List: e.g., ['nosecone', 'fins', 'motor']
            show_edges: Display mesh edges for better depth perception
            background: Background color ('white', 'black', etc.)
            window_size: (width, height) in pixels
            jupyter_backend: 'static', 'trame', 'panel', or 'none'
            return_plotter: If True, return plotter instead of showing
            show: If True, display the window (set False for GUI embedding)
            **kwargs: Additional arguments for plotter.add_mesh()
            
        Returns:
            None, Plotter if return_plotter=True, or image if show=False
        """
        # Set Jupyter backend if in notebook
        if jupyter_backend != 'none':
            try:
                pv.set_jupyter_backend(jupyter_backend)
            except Exception as e:
                print(f"Warning: Could not set Jupyter backend '{jupyter_backend}': {e}")
                print("Falling back to 'static' backend")
                pv.set_jupyter_backend('static')
        
        # Get full mesh
        full_mesh = self.generate_mesh()
        
        # Filter to selected components
        mesh = self._filter_components(full_mesh, components)
        
        # Create plotter
        plotter = pv.Plotter(window_size=window_size)
        plotter.set_background(background)
        
        # Color scheme for components
        colors = {
            'motor_casing': 'silver',
            'motor_nozzle': 'dimgray',
            'motor_closure': 'silver',
            'nosecone': 'red',
            'body': 'white',
            'tail': 'lightgray',
        }
        
        # Fin colors (cycle through colors for multiple fins)
        fin_colors = ['blue', 'blue', 'blue', 'blue']
        
        # Add components to plotter
        for i, key in enumerate(mesh.keys()):
            # Determine color
            if key.startswith('fin_'):
                fin_idx = int(key.split('_')[1]) - 1
                color = fin_colors[fin_idx % len(fin_colors)]
                label = f'Fin {fin_idx + 1}'
            else:
                color = colors.get(key, 'gray')
                label = key.replace('_', ' ').title()
            
            # Add mesh
            plotter.add_mesh(
                mesh[key],
                color=color,
                opacity=1.0,
                show_edges=show_edges,
                label=label,
                smooth_shading=True
            )
        
        # Add legend and axes
        plotter.add_legend(bcolor='white', face=None)
        
        # Axis labels based on coordinate system
        if self.coord_system == 'tail_to_nose':
            z_label = 'Z (m) ↑ UP (tail→nose)'
        else:
            z_label = 'Z (m) ↓ DOWN (nose→tail) ⚠️'
        
        plotter.add_axes(
            xlabel='X (m)',
            ylabel='Y (m)',
            zlabel=z_label,
        )
        plotter.add_camera_orientation_widget()
        
        # Set nice camera angle
        plotter.camera_position = 'iso'
        
        # Return plotter or show
        if return_plotter:
            return plotter
        elif show:
            return plotter.show()
        else:
            return plotter
    
    def save_model(self, filepath: str, components: Union[str, List[str]] = 'all'):
        """
        Export rocket mesh to file.
        
        Args:
            filepath: Output file path (should end in .stl, .obj, .ply, or .vtk)
            components: Components to export (same options as render)
        """
        # Get full mesh
        full_mesh = self.generate_mesh()
        
        # Filter to selected components
        mesh = self._filter_components(full_mesh, components)
        
        # Combine all components into single PolyData mesh
        combined = pv.PolyData()
        for key in mesh.keys():
            combined = combined + mesh[key]
        
        # Save to file
        combined.save(filepath)
        print(f"Rocket model saved to: {filepath}")
    
    def get_mesh_info(self, components: Union[str, List[str]] = 'all') -> dict:
        """
        Get information about the generated mesh.
        
        Args:
            components: Components to get info for (same options as render)
        
        Returns:
            Dictionary with mesh statistics
        """
        # Get full mesh
        full_mesh = self.generate_mesh()
        
        # Filter to selected components
        mesh = self._filter_components(full_mesh, components)
        
        info = {
            'n_components': len(mesh),
            'components': list(mesh.keys()),
            'total_points': sum(mesh[key].n_points for key in mesh.keys()),
            'total_cells': sum(mesh[key].n_cells for key in mesh.keys()),
        }
        
        # Per-component stats
        for key in mesh.keys():
            info[f'{key}_points'] = mesh[key].n_points
            info[f'{key}_cells'] = mesh[key].n_cells
        
        return info
    
    def get_assembly_info(self) -> dict:
        """
        Get information about the rocket assembly.
        
        Returns:
            Dictionary with assembly information
        """
        geometry = self.extract_geometry()
        
        info = {
            'total_length': geometry.total_length,
            'body_radius': geometry.radius,
            'motor_position': geometry.motor_position,
            'nosecone_position': geometry.nosecone_position,
            'fins_position': geometry.fins_position,
            'tail_position': geometry.tail_position,
            'coordinate_system': self.coord_system,
            'flight_simulation_ready': (self.coord_system == 'tail_to_nose'),
            'components': {
                'motor': geometry.motor is not None,
                'nosecone': geometry.nosecone is not None,
                'fins': geometry.fins is not None,
                'tail': geometry.tail is not None,
            }
        }
        
        return info
