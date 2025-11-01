"""
Rocket 3D Viewer GUI Application

A dedicated cross-platform GUI for visualizing RocketPy rockets in 3D.
Built with PySide6 and PyVista for maximum compatibility.

Features:
- Interactive 3D rendering with PyVista
- Component selection for visualization
- Export models to STL/OBJ formats
- Mesh statistics display

Usage:
    from sim.gui import RocketViewerApp
    
    app = RocketViewerApp(rocket=my_rocket)
    app.run()
"""

import sys
import logging
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QCheckBox, QTextEdit, QSplitter,
    QFileDialog, QMessageBox, QFrame
)
from PySide6.QtCore import Qt
from pyvistaqt import QtInteractor
import pyvista as pv

from ..rendering.renderer import RocketRenderer

# Suppress VTK warnings that appear during PyVista shutdown
# These are harmless cleanup warnings from VTK internal objects
pv.set_error_output_file('NUL' if sys.platform == 'win32' else '/dev/null')
logging.getLogger('vtkmodules').setLevel(logging.CRITICAL)


class RocketViewerApp(QMainWindow):
    """
    Main GUI application for 3D rocket visualization.
    
    Provides an interactive interface for:
    - Selecting rocket components to render
    - Viewing 3D models in embedded PyVista viewer
    - Exporting models to file
    - Viewing mesh statistics
    """
    
    def __init__(self, rocket):
        """
        Initialize the Rocket Viewer application.
        
        Args:
            rocket: RocketPy Rocket object to load
        """
        super().__init__()
        
        self.rocket = rocket
        self.rocket_renderer = RocketRenderer(self.rocket)
        
        # Component selection state
        self.component_checkboxes = {}
        self.selected_components = set()
        
        # Setup UI
        self.init_ui()
        
        # Render initial view
        self.render_selected_components()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Rocket 3D Viewer')
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel: Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: 3D Viewer
        right_panel = self.create_viewer_panel()
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (30% controls, 70% viewer)
        splitter.setSizes([420, 980])
        
        # Status bar
        self.statusBar().showMessage('Ready')
    
    def create_control_panel(self) -> QWidget:
        """Create the left control panel with component selection."""
        panel = QWidget()
        panel.setMaximumWidth(500)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel('<h2>Rocket 3D Viewer</h2>')
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Object info section
        info_group = QGroupBox('Rocket Information')
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel(f'<b>Radius:</b> {self.rocket.radius:.4f} m'))
        info_layout.addWidget(QLabel(f'<b>Mass:</b> {self.rocket.mass:.2f} kg'))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Component selection
        component_group = self.create_component_selection()
        layout.addWidget(component_group)
        
        # Action buttons
        button_layout = QVBoxLayout()
        
        # Render button
        render_btn = QPushButton('🔄 Render Selected')
        render_btn.setMinimumHeight(40)
        render_btn.clicked.connect(self.on_render_clicked)
        button_layout.addWidget(render_btn)
        
        # Select All / None buttons
        select_layout = QHBoxLayout()
        select_all_btn = QPushButton('Select All')
        select_all_btn.clicked.connect(self.select_all_components)
        select_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton('Select None')
        select_none_btn.clicked.connect(self.select_no_components)
        select_layout.addWidget(select_none_btn)
        button_layout.addLayout(select_layout)
        
        # Export button
        export_btn = QPushButton('💾 Export to STL')
        export_btn.clicked.connect(self.on_export_clicked)
        button_layout.addWidget(export_btn)
        
        # Clear button
        clear_btn = QPushButton('🗑️ Clear Viewer')
        clear_btn.clicked.connect(self.clear_viewer)
        button_layout.addWidget(clear_btn)
        
        layout.addLayout(button_layout)
        
        # Mesh info display
        info_display_group = QGroupBox('Mesh Information')
        info_display_layout = QVBoxLayout()
        
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setMaximumHeight(200)
        self.info_display.setStyleSheet('font-family: monospace; font-size: 10pt;')
        info_display_layout.addWidget(self.info_display)
        info_display_group.setLayout(info_display_layout)
        layout.addWidget(info_display_group)
        
        # Spacer
        layout.addStretch()
        
        # Help text
        help_text = QLabel(
            '<small><b>Controls:</b><br>'
            '• Left click + drag: Rotate<br>'
            '• Right click + drag: Pan<br>'
            '• Scroll: Zoom<br>'
            '• R: Reset camera</small>'
        )
        help_text.setWordWrap(True)
        help_text.setFrameStyle(QFrame.Box | QFrame.Plain)
        help_text.setMargin(10)
        layout.addWidget(help_text)
        
        return panel
    
    def create_component_selection(self) -> QGroupBox:
        """Create component selection checkboxes for rocket."""
        group = QGroupBox('Select Components')
        layout = QVBoxLayout()
        
        # Get available components
        geometry = self.rocket_renderer.extract_geometry()
        
        # Add checkboxes for each component
        components = [
            ('Motor', 'motor', geometry.motor is not None),
            ('Nose Cone', 'nosecone', geometry.nosecone is not None),
            ('Body Tube', 'body', True),  # Always has body
            ('Fins', 'fins', geometry.fins is not None),
            ('Tail', 'tail', geometry.tail is not None),
        ]
        
        for display_name, component_id, available in components:
            if available:
                checkbox = QCheckBox(display_name)
                checkbox.setChecked(True)  # Default: all selected
                # Store component_id as property to avoid lambda closure issues
                checkbox.setProperty('component_id', component_id)
                checkbox.stateChanged.connect(self.on_component_checkbox_changed)
                self.component_checkboxes[component_id] = checkbox
                self.selected_components.add(component_id)
                layout.addWidget(checkbox)
        
        group.setLayout(layout)
        return group
    
    def create_viewer_panel(self) -> QWidget:
        """Create the right panel with 3D viewer."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create PyVista QtInteractor (embedded viewer)
        self.plotter = QtInteractor(panel)
        self.plotter.set_background('white')
        layout.addWidget(self.plotter.interactor)
        
        # Add axes and camera widget
        self.plotter.add_axes(
            xlabel='X (m)',
            ylabel='Y (m)',
            zlabel='Z (m)',
        )
        self.plotter.add_camera_orientation_widget()
        
        return panel
    
    def on_component_checkbox_changed(self, state: int):
        """Handle component checkbox toggle."""
        checkbox = self.sender()
        if checkbox is None:
            return
        
        component_id = checkbox.property('component_id')
        if component_id is None:
            return
        
        if checkbox.isChecked():
            self.selected_components.add(component_id)
        else:
            self.selected_components.discard(component_id)
    
    def select_all_components(self):
        """Select all available components."""
        for checkbox in self.component_checkboxes.values():
            checkbox.setChecked(True)
    
    def select_no_components(self):
        """Deselect all components."""
        for checkbox in self.component_checkboxes.values():
            checkbox.setChecked(False)
    
    def on_render_clicked(self):
        """Handle render button click."""
        self.render_selected_components()
    
    def render_selected_components(self):
        """Render selected rocket components in the viewer."""
        if not self.selected_components:
            QMessageBox.warning(self, 'No Selection', 'Please select at least one component to render.')
            return
        
        self.statusBar().showMessage('Rendering components...')
        
        try:
            # Clear previous render
            self.plotter.clear()
            
            # Re-add axes and camera widget
            self.plotter.add_axes(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
            self.plotter.add_camera_orientation_widget()
            
            # Get full mesh and filter to selected components
            full_mesh = self.rocket_renderer.generate_mesh()
            components_list = list(self.selected_components)
            filtered_mesh = self.rocket_renderer._filter_components(full_mesh, components_list)
            
            # Color scheme
            colors = {
                'motor_casing': 'silver',
                'motor_nozzle': 'dimgray',
                'motor_closure': 'silver',
                'nosecone': 'red',
                'body': 'white',
                'tail': 'lightgray',
            }
            fin_colors = ['blue', 'blue', 'blue', 'blue']
            
            # Add components to plotter
            for key in filtered_mesh.keys():
                if key.startswith('fin_'):
                    fin_idx = int(key.split('_')[1]) - 1
                    color = fin_colors[fin_idx % len(fin_colors)]
                    label = f'Fin {fin_idx + 1}'
                else:
                    color = colors.get(key, 'gray')
                    label = key.replace('_', ' ').title()
                
                self.plotter.add_mesh(
                    filtered_mesh[key],
                    color=color,
                    show_edges=True,
                    label=label,
                    smooth_shading=True
                )
            
            # Add legend
            self.plotter.add_legend(bcolor='white')
            
            # Reset camera
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            
            # Update info display
            info = self.rocket_renderer.get_mesh_info(components=components_list)
            self.update_info_display(info)
            
            self.statusBar().showMessage(f'Rendered {len(components_list)} component(s) successfully')
            
        except Exception as e:
            QMessageBox.critical(self, 'Render Error', f'Failed to render components:\n{str(e)}')
            self.statusBar().showMessage('Render failed')
    
    def clear_viewer(self):
        """Clear the 3D viewer."""
        self.plotter.clear()
        self.info_display.clear()
        self.statusBar().showMessage('Viewer cleared')
    
    def update_info_display(self, info: dict):
        """Update the mesh information display."""
        text = '<b>Mesh Statistics:</b><br><br>'
        text += f'Components: {info["n_components"]}<br>'
        text += f'Total Vertices: {info["total_points"]:,}<br>'
        text += f'Total Polygons: {info["total_cells"]:,}<br><br>'
        
        text += '<b>Component Breakdown:</b><br>'
        for comp in info['components']:
            points = info.get(f'{comp}_points', 0)
            cells = info.get(f'{comp}_cells', 0)
            text += f'  • {comp}: {points:,} vertices, {cells:,} polygons<br>'
        
        self.info_display.setHtml(text)
    
    def on_export_clicked(self):
        """Handle export button click."""
        if not self.rocket_renderer:
            QMessageBox.warning(self, 'No Object', 'No rocket loaded.')
            return
        
        # Open file dialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            'Export Model',
            '',
            'STL Files (*.stl);;OBJ Files (*.obj);;PLY Files (*.ply);;VTK Files (*.vtk)'
        )
        
        if not filepath:
            return
        
        try:
            self.statusBar().showMessage('Exporting model...')
            components_list = list(self.selected_components) if self.selected_components else 'all'
            self.rocket_renderer.save_model(filepath, components=components_list)
            QMessageBox.information(self, 'Export Successful', f'Model exported to:\n{filepath}')
            self.statusBar().showMessage('Export successful')
        except Exception as e:
            QMessageBox.critical(self, 'Export Error', f'Failed to export model:\n{str(e)}')
            self.statusBar().showMessage('Export failed')
    
    def run(self):
        """Show the application window."""
        self.show()
    
    def closeEvent(self, event):
        """Handle window close event with proper cleanup."""
        try:
            # Close PyVista plotter gracefully to prevent VTK errors
            if hasattr(self, 'plotter') and self.plotter is not None:
                self.plotter.close()
        except Exception:
            # Ignore any errors during cleanup
            pass
        
        # Accept the close event
        event.accept()


def launch_gui(rocket):
    """
    Launch the Rocket 3D Viewer application.
    
    Args:
        rocket: RocketPy Rocket object to visualize
    
    Returns:
        Application exit code
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    viewer = RocketViewerApp(rocket=rocket)
    viewer.show()
    
    return app.exec_()
