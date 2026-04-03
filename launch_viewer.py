"""
Rocket 3D Viewer Launcher

Launch the Rocket 3D Viewer GUI application with the Itzamna rocket.

This launcher uses the shared rocket configuration from the notebook
to ensure the 3D visualization matches the simulated rocket exactly.
"""

import os
import sys


def configure_qt_runtime() -> None:
    """
    Configure Qt before importing GUI modules.

    This avoids mixed Qt bindings and reduces Wayland/X11 interop crashes
    (for example: BadWindow / X_ConfigureWindow on startup).
    """
    # pyvistaqt uses qtpy; keep binding aligned with this app's PySide6 imports.
    os.environ.setdefault("QT_API", "pyside6")

    if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
        # Optional app-level override (kept separate from raw Qt env).
        forced_backend = os.environ.get("ROCKET_VIEWER_QPA_PLATFORM", "").strip()
        if forced_backend:
            os.environ["QT_QPA_PLATFORM"] = forced_backend
        else:
            has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
            has_x11 = bool(os.environ.get("DISPLAY"))

            # Prefer X11/XWayland when both are present.
            # pyvistaqt + VTK can crash on Wayland with:
            # "BadWindow (X_ConfigureWindow)" during startup.
            if has_x11:
                os.environ["QT_QPA_PLATFORM"] = "xcb"
            elif has_wayland:
                os.environ["QT_QPA_PLATFORM"] = "wayland"

    # On some Mesa/X11 stacks this is more stable for VTK/Qt OpenGL widgets.
    if os.environ.get("QT_QPA_PLATFORM", "").startswith("xcb"):
        os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")


def main() -> int:
    configure_qt_runtime()

    repo_root = os.path.dirname(__file__)
    rocket_config_path = os.path.join(repo_root, "notebooks", "Itzamna", "rocket_config.py")

    # Add notebooks/Itzamna to path to import rocket_config
    sys.path.insert(0, os.path.join(repo_root, "notebooks", "Itzamna"))

    from rocket_config import (
        create_rocket,
    )
    from sim import launch_gui

    print("=" * 60)
    print("Rocket 3D Viewer - Itzamna Rocket")
    print("=" * 60)
    print()

    platform = os.environ.get("QT_QPA_PLATFORM", "auto")
    qt_api = os.environ.get("QT_API", "auto")
    print(f"Qt runtime: QT_API={qt_api}, QT_QPA_PLATFORM={platform}")
    print()

    # ==================== CREATE ROCKET FROM SHARED CONFIG ====================
    print("Creating Itzamna rocket from notebook configuration...")
    print("This ensures the 3D viewer shows the exact rocket from the simulation.")
    print()

    # Create rocket using shared configuration (without parachutes for viewer)
    rocket = create_rocket(include_parachutes=False, drag_data_path="data")

    print("✓ Rocket assembled from shared configuration")
    print(f"  - Dry mass: {rocket.mass:.3f} kg")
    print(f"  - Radius: {rocket.radius:.4f} m")
    print(f"  - Surfaces: {len(rocket.aerodynamic_surfaces)}")
    print()

    print("Launching Rocket 3D Viewer...")
    print("  - Static Mode: View rocket geometry")
    print("  - 3D Simulation Mode: Replay and visualize flight kinematics")
    print()

    # ==================== LAUNCH GUI ====================
    # Launch the viewer with rocket geometry; simulation mode consumes replay kinematics.
    return launch_gui(
        rocket,
        geometry_source_name="Itzamna rocket_config.py",
        geometry_source_path=rocket_config_path,
    )


if __name__ == "__main__":
    sys.exit(main())
