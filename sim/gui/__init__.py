"""GUI module for Rocket 3D Viewer application."""

from __future__ import annotations

from typing import Any

__all__ = ["RocketViewerApp", "launch_gui"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .rocket_viewer_app import RocketViewerApp, launch_gui

    exports = {
        "RocketViewerApp": RocketViewerApp,
        "launch_gui": launch_gui,
    }
    return exports[name]
