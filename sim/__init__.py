"""Simulation utilities for RocketPy visualization and analysis."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = ["launch_gui"]


def launch_gui(*args: Any, **kwargs: Any) -> Any:
    """Lazily import the GUI launcher to keep core imports lightweight."""

    from .gui.rocket_viewer_app import launch_gui as _launch_gui

    return _launch_gui(*args, **kwargs)
