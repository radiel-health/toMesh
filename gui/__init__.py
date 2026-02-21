"""
GUI package for toMesh — Cardiovascular Mesh Editor.

Launch as a module:
    python -m gui /path/to/mesh.vtp

Or import and call directly:
    from gui.main_window import launch_gui
    launch_gui(mesh_path)
"""

from .main_window import launch_gui

__all__ = ["launch_gui"]
