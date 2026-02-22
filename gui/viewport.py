"""
PyVista BackgroundPlotter embed for the Qt main window.

The ViewportWidget wraps pyvistaqt.BackgroundPlotter and exposes a clean
API for the rest of the GUI to:
  - Load / replace the displayed mesh
  - Refresh BC colouring
  - Enable / disable picking modes (cell pick, point pick)
  - Display a clip-plane widget
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _require_pyvistaqt() -> None:
    try:
        import pyvistaqt  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pyvistaqt is not installed. Install it with: pip install pyvistaqt"
        ) from exc


class ViewportWidget:
    """Thin wrapper around pyvistaqt.BackgroundPlotter.

    Provides the plotter as a Qt widget via ``self.plotter.interactor``.

    Args:
        parent: Parent Qt widget.
    """

    def __init__(self, parent: Optional[object] = None) -> None:
        _require_pyvistaqt()
        from pyvistaqt import BackgroundPlotter

        self.plotter = BackgroundPlotter(
            show=False,
            title="toMesh Viewport",
            window_size=(800, 600),
        )
        self.plotter.set_background("black")
        self._mesh_actor = None
        self._pick_callback: Optional[Callable] = None

    @property
    def qt_widget(self) -> object:
        """Return the embeddable Qt widget from the plotter."""
        return self.plotter.interactor

    def load_mesh(self, mesh: object, show_edges: bool = False) -> None:
        """Replace the currently displayed mesh.

        Args:
            mesh: PyVista PolyData to display.
            show_edges: Whether to show mesh wireframe edges.
        """
        self.plotter.clear()
        self._mesh_actor = self.plotter.add_mesh(
            mesh,
            scalars="bc_label",
            cmap=["grey", "blue", "red"],
            clim=[0, 2],
            show_edges=show_edges,
            name="mesh",
        )
        self.plotter.reset_camera()
        logger.debug("Viewport mesh reloaded")

    def refresh_bc_colors(self, mesh: object) -> None:
        """Update the colour scalar array without re-adding the mesh.

        Args:
            mesh: Current PyVista PolyData with updated bc_label.
        """
        self.plotter.update_scalars("bc_label", mesh=mesh)
        self.plotter.render()

    def enable_cell_picking(
        self,
        callback: Callable[[int], None],
        flood_fill: bool = False,
    ) -> None:
        """Enable single-cell or flood-fill picking mode.

        Args:
            callback: Function called with the picked face ID.
            flood_fill: Not used at plotter level; passed through to callback.
        """
        self._pick_callback = callback
        self.plotter.enable_cell_picking(
            callback=lambda picked: self._on_cell_picked(picked),
            show_message=False,
            style="wireframe",
            through=False,
        )
        logger.debug("Cell picking enabled (flood_fill=%s)", flood_fill)

    def _on_cell_picked(self, picked: object) -> None:
        """Internal callback bridging PyVista cell pick to our callback.

        Args:
            picked: PyVista PolyData of the single picked cell.
        """
        if self._pick_callback is None:
            return
        try:
            # picked is a single-cell PolyData; get its original cell ID from
            # the 'vtkOriginalCellIds' field array if available
            ids = picked.cell_data.get("vtkOriginalCellIds")
            if ids is not None and len(ids) > 0:
                face_id = int(ids[0])
            else:
                face_id = 0
            self._pick_callback(face_id)
        except Exception as exc:
            logger.warning("Cell pick callback error: %s", exc)

    def enable_point_picking(self, callback: Callable) -> None:
        """Enable point picking (used by sculpt tool).

        Args:
            callback: Function called with (point_position, normal).
        """
        self.plotter.enable_point_picking(
            callback=lambda point: callback(point, None),
            show_message=False,
            use_picker=True,
        )

    def disable_picking(self) -> None:
        """Disable all picking interactions."""
        try:
            self.plotter.disable_picking()
        except Exception:
            pass  # may not exist in all pyvista versions

    def show_clip_widget(self, callback: Callable, normal: str = "z") -> None:
        """Show an interactive clipping plane widget.

        Args:
            callback: Called whenever the plane moves.
            normal: Initial plane normal direction ('x', 'y', or 'z').
        """
        self.plotter.add_plane_widget(callback=callback, normal=normal)

    def remove_clip_widget(self) -> None:
        """Remove the clipping plane widget."""
        try:
            self.plotter.clear_plane_widgets()
        except Exception:
            pass

    def close(self) -> None:
        """Clean up the plotter."""
        try:
            self.plotter.close()
        except Exception:
            pass
