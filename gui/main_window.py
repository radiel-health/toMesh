"""
Main application window for toMesh — Cardiovascular Mesh Editor.

Layout:
  ┌─────────────────────────────────────────────┐
  │  MenuBar + Toolbar (undo/redo/save)          │
  ├──────────┬──────────────────────────────────┤
  │ToolPanel │        PyVista Viewport           │
  │ (270px)  │                                   │
  ├──────────┴──────────────────────────────────┤
  │  StatusBar: vertices | faces | BC counts     │
  └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# BC label constants (mirrors bc_tagger.py)
WALL = 0
INLET = 1
OUTLET = 2


def _apply_dark_theme(app: object) -> None:
    """Apply QDarkStyle dark theme, falling back to a manual stylesheet.

    Args:
        app: QApplication instance.
    """
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6"))
        logger.debug("QDarkStyle applied")
    except Exception:
        # Fallback minimal dark palette
        app.setStyleSheet(
            "QWidget { background-color: #2b2b2b; color: #ddd; }"
            "QPushButton { background: #444; border: 1px solid #666; padding: 4px; }"
            "QPushButton:hover { background: #555; }"
            "QTabBar::tab { background: #333; padding: 6px 12px; }"
            "QTabBar::tab:selected { background: #4a4a4a; }"
        )
        logger.debug("Fallback dark stylesheet applied")


class MainWindow:
    """PyQt6 main window for the mesh editor.

    Coordinates between SessionState, ViewportWidget, and ToolPanel.

    Args:
        mesh_path: Path to the .vtp or .stl mesh to load on startup.
        output_dir: Directory where exported files will be saved.
        source_ct_path: Original CT scan path (stored in graph metadata).
    """

    def __init__(
        self,
        mesh_path: Path,
        output_dir: Optional[Path] = None,
        source_ct_path: Optional[Path] = None,
    ) -> None:
        from PyQt6 import QtWidgets, QtCore, QtGui
        from PyQt6.QtWidgets import (
            QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
            QStatusBar, QToolBar, QAction, QMessageBox, QFileDialog,
        )

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        _apply_dark_theme(self._app)

        self.output_dir = output_dir or mesh_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Load mesh & init session ---
        import pyvista as pv
        from .session import SessionState
        from .bc_tagger import ensure_bc_array

        mesh = pv.read(str(mesh_path))
        mesh = ensure_bc_array(mesh)
        self._session = SessionState(mesh, source_file=source_ct_path)
        self._session.on_mesh_changed = self._on_mesh_changed

        # --- Build main window ---
        self.window = QMainWindow()
        self.window.setWindowTitle("toMesh — Cardiovascular Mesh Editor")
        self.window.resize(1280, 800)

        central = QWidget()
        self.window.setCentralWidget(central)
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        # --- Viewport ---
        from .viewport import ViewportWidget
        self._viewport = ViewportWidget()
        h_layout.addWidget(self._viewport.qt_widget, stretch=1)

        # --- Tool panel ---
        from .toolbar import ToolPanel
        self._toolbar_panel = ToolPanel(
            on_smooth=self._on_smooth,
            on_clip_confirm=self._on_clip,
            on_decimate=self._on_decimate,
            on_sculpt_stroke=self._on_sculpt_activate,
            on_bc_mode_change=self._on_bc_mode_change,
            on_bc_paint=self._on_bc_paint_activate,
            on_bc_auto_detect=self._on_bc_auto_detect,
            on_export=self._on_export,
        )
        h_layout.insertWidget(0, self._toolbar_panel.widget)

        # --- Status bar ---
        self._status_bar = QStatusBar()
        self.window.setStatusBar(self._status_bar)

        # --- Menu bar ---
        self._build_menu_bar()

        # --- Internal state ---
        self._current_bc_mode: int = WALL
        self._sculpt_mode: str = "push"
        self._sculpt_radius: float = 5.0
        self._sculpt_strength: float = 0.5
        self._clip_plane: Optional[object] = None  # stores current clip plane normal+origin

        # --- Initial render ---
        self._viewport.load_mesh(self._session.mesh)
        self._update_status_bar()

    def _build_menu_bar(self) -> None:
        """Build File + Edit menu bar with keyboard shortcuts."""
        from PyQt6.QtWidgets import QMenuBar
        from PyQt6.QtGui import QAction, QKeySequence

        menubar = self.window.menuBar()

        # File
        file_menu = menubar.addMenu("File")
        save_action = QAction("Save Mesh (.vtp)", self.window)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save_mesh)
        file_menu.addAction(save_action)

        export_action = QAction("Export Graph (.pt)…", self.window)
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self.window)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self._app.quit)
        file_menu.addAction(quit_action)

        # Edit
        edit_menu = menubar.addMenu("Edit")
        undo_action = QAction("Undo", self.window)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self._on_undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self.window)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self._on_redo)
        edit_menu.addAction(redo_action)

    # ------------------------------------------------------------------
    # Smooth
    # ------------------------------------------------------------------

    def _on_smooth(self, iterations: int, factor: float, preview: bool = False) -> None:
        """Apply or preview Laplacian smoothing.

        Args:
            iterations: Number of smooth iterations.
            factor: Relaxation factor (0–1).
            preview: If True, does not push to undo stack.
        """
        if not preview:
            self._session.push_snapshot("smooth")

        smoothed = self._session.mesh.smooth(
            n_iter=iterations,
            relaxation_factor=factor,
            feature_smoothing=False,
            boundary_smoothing=True,
        )
        # Preserve BC labels
        if "bc_label" in self._session.mesh.point_data:
            smoothed.point_data["bc_label"] = self._session.mesh.point_data["bc_label"].copy()

        if preview:
            self._viewport.load_mesh(smoothed)
        else:
            self._session.mesh = smoothed
        logger.info("Smooth applied: %d iters, factor=%.2f", iterations, factor)

    # ------------------------------------------------------------------
    # Clip
    # ------------------------------------------------------------------

    def _on_clip(self, action: str, keep_above: Optional[bool]) -> None:
        """Handle clip widget lifecycle.

        Args:
            action: 'activate', 'confirm', or 'cancel'.
            keep_above: Which side to keep (for 'confirm' action).
        """
        if action == "activate":
            self._viewport.show_clip_widget(
                callback=self._on_clip_plane_moved,
                normal="z",
            )
            logger.debug("Clip widget activated")

        elif action == "confirm":
            self._viewport.remove_clip_widget()
            if self._clip_plane is not None:
                self._session.push_snapshot("clip")
                normal, origin = self._clip_plane
                clipped = self._session.mesh.clip(
                    normal=normal,
                    origin=origin,
                    invert=not keep_above,
                )
                if "bc_label" in self._session.mesh.point_data:
                    # After clipping, new vertices won't have bc_label; reindex
                    clipped = _transfer_bc_labels_after_clip(
                        self._session.mesh, clipped
                    )
                self._session.mesh = clipped
                logger.info("Clip confirmed (keep_above=%s)", keep_above)
            self._clip_plane = None

        elif action == "cancel":
            self._viewport.remove_clip_widget()
            self._viewport.load_mesh(self._session.mesh)
            self._clip_plane = None

    def _on_clip_plane_moved(self, normal: object, origin: object) -> None:
        """Store current clip plane position from widget callback.

        Args:
            normal: Plane normal vector.
            origin: Point on the plane.
        """
        self._clip_plane = (normal, origin)

    # ------------------------------------------------------------------
    # Decimate
    # ------------------------------------------------------------------

    def _on_decimate(self, target_faces: int) -> None:
        """Run pymeshlab decimation to target face count.

        Args:
            target_faces: Desired number of triangles after decimation.
        """
        import tempfile
        from pathlib import Path as _Path
        from meshing.cleanup import remesh_to_target

        n_before = self._session.mesh.n_faces
        self._session.push_snapshot("decimate")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_stl = _Path(tmpdir) / "before.stl"
            self._session.mesh.save(str(tmp_stl))
            result_path = remesh_to_target(tmp_stl, target_faces)

            import pyvista as pv
            new_mesh = pv.read(str(result_path))

        # Transfer BC labels (nearest-neighbour)
        if "bc_label" in self._session.mesh.point_data:
            from scipy.spatial import cKDTree
            old_pts = np.array(self._session.mesh.points)
            new_pts = np.array(new_mesh.points)
            _, idx = cKDTree(old_pts).query(new_pts, k=1)
            new_mesh.point_data["bc_label"] = (
                self._session.mesh.point_data["bc_label"][idx]
            )

        n_after = new_mesh.n_faces
        self._session.mesh = new_mesh
        self._toolbar_panel.update_decimate_info(n_before, n_after)
        logger.info("Decimate: %d → %d faces", n_before, n_after)

    # ------------------------------------------------------------------
    # Sculpt
    # ------------------------------------------------------------------

    def _on_sculpt_activate(
        self, mode: str, radius: float, strength: float
    ) -> None:
        """Activate sculpt point-picking mode.

        Args:
            mode: 'push', 'pull', or 'smooth'.
            radius: Brush radius.
            strength: Brush strength.
        """
        self._sculpt_mode = mode
        self._sculpt_radius = radius
        self._sculpt_strength = strength
        self._viewport.enable_point_picking(self._on_sculpt_click)
        logger.debug("Sculpt mode: %s, r=%.2f, s=%.2f", mode, radius, strength)

    def _on_sculpt_click(
        self, point: np.ndarray, normal: Optional[np.ndarray]
    ) -> None:
        """Apply one sculpt stroke at the picked point.

        Args:
            point: World-space 3D position of brush centre.
            normal: Surface normal at pick point (may be None).
        """
        from .sculpt import sculpt_vertices, SculptMode

        mode_map = {
            "push": SculptMode.PUSH,
            "pull": SculptMode.PULL,
            "smooth": SculptMode.SMOOTH,
        }
        mode = mode_map.get(self._sculpt_mode, SculptMode.PUSH)

        self._session.push_snapshot(f"sculpt_{self._sculpt_mode}")
        import copy
        mesh = copy.deepcopy(self._session.mesh)
        sculpt_vertices(
            mesh,
            brush_centre=np.asarray(point),
            brush_radius=self._sculpt_radius,
            brush_strength=self._sculpt_strength,
            mode=mode,
            normal=normal,
        )
        self._session.mesh = mesh

    # ------------------------------------------------------------------
    # BC Tagging
    # ------------------------------------------------------------------

    def _on_bc_mode_change(self, label: int) -> None:
        """Switch the active BC painting label.

        Args:
            label: 0=wall, 1=inlet, 2=outlet.
        """
        self._current_bc_mode = label
        names = {0: "WALL", 1: "INLET", 2: "OUTLET"}
        logger.debug("BC mode → %s", names.get(label, str(label)))

    def _on_bc_paint_activate(self, use_flood: bool) -> None:
        """Activate cell picking in BC painting mode.

        Args:
            use_flood: If True, use flood-fill on click; else single-face.
        """
        def _pick_callback(face_id: int) -> None:
            from .bc_tagger import paint_face, flood_fill_faces
            self._session.push_snapshot("bc_paint")
            import copy
            mesh = copy.deepcopy(self._session.mesh)
            if use_flood:
                angle = self._toolbar_panel.bc_flood_angle
                flood_fill_faces(mesh, face_id, self._current_bc_mode, angle)
            else:
                paint_face(mesh, face_id, self._current_bc_mode)
            self._session.mesh = mesh

        self._viewport.enable_cell_picking(_pick_callback, flood_fill=use_flood)
        logger.debug("BC paint activated (flood=%s)", use_flood)

    def _on_bc_auto_detect(self) -> None:
        """Run the auto-detect BC heuristic."""
        from .bc_tagger import auto_detect_bc
        import copy
        self._session.push_snapshot("bc_auto_detect")
        mesh = copy.deepcopy(self._session.mesh)
        auto_detect_bc(mesh)
        self._session.mesh = mesh
        logger.info("Auto-detect BC complete")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        """Validate, export graph, and save mesh with BC labels."""
        from PyQt6.QtWidgets import QMessageBox

        from export.validators import validate_mesh_for_export
        from export.to_graph import mesh_to_pyg

        mesh = self._session.mesh

        # Run validators
        errors, warnings = validate_mesh_for_export(mesh)
        if warnings:
            for w in warnings:
                logger.warning("Export warning: %s", w)
        if errors:
            QMessageBox.critical(
                self.window,
                "Export Failed",
                "Validation errors:\n" + "\n".join(f"• {e}" for e in errors),
            )
            return

        # Export graph
        graph_path = self.output_dir / "graph.pt"
        vtp_path = self.output_dir / "mesh_with_bcs.vtp"

        try:
            import torch
            source_str = str(self._session.source_file) if self._session.source_file else ""
            data = mesh_to_pyg(mesh, source_file=source_str)
            torch.save(data, str(graph_path))

            mesh.save(str(vtp_path))

            QMessageBox.information(
                self.window,
                "Export Successful",
                f"Graph saved:\n  {graph_path}\n\nMesh saved:\n  {vtp_path}",
            )
            logger.info("Export complete: %s, %s", graph_path, vtp_path)
        except Exception as exc:
            QMessageBox.critical(self.window, "Export Error", str(exc))
            logger.exception("Export failed")

    # ------------------------------------------------------------------
    # Save mesh
    # ------------------------------------------------------------------

    def _on_save_mesh(self) -> None:
        """Save current mesh (with BC labels) as .vtp."""
        vtp_path = self.output_dir / "mesh_edited.vtp"
        self._session.mesh.save(str(vtp_path))
        self._status_bar.showMessage(f"Saved: {vtp_path}", 3000)
        logger.info("Mesh saved: %s", vtp_path)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def _on_undo(self) -> None:
        if not self._session.undo():
            self._status_bar.showMessage("Nothing to undo", 1500)

    def _on_redo(self) -> None:
        if not self._session.redo():
            self._status_bar.showMessage("Nothing to redo", 1500)

    # ------------------------------------------------------------------
    # Mesh change callback
    # ------------------------------------------------------------------

    def _on_mesh_changed(self) -> None:
        """Called by SessionState whenever the mesh is replaced."""
        self._viewport.load_mesh(self._session.mesh)
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        """Refresh the bottom status bar with current mesh stats."""
        mesh = self._session.mesh
        counts = self._session.bc_counts()
        self._toolbar_panel.update_bc_counts(
            counts["wall"], counts["inlet"], counts["outlet"]
        )
        self._status_bar.showMessage(
            f"Vertices: {mesh.n_points:,}  |  Faces: {mesh.n_faces:,}  |  "
            f"Wall: {counts['wall']:,}  Inlet: {counts['inlet']:,}  "
            f"Outlet: {counts['outlet']:,}"
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def show_and_run(self) -> int:
        """Show the window and start the Qt event loop.

        Returns:
            Qt application exit code.
        """
        self.window.show()
        return self._app.exec()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transfer_bc_labels_after_clip(
    original_mesh: object,
    clipped_mesh: object,
) -> object:
    """Transfer BC labels to a clipped mesh via nearest-neighbour mapping.

    Args:
        original_mesh: Pre-clip PyVista PolyData with bc_label.
        clipped_mesh: Post-clip PyVista PolyData.

    Returns:
        clipped_mesh with bc_label assigned.
    """
    from scipy.spatial import cKDTree

    old_pts = np.array(original_mesh.points)
    new_pts = np.array(clipped_mesh.points)
    old_labels = np.array(original_mesh.point_data["bc_label"])

    _, idx = cKDTree(old_pts).query(new_pts, k=1)
    clipped_mesh.point_data["bc_label"] = old_labels[idx]
    return clipped_mesh


# ---------------------------------------------------------------------------
# Module-level launcher
# ---------------------------------------------------------------------------

def launch_gui(
    mesh_path: Path,
    output_dir: Optional[Path] = None,
    source_ct_path: Optional[Path] = None,
) -> int:
    """Launch the mesh editor GUI.

    Args:
        mesh_path: Path to the mesh file to load.
        output_dir: Directory for exported outputs.
        source_ct_path: Original CT scan path (for graph metadata).

    Returns:
        Exit code from the Qt event loop.
    """
    try:
        from PyQt6 import QtWidgets
    except ImportError as exc:
        raise ImportError(
            "PyQt6 is not installed. Install it with: pip install PyQt6"
        ) from exc

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    win = MainWindow(
        mesh_path=mesh_path,
        output_dir=output_dir,
        source_ct_path=source_ct_path,
    )
    return win.show_and_run()
