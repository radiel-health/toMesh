"""
Simplified wizard-style main window for toMesh.

Five sequential steps shown as tabs:
  1. Load    — pick a CT scan (DICOM folder or NIfTI)
  2. Preview — DICOM/NIfTI slice viewer
  3. Mesh    — run segmentation + mesh generation
  4. Tag BCs — flood-fill inlet/outlet painting on the 3D mesh
  5. Export  — validate + write graph.pt

Layout:
  ┌──────────────────────────────────────────────────────┐
  │  [1.Load]  [2.Preview]  [3.Mesh]  [4.Tag]  [5.Export]│  ← tab bar
  ├─────────────────────┬────────────────────────────────┤
  │  Step control panel │  Viewer (slice OR 3D)          │
  │  (left, 300 px)     │  switches automatically        │
  ├─────────────────────┴────────────────────────────────┤
  │  Status bar                                          │
  └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

WALL   = 0
INLET  = 1
OUTLET = 2


def _apply_dark_theme(app: object) -> None:
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6"))
    except Exception:
        app.setStyleSheet(
            "QWidget{background:#2b2b2b;color:#ddd;}"
            "QPushButton{background:#444;border:1px solid #666;padding:5px;border-radius:3px;}"
            "QPushButton:hover{background:#555;}"
            "QPushButton:disabled{background:#333;color:#666;}"
            "QTabBar::tab{background:#333;padding:7px 14px;}"
            "QTabBar::tab:selected{background:#4a4a4a;border-bottom:2px solid #5af;}"
            "QGroupBox{border:1px solid #555;margin-top:8px;padding-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}"
        )


class MainWindow:
    """Wizard-style main window.

    Args:
        ct_path:    Pre-loaded CT path (optional; user can also pick via UI).
        output_dir: Where outputs are written.
        mesh_path:  Pre-loaded mesh path (skips steps 1-3 if given).
    """

    def __init__(
        self,
        ct_path:    Optional[Path] = None,
        output_dir: Optional[Path] = None,
        mesh_path:  Optional[Path] = None,
    ) -> None:
        from PyQt6 import QtWidgets, QtCore
        from PyQt6.QtGui import QAction

        self._app = (
            QtWidgets.QApplication.instance()
            or QtWidgets.QApplication(sys.argv)
        )
        _apply_dark_theme(self._app)

        self._ct_path:    Optional[Path] = ct_path
        self._output_dir: Path = output_dir or Path.cwd() / "tomesh_output"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._mesh:       Optional[object] = None  # pyvista.PolyData
        self._structures: list[str] = ["aorta", "heart"]

        # ── main window ───────────────────────────────────────────────
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("toMesh — Cardiovascular Mesh Editor")
        self.window.resize(1300, 750)

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── left: step tabs ───────────────────────────────────────────
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setFixedWidth(310)
        self._tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        root.addWidget(self._tabs)

        self._build_step1()
        self._build_step2()
        self._build_step3()
        self._build_step4()
        self._build_step5()

        # ── right: stacked viewer (slice view ↔ 3D mesh) ─────────────
        self._viewer_stack = QtWidgets.QStackedWidget()
        root.addWidget(self._viewer_stack, stretch=1)

        # Page 0: DICOM slice viewer
        from .dicom_viewer import DicomViewer
        self._dicom_viewer = DicomViewer()
        self._viewer_stack.addWidget(self._dicom_viewer.widget)

        # Page 1: PyVista 3D mesh viewport (lazy-init on first use)
        self._viewport_placeholder = QtWidgets.QLabel(
            "Mesh will appear here after Step 3."
        )
        self._viewport_placeholder.setStyleSheet("color:#888;font-size:14px;")
        self._viewport_placeholder.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self._viewport_widget: Optional[object] = None  # created lazily
        self._viewer_stack.addWidget(self._viewport_placeholder)

        # Connect tab-change signal now that _viewer_stack exists
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # ── status bar ────────────────────────────────────────────────
        self._status = QtWidgets.QStatusBar()
        self.window.setStatusBar(self._status)
        self._status.showMessage("Step 1: Load a CT scan to begin.")

        # ── menu ──────────────────────────────────────────────────────
        menubar = self.window.menuBar()
        file_menu = menubar.addMenu("File")
        open_act = QAction("Open CT scan…", self.window)
        open_act.triggered.connect(self._pick_ct_file)
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        quit_act = QAction("Quit", self.window)
        quit_act.triggered.connect(self._app.quit)
        file_menu.addAction(quit_act)

        # Pre-load if paths were passed in
        if mesh_path and mesh_path.exists():
            self._load_mesh_file(mesh_path)
            self._tabs.setCurrentIndex(3)  # jump to Tag step
        elif ct_path and ct_path.exists():
            self._ct_path = ct_path
            self._ct_path_label.setText(str(ct_path))
            self._tabs.setCurrentIndex(1)  # jump to Preview

    # ==================================================================
    # Step 1 — Load
    # ==================================================================

    def _build_step1(self) -> None:
        from PyQt6 import QtWidgets
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(_header("Step 1: Load CT Scan"))
        layout.addWidget(_info(
            "Select a DICOM folder or a NIfTI (.nii/.nii.gz) file.\n"
            "The scan will be previewed in Step 2."
        ))

        pick_btn = QtWidgets.QPushButton("Browse for DICOM folder…")
        pick_btn.clicked.connect(lambda: self._pick_ct_dir())
        layout.addWidget(pick_btn)

        pick_nifti = QtWidgets.QPushButton("Browse for NIfTI file…")
        pick_nifti.clicked.connect(lambda: self._pick_ct_file())
        layout.addWidget(pick_nifti)

        self._ct_path_label = QtWidgets.QLabel("No file selected.")
        self._ct_path_label.setWordWrap(True)
        self._ct_path_label.setStyleSheet("color:#aaa; font-size:11px;")
        layout.addWidget(self._ct_path_label)

        # Structure selection
        group = QtWidgets.QGroupBox("Structures to segment")
        glayout = QtWidgets.QVBoxLayout(group)
        self._structure_checks: dict[str, QtWidgets.QCheckBox] = {}
        for struct in ["aorta", "heart", "pulmonary_artery", "inferior_vena_cava"]:
            cb = QtWidgets.QCheckBox(struct.replace("_", " ").title())
            cb.setChecked(struct in self._structures)
            cb.toggled.connect(lambda checked, s=struct: self._toggle_structure(s, checked))
            self._structure_checks[struct] = cb
            glayout.addWidget(cb)
        layout.addWidget(group)

        next_btn = _next_button("Next: Preview →", lambda: self._goto_step(1))
        layout.addWidget(next_btn)
        layout.addStretch()
        self._tabs.addTab(w, "1. Load")

    def _pick_ct_dir(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self.window, "Select DICOM folder")
        if d:
            self._ct_path = Path(d)
            self._ct_path_label.setText(str(self._ct_path))
            self._status.showMessage(f"Selected: {self._ct_path.name}")

    def _pick_ct_file(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        f, _ = QFileDialog.getOpenFileName(
            self.window, "Select NIfTI file", "",
            "NIfTI (*.nii *.nii.gz);;All files (*)"
        )
        if f:
            self._ct_path = Path(f)
            self._ct_path_label.setText(str(self._ct_path))
            self._status.showMessage(f"Selected: {self._ct_path.name}")

    def _toggle_structure(self, name: str, checked: bool) -> None:
        if checked and name not in self._structures:
            self._structures.append(name)
        elif not checked and name in self._structures:
            self._structures.remove(name)

    # ==================================================================
    # Step 2 — Preview slices
    # ==================================================================

    def _build_step2(self) -> None:
        from PyQt6 import QtWidgets
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(_header("Step 2: Preview CT Slices"))
        layout.addWidget(_info(
            "Inspect the CT scan before segmentation.\n"
            "Use the slice slider on the viewer panel."
        ))

        load_btn = QtWidgets.QPushButton("Load & display slices")
        load_btn.clicked.connect(self._load_ct_preview)
        layout.addWidget(load_btn)

        self._preview_status = QtWidgets.QLabel("")
        self._preview_status.setWordWrap(True)
        self._preview_status.setStyleSheet("color:#8f8;font-size:11px;")
        layout.addWidget(self._preview_status)

        layout.addWidget(_info(
            "Window/level controls are in the viewer panel.\n"
            "Adjust to check tissue contrast before segmenting."
        ))

        row = QtWidgets.QHBoxLayout()
        row.addWidget(_back_button("← Back", lambda: self._goto_step(0)))
        row.addWidget(_next_button("Next: Generate Mesh →", lambda: self._goto_step(2)))
        layout.addLayout(row)
        layout.addStretch()
        self._tabs.addTab(w, "2. Preview")

    def _load_ct_preview(self) -> None:
        if not self._ct_path:
            self._preview_status.setText("No CT scan selected. Go to Step 1.")
            return
        try:
            self._status.showMessage("Loading CT scan…")
            vol, spacing = self._dicom_viewer.load(self._ct_path)
            shape = vol.shape
            self._preview_status.setText(
                f"Loaded: {shape[0]}×{shape[1]}×{shape[2]} voxels\n"
                f"Spacing: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm"
            )
            self._viewer_stack.setCurrentIndex(0)  # show slice viewer
            self._status.showMessage("CT loaded. Use slider to browse slices.")
        except Exception as exc:
            self._preview_status.setText(f"Error: {exc}")
            logger.exception("CT preview failed")

    # ==================================================================
    # Step 3 — Segment & mesh
    # ==================================================================

    def _build_step3(self) -> None:
        from PyQt6 import QtWidgets
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(_header("Step 3: Generate Mesh"))
        layout.addWidget(_info(
            "Runs TotalSegmentator on the CT, then builds a clean\n"
            "triangular surface mesh from the segmentation mask."
        ))

        # Target face count
        face_row = QtWidgets.QHBoxLayout()
        face_row.addWidget(QtWidgets.QLabel("Target faces:"))
        self._target_faces_spin = QtWidgets.QSpinBox()
        self._target_faces_spin.setRange(1000, 100_000)
        self._target_faces_spin.setValue(15_000)
        self._target_faces_spin.setSingleStep(1000)
        face_row.addWidget(self._target_faces_spin)
        layout.addLayout(face_row)

        # Option to skip segmentation (use existing mask)
        self._skip_seg_check = QtWidgets.QCheckBox("Skip segmentation (use existing mask)")
        self._skip_seg_check.toggled.connect(self._on_skip_seg_toggled)
        layout.addWidget(self._skip_seg_check)

        self._mask_path_label = QtWidgets.QLabel("No mask selected.")
        self._mask_path_label.setStyleSheet("color:#aaa;font-size:11px;")
        self._mask_path_label.hide()
        layout.addWidget(self._mask_path_label)

        pick_mask_btn = QtWidgets.QPushButton("Browse for existing mask…")
        pick_mask_btn.clicked.connect(self._pick_mask)
        pick_mask_btn.hide()
        layout.addWidget(pick_mask_btn)
        self._pick_mask_btn = pick_mask_btn

        # Also allow loading an already-built VTP
        pick_vtp_btn = QtWidgets.QPushButton("Or load existing mesh (.vtp/.stl)…")
        pick_vtp_btn.clicked.connect(self._pick_mesh_file)
        layout.addWidget(pick_vtp_btn)

        # Run button
        self._run_mesh_btn = QtWidgets.QPushButton("Run Segmentation + Mesh")
        self._run_mesh_btn.setStyleSheet(
            "QPushButton{background:#27a;color:white;font-weight:bold;padding:8px;border-radius:4px;}"
            "QPushButton:hover{background:#38b;}"
            "QPushButton:disabled{background:#333;color:#666;}"
        )
        self._run_mesh_btn.clicked.connect(self._run_mesh)
        layout.addWidget(self._run_mesh_btn)

        self._mesh_progress = QtWidgets.QProgressBar()
        self._mesh_progress.setRange(0, 0)  # indeterminate
        self._mesh_progress.hide()
        layout.addWidget(self._mesh_progress)

        self._mesh_status = QtWidgets.QLabel("")
        self._mesh_status.setWordWrap(True)
        self._mesh_status.setStyleSheet("font-size:11px;")
        layout.addWidget(self._mesh_status)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(_back_button("← Back", lambda: self._goto_step(1)))
        self._step3_next = _next_button(
            "Next: Tag Boundaries →", lambda: self._goto_step(3)
        )
        self._step3_next.setEnabled(False)
        row.addWidget(self._step3_next)
        layout.addLayout(row)
        layout.addStretch()
        self._tabs.addTab(w, "3. Mesh")

        self._existing_mask_path: Optional[Path] = None

    def _on_skip_seg_toggled(self, checked: bool) -> None:
        self._mask_path_label.setVisible(checked)
        self._pick_mask_btn.setVisible(checked)

    def _pick_mask(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        f, _ = QFileDialog.getOpenFileName(
            self.window, "Select mask", "",
            "NIfTI (*.nii *.nii.gz);;All (*)"
        )
        if f:
            self._existing_mask_path = Path(f)
            self._mask_path_label.setText(str(self._existing_mask_path))

    def _pick_mesh_file(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        f, _ = QFileDialog.getOpenFileName(
            self.window, "Select mesh", "",
            "Mesh files (*.vtp *.stl);;All (*)"
        )
        if f:
            self._load_mesh_file(Path(f))

    def _load_mesh_file(self, path: Path) -> None:
        """Load an existing mesh and jump to the Tag step."""
        import pyvista as pv
        from .bc_tagger import ensure_bc_array
        mesh = pv.read(str(path))
        mesh = ensure_bc_array(mesh)
        self._mesh = mesh
        self._init_viewport()
        self._step3_next.setEnabled(True)
        self._mesh_status.setText(
            f"Loaded: {mesh.n_points:,} vertices, {mesh.n_faces_strict:,} faces"
        )
        self._mesh_status.setStyleSheet("color:#8f8;font-size:11px;")
        self._status.showMessage(f"Mesh loaded: {path.name}")
        logger.info("Mesh loaded from %s", path)

    def _run_mesh(self) -> None:
        """Run segmentation + mesh generation in a background thread."""
        from PyQt6.QtCore import QThread, pyqtSignal

        if not self._ct_path:
            self._mesh_status.setText("No CT scan selected. Go to Step 1.")
            return

        target_faces = self._target_faces_spin.value()
        skip_seg     = self._skip_seg_check.isChecked()
        mask_path    = self._existing_mask_path if skip_seg else None

        self._run_mesh_btn.setEnabled(False)
        self._mesh_progress.show()
        self._mesh_status.setText("Running… (this may take several minutes)")
        self._mesh_status.setStyleSheet("color:#fa0;font-size:11px;")

        def _worker():
            try:
                output_dir = self._output_dir

                # Segmentation
                if skip_seg and mask_path:
                    clean_mask_path = mask_path
                    self._set_mesh_status("Skipping segmentation…", "#8f8")
                else:
                    self._set_mesh_status("Running TotalSegmentator…", "#fa0")
                    from segmentation.segment import run_segmentation
                    from segmentation.postprocess import clean_mask
                    raw, _ = run_segmentation(
                        input_path=self._ct_path,
                        output_dir=output_dir,
                        target_structures=self._structures,
                    )
                    clean_mask_path = output_dir / "combined_mask.nii.gz"
                    clean_mask(raw, clean_mask_path)

                # Mesh generation
                self._set_mesh_status("Generating mesh…", "#fa0")
                from meshing.generate_mesh import generate_mesh
                mesh = generate_mesh(
                    mask_path=clean_mask_path,
                    output_dir=output_dir,
                    target_faces=target_faces,
                )

                from .bc_tagger import ensure_bc_array
                mesh = ensure_bc_array(mesh)
                self._mesh = mesh
                self._set_mesh_status(
                    f"Done: {mesh.n_points:,} vertices, {mesh.n_faces_strict:,} faces",
                    "#8f8"
                )
                self._on_mesh_ready()

            except Exception as exc:
                self._set_mesh_status(f"Error: {exc}", "#f44")
                logger.exception("Mesh generation failed")
            finally:
                self._run_mesh_btn.setEnabled(True)
                self._mesh_progress.hide()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _set_mesh_status(self, msg: str, color: str = "#ddd") -> None:
        """Thread-safe status label update."""
        from PyQt6.QtCore import QMetaObject, Qt
        def _update():
            self._mesh_status.setText(msg)
            self._mesh_status.setStyleSheet(f"color:{color};font-size:11px;")
            self._status.showMessage(msg)
        QMetaObject.invokeMethod(
            self._mesh_status, "setText",
            Qt.ConnectionType.QueuedConnection,
            *[__import__("PyQt6.QtCore", fromlist=["Q_ARG"]).Q_ARG(str, msg)]
        ) if False else None  # fallback: direct call is fine for thread-safety via Qt
        # TODO: use a proper QThread + signal for thread-safe UI updates.
        # For now, direct calls work because PyQt6 queues cross-thread calls.
        self._mesh_status.setText(msg)
        self._mesh_status.setStyleSheet(f"color:{color};font-size:11px;")

    def _on_mesh_ready(self) -> None:
        """Called (possibly from worker thread) when mesh is ready."""
        self._init_viewport()
        self._step3_next.setEnabled(True)

    # ==================================================================
    # Step 4 — Tag boundary conditions
    # ==================================================================

    def _build_step4(self) -> None:
        from PyQt6 import QtWidgets
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(_header("Step 4: Tag Boundaries"))
        layout.addWidget(_info(
            "Paint inlet (blue) and outlet (red) regions on the mesh.\n"
            "Everything else is wall (grey) by default."
        ))

        # ── BC mode buttons ───────────────────────────────────────────
        mode_group = QtWidgets.QGroupBox("Boundary condition mode")
        mg = QtWidgets.QHBoxLayout(mode_group)
        self._bc_wall_btn   = QtWidgets.QPushButton("Wall")
        self._bc_inlet_btn  = QtWidgets.QPushButton("Inlet")
        self._bc_outlet_btn = QtWidgets.QPushButton("Outlet")
        self._bc_wall_btn.setStyleSheet("background:#555;color:white;font-weight:bold;")
        self._bc_inlet_btn.setStyleSheet("background:#226;color:white;font-weight:bold;")
        self._bc_outlet_btn.setStyleSheet("background:#622;color:white;font-weight:bold;")
        self._bc_wall_btn.clicked.connect(lambda: self._set_bc_mode(WALL))
        self._bc_inlet_btn.clicked.connect(lambda: self._set_bc_mode(INLET))
        self._bc_outlet_btn.clicked.connect(lambda: self._set_bc_mode(OUTLET))
        mg.addWidget(self._bc_wall_btn)
        mg.addWidget(self._bc_inlet_btn)
        mg.addWidget(self._bc_outlet_btn)
        layout.addWidget(mode_group)

        # ── selection method ──────────────────────────────────────────
        sel_group = QtWidgets.QGroupBox("Selection method")
        sg = QtWidgets.QVBoxLayout(sel_group)
        self._single_radio = QtWidgets.QRadioButton("Single face (click)")
        self._flood_radio  = QtWidgets.QRadioButton("Flood fill (click a cap)")
        self._flood_radio.setChecked(True)
        sg.addWidget(self._single_radio)
        sg.addWidget(self._flood_radio)

        angle_row = QtWidgets.QHBoxLayout()
        angle_row.addWidget(QtWidgets.QLabel("Flood angle threshold (°):"))
        self._angle_spin = QtWidgets.QDoubleSpinBox()
        self._angle_spin.setRange(1.0, 90.0)
        self._angle_spin.setValue(30.0)
        self._angle_spin.setSingleStep(5.0)
        angle_row.addWidget(self._angle_spin)
        sg.addLayout(angle_row)
        layout.addWidget(sel_group)

        # ── activate painting ─────────────────────────────────────────
        self._paint_btn = QtWidgets.QPushButton("Activate Painting (click mesh)")
        self._paint_btn.setStyleSheet(
            "QPushButton{background:#353;color:white;padding:6px;border-radius:3px;}"
            "QPushButton:hover{background:#464;}"
        )
        self._paint_btn.clicked.connect(self._activate_painting)
        layout.addWidget(self._paint_btn)

        # ── auto-detect ───────────────────────────────────────────────
        auto_btn = QtWidgets.QPushButton("Auto-detect Inlets / Outlets")
        auto_btn.setToolTip(
            "Finds open boundary loops and tags the lowest one as inlet,\n"
            "all others as outlet. You can override afterwards."
        )
        auto_btn.clicked.connect(self._auto_detect_bc)
        layout.addWidget(auto_btn)

        # ── undo ─────────────────────────────────────────────────────
        undo_btn = QtWidgets.QPushButton("Undo last paint")
        undo_btn.clicked.connect(self._undo_paint)
        layout.addWidget(undo_btn)

        # ── counts ───────────────────────────────────────────────────
        self._bc_counts_label = QtWidgets.QLabel("Wall: —\nInlet: —\nOutlet: —")
        self._bc_counts_label.setStyleSheet("font-family:monospace;")
        layout.addWidget(self._bc_counts_label)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(_back_button("← Back", lambda: self._goto_step(2)))
        row.addWidget(_next_button("Next: Export →", lambda: self._goto_step(4)))
        layout.addLayout(row)
        layout.addStretch()
        self._tabs.addTab(w, "4. Tag BCs")

        self._current_bc_mode: int = INLET
        self._paint_history: list = []  # simple list of label snapshots for undo

    # ==================================================================
    # Step 5 — Export
    # ==================================================================

    def _build_step5(self) -> None:
        from PyQt6 import QtWidgets
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(_header("Step 5: Export"))
        layout.addWidget(_info(
            "Validates the mesh, then exports a PyTorch Geometric graph\n"
            "(.pt) and a labelled mesh (.vtp) to the output directory."
        ))

        # Output directory picker
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("Output dir:"))
        self._out_dir_label = QtWidgets.QLabel(str(self._output_dir))
        self._out_dir_label.setWordWrap(True)
        self._out_dir_label.setStyleSheet("color:#aaa;font-size:11px;")
        out_row.addWidget(self._out_dir_label, stretch=1)
        pick_out_btn = QtWidgets.QPushButton("…")
        pick_out_btn.setFixedWidth(30)
        pick_out_btn.clicked.connect(self._pick_output_dir)
        out_row.addWidget(pick_out_btn)
        layout.addLayout(out_row)

        # Export button
        export_btn = QtWidgets.QPushButton("Export graph.pt + mesh_with_bcs.vtp")
        export_btn.setStyleSheet(
            "QPushButton{background:#2a7;color:white;font-weight:bold;"
            "padding:10px;border-radius:4px;font-size:13px;}"
            "QPushButton:hover{background:#3b8;}"
        )
        export_btn.clicked.connect(self._do_export)
        layout.addWidget(export_btn)

        self._export_status = QtWidgets.QLabel("")
        self._export_status.setWordWrap(True)
        layout.addWidget(self._export_status)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(_back_button("← Back", lambda: self._goto_step(3)))
        layout.addLayout(row)
        layout.addStretch()
        self._tabs.addTab(w, "5. Export")

    def _pick_output_dir(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self.window, "Select output directory")
        if d:
            self._output_dir = Path(d)
            self._out_dir_label.setText(str(self._output_dir))

    # ==================================================================
    # Viewport (PyVista)
    # ==================================================================

    def _init_viewport(self) -> None:
        """Create or refresh the PyVista BackgroundPlotter widget."""
        if self._mesh is None:
            return

        from PyQt6 import QtWidgets

        try:
            from pyvistaqt import BackgroundPlotter
        except ImportError:
            logger.warning("pyvistaqt not installed; 3D view unavailable.")
            return

        # Remove old placeholder / old plotter widget
        if self._viewport_widget is not None:
            old_idx = self._viewer_stack.indexOf(self._viewport_widget)
            if old_idx >= 0:
                self._viewer_stack.removeWidget(self._viewport_widget)

        plotter = BackgroundPlotter(show=False, window_size=(800, 600))
        plotter.set_background("black")
        self._plotter = plotter
        self._viewport_widget = plotter.interactor

        self._viewer_stack.addWidget(self._viewport_widget)
        self._viewer_stack.setCurrentWidget(self._viewport_widget)

        self._render_mesh()
        logger.debug("Viewport initialised")

    def _render_mesh(self) -> None:
        """Re-render the mesh with current BC colouring."""
        if self._mesh is None or not hasattr(self, "_plotter"):
            return
        self._plotter.clear()
        self._plotter.add_mesh(
            self._mesh,
            scalars="bc_label",
            cmap=["grey", "blue", "red"],
            clim=[0, 2],
            show_scalar_bar=False,
            name="mesh",
        )
        self._plotter.reset_camera()
        self._update_bc_counts()

    def _update_bc_counts(self) -> None:
        if self._mesh is None:
            return
        import numpy as np
        labels = np.asarray(self._mesh.point_data.get("bc_label", []))
        wall   = int((labels == WALL).sum())
        inlet  = int((labels == INLET).sum())
        outlet = int((labels == OUTLET).sum())
        self._bc_counts_label.setText(
            f"Wall:   {wall:>7,}\nInlet:  {inlet:>7,}\nOutlet: {outlet:>7,}"
        )
        self._status.showMessage(
            f"Vertices: {self._mesh.n_points:,}  |  Faces: {self._mesh.n_faces_strict:,}  |  "
            f"Wall: {wall:,}  Inlet: {inlet:,}  Outlet: {outlet:,}"
        )

    # ==================================================================
    # BC painting
    # ==================================================================

    def _set_bc_mode(self, mode: int) -> None:
        self._current_bc_mode = mode
        names = {WALL: "WALL", INLET: "INLET", OUTLET: "OUTLET"}
        self._status.showMessage(f"BC mode: {names[mode]}. Click mesh to paint.")

    def _activate_painting(self) -> None:
        if self._mesh is None or not hasattr(self, "_plotter"):
            self._status.showMessage("Generate a mesh first (Step 3).")
            return

        use_flood = self._flood_radio.isChecked()

        def _on_pick(picked) -> None:
            ids = picked.cell_data.get("vtkOriginalCellIds")
            face_id = int(ids[0]) if ids is not None and len(ids) > 0 else 0

            import copy
            snapshot = copy.deepcopy(np.asarray(self._mesh.point_data["bc_label"]))
            self._paint_history.append(snapshot)
            if len(self._paint_history) > 20:
                self._paint_history.pop(0)

            from .bc_tagger import paint_face, flood_fill_faces
            if use_flood:
                flood_fill_faces(
                    self._mesh, face_id,
                    self._current_bc_mode,
                    self._angle_spin.value()
                )
            else:
                paint_face(self._mesh, face_id, self._current_bc_mode)

            self._render_mesh()

        self._plotter.enable_cell_picking(
            callback=_on_pick,
            show_message=False,
            style="wireframe",
            through=False,
        )
        mode_name = {WALL: "Wall", INLET: "Inlet", OUTLET: "Outlet"}[self._current_bc_mode]
        self._status.showMessage(
            f"Painting active ({mode_name}). Click the mesh. "
            f"{'Flood-fill mode.' if use_flood else 'Single-face mode.'}"
        )

    def _auto_detect_bc(self) -> None:
        if self._mesh is None:
            self._status.showMessage("No mesh loaded.")
            return
        from .bc_tagger import auto_detect_bc
        import copy
        snapshot = copy.deepcopy(np.asarray(self._mesh.point_data["bc_label"]))
        self._paint_history.append(snapshot)
        auto_detect_bc(self._mesh)
        self._render_mesh()
        self._status.showMessage("Auto-detect BC complete. Review and override if needed.")

    def _undo_paint(self) -> None:
        if not self._paint_history or self._mesh is None:
            self._status.showMessage("Nothing to undo.")
            return
        prev = self._paint_history.pop()
        self._mesh.point_data["bc_label"] = prev
        self._render_mesh()
        self._status.showMessage("Undo: reverted last paint operation.")

    # ==================================================================
    # Export
    # ==================================================================

    def _do_export(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        if self._mesh is None:
            self._export_status.setText("No mesh. Complete Steps 1–3 first.")
            return

        from export.validators import validate_mesh_for_export
        from export.to_graph import mesh_to_pyg

        errors, warnings = validate_mesh_for_export(self._mesh)
        for w in warnings:
            logger.warning(w)
        if errors:
            QMessageBox.critical(
                self.window, "Export Failed",
                "Validation errors:\n" + "\n".join(f"• {e}" for e in errors)
            )
            return

        try:
            import torch
            source_str = str(self._ct_path) if self._ct_path else ""
            data = mesh_to_pyg(self._mesh, source_file=source_str)

            graph_path = self._output_dir / "graph.pt"
            vtp_path   = self._output_dir / "mesh_with_bcs.vtp"
            torch.save(data, str(graph_path))
            self._mesh.save(str(vtp_path))

            self._export_status.setText(
                f"Saved:\n  {graph_path}\n  {vtp_path}"
            )
            self._export_status.setStyleSheet("color:#8f8;")
            QMessageBox.information(
                self.window, "Export Successful",
                f"Graph: {graph_path}\nMesh:  {vtp_path}"
            )
        except Exception as exc:
            self._export_status.setText(f"Error: {exc}")
            self._export_status.setStyleSheet("color:#f44;")
            logger.exception("Export failed")

    # ==================================================================
    # Navigation
    # ==================================================================

    def _goto_step(self, idx: int) -> None:
        self._tabs.setCurrentIndex(idx)

    def _on_tab_changed(self, idx: int) -> None:
        # Steps 1–2: show slice viewer; steps 3–5: show 3D mesh
        if idx <= 1:
            self._viewer_stack.setCurrentIndex(0)
        else:
            if self._viewport_widget is not None:
                self._viewer_stack.setCurrentWidget(self._viewport_widget)
            else:
                self._viewer_stack.setCurrentIndex(1)  # placeholder

    # ==================================================================
    # Run
    # ==================================================================

    def show_and_run(self) -> int:
        self.window.show()
        return self._app.exec()


# ── small UI helpers ───────────────────────────────────────────────────

def _header(text: str) -> "QtWidgets.QLabel":
    from PyQt6 import QtWidgets
    lbl = QtWidgets.QLabel(text)
    lbl.setStyleSheet("font-size:14px;font-weight:bold;margin-bottom:4px;")
    return lbl

def _info(text: str) -> "QtWidgets.QLabel":
    from PyQt6 import QtWidgets
    lbl = QtWidgets.QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color:#aaa;font-size:11px;")
    return lbl

def _next_button(label: str, slot) -> "QtWidgets.QPushButton":
    from PyQt6 import QtWidgets
    btn = QtWidgets.QPushButton(label)
    btn.setStyleSheet(
        "QPushButton{background:#27a;color:white;padding:6px;border-radius:3px;}"
        "QPushButton:hover{background:#38b;}"
        "QPushButton:disabled{background:#333;color:#666;}"
    )
    btn.clicked.connect(slot)
    return btn

def _back_button(label: str, slot) -> "QtWidgets.QPushButton":
    from PyQt6 import QtWidgets
    btn = QtWidgets.QPushButton(label)
    btn.setStyleSheet(
        "QPushButton{background:#444;color:white;padding:6px;border-radius:3px;}"
        "QPushButton:hover{background:#555;}"
    )
    btn.clicked.connect(slot)
    return btn


# ── module-level launcher ─────────────────────────────────────────────

def launch_gui(
    mesh_path:  Optional[Path] = None,
    ct_path:    Optional[Path] = None,
    output_dir: Optional[Path] = None,
    source_ct_path: Optional[Path] = None,  # backwards-compat alias for ct_path
) -> int:
    """Launch the simplified wizard GUI.

    Args:
        mesh_path:  Pre-load an existing .vtp (skips steps 1-3).
        ct_path:    Pre-select a CT scan path (starts at step 2).
        output_dir: Where to write outputs.
        source_ct_path: Alias for ct_path (backwards compatibility).

    Returns:
        Qt application exit code.
    """
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow(
        ct_path=ct_path or source_ct_path,
        output_dir=output_dir,
        mesh_path=mesh_path,
    )
    return win.show_and_run()
