"""
Left-side tool panel with tabs for each editing mode.

Panels:
  1. Smooth
  2. Cut / Clip
  3. Decimate / Remesh
  4. Sculpt / Deform
  5. Boundary Condition Tagger
  6. Export (bottom)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _require_pyqt6() -> None:
    try:
        from PyQt6 import QtWidgets  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyQt6 is not installed. Install it with: pip install PyQt6"
        ) from exc


class ToolPanel:
    """Container widget holding all tool tabs + export button.

    Rather than inheriting QWidget directly, we build the panel in build()
    so callers can inject the Qt app before widget creation.

    Args:
        on_smooth: Callback(iterations, factor) called when Apply is clicked.
        on_clip_confirm: Callback(keep_above) after clip confirmation.
        on_decimate: Callback(target_faces) after decimation.
        on_sculpt_stroke: Callback(mode_str, radius, strength) on each brush use.
        on_bc_mode_change: Callback(label_int) when BC mode button pressed.
        on_bc_paint: Callback(flood_fill: bool) on BC paint action.
        on_bc_auto_detect: Callback() for auto-detect button.
        on_export: Callback() for export button.
    """

    def __init__(
        self,
        on_smooth: Callable,
        on_clip_confirm: Callable,
        on_decimate: Callable,
        on_sculpt_stroke: Callable,
        on_bc_mode_change: Callable,
        on_bc_paint: Callable,
        on_bc_auto_detect: Callable,
        on_export: Callable,
    ) -> None:
        _require_pyqt6()
        from PyQt6 import QtWidgets, QtCore, QtGui

        self._callbacks = {
            "smooth": on_smooth,
            "clip": on_clip_confirm,
            "decimate": on_decimate,
            "sculpt": on_sculpt_stroke,
            "bc_mode": on_bc_mode_change,
            "bc_paint": on_bc_paint,
            "bc_auto": on_bc_auto_detect,
            "export": on_export,
        }

        # -- Root widget -------------------------------------------------------
        self.widget = QtWidgets.QWidget()
        self.widget.setFixedWidth(270)
        root_layout = QtWidgets.QVBoxLayout(self.widget)
        root_layout.setContentsMargins(4, 4, 4, 4)

        # -- Tab widget --------------------------------------------------------
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        root_layout.addWidget(self.tabs)

        self._build_smooth_tab()
        self._build_clip_tab()
        self._build_decimate_tab()
        self._build_sculpt_tab()
        self._build_bc_tab()

        # -- Export button (bottom) -------------------------------------------
        self.export_btn = QtWidgets.QPushButton("Export Graph (.pt)")
        self.export_btn.setStyleSheet(
            "QPushButton { background: #2a7; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background: #3b8; }"
        )
        self.export_btn.clicked.connect(on_export)
        root_layout.addWidget(self.export_btn)

    # ------------------------------------------------------------------
    # Smooth panel
    # ------------------------------------------------------------------

    def _build_smooth_tab(self) -> None:
        from PyQt6 import QtWidgets
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("Iterations"))
        self.smooth_iter_slider = QtWidgets.QSlider()
        self.smooth_iter_slider.setOrientation(
            __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.Orientation.Horizontal
        )
        self.smooth_iter_slider.setRange(1, 50)
        self.smooth_iter_slider.setValue(5)
        self.smooth_iter_label = QtWidgets.QLabel("5")
        self.smooth_iter_slider.valueChanged.connect(
            lambda v: self.smooth_iter_label.setText(str(v))
        )
        layout.addWidget(self.smooth_iter_slider)
        layout.addWidget(self.smooth_iter_label)

        layout.addWidget(QtWidgets.QLabel("Smoothing Factor"))
        self.smooth_factor_slider = QtWidgets.QSlider()
        self.smooth_factor_slider.setOrientation(
            __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.Orientation.Horizontal
        )
        self.smooth_factor_slider.setRange(0, 100)
        self.smooth_factor_slider.setValue(50)
        self.smooth_factor_label = QtWidgets.QLabel("0.50")
        self.smooth_factor_slider.valueChanged.connect(
            lambda v: self.smooth_factor_label.setText(f"{v / 100:.2f}")
        )
        layout.addWidget(self.smooth_factor_slider)
        layout.addWidget(self.smooth_factor_label)

        row = QtWidgets.QHBoxLayout()
        preview_btn = QtWidgets.QPushButton("Preview")
        apply_btn = QtWidgets.QPushButton("Apply")
        preview_btn.clicked.connect(self._on_smooth_preview)
        apply_btn.clicked.connect(self._on_smooth_apply)
        row.addWidget(preview_btn)
        row.addWidget(apply_btn)
        layout.addLayout(row)
        layout.addStretch()
        self.tabs.addTab(w, "Smooth")

    def _on_smooth_preview(self) -> None:
        iters = self.smooth_iter_slider.value()
        factor = self.smooth_factor_slider.value() / 100.0
        logger.debug("Smooth preview: iters=%d, factor=%.2f", iters, factor)
        # Preview does not push undo snapshot
        self._callbacks["smooth"](iters, factor, preview=True)

    def _on_smooth_apply(self) -> None:
        iters = self.smooth_iter_slider.value()
        factor = self.smooth_factor_slider.value() / 100.0
        self._callbacks["smooth"](iters, factor, preview=False)

    # ------------------------------------------------------------------
    # Clip panel
    # ------------------------------------------------------------------

    def _build_clip_tab(self) -> None:
        from PyQt6 import QtWidgets
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel(
            "Drag the clipping plane widget in the viewport.\n"
            "Select which side to keep, then Confirm."
        ))

        self.clip_keep_above = QtWidgets.QCheckBox("Keep above plane")
        self.clip_keep_above.setChecked(True)
        layout.addWidget(self.clip_keep_above)

        activate_btn = QtWidgets.QPushButton("Activate Clip Widget")
        activate_btn.clicked.connect(self._on_clip_activate)
        layout.addWidget(activate_btn)

        row = QtWidgets.QHBoxLayout()
        confirm_btn = QtWidgets.QPushButton("Confirm")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        confirm_btn.clicked.connect(self._on_clip_confirm)
        cancel_btn.clicked.connect(self._on_clip_cancel)
        row.addWidget(confirm_btn)
        row.addWidget(cancel_btn)
        layout.addLayout(row)

        self._clip_active = False
        layout.addStretch()
        self.tabs.addTab(w, "Clip")

    def _on_clip_activate(self) -> None:
        self._clip_active = True
        # Signal main window to show clip widget
        self._callbacks["clip"]("activate", None)

    def _on_clip_confirm(self) -> None:
        keep = self.clip_keep_above.isChecked()
        self._callbacks["clip"]("confirm", keep)
        self._clip_active = False

    def _on_clip_cancel(self) -> None:
        self._callbacks["clip"]("cancel", None)
        self._clip_active = False

    # ------------------------------------------------------------------
    # Decimate panel
    # ------------------------------------------------------------------

    def _build_decimate_tab(self) -> None:
        from PyQt6 import QtWidgets
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("Target face count:"))
        self.decimate_target = QtWidgets.QSpinBox()
        self.decimate_target.setRange(500, 500_000)
        self.decimate_target.setValue(15_000)
        self.decimate_target.setSingleStep(1000)
        layout.addWidget(self.decimate_target)

        self.decimate_info = QtWidgets.QLabel("Before: — faces\nAfter: — faces")
        layout.addWidget(self.decimate_info)

        apply_btn = QtWidgets.QPushButton("Decimate")
        apply_btn.clicked.connect(self._on_decimate)
        layout.addWidget(apply_btn)
        layout.addStretch()
        self.tabs.addTab(w, "Decimate")

    def _on_decimate(self) -> None:
        target = self.decimate_target.value()
        self._callbacks["decimate"](target)

    def update_decimate_info(self, before: int, after: int) -> None:
        """Update before/after face count labels.

        Args:
            before: Face count before decimation.
            after: Face count after decimation.
        """
        self.decimate_info.setText(f"Before: {before:,} faces\nAfter: {after:,} faces")

    # ------------------------------------------------------------------
    # Sculpt panel
    # ------------------------------------------------------------------

    def _build_sculpt_tab(self) -> None:
        from PyQt6 import QtWidgets
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("Mode:"))
        self.sculpt_mode_combo = QtWidgets.QComboBox()
        self.sculpt_mode_combo.addItems(["Push", "Pull", "Smooth"])
        layout.addWidget(self.sculpt_mode_combo)

        layout.addWidget(QtWidgets.QLabel("Brush Radius"))
        self.sculpt_radius_slider = QtWidgets.QSlider()
        self.sculpt_radius_slider.setOrientation(
            __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.Orientation.Horizontal
        )
        self.sculpt_radius_slider.setRange(1, 100)
        self.sculpt_radius_slider.setValue(20)
        self.sculpt_radius_label = QtWidgets.QLabel("5.0")
        self.sculpt_radius_slider.valueChanged.connect(
            lambda v: self.sculpt_radius_label.setText(f"{v / 4:.1f}")
        )
        layout.addWidget(self.sculpt_radius_slider)
        layout.addWidget(self.sculpt_radius_label)

        layout.addWidget(QtWidgets.QLabel("Brush Strength"))
        self.sculpt_strength_slider = QtWidgets.QSlider()
        self.sculpt_strength_slider.setOrientation(
            __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.Orientation.Horizontal
        )
        self.sculpt_strength_slider.setRange(1, 100)
        self.sculpt_strength_slider.setValue(50)
        self.sculpt_strength_label = QtWidgets.QLabel("0.50")
        self.sculpt_strength_slider.valueChanged.connect(
            lambda v: self.sculpt_strength_label.setText(f"{v / 100:.2f}")
        )
        layout.addWidget(self.sculpt_strength_slider)
        layout.addWidget(self.sculpt_strength_label)

        activate_btn = QtWidgets.QPushButton("Activate Sculpt Mode")
        activate_btn.clicked.connect(self._on_sculpt_activate)
        layout.addWidget(activate_btn)
        layout.addStretch()
        self.tabs.addTab(w, "Sculpt")

    def _on_sculpt_activate(self) -> None:
        mode = self.sculpt_mode_combo.currentText().lower()
        radius = self.sculpt_radius_slider.value() / 4.0
        strength = self.sculpt_strength_slider.value() / 100.0
        self._callbacks["sculpt"](mode, radius, strength)

    # ------------------------------------------------------------------
    # BC Tagger panel
    # ------------------------------------------------------------------

    def _build_bc_tab(self) -> None:
        from PyQt6 import QtWidgets
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("BC Mode:"))
        btn_row = QtWidgets.QHBoxLayout()
        self.bc_wall_btn = QtWidgets.QPushButton("Wall")
        self.bc_inlet_btn = QtWidgets.QPushButton("Inlet")
        self.bc_outlet_btn = QtWidgets.QPushButton("Outlet")

        self.bc_wall_btn.setStyleSheet("background: #555; color: white;")
        self.bc_inlet_btn.setStyleSheet("background: #339; color: white;")
        self.bc_outlet_btn.setStyleSheet("background: #933; color: white;")

        self.bc_wall_btn.clicked.connect(lambda: self._callbacks["bc_mode"](0))
        self.bc_inlet_btn.clicked.connect(lambda: self._callbacks["bc_mode"](1))
        self.bc_outlet_btn.clicked.connect(lambda: self._callbacks["bc_mode"](2))

        btn_row.addWidget(self.bc_wall_btn)
        btn_row.addWidget(self.bc_inlet_btn)
        btn_row.addWidget(self.bc_outlet_btn)
        layout.addLayout(btn_row)

        layout.addWidget(QtWidgets.QLabel("Selection method:"))
        self.bc_single_radio = QtWidgets.QRadioButton("Single face")
        self.bc_flood_radio = QtWidgets.QRadioButton("Flood fill")
        self.bc_single_radio.setChecked(True)
        layout.addWidget(self.bc_single_radio)
        layout.addWidget(self.bc_flood_radio)

        layout.addWidget(QtWidgets.QLabel("Flood fill angle threshold (°):"))
        self.bc_angle_spin = QtWidgets.QDoubleSpinBox()
        self.bc_angle_spin.setRange(1.0, 90.0)
        self.bc_angle_spin.setValue(30.0)
        self.bc_angle_spin.setSingleStep(5.0)
        layout.addWidget(self.bc_angle_spin)

        paint_btn = QtWidgets.QPushButton("Activate Painting")
        paint_btn.clicked.connect(self._on_bc_paint)
        layout.addWidget(paint_btn)

        auto_btn = QtWidgets.QPushButton("Auto-detect Inlets/Outlets")
        auto_btn.clicked.connect(self._callbacks["bc_auto"])
        layout.addWidget(auto_btn)

        # BC counts display
        self.bc_counts_label = QtWidgets.QLabel(
            "Wall: 0\nInlet: 0\nOutlet: 0"
        )
        self.bc_counts_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.bc_counts_label)

        layout.addStretch()
        self.tabs.addTab(w, "BC Tag")

    def _on_bc_paint(self) -> None:
        use_flood = self.bc_flood_radio.isChecked()
        self._callbacks["bc_paint"](use_flood)

    def update_bc_counts(self, wall: int, inlet: int, outlet: int) -> None:
        """Update the BC count display labels.

        Args:
            wall: Number of wall-tagged vertices.
            inlet: Number of inlet-tagged vertices.
            outlet: Number of outlet-tagged vertices.
        """
        self.bc_counts_label.setText(
            f"Wall:   {wall:>7,}\nInlet:  {inlet:>7,}\nOutlet: {outlet:>7,}"
        )

    @property
    def bc_flood_angle(self) -> float:
        """Current flood fill angle threshold in degrees."""
        return float(self.bc_angle_spin.value())
