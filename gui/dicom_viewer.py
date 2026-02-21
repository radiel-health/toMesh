"""
DICOM / NIfTI slice viewer widget.

Embeds a matplotlib figure in Qt showing axial, coronal, and sagittal
slices of the loaded image volume. A slider scrubs through slices on
the active axis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DicomViewer:
    """PyQt6 widget containing a 3-plane slice viewer for CT volumes.

    Usage:
        viewer = DicomViewer()
        viewer.load(Path("scan.nii.gz"))
        layout.addWidget(viewer.widget)

    Attributes:
        widget: The embeddable QWidget.
    """

    def __init__(self) -> None:
        from PyQt6 import QtWidgets
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        self._volume: Optional[np.ndarray] = None  # (Z, Y, X) float32
        self._spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

        # ── root widget ───────────────────────────────────────────────
        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)

        # Placeholder label shown before any image is loaded
        self._placeholder = QtWidgets.QLabel(
            "No CT scan loaded.\nUse Step 1 to select a DICOM folder or NIfTI file."
        )
        self._placeholder.setStyleSheet("color: #888; font-size: 14px;")
        self._placeholder.setAlignment(
            __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignCenter
        )
        layout.addWidget(self._placeholder)

        # ── matplotlib canvas ─────────────────────────────────────────
        plt.style.use("dark_background")
        self._fig = plt.figure(figsize=(8, 3), tight_layout=True)
        gs = gridspec.GridSpec(1, 3, figure=self._fig)
        self._ax_axial    = self._fig.add_subplot(gs[0, 0])
        self._ax_coronal  = self._fig.add_subplot(gs[0, 1])
        self._ax_sagittal = self._fig.add_subplot(gs[0, 2])
        for ax in (self._ax_axial, self._ax_coronal, self._ax_sagittal):
            ax.axis("off")
        self._ax_axial.set_title("Axial",    color="white", fontsize=9)
        self._ax_coronal.set_title("Coronal", color="white", fontsize=9)
        self._ax_sagittal.set_title("Sagittal", color="white", fontsize=9)

        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.hide()  # hidden until image loaded
        layout.addWidget(self._canvas, stretch=1)

        # ── slice slider ──────────────────────────────────────────────
        slider_row = QtWidgets.QHBoxLayout()
        self._slider_label = QtWidgets.QLabel("Axial slice:")
        self._slider = QtWidgets.QSlider()
        self._slider.setOrientation(
            __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.Orientation.Horizontal
        )
        self._slice_index_label = QtWidgets.QLabel("0")
        self._slider.valueChanged.connect(self._on_slice_changed)

        slider_row.addWidget(self._slider_label)
        slider_row.addWidget(self._slider, stretch=1)
        slider_row.addWidget(self._slice_index_label)
        self._slider_widget = QtWidgets.QWidget()
        self._slider_widget.setLayout(slider_row)
        self._slider_widget.hide()
        layout.addWidget(self._slider_widget)

        # ── windowing controls ────────────────────────────────────────
        wl_row = QtWidgets.QHBoxLayout()
        wl_row.addWidget(QtWidgets.QLabel("Window level:"))
        self._wl_spin = QtWidgets.QSpinBox()
        self._wl_spin.setRange(-2000, 4000)
        self._wl_spin.setValue(40)
        self._wl_spin.setSuffix(" HU")
        wl_row.addWidget(self._wl_spin)
        wl_row.addWidget(QtWidgets.QLabel("Window width:"))
        self._ww_spin = QtWidgets.QSpinBox()
        self._ww_spin.setRange(1, 4000)
        self._ww_spin.setValue(400)
        self._ww_spin.setSuffix(" HU")
        wl_row.addWidget(self._ww_spin)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        wl_row.addWidget(refresh_btn)
        self._wl_widget = QtWidgets.QWidget()
        self._wl_widget.setLayout(wl_row)
        self._wl_widget.hide()
        layout.addWidget(self._wl_widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Load a NIfTI file or DICOM directory into the viewer.

        Args:
            path: Path to a .nii/.nii.gz file or a DICOM folder.

        Returns:
            Tuple of (volume array (Z,Y,X), spacing (sz,sy,sx) in mm).

        Raises:
            ImportError: If SimpleITK is not installed.
        """
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImportError(
                "SimpleITK is not installed: pip install SimpleITK"
            ) from exc

        if path.is_dir():
            # DICOM folder
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(str(path))
            if not series_ids:
                raise ValueError(f"No DICOM series found in {path}")
            files = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
            reader.SetFileNames(files)
            img = reader.Execute()
        else:
            img = sitk.ReadImage(str(path))

        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
        sp_xyz = img.GetSpacing()
        self._spacing = (float(sp_xyz[2]), float(sp_xyz[1]), float(sp_xyz[0]))
        self._volume = arr

        logger.info(
            "CT loaded: shape=%s, spacing(ZYX)=(%.2f,%.2f,%.2f)",
            arr.shape, *self._spacing
        )

        # Initialise slider to middle slice
        z_mid = arr.shape[0] // 2
        self._slider.setRange(0, arr.shape[0] - 1)
        self._slider.setValue(z_mid)

        self._placeholder.hide()
        self._canvas.show()
        self._slider_widget.show()
        self._wl_widget.show()

        self._draw(z_mid)
        return self._volume, self._spacing

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _windowed(self, slice_2d: np.ndarray) -> np.ndarray:
        """Apply window/level to a HU slice for display.

        Args:
            slice_2d: 2-D float32 array in Hounsfield units.

        Returns:
            Float array clipped to [0, 1] for imshow.
        """
        wl = float(self._wl_spin.value())
        ww = float(self._ww_spin.value())
        lo = wl - ww / 2
        hi = wl + ww / 2
        return np.clip((slice_2d - lo) / (hi - lo), 0.0, 1.0)

    def _draw(self, axial_idx: int) -> None:
        """Redraw all three planes at the given axial slice index.

        Args:
            axial_idx: Index along the Z axis (axial).
        """
        if self._volume is None:
            return

        vol = self._volume
        Z, Y, X = vol.shape
        axial_idx = np.clip(axial_idx, 0, Z - 1)
        y_mid = Y // 2
        x_mid = X // 2

        axial_slice    = self._windowed(vol[axial_idx, :, :])
        coronal_slice  = self._windowed(vol[:, y_mid, :])
        sagittal_slice = self._windowed(vol[:, :, x_mid])

        for ax, data, orig in [
            (self._ax_axial,    axial_slice,    "lower"),
            (self._ax_coronal,  coronal_slice,  "lower"),
            (self._ax_sagittal, sagittal_slice, "lower"),
        ]:
            ax.clear()
            ax.imshow(data, cmap="gray", origin=orig, aspect="equal")
            ax.axis("off")

        self._ax_axial.set_title("Axial",     color="white", fontsize=9)
        self._ax_coronal.set_title("Coronal",  color="white", fontsize=9)
        self._ax_sagittal.set_title("Sagittal", color="white", fontsize=9)

        # Mark current axial position on coronal and sagittal
        self._ax_coronal.axhline(y=axial_idx, color="yellow", linewidth=0.8, alpha=0.6)
        self._ax_sagittal.axhline(y=axial_idx, color="yellow", linewidth=0.8, alpha=0.6)

        self._canvas.draw_idle()

    def _on_slice_changed(self, value: int) -> None:
        self._slice_index_label.setText(str(value))
        self._draw(value)

    def _refresh(self) -> None:
        self._draw(self._slider.value())
