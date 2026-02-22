"""Smoke test for wizard GUI."""
import sys
from pathlib import Path
from PyQt6 import QtWidgets

app = QtWidgets.QApplication(sys.argv)

from gui.main_window import MainWindow

win = MainWindow()
print("tabs:", win._tabs.count())

win2 = MainWindow(mesh_path=Path("test_sphere.vtp"))
print("mesh verts:", win2._mesh.n_points, "tab:", win2._tabs.currentIndex())

from gui.dicom_viewer import DicomViewer
DicomViewer()
print("DicomViewer: OK")
print("ALL PASSED")
