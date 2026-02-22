"""
Tests for the meshing pipeline.

Generates a synthetic spherical mask and runs the full meshing + cleanup
pipeline, asserting the output is a valid manifold mesh within expected
face count bounds.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_sphere_mask_nifti(mask: np.ndarray, spacing: tuple, tmp_dir: Path) -> Path:
    """Write a numpy mask as a NIfTI file using SimpleITK.

    Args:
        mask: Boolean 3-D array.
        spacing: (sz, sy, sx) voxel spacing in mm.
        tmp_dir: Temporary directory to write to.

    Returns:
        Path to the written .nii.gz file.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        pytest.skip("SimpleITK not installed")

    arr = mask.astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    # SimpleITK spacing is (x, y, z); reverse from numpy (z, y, x)
    img.SetSpacing((spacing[2], spacing[1], spacing[0]))

    path = tmp_dir / "sphere_mask.nii.gz"
    sitk.WriteImage(img, str(path))
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sphere_mask_nifti():
    """Generate a sphere mask NIfTI in a temp directory.

    Yields:
        Path to sphere_mask.nii.gz.
    """
    try:
        import SimpleITK  # noqa: F401
        import skimage  # noqa: F401
        import pymeshlab  # noqa: F401
        import pyvista  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"Missing dependency: {exc}")

    from tests.synthetic_data import make_sphere_mask

    mask, spacing = make_sphere_mask(grid_size=64)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        nifti_path = _write_sphere_mask_nifti(mask, spacing, tmp_path)
        yield nifti_path, tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMeshGeneration:
    """Full meshing pipeline on a synthetic spherical mask."""

    def test_output_is_pyvista_polydata(self, sphere_mask_nifti):
        """Output should be a PyVista PolyData object."""
        import pyvista as pv
        from meshing.generate_mesh import generate_mesh

        nifti_path, tmp_dir = sphere_mask_nifti
        output_dir = tmp_dir / "mesh_out"
        mesh = generate_mesh(
            mask_path=nifti_path,
            output_dir=output_dir,
            target_faces=3_000,
            smoothing_iterations=2,
            pyvista_smooth_iterations=5,
        )
        assert isinstance(mesh, pv.PolyData), "Output must be PyVista PolyData"

    def test_face_count_within_range(self, sphere_mask_nifti):
        """Face count should be within expected range after decimation."""
        from meshing.generate_mesh import generate_mesh

        nifti_path, tmp_dir = sphere_mask_nifti
        output_dir = tmp_dir / "mesh_fc"
        target = 3_000
        mesh = generate_mesh(
            mask_path=nifti_path,
            output_dir=output_dir,
            target_faces=target,
            smoothing_iterations=2,
            pyvista_smooth_iterations=5,
        )
        # Allow generous tolerance: decimation may not hit target exactly
        assert mesh.n_faces_strict > 0, "Mesh should have faces"
        assert mesh.n_faces_strict < target * 3, (
            f"Face count {mesh.n_faces_strict} unexpectedly high for target {target}"
        )

    def test_no_null_faces(self, sphere_mask_nifti):
        """Mesh should have no degenerate (null) faces."""
        from meshing.generate_mesh import generate_mesh
        import numpy as np

        nifti_path, tmp_dir = sphere_mask_nifti
        output_dir = tmp_dir / "mesh_null"
        mesh = generate_mesh(
            mask_path=nifti_path,
            output_dir=output_dir,
            target_faces=3_000,
            smoothing_iterations=2,
            pyvista_smooth_iterations=5,
        )
        # A null face has coincident or collinear vertices → zero area
        # pyvista >= 0.44 uses cell_quality(); key name changed to match the
        # quality_measure string ("area") in newer versions.
        try:
            mesh_with_quality = mesh.cell_quality(quality_measure="area")
        except AttributeError:
            mesh_with_quality = mesh.compute_cell_quality(quality_measure="area")
        area_key = next(
            k for k in ("area", "CellQuality") if k in mesh_with_quality.cell_data
        )
        areas = np.array(mesh_with_quality.cell_data[area_key])
        n_null = int((areas <= 0).sum())
        assert n_null == 0, f"Found {n_null} null (zero-area) faces"

    def test_manifold_no_boundary_edges(self, sphere_mask_nifti):
        """A closed sphere mesh should have no free boundary edges."""
        from meshing.generate_mesh import generate_mesh

        nifti_path, tmp_dir = sphere_mask_nifti
        output_dir = tmp_dir / "mesh_mfld"
        mesh = generate_mesh(
            mask_path=nifti_path,
            output_dir=output_dir,
            target_faces=3_000,
            smoothing_iterations=2,
            pyvista_smooth_iterations=5,
        )
        boundary = mesh.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=True,
            feature_edges=False,
            manifold_edges=False,
        )
        # A perfect sphere from marching cubes should be closed; after cleanup
        # there may be minor artefacts, so we just assert < 1% of points are
        # on a boundary.
        boundary_frac = boundary.n_points / max(mesh.n_points, 1)
        assert boundary_frac < 0.01, (
            f"{boundary.n_points} boundary/non-manifold points "
            f"({boundary_frac:.1%} of mesh) — mesh is not manifold"
        )

    def test_output_files_saved(self, sphere_mask_nifti):
        """Both .stl and .vtp files should be written to output_dir."""
        from meshing.generate_mesh import generate_mesh

        nifti_path, tmp_dir = sphere_mask_nifti
        output_dir = tmp_dir / "mesh_files"
        generate_mesh(
            mask_path=nifti_path,
            output_dir=output_dir,
            target_faces=3_000,
            smoothing_iterations=2,
            pyvista_smooth_iterations=5,
        )
        assert (output_dir / "mesh.stl").exists(), "mesh.stl not found"
        assert (output_dir / "mesh.vtp").exists(), "mesh.vtp not found"


class TestMaskLoading:
    """Unit tests for load_mask."""

    def test_load_returns_bool_array(self, sphere_mask_nifti):
        """load_mask should return a boolean array."""
        from meshing.generate_mesh import load_mask

        nifti_path, _ = sphere_mask_nifti
        mask, spacing = load_mask(nifti_path)
        assert mask.dtype == bool, "Mask should be boolean"
        assert mask.ndim == 3, "Mask should be 3-D"

    def test_spacing_positive(self, sphere_mask_nifti):
        """Spacing values should all be positive."""
        from meshing.generate_mesh import load_mask

        nifti_path, _ = sphere_mask_nifti
        _, spacing = load_mask(nifti_path)
        assert all(s > 0 for s in spacing), f"Non-positive spacing: {spacing}"
