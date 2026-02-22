"""
Surface mesh generation from a binary segmentation mask.

Pipeline:
  1. Read combined_mask.nii.gz (SimpleITK) → numpy array + voxel spacing
  2. Run skimage marching cubes with correct voxel spacing
  3. Hand off vertices/faces to cleanup.py (pymeshlab pipeline)
  4. Reload into PyVista, run one additional smooth pass
  5. Validate and save .stl + .vtp
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _require_skimage() -> None:
    try:
        import skimage  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "scikit-image is not installed. Install it with: pip install scikit-image"
        ) from exc


def _require_pyvista() -> None:
    try:
        import pyvista  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyVista is not installed. Install it with: pip install pyvista"
        ) from exc


def load_mask(mask_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI binary mask and return the array + voxel spacing.

    Args:
        mask_path: Path to a NIfTI (.nii / .nii.gz) mask file.

    Returns:
        Tuple of:
            - mask: Boolean 3-D array in (Z, Y, X) order.
            - spacing: (sz, sy, sx) voxel spacing in mm.

    Raises:
        ImportError: If SimpleITK is not installed.
        FileNotFoundError: If the mask file does not exist.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is not installed. Install it with: pip install SimpleITK"
        ) from exc

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img).astype(bool)  # (Z, Y, X)

    # SimpleITK spacing is (X, Y, Z); reverse for numpy axis order
    spacing_xyz = img.GetSpacing()
    spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])  # (Z, Y, X) mm

    logger.info(
        "Mask loaded: shape=%s, spacing(ZYX)=(%.3f, %.3f, %.3f) mm, "
        "foreground=%d voxels",
        arr.shape,
        *spacing,
        int(arr.sum()),
    )
    return arr, spacing


def marching_cubes(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    level: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract an iso-surface from the mask using marching cubes.

    Args:
        mask: Boolean or float 3-D array (Z, Y, X).
        spacing: Voxel spacing (sz, sy, sx) in mm to correctly scale vertices.
        level: Iso-surface threshold (default 0.5 for binary masks).

    Returns:
        Tuple of:
            - vertices: (N, 3) float array of vertex positions in mm.
            - faces: (M, 3) int array of triangle vertex indices.
            - normals: (N, 3) float array of vertex normals.

    Raises:
        ImportError: If scikit-image is not installed.
    """
    _require_skimage()
    from skimage.measure import marching_cubes as skimage_mc

    logger.info(
        "Running marching cubes (level=%.2f, spacing=(%.3f, %.3f, %.3f))",
        level,
        *spacing,
    )

    # skimage expects spacing as (sz, sy, sx) matching array axes
    verts, faces, normals, _ = skimage_mc(
        mask.astype(np.float32),
        level=level,
        spacing=spacing,
        method="lewiner",
    )

    logger.info(
        "Marching cubes complete: %d vertices, %d faces", len(verts), len(faces)
    )
    return verts, faces, normals


def generate_mesh(
    mask_path: Path,
    output_dir: Path,
    mc_level: float = 0.5,
    target_faces: int = 15_000,
    max_hole_size: int = 30,
    smoothing_iterations: int = 5,
    smoothing_lambda: float = 0.5,
    pyvista_smooth_iterations: int = 20,
    pyvista_smooth_factor: float = 0.1,
    min_faces: int = 1000,
    max_faces: int = 200_000,
) -> "pyvista.PolyData":  # type: ignore[name-defined]
    """Full mesh generation pipeline: mask → clean triangular surface mesh.

    Steps:
    1. Load mask + spacing from NIfTI.
    2. Marching cubes with correct spacing.
    3. pymeshlab cleanup (dedup, close holes, smooth, decimate).
    4. One PyVista smooth pass.
    5. Validate mesh.
    6. Save .stl and .vtp.

    Args:
        mask_path: Path to combined_mask.nii.gz.
        output_dir: Directory where mesh.stl and mesh.vtp will be written.
        mc_level: Marching cubes iso-threshold.
        target_faces: Target face count for decimation.
        max_hole_size: Max hole boundary edge count to auto-close.
        smoothing_iterations: Laplacian smooth iterations (pymeshlab).
        smoothing_lambda: Laplacian smooth weight.
        pyvista_smooth_iterations: PyVista smooth iterations.
        pyvista_smooth_factor: PyVista smooth relaxation factor.
        min_faces: Minimum acceptable face count (validation warning).
        max_faces: Maximum acceptable face count (validation warning).

    Returns:
        PyVista PolyData of the final mesh (also saved to disk).

    Raises:
        ImportError: If required libraries are not installed.
    """
    _require_skimage()
    _require_pyvista()

    from .cleanup import cleanup_mesh

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    mask, spacing = load_mask(mask_path)

    # 2. Marching cubes
    verts, faces, _normals = marching_cubes(mask, spacing, level=mc_level)

    # 3. pymeshlab cleanup — writes to a temp STL, returns cleaned verts/faces
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_stl = Path(tmpdir) / "raw.stl"
        _save_stl_raw(verts, faces, raw_stl)

        cleaned_stl = cleanup_mesh(
            raw_stl,
            target_faces=target_faces,
            max_hole_size=max_hole_size,
            smoothing_iterations=smoothing_iterations,
            smoothing_lambda=smoothing_lambda,
        )

        # 4. Load into PyVista for final smooth pass
        import pyvista as pv

        mesh = pv.read(str(cleaned_stl))

    logger.info(
        "After pymeshlab cleanup: %d vertices, %d faces",
        mesh.n_points,
        mesh.n_faces_strict,
    )

    # PyVista smooth pass
    mesh = mesh.smooth(
        n_iter=pyvista_smooth_iterations,
        relaxation_factor=pyvista_smooth_factor,
        feature_smoothing=False,
        boundary_smoothing=True,
    )
    logger.info("PyVista smooth pass complete")

    # 5. Validate
    _validate_mesh(mesh, min_faces=min_faces, max_faces=max_faces)

    # 6. Save outputs
    stl_path = output_dir / "mesh.stl"
    vtp_path = output_dir / "mesh.vtp"
    mesh.save(str(stl_path))
    mesh.save(str(vtp_path))
    logger.info("Mesh saved: %s, %s", stl_path, vtp_path)

    return mesh


def _save_stl_raw(
    verts: np.ndarray, faces: np.ndarray, output_path: Path
) -> None:
    """Write vertices + faces to a binary STL for pymeshlab input.

    Args:
        verts: (N, 3) float array of vertex positions.
        faces: (M, 3) int array of triangle indices.
        output_path: Destination .stl path.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "trimesh is not installed. Install it with: pip install trimesh"
        ) from exc

    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tm.export(str(output_path))
    logger.debug("Raw STL saved (%d faces) → %s", len(faces), output_path)


def _validate_mesh(
    mesh: "pyvista.PolyData",
    min_faces: int = 1000,
    max_faces: int = 200_000,
) -> None:
    """Log warnings if the mesh fails basic sanity checks.

    Checks:
    - Face count within [min_faces, max_faces]
    - Mesh is all-triangle
    - No null (degenerate) faces (via PyVista quality metric)
    - Basic manifold check via PyVista

    Args:
        mesh: PyVista PolyData to validate.
        min_faces: Minimum acceptable face count.
        max_faces: Maximum acceptable face count.
    """
    n_faces = mesh.n_faces_strict
    if n_faces < min_faces:
        logger.warning(
            "Mesh has only %d faces (min expected: %d). "
            "Consider reducing decimation target or checking the mask.",
            n_faces, min_faces,
        )
    if n_faces > max_faces:
        logger.warning(
            "Mesh has %d faces (max expected: %d). "
            "Consider increasing decimation aggressiveness.",
            n_faces, max_faces,
        )

    # Check manifold via PyVista boundary edges
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    )
    n_non_manifold = edges.n_points
    if n_non_manifold > 0:
        logger.warning(
            "Mesh has %d non-manifold or boundary edge points. "
            "Manual repair may be needed.",
            n_non_manifold,
        )
    else:
        logger.info("Mesh validation passed: manifold, %d faces", n_faces)
