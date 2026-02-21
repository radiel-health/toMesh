"""
pymeshlab-based mesh cleanup pipeline.

Operations (in order):
  1. Remove duplicate vertices
  2. Remove null / degenerate faces
  3. Close small holes (configurable max hole size)
  4. Laplacian smoothing
  5. Quadric edge collapse decimation
  6. Recompute normals
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _require_pymeshlab() -> None:
    """Raise ImportError with install hint if pymeshlab is missing."""
    try:
        import pymeshlab  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pymeshlab is not installed. Install it with: pip install pymeshlab"
        ) from exc


def cleanup_mesh(
    input_stl: Path,
    target_faces: int = 15_000,
    max_hole_size: int = 30,
    smoothing_iterations: int = 5,
    smoothing_lambda: float = 0.5,
) -> Path:
    """Run the full pymeshlab cleanup pipeline on an STL mesh.

    Args:
        input_stl: Path to the raw .stl file from marching cubes.
        target_faces: Target face count for quadric decimation.
        max_hole_size: Maximum hole boundary edge count to auto-close.
        smoothing_iterations: Number of Laplacian smooth iterations.
        smoothing_lambda: Laplacian smoothing weight (0–1).

    Returns:
        Path to the cleaned .stl file written to a sibling 'cleaned.stl' path
        next to input_stl.

    Raises:
        ImportError: If pymeshlab is not installed.
        FileNotFoundError: If input_stl does not exist.
    """
    _require_pymeshlab()
    import pymeshlab

    if not input_stl.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_stl}")

    logger.info("pymeshlab cleanup pipeline starting on %s", input_stl)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_stl))

    n_before = ms.current_mesh().face_number()
    logger.debug("Loaded mesh: %d vertices, %d faces",
                 ms.current_mesh().vertex_number(), n_before)

    # --- Step 1: Remove duplicate vertices -----------------------------------
    ms.meshing_remove_duplicate_vertices()
    logger.debug("After duplicate vertex removal: %d faces",
                 ms.current_mesh().face_number())

    # --- Step 2: Remove null / degenerate faces ------------------------------
    ms.meshing_remove_null_faces()
    logger.debug("After null face removal: %d faces",
                 ms.current_mesh().face_number())

    # --- Step 3: Close small holes -------------------------------------------
    # 'maxholesize' is the max perimeter (in edges) of holes to close
    try:
        ms.meshing_close_holes(maxholesize=max_hole_size)
        logger.debug(
            "After hole-closing (max_hole_size=%d): %d faces",
            max_hole_size,
            ms.current_mesh().face_number(),
        )
    except Exception as exc:
        # Some mesh topologies cause hole-closing to fail; log and continue
        logger.warning("Hole-closing step raised an exception (skipping): %s", exc)

    # --- Step 4: Laplacian smoothing -----------------------------------------
    ms.apply_coord_laplacian_smoothing(
        stepsmoothnum=smoothing_iterations,
        boundary=True,
        cotangentweight=False,
        # lambda_ parameter name may differ across pymeshlab versions
        # TODO: update if pymeshlab API changes
    )
    logger.debug(
        "After Laplacian smoothing (%d iters): %d faces",
        smoothing_iterations,
        ms.current_mesh().face_number(),
    )

    # --- Step 5: Quadric edge collapse decimation ----------------------------
    current_faces = ms.current_mesh().face_number()
    if target_faces < current_faces:
        target_perc = target_faces / current_faces
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            targetperc=target_perc,
            qualitythr=0.3,
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            planarquadric=True,
        )
        logger.info(
            "Decimation: %d → %d faces (target %d)",
            current_faces,
            ms.current_mesh().face_number(),
            target_faces,
        )
    else:
        logger.info(
            "Skipping decimation: current faces (%d) <= target (%d)",
            current_faces,
            target_faces,
        )

    # --- Step 6: Recompute normals -------------------------------------------
    ms.compute_normal_per_vertex()
    ms.compute_normal_per_face()
    logger.debug("Normals recomputed")

    # --- Save output ----------------------------------------------------------
    output_path = input_stl.parent / "cleaned.stl"
    ms.save_current_mesh(str(output_path))
    logger.info(
        "Cleaned mesh saved → %s (%d faces)",
        output_path,
        ms.current_mesh().face_number(),
    )
    return output_path


def remesh_to_target(
    mesh_path: Path,
    target_faces: int,
    output_path: Path | None = None,
) -> Path:
    """Decimate an existing mesh to a new target face count.

    Convenience function used by the GUI's Decimate panel.

    Args:
        mesh_path: Path to input .stl or .vtp mesh.
        target_faces: Desired face count after decimation.
        output_path: Where to write the result (defaults to mesh_path with
            '_decimated' suffix).

    Returns:
        Path to the decimated mesh.

    Raises:
        ImportError: If pymeshlab is not installed.
    """
    _require_pymeshlab()
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))

    n_before = ms.current_mesh().face_number()
    logger.info("Remeshing %d → %d faces", n_before, target_faces)

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
    )

    if output_path is None:
        output_path = mesh_path.with_stem(mesh_path.stem + "_decimated")

    ms.save_current_mesh(str(output_path))
    logger.info(
        "Decimated mesh saved → %s (%d faces)",
        output_path,
        ms.current_mesh().face_number(),
    )
    return output_path
