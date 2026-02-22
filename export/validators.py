"""
Pre-export mesh validation.

Returns lists of errors (hard failures) and warnings (soft issues).
Callers decide whether to block or just warn.
"""

from __future__ import annotations

import logging
from typing import Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

# BC label constants
WALL = 0
INLET = 1
OUTLET = 2

# Default bounds (overridden by config.yaml at runtime via pipeline.py)
_DEFAULT_MIN_NODES = 1000
_DEFAULT_MAX_NODES = 100_000


def validate_mesh_for_export(
    mesh: "pyvista.PolyData",  # type: ignore
    min_nodes: int = _DEFAULT_MIN_NODES,
    max_nodes: int = _DEFAULT_MAX_NODES,
) -> Tuple[List[str], List[str]]:
    """Run all pre-export sanity checks on a mesh.

    BC labels are optional for ML export. When absent or incomplete, checks
    that require them are demoted to warnings so export is never blocked solely
    due to missing BC tags (which are only needed for CFD simulation setup).

    Checks (ERRORS — block export):
      1. Node count within [min_nodes, max_nodes].

    Checks (WARNINGS — log but allow export):
      2. No 'bc_label' present (ML export still works; CFD setup will need it).
      3. At least 1 inlet + 1 outlet tagged (needed for CFD, not for ML).
      4. No unlabelled faces (bc_label == -1).
      5. Inlet/outlet regions should be on topological boundary (free edges).
      6. Mesh should have no self-intersections (slow for large meshes).

    Args:
        mesh: PyVista PolyData, optionally with bc_label point_data.
        min_nodes: Minimum vertex count.
        max_nodes: Maximum vertex count.

    Returns:
        Tuple of (errors, warnings) — each a list of human-readable strings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # ------------------------------------------------------------------
    # Error: node count bounds (always checked)
    # ------------------------------------------------------------------
    n_nodes = mesh.n_points
    if n_nodes < min_nodes:
        errors.append(
            f"Mesh has only {n_nodes:,} vertices (minimum: {min_nodes:,}). "
            "Consider reducing decimation or checking the segmentation."
        )
    if n_nodes > max_nodes:
        errors.append(
            f"Mesh has {n_nodes:,} vertices (maximum: {max_nodes:,}). "
            "Run further decimation before export."
        )

    if "bc_label" not in mesh.point_data:
        warnings.append(
            "No 'bc_label' point_data found. "
            "ML export will treat all vertices as WALL. "
            "Tag boundary conditions if you need CFD (OpenFOAM) setup."
        )
        # Remaining checks all need the label array — skip them
        if errors:
            logger.error("Export validation FAILED: %d error(s)", len(errors))
        else:
            logger.info("Export validation passed (%d warning(s))", len(warnings))
        return errors, warnings

    labels = np.array(mesh.point_data["bc_label"], dtype=np.int32)

    # ------------------------------------------------------------------
    # Warning: at least 1 inlet + 1 outlet (needed for CFD, not ML)
    # ------------------------------------------------------------------
    n_inlet = int((labels == INLET).sum())
    n_outlet = int((labels == OUTLET).sum())

    if n_inlet == 0:
        warnings.append(
            "No inlet vertices tagged. "
            "Required for CFD (OpenFOAM) setup; not needed for ML export."
        )
    if n_outlet == 0:
        warnings.append(
            "No outlet vertices tagged. "
            "Required for CFD (OpenFOAM) setup; not needed for ML export."
        )

    # ------------------------------------------------------------------
    # Warning: no unlabelled faces (-1)
    # ------------------------------------------------------------------
    n_unlabelled = int((labels == -1).sum())
    if n_unlabelled > 0:
        warnings.append(
            f"{n_unlabelled:,} vertices have label -1 (unlabelled). "
            "All tagged vertices should be WALL (0), INLET (1), or OUTLET (2)."
        )

    # ------------------------------------------------------------------
    # Warning: inlet/outlet should be on free (boundary) edges
    # ------------------------------------------------------------------
    try:
        boundary = mesh.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False,
        )
        boundary_pts = set(
            _find_original_indices(np.array(mesh.points), np.array(boundary.points))
        )

        for label, name in [(INLET, "inlet"), (OUTLET, "outlet")]:
            tagged_indices = set(np.where(labels == label)[0].tolist())
            on_boundary = tagged_indices & boundary_pts
            if len(tagged_indices) > 0 and len(on_boundary) == 0:
                warnings.append(
                    f"The {name} region does not appear to sit on a topological "
                    "boundary (free edge loop). Verify the tag placement."
                )
    except Exception as exc:
        warnings.append(f"Could not check boundary topology: {exc}")

    # ------------------------------------------------------------------
    # Warning: self-intersections (can be slow)
    # ------------------------------------------------------------------
    if mesh.n_faces_strict < 50_000:
        try:
            collision, n_contacts = mesh.collision(mesh, generate_scalars=False)
            if n_contacts > 0:
                warnings.append(
                    f"Mesh has {n_contacts} self-intersection contact(s). "
                    "CFD solvers may fail on self-intersecting geometry."
                )
        except Exception as exc:
            logger.debug("Self-intersection check failed: %s", exc)
    else:
        logger.info(
            "Skipping self-intersection check (mesh > 50k faces; too slow)."
        )

    if errors:
        logger.error("Export validation FAILED: %d error(s)", len(errors))
    else:
        logger.info(
            "Export validation passed (%d warning(s))", len(warnings)
        )

    return errors, warnings


def _find_original_indices(
    all_points: np.ndarray,
    query_points: np.ndarray,
) -> list[int]:
    """Find indices in all_points closest to each point in query_points.

    Args:
        all_points: (N, 3) array of all mesh vertices.
        query_points: (M, 3) array of boundary vertices.

    Returns:
        List of length M with indices into all_points.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(all_points)
    _, idx = tree.query(query_points, k=1)
    return idx.tolist()
