"""
Brush-based sculpt / deform tool.

Implements push, pull, and smooth modes using:
  - Point picking in PyVista to locate brush centre
  - KD-tree radius search for neighbouring vertices
  - Gaussian falloff weighting for smooth deformation
  - BC label array is preserved through sculpt operations
"""

from __future__ import annotations

import logging
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


class SculptMode(Enum):
    """Available sculpt brush modes."""

    PUSH = auto()
    PULL = auto()
    SMOOTH = auto()


def _gaussian_weights(distances: np.ndarray, radius: float) -> np.ndarray:
    """Compute Gaussian falloff weights from brush centre.

    Args:
        distances: 1-D array of distances from the brush centre.
        radius: Brush radius (sigma = radius / 2).

    Returns:
        1-D weight array in [0, 1] — 1 at centre, ~0 at radius.
    """
    sigma = radius / 2.0
    return np.exp(-(distances ** 2) / (2 * sigma ** 2))


def sculpt_vertices(
    mesh: "pyvista.PolyData",  # type: ignore[name-defined]
    brush_centre: np.ndarray,
    brush_radius: float,
    brush_strength: float,
    mode: SculptMode,
    normal: np.ndarray | None = None,
) -> "pyvista.PolyData":  # type: ignore[name-defined]
    """Apply one sculpt brush stroke to the mesh.

    Args:
        mesh: PyVista PolyData to deform (mutated in place).
        brush_centre: (3,) array — world-space position of brush centre.
        brush_radius: Radius of influence in world units.
        brush_strength: Displacement scale factor (0–1 typical).
        mode: SculptMode (PUSH, PULL, or SMOOTH).
        normal: (3,) surface normal at brush centre for PUSH/PULL direction.
            If None, the mean normal of affected vertices is used.

    Returns:
        The modified mesh (same object, mutated in place).
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "scipy is not installed. Install it with: pip install scipy"
        ) from exc

    points = np.array(mesh.points, dtype=float)

    # Find vertices within brush radius
    tree = cKDTree(points)
    indices = tree.query_ball_point(brush_centre, r=brush_radius)

    if not indices:
        return mesh

    indices = np.array(indices, dtype=int)
    distances = np.linalg.norm(points[indices] - brush_centre, axis=1)
    weights = _gaussian_weights(distances, brush_radius) * brush_strength

    if mode == SculptMode.SMOOTH:
        # Move each affected vertex toward the weighted average of its neighbours
        for i, (vi, w) in enumerate(zip(indices, weights)):
            # Find neighbours of vi within the brush
            nbr = [j for j in indices if j != vi]
            if not nbr:
                continue
            centroid = points[nbr].mean(axis=0)
            points[vi] += w * (centroid - points[vi])
    else:
        # Determine displacement direction
        if normal is None:
            # Estimate normal from PCA of the brush neighbourhood
            centred = points[indices] - points[indices].mean(axis=0)
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
            normal = Vt[-1]  # smallest singular vector ≈ local normal

        normal = np.asarray(normal, dtype=float)
        norm_mag = np.linalg.norm(normal)
        if norm_mag > 1e-8:
            normal = normal / norm_mag

        direction = normal if mode == SculptMode.PULL else -normal

        for i, (vi, w) in enumerate(zip(indices, weights)):
            points[vi] += direction * w

    mesh.points = points
    logger.debug(
        "Sculpt %s: %d vertices affected (centre=%s, r=%.2f)",
        mode.name, len(indices), brush_centre, brush_radius,
    )
    return mesh
