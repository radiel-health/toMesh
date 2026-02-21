"""
Boundary condition painting logic.

BC label values:
  0 = wall  (default, grey)
  1 = inlet (blue)
  2 = outlet (red)

Selection modes:
  - SINGLE: paint individual faces on click
  - FLOOD: flood-fill from a seed face to connected faces within angle threshold
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# BC label constants
WALL = 0
INLET = 1
OUTLET = 2

# Colour map (RGBA, 0–1)
BC_COLORS = {
    WALL: (0.6, 0.6, 0.6, 1.0),
    INLET: (0.2, 0.4, 1.0, 1.0),
    OUTLET: (1.0, 0.2, 0.2, 1.0),
}


def ensure_bc_array(mesh: "pyvista.PolyData") -> "pyvista.PolyData":  # type: ignore
    """Ensure the mesh has a 'bc_label' point_data array initialised to WALL.

    Args:
        mesh: PyVista PolyData.

    Returns:
        The same mesh (mutated in place) with bc_label guaranteed to exist.
    """
    if "bc_label" not in mesh.point_data:
        mesh.point_data["bc_label"] = np.zeros(mesh.n_points, dtype=np.int32)
        logger.debug("bc_label array initialised (all WALL) for %d points", mesh.n_points)
    return mesh


def paint_face(
    mesh: "pyvista.PolyData",
    face_id: int,
    label: int,
) -> None:
    """Assign a BC label to all vertices of a single face.

    Args:
        mesh: PyVista PolyData with bc_label point_data.
        face_id: Index of the face to paint.
        label: BC label value (WALL=0, INLET=1, OUTLET=2).
    """
    ensure_bc_array(mesh)
    faces = mesh.faces.reshape(-1, 4)  # [n_tri, 4] with first col = 3
    if face_id >= len(faces):
        logger.warning("face_id %d out of range (%d faces)", face_id, len(faces))
        return
    vertex_ids = faces[face_id, 1:]
    mesh.point_data["bc_label"][vertex_ids] = label


def flood_fill_faces(
    mesh: "pyvista.PolyData",
    seed_face_id: int,
    label: int,
    angle_threshold_deg: float = 30.0,
) -> int:
    """Flood-fill BC label from a seed face to all connected faces within angle threshold.

    Uses BFS over the face adjacency graph. Two faces are adjacent if they share
    an edge. The fill stops at faces whose normal deviates > angle_threshold_deg
    from the seed face's normal.

    Args:
        mesh: PyVista PolyData with bc_label point_data.
        seed_face_id: Index of the starting face.
        label: BC label to assign.
        angle_threshold_deg: Maximum normal deviation to continue fill.

    Returns:
        Number of faces painted.
    """
    ensure_bc_array(mesh)

    faces_arr = mesh.faces.reshape(-1, 4)[:, 1:]  # (M, 3)
    n_faces = len(faces_arr)

    # Build face normals
    # cell_normals requires face normals to be computed
    mesh_with_normals = mesh.compute_normals(cell_normals=True, point_normals=False)
    face_normals = np.array(mesh_with_normals.cell_data["Normals"])  # (M, 3)

    seed_normal = face_normals[seed_face_id]
    cos_thresh = np.cos(np.radians(angle_threshold_deg))

    # Build adjacency: vertex → list of face indices
    vertex_to_faces: dict[int, list[int]] = {}
    for fi, tri in enumerate(faces_arr):
        for v in tri:
            vertex_to_faces.setdefault(int(v), []).append(fi)

    # BFS
    visited = {seed_face_id}
    queue = [seed_face_id]
    painted = 0

    while queue:
        current_face = queue.pop(0)
        # Paint all vertices of this face
        vertex_ids = faces_arr[current_face]
        mesh.point_data["bc_label"][vertex_ids] = label
        painted += 1

        # Find neighbouring faces (share at least one vertex)
        neighbours: set[int] = set()
        for v in faces_arr[current_face]:
            neighbours.update(vertex_to_faces.get(int(v), []))
        neighbours.discard(current_face)

        for nf in neighbours:
            if nf in visited:
                continue
            visited.add(nf)
            # Check angle constraint
            cos_angle = float(np.dot(face_normals[nf], seed_normal))
            if cos_angle >= cos_thresh:
                queue.append(nf)

    logger.debug(
        "Flood fill from face %d: painted %d faces with label %d",
        seed_face_id, painted, label,
    )
    return painted


def auto_detect_bc(
    mesh: "pyvista.PolyData",
) -> "pyvista.PolyData":
    """Auto-tag boundary conditions based on open boundary loops.

    Heuristic:
    1. Find all free boundary edges (edges belonging to only one face).
    2. Group them into loops by connectivity.
    3. For each loop, flood-fill the adjacent faces.
    4. The loop with the lowest mean Z centroid → INLET.
    5. All other loops → OUTLET.
    6. All remaining faces → WALL.

    This works well for aorta / vessel geometries where the lowest opening
    is the inlet. Override via the GUI BC tagger after auto-detection.

    Args:
        mesh: PyVista PolyData (will be mutated in place).

    Returns:
        The same mesh with bc_label point_data assigned.
    """
    ensure_bc_array(mesh)
    labels = mesh.point_data["bc_label"]
    labels[:] = WALL  # reset to all-wall

    # Extract free boundary edges
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False,
    )

    if boundary.n_points == 0:
        logger.info("No boundary edges found — mesh may be closed. No auto-BC applied.")
        return mesh

    # Group boundary points into loops using connected components
    try:
        regions = boundary.connectivity(largest=False)
    except Exception:
        # Older PyVista API
        regions = boundary.connectivity()

    region_ids = np.array(regions.point_data.get("RegionId", []))
    boundary_points = np.array(regions.points)

    if len(region_ids) == 0:
        logger.warning("Could not extract boundary region IDs; skipping auto-BC.")
        return mesh

    unique_regions = np.unique(region_ids)
    logger.info("Found %d boundary loop(s)", len(unique_regions))

    # For each region, find mean Z of its boundary points
    region_centroids: list[tuple[int, float]] = []
    for rid in unique_regions:
        mask = region_ids == rid
        mean_z = float(boundary_points[mask, 2].mean())
        region_centroids.append((int(rid), mean_z))

    # Sort by Z: lowest Z = inlet
    region_centroids.sort(key=lambda x: x[1])

    # Map region points back to mesh points using KD-tree proximity
    from scipy.spatial import cKDTree
    mesh_tree = cKDTree(np.array(mesh.points))
    _, idx_map = mesh_tree.query(boundary_points, k=1)

    for loop_rank, (rid, mean_z) in enumerate(region_centroids):
        loop_label = INLET if loop_rank == 0 else OUTLET
        loop_mask = region_ids == rid
        loop_point_indices = idx_map[loop_mask]

        # Flood-fill from any face that contains one of these boundary vertices
        faces_arr = mesh.faces.reshape(-1, 4)[:, 1:]
        seed_face: Optional[int] = None
        seed_vertex = int(loop_point_indices[0]) if len(loop_point_indices) > 0 else None

        if seed_vertex is not None:
            for fi, tri in enumerate(faces_arr):
                if seed_vertex in tri:
                    seed_face = fi
                    break

        if seed_face is not None:
            painted = flood_fill_faces(mesh, seed_face, loop_label, angle_threshold_deg=45.0)
            logger.info(
                "Loop %d (mean_z=%.2f): %s, painted %d faces",
                rid, mean_z,
                "INLET" if loop_label == INLET else "OUTLET",
                painted,
            )
        else:
            # Fallback: tag the vertex directly
            mesh.point_data["bc_label"][loop_point_indices] = loop_label
            logger.debug("Loop %d: direct vertex tag (no seed face found)", rid)

    return mesh


def get_bc_color_array(mesh: "pyvista.PolyData") -> np.ndarray:
    """Build an RGBA colour array from the bc_label point_data.

    Args:
        mesh: PyVista PolyData with bc_label array.

    Returns:
        (N, 4) uint8 RGBA array suitable for PyVista scalar colouring.
    """
    ensure_bc_array(mesh)
    labels = np.asarray(mesh.point_data["bc_label"])
    rgba = np.zeros((len(labels), 4), dtype=np.uint8)

    for label, color in BC_COLORS.items():
        mask = labels == label
        rgba[mask] = [int(c * 255) for c in color]

    return rgba
