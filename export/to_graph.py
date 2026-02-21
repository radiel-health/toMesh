"""
Convert a PyVista PolyData mesh + BC labels into a PyTorch Geometric Data object.

Node features (N, 9):
  [0:3]  x, y, z      — position normalised to unit bounding box
  [3:6]  nx, ny, nz   — surface normal unit vector
  [6]    mean_curv    — mean curvature (scalar)
  [7]    is_inlet     — BC one-hot
  [8]    is_outlet    — BC one-hot

Edge features (E, 5):
  [0:3]  dx, dy, dz   — relative position vector (src → dst)
  [3]    dist         — Euclidean distance
  [4]    normal_ang   — angle between endpoint normals (radians)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# BC label constants
WALL = 0
INLET = 1
OUTLET = 2


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install it with: pip install torch"
        ) from exc


def _require_torch_geometric() -> None:
    try:
        from torch_geometric.data import Data  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torch-geometric is not installed. See https://pyg.org/install"
        ) from exc


def compute_face_adjacency_edges(faces: np.ndarray) -> np.ndarray:
    """Build edge list from face adjacency (vertices sharing a triangle).

    Args:
        faces: (M, 3) integer array of triangle vertex indices.

    Returns:
        (2, E) edge list with all face-adjacent pairs (bidirectional).
    """
    edge_set: set[tuple[int, int]] = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in [(a, b), (b, c), (a, c)]:
            edge_set.add((u, v))
            edge_set.add((v, u))
    if not edge_set:
        return np.empty((2, 0), dtype=np.int64)
    return np.array(list(edge_set), dtype=np.int64).T  # (2, E)


def compute_radius_edges(
    points: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Connect each vertex to all neighbours within a given radius.

    Args:
        points: (N, 3) vertex positions.
        radius: Radius in world units.

    Returns:
        (2, E) bidirectional edge list.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")  # (E, 2)
    if len(pairs) == 0:
        return np.empty((2, 0), dtype=np.int64)
    # Make bidirectional
    both = np.vstack([pairs, pairs[:, ::-1]])  # (2E, 2)
    return both.T.astype(np.int64)  # (2, 2E)


def compute_mean_curvature(mesh: "pyvista.PolyData") -> np.ndarray:  # type: ignore
    """Compute mean curvature at each vertex via PyVista.

    Falls back to zeros if curvature computation fails (e.g. non-manifold).
    Handles both old API (returns PolyData with point_data) and new API
    (pyvista >= 0.44, returns numpy array directly).

    Args:
        mesh: PyVista PolyData with computed normals.

    Returns:
        (N,) float32 array of mean curvature values.
    """
    try:
        result = mesh.curvature(curv_type="mean")
        # pyvista >= 0.44 returns a numpy array directly
        if isinstance(result, np.ndarray):
            return result.astype(np.float32)
        # older pyvista returns PolyData with point_data
        return np.array(result.point_data["Mean_Curvature"], dtype=np.float32)
    except Exception as exc:
        logger.warning("Curvature computation failed (%s); using zeros.", exc)
        return np.zeros(mesh.n_points, dtype=np.float32)


def compute_mean_edge_length(points: np.ndarray, faces: np.ndarray) -> float:
    """Compute the mean edge length across all triangle edges.

    Args:
        points: (N, 3) vertex positions.
        faces: (M, 3) triangle vertex indices.

    Returns:
        Mean edge length as a float.
    """
    lengths = []
    for tri in faces[:1000]:  # sample first 1000 faces for speed
        for i, j in [(0, 1), (1, 2), (0, 2)]:
            lengths.append(np.linalg.norm(points[tri[i]] - points[tri[j]]))
    return float(np.mean(lengths)) if lengths else 1.0


def mesh_to_pyg(
    mesh: "pyvista.PolyData",  # type: ignore
    source_file: str = "",
    neighbor_radius_multiplier: float = 3.0,
) -> "torch_geometric.data.Data":  # type: ignore
    """Convert a labelled PyVista PolyData to a PyG Data object.

    Args:
        mesh: PyVista PolyData with bc_label point_data.
        source_file: Path to the originating CT scan (stored as metadata).
        neighbor_radius_multiplier: Radius edges are built at
            ``multiplier × mean_edge_length``.

    Returns:
        torch_geometric.data.Data with fields x, edge_index, edge_attr,
        and graph-level metadata attributes.

    Raises:
        ImportError: If PyTorch or torch-geometric are not installed.
        ValueError: If the mesh has no bc_label array.
    """
    _require_torch()
    _require_torch_geometric()

    import torch
    from torch_geometric.data import Data

    if "bc_label" not in mesh.point_data:
        raise ValueError(
            "Mesh has no 'bc_label' point_data. Run BC tagging before export."
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    points = np.array(mesh.points, dtype=np.float64)

    # Compute normals
    mesh_n = mesh.compute_normals(cell_normals=False, point_normals=True)
    normals = np.array(mesh_n.point_data["Normals"], dtype=np.float64)

    # Normalize to unit bounding box
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    scale = bbox_max - bbox_min
    scale[scale < 1e-8] = 1.0  # avoid divide-by-zero on flat dims
    centroid = points.mean(axis=0)

    points_norm = (points - bbox_min) / scale  # in [0, 1]

    # Mean curvature
    mean_curv = compute_mean_curvature(mesh)  # (N,)

    # BC labels
    labels = np.array(mesh.point_data["bc_label"], dtype=np.int32)
    is_inlet = (labels == INLET).astype(np.float32)
    is_outlet = (labels == OUTLET).astype(np.float32)

    # ------------------------------------------------------------------
    # Node features: shape (N, 9)
    # ------------------------------------------------------------------
    x = np.column_stack([
        points_norm.astype(np.float32),  # 3 cols
        normals.astype(np.float32),       # 3 cols
        mean_curv[:, None],               # 1 col
        is_inlet[:, None],                # 1 col
        is_outlet[:, None],               # 1 col
    ])  # (N, 9)

    # ------------------------------------------------------------------
    # Edge construction
    # ------------------------------------------------------------------
    faces_raw = mesh.faces.reshape(-1, 4)[:, 1:]  # (M, 3)

    # 1. Face adjacency edges
    face_edges = compute_face_adjacency_edges(faces_raw)

    # 2. Radius neighbour edges
    mel = compute_mean_edge_length(points, faces_raw)
    radius = neighbor_radius_multiplier * mel
    radius_edges = compute_radius_edges(points, radius)
    logger.debug(
        "Edge construction: face_adj=%d, radius(r=%.3f)=%d",
        face_edges.shape[1] if face_edges.ndim == 2 else 0,
        radius,
        radius_edges.shape[1] if radius_edges.ndim == 2 else 0,
    )

    # Merge and deduplicate
    all_edges = np.hstack([face_edges, radius_edges])  # (2, E_total)
    if all_edges.shape[1] > 0:
        # Deduplicate using a set of tuples
        edge_set = set(map(tuple, all_edges.T.tolist()))
        all_edges = np.array(list(edge_set), dtype=np.int64).T  # (2, E_dedup)
    logger.debug("After dedup: %d edges", all_edges.shape[1] if all_edges.ndim == 2 else 0)

    # ------------------------------------------------------------------
    # Edge features: shape (E, 5) — relative pos (3), dist (1), normal angle (1)
    # ------------------------------------------------------------------
    n_edges = all_edges.shape[1] if all_edges.ndim == 2 and all_edges.shape[1] > 0 else 0
    if n_edges > 0:
        src_idx = all_edges[0]  # (E,)
        dst_idx = all_edges[1]  # (E,)

        src_pts = points[src_idx].astype(np.float32)
        dst_pts = points[dst_idx].astype(np.float32)
        src_nrm = normals[src_idx].astype(np.float32)
        dst_nrm = normals[dst_idx].astype(np.float32)

        diff = dst_pts - src_pts                        # (E, 3)
        dist = np.linalg.norm(diff, axis=1, keepdims=True)  # (E, 1)

        # Angle between normals (clip for numerical safety)
        cos_ang = np.clip(
            (src_nrm * dst_nrm).sum(axis=1, keepdims=True), -1.0, 1.0
        )
        ang = np.arccos(cos_ang).astype(np.float32)  # (E, 1) in radians

        edge_attr = np.hstack([diff, dist, ang])  # (E, 5)
    else:
        edge_attr = np.empty((0, 5), dtype=np.float32)

    # ------------------------------------------------------------------
    # BC face counts
    # ------------------------------------------------------------------
    n_inlet_v = int(is_inlet.sum())
    n_outlet_v = int(is_outlet.sum())
    n_wall_v = int(((labels == WALL).sum()))

    # ------------------------------------------------------------------
    # Assemble PyG Data object
    # ------------------------------------------------------------------
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(all_edges, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )

    # Graph-level metadata
    data.original_scale = torch.tensor(scale, dtype=torch.float32)
    data.original_centroid = torch.tensor(centroid, dtype=torch.float32)
    data.n_inlet_faces = n_inlet_v
    data.n_outlet_faces = n_outlet_v
    data.n_wall_faces = n_wall_v
    data.source_file = source_file
    data.mesh_face_count = int(mesh.n_faces_strict)
    data.mesh_vertex_count = int(mesh.n_points)

    logger.info(
        "Graph assembled: N=%d nodes, E=%d edges, "
        "inlet=%d, outlet=%d, wall=%d",
        mesh.n_points, n_edges, n_inlet_v, n_outlet_v, n_wall_v,
    )
    return data
