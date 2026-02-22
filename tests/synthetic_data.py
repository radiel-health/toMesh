"""
Synthetic data generators for tests.

Provides:
  - make_sphere_mask: binary numpy mask of a filled sphere
  - make_sphere_mesh: PyVista PolyData of a UV sphere with fake BC labels
"""

from __future__ import annotations

import numpy as np


def make_sphere_mask(
    grid_size: int = 64,
    radius_fraction: float = 0.35,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Create a binary 3-D mask of a filled sphere.

    Args:
        grid_size: Size of the cubic grid (default 64 → 64x64x64).
        radius_fraction: Sphere radius as a fraction of grid_size.

    Returns:
        Tuple of:
            - mask: (grid_size, grid_size, grid_size) boolean array.
            - spacing: Voxel spacing in mm (uniform 1.0 mm isotropic).
    """
    r = radius_fraction * grid_size
    centre = grid_size / 2.0

    z, y, x = np.mgrid[:grid_size, :grid_size, :grid_size]
    dist = np.sqrt((x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2)
    mask = dist < r

    spacing = (1.0, 1.0, 1.0)  # 1 mm isotropic
    return mask.astype(bool), spacing


def make_sphere_mesh(
    radius: float = 10.0,
    n_inlet_fraction: float = 0.1,
    n_outlet_fraction: float = 0.1,
) -> "pyvista.PolyData":  # type: ignore[name-defined]
    """Create a PyVista UV-sphere mesh with synthetic BC labels.

    Assigns:
      - Top 10% of vertices (by Z) → outlet (label=2)
      - Bottom 10% of vertices (by Z) → inlet (label=1)
      - Remainder → wall (label=0)

    Args:
        radius: Sphere radius.
        n_inlet_fraction: Fraction of lowest-Z vertices to label as inlet.
        n_outlet_fraction: Fraction of highest-Z vertices to label as outlet.

    Returns:
        PyVista PolyData with bc_label point_data.
    """
    import pyvista as pv
    import numpy as np

    sphere = pv.Sphere(radius=radius, theta_resolution=30, phi_resolution=30)

    z = np.array(sphere.points[:, 2])
    z_sorted = np.sort(z)
    n = len(z)

    n_inlet = max(1, int(n * n_inlet_fraction))
    n_outlet = max(1, int(n * n_outlet_fraction))

    inlet_thresh = z_sorted[n_inlet]
    outlet_thresh = z_sorted[-(n_outlet + 1)]

    labels = np.zeros(n, dtype=np.int32)
    labels[z <= inlet_thresh] = 1   # inlet
    labels[z >= outlet_thresh] = 2  # outlet

    sphere.point_data["bc_label"] = labels
    return sphere
