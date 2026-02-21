"""
Tests for the graph export pipeline.

Generates a synthetic sphere mesh with fake BC labels and runs to_graph.py,
asserting the correct tensor shapes, feature dimensions, and BC one-hot values.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def labelled_sphere():
    """Create a synthetic sphere with BC labels.

    Yields:
        PyVista PolyData with bc_label.
    """
    try:
        import pyvista  # noqa: F401
        import torch  # noqa: F401
        from torch_geometric.data import Data  # noqa: F401
        from scipy.spatial import cKDTree  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"Missing dependency: {exc}")

    from tests.synthetic_data import make_sphere_mesh
    return make_sphere_mesh(radius=10.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNodeFeatures:
    """Node feature tensor shape and content."""

    def test_x_shape(self, labelled_sphere):
        """Node features must have shape (N, 9)."""
        from export.to_graph import mesh_to_pyg

        data = mesh_to_pyg(labelled_sphere)
        N = labelled_sphere.n_points
        assert data.x.shape == (N, 9), (
            f"Expected x shape ({N}, 9), got {data.x.shape}"
        )

    def test_positions_normalised(self, labelled_sphere):
        """Normalised positions should be within [0, 1] (plus small float tolerance)."""
        from export.to_graph import mesh_to_pyg
        import torch

        data = mesh_to_pyg(labelled_sphere)
        pos = data.x[:, :3]
        assert pos.min().item() >= -1e-5, "Normalised positions below 0"
        assert pos.max().item() <= 1.0 + 1e-5, "Normalised positions above 1"

    def test_normals_unit_length(self, labelled_sphere):
        """Surface normals should be approximately unit length."""
        from export.to_graph import mesh_to_pyg
        import torch

        data = mesh_to_pyg(labelled_sphere)
        normals = data.x[:, 3:6]
        norms = normals.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), (
            "Some normals are not unit length"
        )

    def test_bc_onehot_valid(self, labelled_sphere):
        """BC one-hot columns should only contain 0 or 1."""
        from export.to_graph import mesh_to_pyg

        data = mesh_to_pyg(labelled_sphere)
        is_inlet = data.x[:, 7]
        is_outlet = data.x[:, 8]
        assert set(is_inlet.tolist()).issubset({0.0, 1.0}), "is_inlet has non-binary values"
        assert set(is_outlet.tolist()).issubset({0.0, 1.0}), "is_outlet has non-binary values"

    def test_bc_onehot_counts_match_labels(self, labelled_sphere):
        """BC one-hot sums should match the raw label counts."""
        from export.to_graph import mesh_to_pyg
        import numpy as np

        data = mesh_to_pyg(labelled_sphere)
        raw_labels = np.array(labelled_sphere.point_data["bc_label"])

        expected_inlet = int((raw_labels == 1).sum())
        expected_outlet = int((raw_labels == 2).sum())

        assert int(data.x[:, 7].sum().item()) == expected_inlet, (
            "is_inlet sum does not match label count"
        )
        assert int(data.x[:, 8].sum().item()) == expected_outlet, (
            "is_outlet sum does not match label count"
        )


class TestEdgeConnectivity:
    """Edge index and attribute shapes."""

    def test_edge_index_shape(self, labelled_sphere):
        """edge_index should have shape (2, E)."""
        from export.to_graph import mesh_to_pyg

        data = mesh_to_pyg(labelled_sphere)
        assert data.edge_index.ndim == 2, "edge_index must be 2-D"
        assert data.edge_index.shape[0] == 2, (
            f"edge_index first dim must be 2, got {data.edge_index.shape[0]}"
        )

    def test_edge_index_non_empty(self, labelled_sphere):
        """There must be at least one edge."""
        from export.to_graph import mesh_to_pyg

        data = mesh_to_pyg(labelled_sphere)
        assert data.edge_index.shape[1] > 0, "edge_index is empty"

    def test_edge_index_valid_indices(self, labelled_sphere):
        """All edge endpoint indices must be valid vertex indices."""
        from export.to_graph import mesh_to_pyg

        data = mesh_to_pyg(labelled_sphere)
        N = labelled_sphere.n_points
        assert data.edge_index.min().item() >= 0, "Negative edge index"
        assert data.edge_index.max().item() < N, (
            f"Edge index {data.edge_index.max().item()} >= N={N}"
        )

    def test_edge_attr_shape(self, labelled_sphere):
        """edge_attr should have shape (E, 5)."""
        from export.to_graph import mesh_to_pyg

        data = mesh_to_pyg(labelled_sphere)
        E = data.edge_index.shape[1]
        assert data.edge_attr.shape == (E, 5), (
            f"Expected edge_attr shape ({E}, 5), got {data.edge_attr.shape}"
        )

    def test_edge_distances_positive(self, labelled_sphere):
        """Edge distances (column 3 of edge_attr) must be positive."""
        from export.to_graph import mesh_to_pyg

        data = mesh_to_pyg(labelled_sphere)
        distances = data.edge_attr[:, 3]
        assert (distances > 0).all().item(), "Some edge distances are ≤ 0"

    def test_edges_bidirectional(self, labelled_sphere):
        """For every directed edge (u→v) there should be a reverse edge (v→u)."""
        from export.to_graph import mesh_to_pyg

        data = mesh_to_pyg(labelled_sphere)
        edges = set(map(tuple, data.edge_index.T.tolist()))
        for (u, v) in list(edges)[:200]:  # sample check for speed
            assert (v, u) in edges, f"Missing reverse edge ({v}, {u})"


class TestGraphMetadata:
    """Graph-level metadata attributes."""

    def test_metadata_fields_present(self, labelled_sphere):
        """All required metadata fields must be present."""
        from export.to_graph import mesh_to_pyg

        data = mesh_to_pyg(labelled_sphere, source_file="test.nii.gz")
        for field in [
            "original_scale", "original_centroid",
            "n_inlet_faces", "n_outlet_faces", "n_wall_faces",
            "source_file", "mesh_face_count", "mesh_vertex_count",
        ]:
            assert hasattr(data, field), f"Missing metadata field: {field}"

    def test_face_and_vertex_counts(self, labelled_sphere):
        """Metadata counts must match the input mesh."""
        from export.to_graph import mesh_to_pyg

        data = mesh_to_pyg(labelled_sphere)
        assert data.mesh_face_count == labelled_sphere.n_faces_strict
        assert data.mesh_vertex_count == labelled_sphere.n_points

    def test_bc_face_counts_sum_to_total(self, labelled_sphere):
        """n_inlet + n_outlet + n_wall should equal total vertex count."""
        from export.to_graph import mesh_to_pyg

        data = mesh_to_pyg(labelled_sphere)
        total = data.n_inlet_faces + data.n_outlet_faces + data.n_wall_faces
        assert total == labelled_sphere.n_points, (
            f"BC face count sum {total} != n_points {labelled_sphere.n_points}"
        )


class TestValidators:
    """Pre-export validator logic."""

    def test_no_inlet_produces_error(self):
        """Mesh with no inlet should produce an error."""
        try:
            import pyvista as pv
            import numpy as np
        except ImportError:
            pytest.skip("pyvista not installed")

        from export.validators import validate_mesh_for_export

        mesh = pv.Sphere(radius=5.0)
        # All wall — no inlet or outlet
        mesh.point_data["bc_label"] = np.zeros(mesh.n_points, dtype=np.int32)

        errors, _ = validate_mesh_for_export(mesh)
        error_text = " ".join(errors).lower()
        assert "inlet" in error_text, "Expected inlet error not raised"

    def test_no_outlet_produces_error(self):
        """Mesh with no outlet should produce an error."""
        try:
            import pyvista as pv
            import numpy as np
        except ImportError:
            pytest.skip("pyvista not installed")

        from export.validators import validate_mesh_for_export

        mesh = pv.Sphere(radius=5.0)
        labels = np.zeros(mesh.n_points, dtype=np.int32)
        labels[:10] = 1  # some inlet, no outlet
        mesh.point_data["bc_label"] = labels

        errors, _ = validate_mesh_for_export(mesh)
        error_text = " ".join(errors).lower()
        assert "outlet" in error_text, "Expected outlet error not raised"

    def test_valid_mesh_no_errors(self):
        """A properly labelled mesh should pass without errors."""
        try:
            import pyvista  # noqa: F401
        except ImportError:
            pytest.skip("pyvista not installed")

        from tests.synthetic_data import make_sphere_mesh
        from export.validators import validate_mesh_for_export

        # Use a high-resolution sphere so vertex count exceeds the 1000 minimum
        mesh = make_sphere_mesh(radius=10.0)
        errors, _ = validate_mesh_for_export(mesh, min_nodes=100)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_node_count_too_small_error(self):
        """A mesh with too few nodes should produce an error."""
        try:
            import pyvista as pv
            import numpy as np
        except ImportError:
            pytest.skip("pyvista not installed")

        from export.validators import validate_mesh_for_export

        mesh = pv.Sphere(radius=1.0, theta_resolution=5, phi_resolution=5)
        n = mesh.n_points
        labels = np.zeros(n, dtype=np.int32)
        labels[:2] = 1
        labels[2:4] = 2
        mesh.point_data["bc_label"] = labels

        errors, _ = validate_mesh_for_export(mesh, min_nodes=10_000)
        assert any("vertices" in e.lower() for e in errors), (
            "Expected node count error not raised"
        )
