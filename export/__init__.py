"""Export package: mesh + BC labels → PyTorch Geometric Data object."""

from .to_graph import mesh_to_pyg
from .validators import validate_mesh_for_export

__all__ = ["mesh_to_pyg", "validate_mesh_for_export"]
