"""Allow ``python -m gui /path/to/mesh.vtp`` to launch the editor."""

import sys
from pathlib import Path


def main() -> None:
    """Entry point for ``python -m gui``."""
    if len(sys.argv) < 2:
        print("Usage: python -m gui /path/to/mesh.vtp")
        sys.exit(1)

    mesh_path = Path(sys.argv[1])
    if not mesh_path.exists():
        print(f"Error: file not found: {mesh_path}")
        sys.exit(1)

    from .main_window import launch_gui
    launch_gui(mesh_path)


if __name__ == "__main__":
    main()
