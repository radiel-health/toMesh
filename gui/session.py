"""
Session state management: undo/redo stack + current mesh state.

All mesh mutations go through SessionState.apply() so every action is
automatically push-able onto the undo stack.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Maximum number of undo steps to keep in memory
_DEFAULT_MAX_UNDO = 20


@dataclass
class MeshSnapshot:
    """Lightweight snapshot of mesh state stored on the undo stack.

    Storing full PyVista PolyData deep copies can be memory-intensive for
    large meshes. For very large meshes (>50k faces) consider storing only
    the modified point/cell arrays. TODO: investigate lazy snapshot strategy.

    Attributes:
        mesh: Deep copy of the PyVista PolyData at this point in time.
        description: Human-readable label shown in undo history UI.
    """

    mesh: object  # pyvista.PolyData — typed as object to avoid hard import
    description: str = "edit"


class SessionState:
    """Central session state with undo/redo support.

    Usage:
        session = SessionState(initial_mesh, source_file=Path("mesh.vtp"))
        session.push_snapshot("before smooth")
        session.mesh = modified_mesh
        session.undo()   # restores previous mesh
        session.redo()   # re-applies the action

    Attributes:
        source_file: Original input path (used in graph metadata).
        on_mesh_changed: Optional callback fired whenever mesh changes.
    """

    def __init__(
        self,
        initial_mesh: object,
        source_file: Optional[Path] = None,
        max_undo: int = _DEFAULT_MAX_UNDO,
    ) -> None:
        """Initialise session with an initial mesh.

        Args:
            initial_mesh: PyVista PolyData to start with.
            source_file: Path to the source CT scan (stored in graph metadata).
            max_undo: Maximum undo stack depth.
        """
        self._mesh = initial_mesh
        self.source_file = source_file
        self._max_undo = max_undo

        # Stacks store MeshSnapshot objects
        self._undo_stack: List[MeshSnapshot] = []
        self._redo_stack: List[MeshSnapshot] = []

        # Registered callbacks called when mesh changes
        self.on_mesh_changed: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------
    # Mesh property
    # ------------------------------------------------------------------

    @property
    def mesh(self) -> object:
        """Current mesh (PyVista PolyData)."""
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh: object) -> None:
        """Replace current mesh and notify listeners."""
        self._mesh = new_mesh
        if self.on_mesh_changed:
            self.on_mesh_changed()

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def push_snapshot(self, description: str = "edit") -> None:
        """Push the current mesh state onto the undo stack.

        Call this BEFORE applying any modification so the pre-edit state
        is saved.

        Args:
            description: Label for this snapshot (displayed in UI).
        """
        try:
            snap = MeshSnapshot(
                mesh=copy.deepcopy(self._mesh),
                description=description,
            )
        except Exception as exc:
            logger.warning("Failed to create mesh snapshot: %s", exc)
            return

        self._undo_stack.append(snap)
        self._redo_stack.clear()  # new action invalidates redo history

        # Trim stack to max depth
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

        logger.debug(
            "Snapshot pushed: '%s' (undo depth=%d)", description, len(self._undo_stack)
        )

    def undo(self) -> bool:
        """Revert to the previous snapshot.

        Returns:
            True if undo was performed, False if stack was empty.
        """
        if not self._undo_stack:
            logger.debug("Undo stack empty — nothing to undo")
            return False

        # Push current state onto redo stack before reverting
        self._redo_stack.append(
            MeshSnapshot(mesh=copy.deepcopy(self._mesh), description="redo point")
        )

        snap = self._undo_stack.pop()
        self.mesh = snap.mesh
        logger.info("Undo: restored '%s'", snap.description)
        return True

    def redo(self) -> bool:
        """Re-apply the most recently undone action.

        Returns:
            True if redo was performed, False if stack was empty.
        """
        if not self._redo_stack:
            logger.debug("Redo stack empty — nothing to redo")
            return False

        self._undo_stack.append(
            MeshSnapshot(mesh=copy.deepcopy(self._mesh), description="undo point")
        )

        snap = self._redo_stack.pop()
        self.mesh = snap.mesh
        logger.info("Redo: restored '%s'", snap.description)
        return True

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        """True if there are actions to undo."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """True if there are actions to redo."""
        return len(self._redo_stack) > 0

    def bc_counts(self) -> dict[str, int]:
        """Return a dict with counts of BC-tagged vertices.

        Reads the 'bc_label' point_data array from the current mesh.
        0 = wall, 1 = inlet, 2 = outlet.

        Returns:
            Dict with keys 'wall', 'inlet', 'outlet' and integer counts.
        """
        try:
            import numpy as np

            labels = self._mesh.point_data.get("bc_label")
            if labels is None:
                n = self._mesh.n_points
                return {"wall": n, "inlet": 0, "outlet": 0}
            labels = np.asarray(labels)
            return {
                "wall": int((labels == 0).sum()),
                "inlet": int((labels == 1).sum()),
                "outlet": int((labels == 2).sum()),
            }
        except Exception:
            return {"wall": 0, "inlet": 0, "outlet": 0}
