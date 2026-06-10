"""Placement engine v2 — cells, contracts, and proofs.

Replaces the 32-phase v1 optimizer with a 4-stage pure-function
pipeline (certify -> cell generation -> floorplan -> verify). See
``docs/placement_v2_architecture.md`` for the architecture.

Core principle: nothing is true because a stage claimed it; everything
is re-derived from artifacts by the verifier using the same constraint
objects the solvers consumed.
"""

from kicad_pipeline.placement_v2.cells import (
    Cell,
    CellProof,
    PlacedCell,
    PlacedMember,
    Port,
)
from kicad_pipeline.placement_v2.ir import (
    Axis,
    BoardContain,
    CellKeepout,
    ConstraintSet,
    Edge,
    EdgePin,
    IsolationGap,
    KeepoutKind,
    PadRef,
    PinAttach,
    SequenceAlong,
    Severity,
    Violation,
)

__all__ = [
    "Axis",
    "BoardContain",
    "Cell",
    "CellKeepout",
    "CellProof",
    "ConstraintSet",
    "Edge",
    "EdgePin",
    "IsolationGap",
    "KeepoutKind",
    "PadRef",
    "PinAttach",
    "PlacedCell",
    "PlacedMember",
    "Port",
    "SequenceAlong",
    "Severity",
    "Violation",
]
