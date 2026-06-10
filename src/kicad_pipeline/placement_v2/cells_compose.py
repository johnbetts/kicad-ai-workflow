"""Compose a packed group into a single rigid composite cell.

Board packing treats a whole FeatureBlock group as one rigid body —
exactly like a cell. Composition flattens every member into the group
frame; keepouts stay owner-local so they keep transforming with their
owning component through any further rigid moves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.optimization.geometry import convex_hull
from kicad_pipeline.placement_v2.cells import Cell, CellProof

if TYPE_CHECKING:
    from kicad_pipeline.placement_v2.floorplan import GroupPlan
    from kicad_pipeline.placement_v2.ir import CellKeepout


def compose_group_cell(plan: GroupPlan) -> Cell:
    """Flatten a :class:`GroupPlan` into one composite rigid cell."""
    members = tuple(
        m for pc in plan.cells for m in pc.members_in_board()
    )
    ports = tuple(
        p for pc in plan.cells for p in pc.ports_in_board()
    )
    keepouts: tuple[CellKeepout, ...] = tuple(
        ko for pc in plan.cells for ko in pc.cell.keepouts
    )
    corners = tuple(
        pt for pc in plan.cells for pt in pc.polygon_in_board()
    )
    return Cell(
        name=f"group:{plan.name}",
        kind="group",
        members=members,
        polygon=convex_hull(corners),
        ports=ports,
        keepouts=keepouts,
        proof=CellProof(checks=(f"composed from {len(plan.cells)} cells",)),
    )
