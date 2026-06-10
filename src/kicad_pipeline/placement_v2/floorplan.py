"""Stages 2+3 — deterministic floorplanning of cells and groups.

One greedy, fully deterministic packer is used at both levels:

* **Group packing** — a FeatureBlock's cells are packed into a compact
  group-local layout, candidate positions generated beside already-
  placed cells, scored by port-to-port half-perimeter wirelength (HPWL).
* **Board packing** — group composites are packed onto the board the
  same way; groups containing edge-pinned connectors are placed first,
  flush against their edge.

There is no randomness anywhere: identical inputs produce identical
boards, so golden tests can assert byte-equality. Sub-millimeter
residuals are cleaned by :mod:`legalize`; correctness is proven by the
Stage 4 verifier — the packer itself never silently degrades a
constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.optimization.geometry import (
    convex_polygons_overlap,
    polygon_bbox,
    transform_polygon,
)
from kicad_pipeline.placement_v2.cells import PlacedCell

if TYPE_CHECKING:
    from kicad_pipeline.placement_v2.cells import CardinalRotation, Cell
    from kicad_pipeline.placement_v2.ir import ConstraintSet, Edge

_GROUP_CLEARANCE_MM = 1.0
_BOARD_CLEARANCE_MM = 2.0
_EDGE_MARGIN_MM = 1.0
_ROTATIONS: tuple[CardinalRotation, ...] = (0, 90, 180, 270)


@dataclass(frozen=True)
class GroupPlan:
    """A packed FeatureBlock: cells in group-local frame + hull bbox."""

    name: str
    cells: tuple[PlacedCell, ...]
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 local

    @property
    def width(self) -> float:
        """Bounding width of the packed group."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Bounding height of the packed group."""
        return self.bbox[3] - self.bbox[1]


@dataclass(frozen=True)
class Floorplan:
    """Final board-frame placement of every cell + chosen board size."""

    placed: tuple[PlacedCell, ...]
    board_width: float
    board_height: float


def _hpwl(port_groups: dict[str, list[tuple[float, float]]]) -> float:
    """Half-perimeter wirelength over nets with 2+ ports."""
    total = 0.0
    for pts in port_groups.values():
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def _collect_ports(
    placed: list[PlacedCell], candidate: PlacedCell | None = None,
) -> dict[str, list[tuple[float, float]]]:
    groups: dict[str, list[tuple[float, float]]] = {}
    pool = [*placed, candidate] if candidate is not None else list(placed)
    for pc in pool:
        for port in pc.ports_in_board():
            groups.setdefault(port.net, []).append((port.x, port.y))
    return groups


def _candidate_offsets(
    placed: list[PlacedCell], cand_w: float, cand_h: float, clearance: float,
) -> list[tuple[float, float]]:
    """Slots beside each placed cell's bbox: right, below, left, above."""
    slots: list[tuple[float, float]] = []
    for pc in placed:
        x1, y1, x2, y2 = polygon_bbox(pc.polygon_in_board())
        slots.extend((
            (x2 + clearance + cand_w / 2, (y1 + y2) / 2),  # right
            ((x1 + x2) / 2, y2 + clearance + cand_h / 2),  # below
            (x1 - clearance - cand_w / 2, (y1 + y2) / 2),  # left
            ((x1 + x2) / 2, y1 - clearance - cand_h / 2),  # above
            (x2 + clearance + cand_w / 2, y1 + cand_h / 2),  # right-top corner
            (x1 + cand_w / 2, y2 + clearance + cand_h / 2),  # below-left corner
        ))
    return slots


def _center_offset(cell: Cell, rotation: int) -> tuple[float, float, float, float]:
    """Bbox of the cell polygon at the given rotation around origin."""
    poly = transform_polygon(cell.polygon, 0.0, 0.0, float(rotation))
    return polygon_bbox(poly)


def _fits(
    candidate: PlacedCell,
    placed: list[PlacedCell],
    clearance: float,
    board: tuple[float, float] | None,
) -> bool:
    poly = candidate.polygon_in_board()
    if board is not None:
        x1, y1, x2, y2 = polygon_bbox(poly)
        bw, bh = board
        if (x1 < _EDGE_MARGIN_MM or y1 < _EDGE_MARGIN_MM
                or x2 > bw - _EDGE_MARGIN_MM or y2 > bh - _EDGE_MARGIN_MM):
            return False
    return all(
        not convex_polygons_overlap(poly, other.polygon_in_board(),
                                    clearance_mm=clearance)
        for other in placed
    )


def _place_greedy(
    cells: tuple[Cell, ...],
    clearance: float,
    board: tuple[float, float] | None,
    allow_rotation: bool,
) -> list[PlacedCell]:
    """Greedy packer: largest cell first, then best-HPWL slot per cell."""
    if not cells:
        return []
    order = sorted(cells, key=lambda c: (-c.area, c.name))
    first = order[0]
    fx1, fy1, fx2, fy2 = _center_offset(first, 0)
    if board is not None:
        # Anchor the largest cell at the board center.
        bx, by = board[0] / 2, board[1] / 2
        start = PlacedCell(first, bx - (fx1 + fx2) / 2, by - (fy1 + fy2) / 2, 0)
    else:
        start = PlacedCell(first, -(fx1 + fx2) / 2, -(fy1 + fy2) / 2, 0)
    placed: list[PlacedCell] = [start]

    for cell in order[1:]:
        best: PlacedCell | None = None
        best_score = float("inf")
        rotations = _ROTATIONS if allow_rotation else (0,)
        for rot in rotations:
            x1, y1, x2, y2 = _center_offset(cell, rot)
            w, h = x2 - x1, y2 - y1
            for sx, sy in _candidate_offsets(placed, w, h, clearance):
                # Translate so the rotated bbox center lands on the slot.
                dx = sx - (x1 + x2) / 2
                dy = sy - (y1 + y2) / 2
                cand = PlacedCell(cell, dx, dy, rot)
                if not _fits(cand, placed, clearance, board):
                    continue
                score = _hpwl(_collect_ports(placed, cand))
                if score < best_score - 1e-9:
                    best_score = score
                    best = cand
        if best is None:
            raise PCBError(
                f"floorplan: no feasible slot for cell {cell.name!r}"
                + (f" on {board[0]}x{board[1]}mm board" if board else "")
            )
        placed.append(best)
    return placed


def pack_group(
    name: str,
    cells: tuple[Cell, ...],
    clearance_mm: float = _GROUP_CLEARANCE_MM,
) -> GroupPlan:
    """Pack a FeatureBlock's cells into a compact group-local layout."""
    placed = _place_greedy(cells, clearance_mm, board=None, allow_rotation=True)
    if not placed:
        return GroupPlan(name=name, cells=(), bbox=(0.0, 0.0, 0.0, 0.0))
    xs1, ys1, xs2, ys2 = zip(
        *(polygon_bbox(pc.polygon_in_board()) for pc in placed), strict=True
    )
    return GroupPlan(
        name=name,
        cells=tuple(placed),
        bbox=(min(xs1), min(ys1), max(xs2), max(ys2)),
    )


def _has_edge_pin(plan: GroupPlan, edge_pins: dict[str, Edge | None]) -> bool:
    return any(
        ref in edge_pins for pc in plan.cells for ref in pc.cell.refs
    )


def pack_board(
    groups: tuple[GroupPlan, ...],
    constraints: ConstraintSet,
    board_width: float | None = None,
    board_height: float | None = None,
    clearance_mm: float = _BOARD_CLEARANCE_MM,
    shrink_margin_mm: float = 3.0,
) -> Floorplan:
    """Pack groups onto the board; shrink-to-fit when no size is given.

    Groups containing edge-pinned connectors are packed first so they
    claim their edges; the rest follow greedily by ratsnest. With no
    target size, packing happens on an unconstrained canvas and the
    outline becomes the bounding box plus *shrink_margin_mm*.
    """
    from kicad_pipeline.placement_v2.cells_compose import compose_group_cell

    edge_pins: dict[str, Edge | None] = {
        ep.ref: ep.edge for ep in constraints.edge_pins
    }
    composites = {g.name: compose_group_cell(g) for g in groups}

    pinned = [g for g in groups if _has_edge_pin(g, edge_pins)]
    free = [g for g in groups if not _has_edge_pin(g, edge_pins)]
    ordered = (
        sorted(pinned, key=lambda g: (-g.width * g.height, g.name))
        + sorted(free, key=lambda g: (-g.width * g.height, g.name))
    )
    cells = tuple(composites[g.name] for g in ordered)

    board = (
        (board_width, board_height)
        if board_width is not None and board_height is not None
        else None
    )
    placed = _place_greedy(cells, clearance_mm, board, allow_rotation=True)

    # Resolve final outline.
    if board is None:
        xs1, ys1, xs2, ys2 = zip(
            *(polygon_bbox(pc.polygon_in_board()) for pc in placed), strict=True
        )
        x_shift = shrink_margin_mm - min(xs1)
        y_shift = shrink_margin_mm - min(ys1)
        placed = [pc.moved_to(pc.dx + x_shift, pc.dy + y_shift) for pc in placed]
        bw = (max(xs2) - min(xs1)) + 2 * shrink_margin_mm
        bh = (max(ys2) - min(ys1)) + 2 * shrink_margin_mm
    else:
        bw, bh = board

    placed = _snap_pinned_groups(placed, edge_pins, bw, bh)
    return Floorplan(placed=tuple(placed), board_width=bw, board_height=bh)


def _snap_pinned_groups(
    placed: list[PlacedCell],
    edge_pins: dict[str, Edge | None],
    bw: float,
    bh: float,
) -> list[PlacedCell]:
    """Translate groups containing edge-pinned refs flush to their edge.

    The verifier re-checks edge distance afterwards; this snap is a
    solver convenience, not the source of truth.
    """
    from kicad_pipeline.placement_v2.ir import Edge as BoardEdge

    out: list[PlacedCell] = []
    for pc in placed:
        pinned_refs = [r for r in pc.cell.refs if r in edge_pins]
        if not pinned_refs:
            out.append(pc)
            continue
        edge = edge_pins[pinned_refs[0]]
        x1, y1, x2, y2 = polygon_bbox(pc.polygon_in_board())
        if edge is None:
            # Nearest edge by current position.
            dists = {
                BoardEdge.WEST: x1,
                BoardEdge.EAST: bw - x2,
                BoardEdge.NORTH: y1,
                BoardEdge.SOUTH: bh - y2,
            }
            edge = min(sorted(dists, key=lambda e: e.value), key=lambda e: dists[e])
        if edge is BoardEdge.WEST:
            pc = pc.moved_to(pc.dx - (x1 - _EDGE_MARGIN_MM), pc.dy)
        elif edge is BoardEdge.EAST:
            pc = pc.moved_to(pc.dx + (bw - _EDGE_MARGIN_MM - x2), pc.dy)
        elif edge is BoardEdge.NORTH:
            pc = pc.moved_to(pc.dx, pc.dy - (y1 - _EDGE_MARGIN_MM))
        else:  # SOUTH
            pc = pc.moved_to(pc.dx, pc.dy + (bh - _EDGE_MARGIN_MM - y2))
        out.append(pc)
    return out
