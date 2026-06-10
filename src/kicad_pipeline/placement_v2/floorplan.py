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
from kicad_pipeline.placement_v2.ir import Axis, Edge

if TYPE_CHECKING:
    from kicad_pipeline.placement_v2.cells import CardinalRotation, Cell
    from kicad_pipeline.placement_v2.ir import ConstraintSet, SequenceAlong

_GROUP_CLEARANCE_MM = 1.0
_BOARD_CLEARANCE_MM = 2.0
_EDGE_MARGIN_MM = 1.0
_ROTATIONS: tuple[CardinalRotation, ...] = (0, 90, 180, 270)


@dataclass(frozen=True)
class GroupPlan:
    """A packed FeatureBlock: cells in group-local frame + hull bbox.

    ``edge_facing`` records which group-local side carries an edge-
    pinned connector strip (built outermost by construction); board
    packing snaps that side flush to the matching board edge.
    """

    name: str
    cells: tuple[PlacedCell, ...]
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 local
    edge_facing: Edge | None = None

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
    """Bbox of the cell polygon at the given rotation (KiCad convention)."""
    poly = transform_polygon(cell.polygon, 0.0, 0.0, -float(rotation))
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
    return _place_greedy_around(cells, [], clearance, board, allow_rotation)


def _place_greedy_around(
    cells: tuple[Cell, ...],
    preplaced: list[PlacedCell],
    clearance: float,
    board: tuple[float, float] | None,
    allow_rotation: bool,
    outer_limit: tuple[Edge, float] | None = None,
) -> list[PlacedCell]:
    """Greedy packing with immovable pre-placed cells (sequence strips).

    *outer_limit* keeps candidates from extending past an edge-facing
    strip's outer boundary (so connector rows stay outermost).
    """
    placed: list[PlacedCell] = list(preplaced)
    if not cells:
        return placed
    order = sorted(cells, key=lambda c: (-c.area, c.name))
    if not placed:
        first = order[0]
        fx1, fy1, fx2, fy2 = _center_offset(first, 0)
        if board is not None:
            # Anchor the largest cell at the board center.
            bx, by = board[0] / 2, board[1] / 2
            placed.append(
                PlacedCell(first, bx - (fx1 + fx2) / 2, by - (fy1 + fy2) / 2, 0)
            )
        else:
            placed.append(
                PlacedCell(first, -(fx1 + fx2) / 2, -(fy1 + fy2) / 2, 0)
            )
        order = order[1:]

    for cell in order:
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
                if outer_limit is not None:
                    cx1, cy1, cx2, cy2 = polygon_bbox(cand.polygon_in_board())
                    edge_, limit = outer_limit
                    if edge_ is Edge.SOUTH and cy2 > limit + 1e-9:
                        continue
                    if edge_ is Edge.EAST and cx2 > limit + 1e-9:
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


def _strip_extent(
    ref_cells: list[Cell], rotation: CardinalRotation, horizontal: bool,
) -> float:
    """Total along-axis extent of a strip's cells at a uniform rotation."""
    return sum(
        (c.width if horizontal else c.height)
        if rotation in (0, 180) else
        (c.height if horizontal else c.width)
        for c in ref_cells
    )


def _sequence_strips(
    cells: tuple[Cell, ...],
    sequences: tuple[SequenceAlong, ...],
    clearance: float,
    edge_pinned: frozenset[str] = frozenset(),
) -> tuple[list[PlacedCell], set[str], Edge | None]:
    """Arrange sequence-constrained cells as ordered uniform-pitch strips.

    A SequenceAlong whose refs each live in a DISTINCT cell (relay
    array, terminal array) becomes a rigid strip: cells in sequence
    order at uniform pitch, all at the same rotation — the rotation
    (0 or 90) that yields the shorter strip. This is what guarantees
    "K1 K2 K3 K4 in a row" structurally instead of hoping the greedy
    packer discovers it.
    """
    placed: list[PlacedCell] = []
    used: set[str] = set()

    # Resolve each sequence to one distinct cell per ref.
    resolved: list[tuple[SequenceAlong, list[Cell]]] = []
    for seq in sequences:
        ref_cells: list[Cell] = []
        claimed_names = {c.name for r in resolved for c in r[1]}
        ok = True
        for ref in seq.refs:
            owner = next(
                (c for c in cells if ref in c.refs
                 and c.name not in claimed_names and c not in ref_cells),
                None,
            )
            if owner is None:
                ok = False
                break
            ref_cells.append(owner)
        if ok and len(ref_cells) >= 2:
            resolved.append((seq, ref_cells))

    # Ladder pairing: equal-length strips whose elements are pairwise
    # net-connected (relay K_i <-> terminal J_i) share the larger pitch,
    # so element i of each strip lands at the same along-axis coordinate
    # — "the terminal sits under its relay" by construction.
    shared_pitch: dict[int, float] = {}
    naturals: list[float] = []
    for seq, ref_cells in resolved:
        horizontal = seq.axis is Axis.HORIZONTAL
        rotation: CardinalRotation = (
            0 if _strip_extent(ref_cells, 0, horizontal)
            <= _strip_extent(ref_cells, 90, horizontal) else 90
        )
        naturals.append(seq.pitch_mm if seq.pitch_mm is not None else (
            max(
                (c.width if rotation in (0, 180) else c.height)
                if horizontal else
                (c.height if rotation in (0, 180) else c.width)
                for c in ref_cells
            )
            + clearance
        ))
    for i, (seq_a, cells_a) in enumerate(resolved):
        for j in range(i + 1, len(resolved)):
            seq_b, cells_b = resolved[j]
            if len(cells_a) != len(cells_b) or seq_a.axis is not seq_b.axis:
                continue
            if all(
                {p.net for p in ca.ports} & {p.net for p in cb.ports}
                for ca, cb in zip(cells_a, cells_b, strict=True)
            ):
                pitch = max(
                    shared_pitch.get(i, naturals[i]),
                    shared_pitch.get(j, naturals[j]),
                )
                shared_pitch[i] = pitch
                shared_pitch[j] = pitch

    # Edge-pinned strips (connector banks) go LAST so they form the
    # group's outermost row — the side that will be snapped flush to a
    # board edge. They keep native rotation 0 (connector openings are
    # designed outward in the footprint frame).
    def _is_edge_strip(item: tuple[SequenceAlong, list[Cell]]) -> bool:
        return any(
            ref in edge_pinned for c in item[1] for ref in c.refs
        )

    ordered = sorted(
        enumerate(resolved), key=lambda kv: (_is_edge_strip(kv[1]), kv[0])
    )
    facing: Edge | None = None
    cursor_other = 0.0
    for seq_idx, (seq, ref_cells) in ordered:
        horizontal = seq.axis is Axis.HORIZONTAL
        if _is_edge_strip((seq, ref_cells)):
            strip_rot: CardinalRotation = 0
            facing = Edge.SOUTH if horizontal else Edge.EAST
        else:
            strip_rot = (
                0 if _strip_extent(ref_cells, 0, horizontal)
                <= _strip_extent(ref_cells, 90, horizontal) else 90
            )
        pitch = shared_pitch.get(seq_idx, naturals[seq_idx])
        strip_thickness = max(
            (c.height if strip_rot in (0, 180) else c.width)
            if horizontal else
            (c.width if strip_rot in (0, 180) else c.height)
            for c in ref_cells
        )
        base_other = cursor_other + strip_thickness / 2
        cursor_other += strip_thickness + 2 * clearance
        for i, cell in enumerate(ref_cells):
            x1, y1, x2, y2 = _center_offset(cell, strip_rot)
            cx = pitch * i - (x1 + x2) / 2
            cy = base_other - (y1 + y2) / 2
            if not horizontal:
                cx, cy = base_other - (x1 + x2) / 2, pitch * i - (y1 + y2) / 2
            placed.append(PlacedCell(cell, cx, cy, strip_rot))
            used.add(cell.name)
    return placed, used, facing


def pack_group(
    name: str,
    cells: tuple[Cell, ...],
    clearance_mm: float = _GROUP_CLEARANCE_MM,
    sequences: tuple[SequenceAlong, ...] = (),
    edge_pinned: frozenset[str] = frozenset(),
) -> GroupPlan:
    """Pack a FeatureBlock's cells into a compact group-local layout.

    Cross-cell *sequences* (arrays of relay channels, terminal banks)
    are realized as rigid ordered strips first; remaining cells pack
    greedily around them by port wirelength. An edge-pinned strip is
    the group's outermost row and nothing may pack beyond it — that
    side stays clear to meet the board edge.
    """
    strip_cells, used, facing = _sequence_strips(
        cells, sequences, clearance_mm, edge_pinned,
    )
    outer_limit: tuple[Edge, float] | None = None
    if facing is not None and strip_cells:
        bounds = [polygon_bbox(pc.polygon_in_board()) for pc in strip_cells]
        if facing is Edge.SOUTH:
            outer_limit = (facing, max(b[3] for b in bounds))
        else:  # EAST
            outer_limit = (facing, max(b[2] for b in bounds))
    rest = tuple(c for c in cells if c.name not in used)
    placed = _place_greedy_around(
        rest, strip_cells, clearance_mm, board=None, allow_rotation=True,
        outer_limit=outer_limit,
    )
    if not placed:
        return GroupPlan(name=name, cells=(), bbox=(0.0, 0.0, 0.0, 0.0))
    xs1, ys1, xs2, ys2 = zip(
        *(polygon_bbox(pc.polygon_in_board()) for pc in placed), strict=True
    )
    return GroupPlan(
        name=name,
        cells=tuple(placed),
        bbox=(min(xs1), min(ys1), max(xs2), max(ys2)),
        edge_facing=facing,
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

    facing_by_name: dict[str, Edge | None] = {
        f"group:{g.name}": g.edge_facing
        for g in groups if g.edge_facing is not None
    }
    placed = _snap_pinned_groups(placed, edge_pins, facing_by_name, bw, bh)
    return Floorplan(placed=tuple(placed), board_width=bw, board_height=bh)


def _snap_pinned_groups(
    placed: list[PlacedCell],
    edge_pins: dict[str, Edge | None],
    facing_by_name: dict[str, Edge | None],
    bw: float,
    bh: float,
) -> list[PlacedCell]:
    """Translate groups containing edge-pinned refs flush to their edge.

    A group whose plan recorded ``edge_facing`` (its connector strip is
    structurally outermost on that side) snaps that side flush. Others
    fall back to an explicit EdgePin edge or the nearest edge. The
    verifier re-checks edge distance afterwards; this snap is a solver
    convenience, not the source of truth.
    """
    out: list[PlacedCell] = []
    for pc in placed:
        pinned_refs = [r for r in pc.cell.refs if r in edge_pins]
        facing = facing_by_name.get(pc.cell.name)
        if not pinned_refs and facing is None:
            out.append(pc)
            continue
        x1, y1, x2, y2 = polygon_bbox(pc.polygon_in_board())
        edge = facing
        if edge is None and pinned_refs:
            edge = edge_pins[pinned_refs[0]]
        if edge is None:
            # Nearest edge by current position.
            dists = {
                Edge.WEST: x1,
                Edge.EAST: bw - x2,
                Edge.NORTH: y1,
                Edge.SOUTH: bh - y2,
            }
            edge = min(sorted(dists, key=lambda e: e.value), key=lambda e: dists[e])
        if edge is Edge.WEST:
            pc = pc.moved_to(pc.dx - (x1 - _EDGE_MARGIN_MM), pc.dy)
        elif edge is Edge.EAST:
            pc = pc.moved_to(pc.dx + (bw - _EDGE_MARGIN_MM - x2), pc.dy)
        elif edge is Edge.NORTH:
            pc = pc.moved_to(pc.dx, pc.dy - (y1 - _EDGE_MARGIN_MM))
        else:  # SOUTH
            pc = pc.moved_to(pc.dx, pc.dy + (bh - _EDGE_MARGIN_MM - y2))
        out.append(pc)
    return out
