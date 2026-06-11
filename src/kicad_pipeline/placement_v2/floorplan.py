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
from kicad_pipeline.models.pcb import Point
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

_GROUP_CLEARANCE_MM = 0.5
_BOARD_CLEARANCE_MM = 2.0
# Matches the IR's BoardContain margin (0.5mm) — a stricter packing
# margin than the verifier enforces just rejects boards that would pass.
_EDGE_MARGIN_MM = 0.5
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
    # Caller-supplied order is AUTHORITATIVE (pack_board sends lifted
    # connector groups last) — re-sorting here silently defeated it.
    order = list(cells)
    if not placed:
        first = order[0]
        fx1, fy1, fx2, fy2 = _center_offset(first, 0)
        if board is not None and (
            fx2 - fx1 > board[0] - 2 * _EDGE_MARGIN_MM
            or fy2 - fy1 > board[1] - 2 * _EDGE_MARGIN_MM
        ):
            raise PCBError(
                f"floorplan: no feasible slot for cell {first.name!r} "
                f"({fx2 - fx1:.1f}x{fy2 - fy1:.1f}mm) on "
                f"{board[0]}x{board[1]}mm board"
            )
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
        if best is None and board is not None:
            # No overlap-free slot, but the board may still have room
            # once UNPINNED neighbors shift: place at the least-
            # overlapping in-board candidate and let legalization
            # separate them (it reports honestly if it cannot).
            best = _least_overlap_fallback(
                cell, placed, clearance, board,
                _ROTATIONS if allow_rotation else (0,),
            )
        if best is None:
            x1, y1, x2, y2 = _center_offset(cell, 0)
            raise PCBError(
                f"floorplan: no feasible slot for cell {cell.name!r} "
                f"({x2 - x1:.1f}x{y2 - y1:.1f}mm)"
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


def _ladder_facing_rotation(
    ref_cells: list[Cell],
    paired_cells: list[Cell],
    horizontal: bool,
) -> CardinalRotation | None:
    """Rotation pointing ladder-shared ports at the paired connector strip.

    Evaluated over ALL four rotations — choosing the relay's AXIS as
    well as its flip: a relay whose contact pins must reach the
    terminal row needs its long axis perpendicular to that row, even
    when a sideways orientation would give a shorter strip. The shared
    ports' mean coordinate toward the paired strip (stacked at greater
    other-axis position) is maximized; ties prefer the smaller cell
    extent along the strip, then the smaller angle. Returns ``None``
    when the strips share no nets.
    """
    shared = {p.net for p in ref_cells[0].ports} & {
        p.net for p in paired_cells[0].ports
    }
    if not shared:
        return None
    shared_pts = tuple(
        Point(p.x, p.y) for p in ref_cells[0].ports if p.net in shared
    )

    def score(rot: CardinalRotation) -> tuple[float, float, float]:
        pts = transform_polygon(shared_pts, 0.0, 0.0, -float(rot))
        outward = sum(p.y if horizontal else p.x for p in pts) / len(pts)
        extent = _strip_extent(ref_cells, rot, horizontal)
        return (-outward, extent, float(rot))

    best: CardinalRotation = min(_ROTATIONS, key=score)
    return best


def _edge_strip_rotation(
    ref_cells: list[Cell],
    horizontal: bool,
    openings: dict[str, tuple[float, float]],
) -> CardinalRotation:
    """Rotation pointing the connectors' wire openings at the strip's
    facing edge (SOUTH for horizontal strips, EAST for vertical).

    Phoenix-style terminals open toward -Y at rotation 0 and would face
    the board interior on a south edge without this — the opening
    direction comes from part rules (EdgePin.opening) since their
    courtyards are symmetric.
    """
    opening = None
    for c in ref_cells:
        for r in c.refs:
            if r in openings:
                opening = openings[r]
                break
        if opening is not None:
            break
    if opening is None:
        return 0
    target = (0.0, 1.0) if horizontal else (1.0, 0.0)
    best_rot: CardinalRotation = 0
    best_dot = -float("inf")
    for rot in _ROTATIONS:
        pts = transform_polygon(
            (Point(opening[0], opening[1]),), 0.0, 0.0, -float(rot),
        )
        dot = pts[0].x * target[0] + pts[0].y * target[1]
        if dot > best_dot + 1e-9:
            best_dot, best_rot = dot, rot
    return best_rot


def _sequence_strips(
    cells: tuple[Cell, ...],
    sequences: tuple[SequenceAlong, ...],
    clearance: float,
    edge_pinned: frozenset[str] = frozenset(),
    openings: dict[str, tuple[float, float]] | None = None,
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

    def _is_edge_strip(item: tuple[SequenceAlong, list[Cell]]) -> bool:
        return any(
            ref in edge_pinned for c in item[1] for ref in c.refs
        )

    # Ladder pairing FIRST (needs only ports): equal-length strips whose
    # elements are pairwise net-connected (relay K_i <-> terminal J_i).
    ladder_pairs: dict[int, int] = {}
    for i, (seq_a, cells_a) in enumerate(resolved):
        for j in range(i + 1, len(resolved)):
            seq_b, cells_b = resolved[j]
            if len(cells_a) != len(cells_b) or seq_a.axis is not seq_b.axis:
                continue
            if all(
                {p.net for p in ca.ports} & {p.net for p in cb.ports}
                for ca, cb in zip(cells_a, cells_b, strict=True)
            ):
                ladder_pairs[i] = j
                ladder_pairs[j] = i

    # Rotations: edge strips native 0; strips ladder-paired with an
    # edge strip face their shared ports at it (axis AND flip chosen
    # together); the rest minimize extent. Decided before pitch so the
    # natural pitch is computed with the rotation actually used.
    rotations: list[CardinalRotation] = []
    for idx_, (seq, ref_cells) in enumerate(resolved):
        horizontal = seq.axis is Axis.HORIZONTAL
        if _is_edge_strip((seq, ref_cells)):
            rotations.append(_edge_strip_rotation(
                ref_cells, horizontal, openings or {},
            ))
            continue
        pair = ladder_pairs.get(idx_)
        facing_rot: CardinalRotation | None = None
        if pair is not None and _is_edge_strip(resolved[pair]):
            facing_rot = _ladder_facing_rotation(
                ref_cells, resolved[pair][1], horizontal,
            )
        if facing_rot is not None:
            rotations.append(facing_rot)
        else:
            rotations.append(
                0 if _strip_extent(ref_cells, 0, horizontal)
                <= _strip_extent(ref_cells, 90, horizontal) else 90
            )

    # Paired strips share the larger pitch so element i of each strip
    # lands at the same along-axis coordinate — "the terminal sits
    # under its relay" by construction.
    shared_pitch: dict[int, float] = {}
    naturals: list[float] = []
    for idx_, (seq, ref_cells) in enumerate(resolved):
        horizontal = seq.axis is Axis.HORIZONTAL
        rotation = rotations[idx_]
        naturals.append(seq.pitch_mm if seq.pitch_mm is not None else (
            max(
                (c.width if rotation in (0, 180) else c.height)
                if horizontal else
                (c.height if rotation in (0, 180) else c.width)
                for c in ref_cells
            )
            + clearance
        ))
    for i, j in sorted(ladder_pairs.items()):
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
    ordered = sorted(
        enumerate(resolved), key=lambda kv: (_is_edge_strip(kv[1]), kv[0])
    )
    facing: Edge | None = None
    cursor_other = 0.0
    for seq_idx, (seq, ref_cells) in ordered:
        horizontal = seq.axis is Axis.HORIZONTAL
        strip_rot = rotations[seq_idx]
        if _is_edge_strip((seq, ref_cells)):
            facing = Edge.SOUTH if horizontal else Edge.EAST
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


def _least_overlap_fallback(
    cell: Cell,
    placed: list[PlacedCell],
    clearance: float,
    board: tuple[float, float],
    rotations: tuple[CardinalRotation, ...],
) -> PlacedCell | None:
    """Best in-board candidate by (overlap area proxy, HPWL)."""
    bw, bh = board
    best: PlacedCell | None = None
    best_key = (float("inf"), float("inf"))
    for rot in rotations:
        x1, y1, x2, y2 = _center_offset(cell, rot)
        w, h = x2 - x1, y2 - y1
        if w > bw - 2 * _EDGE_MARGIN_MM or h > bh - 2 * _EDGE_MARGIN_MM:
            continue
        slots = [
            *_candidate_offsets(placed, w, h, clearance),
            (_EDGE_MARGIN_MM + w / 2, _EDGE_MARGIN_MM + h / 2),
            (bw - _EDGE_MARGIN_MM - w / 2, _EDGE_MARGIN_MM + h / 2),
            (_EDGE_MARGIN_MM + w / 2, bh - _EDGE_MARGIN_MM - h / 2),
            (bw - _EDGE_MARGIN_MM - w / 2, bh - _EDGE_MARGIN_MM - h / 2),
        ]
        for sx, sy in slots:
            # Clamp the slot center so the cell stays in-board.
            sx = min(max(sx, _EDGE_MARGIN_MM + w / 2), bw - _EDGE_MARGIN_MM - w / 2)
            sy = min(max(sy, _EDGE_MARGIN_MM + h / 2), bh - _EDGE_MARGIN_MM - h / 2)
            cand = PlacedCell(cell, sx - (x1 + x2) / 2, sy - (y1 + y2) / 2, rot)
            cb = polygon_bbox(cand.polygon_in_board())
            overlap = 0.0
            for other in placed:
                ob = polygon_bbox(other.polygon_in_board())
                ow = min(cb[2], ob[2]) - max(cb[0], ob[0])
                oh = min(cb[3], ob[3]) - max(cb[1], ob[1])
                if ow > 0 and oh > 0:
                    overlap += ow * oh
            key = (overlap, _hpwl(_collect_ports(placed, cand)))
            if key < best_key:
                best_key, best = key, cand
    return best


def pack_group(
    name: str,
    cells: tuple[Cell, ...],
    clearance_mm: float = _GROUP_CLEARANCE_MM,
    sequences: tuple[SequenceAlong, ...] = (),
    edge_pinned: frozenset[str] = frozenset(),
    openings: dict[str, tuple[float, float]] | None = None,
) -> GroupPlan:
    """Pack a FeatureBlock's cells into a compact group-local layout.

    Cross-cell *sequences* (arrays of relay channels, terminal banks)
    are realized as rigid ordered strips first; remaining cells pack
    greedily around them by port wirelength. An edge-pinned strip is
    the group's outermost row and nothing may pack beyond it — that
    side stays clear to meet the board edge.
    """
    strip_cells, used, facing = _sequence_strips(
        cells, sequences, clearance_mm, edge_pinned, openings or {},
    )
    outer_limit: tuple[Edge, float] | None = None
    if facing is not None and strip_cells:
        bounds = [polygon_bbox(pc.polygon_in_board()) for pc in strip_cells]
        if facing is Edge.SOUTH:
            outer_limit = (facing, max(b[3] for b in bounds))
        else:  # EAST
            outer_limit = (facing, max(b[2] for b in bounds))
    rest = tuple(sorted(
        (c for c in cells if c.name not in used),
        key=lambda c: (-c.area, c.name),
    ))
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


_EDGE_NORMALS: dict[Edge, tuple[float, float]] = {
    Edge.WEST: (-1.0, 0.0), Edge.EAST: (1.0, 0.0),
    Edge.NORTH: (0.0, -1.0), Edge.SOUTH: (0.0, 1.0),
}


def _aim_at_edge(
    pc: PlacedCell, edge: Edge, bdir: tuple[float, float] | None,
) -> PlacedCell:
    """Rotate a connector cell so its opening (or long axis) faces *edge*."""
    if bdir is not None:
        nx, ny = _EDGE_NORMALS[edge]
        best_rot: CardinalRotation = 0
        best_dot = -float("inf")
        for rot in _ROTATIONS:
            pts = transform_polygon(
                (Point(bdir[0], bdir[1]),), 0.0, 0.0, -float(rot),
            )
            dot = pts[0].x * nx + pts[0].y * ny
            if dot > best_dot + 1e-9:
                best_dot, best_rot = dot, rot
        return pc.rotated(best_rot)
    bb = polygon_bbox(pc.polygon_in_board())
    tall = (bb[3] - bb[1]) > (bb[2] - bb[0])
    wants_horizontal = edge in (Edge.NORTH, Edge.SOUTH)
    if tall == wants_horizontal:
        return pc.rotated(90)
    return pc


def pack_board(
    groups: tuple[GroupPlan, ...],
    constraints: ConstraintSet,
    board_width: float | None = None,
    board_height: float | None = None,
    clearance_mm: float = _BOARD_CLEARANCE_MM,
    shrink_margin_mm: float = 3.0,
    obstacles: tuple[PlacedCell, ...] = (),
    body_dirs: dict[str, tuple[float, float]] | None = None,
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

    board = (
        (board_width, board_height)
        if board_width is not None and board_height is not None
        else None
    )

    # Connectors with an EXPLICIT edge (human lock / part rule) pre-
    # place flush on that edge BEFORE the interior pack, as immovable
    # obstacles — packing the interior first left no coordinated room
    # and legalization pushed groups off the outline (nl-s-3c,
    # 2026-06-11). Shelf order: name-sorted along each edge.
    locked_placed: list[PlacedCell] = []
    locked_names: set[str] = set()
    if board is not None:
        bw0, bh0 = board
        shelf: dict[Edge, float] = dict.fromkeys(Edge, _EDGE_MARGIN_MM)
        for g in sorted(groups, key=lambda g: g.name):
            if not g.name.startswith("conn:"):
                continue
            cell = composites[g.name]
            edge = next(
                (edge_pins[r] for r in cell.refs
                 if edge_pins.get(r) is not None),
                None,
            )
            if edge is None:
                continue
            pc = _aim_at_edge(
                PlacedCell(cell, 0.0, 0.0, 0), edge,
                (body_dirs or {}).get(f"group:{g.name}"),
            )
            bb = polygon_bbox(pc.polygon_in_board())
            horizontal = edge in (Edge.NORTH, Edge.SOUTH)
            span = (bb[2] - bb[0]) if horizontal else (bb[3] - bb[1])
            depth = (bb[3] - bb[1]) if horizontal else (bb[2] - bb[0])
            # Corner-aware: skip past locked cells from PERPENDICULAR
            # edges that reach into this edge's band (the north and
            # west shelves once collided at the NW corner).
            if edge in (Edge.NORTH, Edge.WEST):
                band_lo, band_hi = _EDGE_MARGIN_MM, _EDGE_MARGIN_MM + depth
            elif edge is Edge.EAST:
                band_lo, band_hi = bw0 - _EDGE_MARGIN_MM - depth, bw0 - _EDGE_MARGIN_MM
            else:
                band_lo, band_hi = bh0 - _EDGE_MARGIN_MM - depth, bh0 - _EDGE_MARGIN_MM
            along = shelf[edge]
            for other in locked_placed:
                ob = polygon_bbox(other.polygon_in_board())
                o_perp = (ob[1], ob[3]) if horizontal else (ob[0], ob[2])
                if o_perp[0] >= band_hi + clearance_mm or o_perp[1] <= band_lo - clearance_mm:
                    continue
                o_along = (ob[0], ob[2]) if horizontal else (ob[1], ob[3])
                if o_along[1] + clearance_mm > along and o_along[0] < along + span:
                    along = o_along[1] + clearance_mm
            if horizontal:
                pc = pc.moved_to(pc.dx + (along - bb[0]), pc.dy)
                bb = polygon_bbox(pc.polygon_in_board())
                pc = (pc.moved_to(pc.dx, pc.dy - (bb[1] - _EDGE_MARGIN_MM))
                      if edge is Edge.NORTH else
                      pc.moved_to(pc.dx, pc.dy + (bh0 - _EDGE_MARGIN_MM - bb[3])))
            else:
                pc = pc.moved_to(pc.dx, pc.dy + (along - bb[1]))
                bb = polygon_bbox(pc.polygon_in_board())
                pc = (pc.moved_to(pc.dx - (bb[0] - _EDGE_MARGIN_MM), pc.dy)
                      if edge is Edge.WEST else
                      pc.moved_to(pc.dx + (bw0 - _EDGE_MARGIN_MM - bb[2]), pc.dy))
            shelf[edge] = along + span + clearance_mm
            locked_placed.append(pc)
            locked_names.add(g.name)

    # Functional groups first (largest anchors the interior); lifted
    # connector groups LAST — they end up flush on an edge anyway, so
    # letting a big RJ45 grab the board center starves the real groups.
    ordered = sorted(
        (g for g in groups if f"group:{g.name}" not in locked_names
         and g.name not in locked_names),
        key=lambda g: (
            g.name.startswith("conn:"), -g.width * g.height, g.name,
        ),
    )
    cells = tuple(composites[g.name] for g in ordered)

    # Reserved regions (mounting-hole corners) participate as immovable
    # pre-placed cells; they are only meaningful with explicit board dims.
    pre = (list(obstacles) + locked_placed) if board is not None else []
    placed = _place_greedy_around(
        cells, pre, clearance_mm, board, allow_rotation=True,
    )

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
    placed = _snap_pinned_groups(
        placed, edge_pins, facing_by_name, bw, bh, body_dirs or {},
        final_names=frozenset(f"group:{n}" for n in locked_names),
    )
    return Floorplan(placed=tuple(placed), board_width=bw, board_height=bh)


def _snap_pinned_groups(
    placed: list[PlacedCell],
    edge_pins: dict[str, Edge | None],
    facing_by_name: dict[str, Edge | None],
    bw: float,
    bh: float,
    body_dirs: dict[str, tuple[float, float]] | None = None,
    final_names: frozenset[str] = frozenset(),
) -> list[PlacedCell]:
    """Translate groups containing edge-pinned refs flush to their edge.

    A group whose plan recorded ``edge_facing`` (its connector strip is
    structurally outermost on that side) snaps that side flush. Others
    fall back to an explicit EdgePin edge or the nearest edge. The
    verifier re-checks edge distance afterwards; this snap is a solver
    convenience, not the source of truth.
    """
    def _snap_to(pc: PlacedCell, edge: Edge) -> PlacedCell:
        x1, y1, x2, y2 = polygon_bbox(pc.polygon_in_board())
        if edge is Edge.WEST:
            return pc.moved_to(pc.dx - (x1 - _EDGE_MARGIN_MM), pc.dy)
        if edge is Edge.EAST:
            return pc.moved_to(pc.dx + (bw - _EDGE_MARGIN_MM - x2), pc.dy)
        if edge is Edge.NORTH:
            return pc.moved_to(pc.dx, pc.dy - (y1 - _EDGE_MARGIN_MM))
        return pc.moved_to(pc.dx, pc.dy + (bh - _EDGE_MARGIN_MM - y2))

    def _edge_load(edge: Edge, claims: dict[Edge, float]) -> float:
        return claims.get(edge, 0.0)

    # Two passes: functional groups (own their edge span) first, then
    # lifted connector groups pick the nearest UNCONGESTED edge — a
    # power group covering the whole south edge means the pin headers
    # belong on east/west, not wedged into it.
    out: list[PlacedCell] = list(placed)
    snapped_edge: dict[str, Edge] = {}
    edge_claims: dict[Edge, float] = {}
    movable = [
        i for i in range(len(out))
        if out[i].cell.name.startswith("group:conn:")
        and out[i].cell.name not in final_names  # pre-placed locked cells
    ]
    preferred: dict[str, Edge] = {}
    order = sorted(
        (i for i in range(len(out))
         if i not in movable and out[i].cell.name not in final_names),
        key=lambda i: out[i].cell.name,
    )
    for i in order:
        pc = out[i]
        pinned_refs = [r for r in pc.cell.refs if r in edge_pins]
        facing = facing_by_name.get(pc.cell.name)
        if not pinned_refs and facing is None:
            continue
        x1, y1, x2, y2 = polygon_bbox(pc.polygon_in_board())
        edge = facing
        if edge is None and pinned_refs:
            edge = edge_pins[pinned_refs[0]]
        if edge is None:
            spans = {
                Edge.WEST: y2 - y1, Edge.EAST: y2 - y1,
                Edge.NORTH: x2 - x1, Edge.SOUTH: x2 - x1,
            }
            limits = {
                Edge.WEST: bh, Edge.EAST: bh,
                Edge.NORTH: bw, Edge.SOUTH: bw,
            }
            dists = {
                Edge.WEST: x1, Edge.EAST: bw - x2,
                Edge.NORTH: y1, Edge.SOUTH: bh - y2,
            }
            edge = min(
                sorted(Edge, key=lambda e: e.value),
                key=lambda e: (
                    # Congested edge (claims + me would overflow): last.
                    _edge_load(e, edge_claims) + spans[e]
                    > limits[e] - 2 * _EDGE_MARGIN_MM,
                    dists[e],
                ),
            )
        out[i] = _snap_to(pc, edge)
        snapped_edge[out[i].cell.name] = edge
        b = polygon_bbox(out[i].polygon_in_board())
        span = (b[2] - b[0]) if edge in (Edge.NORTH, Edge.SOUTH) else (b[3] - b[1])
        edge_claims[edge] = edge_claims.get(edge, 0.0) + span + _BOARD_CLEARANCE_MM
    # Explicit EdgePin edges become the connectors' preferred edges.
    for i in movable:
        for r in out[i].cell.refs:
            e = edge_pins.get(r)
            if e is not None:
                preferred[out[i].cell.name] = e
    pinned_names = set(snapped_edge) | set(final_names) | {
        out[i].cell.name for i in range(len(out))
        if out[i].cell.name.startswith("reserved:")
    }
    return _place_conn_groups(
        out, movable, body_dirs or {}, preferred, bw, bh, pinned_names,
    )


def _place_conn_groups(
    cells: list[PlacedCell],
    movable: list[int],
    body_dirs: dict[str, tuple[float, float]],
    preferred: dict[str, Edge],
    bw: float,
    bh: float,
    pinned_names: frozenset[str] | set[str] = frozenset(),
) -> list[PlacedCell]:
    """Place each lifted connector group flush on an edge with room.

    For every movable connector (deterministic order): try its
    preferred edge, then the others by current distance; on each edge,
    rotate the opening outward (body bulge -> edge normal) and search
    for the free interval nearest its current position against ALL
    other cells. The first edge with room wins. A connector that fits
    NOWHERE stays put — legalization reports it honestly.
    """
    normals = {
        Edge.WEST: (-1.0, 0.0), Edge.EAST: (1.0, 0.0),
        Edge.NORTH: (0.0, -1.0), Edge.SOUTH: (0.0, 1.0),
    }
    for i in sorted(movable, key=lambda i: cells[i].cell.name):
        b0 = polygon_bbox(cells[i].polygon_in_board())
        dists = {
            Edge.WEST: b0[0], Edge.EAST: bw - b0[2],
            Edge.NORTH: b0[1], Edge.SOUTH: bh - b0[3],
        }
        pref = preferred.get(cells[i].cell.name)
        bdir0 = body_dirs.get(cells[i].cell.name)

        def _needs_rotation(e: Edge, _bdir0: tuple[float, float] | None = bdir0) -> bool:
            """Does facing *e* require rotating away from the current pose?

            Rotation stability: an ESP32 whose antenna already points
            north should claim the north edge over a nearer west edge —
            rotating flips its pin geometry relative to every already-
            ordered header pinout (mcu_core regression, 2026-06-11).
            """
            if _bdir0 is None:
                return False
            nx, ny = normals[e]
            return _bdir0[0] * nx + _bdir0[1] * ny < 1e-9

        edges = sorted(
            Edge, key=lambda e: (e is not pref, _needs_rotation(e), dists[e], e.value),
        )
        for edge in edges:
            pc = cells[i]
            bdir = body_dirs.get(pc.cell.name)
            if bdir is not None:
                nx, ny = normals[edge]
                best_rot: CardinalRotation = 0
                best_dot = -float("inf")
                for rot in _ROTATIONS:
                    pts = transform_polygon(
                        (Point(bdir[0], bdir[1]),), 0.0, 0.0, -float(rot),
                    )
                    dot = pts[0].x * nx + pts[0].y * ny
                    if dot > best_dot + 1e-9:
                        best_dot, best_rot = dot, rot
                pc = pc.rotated(best_rot)
            else:
                # No opening (symmetric pin header): lay the LONG axis
                # along the edge — a 1x14 header once sat flush south
                # but rotated perpendicular, jutting 37mm into the
                # board (nl-s-3c, 2026-06-11).
                bb0 = polygon_bbox(pc.polygon_in_board())
                tall = (bb0[3] - bb0[1]) > (bb0[2] - bb0[0])
                wants_horizontal = edge in (Edge.NORTH, Edge.SOUTH)
                if tall == wants_horizontal:
                    pc = pc.rotated(90)
            b = polygon_bbox(pc.polygon_in_board())
            horizontal = edge in (Edge.NORTH, Edge.SOUTH)
            span = (b[2] - b[0]) if horizontal else (b[3] - b[1])
            depth = (b[3] - b[1]) if horizontal else (b[2] - b[0])
            limit = bw if horizontal else bh
            # Perpendicular band this connector will occupy at the edge.
            if edge is Edge.WEST:
                band = (_EDGE_MARGIN_MM, _EDGE_MARGIN_MM + depth)
            elif edge is Edge.EAST:
                band = (bw - _EDGE_MARGIN_MM - depth, bw - _EDGE_MARGIN_MM)
            elif edge is Edge.NORTH:
                band = (_EDGE_MARGIN_MM, _EDGE_MARGIN_MM + depth)
            else:
                band = (bh - _EDGE_MARGIN_MM - depth, bh - _EDGE_MARGIN_MM)
            blocked: list[tuple[float, float]] = []
            for j, other in enumerate(cells):
                if j == i:
                    continue
                ob = polygon_bbox(other.polygon_in_board())
                o_perp = (ob[1], ob[3]) if horizontal else (ob[0], ob[2])
                if (o_perp[0] >= band[1] + _BOARD_CLEARANCE_MM
                        or o_perp[1] <= band[0] - _BOARD_CLEARANCE_MM):
                    continue
                o_along = (ob[0], ob[2]) if horizontal else (ob[1], ob[3])
                blocked.append((o_along[0] - _BOARD_CLEARANCE_MM,
                                o_along[1] + _BOARD_CLEARANCE_MM))
            blocked.sort()
            gaps: list[tuple[float, float]] = []
            cursor = _EDGE_MARGIN_MM
            for b_lo, b_hi in blocked:
                if b_lo > cursor:
                    gaps.append((cursor, min(b_lo, limit - _EDGE_MARGIN_MM)))
                cursor = max(cursor, b_hi)
            if cursor < limit - _EDGE_MARGIN_MM:
                gaps.append((cursor, limit - _EDGE_MARGIN_MM))
            cur_lo = b[0] if horizontal else b[1]
            best_pos: float | None = None
            best_dist = float("inf")
            for g_lo, g_hi in gaps:
                if g_hi - g_lo < span:
                    continue
                pos = min(max(cur_lo, g_lo), g_hi - span)
                if abs(pos - cur_lo) < best_dist - 1e-9:
                    best_dist, best_pos = abs(pos - cur_lo), pos
            if best_pos is None:
                # An EXPLICIT preferred edge (human lock / part rule)
                # is hard: force-place flush there at the nearest
                # along-position — the overlap is pushed onto INTERIOR
                # groups by legalization (conn groups are pinned).
                # Politely scanning other edges once sent a locked-
                # south RJ45 to the north edge because the interior
                # pack already covered the south band (nl-s-3c,
                # 2026-06-11). Without a preference, scan all edges
                # and only force on the nearest as a last resort.
                if pref is not None and edge is not pref:
                    continue
                if pref is None and edge is not edges[-1]:
                    continue  # no room on this edge: try the next
                if pref is None:
                    edge = edges[0]
                    pc = cells[i]
                    bdir2 = body_dirs.get(pc.cell.name)
                    if bdir2 is not None:
                        nx, ny = normals[edge]
                        fb_rot: CardinalRotation = 0
                        fb_dot = -float("inf")
                        for rot in _ROTATIONS:
                            pts = transform_polygon(
                                (Point(bdir2[0], bdir2[1]),), 0.0, 0.0, -float(rot),
                            )
                            dot = pts[0].x * nx + pts[0].y * ny
                            if dot > fb_dot + 1e-9:
                                fb_dot, fb_rot = dot, rot
                        pc = pc.rotated(fb_rot)
                    b = polygon_bbox(pc.polygon_in_board())
                    horizontal = edge in (Edge.NORTH, Edge.SOUTH)
                    span = (b[2] - b[0]) if horizontal else (b[3] - b[1])
                    depth = (b[3] - b[1]) if horizontal else (b[2] - b[0])
                    limit = bw if horizontal else bh
                    cur_lo = b[0] if horizontal else b[1]
                    if edge is Edge.WEST or edge is Edge.NORTH:
                        band = (_EDGE_MARGIN_MM, _EDGE_MARGIN_MM + depth)
                    elif edge is Edge.EAST:
                        band = (bw - _EDGE_MARGIN_MM - depth, bw - _EDGE_MARGIN_MM)
                    else:
                        band = (bh - _EDGE_MARGIN_MM - depth, bh - _EDGE_MARGIN_MM)
                # Forced placement coordinates ONLY with fellow pinned
                # cells (other connectors, snapped strips, reserved
                # corners) — interior groups yield via legalization.
                # Blocking on interior groups made every forced south
                # cell clamp to the same spot and overlap its
                # neighbors (nl-s-3c, 2026-06-11).
                pinned_blocked: list[tuple[float, float]] = []
                for j, other in enumerate(cells):
                    if j == i:
                        continue
                    if j not in movable and other.cell.name not in pinned_names:
                        continue
                    ob = polygon_bbox(other.polygon_in_board())
                    o_perp = (ob[1], ob[3]) if horizontal else (ob[0], ob[2])
                    if (o_perp[0] >= band[1] + _BOARD_CLEARANCE_MM
                            or o_perp[1] <= band[0] - _BOARD_CLEARANCE_MM):
                        continue
                    o_along = (ob[0], ob[2]) if horizontal else (ob[1], ob[3])
                    pinned_blocked.append((o_along[0] - _BOARD_CLEARANCE_MM,
                                           o_along[1] + _BOARD_CLEARANCE_MM))
                pinned_blocked.sort()
                p_gaps: list[tuple[float, float]] = []
                cursor2 = _EDGE_MARGIN_MM
                for b_lo, b_hi in pinned_blocked:
                    if b_lo > cursor2:
                        p_gaps.append((cursor2, min(b_lo, limit - _EDGE_MARGIN_MM)))
                    cursor2 = max(cursor2, b_hi)
                if cursor2 < limit - _EDGE_MARGIN_MM:
                    p_gaps.append((cursor2, limit - _EDGE_MARGIN_MM))
                for g_lo, g_hi in p_gaps:
                    if g_hi - g_lo < span:
                        continue
                    pos = min(max(cur_lo, g_lo), g_hi - span)
                    if best_pos is None or abs(pos - cur_lo) < abs(best_pos - cur_lo):
                        best_pos = pos
                if best_pos is None:
                    best_pos = min(
                        max(cur_lo, _EDGE_MARGIN_MM),
                        limit - _EDGE_MARGIN_MM - span,
                    )
            # Flush at the edge, at the chosen along-position.
            if horizontal:
                pc = pc.moved_to(pc.dx + (best_pos - b[0]), pc.dy)
                b = polygon_bbox(pc.polygon_in_board())
                pc = (pc.moved_to(pc.dx, pc.dy - (b[1] - _EDGE_MARGIN_MM))
                      if edge is Edge.NORTH else
                      pc.moved_to(pc.dx, pc.dy + (bh - _EDGE_MARGIN_MM - b[3])))
            else:
                pc = pc.moved_to(pc.dx, pc.dy + (best_pos - b[1]))
                b = polygon_bbox(pc.polygon_in_board())
                pc = (pc.moved_to(pc.dx - (b[0] - _EDGE_MARGIN_MM), pc.dy)
                      if edge is Edge.WEST else
                      pc.moved_to(pc.dx + (bw - _EDGE_MARGIN_MM - b[2]), pc.dy))
            cells[i] = pc
            break
    return cells
