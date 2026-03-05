"""Grid-based PCB autorouter using A* pathfinding.

Provides a simple 2-D occupancy grid and an A* path finder that can route
copper tracks between pad pairs on a single copper layer.  Intended as a
lightweight fallback when FreeRouting is unavailable.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kicad_pipeline.constants import JLCPCB_BOARD_EDGE_CLEARANCE_MM
from kicad_pipeline.models.pcb import Footprint, Pad, Point, Track, Via


def _pad_abs_pos(fp: Footprint, pad: Pad) -> tuple[float, float]:
    """Compute absolute pad position accounting for footprint rotation."""
    rad = math.radians(fp.rotation)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    rx = pad.position.x * cos_r - pad.position.y * sin_r
    ry = pad.position.x * sin_r + pad.position.y * cos_r
    return (fp.position.x + rx, fp.position.y + ry)

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Keepout
    from kicad_pipeline.pcb.netlist import Netlist, NetlistEntry


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteRequest:
    """A single net-routing request: connect all pads in pad_refs."""

    net_number: int
    net_name: str
    pad_refs: tuple[tuple[str, str], ...]  # ((ref, pad_num), ...)
    layer: str  # "F.Cu" or "B.Cu"
    width_mm: float = 0.25
    clearance_mm: float = 0.2


@dataclass(frozen=True)
class RouteResult:
    """Result of routing a single net."""

    net_number: int
    net_name: str
    tracks: tuple[Track, ...]
    vias: tuple[Via, ...]
    routed: bool  # True if all connections were made
    reason: str = ""  # failure reason if not routed


# ---------------------------------------------------------------------------
# Internal grid
# ---------------------------------------------------------------------------


@dataclass
class _Grid:
    """2-D occupancy grid for routing.

    Cells are addressed by their millimetre coordinate; internally stored as
    a flat list indexed by (col, row) where col = round(x / step) and
    row = round(y / step).
    """

    cols: int
    rows: int
    grid_step_mm: float
    _cells: list[list[bool]] = field(default_factory=list)  # _cells[col][row]

    def __post_init__(self) -> None:
        if not self._cells:
            self._cells = [[False] * self.rows for _ in range(self.cols)]

    # ------------------------------------------------------------------
    # Class method constructor
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        width_mm: float,
        height_mm: float,
        grid_step_mm: float = 0.5,
    ) -> _Grid:
        """Allocate a new grid; all cells start unoccupied."""
        cols = max(1, int(width_mm / grid_step_mm) + 1)
        rows = max(1, int(height_mm / grid_step_mm) + 1)
        return cls(cols=cols, rows=rows, grid_step_mm=grid_step_mm)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def to_cell(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        """Return (col, row) clamped to valid grid bounds."""
        col = int(x_mm / self.grid_step_mm)
        row = int(y_mm / self.grid_step_mm)
        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))
        return col, row

    def to_mm(self, col: int, row: int) -> tuple[float, float]:
        """Return (x_mm, y_mm) for a given cell index."""
        return col * self.grid_step_mm, row * self.grid_step_mm

    # ------------------------------------------------------------------
    # Cell operations
    # ------------------------------------------------------------------

    def is_free(self, col: int, row: int) -> bool:
        """Return True if in bounds and not occupied."""
        if col < 0 or col >= self.cols or row < 0 or row >= self.rows:
            return False
        return not self._cells[col][row]

    def mark(self, col: int, row: int) -> None:
        """Mark cell occupied if in bounds."""
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self._cells[col][row] = True

    def unmark(self, col: int, row: int) -> None:
        """Mark cell free if in bounds."""
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self._cells[col][row] = False

    def mark_mm(self, x_mm: float, y_mm: float, radius_cells: int = 1) -> None:
        """Mark cell and all neighbors within radius_cells (Manhattan distance)."""
        base_col, base_row = self.to_cell(x_mm, y_mm)
        for dc in range(-radius_cells, radius_cells + 1):
            for dr in range(-radius_cells, radius_cells + 1):
                self.mark(base_col + dc, base_row + dr)

    def unmark_mm(self, x_mm: float, y_mm: float, radius_cells: int = 1) -> None:
        """Unmark cell and all neighbors within radius_cells (Manhattan distance)."""
        base_col, base_row = self.to_cell(x_mm, y_mm)
        for dc in range(-radius_cells, radius_cells + 1):
            for dr in range(-radius_cells, radius_cells + 1):
                self.unmark(base_col + dc, base_row + dr)


# ---------------------------------------------------------------------------
# Grid preparation helpers
# ---------------------------------------------------------------------------


_PAD_CLEARANCE_MM: float = 0.2
"""Routing clearance around pads in mm (matches KiCad default netclass)."""

_KEEPOUT_MARGIN_CELLS: int = 2
"""Extra grid cells marked around keepout zone bounding boxes."""


def _mark_pad_area(
    grid: _Grid,
    px: float,
    py: float,
    half_w: float,
    half_h: float,
    clearance_mm: float = _PAD_CLEARANCE_MM,
) -> None:
    """Mark a rectangular pad area + clearance on the routing grid."""
    x0 = px - half_w - clearance_mm
    y0 = py - half_h - clearance_mm
    x1 = px + half_w + clearance_mm
    y1 = py + half_h + clearance_mm
    c0, r0 = grid.to_cell(x0, y0)
    c1, r1 = grid.to_cell(x1, y1)
    for cc in range(max(0, c0), min(grid.cols, c1 + 1)):
        for rr in range(max(0, r0), min(grid.rows, r1 + 1)):
            grid.mark(cc, rr)


def _unmark_pad_area(
    grid: _Grid,
    px: float,
    py: float,
    half_w: float,
    half_h: float,
    clearance_mm: float = _PAD_CLEARANCE_MM,
) -> None:
    """Unmark a pad area + clearance so the router can reach and exit it."""
    x0 = px - half_w - clearance_mm
    y0 = py - half_h - clearance_mm
    x1 = px + half_w + clearance_mm
    y1 = py + half_h + clearance_mm
    c0, r0 = grid.to_cell(x0, y0)
    c1, r1 = grid.to_cell(x1, y1)
    for cc in range(max(0, c0), min(grid.cols, c1 + 1)):
        for rr in range(max(0, r0), min(grid.rows, r1 + 1)):
            grid.unmark(cc, rr)


def _restore_pad_marks(
    grid: _Grid,
    footprints: list[Footprint],
    net_clearances: dict[str, float] | None = None,
) -> None:
    """Re-mark all pad areas after temporarily clearing same-net pads.

    This prevents cross-net contamination: when clearing pad A's clearance
    zone for routing, nearby pad B's zone might also get cleared.  After
    routing, this function restores ALL pad marks with correct clearances.
    """
    _half_track_width = 0.125
    for fp in footprints:
        for pad in fp.pads:
            px, py = _pad_abs_pos(fp, pad)
            cl = _PAD_CLEARANCE_MM + _half_track_width
            if net_clearances is not None and pad.net_name:
                cl = max(cl, net_clearances.get(pad.net_name, cl) + _half_track_width)
            _mark_pad_area(grid, px, py, pad.size_x / 2, pad.size_y / 2, cl)


def _prepare_grid(
    grid: _Grid,
    footprints: list[Footprint],
    keepouts: tuple[Keepout, ...] = (),
    net_clearances: dict[str, float] | None = None,
) -> None:
    """Mark pad positions, board-edge margins, and keepout zones on the grid.

    This is called once during grid creation to establish the base occupancy
    before any routing begins.

    Args:
        grid: The grid to prepare.
        footprints: All footprints on the board (pads are marked occupied).
        keepouts: Keepout zones whose areas are marked occupied.
        net_clearances: Optional per-net clearance overrides for pad marking.
    """
    # Mark all pad areas with their actual size + clearance margin.
    # Clearance includes half the default track width (0.125mm) because
    # KiCad measures clearance from copper edge to copper edge, not from
    # track center to pad edge.
    _half_track_width = 0.125  # half of default 0.25mm track
    for fp in footprints:
        for pad in fp.pads:
            px, py = _pad_abs_pos(fp, pad)
            cl = _PAD_CLEARANCE_MM + _half_track_width
            if net_clearances is not None and pad.net_name:
                cl = max(cl, net_clearances.get(pad.net_name, cl) + _half_track_width)
            _mark_pad_area(grid, px, py, pad.size_x / 2, pad.size_y / 2, cl)

    # Mark board-edge margins as occupied
    margin_cells = max(1, int(JLCPCB_BOARD_EDGE_CLEARANCE_MM / grid.grid_step_mm) + 1)
    for col in range(grid.cols):
        for mr in range(margin_cells):
            grid.mark(col, mr)                       # top edge
            grid.mark(col, grid.rows - 1 - mr)       # bottom edge
    for row in range(grid.rows):
        for mc in range(margin_cells):
            grid.mark(mc, row)                        # left edge
            grid.mark(grid.cols - 1 - mc, row)        # right edge

    # Mark keepout zones as occupied (with extra margin for hole_clearance)
    for ko in keepouts:
        if not ko.polygon:
            continue
        ko_xs = [p.x for p in ko.polygon]
        ko_ys = [p.y for p in ko.polygon]
        min_col, min_row = grid.to_cell(min(ko_xs), min(ko_ys))
        max_col, max_row = grid.to_cell(max(ko_xs), max(ko_ys))
        for kc in range(
            max(0, min_col - _KEEPOUT_MARGIN_CELLS),
            min(grid.cols, max_col + _KEEPOUT_MARGIN_CELLS + 1),
        ):
            for kr in range(
                max(0, min_row - _KEEPOUT_MARGIN_CELLS),
                min(grid.rows, max_row + _KEEPOUT_MARGIN_CELLS + 1),
            ):
                grid.mark(kc, kr)


# ---------------------------------------------------------------------------
# A* pathfinder
# ---------------------------------------------------------------------------


def _astar(
    grid: _Grid,
    start_col: int,
    start_row: int,
    goal_col: int,
    goal_row: int,
) -> list[tuple[int, int]] | None:
    """Find a path from start to goal on the grid using A*.

    A cell is traversable if:
    - it is free (grid.is_free), OR
    - it equals the start (start_col, start_row), OR
    - it equals the goal (goal_col, goal_row).

    Args:
        grid: The occupancy grid.
        start_col: Start column index.
        start_row: Start row index.
        goal_col: Goal column index.
        goal_row: Goal row index.

    Returns:
        Ordered list of (col, row) cells from start to goal inclusive,
        or None if no path exists.
    """

    def heuristic(c: int, r: int) -> int:
        return abs(c - goal_col) + abs(r - goal_row)

    # Priority queue: (f, g, col, row)
    open_heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(open_heap, (float(heuristic(start_col, start_row)), 0.0, start_col, start_row))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {(start_col, start_row): 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _f, g, col, row = heapq.heappop(open_heap)
        node = (col, row)

        if node in closed:
            continue
        closed.add(node)

        if col == goal_col and row == goal_row:
            # Reconstruct path
            path: list[tuple[int, int]] = [node]
            cur = node
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        # 4-directional neighbors
        for dc, dr in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nc, nr = col + dc, row + dr
            if nc < 0 or nc >= grid.cols or nr < 0 or nr >= grid.rows:
                continue
            neighbor = (nc, nr)
            if neighbor in closed:
                continue
            is_start_or_goal = (
                (nc == start_col and nr == start_row)
                or (nc == goal_col and nr == goal_row)
            )
            if not (grid.is_free(nc, nr) or is_start_or_goal):
                continue
            tentative_g = g + 1.0
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = node
                f = tentative_g + heuristic(nc, nr)
                heapq.heappush(open_heap, (f, tentative_g, nc, nr))

    return None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PadInfo:
    """Resolved pad position and dimensions for routing."""

    x: float
    y: float
    half_w: float
    half_h: float


def _resolve_pad_positions(
    request: RouteRequest,
    fp_by_ref: dict[str, Footprint],
) -> list[_PadInfo] | str:
    """Resolve pad world positions and sizes for a route request.

    Args:
        request: The routing request.
        fp_by_ref: Lookup from ref to Footprint.

    Returns:
        List of :class:`_PadInfo`, or an error string if resolution fails.
    """
    pads: list[_PadInfo] = []
    for ref, pad_num in request.pad_refs:
        found_fp = fp_by_ref.get(ref)
        if found_fp is None:
            return f"Footprint '{ref}' not found"
        found_pad: _PadInfo | None = None
        for pad in found_fp.pads:
            if pad.number == pad_num:
                px, py = _pad_abs_pos(found_fp, pad)
                found_pad = _PadInfo(
                    x=px, y=py,
                    half_w=pad.size_x / 2,
                    half_h=pad.size_y / 2,
                )
                break
        if found_pad is None:
            return f"Pad '{pad_num}' not found on footprint '{ref}'"
        pads.append(found_pad)
    return pads


def route_net(
    request: RouteRequest,
    footprints: list[Footprint],
    board_width_mm: float,
    board_height_mm: float,
    grid_step_mm: float = 0.5,
    grid: _Grid | None = None,
    keepouts: tuple[Keepout, ...] = (),
    net_clearances: dict[str, float] | None = None,
) -> RouteResult:
    """Route a single net using A* on a 2-D occupancy grid.

    When *grid* is provided, it is used as a shared occupancy grid (for
    multi-net routing where each net's tracks become obstacles for
    subsequent nets).  Same-net pad cells are temporarily unmarked
    before routing so the router can reach its own target pads.

    Args:
        request: The routing request describing the net and its pads.
        footprints: All footprints on the board (used to build occupancy).
        board_width_mm: Board width in mm (defines grid extent).
        board_height_mm: Board height in mm (defines grid extent).
        grid_step_mm: Grid resolution in mm.
        grid: Optional shared grid.  When ``None``, a fresh grid is created
            with all pad positions marked as occupied.
        keepouts: Keepout zones to avoid (only used when creating fresh grid).

    Returns:
        A RouteResult with all generated tracks (or failure info).
    """
    if grid is None:
        grid = _Grid.create(board_width_mm, board_height_mm, grid_step_mm)
        _prepare_grid(grid, list(footprints), keepouts=keepouts)

    # Build a lookup: ref -> Footprint
    fp_by_ref: dict[str, Footprint] = {fp.ref: fp for fp in footprints}

    # Resolve pad world positions and sizes
    resolved = _resolve_pad_positions(request, fp_by_ref)
    if isinstance(resolved, str):
        return RouteResult(
            net_number=request.net_number,
            net_name=request.net_name,
            tracks=(),
            vias=(),
            routed=False,
            reason=resolved,
        )
    pad_infos = resolved

    if len(pad_infos) < 2:
        return RouteResult(
            net_number=request.net_number,
            net_name=request.net_name,
            tracks=(),
            vias=(),
            routed=False,
            reason="insufficient pad positions",
        )

    # Temporarily unmark same-net pad areas (including clearance) so the
    # router can reach and exit them.  After routing, ALL pad areas are
    # restored to prevent cross-net contamination.
    _htw = 0.125  # half default track width, must match _prepare_grid
    pad_cl = max(_PAD_CLEARANCE_MM + _htw, request.clearance_mm + _htw)
    for pi in pad_infos:
        _unmark_pad_area(grid, pi.x, pi.y, pi.half_w, pi.half_h, pad_cl)

    # Track which components have THT pads that might need sibling
    # unmark if routing fails (deferred to retry within the routing loop).
    _tht_refs_in_net: set[str] = set()
    for ref, _ in request.pad_refs:
        fp = fp_by_ref.get(ref)
        if fp is None or len(fp.pads) > 12:
            continue
        if any(p.size_x > 1.0 or p.size_y > 1.0 for p in fp.pads):
            _tht_refs_in_net.add(ref)

    # Dense IC handling: fine-pitch ICs have pad clearance zones that
    # leave no routing space between pins.  We exclude these pads from
    # routing and only route the passive-to-passive connections.
    # Detection: >=6 pads AND minimum pad spacing < 1.0mm (fine pitch).
    ic_refs_in_net: set[str] = set()
    for ref, _ in request.pad_refs:
        fp = fp_by_ref.get(ref)
        if fp is None or len(fp.pads) < 6:
            continue
        # Check minimum pad spacing to distinguish fine-pitch ICs from
        # DIP switches, connectors, etc.
        positions = sorted(
            (p.position.x, p.position.y) for p in fp.pads
        )
        min_spacing = 999.0
        for i in range(len(positions) - 1):
            dx = abs(positions[i + 1][0] - positions[i][0])
            dy = abs(positions[i + 1][1] - positions[i][1])
            d = (dx * dx + dy * dy) ** 0.5
            if d > 0.01:
                min_spacing = min(min_spacing, d)
        if min_spacing < 1.0:
            ic_refs_in_net.add(ref)
    if ic_refs_in_net:
        # Remove IC pads from routing targets — keep only non-IC pads
        non_ic_infos = [
            pi for pi, (ref, _) in zip(pad_infos, request.pad_refs, strict=True)
            if ref not in ic_refs_in_net
        ]
        if len(non_ic_infos) >= 2:
            pad_infos = non_ic_infos
            # Unmark IC pad areas so non-IC pads near the IC can route
            # around it without being blocked by its clearance zones.
            for ic_ref in ic_refs_in_net:
                fp = fp_by_ref[ic_ref]
                for pad in fp.pads:
                    px, py = _pad_abs_pos(fp, pad)
                    _unmark_pad_area(
                        grid, px, py,
                        pad.size_x / 2, pad.size_y / 2, pad_cl,
                    )
        elif len(non_ic_infos) == 1:
            # One non-IC pad + IC pads: keep IC pads in routing targets
            # but unmark their clearance zones so A* can reach them.
            for ic_ref in ic_refs_in_net:
                fp = fp_by_ref[ic_ref]
                for pad in fp.pads:
                    px, py = _pad_abs_pos(fp, pad)
                    _unmark_pad_area(
                        grid, px, py,
                        pad.size_x / 2, pad.size_y / 2, pad_cl,
                    )
        elif len(non_ic_infos) == 0:
            # All pads on dense ICs: skip routing entirely
            _restore_pad_marks(grid, footprints, net_clearances)
            return RouteResult(
                net_number=request.net_number,
                net_name=request.net_name,
                tracks=(),
                vias=(),
                routed=False,
                reason="all pads on dense ICs - needs via routing",
            )

    all_tracks: list[Track] = []

    # Track exclusion: sized per-net to satisfy the netclass clearance.
    # Default (0.2mm) -> 1 cell (0.25mm), HVA (0.3mm) -> 2 cells (0.5mm).
    excl_cells = max(1, math.ceil(request.clearance_mm / grid.grid_step_mm))

    # MST-style routing: always connect closest unrouted pad to routed set
    routed_set: set[int] = {0}  # index into pad_infos
    unrouted: set[int] = set(range(1, len(pad_infos)))

    while unrouted:
        # Find the closest (routed, unrouted) pair by Manhattan distance
        best_from = 0
        best_to = next(iter(unrouted))
        best_dist = float("inf")
        for ri in routed_set:
            for ui in unrouted:
                d = (abs(pad_infos[ri].x - pad_infos[ui].x)
                     + abs(pad_infos[ri].y - pad_infos[ui].y))
                if d < best_dist:
                    best_dist = d
                    best_from = ri
                    best_to = ui

        p1 = pad_infos[best_from]
        p2 = pad_infos[best_to]

        start_col, start_row = grid.to_cell(p1.x, p1.y)
        goal_col, goal_row = grid.to_cell(p2.x, p2.y)

        path = _astar(grid, start_col, start_row, goal_col, goal_row)

        if path is None and _tht_refs_in_net:
            # Retry: shrink clearance zones around sibling THT pads to
            # create routing channels BETWEEN pads (not through them).
            # We unmark the full pad+clearance area then re-mark just
            # the pad copper, leaving only the clearance ring free.
            for ref in _tht_refs_in_net:
                fp = fp_by_ref[ref]
                for pad in fp.pads:
                    px, py = _pad_abs_pos(fp, pad)
                    # Unmark pad + clearance
                    _unmark_pad_area(
                        grid, px, py,
                        pad.size_x / 2, pad.size_y / 2, pad_cl,
                    )
                    # Re-mark just the pad copper (no clearance)
                    _mark_pad_area(grid, px, py, pad.size_x / 2, pad.size_y / 2, 0.0)
            _tht_refs_in_net.clear()  # only retry once
            path = _astar(grid, start_col, start_row, goal_col, goal_row)

        if path is None:
            # Restore all pad markings that may have been cleared
            _restore_pad_marks(grid, footprints, net_clearances)
            return RouteResult(
                net_number=request.net_number,
                net_name=request.net_name,
                tracks=tuple(all_tracks),
                vias=(),
                routed=False,
                reason=f"No path found for net {request.net_name}",
            )

        # Convert path to Track segments
        for j in range(len(path) - 1):
            x1, y1 = grid.to_mm(path[j][0], path[j][1])
            x2, y2 = grid.to_mm(path[j + 1][0], path[j + 1][1])
            all_tracks.append(
                Track(
                    start=Point(x1, y1),
                    end=Point(x2, y2),
                    width=request.width_mm,
                    layer=request.layer,
                    net_number=request.net_number,
                    uuid="",
                )
            )

        # Mark path cells with clearance (half track width + netclass clearance)
        for cell_col, cell_row in path:
            for dc in range(-excl_cells, excl_cells + 1):
                for dr in range(-excl_cells, excl_cells + 1):
                    grid.mark(cell_col + dc, cell_row + dr)

        # Re-unmark same-net pads so subsequent MST connections can still
        # reach unrouted target pads.  IC pads are NOT re-unmarked — the
        # initial unmark provides access, and marked path cells naturally
        # constrain later connections, preventing over-long wandering routes.
        for pi in pad_infos:
            _unmark_pad_area(grid, pi.x, pi.y, pi.half_w, pi.half_h, pad_cl)

        routed_set.add(best_to)
        unrouted.discard(best_to)

    # Restore all pad markings that may have been cleared during unmark
    _restore_pad_marks(grid, footprints, net_clearances)

    return RouteResult(
        net_number=request.net_number,
        net_name=request.net_name,
        tracks=tuple(all_tracks),
        vias=(),
        routed=True,
    )


def route_all_nets(
    netlist: Netlist,
    footprints: list[Footprint],
    board_width_mm: float,
    board_height_mm: float,
    grid_step_mm: float = 0.25,
    net_widths: dict[str, float] | None = None,
    net_clearances: dict[str, float] | None = None,
    keepouts: tuple[Keepout, ...] = (),
) -> tuple[RouteResult, ...]:
    """Route all nets in the netlist using a shared occupancy grid.

    A single grid is created and shared across all nets so that each
    net's routed tracks become obstacles for subsequent nets, preventing
    shorts.  Nets are sorted by pad count ascending (simpler nets first)
    to improve overall routability.

    Nets with fewer than two pads are skipped.  Trace widths are looked up
    from *net_widths* when provided; otherwise Power/GND nets receive a
    wider trace (0.5 mm) and all other nets use 0.25 mm.

    Args:
        netlist: The board netlist.
        footprints: All placed footprints.
        board_width_mm: Board width in mm.
        board_height_mm: Board height in mm.
        grid_step_mm: Grid resolution in mm.
        net_widths: Optional mapping from net name to trace width in mm,
            typically from :func:`~kicad_pipeline.pcb.netclasses.net_width_map`.
        net_clearances: Optional mapping from net name to clearance in mm.
        keepouts: Keepout zones to avoid during routing.

    Returns:
        Tuple of RouteResult, one per routed net entry.
    """
    # Filter to routable nets: skip GND (handled by copper pour) and single-pad nets
    routable = [
        e for e in netlist.entries
        if len(e.pad_refs) >= 2 and e.net.name != "GND"
    ]

    # Sort by estimated route length: shortest first.  This minimises
    # grid congestion — short, local nets consume few cells and leave
    # space for longer nets.  Power nets route last.
    fp_by_ref: dict[str, Footprint] = {fp.ref: fp for fp in footprints}

    def _estimated_length(entry: NetlistEntry) -> float:
        positions: list[tuple[float, float]] = []
        for ref, pad_num in entry.pad_refs:
            fp = fp_by_ref.get(ref)
            if fp is None:
                continue
            for pad in fp.pads:
                if pad.number == pad_num:
                    positions.append(_pad_abs_pos(fp, pad))
                    break
        if len(positions) < 2:
            return 0.0
        max_d = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
                max_d = max(max_d, d)
        return max_d

    def _sort_key(entry: NetlistEntry) -> tuple[int, float]:
        name = entry.net.name.upper()
        is_power = name.startswith("+") or "VDD" in name or "VCC" in name or "VBUS" in name
        tier = 2 if is_power else 0
        return (tier, _estimated_length(entry))

    routable.sort(key=_sort_key)

    # Create a shared grid and prepare it with pads, edge margins, keepouts
    grid = _Grid.create(board_width_mm, board_height_mm, grid_step_mm)
    _prepare_grid(
        grid, list(footprints), keepouts=keepouts,
        net_clearances=net_clearances,
    )

    results: list[RouteResult] = []

    def _route_entry(entry: NetlistEntry) -> RouteResult:
        net_name = entry.net.name
        if net_widths is not None:
            width = net_widths.get(net_name, 0.25)
        else:
            width = 0.5 if "GND" in net_name or "PWR" in net_name else 0.25

        clearance = 0.2  # default netclass clearance
        if net_clearances is not None:
            clearance = net_clearances.get(net_name, 0.2)

        request = RouteRequest(
            net_number=entry.net.number,
            net_name=net_name,
            pad_refs=entry.pad_refs,
            layer="F.Cu",
            width_mm=width,
            clearance_mm=clearance,
        )
        return route_net(
            request, footprints, board_width_mm, board_height_mm,
            grid_step_mm, grid=grid, net_clearances=net_clearances,
        )

    # First pass: route all nets
    failed_entries: list[NetlistEntry] = []
    for entry in routable:
        result = _route_entry(entry)
        if result.routed:
            results.append(result)
        else:
            failed_entries.append(entry)

    # Retry failed nets: earlier routes may have left enough space
    for entry in failed_entries:
        result = _route_entry(entry)
        results.append(result)

    return tuple(results)


def collect_tracks(
    results: tuple[RouteResult, ...],
    *,
    routed_only: bool = True,
    filter_dangling: bool = True,
) -> tuple[Track, ...]:
    """Flatten all Track objects from all RouteResults into a single tuple.

    When *routed_only* is ``True`` (default), tracks from partially-routed
    nets are excluded.  Partial routes create tracks that don't complete
    connections, causing both ``unconnected`` and ``clearance``/``shorting``
    DRC violations — removing them reduces overall violation count.

    When *filter_dangling* is ``True`` (default), single-segment tracks whose
    endpoints don't connect to any other track in the same net are removed.

    Args:
        results: Routing results to collect tracks from.
        routed_only: Only include tracks from fully-routed nets (default True).
        filter_dangling: Remove orphan single-segment stubs (default True).

    Returns:
        Combined tuple of all tracks.
    """
    tracks: list[Track] = []
    for r in results:
        if routed_only and not r.routed:
            continue
        tracks.extend(r.tracks)

    if not filter_dangling or len(tracks) < 2:
        return tuple(tracks)

    # Group tracks by net, then find dangling endpoints
    by_net: dict[int, list[Track]] = {}
    for t in tracks:
        by_net.setdefault(t.net_number, []).append(t)

    keep: list[Track] = []
    for net_tracks in by_net.values():
        if len(net_tracks) <= 1:
            # A single-segment net is OK (direct pad-to-pad)
            keep.extend(net_tracks)
            continue

        # Build endpoint connectivity: count how many tracks touch each point
        eps: dict[tuple[float, float], int] = {}
        for t in net_tracks:
            sk = (round(t.start.x, 4), round(t.start.y, 4))
            ek = (round(t.end.x, 4), round(t.end.y, 4))
            eps[sk] = eps.get(sk, 0) + 1
            eps[ek] = eps.get(ek, 0) + 1

        for t in net_tracks:
            sk = (round(t.start.x, 4), round(t.start.y, 4))
            ek = (round(t.end.x, 4), round(t.end.y, 4))
            # A stub has both endpoints only appearing once (no connections)
            if eps.get(sk, 0) <= 1 and eps.get(ek, 0) <= 1:
                continue  # orphan stub — skip
            keep.append(t)

    return tuple(keep)


def collect_vias(results: tuple[RouteResult, ...]) -> tuple[Via, ...]:
    """Flatten all Via objects from all RouteResults into a single tuple.

    Args:
        results: Routing results to collect vias from.

    Returns:
        Combined tuple of all vias.
    """
    vias: list[Via] = []
    for r in results:
        vias.extend(r.vias)
    return tuple(vias)
