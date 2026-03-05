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
from kicad_pipeline.models.pcb import Footprint, Point, Track, Via

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


def _prepare_grid(
    grid: _Grid,
    footprints: list[Footprint],
    keepouts: tuple[Keepout, ...] = (),
) -> None:
    """Mark pad positions, board-edge margins, and keepout zones on the grid.

    This is called once during grid creation to establish the base occupancy
    before any routing begins.

    Args:
        grid: The grid to prepare.
        footprints: All footprints on the board (pads are marked occupied).
        keepouts: Keepout zones whose areas are marked occupied.
    """
    # Mark all pad positions as occupied
    for fp in footprints:
        for pad in fp.pads:
            px = fp.position.x + pad.position.x
            py = fp.position.y + pad.position.y
            grid.mark_mm(px, py, radius_cells=1)

    # Mark board-edge margins as occupied
    margin_cells = max(1, int(JLCPCB_BOARD_EDGE_CLEARANCE_MM / grid.grid_step_mm) + 1)
    for col in range(grid.cols):
        for r in range(margin_cells):
            grid.mark(col, r)                       # top edge
            grid.mark(col, grid.rows - 1 - r)       # bottom edge
    for row in range(grid.rows):
        for c in range(margin_cells):
            grid.mark(c, row)                        # left edge
            grid.mark(grid.cols - 1 - c, row)        # right edge

    # Mark keepout zones as occupied
    for ko in keepouts:
        if not ko.polygon:
            continue
        ko_xs = [p.x for p in ko.polygon]
        ko_ys = [p.y for p in ko.polygon]
        min_col, min_row = grid.to_cell(min(ko_xs), min(ko_ys))
        max_col, max_row = grid.to_cell(max(ko_xs), max(ko_ys))
        for kc in range(min_col, max_col + 1):
            for kr in range(min_row, max_row + 1):
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


def _resolve_pad_positions(
    request: RouteRequest,
    fp_by_ref: dict[str, Footprint],
) -> list[tuple[float, float]] | str:
    """Resolve pad world positions for a route request.

    Args:
        request: The routing request.
        fp_by_ref: Lookup from ref to Footprint.

    Returns:
        List of ``(x, y)`` positions, or an error string if resolution fails.
    """
    positions: list[tuple[float, float]] = []
    for ref, pad_num in request.pad_refs:
        found_fp = fp_by_ref.get(ref)
        if found_fp is None:
            return f"Footprint '{ref}' not found"
        pad_pos: tuple[float, float] | None = None
        for pad in found_fp.pads:
            if pad.number == pad_num:
                pad_pos = (
                    found_fp.position.x + pad.position.x,
                    found_fp.position.y + pad.position.y,
                )
                break
        if pad_pos is None:
            return f"Pad '{pad_num}' not found on footprint '{ref}'"
        positions.append(pad_pos)
    return positions


def route_net(
    request: RouteRequest,
    footprints: list[Footprint],
    board_width_mm: float,
    board_height_mm: float,
    grid_step_mm: float = 0.5,
    grid: _Grid | None = None,
    keepouts: tuple[Keepout, ...] = (),
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

    # Resolve pad world positions
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
    positions = resolved

    if len(positions) < 2:
        return RouteResult(
            net_number=request.net_number,
            net_name=request.net_name,
            tracks=(),
            vias=(),
            routed=False,
            reason="insufficient pad positions",
        )

    # Temporarily unmark this net's own pads so the router can reach them
    for px, py in positions:
        grid.unmark_mm(px, py, radius_cells=1)

    all_tracks: list[Track] = []

    # MST-style routing: always connect closest unrouted pad to routed set
    routed_set: set[int] = {0}  # index into positions
    unrouted: set[int] = set(range(1, len(positions)))

    while unrouted:
        # Find the closest (routed, unrouted) pair by Manhattan distance
        best_from = 0
        best_to = next(iter(unrouted))
        best_dist = float("inf")
        for ri in routed_set:
            for ui in unrouted:
                d = (abs(positions[ri][0] - positions[ui][0])
                     + abs(positions[ri][1] - positions[ui][1]))
                if d < best_dist:
                    best_dist = d
                    best_from = ri
                    best_to = ui

        p1 = positions[best_from]
        p2 = positions[best_to]

        start_col, start_row = grid.to_cell(p1[0], p1[1])
        goal_col, goal_row = grid.to_cell(p2[0], p2[1])

        path = _astar(grid, start_col, start_row, goal_col, goal_row)

        if path is None:
            # Re-mark pads before returning failure
            for px, py in positions:
                grid.mark_mm(px, py, radius_cells=1)
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

        # Mark path cells with clearance so subsequent nets don't short
        clearance_cells = max(1, round(0.2 / grid.grid_step_mm))
        for cell_col, cell_row in path:
            for dc in range(-clearance_cells, clearance_cells + 1):
                for dr in range(-clearance_cells, clearance_cells + 1):
                    grid.mark(cell_col + dc, cell_row + dr)

        routed_set.add(best_to)
        unrouted.discard(best_to)

    # Re-mark pad cells for subsequent nets
    for px, py in positions:
        grid.mark_mm(px, py, radius_cells=1)

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
        keepouts: Keepout zones to avoid during routing.

    Returns:
        Tuple of RouteResult, one per routed net entry.
    """
    # Filter to routable nets: skip GND (handled by copper pour) and single-pad nets
    routable = [
        e for e in netlist.entries
        if len(e.pad_refs) >= 2 and e.net.name != "GND"
    ]

    # Route short signal nets first, power nets last
    def _sort_key(entry: NetlistEntry) -> tuple[bool, int]:
        name = entry.net.name.upper()
        is_power = name.startswith("+") or "VDD" in name or "VCC" in name or "VBUS" in name
        return (is_power, len(entry.pad_refs))

    routable.sort(key=_sort_key)

    # Create a shared grid and prepare it with pads, edge margins, keepouts
    grid = _Grid.create(board_width_mm, board_height_mm, grid_step_mm)
    _prepare_grid(grid, list(footprints), keepouts=keepouts)

    results: list[RouteResult] = []

    for entry in routable:
        net_name = entry.net.name
        if net_widths is not None:
            width = net_widths.get(net_name, 0.25)
        else:
            width = 0.5 if "GND" in net_name or "PWR" in net_name else 0.25

        request = RouteRequest(
            net_number=entry.net.number,
            net_name=net_name,
            pad_refs=entry.pad_refs,
            layer="F.Cu",
            width_mm=width,
        )
        result = route_net(
            request, footprints, board_width_mm, board_height_mm,
            grid_step_mm, grid=grid,
        )
        results.append(result)

    return tuple(results)


def collect_tracks(results: tuple[RouteResult, ...]) -> tuple[Track, ...]:
    """Flatten all Track objects from all RouteResults into a single tuple.

    Args:
        results: Routing results to collect tracks from.

    Returns:
        Combined tuple of all tracks.
    """
    tracks: list[Track] = []
    for r in results:
        tracks.extend(r.tracks)
    return tuple(tracks)


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
