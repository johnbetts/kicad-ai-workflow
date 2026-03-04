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

from kicad_pipeline.models.pcb import Footprint, Point, Track, Via

if TYPE_CHECKING:
    from kicad_pipeline.pcb.netlist import Netlist


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
    - it equals the goal (goal_col, goal_row), OR
    - it is within Manhattan distance 1 of start (exit start pad zone), OR
    - it is within Manhattan distance 1 of goal (enter goal pad zone).

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
            near_start = (abs(nc - start_col) + abs(nr - start_row)) <= 1
            near_goal = (abs(nc - goal_col) + abs(nr - goal_row)) <= 1
            if not (grid.is_free(nc, nr) or is_start_or_goal or near_start or near_goal):
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


def route_net(
    request: RouteRequest,
    footprints: list[Footprint],
    board_width_mm: float,
    board_height_mm: float,
    grid_step_mm: float = 0.5,
) -> RouteResult:
    """Route a single net using A* on a 2-D occupancy grid.

    Args:
        request: The routing request describing the net and its pads.
        footprints: All footprints on the board (used to build occupancy).
        board_width_mm: Board width in mm (defines grid extent).
        board_height_mm: Board height in mm (defines grid extent).
        grid_step_mm: Grid resolution in mm.

    Returns:
        A RouteResult with all generated tracks (or failure info).
    """
    grid = _Grid.create(board_width_mm, board_height_mm, grid_step_mm)

    # Mark all pad positions as occupied
    for fp in footprints:
        for pad in fp.pads:
            px = fp.position.x + pad.position.x
            py = fp.position.y + pad.position.y
            grid.mark_mm(px, py, radius_cells=1)

    # Build a lookup: ref -> Footprint
    fp_by_ref: dict[str, Footprint] = {fp.ref: fp for fp in footprints}

    # Resolve pad world positions
    positions: list[tuple[float, float]] = []
    for ref, pad_num in request.pad_refs:
        found_fp: Footprint | None = fp_by_ref.get(ref)
        if found_fp is None:
            return RouteResult(
                net_number=request.net_number,
                net_name=request.net_name,
                tracks=(),
                vias=(),
                routed=False,
                reason=f"Footprint '{ref}' not found",
            )
        pad_pos: tuple[float, float] | None = None
        for pad in found_fp.pads:
            if pad.number == pad_num:
                pad_pos = (
                    found_fp.position.x + pad.position.x,
                    found_fp.position.y + pad.position.y,
                )
                break
        if pad_pos is None:
            return RouteResult(
                net_number=request.net_number,
                net_name=request.net_name,
                tracks=(),
                vias=(),
                routed=False,
                reason=f"Pad '{pad_num}' not found on footprint '{ref}'",
            )
        positions.append(pad_pos)

    if len(positions) < 2:
        return RouteResult(
            net_number=request.net_number,
            net_name=request.net_name,
            tracks=(),
            vias=(),
            routed=False,
            reason="insufficient pad positions",
        )

    all_tracks: list[Track] = []

    # Clearance radius used when marking pad cells
    mark_radius: int = 1

    # Connect pads sequentially: positions[0]->positions[1]->positions[2]->...
    for i in range(len(positions) - 1):
        p1 = positions[i]
        p2 = positions[i + 1]

        start_col, start_row = grid.to_cell(p1[0], p1[1])
        goal_col, goal_row = grid.to_cell(p2[0], p2[1])

        # Pre-check: if pad clearance zones overlap, the pads are too close
        # together to route between -- there is no free corridor available.
        col_gap = abs(start_col - goal_col) - 2 * mark_radius
        row_gap = abs(start_row - goal_row) - 2 * mark_radius
        # Zones overlap in column if col_gap < 0 AND in row if row_gap < 0.
        # For a rectangular clearance block, the zones touch/overlap when
        # BOTH dimensions are non-positive (i.e., no free column AND row gap).
        if col_gap <= 0 and row_gap <= 0:
            return RouteResult(
                net_number=request.net_number,
                net_name=request.net_name,
                tracks=tuple(all_tracks),
                vias=(),
                routed=False,
                reason=f"No path found for net {request.net_name}",
            )

        path = _astar(grid, start_col, start_row, goal_col, goal_row)

        if path is None:
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

        # Mark path cells occupied so subsequent nets route around them
        for cell_col, cell_row in path:
            grid.mark(cell_col, cell_row)

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
    grid_step_mm: float = 0.5,
    net_widths: dict[str, float] | None = None,
) -> tuple[RouteResult, ...]:
    """Route all nets in the netlist.

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

    Returns:
        Tuple of RouteResult, one per routed net entry.
    """
    results: list[RouteResult] = []

    for entry in netlist.entries:
        if len(entry.pad_refs) < 2:
            continue

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
        result = route_net(request, footprints, board_width_mm, board_height_mm, grid_step_mm)
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
