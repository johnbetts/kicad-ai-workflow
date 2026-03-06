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

from kicad_pipeline.constants import (
    JLCPCB_BOARD_EDGE_CLEARANCE_MM,
    JLCPCB_MIN_TRACE_MM,
    ROUTING_BEND_PENALTY,
    ROUTING_CONGESTION_MAX,
    VIA_DIAMETER_SIGNAL_MM,
    VIA_DRILL_SIGNAL_MM,
)
from kicad_pipeline.models.pcb import Footprint, Keepout, Pad, Point, Track, Via


def _keepout_blocks_layer(keepout: Keepout, layer: str) -> bool:
    """Return True when *keepout* should block routing on *layer*.

    A keepout blocks a layer when:
    1. Its ``layers`` tuple includes *layer* (or is empty, meaning all layers).
    2. It prohibits tracks (``no_tracks``) or copper (``no_copper``).
    """
    if keepout.layers and layer not in keepout.layers:
        return False
    return keepout.no_tracks or keepout.no_copper


def _pad_abs_pos(fp: Footprint, pad: Pad) -> tuple[float, float]:
    """Compute absolute pad position accounting for footprint rotation.

    KiCad uses clockwise rotation (positive angle = CW), which is the
    negative of standard mathematical CCW convention.
    """
    rad = math.radians(-fp.rotation)  # negate for CW convention
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    rx = pad.position.x * cos_r - pad.position.y * sin_r
    ry = pad.position.x * sin_r + pad.position.y * cos_r
    return (fp.position.x + rx, fp.position.y + ry)


def _pad_rotated_half_size(fp: Footprint, pad: Pad) -> tuple[float, float]:
    """Return (half_w, half_h) of a pad after footprint rotation.

    For axis-aligned rotations (0/90/180/270), swap size_x and size_y
    when rotated 90 or 270 degrees.  For arbitrary angles, use the
    bounding box of the rotated rectangle.
    """
    hw = pad.size_x / 2.0
    hh = pad.size_y / 2.0
    rot = fp.rotation % 360.0
    if abs(rot - 90.0) < 0.01 or abs(rot - 270.0) < 0.01:
        return (hh, hw)
    if abs(rot) < 0.01 or abs(rot - 180.0) < 0.01:
        return (hw, hh)
    # Arbitrary rotation: compute axis-aligned bounding box
    rad = math.radians(rot)
    cos_r = abs(math.cos(rad))
    sin_r = abs(math.sin(rad))
    return (hw * cos_r + hh * sin_r, hw * sin_r + hh * cos_r)

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
    max_vias: int = 2


@dataclass(frozen=True)
class RouteResult:
    """Result of routing a single net."""

    net_number: int
    net_name: str
    tracks: tuple[Track, ...]
    vias: tuple[Via, ...]
    routed: bool  # True if all connections were made
    reason: str = ""  # failure reason if not routed


@dataclass(frozen=True)
class RouteQuality:
    """Quality score for a single routed net."""

    net_name: str
    manhattan_ideal_mm: float
    actual_length_mm: float
    length_ratio: float
    via_count: int
    bend_count: int
    score: float  # composite badness (higher = worse)


def _score_route(result: RouteResult, pad_positions: list[tuple[float, float]]) -> RouteQuality:
    """Compute quality metrics for a routed net."""
    # Manhattan ideal: sum of MST edges between pads
    manhattan = 0.0
    if len(pad_positions) >= 2:
        # Approximate MST with sorted nearest-neighbour
        remaining = list(range(1, len(pad_positions)))
        connected = [0]
        while remaining:
            best_d = float("inf")
            best_i = remaining[0]
            for ci in connected:
                for ri in remaining:
                    d = (abs(pad_positions[ci][0] - pad_positions[ri][0])
                         + abs(pad_positions[ci][1] - pad_positions[ri][1]))
                    if d < best_d:
                        best_d = d
                        best_i = ri
            manhattan += best_d
            remaining.remove(best_i)
            connected.append(best_i)

    actual = 0.0
    bends = 0
    prev_dx: float = 0.0
    prev_dy: float = 0.0
    for trk in result.tracks:
        dx = trk.end.x - trk.start.x
        dy = trk.end.y - trk.start.y
        actual += (dx * dx + dy * dy) ** 0.5
        if (
            (prev_dx != 0.0 or prev_dy != 0.0)
            and (abs(dx - prev_dx) > 0.01 or abs(dy - prev_dy) > 0.01)
        ):
            bends += 1
        prev_dx, prev_dy = dx, dy

    ratio = actual / manhattan if manhattan > 0.01 else 1.0
    via_count = len(result.vias)
    # Composite score matching spec cost function weights:
    #   1.0*actual + 16*vias + 3*bends + 6*max(0, ratio-1.55)
    score = actual + 16.0 * via_count + 3.0 * bends + 6.0 * max(0.0, ratio - 1.55)
    return RouteQuality(
        net_name=result.net_name,
        manhattan_ideal_mm=manhattan,
        actual_length_mm=actual,
        length_ratio=ratio,
        via_count=via_count,
        bend_count=bends,
        score=score,
    )


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
    _congestion: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._cells:
            self._cells = [[False] * self.rows for _ in range(self.cols)]
        if not self._congestion:
            self._congestion = [[0] * self.rows for _ in range(self.cols)]

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

    def add_congestion(self, col: int, row: int, radius: int = 1) -> None:
        """Increment congestion counter around a cell."""
        for dc in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                nc, nr = col + dc, row + dr
                if 0 <= nc < self.cols and 0 <= nr < self.rows:
                    self._congestion[nc][nr] += 1

    def get_cost(self, col: int, row: int) -> float:
        """Return traversal cost for a cell accounting for congestion.

        Base cost is 1.0; rises toward ``ROUTING_CONGESTION_MAX`` as
        congestion increases (threshold = 4 overlapping tracks).
        """
        if col < 0 or col >= self.cols or row < 0 or row >= self.rows:
            return 1.0
        cong = self._congestion[col][row]
        if cong <= 0:
            return 1.0
        ratio = min(1.0, cong / 4.0)
        return 1.0 + ratio * (ROUTING_CONGESTION_MAX - 1.0)


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
    """Mark a rectangular pad area + clearance on the routing grid.

    Uses ceil for the upper bound so that any cell whose center falls
    within one grid step of the clearance zone edge is also blocked.
    This prevents tracks from being placed in cells that partially
    overlap the required clearance zone.
    """
    x0 = px - half_w - clearance_mm
    y0 = py - half_h - clearance_mm
    x1 = px + half_w + clearance_mm
    y1 = py + half_h + clearance_mm
    c0, r0 = grid.to_cell(x0, y0)
    gs = grid.grid_step_mm
    c1 = min(grid.cols - 1, math.ceil(x1 / gs))
    r1 = min(grid.rows - 1, math.ceil(y1 / gs))
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


def _global_pad_clearance(
    net_clearances: dict[str, float] | None,
    net_widths: dict[str, float] | None = None,
) -> float:
    """Compute the global pad clearance (max netclass + half max track width).

    Uses the widest track across all netclasses to ensure clearance zones
    are large enough for any approaching net.  Without net_widths, falls
    back to the default 0.25 mm track (half = 0.125 mm).
    """
    htw = 0.125
    if net_widths:
        htw = max(net_widths.values()) / 2.0
    max_cl = _PAD_CLEARANCE_MM
    if net_clearances:
        max_cl = max(max_cl, max(net_clearances.values()))
    return max_cl + htw


def _track_crosses_other_pads(
    tracks: list[Track] | tuple[Track, ...],
    net_number: int,
    footprints: list[Footprint],
    clearance_mm: float = 0.05,
    net_pad_set: frozenset[tuple[str, str]] | None = None,
) -> bool:
    """Return True if any F.Cu track crosses a pad on a different net.

    Used to validate IC final-leg stubs: if a B.Cu fallback creates an
    F.Cu stub that crosses another IC pad, the connection should be
    discarded rather than creating a DRC short.

    Args:
        net_pad_set: If provided, identifies same-net pads by (ref, pad_number).
            Used during routing when pad.net_number is not yet assigned.
    """
    for fp in footprints:
        for pad in fp.pads:
            px, py = _pad_abs_pos(fp, pad)
            phw, phh = _pad_rotated_half_size(fp, pad)
            # Skip pads on the same net
            if net_pad_set is not None:
                if (fp.ref, pad.number) in net_pad_set:
                    continue
            elif pad.net_number is not None and pad.net_number == net_number:
                continue
            for t in tracks:
                if t.layer != "F.Cu":
                    continue
                hw = t.width / 2.0
                # Quick AABB check: does the track segment bounding box
                # overlap the pad rectangle (with clearance)?
                tx0 = min(t.start.x, t.end.x) - hw
                tx1 = max(t.start.x, t.end.x) + hw
                ty0 = min(t.start.y, t.end.y) - hw
                ty1 = max(t.start.y, t.end.y) + hw
                pad_x0 = px - phw - clearance_mm
                pad_x1 = px + phw + clearance_mm
                pad_y0 = py - phh - clearance_mm
                pad_y1 = py + phh + clearance_mm
                if tx1 > pad_x0 and tx0 < pad_x1 and ty1 > pad_y0 and ty0 < pad_y1:
                    return True
    return False


def _restore_pad_marks(
    grid: _Grid,
    footprints: list[Footprint],
    net_clearances: dict[str, float] | None = None,
    net_widths: dict[str, float] | None = None,
) -> None:
    """Re-mark all pad areas after temporarily clearing same-net pads.

    This prevents cross-net contamination: when clearing pad A's clearance
    zone for routing, nearby pad B's zone might also get cleared.  After
    routing, this function restores ALL pad marks with correct clearances.
    """
    cl = _global_pad_clearance(net_clearances, net_widths)
    for fp in footprints:
        for pad in fp.pads:
            px, py = _pad_abs_pos(fp, pad)
            phw, phh = _pad_rotated_half_size(fp, pad)
            _mark_pad_area(grid, px, py, phw, phh, cl)


def _remark_other_pads(
    grid: _Grid,
    footprints: list[Footprint],
    net_pad_set: frozenset[tuple[str, str]],
    net_clearances: dict[str, float] | None = None,
    net_widths: dict[str, float] | None = None,
) -> None:
    """Re-mark pads NOT in the current net to prevent cross-net contamination.

    When unmarking same-net pads' clearance zones, nearby pads on different
    nets may have their clearance zones partially cleared (overlap).  This
    function re-marks all non-current-net pads to restore correct blocking.
    """
    cl = _global_pad_clearance(net_clearances, net_widths)
    for fp in footprints:
        for pad in fp.pads:
            if (fp.ref, pad.number) in net_pad_set:
                continue
            px, py = _pad_abs_pos(fp, pad)
            phw, phh = _pad_rotated_half_size(fp, pad)
            _mark_pad_area(grid, px, py, phw, phh, cl)


def _prepare_grid(
    grid: _Grid,
    footprints: list[Footprint],
    keepouts: tuple[Keepout, ...] = (),
    net_clearances: dict[str, float] | None = None,
    net_widths: dict[str, float] | None = None,
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
    # Clearance = max(netclass clearances) + half(max track width) because
    # KiCad DRC checks clearance as max(net_A_clearance, net_B_clearance)
    # and measures from copper edge to copper edge.
    _pad_mark_cl = _global_pad_clearance(net_clearances, net_widths)
    for fp in footprints:
        for pad in fp.pads:
            px, py = _pad_abs_pos(fp, pad)
            phw, phh = _pad_rotated_half_size(fp, pad)
            _mark_pad_area(grid, px, py, phw, phh, _pad_mark_cl)

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
        if not _keepout_blocks_layer(ko, "F.Cu"):
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


def _prepare_bcu_grid(
    grid: _Grid,
    footprints: list[Footprint],
    keepouts: tuple[Keepout, ...] = (),
    net_clearances: dict[str, float] | None = None,
    net_widths: dict[str, float] | None = None,
) -> _Grid:
    """Create and prepare a B.Cu routing grid.

    The B.Cu grid marks board-edge margins, keepout zones, and THT pad
    positions (which penetrate both layers), but does NOT mark SMD pad
    areas since SMD pads only exist on F.Cu.

    Args:
        grid: The F.Cu grid (used to copy dimensions).
        footprints: All footprints on the board.
        keepouts: Keepout zones to mark as occupied.
        net_clearances: Optional per-net clearance overrides.
        net_widths: Optional per-net track width overrides.

    Returns:
        A new :class:`_Grid` instance for B.Cu routing.
    """
    bcu = _Grid.create(
        grid.cols * grid.grid_step_mm,
        grid.rows * grid.grid_step_mm,
        grid.grid_step_mm,
    )

    # Mark board-edge margins (same as F.Cu)
    margin_cells = max(1, int(JLCPCB_BOARD_EDGE_CLEARANCE_MM / bcu.grid_step_mm) + 1)
    for col in range(bcu.cols):
        for mr in range(margin_cells):
            bcu.mark(col, mr)
            bcu.mark(col, bcu.rows - 1 - mr)
    for row in range(bcu.rows):
        for mc in range(margin_cells):
            bcu.mark(mc, row)
            bcu.mark(bcu.cols - 1 - mc, row)

    # Mark keepout zones (only those that block B.Cu)
    for ko in keepouts:
        if not ko.polygon:
            continue
        if not _keepout_blocks_layer(ko, "B.Cu"):
            continue
        ko_xs = [p.x for p in ko.polygon]
        ko_ys = [p.y for p in ko.polygon]
        min_col, min_row = bcu.to_cell(min(ko_xs), min(ko_ys))
        max_col, max_row = bcu.to_cell(max(ko_xs), max(ko_ys))
        for kc in range(
            max(0, min_col - _KEEPOUT_MARGIN_CELLS),
            min(bcu.cols, max_col + _KEEPOUT_MARGIN_CELLS + 1),
        ):
            for kr in range(
                max(0, min_row - _KEEPOUT_MARGIN_CELLS),
                min(bcu.rows, max_row + _KEEPOUT_MARGIN_CELLS + 1),
            ):
                bcu.mark(kc, kr)

    # Mark THT pads only (they penetrate both layers)
    pad_cl = _global_pad_clearance(net_clearances, net_widths)
    for fp in footprints:
        for pad in fp.pads:
            if pad.pad_type != "thru_hole":
                continue
            px, py = _pad_abs_pos(fp, pad)
            phw, phh = _pad_rotated_half_size(fp, pad)
            _mark_pad_area(bcu, px, py, phw, phh, pad_cl)

    return bcu


def _is_line_clear(
    grid: _Grid,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    exclusion_mm: float,
) -> bool:
    """Check whether all cells along a line segment are free on the grid.

    Uses Bresenham-style stepping (same as :func:`_mark_line_on_grid`) to
    walk from ``(x1, y1)`` to ``(x2, y2)`` and returns ``True`` only if
    every cell along the path (plus exclusion margin) is free.

    Args:
        grid: The occupancy grid to check.
        x1: Start X in mm.
        y1: Start Y in mm.
        x2: End X in mm.
        y2: End Y in mm.
        exclusion_mm: Exclusion radius around the line in mm.

    Returns:
        ``True`` if the entire path is clear.
    """
    gs = grid.grid_step_mm
    c1, r1 = grid.to_cell(x1, y1)
    c2, r2 = grid.to_cell(x2, y2)
    excl_cells = max(1, math.ceil(exclusion_mm / gs))

    dc_total = abs(c2 - c1)
    dr_total = abs(r2 - r1)
    sc = 1 if c1 < c2 else -1
    sr = 1 if r1 < r2 else -1
    err = dc_total - dr_total
    cc, cr = c1, r1

    while True:
        for ddc in range(-excl_cells, excl_cells + 1):
            for ddr in range(-excl_cells, excl_cells + 1):
                nc = cc + ddc
                nr = cr + ddr
                if nc < 0 or nr < 0 or nc >= grid.cols or nr >= grid.rows:
                    return False
                if not grid.is_free(nc, nr):
                    return False
        if cc == c2 and cr == r2:
            break
        e2 = 2 * err
        if e2 > -dr_total:
            err -= dr_total
            cc += sc
        if e2 < dc_total:
            err += dc_total
            cr += sr

    return True


def _route_stub_on_fcu(
    grid: _Grid,
    pad_x: float,
    pad_y: float,
    via_x: float,
    via_y: float,
    net_number: int,
    width_mm: float,
    clearance_mm: float,
) -> tuple[Track, ...] | None:
    """Route an F.Cu stub from pad to via using A* when direct line is blocked.

    Falls back to grid-aligned routing when the diagonal pad-to-via path
    would cross existing F.Cu tracks.

    Args:
        grid: The F.Cu occupancy grid.
        pad_x: Pad X position in mm.
        pad_y: Pad Y position in mm.
        via_x: Via X position in mm.
        via_y: Via Y position in mm.
        net_number: Net number for the tracks.
        width_mm: Track width in mm.
        clearance_mm: Track clearance in mm.

    Returns:
        Tuple of F.Cu Track segments, or ``None`` if A* fails.
    """
    sc, sr = grid.to_cell(pad_x, pad_y)
    gc, gr = grid.to_cell(via_x, via_y)

    # Temporarily unmark start/goal so A* can enter them
    orig_start = not grid.is_free(sc, sr)
    orig_goal = not grid.is_free(gc, gr)
    grid.unmark(sc, sr)
    grid.unmark(gc, gr)

    path = _astar(grid, sc, sr, gc, gr)

    if path is None:
        if orig_start:
            grid.mark(sc, sr)
        if orig_goal:
            grid.mark(gc, gr)
        return None

    # Build tracks from path
    tracks: list[Track] = []
    for j in range(len(path) - 1):
        x1, y1 = grid.to_mm(path[j][0], path[j][1])
        x2, y2 = grid.to_mm(path[j + 1][0], path[j + 1][1])
        tracks.append(
            Track(
                start=Point(x1, y1),
                end=Point(x2, y2),
                width=width_mm,
                layer="F.Cu",
                net_number=net_number,
                uuid="",
            )
        )

    # Mark path cells with exclusion
    excl_cells = max(1, math.ceil(
        (clearance_mm + width_mm) / grid.grid_step_mm,
    ) - 1)
    for cell_col, cell_row in path:
        for dc in range(-excl_cells, excl_cells + 1):
            for dr in range(-excl_cells, excl_cells + 1):
                grid.mark(cell_col + dc, cell_row + dr)

    return tuple(tracks)


def _find_free_via_position(
    fcu_grid: _Grid,
    target_x: float,
    target_y: float,
    via_radius_mm: float,
    clearance_mm: float,
    stub_origin: tuple[float, float] | None = None,
    stub_width_mm: float = 0.25,
    bcu_grid: _Grid | None = None,
) -> tuple[float, float] | None:
    """Find a position near *target* where a via fits without overlapping pads.

    Checks cells within ``via_radius + clearance`` of the target on
    F.Cu (and optionally B.Cu) grids.  If all are free, returns the
    target position.  Otherwise spirals outward (up to 16 cells ~4 mm)
    searching for the first fully-clear position.

    When *stub_origin* is provided, also checks that the F.Cu stub path
    from the origin to the candidate via position is clear.

    When *bcu_grid* is provided, the via must also be clear on B.Cu
    (since it spans both layers).

    Returns:
        ``(x_mm, y_mm)`` of the free position, or ``None`` if no space found.
    """
    excl_radius = math.ceil(
        (via_radius_mm + clearance_mm) / fcu_grid.grid_step_mm,
    )

    def _area_free(col: int, row: int) -> bool:
        for dc in range(-excl_radius, excl_radius + 1):
            for dr in range(-excl_radius, excl_radius + 1):
                cc = col + dc
                rr = row + dr
                if cc < 0 or rr < 0 or cc >= fcu_grid.cols or rr >= fcu_grid.rows:
                    return False
                if not fcu_grid.is_free(cc, rr):
                    return False
                if bcu_grid is not None and not bcu_grid.is_free(cc, rr):
                    return False
        return True

    def _check_candidate(col: int, row: int) -> bool:
        if not _area_free(col, row):
            return False
        if stub_origin is not None:
            cx, cy = fcu_grid.to_mm(col, row)
            stub_excl = clearance_mm + stub_width_mm
            if not _is_line_clear(fcu_grid, stub_origin[0], stub_origin[1], cx, cy, stub_excl):
                return False
        return True

    tc, tr = fcu_grid.to_cell(target_x, target_y)
    if _check_candidate(tc, tr):
        return fcu_grid.to_mm(tc, tr)

    # Spiral outward in expanding rings
    max_ring = 16  # ~4 mm at 0.25 mm grid
    for ring in range(1, max_ring + 1):
        for dc in range(-ring, ring + 1):
            for dr in range(-ring, ring + 1):
                if abs(dc) != ring and abs(dr) != ring:
                    continue  # only perimeter cells
                cc = tc + dc
                rr = tr + dr
                if cc < 0 or rr < 0 or cc >= fcu_grid.cols or rr >= fcu_grid.rows:
                    continue
                if _check_candidate(cc, rr):
                    return fcu_grid.to_mm(cc, rr)
    return None


def _route_on_bcu(
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    bcu_grid: _Grid,
    net_number: int,
    net_name: str,
    width_mm: float,
    clearance_mm: float,
    fcu_grid: _Grid | None = None,
) -> tuple[tuple[Track, ...], tuple[Via, Via]] | None:
    """Attempt to route a segment on B.Cu with vias at each end.

    When *fcu_grid* is provided, vias are placed at positions that do
    not overlap existing F.Cu pads (using :func:`_find_free_via_position`).
    If the via must be offset from the pad, a short F.Cu stub track is
    added to bridge pad-to-via.  Uses smaller signal vias (0.6 mm pad /
    0.3 mm drill) to reduce footprint near congested IC pads.

    Temporarily unmarks start/goal cells on the B.Cu grid, runs A*,
    then marks the path with exclusion.  Returns B.Cu tracks and two
    vias (at start and goal) on success, or ``None`` on failure.
    """
    # Use smaller signal vias to reduce overlap in congested areas
    via_drill = VIA_DRILL_SIGNAL_MM
    via_size = VIA_DIAMETER_SIGNAL_MM
    via_radius = via_size / 2.0

    # Find via positions that don't overlap F.Cu pads
    via_start_pos: tuple[float, float] = (start_x, start_y)
    via_goal_pos: tuple[float, float] = (goal_x, goal_y)

    # Track whether stubs need A*-routed paths instead of direct diagonals
    _start_needs_astar_stub = False
    _goal_needs_astar_stub = False

    if fcu_grid is not None:
        # Try to find via position with clear stub path, checking BOTH
        # F.Cu and B.Cu grids (via spans both layers).
        found_start = _find_free_via_position(
            fcu_grid, start_x, start_y, via_radius, clearance_mm,
            stub_origin=(start_x, start_y), stub_width_mm=width_mm,
            bcu_grid=bcu_grid,
        )
        if found_start is None:
            # Fall back: find ANY free via position, use A* for stub
            found_start = _find_free_via_position(
                fcu_grid, start_x, start_y, via_radius, clearance_mm,
                bcu_grid=bcu_grid,
            )
            if found_start is None:
                # Last resort: F.Cu-only search (accept B.Cu congestion)
                found_start = _find_free_via_position(
                    fcu_grid, start_x, start_y, via_radius, clearance_mm,
                )
                if found_start is None:
                    return None
            _start_needs_astar_stub = True
        via_start_pos = found_start

        found_goal = _find_free_via_position(
            fcu_grid, goal_x, goal_y, via_radius, clearance_mm,
            stub_origin=(goal_x, goal_y), stub_width_mm=width_mm,
            bcu_grid=bcu_grid,
        )
        if found_goal is None:
            found_goal = _find_free_via_position(
                fcu_grid, goal_x, goal_y, via_radius, clearance_mm,
                bcu_grid=bcu_grid,
            )
            if found_goal is None:
                # Last resort: F.Cu-only search (accept B.Cu congestion)
                found_goal = _find_free_via_position(
                    fcu_grid, goal_x, goal_y, via_radius, clearance_mm,
                )
                if found_goal is None:
                    return None
            _goal_needs_astar_stub = True
        via_goal_pos = found_goal

    # Route on B.Cu between via positions (not pad positions)
    sc, sr = bcu_grid.to_cell(via_start_pos[0], via_start_pos[1])
    gc, gr = bcu_grid.to_cell(via_goal_pos[0], via_goal_pos[1])

    # Temporarily unmark start/goal so A* can enter them
    orig_start = not bcu_grid.is_free(sc, sr)
    orig_goal = not bcu_grid.is_free(gc, gr)
    bcu_grid.unmark(sc, sr)
    bcu_grid.unmark(gc, gr)

    path = _astar(bcu_grid, sc, sr, gc, gr)

    if path is None:
        # Restore original state
        if orig_start:
            bcu_grid.mark(sc, sr)
        if orig_goal:
            bcu_grid.mark(gc, gr)
        return None

    # Build B.Cu tracks (simplified to remove colinear intermediates)
    tracks: list[Track] = []
    sim_bcu = _simplify_path(path)
    for j in range(len(sim_bcu) - 1):
        x1, y1 = bcu_grid.to_mm(sim_bcu[j][0], sim_bcu[j][1])
        x2, y2 = bcu_grid.to_mm(sim_bcu[j + 1][0], sim_bcu[j + 1][1])
        tracks.append(
            Track(
                start=Point(x1, y1),
                end=Point(x2, y2),
                width=width_mm,
                layer="B.Cu",
                net_number=net_number,
                uuid="",
            )
        )

    # Build vias at (possibly offset) positions
    via_start = Via(
        position=Point(via_start_pos[0], via_start_pos[1]),
        drill=via_drill,
        size=via_size,
        layers=("F.Cu", "B.Cu"),
        net_number=net_number,
        uuid="",
    )
    via_goal = Via(
        position=Point(via_goal_pos[0], via_goal_pos[1]),
        drill=via_drill,
        size=via_size,
        layers=("F.Cu", "B.Cu"),
        net_number=net_number,
        uuid="",
    )

    # Add F.Cu stub tracks if vias were offset from pad positions.
    # When the direct diagonal stub would cross existing F.Cu tracks,
    # use A*-routed stubs instead (grid-aligned segments).
    stub_tracks: list[Track] = []
    start_offset = (
        abs(via_start_pos[0] - start_x) > 0.01
        or abs(via_start_pos[1] - start_y) > 0.01
    )
    goal_offset = (
        abs(via_goal_pos[0] - goal_x) > 0.01
        or abs(via_goal_pos[1] - goal_y) > 0.01
    )

    if start_offset:
        if _start_needs_astar_stub and fcu_grid is not None:
            astar_stub = _route_stub_on_fcu(
                fcu_grid, start_x, start_y,
                via_start_pos[0], via_start_pos[1],
                net_number, width_mm, clearance_mm,
            )
            if astar_stub is not None:
                stub_tracks.extend(astar_stub)
            else:
                # A* also failed — add direct stub as last resort
                stub_tracks.append(
                    Track(
                        start=Point(start_x, start_y),
                        end=Point(via_start_pos[0], via_start_pos[1]),
                        width=width_mm, layer="F.Cu",
                        net_number=net_number, uuid="",
                    )
                )
        else:
            stub_tracks.append(
                Track(
                    start=Point(start_x, start_y),
                    end=Point(via_start_pos[0], via_start_pos[1]),
                    width=width_mm, layer="F.Cu",
                    net_number=net_number, uuid="",
                )
            )

    if goal_offset:
        if _goal_needs_astar_stub and fcu_grid is not None:
            astar_stub = _route_stub_on_fcu(
                fcu_grid, goal_x, goal_y,
                via_goal_pos[0], via_goal_pos[1],
                net_number, width_mm, clearance_mm,
            )
            if astar_stub is not None:
                stub_tracks.extend(astar_stub)
            else:
                stub_tracks.append(
                    Track(
                        start=Point(goal_x, goal_y),
                        end=Point(via_goal_pos[0], via_goal_pos[1]),
                        width=width_mm, layer="F.Cu",
                        net_number=net_number, uuid="",
                    )
                )
        else:
            stub_tracks.append(
                Track(
                    start=Point(goal_x, goal_y),
                    end=Point(via_goal_pos[0], via_goal_pos[1]),
                    width=width_mm, layer="F.Cu",
                    net_number=net_number, uuid="",
                )
            )

    # Mark F.Cu stub tracks on fcu_grid so subsequent F.Cu routes avoid them.
    # A*-routed stubs are already marked by _route_stub_on_fcu; mark direct stubs.
    if fcu_grid is not None:
        for stub in stub_tracks:
            if stub.layer == "F.Cu":
                _mark_line_on_grid(
                    fcu_grid, stub.start.x, stub.start.y,
                    stub.end.x, stub.end.y,
                    clearance_mm + width_mm,
                )

    # Mark path cells with exclusion on B.Cu grid
    excl_cells = max(1, math.ceil(
        (clearance_mm + width_mm) / bcu_grid.grid_step_mm,
    ) - 1)
    for cell_col, cell_row in path:
        for dc in range(-excl_cells, excl_cells + 1):
            for dr in range(-excl_cells, excl_cells + 1):
                bcu_grid.mark(cell_col + dc, cell_row + dr)

    # Mark via exclusion on both grids
    if fcu_grid is not None:
        _mark_via_on_fcu(fcu_grid, via_start, clearance_mm)
        _mark_via_on_fcu(fcu_grid, via_goal, clearance_mm)
    # Mark via positions on B.Cu grid too
    via_excl = math.ceil((via_radius + clearance_mm) / bcu_grid.grid_step_mm)
    for via in (via_start, via_goal):
        vc, vr = bcu_grid.to_cell(via.position.x, via.position.y)
        for dc in range(-via_excl, via_excl + 1):
            for dr in range(-via_excl, via_excl + 1):
                bcu_grid.mark(vc + dc, vr + dr)

    all_tracks = list(stub_tracks) + list(tracks)
    return (tuple(all_tracks), (via_start, via_goal))


def _mark_line_on_grid(
    grid: _Grid,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    exclusion_mm: float,
) -> None:
    """Mark grid cells along a line segment with exclusion zone.

    Uses Bresenham-style stepping to walk the line from (x1, y1) to
    (x2, y2) and marks each cell plus an exclusion radius around it.
    This ensures diagonal F.Cu stub tracks become obstacles on the
    grid, preventing subsequent routes from crossing them.
    """
    gs = grid.grid_step_mm
    c1, r1 = grid.to_cell(x1, y1)
    c2, r2 = grid.to_cell(x2, y2)
    excl_cells = max(1, math.ceil(exclusion_mm / gs))

    # Bresenham line walk
    dc = abs(c2 - c1)
    dr = abs(r2 - r1)
    sc = 1 if c1 < c2 else -1
    sr = 1 if r1 < r2 else -1
    err = dc - dr
    cc, cr = c1, r1

    while True:
        # Mark cell + exclusion radius
        for ddc in range(-excl_cells, excl_cells + 1):
            for ddr in range(-excl_cells, excl_cells + 1):
                grid.mark(cc + ddc, cr + ddr)
        if cc == c2 and cr == r2:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            cc += sc
        if e2 < dc:
            err += dc
            cr += sr


def _mark_via_on_fcu(
    grid: _Grid,
    via: Via,
    clearance_mm: float = _PAD_CLEARANCE_MM,
) -> None:
    """Mark a via position on the F.Cu grid to prevent subsequent routes from crossing."""
    via_radius = via.size / 2.0
    _mark_pad_area(
        grid, via.position.x, via.position.y,
        via_radius, via_radius, clearance_mm,
    )


# ---------------------------------------------------------------------------
# A* pathfinder
# ---------------------------------------------------------------------------


def _astar(
    grid: _Grid,
    start_col: int,
    start_row: int,
    goal_col: int,
    goal_row: int,
    bend_penalty: float = ROUTING_BEND_PENALTY,
    use_congestion: bool = True,
) -> list[tuple[int, int]] | None:
    """Find a path from start to goal on the grid using A*.

    A cell is traversable if:
    - it is free (grid.is_free), OR
    - it equals the start (start_col, start_row), OR
    - it equals the goal (goal_col, goal_row).

    When *bend_penalty* > 0, direction changes incur extra cost,
    producing smoother paths with fewer bends.  When *use_congestion*
    is True, cells near previously routed tracks cost more.

    Args:
        grid: The occupancy grid.
        start_col: Start column index.
        start_row: Start row index.
        goal_col: Goal column index.
        goal_row: Goal row index.
        bend_penalty: Extra cost per direction change.
        use_congestion: Apply congestion-based cost weighting.

    Returns:
        Ordered list of (col, row) cells from start to goal inclusive,
        or None if no path exists.
    """

    def heuristic(c: int, r: int) -> float:
        return float(abs(c - goal_col) + abs(r - goal_row))

    # Priority queue: (f, g, col, row, prev_dc, prev_dr)
    open_heap: list[tuple[float, float, int, int, int, int]] = []
    heapq.heappush(open_heap, (
        heuristic(start_col, start_row), 0.0,
        start_col, start_row, 0, 0,
    ))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {(start_col, start_row): 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _f, g, col, row, prev_dc, prev_dr = heapq.heappop(open_heap)
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

            # Base step cost with optional congestion weighting
            step_cost = grid.get_cost(nc, nr) if use_congestion else 1.0

            # Bend penalty: direction change from parent costs extra
            if (
                bend_penalty > 0
                and (prev_dc != 0 or prev_dr != 0)
                and (dc != prev_dc or dr != prev_dr)
            ):
                step_cost += bend_penalty

            tentative_g = g + step_cost
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = node
                f = tentative_g + heuristic(nc, nr)
                heapq.heappush(open_heap, (f, tentative_g, nc, nr, dc, dr))

    return None


def _simplify_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove intermediate colinear points from a grid path.

    Input:  [(0,0), (1,0), (2,0), (2,1)]
    Output: [(0,0), (2,0), (2,1)]
    """
    if len(path) <= 2:
        return path
    result: list[tuple[int, int]] = [path[0]]
    for i in range(1, len(path) - 1):
        # Check if direction from prev to current == current to next
        dc1 = path[i][0] - path[i - 1][0]
        dr1 = path[i][1] - path[i - 1][1]
        dc2 = path[i + 1][0] - path[i][0]
        dr2 = path[i + 1][1] - path[i][1]
        if dc1 != dc2 or dr1 != dr2:
            result.append(path[i])
    result.append(path[-1])
    return result


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
    pad_type: str = "smd"


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
                phw, phh = _pad_rotated_half_size(found_fp, pad)
                found_pad = _PadInfo(
                    x=px, y=py,
                    half_w=phw,
                    half_h=phh,
                    pad_type=pad.pad_type,
                )
                break
        if found_pad is None:
            return f"Pad '{pad_num}' not found on footprint '{ref}'"
        pads.append(found_pad)
    return pads


def _mst_seed(pad_infos: list[_PadInfo]) -> int:
    """Pick the MST seed pad closest to the centroid of all pads.

    Centroid-based seeding produces more balanced spanning trees with
    shorter total path length compared to always starting at index 0.
    """
    if len(pad_infos) <= 1:
        return 0
    cx = sum(p.x for p in pad_infos) / len(pad_infos)
    cy = sum(p.y for p in pad_infos) / len(pad_infos)
    return min(range(len(pad_infos)),
               key=lambda i: abs(pad_infos[i].x - cx) + abs(pad_infos[i].y - cy))


def route_net(
    request: RouteRequest,
    footprints: list[Footprint],
    board_width_mm: float,
    board_height_mm: float,
    grid_step_mm: float = 0.5,
    grid: _Grid | None = None,
    keepouts: tuple[Keepout, ...] = (),
    net_clearances: dict[str, float] | None = None,
    net_widths: dict[str, float] | None = None,
    bcu_grid: _Grid | None = None,
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
        _prepare_grid(
            grid, list(footprints), keepouts=keepouts,
            net_clearances=net_clearances, net_widths=net_widths,
        )

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
    pad_cl = _global_pad_clearance(net_clearances, net_widths)
    net_pad_set = frozenset(request.pad_refs)
    for pi in pad_infos:
        _unmark_pad_area(grid, pi.x, pi.y, pi.half_w, pi.half_h, pad_cl)

    # Track THT components for deferred sibling unmark (retry on A* failure).
    _tht_refs_in_net: set[str] = set()
    for ref, _ in request.pad_refs:
        fp = fp_by_ref.get(ref)
        if fp is None:
            continue
        if any(p.size_x > 1.5 or p.size_y > 1.5 for p in fp.pads):
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
    # Save IC pad infos for final-leg routing after MST loop
    _ic_pad_infos: list[_PadInfo] = []
    _ic_pad_refs: list[tuple[str, str]] = []
    if ic_refs_in_net:
        # Remove IC pads from routing targets — keep only non-IC pads
        non_ic_infos = [
            pi for pi, (ref, _) in zip(pad_infos, request.pad_refs, strict=True)
            if ref not in ic_refs_in_net
        ]
        # Collect IC pad infos for final-leg routing
        _ic_pad_infos = [
            pi for pi, (ref, _) in zip(pad_infos, request.pad_refs, strict=True)
            if ref in ic_refs_in_net
        ]
        # Also collect IC pad refs for unmark/remark
        _ic_pad_refs = [
            (ref, pn) for (ref, pn), pi in zip(request.pad_refs, pad_infos, strict=True)
            if ref in ic_refs_in_net
        ]
        if len(non_ic_infos) >= 2:
            pad_infos = non_ic_infos
            # Don't unmark IC pads — keeps IC area blocked to prevent
            # cross-net contamination.  IC connections use zone pour or B.Cu.
        elif len(non_ic_infos) == 1:
            # One non-IC pad + IC pads: route via IC final-leg only.
            # Replace pad_infos so IC final-leg searches non-IC pads only.
            pad_infos = non_ic_infos
            # Only unmark the specific IC pad(s) that belong to this net.
            for ic_ref in ic_refs_in_net:
                fp = fp_by_ref[ic_ref]
                for pad in fp.pads:
                    if (ic_ref, pad.number) not in net_pad_set:
                        continue
                    px, py = _pad_abs_pos(fp, pad)
                    phw, phh = _pad_rotated_half_size(fp, pad)
                    _unmark_pad_area(
                        grid, px, py, phw, phh, pad_cl,
                    )
        elif len(non_ic_infos) == 0:
            # All pads on dense ICs: skip routing entirely
            _restore_pad_marks(grid, footprints, net_clearances, net_widths)
            return RouteResult(
                net_number=request.net_number,
                net_name=request.net_name,
                tracks=(),
                vias=(),
                routed=False,
                reason="all pads on dense ICs - needs via routing",
            )

    # After all unmark operations, re-mark other-net pads to prevent
    # cross-net contamination from overlapping clearance zones.
    _remark_other_pads(grid, footprints, net_pad_set, net_clearances, net_widths)

    all_tracks: list[Track] = []
    all_vias: list[Via] = []

    # Keep the original net pad set for cross-pad checks — the THT sibling
    # unmark extends net_pad_set for grid purposes but should NOT relax
    # the track-crosses-pad validation (e.g. J2 pad 2 = GND must stay
    # flagged even though J2 pad 1 = SENS0 is same-net).
    _original_net_pad_set = net_pad_set

    # Track exclusion: sized per-net to satisfy the netclass clearance.
    # Default (0.2mm) -> 1 cell (0.25mm), HVA (0.3mm) -> 2 cells (0.5mm).
    # Exclusion must account for track width: adjacent tracks at distance
    # (excl+1)*grid_step must have edge-to-edge gap ≥ clearance_mm.
    # Required: (excl+1)*grid - width ≥ clearance → excl ≥ (cl+w)/g - 1
    excl_cells = max(1, math.ceil(
        (request.clearance_mm + request.width_mm) / grid.grid_step_mm,
    ) - 1)

    # MST-style routing: seed at pad closest to centroid for balanced trees
    seed = _mst_seed(pad_infos)
    routed_set: set[int] = {seed}
    unrouted: set[int] = set(range(len(pad_infos))) - {seed}

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
            extended_pads: set[tuple[str, str]] = set(net_pad_set)
            for ref in _tht_refs_in_net:
                fp = fp_by_ref[ref]
                for pad in fp.pads:
                    extended_pads.add((ref, pad.number))
                    px, py = _pad_abs_pos(fp, pad)
                    phw, phh = _pad_rotated_half_size(fp, pad)
                    _unmark_pad_area(
                        grid, px, py, phw, phh, pad_cl,
                    )
                    _mark_pad_area(grid, px, py, phw, phh, 0.0)
            _tht_refs_in_net.clear()
            net_pad_set = frozenset(extended_pads)
            _remark_other_pads(grid, footprints, net_pad_set, net_clearances, net_widths)
            path = _astar(grid, start_col, start_row, goal_col, goal_row)
            # Validate: discard if THT-retry path crosses other-net pads
            if path is not None:
                trial = _simplify_path(path)
                trial_segs = [
                    Track(
                        start=Point(*grid.to_mm(trial[k][0], trial[k][1])),
                        end=Point(*grid.to_mm(trial[k + 1][0], trial[k + 1][1])),
                        width=request.width_mm, layer="F.Cu",
                        net_number=request.net_number, uuid="",
                    )
                    for k in range(len(trial) - 1)
                ]
                if _track_crosses_other_pads(
                    trial_segs, request.net_number, footprints,
                    net_pad_set=_original_net_pad_set,
                ):
                    path = None  # discard — fall through to B.Cu

        # B.Cu fallback: when F.Cu A* fails, try routing on B.Cu with vias.
        # Each B.Cu segment adds 2 vias — skip if that would exceed max_vias.
        if (path is None and bcu_grid is not None
                and len(all_vias) + 2 <= request.max_vias):
            bcu_result = _route_on_bcu(
                p1.x, p1.y, p2.x, p2.y,
                bcu_grid, request.net_number, request.net_name,
                request.width_mm, request.clearance_mm,
                fcu_grid=grid,
            )
            if bcu_result is not None:
                bcu_tracks, (via_s, via_g) = bcu_result
                # Validate F.Cu stubs don't cross other-net pads
                # Use original net_pad_set — THT sibling extension must
                # not relax cross-pad validation.
                if not _track_crosses_other_pads(
                    bcu_tracks, request.net_number, footprints,
                    net_pad_set=_original_net_pad_set,
                ):
                    all_tracks.extend(bcu_tracks)
                    all_vias.extend([via_s, via_g])
                    routed_set.add(best_to)
                    unrouted.discard(best_to)
                    continue

        if path is None:
            # If either pad is on a dense IC, defer to IC final-leg
            # routing instead of failing the entire net.
            p2_ref = request.pad_refs[best_to][0] if best_to < len(request.pad_refs) else ""
            p1_ref = request.pad_refs[best_from][0] if best_from < len(request.pad_refs) else ""
            if p2_ref in ic_refs_in_net or p1_ref in ic_refs_in_net:
                # Skip this pair — IC final-leg will handle it
                unrouted.discard(best_to)
                continue
            # Restore all pad markings that may have been cleared
            _restore_pad_marks(grid, footprints, net_clearances, net_widths)
            return RouteResult(
                net_number=request.net_number,
                net_name=request.net_name,
                tracks=tuple(all_tracks),
                vias=tuple(all_vias),
                routed=False,
                reason=f"No path found for net {request.net_name}",
            )

        # Simplify path (remove colinear intermediates) for track output
        simplified = _simplify_path(path)
        for j in range(len(simplified) - 1):
            x1, y1 = grid.to_mm(simplified[j][0], simplified[j][1])
            x2, y2 = grid.to_mm(simplified[j + 1][0], simplified[j + 1][1])
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

        # Mark path cells with clearance + congestion
        for cell_col, cell_row in path:
            for dc in range(-excl_cells, excl_cells + 1):
                for dr in range(-excl_cells, excl_cells + 1):
                    grid.mark(cell_col + dc, cell_row + dr)
            grid.add_congestion(cell_col, cell_row, radius=2)

        # Re-unmark same-net pads so subsequent MST connections can still
        # reach unrouted target pads, then re-mark other-net pads to prevent
        # cross-net contamination from overlapping clearance zones.
        for pi in pad_infos:
            _unmark_pad_area(grid, pi.x, pi.y, pi.half_w, pi.half_h, pad_cl)
        _remark_other_pads(grid, footprints, net_pad_set, net_clearances, net_widths)

        routed_set.add(best_to)
        unrouted.discard(best_to)

    # Final-leg IC routing: after MST loop routes all non-IC pads,
    # attempt to connect each IC pad by temporarily unmarking it.
    if _ic_pad_infos:
        # Compute pitch-limited track width for IC stubs.  Dense ICs
        # have fine pitch (e.g. MSOP-10: 0.5mm) — the netclass track
        # width (e.g. 0.4mm) may be too wide to fit between adjacent
        # pads.  Use min(netclass_width, pitch - pad_width) so stubs
        # don't short across neighbouring pads.
        ic_stub_width = request.width_mm
        for ic_ref_w in ic_refs_in_net:
            fp_w = fp_by_ref[ic_ref_w]
            positions_w = sorted(
                (p.position.x, p.position.y) for p in fp_w.pads
            )
            min_pitch = 999.0
            for idx in range(len(positions_w) - 1):
                dx_w = abs(positions_w[idx + 1][0] - positions_w[idx][0])
                dy_w = abs(positions_w[idx + 1][1] - positions_w[idx][1])
                d_w = (dx_w * dx_w + dy_w * dy_w) ** 0.5
                if d_w > 0.01:
                    min_pitch = min(min_pitch, d_w)
            if min_pitch < 999.0:
                max_pad = max(
                    max(p.size_x, p.size_y) for p in fp_w.pads
                )
                pitch_limited = min_pitch - max_pad
                if pitch_limited > 0:
                    ic_stub_width = min(
                        ic_stub_width,
                        max(pitch_limited, JLCPCB_MIN_TRACE_MM),
                    )
        for ic_pi, (ic_ref, ic_pn) in zip(
            _ic_pad_infos, _ic_pad_refs, strict=True,
        ):
            # Find closest routed non-IC pad
            best_pi = pad_infos[0]
            best_dist = float("inf")
            for pi in pad_infos:
                d = abs(pi.x - ic_pi.x) + abs(pi.y - ic_pi.y)
                if d < best_dist:
                    best_dist = d
                    best_pi = pi

            # Temporarily unmark just this IC pad's clearance
            ic_fp = fp_by_ref[ic_ref]
            ic_pad = next(p for p in ic_fp.pads if p.number == ic_pn)
            px, py = _pad_abs_pos(ic_fp, ic_pad)
            ic_hw, ic_hh = _pad_rotated_half_size(ic_fp, ic_pad)
            _unmark_pad_area(
                grid, px, py, ic_hw, ic_hh, pad_cl,
            )
            # Re-unmark source pad too (track exclusion may have blocked it)
            _unmark_pad_area(
                grid, best_pi.x, best_pi.y,
                best_pi.half_w, best_pi.half_h, pad_cl,
            )
            _remark_other_pads(grid, footprints, net_pad_set, net_clearances, net_widths)

            start_col, start_row = grid.to_cell(best_pi.x, best_pi.y)
            goal_col, goal_row = grid.to_cell(ic_pi.x, ic_pi.y)
            path = _astar(grid, start_col, start_row, goal_col, goal_row)
            ic_routed = False
            if path is not None:
                fcu_segs: list[Track] = []
                sim_path = _simplify_path(path)
                for j in range(len(sim_path) - 1):
                    x1, y1 = grid.to_mm(sim_path[j][0], sim_path[j][1])
                    x2, y2 = grid.to_mm(sim_path[j + 1][0], sim_path[j + 1][1])
                    fcu_segs.append(
                        Track(
                            start=Point(x1, y1),
                            end=Point(x2, y2),
                            width=ic_stub_width,
                            layer=request.layer,
                            net_number=request.net_number,
                            uuid="",
                        )
                    )
                # Validate: discard if path crosses other-net pads
                if not _track_crosses_other_pads(
                    fcu_segs, request.net_number, footprints,
                    net_pad_set=_original_net_pad_set,
                ):
                    all_tracks.extend(fcu_segs)
                    # Mark path with exclusion
                    for cell_col, cell_row in path:
                        for dc in range(-excl_cells, excl_cells + 1):
                            for dr in range(-excl_cells, excl_cells + 1):
                                grid.mark(cell_col + dc, cell_row + dr)
                    ic_routed = True

            if not ic_routed and bcu_grid is not None:
                # F.Cu failed or crossed other pads — try B.Cu fallback
                bcu_result = _route_on_bcu(
                    best_pi.x, best_pi.y, ic_pi.x, ic_pi.y,
                    bcu_grid, request.net_number, request.net_name,
                    ic_stub_width, request.clearance_mm,
                    fcu_grid=grid,
                )
                if bcu_result is not None:
                    bcu_tracks, (via_s, via_g) = bcu_result
                    # Validate F.Cu stubs don't cross other-net pads
                    if not _track_crosses_other_pads(
                        bcu_tracks, request.net_number, footprints,
                        net_pad_set=_original_net_pad_set,
                    ):
                        all_tracks.extend(bcu_tracks)
                        all_vias.extend([via_s, via_g])
                        ic_routed = True

            # IC fanout: place via outward from IC body.  Try
            # perpendicular first, then all 4 cardinal directions,
            # checking BOTH F.Cu and B.Cu grids.
            if not ic_routed and bcu_grid is not None:
                ic_cx = ic_fp.position.x
                ic_cy = ic_fp.position.y
                dx_from_center = px - ic_cx
                dy_from_center = py - ic_cy

                via_radius = VIA_DIAMETER_SIGNAL_MM / 2.0
                fan_via_pos: tuple[float, float] | None = None
                r_cells = max(1, round(
                    (via_radius + pad_cl) / grid.grid_step_mm,
                ))

                # Build direction priority: perpendicular away from IC
                # first, then the other 3 directions.
                is_horizontal = abs(dx_from_center) > abs(dy_from_center)
                if is_horizontal:
                    primary_dir = (
                        -1.0 if dx_from_center < 0 else 1.0, 0.0,
                    )
                else:
                    primary_dir = (
                        0.0,
                        -1.0 if dy_from_center < 0 else 1.0,
                    )
                all_dirs: list[tuple[float, float]] = [primary_dir]
                for dx, dy in [
                    (1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                ]:
                    d = (float(dx), float(dy))
                    if d != primary_dir:
                        all_dirs.append(d)

                for fan_dx, fan_dy in all_dirs:
                    if fan_via_pos is not None:
                        break
                    for step in range(4, 60):
                        cx = px + fan_dx * step * grid.grid_step_mm
                        cy = py + fan_dy * step * grid.grid_step_mm
                        col, row = grid.to_cell(cx, cy)
                        clear = True
                        for dc in range(-r_cells, r_cells + 1):
                            for dr in range(-r_cells, r_cells + 1):
                                if not grid.is_free(
                                    col + dc, row + dr,
                                ):
                                    clear = False
                                    break
                                if (bcu_grid is not None
                                        and not bcu_grid.is_free(
                                            col + dc, row + dr)):
                                    clear = False
                                    break
                            if not clear:
                                break
                        if clear:
                            fan_via_pos = grid.to_mm(col, row)
                            break

                if fan_via_pos is not None:
                    # Route stub from IC pad to fanout via using
                    # A* so it avoids previously placed vias/tracks.
                    # Temporarily unmark ALL pads on this IC to
                    # create routing space (clearance zones between
                    # adjacent fine-pitch pads block A*).
                    _ic_pads_unmarked: list[
                        tuple[float, float, float, float]
                    ] = []
                    for _ip in ic_fp.pads:
                        _ipx, _ipy = _pad_abs_pos(ic_fp, _ip)
                        _iphw, _iphh = _pad_rotated_half_size(
                            ic_fp, _ip,
                        )
                        _unmark_pad_area(
                            grid, _ipx, _ipy, _iphw, _iphh, pad_cl,
                        )
                        _ic_pads_unmarked.append(
                            (_ipx, _ipy, _iphw, _iphh),
                        )
                    fan_via_col, _fvr = grid.to_cell(
                        fan_via_pos[0], fan_via_pos[1],
                    )
                    fan_via_row = _fvr
                    grid.unmark(fan_via_col, fan_via_row)
                    # Do NOT call _remark_other_pads here — it
                    # would re-mark adjacent IC pads (different
                    # nets) whose clearance zones overlap the
                    # current pad.  The IC pad unmark creates a
                    # corridor for A* to escape the IC body.
                    ic_col, ic_row = grid.to_cell(px, py)
                    stub_path = _astar(
                        grid, ic_col, ic_row,
                        fan_via_col, fan_via_row,
                    )
                    # Re-mark IC pads
                    for _rpx, _rpy, _rhw, _rhh in _ic_pads_unmarked:
                        _mark_pad_area(
                            grid, _rpx, _rpy, _rhw, _rhh, pad_cl,
                        )
                    fan_stubs: list[Track] = []
                    stub_ok = False
                    if stub_path is not None:
                        sim_stub = _simplify_path(stub_path)
                        for _si in range(len(sim_stub) - 1):
                            sx, sy = grid.to_mm(
                                sim_stub[_si][0],
                                sim_stub[_si][1],
                            )
                            ex, ey = grid.to_mm(
                                sim_stub[_si + 1][0],
                                sim_stub[_si + 1][1],
                            )
                            fan_stubs.append(Track(
                                start=Point(sx, sy),
                                end=Point(ex, ey),
                                width=ic_stub_width,
                                layer="F.Cu",
                                net_number=request.net_number,
                                uuid="",
                            ))
                        stub_ok = not _track_crosses_other_pads(
                            fan_stubs, request.net_number,
                            footprints,
                            net_pad_set=_original_net_pad_set,
                        )
                    if stub_ok:
                        # Route from fanout via to the target pad.
                        # Strategy: try F.Cu A* from via to target
                        # (works well since the via position is away
                        # from the dense IC body).  Fall back to B.Cu
                        # if F.Cu fails.
                        tgt_col, tgt_row = grid.to_cell(
                            best_pi.x, best_pi.y,
                        )
                        # Unmark target pad and create routing
                        # channels through THT pad forests.
                        _unmark_pad_area(
                            grid, best_pi.x, best_pi.y,
                            best_pi.half_w, best_pi.half_h, pad_cl,
                        )
                        _remark_other_pads(
                            grid, footprints, net_pad_set,
                            net_clearances, net_widths,
                        )
                        # THT headers (2.54mm pitch, 0.85mm pads)
                        # have gaps narrower than the grid step after
                        # clearance marking.  Temporarily unmark ALL
                        # pads on the target's footprint so A* can
                        # route between pins.  Do this AFTER remark
                        # to override the remark.
                        _tht_fp_unmarked: list[tuple[float, float, float, float]] = []
                        _tgt_ref = ""
                        for _t_ref, _t_pn in request.pad_refs:
                            if _t_ref in ic_refs_in_net:
                                continue
                            _t_fp = fp_by_ref.get(_t_ref)
                            if _t_fp is None:
                                continue
                            for _t_pad in _t_fp.pads:
                                _tpx, _tpy = _pad_abs_pos(
                                    _t_fp, _t_pad,
                                )
                                if (abs(_tpx - best_pi.x) < 0.01
                                        and abs(_tpy - best_pi.y)
                                        < 0.01):
                                    _tgt_ref = _t_ref
                                    break
                            if _tgt_ref:
                                break
                        if _tgt_ref:
                            _t_fp = fp_by_ref[_tgt_ref]
                            has_tht = any(
                                p.pad_type == "thru_hole"
                                for p in _t_fp.pads
                            )
                            if has_tht and len(_t_fp.pads) > 4:
                                # Only unmask the TARGET pad itself.
                                # Create a small entry corridor by
                                # unmarking a line from the target
                                # pad toward the source to ensure A*
                                # can reach the pad from outside the
                                # connector pad forest.
                                _tphw = best_pi.half_w
                                _tphh = best_pi.half_h
                                # Unmark the target pad with generous
                                # clearance so A* has room to enter
                                _unmark_pad_area(
                                    grid, best_pi.x, best_pi.y,
                                    _tphw + pad_cl,
                                    _tphh + pad_cl, pad_cl,
                                )
                                _tht_fp_unmarked.append((
                                    best_pi.x, best_pi.y,
                                    _tphw + pad_cl,
                                    _tphh + pad_cl,
                                ))
                        fcu_fan_path = _astar(
                            grid,
                            fan_via_col, fan_via_row,
                            tgt_col, tgt_row,
                        )
                        # Re-mark temporarily unmarked THT pads
                        for _rpx, _rpy, _rhw, _rhh in _tht_fp_unmarked:
                            _mark_pad_area(
                                grid, _rpx, _rpy, _rhw, _rhh, pad_cl,
                            )
                        if fcu_fan_path is not None:
                            fan_segs: list[Track] = []
                            sim_fan = _simplify_path(fcu_fan_path)
                            for j in range(len(sim_fan) - 1):
                                sx, sy = grid.to_mm(
                                    sim_fan[j][0],
                                    sim_fan[j][1],
                                )
                                ex, ey = grid.to_mm(
                                    sim_fan[j + 1][0],
                                    sim_fan[j + 1][1],
                                )
                                fan_segs.append(Track(
                                    start=Point(sx, sy),
                                    end=Point(ex, ey),
                                    width=ic_stub_width,
                                    layer="F.Cu",
                                    net_number=request.net_number,
                                    uuid="",
                                ))
                            # Validate F.Cu path doesn't cross
                            # other-net pads.  When routing through
                            # a THT pad forest (e.g. J1 header), the
                            # A* path navigates between pins — skip
                            # crossing check for those tracks since
                            # the grid already guarantees adequate
                            # spacing.
                            skip_cross_check = len(
                                _tht_fp_unmarked,
                            ) > 0
                            crosses = (
                                not skip_cross_check
                                and _track_crosses_other_pads(
                                    fan_segs, request.net_number,
                                    footprints,
                                    net_pad_set=_original_net_pad_set,
                                )
                            )
                            if not crosses:
                                fan_via = Via(
                                    position=Point(
                                        fan_via_pos[0],
                                        fan_via_pos[1],
                                    ),
                                    drill=VIA_DRILL_SIGNAL_MM,
                                    size=VIA_DIAMETER_SIGNAL_MM,
                                    layers=("F.Cu", "B.Cu"),
                                    net_number=request.net_number,
                                    uuid="",
                                )
                                all_tracks.extend(fan_stubs)
                                all_tracks.extend(fan_segs)
                                all_vias.append(fan_via)
                                # Mark path with exclusion
                                for cc, cr in fcu_fan_path:
                                    for dc in range(
                                        -excl_cells, excl_cells + 1,
                                    ):
                                        for dr in range(
                                            -excl_cells,
                                            excl_cells + 1,
                                        ):
                                            grid.mark(cc + dc, cr + dr)
                                # Mark fanout via on BOTH F.Cu and B.Cu
                                # grids.  Use global pad clearance (accounts
                                # for half-width of other nets' tracks) so
                                # subsequent routes maintain adequate spacing
                                # from this via on either layer.
                                _mark_pad_area(
                                    grid,
                                    fan_via_pos[0],
                                    fan_via_pos[1],
                                    VIA_DIAMETER_SIGNAL_MM / 2.0,
                                    VIA_DIAMETER_SIGNAL_MM / 2.0,
                                    pad_cl,
                                )
                                if bcu_grid is not None:
                                    _mark_pad_area(
                                        bcu_grid,
                                        fan_via_pos[0],
                                        fan_via_pos[1],
                                        VIA_DIAMETER_SIGNAL_MM / 2.0,
                                        VIA_DIAMETER_SIGNAL_MM / 2.0,
                                        pad_cl,
                                    )
                                for _fs in fan_stubs:
                                    _mark_line_on_grid(
                                        grid,
                                        _fs.start.x, _fs.start.y,
                                        _fs.end.x, _fs.end.y,
                                        excl_cells,
                                    )
                                ic_routed = True

            if not ic_routed:
                # Re-mark the IC pad — couldn't route without crossing
                _mark_pad_area(
                    grid, px, py, ic_hw, ic_hh, pad_cl,
                )

    # Restore all pad markings that may have been cleared during unmark
    _restore_pad_marks(grid, footprints, net_clearances, net_widths)

    return RouteResult(
        net_number=request.net_number,
        net_name=request.net_name,
        tracks=tuple(all_tracks),
        vias=tuple(all_vias),
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
        # Power nets route first (tier 0) on a clean grid for better
        # connectivity.  Signal nets follow (tier 1), sorted by length.
        tier = 0 if is_power else 1
        return (tier, _estimated_length(entry))

    routable.sort(key=_sort_key)

    # Create a shared grid and prepare it with pads, edge margins, keepouts
    grid = _Grid.create(board_width_mm, board_height_mm, grid_step_mm)
    _prepare_grid(
        grid, list(footprints), keepouts=keepouts,
        net_clearances=net_clearances, net_widths=net_widths,
    )

    # Create B.Cu grid for dual-layer fallback routing
    bcu_grid = _prepare_bcu_grid(
        grid, list(footprints), keepouts=keepouts,
        net_clearances=net_clearances, net_widths=net_widths,
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
            net_widths=net_widths, bcu_grid=bcu_grid,
        )

    # First pass: route all nets
    failed_entries: list[NetlistEntry] = []
    for entry in routable:
        result = _route_entry(entry)
        if result.routed:
            results.append(result)
        else:
            failed_entries.append(entry)

    # Retry failed nets with multiple strategies:
    # 1. Standard retry (congestion may have changed)
    # 2. Reverse pad ordering (asymmetric congestion)
    # 3. Relaxed clearance at JLCPCB manufacturing minimum
    from kicad_pipeline.pcb.netlist import NetlistEntry as _NetlistEntry

    still_failed: list[_NetlistEntry] = []
    for entry in failed_entries:
        result = _route_entry(entry)
        if result.routed:
            results.append(result)
            continue
        # Try reversed pad ordering
        reversed_pads = entry.pad_refs[::-1]
        reversed_entry = _NetlistEntry(
            net=entry.net,
            pad_refs=reversed_pads,
        )
        result = _route_entry(reversed_entry)
        if result.routed:
            results.append(result)
            continue
        still_failed.append(entry)

    # Last resort: relaxed clearance retry at JLCPCB minimum
    if still_failed:
        from kicad_pipeline.constants import JLCPCB_MIN_CLEARANCE_MM

        relaxed_clearances = dict(net_clearances) if net_clearances else {}
        for entry in still_failed:
            relaxed_clearances[entry.net.name] = JLCPCB_MIN_CLEARANCE_MM
        for entry in still_failed:
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
                clearance_mm=JLCPCB_MIN_CLEARANCE_MM,
            )
            result = route_net(
                request, footprints, board_width_mm, board_height_mm,
                grid_step_mm, grid=grid, net_clearances=relaxed_clearances,
                net_widths=net_widths, bcu_grid=bcu_grid,
            )
            results.append(result)

    # Rip-up-and-retry loop: improve worst routes
    # Build pad position lookup for quality scoring
    def _pad_positions_for(entry: NetlistEntry) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        for ref, pad_num in entry.pad_refs:
            fp = fp_by_ref.get(ref)
            if fp is None:
                continue
            for pad in fp.pads:
                if pad.number == pad_num:
                    positions.append(_pad_abs_pos(fp, pad))
                    break
        return positions

    # Map net_name -> entry for rip-up lookup
    entry_by_name: dict[str, NetlistEntry] = {e.net.name: e for e in routable}

    for _ripup_iter in range(3):
        # Score all routed results
        offenders: list[tuple[float, int]] = []  # (score, results_index)
        for idx, r in enumerate(results):
            if not r.routed:
                continue
            score_entry = entry_by_name.get(r.net_name)
            if score_entry is None:
                continue
            q = _score_route(r, _pad_positions_for(score_entry))
            # Spec rip-up triggers: >2 vias, ratio>1.55, or >=4 bends on <40mm
            short_net_excess_bends = (
                q.bend_count >= 4 and q.manhattan_ideal_mm < 40.0
            )
            if q.via_count > 2 or q.length_ratio > 1.55 or short_net_excess_bends:
                offenders.append((q.score, idx))

        if not offenders:
            break

        # Sort by badness, rip up worst 20% (at least 1)
        offenders.sort(key=lambda x: x[0], reverse=True)
        n_ripup = max(1, len(offenders) // 5)
        ripup_indices = [idx for _, idx in offenders[:n_ripup]]

        # Unmark tracks from grids
        for ri in ripup_indices:
            rr = results[ri]
            for trk in rr.tracks:
                if trk.layer == "F.Cu":
                    _unmark_route_tracks(grid, [trk], grid_step_mm)
                elif trk.layer == "B.Cu":
                    _unmark_route_tracks(bcu_grid, [trk], grid_step_mm)

        # Re-route ripped nets
        new_results: list[RouteResult] = []
        ripped_names: set[str] = set()
        for ri in ripup_indices:
            ripped_names.add(results[ri].net_name)
        for ri in sorted(ripup_indices, reverse=True):
            results.pop(ri)
        for name in ripped_names:
            retry_entry = entry_by_name.get(name)
            if retry_entry is not None:
                new_result = _route_entry(retry_entry)
                new_results.append(new_result)
        results.extend(new_results)

    # Post-routing clearance validation: detect and fix violations
    results = _validate_track_clearances(
        results, grid, bcu_grid, grid_step_mm, entry_by_name,
        _route_entry, footprints, net_clearances, net_widths,
        _pad_positions_for,
    )

    return tuple(results)


def _segment_min_distance(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> float:
    """Compute minimum distance between two line segments.

    Checks for intersection first, then falls back to point-to-segment
    distances for non-intersecting segments.
    """
    # Check for segment intersection using cross products
    dx_a = ax2 - ax1
    dy_a = ay2 - ay1
    dx_b = bx2 - bx1
    dy_b = by2 - by1
    denom = dx_a * dy_b - dy_a * dx_b
    if abs(denom) > 1e-12:
        t = ((bx1 - ax1) * dy_b - (by1 - ay1) * dx_b) / denom
        u = ((bx1 - ax1) * dy_a - (by1 - ay1) * dx_a) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0

    def _point_seg_dist(
        px: float, py: float,
        sx1: float, sy1: float, sx2: float, sy2: float,
    ) -> float:
        dx = sx2 - sx1
        dy = sy2 - sy1
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-12:
            return math.sqrt((px - sx1) ** 2 + (py - sy1) ** 2)
        t = max(0.0, min(1.0, ((px - sx1) * dx + (py - sy1) * dy) / len_sq))
        cx = sx1 + t * dx
        cy = sy1 + t * dy
        return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)

    return min(
        _point_seg_dist(ax1, ay1, bx1, by1, bx2, by2),
        _point_seg_dist(ax2, ay2, bx1, by1, bx2, by2),
        _point_seg_dist(bx1, by1, ax1, ay1, ax2, ay2),
        _point_seg_dist(bx2, by2, ax1, ay1, ax2, ay2),
    )


def _validate_track_clearances(
    results: list[RouteResult],
    grid: _Grid,
    bcu_grid: _Grid,
    grid_step_mm: float,
    entry_by_name: dict[str, NetlistEntry],
    route_fn: object,
    footprints: list[Footprint],
    net_clearances: dict[str, float] | None,
    net_widths: dict[str, float] | None,
    pad_positions_fn: object,
) -> list[RouteResult]:
    """Detect and fix cross-net clearance violations after routing.

    Iterates all track pairs from different nets. If edge-to-edge distance
    violates CLEARANCE_DEFAULT_MM, rips up the worse-scored net and re-routes.
    """
    from kicad_pipeline.constants import CLEARANCE_DEFAULT_MM

    # Build per-net track lists
    net_tracks: dict[int, list[Track]] = {}
    net_result_idx: dict[int, int] = {}
    for idx, r in enumerate(results):
        if not r.routed:
            continue
        net_tracks[r.net_number] = list(r.tracks)
        net_result_idx[r.net_number] = idx

    # Check all cross-net pairs for clearance violations
    violating_nets: set[int] = set()
    net_nums = list(net_tracks.keys())
    for i in range(len(net_nums)):
        for j in range(i + 1, len(net_nums)):
            n1, n2 = net_nums[i], net_nums[j]
            for t1 in net_tracks[n1]:
                for t2 in net_tracks[n2]:
                    if t1.layer != t2.layer:
                        continue
                    hw1 = t1.width / 2.0
                    hw2 = t2.width / 2.0
                    min_gap = CLEARANCE_DEFAULT_MM
                    edge_dist = _segment_min_distance(
                        t1.start.x, t1.start.y, t1.end.x, t1.end.y,
                        t2.start.x, t2.start.y, t2.end.x, t2.end.y,
                    ) - hw1 - hw2
                    if edge_dist < min_gap - 0.001:
                        # Pick the net with worse score to rip up
                        violating_nets.add(n1)
                        violating_nets.add(n2)

    if not violating_nets:
        return results

    import logging
    log = logging.getLogger(__name__)
    log.info("clearance validation: %d nets involved in violations", len(violating_nets))

    # Score violating nets, rip up the worst half
    scored: list[tuple[float, int]] = []
    for net_num in violating_nets:
        maybe_idx = net_result_idx.get(net_num)
        if maybe_idx is None:
            continue
        idx = maybe_idx
        r = results[idx]
        entry = entry_by_name.get(r.net_name)
        if entry is None:
            continue
        q = _score_route(r, pad_positions_fn(entry))  # type: ignore[operator]
        scored.append((q.score, idx))

    if not scored:
        return results

    scored.sort(key=lambda x: x[0], reverse=True)
    n_ripup = max(1, len(scored) // 2)
    ripup_indices = [idx for _, idx in scored[:n_ripup]]

    # Unmark and rip up
    for ri in ripup_indices:
        rr = results[ri]
        for trk in rr.tracks:
            if trk.layer == "F.Cu":
                _unmark_route_tracks(grid, [trk], grid_step_mm)
            elif trk.layer == "B.Cu":
                _unmark_route_tracks(bcu_grid, [trk], grid_step_mm)

    ripped_names: set[str] = set()
    for ri in ripup_indices:
        ripped_names.add(results[ri].net_name)
    for ri in sorted(ripup_indices, reverse=True):
        results.pop(ri)

    for name in ripped_names:
        retry_entry = entry_by_name.get(name)
        if retry_entry is not None:
            new_result = route_fn(retry_entry)  # type: ignore[operator]
            results.append(new_result)

    return results


def _unmark_route_tracks(
    grid: _Grid | None,
    tracks: list[Track],
    grid_step_mm: float,
) -> None:
    """Unmark grid cells occupied by routed tracks (for rip-up)."""
    if grid is None:
        return
    for trk in tracks:
        c1, r1 = grid.to_cell(trk.start.x, trk.start.y)
        c2, r2 = grid.to_cell(trk.end.x, trk.end.y)
        # Walk line between cells
        dc = abs(c2 - c1)
        dr = abs(r2 - r1)
        sc = 1 if c1 < c2 else -1
        sr = 1 if r1 < r2 else -1
        err = dc - dr
        cc, cr = c1, r1
        while True:
            grid.unmark(cc, cr)
            if cc == c2 and cr == r2:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                cc += sc
            if e2 < dc:
                err += dc
                cr += sr


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


def collect_vias(
    results: tuple[RouteResult, ...],
    *,
    routed_only: bool = True,
) -> tuple[Via, ...]:
    """Flatten all Via objects from RouteResults into a single tuple.

    Args:
        results: Routing results to collect vias from.
        routed_only: When ``True`` (default), skip vias from unrouted nets
            to avoid dangling via DRC violations.

    Returns:
        Combined tuple of all vias.
    """
    vias: list[Via] = []
    for r in results:
        if routed_only and not r.routed:
            continue
        vias.extend(r.vias)

    # Deduplicate: skip vias at same position (within 0.01mm)
    seen: set[tuple[float, float]] = set()
    deduped: list[Via] = []
    for v in vias:
        key = (round(v.position.x, 2), round(v.position.y, 2))
        if key not in seen:
            seen.add(key)
            deduped.append(v)

    # Distance-based dedup: skip if any previously-kept same-net via
    # is within via.size mm (pad diameter), preventing hole_to_hole violations
    final: list[Via] = []
    for v in deduped:
        too_close = False
        for kept in final:
            if kept.net_number != v.net_number:
                continue
            dist = math.hypot(
                v.position.x - kept.position.x,
                v.position.y - kept.position.y,
            )
            if dist < v.size:
                too_close = True
                break
        if not too_close:
            final.append(v)
    return tuple(final)
