"""Grid-based PCB autorouter using A* pathfinding.

Provides a simple 2-D occupancy grid and an A* path finder that can route
copper tracks between pad pairs on a single copper layer.  Intended as a
lightweight fallback when FreeRouting is unavailable.
"""

from __future__ import annotations

import heapq
import logging
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

_log = logging.getLogger(__name__)


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

    def save_state(self) -> list[list[bool]]:
        """Return a snapshot of the occupancy grid (deep copy)."""
        return [col[:] for col in self._cells]

    def restore_state(self, state: list[list[bool]]) -> None:
        """Restore occupancy grid from a snapshot."""
        for c in range(self.cols):
            self._cells[c][:] = state[c]

    def mark_area(self, col: int, row: int, radius: int) -> None:
        """Mark all cells within *radius* of (col, row), clamped to grid bounds."""
        c_lo = max(0, col - radius)
        c_hi = min(self.cols, col + radius + 1)
        r_lo = max(0, row - radius)
        r_hi = min(self.rows, row + radius + 1)
        cells = self._cells
        for cc in range(c_lo, c_hi):
            col_cells = cells[cc]
            for rr in range(r_lo, r_hi):
                col_cells[rr] = True

    def unmark_area(self, col: int, row: int, radius: int) -> None:
        """Unmark all cells within *radius* of (col, row), clamped to grid bounds."""
        c_lo = max(0, col - radius)
        c_hi = min(self.cols, col + radius + 1)
        r_lo = max(0, row - radius)
        r_hi = min(self.rows, row + radius + 1)
        cells = self._cells
        for cc in range(c_lo, c_hi):
            col_cells = cells[cc]
            for rr in range(r_lo, r_hi):
                col_cells[rr] = False

    def mark_mm(self, x_mm: float, y_mm: float, radius_cells: int = 1) -> None:
        """Mark cell and all neighbors within radius_cells (Manhattan distance)."""
        base_col, base_row = self.to_cell(x_mm, y_mm)
        self.mark_area(base_col, base_row, radius_cells)

    def unmark_mm(self, x_mm: float, y_mm: float, radius_cells: int = 1) -> None:
        """Unmark cell and all neighbors within radius_cells (Manhattan distance)."""
        base_col, base_row = self.to_cell(x_mm, y_mm)
        self.unmark_area(base_col, base_row, radius_cells)

    def add_congestion(self, col: int, row: int, radius: int = 1) -> None:
        """Increment congestion counter around a cell."""
        c_lo = max(0, col - radius)
        c_hi = min(self.cols, col + radius + 1)
        r_lo = max(0, row - radius)
        r_hi = min(self.rows, row + radius + 1)
        congestion = self._congestion
        for nc in range(c_lo, c_hi):
            col_cong = congestion[nc]
            for nr in range(r_lo, r_hi):
                col_cong[nr] += 1

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


@dataclass(frozen=True)
class _CachedPad:
    """Pre-computed absolute position and rotated half-sizes for a pad."""

    ref: str
    pad_number: str
    x: float
    y: float
    half_w: float
    half_h: float
    pad_type: str
    net_number: int | None


def _build_pad_cache(footprints: list[Footprint]) -> list[_CachedPad]:
    """Pre-compute absolute positions and rotated sizes for all pads.

    Avoids repeated trigonometry in inner loops by computing pad world
    positions once up front.
    """
    cache: list[_CachedPad] = []
    for fp in footprints:
        rad = math.radians(-fp.rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        rot_mod = fp.rotation % 360.0
        for pad in fp.pads:
            # Inline _pad_abs_pos
            rx = pad.position.x * cos_r - pad.position.y * sin_r
            ry = pad.position.x * sin_r + pad.position.y * cos_r
            px = fp.position.x + rx
            py = fp.position.y + ry
            # Inline _pad_rotated_half_size
            hw = pad.size_x / 2.0
            hh = pad.size_y / 2.0
            if abs(rot_mod - 90.0) < 0.01 or abs(rot_mod - 270.0) < 0.01:
                phw, phh = hh, hw
            elif abs(rot_mod) < 0.01 or abs(rot_mod - 180.0) < 0.01:
                phw, phh = hw, hh
            else:
                cos_a = abs(math.cos(math.radians(rot_mod)))
                sin_a = abs(math.sin(math.radians(rot_mod)))
                phw = hw * cos_a + hh * sin_a
                phh = hw * sin_a + hh * cos_a
            cache.append(_CachedPad(
                ref=fp.ref,
                pad_number=pad.number,
                x=px, y=py,
                half_w=phw, half_h=phh,
                pad_type=pad.pad_type,
                net_number=pad.net_number,
            ))
    return cache

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
    """Compute the global pad clearance for grid marking.

    Uses the max netclass clearance + half the max track width to
    prevent all clearance violations.
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
    allow_same_ref: str | None = None,
    _pad_cache: list[_CachedPad] | None = None,
) -> bool:
    """Return True if any F.Cu track crosses a pad on a different net.

    Used to validate IC final-leg stubs: if a B.Cu fallback creates an
    F.Cu stub that crosses another IC pad, the connection should be
    discarded rather than creating a DRC short.

    Args:
        net_pad_set: If provided, identifies same-net pads by (ref, pad_number).
            Used during routing when pad.net_number is not yet assigned.
        allow_same_ref: If provided, allow crossings with pads on this
            component (e.g. IC ref).  Intra-footprint clearance violations
            are handled by DRC exclusions for dense ICs.
        _pad_cache: Optional pre-computed pad cache to avoid repeated
            trigonometry. Built via ``_build_pad_cache()``.
    """
    # Filter F.Cu tracks once and precompute their AABBs
    fcu_tracks: list[tuple[float, float, float, float]] = []
    for t in tracks:
        if t.layer != "F.Cu":
            continue
        hw = t.width / 2.0
        fcu_tracks.append((
            min(t.start.x, t.end.x) - hw,
            max(t.start.x, t.end.x) + hw,
            min(t.start.y, t.end.y) - hw,
            max(t.start.y, t.end.y) + hw,
        ))
    if not fcu_tracks:
        return False

    # Use pad cache if available, otherwise compute on the fly
    if _pad_cache is not None:
        pads_iter = _pad_cache
    else:
        pads_iter = _build_pad_cache(footprints)

    for cp in pads_iter:
        if allow_same_ref is not None and cp.ref == allow_same_ref:
            continue
        # Skip pads on the same net
        if net_pad_set is not None:
            if (cp.ref, cp.pad_number) in net_pad_set:
                continue
        elif cp.net_number is not None and cp.net_number == net_number:
            continue
        pad_x0 = cp.x - cp.half_w - clearance_mm
        pad_x1 = cp.x + cp.half_w + clearance_mm
        pad_y0 = cp.y - cp.half_h - clearance_mm
        pad_y1 = cp.y + cp.half_h + clearance_mm
        for tx0, tx1, ty0, ty1 in fcu_tracks:
            if tx1 > pad_x0 and tx0 < pad_x1 and ty1 > pad_y0 and ty0 < pad_y1:
                return True
    return False


def _restore_pad_marks(
    grid: _Grid,
    footprints: list[Footprint],
    net_clearances: dict[str, float] | None = None,
    net_widths: dict[str, float] | None = None,
    _pad_cache: list[_CachedPad] | None = None,
) -> None:
    """Re-mark all pad areas after temporarily clearing same-net pads.

    This prevents cross-net contamination: when clearing pad A's clearance
    zone for routing, nearby pad B's zone might also get cleared.  After
    routing, this function restores ALL pad marks with correct clearances.
    """
    cl = _global_pad_clearance(net_clearances, net_widths)
    pads = _pad_cache if _pad_cache is not None else _build_pad_cache(footprints)
    for cp in pads:
        _mark_pad_area(grid, cp.x, cp.y, cp.half_w, cp.half_h, cl)


def _remark_other_pads(
    grid: _Grid,
    footprints: list[Footprint],
    net_pad_set: frozenset[tuple[str, str]],
    net_clearances: dict[str, float] | None = None,
    net_widths: dict[str, float] | None = None,
    _pad_cache: list[_CachedPad] | None = None,
) -> None:
    """Re-mark pads NOT in the current net to prevent cross-net contamination.

    When unmarking same-net pads' clearance zones, nearby pads on different
    nets may have their clearance zones partially cleared (overlap).  This
    function re-marks all non-current-net pads to restore correct blocking.
    """
    cl = _global_pad_clearance(net_clearances, net_widths)
    pads = _pad_cache if _pad_cache is not None else _build_pad_cache(footprints)
    for cp in pads:
        if (cp.ref, cp.pad_number) in net_pad_set:
            continue
        _mark_pad_area(grid, cp.x, cp.y, cp.half_w, cp.half_h, cl)


def _mark_edge_margins(grid: _Grid) -> None:
    """Mark board-edge margin cells as occupied."""
    margin_cells = max(1, int(JLCPCB_BOARD_EDGE_CLEARANCE_MM / grid.grid_step_mm) + 1)
    for col in range(grid.cols):
        for mr in range(margin_cells):
            grid.mark(col, mr)
            grid.mark(col, grid.rows - 1 - mr)
    for row in range(grid.rows):
        for mc in range(margin_cells):
            grid.mark(mc, row)
            grid.mark(grid.cols - 1 - mc, row)


def _mark_corner_arcs(grid: _Grid, corner_radius_mm: float) -> None:
    """Mark rounded-corner regions as occupied so tracks stay inside the arc."""
    if corner_radius_mm <= 0:
        return
    r_cells = math.ceil(
        (corner_radius_mm + JLCPCB_BOARD_EDGE_CLEARANCE_MM) / grid.grid_step_mm,
    )
    corners = [
        (r_cells, r_cells),
        (grid.cols - 1 - r_cells, r_cells),
        (r_cells, grid.rows - 1 - r_cells),
        (grid.cols - 1 - r_cells, grid.rows - 1 - r_cells),
    ]
    gs = grid.grid_step_mm
    for cx, cy in corners:
        for dc in range(-r_cells, r_cells + 1):
            for dr in range(-r_cells, r_cells + 1):
                cc, cr = cx + dc, cy + dr
                if 0 <= cc < grid.cols and 0 <= cr < grid.rows:
                    dist = math.hypot((cc - cx) * gs, (cr - cy) * gs)
                    if dist > corner_radius_mm:
                        grid.mark(cc, cr)


def _mark_keepouts(grid: _Grid, keepouts: tuple[Keepout, ...], layer: str) -> None:
    """Mark keepout zones as occupied on the given layer."""
    for ko in keepouts:
        if not ko.polygon:
            continue
        if not _keepout_blocks_layer(ko, layer):
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


def _prepare_grid(
    grid: _Grid,
    footprints: list[Footprint],
    keepouts: tuple[Keepout, ...] = (),
    net_clearances: dict[str, float] | None = None,
    net_widths: dict[str, float] | None = None,
    corner_radius_mm: float = 0.0,
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
    _pad_mark_cl = _global_pad_clearance(net_clearances, net_widths)
    pad_cache = _build_pad_cache(footprints)
    for cp in pad_cache:
        _mark_pad_area(grid, cp.x, cp.y, cp.half_w, cp.half_h, _pad_mark_cl)

    _mark_edge_margins(grid)
    _mark_corner_arcs(grid, corner_radius_mm)
    _mark_keepouts(grid, keepouts, "F.Cu")


def _prepare_bcu_grid(
    grid: _Grid,
    footprints: list[Footprint],
    keepouts: tuple[Keepout, ...] = (),
    net_clearances: dict[str, float] | None = None,
    net_widths: dict[str, float] | None = None,
    corner_radius_mm: float = 0.0,
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

    _mark_edge_margins(bcu)
    _mark_corner_arcs(bcu, corner_radius_mm)
    _mark_keepouts(bcu, keepouts, "B.Cu")

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
        grid.mark_area(cell_col, cell_row, excl_cells)

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
        return _is_area_free(fcu_grid, col, row, excl_radius, bcu_grid)

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
    max_ring = 40  # ~10 mm at 0.25 mm grid
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


def _find_via_endpoint(
    pad_x: float,
    pad_y: float,
    is_tht: bool,
    via_in_pad: bool,
    via_radius: float,
    clearance_mm: float,
    width_mm: float,
    fcu_grid: _Grid | None,
    bcu_grid: _Grid,
    net_name: str,
    label: str,
) -> tuple[tuple[float, float], bool] | None:
    """Find a via position for one endpoint of a B.Cu route.

    Returns ``(via_position, needs_astar_stub)`` or ``None`` if no free
    position can be found.
    """
    if fcu_grid is None or is_tht or via_in_pad:
        return (pad_x, pad_y), False

    # Try with clear stub path on both layers
    found = _find_free_via_position(
        fcu_grid, pad_x, pad_y, via_radius, clearance_mm,
        stub_origin=(pad_x, pad_y), stub_width_mm=width_mm,
        bcu_grid=bcu_grid,
    )
    if found is None:
        # Any free position, A* for stub
        found = _find_free_via_position(
            fcu_grid, pad_x, pad_y, via_radius, clearance_mm,
            bcu_grid=bcu_grid,
        )
    if found is None:
        # Last resort: F.Cu-only (accept B.Cu congestion)
        found = _find_free_via_position(
            fcu_grid, pad_x, pad_y, via_radius, clearance_mm,
        )
        if found is None:
            _log.debug(
                "B.Cu via search FAIL %s (%.1f,%.1f) net=%s",
                label, pad_x, pad_y, net_name,
            )
            return None
    return found, True


def _build_bcu_tracks(
    path: list[tuple[int, int]],
    bcu_grid: _Grid,
    width_mm: float,
    net_number: int,
) -> list[Track]:
    """Convert an A* path on B.Cu into simplified track segments."""
    tracks: list[Track] = []
    sim_bcu = _simplify_path(path)
    for j in range(len(sim_bcu) - 1):
        x1, y1 = bcu_grid.to_mm(sim_bcu[j][0], sim_bcu[j][1])
        x2, y2 = bcu_grid.to_mm(sim_bcu[j + 1][0], sim_bcu[j + 1][1])
        tracks.append(
            Track(
                start=Point(x1, y1), end=Point(x2, y2),
                width=width_mm, layer="B.Cu",
                net_number=net_number, uuid="",
            )
        )
    return tracks


def _build_endpoint_via(
    via_pos: tuple[float, float],
    is_tht: bool,
    via_drill: float,
    via_size: float,
    net_number: int,
) -> Via | None:
    """Create a via at *via_pos* unless the endpoint is THT."""
    if is_tht:
        return None
    return Via(
        position=Point(via_pos[0], via_pos[1]),
        drill=via_drill, size=via_size,
        layers=("F.Cu", "B.Cu"),
        net_number=net_number, uuid="",
    )


def _build_stub_track(
    pad_x: float, pad_y: float,
    via_x: float, via_y: float,
    needs_astar: bool,
    fcu_grid: _Grid | None,
    net_number: int,
    width_mm: float,
    clearance_mm: float,
) -> list[Track]:
    """Build F.Cu stub tracks bridging a pad to an offset via.

    Uses A* routing when *needs_astar* is True and *fcu_grid* is available,
    falling back to a direct diagonal stub.
    """
    is_offset = abs(via_x - pad_x) > 0.01 or abs(via_y - pad_y) > 0.01
    if not is_offset:
        return []

    if needs_astar and fcu_grid is not None:
        astar_stub = _route_stub_on_fcu(
            fcu_grid, pad_x, pad_y, via_x, via_y,
            net_number, width_mm, clearance_mm,
        )
        if astar_stub is not None:
            return list(astar_stub)

    # Direct stub as fallback
    return [
        Track(
            start=Point(pad_x, pad_y), end=Point(via_x, via_y),
            width=width_mm, layer="F.Cu",
            net_number=net_number, uuid="",
        )
    ]


def _finalize_bcu_route(
    path: list[tuple[int, int]],
    tracks: list[Track],
    stub_tracks: list[Track],
    via_start: Via | None,
    via_goal: Via | None,
    bcu_grid: _Grid,
    fcu_grid: _Grid | None,
    width_mm: float,
    clearance_mm: float,
    via_radius: float,
) -> tuple[tuple[Track, ...], tuple[Via, ...]]:
    """Mark grids and assemble the final tracks/vias for a B.Cu route."""
    # Mark F.Cu stub tracks on fcu_grid
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
        bcu_grid.mark_area(cell_col, cell_row, excl_cells)

    # Mark via exclusion on both grids
    emitted_vias: list[Via] = []
    via_excl = math.ceil((via_radius + clearance_mm) / bcu_grid.grid_step_mm)
    for via in (via_start, via_goal):
        if via is None:
            continue
        emitted_vias.append(via)
        if fcu_grid is not None:
            _mark_via_on_fcu(fcu_grid, via, clearance_mm)
        vc, vr = bcu_grid.to_cell(via.position.x, via.position.y)
        bcu_grid.mark_area(vc, vr, via_excl)

    all_tracks = list(stub_tracks) + list(tracks)
    return (tuple(all_tracks), tuple(emitted_vias))


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
    start_is_tht: bool = False,
    goal_is_tht: bool = False,
    start_via_in_pad: bool = False,
    goal_via_in_pad: bool = False,
) -> tuple[tuple[Track, ...], tuple[Via, ...]] | None:
    """Route a segment on B.Cu with vias at each end.

    Finds free via positions, runs A* on B.Cu, and adds F.Cu stub tracks
    to bridge offset vias.  Returns ``(tracks, vias)`` or ``None``.
    """
    via_drill = VIA_DRILL_SIGNAL_MM
    via_size = VIA_DIAMETER_SIGNAL_MM
    via_radius = via_size / 2.0

    # Find via positions for start and goal
    start_result = _find_via_endpoint(
        start_x, start_y, start_is_tht, start_via_in_pad,
        via_radius, clearance_mm, width_mm, fcu_grid, bcu_grid, net_name, "start",
    )
    if start_result is None:
        return None
    via_start_pos, _start_needs_astar_stub = start_result

    goal_result = _find_via_endpoint(
        goal_x, goal_y, goal_is_tht, goal_via_in_pad,
        via_radius, clearance_mm, width_mm, fcu_grid, bcu_grid, net_name, "goal",
    )
    if goal_result is None:
        return None
    via_goal_pos, _goal_needs_astar_stub = goal_result

    # Route on B.Cu between via positions
    sc, sr = bcu_grid.to_cell(via_start_pos[0], via_start_pos[1])
    gc, gr = bcu_grid.to_cell(via_goal_pos[0], via_goal_pos[1])

    orig_start = not bcu_grid.is_free(sc, sr)
    orig_goal = not bcu_grid.is_free(gc, gr)
    bcu_grid.unmark(sc, sr)
    bcu_grid.unmark(gc, gr)

    path = _astar(bcu_grid, sc, sr, gc, gr)
    if path is None:
        _log.debug(
            "B.Cu A* FAIL via (%.1f,%.1f)->(%.1f,%.1f) net=%s",
            via_start_pos[0], via_start_pos[1],
            via_goal_pos[0], via_goal_pos[1], net_name,
        )
        if orig_start:
            bcu_grid.mark(sc, sr)
        if orig_goal:
            bcu_grid.mark(gc, gr)
        return None

    # Build tracks, vias, and stubs
    tracks = _build_bcu_tracks(path, bcu_grid, width_mm, net_number)

    via_start = _build_endpoint_via(via_start_pos, start_is_tht, via_drill, via_size, net_number)
    via_goal = _build_endpoint_via(via_goal_pos, goal_is_tht, via_drill, via_size, net_number)

    stub_tracks = _build_stub_track(
        start_x, start_y, via_start_pos[0], via_start_pos[1],
        _start_needs_astar_stub, fcu_grid, net_number, width_mm, clearance_mm,
    )
    stub_tracks.extend(_build_stub_track(
        goal_x, goal_y, via_goal_pos[0], via_goal_pos[1],
        _goal_needs_astar_stub, fcu_grid, net_number, width_mm, clearance_mm,
    ))

    return _finalize_bcu_route(
        path, tracks, stub_tracks, via_start, via_goal,
        bcu_grid, fcu_grid, width_mm, clearance_mm, via_radius,
    )


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


def _is_area_free(
    grid: _Grid,
    col: int,
    row: int,
    radius: int,
    secondary_grid: _Grid | None = None,
) -> bool:
    """Check whether all cells within *radius* of (col, row) are free.

    Optionally checks a secondary grid (e.g. B.Cu) as well.
    Returns False immediately on the first occupied cell.
    """
    c_lo = max(0, col - radius)
    c_hi = min(grid.cols, col + radius + 1)
    r_lo = max(0, row - radius)
    r_hi = min(grid.rows, row + radius + 1)
    # Reject if clamped bounds are smaller than requested (near edge)
    if c_lo > col - radius or c_hi <= col + radius:
        return False
    if r_lo > row - radius or r_hi <= row + radius:
        return False
    cells = grid._cells
    for cc in range(c_lo, c_hi):
        col_cells = cells[cc]
        for rr in range(r_lo, r_hi):
            if col_cells[rr]:
                return False
    if secondary_grid is not None:
        s_cells = secondary_grid._cells
        for cc in range(c_lo, c_hi):
            s_col = s_cells[cc]
            for rr in range(r_lo, r_hi):
                if s_col[rr]:
                    return False
    return True


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
    diag: bool = False,
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

    # Orthogonal + optional diagonal neighbors
    neighbors: tuple[tuple[int, int], ...] = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
    )
    if diag:
        neighbors = (*neighbors, (-1, -1), (-1, 1), (1, -1), (1, 1))

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

        for dc, dr in neighbors:
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

            # Diagonal moves cost sqrt(2), orthogonal cost 1
            is_diag = (dc != 0 and dr != 0)
            base = 1.4142 if is_diag else 1.0

            # Base step cost with optional congestion weighting
            step_cost = (
                grid.get_cost(nc, nr) * base if use_congestion else base
            )

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
# route_net context and extracted helpers
# ---------------------------------------------------------------------------


@dataclass
class _RouteContext:
    """Mutable state threaded through route_net helper functions."""

    request: RouteRequest
    grid: _Grid
    bcu_grid: _Grid | None
    fp_by_ref: dict[str, Footprint]
    pad_infos: list[_PadInfo]
    net_pad_set: frozenset[tuple[str, str]]
    original_net_pad_set: frozenset[tuple[str, str]]
    pad_cl: float
    bcu_pad_cl: float
    excl_cells: int
    all_tracks: list[Track]
    all_vias: list[Via]
    tht_refs_in_net: set[str]
    ic_refs_in_net: set[str]
    ic_pad_infos: list[_PadInfo]
    ic_pad_refs: list[tuple[str, str]]
    pad_cache: dict[str, list[tuple[float, float, float, float, str]]]
    net_clearances: dict[str, float] | None
    net_widths: dict[str, float] | None
    placed_via_positions: list[tuple[float, float]] | None
    footprints: list[Footprint]


def _detect_tht_refs(
    request: RouteRequest,
    fp_by_ref: dict[str, Footprint],
) -> set[str]:
    """Identify THT component refs in the net for deferred sibling unmark."""
    tht_refs: set[str] = set()
    for ref, _ in request.pad_refs:
        fp = fp_by_ref.get(ref)
        if fp is None:
            continue
        if any(p.size_x > 1.5 or p.size_y > 1.5 for p in fp.pads):
            tht_refs.add(ref)
    return tht_refs


def _detect_dense_ic_refs(
    request: RouteRequest,
    fp_by_ref: dict[str, Footprint],
) -> set[str]:
    """Identify dense IC refs (>=6 pads, min spacing < 1.0mm)."""
    ic_refs: set[str] = set()
    for ref, _ in request.pad_refs:
        fp = fp_by_ref.get(ref)
        if fp is None or len(fp.pads) < 6:
            continue
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
            ic_refs.add(ref)
    return ic_refs


def _partition_ic_pads(
    pad_infos: list[_PadInfo],
    pad_refs: tuple[tuple[str, str], ...],
    ic_refs_in_net: set[str],
) -> tuple[list[_PadInfo], list[_PadInfo], list[tuple[str, str]]]:
    """Split pad_infos into non-IC and IC groups.

    Returns:
        (non_ic_infos, ic_pad_infos, ic_pad_refs)
    """
    non_ic_infos = [
        pi for pi, (ref, _) in zip(pad_infos, pad_refs, strict=True)
        if ref not in ic_refs_in_net
    ]
    ic_pad_infos = [
        pi for pi, (ref, _) in zip(pad_infos, pad_refs, strict=True)
        if ref in ic_refs_in_net
    ]
    ic_pad_refs_list = [
        (ref, pn) for (ref, pn), pi in zip(pad_refs, pad_infos, strict=True)
        if ref in ic_refs_in_net
    ]
    return non_ic_infos, ic_pad_infos, ic_pad_refs_list


def _validate_path_segments(
    path: list[tuple[int, int]],
    grid: _Grid,
    request: RouteRequest,
    footprints: list[Footprint],
    net_pad_set: frozenset[tuple[str, str]],
    width: float | None = None,
) -> list[Track]:
    """Simplify path, build Track segments, validate against other-net pads.

    Returns:
        List of Track segments, or empty list if path crosses other-net pads.
    """
    trial = _simplify_path(path)
    segs = [
        Track(
            start=Point(*grid.to_mm(trial[k][0], trial[k][1])),
            end=Point(*grid.to_mm(trial[k + 1][0], trial[k + 1][1])),
            width=width or request.width_mm, layer="F.Cu",
            net_number=request.net_number, uuid="",
        )
        for k in range(len(trial) - 1)
    ]
    if _track_crosses_other_pads(
        segs, request.net_number, footprints,
        net_pad_set=net_pad_set,
    ):
        return []
    return segs


def _try_tht_sibling_unmark(
    ctx: _RouteContext,
    start_col: int, start_row: int,
    goal_col: int, goal_row: int,
) -> list[tuple[int, int]] | None:
    """Retry A* after shrinking clearance zones around sibling THT pads."""
    if not ctx.tht_refs_in_net:
        return None
    _log.debug(
        "MST %s: THT sibling unmark for refs %s",
        ctx.request.net_name, ctx.tht_refs_in_net,
    )
    extended_pads: set[tuple[str, str]] = set(ctx.net_pad_set)
    for ref in ctx.tht_refs_in_net:
        fp = ctx.fp_by_ref[ref]
        for pad in fp.pads:
            extended_pads.add((ref, pad.number))
            px, py = _pad_abs_pos(fp, pad)
            phw, phh = _pad_rotated_half_size(fp, pad)
            _unmark_pad_area(ctx.grid, px, py, phw, phh, ctx.pad_cl)
            _mark_pad_area(ctx.grid, px, py, phw, phh, 0.0)
    ctx.tht_refs_in_net.clear()
    ctx.net_pad_set = frozenset(extended_pads)
    _remark_other_pads(
        ctx.grid, ctx.footprints, ctx.net_pad_set,
        ctx.net_clearances, ctx.net_widths,
        _pad_cache=ctx.pad_cache,
    )
    path = _astar(ctx.grid, start_col, start_row, goal_col, goal_row)
    _log.debug(
        "MST %s: THT retry F.Cu=%s",
        ctx.request.net_name, "OK" if path else "FAIL",
    )
    if path is not None:
        segs = _validate_path_segments(
            path, ctx.grid, ctx.request, ctx.footprints,
            ctx.original_net_pad_set,
        )
        if not segs:
            return None
    return path


def _try_bcu_fallback(
    ctx: _RouteContext,
    p1: _PadInfo,
    p2: _PadInfo,
) -> tuple[list[Track], list[Via]] | None:
    """Attempt B.Cu routing with via fallback for an MST pair.

    Returns:
        (tracks, vias) if successful, None otherwise.
    """
    bcu_grid = ctx.bcu_grid
    if bcu_grid is None:
        return None
    if len(ctx.all_vias) + 2 > ctx.request.max_vias:
        _log.debug(
            "MST %s: B.Cu skip (vias=%d/%d)",
            ctx.request.net_name, len(ctx.all_vias), ctx.request.max_vias,
        )
        return None

    _bcu_saved = bcu_grid.save_state()

    # Unmark same-net THT pads on B.Cu
    for pi in ctx.pad_infos:
        if pi.pad_type == "thru_hole":
            _unmark_pad_area(
                bcu_grid, pi.x, pi.y,
                pi.half_w + ctx.bcu_pad_cl, pi.half_h + ctx.bcu_pad_cl,
                0.0,
            )
    # Open corridors through dense connector pad forests
    _endpoint_conn_refs: set[str] = set()
    for pi in (p1, p2):
        if pi.pad_type == "thru_hole":
            for ref, _pn in ctx.request.pad_refs:
                fp = ctx.fp_by_ref.get(ref)
                if fp is not None and sum(
                    1 for p in fp.pads if p.pad_type == "thru_hole"
                ) > 6:
                    _endpoint_conn_refs.add(ref)
    _conn_inner = max(ctx.bcu_pad_cl * 0.25, 0.1)
    for _cref in _endpoint_conn_refs:
        _cfp = ctx.fp_by_ref[_cref]
        for _cp in _cfp.pads:
            if _cp.pad_type != "thru_hole":
                continue
            _cpx, _cpy = _pad_abs_pos(_cfp, _cp)
            _cphw, _cphh = _pad_rotated_half_size(_cfp, _cp)
            _unmark_pad_area(
                bcu_grid, _cpx, _cpy,
                _cphw + ctx.bcu_pad_cl, _cphh + ctx.bcu_pad_cl,
                _conn_inner,
            )

    bcu_result = _route_on_bcu(
        p1.x, p1.y, p2.x, p2.y,
        bcu_grid, ctx.request.net_number, ctx.request.net_name,
        ctx.request.width_mm, ctx.request.clearance_mm,
        fcu_grid=ctx.grid,
        start_is_tht=p1.pad_type == "thru_hole",
        goal_is_tht=p2.pad_type == "thru_hole",
    )
    bcu_grid.restore_state(_bcu_saved)

    if bcu_result is None:
        _log.debug("MST %s: B.Cu FAIL (None)", ctx.request.net_name)
        return None

    bcu_tracks, bcu_vias = bcu_result
    # Re-mark the successful route on the restored grid
    _via_excl = math.ceil(
        (VIA_DIAMETER_SIGNAL_MM / 2 + ctx.request.clearance_mm)
        / bcu_grid.grid_step_mm,
    )
    for _bt in bcu_tracks:
        if _bt.layer == "B.Cu":
            _mark_line_on_grid(
                bcu_grid, _bt.start.x, _bt.start.y,
                _bt.end.x, _bt.end.y,
                ctx.request.clearance_mm + ctx.request.width_mm,
            )
    for _bv in bcu_vias:
        _vc, _vr = bcu_grid.to_cell(_bv.position.x, _bv.position.y)
        bcu_grid.mark_area(_vc, _vr, _via_excl)

    crosses = _track_crosses_other_pads(
        bcu_tracks, ctx.request.net_number, ctx.footprints,
        net_pad_set=ctx.original_net_pad_set,
    )
    _log.debug(
        "MST %s: B.Cu result=%d tracks, crosses=%s",
        ctx.request.net_name, len(bcu_tracks), crosses,
    )
    if crosses:
        return None
    return list(bcu_tracks), list(bcu_vias)


def _compute_ic_stub_width(
    ic_refs_in_net: set[str],
    fp_by_ref: dict[str, Footprint],
    default_width: float,
) -> float:
    """Compute pitch-limited track width for IC stubs."""
    ic_stub_width = default_width
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
    return ic_stub_width


def _try_ic_fcu_route(
    ctx: _RouteContext,
    ic_pi: _PadInfo,
    ic_ref: str,
    ic_pn: str,
    best_pi: _PadInfo,
    ic_pad_cl: float,
    ic_stub_width: float,
) -> bool:
    """Try F.Cu A* routing from a non-IC pad to an IC pad.

    Returns:
        True if route was successful and tracks/vias added to ctx.
    """
    ic_fp = ctx.fp_by_ref[ic_ref]
    # Temporarily unmark ALL pads on this IC
    for _ip in ic_fp.pads:
        _ipx, _ipy = _pad_abs_pos(ic_fp, _ip)
        _iphw, _iphh = _pad_rotated_half_size(ic_fp, _ip)
        _unmark_pad_area(ctx.grid, _ipx, _ipy, _iphw, _iphh, ic_pad_cl)
    _unmark_pad_area(
        ctx.grid, best_pi.x, best_pi.y,
        best_pi.half_w, best_pi.half_h, ic_pad_cl,
    )
    _remark_other_pads(
        ctx.grid, ctx.footprints, ctx.net_pad_set,
        ctx.net_clearances, ctx.net_widths,
        _pad_cache=ctx.pad_cache,
    )
    # Re-unmark IC pads (remark_other_pads re-marks them)
    for _ip in ic_fp.pads:
        _ipx, _ipy = _pad_abs_pos(ic_fp, _ip)
        _iphw, _iphh = _pad_rotated_half_size(ic_fp, _ip)
        _unmark_pad_area(ctx.grid, _ipx, _ipy, _iphw, _iphh, ic_pad_cl)

    start_col, start_row = ctx.grid.to_cell(best_pi.x, best_pi.y)
    goal_col, goal_row = ctx.grid.to_cell(ic_pi.x, ic_pi.y)
    path = _astar(ctx.grid, start_col, start_row, goal_col, goal_row)
    _log.debug(
        "IC final-leg %s pad %s: F.Cu A* %s, dist=%.1fmm",
        ic_ref, ic_pn,
        "OK" if path is not None else "FAIL",
        ((best_pi.x - ic_pi.x)**2 + (best_pi.y - ic_pi.y)**2)**0.5,
    )
    if path is None:
        return False

    fcu_segs: list[Track] = []
    sim_path = _simplify_path(path)
    for j in range(len(sim_path) - 1):
        x1, y1 = ctx.grid.to_mm(sim_path[j][0], sim_path[j][1])
        x2, y2 = ctx.grid.to_mm(sim_path[j + 1][0], sim_path[j + 1][1])
        fcu_segs.append(
            Track(
                start=Point(x1, y1), end=Point(x2, y2),
                width=ic_stub_width, layer=ctx.request.layer,
                net_number=ctx.request.net_number, uuid="",
            )
        )
    if _track_crosses_other_pads(
        fcu_segs, ctx.request.net_number, ctx.footprints,
        net_pad_set=ctx.original_net_pad_set,
    ):
        return False

    ctx.all_tracks.extend(fcu_segs)
    for cell_col, cell_row in path:
        ctx.grid.mark_area(cell_col, cell_row, ctx.excl_cells)
    return True


def _try_ic_bcu_route(
    ctx: _RouteContext,
    ic_pi: _PadInfo,
    ic_ref: str,
    ic_pn: str,
    best_pi: _PadInfo,
    ic_pad_cl: float,
    ic_stub_width: float,
) -> bool:
    """Try B.Cu fallback routing for IC final-leg.

    Returns:
        True if route was successful and tracks/vias added to ctx.
    """
    bcu_grid = ctx.bcu_grid
    if bcu_grid is None:
        return False

    _ic_bcu_unmarked: list[tuple[float, float, float, float]] = []
    for pi in ctx.pad_infos:
        if pi.pad_type == "thru_hole":
            _unmark_pad_area(
                bcu_grid, pi.x, pi.y,
                pi.half_w + ctx.bcu_pad_cl, pi.half_h + ctx.bcu_pad_cl,
                ctx.bcu_pad_cl * 0.5,
            )
            _ic_bcu_unmarked.append((pi.x, pi.y, pi.half_w, pi.half_h))
    if best_pi.pad_type == "thru_hole":
        _unmark_pad_area(
            bcu_grid, best_pi.x, best_pi.y,
            best_pi.half_w + ctx.bcu_pad_cl,
            best_pi.half_h + ctx.bcu_pad_cl,
            ic_pad_cl,
        )
    bcu_result = _route_on_bcu(
        best_pi.x, best_pi.y, ic_pi.x, ic_pi.y,
        bcu_grid, ctx.request.net_number, ctx.request.net_name,
        ic_stub_width, ctx.request.clearance_mm,
        fcu_grid=ctx.grid,
        start_is_tht=best_pi.pad_type == "thru_hole",
    )
    # Re-mark unmarked THT pads on B.Cu
    for _ux, _uy, _uhw, _uhh in _ic_bcu_unmarked:
        _mark_pad_area(bcu_grid, _ux, _uy, _uhw, _uhh, ctx.bcu_pad_cl)
    if best_pi.pad_type == "thru_hole":
        _mark_pad_area(
            bcu_grid, best_pi.x, best_pi.y,
            best_pi.half_w, best_pi.half_h, ctx.bcu_pad_cl,
        )
    if bcu_result is None:
        return False

    bcu_tracks, bcu_vias = bcu_result
    crosses = _track_crosses_other_pads(
        bcu_tracks, ctx.request.net_number, ctx.footprints,
        net_pad_set=ctx.original_net_pad_set,
    )
    _log.debug(
        "IC final-leg %s pad %s: B.Cu route OK (crosses=%s, %d tracks)",
        ic_ref, ic_pn, crosses, len(bcu_tracks),
    )
    if crosses:
        return False
    ctx.all_tracks.extend(bcu_tracks)
    ctx.all_vias.extend(bcu_vias)
    return True


def _try_ic_fanout(
    ctx: _RouteContext,
    ic_pi: _PadInfo,
    ic_ref: str,
    ic_pn: str,
    best_pi: _PadInfo,
    ic_pad_cl: float,
    ic_stub_width: float,
    ic_via_positions: list[tuple[float, float]],
    vip_min_dist: float,
    px: float,
    py: float,
) -> bool:
    """Attempt IC fanout: place via outward from IC body, route to target.

    Returns:
        True if route was successful and tracks/vias added to ctx.
    """
    bcu_grid = ctx.bcu_grid
    if bcu_grid is None:
        return False

    ic_fp = ctx.fp_by_ref[ic_ref]
    ic_cx = ic_fp.position.x
    ic_cy = ic_fp.position.y
    dx_from_center = px - ic_cx
    dy_from_center = py - ic_cy

    via_radius = VIA_DIAMETER_SIGNAL_MM / 2.0
    r_cells = max(1, round(
        (via_radius + ctx.pad_cl) / ctx.grid.grid_step_mm,
    ))

    # Build direction priority: perpendicular away from IC first
    is_horizontal = abs(dx_from_center) > abs(dy_from_center)
    if is_horizontal:
        primary_dir = (-1.0 if dx_from_center < 0 else 1.0, 0.0)
    else:
        primary_dir = (0.0, -1.0 if dy_from_center < 0 else 1.0)
    all_dirs: list[tuple[float, float]] = [primary_dir]
    for dx, dy in [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]:
        d = (float(dx), float(dy))
        if d != primary_dir:
            all_dirs.append(d)

    # Search for a clear via position
    fan_via_pos, fan_stubs, fan_via_col, fan_via_row = _search_fanout_via(
        ctx, ic_fp, ic_ref, ic_pn, ic_pad_cl, ic_stub_width,
        ic_via_positions, vip_min_dist, px, py,
        r_cells, all_dirs,
    )

    _log.debug(
        "IC fanout %s pad %s: via=%s stub=%s",
        ic_ref, ic_pn,
        f"({fan_via_pos[0]:.1f},{fan_via_pos[1]:.1f})"
        if fan_via_pos else "NONE",
        "OK" if fan_stubs else "FAIL",
    )
    if fan_via_pos is None or not fan_stubs:
        return False

    # Route from fanout via to the target pad
    fan_routed, fan_segs, fan_via_extra, fcu_fan_path = _route_fanout_to_target(
        ctx, best_pi, fan_via_pos, fan_via_col, fan_via_row,
        ic_ref, ic_pn, ic_refs_in_net=ctx.ic_refs_in_net,
        ic_stub_width=ic_stub_width,
    )

    if fan_routed:
        fan_via = Via(
            position=Point(fan_via_pos[0], fan_via_pos[1]),
            drill=VIA_DRILL_SIGNAL_MM,
            size=VIA_DIAMETER_SIGNAL_MM,
            layers=("F.Cu", "B.Cu"),
            net_number=ctx.request.net_number,
            uuid="",
        )
        ctx.all_tracks.extend(fan_stubs)
        ctx.all_tracks.extend(fan_segs)
        ctx.all_vias.append(fan_via)
        ctx.all_vias.extend(fan_via_extra)
        if fcu_fan_path is not None and not fan_via_extra:
            for cc, cr in fcu_fan_path:
                ctx.grid.mark_area(cc, cr, ctx.excl_cells)
        _mark_pad_area(
            ctx.grid,
            fan_via_pos[0], fan_via_pos[1],
            VIA_DIAMETER_SIGNAL_MM / 2.0,
            VIA_DIAMETER_SIGNAL_MM / 2.0,
            ctx.pad_cl,
        )
        if bcu_grid is not None:
            _mark_pad_area(
                bcu_grid,
                fan_via_pos[0], fan_via_pos[1],
                VIA_DIAMETER_SIGNAL_MM / 2.0,
                VIA_DIAMETER_SIGNAL_MM / 2.0,
                ctx.pad_cl,
            )
        for _fs in fan_stubs:
            _mark_line_on_grid(
                ctx.grid,
                _fs.start.x, _fs.start.y,
                _fs.end.x, _fs.end.y,
                ctx.excl_cells,
            )
        ic_via_positions.append(fan_via_pos)
    return fan_routed


def _search_fanout_via(
    ctx: _RouteContext,
    ic_fp: Footprint,
    ic_ref: str,
    ic_pn: str,
    ic_pad_cl: float,
    ic_stub_width: float,
    ic_via_positions: list[tuple[float, float]],
    vip_min_dist: float,
    px: float,
    py: float,
    r_cells: int,
    all_dirs: list[tuple[float, float]],
) -> tuple[
    tuple[float, float] | None,
    list[Track],
    int,
    int,
]:
    """Search for a clear fanout via position and build stub tracks.

    Returns:
        (fan_via_pos, fan_stubs, fan_via_col, fan_via_row)
    """
    fan_via_pos: tuple[float, float] | None = None
    fan_stubs: list[Track] = []
    fan_via_col = 0
    fan_via_row = 0

    for fan_dx, fan_dy in all_dirs:
        if fan_via_pos is not None:
            break
        for step in range(4, 120):
            _cx = px + fan_dx * step * ctx.grid.grid_step_mm
            _cy = py + fan_dy * step * ctx.grid.grid_step_mm
            col, row = ctx.grid.to_cell(_cx, _cy)
            clear = _is_area_free(ctx.grid, col, row, r_cells, ctx.bcu_grid)
            if not clear:
                continue
            _cand_pos = ctx.grid.to_mm(col, row)
            # Skip if too close to an existing via
            _fan_too_close = any(
                ((_cand_pos[0] - vx) ** 2 + (_cand_pos[1] - vy) ** 2) ** 0.5
                < vip_min_dist
                for vx, vy in ic_via_positions
            )
            if not _fan_too_close and ctx.placed_via_positions:
                _fan_too_close = any(
                    ((_cand_pos[0] - vx) ** 2 + (_cand_pos[1] - vy) ** 2) ** 0.5
                    < vip_min_dist
                    for vx, vy in ctx.placed_via_positions
                )
            if _fan_too_close:
                continue
            # Temporarily unmark ALL IC pads, then try A* stub
            _ic_um: list[tuple[float, float, float, float]] = []
            for _ip in ic_fp.pads:
                _ipx, _ipy = _pad_abs_pos(ic_fp, _ip)
                _iphw, _iphh = _pad_rotated_half_size(ic_fp, _ip)
                _unmark_pad_area(
                    ctx.grid, _ipx, _ipy, _iphw, _iphh, ic_pad_cl,
                )
                _ic_um.append((_ipx, _ipy, _iphw, _iphh))
            _fvc, _fvr = ctx.grid.to_cell(_cand_pos[0], _cand_pos[1])
            ctx.grid.unmark(_fvc, _fvr)
            _icc, _icr = ctx.grid.to_cell(px, py)
            _sp = _astar(ctx.grid, _icc, _icr, _fvc, _fvr, diag=True)
            # Re-mark IC pads
            for _rpx, _rpy, _rhw, _rhh in _ic_um:
                _mark_pad_area(ctx.grid, _rpx, _rpy, _rhw, _rhh, ic_pad_cl)
            if _sp is None:
                continue
            # Build stub tracks and validate
            _stubs: list[Track] = []
            _sim = _simplify_path(_sp)
            for _si in range(len(_sim) - 1):
                _sx, _sy = ctx.grid.to_mm(_sim[_si][0], _sim[_si][1])
                _ex, _ey = ctx.grid.to_mm(_sim[_si + 1][0], _sim[_si + 1][1])
                _stubs.append(Track(
                    start=Point(_sx, _sy), end=Point(_ex, _ey),
                    width=ic_stub_width, layer="F.Cu",
                    net_number=ctx.request.net_number, uuid="",
                ))
            if not _track_crosses_other_pads(
                _stubs, ctx.request.net_number, ctx.footprints,
                net_pad_set=ctx.original_net_pad_set,
                allow_same_ref=ic_ref,
            ):
                fan_via_pos = _cand_pos
                fan_stubs = _stubs
                fan_via_col = _fvc
                fan_via_row = _fvr
                break

    return fan_via_pos, fan_stubs, fan_via_col, fan_via_row


def _route_fanout_to_target(
    ctx: _RouteContext,
    best_pi: _PadInfo,
    fan_via_pos: tuple[float, float],
    fan_via_col: int,
    fan_via_row: int,
    ic_ref: str,
    ic_pn: str,
    ic_refs_in_net: set[str],
    ic_stub_width: float,
) -> tuple[bool, list[Track], list[Via], list[tuple[int, int]] | None]:
    """Route from fanout via to the target pad (F.Cu or B.Cu).

    Returns:
        (routed, fan_segs, fan_via_extra, fcu_fan_path)
    """
    tgt_col, tgt_row = ctx.grid.to_cell(best_pi.x, best_pi.y)
    _unmark_pad_area(
        ctx.grid, best_pi.x, best_pi.y,
        best_pi.half_w, best_pi.half_h, ctx.pad_cl,
    )
    _remark_other_pads(
        ctx.grid, ctx.footprints, ctx.net_pad_set,
        ctx.net_clearances, ctx.net_widths,
        _pad_cache=ctx.pad_cache,
    )
    # Unmark THT header pads for A* corridor
    _tht_fp_unmarked: list[tuple[float, float, float, float]] = []
    _tgt_ref = _find_target_ref(ctx, best_pi, ic_refs_in_net)
    if _tgt_ref:
        _t_fp = ctx.fp_by_ref[_tgt_ref]
        has_tht = any(p.pad_type == "thru_hole" for p in _t_fp.pads)
        if has_tht and len(_t_fp.pads) > 4:
            _tphw = best_pi.half_w
            _tphh = best_pi.half_h
            _unmark_pad_area(
                ctx.grid, best_pi.x, best_pi.y,
                _tphw + ctx.pad_cl, _tphh + ctx.pad_cl, ctx.pad_cl,
            )
            _tht_fp_unmarked.append((
                best_pi.x, best_pi.y,
                _tphw + ctx.pad_cl, _tphh + ctx.pad_cl,
            ))

    fcu_fan_path = _astar(
        ctx.grid, fan_via_col, fan_via_row, tgt_col, tgt_row, diag=True,
    )
    # Re-mark temporarily unmarked THT pads
    for _rpx, _rpy, _rhw, _rhh in _tht_fp_unmarked:
        _mark_pad_area(ctx.grid, _rpx, _rpy, _rhw, _rhh, ctx.pad_cl)

    _fan_routed = False
    fan_segs: list[Track] = []
    fan_via_extra: list[Via] = []

    if fcu_fan_path is not None:
        sim_fan = _simplify_path(fcu_fan_path)
        for j in range(len(sim_fan) - 1):
            sx, sy = ctx.grid.to_mm(sim_fan[j][0], sim_fan[j][1])
            ex, ey = ctx.grid.to_mm(sim_fan[j + 1][0], sim_fan[j + 1][1])
            fan_segs.append(Track(
                start=Point(sx, sy), end=Point(ex, ey),
                width=ic_stub_width, layer="F.Cu",
                net_number=ctx.request.net_number, uuid="",
            ))
        crosses = _track_crosses_other_pads(
            fan_segs, ctx.request.net_number, ctx.footprints,
            net_pad_set=ctx.original_net_pad_set,
            allow_same_ref=_tgt_ref or None,
        )
        if not crosses:
            _fan_routed = True

    _log.debug(
        "IC fanout %s pad %s: F.Cu fan A*=%s, crosses=%s",
        ic_ref, ic_pn,
        "OK" if fcu_fan_path is not None else "FAIL",
        "yes" if (fcu_fan_path is not None and not _fan_routed) else "no",
    )

    # B.Cu fallback for fanout-via -> target pad
    if not _fan_routed and ctx.bcu_grid is not None:
        bcu_fan_result = _try_fanout_bcu(
            ctx, best_pi, fan_via_pos, ic_stub_width,
        )
        if bcu_fan_result is not None:
            fan_segs, fan_via_extra = bcu_fan_result
            _fan_routed = True
        else:
            _log.debug(
                "IC fanout %s pad %s: B.Cu fan FAIL",
                ic_ref, ic_pn,
            )

    return _fan_routed, fan_segs, fan_via_extra, fcu_fan_path


def _find_target_ref(
    ctx: _RouteContext,
    best_pi: _PadInfo,
    ic_refs_in_net: set[str],
) -> str:
    """Find the ref designator of the non-IC footprint at best_pi position."""
    for _t_ref, _t_pn in ctx.request.pad_refs:
        if _t_ref in ic_refs_in_net:
            continue
        _t_fp = ctx.fp_by_ref.get(_t_ref)
        if _t_fp is None:
            continue
        for _t_pad in _t_fp.pads:
            _tpx, _tpy = _pad_abs_pos(_t_fp, _t_pad)
            if (abs(_tpx - best_pi.x) < 0.01
                    and abs(_tpy - best_pi.y) < 0.01):
                return _t_ref
    return ""


def _try_fanout_bcu(
    ctx: _RouteContext,
    best_pi: _PadInfo,
    fan_via_pos: tuple[float, float],
    ic_stub_width: float,
) -> tuple[list[Track], list[Via]] | None:
    """Try B.Cu routing from fanout via to target pad."""
    bcu_grid = ctx.bcu_grid
    if bcu_grid is None:
        return None
    _fan_bcu_um: list[tuple[float, float, float, float]] = []
    for pi in ctx.pad_infos:
        if pi.pad_type == "thru_hole":
            _unmark_pad_area(
                bcu_grid, pi.x, pi.y,
                pi.half_w + ctx.bcu_pad_cl,
                pi.half_h + ctx.bcu_pad_cl,
                ctx.bcu_pad_cl * 0.5,
            )
            _fan_bcu_um.append((pi.x, pi.y, pi.half_w, pi.half_h))
    bcu_fan = _route_on_bcu(
        fan_via_pos[0], fan_via_pos[1],
        best_pi.x, best_pi.y,
        bcu_grid,
        ctx.request.net_number,
        ctx.request.net_name,
        ic_stub_width,
        ctx.request.clearance_mm,
        fcu_grid=ctx.grid,
        start_via_in_pad=True,
        goal_is_tht=best_pi.pad_type == "thru_hole",
    )
    for _ux, _uy, _uhw, _uhh in _fan_bcu_um:
        _mark_pad_area(bcu_grid, _ux, _uy, _uhw, _uhh, ctx.pad_cl)
    if bcu_fan is None:
        return None
    bcu_fan_tracks, bcu_fan_vias = bcu_fan
    if _track_crosses_other_pads(
        bcu_fan_tracks, ctx.request.net_number, ctx.footprints,
        net_pad_set=ctx.original_net_pad_set,
    ):
        return None
    return list(bcu_fan_tracks), list(bcu_fan_vias)


def _try_via_in_pad(
    ctx: _RouteContext,
    ic_pi: _PadInfo,
    ic_ref: str,
    ic_pn: str,
    best_pi: _PadInfo,
    ic_pad_cl: float,
    ic_stub_width: float,
    ic_via_positions: list[tuple[float, float]],
    vip_min_dist: float,
    px: float,
    py: float,
) -> bool:
    """Via-in-pad fallback: place via directly on the IC pad, route on B.Cu.

    Returns:
        True if route was successful and tracks/vias added to ctx.
    """
    bcu_grid = ctx.bcu_grid
    if bcu_grid is None:
        return False

    # Check if any existing via is too close
    _vip_too_close = any(
        ((vx - px) ** 2 + (vy - py) ** 2) ** 0.5 < vip_min_dist
        for vx, vy in ic_via_positions
    )
    if not _vip_too_close and ctx.placed_via_positions is not None:
        _vip_too_close = any(
            ((vx - px) ** 2 + (vy - py) ** 2) ** 0.5 < vip_min_dist
            for vx, vy in ctx.placed_via_positions
        )
    if _vip_too_close:
        return False

    _vip_bcu_um: list[tuple[float, float, float, float]] = []
    for pi in ctx.pad_infos:
        if pi.pad_type == "thru_hole":
            _unmark_pad_area(
                bcu_grid, pi.x, pi.y,
                pi.half_w + ctx.bcu_pad_cl,
                pi.half_h + ctx.bcu_pad_cl,
                ctx.bcu_pad_cl * 0.5,
            )
            _vip_bcu_um.append((pi.x, pi.y, pi.half_w, pi.half_h))
    _unmark_pad_area(
        bcu_grid, best_pi.x, best_pi.y,
        best_pi.half_w + ctx.bcu_pad_cl,
        best_pi.half_h + ctx.bcu_pad_cl,
        ic_pad_cl,
    )
    vip_result = _route_on_bcu(
        px, py, best_pi.x, best_pi.y,
        bcu_grid, ctx.request.net_number, ctx.request.net_name,
        ic_stub_width, ctx.request.clearance_mm,
        fcu_grid=None,
        goal_is_tht=best_pi.pad_type == "thru_hole",
    )
    for _ux, _uy, _uhw, _uhh in _vip_bcu_um:
        _mark_pad_area(bcu_grid, _ux, _uy, _uhw, _uhh, ctx.bcu_pad_cl)
    if vip_result is None and best_pi.pad_type == "thru_hole":
        _mark_pad_area(
            bcu_grid, best_pi.x, best_pi.y,
            best_pi.half_w, best_pi.half_h, ctx.bcu_pad_cl,
        )
    if vip_result is None:
        return False

    vip_tracks, vip_vias = vip_result
    if _track_crosses_other_pads(
        vip_tracks, ctx.request.net_number, ctx.footprints,
        net_pad_set=ctx.original_net_pad_set,
    ):
        if best_pi.pad_type == "thru_hole":
            _mark_pad_area(
                bcu_grid, best_pi.x, best_pi.y,
                best_pi.half_w, best_pi.half_h, ctx.pad_cl,
            )
        return False

    ctx.all_tracks.extend(vip_tracks)
    ctx.all_vias.extend(vip_vias)
    _mark_pad_area(
        bcu_grid, px, py,
        VIA_DIAMETER_SIGNAL_MM / 2.0,
        VIA_DIAMETER_SIGNAL_MM / 2.0,
        ctx.pad_cl,
    )
    ic_via_positions.append((px, py))
    _log.debug("IC final-leg %s pad %s: via-in-pad OK", ic_ref, ic_pn)
    return True


def _route_ic_final_legs(ctx: _RouteContext) -> None:
    """Route IC final-leg connections after MST loop."""
    if not ctx.ic_pad_infos:
        return

    from kicad_pipeline.constants import JLCPCB_MIN_CLEARANCE_MM
    ic_pad_cl = min(ctx.pad_cl, JLCPCB_MIN_CLEARANCE_MM)
    ic_stub_width = _compute_ic_stub_width(
        ctx.ic_refs_in_net, ctx.fp_by_ref, ctx.request.width_mm,
    )

    # Sort IC pads outermost-first
    _ic_cx = sum(p.x for p in ctx.ic_pad_infos) / len(ctx.ic_pad_infos)
    _ic_cy = sum(p.y for p in ctx.ic_pad_infos) / len(ctx.ic_pad_infos)
    ic_sorted = sorted(
        zip(ctx.ic_pad_infos, ctx.ic_pad_refs, strict=True),
        key=lambda pr: -((pr[0].x - _ic_cx) ** 2 + (pr[0].y - _ic_cy) ** 2),
    )

    _ic_via_positions: list[tuple[float, float]] = []
    _vip_min_dist = VIA_DIAMETER_SIGNAL_MM + ctx.request.clearance_mm

    for ic_pi, (ic_ref, ic_pn) in ic_sorted:
        # Find closest routed non-IC pad
        best_pi = ctx.pad_infos[0]
        best_dist = float("inf")
        for pi in ctx.pad_infos:
            d = abs(pi.x - ic_pi.x) + abs(pi.y - ic_pi.y)
            if d < best_dist:
                best_dist = d
                best_pi = pi

        ic_fp = ctx.fp_by_ref[ic_ref]
        ic_pad = next(p for p in ic_fp.pads if p.number == ic_pn)
        px, py = _pad_abs_pos(ic_fp, ic_pad)
        ic_hw, ic_hh = _pad_rotated_half_size(ic_fp, ic_pad)

        ic_routed = _try_ic_fcu_route(
            ctx, ic_pi, ic_ref, ic_pn, best_pi, ic_pad_cl, ic_stub_width,
        )

        if not ic_routed:
            ic_routed = _try_ic_bcu_route(
                ctx, ic_pi, ic_ref, ic_pn, best_pi, ic_pad_cl, ic_stub_width,
            )
            if ic_routed:
                for v in ctx.all_vias:
                    _ic_via_positions.append((v.position.x, v.position.y))

        if not ic_routed:
            _log.debug(
                "IC final-leg %s pad %s: B.Cu fallback %s",
                ic_ref, ic_pn,
                "skipped" if ctx.bcu_grid is None else "FAIL",
            )

        if not ic_routed:
            ic_routed = _try_ic_fanout(
                ctx, ic_pi, ic_ref, ic_pn, best_pi,
                ic_pad_cl, ic_stub_width,
                _ic_via_positions, _vip_min_dist,
                px, py,
            )

        if not ic_routed:
            ic_routed = _try_via_in_pad(
                ctx, ic_pi, ic_ref, ic_pn, best_pi,
                ic_pad_cl, ic_stub_width,
                _ic_via_positions, _vip_min_dist,
                px, py,
            )

        if not ic_routed:
            _log.debug("IC final-leg %s pad %s: UNROUTED", ic_ref, ic_pn)
            _mark_pad_area(ctx.grid, px, py, ic_hw, ic_hh, ctx.pad_cl)


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


def _find_mst_nearest_pair(
    pad_infos: list[_PadInfo],
    routed_set: set[int],
    unrouted: set[int],
) -> tuple[int, int]:
    """Find the closest routed-to-unrouted pad pair (Manhattan distance)."""
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
    return best_from, best_to


def _emit_fcu_tracks(
    path: list[tuple[int, int]],
    ctx: _RouteContext,
) -> None:
    """Convert an F.Cu A* path into track segments and mark the grid."""
    grid = ctx.grid
    request = ctx.request
    simplified = _simplify_path(path)
    for j in range(len(simplified) - 1):
        x1, y1 = grid.to_mm(simplified[j][0], simplified[j][1])
        x2, y2 = grid.to_mm(simplified[j + 1][0], simplified[j + 1][1])
        ctx.all_tracks.append(
            Track(
                start=Point(x1, y1), end=Point(x2, y2),
                width=request.width_mm, layer=request.layer,
                net_number=request.net_number, uuid="",
            )
        )

    for cell_col, cell_row in path:
        grid.mark_area(cell_col, cell_row, ctx.excl_cells)
        grid.add_congestion(cell_col, cell_row, radius=2)

    for pi in ctx.pad_infos:
        _unmark_pad_area(grid, pi.x, pi.y, pi.half_w, pi.half_h, ctx.pad_cl)
    _remark_other_pads(
        grid, ctx.footprints, ctx.net_pad_set,
        ctx.net_clearances, ctx.net_widths, _pad_cache=ctx.pad_cache,
    )


def _route_mst_pairs(
    ctx: _RouteContext,
    footprints: list[Footprint],
    net_clearances: dict[str, float] | None,
    net_widths: dict[str, float] | None,
) -> RouteResult | None:
    """Route pad pairs using MST ordering.

    Returns a failed :class:`RouteResult` if any pair cannot be routed,
    or ``None`` on success (results stored in *ctx*).
    """
    request = ctx.request
    grid = ctx.grid
    pad_infos = ctx.pad_infos
    ic_refs_in_net = ctx.ic_refs_in_net

    seed = _mst_seed(pad_infos)
    routed_set: set[int] = {seed}
    unrouted: set[int] = set(range(len(pad_infos))) - {seed}

    while unrouted:
        best_from, best_to = _find_mst_nearest_pair(pad_infos, routed_set, unrouted)
        p1 = pad_infos[best_from]
        p2 = pad_infos[best_to]
        start_col, start_row = grid.to_cell(p1.x, p1.y)
        goal_col, goal_row = grid.to_cell(p2.x, p2.y)

        # Try F.Cu A*
        path = _astar(grid, start_col, start_row, goal_col, goal_row)
        if path is not None:
            segs = _validate_path_segments(
                path, grid, request, footprints, ctx.original_net_pad_set,
            )
            if not segs:
                path = None
        _log.debug(
            "MST %s: (%s) (%.1f,%.1f)->(%s) (%.1f,%.1f) F.Cu=%s",
            request.net_name,
            request.pad_refs[best_from][0] + "." + request.pad_refs[best_from][1],
            p1.x, p1.y,
            request.pad_refs[best_to][0] + "." + request.pad_refs[best_to][1],
            p2.x, p2.y,
            "OK" if path is not None else "FAIL",
        )

        # THT sibling unmark retry
        if path is None:
            path = _try_tht_sibling_unmark(ctx, start_col, start_row, goal_col, goal_row)

        # B.Cu fallback
        if path is None:
            _log.debug(
                "MST %s: F.Cu FAIL, trying B.Cu (vias=%d/%d)",
                request.net_name, len(ctx.all_vias), request.max_vias,
            )
            bcu_result = _try_bcu_fallback(ctx, p1, p2)
            if bcu_result is not None:
                bcu_tracks, bcu_vias = bcu_result
                ctx.all_tracks.extend(bcu_tracks)
                ctx.all_vias.extend(bcu_vias)
                routed_set.add(best_to)
                unrouted.discard(best_to)
                continue

        if path is None:
            p2_ref = request.pad_refs[best_to][0] if best_to < len(request.pad_refs) else ""
            p1_ref = request.pad_refs[best_from][0] if best_from < len(request.pad_refs) else ""
            if p2_ref in ic_refs_in_net or p1_ref in ic_refs_in_net:
                unrouted.discard(best_to)
                continue
            _restore_pad_marks(grid, footprints, net_clearances, net_widths,
                               _pad_cache=ctx.pad_cache)
            return RouteResult(
                net_number=request.net_number, net_name=request.net_name,
                tracks=tuple(ctx.all_tracks), vias=tuple(ctx.all_vias),
                routed=False, reason=f"No path found for net {request.net_name}",
            )

        _emit_fcu_tracks(path, ctx)
        routed_set.add(best_to)
        unrouted.discard(best_to)

    return None


def _build_route_context(
    request: RouteRequest,
    grid: _Grid,
    bcu_grid: _Grid | None,
    fp_by_ref: dict[str, Footprint],
    pad_infos: list[_PadInfo],
    net_pad_set: frozenset[tuple[str, str]],
    pad_cl: float,
    tht_refs_in_net: frozenset[str],
    ic_refs_in_net: frozenset[str],
    ic_pad_infos: list[_PadInfo],
    ic_pad_refs: list[tuple[str, str]],
    pad_cache: list[object],
    net_clearances: dict[str, float] | None,
    net_widths: dict[str, float] | None,
    placed_via_positions: list[tuple[float, float]] | None,
    footprints: list[Footprint],
) -> _RouteContext:
    """Build the shared routing context for a single net."""
    excl_cells = max(1, math.ceil(
        (request.clearance_mm + request.width_mm) / grid.grid_step_mm,
    ) - 1)
    return _RouteContext(
        request=request, grid=grid, bcu_grid=bcu_grid,
        fp_by_ref=fp_by_ref, pad_infos=pad_infos,
        net_pad_set=net_pad_set, original_net_pad_set=net_pad_set,
        pad_cl=pad_cl, bcu_pad_cl=pad_cl,
        excl_cells=excl_cells,
        all_tracks=[], all_vias=[],
        tht_refs_in_net=tht_refs_in_net,
        ic_refs_in_net=ic_refs_in_net,
        ic_pad_infos=ic_pad_infos,
        ic_pad_refs=ic_pad_refs,
        pad_cache=pad_cache,
        net_clearances=net_clearances,
        net_widths=net_widths,
        placed_via_positions=placed_via_positions,
        footprints=footprints,
    )


def _handle_ic_pads(
    ic_refs_in_net: frozenset[str],
    pad_infos: list[_PadInfo],
    request: RouteRequest,
    fp_by_ref: dict[str, Footprint],
    net_pad_set: frozenset[tuple[str, str]],
    grid: _Grid,
    pad_cl: float,
    footprints: list[Footprint],
    net_clearances: dict[str, float] | None,
    net_widths: dict[str, float] | None,
    _pad_cache: list[object],
) -> (
    tuple[list[_PadInfo], list[_PadInfo], list[tuple[str, str]]]
    | RouteResult
):
    """Partition IC pads from non-IC pads.

    Returns ``(pad_infos, ic_pad_infos, ic_pad_refs)`` on success, or a
    failed :class:`RouteResult` if all pads belong to dense ICs.
    """
    _ic_pad_infos: list[_PadInfo] = []
    _ic_pad_refs: list[tuple[str, str]] = []
    if not ic_refs_in_net:
        return pad_infos, _ic_pad_infos, _ic_pad_refs

    non_ic_infos, _ic_pad_infos, _ic_pad_refs = _partition_ic_pads(
        pad_infos, request.pad_refs, ic_refs_in_net,
    )
    if len(non_ic_infos) >= 2:
        return non_ic_infos, _ic_pad_infos, _ic_pad_refs
    if len(non_ic_infos) == 1:
        for ic_ref in ic_refs_in_net:
            fp = fp_by_ref[ic_ref]
            for pad in fp.pads:
                if (ic_ref, pad.number) not in net_pad_set:
                    continue
                px, py = _pad_abs_pos(fp, pad)
                phw, phh = _pad_rotated_half_size(fp, pad)
                _unmark_pad_area(grid, px, py, phw, phh, pad_cl)
        return non_ic_infos, _ic_pad_infos, _ic_pad_refs
    # All pads on dense ICs
    _restore_pad_marks(grid, footprints, net_clearances, net_widths,
                       _pad_cache=_pad_cache)
    return RouteResult(
        net_number=request.net_number, net_name=request.net_name,
        tracks=(), vias=(), routed=False,
        reason="all pads on dense ICs - needs via routing",
    )


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
    placed_via_positions: list[tuple[float, float]] | None = None,
) -> RouteResult:
    """Route a single net using A* on a 2-D occupancy grid.

    Uses a shared *grid* when provided; otherwise creates a fresh one.
    Returns a :class:`RouteResult` with tracks and vias.
    """
    if grid is None:
        grid = _Grid.create(board_width_mm, board_height_mm, grid_step_mm)
        _prepare_grid(
            grid, list(footprints), keepouts=keepouts,
            net_clearances=net_clearances, net_widths=net_widths,
        )

    fp_by_ref: dict[str, Footprint] = {fp.ref: fp for fp in footprints}
    _pad_cache = _build_pad_cache(footprints)

    resolved = _resolve_pad_positions(request, fp_by_ref)
    if isinstance(resolved, str):
        return RouteResult(
            net_number=request.net_number, net_name=request.net_name,
            tracks=(), vias=(), routed=False, reason=resolved,
        )
    pad_infos = resolved

    if len(pad_infos) < 2:
        return RouteResult(
            net_number=request.net_number, net_name=request.net_name,
            tracks=(), vias=(), routed=False, reason="insufficient pad positions",
        )

    pad_cl = _global_pad_clearance(net_clearances, net_widths)
    net_pad_set = frozenset(request.pad_refs)
    for pi in pad_infos:
        _unmark_pad_area(grid, pi.x, pi.y, pi.half_w, pi.half_h, pad_cl)

    tht_refs_in_net = _detect_tht_refs(request, fp_by_ref)
    ic_refs_in_net = _detect_dense_ic_refs(request, fp_by_ref)

    ic_result = _handle_ic_pads(
        ic_refs_in_net, pad_infos, request, fp_by_ref, net_pad_set,
        grid, pad_cl, footprints, net_clearances, net_widths, _pad_cache,
    )
    if isinstance(ic_result, RouteResult):
        return ic_result
    pad_infos, _ic_pad_infos, _ic_pad_refs = ic_result

    _remark_other_pads(grid, footprints, net_pad_set, net_clearances, net_widths,
                       _pad_cache=_pad_cache)

    ctx = _build_route_context(
        request, grid, bcu_grid, fp_by_ref, pad_infos, net_pad_set, pad_cl,
        tht_refs_in_net, ic_refs_in_net, _ic_pad_infos, _ic_pad_refs,
        _pad_cache, net_clearances, net_widths, placed_via_positions, footprints,
    )

    # MST-style routing
    failure = _route_mst_pairs(ctx, footprints, net_clearances, net_widths)
    if failure is not None:
        return failure

    # IC final-leg routing
    _route_ic_final_legs(ctx)

    _restore_pad_marks(grid, footprints, net_clearances, net_widths,
                       _pad_cache=_pad_cache)

    return RouteResult(
        net_number=request.net_number, net_name=request.net_name,
        tracks=tuple(ctx.all_tracks), vias=tuple(ctx.all_vias),
        routed=True,
    )


def _retry_failed_nets(
    failed_entries: list[NetlistEntry],
    route_fn: object,
    record_vias_fn: object,
    results: list[RouteResult],
) -> list[NetlistEntry]:
    """Retry failed nets with standard retry and reversed pad ordering.

    Returns:
        List of entries that still failed after retries.
    """
    from kicad_pipeline.pcb.netlist import NetlistEntry as _NetlistEntry

    still_failed: list[_NetlistEntry] = []
    for entry in failed_entries:
        result = route_fn(entry)  # type: ignore[operator]
        if result.routed:
            results.append(result)
            record_vias_fn(result)  # type: ignore[operator]
            continue
        reversed_pads = entry.pad_refs[::-1]
        reversed_entry = _NetlistEntry(
            net=entry.net,
            pad_refs=reversed_pads,
        )
        result = route_fn(reversed_entry)  # type: ignore[operator]
        if result.routed:
            results.append(result)
            record_vias_fn(result)  # type: ignore[operator]
            continue
        still_failed.append(entry)
    return still_failed


def _retry_relaxed_clearance(
    still_failed: list[NetlistEntry],
    footprints: list[Footprint],
    board_width_mm: float,
    board_height_mm: float,
    grid_step_mm: float,
    grid: _Grid,
    bcu_grid: _Grid | None,
    net_widths: dict[str, float] | None,
    net_clearances: dict[str, float] | None,
    all_placed_vias: list[tuple[float, float]],
    results: list[RouteResult],
    record_vias_fn: object,
) -> None:
    """Last resort retry at JLCPCB minimum clearance."""
    if not still_failed:
        return
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
        n_pads = len(entry.pad_refs)
        via_budget = 2 if n_pads <= 2 else (4 if n_pads <= 4 else 6)
        request = RouteRequest(
            net_number=entry.net.number,
            net_name=net_name,
            pad_refs=entry.pad_refs,
            layer="F.Cu",
            width_mm=width,
            clearance_mm=JLCPCB_MIN_CLEARANCE_MM,
            max_vias=via_budget,
        )
        result = route_net(
            request, footprints, board_width_mm, board_height_mm,
            grid_step_mm, grid=grid, net_clearances=relaxed_clearances,
            net_widths=net_widths, bcu_grid=bcu_grid,
            placed_via_positions=all_placed_vias,
        )
        results.append(result)
        if result.routed:
            record_vias_fn(result)  # type: ignore[operator]


def _rip_up_and_retry(
    results: list[RouteResult],
    entry_by_name: dict[str, NetlistEntry],
    route_fn: object,
    pad_positions_fn: object,
    grid: _Grid,
    bcu_grid: _Grid | None,
    grid_step_mm: float,
) -> None:
    """Rip-up-and-retry loop: improve worst routes (up to 3 iterations)."""
    for _ripup_iter in range(3):
        offenders: list[tuple[float, int]] = []
        for idx, r in enumerate(results):
            if not r.routed:
                continue
            score_entry = entry_by_name.get(r.net_name)
            if score_entry is None:
                continue
            q = _score_route(r, pad_positions_fn(score_entry))  # type: ignore[operator]
            short_net_excess_bends = (
                q.bend_count >= 4 and q.manhattan_ideal_mm < 40.0
            )
            if q.via_count > 2 or q.length_ratio > 1.55 or short_net_excess_bends:
                offenders.append((q.score, idx))

        if not offenders:
            break

        offenders.sort(key=lambda x: x[0], reverse=True)
        n_ripup = max(1, len(offenders) // 5)
        ripup_indices = [idx for _, idx in offenders[:n_ripup]]

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


def _estimated_route_length(
    entry: NetlistEntry,
    fp_by_ref: dict[str, Footprint],
) -> float:
    """Estimate max Manhattan distance between any two pads of *entry*."""
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


def _sort_routable_nets(
    routable: list[NetlistEntry],
    footprints: list[Footprint],
    fp_by_ref: dict[str, Footprint],
) -> None:
    """Sort *routable* in-place by routing priority tier and length."""
    ic_refs = {
        fp.ref for fp in footprints
        if fp.ref.startswith("U") and len(fp.pads) >= 6
    }
    conn_refs = {
        fp.ref for fp in footprints
        if sum(1 for p in fp.pads if p.pad_type == "thru_hole") > 6
    }

    def _sort_key(entry: NetlistEntry) -> tuple[int, float]:
        name = entry.net.name.upper()
        is_power = name.startswith("+") or "VDD" in name or "VCC" in name or "VBUS" in name
        has_ic = any(ref in ic_refs for ref, _ in entry.pad_refs)
        has_conn = any(ref in conn_refs for ref, _ in entry.pad_refs)
        est_len = _estimated_route_length(entry, fp_by_ref)
        if is_power:
            tier = 0
        elif has_ic and has_conn:
            tier = 1
            est_len = -est_len  # longest first
        elif has_ic:
            tier = 2
        else:
            tier = 3
        return (tier, est_len)

    routable.sort(key=_sort_key)


def _make_route_entry_fn(
    footprints: list[Footprint],
    board_width_mm: float,
    board_height_mm: float,
    grid_step_mm: float,
    grid: _Grid,
    bcu_grid: _Grid | None,
    net_widths: dict[str, float] | None,
    net_clearances: dict[str, float] | None,
    all_placed_vias: list[tuple[float, float]],
) -> object:
    """Create a closure that routes a single :class:`NetlistEntry`."""

    def _route_entry(entry: NetlistEntry) -> RouteResult:
        net_name = entry.net.name
        if net_widths is not None:
            width = net_widths.get(net_name, 0.25)
        else:
            width = 0.5 if "GND" in net_name or "PWR" in net_name else 0.25

        clearance = 0.2
        if net_clearances is not None:
            clearance = net_clearances.get(net_name, 0.2)

        n_pads = len(entry.pad_refs)
        if n_pads <= 2:
            via_budget = 2
        elif n_pads <= 4:
            via_budget = 4
        else:
            via_budget = 6

        request = RouteRequest(
            net_number=entry.net.number,
            net_name=net_name,
            pad_refs=entry.pad_refs,
            layer="F.Cu",
            width_mm=width,
            clearance_mm=clearance,
            max_vias=via_budget,
        )
        return route_net(
            request, footprints, board_width_mm, board_height_mm,
            grid_step_mm, grid=grid, net_clearances=net_clearances,
            net_widths=net_widths, bcu_grid=bcu_grid,
            placed_via_positions=all_placed_vias,
        )

    return _route_entry


def _pad_positions_for_entry(
    entry: NetlistEntry,
    fp_by_ref: dict[str, Footprint],
) -> list[tuple[float, float]]:
    """Return absolute pad positions for all pads in *entry*."""
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


def _route_with_retries(
    routable: list[NetlistEntry],
    route_fn: object,
    record_vias_fn: object,
    results: list[RouteResult],
    footprints: list[Footprint],
    board_width_mm: float,
    board_height_mm: float,
    grid_step_mm: float,
    grid: _Grid,
    bcu_grid: _Grid | None,
    net_widths: dict[str, float] | None,
    net_clearances: dict[str, float] | None,
    all_placed_vias: list[tuple[float, float]],
) -> None:
    """Run first-pass routing with standard and relaxed retries."""
    failed_entries: list[NetlistEntry] = []
    for entry in routable:
        result = route_fn(entry)  # type: ignore[operator]
        if result.routed:
            results.append(result)
            record_vias_fn(result)  # type: ignore[operator]
        else:
            failed_entries.append(entry)

    still_failed = _retry_failed_nets(
        failed_entries, route_fn, record_vias_fn, results,
    )

    _retry_relaxed_clearance(
        still_failed, footprints, board_width_mm, board_height_mm,
        grid_step_mm, grid, bcu_grid, net_widths, net_clearances,
        all_placed_vias, results, record_vias_fn,
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
    corner_radius_mm: float = 0.0,
) -> tuple[RouteResult, ...]:
    """Route all nets using a shared occupancy grid.

    Returns a tuple of :class:`RouteResult`, one per routed net entry.
    """
    routable = [
        e for e in netlist.entries
        if len(e.pad_refs) >= 2 and e.net.name != "GND"
    ]

    fp_by_ref: dict[str, Footprint] = {fp.ref: fp for fp in footprints}
    _sort_routable_nets(routable, footprints, fp_by_ref)

    # Create shared grids
    grid = _Grid.create(board_width_mm, board_height_mm, grid_step_mm)
    _prepare_grid(
        grid, list(footprints), keepouts=keepouts,
        net_clearances=net_clearances, net_widths=net_widths,
        corner_radius_mm=corner_radius_mm,
    )
    bcu_grid = _prepare_bcu_grid(
        grid, list(footprints), keepouts=keepouts,
        net_clearances=net_clearances, net_widths=net_widths,
        corner_radius_mm=corner_radius_mm,
    )

    results: list[RouteResult] = []
    all_placed_vias: list[tuple[float, float]] = []

    _route_entry = _make_route_entry_fn(
        footprints, board_width_mm, board_height_mm, grid_step_mm,
        grid, bcu_grid, net_widths, net_clearances, all_placed_vias,
    )

    def _record_vias(result: RouteResult) -> None:
        for v in result.vias:
            all_placed_vias.append((v.position.x, v.position.y))

    _route_with_retries(
        routable, _route_entry, _record_vias, results, footprints,
        board_width_mm, board_height_mm, grid_step_mm,
        grid, bcu_grid, net_widths, net_clearances, all_placed_vias,
    )

    entry_by_name: dict[str, NetlistEntry] = {e.net.name: e for e in routable}
    _rip_up_and_retry(
        results, entry_by_name, _route_entry,
        lambda e: _pad_positions_for_entry(e, fp_by_ref),
        grid, bcu_grid, grid_step_mm,
    )

    results = _validate_track_clearances(
        results, grid, bcu_grid, grid_step_mm, entry_by_name,
        _route_entry, footprints, net_clearances, net_widths,
        lambda e: _pad_positions_for_entry(e, fp_by_ref),
    )
    results = _drop_pad_crossing_tracks(results, footprints)

    return tuple(results)



# Post-route validation and collection functions extracted to post_route.py.
# Re-exported here for backwards compatibility.  Must remain at module
# bottom to avoid circular import (post_route imports RouteResult).
from kicad_pipeline.routing.post_route import (  # noqa: E402, I001
    collect_tracks as collect_tracks,
    collect_vias as collect_vias,
    drop_pad_crossing_tracks as _drop_pad_crossing_tracks,
    point_to_segment_dist as _point_to_segment_dist,  # noqa: F401
    segment_min_distance as _segment_min_distance,  # noqa: F401
    unmark_route_tracks as _unmark_route_tracks,
    validate_track_clearances as _validate_track_clearances,
)
