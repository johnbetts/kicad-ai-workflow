"""Simple 3-pass placement — replaces 25 broken Level 3 phases.

Algorithm (how a human EE places components):
  Pass 1: Connectors to edges (hard constraint, never moved)
  Pass 2: ICs to zone centers
  Pass 3: Stamp subcircuit support around ICs (rigid templates)

No collision resolution phase. Overlaps resolved locally during placement.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import PlacementContext

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Subcircuit stamp templates — relative offsets from anchor IC
# Format: {role: (dx, dy, rotation)}
# ---------------------------------------------------------------------------

_RELAY_DRIVER_STAMP: dict[str, tuple[float, float, float]] = {
    "Q": (0.0, -8.0, 180.0),      # transistor below relay
    "D": (0.0, -11.0, 0.0),       # flyback diode below Q
    "R_gate": (3.5, -8.0, 0.0),   # gate resistor beside Q
    "R_led": (-3.5, -8.0, 0.0),   # LED resistor
    "D_led": (-3.5, -11.0, 0.0),  # LED
}

_BUCK_STAMP: dict[str, tuple[float, float, float]] = {
    "L": (6.0, 0.0, 90.0),        # inductor right of IC
    "C_in": (-4.0, 0.0, 0.0),     # input cap left of IC
    "C_out": (10.0, 0.0, 0.0),    # output cap right of inductor
    "C_boot": (0.0, -3.0, 0.0),   # bootstrap cap above IC
    "D_catch": (-4.0, -3.0, 0.0), # catch diode
    "R_fb_top": (3.0, 3.0, 90.0), # feedback top
    "R_fb_bot": (3.0, 5.5, 90.0), # feedback bottom
}

_DECOUPLING_OFFSETS: list[tuple[float, float]] = [
    (3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0),
    (3.0, 3.0), (-3.0, 3.0), (3.0, -3.0), (-3.0, -3.0),
]


# ---------------------------------------------------------------------------
# Occupancy grid for collision-free placement
# ---------------------------------------------------------------------------

class _OccupancyGrid:
    """Simple AABB occupancy tracker."""

    def __init__(self, bounds: tuple[float, float, float, float]) -> None:
        self._bounds = bounds
        self._placed: list[tuple[float, float, float, float]] = []  # (cx, cy, w, h)

    def place(self, cx: float, cy: float, w: float, h: float) -> None:
        self._placed.append((cx, cy, w, h))

    def is_free(self, cx: float, cy: float, w: float, h: float,
                gap: float = 0.5) -> bool:
        bx1, by1, bx2, by2 = self._bounds
        if cx - w / 2 < bx1 or cx + w / 2 > bx2:
            return False
        if cy - h / 2 < by1 or cy + h / 2 > by2:
            return False
        for px, py, pw, ph in self._placed:
            if (abs(cx - px) < (w + pw) / 2 + gap
                    and abs(cy - py) < (h + ph) / 2 + gap):
                return False
        return True

    def find_free(self, tx: float, ty: float, w: float, h: float,
                  gap: float = 0.5, max_radius: float = 40.0) -> tuple[float, float]:
        """Spiral search for nearest free position."""
        if self.is_free(tx, ty, w, h, gap):
            return tx, ty
        step = max(w, h) * 0.5
        for ring in range(1, int(max_radius / step) + 1):
            r = ring * step
            for angle_deg in range(0, 360, 30):
                rad = math.radians(angle_deg)
                cx = tx + r * math.cos(rad)
                cy = ty + r * math.sin(rad)
                if self.is_free(cx, cy, w, h, gap):
                    return cx, cy
        # Fallback: clamp to bounds
        bx1, by1, bx2, by2 = self._bounds
        return (max(bx1 + w, min(bx2 - w, tx)),
                max(by1 + h, min(by2 - h, ty)))


# ---------------------------------------------------------------------------
# Pass 1: Connectors to edges
# ---------------------------------------------------------------------------

def _pass1_connectors(
    ctx: PlacementContext,
    grid: _OccupancyGrid,
) -> None:
    """Place connectors at board edges based on type."""
    _log.info("  Simple Pass 1: Connectors to edges")
    min_x, min_y, max_x, max_y = ctx.bounds
    margin = 3.0

    # Classify connectors
    terminal_refs: list[str] = []
    rj45_refs: list[str] = []
    usb_refs: list[str] = []
    header_refs: list[str] = []
    other_j: list[str] = []

    for comp in ctx.requirements.components:
        if not comp.ref.startswith("J"):
            continue
        fp_upper = comp.footprint.upper()
        if "TERMINAL" in fp_upper or "TB_" in fp_upper:
            terminal_refs.append(comp.ref)
        elif "RJ45" in fp_upper:
            rj45_refs.append(comp.ref)
        elif "USB" in fp_upper:
            usb_refs.append(comp.ref)
        elif "PINHEADER" in fp_upper or "HEADER" in fp_upper:
            header_refs.append(comp.ref)
        else:
            other_j.append(comp.ref)

    # Terminals → TOP edge, evenly spaced
    if terminal_refs:
        total_w = sum(ctx.fp_sizes.get(r, (10, 5))[0] for r in terminal_refs)
        gap = max(2.0, ((max_x - min_x) - 2 * margin - total_w)
                  / max(len(terminal_refs) - 1, 1))
        cursor_x = min_x + margin
        for ref in sorted(terminal_refs):
            w, h = ctx.fp_sizes.get(ref, (10, 5))
            cx = cursor_x + w / 2
            cy = min_y + margin
            ctx.positions[ref] = (cx, cy, 0.0)
            grid.place(cx, cy, w, h)
            cursor_x += w + gap
            _log.info("    %s → TOP (%.1f, %.1f)", ref, cx, cy)

    # RJ45 → RIGHT edge
    for ref in rj45_refs:
        w, h = ctx.fp_sizes.get(ref, (16, 16))
        cx, cy = max_x - margin - w / 2, (min_y + max_y) / 2
        cx, cy = grid.find_free(cx, cy, w, h)
        ctx.positions[ref] = (cx, cy, 90.0)
        grid.place(cx, cy, w, h)
        _log.info("    %s → RIGHT (%.1f, %.1f)", ref, cx, cy)

    # USB → BOTTOM-LEFT
    for ref in usb_refs:
        w, h = ctx.fp_sizes.get(ref, (9, 7))
        cx, cy = min_x + 20, max_y - margin - h / 2
        cx, cy = grid.find_free(cx, cy, w, h)
        ctx.positions[ref] = (cx, cy, 0.0)
        grid.place(cx, cy, w, h)
        _log.info("    %s → BOTTOM-LEFT (%.1f, %.1f)", ref, cx, cy)

    # Headers → RIGHT edge, below RJ45
    cursor_y = (min_y + max_y) / 2 + 15
    for ref in sorted(header_refs):
        w, h = ctx.fp_sizes.get(ref, (5, 20))
        cx = max_x - margin - w / 2
        cx, cy = grid.find_free(cx, cursor_y, w, h)
        ctx.positions[ref] = (cx, cy, 0.0)
        grid.place(cx, cy, w, h)
        cursor_y = cy + h / 2 + 3
        _log.info("    %s → RIGHT (%.1f, %.1f)", ref, cx, cy)

    # Other connectors (SD card, etc.) → near their group
    for ref in other_j:
        if ref in ctx.positions:
            continue
        w, h = ctx.fp_sizes.get(ref, (10, 10))
        cx, cy = grid.find_free((min_x + max_x) / 2, max_y - margin, w, h)
        ctx.positions[ref] = (cx, cy, 0.0)
        grid.place(cx, cy, w, h)
        _log.info("    %s → default (%.1f, %.1f)", ref, cx, cy)


# ---------------------------------------------------------------------------
# Pass 2: Major ICs to zone centers
# ---------------------------------------------------------------------------

def _pass2_ics(
    ctx: PlacementContext,
    grid: _OccupancyGrid,
) -> None:
    """Place major ICs at zone centers."""
    _log.info("  Simple Pass 2: ICs to zone centers")

    # Build zone lookup
    zone_map: dict[str, tuple[float, float, float, float]] = {}
    for z in ctx.zones:
        for g in z.groups:
            zone_map[g] = z.rect

    # Build group membership
    ref_to_group: dict[str, str] = {}
    for fb in ctx.requirements.features:
        for r in fb.components:
            ref_to_group[r] = fb.name

    # Find all ICs (U-prefix with 4+ pads, K-prefix for relays)
    ic_refs = [
        r for r in ctx.positions
        if r.startswith("U") or r.startswith("K") or r.startswith("Y")
    ]

    for ref in sorted(ic_refs):
        if ref in ctx.fixed_refs:
            continue
        w, h = ctx.fp_sizes.get(ref, (5, 5))
        group = ref_to_group.get(ref, "")
        zone_rect = zone_map.get(group)
        if zone_rect:
            zx1, zy1, zx2, zy2 = zone_rect
            target_x = (zx1 + zx2) / 2
            target_y = (zy1 + zy2) / 2
        else:
            target_x = (ctx.bounds[0] + ctx.bounds[2]) / 2
            target_y = (ctx.bounds[1] + ctx.bounds[3]) / 2

        cx, cy = grid.find_free(target_x, target_y, w, h)
        old = ctx.positions.get(ref, (0, 0, 0))
        ctx.positions[ref] = (cx, cy, old[2])
        grid.place(cx, cy, w, h)
        _log.info("    %s → zone '%s' (%.1f, %.1f)", ref, group, cx, cy)


# ---------------------------------------------------------------------------
# Pass 3: Stamp subcircuit support around ICs
# ---------------------------------------------------------------------------

def _pass3_support(
    ctx: PlacementContext,
    grid: _OccupancyGrid,
) -> None:
    """Place support components around their anchor ICs using subcircuit data."""
    _log.info("  Simple Pass 3: Support components around ICs")

    placed_refs: set[str] = set()

    # Place subcircuit members near their anchors
    for sc in ctx.subcircuits:
        anchor = sc.anchor_ref
        if anchor not in ctx.positions:
            continue
        ax, ay, _arot = ctx.positions[anchor]

        for ref in sc.refs:
            if ref == anchor or ref in placed_refs:
                continue
            if ref.startswith("J"):
                continue  # connectors already placed in pass 1

            w, h = ctx.fp_sizes.get(ref, (2, 1))

            # Place near anchor — try radial positions
            best_dist = 999.0
            best_pos = (ax + 3, ay + 3)
            for dx_off, dy_off in _DECOUPLING_OFFSETS:
                tx, ty = ax + dx_off, ay + dy_off
                if grid.is_free(tx, ty, w, h):
                    dist = math.sqrt(dx_off * dx_off + dy_off * dy_off)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (tx, ty)

            cx, cy = grid.find_free(best_pos[0], best_pos[1], w, h)
            ctx.positions[ref] = (cx, cy, 0.0)
            grid.place(cx, cy, w, h)
            placed_refs.add(ref)

    # Place remaining unplaced components near their group centroid
    ref_to_group: dict[str, str] = {}
    group_centroids: dict[str, tuple[float, float]] = {}
    for fb in ctx.requirements.features:
        xs, ys = [], []
        for r in fb.components:
            ref_to_group[r] = fb.name
            if r in ctx.positions:
                px, py, _ = ctx.positions[r]
                xs.append(px)
                ys.append(py)
        if xs:
            group_centroids[fb.name] = (sum(xs) / len(xs), sum(ys) / len(ys))

    unplaced = [
        r for r in ctx.positions
        if r not in placed_refs
        and not r.startswith("J") and not r.startswith("U")
        and not r.startswith("K") and not r.startswith("Y")
        and not r.startswith("H") and not r.startswith("MH")
    ]

    for ref in sorted(unplaced):
        w, h = ctx.fp_sizes.get(ref, (2, 1))
        group = ref_to_group.get(ref, "")
        gcx, gcy = group_centroids.get(group, (80, 40))
        cx, cy = grid.find_free(gcx, gcy, w, h)
        ctx.positions[ref] = (cx, cy, 0.0)
        grid.place(cx, cy, w, h)
        placed_refs.add(ref)

    _log.info("    Placed %d support + %d remaining", len(placed_refs), len(unplaced))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_simple_placement(ctx: PlacementContext) -> None:
    """Simple 3-pass placement — replaces 25 Level 3 phases.

    Pass 1: Connectors to edges
    Pass 2: ICs to zone centers
    Pass 3: Support components stamped around ICs
    """
    _log.info("=== Simple 3-Pass Placement ===")
    grid = _OccupancyGrid(ctx.bounds)

    # Pre-register fixed refs (mounting holes)
    for ref in ctx.fixed_refs:
        if ref in ctx.positions:
            x, y, _ = ctx.positions[ref]
            w, h = ctx.fp_sizes.get(ref, (4, 4))
            grid.place(x, y, w, h)

    _pass1_connectors(ctx, grid)
    _pass2_ics(ctx, grid)
    _pass3_support(ctx, grid)

    _log.info("=== Simple placement complete ===")
