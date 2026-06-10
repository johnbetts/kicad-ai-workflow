"""Ethernet group placement helpers.

Contains ethernet-specific placement phases that organize components
within the ethernet functional group.

Extracted from ``ee_phases_groups.py`` to reduce module size.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from kicad_pipeline.optimization.collision_resolver import (
    _PlacementGrid,
    _rotation_aware_size,
)
from kicad_pipeline.pcb.pin_map import (
    origin_to_centroid,
)

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import PlacementContext
from kicad_pipeline.pcb.pin_map import (
    origin_to_centroid,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ethernet sub-step helpers
# ---------------------------------------------------------------------------

_DEFAULT_CRYSTAL_SIZE_MM: tuple[float, float] = (3.2, 1.5)
"""Default crystal oscillator footprint size (SMD 3215 package)."""


def _eth_place_caps_column(
    ctx: PlacementContext,
    caps: list[str],
    col_x: float,
    start_y: float,
    grid: _PlacementGrid,
    placed_eth: set[str],
    zone_rect: tuple[float, float, float, float],
) -> float:
    """Place caps in a vertical column within the ethernet zone. Returns bottom Y."""
    ezx1, ezy1, ezx2, ezy2 = zone_rect
    col_y = start_y
    for ref in caps:
        if (ref not in ctx.positions or ref in ctx.fixed_refs
                or ref in placed_eth or ref == ""):
            continue
        cw, ch = ctx.fp_sizes.get(ref, (1.5, 1.0))
        tx = _clamp(col_x, ezx1 + 2.0, ezx2 - 2.0)
        ty = _clamp(col_y + ch / 2.0, ezy1 + 2.0, ezy2 - 2.0)
        px, py = grid.find_free_pos(tx, ty, cw, ch, max_radius=8.0)
        ctx.positions[ref] = (px, py, 0.0)
        grid.place(px, py, cw, ch)
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        col_y = py + ch / 2.0 + 1.0
    return col_y


def _eth_place_crystal_and_caps(
    ctx: PlacementContext,
    crystal_refs: list[str],
    crystal_load_caps: list[str],
    ic_cx: float,
    ic_cy: float,
    ic_h: float,
    eth_grid: _PlacementGrid,
    placed_eth: set[str],
    zone_rect: tuple[float, float, float, float],
) -> None:
    """Place crystal oscillator and its flanking load caps beside the IC.

    The crystal is placed to the right of U1 (not above it) to avoid
    colliding with the RJ45 connector at the top edge.  Load caps are
    stacked above and below the crystal.
    """
    ezx1, ezy1, ezx2, ezy2 = zone_rect

    # Determine IC width and crystal size
    ic_w = ctx.fp_sizes.get(
        next((r for r in placed_eth if r.startswith("U")), ""), (10.0, 10.0)
    )[0]
    crystal_w, crystal_h = (
        ctx.fp_sizes.get(crystal_refs[0], _DEFAULT_CRYSTAL_SIZE_MM)
        if crystal_refs else _DEFAULT_CRYSTAL_SIZE_MM
    )
    # Place crystal to the right of the IC
    crystal_x = ic_cx + ic_w / 2.0 + crystal_w / 2.0 + 1.0
    crystal_y = ic_cy

    for ref in crystal_refs:
        if (ref not in ctx.positions or ref in ctx.fixed_refs
                or ref in placed_eth or ref == ""):
            continue
        w, h = ctx.fp_sizes.get(ref, _DEFAULT_CRYSTAL_SIZE_MM)
        tx = _clamp(crystal_x, ezx1 + w / 2.0 + 2.0, ezx2 - w / 2.0 - 2.0)
        ty = _clamp(crystal_y, ezy1 + h / 2.0 + 2.0, ezy2 - h / 2.0 - 2.0)
        px, py = eth_grid.find_free_pos(tx, ty, w, h, max_radius=8.0)
        ctx.positions[ref] = (px, py, 0.0)
        eth_grid.place(px, py, w, h)
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        crystal_x = px  # Update in case find_free_pos shifted it
        crystal_y = py
        crystal_dist = math.sqrt((px - ic_cx) ** 2 + (py - ic_cy) ** 2)
        _log.info("    %s (crystal) -> (%.1f, %.1f) dist=%.1fmm from IC",
                  ref, px, py, crystal_dist)

    # Load caps stacked above/below the crystal
    cap_idx = 0
    for ref in crystal_load_caps:
        if (ref not in ctx.positions or ref in ctx.fixed_refs
                or ref in placed_eth or ref == ""):
            continue
        w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
        cap_dy = crystal_h / 2.0 + h / 2.0 + 0.5
        ty = crystal_y - cap_dy if cap_idx % 2 == 0 else crystal_y + cap_dy
        tx = crystal_x
        tx = _clamp(tx, ezx1 + 2.0, ezx2 - 2.0)
        ty = _clamp(ty, ezy1 + 2.0, ezy2 - 2.0)
        px, py = eth_grid.find_free_pos(tx, ty, w, h, max_radius=5.0)
        ctx.positions[ref] = (px, py, 0.0)
        eth_grid.place(px, py, w, h)
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        cap_idx += 1


def _eth_place_poe_ic_and_caps(
    ctx: PlacementContext,
    poe_ic: str,
    poe_caps: list[str],
    eth_anchor_x: float,
    eth_grid: _PlacementGrid,
    placed_eth: set[str],
) -> None:
    """Place PoE/PHY IC and its decoupling caps."""
    bounds = ctx.bounds
    if not (poe_ic and poe_ic in ctx.positions and poe_ic not in ctx.fixed_refs):
        return
    poe_w, poe_h = ctx.fp_sizes.get(poe_ic, (8.0, 8.0))
    poe_eff_w, poe_eff_h = poe_h, poe_w
    poe_tx = eth_anchor_x
    poe_ty = bounds[3] - 25.0
    poe_tx = _clamp(poe_tx, bounds[0] + poe_eff_w / 2 + 1,
                    bounds[2] - poe_eff_w / 2 - 1)
    poe_ty = _clamp(poe_ty, bounds[1] + poe_eff_h / 2 + 1,
                    bounds[3] - poe_eff_h / 2 - 1)
    ppx, ppy = eth_grid.find_free_pos(
        poe_tx, poe_ty, poe_eff_w, poe_eff_h, max_radius=15.0,
    )
    ctx.positions[poe_ic] = (ppx, ppy, 270.0)
    eth_grid.place(ppx, ppy, poe_eff_w, poe_eff_h)
    ctx.ethernet_fixed.add(poe_ic)
    placed_eth.add(poe_ic)
    _log.info("    %s (PHY) -> (%.1f, %.1f) rot=270", poe_ic, ppx, ppy)
    cap_y_poe = ppy - poe_eff_h / 2.0 - 2.0
    for cap_ref in poe_caps:
        if (cap_ref not in ctx.positions or cap_ref in ctx.fixed_refs
                or cap_ref in placed_eth):
            continue
        cw, ch = ctx.fp_sizes.get(cap_ref, (1.0, 0.5))
        cpx, cpy = eth_grid.find_free_pos(
            ppx, cap_y_poe, cw, ch, max_radius=8.0,
        )
        ctx.positions[cap_ref] = (cpx, cpy, 0.0)
        eth_grid.place(cpx, cpy, cw, ch)
        ctx.ethernet_fixed.add(cap_ref)
        placed_eth.add(cap_ref)
        cap_y_poe = cpy - ch / 2.0 - 1.0


def _eth_place_rj45_connectors(
    ctx: PlacementContext,
    eth_connectors: list[str],
    eth_anchor_x: float,
    eth_grid: _PlacementGrid,
    placed_eth: set[str],
    zone_rect: tuple[float, float, float, float],
) -> None:
    """Place RJ45 connectors at the RIGHT board edge (or edge_mapped edge).

    Rotation=270 faces the port outward (right / x_max).  After rotation
    the effective width becomes h and the effective height becomes w.
    Origin is adjusted so all pads stay inside the board.
    """
    bounds = ctx.bounds
    _ezx1, ezy1, _ezx2, ezy2 = zone_rect
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space
    zone_cy = (ezy1 + ezy2) / 2.0
    for ref in eth_connectors:
        if ref not in ctx.positions or ref in ctx.fixed_refs:
            continue
        # Respect edge_mapped_connectors if set by phase 3f3
        intended_edge = getattr(ctx, "edge_mapped_connectors", {}).get(ref, "right")
        use_left = intended_edge == "left"
        rotation = 90.0 if use_left else 270.0
        fp_match = None
        for fp in ctx.initial_pcb.footprints:
            if fp.ref == ref:
                fp_match = fp
                break
        w, h = ctx.fp_sizes.get(ref, (19.6, 15.4))
        rot_w, rot_h = h, w
        edge_margin = 1.0
        if use_left:
            cent_x = bounds[0] + rot_w / 2.0 + 1.0
        else:
            cent_x = bounds[2] - rot_w / 2.0 - 1.0
        cent_y = _clamp(zone_cy, ezy1 + rot_h / 2.0 + 1.0, ezy2 - rot_h / 2.0 - 1.0)
        if fp_match is not None:
            if use_left:
                trial_origin_x = bounds[0] + 3.0
                trial_origin_y = zone_cy
                pad_min_x, _, _, _ = pad_extent_in_board_space(
                    fp_match, trial_origin_x, trial_origin_y, rotation,
                )
                if pad_min_x < bounds[0] + edge_margin:
                    trial_origin_x += (bounds[0] + edge_margin - pad_min_x)
            else:
                trial_origin_x = bounds[2] - 3.0
                trial_origin_y = zone_cy
                _, _, pad_max_x, _ = pad_extent_in_board_space(
                    fp_match, trial_origin_x, trial_origin_y, rotation,
                )
                if pad_max_x > bounds[2] - edge_margin:
                    trial_origin_x -= (pad_max_x - bounds[2] + edge_margin)
            cent_x, cent_y = origin_to_centroid(
                fp_match, trial_origin_x, trial_origin_y, rotation,
            )
            cent_y = _clamp(cent_y, ezy1 + rot_h / 2.0 + 1.0, ezy2 - rot_h / 2.0 - 1.0)
        ctx.positions[ref] = (cent_x, cent_y, rotation)
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        edge_name = "left" if use_left else "right"
        _log.info("    %s (RJ45) -> %s edge (%.1f, %.1f) rot=%.0f", ref, edge_name, cent_x, cent_y, rotation)


def _eth_fix_crystal_cap_overlaps(
    ctx: PlacementContext,
    placed_eth: set[str],
    zone_rect: tuple[float, float, float, float],
    crystal_load_caps: list[str] | None = None,
) -> None:
    """Shift caps that overlap crystals in the ethernet group.

    Crystal load caps (C4/C5) are intentionally placed flanking the crystal
    by ``_eth_place_crystal_and_caps`` and must not be repositioned here.
    """
    _ezy1 = zone_rect[1]
    _ezy2 = zone_rect[3]
    _crystal_load: set[str] = set(crystal_load_caps) if crystal_load_caps else set()
    crystal_placed = [r for r in placed_eth if r.startswith("Y")]
    for yref in crystal_placed:
        yx, yy, _yrot = ctx.positions[yref]
        yw, yh = ctx.fp_sizes.get(yref, _DEFAULT_CRYSTAL_SIZE_MM)
        shift_count = 0
        for cref in sorted(placed_eth):
            if cref == yref or not cref.startswith("C"):
                continue
            # Never reposition crystal load caps — they were deliberately placed
            # adjacent to the crystal by _eth_place_crystal_and_caps.
            if cref in _crystal_load:
                continue
            cx, cy, crot = ctx.positions[cref]
            cw, ch = ctx.fp_sizes.get(cref, (1.5, 1.0))
            _margin = 2.0
            if (abs(cx - yx) < (cw + yw) / 2.0 + _margin
                    and abs(cy - yy) < (ch + yh) / 2.0 + _margin):
                # Offset each subsequent cap further to avoid stacking
                cap_offset = (ch + 1.5) * shift_count
                new_cy = yy + yh / 2.0 + ch / 2.0 + 2.0 + cap_offset
                new_cy = _clamp(new_cy, _ezy1 + 2.0, _ezy2 - 2.0)
                ctx.positions[cref] = (cx, new_cy, crot)
                shift_count += 1
                _log.info("    3c4: shifted %s below %s (%.1f,%.1f) -> (%.1f,%.1f)",
                          cref, yref, cx, cy, cx, new_cy)


def _eth_connector_pad_right(
    ctx: PlacementContext,
    eth_connectors: list[str],
    placed_eth: set[str],
    gap_mm: float = 2.0,
) -> float:
    """Return the rightmost pad-extent X of placed RJ45 connectors plus *gap_mm*.

    Uses the actual pad extent in board space (accounting for through-hole
    offsets) rather than the courtyard size, so the IC anchor is placed
    immediately to the right of the connector's real pad footprint — not an
    over-estimate based on the courtyard envelope.

    Falls back to centroid-X + courtyard-half + gap if the footprint is not
    found in initial_pcb.
    """
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space

    max_pad_x: float = ctx.bounds[0]
    for jref in eth_connectors:
        if jref not in placed_eth:
            continue
        pos = ctx.positions.get(jref)
        if pos is None:
            continue
        jx, jy, jrot = pos
        fp_match = next(
            (fp for fp in ctx.initial_pcb.footprints if fp.ref == jref), None,
        )
        if fp_match is not None:
            _, _, pad_x1, _ = pad_extent_in_board_space(fp_match, jx, jy, jrot)
            max_pad_x = max(max_pad_x, pad_x1)
        else:
            jw, _ = ctx.fp_sizes.get(jref, (19.6, 15.4))
            max_pad_x = max(max_pad_x, jx + jw / 2.0)
    return max_pad_x + gap_mm


def _eth_compute_ic_anchor_x(
    ctx: PlacementContext,
    eth_connectors: list[str],
    placed_eth: set[str],
) -> float:
    ic_anchor_x = _eth_connector_pad_right(ctx, eth_connectors, placed_eth, gap_mm=5.0)
    if not eth_connectors or not any(r in placed_eth for r in eth_connectors):
        max_conn_w = max(
            (ctx.fp_sizes.get(r, (19.6, 0.0))[0] for r in eth_connectors),
            default=19.6,
        )
        ic_anchor_x = ctx.bounds[0] + max_conn_w + 5.0
    return ic_anchor_x


# Helper functions that need to be imported
_DEFAULT_CRYSTAL_SIZE_MM: tuple[float, float] = (3.2, 1.5)

def _clamp(val: float, lo: float, hi: float) -> float:
    """Clamp *val* to [lo, hi]."""
    return max(lo, min(hi, val))


def _eth_force_place_main_ic(
    ctx: PlacementContext,
    eth_main_ic: str,
    ic_anchor_x: float,
    ic_anchor_y: float,
    eth_grid: object,
    placed_eth: set[str],
    eth_zone_rect: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Force-place the main ethernet IC to the right of the RJ45 connector.

    The IC is placed at *ic_anchor_x* (derived from the connector's right pad
    extent) and *ic_anchor_y* (vertical center of the ethernet zone), which
    implements the left→right signal-flow layout: J1→U1→headers.

    Returns (ic_cx, ic_cy, ic_w, ic_h).
    """
    ezx1, ezy1, ezx2, ezy2 = eth_zone_rect
    ic_w, ic_h = ctx.fp_sizes.get(eth_main_ic, (10.0, 10.0))
    ic_cx = _clamp(ic_anchor_x + ic_w / 2.0, ezx1 + ic_w / 2.0 + 1.0, ezx2 - ic_w / 2.0 - 1.0)
    ic_cy = _clamp(ic_anchor_y, ezy1 + ic_h / 2.0 + 1.0, ezy2 - ic_h / 2.0 - 1.0)
    if eth_main_ic in ctx.positions and eth_main_ic not in ctx.fixed_refs:
        ctx.positions[eth_main_ic] = (ic_cx, ic_cy, 0.0)
        eth_grid.place(ic_cx, ic_cy, ic_w, ic_h)  # type: ignore[union-attr]
        ctx.ethernet_fixed.add(eth_main_ic)
        ctx.fixed_refs.add(eth_main_ic)
        placed_eth.add(eth_main_ic)
        _log.info("    %s (W5500) force-placed at (%.1f, %.1f) in ethernet zone",
                  eth_main_ic, ic_cx, ic_cy)
    return ic_cx, ic_cy, ic_w, ic_h


def _eth_register_connectors(
    ctx: PlacementContext,
    eth_connectors: list[str],
    eth_anchor_x: float,
    eth_grid: object,
    board_min_y: float,
) -> None:
    """Pre-register RJ45 connector footprints in the ethernet occupancy grid.

    Connectors are reserved at the top edge so the IC/crystal placement logic
    knows the space is taken.
    """
    for jref in eth_connectors:
        jw, jh = ctx.fp_sizes.get(jref, (19.6, 12.5))
        eth_grid.place(eth_anchor_x, board_min_y + jh / 2.0 + 1.0, jw, jh)  # type: ignore[union-attr]


def _eth_place_column(
    ctx: PlacementContext,
    refs: list[str],
    col_x: float,
    start_y: float,
    placed_eth: set[str],
    eth_grid: object,
    eth_zone_rect: tuple[float, float, float, float],
    strip_gap: float = 1.0,
) -> float:
    """Place *refs* in a vertical column within the ethernet zone. Returns bottom Y."""
    ezx1, ezy1, ezx2, ezy2 = eth_zone_rect
    cy = start_y
    for ref in refs:
        if ref not in ctx.positions or ref in ctx.fixed_refs or ref in placed_eth or ref == "":
            continue
        w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        tx = _clamp(col_x, ezx1 + 2.0, ezx2 - 2.0)
        ty = _clamp(cy + h / 2.0, ezy1 + 2.0, ezy2 - 2.0)
        px, py = eth_grid.find_free_pos(tx, ty, w, h, max_radius=6.0)  # type: ignore[union-attr]
        ctx.positions[ref] = (px, py, 0.0)
        eth_grid.place(px, py, w, h)  # type: ignore[union-attr]
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        cy = py + h / 2.0 + strip_gap
    return cy


def _eth_place_signal_chain(
    ctx: PlacementContext,
    crystal_refs: list[str],
    crystal_load_caps: list[str],
    other_caps: list[str],
    poe_ic: str,
    poe_caps: list[str],
    eth_connectors: list[str],
    eth_anchor_x: float,
    ic_cx: float,
    ic_cy: float,
    ic_h: float,
    eth_grid: object,
    placed_eth: set[str],
    eth_zone_rect: tuple[float, float, float, float],
) -> float:
    """Place all ethernet signal-chain components. Returns the bottom column Y.

    RJ45 connectors must already be placed before calling this function
    (done in _phase_ethernet_group so the IC anchor uses the real pad extent).
    """
    col1_bottom = _eth_place_caps_column(
        ctx, other_caps, ic_cx, ic_cy + ic_h / 2.0 + 1.5,
        eth_grid, placed_eth, eth_zone_rect,
    )
    _eth_place_poe_ic_and_caps(ctx, poe_ic, poe_caps, eth_anchor_x, eth_grid, placed_eth)
    # Note: _eth_place_rj45_connectors is called BEFORE this function in
    # _phase_ethernet_group so the IC anchor is based on actual pad extent.
    # Connectors already in placed_eth are skipped automatically here.
    _eth_place_rj45_connectors(
        ctx, eth_connectors, eth_anchor_x, eth_grid, placed_eth, eth_zone_rect,
    )
    return col1_bottom


def _eth_push_non_group_clear(ctx: PlacementContext) -> None:
    """Push non-ethernet components clear of ethernet IC footprints (3c4-post)."""
    skip_refs = ctx.relay_support_refs | {r for r in ctx.positions if r.startswith("K")}
    for eic in [r for r in ctx.ethernet_fixed if r.startswith("U")]:
        _push_non_group_away_from_ic(eic, ctx, ctx.ethernet_fixed, skip_refs)


def _eth_place_headers_bottom(
    ctx: PlacementContext,
    eth_headers: list[str],
    eth_grid: object,
    placed_eth: set[str],
    zone_rect: tuple[float, float, float, float],
) -> None:
    """Place small interface headers (SPI, power) at the RIGHT board edge.

    In the left→right signal-flow layout (J1→U1→headers) the SPI/power
    headers terminate the chain at the right edge.  Headers are stacked
    vertically near x_max, centered within the ethernet zone's y-range.

    Pad extents are checked against the board boundary to ensure no pads
    extend past the board edge.
    """
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space

    bounds = ctx.bounds
    _ezx1, ezy1, ezx2, ezy2 = zone_rect
    unplaced = [r for r in eth_headers
                if r in ctx.positions and r not in ctx.fixed_refs and r not in placed_eth]
    if not unplaced:
        return
    total_h = sum(ctx.fp_sizes.get(r, (2.54, 5.08))[1] for r in unplaced)
    gap = 2.0
    avail_h = ezy2 - ezy1
    if total_h + gap * (len(unplaced) - 1) > avail_h:
        gap = max(1.0, (avail_h - total_h) / max(len(unplaced) - 1, 1))
    cy = ezy1 + (avail_h - (total_h + gap * (len(unplaced) - 1))) / 2.0
    for ref in unplaced:
        w, h = ctx.fp_sizes.get(ref, (2.54, 5.08))
        x = _clamp(bounds[2] - w / 2.0 - 1.0, _ezx1 + w / 2.0 + 1.0, ezx2 - w / 2.0 - 1.0)
        y = _clamp(cy + h / 2.0, ezy1 + h / 2.0 + 1.0, ezy2 - h / 2.0 - 1.0)

        # Check actual pad extents against board bounds and adjust if needed.
        fp_match = None
        for fp in ctx.initial_pcb.footprints:
            if fp.ref == ref:
                fp_match = fp
                break
        if fp_match is not None:
            rot = ctx.positions.get(ref, (0.0, 0.0, 0.0))[2]
            pad_min_x, pad_min_y, pad_max_x, pad_max_y = pad_extent_in_board_space(
                fp_match, x, y, rot,
            )
            edge_margin = 1.0
            # Pull inward if pads exceed board bounds
            if pad_min_x < bounds[0] + edge_margin:
                x += (bounds[0] + edge_margin - pad_min_x)
            if pad_max_x > bounds[2] - edge_margin:
                x -= (pad_max_x - bounds[2] + edge_margin)
            if pad_min_y < bounds[1] + edge_margin:
                y += (bounds[1] + edge_margin - pad_min_y)
            if pad_max_y > bounds[3] - edge_margin:
                y -= (pad_max_y - bounds[3] + edge_margin)

        ctx.positions[ref] = (x, y, 0.0)
        eth_grid.place(x, y, w, h)  # type: ignore[union-attr]
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        _log.info("    %s (header) -> right edge (%.1f, %.1f)", ref, x, y)
        cy += h + gap


def _eth_place_remaining(
    ctx: PlacementContext,
    eth_group_refs: set[str],
    placed_eth: set[str],
    eth_anchor_x: float,
    col1_bottom: float,
    eth_grid: object,
    eth_zone_rect: tuple[float, float, float, float],
) -> None:
    remaining_eth = sorted(eth_group_refs - placed_eth - ctx.fixed_refs)
    if remaining_eth:
        _eth_place_column(
            ctx,
            [r for r in remaining_eth if r in ctx.positions],
            eth_anchor_x, col1_bottom, placed_eth, eth_grid, eth_zone_rect,
        )