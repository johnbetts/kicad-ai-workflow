"""EE placement optimizer — group organization phases.

Contains the group-specific placement phases that organize components
within functional groups (power, MCU, ethernet, ADC channels).

Extracted from ``ee_phases.py`` to reduce module size.
"""

from __future__ import annotations

import logging
import math

from kicad_pipeline.optimization.collision_resolver import (
    _PlacementGrid,
    _rotation_aware_size,
)
from kicad_pipeline.optimization.placement_types import (
    PlacementContext,
)
from kicad_pipeline.pcb.pin_map import (
    origin_to_centroid,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers — reduce cyclomatic complexity of group-phase functions
# ---------------------------------------------------------------------------

_POWER_NET_PREFIXES = (
    "VCC", "VDD", "GND", "AGND", "DGND", "AVCC", "+3V3", "+5V",
    "VIN", "VBUS", "V_", "+12V", "+24V", "I2C_",
)

_SMALL_PREFIXES = ("R", "C", "D", "L", "LED", "SW")

# Default fallback footprint sizes (w, h) in mm for named components
_DEFAULT_CRYSTAL_SIZE_MM: tuple[float, float] = (3.2, 1.5)
"""Default crystal oscillator footprint size (SMD 3215 package)."""

_DEFAULT_J2_SIZE_MM: tuple[float, float] = (9.6, 7.6)
"""Default USB-C connector (J2) footprint size."""

_DEFAULT_J14_SIZE_MM: tuple[float, float] = (2.7, 35.7)
"""Default pin-header connector (J14) footprint size."""

_BOARD_EDGE_MARGIN_MM: float = 2.0
"""Margin from board edge for component placement within groups."""

_ZONE_CLAMP_MARGIN_MM: float = 2.0
"""Margin used when clamping positions inside zone rectangles."""


def _clamp(val: float, lo: float, hi: float) -> float:
    """Clamp *val* to [lo, hi]."""
    return max(lo, min(hi, val))


def _clamp_to_bounds(
    x: float, y: float, bounds: tuple[float, float, float, float],
    margin: float = _BOARD_EDGE_MARGIN_MM,
) -> tuple[float, float]:
    """Clamp (x, y) inside *bounds* with *margin*."""
    return (
        _clamp(x, bounds[0] + margin, bounds[2] - margin),
        _clamp(y, bounds[1] + margin, bounds[3] - margin),
    )


def _collect_feature_refs(
    ctx: PlacementContext, *keywords: str,
) -> set[str]:
    """Return component refs from the first FeatureBlock whose name matches any keyword."""
    for feat in ctx.requirements.features:
        feat_lower = feat.name.lower()
        if any(kw in feat_lower for kw in keywords):
            refs: set[str] = set()
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                refs.add(r)
            return refs
    return set()


def _find_zone_rect(
    ctx: PlacementContext, zone_name: str,
) -> tuple[float, float, float, float] | None:
    """Return the rect of the zone named *zone_name*, or None."""
    for z in ctx.zones:
        if z.name == zone_name:
            return z.rect
    return None


def _build_net_to_group_refs(
    ctx: PlacementContext, group_refs: set[str],
) -> dict[str, set[str]]:
    """Map net-name -> set of refs that are in *group_refs*."""
    mapping: dict[str, set[str]] = {}
    for net in ctx.requirements.nets:
        matched = {c.ref for c in net.connections if c.ref in group_refs}
        if matched:
            mapping[net.name] = matched
    return mapping


def _build_exclusion_grid(
    ctx: PlacementContext,
    exclude_refs: set[str],
) -> _PlacementGrid:
    """Build a PlacementGrid populated with all refs NOT in *exclude_refs*."""
    grid = _PlacementGrid(ctx.bounds)
    for oref, (ox, oy, _orot) in ctx.positions.items():
        if oref in exclude_refs:
            continue
        ow, oh = _rotation_aware_size(oref, ctx.positions, ctx.fp_sizes)
        grid.place(ox, oy, ow, oh)
    return grid


def _is_power_or_bus_net(name: str) -> bool:
    """Return True if *name* looks like a power or bus net."""
    upper = name.upper()
    return any(upper.startswith(p) for p in _POWER_NET_PREFIXES)


def _is_small_passive(ref: str) -> bool:
    """Return True if *ref* is a small passive/LED/switch."""
    return any(ref.startswith(p) for p in _SMALL_PREFIXES)


def _expand_one_hop(
    seed_refs: set[str],
    ctx: PlacementContext,
    already_claimed: set[str],
    allow_small_ics: bool = False,
    max_ic_pins: int = 6,
) -> set[str]:
    """Expand *seed_refs* by one signal-net hop.

    Returns NEW refs discovered (excludes seeds and *already_claimed*).
    """
    new_refs: set[str] = set()
    for ref in seed_refs:
        for net in ctx.requirements.nets:
            if _is_power_or_bus_net(net.name):
                continue
            if not any(c.ref == ref for c in net.connections):
                continue
            for c in net.connections:
                if c.ref == ref or c.ref not in ctx.positions:
                    continue
                if c.ref in already_claimed:
                    continue
                if _is_small_passive(c.ref):
                    new_refs.add(c.ref)
                elif allow_small_ics and c.ref.startswith("U"):
                    comp = next(
                        (comp for comp in ctx.requirements.components
                         if comp.ref == c.ref), None,
                    )
                    if comp and len(comp.pins) <= max_ic_pins:
                        new_refs.add(c.ref)
    return new_refs


def _place_refs_in_column_grid(
    refs: list[str],
    start_x: float,
    start_y: float,
    ctx: PlacementContext,
    target_set: set[str],
    col_gap: float = 5.0,
    row_gap: float = 1.0,
    max_col_height: float = 15.0,
) -> int:
    """Place *refs* in a flowing column-grid layout.

    Returns the number of components placed.
    """
    bounds = ctx.bounds
    cur_x = start_x
    cur_y = start_y
    placed = 0
    for ref in refs:
        if ref not in ctx.positions:
            continue
        _raw_w, _raw_h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        w, h = _raw_w, _raw_h
        tx, ty = _clamp_to_bounds(cur_x, cur_y + h / 2.0, bounds)
        ctx.positions[ref] = (tx, ty, 0.0)
        target_set.add(ref)
        placed += 1
        cur_y = ty + h / 2.0 + row_gap
        if cur_y > start_y + max_col_height:
            cur_x += col_gap
            cur_y = start_y
    return placed


def _classify_refs_by_prefix(
    refs: set[str],
    ctx: PlacementContext,
    *prefixes: str,
) -> dict[str, list[str]]:
    """Partition *refs* into sorted lists keyed by first matching prefix."""
    result: dict[str, list[str]] = {p: [] for p in prefixes}
    for ref in sorted(refs):
        if ref not in ctx.positions:
            continue
        for p in prefixes:
            if ref.startswith(p):
                result[p].append(ref)
                break
    return result


def _push_component_outside_courtyard(
    ref: str,
    ctx: PlacementContext,
    court: tuple[float, float, float, float],
    grid: _PlacementGrid,
) -> None:
    """If *ref* is inside *court*, push it to the nearest edge."""
    rx, ry, rrot = ctx.positions[ref]
    w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
    cx1, cy1, cx2, cy2 = court
    if not (cx1 < rx < cx2 and cy1 < ry < cy2):
        return
    dx_left = rx - cx1 + w / 2.0
    dx_right = cx2 - rx + w / 2.0
    dy_top = ry - cy1 + h / 2.0
    dy_bot = cy2 - ry + h / 2.0
    min_d = min(dx_left, dx_right, dy_top, dy_bot)
    if min_d == dx_left:
        new_x, new_y = cx1 - w / 2.0 - 1.0, ry
    elif min_d == dx_right:
        new_x, new_y = cx2 + w / 2.0 + 1.0, ry
    elif min_d == dy_top:
        new_x, new_y = rx, cy1 - h / 2.0 - 1.0
    else:
        new_x, new_y = rx, cy2 + h / 2.0 + 1.0
    bounds = ctx.bounds
    new_x = _clamp(new_x, bounds[0] + w / 2.0 + 1.0, bounds[2] - w / 2.0 - 1.0)
    new_y = _clamp(new_y, bounds[1] + h / 2.0 + 1.0, bounds[3] - h / 2.0 - 1.0)
    px, py = grid.find_free_pos(new_x, new_y, w, h, max_radius=15.0)
    ctx.positions[ref] = (px, py, rrot)
    grid.place(px, py, w, h)
    _log.info("    %s pushed outside courtyard: (%.1f,%.1f)->(%.1f,%.1f)",
              ref, rx, ry, px, py)


def _place_component_with_grid(
    ref: str,
    tx: float,
    ty: float,
    ctx: PlacementContext,
    grid: _PlacementGrid,
    target_set: set[str],
    rot: float = 0.0,
    max_radius: float = 8.0,
    use_grid_search: bool = True,
) -> tuple[float, float]:
    """Place *ref* at/near (tx, ty), register in *grid* and *target_set*.

    Returns the final (px, py).
    """
    w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
    tx, ty = _clamp_to_bounds(tx, ty, ctx.bounds)
    if use_grid_search:
        px, py = grid.find_free_pos(tx, ty, w, h, max_radius=max_radius)
    else:
        px, py = tx, ty
    ctx.positions[ref] = (px, py, rot)
    grid.place(px, py, w, h)
    target_set.add(ref)
    return px, py


def _push_non_group_away_from_ic(
    ic_ref: str,
    ctx: PlacementContext,
    group_fixed: set[str],
    skip_refs: set[str],
) -> None:
    """Push non-group components that overlap *ic_ref* away horizontally."""
    if ic_ref not in ctx.positions:
        return
    ex, ey, erot = ctx.positions[ic_ref]
    ew, eh = ctx.fp_sizes.get(ic_ref, (5.0, 5.0))
    if erot % 180 in (90, 270):
        ew, eh = eh, ew
    for ref in list(ctx.positions):
        if ref in group_fixed or ref in ctx.fixed_refs:
            continue
        if ref in skip_refs:
            continue
        rx, ry, rrot = ctx.positions[ref]
        rw, rh = ctx.fp_sizes.get(ref, (1.5, 1.0))
        if rrot % 180 in (90, 270):
            rw, rh = rh, rw
        gx = abs(rx - ex) - (rw + ew) / 2.0
        gy = abs(ry - ey) - (rh + eh) / 2.0
        if gx < -0.1 and gy < -0.1:
            push_dist = -gx + 1.0
            if rx < ex:
                ctx.positions[ref] = (rx - push_dist, ry, rrot)
            else:
                ctx.positions[ref] = (rx + push_dist, ry, rrot)
            _log.info("    pushed %s away from %s by %.1fmm", ref, ic_ref, push_dist)


def _build_ref_net_adjacency(
    ctx: PlacementContext,
    group_refs: set[str],
) -> dict[str, set[str]]:
    """Build ref -> set of same-group refs sharing a net."""
    ref_nets: dict[str, set[str]] = {}
    for net in ctx.requirements.nets:
        net_refs = {c.ref for c in net.connections}
        for r in net_refs:
            if r in group_refs:
                ref_nets.setdefault(r, set()).update(net_refs & group_refs)
    return ref_nets


def _place_connector_at_edge(
    ref: str,
    tx: float,
    ty: float,
    ctx: PlacementContext,
    grid: _PlacementGrid,
    target_set: set[str],
    rot: float = 0.0,
    max_radius: float = 25.0,
) -> tuple[float, float]:
    """Place a connector *ref* near (tx, ty) with bounds clamping."""
    w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
    ty = _clamp(ty, ctx.bounds[1] + h / 2.0 + 1.0,
                ctx.bounds[3] - h / 2.0 - 1.0)
    tx = _clamp(tx, ctx.bounds[0] + w / 2.0 + 1.0,
                ctx.bounds[2] - w / 2.0 - 1.0)
    px, py = grid.find_free_pos(tx, ty, w, h, max_radius=max_radius)
    ctx.positions[ref] = (px, py, rot)
    grid.place(px, py, w, h)
    target_set.add(ref)
    return px, py


# ---------------------------------------------------------------------------
# MCU sub-step helpers
# ---------------------------------------------------------------------------

def _mcu_place_u3(
    ctx: PlacementContext,
    mcu_ref: str,
    mcu_w: float,
    mcu_h: float,
) -> tuple[float, float, float, float, float]:
    """Place the MCU IC (Step 0). Returns (mcu_x, mcu_y, eff_w, eff_h, rotation)."""
    bounds = ctx.bounds
    mcu_zone_rect = _find_zone_rect(ctx, "mcu")
    _mcu_rot = 180.0
    eff_w, eff_h = mcu_w, mcu_h

    mcu_fp = None
    for fp in ctx.initial_pcb.footprints:
        if fp.ref == mcu_ref:
            mcu_fp = fp
            break

    if mcu_fp is not None:
        from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space as _pad_ext
        _trial_ox = (bounds[0] + bounds[2]) / 2.0
        _trial_oy = (bounds[1] + bounds[3]) / 2.0
        _te = _pad_ext(mcu_fp, _trial_ox, _trial_oy, _mcu_rot)
        _pad_bot = _te[3] - _trial_oy
        mcu_origin_y = bounds[3] - _pad_bot - 2.0
        if mcu_zone_rect is not None:
            _zx1, _zy1, _zx2, _zy2 = mcu_zone_rect
            mcu_group_refs = _collect_feature_refs(ctx, "mcu", "controller", "processor")
            _j14_w = (ctx.fp_sizes.get("J14", _DEFAULT_J14_SIZE_MM)[0]
                      if "J14" in mcu_group_refs else 0.0)
            mcu_origin_x = (_zx1 + _zx2 - _j14_w) / 2.0
        else:
            mcu_origin_x = bounds[2] - eff_w / 2.0 - 10.0
        _pad_left = _te[0] - _trial_ox
        _pad_right = _te[2] - _trial_ox
        mcu_origin_x = _clamp(mcu_origin_x,
                               bounds[0] - _pad_left + 2.0,
                               bounds[2] - _pad_right - 2.0)
        mcu_x, mcu_y = origin_to_centroid(mcu_fp, mcu_origin_x,
                                           mcu_origin_y, _mcu_rot)
    else:
        if mcu_zone_rect is not None:
            _zx1, _zy1, _zx2, _zy2 = mcu_zone_rect
            mcu_x = (_zx1 + _zx2) / 2.0
        else:
            mcu_x = bounds[2] - eff_w / 2.0 - 10.0
        mcu_y = bounds[3] - eff_h / 2.0 - 5.0

    mcu_y = _clamp(mcu_y, bounds[1] + eff_h / 2.0 + 2.0,
                   bounds[3] - eff_h / 2.0 - 5.0)
    mcu_x = _clamp(mcu_x, bounds[0] + eff_w / 2.0 + 2.0,
                   bounds[2] - eff_w / 2.0 - 2.0)
    ctx.positions[mcu_ref] = (mcu_x, mcu_y, _mcu_rot)
    ctx.mcu_peripheral_refs.add(mcu_ref)
    ctx.fixed_refs.add(mcu_ref)
    _log.info("    U3 centroid at (%.1f, %.1f) rot=180 [eff_w=%.1f, eff_h=%.1f]",
              mcu_x, mcu_y, eff_w, eff_h)
    return mcu_x, mcu_y, eff_w, eff_h, _mcu_rot


def _mcu_place_decoupling(
    ctx: PlacementContext,
    decoupling_refs: list[str],
    mcu_x: float,
    mcu_y: float,
    mcu_left: float,
    grid: _PlacementGrid,
) -> None:
    """Place decoupling caps in a column left of the MCU (Step 3)."""
    bounds = ctx.bounds
    max_cap_w = max(
        (ctx.fp_sizes.get(r, (2.5, 1.5))[0] for r in decoupling_refs),
        default=2.5,
    )
    decoup_col_x = mcu_left - max_cap_w / 2.0 - 2.0
    total_cap_h = sum(
        ctx.fp_sizes.get(r, (1.5, 1.0))[1] + 1.5 for r in decoupling_refs
    )
    decoup_y = mcu_y - total_cap_h / 2.0
    for ref in decoupling_refs:
        w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
        tx, ty = _clamp_to_bounds(decoup_col_x, decoup_y + h / 2.0, bounds)
        px, py = grid.find_free_pos(tx, ty, w, h, max_radius=8.0)
        ctx.positions[ref] = (px, py, 0.0)
        ctx.mcu_peripheral_refs.add(ref)
        grid.place(px, py, w, h)
        decoup_y += h + 1.5
        dist_to_mcu = ((px - mcu_x) ** 2 + (py - mcu_y) ** 2) ** 0.5
        _log.info("    %s (decoupling): ->(%.1f,%.1f) [%.1fmm from U3]",
                  ref, px, py, dist_to_mcu)


def _mcu_place_named_connector(
    ref: str,
    tx: float,
    ty: float,
    ctx: PlacementContext,
    grid: _PlacementGrid,
    rot: float = 0.0,
    max_radius: float = 25.0,
) -> tuple[float, float]:
    """Place a named MCU connector and register it."""
    w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
    bounds = ctx.bounds
    ty = _clamp(ty, bounds[1] + h / 2.0 + 1.0, bounds[3] - h / 2.0 - 1.0)
    tx = _clamp(tx, bounds[0] + w / 2.0 + 1.0, bounds[2] - w / 2.0 - 1.0)
    px, py = grid.find_free_pos(tx, ty, w, h, max_radius=max_radius)
    ctx.positions[ref] = (px, py, rot)
    grid.place(px, py, w, h)
    ctx.mcu_peripheral_refs.add(ref)
    _log.info("    %s -> (%.1f, %.1f)", ref, px, py)
    return px, py


def _mcu_place_connectors(
    ctx: PlacementContext,
    connector_refs: set[str],
    grid: _PlacementGrid,
    mcu_x: float,
    mcu_y: float,
    mcu_left: float,
    mcu_top: float,
    eff_w: float,
) -> None:
    """Place MCU connectors at board edges (Step 4)."""
    bounds = ctx.bounds
    right_edge_x = bounds[2] - 2.0

    if "J14" in connector_refs and "J14" in ctx.positions and "J14" not in ctx.fixed_refs:
        w14, h14 = ctx.fp_sizes.get("J14", _DEFAULT_J14_SIZE_MM)
        _mcu_place_named_connector(
            "J14", bounds[2] - w14 / 2.0 - 1.0, mcu_y, ctx, grid,
        )

    if "J15" in connector_refs and "J15" in ctx.positions and "J15" not in ctx.fixed_refs:
        w15, h15 = ctx.fp_sizes.get("J15", (5.2, 12.9))
        j14_pos = ctx.positions.get("J14")
        if j14_pos:
            j14_bottom = j14_pos[1] + ctx.fp_sizes.get("J14", _DEFAULT_J14_SIZE_MM)[1] / 2.0
            tx = right_edge_x - w15 / 2.0
            ty = j14_bottom + h15 / 2.0 + 2.0
        else:
            tx = right_edge_x - w15 / 2.0
            ty = mcu_y + 10.0
        ty = _clamp(ty, bounds[1] + h15 / 2.0 + 1.5, bounds[3] - h15 / 2.0 - 1.5)
        tx = min(tx, bounds[2] - w15 / 2.0 - 1.5)
        px15, py15 = grid.find_free_pos(tx, ty, w15, h15, max_radius=25.0)
        px15 = min(px15, bounds[2] - w15 / 2.0 - 1.5)
        py15 = min(py15, bounds[3] - h15 / 2.0 - 1.5)
        ctx.positions["J15"] = (px15, py15, 0.0)
        grid.place(px15, py15, w15, h15)
        ctx.mcu_peripheral_refs.add("J15")
        _log.info("    J15 -> right edge, below J14 (%.1f, %.1f)", px15, py15)

    if "J16" in connector_refs and "J16" in ctx.positions and "J16" not in ctx.fixed_refs:
        w16, h16 = ctx.fp_sizes.get("J16", (16.2, 6.9))
        px = bounds[2] - w16 / 2.0
        py = mcu_top - h16 / 2.0 - 2.0
        py = _clamp(py, bounds[1] + h16 / 2.0 + 1.0, bounds[3] - h16 / 2.0 - 1.0)
        ctx.positions["J16"] = (px, py, 0.0)
        grid.place(px, py, w16, h16)
        ctx.mcu_peripheral_refs.add("J16")
        _log.info("    J16 -> right edge, above U3 (%.1f, %.1f)", px, py)

    if "J2" in connector_refs and "J2" in ctx.positions and "J2" not in ctx.fixed_refs:
        w2, h2 = ctx.fp_sizes.get("J2", _DEFAULT_J2_SIZE_MM)
        tx = mcu_left - w2 / 2.0 - 8.0
        ty = bounds[3] - h2 / 2.0 - 1.0
        tx = _clamp(tx, bounds[0] + w2 / 2.0 + 1.0, bounds[2] - w2 / 2.0 - 1.0)
        _mcu_place_named_connector("J2", tx, ty, ctx, grid, rot=180.0, max_radius=20.0)


def _mcu_place_led(
    ctx: PlacementContext,
    other_passive_refs: list[str],
    grid: _PlacementGrid,
    sw_base_x: float,
    sw_base_y: float,
) -> None:
    """Place status LED near switches (Step 6b)."""
    bounds = ctx.bounds
    for led_ref in ["LED1"]:
        if (led_ref in other_passive_refs and led_ref in ctx.positions
                and led_ref not in ctx.fixed_refs):
            lw, lh = ctx.fp_sizes.get(led_ref, (2.0, 1.0))
            led_tx, led_ty = _clamp_to_bounds(sw_base_x + 8.0, sw_base_y, bounds)
            lpx, lpy = grid.find_free_pos(led_tx, led_ty, lw, lh, max_radius=10.0)
            ctx.positions[led_ref] = (lpx, lpy, 0.0)
            ctx.mcu_peripheral_refs.add(led_ref)
            grid.place(lpx, lpy, lw, lh)
            if led_ref in other_passive_refs:
                other_passive_refs.remove(led_ref)
            _log.info("    %s (status LED) -> (%.1f, %.1f)", led_ref, lpx, lpy)


def _mcu_place_usb_subcircuit(
    ctx: PlacementContext,
    other_passive_refs: list[str],
    grid: _PlacementGrid,
) -> None:
    """Place USB ESD + series resistors near J2 (Step 5)."""
    j2_pos = ctx.positions.get("J2")
    bounds = ctx.bounds
    if "U9" in other_passive_refs and "U9" in ctx.positions and j2_pos:
        j2x, j2y, _j2r = j2_pos
        j2w, j2h = ctx.fp_sizes.get("J2", _DEFAULT_J2_SIZE_MM)
        u9w, u9h = ctx.fp_sizes.get("U9", (3.0, 3.0))
        u9_tx = j2x - j2w / 4.0
        u9_ty = j2y - j2h / 2.0 - u9h / 2.0 - 6.0
        u9_tx, u9_ty = _clamp_to_bounds(u9_tx, u9_ty, bounds)
        u9_tx = min(u9_tx, bounds[2] - u9w / 2.0 - 1.0)
        u9x, u9y = grid.find_free_pos(u9_tx, u9_ty, u9w, u9h, max_radius=8.0)
        ctx.positions["U9"] = (u9x, u9y, 0.0)
        ctx.mcu_peripheral_refs.add("U9")
        grid.place(u9x, u9y, u9w, u9h)
        other_passive_refs.remove("U9")
        _log.info("    U9 (ESD) -> near J2 at (%.1f, %.1f)", u9x, u9y)

    usb_r_refs = [r for r in ("R6", "R7") if r in other_passive_refs
                  and r in ctx.positions]
    if usb_r_refs and j2_pos:
        j2x_r, j2y_r, _ = j2_pos
        j2w_r = ctx.fp_sizes.get("J2", _DEFAULT_J2_SIZE_MM)[0]
        j2h_r = ctx.fp_sizes.get("J2", _DEFAULT_J2_SIZE_MM)[1]
        for i, ref in enumerate(usb_r_refs):
            w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
            px = j2x_r - j2w_r / 4.0 + i * (w + 2.0)
            py = j2y_r - j2h_r / 2.0 - h / 2.0 - 1.5
            px, py = _clamp_to_bounds(px, py, bounds)
            ctx.positions[ref] = (px, py, 0.0)
            grid.place(px, py, w, h)
            other_passive_refs.remove(ref)
            _log.info("    %s (USB R) -> (%.1f, %.1f)", ref, px, py)


def _mcu_place_reset_boot(
    ctx: PlacementContext,
    other_passive_refs: list[str],
    ref_nets: dict[str, set[str]],
    grid: _PlacementGrid,
    mcu_x: float,
    mcu_y: float,
    eff_w: float,
    mcu_top: float,
) -> None:
    """Place switch+resistor pairs for reset/boot (Step 6)."""
    bounds = ctx.bounds
    sw_pairs: list[tuple[str, str]] = []
    for sw, res in [("SW1", "R4"), ("SW2", "R5"), ("SW1", "R5"), ("SW2", "R4")]:
        if (sw in other_passive_refs and sw in ctx.positions
                and res in other_passive_refs and res in ctx.positions):
            if res in ref_nets.get(sw, set()):
                sw_pairs.append((sw, res))
    used_sw: set[str] = set()
    unique_pairs: list[tuple[str, str]] = []
    for sw, res in sw_pairs:
        if sw not in used_sw and res not in used_sw:
            unique_pairs.append((sw, res))
            used_sw.add(sw)
            used_sw.add(res)
    unpaired_sw = [r for r in other_passive_refs if r in ctx.positions
                   and r.startswith("SW") and r not in used_sw]

    mcu_left = mcu_x - eff_w / 2.0
    sw_base_x = mcu_left + eff_w / 4.0
    sw_base_y = mcu_top - 4.0

    for i, (sw, res) in enumerate(unique_pairs):
        sw_w, sw_h = ctx.fp_sizes.get(sw, (3.5, 3.5))
        r_w, r_h = ctx.fp_sizes.get(res, (1.0, 0.5))
        tx = sw_base_x - i * (sw_w + 1.5)
        ty = sw_base_y
        tx = _clamp(tx, bounds[0] + sw_w / 2.0 + 2.0, bounds[2] - sw_w / 2.0 - 2.0)
        ty = _clamp(ty, bounds[1] + sw_h / 2.0 + 2.0, bounds[3] - sw_h / 2.0 - 2.0)
        px, py = grid.find_free_pos(tx, ty, sw_w, sw_h, max_radius=10.0)
        px = _clamp(px, mcu_x - 20.0, mcu_x + 20.0)
        py = _clamp(py, max(mcu_y - 20.0, bounds[1] + sw_h / 2.0 + 2.0),
                    bounds[3] - sw_h / 2.0 - 2.0)
        ctx.positions[sw] = (px, py, 0.0)
        ctx.mcu_peripheral_refs.add(sw)
        grid.place(px, py, sw_w, sw_h)
        other_passive_refs.remove(sw)
        r_tx, r_ty = _clamp_to_bounds(px, py - sw_h / 2.0 - r_h / 2.0 - 0.5, bounds)
        rpx, rpy = grid.find_free_pos(r_tx, r_ty, r_w, r_h, max_radius=8.0)
        ctx.positions[res] = (rpx, rpy, 0.0)
        ctx.mcu_peripheral_refs.add(res)
        grid.place(rpx, rpy, r_w, r_h)
        other_passive_refs.remove(res)
        _log.info("    %s+%s (reset/boot) -> (%.1f,%.1f) / (%.1f,%.1f)",
                  sw, res, px, py, rpx, rpy)

    for sw in unpaired_sw:
        sw_w, sw_h = ctx.fp_sizes.get(sw, (3.5, 3.5))
        tx = sw_base_x - len(unique_pairs) * (sw_w + 3.0)
        ty = sw_base_y
        tx = _clamp(tx, bounds[0] + sw_w / 2.0 + 1.0, bounds[2] - 2.0)
        ty = _clamp(ty, bounds[1] + sw_h / 2.0 + 1.0, bounds[3] - 2.0)
        px, py = grid.find_free_pos(tx, ty, sw_w, sw_h, max_radius=15.0)
        ctx.positions[sw] = (px, py, 0.0)
        ctx.mcu_peripheral_refs.add(sw)
        grid.place(px, py, sw_w, sw_h)
        other_passive_refs.remove(sw)
        _log.info("    %s (switch) -> (%.1f, %.1f)", sw, px, py)

    return sw_base_x, sw_base_y  # type: ignore[return-value]


def _mcu_place_remaining(
    ctx: PlacementContext,
    remaining: list[str],
    mcu_ref: str,
    mcu_x: float,
    mcu_y: float,
    eff_w: float,
    eff_h: float,
    grid: _PlacementGrid,
) -> None:
    """Place remaining MCU passives in ring slots around the MCU (Step 7)."""
    bounds = ctx.bounds
    mcu_left = mcu_x - eff_w / 2.0
    mcu_right = mcu_x + eff_w / 2.0
    mcu_top = mcu_y - eff_h / 2.0
    _MCU_TARGET_GAP = 4.0
    _MCU_CLEAR = 3.0

    ring_slots: list[tuple[float, float]] = []
    if mcu_top > bounds[1] + 10.0:
        for dx_off in range(-4, 5):
            ring_slots.append((mcu_x + dx_off * 4.0,
                               mcu_top - _MCU_CLEAR - 3.0))
        for dx_off in range(-3, 4):
            ring_slots.append((mcu_x + dx_off * 4.0,
                               mcu_top - _MCU_CLEAR - 7.0))
    slot_x_left = mcu_left - _MCU_CLEAR - 2.0
    for dy_off in range(-3, 4):
        ring_slots.append((slot_x_left, mcu_y + dy_off * 3.5))
    slot_x_right = mcu_right + _MCU_CLEAR + 2.0
    for dy_off in range(-3, 4):
        ring_slots.append((slot_x_right, mcu_y + dy_off * 3.5))

    slot_idx = 0
    for ref in remaining:
        w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        rrot = ctx.positions[ref][2]

        placed = False
        while slot_idx < len(ring_slots):
            sx, sy = ring_slots[slot_idx]
            slot_idx += 1
            sx, sy = _clamp_to_bounds(sx, sy, bounds)
            if grid.is_free(sx, sy, w, h):
                ctx.positions[ref] = (sx, sy, rrot)
                ctx.mcu_peripheral_refs.add(ref)
                grid.place(sx, sy, w, h)
                placed = True
                break

        if not placed:
            tx = mcu_left - _MCU_TARGET_GAP - w / 2.0
            ty = mcu_y
            tx, ty = _clamp_to_bounds(tx, ty, bounds)
            px, py = grid.find_free_pos(tx, ty, w, h, max_radius=25.0)
            ctx.positions[ref] = (px, py, rrot)
            ctx.mcu_peripheral_refs.add(ref)
            grid.place(px, py, w, h)


# ---------------------------------------------------------------------------
# Ethernet sub-step helpers
# ---------------------------------------------------------------------------

def _nets_connected_to_prefix(
    net_refs: dict[str, set[str]], prefix: str,
) -> set[str]:
    """Return net names that have at least one ref starting with *prefix*."""
    return {n for n, refs in net_refs.items() if any(r.startswith(prefix) for r in refs)}


def _nets_connected_to_ref(
    net_refs: dict[str, set[str]], ref: str,
) -> set[str]:
    """Return net names that contain *ref*."""
    if not ref:
        return set()
    return {n for n, refs in net_refs.items() if ref in refs}


def _caps_on_nets(
    group_refs: set[str],
    net_refs: dict[str, set[str]],
    net_names: set[str],
    ctx: PlacementContext,
) -> list[str]:
    """Return sorted cap refs from *group_refs* on any of *net_names*."""
    return sorted([
        r for r in group_refs
        if r.startswith("C") and r in ctx.positions
        and any(r in net_refs.get(n, set()) for n in net_names)
    ])


def _classify_eth_caps(
    group_refs: set[str],
    net_refs: dict[str, set[str]],
    crystal_refs: list[str],
    poe_ic: str,
    ctx: PlacementContext,
) -> tuple[list[str], list[str], list[str]]:
    """Classify ethernet caps into crystal-load, PoE, and other.

    Returns (crystal_load_caps, poe_caps, other_caps).
    """
    crystal_net_names = _nets_connected_to_prefix(net_refs, "Y")
    crystal_load_caps = _caps_on_nets(group_refs, net_refs, crystal_net_names, ctx)
    poe_net_names = _nets_connected_to_ref(net_refs, poe_ic)
    poe_caps = sorted([
        r for r in group_refs
        if r.startswith("C") and r in ctx.positions
        and r not in crystal_load_caps
        and any(r in net_refs.get(n, set()) for n in poe_net_names)
    ])
    eth_decoupling = sorted([
        r for r in group_refs
        if r.startswith("C") and r in ctx.positions
        and r not in crystal_refs
    ])
    other_caps = sorted(
        [r for r in eth_decoupling
         if r not in crystal_load_caps and r not in poe_caps],
    )
    return crystal_load_caps, poe_caps, other_caps


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
    """Place crystal oscillator and its flanking load caps."""
    bounds = ctx.bounds
    ezx1, ezy1, ezx2, ezy2 = zone_rect
    y_crystal_h = ctx.fp_sizes.get(
        crystal_refs[0], _DEFAULT_CRYSTAL_SIZE_MM)[1] if crystal_refs else 1.5
    crystal_y = ic_cy - ic_h / 2.0 - y_crystal_h / 2.0 - 0.5
    crystal_x = ic_cx

    for ref in crystal_refs:
        if (ref not in ctx.positions or ref in ctx.fixed_refs
                or ref in placed_eth or ref == ""):
            continue
        w, h = ctx.fp_sizes.get(ref, _DEFAULT_CRYSTAL_SIZE_MM)
        tx, ty = _clamp_to_bounds(crystal_x, crystal_y, bounds)
        ctx.positions[ref] = (tx, ty, 0.0)
        eth_grid.place(tx, ty, w, h)
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        crystal_dist = math.sqrt((tx - ic_cx) ** 2 + (ty - ic_cy) ** 2)
        _log.info("    %s (crystal) -> (%.1f, %.1f) dist=%.1fmm from IC",
                  ref, tx, ty, crystal_dist)

    cap_idx = 0
    for ref in crystal_load_caps:
        if (ref not in ctx.positions or ref in ctx.fixed_refs
                or ref in placed_eth or ref == ""):
            continue
        w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
        y_w = ctx.fp_sizes.get(
            crystal_refs[0], _DEFAULT_CRYSTAL_SIZE_MM)[0] if crystal_refs else _DEFAULT_CRYSTAL_SIZE_MM[0]
        if cap_idx % 2 == 0:
            tx = crystal_x - y_w / 2.0 - w / 2.0 - 0.5
        else:
            tx = crystal_x + y_w / 2.0 + w / 2.0 + 0.5
        ty = crystal_y if crystal_refs else ic_cy - ic_h / 2.0 - 3.0
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
    """Place RJ45 connectors at the bottom board edge."""
    bounds = ctx.bounds
    ezx1, _ezy1, ezx2, _ezy2 = zone_rect
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space
    for ref in eth_connectors:
        if ref not in ctx.positions or ref in ctx.fixed_refs:
            continue
        fp_match = None
        for fp in ctx.initial_pcb.footprints:
            if fp.ref == ref:
                fp_match = fp
                break
        w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        cent_y = bounds[3] - h / 2.0 - 1.0
        cent_x = eth_anchor_x
        if fp_match is not None:
            trial_origin_x = eth_anchor_x
            trial_origin_y = bounds[3] - 3.0
            _, _, _, pad_max_y = pad_extent_in_board_space(
                fp_match, trial_origin_x, trial_origin_y, 180.0,
            )
            edge_margin = 2.0
            if pad_max_y > bounds[3] - edge_margin:
                trial_origin_y -= (pad_max_y - bounds[3] + edge_margin)
            cent_x, cent_y = origin_to_centroid(
                fp_match, trial_origin_x, trial_origin_y, 180.0,
            )
        px, py = cent_x, cent_y
        px = _clamp(px, ezx1 + w / 2.0, ezx2 - w / 2.0)
        ctx.positions[ref] = (px, py, 180.0)
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        _log.info("    %s (RJ45) -> bottom edge (%.1f, %.1f)", ref, px, py)


def _eth_fix_crystal_cap_overlaps(
    ctx: PlacementContext,
    placed_eth: set[str],
    zone_rect: tuple[float, float, float, float],
) -> None:
    """Shift caps that overlap crystals in the ethernet group."""
    _ezy1 = zone_rect[1]
    _ezy2 = zone_rect[3]
    crystal_placed = [r for r in placed_eth if r.startswith("Y")]
    for yref in crystal_placed:
        yx, yy, _yrot = ctx.positions[yref]
        yw, yh = ctx.fp_sizes.get(yref, _DEFAULT_CRYSTAL_SIZE_MM)
        for cref in list(placed_eth):
            if cref == yref or not cref.startswith("C"):
                continue
            cx, cy, crot = ctx.positions[cref]
            cw, ch = ctx.fp_sizes.get(cref, (1.5, 1.0))
            _margin = 2.0
            if (abs(cx - yx) < (cw + yw) / 2.0 + _margin
                    and abs(cy - yy) < (ch + yh) / 2.0 + _margin):
                new_cy = yy + yh / 2.0 + ch / 2.0 + 2.0
                new_cy = _clamp(new_cy, _ezy1 + 2.0, _ezy2 - 2.0)
                ctx.positions[cref] = (cx, new_cy, crot)
                _log.info("    3c4: shifted %s below %s (%.1f,%.1f) -> (%.1f,%.1f)",
                          cref, yref, cx, cy, cx, new_cy)


# ---------------------------------------------------------------------------
# ADC sub-step helpers
# ---------------------------------------------------------------------------

def _collect_analog_group_refs(ctx: PlacementContext) -> set[str]:
    """Return the FeatureBlock refs that contain ADC ICs."""
    for feat in ctx.requirements.features:
        feat_refs = set(feat.components)
        if feat_refs & ctx.adc_ic_refs:
            return feat_refs
    return set()


def _find_analog_outliers(
    ctx: PlacementContext,
    analog_group_refs: set[str],
) -> list[str]:
    """Return refs in the analog group that are far from the ADC cluster."""
    placed_analog = [
        ctx.positions[r] for r in (ctx.adc_channel_refs | ctx.adc_ic_refs)
        if r in ctx.positions
    ]
    if not placed_analog:
        return []
    cx = sum(p[0] for p in placed_analog) / len(placed_analog)
    cy = sum(p[1] for p in placed_analog) / len(placed_analog)
    outliers: list[str] = []
    for ref in sorted(analog_group_refs):
        if (ref in ctx.positions
                and ref not in ctx.adc_channel_refs
                and ref not in ctx.adc_ic_refs
                and ref not in ctx.fixed_refs
                and not ref.startswith("J")):
            rx, ry, _rrot = ctx.positions[ref]
            dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
            if dist > 15.0:
                outliers.append(ref)
    return outliers


def _adc_place_channel_strip(
    ch_idx: int,
    ic_pin: str,
    passives: list[str],
    ch_x_start: float,
    channel_spacing: float,
    ic_y: float,
    ic_h: float,
    strip_gap: float,
    ctx: PlacementContext,
) -> None:
    """Place one ADC channel's passives along a radial fan from ADC to connector.

    Learned from human-reference layout:
    - Components are placed along the vector from the ADC IC toward each
      channel's input connector.
    - C_filt (filter cap) closest to ADC (~5-7mm), R_top at medium distance
      (~9-14mm), R_bot offset toward board center, D_tvs along the path.
    - All passives rotated 0deg (matching human convention).
    - When no connector position is known, falls back to a vertical strip.

    The radial placement produces a fan-like layout where each channel's
    signal chain is visually distinct and follows the physical signal path
    from connector to ADC pin.
    """
    bounds = ctx.bounds
    r_refs = sorted([r for r in passives if r.startswith("R")])
    d_refs = [r for r in passives if r.startswith("D")]
    c_refs = [r for r in passives if r.startswith("C")]

    ch_x = ch_x_start + ch_idx * channel_spacing

    # --- Try radial fan placement ---
    # Find the connector for this channel via net connectivity
    connector_pos = _find_channel_connector_pos(passives, ctx)
    # Find ADC IC position (the IC this channel feeds into)
    adc_ic_pos = _find_adc_ic_pos_for_channel(passives, ctx)

    if connector_pos is not None and adc_ic_pos is not None:
        _adc_place_channel_radial(
            r_refs, d_refs, c_refs,
            adc_ic_pos, connector_pos, bounds, ctx,
        )
        return

    # --- Fallback: vertical strip (original behavior) ---
    strip_order: list[str] = []
    if len(r_refs) >= 2:
        strip_order.append(r_refs[1])
    strip_order.extend(d_refs)
    strip_order.extend(c_refs)
    if len(r_refs) >= 1:
        strip_order.append(r_refs[0])

    strip_y = ic_y - ic_h / 2.0 - 1.0
    for ref in strip_order:
        if ref not in ctx.positions or ref in ctx.fixed_refs:
            continue
        _raw_w, _raw_h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        h = _raw_h
        target_x, target_y = _clamp_to_bounds(ch_x, strip_y - h / 2.0, bounds)
        ctx.positions[ref] = (target_x, target_y, 0.0)
        ctx.adc_channel_refs.add(ref)
        strip_y = target_y - (h / 2.0 + strip_gap)


def _find_channel_connector_pos(
    passives: list[str],
    ctx: PlacementContext,
) -> tuple[float, float] | None:
    """Find the connector position for an ADC channel's passives.

    Walks nets from the channel's R refs to find connected J* connectors.
    """
    for net in ctx.requirements.nets:
        j_refs = [c.ref for c in net.connections if c.ref.startswith("J")]
        r_in_ch = [c.ref for c in net.connections if c.ref in passives and c.ref.startswith("R")]
        if j_refs and r_in_ch:
            for j_ref in j_refs:
                if j_ref in ctx.positions:
                    jx, jy, _ = ctx.positions[j_ref]
                    return (jx, jy)
    return None


def _find_adc_ic_pos_for_channel(
    passives: list[str],
    ctx: PlacementContext,
) -> tuple[float, float] | None:
    """Find the ADC IC position this channel connects to."""
    for net in ctx.requirements.nets:
        passive_in_ch = [c.ref for c in net.connections if c.ref in passives]
        ic_in_net = [c.ref for c in net.connections
                     if c.ref.startswith("U") and c.ref in ctx.positions]
        if passive_in_ch and ic_in_net:
            ic_ref = ic_in_net[0]
            ix, iy, _ = ctx.positions[ic_ref]
            return (ix, iy)
    return None


# Radial placement parameters learned from human reference ADC layout.
# (radius_mm, angle_offset_deg) per component type, keyed by connector
# quadrant relative to ADC IC.
_ADC_RADIAL_TOP: dict[str, tuple[float, float]] = {
    # Connector above ADC IC (top edge) — components fan down-left
    "C_filt": (6.0, -40.0),
    "R_top": (11.0, -20.0),
    "R_bot": (14.0, 20.0),
    "D_tvs": (17.0, 0.0),
}
_ADC_RADIAL_BOTTOM: dict[str, tuple[float, float]] = {
    # Connector below or same level as ADC IC — components route leftward
    "C_filt": (5.0, 170.0),
    "R_top": (9.0, -80.0),
    "R_bot": (12.0, 5.0),
    "D_tvs": (17.0, -50.0),
}


def _adc_place_channel_radial(
    r_refs: list[str],
    d_refs: list[str],
    c_refs: list[str],
    adc_pos: tuple[float, float],
    connector_pos: tuple[float, float],
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Place ADC channel passives in a radial fan from ADC IC toward connector.

    Learned from human reference board (60x40mm analog input board):
    - ADC IC at bottom, connectors at top edge
    - Each channel's passives interpolated along the IC-to-connector vector
    - Component type determines radius and angular offset from that vector
    - Connector quadrant (above/below IC) selects different offset tables
    """
    ic_x, ic_y = adc_pos
    jx, jy = connector_pos
    board_h = bounds[3] - bounds[1]

    # Base angle from ADC IC toward connector
    base_angle = math.atan2(jy - ic_y, jx - ic_x)

    # Select radial parameters based on connector quadrant
    params = _ADC_RADIAL_TOP if jy < (bounds[1] + board_h * 0.5) else _ADC_RADIAL_BOTTOM

    # Build ref-to-role mapping
    role_map: dict[str, str] = {}
    if len(r_refs) >= 1:
        role_map[r_refs[0]] = "R_top"    # lower-numbered R = top of divider
    if len(r_refs) >= 2:
        role_map[r_refs[1]] = "R_bot"
    for d in d_refs:
        role_map[d] = "D_tvs"
    for c in c_refs:
        role_map[c] = "C_filt"

    for ref, role in role_map.items():
        if ref not in ctx.positions or ref in ctx.fixed_refs:
            continue
        if role not in params:
            continue
        radius, angle_off_deg = params[role]
        angle = base_angle + math.radians(angle_off_deg)
        new_x = ic_x + radius * math.cos(angle)
        new_y = ic_y + radius * math.sin(angle)
        # Clamp inside board bounds
        new_x, new_y = _clamp_to_bounds(new_x, new_y, bounds)
        ctx.positions[ref] = (new_x, new_y, 0.0)
        ctx.adc_channel_refs.add(ref)


# ---------------------------------------------------------------------------
# Phase functions (refactored)
# ---------------------------------------------------------------------------

def _classify_power_columns(
    ctx: PlacementContext,
    group_refs: set[str],
    power_ics: list[str],
    power_connectors: list[str],
    net_refs: dict[str, set[str]],
    buck1_ic: str,
    buck2_ic: str,
) -> dict[str, list[str]]:
    """Classify power group refs into column lists for placement."""
    # Ferrite detection
    ferrite_refs = sorted(
        [r for r in group_refs
         if r.startswith("L") and r in ctx.positions
         and "ferrite" in (
             next((fp.value for fp in ctx.initial_pcb.footprints
                   if fp.ref == r), "")
         ).lower()],
    )
    ic_conn_set = set(power_ics) | set(power_connectors)

    # Buck #1 column
    vin_passives = sorted(net_refs.get("VIN", set()) - ic_conn_set)
    bst1_passives = sorted(
        (net_refs.get("BST", set()) | net_refs.get("SW", set()))
        - {buck1_ic} - set(power_connectors),
    )
    l1_refs = sorted(
        [r for r in group_refs
         if r.startswith("L") and r in ctx.positions
         and r not in ferrite_refs
         and r in net_refs.get("SW", set())],
    )
    fb_refs = sorted(net_refs.get("FB", set()) - {buck1_ic} - set(power_connectors))
    buck5v_caps = sorted(
        net_refs.get("BUCK_5V", set()) - {buck1_ic}
        - set(power_connectors) - set(fb_refs) - set(l1_refs),
    )

    vin_above = vin_passives
    output_left = bst1_passives + l1_refs + buck5v_caps
    output_right = fb_refs

    # Bridge
    or_diode_refs = sorted(
        net_refs.get("BUCK_5V", set())
        & {r for r in group_refs if r.startswith("D")}
        - set(vin_passives),
    )
    v5_rail_refs = sorted(
        (net_refs.get("+5V", set()) & group_refs)
        - ic_conn_set - set(or_diode_refs),
    )

    # Buck #2 column
    buck2_in_caps = sorted(
        net_refs.get("+5V", set())
        & {r for r in group_refs if r.startswith("C")}
        - {"C4"},
    )
    v5_rail_refs = [r for r in v5_rail_refs if r not in buck2_in_caps]
    bridge = or_diode_refs + v5_rail_refs
    bst2_passives = sorted(
        (net_refs.get("BST2", set()) | net_refs.get("SW2", set()))
        - {buck2_ic} - set(power_connectors),
    )
    l2_refs = sorted(
        [r for r in group_refs
         if r.startswith("L") and r in ctx.positions
         and r not in ferrite_refs
         and r in net_refs.get("SW2", set())],
    )
    v33_caps = sorted(
        net_refs.get("+3V3", set()) & group_refs
        - {buck2_ic} - set(power_connectors),
    )
    buck2_left = buck2_in_caps + [buck2_ic] + bst2_passives + l2_refs
    buck2_right = v33_caps

    # Tail
    led_refs = sorted(
        (net_refs.get("LED_A", set()) & group_refs) - set(power_connectors),
    )
    all_classified = (
        set(vin_above) | {buck1_ic}
        | set(output_left) | set(output_right)
        | set(bridge) | set(buck2_left) | set(buck2_right)
        | set(led_refs) | set(ferrite_refs) | set(power_connectors)
    )
    remaining = sorted(group_refs - all_classified - {""} - ctx.fixed_refs)
    tail = led_refs + ferrite_refs + remaining

    return {
        "vin_above": vin_above,
        "output_left": output_left,
        "output_right": output_right,
        "bridge": bridge,
        "buck2_left": buck2_left,
        "buck2_right": buck2_right,
        "tail": tail,
    }


# ---------------------------------------------------------------------------
# Power chain signal-flow phase (learned from human reference boards)
# ---------------------------------------------------------------------------

# Relative offsets (dx, dy, rotation) for passives around a BUCK IC anchor.
# Learned from human-routed TPS54331 layout on a 50x40mm board.
# Coordinates are relative to the buck IC centroid.
_BUCK_PASSIVE_OFFSETS: dict[str, tuple[float, float, float]] = {
    # (role, (dx, dy, rotation))  — role matched by net name keywords
    "input_cap": (-0.9, -5.3, 0.0),       # C on VIN rail, above IC
    "bootstrap_cap": (0.1, 4.8, 0.0),     # C on BST/BOOT net, below IC
    "inductor": (8.3, -1.2, 0.0),         # L on SW/PH net, right of IC
    "catch_diode": (7.8, 2.2, 180.0),     # D on SW/PH net, right-below IC
    "fb_top_r": (8.0, 6.8, 180.0),        # R on FB/VSNS net (top of divider)
    "fb_bot_r": (8.0, 4.3, 0.0),          # R on FB/VSNS net (bottom of divider)
    "output_cap": (12.6, 1.5, -90.0),     # C at output (past inductor)
}

# Relative offsets for passives around an LDO IC anchor.
# Learned from human-routed AMS1117-3.3 layout.
_LDO_PASSIVE_OFFSETS: dict[str, tuple[float, float, float]] = {
    "input_cap": (-17.4, -1.5, -90.0),    # C on VIN side, left of LDO
    "output_cap": (6.9, 1.1, -90.0),      # C on VOUT side, right of LDO
}

# Net-name keywords used to classify passive roles in power subcircuits.
_VIN_KEYWORDS = ("VIN", "+24V", "+12V", "VBUS", "V_IN")
_BST_KEYWORDS = ("BST", "BOOT", "BOOTSTRAP")
_SW_KEYWORDS = ("SW", "PH", "PHASE")
_FB_KEYWORDS = ("FB", "VSNS", "FEEDBACK", "SENSE")
_OUTPUT_KEYWORDS = ("+5V", "+3V3", "+3.3V", "+1V8", "VOUT", "V_OUT", "BUCK_5V")


def _classify_passive_role_power(
    ref: str,
    ctx: PlacementContext,
    ic_ref: str,
    subcircuit_refs: set[str],
    ic_input_voltage: float = 0.0,
    ic_output_voltage: float = 0.0,
) -> str | None:
    """Classify a passive's role relative to its power IC via net connectivity.

    Returns a role key matching ``_BUCK_PASSIVE_OFFSETS`` /
    ``_LDO_PASSIVE_OFFSETS``, or ``None`` if unclassified.

    Args:
        ic_input_voltage: Estimated input voltage of the regulator (for
            distinguishing input vs output caps when net names are ambiguous).
        ic_output_voltage: Estimated output voltage of the regulator.
    """
    if ref == ic_ref or ref not in ctx.positions:
        return None

    # Collect net names this passive shares with the IC
    shared_nets: list[str] = []
    all_nets_for_ref: list[str] = []
    for net in ctx.requirements.nets:
        conn_refs = {c.ref for c in net.connections}
        if ref in conn_refs:
            all_nets_for_ref.append(net.name.upper())
            if ic_ref in conn_refs:
                shared_nets.append(net.name.upper())

    # Classify by net keyword priority
    prefix = ref[0]

    if prefix == "L":
        for n in all_nets_for_ref:
            if any(kw in n for kw in _SW_KEYWORDS):
                return "inductor"
        return "inductor"  # inductors in power subcircuit are almost always the main inductor

    if prefix == "D":
        for n in all_nets_for_ref:
            if any(kw in n for kw in _SW_KEYWORDS):
                return "catch_diode"
        return "catch_diode"

    if prefix == "C":
        # Bootstrap cap: on BST net
        for n in all_nets_for_ref:
            if any(kw in n for kw in _BST_KEYWORDS):
                return "bootstrap_cap"
        # Input cap: on VIN net shared with IC
        for n in shared_nets:
            if any(kw in n for kw in _VIN_KEYWORDS):
                return "input_cap"
        # For regulators with known voltages, use voltage magnitude to
        # distinguish input vs output caps.  Higher voltage net = input.
        if ic_input_voltage > 0 and ic_output_voltage > 0:
            cap_voltage = _estimate_net_voltage(all_nets_for_ref)
            if cap_voltage is not None:
                # If cap voltage is closer to input voltage, it's input cap
                in_diff = abs(cap_voltage - ic_input_voltage)
                out_diff = abs(cap_voltage - ic_output_voltage)
                if in_diff < out_diff:
                    return "input_cap"
                return "output_cap"
        # Output cap: on output net or not sharing any net with IC directly
        for n in all_nets_for_ref:
            if any(kw in n for kw in _OUTPUT_KEYWORDS):
                return "output_cap"
        # Fallback: if shared net with IC, likely input; otherwise output
        return "input_cap" if shared_nets else "output_cap"

    if prefix == "R":
        for n in all_nets_for_ref:
            if any(kw in n for kw in _FB_KEYWORDS):
                # Determine top vs bottom: top R connects to output rail,
                # bottom R connects to GND
                for n2 in all_nets_for_ref:
                    if n2 in ("GND", "AGND", "DGND", "PGND"):
                        return "fb_bot_r"
                return "fb_top_r"
        return None

    return None


def _estimate_net_voltage(net_names: list[str]) -> float | None:
    """Estimate voltage from net name keywords.

    Returns the voltage in volts, or None if no voltage keyword found.
    """
    for n in net_names:
        # Try common voltage patterns
        for prefix, voltage in (
            ("+24V", 24.0), ("+12V", 12.0), ("+9V", 9.0),
            ("+5V", 5.0), ("+3V3", 3.3), ("+3.3V", 3.3),
            ("+1V8", 1.8), ("+1.8V", 1.8), ("+2V5", 2.5),
        ):
            if prefix in n:
                return voltage
    return None


def _estimate_regulator_voltages(
    ic_ref: str,
    ctx: PlacementContext,
) -> tuple[float, float]:
    """Estimate input and output voltages for a regulator IC from net names.

    Returns (input_voltage, output_voltage).  Both 0.0 if unknown.
    """
    voltages: list[float] = []
    for net in ctx.requirements.nets:
        if not any(c.ref == ic_ref for c in net.connections):
            continue
        if net.name.upper() in ("GND", "AGND", "DGND", "PGND"):
            continue
        v = _estimate_net_voltage([net.name.upper()])
        if v is not None:
            voltages.append(v)
    if len(voltages) >= 2:
        return (max(voltages), min(voltages))
    if len(voltages) == 1:
        return (voltages[0], 0.0)
    return (0.0, 0.0)


def _phase_power_chain_flow(ctx: PlacementContext) -> None:
    """3c1b: Power chain signal-flow ordering.

    Enforces left-to-right signal flow for power conversion chains:
    input connector -> buck IC -> inductor -> output cap -> LDO -> output.

    Learned from human reference boards:
    - Buck IC rotated -90deg, placed at ~17% board width
    - LDO at ~82% board width, rotated 0deg
    - Passives placed at fixed offsets from their parent IC
    - Caps on output side rotated -90deg (vertical, matching horizontal flow)

    This phase runs after ``_phase_power_group`` and applies signal-flow
    corrections to power subcircuit components that were column-placed.
    """
    from kicad_pipeline.optimization.functional_grouper import (
        SubCircuitType,
        compute_power_flow_topology,
    )

    _log.info("  3c1b: Power chain signal-flow ordering")

    # Collect power regulator subcircuits
    buck_scs = [sc for sc in ctx.subcircuits
                if sc.circuit_type == SubCircuitType.BUCK_CONVERTER]
    ldo_scs = [sc for sc in ctx.subcircuits
               if sc.circuit_type == SubCircuitType.LDO_REGULATOR]

    if not (buck_scs or ldo_scs):
        _log.info("    No power regulators found — skipping signal-flow phase")
        return

    # Get power zone bounds.  If the zone is too small for the learned
    # offsets (~25mm width for buck+LDO chain), expand to board bounds.
    min_power_zone_w = 25.0
    power_zone_rect = _find_zone_rect(ctx, "power")
    if power_zone_rect is None:
        power_zone_rect = ctx.bounds
    zx1, zy1, zx2, zy2 = power_zone_rect
    zone_w = zx2 - zx1
    zone_h = zy2 - zy1
    if zone_w < min_power_zone_w and len(buck_scs) + len(ldo_scs) >= 2:
        _log.info("    3c1b: power zone too narrow (%.1fmm) — expanding to board bounds",
                  zone_w)
        zx1, zy1, zx2, zy2 = ctx.bounds
        zone_w = zx2 - zx1
        zone_h = zy2 - zy1

    # Order regulators by power flow topology (highest voltage -> lowest)
    all_reg_scs = buck_scs + ldo_scs
    topology = compute_power_flow_topology(tuple(ctx.subcircuits))

    def _regulator_flow_order(sc: object) -> float:
        """Return flow-order index: earlier in chain = lower index = further left.

        Falls back to input voltage magnitude (higher voltage = earlier in chain).
        """
        if sc.input_domain and sc.input_domain in topology.domain_order:
            return float(list(topology.domain_order).index(sc.input_domain))
        # Fallback: estimate input voltage from net names connected to IC
        max_v = 0.0
        for net in ctx.requirements.nets:
            if not any(c.ref == sc.anchor_ref for c in net.connections):
                continue
            upper = net.name.upper()
            for prefix in ("+24V", "+12V", "+5V", "+3V3", "+3.3V", "+1V8"):
                if prefix in upper:
                    try:
                        v = float(prefix.replace("+", "").replace("V", ".").rstrip("."))
                    except ValueError:
                        continue
                    max_v = max(max_v, v)
            if "VIN" in upper or "V_IN" in upper:
                max_v = max(max_v, 100.0)  # VIN is usually the highest
        # Higher voltage = lower sort key = placed further left
        return -max_v if max_v > 0 else 999.0

    all_reg_scs.sort(key=_regulator_flow_order)

    if not all_reg_scs:
        return

    # Compute horizontal positions: spread regulators left-to-right across zone
    n_regs = len(all_reg_scs)
    # For 1 regulator: center at 30% zone width (leaving room for output passives)
    # For 2 regulators: 17% and 82% (learned from human reference)
    # For N regulators: evenly spread from 15% to 85%
    if n_regs == 1:
        x_fracs = [0.30]
    elif n_regs == 2:
        x_fracs = [0.17, 0.82]
    else:
        x_fracs = [0.15 + 0.70 * i / (n_regs - 1) for i in range(n_regs)]

    for reg_idx, sc in enumerate(all_reg_scs):
        ic_ref = sc.anchor_ref
        if ic_ref not in ctx.positions:
            continue

        is_buck = sc.circuit_type == SubCircuitType.BUCK_CONVERTER
        offsets = _BUCK_PASSIVE_OFFSETS if is_buck else _LDO_PASSIVE_OFFSETS

        # Place IC at signal-flow position
        ic_x = zx1 + zone_w * x_fracs[reg_idx]
        # Force Y to learned zone fractions (from human reference):
        #   Buck IC at ~40% zone height, LDO at ~47% zone height.
        # Previous code kept old_y which was set by _phase_power_group
        # near the zone bottom — causing ~20mm drift from reference.
        y_frac = 0.40 if is_buck else 0.47
        ic_y = zy1 + zone_h * y_frac
        ic_y = _clamp(ic_y, zy1 + 3.0, zy2 - 3.0)
        # Buck ICs are rotated -90deg for signal flow; LDOs stay at 0
        ic_rot = -90.0 if is_buck else 0.0

        ctx.positions[ic_ref] = (ic_x, ic_y, ic_rot)
        ctx.power_group_fixed.add(ic_ref)

        # Gather ALL passives connected to this IC via nets (the subcircuit
        # detector often only captures a subset).  Include power group refs
        # that share a non-GND net with the IC.
        power_group_refs = _collect_feature_refs(ctx, "power", "supply")
        ic_net_refs: set[str] = set(sc.refs)
        for net in ctx.requirements.nets:
            conn_refs = {c.ref for c in net.connections}
            if ic_ref not in conn_refs:
                continue
            # Skip pure GND nets — they connect everything
            if net.name.upper() in ("GND", "AGND", "DGND", "PGND"):
                continue
            for c in net.connections:
                if (c.ref != ic_ref
                        and c.ref in ctx.positions
                        and c.ref[0] in "RCLDF"
                        and (c.ref in power_group_refs or c.ref in sc.refs)):
                    ic_net_refs.add(c.ref)
        # Also look for passives 1 hop away (e.g. FB divider bottom R
        # connects to GND, not to IC directly, but top R connects to IC)
        for net in ctx.requirements.nets:
            if net.name.upper() in ("GND", "AGND", "DGND", "PGND"):
                continue
            conn_refs = {c.ref for c in net.connections}
            # If any ref in ic_net_refs is in this net, grab other small
            # passives from the same power group
            if conn_refs & ic_net_refs:
                for c in net.connections:
                    if (c.ref not in ic_net_refs
                            and c.ref != ic_ref
                            and c.ref in ctx.positions
                            and c.ref[0] in "RCLDF"
                            and c.ref in power_group_refs):
                        ic_net_refs.add(c.ref)

        # Estimate input/output voltages to disambiguate input vs output caps
        in_v, out_v = _estimate_regulator_voltages(ic_ref, ctx)

        # Place passives at learned offsets from IC
        placed_roles: set[str] = set()
        for ref in sorted(ic_net_refs):
            if ref == ic_ref:
                continue
            role = _classify_passive_role_power(
                ref, ctx, ic_ref, ic_net_refs, in_v, out_v,
            )
            if role is None or role not in offsets:
                continue
            # Only place one component per role (first match wins)
            if role in placed_roles:
                continue
            placed_roles.add(role)
            dx, dy, rot = offsets[role]
            px = ic_x + dx
            py = ic_y + dy
            # Clamp inside zone
            px = _clamp(px, zx1 + 1.0, zx2 - 1.0)
            py = _clamp(py, zy1 + 1.0, zy2 - 1.0)
            ctx.positions[ref] = (px, py, rot)
            ctx.power_group_fixed.add(ref)

        _log.info(
            "    3c1b: placed %s (%s) at (%.1f, %.1f, %.0f) with %d passives"
            " (roles: %s)",
            ic_ref,
            "buck" if is_buck else "ldo",
            ic_x, ic_y, ic_rot,
            len(placed_roles),
            ", ".join(sorted(placed_roles)),
        )

    # Place connectors at learned positions from human reference.
    # Only apply on dedicated power boards (few feature blocks, few
    # connectors).  On larger multi-group boards, connectors are
    # handled by _phase_top_edge_connectors and _phase_connector_orientation.
    power_group_refs = _collect_feature_refs(ctx, "power", "supply")
    power_connectors = sorted(
        r for r in power_group_refs
        if r.startswith("J") and r in ctx.positions
    )
    is_power_focused_board = (
        len(ctx.requirements.features) <= 2
        and len(power_connectors) <= 4
    )
    if power_connectors and all_reg_scs and is_power_focused_board:
        first_ic_x = zx1 + zone_w * x_fracs[0]
        first_ic_y = zy1 + zone_h * 0.40  # buck IC Y

        # Connector placement rules learned from human reference:
        #   J1 (input): near top of zone, X = first_ic_x + 7.7, y = 15%
        #   J2 (mid TP): 61% zone width, y = IC_y + 1.7, rot=-90
        #   J3 (output TP): 88% zone width, 73% height, rot=-90
        conn_rules: dict[int, tuple[float, float, float, float, float]] = {
            # index -> (x_frac_or_offset, y_frac, rotation, is_relative_to_ic, ic_dx)
            0: (0.0, 0.15, 0.0, 1.0, 7.7),
            1: (0.61, 0.0, -90.0, 0.0, 0.0),
            2: (0.88, 0.73, -90.0, 0.0, 0.0),
        }
        for idx, j_ref in enumerate(power_connectors):
            if j_ref in ctx.fixed_refs:
                continue
            rule = conn_rules.get(idx)
            if rule is None:
                continue
            x_frac, y_frac, rot, is_rel, ic_dx = rule
            if is_rel > 0.5:
                jx = first_ic_x + ic_dx
                jy = zy1 + zone_h * y_frac
            else:
                jx = zx1 + zone_w * x_frac
                jy = (zy1 + zone_h * y_frac
                      if y_frac > 0.01 else first_ic_y + 1.7)
            jx = _clamp(jx, zx1 + 1.0, zx2 - 1.0)
            jy = _clamp(jy, zy1 + 1.0, zy2 - 1.0)
            ctx.positions[j_ref] = (jx, jy, rot)
            ctx.power_group_fixed.add(j_ref)
            # Also add to fixed_refs so _phase_top_edge_connectors
            # does not override the power-chain connector positions.
            ctx.fixed_refs.add(j_ref)

    _log.info(
        "    3c1b: signal-flow ordered %d regulators across power zone",
        n_regs,
    )


def _phase_power_group(ctx: PlacementContext) -> None:
    """3c1: Power group organization — IC-anchored fork/branch layout."""
    _log.info("  3c1: Power group organization")
    bounds = ctx.bounds
    _min_x, _min_y, max_x, max_y = bounds

    power_group_refs = _collect_feature_refs(ctx, "power", "supply")
    if not power_group_refs:
        return

    power_zone_rect = _find_zone_rect(ctx, "power")
    net_to_pwr_refs = _build_net_to_group_refs(ctx, power_group_refs)

    by_prefix = _classify_refs_by_prefix(power_group_refs, ctx, "U", "J")
    power_ics = by_prefix["U"]
    power_connectors = by_prefix["J"]

    if not (power_ics and power_zone_rect is not None):
        return

    zx1, zy1, zx2, zy2 = power_zone_rect

    _STRIP_GAP = 0.5
    _COL_SPACING = 8.0
    _IC_MARGIN = 3.0

    placed_in_col: set[str] = set()

    buck1_ic = power_ics[0] if power_ics else ""
    buck2_ic = power_ics[1] if len(power_ics) > 1 else ""

    columns = _classify_power_columns(
        ctx, power_group_refs, power_ics, power_connectors,
        net_to_pwr_refs, buck1_ic, buck2_ic,
    )
    vin_above = columns["vin_above"]
    output_left = columns["output_left"]
    output_right = columns["output_right"]
    bridge_column = columns["bridge"]
    buck2_left = columns["buck2_left"]
    buck2_right = columns["buck2_right"]
    tail_column = columns["tail"]

    # --- IC-anchored placement ---
    u1_x, u1_y, _u1_rot = ctx.positions.get(
        buck1_ic, (zx1 + 5.0, zy1 + 15.0, 0.0),
    )
    u1_w, u1_h = ctx.fp_sizes.get(buck1_ic, (5.0, 5.0))
    zone_quarter_x = zx1 + (zx2 - zx1) * 0.25
    anchor_x = max(
        zx1 + 6.0,
        min(zx2 - _COL_SPACING - 3.0, zone_quarter_x),
    )

    # Build occupancy grid of non-power components
    _pwr_grid = _build_exclusion_grid(ctx, power_group_refs)

    _pz_x1 = zx1 + 2.0
    _pz_y1 = zy1 + 2.0
    _pz_x2 = zx2 - 2.0
    _pz_y2 = max_y - 3.0

    def _place_column(
        refs: list[str],
        col_x: float,
        start_y: float,
    ) -> float:
        """Place refs in a vertical column. Returns bottom Y."""
        cy = start_y
        for ref in refs:
            if (ref not in ctx.positions or ref in ctx.fixed_refs
                    or ref in placed_in_col or ref == ""):
                continue
            w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
            tx = max(_pz_x1, min(_pz_x2, col_x))
            ty = max(_pz_y1, min(_pz_y2, cy + h / 2.0))
            px, py = tx, ty
            ctx.positions[ref] = (px, py, 0.0)
            _pwr_grid.place(px, py, w, h)
            ctx.power_group_fixed.add(ref)
            placed_in_col.add(ref)
            cy = py + h / 2.0 + _STRIP_GAP
        return cy

    def _place_column_upward(
        refs: list[str],
        col_x: float,
        start_y: float,
    ) -> float:
        """Place refs in a vertical column going UP. Returns top Y."""
        cy = start_y
        for ref in refs:
            if (ref not in ctx.positions or ref in ctx.fixed_refs
                    or ref in placed_in_col or ref == ""):
                continue
            w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
            tx = max(_pz_x1, min(_pz_x2, col_x))
            ty = max(_pz_y1, min(_pz_y2, cy - h / 2.0))
            px, py = tx, ty
            ctx.positions[ref] = (px, py, 0.0)
            _pwr_grid.place(px, py, w, h)
            ctx.power_group_fixed.add(ref)
            placed_in_col.add(ref)
            cy = py - h / 2.0 - _STRIP_GAP
        return cy

    # Register connectors (don't move them)
    for ref in power_connectors:
        ctx.power_group_fixed.add(ref)

    # Place U1 at its anchor position
    _SUB_COL_OFFSET = 3.5
    if buck1_ic and buck1_ic not in ctx.fixed_refs:
        px = max(_pz_x1, min(_pz_x2, anchor_x))
        py = max(_pz_y1, min(_pz_y2, u1_y))
        ctx.positions[buck1_ic] = (px, py, 0.0)
        _pwr_grid.place(px, py, u1_w, u1_h)
        ctx.power_group_fixed.add(buck1_ic)
        placed_in_col.add(buck1_ic)
        u1_x, u1_y = px, py

    # VIN passives ABOVE U1
    vin_top = u1_y - u1_h / 2.0 - _STRIP_GAP
    _place_column_upward(vin_above, anchor_x, vin_top)

    # Output passives in 2-column grid BELOW U1
    output_top = u1_y + u1_h / 2.0 + _STRIP_GAP
    left_x = anchor_x - _SUB_COL_OFFSET
    right_x = anchor_x + _SUB_COL_OFFSET
    left_bottom = _place_column(output_left, left_x, output_top)
    right_bottom = _place_column(output_right, right_x, output_top)

    # Bridge
    bridge_top = max(left_bottom, right_bottom)
    bridge_left = bridge_column[:len(bridge_column) // 2 + 1]
    bridge_right = bridge_column[len(bridge_column) // 2 + 1:]
    fork_y_l = _place_column(bridge_left, left_x, bridge_top)
    fork_y_r = _place_column(bridge_right, right_x, bridge_top)
    fork_y = max(fork_y_l, fork_y_r)

    # Column 2: Buck #2
    col2_x = anchor_x + _COL_SPACING
    col2_top = output_top
    col2_left_bottom = _place_column(buck2_left, col2_x, col2_top)
    _b2l_max_w = max(
        (ctx.fp_sizes.get(r, (2.0, 2.0))[0] for r in buck2_left if r in ctx.fp_sizes),
        default=3.0,
    )
    _b2r_max_w = max(
        (ctx.fp_sizes.get(r, (2.0, 2.0))[0] for r in buck2_right if r in ctx.fp_sizes),
        default=2.0,
    )
    col2_right_x = col2_x + (_b2l_max_w + _b2r_max_w) / 2.0 + 0.5
    _place_column(buck2_right, col2_right_x, col2_top)
    col2_bottom = col2_left_bottom

    # Tail
    tail_y = max(fork_y, col2_bottom)
    tail_left = tail_column[:len(tail_column) // 2 + 1]
    tail_right = tail_column[len(tail_column) // 2 + 1:]
    _place_column(tail_left, left_x, tail_y)
    _place_column(tail_right, right_x, tail_y)

    _log.info(
        "    3c1: organized %d power components anchored at %s (%.1f, %.1f)",
        len(ctx.power_group_fixed), buck1_ic, u1_x, u1_y,
    )


def _detect_adc_channels(
    ctx: PlacementContext,
) -> list[tuple[str, str, list[str]]]:
    """Detect ADC channels using functional_grouper subcircuits + net tracing.

    Uses the pre-detected ADC_CHANNEL subcircuits from ctx.subcircuits as the
    primary source.  For each subcircuit, finds the ADC IC by tracing nets
    from the channel's passive components to a U* IC.  Returns a list of
    (ic_ref, ic_pin, passives) tuples compatible with the rest of the phase.

    Falls back to the legacy single-net heuristic (1 IC + 2R + 1D + 1C on
    one net) when no subcircuit data is available.
    """
    from kicad_pipeline.optimization.functional_grouper import SubCircuitType

    channels: list[tuple[str, str, list[str]]] = []

    # --- Primary: use pre-detected ADC_CHANNEL subcircuits ---
    adc_scs = [sc for sc in ctx.subcircuits
               if sc.circuit_type == SubCircuitType.ADC_CHANNEL]

    if adc_scs:
        for sc in adc_scs:
            passives = [r for r in sc.refs
                        if r[0] in "RDC" and r in ctx.positions
                        and not r.startswith("U")]
            # Find the ADC IC by tracing nets from passives
            ic_ref: str | None = None
            ic_pin: str = ""
            for net in ctx.requirements.nets:
                net_refs = {c.ref for c in net.connections}
                passive_in_ch = net_refs & set(passives)
                ics_in_net = [c for c in net.connections
                              if c.ref.startswith("U") and c.ref in ctx.positions]
                if passive_in_ch and ics_in_net:
                    ic_ref = ics_in_net[0].ref
                    ic_pin = ics_in_net[0].pin
                    break
            if ic_ref:
                channels.append((ic_ref, ic_pin, passives))
        return channels

    # --- Fallback: legacy single-net heuristic (1 IC + 2R + 1D + 1C) ---
    net_components: dict[str, list[tuple[str, str]]] = {}
    for net in ctx.requirements.nets:
        net_components[net.name] = [(c.ref, c.pin) for c in net.connections]

    for _net_name, conns in net_components.items():
        ic_refs = [(r, p) for r, p in conns if r.startswith("U") and r in ctx.positions]
        passive_refs = [r for r, p in conns
                        if r in ctx.positions and r[0] in "RDC" and not r.startswith("U")]
        if len(ic_refs) != 1 or len(passive_refs) != 4:
            continue
        r_count = sum(1 for r in passive_refs if r.startswith("R"))
        d_count = sum(1 for r in passive_refs if r.startswith("D"))
        c_count = sum(1 for r in passive_refs if r.startswith("C"))
        if r_count == 2 and d_count == 1 and c_count == 1:
            found_ic_ref, found_ic_pin = ic_refs[0]
            channels.append((found_ic_ref, found_ic_pin, passive_refs))
    return channels


def _build_r_top_connector_x(ctx: PlacementContext) -> dict[str, float]:
    """Map R-top refs to their connector's X position."""
    mapping: dict[str, float] = {}
    for net in ctx.requirements.nets:
        j_refs = [c for c in net.connections if c.ref.startswith("J")]
        r_refs = [c for c in net.connections
                  if c.ref.startswith("R") and c.ref in ctx.positions]
        if not (j_refs and r_refs):
            continue
        for j_conn in j_refs:
            j_pos = ctx.positions.get(j_conn.ref)
            if j_pos:
                for r_conn in r_refs:
                    mapping[r_conn.ref] = j_pos[0]
    return mapping


def _move_adc_ics_to_zone(
    ctx: PlacementContext,
    ic_channels: dict[str, list[tuple[str, list[str]]]],
    r_top_connector_x: dict[str, float],
) -> None:
    """Move ADC ICs into the analog zone, spaced evenly."""
    analog_zone_rect = _find_zone_rect(ctx, "analog")
    if not (analog_zone_rect and ctx.adc_ic_refs):
        return

    def _ic_avg_x(ic: str) -> float:
        xs = [r_top_connector_x[r]
              for _pin, passives in ic_channels.get(ic, [])
              for r in passives
              if r.startswith("R") and r in r_top_connector_x]
        return sum(xs) / len(xs) if xs else 999.0

    az_x1, az_y1, az_x2, az_y2 = analog_zone_rect
    ic_list = sorted(ctx.adc_ic_refs, key=_ic_avg_x)
    ic_spacing = (az_x2 - az_x1) / (len(ic_list) + 1)
    for idx, ic_ref in enumerate(ic_list):
        if ic_ref not in ctx.positions:
            continue
        new_x = az_x1 + ic_spacing * (idx + 1)
        new_y = az_y1 + (az_y2 - az_y1) * 0.70
        ctx.positions[ic_ref] = (new_x, new_y, 0.0)
        ctx.fixed_refs.add(ic_ref)
        _log.info("    3c2: moved %s to analog zone (%.1f, %.1f)",
                  ic_ref, new_x, new_y)


def _shift_channel_start_for_overlaps(
    ch_x_start: float,
    total_ch_width: float,
    channel_spacing: float,
    occupied_ranges: list[tuple[float, float]],
) -> tuple[float, float]:
    """Shift channel X range to avoid overlapping occupied ranges."""
    ch_x_end = ch_x_start + total_ch_width
    _comp_half_w = 2.0
    for ox_min, ox_max in occupied_ranges:
        if (ch_x_start - _comp_half_w < ox_max + _comp_half_w
                and ch_x_end + _comp_half_w > ox_min - _comp_half_w):
            shift = (ox_max + _comp_half_w + channel_spacing) - ch_x_start
            if shift > 0:
                ch_x_start += shift
                ch_x_end = ch_x_start + total_ch_width
    return ch_x_start, ch_x_end


def _phase_adc_channels(ctx: PlacementContext) -> None:
    """3c2: ADC channel formation — repeatable channel strips near ADC ICs."""
    _log.info("  3c2: ADC channel formation")

    adc_channels = _detect_adc_channels(ctx)

    # Group channels by IC
    ic_channels: dict[str, list[tuple[str, list[str]]]] = {}
    for ic_ref, ic_pin, passives in adc_channels:
        ic_channels.setdefault(ic_ref, []).append((ic_pin, passives))

    _r_top_connector_x = _build_r_top_connector_x(ctx)

    def _channel_sort_key(ch: tuple[str, list[str]]) -> float:
        _pin, _passives = ch
        for r in sorted(r for r in _passives if r.startswith("R")):
            if r in _r_top_connector_x:
                return _r_top_connector_x[r]
        return float(hash(_pin)) * 1e-6

    for ic_ref in ic_channels:
        ic_channels[ic_ref].sort(key=_channel_sort_key)

    # Collect all ADC channel passive refs and IC refs
    all_adc_passive_refs: set[str] = set()
    for _ic_ref, _ic_pin, passives in adc_channels:
        all_adc_passive_refs.update(passives)
        ctx.adc_ic_refs.add(_ic_ref)

    # Move ADC ICs to the analog zone
    _move_adc_ics_to_zone(ctx, ic_channels, _r_top_connector_x)

    def _ic_avg_connector_x(ic: str) -> float:
        xs = [_r_top_connector_x[r]
              for _pin, passives in ic_channels.get(ic, [])
              for r in passives
              if r.startswith("R") and r in _r_top_connector_x]
        return sum(xs) / len(xs) if xs else 999.0

    _adc_grid = _build_exclusion_grid(ctx, all_adc_passive_refs)

    _CHANNEL_SPACING_MM = 8.0
    _STRIP_GAP_MM = 1.5

    sorted_ic_refs = sorted(ic_channels.keys(), key=_ic_avg_connector_x)
    _occupied_x_ranges: list[tuple[float, float]] = []

    for ic_ref in sorted_ic_refs:
        ch_list = ic_channels[ic_ref]
        if ic_ref not in ctx.positions:
            continue
        ix, iy, _irot = ctx.positions[ic_ref]
        _iw, ih = ctx.fp_sizes.get(ic_ref, (5.0, 5.0))

        total_ch_width = (len(ch_list) - 1) * _CHANNEL_SPACING_MM
        ch_x_start, _ch_x_end = _shift_channel_start_for_overlaps(
            ix - total_ch_width / 2.0, total_ch_width,
            _CHANNEL_SPACING_MM, _occupied_x_ranges,
        )
        _occupied_x_ranges.append((ch_x_start, _ch_x_end))

        for ch_idx, (ic_pin, passives) in enumerate(ch_list):
            _adc_place_channel_strip(
                ch_idx, ic_pin, passives,
                ch_x_start, _CHANNEL_SPACING_MM,
                iy, ih, _STRIP_GAP_MM, ctx,
            )

    _log.info(
        "    3c2: %d channels across %d ICs",
        len(adc_channels), len(ctx.adc_ic_refs),
    )

    # Store on ctx for use by _phase_adc_analog_cluster and late phases
    ctx._adc_channels = adc_channels  # type: ignore[attr-defined]
    ctx._ic_channels = ic_channels  # type: ignore[attr-defined]
    ctx._r_top_connector_x = _r_top_connector_x  # type: ignore[attr-defined]
    ctx._occupied_x_ranges = _occupied_x_ranges  # type: ignore[attr-defined]
    ctx._CHANNEL_SPACING_MM = _CHANNEL_SPACING_MM  # type: ignore[attr-defined]
    ctx._STRIP_GAP_MM = _STRIP_GAP_MM  # type: ignore[attr-defined]


def _phase_adc_analog_cluster(ctx: PlacementContext) -> None:
    """3c3: Analog subcircuit clustering — pull remaining analog passives."""
    _log.info("  3c3: Analog subcircuit clustering")
    bounds = ctx.bounds

    _CHANNEL_SPACING_MM: float = getattr(ctx, "_CHANNEL_SPACING_MM", 8.0)
    _occupied_x_ranges: list[tuple[float, float]] = getattr(
        ctx, "_occupied_x_ranges", [],
    )

    # Find small passives on ADC signal nets (hop 1 — direct IC connection)
    analog_signal_refs: set[str] = set()
    _other_group_refs = ctx.relay_support_refs | ctx.power_group_fixed
    _already_claimed = (
        ctx.adc_channel_refs | ctx.adc_ic_refs | ctx.fixed_refs | _other_group_refs
    )
    for net in ctx.requirements.nets:
        if _is_power_or_bus_net(net.name):
            continue
        if not any(c.ref in ctx.adc_ic_refs and c.ref in ctx.positions
                   for c in net.connections):
            continue
        for c in net.connections:
            if (c.ref in ctx.positions
                    and c.ref not in _already_claimed
                    and _is_small_passive(c.ref)):
                analog_signal_refs.add(c.ref)

    # Hops 2-4: expand outward via signal nets
    hop2_refs = _expand_one_hop(
        analog_signal_refs, ctx,
        _already_claimed | analog_signal_refs,
        allow_small_ics=True,
    )
    small_ics_found = {r for r in hop2_refs if r.startswith("U")}
    hop3_refs = _expand_one_hop(
        small_ics_found, ctx,
        _already_claimed | analog_signal_refs | hop2_refs,
    )
    hop4_refs = _expand_one_hop(
        hop3_refs, ctx,
        _already_claimed | analog_signal_refs | hop2_refs | hop3_refs,
    )

    all_analog_cluster_refs = (
        analog_signal_refs | hop2_refs | hop3_refs | hop4_refs
    ) - ctx.adc_channel_refs - _other_group_refs

    if all_analog_cluster_refs and _occupied_x_ranges:
        last_x_max = max(xmax for _, xmax in _occupied_x_ranges)
        cluster_x = last_x_max + _CHANNEL_SPACING_MM + 2.0

        adc_ys = [ctx.positions[r][1] for r in ctx.adc_channel_refs
                  if r in ctx.positions]
        cluster_y_top = min(adc_ys) - 1.0 if adc_ys else bounds[1] + 5.0

        sorted_cluster = sorted(
            all_analog_cluster_refs,
            key=lambda r: (0 if r.startswith("U") else 2, r),
        )

        placed_count = _place_refs_in_column_grid(
            sorted_cluster, cluster_x, cluster_y_top,
            ctx, ctx.adc_channel_refs,
        )
        _log.info("    3c3: clustered %d analog refs near ADC channels", placed_count)

    # 3c3b. Pull remaining analog group outliers toward the cluster.
    analog_group_refs = _collect_analog_group_refs(ctx)

    if analog_group_refs and ctx.adc_channel_refs:
        outlier_refs = _find_analog_outliers(ctx, analog_group_refs)
        if outlier_refs:
            adc_xs = [ctx.positions[r][0] for r in ctx.adc_channel_refs
                      if r in ctx.positions]
            adc_ys = [ctx.positions[r][1] for r in ctx.adc_channel_refs
                      if r in ctx.positions]
            outlier_x = max(adc_xs) + _CHANNEL_SPACING_MM + 2.0
            outlier_y_top = min(adc_ys) - 1.0

            cnt = _place_refs_in_column_grid(
                outlier_refs, outlier_x, outlier_y_top,
                ctx, ctx.adc_channel_refs,
            )
            _log.info("    3c3b: pulled %d outliers into analog cluster", cnt)


def _phase_mcu_group(ctx: PlacementContext) -> None:
    """3c3: MCU peripheral tightening."""
    _log.info("  3c3: MCU peripheral tightening")
    bounds = ctx.bounds

    from kicad_pipeline.optimization.functional_grouper import _find_mcu_ref as _find_mcu
    mcu_ref_c3 = _find_mcu(ctx.requirements)
    if not (mcu_ref_c3 and mcu_ref_c3 in ctx.positions):
        return
    # Skip if the "MCU" is actually a power regulator IC that has already
    # been placed by _phase_power_chain_flow.
    if mcu_ref_c3 in ctx.power_group_fixed:
        _log.info("    3c3: skipping %s — already placed by power chain phase",
                  mcu_ref_c3)
        return

    _mcu_x, _mcu_y, _mcu_rot_orig = ctx.positions[mcu_ref_c3]
    mcu_w, mcu_h = ctx.fp_sizes.get(mcu_ref_c3, (5.0, 5.0))

    # Find MCU's FeatureBlock group
    mcu_group_refs: set[str] = set()
    for feat in ctx.requirements.features:
        feat_refs: set[str] = set()
        for comp in feat.components:
            r = comp.ref if hasattr(comp, "ref") else comp
            feat_refs.add(r)
        if mcu_ref_c3 in feat_refs:
            mcu_group_refs = feat_refs
            break

    # --- Step 0: Place U3 with antenna on BOTTOM board edge ---
    mcu_x, mcu_y, eff_w, eff_h, _mcu_rot = _mcu_place_u3(
        ctx, mcu_ref_c3, mcu_w, mcu_h,
    )

    mcu_left = mcu_x - eff_w / 2.0
    mcu_top = mcu_y - eff_h / 2.0
    mcu_right = mcu_x + eff_w / 2.0

    # --- Step 1: Build occupancy grid ---
    mcu_grid = _build_exclusion_grid(ctx, mcu_group_refs)
    mcu_grid.place(mcu_x, mcu_y, eff_w, eff_h)

    # --- Step 2: Build net adjacency and classify components ---
    ref_nets = _build_ref_net_adjacency(ctx, mcu_group_refs)

    connector_refs = {r for r in mcu_group_refs if r.startswith("J")}
    decoupling_refs: list[str] = []
    other_passive_refs: list[str] = []
    for ref in sorted(mcu_group_refs):
        if ref == mcu_ref_c3 or ref in connector_refs or ref in ctx.fixed_refs:
            continue
        if ref not in ctx.positions:
            continue
        if ref.startswith("C") and mcu_ref_c3 in ref_nets.get(ref, set()):
            decoupling_refs.append(ref)
        else:
            other_passive_refs.append(ref)

    # --- Step 3: Place decoupling caps ---
    _mcu_place_decoupling(ctx, decoupling_refs, mcu_x, mcu_y, mcu_left, mcu_grid)

    # --- Step 4: Place connectors ---
    _mcu_place_connectors(
        ctx, connector_refs, mcu_grid, mcu_x, mcu_y, mcu_left, mcu_top, eff_w,
    )

    # --- Step 5: USB subcircuit ---
    _mcu_place_usb_subcircuit(ctx, other_passive_refs, mcu_grid)

    # --- Step 6: Reset/Boot subcircuit ---
    sw_base_x, sw_base_y = _mcu_place_reset_boot(
        ctx, other_passive_refs, ref_nets, mcu_grid,
        mcu_x, mcu_y, eff_w, mcu_top,
    )

    # --- Step 6b: Place LED1 ---
    _mcu_place_led(ctx, other_passive_refs, mcu_grid, sw_base_x, sw_base_y)

    # --- Step 7: Place remaining passives ---
    def _mcu_prox_key(ref: str) -> tuple[int, float]:
        connected = mcu_ref_c3 in ref_nets.get(ref, set())
        rx, ry, _ = ctx.positions[ref]
        dist = math.sqrt((rx - mcu_x) ** 2 + (ry - mcu_y) ** 2)
        return (0 if connected else 1, dist)

    remaining = [r for r in other_passive_refs if r in ctx.positions]
    remaining.sort(key=_mcu_prox_key)
    _mcu_place_remaining(ctx, remaining, mcu_ref_c3, mcu_x, mcu_y,
                          eff_w, eff_h, mcu_grid)

    # --- Post-placement: Push components outside U3 courtyard ---
    _COURT_MARGIN = 2.0
    court = (
        mcu_x - eff_w / 2.0 - _COURT_MARGIN,
        mcu_y - eff_h / 2.0 - _COURT_MARGIN,
        mcu_x + eff_w / 2.0 + _COURT_MARGIN,
        mcu_y + eff_h / 2.0 + _COURT_MARGIN,
    )
    for ref in list(ctx.mcu_peripheral_refs):
        if ref == mcu_ref_c3:
            continue
        if ref.startswith("J") and ctx.positions[ref][0] > mcu_x:
            continue
        _push_component_outside_courtyard(ref, ctx, court, mcu_grid)

    _log.info(
        "    3c3: organized %d peripherals around %s at (%.1f, %.1f)",
        len(ctx.mcu_peripheral_refs), mcu_ref_c3, mcu_x, mcu_y,
    )


def _phase_ethernet_group(ctx: PlacementContext) -> None:
    """3c4: Ethernet group organization — vertical signal-chain column."""
    _log.info("  3c4: Ethernet group organization")
    bounds = ctx.bounds

    eth_group_refs = _collect_feature_refs(ctx, "ethernet")
    if not eth_group_refs:
        return

    eth_zone_rect = _find_zone_rect(ctx, "ethernet")
    by_prefix = _classify_refs_by_prefix(eth_group_refs, ctx, "U", "J")
    eth_ics = by_prefix["U"]
    eth_connectors = by_prefix["J"]

    if not (eth_ics and eth_zone_rect is not None):
        return

    ezx1, ezy1, ezx2, ezy2 = eth_zone_rect
    _ETH_STRIP_GAP = 1.0

    eth_net_refs = _build_net_to_group_refs(ctx, eth_group_refs)
    eth_grid = _build_exclusion_grid(ctx, eth_group_refs)

    eth_main_ic = eth_ics[0]
    crystal_refs = sorted([r for r in eth_group_refs
                           if r.startswith("Y") and r in ctx.positions])
    poe_ic = eth_ics[1] if len(eth_ics) > 1 else ""

    # Classify caps by net connectivity
    crystal_load_caps, poe_caps, other_caps = _classify_eth_caps(
        eth_group_refs, eth_net_refs, crystal_refs, poe_ic, ctx,
    )

    eth_anchor_x = (ezx1 + ezx2) / 2.0
    eth_anchor_y = ezy1 + 3.0
    placed_eth: set[str] = set()

    # Force-place main IC
    ic_w, ic_h = ctx.fp_sizes.get(eth_main_ic, (10.0, 10.0))
    ic_cx = _clamp(eth_anchor_x, ezx1 + ic_w / 2.0 + 1.0, ezx2 - ic_w / 2.0 - 1.0)
    ic_cy = _clamp(eth_anchor_y + ic_h / 2.0, ezy1 + ic_h / 2.0 + 1.0,
                   ezy2 - ic_h / 2.0 - 5.0)
    if eth_main_ic in ctx.positions and eth_main_ic not in ctx.fixed_refs:
        ctx.positions[eth_main_ic] = (ic_cx, ic_cy, 0.0)
        eth_grid.place(ic_cx, ic_cy, ic_w, ic_h)
        ctx.ethernet_fixed.add(eth_main_ic)
        ctx.fixed_refs.add(eth_main_ic)
        placed_eth.add(eth_main_ic)
        _log.info("    %s (W5500) force-placed at (%.1f, %.1f) "
                  "in ethernet zone", eth_main_ic, ic_cx, ic_cy)

    # Pre-register connector footprints in grid
    for _j13_ref in eth_connectors:
        _j13_w, _j13_h = ctx.fp_sizes.get(_j13_ref, (19.6, 12.5))
        eth_grid.place(eth_anchor_x, bounds[3] - _j13_h / 2.0 - 1.0,
                       _j13_w, _j13_h)

    def _place_eth_column(
        refs: list[str], col_x: float, start_y: float,
    ) -> float:
        cy = start_y
        for ref in refs:
            if (ref not in ctx.positions or ref in ctx.fixed_refs
                    or ref in placed_eth or ref == ""):
                continue
            w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
            tx = _clamp(col_x, ezx1 + 2.0, ezx2 - 2.0)
            ty = _clamp(cy + h / 2.0, ezy1 + 2.0, ezy2 - 2.0)
            px, py = eth_grid.find_free_pos(tx, ty, w, h, max_radius=6.0)
            ctx.positions[ref] = (px, py, 0.0)
            eth_grid.place(px, py, w, h)
            ctx.ethernet_fixed.add(ref)
            placed_eth.add(ref)
            cy = py + h / 2.0 + _ETH_STRIP_GAP
        return cy

    # Crystal + load caps
    _eth_place_crystal_and_caps(
        ctx, crystal_refs, crystal_load_caps,
        ic_cx, ic_cy, ic_h, eth_grid, placed_eth, eth_zone_rect,
    )

    # Remaining caps below main IC
    col1_bottom = _eth_place_caps_column(
        ctx, other_caps, ic_cx, ic_cy + ic_h / 2.0 + 1.5,
        eth_grid, placed_eth, eth_zone_rect,
    )

    # PoE/PHY module
    _eth_place_poe_ic_and_caps(
        ctx, poe_ic, poe_caps, eth_anchor_x, eth_grid, placed_eth,
    )

    # RJ45 connectors at bottom edge
    _eth_place_rj45_connectors(
        ctx, eth_connectors, eth_anchor_x, eth_grid, placed_eth, eth_zone_rect,
    )

    # Any remaining eth refs
    remaining_eth = sorted(eth_group_refs - placed_eth - ctx.fixed_refs)
    if remaining_eth:
        _place_eth_column(
            [r for r in remaining_eth if r in ctx.positions],
            eth_anchor_x, col1_bottom,
        )

    _log.info(
        "    3c4: organized %d ethernet components in signal chain: %s",
        len(ctx.ethernet_fixed), sorted(ctx.ethernet_fixed),
    )

    # Local overlap fix: shift caps that collide with crystal
    _eth_fix_crystal_cap_overlaps(ctx, placed_eth, eth_zone_rect)

    # Protect all ethernet from collision resolution
    ctx.fixed_refs.update(ctx.ethernet_fixed)

    # 3c4-post: Push non-ethernet components clear of ethernet ICs
    skip_refs = ctx.relay_support_refs | {r for r in ctx.positions if r.startswith("K")}
    for eic in [r for r in ctx.ethernet_fixed if r.startswith("U")]:
        _push_non_group_away_from_ic(eic, ctx, ctx.ethernet_fixed, skip_refs)
