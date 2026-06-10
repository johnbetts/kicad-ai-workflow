"""Shared helper functions for component group placement.

Contains common utilities used across different component group placement modules.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kicad_pipeline.optimization.collision_resolver import (
    _PlacementGrid,
    _rotation_aware_size,
)

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import PlacementContext

_log = logging.getLogger(__name__)

# Power net prefixes for classification
_POWER_NET_PREFIXES = (
    "VCC", "VDD", "GND", "AGND", "DGND", "AVCC", "+3V3", "+5V",
    "VIN", "VBUS", "V_", "+12V", "+24V", "I2C_",
)

# Small passive component prefixes
_SMALL_PREFIXES = ("R", "C", "D", "L", "LED", "SW")


def _clamp(val: float, lo: float, hi: float) -> float:
    """Clamp *val* to [lo, hi]."""
    return max(lo, min(hi, val))


def _clamp_to_bounds(
    x: float, y: float, bounds: tuple[float, float, float, float],
    margin: float = 2.0,
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


def _should_add_ref(
    c_ref: str,
    ctx: PlacementContext,
    already_claimed: set[str],
    allow_small_ics: bool,
    max_ic_pins: int,
) -> bool:
    """Return True if *c_ref* should be added during a one-hop expansion."""
    if c_ref not in ctx.positions or c_ref in already_claimed:
        return False
    if _is_small_passive(c_ref):
        return True
    if not allow_small_ics or not c_ref.startswith("U"):
        return False
    comp = next(
        (comp for comp in ctx.requirements.components if comp.ref == c_ref), None,
    )
    return comp is not None and len(comp.pins) <= max_ic_pins


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
                if c.ref == ref:
                    continue
                if _should_add_ref(c.ref, ctx, already_claimed, allow_small_ics, max_ic_pins):
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
        _, _raw_h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        h = _raw_h
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