"""EE placement optimizer — group organization phases.

Contains the group-specific placement phases that organize components
within functional groups (power, MCU, ethernet, ADC channels).

Extracted from ``ee_phases.py`` to reduce module size.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from kicad_pipeline.optimization.collision_resolver import (
    _PlacementGrid,
    _rotation_aware_size,
)

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import PlacementContext
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
        _pad_top = _te[1] - _trial_oy
        # Leave 10mm at top for USB-C connector + CC resistors
        mcu_origin_y = bounds[3] - _pad_bot - 2.0
        # Ensure top pads are at least 10mm from top edge (room for connectors)
        top_clearance = bounds[1] - _pad_top + 10.0
        if _trial_oy + (mcu_origin_y - _trial_oy) + _pad_top < bounds[1] + 10.0:
            mcu_origin_y = max(mcu_origin_y, top_clearance)
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


def _mcu_power_pin_board_pos(
    ctx: PlacementContext,
    mcu_ref: str,
    mcu_x: float,
    mcu_y: float,
    rotation: float,
) -> tuple[float, float] | None:
    """Return board-space (x, y) of the 3V3/VCC power pad on the MCU footprint.

    Scans all pads for power-net membership.  Returns the centroid of all
    matching pad positions in board space, or ``None`` if not determinable.
    The result is used to position decoupling caps on the correct side of the
    IC regardless of its rotation.
    """
    mcu_fp = next(
        (fp for fp in ctx.initial_pcb.footprints if fp.ref == mcu_ref), None,
    )
    if mcu_fp is None:
        return None

    # Build set of nets attached to the MCU from requirements
    power_pads: list[tuple[float, float]] = []
    rad = math.radians(-rotation)  # KiCad CW convention
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    for pad in mcu_fp.pads:
        pad_net = pad.net_name if hasattr(pad, "net_name") else ""
        # Look up net via requirements connections
        if not pad_net:
            for net in ctx.requirements.nets:
                for conn in net.connections:
                    if conn.ref == mcu_ref and conn.pin == pad.number:
                        pad_net = net.name
                        break
                if pad_net:
                    break
        if not pad_net:
            continue
        net_up = pad_net.upper()
        is_power = any(
            net_up.startswith(pfx) for pfx in
            ("+3V3", "+3.3V", "VCC", "VDD", "AVCC", "DVCC", "3V3")
        )
        if not is_power:
            continue
        # Rotate footprint-local pad position to board space
        bx = mcu_x + pad.position.x * cos_a - pad.position.y * sin_a
        by = mcu_y + pad.position.x * sin_a + pad.position.y * cos_a
        power_pads.append((bx, by))

    if not power_pads:
        return None
    avg_x = sum(p[0] for p in power_pads) / len(power_pads)
    avg_y = sum(p[1] for p in power_pads) / len(power_pads)
    return avg_x, avg_y


def _mcu_place_decoupling(
    ctx: PlacementContext,
    decoupling_refs: list[str],
    mcu_x: float,
    mcu_y: float,
    mcu_left: float,
    grid: _PlacementGrid,
) -> None:
    """Place decoupling caps within 3-5mm of the MCU power pin (Step 3).

    Detects which side of the IC body has the 3V3/VCC power pad and places
    caps on that side.  With rotation=180 the ESP32's left-column 3V3 pad
    (WEST at rot=0) maps to the EAST (right) side in board coordinates, so
    caps are placed to the RIGHT rather than to the left.
    """
    bounds = ctx.bounds

    # Find the MCU ref and courtyard half-width
    mcu_ref = None
    for ref_c in ctx.mcu_peripheral_refs:
        if ref_c.startswith("U"):
            mcu_ref = ref_c
            break
    if mcu_ref and mcu_ref in ctx.fp_sizes:
        mcu_cw, _mcu_ch = ctx.fp_sizes[mcu_ref]
    else:
        mcu_cw = abs(mcu_x - mcu_left) * 2.0

    courtyard_left = mcu_x - mcu_cw / 2.0
    courtyard_right = mcu_x + mcu_cw / 2.0

    # Determine which horizontal side has the 3V3/VCC power pad.
    rotation = ctx.positions.get(mcu_ref, (0.0, 0.0, 0.0))[2] if mcu_ref else 0.0
    power_pos = (
        _mcu_power_pin_board_pos(ctx, mcu_ref, mcu_x, mcu_y, rotation)
        if mcu_ref
        else None
    )

    if power_pos is not None:
        # Place caps on the same horizontal side as the power pad, 3mm away.
        pwr_x, pwr_y = power_pos
        cap_y_start = pwr_y  # align with power pin row
        if pwr_x > mcu_x:
            # Power pin is to the right → place caps right of MCU
            def _cap_x(w: float) -> float:
                return courtyard_right + w / 2.0 + 0.5
        else:
            # Power pin is to the left → place caps left of MCU
            def _cap_x(w: float) -> float:
                return courtyard_left - w / 2.0 - 0.5
        _log.info(
            "    decoupling: power pin at (%.1f,%.1f) → placing caps on %s side",
            pwr_x, pwr_y, "right" if pwr_x > mcu_x else "left",
        )
    else:
        # Fallback: place left of MCU (original behaviour)
        cap_y_start = mcu_y - 2.0
        def _cap_x(w: float) -> float:
            return courtyard_left - w / 2.0 - 0.5
        _log.info("    decoupling: no power pin found, falling back to left of MCU")

    cap_spacing = 2.5

    for i, ref in enumerate(decoupling_refs):
        w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
        cap_x = _cap_x(w)
        tx = _clamp(cap_x, bounds[0] + w / 2.0 + 0.5, bounds[2] - w / 2.0 - 0.5)
        ty = _clamp(cap_y_start + i * cap_spacing,
                    bounds[1] + h / 2.0 + 0.5, bounds[3] - h / 2.0 - 0.5)
        # Force-place without grid search — grid may push them far away
        ctx.positions[ref] = (tx, ty, 0.0)
        ctx.mcu_peripheral_refs.add(ref)
        ctx.fixed_refs.add(ref)  # protect from clamp/collision phases
        grid.place(tx, ty, w, h)
        dist_to_mcu = ((tx - mcu_x) ** 2 + (ty - mcu_y) ** 2) ** 0.5
        _log.info("    %s (decoupling): FORCE->(%.1f,%.1f) [%.1fmm from MCU]",
                  ref, tx, ty, dist_to_mcu)


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

    # J1 (USB-C): FORCE-place at TOP edge, left of MCU (away from antenna).
    # rotation 180 so pads face into the board.
    # We skip grid.find_free_pos because U1's large footprint blocks it.
    # Guard: skip if J1 is owned by the ethernet feature block — ethernet group
    # phase places the RJ45 at the top edge with the correct rotation.
    _eth_refs_j1 = _collect_feature_refs(ctx, "ethernet", "eth")
    if ("J1" in connector_refs and "J1" in ctx.positions
            and "J1" not in ctx.fixed_refs and "J1" not in _eth_refs_j1):
        w1, h1 = ctx.fp_sizes.get("J1", (9.0, 7.5))
        j1_x = mcu_left + 3.0  # left side of MCU, away from antenna
        j1_y = bounds[1] + h1 / 2.0 + 0.5  # near top edge
        j1_x = _clamp(j1_x, bounds[0] + w1 / 2.0 + 1.0, bounds[2] - w1 / 2.0 - 1.0)
        j1_y = _clamp(j1_y, bounds[1] + h1 / 2.0 + 0.5, bounds[3] - h1 / 2.0 - 1.0)
        ctx.positions["J1"] = (j1_x, j1_y, 180.0)
        ctx.mcu_peripheral_refs.add("J1")
        ctx.fixed_refs.add("J1")  # protect from clamp/collision resolution
        _log.info("    J1 (USB-C) -> top edge at (%.1f, %.1f) rot=180", j1_x, j1_y)

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
    """Place USB CC resistors near J1 (USB-C) and ESD near J2 (Step 5)."""
    # --- CC resistors (R3/R4) near J1 (USB-C) ---
    j1_pos = ctx.positions.get("J1")
    bounds = ctx.bounds
    if j1_pos:
        j1x, j1y, _ = j1_pos
        j1w, j1h = ctx.fp_sizes.get("J1", (9.0, 7.5))
        # Find CC resistors: R3/R4 are typical CC1/CC2 resistors
        cc_refs = [r for r in ("R3", "R4") if r in other_passive_refs
                   and r in ctx.positions]
        for i, ref in enumerate(cc_refs):
            w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
            # Place CC resistors just below J1, side by side
            px = j1x - 2.0 + i * (w + 2.0)
            py = j1y + j1h / 2.0 + h / 2.0 + 1.0
            px = _clamp(px, bounds[0] + w / 2.0 + 0.5, bounds[2] - w / 2.0 - 0.5)
            py = _clamp(py, bounds[1] + h / 2.0 + 0.5, bounds[3] - h / 2.0 - 0.5)
            ctx.positions[ref] = (px, py, 0.0)
            ctx.mcu_peripheral_refs.add(ref)
            ctx.fixed_refs.add(ref)
            grid.place(px, py, w, h)
            if ref in other_passive_refs:
                other_passive_refs.remove(ref)
            _log.info("    %s (CC resistor) -> near J1 at (%.1f, %.1f)", ref, px, py)

    # --- USB ESD + series resistors near J2 ---
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
    # Find SW+R pairs via net connectivity (not hardcoded refs)
    sw_refs = [r for r in other_passive_refs if r in ctx.positions
               and r.startswith("SW")]
    r_refs = [r for r in other_passive_refs if r in ctx.positions
              and r.startswith("R")]
    sw_pairs: list[tuple[str, str]] = []
    used_sw: set[str] = set()
    for sw in sw_refs:
        sw_nets = ref_nets.get(sw, set())
        for res in r_refs:
            if res in sw_nets and res not in used_sw:
                sw_pairs.append((sw, res))
                used_sw.add(sw)
                used_sw.add(res)
                break
    unique_pairs = sw_pairs
    unpaired_sw = [r for r in sw_refs if r not in used_sw]

    mcu_x - eff_w / 2.0
    # Place switches on the LEFT side of the board, grouped vertically
    sw_base_x = bounds[0] + 6.0  # near left edge
    sw_base_y = mcu_y - 5.0  # near MCU vertical center

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
    mcu_target_gap = 4.0
    mcu_clear = 3.0

    ring_slots: list[tuple[float, float]] = []
    if mcu_top > bounds[1] + 10.0:
        for dx_off in range(-4, 5):
            ring_slots.append((mcu_x + dx_off * mcu_target_gap,
                               mcu_top - mcu_clear - 3.0))
        for dx_off in range(-3, 4):
            ring_slots.append((mcu_x + dx_off * mcu_target_gap,
                               mcu_top - mcu_clear - 7.0))
    slot_x_left = mcu_left - mcu_clear - 2.0
    for dy_off in range(-3, 4):
        ring_slots.append((slot_x_left, mcu_y + dy_off * 3.5))
    slot_x_right = mcu_right + mcu_clear + 2.0
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
            tx = mcu_left - mcu_target_gap - w / 2.0
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
    """Place RJ45 connectors at the LEFT board edge, vertically centered.

    Signal flow is left→right: J1 (RJ45) sits at the left edge with its
    mating face pointing outward (left).  Rotation=90 is used so the port
    faces x_min.  After a 90° rotation the effective width becomes h and
    the effective height becomes w.  Origin is adjusted so all pads stay
    inside the board.
    """
    bounds = ctx.bounds
    _ezx1, ezy1, _ezx2, ezy2 = zone_rect
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space
    zone_cy = (ezy1 + ezy2) / 2.0
    for ref in eth_connectors:
        if ref not in ctx.positions or ref in ctx.fixed_refs:
            continue
        fp_match = None
        for fp in ctx.initial_pcb.footprints:
            if fp.ref == ref:
                fp_match = fp
                break
        w, h = ctx.fp_sizes.get(ref, (19.6, 15.4))
        # After 90° rotation: effective dims are (h_body → x-axis, w_body → y-axis)
        rot_w, rot_h = h, w
        cent_x = bounds[0] + rot_w / 2.0 + 1.0
        cent_y = _clamp(zone_cy, ezy1 + rot_h / 2.0 + 1.0, ezy2 - rot_h / 2.0 - 1.0)
        if fp_match is not None:
            # Rotation=90: connector port faces left (x_min). Adjust origin so
            # pads don't fall outside the board.
            trial_origin_x = bounds[0] + 3.0
            trial_origin_y = zone_cy
            pad_min_x, _, _, _ = pad_extent_in_board_space(
                fp_match, trial_origin_x, trial_origin_y, 90.0,
            )
            edge_margin = 1.0
            if pad_min_x < bounds[0] + edge_margin:
                trial_origin_x += (bounds[0] + edge_margin - pad_min_x)
            cent_x, cent_y = origin_to_centroid(
                fp_match, trial_origin_x, trial_origin_y, 90.0,
            )
            cent_y = _clamp(cent_y, ezy1 + rot_h / 2.0 + 1.0, ezy2 - rot_h / 2.0 - 1.0)
        ctx.positions[ref] = (cent_x, cent_y, 90.0)
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        _log.info("    %s (RJ45) -> left edge (%.1f, %.1f) rot=90", ref, cent_x, cent_y)


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
    buck2_left = [*buck2_in_caps, buck2_ic, *bst2_passives, *l2_refs]
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
#
# Clearance check (0805 courtyard 2.7x2.0mm, SOD-323 ~2.7x2.0mm, L1210 4.1x3.4mm):
#   input_cap vs bootstrap_cap: same x, dy=10.1 — OK (>>2.0mm)
#   inductor vs catch_diode: dy=3.2, min=(3.4+2.0)/2=2.7 -> OK
#   catch_diode vs fb_bot_r: dy=3.3 > 2.0mm minimum — OK
#   fb_bot_r vs fb_top_r: dy=2.5 > 2.0mm minimum — OK
_BUCK_PASSIVE_OFFSETS: dict[str, tuple[float, float, float]] = {
    # (role, (dx, dy, rotation))  — role matched by net name keywords
    # Offsets are ABSOLUTE distances from the buck IC centroid.
    # IC is placed at +90deg rotation so:
    #   PH/SW pin (pin 7) → RIGHT side (dx=+2.7, dy=-0.6)
    #   VSNS/FB pin (pin 5) → RIGHT side (dx=+2.7, dy=+1.9)
    #   VIN pin (pin 2) → LEFT side (dx=-2.7, dy=-0.6)
    #   BOOT pin (pin 1) → LEFT side (dx=-2.7, dy=-1.9)
    #
    # Footprint sizes (rotation-aware, from estimate_footprint_size):
    #   U1 SOIC-8 at ±90°: 8.9w x 5.9h, half=(4.45, 2.95)
    #   L1210:              3.7w x 3.0h, half=(1.85, 1.50)
    #   SOD-323 (D1):       4.8w x 2.2h, half=(2.40, 1.10)
    #   0805 cap (C1):      4.1w x 2.2h, half=(2.05, 1.10)
    #   0805 cap (C2):      2.5w x 1.8h, half=(1.25, 0.90)
    #   0402 R/C:           1.5w x 1.0h, half=(0.75, 0.50)
    #
    # Min center-to-center (with 0.5mm gap):
    #   U1-L1 dx: 4.45+1.85+0.50=6.80   U1-D1 dx: 4.45+2.40+0.50=7.35
    #   U1-0805 dx: 4.45+2.05+0.50=7.00 U1-0402 dx: 4.45+0.75+0.50=5.70
    #   L1-D1 dy: 1.50+1.10+0.50=3.10   R-R dy: 0.50+0.50+0.50=1.50
    #
    # Signal flow: input(left) -> IC -> PH/inductor/diode(right) -> output(far right)
    # Layout:
    #   C3(bootstrap)  C1(input)   U1(IC)   L1(inductor)  C2(output)
    #                                        D1(diode)
    #                                        R2(fb_bot) R1(fb_top)
    # Collision-safe offsets for SOIC-8 at 90° rotation:
    #   SOIC-8 90°: ~8.9w x 5.9h, half=(4.45, 2.95)
    #   L1210:      3.7w x 3.0h, half=(1.85, 1.50)
    #   0805 cap:   2.5w x 1.8h, half=(1.25, 0.90)
    #   SOD-323:    3.0w x 3.0h, half=(1.50, 1.50)
    #   0402:       1.5w x 1.0h, half=(0.75, 0.50)
    # Min center-to-center with 0.5mm gap:
    #   U1-L1: 4.45+1.85+0.5 = 6.8mm
    #   U1-0805: 4.45+1.25+0.5 = 6.2mm
    #   L1-0805: 1.85+1.25+0.5 = 3.6mm
    "input_cap": (-6.5, 0.5, 0.0),        # C on VIN rail, LEFT of IC
    "bootstrap_cap": (-6.5, -2.5, 0.0),   # C on BST net, LEFT of IC, above C1
    "inductor": (7.0, -0.6, 0.0),         # L on SW net, RIGHT of IC near PH pin
    "catch_diode": (7.0, 3.0, 180.0),     # D on SW net, RIGHT below inductor
    "fb_bot_r": (7.0, 5.5, 0.0),          # R on FB+GND, below catch diode
    "fb_top_r": (7.0, 7.5, 180.0),        # R on FB only, below fb_bot_r
    # C at output (L1@7.0 + L1_half 1.85 + C_half 1.25 + gap 0.9 = 11.0)
    "output_cap": (11.0, 0.0, -90.0),
}

# Relative offsets for passives around an LDO IC anchor.
# Learned from human-routed AMS1117-3.3 layout on a 50x40mm board.
#
# Clearance calculation (actual measured values from easyeda2kicad SOT-223 footprint):
#   fp_size_dict returns U2=9.36x6.70mm (includes tab pad), half-width=4.68mm
#   C_0805 fp_size at -90°: effective width=2.35mm, half=1.175mm
#   No-collision condition: cap_cx ± 1.175 must not overlap [u2_cx ± 4.68]
#   Input cap (left): cap_cx + 1.175 ≤ u2_cx - 4.68 → offset ≤ -5.855 → use -6.5mm
#   Output cap (right): cap_cx - 1.175 ≥ u2_cx + 4.68 → offset ≥ +5.855 → use +6.5mm
#
# The LDO input cap sits just outside the IC tab pad (NOT at the midpoint
# between the buck and LDO — the old -17.4mm offset placed it on top of the
# buck output cap, and -5.5mm still left 0.355mm overlap due to the wide tab).
_LDO_PASSIVE_OFFSETS: dict[str, tuple[float, float, float]] = {
    # LDO IC at 180deg: VIN (pin 3) faces LEFT, VOUT (pin 2) faces LEFT,
    # VOUT_TAB (pin 4) faces RIGHT.
    # SOT-223 (U2): 10.4w x 7.7h at 180deg (no swap), half=(5.2, 3.85)
    #   VIN pin at dx=-3.0, dy=+2.3 from center
    #   VOUT pin at dx=-3.0, dy=0.0 from center
    #   VOUT_TAB at dx=+3.0, dy=0.0 from center
    # 0805 cap at -90deg: 2.2w x 4.1h, half-w=1.1
    # Min dx from IC center = 5.2 + 1.1 + 0.50 = 6.80
    #
    # Output cap near VOUT pin on LEFT side (shortest decoupling path).
    # Input cap on LEFT side below output cap, near VIN pin.
    # 0805 at -90deg: 2.2w x 4.1h. Two caps stacked vertically need
    # dy >= (4.1+4.1)/2 + 0.5 = 4.6mm
    # Input and output caps stacked vertically on LEFT side of LDO.
    # Output cap at dy=0 (near VOUT pin), HF bypass stacks at dy+5.
    # Input cap at dy=-5 (above output, clear of duplicate stacking below).
    "input_cap": (-6.0, -4.5, -90.0),     # C on VIN side (LEFT), above VOUT cap
    "output_cap": (-6.0, 0.0, -90.0),     # C on VOUT side (LEFT), near VOUT pin
}

# Net-name keywords used to classify passive roles in power subcircuits.
_VIN_KEYWORDS = ("VIN", "+24V", "+12V", "VBUS", "V_IN")
_BST_KEYWORDS = ("BST", "BOOT", "BOOTSTRAP")
_SW_KEYWORDS = ("SW", "PH", "PHASE")
_FB_KEYWORDS = ("FB", "VSNS", "FEEDBACK", "SENSE")
_OUTPUT_KEYWORDS = ("+5V", "+3V3", "+3.3V", "+1V8", "VOUT", "V_OUT", "BUCK_5V")


def _collect_passive_nets(
    ref: str,
    ic_ref: str,
    ctx: PlacementContext,
) -> tuple[list[str], list[str]]:
    """Return (all_nets_for_ref, shared_nets_with_ic) for a passive component.

    Both lists contain upper-cased net names.
    """
    all_nets: list[str] = []
    shared: list[str] = []
    for net in ctx.requirements.nets:
        conn_refs = {c.ref for c in net.connections}
        if ref not in conn_refs:
            continue
        name_upper = net.name.upper()
        all_nets.append(name_upper)
        if ic_ref in conn_refs:
            shared.append(name_upper)
    return all_nets, shared


def _classify_cap_role(
    all_nets: list[str],
    shared_nets: list[str],
    ic_input_voltage: float,
    ic_output_voltage: float,
) -> str:
    """Classify a capacitor's role in a power subcircuit.

    Returns one of: "bootstrap_cap", "input_cap", "output_cap".
    """
    # Bootstrap cap: on BST net
    if any(any(kw in n for kw in _BST_KEYWORDS) for n in all_nets):
        return "bootstrap_cap"
    # Input cap: on VIN net shared with IC
    if any(any(kw in n for kw in _VIN_KEYWORDS) for n in shared_nets):
        return "input_cap"
    # Use voltage magnitude when regulator voltages are known
    if ic_input_voltage > 0 and ic_output_voltage > 0:
        cap_voltage = _estimate_net_voltage(all_nets)
        if cap_voltage is not None:
            in_diff = abs(cap_voltage - ic_input_voltage)
            out_diff = abs(cap_voltage - ic_output_voltage)
            return "input_cap" if in_diff < out_diff else "output_cap"
    # Output cap: on output net keyword
    if any(any(kw in n for kw in _OUTPUT_KEYWORDS) for n in all_nets):
        return "output_cap"
    # Fallback: shared net with IC → input, otherwise output
    return "input_cap" if shared_nets else "output_cap"


_GND_NET_NAMES: frozenset[str] = frozenset({"GND", "AGND", "DGND", "PGND"})
"""Canonical GND net names used to classify feedback-divider resistors."""


def _classify_resistor_role_power(all_nets: list[str]) -> str | None:
    """Classify a resistor's role in a power subcircuit (FB divider top/bottom)."""
    for n in all_nets:
        if any(kw in n for kw in _FB_KEYWORDS):
            if any(n2 in _GND_NET_NAMES for n2 in all_nets):
                return "fb_bot_r"
            return "fb_top_r"
    return None


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

    all_nets_for_ref, shared_nets = _collect_passive_nets(ref, ic_ref, ctx)
    prefix = ref[0]

    if prefix == "L":
        # Inductors in a power subcircuit are always the main switching inductor
        return "inductor"

    if prefix == "D":
        # Diodes in a power subcircuit are always the catch/freewheeling diode
        return "catch_diode"

    if prefix == "C":
        return _classify_cap_role(
            all_nets_for_ref, shared_nets, ic_input_voltage, ic_output_voltage,
        )

    if prefix == "R":
        return _classify_resistor_role_power(all_nets_for_ref)

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


def _resolve_power_zone_bounds(
    ctx: PlacementContext,
    n_regs: int,
) -> tuple[float, float, float, float]:
    """Return (zx1, zy1, zx2, zy2) for the power zone, expanding if too narrow."""
    min_power_zone_w = 25.0
    zone_rect = _find_zone_rect(ctx, "power")
    if zone_rect is None:
        return ctx.bounds
    zx1, zy1, zx2, zy2 = zone_rect
    if zx2 - zx1 < min_power_zone_w and n_regs >= 2:
        _log.info(
            "    3c1b: power zone too narrow (%.1fmm) — expanding to board bounds",
            zx2 - zx1,
        )
        return ctx.bounds
    return zone_rect


def _compute_regulator_x_fractions(n_regs: int) -> list[float]:
    """Compute horizontal zone-fraction positions for N regulators.

    - 1 regulator: 30% (room for output passives on right)
    - 2 regulators: 17%, 82% (learned from human reference board)
    - N regulators: evenly spread 15%-85%
    """
    if n_regs == 1:
        return [0.30]
    if n_regs == 2:
        return [0.17, 0.82]
    return [0.15 + 0.70 * i / (n_regs - 1) for i in range(n_regs)]


def _sort_regulators_by_flow(
    all_reg_scs: list[object],
    ctx: PlacementContext,
    topology: object,
) -> None:
    """Sort regulator subcircuits in-place by power-chain flow order.

    Earlier in the chain (higher input voltage) sorts to a lower index
    so it is placed further left in the signal-flow layout.
    """
    def _flow_order(sc: object) -> float:
        if sc.input_domain and sc.input_domain in topology.domain_order:  # type: ignore[union-attr]
            return float(list(topology.domain_order).index(sc.input_domain))  # type: ignore[union-attr]
        max_v = 0.0
        for net in ctx.requirements.nets:
            if not any(c.ref == sc.anchor_ref for c in net.connections):  # type: ignore[union-attr]
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
                max_v = max(max_v, 100.0)
        return -max_v if max_v > 0 else 999.0

    all_reg_scs.sort(key=_flow_order)


def _gather_ic_net_refs(
    ic_ref: str,
    sc_refs: set[str],
    ctx: PlacementContext,
    power_group_refs: set[str],
) -> set[str]:
    """Return all passive refs connected (directly or 1-hop) to *ic_ref*.

    Pass 1: direct connections on non-GND nets shared with *ic_ref*.
    Pass 2: one hop — non-GND nets that share any ref from pass-1 result.
    """
    _gnd_nets = frozenset({"GND", "AGND", "DGND", "PGND"})
    ic_net_refs: set[str] = set(sc_refs)

    for net in ctx.requirements.nets:
        if net.name.upper() in _gnd_nets:
            continue
        conn_refs = {c.ref for c in net.connections}
        if ic_ref not in conn_refs:
            continue
        for c in net.connections:
            if (c.ref != ic_ref
                    and c.ref in ctx.positions
                    and c.ref[0] in "RCLDF"
                    and (c.ref in power_group_refs or c.ref in sc_refs)):
                ic_net_refs.add(c.ref)

    for net in ctx.requirements.nets:
        if net.name.upper() in _gnd_nets:
            continue
        conn_refs = {c.ref for c in net.connections}
        if conn_refs & ic_net_refs:
            for c in net.connections:
                if (c.ref not in ic_net_refs
                        and c.ref != ic_ref
                        and c.ref in ctx.positions
                        and c.ref[0] in "RCLDF"
                        and c.ref in power_group_refs):
                    ic_net_refs.add(c.ref)

    return ic_net_refs


def _place_regulator_passives(
    ic_ref: str,
    ic_x: float,
    ic_y: float,
    ic_net_refs: set[str],
    offsets: dict[str, tuple[float, float, float]],
    zone_bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
    in_v: float,
    out_v: float,
) -> set[str]:
    """Place passives for one regulator IC at learned offsets.

    Returns the set of role names that were placed.
    """
    zx1, zy1, zx2, zy2 = zone_bounds
    # Offsets are absolute physical distances (learned from reference boards).
    # They must NOT scale up with zone size — larger zones should not push
    # power components further apart.  Only scale DOWN if the zone is smaller
    # than the reference to avoid off-board placement.
    zone_w = zx2 - zx1
    zone_h = zy2 - zy1
    ref_zone_w = 35.0  # reference zone width offsets were tuned for
    ref_zone_h = 30.0  # reference zone height offsets were tuned for
    sx = min(1.0, max(0.5, zone_w / ref_zone_w))
    sy = min(1.0, max(0.5, zone_h / ref_zone_h))
    placed_roles: set[str] = set()
    # Track position of first component placed in each role so duplicates
    # (e.g. two output caps: bulk + HF bypass) can be stacked nearby.
    role_positions: dict[str, tuple[float, float]] = {}
    dup_offset_mm = 5.0  # stack duplicates far enough for rotated 0805 caps (4.1mm tall)

    for ref in sorted(ic_net_refs):
        if ref == ic_ref:
            continue
        role = _classify_passive_role_power(ref, ctx, ic_ref, ic_net_refs, in_v, out_v)
        if role is None or role not in offsets:
            continue
        dx, dy, rot = offsets[role]
        if role not in placed_roles:
            # First component with this role — place at learned offset
            placed_roles.add(role)
            px = _clamp(ic_x + dx * sx, zx1 + 1.0, zx2 - 1.0)
            py = _clamp(ic_y + dy * sy, zy1 + 1.0, zy2 - 1.0)
            role_positions[role] = (px, py)
        else:
            # Duplicate role (e.g. C6 is a second output_cap alongside C5).
            # Stack it adjacent to the primary component.
            base_x, base_y = role_positions[role]
            px = _clamp(base_x, zx1 + 1.0, zx2 - 1.0)
            py = _clamp(base_y + dup_offset_mm, zy1 + 1.0, zy2 - 1.0)
        ctx.positions[ref] = (px, py, rot)
        ctx.power_group_fixed.add(ref)
    return placed_roles


def _compute_global_pin_positions(
    ic_ref: str,
    ctx: PlacementContext,
) -> dict[str, tuple[float, float]]:
    """Compute global (x, y) positions of all pads on *ic_ref*.

    Returns a dict mapping pad net_name (upper) to global (x, y).
    For pads with the same net, the first one wins.
    """
    import math as _m

    fp = None
    for f in ctx.initial_pcb.footprints:
        if f.ref == ic_ref:
            fp = f
            break
    if fp is None:
        return {}

    cx, cy, rot_deg = ctx.positions.get(ic_ref, (0.0, 0.0, 0.0))
    rad = _m.radians(rot_deg)
    cos_r = _m.cos(rad)
    sin_r = _m.sin(rad)

    result: dict[str, tuple[float, float]] = {}
    for pad in fp.pads:
        if not pad.net_name:
            continue
        net_upper = pad.net_name.upper()
        if net_upper in result or net_upper in ("GND", "AGND", "DGND", "PGND"):
            continue
        gx = cx + pad.position.x * cos_r - pad.position.y * sin_r
        gy = cy + pad.position.x * sin_r + pad.position.y * cos_r
        result[net_upper] = (gx, gy)
    return result


def _pull_passives_toward_pins(
    ic_ref: str,
    ic_net_refs: set[str],
    zone_bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Pull placed passives toward the IC pin they share a non-GND net with.

    After offset-based placement, passives may be too far from their connected
    pin. This pass computes the global pin position and pulls each passive
    closer — placing it just outside the IC edge nearest that pin, at the
    pin's Y coordinate (for left/right pins) or X coordinate (for top/bottom
    pins), while preserving the offset's side choice (sign of dx/dy).
    """
    import math as _m

    gnd_nets = frozenset({"GND", "AGND", "DGND", "PGND"})
    pin_positions = _compute_global_pin_positions(ic_ref, ctx)
    if not pin_positions:
        return

    zx1, zy1, zx2, zy2 = zone_bounds
    ic_x, ic_y, _ic_rot = ctx.positions[ic_ref]
    ic_w, ic_h = ctx.fp_sizes.get(ic_ref, (6.0, 6.0))
    if abs(_ic_rot) % 180 in (90.0, 270.0):
        ic_w, ic_h = ic_h, ic_w
    ic_half_w = ic_w / 2.0
    ic_half_h = ic_h / 2.0
    gap = 0.2  # tight clearance for power loop components

    for ref in sorted(ic_net_refs):
        if ref == ic_ref or ref not in ctx.positions:
            continue
        # Find the shared non-GND net
        shared_pin_pos: tuple[float, float] | None = None
        for net in ctx.requirements.nets:
            name_u = net.name.upper()
            if name_u in gnd_nets:
                continue
            conn_refs = {c.ref for c in net.connections}
            if ref in conn_refs and ic_ref in conn_refs and name_u in pin_positions:
                shared_pin_pos = pin_positions[name_u]
                break

        if shared_pin_pos is None:
            continue

        px, py, p_rot = ctx.positions[ref]
        pin_x, pin_y = shared_pin_pos
        pw, ph = ctx.fp_sizes.get(ref, (2.0, 2.0))
        if p_rot % 180 in (90.0, 270.0):
            pw, ph = ph, pw

        # Determine which IC edge the pin is closest to
        pin_dx = pin_x - ic_x
        pin_dy = pin_y - ic_y

        # The passive should be placed just outside the IC edge nearest
        # the pin. Place on the side of the IC where the pin is, at min clearance
        if abs(pin_dx) >= abs(pin_dy):
            # Pin is on left or right edge
            side_sign = 1.0 if pin_dx >= 0 else -1.0
            target_x = ic_x + side_sign * (ic_half_w + pw / 2.0 + gap)
            target_y = pin_y  # Align Y with pin
        else:
            # Pin is on top or bottom edge
            side_sign = 1.0 if pin_dy >= 0 else -1.0
            target_x = pin_x  # Align X with pin
            target_y = ic_y + side_sign * (ic_half_h + ph / 2.0 + gap)

        target_x = _clamp(target_x, zx1 + 1.0, zx2 - 1.0)
        target_y = _clamp(target_y, zy1 + 1.0, zy2 - 1.0)

        # Only move if it brings the passive closer to the pin
        old_pin_dist = _m.sqrt((px - pin_x) ** 2 + (py - pin_y) ** 2)
        new_pin_dist = _m.sqrt((target_x - pin_x) ** 2 + (target_y - pin_y) ** 2)

        if new_pin_dist < old_pin_dist:
            ctx.positions[ref] = (target_x, target_y, p_rot)
            _log.info(
                "      pin-pull %s toward %s pin: (%.1f,%.1f) -> (%.1f,%.1f) "
                "[pin_d=%.1f->%.1f]",
                ref, ic_ref, px, py, target_x, target_y,
                old_pin_dist, new_pin_dist,
            )


def _place_power_chain_ic(
    sc: object,
    reg_idx: int,
    x_fracs: list[float],
    zone_bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Place one regulator IC and its passives at the signal-flow position."""
    from kicad_pipeline.optimization.functional_grouper import SubCircuitType

    ic_ref: str = sc.anchor_ref  # type: ignore[union-attr]
    if ic_ref not in ctx.positions:
        return

    zx1, zy1, zx2, zy2 = zone_bounds
    zone_w = zx2 - zx1
    zone_h = zy2 - zy1

    is_buck = sc.circuit_type == SubCircuitType.BUCK_CONVERTER  # type: ignore[union-attr]
    offsets = _BUCK_PASSIVE_OFFSETS if is_buck else _LDO_PASSIVE_OFFSETS
    y_frac = 0.40 if is_buck else 0.47
    # Buck IC at +90deg so PH/SW pin (pin 7) faces RIGHT toward inductor/output.
    # At -90deg the PH pin faces LEFT, forcing inductor placement against signal flow.
    # LDO at 180deg so VIN faces LEFT (input side) and VOUT_TAB faces RIGHT (output).
    ic_rot = 90.0 if is_buck else 180.0

    ic_x = zx1 + zone_w * x_fracs[reg_idx]
    ic_y = _clamp(zy1 + zone_h * y_frac, zy1 + 3.0, zy2 - 3.0)

    ctx.positions[ic_ref] = (ic_x, ic_y, ic_rot)
    ctx.power_group_fixed.add(ic_ref)

    power_group_refs = _collect_feature_refs(ctx, "power", "supply")
    ic_net_refs = _gather_ic_net_refs(
        ic_ref, set(sc.refs), ctx, power_group_refs,  # type: ignore[union-attr]
    )
    in_v, out_v = _estimate_regulator_voltages(ic_ref, ctx)
    placed_roles = _place_regulator_passives(
        ic_ref, ic_x, ic_y, ic_net_refs, offsets, zone_bounds, ctx, in_v, out_v,
    )

    _log.info(
        "    3c1b: placed %s (%s) at (%.1f, %.1f, %.0f) with %d passives (roles: %s)",
        ic_ref,
        "buck" if is_buck else "ldo",
        ic_x, ic_y, ic_rot,
        len(placed_roles),
        ", ".join(sorted(placed_roles)),
    )


def _nudge_connector_clear(
    j_ref: str,
    jx: float,
    jy: float,
    j_rot: float,
    ctx: PlacementContext,
    zone_bounds: tuple[float, float, float, float],
    max_attempts: int = 8,
) -> tuple[float, float]:
    """Nudge a connector position until it no longer collides with any placed component.

    Tries shifting in alternating Y then X directions. Returns the final (x, y).
    """
    zx1, zy1, zx2, zy2 = zone_bounds
    jw, jh = ctx.fp_sizes.get(j_ref, (2.5, 5.0))
    if j_rot % 180 in (90.0, 270.0):
        jw, jh = jh, jw

    clearance = 0.25  # mm

    for _attempt in range(max_attempts):
        collision_found = False
        for ref, (rx, ry, r_rot) in ctx.positions.items():
            if ref == j_ref:
                continue
            rw, rh = ctx.fp_sizes.get(ref, (2.0, 2.0))
            if r_rot % 180 in (90.0, 270.0):
                rw, rh = rh, rw

            overlap_x = (jw + rw) / 2.0 + clearance - abs(jx - rx)
            overlap_y = (jh + rh) / 2.0 + clearance - abs(jy - ry)

            if overlap_x > 0 and overlap_y > 0:
                # Collision detected — nudge in the direction of least overlap
                collision_found = True
                if overlap_y <= overlap_x:
                    # Nudge in Y
                    nudge = overlap_y + 0.5
                    jy = jy + nudge if jy >= ry else jy - nudge
                else:
                    # Nudge in X
                    nudge = overlap_x + 0.5
                    jx = jx + nudge if jx >= rx else jx - nudge
                jx = _clamp(jx, zx1 + 1.0, zx2 - 1.0)
                jy = _clamp(jy, zy1 + 1.0, zy2 - 1.0)
                break  # re-check all after nudge

        if not collision_found:
            break

    return jx, jy


def _place_power_connectors(
    ctx: PlacementContext,
    all_reg_scs: list[object],
    zone_bounds: tuple[float, float, float, float],
) -> None:
    """Place power-group connectors at learned reference-board positions.

    Only applied on dedicated power boards (<=2 features, <=4 connectors).
    On larger boards, _phase_top_edge_connectors handles connector placement.
    """
    zx1, zy1, zx2, zy2 = zone_bounds
    zone_w = zx2 - zx1
    zone_h = zy2 - zy1

    power_group_refs = _collect_feature_refs(ctx, "power", "supply")
    power_connectors = sorted(
        r for r in power_group_refs
        if r.startswith("J") and r in ctx.positions
    )
    is_power_focused_board = (
        len(ctx.requirements.features) <= 2 and len(power_connectors) <= 4
    )
    if not (power_connectors and all_reg_scs and is_power_focused_board):
        return

    # On a dedicated power board the signal-flow phase has authority over
    # connector positions.  Clear any fixed status set by the earlier
    # constraint-placement phase (which only enforced ordering, not absolute
    # positions) so we can place connectors at their correct signal-flow spots.
    for j_ref in power_connectors:
        ctx.fixed_refs.discard(j_ref)

    # index -> (x_frac, y_frac, rotation)
    # All positions are zone-relative fractions for left→right signal flow.
    # Connectors must be near board edges (within 8mm) for wire access.
    #   J1 (input 24V):  left edge, top area → wire entry faces top
    #   J2 (mid test):   between buck and LDO, below center
    #   J3 (output):     right edge, lower area → wire exit
    conn_rules: dict[int, tuple[float, float, float]] = {
        0: (0.10, 0.10, 0.0),     # J1: left edge, top area, wire entry faces top
        1: (0.50, 0.88, -90.0),   # J2: center, near bottom edge
        2: (0.97, 0.75, -90.0),   # J3: right edge, clear of zone right
    }
    for idx, j_ref in enumerate(power_connectors):
        if j_ref in ctx.fixed_refs:
            continue
        rule = conn_rules.get(idx)
        if rule is None:
            continue
        x_frac, y_frac, rot = rule
        jx = zx1 + zone_w * x_frac
        jy = zy1 + zone_h * y_frac
        clamped_x = _clamp(jx, zx1 + 1.0, zx2 - 1.0)
        clamped_y = _clamp(jy, zy1 + 1.0, zy2 - 1.0)
        # Respect placement constraints (proximity, ordering)
        from kicad_pipeline.optimization.constraint_guard import respect_constraints
        clamped_x, clamped_y, rot = respect_constraints(
            j_ref, clamped_x, clamped_y, rot, ctx,
        )
        # Nudge connector away from any component it would collide with
        clamped_x, clamped_y = _nudge_connector_clear(
            j_ref, clamped_x, clamped_y, rot, ctx, zone_bounds,
        )
        ctx.positions[j_ref] = (clamped_x, clamped_y, rot)
        ctx.power_group_fixed.add(j_ref)
        ctx.fixed_refs.add(j_ref)


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

    buck_scs = [sc for sc in ctx.subcircuits if sc.circuit_type == SubCircuitType.BUCK_CONVERTER]
    ldo_scs = [sc for sc in ctx.subcircuits if sc.circuit_type == SubCircuitType.LDO_REGULATOR]

    if not (buck_scs or ldo_scs):
        _log.info("    No power regulators found — skipping signal-flow phase")
        return

    all_reg_scs = buck_scs + ldo_scs
    zone_bounds = _resolve_power_zone_bounds(ctx, len(all_reg_scs))
    topology = compute_power_flow_topology(tuple(ctx.subcircuits))
    _sort_regulators_by_flow(all_reg_scs, ctx, topology)

    if not all_reg_scs:
        return

    x_fracs = _compute_regulator_x_fractions(len(all_reg_scs))

    for reg_idx, sc in enumerate(all_reg_scs):
        _place_power_chain_ic(sc, reg_idx, x_fracs, zone_bounds, ctx)

    _place_power_connectors(ctx, all_reg_scs, zone_bounds)

    _log.info(
        "    3c1b: signal-flow ordered %d regulators across power zone",
        len(all_reg_scs),
    )


def _pwr_place_column_down(
    ctx: PlacementContext,
    refs: list[str],
    col_x: float,
    start_y: float,
    placed_in_col: set[str],
    grid: object,
    pz_bounds: tuple[float, float, float, float],
    strip_gap: float,
) -> float:
    """Place *refs* in a vertical column going DOWN. Returns bottom Y."""
    pz_x1, pz_y1, pz_x2, pz_y2 = pz_bounds
    cy = start_y
    for ref in refs:
        if ref not in ctx.positions or ref in ctx.fixed_refs or ref in placed_in_col or ref == "":
            continue
        w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        px = max(pz_x1, min(pz_x2, col_x))
        py = max(pz_y1, min(pz_y2, cy + h / 2.0))
        ctx.positions[ref] = (px, py, 0.0)
        grid.place(px, py, w, h)
        ctx.power_group_fixed.add(ref)
        placed_in_col.add(ref)
        cy = py + h / 2.0 + strip_gap
    return cy


def _pwr_place_column_up(
    ctx: PlacementContext,
    refs: list[str],
    col_x: float,
    start_y: float,
    placed_in_col: set[str],
    grid: object,
    pz_bounds: tuple[float, float, float, float],
    strip_gap: float,
) -> float:
    """Place *refs* in a vertical column going UP. Returns top Y."""
    pz_x1, pz_y1, pz_x2, pz_y2 = pz_bounds
    cy = start_y
    for ref in refs:
        if ref not in ctx.positions or ref in ctx.fixed_refs or ref in placed_in_col or ref == "":
            continue
        w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        px = max(pz_x1, min(pz_x2, col_x))
        py = max(pz_y1, min(pz_y2, cy - h / 2.0))
        ctx.positions[ref] = (px, py, 0.0)
        grid.place(px, py, w, h)
        ctx.power_group_fixed.add(ref)
        placed_in_col.add(ref)
        cy = py - h / 2.0 - strip_gap
    return cy


def _pwr_place_fork_columns(
    ctx: PlacementContext,
    columns: dict[str, list[str]],
    anchor_x: float,
    sub_col_offset: float,
    col_spacing: float,
    output_top: float,
    placed_in_col: set[str],
    grid: object,
    pz_bounds: tuple[float, float, float, float],
    strip_gap: float,
) -> None:
    """Place output, bridge, buck2, and tail columns around the anchor IC."""
    left_x = anchor_x - sub_col_offset
    right_x = anchor_x + sub_col_offset

    def _col_down(refs: list[str], col_x: float, start_y: float) -> float:
        return _pwr_place_column_down(ctx, refs, col_x, start_y,
                                      placed_in_col, grid, pz_bounds, strip_gap)

    # Output columns
    left_bottom = _col_down(columns["output_left"], left_x, output_top)
    right_bottom = _col_down(columns["output_right"], right_x, output_top)

    # Bridge
    bridge_top = max(left_bottom, right_bottom)
    bridge_col = columns["bridge"]
    mid = len(bridge_col) // 2 + 1
    fork_y_l = _col_down(bridge_col[:mid], left_x, bridge_top)
    fork_y_r = _col_down(bridge_col[mid:], right_x, bridge_top)
    fork_y = max(fork_y_l, fork_y_r)

    # Buck #2 sub-columns
    col2_x = anchor_x + col_spacing
    col2_left_bottom = _col_down(columns["buck2_left"], col2_x, output_top)
    b2l_max_w = max(
        (ctx.fp_sizes.get(r, (2.0, 2.0))[0] for r in columns["buck2_left"] if r in ctx.fp_sizes),
        default=3.0,
    )
    b2r_max_w = max(
        (ctx.fp_sizes.get(r, (2.0, 2.0))[0] for r in columns["buck2_right"] if r in ctx.fp_sizes),
        default=2.0,
    )
    col2_right_x = col2_x + (b2l_max_w + b2r_max_w) / 2.0 + 0.5
    _col_down(columns["buck2_right"], col2_right_x, output_top)

    # Tail
    tail_y = max(fork_y, col2_left_bottom)
    tail_col = columns["tail"]
    mid_t = len(tail_col) // 2 + 1
    _col_down(tail_col[:mid_t], left_x, tail_y)
    _col_down(tail_col[mid_t:], right_x, tail_y)


def _pwr_place_anchor_ic(
    ctx: PlacementContext,
    buck1_ic: str,
    anchor_x: float,
    u1_y: float,
    u1_w: float,
    u1_h: float,
    placed_in_col: set[str],
    grid: object,
    pz_bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Place buck1 IC at its anchor position. Returns updated (u1_x, u1_y)."""
    pz_x1, pz_y1, pz_x2, pz_y2 = pz_bounds
    if not buck1_ic:
        return anchor_x, u1_y
    # Temporarily unfix — power group phase MUST be able to reposition the buck IC
    # to clear connector bodies and maintain proper power chain flow.
    ctx.fixed_refs.discard(buck1_ic)
    px = max(pz_x1, min(pz_x2, anchor_x))
    py = max(pz_y1, min(pz_y2, u1_y))
    ctx.positions[buck1_ic] = (px, py, 0.0)
    grid.place(px, py, u1_w, u1_h)  # type: ignore[union-attr]
    ctx.power_group_fixed.add(buck1_ic)
    ctx.fixed_refs.add(buck1_ic)  # re-fix after placement
    placed_in_col.add(buck1_ic)
    return px, py


def _phase_power_group(ctx: PlacementContext) -> None:
    """3c1: Power group organization — IC-anchored fork/branch layout."""
    _log.info("  3c1: Power group organization")

    power_group_refs = _collect_feature_refs(ctx, "power", "supply")
    if not power_group_refs:
        return

    power_zone_rect = _find_zone_rect(ctx, "power")
    net_to_pwr_refs = _build_net_to_group_refs(ctx, power_group_refs)
    by_prefix = _classify_refs_by_prefix(power_group_refs, ctx, "U", "J")
    power_ics, power_connectors = by_prefix["U"], by_prefix["J"]

    if not (power_ics and power_zone_rect is not None):
        return

    zx1, zy1, zx2, zy2 = power_zone_rect
    # Adaptive spacing derived from actual passive footprint heights
    passive_heights = [
        ctx.fp_sizes.get(r, (2.0, 2.0))[1]
        for r in power_group_refs if r[0] in "CRDL"
    ]
    avg_h = sum(passive_heights) / len(passive_heights) if passive_heights else 2.0
    strip_gap = max(0.3, avg_h * 0.3)   # 30% of avg height, min 0.3mm
    col_spacing = max(5.0, avg_h * 4.0)  # proportional column spacing
    sub_col_offset = max(2.0, avg_h * 1.75)  # proportional sub-column offset
    placed_in_col: set[str] = set()

    buck1_ic = power_ics[0] if power_ics else ""
    buck2_ic = power_ics[1] if len(power_ics) > 1 else ""
    columns = _classify_power_columns(
        ctx, power_group_refs, power_ics, power_connectors,
        net_to_pwr_refs, buck1_ic, buck2_ic,
    )

    u1_x, u1_y, _u1_rot = ctx.positions.get(buck1_ic, (zx1 + 5.0, zy1 + 15.0, 0.0))
    u1_w, u1_h = ctx.fp_sizes.get(buck1_ic, (5.0, 5.0))
    # Leave room for input connector (terminal block body ~13mm from left edge)
    # plus IC half-width (~4.5mm) plus gap (1mm) = 18.5mm minimum from left
    anchor_x = max(zx1 + 18.0, min(zx2 - col_spacing - 3.0, zx1 + (zx2 - zx1) * 0.35))
    pwr_grid = _build_exclusion_grid(ctx, power_group_refs)
    pz_bounds = (zx1 + 2.0, zy1 + 2.0, zx2 - 2.0, ctx.bounds[3] - 3.0)

    for ref in power_connectors:
        ctx.power_group_fixed.add(ref)

    u1_x, u1_y = _pwr_place_anchor_ic(
        ctx, buck1_ic, anchor_x, u1_y, u1_w, u1_h, placed_in_col, pwr_grid, pz_bounds,
    )
    _pwr_place_column_up(
        ctx, columns["vin_above"], anchor_x, u1_y - u1_h / 2.0 - strip_gap,
        placed_in_col, pwr_grid, pz_bounds, strip_gap,
    )
    _pwr_place_fork_columns(
        ctx, columns, anchor_x, sub_col_offset, col_spacing,
        u1_y + u1_h / 2.0 + strip_gap,
        placed_in_col, pwr_grid, pz_bounds, strip_gap,
    )

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


def _find_channel_connector_ref(
    passives: list[str],
    ctx: PlacementContext,
) -> str | None:
    """Find the connector ref (J*) connected to an ADC channel's passives."""
    for net in ctx.requirements.nets:
        j_refs = [c.ref for c in net.connections if c.ref.startswith("J")]
        r_in_ch = [c.ref for c in net.connections
                   if c.ref in passives and c.ref.startswith("R")]
        if j_refs and r_in_ch:
            return j_refs[0]
    return None


def _find_i2c_pullup_refs(
    ic_refs: set[str],
    ctx: PlacementContext,
    already_claimed: set[str],
) -> list[str]:
    """Find I2C pull-up resistors connected to ADC ICs via I2C nets."""
    pullup_refs: set[str] = set()
    for net in ctx.requirements.nets:
        name_upper = net.name.upper()
        if not any(tag in name_upper for tag in ("I2C", "SCL", "SDA")):
            continue
        has_ic = any(c.ref in ic_refs for c in net.connections)
        if not has_ic:
            continue
        for c in net.connections:
            if (c.ref.startswith("R")
                    and c.ref in ctx.positions
                    and c.ref not in already_claimed):
                pullup_refs.add(c.ref)
    return sorted(pullup_refs)


# Per-component dx offsets for horizontal strip below connector.
# Learned from human-routed reference board (60x40mm analog input board).
# Left-to-right order: C_filt, D_tvs, R_bot, R_top.
# (avg_dx_mm, per_channel_slope_dx, rotation_deg)
_ADC_STRIP_OFFSETS: dict[str, tuple[float, float, float]] = {
    "C_filt": (-4.64, 0.75, 90.0),
    "D_tvs":  (-1.77, 0.52, 0.0),
    "R_bot":  (+1.56, 0.57, 90.0),
    "R_top":  (+3.88, 0.76, -90.0),
}

# Vertical drop from connector to passive row (mm, before scaling).
_ADC_STRIP_DY_MM: float = 8.3


def _adc_assign_passive_role(
    ref: str, r_refs: list[str],
) -> str | None:
    """Map a passive ref to its role in the ADC channel strip.

    R refs are ordered numerically; the lower-numbered is R_top (top of
    voltage divider), the higher-numbered is R_bot.
    """
    if ref.startswith("D"):
        return "D_tvs"
    if ref.startswith("C"):
        return "C_filt"
    if ref.startswith("R") and ref in r_refs:
        idx = r_refs.index(ref)
        if idx == 0:
            return "R_top"
        if idx == 1:
            return "R_bot"
    return None


def _adc_init_ctx_defaults(ctx: PlacementContext) -> None:
    """Set ctx ADC attributes to empty defaults when no channels are detected."""
    ctx._adc_channels = []  # type: ignore[attr-defined]
    ctx._ic_channels = {}  # type: ignore[attr-defined]
    ctx._r_top_connector_x = {}  # type: ignore[attr-defined]
    ctx._occupied_x_ranges = []  # type: ignore[attr-defined]
    ctx._CHANNEL_SPACING_MM = 8.0  # type: ignore[attr-defined]
    ctx._STRIP_GAP_MM = 1.5  # type: ignore[attr-defined]


def _adc_group_by_ic(
    adc_channels: list[tuple[str, str, list[str]]],
    ctx: PlacementContext,
) -> dict[str, list[tuple[str, list[str]]]]:
    """Group ADC channels by IC ref and register IC refs on ctx."""
    ic_channels: dict[str, list[tuple[str, list[str]]]] = {}
    for ic_ref, ic_pin, passives in adc_channels:
        ic_channels.setdefault(ic_ref, []).append((ic_pin, passives))
        ctx.adc_ic_refs.add(ic_ref)
    return ic_channels


def _adc_pin_sort_key(ic_pin: str) -> tuple[int, str]:
    """Extract a numeric sort key from an ADC IC pin name.

    E.g. "AIN0" -> (0, "AIN0"), "AIN7" -> (7, "AIN7"), "A3" -> (3, "A3").
    Falls back to (999, pin_name) if no trailing digits are found.
    """
    import re

    m = re.search(r"(\d+)$", ic_pin)
    if m:
        return (int(m.group(1)), ic_pin)
    return (999, ic_pin)


def _adc_pre_move_to_zone(
    adc_channels: list[tuple[str, str, list[str]]],
    ctx: PlacementContext,
) -> int:
    """Move ADC-related components that are outside the analog zone to its center.

    Returns the number of components moved.
    """
    analog_zone = _find_zone_rect(ctx, "analog")
    if analog_zone is None:
        return 0

    az_x1, az_y1, az_x2, az_y2 = analog_zone
    center_x = (az_x1 + az_x2) / 2.0
    center_y = (az_y1 + az_y2) / 2.0
    moved = 0

    # Collect all refs involved in ADC channels (ICs, passives, connectors)
    all_adc_refs: set[str] = set()
    for ic_ref, _ic_pin, passives in adc_channels:
        all_adc_refs.add(ic_ref)
        all_adc_refs.update(passives)
        j_ref = _find_channel_connector_ref(passives, ctx)
        if j_ref:
            all_adc_refs.add(j_ref)

    for ref in all_adc_refs:
        if ref not in ctx.positions:
            continue
        rx, ry, rrot = ctx.positions[ref]
        if rx < az_x1 or rx > az_x2 or ry < az_y1 or ry > az_y2:
            ctx.positions[ref] = (center_x, center_y, rrot)
            moved += 1
            _log.debug(
                "    3c2: pre-moved %s from (%.1f, %.1f) to analog zone center",
                ref, rx, ry,
            )

    return moved


def _adc_build_channel_connector_list(
    adc_channels: list[tuple[str, str, list[str]]],
    ctx: PlacementContext,
) -> list[tuple[str | None, str, str, list[str]]]:
    """Build sorted (connector_ref, ic_ref, ic_pin, passives) list.

    Sorted by IC pin number (AIN0 < AIN1 < ... < AIN7) for deterministic
    left-to-right ordering.  Falls back to connector ref if pin numbers
    are identical.
    """
    channel_with_connectors: list[tuple[str | None, str, str, list[str]]] = [
        (_find_channel_connector_ref(passives, ctx), ic_ref, ic_pin, passives)
        for ic_ref, ic_pin, passives in adc_channels
    ]
    channel_with_connectors.sort(
        key=lambda t: (_adc_pin_sort_key(t[2]), t[0] or "Z999"),
    )
    return channel_with_connectors


def _adc_compute_zone_scale(
    ctx: PlacementContext,
) -> tuple[float, float, float, float, float, float]:
    """Return (az_x1, az_y1, az_x2, az_y2, sx, sy) from the analog zone."""
    analog_zone = _find_zone_rect(ctx, "analog")
    bounds = ctx.bounds
    az_x1 = analog_zone[0] if analog_zone else bounds[0]
    az_y1 = analog_zone[1] if analog_zone else bounds[1]
    az_x2 = analog_zone[2] if analog_zone else bounds[2]
    az_y2 = analog_zone[3] if analog_zone else bounds[3]
    zone_w = az_x2 - az_x1
    zone_h = az_y2 - az_y1
    sx = zone_w / 60.0  # scale relative to 60mm reference board
    sy = zone_h / 40.0  # scale relative to 40mm reference board
    return az_x1, az_y1, az_x2, az_y2, sx, sy


def _adc_place_connectors(
    channel_with_connectors: list[tuple[str | None, str, str, list[str]]],
    az_x1: float,
    az_y1: float,
    sx: float,
    sy: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> tuple[list[str], float]:
    """Place ADC channel connectors at the top edge of the analog zone.

    Returns (connector_refs, j_spacing_mm).
    """
    n_channels = len(channel_with_connectors)
    if n_channels > 1:
        j_x_start = az_x1 + 12.35 * sx
        j_x_end = az_x1 + 46.35 * sx
        j_spacing = (j_x_end - j_x_start) / (n_channels - 1)
    else:
        j_x_start = az_x1 + (az_x1 + 60.0 * sx - az_x1) * 0.5
        j_spacing = 0.0

    j_y = az_y1 + 4.5 * sy
    connector_refs: list[str] = []

    for ch_idx, (j_ref, _ic_ref, _ic_pin, _passives) in enumerate(
        channel_with_connectors,
    ):
        if j_ref is None or j_ref not in ctx.positions:
            continue
        j_x = j_x_start + ch_idx * j_spacing
        j_x_clamped, j_y_clamped = _clamp_to_bounds(j_x, j_y, bounds)
        # ADC connectors sit near top edge — wire entry faces outward (rot=0)
        ctx.positions[j_ref] = (j_x_clamped, j_y_clamped, 0.0)
        ctx.fixed_refs.add(j_ref)
        ctx.adc_channel_refs.add(j_ref)
        connector_refs.append(j_ref)
        _log.info(
            "    3c2: connector %s -> (%.1f, %.1f) rot=0",
            j_ref, j_x_clamped, j_y_clamped,
        )
    return connector_refs, j_spacing


def _adc_place_passive_strips(
    channel_with_connectors: list[tuple[str | None, str, str, list[str]]],
    sx: float,
    sy: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Place each channel's passives in a vertical strip below its connector.

    Vertical order (top to bottom): R_top, R_bot, D_tvs, C_filt.
    Each passive is spaced *_ADC_VERTICAL_STRIP_DY* mm apart vertically,
    centered on the connector's X position.
    """
    # Vertical spacing between passives within a strip (mm, before scaling)
    vertical_dy = 2.5 * sy
    # Vertical offset from connector to first passive
    strip_start_dy = _ADC_STRIP_DY_MM * sy
    # Vertical role order: signal flows from connector down through divider
    vertical_role_order: list[str] = ["R_top", "R_bot", "D_tvs", "C_filt"]
    vertical_role_rot: dict[str, float] = {
        "R_top": 90.0,
        "R_bot": 90.0,
        "D_tvs": 0.0,
        "C_filt": 90.0,
    }

    for ch_idx, (j_ref, _ic_ref, _ic_pin, passives) in enumerate(
        channel_with_connectors,
    ):
        if j_ref is None or j_ref not in ctx.positions:
            continue
        jx, jy, _jrot = ctx.positions[j_ref]
        r_refs = sorted(r for r in passives if r.startswith("R"))

        # Build role -> ref mapping for this channel
        role_to_ref: dict[str, str] = {}
        for ref in passives:
            if ref not in ctx.positions or ref in ctx.fixed_refs:
                continue
            role = _adc_assign_passive_role(ref, r_refs)
            if role is not None:
                role_to_ref[role] = ref

        # Place in vertical order below connector
        for slot_idx, role in enumerate(vertical_role_order):
            ref = role_to_ref.get(role)
            if ref is None:
                continue
            rot = vertical_role_rot.get(role, 0.0)
            new_y = jy + strip_start_dy + slot_idx * vertical_dy
            new_x, new_y_clamped = _clamp_to_bounds(jx, new_y, bounds)
            ctx.positions[ref] = (new_x, new_y_clamped, rot)
            ctx.adc_channel_refs.add(ref)
            ctx.fixed_refs.add(ref)

        _log.info("    3c2: ch%d passives placed as vertical strip below %s", ch_idx, j_ref)


def _adc_place_ics_and_decoupling(
    az_x1: float,
    az_y1: float,
    sx: float,
    sy: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Place ADC IC(s) and their decoupling caps below the channel strips."""
    ic_y = az_y1 + 28.58 * sy
    for ic_ref in sorted(ctx.adc_ic_refs):
        if ic_ref not in ctx.positions:
            continue
        ic_x = az_x1 + 45.75 * sx
        ic_x_clamped, ic_y_clamped = _clamp_to_bounds(ic_x, ic_y, bounds)
        ctx.positions[ic_ref] = (ic_x_clamped, ic_y_clamped, -90.0)
        ctx.fixed_refs.add(ic_ref)
        _log.info(
            "    3c2: ADC IC %s -> (%.1f, %.1f) rot=-90",
            ic_ref, ic_x_clamped, ic_y_clamped,
        )
        _adc_place_decoupling_caps(ic_ref, ic_x_clamped, ic_y_clamped, sy, bounds, ctx)


def _adc_place_decoupling_caps(
    ic_ref: str,
    ic_x: float,
    ic_y: float,
    sy: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Place decoupling caps connected to *ic_ref* directly below it."""
    for net in ctx.requirements.nets:
        if not any(c.ref == ic_ref for c in net.connections):
            continue
        for c in net.connections:
            if (c.ref.startswith("C")
                    and c.ref in ctx.positions
                    and c.ref not in ctx.adc_channel_refs
                    and c.ref not in ctx.fixed_refs):
                cap_x, cap_y = _clamp_to_bounds(ic_x, ic_y + 3.3 * sy, bounds)
                ctx.positions[c.ref] = (cap_x, cap_y, 180.0)
                ctx.adc_channel_refs.add(c.ref)
                ctx.fixed_refs.add(c.ref)
                _log.info("    3c2: ADC decoupling %s -> (%.1f, %.1f)", c.ref, cap_x, cap_y)


def _adc_place_i2c_pullups(
    sx: float,
    sy: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> list[str]:
    """Place I2C pull-up resistors near the first ADC IC.

    Returns the list of placed pull-up refs.
    """
    i2c_pullups = _find_i2c_pullup_refs(
        ctx.adc_ic_refs, ctx, ctx.adc_channel_refs | ctx.fixed_refs,
    )
    if not i2c_pullups or not ctx.adc_ic_refs:
        return i2c_pullups

    anchor_ic = sorted(ctx.adc_ic_refs)[0]
    if anchor_ic not in ctx.positions:
        return i2c_pullups

    aix, aiy, _airot = ctx.positions[anchor_ic]
    for pidx, pr_ref in enumerate(i2c_pullups):
        pr_x, pr_y = _clamp_to_bounds(
            aix + 7.0 * sx, aiy + (-6.0 + pidx * 3.0) * sy, bounds,
        )
        ctx.positions[pr_ref] = (pr_x, pr_y, 0.0)
        ctx.adc_channel_refs.add(pr_ref)
        ctx.fixed_refs.add(pr_ref)
        _log.info("    3c2: I2C pullup %s -> (%.1f, %.1f)", pr_ref, pr_x, pr_y)
    return i2c_pullups


def _adc_build_occupied_x_ranges(
    connector_refs: list[str],
    ctx: PlacementContext,
) -> list[tuple[float, float]]:
    """Return occupied X ranges for downstream phases based on connector positions."""
    if not connector_refs:
        return []
    placed = [ctx.positions[r][0] for r in connector_refs if r in ctx.positions]
    if not placed:
        return []
    return [(min(placed) - 5.0, max(placed) + 5.0)]


def _phase_adc_channels(ctx: PlacementContext) -> None:
    """3c2: ADC channel formation — vertical strip layout within analog zone.

    Layout pattern learned from human-routed reference board:

    1. Pre-move any ADC-related components outside the analog zone to the
       zone center so strip layout calculations use valid positions.

    2. Sort channels by IC pin number (AIN0 < AIN1 < ... < AIN7) for
       deterministic left-to-right ordering.

    3. Place ADC channel connectors at the top edge of the analog zone,
       evenly spaced left-to-right, rotation=0 (wire entry faces outward/top).

    4. For each channel, place passives in a vertical strip below the
       connector: C_filt, D_tvs, R_bot, R_top spaced vertically.

    5. Place ADC IC(s) below the channel strips, centered horizontally.

    6. Place I2C pull-up resistors near the ADC IC.
    """
    _log.info("  3c2: ADC channel formation (vertical strips)")

    adc_channels = _detect_adc_channels(ctx)
    if not adc_channels:
        _log.info("    3c2: no ADC channels detected")
        _adc_init_ctx_defaults(ctx)
        return

    # Pre-move scattered components into the analog zone before layout
    moved_count = _adc_pre_move_to_zone(adc_channels, ctx)
    if moved_count:
        _log.info("    3c2: pre-moved %d components into analog zone", moved_count)

    ic_channels = _adc_group_by_ic(adc_channels, ctx)
    channel_with_connectors = _adc_build_channel_connector_list(adc_channels, ctx)
    az_x1, az_y1, _az_x2, _az_y2, sx, sy = _adc_compute_zone_scale(ctx)
    bounds = ctx.bounds

    connector_refs, j_spacing = _adc_place_connectors(
        channel_with_connectors, az_x1, az_y1, sx, sy, bounds, ctx,
    )
    _adc_place_passive_strips(channel_with_connectors, sx, sy, bounds, ctx)
    _adc_place_ics_and_decoupling(az_x1, az_y1, sx, sy, bounds, ctx)
    i2c_pullups = _adc_place_i2c_pullups(sx, sy, bounds, ctx)

    channel_spacing_mm = j_spacing if j_spacing > 0 else 11.0
    _r_top_connector_x = _build_r_top_connector_x(ctx)
    _occupied_x_ranges = _adc_build_occupied_x_ranges(connector_refs, ctx)

    _log.info(
        "    3c2: %d channels across %d ICs, %d connectors placed, %d I2C pullups",
        len(adc_channels), len(ctx.adc_ic_refs), len(connector_refs), len(i2c_pullups),
    )

    # Store on ctx for use by _phase_adc_analog_cluster and late phases
    ctx._adc_channels = adc_channels  # type: ignore[attr-defined]
    ctx._ic_channels = ic_channels  # type: ignore[attr-defined]
    ctx._r_top_connector_x = _r_top_connector_x  # type: ignore[attr-defined]
    ctx._occupied_x_ranges = _occupied_x_ranges  # type: ignore[attr-defined]
    ctx._CHANNEL_SPACING_MM = channel_spacing_mm  # type: ignore[attr-defined]
    ctx._STRIP_GAP_MM = 1.5 * sy  # type: ignore[attr-defined]  # scale by zone factor


def _phase_adc_analog_cluster(ctx: PlacementContext) -> None:
    """3c3: Analog subcircuit clustering — pull remaining analog passives."""
    _log.info("  3c3: Analog subcircuit clustering")
    bounds = ctx.bounds

    channel_spacing_mm: float = getattr(ctx, "_CHANNEL_SPACING_MM", 8.0)
    occupied_x_ranges: list[tuple[float, float]] = getattr(ctx, "_occupied_x_ranges", [])
    other_group_refs = ctx.relay_support_refs | ctx.power_group_fixed

    all_analog_cluster_refs = _collect_adc_hop_refs(ctx, other_group_refs)

    if all_analog_cluster_refs and occupied_x_ranges:
        last_x_max = max(xmax for _, xmax in occupied_x_ranges)
        cluster_x = last_x_max + channel_spacing_mm + 2.0
        adc_ys = [ctx.positions[r][1] for r in ctx.adc_channel_refs if r in ctx.positions]
        cluster_y_top = min(adc_ys) - 1.0 if adc_ys else bounds[1] + 5.0
        sorted_cluster = sorted(
            all_analog_cluster_refs,
            key=lambda r: (0 if r.startswith("U") else 2, r),
        )
        placed_count = _place_refs_in_column_grid(
            sorted_cluster, cluster_x, cluster_y_top, ctx, ctx.adc_channel_refs,
        )
        _log.info("    3c3: clustered %d analog refs near ADC channels", placed_count)

    # 3c3b. Pull remaining analog group outliers toward the cluster.
    analog_group_refs = _collect_analog_group_refs(ctx)
    if analog_group_refs and ctx.adc_channel_refs:
        outlier_refs = _find_analog_outliers(ctx, analog_group_refs)
        if outlier_refs:
            adc_xs = [ctx.positions[r][0] for r in ctx.adc_channel_refs if r in ctx.positions]
            adc_ys = [ctx.positions[r][1] for r in ctx.adc_channel_refs if r in ctx.positions]
            outlier_x = max(adc_xs) + channel_spacing_mm + 2.0
            outlier_y_top = min(adc_ys) - 1.0
            cnt = _place_refs_in_column_grid(
                outlier_refs, outlier_x, outlier_y_top, ctx, ctx.adc_channel_refs,
            )
            _log.info("    3c3b: pulled %d outliers into analog cluster", cnt)


def _collect_adc_hop_refs(
    ctx: PlacementContext,
    other_group_refs: set[str],
) -> set[str]:
    """Collect analog refs connected to ADC ICs via up to 4 hops."""
    already_claimed = (
        ctx.adc_channel_refs | ctx.adc_ic_refs | ctx.fixed_refs | other_group_refs
    )

    analog_signal_refs: set[str] = set()
    for net in ctx.requirements.nets:
        if _is_power_or_bus_net(net.name):
            continue
        if not any(c.ref in ctx.adc_ic_refs and c.ref in ctx.positions
                   for c in net.connections):
            continue
        for c in net.connections:
            if (c.ref in ctx.positions
                    and c.ref not in already_claimed
                    and _is_small_passive(c.ref)):
                analog_signal_refs.add(c.ref)

    hop2_refs = _expand_one_hop(
        analog_signal_refs, ctx, already_claimed | analog_signal_refs, allow_small_ics=True,
    )
    small_ics_found = {r for r in hop2_refs if r.startswith("U")}
    hop3_refs = _expand_one_hop(
        small_ics_found, ctx, already_claimed | analog_signal_refs | hop2_refs,
    )
    hop4_refs = _expand_one_hop(
        hop3_refs, ctx, already_claimed | analog_signal_refs | hop2_refs | hop3_refs,
    )

    return (
        analog_signal_refs | hop2_refs | hop3_refs | hop4_refs
    ) - ctx.adc_channel_refs - other_group_refs


def _find_mcu_feature_group_refs(
    ctx: PlacementContext,
    mcu_ref: str,
) -> set[str]:
    """Return the set of refs in the FeatureBlock that contains *mcu_ref*."""
    for feat in ctx.requirements.features:
        feat_refs: set[str] = set()
        for comp in feat.components:
            r = comp.ref if hasattr(comp, "ref") else comp
            feat_refs.add(r)
        if mcu_ref in feat_refs:
            return feat_refs
    return set()


def _classify_mcu_group_refs(
    ctx: PlacementContext,
    mcu_ref: str,
    mcu_group_refs: set[str],
    ref_nets: dict[str, set[str]],
) -> tuple[set[str], list[str], list[str]]:
    """Partition MCU group into (connectors, decoupling_caps, other_passives)."""
    connector_refs = {r for r in mcu_group_refs if r.startswith("J")}
    # Build a set of net names per ref for signal/control cap detection
    ref_net_names: dict[str, set[str]] = {}
    for net in ctx.requirements.nets:
        for conn in net.connections:
            ref_net_names.setdefault(conn.ref, set()).add(net.name)
    _signal_keywords = {"EN", "DEB", "RESET", "BOOT", "LED", "UART", "SPI", "I2C"}

    decoupling_refs: list[str] = []
    other_passive_refs: list[str] = []
    for ref in sorted(mcu_group_refs):
        if ref == mcu_ref or ref in connector_refs or ref in ctx.fixed_refs:
            continue
        if ref not in ctx.positions:
            continue
        if ref.startswith("C") and mcu_ref in ref_nets.get(ref, set()):
            # Check if this cap is on a signal/control net (not power decoupling)
            non_gnd_nets = {n for n in ref_net_names.get(ref, set())
                           if "GND" not in n.upper()}
            is_signal_cap = any(
                kw in n.upper() for n in non_gnd_nets for kw in _signal_keywords
            )
            if is_signal_cap:
                other_passive_refs.append(ref)
            else:
                decoupling_refs.append(ref)
        else:
            other_passive_refs.append(ref)
    return connector_refs, decoupling_refs, other_passive_refs


def _mcu_push_courtyard_violations(
    ctx: PlacementContext,
    mcu_ref: str,
    mcu_x: float,
    mcu_y: float,
    eff_w: float,
    eff_h: float,
    mcu_grid: object,
) -> None:
    """Push peripheral refs that overlap the MCU courtyard outward."""
    court_margin = 2.0
    court = (
        mcu_x - eff_w / 2.0 - court_margin,
        mcu_y - eff_h / 2.0 - court_margin,
        mcu_x + eff_w / 2.0 + court_margin,
        mcu_y + eff_h / 2.0 + court_margin,
    )
    for ref in list(ctx.mcu_peripheral_refs):
        if ref == mcu_ref:
            continue
        # Skip refs already in fixed_refs — they were intentionally placed
        # (e.g. J1 at top edge, decoupling caps next to MCU pads).
        if ref in ctx.fixed_refs:
            continue
        if ref.startswith("J") and ctx.positions[ref][0] > mcu_x:
            continue
        _push_component_outside_courtyard(ref, ctx, court, mcu_grid)


def _mcu_place_debounce_caps(
    ctx: PlacementContext, other_passive_refs: list[str],
) -> None:
    ref_net_names: dict[str, set[str]] = {}
    for net in ctx.requirements.nets:
        for conn in net.connections:
            ref_net_names.setdefault(conn.ref, set()).add(net.name)

    for ref in list(other_passive_refs):
        if not ref.startswith("C") or ref not in ctx.positions:
            continue
        cap_nets = ref_net_names.get(ref, set())
        non_gnd = {n for n in cap_nets if "GND" not in n.upper()}
        is_debounce = any(
            kw in n.upper() for n in non_gnd for kw in ("EN", "DEB", "RESET")
        )
        if not is_debounce:
            continue
        sw_candidates = [r for r in ctx.positions
                         if r.startswith("SW") and r in ctx.mcu_peripheral_refs]
        if not sw_candidates:
            continue
        best_sw = sw_candidates[0]
        for sw in sw_candidates:
            sw_nets = ref_net_names.get(sw, set())
            sw_non_gnd = {n for n in sw_nets if "GND" not in n.upper()}
            for cn in non_gnd:
                for sn in sw_non_gnd:
                    if any(kw in cn.upper() and kw in sn.upper()
                           for kw in ("EN", "RESET", "BOOT")):
                        best_sw = sw
        tx, ty, _ = ctx.positions[best_sw]
        tw, th = ctx.fp_sizes.get(ref, (1.0, 0.5))
        sw_w, _sw_h = ctx.fp_sizes.get(best_sw, (4.0, 4.0))
        cx = tx + sw_w / 2.0 + tw / 2.0 + 0.5
        cy = ty
        cx = _clamp(cx, ctx.bounds[0] + tw / 2.0 + 0.5, ctx.bounds[2] - tw / 2.0 - 0.5)
        cy = _clamp(cy, ctx.bounds[1] + th / 2.0 + 0.5, ctx.bounds[3] - th / 2.0 - 0.5)
        ctx.positions[ref] = (cx, cy, 0.0)
        ctx.mcu_peripheral_refs.add(ref)
        ctx.fixed_refs.add(ref)
        other_passive_refs.remove(ref)
        _log.info("    %s (debounce) -> near %s at (%.1f, %.1f)", ref, best_sw, cx, cy)


def _phase_mcu_group(ctx: PlacementContext) -> None:
    """3c3: MCU peripheral tightening."""
    _log.info("  3c3: MCU peripheral tightening")

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
    # Skip if the detected MCU is owned by the ethernet feature block —
    # it's an ethernet IC (e.g. W5500) misidentified as MCU due to pin count.
    # The ethernet group phase handles its placement.
    _eth_refs_mcu = _collect_feature_refs(ctx, "ethernet", "eth")
    if mcu_ref_c3 in _eth_refs_mcu:
        _log.info("    3c3: skipping %s — belongs to ethernet group", mcu_ref_c3)
        return

    mcu_w, mcu_h = ctx.fp_sizes.get(mcu_ref_c3, (5.0, 5.0))
    mcu_group_refs = _find_mcu_feature_group_refs(ctx, mcu_ref_c3)

    # Step 0: Place U3 with antenna on BOTTOM board edge
    mcu_x, mcu_y, eff_w, eff_h, _mcu_rot = _mcu_place_u3(ctx, mcu_ref_c3, mcu_w, mcu_h)
    mcu_left = mcu_x - eff_w / 2.0
    mcu_top = mcu_y - eff_h / 2.0

    # Step 1: Build occupancy grid
    mcu_grid = _build_exclusion_grid(ctx, mcu_group_refs)
    mcu_grid.place(mcu_x, mcu_y, eff_w, eff_h)

    # Step 2: Build net adjacency and classify components
    ref_nets = _build_ref_net_adjacency(ctx, mcu_group_refs)
    connector_refs, decoupling_refs, other_passive_refs = _classify_mcu_group_refs(
        ctx, mcu_ref_c3, mcu_group_refs, ref_nets,
    )

    # Steps 3-6: place sub-groups
    _mcu_place_decoupling(ctx, decoupling_refs, mcu_x, mcu_y, mcu_left, mcu_grid)
    _mcu_place_connectors(ctx, connector_refs, mcu_grid, mcu_x, mcu_y, mcu_left, mcu_top, eff_w)
    _mcu_place_usb_subcircuit(ctx, other_passive_refs, mcu_grid)
    sw_base_x, sw_base_y = _mcu_place_reset_boot(
        ctx, other_passive_refs, ref_nets, mcu_grid, mcu_x, mcu_y, eff_w, mcu_top,
    )
    _mcu_place_led(ctx, other_passive_refs, mcu_grid, sw_base_x, sw_base_y)

    # Step 6c: Place EN/debounce caps near associated switch/resistor
    _mcu_place_debounce_caps(ctx, other_passive_refs)

    # Step 7: Place remaining passives sorted by proximity to MCU
    def _prox_key(ref: str) -> tuple[int, float]:
        connected = mcu_ref_c3 in ref_nets.get(ref, set())
        rx, ry, _ = ctx.positions[ref]
        return (0 if connected else 1, math.sqrt((rx - mcu_x) ** 2 + (ry - mcu_y) ** 2))

    remaining = sorted([r for r in other_passive_refs if r in ctx.positions], key=_prox_key)
    _mcu_place_remaining(ctx, remaining, mcu_ref_c3, mcu_x, mcu_y, eff_w, eff_h, mcu_grid)

    # Post-placement: push peripherals outside U3 courtyard
    _mcu_push_courtyard_violations(ctx, mcu_ref_c3, mcu_x, mcu_y, eff_w, eff_h, mcu_grid)

    _log.info(
        "    3c3: organized %d peripherals around %s at (%.1f, %.1f)",
        len(ctx.mcu_peripheral_refs), mcu_ref_c3, mcu_x, mcu_y,
    )


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
    _eth_place_crystal_and_caps(
        ctx, crystal_refs, crystal_load_caps,
        ic_cx, ic_cy, ic_h, eth_grid, placed_eth, eth_zone_rect,
    )
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


def _phase_ethernet_group(ctx: PlacementContext) -> None:
    """3c4: Ethernet group organization — horizontal left→right signal-chain.

    Layout follows the reference board signal flow:
        J1 (RJ45, left edge) → U1 (W5500, centre-right) → J2/J3 (headers, right edge)

    Crystal and decoupling caps cluster below U1 (centre of board height).
    """
    _log.info("  3c4: Ethernet group organization")

    eth_group_refs = _collect_feature_refs(ctx, "ethernet")
    if not eth_group_refs:
        return

    eth_zone_rect = _find_zone_rect(ctx, "ethernet")
    by_prefix = _classify_refs_by_prefix(eth_group_refs, ctx, "U", "J")
    eth_ics = by_prefix["U"]
    all_j_refs = by_prefix["J"]
    eth_connectors = [r for r in all_j_refs if ctx.fp_sizes.get(r, (0.0, 0.0))[0] > 10.0]
    eth_headers = [r for r in all_j_refs if r not in eth_connectors]

    if not (eth_ics and eth_zone_rect is not None):
        return

    ezx1, ezy1, ezx2, ezy2 = eth_zone_rect
    eth_net_refs = _build_net_to_group_refs(ctx, eth_group_refs)
    eth_grid = _build_exclusion_grid(ctx, eth_group_refs)
    eth_main_ic = eth_ics[0]
    crystal_refs = sorted([r for r in eth_group_refs if r.startswith("Y") and r in ctx.positions])
    poe_ic = eth_ics[1] if len(eth_ics) > 1 else ""
    crystal_load_caps, poe_caps, other_caps = _classify_eth_caps(
        eth_group_refs, eth_net_refs, crystal_refs, poe_ic, ctx,
    )
    eth_anchor_x = (ezx1 + ezx2) / 2.0
    zone_cy = (ezy1 + ezy2) / 2.0
    placed_eth: set[str] = set()

    _eth_place_rj45_connectors(
        ctx, eth_connectors, eth_anchor_x, eth_grid, placed_eth, eth_zone_rect,
    )
    ic_anchor_x = _eth_compute_ic_anchor_x(ctx, eth_connectors, placed_eth)
    ic_cx, ic_cy, ic_w, ic_h = _eth_force_place_main_ic(
        ctx, eth_main_ic, ic_anchor_x, zone_cy, eth_grid, placed_eth, eth_zone_rect,
    )
    _eth_place_headers_bottom(ctx, eth_headers, eth_grid, placed_eth, eth_zone_rect)
    col1_bottom = _eth_place_signal_chain(
        ctx, crystal_refs, crystal_load_caps, other_caps, poe_ic, poe_caps,
        eth_connectors, eth_anchor_x, ic_cx, ic_cy, ic_h,
        eth_grid, placed_eth, eth_zone_rect,
    )
    _eth_place_remaining(
        ctx, eth_group_refs, placed_eth, eth_anchor_x, col1_bottom, eth_grid, eth_zone_rect,
    )

    _log.info(
        "    3c4: organized %d ethernet components in signal chain: %s",
        len(ctx.ethernet_fixed), sorted(ctx.ethernet_fixed),
    )
    _eth_fix_crystal_cap_overlaps(ctx, placed_eth, eth_zone_rect, crystal_load_caps)
    ctx.fixed_refs.update(ctx.ethernet_fixed)
    _eth_push_non_group_clear(ctx)
