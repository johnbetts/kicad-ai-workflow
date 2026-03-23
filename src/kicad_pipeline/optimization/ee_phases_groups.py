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


def _phase_power_group(ctx: PlacementContext) -> None:
    """3c1: Power group organization — IC-anchored fork/branch layout."""
    _log.info("  3c1: Power group organization")
    bounds = ctx.bounds
    min_x, min_y, max_x, max_y = bounds

    # Find power FeatureBlock
    power_group_refs: set[str] = set()
    for feat in ctx.requirements.features:
        feat_lower = feat.name.lower()
        if "power" in feat_lower or "supply" in feat_lower:
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                power_group_refs.add(r)
            break

    if not power_group_refs:
        return

    # Find power zone
    power_zone_rect: tuple[float, float, float, float] | None = None
    for z in ctx.zones:
        if z.name == "power":
            power_zone_rect = z.rect
            break

    # Classify power components by net connectivity
    net_to_pwr_refs: dict[str, set[str]] = {}
    for net in ctx.requirements.nets:
        pwr_in_net = set()
        for conn in net.connections:
            if conn.ref in power_group_refs:
                pwr_in_net.add(conn.ref)
        if pwr_in_net:
            net_to_pwr_refs[net.name] = pwr_in_net

    power_ics = sorted(
        [r for r in power_group_refs
         if r.startswith("U") and r in ctx.positions],
    )
    power_connectors = sorted(
        [r for r in power_group_refs
         if r.startswith("J") and r in ctx.positions],
    )

    if not (power_ics and power_zone_rect is not None):
        return

    zx1, zy1, zx2, zy2 = power_zone_rect

    _STRIP_GAP = 0.5
    _COL_SPACING = 8.0
    _IC_MARGIN = 3.0

    placed_in_col: set[str] = set()

    buck1_ic = power_ics[0] if power_ics else ""
    buck2_ic = power_ics[1] if len(power_ics) > 1 else ""

    # Ferrite detection
    ferrite_refs = sorted(
        [r for r in power_group_refs
         if r.startswith("L") and r in ctx.positions
         and "ferrite" in (
             next((fp.value for fp in ctx.initial_pcb.footprints
                   if fp.ref == r), "")
         ).lower()],
    )

    # Buck #1 column: VIN input -> 5V output
    vin_passives = sorted(
        net_to_pwr_refs.get("VIN", set())
        - set(power_ics) - set(power_connectors),
    )
    bst1_passives = sorted(
        (net_to_pwr_refs.get("BST", set())
         | net_to_pwr_refs.get("SW", set()))
        - {buck1_ic} - set(power_connectors),
    )
    l1_refs = sorted(
        [r for r in power_group_refs
         if r.startswith("L") and r in ctx.positions
         and r not in ferrite_refs
         and r in net_to_pwr_refs.get("SW", set())],
    )
    fb_refs = sorted(
        net_to_pwr_refs.get("FB", set())
        - {buck1_ic} - set(power_connectors),
    )
    buck5v_caps = sorted(
        net_to_pwr_refs.get("BUCK_5V", set())
        - {buck1_ic} - set(power_connectors)
        - set(fb_refs) - set(l1_refs),
    )

    vin_above: list[str] = vin_passives
    output_left: list[str] = bst1_passives + l1_refs + buck5v_caps
    output_right: list[str] = fb_refs

    # Bridge: OR'ing diodes + 5V bulk cap
    or_diode_refs = sorted(
        net_to_pwr_refs.get("BUCK_5V", set())
        & {r for r in power_group_refs if r.startswith("D")}
        - set(vin_passives),
    )
    v5_rail_refs = sorted(
        (net_to_pwr_refs.get("+5V", set()) & power_group_refs)
        - set(power_ics) - set(power_connectors)
        - set(or_diode_refs),
    )

    # Buck #2 column: +5V input -> 3.3V output
    _v5_caps_in_power = sorted(
        net_to_pwr_refs.get("+5V", set())
        & {r for r in power_group_refs if r.startswith("C")}
        - {"C4"},
    )
    buck2_in_caps = _v5_caps_in_power
    v5_rail_refs = [r for r in v5_rail_refs if r not in buck2_in_caps]
    bridge_column: list[str] = or_diode_refs + v5_rail_refs
    bst2_passives = sorted(
        (net_to_pwr_refs.get("BST2", set())
         | net_to_pwr_refs.get("SW2", set()))
        - {buck2_ic} - set(power_connectors),
    )
    l2_refs = sorted(
        [r for r in power_group_refs
         if r.startswith("L") and r in ctx.positions
         and r not in ferrite_refs
         and r in net_to_pwr_refs.get("SW2", set())],
    )
    v33_caps = sorted(
        net_to_pwr_refs.get("+3V3", set())
        & power_group_refs
        - {buck2_ic} - set(power_connectors),
    )

    buck2_left: list[str] = (
        buck2_in_caps + [buck2_ic] + bst2_passives + l2_refs
    )
    buck2_right: list[str] = v33_caps

    # Tail: LED, ferrites, remaining
    led_refs = sorted(
        (net_to_pwr_refs.get("LED_A", set()) & power_group_refs)
        - set(power_connectors),
    )

    all_classified = (
        set(vin_above) | {buck1_ic}
        | set(output_left) | set(output_right)
        | set(bridge_column) | set(buck2_left) | set(buck2_right)
        | set(led_refs) | set(ferrite_refs) | set(power_connectors)
    )
    remaining_refs = sorted(
        power_group_refs - all_classified - {""} - ctx.fixed_refs,
    )
    tail_column: list[str] = led_refs + ferrite_refs + remaining_refs

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
    _pwr_grid = _PlacementGrid(bounds)
    for _oref, (_ox, _oy, _or) in ctx.positions.items():
        if _oref in power_group_refs:
            continue
        _ow, _oh = _rotation_aware_size(_oref, ctx.positions, ctx.fp_sizes)
        _pwr_grid.place(_ox, _oy, _ow, _oh)

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


def _phase_adc_channels(ctx: PlacementContext) -> None:
    """3c2: ADC channel formation — repeatable channel strips near ADC ICs."""
    _log.info("  3c2: ADC channel formation")
    bounds = ctx.bounds

    # Build net -> refs mapping from requirements
    net_components: dict[str, list[tuple[str, str]]] = {}
    for net in ctx.requirements.nets:
        net_components[net.name] = [(c.ref, c.pin) for c in net.connections]

    # Find ADC channel nets: connect U* pin to 2R + 1D + 1C
    adc_channels: list[tuple[str, str, list[str]]] = []
    for net_name, conns in net_components.items():
        ic_refs = [(r, p) for r, p in conns if r.startswith("U") and r in ctx.positions]
        passive_refs = [r for r, p in conns
                        if r in ctx.positions and r[0] in "RDC" and not r.startswith("U")]
        if len(ic_refs) == 1 and len(passive_refs) == 4:
            r_count = sum(1 for r in passive_refs if r.startswith("R"))
            d_count = sum(1 for r in passive_refs if r.startswith("D"))
            c_count = sum(1 for r in passive_refs if r.startswith("C"))
            if r_count == 2 and d_count == 1 and c_count == 1:
                ic_ref, ic_pin = ic_refs[0]
                adc_channels.append((ic_ref, ic_pin, passive_refs))

    # Group channels by IC
    ic_channels: dict[str, list[tuple[str, list[str]]]] = {}
    for ic_ref, ic_pin, passives in adc_channels:
        ic_channels.setdefault(ic_ref, []).append((ic_pin, passives))

    # Build R_top ref -> connector X position
    _r_top_connector_x: dict[str, float] = {}
    for net in ctx.requirements.nets:
        j_refs = [c for c in net.connections if c.ref.startswith("J")]
        r_refs = [c for c in net.connections
                  if c.ref.startswith("R") and c.ref in ctx.positions]
        if j_refs and r_refs:
            for j_conn in j_refs:
                j_pos = ctx.positions.get(j_conn.ref)
                if j_pos:
                    for r_conn in r_refs:
                        _r_top_connector_x[r_conn.ref] = j_pos[0]

    def _channel_sort_key(ch: tuple[str, list[str]]) -> float:
        _pin, passives = ch
        r_refs_ch = sorted(r for r in passives if r.startswith("R"))
        for r in r_refs_ch:
            if r in _r_top_connector_x:
                return _r_top_connector_x[r]
        return float(hash(_pin)) * 1e-6

    for ic_ref in ic_channels:
        ic_channels[ic_ref].sort(key=_channel_sort_key)

    # First pass: collect all ADC channel passive refs and IC refs
    all_adc_passive_refs: set[str] = set()
    for _ic_ref, _ic_pin, passives in adc_channels:
        all_adc_passive_refs.update(passives)
        ctx.adc_ic_refs.add(_ic_ref)

    # Move ADC ICs to the analog zone
    analog_zone = None
    for z in ctx.zones:
        if z.name == "analog":
            analog_zone = z
            break

    def _ic_avg_connector_x(ic: str) -> float:
        xs: list[float] = []
        for _pin, passives in ic_channels.get(ic, []):
            for r in passives:
                if r.startswith("R") and r in _r_top_connector_x:
                    xs.append(_r_top_connector_x[r])
        return sum(xs) / len(xs) if xs else 999.0

    if analog_zone and ctx.adc_ic_refs:
        az_x1, az_y1, az_x2, az_y2 = analog_zone.rect
        ic_list = sorted(ctx.adc_ic_refs, key=_ic_avg_connector_x)
        n_ics = len(ic_list)
        ic_spacing = (az_x2 - az_x1) / (n_ics + 1)
        for idx, ic_ref in enumerate(ic_list):
            if ic_ref not in ctx.positions:
                continue
            iw, ih = ctx.fp_sizes.get(ic_ref, (5.0, 5.0))
            new_x = az_x1 + ic_spacing * (idx + 1)
            new_y = az_y1 + (az_y2 - az_y1) * 0.70
            ctx.positions[ic_ref] = (new_x, new_y, 0.0)
            ctx.fixed_refs.add(ic_ref)
            _log.info(
                "    3c2: moved %s to analog zone (%.1f, %.1f)",
                ic_ref, new_x, new_y,
            )

    # Build occupancy grid WITHOUT ADC channel passives
    adc_grid = _PlacementGrid(bounds)
    for oref, (ox, oy, _orot) in ctx.positions.items():
        if oref in all_adc_passive_refs:
            continue
        ow, oh = _rotation_aware_size(oref, ctx.positions, ctx.fp_sizes)
        adc_grid.place(ox, oy, ow, oh)

    _CHANNEL_SPACING_MM = 8.0
    _STRIP_GAP_MM = 1.5

    sorted_ic_refs = sorted(ic_channels.keys(), key=_ic_avg_connector_x)
    _occupied_x_ranges: list[tuple[float, float]] = []

    for ic_ref in sorted_ic_refs:
        ch_list = ic_channels[ic_ref]
        if ic_ref not in ctx.positions:
            continue
        ix, iy, _irot = ctx.positions[ic_ref]
        iw, ih = ctx.fp_sizes.get(ic_ref, (5.0, 5.0))

        n_ch = len(ch_list)
        total_ch_width = (n_ch - 1) * _CHANNEL_SPACING_MM
        ch_x_start = ix - total_ch_width / 2.0
        ch_x_end = ch_x_start + total_ch_width

        _comp_half_w = 2.0
        for ox_min, ox_max in _occupied_x_ranges:
            if (ch_x_start - _comp_half_w < ox_max + _comp_half_w
                    and ch_x_end + _comp_half_w > ox_min - _comp_half_w):
                shift = (ox_max + _comp_half_w + _CHANNEL_SPACING_MM) - ch_x_start
                if shift > 0:
                    ch_x_start += shift
                    ch_x_end = ch_x_start + total_ch_width

        _occupied_x_ranges.append((ch_x_start, ch_x_end))

        for ch_idx, (ic_pin, passives) in enumerate(ch_list):
            r_refs = sorted([r for r in passives if r.startswith("R")])
            d_refs = [r for r in passives if r.startswith("D")]
            c_refs = [r for r in passives if r.startswith("C")]

            ch_x = ch_x_start + ch_idx * _CHANNEL_SPACING_MM

            strip_order: list[str] = []
            if len(r_refs) >= 2:
                strip_order.append(r_refs[1])
            strip_order.extend(d_refs)
            strip_order.extend(c_refs)
            if len(r_refs) >= 1:
                strip_order.append(r_refs[0])

            strip_y = iy - ih / 2.0 - 1.0

            for ref in strip_order:
                if ref not in ctx.positions or ref in ctx.fixed_refs:
                    continue
                raw_w, raw_h = ctx.fp_sizes.get(ref, (2.0, 2.0))
                w, h = raw_w, raw_h
                rot = 0.0
                target_x = ch_x
                target_y = strip_y - h / 2.0
                target_x = max(bounds[0] + 2.0, min(bounds[2] - 2.0, target_x))
                target_y = max(bounds[1] + 2.0, min(bounds[3] - 2.0, target_y))
                ctx.positions[ref] = (target_x, target_y, rot)
                ctx.adc_channel_refs.add(ref)
                strip_y = target_y - (h / 2.0 + _STRIP_GAP_MM)

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

    _POWER_NET_PREFIXES = (
        "VCC", "VDD", "GND", "AGND", "DGND", "AVCC", "+3V3", "+5V",
        "VIN", "VBUS", "V_", "+12V", "+24V", "I2C_",
    )

    def _is_power_or_bus_net(name: str) -> bool:
        upper = name.upper()
        return any(upper.startswith(p) for p in _POWER_NET_PREFIXES)

    _SMALL_PREFIXES = ("R", "C", "D", "L", "LED", "SW")

    def _is_small_passive(ref: str) -> bool:
        return any(ref.startswith(p) for p in _SMALL_PREFIXES)

    # Find small passives on ADC signal nets
    analog_signal_refs: set[str] = set()
    for net in ctx.requirements.nets:
        if _is_power_or_bus_net(net.name):
            continue
        ic_conn = [c for c in net.connections
                   if c.ref in ctx.adc_ic_refs and c.ref in ctx.positions]
        if not ic_conn:
            continue
        for c in net.connections:
            if (c.ref in ctx.positions
                    and c.ref not in ctx.adc_channel_refs
                    and c.ref not in ctx.adc_ic_refs
                    and c.ref not in ctx.fixed_refs
                    and c.ref not in ctx.relay_support_refs
                    and c.ref not in ctx.power_group_fixed
                    and _is_small_passive(c.ref)):
                analog_signal_refs.add(c.ref)

    _other_group_refs = ctx.relay_support_refs | ctx.power_group_fixed

    # One hop
    hop2_refs: set[str] = set()
    for ref in list(analog_signal_refs):
        for net in ctx.requirements.nets:
            if _is_power_or_bus_net(net.name):
                continue
            if not any(c.ref == ref for c in net.connections):
                continue
            for c in net.connections:
                if (c.ref != ref and c.ref in ctx.positions
                        and c.ref not in ctx.adc_channel_refs
                        and c.ref not in ctx.adc_ic_refs
                        and c.ref not in ctx.fixed_refs
                        and c.ref not in analog_signal_refs
                        and c.ref not in _other_group_refs):
                    if _is_small_passive(c.ref):
                        hop2_refs.add(c.ref)
                    elif c.ref.startswith("U"):
                        comp = next((comp for comp in ctx.requirements.components
                                     if comp.ref == c.ref), None)
                        if comp and len(comp.pins) <= 6:
                            hop2_refs.add(c.ref)

    # Hop 3
    hop3_refs: set[str] = set()
    small_ics_found = {r for r in hop2_refs if r.startswith("U")}
    for ic_ref in small_ics_found:
        for net in ctx.requirements.nets:
            if _is_power_or_bus_net(net.name):
                continue
            if not any(c.ref == ic_ref for c in net.connections):
                continue
            for c in net.connections:
                if (c.ref != ic_ref and c.ref in ctx.positions
                        and c.ref not in ctx.adc_channel_refs
                        and c.ref not in ctx.adc_ic_refs
                        and c.ref not in ctx.fixed_refs
                        and c.ref not in _other_group_refs
                        and _is_small_passive(c.ref)):
                    hop3_refs.add(c.ref)

    # Hop 4
    hop4_refs: set[str] = set()
    for ref in hop3_refs:
        for net in ctx.requirements.nets:
            if _is_power_or_bus_net(net.name):
                continue
            if not any(c.ref == ref for c in net.connections):
                continue
            for c in net.connections:
                if (c.ref != ref and c.ref in ctx.positions
                        and c.ref not in ctx.adc_channel_refs
                        and c.ref not in ctx.adc_ic_refs
                        and c.ref not in ctx.fixed_refs
                        and c.ref not in _other_group_refs
                        and _is_small_passive(c.ref)):
                    hop4_refs.add(c.ref)

    all_analog_cluster_refs = (
        analog_signal_refs | hop2_refs | hop3_refs | hop4_refs
    ) - ctx.adc_channel_refs - _other_group_refs

    if all_analog_cluster_refs and _occupied_x_ranges:
        last_x_max = max(xmax for _, xmax in _occupied_x_ranges)
        cluster_x = last_x_max + _CHANNEL_SPACING_MM + 2.0

        adc_ys = [ctx.positions[r][1] for r in ctx.adc_channel_refs if r in ctx.positions]
        cluster_y_top = min(adc_ys) - 1.0 if adc_ys else bounds[1] + 5.0

        sorted_cluster = sorted(
            all_analog_cluster_refs,
            key=lambda r: (0 if r.startswith("U") else 2, r),
        )

        _COL_GAP = 5.0
        _ROW_GAP = 1.0
        cur_x = cluster_x
        cur_y = cluster_y_top
        placed_count = 0

        for ref in sorted_cluster:
            if ref not in ctx.positions:
                continue
            raw_w, raw_h = ctx.fp_sizes.get(ref, (2.0, 2.0))
            w, h = raw_w, raw_h
            rot = 0.0

            target_x = cur_x
            target_y = cur_y + h / 2.0
            target_x = max(bounds[0] + 2.0, min(bounds[2] - 2.0, target_x))
            target_y = max(bounds[1] + 2.0, min(bounds[3] - 2.0, target_y))

            ctx.positions[ref] = (target_x, target_y, rot)
            ctx.adc_channel_refs.add(ref)
            placed_count += 1

            cur_y = target_y + h / 2.0 + _ROW_GAP

            if cur_y > cluster_y_top + 15.0:
                cur_x += _COL_GAP
                cur_y = cluster_y_top

        _log.info("    3c3: clustered %d analog refs near ADC channels", placed_count)

    # 3c3b. Pull remaining analog group outliers toward the cluster.
    analog_group_refs: set[str] = set()
    for feat in ctx.requirements.features:
        feat_refs = set(feat.components)
        if feat_refs & ctx.adc_ic_refs:
            analog_group_refs = feat_refs
            break

    if analog_group_refs and ctx.adc_channel_refs:
        placed_analog = [
            ctx.positions[r] for r in (ctx.adc_channel_refs | ctx.adc_ic_refs)
            if r in ctx.positions
        ]
        if placed_analog:
            cx = sum(p[0] for p in placed_analog) / len(placed_analog)
            cy = sum(p[1] for p in placed_analog) / len(placed_analog)
            outlier_refs: list[str] = []
            for ref in sorted(analog_group_refs):
                if (ref in ctx.positions
                        and ref not in ctx.adc_channel_refs
                        and ref not in ctx.adc_ic_refs
                        and ref not in ctx.fixed_refs
                        and not ref.startswith("J")):
                    rx, ry, _rrot = ctx.positions[ref]
                    dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
                    if dist > 15.0:
                        outlier_refs.append(ref)

            if outlier_refs:
                adc_xs = [ctx.positions[r][0] for r in ctx.adc_channel_refs
                          if r in ctx.positions]
                adc_ys = [ctx.positions[r][1] for r in ctx.adc_channel_refs
                          if r in ctx.positions]
                outlier_x = max(adc_xs) + _CHANNEL_SPACING_MM + 2.0
                outlier_y_top = min(adc_ys) - 1.0
                oc_x = outlier_x
                oc_y = outlier_y_top
                _OC_COL_GAP = 5.0
                _OC_ROW_GAP = 1.0

                for ref in outlier_refs:
                    raw_w, raw_h = ctx.fp_sizes.get(ref, (2.0, 2.0))
                    w, h = raw_w, raw_h
                    rot = 0.0

                    target_x = oc_x
                    target_y = oc_y + h / 2.0
                    target_x = max(bounds[0] + 2.0, min(bounds[2] - 2.0, target_x))
                    target_y = max(bounds[1] + 2.0, min(bounds[3] - 2.0, target_y))

                    ctx.positions[ref] = (target_x, target_y, rot)
                    ctx.adc_channel_refs.add(ref)

                    oc_y = target_y + h / 2.0 + _OC_ROW_GAP
                    if oc_y > outlier_y_top + 15.0:
                        oc_x += _OC_COL_GAP
                        oc_y = outlier_y_top

                _log.info(
                    "    3c3b: pulled %d outliers into analog cluster",
                    len(outlier_refs),
                )


def _phase_mcu_group(ctx: PlacementContext) -> None:
    """3c3: MCU peripheral tightening."""
    _log.info("  3c3: MCU peripheral tightening")
    bounds = ctx.bounds

    from kicad_pipeline.optimization.functional_grouper import _find_mcu_ref as _find_mcu
    mcu_ref_c3 = _find_mcu(ctx.requirements)
    if not (mcu_ref_c3 and mcu_ref_c3 in ctx.positions):
        return

    mcu_x, mcu_y, _mcu_rot = ctx.positions[mcu_ref_c3]
    mcu_w, mcu_h = ctx.fp_sizes.get(mcu_ref_c3, (5.0, 5.0))

    # Find MCU's FeatureBlock group
    mcu_group_refs: set[str] = set()
    for feat in ctx.requirements.features:
        feat_refs = set()
        for comp in feat.components:
            r = comp.ref if hasattr(comp, "ref") else comp
            feat_refs.add(r)
        if mcu_ref_c3 in feat_refs:
            mcu_group_refs = feat_refs
            break

    # --- Step 0: Place U3 with antenna on BOTTOM board edge ---
    mcu_fp = None
    for fp in ctx.initial_pcb.footprints:
        if fp.ref == mcu_ref_c3:
            mcu_fp = fp
            break
    mcu_zone_rect: tuple[float, float, float, float] | None = None
    for z in ctx.zones:
        if z.name == "mcu":
            mcu_zone_rect = z.rect
            break
    _mcu_rot = 180.0
    eff_w, eff_h = mcu_w, mcu_h

    if mcu_fp is not None:
        from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space as _pad_ext
        _trial_ox = (bounds[0] + bounds[2]) / 2.0
        _trial_oy = (bounds[1] + bounds[3]) / 2.0
        _te = _pad_ext(mcu_fp, _trial_ox, _trial_oy, _mcu_rot)
        _pad_bot = _te[3] - _trial_oy
        mcu_origin_y = bounds[3] - _pad_bot - 2.0
        if mcu_zone_rect is not None:
            _zx1, _zy1, _zx2, _zy2 = mcu_zone_rect
            _j14_w = ctx.fp_sizes.get("J14", (2.7, 35.7))[0] if "J14" in mcu_group_refs else 0.0
            mcu_origin_x = (_zx1 + _zx2 - _j14_w) / 2.0
        else:
            mcu_origin_x = bounds[2] - eff_w / 2.0 - 10.0
        _pad_left = _te[0] - _trial_ox
        _pad_right = _te[2] - _trial_ox
        mcu_origin_x = max(bounds[0] - _pad_left + 2.0,
                           min(bounds[2] - _pad_right - 2.0, mcu_origin_x))
        mcu_x, mcu_y = origin_to_centroid(mcu_fp, mcu_origin_x,
                                           mcu_origin_y, _mcu_rot)
    else:
        if mcu_zone_rect is not None:
            _zx1, _zy1, _zx2, _zy2 = mcu_zone_rect
            mcu_x = (_zx1 + _zx2) / 2.0
        else:
            mcu_x = bounds[2] - eff_w / 2.0 - 10.0
        mcu_y = bounds[3] - eff_h / 2.0 - 5.0

    mcu_y = min(mcu_y, bounds[3] - eff_h / 2.0 - 5.0)
    mcu_y = max(mcu_y, bounds[1] + eff_h / 2.0 + 2.0)
    mcu_x = min(mcu_x, bounds[2] - eff_w / 2.0 - 2.0)
    mcu_x = max(mcu_x, bounds[0] + eff_w / 2.0 + 2.0)
    ctx.positions[mcu_ref_c3] = (mcu_x, mcu_y, _mcu_rot)
    ctx.mcu_peripheral_refs.add(mcu_ref_c3)
    ctx.fixed_refs.add(mcu_ref_c3)
    _log.info("    U3 centroid at (%.1f, %.1f) rot=180 [eff_w=%.1f, eff_h=%.1f]",
              mcu_x, mcu_y, eff_w, eff_h)

    mcu_left = mcu_x - eff_w / 2.0
    mcu_top = mcu_y - eff_h / 2.0
    mcu_bot = mcu_y + eff_h / 2.0
    mcu_right = mcu_x + eff_w / 2.0

    # --- Step 1: Build occupancy grid ---
    mcu_grid = _PlacementGrid(bounds)
    for oref, (ox, oy, _or) in ctx.positions.items():
        if oref in mcu_group_refs:
            continue
        ow, oh = _rotation_aware_size(oref, ctx.positions, ctx.fp_sizes)
        mcu_grid.place(ox, oy, ow, oh)
    mcu_grid.place(mcu_x, mcu_y, eff_w, eff_h)

    # --- Step 2: Build net adjacency ---
    ref_nets: dict[str, set[str]] = {}
    for net in ctx.requirements.nets:
        net_refs = set()
        for conn in net.connections:
            net_refs.add(conn.ref)
        for r in net_refs:
            if r in mcu_group_refs:
                ref_nets.setdefault(r, set()).update(
                    net_refs & mcu_group_refs,
                )

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
    max_cap_w = max((ctx.fp_sizes.get(r, (2.5, 1.5))[0] for r in decoupling_refs),
                    default=2.5)
    decoup_col_x = mcu_left - max_cap_w / 2.0 - 2.0
    total_cap_h = sum(ctx.fp_sizes.get(r, (1.5, 1.0))[1] + 1.5
                      for r in decoupling_refs)
    decoup_y = mcu_y - total_cap_h / 2.0
    for ref in decoupling_refs:
        w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
        tx = decoup_col_x
        ty = decoup_y + h / 2.0
        tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
        ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
        px, py = mcu_grid.find_free_pos(tx, ty, w, h, max_radius=8.0)
        ctx.positions[ref] = (px, py, 0.0)
        ctx.mcu_peripheral_refs.add(ref)
        mcu_grid.place(px, py, w, h)
        decoup_y += h + 1.5
        dist_to_mcu = ((px - mcu_x) ** 2 + (py - mcu_y) ** 2) ** 0.5
        _log.info("    %s (decoupling): ->(%.1f,%.1f) [%.1fmm from U3]",
                  ref, px, py, dist_to_mcu)

    # --- Step 4: Place connectors ---
    right_edge_x = bounds[2] - 2.0

    if "J14" in connector_refs and "J14" in ctx.positions and "J14" not in ctx.fixed_refs:
        w14, h14 = ctx.fp_sizes.get("J14", (2.7, 35.7))
        tx = bounds[2] - w14 / 2.0 - 1.0
        ty = mcu_y
        ty = max(bounds[1] + h14 / 2.0 + 1.0,
                 min(bounds[3] - h14 / 2.0 - 1.0, ty))
        px14, py14 = mcu_grid.find_free_pos(tx, ty, w14, h14,
                                             max_radius=25.0)
        ctx.positions["J14"] = (px14, py14, 0.0)
        mcu_grid.place(px14, py14, w14, h14)
        ctx.mcu_peripheral_refs.add("J14")
        _log.info("    J14 -> right edge (%.1f, %.1f)", px14, py14)

    if "J15" in connector_refs and "J15" in ctx.positions and "J15" not in ctx.fixed_refs:
        w15, h15 = ctx.fp_sizes.get("J15", (5.2, 12.9))
        j14_pos = ctx.positions.get("J14")
        if j14_pos:
            j14_bottom = j14_pos[1] + ctx.fp_sizes.get("J14", (2.7, 35.7))[1] / 2.0
            tx = right_edge_x - w15 / 2.0
            ty = j14_bottom + h15 / 2.0 + 2.0
        else:
            tx = right_edge_x - w15 / 2.0
            ty = mcu_y + 10.0
        ty = max(bounds[1] + h15 / 2.0 + 1.5,
                 min(bounds[3] - h15 / 2.0 - 1.5, ty))
        tx = min(tx, bounds[2] - w15 / 2.0 - 1.5)
        px15, py15 = mcu_grid.find_free_pos(tx, ty, w15, h15, max_radius=25.0)
        px15 = min(px15, bounds[2] - w15 / 2.0 - 1.5)
        py15 = min(py15, bounds[3] - h15 / 2.0 - 1.5)
        ctx.positions["J15"] = (px15, py15, 0.0)
        mcu_grid.place(px15, py15, w15, h15)
        ctx.mcu_peripheral_refs.add("J15")
        _log.info("    J15 -> right edge, below J14 (%.1f, %.1f)", px15, py15)

    if "J16" in connector_refs and "J16" in ctx.positions and "J16" not in ctx.fixed_refs:
        w16, h16 = ctx.fp_sizes.get("J16", (16.2, 6.9))
        px = bounds[2] - w16 / 2.0
        py = mcu_top - h16 / 2.0 - 2.0
        py = max(bounds[1] + h16 / 2.0 + 1.0,
                 min(bounds[3] - h16 / 2.0 - 1.0, py))
        ctx.positions["J16"] = (px, py, 0.0)
        mcu_grid.place(px, py, w16, h16)
        ctx.mcu_peripheral_refs.add("J16")
        _log.info("    J16 -> right edge, above U3 (%.1f, %.1f)", px, py)

    if "J2" in connector_refs and "J2" in ctx.positions and "J2" not in ctx.fixed_refs:
        w2, h2 = ctx.fp_sizes.get("J2", (9.6, 7.6))
        tx = mcu_left - w2 / 2.0 - 8.0
        ty = bounds[3] - h2 / 2.0 - 1.0
        tx = max(bounds[0] + w2 / 2.0 + 1.0,
                 min(bounds[2] - w2 / 2.0 - 1.0, tx))
        px, py = mcu_grid.find_free_pos(tx, ty, w2, h2, max_radius=20.0)
        ctx.positions["J2"] = (px, py, 180.0)
        mcu_grid.place(px, py, w2, h2)
        ctx.mcu_peripheral_refs.add("J2")
        _log.info("    J2 -> bottom edge, left of U3 (%.1f, %.1f)", px, py)

    # --- Step 5: USB subcircuit ---
    j2_pos = ctx.positions.get("J2")
    if "U9" in other_passive_refs and "U9" in ctx.positions and j2_pos:
        j2x, j2y, _j2r = j2_pos
        j2w, j2h = ctx.fp_sizes.get("J2", (9.6, 7.6))
        u9w, u9h = ctx.fp_sizes.get("U9", (3.0, 3.0))
        u9_tx = j2x - j2w / 4.0
        u9_ty = j2y - j2h / 2.0 - u9h / 2.0 - 6.0
        u9_tx = max(bounds[0] + 2.0, min(bounds[2] - u9w / 2.0 - 1.0, u9_tx))
        u9_ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, u9_ty))
        u9x, u9y = mcu_grid.find_free_pos(u9_tx, u9_ty, u9w, u9h,
                                           max_radius=8.0)
        ctx.positions["U9"] = (u9x, u9y, 0.0)
        ctx.mcu_peripheral_refs.add("U9")
        mcu_grid.place(u9x, u9y, u9w, u9h)
        other_passive_refs.remove("U9")
        _log.info("    U9 (ESD) -> near J2 at (%.1f, %.1f)",
                  u9x, u9y)

    usb_r_refs = [r for r in ("R6", "R7") if r in other_passive_refs
                  and r in ctx.positions]
    if usb_r_refs and j2_pos:
        j2x_r, j2y_r, _ = j2_pos
        j2w_r = ctx.fp_sizes.get("J2", (9.6, 7.6))[0]
        j2h_r = ctx.fp_sizes.get("J2", (9.6, 7.6))[1]
        for i, ref in enumerate(usb_r_refs):
            w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
            px = j2x_r - j2w_r / 4.0 + i * (w + 2.0)
            py = j2y_r - j2h_r / 2.0 - h / 2.0 - 1.5
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
            ctx.positions[ref] = (px, py, 0.0)
            mcu_grid.place(px, py, w, h)
            other_passive_refs.remove(ref)
            _log.info("    %s (USB R) -> (%.1f, %.1f)", ref, px, py)

    # --- Step 6: Reset/Boot subcircuit ---
    sw_pairs: list[tuple[str, str]] = []
    for sw, res in [("SW1", "R4"), ("SW2", "R5"), ("SW1", "R5"), ("SW2", "R4")]:
        if (sw in other_passive_refs and sw in ctx.positions
                and res in other_passive_refs and res in ctx.positions):
            sw_nets = ref_nets.get(sw, set())
            if res in sw_nets:
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

    sw_base_x = mcu_left + eff_w / 4.0
    sw_base_y = mcu_top - 4.0
    for i, (sw, res) in enumerate(unique_pairs):
        sw_w, sw_h = ctx.fp_sizes.get(sw, (3.5, 3.5))
        r_w, r_h = ctx.fp_sizes.get(res, (1.0, 0.5))
        tx = sw_base_x - i * (sw_w + 1.5)
        ty = sw_base_y
        tx = max(bounds[0] + sw_w / 2.0 + 2.0, min(bounds[2] - sw_w / 2.0 - 2.0, tx))
        ty = max(bounds[1] + sw_h / 2.0 + 2.0, min(bounds[3] - sw_h / 2.0 - 2.0, ty))
        px, py = mcu_grid.find_free_pos(tx, ty, sw_w, sw_h, max_radius=10.0)
        px = max(mcu_x - 20.0, min(mcu_x + 20.0, px))
        py = max(max(mcu_y - 20.0, bounds[1] + sw_h / 2.0 + 2.0),
                 min(bounds[3] - sw_h / 2.0 - 2.0, py))
        ctx.positions[sw] = (px, py, 0.0)
        ctx.mcu_peripheral_refs.add(sw)
        mcu_grid.place(px, py, sw_w, sw_h)
        other_passive_refs.remove(sw)
        r_tx = px
        r_ty = py - sw_h / 2.0 - r_h / 2.0 - 0.5
        r_tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, r_tx))
        r_ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, r_ty))
        rpx, rpy = mcu_grid.find_free_pos(r_tx, r_ty, r_w, r_h,
                                           max_radius=8.0)
        ctx.positions[res] = (rpx, rpy, 0.0)
        ctx.mcu_peripheral_refs.add(res)
        mcu_grid.place(rpx, rpy, r_w, r_h)
        other_passive_refs.remove(res)
        _log.info("    %s+%s (reset/boot) -> (%.1f,%.1f) / (%.1f,%.1f)",
                  sw, res, px, py, rpx, rpy)

    for sw in unpaired_sw:
        sw_w, sw_h = ctx.fp_sizes.get(sw, (3.5, 3.5))
        tx = sw_base_x - len(unique_pairs) * (sw_w + 3.0)
        ty = sw_base_y
        tx = max(bounds[0] + sw_w / 2.0 + 1.0, min(bounds[2] - 2.0, tx))
        ty = max(bounds[1] + sw_h / 2.0 + 1.0, min(bounds[3] - 2.0, ty))
        px, py = mcu_grid.find_free_pos(tx, ty, sw_w, sw_h, max_radius=15.0)
        ctx.positions[sw] = (px, py, 0.0)
        ctx.mcu_peripheral_refs.add(sw)
        mcu_grid.place(px, py, sw_w, sw_h)
        other_passive_refs.remove(sw)
        _log.info("    %s (switch) -> (%.1f, %.1f)", sw, px, py)

    # --- Step 6b: Place LED1 ---
    led_placed = set()
    for led_ref in ["LED1"]:
        if (led_ref in other_passive_refs and led_ref in ctx.positions
                and led_ref not in ctx.fixed_refs):
            lw, lh = ctx.fp_sizes.get(led_ref, (2.0, 1.0))
            led_tx = sw_base_x + 8.0
            led_ty = sw_base_y
            led_tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, led_tx))
            led_ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, led_ty))
            lpx, lpy = mcu_grid.find_free_pos(
                led_tx, led_ty, lw, lh, max_radius=10.0,
            )
            ctx.positions[led_ref] = (lpx, lpy, 0.0)
            ctx.mcu_peripheral_refs.add(led_ref)
            mcu_grid.place(lpx, lpy, lw, lh)
            if led_ref in other_passive_refs:
                other_passive_refs.remove(led_ref)
            led_placed.add(led_ref)
            _log.info("    %s (status LED) -> (%.1f, %.1f)", led_ref, lpx, lpy)

    # --- Step 7: Place remaining passives ---
    def _mcu_prox_key(ref: str) -> tuple[int, float]:
        connected = mcu_ref_c3 in ref_nets.get(ref, set())
        rx, ry, _ = ctx.positions[ref]
        dist = math.sqrt((rx - mcu_x) ** 2 + (ry - mcu_y) ** 2)
        return (0 if connected else 1, dist)

    remaining = [r for r in other_passive_refs if r in ctx.positions]
    remaining.sort(key=_mcu_prox_key)

    _MCU_TARGET_GAP = 4.0
    ring_slots: list[tuple[float, float]] = []
    _MCU_CLEAR = 3.0

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
            sx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, sx))
            sy = max(bounds[1] + 2.0, min(bounds[3] - 2.0, sy))
            if mcu_grid.is_free(sx, sy, w, h):
                ctx.positions[ref] = (sx, sy, rrot)
                ctx.mcu_peripheral_refs.add(ref)
                mcu_grid.place(sx, sy, w, h)
                placed = True
                break

        if not placed:
            tx = mcu_left - _MCU_TARGET_GAP - w / 2.0
            ty = mcu_y
            tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
            ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
            px, py = mcu_grid.find_free_pos(tx, ty, w, h,
                                            max_radius=25.0)
            ctx.positions[ref] = (px, py, rrot)
            ctx.mcu_peripheral_refs.add(ref)
            mcu_grid.place(px, py, w, h)

    # --- Post-placement: Push any components still inside U3 courtyard ---
    _COURT_MARGIN = 2.0
    court_x1 = mcu_x - eff_w / 2.0 - _COURT_MARGIN
    court_y1 = mcu_y - eff_h / 2.0 - _COURT_MARGIN
    court_x2 = mcu_x + eff_w / 2.0 + _COURT_MARGIN
    court_y2 = mcu_y + eff_h / 2.0 + _COURT_MARGIN
    for ref in list(ctx.mcu_peripheral_refs):
        if ref == mcu_ref_c3:
            continue
        if ref.startswith("J"):
            rx_j = ctx.positions[ref][0]
            if rx_j > mcu_x:
                continue
        rx, ry, rrot = ctx.positions[ref]
        w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        if court_x1 < rx < court_x2 and court_y1 < ry < court_y2:
            dx_left = rx - court_x1 + w / 2.0
            dx_right = court_x2 - rx + w / 2.0
            dy_top = ry - court_y1 + h / 2.0
            dy_bot = court_y2 - ry + h / 2.0
            min_d = min(dx_left, dx_right, dy_top, dy_bot)
            if min_d == dx_left:
                new_x = court_x1 - w / 2.0 - 1.0
                new_y = ry
            elif min_d == dx_right:
                new_x = court_x2 + w / 2.0 + 1.0
                new_y = ry
            elif min_d == dy_top:
                new_x = rx
                new_y = court_y1 - h / 2.0 - 1.0
            else:
                new_x = rx
                new_y = court_y2 + h / 2.0 + 1.0
            new_x = max(bounds[0] + w / 2.0 + 1.0,
                        min(bounds[2] - w / 2.0 - 1.0, new_x))
            new_y = max(bounds[1] + h / 2.0 + 1.0,
                        min(bounds[3] - h / 2.0 - 1.0, new_y))
            px, py = mcu_grid.find_free_pos(new_x, new_y, w, h,
                                             max_radius=15.0)
            ctx.positions[ref] = (px, py, rrot)
            mcu_grid.place(px, py, w, h)
            _log.info("    %s pushed outside U3 courtyard: "
                      "(%.1f,%.1f)->(%.1f,%.1f)", ref, rx, ry, px, py)

    _log.info(
        "    3c3: organized %d peripherals around %s at (%.1f, %.1f)",
        len(ctx.mcu_peripheral_refs), mcu_ref_c3, mcu_x, mcu_y,
    )


def _phase_ethernet_group(ctx: PlacementContext) -> None:
    """3c4: Ethernet group organization — vertical signal-chain column."""
    _log.info("  3c4: Ethernet group organization")
    bounds = ctx.bounds
    min_x, min_y, max_x, max_y = bounds

    eth_group_refs: set[str] = set()
    for feat in ctx.requirements.features:
        if "ethernet" in feat.name.lower():
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                eth_group_refs.add(r)
            break

    if not eth_group_refs:
        return

    eth_zone_rect: tuple[float, float, float, float] | None = None
    for z in ctx.zones:
        if z.name == "ethernet":
            eth_zone_rect = z.rect
            break

    eth_ics = sorted([r for r in eth_group_refs
                      if r.startswith("U") and r in ctx.positions])
    eth_connectors = sorted([r for r in eth_group_refs
                             if r.startswith("J") and r in ctx.positions])

    if not (eth_ics and eth_zone_rect is not None):
        return

    ezx1, ezy1, ezx2, ezy2 = eth_zone_rect
    _ETH_STRIP_GAP = 1.0

    eth_net_refs: dict[str, set[str]] = {}
    for net in ctx.requirements.nets:
        e_refs = set()
        for conn in net.connections:
            if conn.ref in eth_group_refs:
                e_refs.add(conn.ref)
        if e_refs:
            eth_net_refs[net.name] = e_refs

    eth_grid = _PlacementGrid(bounds)
    for oref, (ox, oy, _orot) in ctx.positions.items():
        if oref in eth_group_refs:
            continue
        ow, oh = _rotation_aware_size(oref, ctx.positions, ctx.fp_sizes)
        eth_grid.place(ox, oy, ow, oh)

    eth_main_ic = eth_ics[0]
    crystal_refs = sorted([r for r in eth_group_refs
                           if r.startswith("Y") and r in ctx.positions])
    poe_ic = eth_ics[1] if len(eth_ics) > 1 else ""

    eth_decoupling = sorted([
        r for r in eth_group_refs
        if r.startswith("C") and r in ctx.positions
        and r not in crystal_refs
    ])

    crystal_net_names: set[str] = set()
    for net_name, erefs in eth_net_refs.items():
        if any(r.startswith("Y") for r in erefs):
            crystal_net_names.add(net_name)
    crystal_load_caps = sorted([
        r for r in eth_group_refs
        if r.startswith("C") and r in ctx.positions
        and any(r in eth_net_refs.get(n, set())
                for n in crystal_net_names)
    ])

    poe_net_names: set[str] = set()
    for net_name, erefs in eth_net_refs.items():
        if poe_ic and poe_ic in erefs:
            poe_net_names.add(net_name)
    poe_caps = sorted([
        r for r in eth_group_refs
        if r.startswith("C") and r in ctx.positions
        and r not in crystal_load_caps
        and any(r in eth_net_refs.get(n, set())
                for n in poe_net_names)
    ])

    other_caps = sorted(
        [r for r in eth_decoupling
         if r not in crystal_load_caps and r not in poe_caps],
    )

    eth_anchor_x = (ezx1 + ezx2) / 2.0
    eth_anchor_y = ezy1 + 3.0

    placed_eth: set[str] = set()

    # Force-place main IC
    ic_w, ic_h = ctx.fp_sizes.get(eth_main_ic, (10.0, 10.0))
    ic_cx = eth_anchor_x
    ic_cy = eth_anchor_y + ic_h / 2.0
    ic_cx = max(ezx1 + ic_w / 2.0 + 1.0,
                min(ezx2 - ic_w / 2.0 - 1.0, ic_cx))
    ic_cy = max(ezy1 + ic_h / 2.0 + 1.0,
                min(ezy2 - ic_h / 2.0 - 5.0, ic_cy))
    if eth_main_ic in ctx.positions and eth_main_ic not in ctx.fixed_refs:
        ctx.positions[eth_main_ic] = (ic_cx, ic_cy, 0.0)
        eth_grid.place(ic_cx, ic_cy, ic_w, ic_h)
        ctx.ethernet_fixed.add(eth_main_ic)
        ctx.fixed_refs.add(eth_main_ic)
        placed_eth.add(eth_main_ic)
        _log.info("    %s (W5500) force-placed at (%.1f, %.1f) "
                  "in ethernet zone", eth_main_ic, ic_cx, ic_cy)

    # Pre-register J13
    for _j13_ref in eth_connectors:
        _j13_w, _j13_h = ctx.fp_sizes.get(_j13_ref, (19.6, 12.5))
        _j13_cx = eth_anchor_x
        _j13_cy = bounds[3] - _j13_h / 2.0 - 1.0
        eth_grid.place(_j13_cx, _j13_cy, _j13_w, _j13_h)

    def _place_eth_column(
        refs: list[str], col_x: float, start_y: float,
    ) -> float:
        cy = start_y
        for ref in refs:
            if (ref not in ctx.positions or ref in ctx.fixed_refs
                    or ref in placed_eth or ref == ""):
                continue
            w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
            tx = col_x
            ty = cy + h / 2.0
            tx = max(ezx1 + 2.0, min(ezx2 - 2.0, tx))
            ty = max(ezy1 + 2.0, min(ezy2 - 2.0, ty))
            px, py = eth_grid.find_free_pos(tx, ty, w, h,
                                            max_radius=6.0)
            ctx.positions[ref] = (px, py, 0.0)
            eth_grid.place(px, py, w, h)
            ctx.ethernet_fixed.add(ref)
            placed_eth.add(ref)
            cy = py + h / 2.0 + _ETH_STRIP_GAP
        return cy

    # Crystal + load caps
    y_crystal_h = ctx.fp_sizes.get(
        crystal_refs[0], (3.2, 1.5))[1] if crystal_refs else 1.5
    crystal_y = ic_cy - ic_h / 2.0 - y_crystal_h / 2.0 - 0.5
    crystal_x = ic_cx

    for ref in crystal_refs:
        if (ref not in ctx.positions or ref in ctx.fixed_refs
                or ref in placed_eth or ref == ""):
            continue
        w, h = ctx.fp_sizes.get(ref, (3.2, 1.5))
        tx = crystal_x
        ty = crystal_y
        tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
        ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
        ctx.positions[ref] = (tx, ty, 0.0)
        eth_grid.place(tx, ty, w, h)
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        crystal_dist = math.sqrt(
            (tx - ic_cx) ** 2 + (ty - ic_cy) ** 2)
        _log.info("    %s (crystal) -> (%.1f, %.1f) "
                  "dist=%.1fmm from %s",
                  ref, tx, ty, crystal_dist, eth_main_ic)
        cap_y = ty

    # Crystal load caps flanking the crystal
    cap_idx = 0
    for ref in crystal_load_caps:
        if (ref not in ctx.positions or ref in ctx.fixed_refs
                or ref in placed_eth or ref == ""):
            continue
        w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
        y_w = ctx.fp_sizes.get(crystal_refs[0], (3.2, 1.5))[0] if crystal_refs else 3.2
        if cap_idx % 2 == 0:
            tx = crystal_x - y_w / 2.0 - w / 2.0 - 0.5
        else:
            tx = crystal_x + y_w / 2.0 + w / 2.0 + 0.5
        ty = crystal_y if crystal_refs else ic_cy - ic_h / 2.0 - 3.0
        tx = max(ezx1 + 2.0, min(ezx2 - 2.0, tx))
        ty = max(ezy1 + 2.0, min(ezy2 - 2.0, ty))
        px, py = eth_grid.find_free_pos(tx, ty, w, h,
                                        max_radius=5.0)
        ctx.positions[ref] = (px, py, 0.0)
        eth_grid.place(px, py, w, h)
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        cap_idx += 1

    # Remaining caps
    col_y = ic_cy + ic_h / 2.0 + 1.5
    for ref in other_caps:
        if (ref not in ctx.positions or ref in ctx.fixed_refs
                or ref in placed_eth or ref == ""):
            continue
        cw, ch = ctx.fp_sizes.get(ref, (1.5, 1.0))
        tx = ic_cx
        ty = col_y + ch / 2.0
        tx = max(ezx1 + 2.0, min(ezx2 - 2.0, tx))
        ty = max(ezy1 + 2.0, min(ezy2 - 2.0, ty))
        px, py = eth_grid.find_free_pos(tx, ty, cw, ch,
                                        max_radius=8.0)
        ctx.positions[ref] = (px, py, 0.0)
        eth_grid.place(px, py, cw, ch)
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        col_y = py + ch / 2.0 + 1.0
    col1_bottom = col_y

    # Column 2: PoE/PHY module
    if poe_ic and poe_ic in ctx.positions and poe_ic not in ctx.fixed_refs:
        poe_w, poe_h = ctx.fp_sizes.get(poe_ic, (8.0, 8.0))
        poe_eff_w, poe_eff_h = poe_h, poe_w
        poe_tx = eth_anchor_x
        poe_ty = bounds[3] - 25.0
        poe_tx = max(bounds[0] + poe_eff_w / 2 + 1,
                     min(bounds[2] - poe_eff_w / 2 - 1, poe_tx))
        poe_ty = max(bounds[1] + poe_eff_h / 2 + 1,
                     min(bounds[3] - poe_eff_h / 2 - 1, poe_ty))
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

    # J13 (RJ45): bottom edge
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
        px = max(ezx1 + w / 2.0, min(ezx2 - w / 2.0, px))
        ctx.positions[ref] = (px, py, 180.0)
        ctx.ethernet_fixed.add(ref)
        placed_eth.add(ref)
        _log.info("    %s (RJ45) -> bottom edge (%.1f, %.1f)",
                  ref, px, py)

    # Any remaining eth refs
    remaining_eth = sorted(
        eth_group_refs - placed_eth - ctx.fixed_refs,
    )
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
    _crystal_placed = [r for r in placed_eth if r.startswith("Y")]
    for yref in _crystal_placed:
        yx, yy, yrot = ctx.positions[yref]
        yw, yh = ctx.fp_sizes.get(yref, (3.2, 1.5))
        for cref in list(placed_eth):
            if cref == yref or not cref.startswith("C"):
                continue
            cx, cy, crot = ctx.positions[cref]
            cw, ch = ctx.fp_sizes.get(cref, (1.5, 1.0))
            _margin = 2.0
            if (abs(cx - yx) < (cw + yw) / 2.0 + _margin
                    and abs(cy - yy) < (ch + yh) / 2.0 + _margin):
                new_cy = yy + yh / 2.0 + ch / 2.0 + 2.0
                new_cy = max(ezy1 + 2.0, min(ezy2 - 2.0, new_cy))
                ctx.positions[cref] = (cx, new_cy, crot)
                _log.info("    3c4: shifted %s below %s "
                          "(%.1f,%.1f) -> (%.1f,%.1f)",
                          cref, yref, cx, cy, cx, new_cy)

    # Protect all ethernet from collision resolution
    ctx.fixed_refs.update(ctx.ethernet_fixed)

    # 3c4-post: Push non-ethernet components clear of ethernet ICs
    eth_ic_refs = [r for r in ctx.ethernet_fixed if r.startswith("U")]
    for eic in eth_ic_refs:
        if eic not in ctx.positions:
            continue
        ex, ey, erot = ctx.positions[eic]
        ew, eh = ctx.fp_sizes.get(eic, (5.0, 5.0))
        if erot % 180 in (90, 270):
            ew, eh = eh, ew
        for ref in list(ctx.positions):
            if ref in ctx.ethernet_fixed or ref in ctx.fixed_refs:
                continue
            if ref in ctx.relay_support_refs or ref.startswith("K"):
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
                _log.info("    3c4-post: pushed %s away from %s by %.1fmm",
                          ref, eic, push_dist)
