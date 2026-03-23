"""EE placement optimizer — core placement phases.

Contains zone partitioning, group placement, and level-3 intra-group
refinement phases (relay rows/drivers/LEDs, decoupling, crystal,
RF edge, connector orientation, top-edge connectors, template refinement).

Group organization phases live in ``ee_phases_groups``.
Late refinement and finalization phases live in ``ee_phases_refinement``.
"""

from __future__ import annotations

import logging
import math

from kicad_pipeline.optimization.collision_resolver import (
    _PlacementGrid,
    _rotation_aware_size,
)

# Re-export group phases for backwards compatibility
from kicad_pipeline.optimization.ee_phases_groups import (  # noqa: F401
    _phase_adc_analog_cluster,
    _phase_adc_channels,
    _phase_ethernet_group,
    _phase_mcu_group,
    _phase_power_group,
)

# Re-export refinement phases for backwards compatibility
from kicad_pipeline.optimization.ee_phases_refinement import (  # noqa: F401
    _phase_build_final,
    _phase_collision_resolution,
    _phase_final_clamp,
    _phase_first_clamp,
    _phase_late_adc_realignment,
    _phase_late_decoupling,
    _phase_late_relay_realignment,
    _phase_mcu_decoupling_repull,
    _phase_review_loop,
)
from kicad_pipeline.optimization.functional_grouper import (
    SubCircuitType,
)
from kicad_pipeline.optimization.level3_phases import (
    _apply_template_refinement,
    _orient_connectors,
    _pin_rf_to_edge,
    _place_row_layout,
)
from kicad_pipeline.optimization.placement_types import (
    PlacementContext,
)
from kicad_pipeline.pcb.pin_map import (
    origin_to_centroid,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Level 1 & 2
# ---------------------------------------------------------------------------


def _phase_zone_partitioning(ctx: PlacementContext) -> None:
    """Level 1: Zone Partitioning — partition board into non-overlapping zones."""
    from kicad_pipeline.optimization.functional_grouper import (
        compute_power_flow_topology,
    )
    from kicad_pipeline.optimization.zone_partitioner import partition_board

    topology = compute_power_flow_topology(ctx.subcircuits)
    ctx.zones = partition_board(
        ctx.bounds, list(ctx.requirements.features), topology,
    )
    _log.info("  %d zones created", len(ctx.zones))


def _phase_group_placement(ctx: PlacementContext) -> None:
    """Level 2: Group Placement — place groups as rigid units in zones."""
    has_groups = bool(ctx.requirements.features)
    if not (has_groups and ctx.zones):
        _log.info("  No groups — using initial placement")
        return

    from kicad_pipeline.optimization.group_placer import (
        pin_connectors_to_edge,
        place_groups,
    )

    # Extract internal layouts per group from current positions
    internal_layouts: dict[str, dict[str, tuple[float, float, float]]] = {}
    for block in ctx.requirements.features:
        layout: dict[str, tuple[float, float, float]] = {}
        refs_in_pos = [r for r in block.components if r in ctx.positions]
        if not refs_in_pos:
            continue
        for ref in refs_in_pos:
            x, y, rot = ctx.positions[ref]
            layout[ref] = (x, y, rot)
        internal_layouts[block.name] = layout

    placed_groups = place_groups(
        ctx.zones, list(ctx.requirements.features),
        internal_layouts, ctx.fp_sizes, ctx.bounds,
    )

    # Merge placed group positions back — skip fixed refs
    for pg in placed_groups:
        for ref, (px, py) in pg.positions.items():
            if ref in ctx.fixed_refs:
                continue
            if ref in ctx.positions:
                _, _, rot = ctx.positions[ref]
                ctx.positions[ref] = (px, py, rot)

    # Pin connectors to board edges
    _log.info("  Pinning connectors to board edges")
    edge_positions = pin_connectors_to_edge(
        placed_groups, ctx.fp_sizes, ctx.bounds, ctx.fixed_refs,
    )
    for ref, (px, py) in edge_positions.items():
        if ref.startswith("J") and ref not in ctx.fixed_refs and ref in ctx.positions:
            _, _, rot = ctx.positions[ref]
            ctx.positions[ref] = (px, py, rot)

    _log.info("  %d groups placed", len(placed_groups))


# ---------------------------------------------------------------------------
# Level 3 phases
# ---------------------------------------------------------------------------


def _phase_relay_rows(ctx: PlacementContext) -> None:
    """3a: Relay row formation — arrange relays in 1xN horizontal row."""
    _log.info("  3a: Relay row formation")
    sc_list = list(ctx.subcircuits)
    ctx.positions = _place_row_layout(
        sc_list, ctx.positions, ctx.fp_sizes, ctx.bounds, ctx.fixed_refs,
        zones=ctx.zones,
    )


def _phase_relay_drivers(ctx: PlacementContext) -> None:
    """3b: Relay driver subgroup tightening — Q+D+R within 8mm of K."""
    _log.info("  3b: Relay driver subgroup tightening")
    sc_list = list(ctx.subcircuits)
    bounds = ctx.bounds

    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.RELAY_DRIVER:
            continue
        anchor = sc.anchor_ref
        if anchor not in ctx.positions:
            continue
        kx, ky, _krot = ctx.positions[anchor]
        kw, kh = ctx.fp_sizes.get(anchor, (18.0, 16.0))
        if _krot % 180 in (90.0, 270.0):
            kw, kh = kh, kw

        support_members = [
            r for r in sc.refs
            if r != anchor and r in ctx.positions and r not in ctx.fixed_refs
        ]
        support_members.sort(key=lambda r: (
            0 if r.startswith("Q") else 1 if r.startswith("D") else 2, r,
        ))

        target_y_base = ky + kh / 2.0 + 0.5
        col = 0
        row_y = target_y_base
        row_max_h = 0.0
        cols_per_row = 3

        for ref in support_members:
            w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
            px = kx - kw / 2.0 + (col + 0.5) * (kw / cols_per_row)
            py = row_y + h / 2.0
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
            ctx.positions[ref] = (px, py, 0.0)
            ctx.relay_support_refs.add(ref)
            row_max_h = max(row_max_h, h)
            col += 1
            if col >= cols_per_row:
                col = 0
                row_y += row_max_h + 0.5
                row_max_h = 0.0


def _phase_relay_leds(ctx: PlacementContext) -> tuple[dict[str, list[str]], set[str]]:
    """3b2: Relay LED indicator placement — D_LED + R_LED below support grid.

    Returns:
        Tuple of (relay_leds mapping, relay_led_refs set) for use by later phases.
    """
    _log.info("  3b2: Relay LED indicator placement")
    relay_led_refs: set[str] = set()
    bounds = ctx.bounds

    # Build relay coil net -> relay mapping
    _coil_net_to_relay: dict[str, str] = {}
    for net in ctx.requirements.nets:
        if "_COIL" in net.name.upper():
            for conn in net.connections:
                if conn.ref.startswith("K"):
                    _coil_net_to_relay[net.name] = conn.ref
                    break

    # Find LED+resistor pairs per relay via coil nets
    _relay_leds: dict[str, list[str]] = {}
    for net in ctx.requirements.nets:
        if "_COIL" in net.name.upper():
            k_ref = _coil_net_to_relay.get(net.name)
            if not k_ref:
                continue
            for conn in net.connections:
                if conn.ref.startswith("D") and conn.ref not in ctx.relay_support_refs:
                    _relay_leds.setdefault(k_ref, []).append(conn.ref)
        elif "_LED" in net.name.upper():
            d_refs_in = [c.ref for c in net.connections if c.ref.startswith("D")]
            r_refs_in = [c.ref for c in net.connections if c.ref.startswith("R")]
            for d_ref in d_refs_in:
                for k_ref, led_list in _relay_leds.items():
                    if d_ref in led_list:
                        led_list.extend(r_refs_in)
                        break

    # Place LED pairs below each relay's support row
    for k_ref in sorted(_relay_leds):
        if k_ref not in ctx.positions:
            continue
        kx, ky, _krot = ctx.positions[k_ref]
        kw, kh = ctx.fp_sizes.get(k_ref, (18.0, 16.0))
        if _krot % 180 in (90.0, 270.0):
            kw, kh = kh, kw

        led_members = sorted(set(_relay_leds[k_ref]))
        led_members = [r for r in led_members if r in ctx.positions and r not in ctx.fixed_refs]
        if not led_members:
            continue

        led_y_base = ky + kh / 2.0 + 5.5
        led_col = 0
        led_cols_per_row = min(len(led_members), 2)
        for ref in led_members:
            w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
            px = kx - kw / 4.0 + led_col * (kw / 2.0)
            py = led_y_base + h / 2.0
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
            _, _, rot = ctx.positions[ref]
            ctx.positions[ref] = (px, py, rot)
            ctx.relay_support_refs.add(ref)
            relay_led_refs.add(ref)
            led_col += 1
            if led_col >= led_cols_per_row:
                led_col = 0
                led_y_base += h + 0.5

        _log.info("    3b2: placed %d LED refs for %s", len(led_members), k_ref)

    return _relay_leds, relay_led_refs


def _phase_decoupling(ctx: PlacementContext) -> None:
    """3c: Decoupling cap tightening — within 3-5mm of IC."""
    _log.info("  3c: Decoupling cap tightening")
    sc_list = list(ctx.subcircuits)
    bounds = ctx.bounds

    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.DECOUPLING:
            continue
        ic_ref = sc.anchor_ref
        if ic_ref not in ctx.positions:
            continue
        ix, iy, _irot = ctx.positions[ic_ref]
        iw, ih = ctx.fp_sizes.get(ic_ref, (5.0, 5.0))

        for cap_ref in sc.refs:
            if cap_ref == ic_ref or not cap_ref.startswith("C"):
                continue
            if cap_ref not in ctx.positions or cap_ref in ctx.fixed_refs:
                continue
            cx, cy, crot = ctx.positions[cap_ref]
            cw, ch = ctx.fp_sizes.get(cap_ref, (1.5, 1.0))

            dist = math.sqrt((cx - ix) ** 2 + (cy - iy) ** 2)
            edge_dist = max(0.0, dist - (iw + cw) / 2.0)
            if edge_dist <= 4.0:
                continue

            dx = ix - cx
            dy = iy - cy
            d = math.sqrt(dx * dx + dy * dy) or 1.0
            target_dist = (iw + cw) / 2.0 + 1.5
            tx = ix - dx / d * target_dist
            ty = iy - dy / d * target_dist
            tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
            ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
            ctx.positions[cap_ref] = (tx, ty, crot)


def _phase_crystal_placement(ctx: PlacementContext) -> None:
    """3d: Crystal-IC proximity — within 10mm of connected IC."""
    _log.info("  3d: Crystal-IC proximity")
    sc_list = list(ctx.subcircuits)
    bounds = ctx.bounds

    # Build crystal->IC map via net connectivity
    _crystal_ref_to_nets: dict[str, set[str]] = {}
    _net_to_components: dict[str, set[str]] = {}
    for net in ctx.requirements.nets:
        refs_in_net: set[str] = set()
        for conn in net.connections:
            refs_in_net.add(conn.ref)
        _net_to_components[net.name] = refs_in_net
        for conn in net.connections:
            if conn.ref.startswith("Y"):
                _crystal_ref_to_nets.setdefault(conn.ref, set()).add(net.name)

    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.CRYSTAL_OSC:
            continue
        crystal_ref = sc.anchor_ref
        target_ic: str | None = None
        crystal_nets = _crystal_ref_to_nets.get(crystal_ref, set())
        for net_name in crystal_nets:
            _nl = net_name.upper()
            if _nl in ("GND", "AGND", "DGND", "PGND", "VSS", "AVSS"):
                continue
            for r in _net_to_components.get(net_name, set()):
                if r.startswith("U") and r in ctx.positions and r != crystal_ref:
                    target_ic = r
                    break
            if target_ic:
                break

        if not target_ic:
            from kicad_pipeline.optimization.functional_grouper import (
                _find_mcu_ref,
            )
            target_ic = _find_mcu_ref(ctx.requirements)
        if not target_ic or target_ic not in ctx.positions:
            continue

        ic_x, ic_y, _ic_rot = ctx.positions[target_ic]
        ic_w, ic_h = ctx.fp_sizes.get(target_ic, (5.0, 5.0))
        _log.info("    Crystal %s -> IC %s (%.1f, %.1f)",
                  crystal_ref, target_ic, ic_x, ic_y)

        for ref in sc.refs:
            if ref in ctx.fixed_refs or ref not in ctx.positions:
                continue
            rx, ry, rrot = ctx.positions[ref]
            dist = math.sqrt((rx - ic_x) ** 2 + (ry - ic_y) ** 2)
            if dist <= 10.0:
                continue
            w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
            gap = 1.0
            candidates = [
                (ic_x + (ic_w + w) / 2.0 + gap, ic_y),
                (ic_x - (ic_w + w) / 2.0 - gap, ic_y),
                (ic_x, ic_y + (ic_h + h) / 2.0 + gap),
                (ic_x, ic_y - (ic_h + h) / 2.0 - gap),
            ]
            pull_grid = _PlacementGrid(bounds)
            for oref, (ox, oy, _or) in ctx.positions.items():
                if oref != ref:
                    ow, oh = _rotation_aware_size(oref, ctx.positions, ctx.fp_sizes)
                    pull_grid.place(ox, oy, ow, oh)
            best_pos: tuple[float, float] | None = None
            best_dist = dist
            for txx, tyy in candidates:
                fx, fy = pull_grid.find_free_pos(txx, tyy, w, h)
                new_d = math.sqrt((fx - ic_x) ** 2 + (fy - ic_y) ** 2)
                if new_d < best_dist:
                    best_dist = new_d
                    best_pos = (fx, fy)
            if best_pos is not None:
                ctx.positions[ref] = (best_pos[0], best_pos[1], rrot)
                _log.info("    %s pulled to (%.1f, %.1f) dist=%.1f->%.1f from %s",
                          ref, best_pos[0], best_pos[1], dist, best_dist,
                          target_ic)


def _phase_rf_edge(ctx: PlacementContext) -> None:
    """3e: RF edge pinning — pin RF modules to board edge."""
    _log.info("  3e: RF edge pinning")
    sc_list = list(ctx.subcircuits)
    ctx.positions = _pin_rf_to_edge(
        sc_list, ctx.positions, ctx.fp_sizes, ctx.bounds, ctx.fixed_refs,
    )


def _phase_connector_orientation(ctx: PlacementContext) -> None:
    """3f: Connector orientation — face outward from board edge."""
    _log.info("  3f: Connector orientation")
    ctx.positions = _orient_connectors(
        ctx.positions, ctx.fp_sizes, ctx.bounds, ctx.fixed_refs, ctx.initial_pcb,
    )


def _phase_top_edge_connectors(ctx: PlacementContext) -> None:
    """3f2: Top-edge screw terminal ordering."""
    min_x, min_y, max_x, max_y = ctx.bounds

    _TOP_EDGE_ORDER = ["J6", "J5", "J4", "J3", "J1"]
    _top_refs = [r for r in _TOP_EDGE_ORDER if r in ctx.positions and r not in ctx.fixed_refs]
    if not _top_refs:
        return

    _log.info("  3f2: Top-edge screw terminal ordering (%s)", _top_refs)
    term_gap = 3.0
    term_widths: list[float] = []
    for r in _top_refs:
        w, _h = ctx.fp_sizes.get(r, (2.0, 2.0))
        term_widths.append(w)
    total_w = sum(term_widths) + term_gap * (len(_top_refs) - 1)
    margin = 8.0
    avail_w = (max_x - min_x) - 2 * margin
    if total_w < avail_w:
        start_x = min_x + margin + (avail_w - total_w) / 2.0
    else:
        compressed_gap = max(1.0, (avail_w - sum(term_widths)) / max(len(_top_refs) - 1, 1))
        term_gap = compressed_gap
        total_w = sum(term_widths) + term_gap * (len(_top_refs) - 1)
        start_x = min_x + margin
    cursor_x = start_x
    origin_y_target = min_y + 3.0
    for i, r in enumerate(_top_refs):
        tw = term_widths[i]
        origin_x = cursor_x + tw / 2.0
        fp_match = None
        for fp in ctx.initial_pcb.footprints:
            if fp.ref == r:
                fp_match = fp
                break
        if fp_match is not None:
            cent_x, cent_y = origin_to_centroid(
                fp_match, origin_x, origin_y_target, 0.0,
            )
        else:
            cent_x, cent_y = origin_x, origin_y_target
        ctx.positions[r] = (cent_x, cent_y, 0.0)
        cursor_x += tw + term_gap
        _log.info("    %s -> centroid(%.1f, %.1f) origin(%.1f, %.1f) rot=0",
                  r, cent_x, cent_y, origin_x, origin_y_target)
    ctx.top_edge_connector_refs = set(_top_refs)


def _phase_template_refinement(ctx: PlacementContext) -> None:
    """3h: Template-guided refinement — apply subcircuit layout templates."""
    _log.info("  3h: Template-guided refinement")
    template_protected = (ctx.fixed_refs | ctx.relay_support_refs | ctx.adc_channel_refs
                          | ctx.mcu_peripheral_refs | ctx.power_group_fixed
                          | ctx.ethernet_fixed | ctx.top_edge_connector_refs)
    ctx.positions, ctx.template_fixed = _apply_template_refinement(
        ctx.positions, ctx.fp_sizes, ctx.bounds, ctx.requirements,
        ctx.subcircuits, template_protected,
    )
