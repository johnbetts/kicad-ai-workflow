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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements

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


def _build_connector_to_relay_map(
    requirements: ProjectRequirements,
) -> dict[str, str]:
    """Map connector (J*) refs to their relay (K*) via shared nets.

    A connector serves a relay when they share COM/NO/NC nets.  This is
    determined by finding nets that connect both a J* and a K* component.
    """
    connector_to_relay: dict[str, str] = {}
    for net in requirements.nets:
        j_refs: list[str] = []
        k_refs: list[str] = []
        for conn in net.connections:
            if conn.ref.startswith("J"):
                j_refs.append(conn.ref)
            elif conn.ref.startswith("K"):
                k_refs.append(conn.ref)
        # If a net connects exactly one J to one K, that's a relay terminal
        if len(j_refs) == 1 and len(k_refs) == 1:
            j_ref = j_refs[0]
            k_ref = k_refs[0]
            # Only map if not already mapped (first net wins)
            if j_ref not in connector_to_relay:
                connector_to_relay[j_ref] = k_ref
    return connector_to_relay


def _phase_relay_connector_alignment(ctx: PlacementContext) -> None:
    """3a2: Align relay terminal connectors (J) to their relay (K) X position.

    Design rules (from relay_driver.md):
    - Each J that serves a K gets J.x = K.x (via shared COM/NO/NC nets)
    - J.y = near board top edge (terminal_y = min_y + 5mm)
    - J.rotation = 180 deg (pads face board edge)
    """
    _log.info("  3a2: Relay connector-relay alignment")
    bounds = ctx.bounds
    min_x, min_y, max_x, max_y = bounds

    connector_to_relay = _build_connector_to_relay_map(ctx.requirements)
    if not connector_to_relay:
        _log.info("    No connector-relay associations found")
        return

    # Terminal row Y: near the top board edge
    terminal_y = min_y + 5.0

    aligned = 0
    for j_ref, k_ref in sorted(connector_to_relay.items()):
        if j_ref not in ctx.positions or j_ref in ctx.fixed_refs:
            continue
        if k_ref not in ctx.positions:
            continue

        kx, _ky, _krot = ctx.positions[k_ref]

        # Set J position: same X as relay, near top edge, rotated 180 deg
        px = max(min_x + 2.0, min(max_x - 2.0, kx))
        py = max(min_y + 2.0, min(max_y - 2.0, terminal_y))
        ctx.positions[j_ref] = (px, py, 180.0)
        ctx.relay_support_refs.add(j_ref)
        aligned += 1
        _log.info("    %s -> (%.1f, %.1f) rot=180 aligned to %s",
                   j_ref, px, py, k_ref)

    _log.info("    Aligned %d connectors to their relays", aligned)


def _phase_relay_drivers(ctx: PlacementContext) -> None:
    """3b: Relay driver placement — pad-connectivity-driven two-column layout.

    Two-column layout relative to relay anchor K:

    LEFT column (dx ~ -4.3mm from K.x) — high-current signal chain:
      D_flyback at dy=+10.8, rot=0   (anode down toward Q collector)
      Q transistor at dy=+13.3, rot=180 (collector up toward D, base toward R_gate)

    RIGHT column (dx ~ +4.0mm from K.x) — control:
      R_gate at dy=+15.4, rot=180  (connects Q base to GPIO)

    The logic: D_flyback anode and Q collector share the COIL net —
    placing them vertically with pads facing each other minimises trace
    length.  R_gate on the opposite side creates a routing channel.
    """
    _log.info("  3b: Relay driver subgroup tightening (two-column)")
    sc_list = list(ctx.subcircuits)
    bounds = ctx.bounds

    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.RELAY_DRIVER:
            continue
        anchor = sc.anchor_ref
        if anchor not in ctx.positions:
            continue
        kx, ky, _krot = ctx.positions[anchor]

        support_members = [
            r for r in sc.refs
            if r != anchor and r in ctx.positions and r not in ctx.fixed_refs
        ]

        # Classify support members by type
        q_refs = sorted(r for r in support_members if r.startswith("Q"))
        d_refs = sorted(r for r in support_members if r.startswith("D"))
        all_r_refs = sorted(r for r in support_members if r.startswith("R"))
        other_refs = sorted(
            r for r in support_members
            if not r.startswith("Q") and not r.startswith("D") and not r.startswith("R")
        )

        # Separate gate resistors from LED resistors:
        # Gate resistor shares a DRIVE net with a Q ref in this subcircuit.
        power_nets = {"GND", "+5V", "+5V_RELAY", "+5V_LOGIC", "VCC"}
        r_gate_refs: list[str] = []
        r_other_refs: list[str] = []
        for r_ref in all_r_refs:
            shares_net_with_q = False
            for net in ctx.requirements.nets:
                if net.name.upper() in power_nets:
                    continue
                r_in_net = any(c.ref == r_ref for c in net.connections)
                q_in_net = any(c.ref in q_refs for c in net.connections)
                if r_in_net and q_in_net:
                    shares_net_with_q = True
                    break
            if shares_net_with_q:
                r_gate_refs.append(r_ref)
            else:
                r_other_refs.append(r_ref)

        # Also find gate resistors NOT in subcircuit but sharing a DRIVE net with Q
        for net in ctx.requirements.nets:
            if net.name.upper() in power_nets:
                continue
            if "DRIVE" not in net.name.upper():
                continue
            q_in_net = any(c.ref in q_refs for c in net.connections)
            if not q_in_net:
                continue
            for conn in net.connections:
                if (conn.ref.startswith("R")
                        and conn.ref not in r_gate_refs
                        and conn.ref not in r_other_refs
                        and conn.ref in ctx.positions
                        and conn.ref not in ctx.fixed_refs):
                    r_gate_refs.append(conn.ref)
                    _log.info("    Found external gate R %s via net %s",
                               conn.ref, net.name)

        r_refs = r_gate_refs
        # LED resistors go to other_refs for generic grid placement
        other_refs.extend(r_other_refs)

        # Two-column layout offsets (relative to K centroid)
        left_x = kx - 4.3   # LEFT column: high-current signal chain
        right_x = kx + 4.0  # RIGHT column: control

        # LEFT column: D_flyback above Q (anode facing down toward Q collector)
        for d_ref in d_refs:
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ky + 10.8))
            ctx.positions[d_ref] = (px, py, 0.0)
            ctx.relay_support_refs.add(d_ref)
            _log.info("    D %s -> (%.1f, %.1f) LEFT col, rot=0", d_ref, px, py)

        # LEFT column: Q below D_flyback (collector up toward D, 180 deg)
        for q_ref in q_refs:
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ky + 13.3))
            ctx.positions[q_ref] = (px, py, 180.0)
            ctx.relay_support_refs.add(q_ref)
            _log.info("    Q %s -> (%.1f, %.1f) LEFT col, rot=180", q_ref, px, py)

        # RIGHT column: R_gate (connects Q base to GPIO)
        for r_ref in r_refs:
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, right_x))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ky + 15.4))
            ctx.positions[r_ref] = (px, py, 180.0)
            ctx.relay_support_refs.add(r_ref)
            _log.info("    R %s -> (%.1f, %.1f) RIGHT col, rot=180", r_ref, px, py)

        # Place any remaining components in a grid below
        if other_refs:
            other_y = ky + 18.0
            for i, ref in enumerate(other_refs):
                w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
                px = kx - 4.0 + (i % 2) * 8.0
                py = other_y + (i // 2) * (h + 0.5)
                px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
                py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
                ctx.positions[ref] = (px, py, 0.0)
                ctx.relay_support_refs.add(ref)


def _build_coil_net_to_relay(
    requirements: ProjectRequirements,
) -> dict[str, str]:
    """Map coil net names to relay (K*) refs."""
    coil_net_to_relay: dict[str, str] = {}
    for net in requirements.nets:
        if "_COIL" not in net.name.upper():
            continue
        for conn in net.connections:
            if conn.ref.startswith("K"):
                coil_net_to_relay[net.name] = conn.ref
                break
    return coil_net_to_relay


def _find_relay_led_pairs(
    requirements: ProjectRequirements,
    coil_net_to_relay: dict[str, str],
    relay_support_refs: set[str],
) -> dict[str, list[str]]:
    """Find LED+resistor refs per relay via coil and LED nets.

    Detection strategy:
    1. Find R refs on COIL nets (these are LED resistors connected to the coil)
    2. Trace those R refs to LED nets to find the LED D refs
    3. Also find D refs on COIL nets not already in support (original path)
    """
    relay_leds: dict[str, list[str]] = {}

    # Build ref-to-nets map for tracing
    ref_to_nets: dict[str, set[str]] = {}
    net_to_refs: dict[str, set[str]] = {}
    for net in requirements.nets:
        refs_in_net: set[str] = set()
        for conn in net.connections:
            refs_in_net.add(conn.ref)
            ref_to_nets.setdefault(conn.ref, set()).add(net.name)
        net_to_refs[net.name] = refs_in_net

    # Strategy 1: R refs on COIL nets → trace via LED nets → find D_LED
    for net in requirements.nets:
        name_upper = net.name.upper()
        if "_COIL" not in name_upper:
            continue
        k_ref = coil_net_to_relay.get(net.name)
        if not k_ref:
            continue

        # Find R refs on this COIL net (these are LED current-limiting resistors)
        r_refs_on_coil = [
            c.ref for c in net.connections
            if c.ref.startswith("R")
        ]

        for r_ref in r_refs_on_coil:
            # Trace this R through its other nets to find LED D refs
            for other_net_name in ref_to_nets.get(r_ref, set()):
                if other_net_name == net.name:
                    continue  # Skip the COIL net itself
                other_refs = net_to_refs.get(other_net_name, set())
                for ref in other_refs:
                    if ref.startswith("D") and ref != r_ref and ref not in relay_support_refs:
                        # Check this D is not a flyback diode (not on a COIL net)
                        d_nets = ref_to_nets.get(ref, set())
                        is_flyback = any("_COIL" in n.upper() for n in d_nets)
                        if not is_flyback:
                            led_list = relay_leds.setdefault(k_ref, [])
                            if ref not in led_list:
                                led_list.append(ref)
                            if r_ref not in led_list:
                                led_list.append(r_ref)

    # Strategy 2 (original): D refs on COIL nets not in support → LED nets → R refs
    for net in requirements.nets:
        name_upper = net.name.upper()
        if "_COIL" in name_upper:
            k_ref = coil_net_to_relay.get(net.name)
            if not k_ref:
                continue
            for conn in net.connections:
                if conn.ref.startswith("D") and conn.ref not in relay_support_refs:
                    led_list = relay_leds.setdefault(k_ref, [])
                    if conn.ref not in led_list:
                        led_list.append(conn.ref)
        elif "LED" in name_upper:
            d_refs_in = [c.ref for c in net.connections if c.ref.startswith("D")]
            r_refs_in = [c.ref for c in net.connections if c.ref.startswith("R")]
            for d_ref in d_refs_in:
                for _k_ref, led_list in relay_leds.items():
                    if d_ref in led_list:
                        for r in r_refs_in:
                            if r not in led_list:
                                led_list.append(r)
                        break

    return relay_leds


def _phase_relay_leds(ctx: PlacementContext) -> tuple[dict[str, list[str]], set[str]]:
    """3b2: Relay LED indicator placement — LEFT column below Q.

    Pad-connectivity-driven layout (continuation of two-column pattern):

    LEFT column (dx ~ -4.3mm from K.x):
      R_LED at dy=+15.5, rot=0    (pad 1 up toward Q collector / COIL net)
      D_LED at dy=+17.7, rot=180  (anode up toward R_LED pad 2)

    The logic: R_LED pad 1 connects to the COIL net (same as Q collector),
    so it goes directly below Q.  D_LED anode connects to R_LED pad 2,
    so it goes directly below R_LED with anode facing up (180 deg).

    Returns:
        Tuple of (relay_leds mapping, relay_led_refs set) for use by later phases.
    """
    _log.info("  3b2: Relay LED indicator placement (two-column)")
    relay_led_refs: set[str] = set()
    bounds = ctx.bounds

    coil_net_to_relay = _build_coil_net_to_relay(ctx.requirements)
    _relay_leds = _find_relay_led_pairs(
        ctx.requirements, coil_net_to_relay, ctx.relay_support_refs,
    )

    for k_ref in sorted(_relay_leds):
        if k_ref not in ctx.positions:
            continue
        kx, ky, _krot = ctx.positions[k_ref]

        led_members = sorted(set(_relay_leds[k_ref]))
        led_members = [r for r in led_members if r in ctx.positions and r not in ctx.fixed_refs]
        if not led_members:
            continue

        # LEFT column X — same as D_flyback and Q
        left_x = kx - 4.3

        # Separate R_LED and D_LED refs
        r_led_refs = sorted(r for r in led_members if r.startswith("R"))
        d_led_refs = sorted(r for r in led_members if r.startswith("D"))
        other_led_refs = sorted(
            r for r in led_members
            if not r.startswith("R") and not r.startswith("D")
        )

        # R_LED: below Q, pad 1 facing up toward COIL net (rot=0)
        for ref in r_led_refs:
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ky + 15.5))
            ctx.positions[ref] = (px, py, 0.0)
            ctx.relay_support_refs.add(ref)
            relay_led_refs.add(ref)

        # D_LED: below R_LED, anode facing up (rot=180)
        for ref in d_led_refs:
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ky + 17.7))
            ctx.positions[ref] = (px, py, 180.0)
            ctx.relay_support_refs.add(ref)
            relay_led_refs.add(ref)

        # Remaining LED-related refs below
        for i, ref in enumerate(other_led_refs):
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x + i * 3.0))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ky + 19.5))
            _, _, rot = ctx.positions[ref]
            ctx.positions[ref] = (px, py, rot)
            ctx.relay_support_refs.add(ref)
            relay_led_refs.add(ref)

        _log.info("    3b2: placed %d LED refs for %s (left col at x=%.1f)",
                   len(led_members), k_ref, left_x)

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


_GND_NET_NAMES: frozenset[str] = frozenset(
    {"GND", "AGND", "DGND", "PGND", "VSS", "AVSS"},
)


def _build_crystal_net_maps(
    requirements: ProjectRequirements,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build crystal-ref-to-nets and net-to-components maps.

    Returns (crystal_ref_to_nets, net_to_components).
    """
    crystal_ref_to_nets: dict[str, set[str]] = {}
    net_to_components: dict[str, set[str]] = {}
    for net in requirements.nets:
        refs_in_net: set[str] = set()
        for conn in net.connections:
            refs_in_net.add(conn.ref)
        net_to_components[net.name] = refs_in_net
        for conn in net.connections:
            if conn.ref.startswith("Y"):
                crystal_ref_to_nets.setdefault(conn.ref, set()).add(net.name)
    return crystal_ref_to_nets, net_to_components


def _find_crystal_target_ic(
    crystal_ref: str,
    crystal_ref_to_nets: dict[str, set[str]],
    net_to_components: dict[str, set[str]],
    positions: dict[str, tuple[float, float, float]],
    requirements: ProjectRequirements,
) -> str | None:
    """Find the IC connected to a crystal via signal nets."""
    crystal_nets = crystal_ref_to_nets.get(crystal_ref, set())
    for net_name in crystal_nets:
        if net_name.upper() in _GND_NET_NAMES:
            continue
        for r in net_to_components.get(net_name, set()):
            if r.startswith("U") and r in positions and r != crystal_ref:
                return r
    # Fallback: use MCU ref
    from kicad_pipeline.optimization.functional_grouper import _find_mcu_ref
    return _find_mcu_ref(requirements)


def _find_best_ic_adjacent_position(
    ref: str,
    ic_x: float,
    ic_y: float,
    ic_w: float,
    ic_h: float,
    ctx: PlacementContext,
) -> tuple[float, float] | None:
    """Find the closest free position adjacent to an IC for a crystal component."""
    rx, ry, _rrot = ctx.positions[ref]
    dist = math.sqrt((rx - ic_x) ** 2 + (ry - ic_y) ** 2)
    if dist <= 10.0:
        return None
    w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
    gap = 1.0
    candidates = [
        (ic_x + (ic_w + w) / 2.0 + gap, ic_y),
        (ic_x - (ic_w + w) / 2.0 - gap, ic_y),
        (ic_x, ic_y + (ic_h + h) / 2.0 + gap),
        (ic_x, ic_y - (ic_h + h) / 2.0 - gap),
    ]
    pull_grid = _PlacementGrid(ctx.bounds)
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
    return best_pos


def _phase_crystal_placement(ctx: PlacementContext) -> None:
    """3d: Crystal-IC proximity — within 10mm of connected IC."""
    _log.info("  3d: Crystal-IC proximity")
    sc_list = list(ctx.subcircuits)

    crystal_ref_to_nets, net_to_components = _build_crystal_net_maps(ctx.requirements)

    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.CRYSTAL_OSC:
            continue
        crystal_ref = sc.anchor_ref
        target_ic = _find_crystal_target_ic(
            crystal_ref, crystal_ref_to_nets, net_to_components,
            ctx.positions, ctx.requirements,
        )
        if not target_ic or target_ic not in ctx.positions:
            continue

        ic_x, ic_y, _ic_rot = ctx.positions[target_ic]
        ic_w, ic_h = ctx.fp_sizes.get(target_ic, (5.0, 5.0))
        _log.info("    Crystal %s -> IC %s (%.1f, %.1f)",
                  crystal_ref, target_ic, ic_x, ic_y)

        for ref in sc.refs:
            if ref in ctx.fixed_refs or ref not in ctx.positions:
                continue
            _rrot = ctx.positions[ref][2]
            best_pos = _find_best_ic_adjacent_position(
                ref, ic_x, ic_y, ic_w, ic_h, ctx,
            )
            if best_pos is not None:
                rx, ry = ctx.positions[ref][0], ctx.positions[ref][1]
                old_dist = math.sqrt((rx - ic_x) ** 2 + (ry - ic_y) ** 2)
                new_dist = math.sqrt(
                    (best_pos[0] - ic_x) ** 2 + (best_pos[1] - ic_y) ** 2,
                )
                ctx.positions[ref] = (best_pos[0], best_pos[1], _rrot)
                _log.info("    %s pulled to (%.1f, %.1f) dist=%.1f->%.1f from %s",
                          ref, best_pos[0], best_pos[1], old_dist, new_dist,
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
