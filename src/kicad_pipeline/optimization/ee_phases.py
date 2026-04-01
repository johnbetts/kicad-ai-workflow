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
    from kicad_pipeline.models.requirements import Net, ProjectRequirements
    from kicad_pipeline.optimization.placement_types import PlacementContext

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
    _phase_power_chain_flow,
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
    _phase_pad_facing_optimization,
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


def _apply_ordering_chain(ctx: PlacementContext, chain: object) -> None:
    present = [r for r in chain.refs if r in ctx.positions]  # type: ignore[union-attr]
    if len(present) < 2:
        return
    xs = [ctx.positions[r][0] for r in present]
    ys = [ctx.positions[r][1] for r in present]
    use_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    vals = [ctx.positions[r][0 if use_x else 1] for r in present]
    if all(vals[i] <= vals[i + 1] + 1.0 for i in range(len(vals) - 1)):
        return
    bounds = ctx.bounds
    sorted_vals = sorted(vals)
    min_gap = 4.0
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] - sorted_vals[i - 1] < min_gap:
            sorted_vals[i] = sorted_vals[i - 1] + min_gap
    for i, ref in enumerate(present):
        rx, ry, rrot = ctx.positions[ref]
        w, h = ctx.fp_sizes.get(ref, (2.0, 1.0))
        if use_x:
            new_v = max(bounds[0] + w / 2 + 1, min(bounds[2] - w / 2 - 1, sorted_vals[i]))
            ctx.positions[ref] = (new_v, ry, rrot)
        else:
            new_v = max(bounds[1] + h / 2 + 1, min(bounds[3] - h / 2 - 1, sorted_vals[i]))
            ctx.positions[ref] = (rx, new_v, rrot)
        ctx.fixed_refs.add(ref)
    _log.info("    Reordered chain '%s': %s", chain.group, present)  # type: ignore[union-attr]


def _apply_proximity_constraint(ctx: PlacementContext, prox: object) -> None:
    if prox.ref not in ctx.positions or prox.target_ref not in ctx.positions:  # type: ignore[union-attr]
        return
    rx, ry, rrot = ctx.positions[prox.ref]  # type: ignore[union-attr]
    tx, ty, _ = ctx.positions[prox.target_ref]  # type: ignore[union-attr]
    dist = math.sqrt((rx - tx) ** 2 + (ry - ty) ** 2)
    if dist <= prox.max_distance_mm:  # type: ignore[union-attr]
        return
    ratio = prox.max_distance_mm / max(dist, 0.1)  # type: ignore[union-attr]
    new_x = tx + (rx - tx) * ratio
    new_y = ty + (ry - ty) * ratio
    bounds = ctx.bounds
    w, h = ctx.fp_sizes.get(prox.ref, (2.0, 1.0))  # type: ignore[union-attr]
    new_x = max(bounds[0] + w / 2 + 1, min(bounds[2] - w / 2 - 1, new_x))
    new_y = max(bounds[1] + h / 2 + 1, min(bounds[3] - h / 2 - 1, new_y))
    ctx.positions[prox.ref] = (new_x, new_y, rrot)  # type: ignore[union-attr]
    ctx.fixed_refs.add(prox.ref)  # type: ignore[union-attr]
    _log.info(
        "    %s: moved to (%.1f, %.1f) [proximity to %s]",
        prox.ref, new_x, new_y, prox.target_ref,  # type: ignore[union-attr]
    )


def _phase_constraint_placement(ctx: PlacementContext) -> None:
    """3-constraints: Enforce explicit placement constraints (ordering, proximity).

    Runs early in Level 3 so subsequent type-specific phases refine from a
    constraint-correct starting point.  Is a no-op when ``ctx.constraints``
    is ``None`` or when all constraint collections are empty — existing boards
    with no placement annotations are completely unaffected.
    """
    if ctx.constraints is None:
        return
    if not ctx.constraints.proximity and not ctx.constraints.ordering:
        return

    _log.info("  3-constraints: Enforcing placement constraints")
    constraints = ctx.constraints

    # Ordering runs before proximity so proximity has the final say.
    for chain in constraints.ordering:
        _apply_ordering_chain(ctx, chain)

    for prox in constraints.proximity:
        _apply_proximity_constraint(ctx, prox)


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
        ctx.fixed_refs.add(j_ref)  # protect from phase 3f/3f2 overriding rotation
        aligned += 1
        _log.info("    %s -> (%.1f, %.1f) rot=180 aligned to %s",
                   j_ref, px, py, k_ref)

    _log.info("    Aligned %d connectors to their relays", aligned)


_RELAY_DRIVER_POWER_NETS: frozenset[str] = frozenset(
    {"GND", "+5V", "+5V_RELAY", "+5V_LOGIC", "VCC"}
)
"""Power/rail nets excluded from drive-net classification."""


def _classify_relay_support_members(
    support_members: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split *support_members* into (q_refs, d_refs, all_r_refs, other_refs)."""
    q_refs = sorted(r for r in support_members if r.startswith("Q"))
    d_refs = sorted(r for r in support_members if r.startswith("D"))
    all_r_refs = sorted(r for r in support_members if r.startswith("R"))
    other_refs = sorted(
        r for r in support_members
        if not r.startswith("Q") and not r.startswith("D") and not r.startswith("R")
    )
    return q_refs, d_refs, all_r_refs, other_refs


def _split_flyback_and_led_diodes(
    d_refs: list[str],
    ctx: PlacementContext,
) -> tuple[list[str], list[str]]:
    """Split diode refs into flyback diodes and LED indicator diodes.

    A flyback diode has at least one pin on a ``_COIL`` net.
    A LED indicator diode connects to a ``LED*`` net (not a COIL net directly).

    Returns:
        (flyback_refs, led_indicator_refs)
    """
    flyback_refs: list[str] = []
    led_refs: list[str] = []
    for d_ref in d_refs:
        nets_for_d = {
            net.name for net in ctx.requirements.nets
            if any(c.ref == d_ref for c in net.connections)
        }
        if any("_COIL" in n.upper() for n in nets_for_d):
            flyback_refs.append(d_ref)
        else:
            led_refs.append(d_ref)
    return flyback_refs, led_refs


def _split_gate_resistors(
    all_r_refs: list[str],
    q_refs: list[str],
    ctx: PlacementContext,
) -> tuple[list[str], list[str]]:
    """Separate gate resistors from LED/other resistors.

    A gate resistor is one that shares a non-power, non-COIL net with a Q ref
    in the subcircuit.  LED resistors share the COIL net with Q's collector
    (they sense the coil drive), so they must be excluded by checking that the
    shared net name does not contain ``_COIL``.

    Returns:
        (r_gate_refs, r_other_refs)
    """
    r_gate_refs: list[str] = []
    r_other_refs: list[str] = []
    for r_ref in all_r_refs:
        shares_net_with_q = any(
            net.name.upper() not in _RELAY_DRIVER_POWER_NETS
            and "_COIL" not in net.name.upper()
            and any(c.ref == r_ref for c in net.connections)
            and any(c.ref in q_refs for c in net.connections)
            for net in ctx.requirements.nets
        )
        if shares_net_with_q:
            r_gate_refs.append(r_ref)
        else:
            r_other_refs.append(r_ref)
    return r_gate_refs, r_other_refs


def _find_external_gate_resistors(
    q_refs: list[str],
    r_gate_refs: list[str],
    r_other_refs: list[str],
    ctx: PlacementContext,
) -> list[str]:
    """Find gate resistors outside the subcircuit sharing a DRIVE net with Q.

    Returns list of newly discovered external gate R refs.
    """
    external: list[str] = []
    known = set(r_gate_refs) | set(r_other_refs)
    for net in ctx.requirements.nets:
        if net.name.upper() in _RELAY_DRIVER_POWER_NETS:
            continue
        if "DRIVE" not in net.name.upper():
            continue
        if not any(c.ref in q_refs for c in net.connections):
            continue
        for conn in net.connections:
            if (conn.ref.startswith("R")
                    and conn.ref not in known
                    and conn.ref in ctx.positions
                    and conn.ref not in ctx.fixed_refs):
                external.append(conn.ref)
                known.add(conn.ref)
                _log.info("    Found external gate R %s via net %s",
                           conn.ref, net.name)
    return external


def _find_coil_pin_abs_pos(
    anchor: str,
    ctx: PlacementContext,
    positions: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[float, float] | None:
    """Return the absolute (x, y) of the relay coil pin, or None if unknown.

    Scans nets for ``_COIL`` names referencing *anchor*, then finds the
    matching pad and rotates it into board coordinates.

    Args:
        positions: Override position dict (e.g. ``ctx.best_positions``
            during late refinement).  Falls back to ``ctx.positions``.
    """
    pos_map = positions if positions is not None else ctx.positions
    if anchor not in pos_map:
        return None

    # Find which pin number is the coil pin
    coil_pin: str | None = None
    for net in ctx.requirements.nets:
        if "_COIL" not in net.name.upper():
            continue
        for conn in net.connections:
            if conn.ref == anchor:
                coil_pin = conn.pin
                break
        if coil_pin is not None:
            break
    if coil_pin is None:
        return None

    # Look up pad position in the PCB footprint
    for fp in ctx.initial_pcb.footprints:
        if fp.ref != anchor:
            continue
        for pad in fp.pads:
            if str(pad.number) == str(coil_pin):
                kx, ky, krot = pos_map[anchor]
                rot_rad = math.radians(krot)
                # Rotate local pad coords into board space
                abs_x = kx + (
                    pad.position.x * math.cos(rot_rad)
                    - pad.position.y * math.sin(rot_rad)
                )
                abs_y = ky + (
                    pad.position.x * math.sin(rot_rad)
                    + pad.position.y * math.cos(rot_rad)
                )
                return abs_x, abs_y
    return None


def _place_relay_left_column(
    d_refs: list[str],
    q_refs: list[str],
    left_x: float,
    ky: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
    anchor: str = "",
) -> None:
    """Place D_flyback and Q transistor in the relay driver left column.

    Y offsets are computed from actual component sizes to avoid collisions
    regardless of package size (0402/0603/0805/SOT-23/SOD-123F).

    When a relay coil pin position is available, D is placed just below
    the coil pin (minimising flyback loop area) rather than below the
    relay body centre.
    """
    # Try to place D near coil pin for minimal flyback loop area
    coil_pos = _find_coil_pin_abs_pos(anchor, ctx) if anchor else None
    coil_y = coil_pos[1] if coil_pos is not None else None

    # Compute Y offsets based on actual sizes with rotation awareness
    relay_h = 15.0
    if anchor and anchor in ctx.positions:
        krot = ctx.positions[anchor][2]
        raw_w, raw_h = ctx.fp_sizes.get(anchor, (15.0, 15.0))
        relay_h = raw_w if krot % 180 in (90, 270) else raw_h
    elif any(r.startswith("K") for r in ctx.positions):
        relay_h = max(ctx.fp_sizes.get(r, (2.0, 15.0))[1]
                      for r in ctx.positions if r.startswith("K"))

    gap = 1.5  # mm between component edges
    # Place D just outside the relay body on the coil-pin side.
    # Driver passives ALWAYS go below the relay so the relay can sit
    # directly next to its screw terminal at the top edge.  The coil
    # driving circuit (Q, D, R) connects to coil pins on the bottom.
    cursor_y = ky + relay_h / 2.0 + gap
    direction = 1.0  # downward (increasing Y)

    for d_ref in d_refs:
        _dw, dh = ctx.fp_sizes.get(d_ref, (2.0, 2.0))
        py = cursor_y + direction * dh / 2.0
        px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
        py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
        ctx.positions[d_ref] = (px, py, 0.0)
        ctx.relay_support_refs.add(d_ref)
        cursor_y = py + direction * (dh / 2.0 + gap)
        _log.info("    D %s -> (%.1f, %.1f) LEFT col, rot=0", d_ref, px, py)

    for q_ref in q_refs:
        _qw, qh = ctx.fp_sizes.get(q_ref, (3.0, 3.4))
        py = cursor_y + direction * qh / 2.0
        px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
        py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
        ctx.positions[q_ref] = (px, py, 180.0)
        ctx.relay_support_refs.add(q_ref)
        cursor_y = py + direction * (qh / 2.0 + gap)
        _log.info("    Q %s -> (%.1f, %.1f) LEFT col, rot=180", q_ref, px, py)


def _place_relay_right_column(
    r_refs: list[str],
    right_x: float,
    ky: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
    anchor: str = "",
) -> None:
    """Place R_gate resistors in the relay driver right column."""
    # Rotation-aware relay height
    relay_h = 15.0
    if anchor and anchor in ctx.positions:
        krot = ctx.positions[anchor][2]
        raw_w, raw_h = ctx.fp_sizes.get(anchor, (15.0, 15.0))
        relay_h = raw_w if krot % 180 in (90, 270) else raw_h
    elif any(r.startswith("K") for r in ctx.positions):
        relay_h = max(ctx.fp_sizes.get(r, (2.0, 15.0))[1]
                      for r in ctx.positions if r.startswith("K"))
    gap = 1.5  # mm — matches left column courtyard clearance

    # Driver passives always below relay (same direction as left column)
    cursor_y = ky + relay_h / 2.0 + gap
    direction = 1.0

    for r_ref in r_refs:
        _rw, rh = ctx.fp_sizes.get(r_ref, (2.0, 2.0))
        py = cursor_y + direction * rh / 2.0
        px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, right_x))
        py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
        ctx.positions[r_ref] = (px, py, 180.0)
        ctx.relay_support_refs.add(r_ref)
        cursor_y = py + direction * (rh / 2.0 + gap)
        _log.info("    R %s -> (%.1f, %.1f) RIGHT col, rot=180", r_ref, px, py)


def _place_relay_others_grid(
    other_refs: list[str],
    kx: float,
    ky: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Place remaining relay driver components in a 2-wide grid below the relay."""
    if not other_refs:
        return
    # Derive grid position from relay footprint dimensions
    anchor: str | None = None
    for sc in ctx.subcircuits:
        if (sc.circuit_type == SubCircuitType.RELAY_DRIVER
                and sc.anchor_ref and sc.anchor_ref in ctx.positions):
            anchor = sc.anchor_ref
            break
    relay_h = ctx.fp_sizes.get(anchor, (15.0, 15.0))[1] if anchor else 15.0
    k_w = ctx.fp_sizes.get(anchor, (15.0, 15.0))[0] if anchor else 15.0
    other_y = ky + relay_h / 2.0 + 10.0  # below driver column (D+Q ~8mm)
    grid_spacing_x = max(5.0, k_w / 2.0 + 2.0)
    for i, ref in enumerate(other_refs):
        _w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        px = kx - grid_spacing_x + (i % 2) * (grid_spacing_x * 2)
        py = other_y + (i // 2) * (h + 0.5)
        px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
        py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
        ctx.positions[ref] = (px, py, 0.0)
        ctx.relay_support_refs.add(ref)


def _relay_driver_column_xs(
    anchor: str, kx: float, _krot: float, ctx: PlacementContext,
) -> tuple[float, float]:
    raw_w, raw_h = ctx.fp_sizes.get(anchor, (15.0, 15.0))
    k_w = raw_h if _krot % 180 in (90, 270) else raw_w
    avg_passive_w = 2.0
    left_x = kx - (k_w / 2.0 + avg_passive_w / 2.0 + 1.0)
    right_x = kx + (k_w / 2.0 + avg_passive_w / 2.0 + 1.0)
    return left_x, right_x


def _place_relay_driver_columns(
    anchor: str, kx: float, ky: float, _krot: float,
    d_refs: list[str], q_refs: list[str], r_gate_refs: list[str],
    other_refs: list[str], ctx: PlacementContext,
) -> None:
    bounds = ctx.bounds
    left_x, right_x = _relay_driver_column_xs(anchor, kx, _krot, ctx)
    coil_pos = _find_coil_pin_abs_pos(anchor, ctx)
    if coil_pos is not None and coil_pos[0] > kx:
        _place_relay_left_column(d_refs, q_refs, right_x, ky, bounds, ctx, anchor)
        _place_relay_right_column(r_gate_refs, left_x, ky, bounds, ctx, anchor)
    else:
        _place_relay_left_column(d_refs, q_refs, left_x, ky, bounds, ctx, anchor)
        _place_relay_right_column(r_gate_refs, right_x, ky, bounds, ctx, anchor)
    _place_relay_others_grid(other_refs, kx, ky, bounds, ctx)


def _phase_relay_drivers(ctx: PlacementContext) -> None:
    """3b: Relay driver placement — pad-connectivity-driven two-column layout.

    Two-column layout relative to relay anchor K:

    LEFT column (dx ~ -4.3mm from K.x) — high-current signal chain:
      D_flyback at dy=+11.5, rot=0   (anode down toward Q collector)
      Q transistor at dy=+14.1, rot=180 (collector up toward D, base toward R_gate)

    RIGHT column (dx ~ +4.0mm from K.x) — control:
      R_gate at dy=+15.4, rot=180  (connects Q base to GPIO)

    The logic: D_flyback anode and Q collector share the COIL net —
    placing them vertically with pads facing each other minimises trace
    length.  R_gate on the opposite side creates a routing channel.
    """
    _log.info("  3b: Relay driver subgroup tightening (two-column)")

    for sc in ctx.subcircuits:
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

        q_refs, d_refs, all_r_refs, other_refs = _classify_relay_support_members(
            support_members,
        )
        # ALL passives go below the relay as one unit (merged 3b + 3b2).
        # Split flyback vs LED diodes for column ordering only — both placed here.
        d_flyback_refs, d_led_refs = _split_flyback_and_led_diodes(d_refs, ctx)

        r_gate_refs, r_led_refs = _split_gate_resistors(all_r_refs, q_refs, ctx)
        r_gate_refs.extend(
            _find_external_gate_resistors(q_refs, r_gate_refs, r_led_refs, ctx),
        )
        # LED refs go into other_refs so they're placed in the column below drivers
        other_refs.extend(r_led_refs)
        other_refs.extend(d_led_refs)

        _place_relay_driver_columns(
            anchor, kx, ky, _krot, d_flyback_refs, q_refs, r_gate_refs, other_refs, ctx,
        )


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


def _build_ref_net_maps(
    requirements: ProjectRequirements,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build bidirectional ref↔net lookup maps.

    Returns:
        (ref_to_nets, net_to_refs) — both keyed by name strings.
    """
    ref_to_nets: dict[str, set[str]] = {}
    net_to_refs: dict[str, set[str]] = {}
    for net in requirements.nets:
        refs_in_net: set[str] = set()
        for conn in net.connections:
            refs_in_net.add(conn.ref)
            ref_to_nets.setdefault(conn.ref, set()).add(net.name)
        net_to_refs[net.name] = refs_in_net
    return ref_to_nets, net_to_refs


def _apply_led_via_r_on_coil(
    net: Net,
    k_ref: str,
    ref_to_nets: dict[str, set[str]],
    net_to_refs: dict[str, set[str]],
    relay_support_refs: set[str],
    relay_leds: dict[str, list[str]],
) -> None:
    """Strategy 1 helper: trace R refs on one COIL net to their D_LED refs."""
    coil_net_name = net.name
    r_refs_on_coil = [c.ref for c in net.connections if c.ref.startswith("R")]
    for r_ref in r_refs_on_coil:
        for other_net_name in ref_to_nets.get(r_ref, set()):
            if other_net_name == coil_net_name:
                continue
            for ref in net_to_refs.get(other_net_name, set()):
                if not ref.startswith("D") or ref == r_ref or ref in relay_support_refs:
                    continue
                if any("_COIL" in n.upper() for n in ref_to_nets.get(ref, set())):
                    continue  # flyback diode — skip
                led_list = relay_leds.setdefault(k_ref, [])
                if ref not in led_list:
                    led_list.append(ref)
                if r_ref not in led_list:
                    led_list.append(r_ref)


def _apply_led_strategy2(
    requirements: ProjectRequirements,
    coil_net_to_relay: dict[str, str],
    relay_support_refs: set[str],
    relay_leds: dict[str, list[str]],
) -> None:
    """Strategy 2: D refs on COIL nets + R refs via LED nets."""
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
    ref_to_nets, net_to_refs = _build_ref_net_maps(requirements)

    # Strategy 1: R refs on COIL nets → trace via LED nets → find D_LED
    for net in requirements.nets:
        if "_COIL" not in net.name.upper():
            continue
        k_ref = coil_net_to_relay.get(net.name)
        if not k_ref:
            continue
        _apply_led_via_r_on_coil(
            net, k_ref, ref_to_nets, net_to_refs, relay_support_refs, relay_leds,
        )

    # Strategy 2: D refs on COIL nets not in support → LED nets → R refs
    _apply_led_strategy2(requirements, coil_net_to_relay, relay_support_refs, relay_leds)

    return relay_leds


def _relay_led_cursor_start(
    ctx: PlacementContext, left_x: float, ky: float,
    direction: float = 1.0,
) -> float:
    """Compute cursor Y for LED column start, continuing from Q position.

    Args:
        direction: +1.0 = downward (below relay), -1.0 = upward (above relay).
    """
    gap = 1.5
    q_refs = [
        r for r in ctx.relay_support_refs
        if r.startswith("Q") and r in ctx.positions
        and abs(ctx.positions[r][0] - left_x) < 3.0
        and abs(ctx.positions[r][1] - ky) < 25.0
    ]
    if q_refs:
        if direction < 0:
            # Upward: continue above the topmost Q (smallest Y)
            return min(
                ctx.positions[qr][1] - ctx.fp_sizes.get(qr, (2.0, 2.0))[1] / 2.0
                for qr in q_refs
            ) - gap
        # Downward: continue below the bottommost Q (largest Y)
        return max(
            ctx.positions[qr][1] + ctx.fp_sizes.get(qr, (2.0, 2.0))[1] / 2.0
            for qr in q_refs
        ) + gap
    # Fallback: no Q refs found
    return ky + direction * 16.8


def _place_led_column_refs(
    ctx: PlacementContext,
    r_led_refs: list[str],
    d_led_refs: list[str],
    other_led_refs: list[str],
    left_x: float,
    cursor_y: float,
    relay_led_refs: set[str],
    direction: float = 1.0,
) -> None:
    """Place LED column refs continuing in *direction* from cursor_y.

    Args:
        direction: +1.0 = downward (increasing Y), -1.0 = upward (decreasing Y).
    """
    bounds = ctx.bounds
    gap = 1.5
    for ref in r_led_refs:
        _rw, rh = ctx.fp_sizes.get(ref, (2.0, 2.0))
        py = cursor_y + direction * rh / 2.0
        px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
        py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
        ctx.positions[ref] = (px, py, 0.0)
        ctx.relay_support_refs.add(ref)
        relay_led_refs.add(ref)
        cursor_y = py + direction * (rh / 2.0 + gap)
    for ref in d_led_refs:
        _dw, dh = ctx.fp_sizes.get(ref, (2.0, 2.0))
        py = cursor_y + direction * dh / 2.0
        px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
        py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
        ctx.positions[ref] = (px, py, 180.0)
        ctx.relay_support_refs.add(ref)
        relay_led_refs.add(ref)
        cursor_y = py + direction * (dh / 2.0 + gap)
    for i, ref in enumerate(other_led_refs):
        _ow, oh = ctx.fp_sizes.get(ref, (2.0, 2.0))
        py = cursor_y + direction * oh / 2.0
        px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x + i * 3.0))
        py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
        _, _, rot = ctx.positions[ref]
        ctx.positions[ref] = (px, py, rot)
        ctx.relay_support_refs.add(ref)
        relay_led_refs.add(ref)
        cursor_y = py + direction * (oh / 2.0 + gap)


def _phase_relay_leds(ctx: PlacementContext) -> tuple[dict[str, list[str]], set[str]]:
    """3b2: Relay LED indicator placement — same column & direction as drivers.

    All support components (D_flyback, Q, R_gate, R_LED, D_LED) go on the
    COIL SIDE of the relay.  LEDs continue the left column past Q, in the
    same direction (upward when coil pin is above relay centre, downward
    otherwise).

    Returns:
        Tuple of (relay_leds mapping, relay_led_refs set) for use by later phases.
    """
    _log.info("  3b2: Relay LED indicator placement (coil-side)")
    relay_led_refs: set[str] = set()

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

        left_x = kx - 4.3

        # Determine coil direction — must match driver column direction
        coil_pos = _find_coil_pin_abs_pos(k_ref, ctx)
        direction = -1.0 if (coil_pos is not None and coil_pos[1] < ky) else 1.0

        r_led_refs = sorted(r for r in led_members if r.startswith("R"))
        d_led_refs = sorted(r for r in led_members if r.startswith("D"))
        other_led_refs = sorted(
            r for r in led_members if not r.startswith("R") and not r.startswith("D")
        )

        cursor_y = _relay_led_cursor_start(ctx, left_x, ky, direction)
        _place_led_column_refs(
            ctx, r_led_refs, d_led_refs, other_led_refs,
            left_x, cursor_y, relay_led_refs, direction,
        )
        _log.info("    3b2: placed %d LED refs for %s (left col at x=%.1f, dir=%.0f)",
                   len(led_members), k_ref, left_x, direction)

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
            if edge_dist <= 2.5:
                continue  # already within 2.5mm edge-to-edge — good enough

            dx = ix - cx
            dy = iy - cy
            d = math.sqrt(dx * dx + dy * dy) or 1.0
            target_dist = (iw + cw) / 2.0 + 1.0  # 1mm edge-to-edge gap
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


def _is_usb_connector(ref: str, pcb: object) -> bool:
    """Return True if *ref* is a USB connector (not a screw terminal)."""
    for fp in pcb.footprints:  # type: ignore[union-attr]
        if fp.ref == ref:
            val = (fp.value or "").upper()
            lib = (fp.lib_id or "").upper()
            return "USB" in val or "USB" in lib
    return False


def _is_pin_header(ref: str, pcb: object) -> bool:
    """Return True if *ref* is a pin header (not a screw terminal).

    Pin headers (PinHeader, IDC, JST, Molex) are NOT top-edge screw
    terminals and should not be placed there.  Only genuine screw/push-wire
    terminal blocks belong at the top edge for sensor wiring.
    """
    for fp in pcb.footprints:  # type: ignore[union-attr]
        if fp.ref == ref:
            val = (fp.value or "").upper()
            lib = (fp.lib_id or "").upper()
            return any(kw in val or kw in lib for kw in (
                "PINHEADER", "PIN_HEADER", "HEADER", "IDC", "JST", "MOLEX",
                "CONNECTOR_PINHEADER",
            ))
    return False


def _phase_relay_power_isolation(ctx: PlacementContext) -> None:
    """3b3: Place relay power isolation components (ferrites, bulk caps) at board bottom.

    Pattern: L1/L2 (ferrite beads) and C1/C2 (bulk caps) placed in a row
    near the bottom-left of the board, clear of relay driver columns.
    """
    min_x, min_y, max_x, max_y = ctx.bounds

    # Find ONLY relay-group ferrite beads and bulk caps — NOT all L/C on the board.
    # Previous bug: this grabbed ALL capacitors/inductors, destroying the entire layout.
    relay_block_refs: set[str] = set()
    for fb in ctx.requirements.features:
        if any(kw in fb.name.lower() for kw in ("relay", "output", "switching")):
            relay_block_refs.update(fb.components)

    power_refs = [
        r for r in ctx.positions
        if r in relay_block_refs
        and (r.startswith("L") or r.startswith("C"))
        and r not in ctx.relay_support_refs
        and r not in getattr(ctx, "top_edge_connector_refs", set())
    ]

    if not power_refs:
        return

    _log.info("  3b3: Relay power isolation (%s)", power_refs)
    # Place in a row near bottom-left, clear of mounting holes
    cursor_x = min_x + 10.0
    row_y = max_y - 3.0  # near bottom edge
    gap = 1.5

    for ref in sorted(power_refs):
        w, _h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        px = cursor_x + w / 2.0
        py = row_y
        # Clamp to board
        px = max(min_x + 2.0, min(max_x - 2.0, px))
        py = max(min_y + 2.0, min(max_y - 2.0, py))
        ctx.positions[ref] = (px, py, 90.0 if ref.startswith("L") else 0.0)
        cursor_x = px + w / 2.0 + gap
        _log.info("    %s -> (%.1f, %.1f)", ref, px, py)


def _phase_top_edge_connectors(ctx: PlacementContext) -> None:
    """3f2: Top-edge screw terminal ordering.

    USB connectors (USB-C, USB-A, Micro-USB) and pin headers are excluded —
    the MCU group phase handles their placement and rotation.  Only genuine
    screw/push-wire terminal blocks (TerminalBlock) are placed here.
    """
    min_x, min_y, max_x, max_y = ctx.bounds

    # Collect refs that are in mcu_peripheral_cluster subcircuits so we can
    # exclude them — they belong near the MCU, not at the top sensor edge.
    _mcu_peripheral_sc_refs: set[str] = set()
    from kicad_pipeline.optimization.functional_grouper import SubCircuitType
    for sc in ctx.subcircuits:
        if sc.circuit_type == SubCircuitType.MCU_PERIPHERAL_CLUSTER:
            _mcu_peripheral_sc_refs.update(sc.refs)

    top_edge_order = ["J1", "J2", "J3", "J4", "J5", "J6"]
    # Terminal blocks MUST go to top edge regardless of fixed_refs —
    # earlier phases may have fixed them in wrong positions.
    _top_refs = [
        r for r in top_edge_order
        if r in ctx.positions
        and r not in _mcu_peripheral_sc_refs
        and not _is_usb_connector(r, ctx.initial_pcb)
        and not _is_pin_header(r, ctx.initial_pcb)
    ]
    if not _top_refs:
        return

    _log.info("  3f2: Top-edge screw terminal ordering (%s)", _top_refs)

    # Terminals associated with relays get aligned to their relay's X.
    # Others use evenly-spaced ordering across the top edge.
    connector_to_relay = _build_connector_to_relay_map(ctx.requirements)
    relay_aligned: set[str] = set()
    origin_y_target = min_y + 3.0

    for r in _top_refs:
        k_ref = connector_to_relay.get(r)
        if k_ref and k_ref in ctx.positions:
            kx = ctx.positions[k_ref][0]
            px = max(min_x + 2.0, min(max_x - 2.0, kx))
            ctx.positions[r] = (px, origin_y_target, 0.0)
            relay_aligned.add(r)
            _log.info("    %s -> origin(%.1f, %.1f) rot=0 (aligned to %s)",
                       r, px, origin_y_target, k_ref)

    # Remaining terminals: evenly spaced across the top edge
    non_relay_refs = [r for r in _top_refs if r not in relay_aligned]
    if non_relay_refs:
        term_gap = 3.0
        term_widths = [ctx.fp_sizes.get(r, (2.0, 2.0))[0] for r in non_relay_refs]
        total_w = sum(term_widths) + term_gap * (len(non_relay_refs) - 1)
        margin = 8.0
        avail_w = (max_x - min_x) - 2 * margin
        if total_w < avail_w:
            start_x = min_x + margin + (avail_w - total_w) / 2.0
        else:
            compressed_gap = max(1.0, (avail_w - sum(term_widths)) / max(len(non_relay_refs) - 1, 1))
            term_gap = compressed_gap
            start_x = min_x + margin
        cursor_x = start_x
        for i, r in enumerate(non_relay_refs):
            tw = term_widths[i]
            origin_x = cursor_x + tw / 2.0
            ctx.positions[r] = (origin_x, origin_y_target, 0.0)
            cursor_x += tw + term_gap
            _log.info("    %s -> origin(%.1f, %.1f) rot=0", r, origin_x, origin_y_target)
    ctx.top_edge_connector_refs = set(_top_refs)


def _phase_all_connectors_to_edges(ctx: PlacementContext) -> None:
    """3f3: Pin ALL remaining connectors to the board edge nearest their group.

    Before pushing a connector to an edge, we determine which functional
    group (FeatureBlock) the connector belongs to, compute the centroid of
    that group, and push the connector to the edge closest to the *group
    centroid* — not the connector's current position.  The connector is also
    aligned along that edge at the group centroid's coordinate so it stays
    physically close to its functional group.

    Handles USB connectors, pin headers, and any J-prefix component
    not already placed by ``_phase_top_edge_connectors``.
    """
    min_x, min_y, max_x, max_y = ctx.bounds
    margin = 3.0  # small inset from edge (BOARD_EDGE_MARGIN_MM + 1.0)

    already_placed = getattr(ctx, "top_edge_connector_refs", set())

    # --- Build connector → functional-group map and group centroids ---
    # Map each ref to its FeatureBlock name
    ref_to_group: dict[str, str] = {}
    group_refs: dict[str, list[str]] = {}
    for fb in ctx.requirements.features:
        for comp_ref in fb.components:
            ref_to_group[comp_ref] = fb.name
            group_refs.setdefault(fb.name, []).append(comp_ref)

    # Compute group centroids from current positions
    group_centroids: dict[str, tuple[float, float]] = {}
    for gname, refs in group_refs.items():
        xs: list[float] = []
        ys: list[float] = []
        for r in refs:
            if r in ctx.positions:
                px, py, _ = ctx.positions[r]
                xs.append(px)
                ys.append(py)
        if xs:
            group_centroids[gname] = (sum(xs) / len(xs), sum(ys) / len(ys))

    for ref, (cx, cy, rot) in list(ctx.positions.items()):
        if ref in ctx.fixed_refs or ref in already_placed:
            continue
        if not ref.startswith("J"):
            continue

        # Already within 5mm of an edge — close enough
        d_left = cx - min_x
        d_right = max_x - cx
        d_top = cy - min_y
        d_bottom = max_y - cy
        min_dist = min(d_left, d_right, d_top, d_bottom)
        if min_dist <= 5.0:
            continue

        # Determine target edge based on connector type
        comp = next(
            (c for c in ctx.requirements.components if c.ref == ref), None,
        )
        fp_name = comp.footprint.upper() if comp else ""

        # Screw terminals ALWAYS go to TOP edge (industrial wiring convention)
        if "TERMINAL" in fp_name or "TB_" in fp_name:
            target_edge = "top"
        # RJ45 connectors go to RIGHT edge
        elif "RJ45" in fp_name or "RJ45" in ref.upper():
            target_edge = "right"
        # USB-C goes to BOTTOM or LEFT edge (user-facing)
        elif "USB" in fp_name:
            target_edge = "bottom"
        else:
            # Other connectors: use group centroid to pick nearest edge
            gname = ref_to_group.get(ref)
            gcx, gcy = group_centroids.get(
                gname, (cx, cy),
            ) if gname else (cx, cy)
            gd_left = gcx - min_x
            gd_right = max_x - gcx
            gd_top = gcy - min_y
            gd_bottom = max_y - gcy
            gd_min = min(gd_left, gd_right, gd_top, gd_bottom)
            if gd_min == gd_left:
                target_edge = "left"
            elif gd_min == gd_right:
                target_edge = "right"
            elif gd_min == gd_top:
                target_edge = "top"
            else:
                target_edge = "bottom"

        # Get group centroid for along-edge positioning
        gname = ref_to_group.get(ref)
        gcx, gcy = group_centroids.get(
            gname, (cx, cy),
        ) if gname else (cx, cy)

        # Push to target edge, aligned at group centroid coordinate
        if target_edge == "left":
            along = max(min_y + margin, min(gcy, max_y - margin))
            new_x, new_y, new_rot = min_x + margin, along, rot
        elif target_edge == "right":
            along = max(min_y + margin, min(gcy, max_y - margin))
            new_x, new_y, new_rot = max_x - margin, along, rot
        elif target_edge == "top":
            along = max(min_x + margin, min(gcx, max_x - margin))
            new_x, new_y, new_rot = along, min_y + margin, rot
        else:
            along = max(min_x + margin, min(gcx, max_x - margin))
            new_x, new_y, new_rot = along, max_y - margin, rot

        _log.info(
            "  3f3: %s (group=%s) pushed to edge (%.1f,%.1f) -> (%.1f,%.1f)",
            ref, gname or "?", cx, cy, new_x, new_y,
        )
        ctx.positions[ref] = (new_x, new_y, new_rot)


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
