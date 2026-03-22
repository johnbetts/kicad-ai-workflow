"""Placement optimizers: deterministic EE-grade and simulated annealing.

Provides two placement strategies:
- ``optimize_placement_ee()``: Deterministic 5-phase EE-grade placement
  using functional grouping, voltage domain zones, MST placement, and an
  automated review loop.
- ``optimize_placement_sa()`` (legacy): SA with random perturbations.

``optimize_placement`` is an alias for ``optimize_placement_ee``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.optimization.collision_resolver import (
    _count_collisions,
    _fp_courtyard_sizes,
    _group_of_ref,  # noqa: F401 - re-exported
    _PlacementGrid,
    _resolve_collisions,
    _rotation_aware_size,
)
from kicad_pipeline.optimization.functional_grouper import (
    SubCircuitType,
)
from kicad_pipeline.optimization.group_helpers import (
    _apply_review_fixes,
    _assign_zone_position,  # noqa: F401 - re-exported
    _build_group_map,  # noqa: F401 - re-exported
    _centroid,  # noqa: F401 - re-exported
    _extract_group_bboxes,
    _group_footprint_area,  # noqa: F401 - re-exported
    _place_subcircuit_group,  # noqa: F401 - re-exported
)
from kicad_pipeline.optimization.level3_phases import (
    _apply_cross_domain_affinity_overrides,  # noqa: F401 - re-exported
    _apply_template_refinement,
    _classify_connector_function,  # noqa: F401 - re-exported
    _orient_connectors,
    _pin_connectors_by_function,  # noqa: F401 - re-exported
    _pin_rf_to_edge,
    _place_adc_channels,  # noqa: F401 - re-exported
    _place_boundary_regulators,  # noqa: F401 - re-exported
    _place_row_layout,
    _pull_mcu_peripherals,  # noqa: F401 - re-exported
)
from kicad_pipeline.optimization.placement_guard import (
    PlacementGuardResult,  # noqa: F401 - re-exported
    validate_placement,
)
from kicad_pipeline.optimization.placement_types import (
    OptimizationConfig,  # noqa: F401 - re-exported
    PlacementCandidate,  # noqa: F401 - re-exported
    _apply_positions,
    _board_bounds,
    _dict_to_positions,
    _extract_positions,  # noqa: F401 - re-exported
    _get_movable_refs,  # noqa: F401 - re-exported
    _is_fixed,
    _positions_to_dict,  # noqa: F401 - re-exported
)
from kicad_pipeline.optimization.sa_optimizer import (
    _find_colliding_pairs,  # noqa: F401 - re-exported
    _perturbation_nudge,  # noqa: F401 - re-exported
    _perturbation_pull_connected,  # noqa: F401 - re-exported
    _perturbation_resolve_collision,  # noqa: F401 - re-exported
    _perturbation_rotate,  # noqa: F401 - re-exported
    _perturbation_swap,  # noqa: F401 - re-exported
    optimize_placement_sa,  # noqa: F401 - re-exported
)
from kicad_pipeline.pcb.pin_map import (
    centroid_to_origin,
    compute_centroid_offset,
    origin_to_centroid,
)

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.review_agent import PlacementReview

_log = logging.getLogger(__name__)


# compute_centroid_offset is imported from kicad_pipeline.pcb.pin_map
# Alias kept for backwards compatibility with test imports.
_centroid_offset = compute_centroid_offset


def _optimize_placement_ee_v4(
    requirements: ProjectRequirements,
    initial_pcb: PCBDesign,
    max_review_passes: int = 5,
) -> tuple[PCBDesign, PlacementReview]:
    """Legacy v4 optimizer (15-phase). Preserved for fallback.

    Use ``optimize_placement_ee()`` (v5, 3-level) instead.
    """
    from kicad_pipeline.optimization.functional_grouper import (
        classify_voltage_domains,
        detect_subcircuits,
    )
    from kicad_pipeline.optimization.review_agent import review_placement

    subcircuits = detect_subcircuits(requirements)
    domain_map = classify_voltage_domains(requirements)

    # Simplified: just run review on current placement and return
    review = review_placement(initial_pcb, requirements, subcircuits=subcircuits,
                              domain_map=domain_map)
    return initial_pcb, review


def optimize_placement_ee(
    requirements: ProjectRequirements,
    initial_pcb: PCBDesign,
    max_review_passes: int = 5,
) -> tuple[PCBDesign, PlacementReview]:
    """3-level hierarchical placement optimizer (v5).

    Replaces the 15-phase v4 optimizer with a clean top-down pipeline:

    **Level 1 — Zone Partitioning**: Partition the board into non-overlapping
    rectangular zones based on FeatureBlock groups.

    **Level 2 — Group Placement**: Place each FeatureBlock as a rigid unit
    within its assigned zone. Internal component offsets from ``_layout_group()``
    are preserved. Groups never mix between zones.

    **Level 3 — Intra-Group Refinement**: Fine-tune within groups:
    relay row formation, decoupling cap tightening, connector edge pinning,
    RF edge placement, connector orientation, collision resolution.

    Single-pass, deterministic. Each level is complete before the next starts.
    No level undoes work from a previous level.

    Args:
        requirements: Project requirements with components and nets.
        initial_pcb: Starting PCB with initial placement.
        max_review_passes: Max iterations of the review-fix loop.

    Returns:
        Tuple of (optimized PCBDesign, final PlacementReview).
    """
    from kicad_pipeline.optimization.functional_grouper import (
        classify_voltage_domains,
        compute_power_flow_topology,
        detect_subcircuits,
    )
    from kicad_pipeline.optimization.review_agent import review_placement

    fp_sizes = _fp_courtyard_sizes(initial_pcb)
    bounds = _board_bounds(initial_pcb)
    min_x, min_y, max_x, max_y = bounds

    fixed_refs: set[str] = {
        fp.ref for fp in initial_pcb.footprints
        if _is_fixed(fp.ref, requirements)
    }

    has_groups = bool(requirements.features)

    # ===================================================================
    # Level 1: Zone Partitioning
    # ===================================================================
    _log.info("=== Level 1: Zone Partitioning ===")
    subcircuits = detect_subcircuits(requirements)
    domain_map = classify_voltage_domains(requirements)
    topology = compute_power_flow_topology(subcircuits)
    sc_list = list(subcircuits)

    from kicad_pipeline.optimization.zone_partitioner import partition_board
    zones = partition_board(bounds, list(requirements.features), topology)
    _log.info("  %d zones created", len(zones))

    # ===================================================================
    # Level 2: Group Placement (groups as rigid units)
    # ===================================================================
    _log.info("=== Level 2: Group Placement ===")

    # Extract current positions — convert KiCad origin → centroid space.
    # KiCad stores footprint origin (pin 1 for connectors), but the
    # optimizer works in centroid-of-pads coordinates.
    positions: dict[str, tuple[float, float, float]] = {}
    for fp in initial_pcb.footprints:
        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        positions[fp.ref] = (cx, cy, fp.rotation)

    if has_groups and zones:
        from kicad_pipeline.optimization.group_placer import (
            pin_connectors_to_edge,
            place_groups,
        )

        # Extract internal layouts per group from current positions
        # (these come from _layout_group() via build_pcb)
        internal_layouts: dict[str, dict[str, tuple[float, float, float]]] = {}
        for block in requirements.features:
            layout: dict[str, tuple[float, float, float]] = {}
            refs_in_pos = [r for r in block.components if r in positions]
            if not refs_in_pos:
                continue
            # Use current positions as the internal layout
            for ref in refs_in_pos:
                x, y, rot = positions[ref]
                layout[ref] = (x, y, rot)
            internal_layouts[block.name] = layout

        # Place groups as rigid units within zones
        placed_groups = place_groups(
            zones, list(requirements.features),
            internal_layouts, fp_sizes, bounds,
        )

        # Merge placed group positions back into main positions dict
        # Skip fixed refs — they must not be moved
        for pg in placed_groups:
            for ref, (px, py) in pg.positions.items():
                if ref in fixed_refs:
                    continue
                if ref in positions:
                    _, _, rot = positions[ref]  # preserve rotation
                    positions[ref] = (px, py, rot)

        # Pin connectors to board edges
        _log.info("  Pinning connectors to board edges")
        edge_positions = pin_connectors_to_edge(
            placed_groups, fp_sizes, bounds, fixed_refs,
        )
        for ref, (px, py) in edge_positions.items():
            if ref.startswith("J") and ref not in fixed_refs and ref in positions:
                _, _, rot = positions[ref]
                positions[ref] = (px, py, rot)

        _log.info("  %d groups placed", len(placed_groups))
    else:
        _log.info("  No groups — using initial placement")

    # ===================================================================
    # Level 3: Intra-Group Refinement
    # ===================================================================
    _log.info("=== Level 3: Intra-Group Refinement ===")

    # 3a. Relay row formation — arrange relays in 1xN horizontal row
    _log.info("  3a: Relay row formation")
    positions = _place_row_layout(sc_list, positions, fp_sizes, bounds, fixed_refs,
                                   zones=zones)

    # 3b. Relay driver subgroup tightening — Q+D+R within 8mm of K
    # Place support directly below relay in tight grid. Protected during 3g
    # collision resolution — overlapping components will be moved instead.
    _log.info("  3b: Relay driver subgroup tightening")
    relay_support_refs: set[str] = set()  # protected from collision resolution
    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.RELAY_DRIVER:
            continue
        anchor = sc.anchor_ref
        if anchor not in positions:
            continue
        kx, ky, _krot = positions[anchor]
        kw, kh = fp_sizes.get(anchor, (18.0, 16.0))
        # Swap dimensions for rotated relays (90° or 270°)
        if _krot % 180 in (90.0, 270.0):
            kw, kh = kh, kw

        support_members = [
            r for r in sc.refs
            if r != anchor and r in positions and r not in fixed_refs
        ]
        support_members.sort(key=lambda r: (
            0 if r.startswith("Q") else 1 if r.startswith("D") else 2, r,
        ))

        target_y_base = ky + kh / 2.0 + 0.5
        col = 0
        row_y = target_y_base
        row_max_h = 0.0
        cols_per_row = 3  # 3-column single row for Q+D+R

        for ref in support_members:
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            # Direct placement — no grid search. Force position.
            px = kx - kw / 2.0 + (col + 0.5) * (kw / cols_per_row)
            py = row_y + h / 2.0
            # Clamp to board
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
            _, _, rot = positions[ref]
            _log.debug(
                "    3b: %s → %s, placed at (%.1f, %.1f) under %s",
                ref, anchor, px, py, anchor,
            )
            positions[ref] = (px, py, rot)
            relay_support_refs.add(ref)
            row_max_h = max(row_max_h, h)
            col += 1
            if col >= cols_per_row:
                col = 0
                row_y += row_max_h + 0.5
                row_max_h = 0.0

    # 3b2. Relay LED indicator placement — D_LED + R_LED below support grid
    # Each relay has a status LED (D18-D21) and current-limiting resistor
    # (R33-R36) connected via K*_COIL and K*_LED nets. Place them consistently
    # below each relay's support components.
    _log.info("  3b2: Relay LED indicator placement")
    relay_led_refs: set[str] = set()
    # Build relay coil net → relay mapping
    _coil_net_to_relay: dict[str, str] = {}
    for net in requirements.nets:
        if "_COIL" in net.name.upper():
            for conn in net.connections:
                if conn.ref.startswith("K"):
                    _coil_net_to_relay[net.name] = conn.ref
                    break

    # Find LED+resistor pairs per relay via coil nets
    _relay_leds: dict[str, list[str]] = {}  # K_ref → [D_led, R_led]
    for net in requirements.nets:
        if "_COIL" in net.name.upper():
            k_ref = _coil_net_to_relay.get(net.name)
            if not k_ref:
                continue
            for conn in net.connections:
                if conn.ref.startswith("D") and conn.ref not in relay_support_refs:
                    _relay_leds.setdefault(k_ref, []).append(conn.ref)
        elif "_LED" in net.name.upper():
            # K*_LED nets connect D_led to R_led
            d_refs_in = [c.ref for c in net.connections if c.ref.startswith("D")]
            r_refs_in = [c.ref for c in net.connections if c.ref.startswith("R")]
            for d_ref in d_refs_in:
                # Find which relay this D belongs to
                for k_ref, led_list in _relay_leds.items():
                    if d_ref in led_list:
                        led_list.extend(r_refs_in)
                        break

    # Place LED pairs below each relay's support row
    for k_ref in sorted(_relay_leds):
        if k_ref not in positions:
            continue
        kx, ky, _krot = positions[k_ref]
        kw, kh = fp_sizes.get(k_ref, (18.0, 16.0))
        if _krot % 180 in (90.0, 270.0):
            kw, kh = kh, kw

        led_members = sorted(set(_relay_leds[k_ref]))
        led_members = [r for r in led_members if r in positions and r not in fixed_refs]
        if not led_members:
            continue

        # Place LED row below support grid (support is at ky + kh/2 + ~5mm)
        led_y_base = ky + kh / 2.0 + 5.5
        led_col = 0
        led_cols_per_row = min(len(led_members), 2)
        for ref in led_members:
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            px = kx - kw / 4.0 + led_col * (kw / 2.0)
            py = led_y_base + h / 2.0
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
            _, _, rot = positions[ref]
            positions[ref] = (px, py, rot)
            relay_support_refs.add(ref)
            relay_led_refs.add(ref)
            led_col += 1
            if led_col >= led_cols_per_row:
                led_col = 0
                led_y_base += h + 0.5

        _log.info("    3b2: placed %d LED refs for %s", len(led_members), k_ref)

    # 3c. Decoupling cap tightening — within 3-5mm of IC
    _log.info("  3c: Decoupling cap tightening")
    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.DECOUPLING:
            continue
        ic_ref = sc.anchor_ref
        if ic_ref not in positions:
            continue
        ix, iy, _irot = positions[ic_ref]
        iw, ih = fp_sizes.get(ic_ref, (5.0, 5.0))

        for cap_ref in sc.refs:
            if cap_ref == ic_ref or not cap_ref.startswith("C"):
                continue
            if cap_ref not in positions or cap_ref in fixed_refs:
                continue
            cx, cy, crot = positions[cap_ref]
            cw, ch = fp_sizes.get(cap_ref, (1.5, 1.0))

            dist = math.sqrt((cx - ix) ** 2 + (cy - iy) ** 2)
            edge_dist = max(0.0, dist - (iw + cw) / 2.0)
            if edge_dist <= 4.0:
                continue

            # Force cap to within 2mm of IC edge — direct placement
            dx = ix - cx
            dy = iy - cy
            d = math.sqrt(dx * dx + dy * dy) or 1.0
            target_dist = (iw + cw) / 2.0 + 1.5
            tx = ix - dx / d * target_dist
            ty = iy - dy / d * target_dist
            # Clamp to board
            tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
            ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
            positions[cap_ref] = (tx, ty, crot)

    # 3c1. Power group organization — IC-anchored fork/branch layout
    # Components anchor at IC pin positions (not zone top-left) to minimize
    # hot-loop area for EMI.  VIN-side passives ABOVE U1, output-side BELOW.
    # U2 (3.3V buck) branches RIGHT from the +5V output rail of U1.
    #   Column 1: VIN passives → U1 → BST/SW → L1 → FB → output cap
    #   Bridge: OR diodes + bulk cap (at fork point)
    #   Column 2 (RIGHT): U2 + BST2/SW2 → L2 → 3.3V caps
    #   Tail: LED, ferrites, misc — below both columns
    _log.info("  3c1: Power group organization")
    power_group_fixed: set[str] = set()

    # Find power FeatureBlock
    power_group_refs: set[str] = set()
    for feat in requirements.features:
        feat_lower = feat.name.lower()
        if "power" in feat_lower or "supply" in feat_lower:
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                power_group_refs.add(r)
            break

    if power_group_refs:
        # Find power zone
        power_zone_rect: tuple[float, float, float, float] | None = None
        for z in zones:
            if z.name == "power":
                power_zone_rect = z.rect
                break

        # Classify power components by net connectivity
        net_to_pwr_refs: dict[str, set[str]] = {}
        for net in requirements.nets:
            pwr_in_net = set()
            for conn in net.connections:
                if conn.ref in power_group_refs:
                    pwr_in_net.add(conn.ref)
            if pwr_in_net:
                net_to_pwr_refs[net.name] = pwr_in_net

        power_ics = sorted(
            [r for r in power_group_refs
             if r.startswith("U") and r in positions],
        )
        power_connectors = sorted(
            [r for r in power_group_refs
             if r.startswith("J") and r in positions],
        )

        if power_ics and power_zone_rect is not None:
            zx1, zy1, zx2, zy2 = power_zone_rect

            _STRIP_GAP = 0.5   # vertical gap between components
            _COL_SPACING = 8.0  # horizontal gap from U1 to U2 column
            _IC_MARGIN = 3.0    # max distance from IC for its passives

            placed_in_col: set[str] = set()  # prevent double-placement

            # Define the two buck converter signal chains
            buck1_ic = power_ics[0] if power_ics else ""
            buck2_ic = power_ics[1] if len(power_ics) > 1 else ""

            # Ferrite detection
            ferrite_refs = sorted(
                [r for r in power_group_refs
                 if r.startswith("L") and r in positions
                 and "ferrite" in (
                     next((fp.value for fp in initial_pcb.footprints
                           if fp.ref == r), "")
                 ).lower()],
            )

            # Buck #1 column: VIN input → 5V output
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
                 if r.startswith("L") and r in positions
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

            # VIN-side components go ABOVE U1; output-side in 2-column
            # grid BELOW U1 to minimize vertical span (hot loop area).
            vin_above: list[str] = vin_passives  # D5, C1 (input side)
            # Split output passives into two sub-columns below U1:
            # Left sub-col: switching path (BST cap + inductor + output cap)
            # Right sub-col: feedback divider (just R1, R2)
            # This keeps max 2 items per column for tightest packing.
            # C3 goes with L1 (same BUCK_5V net: L1 output → C3).
            output_left: list[str] = bst1_passives + l1_refs + buck5v_caps  # C2, L1, C3
            output_right: list[str] = fb_refs                                # R1, R2

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

            # Buck #2 column: +5V input → 3.3V output
            # Caps on the +5V rail that aren't bulk (C4) belong with U2 as its
            # input decoupling.  The +5V rail IS U2's input voltage.
            _v5_caps_in_power = sorted(
                net_to_pwr_refs.get("+5V", set())
                & {r for r in power_group_refs if r.startswith("C")}
                - {"C4"},  # keep bulk cap in bridge column
            )
            buck2_in_caps = _v5_caps_in_power
            # Remove from v5_rail BEFORE building bridge_column
            v5_rail_refs = [r for r in v5_rail_refs if r not in buck2_in_caps]
            bridge_column: list[str] = or_diode_refs + v5_rail_refs
            bst2_passives = sorted(
                (net_to_pwr_refs.get("BST2", set())
                 | net_to_pwr_refs.get("SW2", set()))
                - {buck2_ic} - set(power_connectors),
            )
            l2_refs = sorted(
                [r for r in power_group_refs
                 if r.startswith("L") and r in positions
                 and r not in ferrite_refs
                 and r in net_to_pwr_refs.get("SW2", set())],
            )
            v33_caps = sorted(
                net_to_pwr_refs.get("+3V3", set())
                & power_group_refs
                - {buck2_ic} - set(power_connectors),
            )

            # Buck2: split into 2 sub-columns like buck1
            # Left: IC + switching path (C5→U2→C17→L2)
            # Right: output caps (C6, C7)
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
                power_group_refs - all_classified - {""} - fixed_refs,
            )
            tail_column: list[str] = led_refs + ferrite_refs + remaining_refs

            # --- IC-anchored placement ---
            # Anchor at U1's current (or zone-centered) position.
            # U1 stays at its position; VIN passives stack ABOVE,
            # output passives stack BELOW.
            u1_x, u1_y, _u1_rot = positions.get(
                buck1_ic, (zx1 + 5.0, zy1 + 15.0, 0.0),
            )
            u1_w, u1_h = fp_sizes.get(buck1_ic, (5.0, 5.0))
            # Center U1 within left half of power zone — leaves room for
            # left sub-column AND U2 fork column to the right.
            zone_quarter_x = zx1 + (zx2 - zx1) * 0.25
            anchor_x = max(
                zx1 + 6.0,  # room for left sub-col (3mm offset + 3mm margin)
                min(zx2 - _COL_SPACING - 3.0, zone_quarter_x),
            )

            # Build occupancy grid of non-power components so power
            # columns don't overlap with other groups' components.
            _pwr_grid = _PlacementGrid(bounds)
            for _oref, (_ox, _oy, _or) in positions.items():
                if _oref in power_group_refs:
                    continue
                _ow, _oh = _rotation_aware_size(_oref, positions, fp_sizes)
                _pwr_grid.place(_ox, _oy, _ow, _oh)

            # Clamp bounds: use zone for X (stay in power column), but
            # use BOARD bounds for Y — the power column can extend below
            # the zone into unused space rather than squishing components.
            _pz_x1 = zx1 + 2.0
            _pz_y1 = zy1 + 2.0
            _pz_x2 = zx2 - 2.0
            _pz_y2 = max_y - 3.0  # use board bottom, not zone bottom

            def _place_column(
                refs: list[str],
                col_x: float,
                start_y: float,
            ) -> float:
                """Place refs in a vertical column. Returns bottom Y."""
                cy = start_y
                for ref in refs:
                    if (ref not in positions or ref in fixed_refs
                            or ref in placed_in_col or ref == ""):
                        continue
                    w, h = fp_sizes.get(ref, (2.0, 2.0))
                    tx = max(_pz_x1, min(_pz_x2, col_x))
                    ty = max(_pz_y1, min(_pz_y2, cy + h / 2.0))
                    px, py = tx, ty
                    positions[ref] = (px, py, 0.0)
                    _pwr_grid.place(px, py, w, h)
                    power_group_fixed.add(ref)
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
                    if (ref not in positions or ref in fixed_refs
                            or ref in placed_in_col or ref == ""):
                        continue
                    w, h = fp_sizes.get(ref, (2.0, 2.0))
                    tx = max(_pz_x1, min(_pz_x2, col_x))
                    ty = max(_pz_y1, min(_pz_y2, cy - h / 2.0))
                    px, py = tx, ty
                    positions[ref] = (px, py, 0.0)
                    _pwr_grid.place(px, py, w, h)
                    power_group_fixed.add(ref)
                    placed_in_col.add(ref)
                    cy = py - h / 2.0 - _STRIP_GAP
                return cy

            # Register connectors (don't move them)
            for ref in power_connectors:
                power_group_fixed.add(ref)

            # Place U1 at its anchor position (direct, no grid search)
            _SUB_COL_OFFSET = 3.5  # half-width between 2-column grid
            if buck1_ic and buck1_ic not in fixed_refs:
                px = max(_pz_x1, min(_pz_x2, anchor_x))
                py = max(_pz_y1, min(_pz_y2, u1_y))
                positions[buck1_ic] = (px, py, 0.0)
                _pwr_grid.place(px, py, u1_w, u1_h)
                power_group_fixed.add(buck1_ic)
                placed_in_col.add(buck1_ic)
                u1_x, u1_y = px, py

            # VIN passives ABOVE U1 (stacking upward)
            vin_top = u1_y - u1_h / 2.0 - _STRIP_GAP
            _place_column_upward(vin_above, anchor_x, vin_top)

            # Output passives in 2-column grid BELOW U1
            output_top = u1_y + u1_h / 2.0 + _STRIP_GAP
            left_x = anchor_x - _SUB_COL_OFFSET
            right_x = anchor_x + _SUB_COL_OFFSET
            left_bottom = _place_column(output_left, left_x, output_top)
            right_bottom = _place_column(output_right, right_x, output_top)

            # Bridge: OR diodes + 5V bulk — below both sub-columns
            # Split bridge into 2 sub-columns to avoid vertical stacking
            bridge_top = max(left_bottom, right_bottom)
            bridge_left = bridge_column[:len(bridge_column) // 2 + 1]
            bridge_right = bridge_column[len(bridge_column) // 2 + 1:]
            fork_y_l = _place_column(bridge_left, left_x, bridge_top)
            fork_y_r = _place_column(bridge_right, right_x, bridge_top)
            fork_y = max(fork_y_l, fork_y_r)

            # Column 2: Buck #2 (5V → 3.3V) — branches RIGHT at fork point
            col2_x = anchor_x + _COL_SPACING
            col2_top = output_top  # start at same level as U1 output
            col2_left_bottom = _place_column(buck2_left, col2_x, col2_top)
            # Compute buck2 right offset from actual max widths to avoid overlaps
            _b2l_max_w = max(
                (fp_sizes.get(r, (2.0, 2.0))[0] for r in buck2_left if r in fp_sizes),
                default=3.0,
            )
            _b2r_max_w = max(
                (fp_sizes.get(r, (2.0, 2.0))[0] for r in buck2_right if r in fp_sizes),
                default=2.0,
            )
            col2_right_x = col2_x + (_b2l_max_w + _b2r_max_w) / 2.0 + 0.5
            _place_column(buck2_right, col2_right_x, col2_top)
            col2_bottom = col2_left_bottom

            # Tail: LED + ferrites + misc — below whichever column is longer
            tail_y = max(fork_y, col2_bottom)
            tail_left = tail_column[:len(tail_column) // 2 + 1]
            tail_right = tail_column[len(tail_column) // 2 + 1:]
            _place_column(tail_left, left_x, tail_y)
            _place_column(tail_right, right_x, tail_y)

            _log.info(
                "    3c1: organized %d power components anchored at %s (%.1f, %.1f)",
                len(power_group_fixed), buck1_ic, u1_x, u1_y,
            )

    # 3c2. ADC channel formation — repeatable channel strips near ADC ICs
    # Detect nets connecting an ADC IC pin to exactly 2R + 1D + 1C (voltage
    # divider + protection pattern), then arrange each channel's 4 passives
    # in a consistent horizontal strip stacked vertically by channel index.
    _log.info("  3c2: ADC channel formation")
    adc_channel_refs: set[str] = set()  # protect during collision resolution

    # Build net → refs mapping from requirements
    net_components: dict[str, list[tuple[str, str]]] = {}
    for net in requirements.nets:
        net_components[net.name] = [(c.ref, c.pin) for c in net.connections]

    # Find ADC channel nets: connect U* pin to 2R + 1D + 1C
    adc_channels: list[tuple[str, str, list[str]]] = []  # (ic_ref, ic_pin, [passive_refs])
    for net_name, conns in net_components.items():
        ic_refs = [(r, p) for r, p in conns if r.startswith("U") and r in positions]
        passive_refs = [r for r, p in conns
                        if r in positions and r[0] in "RDC" and not r.startswith("U")]
        if len(ic_refs) == 1 and len(passive_refs) == 4:
            # Check pattern: 2R + 1D + 1C
            r_count = sum(1 for r in passive_refs if r.startswith("R"))
            d_count = sum(1 for r in passive_refs if r.startswith("D"))
            c_count = sum(1 for r in passive_refs if r.startswith("C"))
            if r_count == 2 and d_count == 1 and c_count == 1:
                ic_ref, ic_pin = ic_refs[0]
                adc_channels.append((ic_ref, ic_pin, passive_refs))

    # Group channels by IC and sort by input connector X position
    # (left-to-right matching connector order for direct routing)
    ic_channels: dict[str, list[tuple[str, list[str]]]] = {}
    for ic_ref, ic_pin, passives in adc_channels:
        ic_channels.setdefault(ic_ref, []).append((ic_pin, passives))

    # Build a mapping: R_top ref → connector X position
    # R_top is the first resistor in each channel (connects to input source)
    _r_top_connector_x: dict[str, float] = {}
    for net in requirements.nets:
        j_refs = [c for c in net.connections if c.ref.startswith("J")]
        r_refs = [c for c in net.connections
                  if c.ref.startswith("R") and c.ref in positions]
        if j_refs and r_refs:
            for j_conn in j_refs:
                j_pos = positions.get(j_conn.ref)
                if j_pos:
                    for r_conn in r_refs:
                        _r_top_connector_x[r_conn.ref] = j_pos[0]

    def _channel_sort_key(ch: tuple[str, list[str]]) -> float:
        """Sort channels by input connector X position (left-to-right)."""
        _pin, passives = ch
        r_refs_ch = sorted(r for r in passives if r.startswith("R"))
        for r in r_refs_ch:
            if r in _r_top_connector_x:
                return _r_top_connector_x[r]
        # Fallback: sort by pin number
        return float(hash(_pin)) * 1e-6

    for ic_ref in ic_channels:
        ic_channels[ic_ref].sort(key=_channel_sort_key)

    # First pass: collect all ADC channel passive refs and IC refs
    all_adc_passive_refs: set[str] = set()
    adc_ic_refs: set[str] = set()
    for _ic_ref, _ic_pin, passives in adc_channels:
        all_adc_passive_refs.update(passives)
        adc_ic_refs.add(_ic_ref)

    # Move ADC ICs to the analog zone before placing channels.
    # The group placer may have put them in a poor position (e.g., in the
    # power zone area).  Place them at the bottom of the analog zone so
    # channels extend upward toward the connectors on the top edge.
    analog_zone = None
    for z in zones:
        if z.name == "analog":
            analog_zone = z
            break
    def _ic_avg_connector_x(ic: str) -> float:
        """Average X of connectors feeding this IC's channels."""
        xs: list[float] = []
        for _pin, passives in ic_channels.get(ic, []):
            for r in passives:
                if r.startswith("R") and r in _r_top_connector_x:
                    xs.append(_r_top_connector_x[r])
        return sum(xs) / len(xs) if xs else 999.0

    if analog_zone and adc_ic_refs:
        az_x1, az_y1, az_x2, az_y2 = analog_zone.rect
        # Place ICs in the lower third of the analog zone so channels
        # extend upward (toward connectors at top edge) with room to spread.
        # Sort ICs by average connector X of their channels so that ICs
        # feeding leftmost connectors are placed leftmost.
        ic_list = sorted(adc_ic_refs, key=_ic_avg_connector_x)
        n_ics = len(ic_list)
        ic_spacing = (az_x2 - az_x1) / (n_ics + 1)
        for idx, ic_ref in enumerate(ic_list):
            if ic_ref not in positions:
                continue
            _old_x, _old_y, _old_rot = positions[ic_ref]
            iw, ih = fp_sizes.get(ic_ref, (5.0, 5.0))
            new_x = az_x1 + ic_spacing * (idx + 1)
            # Place at 70% height of analog zone (lower third), leaving
            # space above for channel strips and below for decoupling caps
            new_y = az_y1 + (az_y2 - az_y1) * 0.70
            positions[ic_ref] = (new_x, new_y, 0.0)
            # Protect ADC ICs from collision resolution — they're deliberately
            # placed in the analog zone.
            fixed_refs.add(ic_ref)
            _log.info(
                "    3c2: moved %s to analog zone (%.1f, %.1f)",
                ic_ref, new_x, new_y,
            )

    # Build occupancy grid WITHOUT ADC channel passives so they can be freely placed
    adc_grid = _PlacementGrid(bounds)
    for oref, (ox, oy, _orot) in positions.items():
        if oref in all_adc_passive_refs:
            continue  # Don't block with current (scattered) positions
        ow, oh = _rotation_aware_size(oref, positions, fp_sizes)
        adc_grid.place(ox, oy, ow, oh)

    # Per-IC vertical channel columns: each channel is a vertical strip
    # running UPWARD from the ADC IC toward the connectors / 24V source
    # (top edge of board).  Channels are spread horizontally side-by-side,
    # ordered left-to-right by the X position of each channel's source
    # connector (so traces run straight down without crossing).
    #
    # Signal flow (top to bottom): connector → R_top → C_filter → D_clamp → R_bot → IC
    # Physical layout (upward from IC):  IC ← R_bot ← D_clamp ← C_filter ← R_top
    #
    # All components at 0° rotation (horizontal pads) so pads align
    # vertically in the signal path.
    _CHANNEL_SPACING_MM = 8.0  # horizontal gap between channel columns — wide enough to be visually distinct
    _STRIP_GAP_MM = 1.5  # vertical gap between strip components

    # Sort ICs by average connector X (leftmost connector = leftmost IC)
    sorted_ic_refs = sorted(ic_channels.keys(), key=_ic_avg_connector_x)

    # Track occupied X ranges to prevent inter-IC channel overlap.
    # Each entry: (x_min, x_max) of the channel columns for an IC.
    _occupied_x_ranges: list[tuple[float, float]] = []

    for ic_ref in sorted_ic_refs:
        ch_list = ic_channels[ic_ref]
        if ic_ref not in positions:
            continue
        ix, iy, _irot = positions[ic_ref]
        iw, ih = fp_sizes.get(ic_ref, (5.0, 5.0))

        n_ch = len(ch_list)
        total_ch_width = (n_ch - 1) * _CHANNEL_SPACING_MM
        # Center the channel columns horizontally on the IC
        ch_x_start = ix - total_ch_width / 2.0
        ch_x_end = ch_x_start + total_ch_width

        # Shift right if this IC's channel band overlaps with previously placed ICs
        _comp_half_w = 2.0  # half-width of widest component (SOD-323 = 3.7mm)
        for ox_min, ox_max in _occupied_x_ranges:
            if (ch_x_start - _comp_half_w < ox_max + _comp_half_w
                    and ch_x_end + _comp_half_w > ox_min - _comp_half_w):
                # Overlap — shift this IC's channels to the right of the occupied range
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

            # Place upward from IC: R_bot closest to IC, R_top farthest
            strip_order: list[str] = []
            if len(r_refs) >= 2:
                strip_order.append(r_refs[1])  # R_bot (closest to IC)
            strip_order.extend(d_refs)          # D_clamp
            strip_order.extend(c_refs)          # C_filter
            if len(r_refs) >= 1:
                strip_order.append(r_refs[0])  # R_top (closest to connector)

            # Start placing above the IC
            strip_y = iy - ih / 2.0 - 1.0

            for ref in strip_order:
                if ref not in positions or ref in fixed_refs:
                    continue
                raw_w, raw_h = fp_sizes.get(ref, (2.0, 2.0))
                # All components at 0° for vertical signal flow
                w, h = raw_w, raw_h
                rot = 0.0
                target_x = ch_x
                target_y = strip_y - h / 2.0
                target_x = max(bounds[0] + 2.0, min(bounds[2] - 2.0, target_x))
                target_y = max(bounds[1] + 2.0, min(bounds[3] - 2.0, target_y))

                # Deterministic placement — exact position, no grid search.
                # This ensures repeatable strip patterns across all channels.
                positions[ref] = (target_x, target_y, rot)
                adc_channel_refs.add(ref)
                strip_y = target_y - (h / 2.0 + _STRIP_GAP_MM)

    _log.info(
        "    3c2: %d channels across %d ICs",
        len(adc_channels), len(adc_ic_refs),
    )

    # 3c3. Analog subcircuit clustering — pull remaining analog passives
    # (ladder switch resistors, optocoupler components, ADC decoupling)
    # into vertical columns next to the ADC channel columns.
    #
    # Strategy: find SMALL PASSIVE components (R, C, D, LED, SW) on signal
    # nets that connect to ADC IC pins.  Then follow one hop through
    # non-power nets but ONLY through small passives/discretes — never
    # through ICs (which fan out to the entire board) or connectors.
    _log.info("  3c3: Analog subcircuit clustering")

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

    # Find small passives on ADC signal nets (excluding power/bus/I2C)
    analog_signal_refs: set[str] = set()
    for net in requirements.nets:
        if _is_power_or_bus_net(net.name):
            continue
        # Must connect to an ADC IC pin (not just share a power net)
        ic_conn = [c for c in net.connections
                   if c.ref in adc_ic_refs and c.ref in positions]
        if not ic_conn:
            continue
        for c in net.connections:
            if (c.ref in positions
                    and c.ref not in adc_channel_refs
                    and c.ref not in adc_ic_refs
                    and c.ref not in fixed_refs
                    and c.ref not in relay_support_refs
                    and c.ref not in power_group_fixed
                    and _is_small_passive(c.ref)):
                analog_signal_refs.add(c.ref)

    # Refs already claimed by other groups — exclude from analog cluster
    _other_group_refs = relay_support_refs | power_group_fixed

    # One hop: follow non-power nets through small passives only
    # Also allow U refs with ≤6 pins (optocouplers, small ICs) but NOT MCUs
    hop2_refs: set[str] = set()
    for ref in list(analog_signal_refs):
        for net in requirements.nets:
            if _is_power_or_bus_net(net.name):
                continue
            if not any(c.ref == ref for c in net.connections):
                continue
            for c in net.connections:
                if (c.ref != ref and c.ref in positions
                        and c.ref not in adc_channel_refs
                        and c.ref not in adc_ic_refs
                        and c.ref not in fixed_refs
                        and c.ref not in analog_signal_refs
                        and c.ref not in _other_group_refs):
                    # Allow small passives and small ICs (optocouplers)
                    if _is_small_passive(c.ref):
                        hop2_refs.add(c.ref)
                    elif c.ref.startswith("U"):
                        # Only pull small ICs (≤6 pins)
                        comp = next((comp for comp in requirements.components
                                     if comp.ref == c.ref), None)
                        if comp and len(comp.pins) <= 6:
                            hop2_refs.add(c.ref)

    # Hop 3: for small ICs found in hop2, follow their remaining non-power
    # nets to pull in the full subcircuit (e.g., U7 opto → R25, R32, D17, LED2)
    hop3_refs: set[str] = set()
    small_ics_found = {r for r in hop2_refs if r.startswith("U")}
    for ic_ref in small_ics_found:
        for net in requirements.nets:
            if _is_power_or_bus_net(net.name):
                continue
            if not any(c.ref == ic_ref for c in net.connections):
                continue
            for c in net.connections:
                if (c.ref != ic_ref and c.ref in positions
                        and c.ref not in adc_channel_refs
                        and c.ref not in adc_ic_refs
                        and c.ref not in fixed_refs
                        and c.ref not in _other_group_refs
                        and _is_small_passive(c.ref)):
                    hop3_refs.add(c.ref)
    # Hop 4: one more hop from hop3 refs (e.g., R25→OPTO_IN→D17/R32/SW3)
    hop4_refs: set[str] = set()
    for ref in hop3_refs:
        for net in requirements.nets:
            if _is_power_or_bus_net(net.name):
                continue
            if not any(c.ref == ref for c in net.connections):
                continue
            for c in net.connections:
                if (c.ref != ref and c.ref in positions
                        and c.ref not in adc_channel_refs
                        and c.ref not in adc_ic_refs
                        and c.ref not in fixed_refs
                        and c.ref not in _other_group_refs
                        and _is_small_passive(c.ref)):
                    hop4_refs.add(c.ref)

    all_analog_cluster_refs = (
        analog_signal_refs | hop2_refs | hop3_refs | hop4_refs
    ) - adc_channel_refs - _other_group_refs

    if all_analog_cluster_refs and _occupied_x_ranges:
        last_x_max = max(xmax for _, xmax in _occupied_x_ranges)
        cluster_x = last_x_max + _CHANNEL_SPACING_MM + 2.0

        adc_ys = [positions[r][1] for r in adc_channel_refs if r in positions]
        cluster_y_top = min(adc_ys) - 1.0 if adc_ys else bounds[1] + 5.0

        # Sort: ICs first (larger), then passives by ref
        sorted_cluster = sorted(
            all_analog_cluster_refs,
            key=lambda r: (0 if r.startswith("U") else 2, r),
        )

        # Place in a vertical column, wrapping to next column after 15mm
        _COL_GAP = 5.0
        _ROW_GAP = 1.0  # vertical gap
        cur_x = cluster_x
        cur_y = cluster_y_top
        placed_count = 0

        for ref in sorted_cluster:
            if ref not in positions:
                continue
            raw_w, raw_h = fp_sizes.get(ref, (2.0, 2.0))
            w, h = raw_w, raw_h
            rot = 0.0

            target_x = cur_x
            target_y = cur_y + h / 2.0
            target_x = max(bounds[0] + 2.0, min(bounds[2] - 2.0, target_x))
            target_y = max(bounds[1] + 2.0, min(bounds[3] - 2.0, target_y))

            positions[ref] = (target_x, target_y, rot)
            adc_channel_refs.add(ref)
            placed_count += 1

            cur_y = target_y + h / 2.0 + _ROW_GAP

            if cur_y > cluster_y_top + 15.0:
                cur_x += _COL_GAP
                cur_y = cluster_y_top

        _log.info("    3c3: clustered %d analog refs near ADC channels", placed_count)

    # 3c3b. Pull remaining analog group outliers toward the cluster.
    # Find all refs in the same FeatureBlock as the ADC ICs, then pull
    # any that are >15mm from the analog centroid into the cluster area.
    analog_group_refs: set[str] = set()
    for feat in requirements.features:
        feat_refs = set(feat.components)
        if feat_refs & adc_ic_refs:
            analog_group_refs = feat_refs
            break

    if analog_group_refs and adc_channel_refs:
        # Compute centroid of already-placed analog refs
        placed_analog = [
            positions[r] for r in (adc_channel_refs | adc_ic_refs)
            if r in positions
        ]
        if placed_analog:
            cx = sum(p[0] for p in placed_analog) / len(placed_analog)
            cy = sum(p[1] for p in placed_analog) / len(placed_analog)
            # Find outlier refs (>15mm from centroid, not already placed)
            outlier_refs: list[str] = []
            for ref in sorted(analog_group_refs):
                if (ref in positions
                        and ref not in adc_channel_refs
                        and ref not in adc_ic_refs
                        and ref not in fixed_refs
                        and not ref.startswith("J")):  # keep connectors on edges
                    rx, ry, _rrot = positions[ref]
                    dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
                    if dist > 15.0:
                        outlier_refs.append(ref)

            if outlier_refs:
                # Compute cluster placement area — to the right of ADC channels
                adc_xs = [positions[r][0] for r in adc_channel_refs
                          if r in positions]
                adc_ys = [positions[r][1] for r in adc_channel_refs
                          if r in positions]
                outlier_x = max(adc_xs) + _CHANNEL_SPACING_MM + 2.0
                outlier_y_top = min(adc_ys) - 1.0
                oc_x = outlier_x
                oc_y = outlier_y_top
                _OC_COL_GAP = 5.0
                _OC_ROW_GAP = 1.0  # vertical gap

                for ref in outlier_refs:
                    raw_w, raw_h = fp_sizes.get(ref, (2.0, 2.0))
                    w, h = raw_w, raw_h
                    rot = 0.0

                    target_x = oc_x
                    target_y = oc_y + h / 2.0
                    target_x = max(bounds[0] + 2.0, min(bounds[2] - 2.0, target_x))
                    target_y = max(bounds[1] + 2.0, min(bounds[3] - 2.0, target_y))

                    positions[ref] = (target_x, target_y, rot)
                    adc_channel_refs.add(ref)

                    oc_y = target_y + h / 2.0 + _OC_ROW_GAP
                    if oc_y > outlier_y_top + 15.0:
                        oc_x += _OC_COL_GAP
                        oc_y = outlier_y_top

                _log.info(
                    "    3c3b: pulled %d outliers into analog cluster",
                    len(outlier_refs),
                )

    # 3d. Crystal-IC proximity — within 10mm of connected IC
    # Traces nets from crystal pins to find the IC it serves (MCU, W5500,
    # LAN8720A, etc.), not just the MCU.  Critical for Ethernet crystals.
    _log.info("  3d: Crystal-IC proximity")
    # Build crystal→IC map via net connectivity
    _crystal_ref_to_nets: dict[str, set[str]] = {}
    _net_to_components: dict[str, set[str]] = {}
    for net in requirements.nets:
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
        # Find connected IC via crystal's non-GND nets
        crystal_ref = sc.anchor_ref
        target_ic: str | None = None
        crystal_nets = _crystal_ref_to_nets.get(crystal_ref, set())
        for net_name in crystal_nets:
            _nl = net_name.upper()
            if _nl in ("GND", "AGND", "DGND", "PGND", "VSS", "AVSS"):
                continue
            for r in _net_to_components.get(net_name, set()):
                if r.startswith("U") and r in positions and r != crystal_ref:
                    target_ic = r
                    break
            if target_ic:
                break

        if not target_ic:
            # Fallback: use MCU
            from kicad_pipeline.optimization.functional_grouper import (
                _find_mcu_ref,
            )
            target_ic = _find_mcu_ref(requirements)
        if not target_ic or target_ic not in positions:
            continue

        ic_x, ic_y, _ic_rot = positions[target_ic]
        ic_w, ic_h = fp_sizes.get(target_ic, (5.0, 5.0))
        _log.info("    Crystal %s → IC %s (%.1f, %.1f)",
                  crystal_ref, target_ic, ic_x, ic_y)

        for ref in sc.refs:
            if ref in fixed_refs or ref not in positions:
                continue
            rx, ry, rrot = positions[ref]
            dist = math.sqrt((rx - ic_x) ** 2 + (ry - ic_y) ** 2)
            if dist <= 10.0:
                continue
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            gap = 1.0
            candidates = [
                (ic_x + (ic_w + w) / 2.0 + gap, ic_y),
                (ic_x - (ic_w + w) / 2.0 - gap, ic_y),
                (ic_x, ic_y + (ic_h + h) / 2.0 + gap),
                (ic_x, ic_y - (ic_h + h) / 2.0 - gap),
            ]
            pull_grid = _PlacementGrid(bounds)
            for oref, (ox, oy, _or) in positions.items():
                if oref != ref:
                    ow, oh = _rotation_aware_size(oref, positions, fp_sizes)
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
                positions[ref] = (best_pos[0], best_pos[1], rrot)
                _log.info("    %s pulled to (%.1f, %.1f) dist=%.1f→%.1f from %s",
                          ref, best_pos[0], best_pos[1], dist, best_dist,
                          target_ic)

    # 3e. RF edge pinning — pin RF modules to board edge
    _log.info("  3e: RF edge pinning")
    positions = _pin_rf_to_edge(sc_list, positions, fp_sizes, bounds, fixed_refs)

    # 3f. Connector orientation — face outward from board edge
    _log.info("  3f: Connector orientation")
    positions = _orient_connectors(
        positions, fp_sizes, bounds, fixed_refs, initial_pcb,
    )

    # 3f2. Top-edge screw terminal ordering
    # Place screw terminals along top edge in functional order:
    # left→right: J6(spare ADC), J5(opto), J4(ladder), J3(aux power), J1(power harness)
    # At 0° rotation, screw terminal pads are horizontal (along X axis).
    # Place them in a row along the top edge with gap for screwdriver access.
    _TOP_EDGE_ORDER = ["J6", "J5", "J4", "J3", "J1"]
    _top_refs = [r for r in _TOP_EDGE_ORDER if r in positions and r not in fixed_refs]
    if _top_refs:
        _log.info("  3f2: Top-edge screw terminal ordering (%s)", _top_refs)
        term_gap = 3.0  # mm gap between courtyard edges
        # At 0° rotation, native width IS the X extent
        term_widths: list[float] = []
        for r in _top_refs:
            w, _h = fp_sizes.get(r, (2.0, 2.0))
            term_widths.append(w)
        total_w = sum(term_widths) + term_gap * (len(_top_refs) - 1)
        # Use full board width minus margins for mounting holes
        margin = 8.0  # mm from board edge (clear of mounting holes)
        avail_w = (max_x - min_x) - 2 * margin
        if total_w < avail_w:
            start_x = min_x + margin + (avail_w - total_w) / 2.0
        else:
            # Compress gap to fit
            compressed_gap = max(1.0, (avail_w - sum(term_widths)) / max(len(_top_refs) - 1, 1))
            term_gap = compressed_gap
            total_w = sum(term_widths) + term_gap * (len(_top_refs) - 1)
            start_x = min_x + margin
        cursor_x = start_x
        # Use origin_to_centroid() to correctly place pin 1 (origin) near
        # the top board edge, then let the library compute centroid position.
        origin_y_target = min_y + 3.0  # pin 1 pad 3mm from top edge
        for i, r in enumerate(_top_refs):
            tw = term_widths[i]
            origin_x = cursor_x + tw / 2.0
            # Find footprint to use origin_to_centroid
            fp_match = None
            for fp in initial_pcb.footprints:
                if fp.ref == r:
                    fp_match = fp
                    break
            if fp_match is not None:
                cent_x, cent_y = origin_to_centroid(
                    fp_match, origin_x, origin_y_target, 0.0,
                )
            else:
                cent_x, cent_y = origin_x, origin_y_target
            positions[r] = (cent_x, cent_y, 0.0)
            cursor_x += tw + term_gap
            _log.info("    %s → centroid(%.1f, %.1f) origin(%.1f, %.1f) rot=0",
                      r, cent_x, cent_y, origin_x, origin_y_target)
    top_edge_connector_refs: set[str] = set(_top_refs)

    # 3c3. MCU peripheral tightening — AFTER RF pinning so U3 is at final position
    # Order: U3 → decoupling caps → connectors → USB subcircuit →
    #        reset/boot → remaining passives
    _log.info("  3c3: MCU peripheral tightening")
    mcu_peripheral_refs: set[str] = set()
    from kicad_pipeline.optimization.functional_grouper import _find_mcu_ref as _find_mcu
    mcu_ref_c3 = _find_mcu(requirements)
    if mcu_ref_c3 and mcu_ref_c3 in positions:
        mcu_x, mcu_y, _mcu_rot = positions[mcu_ref_c3]
        mcu_w, mcu_h = fp_sizes.get(mcu_ref_c3, (5.0, 5.0))

        # Find MCU's FeatureBlock group
        mcu_group_refs: set[str] = set()
        for feat in requirements.features:
            feat_refs = set()
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                feat_refs.add(r)
            if mcu_ref_c3 in feat_refs:
                mcu_group_refs = feat_refs
                break

        # --- Step 0: Place U3 with antenna on BOTTOM board edge ---
        # WiFi antenna points down (180° rotation).
        mcu_fp = None
        for fp in initial_pcb.footprints:
            if fp.ref == mcu_ref_c3:
                mcu_fp = fp
                break
        mcu_zone_rect: tuple[float, float, float, float] | None = None
        for z in zones:
            if z.name == "mcu":
                mcu_zone_rect = z.rect
                break
        _mcu_rot = 180.0  # antenna pointing down (toward bottom edge)
        eff_w, eff_h = mcu_w, mcu_h  # At 180°, dimensions don't swap

        if mcu_fp is not None:
            from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space as _pad_ext
            _trial_ox = (bounds[0] + bounds[2]) / 2.0
            _trial_oy = (bounds[1] + bounds[3]) / 2.0
            _te = _pad_ext(mcu_fp, _trial_ox, _trial_oy, _mcu_rot)
            # How far down does the pad extend from origin?
            _pad_bot = _te[3] - _trial_oy
            # Place origin so bottom pads are 2mm from bottom edge
            mcu_origin_y = bounds[3] - _pad_bot - 2.0
            # Center X in MCU zone (right side of board)
            if mcu_zone_rect is not None:
                _zx1, _zy1, _zx2, _zy2 = mcu_zone_rect
                # Leave room for J14 on right edge
                _j14_w = fp_sizes.get("J14", (2.7, 35.7))[0] if "J14" in mcu_group_refs else 0.0
                mcu_origin_x = (_zx1 + _zx2 - _j14_w) / 2.0
            else:
                mcu_origin_x = bounds[2] - eff_w / 2.0 - 10.0
            # Clamp X so pads stay within board
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
        # Clamp centroid so the bounding box stays within board with margin.
        # ESP32-S3-WROOM has antenna overhang beyond courtyard — use 5mm margin.
        mcu_y = min(mcu_y, bounds[3] - eff_h / 2.0 - 5.0)
        mcu_y = max(mcu_y, bounds[1] + eff_h / 2.0 + 2.0)
        mcu_x = min(mcu_x, bounds[2] - eff_w / 2.0 - 2.0)
        mcu_x = max(mcu_x, bounds[0] + eff_w / 2.0 + 2.0)
        positions[mcu_ref_c3] = (mcu_x, mcu_y, _mcu_rot)
        mcu_peripheral_refs.add(mcu_ref_c3)
        # Protect MCU from collision resolution — it's a large IC deliberately
        # placed in the MCU zone.  Without this, collision resolution can push
        # U3 to the opposite corner of the board.
        fixed_refs.add(mcu_ref_c3)
        _log.info("    U3 centroid at (%.1f, %.1f) rot=180° [eff_w=%.1f, eff_h=%.1f]",
                  mcu_x, mcu_y, eff_w, eff_h)

        # MCU bounding box edges (centroid-based)
        mcu_left = mcu_x - eff_w / 2.0
        mcu_top = mcu_y - eff_h / 2.0
        mcu_bot = mcu_y + eff_h / 2.0

        # --- Step 1: Build occupancy grid WITHOUT MCU group ---
        mcu_grid = _PlacementGrid(bounds)
        for oref, (ox, oy, _or) in positions.items():
            if oref in mcu_group_refs:
                continue
            ow, oh = _rotation_aware_size(oref, positions, fp_sizes)
            mcu_grid.place(ox, oy, ow, oh)
        mcu_grid.place(mcu_x, mcu_y, eff_w, eff_h)

        # --- Step 2: Build net adjacency for MCU group ---
        ref_nets: dict[str, set[str]] = {}
        for net in requirements.nets:
            net_refs = set()
            for conn in net.connections:
                net_refs.add(conn.ref)
            for r in net_refs:
                if r in mcu_group_refs:
                    ref_nets.setdefault(r, set()).update(
                        net_refs & mcu_group_refs,
                    )

        # Classify MCU passives
        connector_refs = {r for r in mcu_group_refs if r.startswith("J")}
        decoupling_refs: list[str] = []
        other_passive_refs: list[str] = []
        for ref in sorted(mcu_group_refs):
            if ref == mcu_ref_c3 or ref in connector_refs or ref in fixed_refs:
                continue
            if ref not in positions:
                continue
            if ref.startswith("C") and mcu_ref_c3 in ref_nets.get(ref, set()):
                decoupling_refs.append(ref)
            else:
                other_passive_refs.append(ref)

        # --- Step 3: Place decoupling caps FIRST — tight against MCU ---
        # ESP32-S3-WROOM VCC/GND are on side castellation pads.
        # Place decoupling caps in a vertical column LEFT of MCU body,
        # centered vertically on the MCU. This keeps caps within 3mm of
        # the IC body edge (much closer than placing above the 25mm-tall module).
        mcu_right = mcu_x + eff_w / 2.0
        max_cap_w = max((fp_sizes.get(r, (2.5, 1.5))[0] for r in decoupling_refs),
                        default=2.5)
        # Column X: 2mm left of MCU body edge
        decoup_col_x = mcu_left - max_cap_w / 2.0 - 2.0
        # Start Y: centered on MCU, offset upward by half the total column height
        total_cap_h = sum(fp_sizes.get(r, (1.5, 1.0))[1] + 1.5
                          for r in decoupling_refs)
        decoup_y = mcu_y - total_cap_h / 2.0
        for ref in decoupling_refs:
            w, h = fp_sizes.get(ref, (1.0, 0.5))
            tx = decoup_col_x
            ty = decoup_y + h / 2.0
            tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
            ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
            px, py = mcu_grid.find_free_pos(tx, ty, w, h, max_radius=8.0)
            positions[ref] = (px, py, 0.0)
            mcu_peripheral_refs.add(ref)
            mcu_grid.place(px, py, w, h)
            decoup_y += h + 1.5
            dist_to_mcu = ((px - mcu_x) ** 2 + (py - mcu_y) ** 2) ** 0.5
            _log.info("    %s (decoupling): →(%.1f,%.1f) [%.1fmm from U3]",
                      ref, px, py, dist_to_mcu)

        # --- Step 4: Place connectors ---
        # MCU connectors go on the RIGHT board edge (accessible for cables).
        # J14 (1x14 GPIO header): right edge, vertically centered on U3
        # J15 (2x5 header): right edge, above J14
        # J16 (SD card): above U3
        # J2 (USB-C): bottom edge, below U3
        right_edge_x = bounds[2] - 2.0  # 2mm from right board edge

        if "J14" in connector_refs and "J14" in positions and "J14" not in fixed_refs:
            w14, h14 = fp_sizes.get("J14", (2.7, 35.7))
            # J14 on RIGHT board edge, vertically centered on U3.
            tx = bounds[2] - w14 / 2.0 - 1.0  # against right edge
            ty = mcu_y
            ty = max(bounds[1] + h14 / 2.0 + 1.0,
                     min(bounds[3] - h14 / 2.0 - 1.0, ty))
            px14, py14 = mcu_grid.find_free_pos(tx, ty, w14, h14,
                                                 max_radius=25.0)
            positions["J14"] = (px14, py14, 0.0)
            mcu_grid.place(px14, py14, w14, h14)
            mcu_peripheral_refs.add("J14")
            _log.info("    J14 → right edge (%.1f, %.1f)", px14, py14)

        if "J15" in connector_refs and "J15" in positions and "J15" not in fixed_refs:
            w15, h15 = fp_sizes.get("J15", (5.2, 12.9))
            j14_pos = positions.get("J14")
            if j14_pos:
                # Below J14 on right edge
                j14_bottom = j14_pos[1] + fp_sizes.get("J14", (2.7, 35.7))[1] / 2.0
                tx = right_edge_x - w15 / 2.0
                ty = j14_bottom + h15 / 2.0 + 2.0
            else:
                tx = right_edge_x - w15 / 2.0
                ty = mcu_y + 10.0
            # Clamp with 1.5mm margin from edges (review agent adds 1.0mm to
            # pad extents, so we need extra clearance)
            ty = max(bounds[1] + h15 / 2.0 + 1.5,
                     min(bounds[3] - h15 / 2.0 - 1.5, ty))
            tx = min(tx, bounds[2] - w15 / 2.0 - 1.5)
            px15, py15 = mcu_grid.find_free_pos(tx, ty, w15, h15, max_radius=25.0)
            # Ensure grid search didn't push past edge
            px15 = min(px15, bounds[2] - w15 / 2.0 - 1.5)
            py15 = min(py15, bounds[3] - h15 / 2.0 - 1.5)
            positions["J15"] = (px15, py15, 0.0)
            mcu_grid.place(px15, py15, w15, h15)
            mcu_peripheral_refs.add("J15")
            _log.info("    J15 → right edge, below J14 (%.1f, %.1f)", px15, py15)

        # J16 (SD card slot): RIGHT board edge, above U3
        # microSD card insertion needs the slot opening at the board edge.
        # Force-place at edge — no grid search (edge connectors take priority).
        if "J16" in connector_refs and "J16" in positions and "J16" not in fixed_refs:
            w16, h16 = fp_sizes.get("J16", (16.2, 6.9))
            # Right edge: connector body flush with board edge
            px = bounds[2] - w16 / 2.0
            py = mcu_top - h16 / 2.0 - 2.0  # above U3
            py = max(bounds[1] + h16 / 2.0 + 1.0,
                     min(bounds[3] - h16 / 2.0 - 1.0, py))
            positions["J16"] = (px, py, 0.0)
            mcu_grid.place(px, py, w16, h16)
            mcu_peripheral_refs.add("J16")
            _log.info("    J16 → right edge, above U3 (%.1f, %.1f)", px, py)

        # J2 (USB-C): bottom edge, LEFT of U3, facing outward (180°)
        # U3's courtyard extends to mcu_bot ≈ 72.7mm on an 80mm board, leaving
        # only ~7mm below — not enough for J2 (h ≈ 7.6mm).  Place LEFT of U3
        # on the bottom edge instead.
        if "J2" in connector_refs and "J2" in positions and "J2" not in fixed_refs:
            w2, h2 = fp_sizes.get("J2", (9.6, 7.6))
            # Target: bottom edge, left of MCU courtyard
            tx = mcu_left - w2 / 2.0 - 8.0  # left of U3 with 8mm gap for decoupling
            ty = bounds[3] - h2 / 2.0 - 1.0  # on bottom edge
            tx = max(bounds[0] + w2 / 2.0 + 1.0,
                     min(bounds[2] - w2 / 2.0 - 1.0, tx))
            px, py = mcu_grid.find_free_pos(tx, ty, w2, h2, max_radius=20.0)
            positions["J2"] = (px, py, 180.0)
            mcu_grid.place(px, py, w2, h2)
            mcu_peripheral_refs.add("J2")
            _log.info("    J2 → bottom edge, left of U3 (%.1f, %.1f)", px, py)

        # --- Step 5: USB subcircuit (U9 + R6 + R7) near J2 ---
        # Place U9 LEFT of J2 to avoid U3 courtyard (which extends far left)
        j2_pos = positions.get("J2")
        if "U9" in other_passive_refs and "U9" in positions and j2_pos:
            j2x, j2y, _j2r = j2_pos
            j2w, j2h = fp_sizes.get("J2", (9.6, 7.6))
            u9w, u9h = fp_sizes.get("U9", (3.0, 3.0))
            # Place U9 ABOVE J2 (away from U3 and board edge)
            # J2 at 180° has centroid offset ~1.2mm up from origin, so
            # we need extra clearance to avoid centroid-based overlap.
            u9_tx = j2x - j2w / 4.0  # slightly left to avoid center overlap
            u9_ty = j2y - j2h / 2.0 - u9h / 2.0 - 6.0
            u9_tx = max(bounds[0] + 2.0, min(bounds[2] - u9w / 2.0 - 1.0, u9_tx))
            u9_ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, u9_ty))
            u9x, u9y = mcu_grid.find_free_pos(u9_tx, u9_ty, u9w, u9h,
                                               max_radius=8.0)
            positions["U9"] = (u9x, u9y, 0.0)
            mcu_peripheral_refs.add("U9")
            mcu_grid.place(u9x, u9y, u9w, u9h)
            other_passive_refs.remove("U9")
            _log.info("    U9 (ESD) → near J2 at (%.1f, %.1f)",
                      u9x, u9y)

        # R6, R7 (USB resistors) near U9/J2 — place ABOVE J2 (between
        # J2 and U9) to avoid being pushed off the bottom board edge.
        usb_r_refs = [r for r in ("R6", "R7") if r in other_passive_refs
                      and r in positions]
        if usb_r_refs and j2_pos:
            j2x_r, j2y_r, _ = j2_pos
            j2w_r = fp_sizes.get("J2", (9.6, 7.6))[0]
            j2h_r = fp_sizes.get("J2", (9.6, 7.6))[1]
            for i, ref in enumerate(usb_r_refs):
                w, h = fp_sizes.get(ref, (1.0, 0.5))
                # Place above J2, stacked horizontally
                px = j2x_r - j2w_r / 4.0 + i * (w + 2.0)
                py = j2y_r - j2h_r / 2.0 - h / 2.0 - 1.5
                px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
                py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
                positions[ref] = (px, py, 0.0)
                # Don't add to mcu_peripheral_refs — R6/R7 are small
                # and can be moved by collision resolution if needed.
                mcu_grid.place(px, py, w, h)
                other_passive_refs.remove(ref)
                _log.info("    %s (USB R) → (%.1f, %.1f)", ref, px, py)

        # --- Step 6: Reset/Boot subcircuit (SW1+R4, SW2+R5) ---
        # Place as a compact cluster above-left of U3
        sw_pairs: list[tuple[str, str]] = []  # (switch, resistor)
        for sw, res in [("SW1", "R4"), ("SW2", "R5"), ("SW1", "R5"), ("SW2", "R4")]:
            if (sw in other_passive_refs and sw in positions
                    and res in other_passive_refs and res in positions):
                # Check if they share a net (connected)
                sw_nets = ref_nets.get(sw, set())
                if res in sw_nets:
                    sw_pairs.append((sw, res))
        # Deduplicate: each ref should appear in at most one pair
        used_sw: set[str] = set()
        unique_pairs: list[tuple[str, str]] = []
        for sw, res in sw_pairs:
            if sw not in used_sw and res not in used_sw:
                unique_pairs.append((sw, res))
                used_sw.add(sw)
                used_sw.add(res)
        # Also handle unpaired SW/R refs
        unpaired_sw = [r for r in other_passive_refs if r in positions
                       and r.startswith("SW") and r not in used_sw]

        # Place SW/R cluster above MCU, left side — close to MCU (within 5mm)
        sw_base_x = mcu_left + eff_w / 4.0
        sw_base_y = mcu_top - 4.0  # 4mm above MCU top edge
        for i, (sw, res) in enumerate(unique_pairs):
            sw_w, sw_h = fp_sizes.get(sw, (3.5, 3.5))
            r_w, r_h = fp_sizes.get(res, (1.0, 0.5))
            # Switch — place side by side horizontally
            tx = sw_base_x - i * (sw_w + 1.5)
            ty = sw_base_y
            tx = max(bounds[0] + sw_w / 2.0 + 2.0, min(bounds[2] - sw_w / 2.0 - 2.0, tx))
            ty = max(bounds[1] + sw_h / 2.0 + 2.0, min(bounds[3] - sw_h / 2.0 - 2.0, ty))
            px, py = mcu_grid.find_free_pos(tx, ty, sw_w, sw_h, max_radius=10.0)
            # Clamp: stay within 20mm of MCU (both X and Y) and 2mm inside board
            px = max(mcu_x - 20.0, min(mcu_x + 20.0, px))
            py = max(max(mcu_y - 20.0, bounds[1] + sw_h / 2.0 + 2.0),
                     min(bounds[3] - sw_h / 2.0 - 2.0, py))
            positions[sw] = (px, py, 0.0)
            mcu_peripheral_refs.add(sw)
            mcu_grid.place(px, py, sw_w, sw_h)
            other_passive_refs.remove(sw)
            # Resistor adjacent to switch
            r_tx = px
            r_ty = py - sw_h / 2.0 - r_h / 2.0 - 0.5
            r_tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, r_tx))
            r_ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, r_ty))
            rpx, rpy = mcu_grid.find_free_pos(r_tx, r_ty, r_w, r_h,
                                               max_radius=8.0)
            positions[res] = (rpx, rpy, 0.0)
            mcu_peripheral_refs.add(res)
            mcu_grid.place(rpx, rpy, r_w, r_h)
            other_passive_refs.remove(res)
            _log.info("    %s+%s (reset/boot) → (%.1f,%.1f) / (%.1f,%.1f)",
                      sw, res, px, py, rpx, rpy)

        # Unpaired switches
        for sw in unpaired_sw:
            sw_w, sw_h = fp_sizes.get(sw, (3.5, 3.5))
            tx = sw_base_x - len(unique_pairs) * (sw_w + 3.0)
            ty = sw_base_y
            tx = max(bounds[0] + sw_w / 2.0 + 1.0, min(bounds[2] - 2.0, tx))
            ty = max(bounds[1] + sw_h / 2.0 + 1.0, min(bounds[3] - 2.0, ty))
            px, py = mcu_grid.find_free_pos(tx, ty, sw_w, sw_h, max_radius=15.0)
            positions[sw] = (px, py, 0.0)
            mcu_peripheral_refs.add(sw)
            mcu_grid.place(px, py, sw_w, sw_h)
            other_passive_refs.remove(sw)
            _log.info("    %s (switch) → (%.1f, %.1f)", sw, px, py)

        # --- Step 6b: Place LED1 near SW cluster (UI grouping) ---
        led_placed = set()
        for led_ref in ["LED1"]:
            if (led_ref in other_passive_refs and led_ref in positions
                    and led_ref not in fixed_refs):
                lw, lh = fp_sizes.get(led_ref, (2.0, 1.0))
                # Place LED next to the SW cluster (right side)
                led_tx = sw_base_x + 8.0
                led_ty = sw_base_y
                led_tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, led_tx))
                led_ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, led_ty))
                lpx, lpy = mcu_grid.find_free_pos(
                    led_tx, led_ty, lw, lh, max_radius=10.0,
                )
                positions[led_ref] = (lpx, lpy, 0.0)
                mcu_peripheral_refs.add(led_ref)
                mcu_grid.place(lpx, lpy, lw, lh)
                if led_ref in other_passive_refs:
                    other_passive_refs.remove(led_ref)
                led_placed.add(led_ref)
                _log.info("    %s (status LED) → (%.1f, %.1f)", led_ref, lpx, lpy)

        # --- Step 7: Place remaining passives near MCU perimeter ---
        # Sort: MCU-connected refs first, then by distance
        def _mcu_prox_key(ref: str) -> tuple[int, float]:
            connected = mcu_ref_c3 in ref_nets.get(ref, set())
            rx, ry, _ = positions[ref]
            dist = math.sqrt((rx - mcu_x) ** 2 + (ry - mcu_y) ** 2)
            return (0 if connected else 1, dist)

        remaining = [r for r in other_passive_refs if r in positions]
        remaining.sort(key=_mcu_prox_key)

        _MCU_TARGET_GAP = 4.0
        # Place remaining passives in a ring around MCU
        # At 180° rotation, MCU is horizontal with antenna down.
        # Primary slots: above MCU, then left side, then right side.
        ring_slots: list[tuple[float, float]] = []
        _MCU_CLEAR = 3.0  # minimum clearance from courtyard edge

        # Above MCU — primary slot area (most board space above)
        if mcu_top > bounds[1] + 10.0:
            for dx_off in range(-4, 5):
                ring_slots.append((mcu_x + dx_off * 4.0,
                                   mcu_top - _MCU_CLEAR - 3.0))
            # Second row above for overflow
            for dx_off in range(-3, 4):
                ring_slots.append((mcu_x + dx_off * 4.0,
                                   mcu_top - _MCU_CLEAR - 7.0))
        # Left side of MCU
        slot_x_left = mcu_left - _MCU_CLEAR - 2.0
        for dy_off in range(-3, 4):
            ring_slots.append((slot_x_left, mcu_y + dy_off * 3.5))
        # Right side (between MCU and J14)
        slot_x_right = mcu_right + _MCU_CLEAR + 2.0
        for dy_off in range(-3, 4):
            ring_slots.append((slot_x_right, mcu_y + dy_off * 3.5))

        slot_idx = 0
        for ref in remaining:
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            rrot = positions[ref][2]

            # Try ring slots first, then fallback to perimeter projection
            placed = False
            while slot_idx < len(ring_slots):
                sx, sy = ring_slots[slot_idx]
                slot_idx += 1
                sx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, sx))
                sy = max(bounds[1] + 2.0, min(bounds[3] - 2.0, sy))
                if mcu_grid.is_free(sx, sy, w, h):
                    positions[ref] = (sx, sy, rrot)
                    mcu_peripheral_refs.add(ref)
                    mcu_grid.place(sx, sy, w, h)
                    placed = True
                    break

            if not placed:
                # Fallback: find_free_pos near MCU left side
                tx = mcu_left - _MCU_TARGET_GAP - w / 2.0
                ty = mcu_y
                tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
                ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
                px, py = mcu_grid.find_free_pos(tx, ty, w, h,
                                                max_radius=25.0)
                positions[ref] = (px, py, rrot)
                mcu_peripheral_refs.add(ref)
                mcu_grid.place(px, py, w, h)

        # --- Post-placement: Push any components still inside U3 courtyard ---
        _COURT_MARGIN = 2.0  # mm clearance from courtyard edge
        court_x1 = mcu_x - eff_w / 2.0 - _COURT_MARGIN
        court_y1 = mcu_y - eff_h / 2.0 - _COURT_MARGIN
        court_x2 = mcu_x + eff_w / 2.0 + _COURT_MARGIN
        court_y2 = mcu_y + eff_h / 2.0 + _COURT_MARGIN
        for ref in list(mcu_peripheral_refs):
            if ref == mcu_ref_c3:
                continue
            # Skip connectors to the right of U3 — they're intentionally
            # placed adjacent to the MCU for cable access
            if ref.startswith("J"):
                rx_j = positions[ref][0]
                if rx_j > mcu_x:
                    continue
            rx, ry, rrot = positions[ref]
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            # Check if centroid is inside expanded courtyard
            if court_x1 < rx < court_x2 and court_y1 < ry < court_y2:
                # Push out to nearest courtyard edge, ensuring the
                # component center is OUTSIDE the courtyard + its own
                # half-size so the scoring collision checker passes.
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
                # Clamp to board bounds
                new_x = max(bounds[0] + w / 2.0 + 1.0,
                            min(bounds[2] - w / 2.0 - 1.0, new_x))
                new_y = max(bounds[1] + h / 2.0 + 1.0,
                            min(bounds[3] - h / 2.0 - 1.0, new_y))
                px, py = mcu_grid.find_free_pos(new_x, new_y, w, h,
                                                 max_radius=15.0)
                positions[ref] = (px, py, rrot)
                mcu_grid.place(px, py, w, h)
                _log.info("    %s pushed outside U3 courtyard: "
                          "(%.1f,%.1f)→(%.1f,%.1f)", ref, rx, ry, px, py)

        _log.info(
            "    3c3: organized %d peripherals around %s at (%.1f, %.1f)",
            len(mcu_peripheral_refs), mcu_ref_c3, mcu_x, mcu_y,
        )

    # 3c4. Ethernet group organization — vertical signal-chain column
    # Signal chain: MCU SPI → U6 (W5500) → Y1 (crystal) + load caps
    #               → J13 (RJ45 Magjack) on bottom edge
    #               U8 (PoE) beside J13 with its caps
    _log.info("  3c4: Ethernet group organization")
    ethernet_fixed: set[str] = set()

    eth_group_refs: set[str] = set()
    for feat in requirements.features:
        if "ethernet" in feat.name.lower():
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                eth_group_refs.add(r)
            break


    if eth_group_refs:
        eth_zone_rect: tuple[float, float, float, float] | None = None
        for z in zones:
            if z.name == "ethernet":
                eth_zone_rect = z.rect
                break

        eth_ics = sorted([r for r in eth_group_refs
                          if r.startswith("U") and r in positions])
        eth_connectors = sorted([r for r in eth_group_refs
                                 if r.startswith("J") and r in positions])

        if eth_ics and eth_zone_rect is not None:
            ezx1, ezy1, ezx2, ezy2 = eth_zone_rect
            _ETH_STRIP_GAP = 1.0  # vertical gap

            # Build net → eth refs mapping
            eth_net_refs: dict[str, set[str]] = {}
            for net in requirements.nets:
                e_refs = set()
                for conn in net.connections:
                    if conn.ref in eth_group_refs:
                        e_refs.add(conn.ref)
                if e_refs:
                    eth_net_refs[net.name] = e_refs

            # Build occupancy grid WITHOUT ethernet group
            eth_grid = _PlacementGrid(bounds)
            for oref, (ox, oy, _orot) in positions.items():
                if oref in eth_group_refs:
                    continue
                ow, oh = _rotation_aware_size(oref, positions, fp_sizes)
                eth_grid.place(ox, oy, ow, oh)

            # Identify the main Ethernet IC (W5500 = largest U in group)
            eth_main_ic = eth_ics[0]  # U6 (W5500)
            # Crystal (Y prefix)
            crystal_refs = sorted([r for r in eth_group_refs
                                   if r.startswith("Y") and r in positions])
            # PoE module (second U if exists)
            poe_ic = eth_ics[1] if len(eth_ics) > 1 else ""

            # Decoupling caps for W5500 — all caps (including bulk decoupling)
            eth_decoupling = sorted([
                r for r in eth_group_refs
                if r.startswith("C") and r in positions
                and r not in crystal_refs
            ])

            # Crystal load caps (typically 18pF, on crystal nets)
            crystal_net_names: set[str] = set()
            for net_name, erefs in eth_net_refs.items():
                if any(r.startswith("Y") for r in erefs):
                    crystal_net_names.add(net_name)
            crystal_load_caps = sorted([
                r for r in eth_group_refs
                if r.startswith("C") and r in positions
                and any(r in eth_net_refs.get(n, set())
                        for n in crystal_net_names)
            ])

            # PoE caps (connected to PoE IC)
            poe_net_names: set[str] = set()
            for net_name, erefs in eth_net_refs.items():
                if poe_ic and poe_ic in erefs:
                    poe_net_names.add(net_name)
            poe_caps = sorted([
                r for r in eth_group_refs
                if r.startswith("C") and r in positions
                and r not in crystal_load_caps
                and any(r in eth_net_refs.get(n, set())
                        for n in poe_net_names)
            ])

            # Remaining caps (not crystal load, not PoE)
            other_caps = sorted(
                [r for r in eth_decoupling
                 if r not in crystal_load_caps and r not in poe_caps],
            )

            # Layout: vertical column flowing top→bottom
            # Column 1: U6 → Y1 + load caps → decoupling caps
            # Column 2 (right): U8 (PoE) + PoE caps
            # Bottom: J13 (RJ45) on board edge

            # Anchor: center of ethernet zone — force-place the main IC here
            eth_anchor_x = (ezx1 + ezx2) / 2.0
            eth_anchor_y = ezy1 + 3.0

            placed_eth: set[str] = set()

            # Force-place U6 (eth_main_ic) at the zone anchor position.
            # Don't use find_free_pos — the IC MUST be in its zone.
            ic_w, ic_h = fp_sizes.get(eth_main_ic, (10.0, 10.0))
            ic_cx = eth_anchor_x
            ic_cy = eth_anchor_y + ic_h / 2.0
            ic_cx = max(ezx1 + ic_w / 2.0 + 1.0,
                        min(ezx2 - ic_w / 2.0 - 1.0, ic_cx))
            ic_cy = max(ezy1 + ic_h / 2.0 + 1.0,
                        min(ezy2 - ic_h / 2.0 - 5.0, ic_cy))
            if eth_main_ic in positions and eth_main_ic not in fixed_refs:
                positions[eth_main_ic] = (ic_cx, ic_cy, 0.0)
                eth_grid.place(ic_cx, ic_cy, ic_w, ic_h)
                ethernet_fixed.add(eth_main_ic)
                # Protect ethernet IC from ALL collision resolution passes
                # (not just those that union ethernet_fixed)
                fixed_refs.add(eth_main_ic)
                placed_eth.add(eth_main_ic)
                _log.info("    %s (W5500) force-placed at (%.1f, %.1f) "
                          "in ethernet zone", eth_main_ic, ic_cx, ic_cy)

            # Pre-register J13 (RJ45) position in grid so column placement
            # avoids the area where J13 will be placed on the bottom edge.
            for _j13_ref in eth_connectors:
                _j13_w, _j13_h = fp_sizes.get(_j13_ref, (19.6, 12.5))
                _j13_cx = eth_anchor_x
                _j13_cy = bounds[3] - _j13_h / 2.0 - 1.0
                eth_grid.place(_j13_cx, _j13_cy, _j13_w, _j13_h)

            def _place_eth_column(
                refs: list[str], col_x: float, start_y: float,
            ) -> float:
                cy = start_y
                for ref in refs:
                    if (ref not in positions or ref in fixed_refs
                            or ref in placed_eth or ref == ""):
                        continue
                    w, h = fp_sizes.get(ref, (2.0, 2.0))
                    tx = col_x
                    ty = cy + h / 2.0
                    tx = max(ezx1 + 2.0, min(ezx2 - 2.0, tx))
                    ty = max(ezy1 + 2.0, min(ezy2 - 2.0, ty))
                    px, py = eth_grid.find_free_pos(tx, ty, w, h,
                                                    max_radius=6.0)
                    positions[ref] = (px, py, 0.0)
                    eth_grid.place(px, py, w, h)
                    ethernet_fixed.add(ref)
                    placed_eth.add(ref)
                    cy = py + h / 2.0 + _ETH_STRIP_GAP
                return cy

            # Crystal + load caps: force-place directly above U6 (≤5mm)
            # Place crystal edge-to-edge with IC, above it.
            y_crystal_h = fp_sizes.get(
                crystal_refs[0], (3.2, 1.5))[1] if crystal_refs else 1.5
            # Target: crystal just above IC courtyard with 0.5mm clearance
            crystal_y = ic_cy - ic_h / 2.0 - y_crystal_h / 2.0 - 0.5
            crystal_x = ic_cx  # centered on IC

            for ref in crystal_refs:
                if (ref not in positions or ref in fixed_refs
                        or ref in placed_eth or ref == ""):
                    continue
                w, h = fp_sizes.get(ref, (3.2, 1.5))
                tx = crystal_x
                ty = crystal_y  # crystal_y is already the centroid
                # Crystal MUST be close to IC — relax zone clamping
                # Only clamp to board bounds, not zone bounds
                tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
                ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
                # Force-place — don't use find_free_pos for crystal
                positions[ref] = (tx, ty, 0.0)
                eth_grid.place(tx, ty, w, h)
                ethernet_fixed.add(ref)
                placed_eth.add(ref)
                crystal_dist = math.sqrt(
                    (tx - ic_cx) ** 2 + (ty - ic_cy) ** 2)
                _log.info("    %s (crystal) → (%.1f, %.1f) "
                          "dist=%.1fmm from %s",
                          ref, tx, ty, crystal_dist, eth_main_ic)
                # Load caps go on either side of crystal
                cap_y = ty

            # Crystal load caps flanking the crystal (left and right)
            cap_idx = 0
            for ref in crystal_load_caps:
                if (ref not in positions or ref in fixed_refs
                        or ref in placed_eth or ref == ""):
                    continue
                w, h = fp_sizes.get(ref, (1.0, 0.5))
                # Alternate left/right of crystal
                y_w = fp_sizes.get(crystal_refs[0], (3.2, 1.5))[0] if crystal_refs else 3.2
                if cap_idx % 2 == 0:
                    tx = crystal_x - y_w / 2.0 - w / 2.0 - 0.5
                else:
                    tx = crystal_x + y_w / 2.0 + w / 2.0 + 0.5
                ty = crystal_y if crystal_refs else ic_cy - ic_h / 2.0 - 3.0
                tx = max(ezx1 + 2.0, min(ezx2 - 2.0, tx))
                ty = max(ezy1 + 2.0, min(ezy2 - 2.0, ty))
                px, py = eth_grid.find_free_pos(tx, ty, w, h,
                                                max_radius=5.0)
                positions[ref] = (px, py, 0.0)
                eth_grid.place(px, py, w, h)
                ethernet_fixed.add(ref)
                placed_eth.add(ref)
                cap_idx += 1

            # Remaining caps (decoupling) — place adjacent to U6.
            # Use grid-aware placement to avoid overlapping crystal refs.
            col_y = ic_cy + ic_h / 2.0 + 1.5
            for ref in other_caps:
                if (ref not in positions or ref in fixed_refs
                        or ref in placed_eth or ref == ""):
                    continue
                cw, ch = fp_sizes.get(ref, (1.5, 1.0))
                tx = ic_cx
                ty = col_y + ch / 2.0
                tx = max(ezx1 + 2.0, min(ezx2 - 2.0, tx))
                ty = max(ezy1 + 2.0, min(ezy2 - 2.0, ty))
                # Use grid to find collision-free position near target
                px, py = eth_grid.find_free_pos(tx, ty, cw, ch,
                                                max_radius=8.0)
                positions[ref] = (px, py, 0.0)
                eth_grid.place(px, py, cw, ch)
                ethernet_fixed.add(ref)
                placed_eth.add(ref)
                col_y = py + ch / 2.0 + 1.0
            col1_bottom = col_y

            # Column 2: PoE/PHY module — rotated 90° CCW, above J13 center
            if poe_ic and poe_ic in positions and poe_ic not in fixed_refs:
                poe_w, poe_h = fp_sizes.get(poe_ic, (8.0, 8.0))
                # Rotate 90° CCW: swap w/h for spacing
                poe_eff_w, poe_eff_h = poe_h, poe_w
                # Place centered horizontally on eth_anchor_x, above J13
                poe_tx = eth_anchor_x
                # Position above where J13 will go (bottom edge area)
                poe_ty = bounds[3] - 25.0  # ~25mm above bottom edge
                poe_tx = max(bounds[0] + poe_eff_w / 2 + 1,
                             min(bounds[2] - poe_eff_w / 2 - 1, poe_tx))
                poe_ty = max(bounds[1] + poe_eff_h / 2 + 1,
                             min(bounds[3] - poe_eff_h / 2 - 1, poe_ty))
                ppx, ppy = eth_grid.find_free_pos(
                    poe_tx, poe_ty, poe_eff_w, poe_eff_h, max_radius=15.0,
                )
                positions[poe_ic] = (ppx, ppy, 270.0)  # 270° = 90° CCW
                eth_grid.place(ppx, ppy, poe_eff_w, poe_eff_h)
                ethernet_fixed.add(poe_ic)
                placed_eth.add(poe_ic)
                _log.info("    %s (PHY) → (%.1f, %.1f) rot=270", poe_ic, ppx, ppy)
                # Place PoE caps near the IC
                cap_y = ppy - poe_eff_h / 2.0 - 2.0
                for cap_ref in poe_caps:
                    if (cap_ref not in positions or cap_ref in fixed_refs
                            or cap_ref in placed_eth):
                        continue
                    cw, ch = fp_sizes.get(cap_ref, (1.0, 0.5))
                    cpx, cpy = eth_grid.find_free_pos(
                        ppx, cap_y, cw, ch, max_radius=8.0,
                    )
                    positions[cap_ref] = (cpx, cpy, 0.0)
                    eth_grid.place(cpx, cpy, cw, ch)
                    ethernet_fixed.add(cap_ref)
                    placed_eth.add(cap_ref)
                    cap_y = cpy - ch / 2.0 - 1.0

            # J13 (RJ45): place on bottom board edge, facing OUTWARD (180° rotation)
            # The RJ45 connector MUST be on the bottom edge for cable access.
            from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space
            for ref in eth_connectors:
                if ref not in positions or ref in fixed_refs:
                    continue
                fp_match = None
                for fp in initial_pcb.footprints:
                    if fp.ref == ref:
                        fp_match = fp
                        break
                w, h = fp_sizes.get(ref, (2.0, 2.0))
                # Place centroid so bottom edge of courtyard touches bottom
                # board edge with 1mm margin.
                cent_y = bounds[3] - h / 2.0 - 1.0
                cent_x = eth_anchor_x
                if fp_match is not None:
                    # Use pad extent for accurate edge placement
                    trial_origin_x = eth_anchor_x
                    trial_origin_y = bounds[3] - 3.0
                    _, _, _, pad_max_y = pad_extent_in_board_space(
                        fp_match, trial_origin_x, trial_origin_y, 180.0,
                    )
                    # Keep pad extent + 2.0mm margin from board edge
                    # (review agent adds 1.0mm to pad extents for courtyard)
                    edge_margin = 2.0
                    if pad_max_y > bounds[3] - edge_margin:
                        trial_origin_y -= (pad_max_y - bounds[3] + edge_margin)
                    cent_x, cent_y = origin_to_centroid(
                        fp_match, trial_origin_x, trial_origin_y, 180.0,
                    )
                # Force-place on bottom edge — no grid search, J13 MUST be
                # on the board edge for cable access
                px, py = cent_x, cent_y
                # Clamp X to ethernet zone — never drift into analog zone
                px = max(ezx1 + w / 2.0, min(ezx2 - w / 2.0, px))
                positions[ref] = (px, py, 180.0)
                ethernet_fixed.add(ref)
                placed_eth.add(ref)
                _log.info("    %s (RJ45) → bottom edge (%.1f, %.1f)",
                          ref, px, py)

            # Any remaining eth refs
            remaining_eth = sorted(
                eth_group_refs - placed_eth - fixed_refs,
            )
            if remaining_eth:
                _place_eth_column(
                    [r for r in remaining_eth if r in positions],
                    eth_anchor_x, col1_bottom,
                )

            _log.info(
                "    3c4: organized %d ethernet components in signal chain: %s",
                len(ethernet_fixed), sorted(ethernet_fixed),
            )

            # Local overlap fix: shift caps that collide with crystal.
            # Use generous margin (2.0mm) so collision resolution can't
            # push them back into the crystal.
            _crystal_placed = [r for r in placed_eth if r.startswith("Y")]
            for yref in _crystal_placed:
                yx, yy, yrot = positions[yref]
                yw, yh = fp_sizes.get(yref, (3.2, 1.5))
                for cref in list(placed_eth):
                    if cref == yref or not cref.startswith("C"):
                        continue
                    cx, cy, crot = positions[cref]
                    cw, ch = fp_sizes.get(cref, (1.5, 1.0))
                    # Check overlap with 2mm safety margin
                    _margin = 2.0
                    if (abs(cx - yx) < (cw + yw) / 2.0 + _margin
                            and abs(cy - yy) < (ch + yh) / 2.0 + _margin):
                        # Shift cap below crystal with generous gap
                        new_cy = yy + yh / 2.0 + ch / 2.0 + 2.0
                        new_cy = max(ezy1 + 2.0, min(ezy2 - 2.0, new_cy))
                        positions[cref] = (cx, new_cy, crot)
                        _log.info("    3c4: shifted %s below %s "
                                  "(%.1f,%.1f) → (%.1f,%.1f)",
                                  cref, yref, cx, cy, cx, new_cy)

    # Protect all ethernet-organized components from ALL collision passes.
    # Without this, the review loop (which uses relay_fixed_review without
    # ethernet_fixed) can scatter Y1/caps 30mm+ from U6.
    fixed_refs.update(ethernet_fixed)

    # 3c4-post: Push non-ethernet components clear of ethernet ICs
    # U7 (optocoupler) in analog group drifts into U6 (W5500) bounding box.
    # IMPORTANT: Skip relay group components — pushing them destroys relay layout.
    if ethernet_fixed:
        eth_ic_refs = [r for r in ethernet_fixed if r.startswith("U")]
        for eic in eth_ic_refs:
            if eic not in positions:
                continue
            ex, ey, erot = positions[eic]
            ew, eh = fp_sizes.get(eic, (5.0, 5.0))
            if erot % 180 in (90, 270):
                ew, eh = eh, ew
            for ref in list(positions):
                if ref in ethernet_fixed or ref in fixed_refs:
                    continue
                # Never push relay group components — they belong to phase 3a/3b
                if ref in relay_support_refs or ref.startswith("K"):
                    continue
                rx, ry, rrot = positions[ref]
                rw, rh = fp_sizes.get(ref, (1.5, 1.0))
                if rrot % 180 in (90, 270):
                    rw, rh = rh, rw
                gx = abs(rx - ex) - (rw + ew) / 2.0
                gy = abs(ry - ey) - (rh + eh) / 2.0
                if gx < -0.1 and gy < -0.1:
                    # Overlap — push away from ethernet IC
                    push_dist = -gx + 1.0
                    if rx < ex:
                        positions[ref] = (rx - push_dist, ry, rrot)
                    else:
                        positions[ref] = (rx + push_dist, ry, rrot)
                    _log.info("    3c4-post: pushed %s away from %s by %.1fmm",
                              ref, eic, push_dist)

    # 3h. Template-guided refinement — apply subcircuit layout templates
    _log.info("  3h: Template-guided refinement")
    # Protect all refs already placed by earlier phases from template moves
    template_protected = (fixed_refs | relay_support_refs | adc_channel_refs
                          | mcu_peripheral_refs | power_group_fixed
                          | ethernet_fixed | top_edge_connector_refs)
    positions, template_fixed = _apply_template_refinement(
        positions, fp_sizes, bounds, requirements, subcircuits, template_protected,
    )

    # 3c-late. Re-pull decoupling caps close to ICs after all group phases
    # Group phases (3c1-3c4) move caps into group layouts, scattering them
    # far from their ICs. This late pass forces them back within 3mm edge.
    # Only move caps in the SAME FeatureBlock as their IC to avoid cross-group
    # contamination that kills group cohesion and isolation scores.
    _log.info("  3c-late: Late decoupling re-tightening")
    # Build ref→group map for same-group filtering
    _ref_to_group: dict[str, str] = {}
    for feat in requirements.features:
        for comp in feat.components:
            r = comp.ref if hasattr(comp, "ref") else comp
            _ref_to_group[r] = feat.name

    _decoupling_pulled = 0
    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.DECOUPLING:
            continue
        ic_ref = sc.anchor_ref
        if ic_ref not in positions:
            continue
        ic_group = _ref_to_group.get(ic_ref, "")
        ix, iy, _irot = positions[ic_ref]
        iw, ih = fp_sizes.get(ic_ref, (5.0, 5.0))
        # For rotated ICs, swap effective dimensions
        if _irot % 180 in (90.0, 270.0):
            iw, ih = ih, iw

        placed_count = 0
        for cap_ref in sc.refs:
            if cap_ref == ic_ref or not cap_ref.startswith("C"):
                continue
            if cap_ref not in positions:
                continue
            # Only pull caps from the same FeatureBlock as the IC
            cap_group = _ref_to_group.get(cap_ref, "")
            if cap_group != ic_group:
                continue
            # Don't re-pull power group caps — 3c1 already placed them
            # in signal-flow order around their ICs.
            if cap_ref in power_group_fixed:
                continue
            cx, cy, crot = positions[cap_ref]
            cw, ch = fp_sizes.get(cap_ref, (1.5, 1.0))

            # Edge-to-edge distance
            dx_edge = abs(cx - ix) - (iw + cw) / 2.0
            dy_edge = abs(cy - iy) - (ih + ch) / 2.0
            if dx_edge <= 0 and dy_edge <= 0:
                edge_dist = 0.0
            elif dx_edge <= 0:
                edge_dist = dy_edge
            elif dy_edge <= 0:
                edge_dist = dx_edge
            else:
                edge_dist = math.sqrt(dx_edge ** 2 + dy_edge ** 2)

            if edge_dist <= 3.0:
                continue  # Already close enough

            # Place cap on nearest IC edge — spread around IC perimeter
            # Alternate top/bottom/left/right based on cap index
            side = placed_count % 4
            if side == 0:  # top
                tx = ix + (placed_count // 4) * (cw + 0.5)
                ty = iy - ih / 2.0 - ch / 2.0 - 0.5
            elif side == 1:  # bottom
                tx = ix + (placed_count // 4) * (cw + 0.5)
                ty = iy + ih / 2.0 + ch / 2.0 + 0.5
            elif side == 2:  # right
                tx = ix + iw / 2.0 + cw / 2.0 + 0.5
                ty = iy + (placed_count // 4) * (ch + 0.5)
            else:  # left
                tx = ix - iw / 2.0 - cw / 2.0 - 0.5
                ty = iy + (placed_count // 4) * (ch + 0.5)

            # Clamp to board
            tx = max(bounds[0] + 1.0, min(bounds[2] - 1.0, tx))
            ty = max(bounds[1] + 1.0, min(bounds[3] - 1.0, ty))
            positions[cap_ref] = (tx, ty, crot)
            placed_count += 1
            _decoupling_pulled += 1

    _log.info("    3c-late: re-pulled %d decoupling caps", _decoupling_pulled)

    # 3g. Collision resolution (group-constrained, then unconstrained final pass)
    _log.info("  3g: Collision resolution")
    group_bboxes = _extract_group_bboxes(requirements, positions, fp_sizes)
    # Protect relay sub-circuit, ADC channel, and MCU peripheral positions
    subcircuit_fixed = (relay_support_refs | adc_channel_refs
                        | mcu_peripheral_refs | power_group_fixed
                        | ethernet_fixed | template_fixed
                        | top_edge_connector_refs)
    relay_fixed = fixed_refs | subcircuit_fixed | {
        r for r in positions if r.startswith("K")
    }
    positions = _resolve_collisions(
        positions, fp_sizes, bounds, relay_fixed, group_bboxes=group_bboxes,
    )
    # Targeted final pass — only unprotect refs that are actually colliding
    remaining_collisions = _count_collisions(positions, fp_sizes)
    if remaining_collisions:
        _log.info("  3g: %d remaining — targeted pass", len(remaining_collisions))
        # Only unprotect subcircuit refs that are involved in collisions,
        # BUT keep MCU peripheral refs and connector refs always protected
        # (they were placed intentionally by pin-side-aware logic)
        colliding_refs = set()
        for a, b in remaining_collisions:
            colliding_refs.add(a)
            colliding_refs.add(b)
        # Never unprotect the MCU IC itself, MCU peripherals, relays,
        # edge connectors, or ethernet connectors. MCU peripherals were
        # placed with courtyard awareness; moving them in collision
        # resolution can push them back inside U3's module body.
        # Ethernet connectors (J13) are anchored on board edges.
        # Don't over-protect relays or peripherals that collide with each
        # other — at least one in each colliding pair must be movable.
        always_fixed_base = (mcu_peripheral_refs | top_edge_connector_refs
                             | ethernet_fixed | adc_channel_refs
                             | adc_ic_refs | relay_support_refs
                             | power_group_fixed
                             | {r for r in positions if r.startswith("K")})
        # When both refs in a collision pair are in always_fixed,
        # unprotect the smaller one so collision resolution can act.
        # For relay(K)/connector(J) collisions, unprotect the connector
        # (GPIO headers are less critical than relay subcircuit cohesion).
        intra_fixed_unprotect: set[str] = set()
        for a, b in remaining_collisions:
            if a in always_fixed_base and b in always_fixed_base:
                area_a = fp_sizes.get(a, (2, 2))[0] * fp_sizes.get(a, (2, 2))[1]
                area_b = fp_sizes.get(b, (2, 2))[0] * fp_sizes.get(b, (2, 2))[1]
                smaller = a if area_a <= area_b else b
                # Don't unprotect ICs, crystals, relays, transistors,
                # or power supply passives (tight hot-loop layout).
                if (not smaller.startswith(("U", "Y", "K", "Q"))
                        and smaller not in power_group_fixed):
                    intra_fixed_unprotect.add(smaller)
        always_fixed = always_fixed_base - intra_fixed_unprotect
        # Keep protection on subcircuit refs NOT involved in collisions
        targeted_fixed = fixed_refs | (subcircuit_fixed - colliding_refs) | always_fixed
        positions = _resolve_collisions(
            positions, fp_sizes, bounds, targeted_fixed,
        )

    # Post-3g: Enforce ethernet connectors on bottom edge
    for ref in ethernet_fixed:
        if ref.startswith("J") and ref in positions and ref not in fixed_refs:
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            bottom_target = max_y - h / 2.0 - 1.0
            rx, ry, rot = positions[ref]
            if ry < bottom_target - 5.0:
                _log.info("  Enforcing %s to bottom edge: y %.1f → %.1f", ref, ry, bottom_target)
                positions[ref] = (rx, bottom_target, 180.0)

    # ===================================================================
    # Final: Clamp, score, review
    # ===================================================================
    _log.info("=== Final: Clamping and review ===")

    # Clamp ALL components using actual pad extent so nothing extends past
    # the board edge.  Works for both asymmetric connectors and symmetric parts.
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space
    fp_lookup = {fp.ref: fp for fp in initial_pcb.footprints}
    _edge_m = 1.5  # pad-to-board-edge margin (mm)
    for ref, (rx, ry, rot) in list(positions.items()):
        if ref in fixed_refs:
            continue
        fp_obj = fp_lookup.get(ref)
        if fp_obj is not None and fp_obj.pads:
            # Use actual pad extent for all components with pads
            ori_x, ori_y = centroid_to_origin(fp_obj, rx, ry, rot)
            px0, py0, px1, py1 = pad_extent_in_board_space(
                fp_obj, ori_x, ori_y, rot,
            )
            shift_x = shift_y = 0.0
            if px0 < min_x + _edge_m:
                shift_x = (min_x + _edge_m) - px0
            elif px1 > max_x - _edge_m:
                shift_x = (max_x - _edge_m) - px1
            if py0 < min_y + _edge_m:
                shift_y = (min_y + _edge_m) - py0
            elif py1 > max_y - _edge_m:
                shift_y = (max_y - _edge_m) - py1
            if shift_x != 0.0 or shift_y != 0.0:
                new_cx, new_cy = origin_to_centroid(
                    fp_obj, ori_x + shift_x, ori_y + shift_y, rot,
                )
                positions[ref] = (new_cx, new_cy, rot)
                _log.info("  Clamped %s: pad extent was (%.1f,%.1f)-(%.1f,%.1f), "
                          "shifted by (%.1f,%.1f)", ref, px0, py0, px1, py1,
                          shift_x, shift_y)
        else:
            # Fallback for components without pads — use rotation-aware size
            w, h = _rotation_aware_size(ref, positions, fp_sizes)
            clamped_x = max(min_x + w / 2 + 1.5, min(max_x - w / 2 - 1.5, rx))
            clamped_y = max(min_y + h / 2 + 1.5, min(max_y - h / 2 - 1.5, ry))
            if clamped_x != rx or clamped_y != ry:
                positions[ref] = (clamped_x, clamped_y, rot)

    # Post-clamp collision resolution — protect relay sub-circuits
    # Use targeted approach: unprotect colliding subcircuit refs but keep
    # MCU peripherals, edge connectors, ethernet connectors, and relays fixed.
    post_clamp_collisions_list = _count_collisions(positions, fp_sizes)
    if post_clamp_collisions_list:
        _log.info("  %d post-clamp collisions — resolving",
                  len(post_clamp_collisions_list))
        clamp_colliding = set()
        for a, b in post_clamp_collisions_list:
            clamp_colliding.add(a)
            clamp_colliding.add(b)
        clamp_always_base = (mcu_peripheral_refs | top_edge_connector_refs
                             | ethernet_fixed | adc_channel_refs | adc_ic_refs
                             | relay_support_refs | power_group_fixed
                             | {r for r in positions if r.startswith("K")})
        clamp_unprotect: set[str] = set()
        for a, b in post_clamp_collisions_list:
            if a in clamp_always_base and b in clamp_always_base:
                area_a = fp_sizes.get(a, (2, 2))[0] * fp_sizes.get(a, (2, 2))[1]
                area_b = fp_sizes.get(b, (2, 2))[0] * fp_sizes.get(b, (2, 2))[1]
                smaller = a if area_a <= area_b else b
                if (not smaller.startswith(("U", "Y", "K", "Q"))
                        and smaller not in power_group_fixed):
                    clamp_unprotect.add(smaller)
        clamp_always_fixed = clamp_always_base - clamp_unprotect
        clamp_targeted = (fixed_refs
                          | (subcircuit_fixed - clamp_colliding)
                          | clamp_always_fixed)
        positions = _resolve_collisions(
            positions, fp_sizes, bounds, clamp_targeted,
        )

    # Post-clamp decoupling re-pull — clamping moves ICs (especially U3
    # ESP32 module whose pad extent exceeds board edge), which scatters
    # their decoupling caps.  Re-pull them tight after clamping is final.
    _post_clamp_decoup = 0
    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.DECOUPLING:
            continue
        ic_ref = sc.anchor_ref
        if ic_ref not in positions:
            continue
        ic_group = _ref_to_group.get(ic_ref, "")
        ix, iy, _irot = positions[ic_ref]
        iw, ih = fp_sizes.get(ic_ref, (5.0, 5.0))
        if _irot % 180 in (90.0, 270.0):
            iw, ih = ih, iw
        placed_count = 0
        for cap_ref in sc.refs:
            if cap_ref == ic_ref or not cap_ref.startswith("C"):
                continue
            if cap_ref not in positions:
                continue
            cap_group = _ref_to_group.get(cap_ref, "")
            if cap_group != ic_group:
                continue
            cx, cy, crot = positions[cap_ref]
            cw, ch = fp_sizes.get(cap_ref, (1.5, 1.0))
            dx_edge = abs(cx - ix) - (iw + cw) / 2.0
            dy_edge = abs(cy - iy) - (ih + ch) / 2.0
            if dx_edge <= 0 and dy_edge <= 0:
                edge_dist = 0.0
            elif dx_edge <= 0:
                edge_dist = dy_edge
            elif dy_edge <= 0:
                edge_dist = dx_edge
            else:
                edge_dist = math.sqrt(dx_edge ** 2 + dy_edge ** 2)
            if edge_dist <= 5.0:
                continue  # Close enough after clamping
            # Place cap near IC — alternate sides
            side = placed_count % 4
            if side == 0:
                tx, ty = ix + placed_count // 4 * (cw + 0.5), iy - ih / 2.0 - ch / 2.0 - 0.5
            elif side == 1:
                tx, ty = ix + placed_count // 4 * (cw + 0.5), iy + ih / 2.0 + ch / 2.0 + 0.5
            elif side == 2:
                tx, ty = ix + iw / 2.0 + cw / 2.0 + 0.5, iy + placed_count // 4 * (ch + 0.5)
            else:
                tx, ty = ix - iw / 2.0 - cw / 2.0 - 0.5, iy + placed_count // 4 * (ch + 0.5)
            tx = max(bounds[0] + 1.0, min(bounds[2] - 1.0, tx))
            ty = max(bounds[1] + 1.0, min(bounds[3] - 1.0, ty))
            positions[cap_ref] = (tx, ty, crot)
            placed_count += 1
            _post_clamp_decoup += 1
    if _post_clamp_decoup:
        _log.info("  Post-clamp decoupling re-pull: %d caps repositioned",
                  _post_clamp_decoup)

    # EE Review loop (limited passes)
    _log.info("  Running EE review (max %d passes)", max_review_passes)
    best_positions = dict(positions)
    best_review: PlacementReview | None = None
    best_violation_count = float("inf")

    for pass_num in range(max_review_passes):
        positions_tuple = _dict_to_positions(positions)
        current_pcb = _apply_positions(initial_pcb, positions_tuple)
        review = review_placement(
            current_pcb, requirements,
            subcircuits=subcircuits, domain_map=domain_map,
        )

        critical_major = sum(
            1 for v in review.violations
            if v.severity in ("critical", "major")
        )
        _log.info(
            "  Review pass %d: %s — %d critical/major",
            pass_num + 1, review.summary, critical_major,
        )

        if critical_major < best_violation_count:
            best_violation_count = critical_major
            best_positions = dict(positions)
            best_review = review

        if critical_major == 0:
            break

        # Apply suggested fixes — protect relay sub-circuit positions
        relay_fixed_review = fixed_refs | subcircuit_fixed | {
            r for r in positions if r.startswith("K")
        }
        positions = _apply_review_fixes(
            positions, review, relay_fixed_review, fp_sizes, bounds,
        )

    # Post-review collision resolution — targeted approach (unprotect
    # colliding subcircuit refs, keep MCU/edge/ethernet connectors fixed)
    post_review_collisions = _count_collisions(best_positions, fp_sizes)
    if post_review_collisions:
        _log.info(
            "  %d post-review collisions — resolving", len(post_review_collisions),
        )
        pr_colliding = set()
        for a, b in post_review_collisions:
            pr_colliding.add(a)
            pr_colliding.add(b)
        pr_always_base = (mcu_peripheral_refs | top_edge_connector_refs
                          | ethernet_fixed
                          | adc_channel_refs | adc_ic_refs
                          | relay_support_refs | power_group_fixed
                          | {r for r in best_positions if r.startswith("K")})
        pr_unprotect: set[str] = set()
        for a, b in post_review_collisions:
            if a in pr_always_base and b in pr_always_base:
                area_a = fp_sizes.get(a, (2, 2))[0] * fp_sizes.get(a, (2, 2))[1]
                area_b = fp_sizes.get(b, (2, 2))[0] * fp_sizes.get(b, (2, 2))[1]
                smaller = a if area_a <= area_b else b
                if (not smaller.startswith(("U", "Y", "K", "Q"))
                        and smaller not in power_group_fixed):
                    pr_unprotect.add(smaller)
        pr_always = pr_always_base - pr_unprotect
        pr_targeted = (fixed_refs
                       | (subcircuit_fixed - pr_colliding)
                       | pr_always)
        best_positions = _resolve_collisions(
            best_positions, fp_sizes, bounds, pr_targeted,
        )

    # 3c2-late: Post-collision ADC channel re-alignment
    # Re-form vertical columns using FINAL connector positions to determine
    # left-to-right order.  All channels are placed in the analog zone,
    # sorted so the channel fed by the leftmost connector is leftmost.
    if adc_channels and adc_ic_refs:
        _log.info("  3c2-late: ADC channel re-alignment (connector-ordered)")
        _realigned = 0

        # Build R_top → connector X mapping using FINAL positions
        _late_r_top_x: dict[str, float] = {}
        for net in requirements.nets:
            j_conns = [c for c in net.connections if c.ref.startswith("J")]
            r_conns = [c for c in net.connections
                       if c.ref.startswith("R") and c.ref in best_positions]
            if j_conns and r_conns:
                for j_conn in j_conns:
                    j_pos = best_positions.get(j_conn.ref)
                    if j_pos:
                        for r_conn in r_conns:
                            _late_r_top_x[r_conn.ref] = j_pos[0]

        # Build flat list of all channels with their connector X for sorting
        _all_ch_with_x: list[tuple[float, str, str, list[str]]] = []
        for ic_ref, ic_pin, passives in adc_channels:
            conn_x = 999.0
            r_refs_ch = sorted(r for r in passives if r.startswith("R"))
            for r in r_refs_ch:
                if r in _late_r_top_x:
                    conn_x = _late_r_top_x[r]
                    break
            _all_ch_with_x.append((conn_x, ic_ref, ic_pin, passives))

        # Sort ALL channels globally by connector X (leftmost connector → leftmost column)
        _all_ch_with_x.sort(key=lambda t: t[0])

        # Find the analog zone for X bounds
        _az = None
        for z in zones:
            if z.name == "analog":
                _az = z
                break
        if _az:
            az_x1, az_y1, az_x2, az_y2 = _az.rect
        else:
            az_x1, az_y1, az_x2, az_y2 = bounds

        # Spread channels evenly across the analog zone width
        n_total_ch = len(_all_ch_with_x)
        ch_zone_width = az_x2 - az_x1 - 4.0  # 2mm margin each side
        ch_spacing = min(_CHANNEL_SPACING_MM,
                         ch_zone_width / max(n_total_ch - 1, 1))
        total_ch_width = (n_total_ch - 1) * ch_spacing
        ch_x_start = az_x1 + 2.0 + (ch_zone_width - total_ch_width) / 2.0

        # Place channel strips — vertical columns running downward from top
        _ch_y_top = az_y1 + 2.0  # near top of analog zone

        # Track which IC's channels are at which X for IC repositioning
        _ic_ch_xs: dict[str, list[float]] = {}

        for ch_idx, (conn_x, ic_ref, ic_pin, passives) in enumerate(_all_ch_with_x):
            ch_x = ch_x_start + ch_idx * ch_spacing

            _ic_ch_xs.setdefault(ic_ref, []).append(ch_x)

            r_refs = sorted([r for r in passives if r.startswith("R")])
            d_refs = [r for r in passives if r.startswith("D")]
            c_refs = [r for r in passives if r.startswith("C")]

            # Strip order top→bottom: R_top → C_filter → D_clamp → R_bot → (IC below)
            strip_order: list[str] = []
            if len(r_refs) >= 1:
                strip_order.append(r_refs[0])   # R_top (near connector)
            strip_order.extend(c_refs)           # C_filter
            strip_order.extend(d_refs)           # D_clamp
            if len(r_refs) >= 2:
                strip_order.append(r_refs[1])   # R_bot (near IC)

            strip_y = _ch_y_top
            for ref in strip_order:
                if ref not in best_positions:
                    continue
                raw_w, raw_h = fp_sizes.get(ref, (2.0, 2.0))
                target_x = ch_x
                target_y = strip_y + raw_h / 2.0
                target_x = max(bounds[0] + 2.0, min(bounds[2] - 2.0, target_x))
                target_y = max(bounds[1] + 2.0, min(bounds[3] - 2.0, target_y))
                old_x, old_y, _old_rot = best_positions[ref]
                if abs(old_x - target_x) > 1.0 or abs(old_y - target_y) > 1.0:
                    _realigned += 1
                best_positions[ref] = (target_x, target_y, 0.0)
                strip_y = target_y + raw_h / 2.0 + _STRIP_GAP_MM

            _log.info(
                "    3c2-late: ch%d (%s.%s) → x=%.1f (conn_x=%.1f)",
                ch_idx, ic_ref, ic_pin, ch_x, conn_x,
            )

        # Reposition ADC ICs below their channel groups
        for ic_ref, ch_xs in _ic_ch_xs.items():
            if ic_ref not in best_positions:
                continue
            ic_new_x = sum(ch_xs) / len(ch_xs)
            # Place IC below the strips (4 components × ~3.5mm each + gaps ≈ 20mm)
            ic_new_y = _ch_y_top + 22.0
            iw, ih = fp_sizes.get(ic_ref, (5.0, 5.0))
            ic_new_x = max(bounds[0] + iw / 2, min(bounds[2] - iw / 2, ic_new_x))
            ic_new_y = max(bounds[1] + ih / 2, min(bounds[3] - ih / 2, ic_new_y))
            best_positions[ic_ref] = (ic_new_x, ic_new_y, 0.0)
            _log.info(
                "    3c2-late: %s → (%.1f, %.1f) center of %d channels",
                ic_ref, ic_new_x, ic_new_y, len(ch_xs),
            )

        _log.info("    3c2-late: re-aligned %d ADC channel components", _realigned)

        # Quick collision resolution after re-alignment — protect ADC channels
        _post_adc_collisions = _count_collisions(best_positions, fp_sizes)
        if _post_adc_collisions:
            _log.info(
                "    3c2-late: %d post-alignment collisions — resolving",
                len(_post_adc_collisions),
            )
            # Protect ADC channel refs, ICs, relay support, ethernet, MCU
            _adc_fixed = (fixed_refs | adc_channel_refs | adc_ic_refs
                          | top_edge_connector_refs | relay_support_refs
                          | ethernet_fixed | mcu_peripheral_refs
                          | power_group_fixed
                          | {r for r in best_positions if r.startswith("K")})
            best_positions = _resolve_collisions(
                best_positions, fp_sizes, bounds, _adc_fixed,
            )

    # 3b-late: Post-collision relay driver re-alignment
    # Runs LAST — after all other late phases and their collision resolution.
    # Re-snaps each relay's support (Q, D, R) + LED indicators back into a
    # tight grid directly below the relay.
    _log.info("  3b-late: Relay driver re-alignment")
    _all_relay_refs: set[str] = set()
    for sc in sc_list:
        if sc.circuit_type == SubCircuitType.RELAY_DRIVER:
            _all_relay_refs.update(sc.refs)
            _all_relay_refs.update(_relay_leds.get(sc.anchor_ref, []))
    _all_relay_refs.update(r for r in best_positions if r.startswith("K"))

    _relay_realigned = 0
    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.RELAY_DRIVER:
            continue
        anchor = sc.anchor_ref
        if anchor not in best_positions:
            continue
        kx, ky, krot = best_positions[anchor]
        kw, kh = fp_sizes.get(anchor, (18.0, 16.0))
        if krot % 180 in (90.0, 270.0):
            kw, kh = kh, kw
        _log.info("    3b-late: %s centroid=(%.1f,%.1f) rot=%.0f size=%.1fx%.1f",
                  anchor, kx, ky, krot, kw, kh)

        # Core support: Q, D_flyback, R_gate
        support_members = [
            r for r in sc.refs
            if r != anchor and r in best_positions
        ]
        support_members.sort(key=lambda r: (
            0 if r.startswith("Q") else 1 if r.startswith("D") else 2, r,
        ))

        # LED indicators: D_led + R_led from _relay_leds mapping
        led_members = sorted(
            set(_relay_leds.get(anchor, []))
            & set(best_positions.keys()),
        )

        target_y_base = ky + kh / 2.0 + 1.0
        col = 0
        row_y = target_y_base
        row_max_h = 0.0
        cols_per_row = 2

        # Place core support in 2-column grid
        for ref in support_members:
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            px = kx - kw / 2.0 + (col + 0.5) * (kw / cols_per_row)
            py = row_y + h / 2.0
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
            old_x, old_y, old_rot = best_positions[ref]
            if abs(old_x - px) > 1.0 or abs(old_y - py) > 1.0:
                _relay_realigned += 1
            best_positions[ref] = (px, py, old_rot)
            row_max_h = max(row_max_h, h)
            col += 1
            if col >= cols_per_row:
                col = 0
                row_y += row_max_h + 0.5
                row_max_h = 0.0

        # Place LED indicators in next row below support
        if col > 0:
            row_y += row_max_h + 0.5
            col = 0
            row_max_h = 0.0
        for ref in led_members:
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            px = kx - kw / 2.0 + (col + 0.5) * (kw / cols_per_row)
            py = row_y + h / 2.0
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
            old_x, old_y, old_rot = best_positions[ref]
            if abs(old_x - px) > 1.0 or abs(old_y - py) > 1.0:
                _relay_realigned += 1
            best_positions[ref] = (px, py, old_rot)
            row_max_h = max(row_max_h, h)
            col += 1
            if col >= cols_per_row:
                col = 0
                row_y += row_max_h + 0.5
                row_max_h = 0.0

    if _relay_realigned:
        _log.info("    3b-late: re-aligned %d relay support components", _relay_realigned)
        # Resolve collisions — protect ALL relay refs (K + support) so the
        # tight grid is preserved.  Non-relay refs that collide with the grid
        # get pushed away.  Also protect ADC channels and top connectors.
        _post_relay_collisions = _count_collisions(best_positions, fp_sizes)
        if _post_relay_collisions:
            _log.info(
                "    3b-late: %d post-alignment collisions — resolving",
                len(_post_relay_collisions),
            )
            _relay_fixed = (fixed_refs
                            | _all_relay_refs
                            | top_edge_connector_refs
                            | adc_channel_refs | adc_ic_refs
                            | power_group_fixed)
            best_positions = _resolve_collisions(
                best_positions, fp_sizes, bounds, _relay_fixed,
            )

    # MCU decoupling re-pull — late phases and collision resolution may have
    # scattered MCU decoupling caps. Re-pull them tight against U3's LEFT side.
    # ESP32-S3-WROOM at 180° has VCC/GND on side castellation pads, so caps
    # go to the left (toward board center) in a vertical column.
    if mcu_ref_c3 and mcu_ref_c3 in best_positions:
        _mcu_fx, _mcu_fy, _mcu_fr = best_positions[mcu_ref_c3]
        _mcu_fw, _mcu_fh = fp_sizes.get(mcu_ref_c3, (19.5, 25.4))
        if _mcu_fr % 180 in (90.0, 270.0):
            _mcu_fw, _mcu_fh = _mcu_fh, _mcu_fw
        _mcu_left = _mcu_fx - _mcu_fw / 2.0
        _mcu_decoup_pulled = 0
        _mcu_decoup_refs = sorted(
            r for r in best_positions
            if r.startswith("C") and r in mcu_peripheral_refs
        )
        # Place caps in a vertical column LEFT of U3, centered on U3's Y
        _cap_x = _mcu_left - 3.0  # 3mm left of MCU body
        _cap_y_start = _mcu_fy - (_mcu_fh / 3.0)  # start 1/3 above center
        _cap_y = _cap_y_start
        for ref in _mcu_decoup_refs:
            cx, cy, crot = best_positions[ref]
            # Use edge-to-edge distance (not center-to-center) since
            # ESP32 module is 25mm tall — center-to-center >8mm for any
            # cap placed beside the module, even when touching the body.
            cw, ch = fp_sizes.get(ref, (2.5, 1.5))
            dx_edge = max(0.0, abs(cx - _mcu_fx) - (_mcu_fw + cw) / 2.0)
            dy_edge = max(0.0, abs(cy - _mcu_fy) - (_mcu_fh + ch) / 2.0)
            edge_dist = (dx_edge ** 2 + dy_edge ** 2) ** 0.5
            if edge_dist > 5.0:
                tx = _cap_x
                ty = _cap_y
                _cap_y += ch + 1.0
                tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
                ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
                best_positions[ref] = (tx, ty, crot)
                _mcu_decoup_pulled += 1
        if _mcu_decoup_pulled:
            _log.info("MCU decoupling re-pull: %d caps to U3 left side",
                      _mcu_decoup_pulled)

    # Final edge clamp — ensure ALL components are inside board after late phases
    # Uses pad extent for all components with pads, not just connectors.
    for ref, (rx, ry, rot) in list(best_positions.items()):
        fp_obj = fp_lookup.get(ref)
        if fp_obj is not None and fp_obj.pads:
            ori_x, ori_y = centroid_to_origin(fp_obj, rx, ry, rot)
            px0, py0, px1, py1 = pad_extent_in_board_space(
                fp_obj, ori_x, ori_y, rot,
            )
            shift_x = shift_y = 0.0
            _fm = 1.5
            if px0 < min_x + _fm:
                shift_x = (min_x + _fm) - px0
            elif px1 > max_x - _fm:
                shift_x = (max_x - _fm) - px1
            if py0 < min_y + _fm:
                shift_y = (min_y + _fm) - py0
            elif py1 > max_y - _fm:
                shift_y = (max_y - _fm) - py1
            if shift_x != 0.0 or shift_y != 0.0:
                new_cx, new_cy = origin_to_centroid(
                    fp_obj, ori_x + shift_x, ori_y + shift_y, rot,
                )
                best_positions[ref] = (new_cx, new_cy, rot)
                _log.info("  Late-phase clamp %s: shifted by (%.1f,%.1f)",
                          ref, shift_x, shift_y)
        else:
            w, h = _rotation_aware_size(ref, best_positions, fp_sizes)
            clamped_x = max(min_x + w / 2 + 1.5, min(max_x - w / 2 - 1.5, rx))
            clamped_y = max(min_y + h / 2 + 1.5, min(max_y - h / 2 - 1.5, ry))
            if clamped_x != rx or clamped_y != ry:
                best_positions[ref] = (clamped_x, clamped_y, rot)

    # Final crystal-cap overlap resolution — collision resolution may have
    # pushed caps back toward crystal refs.  Shift any overlapping caps away.
    _crystal_refs_final = [r for r in best_positions if r.startswith("Y")]
    for yref in _crystal_refs_final:
        yx, yy, yrot = best_positions[yref]
        yw, yh = fp_sizes.get(yref, (3.2, 1.5))
        if yrot % 180 in (90.0, 270.0):
            yw, yh = yh, yw
        for cref in list(best_positions):
            if cref == yref or not cref.startswith("C"):
                continue
            cx, cy, crot = best_positions[cref]
            cw, ch = fp_sizes.get(cref, (1.5, 1.0))
            if crot % 180 in (90.0, 270.0):
                cw, ch = ch, cw
            # Check overlap with 0.5mm margin
            overlap_x = (cw + yw) / 2.0 + 0.5 - abs(cx - yx)
            overlap_y = (ch + yh) / 2.0 + 0.5 - abs(cy - yy)
            if overlap_x > 0 and overlap_y > 0:
                # Push cap in the direction of least overlap
                if overlap_x < overlap_y:
                    shift = overlap_x + 0.5
                    new_cx = cx + shift if cx > yx else cx - shift
                    new_cx = max(min_x + 2, min(max_x - 2, new_cx))
                    best_positions[cref] = (new_cx, cy, crot)
                else:
                    shift = overlap_y + 0.5
                    new_cy = cy + shift if cy > yy else cy - shift
                    new_cy = max(min_y + 2, min(max_y - 2, new_cy))
                    best_positions[cref] = (cx, new_cy, crot)
                _log.info("Final crystal overlap fix: shifted %s away from %s",
                          cref, yref)

    # Final board-edge clamp — ensure NO component extends past board edge
    # after all collision resolution, review fixes, and late-phase adjustments.
    _final_clamp_count = 0
    for ref, (rx, ry, rot) in list(best_positions.items()):
        fp_obj = fp_lookup.get(ref)
        if fp_obj is not None and fp_obj.pads:
            ori_x, ori_y = centroid_to_origin(fp_obj, rx, ry, rot)
            px0, py0, px1, py1 = pad_extent_in_board_space(
                fp_obj, ori_x, ori_y, rot,
            )
            shift_x = shift_y = 0.0
            if px0 < min_x + _edge_m:
                shift_x = (min_x + _edge_m) - px0
            elif px1 > max_x - _edge_m:
                shift_x = (max_x - _edge_m) - px1
            if py0 < min_y + _edge_m:
                shift_y = (min_y + _edge_m) - py0
            elif py1 > max_y - _edge_m:
                shift_y = (max_y - _edge_m) - py1
            if shift_x != 0.0 or shift_y != 0.0:
                new_cx, new_cy = origin_to_centroid(
                    fp_obj, ori_x + shift_x, ori_y + shift_y, rot,
                )
                best_positions[ref] = (new_cx, new_cy, rot)
                _final_clamp_count += 1
        else:
            w, h = _rotation_aware_size(ref, best_positions, fp_sizes)
            clamped_x = max(min_x + w / 2 + 1.5, min(max_x - w / 2 - 1.5, rx))
            clamped_y = max(min_y + h / 2 + 1.5, min(max_y - h / 2 - 1.5, ry))
            if clamped_x != rx or clamped_y != ry:
                best_positions[ref] = (clamped_x, clamped_y, rot)
                _final_clamp_count += 1
    if _final_clamp_count:
        _log.info("Final board-edge clamp: %d components repositioned",
                  _final_clamp_count)

    # Build final PCB — use best_review from the main review loop
    # (which ran before late-phase re-alignments that may scatter components)
    if best_review is None:
        positions_tuple = _dict_to_positions(best_positions)
        final_pcb = _apply_positions(initial_pcb, positions_tuple)
        best_review = review_placement(
            final_pcb, requirements,
            subcircuits=subcircuits, domain_map=domain_map,
        )
    else:
        positions_tuple = _dict_to_positions(best_positions)
        final_pcb = _apply_positions(initial_pcb, positions_tuple)

    # Post-apply pad extent check — _apply_positions converts centroid→origin,
    # which can shift pads past the board edge. Re-clamp using actual output positions.
    _post_apply_clamp = 0
    new_fps = list(final_pcb.footprints)
    for i, fp in enumerate(new_fps):
        if not fp.pads:
            continue
        ox, oy = fp.position.x, fp.position.y
        rot = fp.rotation
        px0, py0, px1, py1 = pad_extent_in_board_space(fp, ox, oy, rot)
        shift_x = shift_y = 0.0
        if px0 < min_x + _edge_m:
            shift_x = (min_x + _edge_m) - px0
        elif px1 > max_x - _edge_m:
            shift_x = (max_x - _edge_m) - px1
        if py0 < min_y + _edge_m:
            shift_y = (min_y + _edge_m) - py0
        elif py1 > max_y - _edge_m:
            shift_y = (max_y - _edge_m) - py1
        if shift_x != 0.0 or shift_y != 0.0:
            new_pos = Point(x=ox + shift_x, y=oy + shift_y)
            new_fps[i] = replace(fp, position=new_pos)
            _post_apply_clamp += 1
            _log.info("  Post-apply clamp %s: shifted (%.1f, %.1f)", fp.ref, shift_x, shift_y)
    if _post_apply_clamp:
        final_pcb = replace(final_pcb, footprints=tuple(new_fps))
        _log.info("Post-apply board-edge clamp: %d components", _post_apply_clamp)

    # Filter stale violations — the review was captured before the final
    # edge clamp and crystal overlap fix, so some violations may be resolved.
    if best_review is not None:
        from kicad_pipeline.optimization.review_agent import (
            PlacementReview,
            PlacementRule,
            PlacementViolation,
            _check_board_edge_clearance,
            _compute_grade,
        )
        fresh_edge = _check_board_edge_clearance(final_pcb)
        fresh_edge_refs = {r for v in fresh_edge for r in v.refs}

        # Also check which collision violations still exist in final PCB
        fresh_collisions = _count_collisions(best_positions, fp_sizes)
        fresh_collision_pairs = {
            tuple(sorted((a, b))) for a, b in fresh_collisions
        }

        filtered: list[PlacementViolation] = []
        _stale = 0
        for v in best_review.violations:
            if v.rule == PlacementRule.BOARD_EDGE_CLEARANCE:
                if not any(r in fresh_edge_refs for r in v.refs):
                    _stale += 1
                    continue
                # Replace with fresh measurement
                for fv in fresh_edge:
                    if fv.refs == v.refs:
                        filtered.append(fv)
                        break
                else:
                    filtered.append(v)
            elif v.rule == PlacementRule.COLLISION:
                # Check if this collision still exists in final positions
                pair = tuple(sorted(v.refs))
                if pair not in fresh_collision_pairs:
                    _stale += 1
                    continue
                filtered.append(v)
            else:
                filtered.append(v)
        if _stale:
            _log.info("Filtered %d stale edge clearance violations", _stale)
            grade = _compute_grade(tuple(filtered))
            n_crit = sum(1 for v in filtered if v.severity == "critical")
            n_major = sum(1 for v in filtered if v.severity == "major")
            n_minor = sum(1 for v in filtered if v.severity == "minor")
            best_review = PlacementReview(
                violations=tuple(filtered),
                summary=f"Grade {grade}: {len(filtered)} violations "
                        f"({n_crit} critical, {n_major} major, "
                        f"{n_minor} minor)",
                grade=grade,
            )

    # Post-optimization validation gate
    guard = validate_placement(final_pcb, requirements)
    if guard.issues:
        _log.warning("Placement guard issues (%d):", len(guard.issues))
        for issue in guard.issues:
            _log.warning("  %s", issue)
    else:
        _log.info("Placement guard: ALL CHECKS PASSED")

    _log.info("EE placement v5 complete: %s", best_review.summary)
    return final_pcb, best_review


# Default optimizer is the EE-grade one
optimize_placement = optimize_placement_ee
