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
from typing import TYPE_CHECKING

from kicad_pipeline.optimization.collision_resolver import (
    _count_collisions,  # noqa: F401 - re-exported
    _fp_courtyard_sizes,
    _group_of_ref,  # noqa: F401 - re-exported
    _PlacementGrid,  # noqa: F401 - re-exported
    _resolve_collisions,  # noqa: F401 - re-exported
    _rotation_aware_size,  # noqa: F401 - re-exported
)
from kicad_pipeline.optimization.functional_grouper import (
    SubCircuitType,  # noqa: F401 - re-exported
)
from kicad_pipeline.optimization.group_helpers import (
    _apply_review_fixes,  # noqa: F401 - re-exported
    _assign_zone_position,  # noqa: F401 - re-exported
    _build_group_map,  # noqa: F401 - re-exported
    _centroid,  # noqa: F401 - re-exported
    _extract_group_bboxes,  # noqa: F401 - re-exported
    _group_footprint_area,  # noqa: F401 - re-exported
    _place_subcircuit_group,  # noqa: F401 - re-exported
)
from kicad_pipeline.optimization.level3_phases import (
    _apply_cross_domain_affinity_overrides,  # noqa: F401 - re-exported
    _apply_template_refinement,  # noqa: F401 - re-exported
    _classify_connector_function,  # noqa: F401 - re-exported
    _orient_connectors,  # noqa: F401 - re-exported
    _pin_connectors_by_function,  # noqa: F401 - re-exported
    _pin_rf_to_edge,  # noqa: F401 - re-exported
    _place_adc_channels,  # noqa: F401 - re-exported
    _place_boundary_regulators,  # noqa: F401 - re-exported
    _place_row_layout,  # noqa: F401 - re-exported
    _pull_mcu_peripherals,  # noqa: F401 - re-exported
)
from kicad_pipeline.optimization.placement_guard import (
    PlacementGuardResult,  # noqa: F401 - re-exported
    validate_placement,  # noqa: F401 - re-exported
)
from kicad_pipeline.optimization.placement_types import (
    OptimizationConfig,  # noqa: F401 - re-exported
    PlacementCandidate,  # noqa: F401 - re-exported
    PlacementContext,
    PlacementResult,
    _apply_positions,  # noqa: F401 - re-exported
    _board_bounds,
    _dict_to_positions,  # noqa: F401 - re-exported
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
    centroid_to_origin,  # noqa: F401 - re-exported
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


def _build_placement_context(
    requirements: ProjectRequirements,
    initial_pcb: PCBDesign,
    max_review_passes: int,
) -> object:
    """Build the shared PlacementContext for the optimizer."""
    from kicad_pipeline.optimization.functional_grouper import detect_subcircuits
    from kicad_pipeline.optimization.placement_types import PlacementContext

    fp_sizes = _fp_courtyard_sizes(initial_pcb)
    bounds = _board_bounds(initial_pcb)
    fixed_refs: set[str] = {
        fp.ref for fp in initial_pcb.footprints
        if _is_fixed(fp.ref, requirements)
    }
    positions: dict[str, tuple[float, float, float]] = {}
    for fp in initial_pcb.footprints:
        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        positions[fp.ref] = (cx, cy, fp.rotation)

    subcircuits = detect_subcircuits(requirements)

    from kicad_pipeline.optimization.constraint_resolver import resolve_constraints
    constraint_set = resolve_constraints(requirements, subcircuits)

    return PlacementContext(
        positions=positions,
        fp_sizes=fp_sizes,
        bounds=bounds,
        fixed_refs=fixed_refs,
        requirements=requirements,
        initial_pcb=initial_pcb,
        zones=[],
        subcircuits=list(subcircuits),
        max_review_passes=max_review_passes,
        constraints=constraint_set,
    )


def _run_level3_phases(ctx: object, **phases: object) -> object:
    """Execute all Level 3 intra-group refinement phases in order.

    Returns relay_leds data for late-phase use.
    """
    phases["subnet_placement"](ctx)  # type: ignore[operator]
    phases["constraint_placement"](ctx)  # type: ignore[operator]
    phases["relay_rows"](ctx)  # type: ignore[operator]
    phases["relay_connector_align"](ctx)  # type: ignore[operator]
    phases["relay_drivers"](ctx)  # type: ignore[operator]
    relay_leds, _relay_led_refs = phases["relay_leds"](ctx)  # type: ignore[operator]
    phases["relay_power_isolation"](ctx)  # type: ignore[operator]
    phases["decoupling"](ctx)  # type: ignore[operator]
    phases["power_group"](ctx)  # type: ignore[operator]
    phases["power_chain_flow"](ctx)  # type: ignore[operator]
    phases["adc_channels"](ctx)  # type: ignore[operator]
    phases["adc_analog_cluster"](ctx)  # type: ignore[operator]
    phases["crystal"](ctx)  # type: ignore[operator]
    phases["rf_edge"](ctx)  # type: ignore[operator]
    phases["connector_orient"](ctx)  # type: ignore[operator]
    phases["top_edge_connectors"](ctx)  # type: ignore[operator]
    phases["all_connectors_to_edges"](ctx)  # type: ignore[operator]
    phases["mcu_group"](ctx)  # type: ignore[operator]
    phases["ethernet_group"](ctx)  # type: ignore[operator]
    phases["template_refinement"](ctx)  # type: ignore[operator]
    phases["late_decoupling"](ctx)  # type: ignore[operator]
    phases["pad_facing"](ctx)  # type: ignore[operator]
    phases["collision"](ctx)  # type: ignore[operator]
    phases["first_clamp"](ctx)  # type: ignore[operator]
    phases["review_loop"](ctx)  # type: ignore[operator]
    return relay_leds


def optimize_placement_ee(
    requirements: ProjectRequirements,
    initial_pcb: PCBDesign,
    max_review_passes: int = 5,
    level3: str = "legacy",
) -> PlacementResult:
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

    Renders 4 mandatory views (2D, 3D top/iso/iso-back) after placement
    and returns them in ``PlacementResult.render_paths``.

    Args:
        requirements: Project requirements with components and nets.
        initial_pcb: Starting PCB with initial placement.
        max_review_passes: Max iterations of the review-fix loop.

    Returns:
        PlacementResult with PCB, review, render paths, and visual findings.
        Supports tuple unpacking: ``pcb, review = optimize_placement_ee(...)``
    """
    from kicad_pipeline.optimization.ee_phases import (
        _phase_adc_analog_cluster,
        _phase_adc_channels,
        _phase_all_connectors_to_edges,
        _phase_build_final,
        _phase_collision_resolution,
        _phase_connector_orientation,
        _phase_constraint_placement,
        _phase_crystal_placement,
        _phase_decoupling,
        _phase_ethernet_group,
        _phase_final_clamp,
        _phase_first_clamp,
        _phase_group_placement,
        _phase_late_adc_realignment,
        _phase_late_decoupling,
        _phase_late_relay_realignment,
        _phase_mcu_decoupling_repull,
        _phase_mcu_group,
        _phase_pad_facing_optimization,
        _phase_power_chain_flow,
        _phase_power_group,
        _phase_relay_connector_alignment,
        _phase_relay_drivers,
        _phase_relay_leds,
        _phase_relay_power_isolation,
        _phase_relay_rows,
        _phase_review_loop,
        _phase_rf_edge,
        _phase_template_refinement,
        _phase_top_edge_connectors,
        _phase_zone_partitioning,
    )
    from kicad_pipeline.optimization.subnet_placer import (
        _phase_subnet_placement,
    )

    ctx = _build_placement_context(requirements, initial_pcb, max_review_passes)

    # Level 1: Zone Partitioning
    _log.info("=== Level 1: Zone Partitioning ===")
    _phase_zone_partitioning(ctx)

    # Level 2: Group Placement (groups as rigid units)
    _log.info("=== Level 2: Group Placement ===")
    _phase_group_placement(ctx)

    # Record which zone each component belongs to after L2 placement.
    # This map is used by collision resolution and late phases to prevent
    # components from being pushed across zone boundaries.
    for zone in ctx.zones:
        zx1, zy1, zx2, zy2 = zone.rect
        for ref, (rx, ry, _rot) in ctx.positions.items():
            if zx1 <= rx <= zx2 and zy1 <= ry <= zy2:
                ctx.zone_membership[ref] = zone.name
    _log.info("  Zone membership recorded: %d refs assigned to zones",
              len(ctx.zone_membership))

    # Level 2.5: Pre-populate subcircuit protection from detected subcircuits
    # This ensures collision resolution in Level 3 can't scatter subcircuit members.
    _log.info("=== Level 2.5: Subcircuit Pre-Protection ===")
    _preprotect_count = 0
    for sc in ctx.subcircuits:
        sc_type = sc.circuit_type.name
        # Only pre-protect ADC/divider subcircuits — they form tight strips
        # that must stay together. Relay drivers, MCU peripherals, and
        # decoupling caps are managed by dedicated Level 3 phases.
        if "ADC" not in sc_type and "DIVIDER" not in sc_type:
            continue
        for ref in sc.refs:
            if ref in ctx.positions:
                ctx.adc_channel_refs.add(ref)
                _preprotect_count += 1
    _log.info("  Pre-protected %d subcircuit refs from collision scatter",
              _preprotect_count)

    # Level 3: Placement
    if level3 == "simple":
        # Simple 3-pass placement — replaces 25 broken phases
        from kicad_pipeline.optimization.placement_simple import run_simple_placement
        _log.info("=== Level 3: Simple 3-Pass Placement ===")
        run_simple_placement(ctx)
        ctx.best_positions = dict(ctx.positions)
        return _phase_build_final(ctx)

    # Collision guard is activated AFTER the main placement phases (3a-3f)
    # complete and before the late refinement phases. This allows the main
    # phases to freely place components, then the guard prevents late phases
    # (review loop, clamp, THT enforcement) from re-creating collisions.
    # The guard is installed by _phase_collision_resolution after it resolves
    # existing collisions.
    _log.info("=== Level 3: Intra-Group Refinement (legacy 25-phase) ===")
    _relay_leds = _run_level3_phases(
        ctx,
        subnet_placement=_phase_subnet_placement,
        constraint_placement=_phase_constraint_placement,
        relay_rows=_phase_relay_rows,
        relay_connector_align=_phase_relay_connector_alignment,
        relay_drivers=_phase_relay_drivers,
        relay_leds=_phase_relay_leds,
        relay_power_isolation=_phase_relay_power_isolation,
        decoupling=_phase_decoupling,
        power_group=_phase_power_group,
        power_chain_flow=_phase_power_chain_flow,
        adc_channels=_phase_adc_channels,
        adc_analog_cluster=_phase_adc_analog_cluster,
        crystal=_phase_crystal_placement,
        rf_edge=_phase_rf_edge,
        connector_orient=_phase_connector_orientation,
        top_edge_connectors=_phase_top_edge_connectors,
        all_connectors_to_edges=_phase_all_connectors_to_edges,
        mcu_group=_phase_mcu_group,
        ethernet_group=_phase_ethernet_group,
        template_refinement=_phase_template_refinement,  # 3h template
        late_decoupling=_phase_late_decoupling,  # 3c-late Late decoupling
        pad_facing=_phase_pad_facing_optimization,  # 3i pad-facing rotation
        collision=_phase_collision_resolution,
        first_clamp=_phase_first_clamp,
        review_loop=_phase_review_loop,
    )

    # Post-review late refinements
    _phase_late_adc_realignment(ctx)
    _phase_late_relay_realignment(ctx, _relay_leds)
    _phase_mcu_decoupling_repull(ctx)
    _phase_final_clamp(ctx)

    # FINAL enforcement: THT connectors MUST be at board edges.
    # Earlier enforcement (in phase 3g) gets undone by the review loop and
    # late phases.  This is the last word — no phase runs after this.
    from kicad_pipeline.optimization.ee_phases_refinement import (
        _enforce_tht_connectors_to_edge,
    )
    _enforce_tht_connectors_to_edge(ctx)

    # FINAL: clamp subcircuit spread — pull outlier components toward their
    # anchor so relay driver subcircuits stay within 25mm spread limit.
    _clamp_subcircuit_spread(ctx)

    # FINAL: push apart any components whose bodies still overlap.
    # NOTE: _phase_build_final reads from ctx.best_positions, not ctx.positions.
    # After all final fixes, sync positions → best_positions.
    # The connector enforcement and collision resolver check courtyard (pads)
    # but miss 3D body collisions — especially connector bodies extending
    # beyond their pads.  Must run AFTER spread clamp, which can pull
    # components back together and re-create overlaps.
    _final_body_collision_fix(ctx)

    # FINAL: re-pull any decoupling caps that drifted during late phases
    # (body collision fix, subcircuit spread clamp, THT enforcement).
    # This is the last decoupling enforcement — nothing runs after it.
    from kicad_pipeline.optimization.ee_phases_refinement import (
        _post_clamp_decoupling_repull,
    )
    _post_clamp_decoupling_repull(ctx)

    # Log collision guard stats
    from kicad_pipeline.optimization.placement_types import CollisionGuardDict
    if isinstance(ctx.positions, CollisionGuardDict):
        _log.info(
            "Collision guard: %d placements rejected (prevented new collisions)",
            ctx.positions.rejected,
        )

    # FINAL collision resolution — the absolute last pass.
    # Late phases (review loop, clamp, body fix, decoupling repull) can
    # re-create collisions.  This final pass resolves them with no fixed
    # refs except connectors and mounting holes.
    final_collisions = _count_collisions(ctx.positions, ctx.fp_sizes)
    if final_collisions:
        _log.info(
            "FINAL: %d collisions after all phases — running final resolution",
            len(final_collisions),
        )
        final_fixed = ctx.fixed_refs | {
            r for r in ctx.positions
            if r.startswith(("J", "K", "H", "MH"))
        }
        ctx.positions = _resolve_collisions(
            dict(ctx.positions), ctx.fp_sizes, ctx.bounds, final_fixed,
        )
        remaining = _count_collisions(ctx.positions, ctx.fp_sizes)
        _log.info(
            "FINAL: resolved to %d collisions",
            len(remaining),
        )

    # Sync final positions to best_positions — _phase_build_final reads best_positions
    ctx.best_positions = dict(ctx.positions)

    return _phase_build_final(ctx)


_BODY_OVERHANG: dict[str, float] = {"J": 3.0, "P": 3.0, "K": 2.0}


def _body_half_extents(
    ref: str, ctx: PlacementContext, rot: float,
) -> tuple[float, float]:
    w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
    if abs(rot % 180 - 90) < 10:
        w, h = h, w
    oh = _BODY_OVERHANG.get(ref.rstrip("0123456789"), 0.0)
    return w / 2.0 + oh, h / 2.0 + oh


def _resolve_body_pair(
    ctx: PlacementContext,
    ref_a: str, ref_b: str,
    ax: float, ay: float, bx: float, by: float,
    a_hw: float, a_hh: float, b_hw: float, b_hh: float,
    min_x: float, min_y: float, max_x: float, max_y: float,
) -> bool:
    gap = 0.5
    overlap_x = (a_hw + b_hw + gap) - abs(ax - bx)
    overlap_y = (a_hh + b_hh + gap) - abs(ay - by)
    if not (overlap_x > 0 and overlap_y > 0):
        return False
    is_a_conn = ref_a.startswith(("J", "P", "H"))
    is_b_conn = ref_b.startswith(("J", "P", "H"))
    if is_a_conn and is_b_conn:
        return False
    mover = ref_b if is_a_conn else ref_a
    if mover.startswith(("J", "P")):
        return False
    mx, my, mrot = ctx.positions[mover]
    if overlap_x < overlap_y:
        push = overlap_x + 0.5
        nx = mx + push if mx > (ax + bx) / 2.0 else mx - push
        nx = max(min_x + 2.0, min(max_x - 2.0, nx))
        from kicad_pipeline.optimization.ee_phases_refinement import _is_within_zone
        if not _is_within_zone(mover, nx, my, ctx):
            return False
        ctx.positions[mover] = (nx, my, mrot)
    else:
        push = overlap_y + 0.5
        ny = my + push if my > (ay + by) / 2.0 else my - push
        ny = max(min_y + 2.0, min(max_y - 2.0, ny))
        from kicad_pipeline.optimization.ee_phases_refinement import _is_within_zone
        if not _is_within_zone(mover, mx, ny, ctx):
            return False
        ctx.positions[mover] = (mx, ny, mrot)
    _log.info("  Final body fix: pushed %s away from %s (overlap %.1fx%.1f)",
               mover, ref_a if mover == ref_b else ref_b, overlap_x, overlap_y)
    return True


def _clamp_subcircuit_spread(ctx: PlacementContext) -> None:
    """Pull outlier subcircuit members toward their anchor.

    After all collision resolution and body fixes, some subcircuit
    components may have been pushed far from their anchor.  This clamp
    pulls them back so the max pairwise spread stays within limits.

    Only moves components that are farthest from the subcircuit centroid,
    and maintains minimum clearance from the anchor body to avoid
    creating new collisions.
    """
    import math
    from kicad_pipeline.optimization.functional_grouper import (
        SubCircuitType,
    )

    _SPREAD_LIMITS: dict[SubCircuitType, float] = {
        SubCircuitType.RELAY_DRIVER: 25.0,
        SubCircuitType.ADC_CHANNEL: 28.0,
    }

    for sc in ctx.subcircuits:
        limit = _SPREAD_LIMITS.get(sc.circuit_type)
        if limit is None:
            continue
        anchor = sc.anchor_ref
        if anchor not in ctx.positions:
            continue
        ax, ay, _ = ctx.positions[anchor]
        aw, ah = ctx.fp_sizes.get(anchor, (2.0, 2.0))
        # Rotation-aware anchor half-extents
        arot = ctx.positions[anchor][2]
        if arot % 180 in (90, 270):
            aw, ah = ah, aw
        # Minimum distance from anchor center to avoid overlap
        min_clearance = max(aw, ah) / 2.0 + 2.0

        members = [r for r in sc.refs if r != anchor and r in ctx.positions]
        if not members:
            continue

        # Protect the anchor and components physically close to it
        # (within min_clearance — moving them would cause courtyard overlap).
        # Components farther away (e.g. LEDs placed in a separate column)
        # can be pulled inward to meet the spread limit.
        immovable: set[str] = {anchor}
        for r in members:
            rx, ry, _ = ctx.positions[r]
            if math.dist((rx, ry), (ax, ay)) < min_clearance + 2.0:
                immovable.add(r)

        movable = [r for r in members if r not in immovable]
        if not movable:
            continue

        for _ in range(5):
            all_refs = [anchor] + members
            all_xy = [(ctx.positions[r][0], ctx.positions[r][1]) for r in all_refs]

            max_dist = 0.0
            pair_i, pair_j = 0, 0
            for i in range(len(all_xy)):
                for j in range(i + 1, len(all_xy)):
                    d = math.dist(all_xy[i], all_xy[j])
                    if d > max_dist:
                        max_dist = d
                        pair_i, pair_j = i, j

            if max_dist <= limit:
                break

            # Find the movable ref farthest from anchor
            far_ref = ""
            far_dist = 0.0
            for r in movable:
                rx, ry, _ = ctx.positions[r]
                d = math.dist((rx, ry), (ax, ay))
                if d > far_dist:
                    far_dist = d
                    far_ref = r

            if not far_ref or far_dist < min_clearance:
                break

            fx, fy, frot = ctx.positions[far_ref]
            # Pull toward anchor but stop at min_clearance
            overshoot = max_dist - limit
            pull_fraction = min(overshoot / far_dist, 0.7)
            new_x = fx + (ax - fx) * pull_fraction
            new_y = fy + (ay - fy) * pull_fraction

            # Ensure we don't get closer than min_clearance to anchor
            new_dist = math.dist((new_x, new_y), (ax, ay))
            if new_dist < min_clearance:
                scale = min_clearance / new_dist if new_dist > 0 else 1.0
                new_x = ax + (new_x - ax) * scale
                new_y = ay + (new_y - ay) * scale

            bx0, by0, bx1, by1 = ctx.bounds
            new_x = max(bx0 + 2.0, min(bx1 - 2.0, new_x))
            new_y = max(by0 + 2.0, min(by1 - 2.0, new_y))
            ctx.positions[far_ref] = (new_x, new_y, frot)
            new_spread = max_dist - overshoot * pull_fraction
            _log.info(
                "  Spread clamp: pulled %s from (%.1f,%.1f) to (%.1f,%.1f) "
                "[%s spread %.1f→~%.1f target %.1f]",
                far_ref, fx, fy, new_x, new_y,
                sc.circuit_type.name, max_dist, new_spread, limit,
            )


def _final_body_collision_fix(ctx: PlacementContext) -> None:
    """Push apart components whose estimated 3D bodies overlap.

    Connectors have bodies extending 3mm+ beyond pads.  After all placement
    phases complete, check for body-to-body overlap and push the smaller
    component away from the larger one.
    """
    min_x, min_y, max_x, max_y = ctx.bounds

    for _pass in range(3):
        moved = False
        refs = sorted(ctx.positions.keys())
        for i, ref_a in enumerate(refs):
            if ref_a.startswith("H"):
                continue
            ax, ay, arot = ctx.positions[ref_a]
            a_hw, a_hh = _body_half_extents(ref_a, ctx, arot)

            for ref_b in refs[i + 1:]:
                if ref_b.startswith("H"):
                    continue
                bx, by, brot = ctx.positions[ref_b]
                b_hw, b_hh = _body_half_extents(ref_b, ctx, brot)
                if _resolve_body_pair(
                    ctx, ref_a, ref_b, ax, ay, bx, by,
                    a_hw, a_hh, b_hw, b_hh, min_x, min_y, max_x, max_y,
                ):
                    moved = True

        if not moved:
            break


# Default optimizer is the EE-grade one
optimize_placement = optimize_placement_ee
