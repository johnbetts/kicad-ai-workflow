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

    # Level 3: Intra-Group Refinement
    _log.info("=== Level 3: Intra-Group Refinement ===")
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

    # FINAL: push apart any components whose bodies still overlap.
    # NOTE: _phase_build_final reads from ctx.best_positions, not ctx.positions.
    # After all final fixes, sync positions → best_positions.
    # The connector enforcement and collision resolver check courtyard (pads)
    # but miss 3D body collisions — especially connector bodies extending
    # beyond their pads.
    _final_body_collision_fix(ctx)

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
        ctx.positions[mover] = (max(min_x + 2.0, min(max_x - 2.0, nx)), my, mrot)
    else:
        push = overlap_y + 0.5
        ny = my + push if my > (ay + by) / 2.0 else my - push
        ctx.positions[mover] = (mx, max(min_y + 2.0, min(max_y - 2.0, ny)), mrot)
    _log.info("  Final body fix: pushed %s away from %s (overlap %.1fx%.1f)",
               mover, ref_a if mover == ref_b else ref_b, overlap_x, overlap_y)
    return True


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
