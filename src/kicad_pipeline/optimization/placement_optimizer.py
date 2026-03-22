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
        detect_subcircuits,
    )
    from kicad_pipeline.optimization.ee_phases import (
        _phase_adc_analog_cluster,
        _phase_adc_channels,
        _phase_build_final,
        _phase_collision_resolution,
        _phase_connector_orientation,
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
        _phase_power_group,
        _phase_relay_drivers,
        _phase_relay_leds,
        _phase_relay_rows,
        _phase_review_loop,
        _phase_rf_edge,
        _phase_template_refinement,
        _phase_top_edge_connectors,
        _phase_zone_partitioning,
    )
    from kicad_pipeline.optimization.placement_types import PlacementContext

    fp_sizes = _fp_courtyard_sizes(initial_pcb)
    bounds = _board_bounds(initial_pcb)

    fixed_refs: set[str] = {
        fp.ref for fp in initial_pcb.footprints
        if _is_fixed(fp.ref, requirements)
    }

    # Extract current positions — convert KiCad origin → centroid space.
    positions: dict[str, tuple[float, float, float]] = {}
    for fp in initial_pcb.footprints:
        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        positions[fp.ref] = (cx, cy, fp.rotation)

    subcircuits = detect_subcircuits(requirements)

    # Build the shared mutable context
    ctx = PlacementContext(
        positions=positions,
        fp_sizes=fp_sizes,
        bounds=bounds,
        fixed_refs=fixed_refs,
        requirements=requirements,
        initial_pcb=initial_pcb,
        zones=[],
        subcircuits=list(subcircuits),
        max_review_passes=max_review_passes,
    )

    # ===================================================================
    # Level 1: Zone Partitioning
    # ===================================================================
    _log.info("=== Level 1: Zone Partitioning ===")
    _phase_zone_partitioning(ctx)

    # ===================================================================
    # Level 2: Group Placement (groups as rigid units)
    # ===================================================================
    _log.info("=== Level 2: Group Placement ===")
    _phase_group_placement(ctx)

    # ===================================================================
    # Level 3: Intra-Group Refinement
    # ===================================================================
    _log.info("=== Level 3: Intra-Group Refinement ===")

    # 3a. Relay row formation
    _phase_relay_rows(ctx)

    # 3b. Relay driver subgroup tightening
    _phase_relay_drivers(ctx)

    # 3b2. Relay LED indicator placement
    _relay_leds, _relay_led_refs = _phase_relay_leds(ctx)

    # 3c. Decoupling cap tightening
    _phase_decoupling(ctx)

    # 3c1. Power group organization
    _phase_power_group(ctx)

    # 3c2. ADC channel formation
    _phase_adc_channels(ctx)

    # 3c3. Analog subcircuit clustering
    _phase_adc_analog_cluster(ctx)

    # 3d. Crystal-IC proximity
    _phase_crystal_placement(ctx)

    # 3e. RF edge pinning
    _phase_rf_edge(ctx)

    # 3f. Connector orientation
    _phase_connector_orientation(ctx)

    # 3f2. Top-edge screw terminal ordering
    _phase_top_edge_connectors(ctx)

    # 3c3. MCU peripheral tightening
    _phase_mcu_group(ctx)

    # 3c4. Ethernet group organization
    _phase_ethernet_group(ctx)

    # 3h. Template-guided refinement
    _phase_template_refinement(ctx)

    # 3c-late. Late decoupling re-tightening
    _phase_late_decoupling(ctx)

    # 3g. Collision resolution
    _phase_collision_resolution(ctx)

    # First board-edge clamp + post-clamp collision resolution
    _phase_first_clamp(ctx)

    # EE Review loop
    _phase_review_loop(ctx)

    # 3c2-late: ADC channel re-alignment
    _phase_late_adc_realignment(ctx)

    # 3b-late: Relay driver re-alignment
    _phase_late_relay_realignment(ctx, _relay_leds)

    # MCU decoupling re-pull
    _phase_mcu_decoupling_repull(ctx)

    # Final clamp + crystal overlap fix
    _phase_final_clamp(ctx)

    # Build final PCB, filter stale violations, validate
    return _phase_build_final(ctx)


# Default optimizer is the EE-grade one
optimize_placement = optimize_placement_ee
