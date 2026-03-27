"""Quality scoring engine for PCB designs.

Computes a multi-dimensional quality score from validation reports, routing
metrics, and placement analysis.  The overall score uses a weighted geometric
mean so that a single zero-dimension drags the composite down hard.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.routing.metrics import BoardRoutingMetrics
    from kicad_pipeline.validation.report import ValidationReport


# ---------------------------------------------------------------------------
# Score weights
# ---------------------------------------------------------------------------

_WEIGHT_ELECTRICAL: float = 0.30
_WEIGHT_MANUFACTURING: float = 0.25
_WEIGHT_PLACEMENT: float = 0.20
_WEIGHT_SIGNAL_INTEGRITY: float = 0.15
_WEIGHT_THERMAL: float = 0.10

# Fast-path sub-dimension weights (EE-aligned, v4 — 14 dimensions)
# Each original weight is scaled by 0.9 to make room for constraint compliance (0.10).
# Original weights summed to 1.0; new 13 x 0.9 + 0.10 = 1.00 exactly.
# Fast-path sub-dimension weights (EE-aligned, v6 — 18 dimensions)
# Added utilization + compactness + signal flow to catch sprawling/disordered layouts.
_FAST_WEIGHT_COLLISION: float = 0.12
_FAST_WEIGHT_SUBCIRCUIT_COHESION: float = 0.04
_FAST_WEIGHT_VOLTAGE_ISOLATION: float = 0.12
_FAST_WEIGHT_CONNECTOR_EDGE: float = 0.08
_FAST_WEIGHT_DECOUPLING_PROXIMITY: float = 0.08
_FAST_WEIGHT_MCU_PERIPHERAL: float = 0.08
_FAST_WEIGHT_RF_EDGE: float = 0.04
_FAST_WEIGHT_CONNECTOR_ORIENTATION: float = 0.04
_FAST_WEIGHT_REGULATOR_BOUNDARY: float = 0.04
_FAST_WEIGHT_GROUP_COHESION: float = 0.04
_FAST_WEIGHT_SUBGROUP_COHESION: float = 0.04
_FAST_WEIGHT_GROUP_ISOLATION: float = 0.04
_FAST_WEIGHT_PAD_FACING: float = 0.04
_FAST_WEIGHT_CONSTRAINT_COMPLIANCE: float = 0.04
_FAST_WEIGHT_HUMAN_FEEDBACK: float = 0.04
_FAST_WEIGHT_UTILIZATION: float = 0.04
_FAST_WEIGHT_COMPACTNESS: float = 0.04
_FAST_WEIGHT_SIGNAL_FLOW: float = 0.04

# Legacy weight names for backward compatibility
_FAST_WEIGHT_NET_PROXIMITY: float = _FAST_WEIGHT_SUBCIRCUIT_COHESION
_FAST_WEIGHT_PASSIVE_PROXIMITY: float = _FAST_WEIGHT_DECOUPLING_PROXIMITY
_FAST_WEIGHT_BLOCK_COHESION: float = _FAST_WEIGHT_SUBCIRCUIT_COHESION
_FAST_WEIGHT_BOUNDARY: float = 0.05  # used in breakdown display

# Grade thresholds
_GRADE_A: float = 0.9
_GRADE_B: float = 0.75
_GRADE_C: float = 0.6
_GRADE_D: float = 0.4

# Floor to prevent zero in geometric mean
_SCORE_FLOOR: float = 0.01

# Placement: reference distance for normalisation (mm)
_PLACEMENT_IDEAL_DISTANCE_MM: float = 15.0

# Net proximity: maximum useful distance (mm) for normalisation
_NET_PROXIMITY_MAX_MM: float = 50.0

# Collision: per-collision penalty — gradual so each fix is visible
_COLLISION_PENALTY: float = 0.05


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreDetail:
    """Per-category score breakdown.

    Attributes:
        category: Human-readable category name.
        score: Normalised score in [0.0, 1.0].
        weight: Weight used in the overall composite.
        issues: Descriptive issue strings (worst first).
    """

    category: str
    score: float
    weight: float
    issues: tuple[str, ...]


@dataclass(frozen=True)
class QualityScore:
    """Composite quality score for a PCB design.

    Attributes:
        board_cost: Routing cost metric (lower is better), 0.0 if unavailable.
        electrical_score: Electrical / DRC score [0, 1].
        manufacturing_score: Manufacturing constraint score [0, 1].
        thermal_score: Thermal analysis score [0, 1].
        signal_integrity_score: Signal integrity score [0, 1].
        placement_score: Placement quality score [0, 1].
        overall_score: Weighted geometric mean of the above [0, 1].
        grade: Letter grade (A / B / C / D / F).
        breakdown: Per-category detail entries.
    """

    board_cost: float
    electrical_score: float
    manufacturing_score: float
    thermal_score: float
    signal_integrity_score: float
    placement_score: float
    overall_score: float
    grade: str
    breakdown: tuple[ScoreDetail, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def score_to_grade(score: float) -> str:
    """Map a normalised score to a letter grade.

    Args:
        score: Value in [0, 1].

    Returns:
        One of ``'A'``, ``'B'``, ``'C'``, ``'D'``, ``'F'``.
    """
    if score >= _GRADE_A:
        return "A"
    if score >= _GRADE_B:
        return "B"
    if score >= _GRADE_C:
        return "C"
    if score >= _GRADE_D:
        return "D"
    return "F"


def _clamp01(value: float) -> float:
    """Clamp *value* to the range [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def _weighted_geometric_mean(
    scores: tuple[tuple[float, float], ...],
) -> float:
    """Compute the weighted geometric mean of ``(score, weight)`` pairs.

    Each score is floored at :data:`_SCORE_FLOOR` to avoid zeroing out the
    entire composite.
    """
    total_weight = sum(w for _, w in scores)
    if total_weight <= 0.0:
        return 0.0
    log_sum = 0.0
    for s, w in scores:
        log_sum += w * math.log(max(s, _SCORE_FLOOR))
    return math.exp(log_sum / total_weight)


def _score_from_violations(
    error_count: int,
    warning_count: int,
    error_penalty: float,
    warning_penalty: float,
) -> float:
    """Compute a [0, 1] score from violation counts."""
    return _clamp01(1.0 - (error_count * error_penalty + warning_count * warning_penalty))


def _compute_placement_score_from_pcb(pcb: PCBDesign) -> tuple[float, tuple[str, ...]]:
    """Derive a placement score from footprint positions.

    Evaluates passive-to-IC proximity and returns ``(score, issues)``.
    """
    from kicad_pipeline.routing.metrics import compute_passive_proximity

    avg_dist = compute_passive_proximity(list(pcb.footprints))
    issues: list[str] = []

    if avg_dist <= 0.0:
        # No passives or no ICs — neutral score
        return 1.0, ()

    # Normalise: ideal <= _PLACEMENT_IDEAL_DISTANCE_MM → 1.0, worse → lower
    ratio = (avg_dist - _PLACEMENT_IDEAL_DISTANCE_MM) / _PLACEMENT_IDEAL_DISTANCE_MM
    score = _clamp01(1.0 - ratio)
    if score < 0.7:
        issues.append(
            f"Average passive-to-IC distance is {avg_dist:.1f} mm "
            f"(ideal < {_PLACEMENT_IDEAL_DISTANCE_MM:.0f} mm)"
        )
    return score, tuple(issues)


# ---------------------------------------------------------------------------
# Fast-path placement scoring (for SA optimizer loop)
# ---------------------------------------------------------------------------


def _fp_position_dict(pcb: PCBDesign) -> dict[str, tuple[float, float]]:
    """Build ref → (x, y) centroid lookup from PCB footprints.

    Converts KiCad origin-based positions to pad-centroid positions
    for accurate distance and boundary measurements.  Uses
    pin_map.origin_to_centroid() as the single source of truth.
    """
    from kicad_pipeline.pcb.pin_map import origin_to_centroid

    return {
        fp.ref: origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        for fp in pcb.footprints
    }


def _fp_size_dict(pcb: PCBDesign) -> dict[str, tuple[float, float]]:
    """Build ref → (width, height) lookup from PCB footprints.

    Delegates to :func:`~kicad_pipeline.pcb.footprints.estimate_courtyard_mm`
    which accounts for component body extension beyond the pad field.
    """
    from kicad_pipeline.pcb.footprints import estimate_courtyard_mm

    sizes: dict[str, tuple[float, float]] = {}
    for fp in pcb.footprints:
        sizes[fp.ref] = estimate_courtyard_mm(fp)
    return sizes


def _score_collisions(
    pcb: PCBDesign,
) -> tuple[float, list[str]]:
    """Score based on courtyard collision count.

    Uses centroid positions (not KiCad origin) to match the optimizer's
    coordinate space.  The optimizer resolves collisions in centroid space,
    so scoring must check the same positions.

    Returns (score, issues) where score = 1.0 - penalty_per_collision * count.
    """
    from kicad_pipeline.pcb.pin_map import origin_to_centroid

    sizes = _fp_size_dict(pcb)

    # Build centroid-based position dict (matches optimizer coordinate space)
    centroid_positions: dict[str, tuple[float, float]] = {}
    rotations: dict[str, float] = {}
    for fp in pcb.footprints:
        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        centroid_positions[fp.ref] = (cx, cy)
        rotations[fp.ref] = fp.rotation

    collisions: list[str] = []
    refs = list(centroid_positions.keys())

    # Pre-build flat arrays with rotation-adjusted sizes for O(n^2) inner loop
    fp_data: list[tuple[str, float, float, float, float]] = []
    for ref in refs:
        x, y = centroid_positions[ref]
        w, h = sizes.get(ref, (2.0, 2.0))
        rot = rotations.get(ref, 0.0)
        if rot % 180 in (90.0, 270.0):
            w, h = h, w
        fp_data.append((ref, x, y, w, h))

    for i, (ref_a, xa, ya, wa, ha) in enumerate(fp_data):
        for ref_b, xb, yb, wb, hb in fp_data[i + 1:]:
            # AABB overlap check (center-based)
            dx = abs(xa - xb)
            dy = abs(ya - yb)
            gap_x = (wa + wb) / 2.0
            gap_y = (ha + hb) / 2.0
            if dx < gap_x and dy < gap_y:
                collisions.append(f"Collision: {ref_a} overlaps {ref_b}")

    score = _clamp01(1.0 - len(collisions) * _COLLISION_PENALTY)
    return score, collisions


def _score_net_proximity(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, list[str]]:
    """Score based on distances between signal-connected components.

    Components sharing signal nets should be close together.
    """
    from kicad_pipeline.pcb.constraints import build_signal_adjacency

    adj = build_signal_adjacency(requirements)
    pos = _fp_position_dict(pcb)

    total_dist = 0.0
    pair_count = 0
    issues: list[str] = []

    seen: set[tuple[str, str]] = set()
    for ref_a, neighbours in adj.items():
        if ref_a not in pos:
            continue
        xa, ya = pos[ref_a]
        for ref_b in neighbours:
            pair = (min(ref_a, ref_b), max(ref_a, ref_b))
            if pair in seen or ref_b not in pos:
                continue
            seen.add(pair)
            xb, yb = pos[ref_b]
            dist = math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
            total_dist += dist
            pair_count += 1
            if dist > _NET_PROXIMITY_MAX_MM:
                issues.append(
                    f"{ref_a}-{ref_b} signal distance {dist:.1f}mm "
                    f"(max {_NET_PROXIMITY_MAX_MM:.0f}mm)"
                )

    if pair_count == 0:
        return 1.0, []

    avg_dist = total_dist / pair_count
    score = _clamp01(1.0 - avg_dist / _NET_PROXIMITY_MAX_MM)
    return score, issues


def _score_block_cohesion(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, list[str]]:
    """Score based on how tightly feature-block components cluster.

    For each feature block, compute the bounding box of its components
    and normalise against the board area.
    """
    if not requirements.features:
        return 1.0, []

    pos = _fp_position_dict(pcb)

    # Board area for normalisation
    if pcb.outline.polygon:
        bxs = [p.x for p in pcb.outline.polygon]
        bys = [p.y for p in pcb.outline.polygon]
        board_diag = math.sqrt(
            (max(bxs) - min(bxs)) ** 2 + (max(bys) - min(bys)) ** 2
        )
    else:
        board_diag = 100.0

    block_scores: list[float] = []
    issues: list[str] = []

    for block in requirements.features:
        block_positions = [
            pos[ref] for ref in block.components if ref in pos
        ]
        if len(block_positions) < 2:
            block_scores.append(1.0)
            continue

        xs = [p[0] for p in block_positions]
        ys = [p[1] for p in block_positions]
        spread = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)

        # Score: compact cluster relative to board diagonal
        # Ideal: spread <= 25% of board diagonal
        ratio = spread / board_diag if board_diag > 0 else 0.0
        s = _clamp01(1.0 - max(0.0, ratio - 0.25) / 0.75)
        block_scores.append(s)

        if s < 0.6:
            issues.append(
                f"Block '{block.name}' spread {spread:.1f}mm "
                f"({ratio:.0%} of board diagonal)"
            )

    score = sum(block_scores) / len(block_scores) if block_scores else 1.0
    return score, issues


# Group cohesion thresholds (mm)
_GROUP_SPREAD_SMALL_THRESHOLD: float = 30.0  # groups with <= 5 components
_GROUP_SPREAD_LARGE_THRESHOLD: float = 60.0  # groups with > 5 components


def _score_group_cohesion(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, list[str]]:
    """Score based on max spread within each FeatureBlock group.

    For each FeatureBlock, compute the maximum distance between any two
    members. Score 1.0 if within threshold, penalize proportionally beyond.
    Small groups (<=5 components) have tighter threshold (15mm) than
    large groups (25mm).
    """
    if not requirements.features:
        return 1.0, []

    pos = _fp_position_dict(pcb)
    group_scores: list[float] = []
    issues: list[str] = []

    for block in requirements.features:
        block_positions = [
            pos[ref] for ref in block.components if ref in pos
        ]
        if len(block_positions) < 2:
            group_scores.append(1.0)
            continue

        # Compute max pairwise distance via bounding box diagonal (O(n) vs O(n^2))
        xs = [p[0] for p in block_positions]
        ys = [p[1] for p in block_positions]
        max_dist = math.sqrt(
            (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2
        )

        threshold = (
            _GROUP_SPREAD_SMALL_THRESHOLD
            if len(block_positions) <= 5
            else _GROUP_SPREAD_LARGE_THRESHOLD
        )

        if max_dist <= threshold:
            group_scores.append(1.0)
        else:
            # Penalize proportionally: at 2x threshold score = 0.0
            overshoot = (max_dist - threshold) / threshold
            s = _clamp01(1.0 - overshoot)
            group_scores.append(s)
            if s < 0.8:
                issues.append(
                    f"Group '{block.name}' spread {max_dist:.1f}mm "
                    f"(threshold {threshold:.0f}mm)"
                )

    score = sum(group_scores) / len(group_scores) if group_scores else 1.0
    return score, issues


def _score_boundary(pcb: PCBDesign) -> tuple[float, list[str]]:
    """Score based on components staying within board boundary.

    Any component outside the board outline gets a penalty.
    """
    if not pcb.outline.polygon:
        return 1.0, []

    bxs = [p.x for p in pcb.outline.polygon]
    bys = [p.y for p in pcb.outline.polygon]
    min_x, max_x = min(bxs), max(bxs)
    min_y, max_y = min(bys), max(bys)

    out_count = 0
    issues: list[str] = []
    margin = 1.0  # 1mm margin

    pos_dict = _fp_position_dict(pcb)
    for fp in pcb.footprints:
        x, y = pos_dict.get(fp.ref, (fp.position.x, fp.position.y))
        if x < min_x - margin or x > max_x + margin or \
           y < min_y - margin or y > max_y + margin:
            out_count += 1
            issues.append(f"{fp.ref} outside board boundary")

    score = _clamp01(1.0 - out_count * 0.2)
    return score, issues


def _score_subcircuit_cohesion(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, list[str]]:
    """Score based on sub-circuit component clustering.

    Uses the functional grouper to detect sub-circuits and measure how
    tightly each group's components are clustered around their anchor.
    """
    from kicad_pipeline.optimization.functional_grouper import detect_subcircuits
    from kicad_pipeline.optimization.review_agent import SUBCIRCUIT_MAX_SPREAD_MM

    subcircuits = detect_subcircuits(requirements)
    if not subcircuits:
        # Fall back to block cohesion
        return _score_block_cohesion(pcb, requirements)

    pos = _fp_position_dict(pcb)
    scores: list[float] = []
    issues: list[str] = []

    for sc in subcircuits:
        anchor_pos = pos.get(sc.anchor_ref)
        if anchor_pos is None:
            continue
        max_dist = 0.0
        for ref in sc.refs:
            if ref == sc.anchor_ref or ref not in pos:
                continue
            d = math.sqrt(
                (pos[ref][0] - anchor_pos[0]) ** 2 +
                (pos[ref][1] - anchor_pos[1]) ** 2
            )
            max_dist = max(max_dist, d)
        ratio = max_dist / SUBCIRCUIT_MAX_SPREAD_MM if SUBCIRCUIT_MAX_SPREAD_MM > 0 else 0.0
        s = _clamp01(1.0 - max(0.0, ratio - 1.0))
        scores.append(s)
        if s < 0.7:
            issues.append(
                f"{sc.circuit_type.value} ({sc.anchor_ref}) spread {max_dist:.1f}mm"
            )

    score = sum(scores) / len(scores) if scores else 1.0
    return score, issues


def _score_voltage_isolation(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, list[str]]:
    """Score based on voltage domain separation.

    Components in different voltage domains should maintain minimum distance.
    """
    from kicad_pipeline.optimization.functional_grouper import (
        VoltageDomain,
        classify_voltage_domains,
    )
    from kicad_pipeline.optimization.review_agent import VOLTAGE_DOMAIN_MIN_GAP_MM

    domain_map = classify_voltage_domains(requirements)
    pos = _fp_position_dict(pcb)

    # Group refs by domain (skip MIXED)
    domain_refs: dict[VoltageDomain, list[str]] = {}
    for ref, domain in domain_map.items():
        if domain == VoltageDomain.MIXED:
            continue
        if ref in pos:
            domain_refs.setdefault(domain, []).append(ref)

    violations = 0
    total_checks = 0
    issues: list[str] = []
    domains = list(domain_refs.keys())

    # Pre-build position arrays per domain (avoids repeated dict lookups)
    domain_positions: dict[VoltageDomain, list[tuple[float, float]]] = {
        d: [pos[r] for r in refs[:10]]
        for d, refs in domain_refs.items()
    }

    for i, d1 in enumerate(domains):
        for d2 in domains[i + 1:]:
            # Sample: check closest pair per domain pair using pre-built arrays
            min_dist = float("inf")
            for x1, y1 in domain_positions[d1]:
                for x2, y2 in domain_positions[d2]:
                    d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    min_dist = min(min_dist, d)
            total_checks += 1
            if min_dist < VOLTAGE_DOMAIN_MIN_GAP_MM:
                violations += 1
                issues.append(
                    f"{d1.value} vs {d2.value}: {min_dist:.1f}mm "
                    f"(min {VOLTAGE_DOMAIN_MIN_GAP_MM}mm)"
                )

    score = _clamp01(1.0 - violations * 0.2) if total_checks > 0 else 1.0
    return score, issues


def _score_connector_edge(
    pcb: PCBDesign,
) -> tuple[float, list[str]]:
    """Score based on connector proximity to board edges."""
    from kicad_pipeline.optimization.review_agent import CONNECTOR_EDGE_MAX_MM

    connectors = [fp for fp in pcb.footprints if fp.ref.startswith("J")]
    if not connectors:
        return 1.0, []

    if not pcb.outline or not pcb.outline.polygon:
        return 1.0, []

    bxs = [p.x for p in pcb.outline.polygon]
    bys = [p.y for p in pcb.outline.polygon]
    min_x, max_x = min(bxs), max(bxs)
    min_y, max_y = min(bys), max(bys)

    scores: list[float] = []
    issues: list[str] = []

    for fp in connectors:
        edge_dist = min(
            fp.position.x - min_x,
            max_x - fp.position.x,
            fp.position.y - min_y,
            max_y - fp.position.y,
        )
        if edge_dist <= CONNECTOR_EDGE_MAX_MM:
            scores.append(1.0)
        else:
            ratio = edge_dist / CONNECTOR_EDGE_MAX_MM
            s = _clamp01(1.0 - (ratio - 1.0) * 0.3)
            scores.append(s)
            issues.append(f"{fp.ref} is {edge_dist:.1f}mm from edge")

    score = sum(scores) / len(scores) if scores else 1.0
    return score, issues


def _score_decoupling_proximity(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, list[str]]:
    """Score decoupling cap proximity to ICs."""
    from kicad_pipeline.optimization.functional_grouper import (
        SubCircuitType,
        detect_subcircuits,
    )

    subcircuits = detect_subcircuits(requirements)
    decoupling = [s for s in subcircuits if s.circuit_type == SubCircuitType.DECOUPLING]
    if not decoupling:
        return 1.0, []

    pos = _fp_position_dict(pcb)
    scores: list[float] = []
    issues: list[str] = []
    threshold = 3.0  # mm

    sizes = _fp_size_dict(pcb)

    for sc in decoupling:
        ic_pos = pos.get(sc.anchor_ref)
        if ic_pos is None:
            continue
        ic_size = sizes.get(sc.anchor_ref, (3.0, 3.0))
        for ref in sc.refs:
            if ref == sc.anchor_ref or ref not in pos:
                continue
            cap_size = sizes.get(ref, (2.0, 1.0))
            # Edge-to-edge distance (gap between bounding boxes)
            dx = abs(pos[ref][0] - ic_pos[0]) - (ic_size[0] + cap_size[0]) / 2.0
            dy = abs(pos[ref][1] - ic_pos[1]) - (ic_size[1] + cap_size[1]) / 2.0
            if dx <= 0 and dy <= 0:
                d = 0.0
            elif dx <= 0:
                d = dy
            elif dy <= 0:
                d = dx
            else:
                d = math.sqrt(dx * dx + dy * dy)
            if d <= threshold:
                scores.append(1.0)
            else:
                s = _clamp01(1.0 - (d - threshold) / threshold)
                scores.append(s)
                issues.append(f"{ref} is {d:.1f}mm from {sc.anchor_ref} edge")

    score = sum(scores) / len(scores) if scores else 1.0
    return score, issues


def _score_mcu_peripheral_proximity(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, list[str]]:
    """Score MCU peripheral proximity.

    Evaluates how close switches, LEDs, and debug connectors are to their MCU.
    """
    from kicad_pipeline.constants import MCU_PERIPHERAL_MAX_DISTANCE_MM
    from kicad_pipeline.optimization.functional_grouper import (
        SubCircuitType,
        detect_subcircuits,
    )

    subcircuits = detect_subcircuits(requirements)
    mcu_clusters = [s for s in subcircuits
                    if s.circuit_type == SubCircuitType.MCU_PERIPHERAL_CLUSTER]
    if not mcu_clusters:
        return 1.0, []

    pos = _fp_position_dict(pcb)
    scores: list[float] = []
    issues: list[str] = []
    threshold = MCU_PERIPHERAL_MAX_DISTANCE_MM

    for sc in mcu_clusters:
        anchor_pos = pos.get(sc.anchor_ref)
        if anchor_pos is None:
            continue
        for ref in sc.refs:
            if ref == sc.anchor_ref or ref not in pos:
                continue
            d = math.sqrt(
                (pos[ref][0] - anchor_pos[0]) ** 2 +
                (pos[ref][1] - anchor_pos[1]) ** 2,
            )
            if d <= threshold:
                scores.append(1.0)
            else:
                s = _clamp01(1.0 - (d - threshold) / threshold)
                scores.append(s)
                issues.append(f"{ref} is {d:.1f}mm from MCU {sc.anchor_ref}")

    score = sum(scores) / len(scores) if scores else 1.0
    return score, issues


def _score_rf_edge_placement(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, list[str]]:
    """Score RF module edge placement."""
    from kicad_pipeline.constants import RF_EDGE_MAX_MM
    from kicad_pipeline.optimization.functional_grouper import (
        SubCircuitType,
        detect_subcircuits,
    )

    subcircuits = detect_subcircuits(requirements)
    rf_modules = [s for s in subcircuits if s.circuit_type == SubCircuitType.RF_ANTENNA]
    if not rf_modules:
        return 1.0, []

    if not pcb.outline or not pcb.outline.polygon:
        return 1.0, []

    bxs = [p.x for p in pcb.outline.polygon]
    bys = [p.y for p in pcb.outline.polygon]
    min_x, max_x = min(bxs), max(bxs)
    min_y, max_y = min(bys), max(bys)

    scores: list[float] = []
    issues: list[str] = []

    fp_sizes = _fp_size_dict(pcb)
    for sc in rf_modules:
        pos = _fp_position_dict(pcb).get(sc.anchor_ref)
        if pos is None:
            continue
        x, y = pos
        # Use the NEAREST EDGE of the module body, not centroid,
        # since the antenna is at the module's outer edge.
        w, h = fp_sizes.get(sc.anchor_ref, (2.0, 2.0))
        edge_dist = min(
            x - w / 2.0 - min_x,
            max_x - x - w / 2.0,
            y - h / 2.0 - min_y,
            max_y - y - h / 2.0,
        )
        if edge_dist <= RF_EDGE_MAX_MM:
            scores.append(1.0)
        else:
            s = _clamp01(1.0 - (edge_dist - RF_EDGE_MAX_MM) / RF_EDGE_MAX_MM)
            scores.append(s)
            issues.append(f"RF {sc.anchor_ref} is {edge_dist:.1f}mm from edge")

    score = sum(scores) / len(scores) if scores else 1.0
    return score, issues


def _score_subgroup_cohesion(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, tuple[str, ...]]:
    """Score subgroup cohesion -- relay driver, ADC channel, decoupling groups.

    Measures whether components that form functional subgroups (e.g. each
    relay driver's Q+D+R, each ADC channel's resistor ladder) are kept
    tightly together.

    Returns (score, issues) where score is 0-1.
    """
    from kicad_pipeline.optimization.functional_grouper import (
        SubCircuitType,
        detect_subcircuits,
    )

    subcircuits = detect_subcircuits(requirements)
    fp_pos = _fp_position_dict(pcb)

    issues: list[str] = []
    scores: list[float] = []

    # Subgroup types and their max spread thresholds
    # Thresholds account for anchor component sizes:
    #   RELAY_DRIVER: relay footprint ~16-18mm, support must be adjacent → 22mm
    #   DECOUPLING: multiple caps around one IC → 10mm
    #   BUCK_CONVERTER: IC + inductor + caps in chain → 18mm
    thresholds: dict[SubCircuitType, float] = {
        SubCircuitType.RELAY_DRIVER: 22.0,
        SubCircuitType.ADC_CHANNEL: 12.0,
        SubCircuitType.DECOUPLING: 10.0,
        SubCircuitType.BUCK_CONVERTER: 18.0,
        SubCircuitType.CRYSTAL_OSC: 10.0,
    }

    for sc in subcircuits:
        threshold = thresholds.get(sc.circuit_type)
        if threshold is None:
            continue

        positions = [fp_pos[r] for r in sc.refs if r in fp_pos]
        if len(positions) < 2:
            continue

        # Compute spread via bounding box diagonal (O(n) vs O(n^2))
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        max_dist = math.sqrt(
            (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2
        )

        if max_dist <= threshold:
            scores.append(1.0)
        else:
            ratio = threshold / max(max_dist, 0.01)
            scores.append(_clamp01(ratio))
            issues.append(
                f"{sc.circuit_type.value} ({', '.join(sc.refs[:3])}): "
                f"spread {max_dist:.1f}mm > {threshold:.0f}mm"
            )

    if not scores:
        return (1.0, ())

    return (sum(scores) / len(scores), tuple(issues))


def _score_group_isolation(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, tuple[str, ...]]:
    """Score inter-group isolation -- minimum gap between group bounding boxes.

    Score 1.0 if all inter-group gaps >= 10mm, penalize proportionally
    for overlap or insufficient gap.

    Returns (score, issues).
    """
    if not requirements.features or len(requirements.features) < 2:
        return (1.0, ())

    fp_pos = _fp_position_dict(pcb)

    # Compute bounding boxes for each group
    group_bboxes: list[tuple[str, float, float, float, float]] = []
    for block in requirements.features:
        positions = [fp_pos[r] for r in block.components if r in fp_pos]
        if len(positions) < 2:
            continue
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        group_bboxes.append((block.name, min(xs) - 1, min(ys) - 1, max(xs) + 1, max(ys) + 1))

    if len(group_bboxes) < 2:
        return (1.0, ())

    target_gap = 10.0  # mm
    issues: list[str] = []
    scores: list[float] = []

    for i, (name_a, ax1, ay1, ax2, ay2) in enumerate(group_bboxes):
        for name_b, bx1, by1, bx2, by2 in group_bboxes[i + 1:]:
            # Compute minimum gap between two bounding boxes
            dx = max(0.0, max(ax1 - bx2, bx1 - ax2))
            dy = max(0.0, max(ay1 - by2, by1 - ay2))

            if dx == 0.0 and dy == 0.0:
                # Overlapping
                overlap_x = min(ax2, bx2) - max(ax1, bx1)
                overlap_y = min(ay2, by2) - max(ay1, by1)
                gap = -min(overlap_x, overlap_y)
                scores.append(0.0)
                issues.append(f"{name_a} overlaps {name_b} by {abs(gap):.1f}mm")
            else:
                gap = math.sqrt(dx * dx + dy * dy)
                if gap >= target_gap:
                    scores.append(1.0)
                else:
                    scores.append(_clamp01(gap / target_gap))
                    issues.append(
                        f"{name_a} <-> {name_b}: gap {gap:.1f}mm < {target_gap:.0f}mm"
                    )

    if not scores:
        return (1.0, ())

    return (sum(scores) / len(scores), tuple(issues))


def _build_signal_net_connections(
    pcb: PCBDesign,
    power_nets: set[str],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, object]]:
    """Build net-to-pad connections and footprint lookup, excluding power nets.

    Returns (net_connections, fp_map).
    """
    net_connections: dict[str, list[tuple[str, str]]] = {}
    fp_map: dict[str, object] = {fp.ref: fp for fp in pcb.footprints}
    for fp in pcb.footprints:
        for pad in fp.pads:
            if pad.net_name and pad.net_name.upper() not in power_nets:
                net_connections.setdefault(pad.net_name, []).append(
                    (fp.ref, pad.number)
                )
    return net_connections, fp_map


def _score_pad_pair(
    fp_a: object, fp_b: object,
    side_a: object, side_b: object,
    side_vectors: dict[object, tuple[float, float]],
) -> float:
    """Score a single pad pair's facing alignment. Returns score in [0, 1]."""
    ax = fp_a.position.x  # type: ignore[union-attr]
    ay = fp_a.position.y  # type: ignore[union-attr]
    bx = fp_b.position.x  # type: ignore[union-attr]
    by = fp_b.position.y  # type: ignore[union-attr]
    dx, dy = bx - ax, by - ay
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.1:
        return 1.0
    dx /= dist
    dy /= dist

    va = side_vectors[side_a]
    dot_a = va[0] * dx + va[1] * dy
    vb = side_vectors[side_b]
    dot_b = vb[0] * (-dx) + vb[1] * (-dy)
    score_a = (dot_a + 1.0) / 2.0
    score_b = (dot_b + 1.0) / 2.0
    return (score_a + score_b) / 2.0


def _score_pad_facing(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, tuple[str, ...]]:
    """Score how well connected pads face each other.

    For each signal net connecting two pads on different footprints:
    1. Compute which CardinalSide each pad is on (relative to its footprint).
    2. Compute direction vector from footprint A center to footprint B center.
    3. Score: pad on correct side (facing partner) = 1.0, opposite = 0.0,
       perpendicular = 0.5.

    Returns (score, issues).
    """
    from kicad_pipeline.pcb.pin_map import CardinalSide, compute_pin_map
    from kicad_pipeline.visualization.ratsnest import POWER_NETS

    net_connections, fp_map = _build_signal_net_connections(pcb, POWER_NETS)

    side_vectors: dict[CardinalSide, tuple[float, float]] = {
        CardinalSide.NORTH: (0.0, -1.0),
        CardinalSide.SOUTH: (0.0, 1.0),
        CardinalSide.EAST: (1.0, 0.0),
        CardinalSide.WEST: (-1.0, 0.0),
        CardinalSide.CENTER: (0.0, 0.0),
    }

    scores: list[float] = []
    issues: list[str] = []
    pin_maps: dict[str, object] = {}

    for _net_name, connections in net_connections.items():
        if len(connections) != 2:
            continue
        ref_a, pad_a = connections[0]
        ref_b, pad_b = connections[1]
        if ref_a == ref_b:
            continue

        fp_a = fp_map.get(ref_a)
        fp_b = fp_map.get(ref_b)
        if fp_a is None or fp_b is None:
            continue

        if ref_a not in pin_maps:
            pin_maps[ref_a] = compute_pin_map(fp_a, fp_a.rotation)  # type: ignore[arg-type]
        if ref_b not in pin_maps:
            pin_maps[ref_b] = compute_pin_map(fp_b, fp_b.rotation)  # type: ignore[arg-type]

        side_a = pin_maps[ref_a].side_for_pad(pad_a)  # type: ignore[union-attr]
        side_b = pin_maps[ref_b].side_for_pad(pad_b)  # type: ignore[union-attr]

        if side_a is None or side_b is None:
            continue
        if side_a == CardinalSide.CENTER or side_b == CardinalSide.CENTER:
            scores.append(1.0)
            continue

        pair_score = _score_pad_pair(fp_a, fp_b, side_a, side_b, side_vectors)
        scores.append(pair_score)

        if pair_score < 0.4:
            issues.append(
                f"{ref_a}.{pad_a}({side_a.value}) -> "
                f"{ref_b}.{pad_b}({side_b.value}): "
                f"facing score {pair_score:.2f}"
            )

    if not scores:
        return (1.0, ())

    return (_clamp01(sum(scores) / len(scores)), tuple(issues[:10]))


def _score_constraint_compliance(
    positions: dict[str, tuple[float, float]],
    requirements: ProjectRequirements | None,
) -> tuple[float, list[str]]:
    """Score constraint compliance. Returns (score, issues).

    Checks proximity and ordering constraints resolved from requirements.
    Returns 1.0 if there are no constraints or requirements is None.
    """
    if requirements is None:
        return 1.0, []
    try:
        from kicad_pipeline.optimization.constraint_resolver import resolve_constraints

        constraints = resolve_constraints(requirements)
    except Exception:
        return 1.0, []

    if not constraints.proximity and not constraints.ordering and not constraints.groups:
        return 1.0, []

    total = len(constraints.proximity) + len(constraints.ordering) + len(constraints.groups)
    violations = 0
    issues: list[str] = []

    # Check proximity constraints
    for prox in constraints.proximity:
        if prox.ref not in positions or prox.target_ref not in positions:
            continue
        rx, ry = positions[prox.ref]
        tx, ty = positions[prox.target_ref]
        dist = math.sqrt((rx - tx) ** 2 + (ry - ty) ** 2)
        if dist > prox.max_distance_mm:
            violations += 1
            issues.append(
                f"{prox.ref} {dist:.0f}mm from {prox.target_ref} "
                f"(max {prox.max_distance_mm:.0f})"
            )

    # Check ordering constraints
    for chain in constraints.ordering:
        present = [r for r in chain.refs if r in positions]
        if len(present) < 2:
            continue
        xs = [positions[r][0] for r in present]
        ys = [positions[r][1] for r in present]
        use_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))
        vals = [positions[r][0 if use_x else 1] for r in present]
        out_of_order = sum(
            1 for i in range(len(vals) - 1) if vals[i] > vals[i + 1] + 1.0
        )
        if out_of_order > 0:
            violations += 1
            issues.append(
                f"Chain '{chain.group}' has {out_of_order} ordering violation(s)"
            )

    if total == 0:
        return 1.0, []
    score = max(0.0, 1.0 - violations / total)
    return score, issues[:5]


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Human feedback scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HumanConstraint:
    """A single scored constraint from human feedback.

    Attributes:
        constraint_type: One of ``"collision"``, ``"position_lock"``,
            ``"proximity"``, ``"zone"``, ``"custom"``.
        refs: Component references this constraint applies to.
        description: Human-readable description of what to fix/preserve.
        weight: Relative importance (positive = reward when met, negative = penalty when violated).
        target_resolved: Whether the constraint should be resolved (True) or preserved (False).
        tolerance_mm: Position lock tolerance in mm (for ``position_lock`` type).
        baseline_positions: Original positions ``{ref: (x, y)}`` to lock against.
    """

    constraint_type: str
    refs: tuple[str, ...]
    description: str
    weight: float = 1.0
    target_resolved: bool = True
    tolerance_mm: float = 1.0
    baseline_positions: tuple[tuple[str, float, float], ...] = ()


def load_human_constraints(pcb_dir: str | Path) -> tuple[HumanConstraint, ...]:
    """Load human feedback constraints from ``.pcb-review/human_constraints.json``.

    Returns an empty tuple if no file exists (no penalty, no reward).
    """
    path = Path(pcb_dir) / ".pcb-review" / "human_constraints.json"
    if not path.exists():
        return ()
    try:
        data = json.loads(path.read_text())
        constraints: list[HumanConstraint] = []
        for entry in data.get("constraints", []):
            baseline = tuple(
                (bp["ref"], bp["x"], bp["y"])
                for bp in entry.get("baseline_positions", [])
            )
            constraints.append(HumanConstraint(
                constraint_type=entry["type"],
                refs=tuple(entry.get("refs", [])),
                description=entry.get("description", ""),
                weight=entry.get("weight", 1.0),
                target_resolved=entry.get("target_resolved", True),
                tolerance_mm=entry.get("tolerance_mm", 1.0),
                baseline_positions=baseline,
            ))
        return tuple(constraints)
    except Exception:
        _log.warning("Failed to load human_constraints.json from %s", pcb_dir)
        return ()


def save_human_constraints(
    pcb_dir: str | Path,
    constraints: tuple[HumanConstraint, ...],
) -> None:
    """Write human feedback constraints to ``.pcb-review/human_constraints.json``."""
    path = Path(pcb_dir) / ".pcb-review"
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "constraints": [
            {
                "type": hc.constraint_type,
                "refs": list(hc.refs),
                "description": hc.description,
                "weight": hc.weight,
                "target_resolved": hc.target_resolved,
                "tolerance_mm": hc.tolerance_mm,
                "baseline_positions": [
                    {"ref": r, "x": x, "y": y} for r, x, y in hc.baseline_positions
                ],
            }
            for hc in constraints
        ],
    }
    (path / "human_constraints.json").write_text(
        json.dumps(data, indent=2) + "\n"
    )


def _score_human_feedback(
    pos: dict[str, tuple[float, float]],
    pcb_dir: str | Path | None = None,
    constraints: tuple[HumanConstraint, ...] | None = None,
) -> tuple[float, list[str]]:
    """Score placement against human feedback constraints.

    Args:
        pos: Dict of ``{ref: (x, y)}`` current positions.
        pcb_dir: Board directory containing ``.pcb-review/``.
        constraints: Pre-loaded constraints (skips file load if provided).

    Returns:
        ``(score, issues)`` where score is [0, 1].
        1.0 = all human constraints satisfied. 0.0 = all violated.
        Returns (1.0, []) if no constraints exist.
    """
    if constraints is None:
        if pcb_dir is None:
            return 1.0, []
        constraints = load_human_constraints(pcb_dir)
    if not constraints:
        return 1.0, []

    total_weight = 0.0
    earned_weight = 0.0
    issues: list[str] = []

    for hc in constraints:
        total_weight += abs(hc.weight)

        if hc.constraint_type == "position_lock":
            # Check that locked refs haven't moved beyond tolerance
            all_within = True
            for ref, bx, by in hc.baseline_positions:
                cx, cy = pos.get(ref, (bx, by))
                dist = math.hypot(cx - bx, cy - by)
                if dist > hc.tolerance_mm:
                    all_within = False
                    issues.append(
                        f"LOCKED {ref} moved {dist:.1f}mm (max {hc.tolerance_mm}mm)"
                    )
            if all_within:
                earned_weight += abs(hc.weight)

        elif hc.constraint_type == "collision":
            # Check that the named refs are NOT colliding
            # (uses simple distance check — full courtyard check is in _score_collisions)
            if len(hc.refs) >= 2:
                r1, r2 = hc.refs[0], hc.refs[1]
                p1 = pos.get(r1)
                p2 = pos.get(r2)
                if p1 and p2:
                    dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    # Collision resolved if components are > tolerance apart
                    if dist > hc.tolerance_mm:
                        earned_weight += abs(hc.weight)
                    else:
                        issues.append(
                            f"COLLISION {r1}-{r2}: {dist:.1f}mm "
                            f"(need >{hc.tolerance_mm}mm) — {hc.description}"
                        )

        elif hc.constraint_type == "proximity":
            # Check that refs are within tolerance of each other
            if len(hc.refs) >= 2:
                r1, r2 = hc.refs[0], hc.refs[1]
                p1 = pos.get(r1)
                p2 = pos.get(r2)
                if p1 and p2:
                    dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    if dist <= hc.tolerance_mm:
                        earned_weight += abs(hc.weight)
                    else:
                        issues.append(
                            f"PROXIMITY {r1}-{r2}: {dist:.1f}mm "
                            f"(need <{hc.tolerance_mm}mm) — {hc.description}"
                        )

        else:
            # Custom or unknown — score 1.0 (neutral) if target_resolved
            earned_weight += abs(hc.weight) * (1.0 if hc.target_resolved else 0.0)

    if total_weight == 0.0:
        return 1.0, []
    return earned_weight / total_weight, issues[:5]


def _score_utilization(pcb: PCBDesign) -> tuple[float, list[str]]:
    """Score board utilization — penalizes sprawling layouts with wasted space.

    Target: 25-60% utilization is ideal.  Below 15% = sprawl.  Above 75% = packed.
    """
    from kicad_pipeline.pcb.footprints import estimate_footprint_size

    # Board area from outline
    if pcb.outline and pcb.outline.polygon:
        xs = [p.x for p in pcb.outline.polygon]
        ys = [p.y for p in pcb.outline.polygon]
        board_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    else:
        board_area = 10000.0  # 100x100 fallback

    # Component area (sum of footprint bboxes)
    comp_area = 0.0
    for fp in pcb.footprints:
        if not fp.pads:
            continue
        w, h = estimate_footprint_size(fp.lib_id)
        comp_area += w * h

    if board_area < 1.0:
        return 1.0, []

    ratio = comp_area / board_area
    issues: list[str] = []

    if ratio < 0.10:
        score = ratio / 0.10  # 0→1 as ratio goes 0→10%
        issues.append(f"Board utilization {ratio:.0%} — very sparse layout")
    elif ratio < 0.25:
        score = 0.5 + 0.5 * (ratio - 0.10) / 0.15  # 0.5→1 as 10%→25%
    elif ratio <= 0.65:
        score = 1.0  # ideal range
    else:
        score = max(0.5, 1.0 - (ratio - 0.65) / 0.35)
        issues.append(f"Board utilization {ratio:.0%} — very dense")

    return score, issues


def _score_compactness(pcb: PCBDesign) -> tuple[float, list[str]]:
    """Score layout compactness — penalizes scattered component clusters.

    Measures the ratio of component bounding box to board area.
    A compact layout has all components in a tight cluster.
    """
    if not pcb.footprints:
        return 1.0, []

    # Get bounding box of all placed components (excluding mounting holes)
    comp_xs: list[float] = []
    comp_ys: list[float] = []
    for fp in pcb.footprints:
        if not fp.pads or fp.ref.startswith("H"):
            continue
        comp_xs.append(fp.position.x)
        comp_ys.append(fp.position.y)

    if len(comp_xs) < 2:
        return 1.0, []

    comp_bbox_w = max(comp_xs) - min(comp_xs)
    comp_bbox_h = max(comp_ys) - min(comp_ys)
    comp_bbox_area = max(comp_bbox_w * comp_bbox_h, 1.0)

    # Board area
    if pcb.outline and pcb.outline.polygon:
        xs = [p.x for p in pcb.outline.polygon]
        ys = [p.y for p in pcb.outline.polygon]
        board_area = max((max(xs) - min(xs)) * (max(ys) - min(ys)), 1.0)
    else:
        board_area = 10000.0

    ratio = comp_bbox_area / board_area
    issues: list[str] = []

    if ratio > 0.9:
        score = 1.0  # components fill the board — very compact
    elif ratio > 0.5:
        score = 0.8 + 0.2 * (ratio - 0.5) / 0.4
    elif ratio > 0.2:
        score = 0.5 + 0.3 * (ratio - 0.2) / 0.3
    else:
        score = ratio / 0.2 * 0.5  # 0→0.5 as 0%→20%
        issues.append(
            f"Components use {ratio:.0%} of board — layout is scattered"
        )

    return score, issues


def _score_signal_flow(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[float, list[str]]:
    """Score signal flow direction — power should flow left-to-right.

    Checks that power chain components are ordered consistently:
    input connectors on the left, regulators in the middle, output on the right.
    Also checks relay boards: relays should be ordered K1->K2->K3->K4 left-to-right.

    Returns (score, issues) where 1.0 = perfect ordering.
    """
    pos = _fp_position_dict(pcb)
    if not pos:
        return 1.0, []

    checks = 0
    passes = 0
    issues: list[str] = []

    # Check 1: Input connectors should be left of ICs
    j_refs = {r for r in pos if r.startswith("J")}
    u_refs = {r for r in pos if r.startswith("U")}
    if j_refs and u_refs:
        # Find leftmost connector and leftmost IC
        j_xs = [pos[r][0] for r in j_refs]
        u_xs = [pos[r][0] for r in u_refs]
        # At least one connector should be to the left of (or above) the ICs
        # This is a soft check — connectors at edges is already handled
        checks += 1
        if min(j_xs) <= min(u_xs) + 5.0:  # connector within 5mm of leftmost IC
            passes += 1
        else:
            issues.append("No input connector near left/top edge of IC cluster")

    # Check 2: If multiple regulators (U1, U2...), they should be ordered left-to-right
    reg_refs = sorted(
        [r for r in pos if r.startswith("U")],
        key=lambda r: int(r[1:]) if r[1:].isdigit() else 99,
    )
    if len(reg_refs) >= 2:
        checks += 1
        xs = [pos[r][0] for r in reg_refs]
        # Check if X coordinates are monotonically increasing (or close)
        ordered = all(xs[i] <= xs[i + 1] + 3.0 for i in range(len(xs) - 1))
        if ordered:
            passes += 1
        else:
            issues.append(f"Regulators not ordered L-to-R: {reg_refs}")

    # Check 3: If relays exist, they should be ordered K1->K2->... left-to-right
    k_refs = sorted(
        [r for r in pos if r.startswith("K")],
        key=lambda r: int(r[1:]) if r[1:].isdigit() else 99,
    )
    if len(k_refs) >= 2:
        checks += 1
        xs = [pos[r][0] for r in k_refs]
        ordered = all(xs[i] <= xs[i + 1] + 2.0 for i in range(len(xs) - 1))
        if ordered:
            passes += 1
        else:
            issues.append(f"Relays not ordered L-to-R: {k_refs}")

    if checks == 0:
        return 1.0, []
    return passes / checks, issues[:5]


def _gather_placement_subdimensions(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    pcb_dir: str | Path | None = None,
    human_constraints: tuple[HumanConstraint, ...] | None = None,
) -> dict[str, tuple[float, tuple[str, ...]]]:
    """Compute all 18 placement sub-dimension scores.

    Args:
        pcb: The PCB design to evaluate.
        requirements: Project requirements.
        pcb_dir: Board directory for loading human feedback constraints.
        human_constraints: Pre-loaded human constraints (skips file load).

    Returns a dict mapping dimension name to ``(score, issues)`` pairs.
    """
    collision_score, collision_issues = _score_collisions(pcb)
    cohesion_score, cohesion_issues = _score_subcircuit_cohesion(pcb, requirements)
    isolation_score, isolation_issues = _score_voltage_isolation(pcb, requirements)
    connector_score, connector_issues = _score_connector_edge(pcb)
    decoupling_score, decoupling_issues = _score_decoupling_proximity(pcb, requirements)
    boundary_score, _boundary_issues = _score_boundary(pcb)
    mcu_periph_score, mcu_periph_issues = _score_mcu_peripheral_proximity(pcb, requirements)
    rf_edge_score, rf_edge_issues = _score_rf_edge_placement(pcb, requirements)
    group_cohesion_score, group_cohesion_issues = _score_group_cohesion(pcb, requirements)
    subgroup_score, subgroup_issues = _score_subgroup_cohesion(pcb, requirements)
    grp_isolation_score, grp_isolation_issues = _score_group_isolation(pcb, requirements)
    pad_facing_score, pad_facing_issues = _score_pad_facing(pcb, requirements)
    pos_2d = _fp_position_dict(pcb)
    constraint_score, constraint_issues = _score_constraint_compliance(pos_2d, requirements)
    human_score, human_issues = _score_human_feedback(
        pos_2d, pcb_dir=pcb_dir, constraints=human_constraints,
    )
    utilization_score, utilization_issues = _score_utilization(pcb)
    compactness_score, compactness_issues = _score_compactness(pcb)
    signal_flow_score, signal_flow_issues = _score_signal_flow(pcb, requirements)

    return {
        "collision": (collision_score, tuple(collision_issues[:5])),
        "cohesion": (cohesion_score, tuple(cohesion_issues[:5])),
        "isolation": (isolation_score, tuple(isolation_issues[:5])),
        "connector": (connector_score, tuple(connector_issues[:5])),
        "decoupling": (decoupling_score, tuple(decoupling_issues[:5])),
        "boundary": (boundary_score, tuple(_boundary_issues[:5])),
        "mcu_periph": (mcu_periph_score, tuple(mcu_periph_issues[:5])),
        "rf_edge": (rf_edge_score, tuple(rf_edge_issues[:5])),
        "group_cohesion": (group_cohesion_score, tuple(group_cohesion_issues[:5])),
        "subgroup": (subgroup_score, tuple(subgroup_issues[:5])),
        "grp_isolation": (grp_isolation_score, tuple(grp_isolation_issues[:5])),
        "pad_facing": (pad_facing_score, tuple(pad_facing_issues[:5])),
        "constraint_compliance": (constraint_score, tuple(constraint_issues[:5])),
        "human_feedback": (human_score, tuple(human_issues[:5])),
        "utilization": (utilization_score, tuple(utilization_issues[:5])),
        "compactness": (compactness_score, tuple(compactness_issues[:5])),
        "signal_flow": (signal_flow_score, tuple(signal_flow_issues[:5])),
    }


def _build_fast_breakdown(
    dims: dict[str, tuple[float, tuple[str, ...]]],
) -> tuple[ScoreDetail, ...]:
    """Build the ScoreDetail breakdown tuple from sub-dimension scores."""
    _detail_spec: tuple[tuple[str, str, float], ...] = (
        ("collision", "Collisions", _FAST_WEIGHT_COLLISION),
        ("cohesion", "Sub-circuit Cohesion", _FAST_WEIGHT_SUBCIRCUIT_COHESION),
        ("isolation", "Voltage Isolation", _FAST_WEIGHT_VOLTAGE_ISOLATION),
        ("connector", "Connector Edge", _FAST_WEIGHT_CONNECTOR_EDGE),
        ("decoupling", "Decoupling Proximity", _FAST_WEIGHT_DECOUPLING_PROXIMITY),
        ("mcu_periph", "MCU Peripheral", _FAST_WEIGHT_MCU_PERIPHERAL),
        ("rf_edge", "RF Edge", _FAST_WEIGHT_RF_EDGE),
        ("group_cohesion", "Group Cohesion", _FAST_WEIGHT_GROUP_COHESION),
        ("subgroup", "Subgroup Cohesion", _FAST_WEIGHT_SUBGROUP_COHESION),
        ("grp_isolation", "Group Isolation", _FAST_WEIGHT_GROUP_ISOLATION),
        ("pad_facing", "Pad Facing", _FAST_WEIGHT_PAD_FACING),
        ("constraint_compliance", "Constraint Compliance", _FAST_WEIGHT_CONSTRAINT_COMPLIANCE),
        ("human_feedback", "Human Feedback", _FAST_WEIGHT_HUMAN_FEEDBACK),
        ("utilization", "Board Utilization", _FAST_WEIGHT_UTILIZATION),
        ("compactness", "Layout Compactness", _FAST_WEIGHT_COMPACTNESS),
        ("signal_flow", "Signal Flow", _FAST_WEIGHT_SIGNAL_FLOW),
    )
    return tuple(
        ScoreDetail(
            category=label,
            score=dims[key][0],
            weight=weight,
            issues=dims[key][1],
        )
        for key, label, weight in _detail_spec
    )


def compute_fast_placement_score(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    pcb_dir: str | Path | None = None,
    human_constraints: tuple[HumanConstraint, ...] | None = None,
) -> QualityScore:
    """Compute a placement-focused quality score without full validation.

    Evaluates 18 EE-aligned placement sub-dimensions and returns a composite
    :class:`QualityScore`.  The 15th dimension scores compliance with human
    feedback constraints loaded from ``.pcb-review/human_constraints.json``.

    Args:
        pcb: The PCB design to evaluate.
        requirements: Project requirements with nets and feature blocks.
        pcb_dir: Board directory for loading human feedback constraints.
        human_constraints: Pre-loaded human constraints (skips file load).

    Returns:
        A :class:`QualityScore` with placement-derived scores.
    """
    dims = _gather_placement_subdimensions(
        pcb, requirements, pcb_dir=pcb_dir, human_constraints=human_constraints,
    )

    # Weighted placement composite (15 dimensions)
    placement_score = (
        _FAST_WEIGHT_COLLISION * dims["collision"][0]
        + _FAST_WEIGHT_SUBCIRCUIT_COHESION * dims["cohesion"][0]
        + _FAST_WEIGHT_VOLTAGE_ISOLATION * dims["isolation"][0]
        + _FAST_WEIGHT_CONNECTOR_EDGE * dims["connector"][0]
        + _FAST_WEIGHT_DECOUPLING_PROXIMITY * dims["decoupling"][0]
        + _FAST_WEIGHT_MCU_PERIPHERAL * dims["mcu_periph"][0]
        + _FAST_WEIGHT_RF_EDGE * dims["rf_edge"][0]
        + _FAST_WEIGHT_CONNECTOR_ORIENTATION * 1.0  # orientation scored via review
        + _FAST_WEIGHT_REGULATOR_BOUNDARY * dims["boundary"][0]
        + _FAST_WEIGHT_GROUP_COHESION * dims["group_cohesion"][0]
        + _FAST_WEIGHT_SUBGROUP_COHESION * dims["subgroup"][0]
        + _FAST_WEIGHT_GROUP_ISOLATION * dims["grp_isolation"][0]
        + _FAST_WEIGHT_PAD_FACING * dims["pad_facing"][0]
        + _FAST_WEIGHT_CONSTRAINT_COMPLIANCE * dims["constraint_compliance"][0]
        + _FAST_WEIGHT_HUMAN_FEEDBACK * dims["human_feedback"][0]
        + _FAST_WEIGHT_UTILIZATION * dims["utilization"][0]
        + _FAST_WEIGHT_COMPACTNESS * dims["compactness"][0]
        + _FAST_WEIGHT_SIGNAL_FLOW * dims["signal_flow"][0]
    )

    manufacturing_score = _clamp01(0.5 + 0.5 * dims["collision"][0])
    electrical_score = _clamp01(
        0.5 + 0.25 * dims["isolation"][0] + 0.25 * dims["boundary"][0],
    )

    scores = (
        (electrical_score, _WEIGHT_ELECTRICAL),
        (manufacturing_score, _WEIGHT_MANUFACTURING),
        (placement_score, _WEIGHT_PLACEMENT),
        (dims["cohesion"][0], _WEIGHT_SIGNAL_INTEGRITY),
        (1.0, _WEIGHT_THERMAL),
    )
    overall = _weighted_geometric_mean(scores)
    grade = score_to_grade(overall)

    return QualityScore(
        board_cost=0.0,
        electrical_score=round(electrical_score, 4),
        manufacturing_score=round(manufacturing_score, 4),
        thermal_score=1.0,
        signal_integrity_score=round(dims["cohesion"][0], 4),
        placement_score=round(placement_score, 4),
        overall_score=round(overall, 4),
        grade=grade,
        breakdown=_build_fast_breakdown(dims),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _score_validation_dimensions(
    validation_report: ValidationReport,
) -> tuple[
    tuple[float, list[str]],
    tuple[float, list[str]],
    tuple[float, list[str]],
    tuple[float, list[str]],
]:
    """Extract per-dimension scores from a validation report.

    Returns (electrical, manufacturing, thermal, signal_integrity) tuples
    of ``(score, issues)``.
    """
    from kicad_pipeline.validation.drc import Severity

    # Electrical / DRC
    drc_errors = len(validation_report.drc.errors)
    drc_warnings = sum(
        1 for v in validation_report.drc.violations if v.severity == Severity.WARNING
    )
    elec_errors = len(validation_report.electrical.errors)
    elec_warnings = sum(
        1 for v in validation_report.electrical.violations if v.severity == Severity.WARNING
    )
    total_elec_err = drc_errors + elec_errors
    total_elec_warn = drc_warnings + elec_warnings
    electrical_score = _score_from_violations(total_elec_err, total_elec_warn, 0.15, 0.03)
    electrical_issues: list[str] = []
    if total_elec_err > 0:
        electrical_issues.append(f"{total_elec_err} electrical/DRC errors")
    if total_elec_warn > 0:
        electrical_issues.append(f"{total_elec_warn} electrical/DRC warnings")

    # Manufacturing
    mfg_errors = len(validation_report.manufacturing.errors)
    mfg_warnings = sum(
        1 for v in validation_report.manufacturing.violations if v.severity == Severity.WARNING
    )
    manufacturing_score = _score_from_violations(mfg_errors, mfg_warnings, 0.2, 0.05)
    manufacturing_issues: list[str] = []
    if mfg_errors > 0:
        manufacturing_issues.append(f"{mfg_errors} manufacturing errors")
    if mfg_warnings > 0:
        manufacturing_issues.append(f"{mfg_warnings} manufacturing warnings")

    # Thermal
    thermal_errors = sum(
        1 for v in validation_report.thermal.violations if v.severity == Severity.ERROR
    )
    thermal_warnings = sum(
        1 for v in validation_report.thermal.violations if v.severity == Severity.WARNING
    )
    thermal_score = _score_from_violations(thermal_errors, thermal_warnings, 0.15, 0.03)
    thermal_issues: list[str] = []
    if thermal_errors > 0:
        thermal_issues.append(f"{thermal_errors} thermal errors")
    if thermal_warnings > 0:
        thermal_issues.append(f"{thermal_warnings} thermal warnings")

    # Signal integrity
    si_errors = len(validation_report.signal_integrity.errors)
    si_warnings = sum(
        1 for v in validation_report.signal_integrity.violations
        if v.severity == Severity.WARNING
    )
    si_score = _score_from_violations(si_errors, si_warnings, 0.1, 0.02)
    si_issues: list[str] = []
    if si_errors > 0:
        si_issues.append(f"{si_errors} signal integrity errors")
    if si_warnings > 0:
        si_issues.append(f"{si_warnings} signal integrity warnings")

    return (
        (electrical_score, electrical_issues),
        (manufacturing_score, manufacturing_issues),
        (thermal_score, thermal_issues),
        (si_score, si_issues),
    )


def compute_quality_score(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    validation_report: ValidationReport | None = None,
    routing_metrics: BoardRoutingMetrics | None = None,
) -> QualityScore:
    """Compute a composite quality score for *pcb*.

    Args:
        pcb: The PCB design to evaluate.
        requirements: Project requirements (used for cross-reference).
        validation_report: Optional unified validation report.
        routing_metrics: Optional routing metrics for board cost.

    Returns:
        A :class:`QualityScore` summarising all dimensions.
    """
    board_cost = 0.0
    if routing_metrics is not None:
        from kicad_pipeline.routing.metrics import compute_board_cost
        board_cost = compute_board_cost(routing_metrics)

    if validation_report is not None:
        elec, mfg, therm, si = _score_validation_dimensions(validation_report)
        electrical_score, electrical_issues = elec
        manufacturing_score, manufacturing_issues = mfg
        thermal_score, thermal_issues = therm
        si_score, si_issues = si
    else:
        electrical_score = manufacturing_score = thermal_score = si_score = 1.0
        electrical_issues = manufacturing_issues = thermal_issues = si_issues = []

    placement_score, placement_issues = _compute_placement_score_from_pcb(pcb)
    if routing_metrics is not None and routing_metrics.avg_passive_distance_mm > 0.0:
        avg_dist = routing_metrics.avg_passive_distance_mm
        placement_score = _clamp01(
            1.0 - (avg_dist - _PLACEMENT_IDEAL_DISTANCE_MM) / _PLACEMENT_IDEAL_DISTANCE_MM
        )
        p_issues: list[str] = []
        if placement_score < 0.7:
            p_issues.append(
                f"Average passive-to-IC distance is {avg_dist:.1f} mm "
                f"(ideal < {_PLACEMENT_IDEAL_DISTANCE_MM:.0f} mm)"
            )
        placement_issues = tuple(p_issues)

    scores = (
        (electrical_score, _WEIGHT_ELECTRICAL),
        (manufacturing_score, _WEIGHT_MANUFACTURING),
        (placement_score, _WEIGHT_PLACEMENT),
        (si_score, _WEIGHT_SIGNAL_INTEGRITY),
        (thermal_score, _WEIGHT_THERMAL),
    )
    overall = _weighted_geometric_mean(scores)
    grade = score_to_grade(overall)

    breakdown = (
        ScoreDetail("Electrical/DRC", electrical_score, _WEIGHT_ELECTRICAL,
                     tuple(electrical_issues)),
        ScoreDetail("Manufacturing", manufacturing_score, _WEIGHT_MANUFACTURING,
                     tuple(manufacturing_issues)),
        ScoreDetail("Placement", placement_score, _WEIGHT_PLACEMENT,
                     tuple(placement_issues)),
        ScoreDetail("Signal Integrity", si_score, _WEIGHT_SIGNAL_INTEGRITY,
                     tuple(si_issues)),
        ScoreDetail("Thermal", thermal_score, _WEIGHT_THERMAL,
                     tuple(thermal_issues)),
    )

    return QualityScore(
        board_cost=round(board_cost, 2),
        electrical_score=round(electrical_score, 4),
        manufacturing_score=round(manufacturing_score, 4),
        thermal_score=round(thermal_score, 4),
        signal_integrity_score=round(si_score, 4),
        placement_score=round(placement_score, 4),
        overall_score=round(overall, 4),
        grade=grade,
        breakdown=breakdown,
    )
