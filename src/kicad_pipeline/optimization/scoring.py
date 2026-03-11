"""Quality scoring engine for PCB designs.

Computes a multi-dimensional quality score from validation reports, routing
metrics, and placement analysis.  The overall score uses a weighted geometric
mean so that a single zero-dimension drags the composite down hard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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

# Fast-path sub-dimension weights (EE-aligned, v2)
_FAST_WEIGHT_COLLISION: float = 0.20
_FAST_WEIGHT_SUBCIRCUIT_COHESION: float = 0.05
_FAST_WEIGHT_VOLTAGE_ISOLATION: float = 0.15
_FAST_WEIGHT_CONNECTOR_EDGE: float = 0.10
_FAST_WEIGHT_DECOUPLING_PROXIMITY: float = 0.10
_FAST_WEIGHT_MCU_PERIPHERAL: float = 0.10
_FAST_WEIGHT_RF_EDGE: float = 0.05
_FAST_WEIGHT_CONNECTOR_ORIENTATION: float = 0.05
_FAST_WEIGHT_REGULATOR_BOUNDARY: float = 0.05
_FAST_WEIGHT_GROUP_COHESION: float = 0.05
_FAST_WEIGHT_SUBGROUP_COHESION: float = 0.05
_FAST_WEIGHT_GROUP_ISOLATION: float = 0.05

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
    """Build ref → (x, y) lookup from PCB footprints."""
    return {fp.ref: (fp.position.x, fp.position.y) for fp in pcb.footprints}


def _fp_size_dict(pcb: PCBDesign) -> dict[str, tuple[float, float]]:
    """Build ref → (width, height) lookup from PCB footprints.

    Uses pad extents + courtyard margins as a size proxy.
    """
    sizes: dict[str, tuple[float, float]] = {}
    for fp in pcb.footprints:
        if fp.pads:
            xs = [p.position.x for p in fp.pads]
            ys = [p.position.y for p in fp.pads]
            w = max(xs) - min(xs) + 1.0  # pad span + margin
            h = max(ys) - min(ys) + 1.0
            sizes[fp.ref] = (max(w, 1.0), max(h, 1.0))
        else:
            sizes[fp.ref] = (2.0, 2.0)  # default for padless fps
    return sizes


def _score_collisions(
    pcb: PCBDesign,
) -> tuple[float, list[str]]:
    """Score based on courtyard collision count.

    Returns (score, issues) where score = 1.0 - penalty_per_collision * count.
    """
    positions = {fp.ref: fp.position for fp in pcb.footprints}
    sizes = _fp_size_dict(pcb)
    rotations = {fp.ref: fp.rotation for fp in pcb.footprints}

    collisions: list[str] = []
    refs = list(positions.keys())
    for i, ref_a in enumerate(refs):
        xa, ya = positions[ref_a].x, positions[ref_a].y
        wa, ha = sizes.get(ref_a, (2.0, 2.0))
        rot_a = rotations.get(ref_a, 0.0)
        if rot_a % 180 in (90.0, 270.0):
            wa, ha = ha, wa

        for ref_b in refs[i + 1:]:
            xb, yb = positions[ref_b].x, positions[ref_b].y
            wb, hb = sizes.get(ref_b, (2.0, 2.0))
            rot_b = rotations.get(ref_b, 0.0)
            if rot_b % 180 in (90.0, 270.0):
                wb, hb = hb, wb

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

        # Compute max pairwise distance (spread)
        max_dist = 0.0
        for i in range(len(block_positions)):
            for j in range(i + 1, len(block_positions)):
                dx = block_positions[i][0] - block_positions[j][0]
                dy = block_positions[i][1] - block_positions[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d > max_dist:
                    max_dist = d

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

    for fp in pcb.footprints:
        x, y = fp.position.x, fp.position.y
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

    for i, d1 in enumerate(domains):
        for d2 in domains[i + 1:]:
            # Sample: check closest pair per domain pair
            min_dist = float("inf")
            for r1 in domain_refs[d1][:10]:  # cap for performance
                for r2 in domain_refs[d2][:10]:
                    d = math.sqrt(
                        (pos[r1][0] - pos[r2][0]) ** 2 +
                        (pos[r1][1] - pos[r2][1]) ** 2
                    )
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

    for sc in rf_modules:
        pos = _fp_position_dict(pcb).get(sc.anchor_ref)
        if pos is None:
            continue
        x, y = pos
        edge_dist = min(x - min_x, max_x - x, y - min_y, max_y - y)
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
    fp_pos: dict[str, tuple[float, float]] = {
        fp.ref: (fp.position.x, fp.position.y) for fp in pcb.footprints
    }

    issues: list[str] = []
    scores: list[float] = []

    # Subgroup types and their max spread thresholds
    thresholds: dict[SubCircuitType, float] = {
        SubCircuitType.RELAY_DRIVER: 8.0,
        SubCircuitType.ADC_CHANNEL: 10.0,
        SubCircuitType.DECOUPLING: 5.0,
        SubCircuitType.BUCK_CONVERTER: 12.0,
        SubCircuitType.CRYSTAL_OSC: 8.0,
    }

    for sc in subcircuits:
        threshold = thresholds.get(sc.circuit_type)
        if threshold is None:
            continue

        positions = [fp_pos[r] for r in sc.refs if r in fp_pos]
        if len(positions) < 2:
            continue

        # Compute spread (max distance between any two members)
        max_dist = 0.0
        for i, p1 in enumerate(positions):
            for p2 in positions[i + 1:]:
                d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                max_dist = max(max_dist, d)

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

    fp_pos: dict[str, tuple[float, float]] = {
        fp.ref: (fp.position.x, fp.position.y) for fp in pcb.footprints
    }

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


def compute_fast_placement_score(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> QualityScore:
    """Compute a placement-focused quality score without full validation.

    Evaluates 5 EE-aligned placement sub-dimensions:

    - **Collisions** (25%): courtyard overlap detection
    - **Sub-circuit cohesion** (25%): sub-circuit components clustered
    - **Voltage isolation** (20%): domain separation maintained
    - **Connector edge** (15%): connectors near board edges
    - **Decoupling proximity** (15%): caps near their ICs

    Args:
        pcb: The PCB design to evaluate.
        requirements: Project requirements with nets and feature blocks.

    Returns:
        A :class:`QualityScore` with placement-derived scores.
    """
    # Sub-dimension scores
    collision_score, collision_issues = _score_collisions(pcb)
    cohesion_score, cohesion_issues = _score_subcircuit_cohesion(pcb, requirements)
    isolation_score, isolation_issues = _score_voltage_isolation(pcb, requirements)
    connector_score, connector_issues = _score_connector_edge(pcb)
    decoupling_score, decoupling_issues = _score_decoupling_proximity(pcb, requirements)
    boundary_score, _boundary_issues = _score_boundary(pcb)
    mcu_periph_score, mcu_periph_issues = _score_mcu_peripheral_proximity(pcb, requirements)
    rf_edge_score, rf_edge_issues = _score_rf_edge_placement(pcb, requirements)
    group_cohesion_score, group_cohesion_issues = _score_group_cohesion(pcb, requirements)

    # Subgroup cohesion
    subgroup_score, subgroup_issues = _score_subgroup_cohesion(pcb, requirements)

    # Group isolation
    grp_isolation_score, grp_isolation_issues = _score_group_isolation(pcb, requirements)

    # Weighted placement composite (12 dimensions)
    placement_score = (
        _FAST_WEIGHT_COLLISION * collision_score
        + _FAST_WEIGHT_SUBCIRCUIT_COHESION * cohesion_score
        + _FAST_WEIGHT_VOLTAGE_ISOLATION * isolation_score
        + _FAST_WEIGHT_CONNECTOR_EDGE * connector_score
        + _FAST_WEIGHT_DECOUPLING_PROXIMITY * decoupling_score
        + _FAST_WEIGHT_MCU_PERIPHERAL * mcu_periph_score
        + _FAST_WEIGHT_RF_EDGE * rf_edge_score
        + _FAST_WEIGHT_CONNECTOR_ORIENTATION * 1.0  # orientation scored via review
        + _FAST_WEIGHT_REGULATOR_BOUNDARY * boundary_score
        + _FAST_WEIGHT_GROUP_COHESION * group_cohesion_score
        + _FAST_WEIGHT_SUBGROUP_COHESION * subgroup_score
        + _FAST_WEIGHT_GROUP_ISOLATION * grp_isolation_score
    )

    # For fast path, other dimensions are derived from placement sub-scores
    manufacturing_score = _clamp01(
        0.5 + 0.5 * collision_score
    )
    electrical_score = _clamp01(
        0.5 + 0.25 * isolation_score + 0.25 * boundary_score
    )

    # Overall: use full weight system but with placement-derived estimates
    scores = (
        (electrical_score, _WEIGHT_ELECTRICAL),
        (manufacturing_score, _WEIGHT_MANUFACTURING),
        (placement_score, _WEIGHT_PLACEMENT),
        (cohesion_score, _WEIGHT_SIGNAL_INTEGRITY),
        (1.0, _WEIGHT_THERMAL),
    )
    overall = _weighted_geometric_mean(scores)
    grade = score_to_grade(overall)

    breakdown = (
        ScoreDetail(
            category="Collisions",
            score=collision_score,
            weight=_FAST_WEIGHT_COLLISION,
            issues=tuple(collision_issues[:5]),
        ),
        ScoreDetail(
            category="Sub-circuit Cohesion",
            score=cohesion_score,
            weight=_FAST_WEIGHT_SUBCIRCUIT_COHESION,
            issues=tuple(cohesion_issues[:5]),
        ),
        ScoreDetail(
            category="Voltage Isolation",
            score=isolation_score,
            weight=_FAST_WEIGHT_VOLTAGE_ISOLATION,
            issues=tuple(isolation_issues[:5]),
        ),
        ScoreDetail(
            category="Connector Edge",
            score=connector_score,
            weight=_FAST_WEIGHT_CONNECTOR_EDGE,
            issues=tuple(connector_issues[:5]),
        ),
        ScoreDetail(
            category="Decoupling Proximity",
            score=decoupling_score,
            weight=_FAST_WEIGHT_DECOUPLING_PROXIMITY,
            issues=tuple(decoupling_issues[:5]),
        ),
        ScoreDetail(
            category="MCU Peripheral",
            score=mcu_periph_score,
            weight=_FAST_WEIGHT_MCU_PERIPHERAL,
            issues=tuple(mcu_periph_issues[:5]),
        ),
        ScoreDetail(
            category="RF Edge",
            score=rf_edge_score,
            weight=_FAST_WEIGHT_RF_EDGE,
            issues=tuple(rf_edge_issues[:5]),
        ),
        ScoreDetail(
            category="Group Cohesion",
            score=group_cohesion_score,
            weight=_FAST_WEIGHT_GROUP_COHESION,
            issues=tuple(group_cohesion_issues[:5]),
        ),
        ScoreDetail(
            category="Subgroup Cohesion",
            score=subgroup_score,
            weight=_FAST_WEIGHT_SUBGROUP_COHESION,
            issues=tuple(subgroup_issues[:5]),
        ),
        ScoreDetail(
            category="Group Isolation",
            score=grp_isolation_score,
            weight=_FAST_WEIGHT_GROUP_ISOLATION,
            issues=tuple(grp_isolation_issues[:5]),
        ),
    )

    return QualityScore(
        board_cost=0.0,
        electrical_score=round(electrical_score, 4),
        manufacturing_score=round(manufacturing_score, 4),
        thermal_score=1.0,
        signal_integrity_score=round(cohesion_score, 4),
        placement_score=round(placement_score, 4),
        overall_score=round(overall, 4),
        grade=grade,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
        validation_report: Optional unified validation report.  When provided,
            error/warning counts are extracted from each sub-report.
        routing_metrics: Optional routing metrics.  When provided, the board
            cost is computed via :func:`compute_board_cost`.

    Returns:
        A :class:`QualityScore` summarising all dimensions.
    """
    from kicad_pipeline.validation.drc import Severity

    # --- Board cost --------------------------------------------------------
    board_cost = 0.0
    if routing_metrics is not None:
        from kicad_pipeline.routing.metrics import compute_board_cost

        board_cost = compute_board_cost(routing_metrics)

    # --- Per-dimension scoring ---------------------------------------------
    electrical_issues: list[str] = []
    manufacturing_issues: list[str] = []
    thermal_issues: list[str] = []
    si_issues: list[str] = []

    if validation_report is not None:
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
        if si_errors > 0:
            si_issues.append(f"{si_errors} signal integrity errors")
        if si_warnings > 0:
            si_issues.append(f"{si_warnings} signal integrity warnings")
    else:
        # No validation report — assume perfect
        electrical_score = 1.0
        manufacturing_score = 1.0
        thermal_score = 1.0
        si_score = 1.0

    # --- Placement ---------------------------------------------------------
    placement_score, placement_issues = _compute_placement_score_from_pcb(pcb)

    # If routing_metrics provides avg_passive_distance_mm, prefer it
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

    # --- Composite ---------------------------------------------------------
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
        ScoreDetail(
            category="Electrical/DRC",
            score=electrical_score,
            weight=_WEIGHT_ELECTRICAL,
            issues=tuple(electrical_issues),
        ),
        ScoreDetail(
            category="Manufacturing",
            score=manufacturing_score,
            weight=_WEIGHT_MANUFACTURING,
            issues=tuple(manufacturing_issues),
        ),
        ScoreDetail(
            category="Placement",
            score=placement_score,
            weight=_WEIGHT_PLACEMENT,
            issues=tuple(placement_issues),
        ),
        ScoreDetail(
            category="Signal Integrity",
            score=si_score,
            weight=_WEIGHT_SIGNAL_INTEGRITY,
            issues=tuple(si_issues),
        ),
        ScoreDetail(
            category="Thermal",
            score=thermal_score,
            weight=_WEIGHT_THERMAL,
            issues=tuple(thermal_issues),
        ),
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
