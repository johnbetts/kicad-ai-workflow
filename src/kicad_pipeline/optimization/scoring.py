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

# Grade thresholds
_GRADE_A: float = 0.9
_GRADE_B: float = 0.75
_GRADE_C: float = 0.6
_GRADE_D: float = 0.4

# Floor to prevent zero in geometric mean
_SCORE_FLOOR: float = 0.01

# Placement: reference distance for normalisation (mm)
_PLACEMENT_IDEAL_DISTANCE_MM: float = 15.0


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
