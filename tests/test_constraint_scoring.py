"""Tests for constraint compliance scoring (14th placement dimension)."""

from __future__ import annotations

from kicad_pipeline.optimization.scoring import (
    _FAST_WEIGHT_COLLISION,
    _FAST_WEIGHT_CONNECTOR_EDGE,
    _FAST_WEIGHT_CONNECTOR_ORIENTATION,
    _FAST_WEIGHT_CONSTRAINT_COMPLIANCE,
    _FAST_WEIGHT_DECOUPLING_PROXIMITY,
    _FAST_WEIGHT_HUMAN_FEEDBACK,
    _FAST_WEIGHT_GROUP_COHESION,
    _FAST_WEIGHT_GROUP_ISOLATION,
    _FAST_WEIGHT_MCU_PERIPHERAL,
    _FAST_WEIGHT_PAD_FACING,
    _FAST_WEIGHT_REGULATOR_BOUNDARY,
    _FAST_WEIGHT_RF_EDGE,
    _FAST_WEIGHT_SUBCIRCUIT_COHESION,
    _FAST_WEIGHT_SUBGROUP_COHESION,
    _FAST_WEIGHT_VOLTAGE_ISOLATION,
    _score_constraint_compliance,
    compute_fast_placement_score,
)
from tests.helpers import (
    make_component,
    make_pcb_design,
    make_requirements,
)

# ---------------------------------------------------------------------------
# _score_constraint_compliance — unit tests
# ---------------------------------------------------------------------------


def test_constraint_score_no_requirements() -> None:
    """None requirements returns score 1.0 with no issues."""
    score, issues = _score_constraint_compliance({}, None)
    assert score == 1.0
    assert issues == []


def test_constraint_score_no_constraints() -> None:
    """Requirements with no placement constraints return score 1.0."""
    reqs = make_requirements(
        components=(
            make_component("R1", "10k", "R_0805"),
            make_component("R2", "4.7k", "R_0805"),
        ),
    )
    positions: dict[str, tuple[float, float]] = {
        "R1": (10.0, 10.0),
        "R2": (30.0, 10.0),
    }
    score, issues = _score_constraint_compliance(positions, reqs)
    assert score == 1.0
    assert issues == []


def test_constraint_score_all_satisfied() -> None:
    """When all proximity constraints are met, score equals 1.0."""
    from kicad_pipeline.models.requirements import Component

    comp_r1 = Component(
        ref="R1",
        value="10k",
        footprint="R_0805",
        placement_near="R2",
        placement_near_max_mm=20.0,
    )
    comp_r2 = Component(ref="R2", value="4.7k", footprint="R_0805")
    reqs = make_requirements(components=(comp_r1, comp_r2))
    # R1 and R2 are only 5mm apart — well within the 20mm constraint
    positions: dict[str, tuple[float, float]] = {
        "R1": (10.0, 10.0),
        "R2": (15.0, 10.0),
    }
    score, issues = _score_constraint_compliance(positions, reqs)
    assert score == 1.0
    assert issues == []


def test_constraint_score_violations() -> None:
    """Violated proximity constraints reduce score below 1.0."""
    from kicad_pipeline.models.requirements import Component

    comp_r1 = Component(
        ref="R1",
        value="10k",
        footprint="R_0805",
        placement_near="R2",
        placement_near_max_mm=5.0,  # tight: must be within 5mm
    )
    comp_r2 = Component(ref="R2", value="4.7k", footprint="R_0805")
    reqs = make_requirements(components=(comp_r1, comp_r2))
    # R1 and R2 are 50mm apart — violates the 5mm constraint
    positions: dict[str, tuple[float, float]] = {
        "R1": (0.0, 0.0),
        "R2": (50.0, 0.0),
    }
    score, issues = _score_constraint_compliance(positions, reqs)
    assert score < 1.0
    assert len(issues) >= 1
    assert "R1" in issues[0] or "R2" in issues[0]


def test_constraint_score_missing_refs_skipped() -> None:
    """Constraints referencing refs not in positions are skipped gracefully."""
    from kicad_pipeline.models.requirements import Component

    comp_r1 = Component(
        ref="R1",
        value="10k",
        footprint="R_0805",
        placement_near="R99",  # R99 not in positions
        placement_near_max_mm=5.0,
    )
    reqs = make_requirements(components=(comp_r1,))
    positions: dict[str, tuple[float, float]] = {"R1": (10.0, 10.0)}
    # R99 is absent — should not raise, constraint is skipped
    score, issues = _score_constraint_compliance(positions, reqs)
    # No violations counted since R99 is absent
    assert score == 1.0


# ---------------------------------------------------------------------------
# compute_fast_placement_score integration
# ---------------------------------------------------------------------------


def test_constraint_compliance_appears_in_fast_score_breakdown() -> None:
    """Constraint Compliance dimension must appear in fast score breakdown."""
    pcb = make_pcb_design()
    reqs = make_requirements()
    score = compute_fast_placement_score(pcb, reqs)
    categories = [d.category for d in score.breakdown]
    assert "Constraint Compliance" in categories


def test_constraint_compliance_weight_in_breakdown() -> None:
    """Constraint Compliance breakdown entry has the correct weight."""
    pcb = make_pcb_design()
    reqs = make_requirements()
    score = compute_fast_placement_score(pcb, reqs)
    detail = next(d for d in score.breakdown if d.category == "Constraint Compliance")
    assert abs(detail.weight - _FAST_WEIGHT_CONSTRAINT_COMPLIANCE) < 1e-9


# ---------------------------------------------------------------------------
# Weight sum invariant
# ---------------------------------------------------------------------------


def test_weights_sum_to_one() -> None:
    """All 15 fast-path placement weights must sum to 1.0."""
    all_weights = (
        _FAST_WEIGHT_COLLISION,
        _FAST_WEIGHT_SUBCIRCUIT_COHESION,
        _FAST_WEIGHT_VOLTAGE_ISOLATION,
        _FAST_WEIGHT_CONNECTOR_EDGE,
        _FAST_WEIGHT_DECOUPLING_PROXIMITY,
        _FAST_WEIGHT_MCU_PERIPHERAL,
        _FAST_WEIGHT_RF_EDGE,
        _FAST_WEIGHT_CONNECTOR_ORIENTATION,
        _FAST_WEIGHT_REGULATOR_BOUNDARY,
        _FAST_WEIGHT_GROUP_COHESION,
        _FAST_WEIGHT_SUBGROUP_COHESION,
        _FAST_WEIGHT_GROUP_ISOLATION,
        _FAST_WEIGHT_PAD_FACING,
        _FAST_WEIGHT_CONSTRAINT_COMPLIANCE,
        _FAST_WEIGHT_HUMAN_FEEDBACK,
    )
    total = sum(all_weights)
    assert abs(total - 1.0) < 1e-9, (
        f"Weights sum to {total:.10f}, expected 1.0. "
        f"Individual weights: {all_weights}"
    )
