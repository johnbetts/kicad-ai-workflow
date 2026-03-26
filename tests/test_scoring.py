"""Tests for kicad_pipeline.optimization.scoring."""

from __future__ import annotations

import math

import pytest

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    PCBDesign,
    Point,
)
from kicad_pipeline.models.requirements import (
    Net,
    ProjectRequirements,
)
from kicad_pipeline.optimization.scoring import (
    QualityScore,
    ScoreDetail,
    _clamp01,
    _weighted_geometric_mean,
    compute_quality_score,
    score_to_grade,
)
from tests.helpers import (
    make_board_outline,
    make_component,
    make_pcb_design,
    make_requirements,
)

# ---------------------------------------------------------------------------
# Fixtures — thin wrappers over shared conftest helpers
# ---------------------------------------------------------------------------


def _minimal_outline() -> BoardOutline:
    return make_board_outline(w=50.0, h=40.0)


def _minimal_pcb(
    footprints: tuple[Footprint, ...] = (),
) -> PCBDesign:
    return make_pcb_design(footprints=footprints, w=50.0, h=40.0)


def _minimal_requirements() -> ProjectRequirements:
    return make_requirements(
        components=(make_component("R1", "10k", "R_0805"),),
        nets=(Net(name="GND", connections=()),),
        w=50.0,
        h=40.0,
    )


# ---------------------------------------------------------------------------
# Frozen dataclass tests
# ---------------------------------------------------------------------------


def test_score_detail_frozen() -> None:
    sd = ScoreDetail(category="DRC", score=0.9, weight=0.3, issues=())
    with pytest.raises(AttributeError):
        sd.score = 0.5  # type: ignore[misc]


def test_quality_score_frozen() -> None:
    qs = QualityScore(
        board_cost=0.0,
        electrical_score=1.0,
        manufacturing_score=1.0,
        thermal_score=1.0,
        signal_integrity_score=1.0,
        placement_score=1.0,
        overall_score=1.0,
        grade="A",
        breakdown=(),
    )
    with pytest.raises(AttributeError):
        qs.grade = "F"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# score_to_grade
# ---------------------------------------------------------------------------


def test_score_to_grade_boundaries() -> None:
    assert score_to_grade(1.0) == "A"
    assert score_to_grade(0.9) == "A"
    assert score_to_grade(0.89) == "B"
    assert score_to_grade(0.75) == "B"
    assert score_to_grade(0.74) == "C"
    assert score_to_grade(0.6) == "C"
    assert score_to_grade(0.59) == "D"
    assert score_to_grade(0.4) == "D"
    assert score_to_grade(0.39) == "F"
    assert score_to_grade(0.0) == "F"


# ---------------------------------------------------------------------------
# compute_quality_score — minimal cases
# ---------------------------------------------------------------------------


def test_compute_quality_score_minimal_pcb() -> None:
    pcb = _minimal_pcb()
    req = _minimal_requirements()
    result = compute_quality_score(pcb, req)

    assert 0.0 <= result.overall_score <= 1.0
    assert result.grade in ("A", "B", "C", "D", "F")
    assert len(result.breakdown) == 5
    assert result.board_cost == 0.0


def test_compute_quality_score_perfect_scores() -> None:
    """Without a validation report, all scores default to 1.0."""
    pcb = _minimal_pcb()
    req = _minimal_requirements()
    result = compute_quality_score(pcb, req)

    assert result.electrical_score == 1.0
    assert result.manufacturing_score == 1.0
    assert result.thermal_score == 1.0
    assert result.signal_integrity_score == 1.0
    # Placement depends on footprints; with none it should be 1.0
    assert result.placement_score == 1.0
    assert result.overall_score == 1.0
    assert result.grade == "A"


def test_compute_quality_score_with_validation_report() -> None:
    """Scores derived from validation report error/warning counts."""
    from kicad_pipeline.validation.drc import DRCReport, DRCViolation, Severity
    from kicad_pipeline.validation.electrical import ElectricalReport
    from kicad_pipeline.validation.manufacturing import ManufacturingReport
    from kicad_pipeline.validation.report import build_validation_report
    from kicad_pipeline.validation.signal_integrity import SIReport
    from kicad_pipeline.validation.thermal import ThermalReport

    # 2 DRC errors, 1 warning
    drc = DRCReport(
        violations=(
            DRCViolation(
                rule="clearance", message="err1", severity=Severity.ERROR
            ),
            DRCViolation(
                rule="clearance", message="err2", severity=Severity.ERROR
            ),
            DRCViolation(
                rule="silk_overlap", message="w1", severity=Severity.WARNING
            ),
        )
    )
    elec = ElectricalReport(violations=())
    mfg = ManufacturingReport(violations=())
    thermal = ThermalReport(component_thermals=(), violations=())
    si = SIReport(violations=())

    report = build_validation_report(
        drc=drc, electrical=elec, manufacturing=mfg,
        thermal=thermal, si=si,
    )

    pcb = _minimal_pcb()
    req = _minimal_requirements()
    result = compute_quality_score(pcb, req, validation_report=report)

    # 2 errors * 0.15 + 1 warning * 0.03 = 0.33 penalty -> 0.67
    assert abs(result.electrical_score - 0.67) < 0.01
    assert result.manufacturing_score == 1.0
    assert result.grade in ("B", "C")


def test_compute_quality_score_all_failures() -> None:
    """Many errors should push scores toward zero."""
    from kicad_pipeline.validation.drc import DRCReport, DRCViolation, Severity
    from kicad_pipeline.validation.electrical import ElectricalReport
    from kicad_pipeline.validation.manufacturing import (
        ManufacturingReport,
        ManufacturingViolation,
    )
    from kicad_pipeline.validation.report import build_validation_report
    from kicad_pipeline.validation.signal_integrity import SIReport, SIViolation
    from kicad_pipeline.validation.thermal import ThermalReport

    drc_viols = tuple(
        DRCViolation(rule="err", message=f"err{i}", severity=Severity.ERROR)
        for i in range(20)
    )
    elec_viols = tuple(
        DRCViolation(rule="elec", message=f"e{i}", severity=Severity.ERROR)
        for i in range(10)
    )
    mfg_viols = tuple(
        ManufacturingViolation(
            rule="mfg", message=f"m{i}", severity=Severity.ERROR
        )
        for i in range(10)
    )
    thermal_viols = tuple(
        DRCViolation(rule="th", message=f"t{i}", severity=Severity.ERROR)
        for i in range(10)
    )
    si_viols = tuple(
        SIViolation(rule="si", message=f"s{i}", severity=Severity.ERROR)
        for i in range(15)
    )

    report = build_validation_report(
        drc=DRCReport(violations=drc_viols),
        electrical=ElectricalReport(violations=elec_viols),
        manufacturing=ManufacturingReport(violations=mfg_viols),
        thermal=ThermalReport(
            component_thermals=(), violations=thermal_viols
        ),
        si=SIReport(violations=si_viols),
    )

    pcb = _minimal_pcb()
    req = _minimal_requirements()
    result = compute_quality_score(pcb, req, validation_report=report)

    assert result.electrical_score == 0.0
    assert result.manufacturing_score == 0.0
    assert result.overall_score < 0.05
    assert result.grade == "F"


def test_compute_quality_score_with_routing_metrics() -> None:
    """Board cost should be populated from routing metrics."""
    from kicad_pipeline.routing.grid_router import RouteQuality
    from kicad_pipeline.routing.metrics import BoardRoutingMetrics

    metrics = BoardRoutingMetrics(
        total_track_length_mm=100.0,
        total_vias=5,
        nets_routed=10,
        nets_failed=0,
        overall_length_ratio=1.2,
        max_vias_per_net=2,
        per_net=(
            RouteQuality(
                net_name="GND",
                actual_length_mm=50.0,
                manhattan_ideal_mm=40.0,
                length_ratio=1.25,
                via_count=2,
                bend_count=3,
                score=5.0,
            ),
        ),
        total_bends=3,
        avg_passive_distance_mm=8.0,
    )

    pcb = _minimal_pcb()
    req = _minimal_requirements()
    result = compute_quality_score(pcb, req, routing_metrics=metrics)

    assert result.board_cost > 0.0


def test_board_cost_from_metrics() -> None:
    """Board cost from routing metrics should be non-negative."""
    from kicad_pipeline.routing.metrics import BoardRoutingMetrics, compute_board_cost

    metrics = BoardRoutingMetrics(
        total_track_length_mm=200.0,
        total_vias=10,
        nets_routed=5,
        nets_failed=2,
        overall_length_ratio=1.5,
        max_vias_per_net=3,
        per_net=(),
    )

    cost = compute_board_cost(metrics)
    assert cost >= 0.0
    # 200 trace + 160 vias + 400 unrouted = at least 760
    assert cost >= 760.0


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def test_placement_score_computation() -> None:
    """Footprints placed close together yield a better placement score."""
    # Two passives near one IC
    ic = Footprint(
        lib_id="Package_SO:SOIC-8", ref="U1", value="ATtiny85",
        position=Point(25.0, 20.0), pads=(),
    )
    r1 = Footprint(
        lib_id="R_0805:R_0805", ref="R1", value="10k",
        position=Point(27.0, 20.0), pads=(),
    )
    r2 = Footprint(
        lib_id="R_0805:R_0805", ref="R2", value="4k7",
        position=Point(23.0, 20.0), pads=(),
    )

    pcb = _minimal_pcb(footprints=(ic, r1, r2))
    req = _minimal_requirements()
    result = compute_quality_score(pcb, req)

    # Passives are 2mm from IC — well under 15mm ideal
    assert result.placement_score >= 0.85


def test_empty_pcb_returns_valid_score() -> None:
    pcb = _minimal_pcb()
    req = _minimal_requirements()
    result = compute_quality_score(pcb, req)

    assert 0.0 <= result.overall_score <= 1.0
    assert result.grade in ("A", "B", "C", "D", "F")
    assert len(result.breakdown) == 5


# ---------------------------------------------------------------------------
# Weighted geometric mean
# ---------------------------------------------------------------------------


def test_weighted_geometric_mean_calculation() -> None:
    # All 1.0 -> 1.0
    val = _weighted_geometric_mean(((1.0, 1.0), (1.0, 1.0)))
    assert abs(val - 1.0) < 1e-9

    # Mixed: (0.5, 0.5), (1.0, 0.5) -> geometric mean with equal weights
    result = _weighted_geometric_mean(((0.5, 0.5), (1.0, 0.5)))
    expected = math.exp(0.5 * math.log(0.5) + 0.5 * math.log(1.0))
    assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# Grade mapping
# ---------------------------------------------------------------------------


def test_quality_score_grade_a() -> None:
    pcb = _minimal_pcb()
    req = _minimal_requirements()
    result = compute_quality_score(pcb, req)
    # No validation report -> all 1.0 -> grade A
    assert result.grade == "A"
    assert result.overall_score >= 0.9


def test_quality_score_grade_f() -> None:
    from kicad_pipeline.validation.drc import DRCReport, DRCViolation, Severity
    from kicad_pipeline.validation.electrical import ElectricalReport
    from kicad_pipeline.validation.manufacturing import (
        ManufacturingReport,
        ManufacturingViolation,
    )
    from kicad_pipeline.validation.report import build_validation_report
    from kicad_pipeline.validation.signal_integrity import SIReport, SIViolation
    from kicad_pipeline.validation.thermal import ThermalReport

    many_errors = tuple(
        DRCViolation(rule="x", message="x", severity=Severity.ERROR)
        for _ in range(50)
    )
    mfg_errors = tuple(
        ManufacturingViolation(
            rule="x", message="x", severity=Severity.ERROR
        )
        for _ in range(50)
    )
    si_errors = tuple(
        SIViolation(rule="x", message="x", severity=Severity.ERROR)
        for _ in range(50)
    )

    report = build_validation_report(
        drc=DRCReport(violations=many_errors),
        electrical=ElectricalReport(violations=many_errors),
        manufacturing=ManufacturingReport(violations=mfg_errors),
        thermal=ThermalReport(
            component_thermals=(), violations=many_errors
        ),
        si=SIReport(violations=si_errors),
    )

    pcb = _minimal_pcb()
    req = _minimal_requirements()
    result = compute_quality_score(pcb, req, validation_report=report)
    assert result.grade == "F"
    assert result.overall_score < 0.4


# ---------------------------------------------------------------------------
# Clamp
# ---------------------------------------------------------------------------


def test_score_clamp_to_valid_range() -> None:
    assert _clamp01(-0.5) == 0.0
    assert _clamp01(0.5) == 0.5
    assert _clamp01(1.5) == 1.0


# ---------------------------------------------------------------------------
# Pad Facing
# ---------------------------------------------------------------------------


def test_pad_facing_appears_in_fast_score_breakdown() -> None:
    """Pad Facing dimension should appear in compute_fast_placement_score."""
    from kicad_pipeline.optimization.scoring import compute_fast_placement_score

    pcb = _minimal_pcb()
    reqs = _minimal_requirements()
    score = compute_fast_placement_score(pcb, reqs)
    categories = [d.category for d in score.breakdown]
    assert "Pad Facing" in categories


def test_pad_facing_perfect_orientation() -> None:
    """Two resistors with pads facing each other should score high."""
    from kicad_pipeline.models.pcb import Pad
    from kicad_pipeline.optimization.scoring import _score_pad_facing

    # R1 at (10, 20), R2 at (30, 20) — connected via SIG net
    # R1 pad 2 (east) connects to R2 pad 1 (west) — perfect facing
    pad_r1_1 = Pad(
        number="1", pad_type="smd", shape="rect",
        position=Point(x=-0.9, y=0.0), size_x=1.0, size_y=0.5,
        layers=("F.Cu",), net_number=1, net_name="GND",
    )
    pad_r1_2 = Pad(
        number="2", pad_type="smd", shape="rect",
        position=Point(x=0.9, y=0.0), size_x=1.0, size_y=0.5,
        layers=("F.Cu",), net_number=2, net_name="SIG",
    )
    pad_r2_1 = Pad(
        number="1", pad_type="smd", shape="rect",
        position=Point(x=-0.9, y=0.0), size_x=1.0, size_y=0.5,
        layers=("F.Cu",), net_number=2, net_name="SIG",
    )
    pad_r2_2 = Pad(
        number="2", pad_type="smd", shape="rect",
        position=Point(x=0.9, y=0.0), size_x=1.0, size_y=0.5,
        layers=("F.Cu",), net_number=3, net_name="OUT",
    )
    fp1 = Footprint(
        lib_id="test:R", ref="R1", value="10k",
        position=Point(x=10.0, y=20.0), rotation=0.0,
        pads=(pad_r1_1, pad_r1_2),
    )
    fp2 = Footprint(
        lib_id="test:R", ref="R2", value="4.7k",
        position=Point(x=30.0, y=20.0), rotation=0.0,
        pads=(pad_r2_1, pad_r2_2),
    )
    pcb = PCBDesign(
        outline=_minimal_outline(),
        design_rules=DesignRules(),
        nets=(
            NetEntry(number=0, name=""),
            NetEntry(number=1, name="GND"),
            NetEntry(number=2, name="SIG"),
            NetEntry(number=3, name="OUT"),
        ),
        footprints=(fp1, fp2),
        tracks=(), vias=(), zones=(), keepouts=(),
    )
    reqs = _minimal_requirements()

    score, issues = _score_pad_facing(pcb, reqs)
    # R1 pad 2 (EAST) faces R2 pad 1 (WEST) — both facing toward each other
    assert score > 0.8, f"Expected high score for facing pads, got {score}"


def test_pad_facing_no_signal_nets_neutral() -> None:
    """PCB with only power nets should score 1.0 (neutral)."""
    from kicad_pipeline.optimization.scoring import _score_pad_facing

    pcb = _minimal_pcb()
    reqs = _minimal_requirements()
    score, issues = _score_pad_facing(pcb, reqs)
    assert score == 1.0


# ---------------------------------------------------------------------------
# score_to_grade — additional edge cases
# ---------------------------------------------------------------------------


def test_score_to_grade_exact_boundary_a() -> None:
    """Exactly 0.9 → A."""
    assert score_to_grade(0.9) == "A"


def test_score_to_grade_just_below_a() -> None:
    """0.8999 → B (not A)."""
    assert score_to_grade(0.8999) == "B"


def test_score_to_grade_negative() -> None:
    """Negative score → F."""
    assert score_to_grade(-0.5) == "F"


def test_score_to_grade_above_one() -> None:
    """Score > 1.0 → A."""
    assert score_to_grade(1.5) == "A"


# ---------------------------------------------------------------------------
# _clamp01 — additional
# ---------------------------------------------------------------------------


def test_clamp01_exact_boundaries() -> None:
    """0.0 and 1.0 pass through unchanged."""
    assert _clamp01(0.0) == 0.0
    assert _clamp01(1.0) == 1.0


def test_clamp01_midpoint() -> None:
    """0.5 passes through unchanged."""
    assert _clamp01(0.5) == 0.5


# ---------------------------------------------------------------------------
# _weighted_geometric_mean — edge cases
# ---------------------------------------------------------------------------


def test_weighted_geometric_mean_zero_weight() -> None:
    """Total weight of 0 returns 0."""
    result = _weighted_geometric_mean(((0.5, 0.0),))
    assert result == 0.0


def test_weighted_geometric_mean_single_element() -> None:
    """Single element with weight 1.0 returns the element score."""
    result = _weighted_geometric_mean(((0.7, 1.0),))
    assert abs(result - 0.7) < 1e-6


def test_weighted_geometric_mean_floor_prevents_zero() -> None:
    """Zero score is floored at 0.01 to prevent log(0)."""
    result = _weighted_geometric_mean(((0.0, 1.0), (1.0, 1.0)))
    # exp(0.5 * log(0.01) + 0.5 * log(1.0)) = exp(-2.3025...) = ~0.1
    assert result > 0.0
    assert result < 0.2


# ---------------------------------------------------------------------------
# compute_fast_placement_score — basic
# ---------------------------------------------------------------------------


def test_compute_fast_placement_score_empty_board() -> None:
    """Empty board yields valid QualityScore."""
    from kicad_pipeline.optimization.scoring import compute_fast_placement_score

    pcb = _minimal_pcb()
    reqs = _minimal_requirements()
    score = compute_fast_placement_score(pcb, reqs)

    assert 0.0 <= score.overall_score <= 1.0
    assert score.grade in ("A", "B", "C", "D", "F")
    assert len(score.breakdown) > 0


def test_compute_fast_placement_score_has_breakdown_dimensions() -> None:
    """Fast score has multiple breakdown dimensions (at least 10)."""
    from kicad_pipeline.optimization.scoring import compute_fast_placement_score

    pcb = _minimal_pcb()
    reqs = _minimal_requirements()
    score = compute_fast_placement_score(pcb, reqs)
    assert len(score.breakdown) >= 10


def test_compute_fast_placement_score_all_dimensions_valid_range() -> None:
    """All breakdown dimensions are in [0, 1]."""
    from kicad_pipeline.optimization.scoring import compute_fast_placement_score

    pcb = _minimal_pcb()
    reqs = _minimal_requirements()
    score = compute_fast_placement_score(pcb, reqs)
    for dim in score.breakdown:
        assert 0.0 <= dim.score <= 1.0, f"{dim.category} score {dim.score} out of range"


def test_compute_fast_placement_score_board_cost_zero() -> None:
    """Fast placement score always has board_cost = 0 (no routing)."""
    from kicad_pipeline.optimization.scoring import compute_fast_placement_score

    pcb = _minimal_pcb()
    reqs = _minimal_requirements()
    score = compute_fast_placement_score(pcb, reqs)
    assert score.board_cost == 0.0


def test_compute_fast_placement_score_thermal_always_one() -> None:
    """Thermal score is always 1.0 in fast path (not evaluated)."""
    from kicad_pipeline.optimization.scoring import compute_fast_placement_score

    pcb = _minimal_pcb()
    reqs = _minimal_requirements()
    score = compute_fast_placement_score(pcb, reqs)
    assert score.thermal_score == 1.0


# ---------------------------------------------------------------------------
# _score_boundary
# ---------------------------------------------------------------------------


def test_score_boundary_all_inside() -> None:
    """All footprints inside board → score 1.0."""
    from kicad_pipeline.optimization.scoring import _score_boundary

    fp = Footprint(
        lib_id="test:R", ref="R1", value="10k",
        position=Point(25.0, 20.0), pads=(),
    )
    pcb = _minimal_pcb(footprints=(fp,))
    score, issues = _score_boundary(pcb)
    assert score == 1.0
    assert len(issues) == 0


def test_score_boundary_outside_penalized() -> None:
    """Footprint outside board → penalized score."""
    from kicad_pipeline.optimization.scoring import _score_boundary

    fp = Footprint(
        lib_id="test:R", ref="R1", value="10k",
        position=Point(200.0, 200.0), pads=(),
    )
    pcb = _minimal_pcb(footprints=(fp,))
    score, issues = _score_boundary(pcb)
    assert score < 1.0
    assert len(issues) == 1
    assert "R1" in issues[0]


def test_score_boundary_no_outline() -> None:
    """No board outline → score 1.0."""
    from kicad_pipeline.optimization.scoring import _score_boundary

    pcb = PCBDesign(
        outline=BoardOutline(polygon=()),
        design_rules=DesignRules(),
        nets=(NetEntry(number=0, name=""),),
        footprints=(),
        tracks=(), vias=(), zones=(), keepouts=(),
    )
    score, issues = _score_boundary(pcb)
    assert score == 1.0
