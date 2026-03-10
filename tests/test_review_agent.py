"""Tests for review_agent: EE placement critique."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.models.requirements import (
    Component,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.functional_grouper import (
    DetectedSubCircuit,
    SubCircuitType,
    VoltageDomain,
)
from kicad_pipeline.optimization.review_agent import (
    CONNECTOR_EDGE_MAX_MM,
    PlacementReview,
    PlacementRule,
    PlacementViolation,
    review_placement,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pin(num: str, name: str, net: str | None = None) -> Pin:
    return Pin(number=num, name=name, pin_type=PinType.PASSIVE, function=None, net=net)


def _comp(ref: str, value: str, fp: str = "R_0402",
          pins: tuple[Pin, ...] = (), desc: str | None = None) -> Component:
    return Component(ref=ref, value=value, footprint=fp, lcsc=None,
                     description=desc, datasheet=None, pins=pins)


def _pad(num: str, x: float = 0, y: float = 0) -> Pad:
    return Pad(number=num, pad_type="smd", shape="roundrect",
               position=Point(x, y), size_x=1.0, size_y=0.6,
               layers=("F.Cu", "F.Paste", "F.Mask"),
               net_number=0, net_name="")


def _fp(ref: str, x: float, y: float, pads: tuple[Pad, ...] | None = None) -> Footprint:
    if pads is None:
        pads = (_pad("1", -0.5, 0), _pad("2", 0.5, 0))
    return Footprint(
        lib_id=f"test:{ref}",
        ref=ref,
        value=ref,
        position=Point(x, y),
        rotation=0.0,
        layer="F.Cu",
        pads=pads,
        graphics=(),
        texts=(),
        lcsc=None,
    )


def _make_pcb(
    footprints: list[Footprint],
    board_w: float = 100.0,
    board_h: float = 80.0,
) -> PCBDesign:
    outline = BoardOutline(
        polygon=(
            Point(0, 0), Point(board_w, 0),
            Point(board_w, board_h), Point(0, board_h),
            Point(0, 0),
        ),
        width=0.1,
    )
    return PCBDesign(
        outline=outline,
        design_rules=DesignRules(),
        nets=(NetEntry(0, ""), NetEntry(1, "GND")),
        footprints=tuple(footprints),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_requirements(
    components: list[Component],
    nets: list[Net],
) -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="test", author="test", revision="1", description="test"),
        features=(),
        components=tuple(components),
        nets=tuple(nets),
        pin_map=None,
        power_budget=None,
        mechanical=MechanicalConstraints(board_width_mm=100.0, board_height_mm=80.0),
        recommendations=(),
        board_context=None,
    )


# ---------------------------------------------------------------------------
# Decoupling distance
# ---------------------------------------------------------------------------


class TestDecouplingDistance:
    def test_close_cap_no_violation(self) -> None:
        """Cap within 3mm of IC → no violation."""
        pcb = _make_pcb([_fp("U1", 50, 40), _fp("C1", 52, 40)])
        reqs = _make_requirements(
            [_comp("U1", "STM32", pins=(_pin("1", "VDD", "+3V3"), _pin("2", "GND", "GND"))),
             _comp("C1", "100nF", pins=(_pin("1", "1", "+3V3"), _pin("2", "2", "GND")))],
            [Net("+3V3", (NetConnection("U1", "1"), NetConnection("C1", "1"))),
             Net("GND", (NetConnection("U1", "2"), NetConnection("C1", "2")))],
        )
        review = review_placement(pcb, reqs)
        decoup_violations = [v for v in review.violations
                             if v.rule == PlacementRule.DECOUPLING_DISTANCE]
        assert len(decoup_violations) == 0

    def test_far_cap_violation(self) -> None:
        """Cap 20mm from IC → violation with suggested position."""
        pcb = _make_pcb([_fp("U1", 50, 40), _fp("C1", 70, 40)])
        reqs = _make_requirements(
            [_comp("U1", "STM32", pins=(_pin("1", "VDD", "+3V3"), _pin("2", "GND", "GND"))),
             _comp("C1", "100nF", pins=(_pin("1", "1", "+3V3"), _pin("2", "2", "GND")))],
            [Net("+3V3", (NetConnection("U1", "1"), NetConnection("C1", "1"))),
             Net("GND", (NetConnection("U1", "2"), NetConnection("C1", "2")))],
        )
        review = review_placement(pcb, reqs)
        decoup_violations = [v for v in review.violations
                             if v.rule == PlacementRule.DECOUPLING_DISTANCE]
        assert len(decoup_violations) == 1
        v = decoup_violations[0]
        assert v.severity in ("critical", "major")
        assert v.suggested_position is not None


# ---------------------------------------------------------------------------
# Sub-circuit spread
# ---------------------------------------------------------------------------


class TestSubcircuitSpread:
    def test_tight_group_no_violation(self) -> None:
        pcb = _make_pcb([_fp("K1", 50, 40), _fp("Q1", 52, 40)])
        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=("K1", "Q1"),
            anchor_ref="K1",
            net_connections=(),
            domain=VoltageDomain.VIN_24V,
        )
        reqs = _make_requirements(
            [_comp("K1", "RELAY"), _comp("Q1", "2N7002")], [],
        )
        review = review_placement(pcb, reqs, subcircuits=(sc,))
        spread = [v for v in review.violations if v.rule == PlacementRule.SUBCIRCUIT_SPREAD]
        assert len(spread) == 0

    def test_scattered_group_violation(self) -> None:
        pcb = _make_pcb([_fp("K1", 10, 10), _fp("Q1", 80, 70)])
        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=("K1", "Q1"),
            anchor_ref="K1",
            net_connections=(),
            domain=VoltageDomain.VIN_24V,
        )
        reqs = _make_requirements(
            [_comp("K1", "RELAY"), _comp("Q1", "2N7002")], [],
        )
        review = review_placement(pcb, reqs, subcircuits=(sc,))
        spread = [v for v in review.violations if v.rule == PlacementRule.SUBCIRCUIT_SPREAD]
        assert len(spread) == 1
        assert spread[0].suggested_position is not None


# ---------------------------------------------------------------------------
# Connector edge
# ---------------------------------------------------------------------------


class TestConnectorEdge:
    def test_connector_on_edge_no_violation(self) -> None:
        pcb = _make_pcb([_fp("J1", 2, 40)])
        reqs = _make_requirements([_comp("J1", "RJ45")], [])
        review = review_placement(pcb, reqs)
        edge = [v for v in review.violations if v.rule == PlacementRule.CONNECTOR_EDGE]
        assert len(edge) == 0

    def test_connector_center_violation(self) -> None:
        pcb = _make_pcb([_fp("J1", 50, 40)])
        reqs = _make_requirements([_comp("J1", "RJ45")], [])
        review = review_placement(pcb, reqs)
        edge = [v for v in review.violations if v.rule == PlacementRule.CONNECTOR_EDGE]
        assert len(edge) == 1
        assert edge[0].current_value > CONNECTOR_EDGE_MAX_MM


# ---------------------------------------------------------------------------
# Voltage isolation
# ---------------------------------------------------------------------------


class TestVoltageIsolation:
    def test_separated_domains_no_violation(self) -> None:
        pcb = _make_pcb([_fp("K1", 10, 10), _fp("U1", 80, 70)])
        domain_map = {"K1": VoltageDomain.VIN_24V, "U1": VoltageDomain.DIGITAL_3V3}
        reqs = _make_requirements([_comp("K1", "RELAY"), _comp("U1", "STM32")], [])
        review = review_placement(pcb, reqs, subcircuits=(), domain_map=domain_map)
        iso = [v for v in review.violations if v.rule == PlacementRule.VOLTAGE_ISOLATION]
        assert len(iso) == 0

    def test_adjacent_domains_violation(self) -> None:
        pcb = _make_pcb([_fp("K1", 50, 40), _fp("U1", 50.5, 40)])
        domain_map = {"K1": VoltageDomain.VIN_24V, "U1": VoltageDomain.DIGITAL_3V3}
        reqs = _make_requirements([_comp("K1", "RELAY"), _comp("U1", "STM32")], [])
        review = review_placement(pcb, reqs, subcircuits=(), domain_map=domain_map)
        iso = [v for v in review.violations if v.rule == PlacementRule.VOLTAGE_ISOLATION]
        assert len(iso) == 1


# ---------------------------------------------------------------------------
# Crystal proximity
# ---------------------------------------------------------------------------


class TestCrystalProximity:
    def test_close_crystal_no_violation(self) -> None:
        pcb = _make_pcb([_fp("Y1", 50, 40), _fp("U1", 52, 40)])
        reqs = _make_requirements(
            [_comp("Y1", "8MHz", pins=(_pin("1", "1", "XTAL_IN"),)),
             _comp("U1", "STM32", pins=(_pin("1", "OSC_IN", "XTAL_IN"),))],
            [Net("XTAL_IN", (NetConnection("Y1", "1"), NetConnection("U1", "1")))],
        )
        review = review_placement(pcb, reqs, subcircuits=())
        crystal = [v for v in review.violations if v.rule == PlacementRule.CRYSTAL_PROXIMITY]
        assert len(crystal) == 0

    def test_far_crystal_violation(self) -> None:
        pcb = _make_pcb([_fp("Y1", 10, 10), _fp("U1", 80, 70)])
        reqs = _make_requirements(
            [_comp("Y1", "8MHz", pins=(_pin("1", "1", "XTAL_IN"),)),
             _comp("U1", "STM32", pins=(_pin("1", "OSC_IN", "XTAL_IN"),))],
            [Net("XTAL_IN", (NetConnection("Y1", "1"), NetConnection("U1", "1")))],
        )
        review = review_placement(pcb, reqs, subcircuits=())
        crystal = [v for v in review.violations if v.rule == PlacementRule.CRYSTAL_PROXIMITY]
        assert len(crystal) == 1


# ---------------------------------------------------------------------------
# Collisions
# ---------------------------------------------------------------------------


class TestCollisions:
    def test_no_overlap_no_violation(self) -> None:
        pcb = _make_pcb([_fp("R1", 10, 10), _fp("R2", 20, 10)])
        reqs = _make_requirements([_comp("R1", "10k"), _comp("R2", "10k")], [])
        review = review_placement(pcb, reqs, subcircuits=())
        collisions = [v for v in review.violations if v.rule == PlacementRule.COLLISION]
        assert len(collisions) == 0

    def test_overlap_violation(self) -> None:
        pcb = _make_pcb([_fp("R1", 10, 10), _fp("R2", 10, 10)])
        reqs = _make_requirements([_comp("R1", "10k"), _comp("R2", "10k")], [])
        review = review_placement(pcb, reqs, subcircuits=())
        collisions = [v for v in review.violations if v.rule == PlacementRule.COLLISION]
        assert len(collisions) >= 1
        assert collisions[0].severity == "critical"


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


class TestGrading:
    def test_clean_board_grade_a(self) -> None:
        pcb = _make_pcb([_fp("R1", 10, 10)])
        reqs = _make_requirements([_comp("R1", "10k")], [])
        review = review_placement(pcb, reqs, subcircuits=())
        assert review.grade == "A"

    def test_many_violations_grade_f(self) -> None:
        # Put 10 connectors all in the center
        fps = [_fp(f"J{i}", 50, 40) for i in range(1, 11)]
        comps = [_comp(f"J{i}", "CONN") for i in range(1, 11)]
        pcb = _make_pcb(fps)
        reqs = _make_requirements(comps, [])
        review = review_placement(pcb, reqs, subcircuits=())
        # Many collisions (critical) → should be D or F
        assert review.grade in ("D", "F")

    def test_review_summary_format(self) -> None:
        pcb = _make_pcb([_fp("R1", 10, 10)])
        reqs = _make_requirements([_comp("R1", "10k")], [])
        review = review_placement(pcb, reqs, subcircuits=())
        assert "Grade" in review.summary
        assert "violation" in review.summary


# ---------------------------------------------------------------------------
# Data class immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_violation_frozen(self) -> None:
        v = PlacementViolation(
            rule=PlacementRule.COLLISION,
            severity="critical",
            refs=("R1",),
            message="test",
            current_value=0.0,
            threshold=0.0,
            suggested_position=None,
        )
        with pytest.raises(AttributeError):
            v.severity = "minor"  # type: ignore[misc]

    def test_review_frozen(self) -> None:
        r = PlacementReview(violations=(), grade="A", summary="clean")
        with pytest.raises(AttributeError):
            r.grade = "F"  # type: ignore[misc]
