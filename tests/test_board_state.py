"""Tests for board_state module — spatial snapshot + text report."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.models.requirements import (
    FeatureBlock,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.board_state import (
    build_board_state,
)
from kicad_pipeline.optimization.functional_grouper import (
    DetectedSubCircuit,
    SubCircuitType,
    VoltageDomain,
)
from kicad_pipeline.optimization.group_placer import PlacedGroup
from kicad_pipeline.optimization.review_agent import PlacementReview, PlacementViolation
from kicad_pipeline.optimization.zone_partitioner import BoardZone

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_smd_pad(number: str, x: float, y: float, w: float = 1.0, h: float = 1.0) -> Pad:
    """Create a minimal SMD pad."""
    return Pad(
        number=number,
        pad_type="smd",
        shape="rect",
        position=Point(x, y),
        size_x=w,
        size_y=h,
        layers=("F.Cu", "F.Paste", "F.Mask"),
    )


def _make_footprint(
    ref: str,
    value: str,
    lib_id: str,
    x: float,
    y: float,
    rotation: float = 0.0,
    pads: tuple[Pad, ...] = (),
) -> Footprint:
    """Create a minimal footprint."""
    if not pads:
        # Default: four pads forming a 2x2mm rectangle
        pads = (
            _make_smd_pad("1", -1.0, -1.0),
            _make_smd_pad("2", 1.0, -1.0),
            _make_smd_pad("3", 1.0, 1.0),
            _make_smd_pad("4", -1.0, 1.0),
        )
    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(x, y),
        rotation=rotation,
        pads=pads,
    )


def _make_connector(
    ref: str,
    x: float,
    y: float,
    rotation: float = 0.0,
    num_pins: int = 4,
) -> Footprint:
    """Create a connector footprint with pads along Y axis."""
    pads = tuple(
        _make_smd_pad(str(i + 1), 0.0, float(i) * 2.54 - (num_pins - 1) * 1.27)
        for i in range(num_pins)
    )
    return Footprint(
        lib_id="Connector:Conn_01x04",
        ref=ref,
        value="Conn",
        position=Point(x, y),
        rotation=rotation,
        pads=pads,
    )


def _make_pcb(
    footprints: tuple[Footprint, ...],
    width: float = 100.0,
    height: float = 60.0,
) -> PCBDesign:
    """Create a minimal PCB design."""
    outline = BoardOutline(
        polygon=(
            Point(0.0, 0.0),
            Point(width, 0.0),
            Point(width, height),
            Point(0.0, height),
            Point(0.0, 0.0),
        )
    )
    return PCBDesign(
        outline=outline,
        design_rules=DesignRules(),
        nets=(),
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_requirements(
    features: tuple[FeatureBlock, ...] = (),
) -> ProjectRequirements:
    """Create minimal project requirements."""
    return ProjectRequirements(
        project=ProjectInfo(name="test", description="test project"),
        features=features,
        components=(),
        nets=(),
    )


# ---------------------------------------------------------------------------
# Tests: build_board_state basics
# ---------------------------------------------------------------------------

class TestBuildBoardState:
    """Test the factory function."""

    def test_empty_board(self) -> None:
        pcb = _make_pcb(footprints=())
        req = _make_requirements()
        state = build_board_state(pcb, req)

        assert state.board_width_mm == 100.0
        assert state.board_height_mm == 60.0
        assert state.components == ()
        assert state.overlaps == ()
        assert state.off_board_count == 0
        assert state.review_grade == "?"
        assert state.total_utilization_pct == 0.0

    def test_single_component(self) -> None:
        fp = _make_footprint("R1", "10k", "Resistor_SMD:R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        state = build_board_state(pcb, req)

        assert len(state.components) == 1
        c = state.components[0]
        assert c.ref == "R1"
        assert c.value == "10k"
        assert c.footprint_id == "Resistor_SMD:R_0402"
        assert c.rotation == 0.0
        assert c.overlapping_refs == ()
        assert c.group_name == ""
        assert c.voltage_domain == ""

    def test_board_bounds(self) -> None:
        pcb = _make_pcb(footprints=(), width=160.0, height=80.0)
        req = _make_requirements()
        state = build_board_state(pcb, req)

        assert state.board_bounds == (0.0, 0.0, 160.0, 80.0)
        assert state.board_width_mm == 160.0
        assert state.board_height_mm == 80.0

    def test_review_info_propagated(self) -> None:
        pcb = _make_pcb(footprints=())
        req = _make_requirements()
        review = PlacementReview(violations=(), grade="A", summary="ok")
        state = build_board_state(pcb, req, review=review)

        assert state.review_grade == "A"
        assert state.review_violation_count == 0

    def test_review_with_violations(self) -> None:
        pcb = _make_pcb(footprints=())
        req = _make_requirements()
        v = PlacementViolation(
            rule=PlacementViolation.__dataclass_fields__["rule"].type,  # type: ignore[attr-defined]
            severity="minor",
            refs=("R1",),
            message="test",
            current_value=1.0,
            threshold=2.0,
            suggested_position=None,
        )
        # Build a real violation
        from kicad_pipeline.optimization.review_agent import PlacementRule

        v = PlacementViolation(
            rule=PlacementRule.DECOUPLING_DISTANCE,
            severity="minor",
            refs=("R1",),
            message="too far",
            current_value=10.0,
            threshold=5.0,
            suggested_position=None,
        )
        review = PlacementReview(violations=(v,), grade="C", summary="issues")
        state = build_board_state(pcb, req, review=review)

        assert state.review_grade == "C"
        assert state.review_violation_count == 1


class TestOverlapDetection:
    """Test AABB overlap detection."""

    def test_no_overlap(self) -> None:
        fp1 = _make_footprint("R1", "10k", "R_0402", 20.0, 30.0)
        fp2 = _make_footprint("R2", "10k", "R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp1, fp2))
        req = _make_requirements()
        state = build_board_state(pcb, req)

        assert state.overlaps == ()
        for c in state.components:
            assert c.overlapping_refs == ()

    def test_overlap_detected(self) -> None:
        # Two components at same position — guaranteed overlap
        fp1 = _make_footprint("R1", "10k", "R_0402", 30.0, 30.0)
        fp2 = _make_footprint("R2", "10k", "R_0402", 30.5, 30.0)
        pcb = _make_pcb(footprints=(fp1, fp2))
        req = _make_requirements()
        state = build_board_state(pcb, req)

        assert len(state.overlaps) >= 1
        ov = state.overlaps[0]
        assert {ov.ref_a, ov.ref_b} == {"R1", "R2"}
        assert ov.overlap_area_mm2 > 0

        # Check overlapping_refs on components
        r1 = next(c for c in state.components if c.ref == "R1")
        assert "R2" in r1.overlapping_refs

    def test_tiny_overlap_ignored(self) -> None:
        # Components barely touching — overlap < 0.01 mm^2 threshold
        fp1 = _make_footprint("R1", "10k", "R_0402", 30.0, 30.0)
        # Place far enough that pads don't meaningfully overlap
        fp2 = _make_footprint("R2", "10k", "R_0402", 35.0, 30.0)
        pcb = _make_pcb(footprints=(fp1, fp2))
        req = _make_requirements()
        state = build_board_state(pcb, req)

        assert state.overlaps == ()


class TestEdgeViolations:
    """Test off-board and edge proximity detection."""

    def test_component_in_center_no_violation(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        state = build_board_state(pcb, req)

        assert state.edge_violations == ()
        assert state.off_board_count == 0

    def test_component_off_board(self) -> None:
        # Place component at x=-5, outside 0..100 board
        fp = _make_footprint("R1", "10k", "R_0402", -5.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        state = build_board_state(pcb, req)

        assert state.off_board_count > 0
        assert len(state.edge_violations) > 0
        ev = state.edge_violations[0]
        assert ev.ref == "R1"
        assert ev.is_off_board is True

    def test_near_edge_distance(self) -> None:
        # Component near left edge
        fp = _make_footprint("R1", "10k", "R_0402", 2.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        state = build_board_state(pcb, req)

        c = state.components[0]
        assert c.nearest_edge == "left"
        assert c.nearest_edge_distance_mm < 5.0


class TestGroupAndDomainMapping:
    """Test group, zone, domain, and subcircuit assignment."""

    def test_group_assignment(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        features = (
            FeatureBlock(
                name="Power Supply",
                description="Power",
                components=("R1",),
                nets=(),
                subcircuits=(),
            ),
        )
        req = _make_requirements(features=features)
        state = build_board_state(pcb, req)

        assert state.components[0].group_name == "Power Supply"

    def test_domain_from_subcircuit(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.VOLTAGE_DIVIDER,
            refs=("R1",),
            anchor_ref="R1",
            net_connections=("VIN",),
            domain=VoltageDomain.POWER_5V,
        )
        state = build_board_state(pcb, req, subcircuits=(sc,))

        c = state.components[0]
        assert c.voltage_domain == "5v"
        assert "voltage_divider" in c.subcircuit_types

    def test_domain_from_domain_map(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        state = build_board_state(
            pcb, req, domain_map={"R1": VoltageDomain.DIGITAL_3V3}
        )

        assert state.components[0].voltage_domain == "3v3"

    def test_zone_from_placed_group(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        pg = PlacedGroup(
            name="Power",
            zone="power_zone",
            origin=(40.0, 20.0),
            refs=("R1",),
            positions={"R1": (50.0, 30.0)},
            bbox=(40.0, 20.0, 60.0, 40.0),
        )
        state = build_board_state(pcb, req, groups=[pg])

        assert state.components[0].zone_name == "power_zone"


class TestIsolationGaps:
    """Test voltage domain isolation gap calculation."""

    def test_two_domains(self) -> None:
        fp1 = _make_footprint("R1", "10k", "R_0402", 20.0, 30.0)
        fp2 = _make_footprint("R2", "10k", "R_0402", 80.0, 30.0)
        pcb = _make_pcb(footprints=(fp1, fp2))
        req = _make_requirements()
        sc1 = DetectedSubCircuit(
            circuit_type=SubCircuitType.VOLTAGE_DIVIDER,
            refs=("R1",),
            anchor_ref="R1",
            net_connections=(),
            domain=VoltageDomain.VIN_24V,
        )
        sc2 = DetectedSubCircuit(
            circuit_type=SubCircuitType.DECOUPLING,
            refs=("R2",),
            anchor_ref="R2",
            net_connections=(),
            domain=VoltageDomain.DIGITAL_3V3,
        )
        state = build_board_state(pcb, req, subcircuits=(sc1, sc2))

        assert len(state.isolation_gaps) == 1
        gap = state.isolation_gaps[0]
        assert {gap.domain_a, gap.domain_b} == {"24v", "3v3"}
        assert gap.min_gap_mm == pytest.approx(60.0, abs=2.0)
        assert gap.closest_pair == ("R1", "R2")


class TestZoneOccupancy:
    """Test zone utilization calculation."""

    def test_zone_with_components(self) -> None:
        fp1 = _make_footprint("R1", "10k", "R_0402", 25.0, 25.0)
        fp2 = _make_footprint("R2", "10k", "R_0402", 35.0, 25.0)
        pcb = _make_pcb(footprints=(fp1, fp2))
        req = _make_requirements()
        zone = BoardZone(
            name="power",
            rect=(0.0, 0.0, 50.0, 50.0),
            edge_affinity=None,
            groups=(),
        )
        state = build_board_state(pcb, req, zones=[zone])

        assert len(state.zone_occupancy) == 1
        zo = state.zone_occupancy[0]
        assert zo.zone_name == "power"
        assert zo.component_count == 2
        assert zo.utilization_pct > 0


class TestGroupCohesion:
    """Test group spread and density calculation."""

    def test_single_group(self) -> None:
        fp1 = _make_footprint("R1", "10k", "R_0402", 20.0, 30.0)
        fp2 = _make_footprint("R2", "10k", "R_0402", 30.0, 30.0)
        pcb = _make_pcb(footprints=(fp1, fp2))
        features = (
            FeatureBlock(
                name="Power",
                description="Power",
                components=("R1", "R2"),
                nets=(),
                subcircuits=(),
            ),
        )
        req = _make_requirements(features=features)
        state = build_board_state(pcb, req)

        assert len(state.group_cohesion) == 1
        gc = state.group_cohesion[0]
        assert gc.group_name == "Power"
        assert gc.ref_count == 2
        assert gc.spread_mm > 0


class TestConnectorMatingFace:
    """Test connector mating direction detection."""

    def test_connector_gets_mating_face(self) -> None:
        conn = _make_connector("J1", 50.0, 30.0)
        pcb = _make_pcb(footprints=(conn,))
        req = _make_requirements()
        state = build_board_state(pcb, req)

        c = state.components[0]
        assert c.ref == "J1"
        assert c.mating_face != ""  # Should detect a direction

    def test_non_connector_no_mating_face(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        state = build_board_state(pcb, req)

        assert state.components[0].mating_face == ""


# ---------------------------------------------------------------------------
# Tests: text report
# ---------------------------------------------------------------------------

class TestTextReport:
    """Test to_report() output format."""

    def test_report_header(self) -> None:
        pcb = _make_pcb(footprints=(), width=160.0, height=80.0)
        req = _make_requirements()
        state = build_board_state(pcb, req)
        report = state.to_report()

        assert "=== BOARD STATE REPORT ===" in report
        assert "160.0x80.0mm" in report
        assert "0 components" in report

    def test_report_contains_component_map(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        state = build_board_state(pcb, req)
        report = state.to_report()

        assert "COMPONENT MAP" in report
        assert "R1" in report

    def test_report_connector_section(self) -> None:
        conn = _make_connector("J1", 2.0, 30.0)
        pcb = _make_pcb(footprints=(conn,))
        req = _make_requirements()
        state = build_board_state(pcb, req)
        report = state.to_report()

        assert "CONNECTORS" in report
        assert "J1" in report

    def test_report_overlap_section(self) -> None:
        fp1 = _make_footprint("R1", "10k", "R_0402", 30.0, 30.0)
        fp2 = _make_footprint("R2", "10k", "R_0402", 30.5, 30.0)
        pcb = _make_pcb(footprints=(fp1, fp2))
        req = _make_requirements()
        state = build_board_state(pcb, req)
        report = state.to_report()

        assert "OVERLAPS" in report
        assert "CRITICAL ISSUES" in report

    def test_report_no_issues(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        state = build_board_state(pcb, req)
        report = state.to_report()

        assert "NO CRITICAL ISSUES" in report

    def test_report_zones(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", 25.0, 25.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        zone = BoardZone(
            name="power",
            rect=(0.0, 0.0, 50.0, 50.0),
            edge_affinity=None,
            groups=(),
        )
        state = build_board_state(pcb, req, zones=[zone])
        report = state.to_report()

        assert "ZONES" in report
        assert "power" in report

    def test_report_groups(self) -> None:
        fp1 = _make_footprint("R1", "10k", "R_0402", 20.0, 30.0)
        fp2 = _make_footprint("R2", "10k", "R_0402", 30.0, 30.0)
        pcb = _make_pcb(footprints=(fp1, fp2))
        features = (
            FeatureBlock(
                name="MCU",
                description="MCU",
                components=("R1", "R2"),
                nets=(),
                subcircuits=(),
            ),
        )
        req = _make_requirements(features=features)
        state = build_board_state(pcb, req)
        report = state.to_report()

        assert "GROUPS" in report
        assert "MCU" in report

    def test_report_voltage_isolation(self) -> None:
        fp1 = _make_footprint("R1", "10k", "R_0402", 10.0, 30.0)
        fp2 = _make_footprint("R2", "10k", "R_0402", 90.0, 30.0)
        pcb = _make_pcb(footprints=(fp1, fp2))
        req = _make_requirements()
        sc1 = DetectedSubCircuit(
            circuit_type=SubCircuitType.VOLTAGE_DIVIDER,
            refs=("R1",),
            anchor_ref="R1",
            net_connections=(),
            domain=VoltageDomain.VIN_24V,
        )
        sc2 = DetectedSubCircuit(
            circuit_type=SubCircuitType.DECOUPLING,
            refs=("R2",),
            anchor_ref="R2",
            net_connections=(),
            domain=VoltageDomain.DIGITAL_3V3,
        )
        state = build_board_state(pcb, req, subcircuits=(sc1, sc2))
        report = state.to_report()

        assert "VOLTAGE ISOLATION" in report
        assert "24v" in report
        assert "3v3" in report

    def test_report_cell_size(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        state = build_board_state(pcb, req)

        report_5 = state.to_report(cell_size_mm=5.0)
        report_20 = state.to_report(cell_size_mm=20.0)
        # Smaller cells = more grid lines
        assert report_5.count("\n") > report_20.count("\n")

    def test_report_off_board_in_issues(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", -5.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        state = build_board_state(pcb, req)
        report = state.to_report()

        assert "OFF-BOARD" in report
        assert "R1" in report


# ---------------------------------------------------------------------------
# Tests: dataclass immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    """Verify frozen dataclasses."""

    def test_board_state_frozen(self) -> None:
        pcb = _make_pcb(footprints=())
        req = _make_requirements()
        state = build_board_state(pcb, req)
        with pytest.raises(AttributeError):
            state.off_board_count = 99  # type: ignore[misc]

    def test_placed_component_frozen(self) -> None:
        fp = _make_footprint("R1", "10k", "R_0402", 50.0, 30.0)
        pcb = _make_pcb(footprints=(fp,))
        req = _make_requirements()
        state = build_board_state(pcb, req)
        with pytest.raises(AttributeError):
            state.components[0].ref = "R99"  # type: ignore[misc]
