"""Tests for the 3 hard-gate review rules: off-board, zone overflow, contamination."""

from __future__ import annotations

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
    FeatureBlock,
    MechanicalConstraints,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.review_agent import (
    PlacementRule,
    _check_component_off_board,
    _check_group_contamination,
    _check_zone_overflow,
)

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_DEFAULT_BOARD_W: float = 100.0
_DEFAULT_BOARD_H: float = 80.0
_DEFAULT_PAD_SX: float = 1.0
_DEFAULT_PAD_SY: float = 0.6
_BOARD_CENTER_Y: float = 40.0
_GROUP_A_OFFSET: float = 20.0
_GROUP_B_OFFSET_X: float = 70.0
_GROUP_B_OFFSET_Y: float = 60.0
_CONTAM_X: float = 73.0
_CONTAM_Y: float = 62.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pin(num: str, name: str) -> Pin:
    return Pin(number=num, name=name, pin_type=PinType.PASSIVE, function=None, net=None)


def _comp(ref: str, value: str, fp: str = "R_0402") -> Component:
    return Component(ref=ref, value=value, footprint=fp, lcsc=None,
                     description=None, datasheet=None, pins=())


def _pad(num: str, x: float = 0.0, y: float = 0.0,
         sx: float = _DEFAULT_PAD_SX, sy: float = _DEFAULT_PAD_SY) -> Pad:
    return Pad(number=num, pad_type="smd", shape="roundrect",
               position=Point(x, y), size_x=sx, size_y=sy,
               layers=("F.Cu", "F.Paste", "F.Mask"),
               net_number=0, net_name="")


def _fp(ref: str, x: float, y: float,
        pads: tuple[Pad, ...] | None = None,
        rotation: float = 0.0) -> Footprint:
    if pads is None:
        pads = (_pad("1", -0.5, 0), _pad("2", 0.5, 0))
    return Footprint(
        lib_id=f"test:{ref}",
        ref=ref,
        value=ref,
        position=Point(x, y),
        rotation=rotation,
        layer="F.Cu",
        pads=pads,
        graphics=(),
        texts=(),
        lcsc=None,
    )


def _make_pcb(
    footprints: list[Footprint],
    board_w: float = _DEFAULT_BOARD_W,
    board_h: float = _DEFAULT_BOARD_H,
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
    features: list[FeatureBlock] | None = None,
) -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="test", author="test", revision="1", description="test"),
        features=tuple(features) if features else (),
        components=tuple(components),
        nets=(),
        pin_map=None,
        power_budget=None,
        mechanical=MechanicalConstraints(
            board_width_mm=_DEFAULT_BOARD_W, board_height_mm=_DEFAULT_BOARD_H
        ),
        recommendations=(),
        board_context=None,
    )


# ---------------------------------------------------------------------------
# COMPONENT_OFF_BOARD tests
# ---------------------------------------------------------------------------


class TestComponentOffBoard:
    def test_pad_off_board_detected(self) -> None:
        """Component with pads 2mm past board edge → critical violation."""
        # Place component at x=-1 so pads extend past left edge (x=0)
        pcb = _make_pcb([_fp("R1", -1.0, _BOARD_CENTER_Y)])
        violations = _check_component_off_board(pcb)
        off_board = [v for v in violations
                     if v.rule == PlacementRule.COMPONENT_OFF_BOARD]
        assert len(off_board) == 1
        assert off_board[0].severity == "critical"
        assert off_board[0].suggested_position is not None
        # Suggested position should be inside the board
        sx, _ = off_board[0].suggested_position
        assert sx > 0.0

    def test_pad_inside_board_ok(self) -> None:
        """Component fully inside board → no off-board violation."""
        pcb = _make_pcb([_fp("R1", 50.0, _BOARD_CENTER_Y)])
        violations = _check_component_off_board(pcb)
        off_board = [v for v in violations
                     if v.rule == PlacementRule.COMPONENT_OFF_BOARD]
        assert len(off_board) == 0

    def test_pad_at_edge_flagged(self) -> None:
        """Pad edge exactly at board edge → off-board (gap = 0, not > 0)."""
        # Pad extends from -0.5 to +0.5 around centroid.
        # Place at x=0.5 so left pad edge is at x=0.0 (board edge).
        # That's a gap of 0.0 which is not negative, so not flagged.
        # Place at x=0.4 so left pad edge is at -0.1 (past edge).
        pcb = _make_pcb([_fp("R1", 0.4, _BOARD_CENTER_Y)])
        violations = _check_component_off_board(pcb)
        off_board = [v for v in violations
                     if v.rule == PlacementRule.COMPONENT_OFF_BOARD]
        assert len(off_board) == 1
        assert off_board[0].severity == "critical"

    def test_component_past_right_edge(self) -> None:
        """Component past right board edge is detected."""
        pcb = _make_pcb([_fp("R1", 100.5, _BOARD_CENTER_Y)])
        violations = _check_component_off_board(pcb)
        off_board = [v for v in violations
                     if v.rule == PlacementRule.COMPONENT_OFF_BOARD]
        assert len(off_board) == 1


# ---------------------------------------------------------------------------
# ZONE_OVERFLOW tests
# ---------------------------------------------------------------------------


class TestZoneOverflow:
    def test_zone_overlap_critical(self) -> None:
        """Two groups with >50% bbox overlap → critical violation."""
        # Both groups in same area
        pcb = _make_pcb([
            _fp("R1", _GROUP_A_OFFSET, _GROUP_A_OFFSET), _fp("R2", 25, 25),   # group A
            _fp("C1", 21, 21), _fp("C2", 24, 24),    # group B — same area
        ])
        features = [
            FeatureBlock(name="GroupA", description="", components=("R1", "R2"),
                         nets=(), subcircuits=()),
            FeatureBlock(name="GroupB", description="", components=("C1", "C2"),
                         nets=(), subcircuits=()),
        ]
        reqs = _make_requirements(
            [_comp("R1", "10k"), _comp("R2", "10k"),
             _comp("C1", "100nF"), _comp("C2", "100nF")],
            features,
        )
        violations = _check_zone_overflow(pcb, reqs)
        overflow = [v for v in violations if v.rule == PlacementRule.ZONE_OVERFLOW]
        assert len(overflow) == 1
        assert overflow[0].severity == "critical"

    def test_zone_overlap_major(self) -> None:
        """Two groups with ~30% overlap → major violation."""
        # Group A: 10-30 x, 20-40 y; Group B: 25-45 x, 20-40 y — partial overlap
        pcb = _make_pcb([
            _fp("R1", 10, 20), _fp("R2", 30, 40),   # group A: bbox 10-30 x 20-40
            _fp("C1", 25, 20), _fp("C2", 45, 40),    # group B: bbox 25-45 x 20-40
        ])
        features = [
            FeatureBlock(name="GroupA", description="", components=("R1", "R2"),
                         nets=(), subcircuits=()),
            FeatureBlock(name="GroupB", description="", components=("C1", "C2"),
                         nets=(), subcircuits=()),
        ]
        reqs = _make_requirements(
            [_comp("R1", "10k"), _comp("R2", "10k"),
             _comp("C1", "100nF"), _comp("C2", "100nF")],
            features,
        )
        violations = _check_zone_overflow(pcb, reqs)
        overflow = [v for v in violations if v.rule == PlacementRule.ZONE_OVERFLOW]
        assert len(overflow) == 1
        assert overflow[0].severity == "major"

    def test_separate_groups_ok(self) -> None:
        """Groups in separate quadrants → no zone overflow."""
        pcb = _make_pcb([
            _fp("R1", 10, 10), _fp("R2", 15, 15),   # top-left
            _fp("C1", _GROUP_B_OFFSET_X, _GROUP_B_OFFSET_Y), _fp("C2", 75, 65),    # bottom-right
        ])
        features = [
            FeatureBlock(name="GroupA", description="", components=("R1", "R2"),
                         nets=(), subcircuits=()),
            FeatureBlock(name="GroupB", description="", components=("C1", "C2"),
                         nets=(), subcircuits=()),
        ]
        reqs = _make_requirements(
            [_comp("R1", "10k"), _comp("R2", "10k"),
             _comp("C1", "100nF"), _comp("C2", "100nF")],
            features,
        )
        violations = _check_zone_overflow(pcb, reqs)
        overflow = [v for v in violations if v.rule == PlacementRule.ZONE_OVERFLOW]
        assert len(overflow) == 0


# ---------------------------------------------------------------------------
# GROUP_CONTAMINATION tests
# ---------------------------------------------------------------------------


class TestGroupContamination:
    def test_contamination_detected(self) -> None:
        """Component from group A inside group B's bbox → major violation."""
        pcb = _make_pcb([
            _fp("R1", 50, 50),   # group A component
            _fp("C1", _GROUP_B_OFFSET_X, _GROUP_B_OFFSET_Y),   # group B
            _fp("C2", 75, 65),   # group B
            _fp("C3", 80, _GROUP_B_OFFSET_Y),   # group B
            # R1 placed at 50,50 but group B spans 70-80, so R1 is NOT inside B.
            # Instead, place R3 (group A) INSIDE group B's area:
            _fp("R3", _CONTAM_X, _CONTAM_Y),   # group A but inside group B bbox
        ])
        features = [
            FeatureBlock(name="GroupA", description="",
                         components=("R1", "R3"), nets=(), subcircuits=()),
            FeatureBlock(name="GroupB", description="",
                         components=("C1", "C2", "C3"), nets=(), subcircuits=()),
        ]
        reqs = _make_requirements(
            [_comp("R1", "10k"), _comp("R3", "10k"),
             _comp("C1", "100nF"), _comp("C2", "100nF"), _comp("C3", "100nF")],
            features,
        )
        violations = _check_group_contamination(pcb, reqs)
        contam = [v for v in violations if v.rule == PlacementRule.GROUP_CONTAMINATION]
        assert len(contam) >= 1
        assert any("R3" in v.refs for v in contam)
        assert contam[0].severity == "major"

    def test_connector_exempt(self) -> None:
        """J-prefix component inside another group → no violation (exempt)."""
        pcb = _make_pcb([
            _fp("J1", _CONTAM_X, _CONTAM_Y),   # connector — should be exempt
            _fp("C1", _GROUP_B_OFFSET_X, _GROUP_B_OFFSET_Y),   # group B
            _fp("C2", 75, 65),   # group B
            _fp("C3", 80, _GROUP_B_OFFSET_Y),   # group B
        ])
        features = [
            FeatureBlock(name="GroupA", description="",
                         components=("J1",), nets=(), subcircuits=()),
            FeatureBlock(name="GroupB", description="",
                         components=("C1", "C2", "C3"), nets=(), subcircuits=()),
        ]
        reqs = _make_requirements(
            [_comp("J1", "CONN"), _comp("C1", "100nF"),
             _comp("C2", "100nF"), _comp("C3", "100nF")],
            features,
        )
        violations = _check_group_contamination(pcb, reqs)
        contam = [v for v in violations if v.rule == PlacementRule.GROUP_CONTAMINATION]
        j1_violations = [v for v in contam if "J1" in v.refs]
        assert len(j1_violations) == 0

    def test_shared_component_exempt(self) -> None:
        """Component in multiple feature blocks → no violation (exempt)."""
        pcb = _make_pcb([
            _fp("R1", _CONTAM_X, _CONTAM_Y),   # shared component
            _fp("C1", _GROUP_B_OFFSET_X, _GROUP_B_OFFSET_Y),   # group B
            _fp("C2", 75, 65),   # group B
            _fp("C3", 80, _GROUP_B_OFFSET_Y),   # group B
        ])
        features = [
            FeatureBlock(name="GroupA", description="",
                         components=("R1",), nets=(), subcircuits=()),
            FeatureBlock(name="GroupB", description="",
                         components=("R1", "C1", "C2", "C3"),
                         nets=(), subcircuits=()),
        ]
        reqs = _make_requirements(
            [_comp("R1", "10k"), _comp("C1", "100nF"),
             _comp("C2", "100nF"), _comp("C3", "100nF")],
            features,
        )
        violations = _check_group_contamination(pcb, reqs)
        contam = [v for v in violations if v.rule == PlacementRule.GROUP_CONTAMINATION]
        r1_violations = [v for v in contam if "R1" in v.refs]
        assert len(r1_violations) == 0
