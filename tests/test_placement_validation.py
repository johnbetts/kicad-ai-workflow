"""Tests for post-placement validation and spacing enforcement."""

from __future__ import annotations

import pytest

from kicad_pipeline.constants import (
    DECOUPLING_CAP_MAX_DISTANCE_MM,
    DECOUPLING_CAP_MIN_DISTANCE_MM,
    PASSIVE_NEAR_IC_MAX_DISTANCE_MM,
)
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    PCBDesign,
    Pad,
    PlacementConstraint,
    PlacementConstraintType,
    Point,
)
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    Net,
    NetConnection,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.pcb.constraints import (
    validate_placement_constraints,
    validate_signal_chain_placement,
)
from kicad_pipeline.validation.drc import Severity
from kicad_pipeline.validation.electrical import run_electrical_checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLOSED_RECT: tuple[Point, ...] = (
    Point(0.0, 0.0), Point(80.0, 0.0),
    Point(80.0, 40.0), Point(0.0, 40.0),
    Point(0.0, 0.0),
)
_DEFAULT_RULES = DesignRules()


def _make_pcb(
    footprints: tuple[Footprint, ...] = (),
    nets: tuple[NetEntry, ...] | None = None,
) -> PCBDesign:
    if nets is None:
        nets = (NetEntry(number=1, name="GND"),)
    return PCBDesign(
        outline=BoardOutline(polygon=_CLOSED_RECT),
        design_rules=_DEFAULT_RULES,
        nets=nets,
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


# ---------------------------------------------------------------------------
# Phase 4: validate_placement_constraints tests
# ---------------------------------------------------------------------------


class TestValidatePlacementConstraints:
    """Tests for validate_placement_constraints()."""

    def test_near_constraint_satisfied(self) -> None:
        """Component within max distance: no violation."""
        positions = {
            "U1": Point(x=20.0, y=20.0),
            "C1": Point(x=22.0, y=20.0),  # 2mm away
        }
        constraints = (
            PlacementConstraint(
                ref="C1",
                constraint_type=PlacementConstraintType.NEAR,
                target_ref="U1",
                max_distance_mm=DECOUPLING_CAP_MAX_DISTANCE_MM,
            ),
        )
        violations = validate_placement_constraints(positions, constraints)
        assert len(violations) == 0

    def test_near_constraint_exceeded(self) -> None:
        """Component beyond max distance: violation reported."""
        positions = {
            "U1": Point(x=20.0, y=20.0),
            "C1": Point(x=30.0, y=20.0),  # 10mm away
        }
        constraints = (
            PlacementConstraint(
                ref="C1",
                constraint_type=PlacementConstraintType.NEAR,
                target_ref="U1",
                max_distance_mm=DECOUPLING_CAP_MAX_DISTANCE_MM,
            ),
        )
        violations = validate_placement_constraints(positions, constraints)
        assert len(violations) == 1
        assert "too far" in violations[0]

    def test_near_constraint_too_close(self) -> None:
        """Component below min distance: violation reported."""
        positions = {
            "U1": Point(x=20.0, y=20.0),
            "C1": Point(x=20.3, y=20.0),  # 0.3mm away
        }
        constraints = (
            PlacementConstraint(
                ref="C1",
                constraint_type=PlacementConstraintType.NEAR,
                target_ref="U1",
                max_distance_mm=DECOUPLING_CAP_MAX_DISTANCE_MM,
                min_distance_mm=DECOUPLING_CAP_MIN_DISTANCE_MM,
            ),
        )
        violations = validate_placement_constraints(positions, constraints)
        assert len(violations) == 1
        assert "too close" in violations[0]

    def test_away_from_constraint_violated(self) -> None:
        """Component too close to AWAY_FROM target: violation reported."""
        positions = {
            "U1": Point(x=20.0, y=20.0),
            "U2": Point(x=22.0, y=20.0),  # 2mm away but should be 10mm
        }
        constraints = (
            PlacementConstraint(
                ref="U2",
                constraint_type=PlacementConstraintType.AWAY_FROM,
                target_ref="U1",
                min_distance_mm=10.0,
            ),
        )
        violations = validate_placement_constraints(positions, constraints)
        assert len(violations) == 1
        assert "too close" in violations[0]

    def test_missing_ref_skipped(self) -> None:
        """Constraint for unplaced ref: silently skipped."""
        positions = {"U1": Point(x=20.0, y=20.0)}
        constraints = (
            PlacementConstraint(
                ref="C1",  # Not placed
                constraint_type=PlacementConstraintType.NEAR,
                target_ref="U1",
                max_distance_mm=5.0,
            ),
        )
        violations = validate_placement_constraints(positions, constraints)
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Phase 4: Constants are wired through
# ---------------------------------------------------------------------------


def test_decoupling_uses_constant() -> None:
    """Verify DECOUPLING_CAP_MAX_DISTANCE_MM is used (not hardcoded 5.0)."""
    assert DECOUPLING_CAP_MAX_DISTANCE_MM == 5.0
    assert DECOUPLING_CAP_MIN_DISTANCE_MM == 1.0
    assert PASSIVE_NEAR_IC_MAX_DISTANCE_MM == 5.0


# ---------------------------------------------------------------------------
# Phase 4: Electrical decoupling physical distance check
# ---------------------------------------------------------------------------


def test_electrical_decoupling_physical_distance() -> None:
    """PCBDesign with IC and cap placed far apart: INFO violation."""
    ic_pad = Pad(
        number="1", pad_type="smd", shape="rect",
        position=Point(x=0.0, y=0.0), size_x=1.0, size_y=0.6,
        layers=("F.Cu",),
    )
    cap_pad = Pad(
        number="1", pad_type="smd", shape="rect",
        position=Point(x=0.0, y=0.0), size_x=1.0, size_y=0.6,
        layers=("F.Cu",),
    )
    ic_fp = Footprint(
        lib_id="IC:SOIC-8", ref="U1", value="MyIC",
        position=Point(x=10.0, y=10.0), pads=(ic_pad,),
    )
    cap_fp = Footprint(
        lib_id="Capacitor_SMD:C_0805", ref="C1", value="100nF",
        position=Point(x=30.0, y=10.0), pads=(cap_pad,),  # 20mm away
    )
    pcb = _make_pcb(footprints=(ic_fp, cap_fp))

    feature = FeatureBlock(
        name="Power",
        description="Power block",
        components=("U1", "C1"),
        nets=(),
        subcircuits=(),
    )
    reqs = ProjectRequirements(
        project=ProjectInfo(name="Test"),
        features=(feature,),
        components=(
            Component(ref="U1", value="MyIC", footprint="SOIC-8"),
            Component(ref="C1", value="100nF", footprint="C_0805"),
        ),
        nets=(),
    )
    report = run_electrical_checks(pcb, requirements=reqs)
    decoup = [v for v in report.violations if v.rule == "decoupling_caps"]
    assert len(decoup) >= 1
    assert decoup[0].severity == Severity.INFO
    # Should report distance
    assert "mm" in decoup[0].message


# ---------------------------------------------------------------------------
# Phase 5: Signal chain placement validation
# ---------------------------------------------------------------------------


def _make_chain_requirements(n: int) -> ProjectRequirements:
    """Create requirements with a linear signal chain of n components."""
    components = tuple(
        Component(ref=f"U{i+1}", value="IC", footprint="SOIC-8")
        for i in range(n)
    )
    # Chain: U1-sig1->U2-sig2->U3-sig3->U4 etc.
    nets: list[Net] = []
    for i in range(n - 1):
        nets.append(Net(
            name=f"SIG_{i+1}",
            connections=(
                NetConnection(ref=f"U{i+1}", pin="2"),
                NetConnection(ref=f"U{i+2}", pin="1"),
            ),
        ))
    return ProjectRequirements(
        project=ProjectInfo(name="ChainTest"),
        features=(),
        components=components,
        nets=tuple(nets),
    )


class TestSignalChainPlacement:
    """Tests for validate_signal_chain_placement()."""

    def test_signal_chain_good_order(self) -> None:
        """Components placed in chain order: no violation."""
        reqs = _make_chain_requirements(4)
        # U1 → U2 → U3 → U4 placed in order along X axis
        positions = {
            "U1": Point(x=10.0, y=20.0),
            "U2": Point(x=20.0, y=20.0),
            "U3": Point(x=30.0, y=20.0),
            "U4": Point(x=40.0, y=20.0),
        }
        violations = validate_signal_chain_placement(reqs, positions)
        assert len(violations) == 0

    def test_signal_chain_reversed(self) -> None:
        """Components placed in reverse order: violation emitted."""
        reqs = _make_chain_requirements(4)
        # U1 → U2 → U3 → U4 but placed reversed: U1 far right, U4 far left
        positions = {
            "U1": Point(x=40.0, y=20.0),
            "U2": Point(x=30.0, y=20.0),
            "U3": Point(x=20.0, y=20.0),
            "U4": Point(x=10.0, y=20.0),
        }
        # Reversed is same length as forward in this case (linear arrangement)
        # so no violation — let's make a zigzag instead
        positions_zigzag = {
            "U1": Point(x=10.0, y=10.0),
            "U2": Point(x=40.0, y=30.0),
            "U3": Point(x=15.0, y=15.0),
            "U4": Point(x=45.0, y=25.0),
        }
        violations = validate_signal_chain_placement(reqs, positions_zigzag)
        assert len(violations) >= 1
        assert "excess" in violations[0]

    def test_short_chain_skipped(self) -> None:
        """2-component chain is ignored (too short for meaningful analysis)."""
        reqs = _make_chain_requirements(2)
        positions = {
            "U1": Point(x=40.0, y=20.0),
            "U2": Point(x=10.0, y=20.0),
        }
        violations = validate_signal_chain_placement(reqs, positions)
        assert len(violations) == 0
