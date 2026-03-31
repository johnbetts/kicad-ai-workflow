"""Negative tests for DFM gates — deliberately bad PCB data that MUST be rejected.

Proves that the placement DFM gates catch real defects rather than rubber-stamping
everything as passing.  Each test constructs minimal bad data for one specific gate
and asserts ``passed=False``.

These are fast, pure-data tests (no board builds, no I/O).
"""
from __future__ import annotations

from kicad_pipeline.evals.dfm_gates import (
    check_all_within_board,
    check_board_sizing,
    check_no_collisions,
    check_package_match,
    check_schematic_pcb_sync,
)
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.models.requirements import (
    Component,
    ProjectInfo,
    ProjectRequirements,
)

# ---------------------------------------------------------------------------
# Helpers — minimal valid object builders
# ---------------------------------------------------------------------------

def _make_pad(
    number: str = "1",
    x: float = 0.0,
    y: float = 0.0,
    size_x: float = 1.0,
    size_y: float = 0.5,
) -> Pad:
    """Build a minimal SMD pad at a local offset."""
    return Pad(
        number=number,
        pad_type="smd",
        shape="rect",
        position=Point(x, y),
        size_x=size_x,
        size_y=size_y,
        layers=("F.Cu", "F.Paste", "F.Mask"),
    )


def _make_footprint(
    ref: str = "R1",
    lib_id: str = "Resistor_SMD:R_0603_1608Metric",
    value: str = "10k",
    x: float = 10.0,
    y: float = 10.0,
    pads: tuple[Pad, ...] | None = None,
) -> Footprint:
    """Build a minimal footprint with two pads (unless overridden)."""
    if pads is None:
        pads = (
            _make_pad("1", x=-0.5, y=0.0),
            _make_pad("2", x=0.5, y=0.0),
        )
    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(x, y),
        rotation=0.0,
        pads=pads,
    )


def _make_board_outline(
    width: float = 100.0,
    height: float = 80.0,
) -> BoardOutline:
    """Build a rectangular board outline anchored at (0, 0)."""
    return BoardOutline(
        polygon=(
            Point(0.0, 0.0),
            Point(width, 0.0),
            Point(width, height),
            Point(0.0, height),
        ),
    )


def _make_pcb(
    footprints: tuple[Footprint, ...] = (),
    outline: BoardOutline | None = None,
) -> PCBDesign:
    """Build a minimal PCBDesign with given footprints and outline."""
    if outline is None:
        outline = _make_board_outline()
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
    components: tuple[Component, ...] = (),
) -> ProjectRequirements:
    """Build minimal ProjectRequirements with given components."""
    return ProjectRequirements(
        project=ProjectInfo(name="dfm-neg-test"),
        features=(),
        components=components,
        nets=(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollisionDetected:
    """Gate #3: two footprints with overlapping AABBs must be rejected."""

    def test_collision_detected(self) -> None:
        # Place two 0603 resistors at the exact same position so their pad
        # AABBs overlap completely.
        fp_a = _make_footprint(ref="R1", x=10.0, y=10.0)
        fp_b = _make_footprint(ref="R2", x=10.0, y=10.0)

        pcb = _make_pcb(footprints=(fp_a, fp_b))
        reqs = _make_requirements()

        result = check_no_collisions(pcb, reqs)

        assert result.passed is False, (
            f"Expected collision rejection but got passed=True: {result.detail}"
        )
        assert "R1" in result.detail or "R2" in result.detail


class TestOffBoardDetected:
    """Gate #3b: footprint with pads beyond board outline must be rejected."""

    def test_off_board_detected(self) -> None:
        # Board is 100x80 starting at (0,0).
        # Place footprint at (-50, -50) — well outside the board.
        fp_off = _make_footprint(ref="R1", x=-50.0, y=-50.0)

        pcb = _make_pcb(footprints=(fp_off,))
        reqs = _make_requirements()

        result = check_all_within_board(pcb, reqs)

        assert result.passed is False, (
            f"Expected off-board rejection but got passed=True: {result.detail}"
        )
        assert "R1" in result.detail


class TestPackageMismatchDetected:
    """Gate #8: requirements say 0603 but PCB footprint is 0402."""

    def test_package_mismatch_detected(self) -> None:
        # Requirements specify R_0603 footprint.
        comp = Component(
            ref="R1",
            value="10k",
            footprint="R_0603_1608Metric",
        )
        # PCB has R_0402 footprint for the same ref.
        fp = _make_footprint(
            ref="R1",
            lib_id="Resistor_SMD:R_0402_1005Metric",
            x=10.0,
            y=10.0,
        )

        pcb = _make_pcb(footprints=(fp,))
        reqs = _make_requirements(components=(comp,))

        result = check_package_match(pcb, reqs)

        assert result.passed is False, (
            f"Expected package mismatch rejection but got passed=True: {result.detail}"
        )
        assert "R1" in result.detail
        assert "0603" in result.detail
        assert "0402" in result.detail


class TestMissingComponentDetected:
    """Gate #10: requirements has R1 but PCB does not — sync check must fail."""

    def test_missing_component_detected(self) -> None:
        comp = Component(
            ref="R1",
            value="10k",
            footprint="R_0603_1608Metric",
        )

        # PCB has no footprints at all.
        pcb = _make_pcb(footprints=())
        reqs = _make_requirements(components=(comp,))

        result = check_schematic_pcb_sync(pcb, reqs)

        assert result.passed is False, (
            f"Expected missing-component rejection but got passed=True: {result.detail}"
        )
        assert "R1" in result.detail


class TestBoardTooLargeDetected:
    """Gate #4D: 500x500mm board with one tiny component — too large."""

    def test_board_too_large_detected(self) -> None:
        # One tiny 0603 resistor on a 500x500mm board = very low utilization.
        fp = _make_footprint(ref="R1", x=250.0, y=250.0)

        pcb = _make_pcb(
            footprints=(fp,),
            outline=_make_board_outline(width=500.0, height=500.0),
        )
        reqs = _make_requirements()

        result = check_board_sizing(pcb, reqs)

        assert result.passed is False, (
            f"Expected board-too-large rejection but got passed=True: {result.detail}"
        )
        assert "too large" in result.detail.lower()
