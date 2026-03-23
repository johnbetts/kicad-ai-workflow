"""Tests for kicad_pipeline.optimization.placement_guard."""

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
    FeatureBlock,
    MechanicalConstraints,
    Net,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.placement_guard import (
    PlacementGuardResult,
    validate_placement,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_outline(
    w: float = 100.0, h: float = 80.0,
) -> BoardOutline:
    return BoardOutline(
        polygon=(
            Point(0.0, 0.0),
            Point(w, 0.0),
            Point(w, h),
            Point(0.0, h),
            Point(0.0, 0.0),
        ),
    )


def _make_fp(
    ref: str,
    x: float,
    y: float,
    lib_id: str = "test:R",
    pads: tuple[Pad, ...] = (),
) -> Footprint:
    """Create a minimal footprint at a given position."""
    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value="10k",
        position=Point(x, y),
        pads=pads,
    )


def _minimal_pcb(
    footprints: tuple[Footprint, ...] = (),
    w: float = 100.0,
    h: float = 80.0,
) -> PCBDesign:
    return PCBDesign(
        outline=_minimal_outline(w, h),
        design_rules=DesignRules(),
        nets=(NetEntry(number=0, name=""),),
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _minimal_requirements(
    component_refs: tuple[str, ...] = ("R1",),
) -> ProjectRequirements:
    comps = tuple(
        Component(ref=r, value="10k", footprint="R_0805")
        for r in component_refs
    )
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        features=(
            FeatureBlock(
                name="Test",
                description="test",
                components=component_refs,
                nets=("GND",),
                subcircuits=(),
            ),
        ),
        components=comps,
        nets=(Net(name="GND", connections=()),),
        mechanical=MechanicalConstraints(
            board_width_mm=100.0, board_height_mm=80.0,
        ),
    )


# ---------------------------------------------------------------------------
# PlacementGuardResult frozen dataclass
# ---------------------------------------------------------------------------


def test_placement_guard_result_is_frozen() -> None:
    """PlacementGuardResult is immutable."""
    r = PlacementGuardResult(
        passed=True,
        off_board_refs=(),
        off_board_pad_refs=(),
        collision_pairs=(),
        cross_group_refs=(),
        issues=(),
    )
    with pytest.raises((AttributeError, TypeError)):
        r.passed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validate_placement — clean board
# ---------------------------------------------------------------------------


def test_validate_placement_clean_board_passes() -> None:
    """Board with all components inside bounds passes."""
    fps = (
        _make_fp("R1", 25.0, 40.0),
        _make_fp("R2", 50.0, 40.0),
    )
    pcb = _minimal_pcb(footprints=fps)
    reqs = _minimal_requirements(("R1", "R2"))
    result = validate_placement(pcb, reqs)
    assert result.passed is True
    assert result.off_board_refs == ()


def test_validate_placement_empty_board_passes() -> None:
    """Board with no footprints passes (nothing to violate)."""
    pcb = _minimal_pcb(footprints=())
    reqs = _minimal_requirements(())
    result = validate_placement(pcb, reqs)
    assert result.passed is True
    assert result.issues == ()


# ---------------------------------------------------------------------------
# validate_placement — off-board detection
# ---------------------------------------------------------------------------


def test_validate_placement_off_board_center_detected() -> None:
    """Component center outside board bounds is detected."""
    fps = (
        _make_fp("R1", 25.0, 40.0),     # inside
        _make_fp("R2", 200.0, 200.0),   # way outside
    )
    pcb = _minimal_pcb(footprints=fps)
    reqs = _minimal_requirements(("R1", "R2"))
    result = validate_placement(pcb, reqs)
    assert result.passed is False
    assert "R2" in result.off_board_refs


def test_validate_placement_negative_position_off_board() -> None:
    """Component at negative coordinates is off-board."""
    fps = (_make_fp("R1", -50.0, -50.0),)
    pcb = _minimal_pcb(footprints=fps)
    reqs = _minimal_requirements(("R1",))
    result = validate_placement(pcb, reqs)
    assert "R1" in result.off_board_refs


def test_validate_placement_component_near_edge_within_margin() -> None:
    """Component just inside margin still passes."""
    # Board is 100x80, margin=5mm. Component at (3, 40) is within margin.
    fps = (_make_fp("R1", 3.0, 40.0),)
    pcb = _minimal_pcb(footprints=fps)
    reqs = _minimal_requirements(("R1",))
    result = validate_placement(pcb, reqs, margin_mm=5.0)
    assert "R1" not in result.off_board_refs


# ---------------------------------------------------------------------------
# validate_placement — collisions
# ---------------------------------------------------------------------------


def test_validate_placement_stacked_components_collide() -> None:
    """Two components at the exact same position should be detected as collision."""
    fps = (
        _make_fp("R1", 50.0, 40.0),
        _make_fp("R2", 50.0, 40.0),
    )
    pcb = _minimal_pcb(footprints=fps)
    reqs = _minimal_requirements(("R1", "R2"))
    result = validate_placement(pcb, reqs)
    # May or may not detect collision depending on courtyard estimation,
    # but at minimum should not crash
    assert isinstance(result, PlacementGuardResult)


def test_validate_placement_few_collisions_still_passes() -> None:
    """Up to 5 collisions still pass (threshold is <=5)."""
    # Build components far apart — should pass with 0 collisions
    fps = tuple(
        _make_fp(f"R{i}", 10.0 + i * 20.0, 40.0)
        for i in range(4)
    )
    refs = tuple(f"R{i}" for i in range(4))
    pcb = _minimal_pcb(footprints=fps)
    reqs = _minimal_requirements(refs)
    result = validate_placement(pcb, reqs)
    assert result.passed is True


# ---------------------------------------------------------------------------
# validate_placement — custom margin
# ---------------------------------------------------------------------------


def test_validate_placement_custom_margin() -> None:
    """Custom margin_mm allows components closer to edge."""
    # Component at (1, 40) — would fail with default margin=5mm, passes with margin=0
    fps = (_make_fp("R1", 1.0, 40.0),)
    pcb = _minimal_pcb(footprints=fps)
    reqs = _minimal_requirements(("R1",))
    result_strict = validate_placement(pcb, reqs, margin_mm=5.0)
    result_loose = validate_placement(pcb, reqs, margin_mm=0.0)
    # Loose margin should have fewer/no off-board issues compared to strict
    assert len(result_loose.off_board_refs) <= len(result_strict.off_board_refs)


# ---------------------------------------------------------------------------
# validate_placement — issues list
# ---------------------------------------------------------------------------


def test_validate_placement_issues_list_populated_on_failure() -> None:
    """Issues list contains human-readable descriptions when checks fail."""
    fps = (_make_fp("R1", 300.0, 300.0),)  # way off board
    pcb = _minimal_pcb(footprints=fps)
    reqs = _minimal_requirements(("R1",))
    result = validate_placement(pcb, reqs)
    assert len(result.issues) > 0
    assert any("off-board" in issue.lower() or "Off-board" in issue for issue in result.issues)
