"""Tests for enforce_placement — Stage 4 DRC enforcement pass."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import BoardOutline, DesignRules, PCBDesign, Point
from kicad_pipeline.models.requirements import (
    FeatureBlock,
    MechanicalConstraints,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.placement_guard import enforce_placement
from kicad_pipeline.optimization.placement_types import PlacementContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOARD_BOUNDS = (0.0, 0.0, 160.0, 80.0)


def _make_reqs() -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="test", description=""),
        features=(
            FeatureBlock(
                name="Power", description="", components=("R1", "U1", "J1"),
                nets=(), subcircuits=(),
            ),
        ),
        components=(),
        nets=(),
        mechanical=MechanicalConstraints(board_width_mm=160.0, board_height_mm=80.0),
    )


def _make_pcb() -> PCBDesign:
    """Minimal PCBDesign with an outline matching _BOARD_BOUNDS."""
    outline = BoardOutline(
        polygon=(
            Point(x=0.0, y=0.0),
            Point(x=160.0, y=0.0),
            Point(x=160.0, y=80.0),
            Point(x=0.0, y=80.0),
        ),
    )
    return PCBDesign(
        outline=outline,
        design_rules=DesignRules(),
        nets=(),
        footprints=(),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_ctx(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]] | None = None,
    fixed_refs: set[str] | None = None,
) -> PlacementContext:
    if fp_sizes is None:
        fp_sizes = {ref: (2.0, 1.0) for ref in positions}
    return PlacementContext(
        positions=dict(positions),  # mutable copy
        fp_sizes=fp_sizes,
        bounds=_BOARD_BOUNDS,
        fixed_refs=fixed_refs or set(),
        requirements=_make_reqs(),
        initial_pcb=_make_pcb(),
        zones=[],
        subcircuits=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clamp_off_board() -> None:
    """Component at (-10, -10) must be clamped inside board bounds."""
    ctx = _make_ctx({"R1": (-10.0, -10.0, 0.0)})

    result = enforce_placement(ctx)

    x, y, _ = ctx.positions["R1"]
    # Must be inside bounds (with margin + half component size)
    assert x >= 0.0, f"R1 x={x} still off left edge"
    assert y >= 0.0, f"R1 y={y} still off top edge"


def test_collision_pushes_apart() -> None:
    """Two components at the same position must be separated after enforcement."""
    ctx = _make_ctx(
        positions={
            "R1": (80.0, 40.0, 0.0),
            "U1": (80.0, 40.0, 0.0),
        },
        fp_sizes={"R1": (4.0, 2.0), "U1": (8.0, 5.0)},
    )

    enforce_placement(ctx)

    r1x, r1y, _ = ctx.positions["R1"]
    u1x, u1y, _ = ctx.positions["U1"]
    assert (r1x, r1y) != (u1x, u1y), "Colliding components were not pushed apart"


def test_connector_pushed_to_edge() -> None:
    """J1 at board center must be pushed within 5mm of a board edge."""
    ctx = _make_ctx({"J1": (80.0, 40.0, 0.0)})

    enforce_placement(ctx)

    jx, jy, _ = ctx.positions["J1"]
    dist_to_edge = min(
        jx - _BOARD_BOUNDS[0],
        _BOARD_BOUNDS[2] - jx,
        jy - _BOARD_BOUNDS[1],
        _BOARD_BOUNDS[3] - jy,
    )
    assert dist_to_edge <= 5.0, (
        f"J1 at ({jx:.1f}, {jy:.1f}) is {dist_to_edge:.1f}mm from nearest edge"
    )


def test_fixed_refs_not_moved_by_collision() -> None:
    """A fixed ref must not be displaced by collision resolution (RULE-002).

    RULE-001 (clamp) applies unconditionally, but RULE-002 (push-apart)
    respects ``fixed_refs`` — the non-fixed component is moved instead.
    """
    ctx = _make_ctx(
        positions={
            "U1": (80.0, 40.0, 0.0),  # fixed
            "R1": (80.0, 40.0, 0.0),  # overlapping, should be pushed away
        },
        fp_sizes={"U1": (8.0, 5.0), "R1": (4.0, 2.0)},
        fixed_refs={"U1"},
    )
    u1_before = ctx.positions["U1"]

    enforce_placement(ctx)

    assert ctx.positions["U1"] == u1_before, (
        f"Fixed ref U1 was moved from {u1_before} to {ctx.positions['U1']}"
    )
    # R1 should have been pushed away
    r1x, r1y, _ = ctx.positions["R1"]
    u1x, u1y, _ = ctx.positions["U1"]
    assert (r1x, r1y) != (u1x, u1y), "Non-fixed R1 should have been pushed away from U1"
