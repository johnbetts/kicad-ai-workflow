"""Tests for _phase_constraint_placement in ee_phases."""

from __future__ import annotations

import math

from kicad_pipeline.models.pcb import (
    OrderingChain,
    PlacementConstraintSet,
    ProximityConstraint,
)
from kicad_pipeline.models.requirements import ProjectInfo, ProjectRequirements
from kicad_pipeline.optimization.ee_phases import _phase_constraint_placement
from kicad_pipeline.optimization.placement_types import PlacementContext
from tests.helpers import make_component, make_footprint, make_pcb_design

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    positions: dict[str, tuple[float, float, float]],
    constraints: PlacementConstraintSet | None = None,
    fixed_refs: set[str] | None = None,
) -> PlacementContext:
    """Build a minimal PlacementContext for constraint-phase tests."""
    refs = list(positions)
    comps = tuple(make_component(r) for r in refs)
    reqs = ProjectRequirements(
        project=ProjectInfo(name="test"),
        features=(),
        components=comps,
        nets=(),
    )
    footprints = tuple(
        make_footprint(r, x=positions[r][0], y=positions[r][1], rotation=positions[r][2])
        for r in refs
    )
    pcb = make_pcb_design(footprints=footprints, w=100.0, h=100.0)

    return PlacementContext(
        positions=dict(positions),
        fp_sizes={r: (2.0, 1.0) for r in refs},
        bounds=(0.0, 0.0, 100.0, 100.0),
        fixed_refs=fixed_refs if fixed_refs is not None else set(),
        requirements=reqs,
        initial_pcb=pcb,
        zones=[],
        subcircuits=[],
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConstraintPhaseNoop:
    def test_noop_when_none(self) -> None:
        """Phase does nothing when constraints is None."""
        initial = {"R1": (10.0, 10.0, 0.0), "R2": (50.0, 50.0, 0.0)}
        ctx = _make_ctx(positions=initial, constraints=None)

        _phase_constraint_placement(ctx)

        assert ctx.positions["R1"] == (10.0, 10.0, 0.0)
        assert ctx.positions["R2"] == (50.0, 50.0, 0.0)
        assert len(ctx.fixed_refs) == 0

    def test_noop_when_empty(self) -> None:
        """Phase does nothing when constraint collections are all empty."""
        initial = {"R1": (10.0, 10.0, 0.0), "R2": (50.0, 50.0, 0.0)}
        empty = PlacementConstraintSet()
        ctx = _make_ctx(positions=initial, constraints=empty)

        _phase_constraint_placement(ctx)

        assert ctx.positions["R1"] == (10.0, 10.0, 0.0)
        assert ctx.positions["R2"] == (50.0, 50.0, 0.0)
        assert len(ctx.fixed_refs) == 0


class TestProximityEnforcement:
    def test_component_moved_within_max_distance(self) -> None:
        """Component 20 mm from target with max 5 mm should end up ≤ 5 mm away."""
        initial = {
            "C1": (30.0, 10.0, 0.0),   # 20 mm away from U1
            "U1": (10.0, 10.0, 0.0),   # target
        }
        constraints = PlacementConstraintSet(
            proximity=(
                ProximityConstraint(ref="C1", target_ref="U1", max_distance_mm=5.0),
            ),
        )
        ctx = _make_ctx(positions=initial, constraints=constraints)

        _phase_constraint_placement(ctx)

        cx, cy, _ = ctx.positions["C1"]
        tx, ty, _ = ctx.positions["U1"]
        dist = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
        assert dist <= 5.0 + 1e-6

    def test_already_close_component_not_moved(self) -> None:
        """Component already within max distance should not be moved."""
        initial = {
            "C1": (13.0, 10.0, 0.0),   # 3 mm from U1 — already within 5 mm
            "U1": (10.0, 10.0, 0.0),
        }
        constraints = PlacementConstraintSet(
            proximity=(
                ProximityConstraint(ref="C1", target_ref="U1", max_distance_mm=5.0),
            ),
        )
        ctx = _make_ctx(positions=initial, constraints=constraints)

        _phase_constraint_placement(ctx)

        assert ctx.positions["C1"] == (13.0, 10.0, 0.0)

    def test_missing_refs_skipped(self) -> None:
        """If either ref or target is absent from positions, constraint is skipped."""
        initial = {"U1": (10.0, 10.0, 0.0)}
        constraints = PlacementConstraintSet(
            proximity=(
                ProximityConstraint(ref="C99", target_ref="U1", max_distance_mm=5.0),
            ),
        )
        ctx = _make_ctx(positions=initial, constraints=constraints)

        # Should not raise
        _phase_constraint_placement(ctx)

        assert ctx.positions["U1"] == (10.0, 10.0, 0.0)


class TestOrderingEnforcement:
    def test_components_reordered_along_x_axis(self) -> None:
        """Three components out of X-order are reordered to match declared sequence."""
        # Declared order: R1, R2, R3 (left to right)
        # Initial: R1 at x=50, R2 at x=10, R3 at x=30 → out of order
        initial = {
            "R1": (50.0, 10.0, 0.0),
            "R2": (10.0, 10.0, 0.0),
            "R3": (30.0, 10.0, 0.0),
        }
        constraints = PlacementConstraintSet(
            ordering=(
                OrderingChain(group="divider", refs=("R1", "R2", "R3")),
            ),
        )
        ctx = _make_ctx(positions=initial, constraints=constraints)

        _phase_constraint_placement(ctx)

        x1 = ctx.positions["R1"][0]
        x2 = ctx.positions["R2"][0]
        x3 = ctx.positions["R3"][0]
        assert x1 <= x2 + 1.0
        assert x2 <= x3 + 1.0

    def test_ordering_already_correct_no_change(self) -> None:
        """Components already in order should not be moved."""
        initial = {
            "R1": (10.0, 10.0, 0.0),
            "R2": (20.0, 10.0, 0.0),
            "R3": (30.0, 10.0, 0.0),
        }
        constraints = PlacementConstraintSet(
            ordering=(
                OrderingChain(group="divider", refs=("R1", "R2", "R3")),
            ),
        )
        ctx = _make_ctx(positions=initial, constraints=constraints)

        _phase_constraint_placement(ctx)

        assert ctx.positions["R1"] == (10.0, 10.0, 0.0)
        assert ctx.positions["R2"] == (20.0, 10.0, 0.0)
        assert ctx.positions["R3"] == (30.0, 10.0, 0.0)

    def test_single_ref_chain_skipped(self) -> None:
        """A chain with only one present ref is silently skipped."""
        initial = {"R1": (10.0, 10.0, 0.0)}
        constraints = PlacementConstraintSet(
            ordering=(
                OrderingChain(group="divider", refs=("R1", "R99")),
            ),
        )
        ctx = _make_ctx(positions=initial, constraints=constraints)

        _phase_constraint_placement(ctx)

        assert ctx.positions["R1"] == (10.0, 10.0, 0.0)


class TestRefsAddedToFixed:
    def test_proximity_enforced_refs_become_fixed(self) -> None:
        """After proximity enforcement, moved ref is added to fixed_refs."""
        initial = {
            "C1": (40.0, 10.0, 0.0),   # 30 mm from U1
            "U1": (10.0, 10.0, 0.0),
        }
        constraints = PlacementConstraintSet(
            proximity=(
                ProximityConstraint(ref="C1", target_ref="U1", max_distance_mm=5.0),
            ),
        )
        ctx = _make_ctx(positions=initial, constraints=constraints)

        _phase_constraint_placement(ctx)

        assert "C1" in ctx.fixed_refs

    def test_ordering_enforced_refs_become_fixed(self) -> None:
        """After ordering enforcement, reordered refs are added to fixed_refs."""
        initial = {
            "R1": (30.0, 10.0, 0.0),
            "R2": (10.0, 10.0, 0.0),
        }
        constraints = PlacementConstraintSet(
            ordering=(
                OrderingChain(group="chain", refs=("R1", "R2")),
            ),
        )
        ctx = _make_ctx(positions=initial, constraints=constraints)

        _phase_constraint_placement(ctx)

        assert "R1" in ctx.fixed_refs
        assert "R2" in ctx.fixed_refs

    def test_already_fixed_refs_preserved(self) -> None:
        """Phase must not clear pre-existing fixed_refs."""
        initial = {
            "H1": (5.0, 5.0, 0.0),   # mounting hole, pre-fixed
            "C1": (40.0, 10.0, 0.0),
            "U1": (10.0, 10.0, 0.0),
        }
        constraints = PlacementConstraintSet(
            proximity=(
                ProximityConstraint(ref="C1", target_ref="U1", max_distance_mm=5.0),
            ),
        )
        ctx = _make_ctx(
            positions=initial,
            constraints=constraints,
            fixed_refs={"H1"},
        )

        _phase_constraint_placement(ctx)

        assert "H1" in ctx.fixed_refs
        assert "C1" in ctx.fixed_refs
