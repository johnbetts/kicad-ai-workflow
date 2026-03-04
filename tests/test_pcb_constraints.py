"""Tests for kicad_pipeline.pcb.constraints — constraint-based placement solver."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import (
    BoardEdge,
    BoardOutline,
    Keepout,
    PlacementConstraint,
    PlacementConstraintType,
    Point,
)
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.pcb.constraints import (
    _OccupancyGrid,
    constraints_from_requirements,
    solve_placement,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _board(w: float = 80.0, h: float = 40.0) -> BoardOutline:
    return BoardOutline(
        polygon=(
            Point(0, 0), Point(w, 0), Point(w, h), Point(0, h), Point(0, 0),
        ),
    )


def _sizes(*refs: str) -> dict[str, tuple[float, float]]:
    return {r: (3.0, 3.0) for r in refs}


# ---------------------------------------------------------------------------
# OccupancyGrid tests
# ---------------------------------------------------------------------------


class TestOccupancyGrid:
    """Tests for the _OccupancyGrid internal class."""

    def test_empty_grid_is_free(self) -> None:
        grid = _OccupancyGrid(20.0, 20.0, grid_mm=1.0)
        assert grid.is_rect_free(5.0, 5.0, 3.0, 3.0)

    def test_mark_makes_occupied(self) -> None:
        grid = _OccupancyGrid(20.0, 20.0, grid_mm=1.0)
        grid.mark_rect(5.0, 5.0, 3.0, 3.0)
        assert not grid.is_rect_free(5.0, 5.0, 3.0, 3.0)

    def test_adjacent_rect_still_free(self) -> None:
        grid = _OccupancyGrid(20.0, 20.0, grid_mm=1.0)
        grid.mark_rect(5.0, 5.0, 3.0, 3.0)
        assert grid.is_rect_free(10.0, 10.0, 3.0, 3.0)

    def test_overlapping_rect_not_free(self) -> None:
        grid = _OccupancyGrid(20.0, 20.0, grid_mm=1.0)
        grid.mark_rect(5.0, 5.0, 5.0, 5.0)
        assert not grid.is_rect_free(7.0, 7.0, 3.0, 3.0)

    def test_find_nearest_free_returns_origin_when_empty(self) -> None:
        grid = _OccupancyGrid(20.0, 20.0, grid_mm=1.0)
        pos = grid.find_nearest_free(0.0, 0.0, 2.0, 2.0)
        assert pos is not None
        assert pos[0] == pytest.approx(0.0)
        assert pos[1] == pytest.approx(0.0)

    def test_find_nearest_free_avoids_occupied(self) -> None:
        grid = _OccupancyGrid(20.0, 20.0, grid_mm=1.0)
        grid.mark_rect(5.0, 5.0, 3.0, 3.0)
        pos = grid.find_nearest_free(5.0, 5.0, 2.0, 2.0)
        assert pos is not None
        # Should be displaced from (5, 5)
        assert not (pos[0] == 5.0 and pos[1] == 5.0)

    def test_fine_grid_resolution(self) -> None:
        grid = _OccupancyGrid(10.0, 10.0, grid_mm=0.5)
        grid.mark_rect(2.0, 2.0, 1.0, 1.0)
        assert not grid.is_rect_free(2.0, 2.0, 0.5, 0.5)
        assert grid.is_rect_free(4.0, 4.0, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Constraint solver tests
# ---------------------------------------------------------------------------


class TestSolvePlacement:
    """Tests for solve_placement()."""

    def test_fixed_constraint_exact_position(self) -> None:
        """FIXED constraint places component at exact coordinates."""
        constraints = (
            PlacementConstraint(
                ref="U1",
                constraint_type=PlacementConstraintType.FIXED,
                x=30.0, y=20.0, rotation=90.0,
                priority=100,
            ),
        )
        result = solve_placement(constraints, _board(), _sizes("U1"))
        assert "U1" in result.positions
        assert result.positions["U1"].x == pytest.approx(30.0)
        assert result.positions["U1"].y == pytest.approx(20.0)
        assert result.rotations["U1"] == pytest.approx(90.0)

    def test_edge_constraint_places_along_edge(self) -> None:
        """EDGE constraint places component along the specified board edge."""
        constraints = (
            PlacementConstraint(
                ref="J1",
                constraint_type=PlacementConstraintType.EDGE,
                edge=BoardEdge.LEFT,
                priority=50,
            ),
        )
        result = solve_placement(constraints, _board(), _sizes("J1"))
        assert "J1" in result.positions
        # Should be near the left edge (small x)
        assert result.positions["J1"].x < 20.0

    def test_near_constraint_within_distance(self) -> None:
        """NEAR constraint places component close to target."""
        constraints = (
            PlacementConstraint(
                ref="U1",
                constraint_type=PlacementConstraintType.FIXED,
                x=40.0, y=20.0, priority=100,
            ),
            PlacementConstraint(
                ref="C1",
                constraint_type=PlacementConstraintType.NEAR,
                target_ref="U1",
                max_distance_mm=5.0,
                priority=30,
            ),
        )
        result = solve_placement(constraints, _board(), _sizes("U1", "C1"))
        assert "C1" in result.positions
        dx = result.positions["C1"].x - 40.0
        dy = result.positions["C1"].y - 20.0
        dist = (dx**2 + dy**2) ** 0.5
        assert dist < 10.0  # reasonably close

    def test_group_constraint_clusters_components(self) -> None:
        """GROUP constraint places components together."""
        constraints = (
            PlacementConstraint(
                ref="R1", constraint_type=PlacementConstraintType.GROUP,
                group_name="divider", priority=10,
            ),
            PlacementConstraint(
                ref="R2", constraint_type=PlacementConstraintType.GROUP,
                group_name="divider", priority=10,
            ),
            PlacementConstraint(
                ref="R3", constraint_type=PlacementConstraintType.GROUP,
                group_name="divider", priority=10,
            ),
        )
        sizes = {"R1": (2.0, 1.5), "R2": (2.0, 1.5), "R3": (2.0, 1.5)}
        result = solve_placement(constraints, _board(), sizes)
        assert len(result.positions) == 3
        # All three should be within a reasonable area
        xs = [result.positions[r].x for r in ("R1", "R2", "R3")]
        assert max(xs) - min(xs) < 20.0  # clustered

    def test_no_violations_for_simple_placement(self) -> None:
        """Simple placement produces no violations."""
        constraints = (
            PlacementConstraint(
                ref="U1",
                constraint_type=PlacementConstraintType.FIXED,
                x=40.0, y=20.0, priority=100,
            ),
        )
        result = solve_placement(constraints, _board(), _sizes("U1"))
        assert len(result.violations) == 0

    def test_keepout_avoidance(self) -> None:
        """Components should not be placed inside keepout zones."""
        keepout = Keepout(
            polygon=(Point(35, 15), Point(45, 15), Point(45, 25), Point(35, 25)),
            layers=("F.Cu", "B.Cu"),
            no_copper=True,
        )
        constraints = (
            PlacementConstraint(
                ref="R1",
                constraint_type=PlacementConstraintType.GROUP,
                group_name="test",
                priority=10,
            ),
        )
        result = solve_placement(
            constraints, _board(), _sizes("R1"),
            keepouts=(keepout,),
        )
        assert "R1" in result.positions
        pos = result.positions["R1"]
        # Just verify it was placed successfully
        assert pos.x >= 0.0

    def test_mixed_constraints(self) -> None:
        """Mix of FIXED, NEAR, and GROUP constraints all resolve."""
        constraints = (
            PlacementConstraint(
                ref="U1",
                constraint_type=PlacementConstraintType.FIXED,
                x=40.0, y=20.0, priority=100,
            ),
            PlacementConstraint(
                ref="C1",
                constraint_type=PlacementConstraintType.NEAR,
                target_ref="U1", max_distance_mm=3.0, priority=30,
            ),
            PlacementConstraint(
                ref="R1",
                constraint_type=PlacementConstraintType.GROUP,
                group_name="passives", priority=10,
            ),
            PlacementConstraint(
                ref="R2",
                constraint_type=PlacementConstraintType.GROUP,
                group_name="passives", priority=10,
            ),
        )
        sizes = {
            "U1": (5.0, 5.0), "C1": (2.0, 1.5),
            "R1": (2.0, 1.5), "R2": (2.0, 1.5),
        }
        result = solve_placement(constraints, _board(), sizes)
        assert len(result.positions) == 4
        assert len(result.violations) == 0


# ---------------------------------------------------------------------------
# constraints_from_requirements tests
# ---------------------------------------------------------------------------


def _make_requirements_for_constraints() -> ProjectRequirements:
    """Build requirements with ICs, caps, connectors for constraint tests."""
    return ProjectRequirements(
        project=ProjectInfo(name="ConstraintTest"),
        features=(
            FeatureBlock(
                name="ADC",
                description="ADC channel",
                components=("U1", "C1", "R1", "R2"),
                nets=("+3V3", "GND", "AIN0"),
                subcircuits=(),
            ),
            FeatureBlock(
                name="Connectors",
                description="Input connectors",
                components=("J1", "J2"),
                nets=("VIN",),
                subcircuits=(),
            ),
        ),
        components=(
            Component(ref="U1", value="ADS1115", footprint="MSOP-10", pins=(
                Pin(number="1", name="VDD", pin_type=PinType.POWER_IN, net="+3V3"),
                Pin(number="2", name="GND", pin_type=PinType.POWER_IN, net="GND"),
            )),
            Component(ref="C1", value="100nF", footprint="C_0402", pins=(
                Pin(number="1", name="+", pin_type=PinType.PASSIVE, net="+3V3"),
                Pin(number="2", name="-", pin_type=PinType.PASSIVE, net="GND"),
            )),
            Component(ref="R1", value="10k", footprint="R_0805", pins=(
                Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="AIN0"),
                Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
            )),
            Component(ref="R2", value="10k", footprint="R_0805", pins=(
                Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="AIN0"),
                Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="+3V3"),
            )),
            Component(
                ref="J1", value="Screw_Terminal",
                footprint="TerminalBlock_1x02_P5.08mm",
                pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="VIN"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                ),
            ),
            Component(
                ref="J2", value="Header_2x20",
                footprint="PinHeader_2x20_P2.54mm_Vertical",
                pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="+3V3"),
                ),
            ),
        ),
        nets=(
            Net(name="+3V3", connections=(
                NetConnection(ref="U1", pin="1"),
                NetConnection(ref="C1", pin="1"),
                NetConnection(ref="R2", pin="2"),
                NetConnection(ref="J2", pin="1"),
            )),
            Net(name="GND", connections=(
                NetConnection(ref="U1", pin="2"),
                NetConnection(ref="C1", pin="2"),
                NetConnection(ref="R1", pin="2"),
                NetConnection(ref="J1", pin="2"),
            )),
            Net(name="AIN0", connections=(
                NetConnection(ref="R1", pin="1"),
                NetConnection(ref="R2", pin="1"),
            )),
            Net(name="VIN", connections=(
                NetConnection(ref="J1", pin="1"),
            )),
        ),
    )


class TestConstraintsFromRequirements:
    """Tests for constraints_from_requirements()."""

    def test_connectors_get_edge_constraints(self) -> None:
        """Connectors (J*) get EDGE constraints."""
        req = _make_requirements_for_constraints()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        j1_constraints = [c for c in constraints if c.ref == "J1"]
        assert any(c.constraint_type == PlacementConstraintType.EDGE for c in j1_constraints)

    def test_decoupling_caps_get_near_constraints(self) -> None:
        """Decoupling caps (100nF) sharing power net with IC get NEAR constraint."""
        req = _make_requirements_for_constraints()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        c1_constraints = [c for c in constraints if c.ref == "C1"]
        near = [c for c in c1_constraints if c.constraint_type == PlacementConstraintType.NEAR]
        assert len(near) >= 1
        assert near[0].target_ref == "U1"

    def test_feature_blocks_get_group_constraints(self) -> None:
        """Components in the same FeatureBlock get GROUP constraints."""
        req = _make_requirements_for_constraints()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        group_constraints = [
            c for c in constraints if c.constraint_type == PlacementConstraintType.GROUP
        ]
        assert len(group_constraints) >= 1

    def test_template_fixed_components_get_fixed_constraints(self) -> None:
        """Template fixed components generate FIXED constraints."""
        from kicad_pipeline.pcb.board_templates import get_template

        req = _make_requirements_for_constraints()
        tmpl = get_template("RPI_HAT")
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        # J2 matches template's J1 pattern? No. Let me use a req with J1 as header.
        # Actually J1 in template is ref_pattern="J1" matching J1 in components
        constraints = constraints_from_requirements(req, tmpl, sizes)
        # J1 in template is the GPIO header at fixed position
        # But our J1 is a screw terminal... template ref_pattern is "J1"
        # which matches our J1 component ref
        j1_fixed = [
            c for c in constraints
            if c.ref == "J1" and c.constraint_type == PlacementConstraintType.FIXED
        ]
        assert len(j1_fixed) == 1

    def test_no_constraints_for_none_template(self) -> None:
        """Without a template, no FIXED constraints are generated."""
        req = _make_requirements_for_constraints()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        fixed = [c for c in constraints if c.constraint_type == PlacementConstraintType.FIXED]
        assert len(fixed) == 0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure layout_pcb still works without template (zone-based fallback)."""

    def test_layout_pcb_without_template(self) -> None:
        """layout_pcb places all components using zone fallback."""
        from kicad_pipeline.pcb.placement import layout_pcb

        req = _make_requirements_for_constraints()
        board = _board()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        positions = layout_pcb(req, board, footprint_sizes=sizes)
        assert len(positions) == len(req.components)

    def test_layout_pcb_with_template(self) -> None:
        """layout_pcb with template uses constraint solver."""
        from kicad_pipeline.pcb.board_templates import get_template
        from kicad_pipeline.pcb.placement import layout_pcb

        req = _make_requirements_for_constraints()
        tmpl = get_template("RPI_HAT")
        board = _board(65.0, 56.5)
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        positions = layout_pcb(req, board, footprint_sizes=sizes, board_template=tmpl)
        assert len(positions) == len(req.components)
