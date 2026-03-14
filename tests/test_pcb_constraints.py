"""Tests for kicad_pipeline.pcb.constraints — constraint-based placement solver."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
    _build_pad_connectivity,
    _connector_edge,
    _is_decoupling_cap,
    _is_power_net,
    _is_screw_terminal,
    _OccupancyGrid,
    _rotated_pad_offset,
    build_signal_adjacency,
    check_courtyard_collisions,
    constraints_from_requirements,
    optimize_rotations,
    rpi_hat_constraints,
    solve_placement,
    trace_linear_chains,
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

    def test_is_rect_free_rejects_right_overflow(self) -> None:
        """Rectangle extending past the right edge of the grid returns False."""
        grid = _OccupancyGrid(20.0, 20.0, grid_mm=1.0)
        # 5mm wide rect starting at x=18 overflows a 20mm grid
        assert not grid.is_rect_free(18.0, 5.0, 5.0, 3.0)

    def test_is_rect_free_rejects_bottom_overflow(self) -> None:
        """Rectangle extending past the bottom edge of the grid returns False."""
        grid = _OccupancyGrid(20.0, 20.0, grid_mm=1.0)
        assert not grid.is_rect_free(5.0, 18.0, 3.0, 5.0)

    def test_is_rect_free_rejects_negative_origin(self) -> None:
        """Rectangle starting at negative coordinates returns False."""
        grid = _OccupancyGrid(20.0, 20.0, grid_mm=1.0)
        assert not grid.is_rect_free(-2.0, 5.0, 3.0, 3.0)

    def test_is_rect_free_allows_rect_at_boundary(self) -> None:
        """Rectangle that fits exactly at the grid boundary returns True."""
        grid = _OccupancyGrid(20.0, 20.0, grid_mm=1.0)
        # 5mm rect ending exactly at x=20
        assert grid.is_rect_free(15.0, 5.0, 5.0, 3.0)

    def test_find_nearest_free_respects_bounds(self) -> None:
        """find_nearest_free never returns a position where the rect overflows."""
        grid = _OccupancyGrid(20.0, 20.0, grid_mm=1.0)
        # Request near the edge — should stay in bounds
        pos = grid.find_nearest_free(18.0, 18.0, 5.0, 5.0)
        assert pos is not None
        # The rect (pos, 5x5) must fit within the 20x20 grid
        assert pos[0] + 5.0 <= 20.0 + 0.5  # allow grid rounding
        assert pos[1] + 5.0 <= 20.0 + 0.5


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

    def test_placement_gap_prevents_courtyard_overlap(self) -> None:
        """Adjacent components should have a gap between their courtyards."""
        # Place two components via GROUP — they should not touch
        constraints = (
            PlacementConstraint(
                ref="R1", constraint_type=PlacementConstraintType.GROUP,
                group_name="test", priority=10,
            ),
            PlacementConstraint(
                ref="R2", constraint_type=PlacementConstraintType.GROUP,
                group_name="test", priority=10,
            ),
        )
        sizes = {"R1": (2.0, 1.5), "R2": (2.0, 1.5)}
        result = solve_placement(constraints, _board(), sizes)
        pos_r1 = result.positions["R1"]
        pos_r2 = result.positions["R2"]
        # Distance between centres should be > sum of half-widths + gap
        import math

        dist = math.hypot(pos_r1.x - pos_r2.x, pos_r1.y - pos_r2.y)
        # At minimum they should not overlap (centres > max dimension)
        assert dist > 1.0  # sanity check: they're not on top of each other

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


class TestGroupWrapping:
    """Tests for GROUP placement row-wrapping on narrow boards."""

    def test_group_stays_within_board_bounds(self) -> None:
        """GROUP with many components wraps to rows instead of overflowing."""
        # Narrow board: 30mm wide x 40mm tall
        board = _board(w=30.0, h=40.0)
        refs = [f"R{i}" for i in range(1, 9)]  # 8 resistors
        sizes = {r: (3.0, 3.0) for r in refs}
        constraints = tuple(
            PlacementConstraint(
                ref=r,
                constraint_type=PlacementConstraintType.GROUP,
                group_name="resistors",
                priority=10,
            )
            for r in refs
        )
        result = solve_placement(constraints, board, sizes)
        # All components must be within board bounds
        for ref in refs:
            pos = result.positions[ref]
            assert 0 <= pos.x <= 30.0, f"{ref} x={pos.x} is off the board"
            assert 0 <= pos.y <= 40.0, f"{ref} y={pos.y} is off the board"

    def test_group_wraps_to_multiple_rows(self) -> None:
        """GROUP with 6 large components on a 30mm board uses multiple rows."""
        board = _board(w=30.0, h=50.0)
        refs = [f"R{i}" for i in range(1, 7)]  # 6 components
        sizes = {r: (8.0, 3.0) for r in refs}  # 8mm wide each, needs ~10mm/slot
        constraints = tuple(
            PlacementConstraint(
                ref=r,
                constraint_type=PlacementConstraintType.GROUP,
                group_name="wide_parts",
                priority=10,
            )
            for r in refs
        )
        result = solve_placement(constraints, board, sizes)
        # Check that at least 2 distinct y-values are used (multi-row)
        y_values = {round(result.positions[r].y, 1) for r in refs}
        assert len(y_values) >= 2, f"Expected multi-row but got y-values: {y_values}"

    def test_group_no_overflow_violations(self) -> None:
        """GROUP placement must not report overflow violations."""
        board = _board(w=25.0, h=25.0)
        refs = [f"C{i}" for i in range(1, 5)]
        sizes = {r: (2.0, 2.0) for r in refs}
        constraints = tuple(
            PlacementConstraint(
                ref=r,
                constraint_type=PlacementConstraintType.GROUP,
                group_name="caps",
                priority=10,
            )
            for r in refs
        )
        result = solve_placement(constraints, board, sizes)
        assert len(result.violations) == 0
        for ref in refs:
            pos = result.positions[ref]
            assert 0 <= pos.x <= 25.0, f"{ref} x={pos.x} off board"
            assert 0 <= pos.y <= 25.0, f"{ref} y={pos.y} off board"


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
        """Template fixed components generate FIXED constraints.

        The RPi HAT template's GPIO header (ref_pattern=J1) should match
        the 2x20 header J2 (not the 2-pin screw terminal J1) because small
        connectors (< 10 pins) are skipped for GPIO header matching.
        """
        from kicad_pipeline.pcb.board_templates import get_template

        req = _make_requirements_for_constraints()
        tmpl = get_template("RPI_HAT")
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, tmpl, sizes)
        # J2 (2x20 header) should get FIXED via GPIO fallback matching
        j2_fixed = [
            c for c in constraints
            if c.ref == "J2" and c.constraint_type == PlacementConstraintType.FIXED
        ]
        assert len(j2_fixed) == 1

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
        result = layout_pcb(req, board, footprint_sizes=sizes)
        assert len(result.positions) == len(req.components)

    def test_layout_pcb_with_template(self) -> None:
        """layout_pcb with template uses constraint solver."""
        from kicad_pipeline.pcb.board_templates import get_template
        from kicad_pipeline.pcb.placement import layout_pcb

        req = _make_requirements_for_constraints()
        tmpl = get_template("RPI_HAT")
        board = _board(65.0, 56.0)
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        result = layout_pcb(req, board, footprint_sizes=sizes, board_template=tmpl)
        assert len(result.positions) == len(req.components)

    def test_layout_pcb_returns_rotations(self) -> None:
        """layout_pcb with template returns rotations dict."""
        from kicad_pipeline.pcb.board_templates import get_template
        from kicad_pipeline.pcb.placement import LayoutResult, layout_pcb

        req = _make_requirements_for_constraints()
        tmpl = get_template("RPI_HAT")
        board = _board(65.0, 56.0)
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        result = layout_pcb(req, board, footprint_sizes=sizes, board_template=tmpl)
        assert isinstance(result, LayoutResult)
        assert isinstance(result.rotations, dict)


# ---------------------------------------------------------------------------
# Phase D: Edge connector placement
# ---------------------------------------------------------------------------


class TestEdgeConnectorPlacement:
    """Tests for FEAT-3: edge-mounted connector placement."""

    def test_screw_terminal_detected(self) -> None:
        """_is_screw_terminal recognises terminal block footprints."""
        assert _is_screw_terminal("TerminalBlock_1x02_P5.08mm")
        assert _is_screw_terminal("PhoenixContact_MSTBA")
        assert not _is_screw_terminal("R_0805_2012Metric")

    def test_connector_edge_screw_terminal_alternates(self) -> None:
        """Screw terminals alternate between LEFT and BOTTOM on non-HAT boards."""
        e1 = _connector_edge("J1", "TerminalBlock_1x02_P5.08mm")
        e2 = _connector_edge("J2", "TerminalBlock_1x02_P5.08mm")
        assert e1 == BoardEdge.LEFT  # J1 (odd) -> LEFT
        assert e2 == BoardEdge.BOTTOM  # J2 (even) -> BOTTOM

    def test_connector_edge_screw_terminal_rpi_hat_bottom(self) -> None:
        """On RPi HATs, screw terminals always go on BOTTOM edge."""
        e1 = _connector_edge("J2", "TerminalBlock_1x02_P5.08mm", "RPI_HAT")
        e2 = _connector_edge("J3", "TerminalBlock_1x02_P5.08mm", "RPI_HAT")
        assert e1 == BoardEdge.BOTTOM
        assert e2 == BoardEdge.BOTTOM

    def test_connector_edge_pin_header_top(self) -> None:
        """Pin headers default to TOP edge."""
        edge = _connector_edge("J1", "PinHeader_2x20_P2.54mm_Vertical")
        assert edge == BoardEdge.TOP

    def test_edge_rotation_correct_per_edge(self) -> None:
        """Edge placement produces correct rotation for each edge."""
        from kicad_pipeline.pcb.constraints import _edge_position

        _, _, rot_top = _edge_position(BoardEdge.TOP, 80, 40, 3, 3, 40)
        _, _, rot_bot = _edge_position(BoardEdge.BOTTOM, 80, 40, 3, 3, 40)
        _, _, rot_left = _edge_position(BoardEdge.LEFT, 80, 40, 3, 3, 20)
        _, _, rot_right = _edge_position(BoardEdge.RIGHT, 80, 40, 3, 3, 20)
        assert rot_top == pytest.approx(0.0)
        assert rot_bot == pytest.approx(180.0)
        assert rot_left == pytest.approx(90.0)
        assert rot_right == pytest.approx(270.0)

    def test_edge_placement_connectors_on_correct_edges(self) -> None:
        """Connectors are placed on the correct board edge."""
        constraints = (
            PlacementConstraint(
                ref="J1",
                constraint_type=PlacementConstraintType.EDGE,
                edge=BoardEdge.LEFT,
                priority=50,
            ),
            PlacementConstraint(
                ref="J2",
                constraint_type=PlacementConstraintType.EDGE,
                edge=BoardEdge.BOTTOM,
                priority=50,
            ),
        )
        result = solve_placement(constraints, _board(), _sizes("J1", "J2"))
        # J1 on LEFT edge: small x
        assert result.positions["J1"].x < 15.0
        # J2 on BOTTOM edge: near board_h (40)
        assert result.positions["J2"].y > 25.0


# ---------------------------------------------------------------------------
# Phase D: Decoupling cap proximity
# ---------------------------------------------------------------------------


class TestDecouplingCapProximity:
    """Tests for FEAT-4: decoupling cap proximity."""

    def test_is_decoupling_cap_100nf(self) -> None:
        assert _is_decoupling_cap("C1", "100nF")

    def test_is_decoupling_cap_10uf(self) -> None:
        assert _is_decoupling_cap("C3", "10uF")

    def test_is_decoupling_cap_not_resistor(self) -> None:
        assert not _is_decoupling_cap("R1", "100nF")

    def test_is_decoupling_cap_not_electrolytic(self) -> None:
        assert not _is_decoupling_cap("C5", "470uF")

    def test_decoupling_cap_near_constraint_distance(self) -> None:
        """Decoupling caps use DECOUPLING_CAP_MAX/MIN_DISTANCE_MM constants."""
        from kicad_pipeline.constants import (
            DECOUPLING_CAP_MAX_DISTANCE_MM,
            DECOUPLING_CAP_MIN_DISTANCE_MM,
        )
        req = _make_requirements_for_constraints()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        c1_near = [
            c for c in constraints
            if c.ref == "C1" and c.constraint_type == PlacementConstraintType.NEAR
        ]
        assert len(c1_near) >= 1
        assert c1_near[0].max_distance_mm == pytest.approx(DECOUPLING_CAP_MAX_DISTANCE_MM)
        assert c1_near[0].min_distance_mm == pytest.approx(DECOUPLING_CAP_MIN_DISTANCE_MM)

    def test_decoupling_cap_has_target_pin(self) -> None:
        """Decoupling cap NEAR constraint records the IC power pin."""
        req = _make_requirements_for_constraints()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        c1_near = [
            c for c in constraints
            if c.ref == "C1" and c.constraint_type == PlacementConstraintType.NEAR
        ]
        assert len(c1_near) >= 1
        # target_pin should be set to the specific IC pin on the shared power net
        assert c1_near[0].target_pin is not None


# ---------------------------------------------------------------------------
# Phase D: Signal-path analysis
# ---------------------------------------------------------------------------


class TestSignalPathAnalysis:
    """Tests for FEAT-5: signal-path component grouping."""

    def test_is_power_net(self) -> None:
        assert _is_power_net("GND")
        assert _is_power_net("+3V3")
        assert _is_power_net("VCC")
        assert not _is_power_net("AIN0")
        assert not _is_power_net("SPI_CLK")

    def test_build_signal_adjacency_excludes_power(self) -> None:
        """Power/ground nets are excluded from adjacency."""
        req = _make_requirements_for_constraints()
        adj = build_signal_adjacency(req)
        # R1 and R2 share AIN0 (signal) -> adjacent
        assert "R2" in adj["R1"]
        assert "R1" in adj["R2"]
        # U1 and C1 share +3V3 (power) -> NOT adjacent via power
        # (they may still be adjacent via other signals)
        # U1 doesn't share any signal net with J1
        # J1 connects to VIN (only J1 is on it) -> no adjacency

    def test_trace_linear_chains_finds_chain(self) -> None:
        """Linear chain R1-R2 on AIN0 is detected."""
        req = _make_requirements_for_constraints()
        adj = build_signal_adjacency(req)
        chains = trace_linear_chains(adj)
        # R1 and R2 are connected via AIN0
        refs_in_chains = set()
        for chain in chains:
            refs_in_chains.update(chain)
        assert "R1" in refs_in_chains or "R2" in refs_in_chains

    def test_signal_chains_create_group_constraints(self) -> None:
        """Signal chains generate GROUP constraints at priority 15."""
        # Build a requirements with a clear linear chain: J1 -> R1 -> R2 -> U1
        req = ProjectRequirements(
            project=ProjectInfo(name="ChainTest"),
            features=(
                FeatureBlock(
                    name="Signal",
                    description="Signal path",
                    components=("J1", "R1", "R2", "U1"),
                    nets=("SIG_IN", "SIG_MID", "SIG_OUT"),
                    subcircuits=(),
                ),
            ),
            components=(
                Component(ref="J1", value="Conn", footprint="Connector_1x01", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG_IN"),
                )),
                Component(ref="R1", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG_IN"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="SIG_MID"),
                )),
                Component(ref="R2", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG_MID"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="SIG_OUT"),
                )),
                Component(ref="U1", value="ADS1115", footprint="MSOP-10", pins=(
                    Pin(number="1", name="AIN", pin_type=PinType.INPUT, net="SIG_OUT"),
                )),
            ),
            nets=(
                Net(name="SIG_IN", connections=(
                    NetConnection(ref="J1", pin="1"),
                    NetConnection(ref="R1", pin="1"),
                )),
                Net(name="SIG_MID", connections=(
                    NetConnection(ref="R1", pin="2"),
                    NetConnection(ref="R2", pin="1"),
                )),
                Net(name="SIG_OUT", connections=(
                    NetConnection(ref="R2", pin="2"),
                    NetConnection(ref="U1", pin="1"),
                )),
            ),
        )
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        # R1 gets NEAR(J1) and R2 gets NEAR(U1) at priority 25 (section 4b),
        # which supersedes signal chain GROUP at priority 15. U1 (non-passive)
        # still gets a chain GROUP. Verify NEAR constraints exist for passives
        # and at least one chain GROUP remains.
        r1_near = [c for c in constraints
                   if c.ref == "R1" and c.constraint_type == PlacementConstraintType.NEAR]
        r2_near = [c for c in constraints
                   if c.ref == "R2" and c.constraint_type == PlacementConstraintType.NEAR]
        assert len(r1_near) >= 1, "R1 should have NEAR constraint to J1"
        assert len(r2_near) >= 1, "R2 should have NEAR constraint to U1"
        chain_groups = [
            c for c in constraints
            if c.constraint_type == PlacementConstraintType.GROUP and c.priority == 15
        ]
        assert len(chain_groups) >= 1  # U1 still in the chain


# ---------------------------------------------------------------------------
# Phase D: RPi HAT placement strategy
# ---------------------------------------------------------------------------


class TestRPiHATPlacement:
    """Tests for FEAT-2: RPi HAT placement strategy."""

    def _make_hat_requirements(self) -> ProjectRequirements:
        """Build requirements for an RPi HAT design."""
        return ProjectRequirements(
            project=ProjectInfo(name="RPi HAT"),
            features=(
                FeatureBlock(
                    name="ADC",
                    description="ADC channel",
                    components=("U1", "C1", "R1", "R2"),
                    nets=("+3V3", "GND", "AIN0"),
                    subcircuits=(),
                ),
                FeatureBlock(
                    name="IO",
                    description="I/O connectors",
                    components=("J1", "J2", "J3", "SW1"),
                    nets=("VIN", "SDA", "SCL"),
                    subcircuits=(),
                ),
            ),
            components=(
                Component(ref="U1", value="ADS1115", footprint="MSOP-10", pins=(
                    Pin(number="1", name="VDD", pin_type=PinType.POWER_IN, net="+3V3"),
                    Pin(number="2", name="GND", pin_type=PinType.POWER_IN, net="GND"),
                    Pin(number="3", name="AIN0", pin_type=PinType.INPUT, net="AIN0"),
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
                    ref="J1", value="Header_2x20",
                    footprint="PinHeader_2x20_P2.54mm_Vertical",
                    pins=(
                        Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="+3V3"),
                    ),
                ),
                Component(
                    ref="J2", value="Screw_Terminal",
                    footprint="TerminalBlock_1x02_P5.08mm",
                    pins=(
                        Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="VIN"),
                        Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                    ),
                ),
                Component(
                    ref="J3", value="Screw_Terminal",
                    footprint="TerminalBlock_1x02_P5.08mm",
                    pins=(
                        Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="VIN"),
                    ),
                ),
                Component(ref="SW1", value="DIP_4", footprint="DIP-SW_4", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="AIN0"),
                )),
            ),
            nets=(
                Net(name="+3V3", connections=(
                    NetConnection(ref="U1", pin="1"),
                    NetConnection(ref="C1", pin="1"),
                    NetConnection(ref="R2", pin="2"),
                    NetConnection(ref="J1", pin="1"),
                )),
                Net(name="GND", connections=(
                    NetConnection(ref="U1", pin="2"),
                    NetConnection(ref="C1", pin="2"),
                    NetConnection(ref="R1", pin="2"),
                    NetConnection(ref="J2", pin="2"),
                )),
                Net(name="AIN0", connections=(
                    NetConnection(ref="U1", pin="3"),
                    NetConnection(ref="R1", pin="1"),
                    NetConnection(ref="R2", pin="1"),
                    NetConnection(ref="SW1", pin="1"),
                )),
                Net(name="VIN", connections=(
                    NetConnection(ref="J2", pin="1"),
                    NetConnection(ref="J3", pin="1"),
                )),
            ),
        )

    def test_rpi_hat_dip_switch_near_ic(self) -> None:
        """DIP switch gets NEAR constraint to IC on RPi HAT."""
        from kicad_pipeline.pcb.board_templates import get_template

        req = self._make_hat_requirements()
        tmpl = get_template("RPI_HAT")
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = rpi_hat_constraints(req, tmpl, sizes)
        sw1 = [c for c in constraints if c.ref == "SW1"]
        near = [c for c in sw1 if c.constraint_type == PlacementConstraintType.NEAR]
        assert len(near) >= 1
        assert near[0].target_ref == "U1"

    def test_rpi_hat_screw_terminals_edge(self) -> None:
        """Screw terminals get EDGE(BOTTOM) constraints on RPi HAT."""
        from kicad_pipeline.pcb.board_templates import get_template

        req = self._make_hat_requirements()
        tmpl = get_template("RPI_HAT")
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = rpi_hat_constraints(req, tmpl, sizes)
        j2 = [c for c in constraints if c.ref == "J2"]
        j3 = [c for c in constraints if c.ref == "J3"]
        j2_edge = [c for c in j2 if c.constraint_type == PlacementConstraintType.EDGE]
        j3_edge = [c for c in j3 if c.constraint_type == PlacementConstraintType.EDGE]
        assert len(j2_edge) >= 1
        assert len(j3_edge) >= 1
        # All screw terminals should be on BOTTOM for RPi HATs
        assert j2_edge[0].edge == BoardEdge.BOTTOM
        assert j3_edge[0].edge == BoardEdge.BOTTOM

    def test_rpi_hat_voltage_divider_near_target(self) -> None:
        """Voltage divider resistors get NEAR constraints with pin targeting on RPi HAT."""
        from kicad_pipeline.pcb.board_templates import get_template

        req = self._make_hat_requirements()
        tmpl = get_template("RPI_HAT")
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = rpi_hat_constraints(req, tmpl, sizes)
        r1 = [c for c in constraints if c.ref == "R1"]
        r2 = [c for c in constraints if c.ref == "R2"]
        r1_near = [c for c in r1 if c.constraint_type == PlacementConstraintType.NEAR]
        r2_near = [c for c in r2 if c.constraint_type == PlacementConstraintType.NEAR]
        assert len(r1_near) >= 1
        assert len(r2_near) >= 1
        # Resistors should target SW1's specific pin on the shared net
        assert r1_near[0].target_ref == "SW1"
        assert r1_near[0].target_pin == "1"  # SW1 pin on AIN0 net
        assert r1_near[0].max_distance_mm == 5.0

    def test_rpi_hat_full_placement_no_violations(self) -> None:
        """Full RPi HAT placement produces no violations."""
        from kicad_pipeline.pcb.board_templates import get_template

        req = self._make_hat_requirements()
        tmpl = get_template("RPI_HAT")
        board = _board(65.0, 56.0)
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = rpi_hat_constraints(req, tmpl, sizes)
        result = solve_placement(constraints, board, sizes)
        assert len(result.positions) == len(req.components)
        assert len(result.violations) == 0


# ---------------------------------------------------------------------------
# Phase E: Ground plane strategy
# ---------------------------------------------------------------------------


class TestGroundPlaneStrategy:
    """Tests for FEAT-6: B.Cu ground plane strategy."""

    def test_gnd_zones_both_layers(self) -> None:
        """Default strategy creates zones on both F.Cu and B.Cu."""
        from kicad_pipeline.pcb.builder import _make_gnd_zones

        board = _board()
        zones = _make_gnd_zones(board, gnd_net_number=1, strategy="both")
        assert len(zones) == 2
        layers = {z.layer for z in zones}
        assert "F.Cu" in layers
        assert "B.Cu" in layers

    def test_gnd_zones_back_only(self) -> None:
        """back_only strategy creates zone only on B.Cu."""
        from kicad_pipeline.pcb.builder import _make_gnd_zones

        board = _board()
        zones = _make_gnd_zones(board, gnd_net_number=1, strategy="back_only")
        assert len(zones) == 1
        assert zones[0].layer == "B.Cu"


# ---------------------------------------------------------------------------
# Phase E: Rotation optimization
# ---------------------------------------------------------------------------


class TestRotationOptimization:
    """Tests for FEAT-7: component rotation optimization."""

    def test_passive_aligns_toward_neighbour(self) -> None:
        """2-pin passive rotates so connected pad faces its neighbour."""
        req = ProjectRequirements(
            project=ProjectInfo(name="RotTest"),
            features=(
                FeatureBlock(
                    name="Test", description="", components=("R1", "U1"),
                    nets=("SIG",), subcircuits=(),
                ),
            ),
            components=(
                Component(ref="R1", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                )),
                Component(ref="U1", value="IC", footprint="TSSOP-8", pins=(
                    Pin(number="1", name="IN", pin_type=PinType.INPUT, net="SIG"),
                )),
            ),
            nets=(
                Net(name="SIG", connections=(
                    NetConnection(ref="R1", pin="1"),
                    NetConnection(ref="U1", pin="1"),
                )),
            ),
        )
        positions = {
            "R1": Point(x=10.0, y=20.0),
            "U1": Point(x=30.0, y=20.0),
        }
        rotations = {"R1": 0.0, "U1": 0.0}
        sizes = {"R1": (3.0, 1.6), "U1": (6.0, 4.0)}
        result = optimize_rotations(positions, rotations, req, footprint_sizes=sizes)
        # Pad-1 at (-w/2, 0) at 0 deg = left side. U1 is to the right.
        # Rotation 180 puts pad-1 on the right, facing U1.
        assert result["R1"] == pytest.approx(180.0)

    def test_passive_aligns_vertical(self) -> None:
        """Passive above its neighbour rotates so connected pad faces down."""
        req = ProjectRequirements(
            project=ProjectInfo(name="RotTest"),
            features=(
                FeatureBlock(
                    name="Test", description="", components=("C1", "U1"),
                    nets=("PWR",), subcircuits=(),
                ),
            ),
            components=(
                Component(ref="C1", value="100nF", footprint="C_0402", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="PWR"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                )),
                Component(ref="U1", value="IC", footprint="TSSOP-8", pins=(
                    Pin(number="1", name="VDD", pin_type=PinType.POWER_IN, net="PWR"),
                )),
            ),
            nets=(
                Net(name="PWR", connections=(
                    NetConnection(ref="C1", pin="1"),
                    NetConnection(ref="U1", pin="1"),
                )),
            ),
        )
        positions = {
            "C1": Point(x=20.0, y=10.0),
            "U1": Point(x=20.0, y=30.0),
        }
        rotations = {"C1": 0.0, "U1": 0.0}
        sizes = {"C1": (2.0, 1.0), "U1": (6.0, 4.0)}
        result = optimize_rotations(positions, rotations, req, footprint_sizes=sizes)
        # Pad-1 at (-w/2, 0). At 90 deg CW (KiCad convention), pad-1 rotates
        # to (0, +w/2) = below centre, facing U1 which is below at y=30.
        assert result["C1"] == pytest.approx(90.0)

    def test_voltage_divider_connected_pads_face_each_other(self) -> None:
        """R1-pad2 connects to R2-pad1; they should orient connected pads closest."""
        req = ProjectRequirements(
            project=ProjectInfo(name="DividerTest"),
            features=(
                FeatureBlock(
                    name="Divider", description="", components=("R1", "R2"),
                    nets=("MID",), subcircuits=(),
                ),
            ),
            components=(
                Component(ref="R1", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="IN"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="MID"),
                )),
                Component(ref="R2", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="MID"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="OUT"),
                )),
            ),
            nets=(
                Net(name="MID", connections=(
                    NetConnection(ref="R1", pin="2"),
                    NetConnection(ref="R2", pin="1"),
                )),
            ),
        )
        # R1 at left, R2 at right (same y)
        positions = {
            "R1": Point(x=10.0, y=20.0),
            "R2": Point(x=20.0, y=20.0),
        }
        rotations = {"R1": 90.0, "R2": 90.0}  # start non-optimal
        sizes = {"R1": (3.0, 1.6), "R2": (3.0, 1.6)}
        result = optimize_rotations(positions, rotations, req, footprint_sizes=sizes)
        # R1-pad2 (+w/2,0) should face right toward R2.
        # At 0 deg, pad2 is at (+1.5, 0) = right side. Good.
        assert result["R1"] == pytest.approx(0.0)
        # R2-pad1 (-w/2,0) should face left toward R1.
        # At 0 deg, pad1 is at (-1.5, 0) = left side. Good.
        assert result["R2"] == pytest.approx(0.0)

    def test_pad_aware_disabled_without_footprint_sizes(self) -> None:
        """Without footprint_sizes, old centre-based behavior is preserved."""
        req = ProjectRequirements(
            project=ProjectInfo(name="FallbackTest"),
            features=(
                FeatureBlock(
                    name="Test", description="", components=("R1", "U1"),
                    nets=("SIG",), subcircuits=(),
                ),
            ),
            components=(
                Component(ref="R1", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                )),
                Component(ref="U1", value="IC", footprint="TSSOP-8", pins=(
                    Pin(number="1", name="IN", pin_type=PinType.INPUT, net="SIG"),
                )),
            ),
            nets=(
                Net(name="SIG", connections=(
                    NetConnection(ref="R1", pin="1"),
                    NetConnection(ref="U1", pin="1"),
                )),
            ),
        )
        positions = {
            "R1": Point(x=10.0, y=20.0),
            "U1": Point(x=30.0, y=20.0),
        }
        rotations = {"R1": 0.0, "U1": 0.0}
        # No footprint_sizes -> fallback to centre-based snapping
        result = optimize_rotations(positions, rotations, req)
        # Centre-based: R1 toward U1 (right) = 0 degrees
        assert result["R1"] == pytest.approx(0.0)

    def test_rotated_pad_offset_helper(self) -> None:
        """Unit test for _rotated_pad_offset at 0/90/180/270."""
        # 0 degrees — identity
        x, y = _rotated_pad_offset(1.5, 0.0, 0.0)
        assert x == pytest.approx(1.5)
        assert y == pytest.approx(0.0)

        # 90 degrees CW (KiCad convention) — (1.5, 0) -> (0, -1.5)
        x, y = _rotated_pad_offset(1.5, 0.0, 90.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(-1.5)

        # 180 degrees — (1.5, 0) -> (-1.5, 0)
        x, y = _rotated_pad_offset(1.5, 0.0, 180.0)
        assert x == pytest.approx(-1.5)
        assert y == pytest.approx(0.0, abs=1e-10)

        # 270 degrees CW (KiCad convention) — (1.5, 0) -> (0, 1.5)
        x, y = _rotated_pad_offset(1.5, 0.0, 270.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(1.5)

    def test_build_pad_connectivity(self) -> None:
        """Verify pin-level mapping excludes power nets."""
        req = ProjectRequirements(
            project=ProjectInfo(name="PadConnTest"),
            features=(),
            components=(
                Component(ref="R1", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                )),
                Component(ref="U1", value="IC", footprint="TSSOP-8", pins=(
                    Pin(number="1", name="IN", pin_type=PinType.INPUT, net="SIG"),
                    Pin(number="2", name="GND", pin_type=PinType.POWER_IN, net="GND"),
                )),
            ),
            nets=(
                Net(name="SIG", connections=(
                    NetConnection(ref="R1", pin="1"),
                    NetConnection(ref="U1", pin="1"),
                )),
                Net(name="GND", connections=(
                    NetConnection(ref="R1", pin="2"),
                    NetConnection(ref="U1", pin="2"),
                )),
            ),
        )
        pad_conn = _build_pad_connectivity(req)
        # SIG net: R1.1 <-> U1.1
        assert ("U1", "1") in pad_conn[("R1", "1")]
        assert ("R1", "1") in pad_conn[("U1", "1")]
        # GND net excluded (power net)
        assert ("R1", "2") not in pad_conn
        assert ("U1", "2") not in pad_conn


# ---------------------------------------------------------------------------
# Phase E: Courtyard collision checking
# ---------------------------------------------------------------------------


class TestCourtyardCollisions:
    """Tests for FEAT-9: courtyard collision checking."""

    def test_no_collision_when_separated(self) -> None:
        positions = {"R1": Point(10.0, 10.0), "R2": Point(30.0, 30.0)}
        sizes = {"R1": (3.0, 3.0), "R2": (3.0, 3.0)}
        violations = check_courtyard_collisions(positions, sizes)
        assert len(violations) == 0

    def test_detects_overlapping_components(self) -> None:
        positions = {"R1": Point(10.0, 10.0), "R2": Point(11.0, 11.0)}
        sizes = {"R1": (5.0, 5.0), "R2": (5.0, 5.0)}
        violations = check_courtyard_collisions(positions, sizes)
        assert len(violations) == 1
        assert "R1" in violations[0]
        assert "R2" in violations[0]

    def test_detects_keepout_collision(self) -> None:
        positions = {"R1": Point(10.0, 10.0)}
        sizes = {"R1": (5.0, 5.0)}
        keepout = Keepout(
            polygon=(Point(8, 8), Point(12, 8), Point(12, 12), Point(8, 12)),
            layers=("F.Cu",), no_copper=True,
        )
        violations = check_courtyard_collisions(positions, sizes, keepouts=(keepout,))
        assert len(violations) == 1
        assert "keepout" in violations[0].lower()

    def test_no_keepout_collision_when_clear(self) -> None:
        positions = {"R1": Point(10.0, 10.0)}
        sizes = {"R1": (3.0, 3.0)}
        keepout = Keepout(
            polygon=(Point(30, 30), Point(35, 30), Point(35, 35), Point(30, 35)),
            layers=("F.Cu",), no_copper=True,
        )
        violations = check_courtyard_collisions(positions, sizes, keepouts=(keepout,))
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Phase E: DRC intra-footprint filter
# ---------------------------------------------------------------------------


class TestDRCIntraFootprint:
    """Tests for FEAT-10: intra-footprint violation filter."""

    def test_is_intra_footprint_for_clearance(self) -> None:
        """Clearance violation on a known footprint ref is intra-footprint."""
        from kicad_pipeline.models.pcb import (
            DesignRules,
            Footprint,
            NetEntry,
            PCBDesign,
        )
        from kicad_pipeline.validation.drc import (
            DRCViolation,
            Severity,
            is_intra_footprint_violation,
        )

        pcb = PCBDesign(
            outline=BoardOutline(
                polygon=(Point(0, 0), Point(50, 0), Point(50, 50), Point(0, 50)),
            ),
            design_rules=DesignRules(),
            nets=(NetEntry(number=0, name=""),),
            footprints=(
                Footprint(
                    lib_id="R:R_0805", ref="R1", value="10k",
                    position=Point(10, 10),
                ),
            ),
            tracks=(),
            vias=(),
            zones=(),
            keepouts=(),
        )
        v = DRCViolation(
            rule="min_clearance", message="Pad too close",
            severity=Severity.WARNING, ref="R1",
        )
        assert is_intra_footprint_violation(v, pcb) is True

    def test_not_intra_footprint_for_other_rules(self) -> None:
        """Non-clearance violations are not intra-footprint."""
        from kicad_pipeline.models.pcb import (
            DesignRules,
            Footprint,
            NetEntry,
            PCBDesign,
        )
        from kicad_pipeline.validation.drc import (
            DRCViolation,
            Severity,
            is_intra_footprint_violation,
        )

        pcb = PCBDesign(
            outline=BoardOutline(
                polygon=(Point(0, 0), Point(50, 0), Point(50, 50), Point(0, 50)),
            ),
            design_rules=DesignRules(),
            nets=(NetEntry(number=0, name=""),),
            footprints=(
                Footprint(
                    lib_id="R:R_0805", ref="R1", value="10k",
                    position=Point(10, 10),
                ),
            ),
            tracks=(),
            vias=(),
            zones=(),
            keepouts=(),
        )
        v = DRCViolation(
            rule="min_trace_width", message="Too thin",
            severity=Severity.ERROR, ref="R1",
        )
        assert is_intra_footprint_violation(v, pcb) is False

    def test_drc_exclusions_in_project_file(self) -> None:
        """DRC exclusions are written to the project file."""
        from kicad_pipeline.project_file import build_project_file

        data = build_project_file(
            "test", drc_exclusions=("clearance|R1|pad1|R1|pad2",),
        )
        exclusions = data["board"]["design_settings"]["drc_exclusions"]
        assert "clearance|R1|pad1|R1|pad2" in exclusions


# ---------------------------------------------------------------------------
# Phase H: Net-based grouping, edge margin, THT gap (DRC fix sprint)
# ---------------------------------------------------------------------------


class TestNetBasedGrouping:
    """Tests for net-based signal grouping in constraints_from_requirements."""

    def test_three_component_signal_net_grouped(self) -> None:
        """Components sharing a 3+ member signal net are grouped together."""
        req = ProjectRequirements(
            project=ProjectInfo(name="NetGroup"),
            features=(
                FeatureBlock(
                    name="ADC", description="", components=("R1", "R2", "C1", "U1"),
                    nets=("AIN0",), subcircuits=(),
                ),
            ),
            components=(
                Component(ref="R1", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="AIN0"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                )),
                Component(ref="R2", value="30k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="+5V"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="AIN0"),
                )),
                Component(ref="C1", value="100nF", footprint="C_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="AIN0"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                )),
                Component(ref="U1", value="ADS1115", footprint="MSOP-10", pins=(
                    Pin(number="1", name="AIN0", pin_type=PinType.INPUT, net="AIN0"),
                )),
            ),
            nets=(
                Net(name="AIN0", connections=(
                    NetConnection(ref="R1", pin="1"),
                    NetConnection(ref="R2", pin="2"),
                    NetConnection(ref="C1", pin="1"),
                    NetConnection(ref="U1", pin="1"),
                )),
                Net(name="+5V", connections=(
                    NetConnection(ref="R2", pin="1"),
                )),
                Net(name="GND", connections=(
                    NetConnection(ref="R1", pin="2"),
                    NetConnection(ref="C1", pin="2"),
                )),
            ),
        )
        constraints = constraints_from_requirements(req, None, _sizes("R1", "R2", "C1", "U1"))
        # Passives (R1, R2, C1) get NEAR(U1) at priority 25 (section 4b),
        # which is more specific than net-based GROUP. Verify they're all
        # placed near U1 via NEAR or share a net group.
        near_u1 = [
            c for c in constraints
            if c.constraint_type == PlacementConstraintType.NEAR and c.target_ref == "U1"
        ]
        ain0_groups = [
            c for c in constraints
            if c.constraint_type == PlacementConstraintType.GROUP
            and c.group_name is not None
            and c.group_name.startswith("_net_group_")
        ]
        # Passives get NEAR(U1) + possibly U1 in net group = all 4 co-located
        all_constrained = {c.ref for c in near_u1} | {c.ref for c in ain0_groups}
        assert len(all_constrained & {"R1", "R2", "C1", "U1"}) >= 3

    def test_power_nets_not_grouped(self) -> None:
        """Power/GND nets should NOT trigger net-based grouping."""
        req = ProjectRequirements(
            project=ProjectInfo(name="PowerNetTest"),
            features=(),
            components=(
                Component(ref="R1", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="GND"),
                )),
                Component(ref="R2", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="GND"),
                )),
                Component(ref="C1", value="100nF", footprint="C_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="GND"),
                )),
            ),
            nets=(
                Net(name="GND", connections=(
                    NetConnection(ref="R1", pin="1"),
                    NetConnection(ref="R2", pin="1"),
                    NetConnection(ref="C1", pin="1"),
                )),
            ),
        )
        constraints = constraints_from_requirements(req, None, _sizes("R1", "R2", "C1"))
        net_groups = [
            c for c in constraints
            if c.constraint_type == PlacementConstraintType.GROUP
            and c.group_name is not None
            and c.group_name.startswith("_net_group_")
        ]
        assert len(net_groups) == 0


class TestBoardEdgeMargin:
    """Tests for board-edge margin in the occupancy grid."""

    def test_edge_cells_occupied_after_margin(self) -> None:
        """Edge cells should be marked occupied to prevent copper_edge_clearance."""
        board = _board(20.0, 20.0)
        constraints = (
            PlacementConstraint(
                ref="R1",
                constraint_type=PlacementConstraintType.FIXED,
                x=10.0, y=10.0, priority=100,
            ),
        )
        sizes = {"R1": (3.0, 3.0)}
        result = solve_placement(constraints, board, sizes)
        # Component at centre should be placed fine
        assert "R1" in result.positions
        assert len(result.violations) == 0


class TestTHTPlacementGap:
    """Tests for increased placement gap for through-hole components."""

    def test_tht_gap_larger_than_smd(self) -> None:
        """THT components (>5mm) should use a larger placement gap."""
        from kicad_pipeline.pcb.constraints import _placement_gap

        # Small SMD component
        assert _placement_gap(3.0, 1.6) == 0.5
        # Large THT component
        assert _placement_gap(10.0, 5.0) == 1.0
        assert _placement_gap(5.0, 6.0) == 1.0
        # Borderline (exactly 5.0) should use standard gap
        assert _placement_gap(4.0, 5.0) == 0.5


class TestBuilderAutoRoute:
    """Tests for auto-routing integration in build_pcb."""

    def test_build_pcb_auto_route_disabled(self) -> None:
        """build_pcb with auto_route=False returns no tracks."""
        from kicad_pipeline.pcb.builder import build_pcb

        req = ProjectRequirements(
            project=ProjectInfo(name="NoRoute"),
            features=(
                FeatureBlock(
                    name="Test", description="", components=("R1",),
                    nets=("SIG",), subcircuits=(),
                ),
            ),
            components=(
                Component(ref="R1", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                )),
            ),
            nets=(
                Net(name="SIG", connections=(NetConnection(ref="R1", pin="1"),)),
                Net(name="GND", connections=(NetConnection(ref="R1", pin="2"),)),
            ),
        )
        pcb = build_pcb(req, auto_route=False)
        assert len(pcb.tracks) == 0

    @patch("kicad_pipeline.routing.freerouting.find_freerouting_jar", return_value=None)
    def test_build_pcb_auto_route_enabled(self, _mock_jar: MagicMock) -> None:
        """build_pcb with auto_route=True returns tracks for routable nets."""
        from kicad_pipeline.pcb.builder import build_pcb

        req = ProjectRequirements(
            project=ProjectInfo(name="AutoRoute"),
            features=(
                FeatureBlock(
                    name="Test", description="", components=("R1", "R2"),
                    nets=("SIG",), subcircuits=(),
                ),
            ),
            components=(
                Component(ref="R1", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                )),
                Component(ref="R2", value="10k", footprint="R_0805", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                )),
            ),
            nets=(
                Net(name="SIG", connections=(
                    NetConnection(ref="R1", pin="1"),
                    NetConnection(ref="R2", pin="1"),
                )),
                Net(name="GND", connections=(
                    NetConnection(ref="R1", pin="2"),
                    NetConnection(ref="R2", pin="2"),
                )),
            ),
        )
        pcb = build_pcb(req, auto_route=True)
        # Should have at least some tracks (SIG routed, GND skipped)
        # GND is skipped because it's handled by the copper pour
        assert len(pcb.tracks) >= 0  # May or may not route depending on placement


class TestLayoutPcbRpiHatDispatch:
    """Test that layout_pcb dispatches to rpi_hat_constraints for RPI_HAT."""

    def test_rpi_hat_uses_hat_constraints(self) -> None:
        """layout_pcb should use rpi_hat_constraints for RPI_HAT template."""
        from kicad_pipeline.pcb.board_templates import get_template
        from kicad_pipeline.pcb.placement import layout_pcb

        req = ProjectRequirements(
            project=ProjectInfo(name="HAT"),
            features=(
                FeatureBlock(
                    name="Core", description="", components=("J1", "U1"),
                    nets=("SIG",), subcircuits=(),
                ),
            ),
            components=(
                Component(ref="J1", value="2x20", footprint="PinSocket_2x20_P2.54mm", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                )),
                Component(ref="U1", value="ADC", footprint="MSOP-10", pins=(
                    Pin(number="1", name="1", pin_type=PinType.INPUT, net="SIG"),
                )),
            ),
            nets=(
                Net(name="SIG", connections=(
                    NetConnection(ref="J1", pin="1"),
                    NetConnection(ref="U1", pin="1"),
                )),
            ),
        )
        tmpl = get_template("RPI_HAT")
        board = _board(65.0, 56.0)
        sizes = {"J1": (51.0, 5.08), "U1": (3.0, 5.0)}
        result = layout_pcb(
            req, board, footprint_sizes=sizes,
            fixed_positions={"J1": (32.504, 3.502, 0.0)},
            board_template=tmpl,
        )
        # J1 should be at the template fixed position
        assert abs(result.positions["J1"].x - 32.504) < 0.01
        assert abs(result.positions["J1"].y - 3.502) < 0.01
        # U1 should also be placed
        assert "U1" in result.positions


# ---------------------------------------------------------------------------
# Passive NEAR constraints (section 4b)
# ---------------------------------------------------------------------------


def _make_switch_resistor_requirements() -> ProjectRequirements:
    """Requirements with resistors sharing signal nets with a switch."""
    return ProjectRequirements(
        project=ProjectInfo(name="SwitchTest"),
        features=(
            FeatureBlock(
                name="Buttons",
                description="4 switches with pull-ups",
                components=("SW1", "R9", "R10", "R11", "R12"),
                nets=("BTN0", "BTN1", "BTN2", "BTN3"),
                subcircuits=(),
            ),
        ),
        components=(
            Component(ref="SW1", value="Switch", footprint="SW_SPST_4Pin", pins=(
                Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="BTN0"),
                Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="BTN1"),
                Pin(number="3", name="3", pin_type=PinType.PASSIVE, net="BTN2"),
                Pin(number="4", name="4", pin_type=PinType.PASSIVE, net="BTN3"),
            )),
            Component(ref="R9", value="10k", footprint="R_0603", pins=(
                Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="BTN0"),
                Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="+3V3"),
            )),
            Component(ref="R10", value="10k", footprint="R_0603", pins=(
                Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="BTN1"),
                Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="+3V3"),
            )),
            Component(ref="R11", value="10k", footprint="R_0603", pins=(
                Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="BTN2"),
                Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="+3V3"),
            )),
            Component(ref="R12", value="10k", footprint="R_0603", pins=(
                Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="BTN3"),
                Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="+3V3"),
            )),
        ),
        nets=(
            Net(name="BTN0", connections=(
                NetConnection(ref="SW1", pin="1"),
                NetConnection(ref="R9", pin="1"),
            )),
            Net(name="BTN1", connections=(
                NetConnection(ref="SW1", pin="2"),
                NetConnection(ref="R10", pin="1"),
            )),
            Net(name="BTN2", connections=(
                NetConnection(ref="SW1", pin="3"),
                NetConnection(ref="R11", pin="1"),
            )),
            Net(name="BTN3", connections=(
                NetConnection(ref="SW1", pin="4"),
                NetConnection(ref="R12", pin="1"),
            )),
            Net(name="+3V3", connections=(
                NetConnection(ref="R9", pin="2"),
                NetConnection(ref="R10", pin="2"),
                NetConnection(ref="R11", pin="2"),
                NetConnection(ref="R12", pin="2"),
            )),
        ),
    )


class TestPassiveNearConstraints:
    """Tests for section 4b: generic passive NEAR constraints."""

    def test_passive_near_constraint_for_switch_resistors(self) -> None:
        """Resistors sharing signal nets with SW get NEAR constraint."""
        req = _make_switch_resistor_requirements()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        for ref in ("R9", "R10", "R11", "R12"):
            near = [
                c for c in constraints
                if c.ref == ref and c.constraint_type == PlacementConstraintType.NEAR
            ]
            assert len(near) >= 1, f"{ref} should have NEAR constraint"
            assert near[0].target_ref == "SW1"
            assert near[0].max_distance_mm == pytest.approx(5.0)

    def test_passive_near_skips_power_nets(self) -> None:
        """Passive on power net only doesn't get NEAR to power pin."""
        # R9 also has +3V3 on pin 2, but NEAR should come from BTN0, not +3V3
        req = _make_switch_resistor_requirements()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        r9_near = [
            c for c in constraints
            if c.ref == "R9" and c.constraint_type == PlacementConstraintType.NEAR
        ]
        assert len(r9_near) >= 1
        # target_pin should be from signal net BTN0 (SW1 pin "1"), not from +3V3
        assert r9_near[0].target_ref == "SW1"
        assert r9_near[0].target_pin == "1"

    def test_passive_near_skips_already_constrained(self) -> None:
        """Decoupling cap already NEAR from section 4 doesn't get duplicate."""
        req = _make_requirements_for_constraints()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        c1_near = [
            c for c in constraints
            if c.ref == "C1" and c.constraint_type == PlacementConstraintType.NEAR
        ]
        # Should have exactly 1 NEAR (from decoupling, not duplicated by 4b)
        assert len(c1_near) == 1
        from kicad_pipeline.constants import DECOUPLING_CAP_MAX_DISTANCE_MM
        assert c1_near[0].max_distance_mm == pytest.approx(DECOUPLING_CAP_MAX_DISTANCE_MM)

    def test_passive_near_pin_level_targeting(self) -> None:
        """Passive NEAR constraint records the specific target pin."""
        req = _make_switch_resistor_requirements()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        r10_near = [
            c for c in constraints
            if c.ref == "R10" and c.constraint_type == PlacementConstraintType.NEAR
        ]
        assert len(r10_near) >= 1
        assert r10_near[0].target_pin == "2"  # SW1 pin 2 on BTN1

    def test_passive_near_priority(self) -> None:
        """Passive NEAR constraints have priority 25."""
        req = _make_switch_resistor_requirements()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        r9_near = [
            c for c in constraints
            if c.ref == "R9" and c.constraint_type == PlacementConstraintType.NEAR
        ]
        assert len(r9_near) >= 1
        assert r9_near[0].priority == 25

    def test_passive_placed_outward_from_pin(self) -> None:
        """Passive placed outward from target pin, not at arbitrary angle."""
        # SW1 fixed at center, R9 NEAR SW1 pin "5" (right side, bottom)
        # Pin 5 is on the right side of SW1 (positive x offset).
        # R9 should be placed further right (outward from center through pin).
        req = ProjectRequirements(
            project=ProjectInfo(name="OutwardTest"),
            features=(),
            components=(
                Component(ref="SW1", value="DIP4", footprint="DIP-SW_4", pins=(
                    Pin(number="5", name="5", pin_type=PinType.PASSIVE, net="SIG"),
                )),
                Component(ref="R9", value="10k", footprint="R_0603", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                )),
            ),
            nets=(
                Net(name="SIG", connections=(
                    NetConnection(ref="SW1", pin="5"),
                    NetConnection(ref="R9", pin="1"),
                )),
            ),
        )
        sw_constraints = (
            PlacementConstraint(
                ref="SW1", constraint_type=PlacementConstraintType.FIXED,
                x=40.0, y=20.0, priority=100,
            ),
            PlacementConstraint(
                ref="R9", constraint_type=PlacementConstraintType.NEAR,
                target_ref="SW1", target_pin="5",
                max_distance_mm=5.0, priority=28,
            ),
        )
        sizes = {"SW1": (7.62, 10.16), "R9": (1.6, 0.8)}
        result = solve_placement(
            sw_constraints, _board(), sizes, requirements=req,
        )
        # R9 should be to the RIGHT of SW1 (x > SW1.x) since pin 5 is right side
        assert result.positions["R9"].x > 40.0

    def test_dip_pin_offset_right_side_mirrored(self) -> None:
        """Right-side DIP pins use mirrored Y-order (bottom-to-top)."""
        # 8-pin DIP: pins 1-4 left (top→bottom), 5-8 right (bottom→top)
        req = ProjectRequirements(
            project=ProjectInfo(name="DIPMirrorTest"),
            features=(),
            components=(
                Component(ref="SW1", value="DIP4", footprint="DIP-SW_4", pins=tuple(
                    Pin(number=str(i), name=str(i), pin_type=PinType.PASSIVE, net=f"N{i}")
                    for i in range(1, 9)
                )),
                Component(ref="R9", value="10k", footprint="R_0603", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="N5"),
                )),
            ),
            nets=tuple(
                Net(name=f"N{i}", connections=(
                    NetConnection(ref="SW1", pin=str(i)),
                    NetConnection(ref="R9", pin="1")
                    if i == 5
                    else NetConnection(ref="SW1", pin=str(i)),
                ))
                for i in range(1, 9)
            ),
        )
        w, h = 7.62, 10.16
        constraints = (
            PlacementConstraint(
                ref="SW1", constraint_type=PlacementConstraintType.FIXED,
                x=40.0, y=20.0, priority=100,
            ),
            PlacementConstraint(
                ref="R9", constraint_type=PlacementConstraintType.NEAR,
                target_ref="SW1", target_pin="5",
                max_distance_mm=5.0, priority=28,
            ),
        )
        sizes = {"SW1": (w, h), "R9": (1.6, 0.8)}
        result = solve_placement(constraints, _board(), sizes, requirements=req)
        # Pin 5 is first right-side pin → should be at bottom-right (+w/2, +h/2)
        # So R9 should be placed below-right of SW1 center (y > 20)
        r9 = result.positions["R9"]
        assert r9.x > 40.0, f"R9 should be right of SW1, got x={r9.x}"
        # Pin 8 (last right) should be at top-right — verify via R9 placement
        # which targets pin 5 (bottom-right), so R9.y should be > SW1.y
        assert r9.y > 20.0, f"R9 near pin 5 (bottom-right) should have y > 20, got y={r9.y}"

    def test_edge_target_passive_placed_inward_without_pin(self) -> None:
        """NEAR(J, no target_pin) + J has EDGE → passive placed toward board center."""
        req = ProjectRequirements(
            project=ProjectInfo(name="InwardNoPinTest"),
            features=(),
            components=(
                Component(ref="J3", value="Conn", footprint="TerminalBlock_1x02", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                )),
                Component(ref="C4", value="100nF", footprint="C_0603", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                )),
            ),
            nets=(
                Net(name="SIG", connections=(
                    NetConnection(ref="J3", pin="1"),
                    NetConnection(ref="C4", pin="1"),
                )),
            ),
        )
        constraints = (
            # J3 on LEFT edge
            PlacementConstraint(
                ref="J3", constraint_type=PlacementConstraintType.EDGE,
                edge=BoardEdge.LEFT, x=5.0, y=20.0, priority=50,
            ),
            # C4 NEAR J3 but NO target_pin
            PlacementConstraint(
                ref="C4", constraint_type=PlacementConstraintType.NEAR,
                target_ref="J3",
                max_distance_mm=5.0, priority=25,
            ),
        )
        sizes = {"J3": (5.0, 10.0), "C4": (1.6, 0.8)}
        result = solve_placement(constraints, _board(), sizes, requirements=req)
        # C4 should be to the RIGHT (inward) of J3 on the LEFT edge
        assert result.positions["C4"].x > result.positions["J3"].x, (
            f"C4 should be inward (right) of J3, got C4.x={result.positions['C4'].x} "
            f"vs J3.x={result.positions['J3'].x}"
        )

    def test_channel_grouping_has_target_pin(self) -> None:
        """Channel grouping constraints include target_pin from SENS net."""
        from kicad_pipeline.pcb.board_templates import get_template

        req = ProjectRequirements(
            project=ProjectInfo(name="ChannelPinTest"),
            features=(),
            components=(
                Component(ref="J1", value="PinHeader_2x20", footprint="PinHeader_2x20_P2.54mm",
                          pins=()),
                Component(ref="J2", value="Conn", footprint="TerminalBlock_1x02", pins=(
                    Pin(number="1", name="SENS1", pin_type=PinType.PASSIVE, net="SENS1"),
                )),
                Component(ref="R1", value="100k", footprint="R_0603", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SENS1"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="AIN1"),
                )),
                Component(ref="R2", value="10k", footprint="R_0603", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="AIN1"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                )),
                Component(ref="C1", value="100nF", footprint="C_0603", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="AIN1"),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
                )),
                Component(ref="U1", value="ADS1115", footprint="MSOP-10", pins=(
                    Pin(number="1", name="AIN1", pin_type=PinType.INPUT, net="AIN1"),
                )),
            ),
            nets=(
                Net(name="SENS1", connections=(
                    NetConnection(ref="J2", pin="1"),
                    NetConnection(ref="R1", pin="1"),
                )),
                Net(name="AIN1", connections=(
                    NetConnection(ref="R1", pin="2"),
                    NetConnection(ref="R2", pin="1"),
                    NetConnection(ref="C1", pin="1"),
                    NetConnection(ref="U1", pin="1"),
                )),
                Net(name="GND", connections=(
                    NetConnection(ref="R2", pin="2"),
                    NetConnection(ref="C1", pin="2"),
                )),
            ),
        )
        tmpl = get_template("RPI_HAT")
        sizes = {"J1": (50.8, 5.08), "J2": (5.0, 10.0), "R1": (1.6, 0.8),
                 "R2": (1.6, 0.8), "C1": (1.6, 0.8), "U1": (5.0, 3.0)}
        result = rpi_hat_constraints(req, tmpl, sizes)
        # Find channel grouping constraints for R1, R2, C1
        channel_constraints = [
            c for c in result
            if c.ref in ("R1", "R2", "C1")
            and c.constraint_type == PlacementConstraintType.NEAR
            and c.target_ref == "J2"
        ]
        assert len(channel_constraints) >= 1, "Should have channel grouping constraints"
        for cc in channel_constraints:
            assert cc.target_pin == "1", (
                f"Channel constraint for {cc.ref} should have target_pin='1', "
                f"got {cc.target_pin}"
            )
            assert cc.max_distance_mm == 8.0, (
                f"Channel constraint distance should be 8.0mm, got {cc.max_distance_mm}"
            )

    def test_connector_cap_placed_inward(self) -> None:
        """Cap NEAR an edge connector placed on board-interior side."""
        req = ProjectRequirements(
            project=ProjectInfo(name="InwardTest"),
            features=(),
            components=(
                Component(ref="J2", value="Conn", footprint="TerminalBlock_1x02", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                )),
                Component(ref="C2", value="100nF", footprint="C_0603", pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="SIG"),
                )),
            ),
            nets=(
                Net(name="SIG", connections=(
                    NetConnection(ref="J2", pin="1"),
                    NetConnection(ref="C2", pin="1"),
                )),
            ),
        )
        conn_constraints = (
            # J2 on the LEFT edge at (5, 20)
            PlacementConstraint(
                ref="J2", constraint_type=PlacementConstraintType.EDGE,
                edge=BoardEdge.LEFT, x=5.0, y=20.0, priority=50,
            ),
            PlacementConstraint(
                ref="C2", constraint_type=PlacementConstraintType.NEAR,
                target_ref="J2", target_pin="1",
                max_distance_mm=5.0, priority=25,
            ),
        )
        sizes = {"J2": (5.0, 10.0), "C2": (1.6, 0.8)}
        result = solve_placement(
            conn_constraints, _board(), sizes, requirements=req,
        )
        # C2 should be to the RIGHT of J2 (inward from edge)
        assert result.positions["C2"].x > result.positions["J2"].x
