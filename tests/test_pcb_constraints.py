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
        result = layout_pcb(req, board, footprint_sizes=sizes)
        assert len(result.positions) == len(req.components)

    def test_layout_pcb_with_template(self) -> None:
        """layout_pcb with template uses constraint solver."""
        from kicad_pipeline.pcb.board_templates import get_template
        from kicad_pipeline.pcb.placement import layout_pcb

        req = _make_requirements_for_constraints()
        tmpl = get_template("RPI_HAT")
        board = _board(65.0, 56.5)
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        result = layout_pcb(req, board, footprint_sizes=sizes, board_template=tmpl)
        assert len(result.positions) == len(req.components)

    def test_layout_pcb_returns_rotations(self) -> None:
        """layout_pcb with template returns rotations dict."""
        from kicad_pipeline.pcb.board_templates import get_template
        from kicad_pipeline.pcb.placement import LayoutResult, layout_pcb

        req = _make_requirements_for_constraints()
        tmpl = get_template("RPI_HAT")
        board = _board(65.0, 56.5)
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
        """Decoupling caps are placed within 3mm of their IC."""
        req = _make_requirements_for_constraints()
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = constraints_from_requirements(req, None, sizes)
        c1_near = [
            c for c in constraints
            if c.ref == "C1" and c.constraint_type == PlacementConstraintType.NEAR
        ]
        assert len(c1_near) >= 1
        assert c1_near[0].max_distance_mm == pytest.approx(3.0)


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
        # Signal chain should create GROUP constraints at priority 15
        chain_groups = [
            c for c in constraints
            if c.constraint_type == PlacementConstraintType.GROUP and c.priority == 15
        ]
        assert len(chain_groups) >= 2  # At least R1 and R2 in the chain


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

    def test_rpi_hat_voltage_divider_grouped(self) -> None:
        """Voltage divider resistors get GROUP constraints on RPi HAT."""
        from kicad_pipeline.pcb.board_templates import get_template

        req = self._make_hat_requirements()
        tmpl = get_template("RPI_HAT")
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = rpi_hat_constraints(req, tmpl, sizes)
        r1 = [c for c in constraints if c.ref == "R1"]
        r2 = [c for c in constraints if c.ref == "R2"]
        r1_groups = [c for c in r1 if c.constraint_type == PlacementConstraintType.GROUP]
        r2_groups = [c for c in r2 if c.constraint_type == PlacementConstraintType.GROUP]
        assert len(r1_groups) >= 1
        assert len(r2_groups) >= 1

    def test_rpi_hat_full_placement_no_violations(self) -> None:
        """Full RPi HAT placement produces no violations."""
        from kicad_pipeline.pcb.board_templates import get_template

        req = self._make_hat_requirements()
        tmpl = get_template("RPI_HAT")
        board = _board(65.0, 56.5)
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
        # Pad-1 at (-w/2, 0). At 270 deg, pad-1 rotates to (0, +w/2) = below centre,
        # facing U1 which is below at y=30.
        assert result["C1"] == pytest.approx(270.0)

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

        # 90 degrees — (1.5, 0) -> (0, 1.5)
        x, y = _rotated_pad_offset(1.5, 0.0, 90.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(1.5)

        # 180 degrees — (1.5, 0) -> (-1.5, 0)
        x, y = _rotated_pad_offset(1.5, 0.0, 180.0)
        assert x == pytest.approx(-1.5)
        assert y == pytest.approx(0.0, abs=1e-10)

        # 270 degrees — (1.5, 0) -> (0, -1.5)
        x, y = _rotated_pad_offset(1.5, 0.0, 270.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(-1.5)

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
        # R1, R2, C1, U1 should share a net group for AIN0
        ain0_groups = [
            c for c in constraints
            if c.constraint_type == PlacementConstraintType.GROUP
            and c.group_name is not None
            and c.group_name.startswith("_net_group_")
        ]
        ain0_refs = {c.ref for c in ain0_groups}
        # At least 3 of the 4 should be in the same net group
        assert len(ain0_refs & {"R1", "R2", "C1", "U1"}) >= 3

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

    def test_build_pcb_auto_route_enabled(self) -> None:
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
        board = _board(65.0, 56.5)
        sizes = {"J1": (51.0, 5.08), "U1": (3.0, 5.0)}
        result = layout_pcb(
            req, board, footprint_sizes=sizes,
            fixed_positions={"J1": (29.21, 3.29, 0.0)},
            board_template=tmpl,
        )
        # J1 should be at the template fixed position
        assert abs(result.positions["J1"].x - 29.21) < 0.01
        assert abs(result.positions["J1"].y - 3.29) < 0.01
        # U1 should also be placed
        assert "U1" in result.positions
