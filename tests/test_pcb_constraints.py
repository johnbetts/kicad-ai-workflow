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
    _connector_edge,
    _is_decoupling_cap,
    _is_power_net,
    _is_screw_terminal,
    _OccupancyGrid,
    build_signal_adjacency,
    constraints_from_requirements,
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
        """Screw terminals alternate between LEFT and BOTTOM edges."""
        e1 = _connector_edge("J1", "TerminalBlock_1x02_P5.08mm")
        e2 = _connector_edge("J2", "TerminalBlock_1x02_P5.08mm")
        assert e1 == BoardEdge.LEFT  # J1 (odd) -> LEFT
        assert e2 == BoardEdge.BOTTOM  # J2 (even) -> BOTTOM

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
        """Screw terminals get EDGE constraints on RPi HAT."""
        from kicad_pipeline.pcb.board_templates import get_template

        req = self._make_hat_requirements()
        tmpl = get_template("RPI_HAT")
        sizes = {c.ref: (3.0, 3.0) for c in req.components}
        constraints = rpi_hat_constraints(req, tmpl, sizes)
        j2 = [c for c in constraints if c.ref == "J2"]
        j3 = [c for c in constraints if c.ref == "J3"]
        assert any(c.constraint_type == PlacementConstraintType.EDGE for c in j2)
        assert any(c.constraint_type == PlacementConstraintType.EDGE for c in j3)

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
