"""Tests for subnet-to-pin resolver and pad-facing placement engine."""

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
    Net,
    NetConnection,
    Pin,
    PinFunction,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.subnet_resolver import (
    SubnetConnection,
    compute_pad_facing_position,
    place_subnet_components,
    resolve_ic_pin_position,
    resolve_subnets,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLOSED_RECT: tuple[Point, ...] = (
    Point(0.0, 0.0), Point(80.0, 0.0),
    Point(80.0, 60.0), Point(0.0, 60.0),
    Point(0.0, 0.0),
)
_DEFAULT_RULES = DesignRules()


def _make_pad(number: str, x: float, y: float, net_number: int = 0,
              net_name: str = "") -> Pad:
    return Pad(
        number=number,
        pad_type="smd",
        shape="rect",
        position=Point(x, y),
        size_x=1.0,
        size_y=0.6,
        layers=("F.Cu",),
        net_number=net_number,
        net_name=net_name,
    )


def _make_ic_footprint(
    ref: str = "U1",
    x: float = 40.0,
    y: float = 30.0,
    rotation: float = 0.0,
) -> Footprint:
    """Create a simple IC footprint with pads on all 4 sides."""
    # 8-pin SOIC-like: 4 pads on west, 4 on east
    pads = (
        # West side (pin 1-4)
        _make_pad("1", -3.0, -2.0, 1, "VCC_U1"),      # VCC pin
        _make_pad("2", -3.0, -0.67, 2, "BST_U1"),      # BST pin
        _make_pad("3", -3.0, 0.67, 3, "SW_U1"),         # SW pin
        _make_pad("4", -3.0, 2.0, 4, "GND"),            # GND pin
        # East side (pin 5-8)
        _make_pad("5", 3.0, 2.0, 5, "FB_U1"),           # FB pin
        _make_pad("6", 3.0, 0.67, 6, "COMP_U1"),        # COMP
        _make_pad("7", 3.0, -0.67, 7, "EN_U1"),         # EN
        _make_pad("8", 3.0, -2.0, 8, "+3V3_U1_DEC"),    # Another VCC
    )
    return Footprint(
        lib_id="Package_SO:SOIC-8",
        ref=ref,
        value="TPS5430",
        position=Point(x, y),
        rotation=rotation,
        pads=pads,
    )


def _make_passive_footprint(
    ref: str = "C1",
    x: float = 35.0,
    y: float = 30.0,
    value: str = "100nF",
) -> Footprint:
    """Create a simple 2-pad passive footprint (0805-like)."""
    pads = (
        _make_pad("1", -0.9, 0.0),
        _make_pad("2", 0.9, 0.0),
    )
    return Footprint(
        lib_id="Resistor_SMD:R_0805_2012Metric",
        ref=ref,
        value=value,
        position=Point(x, y),
        pads=pads,
    )


def _make_pcb(
    footprints: tuple[Footprint, ...] = (),
    nets: tuple[NetEntry, ...] | None = None,
) -> PCBDesign:
    if nets is None:
        nets = (NetEntry(number=0, name=""),)
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


def _make_requirements(
    components: tuple[Component, ...] = (),
    nets: tuple[Net, ...] = (),
) -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        features=(
            FeatureBlock(
                name="Power",
                description="Power supply",
                components=tuple(c.ref for c in components),
                nets=tuple(n.name for n in nets),
                subcircuits=(),
            ),
        ),
        components=components,
        nets=nets,
    )


# ---------------------------------------------------------------------------
# IC component with pin metadata
# ---------------------------------------------------------------------------

_IC_COMPONENT = Component(
    ref="U1",
    value="TPS5430",
    footprint="SOIC-8",
    pins=(
        Pin(number="1", name="VCC", pin_type=PinType.POWER_IN,
            function=PinFunction.VCC),
        Pin(number="2", name="BST", pin_type=PinType.PASSIVE),
        Pin(number="3", name="SW", pin_type=PinType.OUTPUT),
        Pin(number="4", name="GND", pin_type=PinType.POWER_IN,
            function=PinFunction.GND),
        Pin(number="5", name="FB", pin_type=PinType.INPUT),
        Pin(number="6", name="COMP", pin_type=PinType.PASSIVE),
        Pin(number="7", name="EN", pin_type=PinType.INPUT,
            function=PinFunction.ENABLE),
        Pin(number="8", name="VCC2", pin_type=PinType.POWER_IN,
            function=PinFunction.VCC),
    ),
)

_CAP_COMPONENT = Component(ref="C1", value="100nF", footprint="C_0805")
_CAP2_COMPONENT = Component(ref="C2", value="100nF", footprint="C_0805")
_RES_COMPONENT = Component(ref="R1", value="10k", footprint="R_0805")
_RES2_COMPONENT = Component(ref="R2", value="47k", footprint="R_0805")
_IND_COMPONENT = Component(ref="L1", value="10uH", footprint="L_0805")
_DIODE_COMPONENT = Component(ref="D1", value="SS34", footprint="D_SOD-123")


# ===========================================================================
# Test: resolve_subnets
# ===========================================================================


class TestResolveSubnets:
    """Tests for resolve_subnets()."""

    def test_finds_decoupling_subnet(self) -> None:
        """A private net connecting C1 to U1.VCC is classified as decoupling."""
        nets = (
            Net(name="VCC_U1_DEC", connections=(
                NetConnection(ref="U1", pin="1"),
                NetConnection(ref="C1", pin="1"),
            )),
        )
        reqs = _make_requirements(
            components=(_IC_COMPONENT, _CAP_COMPONENT),
            nets=nets,
        )
        result = resolve_subnets(reqs)
        assert len(result) == 1
        conn = result[0]
        assert conn.passive_ref == "C1"
        assert conn.passive_pin == "1"
        assert conn.ic_ref == "U1"
        assert conn.ic_pin == "1"
        assert conn.subnet_name == "VCC_U1_DEC"
        assert conn.role == "decoupling"

    def test_finds_bootstrap_subnet(self) -> None:
        """A private net connecting C2 to U1.BST is classified as bootstrap."""
        nets = (
            Net(name="BST_U1", connections=(
                NetConnection(ref="U1", pin="2"),
                NetConnection(ref="C2", pin="1"),
            )),
        )
        reqs = _make_requirements(
            components=(_IC_COMPONENT, _CAP2_COMPONENT),
            nets=nets,
        )
        result = resolve_subnets(reqs)
        assert len(result) == 1
        assert result[0].role == "bootstrap"

    def test_finds_feedback_subnet(self) -> None:
        """A net connecting R1 to U1.FB is classified as feedback."""
        nets = (
            Net(name="FB_U1", connections=(
                NetConnection(ref="U1", pin="5"),
                NetConnection(ref="R1", pin="1"),
            )),
        )
        reqs = _make_requirements(
            components=(_IC_COMPONENT, _RES_COMPONENT),
            nets=nets,
        )
        result = resolve_subnets(reqs)
        assert len(result) == 1
        assert result[0].role == "feedback"

    def test_finds_inductor_subnet(self) -> None:
        """A net connecting L1 to U1.SW is classified as inductor."""
        nets = (
            Net(name="SW_U1", connections=(
                NetConnection(ref="U1", pin="3"),
                NetConnection(ref="L1", pin="1"),
            )),
        )
        reqs = _make_requirements(
            components=(_IC_COMPONENT, _IND_COMPONENT),
            nets=nets,
        )
        result = resolve_subnets(reqs)
        assert len(result) == 1
        assert result[0].role == "inductor"

    def test_ignores_shared_rails(self) -> None:
        """A net with 10+ connections (shared power rail) is not a subnet."""
        many_conns = (
            *(NetConnection(ref=f"C{i}", pin="1") for i in range(10)),
            NetConnection(ref="U1", pin="1"),
        )
        nets = (
            Net(name="+3V3", connections=many_conns),
        )
        components = (
            *(Component(ref=f"C{i}", value="100nF", footprint="C_0805")
              for i in range(10)),
            _IC_COMPONENT,
        )
        reqs = _make_requirements(components=components, nets=nets)
        result = resolve_subnets(reqs)
        assert len(result) == 0

    def test_empty_requirements(self) -> None:
        """Empty requirements yield no subnets."""
        reqs = _make_requirements()
        result = resolve_subnets(reqs)
        assert result == []

    def test_net_with_only_passives_not_subnet(self) -> None:
        """A net connecting only passives (no IC) is not a subnet."""
        nets = (
            Net(name="R_DIV", connections=(
                NetConnection(ref="R1", pin="2"),
                NetConnection(ref="R2", pin="1"),
            )),
        )
        reqs = _make_requirements(
            components=(_RES_COMPONENT, _RES2_COMPONENT),
            nets=nets,
        )
        result = resolve_subnets(reqs)
        assert len(result) == 0

    def test_net_with_only_ics_not_subnet(self) -> None:
        """A net connecting only ICs (no passives) is not a subnet."""
        ic2 = Component(
            ref="U2", value="LM1117", footprint="SOT-223",
            pins=(Pin(number="1", name="OUT", pin_type=PinType.OUTPUT),),
        )
        nets = (
            Net(name="IC_LINK", connections=(
                NetConnection(ref="U1", pin="7"),
                NetConnection(ref="U2", pin="1"),
            )),
        )
        reqs = _make_requirements(
            components=(_IC_COMPONENT, ic2),
            nets=nets,
        )
        result = resolve_subnets(reqs)
        assert len(result) == 0

    def test_single_connection_net_ignored(self) -> None:
        """A net with only 1 connection is not a subnet."""
        nets = (
            Net(name="LONELY", connections=(
                NetConnection(ref="U1", pin="1"),
            )),
        )
        reqs = _make_requirements(
            components=(_IC_COMPONENT,),
            nets=nets,
        )
        result = resolve_subnets(reqs)
        assert len(result) == 0

    def test_multiple_passives_on_one_subnet(self) -> None:
        """A subnet with 1 IC pin + 2 passives produces 2 connections."""
        nets = (
            Net(name="VCC_U1_FILT", connections=(
                NetConnection(ref="U1", pin="1"),
                NetConnection(ref="C1", pin="1"),
                NetConnection(ref="C2", pin="1"),
            )),
        )
        reqs = _make_requirements(
            components=(_IC_COMPONENT, _CAP_COMPONENT, _CAP2_COMPONENT),
            nets=nets,
        )
        result = resolve_subnets(reqs)
        assert len(result) == 2
        passive_refs = {c.passive_ref for c in result}
        assert passive_refs == {"C1", "C2"}

    def test_enable_pin_classified_as_associated(self) -> None:
        """An EN pin with no special keyword falls back to 'associated'."""
        nets = (
            Net(name="EN_U1", connections=(
                NetConnection(ref="U1", pin="7"),
                NetConnection(ref="R1", pin="1"),
            )),
        )
        reqs = _make_requirements(
            components=(_IC_COMPONENT, _RES_COMPONENT),
            nets=nets,
        )
        result = resolve_subnets(reqs)
        assert len(result) == 1
        assert result[0].role == "associated"


# ===========================================================================
# Test: resolve_ic_pin_position
# ===========================================================================


class TestResolveIcPinPosition:
    """Tests for resolve_ic_pin_position()."""

    def test_known_pin_position(self) -> None:
        """Pin 1 of U1 at (40,30) with pad at (-3, -2) -> board (37, 28)."""
        ic_fp = _make_ic_footprint(ref="U1", x=40.0, y=30.0, rotation=0.0)
        pcb = _make_pcb(footprints=(ic_fp,))

        result = resolve_ic_pin_position("U1", "1", pcb)
        assert result is not None
        x, y, side = result
        assert abs(x - 37.0) < 0.01
        assert abs(y - 28.0) < 0.01
        assert side == "W"  # pad at -3.0 is on the west side

    def test_east_pin(self) -> None:
        """Pin 8 at (+3, -2) is on the east side."""
        ic_fp = _make_ic_footprint()
        pcb = _make_pcb(footprints=(ic_fp,))

        result = resolve_ic_pin_position("U1", "8", pcb)
        assert result is not None
        x, y, side = result
        assert abs(x - 43.0) < 0.01
        assert abs(y - 28.0) < 0.01
        assert side == "E"

    def test_ic_not_found(self) -> None:
        """Returns None when the IC ref is not in the PCB."""
        pcb = _make_pcb(footprints=())
        result = resolve_ic_pin_position("U99", "1", pcb)
        assert result is None

    def test_pin_not_found(self) -> None:
        """Returns None when the pin number doesn't exist on the IC."""
        ic_fp = _make_ic_footprint()
        pcb = _make_pcb(footprints=(ic_fp,))
        result = resolve_ic_pin_position("U1", "99", pcb)
        assert result is None

    def test_rotated_ic_pin_position(self) -> None:
        """Pin position accounts for IC rotation."""
        ic_fp = _make_ic_footprint(x=40.0, y=30.0, rotation=90.0)
        pcb = _make_pcb(footprints=(ic_fp,))

        result = resolve_ic_pin_position("U1", "1", pcb)
        assert result is not None
        x, y, side = result
        # Pad at local (-3, -2), rotated 90 CW: (x,y) -> (x*cos+y*sin, -x*sin+y*cos)
        # with -angle for CW: cos(-90)=0, sin(-90)=-1
        # rx = -3*0 - (-2)*(-1) = -2, ry = -3*(-1) + (-2)*0 = 3
        assert abs(x - (40.0 + (-2.0))) < 0.01
        assert abs(y - (30.0 + 3.0)) < 0.01


# ===========================================================================
# Test: compute_pad_facing_position
# ===========================================================================


class TestComputePadFacingPosition:
    """Tests for compute_pad_facing_position()."""

    def test_west_pin(self) -> None:
        """Passive placed to the left of a west-side IC pin."""
        x, y, rot = compute_pad_facing_position(
            passive_size=(2.0, 1.0),
            ic_pin_x=10.0, ic_pin_y=20.0,
            ic_pin_side="W",
            gap_mm=1.0,
        )
        # passive center = pin_x - pw/2 - gap = 10 - 1 - 1 = 8
        assert abs(x - 8.0) < 0.01
        assert abs(y - 20.0) < 0.01
        assert abs(rot - 0.0) < 0.01

    def test_east_pin(self) -> None:
        """Passive placed to the right of an east-side IC pin."""
        x, y, rot = compute_pad_facing_position(
            passive_size=(2.0, 1.0),
            ic_pin_x=30.0, ic_pin_y=20.0,
            ic_pin_side="E",
            gap_mm=1.0,
        )
        # passive center = pin_x + pw/2 + gap = 30 + 1 + 1 = 32
        assert abs(x - 32.0) < 0.01
        assert abs(y - 20.0) < 0.01
        assert abs(rot - 180.0) < 0.01

    def test_north_pin(self) -> None:
        """Passive placed above a north-side IC pin."""
        x, y, rot = compute_pad_facing_position(
            passive_size=(2.0, 1.0),
            ic_pin_x=20.0, ic_pin_y=10.0,
            ic_pin_side="N",
            gap_mm=1.0,
        )
        # passive center = pin_y - ph/2 - gap = 10 - 0.5 - 1 = 8.5
        assert abs(x - 20.0) < 0.01
        assert abs(y - 8.5) < 0.01
        assert abs(rot - 90.0) < 0.01

    def test_south_pin(self) -> None:
        """Passive placed below a south-side IC pin."""
        x, y, rot = compute_pad_facing_position(
            passive_size=(2.0, 1.0),
            ic_pin_x=20.0, ic_pin_y=30.0,
            ic_pin_side="S",
            gap_mm=1.0,
        )
        # passive center = pin_y + ph/2 + gap = 30 + 0.5 + 1 = 31.5
        assert abs(x - 20.0) < 0.01
        assert abs(y - 31.5) < 0.01
        assert abs(rot - 270.0) < 0.01

    def test_rotation_correct_for_sides(self) -> None:
        """Each side gets the expected rotation."""
        sides_and_rotations = [
            ("W", 0.0),
            ("E", 180.0),
            ("N", 90.0),
            ("S", 270.0),
        ]
        for side, expected_rot in sides_and_rotations:
            _, _, rot = compute_pad_facing_position(
                passive_size=(2.0, 1.0),
                ic_pin_x=20.0, ic_pin_y=20.0,
                ic_pin_side=side,
            )
            assert abs(rot - expected_rot) < 0.01, (
                f"Side {side}: expected rotation {expected_rot}, got {rot}"
            )

    def test_custom_gap(self) -> None:
        """Gap parameter affects placement distance."""
        x1, _, _ = compute_pad_facing_position(
            passive_size=(2.0, 1.0),
            ic_pin_x=20.0, ic_pin_y=20.0,
            ic_pin_side="W",
            gap_mm=0.5,
        )
        x2, _, _ = compute_pad_facing_position(
            passive_size=(2.0, 1.0),
            ic_pin_x=20.0, ic_pin_y=20.0,
            ic_pin_side="W",
            gap_mm=2.0,
        )
        # Larger gap -> further left (smaller x)
        assert x2 < x1


# ===========================================================================
# Test: place_subnet_components
# ===========================================================================


class TestPlaceSubnetComponents:
    """Tests for place_subnet_components()."""

    def _make_ctx(self) -> object:
        """Build a minimal PlacementContext for testing."""
        from kicad_pipeline.optimization.placement_types import PlacementContext

        ic_fp = _make_ic_footprint(ref="U1", x=40.0, y=30.0)
        c1_fp = _make_passive_footprint(ref="C1", x=10.0, y=10.0)
        pcb = _make_pcb(footprints=(ic_fp, c1_fp))

        positions: dict[str, tuple[float, float, float]] = {
            "U1": (40.0, 30.0, 0.0),
            "C1": (10.0, 10.0, 0.0),
        }
        fp_sizes: dict[str, tuple[float, float]] = {
            "U1": (6.0, 4.0),
            "C1": (2.0, 1.0),
        }

        return PlacementContext(
            positions=positions,
            fp_sizes=fp_sizes,
            bounds=(0.0, 0.0, 80.0, 60.0),
            fixed_refs=set(),
            requirements=_make_requirements(
                components=(_IC_COMPONENT, _CAP_COMPONENT),
                nets=(),
            ),
            initial_pcb=pcb,
            zones=[],
            subcircuits=[],
        )

    def test_places_decoupling_cap(self) -> None:
        """A decoupling cap is placed facing the IC VCC pin."""
        ctx = self._make_ctx()
        connections = [
            SubnetConnection(
                passive_ref="C1",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="1",
                subnet_name="VCC_U1",
                role="decoupling",
            ),
        ]
        placed = place_subnet_components(ctx, connections)  # type: ignore[arg-type]
        assert "C1" in placed
        # C1 should now be near pin 1 of U1 (west side)
        cx, cy, crot = ctx.positions["C1"]  # type: ignore[union-attr]
        # Pin 1 is at board position (37, 28) (west side)
        # Passive should be to the left of that
        assert cx < 37.0

    def test_fixed_refs_not_moved(self) -> None:
        """Passives already in fixed_refs are not moved."""
        ctx = self._make_ctx()
        ctx.fixed_refs.add("C1")  # type: ignore[union-attr]
        original_pos = ctx.positions["C1"]  # type: ignore[union-attr]

        connections = [
            SubnetConnection(
                passive_ref="C1",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="1",
                subnet_name="VCC_U1",
                role="decoupling",
            ),
        ]
        placed = place_subnet_components(ctx, connections)  # type: ignore[arg-type]
        assert "C1" not in placed
        assert ctx.positions["C1"] == original_pos  # type: ignore[union-attr]

    def test_placed_refs_not_added_to_fixed(self) -> None:
        """After placement, placed refs are NOT added to fixed_refs.

        Subnet placement should leave refs unlocked so later type-specific
        phases (relay driver, power chain, etc.) can refine positions.
        """
        ctx = self._make_ctx()
        connections = [
            SubnetConnection(
                passive_ref="C1",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="1",
                subnet_name="VCC_U1",
                role="decoupling",
            ),
        ]
        place_subnet_components(ctx, connections)  # type: ignore[arg-type]
        assert "C1" not in ctx.fixed_refs  # type: ignore[union-attr]

    def test_missing_ic_skipped(self) -> None:
        """Connections referencing missing ICs are skipped gracefully."""
        ctx = self._make_ctx()
        connections = [
            SubnetConnection(
                passive_ref="C1",
                passive_pin="1",
                ic_ref="U99",  # doesn't exist
                ic_pin="1",
                subnet_name="VCC_U99",
                role="decoupling",
            ),
        ]
        placed = place_subnet_components(ctx, connections)  # type: ignore[arg-type]
        assert len(placed) == 0

    def test_missing_fp_size_skipped(self) -> None:
        """Connections for passives with no known size are skipped."""
        ctx = self._make_ctx()
        del ctx.fp_sizes["C1"]  # type: ignore[union-attr]
        connections = [
            SubnetConnection(
                passive_ref="C1",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="1",
                subnet_name="VCC_U1",
                role="decoupling",
            ),
        ]
        placed = place_subnet_components(ctx, connections)  # type: ignore[arg-type]
        assert len(placed) == 0

    def test_close_component_not_moved(self) -> None:
        """Components already within 8mm of IC pin are not disturbed."""
        ctx = self._make_ctx()
        # Place C1 close to pin 1 of U1 (west side at board x=37, y=28)
        ctx.positions["C1"] = (34.0, 28.0, 0.0)  # type: ignore[union-attr]

        connections = [
            SubnetConnection(
                passive_ref="C1",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="1",
                subnet_name="VCC_U1",
                role="decoupling",
            ),
        ]
        placed = place_subnet_components(ctx, connections)  # type: ignore[arg-type]
        assert "C1" not in placed
        # Position should be unchanged
        assert ctx.positions["C1"] == (34.0, 28.0, 0.0)  # type: ignore[union-attr]

    def test_position_clamped_to_board(self) -> None:
        """Placement is clamped to board bounds even if pin is near the edge."""
        ctx = self._make_ctx()
        # Move IC to the very left edge
        ctx.positions["U1"] = (2.0, 30.0, 0.0)  # type: ignore[union-attr]

        # Rebuild PCB with IC near edge
        ic_fp = _make_ic_footprint(ref="U1", x=2.0, y=30.0)
        c1_fp = _make_passive_footprint(ref="C1", x=50.0, y=50.0)
        ctx.initial_pcb = _make_pcb(footprints=(ic_fp, c1_fp))  # type: ignore[union-attr]

        connections = [
            SubnetConnection(
                passive_ref="C1",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="1",  # west side pin
                subnet_name="VCC_U1",
                role="decoupling",
            ),
        ]
        placed = place_subnet_components(ctx, connections)  # type: ignore[arg-type]
        if "C1" in placed:
            cx, cy, _ = ctx.positions["C1"]  # type: ignore[union-attr]
            # Should be clamped: x >= bounds_min + pw/2 = 0 + 1.0 = 1.0
            assert cx >= 1.0
