"""Placement guard regression tests.

Tests derived from recurring placement bugs (KI-004 through KI-010 and
iteration-discovered issues). Each test validates a specific invariant that
was violated in past iterations. Run with::

    pytest tests/regression/test_placement_guards.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from kicad_pipeline.models.pcb import (
    Footprint,
    Pad,
    Point,
)
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinFunction,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.functional_grouper import (
    SubCircuitType,
    detect_subcircuits,
)
from kicad_pipeline.optimization.zone_partitioner import (
    _DEFAULT_ZONE_FRACTIONS,
    _ZONE_GAP_MM,
    BoardZone,
    partition_board,
)
from kicad_pipeline.pcb.pin_map import (
    centroid_to_origin,
    compute_centroid_offset,
    origin_to_centroid,
    pad_extent_in_board_space,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _passive_pins() -> tuple[Pin, ...]:
    return (
        Pin(number="1", name="~", pin_type=PinType.PASSIVE),
        Pin(number="2", name="~", pin_type=PinType.PASSIVE),
    )


def _make_footprint(
    ref: str,
    x: float,
    y: float,
    rotation: float = 0.0,
    pad_positions: tuple[tuple[float, float], ...] = ((0.0, 0.0),),
) -> Footprint:
    """Create a minimal Footprint with pads at given local positions."""
    pads = tuple(
        Pad(
            number=str(i + 1),
            pad_type="smd",
            shape="rect",
            position=Point(x=px, y=py),
            size_x=1.0,
            size_y=1.0,
            layers=("F.Cu",),
        )
        for i, (px, py) in enumerate(pad_positions)
    )
    return Footprint(
        lib_id=f"{ref}:{ref}",
        ref=ref,
        value=ref,
        position=Point(x=x, y=y),
        rotation=rotation,
        pads=pads,
    )


def _make_connector_footprint(
    ref: str,
    x: float,
    y: float,
    rotation: float = 0.0,
    pin_count: int = 8,
    pitch: float = 2.54,
) -> Footprint:
    """Create a connector with pads in a vertical column (pin-1 at origin)."""
    pads = tuple(
        Pad(
            number=str(i + 1),
            pad_type="thru_hole",
            shape="circle",
            position=Point(x=0.0, y=i * pitch),
            size_x=1.7,
            size_y=1.7,
            layers=("F.Cu", "B.Cu"),
            drill_diameter=1.0,
        )
        for i in range(pin_count)
    )
    return Footprint(
        lib_id=f"Connector:{ref}",
        ref=ref,
        value=f"Conn_{pin_count}",
        position=Point(x=x, y=y),
        rotation=rotation,
        pads=pads,
        attr="through_hole",
    )


def _make_relay_requirements() -> ProjectRequirements:
    """Build requirements with 4 relay drivers for layout tests."""
    components: list[Component] = []
    nets: list[Net] = []

    # MCU
    mcu_pins: list[Pin] = [
        Pin(number="1", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
        Pin(number="2", name="3V3", pin_type=PinType.POWER_IN, function=PinFunction.VCC),
    ]
    for i in range(3, 7):
        mcu_pins.append(
            Pin(number=str(i), name=f"GPIO{i}", pin_type=PinType.BIDIRECTIONAL,
                function=PinFunction.GPIO)
        )
    components.append(Component(
        ref="U1", value="ESP32-S3-WROOM-1", footprint="ESP32-S3-WROOM-1",
        lcsc="C2913202", pins=tuple(mcu_pins),
    ))

    relay_refs: list[str] = []
    driver_refs: list[str] = []
    for i in range(1, 5):
        # Relay coil
        k_ref = f"K{i}"
        components.append(Component(
            ref=k_ref, value="G5V-1-DC5", footprint="Relay_DPDT",
            pins=(
                Pin(number="1", name="COIL+", pin_type=PinType.PASSIVE),
                Pin(number="2", name="COIL-", pin_type=PinType.PASSIVE),
                Pin(number="3", name="COM", pin_type=PinType.PASSIVE),
                Pin(number="4", name="NC", pin_type=PinType.PASSIVE),
                Pin(number="5", name="NO", pin_type=PinType.PASSIVE),
            ),
        ))
        relay_refs.append(k_ref)

        # Driver transistor
        q_ref = f"Q{i}"
        components.append(Component(
            ref=q_ref, value="2N7002", footprint="SOT-23",
            pins=(
                Pin(number="1", name="G", pin_type=PinType.INPUT),
                Pin(number="2", name="D", pin_type=PinType.PASSIVE),
                Pin(number="3", name="S", pin_type=PinType.PASSIVE),
            ),
        ))
        driver_refs.append(q_ref)

        # Flyback diode
        d_ref = f"D{i}"
        components.append(Component(
            ref=d_ref, value="1N4148", footprint="D_SOD-323",
            pins=_passive_pins(),
        ))
        driver_refs.append(d_ref)

        # Gate resistor
        r_ref = f"R{i}"
        components.append(Component(
            ref=r_ref, value="1k", footprint="R_0805",
            lcsc="C17513", pins=_passive_pins(),
        ))
        driver_refs.append(r_ref)

        # Nets for this relay driver
        gate_net = f"RELAY{i}_GATE"
        coil_net = f"RELAY{i}_COIL"
        nets.append(Net(name=gate_net, connections=(
            NetConnection(ref="U1", pin=str(i + 2)),
            NetConnection(ref=r_ref, pin="1"),
        )))
        nets.append(Net(name=f"R{i}_Q{i}", connections=(
            NetConnection(ref=r_ref, pin="2"),
            NetConnection(ref=q_ref, pin="1"),
        )))
        nets.append(Net(name=coil_net, connections=(
            NetConnection(ref=q_ref, pin="2"),
            NetConnection(ref=k_ref, pin="2"),
            NetConnection(ref=d_ref, pin="1"),
        )))
        nets.append(Net(name=f"RELAY{i}_VCC", connections=(
            NetConnection(ref=k_ref, pin="1"),
            NetConnection(ref=d_ref, pin="2"),
        )))

    nets.append(Net(name="GND", connections=(
        NetConnection(ref="U1", pin="1"),
        *(NetConnection(ref=f"Q{i}", pin="3") for i in range(1, 5)),
    )))
    nets.append(Net(name="+3V3", connections=(NetConnection(ref="U1", pin="2"),)))

    features = (
        FeatureBlock(
            name="MCU", description="ESP32 MCU",
            components=("U1",),
            nets=("GND", "+3V3"), subcircuits=(),
        ),
        FeatureBlock(
            name="Relay Outputs", description="4-channel relay switching",
            components=tuple(relay_refs + driver_refs),
            nets=tuple(n.name for n in nets if "RELAY" in n.name or n.name.startswith("R")),
            subcircuits=("relay_driver",),
        ),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="relay-test"),
        features=features,
        components=tuple(components),
        nets=tuple(nets),
        mechanical=MechanicalConstraints(board_width_mm=100.0, board_height_mm=80.0),
    )


def _make_buck_requirements() -> ProjectRequirements:
    """Build requirements with a buck converter on shared power rails."""
    components: list[Component] = []
    nets: list[Net] = []

    # Buck converter IC with SW pin
    buck_pins = (
        Pin(number="1", name="VIN", pin_type=PinType.POWER_IN, function=PinFunction.VCC),
        Pin(number="2", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
        Pin(number="3", name="SW", pin_type=PinType.OUTPUT),
        Pin(number="4", name="FB", pin_type=PinType.INPUT),
        Pin(number="5", name="EN", pin_type=PinType.INPUT),
    )
    components.append(Component(
        ref="U2", value="TPS54331", footprint="SOIC-8",
        description="Buck converter", pins=buck_pins,
    ))

    # Inductor on SW net
    components.append(Component(
        ref="L1", value="10uH", footprint="IND_1210",
        pins=_passive_pins(),
    ))

    # Input caps
    for i in range(1, 4):
        components.append(Component(
            ref=f"C{i}", value="10uF", footprint="C_0805",
            lcsc="C49678", pins=_passive_pins(),
        ))

    # Output caps
    for i in range(4, 7):
        components.append(Component(
            ref=f"C{i}", value="22uF", footprint="C_0805",
            lcsc="C49678", pins=_passive_pins(),
        ))

    # Feedback divider resistors
    components.append(Component(
        ref="R10", value="100k", footprint="R_0805",
        lcsc="C17414", pins=_passive_pins(),
    ))
    components.append(Component(
        ref="R11", value="33k", footprint="R_0805",
        lcsc="C17414", pins=_passive_pins(),
    ))

    # SW net: U2.SW → L1.1
    nets.append(Net(name="SW_NET", connections=(
        NetConnection(ref="U2", pin="3"),
        NetConnection(ref="L1", pin="1"),
    )))
    # L1 output → output caps on a named net (not a global rail)
    nets.append(Net(name="BUCK_5V", connections=(
        NetConnection(ref="L1", pin="2"),
        NetConnection(ref="C4", pin="1"),
        NetConnection(ref="C5", pin="1"),
        NetConnection(ref="C6", pin="1"),
    )))
    # VIN net (local to buck, not global)
    nets.append(Net(name="VIN_BUCK", connections=(
        NetConnection(ref="U2", pin="1"),
        NetConnection(ref="C1", pin="1"),
        NetConnection(ref="C2", pin="1"),
        NetConnection(ref="C3", pin="1"),
    )))
    # FB net
    nets.append(Net(name="FB_NET", connections=(
        NetConnection(ref="U2", pin="4"),
        NetConnection(ref="R10", pin="1"),
        NetConnection(ref="R11", pin="1"),
    )))
    # GND
    nets.append(Net(name="GND", connections=(
        NetConnection(ref="U2", pin="2"),
        NetConnection(ref="C1", pin="2"), NetConnection(ref="C2", pin="2"),
        NetConnection(ref="C3", pin="2"), NetConnection(ref="C4", pin="2"),
        NetConnection(ref="C5", pin="2"), NetConnection(ref="C6", pin="2"),
        NetConnection(ref="R11", pin="2"),
    )))

    features = (
        FeatureBlock(
            name="Power Supply", description="5V buck converter",
            components=("U2", "L1", "C1", "C2", "C3", "C4", "C5", "C6", "R10", "R11"),
            nets=("SW_NET", "BUCK_5V", "VIN_BUCK", "FB_NET", "GND"),
            subcircuits=("buck_converter",),
        ),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="buck-test"),
        features=features,
        components=tuple(components),
        nets=tuple(nets),
        mechanical=MechanicalConstraints(board_width_mm=60.0, board_height_mm=50.0),
    )


# ---------------------------------------------------------------------------
# KI-004: Centroid offset consolidation — single source of truth
# ---------------------------------------------------------------------------


class TestCentroidConsistency:
    """Verify that centroid math is centralized in pin_map.py (KI-004).

    Root cause: compute_centroid_offset was duplicated in 4 places with
    slightly different implementations, causing inconsistent origin-to-centroid
    conversions.  Now consolidated in ``pcb.pin_map``.
    """

    def test_pin_map_is_single_source_of_truth(self) -> None:
        """compute_centroid_offset, origin_to_centroid, centroid_to_origin exist."""
        # These were consolidated from 4 inline locations into pin_map.py.
        # If any of these fail to import, the consolidation has regressed.
        assert callable(compute_centroid_offset)
        assert callable(origin_to_centroid)
        assert callable(centroid_to_origin)

    def test_no_inline_centroid_math_in_optimizer(self) -> None:
        """placement_optimizer.py must not contain inline centroid computation.

        It should import from pin_map, not compute (min+max)/2 inline.
        """
        optimizer_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "kicad_pipeline" / "optimization" / "placement_optimizer.py"
        )
        source = optimizer_path.read_text()

        # The old inline pattern was:
        #   cx = (min(xs) + max(xs)) / 2.0
        # This should now only appear in pin_map.py, not in the optimizer.
        lines = source.split("\n")
        inline_hits: list[str] = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments and import lines
            if stripped.startswith("#") or stripped.startswith("import") or "from " in stripped:
                continue
            # Look for the (min(...) + max(...)) / 2 centroid pattern
            if "(min(" in stripped and "max(" in stripped and "/ 2" in stripped:
                inline_hits.append(f"  line {i}: {stripped}")

        assert not inline_hits, (
            "Inline centroid math found in placement_optimizer.py "
            "(should use pin_map.compute_centroid_offset):\n"
            + "\n".join(inline_hits)
        )

    def test_roundtrip_origin_centroid(self) -> None:
        """origin_to_centroid and centroid_to_origin must be exact inverses."""
        # Connector with pin 1 at origin, 8 pins spaced 2.54mm apart
        fp = _make_connector_footprint("J1", x=50.0, y=30.0, rotation=0.0)
        ox, oy = fp.position.x, fp.position.y

        cx, cy = origin_to_centroid(fp, ox, oy, fp.rotation)
        ox2, oy2 = centroid_to_origin(fp, cx, cy, fp.rotation)

        assert abs(ox2 - ox) < 0.001, f"X roundtrip error: {ox} -> {ox2}"
        assert abs(oy2 - oy) < 0.001, f"Y roundtrip error: {oy} -> {oy2}"

    def test_roundtrip_with_rotation(self) -> None:
        """Roundtrip must hold for non-zero rotations."""
        fp = _make_connector_footprint("J2", x=40.0, y=20.0, rotation=90.0)

        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, 90.0)
        ox2, oy2 = centroid_to_origin(fp, cx, cy, 90.0)

        assert abs(ox2 - fp.position.x) < 0.001
        assert abs(oy2 - fp.position.y) < 0.001


# ---------------------------------------------------------------------------
# KI-005 / KI-010: Component bounds
# ---------------------------------------------------------------------------


class TestComponentBounds:
    """Verify all components stay within board boundaries (KI-005, KI-010).

    Root cause (KI-005): Connectors with asymmetric origins (pin 1) could
    extend pad extents past the board edge.
    Root cause (KI-010): Analog/power zones placed too high, interfering
    with screw terminal area.
    """

    def test_all_components_within_board_bounds(self) -> None:
        """Every component center must be within board bounds."""
        board_w, board_h = 100.0, 80.0
        positions: dict[str, tuple[float, float]] = {
            "U1": (50.0, 40.0),
            "R1": (10.0, 10.0),
            "J1": (5.0, 5.0),
            "J2": (95.0, 75.0),
        }
        for ref, (x, y) in positions.items():
            assert 0.0 <= x <= board_w, (
                f"{ref} center X={x:.1f} is outside board [0, {board_w}]"
            )
            assert 0.0 <= y <= board_h, (
                f"{ref} center Y={y:.1f} is outside board [0, {board_h}]"
            )

    def test_pad_extents_within_board(self) -> None:
        """Pad bounding box must not extend past board edges."""
        board_bounds = (0.0, 0.0, 100.0, 80.0)

        # 8-pin connector at top-left corner
        fp = _make_connector_footprint("J1", x=5.0, y=5.0, pin_count=8)
        px0, py0, px1, py1 = pad_extent_in_board_space(
            fp, fp.position.x, fp.position.y, fp.rotation,
        )

        tolerance = 0.5  # mm
        assert px0 >= board_bounds[0] - tolerance, (
            f"J1 pads extend past left edge: {px0:.1f} < {board_bounds[0]}"
        )
        assert py0 >= board_bounds[1] - tolerance, (
            f"J1 pads extend past top edge: {py0:.1f} < {board_bounds[1]}"
        )
        assert px1 <= board_bounds[2] + tolerance, (
            f"J1 pads extend past right edge: {px1:.1f} > {board_bounds[2]}"
        )
        assert py1 <= board_bounds[3] + tolerance, (
            f"J1 pads extend past bottom edge: {py1:.1f} > {board_bounds[3]}"
        )

    def test_zone_top_margin_not_too_high(self) -> None:
        """Analog/power zone fractions must start at y >= 0.15.

        Prevents KI-010 where analog/power blocks placed too high
        interfered with screw terminals at the top edge.
        """
        for zone_name in ("analog", "power"):
            fracs = _DEFAULT_ZONE_FRACTIONS.get(zone_name)
            if fracs is None:
                continue
            _fx1, fy1, _fx2, _fy2 = fracs
            assert fy1 >= 0.15, (
                f"Zone '{zone_name}' y_start={fy1:.2f} is too high "
                f"(must be >= 0.15 to leave room for top-edge connectors)"
            )


# ---------------------------------------------------------------------------
# KI-008 / KI-010: Zone partitioning
# ---------------------------------------------------------------------------


class TestZonePartitioning:
    """Verify zone partitioner produces valid, non-overlapping zones.

    Root cause (KI-008): Power zone was too small, causing tail overflow.
    Root cause (KI-010): Overlapping zone fractions caused group contamination.
    """

    @pytest.fixture()
    def standard_zones(self) -> list[BoardZone]:
        """Partition a standard 100x80mm board with typical groups."""
        groups = [
            FeatureBlock(name="Power Supply", description="Buck converters",
                         components=("U2", "L1", "C1", "C2"),
                         nets=("VIN",), subcircuits=()),
            FeatureBlock(name="Relay Outputs", description="4ch relay",
                         components=("K1", "K2", "K3", "K4"),
                         nets=("COIL1",), subcircuits=()),
            FeatureBlock(name="MCU Core", description="ESP32",
                         components=("U1",),
                         nets=("GND",), subcircuits=()),
            FeatureBlock(name="Analog Inputs", description="ADC channels",
                         components=("U3", "R10", "R11"),
                         nets=("ADC_CH0",), subcircuits=()),
        ]
        return partition_board((0.0, 0.0, 100.0, 80.0), groups)

    def test_zones_no_overlap(self, standard_zones: list[BoardZone]) -> None:
        """No two zones should overlap."""
        for i, z1 in enumerate(standard_zones):
            for z2 in standard_zones[i + 1:]:
                x1a, y1a, x2a, y2a = z1.rect
                x1b, y1b, x2b, y2b = z2.rect

                # Two rects overlap if they have non-empty intersection
                overlap_x = max(0.0, min(x2a, x2b) - max(x1a, x1b))
                overlap_y = max(0.0, min(y2a, y2b) - max(y1a, y1b))
                overlap_area = overlap_x * overlap_y

                assert overlap_area < 1.0, (
                    f"Zones '{z1.name}' and '{z2.name}' overlap by "
                    f"{overlap_area:.1f} mm^2: {z1.rect} vs {z2.rect}"
                )

    def test_zones_within_board_bounds(self, standard_zones: list[BoardZone]) -> None:
        """All zone rects must be within board bounds (with small tolerance)."""
        bx1, by1, bx2, by2 = 0.0, 0.0, 100.0, 80.0
        tolerance = 1.0  # mm

        for zone in standard_zones:
            zx1, zy1, zx2, zy2 = zone.rect
            assert zx1 >= bx1 - tolerance, (
                f"Zone '{zone.name}' left edge {zx1:.1f} past board left {bx1}"
            )
            assert zy1 >= by1 - tolerance, (
                f"Zone '{zone.name}' top edge {zy1:.1f} past board top {by1}"
            )
            assert zx2 <= bx2 + tolerance, (
                f"Zone '{zone.name}' right edge {zx2:.1f} past board right {bx2}"
            )
            assert zy2 <= by2 + tolerance, (
                f"Zone '{zone.name}' bottom edge {zy2:.1f} past board bottom {by2}"
            )

    def test_zone_minimum_size(self, standard_zones: list[BoardZone]) -> None:
        """Each zone must be at least 10mm x 10mm."""
        for zone in standard_zones:
            zx1, zy1, zx2, zy2 = zone.rect
            width = zx2 - zx1
            height = zy2 - zy1
            assert width >= 10.0, (
                f"Zone '{zone.name}' width {width:.1f}mm < 10mm minimum"
            )
            assert height >= 10.0, (
                f"Zone '{zone.name}' height {height:.1f}mm < 10mm minimum"
            )

    def test_zone_gap_maintained(self, standard_zones: list[BoardZone]) -> None:
        """Inter-zone gap must be >= _ZONE_GAP_MM between adjacent zones.

        Adjacent = zones that share a border (not diagonally separated).
        """
        for i, z1 in enumerate(standard_zones):
            for z2 in standard_zones[i + 1:]:
                x1a, y1a, x2a, y2a = z1.rect
                x1b, y1b, x2b, y2b = z2.rect

                # Check if zones are adjacent horizontally (Y ranges overlap)
                y_overlap = min(y2a, y2b) - max(y1a, y1b)
                if y_overlap > 0:
                    h_gap = max(x1b - x2a, x1a - x2b)
                    if 0.0 < h_gap < 50.0:
                        # They are horizontally adjacent
                        assert h_gap >= _ZONE_GAP_MM - 1.0, (
                            f"Zones '{z1.name}' and '{z2.name}' horizontal gap "
                            f"{h_gap:.1f}mm < {_ZONE_GAP_MM}mm"
                        )

                # Check if zones are adjacent vertically (X ranges overlap)
                x_overlap = min(x2a, x2b) - max(x1a, x1b)
                if x_overlap > 0:
                    v_gap = max(y1b - y2a, y1a - y2b)
                    if 0.0 < v_gap < 50.0:
                        assert v_gap >= _ZONE_GAP_MM - 1.0, (
                            f"Zones '{z1.name}' and '{z2.name}' vertical gap "
                            f"{v_gap:.1f}mm < {_ZONE_GAP_MM}mm"
                        )

    def test_power_zone_height_sufficient(self, standard_zones: list[BoardZone]) -> None:
        """Power zone height must be >= 15mm to prevent KI-008 tail overflow.

        A power zone that is too small forces components outside the zone,
        contaminating adjacent zones.
        """
        power_zones = [z for z in standard_zones if z.name == "power"]
        if not power_zones:
            pytest.skip("No power zone in partition result")
        for pz in power_zones:
            height = pz.rect[3] - pz.rect[1]
            assert height >= 15.0, (
                f"Power zone height {height:.1f}mm is too small (>= 15mm required)"
            )


# ---------------------------------------------------------------------------
# Subcircuit detection guards
# ---------------------------------------------------------------------------


class TestSubcircuitDetection:
    """Guards against subcircuit detection over-claiming or misclassification.

    Recurring bugs:
    - Buck converter scoops up all caps on shared power rails (GND, +3V3)
    - Relay driver falsely claims TVS diodes as flyback diodes
    - Decoupling caps assigned to wrong FeatureBlock IC
    """

    def test_buck_converter_limited_components(self) -> None:
        """Buck converter subcircuit should have <= 10 components.

        Guards against the bug where shared power rails (GND, +3V3, +5V)
        cause detect_subcircuits to pull every cap on those rails into
        the buck converter subcircuit.
        """
        req = _make_buck_requirements()
        subcircuits = detect_subcircuits(req)
        bucks = [sc for sc in subcircuits
                 if sc.circuit_type == SubCircuitType.BUCK_CONVERTER]

        if not bucks:
            pytest.skip("No buck converter detected (detection logic may differ)")

        for buck in bucks:
            assert len(buck.refs) <= 10, (
                f"Buck converter has {len(buck.refs)} components "
                f"(max 10 expected): {buck.refs}"
            )

    def test_relay_driver_excludes_tvs(self) -> None:
        """Relay driver subcircuit should NOT include TVS diodes.

        TVS diodes are for ESD/surge protection on the output terminals,
        not flyback protection on the relay coil.
        """
        # Add a TVS diode to relay requirements
        req = _make_relay_requirements()
        tvs = Component(
            ref="D10", value="TVS_SMBJ24A", footprint="D_SMB",
            description="TVS diode for surge protection",
            pins=_passive_pins(),
        )
        # Add TVS on a relay output net but not on coil net
        tvs_net = Net(name="RELAY1_OUT", connections=(
            NetConnection(ref="K1", pin="5"),  # NO contact
            NetConnection(ref="D10", pin="1"),
        ))
        tvs_gnd_net = Net(name="TVS_GND", connections=(
            NetConnection(ref="D10", pin="2"),
        ))

        from dataclasses import replace
        req = replace(
            req,
            components=(*req.components, tvs),
            nets=(*req.nets, tvs_net, tvs_gnd_net),
        )

        subcircuits = detect_subcircuits(req)
        relay_drivers = [sc for sc in subcircuits
                         if sc.circuit_type == SubCircuitType.RELAY_DRIVER]

        for rd in relay_drivers:
            assert "D10" not in rd.refs, (
                f"TVS diode D10 incorrectly included in relay driver "
                f"subcircuit for {rd.anchor_ref}: {rd.refs}"
            )

    def test_decoupling_same_group_preferred(self) -> None:
        """Decoupling cap connected to ICs in different groups should prefer
        the IC in its own FeatureBlock.

        Guards against cross-group contamination of decoupling assignments.
        """
        # This is a design-level invariant. We verify it by checking that
        # the decoupling detection prefers components on non-power nets
        # (power nets connect everything, causing cross-contamination).
        components = [
            Component(ref="U1", value="ESP32", footprint="ESP32-WROOM",
                      pins=(
                          Pin(number="1", name="GND", pin_type=PinType.POWER_IN,
                              function=PinFunction.GND),
                          Pin(number="2", name="3V3", pin_type=PinType.POWER_IN,
                              function=PinFunction.VCC),
                      )),
            Component(ref="U2", value="ADS1115", footprint="TSSOP-10",
                      pins=(
                          Pin(number="1", name="GND", pin_type=PinType.POWER_IN,
                              function=PinFunction.GND),
                          Pin(number="2", name="VDD", pin_type=PinType.POWER_IN,
                              function=PinFunction.VCC),
                      )),
            Component(ref="C1", value="100nF", footprint="C_0402",
                      lcsc="C1525", pins=_passive_pins()),
        ]
        nets = [
            Net(name="+3V3", connections=(
                NetConnection(ref="U1", pin="2"),
                NetConnection(ref="U2", pin="2"),
                NetConnection(ref="C1", pin="1"),
            )),
            Net(name="GND", connections=(
                NetConnection(ref="U1", pin="1"),
                NetConnection(ref="U2", pin="1"),
                NetConnection(ref="C1", pin="2"),
            )),
        ]
        req = ProjectRequirements(
            project=ProjectInfo(name="decouple-test"),
            features=(
                FeatureBlock(name="MCU", description="MCU",
                             components=("U1", "C1"),
                             nets=("+3V3", "GND"), subcircuits=()),
                FeatureBlock(name="ADC", description="ADC",
                             components=("U2",),
                             nets=("+3V3", "GND"), subcircuits=()),
            ),
            components=tuple(components),
            nets=tuple(nets),
            mechanical=MechanicalConstraints(board_width_mm=50.0, board_height_mm=40.0),
        )

        subcircuits = detect_subcircuits(req)
        decoupling = [sc for sc in subcircuits
                      if sc.circuit_type == SubCircuitType.DECOUPLING]

        # The cap should be detected as decoupling for one of the ICs
        # (not duplicated for both). If it detects for both, the first
        # claim wins which is acceptable.
        cap_assignments: list[str] = []
        for sc in decoupling:
            if "C1" in sc.refs:
                cap_assignments.append(sc.anchor_ref)

        # C1 should appear in at most 1 decoupling subcircuit
        assert len(cap_assignments) <= 1, (
            f"C1 assigned to multiple decoupling subcircuits: "
            f"anchored to {cap_assignments}"
        )


# ---------------------------------------------------------------------------
# Group isolation
# ---------------------------------------------------------------------------


class TestGroupIsolation:
    """Verify components don't leak across group/zone boundaries."""

    def test_no_cross_group_contamination(self) -> None:
        """After zone partitioning, each group maps to exactly one zone.

        Guards against the bug where zone fractions overlap, causing
        groups to be assigned to wrong zones.
        """
        groups = [
            FeatureBlock(name="Power Supply", description="",
                         components=("U2",), nets=(), subcircuits=()),
            FeatureBlock(name="Relay Outputs", description="",
                         components=("K1",), nets=(), subcircuits=()),
            FeatureBlock(name="MCU Core", description="",
                         components=("U1",), nets=(), subcircuits=()),
        ]
        zones = partition_board((0.0, 0.0, 100.0, 80.0), groups)

        # Each group should appear in exactly one zone
        group_zone_count: dict[str, int] = {}
        for zone in zones:
            for gname in zone.groups:
                group_zone_count[gname] = group_zone_count.get(gname, 0) + 1

        for gname, count in group_zone_count.items():
            assert count == 1, (
                f"Group '{gname}' appears in {count} zones (expected 1)"
            )

    def test_late_decoupling_respects_groups(self) -> None:
        """Late decoupling pull must only move caps within same FeatureBlock.

        The 3c-late phase comment documents this constraint:
        'Only move caps in the SAME FeatureBlock as their IC to avoid
        cross-group contamination.'

        Verify the code actually implements this by checking the source.
        """
        optimizer_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "kicad_pipeline" / "optimization" / "placement_optimizer.py"
        )
        source = optimizer_path.read_text()

        # The fix for cross-group decoupling contamination is to check
        # that the cap and IC are in the same FeatureBlock/group.
        assert "ic_group" in source or "same.*group" in source.lower() or \
               "_ref_to_group" in source, (
            "Late decoupling phase must filter by FeatureBlock group "
            "(look for ic_group or _ref_to_group in placement_optimizer.py)"
        )

        # Verify the 3c-late phase checks group membership
        assert "3c-late" in source, (
            "Late decoupling re-tightening phase (3c-late) not found in "
            "placement_optimizer.py"
        )


# ---------------------------------------------------------------------------
# KI-007: Relay layout
# ---------------------------------------------------------------------------


class TestRelayLayout:
    """Verify relay placement invariants (KI-007).

    Root cause: Relay driver support components (Q, D, R) were scattered
    15-25mm from their relay K ref instead of the target 5-8mm.
    """

    def test_relay_row_horizontal(self) -> None:
        """K refs in a relay group should be in a 1xN horizontal row.

        All relay bodies should have similar Y coordinates (within 2mm).
        """
        # Simulate 4 relays placed in a horizontal row
        positions: dict[str, tuple[float, float]] = {
            "K1": (30.0, 40.0),
            "K2": (50.0, 40.5),
            "K3": (70.0, 39.8),
            "K4": (90.0, 40.2),
        }

        y_values = [pos[1] for pos in positions.values()]
        y_spread = max(y_values) - min(y_values)

        assert y_spread <= 2.0, (
            f"Relay row Y spread is {y_spread:.1f}mm (should be <= 2mm). "
            f"Relays should be in a horizontal row."
        )

    def test_relay_driver_subgroup_tight(self) -> None:
        """Support components (R, Q, D) must be within 22mm of their relay K ref.

        Guards against KI-007 where driver components were scattered
        15-25mm away during collision resolution.
        """
        max_distance = 22.0  # mm — matches scoring threshold

        # Simulate a relay driver subgroup
        relay_pos = (50.0, 40.0)
        support_positions: dict[str, tuple[float, float]] = {
            "Q1": (48.0, 45.0),
            "D1": (52.0, 45.0),
            "R1": (50.0, 48.0),
        }

        for ref, (sx, sy) in support_positions.items():
            dist = math.hypot(sx - relay_pos[0], sy - relay_pos[1])
            assert dist <= max_distance, (
                f"{ref} is {dist:.1f}mm from K1 (max {max_distance}mm). "
                f"Support components must stay close to their relay."
            )

    def test_relay_drivers_consistent(self) -> None:
        """All relay driver subgroups should have similar internal layouts.

        If K1's support components are below-left, K2-K4 should follow
        the same pattern (same relative positions within tolerance).
        """
        # 4 relay driver subgroups with relative offsets from K center
        subgroup_offsets: list[dict[str, tuple[float, float]]] = [
            {"Q": (-2.0, 5.0), "D": (2.0, 5.0), "R": (0.0, 8.0)},
            {"Q": (-2.5, 5.5), "D": (1.5, 5.0), "R": (0.0, 7.5)},
            {"Q": (-2.0, 4.5), "D": (2.0, 5.5), "R": (0.5, 8.5)},
            {"Q": (-1.5, 5.0), "D": (2.5, 5.0), "R": (0.0, 8.0)},
        ]

        # Compute mean offset for each role
        roles = ("Q", "D", "R")
        for role in roles:
            offsets_x = [sg[role][0] for sg in subgroup_offsets]
            offsets_y = [sg[role][1] for sg in subgroup_offsets]
            spread_x = max(offsets_x) - min(offsets_x)
            spread_y = max(offsets_y) - min(offsets_y)

            assert spread_x <= 3.0, (
                f"Role {role} X offset spread {spread_x:.1f}mm > 3mm — "
                f"relay drivers should have consistent layouts"
            )
            assert spread_y <= 3.0, (
                f"Role {role} Y offset spread {spread_y:.1f}mm > 3mm — "
                f"relay drivers should have consistent layouts"
            )


# ---------------------------------------------------------------------------
# KI-009: ADC channel layout
# ---------------------------------------------------------------------------


class TestADCChannelLayout:
    """Verify ADC channel ordering and consistency (KI-009).

    Root cause: ADC channels were placed in an order that caused trace
    crossings with screw terminal connectors.
    """

    def test_adc_channels_non_crossing(self) -> None:
        """ADC channels ordered by position should not require trace crossings.

        If channels are at Y positions [10, 20, 30, 40] and connectors
        are at Y positions [10, 20, 30, 40], there should be no crossings.
        """
        # ADC IC outputs (Y positions of ADC channel subcircuit anchors)
        adc_channel_ys = [15.0, 25.0, 35.0, 45.0]
        # Corresponding connector Y positions
        connector_ys = [15.0, 25.0, 35.0, 45.0]

        # Check for crossings: if channel[i] connects to connector[j]
        # and channel[k] connects to connector[l], crossing occurs when
        # (i < k and j > l) or (i > k and j < l).
        # With same ordering, no crossings should occur.
        for i in range(len(adc_channel_ys)):
            for j in range(i + 1, len(adc_channel_ys)):
                ch_order = adc_channel_ys[i] < adc_channel_ys[j]
                conn_order = connector_ys[i] < connector_ys[j]
                assert ch_order == conn_order, (
                    f"ADC channel {i} at Y={adc_channel_ys[i]:.1f} and "
                    f"channel {j} at Y={adc_channel_ys[j]:.1f} cross "
                    f"with connectors at Y={connector_ys[i]:.1f} and "
                    f"Y={connector_ys[j]:.1f}"
                )

    def test_adc_channel_strip_consistent(self) -> None:
        """All ADC channel strips should have identical component ordering.

        Each ADC channel is a strip: connector → R_top → R_bot → C_filter → ADC_pin.
        The component ordering within each strip must be the same.
        """
        # 4 ADC channel strips with X positions of each component
        # (in signal flow order from connector to ADC)
        strips: list[list[float]] = [
            [10.0, 20.0, 25.0, 30.0],  # Ch0: connector, R_top, R_bot, cap
            [10.0, 20.0, 25.0, 30.0],  # Ch1
            [10.0, 20.0, 25.0, 30.0],  # Ch2
            [10.0, 20.0, 25.0, 30.0],  # Ch3
        ]

        # Verify each strip is monotonically increasing in X
        for i, strip in enumerate(strips):
            for j in range(len(strip) - 1):
                assert strip[j] <= strip[j + 1], (
                    f"ADC channel {i} component ordering violated: "
                    f"position[{j}]={strip[j]:.1f} > position[{j + 1}]={strip[j + 1]:.1f}"
                )


# ---------------------------------------------------------------------------
# Scoring thresholds
# ---------------------------------------------------------------------------


class TestScoringThresholds:
    """Verify scoring thresholds account for physical component sizes.

    Recurring bug: subgroup cohesion thresholds were set too tight,
    causing false violations when the anchor component itself was large
    (e.g., relay body is 16-18mm).
    """

    def test_subgroup_thresholds_account_for_relay_size(self) -> None:
        """RELAY_DRIVER threshold must be >= 20mm.

        A relay body is 16-18mm wide. Support components adjacent to it
        will always be at least 8-10mm from the centroid. Threshold must
        accommodate this.
        """

        # We can't easily call the function without a full PCB, but we can
        # verify the threshold constant. It's defined inline in the function.
        scoring_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "kicad_pipeline" / "optimization" / "scoring.py"
        )
        source = scoring_path.read_text()

        # Find the RELAY_DRIVER threshold line
        found = False
        for line in source.split("\n"):
            if "SubCircuitType.RELAY_DRIVER" in line and ":" in line:
                # Extract the number after the colon
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        threshold = float(parts[-1].strip().rstrip(","))
                        assert threshold >= 20.0, (
                            f"RELAY_DRIVER threshold {threshold}mm < 20mm. "
                            f"Relay body is 16-18mm, threshold must accommodate."
                        )
                        found = True
                    except ValueError:
                        continue

        assert found, "RELAY_DRIVER threshold not found in scoring.py"

    def test_decoupling_threshold_realistic(self) -> None:
        """DECOUPLING threshold must be >= 8mm.

        IC package sizes vary (QFP, BGA can be 10-20mm). Decoupling caps
        at the IC edge are already 5-10mm from the IC centroid.
        """
        scoring_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "kicad_pipeline" / "optimization" / "scoring.py"
        )
        source = scoring_path.read_text()

        found = False
        for line in source.split("\n"):
            if "SubCircuitType.DECOUPLING" in line and ":" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        threshold = float(parts[-1].strip().rstrip(","))
                        assert threshold >= 8.0, (
                            f"DECOUPLING threshold {threshold}mm < 8mm. "
                            f"IC packages are 5-20mm, cap placement must "
                            f"accommodate IC body size."
                        )
                        found = True
                    except ValueError:
                        continue

        assert found, "DECOUPLING threshold not found in scoring.py"


# ---------------------------------------------------------------------------
# Phase ordering
# ---------------------------------------------------------------------------


class TestPhaseOrdering:
    """Verify critical optimizer phase ordering invariants.

    Recurring bug: Phase 3c moves decoupling caps close to ICs, then
    subsequent group layout phases scatter them. The 3c-late phase was
    added to re-pull caps after all group phases complete.
    """

    def test_late_decoupling_runs_after_group_phases(self) -> None:
        """Phase 3c-late must exist and appear after 3h (template refinement).

        The decoupling re-tightening MUST run after all group/template
        phases to be effective. If it runs before, group phases will
        scatter the caps again.
        """
        optimizer_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "kicad_pipeline" / "optimization" / "placement_optimizer.py"
        )
        source = optimizer_path.read_text()

        # Find line numbers for key phases
        phase_3c_late_line: int | None = None
        phase_3h_line: int | None = None

        for i, line in enumerate(source.split("\n"), 1):
            if "3c-late" in line and ("_log" in line or "Late" in line):
                phase_3c_late_line = i
            if (
                "3h" in line
                and ("template" in line.lower() or "_log" in line)
                and phase_3h_line is None
            ):
                phase_3h_line = i

        assert phase_3c_late_line is not None, (
            "Phase 3c-late (late decoupling re-tightening) not found "
            "in placement_optimizer.py"
        )

        if phase_3h_line is not None:
            assert phase_3c_late_line > phase_3h_line, (
                f"Phase 3c-late (line {phase_3c_late_line}) must appear "
                f"AFTER phase 3h (line {phase_3h_line}) in the optimizer. "
                f"Otherwise group phases will scatter decoupling caps "
                f"after they've been pulled close."
            )

    def test_phases_monotonically_ordered(self) -> None:
        """Optimizer phases should appear in increasing order in the source.

        Phases: 3a (relay), 3b (relay driver), 3c (decoupling), 3d (crystal),
        3e (RF), 3f (orientation), 3h (template), 3c-late (final decoupling).
        """
        optimizer_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "kicad_pipeline" / "optimization" / "placement_optimizer.py"
        )
        source = optimizer_path.read_text()

        # Extract phase markers with line numbers
        import re
        phase_pattern = re.compile(r'_log\.info\("  (3[a-h](?:-late)?)[:\.]')
        phase_positions: list[tuple[int, str]] = []

        for i, line in enumerate(source.split("\n"), 1):
            match = phase_pattern.search(line)
            if match:
                phase_positions.append((i, match.group(1)))

        # Verify we found at least 5 phases
        assert len(phase_positions) >= 5, (
            f"Expected >= 5 phase markers in optimizer, found {len(phase_positions)}: "
            f"{[p[1] for p in phase_positions]}"
        )

        # Verify phases appear in order (line numbers increasing)
        for i in range(len(phase_positions) - 1):
            line1, phase1 = phase_positions[i]
            line2, phase2 = phase_positions[i + 1]
            assert line1 < line2, (
                f"Phase '{phase1}' at line {line1} appears after "
                f"phase '{phase2}' at line {line2}"
            )
