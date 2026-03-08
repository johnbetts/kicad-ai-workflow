"""Tests for the schematic builder, placement, and wiring modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

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
from kicad_pipeline.models.schematic import Schematic
from kicad_pipeline.schematic.builder import (
    build_schematic,
    schematic_to_sexp,
    write_schematic,
)
from kicad_pipeline.schematic.placement import (
    SCHEMATIC_ZONES,
    assign_zones,
    layout_schematic,
    place_in_zone,
    snap_to_grid,
)
from kicad_pipeline.schematic.wiring import (
    make_global_label,
    make_junction,
    make_label,
    make_wire,
    route_net,
)
from kicad_pipeline.sexp.parser import parse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_requirements() -> ProjectRequirements:
    """Return a minimal ProjectRequirements with 1 component and 1 net."""
    comp = Component(
        ref="R1",
        value="10k",
        footprint="R_0805",
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, net="VIN"),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, net="GND"),
        ),
    )
    net_vin = Net(
        name="VIN",
        connections=(NetConnection(ref="R1", pin="1"),),
    )
    net_gnd = Net(
        name="GND",
        connections=(NetConnection(ref="R1", pin="2"),),
    )
    fb = FeatureBlock(
        name="Power",
        description="Power supply",
        components=("R1",),
        nets=("VIN", "GND"),
        subcircuits=(),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="MinimalTest"),
        features=(fb,),
        components=(comp,),
        nets=(net_vin, net_gnd),
    )


def _multi_component_requirements() -> ProjectRequirements:
    """Return requirements with MCU + power components for zone testing."""
    mcu = Component(
        ref="U1",
        value="ESP32",
        footprint="ESP32-WROOM",
        pins=(
            Pin(number="1", name="VCC", pin_type=PinType.POWER_IN, net="+3V3"),
            Pin(number="2", name="GND", pin_type=PinType.POWER_IN, net="GND"),
            Pin(number="3", name="GPIO4", pin_type=PinType.BIDIRECTIONAL, net="LED_NET"),
        ),
    )
    r1 = Component(
        ref="R1",
        value="220R",
        footprint="R_0805",
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, net="LED_NET"),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, net="GND"),
        ),
    )
    nets = (
        Net(name="+3V3", connections=(NetConnection(ref="U1", pin="1"),)),
        Net(name="GND", connections=(
            NetConnection(ref="U1", pin="2"),
            NetConnection(ref="R1", pin="2"),
        )),
        Net(name="LED_NET", connections=(
            NetConnection(ref="U1", pin="3"),
            NetConnection(ref="R1", pin="1"),
        )),
    )
    fb_mcu = FeatureBlock(
        name="MCU",
        description="Microcontroller",
        components=("U1",),
        nets=("+3V3", "GND"),
        subcircuits=(),
    )
    fb_per = FeatureBlock(
        name="Peripherals",
        description="Peripheral components",
        components=("R1",),
        nets=("LED_NET",),
        subcircuits=(),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="MultiTest"),
        features=(fb_mcu, fb_per),
        components=(mcu, r1),
        nets=nets,
    )


# ---------------------------------------------------------------------------
# build_schematic tests
# ---------------------------------------------------------------------------


def test_build_schematic_minimal() -> None:
    """build_schematic produces a result from minimal requirements."""
    reqs = _minimal_requirements()
    result = build_schematic(reqs)
    assert result is not None


def test_build_schematic_returns_schematic_type() -> None:
    """build_schematic returns a Schematic instance."""
    reqs = _minimal_requirements()
    result = build_schematic(reqs)
    assert isinstance(result, Schematic)


def test_build_schematic_has_lib_symbols() -> None:
    """Built schematic has at least one lib_symbol."""
    reqs = _minimal_requirements()
    result = build_schematic(reqs)
    assert len(result.lib_symbols) >= 1


def test_build_schematic_has_symbols() -> None:
    """Built schematic has symbol instances equal to number of components."""
    reqs = _minimal_requirements()
    result = build_schematic(reqs)
    assert len(result.symbols) == len(reqs.components)


def test_build_schematic_symbol_refs_match() -> None:
    """Symbol instance refs match component refs from requirements."""
    reqs = _multi_component_requirements()
    result = build_schematic(reqs)
    inst_refs = {s.ref for s in result.symbols}
    comp_refs = {c.ref for c in reqs.components}
    assert comp_refs.issubset(inst_refs)


def test_build_schematic_no_components_raises() -> None:
    """build_schematic raises SchematicError when no components are present."""
    from kicad_pipeline.exceptions import SchematicError

    reqs = ProjectRequirements(
        project=ProjectInfo(name="Empty"),
        features=(),
        components=(),
        nets=(),
    )
    with pytest.raises(SchematicError):
        build_schematic(reqs)


def test_build_schematic_has_power_symbols_for_gnd() -> None:
    """GND net produces at least one PowerSymbol."""
    reqs = _minimal_requirements()
    result = build_schematic(reqs)
    power_values = {ps.value for ps in result.power_symbols}
    assert "GND" in power_values


# ---------------------------------------------------------------------------
# schematic_to_sexp tests
# ---------------------------------------------------------------------------


def test_schematic_to_sexp_is_list() -> None:
    """schematic_to_sexp returns a list (root S-expression node)."""
    reqs = _minimal_requirements()
    sch = build_schematic(reqs)
    sexp = schematic_to_sexp(sch)
    assert isinstance(sexp, list)


def test_schematic_to_sexp_starts_with_kicad_sch() -> None:
    """First element of the root node is 'kicad_sch'."""
    reqs = _minimal_requirements()
    sch = build_schematic(reqs)
    sexp = schematic_to_sexp(sch)
    assert sexp[0] == "kicad_sch"


def test_schematic_to_sexp_contains_version() -> None:
    """S-expression tree contains a (version ...) node."""
    from kicad_pipeline.constants import KICAD_SCH_VERSION

    reqs = _minimal_requirements()
    sch = build_schematic(reqs)
    sexp = schematic_to_sexp(sch)
    assert ["version", KICAD_SCH_VERSION] in sexp


def test_schematic_to_sexp_contains_lib_symbols() -> None:
    """S-expression tree contains a (lib_symbols ...) node."""
    reqs = _minimal_requirements()
    sch = build_schematic(reqs)
    sexp = schematic_to_sexp(sch)
    has_lib_symbols = any(
        isinstance(n, list) and len(n) > 0 and n[0] == "lib_symbols" for n in sexp
    )
    assert has_lib_symbols


# ---------------------------------------------------------------------------
# write_schematic tests
# ---------------------------------------------------------------------------


def test_write_schematic_creates_file(tmp_path: Path) -> None:
    """write_schematic() creates a file at the given path."""
    reqs = _minimal_requirements()
    sch = build_schematic(reqs)
    dest = tmp_path / "test.kicad_sch"
    write_schematic(sch, dest)
    assert dest.exists()


def test_write_schematic_file_not_empty(tmp_path: Path) -> None:
    """Written .kicad_sch file is non-empty."""
    reqs = _minimal_requirements()
    sch = build_schematic(reqs)
    dest = tmp_path / "test.kicad_sch"
    write_schematic(sch, dest)
    assert dest.stat().st_size > 0


def test_write_schematic_file_starts_with_kicad_sch(tmp_path: Path) -> None:
    """Written file content starts with '(kicad_sch'."""
    reqs = _minimal_requirements()
    sch = build_schematic(reqs)
    dest = tmp_path / "test.kicad_sch"
    write_schematic(sch, dest)
    content = dest.read_text(encoding="utf-8")
    assert content.startswith("(kicad_sch")


def test_schematic_sexp_roundtrip(tmp_path: Path) -> None:
    """Written .kicad_sch can be parsed back by sexp.parser."""
    reqs = _minimal_requirements()
    sch = build_schematic(reqs)
    dest = tmp_path / "roundtrip.kicad_sch"
    write_schematic(sch, dest)
    content = dest.read_text(encoding="utf-8")
    parsed = parse(content)
    assert isinstance(parsed, list)
    assert parsed[0] == "kicad_sch"


# ---------------------------------------------------------------------------
# snap_to_grid tests
# ---------------------------------------------------------------------------


def test_snap_to_grid() -> None:
    """snap_to_grid rounds to the nearest grid multiple."""
    assert abs(snap_to_grid(1.3, 1.27) - 1.27) < 1e-9


def test_snap_to_grid_exact() -> None:
    """snap_to_grid returns exact grid multiples unchanged."""
    assert abs(snap_to_grid(2.54, 1.27) - 2.54) < 1e-9


def test_snap_to_grid_zero() -> None:
    """snap_to_grid(0) → 0."""
    assert snap_to_grid(0.0, 1.27) == 0.0


def test_snap_to_grid_negative() -> None:
    """snap_to_grid handles negative values."""
    snapped = snap_to_grid(-1.3, 1.27)
    assert abs(snapped - (-1.27)) < 1e-9


# ---------------------------------------------------------------------------
# make_wire / make_junction / make_global_label tests
# ---------------------------------------------------------------------------


def test_make_wire_has_uuid() -> None:
    """make_wire() generates a non-empty UUID string."""
    wire = make_wire(0.0, 0.0, 10.0, 0.0)
    assert isinstance(wire.uuid, str)
    assert len(wire.uuid) > 0


def test_make_wire_endpoints() -> None:
    """make_wire() sets start and end points correctly."""
    wire = make_wire(0.0, 0.0, 10.16, 0.0)
    assert wire.start.x == pytest.approx(0.0)
    assert wire.end.x == pytest.approx(10.16)


def test_make_junction_has_uuid() -> None:
    """make_junction() generates a non-empty UUID string."""
    j = make_junction(5.0, 5.0)
    assert len(j.uuid) > 0


def test_make_global_label() -> None:
    """make_global_label creates a GlobalLabel with the correct text."""
    gl = make_global_label("GND", 10.0, 20.0)
    assert gl.text == "GND"


def test_make_global_label_has_uuid() -> None:
    """make_global_label creates a GlobalLabel with a non-empty UUID."""
    gl = make_global_label("VCC", 0.0, 0.0)
    assert len(gl.uuid) > 0


def test_make_label() -> None:
    """make_label creates a Label with the correct text."""
    label = make_label("NET_A", 5.0, 5.0)
    assert label.text == "NET_A"


# ---------------------------------------------------------------------------
# PlacementZone / SCHEMATIC_ZONES tests
# ---------------------------------------------------------------------------


def test_placement_zones_defined() -> None:
    """SCHEMATIC_ZONES has at least MCU and POWER zones."""
    assert "MCU" in SCHEMATIC_ZONES
    assert "POWER" in SCHEMATIC_ZONES


def test_placement_zones_all_defined() -> None:
    """All four standard zones are present."""
    for zone_name in ("POWER", "MCU", "ANALOG", "PERIPHERALS"):
        assert zone_name in SCHEMATIC_ZONES


def test_placement_zone_is_frozen() -> None:
    """PlacementZone is immutable."""
    zone = SCHEMATIC_ZONES["MCU"]
    with pytest.raises((AttributeError, TypeError)):
        zone.name = "HACKED"  # type: ignore[misc]


def test_assign_zones_mcu() -> None:
    """MCU components are assigned to the MCU zone."""
    result = assign_zones([("U1", "MCU")])
    assert result["U1"].name == "MCU"


def test_assign_zones_power() -> None:
    """Power components are assigned to the POWER zone."""
    result = assign_zones([("C1", "Power")])
    assert result["C1"].name == "POWER"


def test_assign_zones_usb() -> None:
    """USB components are assigned to the POWER zone."""
    result = assign_zones([("J1", "USB")])
    assert result["J1"].name == "POWER"


def test_assign_zones_ethernet() -> None:
    """Ethernet components are assigned to the PERIPHERALS zone (slot 3)."""
    result = assign_zones([("U3", "Ethernet")])
    assert result["U3"].name == "PERIPHERALS"


def test_assign_zones_unknown() -> None:
    """Unknown feature falls back to PERIPHERALS zone."""
    result = assign_zones([("Q1", "SomeRandomFeature")])
    assert result["Q1"].name == "PERIPHERALS"


# ---------------------------------------------------------------------------
# layout_schematic tests
# ---------------------------------------------------------------------------


def test_layout_schematic_returns_all_refs() -> None:
    """layout_schematic returns a position for every ref in symbol_refs."""
    refs = ["U1", "R1", "C1"]
    feature_map = {"U1": "MCU", "R1": "Power", "C1": "Power"}
    positions = layout_schematic(refs, feature_map)
    for ref in refs:
        assert ref in positions


def test_layout_schematic_missing_feature_map_entry() -> None:
    """Refs not in feature_map still receive positions (fallback to PERIPHERALS)."""
    refs = ["X1", "X2"]
    positions = layout_schematic(refs, {})
    assert "X1" in positions
    assert "X2" in positions


def test_place_in_zone_grid_snapped() -> None:
    """All positions returned by place_in_zone are grid-snapped."""
    from kicad_pipeline.constants import SCHEMATIC_WIRE_GRID_MM

    zone = SCHEMATIC_ZONES["POWER"]
    positions = place_in_zone(["R1", "R2", "R3", "R4"], zone)
    for pt in positions.values():
        assert abs(pt.x % SCHEMATIC_WIRE_GRID_MM) < 1e-6 or abs(
            pt.x % SCHEMATIC_WIRE_GRID_MM - SCHEMATIC_WIRE_GRID_MM
        ) < 1e-6
        assert abs(pt.y % SCHEMATIC_WIRE_GRID_MM) < 1e-6 or abs(
            pt.y % SCHEMATIC_WIRE_GRID_MM - SCHEMATIC_WIRE_GRID_MM
        ) < 1e-6


# ---------------------------------------------------------------------------
# route_net tests
# ---------------------------------------------------------------------------


def test_route_net_no_positions() -> None:
    """route_net returns empty lists when no pin positions are known."""
    net = Net(
        name="FLOATING",
        connections=(NetConnection(ref="R1", pin="1"),),
    )
    wires, junctions, gls, lbls = route_net(net, {})
    assert wires == []
    assert junctions == []
    assert gls == []
    assert lbls == []


def test_route_net_single_pin_produces_label() -> None:
    """route_net with a single known pin produces a label."""
    from kicad_pipeline.models.schematic import Point

    net = Net(
        name="SIG",
        connections=(NetConnection(ref="R1", pin="1"),),
    )
    positions = {("R1", "1"): Point(x=10.0, y=10.0)}
    wires, _, gls, _ = route_net(net, positions, use_global_labels=True)
    assert len(wires) >= 1
    assert len(gls) >= 1
    assert gls[0].text == "SIG"


def test_route_net_two_same_zone_pins_uses_labels() -> None:
    """Two pins always get label-per-pin routing (no direct wires to avoid crossings)."""
    from kicad_pipeline.models.schematic import Point

    net = Net(
        name="DIRECT",
        connections=(
            NetConnection(ref="R1", pin="1"),
            NetConnection(ref="R2", pin="2"),
        ),
    )
    positions = {
        ("R1", "1"): Point(x=10.0, y=10.0),
        ("R2", "2"): Point(x=20.0, y=20.0),
    }
    wires, _, gls, _ = route_net(net, positions, use_global_labels=True)
    # Label-per-pin: one stub wire + label per pin
    assert len(wires) == 2
    assert len(gls) == 2


# ---------------------------------------------------------------------------
# Page threshold tests
# ---------------------------------------------------------------------------


def test_auto_a3_for_many_components() -> None:
    """Designs with >15 components auto-select A3 page."""
    components = []
    for i in range(16):
        ref = f"R{i + 1}"
        components.append(Component(
            ref=ref, value="10k", footprint="0805",
            pins=(Pin(number="1", name="1", pin_type=PinType.PASSIVE),
                  Pin(number="2", name="2", pin_type=PinType.PASSIVE)),
        ))
    reqs = ProjectRequirements(
        project=ProjectInfo(name="Big"),
        features=(FeatureBlock(
            name="Analog", description="", nets=(), subcircuits=(),
            components=tuple(c.ref for c in components),
        ),),
        components=tuple(components),
        nets=(),
    )
    sch = build_schematic(reqs)
    assert sch.paper == "A3"


def test_auto_a3_for_large_pin_component() -> None:
    """A component with >=20 pins triggers A3 even with few components."""
    pins = tuple(
        Pin(number=str(i + 1), name=f"P{i + 1}", pin_type=PinType.PASSIVE, net="SIG")
        for i in range(20)
    )
    reqs = ProjectRequirements(
        project=ProjectInfo(name="BigChip"),
        features=(FeatureBlock(
            name="MCU", description="", components=("U1",),
            nets=("SIG",), subcircuits=(),
        ),),
        components=(Component(ref="U1", value="IC", footprint="QFP", pins=pins),),
        nets=(Net(name="SIG", connections=(NetConnection(ref="U1", pin="1"),)),),
    )
    sch = build_schematic(reqs)
    assert sch.paper == "A3"


def test_a4_for_small_design() -> None:
    """Designs with <=15 components and <60 active pins stay on A4."""
    reqs = _minimal_requirements()
    sch = build_schematic(reqs)
    assert sch.paper == "A4"


# ---------------------------------------------------------------------------
# Power symbol consolidation tests
# ---------------------------------------------------------------------------


def test_power_consolidation_single_pin_no_bus() -> None:
    """A single GND pin produces one power symbol and one stub wire."""
    reqs = _minimal_requirements()
    sch = build_schematic(reqs)
    gnd_syms = [ps for ps in sch.power_symbols if ps.value == "GND"]
    # At least one GND symbol (from the minimal requirements)
    assert len(gnd_syms) >= 1


def test_power_consolidation_multi_pin_same_side() -> None:
    """Multiple GND pins on the same side of one component produce one symbol."""
    # Build a component with 4 right-side GND pins
    gnd_pins = tuple(
        Pin(number=str(i + 1), name=f"GND{i + 1}", pin_type=PinType.POWER_IN, net="GND")
        for i in range(4)
    )
    sig_pin = Pin(number="5", name="SIG", pin_type=PinType.INPUT, net="SIG1")
    comp = Component(
        ref="J1", value="CONN", footprint="PinHeader_1x05",
        pins=(*gnd_pins, sig_pin),
    )
    reqs = ProjectRequirements(
        project=ProjectInfo(name="MultiGND"),
        features=(FeatureBlock(
            name="Connectors", description="", components=("J1",),
            nets=("GND", "SIG1"), subcircuits=(),
        ),),
        components=(comp,),
        nets=(
            Net(name="GND", connections=tuple(
                NetConnection(ref="J1", pin=str(i + 1)) for i in range(4)
            )),
            Net(name="SIG1", connections=(NetConnection(ref="J1", pin="5"),)),
        ),
    )
    sch = build_schematic(reqs)
    gnd_syms = [ps for ps in sch.power_symbols if ps.value == "GND"]
    # Should be consolidated to exactly 1 GND symbol (all 4 pins same ref+side)
    assert len(gnd_syms) == 1
    # Should have junctions for interior T-connections (4 pins, 2 interior)
    assert len(sch.junctions) >= 2


# ---------------------------------------------------------------------------
# Ref designator '?' validation
# ---------------------------------------------------------------------------


def test_write_schematic_warns_on_unannotated_ref(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """write_schematic emits WARNING when symbols contain '?' refs."""
    import logging
    from dataclasses import replace as dreplace

    from kicad_pipeline.models.schematic import Point, SymbolInstance

    reqs = _minimal_requirements()
    sch = build_schematic(reqs)

    # Inject an unannotated ref into symbols
    bad_inst = SymbolInstance(
        lib_id="Device:R",
        ref="R?",
        value="10k",
        footprint="Resistor_SMD:R_0603_1608Metric",
        position=Point(x=50.0, y=50.0),
    )
    sch_bad = dreplace(sch, symbols=(*sch.symbols, bad_inst))

    dest = tmp_path / "test_warn.kicad_sch"
    with caplog.at_level(logging.WARNING, logger="kicad_pipeline.schematic.builder"):
        write_schematic(sch_bad, dest)

    assert any("R?" in rec.message for rec in caplog.records)
    assert dest.exists()


# ---------------------------------------------------------------------------
# Contextual spacing tests
# ---------------------------------------------------------------------------


def test_place_in_zone_small_passives_tighter_than_ics() -> None:
    """2-pin passives should have tighter vertical spacing than 8+ pin ICs."""
    zone = SCHEMATIC_ZONES["POWER"]

    # 4 passives (2 pins each) — should use tight spacing
    passive_pos = place_in_zone(
        ["R1", "R2", "R3", "R4"],
        zone,
        pin_counts=[2, 2, 2, 2],
        symbols_per_row=2,
    )
    dy_passive = passive_pos["R3"].y - passive_pos["R1"].y

    # 4 ICs (10 pins each) — should use wider spacing
    ic_pos = place_in_zone(
        ["U1", "U2", "U3", "U4"],
        zone,
        pin_counts=[10, 10, 10, 10],
        symbols_per_row=2,
    )
    # U3 is in the small bucket (10 < large_threshold=8 is False, 10>=8 so large)
    # For large: U3 is in second row
    dy_ic = ic_pos["U3"].y - ic_pos["U1"].y

    assert dy_passive < dy_ic, (
        f"Passive vertical spacing ({dy_passive}mm) should be less than "
        f"IC spacing ({dy_ic}mm)"
    )
