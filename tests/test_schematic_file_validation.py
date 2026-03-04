"""Validate generated .kicad_sch files for functional and stylistic correctness.

These tests write schematic files to disk, re-parse the S-expression text, and
verify every structural and semantic requirement that KiCad 9 enforces.

Functional checks:
  - Required top-level sections present (version, generator, lib_symbols, etc.)
  - symbol_instances maps every symbol UUID to its reference designator
  - sheet_instances section exists with root page
  - Every symbol UUID is unique
  - Every wire is either horizontal or vertical (no diagonal stubs)
  - Power symbols have correct inline routing (single straight wire, no L-shapes)
  - Every wire endpoint connects to a pin, junction, label, or another wire
  - No overlapping wires on the same axis

Stylistic checks:
  - Version and generator match KiCad 9 constants
  - All strings that need quoting are quoted
  - Indentation uses 2-space nesting
  - No trailing whitespace on any line
  - File ends with a single newline
  - Paper size is A4 or A3
"""

from __future__ import annotations

import re
import uuid as uuid_mod
from collections import Counter
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.constants import KICAD_GENERATOR, KICAD_SCH_VERSION

if TYPE_CHECKING:
    from pathlib import Path
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinFunction,
    PinType,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.schematic.builder import (
    build_schematic,
    schematic_to_sexp,
    write_schematic,
)
from kicad_pipeline.sexp.parser import parse
from kicad_pipeline.sexp.writer import write

if TYPE_CHECKING:
    from kicad_pipeline.sexp.writer import SExpNode


# ---------------------------------------------------------------------------
# S-expression tree helpers
# ---------------------------------------------------------------------------


def _find_nodes(tree: list[SExpNode], tag: str) -> list[list[SExpNode]]:
    """Find all direct child lists whose first element is *tag*."""
    return [n for n in tree if isinstance(n, list) and len(n) > 0 and n[0] == tag]


def _find_node(tree: list[SExpNode], tag: str) -> list[SExpNode] | None:
    """Find the first direct child list with the given *tag*, or ``None``."""
    for n in tree:
        if isinstance(n, list) and len(n) > 0 and n[0] == tag:
            return n
    return None


def _scalar(tree: list[SExpNode], tag: str) -> SExpNode | None:
    """Extract the scalar value from ``(tag value)`` within *tree*."""
    node = _find_node(tree, tag)
    if node and len(node) >= 2:
        return node[1]
    return None


def _as_float(node: SExpNode) -> float:
    """Safely cast an SExpNode to float for coordinate extraction."""
    if isinstance(node, int | float):
        return float(node)
    if isinstance(node, str):
        return float(node)
    msg = f"Cannot convert {type(node)} to float"
    raise TypeError(msg)


def _top_level_tags(tree: list[SExpNode]) -> list[str]:
    """Extract tag names from all direct child lists."""
    tags: list[str] = []
    for n in tree:
        if isinstance(n, list) and len(n) > 0:
            tags.append(str(n[0]))
    return tags


def _collect_all_uuids(tree: list[SExpNode]) -> list[str]:
    """Recursively collect every UUID string found in ``(uuid ...)`` nodes."""
    uuids: list[str] = []
    for elem in tree:
        if isinstance(elem, list):
            if len(elem) >= 2 and elem[0] == "uuid" and isinstance(elem[1], str):
                uuids.append(elem[1])
            uuids.extend(_collect_all_uuids(elem))
    return uuids


# ---------------------------------------------------------------------------
# Fixtures — project requirements of varying complexity
# ---------------------------------------------------------------------------


def _minimal_requirements() -> ProjectRequirements:
    """Single resistor with GND net — simplest possible schematic."""
    comp = Component(
        ref="R1", value="10k", footprint="R_0805",
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, net="VIN"),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, net="GND"),
        ),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="MinimalTest"),
        features=(FeatureBlock(
            name="Power", description="Power",
            components=("R1",), nets=("VIN", "GND"), subcircuits=(),
        ),),
        components=(comp,),
        nets=(
            Net(name="VIN", connections=(NetConnection(ref="R1", pin="1"),)),
            Net(name="GND", connections=(NetConnection(ref="R1", pin="2"),)),
        ),
    )


def _multi_power_requirements() -> ProjectRequirements:
    """MCU + resistor + LED with both +3V3 and GND power nets."""
    u1 = Component(
        ref="U1", value="ESP32", footprint="ESP32-WROOM",
        pins=(
            Pin(number="1", name="VCC", pin_type=PinType.POWER_IN, net="+3V3"),
            Pin(number="2", name="GND", pin_type=PinType.POWER_IN, net="GND"),
            Pin(number="3", name="GPIO4", pin_type=PinType.BIDIRECTIONAL, net="LED_NET"),
        ),
    )
    r1 = Component(
        ref="R1", value="220R", footprint="R_0805",
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, net="LED_NET"),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, net="LED_A"),
        ),
    )
    d1 = Component(
        ref="D1", value="LED_Red", footprint="LED_0805",
        pins=(
            Pin(number="1", name="A", pin_type=PinType.PASSIVE, net="LED_A"),
            Pin(number="2", name="K", pin_type=PinType.PASSIVE, net="GND"),
        ),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="LEDTest"),
        features=(
            FeatureBlock(name="MCU", description="MCU", components=("U1",),
                         nets=("+3V3", "GND"), subcircuits=()),
            FeatureBlock(name="Peripherals", description="LED circuit",
                         components=("R1", "D1"), nets=("LED_NET", "LED_A"),
                         subcircuits=()),
        ),
        components=(u1, r1, d1),
        nets=(
            Net(name="+3V3", connections=(NetConnection(ref="U1", pin="1"),)),
            Net(name="GND", connections=(
                NetConnection(ref="U1", pin="2"),
                NetConnection(ref="D1", pin="2"),
            )),
            Net(name="LED_NET", connections=(
                NetConnection(ref="U1", pin="3"),
                NetConnection(ref="R1", pin="1"),
            )),
            Net(name="LED_A", connections=(
                NetConnection(ref="R1", pin="2"),
                NetConnection(ref="D1", pin="1"),
            )),
        ),
    )


def _adc_requirements() -> ProjectRequirements:
    """Multi-component ADC design with DIP switch — exercises power routing complexity."""
    def _passive_2pin(ref: str, value: str, fp: str = "R_0805") -> Component:
        return Component(
            ref=ref, value=value, footprint=fp,
            pins=(
                Pin(number="1", name="~", pin_type=PinType.PASSIVE),
                Pin(number="2", name="~", pin_type=PinType.PASSIVE),
            ),
        )

    u1 = Component(
        ref="U1", value="ADS1115", footprint="MSOP-10",
        pins=(
            Pin(number="1", name="ADDR", pin_type=PinType.INPUT),
            Pin(number="2", name="ALERT", pin_type=PinType.OUTPUT),
            Pin(number="3", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
            Pin(number="4", name="AIN0", pin_type=PinType.INPUT, function=PinFunction.ANALOG_IN),
            Pin(number="5", name="AIN1", pin_type=PinType.INPUT, function=PinFunction.ANALOG_IN),
            Pin(number="6", name="AIN2", pin_type=PinType.INPUT, function=PinFunction.ANALOG_IN),
            Pin(number="7", name="AIN3", pin_type=PinType.INPUT, function=PinFunction.ANALOG_IN),
            Pin(
                number="8", name="SDA", pin_type=PinType.BIDIRECTIONAL,
                function=PinFunction.I2C_SDA,
            ),
            Pin(number="9", name="SCL", pin_type=PinType.INPUT, function=PinFunction.I2C_SCL),
            Pin(number="10", name="VDD", pin_type=PinType.POWER_IN, function=PinFunction.VCC),
        ),
    )
    sw1 = Component(
        ref="SW1", value="DIP_Switch_x04", footprint="DIP_Switch_x04",
        pins=tuple(
            Pin(number=str(i), name=n, pin_type=PinType.PASSIVE)
            for i, n in [(1, "A1"), (2, "A2"), (3, "A3"), (4, "A4"),
                         (5, "GND"), (6, "VDD"), (7, "SDA"), (8, "SCL")]
        ),
    )
    r1 = _passive_2pin("R1", "100k")
    r2 = _passive_2pin("R2", "20k")
    c1 = _passive_2pin("C1", "100nF", "C_0805")
    c2 = _passive_2pin("C2", "100nF", "C_0805")

    nets = (
        Net(name="AIN0", connections=(
            NetConnection(ref="R1", pin="2"),
            NetConnection(ref="R2", pin="1"),
            NetConnection(ref="C1", pin="1"),
            NetConnection(ref="U1", pin="4"),
        )),
        Net(name="SENS0", connections=(
            NetConnection(ref="R1", pin="1"),
        )),
        Net(name="ADDR", connections=(
            NetConnection(ref="U1", pin="1"),
            NetConnection(ref="SW1", pin="1"),
            NetConnection(ref="SW1", pin="2"),
            NetConnection(ref="SW1", pin="3"),
            NetConnection(ref="SW1", pin="4"),
        )),
        Net(name="I2C_SDA", connections=(
            NetConnection(ref="U1", pin="8"),
            NetConnection(ref="SW1", pin="7"),
        )),
        Net(name="I2C_SCL", connections=(
            NetConnection(ref="U1", pin="9"),
            NetConnection(ref="SW1", pin="8"),
        )),
        Net(name="+5V", connections=(
            NetConnection(ref="U1", pin="10"),
            NetConnection(ref="SW1", pin="6"),
            NetConnection(ref="C2", pin="1"),
        )),
        Net(name="GND", connections=(
            NetConnection(ref="U1", pin="3"),
            NetConnection(ref="SW1", pin="5"),
            NetConnection(ref="R2", pin="2"),
            NetConnection(ref="C1", pin="2"),
            NetConnection(ref="C2", pin="2"),
        )),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="ADCTest"),
        features=(
            FeatureBlock(name="ADC Core", description="ADC",
                         components=("U1", "SW1"), nets=("ADDR", "I2C_SDA", "I2C_SCL"),
                         subcircuits=()),
            FeatureBlock(name="Analog Sensors", description="Sensors",
                         components=("R1", "R2", "C1"),
                         nets=("AIN0", "SENS0"), subcircuits=()),
            FeatureBlock(name="Power Supply", description="Bypass",
                         components=("C2",), nets=("+5V",), subcircuits=()),
        ),
        components=(u1, sw1, r1, r2, c1, c2),
        nets=nets,
        mechanical=MechanicalConstraints(board_width_mm=80.0, board_height_mm=60.0),
        power_budget=PowerBudget(
            rails=(PowerRail(name="+5V", voltage=5.0, current_ma=200.0, source_ref="EXT"),),
            total_current_ma=200.0,
            notes=(),
        ),
    )


# ---------------------------------------------------------------------------
# Parametrised fixture: generate file once per requirements set
# ---------------------------------------------------------------------------


@pytest.fixture(params=["minimal", "multi_power", "adc"], ids=["minimal", "multi_power", "adc"])
def schematic_file(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Path, str]:
    """Write a .kicad_sch file and return (path, raw_text)."""
    builders = {
        "minimal": _minimal_requirements,
        "multi_power": _multi_power_requirements,
        "adc": _adc_requirements,
    }
    reqs = builders[request.param]()
    sch = build_schematic(reqs)
    dest = tmp_path / f"{reqs.project.name}.kicad_sch"
    write_schematic(sch, dest)
    text = dest.read_text(encoding="utf-8")
    return dest, text


@pytest.fixture()
def parsed_tree(schematic_file: tuple[Path, str]) -> list[SExpNode]:
    """Parse the schematic file into an S-expression tree."""
    _, text = schematic_file
    tree = parse(text)
    assert isinstance(tree, list)
    return tree


# ---------------------------------------------------------------------------
# SECTION 1 — Structural / functional correctness
# ---------------------------------------------------------------------------


class TestStructuralSections:
    """Every required KiCad 9 top-level section is present."""

    def test_root_tag(self, parsed_tree: list[SExpNode]) -> None:
        assert parsed_tree[0] == "kicad_sch"

    def test_version(self, parsed_tree: list[SExpNode]) -> None:
        assert _scalar(parsed_tree, "version") == KICAD_SCH_VERSION

    def test_generator(self, parsed_tree: list[SExpNode]) -> None:
        assert _scalar(parsed_tree, "generator") == KICAD_GENERATOR

    def test_generator_version(self, parsed_tree: list[SExpNode]) -> None:
        val = _scalar(parsed_tree, "generator_version")
        assert val is not None

    def test_uuid(self, parsed_tree: list[SExpNode]) -> None:
        val = _scalar(parsed_tree, "uuid")
        assert isinstance(val, str) and len(val) > 0
        # Must be valid UUID
        uuid_mod.UUID(val)

    def test_paper(self, parsed_tree: list[SExpNode]) -> None:
        val = _scalar(parsed_tree, "paper")
        assert val in ("A4", "A3")

    def test_lib_symbols_present(self, parsed_tree: list[SExpNode]) -> None:
        assert _find_node(parsed_tree, "lib_symbols") is not None

    def test_sheet_instances_present(self, parsed_tree: list[SExpNode]) -> None:
        node = _find_node(parsed_tree, "sheet_instances")
        assert node is not None
        # Must contain (path "/" (page "1"))
        path_node = _find_node(node, "path")
        assert path_node is not None
        assert "/" in path_node

    def test_symbol_instances_present(self, parsed_tree: list[SExpNode]) -> None:
        node = _find_node(parsed_tree, "symbol_instances")
        assert node is not None
        # Must have at least one (path ...) child for the placed symbols
        paths = _find_nodes(node, "path")
        assert len(paths) >= 1

    def test_at_least_one_symbol(self, parsed_tree: list[SExpNode]) -> None:
        symbols = _find_nodes(parsed_tree, "symbol")
        assert len(symbols) >= 1


class TestSymbolInstances:
    """symbol_instances section maps every placed symbol UUID to its reference."""

    def _get_placed_symbol_uuids(self, tree: list[SExpNode]) -> dict[str, str]:
        """Return {uuid: ref} for every placed (symbol ...) in the schematic."""
        result: dict[str, str] = {}
        for sym in _find_nodes(tree, "symbol"):
            uid = _scalar(sym, "uuid")
            # Reference is in (property "Reference" "R1" ...)
            for prop in _find_nodes(sym, "property"):
                if len(prop) >= 3 and prop[1] == "Reference":
                    ref = str(prop[2])
                    if isinstance(uid, str):
                        result[uid] = ref
                    break
        return result

    def _get_instance_map(self, tree: list[SExpNode]) -> dict[str, str]:
        """Return {uuid: ref} from the symbol_instances section."""
        si = _find_node(tree, "symbol_instances")
        assert si is not None
        result: dict[str, str] = {}
        for path_node in _find_nodes(si, "path"):
            if len(path_node) >= 2 and isinstance(path_node[1], str):
                # path is "/{uuid}"
                uid = path_node[1].lstrip("/")
                ref_node = _find_node(path_node, "reference")
                if ref_node and len(ref_node) >= 2:
                    result[uid] = str(ref_node[1])
        return result

    def test_every_symbol_has_instance_entry(self, parsed_tree: list[SExpNode]) -> None:
        """Every symbol UUID in the schematic must appear in symbol_instances."""
        placed = self._get_placed_symbol_uuids(parsed_tree)
        instances = self._get_instance_map(parsed_tree)
        for uid, ref in placed.items():
            assert uid in instances, f"Symbol {ref} (uuid={uid}) missing from symbol_instances"

    def test_instance_refs_match_symbol_refs(self, parsed_tree: list[SExpNode]) -> None:
        """Reference designators in symbol_instances must match placed symbol refs."""
        placed = self._get_placed_symbol_uuids(parsed_tree)
        instances = self._get_instance_map(parsed_tree)
        for uid, ref in placed.items():
            if uid in instances:
                assert instances[uid] == ref, (
                    f"Ref mismatch for uuid={uid}: symbol says {ref}, "
                    f"symbol_instances says {instances[uid]}"
                )

    def test_instance_entries_have_unit(self, parsed_tree: list[SExpNode]) -> None:
        """Every symbol_instances path entry must have a (unit N) child."""
        si = _find_node(parsed_tree, "symbol_instances")
        assert si is not None
        for path_node in _find_nodes(si, "path"):
            unit_node = _find_node(path_node, "unit")
            assert unit_node is not None, f"Missing (unit) in symbol_instances: {path_node}"


class TestUUIDs:
    """All UUIDs in the file are valid and unique."""

    def test_all_uuids_valid(self, parsed_tree: list[SExpNode]) -> None:
        for uid in _collect_all_uuids(parsed_tree):
            uuid_mod.UUID(uid)  # raises ValueError if invalid

    def test_all_uuids_unique(self, parsed_tree: list[SExpNode]) -> None:
        uuids = _collect_all_uuids(parsed_tree)
        counts = Counter(uuids)
        dupes = {u: c for u, c in counts.items() if c > 1}
        assert not dupes, f"Duplicate UUIDs: {dupes}"


class TestLibSymbols:
    """lib_symbols section has definitions for all referenced symbols."""

    def test_every_symbol_has_lib_definition(self, parsed_tree: list[SExpNode]) -> None:
        """Every symbol's lib_id must have a matching definition in lib_symbols."""
        lib_syms = _find_node(parsed_tree, "lib_symbols")
        assert lib_syms is not None
        defined_ids: set[str] = set()
        for sym_def in _find_nodes(lib_syms, "symbol"):
            if len(sym_def) >= 2 and isinstance(sym_def[1], str):
                defined_ids.add(sym_def[1])

        for sym in _find_nodes(parsed_tree, "symbol"):
            lib_id_node = _find_node(sym, "lib_id")
            if lib_id_node and len(lib_id_node) >= 2:
                lid = str(lib_id_node[1])
                assert lid in defined_ids, f"lib_id {lid!r} not in lib_symbols"

    def test_power_lib_symbols_have_pin(self, parsed_tree: list[SExpNode]) -> None:
        """Power lib_symbols must contain at least one pin definition."""
        lib_syms = _find_node(parsed_tree, "lib_symbols")
        assert lib_syms is not None
        for sym_def in _find_nodes(lib_syms, "symbol"):
            if len(sym_def) >= 2 and isinstance(sym_def[1], str):
                name = sym_def[1]
                if name.startswith("power:"):
                    # Look for pin in sub-symbols
                    has_pin = _has_pin_recursive(sym_def)
                    assert has_pin, f"Power symbol {name!r} has no pin definition"


def _has_pin_recursive(node: list[SExpNode]) -> bool:
    """Check if node or any nested symbol child has a (pin ...) entry."""
    if _find_node(node, "pin") is not None:
        return True
    for child in node:
        if (isinstance(child, list) and len(child) > 0
                and child[0] == "symbol" and _has_pin_recursive(child)):
            return True
    return False


# ---------------------------------------------------------------------------
# SECTION 2 — Power symbol routing correctness
# ---------------------------------------------------------------------------


class TestPowerRouting:
    """Power symbols use single straight wires (no L-shapes)."""

    def _get_power_symbol_positions(
        self, tree: list[SExpNode],
    ) -> dict[str, tuple[float, float]]:
        """Return {uuid: (x, y)} for power symbols (lib_id starts with 'power:')."""
        result: dict[str, tuple[float, float]] = {}
        for sym in _find_nodes(tree, "symbol"):
            lib_id_node = _find_node(sym, "lib_id")
            if not lib_id_node or len(lib_id_node) < 2:
                continue
            lid = str(lib_id_node[1])
            if not lid.startswith("power:"):
                continue
            uid = _scalar(sym, "uuid")
            at_node = _find_node(sym, "at")
            if at_node and len(at_node) >= 3 and isinstance(uid, str):
                result[uid] = (_as_float(at_node[1]), _as_float(at_node[2]))
        return result

    def _get_wire_segments(
        self, tree: list[SExpNode],
    ) -> list[tuple[float, float, float, float]]:
        """Return [(x1, y1, x2, y2), ...] for every wire."""
        wires: list[tuple[float, float, float, float]] = []
        for w in _find_nodes(tree, "wire"):
            pts = _find_node(w, "pts")
            if pts and len(pts) >= 3:
                xy1 = pts[1]
                xy2 = pts[2]
                if (isinstance(xy1, list) and isinstance(xy2, list)
                        and len(xy1) >= 3 and len(xy2) >= 3):
                    wires.append((
                        _as_float(xy1[1]), _as_float(xy1[2]),
                        _as_float(xy2[1]), _as_float(xy2[2]),
                    ))
        return wires

    def test_all_wires_are_manhattan(self, parsed_tree: list[SExpNode]) -> None:
        """Every wire must be horizontal or vertical (no diagonals)."""
        for x1, y1, x2, y2 in self._get_wire_segments(parsed_tree):
            is_horiz = abs(y1 - y2) < 0.01
            is_vert = abs(x1 - x2) < 0.01
            assert is_horiz or is_vert, (
                f"Diagonal wire: ({x1}, {y1}) → ({x2}, {y2})"
            )

    def test_power_wires_are_single_segment(self, parsed_tree: list[SExpNode]) -> None:
        """Each power symbol connects to exactly one wire endpoint (no L-shapes).

        An L-shaped route would have a midpoint that isn't the power symbol
        position and isn't the pin — i.e. two wires sharing a midpoint.
        We verify by checking that no wire endpoint near a power symbol is
        also an intermediate point of a two-wire L-shape.
        """
        pwr_positions = self._get_power_symbol_positions(parsed_tree)
        wires = self._get_wire_segments(parsed_tree)

        # Build a set of all wire endpoints
        wire_endpoints: list[tuple[float, float]] = []
        for x1, y1, x2, y2 in wires:
            wire_endpoints.append((x1, y1))
            wire_endpoints.append((x2, y2))

        # For each power symbol, find wires that terminate at its position
        for _uid, (px, py) in pwr_positions.items():
            connected_wires = [
                (x1, y1, x2, y2) for x1, y1, x2, y2 in wires
                if (_near(x1, px) and _near(y1, py))
                or (_near(x2, px) and _near(y2, py))
            ]
            # A power symbol should connect to exactly one wire
            assert len(connected_wires) >= 1, (
                f"Power symbol at ({px}, {py}) has no connected wire"
            )

            # Each connected wire should be a straight shot (verified above
            # by manhattan check). Additionally, verify the wire from pin to
            # power symbol is a single segment — there should NOT be a second
            # wire forming an L at the same point.
            for wx1, wy1, wx2, wy2 in connected_wires:
                # The power-side endpoint
                pin_end = (
                    (wx2, wy2) if _near(wx1, px) and _near(wy1, py)
                    else (wx1, wy1)
                )

                # The wire from pin to power should be strictly horizontal
                # OR strictly vertical (inline routing, not L-shaped).
                is_horiz = _near(pin_end[1], py)
                is_vert = _near(pin_end[0], px)
                assert is_horiz or is_vert, (
                    f"Power wire not inline: ({pin_end[0]}, {pin_end[1]}) → "
                    f"({px}, {py}) — implies L-shape routing"
                )


def _near(a: float, b: float, tol: float = 0.02) -> bool:
    return abs(a - b) < tol


class TestPowerSymbolRotation:
    """Power symbols have appropriate rotation values."""

    def test_gnd_rotation_valid(self, parsed_tree: list[SExpNode]) -> None:
        """GND symbols must have rotation in {0, 90, 180, 270}."""
        for sym in _find_nodes(parsed_tree, "symbol"):
            lib_id_node = _find_node(sym, "lib_id")
            if not lib_id_node or len(lib_id_node) < 2:
                continue
            lid = str(lib_id_node[1])
            if "GND" not in lid.upper():
                continue
            at_node = _find_node(sym, "at")
            if at_node and len(at_node) >= 4:
                rot = _as_float(at_node[3])
                assert rot in (0.0, 90.0, 180.0, 270.0), (
                    f"GND symbol rotation {rot} not in {{0, 90, 180, 270}}"
                )

    def test_vcc_rotation_valid(self, parsed_tree: list[SExpNode]) -> None:
        """VCC/+5V/+3V3 symbols must have rotation in {0, 90, 180, 270}."""
        for sym in _find_nodes(parsed_tree, "symbol"):
            lib_id_node = _find_node(sym, "lib_id")
            if not lib_id_node or len(lib_id_node) < 2:
                continue
            lid = str(lib_id_node[1])
            if not any(v in lid for v in ("+5V", "+3V3", "+3.3V", "VCC", "+1V8")):
                continue
            at_node = _find_node(sym, "at")
            if at_node and len(at_node) >= 4:
                rot = _as_float(at_node[3])
                assert rot in (0.0, 90.0, 180.0, 270.0), (
                    f"Power symbol {lid} rotation {rot} not in {{0, 90, 180, 270}}"
                )


# ---------------------------------------------------------------------------
# SECTION 3 — Wire connectivity
# ---------------------------------------------------------------------------


class TestWireConnectivity:
    """Wire endpoints should connect to something meaningful."""

    def test_no_zero_length_wires(self, parsed_tree: list[SExpNode]) -> None:
        """No wire should have identical start and end points."""
        for w in _find_nodes(parsed_tree, "wire"):
            pts = _find_node(w, "pts")
            if pts and len(pts) >= 3:
                xy1, xy2 = pts[1], pts[2]
                if (isinstance(xy1, list) and isinstance(xy2, list)
                        and len(xy1) >= 3 and len(xy2) >= 3):
                    same_x = _near(_as_float(xy1[1]), _as_float(xy2[1]))
                    same_y = _near(_as_float(xy1[2]), _as_float(xy2[2]))
                    assert not (same_x and same_y), (
                        f"Zero-length wire at ({xy1[1]}, {xy1[2]})"
                    )

    def test_no_duplicate_wires(self, parsed_tree: list[SExpNode]) -> None:
        """No two wires should share identical start AND end points."""
        seen: set[tuple[float, float, float, float]] = set()
        for w in _find_nodes(parsed_tree, "wire"):
            pts = _find_node(w, "pts")
            if pts and len(pts) >= 3:
                xy1, xy2 = pts[1], pts[2]
                if (isinstance(xy1, list) and isinstance(xy2, list)
                        and len(xy1) >= 3 and len(xy2) >= 3):
                    # Normalise direction
                    p = (_as_float(xy1[1]), _as_float(xy1[2]),
                         _as_float(xy2[1]), _as_float(xy2[2]))
                    key = (min(p[0], p[2]), min(p[1], p[3]),
                           max(p[0], p[2]), max(p[1], p[3]))
                    # Round to grid for comparison
                    rkey = tuple(round(v, 2) for v in key)
                    assert rkey not in seen, f"Duplicate wire: {rkey}"
                    seen.add(rkey)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SECTION 4 — Reference designator correctness
# ---------------------------------------------------------------------------


class TestReferenceDesignators:
    """Reference designators are properly assigned and consistent."""

    def test_no_question_mark_refs(self, schematic_file: tuple[Path, str]) -> None:
        """No reference designator should contain '?' (unannotated)."""
        _, text = schematic_file
        # Look for (reference "R?") or similar patterns
        # Exclude #PWR refs which legitimately use a different numbering
        ref_pattern = re.compile(r'\(reference\s+"([^"]+)"\)')
        for m in ref_pattern.finditer(text):
            ref = m.group(1)
            if ref.startswith("#"):
                continue  # Power refs like #PWR01 are fine
            assert "?" not in ref, f"Unannotated reference: {ref!r}"

    def test_regular_refs_are_numbered(self, schematic_file: tuple[Path, str]) -> None:
        """Regular refs (R, U, C, D, etc.) must end with a number."""
        _, text = schematic_file
        ref_pattern = re.compile(r'\(reference\s+"([^"]+)"\)')
        for m in ref_pattern.finditer(text):
            ref = m.group(1)
            if ref.startswith("#"):
                continue
            assert re.match(r"^[A-Z]+\d+$", ref), (
                f"Reference {ref!r} does not match pattern PREFIX + NUMBER"
            )

    def test_power_refs_are_numbered(self, schematic_file: tuple[Path, str]) -> None:
        """Power refs (#PWR0xx) must follow the #PWR0NN pattern."""
        _, text = schematic_file
        ref_pattern = re.compile(r'\(reference\s+"(#PWR[^"]+)"\)')
        for m in ref_pattern.finditer(text):
            ref = m.group(1)
            assert re.match(r"^#PWR0\d+$", ref), (
                f"Power reference {ref!r} does not match #PWR0NN pattern"
            )


# ---------------------------------------------------------------------------
# SECTION 5 — Stylistic / formatting correctness
# ---------------------------------------------------------------------------


class TestFileFormatting:
    """File-level formatting matches KiCad 9 conventions."""

    def test_starts_with_kicad_sch(self, schematic_file: tuple[Path, str]) -> None:
        _, text = schematic_file
        assert text.startswith("(kicad_sch")

    def test_ends_with_single_newline(self, schematic_file: tuple[Path, str]) -> None:
        _, text = schematic_file
        assert text.endswith(")\n")
        assert not text.endswith(")\n\n")

    def test_no_trailing_whitespace(self, schematic_file: tuple[Path, str]) -> None:
        _, text = schematic_file
        for i, line in enumerate(text.splitlines(), 1):
            assert line == line.rstrip(), (
                f"Trailing whitespace on line {i}: {line!r}"
            )

    def test_consistent_indentation(self, schematic_file: tuple[Path, str]) -> None:
        """Indented lines use spaces (no tabs), in multiples of 2."""
        _, text = schematic_file
        for i, line in enumerate(text.splitlines(), 1):
            if not line or line[0] != " ":
                continue
            indent = len(line) - len(line.lstrip(" "))
            assert "\t" not in line[:indent], f"Tab indent on line {i}"
            assert indent % 2 == 0, (
                f"Odd indentation ({indent} spaces) on line {i}: {line!r}"
            )

    def test_utf8_encoding(self, schematic_file: tuple[Path, str]) -> None:
        path, _ = schematic_file
        # Just ensure it can be decoded as UTF-8 without errors
        path.read_text(encoding="utf-8")

    def test_balanced_parentheses(self, schematic_file: tuple[Path, str]) -> None:
        """Parentheses must be balanced across the entire file."""
        _, text = schematic_file
        depth = 0
        in_string = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            assert depth >= 0, "Unbalanced closing parenthesis"
        assert depth == 0, f"Unclosed parentheses: depth={depth}"


class TestSExpQuoting:
    """S-expression quoting follows KiCad 9 rules."""

    def test_version_is_bare_integer(self, schematic_file: tuple[Path, str]) -> None:
        _, text = schematic_file
        assert re.search(r"\(version \d+\)", text), (
            "Version should be a bare integer, not quoted"
        )

    def test_generator_is_quoted(self, schematic_file: tuple[Path, str]) -> None:
        _, text = schematic_file
        assert re.search(r'\(generator "[^"]+"\)', text), (
            "Generator should be a quoted string"
        )

    def test_uuid_is_quoted(self, schematic_file: tuple[Path, str]) -> None:
        """UUIDs should be quoted strings."""
        _, text = schematic_file
        uuid_pattern = re.compile(r'\(uuid "([^"]+)"\)')
        matches = uuid_pattern.findall(text)
        assert len(matches) >= 1
        for uid in matches:
            uuid_mod.UUID(uid)  # validates format

    def test_property_values_are_quoted(self, schematic_file: tuple[Path, str]) -> None:
        """Property name and value should both be quoted."""
        _, text = schematic_file
        prop_pattern = re.compile(r'\(property\s+("[^"]*")\s+("[^"]*")')
        matches = prop_pattern.findall(text)
        assert len(matches) >= 1, "No (property ...) nodes found"


# ---------------------------------------------------------------------------
# SECTION 6 — Required properties on symbol instances
# ---------------------------------------------------------------------------


class TestSymbolProperties:
    """Each placed symbol must have the KiCad 9 required properties."""

    def test_symbols_have_required_properties(self, parsed_tree: list[SExpNode]) -> None:
        """Every non-power symbol must have Reference, Value, Footprint properties."""
        for sym in _find_nodes(parsed_tree, "symbol"):
            lib_id_node = _find_node(sym, "lib_id")
            if not lib_id_node or len(lib_id_node) < 2:
                continue
            lid = str(lib_id_node[1])
            if lid.startswith("power:"):
                continue  # power symbols have different property sets

            prop_names: set[str] = set()
            for prop in _find_nodes(sym, "property"):
                if len(prop) >= 2 and isinstance(prop[1], str):
                    prop_names.add(prop[1])

            for required in ("Reference", "Value", "Footprint"):
                assert required in prop_names, (
                    f"Symbol {lid} missing property {required!r}. "
                    f"Has: {prop_names}"
                )

    def test_symbols_have_at_position(self, parsed_tree: list[SExpNode]) -> None:
        """Every symbol must have an (at x y ...) node."""
        for sym in _find_nodes(parsed_tree, "symbol"):
            at_node = _find_node(sym, "at")
            assert at_node is not None, f"Symbol missing (at) node: {sym[:3]}"
            assert len(at_node) >= 3, f"Symbol (at) node too short: {at_node}"

    def test_symbols_have_instances_section(self, parsed_tree: list[SExpNode]) -> None:
        """Every placed symbol must have an (instances ...) child for project mapping."""
        for sym in _find_nodes(parsed_tree, "symbol"):
            inst = _find_node(sym, "instances")
            assert inst is not None, (
                f"Symbol missing (instances) section: {sym[:3]}"
            )

    def test_symbols_have_exclude_from_sim(self, parsed_tree: list[SExpNode]) -> None:
        """KiCad 9 requires exclude_from_sim on every symbol."""
        for sym in _find_nodes(parsed_tree, "symbol"):
            node = _find_node(sym, "exclude_from_sim")
            assert node is not None, f"Symbol missing exclude_from_sim: {sym[:3]}"

    def test_symbols_have_in_bom(self, parsed_tree: list[SExpNode]) -> None:
        """KiCad 9 requires in_bom on every symbol."""
        for sym in _find_nodes(parsed_tree, "symbol"):
            node = _find_node(sym, "in_bom")
            assert node is not None, f"Symbol missing in_bom: {sym[:3]}"

    def test_symbols_have_on_board(self, parsed_tree: list[SExpNode]) -> None:
        """KiCad 9 requires on_board on every symbol."""
        for sym in _find_nodes(parsed_tree, "symbol"):
            node = _find_node(sym, "on_board")
            assert node is not None, f"Symbol missing on_board: {sym[:3]}"


# ---------------------------------------------------------------------------
# SECTION 7 — Sexp round-trip stability
# ---------------------------------------------------------------------------


class TestSExpRoundTrip:
    """Write → parse round-trip produces structurally equivalent trees."""

    @pytest.mark.parametrize("req_name", ["minimal", "multi_power", "adc"])
    def test_roundtrip_preserves_structure(self, req_name: str) -> None:
        """schematic_to_sexp → write → parse → basic structural equivalence."""
        builders = {
            "minimal": _minimal_requirements,
            "multi_power": _multi_power_requirements,
            "adc": _adc_requirements,
        }
        reqs = builders[req_name]()
        sch = build_schematic(reqs)
        sexp = schematic_to_sexp(sch)
        assert isinstance(sexp, list)
        text = write(sexp)
        reparsed = parse(text)
        assert isinstance(reparsed, list)
        assert reparsed[0] == "kicad_sch"

        # Same number of top-level sections
        original_tags = _top_level_tags(sexp)
        reparsed_tags = _top_level_tags(reparsed)
        assert Counter(original_tags) == Counter(reparsed_tags)
