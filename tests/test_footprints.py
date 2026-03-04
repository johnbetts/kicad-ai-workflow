"""Tests for kicad_pipeline.pcb.footprints."""

from __future__ import annotations

from pathlib import Path

import pytest

from kicad_pipeline.exceptions import ConfigurationError, PCBError
from kicad_pipeline.models.pcb import Footprint, FootprintLine
from kicad_pipeline.pcb.footprints import (
    apply_rotation_offset,
    footprint_for_component,
    load_rotation_offsets,
    make_rj45,
    make_smd_led,
    make_smd_resistor_capacitor,
    make_sot23,
    make_through_hole_2pin,
    make_usbc_connector,
)

# Actual data directory (ROTATION_OFFSETS_FILE constant has a path bug; use real path)
_DATA_DIR = Path(__file__).parent.parent / "data"
_ROTATION_OFFSETS_FILE = _DATA_DIR / "rotation_offsets.json"


# ---------------------------------------------------------------------------
# SMD resistor / capacitor — 0805
# ---------------------------------------------------------------------------


def test_smd_rc_0805() -> None:
    """make_smd_resistor_capacitor 0805 produces exactly 2 pads."""
    fp = make_smd_resistor_capacitor("R1", "10k", "0805")
    assert len(fp.pads) == 2


def test_smd_rc_0805_pad_numbers() -> None:
    """0805 pads are numbered '1' and '2'."""
    fp = make_smd_resistor_capacitor("R1", "10k", "0805")
    numbers = {p.number for p in fp.pads}
    assert numbers == {"1", "2"}


def test_smd_rc_0805_pad_x_positions() -> None:
    """0805 pads are at ±1.0 mm on the x axis (pitch=2.0, pitch/2=1.0)."""
    fp = make_smd_resistor_capacitor("R1", "10k", "0805")
    pad1 = next(p for p in fp.pads if p.number == "1")
    pad2 = next(p for p in fp.pads if p.number == "2")
    assert pad1.position.x == pytest.approx(-1.0)
    assert pad2.position.x == pytest.approx(1.0)


def test_smd_rc_0805_pad_layer_fcu() -> None:
    """0805 pads have F.Cu in their layers tuple."""
    fp = make_smd_resistor_capacitor("R1", "10k", "0805")
    for pad in fp.pads:
        assert "F.Cu" in pad.layers


# ---------------------------------------------------------------------------
# SMD resistor / capacitor — other packages
# ---------------------------------------------------------------------------


def test_smd_rc_0402() -> None:
    """Package 0402 produces 2 pads and smaller dimensions than 0805."""
    fp0402 = make_smd_resistor_capacitor("R2", "1k", "0402")
    fp0805 = make_smd_resistor_capacitor("R3", "1k", "0805")
    assert len(fp0402.pads) == 2
    # 0402 pads should be smaller
    assert fp0402.pads[0].size_x < fp0805.pads[0].size_x


def test_smd_rc_0603() -> None:
    """Package 0603 produces 2 pads."""
    fp = make_smd_resistor_capacitor("C2", "10nF", "0603")
    assert len(fp.pads) == 2


def test_smd_rc_1206() -> None:
    """Package 1206 produces 2 pads."""
    fp = make_smd_resistor_capacitor("R4", "4k7", "1206")
    assert len(fp.pads) == 2


def test_smd_rc_unknown_package() -> None:
    """Unknown package string raises PCBError."""
    with pytest.raises(PCBError):
        make_smd_resistor_capacitor("R1", "10k", "INVALID_PKG")


def test_smd_rc_has_courtyard() -> None:
    """Graphics include lines on F.CrtYd."""
    fp = make_smd_resistor_capacitor("R1", "10k", "0805")
    courtyard_lines = [
        g for g in fp.graphics
        if isinstance(g, FootprintLine) and g.layer == "F.CrtYd"
    ]
    assert len(courtyard_lines) >= 4


def test_smd_rc_has_silkscreen() -> None:
    """Graphics include lines on F.Silkscreen."""
    fp = make_smd_resistor_capacitor("R1", "10k", "0805")
    silk_lines = [
        g for g in fp.graphics
        if isinstance(g, FootprintLine) and g.layer == "F.SilkS"
    ]
    assert len(silk_lines) >= 2


def test_smd_rc_has_texts() -> None:
    """Texts tuple has a reference and a value entry."""
    fp = make_smd_resistor_capacitor("R1", "10k", "0805")
    text_types = {t.text_type for t in fp.texts}
    assert "reference" in text_types
    assert "value" in text_types


def test_footprint_ref_assigned() -> None:
    """Footprint.ref matches the input ref argument."""
    fp = make_smd_resistor_capacitor("C4", "100nF", "0603")
    assert fp.ref == "C4"


def test_footprint_is_frozen() -> None:
    """Footprint is a frozen dataclass — attribute assignment raises."""
    fp = make_smd_resistor_capacitor("R1", "10k")
    with pytest.raises(AttributeError):
        fp.ref = "R99"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# make_smd_led
# ---------------------------------------------------------------------------


def test_smd_led_0805() -> None:
    """LED 0805 footprint has more silkscreen lines than basic RC (polarity triangle)."""
    rc = make_smd_resistor_capacitor("R1", "10k", "0805")
    led = make_smd_led("D1", "RED", "0805")
    rc_silk = [
        g for g in rc.graphics
        if isinstance(g, FootprintLine) and g.layer == "F.SilkS"
    ]
    led_silk = [
        g for g in led.graphics
        if isinstance(g, FootprintLine) and g.layer == "F.SilkS"
    ]
    assert len(led_silk) > len(rc_silk)


def test_smd_led_unknown_package() -> None:
    """Unknown package for LED raises PCBError."""
    with pytest.raises(PCBError):
        make_smd_led("D1", "RED", "UNKNOWNPKG")


# ---------------------------------------------------------------------------
# make_sot23
# ---------------------------------------------------------------------------


def test_sot23_3pin() -> None:
    """SOT-23 variant produces 3 pads."""
    fp = make_sot23("Q1", "S8050", "SOT-23")
    assert len(fp.pads) == 3


def test_sot23_5pin() -> None:
    """SOT-23-5 variant produces 5 pads."""
    fp = make_sot23("U2", "MCP1700", "SOT-23-5")
    assert len(fp.pads) == 5


def test_sot23_6pin() -> None:
    """SOT-23-6 variant produces 6 pads."""
    fp = make_sot23("U3", "FDC6301", "SOT-23-6")
    assert len(fp.pads) == 6


def test_sot23_unknown_variant() -> None:
    """Unknown SOT-23 variant raises PCBError."""
    with pytest.raises(PCBError):
        make_sot23("Q1", "X", "SOT-23-99")


def test_sot23_pads_are_smd() -> None:
    """All SOT-23 pads are SMD type."""
    fp = make_sot23("Q1", "BSS138", "SOT-23")
    for pad in fp.pads:
        assert pad.pad_type == "smd"


# ---------------------------------------------------------------------------
# make_through_hole_2pin
# ---------------------------------------------------------------------------


def test_thru_hole_2pin() -> None:
    """Through-hole 2-pin footprint has exactly 2 pads."""
    fp = make_through_hole_2pin("SW1", "Switch")
    assert len(fp.pads) == 2


def test_thru_hole_2pin_pad_type() -> None:
    """Through-hole pads have pad_type 'thru_hole'."""
    fp = make_through_hole_2pin("SW1", "Switch")
    for pad in fp.pads:
        assert pad.pad_type == "thru_hole"


def test_thru_hole_2pin_drill_not_none() -> None:
    """Through-hole pads have a non-None drill_diameter."""
    fp = make_through_hole_2pin("SW1", "Switch", pitch_mm=2.54, drill_mm=0.8)
    for pad in fp.pads:
        assert pad.drill_diameter is not None
        assert pad.drill_diameter == pytest.approx(0.8)


def test_through_hole_attr() -> None:
    """Through-hole footprint has attr='through_hole'."""
    fp = make_through_hole_2pin("J1", "Conn")
    assert fp.attr == "through_hole"


# ---------------------------------------------------------------------------
# make_usbc_connector
# ---------------------------------------------------------------------------


def test_usbc_connector() -> None:
    """USB-C connector produces exactly 8 pads."""
    fp = make_usbc_connector("J1")
    assert len(fp.pads) == 8


def test_usbc_connector_attr_smd() -> None:
    """USB-C connector has attr='smd'."""
    fp = make_usbc_connector("J1")
    assert fp.attr == "smd"


# ---------------------------------------------------------------------------
# make_rj45
# ---------------------------------------------------------------------------


def test_rj45() -> None:
    """RJ45 produces 10 signal/LED pads + 4 mounting holes = 14 total pads."""
    fp = make_rj45("J2")
    assert len(fp.pads) == 14


def test_rj45_mounting_pads_np_thru() -> None:
    """RJ45 mounting holes are np_thru_hole type."""
    fp = make_rj45("J2")
    mp_pads = [p for p in fp.pads if p.number.startswith("MP")]
    assert len(mp_pads) == 4
    for pad in mp_pads:
        assert pad.pad_type == "np_thru_hole"


# ---------------------------------------------------------------------------
# footprint_for_component routing
# ---------------------------------------------------------------------------


def test_footprint_for_component_resistor() -> None:
    """'R_0805' footprint_id → lib_id contains 'Device'."""
    fp = footprint_for_component("R1", "10k", "R_0805")
    assert "Device" in fp.lib_id


def test_footprint_for_component_led() -> None:
    """'LED_0805' footprint_id → LED lib_id."""
    fp = footprint_for_component("D1", "RED", "LED_0805")
    assert "LED" in fp.lib_id


def test_footprint_for_component_sot23() -> None:
    """'SOT-23' footprint_id → 3 pads."""
    fp = footprint_for_component("Q1", "S8050", "SOT-23")
    assert len(fp.pads) == 3


def test_footprint_for_component_usbc() -> None:
    """'USB-C-SMD' footprint_id → USB-C connector with 8 pads."""
    fp = footprint_for_component("J1", "USB-C", "USB-C-SMD")
    assert len(fp.pads) == 8


def test_footprint_for_component_with_lcsc() -> None:
    """Providing lcsc='C17414' stamps it on the returned footprint."""
    fp = footprint_for_component("R1", "10k", "R_0805", lcsc="C17414")
    assert fp.lcsc == "C17414"


def test_footprint_for_component_fallback() -> None:
    """Unknown footprint_id falls back to 0805 (2 pads)."""
    fp = footprint_for_component("X1", "Thing", "COMPLETELY_UNKNOWN")
    assert len(fp.pads) == 2


def test_footprint_for_component_rj45() -> None:
    """'RJ45' footprint_id → RJ45 with 14 total pads."""
    fp = footprint_for_component("J2", "RJ45", "RJ45")
    assert len(fp.pads) == 14


def test_footprint_for_component_no_lcsc() -> None:
    """footprint_for_component with no LCSC leaves fp.lcsc as None."""
    fp = footprint_for_component("R1", "10k", "R_0805")
    assert fp.lcsc is None


def test_footprint_for_component_returns_footprint_instance() -> None:
    """footprint_for_component always returns a Footprint instance."""
    fp = footprint_for_component("R1", "10k", "R_0805")
    assert isinstance(fp, Footprint)


# ---------------------------------------------------------------------------
# load_rotation_offsets / apply_rotation_offset
# ---------------------------------------------------------------------------


def test_load_rotation_offsets() -> None:
    """load_rotation_offsets loads the bundled file without error."""
    offsets = load_rotation_offsets(_ROTATION_OFFSETS_FILE)
    assert isinstance(offsets, dict)
    assert len(offsets) > 0


def test_load_rotation_offsets_sot23_is_180() -> None:
    """SOT-23 offset is 180.0 degrees in the bundled data file."""
    offsets = load_rotation_offsets(_ROTATION_OFFSETS_FILE)
    assert "SOT-23" in offsets
    assert offsets["SOT-23"] == pytest.approx(180.0)


def test_apply_rotation_offset_sot23() -> None:
    """SOT-23 at 0.0 + 180 = 180.0 degrees."""
    offsets = {"SOT-23": 180.0}
    result = apply_rotation_offset("SOT-23", 0.0, offsets)
    assert result == pytest.approx(180.0)


def test_apply_rotation_offset_0805() -> None:
    """0805 not in offsets → angle returned unchanged."""
    offsets = {"SOT-23": 180.0}
    result = apply_rotation_offset("0805", 45.0, offsets)
    assert result == pytest.approx(45.0)


def test_apply_rotation_offset_unknown() -> None:
    """Completely unknown package → rotation unchanged."""
    offsets = {"SOT-23": 180.0}
    result = apply_rotation_offset("MYSTERY_PKG", 90.0, offsets)
    assert result == pytest.approx(90.0)


def test_apply_rotation_offset_sot23_wraps() -> None:
    """SOT-23 offset applied to 270° wraps correctly to 90°."""
    offsets = load_rotation_offsets(_ROTATION_OFFSETS_FILE)
    result = apply_rotation_offset("SOT-23", 270.0, offsets)
    assert result == pytest.approx(90.0)


def test_load_rotation_offsets_missing_file_raises(tmp_path: Path) -> None:
    """load_rotation_offsets raises ConfigurationError for a missing file."""
    with pytest.raises(ConfigurationError):
        load_rotation_offsets(tmp_path / "nonexistent.json")


def test_load_rotation_offsets_invalid_json_raises(tmp_path: Path) -> None:
    """load_rotation_offsets raises ConfigurationError for invalid JSON."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("NOT JSON {{{", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_rotation_offsets(bad_file)
