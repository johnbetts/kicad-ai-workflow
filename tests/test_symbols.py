"""Tests for kicad_pipeline.schematic.symbols."""

from __future__ import annotations

from kicad_pipeline.models.requirements import Component, Pin, PinType
from kicad_pipeline.models.schematic import LibRectangle, LibSymbol
from kicad_pipeline.schematic.symbols import (
    BUILTIN_SYMBOLS,
    get_or_make_symbol,
    make_lib_symbol,
    make_passive_symbol,
    make_power_symbol,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_component(
    ref: str = "U1",
    value: str = "TestIC",
    pins: tuple[Pin, ...] = (),
    description: str | None = None,
) -> Component:
    """Build a minimal :class:`Component` for testing."""
    return Component(
        ref=ref,
        value=value,
        footprint="Package_SO:SOIC-8",
        pins=pins,
        description=description,
    )


def _input_pin(number: str, name: str) -> Pin:
    return Pin(number=number, name=name, pin_type=PinType.INPUT)


def _output_pin(number: str, name: str) -> Pin:
    return Pin(number=number, name=name, pin_type=PinType.OUTPUT)


# ---------------------------------------------------------------------------
# make_passive_symbol
# ---------------------------------------------------------------------------


def test_make_passive_symbol_has_two_pins() -> None:
    """make_passive_symbol returns a symbol with exactly two pins."""
    sym = make_passive_symbol("Device:R")
    assert len(sym.pins) == 2


def test_make_passive_symbol_lib_id() -> None:
    """make_passive_symbol preserves the given lib_id."""
    sym = make_passive_symbol("Device:C")
    assert sym.lib_id == "Device:C"


# ---------------------------------------------------------------------------
# make_lib_symbol
# ---------------------------------------------------------------------------


def test_make_lib_symbol_basic_component() -> None:
    """Component with 2 input pins and 1 output → symbol with 3 pins."""
    comp = _make_component(
        pins=(
            _input_pin("1", "IN1"),
            _input_pin("2", "IN2"),
            _output_pin("3", "OUT"),
        )
    )
    sym = make_lib_symbol(comp)
    assert len(sym.pins) == 3


def test_make_lib_symbol_pin_positions() -> None:
    """Input pins are on the left (negative x); output pins on the right."""
    comp = _make_component(
        pins=(
            _input_pin("1", "IN"),
            _output_pin("2", "OUT"),
        )
    )
    sym = make_lib_symbol(comp)

    input_pins = [p for p in sym.pins if p.pin_type == PinType.INPUT.value]
    output_pins = [p for p in sym.pins if p.pin_type == PinType.OUTPUT.value]

    assert all(p.at.x < 0 for p in input_pins), "Input pins should have negative x"
    assert all(p.at.x > 0 for p in output_pins), "Output pins should have positive x"


# ---------------------------------------------------------------------------
# make_power_symbol
# ---------------------------------------------------------------------------


def test_make_power_symbol_gnd() -> None:
    """GND power symbol has exactly one pin."""
    sym = make_power_symbol("GND")
    assert len(sym.pins) == 1
    assert sym.lib_id == "power:GND"


def test_make_power_symbol_vcc() -> None:
    """VCC power symbol has exactly one pin."""
    sym = make_power_symbol("VCC")
    assert len(sym.pins) == 1
    assert sym.lib_id == "power:VCC"


# ---------------------------------------------------------------------------
# BUILTIN_SYMBOLS
# ---------------------------------------------------------------------------


def test_builtin_symbols_populated() -> None:
    """BUILTIN_SYMBOLS contains Device:R, Device:C, and Device:LED."""
    assert "Device:R" in BUILTIN_SYMBOLS
    assert "Device:C" in BUILTIN_SYMBOLS
    assert "Device:LED" in BUILTIN_SYMBOLS


# ---------------------------------------------------------------------------
# get_or_make_symbol
# ---------------------------------------------------------------------------


def test_get_or_make_symbol_uses_builtin_for_resistor() -> None:
    """get_or_make_symbol returns Device:R for a resistor component."""
    comp = _make_component(ref="R1", value="10k", description="resistor 10k ohm")
    cache: dict[str, LibSymbol] = {}
    sym = get_or_make_symbol(comp, cache)
    assert sym.lib_id == "Device:R"


def test_get_or_make_symbol_generates_for_unknown() -> None:
    """get_or_make_symbol auto-generates a symbol for an unknown component and caches it."""
    comp = _make_component(
        ref="U99",
        value="MyCustomIC",
        pins=(_input_pin("1", "A"), _output_pin("2", "B")),
    )
    cache: dict[str, LibSymbol] = {}
    sym = get_or_make_symbol(comp, cache)
    # Should be auto-generated, not a built-in
    assert sym.lib_id.startswith("kicad-ai:")
    # Should be added to cache
    assert "U99" in cache
    assert cache["U99"] is sym


# ---------------------------------------------------------------------------
# Symbol shape and structure
# ---------------------------------------------------------------------------


def test_lib_symbol_has_rectangle_shape() -> None:
    """A generated IC symbol has at least one LibRectangle shape."""
    comp = _make_component(
        pins=(_input_pin("1", "IN"), _output_pin("2", "OUT")),
    )
    sym = make_lib_symbol(comp)
    has_rect = any(isinstance(s, LibRectangle) for s in sym.shapes)
    assert has_rect, "Generated IC symbol should contain a LibRectangle"


def test_lib_symbol_pins_are_frozen() -> None:
    """LibSymbol.pins is a tuple (frozen/immutable)."""
    comp = _make_component(pins=(_input_pin("1", "A"),))
    sym = make_lib_symbol(comp)
    assert isinstance(sym.pins, tuple)
