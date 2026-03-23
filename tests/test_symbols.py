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


# ---------------------------------------------------------------------------
# make_lib_symbol — additional coverage
# ---------------------------------------------------------------------------


def _power_in_pin(number: str, name: str) -> Pin:
    return Pin(number=number, name=name, pin_type=PinType.POWER_IN)


def _power_out_pin(number: str, name: str) -> Pin:
    return Pin(number=number, name=name, pin_type=PinType.POWER_OUT)


def _bidirectional_pin(number: str, name: str) -> Pin:
    return Pin(number=number, name=name, pin_type=PinType.BIDIRECTIONAL)


def _no_connect_pin(number: str, name: str) -> Pin:
    return Pin(number=number, name=name, pin_type=PinType.NO_CONNECT)


def _passive_pin(number: str, name: str) -> Pin:
    return Pin(number=number, name=name, pin_type=PinType.PASSIVE)


def test_make_lib_symbol_power_pins_top_bottom() -> None:
    """VCC goes top, GND goes bottom in generated symbol."""
    comp = _make_component(
        pins=(
            _power_in_pin("1", "VCC"),
            _power_in_pin("2", "GND"),
            _input_pin("3", "IN"),
            _output_pin("4", "OUT"),
        )
    )
    sym = make_lib_symbol(comp)
    # VCC pin should have positive Y tip (top), GND negative Y tip (bottom)
    vcc_pin = next(p for p in sym.pins if p.name == "VCC")
    gnd_pin = next(p for p in sym.pins if p.name == "GND")
    assert vcc_pin.at.y > 0 or vcc_pin.rotation == 270.0  # top placement
    assert gnd_pin.at.y < 0 or gnd_pin.rotation == 90.0  # bottom placement


def test_make_lib_symbol_no_connect_pins_excluded() -> None:
    """NO_CONNECT pins are excluded from the symbol."""
    comp = _make_component(
        pins=(
            _input_pin("1", "IN"),
            _output_pin("2", "OUT"),
            _no_connect_pin("3", "NC"),
            _no_connect_pin("4", "NC2"),
        )
    )
    sym = make_lib_symbol(comp)
    assert len(sym.pins) == 2  # only IN and OUT


def test_make_lib_symbol_no_pins() -> None:
    """Component with no pins produces a symbol with no pins."""
    comp = _make_component(pins=())
    sym = make_lib_symbol(comp)
    assert len(sym.pins) == 0


def test_make_lib_symbol_lib_id_format() -> None:
    """lib_id follows 'kicad-ai:{prefix}_{value}' format."""
    comp = _make_component(ref="U3", value="ESP32")
    sym = make_lib_symbol(comp)
    assert sym.lib_id == "kicad-ai:U_ESP32"


def test_make_lib_symbol_connector_ref_prefix() -> None:
    """Connector ref 'J1' yields lib_id starting with 'kicad-ai:J_'."""
    comp = _make_component(ref="J1", value="Conn_01x04", pins=(
        _passive_pin("1", "P1"), _passive_pin("2", "P2"),
    ))
    sym = make_lib_symbol(comp)
    assert sym.lib_id.startswith("kicad-ai:J_")


def test_make_lib_symbol_bidirectional_pins_right() -> None:
    """Bidirectional pins are placed on the right side for 4+ pin components."""
    comp = _make_component(
        pins=(
            _input_pin("1", "IN1"),
            _input_pin("2", "IN2"),
            _output_pin("3", "OUT"),
            _bidirectional_pin("4", "SDA"),
            _bidirectional_pin("5", "SCL"),
        )
    )
    sym = make_lib_symbol(comp)
    sda_pin = next(p for p in sym.pins if p.name == "SDA")
    # Bidirectional pins should be on the right (positive x)
    assert sda_pin.at.x > 0


# ---------------------------------------------------------------------------
# make_passive_symbol — additional
# ---------------------------------------------------------------------------


def test_make_passive_symbol_pin_types_are_passive() -> None:
    """Both pins of a passive symbol have 'passive' pin_type."""
    sym = make_passive_symbol("Device:R")
    assert all(p.pin_type == "passive" for p in sym.pins)


def test_make_passive_symbol_custom_pin_names() -> None:
    """Custom pin names are preserved."""
    sym = make_passive_symbol("Device:C", pin1_name="+", pin2_name="-")
    assert sym.pins[0].name == "+"
    assert sym.pins[1].name == "-"


def test_make_passive_symbol_has_body_shapes() -> None:
    """Passive symbol has 3 polyline shapes (body + end marks)."""
    sym = make_passive_symbol("Device:R")
    assert len(sym.shapes) == 3


# ---------------------------------------------------------------------------
# make_power_symbol — additional
# ---------------------------------------------------------------------------


def test_make_power_symbol_gnd_rotation_270() -> None:
    """GND symbol has pin rotation 270 (pointing down)."""
    sym = make_power_symbol("GND")
    assert sym.pins[0].rotation == 270.0


def test_make_power_symbol_vcc_rotation_90() -> None:
    """VCC symbol has pin rotation 90 (pointing up)."""
    sym = make_power_symbol("VCC")
    assert sym.pins[0].rotation == 90.0


def test_make_power_symbol_vss_is_ground_style() -> None:
    """VSS is treated as ground (rotation 270)."""
    sym = make_power_symbol("AVSS")
    assert sym.pins[0].rotation == 270.0


def test_make_power_symbol_3v3_is_positive_style() -> None:
    """+3.3V has positive supply style (rotation 90)."""
    sym = make_power_symbol("+3.3V")
    assert sym.pins[0].rotation == 90.0


def test_make_power_symbol_custom_name() -> None:
    """Custom net name produces matching lib_id."""
    sym = make_power_symbol("+1.8V")
    assert sym.lib_id == "power:+1.8V"


# ---------------------------------------------------------------------------
# make_led_symbol
# ---------------------------------------------------------------------------

from kicad_pipeline.schematic.symbols import make_led_symbol  # noqa: E402


def test_make_led_symbol_two_pins() -> None:
    """LED symbol has exactly 2 pins."""
    sym = make_led_symbol()
    assert len(sym.pins) == 2


def test_make_led_symbol_anode_cathode_names() -> None:
    """LED pins are named A (anode) and K (cathode)."""
    sym = make_led_symbol()
    names = {p.name for p in sym.pins}
    assert names == {"A", "K"}


def test_make_led_symbol_custom_lib_id() -> None:
    """Custom lib_id is preserved."""
    sym = make_led_symbol("Custom:LED_RGB")
    assert sym.lib_id == "Custom:LED_RGB"


def test_make_led_symbol_has_shapes() -> None:
    """LED symbol has body shapes (triangle + bar)."""
    sym = make_led_symbol()
    assert len(sym.shapes) >= 2


# ---------------------------------------------------------------------------
# get_or_make_symbol — additional
# ---------------------------------------------------------------------------


def test_get_or_make_symbol_uses_builtin_for_capacitor() -> None:
    """Capacitor is matched to Device:C."""
    comp = _make_component(ref="C1", value="100nF", description="capacitor 100nF")
    cache: dict[str, LibSymbol] = {}
    sym = get_or_make_symbol(comp, cache)
    assert sym.lib_id == "Device:C"


def test_get_or_make_symbol_uses_builtin_for_led() -> None:
    """LED description maps to Device:LED."""
    comp = _make_component(ref="D1", value="RED", description="LED indicator red")
    cache: dict[str, LibSymbol] = {}
    sym = get_or_make_symbol(comp, cache)
    assert sym.lib_id == "Device:LED"


def test_get_or_make_symbol_lcsc_cache_hit() -> None:
    """Symbol found by LCSC number in cache."""
    from kicad_pipeline.models.requirements import Component as Comp

    comp = Comp(ref="U5", value="IC", footprint="QFP", lcsc="C12345")
    dummy_sym = make_passive_symbol("cached:sym")
    cache: dict[str, LibSymbol] = {"C12345": dummy_sym}
    sym = get_or_make_symbol(comp, cache)
    assert sym is dummy_sym


def test_get_or_make_symbol_ref_cache_hit() -> None:
    """Symbol found by ref in cache."""
    comp = _make_component(ref="U42", value="CustomIC")
    dummy_sym = make_passive_symbol("cached:ref")
    cache: dict[str, LibSymbol] = {"U42": dummy_sym}
    sym = get_or_make_symbol(comp, cache)
    assert sym is dummy_sym


def test_get_or_make_symbol_ref_prefix_fallback_d() -> None:
    """D prefix (without description) falls back to Device:D."""
    comp = _make_component(ref="D5", value="1N4148")
    cache: dict[str, LibSymbol] = {}
    sym = get_or_make_symbol(comp, cache)
    assert sym.lib_id == "Device:D"


# ---------------------------------------------------------------------------
# BUILTIN_SYMBOLS — additional
# ---------------------------------------------------------------------------


def test_builtin_symbols_contain_power_gnd() -> None:
    """BUILTIN_SYMBOLS contains power:GND."""
    assert "power:GND" in BUILTIN_SYMBOLS


def test_builtin_symbols_contain_device_d() -> None:
    """BUILTIN_SYMBOLS contains Device:D."""
    assert "Device:D" in BUILTIN_SYMBOLS


def test_builtin_symbols_contain_npn() -> None:
    """BUILTIN_SYMBOLS contains Device:Q_NPN_BCE."""
    assert "Device:Q_NPN_BCE" in BUILTIN_SYMBOLS


def test_builtin_npn_has_three_pins() -> None:
    """NPN transistor symbol has 3 pins (B, C, E)."""
    sym = BUILTIN_SYMBOLS["Device:Q_NPN_BCE"]
    assert len(sym.pins) == 3
    names = {p.name for p in sym.pins}
    assert names == {"B", "C", "E"}
