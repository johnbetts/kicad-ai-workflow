"""KiCad lib_symbol generation for the kicad-ai-pipeline.

Generates :class:`~kicad_pipeline.models.schematic.LibSymbol` objects that
populate the ``lib_symbols`` section of a KiCad schematic file.  Symbols are
drawn as rectangular IC-style boxes with pins placed by electrical function
group.
"""

from __future__ import annotations

import logging

from kicad_pipeline.constants import (
    SCHEMATIC_PIN_LENGTH_MM,
    SCHEMATIC_SYMBOL_PIN_SPACING_MM,
    SCHEMATIC_TEXT_SIZE_MM,
)
from kicad_pipeline.models.requirements import Component, Pin, PinType
from kicad_pipeline.models.schematic import (
    FontEffect,
    LibPin,
    LibPolyline,
    LibRectangle,
    LibSymbol,
    Point,
    Stroke,
)

__all__ = [
    "BUILTIN_SYMBOLS",
    "get_or_make_symbol",
    "make_led_symbol",
    "make_lib_symbol",
    "make_passive_symbol",
    "make_power_symbol",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MIN_BODY_HALF_WIDTH_MM: float = 5.08
"""Minimum half-width of the rectangular IC symbol body (mm)."""


def _body_half_width(
    left_pins: list[Pin],
    right_pins: list[Pin],
) -> float:
    """Compute body half-width to fit the longest pin name on each side.

    Each character is approximately 1.0mm at the default font size.  The body
    must be wide enough that left-side pin names (rendered inside the body to
    the right of the pin) and right-side pin names (rendered inside the body to
    the left of the pin) do not overlap.
    """
    left_max = max((len(p.name) for p in left_pins), default=0)
    right_max = max((len(p.name) for p in right_pins), default=0)
    # Each side needs ~1.0mm per character plus a small margin
    needed = max(left_max, right_max) * 1.0 + 2.0
    # If both sides have names, total width must fit both
    if left_pins and right_pins:
        needed = max(needed, (left_max + right_max) * 1.0 / 2.0 + 3.0)
    return max(_MIN_BODY_HALF_WIDTH_MM, round(needed / 1.27) * 1.27)

_DEFAULT_FONT: FontEffect = FontEffect(
    size_x=SCHEMATIC_TEXT_SIZE_MM,
    size_y=SCHEMATIC_TEXT_SIZE_MM,
)

_HIDDEN_FONT: FontEffect = FontEffect(
    size_x=SCHEMATIC_TEXT_SIZE_MM,
    size_y=SCHEMATIC_TEXT_SIZE_MM,
    hidden=True,
)


def _make_font(hidden: bool = False) -> FontEffect:
    """Return a :class:`FontEffect` with standard text size."""
    return _HIDDEN_FONT if hidden else _DEFAULT_FONT


def _classify_pins(
    pins: tuple[Pin, ...],
) -> tuple[list[Pin], list[Pin], list[Pin], list[Pin]]:
    """Split pins into (left, right, top, bottom) placement groups.

    Rules
    -----
    * power_in whose name *contains* GND or VSS → bottom group
    * power_in / power_out → top group
    * output → right group
    * everything else (input, passive, bidirectional, open_collector) → left
    * no_connect pins are skipped entirely

    When the left side would have more than 12 pins (e.g. large connectors),
    the overflow is moved to the right side to keep the symbol compact.

    Args:
        pins: All component pins.

    Returns:
        Four lists: left_pins, right_pins, top_pins, bottom_pins.
    """
    left: list[Pin] = []
    right: list[Pin] = []
    top: list[Pin] = []
    bottom: list[Pin] = []

    # For small components (≤3 pins), use simple horizontal layout
    active_pins = [p for p in pins if p.pin_type is not PinType.NO_CONNECT]
    if len(active_pins) <= 3:
        for i, pin in enumerate(active_pins):
            if i % 2 == 0:
                left.append(pin)
            else:
                right.append(pin)
        return left, right, top, bottom

    for pin in pins:
        if pin.pin_type is PinType.NO_CONNECT:
            continue
        upper_name = pin.name.upper()
        is_gnd = "GND" in upper_name or "VSS" in upper_name
        if pin.pin_type in (PinType.POWER_IN, PinType.POWER_OUT):
            if is_gnd:
                bottom.append(pin)
            else:
                top.append(pin)
        elif pin.pin_type is PinType.OUTPUT:
            right.append(pin)
        elif pin.pin_type is PinType.BIDIRECTIONAL:
            # Communication pins (I2C, SPI, UART) go on the right
            right.append(pin)
        else:
            left.append(pin)

    # Balance left/right: split evenly when all pins end up on one side
    # (common for DIP switches, multi-pin passives, etc.)
    if len(left) > 4 and not right and not top and not bottom:
        half = (len(left) + 1) // 2
        right = left[half:]
        left = left[:half]
    elif len(left) > 12 and len(right) < 12:
        target_per_side = (len(left) + len(right) + 1) // 2
        overflow = len(left) - target_per_side
        if overflow > 0:
            right = left[-overflow:] + right
            left = left[:-overflow]

    return left, right, top, bottom


def _classify_connector_pins(
    pins: tuple[Pin, ...],
) -> tuple[list[Pin], list[Pin], list[Pin], list[Pin]]:
    """Split connector pins into left (odd) and right (even) groups.

    Multi-row connectors (Conn_02xNN) are best shown with odd pins on the
    left and even pins on the right, matching their physical layout.  Single-
    row connectors (Conn_01xNN) just put all pins on the left.

    Args:
        pins: All connector pins.

    Returns:
        Four lists: left_pins, right_pins, top_pins, bottom_pins.
    """
    left: list[Pin] = []
    right: list[Pin] = []
    for pin in pins:
        if pin.pin_type is PinType.NO_CONNECT:
            continue
        try:
            num = int(pin.number)
        except ValueError:
            left.append(pin)
            continue
        if num % 2 == 1:
            left.append(pin)
        else:
            right.append(pin)
    # Sort by pin number within each side
    left.sort(key=lambda p: int(p.number) if p.number.isdigit() else 0)
    right.sort(key=lambda p: int(p.number) if p.number.isdigit() else 0)
    return left, right, [], []


# ---------------------------------------------------------------------------
# Public symbol constructors
# ---------------------------------------------------------------------------


def make_lib_symbol(component: Component) -> LibSymbol:
    """Generate a :class:`LibSymbol` from a :class:`Component` definition.

    Pins are sorted into groups:

    * inputs / passive / bidirectional: left side
    * outputs: right side
    * power_in (VCC/VDD) / power_out: top
    * power_in that contains GND or VSS: bottom
    * no_connect: omitted from the symbol body

    For multi-row connectors (``Conn_02x``), pins are split odd (left) /
    even (right) to match physical layout.

    The symbol body is a filled rectangle.  The ``lib_id`` is formatted as
    ``"kicad-ai:{ref_prefix}_{value}"`` where *ref_prefix* is the leading
    alpha characters of :attr:`Component.ref`.

    Args:
        component: The component to generate a symbol for.

    Returns:
        A :class:`LibSymbol` with a rectangular body and all placed pins.
    """
    # Derive ref prefix (leading alpha chars, e.g. "U" from "U1")
    ref_prefix = "".join(ch for ch in component.ref if ch.isalpha()) or "U"
    lib_id = f"kicad-ai:{ref_prefix}_{component.value}"

    # Use connector-specific pin classification for multi-row connectors
    is_multirow_conn = (
        ref_prefix == "J"
        and "Conn_02x" in component.value
        and len(component.pins) > 4
    )
    if is_multirow_conn:
        left_pins, right_pins, top_pins, bottom_pins = _classify_connector_pins(
            component.pins,
        )
    else:
        left_pins, right_pins, top_pins, bottom_pins = _classify_pins(component.pins)

    # Determine body height from the tallest left/right side
    side_max = max(len(left_pins), len(right_pins), 1)
    body_height = side_max * SCHEMATIC_SYMBOL_PIN_SPACING_MM + SCHEMATIC_SYMBOL_PIN_SPACING_MM

    # Adaptive body width based on pin name lengths
    half_w = _body_half_width(left_pins, right_pins)
    pin_x_left = -(half_w + SCHEMATIC_PIN_LENGTH_MM)
    pin_x_right = half_w + SCHEMATIC_PIN_LENGTH_MM

    # KiCad lib_symbol Y-axis: positive = up (mathematical convention)
    # body_top is the visual top (positive Y), body_bottom is visual bottom (negative Y)
    body_top = body_height / 2.0
    body_bottom = -body_height / 2.0
    rect = LibRectangle(
        start=Point(-half_w, body_top),
        end=Point(half_w, body_bottom),
        stroke=Stroke(),
        fill="background",
    )

    placed_pins: list[LibPin] = []

    # Left-side pins (inputs etc.) — top to bottom (positive to negative Y)
    # KiCad convention: left pins at negative X, rotation=0° (extends RIGHT toward body)
    for idx, pin in enumerate(left_pins):
        y = body_top - SCHEMATIC_SYMBOL_PIN_SPACING_MM * (idx + 1)
        placed_pins.append(
            LibPin(
                number=pin.number,
                name=pin.name,
                pin_type=pin.pin_type.value,
                at=Point(pin_x_left, y),
                rotation=0.0,
                length=SCHEMATIC_PIN_LENGTH_MM,
                name_effects=_make_font(),
                number_effects=_make_font(hidden=True),
            )
        )

    # Right-side pins (outputs) — top to bottom (positive to negative Y)
    # KiCad convention: right pins at positive X, rotation=180° (extends LEFT toward body)
    for idx, pin in enumerate(right_pins):
        y = body_top - SCHEMATIC_SYMBOL_PIN_SPACING_MM * (idx + 1)
        placed_pins.append(
            LibPin(
                number=pin.number,
                name=pin.name,
                pin_type=pin.pin_type.value,
                at=Point(pin_x_right, y),
                rotation=180.0,
                length=SCHEMATIC_PIN_LENGTH_MM,
                name_effects=_make_font(),
                number_effects=_make_font(hidden=True),
            )
        )

    # Top pins (power supply) — above body (positive Y direction)
    # KiCad convention: rotation=270° (extends DOWN toward body)
    for idx, pin in enumerate(top_pins):
        x = (idx - (len(top_pins) - 1) / 2.0) * SCHEMATIC_SYMBOL_PIN_SPACING_MM
        placed_pins.append(
            LibPin(
                number=pin.number,
                name=pin.name,
                pin_type=pin.pin_type.value,
                at=Point(x, body_top + SCHEMATIC_PIN_LENGTH_MM),
                rotation=270.0,
                length=SCHEMATIC_PIN_LENGTH_MM,
                name_effects=_make_font(),
                number_effects=_make_font(hidden=True),
            )
        )

    # Bottom pins (GND / VSS) — below body (negative Y direction)
    # KiCad convention: rotation=90° (extends UP toward body)
    for idx, pin in enumerate(bottom_pins):
        x = (idx - (len(bottom_pins) - 1) / 2.0) * SCHEMATIC_SYMBOL_PIN_SPACING_MM
        placed_pins.append(
            LibPin(
                number=pin.number,
                name=pin.name,
                pin_type=pin.pin_type.value,
                at=Point(x, body_bottom - SCHEMATIC_PIN_LENGTH_MM),
                rotation=90.0,
                length=SCHEMATIC_PIN_LENGTH_MM,
                name_effects=_make_font(),
                number_effects=_make_font(hidden=True),
            )
        )

    log.debug(
        "Generated lib_symbol %s with %d pins",
        lib_id,
        len(placed_pins),
    )

    return LibSymbol(
        lib_id=lib_id,
        pins=tuple(placed_pins),
        shapes=(rect,),
    )


def make_passive_symbol(
    lib_id: str,
    pin1_name: str = "~",
    pin2_name: str = "~",
) -> LibSymbol:
    """Make a two-pin passive symbol (resistor / capacitor / inductor style).

    Pin 1 is placed on the left (rotation=180) and pin 2 on the right
    (rotation=0).  The body is a simple horizontal polyline with end marks
    to indicate the component outline.

    Args:
        lib_id: KiCad lib_id string, e.g. ``"Device:R"``.
        pin1_name: Name for pin 1 (default ``"~"``).
        pin2_name: Name for pin 2 (default ``"~"``).

    Returns:
        A :class:`LibSymbol` with two passive pins and a polyline body.
    """
    # Body: horizontal line from -1.27 to +1.27 with short end marks
    _half: float = 1.27
    body_line = LibPolyline(
        points=(Point(-_half, 0.0), Point(_half, 0.0)),
        stroke=Stroke(),
        fill="none",
    )
    left_mark = LibPolyline(
        points=(Point(-_half, -0.508), Point(-_half, 0.508)),
        stroke=Stroke(),
        fill="none",
    )
    right_mark = LibPolyline(
        points=(Point(_half, -0.508), Point(_half, 0.508)),
        stroke=Stroke(),
        fill="none",
    )

    pin1 = LibPin(
        number="1",
        name=pin1_name,
        pin_type=PinType.PASSIVE.value,
        at=Point(-(SCHEMATIC_PIN_LENGTH_MM + _half), 0.0),
        rotation=0.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(hidden=True),
        number_effects=_make_font(hidden=True),
    )
    pin2 = LibPin(
        number="2",
        name=pin2_name,
        pin_type=PinType.PASSIVE.value,
        at=Point(SCHEMATIC_PIN_LENGTH_MM + _half, 0.0),
        rotation=180.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(hidden=True),
        number_effects=_make_font(hidden=True),
    )

    return LibSymbol(
        lib_id=lib_id,
        pins=(pin1, pin2),
        shapes=(body_line, left_mark, right_mark),
    )


def make_led_symbol(lib_id: str = "Device:LED") -> LibSymbol:
    """Make a two-pin LED symbol with K (cathode) and A (anode).

    The anode (A) is pin 1 on the left; the cathode (K) is pin 2 on the
    right.  The body is represented as a triangle pointing right (diode
    convention) with a short bar at the tip.

    Args:
        lib_id: KiCad lib_id for this symbol (default ``"Device:LED"``).

    Returns:
        A :class:`LibSymbol` representing an LED.
    """
    # Diode triangle body: tip pointing right
    triangle = LibPolyline(
        points=(
            Point(-1.27, -1.27),
            Point(-1.27, 1.27),
            Point(1.27, 0.0),
            Point(-1.27, -1.27),
        ),
        stroke=Stroke(),
        fill="background",
    )
    # Cathode bar
    bar = LibPolyline(
        points=(Point(1.27, -1.27), Point(1.27, 1.27)),
        stroke=Stroke(),
        fill="none",
    )

    anode = LibPin(
        number="1",
        name="A",
        pin_type=PinType.PASSIVE.value,
        at=Point(-(SCHEMATIC_PIN_LENGTH_MM + 1.27), 0.0),
        rotation=0.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )
    cathode = LibPin(
        number="2",
        name="K",
        pin_type=PinType.PASSIVE.value,
        at=Point(SCHEMATIC_PIN_LENGTH_MM + 1.27, 0.0),
        rotation=180.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )

    return LibSymbol(
        lib_id=lib_id,
        pins=(anode, cathode),
        shapes=(triangle, bar),
    )


def make_power_symbol(net_name: str) -> LibSymbol:
    """Make a power supply symbol (VCC, GND, +3V3, etc.).

    GND / VSS symbols have a pin pointing downward (rotation=270) with a
    small downward-pointing triangle body.  All other (positive) power
    symbols have a pin pointing upward (rotation=90) with a short horizontal
    bar body.

    Args:
        net_name: The net / power rail name, e.g. ``"GND"``, ``"+3.3V"``.

    Returns:
        A :class:`LibSymbol` with ``lib_id = "power:{net_name}"``.
    """
    lib_id = f"power:{net_name}"
    upper = net_name.upper()
    is_gnd = "GND" in upper or "VSS" in upper

    if is_gnd:
        # Pin pointing down, triangle body below pin attachment
        pin = LibPin(
            number="1",
            name="~",
            pin_type=PinType.POWER_IN.value,
            at=Point(0.0, 0.0),
            rotation=270.0,
            length=SCHEMATIC_PIN_LENGTH_MM,
            name_effects=_make_font(hidden=True),
            number_effects=_make_font(hidden=True),
        )
        # Triangle below the pin tip
        tip_y = SCHEMATIC_PIN_LENGTH_MM
        triangle = LibPolyline(
            points=(
                Point(-1.27, tip_y),
                Point(1.27, tip_y),
                Point(0.0, tip_y + 1.27),
                Point(-1.27, tip_y),
            ),
            stroke=Stroke(),
            fill="background",
        )
        shapes: tuple[LibPolyline, ...] = (triangle,)
    else:
        # Pin pointing up, horizontal bar at top
        pin = LibPin(
            number="1",
            name="~",
            pin_type=PinType.POWER_IN.value,
            at=Point(0.0, 0.0),
            rotation=90.0,
            length=SCHEMATIC_PIN_LENGTH_MM,
            name_effects=_make_font(hidden=True),
            number_effects=_make_font(hidden=True),
        )
        bar_y = -SCHEMATIC_PIN_LENGTH_MM
        bar = LibPolyline(
            points=(Point(-1.27, bar_y), Point(1.27, bar_y)),
            stroke=Stroke(),
            fill="none",
        )
        shapes = (bar,)

    return LibSymbol(
        lib_id=lib_id,
        pins=(pin,),
        shapes=shapes,
    )


# ---------------------------------------------------------------------------
# Built-in symbol catalogue
# ---------------------------------------------------------------------------

def _make_diode_symbol(lib_id: str = "Device:D") -> LibSymbol:
    """Make a generic two-pin diode symbol.

    Args:
        lib_id: KiCad lib_id (default ``"Device:D"``).

    Returns:
        A :class:`LibSymbol` representing a diode.
    """
    triangle = LibPolyline(
        points=(
            Point(-1.27, -1.27),
            Point(-1.27, 1.27),
            Point(1.27, 0.0),
            Point(-1.27, -1.27),
        ),
        stroke=Stroke(),
        fill="background",
    )
    bar = LibPolyline(
        points=(Point(1.27, -1.27), Point(1.27, 1.27)),
        stroke=Stroke(),
        fill="none",
    )
    anode = LibPin(
        number="1",
        name="A",
        pin_type=PinType.PASSIVE.value,
        at=Point(-(SCHEMATIC_PIN_LENGTH_MM + 1.27), 0.0),
        rotation=0.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )
    cathode = LibPin(
        number="2",
        name="K",
        pin_type=PinType.PASSIVE.value,
        at=Point(SCHEMATIC_PIN_LENGTH_MM + 1.27, 0.0),
        rotation=180.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )
    return LibSymbol(lib_id=lib_id, pins=(anode, cathode), shapes=(triangle, bar))


def _make_npn_symbol() -> LibSymbol:
    """Make a simplified NPN transistor symbol (Device:Q_NPN_BCE).

    Returns:
        A :class:`LibSymbol` with B (base), C (collector), E (emitter) pins.
    """
    lib_id = "Device:Q_NPN_BCE"
    # Simplified body lines
    body = LibPolyline(
        points=(Point(0.0, -2.54), Point(0.0, 2.54)),
        stroke=Stroke(),
        fill="none",
    )
    base = LibPin(
        number="1",
        name="B",
        pin_type=PinType.INPUT.value,
        at=Point(-(SCHEMATIC_PIN_LENGTH_MM + _MIN_BODY_HALF_WIDTH_MM), 0.0),
        rotation=0.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )
    collector = LibPin(
        number="2",
        name="C",
        pin_type=PinType.PASSIVE.value,
        at=Point(SCHEMATIC_PIN_LENGTH_MM + _MIN_BODY_HALF_WIDTH_MM, -2.54),
        rotation=180.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )
    emitter = LibPin(
        number="3",
        name="E",
        pin_type=PinType.PASSIVE.value,
        at=Point(SCHEMATIC_PIN_LENGTH_MM + _MIN_BODY_HALF_WIDTH_MM, 2.54),
        rotation=180.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )
    return LibSymbol(lib_id=lib_id, pins=(base, collector, emitter), shapes=(body,))


BUILTIN_SYMBOLS: dict[str, LibSymbol] = {
    "Device:R": make_passive_symbol("Device:R"),
    "Device:C": make_passive_symbol("Device:C"),
    "Device:L": make_passive_symbol("Device:L"),
    "Device:LED": make_led_symbol("Device:LED"),
    "Device:D": _make_diode_symbol("Device:D"),
    "Device:Q_NPN_BCE": _make_npn_symbol(),
    "power:GND": make_power_symbol("GND"),
    "power:+3.3V": make_power_symbol("+3.3V"),
    "power:+5V": make_power_symbol("+5V"),
    "power:VCC": make_power_symbol("VCC"),
}
"""Pre-populated built-in symbol catalogue matching common KiCad Device/power libs."""


# ---------------------------------------------------------------------------
# Category → built-in symbol heuristic
# ---------------------------------------------------------------------------

_CATEGORY_BUILTINS: dict[str, str] = {
    "resistor": "Device:R",
    "capacitor": "Device:C",
    "inductor": "Device:L",
    "led": "Device:LED",
    "diode_switching": "Device:D",
    "diode_esd": "Device:D",
    "transistor_npn": "Device:Q_NPN_BCE",
}

_REF_PREFIX_BUILTINS: dict[str, str] = {
    "R": "Device:R",
    "C": "Device:C",
    "L": "Device:L",
    "D": "Device:D",
}


def _infer_category(component: Component) -> str | None:
    """Infer a category string from a :class:`Component`.

    The inference order is:

    1. Check the ``description`` field for keyword matches.
    2. Fall back to the reference-designator prefix (``R``, ``C``, ``L``, ``D``).

    Args:
        component: The component to inspect.

    Returns:
        A category key compatible with :data:`_CATEGORY_BUILTINS`, or ``None``
        if no category can be inferred.
    """
    desc = (component.description or "").lower()
    value_lower = component.value.lower()
    combined = f"{desc} {value_lower}"

    if "resistor" in combined or "ohm" in combined:
        return "resistor"
    if "capacitor" in combined or "farad" in combined:
        return "capacitor"
    if "inductor" in combined or "henry" in combined:
        return "inductor"
    if "led" in combined:
        return "led"
    if "diode" in combined and "esd" in combined:
        return "diode_esd"
    if "diode" in combined:
        return "diode_switching"
    if "npn" in combined or "transistor" in combined:
        return "transistor_npn"

    # Ref prefix fallback
    ref_prefix = "".join(ch for ch in component.ref if ch.isalpha())
    if ref_prefix in _REF_PREFIX_BUILTINS:
        return ref_prefix  # use the ref prefix as a pseudo-category key

    return None


def get_or_make_symbol(
    component: Component,
    lib_cache: dict[str, LibSymbol],
) -> LibSymbol:
    """Look up a symbol in the cache, falling back to auto-generation.

    Resolution order:

    1. Category heuristic → :data:`BUILTIN_SYMBOLS`.
    2. ``component.lcsc`` key in *lib_cache*.
    3. ``component.ref`` key in *lib_cache*.
    4. Auto-generate via :func:`make_lib_symbol` and store in *lib_cache*.

    Args:
        component: The component to look up.
        lib_cache: Mutable cache of already-generated symbols (mutated in
            place when a new symbol is generated).

    Returns:
        A :class:`LibSymbol` for *component*.
    """
    # 1. Category heuristic
    category = _infer_category(component)
    if category is not None:
        # Try direct category key first, then ref-prefix-based lookup
        builtin_id = _CATEGORY_BUILTINS.get(category)
        if builtin_id is None:
            # category is a ref prefix (e.g. "R", "C")
            builtin_id = _REF_PREFIX_BUILTINS.get(category)
        if builtin_id is not None and builtin_id in BUILTIN_SYMBOLS:
            log.debug(
                "Component %s mapped to built-in symbol %s via category %r",
                component.ref,
                builtin_id,
                category,
            )
            return BUILTIN_SYMBOLS[builtin_id]

    # 2. LCSC key in lib_cache
    if component.lcsc is not None and component.lcsc in lib_cache:
        return lib_cache[component.lcsc]

    # 3. Ref key in lib_cache
    if component.ref in lib_cache:
        return lib_cache[component.ref]

    # 4. Auto-generate
    symbol = make_lib_symbol(component)
    lib_cache[component.ref] = symbol
    log.debug("Auto-generated lib_symbol %s for component %s", symbol.lib_id, component.ref)
    return symbol
