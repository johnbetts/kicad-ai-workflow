"""Core library symbol generation."""

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
    LibRectangle,
    LibSymbol,
    Point,
    Stroke,
)
from kicad_pipeline.schematic.symbols.utils import _make_font

__all__ = ["make_lib_symbol"]

log = logging.getLogger(__name__)

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


def _place_side_pins(
    placed_pins: list[LibPin],
    pins: list[Pin],
    pin_x: float,
    body_top: float,
    *,
    rotation: float,
) -> None:
    """Place pins along a vertical side (left or right) of the symbol body."""
    for idx, pin in enumerate(pins):
        y = body_top - SCHEMATIC_SYMBOL_PIN_SPACING_MM * (idx + 1)
        placed_pins.append(
            LibPin(
                number=pin.number,
                name=pin.name,
                pin_type=pin.pin_type.value,
                at=Point(pin_x, y),
                rotation=rotation,
                length=SCHEMATIC_PIN_LENGTH_MM,
                name_effects=_make_font(),
                number_effects=_make_font(hidden=True),
            )
        )


def _place_horiz_pins(
    placed_pins: list[LibPin],
    pins: list[Pin],
    pin_y: float,
    rotation: float,
) -> None:
    """Place pins along a horizontal edge (top or bottom) of the symbol body."""
    for idx, pin in enumerate(pins):
        x = (idx - (len(pins) - 1) / 2.0) * SCHEMATIC_SYMBOL_PIN_SPACING_MM
        placed_pins.append(
            LibPin(
                number=pin.number,
                name=pin.name,
                pin_type=pin.pin_type.value,
                at=Point(x, pin_y),
                rotation=rotation,
                length=SCHEMATIC_PIN_LENGTH_MM,
                name_effects=_make_font(),
                number_effects=_make_font(hidden=True),
            )
        )


def _lib_symbol_classify_pins(
    component: Component,
) -> tuple[list[Pin], list[Pin], list[Pin], list[Pin]]:
    ref_prefix = "".join(ch for ch in component.ref if ch.isalpha()) or "U"
    is_multirow_conn = (
        ref_prefix == "J"
        and "Conn_02x" in component.value
        and len(component.pins) > 4
    )
    if is_multirow_conn:
        return _classify_connector_pins(component.pins)
    return _classify_pins(component.pins)


def _lib_symbol_build_body(
    left_pins: list[Pin], right_pins: list[Pin],
) -> tuple[LibRectangle, float, float, float]:
    side_max = max(len(left_pins), len(right_pins), 1)
    body_height = side_max * SCHEMATIC_SYMBOL_PIN_SPACING_MM + SCHEMATIC_SYMBOL_PIN_SPACING_MM
    half_w = _body_half_width(left_pins, right_pins)
    body_top = body_height / 2.0
    body_bottom = -body_height / 2.0
    rect = LibRectangle(
        start=Point(-half_w, body_top),
        end=Point(half_w, body_bottom),
        stroke=Stroke(),
        fill="background",
    )
    return rect, half_w, body_top, body_bottom


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
    ref_prefix = "".join(ch for ch in component.ref if ch.isalpha()) or "U"
    lib_id = f"kicad-ai:{ref_prefix}_{component.value}"

    left_pins, right_pins, top_pins, bottom_pins = _lib_symbol_classify_pins(component)
    rect, half_w, body_top, body_bottom = _lib_symbol_build_body(left_pins, right_pins)

    pin_x_left = -(half_w + SCHEMATIC_PIN_LENGTH_MM)
    pin_x_right = half_w + SCHEMATIC_PIN_LENGTH_MM

    placed_pins: list[LibPin] = []
    _place_side_pins(placed_pins, left_pins, pin_x_left, body_top, rotation=0.0)
    _place_side_pins(placed_pins, right_pins, pin_x_right, body_top, rotation=180.0)
    _place_horiz_pins(placed_pins, top_pins, body_top + SCHEMATIC_PIN_LENGTH_MM, 270.0)
    _place_horiz_pins(placed_pins, bottom_pins, body_bottom - SCHEMATIC_PIN_LENGTH_MM, 90.0)

    log.debug("Generated lib_symbol %s with %d pins", lib_id, len(placed_pins))

    return LibSymbol(
        lib_id=lib_id,
        pins=tuple(placed_pins),
        shapes=(rect,),
    )