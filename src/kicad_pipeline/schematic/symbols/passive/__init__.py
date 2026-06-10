"""Passive component symbols (resistors, capacitors, inductors)."""

from kicad_pipeline.constants import SCHEMATIC_PIN_LENGTH_MM
from kicad_pipeline.models.schematic import LibPin, LibPolyline, LibSymbol, Point, Stroke
from kicad_pipeline.schematic.symbols.utils import _make_font

__all__ = ["make_passive_symbol"]


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
        pin_type=1,  # PASSIVE
        at=Point(-(SCHEMATIC_PIN_LENGTH_MM + _half), 0.0),
        rotation=0.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(hidden=True),
        number_effects=_make_font(hidden=True),
    )
    pin2 = LibPin(
        number="2",
        name=pin2_name,
        pin_type=1,  # PASSIVE
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