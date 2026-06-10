"""Active component symbols (LEDs, diodes, transistors)."""

from kicad_pipeline.constants import SCHEMATIC_PIN_LENGTH_MM
from kicad_pipeline.models.schematic import LibPin, LibPolyline, LibSymbol, Point, Stroke
from kicad_pipeline.schematic.symbols.utils import _make_font

__all__ = ["make_led_symbol", "_make_diode_symbol", "_make_npn_symbol"]


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
        pin_type=1,  # PASSIVE
        at=Point(-(SCHEMATIC_PIN_LENGTH_MM + 1.27), 0.0),
        rotation=0.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )
    cathode = LibPin(
        number="2",
        name="K",
        pin_type=1,  # PASSIVE
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
        pin_type=1,  # PASSIVE
        at=Point(-(SCHEMATIC_PIN_LENGTH_MM + 1.27), 0.0),
        rotation=0.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )
    cathode = LibPin(
        number="2",
        name="K",
        pin_type=1,  # PASSIVE
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
        pin_type=0,  # INPUT
        at=Point(-(SCHEMATIC_PIN_LENGTH_MM + 5.08), 0.0),
        rotation=0.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )
    collector = LibPin(
        number="2",
        name="C",
        pin_type=1,  # PASSIVE
        at=Point(SCHEMATIC_PIN_LENGTH_MM + 5.08, -2.54),
        rotation=180.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )
    emitter = LibPin(
        number="3",
        name="E",
        pin_type=1,  # PASSIVE
        at=Point(SCHEMATIC_PIN_LENGTH_MM + 5.08, 2.54),
        rotation=180.0,
        length=SCHEMATIC_PIN_LENGTH_MM,
        name_effects=_make_font(),
        number_effects=_make_font(hidden=True),
    )
    return LibSymbol(lib_id=lib_id, pins=(base, collector, emitter), shapes=(body,))