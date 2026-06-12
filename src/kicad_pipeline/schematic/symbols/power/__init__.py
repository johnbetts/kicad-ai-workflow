"""Power supply symbols."""

from kicad_pipeline.constants import SCHEMATIC_PIN_LENGTH_MM
from kicad_pipeline.models.schematic import LibPin, LibPolyline, LibSymbol, Point, Stroke
from kicad_pipeline.schematic.symbols.utils import _make_font

__all__ = ["make_power_symbol"]


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
            pin_type="power_in",
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
        shapes = (triangle,)
    else:
        # Pin pointing up, horizontal bar at top
        pin = LibPin(
            number="1",
            name="~",
            pin_type="power_in",
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
