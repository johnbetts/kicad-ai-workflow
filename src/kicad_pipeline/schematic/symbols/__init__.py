"""KiCad lib_symbol generation for the kicad-ai-pipeline.

Generates :class:`~kicad_pipeline.models.schematic.LibSymbol` objects that
populate the ``lib_symbols`` section of a KiCad schematic file.  Symbols are
drawn as rectangular IC-style boxes with pins placed by electrical function
group.
"""

from kicad_pipeline.schematic.symbols.active import make_led_symbol
from kicad_pipeline.schematic.symbols.core import (
    BUILTIN_SYMBOLS,
    get_or_make_symbol,
    make_lib_symbol,
)
from kicad_pipeline.schematic.symbols.passive import make_passive_symbol
from kicad_pipeline.schematic.symbols.power import make_power_symbol

__all__ = [
    "BUILTIN_SYMBOLS",
    "get_or_make_symbol",
    "make_led_symbol",
    "make_lib_symbol",
    "make_passive_symbol",
    "make_power_symbol",
]
