"""Symbol lookup and caching functionality."""

import logging

from kicad_pipeline.models.requirements import Component
from kicad_pipeline.models.schematic import LibSymbol
from kicad_pipeline.schematic.symbols.active import (
    _make_diode_symbol,
    _make_npn_symbol,
    make_led_symbol,
)
from kicad_pipeline.schematic.symbols.core.lib_symbol import make_lib_symbol
from kicad_pipeline.schematic.symbols.passive import make_passive_symbol
from kicad_pipeline.schematic.symbols.power import make_power_symbol

__all__ = ["BUILTIN_SYMBOLS", "get_or_make_symbol"]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in symbol catalogue
# ---------------------------------------------------------------------------

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
