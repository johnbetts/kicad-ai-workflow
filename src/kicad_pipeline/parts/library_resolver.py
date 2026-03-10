"""Resolve JLCPCB parts to KiCad symbol and footprint library references.

Searches the CDFER JLCPCB KiCad Library first, then built-in KiCad
symbol/footprint libraries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.parts.jlcpcb_db import JLCPCBPart

logger = logging.getLogger(__name__)

# Standard CDFER JLCPCB library install paths.
_CDFER_SYMBOL_DIRS: tuple[Path, ...] = (
    Path.home()
    / "Documents"
    / "KiCad"
    / "10.0"
    / "3rdparty"
    / "symbols"
    / "com_github_CDFER_JLCPCB-Kicad-Library",
    Path.home()
    / "Documents"
    / "KiCad"
    / "9.0"
    / "3rdparty"
    / "symbols"
    / "com_github_CDFER_JLCPCB-Kicad-Library",
)

_CDFER_FOOTPRINT_DIRS: tuple[Path, ...] = (
    Path.home()
    / "Documents"
    / "KiCad"
    / "10.0"
    / "3rdparty"
    / "footprints"
    / "com_github_CDFER_JLCPCB-Kicad-Library"
    / "JLCPCB.pretty",
    Path.home()
    / "Documents"
    / "KiCad"
    / "9.0"
    / "3rdparty"
    / "footprints"
    / "com_github_CDFER_JLCPCB-Kicad-Library"
    / "JLCPCB.pretty",
)

# KiCad built-in library paths (macOS).
_KICAD_BUILTIN_SYMBOLS: tuple[Path, ...] = (
    Path("/Applications/KiCad 10/KiCad.app/Contents/SharedSupport/symbols"),
    Path("/Applications/KiCad/KiCad.app/Contents/SharedSupport/symbols"),
)
_KICAD_BUILTIN_FOOTPRINTS: tuple[Path, ...] = (
    Path("/Applications/KiCad 10/KiCad.app/Contents/SharedSupport/footprints"),
    Path("/Applications/KiCad/KiCad.app/Contents/SharedSupport/footprints"),
)

# Category → KiCad built-in library mapping for common passives.
_CATEGORY_TO_BUILTIN_SYMBOL: dict[str, str] = {
    "Resistors": "Device:R",
    "Capacitors": "Device:C",
    "Inductors": "Device:L",
    "Diodes": "Device:D",
    "LEDs": "Device:LED",
}

# Package → KiCad built-in footprint mapping for SMD passives.
_PACKAGE_TO_BUILTIN_FOOTPRINT: dict[str, str] = {
    "0201": "Resistor_SMD:R_0201_0603Metric",
    "0402": "Resistor_SMD:R_0402_1005Metric",
    "0603": "Resistor_SMD:R_0603_1608Metric",
    "0805": "Resistor_SMD:R_0805_2012Metric",
    "1206": "Resistor_SMD:R_1206_3216Metric",
    "1210": "Resistor_SMD:R_1210_3225Metric",
    "2010": "Resistor_SMD:R_2010_5025Metric",
    "2512": "Resistor_SMD:R_2512_6332Metric",
}

# Capacitor-specific package mappings.
_CAP_PACKAGE_TO_FOOTPRINT: dict[str, str] = {
    "0201": "Capacitor_SMD:C_0201_0603Metric",
    "0402": "Capacitor_SMD:C_0402_1005Metric",
    "0603": "Capacitor_SMD:C_0603_1608Metric",
    "0805": "Capacitor_SMD:C_0805_2012Metric",
    "1206": "Capacitor_SMD:C_1206_3216Metric",
    "1210": "Capacitor_SMD:C_1210_3225Metric",
}

# CDFER library file names (without path) → lib nickname.
_CDFER_SYMBOL_FILES: dict[str, str] = {
    "JLCPCB-Resistors.kicad_sym": "JLCPCB-Resistors",
    "JLCPCB-Capacitors.kicad_sym": "JLCPCB-Capacitors",
    "JLCPCB-Inductors.kicad_sym": "JLCPCB-Inductors",
    "JLCPCB-Diodes.kicad_sym": "JLCPCB-Diodes",
    "JLCPCB-Diode-Packages.kicad_sym": "JLCPCB-Diode-Packages",
    "JLCPCB-Transistors.kicad_sym": "JLCPCB-Transistors",
    "JLCPCB-Transistor-Packages.kicad_sym": "JLCPCB-Transistor-Packages",
    "JLCPCB-ICs.kicad_sym": "JLCPCB-ICs",
    "JLCPCB-MCUs.kicad_sym": "JLCPCB-MCUs",
    "JLCPCB-Power.kicad_sym": "JLCPCB-Power",
    "JLCPCB-Connectors_Buttons.kicad_sym": "JLCPCB-Connectors_Buttons",
    "JLCPCB-Crystals.kicad_sym": "JLCPCB-Crystals",
    "JLCPCB-Analog.kicad_sym": "JLCPCB-Analog",
    "JLCPCB-Interface.kicad_sym": "JLCPCB-Interface",
    "JLCPCB-Memory.kicad_sym": "JLCPCB-Memory",
    "JLCPCB-Optocouplers.kicad_sym": "JLCPCB-Optocouplers",
    "JLCPCB-Extended.kicad_sym": "JLCPCB-Extended",
    "JLCPCB-Manufacturing.kicad_sym": "JLCPCB-Manufacturing",
}


@dataclass(frozen=True)
class ResolvedPart:
    """A JLCPCB part resolved to KiCad library references."""

    lcsc: str
    symbol_lib: str        # e.g. "JLCPCB-Resistors" or "Device"
    symbol_name: str       # e.g. "R" or full symbol name
    symbol_ref: str        # e.g. "JLCPCB-Resistors:R_10k_0805"
    footprint_ref: str     # e.g. "Resistor_SMD:R_0805_2012Metric"
    footprint_path: Path | None  # absolute path to .kicad_mod file, if found
    source: str            # "cdfer", "builtin", or "generated"


def _find_cdfer_symbol_dir() -> Path | None:
    """Locate the CDFER JLCPCB symbol library directory."""
    for d in _CDFER_SYMBOL_DIRS:
        if d.is_dir():
            return d
    return None


def _find_cdfer_footprint_dir() -> Path | None:
    """Locate the CDFER JLCPCB footprint library directory."""
    for d in _CDFER_FOOTPRINT_DIRS:
        if d.is_dir():
            return d
    return None


def _category_to_cdfer_file(category: str) -> str | None:
    """Map a JLCPCB category to the CDFER symbol library file name."""
    cat_lower = category.lower()
    mapping: dict[str, str] = {
        "resistors": "JLCPCB-Resistors",
        "capacitors": "JLCPCB-Capacitors",
        "inductors": "JLCPCB-Inductors",
        "diodes": "JLCPCB-Diodes",
        "transistors": "JLCPCB-Transistors",
        "ics": "JLCPCB-ICs",
        "embedded processors & controllers": "JLCPCB-MCUs",
        "power management ics": "JLCPCB-Power",
        "connectors": "JLCPCB-Connectors_Buttons",
        "crystals": "JLCPCB-Crystals",
        "analog ics": "JLCPCB-Analog",
        "interface ics": "JLCPCB-Interface",
        "memory": "JLCPCB-Memory",
        "optocouplers & leds & infrared": "JLCPCB-Optocouplers",
    }
    for key, lib in mapping.items():
        if key in cat_lower:
            return lib
    return None


def _is_passive(category: str) -> bool:
    """Return True if the category is a passive component."""
    return category.lower() in {"resistors", "capacitors", "inductors"}


def _footprint_for_passive(category: str, package: str) -> str | None:
    """Return the built-in KiCad footprint for a passive component."""
    cat_lower = category.lower()
    if cat_lower == "capacitors":
        return _CAP_PACKAGE_TO_FOOTPRINT.get(package)
    if cat_lower in {"resistors", "inductors"}:
        return _PACKAGE_TO_BUILTIN_FOOTPRINT.get(package)
    return None


def resolve_symbol(part: JLCPCBPart) -> str | None:
    """Find a KiCad symbol reference for a JLCPCB part.

    Searches CDFER library first, then built-in KiCad libraries.

    Args:
        part: The JLCPCB part to resolve.

    Returns:
        A KiCad symbol reference string (e.g. ``"Device:R"``), or ``None``
        if no matching symbol can be found.
    """
    # Try CDFER library.
    cdfer_lib = _category_to_cdfer_file(part.category)
    if cdfer_lib:
        cdfer_dir = _find_cdfer_symbol_dir()
        if cdfer_dir:
            sym_file = cdfer_dir / f"{cdfer_lib}.kicad_sym"
            if sym_file.is_file():
                # CDFER symbols are named by LCSC number.
                return f"{cdfer_lib}:{part.lcsc}"

    # Fall back to built-in KiCad symbols for passives.
    builtin = _CATEGORY_TO_BUILTIN_SYMBOL.get(part.category)
    if builtin:
        return builtin

    return None


def resolve_footprint(part: JLCPCBPart) -> str | None:
    """Find a KiCad footprint reference for a JLCPCB part.

    Args:
        part: The JLCPCB part to resolve.

    Returns:
        A KiCad footprint reference string, or ``None`` if not found.
    """
    # Try CDFER footprint library.
    cdfer_fp_dir = _find_cdfer_footprint_dir()
    if cdfer_fp_dir:
        # CDFER footprints are named by LCSC number.
        fp_file = cdfer_fp_dir / f"{part.lcsc}.kicad_mod"
        if fp_file.is_file():
            return f"JLCPCB:{part.lcsc}"

    # Fall back to built-in KiCad footprints for passives.
    if _is_passive(part.category):
        fp = _footprint_for_passive(part.category, part.package)
        if fp:
            return fp

    return None


def resolve_part(part: JLCPCBPart) -> ResolvedPart | None:
    """Fully resolve a JLCPCB part to KiCad library references.

    Args:
        part: The JLCPCB part to resolve.

    Returns:
        A :class:`ResolvedPart` with symbol and footprint references,
        or ``None`` if neither can be resolved.
    """
    symbol = resolve_symbol(part)
    footprint = resolve_footprint(part)

    if not symbol and not footprint:
        return None

    # Determine source.
    source = "generated"
    if symbol and "JLCPCB" in symbol:
        source = "cdfer"
    elif symbol:
        source = "builtin"

    # Determine footprint path.
    fp_path: Path | None = None
    if footprint and footprint.startswith("JLCPCB:"):
        cdfer_fp_dir = _find_cdfer_footprint_dir()
        if cdfer_fp_dir:
            fp_file = cdfer_fp_dir / f"{part.lcsc}.kicad_mod"
            if fp_file.is_file():
                fp_path = fp_file

    symbol_ref = symbol or ""
    symbol_lib = symbol_ref.split(":")[0] if ":" in symbol_ref else ""
    symbol_name = symbol_ref.split(":")[1] if ":" in symbol_ref else symbol_ref

    return ResolvedPart(
        lcsc=part.lcsc,
        symbol_lib=symbol_lib,
        symbol_name=symbol_name,
        symbol_ref=symbol_ref,
        footprint_ref=footprint or "",
        footprint_path=fp_path,
        source=source,
    )


def list_available_libraries() -> dict[str, Path]:
    """Discover installed KiCad symbol and footprint libraries.

    Returns:
        A dict mapping library nickname to directory path.
    """
    libs: dict[str, Path] = {}

    # CDFER libraries.
    cdfer_dir = _find_cdfer_symbol_dir()
    if cdfer_dir:
        for fname, nickname in _CDFER_SYMBOL_FILES.items():
            fpath = cdfer_dir / fname
            if fpath.is_file():
                libs[nickname] = fpath

    cdfer_fp_dir = _find_cdfer_footprint_dir()
    if cdfer_fp_dir and cdfer_fp_dir.is_dir():
        libs["JLCPCB"] = cdfer_fp_dir

    # KiCad built-in symbol libraries.
    for builtin_dir in _KICAD_BUILTIN_SYMBOLS:
        if builtin_dir.is_dir():
            for sym_file in sorted(builtin_dir.glob("*.kicad_sym")):
                nickname = sym_file.stem
                libs[nickname] = sym_file

    return libs
