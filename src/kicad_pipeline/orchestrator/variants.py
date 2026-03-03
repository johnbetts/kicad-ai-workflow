"""Variant system: fork requirements with different package strategies.

Given base :class:`~kicad_pipeline.models.requirements.ProjectRequirements`
and a :class:`~kicad_pipeline.orchestrator.models.PackageStrategy`, this
module produces variant-specific requirements with remapped footprints
and updated LCSC part numbers.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from kicad_pipeline.models.requirements import Component, ProjectRequirements

if TYPE_CHECKING:
    from kicad_pipeline.orchestrator.models import PackageStrategy
    from kicad_pipeline.requirements.component_db import ComponentDB

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Footprint family detection
# ---------------------------------------------------------------------------

# Regex patterns for common passive/LED footprint families
_RESISTOR_RE = re.compile(r"^R[_:]")
_CAPACITOR_RE = re.compile(r"^C[_:]")
_LED_RE = re.compile(r"^LED[_:]")

# Map from family to (strategy attribute, KiCad footprint prefix)
_FAMILY_INFO: dict[str, tuple[str, str]] = {
    "R": ("resistor_package", "R"),
    "C": ("capacitor_package", "C"),
    "LED": ("led_package", "LED"),
}

# Package-to-KiCad footprint mappings for SMD sizes
_SMD_FOOTPRINT_MAP: dict[str, dict[str, str]] = {
    "R": {
        "0402": "R_0402",
        "0603": "R_0603",
        "0805": "R_0805",
        "1206": "R_1206",
        "Axial_DIN0207": "R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm",
    },
    "C": {
        "0402": "C_0402",
        "0603": "C_0603",
        "0805": "C_0805",
        "1206": "C_1206",
        "C_Disc_D5.0mm": "C_Disc_D5.0mm_W2.5mm_P2.50mm",
    },
    "LED": {
        "0402": "LED_0402",
        "0603": "LED_0603",
        "0805": "LED_0805",
        "LED_D3.0mm": "LED_D3.0mm",
        "LED_D5.0mm": "LED_D5.0mm",
    },
}


def detect_footprint_family(footprint: str) -> str | None:
    """Determine the component family from a footprint string.

    Recognises resistors (``R_*``), capacitors (``C_*``), and LEDs
    (``LED_*``).

    Args:
        footprint: KiCad footprint string (e.g. ``"R_0805"``).

    Returns:
        Family key (``"R"``, ``"C"``, ``"LED"``), or ``None`` if
        the footprint doesn't match a remappable family.
    """
    if _RESISTOR_RE.match(footprint):
        return "R"
    if _CAPACITOR_RE.match(footprint):
        return "C"
    if _LED_RE.match(footprint):
        return "LED"
    return None


def remap_footprint(
    footprint: str,
    family: str,
    strategy: PackageStrategy,
) -> str:
    """Remap a footprint string to the variant's target package.

    Args:
        footprint: Original footprint (e.g. ``"R_0805"``).
        family: Component family (``"R"``, ``"C"``, ``"LED"``).
        strategy: The variant's package strategy.

    Returns:
        Remapped footprint string, or the original if no mapping exists.
    """
    attr_name, _prefix = _FAMILY_INFO[family]
    target_package: str = getattr(strategy, attr_name)

    family_map = _SMD_FOOTPRINT_MAP.get(family, {})
    mapped = family_map.get(target_package)
    if mapped is not None:
        return mapped

    # Fallback: construct a simple footprint string
    return f"{_prefix}_{target_package}"


def _remap_component(
    comp: Component,
    strategy: PackageStrategy,
    db: ComponentDB | None = None,
) -> Component:
    """Remap a single component's footprint and optionally update LCSC.

    Args:
        comp: The original component.
        strategy: Package strategy for the target variant.
        db: Optional :class:`ComponentDB` for LCSC part lookup.

    Returns:
        A new :class:`Component` with updated footprint (and LCSC if
        a matching part was found in *db*).
    """
    family = detect_footprint_family(comp.footprint)
    if family is None:
        return comp  # Not a remappable component

    new_footprint = remap_footprint(comp.footprint, family, strategy)
    if new_footprint == comp.footprint:
        return comp  # Same package, no change

    new_lcsc = comp.lcsc
    if db is not None:
        attr_name = _FAMILY_INFO[family][0]
        target_package: str = getattr(strategy, attr_name)
        new_lcsc = _lookup_lcsc(db, comp, family, target_package)

    log.debug(
        "Remapped %s footprint: %s -> %s (lcsc: %s -> %s)",
        comp.ref,
        comp.footprint,
        new_footprint,
        comp.lcsc,
        new_lcsc,
    )

    return Component(
        ref=comp.ref,
        value=comp.value,
        footprint=new_footprint,
        lcsc=new_lcsc,
        description=comp.description,
        datasheet=comp.datasheet,
        pins=comp.pins,
    )


def _lookup_lcsc(
    db: ComponentDB,
    comp: Component,
    family: str,
    target_package: str,
) -> str | None:
    """Try to find a matching LCSC part number for the new package.

    Args:
        db: Component database.
        comp: Original component (used for value matching).
        family: Component family key.
        target_package: Target package string.

    Returns:
        LCSC part number, or the original ``comp.lcsc`` if no match.
    """
    from kicad_pipeline.requirements.component_db import (
        _parse_capacitance_uf,
        _parse_resistance_ohms,
    )

    if family == "R":
        ohms = _parse_resistance_ohms(comp.value)
        if ohms is not None:
            part = db.find_resistor(ohms, package=target_package)
            if part is not None:
                return part.lcsc
    elif family == "C":
        uf = _parse_capacitance_uf(comp.value)
        if uf is not None:
            part = db.find_capacitor(uf, package=target_package)
            if part is not None:
                return part.lcsc
    elif family == "LED":
        # Try to extract color from value
        color = comp.value.lower() if comp.value else "green"
        part = db.find_led(color=color, package=target_package)
        if part is not None:
            return part.lcsc

    return comp.lcsc


def fork_requirements_for_variant(
    base: ProjectRequirements,
    strategy: PackageStrategy,
    db: ComponentDB | None = None,
) -> ProjectRequirements:
    """Create variant-specific requirements by remapping footprints.

    Walks all components in *base*, detects footprint families (R, C, LED),
    and remaps to the package sizes defined in *strategy*.  ICs, connectors,
    and other non-passive components are left untouched.

    If *db* is provided, LCSC part numbers are updated to match the new
    package sizes.

    Args:
        base: The base (package-agnostic) requirements.
        strategy: Package strategy for the variant.
        db: Optional component database for LCSC lookup.

    Returns:
        A new :class:`ProjectRequirements` with remapped footprints.
    """
    remapped_components = tuple(
        _remap_component(c, strategy, db) for c in base.components
    )

    changed = sum(
        1 for orig, new in zip(base.components, remapped_components, strict=False)
        if orig.footprint != new.footprint
    )
    log.info(
        "Forked variant with strategy %r: %d/%d components remapped",
        strategy.name,
        changed,
        len(base.components),
    )

    return ProjectRequirements(
        project=base.project,
        features=base.features,
        components=remapped_components,
        nets=base.nets,
        pin_map=base.pin_map,
        power_budget=base.power_budget,
        mechanical=base.mechanical,
        recommendations=base.recommendations,
    )
