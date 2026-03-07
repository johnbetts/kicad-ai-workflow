"""Suggest and validate JLCPCB parts for project requirements.

Maps components from a :class:`ProjectRequirements` to concrete JLCPCB
parts, with stock checking and compatibility validation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import Component, ProjectRequirements
    from kicad_pipeline.parts.jlcpcb_db import JLCPCBPart, JLCPCBPartsDB

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartSuggestion:
    """A suggested part with match quality metadata."""

    component_ref: str
    component_value: str
    candidates: tuple[JLCPCBPart, ...]
    preferred: JLCPCBPart | None
    match_quality: str  # "exact", "close", "generic", "none"
    notes: str = ""


@dataclass(frozen=True)
class ValidationIssue:
    """An issue found during parts selection validation."""

    ref: str
    severity: str  # "error", "warning", "info"
    message: str


def _extract_package(footprint: str) -> str:
    """Extract package size from a footprint string.

    Examples:
        ``"R_0805"`` → ``"0805"``
        ``"C_0603_1608Metric"`` → ``"0603"``
        ``"SOT-23"`` → ``"SOT-23"``
    """
    # Match 4-digit SMD packages (may follow underscore).
    pkg_pat = r"(?:^|[_\-\s])(0201|0402|0603|0805|1206|1210|2010|2512)(?:[_\-\s]|$)"
    m = re.search(pkg_pat, footprint)
    if m:
        return m.group(1)
    return footprint


def _is_passive(ref: str) -> bool:
    """Return True if the reference designator indicates a passive."""
    return ref[0] in {"R", "C", "L"} if ref else False


def _component_category(ref: str) -> str | None:
    """Map ref designator prefix to JLCPCB category."""
    prefix = ref.rstrip("0123456789") if ref else ""
    mapping: dict[str, str] = {
        "R": "Resistors",
        "C": "Capacitors",
        "L": "Inductors",
        "D": "Diodes",
        "Q": "Transistors",
        "U": "ICs",
        "LED": "LEDs",
    }
    return mapping.get(prefix)


def suggest_parts_for_component(
    component: Component,
    db: JLCPCBPartsDB,
    prefer_basic: bool = True,
) -> PartSuggestion:
    """Suggest JLCPCB parts for a single component.

    Args:
        component: The component to find parts for.
        db: Open JLCPCB parts database.
        prefer_basic: Prefer JLCPCB basic parts when available.

    Returns:
        A :class:`PartSuggestion` with candidates and a preferred pick.
    """
    # If component already has an LCSC number, look it up directly.
    if component.lcsc:
        direct = db.get_part(component.lcsc)
        if direct:
            return PartSuggestion(
                component_ref=component.ref,
                component_value=component.value,
                candidates=(direct,),
                preferred=direct,
                match_quality="exact",
                notes=f"Matched by LCSC number {component.lcsc}",
            )

    package = _extract_package(component.footprint)
    candidates: list[JLCPCBPart] = []

    # Category-specific search.
    if component.ref.startswith("R"):
        candidates = db.find_resistor(
            component.value, package=package, basic_only=prefer_basic
        )
    elif component.ref.startswith("C"):
        candidates = db.find_capacitor(
            component.value, package=package, basic_only=prefer_basic
        )
    elif component.ref.startswith(("U", "IC")):
        candidates = db.find_ic(component.value, basic_only=False)
    else:
        # Generic search.
        query = f"{component.value} {package}"
        candidates = db.search_parts(
            query, basic_only=prefer_basic, in_stock=True
        )

    # Determine match quality.
    if not candidates:
        return PartSuggestion(
            component_ref=component.ref,
            component_value=component.value,
            candidates=(),
            preferred=None,
            match_quality="none",
            notes="No matching parts found in JLCPCB database",
        )

    # Prefer basic parts, then highest stock.
    sorted_candidates = sorted(
        candidates,
        key=lambda p: (p.basic, p.stock),
        reverse=True,
    )
    preferred = sorted_candidates[0]

    quality = "close"
    if preferred.basic:
        quality = "exact" if _is_passive(component.ref) else "close"

    return PartSuggestion(
        component_ref=component.ref,
        component_value=component.value,
        candidates=tuple(sorted_candidates[:10]),
        preferred=preferred,
        match_quality=quality,
    )


def suggest_parts(
    requirements: ProjectRequirements,
    db: JLCPCBPartsDB,
    prefer_basic: bool = True,
) -> dict[str, PartSuggestion]:
    """Suggest JLCPCB parts for all components in requirements.

    Args:
        requirements: Project requirements with components.
        db: Open JLCPCB parts database.
        prefer_basic: Prefer JLCPCB basic parts when available.

    Returns:
        Dict mapping component ref to :class:`PartSuggestion`.
    """
    suggestions: dict[str, PartSuggestion] = {}
    for component in requirements.components:
        suggestions[component.ref] = suggest_parts_for_component(
            component, db, prefer_basic=prefer_basic
        )
    return suggestions


def validate_parts_selection(
    parts: dict[str, JLCPCBPart],
) -> list[ValidationIssue]:
    """Validate a parts selection for manufacturing readiness.

    Args:
        parts: Dict mapping component ref to selected JLCPCB part.

    Returns:
        List of validation issues found.
    """
    issues: list[ValidationIssue] = []

    for ref, part in parts.items():
        # Stock check.
        if not part.is_in_stock:
            issues.append(
                ValidationIssue(
                    ref=ref,
                    severity="error",
                    message=f"{part.lcsc} ({part.mfr_part}) is out of stock",
                )
            )

        # Extended part warning.
        if not part.basic:
            issues.append(
                ValidationIssue(
                    ref=ref,
                    severity="warning",
                    message=(
                        f"{part.lcsc} is an extended part — "
                        f"adds $3 setup fee per unique extended part"
                    ),
                )
            )

        # Low stock warning.
        if part.is_in_stock and part.stock < 100:
            issues.append(
                ValidationIssue(
                    ref=ref,
                    severity="warning",
                    message=f"{part.lcsc} has low stock ({part.stock} remaining)",
                )
            )

    return issues
