"""Interpret DRC violations into actionable suggestions.

Categorises violations from :mod:`kicad_drc` into auto-fixable and
manual-fix groups, providing human-readable explanations and fix actions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    from kicad_pipeline.validation.kicad_drc import DRCReport, DRCViolation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DRCSuggestion:
    """An actionable suggestion derived from a DRC violation."""

    violation_type: str
    severity: str
    auto_fixable: bool
    description: str
    fix_action: str | None = None
    affected_refs: tuple[str, ...] = ()


# Violation types that can be auto-fixed by S-expression mutation.
_AUTO_FIXABLE: dict[str, str] = {
    "silk_over_copper": "Move silkscreen text away from copper pads",
    "silk_overlap": "Reposition overlapping silkscreen labels",
    "silk_edge_clearance": "Move silkscreen text away from board edge",
    "courtyards_overlap": "Adjust component spacing to resolve courtyard overlap",
    "courtyard_overlap": "Adjust component spacing to resolve courtyard overlap",
    "missing_courtyard": "Add courtyard outline to footprint",
}

# Violation types requiring manual intervention.
_MANUAL_FIX: dict[str, str] = {
    "clearance": "Increase spacing between copper objects — re-route affected traces",
    "copper_edge_clearance": "Move tracks/pours away from board edge",
    "track_width": "Widen trace to meet minimum width requirement",
    "via_diameter": "Increase via diameter to meet design rules",
    "via_drill_too_small": "Increase via drill size",
    "annular_width": "Increase via annular ring width",
    "hole_to_hole": "Increase spacing between drill holes",
    "shorting": "Remove short circuit between different nets",
    "unconnected_items": "Route missing connections",
    "solder_mask_bridge": "Increase spacing between solder mask openings",
    "starved_thermal": "Adjust thermal relief connection to zone",
}


def interpret_violations(report: DRCReport) -> list[DRCSuggestion]:
    """Convert DRC violations into actionable suggestions.

    Args:
        report: Parsed DRC report from :func:`kicad_drc.run_drc`.

    Returns:
        List of suggestions, auto-fixable first.
    """
    suggestions: list[DRCSuggestion] = []

    for violation in report.violations:
        if violation.excluded:
            continue
        suggestion = _interpret_single(violation)
        suggestions.append(suggestion)

    for unconnected in report.unconnected:
        if unconnected.excluded:
            continue
        refs = tuple(item.ref for item in unconnected.items if item.ref)
        suggestions.append(
            DRCSuggestion(
                violation_type="unconnected_items",
                severity="error",
                auto_fixable=False,
                description=unconnected.description or "Unconnected net items",
                fix_action="Route the missing connection between pads",
                affected_refs=refs,
            )
        )

    # Sort: auto-fixable first, then by severity.
    severity_order = {"error": 0, "warning": 1, "info": 2}
    suggestions.sort(
        key=lambda s: (
            not s.auto_fixable,
            severity_order.get(s.severity, 3),
        )
    )

    return suggestions


def _interpret_single(violation: DRCViolation) -> DRCSuggestion:
    """Interpret a single DRC violation."""
    vtype = violation.type
    refs = tuple(item.ref for item in violation.items if item.ref)

    # Check auto-fixable.
    if vtype in _AUTO_FIXABLE:
        return DRCSuggestion(
            violation_type=vtype,
            severity=violation.severity,
            auto_fixable=True,
            description=violation.description,
            fix_action=_AUTO_FIXABLE[vtype],
            affected_refs=refs,
        )

    # Check known manual-fix types.
    if vtype in _MANUAL_FIX:
        return DRCSuggestion(
            violation_type=vtype,
            severity=violation.severity,
            auto_fixable=False,
            description=violation.description,
            fix_action=_MANUAL_FIX[vtype],
            affected_refs=refs,
        )

    # Unknown violation type — manual fix.
    return DRCSuggestion(
        violation_type=vtype,
        severity=violation.severity,
        auto_fixable=False,
        description=violation.description,
        fix_action=None,
        affected_refs=refs,
    )


def summarize_suggestions(suggestions: list[DRCSuggestion]) -> str:
    """Generate a human-readable summary of DRC suggestions.

    Args:
        suggestions: List of DRC suggestions.

    Returns:
        Multi-line summary string.
    """
    if not suggestions:
        return "No DRC issues found — board is clean."

    auto = [s for s in suggestions if s.auto_fixable]
    manual = [s for s in suggestions if not s.auto_fixable]

    lines: list[str] = []
    lines.append(f"DRC: {len(suggestions)} issues found")

    if auto:
        lines.append(f"\nAuto-fixable ({len(auto)}):")
        for s in auto:
            lines.append(f"  [{s.severity}] {s.violation_type}: {s.description}")
            if s.fix_action:
                lines.append(f"    → {s.fix_action}")

    if manual:
        lines.append(f"\nManual fix required ({len(manual)}):")
        for s in manual:
            lines.append(f"  [{s.severity}] {s.violation_type}: {s.description}")
            if s.fix_action:
                lines.append(f"    → {s.fix_action}")

    return "\n".join(lines)
