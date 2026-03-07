"""Apply LCSC part replacements to PCBDesign footprints."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.production.parts_validator import PartsValidationReport


def replacement_map_from_report(
    report: PartsValidationReport,
) -> dict[str, str]:
    """Extract {old_lcsc: new_lcsc} map from validation report.

    Only includes parts with status="replaced" and a non-None replacement_lcsc.
    Key is the original LCSC (or ref designator if original LCSC is empty).
    """
    replacements: dict[str, str] = {}
    for part in report.parts:
        if part.status == "replaced" and part.replacement_lcsc is not None:
            if part.lcsc:
                replacements[part.lcsc] = part.replacement_lcsc
            else:
                # No original LCSC — map by ref designators
                for ref in part.ref_designators:
                    replacements[ref] = part.replacement_lcsc
    return replacements


def apply_replacements(
    pcb: PCBDesign,
    replacements: dict[str, str],
) -> PCBDesign:
    """Return a new PCBDesign with LCSC numbers updated per replacements.

    ``replacements`` maps either old LCSC -> new LCSC, or ref -> new LCSC.
    """
    if not replacements:
        return pcb

    new_fps: list[object] = []
    changed = False
    for fp in pcb.footprints:
        new_lcsc: str | None = None
        # Check by current LCSC number
        if fp.lcsc and fp.lcsc in replacements:
            new_lcsc = replacements[fp.lcsc]
        # Check by ref designator
        elif fp.ref in replacements:
            new_lcsc = replacements[fp.ref]

        if new_lcsc is not None and new_lcsc != fp.lcsc:
            new_fps.append(replace(fp, lcsc=new_lcsc))
            changed = True
        else:
            new_fps.append(fp)

    if not changed:
        return pcb

    return replace(pcb, footprints=tuple(new_fps))  # type: ignore[arg-type]
