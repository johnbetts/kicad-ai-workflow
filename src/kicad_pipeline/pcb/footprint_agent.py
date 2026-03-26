"""Unified footprint facade — single entry point for all footprint operations.

Delegates to existing modules without duplicating logic:

- :mod:`~kicad_pipeline.pcb.footprints` — parametric generators, 3D model validation
- :mod:`~kicad_pipeline.pcb.footprint_library` — ``.pretty`` library generation
- :mod:`~kicad_pipeline.pcb.pin_map` — pad-side classification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import FootprintError
from kicad_pipeline.pcb.footprint_library import (
    build_footprint_library,
    remap_footprint_lib_ids,
    write_fp_lib_table,
)
from kicad_pipeline.pcb.footprints import (
    footprint_for_component,
    validate_3d_model_orientation,
)
from kicad_pipeline.pcb.keepout_builder import RF_KEYWORDS as _RF_KEYWORDS
from kicad_pipeline.pcb.pin_map import FootprintPinMap, compute_pin_map

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.pcb import Footprint, Keepout, PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ModelStatus(Enum):
    """3D model presence / validity status."""

    PRESENT = "present"
    MISSING = "missing"
    ROTATION_SUSPECT = "rotation_suspect"


class PropertyStatus(Enum):
    """Status of a single footprint property."""

    PRESENT = "present"
    MISSING = "missing"
    EMPTY = "empty"


class VerificationStatus(Enum):
    """Template verification status for a footprint."""

    VERIFIED = "verified"
    GEOMETRY_ONLY = "geometry_only"
    UNVERIFIED = "unverified"
    NO_TEMPLATE = "no_template"


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelReport:
    """3D model status for a single footprint."""

    ref: str
    lib_id: str
    status: ModelStatus
    model_path: str
    rotation_warnings: tuple[str, ...]


@dataclass(frozen=True)
class PropertyReport:
    """Property completeness for a single footprint."""

    ref: str
    lcsc: PropertyStatus
    datasheet: PropertyStatus
    description: PropertyStatus
    mpn: PropertyStatus
    manufacturer: PropertyStatus


@dataclass(frozen=True)
class KeepoutReport:
    """RF keepout zone audit for a single footprint."""

    ref: str
    requires_keepout: bool
    keepout_present: bool
    keepout_area_mm2: float


@dataclass(frozen=True)
class ComponentAuditEntry:
    """Full audit entry for a single component."""

    ref: str
    lib_id: str
    model: ModelReport
    properties: PropertyReport
    keepout: KeepoutReport
    pin_count: int


@dataclass(frozen=True)
class FootprintAuditReport:
    """Board-wide footprint audit results."""

    entries: tuple[ComponentAuditEntry, ...]
    total_components: int
    models_present: int
    models_missing: int
    properties_complete: int
    properties_incomplete: int
    keepouts_required: int
    keepouts_present: int
    issues: tuple[str, ...]

    def to_report(self) -> str:
        """Render a human-readable text summary."""
        lines: list[str] = [
            "=== Footprint Audit Report ===",
            f"Total components: {self.total_components}",
            f"3D models: {self.models_present} present,"
            f" {self.models_missing} missing",
            f"Properties: {self.properties_complete} complete,"
            f" {self.properties_incomplete} incomplete",
            f"Keepouts: {self.keepouts_present}/{self.keepouts_required}"
            " required present",
        ]
        if self.issues:
            lines.append("")
            lines.append(f"Issues ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        else:
            lines.append("")
            lines.append("No issues found.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers (new logic — small)
# ---------------------------------------------------------------------------


def _check_model(fp: Footprint) -> ModelReport:
    """Check 3D model presence and orientation for a footprint."""
    warnings = validate_3d_model_orientation(fp)
    model_path = fp.models[0].path if fp.models else ""

    if not fp.models:
        status = ModelStatus.MISSING
    elif warnings:
        status = ModelStatus.ROTATION_SUSPECT
    else:
        status = ModelStatus.PRESENT

    return ModelReport(
        ref=fp.ref,
        lib_id=fp.lib_id,
        status=status,
        model_path=model_path,
        rotation_warnings=warnings,
    )


def _prop_status(value: str | None) -> PropertyStatus:
    """Classify a single property value."""
    if value is None:
        return PropertyStatus.MISSING
    if not value.strip():
        return PropertyStatus.EMPTY
    return PropertyStatus.PRESENT


def _check_properties(fp: Footprint) -> PropertyReport:
    """Check property completeness for a footprint."""
    return PropertyReport(
        ref=fp.ref,
        lcsc=_prop_status(fp.lcsc),
        datasheet=_prop_status(fp.datasheet),
        description=_prop_status(fp.description),
        mpn=_prop_status(fp.mpn),
        manufacturer=_prop_status(fp.manufacturer),
    )


def _is_rf_component(lib_id: str) -> bool:
    """Return True if the lib_id matches an RF/wireless keyword."""
    lower = lib_id.lower()
    return any(kw in lower for kw in _RF_KEYWORDS)


def _polygon_area(polygon: tuple[object, ...]) -> float:
    """Compute area of a polygon using the shoelace formula.

    Each element must have ``.x`` and ``.y`` attributes.
    """
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        area += getattr(p1, "x", 0.0) * getattr(p2, "y", 0.0)
        area -= getattr(p2, "x", 0.0) * getattr(p1, "y", 0.0)
    return abs(area) / 2.0


def _find_keepout_for_ref(
    ref: str, fp: Footprint, keepouts: tuple[Keepout, ...]
) -> tuple[bool, float]:
    """Find a keepout zone overlapping the footprint position.

    Returns ``(found, area_mm2)``.
    """
    fx, fy = fp.position.x, fp.position.y
    for ko in keepouts:
        # Simple containment check: is the footprint center inside the keepout?
        pts = ko.polygon
        if not pts:
            continue
        # Point-in-polygon (ray casting)
        inside = False
        n = len(pts)
        j = n - 1
        for i in range(n):
            yi, yj = pts[i].y, pts[j].y
            xi, xj = pts[i].x, pts[j].x
            if ((yi > fy) != (yj > fy)) and (
                fx < (xj - xi) * (fy - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i
        if inside:
            return True, _polygon_area(pts)
    return False, 0.0


def _check_rf_keepout(fp: Footprint, keepouts: tuple[Keepout, ...]) -> KeepoutReport:
    """Check whether an RF component has an appropriate keepout zone."""
    needs = _is_rf_component(fp.lib_id)
    if not needs:
        return KeepoutReport(
            ref=fp.ref,
            requires_keepout=False,
            keepout_present=False,
            keepout_area_mm2=0.0,
        )

    found, area = _find_keepout_for_ref(fp.ref, fp, keepouts)
    return KeepoutReport(
        ref=fp.ref,
        requires_keepout=True,
        keepout_present=found,
        keepout_area_mm2=area,
    )


def _is_property_complete(pr: PropertyReport) -> bool:
    """Return True if all key properties are present (non-missing, non-empty)."""
    return all(
        s == PropertyStatus.PRESENT
        for s in (pr.lcsc, pr.datasheet, pr.description)
    )


# ---------------------------------------------------------------------------
# Public API — delegation only
# ---------------------------------------------------------------------------


def create_footprint(
    ref: str,
    value: str,
    fp_id: str,
    lcsc: str | None = None,
    layer: str = "F.Cu",
    datasheet: str | None = None,
    description: str | None = None,
    mpn: str | None = None,
    manufacturer: str | None = None,
) -> Footprint:
    """Create a footprint with full property population.

    Delegates generation to :func:`footprint_for_component` then enriches
    with datasheet, description, MPN, and manufacturer fields.

    Args:
        ref: Reference designator.
        value: Component value string.
        fp_id: Footprint identifier string.
        lcsc: Optional LCSC part number.
        layer: Copper layer (default ``"F.Cu"``).
        datasheet: Datasheet URL or path.
        description: Component description.
        mpn: Manufacturer part number.
        manufacturer: Manufacturer name.

    Returns:
        Fully constructed :class:`Footprint` with all properties populated.

    Raises:
        FootprintError: If footprint generation fails.
    """
    try:
        fp = footprint_for_component(ref, value, fp_id, lcsc=lcsc, layer=layer)
    except Exception as exc:
        raise FootprintError(
            f"Failed to create footprint for {ref} ({fp_id}): {exc}"
        ) from exc

    # Frozen dataclass — use __class__ constructor to add extra fields
    from dataclasses import asdict

    d = asdict(fp)
    if datasheet is not None:
        d["datasheet"] = datasheet
    if description is not None:
        d["description"] = description
    if mpn is not None:
        d["mpn"] = mpn
    if manufacturer is not None:
        d["manufacturer"] = manufacturer
    # Re-import to reconstruct (avoid circular at module level)
    from kicad_pipeline.models.pcb import Footprint as _Footprint

    return _Footprint(**d)


def audit_board(
    pcb: PCBDesign,
    requirements: ProjectRequirements | None = None,
) -> FootprintAuditReport:
    """Run a comprehensive footprint audit on a PCB design.

    Checks every footprint for:
    - 3D model presence and rotation correctness
    - Property completeness (LCSC, datasheet, description, MPN, manufacturer)
    - RF keepout zone presence for wireless modules

    Args:
        pcb: The PCB design to audit.
        requirements: Optional requirements (reserved for future enrichment).

    Returns:
        A :class:`FootprintAuditReport` with all findings.
    """
    entries: list[ComponentAuditEntry] = []
    issues: list[str] = []
    counts = {"models_present": 0, "models_missing": 0,
              "props_complete": 0, "props_incomplete": 0,
              "keepouts_required": 0, "keepouts_present": 0}

    for fp in pcb.footprints:
        model_report = _check_model(fp)
        prop_report = _check_properties(fp)
        keepout_report = _check_rf_keepout(fp, pcb.keepouts)
        _tally_fp_reports(fp, model_report, prop_report, keepout_report, counts, issues)
        entries.append(ComponentAuditEntry(
            ref=fp.ref,
            lib_id=fp.lib_id,
            model=model_report,
            properties=prop_report,
            keepout=keepout_report,
            pin_count=len(fp.pads),
        ))

    return FootprintAuditReport(
        entries=tuple(entries),
        total_components=len(pcb.footprints),
        models_present=counts["models_present"],
        models_missing=counts["models_missing"],
        properties_complete=counts["props_complete"],
        properties_incomplete=counts["props_incomplete"],
        keepouts_required=counts["keepouts_required"],
        keepouts_present=counts["keepouts_present"],
        issues=tuple(issues),
    )


def _tally_fp_reports(
    fp: Footprint,
    model_report: ModelReport,
    prop_report: PropertyReport,
    keepout_report: KeepoutReport,
    counts: dict[str, int],
    issues: list[str],
) -> None:
    """Update *counts* and *issues* in place for a single footprint's reports."""
    if model_report.status == ModelStatus.MISSING:
        counts["models_missing"] += 1
        issues.append(f"{fp.ref}: missing 3D model")
    else:
        counts["models_present"] += 1
    for w in model_report.rotation_warnings:
        issues.append(w)

    if _is_property_complete(prop_report):
        counts["props_complete"] += 1
    else:
        counts["props_incomplete"] += 1
        missing_props = [
            name for name, status in [
                ("LCSC", prop_report.lcsc),
                ("datasheet", prop_report.datasheet),
                ("description", prop_report.description),
            ]
            if status != PropertyStatus.PRESENT
        ]
        if missing_props:
            issues.append(f"{fp.ref}: missing properties: {', '.join(missing_props)}")

    if keepout_report.requires_keepout:
        counts["keepouts_required"] += 1
        if keepout_report.keepout_present:
            counts["keepouts_present"] += 1
        else:
            issues.append(f"{fp.ref}: RF component missing keepout zone")


def build_library(
    requirements: ProjectRequirements,
    project_dir: Path,
    project_name: str,
) -> Path:
    """Generate a project-local footprint library and fp-lib-table.

    Combines :func:`build_footprint_library` and :func:`write_fp_lib_table`
    into a single call.

    Args:
        requirements: Project requirements with all components.
        project_dir: Path to the project root directory.
        project_name: Project name (used as library name).

    Returns:
        Path to the generated ``.pretty`` directory.
    """
    pretty_dir = build_footprint_library(requirements, project_dir, project_name)
    write_fp_lib_table(project_dir, project_name)
    _log.info("Built library and fp-lib-table at %s", project_dir)
    return pretty_dir


def get_pin_map(fp: Footprint, rotation: float = 0.0) -> FootprintPinMap:
    """Classify footprint pads by cardinal side.

    Delegates to :func:`compute_pin_map`.

    Args:
        fp: Footprint to analyze.
        rotation: Board-level rotation in degrees (clockwise).

    Returns:
        Pad-side classification map.
    """
    return compute_pin_map(fp, rotation)


def remap_lib_ids(
    requirements: ProjectRequirements,
    project_name: str,
) -> dict[str, str]:
    """Build a mapping from current lib_ids to project-local lib_ids.

    Delegates to :func:`remap_footprint_lib_ids`.

    Args:
        requirements: Project requirements with all components.
        project_name: Project name used as the library prefix.

    Returns:
        Dict mapping old lib_id to new ``{project_name}:{footprint_name}``.
    """
    return remap_footprint_lib_ids(requirements, project_name)
