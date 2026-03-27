"""PCB integrity validation — programmatic gates that run after every build.

These checks verify that the generated PCB matches the requirements and that
all components have correct footprints, 3D models, and placement.  Each check
returns a list of issues; an empty list means pass.

Usage::

    from kicad_pipeline.validation.pcb_integrity import validate_pcb_integrity
    issues = validate_pcb_integrity(pcb, requirements)
    if issues:
        for issue in issues:
            print(f"  [{issue.severity}] {issue.message}")
        raise ValidationError(f"{len(issues)} integrity issues found")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntegrityIssue:
    """A single PCB integrity issue."""

    severity: str  # "critical", "major", "minor"
    category: str  # "sync", "3d_model", "pad_size", "keepout", "footprint"
    ref: str  # component ref or "board"
    message: str


# ---------------------------------------------------------------------------
# Package size reference — expected max pad dimension per package code
# ---------------------------------------------------------------------------

_PACKAGE_MAX_PAD_MM: dict[str, float] = {
    "0402": 0.8,
    "0603": 1.2,
    "0805": 1.8,
    "1206": 2.2,
    "1210": 2.5,
}


def _extract_package_code(footprint_id: str) -> str | None:
    """Extract package code (0402/0603/0805/etc.) from footprint_id."""
    upper = footprint_id.upper().replace("_", "").replace("-", "")
    for code in ("0402", "0603", "0805", "1206", "1210"):
        if code in upper:
            return code
    return None


# ---------------------------------------------------------------------------
# Check 1: Requirements ↔ PCB component sync
# ---------------------------------------------------------------------------

def _check_component_sync(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[IntegrityIssue]:
    """Verify every requirements component exists in PCB and vice versa."""
    issues: list[IntegrityIssue] = []

    req_refs = {c.ref for c in requirements.components}
    pcb_refs = {fp.ref for fp in pcb.footprints if fp.pads}

    # Components in requirements but missing from PCB
    missing_from_pcb = req_refs - pcb_refs
    for ref in sorted(missing_from_pcb):
        issues.append(IntegrityIssue(
            severity="critical", category="sync", ref=ref,
            message=f"{ref} in requirements but missing from PCB",
        ))

    # Components in PCB but not in requirements (excluding mounting holes)
    extra_in_pcb = pcb_refs - req_refs
    for ref in sorted(extra_in_pcb):
        if ref.startswith("H"):
            continue  # Mounting holes are auto-generated
        issues.append(IntegrityIssue(
            severity="major", category="sync", ref=ref,
            message=f"{ref} in PCB but not in requirements",
        ))

    return issues


# ---------------------------------------------------------------------------
# Check 2: Pad size matches requested package
# ---------------------------------------------------------------------------

def _check_pad_model_match(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[IntegrityIssue]:
    """Verify pad dimensions match the requested package size."""
    issues: list[IntegrityIssue] = []

    req_fps: dict[str, str] = {c.ref: c.footprint for c in requirements.components}

    for fp in pcb.footprints:
        if not fp.pads or fp.ref.startswith("H"):
            continue

        req_fp = req_fps.get(fp.ref)
        if req_fp is None:
            continue

        pkg = _extract_package_code(req_fp)
        if pkg is None:
            continue

        max_expected = _PACKAGE_MAX_PAD_MM.get(pkg)
        if max_expected is None:
            continue

        max_pad = max(max(p.size_x, p.size_y) for p in fp.pads)

        if max_pad > max_expected * 1.3:  # 30% tolerance
            issues.append(IntegrityIssue(
                severity="critical", category="pad_size", ref=fp.ref,
                message=(
                    f"{fp.ref}: pad {max_pad:.2f}mm but {req_fp} expects "
                    f"<{max_expected:.1f}mm — wrong package from JLCPCB cache?"
                ),
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 3: 3D model exists and size is plausible
# ---------------------------------------------------------------------------

def _check_3d_models(
    pcb: PCBDesign,
) -> list[IntegrityIssue]:
    """Verify all components have 3D models that exist on disk."""
    from kicad_pipeline.pcb.footprints import _step_file_exists

    issues: list[IntegrityIssue] = []

    for fp in pcb.footprints:
        if not fp.pads or fp.ref.startswith("H"):
            continue

        if not fp.models:
            issues.append(IntegrityIssue(
                severity="major", category="3d_model", ref=fp.ref,
                message=f"{fp.ref}: no 3D model assigned",
            ))
            continue

        model_path = fp.models[0].path
        model_name = model_path.split("/")[-1]

        if not _step_file_exists(model_path):
            issues.append(IntegrityIssue(
                severity="critical", category="3d_model", ref=fp.ref,
                message=f"{fp.ref}: STEP file missing: {model_name}",
            ))
            continue

        # Check model size vs pad size (0402 model on 0805 pads or vice versa)
        max_pad = max(max(p.size_x, p.size_y) for p in fp.pads)
        if "0402" in model_name and max_pad > 0.8:
            issues.append(IntegrityIssue(
                severity="critical", category="3d_model", ref=fp.ref,
                message=f"{fp.ref}: 0402 model on {max_pad:.1f}mm pads (0805?)",
            ))
        elif "0805" in model_name and max_pad < 0.7 and max_pad > 0.1:
            issues.append(IntegrityIssue(
                severity="critical", category="3d_model", ref=fp.ref,
                message=f"{fp.ref}: 0805 model on {max_pad:.1f}mm pads (0402?)",
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 4: Keepout zones near their target components
# ---------------------------------------------------------------------------

def _check_keepout_positions(
    pcb: PCBDesign,
) -> list[IntegrityIssue]:
    """Verify keepout zones are near the components they protect."""
    issues: list[IntegrityIssue] = []

    # Find RF module
    _rf_kw = ("esp32", "wroom", "nina", "wifi", "ble", "nrf52")
    rf_fp = next(
        (fp for fp in pcb.footprints
         if any(kw in (fp.value or "").lower() for kw in _rf_kw)
         or any(kw in (fp.lib_id or "").lower() for kw in _rf_kw)),
        None,
    )
    if rf_fp is None:
        return issues

    # Find antenna keepout (largest non-mounting-hole keepout)
    antenna_ko = None
    max_area = 0.0
    for ko in pcb.keepouts:
        xs = [p.x for p in ko.polygon]
        ys = [p.y for p in ko.polygon]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        area = w * h
        # Skip small keepouts (mounting holes ~5x5mm)
        if area > max_area and area > 50.0:
            max_area = area
            antenna_ko = ko

    if antenna_ko is None:
        issues.append(IntegrityIssue(
            severity="major", category="keepout", ref=rf_fp.ref,
            message=f"{rf_fp.ref}: no antenna keepout zone found",
        ))
        return issues

    # Check keepout is near RF module
    ko_xs = [p.x for p in antenna_ko.polygon]
    ko_ys = [p.y for p in antenna_ko.polygon]
    ko_cx = (min(ko_xs) + max(ko_xs)) / 2.0
    ko_cy = (min(ko_ys) + max(ko_ys)) / 2.0

    dist = ((ko_cx - rf_fp.position.x) ** 2 + (ko_cy - rf_fp.position.y) ** 2) ** 0.5
    if dist > 20.0:
        issues.append(IntegrityIssue(
            severity="critical", category="keepout", ref=rf_fp.ref,
            message=(
                f"{rf_fp.ref}: antenna keepout center ({ko_cx:.0f},{ko_cy:.0f}) "
                f"is {dist:.0f}mm from RF module ({rf_fp.position.x:.0f},"
                f"{rf_fp.position.y:.0f}) — should be adjacent"
            ),
        ))

    return issues


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_pcb_integrity(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[IntegrityIssue]:
    """Run all PCB integrity checks.

    Returns a list of issues (empty = all checks pass).
    Call this after every ``build_pcb()`` as a hard gate.
    """
    issues: list[IntegrityIssue] = []
    issues.extend(_check_component_sync(pcb, requirements))
    issues.extend(_check_pad_model_match(pcb, requirements))
    issues.extend(_check_3d_models(pcb))
    issues.extend(_check_keepout_positions(pcb))

    if issues:
        _log.warning(
            "PCB integrity: %d issues (%d critical)",
            len(issues),
            sum(1 for i in issues if i.severity == "critical"),
        )
    else:
        _log.info("PCB integrity: all checks pass")

    return issues


def format_integrity_report(issues: list[IntegrityIssue]) -> str:
    """Format issues as a human-readable report."""
    if not issues:
        return "PCB Integrity: ALL CHECKS PASS"

    lines = [f"PCB Integrity: {len(issues)} issues found"]
    for issue in sorted(issues, key=lambda i: ("critical", "major", "minor").index(i.severity)):
        lines.append(f"  [{issue.severity:8s}] [{issue.category:10s}] {issue.message}")
    return "\n".join(lines)
