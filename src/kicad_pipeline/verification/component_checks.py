"""Component-level check definitions — programmatic + AI check item generation.

Wraps the existing 13 structural checks from ``component_verifier.py`` in the
new ``CheckResult`` format and adds 3 new programmatic checks (courtyard
existence, courtyard size, pad shape validation).

Also defines the per-component AI check items for fabricator and EE personas.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kicad_pipeline.verification.orchestrator import (
    CheckItem,
    CheckResult,
    CheckSeverity,
    Persona,
)

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.pcb import Footprint
    from kicad_pipeline.validation.component_registry import ComponentSpec
    from kicad_pipeline.verification.checklist import VerificationChecklist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check ID mapping: existing verifier check names → new PROG-* IDs
# ---------------------------------------------------------------------------

_VERIFIER_CHECK_MAP: dict[str, str] = {
    "pad_count": "PROG-PAD-COUNT",
    "pad_type": "PROG-PAD-TYPE",
    "no_duplicate_pads": "PROG-NO-DUP-PADS",
    "3d_model_present": "PROG-3D-PRESENT",
    "3d_model_file_exists": "PROG-3D-FILE",
    "model_rotation": "PROG-3D-ROTATION",
    "model_offset": "PROG-3D-OFFSET",
    "pad_symmetry": "PROG-PAD-SYMMETRY",
    "pad_centroid_near_origin": "PROG-CENTROID",
    "tht_drill_present": "PROG-THT-DRILL",
    "3d_body_pad_alignment": "PROG-BODY-PAD-ALIGN",
    "body_covers_pads": "PROG-BODY-COVERS",
    "pad_extent_vs_body": "PROG-PAD-BODY-RATIO",
}

_SEVERITY_MAP: dict[str, CheckSeverity] = {
    "critical": CheckSeverity.CRITICAL,
    "major": CheckSeverity.MAJOR,
    "minor": CheckSeverity.MINOR,
    "info": CheckSeverity.INFO,
}


# ---------------------------------------------------------------------------
# New programmatic checks
# ---------------------------------------------------------------------------


def _check_courtyard_exists(
    fp: Footprint,
    spec: ComponentSpec,
) -> CheckResult:
    """Verify the footprint has a courtyard layer polygon."""
    ref = spec.component_id
    has_courtyard = any(
        hasattr(item, "layer") and "CrtYd" in getattr(item, "layer", "")
        for item in getattr(fp, "graphics", ())
    )
    if has_courtyard:
        return CheckResult(
            item_id=f"PROG-COURTYARD-EXISTS-{ref}",
            passed=True,
            detail="courtyard polygon found",
            severity=CheckSeverity.MINOR,
        )
    return CheckResult(
        item_id=f"PROG-COURTYARD-EXISTS-{ref}",
        passed=False,
        detail="no courtyard (F.CrtYd/B.CrtYd) polygon found in footprint",
        severity=CheckSeverity.MINOR,
    )


def _check_courtyard_size(
    fp: Footprint,
    spec: ComponentSpec,
) -> CheckResult:
    """Verify courtyard extends at least 0.25mm past pads (IPC-7351B)."""
    ref = spec.component_id
    # Compute pad bounding box
    if not fp.pads:
        return CheckResult(
            item_id=f"PROG-COURTYARD-SIZE-{ref}",
            passed=True,
            detail="no pads to check against",
            severity=CheckSeverity.MINOR,
        )

    pad_xs = [p.position.x for p in fp.pads]
    pad_ys = [p.position.y for p in fp.pads]
    pad_min_x = min(pad_xs)
    pad_max_x = max(pad_xs)
    pad_min_y = min(pad_ys)
    pad_max_y = max(pad_ys)

    # Look for courtyard rectangles in graphics
    crtyd_found = False
    for item in getattr(fp, "graphics", ()):
        layer = getattr(item, "layer", "")
        if "CrtYd" not in layer:
            continue
        crtyd_found = True
        # Check if courtyard bounds extend past pad bounds
        # For rect/poly graphics, check start/end points
        start = getattr(item, "start", None)
        end = getattr(item, "end", None)
        if start is not None and end is not None:
            sx = start.x if hasattr(start, "x") else start[0]
            sy = start.y if hasattr(start, "y") else start[1]
            ex = end.x if hasattr(end, "x") else end[0]
            ey = end.y if hasattr(end, "y") else end[1]
            crt_min_x = min(sx, ex)
            crt_max_x = max(sx, ex)
            crt_min_y = min(sy, ey)
            crt_max_y = max(sy, ey)

            margin_left = pad_min_x - crt_min_x
            margin_right = crt_max_x - pad_max_x
            margin_top = pad_min_y - crt_min_y
            margin_bottom = crt_max_y - pad_max_y
            min_margin = min(margin_left, margin_right, margin_top, margin_bottom)

            if min_margin < 0.20:  # 0.25mm target, 0.05mm tolerance
                return CheckResult(
                    item_id=f"PROG-COURTYARD-SIZE-{ref}",
                    passed=False,
                    detail=f"courtyard margin {min_margin:.2f}mm < 0.25mm IPC minimum",
                    severity=CheckSeverity.MINOR,
                )
            return CheckResult(
                item_id=f"PROG-COURTYARD-SIZE-{ref}",
                passed=True,
                detail=f"courtyard margin {min_margin:.2f}mm >= 0.25mm",
                severity=CheckSeverity.MINOR,
            )

    if not crtyd_found:
        return CheckResult(
            item_id=f"PROG-COURTYARD-SIZE-{ref}",
            passed=False,
            detail="no courtyard found to measure",
            severity=CheckSeverity.MINOR,
        )

    return CheckResult(
        item_id=f"PROG-COURTYARD-SIZE-{ref}",
        passed=True,
        detail="courtyard present (complex geometry, not measurable)",
        severity=CheckSeverity.MINOR,
    )


def _check_pad_shape(
    fp: Footprint,
    spec: ComponentSpec,
) -> CheckResult:
    """Verify pad shapes are valid for the component type."""
    ref = spec.component_id
    if not fp.pads:
        return CheckResult(
            item_id=f"PROG-PAD-SHAPE-{ref}",
            passed=True,
            detail="no pads",
            severity=CheckSeverity.MINOR,
        )

    issues: list[str] = []
    for pad in fp.pads:
        shape = getattr(pad, "shape", "rect")
        pad_type = getattr(pad, "pad_type", "smd")

        # THT pads should be oval or circle, not rect
        if pad_type == "thru_hole" and shape == "rect":
            # Thermal pads on THT can be rect — only flag signal pads
            size = getattr(pad, "size", (0, 0))
            if size[0] < 3.0 and size[1] < 3.0:
                issues.append(f"pad {pad.number}: THT with rect shape (should be oval/circle)")

        # SMD pads should not be circle for most passives
        if (
            pad_type == "smd"
            and shape == "circle"
            and spec.ref.startswith(("R", "C", "L", "D", "FB"))
        ):
            issues.append(
                f"pad {pad.number}: SMD passive with circle shape"
                " (should be rect/roundrect)"
            )

    if issues:
        return CheckResult(
            item_id=f"PROG-PAD-SHAPE-{ref}",
            passed=False,
            detail="; ".join(issues[:3]),
            severity=CheckSeverity.MINOR,
        )

    return CheckResult(
        item_id=f"PROG-PAD-SHAPE-{ref}",
        passed=True,
        detail=f"all {len(fp.pads)} pad shapes valid",
        severity=CheckSeverity.MINOR,
    )


# ---------------------------------------------------------------------------
# Convert existing verifier results to new format
# ---------------------------------------------------------------------------


def _convert_verifier_result(
    verifier_check: object,
    component_id: str,
) -> CheckResult:
    """Convert a component_verifier.CheckResult to verification.CheckResult."""
    # The verifier returns its own CheckResult dataclass
    name = getattr(verifier_check, "name", "unknown")
    passed = getattr(verifier_check, "passed", False)
    detail = getattr(verifier_check, "detail", "")
    severity_str = getattr(verifier_check, "severity", "major")

    prefix = _VERIFIER_CHECK_MAP.get(name, f"PROG-{name.upper()}")
    item_id = f"{prefix}-{component_id}"

    return CheckResult(
        item_id=item_id,
        passed=passed,
        detail=detail,
        severity=_SEVERITY_MAP.get(severity_str, CheckSeverity.MAJOR),
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Run all programmatic checks for a board
# ---------------------------------------------------------------------------


def run_programmatic_checks(
    board_path: Path,
    component_refs: list[str] | None = None,
    checklist: VerificationChecklist | None = None,
) -> list[CheckResult]:
    """Run all programmatic structural checks for registered components.

    Uses the same isolation-footprint approach as ``verify_component()`` —
    builds a parametric footprint from the ComponentSpec and runs structural
    checks against it.  This ensures checks match what real boards produce.

    Args:
        board_path: Path to the .kicad_pcb file (used for context/logging).
        component_refs: Optional subset of component IDs to check.
        checklist: Optional checklist for known-bug filtering.

    Returns:
        List of CheckResult for every check on every component.
    """
    from kicad_pipeline.pcb.isolation_board import build_isolation_footprint
    from kicad_pipeline.validation.component_registry import ComponentRegistry
    from kicad_pipeline.validation.component_verifier import verify_structural

    all_results: list[CheckResult] = []
    registry = ComponentRegistry()

    # Determine which components to check
    all_specs = registry.all_components()
    if component_refs:
        specs = [
            s for s in all_specs
            if s.component_id in component_refs or s.ref in component_refs
        ]
    else:
        specs = list(all_specs)

    for spec in specs:
        # Build isolation footprint (same path as real boards)
        try:
            fp = build_isolation_footprint(spec, use_jlcpcb=True)
        except Exception:
            logger.exception(
                "Failed to build isolation footprint for %s", spec.component_id,
            )
            all_results.append(
                CheckResult(
                    item_id=f"PROG-BUILD-FP-{spec.component_id}",
                    passed=False,
                    detail=f"Failed to build footprint for {spec.component_id}",
                    severity=CheckSeverity.CRITICAL,
                )
            )
            continue

        # Run existing 13 structural checks
        verification = verify_structural(fp, spec)
        for check in verification.checks:
            all_results.append(_convert_verifier_result(check, spec.component_id))

        # Run 3 new checks
        all_results.append(_check_courtyard_exists(fp, spec))
        all_results.append(_check_courtyard_size(fp, spec))
        all_results.append(_check_pad_shape(fp, spec))

    return all_results


# ---------------------------------------------------------------------------
# AI check item definitions (for Phase 2)
# ---------------------------------------------------------------------------

# Fabricator visual checks per component
FAB_COMPONENT_CHECKS: tuple[tuple[str, str], ...] = (
    ("BODY-ALIGN", "Body centered on pads (3D top view)"),
    ("BODY-FLAT", "Body flat on board surface, not floating (3D iso view)"),
    ("BODY-ROT", "Pin 1 indicator matches pad 1 position (3D top view)"),
    ("BODY-SIZE", "Body size proportional to pad footprint (3D top view)"),
    ("TYPE-MATCH", "Component looks like correct type: relay=relay, cap=cap (3D iso)"),
    ("SILK", "Ref designator visible, not overlapping pads (2D top)"),
    ("CLEARANCE", "Adequate courtyard gap from neighbors (2D board view)"),
    ("ASSY", "Pick-and-place feasible: orientation, access (3D iso)"),
)

# EE visual checks per component
EE_COMPONENT_CHECKS: tuple[tuple[str, str], ...] = (
    ("NET-ASSIGN", "Power/ground pins on correct nets"),
    ("PIN-FUNC", "Pin assignments match datasheet"),
    ("DECAP-DIST", "Decoupling cap within 3mm of IC VCC pin (ICs only)"),
    ("SIGNAL-FLOW", "Component in correct signal chain position"),
    ("THERMAL", "Thermal pad grounded properly (if applicable)"),
    ("PROTECTION", "ESD/clamp/filter present where needed"),
)


def generate_fab_check_items(
    refs: list[str],
    checklist: VerificationChecklist | None = None,
) -> list[CheckItem]:
    """Generate fabricator visual check items for a batch of components."""
    items: list[CheckItem] = []
    for ref in refs:
        for category, description in FAB_COMPONENT_CHECKS:
            item_id = f"FAB-{category}-{ref}"
            bug_ids: tuple[str, ...] = ()
            if checklist:
                bugs = checklist.bugs_for_check(item_id)
                bug_ids = tuple(b.id for b in bugs)
            items.append(
                CheckItem(
                    item_id=item_id,
                    description=description,
                    component_ref=ref,
                    persona=Persona.FAB,
                    category=category,
                    known_bug_ids=bug_ids,
                )
            )
    return items


def generate_ee_check_items(
    refs: list[str],
    checklist: VerificationChecklist | None = None,
) -> list[CheckItem]:
    """Generate EE visual check items for a batch of components."""
    items: list[CheckItem] = []
    for ref in refs:
        for category, description in EE_COMPONENT_CHECKS:
            item_id = f"EE-{category}-{ref}"
            bug_ids: tuple[str, ...] = ()
            if checklist:
                bugs = checklist.bugs_for_check(item_id)
                bug_ids = tuple(b.id for b in bugs)
            items.append(
                CheckItem(
                    item_id=item_id,
                    description=description,
                    component_ref=ref,
                    persona=Persona.EE,
                    category=category,
                    known_bug_ids=bug_ids,
                )
            )
    return items


# Board-level checks (for Phase 2)
FAB_BOARD_CHECKS: tuple[tuple[str, str], ...] = (
    ("FAB-BOARD-PATTERN", "Repeated subcircuits visually identical"),
    ("FAB-BOARD-GRID", "Components grid-aligned (rows share Y, cols share X)"),
    ("FAB-BOARD-SPACING", "Uniform gaps in repeated elements"),
    ("FAB-BOARD-ORPHAN", "No orphan components outside functional groups"),
    ("FAB-BOARD-SILK", "Board-wide silkscreen readability"),
    ("FAB-BOARD-EDGE", "All connectors within 5mm of board edge"),
    ("FAB-BOARD-AESTHETICS", "Professional, organized appearance"),
    ("FAB-BOARD-FIDUCIALS", "At least 2 fiducials for pick-and-place"),
)

EE_BOARD_CHECKS: tuple[tuple[str, str], ...] = (
    ("EE-BOARD-POWER-FLOW", "Power tree topology matches physical layout"),
    ("EE-BOARD-GND-RETURN", "GND paths do not cross under analog signals"),
    ("EE-BOARD-EMI", "EMI sources have keepout distances"),
    ("EE-BOARD-DOMAINS", "Voltage domain boundaries physically separated"),
    ("EE-BOARD-HOT-LOOP", "All regulator hot loops under 50mm²"),
    ("EE-BOARD-ISOLATION", "Relay-analog gap at least 10mm"),
    ("EE-BOARD-DECAP-ALL", "Every IC has decoupling within 3mm"),
    ("EE-BOARD-CRYSTAL", "Crystal + load caps within 5mm of OSC pins"),
)
