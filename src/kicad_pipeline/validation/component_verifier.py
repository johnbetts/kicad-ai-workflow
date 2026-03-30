"""Component isolation verification — structural and visual checks.

Each check targets a specific failure mode that has historically been caught
too late (during board-level review).  Structural checks run without rendering
and are fast enough for CI.  Visual checks require ``kicad-image-gen``.
"""

from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint
    from kicad_pipeline.validation.component_registry import ComponentRegistry, ComponentSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """Result of a single verification check."""

    name: str
    passed: bool
    detail: str
    severity: str  # "critical", "major", "minor"


@dataclass(frozen=True)
class ComponentVerification:
    """Aggregate result of verifying one component."""

    component_id: str
    passed: bool
    checks: tuple[CheckResult, ...]
    pcb_path: Path | None = None
    render_paths: tuple[tuple[str, Path], ...] = ()


# ---------------------------------------------------------------------------
# Structural checks (no rendering, no file I/O)
# ---------------------------------------------------------------------------

_ROTATION_TOLERANCE = 5.0  # degrees


def _check_pad_count(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """Verify pad count matches registry expectation.

    JLCPCB footprints may include extra ground/shield pads not counted
    in the parametric footprint.  Allow the actual count to be >= expected
    when the footprint source is ``"jlcpcb"``.

    Connectors (J, K refs) and RJ45 footprints also allow >= because they
    may have additional shield/mounting pads beyond the signal pad count.
    """
    actual = len(fp.pads)
    is_jlcpcb = getattr(fp, "footprint_source", "") == "jlcpcb"
    fid = (spec.footprint_id or "").upper()
    is_connector = spec.ref.startswith(("J", "K")) or any(
        kw in fid for kw in ("RJ45", "TERMINAL", "CONNECTOR", "PINHEADER")
    )
    if is_jlcpcb or is_connector:
        ok = actual >= spec.expected_pads
    else:
        ok = actual == spec.expected_pads
    detail = f"expected {spec.expected_pads}, got {actual}"
    if (is_jlcpcb or is_connector) and actual > spec.expected_pads:
        detail += f" (+{actual - spec.expected_pads} extra pads)"
    return CheckResult(
        name="pad_count",
        passed=ok,
        detail=detail,
        severity="critical",
    )


def _check_pad_type(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """Verify all pads match expected type (smd/thru_hole).

    Mixed-technology components (RJ45-SMD with THT signal pins + SMD shield,
    buzzers, tactile switches) are exempt — they legitimately mix pad types.
    """
    fid = (spec.footprint_id or "").upper()
    # Mixed-technology footprints: exempt from pad type check
    _MIXED_TECH_KW = ("RJ45", "BUZZER", "SW_PUSH", "SW_SPST", "SWITCH", "BUTTON")
    if any(kw in fid for kw in _MIXED_TECH_KW):
        return CheckResult(
            name="pad_type", passed=True,
            detail="exempt (mixed-technology footprint)", severity="critical",
        )
    wrong: list[str] = []
    for pad in fp.pads:
        if pad.pad_type == "np_thru_hole":
            continue  # NPTH pads (shields, mounting) are exempt
        if pad.pad_type != spec.expected_pad_type:
            wrong.append(f"pad {pad.number}: {pad.pad_type}")
    ok = len(wrong) == 0
    detail = "all match" if ok else f"mismatches: {', '.join(wrong[:5])}"
    return CheckResult(name="pad_type", passed=ok, detail=detail, severity="critical")


def _check_no_duplicate_pads(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """Verify no duplicate pad numbers (catches BUG-PIPE-005).

    Shield pads ("SH", "") and NPTH pads are exempt — connectors like RJ45
    legitimately have multiple shield/mounting pads with the same designation.
    """
    _EXEMPT_PAD_NAMES = {"SH", "MP", "", "0"}  # shield, mounting post, unnamed, ground tab
    # Tact switches have paired pads (pin 1 and 2 each appear twice) — this
    # is the physical design, not a bug.  Only flag duplicates beyond 2x.
    is_switch = spec.ref.startswith("SW") or "switch" in spec.description.lower()
    numbers = [p.number for p in fp.pads if p.number not in _EXEMPT_PAD_NAMES]
    from collections import Counter
    counts = Counter(numbers)
    max_allowed = 2 if is_switch else 1
    dupes = [n for n, c in counts.items() if c > max_allowed]
    ok = len(dupes) == 0
    detail = "no duplicates" if ok else f"duplicates (>{max_allowed}x): {', '.join(dupes)}"
    return CheckResult(
        name="no_duplicate_pads", passed=ok, detail=detail, severity="critical"
    )


def _check_3d_model_present(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """Verify at least one 3D model is attached."""
    fid = (spec.footprint_id or "").upper()
    # Mounting holes, test points — exempt by ref OR footprint_id
    is_mounting_hole = spec.ref.startswith(("H", "TP")) or "MOUNTINGHOLE" in fid
    if is_mounting_hole:
        return CheckResult(
            name="3d_model_present",
            passed=True,
            detail="exempt (mounting hole / test point)",
            severity="critical",
        )
    ok = len(fp.models) > 0
    # Connectors: exempt by ref OR footprint_id (RJ45, terminal blocks, etc.)
    is_connector = spec.ref.startswith(("J", "K")) or any(
        kw in fid for kw in ("RJ45", "TERMINAL", "CONNECTOR")
    )
    if not ok and is_connector:
        return CheckResult(
            name="3d_model_present",
            passed=True,
            detail="no model (connector exempt — no matching STEP)",
            severity="minor",
        )
    return CheckResult(
        name="3d_model_present",
        passed=ok,
        detail=f"{len(fp.models)} model(s)" if ok else "NO 3D model attached",
        severity="critical",
    )


def _check_3d_model_file_exists(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """Verify the STEP model file exists on disk."""
    if not fp.models or spec.ref.startswith(("H", "TP")):
        return CheckResult(
            name="3d_model_file_exists",
            passed=True,
            detail="skipped (no model or exempt)",
            severity="major",
        )

    from kicad_pipeline.pcb.footprints import _step_file_exists  # type: ignore[attr-defined]

    model = fp.models[0]
    exists = _step_file_exists(model.path)
    return CheckResult(
        name="3d_model_file_exists",
        passed=exists,
        detail=model.path if exists else f"MISSING: {model.path}",
        severity="major",
    )


def _check_model_rotation(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """Verify 3D model Z rotation matches registry expectation.

    JLCPCB footprints may have mirrored pad layouts that add a 180-degree
    correction.  Accept both expected and expected+180 as valid.
    """
    if not fp.models or spec.ref.startswith(("H", "TP")):
        return CheckResult(
            name="model_rotation",
            passed=True,
            detail="skipped",
            severity="major",
        )
    model = fp.models[0]
    actual_z = model.rotate[2] if len(model.rotate) > 2 else 0.0
    expected_z = spec.model_rotation_z
    # Normalize to [0, 360)
    diff = abs((actual_z % 360.0) - (expected_z % 360.0))
    if diff > 180.0:
        diff = 360.0 - diff
    # Also accept expected + 180° (JLCPCB mirror correction)
    diff_mirror = abs((actual_z % 360.0) - ((expected_z + 180.0) % 360.0))
    if diff_mirror > 180.0:
        diff_mirror = 360.0 - diff_mirror
    ok = diff < _ROTATION_TOLERANCE or diff_mirror < _ROTATION_TOLERANCE
    return CheckResult(
        name="model_rotation",
        passed=ok,
        detail=f"Z rotate: expected {expected_z}°, got {actual_z}° (diff {diff:.1f}°)",
        severity="major",
    )


def _check_model_offset(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """Verify 3D model XY offset is within acceptable range."""
    if not fp.models or spec.ref.startswith(("H", "TP")):
        return CheckResult(
            name="model_offset",
            passed=True,
            detail="skipped",
            severity="major",
        )
    model = fp.models[0]
    ox = model.offset[0] if len(model.offset) > 0 else 0.0
    oy = model.offset[1] if len(model.offset) > 1 else 0.0
    dist = math.sqrt(ox * ox + oy * oy)
    ok = dist <= spec.model_offset_xy_max_mm
    return CheckResult(
        name="model_offset",
        passed=ok,
        detail=f"XY offset: ({ox:.2f}, {oy:.2f}) = {dist:.2f}mm "
        f"(max {spec.model_offset_xy_max_mm}mm)",
        severity="major",
    )


def _check_pad_symmetry(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """For 2-pad SMD passives, verify pads are symmetric about origin."""
    if spec.expected_pads != 2 or spec.expected_pad_type != "smd":
        return CheckResult(
            name="pad_symmetry",
            passed=True,
            detail="skipped (not a 2-pad SMD passive)",
            severity="minor",
        )
    if len(fp.pads) < 2:
        return CheckResult(
            name="pad_symmetry",
            passed=False,
            detail="fewer than 2 pads",
            severity="minor",
        )
    p1, p2 = fp.pads[0], fp.pads[1]
    sym_x = abs(p1.position.x + p2.position.x) < 0.2
    sym_y = abs(p1.position.y + p2.position.y) < 0.2
    ok = sym_x or sym_y
    return CheckResult(
        name="pad_symmetry",
        passed=ok,
        detail=f"pad1=({p1.position.x:.2f},{p1.position.y:.2f}), "
        f"pad2=({p2.position.x:.2f},{p2.position.y:.2f})",
        severity="minor",
    )


def _check_pad_centroid_near_origin(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """Verify centroid of all pads is near footprint origin."""
    if not fp.pads:
        return CheckResult(
            name="pad_centroid_near_origin",
            passed=True,
            detail="no pads",
            severity="minor",
        )
    cx = sum(p.position.x for p in fp.pads) / len(fp.pads)
    cy = sum(p.position.y for p in fp.pads) / len(fp.pads)
    dist = math.sqrt(cx * cx + cy * cy)
    # Connectors and modules can have offset origins — relaxed threshold
    max_dist = 5.0 if spec.ref.startswith(("J", "K", "SW")) else 2.0
    ok = dist <= max_dist
    return CheckResult(
        name="pad_centroid_near_origin",
        passed=ok,
        detail=f"centroid=({cx:.2f},{cy:.2f}), distance={dist:.2f}mm (max {max_dist}mm)",
        severity="minor",
    )


def _check_tht_drill_present(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """For through-hole components, verify all THT pads have drill > 0."""
    if spec.expected_pad_type != "thru_hole":
        return CheckResult(
            name="tht_drill_present",
            passed=True,
            detail="skipped (not THT)",
            severity="critical",
        )
    missing_drill: list[str] = []
    for pad in fp.pads:
        if pad.pad_type == "thru_hole" and (
            pad.drill_diameter is None or pad.drill_diameter <= 0
        ):
            missing_drill.append(pad.number)
    ok = len(missing_drill) == 0
    detail = "all drills present" if ok else f"missing drill: pads {', '.join(missing_drill)}"
    return CheckResult(
        name="tht_drill_present", passed=ok, detail=detail, severity="critical"
    )


# ---------------------------------------------------------------------------
# Body-pad alignment checks (math-based, catches DIP/relay/connector offset)
# ---------------------------------------------------------------------------


def _check_3d_body_pad_alignment(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """Verify the 3D model offset would correctly align body with pads.

    Uses the kicad_ref_pad1 values from the registry to check that the 3D
    model offset accounts for any difference between the KiCad library
    pad-1 position and the actual footprint's pad-1 position.
    Only meaningful for JLCPCB footprints — parametric footprints use their
    own pad layout which doesn't need runtime offset correction.
    """
    if not fp.models or spec.ref.startswith(("H", "TP")):
        return CheckResult(
            name="3d_body_pad_alignment",
            passed=True,
            detail="skipped (no model or exempt)",
            severity="major",
        )
    # Only check JLCPCB footprints — parametric footprints don't need offset
    if getattr(fp, "footprint_source", "") != "jlcpcb":
        return CheckResult(
            name="3d_body_pad_alignment",
            passed=True,
            detail="skipped (parametric footprint)",
            severity="major",
        )
    # Skip if no reference pad1 data in registry
    if spec.kicad_ref_pad1_x == 0.0 and spec.kicad_ref_pad1_y == 0.0:
        return CheckResult(
            name="3d_body_pad_alignment",
            passed=True,
            detail="skipped (no kicad_ref_pad1 data)",
            severity="major",
        )

    # Compute JLCPCB pad centroid — the offset application code uses
    # -JLCPCB_centroid as the model offset (centroid-based, NOT pad-1-based).
    if not fp.pads:
        return CheckResult(
            name="3d_body_pad_alignment",
            passed=True,
            detail="skipped (no pads)",
            severity="major",
        )

    jxs = [p.position.x for p in fp.pads]
    jys = [p.position.y for p in fp.pads]
    j_cx = (min(jxs) + max(jxs)) / 2.0
    j_cy = (min(jys) + max(jys)) / 2.0

    # Expected offset = -JLCPCB_centroid (aligns body center with pad center)
    expected_ox = -j_cx
    expected_oy = -j_cy

    model = fp.models[0]
    model_ox = model.offset[0] if len(model.offset) > 0 else 0.0
    model_oy = model.offset[1] if len(model.offset) > 1 else 0.0

    diff = math.sqrt((model_ox - expected_ox) ** 2 + (model_oy - expected_oy) ** 2)

    # Tolerance: 2mm — enough for rounding and JLCPCB pad layout variations
    # Modules (ESP32) have body extending past pads (antenna) so the STEP
    # model origin may be offset from pad centroid by several mm.
    tolerance = 5.0 if spec.ref.startswith("U") else 2.0
    ok = diff <= tolerance
    return CheckResult(
        name="3d_body_pad_alignment",
        passed=ok,
        detail=(
            f"model offset=({model_ox:.2f},{model_oy:.2f}), "
            f"expected=({expected_ox:.2f},{expected_oy:.2f}), "
            f"diff={diff:.2f}mm"
        ),
        severity="major",
    )


def _check_body_covers_pads(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """Verify the 3D body extent covers all pads.

    Uses body dimensions from the registry and the model offset to compute
    where the 3D body sits relative to the footprint origin.  All pads
    should be within the body extent (with margin for edge castellations).
    """
    if not fp.pads or spec.body_width_mm is None or spec.body_height_mm is None:
        return CheckResult(
            name="body_covers_pads", passed=True,
            detail="skipped (no pads or no body dims)", severity="major",
        )
    fid = (spec.footprint_id or "").upper()
    is_mounting_hole = spec.ref.startswith(("H", "TP")) or "MOUNTINGHOLE" in fid
    if is_mounting_hole:
        return CheckResult(
            name="body_covers_pads", passed=True,
            detail="exempt", severity="major",
        )

    # Body extent relative to footprint origin
    # Model offset shifts the 3D body, so body center = model_offset
    if fp.models:
        m = fp.models[0]
        body_cx = m.offset[0] if len(m.offset) > 0 else 0.0
        body_cy = m.offset[1] if len(m.offset) > 1 else 0.0
    else:
        body_cx, body_cy = 0.0, 0.0

    half_w = spec.body_width_mm / 2.0
    half_h = spec.body_height_mm / 2.0
    body_x_min = body_cx - half_w
    body_x_max = body_cx + half_w
    body_y_min = body_cy - half_h
    body_y_max = body_cy + half_h

    # Margin: pads extend past body for ICs (gull-wing, J-lead, QFP/QFN)
    # and connectors (shield pins, mounting posts).
    # IC pads typically extend 1-3mm past plastic body per side.
    # THT components (pin headers, relays) have pads extending well past body.
    is_ic = (spec.ref.startswith("U")
             or (spec.expected_pad_type == "smd" and spec.expected_pads > 2))
    is_connector = spec.ref.startswith(("J", "K")) or any(
        kw in fid for kw in ("TERMINAL", "PINHEADER", "PINSOCKET", "CONNECTOR", "RJ45")
    )
    is_relay = "RELAY" in fid
    is_jlcpcb = getattr(fp, "footprint_source", "") == "jlcpcb"
    if is_ic:
        margin = 8.0 if is_jlcpcb else 4.0
    elif is_connector or is_relay:
        margin = 8.0 if is_jlcpcb else 6.0
    else:
        margin = 4.0 if is_jlcpcb else 2.0

    exposed: list[str] = []
    for pad in fp.pads:
        px, py = pad.position.x, pad.position.y
        if (px < body_x_min - margin or px > body_x_max + margin
                or py < body_y_min - margin or py > body_y_max + margin):
            exposed.append(f"{pad.number}@({px:.1f},{py:.1f})")

    ok = len(exposed) == 0
    if ok:
        detail = f"all pads within body ({body_x_min:.1f},{body_y_min:.1f})-({body_x_max:.1f},{body_y_max:.1f})"
    else:
        detail = (
            f"pads outside body: {', '.join(exposed[:5])}; "
            f"body=({body_x_min:.1f},{body_y_min:.1f})-({body_x_max:.1f},{body_y_max:.1f})"
        )
    return CheckResult(
        name="body_covers_pads", passed=ok, detail=detail, severity="critical",
    )


def _check_pad_extent_vs_body(fp: Footprint, spec: ComponentSpec) -> CheckResult:
    """Verify pad span is proportional to body dimensions.

    If pads span 15mm but body is 5mm, or pads span 2mm but body is 20mm,
    something is wrong with pad positions or body size.
    """
    if not fp.pads or spec.body_width_mm is None or spec.body_height_mm is None:
        return CheckResult(
            name="pad_extent_vs_body", passed=True,
            detail="skipped", severity="major",
        )
    fid = (spec.footprint_id or "").upper()
    is_mounting_hole = spec.ref.startswith(("H", "TP")) or "MOUNTINGHOLE" in fid
    if is_mounting_hole:
        return CheckResult(
            name="pad_extent_vs_body", passed=True,
            detail="exempt", severity="major",
        )

    xs = [p.position.x for p in fp.pads]
    ys = [p.position.y for p in fp.pads]
    pad_span_x = max(xs) - min(xs)
    pad_span_y = max(ys) - min(ys)

    body_w = spec.body_width_mm
    body_h = spec.body_height_mm

    # IC pads (gull-wing, QFP, QFN) extend 1-3mm past the plastic body per
    # side, so pad span can be up to ~200% of body width for small ICs.
    # SOT-223/SOT-89: large tab pad extends ~2x body height — use higher ratio.
    # THT connectors/relays have shield/mounting pins that extend further.
    is_ic = spec.ref.startswith("U") or (spec.expected_pad_type == "smd" and spec.expected_pads > 4)
    is_connector = spec.ref.startswith(("J", "K")) or any(
        kw in fid for kw in ("TERMINAL", "PINHEADER", "PINSOCKET", "CONNECTOR", "RJ45", "RELAY")
    )
    is_tab_package = any(kw in fid for kw in ("SOT-223", "SOT-89", "TO-252", "TO-263", "DPAK"))
    max_ratio = 2.5 if is_tab_package else (2.5 if is_ic else (3.0 if is_connector else 1.5))
    min_ratio = 0.15

    issues: list[str] = []
    if spec.expected_pads > 2:
        if pad_span_x > 0 and pad_span_x > body_w * max_ratio:
            issues.append(f"X span {pad_span_x:.1f}mm > {max_ratio*100:.0f}% of body {body_w:.1f}mm")
        if pad_span_y > 0 and pad_span_y > body_h * max_ratio:
            issues.append(f"Y span {pad_span_y:.1f}mm > {max_ratio*100:.0f}% of body {body_h:.1f}mm")
        if pad_span_x > 0 and pad_span_x < body_w * min_ratio:
            issues.append(f"X span {pad_span_x:.1f}mm < {min_ratio*100:.0f}% of body {body_w:.1f}mm")
        if pad_span_y > 0 and pad_span_y < body_h * min_ratio:
            issues.append(f"Y span {pad_span_y:.1f}mm < {min_ratio*100:.0f}% of body {body_h:.1f}mm")

    ok = len(issues) == 0
    detail = (
        f"pad span ({pad_span_x:.1f}x{pad_span_y:.1f}) vs body ({body_w:.1f}x{body_h:.1f})"
        if ok else "; ".join(issues)
    )
    return CheckResult(
        name="pad_extent_vs_body", passed=ok, detail=detail,
        severity="major",
    )


# ---------------------------------------------------------------------------
# All structural checks
# ---------------------------------------------------------------------------

_STRUCTURAL_CHECKS = (
    _check_pad_count,
    _check_pad_type,
    _check_no_duplicate_pads,
    _check_3d_model_present,
    _check_3d_model_file_exists,
    _check_model_rotation,
    _check_model_offset,
    _check_pad_symmetry,
    _check_pad_centroid_near_origin,
    _check_tht_drill_present,
    _check_3d_body_pad_alignment,
    _check_body_covers_pads,
    _check_pad_extent_vs_body,
)


def verify_structural(
    fp: Footprint,
    spec: ComponentSpec,
) -> ComponentVerification:
    """Run all structural checks on a footprint (no rendering)."""
    checks = tuple(check(fp, spec) for check in _STRUCTURAL_CHECKS)
    passed = all(c.passed for c in checks if c.severity in ("critical", "major"))
    return ComponentVerification(
        component_id=spec.component_id,
        passed=passed,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# Visual verification (requires kicad-image-gen)
# ---------------------------------------------------------------------------


def _render_isolation_board(
    pcb_path: Path,
    output_dir: Path,
    component_id: str,
) -> tuple[tuple[str, Path], ...]:
    """Render 2D + 3D views of an isolation board.

    Returns tuple of (view_name, image_path) pairs.
    """
    safe_id = component_id.replace("/", "_").replace(" ", "_")
    renders: list[tuple[str, Path]] = []

    # Keep renders ≤1600px so multiple images fit within API dimension limits
    views = [
        ("2d_top", ["kicad-image-gen", "2d", str(pcb_path), "-w", "1600"]),
        ("3d_top", ["kicad-image-gen", "3d", str(pcb_path), "--view", "top",
                     "-w", "1600", "--height", "900"]),
        ("3d_iso", ["kicad-image-gen", "3d", str(pcb_path), "--view", "iso",
                     "-w", "1600", "--height", "900"]),
        ("3d_bottom", ["kicad-image-gen", "3d", str(pcb_path), "--view", "bottom",
                        "-w", "1600", "--height", "900"]),
    ]

    for view_name, cmd in views:
        out_path = output_dir / f"{safe_id}_{view_name}.png"
        full_cmd = [*cmd, "-o", str(out_path)]
        try:
            subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            if out_path.exists() and out_path.stat().st_size > 0:
                renders.append((view_name, out_path))
                logger.info("Rendered %s → %s", view_name, out_path)
            else:
                logger.warning("Render %s produced empty file", view_name)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("Failed to render %s: %s", view_name, exc)

    return tuple(renders)


# ---------------------------------------------------------------------------
# Persona-based visual checks (run on rendered images)
# ---------------------------------------------------------------------------


def _check_render_exists(
    render_paths: tuple[tuple[str, Path], ...],
    view: str,
) -> CheckResult:
    """Verify a specific render view was produced and is non-empty."""
    for name, path in render_paths:
        if name == view:
            size_kb = path.stat().st_size / 1024 if path.exists() else 0
            # A valid render should be at least 5KB
            ok = size_kb >= 5.0
            return CheckResult(
                name=f"render_{view}",
                passed=ok,
                detail=f"{size_kb:.0f}KB" if ok else f"too small ({size_kb:.1f}KB)",
                severity="major",
            )
    return CheckResult(
        name=f"render_{view}", passed=False,
        detail="render not generated", severity="major",
    )


def _check_3d_body_visible(
    render_paths: tuple[tuple[str, Path], ...],
    spec: ComponentSpec,
) -> CheckResult:
    """Fabricator check: 3D body should be visible (render not just empty board).

    Uses image file size as a proxy — a render with a 3D component body is
    significantly larger than a bare board render.
    """
    if spec.ref.startswith(("H", "TP")):
        return CheckResult(
            name="fab_3d_body_visible", passed=True,
            detail="exempt", severity="major",
        )
    for name, path in render_paths:
        if name == "3d_iso" and path.exists():
            size_kb = path.stat().st_size / 1024
            # A 3D render with a component body should be >10KB
            # A bare green board with no model is typically <8KB
            ok = size_kb > 10.0
            return CheckResult(
                name="fab_3d_body_visible",
                passed=ok,
                detail=f"3D iso render {size_kb:.0f}KB (body {'visible' if ok else 'may be missing'})",
                severity="major",
            )
    return CheckResult(
        name="fab_3d_body_visible", passed=False,
        detail="no 3D iso render to check", severity="major",
    )


def _check_2d_pads_visible(
    render_paths: tuple[tuple[str, Path], ...],
    spec: ComponentSpec,
) -> CheckResult:
    """EE check: 2D render should show pads (non-trivial image)."""
    for name, path in render_paths:
        if name == "2d_top" and path.exists():
            size_kb = path.stat().st_size / 1024
            # 2D render with pads and labels should be >5KB
            ok = size_kb > 5.0
            return CheckResult(
                name="ee_2d_pads_visible",
                passed=ok,
                detail=f"2D render {size_kb:.0f}KB (pads {'visible' if ok else 'may be missing'})",
                severity="major",
            )
    return CheckResult(
        name="ee_2d_pads_visible", passed=False,
        detail="no 2D render to check", severity="major",
    )


def _run_persona_checks(
    render_paths: tuple[tuple[str, Path], ...],
    spec: ComponentSpec,
) -> tuple[CheckResult, ...]:
    """Run fabricator and EE persona checks on rendered images."""
    checks: list[CheckResult] = [
        # All renders exist
        _check_render_exists(render_paths, "2d_top"),
        _check_render_exists(render_paths, "3d_iso"),
        _check_render_exists(render_paths, "3d_bottom"),
        # Fabricator: is the 3D body actually rendered?
        _check_3d_body_visible(render_paths, spec),
        # EE: are pads visible in 2D?
        _check_2d_pads_visible(render_paths, spec),
    ]
    return tuple(checks)


# ---------------------------------------------------------------------------
# Auto-computation of kicad_ref_pad1 offsets
# ---------------------------------------------------------------------------


_PAD1_TOLERANCE_MM: float = 0.1


def compute_kicad_ref_pad1(spec: ComponentSpec) -> tuple[float, float]:
    """Compute the KiCad reference pad-1 position from the parametric footprint.

    Uses ``use_jlcpcb=False`` to force the parametric path — this gives the
    KiCad-convention pad layout that the STEP 3D models are designed for.

    Returns:
        ``(pad1_x, pad1_y)`` or ``(0.0, 0.0)`` if no pad numbered "1" exists.
    """
    from kicad_pipeline.pcb.isolation_board import build_isolation_footprint

    fp = build_isolation_footprint(spec, use_jlcpcb=False)
    for pad in fp.pads:
        if pad.number == "1":
            return (pad.position.x, pad.position.y)
    return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Full verification
# ---------------------------------------------------------------------------


def verify_component(
    spec: ComponentSpec,
    output_dir: Path,
    render: bool = True,
    registry: ComponentRegistry | None = None,
) -> ComponentVerification:
    """Full verification: build isolation board, run structural checks, optionally render.

    When ``render=True``, also runs fabricator and EE persona checks on the
    generated images — verifying 3D body visibility, pad rendering, and
    render completeness.

    After structural checks pass, auto-computes ``kicad_ref_pad1_x/y`` from
    the parametric footprint and updates the registry if the values differ by
    more than 0.1 mm.

    Args:
        spec: Component specification from registry.
        output_dir: Directory for PCB file and renders.
        render: If True, generate 2D+3D renders via kicad-image-gen.
        registry: If provided and pad-1 values need updating, mutates and
            saves the registry.

    Returns:
        :class:`ComponentVerification` with all check results and render paths.
    """
    from kicad_pipeline.pcb.isolation_board import (
        build_isolation_board,
        build_isolation_footprint,
    )

    # Generate footprint using the SAME path as real boards (JLCPCB when
    # available).  This ensures evidence images match what actually ships.
    fp = build_isolation_footprint(spec, use_jlcpcb=True)
    structural = verify_structural(fp, spec)

    pcb_path: Path | None = None
    render_paths: tuple[tuple[str, Path], ...] = ()
    all_checks = list(structural.checks)

    # Auto-compute kicad_ref_pad1 from PARAMETRIC path (not JLCPCB) since
    # the KiCad STEP model origin matches the parametric pad layout.
    computed_x, computed_y = compute_kicad_ref_pad1(spec)
    dx = abs(computed_x - spec.kicad_ref_pad1_x)
    dy = abs(computed_y - spec.kicad_ref_pad1_y)
    if dx > _PAD1_TOLERANCE_MM or dy > _PAD1_TOLERANCE_MM:
        logger.info(
            "%s: kicad_ref_pad1 updated (%+.2f,%+.2f) -> (%+.2f,%+.2f)",
            spec.component_id,
            spec.kicad_ref_pad1_x,
            spec.kicad_ref_pad1_y,
            computed_x,
            computed_y,
        )
        if registry is not None:
            registry._replace_spec(
                spec.component_id,
                kicad_ref_pad1_x=computed_x,
                kicad_ref_pad1_y=computed_y,
            )
            registry.save()

    if render:
        # Build isolation board and render it
        pcb_path = build_isolation_board(spec, output_dir)
        render_paths = _render_isolation_board(pcb_path, output_dir, spec.component_id)

        # Run persona-based visual checks on the renders
        persona_checks = _run_persona_checks(render_paths, spec)
        all_checks.extend(persona_checks)

    checks_tuple = tuple(all_checks)
    passed = all(c.passed for c in checks_tuple if c.severity in ("critical", "major"))

    return ComponentVerification(
        component_id=spec.component_id,
        passed=passed,
        checks=checks_tuple,
        pcb_path=pcb_path,
        render_paths=render_paths,
    )
