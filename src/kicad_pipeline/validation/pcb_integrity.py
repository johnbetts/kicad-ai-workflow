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
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.pcb import Footprint, PCBDesign
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
# Check 5: Per-component 3D body-to-pad alignment (BUG-PIPE-004)
# ---------------------------------------------------------------------------

# Maximum acceptable 3D offset magnitude by component category (mm).
# Offsets beyond these thresholds indicate misaligned models.
_MAX_OFFSET_MM: dict[str, float] = {
    "passive": 1.5,   # R, C, L — small bodies, offset should be minimal
    "ic": 5.0,        # ICs/modules — larger bodies, JLCPCB→KiCad offsets common
    "module": 8.0,    # ESP32, W5500 etc — large modules with big origin shifts
    "connector": 3.0,
    "default": 3.0,
}


def _classify_component(fp: Footprint) -> str:
    """Classify a footprint into a component category for 3D checks."""
    ref_prefix = fp.ref.rstrip("0123456789")
    lib_lower = (fp.lib_id or "").lower()

    if ref_prefix in ("R", "C", "L", "FB"):
        return "passive"
    if any(kw in lib_lower for kw in ("esp32", "wroom", "nina", "w5500", "lan87")):
        return "module"
    if ref_prefix in ("J", "P"):
        return "connector"
    if ref_prefix in ("U", "IC"):
        return "ic"
    return "default"


def _pad_centroid(fp: Footprint) -> tuple[float, float]:
    """Return the centroid of all pads in footprint-local coordinates."""
    if not fp.pads:
        return (0.0, 0.0)
    xs = [p.position.x for p in fp.pads]
    ys = [p.position.y for p in fp.pads]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _check_3d_alignment(
    pcb: PCBDesign,
) -> list[IntegrityIssue]:
    """Verify 3D model offset is plausible for every component.

    Checks per component:
      - Offset magnitude within category threshold
      - Z-offset is 0 for SMD components
      - Pad centroid is near origin (large centroid-origin gap + zero offset = misaligned)
    """
    issues: list[IntegrityIssue] = []

    for fp in pcb.footprints:
        if not fp.pads or fp.ref.startswith("H"):
            continue
        if not fp.models:
            continue  # Missing models caught by check 3

        model = fp.models[0]
        ox, oy, oz = model.offset
        offset_mag = (ox ** 2 + oy ** 2) ** 0.5
        category = _classify_component(fp)
        max_off = _MAX_OFFSET_MM.get(category, _MAX_OFFSET_MM["default"])

        # Check 5a: Offset magnitude
        if offset_mag > max_off:
            issues.append(IntegrityIssue(
                severity="major", category="3d_alignment", ref=fp.ref,
                message=(
                    f"{fp.ref}: 3D offset ({ox:.2f}, {oy:.2f}) = {offset_mag:.1f}mm "
                    f"exceeds {category} limit {max_off:.0f}mm"
                ),
            ))

        # Check 5b: Z-offset for SMD components (should be 0)
        if fp.attr == "smd" and abs(oz) > 0.5:
            issues.append(IntegrityIssue(
                severity="major", category="3d_alignment", ref=fp.ref,
                message=f"{fp.ref}: SMD component has Z-offset {oz:.2f}mm (expect 0)",
            ))

        # Check 5c: Pad centroid far from origin but NO compensating offset
        # This catches JLCPCB footprints with non-center origins that forgot
        # to set a 3D offset.
        cx, cy = _pad_centroid(fp)
        centroid_dist = (cx ** 2 + cy ** 2) ** 0.5
        if centroid_dist > 3.0 and offset_mag < 0.5:
            issues.append(IntegrityIssue(
                severity="minor", category="3d_alignment", ref=fp.ref,
                message=(
                    f"{fp.ref}: pad centroid offset {centroid_dist:.1f}mm from origin "
                    f"but 3D offset is ~0 — model may be misaligned"
                ),
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 6: No duplicate pad numbers within a footprint
# ---------------------------------------------------------------------------

def _check_duplicate_pads(
    pcb: PCBDesign,
) -> list[IntegrityIssue]:
    """Verify no footprint has duplicate pad numbers.

    Duplicate pads cause manufacturing failures — pick-and-place machines
    and DRC tools assume unique pad numbers.  This catches the thermal-pad
    duplication bug (BUG-PIPE-005).
    """
    issues: list[IntegrityIssue] = []

    for fp in pcb.footprints:
        if not fp.pads or fp.ref.startswith("H"):
            continue
        seen: dict[str, int] = {}
        for p in fp.pads:
            if not p.number or p.number.strip() == "":
                continue  # skip unnamed pads (shield, mounting)
            seen[p.number] = seen.get(p.number, 0) + 1
        dupes = {num: cnt for num, cnt in seen.items() if cnt > 1}
        if dupes:
            dupe_str = ", ".join(f"pad {n} x{c}" for n, c in dupes.items())
            issues.append(IntegrityIssue(
                severity="critical", category="duplicate_pad", ref=fp.ref,
                message=f"{fp.ref}: duplicate pad numbers: {dupe_str}",
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 7: Mounting hole clearance — no component overlaps mounting holes
# ---------------------------------------------------------------------------

_MOUNTING_HOLE_CLEARANCE_MM = 1.0  # min edge-to-edge gap (relaxed for connectors near corners)


def _fp_bbox_abs(fp: Footprint) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) in board coordinates.

    Uses pad positions + pad sizes, rotated by footprint rotation.
    """
    rot_rad = math.radians(fp.rotation)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    ox, oy = fp.position.x, fp.position.y

    xs: list[float] = []
    ys: list[float] = []
    for p in fp.pads:
        # Rotate pad position by footprint rotation
        rx = p.position.x * cos_r - p.position.y * sin_r
        ry = p.position.x * sin_r + p.position.y * cos_r
        half_w = max(p.size_x, p.size_y) / 2.0
        xs.extend([ox + rx - half_w, ox + rx + half_w])
        ys.extend([oy + ry - half_w, oy + ry + half_w])

    if not xs:
        return (ox, oy, ox, oy)
    return (min(xs), min(ys), max(xs), max(ys))


def _check_mounting_hole_clearance(
    pcb: PCBDesign,
) -> list[IntegrityIssue]:
    """Verify no component courtyard overlaps a mounting hole.

    Mounting holes need clearance for screw heads and standoffs.
    A component placed on top of a mounting hole cannot be assembled.
    """
    issues: list[IntegrityIssue] = []

    # Find mounting holes
    holes: list[Footprint] = [
        fp for fp in pcb.footprints
        if fp.ref.startswith("H") or "mountinghole" in (fp.lib_id or "").lower().replace("_", "")
    ]
    if not holes:
        return issues

    for fp in pcb.footprints:
        if not fp.pads or fp.ref.startswith("H"):
            continue

        fp_x1, fp_y1, fp_x2, fp_y2 = _fp_bbox_abs(fp)

        for hole in holes:
            hx, hy = hole.position.x, hole.position.y
            # Mounting hole effective radius: largest pad / 2 + clearance
            h_radius = _MOUNTING_HOLE_CLEARANCE_MM
            if hole.pads:
                h_radius = max(max(p.size_x, p.size_y) for p in hole.pads) / 2.0
                h_radius += _MOUNTING_HOLE_CLEARANCE_MM

            # Check overlap: expand footprint bbox by clearance, check if hole center is inside
            mh_clr = _MOUNTING_HOLE_CLEARANCE_MM
            if (fp_x1 - mh_clr <= hx <= fp_x2 + mh_clr
                    and fp_y1 - mh_clr <= hy <= fp_y2 + mh_clr):
                dist_x = max(fp_x1 - hx, 0, hx - fp_x2)
                dist_y = max(fp_y1 - hy, 0, hy - fp_y2)
                gap = (dist_x ** 2 + dist_y ** 2) ** 0.5
                issues.append(IntegrityIssue(
                    severity="critical",
                    category="mounting_hole",
                    ref=fp.ref,
                    message=(
                        f"{fp.ref}: overlaps or too close to {hole.ref} "
                        f"(gap={gap:.1f}mm, need {_MOUNTING_HOLE_CLEARANCE_MM}mm)"
                    ),
                ))

    return issues


# ---------------------------------------------------------------------------
# Check 8: Connector edge placement — THT connectors must be at board edge
# ---------------------------------------------------------------------------

_CONNECTOR_EDGE_MAX_MM = 5.0  # max distance from nearest board edge


def _check_connector_edge_placement(
    pcb: PCBDesign,
) -> list[IntegrityIssue]:
    """Verify THT connectors are placed at the board edge.

    Connectors floating in the middle of a board are useless for wiring.
    Pin headers / terminal blocks must have pads within 5mm of a board edge.
    """
    issues: list[IntegrityIssue] = []

    # Determine board outline bounds
    if not pcb.outline or not pcb.outline.polygon:
        return issues
    bx = [p.x for p in pcb.outline.polygon]
    by = [p.y for p in pcb.outline.polygon]
    board_x1, board_x2 = min(bx), max(bx)
    board_y1, board_y2 = min(by), max(by)

    for fp in pcb.footprints:
        if not fp.pads:
            continue
        # Only check THT connectors (J/P refs with thru_hole pads)
        ref_prefix = fp.ref.rstrip("0123456789")
        if ref_prefix not in ("J", "P"):
            continue
        has_tht = any(p.pad_type == "thru_hole" for p in fp.pads)
        if not has_tht:
            continue

        # Nearest edge distance from footprint center
        cx, cy = fp.position.x, fp.position.y
        dist_left = cx - board_x1
        dist_right = board_x2 - cx
        dist_top = cy - board_y1
        dist_bottom = board_y2 - cy
        nearest_edge = min(dist_left, dist_right, dist_top, dist_bottom)

        if nearest_edge > _CONNECTOR_EDGE_MAX_MM:
            issues.append(IntegrityIssue(
                severity="major",
                category="connector_edge",
                ref=fp.ref,
                message=(
                    f"{fp.ref}: THT connector {nearest_edge:.1f}mm from nearest "
                    f"board edge (max {_CONNECTOR_EDGE_MAX_MM}mm)"
                ),
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 9a: Component body overhang — 3D bodies must not extend past board edge
# ---------------------------------------------------------------------------

# Estimated body overhang beyond pad field for common component types (mm).
# IC pad rows extend BEYOND the body (gull-wing leads stick out), so IC
# overhang is 0 — the pad bbox already covers the body.  Connectors and
# relays have plastic housings that extend past the pads.
_BODY_OVERHANG_MM: dict[str, float] = {
    "J": 3.0,   # terminal blocks, pin headers — plastic body extends beyond pads
    "P": 3.0,   # connectors
    "K": 2.0,   # relays — body extends beyond pads
    "U": 0.0,   # ICs — pads extend BEYOND body (gull-wing leads)
    "SW": 0.5,  # switches — slight body overhang
    "R": 0.0,   # passives — body matches pad field
    "C": 0.0,
    "L": 0.0,
    "D": 0.0,
    "FB": 0.0,
    "default": 0.2,
}


def _check_body_overhang(
    pcb: PCBDesign,
) -> list[IntegrityIssue]:
    """Verify no component 3D body extends past the board outline.

    Pad-extent checks miss connector bodies that overhang the board edge.
    A terminal block's plastic housing is much larger than its pad field.
    """
    issues: list[IntegrityIssue] = []

    if not pcb.outline or not pcb.outline.polygon:
        return issues

    bx = [p.x for p in pcb.outline.polygon]
    by = [p.y for p in pcb.outline.polygon]
    board_x1, board_x2 = min(bx), max(bx)
    board_y1, board_y2 = min(by), max(by)

    for fp in pcb.footprints:
        if not fp.pads or fp.ref.startswith("H"):
            continue

        # Skip components whose pad center is already far outside the board
        # — that's an off-board placement issue, not a body overhang issue.
        pad_cx = sum(p.position.x for p in fp.pads) / len(fp.pads) + fp.position.x
        pad_cy = sum(p.position.y for p in fp.pads) / len(fp.pads) + fp.position.y
        margin = 10.0  # generous margin for edge components
        if (pad_cx < board_x1 - margin or pad_cx > board_x2 + margin
                or pad_cy < board_y1 - margin or pad_cy > board_y2 + margin):
            continue  # off-board placement — checked by other rules

        prefix = fp.ref.rstrip("0123456789")
        overhang = _BODY_OVERHANG_MM.get(prefix, _BODY_OVERHANG_MM["default"])

        # Compute pad bounding box in board coords
        rot_rad = math.radians(fp.rotation)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        ox, oy = fp.position.x, fp.position.y

        body_xs: list[float] = []
        body_ys: list[float] = []
        for p in fp.pads:
            rx = p.position.x * cos_r - p.position.y * sin_r
            ry = p.position.x * sin_r + p.position.y * cos_r
            half = max(p.size_x, p.size_y) / 2.0 + overhang
            body_xs.extend([ox + rx - half, ox + rx + half])
            body_ys.extend([oy + ry - half, oy + ry + half])

        if not body_xs:
            continue

        body_x1, body_x2 = min(body_xs), max(body_xs)
        body_y1, body_y2 = min(body_ys), max(body_ys)

        # Check each edge
        overhangs: list[str] = []
        if body_x1 < board_x1 - 0.5:
            overhangs.append(f"left by {board_x1 - body_x1:.1f}mm")
        if body_x2 > board_x2 + 0.5:
            overhangs.append(f"right by {body_x2 - board_x2:.1f}mm")
        if body_y1 < board_y1 - 0.5:
            overhangs.append(f"top by {board_y1 - body_y1:.1f}mm")
        if body_y2 > board_y2 + 0.5:
            overhangs.append(f"bottom by {body_y2 - board_y2:.1f}mm")

        if overhangs:
            # Connectors are designed to overhang (USB, RJ45, terminal
            # blocks, pin headers).  Only non-connector overhang is critical.
            sev = "minor" if prefix in ("J", "P", "K") else "critical"
            issues.append(IntegrityIssue(
                severity=sev,
                category="body_overhang",
                ref=fp.ref,
                message=f"{fp.ref}: body extends past board edge: {', '.join(overhangs)}",
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 9b: Passive rotation consistency within groups
# ---------------------------------------------------------------------------

def _check_passive_rotation_consistency(
    pcb: PCBDesign,
) -> list[IntegrityIssue]:
    """Flag passives at inconsistent rotations within 15mm of each other.

    Passives near each other should share a common rotation (0/180 or 90/270)
    for clean, professional appearance and easier assembly.
    """
    issues: list[IntegrityIssue] = []

    passives: list[Footprint] = []
    for fp in pcb.footprints:
        if not fp.pads or fp.ref.startswith("H"):
            continue
        prefix = fp.ref.rstrip("0123456789")
        if prefix in ("R", "C", "L", "D", "FB"):
            passives.append(fp)

    # Group passives by proximity (within 15mm)
    checked: set[str] = set()
    for fp_a in passives:
        if fp_a.ref in checked:
            continue
        # Find nearby passives
        cluster = [fp_a]
        for fp_b in passives:
            if fp_b.ref == fp_a.ref:
                continue
            dist = math.sqrt(
                (fp_a.position.x - fp_b.position.x) ** 2
                + (fp_a.position.y - fp_b.position.y) ** 2
            )
            if dist < 15.0:
                cluster.append(fp_b)

        if len(cluster) < 3:
            continue

        # Check rotation consistency: normalize to 0-180 range
        rots = set()
        for fp in cluster:
            r = fp.rotation % 180
            rots.add(round(r / 45) * 45)  # bucket to nearest 45°

        if len(rots) > 2:
            refs = [fp.ref for fp in cluster[:5]]
            checked.update(fp.ref for fp in cluster)
            issues.append(IntegrityIssue(
                severity="minor",
                category="rotation_consistency",
                ref=refs[0],
                message=(
                    f"Passives {', '.join(refs)} have {len(rots)} different rotations "
                    f"— should be consistent within a group"
                ),
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 9c: Component body-to-body collisions (3D overlap)
# ---------------------------------------------------------------------------

_BODY_COLLISION_GAP_MM = 0.3  # minimum gap between component bodies


def _check_body_collisions(
    pcb: PCBDesign,
) -> list[IntegrityIssue]:
    """Detect components whose 3D bodies physically overlap.

    The scoring system checks courtyard overlap (pad-based) but misses
    3D body collisions where bodies extend beyond pads.  A diode touching
    an IC body is a manufacturing and reliability failure.
    """
    issues: list[IntegrityIssue] = []

    # Build body bounding boxes (pad bbox + body overhang estimate)
    # Board extent for off-board filtering
    bx = [p.x for p in pcb.outline.polygon] if pcb.outline and pcb.outline.polygon else []
    by = [p.y for p in pcb.outline.polygon] if pcb.outline and pcb.outline.polygon else []
    bx1, bx2 = (min(bx), max(bx)) if bx else (0, 1000)
    by1, by2 = (min(by), max(by)) if by else (0, 1000)

    body_boxes: list[tuple[str, float, float, float, float]] = []
    for fp in pcb.footprints:
        if not fp.pads or fp.ref.startswith("H"):
            continue
        # Skip off-board components
        pc_x = sum(p.position.x for p in fp.pads) / len(fp.pads) + fp.position.x
        pc_y = sum(p.position.y for p in fp.pads) / len(fp.pads) + fp.position.y
        if (pc_x < bx1 - 10 or pc_x > bx2 + 10 or pc_y < by1 - 10 or pc_y > by2 + 10):
            continue
        prefix = fp.ref.rstrip("0123456789")
        overhang = _BODY_OVERHANG_MM.get(prefix, _BODY_OVERHANG_MM["default"])

        rot_rad = math.radians(fp.rotation)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        ox, oy = fp.position.x, fp.position.y

        xs: list[float] = []
        ys: list[float] = []
        for p in fp.pads:
            rx = p.position.x * cos_r - p.position.y * sin_r
            ry = p.position.x * sin_r + p.position.y * cos_r
            half = max(p.size_x, p.size_y) / 2.0 + overhang
            xs.extend([ox + rx - half, ox + rx + half])
            ys.extend([oy + ry - half, oy + ry + half])

        if xs:
            body_boxes.append((fp.ref, min(xs), min(ys), max(xs), max(ys)))

    # Pairwise overlap check
    checked: set[tuple[str, str]] = set()
    for i, (ref_a, ax1, ay1, ax2, ay2) in enumerate(body_boxes):
        for ref_b, bx1, by1, bx2, by2 in body_boxes[i + 1:]:
            pair = (min(ref_a, ref_b), max(ref_a, ref_b))
            if pair in checked:
                continue
            checked.add(pair)

            # AABB overlap with gap
            gap = _BODY_COLLISION_GAP_MM
            if ax1 - gap < bx2 and ax2 + gap > bx1 and ay1 - gap < by2 and ay2 + gap > by1:
                overlap_x = min(ax2, bx2) - max(ax1, bx1)
                overlap_y = min(ay2, by2) - max(ay1, by1)
                if overlap_x > 0 and overlap_y > 0:
                    # Small overlaps near connectors or between small passives
                    # are major, not critical — body estimates use pad bbox +
                    # overhang which overestimates for small components.
                    is_connector = (ref_a.startswith(("J", "P", "K"))
                                    or ref_b.startswith(("J", "P", "K")))
                    small = overlap_x < 2.0 and overlap_y < 2.0
                    # Marginal: one dimension < 1.5mm indicates near-miss,
                    # not a solid stack — downgrade for small passives.
                    marginal = min(overlap_x, overlap_y) < 1.5
                    is_small_passive = all(
                        r.rstrip("0123456789") in ("Q", "D", "R", "C", "L")
                        for r in (ref_a, ref_b)
                    )
                    sev = ("major" if (is_connector or small
                                       or (marginal and is_small_passive))
                           else "critical")
                    issues.append(IntegrityIssue(
                        severity=sev,
                        category="body_collision",
                        ref=f"{ref_a}/{ref_b}",
                        message=(
                            f"{ref_a} and {ref_b}: 3D bodies overlap "
                            f"by {overlap_x:.1f}x{overlap_y:.1f}mm"
                        ),
                    ))

    return issues


# ---------------------------------------------------------------------------
# Check 10: Board utilization — components must fill the board, not float
# ---------------------------------------------------------------------------

_MIN_UTILIZATION_PCT = 25.0  # minimum component-spread / board-area ratio


def _check_board_utilization(
    pcb: PCBDesign,
) -> list[IntegrityIssue]:
    """Verify components fill a reasonable fraction of the board.

    A 60x40mm board with 15 components clustered in one corner is oversized.
    Either shrink the board or spread components to fill the space.
    """
    issues: list[IntegrityIssue] = []

    if not pcb.outline or not pcb.outline.polygon:
        return issues

    bx = [p.x for p in pcb.outline.polygon]
    by = [p.y for p in pcb.outline.polygon]
    board_area = (max(bx) - min(bx)) * (max(by) - min(by))
    if board_area < 1.0:
        return issues

    # Compute total component footprint area (sum of pad bounding boxes)
    total_comp_area = 0.0
    for fp in pcb.footprints:
        if not fp.pads or fp.ref.startswith("H"):
            continue
        xs = [p.position.x for p in fp.pads]
        ys = [p.position.y for p in fp.pads]
        w = max(xs) - min(xs) + max(p.size_x for p in fp.pads)
        h = max(ys) - min(ys) + max(p.size_y for p in fp.pads)
        total_comp_area += w * h

    fill_ratio = total_comp_area / board_area * 100.0

    if fill_ratio < _MIN_UTILIZATION_PCT:
        issues.append(IntegrityIssue(
            severity="major",
            category="utilization",
            ref="board",
            message=(
                f"Board utilization {fill_ratio:.0f}% (min {_MIN_UTILIZATION_PCT:.0f}%) "
                f"-- board is oversized for "
                f"{len([f for f in pcb.footprints if f.pads and not f.ref.startswith('H')])} "
                f"components "
                f"or components are too spread out"
            ),
        ))

    return issues


# ---------------------------------------------------------------------------
# Check 10: Orphan passives — every R/C/L/D must be near an IC
# ---------------------------------------------------------------------------

_ORPHAN_MAX_DISTANCE_MM = 10.0  # max distance from passive to nearest IC


def _check_orphan_passives(
    pcb: PCBDesign,
) -> list[IntegrityIssue]:
    """Verify every passive component is near at least one IC.

    Isolated passives floating between groups indicate broken placement.
    Every R, C, L, D should be within 15mm of an IC it's connected to.
    """
    issues: list[IntegrityIssue] = []

    # Find all IC positions
    ic_positions: list[tuple[float, float]] = []
    for fp in pcb.footprints:
        if not fp.pads:
            continue
        prefix = fp.ref.rstrip("0123456789")
        if prefix in ("U", "IC", "K"):
            ic_positions.append((fp.position.x, fp.position.y))

    if not ic_positions:
        return issues

    # Check each passive
    for fp in pcb.footprints:
        if not fp.pads or fp.ref.startswith("H"):
            continue
        prefix = fp.ref.rstrip("0123456789")
        if prefix not in ("R", "C", "L", "D", "FB"):
            continue

        # Find nearest IC
        min_dist = min(
            math.sqrt((fp.position.x - ix) ** 2 + (fp.position.y - iy) ** 2)
            for ix, iy in ic_positions
        )

        if min_dist > _ORPHAN_MAX_DISTANCE_MM:
            issues.append(IntegrityIssue(
                severity="major",
                category="orphan",
                ref=fp.ref,
                message=(
                    f"{fp.ref}: nearest IC is {min_dist:.0f}mm away "
                    f"(max {_ORPHAN_MAX_DISTANCE_MM:.0f}mm) — orphan component?"
                ),
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 11: Power chain compactness — connected stages shouldn't be far apart
# ---------------------------------------------------------------------------

_POWER_CHAIN_MAX_GAP_MM = 20.0  # max gap between regulator ICs on same chain


def _check_power_chain_gap(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[IntegrityIssue]:
    """Verify regulator ICs on the same power chain are reasonably close.

    A buck converter feeding an LDO with a 30mm gap creates long power traces,
    voltage drop, and wasted board space.
    """
    issues: list[IntegrityIssue] = []

    # Find regulator ICs (U refs that have power-related values)
    _pwr_kw = ("tps", "ams", "lm", "ldo", "buck", "boost", "reg", "117", "78", "33")
    reg_fps: list[Footprint] = []
    for fp in pcb.footprints:
        if not fp.pads or not fp.ref.startswith("U"):
            continue
        val_lower = (fp.value or "").lower()
        if any(kw in val_lower for kw in _pwr_kw):
            reg_fps.append(fp)

    # Check pairwise distances between regulators
    for i, fp_a in enumerate(reg_fps):
        for fp_b in reg_fps[i + 1:]:
            dist = math.sqrt(
                (fp_a.position.x - fp_b.position.x) ** 2
                + (fp_a.position.y - fp_b.position.y) ** 2
            )
            if dist > _POWER_CHAIN_MAX_GAP_MM:
                issues.append(IntegrityIssue(
                    severity="major",
                    category="power_chain",
                    ref=f"{fp_a.ref}/{fp_b.ref}",
                    message=(
                        f"{fp_a.ref} ({fp_a.value}) to {fp_b.ref} ({fp_b.value}): "
                        f"{dist:.0f}mm apart (max {_POWER_CHAIN_MAX_GAP_MM:.0f}mm)"
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
    issues.extend(_check_3d_alignment(pcb))
    issues.extend(_check_keepout_positions(pcb))
    issues.extend(_check_duplicate_pads(pcb))
    issues.extend(_check_mounting_hole_clearance(pcb))
    issues.extend(_check_connector_edge_placement(pcb))
    issues.extend(_check_body_overhang(pcb))
    issues.extend(_check_body_collisions(pcb))
    issues.extend(_check_passive_rotation_consistency(pcb))
    issues.extend(_check_board_utilization(pcb))
    issues.extend(_check_orphan_passives(pcb))
    issues.extend(_check_power_chain_gap(pcb, requirements))

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


def assert_renders_exist(
    pcb_path: str | Path,
    max_age_seconds: float = 300.0,
) -> dict[str, Path]:
    """Assert that fresh render PNGs exist for a PCB file.

    Checks for 2D and 3D renders in the same directory as ``pcb_path``,
    or in a ``placement_renders`` subdirectory.  Raises ``AssertionError``
    if no fresh renders are found.

    Args:
        pcb_path: Path to the .kicad_pcb file.
        max_age_seconds: Maximum age in seconds for renders to be
            considered fresh (default: 5 minutes).

    Returns:
        Dict mapping view name to Path for found renders.

    Raises:
        AssertionError: If no fresh renders exist.
    """
    import time
    from pathlib import Path as _Path

    pcb = _Path(pcb_path)
    now = time.time()

    search_dirs = [pcb.parent, pcb.parent / "placement_renders"]
    render_patterns = {
        "2d": "*2d*.png",
        "3d_top": "*3d_top*.png",
        "3d_iso": "*3d_iso*.png",
        "3d_isoback": "*3d_isoback*.png",
    }

    found: dict[str, _Path] = {}
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for view_name, pattern in render_patterns.items():
            if view_name in found:
                continue
            for p in search_dir.glob(pattern):
                if p.is_file() and (now - p.stat().st_mtime) < max_age_seconds:
                    found[view_name] = p
                    break

    if not found:
        msg = (
            f"No fresh render PNGs found for {pcb.name}. "
            f"Searched: {[str(d) for d in search_dirs]}. "
            f"The placement optimizer should produce these automatically. "
            f"Max age: {max_age_seconds}s."
        )
        raise AssertionError(msg)

    _log.info(
        "Render evidence: %d views found for %s: %s",
        len(found), pcb.name, list(found.keys()),
    )
    return found
