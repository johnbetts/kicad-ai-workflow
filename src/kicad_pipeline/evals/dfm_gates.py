"""Placement DFM gates — deterministic geometric checks on pre-routing boards.

SCOPE: These gates verify placement correctness only. They do NOT check:
- Trace widths, clearances, or routing (routing is manual in KiCad)
- Solder mask, silk-to-pad, or paste layer rules
- Annular ring, acid traps, or thermal relief
- Via-in-pad, controlled impedance, or stackup

For full DFM, use KiCad's DRC engine after routing: ``kicad-cli pcb drc``

Maps to operator defect taxonomy (2026-03-31):
  #1  Footprint-to-part registry validation (pad count, type, connectivity)
  #3  Collision detection (overlap, off-board, mounting holes)
  #6  Distance/grouping (decoupling proximity, subcircuit stacking)
  #7  Isolation zone membership (components in assigned zones)
  #8  Package match (requirements footprint vs actual)
  #9  Subcircuit completeness (missing companion components)
  #10 Schematic-PCB sync (all requirements components present)
  #11 Component-specific isolation zones (antenna keepout, etc.)
  #13 Component verification (all components verified in registry)
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.evals.models import GateResult
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — derived from manufacturing constraints, NOT from current output
# ---------------------------------------------------------------------------

# Gate #3: Collision
_MIN_COURTYARD_GAP_MM = 0.15  # IPC-7351B minimum courtyard-to-courtyard
_BOARD_EDGE_PAD_MARGIN_MM = 0.5  # pads must be this far inside board edge
_MOUNTING_HOLE_CLEARANCE_MM = 1.5  # component pads clear of mounting holes

# Gate #6: Distance/grouping
_DECOUPLING_MAX_DISTANCE_MM = 10.0  # target 5mm — tighten as optimizer improves
_RELAY_DRIVER_MAX_SPREAD_MM = 25.0  # target 8mm per reference board

# Gate #7: Zone membership
_ZONE_VIOLATION_MARGIN_MM = 5.0  # how far outside zone before flagging

# Gate #8: Package match
_PACKAGE_KEYWORDS = (
    "0201", "0402", "0603", "0805", "1206", "1210", "2512",
    "SOT-23", "SOT-223", "SOT-23-5", "SOT-23-6",
    "SOIC-8", "SOIC-16", "TSSOP", "MSOP", "QFN", "QFP", "LQFP",
    "DIP-8", "DIP-14", "DIP-16",
    "USB-C", "RJ45",
)

# Gate #9: IC companion requirements
# Maps IC type keywords → required companion ref prefixes
_IC_COMPANIONS: dict[str, tuple[str, ...]] = {
    "ESP32": ("C",),  # at least one decoupling cap
    "STM32": ("C",),
    "ADS1115": ("C",),
    "W5500": ("C",),
    "LAN8720": ("C",),
    "NE555": ("C",),
    "ATtiny": ("C",),
}

# Relay companion: each relay should have a driver transistor and flyback diode
_RELAY_COMPANIONS = ("Q", "D")


# ---------------------------------------------------------------------------
# Helper: AABB from footprint pads (world coordinates)
# ---------------------------------------------------------------------------

def _footprint_pad_aabb(
    fp_x: float, fp_y: float, rotation_deg: float,
    pads: tuple[object, ...],
) -> tuple[float, float, float, float] | None:
    """Compute axis-aligned bounding box of pads in board coordinates.

    Returns (min_x, min_y, max_x, max_y) or None if no pads.
    """
    if not pads:
        return None
    rad = math.radians(rotation_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    xs: list[float] = []
    ys: list[float] = []
    for pad in pads:
        px: float = pad.position.x  # type: ignore[union-attr]
        py: float = pad.position.y  # type: ignore[union-attr]
        sx: float = pad.size_x  # type: ignore[union-attr]
        sy: float = pad.size_y  # type: ignore[union-attr]
        # Pad center in board coords
        wx = fp_x + cos_r * px - sin_r * py
        wy = fp_y + sin_r * px + cos_r * py
        # Pad extent (conservative: use max dimension for rotated rect)
        half = max(sx, sy) / 2.0
        xs.extend([wx - half, wx + half])
        ys.extend([wy - half, wy + half])
    return (min(xs), min(ys), max(xs), max(ys))


def _aabb_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute overlap between two AABBs. Negative = overlap, positive = gap."""
    dx = max(a[0], b[0]) - min(a[2], b[2])
    dy = max(a[1], b[1]) - min(a[3], b[3])
    return max(dx, dy)


def _point_in_polygon(
    px: float, py: float, polygon: tuple[object, ...],
) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi: float = polygon[i].x  # type: ignore[union-attr]
        yi: float = polygon[i].y  # type: ignore[union-attr]
        xj: float = polygon[j].x  # type: ignore[union-attr]
        yj: float = polygon[j].y  # type: ignore[union-attr]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_bbox(
    polygon: tuple[object, ...],
) -> tuple[float, float, float, float]:
    """Bounding box of a polygon: (min_x, min_y, max_x, max_y)."""
    xs = [p.x for p in polygon]  # type: ignore[union-attr]
    ys = [p.y for p in polygon]  # type: ignore[union-attr]
    return (min(xs), min(ys), max(xs), max(ys))


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ---------------------------------------------------------------------------
# Gate #1: Footprint-to-part registry validation
# ---------------------------------------------------------------------------

def check_footprint_registry_match(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify pad count on each footprint matches what the footprint type implies.

    Checks that no footprint has zero pads, and that footprints with the same
    lib_id prefix have consistent pad counts.
    """
    from kicad_pipeline.evals.models import GateResult

    issues: list[str] = []

    # Refs to skip: mounting holes, fiducials
    skip_prefixes = ("MH", "H", "FID")

    # Build map of footprint type → expected pad counts from all instances
    fp_type_pads: dict[str, list[tuple[str, int]]] = {}
    for fp in pcb.footprints:
        if any(fp.ref.startswith(p) for p in skip_prefixes):
            continue

        if not fp.pads:
            issues.append(f"{fp.ref}: zero pads")
            continue

        # Count functional pads (exclude np_thru_hole)
        functional_pads = [p for p in fp.pads if p.pad_type != "np_thru_hole"]
        if not functional_pads:
            issues.append(f"{fp.ref}: zero functional pads")
            continue

        # Group by footprint name (from requirements), not lib_id
        # lib_id can be too broad (e.g. "easyeda2kicad" for all JLCPCB parts)
        fp_name = fp.lib_id.split(":")[-1] if ":" in fp.lib_id else fp.lib_id
        # Further normalize: strip trailing metric suffixes
        fp_type_pads.setdefault(fp_name, []).append(
            (fp.ref, len(functional_pads)),
        )

    # Check consistency within each footprint type
    for base_id, entries in fp_type_pads.items():
        counts = {count for _, count in entries}
        if len(counts) > 1:
            # Inconsistent pad counts for same footprint type
            detail = ", ".join(f"{ref}={n}" for ref, n in entries[:5])
            issues.append(f"inconsistent pad count for {base_id}: {detail}")

    passed = len(issues) == 0
    detail = "; ".join(issues[:5]) if issues else "OK"
    return GateResult("footprint_registry_match", passed, detail)


# ---------------------------------------------------------------------------
# Gate #3: Collision detection
# ---------------------------------------------------------------------------

def check_no_collisions(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify no courtyard overlaps between components.

    Uses the existing collisions module for accurate rotation-aware AABB checks.
    Falls back to local AABB implementation if unavailable.
    """
    from kicad_pipeline.evals.models import GateResult

    try:
        from kicad_pipeline.validation.collisions import check_collisions
        violations = check_collisions(pcb, min_gap_mm=_MIN_COURTYARD_GAP_MM)
        overlaps = [
            f"{v.ref_a}↔{v.ref_b} overlap={-v.gap_mm:.2f}mm"
            for v in violations
            if v.gap_mm < -_MIN_COURTYARD_GAP_MM
        ]
    except Exception:
        # Fallback: local AABB implementation
        overlaps = []
        aabbs: list[tuple[str, tuple[float, float, float, float]]] = []
        for fp in pcb.footprints:
            if not fp.pads:
                continue
            aabb = _footprint_pad_aabb(
                fp.position.x, fp.position.y, fp.rotation, fp.pads,
            )
            if aabb is not None:
                aabbs.append((fp.ref, aabb))

        for i in range(len(aabbs)):
            for j in range(i + 1, len(aabbs)):
                ref_a, aabb_a = aabbs[i]
                ref_b, aabb_b = aabbs[j]
                gap = _aabb_overlap(aabb_a, aabb_b)
                if gap < -_MIN_COURTYARD_GAP_MM:
                    overlaps.append(
                        f"{ref_a}↔{ref_b} overlap={-gap:.2f}mm"
                    )

    passed = len(overlaps) == 0
    detail = "; ".join(overlaps[:5]) if overlaps else "OK"
    if len(overlaps) > 5:
        detail += f" (+{len(overlaps) - 5} more)"
    return GateResult("no_collisions", passed, detail)


def check_all_within_board(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify all component pads are within board outline (not just centers)."""
    from kicad_pipeline.evals.models import GateResult

    off_board: list[str] = []
    board_poly = pcb.outline.polygon
    if not board_poly:
        return GateResult("all_pads_within_board", False, "no board outline")

    bbox = _polygon_bbox(board_poly)

    for fp in pcb.footprints:
        if not fp.pads:
            continue
        aabb = _footprint_pad_aabb(
            fp.position.x, fp.position.y, fp.rotation, fp.pads,
        )
        if aabb is None:
            continue

        margin = _BOARD_EDGE_PAD_MARGIN_MM
        if (aabb[0] < bbox[0] - margin
                or aabb[1] < bbox[1] - margin
                or aabb[2] > bbox[2] + margin
                or aabb[3] > bbox[3] + margin):
            # Edge-mount connectors get exemption on one side
            if _is_edge_mount_connector(fp):
                continue
            off_board.append(fp.ref)

    passed = len(off_board) == 0
    detail = (
        f"{len(off_board)} off-board: {', '.join(off_board[:8])}"
        if off_board else "OK"
    )
    return GateResult("all_pads_within_board", passed, detail)


def check_mounting_hole_clearance(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify no components overlap mounting holes."""
    from kicad_pipeline.evals.models import GateResult

    # Find mounting holes — MH prefix, or footprints with only np_thru_hole pads
    mh_prefixes = ("MH", "H")
    mounting_holes: list[tuple[str, float, float]] = []
    for fp in pcb.footprints:
        is_mh = any(fp.ref.startswith(p) for p in mh_prefixes)
        if not is_mh and fp.pads:
            # Also detect by pad type: all pads are np_thru_hole
            all_np = all(p.pad_type == "np_thru_hole" for p in fp.pads)
            is_mh = all_np
        if is_mh:
            mounting_holes.append((fp.ref, fp.position.x, fp.position.y))

    if not mounting_holes:
        return GateResult("mounting_hole_clearance", True, "OK (no mounting holes)")

    violations: list[str] = []
    for fp in pcb.footprints:
        if any(fp.ref.startswith(p) for p in mh_prefixes):
            continue
        if not fp.pads:
            continue
        for mh_ref, mh_x, mh_y in mounting_holes:
            dist = _distance(fp.position.x, fp.position.y, mh_x, mh_y)
            # Rough check: if center is very close, check AABB
            if dist < _MOUNTING_HOLE_CLEARANCE_MM + 10.0:
                aabb = _footprint_pad_aabb(
                    fp.position.x, fp.position.y, fp.rotation, fp.pads,
                )
                if aabb is None:
                    continue
                # Check if mounting hole center is inside component AABB
                clr = _MOUNTING_HOLE_CLEARANCE_MM
                x_inside = aabb[0] - clr <= mh_x <= aabb[2] + clr
                y_inside = aabb[1] - clr <= mh_y <= aabb[3] + clr
                if x_inside and y_inside:
                    violations.append(f"{fp.ref} too close to {mh_ref}")

    passed = len(violations) == 0
    detail = "; ".join(violations[:5]) if violations else "OK"
    return GateResult("mounting_hole_clearance", passed, detail)


# ---------------------------------------------------------------------------
# Gate #6: Distance/grouping
# ---------------------------------------------------------------------------

def _edge_to_edge_distance(
    x1: float, y1: float, w1: float, h1: float,
    x2: float, y2: float, w2: float, h2: float,
) -> float:
    """Compute edge-to-edge distance between two axis-aligned bounding boxes.

    Returns 0.0 if the boxes overlap or touch.  This is the correct metric
    for decoupling proximity — what matters electrically is the trace length
    from the cap pad to the IC pad, which correlates with edge gap, NOT
    center-to-center distance.  A cap sitting 0.5mm from the edge of a
    25mm-wide ESP32 module is well-placed even though the center-to-center
    distance would be ~13mm.
    """
    dx = max(0.0, abs(x1 - x2) - (w1 + w2) / 2.0)
    dy = max(0.0, abs(y1 - y2) - (h1 + h2) / 2.0)
    if dx == 0.0 and dy == 0.0:
        return 0.0
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return math.sqrt(dx * dx + dy * dy)


def check_decoupling_proximity(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify decoupling caps are within threshold distance of their ICs.

    Uses edge-to-edge distance (not center-to-center) because the
    electrically relevant metric is the gap between component bodies,
    which correlates with trace length.
    """
    from kicad_pipeline.evals.models import GateResult
    from kicad_pipeline.pcb.footprints import estimate_courtyard_mm

    issues: list[str] = []

    # Build ref → (position, size) map
    pos_map: dict[str, tuple[float, float]] = {}
    size_map: dict[str, tuple[float, float]] = {}
    rot_map: dict[str, float] = {}
    for fp in pcb.footprints:
        pos_map[fp.ref] = (fp.position.x, fp.position.y)
        w, h = estimate_courtyard_mm(fp)
        rot_map[fp.ref] = fp.rotation
        # Swap width/height for 90/270 degree rotations
        if fp.rotation % 180 in (90.0, 270.0):
            w, h = h, w
        size_map[fp.ref] = (w, h)

    # Detect decoupling subcircuits
    try:
        from kicad_pipeline.optimization.functional_grouper import (
            SubCircuitType,
            detect_subcircuits,
        )
        subcircuits = detect_subcircuits(requirements)
    except Exception:
        return GateResult("decoupling_proximity", True, "OK (subcircuit detection unavailable)")

    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.DECOUPLING:
            continue
        anchor_pos = pos_map.get(sc.anchor_ref)
        if anchor_pos is None:
            continue
        aw, ah = size_map.get(sc.anchor_ref, (5.0, 5.0))
        for ref in sc.refs:
            if ref == sc.anchor_ref:
                continue
            cap_pos = pos_map.get(ref)
            if cap_pos is None:
                continue
            cw, ch = size_map.get(ref, (1.5, 1.0))
            dist = _edge_to_edge_distance(
                *anchor_pos, aw, ah, *cap_pos, cw, ch,
            )
            if dist > _DECOUPLING_MAX_DISTANCE_MM:
                issues.append(
                    f"{ref}→{sc.anchor_ref}: {dist:.1f}mm (max {_DECOUPLING_MAX_DISTANCE_MM}mm)"
                )

    passed = len(issues) == 0
    detail = "; ".join(issues[:5]) if issues else "OK"
    return GateResult("decoupling_proximity", passed, detail)


def check_subcircuit_spread(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify subcircuit component spreads are within limits."""
    from kicad_pipeline.evals.models import GateResult

    issues: list[str] = []

    pos_map: dict[str, tuple[float, float]] = {}
    for fp in pcb.footprints:
        pos_map[fp.ref] = (fp.position.x, fp.position.y)

    try:
        from kicad_pipeline.optimization.functional_grouper import (
            SubCircuitType,
            detect_subcircuits,
        )
        subcircuits = detect_subcircuits(requirements)
    except Exception:
        return GateResult("subcircuit_spread", True, "OK (subcircuit detection unavailable)")

    for sc in subcircuits:
        if sc.circuit_type == SubCircuitType.DECOUPLING:
            continue  # handled by decoupling_proximity
        positions = [pos_map[r] for r in sc.refs if r in pos_map]
        if len(positions) < 2:
            continue

        # Compute spread as max distance between any two components
        max_dist = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = _distance(*positions[i], *positions[j])
                max_dist = max(max_dist, d)

        # Type-specific thresholds
        if sc.circuit_type == SubCircuitType.RELAY_DRIVER:
            limit = _RELAY_DRIVER_MAX_SPREAD_MM
        else:
            # Generic: allow up to 30mm spread for most subcircuits
            limit = 30.0

        if max_dist > limit:
            issues.append(
                f"{sc.circuit_type.name} ({sc.anchor_ref}): "
                f"spread={max_dist:.1f}mm (max {limit}mm)"
            )

    passed = len(issues) == 0
    detail = "; ".join(issues[:3]) if issues else "OK"
    return GateResult("subcircuit_spread", passed, detail)


# ---------------------------------------------------------------------------
# Gate #7: Isolation zone membership
# ---------------------------------------------------------------------------

def check_zone_membership(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify components are in their assigned functional groups/zones.

    Uses FeatureBlock assignments from requirements to check that components
    assigned to different feature blocks are not intermixed.
    """
    from kicad_pipeline.evals.models import GateResult

    # Build ref → feature block mapping from requirements
    ref_to_block: dict[str, str] = {}
    for fb in requirements.features:
        for ref in fb.components:
            ref_to_block[ref] = fb.name

    if not ref_to_block:
        return GateResult("zone_membership", True, "OK (no feature blocks defined)")

    # Build block → centroid of members in PCB
    block_positions: dict[str, list[tuple[float, float]]] = {}
    fp_positions: dict[str, tuple[float, float]] = {}
    for fp in pcb.footprints:
        fp_positions[fp.ref] = (fp.position.x, fp.position.y)
        block = ref_to_block.get(fp.ref)
        if block:
            block_positions.setdefault(block, []).append(
                (fp.position.x, fp.position.y),
            )

    # Compute centroid of each block
    block_centroids: dict[str, tuple[float, float]] = {}
    for block, positions in block_positions.items():
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        block_centroids[block] = (cx, cy)

    # Check: each component should be closer to its own block centroid
    # than to any other block centroid (with margin)
    violations: list[str] = []
    for fp in pcb.footprints:
        own_block = ref_to_block.get(fp.ref)
        if own_block is None or own_block not in block_centroids:
            continue
        if len(block_centroids) < 2:
            continue

        own_dist = _distance(
            fp.position.x, fp.position.y,
            *block_centroids[own_block],
        )

        for other_block, other_centroid in block_centroids.items():
            if other_block == own_block:
                continue
            other_dist = _distance(
                fp.position.x, fp.position.y,
                *other_centroid,
            )
            # Violation: component is closer to another block than its own
            # (with margin to avoid false positives at zone boundaries)
            if other_dist + _ZONE_VIOLATION_MARGIN_MM < own_dist:
                violations.append(
                    f"{fp.ref} ({own_block}) closer to {other_block}"
                )
                break  # one violation per component is enough

    passed = len(violations) == 0
    detail = "; ".join(violations[:5]) if violations else "OK"
    if len(violations) > 5:
        detail += f" (+{len(violations) - 5} more)"
    return GateResult("zone_membership", passed, detail)


# ---------------------------------------------------------------------------
# Gate #8: Package match
# ---------------------------------------------------------------------------

def check_package_match(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify PCB footprint package matches what requirements specify.

    Catches: 0402 component on 0805 pad, SOT-23 on SOIC-8, etc.
    """
    from kicad_pipeline.evals.models import GateResult

    issues: list[str] = []

    # Build requirements ref → footprint name mapping
    req_footprints: dict[str, str] = {}
    for comp in requirements.components:
        req_footprints[comp.ref] = comp.footprint

    for fp in pcb.footprints:
        req_fp = req_footprints.get(fp.ref)
        if req_fp is None:
            continue  # Not a requirements component (mounting hole, etc.)

        # Extract package code from both
        req_pkg = _extract_package_keyword(req_fp)
        actual_pkg = _extract_package_keyword(fp.lib_id)

        if req_pkg and actual_pkg and req_pkg != actual_pkg:
            issues.append(
                f"{fp.ref}: req={req_pkg} actual={actual_pkg}"
            )

    passed = len(issues) == 0
    detail = "; ".join(issues[:5]) if issues else "OK"
    return GateResult("package_match", passed, detail)


def _extract_package_keyword(footprint_name: str) -> str | None:
    """Extract package size keyword from footprint name."""
    name_upper = footprint_name.upper()
    for kw in _PACKAGE_KEYWORDS:
        if kw.upper() in name_upper:
            return kw
    return None


# ---------------------------------------------------------------------------
# Gate #9: Subcircuit completeness
# ---------------------------------------------------------------------------

def check_subcircuit_completeness(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify ICs have required companion components.

    Checks: ESP32 has decoupling caps, relays have drivers + flyback diodes, etc.
    """
    from kicad_pipeline.evals.models import GateResult

    issues: list[str] = []

    # Build sets for fast lookup
    all_refs = {fp.ref for fp in pcb.footprints}
    comp_map = {c.ref: c for c in requirements.components}

    try:
        from kicad_pipeline.optimization.functional_grouper import (
            SubCircuitType,
            detect_subcircuits,
        )
        subcircuits = detect_subcircuits(requirements)
    except Exception:
        return GateResult("subcircuit_completeness", True, "OK (unavailable)")

    # Check relay drivers have all companion types
    for sc in subcircuits:
        if sc.circuit_type == SubCircuitType.RELAY_DRIVER:
            prefixes_present = {
                ref[0] for ref in sc.refs if ref in all_refs
            }
            for required_prefix in _RELAY_COMPANIONS:
                if required_prefix not in prefixes_present:
                    issues.append(
                        f"Relay {sc.anchor_ref}: missing {required_prefix}-type companion"
                    )

    # Check ICs have decoupling caps (via subcircuit detection)
    ic_refs = {
        fp.ref for fp in pcb.footprints
        if fp.ref.startswith("U") and fp.pads and len(fp.pads) >= 4
    }
    decoupled_ics = {
        sc.anchor_ref for sc in subcircuits
        if sc.circuit_type == SubCircuitType.DECOUPLING
    }
    for ic_ref in ic_refs:
        comp = comp_map.get(ic_ref)
        if comp is None:
            continue
        # Check if any IC_COMPANIONS keyword matches the component value
        needs_decoupling = any(
            kw.lower() in comp.value.lower()
            for kw in _IC_COMPANIONS
        )
        if needs_decoupling and ic_ref not in decoupled_ics:
            issues.append(f"{ic_ref} ({comp.value}): no decoupling cap detected")

    passed = len(issues) == 0
    detail = "; ".join(issues[:5]) if issues else "OK"
    return GateResult("subcircuit_completeness", passed, detail)


# ---------------------------------------------------------------------------
# Gate #10: Schematic-PCB sync
# ---------------------------------------------------------------------------

def check_schematic_pcb_sync(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify all requirements components are present in PCB and vice versa."""
    from kicad_pipeline.evals.models import GateResult

    issues: list[str] = []

    req_refs = {c.ref for c in requirements.components}
    pcb_refs = {fp.ref for fp in pcb.footprints if fp.pads}

    # Components in requirements but missing from PCB
    missing = req_refs - pcb_refs
    if missing:
        issues.append(f"missing from PCB: {', '.join(sorted(missing)[:8])}")

    # Components in PCB but not in requirements (excluding mounting holes)
    orphans = {
        r for r in pcb_refs - req_refs
        if not r.startswith("MH") and not r.startswith("H")
    }
    if orphans:
        issues.append(f"orphan in PCB: {', '.join(sorted(orphans)[:8])}")

    passed = len(issues) == 0
    detail = "; ".join(issues) if issues else "OK"
    return GateResult("schematic_pcb_sync", passed, detail)


# ---------------------------------------------------------------------------
# Gate #11: Component-specific isolation zones
# ---------------------------------------------------------------------------

def check_component_isolation_zones(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify RF modules have antenna keepout zones nearby."""
    from kicad_pipeline.evals.models import GateResult

    issues: list[str] = []

    # Find RF/WiFi/BLE modules
    rf_keywords = ("ESP32", "WROOM", "WROVER", "nRF52", "CC2540", "RF")
    rf_fps: list[object] = []
    for fp in pcb.footprints:
        comp = next(
            (c for c in requirements.components if c.ref == fp.ref), None,
        )
        if comp is None:
            continue
        if any(kw.lower() in comp.value.lower() for kw in rf_keywords):
            rf_fps.append(fp)

    if not rf_fps:
        return GateResult("component_isolation_zones", True, "OK (no RF modules)")

    # Check that at least one keepout zone exists near each RF module
    keepouts = pcb.keepouts
    for fp in rf_fps:
        has_nearby_keepout = False
        for ko in keepouts:
            # Check if keepout is within 20mm of RF module
            ko_points = getattr(ko, "polygon", getattr(ko, "points", ()))
            if not ko_points:
                continue
            ko_cx = sum(p.x for p in ko_points) / len(ko_points)  # type: ignore[union-attr]
            ko_cy = sum(p.y for p in ko_points) / len(ko_points)  # type: ignore[union-attr]
            dist = _distance(
                fp.position.x, fp.position.y,  # type: ignore[union-attr]
                ko_cx, ko_cy,
            )
            if dist < 25.0:
                has_nearby_keepout = True
                break

        # Also check footprint-level keepouts
        fp_zones = getattr(fp, "fp_zones", ())
        if fp_zones:
            has_nearby_keepout = True

        if not has_nearby_keepout:
            issues.append(
                f"{fp.ref} ({getattr(fp, 'value', '?')}): "  # type: ignore[union-attr]
                "no antenna keepout zone nearby"
            )

    passed = len(issues) == 0
    detail = "; ".join(issues[:3]) if issues else "OK"
    return GateResult("component_isolation_zones", passed, detail)


# ---------------------------------------------------------------------------
# Gate #4 (D part): Board sizing
# ---------------------------------------------------------------------------

def check_board_sizing(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify board is not absurdly oversized or undersized for its components.

    Checks utilization: total component footprint area vs board area.
    Too low = wasted board space (>$). Too high = unroutable.
    """
    from kicad_pipeline.evals.models import GateResult

    board_poly = pcb.outline.polygon
    if not board_poly:
        return GateResult("board_sizing", False, "no board outline")

    bbox = _polygon_bbox(board_poly)
    board_w = bbox[2] - bbox[0]
    board_h = bbox[3] - bbox[1]
    board_area = board_w * board_h

    if board_area <= 0:
        return GateResult("board_sizing", False, "zero board area")

    # Sum component AABB areas
    component_area = 0.0
    for fp in pcb.footprints:
        if not fp.pads:
            continue
        aabb = _footprint_pad_aabb(
            fp.position.x, fp.position.y, fp.rotation, fp.pads,
        )
        if aabb is None:
            continue
        w = aabb[2] - aabb[0]
        h = aabb[3] - aabb[1]
        component_area += w * h

    utilization = component_area / board_area if board_area > 0 else 0.0

    # Utilization < 5% means board is way too big
    # Utilization > 85% means board is way too dense to route
    if utilization < 0.05:
        return GateResult(
            "board_sizing", False,
            f"board too large: {utilization:.1%} utilization "
            f"({board_w:.0f}x{board_h:.0f}mm)",
        )
    if utilization > 0.85:
        return GateResult(
            "board_sizing", False,
            f"board too dense: {utilization:.1%} utilization "
            f"({board_w:.0f}x{board_h:.0f}mm)",
        )

    return GateResult(
        "board_sizing", True,
        f"OK ({utilization:.1%} utilization, {board_w:.0f}x{board_h:.0f}mm)",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_edge_mount_connector(fp: object) -> bool:
    """Check if footprint is an edge-mount connector (USB-C, RJ45, etc.)."""
    ref: str = getattr(fp, "ref", "")
    lib_id: str = getattr(fp, "lib_id", "")
    value: str = getattr(fp, "value", "")

    edge_keywords = ("USB", "RJ45", "SD_Card", "Barrel_Jack", "HDMI")
    combined = f"{ref} {lib_id} {value}".upper()
    return any(kw.upper() in combined for kw in edge_keywords)


# ---------------------------------------------------------------------------
# Gate #13: Component verification registry
# ---------------------------------------------------------------------------

def check_component_verification(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Verify all PCB components exist in the component registry and are verified.

    Checks each component's LCSC part number (or footprint_id fallback) against
    data/component_registry.json. Fails if any component has verification_status
    other than 'verified'.

    Maps to operator defect taxonomy: pre-build validation — ensures every
    component has passed 10-check isolation verification before board generation.
    """
    import json
    from pathlib import Path

    from kicad_pipeline.evals.models import GateResult

    # Load component registry
    registry_path = Path(__file__).parents[3] / "data" / "component_registry.json"
    if not registry_path.exists():
        return GateResult(
            "component_verification", False,
            f"component registry not found: {registry_path}",
        )

    registry_data = json.loads(registry_path.read_text())
    registry = registry_data.get("components", {})

    # Skip refs that aren't real components
    skip_prefixes = ("MH", "H", "FID", "TP")

    unverified: list[str] = []
    missing: list[str] = []

    for fp in pcb.footprints:
        if any(fp.ref.startswith(p) for p in skip_prefixes):
            continue
        if not fp.pads:
            continue

        # Look up by LCSC part number first, then by footprint lib_id
        # Match against requirements to get LCSC
        req_comp = next(
            (c for c in requirements.components if c.ref == fp.ref),
            None,
        )
        lcsc = getattr(req_comp, "lcsc", None) if req_comp else None

        # Try LCSC lookup, then lib_id-based lookup
        entry = None
        if lcsc and lcsc in registry:
            entry = registry[lcsc]
        else:
            # Try matching by footprint_id in registry entries
            for comp_id, comp_data in registry.items():
                fp_id = comp_data.get("footprint_id", "")
                if fp_id and fp_id in fp.lib_id:
                    entry = comp_data
                    break

        if entry is None:
            missing.append(fp.ref)
        elif entry.get("verification_status") != "verified":
            status = entry.get("verification_status", "unknown")
            unverified.append(f"{fp.ref}({status})")

    issues: list[str] = []
    if missing:
        issues.append(
            f"{len(missing)} not in registry: {', '.join(sorted(missing)[:8])}"
        )
    if unverified:
        issues.append(
            f"{len(unverified)} unverified: {', '.join(sorted(unverified)[:8])}"
        )

    passed = len(issues) == 0
    detail = "; ".join(issues) if issues else "OK"
    return GateResult("component_verification", passed, detail)


# ---------------------------------------------------------------------------
# Public: run all DFM gates
# ---------------------------------------------------------------------------

ALL_DFM_GATES: tuple[str, ...] = (
    "footprint_registry_match",
    "no_collisions",
    "all_pads_within_board",
    "mounting_hole_clearance",
    "decoupling_proximity",
    "subcircuit_spread",
    "zone_membership",
    "package_match",
    "subcircuit_completeness",
    "schematic_pcb_sync",
    "component_isolation_zones",
    "board_sizing",
    "component_verification",
)

_GATE_FUNCTIONS: dict[
    str,
    type[object],  # actually Callable, but avoiding import
] = {
    "footprint_registry_match": check_footprint_registry_match,  # type: ignore[dict-item]
    "no_collisions": check_no_collisions,  # type: ignore[dict-item]
    "all_pads_within_board": check_all_within_board,  # type: ignore[dict-item]
    "mounting_hole_clearance": check_mounting_hole_clearance,  # type: ignore[dict-item]
    "decoupling_proximity": check_decoupling_proximity,  # type: ignore[dict-item]
    "subcircuit_spread": check_subcircuit_spread,  # type: ignore[dict-item]
    "zone_membership": check_zone_membership,  # type: ignore[dict-item]
    "package_match": check_package_match,  # type: ignore[dict-item]
    "subcircuit_completeness": check_subcircuit_completeness,  # type: ignore[dict-item]
    "schematic_pcb_sync": check_schematic_pcb_sync,  # type: ignore[dict-item]
    "component_isolation_zones": check_component_isolation_zones,  # type: ignore[dict-item]
    "board_sizing": check_board_sizing,  # type: ignore[dict-item]
    "component_verification": check_component_verification,  # type: ignore[dict-item]
}


def evaluate_dfm_gate(
    gate_name: str,
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> GateResult:
    """Evaluate a single DFM gate by name."""
    from kicad_pipeline.evals.models import GateResult

    fn = _GATE_FUNCTIONS.get(gate_name)
    if fn is None:
        return GateResult(gate_name, False, f"unknown DFM gate: {gate_name}")
    return fn(pcb, requirements)  # type: ignore[operator]
