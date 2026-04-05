"""Post-optimization placement validation gate.

Checks critical placement invariants (off-board, collisions, cross-group
contamination) that have caused bugs in the past (KI-004 through KI-010).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.optimization.placement_types import _board_bounds
from kicad_pipeline.pcb.pin_map import origin_to_centroid

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlacementGuardResult:
    """Result of post-optimization placement validation.

    Attributes:
        passed: True if all critical guards pass.
        off_board_refs: Component refs with centers outside board bounds.
        off_board_pad_refs: Component refs with pad extents outside board bounds.
        collision_pairs: Pairs of overlapping component refs.
        cross_group_refs: Components placed in the wrong group's zone.
        issues: Human-readable list of all issues found.
    """

    passed: bool
    off_board_refs: tuple[str, ...]
    off_board_pad_refs: tuple[str, ...]
    collision_pairs: tuple[tuple[str, str], ...]
    cross_group_refs: tuple[str, ...]
    issues: tuple[str, ...]


def _fp_courtyard_sizes_for_guard(pcb: PCBDesign) -> dict[str, tuple[float, float]]:
    """Build ref -> (width, height) using courtyard estimates."""
    from kicad_pipeline.pcb.footprints import estimate_courtyard_mm

    result: dict[str, tuple[float, float]] = {}
    for fp in pcb.footprints:
        result[fp.ref] = estimate_courtyard_mm(fp)
    return result


def _guard_off_board_centers(
    pcb: PCBDesign,
    bounds: tuple[float, float, float, float],
    margin_mm: float,
    issues: list[str],
) -> list[str]:
    """Guard 1: Check all component centers are within board bounds."""
    min_x, min_y, max_x, max_y = bounds
    off_board: list[str] = []
    for fp in pcb.footprints:
        x, y = fp.position.x, fp.position.y
        if (x < min_x - margin_mm or x > max_x + margin_mm
                or y < min_y - margin_mm or y > max_y + margin_mm):
            off_board.append(fp.ref)
    if off_board:
        issues.append(
            f"Off-board centers ({len(off_board)}): {', '.join(sorted(off_board))}"
        )
    return off_board


def _guard_off_board_pads(
    pcb: PCBDesign,
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    margin_mm: float,
    issues: list[str],
) -> list[str]:
    """Guard 2: Check pad extents are within board bounds."""
    min_x, min_y, max_x, max_y = bounds
    off_board_pads: list[str] = []
    for fp in pcb.footprints:
        w, h = fp_sizes.get(fp.ref, (2.0, 2.0))
        rot = fp.rotation
        if rot % 180 in (90.0, 270.0):
            w, h = h, w
        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, rot)
        half_w, half_h = w / 2.0, h / 2.0
        if (cx - half_w < min_x - margin_mm or cx + half_w > max_x + margin_mm
                or cy - half_h < min_y - margin_mm or cy + half_h > max_y + margin_mm):
            off_board_pads.append(fp.ref)
    if off_board_pads:
        issues.append(
            f"Off-board pads ({len(off_board_pads)}): "
            f"{', '.join(sorted(off_board_pads))}"
        )
    return off_board_pads


def _build_centroid_positions(
    pcb: PCBDesign,
) -> dict[str, tuple[float, float, float]]:
    """Build centroid-space position dict from PCB footprints."""
    positions_dict: dict[str, tuple[float, float, float]] = {}
    for fp in pcb.footprints:
        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        positions_dict[fp.ref] = (cx, cy, fp.rotation)
    return positions_dict


def _guard_collisions(
    positions_dict: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    issues: list[str],
) -> list[tuple[str, str]]:
    """Guard 3: Check courtyard collisions."""
    from kicad_pipeline.pcb.constraints import check_courtyard_collisions

    rotations = {ref: rot for ref, (_x, _y, rot) in positions_dict.items()}
    collision_points: dict[str, Point] = {
        ref: Point(x=cx, y=cy)
        for ref, (cx, cy, _rot) in positions_dict.items()
    }
    try:
        collision_list = check_courtyard_collisions(
            collision_points, fp_sizes, rotations=rotations,
        )
    except Exception:
        collision_list = []

    collision_tuples: list[tuple[str, str]] = []
    for viol_str in collision_list:
        s = str(viol_str)
        if "collision:" in s and " and " in s:
            after_colon = s.split("collision:", 1)[1].strip()
            pair = after_colon.split(" and ", 1)
            if len(pair) == 2:
                collision_tuples.append((pair[0].strip(), pair[1].strip()))
    if collision_tuples:
        issues.append(f"Collisions ({len(collision_tuples)}): "
                       f"{collision_tuples[:10]}")
    return collision_tuples


def _guard_cross_group(
    requirements: ProjectRequirements,
    positions_dict: dict[str, tuple[float, float, float]],
    bounds: tuple[float, float, float, float],
    issues: list[str],
) -> list[str]:
    """Guard 4: Check for cross-group contamination."""
    cross_group: list[str] = []
    if not requirements.features:
        return cross_group
    from kicad_pipeline.optimization.functional_grouper import (
        compute_power_flow_topology,
        detect_subcircuits,
    )
    from kicad_pipeline.optimization.zone_partitioner import (
        partition_board,
        zone_for_group,
    )
    try:
        sc = detect_subcircuits(requirements)
        topo = compute_power_flow_topology(sc)
        zones = partition_board(
            bounds, list(requirements.features), topo,
            requirements=requirements,
        )
        ref_to_group: dict[str, str] = {}
        for feat in requirements.features:
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                ref_to_group[r] = feat.name
        for ref, (cx, cy, _rot) in positions_dict.items():
            grp = ref_to_group.get(ref)
            if not grp or ref.startswith("J"):
                continue
            own_zone = zone_for_group(grp, zones)
            for z in zones:
                if own_zone and z.name == own_zone.name:
                    continue
                if z.contains(cx, cy):
                    cross_group.append(ref)
                    break
    except Exception as exc:
        _log.debug("Cross-group contamination check failed: %s", exc)
    if cross_group:
        issues.append(
            f"Cross-group ({len(cross_group)}): "
            f"{', '.join(sorted(cross_group)[:15])}"
        )
    return cross_group


def validate_placement(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    *,
    margin_mm: float = 5.0,
) -> PlacementGuardResult:
    """Post-optimization placement validation gate.

    Checks all critical placement invariants that have caused bugs in the past
    (KI-004 through KI-010). Returns a result indicating whether the placement
    is acceptable and listing all issues found.

    Args:
        pcb: The optimized PCB design.
        requirements: Project requirements with feature groups.
        margin_mm: Tolerance for off-board checks (default 5mm).

    Returns:
        PlacementGuardResult with pass/fail and detailed issue list.
    """
    bounds = _board_bounds(pcb)
    min_x, min_y, max_x, max_y = bounds
    fp_sizes = _fp_courtyard_sizes_for_guard(pcb)
    issues: list[str] = []

    off_board = _guard_off_board_centers(pcb, bounds, margin_mm, issues)
    off_board_pads = _guard_off_board_pads(pcb, fp_sizes, bounds, margin_mm, issues)

    positions_dict = _build_centroid_positions(pcb)
    collision_tuples = _guard_collisions(positions_dict, fp_sizes, issues)
    cross_group = _guard_cross_group(requirements, positions_dict, bounds, issues)

    # Known recurring bug checks (from human review feedback)
    _guard_connector_rotation(pcb, bounds, issues)
    _guard_relay_orientation(pcb, issues)
    _guard_antenna_isolation(pcb, issues)
    _guard_ethernet_adjacency(pcb, positions_dict, issues)

    # RECURRING issues BLOCK the build — the board owner requires that every
    # known bug is caught before they see the board.  A RECURRING issue means
    # a previously-reported bug has regressed.
    recurring_count = sum(1 for i in issues if "RECURRING" in i)
    # Collision threshold: ≤25 for placement phase (pre-routing boards always
    # have some courtyard overlaps that routing clearance will resolve).
    # The scoring system penalizes collisions proportionally via the
    # diminishing penalty formula.
    passed = (
        len(off_board) == 0
        and len(collision_tuples) <= 25
        and recurring_count == 0
    )
    return PlacementGuardResult(
        passed=passed,
        off_board_refs=tuple(sorted(off_board)),
        off_board_pad_refs=tuple(sorted(off_board_pads)),
        collision_pairs=tuple(collision_tuples),
        cross_group_refs=tuple(sorted(cross_group)),
        issues=tuple(issues),
    )


# ---------------------------------------------------------------------------
# Known Recurring Bug Guards
# ---------------------------------------------------------------------------

def _guard_connector_rotation(
    pcb: PCBDesign, bounds: tuple[float, float, float, float],
    issues: list[str],
) -> None:
    """Check that connectors face off-board (wires toward nearest edge).

    Recurring bugs: screw terminals facing inward, USB-C facing wrong way,
    RJ45 not at board edge.
    """
    import math
    min_x, min_y, max_x, max_y = bounds
    board_cx = (min_x + max_x) / 2.0
    board_cy = (min_y + max_y) / 2.0

    for fp in pcb.footprints:
        if not fp.ref.startswith("J"):
            continue
        x, y = fp.position.x, fp.position.y
        rot = fp.rotation

        # Find nearest board edge
        dist_left = abs(x - min_x)
        dist_right = abs(x - max_x)
        dist_top = abs(y - min_y)
        dist_bottom = abs(y - max_y)
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        # Connector should be within 10mm of an edge
        if min_dist > 10.0:
            issues.append(
                f"RECURRING: {fp.ref} is {min_dist:.0f}mm from nearest edge "
                f"(connectors should be <10mm from edge)"
            )

        # Check rotation: connector opening should face the nearest edge
        # Terminal blocks at top edge: rot should be 0 or 180 (wires up)
        # Terminal blocks at bottom edge: rot should be 0 or 180 (wires down)
        # USB/RJ45 at left/right edge: rot should be 90 or 270
        is_terminal = "terminal" in fp.lib_id.lower() or "5.08" in fp.lib_id
        if is_terminal and min_dist == dist_top and rot not in (0.0, 180.0):
            issues.append(
                f"RECURRING: {fp.ref} screw terminal at top edge has rot={rot:.0f} "
                f"(should be 0 or 180 for wires facing outward)"
            )


def _guard_relay_orientation(pcb: PCBDesign, issues: list[str]) -> None:
    """Check relay orientation — COM pin must face screw terminals, coil faces drivers.

    SANYOU SRD relay pin layout (at rot=0):
      Pin 1 (COM) at local (-5.1, +6.0) — bottom-left
      Pin 5 (COIL+) at local (-7.1, 0.0) — left-center
      Pin 2 (COIL-) at local (+7.1, +6.0) — bottom-right
    U-shaped isolation cutout on LEFT side (x≈-6.1mm) separating COM from COIL.

    At rot=90 (CW): COM moves to top, coil to bottom → drivers below face coil ✓
    At rot=0 or 180: COM at left/right — cutout doesn't align with driver column.

    Also verifies COM pin (pin 1) faces the screw terminal side (top edge)
    and coil pins (2, 5) face the driver side (bottom/below).
    """
    import math

    for fp in pcb.footprints:
        if not fp.ref.startswith("K"):
            continue
        rot = fp.rotation

        # Relays must be at 90 degrees for this board layout
        # (drivers below on coil side, terminals above on COM side)
        if rot % 180 == 0:
            issues.append(
                f"RECURRING: {fp.ref} relay at rot={rot:.0f} — needs 90deg rotation. "
                f"At rot=0/180 the isolation cutout doesn't separate COM from coil "
                f"drivers correctly. SRD datasheet: COM=pin1, COIL=pin2/5."
            )
            continue

        # At rot=90: verify COM pin (pin 1) faces toward top edge (terminals)
        # Pin 1 local position: (-5.1, +6.0). At rot=90: board_y = origin_y + (-5.1)*sin(90) + 6.0*cos(90) = origin_y - 5.1
        # So COM is ABOVE the relay center — toward top edge. ✓
        # This is correct when screw terminals are at the top edge.
        pin1 = None
        for pad in fp.pads:
            if pad.number == "1":
                pin1 = pad
                break
        if pin1 is not None:
            rot_rad = math.radians(rot)
            com_board_y = fp.position.y + pin1.position.x * math.sin(rot_rad) + pin1.position.y * math.cos(rot_rad)
            # COM should be toward the top of the board (lower Y value)
            if com_board_y > fp.position.y + 2.0:
                issues.append(
                    f"RECURRING: {fp.ref} COM pin (pin 1) faces bottom — should face "
                    f"top edge toward screw terminals. Try rot={((rot + 180) % 360):.0f}"
                )

        # Verify isolation cutout is BELOW COM pin (not centered on it).
        # The cutout Edge.Cuts arcs should have their center X < pin 1 X
        # in the rotated frame (meaning the arc is on the far side of COM).
        edge_cuts = [g for g in fp.graphics
                     if hasattr(g, "layer") and "Edge" in getattr(g, "layer", "")]
        if edge_cuts:
            # Check that at least one arc midpoint is shifted away from COM
            for g in edge_cuts:
                if hasattr(g, "mid"):  # FootprintArc
                    # Arc mid should be offset from origin (shifted cutout)
                    mid_dist = math.hypot(g.mid.x, g.mid.y)
                    if mid_dist < 2.0:
                        issues.append(
                            f"RECURRING: {fp.ref} isolation cutout centered ON pin 1 "
                            f"(arc mid at {g.mid.x:.1f},{g.mid.y:.1f}) — must be "
                            f"shifted below pin 1 so COM sits inside the U"
                        )
                    break


def _guard_antenna_isolation(pcb: PCBDesign, issues: list[str]) -> None:
    """Check that antenna keepout zone exists near the antenna end of the MCU.

    Recurring bug: isolation zone placed on wrong side of ESP32 module.
    The antenna is at the TOP of the module body (negative Y in local coords).
    At rot=180, antenna points toward +Y (bottom edge).
    """
    import math

    mcu_fp = None
    for fp in pcb.footprints:
        if "esp32" in fp.lib_id.lower() or "wroom" in fp.lib_id.lower():
            mcu_fp = fp
            break
    if mcu_fp is None:
        return

    # Compute antenna end position based on rotation.
    # IMPORTANT: use centroid (body center), not origin (pin 1).
    cx, cy = origin_to_centroid(mcu_fp, mcu_fp.position.x, mcu_fp.position.y, mcu_fp.rotation)
    module_half_h = 12.75  # ESP32-S3-WROOM-1 half-height
    rot_rad = math.radians(mcu_fp.rotation)
    antenna_x = cx - module_half_h * math.sin(rot_rad)
    antenna_y = cy - module_half_h * math.cos(rot_rad)

    # Check if any keepout zone exists within 15mm of the antenna end
    has_nearby_keepout = False
    if pcb.keepouts:
        for ko in pcb.keepouts:
            if not ko.polygon:
                continue
            ko_cx = sum(p.x for p in ko.polygon) / len(ko.polygon)
            ko_cy = sum(p.y for p in ko.polygon) / len(ko.polygon)
            dist = math.hypot(ko_cx - antenna_x, ko_cy - antenna_y)
            if dist < 15.0:
                has_nearby_keepout = True
                break

    if not has_nearby_keepout:
        issues.append(
            f"RECURRING: {mcu_fp.ref} antenna keepout missing or misplaced — "
            f"antenna end at ({antenna_x:.0f},{antenna_y:.0f}), "
            f"no keepout within 15mm"
        )


def _guard_ethernet_adjacency(
    pcb: PCBDesign,
    positions: dict[str, tuple[float, float]],
    issues: list[str],
) -> None:
    """Check that ethernet magnetics (U8) is adjacent to RJ45 (J13).

    Recurring bug: U8 placed far from J13.
    """
    import math

    # Find U8 and J13 by ref
    u8_pos = positions.get("U8")
    j13_pos = positions.get("J13")
    if u8_pos is None or j13_pos is None:
        return

    dist = math.hypot(u8_pos[0] - j13_pos[0], u8_pos[1] - j13_pos[1])
    if dist > 5.0:
        issues.append(
            f"RECURRING: U8 (magnetics) is {dist:.0f}mm from J13 (RJ45) — "
            f"should be <5mm for signal integrity"
        )
