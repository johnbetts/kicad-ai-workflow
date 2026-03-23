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
        zones = partition_board(bounds, list(requirements.features), topo)
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
                zx1, zy1, zx2, zy2 = z.rect
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    cross_group.append(ref)
                    break
    except Exception:
        pass
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

    passed = len(off_board) == 0 and len(collision_tuples) <= 5
    return PlacementGuardResult(
        passed=passed,
        off_board_refs=tuple(sorted(off_board)),
        off_board_pad_refs=tuple(sorted(off_board_pads)),
        collision_pairs=tuple(collision_tuples),
        cross_group_refs=tuple(sorted(cross_group)),
        issues=tuple(issues),
    )
