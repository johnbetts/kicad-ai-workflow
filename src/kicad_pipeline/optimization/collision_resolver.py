"""Collision detection and resolution for PCB placement.

Provides ``_PlacementGrid`` for occupancy tracking and ``_resolve_collisions``
for iterative AABB-based collision resolution with group-aware clamping.
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    BOARD_EDGE_MARGIN_MM,
    COLLISION_GROUP_EXPANSION_MM,
    COLLISION_MAX_PASSES,
    COMPONENT_CLEARANCE_GAP_MM,
    DEFAULT_FP_SIZE_MM,
    SPIRAL_SEARCH_MAX_RINGS,
)

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import GroupBoundingBox

_log = logging.getLogger(__name__)


class _PlacementGrid:
    """Simple occupancy tracker to prevent component overlaps.

    Tracks placed component bounding boxes and checks for collisions
    before accepting new placements.
    """

    def __init__(self, board_bounds: tuple[float, float, float, float]) -> None:
        self.min_x, self.min_y, self.max_x, self.max_y = board_bounds
        # (cx, cy, half_w, half_h) for each placed component
        self._placed: list[tuple[float, float, float, float]] = []
        self._margin = COMPONENT_CLEARANCE_GAP_MM

    def is_free(self, cx: float, cy: float, w: float, h: float) -> bool:
        """Check if placing a component here would overlap any existing one."""
        hw = w / 2.0 + self._margin
        hh = h / 2.0 + self._margin
        for px, py, phw, phh in self._placed:
            if abs(cx - px) < hw + phw and abs(cy - py) < hh + phh:
                return False
        return True

    def place(self, cx: float, cy: float, w: float, h: float) -> None:
        """Register a component at (cx, cy) with size (w, h)."""
        self._placed.append((cx, cy, w / 2.0, h / 2.0))

    def find_free_pos(
        self,
        target_x: float,
        target_y: float,
        w: float,
        h: float,
        max_radius: float = 0.0,
    ) -> tuple[float, float]:
        """Find nearest free position to (target_x, target_y).

        Searches in expanding concentric rings around the target.

        Args:
            max_radius: If > 0, limit search to within this distance of target.
                Returns clamped target if no free position found within radius.
        """
        bmin_x = self.min_x + BOARD_EDGE_MARGIN_MM
        bmin_y = self.min_y + BOARD_EDGE_MARGIN_MM
        bmax_x = self.max_x - BOARD_EDGE_MARGIN_MM
        bmax_y = self.max_y - BOARD_EDGE_MARGIN_MM

        # Clamp target to board
        tx = max(bmin_x, min(bmax_x, target_x))
        ty = max(bmin_y, min(bmax_y, target_y))

        if self.is_free(tx, ty, w, h):
            return (tx, ty)

        # Spiral search — try 8 directions at increasing radii
        step = max(w, h) * 0.5 + self._margin
        max_rings = SPIRAL_SEARCH_MAX_RINGS
        if max_radius > 0:
            max_rings = min(max_rings, max(3, int(max_radius / step) + 1))
        for ring in range(1, max_rings):
            r = step * ring
            if max_radius > 0 and r > max_radius:
                break
            for angle_idx in range(8 * ring):
                angle = 2 * math.pi * angle_idx / (8 * ring)
                cx = tx + r * math.cos(angle)
                cy = ty + r * math.sin(angle)
                cx = max(bmin_x, min(bmax_x, cx))
                cy = max(bmin_y, min(bmax_y, cy))
                if self.is_free(cx, cy, w, h):
                    return (cx, cy)

        # If max_radius limited the search, retry without radius limit
        if max_radius > 0:
            return self.find_free_pos(target_x, target_y, w, h, max_radius=0.0)

        # Fallback — return clamped target (will collide but won't crash)
        _log.warning("find_free_pos: no free position found for (%.1f, %.1f) "
                     "size (%.1f, %.1f) — returning target (will collide)",
                     target_x, target_y, w, h)
        return (tx, ty)


def _fp_courtyard_sizes(pcb: object) -> dict[str, tuple[float, float]]:
    """Build ref -> (width, height) using accurate courtyard estimates.

    Delegates to :func:`~kicad_pipeline.pcb.footprints.estimate_courtyard_mm`
    which accounts for component body extension beyond the pad field.
    """
    from kicad_pipeline.pcb.footprints import estimate_courtyard_mm

    result: dict[str, tuple[float, float]] = {}
    for fp in pcb.footprints:  # type: ignore[attr-defined]
        result[fp.ref] = estimate_courtyard_mm(fp)
    return result


def _rotation_aware_size(
    ref: str,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Get rotation-aware bounding box for a component."""
    w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
    if ref in positions:
        rot = positions[ref][2]
        if rot % 180 in (90.0, 270.0):
            w, h = h, w
    return w, h


def _count_collisions(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> list[tuple[str, str]]:
    """Detect all AABB collisions (rotation-aware, matching scoring)."""
    collisions: list[tuple[str, str]] = []
    refs = list(positions.keys())
    for i, ref_a in enumerate(refs):
        xa, ya, rot_a = positions[ref_a]
        wa, ha = fp_sizes.get(ref_a, DEFAULT_FP_SIZE_MM)
        if rot_a % 180 in (90.0, 270.0):
            wa, ha = ha, wa

        for ref_b in refs[i + 1:]:
            xb, yb, rot_b = positions[ref_b]
            wb, hb = fp_sizes.get(ref_b, DEFAULT_FP_SIZE_MM)
            if rot_b % 180 in (90.0, 270.0):
                wb, hb = hb, wb

            gap = COMPONENT_CLEARANCE_GAP_MM
            if (abs(xa - xb) < (wa + wb) / 2.0 + gap
                    and abs(ya - yb) < (ha + hb) / 2.0 + gap):
                collisions.append((ref_a, ref_b))
    return collisions


def _group_of_ref(
    ref: str,
    group_bboxes: list[GroupBoundingBox],
) -> GroupBoundingBox | None:
    """Find the GroupBoundingBox containing a given ref."""
    for grp in group_bboxes:
        if ref in grp.refs:
            return grp
    return None


def _get_movable_colliding_refs(
    collisions: list[tuple[str, str]],
    fixed_refs: set[str],
    fp_sizes: dict[str, tuple[float, float]],
) -> list[str]:
    """Get non-fixed colliding refs sorted by area (smallest first)."""
    colliding_refs: set[str] = set()
    for ref_a, ref_b in collisions:
        if ref_a not in fixed_refs:
            colliding_refs.add(ref_a)
        if ref_b not in fixed_refs:
            colliding_refs.add(ref_b)
    return sorted(
        colliding_refs,
        key=lambda r: (
            fp_sizes.get(r, DEFAULT_FP_SIZE_MM)[0]
            * fp_sizes.get(r, DEFAULT_FP_SIZE_MM)[1]
        ),
    )


def _ref_has_collision(
    ref: str,
    rx: float,
    ry: float,
    w: float,
    h: float,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> bool:
    """Check if a specific ref still collides with any other component."""
    for other_ref, (ox, oy, _orot) in positions.items():
        if other_ref == ref:
            continue
        ow, oh = _rotation_aware_size(other_ref, positions, fp_sizes)
        gap = COMPONENT_CLEARANCE_GAP_MM
        if (abs(rx - ox) < (w + ow) / 2.0 + gap
                and abs(ry - oy) < (h + oh) / 2.0 + gap):
            return True
    return False


def _count_collisions_at(
    ref: str,
    cx: float,
    cy: float,
    w: float,
    h: float,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> int:
    """Count how many components would collide with *ref* placed at (cx, cy)."""
    count = 0
    gap = COMPONENT_CLEARANCE_GAP_MM
    for other_ref, (ox, oy, _orot) in positions.items():
        if other_ref == ref:
            continue
        ow, oh = _rotation_aware_size(other_ref, positions, fp_sizes)
        if abs(cx - ox) < (w + ow) / 2.0 + gap and abs(cy - oy) < (h + oh) / 2.0 + gap:
            count += 1
    return count


def _violates_proximity(
    ref: str,
    cx: float,
    cy: float,
    proximity_constraints: dict[str, tuple[str, float]] | None,
    positions: dict[str, tuple[float, float, float]],
) -> bool:
    """Return True if placing *ref* at (cx, cy) violates any proximity constraint.

    A proximity constraint ``proximity_constraints[ref] = (other_ref, max_dist)``
    requires that the Manhattan distance between *ref* and *other_ref* stays at
    or below *max_dist*.
    """
    if proximity_constraints is None:
        return False
    if ref not in proximity_constraints:
        return False
    other_ref, max_dist = proximity_constraints[ref]
    if other_ref not in positions:
        return False
    ox, oy, _ = positions[other_ref]
    dist = math.hypot(cx - ox, cy - oy)
    return dist > max_dist


_NUDGE_COMPASS: tuple[tuple[float, float], ...] = (
    (0.0, -1.0),   # N
    (1.0, -1.0),   # NE
    (1.0, 0.0),    # E
    (1.0, 1.0),    # SE
    (0.0, 1.0),    # S
    (-1.0, 1.0),   # SW
    (-1.0, 0.0),   # W
    (-1.0, -1.0),  # NW
)
_NUDGE_SMALL_MM = (2.0, 3.5, 5.0)
_NUDGE_LARGE_MM = (5.0, 7.5, 10.0)


def _random_nudge_fallback(
    ref: str,
    rx: float,
    ry: float,
    rot: float,
    w: float,
    h: float,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    proximity_constraints: dict[str, tuple[str, float]] | None,
    zone_bboxes: dict[str, tuple[float, float, float, float]] | None = None,
    zone_membership: dict[str, str] | None = None,
) -> tuple[float, float] | None:
    """Try 8-direction nudges to find an improvement when grid relocation fails.

    Attempts small nudges (2-5 mm) first, then larger nudges (5-10 mm).
    Picks the candidate with the fewest remaining collisions (greedy).
    Accepts any candidate that is strictly better than the current position.

    Returns:
        (new_x, new_y) if an improvement was found, else None.
    """
    bmin_x = bounds[0] + BOARD_EDGE_MARGIN_MM
    bmin_y = bounds[1] + BOARD_EDGE_MARGIN_MM
    bmax_x = bounds[2] - BOARD_EDGE_MARGIN_MM
    bmax_y = bounds[3] - BOARD_EDGE_MARGIN_MM

    current_hits = _count_collisions_at(ref, rx, ry, w, h, positions, fp_sizes)

    best_pos: tuple[float, float] | None = None
    best_hits = current_hits  # must beat current to be accepted

    for nudge_set in (_NUDGE_SMALL_MM, _NUDGE_LARGE_MM):
        # Randomise compass order so repeated calls explore different directions
        compass_order = list(_NUDGE_COMPASS)
        random.shuffle(compass_order)
        for dist in nudge_set:
            for dx_unit, dy_unit in compass_order:
                cx = rx + dx_unit * dist
                cy = ry + dy_unit * dist
                # Clamp to board
                cx = max(bmin_x, min(bmax_x, cx))
                cy = max(bmin_y, min(bmax_y, cy))
                # Respect proximity constraints
                if _violates_proximity(ref, cx, cy, proximity_constraints, positions):
                    continue
                # Respect zone boundary — reject nudge if it exits the assigned zone
                if (zone_membership is not None and zone_bboxes is not None
                        and ref in zone_membership):
                    zname = zone_membership[ref]
                    zbbox = zone_bboxes.get(zname)
                    if zbbox is not None:
                        zx1, zy1, zx2, zy2 = zbbox
                        if not (zx1 <= cx <= zx2 and zy1 <= cy <= zy2):
                            continue
                hits = _count_collisions_at(ref, cx, cy, w, h, positions, fp_sizes)
                if hits < best_hits:
                    best_hits = hits
                    best_pos = (cx, cy)
                    if hits == 0:
                        # Collision-free — accept immediately
                        return best_pos
        if best_pos is not None:
            # Found improvement in small nudge set — stop here
            return best_pos

    return best_pos


def _build_exclusion_grid(
    ref: str,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> _PlacementGrid:
    """Build an occupancy grid with all components except *ref*."""
    grid = _PlacementGrid(bounds)
    for other_ref, (ox, oy, _orot) in positions.items():
        if other_ref == ref:
            continue
        ow, oh = _rotation_aware_size(other_ref, positions, fp_sizes)
        grid.place(ox, oy, ow, oh)
    return grid


def _compute_large_ic_push(
    ref: str,
    rx: float,
    ry: float,
    w: float,
    h: float,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """If ref collides with a large IC (>100mm^2), push away from its center.

    Returns:
        Target (x, y) for relocation.
    """
    target_x, target_y = rx, ry
    for other_ref, (ox, oy, _orot) in positions.items():
        if other_ref == ref:
            continue
        ow, oh = _rotation_aware_size(other_ref, positions, fp_sizes)
        if ow * oh < 100.0:
            continue
        gap = COMPONENT_CLEARANCE_GAP_MM
        if (abs(rx - ox) < (w + ow) / 2.0 + gap
                and abs(ry - oy) < (h + oh) / 2.0 + gap):
            dx = rx - ox
            dy = ry - oy
            push_gap = COMPONENT_CLEARANCE_GAP_MM + 1.0  # extra margin
            if abs(dx) * oh > abs(dy) * ow:
                push_x = ow / 2.0 + w / 2.0 + push_gap
                target_x = ox + push_x if dx >= 0 else ox - push_x
            else:
                push_y = oh / 2.0 + h / 2.0 + push_gap
                target_y = oy + push_y if dy >= 0 else oy - push_y
            break
    return target_x, target_y


def _clamp_to_group(
    ref: str,
    fx: float,
    fy: float,
    w: float,
    h: float,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    group_bboxes: list[GroupBoundingBox] | None,
    grid: _PlacementGrid,
    group_rect_fn: object,
) -> tuple[float, float]:
    """Clamp position to group bounding box if applicable.

    Returns:
        (fx, fy) — clamped or original position.
    """
    if group_bboxes is None:
        return fx, fy
    grp = _group_of_ref(ref, group_bboxes)
    if grp is None:
        return fx, fy
    grx1, gry1, grx2, gry2 = group_rect_fn(grp, positions)  # type: ignore[operator]
    cx = max(grx1 + w / 2, min(grx2 - w / 2, fx))
    cy = max(gry1 + h / 2, min(gry2 - h / 2, fy))
    if grid.is_free(cx, cy, w, h):
        return cx, cy
    return fx, fy


def _group_rect(
    grp: GroupBoundingBox,
    pos: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Return current bounding rect of group members with expansion margin."""
    gmin_x = float("inf")
    gmin_y = float("inf")
    gmax_x = float("-inf")
    gmax_y = float("-inf")
    for r in grp.refs:
        if r not in pos:
            continue
        rx, ry, _rot = pos[r]
        w, h = fp_sizes.get(r, DEFAULT_FP_SIZE_MM)
        if _rot % 180 in (90.0, 270.0):
            w, h = h, w
        gmin_x = min(gmin_x, rx - w / 2)
        gmin_y = min(gmin_y, ry - h / 2)
        gmax_x = max(gmax_x, rx + w / 2)
        gmax_y = max(gmax_y, ry + h / 2)
    margin = COLLISION_GROUP_EXPANSION_MM
    return (gmin_x - margin, gmin_y - margin, gmax_x + margin, gmax_y + margin)


def _run_collision_pass(
    result: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    group_bboxes: list[GroupBoundingBox] | None,
    pass_num: int,
    proximity_constraints: dict[str, tuple[str, float]] | None = None,
    zone_bboxes: dict[str, tuple[float, float, float, float]] | None = None,
    zone_membership: dict[str, str] | None = None,
) -> int:
    """Execute one collision-resolution pass; return number of components moved."""
    current_collisions = _count_collisions(result, fp_sizes)
    if not current_collisions:
        return 0

    sorted_refs = _get_movable_colliding_refs(current_collisions, fixed_refs, fp_sizes)

    def _group_rect_fn(
        grp: GroupBoundingBox,
        pos: dict[str, tuple[float, float, float]],
    ) -> tuple[float, float, float, float]:
        return _group_rect(grp, pos, fp_sizes)

    moved = 0
    for ref in sorted_refs:
        if ref in fixed_refs:
            continue
        rx, ry, rot = result[ref]
        w, h = _rotation_aware_size(ref, result, fp_sizes)
        if not _ref_has_collision(ref, rx, ry, w, h, result, fp_sizes):
            continue
        grid = _build_exclusion_grid(ref, result, fp_sizes, bounds)
        target_x, target_y = _compute_large_ic_push(ref, rx, ry, w, h, result, fp_sizes)
        fx, fy = grid.find_free_pos(target_x, target_y, w, h)
        fx, fy = _clamp_to_group(ref, fx, fy, w, h, result, fp_sizes,
                                 group_bboxes, grid, _group_rect_fn)

        # If grid relocation returned the same position, try random nudge fallback.
        if fx == rx and fy == ry:
            nudge = _random_nudge_fallback(
                ref, rx, ry, rot, w, h, result, fp_sizes, bounds, proximity_constraints,
                zone_bboxes=zone_bboxes, zone_membership=zone_membership,
            )
            if nudge is not None:
                fx, fy = nudge
                _log.debug(
                    "  Nudge fallback moved %s from (%.1f, %.1f) to (%.1f, %.1f)",
                    ref, rx, ry, fx, fy,
                )

        result[ref] = (fx, fy, rot)
        moved += 1

    remaining = len(_count_collisions(result, fp_sizes))
    _log.info(
        "  Collision resolution pass %d: relocated %d, %d remaining",
        pass_num + 1, moved, remaining,
    )
    return moved


def _resolve_collisions(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    group_bboxes: list[GroupBoundingBox] | None = None,
    proximity_constraints: dict[str, tuple[str, float]] | None = None,
    zone_bboxes: dict[str, tuple[float, float, float, float]] | None = None,
    zone_membership: dict[str, str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Resolve courtyard collisions using grid-based relocation.

    For each colliding component (smaller one in the pair), relocate it
    to the nearest free position using the occupancy grid. This guarantees
    the relocated component won't collide with anything already placed.

    When *group_bboxes* is provided, relocated positions are clamped to stay
    within their group's current bounding box + a small margin so that
    collision resolution doesn't scatter group members.
    """
    result = dict(positions)

    collisions = _count_collisions(result, fp_sizes)
    if not collisions:
        _log.info("  Collision resolution: no collisions found")
        return result

    _log.info("  Collision resolution: %d initial collisions", len(collisions))

    for _pass in range(COLLISION_MAX_PASSES):
        if not _count_collisions(result, fp_sizes):
            break
        _run_collision_pass(
            result, fp_sizes, bounds, fixed_refs, group_bboxes, _pass,
            proximity_constraints, zone_bboxes=zone_bboxes, zone_membership=zone_membership,
        )
        if not _count_collisions(result, fp_sizes):
            break

    return result
