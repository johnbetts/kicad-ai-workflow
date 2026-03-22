"""Collision detection and resolution for PCB placement.

Provides ``_PlacementGrid`` for occupancy tracking and ``_resolve_collisions``
for iterative AABB-based collision resolution with group-aware clamping.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

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
        self._margin = 0.5  # mm clearance between components (courtyard-safe)

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
        margin = 2.0
        bmin_x = self.min_x + margin
        bmin_y = self.min_y + margin
        bmax_x = self.max_x - margin
        bmax_y = self.max_y - margin

        # Clamp target to board
        tx = max(bmin_x, min(bmax_x, target_x))
        ty = max(bmin_y, min(bmax_y, target_y))

        if self.is_free(tx, ty, w, h):
            return (tx, ty)

        # Spiral search — try 8 directions at increasing radii
        step = max(w, h) * 0.5 + self._margin
        max_rings = 40
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
    w, h = fp_sizes.get(ref, (2.0, 2.0))
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
        wa, ha = fp_sizes.get(ref_a, (2.0, 2.0))
        if rot_a % 180 in (90.0, 270.0):
            wa, ha = ha, wa

        for ref_b in refs[i + 1:]:
            xb, yb, rot_b = positions[ref_b]
            wb, hb = fp_sizes.get(ref_b, (2.0, 2.0))
            if rot_b % 180 in (90.0, 270.0):
                wb, hb = hb, wb

            if (abs(xa - xb) < (wa + wb) / 2.0
                    and abs(ya - yb) < (ha + hb) / 2.0):
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


def _resolve_collisions(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    group_bboxes: list[GroupBoundingBox] | None = None,
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

    # Build current group rect lookup (re-computed from positions each pass)
    def _group_rect(
        grp: GroupBoundingBox,
        pos: dict[str, tuple[float, float, float]],
    ) -> tuple[float, float, float, float]:
        """Current bounding rect of group members in absolute coords."""
        gmin_x = float("inf")
        gmin_y = float("inf")
        gmax_x = float("-inf")
        gmax_y = float("-inf")
        for r in grp.refs:
            if r not in pos:
                continue
            rx, ry, _rot = pos[r]
            w, h = fp_sizes.get(r, (2.0, 2.0))
            gmin_x = min(gmin_x, rx - w / 2)
            gmin_y = min(gmin_y, ry - h / 2)
            gmax_x = max(gmax_x, rx + w / 2)
            gmax_y = max(gmax_y, ry + h / 2)
        margin = 5.0  # allow 5mm expansion for collision resolution
        return (gmin_x - margin, gmin_y - margin,
                gmax_x + margin, gmax_y + margin)

    # Iteratively relocate colliding components
    for _pass in range(12):
        # Recompute colliding refs each pass (relocations may create new collisions)
        current_collisions = _count_collisions(result, fp_sizes)
        if not current_collisions:
            break
        colliding_refs: set[str] = set()
        for ref_a, ref_b in current_collisions:
            if ref_a not in fixed_refs:
                colliding_refs.add(ref_a)
            if ref_b not in fixed_refs:
                colliding_refs.add(ref_b)

        # Sort: move smaller components first (less disruptive)
        sorted_refs = sorted(
            colliding_refs,
            key=lambda r: (
                fp_sizes.get(r, (2.0, 2.0))[0] * fp_sizes.get(r, (2.0, 2.0))[1]
            ),
        )

        moved = 0
        for ref in sorted_refs:
            if ref in fixed_refs:
                continue
            rx, ry, rot = result[ref]
            w, h = _rotation_aware_size(ref, result, fp_sizes)

            # Check if this ref still collides
            has_collision = False
            for other_ref, (ox, oy, _orot) in result.items():
                if other_ref == ref:
                    continue
                ow, oh = _rotation_aware_size(other_ref, result, fp_sizes)
                if (abs(rx - ox) < (w + ow) / 2.0
                        and abs(ry - oy) < (h + oh) / 2.0):
                    has_collision = True
                    break

            if not has_collision:
                continue

            # Build grid WITHOUT this component
            grid = _PlacementGrid(bounds)
            for other_ref, (ox, oy, _orot) in result.items():
                if other_ref == ref:
                    continue
                ow, oh = _rotation_aware_size(other_ref, result, fp_sizes)
                grid.place(ox, oy, ow, oh)

            # If colliding with a large IC (>100mm^2), push away from its center
            # to avoid landing right at the edge of its courtyard
            target_x, target_y = rx, ry
            for other_ref, (ox, oy, _orot) in result.items():
                if other_ref == ref:
                    continue
                ow, oh = _rotation_aware_size(other_ref, result, fp_sizes)
                if ow * oh < 100.0:
                    continue  # not a large IC
                if (abs(rx - ox) < (w + ow) / 2.0
                        and abs(ry - oy) < (h + oh) / 2.0):
                    # Inside large IC — push to nearest edge + margin
                    dx = rx - ox
                    dy = ry - oy
                    if abs(dx) * oh > abs(dy) * ow:
                        # Closer to left/right edge
                        push_x = (ow / 2.0 + w / 2.0 + 2.0)
                        target_x = ox + push_x if dx >= 0 else ox - push_x
                    else:
                        # Closer to top/bottom edge
                        push_y = (oh / 2.0 + h / 2.0 + 2.0)
                        target_y = oy + push_y if dy >= 0 else oy - push_y
                    break

            # Find nearest free position to target
            fx, fy = grid.find_free_pos(target_x, target_y, w, h)

            # Clamp to group bounding box if group constraints active,
            # but ONLY if clamping doesn't re-create the collision.
            if group_bboxes is not None:
                grp = _group_of_ref(ref, group_bboxes)
                if grp is not None:
                    grx1, gry1, grx2, gry2 = _group_rect(grp, result)
                    cx = max(grx1 + w / 2, min(grx2 - w / 2, fx))
                    cy = max(gry1 + h / 2, min(gry2 - h / 2, fy))
                    # Only apply group clamp if it doesn't cause overlap
                    # with the same component we're trying to escape
                    if grid.is_free(cx, cy, w, h):
                        fx, fy = cx, cy

            result[ref] = (fx, fy, rot)
            moved += 1

        remaining = len(_count_collisions(result, fp_sizes))
        _log.info(
            "  Collision resolution pass %d: relocated %d, %d remaining",
            _pass + 1, moved, remaining,
        )
        if remaining == 0:
            break

    return result
