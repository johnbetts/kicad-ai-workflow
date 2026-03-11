"""Group placer — places feature groups as rigid units within board zones.

Each FeatureBlock group is treated as a rigid rectangular unit with fixed
internal component offsets (from ``_layout_group()``).  The placer positions
each group within its assigned zone using collision-free grid placement.

Special handling:
- Connectors within groups are slid to the nearest zone/board edge.
- Ferrites that bridge isolation zones are placed at zone boundaries.
- Groups are sorted largest-first for priority placement.

This is Level 2 of the 3-level hierarchical placement engine.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import FeatureBlock
    from kicad_pipeline.optimization.zone_partitioner import BoardZone

_log = logging.getLogger(__name__)

# Edge margin for connector pinning (mm)
_CONNECTOR_EDGE_MARGIN_MM: float = 3.0

# Margin when placing groups within zones (mm from zone edge)
_ZONE_INSET_MM: float = 2.0


@dataclass(frozen=True)
class PlacedGroup:
    """A feature group placed on the board as a rigid unit.

    Attributes:
        name: FeatureBlock name.
        zone: Zone name this group was placed in.
        origin: Group origin (top-left corner) on the board (x, y).
        refs: Component refs in this group.
        positions: Mapping from ref to absolute (x, y) position.
        bbox: Bounding box (x_min, y_min, x_max, y_max) on the board.
    """

    name: str
    zone: str
    origin: tuple[float, float]
    refs: tuple[str, ...]
    positions: dict[str, tuple[float, float]]
    bbox: tuple[float, float, float, float]


class _GroupGrid:
    """Occupancy tracker for placing groups (rectangular blocks) without overlap.

    Similar to _PlacementGrid but operates on group-sized rectangles.
    """

    def __init__(self, bounds: tuple[float, float, float, float]) -> None:
        self.min_x, self.min_y, self.max_x, self.max_y = bounds
        self._placed: list[tuple[float, float, float, float]] = []
        self._margin = 3.0  # mm clearance between groups

    def is_free(self, cx: float, cy: float, w: float, h: float) -> bool:
        """Check if placing a group here would overlap any existing one."""
        hw = w / 2.0 + self._margin
        hh = h / 2.0 + self._margin
        for px, py, phw, phh in self._placed:
            if abs(cx - px) < hw + phw and abs(cy - py) < hh + phh:
                return False
        return True

    def place(self, cx: float, cy: float, w: float, h: float) -> None:
        """Register a group at center (cx, cy) with size (w, h)."""
        self._placed.append((cx, cy, w / 2.0, h / 2.0))

    def find_free_pos(
        self,
        target_x: float,
        target_y: float,
        w: float,
        h: float,
    ) -> tuple[float, float]:
        """Find nearest free position to target using spiral search."""
        margin = self._margin
        bmin_x = self.min_x + margin
        bmin_y = self.min_y + margin
        bmax_x = self.max_x - margin
        bmax_y = self.max_y - margin

        tx = max(bmin_x + w / 2, min(bmax_x - w / 2, target_x))
        ty = max(bmin_y + h / 2, min(bmax_y - h / 2, target_y))

        if self.is_free(tx, ty, w, h):
            return (tx, ty)

        step = max(w, h) * 0.3 + self._margin
        for ring in range(1, 30):
            r = step * ring
            for angle_idx in range(8 * ring):
                angle = 2 * math.pi * angle_idx / (8 * ring)
                cx = tx + r * math.cos(angle)
                cy = ty + r * math.sin(angle)
                cx = max(bmin_x + w / 2, min(bmax_x - w / 2, cx))
                cy = max(bmin_y + h / 2, min(bmax_y - h / 2, cy))
                if self.is_free(cx, cy, w, h):
                    return (cx, cy)

        return (tx, ty)


def _group_dimensions(
    internal_layout: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Compute (width, height) of a group from its internal layout.

    Args:
        internal_layout: ref → (rel_x, rel_y, rotation) relative offsets.
        fp_sizes: ref → (w, h) courtyard sizes.
    """
    if not internal_layout:
        return (5.0, 5.0)

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    for ref, (rx, ry, rot) in internal_layout.items():
        w, h = fp_sizes.get(ref, (2.0, 2.0))
        if rot % 180 in (90.0, 270.0):
            w, h = h, w
        min_x = min(min_x, rx - w / 2)
        min_y = min(min_y, ry - h / 2)
        max_x = max(max_x, rx + w / 2)
        max_y = max(max_y, ry + h / 2)

    return (max_x - min_x + 1.0, max_y - min_y + 1.0)


def _group_internal_origin(
    internal_layout: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Compute the (min_x, min_y) of the internal layout bounding box."""
    if not internal_layout:
        return (0.0, 0.0)

    min_x = float("inf")
    min_y = float("inf")
    for ref, (rx, ry, rot) in internal_layout.items():
        w, h = fp_sizes.get(ref, (2.0, 2.0))
        if rot % 180 in (90.0, 270.0):
            w, h = h, w
        min_x = min(min_x, rx - w / 2)
        min_y = min(min_y, ry - h / 2)

    return (min_x, min_y)


def place_groups(
    zones: list[BoardZone],
    groups: list[FeatureBlock],
    internal_layouts: dict[str, dict[str, tuple[float, float, float]]],
    fp_sizes: dict[str, tuple[float, float]],
    board_bounds: tuple[float, float, float, float],
) -> list[PlacedGroup]:
    """Place each feature group as a rigid unit within its assigned zone.

    Args:
        zones: Board zones from ``partition_board()``.
        groups: FeatureBlock instances.
        internal_layouts: group_name → {ref → (rel_x, rel_y, rotation)}.
            These come from ``_layout_group()`` and define the fixed internal
            arrangement of components within each group.
        fp_sizes: ref → (width, height) courtyard sizes.
        board_bounds: (min_x, min_y, max_x, max_y) board outline.

    Returns:
        List of PlacedGroup instances with absolute positions.
    """
    bx1, by1, bx2, by2 = board_bounds
    grid = _GroupGrid(board_bounds)
    placed: list[PlacedGroup] = []

    # Build zone lookup
    zone_map: dict[str, BoardZone] = {}
    for bz in zones:
        for gname in bz.groups:
            zone_map[gname] = bz

    # Sort groups: largest (most components) first for priority placement
    sorted_groups = sorted(groups, key=lambda g: len(g.components), reverse=True)

    for group in sorted_groups:
        layout = internal_layouts.get(group.name, {})
        if not layout:
            _log.warning("No internal layout for group '%s' — skipping", group.name)
            continue

        gw, gh = _group_dimensions(layout, fp_sizes)
        gox, goy = _group_internal_origin(layout, fp_sizes)

        # Find target zone
        zone: BoardZone | None = zone_map.get(group.name)
        if zone is not None:
            zx1, zy1, zx2, zy2 = zone.rect
            # Target center of zone
            target_x = (zx1 + zx2) / 2.0
            target_y = (zy1 + zy2) / 2.0
            zone_name = zone.name
        else:
            # No zone assigned — place in board center
            target_x = (bx1 + bx2) / 2.0
            target_y = (by1 + by2) / 2.0
            zone_name = "unassigned"

        # Find collision-free position for the group bounding box
        cx, cy = grid.find_free_pos(target_x, target_y, gw, gh)
        grid.place(cx, cy, gw, gh)

        # Compute absolute positions: shift internal layout to center at (cx, cy)
        # Internal layout origin is at (gox, goy), center is at (gox + gw/2, goy + gh/2)
        offset_x = cx - (gox + gw / 2.0)
        offset_y = cy - (goy + gh / 2.0)

        abs_positions: dict[str, tuple[float, float]] = {}
        for ref, (rx, ry, _rot) in layout.items():
            abs_x = rx + offset_x
            abs_y = ry + offset_y
            # Clamp to board
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            abs_x = max(bx1 + w / 2 + 1, min(bx2 - w / 2 - 1, abs_x))
            abs_y = max(by1 + h / 2 + 1, min(by2 - h / 2 - 1, abs_y))
            abs_positions[ref] = (abs_x, abs_y)

        # Compute bounding box
        all_x = [p[0] for p in abs_positions.values()]
        all_y = [p[1] for p in abs_positions.values()]
        if all_x and all_y:
            bbox = (min(all_x) - 1, min(all_y) - 1, max(all_x) + 1, max(all_y) + 1)
        else:
            bbox = (cx - gw / 2, cy - gh / 2, cx + gw / 2, cy + gh / 2)

        origin = (cx - gw / 2.0, cy - gh / 2.0)

        placed.append(PlacedGroup(
            name=group.name,
            zone=zone_name,
            origin=origin,
            refs=tuple(sorted(abs_positions.keys())),
            positions=abs_positions,
            bbox=bbox,
        ))

        _log.info(
            "  Placed group '%s' in zone '%s' at (%.1f, %.1f) [%.0fx%.0fmm]",
            group.name, zone_name, cx, cy, gw, gh,
        )

    _log.info("Group placement complete: %d groups placed", len(placed))
    return placed


def pin_connectors_to_edge(
    placed_groups: list[PlacedGroup],
    fp_sizes: dict[str, tuple[float, float]],
    board_bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
) -> dict[str, tuple[float, float]]:
    """Slide connectors in placed groups to the nearest board edge.

    Modifies connector positions in-place while preserving other component
    positions. Returns a merged dict of all ref → (x, y) positions.

    Args:
        placed_groups: Groups from ``place_groups()``.
        fp_sizes: ref → (width, height).
        board_bounds: Board outline bounds.
        fixed_refs: Refs that must not be moved.

    Returns:
        Merged position dict for all refs across all groups.
    """
    bx1, by1, bx2, by2 = board_bounds
    margin = _CONNECTOR_EDGE_MARGIN_MM

    all_positions: dict[str, tuple[float, float]] = {}
    for pg in placed_groups:
        all_positions.update(pg.positions)

    # Slide connectors to nearest edge
    for pg in placed_groups:
        for ref, (rx, ry) in pg.positions.items():
            if ref in fixed_refs or not ref.startswith("J"):
                continue
            w, h = fp_sizes.get(ref, (2.0, 2.0))

            dist_left = rx - bx1
            dist_right = bx2 - rx
            dist_top = ry - by1
            dist_bottom = by2 - ry
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

            # Only pin if not already near edge
            if min_dist <= 8.0:
                continue

            new_x, new_y = rx, ry
            if min_dist == dist_left:
                new_x = bx1 + margin + w / 2
            elif min_dist == dist_right:
                new_x = bx2 - margin - w / 2
            elif min_dist == dist_top:
                new_y = by1 + margin + h / 2
            else:
                new_y = by2 - margin - h / 2

            all_positions[ref] = (new_x, new_y)

    return all_positions


def place_ferrites_at_boundaries(
    ferrite_refs: list[str],
    group_positions: dict[str, tuple[float, float]],
    zones: list[BoardZone],
    group_map: dict[str, str],
    fp_sizes: dict[str, tuple[float, float]],
    board_bounds: tuple[float, float, float, float],
) -> dict[str, tuple[float, float]]:
    """Place ferrites (L* refs) at boundaries between zones they bridge.

    A ferrite bridges two zones if the components it connects (via netlist)
    belong to different zones. Place the ferrite at the midpoint between
    the two zone centers.

    Args:
        ferrite_refs: List of ferrite/inductor refs to potentially relocate.
        group_positions: All ref → (x, y) positions.
        zones: Board zones.
        group_map: ref → group_name mapping.
        fp_sizes: ref → (width, height).
        board_bounds: Board outline bounds.

    Returns:
        Updated positions for ferrite refs only.
    """
    from kicad_pipeline.optimization.zone_partitioner import zone_center, zone_for_group

    result: dict[str, tuple[float, float]] = {}
    bx1, by1, bx2, by2 = board_bounds

    for ref in ferrite_refs:
        gname = group_map.get(ref)
        if gname is None:
            continue

        source_zone = zone_for_group(gname, zones)
        if source_zone is None:
            continue

        # For now, place ferrites at the edge of their zone nearest to board center
        # Future: detect which zone boundary the ferrite bridges via netlist
        zx1, zy1, zx2, zy2 = source_zone.rect
        zcx, zcy = zone_center(source_zone)
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        # Place at zone edge nearest to board center
        w, h = fp_sizes.get(ref, (2.0, 2.0))
        if abs(zcx - bcx) > abs(zcy - bcy):
            # Horizontal separation — place at left or right zone edge
            fx = zx1 + w / 2 if zcx > bcx else zx2 - w / 2
            fy = zcy
        else:
            # Vertical separation — place at top or bottom zone edge
            fx = zcx
            fy = zy1 + h / 2 if zcy > bcy else zy2 - h / 2

        result[ref] = (fx, fy)

    return result
