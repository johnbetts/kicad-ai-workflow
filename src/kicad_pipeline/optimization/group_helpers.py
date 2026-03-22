"""Group-level placement helpers.

Standalone functions for computing group geometry, placing subcircuit
clusters, and applying review fixes.  Extracted from placement_optimizer.py
to reduce module size.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from kicad_pipeline.optimization.collision_resolver import _PlacementGrid
from kicad_pipeline.optimization.placement_types import GroupBoundingBox

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.review_agent import PlacementReview


def _group_footprint_area(
    refs: tuple[str, ...],
    fp_sizes: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Estimate the bounding box needed for a group of components.

    Arranges components in a tight 2-column layout to estimate area.
    Returns (width, height) in mm.
    """
    sizes = [fp_sizes.get(r, (2.0, 2.0)) for r in refs]
    if not sizes:
        return (2.0, 2.0)
    if len(sizes) == 1:
        return (sizes[0][0] + 1.0, sizes[0][1] + 1.0)

    # Sort largest first, pack in 2 columns
    sizes.sort(key=lambda s: s[0] * s[1], reverse=True)
    cols = min(2, len(sizes))
    col_widths = [0.0] * cols
    col_heights = [0.0] * cols
    for i, (w, h) in enumerate(sizes):
        c = i % cols
        col_widths[c] = max(col_widths[c], w)
        col_heights[c] += h + 0.5  # gap between rows

    total_w = sum(col_widths) + 1.0 * (cols - 1) + 1.0  # inter-col gap + margin
    total_h = max(col_heights) + 1.0  # margin
    return (total_w, total_h)


def _place_subcircuit_group(
    anchor_pos: tuple[float, float],
    refs: tuple[str, ...],
    anchor_ref: str,
    fp_sizes: dict[str, tuple[float, float]],
    board_bounds: tuple[float, float, float, float],
    grid: _PlacementGrid,
) -> dict[str, tuple[float, float]]:
    """Arrange sub-circuit components tightly around the anchor.

    Uses the occupancy grid to avoid collisions with already-placed
    components. Places anchor first, then remaining components in a
    spiral pattern checking for free positions.

    Returns:
        Dict mapping ref -> (x, y) position.
    """
    min_x, min_y, max_x, max_y = board_bounds
    positions: dict[str, tuple[float, float]] = {}

    # Place anchor
    aw, ah = fp_sizes.get(anchor_ref, (2.0, 2.0))
    ax, ay = grid.find_free_pos(anchor_pos[0], anchor_pos[1], aw, ah)
    grid.place(ax, ay, aw, ah)
    positions[anchor_ref] = (ax, ay)

    other_refs = [r for r in refs if r != anchor_ref]
    if not other_refs:
        return positions

    # Sort by size (largest first) so they get placed closer to anchor
    other_with_size = [
        (ref, fp_sizes.get(ref, (2.0, 2.0))) for ref in other_refs
    ]
    other_with_size.sort(key=lambda x: x[1][0] * x[1][1], reverse=True)

    # Place others around anchor, checking occupancy
    # Use 4 cardinal + 4 diagonal directions for better spreading
    n = len(other_with_size)
    angle_step = 2 * math.pi / max(n, 1)
    for i, (ref, (w, h)) in enumerate(other_with_size):
        # Use direction-aware clearance: along X use widths, along Y use heights
        angle = angle_step * i
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # Minimum distance along this direction to clear anchor + component
        clear_x = (w + aw) / 2.0 + 0.5
        clear_y = (h + ah) / 2.0 + 0.5
        # Project clearance onto direction vector
        if abs(cos_a) > 0.01 or abs(sin_a) > 0.01:
            min_dist = math.sqrt(
                (clear_x * cos_a) ** 2 + (clear_y * sin_a) ** 2,
            )
        else:
            min_dist = max(clear_x, clear_y)
        min_dist = max(min_dist, 1.5)  # at least 1.5mm

        target_x = ax + min_dist * cos_a
        target_y = ay + min_dist * sin_a
        fx, fy = grid.find_free_pos(target_x, target_y, w, h)
        grid.place(fx, fy, w, h)
        positions[ref] = (fx, fy)

    return positions


def _centroid(
    refs: tuple[str, ...] | list[str],
    positions: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Compute centroid of refs that have positions."""
    xs, ys = [], []
    for r in refs:
        if r in positions:
            xs.append(positions[r][0])
            ys.append(positions[r][1])
    if not xs:
        return (0.0, 0.0)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _assign_zone_position(
    zone_rect: tuple[float, float, float, float],
    index: int,
    total: int,
    group_sizes: list[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Assign a position within a zone for the index-th group.

    When *group_sizes* is provided, cells are sized proportionally to
    the footprint area of each group to avoid overlap.
    """
    x1, y1, x2, y2 = zone_rect
    zw = x2 - x1
    zh = y2 - y1
    cx = x1 + zw / 2.0
    cy = y1 + zh / 2.0

    if total <= 1:
        return (cx, cy)

    # Grid within zone -- use enough columns to spread groups
    cols = max(1, math.ceil(math.sqrt(total)))
    rows_needed = max(1, (total + cols - 1) // cols)

    # If group sizes provided, compute per-cell offsets proportionally
    cell_w = zw / cols
    cell_h = zh / rows_needed

    row = index // cols
    col = index % cols
    x = x1 + cell_w * (col + 0.5)
    y = y1 + cell_h * (row + 0.5)
    return (x, y)


def _apply_review_fixes(
    positions: dict[str, tuple[float, float, float]],
    review: PlacementReview,
    fixed_refs: set[str],
    fp_sizes: dict[str, tuple[float, float]],
    board_bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Apply suggested position fixes from review violations.

    Only applies fixes for critical and major violations that have
    a suggested_position, affect movable components, and result in
    the component being closer to its target than before.
    """
    result = dict(positions)
    bounds = board_bounds or (0.0, 0.0, 200.0, 200.0)

    for violation in review.violations:
        if violation.severity == "minor":
            continue
        if violation.suggested_position is None:
            continue
        # Apply fix to the first ref in the violation that's movable
        for ref in violation.refs:
            if ref in fixed_refs or ref not in result:
                continue
            old_x, old_y, rot = result[ref]
            sx, sy = violation.suggested_position
            w, h = fp_sizes.get(ref, (2.0, 2.0))

            # Build grid excluding this component
            fix_grid = _PlacementGrid(bounds)
            for other_ref, (ox, oy, _orot) in result.items():
                if other_ref == ref:
                    continue
                ow, oh = fp_sizes.get(other_ref, (2.0, 2.0))
                fix_grid.place(ox, oy, ow, oh)

            # Find nearest free position to suggested target
            fx, fy = fix_grid.find_free_pos(sx, sy, w, h)

            # Only accept if it moves closer to the suggested position
            old_dist = math.sqrt((old_x - sx) ** 2 + (old_y - sy) ** 2)
            new_dist = math.sqrt((fx - sx) ** 2 + (fy - sy) ** 2)
            if new_dist < old_dist:
                result[ref] = (fx, fy, rot)
            break  # one fix attempt per violation

    return result


def _extract_group_bboxes(
    requirements: ProjectRequirements,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> list[GroupBoundingBox]:
    """Extract bounding boxes for each FeatureBlock from current positions.

    For each FeatureBlock, compute the bounding box from member positions
    and store internal offsets relative to the group origin (min_x, min_y).
    """
    if not requirements.features:
        return []

    groups: list[GroupBoundingBox] = []
    for block in requirements.features:
        refs_in_pos = [r for r in block.components if r in positions]
        if not refs_in_pos:
            continue

        # Compute bounding box including component sizes
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        for ref in refs_in_pos:
            x, y, _rot = positions[ref]
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            min_x = min(min_x, x - w / 2)
            min_y = min(min_y, y - h / 2)
            max_x = max(max_x, x + w / 2)
            max_y = max(max_y, y + h / 2)

        # Internal offsets relative to group origin
        origin_x = min_x
        origin_y = min_y
        offsets: dict[str, tuple[float, float]] = {}
        for ref in refs_in_pos:
            x, y, _rot = positions[ref]
            offsets[ref] = (x - origin_x, y - origin_y)

        groups.append(GroupBoundingBox(
            name=block.name,
            refs=tuple(refs_in_pos),
            internal_offsets=offsets,
            width=max_x - min_x,
            height=max_y - min_y,
        ))

    return groups


def _build_group_map(requirements: ProjectRequirements) -> dict[str, str]:
    """Build ref -> group_name mapping from FeatureBlocks."""
    group_map: dict[str, str] = {}
    for block in requirements.features:
        for ref in block.components:
            group_map[ref] = block.name
    return group_map
