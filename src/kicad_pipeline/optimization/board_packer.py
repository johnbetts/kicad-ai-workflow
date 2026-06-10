"""Board packer — places group polygons on the board with edge pinning.

Stage 3 of the 4-stage bottom-up placement pipeline.  Takes rigid
``PlacedGroup`` units (from Stage 2) and positions them within their
assigned ``BoardZone`` regions, pinning connector-bearing groups to
the nearest board edge.  Produces absolute ``(x, y, rotation)`` for
every component reference.

Pipeline:
  Stage 1 — zone_partitioner: partition board into zones
  Stage 2 — group_placer: lay out components within each group
  **Stage 3 — board_packer: place groups on board, pin connectors**
  Stage 4 — refinement / collision resolution
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.group_placer import PlacedGroup
    from kicad_pipeline.optimization.zone_partitioner import BoardZone

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum clearance from board edge for all components (mm).
_BOARD_EDGE_MARGIN_MM: float = 3.0

# Connectors are pinned within this distance of the board edge (mm).
_CONNECTOR_EDGE_MARGIN_MM: float = 3.0

# Reference designator prefixes that indicate connector components.
_CONNECTOR_PREFIXES: tuple[str, ...] = ("J", "P", "CN", "TB")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pack_groups_on_board(
    groups: list[PlacedGroup],
    bounds: tuple[float, float, float, float],
    zones: list[BoardZone],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """Place group polygons on the board and return absolute positions.

    Parameters
    ----------
    groups:
        Rigid groups from Stage 2 (group_placer).
    bounds:
        Board outline ``(x_min, y_min, x_max, y_max)`` in mm.
    zones:
        Non-overlapping board zones from Stage 1 (zone_partitioner).
    requirements:
        Project requirements (used for future net-aware heuristics).

    Returns
    -------
    dict mapping each component ref to ``(x, y, rotation_deg)`` in
    absolute board coordinates.
    """
    zone_lookup: dict[str, BoardZone] = {z.name: z for z in zones}
    board_cx = (bounds[0] + bounds[2]) / 2.0
    board_cy = (bounds[1] + bounds[3]) / 2.0

    result: dict[str, tuple[float, float, float]] = {}

    for group in groups:
        # --- Determine target center for this group -----------------------
        zone = zone_lookup.get(group.zone)
        if zone is not None:
            target_cx, target_cy = zone.center
            _log.debug(
                "Group %r → zone %r center (%.1f, %.1f)",
                group.name, zone.name, target_cx, target_cy,
            )
        else:
            target_cx, target_cy = board_cx, board_cy
            _log.warning(
                "Group %r has no matching zone %r — placing at board center",
                group.name, group.zone,
            )

        # --- Compute translation from current group center to target ------
        group_cx, group_cy = _group_center(group)
        dx = target_cx - group_cx
        dy = target_cy - group_cy

        # Group positions may be (x, y) or (x, y, rot) — handle both
        translated: dict[str, tuple[float, float, float]] = {}
        for ref, pos in group.positions.items():
            if len(pos) == 3:
                x, y, rot = pos[0], pos[1], pos[2]
            else:
                x, y, rot = pos[0], pos[1], 0.0
            translated[ref] = (x + dx, y + dy, rot)

        # --- Pin connector groups to nearest board edge -------------------
        if _group_has_connectors(group):
            edge = _find_nearest_edge(target_cx, target_cy, bounds)
            shifted = _shift_group_to_edge(group, edge, bounds,
                                           {r: (p[0], p[1]) for r, p in translated.items()})
            translated = {r: (shifted[r][0], shifted[r][1], translated[r][2])
                          for r in translated}
            _log.debug("Group %r pinned to %s edge", group.name, edge)

        # --- Clamp all positions inside board bounds ----------------------
        for ref, (x, y, rot) in translated.items():
            cx = _clamp(x, bounds[0] + _BOARD_EDGE_MARGIN_MM,
                        bounds[2] - _BOARD_EDGE_MARGIN_MM)
            cy = _clamp(y, bounds[1] + _BOARD_EDGE_MARGIN_MM,
                        bounds[3] - _BOARD_EDGE_MARGIN_MM)
            result[ref] = (cx, cy, rot)

    _log.info(
        "Packed %d groups (%d refs) onto board (%.0f x %.0f mm)",
        len(groups),
        len(result),
        bounds[2] - bounds[0],
        bounds[3] - bounds[1],
    )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_nearest_edge(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
) -> str:
    """Return ``'top'``, ``'bottom'``, ``'left'``, or ``'right'``.

    Computes the distance from ``(x, y)`` to each board edge and returns
    the name of the closest one.  In KiCad coordinates, *top* is the
    minimum-Y edge and *bottom* is the maximum-Y edge.
    """
    x_min, y_min, x_max, y_max = bounds
    distances: dict[str, float] = {
        "top": abs(y - y_min),
        "bottom": abs(y - y_max),
        "left": abs(x - x_min),
        "right": abs(x - x_max),
    }
    return min(distances, key=distances.__getitem__)


def _shift_group_to_edge(
    group: PlacedGroup,
    edge: str,
    bounds: tuple[float, float, float, float],
    positions: dict[str, tuple[float, float]],
    margin: float = _CONNECTOR_EDGE_MARGIN_MM,
) -> dict[str, tuple[float, float]]:
    """Shift *positions* so the edge-facing side is within *margin* of *bounds*.

    Only the axis perpendicular to the target edge is adjusted; the
    parallel axis is left unchanged to preserve intra-group layout.
    """
    x_min_b, y_min_b, x_max_b, y_max_b = bounds

    # Determine current extent of the group along the relevant axis.
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    g_xmin, g_xmax = min(xs), max(xs)
    g_ymin, g_ymax = min(ys), max(ys)

    shift_x = 0.0
    shift_y = 0.0

    if edge == "top":
        # Move group so its top row is *margin* below the top board edge.
        shift_y = (y_min_b + margin) - g_ymin
    elif edge == "bottom":
        # Move group so its bottom row is *margin* above the bottom edge.
        shift_y = (y_max_b - margin) - g_ymax
    elif edge == "left":
        shift_x = (x_min_b + margin) - g_xmin
    elif edge == "right":
        shift_x = (x_max_b - margin) - g_xmax

    return {
        ref: (p[0] + shift_x, p[1] + shift_y) for ref, p in positions.items()
    }


def _group_center(group: PlacedGroup) -> tuple[float, float]:
    """Return the centroid of all component positions in *group*."""
    if not group.positions:
        return group.origin
    xs = [p[0] for p in group.positions.values()]
    ys = [p[1] for p in group.positions.values()]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _group_has_connectors(group: PlacedGroup) -> bool:
    """True if *group* contains at least one connector reference."""
    return any(
        ref.lstrip("0123456789") == "" or _ref_is_connector(ref)
        for ref in group.refs
    )


def _ref_is_connector(ref: str) -> bool:
    """True if *ref* starts with a known connector prefix."""
    # Strip trailing digits to get the prefix letters.
    prefix = ref.rstrip("0123456789")
    return prefix in _CONNECTOR_PREFIXES


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to ``[lo, hi]``."""
    if lo > hi:
        # Degenerate case: board too small for margin — use midpoint.
        return (lo + hi) / 2.0
    return max(lo, min(hi, value))
