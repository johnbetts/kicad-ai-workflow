"""Centralized 2D geometry utilities for placement optimization.

Provides polygon containment, centroid, area, bounding box, and
clamping operations used by the zone partitioner, group placer,
collision resolver, and placement guard.

Consolidates duplicated implementations from ``evals/dfm_gates.py``,
``pcb/keepout_builder.py``, and ``optimization/board_state.py``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Point


def point_in_polygon(
    px: float, py: float, polygon: tuple[Point, ...],
) -> bool:
    """Ray-casting point-in-polygon test.

    Returns ``True`` if *(px, py)* is strictly inside or on the boundary
    of the polygon defined by *polygon* (ordered vertices).
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[j].x, polygon[j].y
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def polygon_bbox(
    polygon: tuple[Point, ...],
) -> tuple[float, float, float, float]:
    """Return axis-aligned bounding box ``(x_min, y_min, x_max, y_max)``."""
    if not polygon:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p.x for p in polygon]
    ys = [p.y for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def polygon_area(polygon: tuple[Point, ...]) -> float:
    """Compute area of a simple polygon using the shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i].x * polygon[j].y
        area -= polygon[j].x * polygon[i].y
    return abs(area) / 2.0


def polygon_centroid(polygon: tuple[Point, ...]) -> tuple[float, float]:
    """Compute centroid of a simple polygon.

    Uses the signed-area-weighted vertex formula. Falls back to the
    AABB center if the polygon is degenerate (zero area).
    """
    n = len(polygon)
    if n == 0:
        return (0.0, 0.0)
    if n < 3:
        cx = sum(p.x for p in polygon) / n
        cy = sum(p.y for p in polygon) / n
        return (cx, cy)

    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y
        signed_area += cross
        cx += (polygon[i].x + polygon[j].x) * cross
        cy += (polygon[i].y + polygon[j].y) * cross

    if abs(signed_area) < 1e-12:
        # Degenerate — fall back to AABB center
        bbox = polygon_bbox(polygon)
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    signed_area /= 2.0
    cx /= (6.0 * signed_area)
    cy /= (6.0 * signed_area)
    return (cx, cy)


def polygon_dimensions(polygon: tuple[Point, ...]) -> tuple[float, float]:
    """Return ``(width, height)`` of the axis-aligned bounding box."""
    x1, y1, x2, y2 = polygon_bbox(polygon)
    return (x2 - x1, y2 - y1)


def closest_point_on_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> tuple[float, float]:
    """Return the closest point on segment AB to point P.

    Projects P onto the line through A and B, then clamps the
    parameter to [0, 1] so the result lies on the segment.
    """
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return (ax, ay)  # degenerate segment
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy)


def clamp_to_polygon(
    px: float, py: float, polygon: tuple[Point, ...],
) -> tuple[float, float]:
    """Project a point to the nearest polygon edge if outside.

    If *(px, py)* is inside the polygon, returns it unchanged.
    Otherwise, returns the closest point on any polygon edge.
    """
    if point_in_polygon(px, py, polygon):
        return (px, py)

    n = len(polygon)
    if n < 2:
        return (px, py)

    best_x, best_y = px, py
    best_dist_sq = math.inf
    for i in range(n):
        j = (i + 1) % n
        cx, cy = closest_point_on_segment(
            px, py,
            polygon[i].x, polygon[i].y,
            polygon[j].x, polygon[j].y,
        )
        dist_sq = (cx - px) ** 2 + (cy - py) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_x, best_y = cx, cy

    return (best_x, best_y)
