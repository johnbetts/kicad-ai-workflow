"""Centralized 2D geometry utilities for placement optimization.

Provides polygon containment, centroid, area, bounding box, and
clamping operations used by the zone partitioner, group placer,
collision resolver, and placement guard.

Consolidates duplicated implementations from ``evals/dfm_gates.py``,
``pcb/keepout_builder.py``, and ``optimization/board_state.py``.
"""

from __future__ import annotations

import math

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


def convex_hull(points: tuple[Point, ...]) -> tuple[Point, ...]:
    """Compute the convex hull of a point set (Andrew's monotone chain).

    Returns hull vertices in counter-clockwise order (in the PCB
    coordinate convention where Y grows downward, this is screen-space
    clockwise). Collinear interior points are dropped. Degenerate
    inputs (<3 distinct points) are returned deduplicated and sorted.
    """
    pts = sorted({(p.x, p.y) for p in points})
    if len(pts) < 3:
        return tuple(Point(x, y) for x, y in pts)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return tuple(Point(x, y) for x, y in hull)


def transform_polygon(
    polygon: tuple[Point, ...], dx: float, dy: float, rotation_deg: float = 0.0,
) -> tuple[Point, ...]:
    """Rotate a polygon about the origin, then translate it.

    Rotation is counter-clockwise in mathematical convention; in the
    PCB coordinate system (Y down) a positive angle appears clockwise
    on screen, matching KiCad footprint rotation.
    """
    if rotation_deg % 360.0 == 0.0:
        return tuple(Point(p.x + dx, p.y + dy) for p in polygon)
    rad = math.radians(rotation_deg)
    c, s = math.cos(rad), math.sin(rad)
    return tuple(
        Point(p.x * c - p.y * s + dx, p.x * s + p.y * c + dy) for p in polygon
    )


def inflate_convex_polygon(
    polygon: tuple[Point, ...], margin_mm: float,
) -> tuple[Point, ...]:
    """Offset a convex polygon outward by *margin_mm* (miter joins).

    Each edge is shifted along its outward normal and consecutive
    offset edges are re-intersected. The polygon may be given in either
    winding order. Margin 0 returns the polygon unchanged.
    """
    n = len(polygon)
    if n < 3 or margin_mm == 0.0:
        return polygon

    # Determine winding via the shoelace sign so normals point outward.
    signed = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed += polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y
    sign = 1.0 if signed > 0 else -1.0

    offset_lines: list[tuple[float, float, float, float]] = []
    for i in range(n):
        j = (i + 1) % n
        ex, ey = polygon[j].x - polygon[i].x, polygon[j].y - polygon[i].y
        length = math.hypot(ex, ey)
        if length < 1e-12:
            continue
        # Outward normal for this winding.
        nx, ny = sign * ey / length, -sign * ex / length
        offset_lines.append(
            (polygon[i].x + nx * margin_mm, polygon[i].y + ny * margin_mm, ex, ey)
        )

    result: list[Point] = []
    m = len(offset_lines)
    for i in range(m):
        ax, ay, adx, ady = offset_lines[i - 1]
        bx, by, bdx, bdy = offset_lines[i]
        denom = adx * bdy - ady * bdx
        if abs(denom) < 1e-12:
            result.append(Point(bx, by))  # parallel edges — use edge start
            continue
        t = ((bx - ax) * bdy - (by - ay) * bdx) / denom
        result.append(Point(ax + t * adx, ay + t * ady))
    return tuple(result)


def _project_polygon(
    polygon: tuple[Point, ...], ax: float, ay: float,
) -> tuple[float, float]:
    """Project polygon vertices onto axis (ax, ay); return (min, max)."""
    dots = [p.x * ax + p.y * ay for p in polygon]
    return (min(dots), max(dots))


def _sat_axes(polygon: tuple[Point, ...]) -> list[tuple[float, float]]:
    """Edge-normal axes for separating-axis tests."""
    axes: list[tuple[float, float]] = []
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        ex, ey = polygon[j].x - polygon[i].x, polygon[j].y - polygon[i].y
        length = math.hypot(ex, ey)
        if length > 1e-12:
            axes.append((ey / length, -ex / length))
    return axes


def convex_polygons_overlap(
    a: tuple[Point, ...], b: tuple[Point, ...], clearance_mm: float = 0.0,
) -> bool:
    """Separating-axis overlap test for two convex polygons.

    With a positive *clearance_mm*, also returns ``True`` when the
    polygons are closer than the clearance (treats near-touching as
    overlap).
    """
    if len(a) < 3 or len(b) < 3:
        return False
    for ax, ay in _sat_axes(a) + _sat_axes(b):
        a_min, a_max = _project_polygon(a, ax, ay)
        b_min, b_max = _project_polygon(b, ax, ay)
        if a_max + clearance_mm < b_min or b_max + clearance_mm < a_min:
            return False
    return True


def segment_distance(
    a1x: float, a1y: float, a2x: float, a2y: float,
    b1x: float, b1y: float, b2x: float, b2y: float,
) -> float:
    """Minimum distance between two line segments."""
    d = math.inf
    for px, py, sx1, sy1, sx2, sy2 in (
        (a1x, a1y, b1x, b1y, b2x, b2y),
        (a2x, a2y, b1x, b1y, b2x, b2y),
        (b1x, b1y, a1x, a1y, a2x, a2y),
        (b2x, b2y, a1x, a1y, a2x, a2y),
    ):
        cx, cy = closest_point_on_segment(px, py, sx1, sy1, sx2, sy2)
        d = min(d, math.hypot(cx - px, cy - py))
    return d


def convex_polygon_gap(a: tuple[Point, ...], b: tuple[Point, ...]) -> float:
    """Minimum edge-to-edge gap between two convex polygons.

    Returns 0.0 when the polygons overlap or touch.
    """
    if convex_polygons_overlap(a, b):
        return 0.0
    na, nb = len(a), len(b)
    gap = math.inf
    for i in range(na):
        i2 = (i + 1) % na
        for j in range(nb):
            j2 = (j + 1) % nb
            gap = min(
                gap,
                segment_distance(
                    a[i].x, a[i].y, a[i2].x, a[i2].y,
                    b[j].x, b[j].y, b[j2].x, b[j2].y,
                ),
            )
    return gap
