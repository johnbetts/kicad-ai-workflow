"""Tests for the centralized geometry module."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.optimization.geometry import (
    clamp_to_polygon,
    closest_point_on_segment,
    convex_hull,
    convex_polygon_gap,
    convex_polygons_overlap,
    inflate_convex_polygon,
    point_in_polygon,
    polygon_area,
    polygon_bbox,
    polygon_centroid,
    polygon_dimensions,
    transform_polygon,
)


def _unit_square(cx: float, cy: float, half: float) -> tuple[Point, ...]:
    return (
        Point(cx - half, cy - half),
        Point(cx + half, cy - half),
        Point(cx + half, cy + half),
        Point(cx - half, cy + half),
    )


@pytest.fixture
def square() -> tuple[Point, ...]:
    """10x10 square at origin."""
    return (Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10))


@pytest.fixture
def l_shape() -> tuple[Point, ...]:
    """L-shaped polygon: 20x10 bottom, 10x10 top-left."""
    return (
        Point(0, 0), Point(20, 0), Point(20, 10),
        Point(10, 10), Point(10, 20), Point(0, 20),
    )


class TestPointInPolygon:
    def test_inside_square(self, square: tuple[Point, ...]) -> None:
        assert point_in_polygon(5, 5, square) is True

    def test_outside_square(self, square: tuple[Point, ...]) -> None:
        assert point_in_polygon(15, 5, square) is False

    def test_inside_l_shape(self, l_shape: tuple[Point, ...]) -> None:
        assert point_in_polygon(5, 5, l_shape) is True
        assert point_in_polygon(5, 15, l_shape) is True  # upper-left arm

    def test_outside_l_shape(self, l_shape: tuple[Point, ...]) -> None:
        assert point_in_polygon(15, 15, l_shape) is False  # notch

    def test_degenerate(self) -> None:
        assert point_in_polygon(0, 0, ()) is False
        assert point_in_polygon(0, 0, (Point(0, 0), Point(1, 0))) is False


class TestPolygonBbox:
    def test_square(self, square: tuple[Point, ...]) -> None:
        assert polygon_bbox(square) == (0, 0, 10, 10)

    def test_l_shape(self, l_shape: tuple[Point, ...]) -> None:
        assert polygon_bbox(l_shape) == (0, 0, 20, 20)

    def test_empty(self) -> None:
        assert polygon_bbox(()) == (0.0, 0.0, 0.0, 0.0)


class TestPolygonArea:
    def test_square(self, square: tuple[Point, ...]) -> None:
        assert polygon_area(square) == pytest.approx(100.0)

    def test_l_shape(self, l_shape: tuple[Point, ...]) -> None:
        # 20x10 + 10x10 = 300
        assert polygon_area(l_shape) == pytest.approx(300.0)


class TestPolygonCentroid:
    def test_square(self, square: tuple[Point, ...]) -> None:
        cx, cy = polygon_centroid(square)
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    def test_degenerate(self) -> None:
        assert polygon_centroid(()) == (0.0, 0.0)


class TestPolygonDimensions:
    def test_square(self, square: tuple[Point, ...]) -> None:
        assert polygon_dimensions(square) == (10, 10)


class TestClosestPointOnSegment:
    def test_midpoint(self) -> None:
        cx, cy = closest_point_on_segment(5, 5, 0, 0, 10, 0)
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(0.0)

    def test_clamped_to_start(self) -> None:
        cx, cy = closest_point_on_segment(-5, 0, 0, 0, 10, 0)
        assert cx == pytest.approx(0.0)
        assert cy == pytest.approx(0.0)

    def test_clamped_to_end(self) -> None:
        cx, cy = closest_point_on_segment(15, 0, 0, 0, 10, 0)
        assert cx == pytest.approx(10.0)
        assert cy == pytest.approx(0.0)


class TestClampToPolygon:
    def test_inside_unchanged(self, square: tuple[Point, ...]) -> None:
        assert clamp_to_polygon(5, 5, square) == (5, 5)

    def test_outside_projects_to_edge(self, square: tuple[Point, ...]) -> None:
        cx, cy = clamp_to_polygon(15, 5, square)
        assert cx == pytest.approx(10.0)
        assert cy == pytest.approx(5.0)

    def test_corner(self, square: tuple[Point, ...]) -> None:
        cx, cy = clamp_to_polygon(15, 15, square)
        assert cx == pytest.approx(10.0)
        assert cy == pytest.approx(10.0)

    def test_l_shape_notch(self, l_shape: tuple[Point, ...]) -> None:
        # Point in the notch (15, 15) — should clamp to nearest edge
        cx, cy = clamp_to_polygon(15, 15, l_shape)
        # Nearest edge is either (10, 15) on the vertical edge or (15, 10) on horizontal
        dist_to_vertical = abs(cx - 10)
        dist_to_horizontal = abs(cy - 10)
        assert min(dist_to_vertical, dist_to_horizontal) < 0.1


class TestConvexHull:
    def test_square_with_interior_point(self) -> None:
        hull = convex_hull((*_unit_square(0, 0, 1), Point(0.0, 0.0)))
        assert len(hull) == 4
        assert {(p.x, p.y) for p in hull} == {(-1, -1), (1, -1), (1, 1), (-1, 1)}

    def test_drops_collinear_points(self) -> None:
        pts = (Point(0, 0), Point(1, 0), Point(2, 0), Point(2, 1), Point(0, 1))
        hull = convex_hull(pts)
        assert {(p.x, p.y) for p in hull} == {(0, 0), (2, 0), (2, 1), (0, 1)}

    def test_degenerate_two_points(self) -> None:
        assert len(convex_hull((Point(1, 1), Point(2, 2), Point(1, 1)))) == 2

    def test_hull_area(self) -> None:
        hull = convex_hull(_unit_square(5, 5, 2))
        assert polygon_area(hull) == pytest.approx(16.0)


class TestTransformPolygon:
    def test_translate_only(self) -> None:
        moved = transform_polygon(_unit_square(0, 0, 1), 10.0, 5.0)
        assert {(p.x, p.y) for p in moved} == {(9, 4), (11, 4), (11, 6), (9, 6)}

    def test_rotate_90_about_origin(self) -> None:
        (p,) = transform_polygon((Point(1.0, 0.0),), 0.0, 0.0, 90.0)
        assert p.x == pytest.approx(0.0, abs=1e-9)
        assert p.y == pytest.approx(1.0)

    def test_rotation_preserves_area(self) -> None:
        rotated = transform_polygon(_unit_square(0, 0, 1.5), 3.0, 4.0, 45.0)
        assert polygon_area(rotated) == pytest.approx(9.0)

    def test_full_turn_is_identity(self) -> None:
        poly = _unit_square(2, 3, 1)
        out = transform_polygon(poly, 0.0, 0.0, 360.0)
        for a, b in zip(poly, out, strict=False):
            assert a.x == pytest.approx(b.x)
            assert a.y == pytest.approx(b.y)


class TestInflateConvexPolygon:
    def test_inflate_square_grows_to_3x3(self) -> None:
        out = inflate_convex_polygon(_unit_square(0, 0, 1), 0.5)
        assert polygon_area(out) == pytest.approx(9.0)

    def test_inflate_zero_is_identity(self) -> None:
        poly = _unit_square(0, 0, 1)
        assert inflate_convex_polygon(poly, 0.0) == poly

    def test_inflate_handles_either_winding(self) -> None:
        cw = tuple(reversed(_unit_square(0, 0, 1)))
        out = inflate_convex_polygon(cw, 0.5)
        assert polygon_area(out) == pytest.approx(9.0)


class TestOverlapAndGap:
    def test_disjoint_squares_do_not_overlap(self) -> None:
        assert not convex_polygons_overlap(_unit_square(0, 0, 1), _unit_square(5, 0, 1))

    def test_intersecting_squares_overlap(self) -> None:
        assert convex_polygons_overlap(_unit_square(0, 0, 1), _unit_square(1.5, 0, 1))

    def test_clearance_treats_near_touch_as_overlap(self) -> None:
        a, b = _unit_square(0, 0, 1), _unit_square(2.4, 0, 1)  # gap = 0.4
        assert not convex_polygons_overlap(a, b)
        assert convex_polygons_overlap(a, b, clearance_mm=0.5)

    def test_gap_between_separated_squares(self) -> None:
        gap = convex_polygon_gap(_unit_square(0, 0, 1), _unit_square(4, 0, 1))
        assert gap == pytest.approx(2.0)

    def test_gap_zero_when_overlapping(self) -> None:
        assert convex_polygon_gap(_unit_square(0, 0, 1), _unit_square(0.5, 0.5, 1)) == 0.0

    def test_diagonal_gap(self) -> None:
        gap = convex_polygon_gap(_unit_square(0, 0, 1), _unit_square(3, 3, 1))
        assert gap == pytest.approx(2.0**0.5)
