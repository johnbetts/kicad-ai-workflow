"""Tests for the centralized geometry module."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.optimization.geometry import (
    clamp_to_polygon,
    closest_point_on_segment,
    point_in_polygon,
    polygon_area,
    polygon_bbox,
    polygon_centroid,
    polygon_dimensions,
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
