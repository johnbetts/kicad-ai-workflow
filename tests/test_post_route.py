"""Tests for kicad_pipeline.routing.post_route."""

from __future__ import annotations

import math

import pytest

from kicad_pipeline.models.pcb import Point, Track, Via
from kicad_pipeline.routing.grid_router import RouteResult
from kicad_pipeline.routing.post_route import (
    collect_tracks,
    collect_vias,
    point_to_segment_dist,
    segment_min_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _track(
    x1: float, y1: float, x2: float, y2: float,
    net: int = 1, layer: str = "F.Cu", width: float = 0.25,
) -> Track:
    return Track(
        start=Point(x1, y1),
        end=Point(x2, y2),
        width=width,
        layer=layer,
        net_number=net,
    )


def _via(x: float, y: float, net: int = 1, size: float = 0.6) -> Via:
    return Via(
        position=Point(x, y),
        drill=0.3,
        size=size,
        layers=("F.Cu", "B.Cu"),
        net_number=net,
    )


def _route_result(
    net_num: int,
    net_name: str,
    tracks: tuple[Track, ...] = (),
    vias: tuple[Via, ...] = (),
    routed: bool = True,
) -> RouteResult:
    return RouteResult(
        net_number=net_num,
        net_name=net_name,
        tracks=tracks,
        vias=vias,
        routed=routed,
    )


# ---------------------------------------------------------------------------
# point_to_segment_dist
# ---------------------------------------------------------------------------


def test_point_to_segment_dist_on_segment() -> None:
    """Point on the segment has zero distance."""
    d = point_to_segment_dist(5.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_point_to_segment_dist_perpendicular() -> None:
    """Perpendicular distance from point to horizontal segment."""
    d = point_to_segment_dist(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
    assert d == pytest.approx(3.0, abs=1e-9)


def test_point_to_segment_dist_beyond_endpoint() -> None:
    """Point beyond segment end → distance to nearest endpoint."""
    d = point_to_segment_dist(15.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert d == pytest.approx(5.0, abs=1e-9)


def test_point_to_segment_dist_before_startpoint() -> None:
    """Point before segment start → distance to start."""
    d = point_to_segment_dist(-3.0, 4.0, 0.0, 0.0, 10.0, 0.0)
    assert d == pytest.approx(5.0, abs=1e-9)


def test_point_to_segment_dist_zero_length_segment() -> None:
    """Zero-length segment → distance is Euclidean to that point."""
    d = point_to_segment_dist(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
    assert d == pytest.approx(5.0, abs=1e-9)


def test_point_to_segment_dist_diagonal_segment() -> None:
    """Point near diagonal segment."""
    # Segment from (0,0) to (10,10), point at (0,10)
    d = point_to_segment_dist(0.0, 10.0, 0.0, 0.0, 10.0, 10.0)
    # Closest point on segment is (5,5), distance = sqrt(25+25) = ~7.07
    expected = math.sqrt(50.0)
    assert d == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# segment_min_distance
# ---------------------------------------------------------------------------


def test_segment_min_distance_intersecting() -> None:
    """Intersecting segments have distance 0."""
    d = segment_min_distance(0, 0, 10, 10, 0, 10, 10, 0)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_segment_min_distance_parallel_horizontal() -> None:
    """Two parallel horizontal segments with known gap."""
    d = segment_min_distance(0, 0, 10, 0, 0, 5, 10, 5)
    assert d == pytest.approx(5.0, abs=1e-9)


def test_segment_min_distance_collinear_gap() -> None:
    """Two collinear segments with a gap between them."""
    d = segment_min_distance(0, 0, 5, 0, 8, 0, 15, 0)
    assert d == pytest.approx(3.0, abs=1e-9)


def test_segment_min_distance_perpendicular_non_intersecting() -> None:
    """Perpendicular segments that don't intersect."""
    d = segment_min_distance(0, 0, 5, 0, 3, 2, 3, 10)
    assert d == pytest.approx(2.0, abs=1e-9)


def test_segment_min_distance_same_segment() -> None:
    """Distance of a segment to itself is 0."""
    d = segment_min_distance(0, 0, 10, 0, 0, 0, 10, 0)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_segment_min_distance_zero_length_both() -> None:
    """Two zero-length segments (points)."""
    d = segment_min_distance(3, 0, 3, 0, 0, 4, 0, 4)
    assert d == pytest.approx(5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# collect_tracks
# ---------------------------------------------------------------------------


def test_collect_tracks_empty_results() -> None:
    """Empty results yield empty tracks."""
    assert collect_tracks(()) == ()


def test_collect_tracks_single_routed() -> None:
    """Single routed result returns its tracks."""
    t = _track(0, 0, 10, 0)
    r = _route_result(1, "NET1", tracks=(t,))
    result = collect_tracks((r,))
    assert len(result) == 1


def test_collect_tracks_skips_unrouted() -> None:
    """Unrouted results are skipped by default."""
    t = _track(0, 0, 10, 0)
    r = _route_result(1, "NET1", tracks=(t,), routed=False)
    result = collect_tracks((r,))
    assert len(result) == 0


def test_collect_tracks_includes_unrouted_when_disabled() -> None:
    """routed_only=False includes unrouted tracks."""
    t = _track(0, 0, 10, 0)
    r = _route_result(1, "NET1", tracks=(t,), routed=False)
    result = collect_tracks((r,), routed_only=False)
    assert len(result) == 1


def test_collect_tracks_multiple_results() -> None:
    """Tracks from multiple results are flattened."""
    t1 = _track(0, 0, 10, 0, net=1)
    t2 = _track(20, 0, 30, 0, net=2)
    r1 = _route_result(1, "N1", tracks=(t1,))
    r2 = _route_result(2, "N2", tracks=(t2,))
    result = collect_tracks((r1, r2))
    assert len(result) == 2


def test_collect_tracks_filters_dangling_stubs() -> None:
    """Dangling single-segment stubs are filtered by default."""
    # Two tracks in the same net: one connected, one orphan
    t1 = _track(0, 0, 10, 0, net=1)
    t2 = _track(10, 0, 20, 0, net=1)
    t3 = _track(50, 50, 60, 60, net=1)  # orphan — neither endpoint shared
    r = _route_result(1, "N1", tracks=(t1, t2, t3))
    result = collect_tracks((r,))
    # t3 is orphan (both endpoints unique), should be filtered
    assert len(result) == 2


def test_collect_tracks_no_dangling_filter() -> None:
    """filter_dangling=False keeps all tracks."""
    t1 = _track(0, 0, 10, 0, net=1)
    t2 = _track(50, 50, 60, 60, net=1)  # orphan
    r = _route_result(1, "N1", tracks=(t1, t2))
    result = collect_tracks((r,), filter_dangling=False)
    assert len(result) == 2


def test_collect_tracks_single_segment_net_kept() -> None:
    """Single-segment net (direct pad-to-pad) is kept, not filtered."""
    t = _track(0, 0, 10, 0, net=1)
    r = _route_result(1, "N1", tracks=(t,))
    result = collect_tracks((r,))
    assert len(result) == 1


# ---------------------------------------------------------------------------
# collect_vias
# ---------------------------------------------------------------------------


def test_collect_vias_empty() -> None:
    """No results → no vias."""
    assert collect_vias(()) == ()


def test_collect_vias_single() -> None:
    """Single via from a routed result."""
    v = _via(10, 20, net=1)
    r = _route_result(1, "N1", vias=(v,))
    result = collect_vias((r,))
    assert len(result) == 1


def test_collect_vias_skips_unrouted() -> None:
    """Vias from unrouted nets are skipped by default."""
    v = _via(10, 20, net=1)
    r = _route_result(1, "N1", vias=(v,), routed=False)
    result = collect_vias((r,))
    assert len(result) == 0


def test_collect_vias_includes_unrouted_when_disabled() -> None:
    """routed_only=False includes vias from unrouted nets."""
    v = _via(10, 20, net=1)
    r = _route_result(1, "N1", vias=(v,), routed=False)
    result = collect_vias((r,), routed_only=False)
    assert len(result) == 1


def test_collect_vias_deduplicates_same_position() -> None:
    """Vias at the same position are deduplicated."""
    v1 = _via(10.0, 20.0, net=1)
    v2 = _via(10.0, 20.0, net=1)  # same position
    r = _route_result(1, "N1", vias=(v1, v2))
    result = collect_vias((r,))
    assert len(result) == 1


def test_collect_vias_keeps_different_positions() -> None:
    """Vias at different positions are all kept."""
    v1 = _via(10, 20, net=1)
    v2 = _via(30, 40, net=1)
    r = _route_result(1, "N1", vias=(v1, v2))
    result = collect_vias((r,))
    assert len(result) == 2


def test_collect_vias_deduplicates_close_same_net() -> None:
    """Vias too close on the same net are deduplicated (hole-to-hole safety)."""
    v1 = _via(10.0, 20.0, net=1, size=0.6)
    # v2 is 0.3mm away — within via.size (0.6mm)
    v2 = _via(10.3, 20.0, net=1, size=0.6)
    r = _route_result(1, "N1", vias=(v1, v2))
    result = collect_vias((r,))
    assert len(result) == 1


def test_collect_vias_different_nets_not_deduplicated() -> None:
    """Close vias on different nets are not distance-deduplicated."""
    v1 = _via(10.0, 20.0, net=1, size=0.6)
    v2 = _via(10.3, 20.0, net=2, size=0.6)
    r1 = _route_result(1, "N1", vias=(v1,))
    r2 = _route_result(2, "N2", vias=(v2,))
    result = collect_vias((r1, r2))
    assert len(result) == 2
