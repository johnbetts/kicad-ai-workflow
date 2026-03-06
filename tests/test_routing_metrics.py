"""Tests for kicad_pipeline.routing.metrics."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Footprint, Pad, Point, Track, Via
from kicad_pipeline.routing.grid_router import RouteResult
from kicad_pipeline.routing.metrics import (
    BoardRoutingMetrics,
    compute_board_cost,
    compute_board_metrics,
    compute_passive_proximity,
    count_detours,
    count_via_ping_pongs,
)


def _make_footprint(ref: str, x: float, y: float) -> Footprint:
    return Footprint(
        lib_id="Test:Test", ref=ref, value="1k",
        position=Point(x, y), rotation=0.0, layer="F.Cu",
        pads=(
            Pad(number="1", pad_type="smd", shape="rect",
                position=Point(-0.5, 0.0), size_x=0.6, size_y=0.6,
                layers=("F.Cu",)),
            Pad(number="2", pad_type="smd", shape="rect",
                position=Point(0.5, 0.0), size_x=0.6, size_y=0.6,
                layers=("F.Cu",)),
        ),
    )


def test_compute_board_metrics_basic() -> None:
    """Basic metrics computation for a single routed net."""
    results = (
        RouteResult(
            net_number=1, net_name="NET1",
            tracks=(
                Track(start=Point(0, 0), end=Point(10, 0),
                      width=0.25, layer="F.Cu", net_number=1),
            ),
            vias=(),
            routed=True,
        ),
    )
    fps = [_make_footprint("R1", 0, 0), _make_footprint("R2", 10, 0)]
    m = compute_board_metrics(results, fps)
    assert isinstance(m, BoardRoutingMetrics)
    assert m.nets_routed == 1
    assert m.nets_failed == 0
    assert m.total_vias == 0
    assert m.total_track_length_mm > 0


def test_compute_board_metrics_with_failure() -> None:
    """Failed nets are counted correctly."""
    results = (
        RouteResult(
            net_number=1, net_name="NET1",
            tracks=(), vias=(), routed=False, reason="no path",
        ),
        RouteResult(
            net_number=2, net_name="NET2",
            tracks=(
                Track(start=Point(0, 0), end=Point(5, 0),
                      width=0.25, layer="F.Cu", net_number=2),
            ),
            vias=(),
            routed=True,
        ),
    )
    fps = [_make_footprint("R1", 0, 0), _make_footprint("R2", 5, 0)]
    m = compute_board_metrics(results, fps)
    assert m.nets_routed == 1
    assert m.nets_failed == 1


def test_compute_board_metrics_via_count() -> None:
    """Vias are counted in metrics."""
    results = (
        RouteResult(
            net_number=1, net_name="NET1",
            tracks=(
                Track(start=Point(0, 0), end=Point(10, 0),
                      width=0.25, layer="F.Cu", net_number=1),
            ),
            vias=(
                Via(position=Point(5, 0), drill=0.3, size=0.6,
                    layers=("F.Cu", "B.Cu"), net_number=1),
                Via(position=Point(8, 0), drill=0.3, size=0.6,
                    layers=("F.Cu", "B.Cu"), net_number=1),
            ),
            routed=True,
        ),
    )
    fps = [_make_footprint("R1", 0, 0), _make_footprint("R2", 10, 0)]
    m = compute_board_metrics(results, fps)
    assert m.total_vias == 2
    assert m.max_vias_per_net == 2


def test_compute_board_cost_spec_formula() -> None:
    """Board cost follows spec: 1*length + 16*vias + 3*bends + 6*excess + ..."""
    results = (
        RouteResult(
            net_number=1, net_name="NET1",
            tracks=(
                Track(start=Point(0, 0), end=Point(10, 0),
                      width=0.25, layer="F.Cu", net_number=1),
            ),
            vias=(
                Via(position=Point(5, 0), drill=0.3, size=0.6,
                    layers=("F.Cu", "B.Cu"), net_number=1),
            ),
            routed=True,
        ),
    )
    fps = [_make_footprint("R1", 0, 0), _make_footprint("R2", 10, 0)]
    m = compute_board_metrics(results, fps)
    cost = compute_board_cost(m)
    # 10mm length + 16*1 via + 3*0 bends + 0 ratio penalty (ratio=1.0 < 1.55)
    # No failures, no ping-pong, no passive dist penalty in this simple case
    expected_base = m.total_track_length_mm + 16.0 * m.total_vias
    assert cost >= expected_base
    assert cost > 0.0


def test_compute_board_cost_zero_nets() -> None:
    """Board cost is 0 when no nets are routed."""
    m = BoardRoutingMetrics(
        total_track_length_mm=0.0,
        total_vias=0,
        nets_routed=0,
        nets_failed=0,
        overall_length_ratio=1.0,
        max_vias_per_net=0,
        per_net=(),
    )
    assert compute_board_cost(m) == 0.0


def test_compute_board_cost_drc_penalty() -> None:
    """DRC violations add 70x penalty."""
    m = BoardRoutingMetrics(
        total_track_length_mm=0.0,
        total_vias=0,
        nets_routed=1,
        nets_failed=0,
        overall_length_ratio=1.0,
        max_vias_per_net=0,
        per_net=(),
    )
    assert compute_board_cost(m, drc_violations=2) == pytest.approx(140.0)


def test_compute_board_cost_unrouted_penalty() -> None:
    """Unrouted nets add 200x penalty."""
    m = BoardRoutingMetrics(
        total_track_length_mm=0.0,
        total_vias=0,
        nets_routed=0,
        nets_failed=3,
        overall_length_ratio=1.0,
        max_vias_per_net=0,
        per_net=(),
    )
    assert compute_board_cost(m) == pytest.approx(600.0)


def test_compute_board_cost_ping_pong_penalty() -> None:
    """Via ping-pongs add 25x penalty."""
    m = BoardRoutingMetrics(
        total_track_length_mm=0.0,
        total_vias=0,
        nets_routed=1,
        nets_failed=0,
        overall_length_ratio=1.0,
        max_vias_per_net=0,
        per_net=(),
        via_ping_pong_count=2,
    )
    assert compute_board_cost(m) == pytest.approx(50.0)


def test_compute_board_cost_gnd_pour_penalty() -> None:
    """Missing GND pour adds 10x penalty per layer."""
    m = BoardRoutingMetrics(
        total_track_length_mm=0.0,
        total_vias=0,
        nets_routed=1,
        nets_failed=0,
        overall_length_ratio=1.0,
        max_vias_per_net=0,
        per_net=(),
    )
    assert compute_board_cost(m, gnd_pour_missing_layers=2) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Via ping-pong detection
# ---------------------------------------------------------------------------


def test_count_via_ping_pongs_none() -> None:
    """No ping-pong in single-layer route."""
    r = RouteResult(
        net_number=1, net_name="NET1",
        tracks=(
            Track(start=Point(0, 0), end=Point(10, 0), width=0.25,
                  layer="F.Cu", net_number=1),
        ),
        vias=(), routed=True,
    )
    assert count_via_ping_pongs(r) == 0


def test_count_via_ping_pongs_single_transition() -> None:
    """F.Cu→B.Cu without return is not ping-pong."""
    r = RouteResult(
        net_number=1, net_name="NET1",
        tracks=(
            Track(start=Point(0, 0), end=Point(5, 0), width=0.25,
                  layer="F.Cu", net_number=1),
            Track(start=Point(5, 0), end=Point(10, 0), width=0.25,
                  layer="B.Cu", net_number=1),
        ),
        vias=(), routed=True,
    )
    assert count_via_ping_pongs(r) == 0


def test_count_via_ping_pongs_detected() -> None:
    """F.Cu→B.Cu→F.Cu is one ping-pong."""
    r = RouteResult(
        net_number=1, net_name="NET1",
        tracks=(
            Track(start=Point(0, 0), end=Point(3, 0), width=0.25,
                  layer="F.Cu", net_number=1),
            Track(start=Point(3, 0), end=Point(6, 0), width=0.25,
                  layer="B.Cu", net_number=1),
            Track(start=Point(6, 0), end=Point(10, 0), width=0.25,
                  layer="F.Cu", net_number=1),
        ),
        vias=(), routed=True,
    )
    assert count_via_ping_pongs(r) == 1


# ---------------------------------------------------------------------------
# Passive proximity
# ---------------------------------------------------------------------------


def test_passive_proximity_basic() -> None:
    """Passives near ICs should have small distance."""
    fps = [
        _make_footprint("U1", 10.0, 10.0),
        _make_footprint("R1", 12.0, 10.0),
        _make_footprint("C1", 11.0, 12.0),
    ]
    dist = compute_passive_proximity(fps)
    assert 0.0 < dist < 5.0


def test_passive_proximity_no_passives() -> None:
    """No passives returns 0."""
    fps = [_make_footprint("U1", 10.0, 10.0)]
    assert compute_passive_proximity(fps) == 0.0


# ---------------------------------------------------------------------------
# Detour detection
# ---------------------------------------------------------------------------


def test_count_detours_straight_route() -> None:
    """Straight route has no detours."""
    results = (
        RouteResult(
            net_number=1, net_name="NET1",
            tracks=(
                Track(start=Point(0, 0), end=Point(10, 0), width=0.25,
                      layer="F.Cu", net_number=1),
            ),
            vias=(), routed=True,
        ),
    )
    assert count_detours(results, threshold=1.4) == 0


def test_count_detours_detoured_route() -> None:
    """Circuitous route is flagged as detour."""
    # Route that goes 0→10 in X but zigzags 0→5 in Y and back
    results = (
        RouteResult(
            net_number=1, net_name="NET1",
            tracks=(
                Track(start=Point(0, 0), end=Point(0, 10), width=0.25,
                      layer="F.Cu", net_number=1),
                Track(start=Point(0, 10), end=Point(10, 10), width=0.25,
                      layer="F.Cu", net_number=1),
                Track(start=Point(10, 10), end=Point(10, 0), width=0.25,
                      layer="F.Cu", net_number=1),
            ),
            vias=(), routed=True,
        ),
    )
    # Manhattan span = 10+10 = 20, actual = 10+10+10 = 30, ratio = 1.5
    assert count_detours(results, threshold=1.4) == 1
