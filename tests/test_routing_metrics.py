"""Tests for kicad_pipeline.routing.metrics."""

from __future__ import annotations

from kicad_pipeline.models.pcb import Footprint, Pad, Point, Track, Via
from kicad_pipeline.routing.grid_router import RouteResult
from kicad_pipeline.routing.metrics import BoardRoutingMetrics, compute_board_metrics


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
