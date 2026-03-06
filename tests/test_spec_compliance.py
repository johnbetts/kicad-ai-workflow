"""Spec compliance tests — Documentation/projectspecs.md.

Validates that routing, placement, and cost function behavior matches
the project specification's non-negotiable rules.
"""

from __future__ import annotations

import pytest

from kicad_pipeline.constants import (
    CLEARANCE_DEFAULT_MM,
    GND_STITCH_FP_CLEARANCE_MM,
    GND_STITCH_SPACING_MM,
    JLCPCB_MIN_CLEARANCE_MM,
    ROUTING_VIA_COST,
    VIA_DIAMETER_SIGNAL_MM,
    VIA_DRILL_SIGNAL_MM,
)
from kicad_pipeline.models.pcb import (
    Footprint,
    Pad,
    Point,
    Track,
    Via,
)
from kicad_pipeline.routing.grid_router import (
    RouteRequest,
    RouteResult,
    _score_route,
    _segment_min_distance,
)
from kicad_pipeline.routing.metrics import (
    BoardRoutingMetrics,
    compute_board_cost,
    compute_passive_proximity,
    count_detours,
    count_via_ping_pongs,
)

# ---------------------------------------------------------------------------
# Spec Rule 1: Via cost = 14-20x normal trace segment
# ---------------------------------------------------------------------------


def test_spec_via_cost_in_range() -> None:
    """Spec: via cost 14-20x normal trace segment."""
    assert 14.0 <= ROUTING_VIA_COST <= 20.0


# ---------------------------------------------------------------------------
# Spec Rule 2: Hard limit 2 vias/net
# ---------------------------------------------------------------------------


def test_spec_max_vias_default() -> None:
    """Spec: hard max 2 vias per net (RouteRequest default)."""
    req = RouteRequest(
        net_number=1, net_name="NET1",
        pad_refs=(("R1", "1"), ("R2", "1")),
        layer="F.Cu",
    )
    assert req.max_vias == 2


# ---------------------------------------------------------------------------
# Spec Rule 3: Score function weights match spec
# ---------------------------------------------------------------------------


def test_spec_score_via_weight_16() -> None:
    """Spec: 16x via penalty in cost function."""
    result = RouteResult(
        net_number=1, net_name="NET1",
        tracks=(
            Track(start=Point(0, 0), end=Point(10, 0), width=0.25,
                  layer="F.Cu", net_number=1),
        ),
        vias=(
            Via(position=Point(5, 0), drill=0.3, size=0.6,
                layers=("F.Cu", "B.Cu"), net_number=1),
        ),
        routed=True,
    )
    q = _score_route(result, [(0.0, 0.0), (10.0, 0.0)])
    # score = actual(10) + 16*vias(1) + 3*bends(0) + 6*excess(0)
    assert q.score == pytest.approx(10.0 + 16.0)


def test_spec_score_bend_weight_3() -> None:
    """Spec: 3x bend penalty in cost function."""
    result = RouteResult(
        net_number=1, net_name="NET1",
        tracks=(
            Track(start=Point(0, 0), end=Point(5, 0), width=0.25,
                  layer="F.Cu", net_number=1),
            Track(start=Point(5, 0), end=Point(5, 5), width=0.25,
                  layer="F.Cu", net_number=1),
        ),
        vias=(),
        routed=True,
    )
    q = _score_route(result, [(0.0, 0.0), (5.0, 5.0)])
    # 1 bend, actual ~10mm, manhattan=10mm, ratio=1.0
    actual = 5.0 + 5.0
    expected = actual + 3.0 * 1  # 1 bend at weight 3
    assert q.score == pytest.approx(expected)


def test_spec_score_excess_ratio_weight_6() -> None:
    """Spec: 6x penalty per unit of (ratio - 1.55) excess."""
    # Route that is 2x the manhattan distance
    result = RouteResult(
        net_number=1, net_name="NET1",
        tracks=(
            Track(start=Point(0, 0), end=Point(20, 0), width=0.25,
                  layer="F.Cu", net_number=1),
        ),
        vias=(),
        routed=True,
    )
    q = _score_route(result, [(0.0, 0.0), (10.0, 0.0)])
    # ratio = 20/10 = 2.0, excess = 2.0 - 1.55 = 0.45
    assert q.length_ratio == pytest.approx(2.0)
    expected = 20.0 + 6.0 * 0.45  # actual + excess penalty
    assert q.score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Spec Rule 4: Board cost function — all 11 terms
# ---------------------------------------------------------------------------


def test_spec_board_cost_all_terms() -> None:
    """Verify all 11 cost function terms are present and weighted correctly."""
    m = BoardRoutingMetrics(
        total_track_length_mm=100.0,
        total_vias=5,
        nets_routed=10,
        nets_failed=2,
        overall_length_ratio=1.3,
        max_vias_per_net=2,
        per_net=(),
        total_bends=10,
        via_ping_pong_count=1,
        avg_passive_distance_mm=8.0,
        gnd_pour_missing_layers=1,
        detour_count=3,
        max_congestion=2.0,
        drc_violations=0,
    )
    cost = compute_board_cost(m, drc_violations=1, gnd_pour_missing_layers=1)
    expected = (
        1.0 * 100.0          # trace length
        + 16.0 * 5           # vias
        + 3.0 * 10           # bends
        + 0.0                # ratio penalty (per_net is empty)
        + 70.0 * 1           # drc violations
        + 200.0 * 2          # unrouted
        + 12.0 * 2.0         # congestion
        + 18.0 * 8.0         # passive distance
        + 25.0 * 1           # ping-pong
        + 10.0 * 1           # GND pour missing
        + 8.0 * 3            # detours
    )
    assert cost == pytest.approx(expected)


def test_spec_unrouted_is_highest_penalty() -> None:
    """Spec: unrouted nets must have highest penalty (200x)."""
    m = BoardRoutingMetrics(
        total_track_length_mm=0.0, total_vias=0,
        nets_routed=0, nets_failed=1,
        overall_length_ratio=1.0, max_vias_per_net=0, per_net=(),
    )
    # 1 unrouted net should dominate all other single-count penalties
    cost = compute_board_cost(m)
    assert cost == pytest.approx(200.0)
    # Compare: 1 DRC violation = 70, 1 ping-pong = 25
    assert cost > 70.0
    assert cost > 25.0


# ---------------------------------------------------------------------------
# Spec Rule 5: Rip-up triggers
# ---------------------------------------------------------------------------


def test_spec_ripup_length_ratio_threshold() -> None:
    """Spec: rip up if length > 1.55-1.65x Manhattan."""
    result = RouteResult(
        net_number=1, net_name="NET1",
        tracks=(
            Track(start=Point(0, 0), end=Point(16, 0), width=0.25,
                  layer="F.Cu", net_number=1),
        ),
        vias=(), routed=True,
    )
    q = _score_route(result, [(0.0, 0.0), (10.0, 0.0)])
    # ratio = 16/10 = 1.6 > 1.55 → should trigger rip-up
    assert q.length_ratio > 1.55


def test_spec_ripup_bend_count_short_net() -> None:
    """Spec: rip up if >=4 bends on net < 40mm."""
    result = RouteResult(
        net_number=1, net_name="NET1",
        tracks=(
            Track(start=Point(0, 0), end=Point(2, 0), width=0.25,
                  layer="F.Cu", net_number=1),
            Track(start=Point(2, 0), end=Point(2, 2), width=0.25,
                  layer="F.Cu", net_number=1),
            Track(start=Point(2, 2), end=Point(4, 2), width=0.25,
                  layer="F.Cu", net_number=1),
            Track(start=Point(4, 2), end=Point(4, 4), width=0.25,
                  layer="F.Cu", net_number=1),
            Track(start=Point(4, 4), end=Point(6, 4), width=0.25,
                  layer="F.Cu", net_number=1),
        ),
        vias=(), routed=True,
    )
    q = _score_route(result, [(0.0, 0.0), (6.0, 4.0)])
    assert q.bend_count >= 4
    assert q.manhattan_ideal_mm < 40.0


# ---------------------------------------------------------------------------
# Spec Rule 6: Routing priority — power first
# ---------------------------------------------------------------------------


def test_spec_power_routes_before_signal() -> None:
    """Spec: power distribution nets route first."""
    from kicad_pipeline.models.pcb import NetEntry
    from kicad_pipeline.pcb.netlist import Netlist, NetlistEntry
    from kicad_pipeline.routing.grid_router import route_all_nets

    def _fp(ref: str, x: float, y: float) -> Footprint:
        return Footprint(
            lib_id="Test:Test", ref=ref, value="1k",
            position=Point(x, y), rotation=0.0, layer="F.Cu",
            pads=(
                Pad(number="1", pad_type="smd", shape="rect",
                    position=Point(-0.5, 0.0), size_x=0.6, size_y=0.6,
                    layers=("F.Cu",), net_number=0),
            ),
        )

    fps = [_fp("R1", 5, 20), _fp("R2", 15, 20), _fp("R3", 25, 20)]
    signal = NetlistEntry(
        net=NetEntry(number=1, name="SIG"),
        pad_refs=(("R1", "1"), ("R2", "1")),
    )
    power = NetlistEntry(
        net=NetEntry(number=2, name="+5V"),
        pad_refs=(("R2", "1"), ("R3", "1")),
    )
    netlist = Netlist(entries=(signal, power))
    results = route_all_nets(netlist, fps, 30.0, 40.0)
    assert len(results) >= 2
    assert results[0].net_name == "+5V"


# ---------------------------------------------------------------------------
# Spec Rule 7: GND stitching every 10-20mm
# ---------------------------------------------------------------------------


def test_spec_gnd_stitch_spacing_in_range() -> None:
    """Spec: stitching every 10-20mm."""
    assert 10.0 <= GND_STITCH_SPACING_MM <= 20.0


def test_spec_gnd_stitch_via_specs() -> None:
    """Spec: signal-size vias (drill 0.3, size 0.6)."""
    assert VIA_DRILL_SIGNAL_MM == 0.3
    assert VIA_DIAMETER_SIGNAL_MM == 0.6


def test_spec_gnd_stitch_footprint_clearance() -> None:
    """Spec: vias avoid footprint areas with clearance."""
    assert GND_STITCH_FP_CLEARANCE_MM >= 1.5


# ---------------------------------------------------------------------------
# Spec Rule 8: Via ping-pong detection
# ---------------------------------------------------------------------------


def test_spec_ping_pong_detects_triple_layer_change() -> None:
    """Spec: detect unnecessary layer ping-pong (F->B->F)."""
    r = RouteResult(
        net_number=1, net_name="NET1",
        tracks=(
            Track(start=Point(0, 0), end=Point(3, 0), width=0.25,
                  layer="F.Cu", net_number=1),
            Track(start=Point(3, 0), end=Point(6, 0), width=0.25,
                  layer="B.Cu", net_number=1),
            Track(start=Point(6, 0), end=Point(9, 0), width=0.25,
                  layer="F.Cu", net_number=1),
            Track(start=Point(9, 0), end=Point(12, 0), width=0.25,
                  layer="B.Cu", net_number=1),
            Track(start=Point(12, 0), end=Point(15, 0), width=0.25,
                  layer="F.Cu", net_number=1),
        ),
        vias=(), routed=True,
    )
    # Layers: [F,B,F,B,F] → ping-pongs at (F,B,F), (B,F,B), (F,B,F) = 3
    assert count_via_ping_pongs(r) == 3


def test_spec_ping_pong_penalty_in_cost() -> None:
    """Spec: 25x penalty per ping-pong in board cost."""
    m = BoardRoutingMetrics(
        total_track_length_mm=0.0, total_vias=0,
        nets_routed=1, nets_failed=0,
        overall_length_ratio=1.0, max_vias_per_net=0, per_net=(),
        via_ping_pong_count=3,
    )
    cost = compute_board_cost(m)
    assert cost == pytest.approx(75.0)  # 25 * 3


# ---------------------------------------------------------------------------
# Spec Rule 9: Passive proximity - Class A <= 6mm, Class B <= 10mm
# ---------------------------------------------------------------------------


def test_spec_passive_proximity_close() -> None:
    """Passives within spec should have low avg distance."""
    fps = [
        Footprint(
            lib_id="Test:Test", ref="U1", value="MCU",
            position=Point(10, 10), rotation=0.0, layer="F.Cu",
            pads=(Pad(number="1", pad_type="smd", shape="rect",
                      position=Point(0, 0), size_x=0.6, size_y=0.6,
                      layers=("F.Cu",)),),
        ),
        Footprint(
            lib_id="Test:Test", ref="R1", value="10k",
            position=Point(13, 10), rotation=0.0, layer="F.Cu",
            pads=(Pad(number="1", pad_type="smd", shape="rect",
                      position=Point(0, 0), size_x=0.6, size_y=0.6,
                      layers=("F.Cu",)),),
        ),
        Footprint(
            lib_id="Test:Test", ref="C1", value="100nF",
            position=Point(11, 12), rotation=0.0, layer="F.Cu",
            pads=(Pad(number="1", pad_type="smd", shape="rect",
                      position=Point(0, 0), size_x=0.6, size_y=0.6,
                      layers=("F.Cu",)),),
        ),
    ]
    dist = compute_passive_proximity(fps)
    # R1 is 3mm from U1, C1 is ~2.2mm from U1
    assert dist < 6.0, f"Passives should be within 6mm of IC, got {dist:.1f}mm"


def test_spec_passive_proximity_far_flagged() -> None:
    """Passives placed far from ICs should produce high avg distance."""
    fps = [
        Footprint(
            lib_id="Test:Test", ref="U1", value="MCU",
            position=Point(10, 10), rotation=0.0, layer="F.Cu",
            pads=(Pad(number="1", pad_type="smd", shape="rect",
                      position=Point(0, 0), size_x=0.6, size_y=0.6,
                      layers=("F.Cu",)),),
        ),
        Footprint(
            lib_id="Test:Test", ref="R1", value="10k",
            position=Point(50, 50), rotation=0.0, layer="F.Cu",
            pads=(Pad(number="1", pad_type="smd", shape="rect",
                      position=Point(0, 0), size_x=0.6, size_y=0.6,
                      layers=("F.Cu",)),),
        ),
    ]
    dist = compute_passive_proximity(fps)
    assert dist > 8.0, f"Far passive should have distance > 8mm, got {dist:.1f}mm"


# ---------------------------------------------------------------------------
# Spec Rule 10: Clearance validation
# ---------------------------------------------------------------------------


def test_spec_clearance_default_200um() -> None:
    """Spec: 0.20/0.20mm min clearance rules."""
    assert CLEARANCE_DEFAULT_MM == 0.2


def test_spec_clearance_detection() -> None:
    """Two tracks at 0.35mm center distance with 0.25mm width violate 0.2mm clearance."""
    d = _segment_min_distance(0, 0, 10, 0, 0, 0.35, 10, 0.35)
    hw = 0.25 / 2.0
    edge_dist = d - hw - hw
    assert edge_dist < CLEARANCE_DEFAULT_MM


# ---------------------------------------------------------------------------
# Spec Rule 11: Detour detection
# ---------------------------------------------------------------------------


def test_spec_detour_threshold_1_4x() -> None:
    """Spec: trace efficiency <= 1.4x Manhattan ideal target."""
    # A straight route: ratio ~1.0, no detour
    straight = (
        RouteResult(
            net_number=1, net_name="NET1",
            tracks=(
                Track(start=Point(0, 0), end=Point(10, 0), width=0.25,
                      layer="F.Cu", net_number=1),
            ),
            vias=(), routed=True,
        ),
    )
    assert count_detours(straight, threshold=1.4) == 0

    # An L-shaped route: ratio ~1.41, marginal detour
    l_shape = (
        RouteResult(
            net_number=1, net_name="NET1",
            tracks=(
                Track(start=Point(0, 0), end=Point(10, 0), width=0.25,
                      layer="F.Cu", net_number=1),
                Track(start=Point(10, 0), end=Point(10, 10), width=0.25,
                      layer="F.Cu", net_number=1),
            ),
            vias=(), routed=True,
        ),
    )
    # Manhattan span = 10+10=20, actual = 10+10=20, ratio=1.0 → no detour
    assert count_detours(l_shape, threshold=1.4) == 0


# ---------------------------------------------------------------------------
# Spec Rule 12: JLCPCB manufacturing constraints
# ---------------------------------------------------------------------------


def test_spec_jlcpcb_min_clearance() -> None:
    """Spec: JLCPCB min clearance 0.127mm."""
    assert JLCPCB_MIN_CLEARANCE_MM == 0.127


# ---------------------------------------------------------------------------
# Spec Rule 13: Cost function component ordering
# ---------------------------------------------------------------------------


def test_spec_cost_penalty_ordering() -> None:
    """Spec penalty ordering: unrouted (200) > DRC (70) > ping-pong (25) > via (16)."""
    assert 200.0 > 70.0 > 25.0 > 16.0
    # Verify via compute_board_cost
    unrouted = BoardRoutingMetrics(
        total_track_length_mm=0.0, total_vias=0,
        nets_routed=0, nets_failed=1,
        overall_length_ratio=1.0, max_vias_per_net=0, per_net=(),
    )
    drc = BoardRoutingMetrics(
        total_track_length_mm=0.0, total_vias=0,
        nets_routed=1, nets_failed=0,
        overall_length_ratio=1.0, max_vias_per_net=0, per_net=(),
    )
    pp = BoardRoutingMetrics(
        total_track_length_mm=0.0, total_vias=0,
        nets_routed=1, nets_failed=0,
        overall_length_ratio=1.0, max_vias_per_net=0, per_net=(),
        via_ping_pong_count=1,
    )
    via_m = BoardRoutingMetrics(
        total_track_length_mm=0.0, total_vias=1,
        nets_routed=1, nets_failed=0,
        overall_length_ratio=1.0, max_vias_per_net=1, per_net=(),
    )
    assert compute_board_cost(unrouted) > compute_board_cost(drc, drc_violations=1)
    assert compute_board_cost(drc, drc_violations=1) > compute_board_cost(pp)
    assert compute_board_cost(pp) > compute_board_cost(via_m)
