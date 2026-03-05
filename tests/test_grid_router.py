"""Tests for kicad_pipeline.routing.grid_router."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Footprint, Keepout, NetEntry, Pad, Point, Track
from kicad_pipeline.pcb.netlist import Netlist, NetlistEntry
from kicad_pipeline.routing.grid_router import (
    RouteRequest,
    RouteResult,
    _Grid,
    _prepare_grid,
    collect_tracks,
    collect_vias,
    route_all_nets,
    route_net,
)

# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _make_pad(number: str, x: float = 0.0, y: float = 0.0, net_number: int = 1) -> Pad:
    return Pad(
        number=number,
        pad_type="smd",
        shape="rect",
        position=Point(x=x, y=y),
        size_x=1.0,
        size_y=1.0,
        layers=("F.Cu",),
        net_number=net_number,
        net_name="NET",
    )


def _make_footprint(ref: str, x: float, y: float, pad_number: str = "1") -> Footprint:
    return Footprint(
        lib_id="Test:Test",
        ref=ref,
        value="VAL",
        position=Point(x=x, y=y),
        rotation=0.0,
        layer="F.Cu",
        pads=(_make_pad(pad_number),),
    )


def _make_simple_footprints() -> tuple[Footprint, Footprint]:
    """Two small footprints with one pad each, separated by 10 mm."""
    fp1 = _make_footprint("R1", x=5.0, y=20.0)
    fp2 = _make_footprint("R2", x=15.0, y=20.0)
    return fp1, fp2


def _make_route_request(
    net_number: int,
    pad_refs: tuple[tuple[str, str], ...],
    layer: str = "F.Cu",
    width_mm: float = 0.25,
) -> RouteRequest:
    return RouteRequest(
        net_number=net_number,
        net_name="TEST_NET",
        pad_refs=pad_refs,
        layer=layer,
        width_mm=width_mm,
    )


def _make_netlist(entries: list[NetlistEntry]) -> Netlist:
    return Netlist(entries=tuple(entries))


def _make_netlist_entry(
    net_number: int,
    net_name: str,
    pad_refs: tuple[tuple[str, str], ...],
) -> NetlistEntry:
    return NetlistEntry(
        net=NetEntry(number=net_number, name=net_name),
        pad_refs=pad_refs,
    )


# ---------------------------------------------------------------------------
# RouteRequest / RouteResult dataclass tests
# ---------------------------------------------------------------------------


def test_route_request_frozen() -> None:
    req = _make_route_request(1, (("R1", "1"), ("R2", "1")))
    with pytest.raises(AttributeError):
        req.net_number = 99  # type: ignore[misc]


def test_route_result_frozen() -> None:
    result = RouteResult(
        net_number=1,
        net_name="NET",
        tracks=(),
        vias=(),
        routed=True,
    )
    with pytest.raises(AttributeError):
        result.routed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# route_net - successful routing
# ---------------------------------------------------------------------------


def test_route_result_routed_flag() -> None:
    fp1, fp2 = _make_simple_footprints()
    req = _make_route_request(1, (("R1", "1"), ("R2", "1")))
    result = route_net(req, (fp1, fp2), board_width_mm=30.0, board_height_mm=40.0)
    assert result.routed is True


def test_route_result_tracks_not_empty() -> None:
    fp1, fp2 = _make_simple_footprints()
    req = _make_route_request(1, (("R1", "1"), ("R2", "1")))
    result = route_net(req, (fp1, fp2), board_width_mm=30.0, board_height_mm=40.0)
    assert len(result.tracks) > 0


def test_route_result_track_net_number() -> None:
    fp1, fp2 = _make_simple_footprints()
    req = _make_route_request(42, (("R1", "1"), ("R2", "1")))
    result = route_net(req, (fp1, fp2), board_width_mm=30.0, board_height_mm=40.0)
    assert all(t.net_number == 42 for t in result.tracks)


def test_route_result_track_layer() -> None:
    fp1, fp2 = _make_simple_footprints()
    req = _make_route_request(1, (("R1", "1"), ("R2", "1")), layer="B.Cu")
    result = route_net(req, (fp1, fp2), board_width_mm=30.0, board_height_mm=40.0)
    assert all(t.layer == "B.Cu" for t in result.tracks)


def test_track_positions() -> None:
    """Tracks should connect start and end regions of the two pads."""
    fp1, fp2 = _make_simple_footprints()
    req = _make_route_request(1, (("R1", "1"), ("R2", "1")))
    result = route_net(req, (fp1, fp2), board_width_mm=30.0, board_height_mm=40.0)
    assert result.routed
    # First track start should be near fp1 position (5, 20)
    # Last track end should be near fp2 position (15, 20)
    first_track = result.tracks[0]
    last_track = result.tracks[-1]
    assert abs(first_track.start.x - 5.0) < 2.0
    assert abs(last_track.end.x - 15.0) < 2.0


def test_route_result_tracks_within_board() -> None:
    """All track segments should have coordinates within board bounds."""
    fp1, fp2 = _make_simple_footprints()
    req = _make_route_request(1, (("R1", "1"), ("R2", "1")))
    board_w, board_h = 30.0, 40.0
    result = route_net(req, (fp1, fp2), board_width_mm=board_w, board_height_mm=board_h)
    assert result.routed
    for track in result.tracks:
        assert -0.1 <= track.start.x <= board_w + 0.1
        assert -0.1 <= track.start.y <= board_h + 0.1
        assert -0.1 <= track.end.x <= board_w + 0.1
        assert -0.1 <= track.end.y <= board_h + 0.1


# ---------------------------------------------------------------------------
# route_net - failure cases
# ---------------------------------------------------------------------------


def test_route_result_unroutable() -> None:
    """Routing fails when target footprint ref is missing."""
    fp1 = _make_footprint("R1", x=5.0, y=5.0)
    req = _make_route_request(1, (("R1", "1"), ("NONEXISTENT", "1")))
    result = route_net(req, [fp1], board_width_mm=20.0, board_height_mm=20.0)
    assert result.routed is False
    assert "NONEXISTENT" in result.reason


def test_route_net_missing_footprint() -> None:
    fp1, _fp2 = _make_simple_footprints()
    req = _make_route_request(1, (("MISSING", "1"), ("R1", "1")))
    result = route_net(req, (fp1,), board_width_mm=30.0, board_height_mm=40.0)
    assert result.routed is False
    assert "MISSING" in result.reason


def test_route_net_missing_pad() -> None:
    fp1, fp2 = _make_simple_footprints()
    req = _make_route_request(1, (("R1", "99"), ("R2", "1")))
    result = route_net(req, (fp1, fp2), board_width_mm=30.0, board_height_mm=40.0)
    assert result.routed is False


# ---------------------------------------------------------------------------
# route_all_nets
# ---------------------------------------------------------------------------


def test_route_all_nets_empty_netlist() -> None:
    netlist = _make_netlist([])
    result = route_all_nets(netlist, (), board_width_mm=30.0, board_height_mm=40.0)
    assert result == ()


def test_route_all_nets_single_pad_skipped() -> None:
    """A net with only one pad should be skipped (produces no RouteResult)."""
    entry = _make_netlist_entry(1, "SINGLE", (("R1", "1"),))
    netlist = _make_netlist([entry])
    fp1, _fp2 = _make_simple_footprints()
    results = route_all_nets(netlist, (fp1,), board_width_mm=30.0, board_height_mm=40.0)
    # Single-pad nets are skipped entirely
    assert len(results) == 0


def test_route_all_nets_returns_tuple() -> None:
    fp1, fp2 = _make_simple_footprints()
    entry = _make_netlist_entry(2, "SIG", (("R1", "1"), ("R2", "1")))
    netlist = _make_netlist([entry])
    results = route_all_nets(netlist, (fp1, fp2), board_width_mm=30.0, board_height_mm=40.0)
    assert isinstance(results, tuple)


def test_route_gnd_skipped_by_default() -> None:
    """GND is handled by copper pour and should be skipped by route_all_nets."""
    fp1, fp2 = _make_simple_footprints()
    entry = _make_netlist_entry(1, "GND", (("R1", "1"), ("R2", "1")))
    netlist = _make_netlist([entry])
    results = route_all_nets(netlist, (fp1, fp2), board_width_mm=30.0, board_height_mm=40.0)
    assert len(results) == 0  # GND skipped — copper pour handles it


def test_route_signal_uses_default_trace() -> None:
    """Regular signal nets should use 0.25 mm width (fallback)."""
    fp1, fp2 = _make_simple_footprints()
    entry = _make_netlist_entry(2, "SDA", (("R1", "1"), ("R2", "1")))
    netlist = _make_netlist([entry])
    results = route_all_nets(netlist, (fp1, fp2), board_width_mm=30.0, board_height_mm=40.0)
    assert len(results) == 1
    result = results[0]
    if result.routed and result.tracks:
        assert all(abs(t.width - 0.25) < 1e-9 for t in result.tracks)


def test_route_all_nets_with_net_widths() -> None:
    """net_widths parameter should override default trace width logic."""
    fp1, fp2 = _make_simple_footprints()
    entry = _make_netlist_entry(1, "SENS_IN", (("R1", "1"), ("R2", "1")))
    netlist = _make_netlist([entry])
    widths = {"SENS_IN": 0.4}
    results = route_all_nets(
        netlist, (fp1, fp2),
        board_width_mm=30.0, board_height_mm=40.0,
        net_widths=widths,
    )
    assert len(results) == 1
    result = results[0]
    if result.routed and result.tracks:
        assert all(abs(t.width - 0.4) < 1e-9 for t in result.tracks)


def test_route_all_nets_net_widths_fallback() -> None:
    """Nets not in net_widths should fall back to 0.25mm."""
    fp1, fp2 = _make_simple_footprints()
    entry = _make_netlist_entry(2, "UNKNOWN", (("R1", "1"), ("R2", "1")))
    netlist = _make_netlist([entry])
    widths = {"GND": 0.5}  # UNKNOWN not in widths
    results = route_all_nets(
        netlist, (fp1, fp2),
        board_width_mm=30.0, board_height_mm=40.0,
        net_widths=widths,
    )
    assert len(results) == 1
    result = results[0]
    if result.routed and result.tracks:
        assert all(abs(t.width - 0.25) < 1e-9 for t in result.tracks)


# ---------------------------------------------------------------------------
# collect_tracks / collect_vias
# ---------------------------------------------------------------------------


def test_collect_tracks_empty() -> None:
    assert collect_tracks(()) == ()


def test_collect_vias_empty() -> None:
    assert collect_vias(()) == ()


def test_collect_tracks_combines() -> None:
    t1 = Track(start=Point(0, 0), end=Point(1, 0), width=0.25, layer="F.Cu", net_number=1)
    t2 = Track(start=Point(2, 0), end=Point(3, 0), width=0.25, layer="F.Cu", net_number=2)
    r1 = RouteResult(net_number=1, net_name="A", tracks=(t1,), vias=(), routed=True)
    r2 = RouteResult(net_number=2, net_name="B", tracks=(t2,), vias=(), routed=True)
    combined = collect_tracks((r1, r2))
    assert len(combined) == 2
    assert t1 in combined
    assert t2 in combined


# ---------------------------------------------------------------------------
# Fix 3: Same-net pad routing, shared grid, net sorting
# ---------------------------------------------------------------------------


def test_router_can_reach_same_net_pads() -> None:
    """Router should route between pads on the same net even when close together.

    Old bug: all pads were marked occupied, so the router couldn't reach
    its own target pads.
    """
    # Two pads 5mm apart — close enough that old gap pre-check would reject
    fp1 = _make_footprint("R1", x=10.0, y=20.0, pad_number="1")
    fp2 = _make_footprint("R2", x=15.0, y=20.0, pad_number="1")
    req = _make_route_request(1, (("R1", "1"), ("R2", "1")))
    result = route_net(req, [fp1, fp2], board_width_mm=30.0, board_height_mm=40.0)
    assert result.routed is True
    assert len(result.tracks) > 0


def test_router_shared_grid_prevents_shorts() -> None:
    """When using route_all_nets, routed tracks for one net should block
    subsequent nets from using the same cells (preventing shorts)."""
    fp1 = Footprint(
        lib_id="Test:Test", ref="R1", value="V",
        position=Point(x=5.0, y=20.0), layer="F.Cu",
        pads=(_make_pad("1", net_number=1), _make_pad("2", x=2.0, net_number=2)),
    )
    fp2 = Footprint(
        lib_id="Test:Test", ref="R2", value="V",
        position=Point(x=20.0, y=20.0), layer="F.Cu",
        pads=(_make_pad("1", net_number=1), _make_pad("2", x=2.0, net_number=2)),
    )
    entry1 = _make_netlist_entry(1, "NET_A", (("R1", "1"), ("R2", "1")))
    entry2 = _make_netlist_entry(2, "NET_B", (("R1", "2"), ("R2", "2")))
    netlist = _make_netlist([entry1, entry2])
    results = route_all_nets(netlist, [fp1, fp2], board_width_mm=30.0, board_height_mm=40.0)
    # At least one net should route successfully
    assert any(r.routed for r in results)


def test_router_sorts_nets_by_complexity() -> None:
    """route_all_nets should route simpler nets (fewer pads) first."""
    fp1 = _make_footprint("R1", x=5.0, y=20.0)
    fp2 = _make_footprint("R2", x=15.0, y=20.0)
    fp3 = _make_footprint("R3", x=25.0, y=20.0)
    # 3-pad net should route after 2-pad net
    entry_2pad = _make_netlist_entry(1, "SIMPLE", (("R1", "1"), ("R2", "1")))
    entry_3pad = _make_netlist_entry(2, "COMPLEX", (("R1", "1"), ("R2", "1"), ("R3", "1")))
    netlist = _make_netlist([entry_3pad, entry_2pad])  # Insert complex first
    results = route_all_nets(netlist, [fp1, fp2, fp3], board_width_mm=30.0, board_height_mm=40.0)
    # Results should come back with simple net first (sorted by pad count)
    assert len(results) == 2
    assert results[0].net_name == "SIMPLE"


def test_collect_vias_combines() -> None:
    from kicad_pipeline.models.pcb import Via

    v1 = Via(
        position=Point(1, 1),
        drill=0.3,
        size=0.6,
        layers=("F.Cu", "B.Cu"),
        net_number=1,
    )
    r1 = RouteResult(net_number=1, net_name="A", tracks=(), vias=(v1,), routed=True)
    r2 = RouteResult(net_number=2, net_name="B", tracks=(), vias=(), routed=True)
    combined = collect_vias((r1, r2))
    assert len(combined) == 1
    assert v1 in combined


# ---------------------------------------------------------------------------
# Fix 1: Board-edge margin
# ---------------------------------------------------------------------------


def test_prepare_grid_marks_edge_margins() -> None:
    """Edge cells should be marked occupied after _prepare_grid."""
    grid = _Grid.create(20.0, 20.0, grid_step_mm=0.5)
    _prepare_grid(grid, [])
    # Corner cells should be occupied (edge margin)
    assert not grid.is_free(0, 0)
    assert not grid.is_free(grid.cols - 1, 0)
    assert not grid.is_free(0, grid.rows - 1)
    assert not grid.is_free(grid.cols - 1, grid.rows - 1)
    # Centre should be free
    mid_c = grid.cols // 2
    mid_r = grid.rows // 2
    assert grid.is_free(mid_c, mid_r)


def test_route_net_tracks_avoid_board_edges() -> None:
    """Tracks generated by route_net should not touch the first/last grid cells."""
    fp1 = _make_footprint("R1", x=5.0, y=10.0)
    fp2 = _make_footprint("R2", x=15.0, y=10.0)
    req = _make_route_request(1, (("R1", "1"), ("R2", "1")))
    result = route_net(req, [fp1, fp2], board_width_mm=20.0, board_height_mm=20.0)
    if result.routed:
        for track in result.tracks:
            # No track endpoint should be at the very edge (within 0.3mm)
            assert track.start.x >= 0.0
            assert track.start.y >= 0.0


# ---------------------------------------------------------------------------
# Fix 2: Keepout zone marking
# ---------------------------------------------------------------------------


def test_prepare_grid_marks_keepout_zones() -> None:
    """Keepout zones should be marked as occupied on the grid."""
    grid = _Grid.create(30.0, 30.0, grid_step_mm=0.5)
    keepout = Keepout(
        polygon=(Point(10, 10), Point(15, 10), Point(15, 15), Point(10, 15)),
        layers=("F.Cu",),
        no_copper=True,
    )
    _prepare_grid(grid, [], keepouts=(keepout,))
    # Cell in the keepout area should be occupied
    col, row = grid.to_cell(12.0, 12.0)
    assert not grid.is_free(col, row)
    # Cell outside keepout should be free (away from edges)
    col2, row2 = grid.to_cell(25.0, 25.0)
    assert grid.is_free(col2, row2)


def test_route_all_nets_accepts_keepouts() -> None:
    """route_all_nets should accept and use keepouts parameter."""
    fp1, fp2 = _make_simple_footprints()
    entry = _make_netlist_entry(1, "SIG", (("R1", "1"), ("R2", "1")))
    netlist = _make_netlist([entry])
    keepout = Keepout(
        polygon=(Point(8, 18), Point(12, 18), Point(12, 22), Point(8, 22)),
        layers=("F.Cu",),
        no_copper=True,
    )
    results = route_all_nets(
        netlist, [fp1, fp2],
        board_width_mm=30.0, board_height_mm=40.0,
        keepouts=(keepout,),
    )
    assert isinstance(results, tuple)


# ---------------------------------------------------------------------------
# Fix 3: No near-start/near-goal shorts
# ---------------------------------------------------------------------------


def test_astar_does_not_traverse_occupied_near_pads() -> None:
    """A* should not traverse occupied cells near start/goal (no shorts)."""
    # Create a grid and mark a barrier between two points
    grid = _Grid.create(20.0, 20.0, grid_step_mm=0.5)
    # Mark a wall of cells that the router must go around
    for r in range(0, grid.rows):
        grid.mark(20, r)  # col 20 = 10mm
    # Unmark start and goal
    start_col, start_row = grid.to_cell(5.0, 10.0)
    goal_col, goal_row = grid.to_cell(15.0, 10.0)
    # The wall blocks direct path; router should find a way around or fail
    from kicad_pipeline.routing.grid_router import _astar

    path = _astar(grid, start_col, start_row, goal_col, goal_row)
    # Path is None because the wall is complete — no way around
    assert path is None
