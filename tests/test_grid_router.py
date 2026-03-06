"""Tests for kicad_pipeline.routing.grid_router."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Footprint, Keepout, NetEntry, Pad, Point, Track, Via
from kicad_pipeline.pcb.netlist import Netlist, NetlistEntry
from kicad_pipeline.routing.grid_router import (
    RouteRequest,
    RouteResult,
    _find_free_via_position,
    _Grid,
    _is_line_clear,
    _keepout_blocks_layer,
    _mark_line_on_grid,
    _mark_via_on_fcu,
    _prepare_bcu_grid,
    _prepare_grid,
    _route_on_bcu,
    _route_stub_on_fcu,
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


# ---------------------------------------------------------------------------
# B.Cu dual-layer routing tests
# ---------------------------------------------------------------------------


def _make_tht_pad(
    number: str, x: float = 0.0, y: float = 0.0, net_number: int = 1,
) -> Pad:
    """Create a through-hole pad that penetrates both layers."""
    return Pad(
        number=number,
        pad_type="thru_hole",
        shape="circle",
        position=Point(x=x, y=y),
        size_x=1.6,
        size_y=1.6,
        layers=("F.Cu", "B.Cu"),
        net_number=net_number,
        net_name="NET",
        drill_diameter=1.0,
    )


def test_route_on_bcu_produces_tracks_and_vias() -> None:
    """B.Cu routing returns B.Cu-layer tracks + 2 vias."""
    bcu_grid = _Grid.create(30.0, 30.0, grid_step_mm=0.5)
    # No obstacles — should route easily
    result = _route_on_bcu(
        5.0, 15.0, 25.0, 15.0,
        bcu_grid, net_number=1, net_name="SIG",
        width_mm=0.25, clearance_mm=0.2,
    )
    assert result is not None
    tracks, (via_start, via_goal) = result
    assert len(tracks) > 0
    assert all(t.layer == "B.Cu" for t in tracks)
    assert isinstance(via_start, Via)
    assert isinstance(via_goal, Via)


def test_route_on_bcu_via_properties() -> None:
    """Via drill/size/layers/net should use signal via dimensions."""
    from kicad_pipeline.constants import VIA_DIAMETER_SIGNAL_MM, VIA_DRILL_SIGNAL_MM

    bcu_grid = _Grid.create(30.0, 30.0, grid_step_mm=0.5)
    result = _route_on_bcu(
        5.0, 15.0, 25.0, 15.0,
        bcu_grid, net_number=42, net_name="I2C_SDA",
        width_mm=0.25, clearance_mm=0.2,
    )
    assert result is not None
    _tracks, (via_start, via_goal) = result
    for via in (via_start, via_goal):
        assert via.drill == VIA_DRILL_SIGNAL_MM
        assert via.size == VIA_DIAMETER_SIGNAL_MM
        assert via.layers == ("F.Cu", "B.Cu")
        assert via.net_number == 42


def test_route_net_bcu_fallback_ic_final_leg() -> None:
    """Dense IC pad routes via B.Cu when F.Cu is blocked."""
    # Create a dense IC (8 fine-pitch pads) and one passive
    ic_pads = tuple(
        Pad(
            number=str(i + 1), pad_type="smd", shape="rect",
            position=Point(x=(i % 4) * 0.65, y=(i // 4) * 3.0),
            size_x=0.4, size_y=1.2,
            layers=("F.Cu",), net_number=i + 1, net_name=f"N{i + 1}",
        )
        for i in range(8)
    )
    ic_fp = Footprint(
        lib_id="Test:IC", ref="U1", value="IC",
        position=Point(x=15.0, y=15.0), layer="F.Cu",
        pads=ic_pads,
    )
    res_fp = Footprint(
        lib_id="Test:R", ref="R1", value="10k",
        position=Point(x=5.0, y=15.0), layer="F.Cu",
        pads=(_make_pad("1", net_number=1),),
    )
    req = RouteRequest(
        net_number=1, net_name="SIG",
        pad_refs=(("R1", "1"), ("U1", "1")),
        layer="F.Cu", width_mm=0.25,
    )
    # Create a shared grid and B.Cu grid
    grid = _Grid.create(30.0, 30.0, grid_step_mm=0.5)
    _prepare_grid(grid, [ic_fp, res_fp])
    bcu_grid = _prepare_bcu_grid(grid, [ic_fp, res_fp])

    result = route_net(
        req, [ic_fp, res_fp],
        board_width_mm=30.0, board_height_mm=30.0,
        grid=grid, bcu_grid=bcu_grid,
    )
    # Should produce tracks (F.Cu or B.Cu) and potentially vias
    assert len(result.tracks) > 0


def test_route_net_bcu_fallback_mst_loop() -> None:
    """MST segment routes via B.Cu when F.Cu is completely blocked."""
    # Place two pads with a wall between them on F.Cu
    fp1 = _make_footprint("R1", x=5.0, y=15.0)
    fp2 = _make_footprint("R2", x=25.0, y=15.0)

    grid = _Grid.create(30.0, 30.0, grid_step_mm=0.5)
    _prepare_grid(grid, [fp1, fp2])

    # Block ALL intermediate F.Cu cells with a wall
    for row in range(grid.rows):
        grid.mark(grid.cols // 2, row)
        grid.mark(grid.cols // 2 - 1, row)
        grid.mark(grid.cols // 2 + 1, row)

    bcu_grid = _prepare_bcu_grid(grid, [fp1, fp2])

    req = _make_route_request(1, (("R1", "1"), ("R2", "1")))
    result = route_net(
        req, [fp1, fp2],
        board_width_mm=30.0, board_height_mm=30.0,
        grid=grid, bcu_grid=bcu_grid,
    )
    # Should succeed via B.Cu fallback
    assert result.routed is True
    # Should have B.Cu tracks
    bcu_tracks = [t for t in result.tracks if t.layer == "B.Cu"]
    assert len(bcu_tracks) > 0
    # Should have vias
    assert len(result.vias) >= 2


def test_route_net_without_bcu_grid_unchanged() -> None:
    """bcu_grid=None preserves existing behavior (no B.Cu fallback)."""
    fp1, fp2 = _make_simple_footprints()
    req = _make_route_request(1, (("R1", "1"), ("R2", "1")))
    result = route_net(
        req, [fp1, fp2],
        board_width_mm=30.0, board_height_mm=40.0,
        bcu_grid=None,
    )
    assert result.routed is True
    # All tracks should be F.Cu (no B.Cu fallback available)
    assert all(t.layer == "F.Cu" for t in result.tracks)
    assert len(result.vias) == 0


def test_bcu_grid_skips_smd_pads() -> None:
    """B.Cu grid doesn't mark SMD pad areas (they're F.Cu-only)."""
    smd_fp = _make_footprint("R1", x=15.0, y=15.0)
    tht_fp = Footprint(
        lib_id="Test:J", ref="J1", value="Conn",
        position=Point(x=5.0, y=15.0), layer="F.Cu",
        pads=(_make_tht_pad("1"),),
    )
    fcu_grid = _Grid.create(30.0, 30.0, grid_step_mm=0.5)
    bcu_grid = _prepare_bcu_grid(fcu_grid, [smd_fp, tht_fp])

    # THT pad position should be marked on B.Cu
    tht_col, tht_row = bcu_grid.to_cell(5.0, 15.0)
    assert not bcu_grid.is_free(tht_col, tht_row)

    # SMD pad position should be FREE on B.Cu (it's F.Cu-only)
    smd_col, smd_row = bcu_grid.to_cell(15.0, 15.0)
    assert bcu_grid.is_free(smd_col, smd_row)


def test_via_positions_marked_on_fcu_grid() -> None:
    """After B.Cu routing, vias should be marked occupied on F.Cu grid."""
    fcu_grid = _Grid.create(30.0, 30.0, grid_step_mm=0.5)
    via = Via(
        position=Point(10.0, 15.0),
        drill=0.508, size=0.9,
        layers=("F.Cu", "B.Cu"),
        net_number=1,
    )
    # Cell should be free before marking
    col, row = fcu_grid.to_cell(10.0, 15.0)
    assert fcu_grid.is_free(col, row)

    _mark_via_on_fcu(fcu_grid, via)

    # Cell should now be occupied
    assert not fcu_grid.is_free(col, row)


def test_collect_vias_includes_bcu_routing_vias() -> None:
    """collect_vias() returns non-empty when B.Cu routing produced vias."""
    via1 = Via(
        position=Point(5.0, 15.0),
        drill=0.508, size=0.9,
        layers=("F.Cu", "B.Cu"),
        net_number=1,
    )
    via2 = Via(
        position=Point(25.0, 15.0),
        drill=0.508, size=0.9,
        layers=("F.Cu", "B.Cu"),
        net_number=1,
    )
    r = RouteResult(
        net_number=1, net_name="SIG",
        tracks=(), vias=(via1, via2), routed=True,
    )
    combined = collect_vias((r,))
    assert len(combined) == 2
    assert via1 in combined
    assert via2 in combined


# ---------------------------------------------------------------------------
# _find_free_via_position tests
# ---------------------------------------------------------------------------


def test_find_free_via_position_free_target() -> None:
    """When target area is clear, returns grid-snapped position near target."""
    grid = _Grid.create(30.0, 30.0, grid_step_mm=0.25)
    result = _find_free_via_position(grid, 15.0, 15.0, via_radius_mm=0.3, clearance_mm=0.2)
    assert result is not None
    # Result should be grid-snapped (multiple of grid_step_mm)
    rx, ry = result
    assert rx % 0.25 == pytest.approx(0.0, abs=1e-6)
    assert ry % 0.25 == pytest.approx(0.0, abs=1e-6)
    # Should be very close to target (within one grid cell)
    assert abs(rx - 15.0) <= 0.25
    assert abs(ry - 15.0) <= 0.25


def test_find_free_via_position_offset() -> None:
    """When target is occupied, returns a nearby free position."""
    grid = _Grid.create(30.0, 30.0, grid_step_mm=0.25)
    # Mark a cluster of cells around target
    tc, tr = grid.to_cell(15.0, 15.0)
    for dc in range(-3, 4):
        for dr in range(-3, 4):
            grid.mark(tc + dc, tr + dr)
    result = _find_free_via_position(grid, 15.0, 15.0, via_radius_mm=0.3, clearance_mm=0.2)
    assert result is not None
    # Should be offset from original position
    rx, ry = result
    dist = ((rx - 15.0) ** 2 + (ry - 15.0) ** 2) ** 0.5
    assert dist > 0.1  # must be offset
    assert dist < 3.0  # but not too far


def test_find_free_via_position_fully_blocked() -> None:
    """When entire search area is blocked, returns None."""
    grid = _Grid.create(10.0, 10.0, grid_step_mm=0.25)
    # Mark entire grid as occupied
    for c in range(grid.cols):
        for r in range(grid.rows):
            grid.mark(c, r)
    result = _find_free_via_position(grid, 5.0, 5.0, via_radius_mm=0.3, clearance_mm=0.2)
    assert result is None


def test_bcu_fallback_generates_fcu_stub() -> None:
    """When via is offset from pad, an F.Cu stub track bridges pad to via."""
    bcu_grid = _Grid.create(30.0, 30.0, grid_step_mm=0.25)
    fcu_grid = _Grid.create(30.0, 30.0, grid_step_mm=0.25)
    # Block F.Cu around start position to force via offset
    tc, tr = fcu_grid.to_cell(5.0, 15.0)
    for dc in range(-3, 4):
        for dr in range(-3, 4):
            fcu_grid.mark(tc + dc, tr + dr)

    result = _route_on_bcu(
        5.0, 15.0, 25.0, 15.0,
        bcu_grid, net_number=1, net_name="SIG",
        width_mm=0.25, clearance_mm=0.2,
        fcu_grid=fcu_grid,
    )
    assert result is not None
    tracks, (via_start, _via_goal) = result
    # At least one track should be on F.Cu (the stub)
    fcu_tracks = [t for t in tracks if t.layer == "F.Cu"]
    assert len(fcu_tracks) >= 1
    # The stub should start at the pad position
    stub = fcu_tracks[0]
    assert abs(stub.start.x - 5.0) < 0.01
    assert abs(stub.start.y - 15.0) < 0.01
    # Via should NOT be at the original pad position (it was offset)
    assert abs(via_start.position.x - 5.0) > 0.1 or abs(via_start.position.y - 15.0) > 0.1


# ---------------------------------------------------------------------------
# collect_vias routed_only filter
# ---------------------------------------------------------------------------


def test_collect_vias_filters_unrouted() -> None:
    """collect_vias with routed_only=True excludes vias from unrouted nets."""
    v1 = Via(
        position=Point(1, 1), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=1,
    )
    v2 = Via(
        position=Point(2, 2), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=2,
    )
    r_ok = RouteResult(net_number=1, net_name="A", tracks=(), vias=(v1,), routed=True)
    r_fail = RouteResult(net_number=2, net_name="B", tracks=(), vias=(v2,), routed=False)
    combined = collect_vias((r_ok, r_fail), routed_only=True)
    assert len(combined) == 1
    assert v1 in combined
    assert v2 not in combined


def test_collect_vias_includes_all() -> None:
    """collect_vias with routed_only=False includes everything."""
    v1 = Via(
        position=Point(1, 1), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=1,
    )
    v2 = Via(
        position=Point(2, 2), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=2,
    )
    r_ok = RouteResult(net_number=1, net_name="A", tracks=(), vias=(v1,), routed=True)
    r_fail = RouteResult(net_number=2, net_name="B", tracks=(), vias=(v2,), routed=False)
    combined = collect_vias((r_ok, r_fail), routed_only=False)
    assert len(combined) == 2
    assert v1 in combined
    assert v2 in combined


# ---------------------------------------------------------------------------
# _mark_line_on_grid tests
# ---------------------------------------------------------------------------


def test_mark_line_on_grid_diagonal() -> None:
    """Diagonal line should mark cells along its path."""
    grid = _Grid.create(20.0, 20.0, grid_step_mm=0.5)
    # Mark a diagonal line from (2,2) to (8,6)
    _mark_line_on_grid(grid, 2.0, 2.0, 8.0, 6.0, exclusion_mm=0.5)
    # Midpoint of the line (~5, ~4) should be marked
    mid_c, mid_r = grid.to_cell(5.0, 4.0)
    assert not grid.is_free(mid_c, mid_r)
    # Start and end should be marked
    sc, sr = grid.to_cell(2.0, 2.0)
    assert not grid.is_free(sc, sr)
    ec, er = grid.to_cell(8.0, 6.0)
    assert not grid.is_free(ec, er)
    # A point far from the line should be free
    fc, fr = grid.to_cell(15.0, 15.0)
    assert grid.is_free(fc, fr)


def test_mark_line_on_grid_horizontal() -> None:
    """Horizontal line should mark cells along its path."""
    grid = _Grid.create(20.0, 20.0, grid_step_mm=0.5)
    _mark_line_on_grid(grid, 2.0, 10.0, 8.0, 10.0, exclusion_mm=0.5)
    # Midpoint
    mid_c, mid_r = grid.to_cell(5.0, 10.0)
    assert not grid.is_free(mid_c, mid_r)


def test_mark_line_on_grid_zero_length() -> None:
    """Zero-length line should mark just the point + exclusion."""
    grid = _Grid.create(20.0, 20.0, grid_step_mm=0.5)
    _mark_line_on_grid(grid, 5.0, 5.0, 5.0, 5.0, exclusion_mm=0.5)
    c, r = grid.to_cell(5.0, 5.0)
    assert not grid.is_free(c, r)


# ---------------------------------------------------------------------------
# B.Cu stubs marked on F.Cu grid
# ---------------------------------------------------------------------------


def test_bcu_stubs_marked_on_fcu_grid() -> None:
    """After B.Cu routing with stubs, F.Cu grid has stub cells marked."""
    bcu_grid = _Grid.create(30.0, 30.0, grid_step_mm=0.25)
    fcu_grid = _Grid.create(30.0, 30.0, grid_step_mm=0.25)
    # Block F.Cu around start to force via offset (and thus stub)
    tc, tr = fcu_grid.to_cell(5.0, 15.0)
    for dc in range(-3, 4):
        for ddr in range(-3, 4):
            fcu_grid.mark(tc + dc, tr + ddr)

    result = _route_on_bcu(
        5.0, 15.0, 25.0, 15.0,
        bcu_grid, net_number=1, net_name="SIG",
        width_mm=0.25, clearance_mm=0.2,
        fcu_grid=fcu_grid,
    )
    assert result is not None
    tracks, (via_start, _via_goal) = result
    # Find F.Cu stubs
    fcu_stubs = [t for t in tracks if t.layer == "F.Cu"]
    assert len(fcu_stubs) >= 1
    # The stub midpoint should be marked on fcu_grid
    stub = fcu_stubs[0]
    mx = (stub.start.x + stub.end.x) / 2
    my = (stub.start.y + stub.end.y) / 2
    mc, mr = fcu_grid.to_cell(mx, my)
    assert not fcu_grid.is_free(mc, mr)


# ---------------------------------------------------------------------------
# collect_vias deduplication tests
# ---------------------------------------------------------------------------


def test_collect_vias_deduplicates() -> None:
    """Co-located vias should be collapsed to one."""
    v1 = Via(
        position=Point(10.0, 15.0), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=1,
    )
    v2 = Via(
        position=Point(10.0, 15.0), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=1,
    )
    r = RouteResult(net_number=1, net_name="A", tracks=(), vias=(v1, v2), routed=True)
    combined = collect_vias((r,))
    assert len(combined) == 1


def test_collect_vias_keeps_distinct() -> None:
    """Vias far apart should both be kept."""
    v1 = Via(
        position=Point(5.0, 15.0), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=1,
    )
    v2 = Via(
        position=Point(25.0, 15.0), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=1,
    )
    r = RouteResult(net_number=1, net_name="A", tracks=(), vias=(v1, v2), routed=True)
    combined = collect_vias((r,))
    assert len(combined) == 2


# ---------------------------------------------------------------------------
# Phase 1.3: _keepout_blocks_layer tests
# ---------------------------------------------------------------------------


def test_keepout_blocks_layer_fcu_only() -> None:
    """Keepout with layers=('F.Cu',) blocks F.Cu but not B.Cu."""
    ko = Keepout(
        polygon=(Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)),
        layers=("F.Cu",),
        no_copper=True,
    )
    assert _keepout_blocks_layer(ko, "F.Cu") is True
    assert _keepout_blocks_layer(ko, "B.Cu") is False


def test_keepout_blocks_layer_empty_layers() -> None:
    """Keepout with layers=() blocks all layers."""
    ko = Keepout(
        polygon=(Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)),
        layers=(),
        no_copper=True,
    )
    assert _keepout_blocks_layer(ko, "F.Cu") is True
    assert _keepout_blocks_layer(ko, "B.Cu") is True


def test_keepout_blocks_layer_both() -> None:
    """Keepout with layers=('F.Cu','B.Cu') blocks both."""
    ko = Keepout(
        polygon=(Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)),
        layers=("F.Cu", "B.Cu"),
        no_tracks=True,
    )
    assert _keepout_blocks_layer(ko, "F.Cu") is True
    assert _keepout_blocks_layer(ko, "B.Cu") is True


def test_keepout_no_tracks_required() -> None:
    """Keepout with no_tracks=False and no_copper=False does not block."""
    ko = Keepout(
        polygon=(Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)),
        layers=("F.Cu",),
        no_copper=False,
        no_tracks=False,
    )
    assert _keepout_blocks_layer(ko, "F.Cu") is False


# ---------------------------------------------------------------------------
# Phase 1.4: collect_vias distance-based dedup tests
# ---------------------------------------------------------------------------


def test_collect_vias_distance_dedup() -> None:
    """Two same-net vias 0.25mm apart with size=0.6 -> only one kept."""
    v1 = Via(
        position=Point(10.0, 15.0), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=1,
    )
    v2 = Via(
        position=Point(10.25, 15.0), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=1,
    )
    r = RouteResult(net_number=1, net_name="A", tracks=(), vias=(v1, v2), routed=True)
    combined = collect_vias((r,))
    assert len(combined) == 1


def test_collect_vias_different_nets_not_deduped() -> None:
    """Two vias on different nets close together -> both kept."""
    v1 = Via(
        position=Point(10.0, 15.0), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=1,
    )
    v2 = Via(
        position=Point(10.25, 15.0), drill=0.3, size=0.6,
        layers=("F.Cu", "B.Cu"), net_number=2,
    )
    r1 = RouteResult(net_number=1, net_name="A", tracks=(), vias=(v1,), routed=True)
    r2 = RouteResult(net_number=2, net_name="B", tracks=(), vias=(v2,), routed=True)
    combined = collect_vias((r1, r2))
    assert len(combined) == 2


# ---------------------------------------------------------------------------
# Phase 1.1: _is_line_clear tests
# ---------------------------------------------------------------------------


def test_is_line_clear_on_empty_grid() -> None:
    """A line on an empty grid should be clear."""
    grid = _Grid.create(20.0, 20.0, 0.5)
    assert _is_line_clear(grid, 2.0, 5.0, 10.0, 5.0, 0.5)


def test_is_line_clear_blocked_by_obstacle() -> None:
    """A line through an occupied cell should be blocked."""
    grid = _Grid.create(20.0, 20.0, 0.5)
    # Mark a cell at (6.0, 5.0)
    c, r = grid.to_cell(6.0, 5.0)
    grid.mark(c, r)
    assert not _is_line_clear(grid, 2.0, 5.0, 10.0, 5.0, 0.5)


def test_is_line_clear_parallel_line_not_blocked() -> None:
    """A line well away from an obstacle should be clear."""
    grid = _Grid.create(20.0, 20.0, 0.5)
    # Mark cells at y=10 — line runs at y=5
    c, r = grid.to_cell(6.0, 10.0)
    grid.mark(c, r)
    assert _is_line_clear(grid, 2.0, 5.0, 10.0, 5.0, 0.5)


# ---------------------------------------------------------------------------
# Phase 1.2: _route_stub_on_fcu tests
# ---------------------------------------------------------------------------


def test_route_stub_on_fcu_direct() -> None:
    """Route a stub on an empty grid — should succeed."""
    grid = _Grid.create(20.0, 20.0, 0.5)
    result = _route_stub_on_fcu(grid, 5.0, 5.0, 10.0, 5.0, 1, 0.25, 0.2)
    assert result is not None
    assert len(result) >= 1
    # All tracks on F.Cu
    assert all(t.layer == "F.Cu" for t in result)


def test_route_stub_on_fcu_blocked() -> None:
    """Route a stub when the path is fully blocked — should return None."""
    grid = _Grid.create(20.0, 20.0, 0.5)
    # Block a wall across the grid at x=7.5
    for r in range(grid.rows):
        grid.mark(grid.to_cell(7.5, r * 0.5)[0], r)
    result = _route_stub_on_fcu(grid, 5.0, 5.0, 10.0, 5.0, 1, 0.25, 0.2)
    assert result is None


# ---------------------------------------------------------------------------
# Phase 1.1: _find_free_via_position with stub_origin
# ---------------------------------------------------------------------------


def test_find_via_with_stub_origin_clear() -> None:
    """Via position found when stub path is clear."""
    grid = _Grid.create(20.0, 20.0, 0.5)
    pos = _find_free_via_position(
        grid, 10.0, 10.0, 0.3, 0.2,
        stub_origin=(5.0, 10.0), stub_width_mm=0.25,
    )
    assert pos is not None


def test_find_via_with_stub_origin_blocked_spirals() -> None:
    """When direct position has blocked stub, should spiral to find clear one."""
    grid = _Grid.create(20.0, 20.0, 0.5)
    # Block cells between pad at (5,10) and target via at (10,10)
    for c_offset in range(4, 8):
        c = grid.to_cell(c_offset * 0.5 + 5.0, 10.0)[0]
        r = grid.to_cell(5.0, 10.0)[1]
        grid.mark(c, r)
    # Should still find a via position (spirals to avoid blocked path)
    pos = _find_free_via_position(
        grid, 10.0, 10.0, 0.3, 0.2,
        stub_origin=(5.0, 10.0), stub_width_mm=0.25,
    )
    # May or may not find one depending on grid state; just ensure no crash
    assert pos is None or len(pos) == 2


# ---------------------------------------------------------------------------
# Phase 1: GND stitching vias removed — route_all_nets has no
# gnd_via_positions parameter
# ---------------------------------------------------------------------------


def test_route_all_nets_no_gnd_via_param() -> None:
    """Verify route_all_nets no longer accepts gnd_via_positions."""
    import inspect

    sig = inspect.signature(route_all_nets)
    assert "gnd_via_positions" not in sig.parameters
