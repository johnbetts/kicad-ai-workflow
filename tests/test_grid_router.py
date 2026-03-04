"""Tests for kicad_pipeline.routing.grid_router."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Footprint, NetEntry, Pad, Point, Track
from kicad_pipeline.pcb.netlist import Netlist, NetlistEntry
from kicad_pipeline.routing.grid_router import (
    RouteRequest,
    RouteResult,
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
    """A tiny board should make routing impossible for far-apart pads."""
    fp1 = _make_footprint("R1", x=0.5, y=0.5)
    fp2 = _make_footprint("R2", x=1.5, y=1.5)
    req = _make_route_request(1, (("R1", "1"), ("R2", "1")))
    # Board is only 2x2 mm but pads are in opposite corners with clearance
    result = route_net(req, (fp1, fp2), board_width_mm=2.0, board_height_mm=2.0)
    assert result.routed is False


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


def test_route_gnd_uses_wider_trace() -> None:
    """Nets with 'GND' in the name should use 0.5 mm width (fallback)."""
    fp1, fp2 = _make_simple_footprints()
    entry = _make_netlist_entry(1, "GND", (("R1", "1"), ("R2", "1")))
    netlist = _make_netlist([entry])
    results = route_all_nets(netlist, (fp1, fp2), board_width_mm=30.0, board_height_mm=40.0)
    assert len(results) == 1
    result = results[0]
    if result.routed and result.tracks:
        assert all(abs(t.width - 0.5) < 1e-9 for t in result.tracks)


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
