"""Tests for kicad_pipeline.schematic.wiring."""

from __future__ import annotations

import pytest

from kicad_pipeline.constants import SCHEMATIC_WIRE_GRID_MM
from kicad_pipeline.models.requirements import Net, NetConnection
from kicad_pipeline.models.schematic import (
    GlobalLabel,
    Junction,
    Label,
    Point,
    Wire,
)
from kicad_pipeline.schematic.wiring import (
    connect_pin_to_label,
    make_global_label,
    make_junction,
    make_label,
    make_wire,
    route_net,
    snap_to_grid,
)

# ---------------------------------------------------------------------------
# snap_to_grid
# ---------------------------------------------------------------------------


def test_snap_to_grid_exact_multiple() -> None:
    """Value on grid returns unchanged."""
    grid = SCHEMATIC_WIRE_GRID_MM  # 1.27
    assert snap_to_grid(grid * 3) == pytest.approx(grid * 3)


def test_snap_to_grid_rounds_up() -> None:
    """Value just above a grid step rounds to the nearest multiple."""
    grid = SCHEMATIC_WIRE_GRID_MM
    val = grid * 2 + grid * 0.4  # slightly past 2*grid
    result = snap_to_grid(val)
    # Should round to nearest grid step
    assert result == pytest.approx(round(val / grid) * grid)


def test_snap_to_grid_rounds_down() -> None:
    """Value just below a grid step rounds to the nearest multiple."""
    grid = SCHEMATIC_WIRE_GRID_MM
    val = grid * 3 - grid * 0.1
    result = snap_to_grid(val)
    assert result == pytest.approx(round(val / grid) * grid)


def test_snap_to_grid_zero() -> None:
    """Zero stays zero."""
    assert snap_to_grid(0.0) == 0.0


def test_snap_to_grid_negative() -> None:
    """Negative values snap correctly."""
    grid = SCHEMATIC_WIRE_GRID_MM
    val = -grid * 5
    assert snap_to_grid(val) == pytest.approx(val)


def test_snap_to_grid_custom_grid() -> None:
    """Custom grid pitch works."""
    assert snap_to_grid(7.3, grid=5.0) == pytest.approx(5.0)
    assert snap_to_grid(8.0, grid=5.0) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# make_wire
# ---------------------------------------------------------------------------


def test_make_wire_returns_wire() -> None:
    """make_wire returns a Wire instance."""
    w = make_wire(0.0, 0.0, 10.0, 10.0)
    assert isinstance(w, Wire)


def test_make_wire_endpoints_snapped() -> None:
    """Wire endpoints are snapped to the schematic grid."""
    w = make_wire(0.1, 0.1, 10.1, 10.1)
    grid = SCHEMATIC_WIRE_GRID_MM
    assert w.start.x == pytest.approx(round(0.1 / grid) * grid)
    assert w.end.x == pytest.approx(round(10.1 / grid) * grid)


def test_make_wire_has_uuid() -> None:
    """Each wire gets a unique UUID."""
    w1 = make_wire(0, 0, 10, 10)
    w2 = make_wire(0, 0, 10, 10)
    assert w1.uuid != ""
    assert w2.uuid != ""
    assert w1.uuid != w2.uuid


def test_make_wire_zero_length() -> None:
    """Zero-length wire (start==end) is allowed."""
    w = make_wire(5.08, 5.08, 5.08, 5.08)
    assert w.start.x == w.end.x
    assert w.start.y == w.end.y


# ---------------------------------------------------------------------------
# make_junction
# ---------------------------------------------------------------------------


def test_make_junction_returns_junction() -> None:
    """make_junction returns a Junction instance."""
    j = make_junction(10.0, 20.0)
    assert isinstance(j, Junction)


def test_make_junction_position_snapped() -> None:
    """Junction position is snapped to grid."""
    j = make_junction(0.1, 0.2)
    grid = SCHEMATIC_WIRE_GRID_MM
    assert j.position.x == pytest.approx(round(0.1 / grid) * grid)
    assert j.position.y == pytest.approx(round(0.2 / grid) * grid)


def test_make_junction_has_uuid() -> None:
    """Junction gets a unique UUID."""
    j = make_junction(0, 0)
    assert j.uuid != ""


# ---------------------------------------------------------------------------
# make_global_label / make_label
# ---------------------------------------------------------------------------


def test_make_global_label_returns_global_label() -> None:
    """make_global_label returns a GlobalLabel instance."""
    gl = make_global_label("NET1", 10.0, 20.0)
    assert isinstance(gl, GlobalLabel)
    assert gl.text == "NET1"


def test_make_global_label_default_shape() -> None:
    """Default shape is bidirectional."""
    gl = make_global_label("SIG", 0.0, 0.0)
    assert gl.shape == "bidirectional"


def test_make_global_label_custom_shape() -> None:
    """Custom shape is preserved."""
    gl = make_global_label("SIG", 0.0, 0.0, shape="input")
    assert gl.shape == "input"


def test_make_global_label_position_snapped() -> None:
    """GlobalLabel position is snapped to grid."""
    gl = make_global_label("NET", 0.1, 0.2)
    grid = SCHEMATIC_WIRE_GRID_MM
    assert gl.position.x == pytest.approx(round(0.1 / grid) * grid)


def test_make_label_returns_label() -> None:
    """make_label returns a Label instance."""
    lbl = make_label("SIG_A", 10.0, 20.0)
    assert isinstance(lbl, Label)
    assert lbl.text == "SIG_A"


def test_make_label_position_snapped() -> None:
    """Label position is snapped to grid."""
    lbl = make_label("NET", 0.1, 0.2)
    grid = SCHEMATIC_WIRE_GRID_MM
    assert lbl.position.x == pytest.approx(round(0.1 / grid) * grid)


# ---------------------------------------------------------------------------
# connect_pin_to_label
# ---------------------------------------------------------------------------


def test_connect_pin_to_label_global_returns_wire_and_global_label() -> None:
    """Global label mode produces 1 wire + 1 global label."""
    wires, gls, lls = connect_pin_to_label(
        Point(10.0, 20.0), "NET1", is_global=True, pin_side="left",
    )
    assert len(wires) == 1
    assert len(gls) == 1
    assert len(lls) == 0
    assert gls[0].text == "NET1"


def test_connect_pin_to_label_local_returns_wire_and_local_label() -> None:
    """Local label mode produces 1 wire + 1 local label."""
    wires, gls, lls = connect_pin_to_label(
        Point(10.0, 20.0), "NET2", is_global=False, pin_side="right",
    )
    assert len(wires) == 1
    assert len(gls) == 0
    assert len(lls) == 1
    assert lls[0].text == "NET2"


def test_connect_pin_to_label_right_extends_right() -> None:
    """pin_side='right' extends the wire to the right of the pin."""
    wires, _, _ = connect_pin_to_label(
        Point(10.0, 20.0), "NET", is_global=True, pin_side="right",
    )
    assert wires[0].end.x > wires[0].start.x or wires[0].start.x > 10.0


def test_connect_pin_to_label_top_extends_upward() -> None:
    """pin_side='top' extends the wire upward (negative Y)."""
    wires, _, _ = connect_pin_to_label(
        Point(10.0, 20.0), "NET", is_global=True, pin_side="top",
    )
    # Y should decrease (upward in schematic coords)
    assert wires[0].end.y < wires[0].start.y or wires[0].start.y < 20.0


# ---------------------------------------------------------------------------
# route_net
# ---------------------------------------------------------------------------


def _make_net(name: str, connections: list[tuple[str, str]]) -> Net:
    return Net(
        name=name,
        connections=tuple(
            NetConnection(ref=r, pin=p) for r, p in connections
        ),
    )


def test_route_net_empty_positions_returns_empty() -> None:
    """No known pin positions yields empty routing result."""
    net = _make_net("SIG", [("R1", "1"), ("R2", "2")])
    wires, junctions, gls, lls = route_net(net, {})
    assert wires == []
    assert junctions == []
    assert gls == []
    assert lls == []


def test_route_net_single_pin_produces_one_label() -> None:
    """Single known pin produces one wire stub + one label."""
    net = _make_net("SIG", [("R1", "1"), ("R2", "2")])
    pin_positions = {("R1", "1"): Point(50.0, 50.0)}
    wires, junctions, gls, lls = route_net(net, pin_positions)
    assert len(wires) == 1
    assert len(gls) == 1


def test_route_net_two_pins_produces_two_labels() -> None:
    """Two pins both get label-per-pin stubs."""
    net = _make_net("SIG", [("R1", "1"), ("R2", "2")])
    pin_positions = {
        ("R1", "1"): Point(50.0, 50.0),
        ("R2", "2"): Point(100.0, 50.0),
    }
    wires, _, gls, _ = route_net(net, pin_positions)
    assert len(wires) == 2
    assert len(gls) == 2
    assert all(gl.text == "SIG" for gl in gls)


def test_route_net_local_labels() -> None:
    """use_global_labels=False produces local labels."""
    net = _make_net("SIG", [("R1", "1")])
    pin_positions = {("R1", "1"): Point(50.0, 50.0)}
    wires, _, gls, lls = route_net(
        net, pin_positions, use_global_labels=False,
    )
    assert len(gls) == 0
    assert len(lls) == 1
    assert lls[0].text == "SIG"


def test_route_net_pin_sides_respected() -> None:
    """Custom pin_sides are forwarded to connect_pin_to_label."""
    net = _make_net("SIG", [("R1", "1")])
    pin_positions = {("R1", "1"): Point(50.0, 50.0)}
    pin_sides = {("R1", "1"): "right"}
    wires, _, gls, _ = route_net(
        net, pin_positions, pin_sides=pin_sides,
    )
    assert len(wires) == 1
    # Wire should extend to the right
    w = wires[0]
    end_x = max(w.start.x, w.end.x)
    assert end_x > 50.0
