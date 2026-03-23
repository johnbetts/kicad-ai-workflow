"""Tests for kicad_pipeline.pcb.zones."""

from __future__ import annotations

import math

import pytest

from kicad_pipeline.constants import (
    LAYER_B_CU,
    LAYER_F_CU,
    THERMAL_RELIEF_BRIDGE_MM,
    THERMAL_RELIEF_GAP_MM,
)
from kicad_pipeline.models.pcb import BoardOutline, ZoneFill, ZonePolygon
from kicad_pipeline.models.pcb import Keepout, NetEntry
from kicad_pipeline.pcb.netlist import Netlist, NetlistEntry
from kicad_pipeline.pcb.zones import (
    _CIRCLE_SEGMENTS,
    gnd_pours_both_layers,
    make_antenna_keepout,
    make_board_outline,
    make_edge_keepout,
    make_gnd_pour,
    make_mounting_hole_keepout,
    make_power_pour,
    make_standard_keepouts,
    make_standard_zones,
)

# ---------------------------------------------------------------------------
# Board outline
# ---------------------------------------------------------------------------


def test_make_board_outline_rectangle() -> None:
    """Rectangular board outline must have exactly 4 polygon points."""
    outline = make_board_outline(100.0, 80.0)
    assert len(outline.polygon) == 4


def test_board_outline_dimensions() -> None:
    """Board outline polygon must span the full width and height."""
    w, h = 100.0, 80.0
    outline = make_board_outline(w, h)
    xs = [p.x for p in outline.polygon]
    ys = [p.y for p in outline.polygon]
    assert max(xs) == pytest.approx(w)
    assert max(ys) == pytest.approx(h)
    assert min(xs) == pytest.approx(0.0)
    assert min(ys) == pytest.approx(0.0)


def test_board_outline_first_point_at_origin() -> None:
    """The first point of the board outline must be (0, 0)."""
    outline = make_board_outline(50.0, 30.0)
    assert outline.polygon[0].x == pytest.approx(0.0)
    assert outline.polygon[0].y == pytest.approx(0.0)


def test_board_outline_corner_radius_nonzero_raises() -> None:
    """Passing corner_radius_mm != 0 must raise ValueError."""
    with pytest.raises(ValueError):
        make_board_outline(100.0, 80.0, corner_radius_mm=2.0)


def test_board_outline_is_frozen() -> None:
    """BoardOutline must be immutable (frozen dataclass)."""
    outline = make_board_outline(100.0, 80.0)
    with pytest.raises(AttributeError):
        outline.width = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GND pour
# ---------------------------------------------------------------------------


def test_make_gnd_pour_layer() -> None:
    """GND pour must be placed on the specified layer."""
    outline = make_board_outline(100.0, 80.0)
    pour = make_gnd_pour(outline, net_number=1, layer=LAYER_F_CU)
    assert pour.layer == LAYER_F_CU


def test_make_gnd_pour_back_layer() -> None:
    """GND pour placed on B.Cu must report B.Cu as its layer."""
    outline = make_board_outline(100.0, 80.0)
    pour = make_gnd_pour(outline, net_number=1, layer=LAYER_B_CU)
    assert pour.layer == LAYER_B_CU


def test_make_gnd_pour_net_number() -> None:
    """GND pour must carry the net number passed as argument."""
    outline = make_board_outline(100.0, 80.0)
    pour = make_gnd_pour(outline, net_number=3, layer=LAYER_F_CU)
    assert pour.net_number == 3


def test_make_gnd_pour_net_name() -> None:
    """GND pour must have net_name='GND'."""
    outline = make_board_outline(100.0, 80.0)
    pour = make_gnd_pour(outline, net_number=1, layer=LAYER_F_CU)
    assert pour.net_name == "GND"


def test_make_gnd_pour_solid_fill() -> None:
    """GND pour fill style must be SOLID."""
    outline = make_board_outline(100.0, 80.0)
    pour = make_gnd_pour(outline, net_number=1, layer=LAYER_F_CU)
    assert pour.fill == ZoneFill.SOLID


def test_make_gnd_pour_thermal_relief() -> None:
    """GND pour thermal relief values match constants."""
    outline = make_board_outline(100.0, 80.0)
    pour = make_gnd_pour(outline, net_number=1, layer=LAYER_F_CU)
    assert pour.thermal_relief_gap == pytest.approx(THERMAL_RELIEF_GAP_MM)
    assert pour.thermal_relief_bridge == pytest.approx(THERMAL_RELIEF_BRIDGE_MM)


def test_make_gnd_pour_polygon_matches_outline() -> None:
    """GND pour polygon must be the same as the board outline polygon."""
    outline = make_board_outline(100.0, 80.0)
    pour = make_gnd_pour(outline, net_number=1, layer=LAYER_F_CU)
    assert pour.polygon == outline.polygon


def test_zone_polygon_is_frozen() -> None:
    """ZonePolygon must be immutable (frozen dataclass)."""
    outline = make_board_outline(50.0, 40.0)
    pour = make_gnd_pour(outline, net_number=1, layer=LAYER_F_CU)
    with pytest.raises(AttributeError):
        pour.net_number = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GND pours — both layers
# ---------------------------------------------------------------------------


def test_gnd_pours_both_layers() -> None:
    """gnd_pours_both_layers returns two zones on F.Cu and B.Cu."""
    outline = make_board_outline(100.0, 80.0)
    front, back = gnd_pours_both_layers(outline, gnd_net_number=1)
    assert front.layer == LAYER_F_CU
    assert back.layer == LAYER_B_CU


def test_gnd_pours_both_layers_same_net() -> None:
    """Both pours from gnd_pours_both_layers must share the given net number."""
    outline = make_board_outline(100.0, 80.0)
    front, back = gnd_pours_both_layers(outline, gnd_net_number=1)
    assert front.net_number == 1
    assert back.net_number == 1


def test_gnd_pours_both_layers_returns_zone_polygons() -> None:
    """gnd_pours_both_layers must return ZonePolygon instances."""
    outline = make_board_outline(100.0, 80.0)
    result = gnd_pours_both_layers(outline, gnd_net_number=1)
    assert all(isinstance(z, ZonePolygon) for z in result)


# ---------------------------------------------------------------------------
# Antenna keepout
# ---------------------------------------------------------------------------


def test_make_antenna_keepout_no_copper() -> None:
    """Antenna keepout must have no_copper=True."""
    ko = make_antenna_keepout(10.0, 10.0)
    assert ko.no_copper is True


def test_make_antenna_keepout_no_vias() -> None:
    """Antenna keepout must have no_vias=True."""
    ko = make_antenna_keepout(10.0, 10.0)
    assert ko.no_vias is True


def test_make_antenna_keepout_no_tracks() -> None:
    """Antenna keepout must have no_tracks=True."""
    ko = make_antenna_keepout(10.0, 10.0)
    assert ko.no_tracks is True


def test_make_antenna_keepout_polygon() -> None:
    """Antenna keepout polygon must have at least 4 points."""
    ko = make_antenna_keepout(10.0, 10.0, width_mm=20.0, height_mm=12.0)
    assert len(ko.polygon) >= 4


def test_make_antenna_keepout_dimensions() -> None:
    """Antenna keepout polygon spans the given width and height (center-based)."""
    cx, cy, w, h = 15.0, 9.0, 20.0, 12.0
    ko = make_antenna_keepout(cx, cy, width_mm=w, height_mm=h)
    xs = [p.x for p in ko.polygon]
    ys = [p.y for p in ko.polygon]
    assert max(xs) == pytest.approx(cx + w / 2.0)
    assert max(ys) == pytest.approx(cy + h / 2.0)
    assert min(xs) == pytest.approx(cx - w / 2.0)
    assert min(ys) == pytest.approx(cy - h / 2.0)


# ---------------------------------------------------------------------------
# Mounting hole keepout
# ---------------------------------------------------------------------------


def test_make_mounting_hole_keepout() -> None:
    """Mounting hole keepout polygon must approximate a circle."""
    ko = make_mounting_hole_keepout(10.0, 10.0, diameter_mm=3.2, clearance_mm=1.0)
    assert len(ko.polygon) == _CIRCLE_SEGMENTS


def test_mounting_hole_keepout_radius() -> None:
    """Mounting hole keepout polygon points are approx at the correct radius."""
    cx, cy = 10.0, 10.0
    diameter = 3.2
    clearance = 1.0
    expected_r = diameter / 2.0 + clearance
    ko = make_mounting_hole_keepout(cx, cy, diameter_mm=diameter, clearance_mm=clearance)
    for pt in ko.polygon:
        r = math.hypot(pt.x - cx, pt.y - cy)
        assert r == pytest.approx(expected_r, abs=1e-9)


def test_mounting_hole_keepout_no_copper() -> None:
    """Mounting hole keepout must have no_copper=True."""
    ko = make_mounting_hole_keepout(0.0, 0.0)
    assert ko.no_copper is True


def test_board_outline_type() -> None:
    """make_board_outline must return a BoardOutline instance."""
    outline = make_board_outline(100.0, 80.0)
    assert isinstance(outline, BoardOutline)


# ---------------------------------------------------------------------------
# BUG-18: filled_polygons
# ---------------------------------------------------------------------------


def test_make_gnd_pour_has_filled_polygons() -> None:
    """make_gnd_pour returns a zone with non-empty filled_polygons."""
    outline = make_board_outline(50.0, 30.0)
    zone = make_gnd_pour(outline, net_number=1, net_name="GND", layer="B.Cu")
    assert len(zone.filled_polygons) == 1
    fill_poly = zone.filled_polygons[0]
    # Should be a closed 5-point inset rectangle
    assert len(fill_poly) == 5
    # Inset should be smaller than outline
    xs = [p.x for p in fill_poly]
    ys = [p.y for p in fill_poly]
    assert min(xs) > 0.0
    assert min(ys) > 0.0
    assert max(xs) < 50.0
    assert max(ys) < 30.0


# ---------------------------------------------------------------------------
# make_power_pour
# ---------------------------------------------------------------------------


def test_make_power_pour_layer() -> None:
    """Power pour defaults to F.Cu."""
    outline = make_board_outline(80.0, 60.0)
    pour = make_power_pour(outline, net_number=5, net_name="+3V3")
    assert pour.layer == LAYER_F_CU


def test_make_power_pour_custom_layer() -> None:
    """Power pour accepts a custom layer."""
    outline = make_board_outline(80.0, 60.0)
    pour = make_power_pour(outline, net_number=5, net_name="+5V", layer=LAYER_B_CU)
    assert pour.layer == LAYER_B_CU


def test_make_power_pour_net_info() -> None:
    """Power pour carries correct net number and name."""
    outline = make_board_outline(80.0, 60.0)
    pour = make_power_pour(outline, net_number=7, net_name="+12V")
    assert pour.net_number == 7
    assert pour.net_name == "+12V"


def test_make_power_pour_solid_fill() -> None:
    outline = make_board_outline(50.0, 40.0)
    pour = make_power_pour(outline, net_number=2, net_name="+5V")
    assert pour.fill == ZoneFill.SOLID


def test_make_power_pour_polygon_matches_outline() -> None:
    outline = make_board_outline(60.0, 40.0)
    pour = make_power_pour(outline, net_number=3, net_name="+3V3")
    assert pour.polygon == outline.polygon


def test_make_power_pour_thermal_relief() -> None:
    outline = make_board_outline(50.0, 30.0)
    pour = make_power_pour(outline, net_number=2, net_name="+5V")
    assert pour.thermal_relief_gap == pytest.approx(THERMAL_RELIEF_GAP_MM)
    assert pour.thermal_relief_bridge == pytest.approx(THERMAL_RELIEF_BRIDGE_MM)


# ---------------------------------------------------------------------------
# make_edge_keepout
# ---------------------------------------------------------------------------


def test_make_edge_keepout_rectangular() -> None:
    """Edge keepout for a rectangular board returns an inset rectangle."""
    outline = make_board_outline(100.0, 80.0)
    ko = make_edge_keepout(outline, margin_mm=1.0)
    assert isinstance(ko, Keepout)
    xs = [p.x for p in ko.polygon]
    ys = [p.y for p in ko.polygon]
    assert min(xs) == pytest.approx(1.0)
    assert min(ys) == pytest.approx(1.0)
    assert max(xs) == pytest.approx(99.0)
    assert max(ys) == pytest.approx(79.0)


def test_make_edge_keepout_no_copper_false() -> None:
    """Edge keepout does not block copper (only tracks)."""
    outline = make_board_outline(50.0, 40.0)
    ko = make_edge_keepout(outline)
    assert ko.no_copper is False
    assert ko.no_tracks is True


def test_make_edge_keepout_default_margin() -> None:
    """Default margin is 0.3mm."""
    outline = make_board_outline(50.0, 40.0)
    ko = make_edge_keepout(outline)
    xs = [p.x for p in ko.polygon]
    assert min(xs) == pytest.approx(0.3)
    assert max(xs) == pytest.approx(49.7)


def test_make_edge_keepout_non_rectangular() -> None:
    """Non-rectangular outline reuses original polygon."""
    from kicad_pipeline.models.pcb import Point

    triangle = BoardOutline(
        polygon=(Point(0.0, 0.0), Point(50.0, 0.0), Point(25.0, 40.0)),
    )
    ko = make_edge_keepout(triangle)
    # Should use original polygon (3 points, not 5)
    assert len(ko.polygon) == 3
    assert ko.no_tracks is True


def test_make_edge_keepout_both_layers() -> None:
    """Edge keepout covers both copper layers."""
    outline = make_board_outline(50.0, 40.0)
    ko = make_edge_keepout(outline)
    assert "F.Cu" in ko.layers
    assert "B.Cu" in ko.layers


# ---------------------------------------------------------------------------
# make_standard_zones
# ---------------------------------------------------------------------------


def _make_netlist_with_entries(*entries: NetlistEntry) -> Netlist:
    return Netlist(entries=tuple(entries))


def test_make_standard_zones_gnd_only() -> None:
    """With no power net, standard zones has exactly one GND pour."""
    outline = make_board_outline(60.0, 40.0)
    nl = _make_netlist_with_entries(
        NetlistEntry(net=NetEntry(number=1, name="GND"), pad_refs=()),
    )
    zones = make_standard_zones(outline, nl)
    assert len(zones) == 1
    assert zones[0].net_name == "GND"


def test_make_standard_zones_with_3v3() -> None:
    """When +3V3 net exists, standard zones includes a power zone."""
    outline = make_board_outline(60.0, 40.0)
    nl = _make_netlist_with_entries(
        NetlistEntry(net=NetEntry(number=1, name="GND"), pad_refs=()),
        NetlistEntry(net=NetEntry(number=2, name="+3V3"), pad_refs=()),
    )
    zones = make_standard_zones(outline, nl)
    assert len(zones) == 2
    names = {z.net_name for z in zones}
    assert "GND" in names
    assert "+3V3" in names


def test_make_standard_zones_with_3_3v() -> None:
    """Alternative +3.3V naming also triggers power zone."""
    outline = make_board_outline(60.0, 40.0)
    nl = _make_netlist_with_entries(
        NetlistEntry(net=NetEntry(number=1, name="GND"), pad_refs=()),
        NetlistEntry(net=NetEntry(number=2, name="+3.3V"), pad_refs=()),
    )
    zones = make_standard_zones(outline, nl)
    assert len(zones) == 2


def test_make_standard_zones_empty_netlist() -> None:
    """With empty netlist, still get the GND pour."""
    outline = make_board_outline(60.0, 40.0)
    nl = Netlist(entries=())
    zones = make_standard_zones(outline, nl)
    assert len(zones) == 1
    assert zones[0].net_name == "GND"


def test_make_standard_zones_returns_zone_polygons() -> None:
    outline = make_board_outline(50.0, 30.0)
    nl = Netlist(entries=())
    zones = make_standard_zones(outline, nl)
    assert all(isinstance(z, ZonePolygon) for z in zones)


# ---------------------------------------------------------------------------
# make_standard_keepouts
# ---------------------------------------------------------------------------


def test_make_standard_keepouts_returns_edge_keepout() -> None:
    outline = make_board_outline(60.0, 40.0)
    keepouts = make_standard_keepouts(outline)
    assert len(keepouts) == 1
    assert isinstance(keepouts[0], Keepout)
    assert keepouts[0].no_tracks is True


def test_make_standard_keepouts_returns_tuple() -> None:
    outline = make_board_outline(50.0, 30.0)
    keepouts = make_standard_keepouts(outline)
    assert isinstance(keepouts, tuple)
