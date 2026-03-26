"""Tests for kicad_pipeline.models.pcb — PCBDesign, FootprintBBox, etc."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    FootprintBBox,
    NetEntry,
    PCBDesign,
    Point,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pcb(
    footprints: tuple[Footprint, ...] = (),
    nets: tuple[NetEntry, ...] = (),
) -> PCBDesign:
    """Build a minimal PCBDesign for testing."""
    outline = BoardOutline(
        polygon=(
            Point(0.0, 0.0), Point(100.0, 0.0),
            Point(100.0, 80.0), Point(0.0, 80.0),
        )
    )
    return PCBDesign(
        outline=outline,
        design_rules=DesignRules(),
        nets=nets,
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_footprint(ref: str = "R1", value: str = "10k") -> Footprint:
    return Footprint(
        lib_id="Device:R_0805",
        ref=ref,
        value=value,
        position=Point(10.0, 20.0),
    )


# ---------------------------------------------------------------------------
# PCBDesign.get_footprint
# ---------------------------------------------------------------------------


def test_get_footprint_found() -> None:
    """get_footprint returns matching Footprint when ref exists."""
    fp = _make_footprint("R1")
    pcb = _make_pcb(footprints=(fp,))
    result = pcb.get_footprint("R1")
    assert result is not None
    assert result.ref == "R1"


def test_get_footprint_not_found() -> None:
    """get_footprint returns None for unknown ref."""
    fp = _make_footprint("R1")
    pcb = _make_pcb(footprints=(fp,))
    assert pcb.get_footprint("C99") is None


def test_get_footprint_empty_design() -> None:
    """get_footprint returns None on empty design."""
    pcb = _make_pcb()
    assert pcb.get_footprint("R1") is None


def test_get_footprint_multiple() -> None:
    """get_footprint returns correct one among several."""
    fps = (_make_footprint("R1"), _make_footprint("R2", "4.7k"), _make_footprint("C1", "100nF"))
    pcb = _make_pcb(footprints=fps)
    result = pcb.get_footprint("R2")
    assert result is not None
    assert result.value == "4.7k"


# ---------------------------------------------------------------------------
# PCBDesign.get_net_number
# ---------------------------------------------------------------------------


def test_get_net_number_found() -> None:
    """get_net_number returns correct number for known net."""
    nets = (NetEntry(0, ""), NetEntry(1, "GND"), NetEntry(2, "+3V3"))
    pcb = _make_pcb(nets=nets)
    assert pcb.get_net_number("GND") == 1
    assert pcb.get_net_number("+3V3") == 2


def test_get_net_number_not_found() -> None:
    """get_net_number returns None for unknown net name."""
    nets = (NetEntry(0, ""), NetEntry(1, "GND"))
    pcb = _make_pcb(nets=nets)
    assert pcb.get_net_number("NONEXISTENT") is None


def test_get_net_number_empty_nets() -> None:
    """get_net_number returns None on design with no nets."""
    pcb = _make_pcb()
    assert pcb.get_net_number("GND") is None


# ---------------------------------------------------------------------------
# FootprintBBox properties
# ---------------------------------------------------------------------------


def test_bbox_width() -> None:
    """width property returns max_x - min_x."""
    bbox = FootprintBBox(min_x=-1.0, min_y=-0.5, max_x=1.0, max_y=0.5)
    assert bbox.width == pytest.approx(2.0)


def test_bbox_height() -> None:
    """height property returns max_y - min_y."""
    bbox = FootprintBBox(min_x=-1.0, min_y=-0.5, max_x=1.0, max_y=0.5)
    assert bbox.height == pytest.approx(1.0)


def test_bbox_center_offset_symmetric() -> None:
    """center_offset is (0,0) for a symmetric box."""
    bbox = FootprintBBox(min_x=-2.0, min_y=-1.0, max_x=2.0, max_y=1.0)
    dx, dy = bbox.center_offset
    assert dx == pytest.approx(0.0)
    assert dy == pytest.approx(0.0)


def test_bbox_center_offset_asymmetric() -> None:
    """center_offset is non-zero for an asymmetric box (pin-1 origin)."""
    bbox = FootprintBBox(min_x=0.0, min_y=0.0, max_x=10.0, max_y=4.0)
    dx, dy = bbox.center_offset
    assert dx == pytest.approx(5.0)
    assert dy == pytest.approx(2.0)


def test_bbox_rotated_0_unchanged() -> None:
    """rotated(0) returns the same bbox."""
    bbox = FootprintBBox(min_x=-1.0, min_y=-0.5, max_x=1.0, max_y=0.5)
    r = bbox.rotated(0.0)
    assert r.width == pytest.approx(bbox.width, abs=1e-9)
    assert r.height == pytest.approx(bbox.height, abs=1e-9)


def test_bbox_rotated_90_swaps_dims() -> None:
    """rotated(90) swaps width and height."""
    bbox = FootprintBBox(min_x=-2.0, min_y=-0.5, max_x=2.0, max_y=0.5)
    r = bbox.rotated(90.0)
    assert r.width == pytest.approx(bbox.height, abs=1e-6)
    assert r.height == pytest.approx(bbox.width, abs=1e-6)


def test_bbox_rotated_360_identity() -> None:
    """rotated(360) returns same bbox as original."""
    bbox = FootprintBBox(min_x=-3.0, min_y=-1.0, max_x=3.0, max_y=1.0)
    r = bbox.rotated(360.0)
    assert r.min_x == pytest.approx(bbox.min_x, abs=1e-9)
    assert r.max_x == pytest.approx(bbox.max_x, abs=1e-9)


# ---------------------------------------------------------------------------
# Frozen dataclass enforcement
# ---------------------------------------------------------------------------


def test_pcb_design_frozen() -> None:
    """PCBDesign is frozen — attribute assignment raises."""
    pcb = _make_pcb()
    with pytest.raises(AttributeError):
        pcb.title = "changed"  # type: ignore[misc]


def test_footprint_frozen() -> None:
    """Footprint is frozen — attribute assignment raises."""
    fp = _make_footprint()
    with pytest.raises(AttributeError):
        fp.ref = "changed"  # type: ignore[misc]


def test_point_frozen() -> None:
    """Point is frozen — attribute assignment raises."""
    p = Point(1.0, 2.0)
    with pytest.raises(AttributeError):
        p.x = 99.0  # type: ignore[misc]
