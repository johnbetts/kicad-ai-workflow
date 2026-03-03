"""Tests for Excellon drill file generator."""

from __future__ import annotations

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.production.drill import generate_drill_file, generate_drill_files


def _make_pcb_with_holes() -> PCBDesign:
    """Build a PCBDesign with through-hole pads."""
    outline = BoardOutline(
        polygon=(
            Point(0.0, 0.0),
            Point(50.0, 0.0),
            Point(50.0, 30.0),
            Point(0.0, 30.0),
        )
    )
    design_rules = DesignRules()
    nets = (NetEntry(0, "GND"),)

    th_pad1 = Pad(
        number="1",
        pad_type="thru_hole",
        shape="circle",
        position=Point(0.0, 0.0),
        size_x=1.6,
        size_y=1.6,
        layers=("F.Cu", "B.Cu"),
        drill_diameter=0.8,
    )
    th_pad2 = Pad(
        number="2",
        pad_type="thru_hole",
        shape="circle",
        position=Point(2.54, 0.0),
        size_x=1.6,
        size_y=1.6,
        layers=("F.Cu", "B.Cu"),
        drill_diameter=0.8,
    )
    footprint = Footprint(
        lib_id="Connector:PinHeader_2.54mm",
        ref="J1",
        value="Connector",
        position=Point(25.0, 15.0),
        layer="F.Cu",
        pads=(th_pad1, th_pad2),
        attr="through_hole",
    )

    return PCBDesign(
        outline=outline,
        design_rules=design_rules,
        nets=nets,
        footprints=(footprint,),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_pcb_no_holes() -> PCBDesign:
    """Build a PCBDesign with no through-hole pads."""
    outline = BoardOutline(polygon=(Point(0.0, 0.0), Point(10.0, 10.0)))
    design_rules = DesignRules()
    smd_pad = Pad(
        number="1",
        pad_type="smd",
        shape="rect",
        position=Point(0.0, 0.0),
        size_x=1.0,
        size_y=0.5,
        layers=("F.Cu",),
    )
    fp = Footprint(
        lib_id="Device:R",
        ref="R1",
        value="10k",
        position=Point(5.0, 5.0),
        layer="F.Cu",
        pads=(smd_pad,),
    )
    return PCBDesign(
        outline=outline,
        design_rules=design_rules,
        nets=(),
        footprints=(fp,),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def test_generate_drill_file_returns_string() -> None:
    pcb = _make_pcb_with_holes()
    result = generate_drill_file(pcb)
    assert isinstance(result, str)


def test_drill_header_present() -> None:
    pcb = _make_pcb_with_holes()
    result = generate_drill_file(pcb)
    assert "M48" in result


def test_drill_footer_present() -> None:
    pcb = _make_pcb_with_holes()
    result = generate_drill_file(pcb)
    assert "M30" in result


def test_drill_has_tool_table() -> None:
    pcb = _make_pcb_with_holes()
    result = generate_drill_file(pcb)
    assert "T1C" in result


def test_drill_has_coordinates() -> None:
    pcb = _make_pcb_with_holes()
    result = generate_drill_file(pcb)
    assert "X" in result
    assert "Y" in result


def test_generate_drill_files_returns_dict() -> None:
    pcb = _make_pcb_with_holes()
    result = generate_drill_files(pcb)
    assert isinstance(result, dict)
    assert len(result) == 2


def test_drill_files_have_pth_npth() -> None:
    pcb = _make_pcb_with_holes()
    result = generate_drill_files(pcb)
    keys = list(result.keys())
    assert any("PTH" in k for k in keys)
    assert any("NPTH" in k for k in keys)


def test_generate_drill_file_no_holes() -> None:
    pcb = _make_pcb_no_holes()
    result = generate_drill_file(pcb)
    assert isinstance(result, str)
    assert "M48" in result
    assert "M30" in result
