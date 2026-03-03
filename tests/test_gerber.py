"""Tests for RS-274X Gerber file generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
    Track,
)
from kicad_pipeline.production.gerber import (
    _mm_to_gerber,
    generate_all_gerbers,
    generate_copper_layer,
    generate_edge_cuts,
    write_gerbers,
)


def _make_pcb() -> PCBDesign:
    """Build a minimal PCBDesign with tracks, pads, and outline."""
    outline = BoardOutline(
        polygon=(
            Point(0.0, 0.0),
            Point(50.0, 0.0),
            Point(50.0, 30.0),
            Point(0.0, 30.0),
        )
    )
    design_rules = DesignRules()
    nets = (NetEntry(0, "GND"), NetEntry(1, "VCC"))

    smd_pad = Pad(
        number="1",
        pad_type="smd",
        shape="rect",
        position=Point(0.0, 0.0),
        size_x=1.5,
        size_y=1.0,
        layers=("F.Cu",),
        net_number=1,
        net_name="VCC",
    )
    footprint = Footprint(
        lib_id="Device:R_0805",
        ref="R1",
        value="10k",
        position=Point(10.0, 10.0),
        rotation=0.0,
        layer="F.Cu",
        pads=(smd_pad,),
    )

    track = Track(
        start=Point(5.0, 5.0),
        end=Point(15.0, 5.0),
        width=0.25,
        layer="F.Cu",
        net_number=1,
    )

    return PCBDesign(
        outline=outline,
        design_rules=design_rules,
        nets=nets,
        footprints=(footprint,),
        tracks=(track,),
        vias=(),
        zones=(),
        keepouts=(),
    )


def test_generate_copper_returns_string() -> None:
    pcb = _make_pcb()
    result = generate_copper_layer(pcb, "F.Cu")
    assert isinstance(result, str)


def test_gerber_header_present() -> None:
    pcb = _make_pcb()
    result = generate_copper_layer(pcb, "F.Cu")
    assert "%FSLAX46Y46*%" in result


def test_gerber_footer_present() -> None:
    pcb = _make_pcb()
    result = generate_copper_layer(pcb, "F.Cu")
    assert "M02*" in result


def test_gerber_has_track() -> None:
    pcb = _make_pcb()
    result = generate_copper_layer(pcb, "F.Cu")
    assert "D01*" in result


def test_gerber_pad_flash() -> None:
    pcb = _make_pcb()
    result = generate_copper_layer(pcb, "F.Cu")
    assert "D03*" in result


def test_edge_cuts_returns_string() -> None:
    pcb = _make_pcb()
    result = generate_edge_cuts(pcb)
    assert isinstance(result, str)


def test_edge_cuts_has_moves() -> None:
    pcb = _make_pcb()
    result = generate_edge_cuts(pcb)
    assert "D02*" in result
    assert "D01*" in result


def test_generate_all_gerbers_dict() -> None:
    pcb = _make_pcb()
    result = generate_all_gerbers(pcb)
    assert isinstance(result, dict)
    assert len(result) == 7


def test_gerber_filenames() -> None:
    pcb = _make_pcb()
    result = generate_all_gerbers(pcb, project_name="myproj")
    keys = list(result.keys())
    assert any("F_Cu" in k for k in keys)
    assert any("B_Cu" in k for k in keys)


def test_write_gerbers_creates_files(tmp_path: Path) -> None:
    pcb = _make_pcb()
    gerbers = generate_all_gerbers(pcb, project_name="test")
    write_gerbers(gerbers, tmp_path)
    created = list(tmp_path.iterdir())
    assert len(created) == 7


def test_mm_to_gerber_positive() -> None:
    assert _mm_to_gerber(1.0) == "1000000"


def test_mm_to_gerber_negative() -> None:
    assert _mm_to_gerber(-1.0) == "-1000000"
