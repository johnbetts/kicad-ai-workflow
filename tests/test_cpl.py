"""Tests for CPL (Component Placement List) generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    PCBDesign,
    Point,
)
from kicad_pipeline.production.cpl import (
    CPLRow,
    cpl_to_csv,
    generate_cpl,
    write_cpl,
)


def _make_pcb_simple() -> PCBDesign:
    """Build a simple PCBDesign for CPL testing."""
    outline = BoardOutline(polygon=(Point(0.0, 0.0), Point(50.0, 30.0)))
    design_rules = DesignRules()

    fp_front = Footprint(
        lib_id="Device:R_0805",
        ref="R1",
        value="10k",
        position=Point(10.0, 20.0),
        rotation=0.0,
        layer="F.Cu",
    )
    fp_back = Footprint(
        lib_id="Device:C_0402",
        ref="C1",
        value="100nF",
        position=Point(30.0, 15.0),
        rotation=90.0,
        layer="B.Cu",
    )

    return PCBDesign(
        outline=outline,
        design_rules=design_rules,
        nets=(),
        footprints=(fp_front, fp_back),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def test_generate_cpl_returns_tuple() -> None:
    pcb = _make_pcb_simple()
    result = generate_cpl(pcb)
    assert isinstance(result, tuple)


def test_cpl_row_frozen() -> None:
    row = CPLRow(
        designator="R1",
        val="10k",
        package="R_0805",
        mid_x=10.0,
        mid_y=20.0,
        rotation=0.0,
        layer="top",
    )
    with pytest.raises(AttributeError):
        row.designator = "changed"  # type: ignore[misc]


def test_cpl_layer_front() -> None:
    pcb = _make_pcb_simple()
    rows = generate_cpl(pcb)
    r1 = next(r for r in rows if r.designator == "R1")
    assert r1.layer == "top"


def test_cpl_layer_back() -> None:
    pcb = _make_pcb_simple()
    rows = generate_cpl(pcb)
    c1 = next(r for r in rows if r.designator == "C1")
    assert c1.layer == "bottom"


def test_cpl_position() -> None:
    pcb = _make_pcb_simple()
    rows = generate_cpl(pcb)
    r1 = next(r for r in rows if r.designator == "R1")
    assert r1.mid_x == pytest.approx(10.0)
    assert r1.mid_y == pytest.approx(20.0)


def test_cpl_rotation_default() -> None:
    pcb = _make_pcb_simple()
    rows = generate_cpl(pcb)
    r1 = next(r for r in rows if r.designator == "R1")
    # Rotation should be 0 + offset for R_0805 (which is 0)
    assert isinstance(r1.rotation, float)
    assert 0.0 <= r1.rotation < 360.0


def test_cpl_to_csv_header() -> None:
    pcb = _make_pcb_simple()
    rows = generate_cpl(pcb)
    csv_str = cpl_to_csv(rows)
    first_line = csv_str.splitlines()[0]
    assert "Designator" in first_line
    assert "Val" in first_line
    assert "Package" in first_line
    assert "Mid X" in first_line
    assert "Mid Y" in first_line
    assert "Rotation" in first_line
    assert "Layer" in first_line


def test_cpl_to_csv_data_row() -> None:
    pcb = _make_pcb_simple()
    rows = generate_cpl(pcb)
    csv_str = cpl_to_csv(rows)
    lines = csv_str.splitlines()
    assert len(lines) > 1


def test_write_cpl_creates_file(tmp_path: Path) -> None:
    pcb = _make_pcb_simple()
    rows = generate_cpl(pcb)
    out_file = tmp_path / "cpl.csv"
    write_cpl(rows, out_file)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Designator" in content


def test_cpl_package_extraction() -> None:
    pcb = _make_pcb_simple()
    rows = generate_cpl(pcb)
    r1 = next(r for r in rows if r.designator == "R1")
    # lib_id "Device:R_0805" -> package "R_0805"
    assert r1.package == "R_0805"
