"""Tests for BOM (Bill of Materials) generator."""

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
from kicad_pipeline.production.bom import BOMRow, bom_to_csv, generate_bom, write_bom


def _make_pcb_with_components() -> PCBDesign:
    """Build PCBDesign with several footprints for BOM testing."""
    outline = BoardOutline(polygon=(Point(0.0, 0.0), Point(50.0, 30.0)))
    design_rules = DesignRules()

    fp_r1 = Footprint(
        lib_id="Device:R_0805",
        ref="R1",
        value="10k",
        position=Point(10.0, 10.0),
        layer="F.Cu",
        lcsc="C17414",
    )
    fp_r2 = Footprint(
        lib_id="Device:R_0805",
        ref="R2",
        value="10k",
        position=Point(20.0, 10.0),
        layer="F.Cu",
        lcsc="C17414",
    )
    fp_c1 = Footprint(
        lib_id="Device:C_0402",
        ref="C1",
        value="100nF",
        position=Point(30.0, 10.0),
        layer="F.Cu",
        lcsc=None,
    )

    return PCBDesign(
        outline=outline,
        design_rules=design_rules,
        nets=(),
        footprints=(fp_r1, fp_r2, fp_c1),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_pcb_three_identical() -> PCBDesign:
    """Build PCBDesign with three identical components."""
    outline = BoardOutline(polygon=(Point(0.0, 0.0), Point(50.0, 30.0)))
    design_rules = DesignRules()

    fps = []
    for i in range(1, 4):
        fps.append(
            Footprint(
                lib_id="Device:R_0805",
                ref=f"R{i}",
                value="10k",
                position=Point(float(i * 10), 10.0),
                layer="F.Cu",
                lcsc="C17414",
            )
        )

    return PCBDesign(
        outline=outline,
        design_rules=design_rules,
        nets=(),
        footprints=tuple(fps),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def test_generate_bom_returns_tuple() -> None:
    pcb = _make_pcb_with_components()
    result = generate_bom(pcb)
    assert isinstance(result, tuple)


def test_bom_row_frozen() -> None:
    row = BOMRow(
        comment="10k",
        designator="R1",
        footprint="R_0805",
        lcsc="C17414",
        quantity=1,
    )
    with pytest.raises(AttributeError):
        row.comment = "changed"  # type: ignore[misc]


def test_bom_groups_identical_components() -> None:
    pcb = _make_pcb_with_components()
    rows = generate_bom(pcb)
    # R1 and R2 have same value, lib_id, lcsc -> should be 1 row
    r_rows = [r for r in rows if r.comment == "10k"]
    assert len(r_rows) == 1
    assert r_rows[0].quantity == 2


def test_bom_designators_sorted() -> None:
    pcb = _make_pcb_with_components()
    rows = generate_bom(pcb)
    r_rows = [r for r in rows if r.comment == "10k"]
    assert len(r_rows) == 1
    designators = r_rows[0].designator.split()
    assert designators == sorted(designators)


def test_bom_to_csv_header() -> None:
    pcb = _make_pcb_with_components()
    rows = generate_bom(pcb)
    csv_str = bom_to_csv(rows)
    first_line = csv_str.splitlines()[0]
    assert "Comment" in first_line
    assert "Designator" in first_line
    assert "Footprint" in first_line


def test_bom_to_csv_data_row() -> None:
    pcb = _make_pcb_with_components()
    rows = generate_bom(pcb)
    csv_str = bom_to_csv(rows)
    lines = csv_str.splitlines()
    assert len(lines) > 1


def test_bom_lcsc_empty_string() -> None:
    pcb = _make_pcb_with_components()
    rows = generate_bom(pcb)
    c_rows = [r for r in rows if r.comment == "100nF"]
    assert len(c_rows) == 1
    assert c_rows[0].lcsc == ""


def test_write_bom_creates_file(tmp_path: Path) -> None:
    pcb = _make_pcb_with_components()
    rows = generate_bom(pcb)
    out_file = tmp_path / "bom.csv"
    write_bom(rows, out_file)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Comment" in content


def test_bom_quantity_correct() -> None:
    pcb = _make_pcb_three_identical()
    rows = generate_bom(pcb)
    assert len(rows) == 1
    assert rows[0].quantity == 3


def test_bom_to_csv_quoted_designators() -> None:
    pcb = _make_pcb_with_components()
    rows = generate_bom(pcb)
    csv_str = bom_to_csv(rows)
    # Row with two designators "R1 R2" should be quoted in CSV
    assert "R1 R2" in csv_str or '"R1 R2"' in csv_str
