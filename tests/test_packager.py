"""Tests for production package builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    PCBDesign,
    Point,
)
from kicad_pipeline.production.bom import BOMRow
from kicad_pipeline.production.packager import (
    ProductionPackage,
    build_production_package,
    generate_order_guide,
    write_production_package,
)


def _make_pcb() -> PCBDesign:
    """Build a minimal PCBDesign for packager testing."""
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

    fp1 = Footprint(
        lib_id="Device:R_0805",
        ref="R1",
        value="10k",
        position=Point(10.0, 10.0),
        layer="F.Cu",
        lcsc="C17414",
    )
    fp2 = Footprint(
        lib_id="Device:C_0402",
        ref="C1",
        value="100nF",
        position=Point(25.0, 15.0),
        layer="F.Cu",
        lcsc=None,
    )

    return PCBDesign(
        outline=outline,
        design_rules=design_rules,
        nets=nets,
        footprints=(fp1, fp2),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def test_build_production_package_returns_dataclass() -> None:
    pcb = _make_pcb()
    pkg = build_production_package(pcb, "testproj")
    assert isinstance(pkg, ProductionPackage)


def test_production_package_has_gerbers() -> None:
    pcb = _make_pcb()
    pkg = build_production_package(pcb, "testproj")
    assert len(pkg.gerbers) > 0


def test_production_package_has_bom_csv() -> None:
    pcb = _make_pcb()
    pkg = build_production_package(pcb, "testproj")
    assert "Comment" in pkg.bom_csv


def test_production_package_has_cpl_csv() -> None:
    pcb = _make_pcb()
    pkg = build_production_package(pcb, "testproj")
    assert "Designator" in pkg.cpl_csv


def test_production_package_has_order_guide() -> None:
    pcb = _make_pcb()
    pkg = build_production_package(pcb, "testproj")
    assert "JLCPCB" in pkg.order_guide


def test_production_package_has_cost_estimate() -> None:
    pcb = _make_pcb()
    pkg = build_production_package(pcb, "testproj")
    assert "Qty" in pkg.cost_estimate


def test_generate_order_guide_returns_string() -> None:
    bom_rows: tuple[BOMRow, ...] = (
        BOMRow(
            comment="10k",
            designator="R1",
            footprint="R_0805",
            lcsc="C17414",
            quantity=1,
            unit_price_usd=0.01,
        ),
    )
    result = generate_order_guide("myproj", bom_rows, (50.0, 30.0))
    assert isinstance(result, str)
    assert "JLCPCB" in result


def test_write_production_package_creates_dirs(tmp_path: Path) -> None:
    pcb = _make_pcb()
    pkg = build_production_package(pcb, "testproj")
    write_production_package(pkg, tmp_path)
    gerbers_dir = tmp_path / "gerbers"
    assert gerbers_dir.exists()
    assert gerbers_dir.is_dir()
