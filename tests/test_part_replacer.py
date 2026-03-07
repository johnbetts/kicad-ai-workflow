"""Tests for part replacement logic."""

from __future__ import annotations

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    PCBDesign,
    Point,
)
from kicad_pipeline.production.part_replacer import (
    apply_replacements,
    replacement_map_from_report,
)
from kicad_pipeline.production.parts_validator import (
    PartStatus,
    PartsValidationReport,
)


def _make_pcb() -> PCBDesign:
    """Build a simple PCBDesign for replacement testing."""
    outline = BoardOutline(polygon=(Point(0.0, 0.0), Point(50.0, 30.0)))
    return PCBDesign(
        outline=outline,
        design_rules=DesignRules(),
        nets=(),
        footprints=(
            Footprint(
                lib_id="Device:R_0805", ref="R1", value="10k",
                position=Point(10.0, 10.0), layer="F.Cu", lcsc="C17414",
            ),
            Footprint(
                lib_id="Device:C_0402", ref="C1", value="100nF",
                position=Point(20.0, 10.0), layer="F.Cu", lcsc=None,
            ),
            Footprint(
                lib_id="Device:R_0805", ref="R2", value="10k",
                position=Point(30.0, 10.0), layer="F.Cu", lcsc="C17414",
            ),
        ),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


class TestApplyReplacements:
    def test_no_replacements_returns_same(self) -> None:
        pcb = _make_pcb()
        result = apply_replacements(pcb, {})
        assert result is pcb

    def test_replace_by_lcsc(self) -> None:
        pcb = _make_pcb()
        result = apply_replacements(pcb, {"C17414": "C25752"})
        # R1 and R2 both had C17414
        for fp in result.footprints:
            if fp.ref in ("R1", "R2"):
                assert fp.lcsc == "C25752"
            elif fp.ref == "C1":
                assert fp.lcsc is None  # untouched

    def test_replace_by_ref(self) -> None:
        pcb = _make_pcb()
        result = apply_replacements(pcb, {"C1": "C99999"})
        c1 = next(fp for fp in result.footprints if fp.ref == "C1")
        assert c1.lcsc == "C99999"
        # Others unchanged
        r1 = next(fp for fp in result.footprints if fp.ref == "R1")
        assert r1.lcsc == "C17414"

    def test_no_matching_replacement(self) -> None:
        pcb = _make_pcb()
        result = apply_replacements(pcb, {"CXXXXX": "C99999"})
        assert result is pcb  # no change


class TestReplacementMapFromReport:
    def test_extracts_replaced_parts(self) -> None:
        report = PartsValidationReport(
            project_name="test",
            timestamp="2026-03-06T00:00:00Z",
            parts=(
                PartStatus(
                    lcsc="C17414", ref_designators=("R1",), comment="10k",
                    footprint="R_0805", tier=3, status="replaced", in_stock=True,
                    stock_qty=None, unit_price_usd=0.005,
                    replacement_lcsc="C25752", replacement_reason="auto",
                    manual_url=None,
                ),
                PartStatus(
                    lcsc="C37593", ref_designators=("U1",), comment="ADS1115",
                    footprint="MSOP-10", tier=1, status="ok", in_stock=True,
                    stock_qty=None, unit_price_usd=2.50,
                    replacement_lcsc=None, replacement_reason=None,
                    manual_url=None,
                ),
            ),
            total_bom_cost_usd=2.505,
            all_parts_available=True,
            unresolved_count=0,
            summary_text="ok",
        )
        repl_map = replacement_map_from_report(report)
        assert repl_map == {"C17414": "C25752"}

    def test_replacement_by_ref_when_no_lcsc(self) -> None:
        report = PartsValidationReport(
            project_name="test",
            timestamp="2026-03-06T00:00:00Z",
            parts=(
                PartStatus(
                    lcsc="", ref_designators=("C1", "C2"), comment="100nF",
                    footprint="C_0402", tier=3, status="replaced", in_stock=True,
                    stock_qty=None, unit_price_usd=0.003,
                    replacement_lcsc="C25804",
                    replacement_reason="auto-replacement",
                    manual_url=None,
                ),
            ),
            total_bom_cost_usd=0.006,
            all_parts_available=True,
            unresolved_count=0,
            summary_text="ok",
        )
        repl_map = replacement_map_from_report(report)
        assert repl_map == {"C1": "C25804", "C2": "C25804"}

    def test_empty_report(self) -> None:
        report = PartsValidationReport(
            project_name="test",
            timestamp="2026-03-06T00:00:00Z",
            parts=(),
            total_bom_cost_usd=None,
            all_parts_available=True,
            unresolved_count=0,
            summary_text="empty",
        )
        assert replacement_map_from_report(report) == {}
