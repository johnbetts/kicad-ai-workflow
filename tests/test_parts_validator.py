"""Tests for parts validation orchestrator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.production.bom import BOMRow
from kicad_pipeline.production.parts_validator import (
    PartStatus,
    PartsValidationReport,
    _extract_package,
    _make_search_url,
    _ref_category,
    report_to_json,
    report_to_text,
    validate_bom_parts,
)


def _sample_bom_rows() -> tuple[BOMRow, ...]:
    return (
        BOMRow(comment="10k", designator="R1 R2", footprint="R_0805_2012Metric",
               lcsc="C17414", quantity=2, unit_price_usd=0.005),
        BOMRow(comment="100nF", designator="C1", footprint="C_0402_1005Metric",
               lcsc="", quantity=1),
        BOMRow(comment="ADS1115", designator="U1", footprint="MSOP-10",
               lcsc="C37593", quantity=1),
    )


class TestHelpers:
    def test_extract_package_0805(self) -> None:
        assert _extract_package("R_0805_2012Metric") == "0805"

    def test_extract_package_0402(self) -> None:
        assert _extract_package("C_0402_1005Metric") == "0402"

    def test_extract_package_no_match(self) -> None:
        assert _extract_package("MSOP-10") == ""

    def test_ref_category_resistor(self) -> None:
        assert _ref_category(("R1",)) == "resistor"

    def test_ref_category_capacitor(self) -> None:
        assert _ref_category(("C1",)) == "capacitor"

    def test_ref_category_ic(self) -> None:
        assert _ref_category(("U1",)) == "ic"

    def test_ref_category_unknown(self) -> None:
        assert _ref_category(("X1",)) == ""

    def test_ref_category_empty(self) -> None:
        assert _ref_category(()) == ""

    def test_make_search_url(self) -> None:
        url = _make_search_url("10k", "R_0805")
        assert "10k+R_0805" in url
        assert url.startswith("https://")


class TestPartStatusFrozen:
    def test_frozen(self) -> None:
        ps = PartStatus(
            lcsc="C17414", ref_designators=("R1",), comment="10k",
            footprint="R_0805", tier=1, status="ok", in_stock=True,
            stock_qty=5000, unit_price_usd=0.005, replacement_lcsc=None,
            replacement_reason=None, manual_url=None,
        )
        with pytest.raises(AttributeError):
            ps.lcsc = "other"  # type: ignore[misc]


class TestValidateBomParts:
    def test_tier1_local_db_match(self) -> None:
        """Parts found in local DB should be tier 1."""
        mock_db = MagicMock()
        mock_part = MagicMock()
        mock_part.price_usd = 0.005
        mock_part.in_stock = True
        mock_db.find_by_lcsc.return_value = mock_part

        rows = (
            BOMRow(comment="10k", designator="R1", footprint="R_0805",
                   lcsc="C17414", quantity=1),
        )
        report = validate_bom_parts(rows, db=mock_db, check_web_stock=False)
        assert report.parts[0].tier == 1
        assert report.parts[0].status == "ok"
        assert report.all_parts_available is True

    @patch("kicad_pipeline.production.parts_validator.fetch_lcsc_stock")
    def test_tier2_web_stock(self, mock_fetch: MagicMock) -> None:
        """Parts verified via web API should be tier 2."""
        from kicad_pipeline.production.lcsc_client import LCSCStockInfo

        mock_fetch.return_value = LCSCStockInfo(
            lcsc="C37593", in_stock=True, stock_qty=1000,
            unit_price_usd=2.50, description="ADS1115", package="MSOP-10",
        )
        # DB returns None so tier 1 skipped
        mock_db = MagicMock()
        mock_db.find_by_lcsc.return_value = None

        rows = (
            BOMRow(comment="ADS1115", designator="U1", footprint="MSOP-10",
                   lcsc="C37593", quantity=1),
        )
        report = validate_bom_parts(rows, db=mock_db, check_web_stock=True)
        assert report.parts[0].tier == 2
        assert report.parts[0].status == "ok"

    def test_tier3_replacement(self) -> None:
        """Parts missing LCSC should get auto-replacement from DB."""
        mock_db = MagicMock()
        mock_db.find_by_lcsc.return_value = None

        replacement_part = MagicMock()
        replacement_part.lcsc = "C25752"
        replacement_part.value = "100nF"
        replacement_part.package = "0402"
        replacement_part.mfr = "Samsung"
        replacement_part.price_usd = 0.003
        mock_db.find_capacitor.return_value = replacement_part

        rows = (
            BOMRow(comment="100nF", designator="C1", footprint="C_0402_1005Metric",
                   lcsc="", quantity=1),
        )
        report = validate_bom_parts(rows, db=mock_db, check_web_stock=False)
        assert report.parts[0].tier == 3
        assert report.parts[0].status == "replaced"
        assert report.parts[0].replacement_lcsc == "C25752"

    def test_tier4_manual(self) -> None:
        """Parts with no DB and no web should fall to tier 4."""
        rows = (
            BOMRow(comment="CustomIC", designator="U1", footprint="QFP-48",
                   lcsc="", quantity=1),
        )
        report = validate_bom_parts(rows, db=None, check_web_stock=False)
        assert report.parts[0].tier == 4
        assert report.parts[0].status == "missing_lcsc"
        assert report.parts[0].manual_url is not None
        assert report.all_parts_available is False
        assert report.unresolved_count == 1

    def test_tier4_unavailable_with_lcsc(self) -> None:
        """Parts with LCSC but out of stock everywhere → tier 4 unavailable."""
        mock_db = MagicMock()
        mock_db.find_by_lcsc.return_value = None
        mock_db.find_resistor.return_value = None
        mock_db.find_capacitor.return_value = None

        rows = (
            BOMRow(comment="SpecialR", designator="R1", footprint="R_0805",
                   lcsc="C99999", quantity=1),
        )
        report = validate_bom_parts(rows, db=mock_db, check_web_stock=False)
        assert report.parts[0].tier == 4
        assert report.parts[0].status == "unavailable"

    def test_total_cost_calculated(self) -> None:
        mock_db = MagicMock()
        mock_part = MagicMock()
        mock_part.price_usd = 0.01
        mock_part.in_stock = True
        mock_db.find_by_lcsc.return_value = mock_part

        rows = (
            BOMRow(comment="10k", designator="R1 R2", footprint="R_0805",
                   lcsc="C17414", quantity=2),
        )
        report = validate_bom_parts(rows, db=mock_db, check_web_stock=False)
        assert report.total_bom_cost_usd is not None
        assert report.total_bom_cost_usd == pytest.approx(0.02)

    def test_empty_bom(self) -> None:
        report = validate_bom_parts((), db=None, check_web_stock=False)
        assert len(report.parts) == 0
        assert report.all_parts_available is True
        assert report.unresolved_count == 0


class TestReportFormatters:
    def test_report_to_text_contains_header(self) -> None:
        report = PartsValidationReport(
            project_name="test",
            timestamp="2026-03-06T00:00:00Z",
            parts=(
                PartStatus(
                    lcsc="C17414", ref_designators=("R1",), comment="10k",
                    footprint="R_0805", tier=1, status="ok", in_stock=True,
                    stock_qty=None, unit_price_usd=0.005,
                    replacement_lcsc=None, replacement_reason=None, manual_url=None,
                ),
            ),
            total_bom_cost_usd=0.005,
            all_parts_available=True,
            unresolved_count=0,
            summary_text="All parts available.",
        )
        text = report_to_text(report)
        assert "Parts Validation Report: test" in text
        assert "OK" in text
        assert "R1" in text

    def test_report_to_json_parses(self) -> None:
        report = PartsValidationReport(
            project_name="test",
            timestamp="2026-03-06T00:00:00Z",
            parts=(),
            total_bom_cost_usd=None,
            all_parts_available=True,
            unresolved_count=0,
            summary_text="All parts available.",
        )
        json_str = report_to_json(report)
        data = json.loads(json_str)
        assert data["project_name"] == "test"
        assert data["all_parts_available"] is True
        assert isinstance(data["parts"], list)

    def test_report_to_json_with_parts(self) -> None:
        report = PartsValidationReport(
            project_name="test",
            timestamp="2026-03-06T00:00:00Z",
            parts=(
                PartStatus(
                    lcsc="C17414", ref_designators=("R1", "R2"), comment="10k",
                    footprint="R_0805", tier=1, status="ok", in_stock=True,
                    stock_qty=5000, unit_price_usd=0.005,
                    replacement_lcsc=None, replacement_reason=None, manual_url=None,
                ),
            ),
            total_bom_cost_usd=0.01,
            all_parts_available=True,
            unresolved_count=0,
            summary_text="All good.",
        )
        data = json.loads(report_to_json(report))
        assert len(data["parts"]) == 1
        assert data["parts"][0]["ref_designators"] == ["R1", "R2"]
