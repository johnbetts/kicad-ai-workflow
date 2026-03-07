"""Tests for parts selector module."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.models.requirements import Component
from kicad_pipeline.parts.jlcpcb_db import JLCPCBPart, JLCPCBPartsDB
from kicad_pipeline.parts.selector import (
    _component_category,
    _extract_package,
    _is_passive,
    suggest_parts_for_component,
    validate_parts_selection,
)

if TYPE_CHECKING:
    from pathlib import Path


def _create_mock_db(db_path: Path) -> None:
    """Create a minimal FTS5 database for testing."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        'CREATE VIRTUAL TABLE parts USING fts5('
        '"LCSC Part", "MFR.Part", "Package", "Solder Joint", '
        '"Library Type", "Stock", "Manufacturer", "Description", '
        '"Price", "First Category", "Second Category"'
        ')'
    )
    conn.execute(
        "INSERT INTO parts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "C25804", "RC0805FR-0710KL", "0805", "2",
            "Basic", "500000", "YAGEO",
            "10kOhms 1% 0.125W 0805 Chip Resistor",
            "1-9:0.001,10-:0.0008",
            "Resistors", "Chip Resistor",
        ),
    )
    conn.execute(
        "INSERT INTO parts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "C14663", "CL21B104KBCNNNC", "0805", "2",
            "Basic", "300000", "Samsung",
            "100nF 50V X7R 0805 MLCC",
            "1-9:0.002,10-:0.001",
            "Capacitors", "Multilayer Ceramic Capacitors",
        ),
    )
    conn.commit()
    conn.close()


class TestExtractPackage:
    def test_0805_in_footprint(self) -> None:
        assert _extract_package("R_0805") == "0805"

    def test_0603_with_metric(self) -> None:
        assert _extract_package("C_0603_1608Metric") == "0603"

    def test_no_match(self) -> None:
        assert _extract_package("SOT-23") == "SOT-23"


class TestIsPassive:
    def test_resistor(self) -> None:
        assert _is_passive("R1") is True

    def test_capacitor(self) -> None:
        assert _is_passive("C3") is True

    def test_ic(self) -> None:
        assert _is_passive("U1") is False

    def test_empty(self) -> None:
        assert _is_passive("") is False


class TestComponentCategory:
    def test_resistor(self) -> None:
        assert _component_category("R1") == "Resistors"

    def test_capacitor(self) -> None:
        assert _component_category("C5") == "Capacitors"

    def test_ic(self) -> None:
        assert _component_category("U2") == "ICs"

    def test_unknown(self) -> None:
        assert _component_category("X1") is None


class TestSuggestPartsForComponent:
    @pytest.fixture()
    def mock_db(self, tmp_path: Path) -> JLCPCBPartsDB:
        db_path = tmp_path / "selector-test.db"
        _create_mock_db(db_path)
        return JLCPCBPartsDB(db_path)

    def test_suggest_resistor(self, mock_db: JLCPCBPartsDB) -> None:
        comp = Component(ref="R1", value="10k", footprint="R_0805")
        suggestion = suggest_parts_for_component(comp, mock_db)
        assert suggestion.component_ref == "R1"
        assert suggestion.preferred is not None
        assert suggestion.match_quality != "none"

    def test_suggest_capacitor(self, mock_db: JLCPCBPartsDB) -> None:
        comp = Component(ref="C1", value="100nF", footprint="C_0805")
        suggestion = suggest_parts_for_component(comp, mock_db)
        assert suggestion.component_ref == "C1"
        assert len(suggestion.candidates) >= 1

    def test_suggest_with_lcsc(self, mock_db: JLCPCBPartsDB) -> None:
        comp = Component(
            ref="R1", value="10k", footprint="R_0805", lcsc="C25804"
        )
        suggestion = suggest_parts_for_component(comp, mock_db)
        assert suggestion.match_quality == "exact"
        assert suggestion.preferred is not None
        assert suggestion.preferred.lcsc == "C25804"

    def test_suggest_no_match(self, mock_db: JLCPCBPartsDB) -> None:
        comp = Component(ref="U1", value="XYZABC123", footprint="QFP-100")
        suggestion = suggest_parts_for_component(comp, mock_db)
        assert suggestion.match_quality == "none"
        assert suggestion.preferred is None


class TestValidatePartsSelection:
    def test_in_stock_basic_part(self) -> None:
        part = JLCPCBPart(
            lcsc="C25804", mfr="YAGEO", mfr_part="RC0805FR-0710KL",
            description="10k", package="0805", category="Resistors",
            subcategory="", solder_joints=2, stock=50000,
            price=0.001, basic=True,
        )
        issues = validate_parts_selection({"R1": part})
        assert all(i.severity != "error" for i in issues)

    def test_out_of_stock(self) -> None:
        part = JLCPCBPart(
            lcsc="C99999", mfr="Test", mfr_part="TEST",
            description="test", package="0805", category="Resistors",
            subcategory="", solder_joints=2, stock=0,
            price=0.001, basic=True,
        )
        issues = validate_parts_selection({"R1": part})
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 1
        assert "out of stock" in errors[0].message

    def test_extended_part_warning(self) -> None:
        part = JLCPCBPart(
            lcsc="C2040", mfr="Espressif", mfr_part="ESP32",
            description="ESP32", package="Module", category="ICs",
            subcategory="", solder_joints=39, stock=10000,
            price=2.50, basic=False,
        )
        issues = validate_parts_selection({"U1": part})
        warnings = [i for i in issues if i.severity == "warning"]
        assert any("extended part" in w.message for w in warnings)

    def test_low_stock_warning(self) -> None:
        part = JLCPCBPart(
            lcsc="C11111", mfr="Test", mfr_part="LOW",
            description="low stock", package="0805", category="Resistors",
            subcategory="", solder_joints=2, stock=50,
            price=0.001, basic=True,
        )
        issues = validate_parts_selection({"R1": part})
        warnings = [i for i in issues if i.severity == "warning"]
        assert any("low stock" in w.message for w in warnings)
