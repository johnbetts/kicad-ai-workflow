"""Tests for JLCPCB parts database adapter."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.exceptions import ConfigurationError
from kicad_pipeline.parts.jlcpcb_db import (
    JLCPCBPart,
    JLCPCBPartsDB,
    _parse_price,
    _parse_solder_joints,
    _parse_stock,
    discover_db_path,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestParsePrice:
    """Tests for price string parsing."""

    def test_standard_price_breaks(self) -> None:
        assert _parse_price("1-9:0.0123,10-99:0.0098,100-:0.0078") == pytest.approx(
            0.0123
        )

    def test_single_price(self) -> None:
        assert _parse_price("1-:0.05") == pytest.approx(0.05)

    def test_empty_string(self) -> None:
        assert _parse_price("") is None

    def test_whitespace(self) -> None:
        assert _parse_price("  ") is None

    def test_malformed(self) -> None:
        assert _parse_price("not-a-price") is None


class TestParseStock:
    """Tests for stock value parsing."""

    def test_integer_stock(self) -> None:
        assert _parse_stock("12345") == 12345

    def test_zero_stock(self) -> None:
        assert _parse_stock("0") == 0

    def test_invalid_stock(self) -> None:
        assert _parse_stock("N/A") == 0

    def test_none_stock(self) -> None:
        assert _parse_stock(None) == 0  # type: ignore[arg-type]


class TestParseSolderJoints:
    """Tests for solder joints parsing."""

    def test_integer(self) -> None:
        assert _parse_solder_joints("2") == 2

    def test_invalid(self) -> None:
        assert _parse_solder_joints("N/A") == 0


class TestJLCPCBPart:
    """Tests for the JLCPCBPart dataclass."""

    def test_is_in_stock_positive(self) -> None:
        part = JLCPCBPart(
            lcsc="C25804",
            mfr="YAGEO",
            mfr_part="RC0805FR-0710KL",
            description="10k resistor",
            package="0805",
            category="Resistors",
            subcategory="Chip Resistor",
            solder_joints=2,
            stock=50000,
            price=0.001,
            basic=True,
        )
        assert part.is_in_stock is True

    def test_is_in_stock_zero(self) -> None:
        part = JLCPCBPart(
            lcsc="C99999",
            mfr="Test",
            mfr_part="TEST-1",
            description="test",
            package="0805",
            category="Test",
            subcategory="",
            solder_joints=2,
            stock=0,
            price=None,
            basic=False,
        )
        assert part.is_in_stock is False

    def test_frozen(self) -> None:
        part = JLCPCBPart(
            lcsc="C1",
            mfr="",
            mfr_part="",
            description="",
            package="",
            category="",
            subcategory="",
            solder_joints=0,
            stock=0,
            price=None,
            basic=False,
        )
        with pytest.raises(AttributeError):
            part.lcsc = "C2"  # type: ignore[misc]


class TestDiscoverDbPath:
    """Tests for database path discovery."""

    def test_env_var_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_file = tmp_path / "parts-fts5.db"
        db_file.touch()
        monkeypatch.setenv("JLCPCB_PARTS_DB", str(db_file))
        assert discover_db_path() == db_file

    def test_env_var_missing_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JLCPCB_PARTS_DB", "/nonexistent/db.sqlite")
        with pytest.raises(ConfigurationError, match="non-existent file"):
            discover_db_path()

    def test_no_db_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("JLCPCB_PARTS_DB", raising=False)
        # Patch search paths to empty.
        monkeypatch.setattr(
            "kicad_pipeline.parts.jlcpcb_db._DB_SEARCH_PATHS", ()
        )
        with pytest.raises(ConfigurationError, match="Cannot find"):
            discover_db_path()


def _create_mock_db(db_path: Path) -> None:
    """Create a minimal SQLite FTS5 database for testing."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        'CREATE VIRTUAL TABLE parts USING fts5('
        '"LCSC Part", "MFR.Part", "Package", "Solder Joint", '
        '"Library Type", "Stock", "Manufacturer", "Description", '
        '"Price", "First Category", "Second Category"'
        ')'
    )
    # Insert test parts.
    conn.execute(
        "INSERT INTO parts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "C25804", "RC0805FR-0710KL", "0805", "2",
            "Basic", "500000", "YAGEO",
            "10kOhms 1% 0.125W 0805 Chip Resistor",
            "1-9:0.001,10-:0.0008",
            "Resistors", "Chip Resistor - Surface Mount",
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
    conn.execute(
        "INSERT INTO parts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "C2040", "ESP32-S3-WROOM-1-N8", "ESP32-S3-WROOM-1", "39",
            "Extended", "10000", "Espressif",
            "ESP32-S3 WiFi+BLE Module 8MB Flash",
            "1-:2.50",
            "Embedded Processors & Controllers", "WiFi Modules",
        ),
    )
    conn.execute(
        "INSERT INTO parts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "C99999", "OUT-OF-STOCK-RES", "0805", "2",
            "Basic", "0", "TestMfr",
            "Out of stock resistor 1k",
            "1-:0.001",
            "Resistors", "Chip Resistor - Surface Mount",
        ),
    )
    conn.commit()
    conn.close()


class TestJLCPCBPartsDB:
    """Tests for the database query interface."""

    @pytest.fixture()
    def mock_db(self, tmp_path: Path) -> JLCPCBPartsDB:
        db_path = tmp_path / "test-parts.db"
        _create_mock_db(db_path)
        return JLCPCBPartsDB(db_path)

    def test_search_parts_basic(self, mock_db: JLCPCBPartsDB) -> None:
        results = mock_db.search_parts("10k 0805")
        assert len(results) >= 1
        assert any(p.lcsc == "C25804" for p in results)

    def test_search_parts_by_category(self, mock_db: JLCPCBPartsDB) -> None:
        results = mock_db.search_parts("resistor", category="Resistors")
        assert all(p.category == "Resistors" for p in results)

    def test_search_parts_basic_only(self, mock_db: JLCPCBPartsDB) -> None:
        results = mock_db.search_parts("0805", basic_only=True)
        assert all(p.basic for p in results)

    def test_search_parts_in_stock(self, mock_db: JLCPCBPartsDB) -> None:
        results = mock_db.search_parts("resistor 0805", in_stock=True)
        assert all(p.stock > 0 for p in results)

    def test_search_parts_empty_query(self, mock_db: JLCPCBPartsDB) -> None:
        results = mock_db.search_parts("")
        assert results == []

    def test_get_part_found(self, mock_db: JLCPCBPartsDB) -> None:
        part = mock_db.get_part("C25804")
        assert part is not None
        assert part.lcsc == "C25804"
        assert part.mfr == "YAGEO"
        assert part.basic is True

    def test_get_part_without_prefix(self, mock_db: JLCPCBPartsDB) -> None:
        part = mock_db.get_part("25804")
        assert part is not None
        assert part.lcsc == "C25804"

    def test_get_part_not_found(self, mock_db: JLCPCBPartsDB) -> None:
        part = mock_db.get_part("C00000")
        assert part is None

    def test_find_resistor(self, mock_db: JLCPCBPartsDB) -> None:
        results = mock_db.find_resistor("10k", package="0805")
        assert len(results) >= 1

    def test_find_capacitor(self, mock_db: JLCPCBPartsDB) -> None:
        results = mock_db.find_capacitor("100nF", package="0805")
        assert len(results) >= 1

    def test_find_ic(self, mock_db: JLCPCBPartsDB) -> None:
        results = mock_db.find_ic("ESP32-S3")
        assert len(results) >= 1
        assert any(p.lcsc == "C2040" for p in results)

    def test_context_manager(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctx-test.db"
        _create_mock_db(db_path)
        with JLCPCBPartsDB(db_path) as db:
            result = db.search_parts("0805")
            assert len(result) >= 1

    def test_find_by_category(self, mock_db: JLCPCBPartsDB) -> None:
        results = mock_db.find_by_category("Capacitors")
        assert len(results) >= 1
