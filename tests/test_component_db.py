"""Tests for kicad_pipeline.requirements.component_db."""

from __future__ import annotations

import pytest

from kicad_pipeline.requirements.component_db import (
    ComponentDB,
    ESeries,
    _parse_capacitance_uf,
    _parse_resistance_ohms,
    load_e_series,
    nearest_e_series_value,
)

# ---------------------------------------------------------------------------
# ComponentDB loading
# ---------------------------------------------------------------------------


def test_load_db_from_default_file() -> None:
    """ComponentDB loads from the default bundled file without error."""
    db = ComponentDB()
    assert db is not None


def test_all_parts_not_empty() -> None:
    """all_parts() returns a non-empty list."""
    db = ComponentDB()
    parts = db.all_parts()
    assert len(parts) > 0


# ---------------------------------------------------------------------------
# find_by_lcsc
# ---------------------------------------------------------------------------


def test_find_by_lcsc_found() -> None:
    """find_by_lcsc returns the correct part for LCSC C17414 (10k resistor)."""
    db = ComponentDB()
    part = db.find_by_lcsc("C17414")
    assert part is not None
    assert part.lcsc == "C17414"
    assert part.value == "10k"
    assert part.category == "resistor"


def test_find_by_lcsc_not_found() -> None:
    """find_by_lcsc returns None for an unknown LCSC number."""
    db = ComponentDB()
    part = db.find_by_lcsc("C99999999")
    assert part is None


# ---------------------------------------------------------------------------
# find_by_category
# ---------------------------------------------------------------------------


def test_find_by_category_resistor() -> None:
    """find_by_category('resistor') returns a non-empty list of resistors."""
    db = ComponentDB()
    resistors = db.find_by_category("resistor")
    assert len(resistors) > 0
    for r in resistors:
        assert r.category == "resistor"


# ---------------------------------------------------------------------------
# find_resistor
# ---------------------------------------------------------------------------


def test_find_resistor_10k() -> None:
    """find_resistor finds the 10k resistor in 0805 package."""
    db = ComponentDB()
    part = db.find_resistor(10_000.0, package="0805")
    assert part is not None
    assert part.package == "0805"
    # The value should be '10k'
    assert part.value == "10k"


def test_find_resistor_not_found() -> None:
    """find_resistor returns None for an exotic value in a non-existent package."""
    db = ComponentDB()
    # Use a package that is definitely not in the database
    part = db.find_resistor(10_000.0, package="0201_EXOTIC_FAKE")
    assert part is None


# ---------------------------------------------------------------------------
# _parse_resistance_ohms
# ---------------------------------------------------------------------------


def test_parse_resistance_100r() -> None:
    """_parse_resistance_ohms('100R') == 100.0."""
    result = _parse_resistance_ohms("100R")
    assert result == pytest.approx(100.0)


def test_parse_resistance_10k() -> None:
    """_parse_resistance_ohms('10k') == 10000.0."""
    result = _parse_resistance_ohms("10k")
    assert result == pytest.approx(10_000.0)


def test_parse_resistance_4_7k() -> None:
    """_parse_resistance_ohms('4.7k') == 4700.0."""
    result = _parse_resistance_ohms("4.7k")
    assert result == pytest.approx(4_700.0)


# ---------------------------------------------------------------------------
# _parse_capacitance_uf
# ---------------------------------------------------------------------------


def test_parse_capacitance_100nf() -> None:
    """_parse_capacitance_uf('100nF') == 0.1."""
    result = _parse_capacitance_uf("100nF")
    assert result == pytest.approx(0.1)


def test_parse_capacitance_10uf() -> None:
    """_parse_capacitance_uf('10uF') == 10.0."""
    result = _parse_capacitance_uf("10uF")
    assert result == pytest.approx(10.0)


def test_parse_capacitance_22pf() -> None:
    """_parse_capacitance_uf('22pF') is approximately 2.2e-5 µF."""
    result = _parse_capacitance_uf("22pF")
    assert result is not None
    assert result == pytest.approx(2.2e-5, rel=1e-4)


# ---------------------------------------------------------------------------
# find_capacitor
# ---------------------------------------------------------------------------


def test_find_capacitor_100nf() -> None:
    """find_capacitor finds a 100nF capacitor in 0805 package."""
    db = ComponentDB()
    part = db.find_capacitor(0.1, package="0805")
    assert part is not None
    assert part.package == "0805"
    assert "100nF" in part.value or "100" in part.value


# ---------------------------------------------------------------------------
# find_ldo
# ---------------------------------------------------------------------------


def test_find_ldo_3v3() -> None:
    """find_ldo finds a 3.3V LDO."""
    db = ComponentDB()
    part = db.find_ldo(3.3)
    assert part is not None
    assert part.vout is not None
    assert part.vout == pytest.approx(3.3, abs=0.05)


# ---------------------------------------------------------------------------
# E-series
# ---------------------------------------------------------------------------


def test_e_series_loaded() -> None:
    """load_e_series() returns ESeries with E24 containing 24 values."""
    e = load_e_series()
    assert isinstance(e, ESeries)
    assert len(e.E24) == 24


def test_nearest_e24_value() -> None:
    """nearest_e_series_value(9500) returns a value close to 9100 or 10000."""
    result = nearest_e_series_value(9500.0, series="E24")
    # E24 values near 9500 are 9100 and 10000; accept either decade-scaled
    assert result == pytest.approx(9100.0, rel=0.05) or result == pytest.approx(10_000.0, rel=0.05)
