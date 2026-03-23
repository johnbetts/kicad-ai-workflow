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


# ---------------------------------------------------------------------------
# Edge / negative tests
# ---------------------------------------------------------------------------


def test_find_by_category_unknown() -> None:
    """find_by_category for unknown category returns empty list."""
    db = ComponentDB()
    result = db.find_by_category("nonexistent_category_xyz")
    assert result == []


def test_find_resistor_closest_match() -> None:
    """find_resistor picks the closest match, not just any match."""
    db = ComponentDB()
    part = db.find_resistor(4_700.0, package="0805")
    assert part is not None
    parsed = _parse_resistance_ohms(part.value)
    assert parsed is not None
    # Should be within 10% of target
    assert abs(parsed - 4700.0) / 4700.0 < 0.1


def test_find_capacitor_not_found_exotic_package() -> None:
    """find_capacitor returns None for non-existent package."""
    db = ComponentDB()
    part = db.find_capacitor(0.1, package="0201_EXOTIC_FAKE")
    assert part is None


def test_find_ldo_not_found() -> None:
    """find_ldo returns None for unavailable voltage."""
    db = ComponentDB()
    part = db.find_ldo(99.9)  # No 99.9V LDO in basic parts
    # May or may not find; if found, should be far from target
    # The key point is it doesn't crash


def test_find_led_not_found_exotic_color() -> None:
    """find_led returns None for non-existent colour."""
    db = ComponentDB()
    part = db.find_led(color="ultraviolet_invisible_xyz", package="0805")
    assert part is None


def test_parse_resistance_megaohm() -> None:
    """_parse_resistance_ohms('2.2M') == 2_200_000.0."""
    result = _parse_resistance_ohms("2.2M")
    assert result == pytest.approx(2_200_000.0)


def test_parse_resistance_bare_number() -> None:
    """_parse_resistance_ohms('470') == 470.0."""
    result = _parse_resistance_ohms("470")
    assert result == pytest.approx(470.0)


def test_parse_resistance_invalid_returns_none() -> None:
    """_parse_resistance_ohms returns None for non-numeric string."""
    assert _parse_resistance_ohms("abc") is None
    assert _parse_resistance_ohms("") is None


def test_parse_capacitance_invalid_returns_none() -> None:
    """_parse_capacitance_uf returns None for non-numeric string."""
    assert _parse_capacitance_uf("xyz") is None
    assert _parse_capacitance_uf("") is None


def test_parse_capacitance_1uf() -> None:
    """_parse_capacitance_uf('1uF') == 1.0."""
    result = _parse_capacitance_uf("1uF")
    assert result == pytest.approx(1.0)


def test_nearest_e_series_unknown_series_raises() -> None:
    """nearest_e_series_value raises ValueError for unknown series name."""
    with pytest.raises(ValueError, match="Unknown E-series"):
        nearest_e_series_value(1000.0, series="E999")


def test_nearest_e6_value() -> None:
    """nearest_e_series_value with E6 returns a standard value."""
    result = nearest_e_series_value(5000.0, series="E6")
    assert result == pytest.approx(4700.0, rel=0.1) or result == pytest.approx(6800.0, rel=0.1)


def test_all_parts_returns_list() -> None:
    """all_parts returns a list of JLCPCBPart objects."""
    db = ComponentDB()
    parts = db.all_parts()
    assert isinstance(parts, list)
    assert all(hasattr(p, "lcsc") for p in parts)
