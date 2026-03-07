"""Tests for KiCad library resolver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.parts.jlcpcb_db import JLCPCBPart
from kicad_pipeline.parts.library_resolver import (
    _category_to_cdfer_file,
    _footprint_for_passive,
    _is_passive,
    list_available_libraries,
    resolve_footprint,
    resolve_part,
    resolve_symbol,
)

if TYPE_CHECKING:
    import pytest


def _make_part(
    lcsc: str = "C25804",
    category: str = "Resistors",
    package: str = "0805",
    mfr_part: str = "RC0805FR-0710KL",
) -> JLCPCBPart:
    return JLCPCBPart(
        lcsc=lcsc,
        mfr="YAGEO",
        mfr_part=mfr_part,
        description="10k resistor",
        package=package,
        category=category,
        subcategory="Chip Resistor",
        solder_joints=2,
        stock=50000,
        price=0.001,
        basic=True,
    )


class TestIsPassive:
    def test_resistor(self) -> None:
        assert _is_passive("Resistors") is True

    def test_capacitor(self) -> None:
        assert _is_passive("Capacitors") is True

    def test_inductor(self) -> None:
        assert _is_passive("Inductors") is True

    def test_ic(self) -> None:
        assert _is_passive("ICs") is False


class TestCategoryToCdferFile:
    def test_resistors(self) -> None:
        assert _category_to_cdfer_file("Resistors") == "JLCPCB-Resistors"

    def test_capacitors(self) -> None:
        assert _category_to_cdfer_file("Capacitors") == "JLCPCB-Capacitors"

    def test_unknown(self) -> None:
        assert _category_to_cdfer_file("UnknownCategory") is None


class TestFootprintForPassive:
    def test_resistor_0805(self) -> None:
        fp = _footprint_for_passive("Resistors", "0805")
        assert fp == "Resistor_SMD:R_0805_2012Metric"

    def test_capacitor_0603(self) -> None:
        fp = _footprint_for_passive("Capacitors", "0603")
        assert fp == "Capacitor_SMD:C_0603_1608Metric"

    def test_unknown_package(self) -> None:
        fp = _footprint_for_passive("Resistors", "9999")
        assert fp is None


class TestResolveSymbol:
    def test_builtin_resistor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patch CDFER dir to not exist.
        monkeypatch.setattr(
            "kicad_pipeline.parts.library_resolver._CDFER_SYMBOL_DIRS", ()
        )
        part = _make_part(category="Resistors")
        sym = resolve_symbol(part)
        assert sym == "Device:R"

    def test_builtin_capacitor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "kicad_pipeline.parts.library_resolver._CDFER_SYMBOL_DIRS", ()
        )
        part = _make_part(category="Capacitors")
        sym = resolve_symbol(part)
        assert sym == "Device:C"

    def test_unknown_category(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "kicad_pipeline.parts.library_resolver._CDFER_SYMBOL_DIRS", ()
        )
        part = _make_part(category="FancyWidgets")
        sym = resolve_symbol(part)
        assert sym is None


class TestResolveFootprint:
    def test_builtin_resistor_footprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "kicad_pipeline.parts.library_resolver._CDFER_FOOTPRINT_DIRS", ()
        )
        part = _make_part(category="Resistors", package="0805")
        fp = resolve_footprint(part)
        assert fp == "Resistor_SMD:R_0805_2012Metric"

    def test_builtin_capacitor_footprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "kicad_pipeline.parts.library_resolver._CDFER_FOOTPRINT_DIRS", ()
        )
        part = _make_part(category="Capacitors", package="0603")
        fp = resolve_footprint(part)
        assert fp == "Capacitor_SMD:C_0603_1608Metric"


class TestResolvePart:
    def test_full_resolve_passive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "kicad_pipeline.parts.library_resolver._CDFER_SYMBOL_DIRS", ()
        )
        monkeypatch.setattr(
            "kicad_pipeline.parts.library_resolver._CDFER_FOOTPRINT_DIRS", ()
        )
        part = _make_part(category="Resistors", package="0805")
        resolved = resolve_part(part)
        assert resolved is not None
        assert resolved.symbol_ref == "Device:R"
        assert resolved.footprint_ref == "Resistor_SMD:R_0805_2012Metric"
        assert resolved.source == "builtin"

    def test_unresolvable_part(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "kicad_pipeline.parts.library_resolver._CDFER_SYMBOL_DIRS", ()
        )
        monkeypatch.setattr(
            "kicad_pipeline.parts.library_resolver._CDFER_FOOTPRINT_DIRS", ()
        )
        part = _make_part(category="FancyWidgets", package="CUSTOM")
        resolved = resolve_part(part)
        assert resolved is None


class TestListAvailableLibraries:
    def test_returns_dict(self) -> None:
        libs = list_available_libraries()
        assert isinstance(libs, dict)
