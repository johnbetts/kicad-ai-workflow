"""Tests for the schematic-PCB consistency validator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.validation.consistency import (
    ConsistencyReport,
    PCBComponent,
    SchematicComponent,
    check_consistency,
    check_requirements_hash,
    compute_requirements_hash,
    consistency_report_to_text,
    extract_pcb_components,
    extract_schematic_components,
    extract_schematic_components_recursive,
    footprints_match,
    normalize_footprint,
)
from kicad_pipeline.validation.drc import Severity

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers — build minimal S-expression strings
# ---------------------------------------------------------------------------

_FX = '(effects (font (size 1 1)))'
_FX_HIDE = '(effects (font (size 1 1)) hide)'


def _sch_symbol(
    lib_id: str, ref: str, value: str, footprint: str,
) -> str:
    return (
        f'  (symbol (lib_id "{lib_id}")\n'
        f'    (at 100 50 0)\n'
        f'    (property "Reference" "{ref}" (at 0 0 0) {_FX})\n'
        f'    (property "Value" "{value}" (at 0 0 0) {_FX})\n'
        f'    (property "Footprint" "{footprint}" (at 0 0 0) {_FX_HIDE})\n'
        f'  )\n'
    )


def _pcb_fp(lib_id: str, ref: str, value: str) -> str:
    return (
        f'  (footprint "{lib_id}"\n'
        f'    (at 100 50)\n'
        f'    (property "Reference" "{ref}" (at 0 0) (layer "F.SilkS") {_FX})\n'
        f'    (property "Value" "{value}" (at 0 0) (layer "F.Fab") {_FX})\n'
        f'  )\n'
    )


def _wrap_sch(*symbols: str) -> str:
    body = "".join(symbols)
    return (
        '(kicad_sch (version 20250114) (generator "kicad_pipeline")\n'
        f'{body})\n'
    )


def _wrap_pcb(*fps: str) -> str:
    body = "".join(fps)
    return (
        '(kicad_pcb (version 20241229) (generator "kicad_pipeline")\n'
        f'{body})\n'
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_SCH = _wrap_sch(
    _sch_symbol("Device:R", "R1", "10k", "Resistor_SMD:R_0805_2012Metric"),
    _sch_symbol("Device:C", "C1", "100nF", "Capacitor_SMD:C_0805_2012Metric"),
)

MINIMAL_SCH_WITH_POWER = _wrap_sch(
    _sch_symbol("Device:R", "R1", "10k", "Resistor_SMD:R_0805_2012Metric"),
    _sch_symbol("power:GND", "#PWR01", "GND", ""),
)

MINIMAL_PCB = _wrap_pcb(
    _pcb_fp("Resistor_SMD:R_0805_2012Metric", "R1", "10k"),
    _pcb_fp("Capacitor_SMD:C_0805_2012Metric", "C1", "100nF"),
)

MINIMAL_PCB_EXTRA = _wrap_pcb(
    _pcb_fp("Resistor_SMD:R_0805_2012Metric", "R1", "10k"),
    _pcb_fp("Capacitor_SMD:C_0805_2012Metric", "C1", "100nF"),
    _pcb_fp("LED_SMD:LED_0805_2012Metric", "D1", "RED"),
)

SUB_SHEET_SCH = _wrap_sch(
    _sch_symbol("Device:LED", "D1", "RED", "LED_SMD:LED_0805_2012Metric"),
)

ROOT_WITH_SUBSHEET = (
    '(kicad_sch (version 20250114) (generator "kicad_pipeline")\n'
    + _sch_symbol("Device:R", "R1", "10k", "Resistor_SMD:R_0805_2012Metric")
    + '  (sheet (at 150 50) (size 20 15)\n'
    '    (property "Sheetname" "LEDs" (at 0 0 0) '
    + _FX
    + ")\n"
    '    (property "Sheetfile" "leds.kicad_sch" (at 0 0 0) '
    + _FX
    + ")\n"
    "  )\n"
    ")\n"
)


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------


def test_extract_schematic_components_simple(tmp_path: Path) -> None:
    """Two components from a flat schematic."""
    sch = tmp_path / "test.kicad_sch"
    sch.write_text(MINIMAL_SCH, encoding="utf-8")

    comps = extract_schematic_components(sch)
    assert len(comps) == 2
    refs = {c.ref for c in comps}
    assert refs == {"R1", "C1"}
    r1 = next(c for c in comps if c.ref == "R1")
    assert r1.value == "10k"
    assert r1.footprint == "Resistor_SMD:R_0805_2012Metric"
    assert r1.source_file == "test.kicad_sch"


def test_extract_schematic_skips_power_symbols(tmp_path: Path) -> None:
    """Power symbols (power:GND, #PWR refs) are excluded."""
    sch = tmp_path / "test.kicad_sch"
    sch.write_text(MINIMAL_SCH_WITH_POWER, encoding="utf-8")

    comps = extract_schematic_components(sch)
    assert len(comps) == 1
    assert comps[0].ref == "R1"


def test_extract_pcb_components(tmp_path: Path) -> None:
    """Two footprints from a PCB file."""
    pcb = tmp_path / "test.kicad_pcb"
    pcb.write_text(MINIMAL_PCB, encoding="utf-8")

    comps = extract_pcb_components(pcb)
    assert len(comps) == 2
    refs = {c.ref for c in comps}
    assert refs == {"R1", "C1"}
    r1 = next(c for c in comps if c.ref == "R1")
    assert r1.value == "10k"
    assert r1.lib_id == "Resistor_SMD:R_0805_2012Metric"


def test_extract_schematic_hierarchical(tmp_path: Path) -> None:
    """Sub-sheet components are included via recursive extraction."""
    root = tmp_path / "root.kicad_sch"
    root.write_text(ROOT_WITH_SUBSHEET, encoding="utf-8")
    sub = tmp_path / "leds.kicad_sch"
    sub.write_text(SUB_SHEET_SCH, encoding="utf-8")

    comps = extract_schematic_components_recursive(root)
    refs = {c.ref for c in comps}
    assert refs == {"R1", "D1"}


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


def test_normalize_footprint_strips_prefix() -> None:
    result = normalize_footprint("Resistor_SMD:R_0805_2012Metric")
    assert result == "R_0805_2012Metric"
    assert normalize_footprint("R_0805_2012Metric") == "R_0805_2012Metric"


def test_footprints_match_exact() -> None:
    assert footprints_match(
        "Resistor_SMD:R_0805_2012Metric",
        "Resistor_SMD:R_0805_2012Metric",
    )


def test_footprints_match_prefix() -> None:
    """One footprint name is a prefix of the other."""
    assert footprints_match("R_0805", "R_0805_2012Metric")
    assert footprints_match(
        "Resistor_SMD:R_0805_2012Metric", "R_0805",
    )


def test_footprints_no_match() -> None:
    assert not footprints_match(
        "Resistor_SMD:R_0805_2012Metric",
        "Resistor_SMD:R_0603_1608Metric",
    )


# ---------------------------------------------------------------------------
# Consistency check tests
# ---------------------------------------------------------------------------


def test_consistency_all_match(tmp_path: Path) -> None:
    """No violations when schematic and PCB match."""
    sch = tmp_path / "test.kicad_sch"
    sch.write_text(MINIMAL_SCH, encoding="utf-8")
    pcb = tmp_path / "test.kicad_pcb"
    pcb.write_text(MINIMAL_PCB, encoding="utf-8")

    report = check_consistency(sch, pcb)
    assert report.passed
    assert len(report.violations) == 0
    assert set(report.schematic_refs) == {"R1", "C1"}
    assert set(report.pcb_refs) == {"R1", "C1"}


def test_consistency_missing_in_pcb(tmp_path: Path) -> None:
    """Component in schematic but not in PCB produces an ERROR."""
    sch = tmp_path / "test.kicad_sch"
    sch.write_text(MINIMAL_SCH, encoding="utf-8")

    pcb_only_r1 = _wrap_pcb(
        _pcb_fp("Resistor_SMD:R_0805_2012Metric", "R1", "10k"),
    )
    pcb = tmp_path / "test.kicad_pcb"
    pcb.write_text(pcb_only_r1, encoding="utf-8")

    report = check_consistency(sch, pcb)
    assert not report.passed
    assert len(report.errors) == 1
    assert report.errors[0].rule == "consistency_missing_in_pcb"
    assert "C1" in report.errors[0].message


def test_consistency_extra_in_pcb(tmp_path: Path) -> None:
    """Component in PCB but not in schematic produces an ERROR."""
    sch = tmp_path / "test.kicad_sch"
    sch.write_text(MINIMAL_SCH, encoding="utf-8")
    pcb = tmp_path / "test.kicad_pcb"
    pcb.write_text(MINIMAL_PCB_EXTRA, encoding="utf-8")

    report = check_consistency(sch, pcb)
    assert not report.passed
    missing = [
        v for v in report.errors
        if v.rule == "consistency_missing_in_schematic"
    ]
    assert len(missing) == 1
    assert "D1" in missing[0].message


def test_consistency_footprint_mismatch(tmp_path: Path) -> None:
    """Same ref with different footprint produces an ERROR."""
    sch = tmp_path / "test.kicad_sch"
    sch.write_text(MINIMAL_SCH, encoding="utf-8")

    pcb_text = _wrap_pcb(
        _pcb_fp("Resistor_SMD:R_0603_1608Metric", "R1", "10k"),
        _pcb_fp("Capacitor_SMD:C_0805_2012Metric", "C1", "100nF"),
    )
    pcb = tmp_path / "test.kicad_pcb"
    pcb.write_text(pcb_text, encoding="utf-8")

    report = check_consistency(sch, pcb)
    assert not report.passed
    fp_mm = [
        v for v in report.errors
        if v.rule == "consistency_footprint_mismatch"
    ]
    assert len(fp_mm) == 1
    assert "R1" in fp_mm[0].message
    assert "0805" in fp_mm[0].message
    assert "0603" in fp_mm[0].message


def test_consistency_value_mismatch(tmp_path: Path) -> None:
    """Same ref with different value produces a WARNING only."""
    sch = tmp_path / "test.kicad_sch"
    sch.write_text(MINIMAL_SCH, encoding="utf-8")

    pcb_text = _wrap_pcb(
        _pcb_fp("Resistor_SMD:R_0805_2012Metric", "R1", "4.7k"),
        _pcb_fp("Capacitor_SMD:C_0805_2012Metric", "C1", "100nF"),
    )
    pcb = tmp_path / "test.kicad_pcb"
    pcb.write_text(pcb_text, encoding="utf-8")

    report = check_consistency(sch, pcb)
    assert report.passed  # WARNING only
    assert len(report.warnings) == 1
    assert report.warnings[0].rule == "consistency_value_mismatch"
    assert "R1" in report.warnings[0].message


# ---------------------------------------------------------------------------
# Requirements hash tests
# ---------------------------------------------------------------------------


def test_requirements_hash_deterministic(tmp_path: Path) -> None:
    """Same JSON content produces same hash regardless of formatting."""
    data = {"components": [{"ref": "R1", "value": "10k"}], "nets": []}

    p1 = tmp_path / "req1.json"
    p1.write_text(json.dumps(data, indent=2), encoding="utf-8")

    p2 = tmp_path / "req2.json"
    p2.write_text(
        json.dumps(data, indent=4, sort_keys=True), encoding="utf-8",
    )

    assert compute_requirements_hash(p1) == compute_requirements_hash(p2)


def test_requirements_hash_changed(tmp_path: Path) -> None:
    """Changed content produces different hash and a warning."""
    data1 = {"components": [{"ref": "R1", "value": "10k"}]}
    data2 = {"components": [{"ref": "R1", "value": "4.7k"}]}

    p1 = tmp_path / "req.json"
    p1.write_text(json.dumps(data1), encoding="utf-8")
    stored_hash = compute_requirements_hash(p1)

    p1.write_text(json.dumps(data2), encoding="utf-8")
    violation = check_requirements_hash(stored_hash, p1)

    assert violation is not None
    assert violation.severity == Severity.WARNING
    assert "changed" in violation.message


def test_requirements_hash_unchanged(tmp_path: Path) -> None:
    """Unchanged content returns None (no violation)."""
    data = {"components": [{"ref": "R1", "value": "10k"}]}
    p = tmp_path / "req.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    stored_hash = compute_requirements_hash(p)

    assert check_requirements_hash(stored_hash, p) is None


# ---------------------------------------------------------------------------
# Report text output
# ---------------------------------------------------------------------------


def test_consistency_report_to_text(tmp_path: Path) -> None:
    """Report text includes summary and violations."""
    sch = tmp_path / "test.kicad_sch"
    sch.write_text(MINIMAL_SCH, encoding="utf-8")
    pcb = tmp_path / "test.kicad_pcb"
    pcb.write_text(MINIMAL_PCB_EXTRA, encoding="utf-8")

    report = check_consistency(sch, pcb)
    text = consistency_report_to_text(report)

    assert "FAIL" in text
    assert "Errors: 1" in text
    assert "D1" in text


def test_consistency_report_pass_text(tmp_path: Path) -> None:
    """Passing report shows PASS."""
    sch = tmp_path / "test.kicad_sch"
    sch.write_text(MINIMAL_SCH, encoding="utf-8")
    pcb = tmp_path / "test.kicad_pcb"
    pcb.write_text(MINIMAL_PCB, encoding="utf-8")

    report = check_consistency(sch, pcb)
    text = consistency_report_to_text(report)
    assert "PASS" in text
    assert "Errors: 0" in text


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


def test_consistency_report_properties() -> None:
    """ConsistencyReport properties filter by severity."""
    from kicad_pipeline.validation.drc import DRCViolation, Severity

    violations = (
        DRCViolation(rule="a", message="err", severity=Severity.ERROR),
        DRCViolation(
            rule="b", message="warn", severity=Severity.WARNING,
        ),
        DRCViolation(rule="c", message="err2", severity=Severity.ERROR),
    )
    report = ConsistencyReport(
        violations=violations,
        schematic_refs=("R1",),
        pcb_refs=("R1",),
    )
    assert len(report.errors) == 2
    assert len(report.warnings) == 1
    assert not report.passed


def test_schematic_component_frozen() -> None:
    comp = SchematicComponent(
        ref="R1", value="10k", footprint="R_0805", source_file="a.sch",
    )
    with pytest.raises(AttributeError):
        comp.ref = "R2"  # type: ignore[misc]


def test_pcb_component_frozen() -> None:
    comp = PCBComponent(ref="R1", value="10k", lib_id="R_0805")
    with pytest.raises(AttributeError):
        comp.ref = "R2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edge cases: normalize_footprint
# ---------------------------------------------------------------------------


def test_normalize_footprint_empty() -> None:
    """Empty string stays empty."""
    assert normalize_footprint("") == ""


def test_normalize_footprint_no_colon() -> None:
    """No colon means no prefix to strip."""
    assert normalize_footprint("R_0805_2012Metric") == "R_0805_2012Metric"


def test_normalize_footprint_multiple_colons() -> None:
    """Only the first colon is used as prefix separator."""
    assert normalize_footprint("Lib:Sub:Pkg") == "Sub:Pkg"


def test_footprints_match_both_empty() -> None:
    """Two empty strings match."""
    assert footprints_match("", "")


def test_footprints_match_one_empty() -> None:
    """Empty string is a prefix of anything."""
    assert footprints_match("", "R_0805_2012Metric")
    assert footprints_match("R_0805_2012Metric", "")


# ---------------------------------------------------------------------------
# Edge cases: ConsistencyReport
# ---------------------------------------------------------------------------


def test_consistency_report_empty_violations_passes() -> None:
    """A report with zero violations should pass."""
    report = ConsistencyReport(violations=(), schematic_refs=(), pcb_refs=())
    assert report.passed is True
    assert len(report.errors) == 0
    assert len(report.warnings) == 0


# ---------------------------------------------------------------------------
# Edge cases: extraction with empty files
# ---------------------------------------------------------------------------


def test_extract_schematic_empty_file(tmp_path: Path) -> None:
    """An empty-body schematic returns no components."""
    sch = tmp_path / "empty.kicad_sch"
    sch.write_text(_wrap_sch(), encoding="utf-8")
    comps = extract_schematic_components(sch)
    assert comps == ()


def test_extract_pcb_empty_file(tmp_path: Path) -> None:
    """An empty-body PCB returns no components."""
    pcb = tmp_path / "empty.kicad_pcb"
    pcb.write_text(_wrap_pcb(), encoding="utf-8")
    comps = extract_pcb_components(pcb)
    assert comps == ()


def test_extract_schematic_recursive_no_subsheets(tmp_path: Path) -> None:
    """Recursive extraction of a flat schematic returns same as single."""
    sch = tmp_path / "flat.kicad_sch"
    sch.write_text(MINIMAL_SCH, encoding="utf-8")
    comps = extract_schematic_components_recursive(sch)
    assert len(comps) == 2


def test_extract_schematic_recursive_missing_subsheet(tmp_path: Path) -> None:
    """Missing sub-sheet file is silently skipped."""
    root = tmp_path / "root.kicad_sch"
    root.write_text(ROOT_WITH_SUBSHEET, encoding="utf-8")
    # Deliberately not creating leds.kicad_sch
    comps = extract_schematic_components_recursive(root)
    # Only root-level component R1
    assert len(comps) == 1
    assert comps[0].ref == "R1"


# ---------------------------------------------------------------------------
# Edge cases: requirements hash
# ---------------------------------------------------------------------------


def test_requirements_hash_different_order_same_hash(tmp_path: Path) -> None:
    """Key ordering difference should NOT affect the hash."""
    p1 = tmp_path / "req1.json"
    p1.write_text('{"b": 2, "a": 1}', encoding="utf-8")
    p2 = tmp_path / "req2.json"
    p2.write_text('{"a": 1, "b": 2}', encoding="utf-8")
    assert compute_requirements_hash(p1) == compute_requirements_hash(p2)


# ---------------------------------------------------------------------------
# Edge cases: mechanical values in PCB
# ---------------------------------------------------------------------------


def test_consistency_mechanical_value_not_flagged(tmp_path: Path) -> None:
    """PCB-only components with 'MountingHole' value should not be errors."""
    sch = tmp_path / "test.kicad_sch"
    sch.write_text(
        _wrap_sch(
            _sch_symbol("Device:R", "R1", "10k", "Resistor_SMD:R_0805_2012Metric"),
        ),
        encoding="utf-8",
    )
    pcb = tmp_path / "test.kicad_pcb"
    pcb.write_text(
        _wrap_pcb(
            _pcb_fp("Resistor_SMD:R_0805_2012Metric", "R1", "10k"),
            _pcb_fp("MountingHole:MountingHole_3.2mm", "H1", "MountingHole"),
        ),
        encoding="utf-8",
    )
    report = check_consistency(sch, pcb)
    # H1 with MountingHole value should be excluded from "missing in schematic" errors
    assert report.passed is True
