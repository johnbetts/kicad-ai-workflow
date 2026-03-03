"""Tests for kicad_pipeline.schematic.erc."""

from __future__ import annotations

from kicad_pipeline.models.schematic import (
    GlobalLabel,
    Point,
    Schematic,
    SymbolInstance,
)
from kicad_pipeline.schematic.erc import (
    ERCReport,
    ERCSeverity,
    ERCViolation,
    run_erc,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_schematic() -> Schematic:
    """Return a schematic with no elements."""
    return Schematic(
        lib_symbols=(),
        symbols=(),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )


def _make_sym(
    ref: str,
    value: str = "10k",
    lib_id: str = "Device:R",
    x: float = 0.0,
    y: float = 0.0,
) -> SymbolInstance:
    """Build a minimal :class:`SymbolInstance` for testing."""
    return SymbolInstance(
        lib_id=lib_id,
        ref=ref,
        value=value,
        footprint="R_0805",
        position=Point(x, y),
    )


def _make_global_label(text: str, x: float = 0.0, y: float = 0.0) -> GlobalLabel:
    return GlobalLabel(
        text=text,
        shape="input",
        position=Point(x, y),
    )


# ---------------------------------------------------------------------------
# test_erc_empty_schematic_passes
# ---------------------------------------------------------------------------


def test_erc_empty_schematic_passes() -> None:
    """An empty schematic with no symbols should pass ERC."""
    report = run_erc(_empty_schematic())
    assert report.passed
    assert len(report.errors) == 0


# ---------------------------------------------------------------------------
# duplicate_ref
# ---------------------------------------------------------------------------


def test_erc_no_duplicates_passes() -> None:
    """Schematic with unique refs passes the duplicate_ref check."""
    sch = Schematic(
        lib_symbols=(),
        symbols=(_make_sym("R1"), _make_sym("R2")),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )
    report = run_erc(sch)
    dup_errors = [v for v in report.errors if v.rule == "duplicate_ref"]
    assert len(dup_errors) == 0


def test_erc_duplicate_ref_is_error() -> None:
    """Two symbols sharing the same ref → ERROR violation with rule 'duplicate_ref'."""
    sch = Schematic(
        lib_symbols=(),
        symbols=(_make_sym("R1", x=0.0), _make_sym("R1", x=10.0)),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )
    report = run_erc(sch)
    dup_errors = [v for v in report.errors if v.rule == "duplicate_ref"]
    assert len(dup_errors) >= 1
    assert dup_errors[0].ref == "R1"


# ---------------------------------------------------------------------------
# missing_value
# ---------------------------------------------------------------------------


def test_erc_missing_value_is_error() -> None:
    """A symbol with an empty value field → ERROR violation with rule 'missing_value'."""
    sch = Schematic(
        lib_symbols=(),
        symbols=(_make_sym("R1", value=""),),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )
    report = run_erc(sch)
    val_errors = [v for v in report.errors if v.rule == "missing_value"]
    assert len(val_errors) == 1
    assert val_errors[0].ref == "R1"


# ---------------------------------------------------------------------------
# ERCReport properties
# ---------------------------------------------------------------------------


def test_erc_report_passed_property() -> None:
    """passed is True when there are only WARNING-severity violations."""
    warning = ERCViolation(
        severity=ERCSeverity.WARNING,
        rule="some_warning",
        message="Just a warning",
    )
    report = ERCReport(violations=(warning,))
    assert report.passed is True


def test_erc_report_errors_filter() -> None:
    """errors property returns only ERROR-severity violations."""
    error = ERCViolation(severity=ERCSeverity.ERROR, rule="r1", message="e")
    warning = ERCViolation(severity=ERCSeverity.WARNING, rule="r2", message="w")
    report = ERCReport(violations=(error, warning))
    assert report.errors == (error,)
    assert len(report.errors) == 1


def test_erc_report_warnings_filter() -> None:
    """warnings property returns only WARNING-severity violations."""
    error = ERCViolation(severity=ERCSeverity.ERROR, rule="r1", message="e")
    warning = ERCViolation(severity=ERCSeverity.WARNING, rule="r2", message="w")
    report = ERCReport(violations=(error, warning))
    assert report.warnings == (warning,)
    assert len(report.warnings) == 1


# ---------------------------------------------------------------------------
# unmatched_global_label
# ---------------------------------------------------------------------------


def test_erc_unmatched_global_label_warning() -> None:
    """A single GlobalLabel with a unique text → WARNING violation."""
    sch = Schematic(
        lib_symbols=(),
        symbols=(),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(_make_global_label("SPI_CLK"),),
    )
    report = run_erc(sch)
    gl_warnings = [v for v in report.warnings if v.rule == "unmatched_global_label"]
    assert len(gl_warnings) == 1
    assert "SPI_CLK" in gl_warnings[0].message


def test_erc_matched_global_labels_pass() -> None:
    """Two GlobalLabels with the same text → no unmatched_global_label violation."""
    sch = Schematic(
        lib_symbols=(),
        symbols=(),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(
            _make_global_label("SPI_CLK", x=0.0),
            _make_global_label("SPI_CLK", x=100.0),
        ),
    )
    report = run_erc(sch)
    gl_violations = [v for v in report.violations if v.rule == "unmatched_global_label"]
    assert len(gl_violations) == 0
