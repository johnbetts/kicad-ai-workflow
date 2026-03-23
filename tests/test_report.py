"""Tests for kicad_pipeline.validation.report."""

from __future__ import annotations

import pytest

from kicad_pipeline.validation.drc import DRCReport, DRCViolation, Severity
from kicad_pipeline.validation.electrical import ElectricalReport
from kicad_pipeline.validation.manufacturing import (
    ManufacturingReport,
    ManufacturingViolation,
)
from kicad_pipeline.validation.report import (
    OverallStatus,
    build_validation_report,
    format_report_markdown,
    report_to_dict,
)
from kicad_pipeline.validation.signal_integrity import SIReport, SIViolation
from kicad_pipeline.validation.thermal import ThermalReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_drc() -> DRCReport:
    return DRCReport(violations=())


def _clean_electrical() -> ElectricalReport:
    return ElectricalReport(violations=())


def _clean_manufacturing() -> ManufacturingReport:
    return ManufacturingReport(violations=())


def _clean_thermal() -> ThermalReport:
    return ThermalReport(component_thermals=(), violations=())


def _clean_si() -> SIReport:
    return SIReport(violations=())


def _drc_with_error() -> DRCReport:
    return DRCReport(
        violations=(
            DRCViolation(
                rule="clearance",
                message="Clearance violation",
                severity=Severity.ERROR,
            ),
        )
    )


def _manufacturing_with_warning() -> ManufacturingReport:
    return ManufacturingReport(
        violations=(
            ManufacturingViolation(
                rule="lcsc_check",
                message="Component R1 has no LCSC part number",
                severity=Severity.WARNING,
            ),
        )
    )


def _si_with_warning() -> SIReport:
    return SIReport(
        violations=(
            SIViolation(
                rule="trace_length_check",
                message="Long SPI trace: 150mm",
                severity=Severity.WARNING,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_report_all_pass() -> None:
    """All clean sub-reports should produce PASS status."""
    report = build_validation_report(
        _clean_drc(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _clean_si(),
    )
    assert report.overall_status == OverallStatus.PASS
    assert report.total_errors == 0
    assert report.total_warnings == 0
    assert report.passed


def test_build_report_with_errors() -> None:
    """A DRC error should produce FAIL status."""
    report = build_validation_report(
        _drc_with_error(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _clean_si(),
    )
    assert report.overall_status == OverallStatus.FAIL
    assert report.total_errors == 1
    assert not report.passed


def test_build_report_warnings_only() -> None:
    """Warnings with no errors should produce PASS_WITH_WARNINGS."""
    report = build_validation_report(
        _clean_drc(),
        _clean_electrical(),
        _manufacturing_with_warning(),
        _clean_thermal(),
        _clean_si(),
    )
    assert report.overall_status == OverallStatus.PASS_WITH_WARNINGS
    assert report.total_errors == 0
    assert report.total_warnings == 1
    assert report.passed


def test_validation_report_frozen() -> None:
    """ValidationReport should be immutable."""
    report = build_validation_report(
        _clean_drc(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _clean_si(),
    )
    with pytest.raises(AttributeError):
        report.total_errors = 99  # type: ignore[misc]


def test_overall_status_pass() -> None:
    """Zero errors and zero warnings should yield PASS."""
    report = build_validation_report(
        _clean_drc(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _clean_si(),
    )
    assert report.overall_status == OverallStatus.PASS


def test_overall_status_fail() -> None:
    """Presence of errors should yield FAIL regardless of warnings."""
    report = build_validation_report(
        _drc_with_error(),
        _clean_electrical(),
        _manufacturing_with_warning(),
        _clean_thermal(),
        _clean_si(),
    )
    assert report.overall_status == OverallStatus.FAIL


def test_overall_status_warnings() -> None:
    """Warnings only (no errors) should yield PASS_WITH_WARNINGS."""
    report = build_validation_report(
        _clean_drc(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _si_with_warning(),
    )
    assert report.overall_status == OverallStatus.PASS_WITH_WARNINGS
    assert report.total_warnings == 1


def test_report_to_dict_keys() -> None:
    """report_to_dict output should contain all required top-level keys."""
    report = build_validation_report(
        _clean_drc(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _clean_si(),
    )
    d = report_to_dict(report)
    assert "overall_status" in d
    assert "total_errors" in d
    assert "total_warnings" in d
    assert "drc" in d
    assert "electrical" in d
    assert "manufacturing" in d
    assert "thermal" in d
    assert "signal_integrity" in d


def test_report_to_dict_sub_report_keys() -> None:
    """Each sub-report dict should have passed, error_count, warning_count, violations."""
    report = build_validation_report(
        _clean_drc(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _clean_si(),
    )
    d = report_to_dict(report)
    for key in ("drc", "electrical", "manufacturing", "thermal", "signal_integrity"):
        sub = d[key]
        assert isinstance(sub, dict)
        assert "passed" in sub
        assert "error_count" in sub
        assert "warning_count" in sub
        assert "violations" in sub


def test_format_report_markdown_contains_status() -> None:
    """Formatted markdown should contain 'Status'."""
    report = build_validation_report(
        _clean_drc(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _clean_si(),
    )
    md = format_report_markdown(report)
    assert "Status" in md


def test_format_report_markdown_contains_sections() -> None:
    """Formatted markdown should contain '## DRC' and '## Electrical' sections."""
    report = build_validation_report(
        _clean_drc(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _clean_si(),
    )
    md = format_report_markdown(report)
    assert "## DRC" in md
    assert "## Electrical" in md


# ---------------------------------------------------------------------------
# Additional: report_to_dict with violations
# ---------------------------------------------------------------------------


def test_report_to_dict_with_violations() -> None:
    """Violations should appear in the serialized dict."""
    report = build_validation_report(
        _drc_with_error(),
        _clean_electrical(),
        _manufacturing_with_warning(),
        _clean_thermal(),
        _si_with_warning(),
    )
    d = report_to_dict(report)
    assert d["overall_status"] == "FAIL"
    assert d["total_errors"] == 1
    assert d["total_warnings"] == 2

    drc_sub = d["drc"]
    assert isinstance(drc_sub, dict)
    assert drc_sub["error_count"] == 1
    assert len(drc_sub["violations"]) == 1
    assert drc_sub["violations"][0]["severity"] == "error"

    mfg_sub = d["manufacturing"]
    assert isinstance(mfg_sub, dict)
    assert mfg_sub["warning_count"] == 1


def test_report_to_dict_all_pass_zeros() -> None:
    """All-pass report has zero counts everywhere."""
    report = build_validation_report(
        _clean_drc(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _clean_si(),
    )
    d = report_to_dict(report)
    assert d["total_errors"] == 0
    assert d["total_warnings"] == 0
    for key in ("drc", "electrical", "manufacturing", "thermal", "signal_integrity"):
        sub = d[key]
        assert isinstance(sub, dict)
        assert sub["error_count"] == 0
        assert sub["warning_count"] == 0
        assert sub["violations"] == []


# ---------------------------------------------------------------------------
# Additional: format_report_markdown with violations
# ---------------------------------------------------------------------------


def test_format_report_markdown_with_violations() -> None:
    """Violations in a sub-report should appear in the markdown."""
    report = build_validation_report(
        _drc_with_error(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _clean_si(),
    )
    md = format_report_markdown(report)
    assert "FAIL" in md
    assert "clearance" in md.lower()
    assert "[ERROR]" in md


def test_format_report_markdown_pass_section() -> None:
    """All-pass sub-report should show '- PASS'."""
    report = build_validation_report(
        _clean_drc(),
        _clean_electrical(),
        _clean_manufacturing(),
        _clean_thermal(),
        _clean_si(),
    )
    md = format_report_markdown(report)
    assert "- PASS" in md


# ---------------------------------------------------------------------------
# Additional: multiple error sources
# ---------------------------------------------------------------------------


def test_build_report_multiple_error_sources() -> None:
    """Errors from multiple sub-reports should sum correctly."""
    drc_err = DRCReport(
        violations=(
            DRCViolation(rule="a", message="a", severity=Severity.ERROR),
            DRCViolation(rule="b", message="b", severity=Severity.ERROR),
        ),
    )
    elec_err = ElectricalReport(
        violations=(
            DRCViolation(rule="c", message="c", severity=Severity.ERROR),
        ),
    )
    report = build_validation_report(
        drc_err, elec_err, _clean_manufacturing(), _clean_thermal(), _clean_si(),
    )
    assert report.total_errors == 3
    assert report.overall_status == OverallStatus.FAIL


def test_build_report_thermal_error_counted() -> None:
    """Thermal ERROR violations should increment total_errors."""
    thermal = ThermalReport(
        component_thermals=(),
        violations=(
            DRCViolation(rule="thermal", message="hot", severity=Severity.ERROR),
        ),
    )
    report = build_validation_report(
        _clean_drc(), _clean_electrical(), _clean_manufacturing(),
        thermal, _clean_si(),
    )
    assert report.total_errors == 1
    assert report.overall_status == OverallStatus.FAIL


def test_validation_report_passed_property() -> None:
    """passed is True for PASS and PASS_WITH_WARNINGS, False for FAIL."""
    pass_report = build_validation_report(
        _clean_drc(), _clean_electrical(), _clean_manufacturing(),
        _clean_thermal(), _clean_si(),
    )
    assert pass_report.passed is True

    warn_report = build_validation_report(
        _clean_drc(), _clean_electrical(), _manufacturing_with_warning(),
        _clean_thermal(), _clean_si(),
    )
    assert warn_report.passed is True

    fail_report = build_validation_report(
        _drc_with_error(), _clean_electrical(), _clean_manufacturing(),
        _clean_thermal(), _clean_si(),
    )
    assert fail_report.passed is False
