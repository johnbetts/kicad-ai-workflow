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
