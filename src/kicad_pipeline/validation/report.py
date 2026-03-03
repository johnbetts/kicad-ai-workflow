"""Unified validation report aggregating all sub-system checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kicad_pipeline.validation.drc import DRCReport, Severity
from kicad_pipeline.validation.electrical import ElectricalReport  # noqa: TC001
from kicad_pipeline.validation.manufacturing import ManufacturingReport  # noqa: TC001
from kicad_pipeline.validation.signal_integrity import SIReport  # noqa: TC001
from kicad_pipeline.validation.thermal import ThermalReport  # noqa: TC001

# ---------------------------------------------------------------------------
# Overall status
# ---------------------------------------------------------------------------


class OverallStatus(Enum):
    """Overall validation result status."""

    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReport:
    """Aggregated validation report from all check modules."""

    overall_status: OverallStatus
    drc: DRCReport
    electrical: ElectricalReport
    manufacturing: ManufacturingReport
    thermal: ThermalReport
    signal_integrity: SIReport
    total_errors: int
    total_warnings: int

    @property
    def passed(self) -> bool:
        """True if overall status is not FAIL."""
        return self.overall_status != OverallStatus.FAIL


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_validation_report(
    drc: DRCReport,
    electrical: ElectricalReport,
    manufacturing: ManufacturingReport,
    thermal: ThermalReport,
    si: SIReport,
) -> ValidationReport:
    """Build a :class:`ValidationReport` from all sub-reports.

    Args:
        drc: DRC check results.
        electrical: Electrical check results.
        manufacturing: Manufacturing constraint check results.
        thermal: Thermal analysis results.
        si: Signal integrity check results.

    Returns:
        A :class:`ValidationReport` with computed totals and overall status.
    """
    # Count errors
    thermal_errors = sum(
        1 for v in thermal.violations if v.severity == Severity.ERROR
    )
    total_errors = (
        len(drc.errors)
        + len(electrical.errors)
        + len(manufacturing.errors)
        + thermal_errors
        + len(si.errors)
    )

    # Count warnings
    drc_warnings = sum(1 for v in drc.violations if v.severity == Severity.WARNING)
    electrical_warnings = sum(
        1 for v in electrical.violations if v.severity == Severity.WARNING
    )
    manufacturing_warnings = sum(
        1 for v in manufacturing.violations if v.severity == Severity.WARNING
    )
    thermal_warnings = sum(
        1 for v in thermal.violations if v.severity == Severity.WARNING
    )
    si_warnings = sum(1 for v in si.violations if v.severity == Severity.WARNING)
    total_warnings = (
        drc_warnings
        + electrical_warnings
        + manufacturing_warnings
        + thermal_warnings
        + si_warnings
    )

    if total_errors > 0:
        status = OverallStatus.FAIL
    elif total_warnings > 0:
        status = OverallStatus.PASS_WITH_WARNINGS
    else:
        status = OverallStatus.PASS

    return ValidationReport(
        overall_status=status,
        drc=drc,
        electrical=electrical,
        manufacturing=manufacturing,
        thermal=thermal,
        signal_integrity=si,
        total_errors=total_errors,
        total_warnings=total_warnings,
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _violations_to_list(
    violations: tuple[object, ...],
) -> list[dict[str, str]]:
    """Convert a tuple of violation objects to a list of dicts.

    Each object must have ``rule``, ``message``, and ``severity`` attributes.
    """
    result = []
    for v in violations:
        rule = getattr(v, "rule", "")
        message = getattr(v, "message", "")
        sev = getattr(v, "severity", None)
        severity_str = sev.value if isinstance(sev, Severity) else str(sev)
        result.append({"rule": str(rule), "message": str(message), "severity": severity_str})
    return result


def report_to_dict(report: ValidationReport) -> dict[str, object]:
    """Convert *report* to a JSON-serialisable dict.

    Args:
        report: The validation report to serialise.

    Returns:
        A dict with keys ``overall_status``, ``total_errors``,
        ``total_warnings``, and one key per sub-report.
    """
    drc_warnings = sum(
        1 for v in report.drc.violations if v.severity == Severity.WARNING
    )
    electrical_warnings = sum(
        1 for v in report.electrical.violations if v.severity == Severity.WARNING
    )
    manufacturing_warnings = sum(
        1 for v in report.manufacturing.violations if v.severity == Severity.WARNING
    )
    thermal_warnings = sum(
        1 for v in report.thermal.violations if v.severity == Severity.WARNING
    )
    si_warnings = sum(
        1 for v in report.signal_integrity.violations if v.severity == Severity.WARNING
    )

    return {
        "overall_status": report.overall_status.value,
        "total_errors": report.total_errors,
        "total_warnings": report.total_warnings,
        "drc": {
            "passed": report.drc.passed,
            "error_count": len(report.drc.errors),
            "warning_count": drc_warnings,
            "violations": _violations_to_list(report.drc.violations),
        },
        "electrical": {
            "passed": report.electrical.passed,
            "error_count": len(report.electrical.errors),
            "warning_count": electrical_warnings,
            "violations": _violations_to_list(report.electrical.violations),
        },
        "manufacturing": {
            "passed": report.manufacturing.passed,
            "error_count": len(report.manufacturing.errors),
            "warning_count": manufacturing_warnings,
            "violations": _violations_to_list(report.manufacturing.violations),
        },
        "thermal": {
            "passed": report.thermal.passed,
            "error_count": sum(
                1 for v in report.thermal.violations if v.severity == Severity.ERROR
            ),
            "warning_count": thermal_warnings,
            "violations": _violations_to_list(report.thermal.violations),
        },
        "signal_integrity": {
            "passed": report.signal_integrity.passed,
            "error_count": len(report.signal_integrity.errors),
            "warning_count": si_warnings,
            "violations": _violations_to_list(report.signal_integrity.violations),
        },
    }


def format_report_markdown(report: ValidationReport) -> str:
    """Format *report* as a Markdown string.

    Args:
        report: The validation report to format.

    Returns:
        A Markdown-formatted string summarising the report.
    """
    lines: list[str] = [
        "# Validation Report",
        "",
        f"**Status**: {report.overall_status.value}",
        f"**Total Errors**: {report.total_errors}",
        f"**Total Warnings**: {report.total_warnings}",
        "",
    ]

    def _section(
        title: str,
        passed: bool,
        errors: int,
        warnings: int,
        violations: tuple[object, ...],
    ) -> None:
        lines.append(f"## {title}")
        if passed and warnings == 0:
            lines.append("- PASS")
        else:
            lines.append(f"- {errors} errors, {warnings} warnings")
        for v in violations:
            rule = getattr(v, "rule", "")
            message = getattr(v, "message", "")
            sev = getattr(v, "severity", None)
            severity_str = sev.value if isinstance(sev, Severity) else str(sev)
            lines.append(f"  - [{severity_str.upper()}] {rule}: {message}")
        lines.append("")

    drc_warnings = sum(
        1 for v in report.drc.violations if v.severity == Severity.WARNING
    )
    electrical_warnings = sum(
        1 for v in report.electrical.violations if v.severity == Severity.WARNING
    )
    manufacturing_warnings = sum(
        1 for v in report.manufacturing.violations if v.severity == Severity.WARNING
    )
    thermal_warnings = sum(
        1 for v in report.thermal.violations if v.severity == Severity.WARNING
    )
    si_warnings = sum(
        1 for v in report.signal_integrity.violations if v.severity == Severity.WARNING
    )

    _section(
        "DRC",
        report.drc.passed,
        len(report.drc.errors),
        drc_warnings,
        report.drc.violations,
    )
    _section(
        "Electrical",
        report.electrical.passed,
        len(report.electrical.errors),
        electrical_warnings,
        report.electrical.violations,
    )
    _section(
        "Manufacturing",
        report.manufacturing.passed,
        len(report.manufacturing.errors),
        manufacturing_warnings,
        report.manufacturing.violations,
    )
    _section(
        "Thermal",
        report.thermal.passed,
        sum(1 for v in report.thermal.violations if v.severity == Severity.ERROR),
        thermal_warnings,
        report.thermal.violations,
    )
    _section(
        "Signal Integrity",
        report.signal_integrity.passed,
        len(report.signal_integrity.errors),
        si_warnings,
        report.signal_integrity.violations,
    )

    return "\n".join(lines)
