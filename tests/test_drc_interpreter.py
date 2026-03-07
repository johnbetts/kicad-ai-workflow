"""Tests for DRC interpreter module."""

from __future__ import annotations

from kicad_pipeline.validation.drc_interpreter import (
    DRCSuggestion,
    interpret_violations,
    summarize_suggestions,
)
from kicad_pipeline.validation.kicad_drc import (
    DRCItem,
    DRCReport,
    DRCViolation,
)


def _make_violation(
    vtype: str = "clearance",
    severity: str = "error",
    desc: str = "test violation",
    excluded: bool = False,
) -> DRCViolation:
    return DRCViolation(
        type=vtype,
        severity=severity,
        description=desc,
        items=(
            DRCItem(description="item1", ref="ref-1"),
        ),
        excluded=excluded,
    )


def _make_report(
    violations: tuple[DRCViolation, ...] = (),
    unconnected: tuple[DRCViolation, ...] = (),
) -> DRCReport:
    return DRCReport(
        violations=violations,
        unconnected=unconnected,
    )


class TestInterpretViolations:
    def test_auto_fixable_silk(self) -> None:
        v = _make_violation("silk_over_copper", "warning", "Silk on pad")
        report = _make_report(violations=(v,))
        suggestions = interpret_violations(report)
        assert len(suggestions) == 1
        assert suggestions[0].auto_fixable is True
        assert suggestions[0].violation_type == "silk_over_copper"

    def test_manual_clearance(self) -> None:
        v = _make_violation("clearance", "error", "Clearance too small")
        report = _make_report(violations=(v,))
        suggestions = interpret_violations(report)
        assert len(suggestions) == 1
        assert suggestions[0].auto_fixable is False

    def test_unconnected_items(self) -> None:
        u = _make_violation("unconnected_items", "error", "Missing connection")
        report = _make_report(unconnected=(u,))
        suggestions = interpret_violations(report)
        assert len(suggestions) == 1
        assert suggestions[0].violation_type == "unconnected_items"
        assert suggestions[0].auto_fixable is False

    def test_excluded_skipped(self) -> None:
        v = _make_violation("clearance", "error", excluded=True)
        report = _make_report(violations=(v,))
        suggestions = interpret_violations(report)
        assert len(suggestions) == 0

    def test_sorting_auto_first(self) -> None:
        v1 = _make_violation("clearance", "error", "Manual fix")
        v2 = _make_violation("silk_overlap", "warning", "Auto fix")
        report = _make_report(violations=(v1, v2))
        suggestions = interpret_violations(report)
        assert suggestions[0].auto_fixable is True
        assert suggestions[1].auto_fixable is False

    def test_unknown_type(self) -> None:
        v = _make_violation("new_drc_type_2026", "error", "New DRC rule")
        report = _make_report(violations=(v,))
        suggestions = interpret_violations(report)
        assert len(suggestions) == 1
        assert suggestions[0].auto_fixable is False
        assert suggestions[0].fix_action is None

    def test_empty_report(self) -> None:
        report = _make_report()
        suggestions = interpret_violations(report)
        assert suggestions == []

    def test_affected_refs(self) -> None:
        v = _make_violation("clearance", "error")
        report = _make_report(violations=(v,))
        suggestions = interpret_violations(report)
        assert "ref-1" in suggestions[0].affected_refs


class TestSummarizeSuggestions:
    def test_clean_board(self) -> None:
        summary = summarize_suggestions([])
        assert "clean" in summary.lower()

    def test_with_issues(self) -> None:
        suggestions = [
            DRCSuggestion(
                violation_type="silk_overlap",
                severity="warning",
                auto_fixable=True,
                description="Silk overlap",
                fix_action="Move silk text",
            ),
            DRCSuggestion(
                violation_type="clearance",
                severity="error",
                auto_fixable=False,
                description="Clearance violation",
                fix_action="Re-route trace",
            ),
        ]
        summary = summarize_suggestions(suggestions)
        assert "Auto-fixable" in summary
        assert "Manual" in summary
        assert "2 issues" in summary
