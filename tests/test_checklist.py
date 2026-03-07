"""Tests for pre-fabrication validation checklist."""

from __future__ import annotations

from kicad_pipeline.validation.checklist import (
    ChecklistReport,
    CheckResult,
    CheckStatus,
    format_checklist,
    run_checklist,
)
from kicad_pipeline.validation.kicad_drc import DRCReport, DRCViolation


def _make_minimal_pcb() -> object:
    """Create a minimal PCB-like object for testing.

    Uses a SimpleNamespace to avoid importing the full PCBDesign model,
    providing only the fields the checklist checks.
    """
    from types import SimpleNamespace

    # Minimal net
    net0 = SimpleNamespace(number=0, name="")
    net1 = SimpleNamespace(number=1, name="GND")
    net2 = SimpleNamespace(number=2, name="+3V3")

    # Minimal footprint
    fp1 = SimpleNamespace(ref="R1")
    fp2 = SimpleNamespace(ref="C1")

    # Minimal track
    track1 = SimpleNamespace(width=0.25)

    # Minimal via — Via uses position: Point, drill, size
    via_pos = SimpleNamespace(x=10.0, y=10.0)
    via1 = SimpleNamespace(position=via_pos, size=0.9, drill=0.508)

    # Board outline — PCBDesign uses outline.polygon of Point objects.
    outline_points = (
        SimpleNamespace(x=0.0, y=0.0),
        SimpleNamespace(x=50.0, y=0.0),
        SimpleNamespace(x=50.0, y=30.0),
        SimpleNamespace(x=0.0, y=30.0),
        SimpleNamespace(x=0.0, y=0.0),
    )
    outline = SimpleNamespace(polygon=outline_points)

    return SimpleNamespace(
        outline=outline,
        tracks=(track1,),
        vias=(via1,),
        nets=(net0, net1, net2),
        footprints=(fp1, fp2),
    )


class TestCheckStatus:
    def test_enum_values(self) -> None:
        assert CheckStatus.PASS.value == "pass"
        assert CheckStatus.FAIL.value == "fail"
        assert CheckStatus.WARN.value == "warn"
        assert CheckStatus.SKIP.value == "skip"


class TestChecklistReport:
    def test_passed_all_pass(self) -> None:
        results = (
            CheckResult("Test", "Check1", CheckStatus.PASS, "ok"),
            CheckResult("Test", "Check2", CheckStatus.PASS, "ok"),
        )
        report = ChecklistReport(results=results)
        assert report.passed is True
        assert report.pass_count == 2
        assert report.fail_count == 0

    def test_passed_with_warnings(self) -> None:
        results = (
            CheckResult("Test", "Check1", CheckStatus.PASS, "ok"),
            CheckResult("Test", "Check2", CheckStatus.WARN, "warning"),
        )
        report = ChecklistReport(results=results)
        assert report.passed is True

    def test_failed_with_error(self) -> None:
        results = (
            CheckResult("Test", "Check1", CheckStatus.PASS, "ok"),
            CheckResult("Test", "Check2", CheckStatus.FAIL, "failed"),
        )
        report = ChecklistReport(results=results)
        assert report.passed is False
        assert report.fail_count == 1


class TestRunChecklist:
    def test_minimal_valid_pcb(self) -> None:
        pcb = _make_minimal_pcb()
        report = run_checklist(pcb)  # type: ignore[arg-type]
        assert isinstance(report, ChecklistReport)
        # Should pass basic checks.
        assert report.pass_count >= 3

    def test_board_dimensions_pass(self) -> None:
        pcb = _make_minimal_pcb()
        report = run_checklist(pcb)  # type: ignore[arg-type]
        dim_results = [r for r in report.results if r.name == "Board dimensions"]
        assert len(dim_results) == 1
        assert dim_results[0].status == CheckStatus.PASS

    def test_outline_closure(self) -> None:
        pcb = _make_minimal_pcb()
        report = run_checklist(pcb)  # type: ignore[arg-type]
        closure = [r for r in report.results if r.name == "Outline closure"]
        assert len(closure) == 1
        assert closure[0].status == CheckStatus.PASS

    def test_trace_width_pass(self) -> None:
        pcb = _make_minimal_pcb()
        report = run_checklist(pcb)  # type: ignore[arg-type]
        trace_results = [r for r in report.results if "trace" in r.name.lower()]
        assert len(trace_results) >= 1
        assert trace_results[0].status == CheckStatus.PASS

    def test_no_tracks_skips(self) -> None:

        pcb = _make_minimal_pcb()
        pcb.tracks = ()  # type: ignore[attr-defined]
        report = run_checklist(pcb)  # type: ignore[arg-type]
        trace_results = [r for r in report.results if "trace" in r.name.lower()]
        assert trace_results[0].status == CheckStatus.SKIP

    def test_duplicate_refs_fail(self) -> None:
        from types import SimpleNamespace

        pcb = _make_minimal_pcb()
        pcb.footprints = (  # type: ignore[attr-defined]
            SimpleNamespace(ref="R1"),
            SimpleNamespace(ref="R1"),
        )
        report = run_checklist(pcb)  # type: ignore[arg-type]
        dup_results = [r for r in report.results if "uplicate" in r.name]
        assert len(dup_results) == 1
        assert dup_results[0].status == CheckStatus.FAIL

    def test_with_drc_report_passed(self) -> None:
        pcb = _make_minimal_pcb()
        drc = DRCReport()
        report = run_checklist(pcb, drc_report=drc)  # type: ignore[arg-type]
        drc_results = [r for r in report.results if r.category == "DRC"]
        assert len(drc_results) >= 1
        assert drc_results[0].status == CheckStatus.PASS

    def test_with_drc_report_failed(self) -> None:
        pcb = _make_minimal_pcb()
        drc = DRCReport(
            violations=(
                DRCViolation(
                    type="clearance", severity="error",
                    description="test", excluded=False,
                ),
            ),
        )
        report = run_checklist(pcb, drc_report=drc)  # type: ignore[arg-type]
        drc_results = [r for r in report.results if r.category == "DRC"]
        assert any(r.status == CheckStatus.FAIL for r in drc_results)

    def test_no_drc_skips(self) -> None:
        pcb = _make_minimal_pcb()
        report = run_checklist(pcb)  # type: ignore[arg-type]
        drc_results = [r for r in report.results if r.category == "DRC"]
        assert drc_results[0].status == CheckStatus.SKIP


class TestFormatChecklist:
    def test_format_includes_status(self) -> None:
        results = (
            CheckResult("Mechanical", "Board size", CheckStatus.PASS, "50x30mm OK"),
        )
        report = ChecklistReport(results=results)
        formatted = format_checklist(report)
        assert "READY" in formatted
        assert "[PASS]" in formatted
        assert "Board size" in formatted

    def test_format_failed(self) -> None:
        results = (
            CheckResult("Manufacturing", "Trace width", CheckStatus.FAIL, "Too thin"),
        )
        report = ChecklistReport(results=results)
        formatted = format_checklist(report)
        assert "NOT READY" in formatted
        assert "[FAIL]" in formatted
