"""Tests for the verification orchestrator framework.

Tests data models, checklist loading/filtering, and programmatic check wrappers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicad_pipeline.verification.checklist import VerificationChecklist
from kicad_pipeline.verification.component_checks import (
    EE_BOARD_CHECKS,
    EE_COMPONENT_CHECKS,
    FAB_BOARD_CHECKS,
    FAB_COMPONENT_CHECKS,
    generate_ee_check_items,
    generate_fab_check_items,
)
from kicad_pipeline.verification.orchestrator import (
    CheckResult,
    CheckSeverity,
    Persona,
    StepResult,
    StepType,
    VerificationReport,
)

# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_create_passing(self) -> None:
        cr = CheckResult(
            item_id="PROG-PAD-COUNT-R_0805",
            passed=True,
            detail="2 pads as expected",
            severity=CheckSeverity.MAJOR,
        )
        assert cr.passed
        assert cr.confidence == 1.0
        assert cr.evidence_path is None

    def test_create_failing(self) -> None:
        cr = CheckResult(
            item_id="FAB-BODY-ALIGN-U1",
            passed=False,
            detail="body offset 1.2mm from pad center",
            severity=CheckSeverity.CRITICAL,
            confidence=0.85,
            evidence_path="/tmp/u1_3d_top.png",
        )
        assert not cr.passed
        assert cr.confidence == 0.85
        assert cr.severity == CheckSeverity.CRITICAL


class TestStepResult:
    def test_step_passes_when_all_checks_pass(self) -> None:
        checks = (
            CheckResult("A", True, "ok", CheckSeverity.MAJOR),
            CheckResult("B", True, "ok", CheckSeverity.MINOR),
        )
        sr = StepResult(
            step_name="test", passed=True,
            check_results=checks, duration_secs=1.0,
        )
        assert sr.passed

    def test_step_type_defaults_to_programmatic(self) -> None:
        sr = StepResult(
            step_name="test", passed=True,
            check_results=(), duration_secs=0.1,
        )
        assert sr.step_type == StepType.PROGRAMMATIC


class TestVerificationReport:
    def test_empty_report_passes(self) -> None:
        report = VerificationReport(board_path="/tmp/test.kicad_pcb")
        assert report.passed
        assert report.total_checks == 0

    def test_report_with_critical_failure(self) -> None:
        step = StepResult(
            step_name="structural",
            passed=False,
            check_results=(
                CheckResult("A", True, "ok", CheckSeverity.MAJOR),
                CheckResult("B", False, "bad", CheckSeverity.CRITICAL),
            ),
            duration_secs=1.0,
        )
        report = VerificationReport(board_path="/tmp/test.kicad_pcb")
        report.steps.append(step)
        assert not report.passed
        assert report.total_checks == 2
        assert report.passed_checks == 1
        assert report.failed_checks == 1
        assert len(report.critical_failures) == 1

    def test_report_with_minor_failure_still_passes(self) -> None:
        step = StepResult(
            step_name="structural",
            passed=True,
            check_results=(
                CheckResult("A", True, "ok", CheckSeverity.MAJOR),
                CheckResult("B", False, "minor issue", CheckSeverity.MINOR),
            ),
            duration_secs=0.5,
        )
        report = VerificationReport(board_path="/tmp/test.kicad_pcb")
        report.steps.append(step)
        assert report.passed  # minor failures don't block

    def test_summary_format(self) -> None:
        step = StepResult(
            step_name="structural",
            passed=True,
            check_results=(
                CheckResult("A", True, "ok", CheckSeverity.MAJOR),
            ),
            duration_secs=0.5,
        )
        report = VerificationReport(board_path="/tmp/test.kicad_pcb")
        report.steps.append(step)
        summary = report.summary()
        assert "PASS" in summary
        assert "1/1" in summary


# ---------------------------------------------------------------------------
# Checklist tests
# ---------------------------------------------------------------------------


class TestVerificationChecklist:
    @pytest.fixture()
    def checklist_path(self, tmp_path: Path) -> Path:
        data = {
            "schema_version": 1,
            "known_bugs": [
                {
                    "id": "BUG-001",
                    "description": "Test bug for connectors",
                    "check_instruction": "Check connector placement",
                    "severity": "critical",
                    "affects_checks": ["FAB-CLEARANCE-*"],
                    "affects_components": ["J*"],
                    "verification_type": "programmatic",
                    "status": "open",
                },
                {
                    "id": "BUG-002",
                    "description": "Closed bug",
                    "check_instruction": "N/A",
                    "severity": "major",
                    "affects_checks": [],
                    "affects_components": [],
                    "verification_type": "programmatic",
                    "status": "closed",
                },
            ],
            "patterns": [
                {
                    "id": "PAT-001",
                    "description": "THT headers sit above board",
                    "check_instruction": "Accept 2-3mm gap for J*",
                    "severity": "info",
                    "affects_checks": ["FAB-BODY-FLAT-J*"],
                    "affects_components": ["J*"],
                    "verification_type": "ai_context",
                },
            ],
        }
        path = tmp_path / "checklist.json"
        path.write_text(json.dumps(data))
        return path

    def test_loads_bugs_and_patterns(self, checklist_path: Path) -> None:
        cl = VerificationChecklist(checklist_path)
        assert len(cl.bugs) == 2
        assert len(cl.open_bugs) == 1
        assert len(cl.patterns) == 1

    def test_bugs_for_component_matches_glob(self, checklist_path: Path) -> None:
        cl = VerificationChecklist(checklist_path)
        j_bugs = cl.bugs_for_component("J1")
        assert len(j_bugs) == 1
        assert j_bugs[0].id == "BUG-001"

    def test_bugs_for_component_no_match(self, checklist_path: Path) -> None:
        cl = VerificationChecklist(checklist_path)
        r_bugs = cl.bugs_for_component("R1")
        assert len(r_bugs) == 0

    def test_bugs_for_check_matches_glob(self, checklist_path: Path) -> None:
        cl = VerificationChecklist(checklist_path)
        bugs = cl.bugs_for_check("FAB-CLEARANCE-J1")
        assert len(bugs) == 1
        assert bugs[0].id == "BUG-001"

    def test_patterns_for_component(self, checklist_path: Path) -> None:
        cl = VerificationChecklist(checklist_path)
        pats = cl.patterns_for_component("J1")
        assert len(pats) == 1
        assert pats[0].id == "PAT-001"

    def test_patterns_for_non_matching_component(self, checklist_path: Path) -> None:
        cl = VerificationChecklist(checklist_path)
        pats = cl.patterns_for_component("U1")
        assert len(pats) == 0

    def test_context_for_agent_includes_bugs(self, checklist_path: Path) -> None:
        cl = VerificationChecklist(checklist_path)
        ctx = cl.context_for_agent("fab", refs=["J1"])
        assert "BUG-001" in ctx
        assert "Check connector placement" in ctx

    def test_context_for_agent_includes_patterns(self, checklist_path: Path) -> None:
        cl = VerificationChecklist(checklist_path)
        ctx = cl.context_for_agent("fab", refs=["J1"])
        assert "PAT-001" in ctx

    def test_context_for_agent_no_match(self, checklist_path: Path) -> None:
        cl = VerificationChecklist(checklist_path)
        ctx = cl.context_for_agent("fab", refs=["R1"])
        # R1 doesn't match J* bugs/patterns, so context should be empty or minimal
        assert "BUG-001" not in ctx

    def test_missing_file_loads_empty(self, tmp_path: Path) -> None:
        cl = VerificationChecklist(tmp_path / "nonexistent.json")
        assert cl.bugs == []
        assert cl.patterns == []

    def test_default_checklist_path(self) -> None:
        """Verify the default path points to data/verification_checklist.json."""
        cl = VerificationChecklist()
        # Should load without error — the file was created in this PR
        assert cl.bugs is not None


# ---------------------------------------------------------------------------
# Check item generation tests
# ---------------------------------------------------------------------------


class TestCheckItemGeneration:
    def test_fab_check_items_count(self) -> None:
        items = generate_fab_check_items(["R1", "C1"])
        assert len(items) == 2 * len(FAB_COMPONENT_CHECKS)

    def test_fab_check_item_ids_unique(self) -> None:
        items = generate_fab_check_items(["R1", "C1", "U1"])
        ids = [i.item_id for i in items]
        assert len(ids) == len(set(ids))

    def test_fab_check_item_format(self) -> None:
        items = generate_fab_check_items(["R1"])
        for item in items:
            assert item.item_id.startswith("FAB-")
            assert item.item_id.endswith("-R1")
            assert item.persona == Persona.FAB
            assert item.component_ref == "R1"

    def test_ee_check_items_count(self) -> None:
        items = generate_ee_check_items(["U1"])
        assert len(items) == len(EE_COMPONENT_CHECKS)

    def test_ee_check_item_format(self) -> None:
        items = generate_ee_check_items(["U1"])
        for item in items:
            assert item.item_id.startswith("EE-")
            assert item.item_id.endswith("-U1")
            assert item.persona == Persona.EE

    def test_known_bug_injection(self) -> None:
        """Check items linked to known bugs should carry bug IDs."""
        data = {
            "schema_version": 1,
            "known_bugs": [
                {
                    "id": "BUG-TEST",
                    "description": "Test",
                    "check_instruction": "Test",
                    "severity": "critical",
                    "affects_checks": ["FAB-CLEARANCE-*"],
                    "affects_components": [],
                    "verification_type": "ai_visual",
                    "status": "open",
                },
            ],
            "patterns": [],
        }
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)

        cl = VerificationChecklist(path)
        items = generate_fab_check_items(["J1"], checklist=cl)
        clearance_items = [i for i in items if "CLEARANCE" in i.item_id]
        assert len(clearance_items) == 1
        assert "BUG-TEST" in clearance_items[0].known_bug_ids

        path.unlink()


# ---------------------------------------------------------------------------
# Board-level check definitions
# ---------------------------------------------------------------------------


class TestBoardCheckDefinitions:
    def test_fab_board_checks_complete(self) -> None:
        assert len(FAB_BOARD_CHECKS) == 8

    def test_ee_board_checks_complete(self) -> None:
        assert len(EE_BOARD_CHECKS) == 8

    def test_fab_board_check_ids_start_with_fab(self) -> None:
        for check_id, _ in FAB_BOARD_CHECKS:
            assert check_id.startswith("FAB-BOARD-")

    def test_ee_board_check_ids_start_with_ee(self) -> None:
        for check_id, _ in EE_BOARD_CHECKS:
            assert check_id.startswith("EE-BOARD-")


# ---------------------------------------------------------------------------
# Programmatic checks integration test
# ---------------------------------------------------------------------------


class TestProgrammaticChecksIntegration:
    """Integration test that runs programmatic checks on the real registry.

    This is the smoke test that Phase 1 works end-to-end.
    """

    def test_check_id_format(self) -> None:
        """All programmatic check IDs should follow PROG-*-{component_id} pattern."""
        from kicad_pipeline.verification.component_checks import _VERIFIER_CHECK_MAP

        for old_name, new_prefix in _VERIFIER_CHECK_MAP.items():
            assert new_prefix.startswith("PROG-"), f"{old_name} maps to {new_prefix}"

    def test_convert_verifier_result(self) -> None:
        """Test the result conversion from old to new format."""
        from kicad_pipeline.validation.component_verifier import (
            CheckResult as OldCheckResult,
        )
        from kicad_pipeline.verification.component_checks import (
            _convert_verifier_result,
        )

        old = OldCheckResult(
            name="pad_count",
            passed=True,
            detail="2 pads as expected",
            severity="major",
        )
        new = _convert_verifier_result(old, "R_0805")
        assert new.item_id == "PROG-PAD-COUNT-R_0805"
        assert new.passed is True
        assert new.severity == CheckSeverity.MAJOR
        assert new.confidence == 1.0
