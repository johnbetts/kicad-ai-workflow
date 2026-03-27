"""Tests for the evidence-hardened process runner extensions."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the process runner as a module (it's a script, not a package)
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "process_runner.py"
_spec = importlib.util.spec_from_file_location("process_runner", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
process_runner = importlib.util.module_from_spec(_spec)
sys.modules["process_runner"] = process_runner  # required for dataclass introspection
_spec.loader.exec_module(process_runner)  # type: ignore[union-attr]

# Pull symbols we need
EvidenceStep = process_runner.EvidenceStep
Step = process_runner.Step
StepResult = process_runner.StepResult
VerifyResult = process_runner.VerifyResult
run_evidence_step = process_runner.run_evidence_step
run_verification_step = process_runner.run_verification_step
run_hardened_process = process_runner.run_hardened_process
PCB_REVIEW_EVIDENCE_STEPS = process_runner.PCB_REVIEW_EVIDENCE_STEPS

# Evidence imports (E402 unavoidable — must come after dynamic module load)
from kicad_pipeline.evidence.ledger import append_record, load_ledger  # noqa: E402
from kicad_pipeline.evidence.models import (  # noqa: E402
    EvidenceKind,
    EvidenceRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def board_path(tmp_path: Path) -> str:
    """Create a minimal board file and return its path as a string."""
    board_dir = tmp_path / "output" / "test_board"
    board_dir.mkdir(parents=True)
    board_file = board_dir / "test_board.kicad_pcb"
    board_file.write_text(
        '(kicad_pcb (version 20241229)\n'
        '  (footprint "R0805"\n'
        '    (property "Reference" "R1")\n'
        '  )\n'
        ')\n'
    )
    return str(board_file)


def _always_pass_verify(board_path: str) -> VerifyResult:
    return VerifyResult(passed=True, message="OK")


def _always_fail_verify(board_path: str) -> VerifyResult:
    return VerifyResult(
        passed=False, message="Failed",
        errors=["something went wrong"],
    )


# ---------------------------------------------------------------------------
# EvidenceStep creation and defaults
# ---------------------------------------------------------------------------

class TestEvidenceStepCreation:
    """Test EvidenceStep dataclass field defaults."""

    def test_defaults(self) -> None:
        step = EvidenceStep(
            name="test",
            description="A test step",
            agent_prompt="Do something with {board}",
            verify=_always_pass_verify,
        )
        assert step.stage == "pcb"
        assert step.evidence_kind == EvidenceKind.REVIEW
        assert step.producer == "harness"
        assert step.requires_gate is False
        assert step.inject_known_issues is True

    def test_custom_fields(self) -> None:
        step = EvidenceStep(
            name="render",
            description="Render board",
            agent_prompt="Render {board}",
            verify=_always_pass_verify,
            stage="schematic",
            evidence_kind=EvidenceKind.RENDER,
            producer="render-agent",
            requires_gate=True,
            inject_known_issues=False,
        )
        assert step.stage == "schematic"
        assert step.evidence_kind == EvidenceKind.RENDER
        assert step.producer == "render-agent"
        assert step.requires_gate is True
        assert step.inject_known_issues is False


# ---------------------------------------------------------------------------
# PCB_REVIEW_EVIDENCE_STEPS conversion
# ---------------------------------------------------------------------------

class TestPCBReviewEvidenceSteps:
    """Test that evidence steps were correctly built from PCB_REVIEW_STEPS."""

    def test_count(self) -> None:
        assert len(PCB_REVIEW_EVIDENCE_STEPS) == 4

    def test_render_step(self) -> None:
        s = PCB_REVIEW_EVIDENCE_STEPS[0]
        assert s.name == "render"
        assert s.evidence_kind == EvidenceKind.RENDER
        assert s.producer == "harness"

    def test_fab_review_step(self) -> None:
        s = PCB_REVIEW_EVIDENCE_STEPS[1]
        assert s.name == "fab-review"
        assert s.evidence_kind == EvidenceKind.REVIEW
        assert s.producer == "fab-agent"

    def test_ee_review_step(self) -> None:
        s = PCB_REVIEW_EVIDENCE_STEPS[2]
        assert s.name == "ee-review"
        assert s.evidence_kind == EvidenceKind.REVIEW
        assert s.producer == "ee-agent"

    def test_3d_verify_step(self) -> None:
        s = PCB_REVIEW_EVIDENCE_STEPS[3]
        assert s.name == "3d-component-verify"
        assert s.evidence_kind == EvidenceKind.VERIFICATION
        assert s.producer == "verify-agent"


# ---------------------------------------------------------------------------
# run_evidence_step
# ---------------------------------------------------------------------------

class TestRunEvidenceStep:
    """Test run_evidence_step writes evidence records."""

    @patch.object(process_runner, "run_claude", return_value="Agent completed OK")
    def test_writes_evidence_on_success(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        step = EvidenceStep(
            name="test-step",
            description="Test",
            agent_prompt="Do {board}",
            verify=_always_pass_verify,
            inject_known_issues=False,
        )
        result = run_evidence_step(step, board_path)
        assert result.passed is True

        # Check ledger has a record
        ledger = load_ledger(Path(board_path))
        assert len(ledger.records) == 1
        rec = ledger.records[0]
        assert rec.kind == EvidenceKind.REVIEW
        assert rec.stage == "pcb"
        assert rec.step == "test-step"
        assert rec.passed is True

    @patch.object(process_runner, "run_claude", return_value="Agent output")
    def test_writes_evidence_on_failure(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        step = EvidenceStep(
            name="fail-step",
            description="Fails",
            agent_prompt="Do {board}",
            verify=_always_fail_verify,
            max_retries=1,
            inject_known_issues=False,
        )
        result = run_evidence_step(step, board_path)
        assert result.passed is False

        ledger = load_ledger(Path(board_path))
        assert len(ledger.records) == 1
        rec = ledger.records[0]
        assert rec.passed is False
        assert len(rec.issues) > 0

    @patch.object(process_runner, "run_claude", return_value="Agent output")
    def test_gate_blocks_when_evidence_missing(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        step = EvidenceStep(
            name="gated-step",
            description="Needs gate",
            agent_prompt="Do {board}",
            verify=_always_pass_verify,
            requires_gate=True,
            inject_known_issues=False,
            stage="pcb",
        )
        result = run_evidence_step(step, board_path)
        assert result.passed is False
        assert "blocked" in result.final_message.lower() or "gate" in result.final_message.lower()

        # Agent should NOT have been called
        mock_claude.assert_not_called()

    @patch.object(process_runner, "run_claude", return_value="Agent output")
    def test_known_issues_injected_into_prompt(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        step = EvidenceStep(
            name="inject-step",
            description="Has injection",
            agent_prompt="Do {board}",
            verify=_always_pass_verify,
            inject_known_issues=True,
        )

        with patch.object(
            process_runner,
            "load_known_issues",
            return_value="## KNOWN ISSUES\nDo not repeat KI-099",
        ):
            result = run_evidence_step(step, board_path)

        assert result.passed is True
        # The prompt should have been prepended with known issues
        call_args = mock_claude.call_args
        prompt_sent = call_args[0][0]
        assert "KNOWN ISSUES" in prompt_sent


# ---------------------------------------------------------------------------
# run_verification_step
# ---------------------------------------------------------------------------

class TestRunVerificationStep:
    """Test verification step creates VERIFICATION records."""

    @patch.object(
        process_runner, "run_claude",
        return_value="Verified: output matches files",
    )
    def test_creates_verification_record(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        result = run_verification_step(
            board_path,
            original_step="fab-review",
            original_output="Found 3 issues",
        )
        assert result.passed is True
        assert result.step_name == "verify-fab-review"

        ledger = load_ledger(Path(board_path))
        assert len(ledger.records) == 1
        rec = ledger.records[0]
        assert rec.kind == EvidenceKind.VERIFICATION
        assert rec.step == "verify-fab-review"
        assert rec.producer == "verify-agent"

    @patch.object(
        process_runner, "run_claude",
        return_value="Verified: output matches files",
    )
    def test_verification_prompt_contains_original(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        run_verification_step(
            board_path,
            original_step="ee-review",
            original_output="Signal integrity OK",
        )
        prompt_sent = mock_claude.call_args[0][0]
        assert "ee-review" in prompt_sent
        assert "Signal integrity OK" in prompt_sent
        assert "verification agent" in prompt_sent.lower()

    @patch.object(process_runner, "run_claude", return_value="[TIMEOUT]")
    def test_timeout_marks_failed(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        result = run_verification_step(
            board_path,
            original_step="fab-review",
            original_output="stuff",
        )
        assert result.passed is False


# ---------------------------------------------------------------------------
# run_hardened_process
# ---------------------------------------------------------------------------

class TestRunHardenedProcess:
    """Test the full hardened process loop."""

    @patch.object(process_runner, "run_claude", return_value="Agent OK")
    def test_stops_on_gate_failure(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        steps = [
            EvidenceStep(
                name="gated",
                description="Blocked by gate",
                agent_prompt="Do {board}",
                verify=_always_pass_verify,
                requires_gate=True,
                inject_known_issues=False,
                stage="pcb",
            ),
        ]
        result = run_hardened_process("test", steps, board_path)
        assert result.passed is False
        assert len(result.steps) == 1
        assert result.steps[0].passed is False

    @patch.object(process_runner, "run_claude", return_value="Agent OK")
    def test_all_pass_ungated(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        steps = [
            EvidenceStep(
                name="step-a",
                description="First",
                agent_prompt="Do {board}",
                verify=_always_pass_verify,
                inject_known_issues=False,
            ),
            EvidenceStep(
                name="step-b",
                description="Second",
                agent_prompt="Do {board}",
                verify=_always_pass_verify,
                inject_known_issues=False,
            ),
        ]
        result = run_hardened_process("test", steps, board_path)
        assert result.passed is True
        assert len(result.steps) == 2

        # Both should have evidence
        ledger = load_ledger(Path(board_path))
        assert len(ledger.records) == 2

    @patch.object(process_runner, "run_claude", return_value="Agent OK")
    def test_verification_runs_after_review_steps(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        """fab-review and ee-review trigger verification steps."""
        steps = [
            EvidenceStep(
                name="fab-review",
                description="Fab review",
                agent_prompt="Review {board}",
                verify=_always_pass_verify,
                inject_known_issues=False,
                producer="fab-agent",
            ),
        ]
        result = run_hardened_process("test", steps, board_path)
        assert result.passed is True
        # Should have 2 steps: fab-review + verify-fab-review
        assert len(result.steps) == 2
        assert result.steps[1].step_name == "verify-fab-review"

    @patch.object(process_runner, "run_claude", return_value="Agent OK")
    def test_stops_on_step_failure(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        steps = [
            EvidenceStep(
                name="fail-step",
                description="Will fail",
                agent_prompt="Do {board}",
                verify=_always_fail_verify,
                max_retries=1,
                inject_known_issues=False,
            ),
            EvidenceStep(
                name="never-reached",
                description="Should not run",
                agent_prompt="Do {board}",
                verify=_always_pass_verify,
                inject_known_issues=False,
            ),
        ]
        result = run_hardened_process("test", steps, board_path)
        assert result.passed is False
        assert len(result.steps) == 1
        assert result.steps[0].step_name == "fail-step"

    @patch.object(process_runner, "run_claude", return_value="Agent OK")
    def test_stage_transition_writes_gate_record(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        """When stage changes between steps, a GATE_RESULT is written."""
        # Pre-populate ledger with passing evidence for "requirements"
        rec = EvidenceRecord(
            kind=EvidenceKind.REVIEW,
            stage="requirements",
            step="req-review",
            board="test_board",
            passed=True,
            summary="OK",
        )
        append_record(Path(board_path), rec)

        steps = [
            EvidenceStep(
                name="req-step",
                description="Requirements",
                agent_prompt="Do {board}",
                verify=_always_pass_verify,
                inject_known_issues=False,
                stage="requirements",
            ),
            EvidenceStep(
                name="sch-step",
                description="Schematic",
                agent_prompt="Do {board}",
                verify=_always_pass_verify,
                inject_known_issues=False,
                stage="schematic",
            ),
        ]
        result = run_hardened_process("test", steps, board_path)
        assert result.passed is True

        ledger = load_ledger(Path(board_path))
        gate_records = [
            r for r in ledger.records
            if r.kind == EvidenceKind.GATE_RESULT
        ]
        assert len(gate_records) == 1
        assert gate_records[0].stage == "requirements"


# ---------------------------------------------------------------------------
# EVIDENCE_PROCESSES registration
# ---------------------------------------------------------------------------

class TestEvidenceProcessesRegistration:
    """Test that EVIDENCE_PROCESSES contains the hardened process."""

    def test_hardened_process_registered(self) -> None:
        assert "pcb-review-hardened" in process_runner.EVIDENCE_PROCESSES

    def test_hardened_steps_are_evidence_steps(self) -> None:
        steps = process_runner.EVIDENCE_PROCESSES["pcb-review-hardened"]
        for s in steps:
            assert isinstance(s, EvidenceStep)
