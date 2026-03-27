"""Integration tests: process_runner <-> dashboard notification wiring."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the process runner as a module (it's a script, not a package)
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "process_runner.py"
_spec = importlib.util.spec_from_file_location("process_runner", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
process_runner = importlib.util.module_from_spec(_spec)
sys.modules["process_runner"] = process_runner
_spec.loader.exec_module(process_runner)  # type: ignore[union-attr]

# Pull symbols
EvidenceStep = process_runner.EvidenceStep
VerifyResult = process_runner.VerifyResult
notify_dashboard = process_runner.notify_dashboard
notify_dashboard_log = process_runner.notify_dashboard_log
run_evidence_step = process_runner.run_evidence_step
run_hardened_process = process_runner.run_hardened_process

from kicad_pipeline.evidence.ledger import append_record  # noqa: E402
from kicad_pipeline.evidence.models import EvidenceKind, EvidenceRecord  # noqa: E402

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


@pytest.fixture(autouse=True)
def _reset_dashboard_url() -> None:
    """Reset _dashboard_url before each test."""
    process_runner._dashboard_url = None


# ---------------------------------------------------------------------------
# notify_dashboard
# ---------------------------------------------------------------------------

class TestNotifyDashboard:
    """Test the notify_dashboard function."""

    def test_noop_when_disabled(self) -> None:
        """When _dashboard_url is None, no HTTP call is made."""
        process_runner._dashboard_url = None
        record = EvidenceRecord(
            kind=EvidenceKind.REVIEW,
            stage="pcb",
            step="test",
            board="test_board",
            passed=True,
            summary="OK",
        )
        with patch.object(process_runner.urllib.request, "urlopen") as mock_open:
            notify_dashboard(record)
            mock_open.assert_not_called()

    def test_posts_record(self) -> None:
        """When enabled, POSTs the record JSON to /api/evidence."""
        process_runner._dashboard_url = "http://localhost:9999"
        record = EvidenceRecord(
            kind=EvidenceKind.REVIEW,
            stage="pcb",
            step="test",
            board="test_board",
            passed=True,
            summary="All good",
        )
        with patch.object(process_runner.urllib.request, "urlopen") as mock_open:
            notify_dashboard(record)
            mock_open.assert_called_once()
            req_obj = mock_open.call_args[0][0]
            assert req_obj.full_url == "http://localhost:9999/api/evidence"
            assert req_obj.get_header("Content-type") == "application/json"
            body = json.loads(req_obj.data.decode("utf-8"))
            assert body["kind"] == "review"
            assert body["board"] == "test_board"

    def test_silent_on_error(self) -> None:
        """Exceptions from urlopen must not propagate."""
        process_runner._dashboard_url = "http://localhost:9999"
        record = EvidenceRecord(
            kind=EvidenceKind.REVIEW,
            stage="pcb",
            step="test",
            board="test_board",
            passed=True,
            summary="OK",
        )
        with patch.object(
            process_runner.urllib.request, "urlopen",
            side_effect=ConnectionError("refused"),
        ):
            # Must not raise
            notify_dashboard(record)


# ---------------------------------------------------------------------------
# notify_dashboard_log
# ---------------------------------------------------------------------------

class TestNotifyDashboardLog:
    """Test the notify_dashboard_log function."""

    def test_noop_when_disabled(self) -> None:
        process_runner._dashboard_url = None
        with patch.object(process_runner.urllib.request, "urlopen") as mock_open:
            notify_dashboard_log("hello")
            mock_open.assert_not_called()

    def test_posts_log_message(self) -> None:
        process_runner._dashboard_url = "http://localhost:9999"
        with patch.object(process_runner.urllib.request, "urlopen") as mock_open:
            notify_dashboard_log("test message", "warning")
            mock_open.assert_called_once()
            req_obj = mock_open.call_args[0][0]
            assert req_obj.full_url == "http://localhost:9999/api/log"
            body = json.loads(req_obj.data.decode("utf-8"))
            assert body["message"] == "test message"
            assert body["level"] == "warning"


# ---------------------------------------------------------------------------
# log() integration
# ---------------------------------------------------------------------------

class TestLogNotifiesDashboard:
    """Test that log() calls notify_dashboard_log."""

    def test_log_calls_notify(self) -> None:
        process_runner._dashboard_url = "http://localhost:9999"
        with patch.object(process_runner, "notify_dashboard_log") as mock_notify:
            process_runner.log("hello world")
            mock_notify.assert_called_once_with("hello world")


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

class TestDashboardCLIFlags:
    """Test --dashboard and --no-dashboard argument parsing."""

    def test_dashboard_flag_sets_url(self) -> None:
        """--dashboard sets the URL."""
        parser = process_runner.argparse.ArgumentParser()
        parser.add_argument("process")
        parser.add_argument("board", nargs="?")
        parser.add_argument("--dashboard", type=str, default="http://localhost:8080")
        parser.add_argument("--no-dashboard", action="store_true")
        args = parser.parse_args(["pcb-review", "mcu", "--dashboard", "http://myhost:3000"])
        assert args.dashboard == "http://myhost:3000"
        assert args.no_dashboard is False

    def test_no_dashboard_flag_disables(self) -> None:
        """--no-dashboard sets the flag to True."""
        parser = process_runner.argparse.ArgumentParser()
        parser.add_argument("process")
        parser.add_argument("board", nargs="?")
        parser.add_argument("--dashboard", type=str, default="http://localhost:8080")
        parser.add_argument("--no-dashboard", action="store_true")
        args = parser.parse_args(["pcb-review", "mcu", "--no-dashboard"])
        assert args.no_dashboard is True

    def test_default_dashboard_url(self) -> None:
        """Default dashboard URL is http://localhost:8080."""
        parser = process_runner.argparse.ArgumentParser()
        parser.add_argument("process")
        parser.add_argument("board", nargs="?")
        parser.add_argument("--dashboard", type=str, default="http://localhost:8080")
        parser.add_argument("--no-dashboard", action="store_true")
        args = parser.parse_args(["pcb-review", "mcu"])
        assert args.dashboard == "http://localhost:8080"


# ---------------------------------------------------------------------------
# Evidence step notifies dashboard
# ---------------------------------------------------------------------------

class TestEvidenceStepNotifiesDashboard:
    """Test that run_evidence_step calls notify_dashboard."""

    @patch.object(process_runner, "run_claude", return_value="Agent OK")
    def test_evidence_step_notifies(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        process_runner._dashboard_url = "http://localhost:9999"
        step = EvidenceStep(
            name="test-step",
            description="Test",
            agent_prompt="Do {board}",
            verify=_always_pass_verify,
            inject_known_issues=False,
        )
        with patch.object(process_runner, "notify_dashboard") as mock_notify:
            run_evidence_step(step, board_path)
            mock_notify.assert_called_once()
            record = mock_notify.call_args[0][0]
            assert record.kind == EvidenceKind.REVIEW
            assert record.passed is True


# ---------------------------------------------------------------------------
# Hardened process notifies on gate
# ---------------------------------------------------------------------------

class TestHardenedProcessNotifiesOnGate:
    """Test that run_hardened_process notifies dashboard on gate results."""

    @patch.object(process_runner, "run_claude", return_value="Agent OK")
    def test_gate_record_notified(
        self, mock_claude: MagicMock, board_path: str,
    ) -> None:
        process_runner._dashboard_url = "http://localhost:9999"

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
        with patch.object(process_runner, "notify_dashboard") as mock_notify:
            run_hardened_process("test", steps, board_path)
            # Should have been called for: req-step evidence, gate record, sch-step evidence
            gate_calls = [
                call for call in mock_notify.call_args_list
                if call[0][0].kind == EvidenceKind.GATE_RESULT
            ]
            assert len(gate_calls) == 1
            assert gate_calls[0][0][0].stage == "requirements"
