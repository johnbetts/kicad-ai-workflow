"""Tests for kicad_pipeline.agents.reporter.AgentReporter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.agents.commands import issue_command
from kicad_pipeline.agents.models import (
    AgentCommand,
    AgentState,
    BugSeverity,
    CommandType,
    DRCSummary,
    PipelineVersion,
    RunOutcome,
)
from kicad_pipeline.agents.registry import (
    load_registry,
    save_pipeline_version,
)
from kicad_pipeline.agents.reporter import AgentReporter
from kicad_pipeline.agents.status import load_status
from kicad_pipeline.constants import AGENT_COMMANDS_FILENAME, AGENT_STATUS_FILENAME

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def registry_dir(tmp_path: Path) -> Path:
    """Create a minimal registry directory structure."""
    (tmp_path / "agents").mkdir()
    return tmp_path


@pytest.fixture()
def reporter(registry_dir: Path) -> AgentReporter:
    """Create an AgentReporter backed by a temporary registry directory."""
    return AgentReporter("test-agent", registry_dir=registry_dir)


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_creates_status_file(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """register() must create the agent's status.json."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")

    status_path = registry_dir / "agents" / "test-agent" / AGENT_STATUS_FILENAME
    assert status_path.exists()

    status = load_status(status_path)
    assert status.agent_id == "test-agent"
    assert status.state == AgentState.REGISTERED


def test_register_updates_registry(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """register() must add the agent to registry.json."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")

    reg = load_registry(registry_dir / "registry.json")
    assert len(reg.agents) == 1
    assert reg.agents[0].agent_id == "test-agent"
    assert reg.agents[0].project_name == "Test Project"
    assert reg.agents[0].state == AgentState.REGISTERED


def test_register_with_variant(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """register() passes the active_variant through to registry and status."""
    reporter.register(
        project_path="/tmp/proj",
        project_name="Test Project",
        variant="smd-0603",
    )

    reg = load_registry(registry_dir / "registry.json")
    assert reg.agents[0].active_variant == "smd-0603"

    status_path = registry_dir / "agents" / "test-agent" / AGENT_STATUS_FILENAME
    status = load_status(status_path)
    assert status.active_variant == "smd-0603"


# ---------------------------------------------------------------------------
# update_state
# ---------------------------------------------------------------------------


def test_update_state_changes_status(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """update_state() must update the status file's state field."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")
    reporter.update_state(AgentState.RUNNING, message="building PCB")

    status_path = registry_dir / "agents" / "test-agent" / AGENT_STATUS_FILENAME
    status = load_status(status_path)
    assert status.state == AgentState.RUNNING
    assert status.message == "building PCB"


def test_update_state_changes_registry(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """update_state() must also propagate the new state to registry.json."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")
    reporter.update_state(AgentState.IDLE)

    reg = load_registry(registry_dir / "registry.json")
    assert reg.agents[0].state == AgentState.IDLE


# ---------------------------------------------------------------------------
# report_bug
# ---------------------------------------------------------------------------


def test_report_bug_adds_to_status(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """report_bug() must append a BugReport to the status file."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")
    bug_id = reporter.report_bug(
        title="Pad overlap",
        severity=BugSeverity.HIGH,
        module="pcb.placement",
        description="Pads overlap at (10, 20)",
        function="place_component",
    )

    assert isinstance(bug_id, str)
    assert len(bug_id) > 0

    status_path = registry_dir / "agents" / "test-agent" / AGENT_STATUS_FILENAME
    status = load_status(status_path)
    assert len(status.bugs) == 1
    assert status.bugs[0].bug_id == bug_id
    assert status.bugs[0].title == "Pad overlap"
    assert status.bugs[0].severity == BugSeverity.HIGH
    assert status.bugs[0].pipeline_module == "pcb.placement"
    assert status.bugs[0].pipeline_function == "place_component"


def test_report_bug_returns_unique_ids(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """Each call to report_bug() must return a distinct bug_id."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")
    id1 = reporter.report_bug(
        title="Bug 1", severity=BugSeverity.LOW, module="m", description="d",
    )
    id2 = reporter.report_bug(
        title="Bug 2", severity=BugSeverity.MEDIUM, module="m", description="d",
    )
    assert id1 != id2


# ---------------------------------------------------------------------------
# record_run
# ---------------------------------------------------------------------------


def test_record_run_adds_to_status(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """record_run() must append a RunRecord to the status file."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")
    drc = DRCSummary(total_violations=3, errors=1, warnings=2, unconnected=0)
    run_id = reporter.record_run(
        outcome=RunOutcome.DRC_ERRORS,
        drc_summary=drc,
        stages=("schematic", "pcb"),
        error_message="DRC failed",
    )

    assert isinstance(run_id, str)
    assert len(run_id) > 0

    status_path = registry_dir / "agents" / "test-agent" / AGENT_STATUS_FILENAME
    status = load_status(status_path)
    assert len(status.runs) == 1
    assert status.runs[0].run_id == run_id
    assert status.runs[0].outcome == RunOutcome.DRC_ERRORS
    assert status.runs[0].drc_summary is not None
    assert status.runs[0].drc_summary.errors == 1
    assert status.runs[0].stages_completed == ("schematic", "pcb")
    assert status.runs[0].error_message == "DRC failed"


def test_record_run_returns_unique_ids(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """Each call to record_run() must return a distinct run_id."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")
    id1 = reporter.record_run(outcome=RunOutcome.SUCCESS)
    id2 = reporter.record_run(outcome=RunOutcome.SUCCESS)
    assert id1 != id2


# ---------------------------------------------------------------------------
# check_commands
# ---------------------------------------------------------------------------


def test_check_commands_returns_unacknowledged(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """check_commands() must return only unacknowledged commands."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")

    agent_dir = registry_dir / "agents" / "test-agent"
    cmd = AgentCommand(
        command_id="cmd-001",
        command_type=CommandType.RERUN,
        issued_at="2026-03-07T00:00:00+00:00",
        args={"stage": "pcb"},
        reason="Fix applied",
    )
    issue_command(agent_dir, cmd)

    pending = reporter.check_commands()
    assert len(pending) == 1
    assert pending[0].command_id == "cmd-001"
    assert pending[0].command_type == CommandType.RERUN


def test_check_commands_empty_when_no_commands(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """check_commands() returns empty tuple when no commands exist."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")
    pending = reporter.check_commands()
    assert pending == ()


# ---------------------------------------------------------------------------
# acknowledge_command
# ---------------------------------------------------------------------------


def test_acknowledge_command_marks_acknowledged(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """acknowledge_command() must set acknowledged=True on the command."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")

    agent_dir = registry_dir / "agents" / "test-agent"
    cmd = AgentCommand(
        command_id="cmd-002",
        command_type=CommandType.RELOAD,
        issued_at="2026-03-07T00:00:00+00:00",
        reason="New pipeline version",
    )
    issue_command(agent_dir, cmd)

    # Before acknowledgment: 1 pending
    assert len(reporter.check_commands()) == 1

    reporter.acknowledge_command("cmd-002")

    # After acknowledgment: 0 pending
    assert len(reporter.check_commands()) == 0

    # Verify the command still exists but is acknowledged
    from kicad_pipeline.agents.commands import load_commands

    all_cmds = load_commands(agent_dir / AGENT_COMMANDS_FILENAME)
    assert len(all_cmds) == 1
    assert all_cmds[0].acknowledged is True


# ---------------------------------------------------------------------------
# check_pipeline_update
# ---------------------------------------------------------------------------


def test_check_pipeline_update_detects_change(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """check_pipeline_update() returns True when pipeline version differs."""
    # Save an initial pipeline version and register the agent (status gets "abc123")
    save_pipeline_version(
        PipelineVersion(
            git_hash="abc123", git_tag="v0.1.0", timestamp="2026-03-07T00:00:00+00:00",
        ),
        registry_dir=registry_dir,
    )
    reporter.register(project_path="/tmp/proj", project_name="Test Project")

    # Pipeline version matches status -> no update
    assert reporter.check_pipeline_update() is False

    # Now save a new pipeline version
    save_pipeline_version(
        PipelineVersion(
            git_hash="def456", git_tag="v0.2.0", timestamp="2026-03-07T01:00:00+00:00",
        ),
        registry_dir=registry_dir,
    )

    # Pipeline version differs from status -> update detected
    assert reporter.check_pipeline_update() is True


def test_check_pipeline_update_no_version_file(
    reporter: AgentReporter, registry_dir: Path,
) -> None:
    """check_pipeline_update() returns False when no version file exists."""
    reporter.register(project_path="/tmp/proj", project_name="Test Project")
    assert reporter.check_pipeline_update() is False
