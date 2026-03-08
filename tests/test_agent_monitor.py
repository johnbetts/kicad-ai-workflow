"""Tests for kicad_pipeline.agents.monitor module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from kicad_pipeline.agents.models import (
    AgentRegistration,
    AgentRegistry,
    AgentState,
    AgentStatus,
    BugReport,
    BugSeverity,
    BugStatus,
    CommandType,
)
from kicad_pipeline.agents.monitor import AgentMonitor
from kicad_pipeline.agents.registry import save_registry
from kicad_pipeline.agents.status import save_status
from kicad_pipeline.constants import AGENT_COMMANDS_FILENAME, AGENT_STATUS_FILENAME

if TYPE_CHECKING:
    from pathlib import Path


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _hours_ago_iso(hours: float) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()


def _make_registration(
    agent_id: str = "agent-1",
    state: AgentState = AgentState.RUNNING,
    last_seen: str | None = None,
) -> AgentRegistration:
    return AgentRegistration(
        agent_id=agent_id,
        project_path="/tmp/test-project",
        project_name="test-project",
        description="Test board agent",
        registered_at="2026-03-07T10:00:00+00:00",
        last_seen=last_seen or _now_iso(),
        state=state,
    )


def _make_status(
    agent_id: str = "agent-1",
    bugs: tuple[BugReport, ...] = (),
) -> AgentStatus:
    return AgentStatus(
        agent_id=agent_id,
        state=AgentState.RUNNING,
        updated_at=_now_iso(),
        pipeline_version="abc123",
        bugs=bugs,
    )


def _make_bug(
    bug_id: str = "bug-1",
    status: BugStatus = BugStatus.OPEN,
    severity: BugSeverity = BugSeverity.HIGH,
) -> BugReport:
    return BugReport(
        bug_id=bug_id,
        title="Test bug",
        severity=severity,
        status=status,
        description="Something broke",
        pipeline_module="pcb.builder",
        pipeline_function="build_pcb",
        reported_at="2026-03-07T11:00:00+00:00",
    )


def _setup_registry(tmp_path: Path, registrations: tuple[AgentRegistration, ...]) -> None:
    """Save a registry and create agent directories."""
    registry = AgentRegistry(
        agents=registrations,
        updated_at=_now_iso(),
    )
    save_registry(registry, tmp_path / "registry.json")
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(exist_ok=True)
    for reg in registrations:
        (agents_dir / reg.agent_id).mkdir(exist_ok=True)


def _setup_status(tmp_path: Path, agent_id: str, status: AgentStatus) -> None:
    """Write a status file for a given agent."""
    status_path = tmp_path / "agents" / agent_id / AGENT_STATUS_FILENAME
    status_path.parent.mkdir(parents=True, exist_ok=True)
    save_status(status, status_path)


# ---------------------------------------------------------------------------
# scan_all — empty registry
# ---------------------------------------------------------------------------


def test_scan_all_returns_empty_when_no_agents(tmp_path: Path) -> None:
    _setup_registry(tmp_path, ())
    monitor = AgentMonitor(registry_dir=tmp_path)
    result = monitor.scan_all()
    assert result == ()


# ---------------------------------------------------------------------------
# scan_all — returns registration + status pairs
# ---------------------------------------------------------------------------


def test_scan_all_returns_registration_and_status_pairs(tmp_path: Path) -> None:
    reg = _make_registration(agent_id="agent-1")
    _setup_registry(tmp_path, (reg,))

    status = _make_status(agent_id="agent-1")
    _setup_status(tmp_path, "agent-1", status)

    monitor = AgentMonitor(registry_dir=tmp_path)
    result = monitor.scan_all()

    assert len(result) == 1
    returned_reg, returned_status = result[0]
    assert returned_reg.agent_id == "agent-1"
    assert returned_status is not None
    assert returned_status.agent_id == "agent-1"


def test_scan_all_returns_none_status_when_missing(tmp_path: Path) -> None:
    reg = _make_registration(agent_id="agent-2")
    _setup_registry(tmp_path, (reg,))

    monitor = AgentMonitor(registry_dir=tmp_path)
    result = monitor.scan_all()

    assert len(result) == 1
    _, returned_status = result[0]
    assert returned_status is None


# ---------------------------------------------------------------------------
# find_open_bugs
# ---------------------------------------------------------------------------


def test_find_open_bugs_finds_open_bugs(tmp_path: Path) -> None:
    reg = _make_registration(agent_id="agent-1")
    _setup_registry(tmp_path, (reg,))

    open_bug = _make_bug(bug_id="bug-open", status=BugStatus.OPEN)
    fixed_bug = _make_bug(bug_id="bug-fixed", status=BugStatus.FIXED)
    status = _make_status(agent_id="agent-1", bugs=(open_bug, fixed_bug))
    _setup_status(tmp_path, "agent-1", status)

    monitor = AgentMonitor(registry_dir=tmp_path)
    bugs = monitor.find_open_bugs()

    assert len(bugs) == 1
    agent_id, bug = bugs[0]
    assert agent_id == "agent-1"
    assert bug.bug_id == "bug-open"
    assert bug.status == BugStatus.OPEN


def test_find_open_bugs_returns_empty_when_all_fixed(tmp_path: Path) -> None:
    reg = _make_registration(agent_id="agent-1")
    _setup_registry(tmp_path, (reg,))

    fixed_bug = _make_bug(bug_id="bug-fixed", status=BugStatus.FIXED)
    status = _make_status(agent_id="agent-1", bugs=(fixed_bug,))
    _setup_status(tmp_path, "agent-1", status)

    monitor = AgentMonitor(registry_dir=tmp_path)
    bugs = monitor.find_open_bugs()
    assert bugs == ()


# ---------------------------------------------------------------------------
# find_stale_agents
# ---------------------------------------------------------------------------


def test_find_stale_agents_detects_old_last_seen(tmp_path: Path) -> None:
    stale_reg = _make_registration(
        agent_id="stale-agent",
        last_seen=_hours_ago_iso(48),
    )
    fresh_reg = _make_registration(
        agent_id="fresh-agent",
        last_seen=_now_iso(),
    )
    _setup_registry(tmp_path, (stale_reg, fresh_reg))

    monitor = AgentMonitor(registry_dir=tmp_path)
    stale = monitor.find_stale_agents(stale_hours=24.0)

    assert len(stale) == 1
    assert stale[0].agent_id == "stale-agent"


def test_find_stale_agents_skips_completed(tmp_path: Path) -> None:
    completed_reg = _make_registration(
        agent_id="done-agent",
        state=AgentState.COMPLETED,
        last_seen=_hours_ago_iso(100),
    )
    _setup_registry(tmp_path, (completed_reg,))

    monitor = AgentMonitor(registry_dir=tmp_path)
    stale = monitor.find_stale_agents(stale_hours=24.0)
    assert stale == ()


# ---------------------------------------------------------------------------
# issue_rerun — creates command file
# ---------------------------------------------------------------------------


def test_issue_rerun_creates_command_file(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agents" / "agent-1"
    agent_dir.mkdir(parents=True)

    monitor = AgentMonitor(registry_dir=tmp_path)
    monitor.issue_rerun("agent-1", stage="pcb", reason="DRC regression")

    cmd_path = agent_dir / AGENT_COMMANDS_FILENAME
    assert cmd_path.exists()

    from kicad_pipeline.agents.commands import load_commands

    commands = load_commands(cmd_path)
    assert len(commands) == 1
    assert commands[0].command_type == CommandType.RERUN
    assert commands[0].args["stage"] == "pcb"
    assert commands[0].reason == "DRC regression"
    assert commands[0].acknowledged is False


# ---------------------------------------------------------------------------
# issue_bug_update — creates command file
# ---------------------------------------------------------------------------


def test_issue_bug_update_creates_command_file(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agents" / "agent-1"
    agent_dir.mkdir(parents=True)

    monitor = AgentMonitor(registry_dir=tmp_path)
    monitor.issue_bug_update(
        agent_id="agent-1",
        bug_id="bug-42",
        new_status=BugStatus.FIXED,
        fix_commit="deadbeef",
    )

    cmd_path = agent_dir / AGENT_COMMANDS_FILENAME
    assert cmd_path.exists()

    from kicad_pipeline.agents.commands import load_commands

    commands = load_commands(cmd_path)
    assert len(commands) == 1
    assert commands[0].command_type == CommandType.BUG_UPDATE
    assert commands[0].args["bug_id"] == "bug-42"
    assert commands[0].args["new_status"] == "fixed"
    assert commands[0].args["fix_commit"] == "deadbeef"
