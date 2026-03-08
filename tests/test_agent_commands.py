"""Tests for kicad_pipeline.agents.commands module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.agents.commands import (
    _command_from_dict,
    _command_to_dict,
    acknowledge_command,
    issue_command,
    load_commands,
    save_commands,
)
from kicad_pipeline.agents.models import AgentCommand, CommandType
from kicad_pipeline.exceptions import AgentError

if TYPE_CHECKING:
    from pathlib import Path


def _make_agent_dir(tmp_path: Path) -> Path:
    """Create and return the standard agent directory structure."""
    agent_dir = tmp_path / "agents" / "test-agent"
    agent_dir.mkdir(parents=True)
    return agent_dir


def _sample_command(
    command_id: str = "cmd-001",
    command_type: CommandType = CommandType.RERUN,
    issued_at: str = "2026-03-07T12:00:00+00:00",
    reason: str = "DRC regression",
) -> AgentCommand:
    return AgentCommand(
        command_id=command_id,
        command_type=command_type,
        issued_at=issued_at,
        args={"stage": "pcb"},
        reason=reason,
    )


# ---------------------------------------------------------------------------
# _command_to_dict / _command_from_dict roundtrip
# ---------------------------------------------------------------------------


def test_command_to_dict_from_dict_roundtrip() -> None:
    original = _sample_command()
    d = _command_to_dict(original)
    restored = _command_from_dict(d)

    assert restored.command_id == original.command_id
    assert restored.command_type == original.command_type
    assert restored.issued_at == original.issued_at
    assert restored.args == original.args
    assert restored.reason == original.reason
    assert restored.acknowledged == original.acknowledged


def test_command_roundtrip_with_acknowledged() -> None:
    original = AgentCommand(
        command_id="cmd-ack",
        command_type=CommandType.BUG_UPDATE,
        issued_at="2026-03-07T14:00:00+00:00",
        args={"bug_id": "bug-1", "new_status": "fixed"},
        reason="Fixed in commit abc123",
        acknowledged=True,
    )
    d = _command_to_dict(original)
    restored = _command_from_dict(d)

    assert restored.acknowledged is True
    assert restored.command_type == CommandType.BUG_UPDATE
    assert restored.args["bug_id"] == "bug-1"


# ---------------------------------------------------------------------------
# load_commands — file missing
# ---------------------------------------------------------------------------


def test_load_commands_returns_empty_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent" / "commands.json"
    result = load_commands(missing)
    assert result == ()


# ---------------------------------------------------------------------------
# save_commands / load_commands roundtrip
# ---------------------------------------------------------------------------


def test_save_load_commands_roundtrip(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    path = agent_dir / "commands.json"

    cmd1 = _sample_command(command_id="cmd-001")
    cmd2 = _sample_command(
        command_id="cmd-002", command_type=CommandType.RELOAD, reason="hot reload",
    )

    save_commands((cmd1, cmd2), path)
    loaded = load_commands(path)

    assert len(loaded) == 2
    assert loaded[0].command_id == "cmd-001"
    assert loaded[1].command_id == "cmd-002"
    assert loaded[1].command_type == CommandType.RELOAD


# ---------------------------------------------------------------------------
# issue_command appends to existing
# ---------------------------------------------------------------------------


def test_issue_command_appends_to_existing(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    path = agent_dir / "commands.json"

    first = _sample_command(command_id="cmd-first")
    save_commands((first,), path)

    second = _sample_command(command_id="cmd-second", reason="second command")
    issue_command(agent_dir, second)

    loaded = load_commands(path)
    assert len(loaded) == 2
    assert loaded[0].command_id == "cmd-first"
    assert loaded[1].command_id == "cmd-second"


# ---------------------------------------------------------------------------
# acknowledge_command marks as acknowledged
# ---------------------------------------------------------------------------


def test_acknowledge_command_marks_acknowledged(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    path = agent_dir / "commands.json"

    cmd = _sample_command(command_id="cmd-to-ack")
    save_commands((cmd,), path)

    assert load_commands(path)[0].acknowledged is False

    acknowledge_command(agent_dir, "cmd-to-ack")

    loaded = load_commands(path)
    assert len(loaded) == 1
    assert loaded[0].acknowledged is True
    assert loaded[0].command_id == "cmd-to-ack"


# ---------------------------------------------------------------------------
# acknowledge_command raises AgentError for unknown id
# ---------------------------------------------------------------------------


def test_acknowledge_command_raises_for_unknown_id(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    path = agent_dir / "commands.json"

    cmd = _sample_command(command_id="cmd-known")
    save_commands((cmd,), path)

    with pytest.raises(AgentError, match="not found"):
        acknowledge_command(agent_dir, "cmd-nonexistent")
