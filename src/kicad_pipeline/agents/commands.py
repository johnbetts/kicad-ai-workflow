"""Persistence layer for per-agent command queues.

The pipeline agent writes ``commands.json`` inside each board agent's
directory.  Board agents read and acknowledge commands.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from kicad_pipeline.agents.models import AgentCommand, CommandType
from kicad_pipeline.constants import AGENT_COMMANDS_FILENAME
from kicad_pipeline.exceptions import AgentError

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _command_to_dict(cmd: AgentCommand) -> dict[str, object]:
    return {
        "command_id": cmd.command_id,
        "command_type": cmd.command_type.value,
        "issued_at": cmd.issued_at,
        "args": dict(cmd.args),
        "reason": cmd.reason,
        "acknowledged": cmd.acknowledged,
    }


def _command_from_dict(data: dict[str, object]) -> AgentCommand:
    args_raw = data.get("args", {})
    if not isinstance(args_raw, dict):
        args_raw = {}
    return AgentCommand(
        command_id=str(data["command_id"]),
        command_type=CommandType(str(data["command_type"])),
        issued_at=str(data["issued_at"]),
        args={str(k): str(v) for k, v in args_raw.items()},
        reason=str(data.get("reason", "")),
        acknowledged=bool(data.get("acknowledged", False)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_commands(path: Path) -> tuple[AgentCommand, ...]:
    """Load command queue from a ``commands.json`` file.

    Args:
        path: Path to the commands file.

    Returns:
        Tuple of commands, possibly empty.
    """
    if not path.exists():
        return ()

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentError(f"Cannot read commands file {path}: {exc}") from exc

    if not isinstance(data, list):
        return ()

    return tuple(_command_from_dict(d) for d in data if isinstance(d, dict))


def save_commands(commands: tuple[AgentCommand, ...], path: Path) -> None:
    """Persist command queue atomically.

    Args:
        commands: The commands to save.
        path: Path to write ``commands.json``.
    """
    import contextlib
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump([_command_to_dict(c) for c in commands], f, indent=2)
            f.write("\n")
        os.replace(tmp_path, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    log.info("Saved %d commands to %s", len(commands), path)


def issue_command(agent_dir: Path, command: AgentCommand) -> None:
    """Append a command to a board agent's command queue.

    Args:
        agent_dir: The agent's directory (e.g. ``agents/{agent-id}/``).
        command: The command to issue.
    """
    path = agent_dir / AGENT_COMMANDS_FILENAME
    existing = load_commands(path)
    save_commands((*existing, command), path)
    log.info("Issued command %s to %s", command.command_id, agent_dir.name)


def acknowledge_command(agent_dir: Path, command_id: str) -> None:
    """Mark a command as acknowledged in the agent's command queue.

    Args:
        agent_dir: The agent's directory.
        command_id: The ID of the command to acknowledge.

    Raises:
        AgentError: If the command is not found.
    """
    path = agent_dir / AGENT_COMMANDS_FILENAME
    commands = load_commands(path)
    updated: list[AgentCommand] = []
    found = False
    for cmd in commands:
        if cmd.command_id == command_id:
            updated.append(replace(cmd, acknowledged=True))
            found = True
        else:
            updated.append(cmd)

    if not found:
        raise AgentError(f"Command {command_id!r} not found in {path}")

    save_commands(tuple(updated), path)
    log.info("Acknowledged command %s in %s", command_id, agent_dir.name)
