"""Pipeline-side agent monitor.

Scans the agent registry, finds open bugs, detects stale agents,
and issues commands to board agents.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from kicad_pipeline.agents.commands import issue_command
from kicad_pipeline.agents.models import (
    AgentCommand,
    AgentRegistration,
    AgentState,
    AgentStatus,
    BugReport,
    BugStatus,
    CommandType,
    PipelineVersion,
)
from kicad_pipeline.agents.registry import (
    get_pipeline_version,
    get_registry_dir,
    load_registry,
    save_pipeline_version,
    save_registry,
)
from kicad_pipeline.agents.status import load_status
from kicad_pipeline.constants import AGENT_STATUS_FILENAME

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class AgentMonitor:
    """Pipeline-side interface for monitoring board agents."""

    def __init__(self, registry_dir: Path | None = None) -> None:
        self._dir = registry_dir or get_registry_dir()

    @property
    def registry_dir(self) -> Path:
        """The registry directory this monitor operates on."""
        return self._dir

    def scan_all(self) -> tuple[tuple[AgentRegistration, AgentStatus | None], ...]:
        """Scan all registered agents and load their status files.

        Returns:
            Tuple of ``(registration, status_or_None)`` pairs.
        """
        registry = load_registry(self._dir / "registry.json")
        results: list[tuple[AgentRegistration, AgentStatus | None]] = []
        for reg in registry.agents:
            status_path = self._dir / "agents" / reg.agent_id / AGENT_STATUS_FILENAME
            status: AgentStatus | None = None
            if status_path.exists():
                try:
                    status = load_status(status_path)
                except Exception:
                    log.warning("Failed to load status for agent %s", reg.agent_id)
            results.append((reg, status))
        return tuple(results)

    def find_open_bugs(self) -> tuple[tuple[str, BugReport], ...]:
        """Find all open bugs across all agents.

        Returns:
            Tuple of ``(agent_id, bug)`` pairs where bug status is OPEN.
        """
        bugs: list[tuple[str, BugReport]] = []
        for reg, status in self.scan_all():
            if status is None:
                continue
            for bug in status.bugs:
                if bug.status == BugStatus.OPEN:
                    bugs.append((reg.agent_id, bug))
        return tuple(bugs)

    def find_stale_agents(self, stale_hours: float = 24.0) -> tuple[AgentRegistration, ...]:
        """Find agents that haven't reported in recently.

        Args:
            stale_hours: Hours since last_seen to consider stale.

        Returns:
            Tuple of stale registrations.
        """
        registry = load_registry(self._dir / "registry.json")
        now = datetime.now(tz=timezone.utc)
        stale: list[AgentRegistration] = []
        for reg in registry.agents:
            if reg.state == AgentState.COMPLETED:
                continue
            try:
                last = datetime.fromisoformat(reg.last_seen)
                if (now - last).total_seconds() > stale_hours * 3600:
                    stale.append(reg)
            except ValueError:
                stale.append(reg)
        return tuple(stale)

    def issue_rerun(self, agent_id: str, stage: str, reason: str) -> None:
        """Issue a rerun command to a board agent.

        Args:
            agent_id: Target agent ID.
            stage: The pipeline stage to rerun.
            reason: Why the rerun is needed.
        """
        agent_dir = self._dir / "agents" / agent_id
        cmd = AgentCommand(
            command_id=str(uuid.uuid4()),
            command_type=CommandType.RERUN,
            issued_at=_now_iso(),
            args={"stage": stage},
            reason=reason,
        )
        issue_command(agent_dir, cmd)

    def issue_bug_update(
        self,
        agent_id: str,
        bug_id: str,
        new_status: BugStatus,
        fix_commit: str | None = None,
    ) -> None:
        """Notify a board agent that a bug status has changed.

        Args:
            agent_id: Target agent ID.
            bug_id: The bug to update.
            new_status: The new status for the bug.
            fix_commit: Git commit hash that fixes the bug, if applicable.
        """
        agent_dir = self._dir / "agents" / agent_id
        args: dict[str, str] = {"bug_id": bug_id, "new_status": new_status.value}
        if fix_commit is not None:
            args["fix_commit"] = fix_commit
        cmd = AgentCommand(
            command_id=str(uuid.uuid4()),
            command_type=CommandType.BUG_UPDATE,
            issued_at=_now_iso(),
            args=args,
            reason=f"Bug {bug_id} -> {new_status.value}",
        )
        issue_command(agent_dir, cmd)

    def update_pipeline_version(self, pipeline_root: Path | None = None) -> PipelineVersion:
        """Update the pipeline version marker file.

        Args:
            pipeline_root: Root of the pipeline repo.  Defaults to cwd.

        Returns:
            The new pipeline version.
        """
        root = pipeline_root or Path.cwd()
        version = get_pipeline_version(root)
        save_pipeline_version(version, self._dir)

        # Also update the registry's version field
        registry = load_registry(self._dir / "registry.json")
        registry = replace(
            registry, pipeline_version=version, updated_at=_now_iso(),
        )
        save_registry(registry, self._dir / "registry.json")

        return version
