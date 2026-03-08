"""Board-agent-side reporter for the multi-agent coordination system.

A board agent uses :class:`AgentReporter` to register itself, report
bugs, record runs, and check for commands from the pipeline agent.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from kicad_pipeline.agents.commands import acknowledge_command, load_commands
from kicad_pipeline.agents.models import (
    AgentCommand,
    AgentRegistration,
    AgentState,
    AgentStatus,
    BugReport,
    BugSeverity,
    BugStatus,
    DRCSummary,
    RunOutcome,
    RunRecord,
)
from kicad_pipeline.agents.registry import (
    get_registry_dir,
    load_pipeline_version,
    load_registry,
    save_registry,
)
from kicad_pipeline.agents.status import load_status, save_status
from kicad_pipeline.constants import AGENT_COMMANDS_FILENAME, AGENT_STATUS_FILENAME

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class AgentReporter:
    """Board-agent-side interface for reporting to the pipeline agent."""

    def __init__(
        self,
        agent_id: str,
        registry_dir: Path | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._dir = registry_dir or get_registry_dir()
        self._agent_dir = self._dir / "agents" / agent_id

    @property
    def agent_id(self) -> str:
        """The agent's unique identifier."""
        return self._agent_id

    @property
    def agent_dir(self) -> Path:
        """The agent's directory within the registry."""
        return self._agent_dir

    def register(
        self,
        project_path: str,
        project_name: str,
        variant: str | None = None,
        description: str = "",
    ) -> None:
        """Register this agent in the global registry.

        Args:
            project_path: Absolute path to the board project.
            project_name: Human-readable project name.
            variant: Active variant name, if any.
            description: Optional description.
        """
        self._agent_dir.mkdir(parents=True, exist_ok=True)
        now = _now_iso()

        reg = AgentRegistration(
            agent_id=self._agent_id,
            project_path=project_path,
            project_name=project_name,
            description=description,
            registered_at=now,
            last_seen=now,
            state=AgentState.REGISTERED,
            active_variant=variant,
        )

        # Update global registry
        registry = load_registry(self._dir / "registry.json")
        existing = tuple(a for a in registry.agents if a.agent_id != self._agent_id)
        registry = replace(
            registry, agents=(*existing, reg), updated_at=now,
        )
        save_registry(registry, self._dir / "registry.json")

        # Create initial status file
        pv = load_pipeline_version(self._dir)
        status = AgentStatus(
            agent_id=self._agent_id,
            state=AgentState.REGISTERED,
            updated_at=now,
            pipeline_version=pv.git_hash if pv else "",
            active_variant=variant,
        )
        save_status(status, self._agent_dir / AGENT_STATUS_FILENAME)
        log.info("Registered agent %s for project %s", self._agent_id, project_name)

    def update_state(self, state: AgentState, message: str = "") -> None:
        """Update this agent's state and optional message.

        Args:
            state: New agent state.
            message: Optional status message.
        """
        status = self._load_status()
        status = replace(
            status, state=state, message=message, updated_at=_now_iso(),
        )
        save_status(status, self._agent_dir / AGENT_STATUS_FILENAME)
        self._update_registry_state(state)

    def report_bug(
        self,
        title: str,
        severity: BugSeverity,
        module: str,
        description: str,
        function: str = "",
    ) -> str:
        """Report a pipeline bug.

        Args:
            title: Short bug title.
            severity: Bug severity level.
            module: Pipeline module where the bug occurs.
            description: Detailed description.
            function: Specific function name, if known.

        Returns:
            The generated bug ID.
        """
        bug_id = str(uuid.uuid4())[:8]
        bug = BugReport(
            bug_id=bug_id,
            title=title,
            severity=severity,
            status=BugStatus.OPEN,
            description=description,
            pipeline_module=module,
            pipeline_function=function,
            reported_at=_now_iso(),
        )

        status = self._load_status()
        status = replace(
            status, bugs=(*status.bugs, bug), updated_at=_now_iso(),
        )
        save_status(status, self._agent_dir / AGENT_STATUS_FILENAME)
        log.info("Reported bug %s: %s", bug_id, title)
        return bug_id

    def record_run(
        self,
        outcome: RunOutcome,
        drc_summary: DRCSummary | None = None,
        stages: tuple[str, ...] = (),
        error_message: str | None = None,
    ) -> str:
        """Record a pipeline run.

        Args:
            outcome: The run outcome.
            drc_summary: Optional DRC results summary.
            stages: Stages completed during this run.
            error_message: Error message if the run failed.

        Returns:
            The generated run ID.
        """
        run_id = str(uuid.uuid4())[:8]
        now = _now_iso()

        pv = load_pipeline_version(self._dir)
        run = RunRecord(
            run_id=run_id,
            started_at=now,
            completed_at=now,
            outcome=outcome,
            pipeline_version=pv.git_hash if pv else "",
            drc_summary=drc_summary,
            stages_completed=stages,
            error_message=error_message,
        )

        status = self._load_status()
        status = replace(
            status, runs=(*status.runs, run), updated_at=now,
        )
        save_status(status, self._agent_dir / AGENT_STATUS_FILENAME)
        log.info("Recorded run %s: %s", run_id, outcome.value)
        return run_id

    def check_commands(self) -> tuple[AgentCommand, ...]:
        """Check for pending (unacknowledged) commands from the pipeline.

        Returns:
            Tuple of unacknowledged commands.
        """
        path = self._agent_dir / AGENT_COMMANDS_FILENAME
        all_cmds = load_commands(path)
        return tuple(c for c in all_cmds if not c.acknowledged)

    def acknowledge_command(self, command_id: str) -> None:
        """Mark a command as acknowledged.

        Args:
            command_id: The command to acknowledge.
        """
        acknowledge_command(self._agent_dir, command_id)

    def check_pipeline_update(self) -> bool:
        """Check if the pipeline version has changed since last status update.

        Returns:
            True if the pipeline has been updated.
        """
        pv = load_pipeline_version(self._dir)
        if pv is None:
            return False
        status = self._load_status()
        return pv.git_hash != status.pipeline_version

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _load_status(self) -> AgentStatus:
        """Load this agent's status file, creating a default if missing."""
        path = self._agent_dir / AGENT_STATUS_FILENAME
        if path.exists():
            return load_status(path)
        return AgentStatus(
            agent_id=self._agent_id,
            state=AgentState.REGISTERED,
            updated_at=_now_iso(),
            pipeline_version="",
        )

    def _update_registry_state(self, state: AgentState) -> None:
        """Update this agent's state in the global registry."""
        registry = load_registry(self._dir / "registry.json")
        now = _now_iso()
        updated_agents: list[AgentRegistration] = []
        for reg in registry.agents:
            if reg.agent_id == self._agent_id:
                updated_agents.append(replace(reg, state=state, last_seen=now))
            else:
                updated_agents.append(reg)
        registry = replace(registry, agents=tuple(updated_agents), updated_at=now)
        save_registry(registry, self._dir / "registry.json")
