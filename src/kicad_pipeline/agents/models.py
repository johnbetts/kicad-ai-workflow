"""Data models for multi-agent coordination.

Defines the enums and frozen dataclasses used by both pipeline agents
(monitor side) and board agents (reporter side) to communicate via
the file-based registry at ``~/.claude/kicad-agents/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentState(Enum):
    """Lifecycle state of a board agent."""

    REGISTERED = "registered"
    IDLE = "idle"
    RUNNING = "running"
    AWAITING_FIX = "awaiting_fix"
    ERROR = "error"
    COMPLETED = "completed"


class BugSeverity(Enum):
    """Severity classification for pipeline bugs reported by board agents."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BugStatus(Enum):
    """Lifecycle status of a bug report."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    FIXED = "fixed"
    WONT_FIX = "wont_fix"


class RunOutcome(Enum):
    """Outcome of a single pipeline run."""

    SUCCESS = "success"
    DRC_ERRORS = "drc_errors"
    BUILD_FAILURE = "build_failure"
    VALIDATION_FAILURE = "validation_failure"


class CommandType(Enum):
    """Type of command the pipeline agent can issue to a board agent."""

    RERUN = "rerun"
    BUG_UPDATE = "bug_update"
    RELOAD = "reload"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineVersion:
    """Snapshot of the pipeline source version."""

    git_hash: str
    git_tag: str
    timestamp: str


@dataclass(frozen=True)
class DRCSummary:
    """Condensed DRC results from a pipeline run."""

    total_violations: int
    errors: int
    warnings: int
    unconnected: int


@dataclass(frozen=True)
class BugReport:
    """A bug filed by a board agent against the pipeline."""

    bug_id: str
    title: str
    severity: BugSeverity
    status: BugStatus
    description: str
    pipeline_module: str
    pipeline_function: str
    reported_at: str
    resolved_at: str | None = None
    fix_commit: str | None = None


@dataclass(frozen=True)
class RunRecord:
    """Record of a single pipeline execution."""

    run_id: str
    started_at: str
    completed_at: str | None
    outcome: RunOutcome
    pipeline_version: str
    drc_summary: DRCSummary | None = None
    stages_completed: tuple[str, ...] = ()
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Agent-level records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRegistration:
    """Lightweight entry stored in the global registry."""

    agent_id: str
    project_path: str
    project_name: str
    description: str
    registered_at: str
    last_seen: str
    state: AgentState
    active_variant: str | None = None


@dataclass(frozen=True)
class AgentStatus:
    """Full status file owned by a board agent."""

    agent_id: str
    state: AgentState
    updated_at: str
    pipeline_version: str
    active_variant: str | None = None
    current_stage: str | None = None
    bugs: tuple[BugReport, ...] = ()
    runs: tuple[RunRecord, ...] = ()
    message: str = ""
    needs_pipeline_update: bool = False


@dataclass(frozen=True)
class AgentCommand:
    """A command issued by the pipeline agent to a board agent."""

    command_id: str
    command_type: CommandType
    issued_at: str
    args: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    acknowledged: bool = False


@dataclass(frozen=True)
class AgentRegistry:
    """Top-level registry of all known board agents."""

    schema_version: int = 1
    pipeline_project_path: str = ""
    pipeline_version: PipelineVersion | None = None
    agents: tuple[AgentRegistration, ...] = ()
    updated_at: str = ""
