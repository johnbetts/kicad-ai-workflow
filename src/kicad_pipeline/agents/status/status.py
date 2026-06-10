"""Persistence layer for per-agent status files.

Each board agent owns its ``status.json`` inside
``~/.claude/kicad-agents/agents/{agent-id}/``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from kicad_pipeline.agents.models import (
    AgentState,
    AgentStatus,
    BugReport,
    BugSeverity,
    BugStatus,
    DRCSummary,
    RunOutcome,
    RunRecord,
)
from kicad_pipeline.exceptions import AgentError

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _drc_summary_to_dict(ds: DRCSummary) -> dict[str, object]:
    return {
        "total_violations": ds.total_violations,
        "errors": ds.errors,
        "warnings": ds.warnings,
        "unconnected": ds.unconnected,
    }


def _drc_summary_from_dict(data: dict[str, object]) -> DRCSummary:
    return DRCSummary(
        total_violations=int(str(data.get("total_violations", 0))),
        errors=int(str(data.get("errors", 0))),
        warnings=int(str(data.get("warnings", 0))),
        unconnected=int(str(data.get("unconnected", 0))),
    )


def _bug_to_dict(b: BugReport) -> dict[str, object]:
    return {
        "bug_id": b.bug_id,
        "title": b.title,
        "severity": b.severity.value,
        "status": b.status.value,
        "description": b.description,
        "pipeline_module": b.pipeline_module,
        "pipeline_function": b.pipeline_function,
        "reported_at": b.reported_at,
        "resolved_at": b.resolved_at,
        "fix_commit": b.fix_commit,
    }


def _bug_from_dict(data: dict[str, object]) -> BugReport:
    resolved = data.get("resolved_at")
    fix = data.get("fix_commit")
    return BugReport(
        bug_id=str(data["bug_id"]),
        title=str(data["title"]),
        severity=BugSeverity(str(data["severity"])),
        status=BugStatus(str(data["status"])),
        description=str(data.get("description", "")),
        pipeline_module=str(data.get("pipeline_module", "")),
        pipeline_function=str(data.get("pipeline_function", "")),
        reported_at=str(data["reported_at"]),
        resolved_at=str(resolved) if resolved is not None else None,
        fix_commit=str(fix) if fix is not None else None,
    )


def _run_to_dict(r: RunRecord) -> dict[str, object]:
    drc: dict[str, object] | None = None
    if r.drc_summary is not None:
        drc = _drc_summary_to_dict(r.drc_summary)
    return {
        "run_id": r.run_id,
        "started_at": r.started_at,
        "completed_at": r.completed_at,
        "outcome": r.outcome.value,
        "pipeline_version": r.pipeline_version,
        "drc_summary": drc,
        "stages_completed": list(r.stages_completed),
        "error_message": r.error_message,
    }


def _run_from_dict(data: dict[str, object]) -> RunRecord:
    drc_data = data.get("drc_summary")
    drc: DRCSummary | None = None
    if drc_data is not None and isinstance(drc_data, dict):
        drc = _drc_summary_from_dict(drc_data)

    completed = data.get("completed_at")
    err = data.get("error_message")
    stages_raw = data.get("stages_completed", [])
    if not isinstance(stages_raw, list):
        stages_raw = []

    return RunRecord(
        run_id=str(data["run_id"]),
        started_at=str(data["started_at"]),
        completed_at=str(completed) if completed is not None else None,
        outcome=RunOutcome(str(data["outcome"])),
        pipeline_version=str(data.get("pipeline_version", "")),
        drc_summary=drc,
        stages_completed=tuple(str(s) for s in stages_raw),
        error_message=str(err) if err is not None else None,
    )


def _status_to_dict(status: AgentStatus) -> dict[str, object]:
    return {
        "agent_id": status.agent_id,
        "state": status.state.value,
        "updated_at": status.updated_at,
        "pipeline_version": status.pipeline_version,
        "active_variant": status.active_variant,
        "current_stage": status.current_stage,
        "bugs": [_bug_to_dict(b) for b in status.bugs],
        "runs": [_run_to_dict(r) for r in status.runs],
        "message": status.message,
        "needs_pipeline_update": status.needs_pipeline_update,
    }


def _status_from_dict(data: dict[str, object]) -> AgentStatus:
    bugs_raw = data.get("bugs", [])
    if not isinstance(bugs_raw, list):
        bugs_raw = []
    runs_raw = data.get("runs", [])
    if not isinstance(runs_raw, list):
        runs_raw = []

    return AgentStatus(
        agent_id=str(data["agent_id"]),
        state=AgentState(str(data.get("state", "registered"))),
        updated_at=str(data.get("updated_at", "")),
        pipeline_version=str(data.get("pipeline_version", "")),
        active_variant=str(data["active_variant"]) if data.get("active_variant") else None,
        current_stage=str(data["current_stage"]) if data.get("current_stage") else None,
        bugs=tuple(_bug_from_dict(b) for b in bugs_raw if isinstance(b, dict)),
        runs=tuple(_run_from_dict(r) for r in runs_raw if isinstance(r, dict)),
        message=str(data.get("message", "")),
        needs_pipeline_update=bool(data.get("needs_pipeline_update", False)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_status(path: Path) -> AgentStatus:
    """Load an agent status file.

    Args:
        path: Path to the ``status.json`` file.

    Returns:
        The deserialized :class:`AgentStatus`.

    Raises:
        AgentError: If the file cannot be read or parsed.
    """
    try:
        raw = path.read_text(encoding="utf-8")
        data: dict[str, object] = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentError(f"Cannot read agent status {path}: {exc}") from exc

    return _status_from_dict(data)


def save_status(status: AgentStatus, path: Path) -> None:
    """Persist an agent status file atomically.

    Args:
        status: The status to save.
        path: Path to write ``status.json``.
    """
    from kicad_pipeline.agents.registry import _write_atomic

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, _status_to_dict(status))
    log.info("Saved agent status to %s", path)
