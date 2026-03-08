"""Tests for kicad_pipeline.agents.status persistence layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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
from kicad_pipeline.agents.status import (
    _bug_from_dict,
    _bug_to_dict,
    _drc_summary_from_dict,
    _drc_summary_to_dict,
    _run_from_dict,
    _run_to_dict,
    _status_from_dict,
    _status_to_dict,
    load_status,
    save_status,
)
from kicad_pipeline.exceptions import AgentError

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_drc_summary() -> DRCSummary:
    return DRCSummary(
        total_violations=10,
        errors=3,
        warnings=5,
        unconnected=2,
    )


@pytest.fixture()
def sample_bug() -> BugReport:
    return BugReport(
        bug_id="bug-001",
        title="Pad net assignment missing",
        severity=BugSeverity.HIGH,
        status=BugStatus.OPEN,
        description="Pads have no net numbers after build_pcb",
        pipeline_module="pcb.builder",
        pipeline_function="build_pcb",
        reported_at="2026-03-07T10:00:00+00:00",
        resolved_at=None,
        fix_commit=None,
    )


@pytest.fixture()
def sample_run(sample_drc_summary: DRCSummary) -> RunRecord:
    return RunRecord(
        run_id="run-abc123",
        started_at="2026-03-07T11:00:00+00:00",
        completed_at="2026-03-07T11:05:00+00:00",
        outcome=RunOutcome.DRC_ERRORS,
        pipeline_version="abc123def456",
        drc_summary=sample_drc_summary,
        stages_completed=("requirements", "schematic", "pcb"),
        error_message=None,
    )


@pytest.fixture()
def sample_status(sample_bug: BugReport, sample_run: RunRecord) -> AgentStatus:
    return AgentStatus(
        agent_id="agent-led-blinker",
        state=AgentState.RUNNING,
        updated_at="2026-03-07T12:00:00+00:00",
        pipeline_version="abc123def456",
        active_variant="smd-0603",
        current_stage="pcb",
        bugs=(sample_bug,),
        runs=(sample_run,),
        message="DRC run completed with errors",
        needs_pipeline_update=True,
    )


# ---------------------------------------------------------------------------
# DRCSummary roundtrip
# ---------------------------------------------------------------------------


def test_drc_summary_roundtrip(sample_drc_summary: DRCSummary) -> None:
    d = _drc_summary_to_dict(sample_drc_summary)
    restored = _drc_summary_from_dict(d)

    assert restored == sample_drc_summary
    assert restored.total_violations == 10
    assert restored.errors == 3
    assert restored.warnings == 5
    assert restored.unconnected == 2


# ---------------------------------------------------------------------------
# BugReport roundtrip
# ---------------------------------------------------------------------------


def test_bug_roundtrip(sample_bug: BugReport) -> None:
    d = _bug_to_dict(sample_bug)
    restored = _bug_from_dict(d)

    assert restored == sample_bug
    assert restored.severity == BugSeverity.HIGH
    assert restored.status == BugStatus.OPEN
    assert restored.resolved_at is None
    assert restored.fix_commit is None


def test_bug_roundtrip_with_resolution() -> None:
    bug = BugReport(
        bug_id="bug-002",
        title="Zone clearance too large",
        severity=BugSeverity.MEDIUM,
        status=BugStatus.FIXED,
        description="Zone clearance was 0.5mm instead of 0.2mm",
        pipeline_module="pcb.zones",
        pipeline_function="make_zones",
        reported_at="2026-03-06T09:00:00+00:00",
        resolved_at="2026-03-07T08:00:00+00:00",
        fix_commit="deadbeef",
    )
    d = _bug_to_dict(bug)
    restored = _bug_from_dict(d)

    assert restored == bug
    assert restored.resolved_at == "2026-03-07T08:00:00+00:00"
    assert restored.fix_commit == "deadbeef"


# ---------------------------------------------------------------------------
# RunRecord roundtrip
# ---------------------------------------------------------------------------


def test_run_roundtrip(sample_run: RunRecord) -> None:
    d = _run_to_dict(sample_run)
    restored = _run_from_dict(d)

    assert restored == sample_run
    assert restored.outcome == RunOutcome.DRC_ERRORS
    assert restored.drc_summary is not None
    assert restored.drc_summary.errors == 3
    assert restored.stages_completed == ("requirements", "schematic", "pcb")
    assert restored.error_message is None


def test_run_roundtrip_no_drc() -> None:
    run = RunRecord(
        run_id="run-xyz",
        started_at="2026-03-07T14:00:00+00:00",
        completed_at=None,
        outcome=RunOutcome.BUILD_FAILURE,
        pipeline_version="v0.5.0",
        drc_summary=None,
        stages_completed=("requirements",),
        error_message="Schematic build failed",
    )
    d = _run_to_dict(run)
    restored = _run_from_dict(d)

    assert restored == run
    assert restored.drc_summary is None
    assert restored.completed_at is None
    assert restored.error_message == "Schematic build failed"


# ---------------------------------------------------------------------------
# AgentStatus roundtrip
# ---------------------------------------------------------------------------


def test_status_roundtrip(sample_status: AgentStatus) -> None:
    d = _status_to_dict(sample_status)
    restored = _status_from_dict(d)

    assert restored == sample_status
    assert restored.agent_id == "agent-led-blinker"
    assert restored.state == AgentState.RUNNING
    assert restored.active_variant == "smd-0603"
    assert restored.current_stage == "pcb"
    assert len(restored.bugs) == 1
    assert len(restored.runs) == 1
    assert restored.needs_pipeline_update is True


def test_status_roundtrip_minimal() -> None:
    status = AgentStatus(
        agent_id="agent-minimal",
        state=AgentState.IDLE,
        updated_at="2026-03-07T12:00:00+00:00",
        pipeline_version="v0.1.0",
    )
    d = _status_to_dict(status)
    restored = _status_from_dict(d)

    assert restored == status
    assert restored.active_variant is None
    assert restored.current_stage is None
    assert restored.bugs == ()
    assert restored.runs == ()


# ---------------------------------------------------------------------------
# load_status / save_status
# ---------------------------------------------------------------------------


def test_load_save_status_roundtrip(
    tmp_path: Path, sample_status: AgentStatus
) -> None:
    path = tmp_path / "agents" / "agent-led-blinker" / "status.json"
    save_status(sample_status, path)
    loaded = load_status(path)

    assert loaded == sample_status


def test_load_status_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent" / "status.json"

    with pytest.raises(AgentError, match="Cannot read agent status"):
        load_status(missing)
