"""Tests for kicad_pipeline.agents.models."""

from __future__ import annotations

import dataclasses

import pytest

from kicad_pipeline.agents.models import (
    AgentCommand,
    AgentRegistration,
    AgentRegistry,
    AgentState,
    AgentStatus,
    BugReport,
    BugSeverity,
    BugStatus,
    CommandType,
    DRCSummary,
    PipelineVersion,
    RunOutcome,
    RunRecord,
)

# ---------------------------------------------------------------------------
# Enum value tests
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_registered(self) -> None:
        assert AgentState.REGISTERED.value == "registered"

    def test_idle(self) -> None:
        assert AgentState.IDLE.value == "idle"

    def test_running(self) -> None:
        assert AgentState.RUNNING.value == "running"

    def test_awaiting_fix(self) -> None:
        assert AgentState.AWAITING_FIX.value == "awaiting_fix"

    def test_error(self) -> None:
        assert AgentState.ERROR.value == "error"

    def test_completed(self) -> None:
        assert AgentState.COMPLETED.value == "completed"

    def test_member_count(self) -> None:
        assert len(AgentState) == 6


class TestBugSeverity:
    def test_critical(self) -> None:
        assert BugSeverity.CRITICAL.value == "critical"

    def test_high(self) -> None:
        assert BugSeverity.HIGH.value == "high"

    def test_medium(self) -> None:
        assert BugSeverity.MEDIUM.value == "medium"

    def test_low(self) -> None:
        assert BugSeverity.LOW.value == "low"

    def test_member_count(self) -> None:
        assert len(BugSeverity) == 4


class TestBugStatus:
    def test_open(self) -> None:
        assert BugStatus.OPEN.value == "open"

    def test_acknowledged(self) -> None:
        assert BugStatus.ACKNOWLEDGED.value == "acknowledged"

    def test_fixed(self) -> None:
        assert BugStatus.FIXED.value == "fixed"

    def test_wont_fix(self) -> None:
        assert BugStatus.WONT_FIX.value == "wont_fix"

    def test_member_count(self) -> None:
        assert len(BugStatus) == 4


class TestRunOutcome:
    def test_success(self) -> None:
        assert RunOutcome.SUCCESS.value == "success"

    def test_drc_errors(self) -> None:
        assert RunOutcome.DRC_ERRORS.value == "drc_errors"

    def test_build_failure(self) -> None:
        assert RunOutcome.BUILD_FAILURE.value == "build_failure"

    def test_validation_failure(self) -> None:
        assert RunOutcome.VALIDATION_FAILURE.value == "validation_failure"

    def test_member_count(self) -> None:
        assert len(RunOutcome) == 4


class TestCommandType:
    def test_rerun(self) -> None:
        assert CommandType.RERUN.value == "rerun"

    def test_bug_update(self) -> None:
        assert CommandType.BUG_UPDATE.value == "bug_update"

    def test_reload(self) -> None:
        assert CommandType.RELOAD.value == "reload"

    def test_optimize(self) -> None:
        assert CommandType.OPTIMIZE.value == "optimize"

    def test_apply_optimization(self) -> None:
        assert CommandType.APPLY_OPTIMIZATION.value == "apply_optimization"

    def test_member_count(self) -> None:
        assert len(CommandType) == 5


# ---------------------------------------------------------------------------
# Dataclass instantiation + immutability
# ---------------------------------------------------------------------------


class TestPipelineVersion:
    def test_instantiation(self) -> None:
        pv = PipelineVersion(git_hash="abc123", git_tag="v0.1.0", timestamp="2026-03-07T00:00:00Z")
        assert pv.git_hash == "abc123"
        assert pv.git_tag == "v0.1.0"
        assert pv.timestamp == "2026-03-07T00:00:00Z"

    def test_frozen(self) -> None:
        pv = PipelineVersion(git_hash="abc123", git_tag="v0.1.0", timestamp="2026-03-07T00:00:00Z")
        with pytest.raises(dataclasses.FrozenInstanceError):
            pv.git_hash = "new"  # type: ignore[misc]


class TestDRCSummary:
    def test_instantiation(self) -> None:
        ds = DRCSummary(total_violations=10, errors=3, warnings=5, unconnected=2)
        assert ds.total_violations == 10
        assert ds.errors == 3
        assert ds.warnings == 5
        assert ds.unconnected == 2

    def test_frozen(self) -> None:
        ds = DRCSummary(total_violations=0, errors=0, warnings=0, unconnected=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ds.total_violations = 99  # type: ignore[misc]


class TestBugReport:
    @pytest.fixture()
    def bug(self) -> BugReport:
        return BugReport(
            bug_id="BUG-001",
            title="Bad pad clearance",
            severity=BugSeverity.HIGH,
            status=BugStatus.OPEN,
            description="Pad clearance too small on U1",
            pipeline_module="pcb.placement",
            pipeline_function="place_component",
            reported_at="2026-03-07T12:00:00Z",
        )

    def test_instantiation(self, bug: BugReport) -> None:
        assert bug.bug_id == "BUG-001"
        assert bug.severity is BugSeverity.HIGH
        assert bug.status is BugStatus.OPEN

    def test_defaults(self, bug: BugReport) -> None:
        assert bug.resolved_at is None
        assert bug.fix_commit is None

    def test_frozen(self, bug: BugReport) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            bug.status = BugStatus.FIXED  # type: ignore[misc]


class TestRunRecord:
    @pytest.fixture()
    def record(self) -> RunRecord:
        return RunRecord(
            run_id="run-001",
            started_at="2026-03-07T12:00:00Z",
            completed_at="2026-03-07T12:05:00Z",
            outcome=RunOutcome.SUCCESS,
            pipeline_version="v0.1.0",
        )

    def test_instantiation(self, record: RunRecord) -> None:
        assert record.run_id == "run-001"
        assert record.outcome is RunOutcome.SUCCESS

    def test_defaults(self, record: RunRecord) -> None:
        assert record.drc_summary is None
        assert record.stages_completed == ()
        assert record.error_message is None

    def test_frozen(self, record: RunRecord) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.outcome = RunOutcome.DRC_ERRORS  # type: ignore[misc]


class TestAgentRegistration:
    @pytest.fixture()
    def reg(self) -> AgentRegistration:
        return AgentRegistration(
            agent_id="agent-001",
            project_path="/tmp/project",
            project_name="test-board",
            description="Test board agent",
            registered_at="2026-03-07T12:00:00Z",
            last_seen="2026-03-07T12:00:00Z",
            state=AgentState.IDLE,
        )

    def test_instantiation(self, reg: AgentRegistration) -> None:
        assert reg.agent_id == "agent-001"
        assert reg.state is AgentState.IDLE

    def test_defaults(self, reg: AgentRegistration) -> None:
        assert reg.active_variant is None

    def test_frozen(self, reg: AgentRegistration) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            reg.state = AgentState.RUNNING  # type: ignore[misc]


class TestAgentStatus:
    @pytest.fixture()
    def status(self) -> AgentStatus:
        return AgentStatus(
            agent_id="agent-001",
            state=AgentState.RUNNING,
            updated_at="2026-03-07T12:00:00Z",
            pipeline_version="v0.1.0",
        )

    def test_instantiation(self, status: AgentStatus) -> None:
        assert status.agent_id == "agent-001"
        assert status.state is AgentState.RUNNING

    def test_defaults(self, status: AgentStatus) -> None:
        assert status.active_variant is None
        assert status.current_stage is None
        assert status.bugs == ()
        assert status.runs == ()
        assert status.message == ""
        assert status.needs_pipeline_update is False

    def test_frozen(self, status: AgentStatus) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            status.state = AgentState.COMPLETED  # type: ignore[misc]


class TestAgentCommand:
    @pytest.fixture()
    def cmd(self) -> AgentCommand:
        return AgentCommand(
            command_id="cmd-001",
            command_type=CommandType.RERUN,
            issued_at="2026-03-07T12:00:00Z",
        )

    def test_instantiation(self, cmd: AgentCommand) -> None:
        assert cmd.command_id == "cmd-001"
        assert cmd.command_type is CommandType.RERUN

    def test_args_default_factory_produces_empty_dict(self, cmd: AgentCommand) -> None:
        assert cmd.args == {}
        assert isinstance(cmd.args, dict)

    def test_args_default_is_independent_per_instance(self) -> None:
        cmd1 = AgentCommand(
            command_id="cmd-a", command_type=CommandType.RERUN, issued_at="t1"
        )
        cmd2 = AgentCommand(
            command_id="cmd-b", command_type=CommandType.RELOAD, issued_at="t2"
        )
        assert cmd1.args is not cmd2.args

    def test_defaults(self, cmd: AgentCommand) -> None:
        assert cmd.reason == ""
        assert cmd.acknowledged is False

    def test_frozen(self, cmd: AgentCommand) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            cmd.acknowledged = True  # type: ignore[misc]


class TestAgentRegistry:
    def test_instantiation_all_defaults(self) -> None:
        registry = AgentRegistry()
        assert registry.schema_version == 1
        assert registry.pipeline_project_path == ""
        assert registry.pipeline_version is None
        assert registry.agents == ()
        assert registry.updated_at == ""

    def test_instantiation_with_values(self) -> None:
        pv = PipelineVersion(git_hash="abc", git_tag="v1", timestamp="t")
        reg = AgentRegistration(
            agent_id="a1",
            project_path="/p",
            project_name="n",
            description="d",
            registered_at="t",
            last_seen="t",
            state=AgentState.REGISTERED,
        )
        registry = AgentRegistry(
            schema_version=2,
            pipeline_project_path="/src",
            pipeline_version=pv,
            agents=(reg,),
            updated_at="2026-03-07",
        )
        assert registry.schema_version == 2
        assert registry.pipeline_version is pv
        assert len(registry.agents) == 1

    def test_frozen(self) -> None:
        registry = AgentRegistry()
        with pytest.raises(dataclasses.FrozenInstanceError):
            registry.schema_version = 99  # type: ignore[misc]
