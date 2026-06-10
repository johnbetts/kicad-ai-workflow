"""Framework <-> board project pipeline coordination.

Manages the upstream (board -> framework) bug reporting flow and the
downstream (framework -> board) fix distribution flow.

Upstream: Board agents file bugs -> PM triages -> Developer fixes -> Tester certifies
Downstream: Framework fix certified -> PM issues RERUN to affected boards -> Board rebuilds
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from kicad_pipeline.agents.planner.roles import (
    AgentRole,
    Evidence,
    EvidenceType,
    Task,
    TaskBoard,
    TaskStatus,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpstreamBugReport:
    """A bug report from a board agent to the framework.

    Named ``UpstreamBugReport`` to avoid collision with
    ``kicad_pipeline.agents.models.BugReport`` which models the
    registry-level bug record with different fields.
    """

    bug_id: str
    source_board: str  # Board project name
    classification: str  # "CORE" or "DEPLOYMENT"
    title: str
    description: str
    severity: str  # "critical", "major", "minor"
    affected_components: tuple[str, ...]  # Component refs
    evidence_paths: tuple[str, ...]  # Render PNGs, test output files
    reported_by: str  # Agent ID


# Convenience alias matching the conceptual name in the spec.
BugReport = UpstreamBugReport


# ---------------------------------------------------------------------------
# Priority mapping
# ---------------------------------------------------------------------------

_SEVERITY_TO_PRIORITY: dict[str, str] = {
    "critical": "P0",
    "major": "P1",
    "minor": "P2",
}


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class PipelineCoordinator:
    """Coordinates work between framework and board project agents.

    Sits between board agents (which file bugs) and the framework's
    internal TaskBoard (where PM, Developer, Tester, Reviewer roles
    collaborate to fix and certify issues).
    """

    def __init__(self, task_board: TaskBoard) -> None:
        self._board = task_board

    # -- upstream: bug triage ------------------------------------------------

    def triage_bug_report(self, report: UpstreamBugReport) -> Task | None:
        """Triage a bug report and create a task if it is a CORE issue.

        DEPLOYMENT issues return ``None`` (handled by the board project).
        CORE issues create a Developer task on the TaskBoard.
        """
        if report.classification != "CORE":
            _log.info(
                "Bug %s is DEPLOYMENT -- stays in board project %s",
                report.bug_id,
                report.source_board,
            )
            return None

        priority = _SEVERITY_TO_PRIORITY.get(report.severity, "P2")

        task = self._board.create_task(
            title=f"[{report.bug_id}] {report.title}",
            description=(
                f"Source: {report.source_board}\n"
                f"Severity: {report.severity}\n"
                f"Components: {', '.join(report.affected_components)}\n\n"
                f"{report.description}"
            ),
            owner_role=AgentRole.DEVELOPER,
            priority=priority,
            created_by=f"pm:triage:{report.reported_by}",
            tags=("core-bug", report.source_board),
        )
        _log.info(
            "Triaged %s as %s task %s",
            report.bug_id,
            priority,
            task.task_id,
        )
        return task

    # -- handoffs ------------------------------------------------------------

    def developer_handoff_to_tester(
        self,
        task_id: str,
        agent_id: str,
        test_output: str,
        code_diff: str,
    ) -> Task:
        """Developer completes work; hands off to tester with evidence."""
        now = _now_iso()
        evidence = (
            Evidence(EvidenceType.TEST_RESULT, test_output, agent_id, now),
            Evidence(EvidenceType.CODE_DIFF, code_diff, agent_id, now),
        )
        return self._board.transition(
            task_id,
            TaskStatus.TESTING,
            agent_id,
            evidence,
        )

    def tester_handoff_to_reviewer(
        self,
        task_id: str,
        agent_id: str,
        test_output: str,
        render_2d: str,
        render_3d: str,
    ) -> Task:
        """Tester validates and hands off to reviewer with visual evidence."""
        now = _now_iso()
        evidence = (
            Evidence(EvidenceType.TEST_RESULT, test_output, agent_id, now),
            Evidence(EvidenceType.RENDER_2D, render_2d, agent_id, now),
            Evidence(EvidenceType.RENDER_3D, render_3d, agent_id, now),
        )
        return self._board.transition(
            task_id,
            TaskStatus.REVIEW,
            agent_id,
            evidence,
        )

    def reviewer_sign_off(
        self,
        task_id: str,
        agent_id: str,
        review_findings: str,
    ) -> Task:
        """Reviewer approves -- task moves to DONE."""
        now = _now_iso()
        evidence = (
            Evidence(EvidenceType.REVIEW_FINDING, review_findings, agent_id, now),
        )
        return self._board.transition(
            task_id,
            TaskStatus.DONE,
            agent_id,
            evidence,
        )

    def reviewer_reject(
        self,
        task_id: str,
        agent_id: str,
        reason: str,
    ) -> Task:
        """Reviewer rejects -- task goes back to developer."""
        return self._board.transition(
            task_id,
            TaskStatus.REJECTED,
            agent_id,
            rejection_reason=reason,
        )

    # -- downstream: fix distribution ----------------------------------------

    def notify_boards_of_fix(
        self,
        task: Task,
        affected_boards: tuple[str, ...],
    ) -> list[str]:
        """After a framework fix is certified, notify affected board agents.

        Returns list of board names that were notified.  In practice this
        would use the agent command queue from
        ``kicad_pipeline.agents.commands`` to issue RERUN commands.
        """
        notified: list[str] = []
        for board in affected_boards:
            _log.info(
                "Notifying board %s of fix: %s",
                board,
                task.title,
            )
            # Board notification dispatch — future: commands.issue_command()
            notified.append(board)
        return notified

    # -- queries -------------------------------------------------------------

    def pipeline_status(self) -> str:
        """Human-readable pipeline status summary."""
        lines: list[str] = [self._board.summary(), ""]

        by_role: dict[str, list[Task]] = {}
        for task in self._board.open_tasks():
            role = task.owner_role.value
            by_role.setdefault(role, []).append(task)

        for role, tasks in sorted(by_role.items()):
            lines.append(f"  {role}: {len(tasks)} open tasks")
            for t in tasks[:5]:
                lines.append(
                    f"    [{t.priority}] {t.status.value}: {t.title}"
                )

        return "\n".join(lines)
