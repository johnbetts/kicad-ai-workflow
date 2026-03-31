"""Multi-agent role coordination with task tracking and evidence gates.

Defines PM, Developer, Tester, Reviewer roles with a persistent TaskBoard
for tracking tasks through the Development -> Testing -> Review pipeline.
"""

from __future__ import annotations

import enum
import json
import logging
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentRole(enum.Enum):
    """Agent roles in the development pipeline."""

    PM = "project_manager"
    DEVELOPER = "developer"
    TESTER = "tester"
    REVIEWER = "reviewer"
    BOARD_AGENT = "board_agent"


class TaskStatus(enum.Enum):
    """Task lifecycle states."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"
    REVIEW = "review"
    DONE = "done"
    BLOCKED = "blocked"
    REJECTED = "rejected"


class EvidenceType(enum.Enum):
    """Types of evidence that can be attached to a task."""

    TEST_RESULT = "test_result"
    RENDER_2D = "render_2d"
    RENDER_3D = "render_3d"
    REVIEW_FINDING = "review_finding"
    CODE_DIFF = "code_diff"
    LINT_RESULT = "lint_result"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """A piece of evidence attached to a task."""

    evidence_type: EvidenceType
    content: str
    created_by: str
    timestamp: str


@dataclass(frozen=True)
class Task:
    """A tracked work item flowing through the agent pipeline."""

    task_id: str
    title: str
    description: str
    owner_role: AgentRole
    priority: str
    status: TaskStatus = TaskStatus.OPEN
    assigned_to: str | None = None
    evidence: tuple[Evidence, ...] = ()
    depends_on: tuple[str, ...] = ()
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""
    tags: tuple[str, ...] = ()
    rejection_reason: str | None = None

    def to_json(self) -> str:
        """Serialize to a JSON string with enums converted to values."""
        d = asdict(self)
        d["owner_role"] = self.owner_role.value
        d["status"] = self.status.value
        d["evidence"] = [
            {**asdict(e), "evidence_type": e.evidence_type.value}
            for e in self.evidence
        ]
        return json.dumps(d, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Task:
        """Deserialize from a plain dict (e.g. parsed JSON)."""
        evidence_raw: list[dict[str, str]] = d.get("evidence", [])  # type: ignore[assignment]
        depends_raw: list[str] = d.get("depends_on", [])  # type: ignore[assignment]
        tags_raw: list[str] = d.get("tags", [])  # type: ignore[assignment]
        return cls(
            task_id=str(d["task_id"]),
            title=str(d["title"]),
            description=str(d["description"]),
            owner_role=AgentRole(d["owner_role"]),
            priority=str(d["priority"]),
            status=TaskStatus(d["status"]),
            assigned_to=d.get("assigned_to"),  # type: ignore[arg-type]
            evidence=tuple(
                Evidence(
                    evidence_type=EvidenceType(e["evidence_type"]),
                    content=e["content"],
                    created_by=e["created_by"],
                    timestamp=e["timestamp"],
                )
                for e in evidence_raw
            ),
            depends_on=tuple(depends_raw),
            created_by=str(d.get("created_by", "")),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            tags=tuple(tags_raw),
            rejection_reason=d.get("rejection_reason"),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Evidence requirements per transition
# ---------------------------------------------------------------------------

_REQUIRED_EVIDENCE: dict[tuple[TaskStatus, TaskStatus], tuple[EvidenceType, ...]] = {
    # Developer -> Tester: must have test results and code diff
    (TaskStatus.IN_PROGRESS, TaskStatus.TESTING): (
        EvidenceType.TEST_RESULT,
        EvidenceType.CODE_DIFF,
    ),
    # Tester -> Reviewer: must have test results + renders
    (TaskStatus.TESTING, TaskStatus.REVIEW): (
        EvidenceType.TEST_RESULT,
        EvidenceType.RENDER_2D,
        EvidenceType.RENDER_3D,
    ),
    # Reviewer -> Done: must have review findings
    (TaskStatus.REVIEW, TaskStatus.DONE): (
        EvidenceType.REVIEW_FINDING,
    ),
}

# Transitions that bypass evidence gates
_GATE_EXEMPT_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.BLOCKED, TaskStatus.REJECTED}
)


# ---------------------------------------------------------------------------
# TaskBoard
# ---------------------------------------------------------------------------


class TaskBoard:
    """Persistent task tracking with role-based views and evidence gates.

    Tasks are stored in an append-only JSONL file.  The latest entry
    per ``task_id`` wins (same pattern as eval baselines).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path(".claude/taskboard.jsonl")
        self._tasks: dict[str, Task] = {}
        self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                task = Task.from_dict(d)
                self._tasks[task.task_id] = task
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                _log.warning("taskboard parse error: %s", exc)

    def _save(self, task: Task) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(task.to_json() + "\n")

    # -- mutations ----------------------------------------------------------

    def create_task(
        self,
        title: str,
        description: str,
        owner_role: AgentRole,
        priority: str = "P2",
        created_by: str = "pm",
        depends_on: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
    ) -> Task:
        """Create a new task and persist it."""
        now = datetime.now(timezone.utc).isoformat()
        task = Task(
            task_id=uuid.uuid4().hex[:12],
            title=title,
            description=description,
            owner_role=owner_role,
            priority=priority,
            created_by=created_by,
            created_at=now,
            updated_at=now,
            depends_on=depends_on,
            tags=tags,
        )
        self._tasks[task.task_id] = task
        self._save(task)
        _log.info("Task created: %s [%s] %s", task.task_id, priority, title)
        return task

    def transition(
        self,
        task_id: str,
        new_status: TaskStatus,
        agent_id: str = "",
        evidence: tuple[Evidence, ...] = (),
        rejection_reason: str | None = None,
    ) -> Task:
        """Move a task to a new status, enforcing evidence requirements."""
        task = self._tasks.get(task_id)
        if task is None:
            msg = f"Task {task_id} not found"
            raise ValueError(msg)

        # Check evidence requirements for this transition
        key = (task.status, new_status)
        required = _REQUIRED_EVIDENCE.get(key, ())
        existing_types = {e.evidence_type for e in (*task.evidence, *evidence)}
        missing = [r for r in required if r not in existing_types]
        if missing and new_status not in _GATE_EXEMPT_STATUSES:
            missing_names = [m.value for m in missing]
            msg = (
                f"Cannot transition {task_id} from {task.status.value} to "
                f"{new_status.value}: missing evidence: {', '.join(missing_names)}"
            )
            raise ValueError(msg)

        now = datetime.now(timezone.utc).isoformat()
        updated = replace(
            task,
            status=new_status,
            assigned_to=agent_id or task.assigned_to,
            evidence=(*task.evidence, *evidence),
            updated_at=now,
            rejection_reason=rejection_reason,
        )
        self._tasks[task_id] = updated
        self._save(updated)
        _log.info(
            "Task %s: %s -> %s (by %s)",
            task_id,
            task.status.value,
            new_status.value,
            agent_id,
        )
        return updated

    # -- queries ------------------------------------------------------------

    def get_task(self, task_id: str) -> Task | None:
        """Look up a task by ID."""
        return self._tasks.get(task_id)

    def tasks_for_role(self, role: AgentRole) -> tuple[Task, ...]:
        """Return all non-done tasks owned by *role*."""
        return tuple(
            t for t in self._tasks.values()
            if t.owner_role == role and t.status != TaskStatus.DONE
        )

    def open_tasks(self) -> tuple[Task, ...]:
        """Return all tasks that are neither done nor rejected."""
        return tuple(
            t for t in self._tasks.values()
            if t.status not in (TaskStatus.DONE, TaskStatus.REJECTED)
        )

    def summary(self) -> str:
        """One-line summary of the board state."""
        by_status: dict[str, int] = {}
        for t in self._tasks.values():
            by_status[t.status.value] = by_status.get(t.status.value, 0) + 1
        parts = [f"{k}: {v}" for k, v in sorted(by_status.items())]
        return f"TaskBoard: {len(self._tasks)} tasks ({', '.join(parts)})"
