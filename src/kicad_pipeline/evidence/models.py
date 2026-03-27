"""Pydantic models for the evidence-based stage-gate system.

All pipeline stages produce EvidenceRecords that are appended to a per-board
EvidenceLedger (JSONL file). Gates check the ledger before allowing stage
transitions. The dashboard reads the same ledger for display.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceKind(str, enum.Enum):
    """What kind of proof this record represents."""

    RENDER = "render"
    DRC_REPORT = "drc_report"
    REVIEW = "review"
    SCORE = "score"
    VERIFICATION = "verification"
    HUMAN_APPROVAL = "human_approval"
    HUMAN_REJECTION = "human_rejection"
    KNOWN_ISSUE_CHECK = "known_issue_check"
    GATE_RESULT = "gate_result"


class Severity(str, enum.Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class Issue(BaseModel):
    """A single finding within a review."""

    ref: str = ""
    rule: str = ""
    severity: Severity = Severity.INFO
    description: str = ""
    fix: str = ""


class ScoreSnapshot(BaseModel):
    """Captures QualityScore at a point in time for trending."""

    overall_score: float
    grade: str
    breakdown: dict[str, float] = Field(default_factory=dict)


def _generate_id() -> str:
    return uuid4().hex[:12]


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class EvidenceRecord(BaseModel):
    """Single proof item — immutable once written to ledger."""

    id: str = Field(default_factory=_generate_id)
    kind: EvidenceKind
    stage: str
    step: str
    board: str
    timestamp: datetime = Field(default_factory=_utc_now)
    producer: str = "harness"
    passed: bool | None = None
    summary: str = ""
    details: dict[str, object] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    feedback: str = ""


class GateResult(BaseModel):
    """Result of a blocking gate check."""

    gate_name: str
    stage: str
    passed: bool
    required_evidence: list[str]
    present_evidence: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    feedback: str = ""


class EvidenceLedger(BaseModel):
    """Append-only collection of evidence for one board."""

    schema_version: int = 1
    board: str
    project: str = ""
    records: list[EvidenceRecord] = Field(default_factory=list)

    def append(self, record: EvidenceRecord) -> None:
        """Append a record to the in-memory ledger."""
        self.records.append(record)

    def filter_by_stage(self, stage: str) -> list[EvidenceRecord]:
        """Return records matching a stage."""
        return [r for r in self.records if r.stage == stage]

    def filter_by_kind(self, kind: EvidenceKind) -> list[EvidenceRecord]:
        """Return records matching an evidence kind."""
        return [r for r in self.records if r.kind == kind]

    def has_passing(self, stage: str, kind: EvidenceKind) -> bool:
        """Check if a passing record exists for a stage + kind."""
        return any(
            r.passed is True
            for r in self.records
            if r.stage == stage and r.kind == kind
        )

    def latest_score(self) -> ScoreSnapshot | None:
        """Return the most recent score snapshot, or None."""
        scores = [r for r in self.records if r.kind == EvidenceKind.SCORE]
        if not scores:
            return None
        latest = scores[-1]
        return ScoreSnapshot.model_validate(latest.details)

    def latest_by_kind(
        self, stage: str, kind: EvidenceKind
    ) -> EvidenceRecord | None:
        """Return the most recent record for a stage + kind."""
        matches = [
            r for r in self.records if r.stage == stage and r.kind == kind
        ]
        return matches[-1] if matches else None
