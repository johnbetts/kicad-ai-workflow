"""Tests for the evidence model Pydantic types."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from kicad_pipeline.evidence.models import (
    EvidenceKind,
    EvidenceLedger,
    EvidenceRecord,
    GateResult,
    Issue,
    ScoreSnapshot,
    Severity,
)


def _make_record(**overrides: object) -> EvidenceRecord:
    defaults: dict[str, object] = {
        "kind": EvidenceKind.RENDER,
        "stage": "pcb",
        "step": "render",
        "board": "test_board",
        "producer": "harness",
        "passed": True,
    }
    defaults.update(overrides)
    return EvidenceRecord(**defaults)  # type: ignore[arg-type]


class TestEvidenceRecord:
    def test_roundtrip_json(self) -> None:
        record = _make_record(summary="Test render")
        dumped = record.model_dump_json()
        restored = EvidenceRecord.model_validate_json(dumped)
        assert restored.kind == EvidenceKind.RENDER
        assert restored.summary == "Test render"
        assert restored.passed is True

    def test_auto_generates_id(self) -> None:
        r1 = _make_record()
        r2 = _make_record()
        assert r1.id != r2.id
        assert len(r1.id) == 12

    def test_auto_generates_timestamp(self) -> None:
        record = _make_record()
        assert isinstance(record.timestamp, datetime)
        assert record.timestamp.tzinfo == timezone.utc

    def test_passed_can_be_none(self) -> None:
        record = _make_record(passed=None, kind=EvidenceKind.KNOWN_ISSUE_CHECK)
        assert record.passed is None

    def test_issues_list(self) -> None:
        issues = [
            Issue(ref="C1", rule="KI-005", severity=Severity.CRITICAL, description="Too far"),
            Issue(ref="R2", severity=Severity.MINOR, description="Slight offset"),
        ]
        record = _make_record(issues=issues)
        assert len(record.issues) == 2
        assert record.issues[0].severity == Severity.CRITICAL

    def test_artifacts_list(self) -> None:
        record = _make_record(artifacts=["2d_top.png", "3d_iso.png"])
        assert len(record.artifacts) == 2

    def test_details_dict(self) -> None:
        record = _make_record(details={"overall_score": 0.85, "grade": "B"})
        assert record.details["grade"] == "B"


class TestEvidenceKind:
    def test_all_kinds_serialize(self) -> None:
        for kind in EvidenceKind:
            record = _make_record(kind=kind)
            dumped = json.loads(record.model_dump_json())
            assert dumped["kind"] == kind.value

    def test_string_values(self) -> None:
        assert EvidenceKind.RENDER.value == "render"
        assert EvidenceKind.HUMAN_APPROVAL.value == "human_approval"


class TestSeverity:
    def test_all_severities(self) -> None:
        for sev in Severity:
            issue = Issue(severity=sev, description="test")
            assert issue.severity == sev


class TestIssue:
    def test_defaults(self) -> None:
        issue = Issue()
        assert issue.ref == ""
        assert issue.rule == ""
        assert issue.severity == Severity.INFO


class TestScoreSnapshot:
    def test_basic(self) -> None:
        snap = ScoreSnapshot(overall_score=0.92, grade="A", breakdown={"electrical": 0.95})
        assert snap.grade == "A"
        assert snap.breakdown["electrical"] == 0.95

    def test_from_dict(self) -> None:
        data = {"overall_score": 0.78, "grade": "B", "breakdown": {}}
        snap = ScoreSnapshot.model_validate(data)
        assert snap.overall_score == 0.78


class TestGateResult:
    def test_pass(self) -> None:
        gate = GateResult(
            gate_name="pcb-gate",
            stage="pcb",
            passed=True,
            required_evidence=["render", "review"],
            present_evidence=["render", "review"],
        )
        assert gate.passed is True
        assert not gate.missing

    def test_fail_with_missing(self) -> None:
        gate = GateResult(
            gate_name="pcb-gate",
            stage="pcb",
            passed=False,
            required_evidence=["render", "review", "score"],
            present_evidence=["render"],
            missing=["review", "score"],
            feedback="Missing: review, score",
        )
        assert gate.passed is False
        assert len(gate.missing) == 2


class TestEvidenceLedger:
    def test_append(self) -> None:
        ledger = EvidenceLedger(board="test")
        assert len(ledger.records) == 0
        ledger.append(_make_record())
        assert len(ledger.records) == 1

    def test_filter_by_stage(self) -> None:
        ledger = EvidenceLedger(board="test")
        ledger.append(_make_record(stage="pcb"))
        ledger.append(_make_record(stage="schematic"))
        ledger.append(_make_record(stage="pcb"))
        assert len(ledger.filter_by_stage("pcb")) == 2
        assert len(ledger.filter_by_stage("schematic")) == 1
        assert len(ledger.filter_by_stage("production")) == 0

    def test_filter_by_kind(self) -> None:
        ledger = EvidenceLedger(board="test")
        ledger.append(_make_record(kind=EvidenceKind.RENDER))
        ledger.append(_make_record(kind=EvidenceKind.REVIEW))
        ledger.append(_make_record(kind=EvidenceKind.RENDER))
        assert len(ledger.filter_by_kind(EvidenceKind.RENDER)) == 2
        assert len(ledger.filter_by_kind(EvidenceKind.SCORE)) == 0

    def test_has_passing_true(self) -> None:
        ledger = EvidenceLedger(board="test")
        ledger.append(_make_record(stage="pcb", kind=EvidenceKind.RENDER, passed=True))
        assert ledger.has_passing("pcb", EvidenceKind.RENDER) is True

    def test_has_passing_false_when_failed(self) -> None:
        ledger = EvidenceLedger(board="test")
        ledger.append(_make_record(stage="pcb", kind=EvidenceKind.RENDER, passed=False))
        assert ledger.has_passing("pcb", EvidenceKind.RENDER) is False

    def test_has_passing_false_when_none(self) -> None:
        ledger = EvidenceLedger(board="test")
        ledger.append(_make_record(stage="pcb", kind=EvidenceKind.RENDER, passed=None))
        assert ledger.has_passing("pcb", EvidenceKind.RENDER) is False

    def test_has_passing_false_when_empty(self) -> None:
        ledger = EvidenceLedger(board="test")
        assert ledger.has_passing("pcb", EvidenceKind.RENDER) is False

    def test_latest_score(self) -> None:
        ledger = EvidenceLedger(board="test")
        ledger.append(_make_record(
            kind=EvidenceKind.SCORE,
            details={"overall_score": 0.7, "grade": "C", "breakdown": {}},
        ))
        ledger.append(_make_record(
            kind=EvidenceKind.SCORE,
            details={"overall_score": 0.92, "grade": "A", "breakdown": {"electrical": 0.95}},
        ))
        score = ledger.latest_score()
        assert score is not None
        assert score.grade == "A"
        assert score.overall_score == 0.92

    def test_latest_score_none(self) -> None:
        ledger = EvidenceLedger(board="test")
        assert ledger.latest_score() is None

    def test_latest_by_kind(self) -> None:
        ledger = EvidenceLedger(board="test")
        ledger.append(_make_record(stage="pcb", kind=EvidenceKind.REVIEW, summary="first"))
        ledger.append(_make_record(stage="pcb", kind=EvidenceKind.REVIEW, summary="second"))
        latest = ledger.latest_by_kind("pcb", EvidenceKind.REVIEW)
        assert latest is not None
        assert latest.summary == "second"

    def test_latest_by_kind_none(self) -> None:
        ledger = EvidenceLedger(board="test")
        assert ledger.latest_by_kind("pcb", EvidenceKind.RENDER) is None
