"""Tests for stage gate checking logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.evidence.gates import ALL_STAGES, STAGE_GATES, check_gate

if TYPE_CHECKING:
    from pathlib import Path
from kicad_pipeline.evidence.ledger import append_record
from kicad_pipeline.evidence.models import EvidenceKind, EvidenceRecord


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


def _board_path(tmp_path: Path) -> Path:
    board_dir = tmp_path / "output" / "test_board"
    board_dir.mkdir(parents=True)
    return board_dir / "test_board.kicad_pcb"


class TestStageGateDefinitions:
    def test_all_stages_have_gates(self) -> None:
        for stage in ALL_STAGES:
            assert stage in STAGE_GATES, f"No gate defined for stage: {stage}"

    def test_gate_values_are_evidence_kinds(self) -> None:
        for stage, kinds in STAGE_GATES.items():
            for kind in kinds:
                assert isinstance(kind, EvidenceKind), (
                    f"Non-EvidenceKind in {stage} gate: {kind}"
                )

    def test_pcb_gate_is_strictest(self) -> None:
        assert len(STAGE_GATES["pcb"]) >= 4


class TestCheckGate:
    def test_passes_when_all_evidence_present(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        # Requirements gate only needs REVIEW
        append_record(bp, _make_record(
            stage="requirements", kind=EvidenceKind.REVIEW, passed=True,
        ))
        result = check_gate(bp, "requirements")
        assert result.passed is True
        assert not result.missing

    def test_fails_when_evidence_missing(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        # PCB gate needs RENDER, DRC, REVIEW, VERIFICATION, SCORE
        append_record(bp, _make_record(
            stage="pcb", kind=EvidenceKind.RENDER, passed=True,
        ))
        result = check_gate(bp, "pcb")
        assert result.passed is False
        assert "drc_report" in result.missing
        assert "review" in result.missing
        assert "render" not in result.missing

    def test_failed_evidence_does_not_count(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        append_record(bp, _make_record(
            stage="requirements", kind=EvidenceKind.REVIEW, passed=False,
        ))
        result = check_gate(bp, "requirements")
        assert result.passed is False
        assert "review" in result.missing

    def test_none_passed_does_not_count(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        append_record(bp, _make_record(
            stage="requirements", kind=EvidenceKind.REVIEW, passed=None,
        ))
        result = check_gate(bp, "requirements")
        assert result.passed is False

    def test_feedback_describes_missing(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        result = check_gate(bp, "validation")
        assert result.passed is False
        assert "human_approval" in result.feedback

    def test_full_pcb_gate(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        for kind in STAGE_GATES["pcb"]:
            append_record(bp, _make_record(stage="pcb", kind=kind, passed=True))
        result = check_gate(bp, "pcb")
        assert result.passed is True
        assert len(result.missing) == 0

    def test_empty_ledger_fails_all(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        for stage in ALL_STAGES:
            result = check_gate(bp, stage)
            assert result.passed is False, f"Stage {stage} should fail with empty ledger"

    def test_unknown_stage_passes(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        result = check_gate(bp, "nonexistent")
        assert result.passed is True
