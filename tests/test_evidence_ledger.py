"""Tests for JSONL-based evidence ledger persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.evidence.ledger import (
    append_record,
    clear_ledger,
    ledger_path,
    load_ledger,
)
from kicad_pipeline.evidence.models import EvidenceKind, EvidenceRecord

if TYPE_CHECKING:
    from pathlib import Path


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


def _board_path(tmp_path: Path, name: str = "test_board") -> Path:
    board_dir = tmp_path / "output" / name
    board_dir.mkdir(parents=True)
    return board_dir / f"{name}.kicad_pcb"


class TestLedgerPath:
    def test_convention(self, tmp_path: Path) -> None:
        bp = tmp_path / "output" / "my_board" / "my_board.kicad_pcb"
        path = ledger_path(bp)
        assert path.name == "my_board.jsonl"
        assert path.parent.name == ".evidence"
        assert path.parent.parent == bp.parent


class TestAppendRecord:
    def test_creates_file(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        record = _make_record()
        result = append_record(bp, record)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_append_only(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        append_record(bp, _make_record(summary="first"))
        append_record(bp, _make_record(summary="second"))
        append_record(bp, _make_record(summary="third"))
        path = ledger_path(bp)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path, "nested")
        append_record(bp, _make_record())
        assert ledger_path(bp).exists()


class TestLoadLedger:
    def test_empty_when_no_file(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        ledger = load_ledger(bp)
        assert ledger.board == "test_board"
        assert len(ledger.records) == 0

    def test_roundtrip(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        r1 = _make_record(summary="render done", kind=EvidenceKind.RENDER)
        r2 = _make_record(summary="review done", kind=EvidenceKind.REVIEW, passed=False)
        append_record(bp, r1)
        append_record(bp, r2)

        ledger = load_ledger(bp)
        assert len(ledger.records) == 2
        assert ledger.records[0].summary == "render done"
        assert ledger.records[1].passed is False

    def test_skips_corrupted_lines(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        append_record(bp, _make_record(summary="good"))
        path = ledger_path(bp)
        with path.open("a") as f:
            f.write("this is not valid json\n")
        append_record(bp, _make_record(summary="also good"))

        ledger = load_ledger(bp)
        assert len(ledger.records) == 2
        assert ledger.records[0].summary == "good"
        assert ledger.records[1].summary == "also good"

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        append_record(bp, _make_record())
        path = ledger_path(bp)
        with path.open("a") as f:
            f.write("\n\n")
        append_record(bp, _make_record())

        ledger = load_ledger(bp)
        assert len(ledger.records) == 2

    def test_preserves_record_fields(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        original = _make_record(
            kind=EvidenceKind.SCORE,
            stage="validation",
            step="scoring",
            producer="harness",
            passed=True,
            summary="Grade A",
            details={"overall_score": 0.95, "grade": "A", "breakdown": {}},
            artifacts=["score.json"],
        )
        append_record(bp, original)

        ledger = load_ledger(bp)
        loaded = ledger.records[0]
        assert loaded.kind == EvidenceKind.SCORE
        assert loaded.stage == "validation"
        assert loaded.producer == "harness"
        assert loaded.details["overall_score"] == 0.95
        assert loaded.artifacts == ["score.json"]


class TestClearLedger:
    def test_clears(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        append_record(bp, _make_record())
        assert ledger_path(bp).exists()
        clear_ledger(bp)
        assert not ledger_path(bp).exists()

    def test_no_error_if_missing(self, tmp_path: Path) -> None:
        bp = _board_path(tmp_path)
        clear_ledger(bp)  # should not raise
