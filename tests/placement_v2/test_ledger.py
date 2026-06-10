"""Tests for the placement_v2 build ledger."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.placement_v2.ir import Severity, Violation
from kicad_pipeline.placement_v2.ledger import (
    BuildLedger,
    LedgerError,
    StageRecord,
    sha256_text,
)


def _record(stage: str, passed: bool, violations: tuple[Violation, ...] = ()) -> StageRecord:
    return StageRecord(
        stage=stage,
        input_sha256=sha256_text("in"),
        output_sha256=sha256_text("out"),
        checks_run=("check_a x3",),
        violations=violations,
        passed=passed,
        timestamp="2026-06-10T00:00:00",
    )


def _violation() -> Violation:
    return Violation(
        constraint="PinAttach(C1.1->U1.8)",
        refs=("C1", "U1"),
        severity=Severity.MAJOR,
        measured=7.5,
        limit=5.0,
        message="C1.1 is 7.50mm from U1.8",
    )


class TestStageRecord:
    def test_json_round_trip(self) -> None:
        rec = _record("gate_a", False, (_violation(),))
        back = StageRecord.from_json(rec.to_json())
        assert back == rec

    def test_malformed_line_raises(self) -> None:
        with pytest.raises(LedgerError):
            StageRecord.from_json('{"stage": "x"}')


class TestBuildLedger:
    def test_append_and_read(self, tmp_path: Path) -> None:
        ledger = BuildLedger(tmp_path / "build.jsonl")
        ledger.append(_record("certify", True))
        ledger.append(_record("cells", True))
        assert [r.stage for r in ledger.records()] == ["certify", "cells"]

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert BuildLedger(tmp_path / "nope.jsonl").records() == ()

    def test_latest_wins_per_stage(self, tmp_path: Path) -> None:
        ledger = BuildLedger(tmp_path / "build.jsonl")
        ledger.append(_record("gate_a", False, (_violation(),)))
        ledger.append(_record("gate_a", True))
        latest = ledger.latest("gate_a")
        assert latest is not None and latest.passed

    def test_all_green_requires_every_stage(self, tmp_path: Path) -> None:
        ledger = BuildLedger(tmp_path / "build.jsonl")
        ledger.append(_record("certify", True))
        ledger.append(_record("cells", True))
        assert not ledger.all_green(("certify", "cells", "gate_a"))
        ledger.append(_record("gate_a", True))
        assert ledger.all_green(("certify", "cells", "gate_a"))

    def test_failed_latest_blocks_green(self, tmp_path: Path) -> None:
        ledger = BuildLedger(tmp_path / "build.jsonl")
        ledger.append(_record("gate_a", True))
        ledger.append(_record("gate_a", False, (_violation(),)))
        assert not ledger.all_green(("gate_a",))

    def test_summary_shows_latest_status(self, tmp_path: Path) -> None:
        ledger = BuildLedger(tmp_path / "build.jsonl")
        ledger.append(_record("certify", True))
        ledger.append(_record("gate_a", False, (_violation(),)))
        text = ledger.summary()
        assert "certify: PASS" in text
        assert "gate_a: FAIL (1 violations)" in text

    def test_truncated_tail_does_not_destroy_history(self, tmp_path: Path) -> None:
        path = tmp_path / "build.jsonl"
        ledger = BuildLedger(path)
        ledger.append(_record("certify", True))
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"stage": "broken"')  # no newline, truncated write
        with pytest.raises(LedgerError):
            ledger.records()
        # The valid prefix is still intact on disk.
        first_line = path.read_text().splitlines()[0]
        assert StageRecord.from_json(first_line).stage == "certify"
