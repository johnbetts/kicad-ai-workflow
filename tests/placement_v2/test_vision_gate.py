"""Tests for the Gate B vision protocol."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.placement_v2.ir import Severity
from kicad_pipeline.placement_v2.vision_gate import (
    REQUIRED_VIEWS,
    VISION_CHECKLIST,
    VisionProtocolError,
    VisionVerdict,
    gate_b_record,
    golden_diff,
    parse_verdicts,
    reviewer_instructions,
    verdicts_to_violations,
)

if TYPE_CHECKING:
    from pathlib import Path


def _all_pass_json() -> str:
    return json.dumps([
        {"check_id": c.check_id, "passed": True, "refs": [], "evidence": "ok"}
        for c in VISION_CHECKLIST
    ])


def _verdicts(failing: str | None = None) -> tuple[VisionVerdict, ...]:
    return tuple(
        VisionVerdict(
            check_id=c.check_id,
            passed=c.check_id != failing,
            refs=("K1",) if c.check_id == failing else (),
            evidence="relay body overlaps Q1" if c.check_id == failing else "ok",
        )
        for c in VISION_CHECKLIST
    )


class TestParseVerdicts:
    def test_full_pass_parses(self) -> None:
        verdicts = parse_verdicts(_all_pass_json())
        assert len(verdicts) == len(VISION_CHECKLIST)
        assert all(v.passed for v in verdicts)

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(VisionProtocolError, match="not valid JSON"):
            parse_verdicts("the board looks great!")

    def test_missing_check_raises(self) -> None:
        data = json.loads(_all_pass_json())[:-1]
        with pytest.raises(VisionProtocolError, match="missing="):
            parse_verdicts(json.dumps(data))

    def test_unknown_check_raises(self) -> None:
        data = json.loads(_all_pass_json())
        data[0]["check_id"] = "made_up_check"
        with pytest.raises(VisionProtocolError, match="extra="):
            parse_verdicts(json.dumps(data))

    def test_prose_evidence_not_accepted_as_array(self) -> None:
        with pytest.raises(VisionProtocolError, match="JSON array"):
            parse_verdicts('{"summary": "all good"}')


class TestVerdictsToViolations:
    def test_pass_produces_no_violations(self) -> None:
        assert verdicts_to_violations(_verdicts()) == ()

    def test_critical_check_fails_critical(self) -> None:
        (v,) = verdicts_to_violations(_verdicts(failing="bodies_no_overlap"))
        assert v.severity is Severity.CRITICAL
        assert v.refs == ("K1",)
        assert "overlaps" in v.message


class TestGateBRecord:
    def _renders(self, tmp_path: Path) -> dict[str, Path]:
        renders = {}
        for view in REQUIRED_VIEWS:
            p = tmp_path / f"{view}.png"
            p.write_bytes(view.encode())
            renders[view] = p
        return renders

    def test_record_passes_when_all_verdicts_pass(self, tmp_path: Path) -> None:
        rec = gate_b_record(_verdicts(), self._renders(tmp_path), "abc", "2026-06-10")
        assert rec.passed and rec.stage == "gate_b"
        assert len(rec.checks_run) == len(VISION_CHECKLIST)

    def test_record_fails_on_failed_verdict(self, tmp_path: Path) -> None:
        rec = gate_b_record(
            _verdicts(failing="antenna_zone_clear"),
            self._renders(tmp_path), "abc", "2026-06-10",
        )
        assert not rec.passed
        assert rec.violations[0].constraint == "vision:antenna_zone_clear"

    def test_missing_view_raises(self, tmp_path: Path) -> None:
        renders = self._renders(tmp_path)
        del renders["3d_iso_back"]
        with pytest.raises(VisionProtocolError, match="3d_iso_back"):
            gate_b_record(_verdicts(), renders, "abc", "2026-06-10")


class TestGoldenDiff:
    def test_identical_files_match(self, tmp_path: Path) -> None:
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        a.write_bytes(b"render")
        b.write_bytes(b"render")
        assert golden_diff(a, b)

    def test_different_files_differ(self, tmp_path: Path) -> None:
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        a.write_bytes(b"render")
        b.write_bytes(b"changed")
        assert not golden_diff(a, b)

    def test_missing_baseline_is_no_match(self, tmp_path: Path) -> None:
        a = tmp_path / "a.png"
        a.write_bytes(b"render")
        assert not golden_diff(a, tmp_path / "missing.png")


class TestReviewerInstructions:
    def test_instructions_cover_every_check_and_view(self, tmp_path: Path) -> None:
        renders = {v: tmp_path / f"{v}.png" for v in REQUIRED_VIEWS}
        text = reviewer_instructions(renders)
        for c in VISION_CHECKLIST:
            assert c.check_id in text
        assert "JSON array" in text
