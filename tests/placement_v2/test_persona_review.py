"""Dual-persona review gate tests (Gate C feedback 2026-06-11 item 6)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from kicad_pipeline.placement_v2.ir import Severity
from kicad_pipeline.placement_v2.persona_review import (
    EE_CHECKLIST,
    FAB_CHECKLIST,
    PERSONAS,
    PRESENTATION_STAGES,
    checklist_for,
    parse_persona_verdicts,
    persona_instructions,
    persona_record,
    persona_violations,
)
from kicad_pipeline.placement_v2.vision_gate import (
    REQUIRED_VIEWS,
    VisionProtocolError,
)


def _all_pass(persona: str) -> str:
    return json.dumps([
        {"check_id": c.check_id, "passed": True, "refs": [], "evidence": "ok"}
        for c in checklist_for(persona)
    ])


def _renders(tmp_path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for view in REQUIRED_VIEWS:
        p = tmp_path / f"{view}.png"
        p.write_bytes(b"png" + view.encode())
        out[view] = p
    return out


def test_both_personas_have_distinct_checklists() -> None:
    assert PERSONAS == ("fab", "ee")
    fab_ids = {c.check_id for c in FAB_CHECKLIST}
    ee_ids = {c.check_id for c in EE_CHECKLIST}
    assert not fab_ids & ee_ids
    assert all(c.check_id.startswith("fab_") for c in FAB_CHECKLIST)
    assert all(c.check_id.startswith("ee_") for c in EE_CHECKLIST)


def test_instructions_name_every_check_and_render(tmp_path: Path) -> None:
    renders = _renders(tmp_path)
    for persona in PERSONAS:
        text = persona_instructions(persona, renders)
        for check in checklist_for(persona):
            assert check.check_id in text
        for path in renders.values():
            assert str(path) in text


def test_parse_round_trip_all_pass() -> None:
    verdicts = parse_persona_verdicts("fab", _all_pass("fab"))
    assert len(verdicts) == len(FAB_CHECKLIST)
    assert persona_violations("fab", verdicts) == ()


def test_parse_rejects_missing_check() -> None:
    partial = json.dumps([
        {"check_id": FAB_CHECKLIST[0].check_id, "passed": True,
         "refs": [], "evidence": "ok"},
    ])
    with pytest.raises(VisionProtocolError, match="missing"):
        parse_persona_verdicts("fab", partial)


def test_parse_rejects_foreign_persona_checks() -> None:
    with pytest.raises(VisionProtocolError, match="mismatch"):
        parse_persona_verdicts("fab", _all_pass("ee"))


def test_unknown_persona_rejected() -> None:
    with pytest.raises(VisionProtocolError, match="unknown persona"):
        checklist_for("manager")


def test_failed_critical_check_blocks(tmp_path: Path) -> None:
    raw = json.loads(_all_pass("ee"))
    for item in raw:
        if item["check_id"] == "ee_isolation":
            item["passed"] = False
            item["refs"] = ["K1"]
            item["evidence"] = "logic resistor inside the mains gap"
    verdicts = parse_persona_verdicts("ee", json.dumps(raw))
    record = persona_record("ee", verdicts, _renders(tmp_path), "sha", "t")
    assert record.stage == "review_ee"
    assert not record.passed
    assert record.violations[0].severity is Severity.CRITICAL
    assert record.violations[0].refs == ("K1",)


def test_minor_only_failure_still_passes(tmp_path: Path) -> None:
    raw = json.loads(_all_pass("fab"))
    for item in raw:
        if item["check_id"] == "fab_silkscreen":
            item["passed"] = False
            item["evidence"] = "ref text overlaps pad"
    verdicts = parse_persona_verdicts("fab", json.dumps(raw))
    record = persona_record("fab", verdicts, _renders(tmp_path), "sha", "t")
    assert record.passed  # MINOR does not block
    assert len(record.violations) == 1


def test_record_requires_all_four_views(tmp_path: Path) -> None:
    verdicts = parse_persona_verdicts("fab", _all_pass("fab"))
    renders = _renders(tmp_path)
    del renders["3d_iso_back"]
    with pytest.raises(VisionProtocolError, match="missing required render"):
        persona_record("fab", verdicts, renders, "sha", "t")


def test_presentation_stages_include_both_reviews_and_sync() -> None:
    """The machine definition of 'reviewed' (Gate C item 6): both
    persona reviews, the sync gate, and Gate B are required — prose
    claims of review do not count."""
    for stage in ("review_fab", "review_ee", "gate_b", "sync", "gate_a"):
        assert stage in PRESENTATION_STAGES
