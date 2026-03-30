"""Tests for kicad_pipeline.service.models."""

from __future__ import annotations

from pathlib import Path

from kicad_pipeline.service.models import (
    PipelineError,
    PipelineRequest,
    PipelineResult,
    StageOutcome,
)


def test_pipeline_error_construction() -> None:
    """PipelineError can be constructed with all required fields."""
    err = PipelineError(
        code="KAP-001",
        message="Bad requirements",
        cause="Missing field",
        fix="Add the field",
        severity="fatal",
    )
    assert err.code == "KAP-001"
    assert err.message == "Bad requirements"
    assert err.cause == "Missing field"
    assert err.fix == "Add the field"
    assert err.severity == "fatal"


def test_pipeline_error_defaults() -> None:
    """PipelineError optional fields have correct defaults."""
    err = PipelineError(
        code="KAP-010",
        message="test",
        cause="test cause",
        fix="test fix",
        severity="warning",
    )
    assert err.ref == ""
    assert err.stage == ""


def test_pipeline_error_is_frozen() -> None:
    """PipelineError is immutable (frozen dataclass)."""
    err = PipelineError(
        code="KAP-001",
        message="x",
        cause="y",
        fix="z",
        severity="fatal",
    )
    import dataclasses

    try:
        err.code = "KAP-999"  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except dataclasses.FrozenInstanceError:
        pass


def test_stage_outcome_success() -> None:
    """StageOutcome with no errors is successful."""
    outcome = StageOutcome(
        stage="requirements",
        success=True,
        artifacts=("req.json",),
        errors=(),
        warnings=(),
        duration_secs=0.5,
    )
    assert outcome.success is True
    assert outcome.errors == ()
    assert outcome.score_overall is None
    assert outcome.score_grade is None


def test_stage_outcome_with_errors() -> None:
    """StageOutcome with errors has success=False."""
    err = PipelineError(
        code="KAP-015",
        message="PCB failed",
        cause="Layout error",
        fix="Fix it",
        severity="fatal",
        stage="pcb",
    )
    outcome = StageOutcome(
        stage="pcb",
        success=False,
        artifacts=(),
        errors=(err,),
        warnings=("minor issue",),
        duration_secs=1.2,
    )
    assert outcome.success is False
    assert len(outcome.errors) == 1
    assert outcome.errors[0].code == "KAP-015"
    assert outcome.warnings == ("minor issue",)


def test_stage_outcome_with_scores() -> None:
    """StageOutcome can carry placement scores (PCB stage)."""
    outcome = StageOutcome(
        stage="pcb",
        success=True,
        artifacts=("board.kicad_pcb",),
        errors=(),
        warnings=(),
        duration_secs=3.0,
        score_overall=0.87,
        score_grade="B",
    )
    assert outcome.score_overall == 0.87
    assert outcome.score_grade == "B"


def test_pipeline_request_defaults() -> None:
    """PipelineRequest has sensible defaults."""
    req = PipelineRequest()
    assert req.requirements_path is None
    assert req.requirements is None
    assert req.output_dir == Path("output")
    assert req.board_name == "board"
    assert req.variant == "default"
    assert req.placement_mode == "grouped"
    assert req.board_width_mm is None
    assert req.board_height_mm is None
    assert req.auto_route is False
    assert req.validate_parts is False
    assert req.web_check is False


def test_pipeline_result_construction() -> None:
    """PipelineResult with mixed outcomes tracks overall_success."""
    ok = StageOutcome(
        stage="requirements",
        success=True,
        artifacts=("req.json",),
        errors=(),
        warnings=(),
        duration_secs=0.1,
    )
    fail = StageOutcome(
        stage="schematic",
        success=False,
        artifacts=(),
        errors=(
            PipelineError(
                code="KAP-010",
                message="fail",
                cause="c",
                fix="f",
                severity="fatal",
            ),
        ),
        warnings=(),
        duration_secs=0.2,
    )
    result = PipelineResult(
        outcomes=(ok, fail),
        overall_success=False,
        board_path=None,
        production_path=None,
    )
    assert result.overall_success is False
    assert len(result.outcomes) == 2
    assert result.outcomes[0].success is True
    assert result.outcomes[1].success is False


def test_pipeline_result_all_success() -> None:
    """PipelineResult overall_success=True when all stages pass."""
    ok1 = StageOutcome(
        stage="requirements",
        success=True,
        artifacts=(),
        errors=(),
        warnings=(),
        duration_secs=0.1,
    )
    ok2 = StageOutcome(
        stage="schematic",
        success=True,
        artifacts=(),
        errors=(),
        warnings=(),
        duration_secs=0.2,
    )
    result = PipelineResult(
        outcomes=(ok1, ok2),
        overall_success=True,
        board_path="/tmp/board.kicad_pcb",
        production_path=None,
    )
    assert result.overall_success is True
    assert result.board_path == "/tmp/board.kicad_pcb"
    assert result.production_path is None
