"""Tests for kicad_pipeline.service.pipeline.PipelineService."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from kicad_pipeline.exceptions import RequirementsError
from kicad_pipeline.service.models import PipelineRequest, StageOutcome
from kicad_pipeline.service.pipeline import PipelineService
from tests.helpers import make_requirements

# ---------------------------------------------------------------------------
# resolve_requirements
# ---------------------------------------------------------------------------


def test_resolve_requirements_from_object() -> None:
    """resolve_requirements returns passed-in requirements directly."""
    req = make_requirements()
    service = PipelineService()
    request = PipelineRequest(requirements=req)
    result = service.resolve_requirements(request)
    assert result is req


def test_resolve_requirements_from_path(tmp_path: Path) -> None:
    """resolve_requirements loads from file when requirements_path given."""
    from kicad_pipeline.requirements.decomposer import save_requirements

    req = make_requirements()
    req_path = tmp_path / "requirements.json"
    save_requirements(req, req_path)

    service = PipelineService()
    request = PipelineRequest(requirements_path=req_path)
    loaded = service.resolve_requirements(request)

    # Should have the same components
    assert len(loaded.components) == len(req.components)
    assert loaded.project.name == req.project.name


def test_resolve_requirements_raises_without_input() -> None:
    """resolve_requirements raises when neither requirements nor path given."""
    service = PipelineService()
    request = PipelineRequest()
    with pytest.raises(RequirementsError, match="must provide either"):
        service.resolve_requirements(request)


# ---------------------------------------------------------------------------
# run_requirements
# ---------------------------------------------------------------------------


def test_run_requirements_success(tmp_path: Path) -> None:
    """run_requirements with valid requirements succeeds."""
    req = make_requirements()
    service = PipelineService()
    request = PipelineRequest(
        requirements=req,
        output_dir=tmp_path,
        board_name="test_board",
    )
    outcome = service.run_requirements(request)

    assert isinstance(outcome, StageOutcome)
    assert outcome.stage == "requirements"
    assert outcome.success is True
    assert len(outcome.errors) == 0
    assert len(outcome.artifacts) == 1
    assert outcome.artifacts[0].endswith("requirements.json")
    assert outcome.duration_secs >= 0


def test_run_requirements_empty_components_warns(tmp_path: Path) -> None:
    """run_requirements warns when requirements have no components."""
    req = make_requirements(components=())
    service = PipelineService()
    request = PipelineRequest(
        requirements=req,
        output_dir=tmp_path,
        board_name="empty_board",
    )
    outcome = service.run_requirements(request)

    assert outcome.success is True
    assert any("no components" in w.lower() for w in outcome.warnings)


def test_run_requirements_failure_on_bad_path(tmp_path: Path) -> None:
    """run_requirements fails gracefully when requirements_path is invalid."""
    service = PipelineService()
    request = PipelineRequest(
        requirements_path=tmp_path / "nonexistent.json",
        output_dir=tmp_path,
        board_name="bad",
    )
    outcome = service.run_requirements(request)

    assert outcome.success is False
    assert len(outcome.errors) == 1
    assert outcome.errors[0].code == "KAP-001"


# ---------------------------------------------------------------------------
# run_schematic
# ---------------------------------------------------------------------------


def test_run_schematic_success(tmp_path: Path) -> None:
    """run_schematic produces artifacts and succeeds."""
    req = make_requirements()
    service = PipelineService()
    request = PipelineRequest(
        requirements=req,
        output_dir=tmp_path,
        board_name="sch_test",
    )
    outcome = service.run_schematic(request)

    assert isinstance(outcome, StageOutcome)
    assert outcome.stage == "schematic"
    assert outcome.success is True
    assert len(outcome.artifacts) >= 1
    assert any(a.endswith(".kicad_sch") for a in outcome.artifacts)
    assert outcome.duration_secs >= 0


# ---------------------------------------------------------------------------
# run_pcb (mocked heavy builders)
# ---------------------------------------------------------------------------


def test_run_pcb_success(tmp_path: Path) -> None:
    """run_pcb produces PCB file and score."""
    req = make_requirements()
    service = PipelineService()
    request = PipelineRequest(
        requirements=req,
        output_dir=tmp_path,
        board_name="pcb_test",
    )

    # Create mock quality score
    mock_quality = MagicMock()
    mock_quality.overall_score = 0.85
    mock_quality.grade = "B"

    # Mock heavy PCB builders to avoid slow builds
    with (
        patch(
            "kicad_pipeline.service.pipeline.PipelineService.resolve_requirements",
            return_value=req,
        ),
        patch("kicad_pipeline.pcb.builder.build_pcb") as mock_build,
        patch(
            "kicad_pipeline.optimization.placement_optimizer.optimize_placement_ee"
        ) as mock_opt,
        patch("kicad_pipeline.pcb.builder.write_pcb"),
        patch(
            "kicad_pipeline.optimization.scoring.compute_fast_placement_score",
            return_value=mock_quality,
        ),
        patch(
            "kicad_pipeline.validation.containment.check_board_containment",
            return_value=[],
        ),
        patch(
            "kicad_pipeline.validation.collisions.check_collisions",
            return_value=[],
        ),
    ):
        mock_pcb = MagicMock()
        mock_build.return_value = mock_pcb
        mock_opt.return_value = (mock_pcb, MagicMock())

        outcome = service.run_pcb(request)

    assert outcome.stage == "pcb"
    assert outcome.success is True
    assert outcome.score_overall == 0.85
    assert outcome.score_grade == "B"
    assert outcome.duration_secs >= 0


# ---------------------------------------------------------------------------
# run_full
# ---------------------------------------------------------------------------


def test_run_full_stops_on_fatal_error(tmp_path: Path) -> None:
    """run_full stops when a stage produces a fatal error."""
    service = PipelineService()
    req = make_requirements()
    request = PipelineRequest(
        requirements=req,
        output_dir=tmp_path,
        board_name="fatal_test",
    )

    # Make run_schematic raise so pipeline stops before PCB
    with patch.object(
        service,
        "run_schematic",
        return_value=StageOutcome(
            stage="schematic",
            success=False,
            artifacts=(),
            errors=(
                MagicMock(severity="fatal", code="KAP-010", message="fail"),
            ),
            warnings=(),
            duration_secs=0.1,
        ),
    ):
        result = service.run_full(request)

    assert result.overall_success is False
    # Should have requirements (real) + schematic (mocked fatal) and then stop
    stage_names = [o.stage for o in result.outcomes]
    assert "requirements" in stage_names
    assert "schematic" in stage_names
    # PCB should NOT have been attempted
    assert "pcb" not in stage_names


def test_stage_outcome_has_timing(tmp_path: Path) -> None:
    """Every StageOutcome has duration_secs >= 0."""
    req = make_requirements()
    service = PipelineService()
    request = PipelineRequest(
        requirements=req,
        output_dir=tmp_path,
        board_name="timing_test",
    )
    outcome = service.run_requirements(request)
    assert outcome.duration_secs >= 0
