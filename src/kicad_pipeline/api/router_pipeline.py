"""Pipeline execution endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from kicad_pipeline.api.schemas import (
    PipelineErrorSchema,
    PipelineRequestSchema,
    PipelineResultSchema,
    StageOutcomeSchema,
)

router = APIRouter()

_VALID_STAGES = {"requirements", "schematic", "pcb", "validation", "production"}


@router.post("/run/{stage}", response_model=StageOutcomeSchema)
async def run_stage(
    stage: str,
    request: PipelineRequestSchema,
    requirements_json: str = "",
) -> StageOutcomeSchema:
    """Run a single pipeline stage.

    The requirements_json body parameter contains the requirements JSON as a string.
    Stage must be one of: requirements, schematic, pcb, validation, production.
    """
    if stage not in _VALID_STAGES:
        raise HTTPException(400, f"Invalid stage: {stage}. Must be one of {_VALID_STAGES}")

    from kicad_pipeline.requirements.decomposer import requirements_from_dict
    from kicad_pipeline.service.models import PipelineRequest
    from kicad_pipeline.service.pipeline import PipelineService

    try:
        if requirements_json:
            req_dict = json.loads(requirements_json)
            req = requirements_from_dict(req_dict)
        else:
            raise HTTPException(400, "requirements_json is required")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON: {e}") from e

    service = PipelineService()
    pipe_request = PipelineRequest(
        requirements=req,
        output_dir=Path(tempfile.mkdtemp(prefix="kicad-api-")),
        board_name=request.board_name,
        variant=request.variant,
        placement_mode=request.placement_mode,
        board_width_mm=request.board_width_mm,
        board_height_mm=request.board_height_mm,
        auto_route=request.auto_route,
        validate_parts=request.validate_parts,
    )

    dispatch = {
        "requirements": service.run_requirements,
        "schematic": service.run_schematic,
        "pcb": service.run_pcb,
        "validation": service.run_validation,
        "production": service.run_production,
    }

    outcome = dispatch[stage](pipe_request)
    return _outcome_to_schema(outcome)


@router.post("/run-full", response_model=PipelineResultSchema)
async def run_full_pipeline(
    request: PipelineRequestSchema,
    requirements_json: str = "",
) -> PipelineResultSchema:
    """Run the full pipeline end-to-end."""
    from kicad_pipeline.requirements.decomposer import requirements_from_dict
    from kicad_pipeline.service.models import PipelineRequest
    from kicad_pipeline.service.pipeline import PipelineService

    try:
        req_dict = json.loads(requirements_json)
        req = requirements_from_dict(req_dict)
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(400, f"Invalid requirements: {e}") from e

    service = PipelineService()
    pipe_request = PipelineRequest(
        requirements=req,
        output_dir=Path(tempfile.mkdtemp(prefix="kicad-api-")),
        board_name=request.board_name,
        variant=request.variant,
        placement_mode=request.placement_mode,
    )

    result = service.run_full(pipe_request)

    return PipelineResultSchema(
        outcomes=[_outcome_to_schema(o) for o in result.outcomes],
        overall_success=result.overall_success,
        board_path=result.board_path,
        production_path=result.production_path,
    )


def _outcome_to_schema(outcome: Any) -> StageOutcomeSchema:
    """Convert internal StageOutcome to HTTP schema."""
    return StageOutcomeSchema(
        stage=outcome.stage,
        success=outcome.success,
        artifacts=list(outcome.artifacts),
        errors=[
            PipelineErrorSchema(
                code=e.code,
                message=e.message,
                cause=e.cause,
                fix=e.fix,
                severity=e.severity,
                ref=e.ref,
                stage=e.stage,
            )
            for e in outcome.errors
        ],
        warnings=list(outcome.warnings),
        duration_secs=outcome.duration_secs,
        score_overall=outcome.score_overall,
        score_grade=outcome.score_grade,
    )
