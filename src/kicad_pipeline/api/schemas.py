"""Pydantic models for request/response serialization.

These are the HTTP-facing models (separate from internal pipeline models).
"""

from __future__ import annotations

from pydantic import BaseModel


class PipelineRequestSchema(BaseModel):
    """HTTP request to run pipeline stages."""

    board_name: str = "board"
    variant: str = "default"
    placement_mode: str = "grouped"
    board_width_mm: float | None = None
    board_height_mm: float | None = None
    auto_route: bool = False
    validate_parts: bool = False


class PipelineErrorSchema(BaseModel):
    """Structured error in HTTP response."""

    code: str
    message: str
    cause: str
    fix: str
    severity: str
    ref: str = ""
    stage: str = ""


class StageOutcomeSchema(BaseModel):
    """HTTP response for a single stage result."""

    stage: str
    success: bool
    artifacts: list[str]
    errors: list[PipelineErrorSchema]
    warnings: list[str]
    duration_secs: float
    score_overall: float | None = None
    score_grade: str | None = None


class PipelineResultSchema(BaseModel):
    """HTTP response for full pipeline run."""

    outcomes: list[StageOutcomeSchema]
    overall_success: bool
    board_path: str | None = None
    production_path: str | None = None


class BoardSummarySchema(BaseModel):
    """Summary info for one board."""

    name: str
    grade: str = ""
    score: float | None = None
    stage: str = ""  # Current stage
    has_evidence: bool = False


class PartSearchResultSchema(BaseModel):
    """Single part search result."""

    lcsc: str
    mfr: str = ""
    description: str = ""
    package: str = ""
    stock: int = 0
    price: float = 0.0
    basic: bool = False


class PartSearchResponseSchema(BaseModel):
    """Response from parts search endpoint."""

    query: str
    results: list[PartSearchResultSchema]
    total: int


class EvidenceSubmitSchema(BaseModel):
    """Submit evidence record."""

    kind: str  # EvidenceKind value
    stage: str
    step: str = ""
    board: str
    passed: bool | None = None
    summary: str = ""
    feedback: str = ""


class BoardComponentSchema(BaseModel):
    """Component position for 3D interactive view."""

    ref: str
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    type: str = "other"  # ic, passive, connector, other


class BoardDetailSchema(BaseModel):
    """Full detail for a single board including image URLs."""

    name: str
    images: dict[str, str] = {}
    crops: list[dict[str, str]] = []
    score: dict[str, object] | None = None
    files: list[str] = []
    pcb_file_url: str = ""
    sch_file_url: str = ""
    has_pcb: bool = False
    has_schematic: bool = False
    has_requirements: bool = False
    board_size: dict[str, float] | None = None
    components: list[dict[str, object]] = []


class HealthSchema(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "1.0.0"
    test_count: int = 0  # Number of eval baselines
