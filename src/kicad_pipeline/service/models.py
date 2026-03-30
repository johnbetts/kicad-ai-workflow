"""Service-layer data models for the pipeline facade.

All models are frozen dataclasses with tuple fields for immutability.
These are the primary exchange types between consumers (CLI, API,
dashboard, tests) and the pipeline internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements


@dataclass(frozen=True)
class PipelineError:
    """Structured error with code, cause, and fix suggestion.

    Every exception raised during a pipeline run is converted into one
    of these before being surfaced to the caller, ensuring a consistent
    error interface regardless of which stage failed.
    """

    code: str
    """Error code in ``KAP-NNN`` format."""

    message: str
    """Human-readable description of what went wrong."""

    cause: str
    """Why the error occurred."""

    fix: str
    """What the user should do to resolve it."""

    severity: str
    """One of ``"fatal"``, ``"recoverable"``, ``"warning"``."""

    ref: str = ""
    """Component reference designator, if applicable (e.g. ``"U1"``)."""

    stage: str = ""
    """Pipeline stage where the error occurred (e.g. ``"pcb"``)."""


@dataclass(frozen=True)
class StageOutcome:
    """Result from a single pipeline stage execution.

    Each stage (requirements, schematic, pcb, validation, production)
    produces exactly one ``StageOutcome`` containing its artifacts,
    errors, and optional quality scores.
    """

    stage: str
    """Stage name: requirements, schematic, pcb, validation, or production."""

    success: bool
    """Whether the stage completed without fatal errors."""

    artifacts: tuple[str, ...]
    """Absolute file paths of generated artifacts."""

    errors: tuple[PipelineError, ...]
    """Structured errors encountered during the stage."""

    warnings: tuple[str, ...]
    """Non-fatal warning messages."""

    duration_secs: float
    """Wall-clock time spent in this stage."""

    score_overall: float | None = None
    """Placement quality score (PCB stage only)."""

    score_grade: str | None = None
    """Letter grade A-F (PCB stage only)."""


@dataclass(frozen=True)
class PipelineRequest:
    """Input to any pipeline stage.

    Callers provide either ``requirements_path`` (load from JSON file)
    or ``requirements`` (pass a pre-built model).  The remaining fields
    control output location and pipeline behaviour.
    """

    requirements_path: Path | None = None
    """Path to a ``requirements.json`` file to load."""

    requirements: ProjectRequirements | None = None
    """Pre-built requirements model (alternative to ``requirements_path``)."""

    output_dir: Path = Path("output")
    """Directory for all generated artifacts."""

    board_name: str = "board"
    """Base name for generated KiCad files."""

    variant: str = "default"
    """Build variant (used by the orchestrator for multi-variant boards)."""

    placement_mode: str = "grouped"
    """Placement algorithm: ``"grouped"`` (hierarchical) or ``"flat"``."""

    board_width_mm: float | None = None
    """Override board width in millimetres (auto-sized when ``None``)."""

    board_height_mm: float | None = None
    """Override board height in millimetres (auto-sized when ``None``)."""

    auto_route: bool = False
    """Whether to invoke the autorouter after placement."""

    validate_parts: bool = False
    """Whether to check JLCPCB part availability."""

    web_check: bool = False
    """Whether to perform live web lookups for part validation."""


@dataclass(frozen=True)
class PipelineResult:
    """Full pipeline run result.

    Contains the ordered sequence of :class:`StageOutcome` instances
    plus convenience accessors for the final board and production paths.
    """

    outcomes: tuple[StageOutcome, ...]
    """One ``StageOutcome`` per executed stage, in execution order."""

    overall_success: bool
    """``True`` only if every stage succeeded."""

    board_path: str | None = None
    """Absolute path to the generated ``.kicad_pcb`` file, if any."""

    production_path: str | None = None
    """Absolute path to the production ZIP, if any."""
