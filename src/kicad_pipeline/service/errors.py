"""Error catalog — maps pipeline exceptions to structured PipelineError instances.

Every exception class in :mod:`kicad_pipeline.exceptions` has a
corresponding :class:`ErrorTemplate` entry in :data:`ERROR_REGISTRY`.
The :func:`classify_error` function walks an exception's MRO to find
the most specific match, then builds a :class:`PipelineError` from
the template and the actual exception message.
"""

from __future__ import annotations

from dataclasses import dataclass

from kicad_pipeline.exceptions import (
    AgentError,
    ComponentError,
    ConfigurationError,
    DRCError,
    ERCError,
    FileFormatError,
    FootprintError,
    GerberError,
    GitHubError,
    IPCConnectionError,
    IPCError,
    IPCSyncError,
    IPCUnavailableError,
    KiCadPipelineError,
    OptimizationError,
    OrchestrationError,
    PartsError,
    PCBError,
    ProductionError,
    RequirementsError,
    RoutingError,
    SchematicError,
    SExpError,
    SExpParseError,
    SExpWriteError,
    ValidationError,
)
from kicad_pipeline.service.models import PipelineError

# ---------------------------------------------------------------------------
# Error template
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorTemplate:
    """Template for converting exceptions to PipelineError."""

    code: str
    """Error code in ``KAP-NNN`` format."""

    severity: str
    """One of ``"fatal"``, ``"recoverable"``, ``"warning"``."""

    cause: str
    """Default cause text (may be overridden by the exception message)."""

    fix: str
    """Default fix suggestion for the user."""


# ---------------------------------------------------------------------------
# Registry: exception class → ErrorTemplate
# ---------------------------------------------------------------------------

ERROR_REGISTRY: dict[type[Exception], ErrorTemplate] = {
    # -- Requirements / Component (KAP-001 - KAP-009) ----------------------
    RequirementsError: ErrorTemplate(
        code="KAP-001",
        severity="fatal",
        cause="Requirements file is missing, malformed, or contains invalid data.",
        fix="Check requirements.json for syntax errors and missing required fields.",
    ),
    ComponentError: ErrorTemplate(
        code="KAP-002",
        severity="fatal",
        cause="A component could not be resolved or has invalid parameters.",
        fix="Verify the component reference, value, and footprint in requirements.",
    ),
    # -- Schematic / ERC (KAP-010 - KAP-014) -------------------------------
    SchematicError: ErrorTemplate(
        code="KAP-010",
        severity="fatal",
        cause="Schematic generation failed.",
        fix="Check the schematic builder logs and ensure all nets are connected.",
    ),
    ERCError: ErrorTemplate(
        code="KAP-011",
        severity="fatal",
        cause="Electrical rules check found hard errors in the schematic.",
        fix="Review ERC report: fix unconnected pins, conflicting types, or missing power flags.",
    ),
    # -- PCB / Footprint / Routing (KAP-015 - KAP-019) --------------------
    PCBError: ErrorTemplate(
        code="KAP-015",
        severity="fatal",
        cause="PCB layout generation failed.",
        fix="Check PCB builder logs for placement or netlist errors.",
    ),
    FootprintError: ErrorTemplate(
        code="KAP-016",
        severity="fatal",
        cause="Footprint creation or verification failed.",
        fix="Verify the footprint name, pad geometry, and 3D model path.",
    ),
    RoutingError: ErrorTemplate(
        code="KAP-017",
        severity="recoverable",
        cause="Autorouter could not complete all connections.",
        fix="Review unrouted nets; consider manual routing or board resizing.",
    ),
    # -- Validation / DRC (KAP-020 - KAP-024) -----------------------------
    ValidationError: ErrorTemplate(
        code="KAP-020",
        severity="fatal",
        cause="Board validation failed.",
        fix="Run the validation report and address each finding.",
    ),
    DRCError: ErrorTemplate(
        code="KAP-021",
        severity="fatal",
        cause="Design rules check found clearance, annular ring, or drill violations.",
        fix="Review DRC report: fix clearance violations, drill sizes, and minimum widths.",
    ),
    # -- Production / Gerber (KAP-025 - KAP-029) --------------------------
    ProductionError: ErrorTemplate(
        code="KAP-025",
        severity="fatal",
        cause="Production artifact generation failed.",
        fix="Check that the board passes DRC before generating production files.",
    ),
    GerberError: ErrorTemplate(
        code="KAP-026",
        severity="fatal",
        cause="Gerber file generation failed.",
        fix="Verify board layers and zone fills; re-run DRC before Gerber export.",
    ),
    # -- Orchestration / Agent (KAP-030 - KAP-034) ------------------------
    OrchestrationError: ErrorTemplate(
        code="KAP-030",
        severity="recoverable",
        cause="Pipeline orchestration encountered an error.",
        fix="Check the manifest and workflow configuration.",
    ),
    AgentError: ErrorTemplate(
        code="KAP-031",
        severity="recoverable",
        cause="A sub-agent failed during multi-agent coordination.",
        fix="Review agent logs; the failing agent's task may need manual intervention.",
    ),
    # -- Parts / IPC (KAP-035 - KAP-039) ----------------------------------
    PartsError: ErrorTemplate(
        code="KAP-035",
        severity="recoverable",
        cause="Part lookup or selection failed.",
        fix="Check the LCSC part number and verify availability on JLCPCB.",
    ),
    IPCError: ErrorTemplate(
        code="KAP-036",
        severity="recoverable",
        cause="IPC communication with KiCad failed.",
        fix="Ensure KiCad is running with the board open and IPC API is enabled.",
    ),
    IPCUnavailableError: ErrorTemplate(
        code="KAP-037",
        severity="recoverable",
        cause="kicad-python is not installed or KiCad is not running.",
        fix="Install kicad-python and launch KiCad with the target board open.",
    ),
    IPCConnectionError: ErrorTemplate(
        code="KAP-038",
        severity="recoverable",
        cause="Could not connect to the KiCad IPC socket.",
        fix="Restart KiCad and verify the IPC API port is not blocked.",
    ),
    IPCSyncError: ErrorTemplate(
        code="KAP-039",
        severity="recoverable",
        cause="Board state synchronisation between pipeline and KiCad failed.",
        fix="Close and reopen the board in KiCad, then retry the sync.",
    ),
    # -- File / Config / SExp (KAP-040 - KAP-044) -------------------------
    FileFormatError: ErrorTemplate(
        code="KAP-040",
        severity="fatal",
        cause="A file has an unrecognised or malformed format.",
        fix="Verify the file was generated by a supported KiCad version (9 or 10).",
    ),
    ConfigurationError: ErrorTemplate(
        code="KAP-041",
        severity="fatal",
        cause="Pipeline configuration is invalid.",
        fix="Check pyproject.toml, pipeline.yaml, or CLI arguments for typos.",
    ),
    SExpError: ErrorTemplate(
        code="KAP-042",
        severity="fatal",
        cause="S-expression processing failed.",
        fix="Check that the file is valid KiCad S-expression format.",
    ),
    SExpParseError: ErrorTemplate(
        code="KAP-043",
        severity="fatal",
        cause="S-expression could not be parsed.",
        fix="Check for unmatched parentheses, invalid tokens, or encoding issues.",
    ),
    SExpWriteError: ErrorTemplate(
        code="KAP-044",
        severity="fatal",
        cause="S-expression could not be serialised.",
        fix="Check that all node values are valid types (str, int, float, bool).",
    ),
    # -- Optimization (KAP-045) --------------------------------------------
    OptimizationError: ErrorTemplate(
        code="KAP-045",
        severity="recoverable",
        cause="Placement optimisation failed to converge.",
        fix="Try a different placement mode or increase the iteration budget.",
    ),
    # -- GitHub (KAP-046) --------------------------------------------------
    GitHubError: ErrorTemplate(
        code="KAP-046",
        severity="recoverable",
        cause="A GitHub API operation failed.",
        fix="Check your authentication token and network connectivity.",
    ),
    # -- Base / unknown (KAP-050) ------------------------------------------
    KiCadPipelineError: ErrorTemplate(
        code="KAP-050",
        severity="fatal",
        cause="An unclassified pipeline error occurred.",
        fix="Check the full traceback for details.",
    ),
}

_FALLBACK_TEMPLATE = ErrorTemplate(
    code="KAP-999",
    severity="fatal",
    cause="An unexpected error occurred outside the pipeline exception hierarchy.",
    fix="Check the full traceback and report this as a bug.",
)


# ---------------------------------------------------------------------------
# Classification function
# ---------------------------------------------------------------------------


def classify_error(exc: Exception, stage: str = "") -> PipelineError:
    """Convert any exception into a structured :class:`PipelineError`.

    Walks the exception's MRO to find the most specific matching entry
    in :data:`ERROR_REGISTRY`.  Falls back to :data:`_FALLBACK_TEMPLATE`
    for exceptions outside the pipeline hierarchy.

    Args:
        exc: The exception to classify.
        stage: Pipeline stage name (e.g. ``"pcb"``).  Attached to the
            resulting ``PipelineError.stage`` field.

    Returns:
        A fully populated ``PipelineError`` instance.
    """
    template = _FALLBACK_TEMPLATE

    # Walk MRO from most specific to least specific.
    for cls in type(exc).__mro__:
        if cls in ERROR_REGISTRY:
            template = ERROR_REGISTRY[cls]
            break

    return PipelineError(
        code=template.code,
        message=str(exc) if str(exc) else template.cause,
        cause=template.cause,
        fix=template.fix,
        severity=template.severity,
        stage=stage,
    )
