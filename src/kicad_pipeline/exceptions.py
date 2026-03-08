"""Project-wide exception hierarchy for kicad-ai-pipeline.

All exceptions ultimately inherit from :class:`KiCadPipelineError` so that
callers can catch the entire family with a single ``except`` clause when
needed.

Hierarchy
---------
::

    KiCadPipelineError
    ├── RequirementsError
    │   └── ComponentError
    ├── SchematicError
    │   └── ERCError
    ├── PCBError
    │   └── RoutingError
    ├── ValidationError
    │   └── DRCError
    ├── ProductionError
    │   └── GerberError
    ├── GitHubError
    ├── SExpError
    │   ├── SExpParseError   (legacy alias kept for back-compat)
    │   └── SExpWriteError   (legacy alias kept for back-compat)
    ├── OrchestrationError
    ├── FileFormatError
    └── ConfigurationError
"""


class KiCadPipelineError(Exception):
    """Base exception for all pipeline errors."""


# ---------------------------------------------------------------------------
# Requirements / component selection
# ---------------------------------------------------------------------------


class RequirementsError(KiCadPipelineError):
    """Raised when requirements parsing or validation fails."""


class ComponentError(RequirementsError):
    """Raised for component selection or validation failures."""


# ---------------------------------------------------------------------------
# Schematic
# ---------------------------------------------------------------------------


class SchematicError(KiCadPipelineError):
    """Raised when schematic generation or validation fails."""


class ERCError(SchematicError):
    """Raised when ERC validation fails with hard errors."""


# ---------------------------------------------------------------------------
# PCB
# ---------------------------------------------------------------------------


class PCBError(KiCadPipelineError):
    """Raised when PCB layout or generation fails."""


class RoutingError(PCBError):
    """Raised when autorouting fails or produces unroutable nets."""


# ---------------------------------------------------------------------------
# Validation (DRC, electrical, manufacturing)
# ---------------------------------------------------------------------------


class ValidationError(KiCadPipelineError):
    """Raised when DRC, electrical, or manufacturing validation fails."""


class DRCError(ValidationError):
    """Raised when DRC produces hard errors."""


# ---------------------------------------------------------------------------
# Production artifact generation
# ---------------------------------------------------------------------------


class ProductionError(KiCadPipelineError):
    """Raised when production artifact generation fails."""


class GerberError(ProductionError):
    """Raised when Gerber generation fails."""


# ---------------------------------------------------------------------------
# External services
# ---------------------------------------------------------------------------


class GitHubError(KiCadPipelineError):
    """Raised when GitHub operations fail."""


# ---------------------------------------------------------------------------
# S-expression parsing / writing
# ---------------------------------------------------------------------------


class SExpError(KiCadPipelineError):
    """Raised when S-expression parsing or writing fails."""


class SExpParseError(SExpError):
    """Raised when an S-expression cannot be parsed.

    Args:
        message: Human-readable description of the parse failure.
        position: Zero-based character offset in the source text, if known.
    """

    def __init__(self, message: str, *, position: int | None = None) -> None:
        """Initialise with an optional character-position hint.

        Args:
            message: Human-readable description of the parse failure.
            position: Zero-based character offset in the source text, if known.
        """
        self.position = position
        detail = f" (at position {position})" if position is not None else ""
        super().__init__(f"{message}{detail}")


class SExpWriteError(SExpError):
    """Raised when an S-expression node cannot be serialised."""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class OrchestrationError(KiCadPipelineError):
    """Raised when orchestration operations fail (manifest, workflow, variants)."""


# ---------------------------------------------------------------------------
# File format / configuration
# ---------------------------------------------------------------------------


class FileFormatError(KiCadPipelineError):
    """Raised when a file format is unrecognized or malformed."""


class ConfigurationError(KiCadPipelineError):
    """Raised when pipeline configuration is invalid."""


class PartsError(KiCadPipelineError):
    """Raised when part lookup or selection fails."""


# ---------------------------------------------------------------------------
# IPC (KiCad 9 live connection)
# ---------------------------------------------------------------------------


class IPCError(KiCadPipelineError):
    """Base exception for all IPC communication failures."""


class IPCUnavailableError(IPCError):
    """Raised when kicad-python is not installed or KiCad is not running."""


class IPCConnectionError(IPCError):
    """Raised when the IPC socket connection fails."""


class IPCSyncError(IPCError):
    """Raised when board state synchronisation fails."""


# ---------------------------------------------------------------------------
# Multi-agent coordination
# ---------------------------------------------------------------------------


class AgentError(OrchestrationError):
    """Raised when multi-agent coordination operations fail."""
