"""Tests for kicad_pipeline.service.errors."""

from __future__ import annotations

from kicad_pipeline.exceptions import (
    ComponentError,
    ConfigurationError,
    DRCError,
    ERCError,
    FileFormatError,
    FootprintError,
    GerberError,
    GitHubError,
    KiCadPipelineError,
    OptimizationError,
    OrchestrationError,
    PCBError,
    ProductionError,
    RequirementsError,
    RoutingError,
    SchematicError,
    SExpError,
    ValidationError,
)
from kicad_pipeline.service.errors import ERROR_REGISTRY, classify_error
from kicad_pipeline.service.models import PipelineError


def test_classify_requirements_error() -> None:
    """RequirementsError maps to KAP-001."""
    exc = RequirementsError("bad requirements file")
    result = classify_error(exc)
    assert result.code == "KAP-001"
    assert result.severity == "fatal"


def test_classify_component_error() -> None:
    """ComponentError (subclass of RequirementsError) maps to KAP-002."""
    exc = ComponentError("missing footprint for U1")
    result = classify_error(exc)
    assert result.code == "KAP-002"
    assert result.severity == "fatal"


def test_classify_pcb_error() -> None:
    """PCBError maps to KAP-015."""
    exc = PCBError("layout failed")
    result = classify_error(exc)
    assert result.code == "KAP-015"


def test_classify_footprint_error() -> None:
    """FootprintError (subclass of PCBError) maps to KAP-016."""
    exc = FootprintError("pad geometry invalid")
    result = classify_error(exc)
    assert result.code == "KAP-016"


def test_classify_schematic_error() -> None:
    """SchematicError maps to KAP-010."""
    result = classify_error(SchematicError("net mismatch"))
    assert result.code == "KAP-010"


def test_classify_erc_error() -> None:
    """ERCError (subclass of SchematicError) maps to KAP-011."""
    result = classify_error(ERCError("unconnected pin"))
    assert result.code == "KAP-011"


def test_classify_routing_error() -> None:
    """RoutingError maps to KAP-017 (recoverable)."""
    result = classify_error(RoutingError("unrouted nets"))
    assert result.code == "KAP-017"
    assert result.severity == "recoverable"


def test_classify_validation_error() -> None:
    """ValidationError maps to KAP-020."""
    result = classify_error(ValidationError("check failed"))
    assert result.code == "KAP-020"


def test_classify_drc_error() -> None:
    """DRCError maps to KAP-021."""
    result = classify_error(DRCError("clearance violation"))
    assert result.code == "KAP-021"


def test_classify_production_error() -> None:
    """ProductionError maps to KAP-025."""
    result = classify_error(ProductionError("gerber export failed"))
    assert result.code == "KAP-025"


def test_classify_gerber_error() -> None:
    """GerberError maps to KAP-026."""
    result = classify_error(GerberError("layer missing"))
    assert result.code == "KAP-026"


def test_classify_unknown_exception() -> None:
    """Non-pipeline exception maps to KAP-999 fallback."""
    exc = RuntimeError("something unexpected")
    result = classify_error(exc)
    assert result.code == "KAP-999"
    assert result.severity == "fatal"


def test_classify_preserves_message() -> None:
    """classify_error includes the original exception message."""
    exc = PCBError("netlist mismatch on U3")
    result = classify_error(exc)
    assert "netlist mismatch on U3" in result.message


def test_classify_includes_stage() -> None:
    """classify_error propagates the stage parameter."""
    exc = PCBError("test")
    result = classify_error(exc, stage="pcb")
    assert result.stage == "pcb"


def test_classify_empty_message_uses_cause() -> None:
    """classify_error falls back to template cause when exception message is empty."""
    exc = PCBError("")
    result = classify_error(exc)
    # When str(exc) is empty, message should be the template cause
    assert result.message == result.cause


def test_classify_returns_pipeline_error_type() -> None:
    """classify_error always returns a PipelineError instance."""
    result = classify_error(ValueError("oops"))
    assert isinstance(result, PipelineError)


def test_error_registry_covers_all_base_types() -> None:
    """Every direct KiCadPipelineError subclass has a registry entry."""
    # Get all direct subclasses of KiCadPipelineError
    direct_subclasses = set(KiCadPipelineError.__subclasses__())
    registered_types = set(ERROR_REGISTRY.keys())

    # Every direct subclass should be in the registry
    missing = direct_subclasses - registered_types
    assert missing == set(), (
        f"Missing registry entries for direct subclasses: "
        f"{[cls.__name__ for cls in missing]}"
    )


def test_error_registry_codes_are_unique() -> None:
    """Every error code in the registry is unique."""
    codes = [t.code for t in ERROR_REGISTRY.values()]
    assert len(codes) == len(set(codes)), (
        f"Duplicate codes found: {[c for c in codes if codes.count(c) > 1]}"
    )


def test_classify_optimization_error() -> None:
    """OptimizationError maps to KAP-045 (recoverable)."""
    result = classify_error(OptimizationError("convergence failed"))
    assert result.code == "KAP-045"
    assert result.severity == "recoverable"


def test_classify_github_error() -> None:
    """GitHubError maps to KAP-046 (recoverable)."""
    result = classify_error(GitHubError("auth failed"))
    assert result.code == "KAP-046"
    assert result.severity == "recoverable"
