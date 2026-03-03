"""Tests for the workflow engine (orchestrator.workflow)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.exceptions import OrchestrationError
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    Net,
    NetConnection,
    Pin,
    PinFunction,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.orchestrator.manifest import save_manifest
from kicad_pipeline.orchestrator.models import (
    PackageStrategy,
    ProjectManifest,
    StageId,
    StageState,
    VariantRecord,
    VariantStatus,
    default_stages,
)
from kicad_pipeline.orchestrator.workflow import WorkflowEngine
from kicad_pipeline.requirements.decomposer import save_requirements

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_requirements() -> ProjectRequirements:
    """Build minimal valid requirements with 2 components and 1 net."""
    r1 = Component(
        ref="R1",
        value="10k",
        footprint="R_0805",
        lcsc="C17414",
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, function=PinFunction.NC),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, function=PinFunction.NC),
        ),
    )
    r2 = Component(
        ref="R2",
        value="4.7k",
        footprint="R_0805",
        lcsc="C17673",
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, function=PinFunction.NC),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, function=PinFunction.NC),
        ),
    )
    net = Net(
        name="SIG",
        connections=(
            NetConnection(ref="R1", pin="2"),
            NetConnection(ref="R2", pin="1"),
        ),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="test-project"),
        features=(
            FeatureBlock(
                name="basic",
                description="test",
                components=("R1", "R2"),
                nets=("SIG",),
                subcircuits=(),
            ),
        ),
        components=(r1, r2),
        nets=(net,),
    )


def _setup_project(tmp_path: Path) -> Path:
    """Create a minimal project directory with one variant and requirements."""
    variant_name = "test-0805"
    variant = VariantRecord(
        name=variant_name,
        display_name="Test 0805",
        description="Test variant",
        status=VariantStatus.DRAFT,
        package_strategy=PackageStrategy(name="0805"),
        stages=default_stages(),
    )
    manifest = ProjectManifest(
        project_name="test-project",
        variants=(variant,),
        active_variant=variant_name,
    )
    save_manifest(manifest, tmp_path)

    # Create variant dir with requirements
    vdir = tmp_path / "variants" / variant_name
    vdir.mkdir(parents=True)
    req = _make_requirements()
    save_requirements(req, vdir / "requirements.json")

    return tmp_path


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Return a ready-to-use project directory."""
    return _setup_project(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_current_stage_returns_first_pending(project_dir: Path) -> None:
    """Current stage should be REQUIREMENTS when all stages are pending."""
    engine = WorkflowEngine(project_dir)
    assert engine.get_current_stage("test-0805") == StageId.REQUIREMENTS


def test_generate_requirements_succeeds(project_dir: Path) -> None:
    """Generating requirements stage should succeed when file exists."""
    engine = WorkflowEngine(project_dir)
    result = engine.generate_stage("test-0805", StageId.REQUIREMENTS)
    assert result.success is True
    assert result.stage == StageId.REQUIREMENTS


def test_generate_requirements_updates_stage_state(project_dir: Path) -> None:
    """After generation, stage state should be GENERATED with count=1."""
    engine = WorkflowEngine(project_dir)
    engine.generate_stage("test-0805", StageId.REQUIREMENTS)

    # Reload to verify persistence
    engine2 = WorkflowEngine(project_dir)
    variant = engine2._get_variant("test-0805")
    sr = engine2._get_stage(variant, StageId.REQUIREMENTS)
    assert sr.state == StageState.GENERATED
    assert sr.generation_count == 1
    assert sr.generated_at is not None


def test_generate_schematic_requires_requirements_approved(
    project_dir: Path,
) -> None:
    """Generating schematic should fail if requirements is not approved."""
    engine = WorkflowEngine(project_dir)
    result = engine.generate_stage("test-0805", StageId.SCHEMATIC)
    assert result.success is False
    assert "not approved" in result.message


def test_generate_schematic_succeeds_after_requirements_approved(
    project_dir: Path,
) -> None:
    """Schematic generation succeeds once requirements is approved."""
    engine = WorkflowEngine(project_dir)
    engine.generate_stage("test-0805", StageId.REQUIREMENTS)
    engine.approve_stage("test-0805", StageId.REQUIREMENTS)
    result = engine.generate_stage("test-0805", StageId.SCHEMATIC)
    assert result.success is True
    assert result.stage == StageId.SCHEMATIC

    # Verify schematic file was created
    sch_path = project_dir / "variants" / "test-0805" / "test-0805.kicad_sch"
    assert sch_path.exists()


def test_approve_stage_requires_generated_state(project_dir: Path) -> None:
    """Approving a PENDING stage should fail."""
    engine = WorkflowEngine(project_dir)
    result = engine.approve_stage("test-0805", StageId.REQUIREMENTS)
    assert result.success is False
    assert "pending" in result.message


def test_approve_updates_state_and_timestamp(project_dir: Path) -> None:
    """Approve should set state=APPROVED and approved_at timestamp."""
    engine = WorkflowEngine(project_dir)
    engine.generate_stage("test-0805", StageId.REQUIREMENTS)
    result = engine.approve_stage("test-0805", StageId.REQUIREMENTS)
    assert result.success is True

    engine2 = WorkflowEngine(project_dir)
    variant = engine2._get_variant("test-0805")
    sr = engine2._get_stage(variant, StageId.REQUIREMENTS)
    assert sr.state == StageState.APPROVED
    assert sr.approved_at is not None


def test_rollback_resets_later_stages(project_dir: Path) -> None:
    """Rolling back to requirements should reset schematic and later stages."""
    engine = WorkflowEngine(project_dir)

    # Advance through requirements and schematic
    engine.generate_stage("test-0805", StageId.REQUIREMENTS)
    engine.approve_stage("test-0805", StageId.REQUIREMENTS)
    engine.generate_stage("test-0805", StageId.SCHEMATIC)
    engine.approve_stage("test-0805", StageId.SCHEMATIC)

    # Rollback to requirements
    result = engine.rollback_stage("test-0805", StageId.REQUIREMENTS)
    assert result.success is True

    # Verify all stages from requirements onward are PENDING
    engine2 = WorkflowEngine(project_dir)
    variant = engine2._get_variant("test-0805")
    for sr in variant.stages:
        assert sr.state == StageState.PENDING
        assert sr.generation_count == 0


def test_rollback_to_current_stage(project_dir: Path) -> None:
    """Rolling back to the current stage should reset it and later stages."""
    engine = WorkflowEngine(project_dir)
    engine.generate_stage("test-0805", StageId.REQUIREMENTS)

    result = engine.rollback_stage("test-0805", StageId.REQUIREMENTS)
    assert result.success is True

    variant = engine._get_variant("test-0805")
    sr = engine._get_stage(variant, StageId.REQUIREMENTS)
    assert sr.state == StageState.PENDING
    assert sr.generation_count == 0


def test_review_requirements_returns_summary(project_dir: Path) -> None:
    """Review should return component/net counts from requirements."""
    engine = WorkflowEngine(project_dir)
    summary = engine.review_stage("test-0805", StageId.REQUIREMENTS)
    assert summary["stage"] == "requirements"
    assert summary["component_count"] == 2
    assert summary["net_count"] == 1


def test_full_workflow_requirements_through_production(
    project_dir: Path,
) -> None:
    """Integration: walk all stages from requirements through production."""
    engine = WorkflowEngine(project_dir)

    # Requirements
    assert engine.get_current_stage("test-0805") == StageId.REQUIREMENTS
    result = engine.generate_stage("test-0805")
    assert result.success is True
    result = engine.approve_stage("test-0805")
    assert result.success is True

    # Schematic
    assert engine.get_current_stage("test-0805") == StageId.SCHEMATIC
    result = engine.generate_stage("test-0805")
    assert result.success is True
    result = engine.approve_stage("test-0805")
    assert result.success is True

    # PCB
    assert engine.get_current_stage("test-0805") == StageId.PCB
    result = engine.generate_stage("test-0805")
    assert result.success is True
    result = engine.approve_stage("test-0805")
    assert result.success is True

    # Validation (placeholder)
    assert engine.get_current_stage("test-0805") == StageId.VALIDATION
    result = engine.generate_stage("test-0805")
    assert result.success is True
    assert len(result.warnings) > 0
    assert "not yet wired" in result.warnings[0].lower()
    result = engine.approve_stage("test-0805")
    assert result.success is True

    # Production
    assert engine.get_current_stage("test-0805") == StageId.PRODUCTION
    result = engine.generate_stage("test-0805")
    assert result.success is True
    result = engine.approve_stage("test-0805")
    assert result.success is True

    # All done — current stage should be PRODUCTION (last, all approved)
    assert engine.get_current_stage("test-0805") == StageId.PRODUCTION

    # Verify files exist
    vdir = project_dir / "variants" / "test-0805"
    assert (vdir / "test-0805.kicad_sch").exists()
    assert (vdir / "test-0805.kicad_pcb").exists()
    assert (vdir / "production").is_dir()


def test_generate_unknown_variant_raises(project_dir: Path) -> None:
    """Generating for a non-existent variant should raise OrchestrationError."""
    engine = WorkflowEngine(project_dir)
    with pytest.raises(OrchestrationError, match="Variant not found"):
        engine.generate_stage("nonexistent-variant", StageId.REQUIREMENTS)
