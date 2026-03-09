"""Tests for VALIDATION stage wiring in the workflow engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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


def _make_requirements(lcsc_r1: str | None = "C17414") -> ProjectRequirements:
    """Build minimal requirements. Set lcsc_r1=None for missing-LCSC scenario."""
    r1 = Component(
        ref="R1",
        value="10k",
        footprint="R_0805",
        lcsc=lcsc_r1,
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


def _setup_project(
    tmp_path: Path,
    lcsc_r1: str | None = "C17414",
) -> Path:
    """Create a project dir with variant, advance to VALIDATION-ready state."""
    variant_name = "test-0805"
    # Build stages with REQUIREMENTS, SCHEMATIC, PCB all approved
    stages = list(default_stages())
    for i, sr in enumerate(stages):
        if sr.stage in (StageId.REQUIREMENTS, StageId.SCHEMATIC, StageId.PCB):
            from dataclasses import replace

            stages[i] = replace(sr, state=StageState.APPROVED)

    variant = VariantRecord(
        name=variant_name,
        display_name="Test 0805",
        description="Test variant",
        status=VariantStatus.DRAFT,
        package_strategy=PackageStrategy(name="0805"),
        stages=tuple(stages),
    )
    manifest = ProjectManifest(
        project_name="test-project",
        variants=(variant,),
        active_variant=variant_name,
    )
    save_manifest(manifest, tmp_path)

    vdir = tmp_path / "variants" / variant_name
    vdir.mkdir(parents=True)
    req = _make_requirements(lcsc_r1=lcsc_r1)
    save_requirements(req, vdir / "requirements.json")

    return tmp_path


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Project with all parts having valid LCSC numbers."""
    return _setup_project(tmp_path, lcsc_r1="C17414")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_validation_stage_passes(project_dir: Path) -> None:
    """VALIDATION succeeds when all parts are available in bundled DB."""
    engine = WorkflowEngine(project_dir)
    result = engine.generate_stage("test-0805", StageId.VALIDATION)
    assert result.success is True
    assert result.stage == StageId.VALIDATION


def test_validation_stage_blocks_on_missing(tmp_path: Path) -> None:
    """VALIDATION raises OrchestrationError when parts are unresolved."""
    # Use a component with no LCSC and a value that won't match bundled DB
    from dataclasses import replace

    proj = _setup_project(tmp_path, lcsc_r1=None)
    # Overwrite requirements with an unmatchable component (IC, not passive)
    vdir = proj / "variants" / "test-0805"
    req = _make_requirements(lcsc_r1=None)
    # Use an IC ref so bundled ComponentDB passive lookup won't match
    unmatchable = Component(
        ref="U99",
        value="NONEXISTENT_IC_XYZ",
        footprint="QFP-999",
        lcsc=None,
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, function=PinFunction.NC),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, function=PinFunction.NC),
        ),
    )
    req = replace(
        req,
        components=(unmatchable, req.components[1]),
        features=(
            FeatureBlock(
                name="basic",
                description="test",
                components=("U99", "R2"),
                nets=("SIG",),
                subcircuits=(),
            ),
        ),
        nets=(
            Net(
                name="SIG",
                connections=(
                    NetConnection(ref="U99", pin="2"),
                    NetConnection(ref="R2", pin="1"),
                ),
            ),
        ),
    )
    save_requirements(req, vdir / "requirements.json")

    engine = WorkflowEngine(proj)
    result = engine.generate_stage("test-0805", StageId.VALIDATION)
    # The stage should fail (OrchestrationError caught by generate_stage)
    assert result.success is False
    assert "validation failed" in result.message.lower() or "failed" in result.message.lower()


def test_validation_writes_reports(project_dir: Path) -> None:
    """VALIDATION stage creates report files in validation/ directory."""
    engine = WorkflowEngine(project_dir)
    engine.generate_stage("test-0805", StageId.VALIDATION)

    val_dir = project_dir / "variants" / "test-0805" / "validation"
    assert (val_dir / "parts_validation_report.txt").exists()
    assert (val_dir / "parts_validation_report.json").exists()

    # JSON should be parseable
    import json

    data = json.loads(
        (val_dir / "parts_validation_report.json").read_text(encoding="utf-8")
    )
    assert "all_parts_available" in data
    assert "parts" in data


def test_review_validation_returns_summary(project_dir: Path) -> None:
    """review_stage for VALIDATION returns cost/availability data."""
    engine = WorkflowEngine(project_dir)
    engine.generate_stage("test-0805", StageId.VALIDATION)
    summary = engine.review_stage("test-0805", StageId.VALIDATION)
    assert summary["stage"] == "validation"
    assert "all_parts_available" in summary
    assert "unresolved_count" in summary
