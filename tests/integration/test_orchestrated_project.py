"""End-to-end integration test for the orchestrated project workflow.

Exercises the full lifecycle: init -> create variants -> walk all stages ->
create revisions -> release.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.cli.main import main
from kicad_pipeline.models.requirements import (
    Component,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.orchestrator.manifest import load_manifest
from kicad_pipeline.orchestrator.models import (
    StageState,
    VariantStatus,
)
from kicad_pipeline.orchestrator.variants import fork_requirements_for_variant
from kicad_pipeline.orchestrator.workflow import WorkflowEngine
from kicad_pipeline.requirements.decomposer import save_requirements

if TYPE_CHECKING:
    from pathlib import Path


def _make_test_requirements() -> ProjectRequirements:
    """Build a minimal valid requirements set with 3 components."""
    return ProjectRequirements(
        project=ProjectInfo(name="e2e-test-board", author="CI"),
        features=(),
        components=(
            Component(
                ref="R1",
                value="10k",
                footprint="R_0805",
                lcsc="C17414",
                pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE),
                ),
            ),
            Component(
                ref="D1",
                value="green",
                footprint="LED_0805",
                lcsc="C2297",
                pins=(
                    Pin(number="1", name="A", pin_type=PinType.PASSIVE),
                    Pin(number="2", name="K", pin_type=PinType.PASSIVE),
                ),
            ),
            Component(
                ref="C1",
                value="100nF",
                footprint="C_0805",
                lcsc="C49678",
                pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE),
                ),
            ),
        ),
        nets=(
            Net(
                name="GND",
                connections=(
                    NetConnection(ref="R1", pin="1"),
                    NetConnection(ref="C1", pin="2"),
                ),
            ),
            Net(
                name="LED_OUT",
                connections=(
                    NetConnection(ref="R1", pin="2"),
                    NetConnection(ref="D1", pin="1"),
                ),
            ),
        ),
    )


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Set up a temp dir as working directory and return it."""
    original = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original)


class TestOrchestratedProjectE2E:
    """Full lifecycle integration test."""

    def test_full_lifecycle_two_variants(self, project_dir: Path) -> None:
        """Walk the complete workflow with two package variants."""
        # ---- Step 1: Initialize project ----
        assert main(["project", "init", "--name", "led-blinker"]) == 0
        manifest = load_manifest(project_dir)
        assert manifest.project_name == "led-blinker"

        # ---- Step 2: Create two variants ----
        assert main([
            "project", "variant", "create",
            "--name", "standard-0805",
            "--strategy", "0805",
        ]) == 0
        assert main([
            "project", "variant", "create",
            "--name", "compact-0603",
            "--strategy", "0603",
        ]) == 0

        manifest = load_manifest(project_dir)
        assert len(manifest.variants) == 2
        assert manifest.active_variant == "standard-0805"

        # ---- Step 3: Write base requirements ----
        base_req = _make_test_requirements()
        base_dir = project_dir / "base"
        save_requirements(base_req, base_dir / "requirements.json")

        # ---- Step 4: Fork requirements for each variant ----
        from kicad_pipeline.orchestrator.models import get_strategy_by_name

        for variant_name, strategy_name in [
            ("standard-0805", "0805"),
            ("compact-0603", "0603"),
        ]:
            strategy = get_strategy_by_name(strategy_name)
            assert strategy is not None
            variant_req = fork_requirements_for_variant(base_req, strategy)
            vdir = project_dir / "variants" / variant_name
            save_requirements(variant_req, vdir / "requirements.json")

        # Verify 0603 variant has remapped footprints
        from kicad_pipeline.requirements.decomposer import load_requirements

        compact_req = load_requirements(
            project_dir / "variants" / "compact-0603" / "requirements.json"
        )
        r1 = compact_req.get_component("R1")
        assert r1 is not None
        assert r1.footprint == "R_0603"

        # ---- Step 5: Walk all stages for standard-0805 ----
        engine = WorkflowEngine(project_dir)

        # Requirements
        result = engine.generate_stage("standard-0805")
        assert result.success
        result = engine.approve_stage("standard-0805")
        assert result.success

        # Schematic
        result = engine.generate_stage("standard-0805")
        assert result.success
        sch_path = project_dir / "variants" / "standard-0805" / "standard-0805.kicad_sch"
        assert sch_path.exists()
        result = engine.approve_stage("standard-0805")
        assert result.success

        # PCB
        result = engine.generate_stage("standard-0805")
        assert result.success
        pcb_path = project_dir / "variants" / "standard-0805" / "standard-0805.kicad_pcb"
        assert pcb_path.exists()
        result = engine.approve_stage("standard-0805")
        assert result.success

        # Validation (placeholder)
        result = engine.generate_stage("standard-0805")
        assert result.success
        assert len(result.warnings) > 0  # "Validation not yet wired"
        result = engine.approve_stage("standard-0805")
        assert result.success

        # Production
        result = engine.generate_stage("standard-0805")
        assert result.success
        prod_dir = project_dir / "variants" / "standard-0805" / "production"
        assert prod_dir.exists()
        result = engine.approve_stage("standard-0805")
        assert result.success

        # ---- Step 6: Create a revision ----
        assert main([
            "project", "revision", "create",
            "--variant", "standard-0805",
            "--notes", "First production run",
        ]) == 0

        manifest = load_manifest(project_dir)
        v0805 = next(v for v in manifest.variants if v.name == "standard-0805")
        assert len(v0805.revisions) == 1
        assert v0805.revisions[0].number == 1
        assert v0805.revisions[0].git_tag == "standard-0805/rev1"

        # Verify revision directory exists with production files
        rev_dir = project_dir / "variants" / "standard-0805" / "revisions" / "rev1" / "production"
        assert rev_dir.exists()

        # ---- Step 7: Release the variant ----
        assert main([
            "project", "release",
            "--variant", "standard-0805",
            "--version", "v1.0",
        ]) == 0

        manifest = load_manifest(project_dir)
        v0805 = next(v for v in manifest.variants if v.name == "standard-0805")
        assert v0805.status == VariantStatus.RELEASED
        assert v0805.released_tag == "standard-0805/v1.0"

        # ---- Step 8: Walk compact-0603 stages via CLI ----
        assert main([
            "project", "stage", "generate",
            "--variant", "compact-0603",
        ]) == 0
        assert main([
            "project", "stage", "approve",
            "--variant", "compact-0603",
        ]) == 0

        # Verify compact-0603 is at schematic stage now
        manifest = load_manifest(project_dir)
        v0603 = next(v for v in manifest.variants if v.name == "compact-0603")
        req_stage = next(s for s in v0603.stages if s.stage.value == "requirements")
        assert req_stage.state == StageState.APPROVED

    def test_variant_forking_preserves_component_count(
        self, project_dir: Path
    ) -> None:
        """Verify that forking doesn't lose or duplicate components."""
        from kicad_pipeline.orchestrator.models import get_strategy_by_name

        base = _make_test_requirements()
        strategy = get_strategy_by_name("0603")
        assert strategy is not None

        forked = fork_requirements_for_variant(base, strategy)
        assert len(forked.components) == len(base.components)
        assert len(forked.nets) == len(base.nets)

    def test_rollback_and_regenerate(self, project_dir: Path) -> None:
        """Test rollback then regenerate cycle."""
        main(["project", "init", "--name", "rollback-test"])
        main(["project", "variant", "create", "--name", "v1"])

        base_req = _make_test_requirements()
        vdir = project_dir / "variants" / "v1"
        save_requirements(base_req, vdir / "requirements.json")

        engine = WorkflowEngine(project_dir)

        # Generate and approve requirements
        engine.generate_stage("v1")
        engine.approve_stage("v1")

        # Generate schematic
        engine.generate_stage("v1")
        engine.approve_stage("v1")

        # Now rollback to requirements
        result = engine.rollback_stage("v1", engine.get_current_stage("v1"))
        assert result.success

        # Rollback to requirements
        from kicad_pipeline.orchestrator.models import StageId

        result = engine.rollback_stage("v1", StageId.REQUIREMENTS)
        assert result.success

        # Re-walk from the beginning
        engine2 = WorkflowEngine(project_dir)
        current = engine2.get_current_stage("v1")
        assert current == StageId.REQUIREMENTS

        result = engine2.generate_stage("v1")
        assert result.success

    def test_review_returns_data(self, project_dir: Path) -> None:
        """Test that review returns meaningful stage data."""
        main(["project", "init", "--name", "review-test"])
        main(["project", "variant", "create", "--name", "v1"])

        base_req = _make_test_requirements()
        vdir = project_dir / "variants" / "v1"
        save_requirements(base_req, vdir / "requirements.json")

        engine = WorkflowEngine(project_dir)
        engine.generate_stage("v1")

        review = engine.review_stage("v1")
        assert review["stage"] == "requirements"
        assert review["component_count"] == 3
        assert review["net_count"] == 2

    def test_status_shows_all_info(
        self, project_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test status output includes variant and stage info."""
        main(["project", "init", "--name", "status-test"])
        main(["project", "variant", "create", "--name", "v1", "--strategy", "0805"])
        main(["project", "status"])

        output = capsys.readouterr().out
        assert "status-test" in output
        assert "v1" in output
        assert "requirements" in output
        assert "0805" in output
