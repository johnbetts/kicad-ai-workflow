"""Tests for CLI project commands."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.cli.main import main
from kicad_pipeline.orchestrator.manifest import (
    MANIFEST_FILENAME,
    load_manifest,
)
from kicad_pipeline.orchestrator.models import (
    StageState,
    VariantStatus,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Change to a temp dir and return it."""
    original = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original)


@pytest.fixture()
def initialized_project(project_dir: Path) -> Path:
    """A project dir with init already run."""
    main(["project", "init", "--name", "test-board"])
    return project_dir


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestProjectInit:
    def test_init_creates_manifest(self, project_dir: Path) -> None:
        result = main(["project", "init", "--name", "my-board"])
        assert result == 0
        assert (project_dir / MANIFEST_FILENAME).exists()

    def test_init_creates_directories(self, project_dir: Path) -> None:
        main(["project", "init", "--name", "my-board"])
        assert (project_dir / "base").is_dir()
        assert (project_dir / "variants").is_dir()

    def test_init_stores_name(self, project_dir: Path) -> None:
        main(["project", "init", "--name", "my-board", "-d", "A test board"])
        manifest = load_manifest(project_dir)
        assert manifest.project_name == "my-board"
        assert manifest.description == "A test board"

    def test_init_twice_fails(self, project_dir: Path) -> None:
        main(["project", "init", "--name", "my-board"])
        result = main(["project", "init", "--name", "my-board"])
        assert result == 1


# ---------------------------------------------------------------------------
# Variant
# ---------------------------------------------------------------------------


class TestProjectVariant:
    def test_create_variant(self, initialized_project: Path) -> None:
        result = main([
            "project", "variant", "create",
            "--name", "standard-0805",
            "--strategy", "0805",
        ])
        assert result == 0
        manifest = load_manifest(initialized_project)
        assert len(manifest.variants) == 1
        assert manifest.variants[0].name == "standard-0805"

    def test_create_variant_directory(self, initialized_project: Path) -> None:
        main(["project", "variant", "create", "--name", "compact-0603"])
        assert (initialized_project / "variants" / "compact-0603").is_dir()

    def test_create_variant_sets_active(self, initialized_project: Path) -> None:
        main(["project", "variant", "create", "--name", "v1"])
        manifest = load_manifest(initialized_project)
        assert manifest.active_variant == "v1"

    def test_create_duplicate_fails(self, initialized_project: Path) -> None:
        main(["project", "variant", "create", "--name", "v1"])
        result = main(["project", "variant", "create", "--name", "v1"])
        assert result == 1

    def test_variant_list(
        self, initialized_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["project", "variant", "create", "--name", "v1"])
        main(["project", "variant", "create", "--name", "v2"])
        main(["project", "variant", "list"])
        output = capsys.readouterr().out
        assert "v1" in output
        assert "v2" in output

    def test_variant_activate(self, initialized_project: Path) -> None:
        main(["project", "variant", "create", "--name", "v1"])
        main(["project", "variant", "create", "--name", "v2"])
        main(["project", "variant", "activate", "v2"])
        manifest = load_manifest(initialized_project)
        assert manifest.active_variant == "v2"

    def test_activate_unknown_fails(self, initialized_project: Path) -> None:
        result = main(["project", "variant", "activate", "nonexistent"])
        assert result == 1

    def test_create_variant_with_stages(self, initialized_project: Path) -> None:
        main(["project", "variant", "create", "--name", "v1"])
        manifest = load_manifest(initialized_project)
        assert len(manifest.variants[0].stages) == 5


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestProjectStatus:
    def test_status_empty(
        self, initialized_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = main(["project", "status"])
        assert result == 0
        output = capsys.readouterr().out
        assert "test-board" in output

    def test_status_with_variant(
        self, initialized_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["project", "variant", "create", "--name", "v1"])
        main(["project", "status"])
        output = capsys.readouterr().out
        assert "v1" in output
        assert "requirements" in output


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class TestProjectStage:
    def _setup_variant_with_requirements(self, project_dir: Path) -> None:
        """Create a variant with a valid requirements.json."""
        from kicad_pipeline.models.requirements import (
            Component,
            Net,
            NetConnection,
            Pin,
            PinType,
            ProjectInfo,
            ProjectRequirements,
        )
        from kicad_pipeline.requirements.decomposer import save_requirements

        main(["project", "variant", "create", "--name", "v1"])
        vdir = project_dir / "variants" / "v1"

        req = ProjectRequirements(
            project=ProjectInfo(name="test"),
            features=(),
            components=(
                Component(
                    ref="R1", value="10k", footprint="R_0805",
                    pins=(
                        Pin(number="1", name="1", pin_type=PinType.PASSIVE),
                        Pin(number="2", name="2", pin_type=PinType.PASSIVE),
                    ),
                ),
                Component(
                    ref="R2", value="330R", footprint="R_0805",
                    pins=(
                        Pin(number="1", name="1", pin_type=PinType.PASSIVE),
                        Pin(number="2", name="2", pin_type=PinType.PASSIVE),
                    ),
                ),
            ),
            nets=(
                Net(name="GND", connections=(
                    NetConnection(ref="R1", pin="1"),
                    NetConnection(ref="R2", pin="1"),
                )),
            ),
        )
        save_requirements(req, vdir / "requirements.json")

    def test_generate_requirements(self, initialized_project: Path) -> None:
        self._setup_variant_with_requirements(initialized_project)
        result = main(["project", "stage", "generate", "--variant", "v1"])
        assert result == 0

    def test_approve_after_generate(self, initialized_project: Path) -> None:
        self._setup_variant_with_requirements(initialized_project)
        main(["project", "stage", "generate", "--variant", "v1"])
        result = main(["project", "stage", "approve", "--variant", "v1"])
        assert result == 0

    def test_approve_without_generate_fails(self, initialized_project: Path) -> None:
        self._setup_variant_with_requirements(initialized_project)
        result = main(["project", "stage", "approve", "--variant", "v1"])
        assert result == 1

    def test_review_requirements(
        self,
        initialized_project: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._setup_variant_with_requirements(initialized_project)
        main(["project", "stage", "generate", "--variant", "v1"])
        result = main(["project", "stage", "review", "--variant", "v1"])
        assert result == 0
        output = capsys.readouterr().out
        assert "component_count" in output

    def test_rollback(self, initialized_project: Path) -> None:
        self._setup_variant_with_requirements(initialized_project)
        main(["project", "stage", "generate", "--variant", "v1"])
        main(["project", "stage", "approve", "--variant", "v1"])
        result = main([
            "project", "stage", "rollback",
            "--variant", "v1", "--to", "requirements",
        ])
        assert result == 0
        manifest = load_manifest(initialized_project)
        v = manifest.variants[0]
        for sr in v.stages:
            assert sr.state == StageState.PENDING


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------


class TestProjectRelease:
    def test_release_updates_status(self, initialized_project: Path) -> None:
        main(["project", "variant", "create", "--name", "v1"])
        result = main(["project", "release", "--variant", "v1", "--version", "v1.0"])
        assert result == 0
        manifest = load_manifest(initialized_project)
        assert manifest.variants[0].status == VariantStatus.RELEASED
        assert manifest.variants[0].released_tag == "v1/v1.0"
