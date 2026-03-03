"""Tests for manifest serialization and file I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.exceptions import OrchestrationError
from kicad_pipeline.orchestrator.manifest import (
    MANIFEST_FILENAME,
    load_manifest,
    manifest_from_dict,
    manifest_to_dict,
    save_manifest,
)
from kicad_pipeline.orchestrator.models import (
    PackageStrategy,
    ProjectManifest,
    RevisionRecord,
    StageId,
    StageRecord,
    StageState,
    VariantRecord,
    VariantStatus,
    default_stages,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_manifest() -> ProjectManifest:
    """A fully populated manifest for roundtrip testing."""
    return ProjectManifest(
        schema_version=1,
        project_name="test-board",
        description="A test PCB project",
        original_spec="spec.md",
        created_at="2026-03-03T10:00:00",
        updated_at="2026-03-03T14:30:00",
        active_variant="standard-0805",
        variants=(
            VariantRecord(
                name="standard-0805",
                display_name="Standard 0805",
                description="Standard size passives",
                status=VariantStatus.REVIEWING,
                package_strategy=PackageStrategy(
                    name="0805",
                    resistor_package="0805",
                    capacitor_package="0805",
                    led_package="0805",
                    notes="Easy to solder",
                ),
                stages=(
                    StageRecord(
                        stage=StageId.REQUIREMENTS,
                        state=StageState.APPROVED,
                        generated_at="2026-03-03T10:00:00",
                        approved_at="2026-03-03T10:30:00",
                        generation_count=1,
                    ),
                    StageRecord(
                        stage=StageId.SCHEMATIC,
                        state=StageState.GENERATED,
                        generated_at="2026-03-03T11:00:00",
                        generation_count=2,
                        notes=("added decoupling caps",),
                    ),
                    StageRecord(stage=StageId.PCB),
                    StageRecord(stage=StageId.VALIDATION),
                    StageRecord(stage=StageId.PRODUCTION),
                ),
                revisions=(
                    RevisionRecord(
                        number=1,
                        created_at="2026-03-03T12:00:00",
                        git_tag="standard-0805/rev1",
                        commit_hash="abc123def456",
                        notes="Initial production run",
                        sent_to_fab=True,
                        fab_order_id="JLCPCB-99999",
                    ),
                ),
                created_at="2026-03-03T10:00:00",
                updated_at="2026-03-03T14:30:00",
            ),
            VariantRecord(
                name="compact-0603",
                display_name="Compact 0603",
                description="Space-constrained variant",
                status=VariantStatus.DRAFT,
                package_strategy=PackageStrategy(name="0603"),
                stages=default_stages(),
                created_at="2026-03-03T13:00:00",
                updated_at="2026-03-03T13:00:00",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


class TestManifestSerialization:
    def test_roundtrip(self, sample_manifest: ProjectManifest) -> None:
        data = manifest_to_dict(sample_manifest)
        restored = manifest_from_dict(data)
        assert restored == sample_manifest

    def test_minimal_manifest_roundtrip(self) -> None:
        m = ProjectManifest(project_name="bare")
        data = manifest_to_dict(m)
        restored = manifest_from_dict(data)
        assert restored.project_name == "bare"
        assert restored.variants == ()
        assert restored.active_variant is None

    def test_to_dict_structure(self, sample_manifest: ProjectManifest) -> None:
        data = manifest_to_dict(sample_manifest)
        assert data["schema_version"] == 1
        assert data["project_name"] == "test-board"
        assert data["active_variant"] == "standard-0805"
        variants = data["variants"]
        assert isinstance(variants, list)
        assert len(variants) == 2

    def test_variant_dict_structure(self, sample_manifest: ProjectManifest) -> None:
        data = manifest_to_dict(sample_manifest)
        variants = data["variants"]
        assert isinstance(variants, list)
        v0 = variants[0]
        assert isinstance(v0, dict)
        assert v0["name"] == "standard-0805"
        assert v0["status"] == "reviewing"
        stages = v0["stages"]
        assert isinstance(stages, list)
        assert len(stages) == 5
        assert isinstance(stages[0], dict)
        assert stages[0]["stage"] == "requirements"
        assert stages[0]["state"] == "approved"

    def test_revision_dict_structure(self, sample_manifest: ProjectManifest) -> None:
        data = manifest_to_dict(sample_manifest)
        variants = data["variants"]
        assert isinstance(variants, list)
        v0 = variants[0]
        assert isinstance(v0, dict)
        revisions = v0["revisions"]
        assert isinstance(revisions, list)
        assert len(revisions) == 1
        rev = revisions[0]
        assert isinstance(rev, dict)
        assert rev["number"] == 1
        assert rev["sent_to_fab"] is True
        assert rev["fab_order_id"] == "JLCPCB-99999"

    def test_stage_notes_preserved(self, sample_manifest: ProjectManifest) -> None:
        data = manifest_to_dict(sample_manifest)
        restored = manifest_from_dict(data)
        sch_stage = restored.variants[0].stages[1]
        assert sch_stage.notes == ("added decoupling caps",)


class TestManifestFromDictErrors:
    def test_missing_project_name_raises(self) -> None:
        with pytest.raises(OrchestrationError, match="Failed to deserialize"):
            manifest_from_dict({})

    def test_bad_variant_status_raises(self) -> None:
        data = {
            "project_name": "test",
            "variants": [
                {
                    "name": "v1",
                    "display_name": "V1",
                    "status": "invalid_status",
                    "package_strategy": {"name": "0805"},
                }
            ],
        }
        with pytest.raises(OrchestrationError):
            manifest_from_dict(data)

    def test_bad_stage_id_raises(self) -> None:
        data = {
            "project_name": "test",
            "variants": [
                {
                    "name": "v1",
                    "display_name": "V1",
                    "status": "draft",
                    "package_strategy": {"name": "0805"},
                    "stages": [{"stage": "nonexistent", "state": "pending"}],
                }
            ],
        }
        with pytest.raises(OrchestrationError):
            manifest_from_dict(data)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


class TestManifestFileIO:
    def test_save_and_load(
        self, tmp_path: Path, sample_manifest: ProjectManifest
    ) -> None:
        save_manifest(sample_manifest, tmp_path)
        assert (tmp_path / MANIFEST_FILENAME).exists()
        loaded = load_manifest(tmp_path)
        assert loaded == sample_manifest

    def test_save_creates_json_file(
        self, tmp_path: Path, sample_manifest: ProjectManifest
    ) -> None:
        save_manifest(sample_manifest, tmp_path)
        path = tmp_path / MANIFEST_FILENAME
        content = path.read_text(encoding="utf-8")
        assert content.startswith("{")
        assert '"project_name"' in content

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OrchestrationError, match="Cannot read manifest"):
            load_manifest(tmp_path)

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / MANIFEST_FILENAME).write_text("not json", encoding="utf-8")
        with pytest.raises(OrchestrationError, match="not valid JSON"):
            load_manifest(tmp_path)

    def test_load_malformed_data_raises(self, tmp_path: Path) -> None:
        (tmp_path / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
        with pytest.raises(OrchestrationError):
            load_manifest(tmp_path)

    def test_multiple_save_overwrites(
        self, tmp_path: Path, sample_manifest: ProjectManifest
    ) -> None:
        save_manifest(sample_manifest, tmp_path)
        updated = ProjectManifest(
            project_name="updated-board",
            created_at="2026-03-03T15:00:00",
        )
        save_manifest(updated, tmp_path)
        loaded = load_manifest(tmp_path)
        assert loaded.project_name == "updated-board"
