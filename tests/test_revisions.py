"""Tests for kicad_pipeline.orchestrator.revisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.exceptions import OrchestrationError
from kicad_pipeline.orchestrator.models import (
    PackageStrategy,
    ProjectManifest,
    RevisionRecord,
    VariantRecord,
    VariantStatus,
)
from kicad_pipeline.orchestrator.revisions import RevisionManager

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_variant(
    name: str = "standard-0805",
    revisions: tuple[RevisionRecord, ...] = (),
) -> VariantRecord:
    return VariantRecord(
        name=name,
        display_name="Standard 0805",
        description="Test variant",
        status=VariantStatus.DRAFT,
        package_strategy=PackageStrategy(name="0805"),
        revisions=revisions,
    )


def _make_manifest(
    variants: tuple[VariantRecord, ...] = (),
) -> ProjectManifest:
    return ProjectManifest(
        project_name="test-project",
        variants=variants,
    )


def _setup_production(tmp_path: Path, variant_name: str = "standard-0805") -> Path:
    """Create a variant production directory with dummy files."""
    prod = tmp_path / "variants" / variant_name / "production"
    prod.mkdir(parents=True)
    (prod / "gerber.zip").write_text("fake gerber")
    (prod / "bom.csv").write_text("ref,value\nR1,10k")
    return prod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateRevision:
    def test_copies_production_files(self, tmp_path: Path) -> None:
        _setup_production(tmp_path)
        variant = _make_variant()
        manifest = _make_manifest(variants=(variant,))

        mgr = RevisionManager(tmp_path)
        record = mgr.create_revision("standard-0805", manifest, notes="initial")

        assert record.number == 1
        assert record.git_tag == "standard-0805/rev1"
        assert record.notes == "initial"
        assert record.sent_to_fab is False

        rev_prod = tmp_path / "variants" / "standard-0805" / "revisions" / "rev1" / "production"
        assert rev_prod.is_dir()
        assert (rev_prod / "gerber.zip").read_text() == "fake gerber"
        assert (rev_prod / "bom.csv").exists()

    def test_increments_number(self, tmp_path: Path) -> None:
        _setup_production(tmp_path)
        existing_rev = RevisionRecord(
            number=2,
            created_at="2026-01-01T00:00:00+00:00",
            git_tag="standard-0805/rev2",
            commit_hash="abc123",
        )
        variant = _make_variant(revisions=(existing_rev,))
        manifest = _make_manifest(variants=(variant,))

        mgr = RevisionManager(tmp_path)
        record = mgr.create_revision("standard-0805", manifest)

        assert record.number == 3
        assert record.git_tag == "standard-0805/rev3"

    def test_no_production_dir_raises(self, tmp_path: Path) -> None:
        variant = _make_variant()
        manifest = _make_manifest(variants=(variant,))

        mgr = RevisionManager(tmp_path)
        with pytest.raises(OrchestrationError, match="Production directory does not exist"):
            mgr.create_revision("standard-0805", manifest)


class TestGetRevisionPath:
    def test_returns_expected_path(self, tmp_path: Path) -> None:
        mgr = RevisionManager(tmp_path)
        path = mgr.get_revision_path("standard-0805", 3)
        assert path == tmp_path / "variants" / "standard-0805" / "revisions" / "rev3"


class TestListRevisions:
    def test_from_manifest(self, tmp_path: Path) -> None:
        rev1 = RevisionRecord(
            number=1,
            created_at="2026-01-01T00:00:00+00:00",
            git_tag="standard-0805/rev1",
            commit_hash="aaa",
        )
        rev2 = RevisionRecord(
            number=2,
            created_at="2026-02-01T00:00:00+00:00",
            git_tag="standard-0805/rev2",
            commit_hash="bbb",
        )
        variant = _make_variant(revisions=(rev1, rev2))
        manifest = _make_manifest(variants=(variant,))

        mgr = RevisionManager(tmp_path)
        revisions = mgr.list_revisions("standard-0805", manifest)

        assert len(revisions) == 2
        assert revisions[0].number == 1
        assert revisions[1].number == 2

    def test_unknown_variant_returns_empty(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        mgr = RevisionManager(tmp_path)
        assert mgr.list_revisions("nonexistent", manifest) == ()


class TestMarkSentToFab:
    def test_marks_revision(self, tmp_path: Path) -> None:
        rev = RevisionRecord(
            number=1,
            created_at="2026-01-01T00:00:00+00:00",
            git_tag="standard-0805/rev1",
            commit_hash="aaa",
        )
        variant = _make_variant(revisions=(rev,))
        manifest = _make_manifest(variants=(variant,))

        mgr = RevisionManager(tmp_path)
        new_manifest, updated_rev = mgr.mark_sent_to_fab(
            "standard-0805", 1, manifest, order_id="JLCPCB-12345"
        )

        assert updated_rev.sent_to_fab is True
        assert updated_rev.fab_order_id == "JLCPCB-12345"
        # Manifest is also updated
        assert new_manifest.variants[0].revisions[0].sent_to_fab is True

    def test_unknown_revision_raises(self, tmp_path: Path) -> None:
        variant = _make_variant()
        manifest = _make_manifest(variants=(variant,))

        mgr = RevisionManager(tmp_path)
        with pytest.raises(OrchestrationError, match="Revision 99 not found"):
            mgr.mark_sent_to_fab("standard-0805", 99, manifest)
