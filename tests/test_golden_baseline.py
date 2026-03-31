"""Tests for golden image baseline comparison."""
from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003

import pytest

from kicad_pipeline.validation.golden_baseline import (
    compare_to_baseline,
    has_baseline,
    promote_to_baseline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def evidence_dir(tmp_path: Path) -> Path:
    """Create a fake evidence directory with render PNGs."""
    edir = tmp_path / "evidence" / "R_0805"
    edir.mkdir(parents=True)
    for view in ("3d_iso", "3d_top", "2d_top", "3d_bottom"):
        (edir / f"R_0805_{view}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + f"fake_{view}".encode()
        )
    return edir


@pytest.fixture()
def baselines_dir(tmp_path: Path) -> Path:
    """Create a baselines root directory."""
    bdir = tmp_path / "baselines"
    bdir.mkdir()
    return bdir


# ---------------------------------------------------------------------------
# has_baseline()
# ---------------------------------------------------------------------------

class TestHasBaseline:
    def test_no_baseline(self, baselines_dir: Path) -> None:
        assert has_baseline("R_0805", baselines_dir) is False

    def test_with_baseline(self, baselines_dir: Path) -> None:
        (baselines_dir / "R_0805").mkdir()
        (baselines_dir / "R_0805" / "manifest.json").write_text("{}")
        assert has_baseline("R_0805", baselines_dir) is True


# ---------------------------------------------------------------------------
# promote_to_baseline()
# ---------------------------------------------------------------------------

class TestPromoteToBaseline:
    def test_promote_copies_files(
        self, evidence_dir: Path, baselines_dir: Path,
    ) -> None:
        manifest = promote_to_baseline("R_0805", evidence_dir, baselines_dir)

        assert manifest.component_id == "R_0805"
        assert len(manifest.views) == 4  # 3d_iso, 3d_top, 2d_top, 3d_bottom
        assert manifest.promoted_at  # non-empty ISO timestamp
        assert manifest.commit_hash  # non-empty

        # Files were copied
        for view in ("3d_iso", "3d_top", "2d_top", "3d_bottom"):
            assert (baselines_dir / "R_0805" / f"R_0805_{view}.png").exists()

        # Manifest written
        manifest_path = baselines_dir / "R_0805" / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["component_id"] == "R_0805"
        assert len(data["views"]) == 4

    def test_promote_no_views_raises(
        self, tmp_path: Path, baselines_dir: Path,
    ) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No baseline views"):
            promote_to_baseline("R_0805", empty_dir, baselines_dir)

    def test_promote_partial_views(
        self, tmp_path: Path, baselines_dir: Path,
    ) -> None:
        """Only some views present — promotes what exists."""
        edir = tmp_path / "partial"
        edir.mkdir()
        (edir / "R_0805_3d_iso.png").write_bytes(b"\x89PNG\r\n\x1a\niso")

        manifest = promote_to_baseline("R_0805", edir, baselines_dir)
        assert len(manifest.views) == 1
        assert manifest.views[0][0] == "3d_iso"


# ---------------------------------------------------------------------------
# compare_to_baseline()
# ---------------------------------------------------------------------------

class TestCompareToBaseline:
    def test_no_baseline_returns_pass(
        self, evidence_dir: Path, baselines_dir: Path,
    ) -> None:
        result = compare_to_baseline("R_0805", evidence_dir, baselines_dir)
        assert result.has_baseline is False
        assert result.all_match is True
        assert result.matches == ()

    def test_matching_baseline(
        self, evidence_dir: Path, baselines_dir: Path,
    ) -> None:
        # Promote first, then compare — should match exactly
        promote_to_baseline("R_0805", evidence_dir, baselines_dir)
        result = compare_to_baseline("R_0805", evidence_dir, baselines_dir)

        assert result.has_baseline is True
        assert result.all_match is True
        assert len(result.matches) == 4
        assert all(matched for _, matched in result.matches)

    def test_modified_render_detected(
        self, evidence_dir: Path, baselines_dir: Path,
    ) -> None:
        # Promote baseline
        promote_to_baseline("R_0805", evidence_dir, baselines_dir)

        # Modify one evidence file
        (evidence_dir / "R_0805_3d_iso.png").write_bytes(b"MODIFIED CONTENT")

        result = compare_to_baseline("R_0805", evidence_dir, baselines_dir)

        assert result.has_baseline is True
        assert result.all_match is False
        # Find the failing view
        iso_match = next(m for v, m in result.matches if v == "3d_iso")
        assert iso_match is False
        # Other views should still match
        other_matches = [m for v, m in result.matches if v != "3d_iso"]
        assert all(other_matches)

    def test_missing_evidence_file(
        self, evidence_dir: Path, baselines_dir: Path,
    ) -> None:
        promote_to_baseline("R_0805", evidence_dir, baselines_dir)

        # Delete one evidence file
        (evidence_dir / "R_0805_3d_top.png").unlink()

        result = compare_to_baseline("R_0805", evidence_dir, baselines_dir)
        assert result.all_match is False
        top_match = next(m for v, m in result.matches if v == "3d_top")
        assert top_match is False

    def test_corrupt_manifest(
        self, evidence_dir: Path, baselines_dir: Path,
    ) -> None:
        (baselines_dir / "R_0805").mkdir()
        (baselines_dir / "R_0805" / "manifest.json").write_text("not json")

        result = compare_to_baseline("R_0805", evidence_dir, baselines_dir)
        assert result.has_baseline is False
        assert result.all_match is True
