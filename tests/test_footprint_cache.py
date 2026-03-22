"""Tests for footprint_cache — JLCPCB footprint download and caching."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from kicad_pipeline.parts.footprint_cache import (
    _find_in_cache,
    _is_negative_cached,
    _mark_negative_cache,
    get_jlcpcb_footprint,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def fake_cache(tmp_path: Path) -> Path:
    """Create a temporary cache directory with a pre-cached footprint."""
    cache = tmp_path / "footprints"
    pretty = cache / "C12345.pretty"
    pretty.mkdir(parents=True)
    mod = pretty / "TestPart.kicad_mod"
    mod.write_text(textwrap.dedent("""\
        (module test:TestPart (layer F.Cu)
            (attr smd)
            (pad 1 smd rect (at -1 0) (size 1 1) (layers F.Cu F.Paste F.Mask))
        )
    """))
    return cache


class TestFindInCache:
    """Test cache lookup logic."""

    def test_finds_cached_file(self, fake_cache: Path) -> None:
        with patch("kicad_pipeline.parts.footprint_cache._CACHE_DIR", fake_cache):
            result = _find_in_cache("C12345")
            assert result is not None
            assert result.suffix == ".kicad_mod"
            assert "C12345" in str(result.parent)

    def test_returns_none_for_missing(self, fake_cache: Path) -> None:
        with patch("kicad_pipeline.parts.footprint_cache._CACHE_DIR", fake_cache):
            result = _find_in_cache("C99999")
            assert result is None

    def test_returns_none_for_no_cache_dir(self, tmp_path: Path) -> None:
        with patch(
            "kicad_pipeline.parts.footprint_cache._CACHE_DIR",
            tmp_path / "nonexistent",
        ):
            result = _find_in_cache("C12345")
            assert result is None


class TestGetJlcpcbFootprint:
    """Test the main resolution function."""

    def test_returns_cached(self, fake_cache: Path) -> None:
        with patch("kicad_pipeline.parts.footprint_cache._CACHE_DIR", fake_cache):
            result = get_jlcpcb_footprint("C12345")
            assert result is not None
            assert result.exists()

    def test_empty_lcsc_returns_none(self) -> None:
        assert get_jlcpcb_footprint("") is None

    def test_download_fallback(self, tmp_path: Path) -> None:
        """Test that download is attempted when cache misses."""
        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir()

        with (
            patch("kicad_pipeline.parts.footprint_cache._CACHE_DIR", empty_cache),
            patch("kicad_pipeline.parts.footprint_cache._CDFER_BASE", tmp_path / "nocdfer"),
            patch("kicad_pipeline.parts.footprint_cache._download_footprint") as mock_dl,
        ):
            mock_dl.return_value = None
            result = get_jlcpcb_footprint("C99999")
            assert result is None
            mock_dl.assert_called_once_with("C99999")


class TestNegativeCache:
    """Test negative cache prevents repeated failed downloads."""

    def test_mark_and_check(self, tmp_path: Path) -> None:
        cache = tmp_path / "footprints"
        cache.mkdir()
        with patch("kicad_pipeline.parts.footprint_cache._CACHE_DIR", cache):
            assert not _is_negative_cached("C44857")
            _mark_negative_cache("C44857")
            assert _is_negative_cached("C44857")

    def test_negative_cache_skips_download(self, tmp_path: Path) -> None:
        """Once a part fails download, get_jlcpcb_footprint skips it."""
        cache = tmp_path / "footprints"
        cache.mkdir()
        # Write negative marker
        (cache / "C44857.notfound").touch()

        with (
            patch("kicad_pipeline.parts.footprint_cache._CACHE_DIR", cache),
            patch("kicad_pipeline.parts.footprint_cache._download_footprint") as mock_dl,
        ):
            result = get_jlcpcb_footprint("C44857")
            assert result is None
            mock_dl.assert_not_called()

    def test_positive_cache_still_works(self, fake_cache: Path) -> None:
        """Positive cache hit is not affected by negative cache logic."""
        with patch("kicad_pipeline.parts.footprint_cache._CACHE_DIR", fake_cache):
            result = get_jlcpcb_footprint("C12345")
            assert result is not None


class TestIntegration:
    """Integration tests with real easyeda2kicad (skipped if not available)."""

    @pytest.fixture()
    def has_easyeda2kicad(self) -> bool:
        import shutil
        return shutil.which("easyeda2kicad") is not None

    def test_real_download_esp32(self, has_easyeda2kicad: bool) -> None:
        if not has_easyeda2kicad:
            pytest.skip("easyeda2kicad not installed")
        result = get_jlcpcb_footprint("C2913202")
        assert result is not None
        assert result.exists()
        assert result.suffix == ".kicad_mod"
