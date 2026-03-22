"""Tests for the datasheet downloader module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.models.requirements import (
    Component,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.research.datasheets import (
    _lcsc_url,
    _sanitize_filename,
    download_datasheets,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_requirements(*components: Component) -> ProjectRequirements:
    """Build minimal requirements with given components."""
    return ProjectRequirements(
        project=ProjectInfo(name="test-project"),
        features=(),
        components=components,
        nets=(),
    )


def _make_component(
    ref: str,
    value: str = "10k",
    datasheet: str | None = None,
    lcsc: str | None = None,
) -> Component:
    return Component(
        ref=ref,
        value=value,
        footprint="R_0805",
        datasheet=datasheet,
        lcsc=lcsc,
        pins=(
            Pin(number="1", name="1", pin_type=PinType.PASSIVE),
            Pin(number="2", name="2", pin_type=PinType.PASSIVE),
        ),
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_replaces_spaces(self) -> None:
        assert _sanitize_filename("hello world") == "hello_world"

    def test_replaces_special_chars(self) -> None:
        assert _sanitize_filename("a/b:c") == "a_b_c"

    def test_keeps_safe_chars(self) -> None:
        assert _sanitize_filename("R1_10k.pdf") == "R1_10k.pdf"


class TestLcscUrl:
    def test_builds_url(self) -> None:
        url = _lcsc_url("C17414")
        assert url == "https://www.lcsc.com/product-detail/C17414.html"


# ---------------------------------------------------------------------------
# Download tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestDownloadDatasheets:
    def test_no_url_components(self, tmp_path: Path) -> None:
        """Components without datasheet or lcsc get status 'no_url'."""
        req = _make_requirements(
            _make_component("R1", datasheet=None, lcsc=None),
        )
        results = download_datasheets(req, tmp_path / "datasheets")
        assert len(results) == 1
        assert results[0].status == "no_url"
        assert results[0].local_path is None

    def test_cached_file_skipped(self, tmp_path: Path) -> None:
        """Pre-existing PDFs are marked 'cached' without downloading."""
        ds_dir = tmp_path / "datasheets"
        ds_dir.mkdir()
        (ds_dir / "R1_10k.pdf").write_bytes(b"%PDF-fake")

        req = _make_requirements(
            _make_component("R1", datasheet="https://example.com/r.pdf"),
        )
        results = download_datasheets(req, ds_dir)
        assert len(results) == 1
        assert results[0].status == "cached"
        assert results[0].local_path == ds_dir / "R1_10k.pdf"

    @patch("kicad_pipeline.research.datasheets.urllib.request.urlopen")
    def test_successful_download(
        self, mock_urlopen: MagicMock, tmp_path: Path,
    ) -> None:
        """Successful HTTP fetch writes file and returns 'downloaded'."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"%PDF-1.4 fake datasheet"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        req = _make_requirements(
            _make_component("U1", value="ADS1115", datasheet="https://example.com/ads1115.pdf"),
        )
        results = download_datasheets(req, tmp_path / "datasheets")
        assert len(results) == 1
        assert results[0].status == "downloaded"
        assert results[0].local_path is not None
        assert results[0].local_path.exists()
        assert results[0].local_path.read_bytes() == b"%PDF-1.4 fake datasheet"

    @patch("kicad_pipeline.research.datasheets.urllib.request.urlopen")
    def test_failed_download(
        self, mock_urlopen: MagicMock, tmp_path: Path,
    ) -> None:
        """HTTP error results in 'failed' status."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        req = _make_requirements(
            _make_component("U1", value="ADS1115", datasheet="https://example.com/ads.pdf"),
        )
        results = download_datasheets(req, tmp_path / "datasheets")
        assert len(results) == 1
        assert results[0].status == "failed"
        assert results[0].local_path is None

    @patch("kicad_pipeline.research.datasheets.urllib.request.urlopen")
    def test_lcsc_fallback(
        self, mock_urlopen: MagicMock, tmp_path: Path,
    ) -> None:
        """Component with lcsc but no datasheet URL falls back to LCSC page."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html>product page</html>"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        req = _make_requirements(
            _make_component("R1", lcsc="C17414", datasheet=None),
        )
        results = download_datasheets(req, tmp_path / "datasheets")
        assert len(results) == 1
        assert results[0].status == "downloaded"
        assert "C17414" in results[0].url

    def test_multiple_components(self, tmp_path: Path) -> None:
        """Multiple components each get their own result."""
        req = _make_requirements(
            _make_component("R1"),
            _make_component("R2"),
            _make_component("C1", value="100nF"),
        )
        results = download_datasheets(req, tmp_path / "datasheets")
        assert len(results) == 3
        refs = {r.ref for r in results}
        assert refs == {"R1", "R2", "C1"}
