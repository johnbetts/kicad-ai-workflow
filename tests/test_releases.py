"""Tests for kicad_pipeline.github.releases."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.github.releases import (
    ReleaseAsset,
    ReleaseCreateResult,
    create_production_release,
    create_release,
)

if TYPE_CHECKING:
    from pathlib import Path

_RELEASE_URL = "https://github.com/user/repo/releases/tag/v1.0.0"


# ---------------------------------------------------------------------------
# Frozen dataclass tests
# ---------------------------------------------------------------------------


def test_release_asset_frozen() -> None:
    asset = ReleaseAsset(name="gerbers.zip", path="/output/gerbers.zip")
    with pytest.raises(dataclasses.FrozenInstanceError):
        asset.name = "other.zip"  # type: ignore[misc]


def test_release_create_result_frozen() -> None:
    result = ReleaseCreateResult(
        success=True,
        tag="v1.0.0",
        url=_RELEASE_URL,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# create_release
# ---------------------------------------------------------------------------


def test_create_release_success() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _RELEASE_URL + "\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = create_release(
            tag="v1.0.0",
            title="Release 1.0.0",
            notes="Initial release",
        )

    assert result.success is True
    assert result.url != ""
    assert "v1.0.0" in result.url


def test_create_release_failure() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "gh: release already exists"

    with patch("subprocess.run", return_value=mock_result):
        result = create_release(
            tag="v1.0.0",
            title="Release 1.0.0",
            notes="notes",
        )

    assert result.success is False
    assert result.url == ""
    assert result.error != ""


def test_release_create_result_has_tag() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _RELEASE_URL + "\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = create_release(
            tag="v2.3.4",
            title="Some Title",
            notes="notes",
        )

    assert result.tag == "v2.3.4"


def test_create_release_builds_command() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _RELEASE_URL + "\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        create_release(
            tag="v1.0.0",
            title="My Title",
            notes="release notes",
        )

    call_args = mock_run.call_args[0][0]
    cmd_str = " ".join(call_args)
    assert "v1.0.0" in cmd_str
    assert "My Title" in cmd_str


# ---------------------------------------------------------------------------
# create_production_release
# ---------------------------------------------------------------------------


def test_create_production_release_title_format(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _RELEASE_URL + "\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        create_production_release(
            tag="v1.0.0",
            project_name="MyPCB",
            production_dir=str(tmp_path),
        )

    call_args = mock_run.call_args[0][0]
    title_idx = call_args.index("--title") + 1
    assert "MyPCB" in call_args[title_idx]
    assert "v1.0.0" in call_args[title_idx]
    assert "Production Release" in call_args[title_idx]


def test_create_production_release_no_assets_ok(tmp_path: Path) -> None:
    """An empty production dir should still create a release."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _RELEASE_URL + "\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = create_production_release(
            tag="v1.0.0",
            project_name="EmptyProject",
            production_dir=str(tmp_path),
        )

    assert result.success is True


def test_create_production_release_includes_zip_assets(tmp_path: Path) -> None:
    (tmp_path / "gerbers.zip").write_text("zip content", encoding="utf-8")
    (tmp_path / "bom.csv").write_text("ref,value\n", encoding="utf-8")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _RELEASE_URL + "\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        create_production_release(
            tag="v1.0.0",
            project_name="TestProject",
            production_dir=str(tmp_path),
        )

    call_args = mock_run.call_args[0][0]
    cmd_str = " ".join(call_args)
    assert "gerbers.zip" in cmd_str
    assert "bom.csv" in cmd_str
