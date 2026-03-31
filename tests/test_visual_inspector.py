"""Tests for Claude-powered visual inspection of component renders."""
from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.validation.visual_inspector import (
    _parse_findings,
    inspect_component,
    is_enabled,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _enable_visual(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISUAL_INSPECT_ENABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")


@pytest.fixture()
def fake_spec() -> Any:
    """Minimal ComponentSpec-like object for testing."""
    spec = MagicMock()
    spec.component_id = "R_0805"
    spec.description = "0805 SMD resistor"
    spec.expected_pads = 2
    spec.expected_pad_type = "smd"
    spec.body_width_mm = 2.0
    spec.body_height_mm = 1.25
    return spec


@pytest.fixture()
def render_paths(tmp_path: Path) -> list[tuple[str, Path]]:
    """Create fake render PNGs."""
    paths: list[tuple[str, Path]] = []
    for view in ("3d_iso", "3d_top", "2d_top"):
        p = tmp_path / f"R_0805_{view}.png"
        # Write a minimal PNG header (8 bytes magic + minimal IHDR)
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        paths.append((view, p))
    return paths


# ---------------------------------------------------------------------------
# is_enabled()
# ---------------------------------------------------------------------------

class TestIsEnabled:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VISUAL_INSPECT_ENABLED", raising=False)
        assert is_enabled() is False

    def test_enabled_with_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VISUAL_INSPECT_ENABLED", "1")
        assert is_enabled() is True

    def test_enabled_with_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VISUAL_INSPECT_ENABLED", "true")
        assert is_enabled() is True

    def test_enabled_with_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VISUAL_INSPECT_ENABLED", "yes")
        assert is_enabled() is True

    def test_disabled_with_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VISUAL_INSPECT_ENABLED", "0")
        assert is_enabled() is False


# ---------------------------------------------------------------------------
# _parse_findings()
# ---------------------------------------------------------------------------

class TestParseFindings:
    def test_valid_json_array(self) -> None:
        text = json.dumps([
            {"check": "body_centered", "passed": True, "confidence": 0.95, "detail": "ok"},
            {"check": "pins_aligned", "passed": False, "confidence": 0.8, "detail": "off by 1mm"},
        ])
        findings = _parse_findings(text)
        assert len(findings) == 2
        assert findings[0].check == "body_centered"
        assert findings[0].passed is True
        assert findings[0].confidence == 0.95
        assert findings[1].passed is False

    def test_json_with_surrounding_text(self) -> None:
        text = (
            'Here are the results:\n'
            '[{"check": "flat_on_pcb", "passed": true, '
            '"confidence": 0.9, "detail": "good"}]\nDone.'
        )
        findings = _parse_findings(text)
        assert len(findings) == 1
        assert findings[0].check == "flat_on_pcb"

    def test_invalid_check_name_filtered(self) -> None:
        text = json.dumps([
            {"check": "body_centered", "passed": True, "confidence": 0.9, "detail": "ok"},
            {"check": "invalid_check", "passed": False, "confidence": 0.5, "detail": "bad"},
        ])
        findings = _parse_findings(text)
        assert len(findings) == 1
        assert findings[0].check == "body_centered"

    def test_no_json_returns_empty(self) -> None:
        findings = _parse_findings("No JSON here, just text.")
        assert findings == ()

    def test_malformed_json_returns_empty(self) -> None:
        findings = _parse_findings("[{broken json]")
        assert findings == ()

    def test_empty_array(self) -> None:
        findings = _parse_findings("[]")
        assert findings == ()


# ---------------------------------------------------------------------------
# inspect_component()
# ---------------------------------------------------------------------------

class TestInspectComponent:
    def test_skipped_when_disabled(
        self,
        fake_spec: Any,
        render_paths: list[tuple[str, Path]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("VISUAL_INSPECT_ENABLED", raising=False)
        result = inspect_component(render_paths, fake_spec)
        assert result.skipped is True
        assert result.passed is True
        assert result.findings == ()

    def test_skipped_when_no_images(
        self,
        fake_spec: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VISUAL_INSPECT_ENABLED", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        result = inspect_component([], fake_spec)
        assert result.skipped is True

    @pytest.mark.usefixtures("_enable_visual")
    def test_successful_inspection(
        self,
        fake_spec: Any,
        render_paths: list[tuple[str, Path]],
    ) -> None:
        findings = [
            {"check": "body_centered", "passed": True,
             "confidence": 0.95, "detail": "centered"},
            {"check": "pins_aligned", "passed": True,
             "confidence": 0.9, "detail": "aligned"},
            {"check": "orientation", "passed": True,
             "confidence": 0.85, "detail": "correct"},
            {"check": "flat_on_pcb", "passed": True,
             "confidence": 0.9, "detail": "flat"},
            {"check": "size_match", "passed": True,
             "confidence": 0.8, "detail": "proportional"},
        ]
        api_response = json.dumps({
            "content": [{"type": "text",
                         "text": json.dumps(findings)}],
        })

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = inspect_component(render_paths, fake_spec)

        assert result.skipped is False
        assert result.passed is True
        assert len(result.findings) == 5
        assert result.model_used != "none"

    @pytest.mark.usefixtures("_enable_visual")
    def test_failed_inspection(
        self,
        fake_spec: Any,
        render_paths: list[tuple[str, Path]],
    ) -> None:
        findings = [
            {"check": "body_centered", "passed": False,
             "confidence": 0.9, "detail": "offset 2mm"},
            {"check": "pins_aligned", "passed": True,
             "confidence": 0.9, "detail": "ok"},
            {"check": "orientation", "passed": True,
             "confidence": 0.85, "detail": "ok"},
            {"check": "flat_on_pcb", "passed": True,
             "confidence": 0.9, "detail": "ok"},
            {"check": "size_match", "passed": True,
             "confidence": 0.8, "detail": "ok"},
        ]
        api_response = json.dumps({
            "content": [{"type": "text",
                         "text": json.dumps(findings)}],
        })

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = inspect_component(render_paths, fake_spec)

        assert result.passed is False
        assert any(not f.passed for f in result.findings)

    @pytest.mark.usefixtures("_enable_visual")
    def test_api_failure_graceful(
        self,
        fake_spec: Any,
        render_paths: list[tuple[str, Path]],
    ) -> None:
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timeout")):
            result = inspect_component(render_paths, fake_spec)

        assert result.skipped is True
        assert result.passed is True

    @pytest.mark.usefixtures("_enable_visual")
    def test_missing_checks_filled(
        self,
        fake_spec: Any,
        render_paths: list[tuple[str, Path]],
    ) -> None:
        """If API returns only 3 of 5 checks, the missing 2 are filled."""
        api_response = json.dumps({
            "content": [{"type": "text", "text": json.dumps([
                {"check": "body_centered", "passed": True, "confidence": 0.9, "detail": "ok"},
                {"check": "pins_aligned", "passed": True, "confidence": 0.9, "detail": "ok"},
                {"check": "orientation", "passed": True, "confidence": 0.9, "detail": "ok"},
            ])}],
        })

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = inspect_component(render_paths, fake_spec)

        assert len(result.findings) == 5
        filled = [f for f in result.findings if f.confidence == 0.0]
        assert len(filled) == 2
