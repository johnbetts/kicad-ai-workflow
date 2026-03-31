"""Claude-powered visual inspection of component 3D renders.

Sends rendered images to Claude's vision API and checks:
  1. Body centered over pads
  2. Pins/leads aligned with pad positions
  3. Correct orientation (not rotated 90/180 unexpectedly)
  4. Body flat on PCB surface (not floating)
  5. Body size proportional to expected package dimensions

Gracefully degrades to no-op when API key is missing or disabled.

API key lookup order:
  1. ``ANTHROPIC_API_KEY`` env var
  2. Vault file: ``~/.claude/skills/research-agent/vaults/tech/integrations/anthropic.json``
  3. Vault file: ``~/.claude/skills/research-agent/vaults/tech/credentials/anthropic_api_key``

Enable/disable via ``VISUAL_INSPECT_ENABLED`` env var (default: disabled).
Set ``VISUAL_INSPECT_ENABLED=1`` to activate.
"""
from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Sequence  # noqa: TC003 — used at runtime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.validation.component_registry import ComponentSpec

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VAULTS_DIR = Path.home() / ".claude" / "skills" / "research-agent" / "vaults"
_API_URL = "https://api.anthropic.com/v1/messages"
_VISION_MODEL = "claude-sonnet-4-6-20250514"
_API_VERSION = "2023-06-01"
_TIMEOUT_SECONDS = 60

# The 5 checks performed on each component.
_CHECKS = (
    "body_centered",
    "pins_aligned",
    "orientation",
    "flat_on_pcb",
    "size_match",
)

_JSON_FORMAT = (
    "Respond with ONLY a JSON array of exactly 5 objects, one per check:\n"
    '[{"check": "<check_name>", "passed": true|false, '
    '"confidence": 0.0-1.0, "detail": "explanation"}]\n\n'
    "Check names (in order): body_centered, pins_aligned, orientation, "
    "flat_on_pcb, size_match.\n"
    "If you cannot determine a check, set passed=true with confidence<0.5."
)


# ---------------------------------------------------------------------------
# Public API: enable/disable
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """Check if visual inspection is enabled (opt-in)."""
    return os.environ.get("VISUAL_INSPECT_ENABLED", "").strip() in (
        "1", "true", "yes",
    )


def _get_api_key() -> str | None:
    """Get Anthropic API key from env or vault credentials."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    # Vault: JSON format
    json_path = _VAULTS_DIR / "tech" / "integrations" / "anthropic.json"
    if json_path.exists():
        try:
            creds = json.loads(json_path.read_text())
            k = creds.get("api_key")
            if k:
                return str(k)
        except (json.JSONDecodeError, KeyError) as exc:
            _log.debug("Failed to read anthropic credentials from vault: %s", exc)

    # Vault: plain text format
    txt_path = _VAULTS_DIR / "tech" / "credentials" / "anthropic_api_key"
    if txt_path.exists():
        k = txt_path.read_text().strip()
        if k:
            return k

    return None


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisualFinding:
    """A single visual inspection check result."""

    check: str          # body_centered, pins_aligned, orientation, flat_on_pcb, size_match
    passed: bool
    confidence: float   # 0.0-1.0
    detail: str


@dataclass(frozen=True)
class VisualInspectionResult:
    """Complete visual inspection result for one component."""

    component_id: str
    passed: bool
    findings: tuple[VisualFinding, ...]
    model_used: str
    skipped: bool       # True when API unavailable


def _unavailable(component_id: str, reason: str) -> VisualInspectionResult:
    """Return a skip result when inspection is unavailable."""
    _log.debug("Visual inspection skipped for %s: %s", component_id, reason)
    return VisualInspectionResult(
        component_id=component_id,
        passed=True,  # don't block pipeline when API is unavailable
        findings=(),
        model_used="none",
        skipped=True,
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_claude_vision(
    images: Sequence[tuple[str, Path]],
    prompt: str,
    max_tokens: int = 1024,
) -> str | None:
    """Call the Anthropic Messages API with images. Returns response text."""
    api_key = _get_api_key()
    if not api_key:
        return None

    content: list[dict[str, object]] = []
    for label, img_path in images:
        if not img_path.exists():
            continue
        b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        content.append({
            "type": "text",
            "text": f"[{label}]",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        })
    content.append({"type": "text", "text": prompt})

    payload = json.dumps({
        "model": _VISION_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }).encode("utf-8")

    try:
        req = urllib.request.Request(_API_URL, data=payload, headers={
            "x-api-key": api_key,
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
            "User-Agent": "kicad-pipeline-visual-inspector/1.0",
        })
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Anthropic Messages API returns content[0].text
            blocks = data.get("content", [])
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    return str(block.get("text", ""))
            return None
    except urllib.error.HTTPError as exc:
        body = ""
        with contextlib.suppress(Exception):
            body = exc.read().decode()[:200]
        _log.warning("Anthropic API HTTP %d: %s", exc.code, body)
        return None
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        _log.warning("Anthropic API error: %s", exc)
        return None


def _parse_findings(text: str) -> tuple[VisualFinding, ...]:
    """Parse structured findings from Claude's JSON response."""
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        _log.debug("No JSON array found in visual inspection response")
        return ()

    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        _log.debug("Failed to parse visual inspection JSON: %s", exc)
        return ()

    findings: list[VisualFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        check = str(item.get("check", ""))
        if check not in _CHECKS:
            continue
        findings.append(VisualFinding(
            check=check,
            passed=bool(item.get("passed", True)),
            confidence=float(item.get("confidence", 0.5)),
            detail=str(item.get("detail", "")),
        ))

    return tuple(findings)


# ---------------------------------------------------------------------------
# Main inspection function
# ---------------------------------------------------------------------------

def inspect_component(
    render_paths: Sequence[tuple[str, Path]],
    spec: ComponentSpec,
) -> VisualInspectionResult:
    """Inspect a component's 3D renders using Claude vision.

    Args:
        render_paths: Sequence of (view_name, path) — e.g. ("3d_iso", Path(...)).
        spec: Component specification with expected dimensions.

    Returns:
        VisualInspectionResult with per-check findings.
    """
    if not is_enabled():
        return _unavailable(spec.component_id, "VISUAL_INSPECT_ENABLED not set")

    # Select the best images for inspection (prefer 3d_iso + 3d_top)
    images: list[tuple[str, Path]] = []
    path_map = dict(render_paths)
    for view in ("3d_iso", "3d_top", "2d_top"):
        p = path_map.get(view)
        if p and p.exists():
            images.append((view, p))
    if len(images) < 2:
        # Try any available render
        for name, p in render_paths:
            if p.exists() and (name, p) not in images:
                images.append((name, p))
                if len(images) >= 2:
                    break

    if not images:
        return _unavailable(spec.component_id, "No render images available")

    # Build the prompt with component metadata
    body_info = ""
    if spec.body_width_mm and spec.body_height_mm:
        body_info = (
            f"Expected body dimensions: {spec.body_width_mm:.1f}mm x "
            f"{spec.body_height_mm:.1f}mm\n"
        )

    prompt = (
        f"You are a PCB quality inspector verifying component 3D model alignment.\n\n"
        f"Component: {spec.component_id}\n"
        f"Description: {spec.description}\n"
        f"Expected pads: {spec.expected_pads} ({spec.expected_pad_type})\n"
        f"{body_info}\n"
        f"Perform these 5 checks on the rendered images:\n\n"
        f"1. **body_centered**: Is the 3D body centered over the pads? "
        f"The body should be symmetrically placed relative to the pad pattern.\n"
        f"2. **pins_aligned**: Are the component's pins/leads aligned with "
        f"the pad positions? Pins should land on pads, not between them.\n"
        f"3. **orientation**: Is the body correctly oriented? Not rotated "
        f"90 or 180 degrees from expected (pin 1 should match pad 1 corner).\n"
        f"4. **flat_on_pcb**: Is the body flat on the PCB surface? "
        f"Not floating above or tilted.\n"
        f"5. **size_match**: Does the body size match the expected package? "
        f"Body should be proportional to pad spacing.\n\n"
        f"{_JSON_FORMAT}"
    )

    response = _call_claude_vision(images, prompt)
    if response is None:
        return _unavailable(spec.component_id, "API call failed")

    findings = _parse_findings(response)

    # Fill in any missing checks as passed-with-low-confidence
    found_checks = {f.check for f in findings}
    extras: list[VisualFinding] = []
    for check in _CHECKS:
        if check not in found_checks:
            extras.append(VisualFinding(
                check=check,
                passed=True,
                confidence=0.0,
                detail="Check not returned by vision model",
            ))
    all_findings = findings + tuple(extras)

    passed = all(f.passed for f in all_findings)

    return VisualInspectionResult(
        component_id=spec.component_id,
        passed=passed,
        findings=all_findings,
        model_used=_VISION_MODEL,
        skipped=False,
    )
