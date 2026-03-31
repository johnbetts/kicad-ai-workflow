"""Grok-powered review — code review + PCB design review with vision.

Uses the xAI API (OpenAI-compatible) with ``grok-2-vision`` for image
analysis and ``grok-3`` for code review.  Gracefully degrades to no-op
when API key is missing or credits are exhausted.

API key lookup order:
  1. ``GROK_API_KEY`` env var
  2. ``XAI_API_KEY`` env var
  3. Vault file: ``~/.claude/skills/research-agent/vaults/tech/integrations/grok.json``
  4. Vault file: ``~/.claude/skills/research-agent/vaults/tech/credentials/grok_api_key``

Enable/disable via ``GROK_REVIEW_ENABLED`` env var (default: disabled).
Set ``GROK_REVIEW_ENABLED=1`` to activate once you have API credits.
"""
from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

# Shared JSON output format instruction for all review prompts.
_JSON_FORMAT = (
    'Respond with a JSON array of findings:\n'
    '[{"severity": "critical|major|minor|info", '
    '"category": "<category>", '
    '"description": "...", "suggestion": "..."}]\n'
    'If everything looks good, return [].'
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VAULTS_DIR = Path.home() / ".claude" / "skills" / "research-agent" / "vaults"
_API_URL = "https://api.x.ai/v1/chat/completions"
_CODE_MODEL = "grok-3"
_VISION_MODEL = "grok-2-vision"
_TIMEOUT_SECONDS = 60


def is_enabled() -> bool:
    """Check if Grok review is enabled (opt-in)."""
    return os.environ.get("GROK_REVIEW_ENABLED", "").strip() in ("1", "true", "yes")


def _get_api_key() -> str | None:
    """Get Grok/xAI API key from env or vault credentials."""
    key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
    if key:
        return key

    # Vault: JSON format
    json_path = _VAULTS_DIR / "tech" / "integrations" / "grok.json"
    if json_path.exists():
        try:
            creds = json.loads(json_path.read_text())
            k = creds.get("api_key")
            if k:
                return str(k)
        except (json.JSONDecodeError, KeyError) as exc:
            _log.debug("Failed to read grok credentials from JSON vault: %s", exc)

    # Vault: plain text format
    txt_path = _VAULTS_DIR / "tech" / "credentials" / "grok_api_key"
    if txt_path.exists():
        k = txt_path.read_text().strip()
        if k:
            return k

    return None


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GrokFinding:
    """A single finding from Grok review."""

    severity: str  # "critical", "major", "minor", "info"
    category: str  # "placement", "3d_alignment", "code_quality", etc.
    description: str
    suggestion: str = ""


@dataclass(frozen=True)
class GrokReviewResult:
    """Complete Grok review result."""

    review_type: str  # "pcb_design", "code", "3d_model"
    model_used: str
    passed: bool
    summary: str
    findings: tuple[GrokFinding, ...]
    raw_response: str = ""
    error: str | None = None


def _unavailable(review_type: str, reason: str) -> GrokReviewResult:
    """Return a skip result when Grok is unavailable."""
    return GrokReviewResult(
        review_type=review_type,
        model_used="none",
        passed=True,  # don't block pipeline when Grok is unavailable
        summary=f"Grok review skipped: {reason}",
        findings=(),
        error=reason,
    )


# ---------------------------------------------------------------------------
# API call helpers
# ---------------------------------------------------------------------------

def _call_grok(
    messages: list[dict[str, object]],
    model: str = _CODE_MODEL,
    max_tokens: int = 2048,
) -> str | None:
    """Call the xAI chat completions API. Returns response text or None."""
    api_key = _get_api_key()
    if not api_key:
        return None

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(_API_URL, data=payload, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "kicad-pipeline-evals/1.0",
        })
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
    except urllib.error.HTTPError as exc:
        body = ""
        with contextlib.suppress(Exception):
            body = exc.read().decode()[:200]
        _log.warning("Grok API HTTP %d: %s", exc.code, body)
        return None
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        _log.warning("Grok API error: %s", exc)
        return None


def _encode_image(image_path: Path) -> str:
    """Base64-encode an image file for the vision API."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _parse_findings(text: str) -> tuple[GrokFinding, ...]:
    """Parse structured findings from Grok's JSON response."""
    # Try to extract JSON from the response
    findings: list[GrokFinding] = []

    # Look for JSON array in the response
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            items = json.loads(text[start:end + 1])
            for item in items:
                if isinstance(item, dict):
                    findings.append(GrokFinding(
                        severity=str(item.get("severity", "info")),
                        category=str(item.get("category", "general")),
                        description=str(item.get("description", "")),
                        suggestion=str(item.get("suggestion", "")),
                    ))
        except json.JSONDecodeError as exc:
            _log.debug("Failed to parse Grok JSON response as findings list: %s", exc)

    if not findings:
        # Fallback: treat whole response as a single finding
        findings.append(GrokFinding(
            severity="info",
            category="general",
            description=text[:500],
        ))

    return tuple(findings)


# ---------------------------------------------------------------------------
# Review functions
# ---------------------------------------------------------------------------

def review_pcb_design(
    image_2d: Path,
    image_3d: Path,
    board_name: str = "",
    component_count: int = 0,
) -> GrokReviewResult:
    """Review PCB layout using Grok vision on 2D and 3D renders.

    Sends both the 2D editor view (pad labels, ratsnest) and 3D render
    (body alignment, physical plausibility) to Grok for analysis.

    Args:
        image_2d: Path to 2D top-view PNG.
        image_3d: Path to 3D isometric PNG.
        board_name: Board identifier for context.
        component_count: Number of components for context.

    Returns:
        GrokReviewResult with findings.
    """
    if not is_enabled():
        return _unavailable("pcb_design", "GROK_REVIEW_ENABLED not set")

    if not image_2d.exists() or not image_3d.exists():
        return _unavailable(
            "pcb_design",
            f"Missing images: 2d={image_2d.exists()}, 3d={image_3d.exists()}",
        )

    prompt = (
        f"You are an expert PCB fabricator and electrical engineer reviewing a board layout.\n"
        f"Board: {board_name or 'unknown'} ({component_count} components)\n\n"
        f"I'm sending you two views:\n"
        f"1. 2D editor view — shows pads, silkscreen, pad labels, ratsnest lines\n"
        f"2. 3D isometric view — shows component bodies, physical alignment\n\n"
        f"Review for:\n"
        f"- Component placement quality (grouping, signal flow)\n"
        f"- 3D body alignment (bodies centered on pads, correct orientation)\n"
        f"- Connector placement (near board edges?)\n"
        f"- Decoupling cap proximity to ICs\n"
        f"- Any overlapping components or off-board parts\n"
        f"- Overall manufacturability\n\n"
        f"{_JSON_FORMAT}"
    )

    messages: list[dict[str, object]] = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_encode_image(image_2d)}",
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_encode_image(image_3d)}",
                },
            },
        ],
    }]

    response = _call_grok(messages, model=_VISION_MODEL, max_tokens=2048)
    if response is None:
        return _unavailable("pcb_design", "API call failed (check credits/key)")

    findings = _parse_findings(response)
    has_critical = any(f.severity == "critical" for f in findings)

    return GrokReviewResult(
        review_type="pcb_design",
        model_used=_VISION_MODEL,
        passed=not has_critical,
        summary=(
            f"{len(findings)} findings"
            f" ({sum(1 for f in findings if f.severity == 'critical')} critical)"
        ),
        findings=findings,
        raw_response=response,
    )


def review_3d_component(
    image_3d: Path,
    ref: str,
    component_type: str = "",
) -> GrokReviewResult:
    """Review a single component's 3D alignment using Grok vision.

    Args:
        image_3d: Path to per-component 3D crop PNG.
        ref: Component reference designator (e.g. "U1").
        component_type: Component description for context.

    Returns:
        GrokReviewResult with findings.
    """
    if not is_enabled():
        return _unavailable("3d_model", "GROK_REVIEW_ENABLED not set")

    if not image_3d.exists():
        return _unavailable("3d_model", f"Missing image: {image_3d}")

    prompt = (
        f"You are a PCB quality inspector. This is a 3D render of component {ref}"
        f"{f' ({component_type})' if component_type else ''} on a PCB.\n\n"
        f"Check:\n"
        f"- Is the 3D body centered on its pads?\n"
        f"- Is the body the correct size for the pad footprint?\n"
        f"- Is the component flat on the board (not floating/tilted)?\n"
        f"- Is the component the right type for the footprint "
        f"(e.g. not a capacitor body on resistor pads)?\n\n"
        f"{_JSON_FORMAT}"
    )

    messages: list[dict[str, object]] = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_encode_image(image_3d)}",
                },
            },
        ],
    }]

    response = _call_grok(messages, model=_VISION_MODEL, max_tokens=1024)
    if response is None:
        return _unavailable("3d_model", "API call failed")

    findings = _parse_findings(response)
    has_critical = any(f.severity == "critical" for f in findings)

    return GrokReviewResult(
        review_type="3d_model",
        model_used=_VISION_MODEL,
        passed=not has_critical,
        summary=f"{ref}: {len(findings)} findings",
        findings=findings,
        raw_response=response,
    )


def review_code(
    code: str,
    filepath: str = "",
    context: str = "",
) -> GrokReviewResult:
    """Review Python code using Grok for quality, bugs, and improvements.

    Args:
        code: Source code to review.
        filepath: File path for context.
        context: Additional context about what the code does.

    Returns:
        GrokReviewResult with findings.
    """
    if not is_enabled():
        return _unavailable("code", "GROK_REVIEW_ENABLED not set")

    prompt = (
        f"You are a senior Python engineer reviewing code for a KiCad PCB generation pipeline.\n"
        f"{f'File: {filepath}' if filepath else ''}\n"
        f"{f'Context: {context}' if context else ''}\n\n"
        f"Review for:\n"
        f"- Bugs and logic errors\n"
        f"- Security issues (command injection, path traversal)\n"
        f"- Type safety issues\n"
        f"- Performance concerns\n"
        f"- Missing edge cases\n\n"
        f"{_JSON_FORMAT}\n\n"
        f"```python\n{code}\n```"
    )

    messages: list[dict[str, object]] = [{"role": "user", "content": prompt}]

    response = _call_grok(messages, model=_CODE_MODEL, max_tokens=2048)
    if response is None:
        return _unavailable("code", "API call failed")

    findings = _parse_findings(response)
    has_critical = any(f.severity == "critical" for f in findings)

    return GrokReviewResult(
        review_type="code",
        model_used=_CODE_MODEL,
        passed=not has_critical,
        summary=(
            f"{len(findings)} findings"
            f" ({sum(1 for f in findings if f.severity in ('critical', 'major'))} serious)"
        ),
        findings=findings,
        raw_response=response,
    )


def review_diff(
    diff: str,
    context: str = "",
) -> GrokReviewResult:
    """Review a git diff using Grok — pre-commit code review.

    Args:
        diff: Git diff output.
        context: Description of the change.

    Returns:
        GrokReviewResult with findings.
    """
    if not is_enabled():
        return _unavailable("code", "GROK_REVIEW_ENABLED not set")

    if not diff.strip():
        return _unavailable("code", "Empty diff")

    # Truncate very large diffs
    if len(diff) > 15000:
        diff = diff[:15000] + "\n\n... (truncated, showing first 15000 chars)"

    prompt = (
        f"You are a senior engineer reviewing a git diff for a KiCad PCB generation pipeline.\n"
        f"{f'Context: {context}' if context else ''}\n\n"
        f"Review the changes for:\n"
        f"- Bugs introduced by the change\n"
        f"- Regressions (does this break existing behavior?)\n"
        f"- Missing test coverage for the change\n"
        f"- Security issues\n\n"
        f"{_JSON_FORMAT}\n\n"
        f"```diff\n{diff}\n```"
    )

    messages: list[dict[str, object]] = [{"role": "user", "content": prompt}]

    response = _call_grok(messages, model=_CODE_MODEL, max_tokens=2048)
    if response is None:
        return _unavailable("code", "API call failed")

    findings = _parse_findings(response)
    has_critical = any(f.severity == "critical" for f in findings)

    return GrokReviewResult(
        review_type="code",
        model_used=_CODE_MODEL,
        passed=not has_critical,
        summary=f"Diff review: {len(findings)} findings",
        findings=findings,
        raw_response=response,
    )
