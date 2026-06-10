"""Suggestion reporting system for multi-agent pipeline improvements.

Agents log improvement suggestions to a JSONL file. Each line is a
self-contained JSON object. No file locking needed -- agents just append.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

_DEFAULT_SUGGESTIONS_PATH = Path.home() / ".claude" / "kicad-agents" / "suggestions.jsonl"


@dataclass(frozen=True)
class Suggestion:
    """A pipeline improvement suggestion from an agent."""

    suggestion_id: str
    agent_id: str
    title: str
    description: str
    category: str  # performance, quality, ux, architecture, testing, documentation
    priority: str  # critical, high, medium, low
    effort: str  # trivial, small, medium, large
    affected_module: str
    evidence: str
    status: str  # proposed, accepted, implemented, verified
    created_at: str
    updated_at: str


class SuggestionReporter:
    """Agent-side interface for logging improvement suggestions.

    Usage::

        reporter = SuggestionReporter("optimizer-agent")
        sid = reporter.suggest(
            title="Move C3 closer to U1 VCC pin",
            description="Decoupling cap C3 is 8.2mm from U1 (max: 5mm)",
            category="quality",
            priority="high",
        )
    """

    def __init__(
        self,
        agent_id: str,
        suggestions_path: Path | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._path = suggestions_path or _DEFAULT_SUGGESTIONS_PATH

    def suggest(
        self,
        title: str,
        description: str,
        category: Literal[
            "performance", "quality", "ux", "architecture", "testing", "documentation"
        ],
        priority: Literal["critical", "high", "medium", "low"] = "medium",
        effort: Literal["trivial", "small", "medium", "large"] = "small",
        affected_module: str = "",
        evidence: str = "",
    ) -> str:
        """Log a suggestion and return its ID."""
        now = datetime.now(tz=timezone.utc).isoformat()
        suggestion = Suggestion(
            suggestion_id=f"sug-{uuid.uuid4().hex[:12]}",
            agent_id=self._agent_id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            effort=effort,
            affected_module=affected_module,
            evidence=evidence,
            status="proposed",
            created_at=now,
            updated_at=now,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(suggestion)) + "\n")
        return suggestion.suggestion_id

    def load_all(self) -> tuple[Suggestion, ...]:
        """Load all suggestions from the JSONL file."""
        if not self._path.exists():
            return ()
        suggestions: list[Suggestion] = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                data = json.loads(line)
                suggestions.append(Suggestion(**data))
        return tuple(suggestions)

    def update_status(
        self,
        suggestion_id: str,
        new_status: Literal["proposed", "accepted", "implemented", "verified"],
    ) -> bool:
        """Update a suggestion's status. Returns True if found and updated."""
        if not self._path.exists():
            return False
        lines = self._path.read_text(encoding="utf-8").strip().splitlines()
        updated = False
        new_lines: list[str] = []
        now = datetime.now(tz=timezone.utc).isoformat()
        for line in lines:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("suggestion_id") == suggestion_id:
                data["status"] = new_status
                data["updated_at"] = now
                updated = True
            new_lines.append(json.dumps(data))
        if updated:
            self._path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return updated

    def filter_by_status(self, status: str) -> tuple[Suggestion, ...]:
        """Return suggestions matching the given status."""
        return tuple(s for s in self.load_all() if s.status == status)

    def filter_by_category(self, category: str) -> tuple[Suggestion, ...]:
        """Return suggestions matching the given category."""
        return tuple(s for s in self.load_all() if s.category == category)
