"""Verification checklist — loads known bugs and patterns for agent injection.

Reads ``data/verification_checklist.json`` and provides filtered views
so each AI agent gets ONLY the bugs/patterns relevant to its batch of
components and check types.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CHECKLIST_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "verification_checklist.json"
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownBug:
    """A known bug that verification should check for."""

    id: str
    description: str
    check_instruction: str
    severity: str
    affects_checks: tuple[str, ...]  # glob patterns like "FAB-CLEARANCE-*"
    affects_components: tuple[str, ...]  # glob patterns like "J*", "R1"
    verification_type: str  # "programmatic", "ai_visual", "cross_reference"
    status: str = "open"


@dataclass(frozen=True)
class Pattern:
    """A known pattern that agents should be aware of."""

    id: str
    description: str
    check_instruction: str
    severity: str
    affects_checks: tuple[str, ...]
    affects_components: tuple[str, ...]
    verification_type: str  # "ai_context", "cross_reference"


# ---------------------------------------------------------------------------
# Checklist loader
# ---------------------------------------------------------------------------


class VerificationChecklist:
    """Loads and filters verification_checklist.json for agent injection."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_CHECKLIST_PATH
        self._bugs: list[KnownBug] = []
        self._patterns: list[Pattern] = []
        self._load()

    def _load(self) -> None:
        """Load and parse the checklist JSON."""
        if not self._path.exists():
            logger.warning("Verification checklist not found: %s", self._path)
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load verification checklist: %s", exc)
            return

        for bug_data in data.get("known_bugs", []):
            self._bugs.append(
                KnownBug(
                    id=bug_data["id"],
                    description=bug_data["description"],
                    check_instruction=bug_data.get("check_instruction", ""),
                    severity=bug_data.get("severity", "major"),
                    affects_checks=tuple(bug_data.get("affects_checks", ())),
                    affects_components=tuple(bug_data.get("affects_components", ())),
                    verification_type=bug_data.get("verification_type", "ai_visual"),
                    status=bug_data.get("status", "open"),
                )
            )

        for pat_data in data.get("patterns", []):
            self._patterns.append(
                Pattern(
                    id=pat_data["id"],
                    description=pat_data["description"],
                    check_instruction=pat_data.get("check_instruction", ""),
                    severity=pat_data.get("severity", "info"),
                    affects_checks=tuple(pat_data.get("affects_checks", ())),
                    affects_components=tuple(pat_data.get("affects_components", ())),
                    verification_type=pat_data.get("verification_type", "ai_context"),
                )
            )

    @property
    def bugs(self) -> list[KnownBug]:
        """All known bugs (including closed for history)."""
        return list(self._bugs)

    @property
    def open_bugs(self) -> list[KnownBug]:
        """Only open bugs."""
        return [b for b in self._bugs if b.status == "open"]

    @property
    def patterns(self) -> list[Pattern]:
        """All known patterns."""
        return list(self._patterns)

    def bugs_for_component(self, ref: str) -> list[KnownBug]:
        """Return open bugs that affect a specific component ref."""
        result: list[KnownBug] = []
        for bug in self.open_bugs:
            if not bug.affects_components:
                result.append(bug)
                continue
            for pattern in bug.affects_components:
                if fnmatch.fnmatch(ref, pattern):
                    result.append(bug)
                    break
        return result

    def bugs_for_check(self, check_id: str) -> list[KnownBug]:
        """Return open bugs that affect a specific check ID."""
        result: list[KnownBug] = []
        for bug in self.open_bugs:
            if not bug.affects_checks:
                result.append(bug)
                continue
            for pattern in bug.affects_checks:
                if fnmatch.fnmatch(check_id, pattern):
                    result.append(bug)
                    break
        return result

    def patterns_for_component(self, ref: str) -> list[Pattern]:
        """Return patterns relevant to a specific component ref."""
        result: list[Pattern] = []
        for pat in self._patterns:
            if not pat.affects_components:
                result.append(pat)
                continue
            for pattern in pat.affects_components:
                if fnmatch.fnmatch(ref, pattern):
                    result.append(pat)
                    break
        return result

    def context_for_agent(
        self,
        persona: str,
        refs: list[str] | None = None,
        check_ids: list[str] | None = None,
    ) -> str:
        """Format relevant bugs/patterns as a prompt block for an AI agent.

        Args:
            persona: "fab" or "ee" — filters by verification_type relevance.
            refs: Component refs to filter for (None = all).
            check_ids: Check IDs to filter for (None = all).

        Returns:
            Markdown-formatted string for injection into agent prompt.
        """
        sections: list[str] = []

        # Collect relevant bugs
        relevant_bugs: list[KnownBug] = []
        for bug in self.open_bugs:
            if refs:
                if any(
                    fnmatch.fnmatch(ref, pat)
                    for ref in refs
                    for pat in bug.affects_components
                ) or not bug.affects_components:
                    relevant_bugs.append(bug)
            elif check_ids:
                if any(
                    fnmatch.fnmatch(cid, pat)
                    for cid in check_ids
                    for pat in bug.affects_checks
                ) or not bug.affects_checks:
                    relevant_bugs.append(bug)
            else:
                relevant_bugs.append(bug)

        if relevant_bugs:
            lines = ["## KNOWN BUGS — check these explicitly"]
            for bug in relevant_bugs:
                lines.append(
                    f"- **[{bug.id}]** ({bug.severity}): {bug.description}"
                )
                if bug.check_instruction:
                    lines.append(f"  CHECK: {bug.check_instruction}")
            sections.append("\n".join(lines))

        # Collect relevant patterns
        relevant_patterns: list[Pattern] = []
        for pat in self._patterns:
            if refs:
                if any(
                    fnmatch.fnmatch(ref, cpat)
                    for ref in refs
                    for cpat in pat.affects_components
                ) or not pat.affects_components:
                    relevant_patterns.append(pat)
            else:
                relevant_patterns.append(pat)

        if relevant_patterns:
            lines = ["## KNOWN PATTERNS — apply these rules"]
            for pat in relevant_patterns:
                lines.append(f"- **[{pat.id}]**: {pat.description}")
                if pat.check_instruction:
                    lines.append(f"  RULE: {pat.check_instruction}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)
