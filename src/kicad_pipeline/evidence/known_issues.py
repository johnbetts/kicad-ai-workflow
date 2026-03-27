"""Known issues injector — loads institutional knowledge for agent prompts.

Reads from three sources and formats a context block to prepend to agent
prompts so agents don't repeat known mistakes.

Sources:
1. docs/known_issues.md — project-wide issue registry
2. .pcb-review/lessons.md — board-specific lessons learned
3. .pcb-review/failed_approaches.jsonl — approaches that didn't work
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 2000


def _read_file(path: Path) -> str:
    """Read a file, return empty string if missing."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _filter_known_issues(content: str, stage: str | None) -> str:
    """Filter known_issues.md to OPEN issues relevant to the stage."""
    if not content:
        return ""

    lines = content.splitlines()
    header_lines: list[str] = []
    relevant_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            header_lines.append(line)
            continue
        if stripped.startswith("|") and "OPEN" in stripped:
            relevant_lines.append(stripped)

    if not relevant_lines:
        return ""

    return "\n".join(relevant_lines)


def _filter_failed_approaches(content: str, board: str | None) -> str:
    """Filter failed_approaches.jsonl to board-relevant entries."""
    if not content:
        return ""

    entries: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if board and data.get("board", "") and board not in data["board"]:
                continue
            desc = data.get("description", str(data))
            reason = data.get("reason", "")
            entry = f"- {desc}"
            if reason:
                entry += f" (reason: {reason})"
            entries.append(entry)
        except json.JSONDecodeError:
            continue

    return "\n".join(entries[-10:])


def load_known_issues(
    project_root: Path | None = None,
    board_dir: Path | None = None,
    stage: str | None = None,
    board: str | None = None,
) -> str:
    """Load and filter known issues for injection into agent prompts.

    Args:
        project_root: Project root directory (defaults to cwd).
        board_dir: Board-specific output directory for per-board lessons.
        stage: Pipeline stage to filter for (e.g. "pcb", "schematic").
        board: Board name to filter failed approaches.

    Returns:
        Formatted markdown string to prepend to agent prompts.
        Capped at MAX_CONTEXT_CHARS to avoid prompt bloat.
    """
    if project_root is None:
        project_root = Path.cwd()

    sections: list[str] = []

    # Source 1: docs/known_issues.md
    known_issues_path = project_root / "docs" / "known_issues.md"
    known_content = _read_file(known_issues_path)
    filtered = _filter_known_issues(known_content, stage)
    if filtered:
        sections.append(
            "## KNOWN ISSUES — DO NOT REPEAT THESE MISTAKES\n" + filtered
        )

    # Source 2: .pcb-review/lessons.md (project-level)
    project_lessons = _read_file(project_root / ".pcb-review" / "lessons.md")
    if project_lessons:
        sections.append("## LESSONS LEARNED (project)\n" + project_lessons)

    # Source 2b: board-specific lessons
    if board_dir:
        board_lessons = _read_file(board_dir / ".pcb-review" / "lessons.md")
        if board_lessons:
            sections.append(
                "## LESSONS LEARNED (this board)\n" + board_lessons
            )

    # Source 3: .pcb-review/failed_approaches.jsonl
    failed_path = project_root / ".pcb-review" / "failed_approaches.jsonl"
    failed_content = _read_file(failed_path)
    failed_filtered = _filter_failed_approaches(failed_content, board)
    if failed_filtered:
        sections.append(
            "## FAILED APPROACHES (do NOT try these again)\n" + failed_filtered
        )

    result = "\n\n".join(sections)

    if len(result) > MAX_CONTEXT_CHARS:
        result = result[:MAX_CONTEXT_CHARS] + "\n... (truncated)"

    return result
