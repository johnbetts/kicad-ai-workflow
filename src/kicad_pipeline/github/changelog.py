"""CHANGELOG.md management."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.github.git_ops import GitCommit

CHANGELOG_HEADER = """# Changelog

All notable changes to this project are documented in this file.

Format: [semantic version] YYYY-MM-DD

"""


@dataclass(frozen=True)
class ChangelogEntry:
    """A single versioned entry in the changelog."""

    version: str
    date: str
    changes: tuple[str, ...]
    breaking: tuple[str, ...] = ()


def format_entry(entry: ChangelogEntry) -> str:
    """Format *entry* as a Markdown changelog section.

    Args:
        entry: The changelog entry to format.

    Returns:
        A Markdown-formatted string for the entry.
    """
    lines: list[str] = [
        f"## [{entry.version}] - {entry.date}",
        "",
        "### Changes",
    ]
    for change in entry.changes:
        lines.append(f"- {change}")

    if entry.breaking:
        lines.append("")
        lines.append("### Breaking Changes")
        for breaking_change in entry.breaking:
            lines.append(f"- {breaking_change}")

    lines.append("")
    return "\n".join(lines)


def add_entry(changelog_path: str | Path, entry: ChangelogEntry) -> None:
    """Prepend *entry* to the changelog file.

    If the file does not exist it is created with the standard header.

    Args:
        changelog_path: Path to the ``CHANGELOG.md`` file.
        entry: The entry to prepend.
    """
    path = Path(changelog_path)
    existing = read_changelog(path)

    if not existing:
        existing = CHANGELOG_HEADER

    new_section = format_entry(entry)

    # If file already has the header, insert after it; otherwise prepend
    if existing.startswith(CHANGELOG_HEADER):
        after_header = existing[len(CHANGELOG_HEADER):]
        content = CHANGELOG_HEADER + new_section + "\n" + after_header
    else:
        content = new_section + "\n" + existing

    path.write_text(content, encoding="utf-8")


def generate_entry_from_commits(
    commits: tuple[GitCommit, ...],
    version: str,
) -> ChangelogEntry:
    """Build a :class:`ChangelogEntry` from a tuple of commits.

    Merge commits (messages starting with "Merge") are excluded.

    Args:
        commits: Recent commits to summarise.
        version: Version string for the entry.

    Returns:
        A :class:`ChangelogEntry` dated today.
    """
    changes: list[str] = []
    for c in commits:
        if c.message.lower().startswith("merge"):
            continue
        changes.append(c.message)

    today = datetime.date.today().isoformat()
    return ChangelogEntry(
        version=version,
        date=today,
        changes=tuple(changes),
    )


def read_changelog(path: str | Path) -> str:
    """Read the changelog file at *path*.

    Args:
        path: Path to the changelog file.

    Returns:
        File contents as a string, or an empty string if not found.
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
