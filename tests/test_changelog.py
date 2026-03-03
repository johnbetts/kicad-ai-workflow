"""Tests for kicad_pipeline.github.changelog."""

from __future__ import annotations

import dataclasses
import datetime
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.github.changelog import (
    ChangelogEntry,
    add_entry,
    format_entry,
    generate_entry_from_commits,
    read_changelog,
)
from kicad_pipeline.github.git_ops import GitCommit

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Frozen dataclass tests
# ---------------------------------------------------------------------------


def test_changelog_entry_frozen() -> None:
    entry = ChangelogEntry(
        version="1.0.0",
        date="2024-01-01",
        changes=("Initial release",),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.version = "2.0.0"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# format_entry
# ---------------------------------------------------------------------------


def test_format_entry_version_in_output() -> None:
    entry = ChangelogEntry(
        version="1.0.0",
        date="2024-01-01",
        changes=("Add feature X",),
    )
    output = format_entry(entry)
    assert "1.0.0" in output


def test_format_entry_changes_listed() -> None:
    entry = ChangelogEntry(
        version="1.2.3",
        date="2024-06-15",
        changes=("Fix bug A", "Add feature B"),
    )
    output = format_entry(entry)
    assert "Fix bug A" in output
    assert "Add feature B" in output


def test_format_entry_breaking_changes_section() -> None:
    entry = ChangelogEntry(
        version="2.0.0",
        date="2024-01-01",
        changes=("Refactor API",),
        breaking=("Remove deprecated `foo` function",),
    )
    output = format_entry(entry)
    assert "Breaking Changes" in output
    assert "Remove deprecated" in output


def test_format_entry_no_breaking_section() -> None:
    entry = ChangelogEntry(
        version="1.0.0",
        date="2024-01-01",
        changes=("Normal change",),
        breaking=(),
    )
    output = format_entry(entry)
    assert "Breaking Changes" not in output


# ---------------------------------------------------------------------------
# add_entry
# ---------------------------------------------------------------------------


def test_add_entry_creates_file(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    entry = ChangelogEntry(
        version="1.0.0",
        date="2024-01-01",
        changes=("Initial release",),
    )
    add_entry(changelog, entry)
    assert changelog.exists()


def test_add_entry_content_correct(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    entry = ChangelogEntry(
        version="1.0.0",
        date="2024-01-01",
        changes=("Initial release", "Add routing"),
    )
    add_entry(changelog, entry)
    content = changelog.read_text(encoding="utf-8")
    assert "1.0.0" in content
    assert "Initial release" in content
    assert "Add routing" in content


def test_add_entry_prepends_to_existing(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    # Write an existing entry
    old_entry = ChangelogEntry(
        version="0.9.0",
        date="2023-12-01",
        changes=("Old feature",),
    )
    add_entry(changelog, old_entry)

    new_entry = ChangelogEntry(
        version="1.0.0",
        date="2024-01-01",
        changes=("New feature",),
    )
    add_entry(changelog, new_entry)

    content = changelog.read_text(encoding="utf-8")
    pos_new = content.index("1.0.0")
    pos_old = content.index("0.9.0")
    assert pos_new < pos_old


# ---------------------------------------------------------------------------
# read_changelog
# ---------------------------------------------------------------------------


def test_read_changelog_missing_returns_empty(tmp_path: Path) -> None:
    result = read_changelog(tmp_path / "NONEXISTENT.md")
    assert result == ""


def test_read_changelog_returns_content(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\nSome content\n", encoding="utf-8")
    content = read_changelog(changelog)
    assert "Changelog" in content
    assert "Some content" in content


# ---------------------------------------------------------------------------
# generate_entry_from_commits
# ---------------------------------------------------------------------------


def test_generate_entry_from_commits() -> None:
    commits = (
        GitCommit(
            hash="abc1234",
            message="feat(pcb): add zones",
            author="Author",
            date="2024-01-01",
        ),
        GitCommit(
            hash="def5678",
            message="fix(routing): correct trace",
            author="Author",
            date="2024-01-02",
        ),
    )
    entry = generate_entry_from_commits(commits, version="1.0.0")
    assert entry.version == "1.0.0"
    assert len(entry.changes) == 2
    assert "feat(pcb): add zones" in entry.changes
    assert "fix(routing): correct trace" in entry.changes


def test_generate_entry_filters_merges() -> None:
    commits = (
        GitCommit(
            hash="abc1234",
            message="Merge branch 'feature' into main",
            author="Author",
            date="2024-01-01",
        ),
        GitCommit(
            hash="def5678",
            message="feat: real change",
            author="Author",
            date="2024-01-02",
        ),
    )
    entry = generate_entry_from_commits(commits, version="1.1.0")
    assert len(entry.changes) == 1
    assert "feat: real change" in entry.changes
    # The merge commit should be excluded
    for change in entry.changes:
        assert "Merge" not in change


def test_generate_entry_date_is_today() -> None:
    commits = (
        GitCommit(
            hash="abc",
            message="feat: something",
            author="A",
            date="2024-01-01",
        ),
    )
    entry = generate_entry_from_commits(commits, version="1.0.0")
    today = datetime.date.today().isoformat()
    assert entry.date == today
