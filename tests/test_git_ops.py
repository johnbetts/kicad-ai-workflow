"""Tests for kicad_pipeline.github.git_ops."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.github.git_ops import (
    GitCommit,
    GitStatus,
    _run_git,
    commit,
    create_tag,
    generate_semantic_commit_message,
    get_recent_commits,
    get_status,
    stage_files,
)

# ---------------------------------------------------------------------------
# Frozen dataclass tests
# ---------------------------------------------------------------------------


def test_git_status_frozen() -> None:
    status = GitStatus(
        branch="main",
        staged=(),
        unstaged=(),
        untracked=(),
        is_clean=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        status.branch = "other"  # type: ignore[misc]


def test_git_commit_frozen() -> None:
    c = GitCommit(
        hash="abc1234",
        message="feat: test",
        author="Author",
        date="2024-01-01",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.hash = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# generate_semantic_commit_message
# ---------------------------------------------------------------------------


def test_generate_semantic_commit_requirements() -> None:
    msg = generate_semantic_commit_message("requirements", "add capacitor specs")
    assert msg == "feat(requirements): add capacitor specs"


def test_generate_semantic_commit_unknown_phase() -> None:
    msg = generate_semantic_commit_message("unknown_phase", "some change")
    assert msg == "feat: some change"


def test_generate_semantic_commit_valid_phases() -> None:
    for phase in ("schematic", "pcb", "routing", "validation", "production", "github"):
        msg = generate_semantic_commit_message(phase, "desc")
        assert f"feat({phase}):" in msg


# ---------------------------------------------------------------------------
# _run_git
# ---------------------------------------------------------------------------


def test_run_git_returns_tuple() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "stdout text"
    mock_result.stderr = "stderr text"

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        rc, out, err = _run_git(["status"])

    assert rc == 0
    assert out == "stdout text"
    assert err == "stderr text"
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


_STATUS_OUTPUT = """\
## main...origin/main
M  staged_file.py
 M unstaged_file.py
?? untracked_file.py
"""


def test_get_status_success() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _STATUS_OUTPUT
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        status = get_status("/repo")

    assert status.branch == "main"
    assert "staged_file.py" in status.staged
    assert "unstaged_file.py" in status.unstaged
    assert "untracked_file.py" in status.untracked
    assert status.is_clean is False


def test_get_status_failure() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "fatal: not a git repo"

    with patch("subprocess.run", return_value=mock_result):
        status = get_status("/nonexistent")

    assert status.is_clean is False
    assert status.branch == ""
    assert status.staged == ()


# ---------------------------------------------------------------------------
# stage_files
# ---------------------------------------------------------------------------


def test_stage_files_success() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = stage_files(["file.py"], repo_path="/repo")

    assert result is True


def test_stage_files_failure() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "error"

    with patch("subprocess.run", return_value=mock_result):
        result = stage_files(["missing.py"], repo_path="/repo")

    assert result is False


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------


def test_commit_success() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "[main abc1234] feat: test"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = commit("feat: test", repo_path="/repo")

    assert result is True


def test_commit_failure() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "nothing to commit"

    with patch("subprocess.run", return_value=mock_result):
        result = commit("feat: nothing", repo_path="/repo")

    assert result is False


# ---------------------------------------------------------------------------
# create_tag
# ---------------------------------------------------------------------------


def test_create_tag_success() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = create_tag("v1.0.0", message="Release 1.0.0", repo_path="/repo")

    assert result is True


# ---------------------------------------------------------------------------
# get_recent_commits
# ---------------------------------------------------------------------------


_LOG_LINE_1 = "abc1234|feat(pcb): add zones|Author Name|2024-01-01T12:00:00+00:00"
_LOG_LINE_2 = "def5678|fix(routing): correct trace width|Jane Doe|2024-01-02T09:00:00+00:00"
_LOG_OUTPUT = f"{_LOG_LINE_1}\n{_LOG_LINE_2}"


def test_get_recent_commits_parses() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _LOG_OUTPUT
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        commits = get_recent_commits(n=2, repo_path="/repo")

    assert len(commits) == 2
    assert commits[0].hash == "abc1234"
    assert commits[0].message == "feat(pcb): add zones"
    assert commits[0].author == "Author Name"
    assert commits[1].hash == "def5678"


def test_get_recent_commits_empty_on_failure() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "fatal: not a git repo"

    with patch("subprocess.run", return_value=mock_result):
        commits = get_recent_commits(repo_path="/nonexistent")

    assert commits == ()
