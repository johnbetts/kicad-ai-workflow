"""Tests for kicad_pipeline.github.issues."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.github.issues import (
    GitHubIssue,
    IssueCreateResult,
    _run_gh,
    create_drc_issue,
    create_issue,
    create_validation_issues,
)

# ---------------------------------------------------------------------------
# Frozen dataclass tests
# ---------------------------------------------------------------------------


def test_github_issue_frozen() -> None:
    issue = GitHubIssue(
        number=1,
        title="Test",
        body="body",
        labels=("bug",),
        url="https://github.com/user/repo/issues/1",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        issue.number = 2  # type: ignore[misc]


def test_issue_create_result_frozen() -> None:
    result = IssueCreateResult(
        success=True,
        issue_number=42,
        url="https://github.com/user/repo/issues/42",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _run_gh
# ---------------------------------------------------------------------------


def test_run_gh_returns_tuple() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "https://github.com/user/repo/issues/1"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        rc, out, err = _run_gh(["issue", "list"])

    assert rc == 0
    assert "github.com" in out
    assert err == ""
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# create_issue
# ---------------------------------------------------------------------------


def test_create_issue_success() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "https://github.com/user/repo/issues/42\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = create_issue(
            title="Test Issue",
            body="Test body",
            labels=["bug"],
        )

    assert result.success is True
    assert result.issue_number == 42
    assert "42" in result.url


def test_create_issue_failure() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "gh: authentication required"

    with patch("subprocess.run", return_value=mock_result):
        result = create_issue(title="Test", body="body")

    assert result.success is False
    assert result.issue_number == 0
    assert result.error != ""


# ---------------------------------------------------------------------------
# create_drc_issue
# ---------------------------------------------------------------------------


def test_create_drc_issue_title_contains_rule() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "https://github.com/user/repo/issues/10\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = create_drc_issue(
            violation_message="Trace too close to copper pour",
            rule="clearance",
            severity="error",
        )

    assert result.success is True
    # Check that the title was built with the rule name
    call_args = mock_run.call_args[0][0]
    title_idx = call_args.index("--title") + 1
    assert "clearance" in call_args[title_idx]


def test_create_drc_issue_labels() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "https://github.com/user/repo/issues/10\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        create_drc_issue(
            violation_message="Drill too large",
            rule="drill_size",
            severity="warning",
        )

    call_args = mock_run.call_args[0][0]
    cmd_str = " ".join(call_args)
    assert "drc" in cmd_str
    assert "ai-generated" in cmd_str


def test_create_drc_issue_body_contains_severity() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "https://github.com/user/repo/issues/5\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        create_drc_issue(
            violation_message="Annular ring too small",
            rule="annular_ring",
            severity="error",
        )

    call_args = mock_run.call_args[0][0]
    body_idx = call_args.index("--body") + 1
    assert "error" in call_args[body_idx]


# ---------------------------------------------------------------------------
# create_validation_issues
# ---------------------------------------------------------------------------


def test_create_validation_issues_empty_report() -> None:
    results = create_validation_issues({})
    assert results == ()


def test_create_validation_issues_returns_tuple() -> None:
    report_dict: dict[str, object] = {
        "drc": {
            "violations": [
                {"rule": "clearance", "message": "too close", "severity": "error"},
            ],
        },
        "electrical": {
            "violations": [],
        },
    }

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "https://github.com/user/repo/issues/1\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        results = create_validation_issues(report_dict)

    assert isinstance(results, tuple)
    assert len(results) == 1
    assert results[0].success is True


def test_create_validation_issues_skips_warnings() -> None:
    report_dict: dict[str, object] = {
        "drc": {
            "violations": [
                {"rule": "clearance", "message": "borderline", "severity": "warning"},
            ],
        },
    }

    results = create_validation_issues(report_dict)
    assert results == ()
