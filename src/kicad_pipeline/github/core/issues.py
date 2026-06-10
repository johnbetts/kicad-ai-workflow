"""GitHub issue management via gh CLI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GitHubIssue:
    """Represents a GitHub issue."""

    number: int
    title: str
    body: str
    labels: tuple[str, ...]
    url: str = ""


@dataclass(frozen=True)
class IssueCreateResult:
    """Result of a GitHub issue creation attempt."""

    success: bool
    issue_number: int
    url: str
    error: str = ""


def _run_gh(args: list[str]) -> tuple[int, str, str]:
    """Run a gh subprocess command.

    Args:
        args: Arguments to pass after ``gh``.

    Returns:
        A tuple of (returncode, stdout, stderr).
    """
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


def _parse_issue_number_from_url(url: str) -> int:
    """Extract an issue number from a GitHub issue URL.

    Args:
        url: A URL like ``https://github.com/user/repo/issues/42``.

    Returns:
        The issue number, or 0 if not parseable.
    """
    url = url.strip()
    parts = url.rstrip("/").split("/")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def create_issue(
    title: str,
    body: str,
    labels: list[str] | None = None,
    repo: str | None = None,
) -> IssueCreateResult:
    """Create a GitHub issue via the gh CLI.

    Args:
        title: Issue title.
        body: Issue body (Markdown).
        labels: Optional list of label names to apply.
        repo: Optional ``owner/repo`` to target instead of the current repo.

    Returns:
        An :class:`IssueCreateResult` indicating success or failure.
    """
    cmd: list[str] = [
        "issue",
        "create",
        "--title",
        title,
        "--body",
        body,
    ]
    if labels:
        for label in labels:
            cmd += ["--label", label]
    if repo:
        cmd += ["--repo", repo]

    rc, stdout, stderr = _run_gh(cmd)
    if rc != 0:
        return IssueCreateResult(
            success=False,
            issue_number=0,
            url="",
            error=stderr,
        )

    url = stdout.strip()
    number = _parse_issue_number_from_url(url)
    return IssueCreateResult(
        success=True,
        issue_number=number,
        url=url,
    )


def create_drc_issue(
    violation_message: str,
    rule: str,
    severity: str,
    repo: str | None = None,
) -> IssueCreateResult:
    """Create a GitHub issue for a DRC violation.

    Args:
        violation_message: Human-readable description of the violation.
        rule: DRC rule name that was violated.
        severity: Severity level string (e.g. ``"error"``, ``"warning"``).
        repo: Optional ``owner/repo`` to target.

    Returns:
        An :class:`IssueCreateResult`.
    """
    title = f"DRC Violation: {rule}"
    body = (
        f"## DRC Violation\n\n"
        f"**Severity**: {severity}\n\n"
        f"**Rule**: {rule}\n\n"
        f"**Message**: {violation_message}\n"
    )
    labels: list[str] = ["drc", "ai-generated", f"severity-{severity}"]
    return create_issue(title=title, body=body, labels=labels, repo=repo)


def create_validation_issues(
    report_dict: dict[str, object],
    repo: str | None = None,
) -> tuple[IssueCreateResult, ...]:
    """Create GitHub issues for all error-severity violations in *report_dict*.

    Args:
        report_dict: A dict as produced by
            :func:`kicad_pipeline.validation.report.report_to_dict`.
        repo: Optional ``owner/repo`` to target.

    Returns:
        A tuple of :class:`IssueCreateResult` objects, one per issue created.
    """
    section_keys = (
        "drc",
        "electrical",
        "manufacturing",
        "signal_integrity",
        "thermal",
    )
    results: list[IssueCreateResult] = []

    for key in section_keys:
        section = report_dict.get(key)
        if not isinstance(section, dict):
            continue
        violations = section.get("violations", [])
        if not isinstance(violations, list):
            continue
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            severity = str(violation.get("severity", "")).lower()
            if severity != "error":
                continue
            rule = str(violation.get("rule", "unknown"))
            message = str(violation.get("message", ""))
            result = create_drc_issue(
                violation_message=message,
                rule=rule,
                severity=severity,
                repo=repo,
            )
            results.append(result)

    return tuple(results)
