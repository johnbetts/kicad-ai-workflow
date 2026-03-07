"""Git operations via subprocess (git CLI)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

_VALID_PHASES = {
    "requirements",
    "schematic",
    "pcb",
    "routing",
    "validation",
    "production",
    "github",
}


@dataclass(frozen=True)
class GitStatus:
    """Result of git status --porcelain -b."""

    branch: str
    staged: tuple[str, ...]
    unstaged: tuple[str, ...]
    untracked: tuple[str, ...]
    is_clean: bool


@dataclass(frozen=True)
class GitCommit:
    """A single git commit record."""

    hash: str
    message: str
    author: str
    date: str


def _run_git(
    args: list[str], cwd: str | None = None
) -> tuple[int, str, str]:
    """Run a git subprocess command.

    Args:
        args: Arguments to pass after ``git``.
        cwd: Working directory for the subprocess.

    Returns:
        A tuple of (returncode, stdout, stderr).
    """
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd,
    )
    return result.returncode, result.stdout, result.stderr


def get_status(repo_path: str = ".") -> GitStatus:
    """Return the current git status for *repo_path*.

    Args:
        repo_path: Path to the git repository root.

    Returns:
        A :class:`GitStatus` populated from ``git status --porcelain -b``.
        On failure, returns a :class:`GitStatus` with empty collections and
        ``is_clean=False``.
    """
    rc, stdout, _stderr = _run_git(
        ["status", "--porcelain", "-b"], cwd=repo_path
    )
    if rc != 0:
        return GitStatus(
            branch="",
            staged=(),
            unstaged=(),
            untracked=(),
            is_clean=False,
        )

    branch = ""
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []

    for line in stdout.splitlines():
        if line.startswith("## "):
            # e.g. "## main...origin/main"
            branch_part = line[3:]
            branch = branch_part.split("...")[0].split(" ")[0]
            continue
        if len(line) < 2:
            continue
        xy = line[:2]
        path = line[3:]
        x, y = xy[0], xy[1]
        if xy == "??":
            untracked.append(path)
        else:
            if x != " " and x != "?":
                staged.append(path)
            if y != " " and y != "?":
                unstaged.append(path)

    is_clean = not staged and not unstaged and not untracked
    return GitStatus(
        branch=branch,
        staged=tuple(staged),
        unstaged=tuple(unstaged),
        untracked=tuple(untracked),
        is_clean=is_clean,
    )


def stage_files(files: list[str], repo_path: str = ".") -> bool:
    """Stage *files* with ``git add``.

    Args:
        files: List of file paths to stage.
        repo_path: Path to the git repository root.

    Returns:
        ``True`` if the command succeeded, ``False`` otherwise.
    """
    rc, _out, _err = _run_git(["add", *files], cwd=repo_path)
    return rc == 0


def commit(message: str, repo_path: str = ".") -> bool:
    """Create a commit with *message*.

    Args:
        message: Commit message.
        repo_path: Path to the git repository root.

    Returns:
        ``True`` if the commit succeeded, ``False`` otherwise.
    """
    rc, _out, _err = _run_git(["commit", "-m", message], cwd=repo_path)
    return rc == 0


def create_tag(
    tag: str, message: str = "", repo_path: str = "."
) -> bool:
    """Create an annotated tag.

    Args:
        tag: Tag name.
        message: Tag annotation message.
        repo_path: Path to the git repository root.

    Returns:
        ``True`` if the tag was created, ``False`` otherwise.
    """
    rc, _out, _err = _run_git(
        ["tag", "-a", tag, "-m", message], cwd=repo_path
    )
    return rc == 0


def get_recent_commits(
    n: int = 10, repo_path: str = "."
) -> tuple[GitCommit, ...]:
    """Return the *n* most recent commits.

    Args:
        n: Number of commits to retrieve.
        repo_path: Path to the git repository root.

    Returns:
        A tuple of :class:`GitCommit` objects, or an empty tuple on failure.
    """
    rc, stdout, _stderr = _run_git(
        ["log", "--oneline", f"-{n}", "--pretty=format:%H|%s|%an|%ai"],
        cwd=repo_path,
    )
    if rc != 0 or not stdout.strip():
        return ()

    commits: list[GitCommit] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        commits.append(
            GitCommit(
                hash=parts[0],
                message=parts[1],
                author=parts[2],
                date=parts[3],
            )
        )
    return tuple(commits)


def push(repo_path: str = ".", remote: str = "origin", branch: str = "") -> bool:
    """Push current branch to remote.

    Args:
        repo_path: Path to the git repository root.
        remote: Remote name.
        branch: Branch name (empty = current branch).

    Returns:
        ``True`` if the push succeeded, ``False`` otherwise.
    """
    args = ["push", remote]
    if branch:
        args.append(branch)
    rc, _out, _err = _run_git(args, cwd=repo_path)
    return rc == 0


def init_from_github(url: str, local_path: str) -> bool:
    """Clone a GitHub repository to a local path.

    Args:
        url: GitHub repository URL.
        local_path: Local directory to clone into.

    Returns:
        ``True`` if the clone succeeded, ``False`` otherwise.
    """
    result = subprocess.run(
        ["git", "clone", url, local_path],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode == 0


def checkpoint(repo_path: str, message: str) -> bool:
    """Stage all KiCad files, commit, and push.

    Args:
        repo_path: Path to the git repository root.
        message: Commit message.

    Returns:
        ``True`` if the commit and push succeeded.
    """
    # Stage KiCad project files.
    kicad_patterns = [
        "*.kicad_sch", "*.kicad_pcb", "*.kicad_pro",
        "*.kicad_sym", "*.kicad_mod",
        "output/",
    ]
    _run_git(["add", *kicad_patterns], cwd=repo_path)

    # Also stage any production output.
    _run_git(["add", "output/"], cwd=repo_path)

    # Check if there's anything to commit.
    status = get_status(repo_path)
    if status.is_clean:
        return True

    if not commit(message, repo_path):
        return False

    return push(repo_path)


def create_design_review_pr(
    repo_path: str,
    title: str,
    body: str,
    base: str = "main",
) -> str:
    """Create a GitHub pull request for design review.

    Args:
        repo_path: Path to the git repository root.
        title: PR title.
        body: PR body (markdown).
        base: Base branch for the PR.

    Returns:
        PR URL string, or empty string on failure.
    """
    result = subprocess.run(
        [
            "gh", "pr", "create",
            "--title", title,
            "--body", body,
            "--base", base,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=repo_path,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return ""


def generate_semantic_commit_message(phase: str, description: str) -> str:
    """Generate a conventional commit message.

    Args:
        phase: Pipeline phase name (e.g. "pcb", "schematic").
        description: Short description of the change.

    Returns:
        A conventional commit message string.
    """
    if phase in _VALID_PHASES:
        return f"feat({phase}): {description}"
    return f"feat: {description}"
