"""GitHub release management via gh CLI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleaseAsset:
    """A file asset to attach to a GitHub release."""

    name: str
    path: str


@dataclass(frozen=True)
class ReleaseCreateResult:
    """Result of a GitHub release creation attempt."""

    success: bool
    tag: str
    url: str
    error: str = ""


def _run_gh_release(args: list[str]) -> tuple[int, str, str]:
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
        timeout=60,
    )
    return result.returncode, result.stdout, result.stderr


def create_release(
    tag: str,
    title: str,
    notes: str,
    assets: list[ReleaseAsset] | None = None,
    repo: str | None = None,
) -> ReleaseCreateResult:
    """Create a GitHub release via the gh CLI.

    Args:
        tag: The git tag to create the release from.
        title: Release title.
        notes: Release notes (Markdown).
        assets: Optional list of file assets to attach.
        repo: Optional ``owner/repo`` to target instead of the current repo.

    Returns:
        A :class:`ReleaseCreateResult` indicating success or failure.
    """
    cmd: list[str] = [
        "release",
        "create",
        tag,
        "--title",
        title,
        "--notes",
        notes,
    ]
    if repo:
        cmd += ["--repo", repo]
    if assets:
        for asset in assets:
            cmd.append(asset.path)

    rc, stdout, stderr = _run_gh_release(cmd)
    if rc != 0:
        return ReleaseCreateResult(
            success=False,
            tag=tag,
            url="",
            error=stderr,
        )

    url = stdout.strip()
    return ReleaseCreateResult(
        success=True,
        tag=tag,
        url=url,
    )


def create_production_release(
    tag: str,
    project_name: str,
    production_dir: str,
    repo: str | None = None,
) -> ReleaseCreateResult:
    """Create a GitHub release with production artifact assets.

    Scans *production_dir* for ``.zip``, ``.pdf``, and ``.csv`` files and
    attaches them as release assets.

    Args:
        tag: The git tag for the release.
        project_name: Human-readable project name for the release title.
        production_dir: Directory containing production artifacts.
        repo: Optional ``owner/repo`` to target.

    Returns:
        A :class:`ReleaseCreateResult`.
    """
    title = f"{project_name} {tag} - Production Release"

    # Auto-generate notes from the tag name
    notes = (
        f"## Production Release {tag}\n\n"
        f"Automated production release for **{project_name}**.\n\n"
        "### Included Artifacts\n"
        "- Gerber files (.zip)\n"
        "- Assembly drawings (.pdf)\n"
        "- Bill of materials (.csv)\n"
    )

    prod_path = Path(production_dir)
    assets: list[ReleaseAsset] = []
    for pattern in ("*.zip", "*.pdf", "*.csv"):
        for file_path in sorted(prod_path.rglob(pattern)):
            assets.append(
                ReleaseAsset(name=file_path.name, path=str(file_path))
            )

    return create_release(
        tag=tag,
        title=title,
        notes=notes,
        assets=assets if assets else None,
        repo=repo,
    )
