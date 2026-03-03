"""Git conventions for orchestrated projects.

Thin wrappers around :mod:`kicad_pipeline.github.git_ops` that apply the
project's naming conventions for commits and tags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.github import git_ops

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.orchestrator.models import StageId


def commit_stage(
    project_root: Path,
    variant_name: str,
    stage: StageId,
    description: str,
) -> bool:
    """Create a conventional commit for a stage action.

    Stages all files under ``variants/{variant_name}/`` and creates a commit
    with the message ``feat({stage.value}/{variant_name}): {description}``.

    Args:
        project_root: Root of the git repository.
        variant_name: Variant being modified.
        stage: Pipeline stage that produced the change.
        description: Short imperative description of the change.

    Returns:
        ``True`` if the commit succeeded, ``False`` otherwise.
    """
    repo = str(project_root)
    variant_path = f"variants/{variant_name}/"
    git_ops.stage_files([variant_path], repo_path=repo)
    message = f"feat({stage.value}/{variant_name}): {description}"
    return git_ops.commit(message, repo_path=repo)


def tag_revision(
    project_root: Path,
    variant_name: str,
    revision: int,
) -> bool:
    """Create a git tag for a production revision.

    Tag format: ``{variant_name}/rev{revision}``.

    Args:
        project_root: Root of the git repository.
        variant_name: Variant name.
        revision: Revision number.

    Returns:
        ``True`` if the tag was created, ``False`` otherwise.
    """
    tag = f"{variant_name}/rev{revision}"
    return git_ops.create_tag(
        tag,
        message=f"Production revision {revision} for {variant_name}",
        repo_path=str(project_root),
    )


def tag_release(
    project_root: Path,
    variant_name: str,
    version: str,
) -> bool:
    """Create a git tag for a variant release.

    Tag format: ``{variant_name}/{version}``.

    Args:
        project_root: Root of the git repository.
        variant_name: Variant name.
        version: Semantic version string (e.g. "v1.0.0").

    Returns:
        ``True`` if the tag was created, ``False`` otherwise.
    """
    tag = f"{variant_name}/{version}"
    return git_ops.create_tag(
        tag,
        message=f"Release {version} for {variant_name}",
        repo_path=str(project_root),
    )


def commit_revision(
    project_root: Path,
    variant_name: str,
    revision: int,
) -> bool:
    """Create a commit for a production revision.

    Message format: ``feat(production/{variant_name}): revision {revision}``.

    Args:
        project_root: Root of the git repository.
        variant_name: Variant name.
        revision: Revision number.

    Returns:
        ``True`` if the commit succeeded, ``False`` otherwise.
    """
    repo = str(project_root)
    variant_path = f"variants/{variant_name}/"
    git_ops.stage_files([variant_path], repo_path=repo)
    message = f"feat(production/{variant_name}): revision {revision}"
    return git_ops.commit(message, repo_path=repo)
