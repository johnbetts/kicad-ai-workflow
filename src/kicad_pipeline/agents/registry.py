"""Persistence layer for the global agent registry and pipeline version.

Follows the same ``_to_dict`` / ``_from_dict`` pattern as
:mod:`kicad_pipeline.orchestrator.manifest`.  All writes use atomic
tmp-file + ``os.replace`` to avoid partial reads.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from kicad_pipeline.agents.models import (
    AgentRegistration,
    AgentRegistry,
    AgentState,
    PipelineVersion,
)
from kicad_pipeline.constants import (
    AGENT_PIPELINE_VERSION_FILENAME,
    AGENT_REGISTRY_DIR,
    AGENT_REGISTRY_FILENAME,
)
from kicad_pipeline.exceptions import AgentError

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, data: dict[str, object]) -> None:
    """Write *data* as JSON to *path* atomically via tmp + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, str(path))
    except BaseException:
        with _SuppressUnlink():
            os.unlink(tmp_path)
        raise


class _SuppressUnlink:
    """Context manager that silences ``OSError`` during cleanup."""

    def __enter__(self) -> None:
        pass

    def __exit__(self, *_args: object) -> bool:
        return True


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _pipeline_version_to_dict(pv: PipelineVersion) -> dict[str, object]:
    return {"git_hash": pv.git_hash, "git_tag": pv.git_tag, "timestamp": pv.timestamp}


def _pipeline_version_from_dict(data: dict[str, object]) -> PipelineVersion:
    return PipelineVersion(
        git_hash=str(data.get("git_hash", "")),
        git_tag=str(data.get("git_tag", "")),
        timestamp=str(data.get("timestamp", "")),
    )


def _registration_to_dict(reg: AgentRegistration) -> dict[str, object]:
    return {
        "agent_id": reg.agent_id,
        "project_path": reg.project_path,
        "project_name": reg.project_name,
        "description": reg.description,
        "registered_at": reg.registered_at,
        "last_seen": reg.last_seen,
        "state": reg.state.value,
        "active_variant": reg.active_variant,
    }


def _registration_from_dict(data: dict[str, object]) -> AgentRegistration:
    return AgentRegistration(
        agent_id=str(data["agent_id"]),
        project_path=str(data["project_path"]),
        project_name=str(data["project_name"]),
        description=str(data.get("description", "")),
        registered_at=str(data["registered_at"]),
        last_seen=str(data["last_seen"]),
        state=AgentState(str(data.get("state", "registered"))),
        active_variant=str(data["active_variant"]) if data.get("active_variant") else None,
    )


def _registry_to_dict(registry: AgentRegistry) -> dict[str, object]:
    pv: dict[str, object] | None = None
    if registry.pipeline_version is not None:
        pv = _pipeline_version_to_dict(registry.pipeline_version)
    return {
        "schema_version": registry.schema_version,
        "pipeline_project_path": registry.pipeline_project_path,
        "pipeline_version": pv,
        "agents": [_registration_to_dict(a) for a in registry.agents],
        "updated_at": registry.updated_at,
    }


def _registry_from_dict(data: dict[str, object]) -> AgentRegistry:
    pv_data = data.get("pipeline_version")
    pv: PipelineVersion | None = None
    if pv_data is not None and isinstance(pv_data, dict):
        pv = _pipeline_version_from_dict(pv_data)

    agents_raw = data.get("agents", [])
    if not isinstance(agents_raw, list):
        agents_raw = []

    return AgentRegistry(
        schema_version=int(str(data.get("schema_version", 1))),
        pipeline_project_path=str(data.get("pipeline_project_path", "")),
        pipeline_version=pv,
        agents=tuple(_registration_from_dict(a) for a in agents_raw if isinstance(a, dict)),
        updated_at=str(data.get("updated_at", "")),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_registry_dir() -> Path:
    """Resolve and create the registry directory.

    Returns:
        The absolute path to ``~/.claude/kicad-agents/``.
    """
    d = Path(AGENT_REGISTRY_DIR).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    (d / "agents").mkdir(exist_ok=True)
    return d


def load_registry(path: Path | None = None) -> AgentRegistry:
    """Load the global agent registry from disk.

    Args:
        path: Explicit path to ``registry.json``.  If *None*, uses the
            default registry directory.

    Returns:
        The deserialized registry, or a fresh empty one if the file
        does not yet exist.
    """
    if path is None:
        path = get_registry_dir() / AGENT_REGISTRY_FILENAME

    if not path.exists():
        return AgentRegistry()

    try:
        raw = path.read_text(encoding="utf-8")
        data: dict[str, object] = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentError(f"Cannot read agent registry {path}: {exc}") from exc

    return _registry_from_dict(data)


def save_registry(registry: AgentRegistry, path: Path | None = None) -> None:
    """Persist the global agent registry to disk.

    Args:
        registry: The registry to save.
        path: Explicit path.  If *None*, uses the default registry directory.
    """
    if path is None:
        path = get_registry_dir() / AGENT_REGISTRY_FILENAME

    _write_atomic(path, _registry_to_dict(registry))
    log.info("Saved agent registry to %s", path)


def get_pipeline_version(pipeline_root: Path) -> PipelineVersion:
    """Read the current pipeline git HEAD and return a version snapshot.

    Args:
        pipeline_root: Root directory of the pipeline repository.

    Returns:
        A :class:`PipelineVersion` with the current hash, tag, and timestamp.
    """
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(pipeline_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"

    try:
        git_tag = subprocess.check_output(
            ["git", "describe", "--tags", "--always"],
            cwd=str(pipeline_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_tag = ""

    return PipelineVersion(
        git_hash=git_hash,
        git_tag=git_tag,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


def save_pipeline_version(version: PipelineVersion, registry_dir: Path | None = None) -> None:
    """Write the pipeline version marker file.

    Args:
        version: The version to persist.
        registry_dir: Directory containing the registry.  Defaults to
            :func:`get_registry_dir`.
    """
    if registry_dir is None:
        registry_dir = get_registry_dir()

    path = registry_dir / AGENT_PIPELINE_VERSION_FILENAME
    _write_atomic(path, _pipeline_version_to_dict(version))
    log.info("Saved pipeline version to %s", path)


def load_pipeline_version(registry_dir: Path | None = None) -> PipelineVersion | None:
    """Load the pipeline version marker file.

    Returns:
        The version, or *None* if the file does not exist.
    """
    if registry_dir is None:
        registry_dir = get_registry_dir()

    path = registry_dir / AGENT_PIPELINE_VERSION_FILENAME
    if not path.exists():
        return None

    try:
        raw = path.read_text(encoding="utf-8")
        data: dict[str, object] = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentError(f"Cannot read pipeline version {path}: {exc}") from exc

    return _pipeline_version_from_dict(data)
