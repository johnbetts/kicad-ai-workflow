"""Agent registry persistence layer.

Handles serialization/deserialization of agent registry data to/from JSON files.
Provides atomic write operations and versioned schema support.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.agents.models import (
    AgentRegistration,
    AgentRegistry,
    PipelineVersion,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Path management
# ---------------------------------------------------------------------------


def get_registry_dir() -> Path:
    """Get the directory where registry files are stored."""
    base_dir = Path.home() / ".claude" / "kicad-agents"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _pipeline_version_to_dict(version: PipelineVersion) -> dict[str, str]:
    """Convert PipelineVersion to dict for JSON serialization."""
    return {
        "git_hash": version.git_hash,
        "git_tag": version.git_tag,
        "timestamp": version.timestamp,
    }


def _pipeline_version_from_dict(data: dict[str, str]) -> PipelineVersion:
    """Convert dict to PipelineVersion."""
    return PipelineVersion(
        git_hash=data["git_hash"],
        git_tag=data["git_tag"],
        timestamp=data["timestamp"],
    )


def _registration_to_dict(registration: AgentRegistration) -> dict[str, str | None]:
    """Convert AgentRegistration to dict for JSON serialization."""
    return {
        "agent_id": registration.agent_id,
        "project_path": registration.project_path,
        "project_name": registration.project_name,
        "description": registration.description,
        "registered_at": registration.registered_at,
        "last_seen": registration.last_seen,
        "state": registration.state.value,
        "active_variant": registration.active_variant,
    }


def _registration_from_dict(data: dict[str, str | None]) -> AgentRegistration:
    """Convert dict to AgentRegistration."""
    from kicad_pipeline.agents.models import AgentState

    return AgentRegistration(
        agent_id=str(data["agent_id"]),
        project_path=str(data["project_path"]),
        project_name=str(data["project_name"]),
        description=str(data["description"]),
        registered_at=str(data["registered_at"]),
        last_seen=str(data["last_seen"]),
        state=AgentState(str(data["state"])),
        active_variant=data.get("active_variant"),
    )


def _registry_to_dict(registry: AgentRegistry) -> dict[str, str | int | list[dict[str, str | None]] | dict[str, str] | None]:
    """Convert AgentRegistry to dict for JSON serialization."""
    return {
        "schema_version": registry.schema_version,
        "pipeline_project_path": registry.pipeline_project_path,
        "pipeline_version": _pipeline_version_to_dict(registry.pipeline_version) if registry.pipeline_version else None,
        "agents": [_registration_to_dict(agent) for agent in registry.agents],
        "updated_at": registry.updated_at,
    }


def _registry_from_dict(data: dict[str, str | int | list[dict[str, str | None]] | dict[str, str] | None]) -> AgentRegistry:
    """Convert dict to AgentRegistry."""
    pipeline_version = None
    if data.get("pipeline_version"):
        pipeline_version = _pipeline_version_from_dict(data["pipeline_version"])  # type: ignore

    return AgentRegistry(
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        pipeline_project_path=str(data.get("pipeline_project_path", "")),
        pipeline_version=pipeline_version,
        agents=tuple(_registration_from_dict(agent_data) for agent_data in data.get("agents", [])),  # type: ignore
        updated_at=str(data.get("updated_at", "")),
    )


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def _write_atomic(file_path: Path, content: str) -> None:
    """Write content to file atomically using a temporary file."""
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

    try:
        # Write to temporary file first
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # Atomic move
        temp_path.replace(file_path)

    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_pipeline_version() -> PipelineVersion | None:
    """Load pipeline version from disk."""
    version_file = get_registry_dir() / "pipeline_version.json"

    if not version_file.exists():
        return None

    try:
        with open(version_file, encoding="utf-8") as f:
            data = json.load(f)
        return _pipeline_version_from_dict(data)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def save_pipeline_version(version: PipelineVersion) -> None:
    """Save pipeline version to disk."""
    version_file = get_registry_dir() / "pipeline_version.json"
    data = _pipeline_version_to_dict(version)
    _write_atomic(version_file, json.dumps(data, indent=2, ensure_ascii=False))


def load_registry() -> AgentRegistry:
    """Load agent registry from disk."""
    registry_file = get_registry_dir() / "registry.json"

    if not registry_file.exists():
        return AgentRegistry()

    try:
        with open(registry_file, encoding="utf-8") as f:
            data = json.load(f)
        return _registry_from_dict(data)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return AgentRegistry()


def save_registry(registry: AgentRegistry) -> None:
    """Save agent registry to disk."""
    registry_file = get_registry_dir() / "registry.json"
    data = _registry_to_dict(registry)
    _write_atomic(registry_file, json.dumps(data, indent=2, ensure_ascii=False))