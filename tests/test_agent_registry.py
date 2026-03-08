"""Tests for kicad_pipeline.agents.registry persistence layer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.agents.models import (
    AgentRegistration,
    AgentRegistry,
    AgentState,
    PipelineVersion,
)
from kicad_pipeline.agents.registry import (
    _pipeline_version_from_dict,
    _pipeline_version_to_dict,
    _registration_from_dict,
    _registration_to_dict,
    _registry_from_dict,
    _registry_to_dict,
    _write_atomic,
    get_registry_dir,
    load_pipeline_version,
    load_registry,
    save_pipeline_version,
    save_registry,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_pipeline_version() -> PipelineVersion:
    return PipelineVersion(
        git_hash="abc123def456",
        git_tag="v0.5.0",
        timestamp="2026-03-07T12:00:00+00:00",
    )


@pytest.fixture()
def sample_registration() -> AgentRegistration:
    return AgentRegistration(
        agent_id="agent-led-blinker",
        project_path="/home/user/led-blinker",
        project_name="led-blinker",
        description="Simple LED blinker board",
        registered_at="2026-03-07T10:00:00+00:00",
        last_seen="2026-03-07T11:30:00+00:00",
        state=AgentState.RUNNING,
        active_variant="smd-0603",
    )


@pytest.fixture()
def sample_registry(
    sample_pipeline_version: PipelineVersion,
    sample_registration: AgentRegistration,
) -> AgentRegistry:
    return AgentRegistry(
        schema_version=1,
        pipeline_project_path="/home/user/kicad-ai-workflow",
        pipeline_version=sample_pipeline_version,
        agents=(sample_registration,),
        updated_at="2026-03-07T12:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# _write_atomic
# ---------------------------------------------------------------------------


def test_write_atomic_writes_valid_json(tmp_path: Path) -> None:
    target = tmp_path / "subdir" / "data.json"
    payload: dict[str, object] = {"key": "value", "count": 42}

    _write_atomic(target, payload)

    assert target.exists()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == payload


def test_write_atomic_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    _write_atomic(target, {"version": 1})
    _write_atomic(target, {"version": 2})

    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["version"] == 2


# ---------------------------------------------------------------------------
# PipelineVersion roundtrip
# ---------------------------------------------------------------------------


def test_pipeline_version_roundtrip(sample_pipeline_version: PipelineVersion) -> None:
    d = _pipeline_version_to_dict(sample_pipeline_version)
    restored = _pipeline_version_from_dict(d)

    assert restored == sample_pipeline_version
    assert restored.git_hash == "abc123def456"
    assert restored.git_tag == "v0.5.0"


# ---------------------------------------------------------------------------
# AgentRegistration roundtrip
# ---------------------------------------------------------------------------


def test_registration_roundtrip(sample_registration: AgentRegistration) -> None:
    d = _registration_to_dict(sample_registration)
    restored = _registration_from_dict(d)

    assert restored == sample_registration
    assert restored.agent_id == "agent-led-blinker"
    assert restored.state == AgentState.RUNNING
    assert restored.active_variant == "smd-0603"


# ---------------------------------------------------------------------------
# AgentRegistry roundtrip
# ---------------------------------------------------------------------------


def test_registry_roundtrip(sample_registry: AgentRegistry) -> None:
    d = _registry_to_dict(sample_registry)
    restored = _registry_from_dict(d)

    assert restored == sample_registry
    assert restored.schema_version == 1
    assert len(restored.agents) == 1
    assert restored.pipeline_version is not None
    assert restored.pipeline_version.git_hash == "abc123def456"


def test_registry_roundtrip_no_pipeline_version() -> None:
    registry = AgentRegistry(
        schema_version=1,
        pipeline_project_path="/tmp/proj",
        pipeline_version=None,
        agents=(),
        updated_at="2026-03-07T12:00:00+00:00",
    )
    d = _registry_to_dict(registry)
    restored = _registry_from_dict(d)

    assert restored.pipeline_version is None
    assert restored.agents == ()


# ---------------------------------------------------------------------------
# load_registry / save_registry
# ---------------------------------------------------------------------------


def test_load_registry_returns_empty_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent" / "registry.json"
    result = load_registry(missing)

    assert isinstance(result, AgentRegistry)
    assert result.agents == ()
    assert result.pipeline_version is None


def test_load_save_registry_roundtrip(
    tmp_path: Path, sample_registry: AgentRegistry
) -> None:
    path = tmp_path / "registry.json"
    save_registry(sample_registry, path)
    loaded = load_registry(path)

    assert loaded == sample_registry


# ---------------------------------------------------------------------------
# get_registry_dir
# ---------------------------------------------------------------------------


def test_get_registry_dir_creates_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake_dir = str(tmp_path / ".claude" / "kicad-agents")
    monkeypatch.setattr(
        "kicad_pipeline.agents.registry.AGENT_REGISTRY_DIR", fake_dir
    )

    result = get_registry_dir()

    assert result.exists()
    assert (result / "agents").exists()


# ---------------------------------------------------------------------------
# save_pipeline_version / load_pipeline_version
# ---------------------------------------------------------------------------


def test_save_load_pipeline_version_roundtrip(
    tmp_path: Path, sample_pipeline_version: PipelineVersion
) -> None:
    save_pipeline_version(sample_pipeline_version, registry_dir=tmp_path)
    loaded = load_pipeline_version(registry_dir=tmp_path)

    assert loaded is not None
    assert loaded == sample_pipeline_version


def test_load_pipeline_version_returns_none_when_missing(tmp_path: Path) -> None:
    result = load_pipeline_version(registry_dir=tmp_path)

    assert result is None
