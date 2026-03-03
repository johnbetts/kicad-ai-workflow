"""Persistence layer for the project manifest (kicad-project.json).

The manifest tracks orchestration state: variants, stages, revisions.
Serialization follows the same pattern as
:mod:`kicad_pipeline.requirements.decomposer` — explicit ``_to_dict`` /
``_from_dict`` converters with no third-party dependencies.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import OrchestrationError
from kicad_pipeline.orchestrator.models import (
    PackageStrategy,
    ProjectManifest,
    RevisionRecord,
    StageId,
    StageRecord,
    StageState,
    VariantRecord,
    VariantStatus,
)

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

MANIFEST_FILENAME: str = "kicad-project.json"
"""Default filename for the project manifest."""


# ---------------------------------------------------------------------------
# Serialization: models → dict
# ---------------------------------------------------------------------------


def _package_strategy_to_dict(ps: PackageStrategy) -> dict[str, object]:
    return {
        "name": ps.name,
        "resistor_package": ps.resistor_package,
        "capacitor_package": ps.capacitor_package,
        "led_package": ps.led_package,
        "prefer_smd": ps.prefer_smd,
        "notes": ps.notes,
    }


def _stage_record_to_dict(sr: StageRecord) -> dict[str, object]:
    return {
        "stage": sr.stage.value,
        "state": sr.state.value,
        "generated_at": sr.generated_at,
        "approved_at": sr.approved_at,
        "generation_count": sr.generation_count,
        "notes": list(sr.notes),
    }


def _revision_record_to_dict(rr: RevisionRecord) -> dict[str, object]:
    return {
        "number": rr.number,
        "created_at": rr.created_at,
        "git_tag": rr.git_tag,
        "commit_hash": rr.commit_hash,
        "notes": rr.notes,
        "sent_to_fab": rr.sent_to_fab,
        "fab_order_id": rr.fab_order_id,
    }


def _variant_record_to_dict(vr: VariantRecord) -> dict[str, object]:
    return {
        "name": vr.name,
        "display_name": vr.display_name,
        "description": vr.description,
        "status": vr.status.value,
        "package_strategy": _package_strategy_to_dict(vr.package_strategy),
        "stages": [_stage_record_to_dict(s) for s in vr.stages],
        "revisions": [_revision_record_to_dict(r) for r in vr.revisions],
        "created_at": vr.created_at,
        "updated_at": vr.updated_at,
        "released_tag": vr.released_tag,
    }


def manifest_to_dict(manifest: ProjectManifest) -> dict[str, object]:
    """Serialize a :class:`ProjectManifest` to a JSON-compatible dict.

    Args:
        manifest: The manifest to serialize.

    Returns:
        A plain dict suitable for :func:`json.dumps`.
    """
    return {
        "schema_version": manifest.schema_version,
        "project_name": manifest.project_name,
        "description": manifest.description,
        "original_spec": manifest.original_spec,
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
        "active_variant": manifest.active_variant,
        "variants": [_variant_record_to_dict(v) for v in manifest.variants],
    }


# ---------------------------------------------------------------------------
# Deserialization: dict → models
# ---------------------------------------------------------------------------


def _package_strategy_from_dict(data: dict[str, object]) -> PackageStrategy:
    return PackageStrategy(
        name=str(data["name"]),
        resistor_package=str(data.get("resistor_package", "0805")),
        capacitor_package=str(data.get("capacitor_package", "0805")),
        led_package=str(data.get("led_package", "0805")),
        prefer_smd=bool(data.get("prefer_smd", True)),
        notes=str(data.get("notes", "")),
    )


def _stage_record_from_dict(data: dict[str, object]) -> StageRecord:
    generated_at = data.get("generated_at")
    approved_at = data.get("approved_at")
    return StageRecord(
        stage=StageId(str(data["stage"])),
        state=StageState(str(data["state"])),
        generated_at=str(generated_at) if generated_at is not None else None,
        approved_at=str(approved_at) if approved_at is not None else None,
        generation_count=int(str(data.get("generation_count", 0))),
        notes=tuple(str(x) for x in _as_list(data.get("notes", []))),
    )


def _revision_record_from_dict(data: dict[str, object]) -> RevisionRecord:
    fab_order_id = data.get("fab_order_id")
    return RevisionRecord(
        number=int(str(data["number"])),
        created_at=str(data["created_at"]),
        git_tag=str(data["git_tag"]),
        commit_hash=str(data["commit_hash"]),
        notes=str(data.get("notes", "")),
        sent_to_fab=bool(data.get("sent_to_fab", False)),
        fab_order_id=str(fab_order_id) if fab_order_id is not None else None,
    )


def _variant_record_from_dict(data: dict[str, object]) -> VariantRecord:
    ps_data = _as_dict(data["package_strategy"])
    released_tag = data.get("released_tag")
    return VariantRecord(
        name=str(data["name"]),
        display_name=str(data["display_name"]),
        description=str(data.get("description", "")),
        status=VariantStatus(str(data["status"])),
        package_strategy=_package_strategy_from_dict(ps_data),
        stages=tuple(
            _stage_record_from_dict(_as_dict(s))
            for s in _as_list(data.get("stages", []))
        ),
        revisions=tuple(
            _revision_record_from_dict(_as_dict(r))
            for r in _as_list(data.get("revisions", []))
        ),
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", "")),
        released_tag=str(released_tag) if released_tag is not None else None,
    )


def manifest_from_dict(data: dict[str, object]) -> ProjectManifest:
    """Deserialize a :class:`ProjectManifest` from a plain dict.

    Args:
        data: A dict with the same structure produced by
            :func:`manifest_to_dict`.

    Returns:
        A :class:`ProjectManifest` instance.

    Raises:
        OrchestrationError: If *data* is malformed.
    """
    try:
        active = data.get("active_variant")
        return ProjectManifest(
            schema_version=int(str(data.get("schema_version", 1))),
            project_name=str(data["project_name"]),
            description=str(data.get("description", "")),
            original_spec=str(data.get("original_spec", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            active_variant=str(active) if active is not None else None,
            variants=tuple(
                _variant_record_from_dict(_as_dict(v))
                for v in _as_list(data.get("variants", []))
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OrchestrationError(
            f"Failed to deserialize ProjectManifest: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def save_manifest(manifest: ProjectManifest, project_root: Path) -> None:
    """Write the manifest to ``kicad-project.json`` in *project_root*.

    Args:
        manifest: The manifest to persist.
        project_root: Directory containing the project.
    """
    path = project_root / MANIFEST_FILENAME
    data = manifest_to_dict(manifest)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    log.info("Saved manifest to %s", path)


def load_manifest(project_root: Path) -> ProjectManifest:
    """Load the manifest from ``kicad-project.json`` in *project_root*.

    Args:
        project_root: Directory containing the project.

    Returns:
        The deserialized :class:`ProjectManifest`.

    Raises:
        OrchestrationError: If the file is missing, not valid JSON, or
            fails deserialization.
    """
    path = project_root / MANIFEST_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OrchestrationError(
            f"Cannot read manifest file {path}: {exc}"
        ) from exc

    try:
        data: dict[str, object] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OrchestrationError(
            f"Manifest file {path} is not valid JSON: {exc}"
        ) from exc

    log.info("Loaded manifest from %s", path)
    return manifest_from_dict(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_dict(value: object) -> dict[str, object]:
    """Assert *value* is a dict and return it typed."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict, got {type(value).__name__!r}")
    return value


def _as_list(value: object) -> list[object]:
    """Assert *value* is a list and return it."""
    if not isinstance(value, list):
        raise TypeError(f"Expected list, got {type(value).__name__!r}")
    return value
