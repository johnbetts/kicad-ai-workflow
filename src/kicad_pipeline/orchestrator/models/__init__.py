"""Data models for the orchestration layer.

All models use frozen dataclasses with tuple fields, matching the project
convention.  The central type is :class:`ProjectManifest`, serialized as
``kicad-project.json`` at the project root.
"""

from ..models import (
    PackageStrategy,
    RevisionRecord,
    StageId,
    StageRecord,
    StageState,
    VariantStatus,
)

__all__ = [
    "PackageStrategy",
    "RevisionRecord",
    "StageId",
    "StageRecord",
    "StageState",
    "VariantStatus",
]
