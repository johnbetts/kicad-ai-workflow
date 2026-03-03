"""Data models for the orchestration layer.

All models use frozen dataclasses with tuple fields, matching the project
convention.  The central type is :class:`ProjectManifest`, serialized as
``kicad-project.json`` at the project root.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VariantStatus(Enum):
    """Lifecycle status of a design variant."""

    DRAFT = "draft"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    RELEASED = "released"
    ARCHIVED = "archived"


class StageId(Enum):
    """Pipeline stages in execution order."""

    REQUIREMENTS = "requirements"
    SCHEMATIC = "schematic"
    PCB = "pcb"
    VALIDATION = "validation"
    PRODUCTION = "production"


class StageState(Enum):
    """State of a single stage within a variant."""

    PENDING = "pending"
    GENERATED = "generated"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------

STAGE_ORDER: tuple[StageId, ...] = (
    StageId.REQUIREMENTS,
    StageId.SCHEMATIC,
    StageId.PCB,
    StageId.VALIDATION,
    StageId.PRODUCTION,
)
"""Canonical stage ordering used for transition validation."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackageStrategy:
    """Defines how a variant selects component packages."""

    name: str
    resistor_package: str = "0805"
    capacitor_package: str = "0805"
    led_package: str = "0805"
    prefer_smd: bool = True
    notes: str = ""


@dataclass(frozen=True)
class StageRecord:
    """State of a single pipeline stage for a variant."""

    stage: StageId
    state: StageState = StageState.PENDING
    generated_at: str | None = None
    approved_at: str | None = None
    generation_count: int = 0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RevisionRecord:
    """A production revision snapshot for a variant."""

    number: int
    created_at: str
    git_tag: str
    commit_hash: str
    notes: str = ""
    sent_to_fab: bool = False
    fab_order_id: str | None = None


@dataclass(frozen=True)
class VariantRecord:
    """Complete state for one design variant."""

    name: str
    display_name: str
    description: str
    status: VariantStatus
    package_strategy: PackageStrategy
    stages: tuple[StageRecord, ...] = ()
    revisions: tuple[RevisionRecord, ...] = ()
    created_at: str = ""
    updated_at: str = ""
    released_tag: str | None = None


@dataclass(frozen=True)
class ProjectManifest:
    """Top-level orchestration state for the entire project."""

    schema_version: int = 1
    project_name: str = ""
    description: str = ""
    original_spec: str = ""
    created_at: str = ""
    updated_at: str = ""
    variants: tuple[VariantRecord, ...] = ()
    active_variant: str | None = None


# ---------------------------------------------------------------------------
# Default package strategies
# ---------------------------------------------------------------------------

DEFAULT_PACKAGE_STRATEGIES: tuple[PackageStrategy, ...] = (
    PackageStrategy(
        name="0805",
        resistor_package="0805",
        capacitor_package="0805",
        led_package="0805",
        notes="Standard SMD size, easy to hand-solder",
    ),
    PackageStrategy(
        name="0603",
        resistor_package="0603",
        capacitor_package="0603",
        led_package="0603",
        notes="Compact SMD size, good for space-constrained designs",
    ),
    PackageStrategy(
        name="0402",
        resistor_package="0402",
        capacitor_package="0402",
        led_package="0402",
        notes="Ultra-compact SMD, reflow only",
    ),
    PackageStrategy(
        name="through-hole",
        resistor_package="Axial_DIN0207",
        capacitor_package="C_Disc_D5.0mm",
        led_package="LED_D3.0mm",
        prefer_smd=False,
        notes="Through-hole for prototyping and breadboard compatibility",
    ),
)


def default_stages() -> tuple[StageRecord, ...]:
    """Return a fresh set of stage records, all in PENDING state."""
    return tuple(StageRecord(stage=sid) for sid in STAGE_ORDER)


def get_strategy_by_name(name: str) -> PackageStrategy | None:
    """Look up a default package strategy by name.

    Args:
        name: Strategy name (e.g. "0805", "through-hole").

    Returns:
        The matching strategy, or None if not found.
    """
    for s in DEFAULT_PACKAGE_STRATEGIES:
        if s.name == name:
            return s
    return None
