"""Electrical Rules Check (ERC) for KiCad schematics.

Validates a completed :class:`~kicad_pipeline.models.schematic.Schematic`
against a set of electrical correctness rules and returns an
:class:`ERCReport` describing any violations found.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.schematic import (
        GlobalLabel,
        Schematic,
        SymbolInstance,
        Wire,
    )

__all__ = [
    "ERCReport",
    "ERCSeverity",
    "ERCViolation",
    "run_erc",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ERC data types
# ---------------------------------------------------------------------------


class ERCSeverity(Enum):
    """Severity level of an ERC violation."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ERCViolation:
    """A single ERC rule violation.

    Attributes:
        severity: How critical the violation is.
        rule: Machine-readable rule identifier (e.g. ``"duplicate_ref"``).
        message: Human-readable description of the violation.
        ref: The affected component reference designator, if applicable.
    """

    severity: ERCSeverity
    rule: str
    message: str
    ref: str | None = None


@dataclass(frozen=True)
class ERCReport:
    """The result of an ERC run against a schematic.

    Attributes:
        violations: All violations found, in check order.
    """

    violations: tuple[ERCViolation, ...]

    @property
    def errors(self) -> tuple[ERCViolation, ...]:
        """Return only ERROR-severity violations."""
        return tuple(v for v in self.violations if v.severity == ERCSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ERCViolation, ...]:
        """Return only WARNING-severity violations."""
        return tuple(v for v in self.violations if v.severity == ERCSeverity.WARNING)

    @property
    def passed(self) -> bool:
        """True when there are no ERROR-severity violations."""
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Individual ERC checks
# ---------------------------------------------------------------------------


def _check_duplicate_refs(
    symbols: tuple[SymbolInstance, ...],
) -> list[ERCViolation]:
    """Check for duplicate reference designators.

    Power symbols (ref starting with ``#``) are excluded from this check
    because KiCad auto-assigns them sequential ``#PWR`` identifiers.

    Args:
        symbols: All symbol instances in the schematic.

    Returns:
        List of ERROR violations for each duplicate ref pair found.
    """
    seen: dict[str, str] = {}  # ref → lib_id of first occurrence
    violations: list[ERCViolation] = []

    for sym in symbols:
        if sym.ref.startswith("#"):
            continue
        if sym.ref in seen:
            violations.append(
                ERCViolation(
                    severity=ERCSeverity.ERROR,
                    rule="duplicate_ref",
                    message=(
                        f"Reference '{sym.ref}' is used by more than one symbol "
                        f"(lib_ids: {seen[sym.ref]!r} and {sym.lib_id!r})"
                    ),
                    ref=sym.ref,
                )
            )
        else:
            seen[sym.ref] = sym.lib_id

    return violations


def _check_missing_values(
    symbols: tuple[SymbolInstance, ...],
) -> list[ERCViolation]:
    """Check for symbol instances with an empty value field.

    Args:
        symbols: All symbol instances in the schematic.

    Returns:
        List of ERROR violations for each symbol with an empty value.
    """
    violations: list[ERCViolation] = []
    for sym in symbols:
        if not sym.value.strip():
            violations.append(
                ERCViolation(
                    severity=ERCSeverity.ERROR,
                    rule="missing_value",
                    message=f"Symbol '{sym.ref}' has an empty value field.",
                    ref=sym.ref,
                )
            )
    return violations


def _check_no_lib_symbol(schematic: Schematic) -> list[ERCViolation]:
    """Check that every placed symbol has a corresponding lib_symbol entry.

    This is a WARNING-only check because the symbol may live in a standard
    KiCad system library and simply not be embedded in the schematic file.

    Args:
        schematic: The schematic to validate.

    Returns:
        List of WARNING violations for each missing lib_symbol entry.
    """
    lib_ids = {ls.lib_id for ls in schematic.lib_symbols}
    violations: list[ERCViolation] = []

    for sym in schematic.symbols:
        if sym.lib_id not in lib_ids:
            violations.append(
                ERCViolation(
                    severity=ERCSeverity.WARNING,
                    rule="no_lib_symbol",
                    message=(
                        f"Symbol '{sym.ref}' references lib_id '{sym.lib_id}' "
                        "which is not embedded in this schematic."
                    ),
                    ref=sym.ref,
                )
            )

    return violations


def _check_floating_wires(wires: tuple[Wire, ...]) -> list[ERCViolation]:
    """Check for wire endpoints that appear only once across all wires.

    A wire endpoint that is unique (appears in no other wire start/end)
    may indicate a floating wire not connected to any pin or junction.
    This is a heuristic check — it does not cross-reference symbol pin
    positions.

    Args:
        wires: All wire segments in the schematic.

    Returns:
        List of WARNING violations for each detected floating wire.
    """
    if len(wires) < 2:
        # A single wire or no wires cannot self-connect; skip.
        return []

    # Count how many times each endpoint coordinate appears
    endpoint_count: dict[tuple[float, float], int] = {}
    for wire in wires:
        for pt in (wire.start, wire.end):
            key = (pt.x, pt.y)
            endpoint_count[key] = endpoint_count.get(key, 0) + 1

    violations: list[ERCViolation] = []
    reported: set[tuple[float, float]] = set()

    for wire in wires:
        for pt in (wire.start, wire.end):
            key = (pt.x, pt.y)
            if endpoint_count[key] == 1 and key not in reported:
                reported.add(key)
                violations.append(
                    ERCViolation(
                        severity=ERCSeverity.WARNING,
                        rule="floating_wire",
                        message=(
                            f"Wire endpoint at ({pt.x}, {pt.y}) appears only once "
                            "and may not be connected to a pin or another wire."
                        ),
                    )
                )

    return violations


def _check_unmatched_global_labels(
    global_labels: tuple[GlobalLabel, ...],
) -> list[ERCViolation]:
    """Check for global labels whose text appears only once.

    A global label text that appears on exactly one sheet suggests a
    dangling off-sheet connection.  This is a WARNING because the
    connection may be intentional (e.g. a connector export).

    Args:
        global_labels: All global labels in the schematic.

    Returns:
        List of WARNING violations for each unique global label text.
    """
    label_counts: dict[str, int] = {}
    for lbl in global_labels:
        label_counts[lbl.text] = label_counts.get(lbl.text, 0) + 1

    violations: list[ERCViolation] = []
    for text, count in label_counts.items():
        if count == 1:
            violations.append(
                ERCViolation(
                    severity=ERCSeverity.WARNING,
                    rule="unmatched_global_label",
                    message=(
                        f"Global label '{text}' appears only once. "
                        "It may be an unmatched off-sheet connection."
                    ),
                )
            )

    return violations


# ---------------------------------------------------------------------------
# Public ERC runner
# ---------------------------------------------------------------------------


def run_erc(schematic: Schematic) -> ERCReport:
    """Run all ERC checks on a schematic and return an :class:`ERCReport`.

    Checks performed (in order):

    1. **duplicate_ref** — two :class:`SymbolInstance` objects share the same
       reference designator (power symbols with ``#`` prefix are excluded).
    2. **missing_value** — a :class:`SymbolInstance` has an empty value field.
    3. **no_lib_symbol** — a placed symbol's ``lib_id`` is not found in
       ``schematic.lib_symbols`` (WARNING only).
    4. **floating_wire** — wire endpoints that appear only once across all wire
       segments (WARNING only — heuristic, does not check pin positions).
    5. **unmatched_global_label** — a global label text appears only once
       (WARNING only).

    Args:
        schematic: The completed schematic to validate.

    Returns:
        An :class:`ERCReport` containing all violations found.
    """
    violations: list[ERCViolation] = []

    violations.extend(_check_duplicate_refs(schematic.symbols))
    violations.extend(_check_missing_values(schematic.symbols))
    violations.extend(_check_no_lib_symbol(schematic))
    violations.extend(_check_floating_wires(schematic.wires))
    violations.extend(_check_unmatched_global_labels(schematic.global_labels))

    report = ERCReport(violations=tuple(violations))
    error_count = len(report.errors)
    warn_count = len(report.warnings)

    log.info(
        "ERC complete: %d error(s), %d warning(s)",
        error_count,
        warn_count,
    )

    return report
