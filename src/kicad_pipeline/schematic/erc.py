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
        LibSymbol,
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


def _compute_pin_position(
    sym_pos: tuple[float, float],
    sym_rotation: float,
    pin_at: tuple[float, float],
    pin_rotation: float,
    pin_length: float,
) -> tuple[float, float]:
    """Compute the absolute connection-point position of a pin.

    In KiCad schematics, ``LibPin.at`` gives the pin stub origin relative to
    the symbol origin (with Y-negation applied when placed).  The actual
    connection point is at the *tip* of the pin — ``pin_length`` mm away in
    the direction given by ``pin_rotation`` (0=right, 90=up, 180=left,
    270=down in KiCad convention, but the schematic Y axis is inverted).

    This function accounts for both the symbol's placement rotation and the
    per-pin rotation/length to yield the wire-connection coordinate.
    """
    import math

    # Pin connection point relative to symbol origin:
    # Start from pin.at, then extend by pin_length along pin_rotation.
    # KiCad pin rotation: 0°=right, 90°=up, 180°=left, 270°=down
    # In schematic coords (Y-down), 90° means -Y.
    total_pin_angle = math.radians(pin_rotation)
    # Connection point relative to symbol origin (Y-negated for schematic)
    rel_x = pin_at[0] + pin_length * math.cos(total_pin_angle)
    rel_y = -pin_at[1] - pin_length * math.sin(total_pin_angle)

    # Apply symbol rotation
    sym_angle = math.radians(sym_rotation)
    cos_s = math.cos(sym_angle)
    sin_s = math.sin(sym_angle)
    rotated_x = rel_x * cos_s - rel_y * sin_s
    rotated_y = rel_x * sin_s + rel_y * cos_s

    abs_x = sym_pos[0] + rotated_x
    abs_y = sym_pos[1] + rotated_y
    return (abs_x, abs_y)


_WIRE_PIN_TOLERANCE_MM = 0.01


def _check_wire_pin_alignment(schematic: Schematic) -> list[ERCViolation]:
    """Check that every wire endpoint lands on a valid connection point.

    Valid connection points include: symbol pin tips, power symbol positions,
    label positions, global label positions, junction positions, no-connect
    positions, and wire endpoints that appear more than once (wire-to-wire
    junctions).

    Args:
        schematic: The schematic to validate.

    Returns:
        List of WARNING violations for wire endpoints not near any connection
        point.
    """
    if not schematic.wires:
        return []

    # Build set of valid connection points ----------------------------------

    valid_points: set[tuple[float, float]] = set()

    # Pin connection points from placed symbols
    lib_map: dict[str, LibSymbol] = {ls.lib_id: ls for ls in schematic.lib_symbols}
    for sym in schematic.symbols:
        ls = lib_map.get(sym.lib_id)
        if ls is None:
            continue
        for pin in ls.pins:
            pt = _compute_pin_position(
                (sym.position.x, sym.position.y),
                sym.rotation,
                (pin.at.x, pin.at.y),
                pin.rotation,
                pin.length,
            )
            valid_points.add(pt)

    # Power symbol positions (connection point is the symbol position)
    for ps in schematic.power_symbols:
        valid_points.add((ps.position.x, ps.position.y))

    # Labels, global labels, junctions, no-connects
    for lbl in schematic.labels:
        valid_points.add((lbl.position.x, lbl.position.y))
    for gl in schematic.global_labels:
        valid_points.add((gl.position.x, gl.position.y))
    for jn in schematic.junctions:
        valid_points.add((jn.position.x, jn.position.y))
    for nc in schematic.no_connects:
        valid_points.add((nc.position.x, nc.position.y))

    # Wire-to-wire junctions (endpoints appearing more than once)
    endpoint_count: dict[tuple[float, float], int] = {}
    for w in schematic.wires:
        for wpt in (w.start, w.end):
            wkey = (wpt.x, wpt.y)
            endpoint_count[wkey] = endpoint_count.get(wkey, 0) + 1
    for ekey, ecount in endpoint_count.items():
        if ecount > 1:
            valid_points.add(ekey)

    # Check each wire endpoint against valid points -------------------------

    violations: list[ERCViolation] = []
    reported: set[tuple[float, float]] = set()

    for wire in schematic.wires:
        for wep in (wire.start, wire.end):
            key = (wep.x, wep.y)
            if key in reported:
                continue
            # Fast exact match
            if key in valid_points:
                continue
            # Tolerance match
            matched = False
            for vp in valid_points:
                dx = key[0] - vp[0]
                dy = key[1] - vp[1]
                if dx * dx + dy * dy <= _WIRE_PIN_TOLERANCE_MM**2:
                    matched = True
                    break
            if not matched:
                reported.add(key)
                violations.append(
                    ERCViolation(
                        severity=ERCSeverity.WARNING,
                        rule="wire_pin_misaligned",
                        message=(
                            f"Wire endpoint at ({wep.x}, {wep.y}) does not align "
                            "with any pin, label, junction, or other wire endpoint."
                        ),
                    )
                )

    return violations


def _check_symbol_overlap(schematic: Schematic) -> list[ERCViolation]:
    """Check for overlapping symbol instances.

    Estimates bounding boxes from ``LibSymbol`` pin extents and checks all
    pairs for AABB overlap.  Power symbols (ref starting with ``#``) are
    excluded because they intentionally share coordinates.

    Args:
        schematic: The schematic to validate.

    Returns:
        List of WARNING violations for each overlapping symbol pair.
    """
    lib_map: dict[str, LibSymbol] = {ls.lib_id: ls for ls in schematic.lib_symbols}

    # Build bounding boxes: (min_x, min_y, max_x, max_y) for each non-power symbol
    bboxes: list[tuple[str, float, float, float, float]] = []

    for sym in schematic.symbols:
        if sym.ref.startswith("#"):
            continue
        ls = lib_map.get(sym.lib_id)
        if ls is None or not ls.pins:
            continue

        # Compute extent from pin positions (relative to symbol origin)
        pin_xs = [p.at.x for p in ls.pins]
        pin_ys = [p.at.y for p in ls.pins]
        half_w = max(abs(max(pin_xs)), abs(min(pin_xs)))
        half_h = max(abs(max(pin_ys)), abs(min(pin_ys)))

        # Add a small margin for pin length
        half_w += 2.54
        half_h += 2.54

        bboxes.append(
            (
                sym.ref,
                sym.position.x - half_w,
                sym.position.y - half_h,
                sym.position.x + half_w,
                sym.position.y + half_h,
            )
        )

    violations: list[ERCViolation] = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            ref_a, ax1, ay1, ax2, ay2 = bboxes[i]
            ref_b, bx1, by1, bx2, by2 = bboxes[j]
            # AABB overlap test
            if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                violations.append(
                    ERCViolation(
                        severity=ERCSeverity.WARNING,
                        rule="symbol_overlap",
                        message=(
                            f"Symbols '{ref_a}' and '{ref_b}' have overlapping "
                            "bounding boxes and may visually collide."
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
    6. **wire_pin_misaligned** — wire endpoint doesn't land on any pin,
       label, junction, or other wire endpoint (WARNING only).
    7. **symbol_overlap** — two non-power symbol bounding boxes overlap
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
    violations.extend(_check_wire_pin_alignment(schematic))
    violations.extend(_check_symbol_overlap(schematic))

    report = ERCReport(violations=tuple(violations))
    error_count = len(report.errors)
    warn_count = len(report.warnings)

    log.info(
        "ERC complete: %d error(s), %d warning(s)",
        error_count,
        warn_count,
    )

    return report
