"""Placement constraint auditor — standalone checker for constraint compliance.

Callable from CLI, review agent, or tests. Checks that all placement
constraints (proximity, ordering, grouping) are satisfied by the current
PCB component positions.

Usage::

    from kicad_pipeline.validation.constraint_auditor import audit_placement_constraints
    violations = audit_placement_constraints(pcb, requirements)
    for v in violations:
        print(f"  {v.severity}: {v.message}")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import (
        OrderingChain,
        PCBDesign,
        ProximityConstraint,
    )
    from kicad_pipeline.models.requirements import ProjectRequirements

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConstraintViolation:
    """A single placement constraint violation."""

    constraint_type: str  # "proximity", "ordering", "group", "trace_length"
    severity: str  # "critical", "major", "minor"
    refs: tuple[str, ...]
    message: str
    measured_value: float
    threshold: float
    suggested_fix: str


def _positions_from_pcb(
    pcb: PCBDesign,
) -> dict[str, tuple[float, float, float]]:
    """Extract {ref: (x, y, rotation)} from PCB footprints."""
    return {fp.ref: (fp.position.x, fp.position.y, fp.rotation) for fp in pcb.footprints}


def _find_pin_number(
    requirements: ProjectRequirements | None,
    ref: str,
    pin_name: str,
) -> str | None:
    """Look up a pin number from requirements by component ref and pin name."""
    if requirements is None:
        return None
    for comp in requirements.components:
        if comp.ref != ref:
            continue
        for pin in comp.pins:
            if pin.name.upper() == pin_name.upper():
                return pin.number
    return None


def _pad_world_position(
    pcb: PCBDesign | None,
    ref: str,
    pin_name: str,
    fallback_x: float,
    fallback_y: float,
    requirements: ProjectRequirements | None = None,
) -> tuple[float, float]:
    """Return the world (x, y) of a specific pad on a footprint.

    Matches pads by: (1) pad number, (2) pin name from requirements -> pad
    number lookup, (3) net name containing the pin name.
    Returns (*fallback_x*, *fallback_y*) if the pad or footprint is not found.
    """
    if pcb is None:
        return fallback_x, fallback_y
    fp = None
    for f in pcb.footprints:
        if f.ref == ref:
            fp = f
            break
    if fp is None:
        return fallback_x, fallback_y

    # Try direct pad number match (e.g. pin_name="2")
    target_number = pin_name
    # Also try looking up via requirements pin name -> number
    req_number = _find_pin_number(requirements, ref, pin_name)
    if req_number is not None:
        target_number = req_number

    for pad in fp.pads:
        if pad.number == target_number or pad.number == pin_name:
            rot_rad = math.radians(fp.rotation)
            cos_r = math.cos(rot_rad)
            sin_r = math.sin(rot_rad)
            wx = fp.position.x + pad.position.x * cos_r - pad.position.y * sin_r
            wy = fp.position.y + pad.position.x * sin_r + pad.position.y * cos_r
            return wx, wy
    return fallback_x, fallback_y


def _check_proximity(
    constraint: ProximityConstraint,
    positions: dict[str, tuple[float, float, float]],
    pcb: PCBDesign | None = None,
    requirements: ProjectRequirements | None = None,
) -> ConstraintViolation | None:
    """Check a single proximity constraint.

    When *constraint.target_pin* is set and *pcb* is provided, measures
    distance from the component center to the specific pad on the target
    footprint (not the target center).  This gives more accurate results
    for large ICs where pin positions differ significantly from the center.
    """
    if constraint.ref not in positions:
        return None
    if constraint.target_ref not in positions:
        return None

    rx, ry, _ = positions[constraint.ref]
    tx, ty, _ = positions[constraint.target_ref]
    dist_center = math.sqrt((rx - tx) ** 2 + (ry - ty) ** 2)

    # When target_pin is specified, also measure to the actual pad position
    # and use the shorter of center-to-center vs center-to-pin distance.
    # This prevents false positives when the component is close to the IC
    # but the specific pin is on the far side.
    dist = dist_center
    if constraint.target_pin and pcb is not None:
        px, py = _pad_world_position(
            pcb, constraint.target_ref, constraint.target_pin, tx, ty,
            requirements=requirements,
        )
        dist_pin = math.sqrt((rx - px) ** 2 + (ry - py) ** 2)
        dist = min(dist_center, dist_pin)

    if dist <= constraint.max_distance_mm:
        return None

    pin_str = f":{constraint.target_pin}" if constraint.target_pin else ""
    severity = "critical" if dist > constraint.max_distance_mm * 2 else "major"
    return ConstraintViolation(
        constraint_type="proximity",
        severity=severity,
        refs=(constraint.ref, constraint.target_ref),
        message=(
            f"{constraint.ref} is {dist:.1f}mm from "
            f"{constraint.target_ref}{pin_str} "
            f"(max {constraint.max_distance_mm:.1f}mm)"
        ),
        measured_value=dist,
        threshold=constraint.max_distance_mm,
        suggested_fix=(
            f"Move {constraint.ref} within "
            f"{constraint.max_distance_mm:.1f}mm of "
            f"{constraint.target_ref}{pin_str}"
        ),
    )


def _check_ordering(
    chain: OrderingChain,
    positions: dict[str, tuple[float, float, float]],
) -> list[ConstraintViolation]:
    """Check that components in a chain follow the declared order.

    Uses X-coordinate as the primary axis (left-to-right flow).
    Falls back to Y-coordinate if X spread is less than Y spread.
    """
    violations: list[ConstraintViolation] = []
    present_refs = [r for r in chain.refs if r in positions]
    if len(present_refs) < 2:
        return violations

    # Determine primary axis (X or Y) based on spread
    xs = [positions[r][0] for r in present_refs]
    ys = [positions[r][1] for r in present_refs]
    x_spread = max(xs) - min(xs)
    y_spread = max(ys) - min(ys)
    use_x = x_spread >= y_spread

    for i in range(len(present_refs) - 1):
        ref_a = present_refs[i]
        ref_b = present_refs[i + 1]
        if use_x:
            val_a = positions[ref_a][0]
            val_b = positions[ref_b][0]
        else:
            val_a = positions[ref_a][1]
            val_b = positions[ref_b][1]

        if val_a > val_b + 1.0:  # 1mm tolerance
            axis = "X" if use_x else "Y"
            violations.append(ConstraintViolation(
                constraint_type="ordering",
                severity="major",
                refs=(ref_a, ref_b),
                message=(
                    f"{ref_a} ({axis}={val_a:.1f}) should be before "
                    f"{ref_b} ({axis}={val_b:.1f}) in group '{chain.group}'"
                ),
                measured_value=val_a - val_b,
                threshold=0.0,
                suggested_fix=(
                    f"Swap {ref_a} and {ref_b} positions, or reorder "
                    f"the chain in group '{chain.group}'"
                ),
            ))
    return violations


def _check_group_cohesion(
    group_name: str,
    refs: tuple[str, ...],
    positions: dict[str, tuple[float, float, float]],
    max_spread_mm: float = 40.0,
) -> ConstraintViolation | None:
    """Check that all members of a group are co-located."""
    present = [r for r in refs if r in positions]
    if len(present) < 2:
        return None

    xs = [positions[r][0] for r in present]
    ys = [positions[r][1] for r in present]
    spread = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)

    if spread <= max_spread_mm:
        return None

    farthest = max(present, key=lambda r: math.sqrt(
        (positions[r][0] - sum(xs) / len(xs)) ** 2
        + (positions[r][1] - sum(ys) / len(ys)) ** 2
    ))
    return ConstraintViolation(
        constraint_type="group",
        severity="major" if spread > max_spread_mm * 1.5 else "minor",
        refs=tuple(present),
        message=(
            f"Group '{group_name}' spread is {spread:.1f}mm "
            f"(max {max_spread_mm:.1f}mm). "
            f"Farthest: {farthest}"
        ),
        measured_value=spread,
        threshold=max_spread_mm,
        suggested_fix=f"Move {farthest} closer to the group center",
    )


def audit_placement_constraints(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[ConstraintViolation, ...]:
    """Audit all placement constraints and return violations.

    This is the main entry point. It resolves constraints from the
    requirements (explicit properties + subnet topology) and checks
    each against the current PCB positions.

    Returns:
        Tuple of violations sorted by severity (critical first).
    """
    from kicad_pipeline.optimization.constraint_resolver import resolve_constraints

    constraints = resolve_constraints(requirements)
    positions = _positions_from_pcb(pcb)
    violations: list[ConstraintViolation] = []

    # Check proximity constraints
    for prox in constraints.proximity:
        v = _check_proximity(prox, positions, pcb=pcb, requirements=requirements)
        if v is not None:
            violations.append(v)

    # Check ordering chains
    for chain in constraints.ordering:
        violations.extend(_check_ordering(chain, positions))

    # Check group cohesion
    for group_name, refs in constraints.groups:
        v = _check_group_cohesion(group_name, refs, positions)
        if v is not None:
            violations.append(v)

    # Sort by severity: critical > major > minor
    severity_order = {"critical": 0, "major": 1, "minor": 2}
    violations.sort(key=lambda v: severity_order.get(v.severity, 3))

    if violations:
        _log.info(
            "Constraint audit: %d violations (%d critical, %d major, %d minor)",
            len(violations),
            sum(1 for v in violations if v.severity == "critical"),
            sum(1 for v in violations if v.severity == "major"),
            sum(1 for v in violations if v.severity == "minor"),
        )
    else:
        _log.info("Constraint audit: all constraints satisfied")

    return tuple(violations)
