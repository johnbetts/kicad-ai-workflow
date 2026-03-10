"""EE review agent: rule-based placement critique with suggested fixes.

Analyses a PCB placement against EE best practices and produces
coordinate-specific violations with suggested positions for each
offending component.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    DECOUPLING_CAP_MAX_DISTANCE_MM,
)
from kicad_pipeline.optimization.functional_grouper import (
    DetectedSubCircuit,
    SubCircuitType,
    VoltageDomain,
    classify_voltage_domains,
    detect_subcircuits,
)
from kicad_pipeline.pcb.constraints import (
    check_courtyard_collisions,
)

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBCIRCUIT_MAX_SPREAD_MM = 15.0
VOLTAGE_DOMAIN_MIN_GAP_MM = 2.0
CONNECTOR_EDGE_MAX_MM = 5.0
CRYSTAL_MAX_DISTANCE_MM = 5.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PlacementRule(enum.Enum):
    """Categories of placement rules."""

    DECOUPLING_DISTANCE = "decoupling_distance"
    SUBCIRCUIT_SPREAD = "subcircuit_spread"
    VOLTAGE_ISOLATION = "voltage_isolation"
    CONNECTOR_EDGE = "connector_edge"
    COLLISION = "collision"
    THERMAL_ADJACENCY = "thermal_adjacency"
    CRYSTAL_PROXIMITY = "crystal_proximity"
    SIGNAL_FLOW = "signal_flow"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacementViolation:
    """A single placement rule violation with suggested fix."""

    rule: PlacementRule
    severity: str  # "critical" | "major" | "minor"
    refs: tuple[str, ...]
    message: str
    current_value: float
    threshold: float
    suggested_position: tuple[float, float] | None


@dataclass(frozen=True)
class PlacementReview:
    """Complete placement review result."""

    violations: tuple[PlacementViolation, ...]
    grade: str  # A/B/C/D/F
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _edge_dist(
    p1: tuple[float, float], s1: tuple[float, float],
    p2: tuple[float, float], s2: tuple[float, float],
) -> float:
    """Edge-to-edge distance between two rectangular footprints.

    Args:
        p1, p2: Center positions (x, y).
        s1, s2: Sizes (width, height).
    Returns:
        Minimum gap between bounding boxes (0.0 if overlapping).
    """
    dx = abs(p1[0] - p2[0]) - (s1[0] + s2[0]) / 2.0
    dy = abs(p1[1] - p2[1]) - (s1[1] + s2[1]) / 2.0
    if dx <= 0 and dy <= 0:
        return 0.0  # overlapping
    if dx <= 0:
        return dy
    if dy <= 0:
        return dx
    return math.sqrt(dx * dx + dy * dy)


def _fp_positions(pcb: PCBDesign) -> dict[str, tuple[float, float]]:
    """Extract footprint ref → (x, y) position map."""
    return {fp.ref: (fp.position.x, fp.position.y) for fp in pcb.footprints}


def _fp_size_dict(pcb: PCBDesign) -> dict[str, tuple[float, float]]:
    """Extract footprint ref → (width, height) map from pad extents."""
    result: dict[str, tuple[float, float]] = {}
    for fp in pcb.footprints:
        if not fp.pads:
            result[fp.ref] = (3.0, 3.0)
            continue
        xs = [p.position.x - p.size_x / 2 for p in fp.pads] + \
             [p.position.x + p.size_x / 2 for p in fp.pads]
        ys = [p.position.y - p.size_y / 2 for p in fp.pads] + \
             [p.position.y + p.size_y / 2 for p in fp.pads]
        w = max(xs) - min(xs) + 1.0
        h = max(ys) - min(ys) + 1.0
        result[fp.ref] = (w, h)
    return result


def _board_bounds(pcb: PCBDesign) -> tuple[float, float, float, float]:
    """Get board outline bounds: (min_x, min_y, max_x, max_y)."""
    if not pcb.outline or not pcb.outline.polygon:
        return (0.0, 0.0, 100.0, 80.0)
    pts = pcb.outline.polygon
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _ref_prefix(ref: str) -> str:
    """Get alpha prefix of reference designator."""
    return "".join(c for c in ref if c.isalpha())


def _point_toward(
    source: tuple[float, float],
    target: tuple[float, float],
    distance: float,
) -> tuple[float, float]:
    """Return a point *distance* mm from *source* toward *target*."""
    d = _dist(source, target)
    if d < 0.01:
        return (source[0] + distance, source[1])
    ratio = distance / d
    return (
        source[0] + (target[0] - source[0]) * ratio,
        source[1] + (target[1] - source[1]) * ratio,
    )


# ---------------------------------------------------------------------------
# Rule checks
# ---------------------------------------------------------------------------


def _check_decoupling_distance(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    subcircuits: tuple[DetectedSubCircuit, ...],
) -> list[PlacementViolation]:
    """Check that decoupling caps are within threshold of their IC.

    Uses edge-to-edge distance (gap between bounding boxes), not
    center-to-center, so large ICs (e.g. ESP32 26x16mm) aren't
    penalized when caps are right at their body edge.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    sizes = _fp_size_dict(pcb)
    threshold = DECOUPLING_CAP_MAX_DISTANCE_MM

    # Use decoupling subcircuits for IC-cap pairs
    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.DECOUPLING:
            continue
        ic_ref = sc.anchor_ref
        cap_refs = [r for r in sc.refs if r != ic_ref]
        if ic_ref not in positions:
            continue
        ic_pos = positions[ic_ref]
        ic_size = sizes.get(ic_ref, (3.0, 3.0))

        for cap_ref in cap_refs:
            if cap_ref not in positions:
                continue
            cap_pos = positions[cap_ref]
            cap_size = sizes.get(cap_ref, (2.0, 1.0))
            d = _edge_dist(ic_pos, ic_size, cap_pos, cap_size)
            if d > threshold:
                # Suggest moving cap close to IC edge
                suggested = _point_toward(ic_pos, cap_pos, threshold * 0.8)
                violations.append(PlacementViolation(
                    rule=PlacementRule.DECOUPLING_DISTANCE,
                    severity="critical" if d > threshold * 3 else "major",
                    refs=(cap_ref, ic_ref),
                    message=f"{cap_ref} is {d:.1f}mm from {ic_ref} edge "
                            f"(max {threshold}mm for decoupling)",
                    current_value=d,
                    threshold=threshold,
                    suggested_position=suggested,
                ))

    return violations


def _check_subcircuit_spread(
    pcb: PCBDesign,
    subcircuits: tuple[DetectedSubCircuit, ...],
) -> list[PlacementViolation]:
    """Check that sub-circuit components are clustered near anchor."""
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    threshold = SUBCIRCUIT_MAX_SPREAD_MM

    for sc in subcircuits:
        if sc.circuit_type == SubCircuitType.DECOUPLING:
            continue  # Handled by decoupling check
        anchor_pos = positions.get(sc.anchor_ref)
        if anchor_pos is None:
            continue

        for ref in sc.refs:
            if ref == sc.anchor_ref:
                continue
            pos = positions.get(ref)
            if pos is None:
                continue
            d = _dist(anchor_pos, pos)
            if d > threshold:
                suggested = _point_toward(anchor_pos, pos, threshold * 0.8)
                violations.append(PlacementViolation(
                    rule=PlacementRule.SUBCIRCUIT_SPREAD,
                    severity="major",
                    refs=(ref, sc.anchor_ref),
                    message=f"{ref} is {d:.1f}mm from anchor {sc.anchor_ref} "
                            f"in {sc.circuit_type.value} (max {threshold}mm)",
                    current_value=d,
                    threshold=threshold,
                    suggested_position=suggested,
                ))

    return violations


def _check_voltage_isolation(
    pcb: PCBDesign,
    domain_map: dict[str, VoltageDomain],
) -> list[PlacementViolation]:
    """Check minimum gap between different voltage domain components."""
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    threshold = VOLTAGE_DOMAIN_MIN_GAP_MM

    # Group refs by domain (skip MIXED)
    domain_refs: dict[VoltageDomain, list[str]] = {}
    for ref, domain in domain_map.items():
        if domain == VoltageDomain.MIXED:
            continue
        domain_refs.setdefault(domain, []).append(ref)

    # Check inter-domain distances
    domains = list(domain_refs.keys())
    for i, d1 in enumerate(domains):
        for d2 in domains[i + 1:]:
            # Only check domains that should be isolated
            if d1 == d2:
                continue
            for r1 in domain_refs[d1]:
                p1 = positions.get(r1)
                if p1 is None:
                    continue
                for r2 in domain_refs[d2]:
                    p2 = positions.get(r2)
                    if p2 is None:
                        continue
                    d = _dist(p1, p2)
                    if d < threshold:
                        violations.append(PlacementViolation(
                            rule=PlacementRule.VOLTAGE_ISOLATION,
                            severity="major",
                            refs=(r1, r2),
                            message=f"{r1} ({d1.value}) and {r2} ({d2.value}) "
                                    f"only {d:.1f}mm apart (min {threshold}mm)",
                            current_value=d,
                            threshold=threshold,
                            suggested_position=None,
                        ))

    return violations


def _check_connector_edge(
    pcb: PCBDesign,
) -> list[PlacementViolation]:
    """Check that connectors (J*) are within threshold of board edge."""
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    bounds = _board_bounds(pcb)
    min_x, min_y, max_x, max_y = bounds
    threshold = CONNECTOR_EDGE_MAX_MM

    for fp in pcb.footprints:
        if _ref_prefix(fp.ref) != "J":
            continue
        pos = positions.get(fp.ref)
        if pos is None:
            continue
        x, y = pos
        # Distance to nearest edge
        edge_dist = min(
            x - min_x,
            max_x - x,
            y - min_y,
            max_y - y,
        )
        if edge_dist > threshold:
            # Suggest moving to nearest edge
            nearest_edge_x = min_x if (x - min_x) < (max_x - x) else max_x
            nearest_edge_y = min_y if (y - min_y) < (max_y - y) else max_y
            # Choose closest edge direction
            dx = abs(x - nearest_edge_x)
            dy = abs(y - nearest_edge_y)
            if dx < dy:
                # Move to left/right edge
                suggested = (nearest_edge_x + (2.0 if nearest_edge_x == min_x else -2.0), y)
            else:
                suggested = (x, nearest_edge_y + (2.0 if nearest_edge_y == min_y else -2.0))

            violations.append(PlacementViolation(
                rule=PlacementRule.CONNECTOR_EDGE,
                severity="major",
                refs=(fp.ref,),
                message=f"{fp.ref} is {edge_dist:.1f}mm from nearest board edge "
                        f"(max {threshold}mm for connectors)",
                current_value=edge_dist,
                threshold=threshold,
                suggested_position=suggested,
            ))

    return violations


def _check_collisions(
    pcb: PCBDesign,
) -> list[PlacementViolation]:
    """Check for courtyard collisions between components."""
    from kicad_pipeline.models.pcb import Point as PcbPoint

    violations: list[PlacementViolation] = []
    positions = {fp.ref: PcbPoint(fp.position.x, fp.position.y) for fp in pcb.footprints}
    fp_sizes = _fp_size_dict(pcb)

    collision_strs = check_courtyard_collisions(positions, fp_sizes)
    for msg in collision_strs:
        # Parse refs from collision message (format: "REF1 vs REF2: ...")
        parts = msg.split(":")
        refs_part = parts[0] if parts else msg
        refs = tuple(r.strip() for r in refs_part.replace(" vs ", ",").split(",")
                     if r.strip())
        violations.append(PlacementViolation(
            rule=PlacementRule.COLLISION,
            severity="critical",
            refs=refs,
            message=msg,
            current_value=0.0,
            threshold=0.0,
            suggested_position=None,
        ))

    return violations


def _check_crystal_proximity(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[PlacementViolation]:
    """Check that crystals are within threshold of their MCU."""
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    threshold = CRYSTAL_MAX_DISTANCE_MM

    crystals = [fp for fp in pcb.footprints if _ref_prefix(fp.ref) == "Y"]
    mcus = [fp for fp in pcb.footprints if _ref_prefix(fp.ref) == "U"]

    if not crystals or not mcus:
        return violations

    # Find MCU with clock pins connected to crystal
    crystal_nets: set[str] = set()
    for net in requirements.nets:
        for conn in net.connections:
            if _ref_prefix(conn.ref) == "Y":
                crystal_nets.add(net.name)

    # Find which MCU connects to crystal nets
    mcu_ref: str | None = None
    for net in requirements.nets:
        if net.name not in crystal_nets:
            continue
        for conn in net.connections:
            if conn.ref.startswith("U"):
                mcu_ref = conn.ref
                break
        if mcu_ref:
            break

    if not mcu_ref or mcu_ref not in positions:
        return violations

    mcu_pos = positions[mcu_ref]
    for crystal in crystals:
        if crystal.ref not in positions:
            continue
        crystal_pos = positions[crystal.ref]
        d = _dist(mcu_pos, crystal_pos)
        if d > threshold:
            suggested = _point_toward(mcu_pos, crystal_pos, threshold * 0.8)
            violations.append(PlacementViolation(
                rule=PlacementRule.CRYSTAL_PROXIMITY,
                severity="major",
                refs=(crystal.ref, mcu_ref),
                message=f"{crystal.ref} is {d:.1f}mm from MCU {mcu_ref} "
                        f"(max {threshold}mm for crystals)",
                current_value=d,
                threshold=threshold,
                suggested_position=suggested,
            ))

    return violations


def _check_thermal_adjacency(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[PlacementViolation]:
    """Check high-power components aren't adjacent to sensitive parts."""
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)

    power_keywords = {"RELAY", "REGULATOR", "BUCK", "LDO", "MOSFET", "FET"}
    sensitive_keywords = {"ADC", "DAC", "VREF", "CRYSTAL", "SENSOR", "OPAMP"}

    # Identify power and sensitive components
    power_refs: list[str] = []
    sensitive_refs: list[str] = []

    for comp in requirements.components:
        val_desc = f"{comp.value} {comp.description or ''}".upper()
        if _ref_prefix(comp.ref) == "K" or any(kw in val_desc for kw in power_keywords):
            power_refs.append(comp.ref)
        if any(kw in val_desc for kw in sensitive_keywords):
            sensitive_refs.append(comp.ref)

    threshold = 5.0  # minimum mm between power and sensitive
    for pr in power_refs:
        p1 = positions.get(pr)
        if p1 is None:
            continue
        for sr in sensitive_refs:
            p2 = positions.get(sr)
            if p2 is None:
                continue
            d = _dist(p1, p2)
            if d < threshold:
                violations.append(PlacementViolation(
                    rule=PlacementRule.THERMAL_ADJACENCY,
                    severity="minor",
                    refs=(pr, sr),
                    message=f"Power component {pr} is {d:.1f}mm from "
                            f"sensitive {sr} (min {threshold}mm recommended)",
                    current_value=d,
                    threshold=threshold,
                    suggested_position=None,
                ))

    return violations


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def _compute_grade(violations: tuple[PlacementViolation, ...]) -> str:
    """Compute letter grade from violations."""
    critical = sum(1 for v in violations if v.severity == "critical")
    major = sum(1 for v in violations if v.severity == "major")
    minor = sum(1 for v in violations if v.severity == "minor")

    if critical == 0 and major == 0 and minor <= 2:
        return "A"
    if critical == 0 and major <= 3:
        return "B"
    if critical <= 2 and major <= 10:
        return "C"
    if critical <= 5:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def review_placement(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    subcircuits: tuple[DetectedSubCircuit, ...] | None = None,
    domain_map: dict[str, VoltageDomain] | None = None,
) -> PlacementReview:
    """Review a PCB placement against EE best practices.

    Runs all placement rules and returns a graded review with
    coordinate-specific violations and suggested fixes.

    Args:
        pcb: The PCB design to review.
        requirements: Project requirements.
        subcircuits: Pre-detected sub-circuits (will be detected if None).
        domain_map: Pre-classified voltage domains (will be classified if None).

    Returns:
        PlacementReview with violations, grade, and summary.
    """
    if subcircuits is None:
        subcircuits = detect_subcircuits(requirements)
    if domain_map is None:
        domain_map = classify_voltage_domains(requirements)

    all_violations: list[PlacementViolation] = []

    # Run all checks
    all_violations.extend(
        _check_decoupling_distance(pcb, requirements, subcircuits)
    )
    all_violations.extend(
        _check_subcircuit_spread(pcb, subcircuits)
    )
    all_violations.extend(
        _check_voltage_isolation(pcb, domain_map)
    )
    all_violations.extend(
        _check_connector_edge(pcb)
    )
    all_violations.extend(
        _check_collisions(pcb)
    )
    all_violations.extend(
        _check_crystal_proximity(pcb, requirements)
    )
    all_violations.extend(
        _check_thermal_adjacency(pcb, requirements)
    )

    violations = tuple(all_violations)
    grade = _compute_grade(violations)

    critical = sum(1 for v in violations if v.severity == "critical")
    major = sum(1 for v in violations if v.severity == "major")
    minor = sum(1 for v in violations if v.severity == "minor")
    summary = (
        f"Grade {grade}: {len(violations)} violations "
        f"({critical} critical, {major} major, {minor} minor)"
    )

    _log.info("Placement review: %s", summary)

    return PlacementReview(
        violations=violations,
        grade=grade,
        summary=summary,
    )
