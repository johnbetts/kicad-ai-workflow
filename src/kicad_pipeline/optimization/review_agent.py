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
    CONNECTOR_EDGE_MAX_MM,
    CONNECTOR_FUNCTIONAL_PROXIMITY_MAX_MM,
    DECOUPLING_CAP_MAX_DISTANCE_MM,
    MCU_PERIPHERAL_MAX_DISTANCE_MM,
    REGULATOR_BOUNDARY_TOLERANCE_MM,
    RF_EDGE_MAX_MM,
)
from kicad_pipeline.optimization.functional_grouper import (
    DetectedSubCircuit,
    DomainAffinity,
    SubCircuitType,
    VoltageDomain,
    classify_voltage_domains,
    detect_cross_domain_affinities,
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

SUBCIRCUIT_MAX_SPREAD_MM = 20.0
VOLTAGE_DOMAIN_MIN_GAP_MM = 2.0
CRYSTAL_MAX_DISTANCE_MM = 10.0


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
    MCU_PERIPHERAL_PROXIMITY = "mcu_peripheral_proximity"
    RF_EDGE_PLACEMENT = "rf_edge_placement"
    CONNECTOR_ORIENTATION = "connector_orientation"
    REGULATOR_BOUNDARY = "regulator_boundary"
    CONNECTOR_FUNCTIONAL_PROXIMITY = "connector_functional_proximity"
    BOARD_EDGE_CLEARANCE = "board_edge_clearance"
    DIODE_ORIENTATION_CONSISTENCY = "diode_orientation_consistency"


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
    """Extract footprint ref → (x, y) centroid position map.

    Converts KiCad origin-based positions to pad-centroid positions
    for accurate distance measurements.  Uses pin_map.origin_to_centroid()
    as the single source of truth for origin ↔ centroid conversion.
    """
    from kicad_pipeline.pcb.pin_map import origin_to_centroid

    return {
        fp.ref: origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        for fp in pcb.footprints
    }


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
    affinities: tuple[DomainAffinity, ...] = (),
) -> list[PlacementViolation]:
    """Check minimum gap between different voltage domain components.

    Components linked by cross-domain affinities (e.g. analog monitoring
    circuits measuring relay outputs) are exempted from violations.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    threshold = VOLTAGE_DOMAIN_MIN_GAP_MM

    # Build exemption set from cross-domain affinities
    exempt_pairs: set[tuple[str, str]] = set()
    for aff in affinities:
        for sr in aff.source_refs:
            for tr in aff.target_refs:
                pair = (min(sr, tr), max(sr, tr))
                exempt_pairs.add(pair)

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
                    # Skip exempt pairs (cross-domain affinities)
                    pair = (min(r1, r2), max(r1, r2))
                    if pair in exempt_pairs:
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
    """Check for courtyard collisions between components.

    Provides suggested_position for the smaller component: pushes it
    away from the larger one in the clearance direction.
    """
    from kicad_pipeline.models.pcb import Point as PcbPoint

    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    # Use centroid positions for collision detection too
    pcb_positions = {ref: PcbPoint(x, y) for ref, (x, y) in positions.items()}
    fp_sizes = _fp_size_dict(pcb)
    fp_rotations = {fp.ref: fp.rotation for fp in pcb.footprints}

    collision_strs = check_courtyard_collisions(
        pcb_positions, fp_sizes, rotations=fp_rotations,
    )
    for msg in collision_strs:
        # Parse refs from collision message
        # Format: "Courtyard collision: REF1 and REF2"
        import re as _re
        ref_match = _re.findall(r"\b([A-Z]+\d+)\b", msg)
        refs = tuple(dict.fromkeys(ref_match))  # deduplicate, preserve order

        # Compute suggested position: push smaller component away
        suggested: tuple[float, float] | None = None
        if len(refs) >= 2 and refs[0] in positions and refs[1] in positions:
            r1, r2 = refs[0], refs[1]
            s1 = fp_sizes.get(r1, (2.0, 2.0))
            s2 = fp_sizes.get(r2, (2.0, 2.0))

            # Move the smaller component
            if s1[0] * s1[1] <= s2[0] * s2[1]:
                to_move, anchor = r1, r2
                ms, as_ = s1, s2
            else:
                to_move, anchor = r2, r1
                ms, as_ = s2, s1

            mx, my = positions[to_move]
            ax, ay = positions[anchor]
            dx = mx - ax
            dy = my - ay
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.01:
                dx, dy, dist = 1.0, 0.0, 1.0
            # Push away: enough to clear half-widths + gap
            needed = (ms[0] + as_[0]) / 2.0 + 0.5
            scale = needed / dist
            suggested = (ax + dx * scale, ay + dy * scale)

        violations.append(PlacementViolation(
            rule=PlacementRule.COLLISION,
            severity="critical",
            refs=refs,
            message=msg,
            current_value=0.0,
            threshold=0.0,
            suggested_position=suggested,
        ))

    return violations


def _check_crystal_proximity(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[PlacementViolation]:
    """Check that crystals are within threshold of their connected IC.

    Traces nets from each crystal to find the specific IC it serves
    (MCU, W5500, LAN8720A, etc.), not just the MCU.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    threshold = CRYSTAL_MAX_DISTANCE_MM

    crystals = [fp for fp in pcb.footprints if _ref_prefix(fp.ref) == "Y"]
    if not crystals:
        return violations

    # Build crystal→IC map via net connectivity
    # Each crystal connects to a specific IC via its oscillator pins
    crystal_to_ic: dict[str, str] = {}
    for crystal in crystals:
        crystal_nets: set[str] = set()
        for net in requirements.nets:
            for conn in net.connections:
                if conn.ref == crystal.ref:
                    crystal_nets.add(net.name)
        # Find connected IC on crystal's non-GND nets
        for net in requirements.nets:
            if net.name not in crystal_nets:
                continue
            _nl = net.name.upper()
            if _nl in ("GND", "AGND", "DGND", "PGND", "VSS", "AVSS"):
                continue
            for conn in net.connections:
                if conn.ref.startswith("U") and conn.ref in positions:
                    crystal_to_ic[crystal.ref] = conn.ref
                    break
            if crystal.ref in crystal_to_ic:
                break

    for crystal in crystals:
        if crystal.ref not in positions:
            continue
        ic_ref = crystal_to_ic.get(crystal.ref)
        if not ic_ref or ic_ref not in positions:
            continue
        crystal_pos = positions[crystal.ref]
        ic_pos = positions[ic_ref]
        d = _dist(ic_pos, crystal_pos)
        if d > threshold:
            suggested = _point_toward(ic_pos, crystal_pos, threshold * 0.8)
            violations.append(PlacementViolation(
                rule=PlacementRule.CRYSTAL_PROXIMITY,
                severity="major",
                refs=(crystal.ref, ic_ref),
                message=f"{crystal.ref} is {d:.1f}mm from IC {ic_ref} "
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


def _check_mcu_peripheral_proximity(
    pcb: PCBDesign,
    subcircuits: tuple[DetectedSubCircuit, ...],
) -> list[PlacementViolation]:
    """Check that MCU peripherals are within threshold of their MCU."""
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    threshold = MCU_PERIPHERAL_MAX_DISTANCE_MM

    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.MCU_PERIPHERAL_CLUSTER:
            continue
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
                    rule=PlacementRule.MCU_PERIPHERAL_PROXIMITY,
                    severity="major",
                    refs=(ref, sc.anchor_ref),
                    message=f"{ref} is {d:.1f}mm from MCU {sc.anchor_ref} "
                            f"(max {threshold}mm for peripherals)",
                    current_value=d,
                    threshold=threshold,
                    suggested_position=suggested,
                ))

    return violations


def _check_rf_edge_placement(
    pcb: PCBDesign,
    subcircuits: tuple[DetectedSubCircuit, ...],
) -> list[PlacementViolation]:
    """Check that RF antenna modules are on board edge."""
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    bounds = _board_bounds(pcb)
    min_x, min_y, max_x, max_y = bounds
    threshold = RF_EDGE_MAX_MM

    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.RF_ANTENNA:
            continue
        pos = positions.get(sc.anchor_ref)
        if pos is None:
            continue
        x, y = pos
        edge_dist = min(x - min_x, max_x - x, y - min_y, max_y - y)
        if edge_dist > threshold:
            # Suggest nearest edge
            nearest_x = min_x + 2.0 if (x - min_x) < (max_x - x) else max_x - 2.0
            nearest_y = min_y + 2.0 if (y - min_y) < (max_y - y) else max_y - 2.0
            if abs(x - nearest_x) < abs(y - nearest_y):
                suggested = (nearest_x, y)
            else:
                suggested = (x, nearest_y)

            violations.append(PlacementViolation(
                rule=PlacementRule.RF_EDGE_PLACEMENT,
                severity="critical",
                refs=(sc.anchor_ref,),
                message=f"RF module {sc.anchor_ref} is {edge_dist:.1f}mm from "
                        f"nearest edge (max {threshold}mm for antenna)",
                current_value=edge_dist,
                threshold=threshold,
                suggested_position=suggested,
            ))

    return violations


def _check_connector_functional_proximity(
    pcb: PCBDesign,
    subcircuits: tuple[DetectedSubCircuit, ...],
) -> list[PlacementViolation]:
    """Check that connectors are near their functional group.

    Flags connectors that are on a different edge than the centroid of
    their functional subcircuit group.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    threshold = CONNECTOR_FUNCTIONAL_PROXIMITY_MAX_MM

    # Map connector refs to their subcircuit group centroid
    for sc in subcircuits:
        connector_refs = [r for r in sc.refs if _ref_prefix(r) == "J"]
        if not connector_refs:
            continue
        # Compute centroid of non-connector members
        non_conn = [r for r in sc.refs if _ref_prefix(r) != "J" and r in positions]
        if not non_conn:
            continue
        cx = sum(positions[r][0] for r in non_conn) / len(non_conn)
        cy = sum(positions[r][1] for r in non_conn) / len(non_conn)

        for ref in connector_refs:
            pos = positions.get(ref)
            if pos is None:
                continue
            d = _dist(pos, (cx, cy))
            if d > threshold:
                suggested = _point_toward((cx, cy), pos, threshold * 0.8)
                violations.append(PlacementViolation(
                    rule=PlacementRule.CONNECTOR_FUNCTIONAL_PROXIMITY,
                    severity="major",
                    refs=(ref,),
                    message=f"{ref} is {d:.1f}mm from its functional group "
                            f"({sc.circuit_type.value}) centroid "
                            f"(max {threshold}mm)",
                    current_value=d,
                    threshold=threshold,
                    suggested_position=suggested,
                ))

    return violations


def _check_regulator_boundary(
    pcb: PCBDesign,
    subcircuits: tuple[DetectedSubCircuit, ...],
    domain_map: dict[str, VoltageDomain],
) -> list[PlacementViolation]:
    """Check that regulators are placed at domain boundaries."""
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    threshold = REGULATOR_BOUNDARY_TOLERANCE_MM

    # Compute domain centroids
    domain_positions: dict[VoltageDomain, list[tuple[float, float]]] = {}
    for ref, ref_pos in positions.items():
        dom = domain_map.get(ref)
        if dom is not None and dom != VoltageDomain.MIXED:
            domain_positions.setdefault(dom, []).append(ref_pos)

    domain_centroids: dict[VoltageDomain, tuple[float, float]] = {}
    for dom, pts in domain_positions.items():
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        domain_centroids[dom] = (cx, cy)

    for sc in subcircuits:
        if sc.circuit_type not in (
            SubCircuitType.BUCK_CONVERTER,
            SubCircuitType.LDO_REGULATOR,
        ):
            continue
        if sc.input_domain is None or sc.output_domain is None:
            continue
        sc_pos = positions.get(sc.anchor_ref)
        if sc_pos is None:
            continue

        in_c = domain_centroids.get(sc.input_domain)
        out_c = domain_centroids.get(sc.output_domain)
        if in_c is None or out_c is None:
            continue

        # Ideal position: midpoint between domain centroids
        mid_x = (in_c[0] + out_c[0]) / 2.0
        mid_y = (in_c[1] + out_c[1]) / 2.0
        dist_to_mid = _dist(sc_pos, (mid_x, mid_y))

        if dist_to_mid > threshold:
            violations.append(PlacementViolation(
                rule=PlacementRule.REGULATOR_BOUNDARY,
                severity="major",
                refs=(sc.anchor_ref,),
                message=f"Regulator {sc.anchor_ref} is {dist_to_mid:.1f}mm from "
                        f"domain boundary (tolerance {threshold}mm)",
                current_value=dist_to_mid,
                threshold=threshold,
                suggested_position=(mid_x, mid_y),
            ))

    return violations


def _check_board_edge_clearance(
    pcb: PCBDesign,
) -> list[PlacementViolation]:
    """Check that all components have ≥1mm clearance from board edges.

    Uses rotation-aware bounding box to detect components whose body
    extends within the warning threshold (1mm) or critical threshold (0.3mm).
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    fp_sizes = _fp_size_dict(pcb)

    bx1, by1, bx2, by2 = _board_bounds(pcb)
    warn_margin = 1.0
    crit_margin = 0.3

    for fp in pcb.footprints:
        ref = fp.ref
        if ref not in positions:
            continue
        x, y = positions[ref]
        w, h = fp_sizes.get(ref, (2.0, 2.0))
        rot = 0.0
        for fpp in pcb.footprints:
            if fpp.ref == ref:
                rot = fpp.rotation
                break
        if rot % 180 in (90.0, 270.0):
            w, h = h, w

        # Check each edge
        left_gap = (x - w / 2.0) - bx1
        right_gap = bx2 - (x + w / 2.0)
        top_gap = (y - h / 2.0) - by1
        bottom_gap = by2 - (y + h / 2.0)
        min_gap = min(left_gap, right_gap, top_gap, bottom_gap)

        if min_gap < crit_margin:
            # Suggest moving inward
            sx, sy = x, y
            if left_gap == min_gap:
                sx = bx1 + w / 2.0 + warn_margin
            elif right_gap == min_gap:
                sx = bx2 - w / 2.0 - warn_margin
            elif top_gap == min_gap:
                sy = by1 + h / 2.0 + warn_margin
            else:
                sy = by2 - h / 2.0 - warn_margin
            violations.append(PlacementViolation(
                rule=PlacementRule.BOARD_EDGE_CLEARANCE,
                severity="critical",
                refs=(ref,),
                message=f"{ref} is {min_gap:.1f}mm from board edge "
                        f"(min {crit_margin}mm)",
                current_value=min_gap,
                threshold=crit_margin,
                suggested_position=(sx, sy),
            ))
        elif min_gap < warn_margin:
            violations.append(PlacementViolation(
                rule=PlacementRule.BOARD_EDGE_CLEARANCE,
                severity="minor",
                refs=(ref,),
                message=f"{ref} is {min_gap:.1f}mm from board edge "
                        f"(recommended ≥{warn_margin}mm)",
                current_value=min_gap,
                threshold=warn_margin,
                suggested_position=None,
            ))

    return violations


def _check_diode_orientation_consistency(
    pcb: PCBDesign,
    subcircuits: tuple[DetectedSubCircuit, ...],
) -> list[PlacementViolation]:
    """Check that diodes in repeating subcircuits have consistent orientation.

    For subcircuit types that repeat (e.g., ADC channels), all diodes within
    instances should have the same rotation to maintain consistent cathode
    direction for manufacturing clarity.
    """
    violations: list[PlacementViolation] = []

    # Group subcircuits by type
    sc_by_type: dict[str, list[DetectedSubCircuit]] = {}
    for sc in subcircuits:
        sc_by_type.setdefault(sc.circuit_type.value, []).append(sc)

    # Check types with multiple instances
    for sc_type, instances in sc_by_type.items():
        if len(instances) < 2:
            continue

        # Collect diode rotations per instance
        diode_rots: dict[str, float] = {}
        for sc in instances:
            for ref in sc.refs:
                if ref.startswith("D"):
                    for fp in pcb.footprints:
                        if fp.ref == ref:
                            diode_rots[ref] = fp.rotation
                            break

        if len(diode_rots) < 2:
            continue

        # Check consistency — all should have same rotation
        rot_values = list(diode_rots.values())
        majority_rot = max(set(rot_values), key=rot_values.count)
        for ref, rot in diode_rots.items():
            if rot != majority_rot:
                violations.append(PlacementViolation(
                    rule=PlacementRule.DIODE_ORIENTATION_CONSISTENCY,
                    severity="minor",
                    refs=(ref,),
                    message=f"{ref} rotation {rot}° differs from majority "
                            f"{majority_rot}° in {sc_type} subcircuits",
                    current_value=rot,
                    threshold=majority_rot,
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

    # Detect cross-domain affinities for voltage isolation exemptions
    affinities = detect_cross_domain_affinities(requirements, domain_map)

    all_violations: list[PlacementViolation] = []

    # Run all checks
    all_violations.extend(
        _check_decoupling_distance(pcb, requirements, subcircuits)
    )
    all_violations.extend(
        _check_subcircuit_spread(pcb, subcircuits)
    )
    all_violations.extend(
        _check_voltage_isolation(pcb, domain_map, affinities)
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
    all_violations.extend(
        _check_mcu_peripheral_proximity(pcb, subcircuits)
    )
    all_violations.extend(
        _check_rf_edge_placement(pcb, subcircuits)
    )
    all_violations.extend(
        _check_regulator_boundary(pcb, subcircuits, domain_map)
    )
    all_violations.extend(
        _check_connector_functional_proximity(pcb, subcircuits)
    )
    all_violations.extend(
        _check_board_edge_clearance(pcb)
    )
    all_violations.extend(
        _check_diode_orientation_consistency(pcb, subcircuits)
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
