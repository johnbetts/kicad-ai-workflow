"""EE review agent: rule-based placement critique with suggested fixes.

Analyses a PCB placement against EE best practices and produces
coordinate-specific violations with suggested positions for each
offending component.
"""

from __future__ import annotations

import enum
import logging
import math
import subprocess
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
    from pathlib import Path

    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

# Standard 4-view render set for visual review
_RENDER_VIEWS: tuple[tuple[str, ...], ...] = (
    ("2d",),                          # 2D editor view
    ("3d", "--view", "top"),           # 3D top-down
    ("3d", "--view", "iso"),           # 3D isometric front
    ("3d", "--view", "iso-back"),      # 3D isometric back
)
_RENDER_SUFFIXES: tuple[str, ...] = (
    "_2d_top.png",
    "_3d_top.png",
    "_3d_iso.png",
    "_3d_isoback.png",
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBCIRCUIT_MAX_SPREAD_MM = 20.0
VOLTAGE_DOMAIN_MIN_GAP_MM = 2.0
CRYSTAL_MAX_DISTANCE_MM = 10.0

# Collision resolution defaults
_COLLISION_GAP_MM: float = 0.5
"""Clearance gap added when suggesting collision resolution positions."""

_DEFAULT_FP_SIZE: tuple[float, float] = (2.0, 2.0)
"""Fallback footprint size (w, h) when actual size is unknown."""

# Footprint name substrings that indicate edge-mount connectors.
# These connectors are designed to overhang the board edge (0mm margin per KI-021).
_EDGE_MOUNT_FOOTPRINT_PATTERNS: tuple[str, ...] = (
    "RJ45",
    "USB-C",
    "USB_C",
    "USB-A",
    "USB_A",
    "Micro-USB",
    "Micro_USB",
    "Mini-USB",
    "Mini_USB",
    "HDMI",
    "SD_Card",
    "SDCard",
    "HR911105A",
)


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
    COMPONENT_OFF_BOARD = "component_off_board"
    ZONE_OVERFLOW = "zone_overflow"
    GROUP_CONTAMINATION = "group_contamination"
    CONSTRAINT_COMPLIANCE = "constraint_compliance"
    POWER_LOOP_AREA = "power_loop_area"
    FLYBACK_DIODE_PROXIMITY = "flyback_diode_proximity"
    TERMINAL_ORIENTATION = "terminal_orientation"
    INTEGRITY_ISSUE = "integrity_issue"


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
class PersonaFinding:
    """A single finding from a fabricator or EE persona review."""

    persona: str  # "fab" | "ee"
    severity: str  # "critical" | "major" | "minor"
    category: str
    description: str
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlacementReview:
    """Complete placement review result.

    A review is only considered complete when:
    1. Programmatic rules have been run (always — violations/grade/summary)
    2. Renders have been generated (render_paths is non-empty)
    3. Both fab and EE persona reviews have been recorded

    Use :meth:`visual_review_complete` to check whether the review has
    been fully validated.  Use :func:`dataclasses.replace` to add
    persona findings after visual inspection::

        review = review_placement(pcb, req, render_dir=Path("output/"))
        # ... agent reads images, produces findings ...
        review = replace(review, fab_findings=fab, ee_findings=ee)
    """

    violations: tuple[PlacementViolation, ...]
    grade: str  # A/B/C/D/F
    summary: str
    render_paths: tuple[Path, ...] = ()
    fab_findings: tuple[PersonaFinding, ...] = ()
    ee_findings: tuple[PersonaFinding, ...] = ()

    @property
    def visual_review_complete(self) -> bool:
        """True only when renders exist AND both personas have reviewed."""
        return (
            len(self.render_paths) >= 4
            and len(self.fab_findings) > 0
            and len(self.ee_findings) > 0
        )

    @property
    def all_findings(self) -> tuple[PersonaFinding, ...]:
        """Combined findings from both personas."""
        return (*self.fab_findings, *self.ee_findings)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_edge_mount_connector(ref: str, pcb: PCBDesign) -> bool:
    """Return True if this footprint is an edge-mount connector.

    Edge-mount connectors (RJ45, USB-C, HDMI, etc.) are designed to
    overhang the board edge.  They should NOT be flagged for off-board
    or edge-clearance violations (KI-021: 0mm margin).
    """
    if not ref.startswith("J"):
        return False
    for fp in pcb.footprints:
        if fp.ref == ref:
            # Check footprint source (lib_id / footprint path)
            fp_src = (fp.footprint_source or "").upper()
            for pattern in _EDGE_MOUNT_FOOTPRINT_PATTERNS:
                if pattern.upper() in fp_src:
                    return True
            # Also check value field (e.g. "HR911105A")
            val = (fp.value or "").upper()
            for pattern in _EDGE_MOUNT_FOOTPRINT_PATTERNS:
                if pattern.upper() in val:
                    return True
            # Check lib_id if available
            lib_id = (fp.lib_id or "").upper()
            for pattern in _EDGE_MOUNT_FOOTPRINT_PATTERNS:
                if pattern.upper() in lib_id:
                    return True
            break
    return False


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
    """Extract footprint ref -> (width, height) map.

    Delegates to :func:`~kicad_pipeline.pcb.footprints.estimate_courtyard_mm`
    which uses a 3-tier resolution: courtyard graphics, fab body outline,
    then pad extents + package-type body extension.  This ensures the
    review agent uses the same size estimates as the placement optimizer,
    preventing false RF-edge violations on modules where the physical body
    (e.g. antenna) extends well beyond the pad field.
    """
    from kicad_pipeline.pcb.footprints import estimate_courtyard_mm

    return {fp.ref: estimate_courtyard_mm(fp) for fp in pcb.footprints}


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

    Uses the minimum of edge-to-edge distance (gap between bounding
    boxes) and minimum pad-to-pad distance, so large ICs (e.g. ESP32
    26x16mm) aren't penalized when a cap pad is close to an IC power pin
    even if centroids are far apart.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    sizes = _fp_size_dict(pcb)
    threshold = DECOUPLING_CAP_MAX_DISTANCE_MM

    # Build ref -> footprint lookup for pad access
    fp_by_ref: dict[str, object] = {fp.ref: fp for fp in pcb.footprints}

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
            edge_d = _edge_dist(ic_pos, ic_size, cap_pos, cap_size)

            # Also compute minimum pad-to-pad distance
            min_pad_d = edge_d  # fallback to edge distance
            ic_fp = fp_by_ref.get(ic_ref)
            cap_fp = fp_by_ref.get(cap_ref)
            if ic_fp is not None and cap_fp is not None:
                ic_pads = getattr(ic_fp, "pads", ())
                cap_pads = getattr(cap_fp, "pads", ())
                ic_origin = getattr(ic_fp, "position", None)
                cap_origin = getattr(cap_fp, "position", None)
                if ic_pads and cap_pads and ic_origin and cap_origin:
                    for ip in ic_pads:
                        ip_abs_x = ic_origin.x + ip.position.x
                        ip_abs_y = ic_origin.y + ip.position.y
                        for cp in cap_pads:
                            cp_abs_x = cap_origin.x + cp.position.x
                            cp_abs_y = cap_origin.y + cp.position.y
                            pd = math.sqrt(
                                (ip_abs_x - cp_abs_x) ** 2
                                + (ip_abs_y - cp_abs_y) ** 2
                            )
                            if pd < min_pad_d:
                                min_pad_d = pd

            d = min(edge_d, min_pad_d)
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
    """Check that sub-circuit components are clustered near anchor.

    Uses edge-to-edge distance for subcircuit types whose anchor is a
    large IC (MCU peripheral clusters, RF antenna), so modules like
    ESP32-S3-WROOM are not penalised for centre distance.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    sizes = _fp_size_dict(pcb)
    threshold = SUBCIRCUIT_MAX_SPREAD_MM

    # Relay drivers have inherently tall vertical extent: relay body (~16mm)
    # plus driver column (D + Q + R_LED + D_LED ≈ 20mm below center).
    # Use a relaxed threshold to avoid false positives on correct layouts.
    relay_driver_spread_mm = 25.0

    # Subcircuit types anchored on large ICs — use edge-to-edge distance
    edge_dist_types = frozenset({
        SubCircuitType.MCU_PERIPHERAL_CLUSTER,
        SubCircuitType.RF_ANTENNA,
    })

    for sc in subcircuits:
        if sc.circuit_type == SubCircuitType.DECOUPLING:
            continue  # Handled by decoupling check
        anchor_pos = positions.get(sc.anchor_ref)
        if anchor_pos is None:
            continue

        sc_threshold = (
            relay_driver_spread_mm
            if sc.circuit_type == SubCircuitType.RELAY_DRIVER
            else threshold
        )
        use_edge = sc.circuit_type in edge_dist_types
        anchor_size = sizes.get(sc.anchor_ref, (3.0, 3.0))

        for ref in sc.refs:
            if ref == sc.anchor_ref:
                continue
            pos = positions.get(ref)
            if pos is None:
                continue
            if use_edge:
                ref_size = sizes.get(ref, (2.0, 1.0))
                d = _edge_dist(anchor_pos, anchor_size, pos, ref_size)
            else:
                d = _dist(anchor_pos, pos)
            if d > sc_threshold:
                suggested = _point_toward(anchor_pos, pos, sc_threshold * 0.8)
                violations.append(PlacementViolation(
                    rule=PlacementRule.SUBCIRCUIT_SPREAD,
                    severity="major",
                    refs=(ref, sc.anchor_ref),
                    message=f"{ref} is {d:.1f}mm from anchor {sc.anchor_ref} "
                            f"in {sc.circuit_type.value} (max {sc_threshold}mm)",
                    current_value=d,
                    threshold=sc_threshold,
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

    # Pre-filter domain_refs to only include refs with positions
    domain_refs_positioned: dict[VoltageDomain, list[tuple[str, tuple[float, float]]]] = {}
    for domain, refs in domain_refs.items():
        positioned = [(r, positions[r]) for r in refs if r in positions]
        if positioned:
            domain_refs_positioned[domain] = positioned

    # Check inter-domain distances
    domains = list(domain_refs_positioned.keys())
    for i, d1 in enumerate(domains):
        for d2 in domains[i + 1:]:
            for r1, p1 in domain_refs_positioned[d1]:
                for r2, p2 in domain_refs_positioned[d2]:
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
    """Check that connectors (J*) are within threshold of board edge.

    Uses body/pad extent (not centroid) to measure distance, so tall
    vertical headers whose body reaches the board edge are not penalized.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    fp_sizes = _fp_size_dict(pcb)
    bounds = _board_bounds(pcb)
    min_x, min_y, max_x, max_y = bounds
    threshold = CONNECTOR_EDGE_MAX_MM

    _fp_rotations: dict[str, float] = {fp.ref: fp.rotation for fp in pcb.footprints}

    for fp in pcb.footprints:
        if _ref_prefix(fp.ref) != "J":
            continue
        pos = positions.get(fp.ref)
        if pos is None:
            continue
        x, y = pos
        w, h = fp_sizes.get(fp.ref, (2.0, 2.0))
        rot = _fp_rotations.get(fp.ref, 0.0)
        if rot % 180 in (90.0, 270.0):
            w, h = h, w

        # Distance from body edge to nearest board edge (not centroid)
        edge_dist = min(
            (x - w / 2.0) - min_x,   # left body edge to left board edge
            max_x - (x + w / 2.0),    # right body edge to right board edge
            (y - h / 2.0) - min_y,    # top body edge to top board edge
            max_y - (y + h / 2.0),    # bottom body edge to bottom board edge
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

            # THT connectors require board edge access — escalate severity
            is_tht = any(
                pad.pad_type == "thru_hole"
                for pad in fp.pads
            )
            severity = "critical" if is_tht else "major"
            msg_suffix = (
                " — THT connector requires board edge access"
                if is_tht
                else ""
            )

            violations.append(PlacementViolation(
                rule=PlacementRule.CONNECTOR_EDGE,
                severity=severity,
                refs=(fp.ref,),
                message=f"{fp.ref} is {edge_dist:.1f}mm from nearest board edge "
                        f"(max {threshold}mm for connectors){msg_suffix}",
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
            s1 = fp_sizes.get(r1, _DEFAULT_FP_SIZE)
            s2 = fp_sizes.get(r2, _DEFAULT_FP_SIZE)

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
            needed = (ms[0] + as_[0]) / 2.0 + _COLLISION_GAP_MM
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

    # Pre-build ref -> set[net_name] and net -> set[ref] indices
    _ref_nets: dict[str, set[str]] = {}
    _net_refs: dict[str, set[str]] = {}
    for net in requirements.nets:
        refs_in_net = {conn.ref for conn in net.connections}
        _net_refs[net.name] = refs_in_net
        for ref in refs_in_net:
            _ref_nets.setdefault(ref, set()).add(net.name)

    _gnd_set = frozenset({"GND", "AGND", "DGND", "PGND", "VSS", "AVSS"})

    # Build crystal->IC map via pre-built indices
    crystal_to_ic: dict[str, str] = {}
    for crystal in crystals:
        crystal_nets = _ref_nets.get(crystal.ref, set())
        for net_name in crystal_nets:
            if net_name.upper() in _gnd_set:
                continue
            for ref in _net_refs.get(net_name, set()):
                if ref.startswith("U") and ref in positions:
                    crystal_to_ic[crystal.ref] = ref
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

    # Only flag components that actually dissipate significant power —
    # relay coils (K), buck/LDO ICs (U with regulator keywords), and
    # power inductors (L).  Small-signal transistors (Q) and gate
    # resistors (R) in relay driver circuits are NOT thermal sources.
    power_keywords = {"REGULATOR", "BUCK", "LDO"}
    sensitive_keywords = {"ADC", "DAC", "VREF", "CRYSTAL", "SENSOR", "OPAMP"}

    # Identify power and sensitive components
    power_refs: list[str] = []
    sensitive_refs: list[str] = []

    for comp in requirements.components:
        val_desc = f"{comp.value} {comp.description or ''}".upper()
        prefix = _ref_prefix(comp.ref)
        # Only large thermal sources: relays, power ICs, power inductors
        if (prefix == "K"
                or (prefix == "U" and any(kw in val_desc for kw in power_keywords))
                or (prefix == "L" and "INDUCTOR" in val_desc and "FERRITE" not in val_desc)):
            power_refs.append(comp.ref)
        if any(kw in val_desc for kw in sensitive_keywords):
            sensitive_refs.append(comp.ref)

    threshold = 5.0  # minimum mm between power and sensitive
    # Pre-filter to only refs with positions
    power_positioned = [(r, positions[r]) for r in power_refs if r in positions]
    sensitive_positioned = [(r, positions[r]) for r in sensitive_refs if r in positions]

    for pr, p1 in power_positioned:
        for sr, p2 in sensitive_positioned:
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
    """Check that MCU peripherals are within threshold of their MCU.

    Uses edge-to-edge distance (gap between bounding boxes) so large
    MCU modules (e.g. ESP32-S3-WROOM ~20x26mm) are not penalised for
    centre distance when peripherals are physically right at their body
    edge.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    sizes = _fp_size_dict(pcb)
    threshold = MCU_PERIPHERAL_MAX_DISTANCE_MM

    # Track R* refs already checked via subcircuits to avoid duplicates
    checked_resistors: set[str] = set()

    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.MCU_PERIPHERAL_CLUSTER:
            continue
        anchor_pos = positions.get(sc.anchor_ref)
        if anchor_pos is None:
            continue
        anchor_size = sizes.get(sc.anchor_ref, (3.0, 3.0))

        for ref in sc.refs:
            if ref == sc.anchor_ref:
                continue
            if _ref_prefix(ref) == "R":
                checked_resistors.add(ref)
            pos = positions.get(ref)
            if pos is None:
                continue
            ref_size = sizes.get(ref, (2.0, 1.0))
            d = _edge_dist(anchor_pos, anchor_size, pos, ref_size)
            if d > threshold:
                is_pullup = _ref_prefix(ref) == "R"
                msg = (
                    f"Pull-up resistor {ref} is {d:.1f}mm edge-to-edge from "
                    f"bus master {sc.anchor_ref} (max {threshold}mm)"
                    if is_pullup
                    else f"{ref} is {d:.1f}mm edge-to-edge from MCU "
                         f"{sc.anchor_ref} (max {threshold}mm)"
                )
                suggested = _point_toward(anchor_pos, pos, threshold * 0.8)
                violations.append(PlacementViolation(
                    rule=PlacementRule.MCU_PERIPHERAL_PROXIMITY,
                    severity="major",
                    refs=(ref, sc.anchor_ref),
                    message=msg,
                    current_value=d,
                    threshold=threshold,
                    suggested_position=suggested,
                ))

    # Bug 007: Also check standalone R* refs near large MCUs (>20 pads)
    # that weren't already part of an MCU_PERIPHERAL_CLUSTER subcircuit.
    mcu_fps = [
        fp for fp in pcb.footprints
        if _ref_prefix(fp.ref) == "U" and len(fp.pads) > 20
    ]
    if mcu_fps:
        resistor_fps = [
            fp for fp in pcb.footprints
            if _ref_prefix(fp.ref) == "R" and fp.ref not in checked_resistors
        ]
        for r_fp in resistor_fps:
            r_pos = positions.get(r_fp.ref)
            if r_pos is None:
                continue
            r_size = sizes.get(r_fp.ref, (2.0, 1.0))
            # Check if this resistor is far from ALL MCUs
            nearest_mcu_ref: str | None = None
            nearest_d = float("inf")
            for mcu_fp in mcu_fps:
                mcu_pos = positions.get(mcu_fp.ref)
                if mcu_pos is None:
                    continue
                mcu_size = sizes.get(mcu_fp.ref, (3.0, 3.0))
                d = _edge_dist(mcu_pos, mcu_size, r_pos, r_size)
                if d < nearest_d:
                    nearest_d = d
                    nearest_mcu_ref = mcu_fp.ref
            # Only flag if far from all MCUs — these are likely bus pull-ups
            # that weren't detected as part of a subcircuit
            if nearest_mcu_ref and nearest_d > threshold:
                suggested = _point_toward(
                    positions[nearest_mcu_ref], r_pos, threshold * 0.8,
                )
                violations.append(PlacementViolation(
                    rule=PlacementRule.MCU_PERIPHERAL_PROXIMITY,
                    severity="major",
                    refs=(r_fp.ref, nearest_mcu_ref),
                    message=f"Pull-up resistor {r_fp.ref} is {nearest_d:.1f}mm "
                            f"edge-to-edge from bus master {nearest_mcu_ref} "
                            f"(max {threshold}mm)",
                    current_value=nearest_d,
                    threshold=threshold,
                    suggested_position=suggested,
                ))

    return violations


def _check_rf_edge_placement(
    pcb: PCBDesign,
    subcircuits: tuple[DetectedSubCircuit, ...],
) -> list[PlacementViolation]:
    """Check that RF antenna modules are on board edge.

    Uses edge-to-edge distance (centroid minus half-size) so large
    modules like ESP32-S3-WROOM are not penalised for having a centroid
    far from the edge when their body actually touches it.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    fp_sizes = _fp_size_dict(pcb)
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
        w, h = fp_sizes.get(sc.anchor_ref, (2.0, 2.0))
        # Edge-to-edge distance (body edge to board edge)
        edge_dist = min(
            (x - w / 2.0) - min_x,
            max_x - (x + w / 2.0),
            (y - h / 2.0) - min_y,
            max_y - (y + h / 2.0),
        )
        edge_dist = max(0.0, edge_dist)  # clamp negative (already touching)
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

    # Pre-build ref -> rotation index to avoid O(N^2) inner lookup
    _fp_rotations: dict[str, float] = {fp.ref: fp.rotation for fp in pcb.footprints}

    for fp in pcb.footprints:
        ref = fp.ref
        if ref not in positions:
            continue
        x, y = positions[ref]
        w, h = fp_sizes.get(ref, (2.0, 2.0))
        rot = _fp_rotations.get(ref, 0.0)
        if rot % 180 in (90.0, 270.0):
            w, h = h, w

        # Check each edge
        left_gap = (x - w / 2.0) - bx1
        right_gap = bx2 - (x + w / 2.0)
        top_gap = (y - h / 2.0) - by1
        bottom_gap = by2 - (y + h / 2.0)
        min_gap = min(left_gap, right_gap, top_gap, bottom_gap)

        if min_gap < crit_margin:
            # Edge-mount connectors are designed to overhang the board
            # edge — do not flag them for clearance violations (KI-021).
            if _is_edge_mount_connector(ref, pcb):
                continue
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
            # Connectors intentionally sit on board edges — only warn
            # for non-connector components
            if ref.startswith("J"):
                continue
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
        fp_rot_map = {fp.ref: fp.rotation for fp in pcb.footprints}
        diode_rots: dict[str, float] = {}
        for sc in instances:
            for ref in sc.refs:
                if ref.startswith("D") and ref in fp_rot_map:
                    diode_rots[ref] = fp_rot_map[ref]

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


def _check_power_loop_area(
    pcb: PCBDesign,
    subcircuits: tuple[DetectedSubCircuit, ...],
) -> list[PlacementViolation]:
    """Check that buck converter hot loop (IC + inductor + catch diode) is tight.

    The IC, inductor, and catch/Schottky diode must form a small triangle
    for low EMI.  If the perimeter exceeds 25mm, flag as CRITICAL.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    perimeter_threshold = 25.0

    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.BUCK_CONVERTER:
            continue
        # Identify key components: IC (U*), inductor (L*), catch diode (D*)
        ic_refs = [r for r in sc.refs if _ref_prefix(r) == "U"]
        l_refs = [r for r in sc.refs if _ref_prefix(r) == "L"]
        d_refs = [r for r in sc.refs if _ref_prefix(r) == "D"]

        if not ic_refs or not l_refs or not d_refs:
            continue

        ic_ref = ic_refs[0]
        l_ref = l_refs[0]
        d_ref = d_refs[0]

        ic_pos = positions.get(ic_ref)
        l_pos = positions.get(l_ref)
        d_pos = positions.get(d_ref)
        if ic_pos is None or l_pos is None or d_pos is None:
            continue

        # Measure triangle perimeter
        perimeter = _dist(ic_pos, l_pos) + _dist(l_pos, d_pos) + _dist(d_pos, ic_pos)
        if perimeter > perimeter_threshold:
            violations.append(PlacementViolation(
                rule=PlacementRule.POWER_LOOP_AREA,
                severity="critical",
                refs=(ic_ref, l_ref, d_ref),
                message=f"Buck converter hot loop too large: "
                        f"{ic_ref}/{l_ref}/{d_ref} perimeter={perimeter:.1f}mm "
                        f"(max {perimeter_threshold}mm)",
                current_value=perimeter,
                threshold=perimeter_threshold,
                suggested_position=None,
            ))

    return violations


def _check_flyback_diode_proximity(
    pcb: PCBDesign,
    subcircuits: tuple[DetectedSubCircuit, ...],
    requirements: ProjectRequirements | None = None,
) -> list[PlacementViolation]:
    """Check that flyback diodes are close to their relay coil pins.

    In relay driver subcircuits, the flyback/snubber diode must be near the
    relay coil pins to minimise inductive spike loop area.  Threshold: 8mm
    measured from the relay's coil pin (not the relay centre).

    Only checks actual flyback diodes (connected to ``_COIL`` nets), not
    LED indicator diodes that happen to share the subcircuit.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    threshold = 8.0

    # Build set of D refs that are on COIL nets (actual flyback diodes)
    # and map relay ref → coil pin number for pin-level distance measurement
    flyback_d_refs: set[str] | None = None
    relay_coil_pins: dict[str, str] = {}  # K_ref → pin number
    if requirements is not None:
        flyback_d_refs = set()
        for net in requirements.nets:
            if "_COIL" not in net.name.upper():
                continue
            for conn in net.connections:
                if conn.ref.startswith("D"):
                    flyback_d_refs.add(conn.ref)
                if conn.ref.startswith("K"):
                    relay_coil_pins[conn.ref] = conn.pin

    # Build relay footprint lookup for coil pin positions
    fp_by_ref: dict[str, object] = {fp.ref: fp for fp in pcb.footprints}

    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.RELAY_DRIVER:
            continue
        k_refs = [r for r in sc.refs if _ref_prefix(r) == "K"]
        d_refs = [r for r in sc.refs if _ref_prefix(r) == "D"]

        # Filter to actual flyback diodes when requirements are available
        if flyback_d_refs is not None:
            d_refs = [r for r in d_refs if r in flyback_d_refs]

        if not k_refs or not d_refs:
            continue

        for k_ref in k_refs:
            # Use coil pin position if available, otherwise relay centroid
            measure_pos = positions.get(k_ref)
            coil_pin_num = relay_coil_pins.get(k_ref)
            fp_obj = fp_by_ref.get(k_ref)
            if coil_pin_num is not None and fp_obj is not None:
                import math as _math
                for pad in fp_obj.pads:  # type: ignore[union-attr]
                    if str(pad.number) == str(coil_pin_num):
                        rot_rad = _math.radians(fp_obj.rotation)  # type: ignore[union-attr]
                        abs_x = fp_obj.position.x + (  # type: ignore[union-attr]
                            pad.position.x * _math.cos(rot_rad)
                            - pad.position.y * _math.sin(rot_rad)
                        )
                        abs_y = fp_obj.position.y + (  # type: ignore[union-attr]
                            pad.position.x * _math.sin(rot_rad)
                            + pad.position.y * _math.cos(rot_rad)
                        )
                        measure_pos = (abs_x, abs_y)
                        break

            if measure_pos is None:
                continue
            for d_ref in d_refs:
                d_pos = positions.get(d_ref)
                if d_pos is None:
                    continue
                d = _dist(measure_pos, d_pos)
                if d > threshold:
                    violations.append(PlacementViolation(
                        rule=PlacementRule.FLYBACK_DIODE_PROXIMITY,
                        severity="major",
                        refs=(d_ref, k_ref),
                        message=f"Flyback diode {d_ref} too far from relay coil "
                                f"pin {k_ref}: {d:.1f}mm (max {threshold}mm)",
                        current_value=d,
                        threshold=threshold,
                        suggested_position=None,
                    ))

    return violations


# Footprint name patterns that indicate screw terminal connectors
_SCREW_TERMINAL_PATTERNS: tuple[str, ...] = (
    "WJ128", "WJ500", "Terminal", "P5.00", "Screw",
)


def _check_terminal_orientation(
    pcb: PCBDesign,
) -> list[PlacementViolation]:
    """Check that screw terminal connectors face outward from the board edge.

    Terminals should have their wire entry facing away from the board
    interior.  The expected rotation depends on which board edge the
    terminal is nearest to.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    bounds = _board_bounds(pcb)
    min_x, min_y, max_x, max_y = bounds
    board_w = max_x - min_x
    board_h = max_y - min_y

    for fp in pcb.footprints:
        # Identify screw terminals by lib_id/value/footprint_source
        fp_text = f"{fp.lib_id or ''} {fp.value or ''} {fp.footprint_source or ''}".upper()
        is_terminal = any(pat.upper() in fp_text for pat in _SCREW_TERMINAL_PATTERNS)
        if not is_terminal:
            continue

        pos = positions.get(fp.ref)
        if pos is None:
            continue
        x, y = pos
        rot = fp.rotation % 360.0

        # Determine nearest edge
        dist_top = y - min_y
        dist_bottom = max_y - y
        dist_left = x - min_x
        dist_right = max_x - x
        min_edge_dist = min(dist_top, dist_bottom, dist_left, dist_right)

        # Expected rotation for wire entry to face outward
        if min_edge_dist == dist_top and dist_top < board_h / 4.0:
            expected_rot = 0.0
        elif min_edge_dist == dist_bottom and dist_bottom < board_h / 4.0:
            expected_rot = 180.0
        elif min_edge_dist == dist_left and dist_left < board_w / 4.0:
            expected_rot = 90.0
        elif min_edge_dist == dist_right and dist_right < board_w / 4.0:
            expected_rot = 270.0
        else:
            # Not clearly near an edge — skip
            continue

        # Angular difference (handle wraparound)
        diff = abs(rot - expected_rot)
        if diff > 180.0:
            diff = 360.0 - diff

        if diff > 45.0:
            violations.append(PlacementViolation(
                rule=PlacementRule.TERMINAL_ORIENTATION,
                severity="major",
                refs=(fp.ref,),
                message=f"Screw terminal {fp.ref} wire entry faces board interior "
                        f"(rotation={rot:.0f}°, expected≈{expected_rot:.0f}° "
                        f"for nearest edge)",
                current_value=rot,
                threshold=expected_rot,
                suggested_position=None,
            ))

    return violations


def _fp_raw_pad_bbox(pcb: PCBDesign) -> dict[str, tuple[float, float, float, float]]:
    """Compute raw pad bounding box per footprint (no margin).

    Returns ref -> (half_w, half_h, centroid_x, centroid_y) where
    half_w/half_h are the half-extents from the centroid.
    """
    from kicad_pipeline.pcb.pin_map import origin_to_centroid

    result: dict[str, tuple[float, float, float, float]] = {}
    for fp in pcb.footprints:
        if not fp.pads:
            result[fp.ref] = (1.5, 1.5, fp.position.x, fp.position.y)
            continue
        # Compute raw pad extents in local coordinates (before rotation)
        xs_min = [p.position.x - p.size_x / 2.0 for p in fp.pads]
        xs_max = [p.position.x + p.size_x / 2.0 for p in fp.pads]
        ys_min = [p.position.y - p.size_y / 2.0 for p in fp.pads]
        ys_max = [p.position.y + p.size_y / 2.0 for p in fp.pads]
        raw_w = max(xs_max) - min(xs_min)
        raw_h = max(ys_max) - min(ys_min)
        # Apply rotation: swap w/h for 90/270
        rot = fp.rotation % 360.0
        if 80.0 <= rot <= 100.0 or 260.0 <= rot <= 280.0:
            raw_w, raw_h = raw_h, raw_w
        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        result[fp.ref] = (raw_w / 2.0, raw_h / 2.0, cx, cy)
    return result


def _check_component_off_board(
    pcb: PCBDesign,
) -> list[PlacementViolation]:
    """Check if any component's pads physically extend past the board outline.

    Uses raw pad extents (no +1mm margin from _fp_size_dict) to detect
    components that cannot be manufactured because pads are off-board.
    """
    violations: list[PlacementViolation] = []
    bx1, by1, bx2, by2 = _board_bounds(pcb)
    pad_bboxes = _fp_raw_pad_bbox(pcb)
    margin = 1.0  # desired minimum margin from board edge

    for fp in pcb.footprints:
        ref = fp.ref
        bbox = pad_bboxes.get(ref)
        if bbox is None:
            continue
        half_w, half_h, cx, cy = bbox

        # Compute how far each pad edge extends past the board edge
        left_gap = (cx - half_w) - bx1
        right_gap = bx2 - (cx + half_w)
        top_gap = (cy - half_h) - by1
        bottom_gap = by2 - (cy + half_h)
        min_gap = min(left_gap, right_gap, top_gap, bottom_gap)

        if min_gap < 0.0:
            # Edge-mount connectors are designed to overhang — skip (KI-021).
            if _is_edge_mount_connector(ref, pcb):
                continue
            # Pads physically off-board — compute suggested position
            sx, sy = cx, cy
            if left_gap < 0.0:
                sx = bx1 + half_w + margin
            elif right_gap < 0.0:
                sx = bx2 - half_w - margin
            if top_gap < 0.0:
                sy = by1 + half_h + margin
            elif bottom_gap < 0.0:
                sy = by2 - half_h - margin

            violations.append(PlacementViolation(
                rule=PlacementRule.COMPONENT_OFF_BOARD,
                severity="critical",
                refs=(ref,),
                message=f"{ref} pads extend {abs(min_gap):.1f}mm past board edge "
                        f"(cannot be manufactured)",
                current_value=min_gap,
                threshold=0.0,
                suggested_position=(sx, sy),
            ))

    return violations


def _check_zone_overflow(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[PlacementViolation]:
    """Detect when two FeatureBlock groups' component bounding boxes overlap.

    Significant overlap means zones are not properly partitioned and
    component nudging cannot fix the layout.
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    group_bboxes = _build_feature_bboxes(requirements, positions, shrink=0.0)

    # Check all pairs for overlap
    names = list(group_bboxes.keys())
    for i, name_a in enumerate(names):
        ax1, ay1, ax2, ay2 = group_bboxes[name_a]
        a_area = max((ax2 - ax1) * (ay2 - ay1), 0.01)
        for name_b in names[i + 1:]:
            bx1, by1, bx2, by2 = group_bboxes[name_b]
            b_area = max((bx2 - bx1) * (by2 - by1), 0.01)

            # Intersection
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue  # no overlap
            intersect_area = (ix2 - ix1) * (iy2 - iy1)
            smaller_area = min(a_area, b_area)
            pct = intersect_area / smaller_area * 100.0

            if pct > 20.0:
                severity = "critical" if pct > 50.0 else "major"
                violations.append(PlacementViolation(
                    rule=PlacementRule.ZONE_OVERFLOW,
                    severity=severity,
                    refs=(),
                    message=f"Group '{name_a}' overlaps group '{name_b}' "
                            f"by {pct:.0f}%",
                    current_value=pct,
                    threshold=20.0,
                    suggested_position=None,
                ))

    return violations


def _build_feature_bboxes(
    requirements: ProjectRequirements,
    positions: dict[str, tuple[float, float]],
    shrink: float = 0.0,
) -> dict[str, tuple[float, float, float, float]]:
    """Compute axis-aligned bbox per FeatureBlock from placed positions.

    Args:
        requirements: Project requirements with feature blocks.
        positions: ref -> (x, y) position mapping.
        shrink: Inset amount in mm from each edge (for contamination tolerance).

    Returns:
        Mapping from feature name to (x1, y1, x2, y2) bbox.
    """
    bboxes: dict[str, tuple[float, float, float, float]] = {}
    for fb in requirements.features:
        member_positions = [positions[r] for r in fb.components if r in positions]
        if len(member_positions) < 2:
            continue
        xs = [p[0] for p in member_positions]
        ys = [p[1] for p in member_positions]
        x1 = min(xs) + shrink
        y1 = min(ys) + shrink
        x2 = max(xs) - shrink
        y2 = max(ys) - shrink
        if x1 < x2 and y1 < y2:
            bboxes[fb.name] = (x1, y1, x2, y2)
    return bboxes


def _build_ref_groups(
    requirements: ProjectRequirements,
) -> dict[str, list[str]]:
    """Build ref -> list[group_name] mapping."""
    ref_groups: dict[str, list[str]] = {}
    for fb in requirements.features:
        for r in fb.components:
            ref_groups.setdefault(r, []).append(fb.name)
    return ref_groups


def _check_group_contamination(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[PlacementViolation]:
    """Detect when a component from one group is inside another group's region.

    Exemptions:
    - Connectors (J*) are exempt (often legitimately span boundaries)
    - Components in multiple FeatureBlocks are exempt
    """
    violations: list[PlacementViolation] = []
    positions = _fp_positions(pcb)
    ref_groups = _build_ref_groups(requirements)
    group_bboxes = _build_feature_bboxes(requirements, positions, shrink=2.0)

    # Check each component against groups it doesn't belong to
    for ref, pos in positions.items():
        # Exemption: connectors
        if _ref_prefix(ref) == "J":
            continue
        own_groups = ref_groups.get(ref, [])
        # Exemption: component in multiple feature blocks
        if len(own_groups) > 1:
            continue
        own_group = own_groups[0] if own_groups else ""
        x, y = pos
        for group_name, (gx1, gy1, gx2, gy2) in group_bboxes.items():
            if group_name == own_group:
                continue
            if gx1 <= x <= gx2 and gy1 <= y <= gy2:
                violations.append(PlacementViolation(
                    rule=PlacementRule.GROUP_CONTAMINATION,
                    severity="major",
                    refs=(ref,),
                    message=f"{ref} (group '{own_group}') is inside "
                            f"group '{group_name}' region",
                    current_value=0.0,
                    threshold=0.0,
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


def _check_constraint_compliance(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[PlacementViolation]:
    """Check placement constraint compliance using the constraint auditor."""
    try:
        from kicad_pipeline.validation.constraint_auditor import (
            audit_placement_constraints,
        )
        audit_violations = audit_placement_constraints(pcb, requirements)
    except Exception:
        return []

    violations: list[PlacementViolation] = []
    for av in audit_violations:
        violations.append(PlacementViolation(
            rule=PlacementRule.CONSTRAINT_COMPLIANCE,
            severity=av.severity,
            refs=av.refs,
            message=av.message,
            current_value=av.measured_value,
            threshold=av.threshold,
            suggested_position=None,
        ))
    return violations


# ---------------------------------------------------------------------------
# Image rendering helpers
# ---------------------------------------------------------------------------


def _render_review_images(
    pcb: PCBDesign,
    render_dir: Path,
    board_name: str,
) -> tuple[Path, ...]:
    """Write the PCB to a temp file and render 4 standard views.

    Returns paths to the generated PNG files (2D top, 3D top, 3D iso,
    3D iso-back).  Returns only the paths that actually rendered
    successfully — callers should check ``len(paths) >= 4``.
    """
    from kicad_pipeline.pcb.builder import write_pcb

    render_dir.mkdir(parents=True, exist_ok=True)

    # Write PCB to render dir (temp file if not already present)
    pcb_path = render_dir / f"{board_name}.kicad_pcb"
    write_pcb(pcb, pcb_path)

    rendered: list[Path] = []
    for view_args, suffix in zip(_RENDER_VIEWS, _RENDER_SUFFIXES, strict=True):
        out_path = render_dir / f"{board_name}{suffix}"
        cmd = ["kicad-image-gen", *view_args, str(pcb_path), "-o", str(out_path)]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and out_path.exists():
                rendered.append(out_path)
                _log.debug("Rendered: %s", out_path)
            else:
                _log.warning(
                    "kicad-image-gen failed for %s: %s",
                    suffix, result.stderr[:200],
                )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            _log.warning("kicad-image-gen unavailable: %s", exc)
            break  # No point trying other views if the tool is missing

    return tuple(rendered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def review_placement(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    subcircuits: tuple[DetectedSubCircuit, ...] | None = None,
    domain_map: dict[str, VoltageDomain] | None = None,
    *,
    render_dir: Path | None = None,
    board_name: str = "board",
    integrity_issues: tuple[object, ...] = (),
) -> PlacementReview:
    """Review a PCB placement against EE best practices.

    Runs all programmatic placement rules.  When *render_dir* is provided,
    also generates 2D + 3D renders (4 standard views) via ``kicad-image-gen``
    and includes the paths in the returned :class:`PlacementReview`.

    A review is **not considered complete** until both fabricator and EE
    persona findings have been recorded via :func:`dataclasses.replace`::

        review = review_placement(pcb, req, render_dir=Path("output/board"))
        # ... agent reads review.render_paths images ...
        review = replace(review, fab_findings=fab_results, ee_findings=ee_results)
        assert review.visual_review_complete

    For fast inner-loop calls during optimization, omit *render_dir* to
    skip rendering (the programmatic rules still run).

    Args:
        pcb: The PCB design to review.
        requirements: Project requirements.
        subcircuits: Pre-detected sub-circuits (will be detected if None).
        domain_map: Pre-classified voltage domains (will be classified if None).
        render_dir: Directory for rendered PNG images.  When ``None``,
            no images are generated and ``render_paths`` will be empty.
        board_name: Stem used for render filenames (e.g. ``"train_mcu_core"``).
        integrity_issues: Optional tuple of integrity issue objects (with
            ``.severity`` attribute).  Critical issues are injected as
            synthetic :data:`PlacementRule.INTEGRITY_ISSUE` violations so
            they affect the placement grade.

    Returns:
        PlacementReview with violations, grade, summary, and (if
        *render_dir* was set) render image paths.
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
    all_violations.extend(
        _check_component_off_board(pcb)
    )
    all_violations.extend(
        _check_zone_overflow(pcb, requirements)
    )
    all_violations.extend(
        _check_group_contamination(pcb, requirements)
    )
    all_violations.extend(
        _check_constraint_compliance(pcb, requirements)
    )
    all_violations.extend(
        _check_power_loop_area(pcb, subcircuits)
    )
    all_violations.extend(
        _check_flyback_diode_proximity(pcb, subcircuits, requirements)
    )
    all_violations.extend(
        _check_terminal_orientation(pcb)
    )

    # Bug 004: Inject critical integrity issues as synthetic violations
    for issue in integrity_issues:
        sev = getattr(issue, "severity", "minor")
        if sev == "critical":
            all_violations.append(PlacementViolation(
                rule=PlacementRule.INTEGRITY_ISSUE,
                severity=sev,
                refs=(getattr(issue, "ref", "?"),),
                message=getattr(issue, "message", str(issue)),
                current_value=0.0,
                threshold=0.0,
                suggested_position=None,
            ))

    violations = tuple(all_violations)
    grade = _compute_grade(violations)

    critical = sum(1 for v in violations if v.severity == "critical")
    major = sum(1 for v in violations if v.severity == "major")
    minor = sum(1 for v in violations if v.severity == "minor")
    summary = (
        f"Grade {grade}: {len(violations)} violations "
        f"({critical} critical, {major} major, {minor} minor)"
    )

    # Render images when requested
    render_paths: tuple[Path, ...] = ()
    if render_dir is not None:
        render_paths = _render_review_images(pcb, render_dir, board_name)
        if len(render_paths) < 4:
            _log.warning(
                "Only %d/4 renders generated — visual review incomplete",
                len(render_paths),
            )

    _log.info("Placement review: %s", summary)

    return PlacementReview(
        violations=violations,
        grade=grade,
        summary=summary,
        render_paths=render_paths,
    )
