"""Board state snapshot — spatial relationships as queryable data + text report.

Captures every component's position, bounding box, edge distances, group/zone/domain
membership, and inter-component spatial relationships (overlaps, isolation gaps).
The text report gives an LLM a complete picture of board layout without rendering PNGs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.pcb.footprints import estimate_courtyard_mm
from kicad_pipeline.pcb.pin_map import (
    CardinalSide,
    compute_pin_map,
    origin_to_centroid,
    pad_extent_in_board_space,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kicad_pipeline.models.pcb import Footprint, PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.functional_grouper import (
        DetectedSubCircuit,
        VoltageDomain,
    )
    from kicad_pipeline.optimization.group_placer import PlacedGroup
    from kicad_pipeline.optimization.review_agent import PlacementReview
    from kicad_pipeline.optimization.zone_partitioner import BoardZone


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacedComponent:
    """Spatial snapshot of a single placed component."""

    ref: str
    value: str
    footprint_id: str
    centroid: tuple[float, float]
    origin: tuple[float, float]
    rotation: float
    bbox: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    courtyard_size: tuple[float, float]  # (w, h)
    nearest_edge: str  # "top"|"bottom"|"left"|"right"
    nearest_edge_distance_mm: float
    mating_face: str  # "N"|"S"|"E"|"W"|""
    group_name: str
    zone_name: str
    voltage_domain: str
    subcircuit_types: tuple[str, ...]
    overlapping_refs: tuple[str, ...]


@dataclass(frozen=True)
class OverlapPair:
    """Two components whose pad-extent bounding boxes overlap."""

    ref_a: str
    ref_b: str
    overlap_area_mm2: float


@dataclass(frozen=True)
class EdgeViolation:
    """A component too close to — or past — a board edge."""

    ref: str
    edge: str
    clearance_mm: float
    is_off_board: bool


@dataclass(frozen=True)
class IsolationGap:
    """Minimum distance between two voltage domains."""

    domain_a: str
    domain_b: str
    min_gap_mm: float
    closest_pair: tuple[str, str]  # (ref_a, ref_b)


@dataclass(frozen=True)
class ZoneOccupancy:
    """How fully a zone is utilized."""

    zone_name: str
    rect: tuple[float, float, float, float]
    area_mm2: float
    component_count: int
    utilization_pct: float
    refs: tuple[str, ...]


@dataclass(frozen=True)
class GroupCohesion:
    """Spread and density metrics for a functional group."""

    group_name: str
    ref_count: int
    spread_mm: float
    bbox: tuple[float, float, float, float]
    density_pct: float


@dataclass(frozen=True)
class BoardState:
    """Complete spatial snapshot of the board after placement."""

    board_bounds: tuple[float, float, float, float]
    board_width_mm: float
    board_height_mm: float
    components: tuple[PlacedComponent, ...]
    overlaps: tuple[OverlapPair, ...]
    edge_violations: tuple[EdgeViolation, ...]
    isolation_gaps: tuple[IsolationGap, ...]
    zone_occupancy: tuple[ZoneOccupancy, ...]
    group_cohesion: tuple[GroupCohesion, ...]
    total_utilization_pct: float
    off_board_count: int
    review_grade: str
    review_violation_count: int

    def to_report(self, cell_size_mm: float = 10.0) -> str:
        """Generate a text report suitable for LLM consumption."""
        return _format_report(self, cell_size_mm)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EDGE_NAMES: tuple[str, ...] = ("top", "bottom", "left", "right")

_MATING_ARROWS: dict[str, str] = {
    "N": "\u2191NORTH",
    "S": "\u2193SOUTH",
    "E": "\u2192EAST",
    "W": "\u2190WEST",
}


def _board_bounds_from_outline(pcb: PCBDesign) -> tuple[float, float, float, float]:
    """Extract axis-aligned bounding box from board outline polygon."""
    pts = pcb.outline.polygon
    if not pts:
        return (0.0, 0.0, 100.0, 100.0)
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _edge_distances(
    bbox: tuple[float, float, float, float],
    board: tuple[float, float, float, float],
) -> dict[str, float]:
    """Distance from component bbox edges to each board edge (negative = off-board)."""
    bx0, by0, bx1, by1 = board
    cx0, cy0, cx1, cy1 = bbox
    return {
        "top": cy0 - by0,
        "bottom": by1 - cy1,
        "left": cx0 - bx0,
        "right": bx1 - cx1,
    }


def _nearest_edge(distances: dict[str, float]) -> tuple[str, float]:
    """Return (edge_name, distance) for the closest board edge."""
    return min(distances.items(), key=lambda kv: abs(kv[1]))


def _mating_face_for_connector(fp: Footprint, rotation: float) -> str:
    """Determine connector mating direction from pad distribution."""
    pin_map = compute_pin_map(fp, rotation)
    if not pin_map.entries:
        return ""
    # Count pads on each side; the side with fewest pads (often 0) opposite the
    # mating direction. Convention: mating face is the side with the most pads.
    side_counts: dict[CardinalSide, int] = {}
    for entry in pin_map.entries:
        side_counts[entry.side] = side_counts.get(entry.side, 0) + 1
    # Remove CENTER — not useful for mating direction
    side_counts.pop(CardinalSide.CENTER, None)
    if not side_counts:
        return ""
    dominant = max(side_counts, key=lambda s: side_counts[s])
    mapping = {
        CardinalSide.NORTH: "N",
        CardinalSide.SOUTH: "S",
        CardinalSide.EAST: "E",
        CardinalSide.WEST: "W",
    }
    return mapping.get(dominant, "")


def _bbox_overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute AABB overlap area. Returns 0.0 if no overlap."""
    ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ox * oy


def _build_ref_to_group(
    requirements: ProjectRequirements,
) -> dict[str, str]:
    """Map component ref → FeatureBlock name."""
    mapping: dict[str, str] = {}
    for fb in requirements.features:
        for ref in fb.components:
            mapping[ref] = fb.name
    return mapping


def _build_ref_to_zone(
    zones: Sequence[BoardZone] | None,
    groups: Sequence[PlacedGroup] | None,
) -> dict[str, str]:
    """Map component ref → zone name."""
    mapping: dict[str, str] = {}
    if groups is not None:
        for g in groups:
            mapping.update({ref: g.zone for ref in g.refs})
    elif zones is not None:
        for z in zones:
            for gname in z.groups:
                mapping[gname] = z.name  # group name → zone name (fallback)
    return mapping


def _build_ref_to_domain(
    subcircuits: Sequence[DetectedSubCircuit] | None,
    domain_map: dict[str, VoltageDomain] | None,
) -> dict[str, str]:
    """Map component ref → voltage domain string."""
    mapping: dict[str, str] = {}
    if domain_map is not None:
        for ref, dom in domain_map.items():
            mapping[ref] = dom.value
    if subcircuits is not None:
        for sc in subcircuits:
            for ref in sc.refs:
                if ref not in mapping:
                    mapping[ref] = sc.domain.value
    return mapping


def _build_ref_to_subcircuits(
    subcircuits: Sequence[DetectedSubCircuit] | None,
) -> dict[str, list[str]]:
    """Map component ref → list of SubCircuitType values."""
    mapping: dict[str, list[str]] = {}
    if subcircuits is not None:
        for sc in subcircuits:
            for ref in sc.refs:
                mapping.setdefault(ref, []).append(sc.circuit_type.value)
    return mapping


def _compute_isolation_gaps(
    components: tuple[PlacedComponent, ...],
) -> tuple[IsolationGap, ...]:
    """Find minimum gap between each pair of voltage domains."""
    # Group components by domain
    domain_comps: dict[str, list[PlacedComponent]] = {}
    for c in components:
        if c.voltage_domain:
            domain_comps.setdefault(c.voltage_domain, []).append(c)

    domains = sorted(domain_comps.keys())
    gaps: list[IsolationGap] = []
    for i, da in enumerate(domains):
        for db in domains[i + 1 :]:
            min_dist = float("inf")
            closest = ("", "")
            for ca in domain_comps[da]:
                for cb in domain_comps[db]:
                    dx = ca.centroid[0] - cb.centroid[0]
                    dy = ca.centroid[1] - cb.centroid[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < min_dist:
                        min_dist = dist
                        closest = (ca.ref, cb.ref)
            if min_dist < float("inf"):
                gaps.append(IsolationGap(da, db, round(min_dist, 1), closest))
    return tuple(gaps)


def _compute_zone_occupancy(
    zones: Sequence[BoardZone] | None,
    components: tuple[PlacedComponent, ...],
) -> tuple[ZoneOccupancy, ...]:
    """Compute utilization per zone."""
    if zones is None:
        return ()
    results: list[ZoneOccupancy] = []
    for z in zones:
        zx0, zy0, zx1, zy1 = z.rect
        zone_area = (zx1 - zx0) * (zy1 - zy0)
        if zone_area <= 0:
            continue
        refs_in_zone: list[str] = []
        comp_area = 0.0
        for c in components:
            cx, cy = c.centroid
            if z.contains(cx, cy):
                refs_in_zone.append(c.ref)
                cw, ch = c.courtyard_size
                comp_area += cw * ch
        util = min(100.0, (comp_area / zone_area) * 100.0) if zone_area > 0 else 0.0
        results.append(
            ZoneOccupancy(
                zone_name=z.name,
                rect=z.rect,
                area_mm2=round(zone_area, 1),
                component_count=len(refs_in_zone),
                utilization_pct=round(util, 1),
                refs=tuple(sorted(refs_in_zone)),
            )
        )
    return tuple(results)


def _compute_group_cohesion(
    components: tuple[PlacedComponent, ...],
) -> tuple[GroupCohesion, ...]:
    """Compute spread and density per group."""
    groups: dict[str, list[PlacedComponent]] = {}
    for c in components:
        if c.group_name:
            groups.setdefault(c.group_name, []).append(c)

    results: list[GroupCohesion] = []
    for gname in sorted(groups):
        comps = groups[gname]
        xs = [c.centroid[0] for c in comps]
        ys = [c.centroid[1] for c in comps]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        spread = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        bbox_area = max((x1 - x0), 1.0) * max((y1 - y0), 1.0)
        comp_area = sum(c.courtyard_size[0] * c.courtyard_size[1] for c in comps)
        density = min(100.0, (comp_area / bbox_area) * 100.0)
        results.append(
            GroupCohesion(
                group_name=gname,
                ref_count=len(comps),
                spread_mm=round(spread, 1),
                bbox=(round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)),
                density_pct=round(density, 1),
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_placed_component(
    fp: Footprint,
    board: tuple[float, float, float, float],
    ref_to_group: dict[str, str],
    ref_to_zone: dict[str, str],
    ref_to_domain: dict[str, str],
    ref_to_subcircuits: dict[str, list[str]],
) -> tuple[str, tuple[float, float, float, float], PlacedComponent]:
    """Build a single PlacedComponent from a footprint (without overlap info)."""
    ox, oy = fp.position.x, fp.position.y
    rot = fp.rotation

    centroid = origin_to_centroid(fp, ox, oy, rot)
    bbox = pad_extent_in_board_space(fp, ox, oy, rot)
    courtyard = estimate_courtyard_mm(fp)
    edges = _edge_distances(bbox, board)
    edge_name, edge_dist = _nearest_edge(edges)

    mating = _mating_face_for_connector(fp, rot) if fp.ref.startswith("J") else ""

    pc = PlacedComponent(
        ref=fp.ref,
        value=fp.value,
        footprint_id=fp.lib_id,
        centroid=(round(centroid[0], 2), round(centroid[1], 2)),
        origin=(round(ox, 2), round(oy, 2)),
        rotation=rot,
        bbox=(round(bbox[0], 2), round(bbox[1], 2), round(bbox[2], 2), round(bbox[3], 2)),
        courtyard_size=(round(courtyard[0], 2), round(courtyard[1], 2)),
        nearest_edge=edge_name,
        nearest_edge_distance_mm=round(edge_dist, 2),
        mating_face=mating,
        group_name=ref_to_group.get(fp.ref, ""),
        zone_name=ref_to_zone.get(fp.ref, ""),
        voltage_domain=ref_to_domain.get(fp.ref, ""),
        subcircuit_types=tuple(ref_to_subcircuits.get(fp.ref, [])),
        overlapping_refs=(),
    )
    return (fp.ref, bbox, pc)


def _build_placed_components(
    pcb: PCBDesign,
    board: tuple[float, float, float, float],
    ref_to_group: dict[str, str],
    ref_to_zone: dict[str, str],
    ref_to_domain: dict[str, str],
    ref_to_subcircuits: dict[str, list[str]],
) -> list[tuple[str, tuple[float, float, float, float], PlacedComponent]]:
    """Build PlacedComponent entries for all footprints (without overlap info)."""
    return [
        _build_placed_component(
            fp, board, ref_to_group, ref_to_zone, ref_to_domain, ref_to_subcircuits,
        )
        for fp in pcb.footprints
    ]


def _detect_overlaps(
    fp_data: list[tuple[str, tuple[float, float, float, float], PlacedComponent]],
) -> tuple[tuple[PlacedComponent, ...], list[OverlapPair]]:
    """Detect AABB overlaps and return components with overlap refs populated.

    Returns:
        Tuple of (sorted components with overlap info, list of overlap pairs).
    """
    overlap_threshold_mm2 = 0.01

    overlaps: list[OverlapPair] = []
    overlap_map: dict[str, list[str]] = {}
    n = len(fp_data)
    for i in range(n):
        ref_a, bbox_a, _ = fp_data[i]
        for j in range(i + 1, n):
            ref_b, bbox_b, _ = fp_data[j]
            area = _bbox_overlap_area(bbox_a, bbox_b)
            if area > overlap_threshold_mm2:
                overlaps.append(OverlapPair(ref_a, ref_b, round(area, 2)))
                overlap_map.setdefault(ref_a, []).append(ref_b)
                overlap_map.setdefault(ref_b, []).append(ref_a)

    components: list[PlacedComponent] = []
    for ref, _bbox, pc in fp_data:
        if ref in overlap_map:
            pc = PlacedComponent(
                ref=pc.ref,
                value=pc.value,
                footprint_id=pc.footprint_id,
                centroid=pc.centroid,
                origin=pc.origin,
                rotation=pc.rotation,
                bbox=pc.bbox,
                courtyard_size=pc.courtyard_size,
                nearest_edge=pc.nearest_edge,
                nearest_edge_distance_mm=pc.nearest_edge_distance_mm,
                mating_face=pc.mating_face,
                group_name=pc.group_name,
                zone_name=pc.zone_name,
                voltage_domain=pc.voltage_domain,
                subcircuit_types=pc.subcircuit_types,
                overlapping_refs=tuple(sorted(overlap_map[ref])),
            )
        components.append(pc)

    return tuple(sorted(components, key=lambda c: c.ref)), overlaps


def build_board_state(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    subcircuits: tuple[DetectedSubCircuit, ...] | None = None,
    domain_map: dict[str, VoltageDomain] | None = None,
    zones: Sequence[BoardZone] | None = None,
    groups: Sequence[PlacedGroup] | None = None,
    review: PlacementReview | None = None,
) -> BoardState:
    """Build a complete spatial snapshot of the board.

    Calls existing geometry functions — does NOT reimplement centroid/bbox math.
    O(N^2) for overlap detection; ~10ms for 80-component boards.
    """
    board = _board_bounds_from_outline(pcb)
    bx0, by0, bx1, by1 = board
    board_w = bx1 - bx0
    board_h = by1 - by0

    # Build lookup maps
    ref_to_group = _build_ref_to_group(requirements)
    ref_to_zone = _build_ref_to_zone(zones, groups)
    ref_to_domain = _build_ref_to_domain(subcircuits, domain_map)
    ref_to_subcircuits = _build_ref_to_subcircuits(subcircuits)

    # Phase 1: build PlacedComponent for each footprint (without overlaps yet)
    fp_data = _build_placed_components(
        pcb, board, ref_to_group, ref_to_zone, ref_to_domain, ref_to_subcircuits,
    )

    # Phase 2: detect AABB overlaps and rebuild components with overlap info
    components_t, overlaps = _detect_overlaps(fp_data)

    # Phase 3: edge violations
    edge_violations: list[EdgeViolation] = []
    off_board = 0
    for c in components_t:
        edges = _edge_distances(c.bbox, board)
        for edge_name, dist in edges.items():
            if dist < 0:
                edge_violations.append(
                    EdgeViolation(c.ref, edge_name, round(dist, 2), is_off_board=True)
                )
                off_board += 1
                break  # one violation per component is enough

    # Phase 4: isolation gaps, zone occupancy, group cohesion
    isolation_gaps = _compute_isolation_gaps(components_t)
    zone_occ = _compute_zone_occupancy(zones, components_t)
    group_coh = _compute_group_cohesion(components_t)

    # Total utilization
    total_comp_area = sum(c.courtyard_size[0] * c.courtyard_size[1] for c in components_t)
    board_area = board_w * board_h
    total_util = min(100.0, (total_comp_area / board_area) * 100.0) if board_area > 0 else 0.0

    # Review info
    grade = review.grade if review is not None else "?"
    violation_count = len(review.violations) if review is not None else 0

    return BoardState(
        board_bounds=board,
        board_width_mm=round(board_w, 1),
        board_height_mm=round(board_h, 1),
        components=components_t,
        overlaps=tuple(overlaps),
        edge_violations=tuple(edge_violations),
        isolation_gaps=isolation_gaps,
        zone_occupancy=zone_occ,
        group_cohesion=group_coh,
        total_utilization_pct=round(total_util, 1),
        off_board_count=off_board,
        review_grade=grade,
        review_violation_count=violation_count,
    )


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def _format_component_map(
    state: BoardState, cell_size_mm: float, lines: list[str],
) -> None:
    """Append ASCII component map grid to lines."""
    bb = state.board_bounds
    cols = max(1, math.ceil(state.board_width_mm / cell_size_mm))
    rows = max(1, math.ceil(state.board_height_mm / cell_size_mm))

    grid: list[list[list[str]]] = [[[] for _ in range(cols)] for _ in range(rows)]
    for c in state.components:
        cx, cy = c.centroid
        col = max(0, min(int((cx - bb[0]) / cell_size_mm), cols - 1))
        row = max(0, min(int((cy - bb[1]) / cell_size_mm), rows - 1))
        label = c.ref
        if c.mating_face:
            arrow = {"N": "\u2191", "S": "\u2193", "E": "\u2192", "W": "\u2190"}.get(
                c.mating_face, ""
            )
            label = f"{c.ref}{arrow}"
        grid[row][col].append(label)

    header_parts = ["     "]
    for col_idx in range(cols):
        header_parts.append(f"{int(bb[0] + col_idx * cell_size_mm):>5}")
    lines.append("".join(header_parts))

    for row_idx in range(rows):
        y_val = int(bb[1] + row_idx * cell_size_mm)
        row_parts = [f"{y_val:>4} "]
        for col_idx in range(cols):
            cell = grid[row_idx][col_idx]
            if not cell:
                row_parts.append("    .")
            elif len(cell) == 1:
                row_parts.append(f"{cell[0]:>5}")
            else:
                row_parts.append(f" {len(cell)}x  ")
        lines.append("".join(row_parts))
    lines.append("")


_EDGE_TO_EXPECTED_FACE = {"top": "N", "bottom": "S", "left": "W", "right": "E"}


def _format_connectors_table(
    connectors: list, lines: list[str],
) -> None:
    """Append connectors table to lines."""
    if not connectors:
        return
    lines.append("--- CONNECTORS ---")
    lines.append(f"{'Ref':<6}{'Edge':<8}{'Mating':<10}{'Dist':<8}Status")
    for c in sorted(connectors, key=lambda x: x.ref):
        arrow = _MATING_ARROWS.get(c.mating_face, "?")
        dist_str = f"{c.nearest_edge_distance_mm:.1f}mm"
        expected = _EDGE_TO_EXPECTED_FACE.get(c.nearest_edge, "")
        if c.nearest_edge_distance_mm <= 0.5 and c.mating_face == expected:
            status = "OK (flush)"
        elif c.mating_face == expected:
            status = "OK"
        elif c.mating_face:
            status = "\u2717 FACING INWARD"
        else:
            status = "?"
        lines.append(f"{c.ref:<6}{c.nearest_edge:<8}{arrow:<10}{dist_str:<8}{status}")
    lines.append("")


def _format_sections(state: BoardState, lines: list[str]) -> None:
    """Append RF, overlaps, zones, groups, and isolation sections."""
    # RF module
    rf_comps = [c for c in state.components if "rf_antenna" in c.subcircuit_types]
    if rf_comps:
        lines.append("--- RF MODULE ---")
        for c in rf_comps:
            if c.ref.startswith("U"):
                face_arrow = _MATING_ARROWS.get(c.mating_face, "")
                edge_ok = "\u2713" if c.nearest_edge_distance_mm < 5.0 else "\u2717"
                lines.append(
                    f"{c.ref} ({c.footprint_id}) antenna faces {face_arrow or '?'}, "
                    f"{c.nearest_edge_distance_mm:.1f}mm from {c.nearest_edge} edge {edge_ok}"
                )
        lines.append("")

    # Overlaps
    if state.overlaps:
        lines.append(f"--- OVERLAPS ({len(state.overlaps)}) ---")
        for ov in state.overlaps:
            lines.append(f"{ov.ref_a} \u2194 {ov.ref_b}  ~{ov.overlap_area_mm2:.1f}mm\u00b2")
        lines.append("")

    # Zones
    if state.zone_occupancy:
        lines.append("--- ZONES ---")
        lines.append(f"{'Zone':<14}{'Rect':<30}{'Comps':>6}{'Util':>6}")
        for z in state.zone_occupancy:
            r = z.rect
            rect_str = f"({r[0]:.0f},{r[1]:.0f},{r[2]:.0f},{r[3]:.0f})"
            util = f"{z.utilization_pct:.0f}%"
            lines.append(
                f"{z.zone_name:<14}{rect_str:<30}{z.component_count:>5} {util:>6}"
            )
        lines.append("")

    # Groups
    if state.group_cohesion:
        lines.append("--- GROUPS ---")
        lines.append(f"{'Group':<22}{'Refs':>5}{'Spread':>8}{'Density':>9}{'Zone':>8}")
        group_zones: dict[str, str] = {}
        for c in state.components:
            if c.group_name and c.zone_name:
                group_zones.setdefault(c.group_name, c.zone_name)
        for g in state.group_cohesion:
            zone = group_zones.get(g.group_name, "")
            lines.append(
                f"{g.group_name:<22}{g.ref_count:>5}{g.spread_mm:>7.0f}mm"
                f"{g.density_pct:>8.0f}%{zone:>8}"
            )
        lines.append("")

    # Voltage isolation
    if state.isolation_gaps:
        lines.append("--- VOLTAGE ISOLATION ---")
        for gap in state.isolation_gaps:
            ok = "\u2713" if gap.min_gap_mm >= 3.0 else "\u2717"
            lines.append(
                f"{gap.domain_a} \u2194 {gap.domain_b}  "
                f"{gap.min_gap_mm:.1f}mm ({gap.closest_pair[0]}\u2194{gap.closest_pair[1]}) {ok}"
            )
        lines.append("")


def _format_critical_issues(
    state: BoardState, connectors: list, lines: list[str],
) -> None:
    """Append critical issues section."""
    issues: list[str] = []
    for ov in state.overlaps:
        issues.append(f"[OVERLAP] {ov.ref_a} \u2194 {ov.ref_b}: courtyard collision")
    for ev in state.edge_violations:
        if ev.is_off_board:
            dist = abs(ev.clearance_mm)
            issues.append(
                f"[OFF-BOARD] {ev.ref}: extends past {ev.edge} edge by {dist:.1f}mm"
            )
    for c in sorted(connectors, key=lambda x: x.ref) if connectors else []:
        expected = _EDGE_TO_EXPECTED_FACE.get(c.nearest_edge, "")
        if c.mating_face and c.mating_face != expected:
            issues.append(f"[CONNECTOR] {c.ref}: mating face points INTO board")

    if issues:
        lines.append("--- CRITICAL ISSUES ---")
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. {issue}")
    else:
        lines.append("--- NO CRITICAL ISSUES ---")
    lines.append("")


def _format_report(state: BoardState, cell_size_mm: float) -> str:
    """Format BoardState as a structured text report."""
    lines: list[str] = []

    # Header
    lines.append("=== BOARD STATE REPORT ===")
    lines.append(
        f"Board: {state.board_width_mm}x{state.board_height_mm}mm "
        f"({round(state.board_width_mm * state.board_height_mm)}mm\u00b2) | "
        f"{len(state.components)} components | "
        f"{state.total_utilization_pct}% utilization"
    )
    lines.append(
        f"Review: Grade {state.review_grade} \u2014 "
        f"{state.review_violation_count} violations | "
        f"Off-board: {state.off_board_count} | "
        f"Overlaps: {len(state.overlaps)}"
    )
    lines.append("")

    # Component map
    lines.append(f"--- COMPONENT MAP ({cell_size_mm}mm cells) ---")
    _format_component_map(state, cell_size_mm, lines)

    # Connectors
    connectors = [c for c in state.components if c.ref.startswith("J")]
    _format_connectors_table(connectors, lines)

    # Sections: RF, overlaps, zones, groups, isolation
    _format_sections(state, lines)

    # Critical issues
    _format_critical_issues(state, connectors, lines)

    return "\n".join(lines)
