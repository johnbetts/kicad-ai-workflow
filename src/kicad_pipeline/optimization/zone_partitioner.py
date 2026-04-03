"""Board zone partitioner for hierarchical placement.

Partitions the PCB board area into non-overlapping rectangular zones based
on FeatureBlock functional groups. Each zone is assigned to one or more
groups using keyword matching, enabling top-down placement where groups
are placed as rigid units within their zones.

This is Level 1 of the 3-level hierarchical placement engine.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import (
        FeatureBlock,
        ProjectRequirements,
    )
    from kicad_pipeline.optimization.functional_grouper import PowerFlowTopology

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Component area classification for bottom-up zone sizing
# ---------------------------------------------------------------------------

# Estimated footprint area (mm²) per component by reference designator prefix.
# Categories: SMD passives ≈ 4mm², THT connectors ≈ 50mm²,
# ICs/relays ≈ 100mm², modules ≈ 200mm².
_AREA_BY_REF_PREFIX: dict[str, float] = {
    "R": 4.0,    # SMD resistor
    "C": 4.0,    # SMD capacitor
    "L": 4.0,    # SMD inductor / ferrite bead
    "F": 4.0,    # ferrite bead
    "D": 4.0,    # SMD diode / Zener
    "LED": 4.0,  # discrete LED (must check before "L")
    "Q": 9.0,    # SMD transistor (SOT-23 etc.)
    "T": 9.0,    # transistor variant
    "Y": 9.0,    # crystal / oscillator
    "SW": 16.0,  # SMD switch
    "TP": 1.0,   # test point (negligible area)
    "H": 1.0,    # mounting hole
    "MH": 1.0,   # mounting hole variant
    "J": 50.0,   # connector (THT)
    "P": 50.0,   # connector (THT)
    "CN": 50.0,  # connector
    "TB": 50.0,  # terminal block (THT)
    "K": 100.0,  # relay
    "U": 100.0,  # IC (LDO, MCU, Ethernet PHY, etc.)
}

# When no prefix matches, assume SMD passive.
_DEFAULT_COMPONENT_AREA_MM2: float = 4.0

# Spacing density factor: multiply raw component area to account for
# courtyard clearances, routing channels, and component-to-component gaps.
# At 5×, a group with 1000mm² of footprint area gets a 5000mm² zone,
# which is ~70×70mm — enough for routing channels around clustered components.
_DENSITY_FACTOR: float = 5.0

# Minimum zone dimension (mm) — prevents zones from collapsing to zero.
_MIN_ZONE_DIM_MM: float = 15.0

def _component_area_mm2(ref: str) -> float:
    """Return estimated footprint area (mm²) for a component reference.

    Classification uses the leading alphabetic prefix of the ref designator:
    ``R``, ``C``, ``L``, ``D`` → SMD passive (~4mm²);
    ``J``, ``P``, ``TB`` → THT connector (~50mm²);
    ``K``, ``U`` → relay or IC (~100mm²).
    Multi-letter prefixes (``LED``, ``MH``, ``TB``, ``SW``, ``CN``) are checked
    before single-letter ones to avoid partial matches.
    """
    prefix = ""
    for ch in ref:
        if ch.isalpha():
            prefix += ch
        else:
            break
    # Check longest prefix first for correct disambiguation (e.g. "LED" vs "L").
    for length in (3, 2, 1):
        if len(prefix) >= length:
            candidate = prefix[:length].upper()
            if candidate in _AREA_BY_REF_PREFIX:
                return _AREA_BY_REF_PREFIX[candidate]
    return _DEFAULT_COMPONENT_AREA_MM2


def _zone_footprint_area_mm2(refs: tuple[str, ...]) -> float:
    """Return the total zone area (mm²) needed for a set of component refs.

    Sums per-component footprint areas and applies the density factor to
    account for courtyard clearances and component spacing.
    """
    raw = sum(_component_area_mm2(r) for r in refs)
    return raw * _DENSITY_FACTOR


def _zone_footprint_area_from_requirements(
    refs: tuple[str, ...],
    ref_to_footprint: dict[str, str],
) -> float:
    """Compute zone area from actual footprint sizes (not ref-prefix estimates).

    Uses ``estimate_footprint_size()`` to get real (width, height) per component
    from its footprint ID, then computes the needed zone area as::

        max(sum_of_areas × density_factor,
            largest_side² × 4)

    The second term ensures the zone is large enough for the physically largest
    component (e.g., ESP32 at 26mm or relays at 20mm) plus surrounding passives.
    """
    from kicad_pipeline.pcb.footprints import estimate_footprint_size

    total = 0.0
    max_w = 0.0
    max_h = 0.0
    for ref in refs:
        fp_id = ref_to_footprint.get(ref)
        if fp_id:
            w, h = estimate_footprint_size(fp_id)
            total += w * h
            # Track the largest individual component dimensions.
            if w * h > max_w * max_h:
                max_w, max_h = w, h
        else:
            total += _component_area_mm2(ref)
    # Area from density factor
    area_from_density = total * _DENSITY_FACTOR
    # Area from largest component — use actual aspect ratio, not square.
    # A 36mm pin header needs a 36×10mm zone, not a 90×90mm zone.
    # Formula: (long_side + margin) × (short_side × 3) where margin accounts
    # for passives alongside the connector/IC, and ×3 accounts for components
    # on both sides plus routing.
    if max_w > 5.0 and max_h > 5.0:
        long_side = max(max_w, max_h)
        short_side = min(max_w, max_h)
        area_from_largest = (long_side + 10.0) * (short_side * 3.0)
    else:
        area_from_largest = 0.0
    return max(area_from_density, area_from_largest)


# ---------------------------------------------------------------------------
# Zone keyword mapping — maps zone names to keywords found in FeatureBlock names
# ---------------------------------------------------------------------------

_ZONE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "input_connectors": ("24v", "input connector", "harness"),
    "power": ("power", "supply", "regulator", "buck", "ldo"),
    "relay": ("relay", "output", "switching"),
    "analog": ("analog", "adc", "input", "sensor"),
    "mcu": ("mcu", "micro", "esp", "cpu", "peripheral"),
    "ethernet": ("ethernet", "eth", "poe", "network"),
    "display": ("display", "lcd", "oled", "screen"),
}

# Default zone layout proportions (fraction of board) — derived from
# reference board analysis.  Format: (x_start, y_start, x_end, y_end)
# as fractions of board width/height.
_DEFAULT_ZONE_FRACTIONS: dict[str, tuple[float, float, float, float]] = {
    # 3-row layout (160x80mm board) — Run 3 config (best: 680 crossings):
    #   Row 0: input_connectors (full width, 18% height)
    #   Row 1: power (left) + relay (right)
    #   Row 2: analog (left) + ethernet (center) + mcu (right)
    "input_connectors": (0.00, 0.00, 1.00, 0.18),
    "power":            (0.00, 0.18, 0.35, 0.55),
    "relay":            (0.35, 0.18, 1.00, 0.55),
    "analog":           (0.00, 0.55, 0.30, 0.90),
    "mcu":              (0.55, 0.55, 1.00, 0.90),
    "ethernet":         (0.30, 0.55, 0.55, 0.90),
}

# Minimum inter-zone gap (mm)
_ZONE_GAP_MM: float = 5.0


@dataclass(frozen=True)
class BoardZone:
    """A rectangular zone on the board assigned to one or more feature groups.

    Attributes:
        name: Zone identifier (e.g. "power", "relay", "mcu").
        rect: Absolute board coordinates (x_min, y_min, x_max, y_max) in mm.
        edge_affinity: Preferred board edge ("top", "bottom", "left", "right")
            or None if no edge preference.
        groups: FeatureBlock names assigned to this zone.
    """

    name: str
    rect: tuple[float, float, float, float]
    edge_affinity: str | None
    groups: tuple[str, ...]


def _match_group_to_zone(group_name: str) -> str:
    """Map a FeatureBlock name to a zone name using keyword matching.

    Returns the zone name with the best keyword match, or "mcu" as fallback
    (MCU zone is the general-purpose catch-all).
    """
    lower = group_name.lower()
    for zone_name, keywords in _ZONE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return zone_name
    return "mcu"  # fallback


def _edge_affinity_for_zone(zone_name: str) -> str | None:
    """Return the preferred board edge for a zone, or None."""
    affinities: dict[str, str] = {
        "power": "top",
        "relay": "right",
        "ethernet": "top",
        "display": "left",
    }
    return affinities.get(zone_name)


def _reorder_rows_by_traffic(
    row_definitions: list[tuple[str, list[str]]],
    zone_groups: dict[str, list[str]],
    requirements: ProjectRequirements,
    fixed_rows: set[str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Reorder zones within each row by inter-group net traffic.

    For each multi-zone row, sort zones so that the zone with the most
    inter-group nets to zones in the row above is placed on the left
    (closest to the high-traffic side).  This minimizes crossing distance
    for the most-connected group pairs.
    """
    from collections import defaultdict

    # Build ref → group name map
    ref_to_group: dict[str, str] = {}
    for feat in requirements.features:
        for comp in feat.components:
            ref_to_group[comp.ref if hasattr(comp, "ref") else comp] = feat.name

    # Build group → zone name map
    group_to_zone: dict[str, str] = {}
    for zn, gnames in zone_groups.items():
        for gn in gnames:
            group_to_zone[gn] = zn

    # Build inter-zone traffic matrix
    power_nets = {"GND", "+5V", "+3V3", "+24V", "+12V", "VCC", "VBUS", ""}
    zone_traffic: dict[tuple[str, str], int] = defaultdict(int)
    for net in requirements.nets:
        if net.name.upper() in power_nets or not net.connections:
            continue
        zones_in_net: set[str] = set()
        for conn in net.connections:
            g = ref_to_group.get(conn.ref)
            if g:
                z = group_to_zone.get(g)
                if z:
                    zones_in_net.add(z)
        zlist = sorted(zones_in_net)
        for i in range(len(zlist)):
            for j in range(i + 1, len(zlist)):
                zone_traffic[(zlist[i], zlist[j])] += 1
                zone_traffic[(zlist[j], zlist[i])] += 1

    # For each non-fixed row with >1 zone, sort by traffic to the row above.
    _fixed = fixed_rows or set()
    result: list[tuple[str, list[str]]] = []
    prev_row_zones: set[str] = set()
    for row_name, row_zones in row_definitions:
        active = [z for z in row_zones if z in zone_groups]
        if len(active) > 1 and prev_row_zones and row_name not in _fixed:
            # Score each zone by its traffic to the previous row's zones
            def _traffic_to_prev(z: str) -> int:
                return sum(
                    zone_traffic.get((z, pz), 0) for pz in prev_row_zones
                )
            active.sort(key=_traffic_to_prev, reverse=True)
            _log.debug(
                "  Row %s reordered by traffic: %s",
                row_name,
                [(z, _traffic_to_prev(z)) for z in active],
            )
        result.append((row_name, active if active else row_zones))
        prev_row_zones = set(active) if active else set(row_zones)
    return result


def partition_board(
    board_bounds: tuple[float, float, float, float],
    groups: list[FeatureBlock],
    topology: PowerFlowTopology | None = None,
    requirements: ProjectRequirements | None = None,
) -> list[BoardZone]:
    """Partition board into non-overlapping rectangular zones.

    Strategy:
    1. Map each FeatureBlock to a zone by keyword matching on name.
    2. Compute needed footprint area per zone bottom-up from actual footprint
       sizes (via ``estimate_footprint_size()``) when *requirements* is provided,
       falling back to ref-prefix heuristics otherwise.
    3. Use adaptive row heights: row heights are proportional to the total
       footprint area of zones in that row, not fixed fractions.
    4. Within each row, distribute widths proportionally to zone footprint area.
    5. Enforce a minimum zone dimension of 15mm.
    6. Ensure inter-zone gaps of ``_ZONE_GAP_MM``.
    7. Only create zones that have at least one assigned group.

    Args:
        board_bounds: (min_x, min_y, max_x, max_y) in mm.
        groups: FeatureBlock instances to partition.
        topology: Optional power flow topology for domain ordering
            (reserved for future use).
        requirements: Optional full project requirements for accurate footprint
            size lookup. When provided, uses ``estimate_footprint_size()`` on
            each component's footprint ID instead of ref-prefix heuristics.

    Returns:
        List of BoardZone instances with absolute board coordinates.
    """
    bx1, by1, bx2, by2 = board_bounds
    board_w = bx2 - bx1
    board_h = by2 - by1

    if not groups:
        return []

    # Build ref → footprint_id map from requirements (for accurate sizing).
    ref_to_footprint: dict[str, str] = {}
    if requirements is not None:
        for comp in requirements.components:
            ref_to_footprint[comp.ref] = comp.footprint

    # Step 1: Map groups to zones and compute needed footprint area bottom-up.
    zone_groups: dict[str, list[str]] = {}
    zone_component_count: dict[str, int] = {}
    zone_footprint_area: dict[str, float] = {}
    # Minimum zone height needed: the short side of the tallest-area component
    # (components can be rotated, so min(w,h) determines the height needed).
    zone_min_height: dict[str, float] = {}
    for group in groups:
        zone_name = _match_group_to_zone(group.name)
        zone_groups.setdefault(zone_name, []).append(group.name)
        zone_component_count[zone_name] = (
            zone_component_count.get(zone_name, 0) + len(group.components)
        )
        # Use accurate footprint sizes when requirements available.
        if ref_to_footprint:
            area = _zone_footprint_area_from_requirements(
                group.components, ref_to_footprint,
            )
            # Track the short side of the largest component in this zone.
            # Components can be rotated, so min(w,h) is the minimum zone
            # height needed to fit any single component.
            from kicad_pipeline.pcb.footprints import estimate_footprint_size
            for ref in group.components:
                fp_id = ref_to_footprint.get(ref)
                if fp_id:
                    w, h = estimate_footprint_size(fp_id)
                    short_side = min(w, h)
                    zone_min_height[zone_name] = max(
                        zone_min_height.get(zone_name, 0.0), short_side,
                    )
        else:
            area = _zone_footprint_area_mm2(group.components)
        zone_footprint_area[zone_name] = (
            zone_footprint_area.get(zone_name, 0.0) + area
        )

    if not zone_groups:
        return []

    _log.info(
        "Zone partitioning: %d groups → %d zones",
        len(groups),
        len(zone_groups),
    )
    for zn, gnames in zone_groups.items():
        _log.info(
            "  Zone '%s': %s (%d components, %.0fmm² needed)",
            zn,
            gnames,
            zone_component_count.get(zn, 0),
            zone_footprint_area.get(zn, 0.0),
        )

    # Step 2: Compute zone rects using fully area-proportional allocation.
    #
    # Row heights AND column widths are both proportional to zone content.
    # This replaces the old fixed-fraction layout that produced 25mm-tall
    # strips regardless of content.
    half_gap = _ZONE_GAP_MM / 2.0

    # When there's only one zone, give it the full board area
    single_zone = len(zone_groups) == 1

    # 3-row layout preserves the connector row for edge-pinned terminals.
    # Row order is fixed — L3 phases have implicit dependencies on zone positions.
    # Traffic analysis (Run 3, iter 26-28) confirmed: reordering rows makes
    # crossings WORSE because L3 relay/MCU/ethernet phases assume specific
    # zone positions. The existing order (power-relay, analog-ethernet-mcu)
    # already satisfies the strongest adjacency (ethernet↔mcu = 8 nets).
    row_definitions: list[tuple[str, list[str]]] = [
        ("row0", ["input_connectors"]),
        ("row1", ["power", "relay"]),
        ("row2", ["analog", "ethernet", "mcu"]),
    ]

    # Compute which rows are active and their total areas.
    active_rows: list[tuple[str, list[str], float, float]] = []
    for row_name, row_zone_names in row_definitions:
        active = [z for z in row_zone_names if z in zone_groups]
        if not active:
            continue
        row_area = sum(zone_footprint_area.get(z, 0.0) for z in active)
        # Minimum row height: short side of tallest component × 2.5
        # (component height + clearance + surrounding passives).
        # Uses min(w,h) since components can be rotated.
        row_min_h_mm = max(
            zone_min_height.get(z, 0.0) for z in active
        ) * 2.5 + _ZONE_GAP_MM
        active_rows.append(
            (row_name, active, max(row_area, 1.0), row_min_h_mm),
        )

    # Distribute row heights: first satisfy minimums, then distribute
    # remaining space proportionally to area.
    total_min_h = sum(min_h for _, _, _, min_h in active_rows)
    total_row_area = sum(ra for _, _, ra, _ in active_rows) or 1.0

    # If minimums exceed board height, scale them down proportionally.
    if total_min_h > board_h:
        scale = board_h / total_min_h
        active_rows = [
            (n, z, a, m * scale) for n, z, a, m in active_rows
        ]
        total_min_h = board_h

    remaining_h = board_h - total_min_h

    # Build fraction rects for each zone.
    adjusted_fracs: dict[str, tuple[float, float, float, float]] = {}
    y_cursor = 0.0
    for _row_name, row_zones, row_area, row_min_h in active_rows:
        # Base height from minimum + proportional share of remaining space.
        row_h_mm = row_min_h + remaining_h * (row_area / total_row_area)
        row_h_frac = row_h_mm / board_h
        y_start = y_cursor
        y_end = y_cursor + row_h_frac
        y_cursor = y_end

        if len(row_zones) == 1:
            adjusted_fracs[row_zones[0]] = (0.0, y_start, 1.0, y_end)
            continue

        # Multiple zones in row — distribute width proportionally to area.
        areas = [
            max(zone_footprint_area.get(z, 1.0), 1.0) for z in row_zones
        ]
        total_area = sum(areas) or 1.0
        min_x_frac = 0.12
        remaining_x = 1.0 - min_x_frac * len(row_zones)
        x_cursor = 0.0
        for i, z in enumerate(row_zones):
            frac = min_x_frac + remaining_x * (areas[i] / total_area)
            adjusted_fracs[z] = (x_cursor, y_start, x_cursor + frac, y_end)
            x_cursor += frac

    board_area = board_w * board_h

    zones: list[BoardZone] = []
    for zone_name, group_names in zone_groups.items():
        if single_zone:
            fracs = (0.0, 0.0, 1.0, 1.0)
        elif zone_name in adjusted_fracs:
            fracs = adjusted_fracs[zone_name]
        else:
            fracs = _DEFAULT_ZONE_FRACTIONS.get(zone_name)
            if fracs is None:
                fracs = (0.30, 0.30, 0.70, 0.70)

        fx1, fy1, fx2, fy2 = fracs

        zx1 = max(0.0, fx1)
        zy1 = max(0.0, fy1)
        zx2 = min(1.0, fx2)
        zy2 = min(1.0, fy2)

        # Convert fractions to absolute coordinates with gap inset
        abs_x1 = bx1 + zx1 * board_w + half_gap
        abs_y1 = by1 + zy1 * board_h + half_gap
        abs_x2 = bx1 + zx2 * board_w - half_gap
        abs_y2 = by1 + zy2 * board_h - half_gap

        # Enforce minimum zone dimension — prevents collapsed zones.
        if abs_x2 - abs_x1 < _MIN_ZONE_DIM_MM:
            abs_x2 = abs_x1 + _MIN_ZONE_DIM_MM
        if abs_y2 - abs_y1 < _MIN_ZONE_DIM_MM:
            abs_y2 = abs_y1 + _MIN_ZONE_DIM_MM

        # Warn when the zone's needed footprint area exceeds its allocated space.
        zone_area_allocated = (abs_x2 - abs_x1) * (abs_y2 - abs_y1)
        zone_area_needed = zone_footprint_area.get(zone_name, 0.0)
        if zone_area_needed > zone_area_allocated:
            board_area_needed_total = sum(zone_footprint_area.values())
            suggested_scale = (board_area_needed_total / board_area) ** 0.5
            _log.warning(
                "Zone '%s' needs %.0fmm² but only %.0fmm² allocated "
                "(%.0f%% overflow). Groups: %s. "
                "Board may be undersized — try scaling by %.1fx.",
                zone_name,
                zone_area_needed,
                zone_area_allocated,
                100.0 * (zone_area_needed / zone_area_allocated - 1.0),
                ", ".join(group_names),
                suggested_scale,
            )

        zones.append(BoardZone(
            name=zone_name,
            rect=(abs_x1, abs_y1, abs_x2, abs_y2),
            edge_affinity=_edge_affinity_for_zone(zone_name),
            groups=tuple(sorted(group_names)),
        ))

    _log.info("Zone partitioning complete: %d zones", len(zones))
    return zones


def zone_for_group(
    group_name: str,
    zones: list[BoardZone],
) -> BoardZone | None:
    """Find the zone containing a given group name."""
    for zone in zones:
        if group_name in zone.groups:
            return zone
    return None


def zone_center(zone: BoardZone) -> tuple[float, float]:
    """Return the center point of a zone."""
    x1, y1, x2, y2 = zone.rect
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
