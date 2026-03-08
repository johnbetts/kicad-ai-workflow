"""Schematic component placement engine.

Assigns 2-D positions to schematic symbols by first grouping them into named
functional zones and then distributing them on a regular grid within each zone.

Zone layout (millimetres, A4 landscape 297x210, margins 30mm):

+-------------+-------------+
|  POWER      |   MCU       |
| (30,30)     | (155,30)    |
+-------------+-------------+
| ANALOG      | PERIPHERALS |
| (30,110)    | (155,110)   |
+-------------+-------------+
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kicad_pipeline.constants import SCHEMATIC_WIRE_GRID_MM
from kicad_pipeline.models.schematic import Point

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placement zone descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacementZone:
    """A named rectangular area in the schematic canvas.

    Attributes:
        name: Zone identifier, e.g. ``'POWER'``, ``'MCU'``.
        origin_x: Left edge of the zone in mm.
        origin_y: Top edge of the zone in mm.
        width: Zone width in mm.
        height: Zone height in mm.
    """

    name: str
    origin_x: float
    origin_y: float
    width: float
    height: float


# ---------------------------------------------------------------------------
# Standard schematic zone layout
# ---------------------------------------------------------------------------

SCHEMATIC_ZONES: dict[str, PlacementZone] = {
    "POWER": PlacementZone("POWER", 25.40, 27.94, 116.84, 69.85),
    "MCU": PlacementZone("MCU", 149.86, 27.94, 120.65, 69.85),
    "ANALOG": PlacementZone("ANALOG", 25.40, 106.68, 116.84, 69.85),
    "PERIPHERALS": PlacementZone("PERIPHERALS", 149.86, 106.68, 120.65, 69.85),
}
"""Standard schematic zones keyed by name (A4 layout).

A4 landscape (297x210mm) with 4 roughly equal quadrants:
- POWER: top-left (supply, regulation, bypass caps)
- MCU: top-right (ICs, DIP switches, pull-ups)
- ANALOG: bottom-left (sensors, voltage dividers, ADC channels)
- PERIPHERALS: bottom-right (large connectors, headers)
~8mm gap between rows to prevent label overlap.
"""

_A3_ZONES: dict[str, PlacementZone] = {
    "POWER": PlacementZone("POWER", 20.32, 20.32, 180.34, 119.38),
    "MCU": PlacementZone("MCU", 209.55, 20.32, 180.34, 119.38),
    "ANALOG": PlacementZone("ANALOG", 20.32, 149.86, 180.34, 119.38),
    "PERIPHERALS": PlacementZone("PERIPHERALS", 209.55, 149.86, 180.34, 119.38),
}
"""Schematic zones for A3 landscape page (420x297mm) with 20mm margins.

Four non-overlapping quadrants.  ETHERNET is removed (it shared
coordinates with PERIPHERALS).
"""


def zones_for_page(paper: str = "A4") -> dict[str, PlacementZone]:
    """Return the schematic zone layout for the given paper size.

    Args:
        paper: Paper size identifier (``"A4"`` or ``"A3"``).

    Returns:
        Zone dictionary for the given paper size.
    """
    if paper == "A3":
        return _A3_ZONES
    return SCHEMATIC_ZONES

# Feature-name keyword hints for preferred zone placement.
# Each keyword maps to a preferred zone slot index (0-3 in the 2x2 grid:
# 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right).
_ZONE_HINT_KEYWORDS: list[tuple[tuple[str, ...], int]] = [
    (("power", "supply", "regulator", "usb", "battery"), 0),  # top-left
    (("mcu", "core", "processor", "cpu", "adc", "dac", "ic"), 1),  # top-right
    (("analog", "sensor", "divider", "amplifier", "filter", "opamp"), 2),  # bottom-left
    (("connect", "terminal", "header", "interface", "periph", "io"), 3),  # bottom-right
    (("ethernet", "wifi", "radio", "rf", "bluetooth"), 3),  # bottom-right
]

# The 4 zone slot names in order (matches _ZONE_HINT_KEYWORDS indices)
_ZONE_SLOT_NAMES = ("POWER", "MCU", "ANALOG", "PERIPHERALS")

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _feature_to_slot(feature: str) -> int | None:
    """Return the preferred zone slot for a feature name, or None."""
    fl = feature.lower()
    for keywords, slot in _ZONE_HINT_KEYWORDS:
        if any(fl.startswith(kw) or kw in fl for kw in keywords):
            return slot
    return None


def assign_zones(
    components: list[tuple[str, str]],
    paper: str = "A4",
    adjacency: dict[str, set[str]] | None = None,
) -> dict[str, PlacementZone]:
    """Assign each component reference to a :class:`PlacementZone`.

    Groups components by feature name, then distributes feature groups
    across 4 non-overlapping zone slots.  Keyword hints guide placement
    (e.g. "Power" features go to the POWER zone), but unmatched features
    are distributed to remaining unused slots instead of all being
    crammed into PERIPHERALS.

    When a zone is overloaded (>8 refs) and another zone is empty,
    the overloaded zone is split using connectivity-aware ordering so
    tightly-connected subcircuits stay together.

    Args:
        components: List of ``(ref, feature_name)`` tuples.
        paper: Page size (``"A4"`` or ``"A3"``).
        adjacency: Optional connectivity map for smart splitting.

    Returns:
        Mapping from component ref to the assigned :class:`PlacementZone`.
    """
    active_zones = zones_for_page(paper)

    # Group refs by feature name (preserving order)
    feature_groups: dict[str, list[str]] = {}
    for ref, feature in components:
        feature_groups.setdefault(feature, []).append(ref)

    # Assign each feature group to a zone slot
    feature_to_zone: dict[str, str] = {}
    used_slots: set[int] = set()

    # First pass: features with keyword hints get their preferred slot
    for feature in feature_groups:
        slot = _feature_to_slot(feature)
        if slot is not None and slot not in used_slots:
            feature_to_zone[feature] = _ZONE_SLOT_NAMES[slot]
            used_slots.add(slot)

    # Second pass: distribute remaining features to unused slots
    # Prefer PERIPHERALS (3) first, then ANALOG (2), MCU (1), POWER (0)
    available_slots = [i for i in (3, 2, 1, 0) if i not in used_slots]
    for feature in feature_groups:
        if feature not in feature_to_zone:
            if available_slots:
                slot = available_slots.pop(0)
                feature_to_zone[feature] = _ZONE_SLOT_NAMES[slot]
                used_slots.add(slot)
            else:
                # More features than zones: pick the zone with fewest refs
                zone_counts = {z: 0 for z in _ZONE_SLOT_NAMES}
                for z in feature_to_zone.values():
                    zone_counts[z] = zone_counts.get(z, 0) + 1
                least_used = min(zone_counts, key=lambda z: zone_counts[z])
                feature_to_zone[feature] = least_used

    # Rebalance: if any zone is empty and another is overloaded, redistribute.
    zone_ref_counts: dict[str, int] = {z: 0 for z in _ZONE_SLOT_NAMES}
    for feature, refs in feature_groups.items():
        zone_ref_counts[feature_to_zone[feature]] += len(refs)

    empty_zones = [z for z in _ZONE_SLOT_NAMES if zone_ref_counts[z] == 0]
    ref_overrides: dict[str, str] = {}

    if empty_zones:
        overloaded = max(_ZONE_SLOT_NAMES, key=lambda z: zone_ref_counts[z])
        if zone_ref_counts[overloaded] > 12:
            # Collect all refs in the overloaded zone
            overloaded_refs: list[str] = []
            for feature, refs in feature_groups.items():
                if feature_to_zone[feature] == overloaded:
                    overloaded_refs.extend(refs)

            # Sort by connectivity so connected components are adjacent
            if adjacency is not None:
                overloaded_refs = _sort_by_connectivity(overloaded_refs, adjacency)

            # Split in half — second half goes to the empty zone.
            # Choose target zone that's adjacent (same column = above/below).
            # ANALOG(2) ↔ POWER(0) share left column
            # PERIPHERALS(3) ↔ MCU(1) share right column
            adjacent_map = {"ANALOG": "POWER", "POWER": "ANALOG",
                            "PERIPHERALS": "MCU", "MCU": "PERIPHERALS"}
            preferred_target = adjacent_map.get(overloaded, "")
            target_zone = preferred_target if preferred_target in empty_zones else empty_zones[0]

            split_point = len(overloaded_refs) // 2
            for ref in overloaded_refs[split_point:]:
                ref_overrides[ref] = target_zone
            log.debug(
                "assign_zones: split %d refs from %s to %s (connectivity-aware)",
                len(overloaded_refs) - split_point, overloaded, target_zone,
            )

    # Build result: ref -> PlacementZone
    result: dict[str, PlacementZone] = {}
    for feature, refs in feature_groups.items():
        zone_name = feature_to_zone[feature]
        for ref in refs:
            actual_zone = ref_overrides.get(ref, zone_name)
            result[ref] = active_zones[actual_zone]
            log.debug("assign_zones: %s (feature=%s) -> %s", ref, feature, actual_zone)

    return result


def _h_spacing_for_pins(pin_count: int, grid: float) -> float:
    """Compute horizontal spacing for a component based on pin count."""
    if pin_count > 10:
        raw = pin_count * 1.5 + 15.0
    elif pin_count > 4:
        raw = pin_count * 2.5 + 10.0
    else:
        # 2-pin passives: body ~9mm + labels ~10mm
        raw = 20.0
    return round(raw / grid) * grid


def place_in_zone(
    refs: list[str],
    zone: PlacementZone,
    grid: float = SCHEMATIC_WIRE_GRID_MM,
    symbols_per_row: int | None = None,
    pin_counts: list[int] | None = None,
) -> dict[str, Point]:
    """Place a list of component refs in a grid within *zone*.

    Large components (>10 pins) are placed first, each in their own row.
    Remaining small components are packed in a compact grid below.

    Args:
        refs: Ordered list of component reference designators to place.
        zone: The :class:`PlacementZone` that bounds the placement.
        grid: Grid size for snapping (default: ``SCHEMATIC_WIRE_GRID_MM``).
        symbols_per_row: Number of symbols per row.  If ``None``, derived
            from zone width / horizontal spacing (at least 1).
        pin_counts: Optional list of pin counts parallel to *refs*.
            When provided, vertical spacing adapts to symbol height.

    Returns:
        Mapping from component ref to grid-snapped :class:`Point`.
    """
    result: dict[str, Point] = {}

    if pin_counts is None or len(pin_counts) != len(refs):
        # No pin info: compact grid with tighter defaults
        h_spacing = snap_to_grid(20.32, grid)  # 16 * 1.27mm
        v_spacing = snap_to_grid(15.24, grid)  # 12 * 1.27mm
        if symbols_per_row is None:
            symbols_per_row = max(1, int(zone.width / h_spacing))
        for idx, ref in enumerate(refs):
            col = idx % symbols_per_row
            row = idx // symbols_per_row
            raw_x = zone.origin_x + col * h_spacing
            raw_y = zone.origin_y + row * v_spacing
            x = snap_to_grid(raw_x, grid)
            y = snap_to_grid(raw_y, grid)
            result[ref] = Point(x=x, y=y)
            log.debug("place_in_zone: %s → (%.3f, %.3f) in zone %s", ref, x, y, zone.name)
        return result

    # Separate large components (>=8 pins) from small ones
    large_threshold = 8
    large_indices = [i for i, pc in enumerate(pin_counts) if pc >= large_threshold]
    small_indices = [i for i, pc in enumerate(pin_counts) if pc < large_threshold]

    cumulative_y = 0.0

    # Place large components: use pin-count-aware horizontal spacing
    for row_start in range(0, len(large_indices), 2):
        row_items = large_indices[row_start:row_start + 2]
        # Compute per-row horizontal spacing from the largest component in this row
        row_max_pc = max(pin_counts[i] for i in row_items)
        large_h = max(snap_to_grid(_h_spacing_for_pins(row_max_pc, grid), grid), 25.4)
        max_row_h = 0.0
        for col_idx, i in enumerate(row_items):
            ref = refs[i]
            pc = pin_counts[i]
            x = snap_to_grid(zone.origin_x + col_idx * large_h, grid)
            y = snap_to_grid(zone.origin_y + cumulative_y, grid)
            result[ref] = Point(x=x, y=y)
            log.debug("place_in_zone: %s → (%.3f, %.3f) in zone %s [large]", ref, x, y, zone.name)
            effective_pins = pc // 2 if pc > 10 else pc
            body_h = max(effective_pins * 2.54, 5.08)
            body_h = min(body_h, zone.height * 0.5)
            max_row_h = max(max_row_h, body_h)
        cumulative_y += max_row_h + 8.0

    # Place small components: pin-count-aware spacing per row
    if small_indices:
        n_small = len(small_indices)

        # Compute per-row horizontal spacing from max pin count in first row
        first_row_pcs = [pin_counts[small_indices[j]] for j in range(min(n_small, 6))]
        max_pc_first = max(first_row_pcs) if first_row_pcs else 2
        h_spacing = snap_to_grid(_h_spacing_for_pins(max_pc_first, grid), grid)
        cols = max(1, min(n_small, int(zone.width / h_spacing)))

        for local_idx, global_idx in enumerate(small_indices):
            ref = refs[global_idx]
            pc = pin_counts[global_idx]
            col = local_idx % cols
            row = local_idx // cols

            # Compute row height: body height + size-scaled label clearance
            row_start_idx = row * cols
            row_end_idx = min(row_start_idx + cols, n_small)
            row_pcs = [pin_counts[small_indices[j]] for j in range(row_start_idx, row_end_idx)]
            max_pc = max(row_pcs) if row_pcs else 2
            body_h = max(max_pc * 2.54, 5.08)
            # Scale label clearance by body size
            if body_h <= 5.08:
                clearance = 10.0  # 2-pin passives: tight
            elif body_h <= 10.0:
                clearance = 12.0  # 4-6 pin: moderate
            else:
                clearance = 15.0  # 8+ pin: generous
            row_height = body_h + clearance

            # Recompute h_spacing per row from max pin count
            row_h_spacing = snap_to_grid(_h_spacing_for_pins(max_pc, grid), grid)
            raw_x = zone.origin_x + col * row_h_spacing
            if col == 0 and local_idx > 0:
                cumulative_y += row_height
            raw_y = zone.origin_y + cumulative_y

            x = snap_to_grid(raw_x, grid)
            y = snap_to_grid(raw_y, grid)
            result[ref] = Point(x=x, y=y)
            log.debug("place_in_zone: %s → (%.3f, %.3f) in zone %s", ref, x, y, zone.name)

    return result


def _sort_by_connectivity(
    refs: list[str],
    adjacency: dict[str, set[str]],
) -> list[str]:
    """Sort refs so connected components are adjacent (greedy nearest-neighbor).

    Starts with the ref that has the most connections to other zone members,
    then greedily picks the most-connected remaining neighbor.
    """
    if len(refs) <= 2:
        return refs
    ref_set = set(refs)

    # Seed with an input connector (J ref) if available — they represent
    # signal entry points and give natural left-to-right signal flow.
    # Use (count, ref_name) tuples as sort keys for deterministic tie-breaking.
    j_refs = sorted(r for r in refs if r.startswith("J"))
    if j_refs:
        seed = min(j_refs, key=lambda r: (len(adjacency.get(r, set()) & ref_set), r))
    else:
        seed = max(refs, key=lambda r: (len(adjacency.get(r, set()) & ref_set), r))

    result: list[str] = [seed]
    remaining = set(refs) - {seed}
    while remaining:
        current = result[-1]
        # Find the remaining ref most connected to current
        nbrs = adjacency.get(current, set()) & remaining
        if nbrs:
            nxt = max(nbrs, key=lambda r: (len(adjacency.get(r, set()) & ref_set), r))
        else:
            # Chain break — start a new chain from a remaining connector
            # if available, to preserve signal-flow ordering (alphabetical)
            remaining_j = sorted(r for r in remaining if r.startswith("J"))
            if remaining_j:
                nxt = min(
                    remaining_j,
                    key=lambda r: (len(adjacency.get(r, set()) & ref_set), r),
                )
            else:
                # No connectors left — pick ref with fewest connections
                # (leaf node) to start a clean chain
                nxt = min(
                    sorted(remaining),
                    key=lambda r: (len(adjacency.get(r, set()) & ref_set), r),
                )
        result.append(nxt)
        remaining.remove(nxt)
    return result


def layout_schematic(
    symbol_refs: list[str],
    feature_map: dict[str, str],
    pin_count_map: dict[str, int] | None = None,
    paper: str = "A4",
    adjacency: dict[str, set[str]] | None = None,
) -> dict[str, Point]:
    """Compute the full schematic layout for all symbol references.

    Assigns each ref to a zone using :func:`assign_zones`, then calls
    :func:`place_in_zone` for each zone group.

    Args:
        symbol_refs: All component reference designators to place.
        feature_map: Mapping from ref to feature name, e.g.
            ``{'U1': 'MCU', 'C1': 'Power'}``.  Refs missing from the map
            are assigned to the ``'PERIPHERALS'`` zone.
        pin_count_map: Optional mapping from ref to number of pins.
            When provided, vertical spacing adapts to symbol height.

    Returns:
        Mapping from component ref to :class:`Point` for every ref in
        *symbol_refs*.
    """
    active_zones = zones_for_page(paper)

    # Build (ref, feature) list — fall back to 'Peripherals' for unknowns
    tagged = [(ref, feature_map.get(ref, "Peripherals")) for ref in symbol_refs]
    zone_map = assign_zones(tagged, paper=paper, adjacency=adjacency)

    # Group refs by zone
    groups: dict[str, list[str]] = {}
    for ref, zone in zone_map.items():
        groups.setdefault(zone.name, []).append(ref)

    positions: dict[str, Point] = {}
    for zone_name, zone_refs in groups.items():
        zone = active_zones[zone_name]
        # Sort refs by connectivity so connected components are adjacent
        if adjacency is not None:
            zone_refs = _sort_by_connectivity(zone_refs, adjacency)
        zone_pin_counts: list[int] | None = None
        if pin_count_map is not None:
            zone_pin_counts = [pin_count_map.get(ref, 2) for ref in zone_refs]
        zone_positions = place_in_zone(zone_refs, zone, pin_counts=zone_pin_counts)
        positions.update(zone_positions)

    # Any refs not covered (shouldn't happen, but be safe)
    unplaced = [ref for ref in symbol_refs if ref not in positions]
    if unplaced:
        log.warning("layout_schematic: %d refs not placed; adding to PERIPHERALS", len(unplaced))
        fallback_zone = active_zones["PERIPHERALS"]
        fallback_positions = place_in_zone(unplaced, fallback_zone)
        positions.update(fallback_positions)

    return positions


# ---------------------------------------------------------------------------
# Grid snap utility (also used by wiring.py)
# ---------------------------------------------------------------------------


def snap_to_grid(value: float, grid: float = SCHEMATIC_WIRE_GRID_MM) -> float:
    """Round *value* to the nearest multiple of *grid*.

    Args:
        value: Coordinate value in mm.
        grid: Grid size in mm (default ``SCHEMATIC_WIRE_GRID_MM``).

    Returns:
        Grid-snapped coordinate.
    """
    return round(value / grid) * grid
