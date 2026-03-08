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

from kicad_pipeline.constants import (
    SCHEMATIC_LABEL_CHAR_WIDTH_MM,
    SCHEMATIC_MAX_LABEL_CHARS,
    SCHEMATIC_SYMBOL_GAP_MM,
    SCHEMATIC_WIRE_GRID_MM,
)
from kicad_pipeline.models.schematic import LibRectangle, LibSymbol, Point

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


@dataclass(frozen=True)
class SymbolExtent:
    """Full visual extent of a placed symbol from its centre origin (mm).

    Each field is a positive distance from the symbol origin in that direction.
    The total width is ``left + right``; total height is ``top + bottom``.

    Attributes:
        left: Leftward extent (positive = extends left of origin).
        right: Rightward extent.
        top: Upward extent (positive = extends above origin in schematic Y-down coords).
        bottom: Downward extent.
    """

    left: float
    right: float
    top: float
    bottom: float

    @property
    def width(self) -> float:
        """Total horizontal span (mm)."""
        return self.left + self.right

    @property
    def height(self) -> float:
        """Total vertical span (mm)."""
        return self.top + self.bottom


def compute_symbol_extent(
    lib_sym: LibSymbol,
    ref_text: str,
    value_text: str,
) -> SymbolExtent:
    """Compute the full visual extent of a symbol from its centre origin.

    Takes into account:
    - Body rectangle dimensions
    - Pin tip positions + wire stub length (7.62mm)
    - Net label text width estimate on sides with pins
    - Reference / value label clearance above / below
    - Power pin extra clearance for power symbol body

    Args:
        lib_sym: The library symbol definition.
        ref_text: Reference designator text (e.g. ``"R1"``).
        value_text: Component value text (e.g. ``"10k"``).

    Returns:
        A :class:`SymbolExtent` describing the full visual bounding box.
    """
    # --- Body extent from LibRectangle shapes ---
    body_half_w: float = 0.0
    body_half_h: float = 0.0
    for shape in lib_sym.shapes:
        if isinstance(shape, LibRectangle):
            body_half_w = max(
                body_half_w,
                abs(shape.start.x),
                abs(shape.end.x),
            )
            body_half_h = max(
                body_half_h,
                abs(shape.start.y),
                abs(shape.end.y),
            )

    # --- Pin tip positions per side (in lib-symbol coords: Y-up) ---
    has_left = False
    has_right = False
    has_top = False
    has_bottom = False
    max_pin_left: float = 0.0
    max_pin_right: float = 0.0
    max_pin_top: float = 0.0
    max_pin_bottom: float = 0.0
    has_power_top = False
    has_power_bottom = False

    for pin in lib_sym.pins:
        rot = pin.rotation % 360.0
        tip_x = pin.at.x
        tip_y = pin.at.y
        is_power = pin.pin_type in ("power_in", "power_out")

        if abs(rot) < 1.0:
            # Left-side pin (extends right from negative X)
            has_left = True
            max_pin_left = max(max_pin_left, abs(tip_x))
        elif abs(rot - 180.0) < 1.0:
            # Right-side pin
            has_right = True
            max_pin_right = max(max_pin_right, abs(tip_x))
        elif abs(rot - 270.0) < 1.0:
            # Top pin
            has_top = True
            max_pin_top = max(max_pin_top, abs(tip_y))
            if is_power:
                has_power_top = True
        elif abs(rot - 90.0) < 1.0:
            # Bottom pin
            has_bottom = True
            max_pin_bottom = max(max_pin_bottom, abs(tip_y))
            if is_power:
                has_power_bottom = True

    # --- Wire stub + label text on sides with pins ---
    wire_stub = 7.62  # mm (standard wire stub from pin tip)
    label_chars = max(len(ref_text), len(value_text), SCHEMATIC_MAX_LABEL_CHARS)
    label_width = label_chars * SCHEMATIC_LABEL_CHAR_WIDTH_MM

    left_extent = max_pin_left if has_left else body_half_w
    right_extent = max_pin_right if has_right else body_half_w

    if has_left:
        left_extent += wire_stub + label_width
    if has_right:
        right_extent += wire_stub + label_width

    # --- Vertical: body + ref/value labels ---
    # In schematic coords (Y-down), top extent = body_half_h + ref label clearance
    ref_label_clearance = 2.54  # mm above body for ref designator
    value_label_clearance = 2.54  # mm below body for value

    top_extent = body_half_h + ref_label_clearance
    bottom_extent = body_half_h + value_label_clearance

    # Power pins extend vertically: pin tip + wire stub + power symbol body (~5mm)
    power_extra = wire_stub + 5.08  # wire stub + power symbol body
    if has_power_top:
        top_extent = max(top_extent, max_pin_top + power_extra)
    if has_power_bottom:
        bottom_extent = max(bottom_extent, max_pin_bottom + power_extra)

    # Top/bottom pins also extend vertically (pin tip already included)
    if has_top:
        top_extent = max(top_extent, max_pin_top + wire_stub)
    if has_bottom:
        bottom_extent = max(bottom_extent, max_pin_bottom + wire_stub)

    return SymbolExtent(
        left=left_extent,
        right=right_extent,
        top=top_extent,
        bottom=bottom_extent,
    )


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


def _extent_h_spacing(
    ext_i: SymbolExtent,
    ext_j: SymbolExtent,
    gap: float = SCHEMATIC_SYMBOL_GAP_MM,
) -> float:
    """Horizontal centre-to-centre distance so extents *i* and *j* don't overlap."""
    return ext_i.right + ext_j.left + gap


def _extent_v_spacing(
    row_bottoms: float,
    next_tops: float,
    gap: float = SCHEMATIC_SYMBOL_GAP_MM,
) -> float:
    """Vertical centre-to-centre distance between two rows."""
    return row_bottoms + next_tops + gap


def place_in_zone(
    refs: list[str],
    zone: PlacementZone,
    grid: float = SCHEMATIC_WIRE_GRID_MM,
    symbols_per_row: int | None = None,
    pin_counts: list[int] | None = None,
    symbol_extents: dict[str, SymbolExtent] | None = None,
) -> dict[str, Point]:
    """Place a list of component refs in a grid within *zone*.

    Large components (>10 pins) are placed first, each in their own row.
    Remaining small components are packed in a compact grid below.

    When *symbol_extents* is provided, horizontal and vertical spacing is
    derived from the actual visual extent of each symbol rather than from
    pin-count heuristics, eliminating overlaps.

    Args:
        refs: Ordered list of component reference designators to place.
        zone: The :class:`PlacementZone` that bounds the placement.
        grid: Grid size for snapping (default: ``SCHEMATIC_WIRE_GRID_MM``).
        symbols_per_row: Number of symbols per row.  If ``None``, derived
            from zone width / horizontal spacing (at least 1).
        pin_counts: Optional list of pin counts parallel to *refs*.
            When provided, vertical spacing adapts to symbol height.
        symbol_extents: Optional mapping from ref to :class:`SymbolExtent`.
            When provided, spacing is computed from actual symbol visual
            extents instead of pin-count heuristics.

    Returns:
        Mapping from component ref to grid-snapped :class:`Point`.
    """
    result: dict[str, Point] = {}

    # --- Extent-based placement path ---
    if symbol_extents is not None and all(r in symbol_extents for r in refs):
        return _place_in_zone_with_extents(refs, zone, grid, symbol_extents)

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


def _place_in_zone_with_extents(
    refs: list[str],
    zone: PlacementZone,
    grid: float,
    symbol_extents: dict[str, SymbolExtent],
) -> dict[str, Point]:
    """Place refs using actual symbol extents for spacing.

    Packs symbols left-to-right in rows, wrapping when the zone width is
    exceeded.  Vertical spacing between rows uses the maximum bottom extent
    of the current row plus the maximum top extent of the next row.

    Args:
        refs: Component references to place.
        zone: Bounding zone.
        grid: Snap grid.
        symbol_extents: Mapping from ref to :class:`SymbolExtent`.

    Returns:
        Mapping from ref to grid-snapped :class:`Point`.
    """
    result: dict[str, Point] = {}
    gap = SCHEMATIC_SYMBOL_GAP_MM

    # Build rows greedily by fitting symbols within zone width
    rows: list[list[str]] = []
    current_row: list[str] = []
    current_row_width: float = 0.0

    for ref in refs:
        ext = symbol_extents[ref]
        if not current_row:
            # First symbol in row: its left extent determines the starting offset
            needed = ext.left + ext.right
        else:
            prev_ext = symbol_extents[current_row[-1]]
            needed = _extent_h_spacing(prev_ext, ext, gap)

        if current_row and current_row_width + needed > zone.width:
            rows.append(current_row)
            current_row = [ref]
            current_row_width = ext.left + ext.right
        else:
            current_row.append(ref)
            current_row_width += needed if len(current_row) > 1 else ext.left + ext.right
    if current_row:
        rows.append(current_row)

    # Place each row
    cumulative_y: float = 0.0
    for row_idx, row_refs in enumerate(rows):
        # Vertical spacing from previous row
        if row_idx > 0:
            prev_max_bottom = max(symbol_extents[r].bottom for r in rows[row_idx - 1])
            cur_max_top = max(symbol_extents[r].top for r in row_refs)
            cumulative_y += _extent_v_spacing(prev_max_bottom, cur_max_top, gap)

        # Horizontal placement: accumulate from zone origin
        cur_x: float = 0.0
        for col_idx, ref in enumerate(row_refs):
            ext = symbol_extents[ref]
            if col_idx == 0:
                # First symbol: offset by its left extent from zone edge
                cur_x = ext.left
            else:
                prev_ext = symbol_extents[row_refs[col_idx - 1]]
                cur_x += _extent_h_spacing(prev_ext, ext, gap)

            x = snap_to_grid(zone.origin_x + cur_x, grid)
            y = snap_to_grid(zone.origin_y + cumulative_y, grid)
            result[ref] = Point(x=x, y=y)
            log.debug(
                "place_in_zone[extent]: %s → (%.3f, %.3f) in zone %s",
                ref, x, y, zone.name,
            )

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
    symbol_extents: dict[str, SymbolExtent] | None = None,
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
        symbol_extents: Optional mapping from ref to :class:`SymbolExtent`.
            When provided, spacing is computed from actual visual extents.

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
        zone_positions = place_in_zone(
            zone_refs, zone, pin_counts=zone_pin_counts,
            symbol_extents=symbol_extents,
        )
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


# ---------------------------------------------------------------------------
# Compact layout for sub-sheets
# ---------------------------------------------------------------------------

# Default margins for compact sub-sheet placement (mm).
_COMPACT_MARGIN_LEFT: float = 30.0  # room for hierarchical labels at x=5..25
_COMPACT_MARGIN_TOP: float = 25.0
_COMPACT_MAX_WIDTH: float = 180.0  # leave right margin on A4


def layout_compact(
    symbol_refs: list[str],
    pin_count_map: dict[str, int] | None = None,
    adjacency: dict[str, set[str]] | None = None,
    margin_left: float = _COMPACT_MARGIN_LEFT,
    margin_top: float = _COMPACT_MARGIN_TOP,
    max_width: float = _COMPACT_MAX_WIDTH,
    symbol_extents: dict[str, SymbolExtent] | None = None,
) -> dict[str, Point]:
    """Place components in a compact grid for sub-sheet schematics.

    Unlike :func:`layout_schematic`, this does **not** spread components
    across 4 named zones.  Instead it packs them in rows starting from
    ``(margin_left, margin_top)``, wrapping when exceeding *max_width*.
    The left margin is intentionally generous to leave room for
    hierarchical labels.

    When *symbol_extents* is provided, spacing is derived from actual
    symbol visual extents instead of pin-count heuristics.

    Args:
        symbol_refs: Component reference designators to place.
        pin_count_map: Mapping from ref to pin count for spacing.
        adjacency: Optional connectivity map for ordering.
        margin_left: Left margin in mm (default 30).
        margin_top: Top margin in mm (default 25).
        max_width: Maximum X extent before wrapping (default 180).
        symbol_extents: Optional mapping from ref to :class:`SymbolExtent`.

    Returns:
        Mapping from component ref to grid-snapped :class:`Point`.
    """
    grid = SCHEMATIC_WIRE_GRID_MM
    gap = SCHEMATIC_SYMBOL_GAP_MM

    # Order by connectivity if available
    refs = list(symbol_refs)
    if adjacency is not None and len(refs) > 2:
        refs = _sort_by_connectivity(refs, adjacency)

    # --- Extent-based compact layout ---
    if symbol_extents is not None and all(r in symbol_extents for r in refs):
        positions: dict[str, Point] = {}
        cur_x: float = margin_left
        cur_y: float = margin_top
        row_max_bottom: float = 0.0
        row_refs: list[str] = []

        for ref in refs:
            ext = symbol_extents[ref]
            sym_width = ext.left + ext.right + gap

            # Wrap to next row if exceeding max width
            if cur_x + sym_width > margin_left + max_width and row_refs:
                # Advance Y by max bottom of current row + max top of next ref
                cur_y += row_max_bottom + ext.top + gap
                cur_x = margin_left
                row_max_bottom = 0.0
                row_refs = []

            x = snap_to_grid(cur_x + ext.left, grid)
            y = snap_to_grid(cur_y, grid)
            positions[ref] = Point(x=x, y=y)

            row_max_bottom = max(row_max_bottom, ext.bottom)
            row_refs.append(ref)
            cur_x += ext.left + ext.right + gap

        log.debug("layout_compact[extent]: placed %d refs", len(positions))
        return positions

    # --- Fallback: pin-count heuristic layout ---
    positions = {}
    cur_x = margin_left
    cur_y = margin_top
    row_max_h: float = 0.0

    for ref in refs:
        pc = (pin_count_map or {}).get(ref, 2)
        h_spacing = _h_spacing_for_pins(pc, grid)

        # Wrap to next row if exceeding max width
        if cur_x + h_spacing > margin_left + max_width and cur_x > margin_left:
            cur_y += row_max_h + 10.0  # inter-row gap
            cur_x = margin_left
            row_max_h = 0.0

        x = snap_to_grid(cur_x, grid)
        y = snap_to_grid(cur_y, grid)
        positions[ref] = Point(x=x, y=y)

        # Estimate body height from pin count
        effective_pins = pc // 2 if pc > 10 else pc
        body_h = max(effective_pins * 2.54, 5.08) + 12.0  # body + label clearance
        row_max_h = max(row_max_h, body_h)
        cur_x += h_spacing

    log.debug("layout_compact: placed %d refs in compact grid", len(positions))
    return positions
