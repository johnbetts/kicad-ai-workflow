"""Board zone partitioner for hierarchical placement.

Partitions the PCB board area into non-overlapping rectangular zones based
on FeatureBlock functional groups. Each zone is assigned to one or more
groups using keyword matching, enabling top-down placement where groups
are placed as rigid units within their zones.

This is Level 1 of the 3-level hierarchical placement engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import FeatureBlock
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
# courtyard clearances and component-to-component gaps.
_DENSITY_FACTOR: float = 2.0

# Minimum zone dimension (mm) — prevents zones from collapsing to zero.
_MIN_ZONE_DIM_MM: float = 25.0

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
    # Tiled layout (160x80mm board):
    #   Row 0 (0-100% x, 0-18% y): input_connectors (screw terminals)
    #   Row 1 left (0-35% x, 18-55% y): power
    #   Row 1 right (35-100% x, 18-55% y): relay (includes driver support)
    #   Row 2 left (0-30% x, 55-90% y): analog
    #   Row 2 center (30-55% x, 55-90% y): ethernet
    #   Row 2 right (55-100% x, 55-90% y): mcu
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


def partition_board(
    board_bounds: tuple[float, float, float, float],
    groups: list[FeatureBlock],
    topology: PowerFlowTopology | None = None,
) -> list[BoardZone]:
    """Partition board into non-overlapping rectangular zones.

    Strategy:
    1. Map each FeatureBlock to a zone by keyword matching on name.
    2. Compute needed footprint area per zone bottom-up from component refs
       (SMD passive 4mm², THT 50mm², IC/relay 100mm², module 200mm²) scaled by
       a density factor of 2.0 for component spacing.
    3. Preserve the 3-row layout Y spans from the reference board fractions.
       Within each row, redistribute widths proportionally to zone footprint area
       (not component count) so large-component zones get more space.
    4. Enforce a minimum zone dimension of 25mm to prevent collapsed zones.
    5. Warn when a zone's allocated area is less than its needed footprint area.
    6. Ensure inter-zone gaps of ``_ZONE_GAP_MM``.
    7. Only create zones that have at least one assigned group.

    Args:
        board_bounds: (min_x, min_y, max_x, max_y) in mm.
        groups: FeatureBlock instances to partition.
        topology: Optional power flow topology for domain ordering
            (reserved for future use).

    Returns:
        List of BoardZone instances with absolute board coordinates.
    """
    bx1, by1, bx2, by2 = board_bounds
    board_w = bx2 - bx1
    board_h = by2 - by1

    if not groups:
        return []

    # Step 1: Map groups to zones and compute needed footprint area bottom-up.
    zone_groups: dict[str, list[str]] = {}
    zone_component_count: dict[str, int] = {}
    zone_footprint_area: dict[str, float] = {}
    for group in groups:
        zone_name = _match_group_to_zone(group.name)
        zone_groups.setdefault(zone_name, []).append(group.name)
        zone_component_count[zone_name] = (
            zone_component_count.get(zone_name, 0) + len(group.components)
        )
        zone_footprint_area[zone_name] = (
            zone_footprint_area.get(zone_name, 0.0)
            + _zone_footprint_area_mm2(group.components)
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

    # Step 2: Compute zone rects using area-proportional allocation.
    #
    # The 3-row layout structure is preserved (row heights from _DEFAULT_ZONE_FRACTIONS).
    # Within each row, widths are now allocated proportionally to zone footprint area
    # (sum of component areas × density factor) rather than component count.
    # This prevents the MCU catch-all zone from consuming disproportionate space.
    half_gap = _ZONE_GAP_MM / 2.0

    # When there's only one zone, give it the full board area
    single_zone = len(zone_groups) == 1

    # 3-row layout: row0 is full-width, rows 1-2 subdivide by area.
    row_groups: dict[str, list[str]] = {
        "row0": ["input_connectors"],
        "row1": ["power", "relay"],
        "row2": ["analog", "ethernet", "mcu"],
    }

    # Build area-proportional fractions for each zone within its row.
    adjusted_fracs: dict[str, tuple[float, float, float, float]] = {}
    for _row_name, row_zones in row_groups.items():
        # Only consider zones that have assigned groups
        active = [z for z in row_zones if z in zone_groups]
        if not active:
            continue
        if len(active) == 1:
            # Single zone in row — give it the full row width
            base = _DEFAULT_ZONE_FRACTIONS.get(active[0])
            if base:
                row_ys = [
                    _DEFAULT_ZONE_FRACTIONS[z]
                    for z in row_zones
                    if z in _DEFAULT_ZONE_FRACTIONS
                ]
                y1 = min(f[1] for f in row_ys)
                y2 = max(f[3] for f in row_ys)
                adjusted_fracs[active[0]] = (0.0, y1, 1.0, y2)
            continue

        # Multiple zones in row — distribute width proportionally to footprint area.
        areas = [max(zone_footprint_area.get(z, _DEFAULT_COMPONENT_AREA_MM2), _DEFAULT_COMPONENT_AREA_MM2)
                 for z in active]
        total_area = sum(areas) or 1.0

        # Get row Y span from default fractions
        row_ys = [
            _DEFAULT_ZONE_FRACTIONS[z]
            for z in active
            if z in _DEFAULT_ZONE_FRACTIONS
        ]
        if not row_ys:
            continue
        y1 = min(f[1] for f in row_ys)
        y2 = max(f[3] for f in row_ys)

        # Distribute X proportionally to area with a minimum 15% floor per zone.
        min_frac = 0.15
        remaining = 1.0 - min_frac * len(active)
        x_cursor = 0.0
        for i, z in enumerate(active):
            frac = min_frac + remaining * (areas[i] / total_area)
            adjusted_fracs[z] = (x_cursor, y1, x_cursor + frac, y2)
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

        # Enforce minimum zone dimension (25mm) — prevents collapsed zones.
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
