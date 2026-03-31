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
    2. Use reference-board proportions as default zone rects.
    3. Scale zone sizes proportionally to total component count per zone.
    4. Ensure inter-zone gaps of ``_ZONE_GAP_MM``.
    5. Only create zones that have at least one assigned group.

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

    # Step 1: Map groups to zones
    zone_groups: dict[str, list[str]] = {}
    zone_component_count: dict[str, int] = {}
    for group in groups:
        zone_name = _match_group_to_zone(group.name)
        zone_groups.setdefault(zone_name, []).append(group.name)
        zone_component_count[zone_name] = (
            zone_component_count.get(zone_name, 0) + len(group.components)
        )

    if not zone_groups:
        return []

    _log.info(
        "Zone partitioning: %d groups → %d zones",
        len(groups),
        len(zone_groups),
    )
    for zn, gnames in zone_groups.items():
        _log.info("  Zone '%s': %s (%d components)", zn, gnames,
                   zone_component_count.get(zn, 0))

    # Step 2: Compute zone rects — scale default fractions by component area
    half_gap = _ZONE_GAP_MM / 2.0

    # When there's only one zone, give it the full board area
    single_zone = len(zone_groups) == 1

    # Compute area-proportional scaling per row.
    # The default layout has 3 rows:
    #   Row 0: input_connectors (full width)
    #   Row 1: power (left) + relay (right)
    #   Row 2: analog (left) + ethernet (center) + mcu (right)
    # Within each row, redistribute width proportionally to component count.
    row_groups: dict[str, list[str]] = {
        "row0": ["input_connectors"],
        "row1": ["power", "relay"],
        "row2": ["analog", "ethernet", "mcu"],
    }

    # Build adjusted fractions based on component counts within each row
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
                # Find row Y span from any zone in this row
                row_ys = [
                    _DEFAULT_ZONE_FRACTIONS[z]
                    for z in row_zones
                    if z in _DEFAULT_ZONE_FRACTIONS
                ]
                y1 = min(f[1] for f in row_ys)
                y2 = max(f[3] for f in row_ys)
                adjusted_fracs[active[0]] = (0.0, y1, 1.0, y2)
            continue

        # Multiple zones in row — distribute width by component count
        counts = [zone_component_count.get(z, 1) for z in active]
        total = sum(counts) or 1

        # Get row Y span
        row_ys = [
            _DEFAULT_ZONE_FRACTIONS[z]
            for z in active
            if z in _DEFAULT_ZONE_FRACTIONS
        ]
        if not row_ys:
            continue
        y1 = min(f[1] for f in row_ys)
        y2 = max(f[3] for f in row_ys)

        # Distribute X proportionally with minimum 15% per zone
        min_frac = 0.15
        remaining = 1.0 - min_frac * len(active)
        x_cursor = 0.0
        for i, z in enumerate(active):
            frac = min_frac + remaining * (counts[i] / total)
            adjusted_fracs[z] = (x_cursor, y1, x_cursor + frac, y2)
            x_cursor += frac

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

        # No additional scaling needed — fractions already proportional
        zx1 = max(0.0, fx1)
        zy1 = max(0.0, fy1)
        zx2 = min(1.0, fx2)
        zy2 = min(1.0, fy2)

        # Convert fractions to absolute coordinates with gap inset
        abs_x1 = bx1 + zx1 * board_w + half_gap
        abs_y1 = by1 + zy1 * board_h + half_gap
        abs_x2 = bx1 + zx2 * board_w - half_gap
        abs_y2 = by1 + zy2 * board_h - half_gap

        # Ensure minimum zone size
        if abs_x2 - abs_x1 < 10.0:
            abs_x2 = abs_x1 + 10.0
        if abs_y2 - abs_y1 < 10.0:
            abs_y2 = abs_y1 + 10.0

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
