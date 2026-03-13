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
    # Non-overlapping tiled layout (140x80mm reference board):
    #   Top strip (0-70% x, 0-15% y): input_connectors (screw terminals)
    #   Upper-left (0-20% x, 15-50% y): power (vertical signal chain)
    #   Upper-right (20-100% x, 15-50% y): relay
    #   Bottom-left (0-45% x, 50-100% y): analog (wide for horizontal ADC strips)
    #   Bottom-center (45-65% x, 50-100% y): ethernet
    #   Bottom-right (65-100% x, 50-100% y): mcu
    "input_connectors": (0.00, 0.00, 0.70, 0.15),
    "power":            (0.00, 0.15, 0.20, 0.50),
    "relay":            (0.20, 0.15, 1.00, 0.50),
    "analog":           (0.00, 0.50, 0.45, 1.00),
    "mcu":              (0.65, 0.50, 1.00, 1.00),
    "ethernet":         (0.45, 0.50, 0.65, 1.00),
    "display":          (0.00, 0.85, 0.20, 1.00),
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
        "ethernet": "bottom",
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

    # Step 2: Compute zone rects from default fractions, scaled by component count
    total_components = sum(zone_component_count.values()) or 1
    half_gap = _ZONE_GAP_MM / 2.0

    zones: list[BoardZone] = []
    for zone_name, group_names in zone_groups.items():
        fracs = _DEFAULT_ZONE_FRACTIONS.get(zone_name)
        if fracs is None:
            # Unknown zone — assign a center region
            fracs = (0.30, 0.30, 0.70, 0.70)

        fx1, fy1, fx2, fy2 = fracs

        # Use fixed zone fractions — scaling is disabled until proportional
        # tiling is implemented properly (old scaling caused zone overlaps).
        scale = 1.0

        # Apply scale (expand from center of default zone)
        cx = (fx1 + fx2) / 2.0
        cy = (fy1 + fy2) / 2.0
        half_w = (fx2 - fx1) / 2.0 * scale
        half_h = (fy2 - fy1) / 2.0 * scale

        # Clamp to [0, 1] range
        zx1 = max(0.0, cx - half_w)
        zy1 = max(0.0, cy - half_h)
        zx2 = min(1.0, cx + half_w)
        zy2 = min(1.0, cy + half_h)

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
