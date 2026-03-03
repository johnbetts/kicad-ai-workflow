"""Schematic component placement engine.

Assigns 2-D positions to schematic symbols by first grouping them into named
functional zones and then distributing them on a regular grid within each zone.

Zone layout (millimetres, origin at top-left):

+-----------+-----------+
|  POWER    |   MCU     |
| (0,0)     | (130,0)   |
+-----------+-----------+
| ETHERNET  |  ANALOG   |
| (0,90)    | (280,90)  |
+-----------+-----------+
|       PERIPHERALS     |
|       (0,200)         |
+-----------------------+
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
    "POWER": PlacementZone("POWER", 0.0, 0.0, 120.0, 80.0),
    "MCU": PlacementZone("MCU", 130.0, 0.0, 140.0, 120.0),
    "ETHERNET": PlacementZone("ETHERNET", 0.0, 90.0, 120.0, 100.0),
    "ANALOG": PlacementZone("ANALOG", 280.0, 90.0, 140.0, 100.0),
    "PERIPHERALS": PlacementZone("PERIPHERALS", 0.0, 200.0, 420.0, 80.0),
}
"""Standard schematic zones keyed by name.

Matches the zone plan defined in the project architecture document section 2.4.
"""

# Feature-name to zone-name mapping (case-insensitive prefix matching)
_FEATURE_ZONE_MAP: list[tuple[tuple[str, ...], str]] = [
    (("power", "usb"), "POWER"),
    (("mcu", "core"), "MCU"),
    (("ethernet",), "ETHERNET"),
    (("analog",), "ANALOG"),
]
"""Ordered list of (feature_prefix_tuple, zone_name) pairs.

The first match wins.  Any unmatched feature falls back to ``'PERIPHERALS'``.
"""

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def assign_zones(
    components: list[tuple[str, str]],
) -> dict[str, PlacementZone]:
    """Assign each component reference to a :class:`PlacementZone`.

    The zone is selected by matching *feature_name* (case-insensitively) against
    the keywords in :data:`_FEATURE_ZONE_MAP`.  Components that do not match any
    keyword are placed in the ``'PERIPHERALS'`` zone.

    Args:
        components: List of ``(ref, feature_name)`` tuples, e.g.
            ``[('U1', 'MCU'), ('C1', 'Power'), ('J1', 'USB')]``.

    Returns:
        Mapping from component ref to the assigned :class:`PlacementZone`.
    """
    result: dict[str, PlacementZone] = {}
    for ref, feature in components:
        feature_lower = feature.lower()
        zone_name = "PERIPHERALS"
        for keywords, candidate_zone in _FEATURE_ZONE_MAP:
            if any(feature_lower.startswith(kw) for kw in keywords):
                zone_name = candidate_zone
                break
        result[ref] = SCHEMATIC_ZONES[zone_name]
        log.debug("assign_zones: %s (feature=%s) → %s", ref, feature, zone_name)
    return result


def place_in_zone(
    refs: list[str],
    zone: PlacementZone,
    grid: float = SCHEMATIC_WIRE_GRID_MM,
    symbols_per_row: int = 3,
) -> dict[str, Point]:
    """Place a list of component refs in a grid within *zone*.

    Symbols are arranged left-to-right, top-to-bottom with a spacing of
    25.4 mm horizontally and 30 mm vertically between symbol origins.
    All positions are snapped to the schematic grid.

    Args:
        refs: Ordered list of component reference designators to place.
        zone: The :class:`PlacementZone` that bounds the placement.
        grid: Grid size for snapping (default: ``SCHEMATIC_WIRE_GRID_MM``).
        symbols_per_row: Number of symbols per row (default ``3``).

    Returns:
        Mapping from component ref to grid-snapped :class:`Point`.
    """
    h_spacing = 25.4  # mm between symbol origins horizontally
    v_spacing = 30.0  # mm between symbol origins vertically

    result: dict[str, Point] = {}
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


def layout_schematic(
    symbol_refs: list[str],
    feature_map: dict[str, str],
) -> dict[str, Point]:
    """Compute the full schematic layout for all symbol references.

    Assigns each ref to a zone using :func:`assign_zones`, then calls
    :func:`place_in_zone` for each zone group.

    Args:
        symbol_refs: All component reference designators to place.
        feature_map: Mapping from ref to feature name, e.g.
            ``{'U1': 'MCU', 'C1': 'Power'}``.  Refs missing from the map
            are assigned to the ``'PERIPHERALS'`` zone.

    Returns:
        Mapping from component ref to :class:`Point` for every ref in
        *symbol_refs*.
    """
    # Build (ref, feature) list — fall back to 'Peripherals' for unknowns
    tagged = [(ref, feature_map.get(ref, "Peripherals")) for ref in symbol_refs]
    zone_map = assign_zones(tagged)

    # Group refs by zone
    groups: dict[str, list[str]] = {}
    for ref, zone in zone_map.items():
        groups.setdefault(zone.name, []).append(ref)

    positions: dict[str, Point] = {}
    for zone_name, zone_refs in groups.items():
        zone = SCHEMATIC_ZONES[zone_name]
        zone_positions = place_in_zone(zone_refs, zone)
        positions.update(zone_positions)

    # Any refs not covered (shouldn't happen, but be safe)
    for ref in symbol_refs:
        if ref not in positions:
            log.warning("layout_schematic: %s not placed; adding to PERIPHERALS", ref)
            fallback_zone = SCHEMATIC_ZONES["PERIPHERALS"]
            offset = len(positions) * 25.4
            positions[ref] = Point(
                x=snap_to_grid(fallback_zone.origin_x + offset, SCHEMATIC_WIRE_GRID_MM),
                y=snap_to_grid(fallback_zone.origin_y, SCHEMATIC_WIRE_GRID_MM),
            )

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
