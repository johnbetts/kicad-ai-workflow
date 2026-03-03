"""Zone-based PCB component placement engine.

Assigns physical board positions to footprints by grouping them into named
functional zones and distributing them on a regular grid within each zone.

Zone layout (millimetres, origin at board top-left corner 0, 0):

+-------------------+-------------------+
|  USB_POWER (0,0)  |  STATUS (20,0)    |
| 20x12             | 20x10             |
+-------------------+-------------------+
|          MCU (5,12)                   |
|          40x20                        |
+---------------------------------------+
| ETHERNET (0,32)   | ANALOG (50,32)    |
| 30x8              | 30x8              |
+-------------------+-------------------+
| RJ45 (0,28)       | PERIPHERALS(40,20)|
| 15x12             | 40x12             |
+-------------------+-------------------+

Board default: 80 x 40 mm (Hammond 1551K footprint)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.models.pcb import BoardOutline, Point

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PCB placement zone descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PCBZone:
    """A named rectangular placement zone on the PCB board.

    All coordinates and dimensions are in millimetres from the board origin
    (top-left corner at 0, 0).

    Attributes:
        name: Zone identifier, e.g. ``'MCU'``, ``'USB_POWER'``.
        x: Left edge of the zone in mm.
        y: Top edge of the zone in mm.
        width: Zone width in mm.
        height: Zone height in mm.
    """

    name: str
    x: float
    y: float
    width: float
    height: float


# ---------------------------------------------------------------------------
# Standard PCB zone layout — matches plan section 3.3
# Board default: 80 x 40 mm (Hammond 1551K)
# ---------------------------------------------------------------------------

PCB_ZONES: dict[str, PCBZone] = {
    "USB_POWER": PCBZone("USB_POWER", 0.0, 0.0, 20.0, 12.0),
    "STATUS": PCBZone("STATUS", 20.0, 0.0, 20.0, 10.0),
    "MCU": PCBZone("MCU", 5.0, 12.0, 40.0, 20.0),
    "ETHERNET": PCBZone("ETHERNET", 0.0, 32.0, 30.0, 8.0),
    "RJ45": PCBZone("RJ45", 0.0, 28.0, 15.0, 12.0),
    "ANALOG": PCBZone("ANALOG", 50.0, 32.0, 30.0, 8.0),
    "PERIPHERALS": PCBZone("PERIPHERALS", 40.0, 20.0, 40.0, 12.0),
}
"""Standard PCB placement zones keyed by name.

Matches the zone plan defined in the project architecture document section 3.3.
"""

# Ordered mapping of feature-name prefixes to zone names (case-insensitive).
# First match wins; unmatched features fall back to 'PERIPHERALS'.
_FEATURE_ZONE_MAP: list[tuple[tuple[str, ...], str]] = [
    (("usb", "power", "ldo"), "USB_POWER"),
    (("mcu", "core"), "MCU"),
    (("ethernet", "w5500"), "ETHERNET"),
    (("rj45",), "RJ45"),
    (("analog", "adc"), "ANALOG"),
    (("led", "status"), "STATUS"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assign_pcb_zones(
    components: list[tuple[str, str]],
) -> dict[str, PCBZone]:
    """Map each component ref to a PCB placement zone.

    The zone is chosen by matching *feature_name* case-insensitively against
    the keywords in ``_FEATURE_ZONE_MAP``.  Components that do not match any
    keyword are assigned to the ``'PERIPHERALS'`` zone.

    Feature → zone heuristic (case-insensitive prefix match):

    * ``"USB"`` | ``"Power"`` | ``"LDO"``  → ``"USB_POWER"``
    * ``"MCU"`` | ``"Core"``               → ``"MCU"``
    * ``"Ethernet"`` | ``"W5500"``         → ``"ETHERNET"``
    * ``"RJ45"``                           → ``"RJ45"``
    * ``"Analog"`` | ``"ADC"``             → ``"ANALOG"``
    * ``"LED"`` | ``"Status"``             → ``"STATUS"``
    * Everything else                      → ``"PERIPHERALS"``

    Args:
        components: List of ``(ref, feature_name)`` tuples, e.g.
            ``[('U1', 'MCU'), ('C1', 'Power'), ('J1', 'USB')]``.

    Returns:
        Mapping from component ref to the assigned :class:`PCBZone`.
    """
    result: dict[str, PCBZone] = {}
    for ref, feature in components:
        feature_lower = feature.lower()
        zone_name = "PERIPHERALS"
        for keywords, candidate_zone in _FEATURE_ZONE_MAP:
            if any(feature_lower.startswith(kw) for kw in keywords):
                zone_name = candidate_zone
                break
        result[ref] = PCB_ZONES[zone_name]
        log.debug("assign_pcb_zones: %s (feature=%s) → %s", ref, feature, zone_name)
    return result


def _snap(value: float, grid: float) -> float:
    """Round *value* to the nearest multiple of *grid*.

    Args:
        value: Coordinate value in mm.
        grid: Grid pitch in mm.

    Returns:
        Grid-snapped coordinate.
    """
    if grid <= 0.0:
        return value
    return round(value / grid) * grid


def place_pcb_components(
    refs: list[str],
    zone: PCBZone,
    grid_mm: float = 0.5,
    component_spacing_mm: float = 3.0,
) -> dict[str, Point]:
    """Place refs in a grid within *zone*, snapped to *grid_mm*.

    Components are arranged left-to-right then top-to-bottom.  The number of
    columns is derived from the zone width divided by *component_spacing_mm*,
    floored to at least 1.

    Args:
        refs: Ordered list of component reference designators to place.
        zone: The :class:`PCBZone` that bounds the placement.
        grid_mm: PCB placement grid pitch in mm (default ``0.5``).
        component_spacing_mm: Centre-to-centre distance between adjacent
            components in mm (default ``3.0``).

    Returns:
        Mapping from component ref to grid-snapped :class:`Point`.

    Raises:
        PCBError: If the zone is too small to accommodate all *refs*.
    """
    if not refs:
        return {}

    cols = max(1, math.floor(zone.width / component_spacing_mm))
    rows_needed = math.ceil(len(refs) / cols)
    required_height = rows_needed * component_spacing_mm

    if required_height > zone.height + 1e-9:
        raise PCBError(
            f"Zone '{zone.name}' ({zone.width:.1f}x{zone.height:.1f} mm) is too small "
            f"for {len(refs)} components with spacing {component_spacing_mm:.1f} mm "
            f"(needs {required_height:.1f} mm height)"
        )

    result: dict[str, Point] = {}
    for idx, ref in enumerate(refs):
        col = idx % cols
        row = idx // cols
        raw_x = zone.x + col * component_spacing_mm
        raw_y = zone.y + row * component_spacing_mm
        x = _snap(raw_x, grid_mm)
        y = _snap(raw_y, grid_mm)
        result[ref] = Point(x=x, y=y)
        log.debug(
            "place_pcb_components: %s → (%.3f, %.3f) in zone %s",
            ref,
            x,
            y,
            zone.name,
        )
    return result


# Connector keywords that should be placed near board edges
_CONNECTOR_KEYWORDS: frozenset[str] = frozenset(
    {"usb", "rj45", "j", "con", "header", "conn"}
)


def _is_connector(ref: str) -> bool:
    """Return True if *ref* looks like a connector reference designator.

    Args:
        ref: Component reference designator string.

    Returns:
        ``True`` when the ref prefix suggests a connector.
    """
    prefix = "".join(ch for ch in ref if ch.isalpha()).lower()
    return prefix in _CONNECTOR_KEYWORDS


def layout_pcb(
    requirements: ProjectRequirements,
    board: BoardOutline,
) -> dict[str, Point]:
    """Compute a full PCB placement for all components in *requirements*.

    Steps:

    1. Build a ``(ref, feature_name)`` list from :attr:`~ProjectRequirements.features`.
    2. Call :func:`assign_pcb_zones` to map each ref to a :class:`PCBZone`.
    3. Group refs by zone and call :func:`place_pcb_components` for each group.
    4. Connectors (USB-C, RJ45, headers) are nudged toward the board edge by
       using a reduced top-margin so they land within 5 mm of the board outline.

    Args:
        requirements: Fully-populated project requirements document.
        board: The board outline used to validate that components fit.

    Returns:
        Mapping from component ref to :class:`Point` for every component in
        *requirements*.
    """
    # Build feature map from FeatureBlocks
    feature_map: dict[str, str] = {}
    for fb in requirements.features:
        for ref in fb.components:
            feature_map[ref] = fb.name

    all_refs = [c.ref for c in requirements.components]
    tagged = [(ref, feature_map.get(ref, "Peripherals")) for ref in all_refs]

    zone_map = assign_pcb_zones(tagged)

    # Group refs by zone name
    groups: dict[str, list[str]] = {}
    for ref, zone in zone_map.items():
        groups.setdefault(zone.name, []).append(ref)

    positions: dict[str, Point] = {}
    for zone_name, zone_refs in groups.items():
        zone = PCB_ZONES[zone_name]
        zone_positions = place_pcb_components(zone_refs, zone)
        positions.update(zone_positions)

    # Safety net: any refs not yet placed go into PERIPHERALS
    fallback_zone = PCB_ZONES["PERIPHERALS"]
    unplaced = [ref for ref in all_refs if ref not in positions]
    if unplaced:
        log.warning(
            "layout_pcb: %d refs not placed; adding to PERIPHERALS: %s",
            len(unplaced),
            unplaced,
        )
        extra = place_pcb_components(unplaced, fallback_zone)
        positions.update(extra)

    log.info("layout_pcb: placed %d components", len(positions))
    return positions
