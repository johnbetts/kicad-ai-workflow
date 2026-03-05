"""Zone-based PCB component placement engine.

Assigns physical board positions to footprints by grouping them into named
functional zones and distributing them on a regular grid within each zone.

Zone layout (millimetres, origin at board top-left corner 0, 0):

+------------+----------+----------+
| POWER      |   MCU    | STATUS   |
| (8,8)      | (35,8)   | (57,8)   |
| 25x12      | 20x12    | 13x12    |
+------------+----------+----------+
| PERIPHERALS          | CONNECTORS|
| (8,22)               | (40,22)  |
| 30x12                | 30x12    |
+----------------------+-----------+

Board default: 80 x 40 mm (Hammond 1551K footprint)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.models.pcb import BoardOutline, Point


@dataclass(frozen=True)
class LayoutResult:
    """Result of PCB layout placement.

    Attributes:
        positions: Mapping from component ref to board position.
        rotations: Mapping from component ref to rotation in degrees.
    """

    positions: dict[str, Point]
    rotations: dict[str, float]

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
    "USB_POWER": PCBZone("USB_POWER", 8.0, 8.0, 25.0, 12.0),
    "STATUS": PCBZone("STATUS", 57.0, 8.0, 13.0, 12.0),
    "MCU": PCBZone("MCU", 35.0, 8.0, 20.0, 12.0),
    "ETHERNET": PCBZone("ETHERNET", 8.0, 22.0, 30.0, 12.0),
    "RJ45": PCBZone("RJ45", 40.0, 22.0, 30.0, 12.0),
    "ANALOG": PCBZone("ANALOG", 8.0, 22.0, 30.0, 12.0),
    "PERIPHERALS": PCBZone("PERIPHERALS", 8.0, 22.0, 30.0, 12.0),
    "CONNECTORS": PCBZone("CONNECTORS", 40.0, 22.0, 30.0, 12.0),
}
"""Standard PCB placement zones keyed by name.

Non-overlapping zones within 80x40mm board, inset from mounting holes.
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
    component_spacing_mm: float = 5.0,
    footprint_sizes: dict[str, tuple[float, float]] | None = None,
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
            components in mm (default ``5.0``).
        footprint_sizes: Optional mapping from ref to ``(width, height)``
            in mm.  When provided, spacing is increased if the largest
            footprint in the zone exceeds the default spacing.

    Returns:
        Mapping from component ref to grid-snapped :class:`Point`.

    Raises:
        PCBError: If the zone is too small to accommodate all *refs*.
    """
    if not refs:
        return {}

    # Adjust spacing for large footprints in this zone
    if footprint_sizes is not None:
        zone_sizes = [footprint_sizes[r] for r in refs if r in footprint_sizes]
        if zone_sizes:
            max_w = max(s[0] for s in zone_sizes)
            max_h = max(s[1] for s in zone_sizes)
            max_dim = max(max_w, max_h)
            if max_dim + 4.0 > component_spacing_mm:
                component_spacing_mm = max_dim + 4.0
                log.info(
                    "place_pcb_components: increased spacing to %.1f mm "
                    "for zone '%s' (largest footprint %.1f mm)",
                    component_spacing_mm,
                    zone.name,
                    max_dim,
                )

    cols = max(1, math.floor(zone.width / component_spacing_mm))
    rows_needed = math.ceil(len(refs) / cols)
    required_height = rows_needed * component_spacing_mm

    # Auto-reduce spacing if zone is too small, down to a minimum of 2.0 mm
    if required_height > zone.height + 1e-9:
        reduced = component_spacing_mm
        while reduced > 2.0:
            reduced -= 0.5
            cols = max(1, math.floor(zone.width / reduced))
            rows_needed = math.ceil(len(refs) / cols)
            required_height = rows_needed * reduced
            if required_height <= zone.height + 1e-9:
                log.info(
                    "place_pcb_components: auto-reduced spacing from %.1f to %.1f mm "
                    "for zone '%s' (%d components)",
                    component_spacing_mm,
                    reduced,
                    zone.name,
                    len(refs),
                )
                component_spacing_mm = reduced
                break
        else:
            raise PCBError(
                f"Zone '{zone.name}' ({zone.width:.1f}x{zone.height:.1f} mm) is too small "
                f"for {len(refs)} components even with spacing {reduced:.1f} mm "
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


def _scale_zones(
    board: BoardOutline,
    base_width: float = 80.0,
    base_height: float = 40.0,
) -> dict[str, PCBZone]:
    """Scale the default PCB zones to fit the actual board dimensions.

    When the board is larger than the default 80x40mm, zones are scaled
    proportionally so that components spread across the full board area.

    Args:
        board: Actual board outline.
        base_width: Width the default zones were designed for.
        base_height: Height the default zones were designed for.

    Returns:
        Scaled zone dictionary.
    """
    # Compute actual board dimensions from outline polygon
    xs = [p.x for p in board.polygon]
    ys = [p.y for p in board.polygon]
    actual_w = max(xs) - min(xs)
    actual_h = max(ys) - min(ys)

    if actual_w <= base_width + 0.1 and actual_h <= base_height + 0.1:
        return dict(PCB_ZONES)

    scale_x = actual_w / base_width
    scale_y = actual_h / base_height
    log.info(
        "Scaling PCB zones: %.2fx horizontal, %.2fx vertical",
        scale_x,
        scale_y,
    )

    scaled: dict[str, PCBZone] = {}
    for name, zone in PCB_ZONES.items():
        scaled[name] = PCBZone(
            name=name,
            x=zone.x * scale_x,
            y=zone.y * scale_y,
            width=zone.width * scale_x,
            height=zone.height * scale_y,
        )
    return scaled


def _dynamic_zones(
    groups: dict[str, list[str]],
    board_width: float,
    board_height: float,
    footprint_sizes: dict[str, tuple[float, float]] | None = None,
    margin: float = 5.0,
) -> dict[str, PCBZone]:
    """Create non-overlapping placement zones sized for actual component groups.

    Distributes the available board area among the groups that actually have
    components, ensuring no two groups share the same space.

    Args:
        groups: Mapping from zone name to list of component refs.
        board_width: Usable board width in mm.
        board_height: Usable board height in mm.
        footprint_sizes: Optional mapping from ref to ``(width, height)`` in mm.
        margin: Inset from board edges in mm.

    Returns:
        Mapping from zone name to a dynamically computed :class:`PCBZone`.
    """
    if not groups:
        return {}

    usable_w = board_width - 2.0 * margin
    usable_h = board_height - 2.0 * margin
    if usable_w < 10.0 or usable_h < 10.0:
        usable_w = max(usable_w, board_width * 0.8)
        usable_h = max(usable_h, board_height * 0.8)
        margin = (board_width - usable_w) / 2.0

    # Estimate area needed per group
    group_areas: dict[str, float] = {}
    for gname, refs in groups.items():
        area = 0.0
        for ref in refs:
            if footprint_sizes and ref in footprint_sizes:
                w, h = footprint_sizes[ref]
                # Add clearance around each component
                area += (w + 3.0) * (h + 3.0)
            else:
                area += 7.0 * 7.0  # default 7x7mm per component
        group_areas[gname] = area

    total_area = sum(group_areas.values())
    if total_area < 1.0:
        total_area = 1.0

    # Lay out groups in rows, allocating width proportional to area
    sorted_groups = sorted(groups.keys(), key=lambda g: -group_areas[g])

    # Use up to 2 rows; put largest groups in row 1
    row1_groups: list[str] = []
    row2_groups: list[str] = []
    row1_area = 0.0
    half_area = total_area / 2.0
    for gname in sorted_groups:
        if row1_area < half_area or not row1_groups:
            row1_groups.append(gname)
            row1_area += group_areas[gname]
        else:
            row2_groups.append(gname)

    # If only 1 row needed, give it full height
    if not row2_groups:
        row_heights = [usable_h]
        rows_list = [row1_groups]
    else:
        # Split height proportionally
        row1_frac = row1_area / total_area
        row1_h = max(usable_h * row1_frac, 15.0)
        row2_h = max(usable_h - row1_h, 15.0)
        # Redistribute if one row is too small
        total_rh = row1_h + row2_h
        row1_h = usable_h * row1_h / total_rh
        row2_h = usable_h - row1_h
        row_heights = [row1_h, row2_h]
        rows_list = [row1_groups, row2_groups]

    result: dict[str, PCBZone] = {}
    y_offset = margin
    for row_groups, row_h in zip(rows_list, row_heights, strict=True):
        row_total = sum(group_areas[g] for g in row_groups)
        if row_total < 1.0:
            row_total = 1.0
        x_offset = margin
        for gname in row_groups:
            frac = group_areas[gname] / row_total
            zone_w = max(usable_w * frac, 10.0)
            # Clamp to remaining width
            zone_w = min(zone_w, board_width - x_offset - margin)
            result[gname] = PCBZone(
                name=gname,
                x=x_offset,
                y=y_offset,
                width=max(zone_w, 5.0),
                height=max(row_h, 5.0),
            )
            x_offset += zone_w
        y_offset += row_h

    return result


def layout_pcb(
    requirements: ProjectRequirements,
    board: BoardOutline,
    footprint_sizes: dict[str, tuple[float, float]] | None = None,
    fixed_positions: dict[str, tuple[float, float, float]] | None = None,
    board_template: object | None = None,
    keepouts: tuple[object, ...] = (),
) -> LayoutResult:
    """Compute a full PCB placement for all components in *requirements*.

    When *board_template* is provided, uses the constraint-based solver
    for intelligent placement. Otherwise falls back to the zone-based
    grid-fill approach for backward compatibility.

    Args:
        requirements: Fully-populated project requirements document.
        board: The board outline used to validate that components fit.
        footprint_sizes: Optional mapping from ref to ``(width, height)``
            in mm.  Passed through to :func:`place_pcb_components`.
        fixed_positions: Optional mapping from ref to ``(x, y, rotation)``
            for components with fixed board positions (e.g. from a board
            template).  These refs are excluded from dynamic placement.
        board_template: Optional :class:`BoardTemplate` for constraint-based
            placement. When provided, the constraint solver is used.
        keepouts: Optional keepout zones to avoid during placement.

    Returns:
        :class:`LayoutResult` with positions and rotations for every
        component in *requirements*.
    """
    # Constraint-based path when template is available
    if board_template is not None and footprint_sizes is not None:
        from kicad_pipeline.models.pcb import Keepout as KeepoutModel
        from kicad_pipeline.pcb.board_templates import BoardTemplate as BTClass
        from kicad_pipeline.pcb.constraints import (
            constraints_from_requirements,
            solve_placement,
        )

        if isinstance(board_template, BTClass):
            log.info(
                "layout_pcb: using constraint-based solver (template=%s)",
                board_template.name,
            )
            # Inject fixed_positions as FIXED constraints
            from kicad_pipeline.models.pcb import (
                PlacementConstraint,
                PlacementConstraintType,
            )
            extra_constraints: list[PlacementConstraint] = []
            if fixed_positions:
                for ref, (fx, fy, frot) in fixed_positions.items():
                    extra_constraints.append(PlacementConstraint(
                        ref=ref,
                        constraint_type=PlacementConstraintType.FIXED,
                        x=fx,
                        y=fy,
                        rotation=frot,
                        priority=100,
                    ))
            # Use HAT-specific constraints for RPi HAT boards
            if board_template.name == "RPI_HAT":
                from kicad_pipeline.pcb.constraints import rpi_hat_constraints
                constraint_list = rpi_hat_constraints(
                    requirements, board_template, footprint_sizes,
                )
            else:
                constraint_list = constraints_from_requirements(
                    requirements, board_template, footprint_sizes,
                )
            if extra_constraints:
                # Merge: extra_constraints override by ref
                extra_refs = {c.ref for c in extra_constraints}
                constraint_list = tuple(
                    c for c in constraint_list if c.ref not in extra_refs
                ) + tuple(extra_constraints)
            typed_keepouts = tuple(
                k for k in keepouts if isinstance(k, KeepoutModel)
            )
            result = solve_placement(
                constraint_list, board, footprint_sizes,
                keepouts=typed_keepouts, grid_mm=0.5,
                requirements=requirements,
            )
            positions = dict(result.positions)
            rotations = dict(result.rotations)

            # Log any placement violations
            if result.violations:
                for violation in result.violations:
                    log.warning("layout_pcb: placement violation: %s", violation)

            # Ensure all component refs are placed
            all_refs = {c.ref for c in requirements.components}
            unplaced = all_refs - set(positions.keys())
            if unplaced:
                log.warning(
                    "layout_pcb: constraint solver missed %d refs, adding: %s",
                    len(unplaced), list(unplaced),
                )
                xs = [p.x for p in board.polygon]
                ys = [p.y for p in board.polygon]
                board_w = max(xs) - min(xs)
                board_h = max(ys) - min(ys)
                fallback = PCBZone("FALLBACK", 5.0, board_h * 0.6, board_w - 10.0, board_h * 0.35)
                extra = place_pcb_components(
                    list(unplaced), fallback, footprint_sizes=footprint_sizes,
                )
                positions.update(extra)

            log.info("layout_pcb: placed %d components (constraint solver)", len(positions))
            return LayoutResult(positions=positions, rotations=rotations)
    # --- Zone-based fallback path (no template) ---
    # Pre-populate positions from fixed_positions (board template)
    zone_positions: dict[str, Point] = {}
    zone_rotations: dict[str, float] = {}
    fixed_refs: set[str] = set()
    if fixed_positions:
        for ref, (fx, fy, frot) in fixed_positions.items():
            zone_positions[ref] = Point(x=fx, y=fy)
            zone_rotations[ref] = frot
            fixed_refs.add(ref)
            log.info("layout_pcb: fixed position for %s at (%.2f, %.2f)", ref, fx, fy)

    # Build feature map from FeatureBlocks
    feature_map: dict[str, str] = {}
    for fb in requirements.features:
        for ref in fb.components:
            feature_map[ref] = fb.name

    zone_refs_list = [c.ref for c in requirements.components if c.ref not in fixed_refs]
    tagged = [(ref, feature_map.get(ref, "Peripherals")) for ref in zone_refs_list]

    zone_map = assign_pcb_zones(tagged)

    # Group refs by zone name
    groups: dict[str, list[str]] = {}
    for ref, zone in zone_map.items():
        groups.setdefault(zone.name, []).append(ref)

    # Compute actual board dimensions
    xs = [p.x for p in board.polygon]
    ys = [p.y for p in board.polygon]
    board_w = max(xs) - min(xs)
    board_h = max(ys) - min(ys)

    # Create dynamic non-overlapping zones based on actual groups
    dynamic_zones = _dynamic_zones(groups, board_w, board_h, footprint_sizes)

    for zone_name, zone_refs in groups.items():
        zone = dynamic_zones[zone_name]
        zone_pos = place_pcb_components(
            zone_refs, zone, footprint_sizes=footprint_sizes,
        )
        zone_positions.update(zone_pos)

    # Safety net: any refs not yet placed (check all components, not just dynamic)
    all_component_refs = [c.ref for c in requirements.components]
    unplaced_refs = [ref for ref in all_component_refs if ref not in zone_positions]
    if unplaced_refs:
        log.warning(
            "layout_pcb: %d refs not placed; adding fallback placement: %s",
            len(unplaced_refs),
            unplaced_refs,
        )
        # Create a fallback zone from remaining board space
        fallback = PCBZone("FALLBACK", 5.0, board_h * 0.6, board_w - 10.0, board_h * 0.35)
        extra = place_pcb_components(
            unplaced_refs, fallback, footprint_sizes=footprint_sizes,
        )
        zone_positions.update(extra)

    log.info("layout_pcb: placed %d components", len(zone_positions))
    return LayoutResult(positions=zone_positions, rotations=zone_rotations)
