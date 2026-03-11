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
        layers: Mapping from component ref to layer override (e.g. ``"B.Cu"``).
    """

    positions: dict[str, Point]
    rotations: dict[str, float]
    layers: dict[str, str] | None = None

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
    "RELAY": PCBZone("RELAY", 40.0, 22.0, 30.0, 12.0),
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
    (("relay",), "RELAY"),
    (("analog", "adc"), "ANALOG"),
    (("led", "status"), "STATUS"),
    (("display", "tft", "lcd"), "PERIPHERALS"),
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


# ---------------------------------------------------------------------------
# Subcircuit sorting — group related components within a zone
# ---------------------------------------------------------------------------

# Keywords that identify WiFi/BLE modules needing edge placement
_WIFI_KEYWORDS: frozenset[str] = frozenset({
    "esp32", "wroom", "wrover", "wifi", "ble", "nrf52", "cc2640",
    "sx1276", "sx1262", "rfm95", "rfm96",
})

# Keywords that identify edge-sensitive connectors
_EDGE_CONNECTOR_KEYWORDS: frozenset[str] = frozenset({
    "usb", "rj45", "magjack", "sd_card", "microsd", "sim",
})


def _is_wifi_module(value: str, footprint: str) -> bool:
    """Return True if component is a WiFi/BLE module needing edge placement."""
    combined = f"{value} {footprint}".lower()
    return any(kw in combined for kw in _WIFI_KEYWORDS)


def _is_edge_connector(value: str, footprint: str) -> bool:
    """Return True if component is an edge-sensitive connector."""
    combined = f"{value} {footprint}".lower()
    return any(kw in combined for kw in _EDGE_CONNECTOR_KEYWORDS)


def _subcircuit_sort(
    refs: list[str],
    requirements: ProjectRequirements,
) -> list[str]:
    """Reorder component refs within a zone to group related parts together.

    Groups decoupling caps next to their ICs, passives with their nearest
    IC/connector by shared net count, and sorts groups by signal flow.

    Args:
        refs: List of component refs assigned to this zone.
        requirements: Full project requirements with nets for adjacency.

    Returns:
        Reordered list of refs with related components adjacent.
    """
    if len(refs) <= 2:
        return refs

    from kicad_pipeline.pcb.constraints import build_signal_adjacency

    ref_set = set(refs)
    adj = build_signal_adjacency(requirements)

    # Build net-based adjacency count between components in this zone
    net_sharing: dict[tuple[str, str], int] = {}
    for net in requirements.nets:
        zone_refs_in_net = [c.ref for c in net.connections if c.ref in ref_set]
        for i, r1 in enumerate(zone_refs_in_net):
            for r2 in zone_refs_in_net[i + 1:]:
                key = (min(r1, r2), max(r1, r2))
                net_sharing[key] = net_sharing.get(key, 0) + 1

    # Identify ICs and connectors as "anchor" components
    anchors: list[str] = []
    passives: list[str] = []
    for ref in refs:
        prefix = "".join(ch for ch in ref if ch.isalpha()).upper()
        if prefix in ("U", "J", "K", "Q", "Y"):
            anchors.append(ref)
        else:
            passives.append(ref)

    # Group each passive with its best anchor (most shared nets)
    anchor_groups: dict[str, list[str]] = {a: [] for a in anchors}
    ungrouped: list[str] = []
    for p in passives:
        best_anchor = ""
        best_count = 0
        for a in anchors:
            key = (min(p, a), max(p, a))
            count = net_sharing.get(key, 0)
            # Also count signal adjacency
            if a in adj.get(p, set()):
                count += 1
            if count > best_count:
                best_count = count
                best_anchor = a
        if best_anchor:
            anchor_groups[best_anchor].append(p)
        else:
            ungrouped.append(p)

    # Sort anchors: connectors first (inputs), then ICs, then others
    def _anchor_sort_key(ref: str) -> tuple[int, str]:
        prefix = "".join(ch for ch in ref if ch.isalpha()).upper()
        if prefix == "J":
            return (0, ref)
        if prefix == "U":
            return (1, ref)
        return (2, ref)

    sorted_anchors = sorted(anchors, key=_anchor_sort_key)

    # Build final sorted list: anchor followed by its grouped passives
    result: list[str] = []
    for anchor in sorted_anchors:
        result.append(anchor)
        # Sort passives within group: decoupling caps first, then by ref
        group = anchor_groups[anchor]
        group.sort(key=lambda r: (0 if r.startswith("C") else 1, r))
        result.extend(group)
    result.extend(ungrouped)

    log.debug("_subcircuit_sort: %s → %s", refs, result)
    return result


def _edge_priority_sort(
    groups: dict[str, list[str]],
    requirements: ProjectRequirements,
) -> dict[str, list[str]]:
    """Move edge-sensitive components to edge-adjacent zones.

    WiFi modules, USB connectors, RJ45 jacks, and SD card slots should
    be at board edges. Their immediate dependent passives stay with them.

    This modifies zone membership, moving edge-sensitive refs to edge zones
    while keeping their associated passives nearby.

    Args:
        groups: Zone name → list of refs (mutable, modified in-place).
        requirements: Full project requirements.

    Returns:
        Updated groups dict (same object, mutated).
    """
    comp_map = {c.ref: c for c in requirements.components}
    all_zone_refs = set()
    for refs in groups.values():
        all_zone_refs.update(refs)

    # Identify edge-sensitive components across all zones
    edge_refs: list[str] = []
    for ref in all_zone_refs:
        comp = comp_map.get(ref)
        if comp is None:
            continue
        if _is_wifi_module(comp.value, comp.footprint) or _is_edge_connector(
            comp.value, comp.footprint
        ):
            edge_refs.append(ref)

    if not edge_refs:
        return groups

    # Ensure CONNECTORS zone exists for edge components not already in one
    if "CONNECTORS" not in groups:
        groups["CONNECTORS"] = []

    # Move edge-sensitive refs from interior zones to CONNECTORS
    for ref in edge_refs:
        for zone_name, zone_refs in groups.items():
            if ref in zone_refs and zone_name not in ("CONNECTORS", "RJ45", "USB_POWER"):
                zone_refs.remove(ref)
                groups["CONNECTORS"].append(ref)
                log.info(
                    "_edge_priority_sort: moved %s from %s to CONNECTORS (edge-sensitive)",
                    ref, zone_name,
                )
                break

    return groups


def layout_pcb(
    requirements: ProjectRequirements,
    board: BoardOutline,
    footprint_sizes: dict[str, tuple[float, float]] | None = None,
    fixed_positions: dict[str, tuple[float, float, float]] | None = None,
    board_template: object | None = None,
    keepouts: tuple[object, ...] = (),
    footprint_bboxes: dict[str, object] | None = None,
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
            # Pass bboxes through (typed loosely here; solver uses FootprintBBox)
            _bboxes = footprint_bboxes  # type: ignore[assignment]
            result = solve_placement(
                constraint_list, board, footprint_sizes,
                keepouts=typed_keepouts, grid_mm=0.5,
                requirements=requirements,
                footprint_bboxes=_bboxes,
            )
            positions = dict(result.positions)
            rotations = dict(result.rotations)

            # Log any placement violations
            if result.violations:
                for violation in result.violations:
                    log.warning("layout_pcb: placement violation: %s", violation)

            # Post-placement constraint validation
            from kicad_pipeline.pcb.constraints import validate_placement_constraints

            post_violations = validate_placement_constraints(
                result.positions, constraint_list,
            )
            for pv in post_violations:
                log.info("layout_pcb: post-placement: %s", pv)

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

            # Collect layer overrides from constraints
            layer_overrides: dict[str, str] = {}
            for c in constraint_list:
                if c.layer is not None:
                    layer_overrides[c.ref] = c.layer

            log.info("layout_pcb: placed %d components (constraint solver)", len(positions))
            return LayoutResult(
                positions=positions, rotations=rotations,
                layers=layer_overrides if layer_overrides else None,
            )
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

    # Move edge-sensitive components to edge zones
    _edge_priority_sort(groups, requirements)

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}

    # Create dynamic non-overlapping zones based on actual groups
    dynamic_zones = _dynamic_zones(groups, board_w, board_h, footprint_sizes)

    for zone_name, zone_refs in groups.items():
        zone = dynamic_zones[zone_name]
        # Sort components within zone to group related parts
        sorted_refs = _subcircuit_sort(zone_refs, requirements)
        zone_pos = place_pcb_components(
            sorted_refs, zone, footprint_sizes=footprint_sizes,
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


# ---------------------------------------------------------------------------
# Off-board grouped placement
# ---------------------------------------------------------------------------

# Gap between groups (mm)
_GROUP_GAP_MM: float = 20.0
_GROUP_START_OFFSET_MM: float = 25.0
_GROUP_ROW_MAX_WIDTH_MM: float = 300.0
_GROUP_PACKING_FACTOR: float = 2.5
_GROUP_MARGIN_MM: float = 2.0


_CLUSTER_GAP_MM: float = 1.5
"""Horizontal gap between components within a subcircuit cluster."""

_CLUSTER_ROW_GAP_MM: float = 4.0
"""Vertical gap between subcircuit cluster rows within a group."""

# Anchor priority by ref prefix — lower number = higher priority anchor
_ANCHOR_PRIORITY: dict[str, int] = {
    "K": 0,  # Relay
    "U": 1,  # IC
    "Q": 2,  # Transistor
    "J": 3,  # Connector
    "Y": 4,  # Crystal
}

_PASSIVE_STANDOFF_MM: float = 1.0
"""Gap between anchor pad edge and passive pad edge."""

_PASSIVE_STACK_GAP_MM: float = 0.5
"""Gap between stacked passives assigned to the same anchor pin."""

_ANCHOR_GAP_MM: float = 2.0
"""Horizontal gap between anchor components in the group layout."""

_OVERLAP_MAX_PASSES: int = 10
"""Maximum nudge passes for overlap resolution."""


def _ref_prefix(ref: str) -> str:
    """Extract alphabetic prefix from a reference designator."""
    return "".join(ch for ch in ref if ch.isalpha()).upper()


def _is_anchor_ref(ref: str) -> bool:
    """Return True if *ref* is an anchor component (K, U, Q, J, Y)."""
    return _ref_prefix(ref) in _ANCHOR_PRIORITY


def _anchor_priority(ref: str) -> int:
    """Return anchor priority for *ref* (lower = higher priority)."""
    return _ANCHOR_PRIORITY.get(_ref_prefix(ref), 99)


def _classify_pin_side(
    dx: float, dy: float, half_w: float, half_h: float,
) -> str:
    """Classify which side of a footprint a pad is on.

    Uses normalized distance from center to determine left/right/top/bottom.

    Args:
        dx: Pad X offset from footprint center.
        dy: Pad Y offset from footprint center.
        half_w: Half the footprint width.
        half_h: Half the footprint height.

    Returns:
        One of ``"left"``, ``"right"``, ``"top"``, ``"bottom"``.
    """
    # Avoid division by zero for point-like footprints
    norm_x = abs(dx) / max(half_w, 0.01)
    norm_y = abs(dy) / max(half_h, 0.01)

    if norm_x >= norm_y:
        return "right" if dx >= 0 else "left"
    return "bottom" if dy >= 0 else "top"


def _passive_rotation_for_side(
    side: str, connected_pin: str,
) -> float:
    """Compute rotation for a 2-pin passive so connected pad faces anchor.

    At rot=0 a 2-pin passive has pin 1 at (-w/2, 0) and pin 2 at (+w/2, 0).
    We want the connected pin to face *toward* the anchor (inward).

    Args:
        side: Which side of the anchor the passive is placed on.
        connected_pin: Which passive pin connects to the anchor (``"1"`` or ``"2"``).

    Returns:
        Rotation in degrees.
    """
    # Map: side → {pin: rotation} so connected pin faces inward.
    # At rot=0: pin 1 at (-w/2, 0) faces left, pin 2 at (+w/2, 0) faces right.
    # At rot=180: pin 1 at (+w/2, 0) faces right, pin 2 at (-w/2, 0) faces left.
    # At rot=90: pin 1 at (0, -w/2) faces up, pin 2 at (0, +w/2) faces down.
    # At rot=270: pin 1 at (0, +w/2) faces down, pin 2 at (0, -w/2) faces up.
    rotation_table: dict[str, dict[str, float]] = {
        "right": {"1": 0.0,   "2": 180.0},  # connected pin faces left (toward anchor)
        "left":  {"1": 180.0, "2": 0.0},    # connected pin faces right (toward anchor)
        "bottom": {"1": 90.0, "2": 270.0},  # connected pin faces up (toward anchor)
        "top":    {"1": 270.0, "2": 90.0},   # connected pin faces down (toward anchor)
    }
    return rotation_table.get(side, {}).get(connected_pin, 0.0)


def _resolve_overlaps(
    layout: dict[str, tuple[float, float, float]],
    footprint_sizes: dict[str, tuple[float, float]],
) -> None:
    """Nudge overlapping components apart (AABB check, in-place).

    Performs up to ``_OVERLAP_MAX_PASSES`` passes, nudging along the
    smaller overlap axis each time.
    """
    refs = list(layout.keys())
    for _ in range(_OVERLAP_MAX_PASSES):
        moved = False
        for i, r1 in enumerate(refs):
            x1, y1, rot1 = layout[r1]
            w1, h1 = footprint_sizes.get(r1, (5.0, 5.0))
            if rot1 in (90.0, 270.0):
                w1, h1 = h1, w1
            for r2 in refs[i + 1:]:
                x2, y2, rot2 = layout[r2]
                w2, h2 = footprint_sizes.get(r2, (5.0, 5.0))
                if rot2 in (90.0, 270.0):
                    w2, h2 = h2, w2
                min_dx = (w1 + w2) / 2.0 + 0.1
                min_dy = (h1 + h2) / 2.0 + 0.1
                dx = abs(x1 - x2)
                dy = abs(y1 - y2)
                if dx < min_dx and dy < min_dy:
                    # Overlap — nudge along smaller overlap axis
                    overlap_x = min_dx - dx
                    overlap_y = min_dy - dy
                    if overlap_x <= overlap_y:
                        shift = overlap_x / 2.0 + 0.1
                        if x1 <= x2:
                            layout[r1] = (x1 - shift, y1, rot1)
                            layout[r2] = (x2 + shift, y2, rot2)
                        else:
                            layout[r1] = (x1 + shift, y1, rot1)
                            layout[r2] = (x2 - shift, y2, rot2)
                    else:
                        shift = overlap_y / 2.0 + 0.1
                        if y1 <= y2:
                            layout[r1] = (x1, y1 - shift, rot1)
                            layout[r2] = (x2, y2 + shift, rot2)
                        else:
                            layout[r1] = (x1, y1 + shift, rot1)
                            layout[r2] = (x2, y2 - shift, rot2)
                    moved = True
        if not moved:
            break


def _layout_group(
    group_refs: list[str],
    all_constraints: tuple[object, ...],
    footprint_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """Lay out a feature group using pin-aware spatial placement.

    Places each passive adjacent to the specific anchor pin it connects to,
    rotated so connected pads face each other. Anchors are spaced apart with
    secondary anchors (Q) placed pin-adjacent to the primaries they connect to.

    Args:
        group_refs: Component refs belonging to this group.
        all_constraints: Full constraint list (used for subcircuit cluster
            detection — ``_subcircuit_`` prefix GROUP constraints).
        footprint_sizes: Mapping from ref to ``(width, height)`` in mm.
        requirements: Full project requirements for net/pin connectivity.

    Returns:
        Mapping from ref to ``(relative_x, relative_y, rotation)``.
    """
    from kicad_pipeline.pcb.constraints import (
        _build_pad_connectivity,
        _get_component_pad_offsets,
        _rotated_pad_offset,
    )

    group_ref_set = frozenset(group_refs)

    # ------------------------------------------------------------------
    # Step 1: Classify anchors vs passives
    # ------------------------------------------------------------------
    anchors: list[str] = []
    passives: list[str] = []
    for ref in group_refs:
        if _is_anchor_ref(ref):
            anchors.append(ref)
        else:
            passives.append(ref)

    # Sort anchors by priority then ref for determinism
    anchors.sort(key=lambda r: (_anchor_priority(r), r))

    # ------------------------------------------------------------------
    # Step 2: Build pin-level connectivity + cache pad offsets
    # ------------------------------------------------------------------
    pad_conn = _build_pad_connectivity(requirements)

    pad_offsets_cache: dict[str, dict[str, tuple[float, float]] | None] = {}

    def _cached_offsets(ref: str) -> dict[str, tuple[float, float]] | None:
        if ref not in pad_offsets_cache:
            pad_offsets_cache[ref] = _get_component_pad_offsets(ref, requirements)
        return pad_offsets_cache[ref]

    # ------------------------------------------------------------------
    # Step 3: Assign each passive to an anchor + specific pin
    # ------------------------------------------------------------------
    @dataclass
    class _PassiveAssignment:
        passive_ref: str
        passive_pin: str
        anchor_ref: str
        anchor_pin: str

    assignments: list[_PassiveAssignment] = []
    overflow: list[str] = []

    for pref in passives:
        best_anchor: str = ""
        best_anchor_pin: str = ""
        best_passive_pin: str = ""
        best_priority: int = 999

        # Check all pins of this passive for signal-net connections to anchors
        comp = next((c for c in requirements.components if c.ref == pref), None)
        if comp is None:
            overflow.append(pref)
            continue

        for pin in comp.pins:
            neighbours = pad_conn.get((pref, pin.number), [])
            for nb_ref, nb_pin in neighbours:
                if nb_ref not in group_ref_set or not _is_anchor_ref(nb_ref):
                    continue
                pri = _anchor_priority(nb_ref)
                if pri < best_priority:
                    best_priority = pri
                    best_anchor = nb_ref
                    best_anchor_pin = nb_pin
                    best_passive_pin = pin.number

        if best_anchor:
            assignments.append(_PassiveAssignment(
                passive_ref=pref,
                passive_pin=best_passive_pin,
                anchor_ref=best_anchor,
                anchor_pin=best_anchor_pin,
            ))
        else:
            # Check for indirect connection through other passives to anchors
            found = False
            for pin in comp.pins:
                neighbours = pad_conn.get((pref, pin.number), [])
                for nb_ref, _nb_pin in neighbours:
                    if nb_ref in group_ref_set and not _is_anchor_ref(nb_ref):
                        # This passive connects to another passive —
                        # find which anchor the intermediary connects to
                        nb_comp = next(
                            (c for c in requirements.components if c.ref == nb_ref),
                            None,
                        )
                        if nb_comp is None:
                            continue
                        for nb_p in nb_comp.pins:
                            for nn_ref, nn_pin in pad_conn.get((nb_ref, nb_p.number), []):
                                if nn_ref in group_ref_set and _is_anchor_ref(nn_ref):
                                    pri = _anchor_priority(nn_ref)
                                    if pri < best_priority:
                                        best_priority = pri
                                        best_anchor = nn_ref
                                        best_anchor_pin = nn_pin
                                        best_passive_pin = pin.number
                                        found = True
                    if found:
                        break
                if found:
                    break
            if best_anchor:
                assignments.append(_PassiveAssignment(
                    passive_ref=pref,
                    passive_pin=best_passive_pin,
                    anchor_ref=best_anchor,
                    anchor_pin=best_anchor_pin,
                ))
            else:
                overflow.append(pref)

    # ------------------------------------------------------------------
    # Step 4: Classify anchor pins into sides using pad geometry
    # ------------------------------------------------------------------
    anchor_pin_sides: dict[str, dict[str, str]] = {}  # anchor_ref → {pin: side}
    for aref in anchors:
        offsets = _cached_offsets(aref)
        if offsets is None:
            anchor_pin_sides[aref] = {}
            continue
        w, h = footprint_sizes.get(aref, (5.0, 5.0))
        half_w, half_h = w / 2.0, h / 2.0
        sides: dict[str, str] = {}
        for pin_num, (dx, dy) in offsets.items():
            sides[pin_num] = _classify_pin_side(dx, dy, half_w, half_h)
        anchor_pin_sides[aref] = sides

    # ------------------------------------------------------------------
    # Step 5: Place anchors
    # ------------------------------------------------------------------
    layout: dict[str, tuple[float, float, float]] = {}

    # Separate primary anchors (K, U, J) from secondary (Q)
    primary_anchors = [a for a in anchors if _ref_prefix(a) in ("K", "U", "J", "Y")]
    secondary_anchors = [a for a in anchors if _ref_prefix(a) not in ("K", "U", "J", "Y")]

    # Detect relay groups: if ALL primary anchors are relays (K refs), use
    # single-row layout (1xN) — relays should never wrap to multiple rows.
    is_relay_group = (
        len(primary_anchors) >= 2
        and all(_ref_prefix(a) == "K" for a in primary_anchors)
    )

    # Place primaries in rows, wrapping to 2D grid when row exceeds threshold
    # Relay groups: no wrapping (single row), support components go below
    max_row_width = 80.0 if is_relay_group else 35.0
    cursor_x = _GROUP_MARGIN_MM
    anchor_y = _GROUP_MARGIN_MM
    max_anchor_h = 0.0
    row_h = 0.0  # height of current row for wrapping

    for aref in primary_anchors:
        w, h = footprint_sizes.get(aref, (5.0, 5.0))
        # Wrap to next row if this anchor would exceed max width
        if cursor_x + w > max_row_width and cursor_x > _GROUP_MARGIN_MM + 1:
            anchor_y += row_h + _ANCHOR_GAP_MM
            cursor_x = _GROUP_MARGIN_MM
            row_h = 0.0
        layout[aref] = (cursor_x + w / 2.0, anchor_y + h / 2.0, 0.0)
        cursor_x += w + _ANCHOR_GAP_MM
        row_h = max(row_h, h)
        max_anchor_h = max(max_anchor_h, anchor_y + h - _GROUP_MARGIN_MM)

    # Place secondary anchors (Q) near the primary they connect to
    for sref in secondary_anchors:
        connected_primary: str = ""
        connected_primary_pin: str = ""
        s_comp = next((c for c in requirements.components if c.ref == sref), None)
        if s_comp is not None:
            for pin in s_comp.pins:
                for nb_ref, nb_pin in pad_conn.get((sref, pin.number), []):
                    if nb_ref in primary_anchors:
                        connected_primary = nb_ref
                        connected_primary_pin = nb_pin
                        break
                if connected_primary:
                    break

        if connected_primary and connected_primary in layout:
            # Place Q near the connected pin of the primary
            prim_x, prim_y, _ = layout[connected_primary]
            offsets = _cached_offsets(connected_primary)
            sw, sh = footprint_sizes.get(sref, (3.0, 3.0))
            pw, ph = footprint_sizes.get(connected_primary, (5.0, 5.0))

            if offsets and connected_primary_pin in offsets:
                pdx, pdy = offsets[connected_primary_pin]
                side = _classify_pin_side(pdx, pdy, pw / 2.0, ph / 2.0)
                if side == "right":
                    sx = prim_x + pw / 2.0 + sw / 2.0 + _PASSIVE_STANDOFF_MM
                    sy = prim_y + pdy
                elif side == "left":
                    sx = prim_x - pw / 2.0 - sw / 2.0 - _PASSIVE_STANDOFF_MM
                    sy = prim_y + pdy
                elif side == "bottom":
                    sx = prim_x + pdx
                    sy = prim_y + ph / 2.0 + sh / 2.0 + _PASSIVE_STANDOFF_MM
                else:  # top
                    sx = prim_x + pdx
                    sy = prim_y - ph / 2.0 - sh / 2.0 - _PASSIVE_STANDOFF_MM
                layout[sref] = (sx, sy, 0.0)
            else:
                # Fallback: place below primary
                layout[sref] = (
                    prim_x,
                    prim_y + ph / 2.0 + sh / 2.0 + _PASSIVE_STANDOFF_MM,
                    0.0,
                )
        else:
            # Unconnected secondary — place at end of primary row
            w, h = footprint_sizes.get(sref, (3.0, 3.0))
            layout[sref] = (cursor_x + w / 2.0, anchor_y + h / 2.0, 0.0)
            cursor_x += w + _ANCHOR_GAP_MM

    # ------------------------------------------------------------------
    # Step 6: Place passives at anchor pins
    # ------------------------------------------------------------------
    # Group assignments by (anchor_ref, anchor_pin)
    pin_assignments: dict[tuple[str, str], list[_PassiveAssignment]] = {}
    for asn in assignments:
        key = (asn.anchor_ref, asn.anchor_pin)
        pin_assignments.setdefault(key, []).append(asn)

    for (anchor_ref, anchor_pin), asn_list in pin_assignments.items():
        if anchor_ref not in layout:
            # Anchor not placed (shouldn't happen, but guard)
            continue

        anchor_x, anchor_y_pos, anchor_rot = layout[anchor_ref]
        offsets = _cached_offsets(anchor_ref)
        aw, ah = footprint_sizes.get(anchor_ref, (5.0, 5.0))

        if offsets and anchor_pin in offsets:
            pdx, pdy = offsets[anchor_pin]
            rot_dx, rot_dy = _rotated_pad_offset(pdx, pdy, anchor_rot)
            pad_abs_x = anchor_x + rot_dx
            pad_abs_y = anchor_y_pos + rot_dy
            side = anchor_pin_sides.get(anchor_ref, {}).get(anchor_pin, "right")
        else:
            # Fallback: place to the right
            pad_abs_x = anchor_x + aw / 2.0
            pad_abs_y = anchor_y_pos
            side = "right"

        # Place each passive outward from the pin, stacking if multiple
        for stack_idx, asn in enumerate(asn_list):
            pw, ph = footprint_sizes.get(asn.passive_ref, (1.6, 0.8))
            rot = _passive_rotation_for_side(side, asn.passive_pin)

            # Effective passive size after rotation
            eff_w, eff_h = pw, ph
            if rot in (90.0, 270.0):
                eff_w, eff_h = ph, pw

            stack_offset = stack_idx * (eff_w + _PASSIVE_STACK_GAP_MM)

            if side == "right":
                px = pad_abs_x + _PASSIVE_STANDOFF_MM + eff_w / 2.0 + stack_offset
                py = pad_abs_y
            elif side == "left":
                px = pad_abs_x - _PASSIVE_STANDOFF_MM - eff_w / 2.0 - stack_offset
                py = pad_abs_y
            elif side == "bottom":
                px = pad_abs_x
                py = pad_abs_y + _PASSIVE_STANDOFF_MM + eff_h / 2.0 + stack_offset
            else:  # top
                px = pad_abs_x
                py = pad_abs_y - _PASSIVE_STANDOFF_MM - eff_h / 2.0 - stack_offset

            layout[asn.passive_ref] = (px, py, rot)

    # ------------------------------------------------------------------
    # Step 7: Resolve overlaps
    # ------------------------------------------------------------------
    _resolve_overlaps(layout, footprint_sizes)

    # ------------------------------------------------------------------
    # Step 7.5: Relay group post-processing — support below relay row
    # ------------------------------------------------------------------
    if is_relay_group:
        # Reorganize: K refs in a single top row, all support components
        # (Q, D, R, C) placed directly below their associated relay.
        relay_refs = sorted(
            [r for r in layout if _ref_prefix(r) == "K"],
            key=lambda r: layout[r][0],  # left to right
        )
        if relay_refs:
            # Find the bottom of the relay row
            relay_bottom = max(
                layout[r][1] + footprint_sizes.get(r, (18.0, 15.0))[1] / 2.0
                for r in relay_refs
            )
            support_y_start = relay_bottom + _PASSIVE_STANDOFF_MM

            # Map each non-relay ref to its associated relay
            for ref in list(layout.keys()):
                if _ref_prefix(ref) == "K":
                    continue
                # Find which relay this ref is assigned to (via pad_conn)
                owning_relay = ""
                comp = next(
                    (c for c in requirements.components if c.ref == ref), None,
                )
                if comp:
                    for pin in comp.pins:
                        for nb_ref, _nb_pin in pad_conn.get(
                            (ref, pin.number), [],
                        ):
                            if nb_ref in relay_refs:
                                owning_relay = nb_ref
                                break
                            # Indirect: check through any non-K ref
                            if _ref_prefix(nb_ref) != "K":
                                nb_comp = next(
                                    (c for c in requirements.components
                                     if c.ref == nb_ref),
                                    None,
                                )
                                if nb_comp:
                                    for nb_p in nb_comp.pins:
                                        for nn_ref, _ in pad_conn.get(
                                            (nb_ref, nb_p.number), [],
                                        ):
                                            if nn_ref in relay_refs:
                                                owning_relay = nn_ref
                                                break
                                        if owning_relay:
                                            break
                        if owning_relay:
                            break

                if owning_relay and owning_relay in layout:
                    relay_x = layout[owning_relay][0]
                else:
                    # Fallback: place under first relay
                    relay_x = layout[relay_refs[0]][0]

                # Place below the relay, stacking support components
                w, h = footprint_sizes.get(ref, (2.0, 2.0))
                # Find next free Y slot under this relay
                occupied_ys = [
                    layout[r][1] + footprint_sizes.get(r, (2.0, 2.0))[1] / 2.0
                    for r in layout
                    if r != ref
                    and _ref_prefix(r) != "K"
                    and abs(layout[r][0] - relay_x) < 10.0
                    and layout[r][1] > relay_bottom
                ]
                slot_y = (
                    max(occupied_ys) + h / 2.0 + 1.5
                    if occupied_ys
                    else support_y_start + h / 2.0
                )
                layout[ref] = (relay_x, slot_y, 0.0)

    # ------------------------------------------------------------------
    # Step 8: Overflow row for unassigned passives
    # ------------------------------------------------------------------
    if overflow:
        # Place below all existing layout
        if layout:
            max_y = max(
                pos[1] + footprint_sizes.get(ref, (5.0, 5.0))[1] / 2.0
                for ref, pos in layout.items()
            )
        else:
            max_y = _GROUP_MARGIN_MM
        overflow_y = max_y + _CLUSTER_ROW_GAP_MM
        overflow_x = _GROUP_MARGIN_MM
        for ref in sorted(overflow):
            w, h = footprint_sizes.get(ref, (5.0, 5.0))
            layout[ref] = (overflow_x + w / 2.0, overflow_y + h / 2.0, 0.0)
            overflow_x += w + _CLUSTER_GAP_MM

    return layout


def place_groups_off_board(
    footprints: tuple[object, ...],
    features: tuple[object, ...],
    requirements: ProjectRequirements,
    board_height_mm: float,
    footprint_sizes: dict[str, tuple[float, float]],
    fixed_positions: dict[str, tuple[float, float, float]] | None = None,
) -> LayoutResult:
    """Place component groups off-board with subcircuit-aware internal layout.

    Each FeatureBlock becomes a group placed below the board outline.
    Within each group, subcircuit clusters (relay drivers, NPN drivers, etc.)
    are detected from constraints and laid out as tight horizontal rows.

    Args:
        footprints: All footprints in the design.
        features: FeatureBlock objects from requirements.
        requirements: Full project requirements.
        board_height_mm: Height of the real board in mm.
        footprint_sizes: Mapping from ref to ``(width, height)`` in mm.
        fixed_positions: Optional mapping from ref to ``(x, y, rotation)``
            for preserved positions.

    Returns:
        :class:`LayoutResult` with all components placed below the board.
    """
    from kicad_pipeline.models.requirements import FeatureBlock
    from kicad_pipeline.pcb.constraints import constraints_from_requirements

    typed_features = tuple(f for f in features if isinstance(f, FeatureBlock))

    # 1. Build group membership: ref → group name
    feature_map: dict[str, str] = {}
    for fb in typed_features:
        for ref in fb.components:
            feature_map[ref] = fb.name

    all_refs = {c.ref for c in requirements.components}

    # 2. Handle fixed positions (preserve_from support)
    positions: dict[str, Point] = {}
    rotations: dict[str, float] = {}
    fixed_refs: set[str] = set()
    if fixed_positions:
        for ref, (fx, fy, frot) in fixed_positions.items():
            if ref in all_refs:
                positions[ref] = Point(x=fx, y=fy)
                rotations[ref] = frot
                fixed_refs.add(ref)

    # 3. Generate full constraints for subcircuit cluster detection
    full_constraints = constraints_from_requirements(
        requirements, None, footprint_sizes,
    )

    # 4. Group remaining refs by FeatureBlock
    groups: dict[str, list[str]] = {}
    for ref in all_refs:
        if ref in fixed_refs:
            continue
        group_name = feature_map.get(ref, "Ungrouped")
        groups.setdefault(group_name, []).append(ref)

    # Sort refs within each group for determinism
    for refs in groups.values():
        refs.sort()

    # 5. For each group, use subcircuit-aware layout
    group_layouts: dict[str, dict[str, tuple[float, float, float]]] = {}
    group_dimensions: dict[str, tuple[float, float]] = {}

    for group_name, group_refs in groups.items():
        if not group_refs:
            continue

        layout = _layout_group(
            group_refs, full_constraints, footprint_sizes, requirements,
        )
        group_layouts[group_name] = layout

        # Compute bounding box
        if layout:
            min_x = min(pos[0] - footprint_sizes.get(ref, (5.0, 5.0))[0] / 2.0
                        for ref, pos in layout.items())
            max_x = max(pos[0] + footprint_sizes.get(ref, (5.0, 5.0))[0] / 2.0
                        for ref, pos in layout.items())
            min_y = min(pos[1] - footprint_sizes.get(ref, (5.0, 5.0))[1] / 2.0
                        for ref, pos in layout.items())
            max_y = max(pos[1] + footprint_sizes.get(ref, (5.0, 5.0))[1] / 2.0
                        for ref, pos in layout.items())
            group_dimensions[group_name] = (max_x - min_x + 4.0, max_y - min_y + 4.0)
        else:
            group_dimensions[group_name] = (20.0, 20.0)

    # 6. Arrange groups below the real board
    start_y = board_height_mm + _GROUP_START_OFFSET_MM
    cursor_x = 0.0
    cursor_y = start_y
    row_max_h = 0.0

    for group_name in sorted(group_layouts.keys()):
        layout = group_layouts[group_name]
        gw, gh = group_dimensions[group_name]

        # Wrap to next row if needed
        if cursor_x + gw > _GROUP_ROW_MAX_WIDTH_MM and cursor_x > 0.0:
            cursor_y += row_max_h + _GROUP_GAP_MM
            cursor_x = 0.0
            row_max_h = 0.0

        # Translate group positions to final off-board coordinates
        for ref, (rx, ry, rrot) in layout.items():
            positions[ref] = Point(x=cursor_x + rx, y=cursor_y + ry)
            rotations[ref] = rrot

        cursor_x += gw + _GROUP_GAP_MM
        row_max_h = max(row_max_h, gh)

    log.info(
        "place_groups_off_board: placed %d components in %d groups below board",
        len(positions), len(group_layouts),
    )
    return LayoutResult(positions=positions, rotations=rotations, layers=None)
