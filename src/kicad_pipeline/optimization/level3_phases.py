"""Level-3 intra-group refinement phases for the EE placement optimizer.

Contains standalone placement functions that run as sub-phases of Level 3
in ``optimize_placement_ee()``.  Extracted from placement_optimizer.py to
reduce module size.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    BOARD_EDGE_MARGIN_MM,
    CONNECTOR_EDGE_MARGIN_MM,
    DEFAULT_FP_SIZE_MM,
    DEFAULT_IC_SIZE_MM,
)
from kicad_pipeline.optimization.collision_resolver import (
    _PlacementGrid,
    _rotation_aware_size,
)
from kicad_pipeline.optimization.functional_grouper import (
    BoardZoneAssignment,
    DetectedSubCircuit,
    DomainAffinity,
    SubCircuitType,
    VoltageDomain,
)
from kicad_pipeline.pcb.pin_map import (
    centroid_to_origin,
    origin_to_centroid,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.zone_partitioner import BoardZone

_log = logging.getLogger(__name__)


def _place_row_layout(
    subcircuits: Sequence[DetectedSubCircuit],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    zones: Sequence[BoardZone] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Place same-type subcircuits with layout_hint='row' in a 1xN horizontal row.

    Used for relay banks -- places relay anchors in a horizontal line with
    tight spacing, then places each relay's sub-components around it.
    """
    min_x, min_y, max_x, max_y = bounds

    # Group by circuit_type for row layout
    type_groups: dict[str, list[DetectedSubCircuit]] = {}
    for sc in subcircuits:
        if sc.layout_hint != "row":
            continue
        type_groups.setdefault(sc.circuit_type.value, []).append(sc)

    for _circuit_type, group in type_groups.items():
        if len(group) < 2:
            continue

        # Find the current average position of anchors
        anchor_positions = []
        for sc in group:
            if sc.anchor_ref in positions and sc.anchor_ref not in fixed_refs:
                x, y, rot = positions[sc.anchor_ref]
                anchor_positions.append((x, y, sc))

        if len(anchor_positions) < 2:
            continue

        # Compute the row center and direction
        avg_y = sum(p[1] for p in anchor_positions) / len(anchor_positions)

        # Sort anchors left-to-right
        anchor_positions.sort(key=lambda p: p[0])

        # Relay rotation: 90 deg for vertical coil orientation
        relay_rotation = 90.0

        # Compute row spacing based on anchor widths (swapped for 90 deg rotation)
        total_width = 0.0
        for _, _, sc in anchor_positions:
            aw, ah = fp_sizes.get(sc.anchor_ref, DEFAULT_IC_SIZE_MM)
            # Swap w/h because relay is rotated 90 deg
            aw, ah = ah, aw
            total_width += aw + 2.0  # gap between relays

        # Place anchors in a row, constrained to relay zone if available
        relay_zone_x1 = min_x + BOARD_EDGE_MARGIN_MM
        relay_zone_x2 = max_x - 15.0  # leave room for edge connectors
        if zones:
            for z in zones:
                if z.name == "relay":
                    relay_zone_x1 = max(min_x + BOARD_EDGE_MARGIN_MM, z.rect[0])
                    relay_zone_x2 = min(max_x - 15.0, z.rect[2] - 5.0)
                    break
        # Ensure rightmost relay stays within board with margin
        # Each relay at 90 deg has X half-width = aw/2 ~ 8.8mm
        max_relay_half_w = max(
            (fp_sizes.get(sc.anchor_ref, DEFAULT_IC_SIZE_MM)[1] for _, _, sc in anchor_positions),
            default=8.8,
        ) / 2.0
        relay_zone_x2 = min(relay_zone_x2, max_x - max_relay_half_w - 1.5)
        # Center the row within the available zone X range
        zone_center_x = (relay_zone_x1 + relay_zone_x2) / 2.0
        start_x = zone_center_x - total_width / 2.0
        # Clamp to zone bounds
        start_x = max(relay_zone_x1, start_x)
        if start_x + total_width > relay_zone_x2:
            start_x = relay_zone_x2 - total_width
        row_grid = _PlacementGrid(bounds)

        # Register all non-row components first
        row_refs: set[str] = set()
        for _, _, sc in anchor_positions:
            row_refs.update(sc.refs)
        for ref, (ox, oy, _orot) in positions.items():
            if ref not in row_refs:
                ow, oh = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
                row_grid.place(ox, oy, ow, oh)

        # Force all relay anchors to the same Y (avg_y) in a tight row
        # Ensure minimum Y so relays don't overlap top-edge connectors.
        # Find the lowest bottom edge of any component above the relay row
        # in the relay X range to prevent vertical overlap.
        max_relay_h = max(
            (lambda w, h: h if relay_rotation % 180 == 0 else w)(
                *fp_sizes.get(sc.anchor_ref, (18.0, 16.0))
            )
            for _, _, sc in anchor_positions
        )
        # Check for components in the top area that overlap the relay X range
        relay_x_min = start_x
        relay_x_max = start_x + total_width
        top_obstacle_y = min_y  # highest bottom edge of obstacles above
        for ref, (ox, oy, _orot) in positions.items():
            if ref in row_refs:
                continue
            ow, oh = _rotation_aware_size(ref, positions, fp_sizes)
            # Check X overlap with relay row
            if ox + ow / 2 > relay_x_min and ox - ow / 2 < relay_x_max:
                obstacle_bot = oy + oh / 2.0
                if obstacle_bot < avg_y:  # obstacle is above relay row
                    top_obstacle_y = max(top_obstacle_y, obstacle_bot)
        min_relay_y = top_obstacle_y + max_relay_h / 2.0 + 2.0
        # Use relay zone Y if available -- ensures relays are placed within
        # their assigned zone, well below screw terminals.
        if zones:
            for z in zones:
                if z.name == "relay":
                    # Place relay centroids at the vertical center of the relay zone
                    zone_center_y = (z.rect[1] + z.rect[3]) / 2.0
                    min_relay_y = max(min_relay_y, z.rect[1] + max_relay_h / 2.0 + 2.0)
                    avg_y = max(avg_y, zone_center_y)
                    break
        # Fallback: enforce minimum 25mm from top edge (below screw terminal zone)
        min_relay_y = max(min_relay_y, min_y + max_relay_h / 2.0 + 15.0)
        row_y = max(min_relay_y, avg_y)

        cursor_x = start_x
        for _, _, sc in anchor_positions:
            anchor_ref = sc.anchor_ref
            aw, ah = fp_sizes.get(anchor_ref, DEFAULT_IC_SIZE_MM)
            # Swap w/h for 90 deg rotation
            aw, ah = ah, aw
            target_x = cursor_x + aw / 2.0
            # Leave 15mm margin on right for edge connectors (J14, J15)
            target_x = max(min_x + BOARD_EDGE_MARGIN_MM, min(max_x - 15.0, target_x))
            target_y = max(min_y + BOARD_EDGE_MARGIN_MM, min(max_y - BOARD_EDGE_MARGIN_MM, row_y))

            # Place relay at 90 deg rotation for vertical coil orientation
            positions[anchor_ref] = (target_x, target_y, relay_rotation)
            row_grid.place(target_x, target_y, aw, ah)

            # Support components are placed by Level 3b -- skip here to
            # avoid double-placement and congestion.

            cursor_x += aw + 2.0

    return positions


def _place_boundary_regulators(
    subcircuits: Sequence[DetectedSubCircuit],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    domain_map: dict[str, VoltageDomain],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    zone_assignments: tuple[BoardZoneAssignment, ...] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Place regulators at the boundary between their input and output domains.

    When *zone_assignments* are available, uses the shared edge between
    input and output zone rects. Falls back to midpoint between domain
    centroids.
    """
    min_x, min_y, max_x, max_y = bounds

    # Build zone rect lookup
    zone_rects: dict[VoltageDomain, tuple[float, float, float, float]] = {}
    if zone_assignments:
        for za in zone_assignments:
            zone_rects[za.domain] = za.zone_rect

    # Compute domain centroids (fallback)
    domain_positions: dict[VoltageDomain, list[tuple[float, float]]] = {}
    for ref, (x, y, _rot) in positions.items():
        d = domain_map.get(ref)
        if d is not None and d != VoltageDomain.MIXED:
            domain_positions.setdefault(d, []).append((x, y))

    domain_centroids: dict[VoltageDomain, tuple[float, float]] = {}
    for d, pts in domain_positions.items():
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        domain_centroids[d] = (cx, cy)

    for sc in subcircuits:
        if sc.layout_hint != "boundary":
            continue
        if sc.input_domain is None or sc.output_domain is None:
            continue
        if sc.anchor_ref in fixed_refs or sc.anchor_ref not in positions:
            continue

        # Try zone rects first: find shared edge between input/output zones
        in_rect = zone_rects.get(sc.input_domain)
        out_rect = zone_rects.get(sc.output_domain)

        if in_rect and out_rect:
            # Find the shared boundary between the two zone rects
            # Check if they share a vertical boundary (left-right layout)
            ix1, iy1, ix2, iy2 = in_rect
            ox1, oy1, ox2, oy2 = out_rect
            # Input right edge meets output left edge
            if abs(ix2 - ox1) < 12.0:
                target_x = (ix2 + ox1) / 2.0 + min_x
                target_y = (max(iy1, oy1) + min(iy2, oy2)) / 2.0 + min_y
            # Input bottom edge meets output top edge
            elif abs(iy2 - oy1) < 12.0:
                target_x = (max(ix1, ox1) + min(ix2, ox2)) / 2.0 + min_x
                target_y = (iy2 + oy1) / 2.0 + min_y
            else:
                # No clear shared edge -- use centroid midpoint
                in_c = domain_centroids.get(sc.input_domain)
                out_c = domain_centroids.get(sc.output_domain)
                if in_c is None or out_c is None:
                    continue
                target_x = (in_c[0] + out_c[0]) / 2.0
                target_y = (in_c[1] + out_c[1]) / 2.0
        else:
            in_centroid = domain_centroids.get(sc.input_domain)
            out_centroid = domain_centroids.get(sc.output_domain)
            if in_centroid is None or out_centroid is None:
                continue
            # Target: midpoint between domain centroids
            target_x = (in_centroid[0] + out_centroid[0]) / 2.0
            target_y = (in_centroid[1] + out_centroid[1]) / 2.0
        target_x = max(min_x + BOARD_EDGE_MARGIN_MM, min(max_x - BOARD_EDGE_MARGIN_MM, target_x))
        target_y = max(min_y + BOARD_EDGE_MARGIN_MM, min(max_y - BOARD_EDGE_MARGIN_MM, target_y))

        # Check if moving is actually closer to boundary
        ax, ay, arot = positions[sc.anchor_ref]
        aw, ah = fp_sizes.get(sc.anchor_ref, DEFAULT_FP_SIZE_MM)
        current_dist = math.sqrt((ax - target_x) ** 2 + (ay - target_y) ** 2)

        if current_dist < 3.0:
            continue  # Already near boundary

        # Build grid without this subcircuit's refs
        move_grid = _PlacementGrid(bounds)
        sc_refs_set = set(sc.refs)
        for ref, (ox, oy, _orot) in positions.items():
            if ref in sc_refs_set:
                continue
            ow, oh = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
            move_grid.place(ox, oy, ow, oh)

        fx, fy = move_grid.find_free_pos(target_x, target_y, aw, ah)
        new_dist = math.sqrt((fx - target_x) ** 2 + (fy - target_y) ** 2)
        if new_dist < current_dist:
            positions[sc.anchor_ref] = (fx, fy, arot)
            move_grid.place(fx, fy, aw, ah)

            # Pull sub-circuit members toward new anchor position
            for ref in sc.refs:
                if ref == sc.anchor_ref or ref in fixed_refs or ref not in positions:
                    continue
                rx, ry, rrot = positions[ref]
                w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
                ideal_dist = (w + aw) / 2.0 + 1.0
                rdist = math.sqrt((rx - fx) ** 2 + (ry - fy) ** 2)
                if rdist <= ideal_dist + 1.0:
                    continue
                if rdist < 0.01:
                    continue
                dx = (fx - rx) / rdist
                dy = (fy - ry) / rdist
                tx = fx - dx * ideal_dist
                ty = fy - dy * ideal_dist
                tx = max(min_x + BOARD_EDGE_MARGIN_MM, min(max_x - BOARD_EDGE_MARGIN_MM, tx))
                ty = max(min_y + BOARD_EDGE_MARGIN_MM, min(max_y - BOARD_EDGE_MARGIN_MM, ty))
                mrx, mry = move_grid.find_free_pos(tx, ty, w, h)
                if math.sqrt((mrx - fx) ** 2 + (mry - fy) ** 2) < rdist:
                    move_grid.place(mrx, mry, w, h)
                    positions[ref] = (mrx, mry, rrot)

    return positions


def _pin_rf_to_edge(
    subcircuits: Sequence[DetectedSubCircuit],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
) -> dict[str, tuple[float, float, float]]:
    """Pin RF antenna modules to the nearest board edge.

    Sets rotation so antenna faces outward.
    """
    min_x, min_y, max_x, max_y = bounds
    edge_margin = BOARD_EDGE_MARGIN_MM

    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.RF_ANTENNA:
            continue
        if sc.anchor_ref in fixed_refs or sc.anchor_ref not in positions:
            continue

        cx, cy, rot = positions[sc.anchor_ref]
        w, h = fp_sizes.get(sc.anchor_ref, DEFAULT_IC_SIZE_MM)

        # Find nearest edge
        dist_left = cx - min_x
        dist_right = max_x - cx
        dist_top = cy - min_y
        dist_bottom = max_y - cy
        min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        # Determine target position and rotation for antenna facing outward.
        # ESP32-S3-WROOM-1: antenna is at the TOP of the package (opposite
        # pin 1). KiCad rotation convention: 0 deg=antenna up, 90 deg=antenna left,
        # 180 deg=antenna down, 270 deg=antenna right.
        target_x, target_y = cx, cy
        new_rot = rot
        if dist_right == min_edge_dist or dist_right <= dist_left:
            target_x = max_x - edge_margin - w / 2.0
            new_rot = 270.0  # Antenna pointing right (toward right edge)
        elif dist_left == min_edge_dist:
            target_x = min_x + edge_margin + w / 2.0
            new_rot = 90.0  # Antenna pointing left
        elif dist_top == min_edge_dist:
            target_y = min_y + edge_margin + h / 2.0
            new_rot = 0.0  # Antenna pointing up (toward top edge)
        else:
            target_y = max_y - edge_margin - h / 2.0
            new_rot = 180.0  # Antenna pointing down

        # Build grid without RF module
        move_grid = _PlacementGrid(bounds)
        for ref, (ox, oy, _orot) in positions.items():
            if ref == sc.anchor_ref:
                continue
            ow, oh = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
            move_grid.place(ox, oy, ow, oh)

        fx, fy = move_grid.find_free_pos(target_x, target_y, w, h)
        positions[sc.anchor_ref] = (fx, fy, new_rot)

    return positions


def _pull_mcu_peripherals(
    subcircuits: Sequence[DetectedSubCircuit],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
) -> dict[str, tuple[float, float, float]]:
    """Pull MCU peripheral cluster members tight to the MCU.

    Ensures switches, LEDs, and debug connectors are within the
    MCU_PERIPHERAL_MAX_DISTANCE_MM threshold.
    """
    from kicad_pipeline.constants import MCU_PERIPHERAL_MAX_DISTANCE_MM

    min_x, min_y, max_x, max_y = bounds

    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.MCU_PERIPHERAL_CLUSTER:
            continue
        anchor = sc.anchor_ref
        if anchor not in positions:
            continue
        ax, ay, _arot = positions[anchor]
        aw, ah = fp_sizes.get(anchor, DEFAULT_IC_SIZE_MM)

        for ref in sc.refs:
            if ref == anchor or ref in fixed_refs or ref not in positions:
                continue
            rx, ry, rrot = positions[ref]
            w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
            current_dist = math.sqrt((rx - ax) ** 2 + (ry - ay) ** 2)

            if current_dist <= MCU_PERIPHERAL_MAX_DISTANCE_MM:
                continue

            # Target: just outside MCU body
            ideal_dist = (w + aw) / 2.0 + 2.0
            if current_dist < 0.01:
                continue
            dx = (ax - rx) / current_dist
            dy = (ay - ry) / current_dist
            target_x = ax - dx * ideal_dist
            target_y = ay - dy * ideal_dist
            target_x = max(min_x + BOARD_EDGE_MARGIN_MM, min(max_x - BOARD_EDGE_MARGIN_MM, target_x))
            target_y = max(min_y + BOARD_EDGE_MARGIN_MM, min(max_y - BOARD_EDGE_MARGIN_MM, target_y))

            move_grid = _PlacementGrid(bounds)
            for other_ref, (ox, oy, _orot) in positions.items():
                if other_ref == ref:
                    continue
                ow, oh = fp_sizes.get(other_ref, DEFAULT_FP_SIZE_MM)
                move_grid.place(ox, oy, ow, oh)

            fx, fy = move_grid.find_free_pos(target_x, target_y, w, h)
            new_dist = math.sqrt((fx - ax) ** 2 + (fy - ay) ** 2)
            if new_dist < current_dist:
                positions[ref] = (fx, fy, rrot)

    return positions


def _orient_connectors(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    pcb: PCBDesign,
) -> dict[str, tuple[float, float, float]]:
    """Position and orient connectors at their nearest board edge.

    Uses pin_map.origin_to_centroid() and pad_extent_in_board_space() to
    correctly handle asymmetric footprints (connectors with pin-1 origin).

    Screw terminal rotation conventions (mating face outward):
    - Top edge: rot=90 deg (pads run vertically, screws accessible from top)
    - Bottom edge: rot=270 deg (screws accessible from bottom)
    - Left edge: rot=0 deg (screws accessible from left)
    - Right edge: rot=180 deg (screws accessible from right)
    """
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space

    min_x, min_y, max_x, max_y = bounds
    edge_margin = CONNECTOR_EDGE_MARGIN_MM

    for fp in pcb.footprints:
        ref = fp.ref
        if ref in fixed_refs or not ref.startswith("J"):
            continue
        if ref not in positions:
            continue
        cx, cy, rot = positions[ref]
        w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)

        # Find nearest edge (using centroid position)
        dist_left = cx - min_x
        dist_right = max_x - cx
        dist_top = cy - min_y
        dist_bottom = max_y - cy
        min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_edge_dist > 20.0:
            continue  # Too far from any edge -- not a board-edge connector

        # Determine target edge and rotation
        is_wide = w > h * 1.5  # Multi-pin in a row (native, unrotated)

        if dist_top == min_edge_dist:
            target_edge = "top"
            new_rot = 90.0 if is_wide else 0.0
        elif dist_bottom == min_edge_dist:
            target_edge = "bottom"
            new_rot = 270.0 if is_wide else 0.0
        elif dist_left == min_edge_dist:
            target_edge = "left"
            new_rot = 0.0 if is_wide else 270.0
        else:
            target_edge = "right"
            new_rot = 180.0 if is_wide else 90.0

        # Use centroid_to_origin to find where origin would be at current
        # centroid, then compute pad extent to find how far pads extend
        origin_x, origin_y = centroid_to_origin(fp, cx, cy, new_rot)

        # Get actual pad extent at this position and rotation
        px0, py0, px1, py1 = pad_extent_in_board_space(
            fp, origin_x, origin_y, new_rot,
        )

        # Shift origin so pads are flush to target edge with margin
        if target_edge == "top":
            # Move so topmost pad is at min_y + margin
            shift_y = (min_y + edge_margin) - py0
            origin_y += shift_y
        elif target_edge == "bottom":
            # Move so bottommost pad is at max_y - margin
            shift_y = (max_y - edge_margin) - py1
            origin_y += shift_y
        elif target_edge == "left":
            shift_x = (min_x + edge_margin) - px0
            origin_x += shift_x
        else:  # right
            shift_x = (max_x - edge_margin) - px1
            origin_x += shift_x

        # Clamp: verify all pads are within board after shift
        px0, py0, px1, py1 = pad_extent_in_board_space(
            fp, origin_x, origin_y, new_rot,
        )
        if px0 < min_x + 1.0:
            origin_x += (min_x + 1.0) - px0
        if px1 > max_x - 1.0:
            origin_x -= px1 - (max_x - 1.0)
        if py0 < min_y + 1.0:
            origin_y += (min_y + 1.0) - py0
        if py1 > max_y - 1.0:
            origin_y -= py1 - (max_y - 1.0)

        # Convert back to centroid space for the optimizer
        new_cx, new_cy = origin_to_centroid(fp, origin_x, origin_y, new_rot)
        positions[ref] = (new_cx, new_cy, new_rot)
        _log.debug("  %s: edge=%s -> origin(%.1f,%.1f) centroid(%.1f,%.1f) rot=%.0f",
                   ref, target_edge, origin_x, origin_y, new_cx, new_cy, new_rot)

    return positions


def _classify_connector_function(
    ref: str,
    subcircuits: tuple[DetectedSubCircuit, ...],
    adj: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
) -> str:
    """Classify a connector by its functional role based on signal adjacency.

    Returns one of: "relay_terminal", "mcu_peripheral", "power_input",
    "analog_input", "general".
    """
    from kicad_pipeline.pcb.constraints import _is_power_net

    # Check if connector is in a subcircuit
    for sc in subcircuits:
        if ref in sc.refs:
            if sc.circuit_type == SubCircuitType.RELAY_DRIVER:
                return "relay_terminal"
            if sc.circuit_type == SubCircuitType.MCU_PERIPHERAL_CLUSTER:
                return "mcu_peripheral"
            if sc.circuit_type == SubCircuitType.ADC_CHANNEL:
                return "analog_input"

    # Check signal adjacency
    neighbours = adj.get(ref, set())
    relay_refs = {sc.anchor_ref for sc in subcircuits
                  if sc.circuit_type == SubCircuitType.RELAY_DRIVER}
    mcu_refs = {sc.anchor_ref for sc in subcircuits
                if sc.circuit_type == SubCircuitType.MCU_PERIPHERAL_CLUSTER}

    # Check if connector shares nets with relays
    for nb in neighbours:
        if nb in relay_refs or any(nb in sc.refs for sc in subcircuits
                                    if sc.circuit_type == SubCircuitType.RELAY_DRIVER):
            return "relay_terminal"

    # Check if connected to MCU
    for nb in neighbours:
        if nb in mcu_refs:
            return "mcu_peripheral"

    # Check for analog nets
    conn_nets = ref_to_nets.get(ref, set())
    from kicad_pipeline.optimization.functional_grouper import _ANALOG_KEYWORDS
    if any(any(kw in n.upper() for kw in _ANALOG_KEYWORDS) for n in conn_nets):
        return "analog_input"

    # Check for power-only connector
    if all(_is_power_net(n) or n.upper() in {"GND", "VCC"}
           for n in conn_nets if n.strip()):
        return "power_input"

    return "general"


def _pin_connectors_by_function(
    subcircuits: tuple[DetectedSubCircuit, ...],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    adj: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    pcb: PCBDesign,
    group_map: dict[str, str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Pin connectors to board edges based on their functional classification.

    Instead of pushing every connector to its nearest edge, classifies
    each connector by function and pins it to the edge that makes sense:
    - relay_terminal: same edge as relay bank
    - mcu_peripheral: edge nearest MCU
    - power_input: edge nearest power zone
    - analog_input: same edge as relay terminals (measuring those circuits)
    - general: nearest edge (fallback)

    When *group_map* is provided, connector moves are constrained to stay
    near their group's centroid (edge nearest to group, not global).
    """
    min_x, min_y, max_x, max_y = bounds
    edge_margin = CONNECTOR_EDGE_MARGIN_MM

    # Compute functional group centroids for edge targeting
    relay_centroid: tuple[float, float] | None = None
    relay_positions = []
    for sc in subcircuits:
        if sc.circuit_type == SubCircuitType.RELAY_DRIVER and sc.anchor_ref in positions:
            rx, ry, _ = positions[sc.anchor_ref]
            relay_positions.append((rx, ry))
    if relay_positions:
        relay_centroid = (
            sum(p[0] for p in relay_positions) / len(relay_positions),
            sum(p[1] for p in relay_positions) / len(relay_positions),
        )

    mcu_centroid: tuple[float, float] | None = None
    for sc in subcircuits:
        if (sc.circuit_type == SubCircuitType.MCU_PERIPHERAL_CLUSTER
                and sc.anchor_ref in positions):
            mx, my, _ = positions[sc.anchor_ref]
            mcu_centroid = (mx, my)
            break

    # Find dominant edge for relay group (edge closest to relay centroid)
    def _nearest_edge(
        cx: float, cy: float,
    ) -> str:
        dists = {
            "left": cx - min_x,
            "right": max_x - cx,
            "top": cy - min_y,
            "bottom": max_y - cy,
        }
        return min(dists, key=lambda k: dists[k])

    relay_edge = _nearest_edge(*relay_centroid) if relay_centroid else "left"

    for fp in pcb.footprints:
        ref = fp.ref
        if ref in fixed_refs or not ref.startswith("J"):
            continue
        if ref not in positions:
            continue

        cx, cy, rot = positions[ref]
        w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)

        # Classify connector function
        func = _classify_connector_function(ref, subcircuits, adj, ref_to_nets)

        # Determine target edge based on function
        if func == "relay_terminal":
            target_edge = relay_edge
        elif func == "analog_input":
            target_edge = relay_edge  # analog inputs near relay terminals
        elif func == "mcu_peripheral" and mcu_centroid:
            target_edge = _nearest_edge(*mcu_centroid)
        elif func == "power_input":
            # Power connectors toward the high-voltage zone
            is_left = relay_centroid and relay_centroid[0] < (max_x + min_x) / 2
            target_edge = "left" if is_left else "right"
        else:
            # General: nearest edge to current position
            target_edge = _nearest_edge(cx, cy)

        # When group_map is provided, use group centroid Y for the
        # connector's secondary axis so it stays near its group
        group_cy: float | None = None
        group_cx: float | None = None
        if group_map and ref in group_map:
            gname = group_map[ref]
            gpositions = [
                (positions[r][0], positions[r][1])
                for r, g in group_map.items()
                if g == gname and r in positions
            ]
            if gpositions:
                group_cx = sum(p[0] for p in gpositions) / len(gpositions)
                group_cy = sum(p[1] for p in gpositions) / len(gpositions)

        # Compute target position on target edge
        target_x, target_y = cx, cy
        if target_edge == "left":
            target_x = min_x + edge_margin + w / 2.0
            if group_cy is not None:
                target_y = group_cy
        elif target_edge == "right":
            target_x = max_x - edge_margin - w / 2.0
            if group_cy is not None:
                target_y = group_cy
        elif target_edge == "top":
            target_y = min_y + edge_margin + h / 2.0
            if group_cx is not None:
                target_x = group_cx
        else:
            target_y = max_y - edge_margin - h / 2.0
            if group_cx is not None:
                target_x = group_cx

        # Only move if not already on the target edge
        current_edge_dist = min(
            cx - min_x, max_x - cx, cy - min_y, max_y - cy,
        )
        if current_edge_dist <= 5.0:
            # Already on an edge -- check if it's the right edge
            current_edge = _nearest_edge(cx, cy)
            if current_edge == target_edge:
                continue  # Already on correct edge

        # Use grid-based collision-aware placement
        edge_grid = _PlacementGrid(bounds)
        for other_ref, (ox, oy, _orot) in positions.items():
            if other_ref == ref:
                continue
            ow, oh = _rotation_aware_size(other_ref, positions, fp_sizes)
            edge_grid.place(ox, oy, ow, oh)

        rw, rh = _rotation_aware_size(ref, positions, fp_sizes)
        fx, fy = edge_grid.find_free_pos(target_x, target_y, rw, rh)

        # Only accept if closer to target edge than before
        new_edge_dist = min(
            fx - min_x, max_x - fx, fy - min_y, max_y - fy,
        )
        if new_edge_dist < current_edge_dist or _nearest_edge(fx, fy) == target_edge:
            positions[ref] = (fx, fy, rot)

    return positions


def _place_adc_channels(
    subcircuits: tuple[DetectedSubCircuit, ...],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
) -> dict[str, tuple[float, float, float]]:
    """Pull ADC channel subcircuit members near their input terminal connector.

    ADC_CHANNEL subcircuits have the connector as anchor -- pull divider
    and protection components close to it.
    """
    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.ADC_CHANNEL:
            continue
        anchor = sc.anchor_ref
        if anchor not in positions or anchor in fixed_refs:
            continue
        ax, ay, _ = positions[anchor]
        aw, ah = fp_sizes.get(anchor, DEFAULT_FP_SIZE_MM)

        for ref in sc.refs:
            if ref == anchor or ref in fixed_refs or ref not in positions:
                continue
            rx, ry, rrot = positions[ref]
            w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
            current_dist = math.sqrt((rx - ax) ** 2 + (ry - ay) ** 2)

            from kicad_pipeline.constants import ADC_CHANNEL_MAX_SPREAD_MM
            if current_dist <= ADC_CHANNEL_MAX_SPREAD_MM:
                continue

            # Target: tight to connector
            ideal_dist = (w + aw) / 2.0 + 1.0
            if current_dist < 0.01:
                continue
            dx = (ax - rx) / current_dist
            dy = (ay - ry) / current_dist
            tx = ax - dx * ideal_dist
            ty = ay - dy * ideal_dist
            tx = max(bounds[0] + BOARD_EDGE_MARGIN_MM, min(bounds[2] - BOARD_EDGE_MARGIN_MM, tx))
            ty = max(bounds[1] + BOARD_EDGE_MARGIN_MM, min(bounds[3] - BOARD_EDGE_MARGIN_MM, ty))

            move_grid = _PlacementGrid(bounds)
            for oref, (ox, oy, _orot) in positions.items():
                if oref == ref:
                    continue
                ow, oh = fp_sizes.get(oref, DEFAULT_FP_SIZE_MM)
                move_grid.place(ox, oy, ow, oh)

            fx, fy = move_grid.find_free_pos(tx, ty, w, h)
            new_dist = math.sqrt((fx - ax) ** 2 + (fy - ay) ** 2)
            if new_dist < current_dist:
                positions[ref] = (fx, fy, rrot)

    return positions


def _apply_cross_domain_affinity_overrides(
    affinities: tuple[DomainAffinity, ...],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    domain_map: dict[str, VoltageDomain],
) -> dict[str, tuple[float, float, float]]:
    """Move cross-domain monitoring components closer to their target domain.

    For "measurement" affinities (e.g. ADC monitoring 24V relay outputs),
    moves source_refs toward the centroid of target_refs so they're placed
    near the domain boundary they're monitoring.
    """
    result = dict(positions)

    for aff in affinities:
        if aff.reason != "measurement":
            continue

        # Compute centroid of target refs (the domain being measured)
        target_positions = [
            (result[r][0], result[r][1])
            for r in aff.target_refs if r in result
        ]
        if not target_positions:
            continue
        target_cx = sum(p[0] for p in target_positions) / len(target_positions)
        target_cy = sum(p[1] for p in target_positions) / len(target_positions)

        # Move source refs toward target domain boundary
        for ref in aff.source_refs:
            if ref in fixed_refs or ref not in result:
                continue
            rx, ry, rot = result[ref]
            w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)

            current_dist = math.sqrt((rx - target_cx) ** 2 + (ry - target_cy) ** 2)
            if current_dist < 10.0:
                continue  # Already reasonably close

            # Target: partway toward the target domain (60% of the way)
            tx = rx + (target_cx - rx) * 0.6
            ty = ry + (target_cy - ry) * 0.6

            # Build grid without this component
            move_grid = _PlacementGrid(bounds)
            for oref, (ox, oy, _orot) in result.items():
                if oref == ref:
                    continue
                ow, oh = fp_sizes.get(oref, DEFAULT_FP_SIZE_MM)
                move_grid.place(ox, oy, ow, oh)

            fx, fy = move_grid.find_free_pos(tx, ty, w, h)
            new_dist = math.sqrt((fx - target_cx) ** 2 + (fy - target_cy) ** 2)
            if new_dist < current_dist:
                result[ref] = (fx, fy, rot)
                _log.info(
                    "  Affinity override: moved %s %.1fmm closer to %s domain",
                    ref, current_dist - new_dist, aff.target_domain.value,
                )

    return result


def _apply_template_refinement(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    requirements: ProjectRequirements,
    subcircuits: tuple[DetectedSubCircuit, ...],
    fixed_refs: set[str],
) -> tuple[dict[str, tuple[float, float, float]], set[str]]:
    """Apply subcircuit layout templates to refine component positions.

    For each detected subcircuit with a matching template, maps detected
    component refs to template slots by role matching, computes target
    positions relative to the subcircuit anchor, and moves components
    to template positions (respecting existing fixed refs).

    Args:
        positions: Current component positions {ref: (x, y, rot)}.
        fp_sizes: Component courtyard sizes {ref: (w, h)}.
        bounds: Board bounds (min_x, min_y, max_x, max_y).
        requirements: Project requirements.
        subcircuits: Detected subcircuit patterns.
        fixed_refs: Refs that must not be moved.

    Returns:
        Tuple of (updated positions, set of refs placed by templates).
    """
    from kicad_pipeline.pcb.layout_templates import (
        ComponentRole,
        get_subcircuit_template_by_type,
    )

    template_fixed: set[str] = set()
    min_x, min_y, max_x, max_y = bounds

    for sc in subcircuits:
        tmpl = get_subcircuit_template_by_type(sc.circuit_type)
        if tmpl is None:
            continue

        # Find anchor position -- the anchor_ref is the reference point
        anchor_ref = sc.anchor_ref
        if anchor_ref not in positions:
            continue
        ax, ay, arot = positions[anchor_ref]

        # Simple role-based matching: map subcircuit refs to template slots
        # For now, use positional matching (first SERIES ref -> first SERIES slot, etc.)
        role_refs: dict[str, list[str]] = {}
        for ref in sc.refs:
            if ref == anchor_ref:
                continue
            if ref not in positions or ref in fixed_refs:
                continue
            # Classify by ref prefix
            r_upper = ref.upper()
            if r_upper.startswith("R"):
                role_refs.setdefault("series", []).append(ref)
            elif r_upper.startswith("C"):
                role_refs.setdefault("shunt", []).append(ref)
            elif r_upper.startswith("D"):
                role_refs.setdefault("shunt_d", []).append(ref)
            elif r_upper.startswith("Q"):
                role_refs.setdefault("switch", []).append(ref)
            elif r_upper.startswith("L"):
                role_refs.setdefault("series_l", []).append(ref)

        # Match slots to available refs
        series_slots = [s for s in tmpl.slots if s.role == ComponentRole.SERIES]
        shunt_slots = [s for s in tmpl.slots if s.role == ComponentRole.SHUNT]
        switch_slots = [s for s in tmpl.slots if s.role == ComponentRole.SWITCH]

        placed_count = 0

        def _try_place(slot: object, ref: str) -> bool:
            """Place ref at slot offset, skipping if it would collide."""
            nx = ax + slot.offset_x  # type: ignore[union-attr]
            ny = ay + slot.offset_y  # type: ignore[union-attr]
            w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
            nx = max(min_x + w / 2, min(max_x - w / 2, nx))
            ny = max(min_y + h / 2, min(max_y - h / 2, ny))
            # Check for collisions with existing positions
            for other_ref, (ox, oy, _orot) in positions.items():
                if other_ref == ref:
                    continue
                ow, oh = fp_sizes.get(other_ref, DEFAULT_FP_SIZE_MM)
                gap_x = abs(nx - ox) - (w + ow) / 2
                gap_y = abs(ny - oy) - (h + oh) / 2
                if gap_x < 0.2 and gap_y < 0.2:
                    return False  # Would collide -- skip
            rot = slot.rotation if slot.rotation != 0.0 else positions[ref][2]  # type: ignore[union-attr]
            positions[ref] = (nx, ny, rot)
            template_fixed.add(ref)
            return True

        # Place series components (R, L)
        series_refs = role_refs.get("series", []) + role_refs.get("series_l", [])
        for slot, ref in zip(series_slots, series_refs, strict=False):
            if _try_place(slot, ref):
                placed_count += 1

        # Place shunt components (C, D)
        shunt_refs = role_refs.get("shunt", []) + role_refs.get("shunt_d", [])
        for slot, ref in zip(shunt_slots, shunt_refs, strict=False):
            if _try_place(slot, ref):
                placed_count += 1

        # Place switch components (Q)
        switch_refs = role_refs.get("switch", [])
        for slot, ref in zip(switch_slots, switch_refs, strict=False):
            if _try_place(slot, ref):
                placed_count += 1

        if placed_count > 0:
            _log.info(
                "    Template %s: placed %d/%d refs around %s",
                tmpl.name, placed_count, len(sc.refs) - 1, anchor_ref,
            )

    return positions, template_fixed
