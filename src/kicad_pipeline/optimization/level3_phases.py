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


def _find_relay_zone(
    zones: Sequence[BoardZone] | None,
) -> BoardZone | None:
    """Find the relay zone from a sequence of BoardZones."""
    if not zones:
        return None
    for z in zones:
        if z.name == "relay":
            return z
    return None


def _compute_relay_row_x_range(
    min_x: float,
    max_x: float,
    relay_zone: BoardZone | None,
    max_relay_half_w: float,
) -> tuple[float, float]:
    """Compute the X range for relay row placement."""
    x1 = min_x + BOARD_EDGE_MARGIN_MM
    x2 = max_x - 15.0  # leave room for edge connectors
    if relay_zone is not None:
        x1 = max(x1, relay_zone.rect[0])
        x2 = min(x2, relay_zone.rect[2] - 5.0)
    x2 = min(x2, max_x - max_relay_half_w - 1.5)
    return x1, x2


def _compute_relay_row_y(
    avg_y: float,
    min_y: float,
    max_relay_h: float,
    start_x: float,
    total_width: float,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    row_refs: set[str],
    relay_zone: BoardZone | None,
) -> float:
    """Compute the Y position for the relay row, avoiding top-edge obstacles."""
    # Find the lowest bottom edge of any component above the relay row
    relay_x_min = start_x
    relay_x_max = start_x + total_width
    top_obstacle_y = min_y
    for ref, (ox, oy, _orot) in positions.items():
        if ref in row_refs:
            continue
        ow, oh = _rotation_aware_size(ref, positions, fp_sizes)
        if ox + ow / 2 > relay_x_min and ox - ow / 2 < relay_x_max:
            obstacle_bot = oy + oh / 2.0
            if obstacle_bot < avg_y:
                top_obstacle_y = max(top_obstacle_y, obstacle_bot)

    min_relay_y = top_obstacle_y + max_relay_h / 2.0 + 2.0

    if relay_zone is not None:
        zone_center_y = (relay_zone.rect[1] + relay_zone.rect[3]) / 2.0
        min_relay_y = max(min_relay_y, relay_zone.rect[1] + max_relay_h / 2.0 + 2.0)
        avg_y = max(avg_y, zone_center_y)

    # Fallback: enforce minimum 25mm from top edge
    min_relay_y = max(min_relay_y, min_y + max_relay_h / 2.0 + 15.0)
    return max(min_relay_y, avg_y)


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

    relay_zone = _find_relay_zone(zones)

    for _circuit_type, group in type_groups.items():
        if len(group) < 2:
            continue

        anchor_positions: list[tuple[float, float, DetectedSubCircuit]] = []
        for sc in group:
            if sc.anchor_ref in positions and sc.anchor_ref not in fixed_refs:
                x, y, _rot = positions[sc.anchor_ref]
                anchor_positions.append((x, y, sc))

        if len(anchor_positions) < 2:
            continue

        avg_y = sum(p[1] for p in anchor_positions) / len(anchor_positions)
        anchor_positions.sort(key=lambda p: p[0])
        relay_rotation = 90.0

        # Compute total row width (swapped w/h for 90 deg rotation)
        # Use actual relay widths + courtyard gap (min 1mm) for spacing
        courtyard_gap = 1.0
        relay_widths: list[float] = []
        for _, _, sc in anchor_positions:
            aw, ah = fp_sizes.get(sc.anchor_ref, DEFAULT_IC_SIZE_MM)
            relay_widths.append(ah)  # ah because rotated 90 deg
        total_width = sum(relay_widths) + courtyard_gap * (len(relay_widths) - 1)

        max_relay_half_w = max(
            (fp_sizes.get(sc.anchor_ref, DEFAULT_IC_SIZE_MM)[1] for _, _, sc in anchor_positions),
            default=8.8,
        ) / 2.0

        zone_x1, zone_x2 = _compute_relay_row_x_range(
            min_x, max_x, relay_zone, max_relay_half_w,
        )
        zone_center_x = (zone_x1 + zone_x2) / 2.0
        start_x = max(zone_x1, zone_center_x - total_width / 2.0)
        if start_x + total_width > zone_x2:
            start_x = zone_x2 - total_width

        # Collect row refs and register non-row components on grid
        row_refs: set[str] = set()
        for _, _, sc in anchor_positions:
            row_refs.update(sc.refs)
        row_grid = _PlacementGrid(bounds)
        for ref, (ox, oy, _orot) in positions.items():
            if ref not in row_refs:
                ow, oh = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
                row_grid.place(ox, oy, ow, oh)

        # relay_rotation is 90 deg, so effective height = w (swapped)
        max_relay_h = max(
            fp_sizes.get(sc.anchor_ref, (18.0, 16.0))[0]  # w becomes h at 90 deg
            for _, _, sc in anchor_positions
        )

        row_y = _compute_relay_row_y(
            avg_y, min_y, max_relay_h, start_x, total_width,
            positions, fp_sizes, row_refs, relay_zone,
        )

        # Even spacing: divide usable width evenly among relays.
        # If the zone is too narrow for all relays, fall back to full board width.
        n_relays = len(anchor_positions)
        usable_width = zone_x2 - zone_x1
        if usable_width < total_width:
            # Zone too narrow — use full board width with margins
            board_margin = max_relay_half_w + 2.0
            zone_x1 = min_x + board_margin
            zone_x2 = max_x - board_margin
            usable_width = zone_x2 - zone_x1
            start_x = zone_x1
        # Ensure spacing is at least relay_width + courtyard gap
        min_spacing = max(relay_widths) + courtyard_gap
        relay_spacing = max(min_spacing, usable_width / n_relays)
        for idx, (_, _, sc) in enumerate(anchor_positions):
            anchor_ref = sc.anchor_ref
            aw, ah = fp_sizes.get(anchor_ref, DEFAULT_IC_SIZE_MM)
            aw, ah = ah, aw  # Swap for 90 deg rotation
            # Center each relay in its slot
            target_x = zone_x1 + relay_spacing * (idx + 0.5)
            target_x = max(min_x + BOARD_EDGE_MARGIN_MM, min(max_x - 15.0, target_x))
            target_y = max(min_y + BOARD_EDGE_MARGIN_MM, min(max_y - BOARD_EDGE_MARGIN_MM, row_y))
            positions[anchor_ref] = (target_x, target_y, relay_rotation)
            row_grid.place(target_x, target_y, aw, ah)
            _log.info("    relay %s -> (%.1f, %.1f) rot=%.0f spacing=%.1f",
                       anchor_ref, target_x, target_y, relay_rotation, relay_spacing)

    return positions


def _boundary_target_from_zones(
    sc: DetectedSubCircuit,
    zone_rects: dict[VoltageDomain, tuple[float, float, float, float]],
    domain_centroids: dict[VoltageDomain, tuple[float, float]],
    min_x: float,
    min_y: float,
) -> tuple[float, float] | None:
    """Compute boundary target position from zone rects or domain centroids.

    Returns (target_x, target_y) or None if insufficient data.
    """
    assert sc.input_domain is not None and sc.output_domain is not None
    in_rect = zone_rects.get(sc.input_domain)
    out_rect = zone_rects.get(sc.output_domain)

    if in_rect and out_rect:
        ix1, iy1, ix2, iy2 = in_rect
        ox1, oy1, ox2, oy2 = out_rect
        if abs(ix2 - ox1) < 12.0:
            return ((ix2 + ox1) / 2.0 + min_x,
                    (max(iy1, oy1) + min(iy2, oy2)) / 2.0 + min_y)
        if abs(iy2 - oy1) < 12.0:
            return ((max(ix1, ox1) + min(ix2, ox2)) / 2.0 + min_x,
                    (iy2 + oy1) / 2.0 + min_y)
        # No clear shared edge -- fall through to centroid midpoint

    in_c = domain_centroids.get(sc.input_domain)
    out_c = domain_centroids.get(sc.output_domain)
    if in_c is None or out_c is None:
        return None
    return ((in_c[0] + out_c[0]) / 2.0, (in_c[1] + out_c[1]) / 2.0)


def _pull_members_toward_anchor(
    sc: DetectedSubCircuit,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    move_grid: _PlacementGrid,
    fx: float,
    fy: float,
    aw: float,
) -> None:
    """Pull subcircuit members toward the new anchor position *fx, fy*."""
    min_x, min_y, max_x, max_y = bounds
    for ref in sc.refs:
        if ref == sc.anchor_ref or ref in fixed_refs or ref not in positions:
            continue
        rx, ry, rrot = positions[ref]
        w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
        ideal_dist = (w + aw) / 2.0 + 1.0
        rdist = math.sqrt((rx - fx) ** 2 + (ry - fy) ** 2)
        if rdist <= ideal_dist + 1.0 or rdist < 0.01:
            continue
        dx = (fx - rx) / rdist
        dy = (fy - ry) / rdist
        tx = max(min_x + BOARD_EDGE_MARGIN_MM,
                 min(max_x - BOARD_EDGE_MARGIN_MM, fx - dx * ideal_dist))
        ty = max(min_y + BOARD_EDGE_MARGIN_MM,
                 min(max_y - BOARD_EDGE_MARGIN_MM, fy - dy * ideal_dist))
        mrx, mry = move_grid.find_free_pos(tx, ty, w, h)
        if math.sqrt((mrx - fx) ** 2 + (mry - fy) ** 2) < rdist:
            move_grid.place(mrx, mry, w, h)
            positions[ref] = (mrx, mry, rrot)


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

    zone_rects: dict[VoltageDomain, tuple[float, float, float, float]] = {}
    if zone_assignments:
        for za in zone_assignments:
            zone_rects[za.domain] = za.zone_rect

    domain_positions: dict[VoltageDomain, list[tuple[float, float]]] = {}
    for ref, (x, y, _rot) in positions.items():
        d = domain_map.get(ref)
        if d is not None and d != VoltageDomain.MIXED:
            domain_positions.setdefault(d, []).append((x, y))

    domain_centroids: dict[VoltageDomain, tuple[float, float]] = {}
    for d, pts in domain_positions.items():
        domain_centroids[d] = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
        )

    for sc in subcircuits:
        if sc.layout_hint != "boundary":
            continue
        if sc.input_domain is None or sc.output_domain is None:
            continue
        if sc.anchor_ref in fixed_refs or sc.anchor_ref not in positions:
            continue

        result = _boundary_target_from_zones(
            sc, zone_rects, domain_centroids, min_x, min_y,
        )
        if result is None:
            continue
        target_x = max(min_x + BOARD_EDGE_MARGIN_MM,
                       min(max_x - BOARD_EDGE_MARGIN_MM, result[0]))
        target_y = max(min_y + BOARD_EDGE_MARGIN_MM,
                       min(max_y - BOARD_EDGE_MARGIN_MM, result[1]))

        ax, ay, arot = positions[sc.anchor_ref]
        aw, ah = fp_sizes.get(sc.anchor_ref, DEFAULT_FP_SIZE_MM)
        current_dist = math.sqrt((ax - target_x) ** 2 + (ay - target_y) ** 2)

        if current_dist < 3.0:
            continue

        move_grid = _PlacementGrid(bounds)
        sc_refs_set = set(sc.refs)
        for ref, (ox, oy, _orot) in positions.items():
            if ref not in sc_refs_set:
                ow, oh = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
                move_grid.place(ox, oy, ow, oh)

        fx, fy = move_grid.find_free_pos(target_x, target_y, aw, ah)
        new_dist = math.sqrt((fx - target_x) ** 2 + (fy - target_y) ** 2)
        if new_dist < current_dist:
            positions[sc.anchor_ref] = (fx, fy, arot)
            move_grid.place(fx, fy, aw, ah)
            _pull_members_toward_anchor(
                sc, positions, fp_sizes, bounds, fixed_refs, move_grid, fx, fy, aw,
            )

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
            target_x = max(
                min_x + BOARD_EDGE_MARGIN_MM, min(max_x - BOARD_EDGE_MARGIN_MM, target_x)
            )
            target_y = max(
                min_y + BOARD_EDGE_MARGIN_MM, min(max_y - BOARD_EDGE_MARGIN_MM, target_y)
            )

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


def _nearest_edge_and_rotation(
    cx: float, cy: float,
    bounds: tuple[float, float, float, float],
    is_wide: bool,
) -> tuple[str, float]:
    """Return (edge_name, rotation) for the nearest board edge.

    Rotation values are for generic connectors (pin headers, USB, etc.).
    Screw terminals use a different convention — see _orient_connectors.
    """
    min_x, min_y, max_x, max_y = bounds
    distances = {
        "top": cy - min_y,
        "bottom": max_y - cy,
        "left": cx - min_x,
        "right": max_x - cx,
    }
    target_edge = min(distances, key=lambda k: distances[k])
    edge_rotations: dict[str, tuple[float, float]] = {
        # (wide_rot, narrow_rot) — for generic connectors
        "top": (90.0, 0.0),
        "bottom": (270.0, 0.0),
        "left": (0.0, 270.0),
        "right": (180.0, 90.0),
    }
    wide_rot, narrow_rot = edge_rotations[target_edge]
    return target_edge, wide_rot if is_wide else narrow_rot


# Rotation to use for screw terminals at each board edge so that wire
# entry faces OUTWARD (away from the board interior).
#   top edge:    rot=0   → wire entry faces up (toward min_y = top)
#   bottom edge: rot=180 → wire entry faces down (toward max_y = bottom)
#   left edge:   rot=90  → wire entry faces left (toward min_x = left)
#   right edge:  rot=270 → wire entry faces right (toward max_x = right)
_SCREW_TERMINAL_EDGE_ROTATION: dict[str, float] = {
    "top": 0.0,
    "bottom": 180.0,
    "left": 90.0,
    "right": 270.0,
}


def _shift_origin_to_edge(
    target_edge: str,
    origin_x: float, origin_y: float,
    pad_extent: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
    edge_margin: float,
) -> tuple[float, float]:
    """Shift origin so pads are flush to the target edge with margin."""
    min_x, min_y, max_x, max_y = bounds
    px0, py0, px1, py1 = pad_extent
    if target_edge == "top":
        origin_y += (min_y + edge_margin) - py0
    elif target_edge == "bottom":
        origin_y += (max_y - edge_margin) - py1
    elif target_edge == "left":
        origin_x += (min_x + edge_margin) - px0
    else:  # right
        origin_x += (max_x - edge_margin) - px1
    return origin_x, origin_y


def _clamp_origin_to_board(
    origin_x: float, origin_y: float,
    pad_extent: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Clamp origin so all pads stay within board bounds."""
    min_x, min_y, max_x, max_y = bounds
    px0, py0, px1, py1 = pad_extent
    if px0 < min_x + 1.0:
        origin_x += (min_x + 1.0) - px0
    if px1 > max_x - 1.0:
        origin_x -= px1 - (max_x - 1.0)
    if py0 < min_y + 1.0:
        origin_y += (min_y + 1.0) - py0
    if py1 > max_y - 1.0:
        origin_y -= py1 - (max_y - 1.0)
    return origin_x, origin_y


def _is_screw_terminal_fp(fp: object) -> bool:
    """Return True if *fp* is a screw/cage-clamp terminal block.

    Screw terminals (TerminalBlock, WJ*, CONN-TH with cage clamp) have their
    wire entry at local +y in the footprint coordinate system, which requires
    a different rotation convention than pin headers.
    """
    lib = (getattr(fp, "lib_id", None) or "").upper()
    val = (getattr(fp, "value", None) or "").upper()
    return any(kw in lib or kw in val for kw in (
        "TERMINALBLOCK", "TERMINAL_BLOCK", "WJ", "CONN-TH", "TB_",
        "SCREW_TERM", "SCREWTERM", "CAGE_CLAMP", "P5.00",
    ))


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

    Screw terminal rotation conventions (wire entry facing outward):
    - Top edge: rot=0    (wire entry faces up/north)
    - Bottom edge: rot=180 (wire entry faces down/south)
    - Left edge:  rot=90  (wire entry faces left/west)
    - Right edge: rot=270 (wire entry faces right/east)

    Pin headers and other generic connectors use a separate rotation table
    from _nearest_edge_and_rotation().
    """
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space

    min_x, min_y, max_x, max_y = bounds
    edge_margin = CONNECTOR_EDGE_MARGIN_MM

    for fp in pcb.footprints:
        ref = fp.ref
        if ref in fixed_refs or not ref.startswith("J") or ref not in positions:
            continue
        cx, cy, rot = positions[ref]
        w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)

        # Find nearest edge distance
        min_edge_dist = min(cx - min_x, max_x - cx, cy - min_y, max_y - cy)
        if min_edge_dist > 20.0:
            continue  # Too far from any edge -- not a board-edge connector

        is_wide = w > h * 1.5
        target_edge, new_rot = _nearest_edge_and_rotation(cx, cy, bounds, is_wide)

        # Screw terminals have wire entry at local +y — use the dedicated table.
        if _is_screw_terminal_fp(fp):
            new_rot = _SCREW_TERMINAL_EDGE_ROTATION[target_edge]

        origin_x, origin_y = centroid_to_origin(fp, cx, cy, new_rot)
        pad_ext = pad_extent_in_board_space(fp, origin_x, origin_y, new_rot)

        origin_x, origin_y = _shift_origin_to_edge(
            target_edge, origin_x, origin_y, pad_ext, bounds, edge_margin,
        )

        pad_ext = pad_extent_in_board_space(fp, origin_x, origin_y, new_rot)
        origin_x, origin_y = _clamp_origin_to_board(
            origin_x, origin_y, pad_ext, bounds,
        )

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


def _nearest_edge(
    cx: float,
    cy: float,
    bounds: tuple[float, float, float, float],
) -> str:
    """Return the name of the nearest board edge to (cx, cy)."""
    min_x, min_y, max_x, max_y = bounds
    dists = {
        "left": cx - min_x,
        "right": max_x - cx,
        "top": cy - min_y,
        "bottom": max_y - cy,
    }
    return min(dists, key=lambda k: dists[k])


def _compute_group_centroids(
    subcircuits: tuple[DetectedSubCircuit, ...],
    positions: dict[str, tuple[float, float, float]],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Compute relay and MCU centroids from subcircuit anchors.

    Returns:
        (relay_centroid, mcu_centroid)
    """
    relay_positions: list[tuple[float, float]] = []
    mcu_centroid: tuple[float, float] | None = None

    for sc in subcircuits:
        if sc.circuit_type == SubCircuitType.RELAY_DRIVER and sc.anchor_ref in positions:
            rx, ry, _ = positions[sc.anchor_ref]
            relay_positions.append((rx, ry))
        elif (sc.circuit_type == SubCircuitType.MCU_PERIPHERAL_CLUSTER
                and sc.anchor_ref in positions and mcu_centroid is None):
            mx, my, _ = positions[sc.anchor_ref]
            mcu_centroid = (mx, my)

    relay_centroid: tuple[float, float] | None = None
    if relay_positions:
        relay_centroid = (
            sum(p[0] for p in relay_positions) / len(relay_positions),
            sum(p[1] for p in relay_positions) / len(relay_positions),
        )

    return relay_centroid, mcu_centroid


def _target_edge_for_function(
    func: str,
    relay_edge: str,
    mcu_centroid: tuple[float, float] | None,
    relay_centroid: tuple[float, float] | None,
    cx: float,
    cy: float,
    bounds: tuple[float, float, float, float],
) -> str:
    """Determine target board edge for a connector based on its function."""
    min_x, _, max_x, _ = bounds
    if func in ("relay_terminal", "analog_input"):
        return relay_edge
    if func == "mcu_peripheral" and mcu_centroid:
        return _nearest_edge(*mcu_centroid, bounds)
    if func == "power_input":
        is_left = relay_centroid and relay_centroid[0] < (max_x + min_x) / 2
        return "left" if is_left else "right"
    return _nearest_edge(cx, cy, bounds)


def _compute_edge_target_position(
    target_edge: str,
    cx: float,
    cy: float,
    w: float,
    h: float,
    bounds: tuple[float, float, float, float],
    edge_margin: float,
    group_cx: float | None,
    group_cy: float | None,
) -> tuple[float, float]:
    """Compute the target (x, y) position on the given board edge."""
    min_x, min_y, max_x, max_y = bounds
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
    return target_x, target_y


def _group_centroid_for_ref(
    ref: str,
    group_map: dict[str, str] | None,
    positions: dict[str, tuple[float, float, float]],
) -> tuple[float | None, float | None]:
    """Compute the group centroid for a connector ref, if group_map is available."""
    if not group_map or ref not in group_map:
        return None, None
    gname = group_map[ref]
    gpositions = [
        (positions[r][0], positions[r][1])
        for r, g in group_map.items()
        if g == gname and r in positions
    ]
    if not gpositions:
        return None, None
    return (
        sum(p[0] for p in gpositions) / len(gpositions),
        sum(p[1] for p in gpositions) / len(gpositions),
    )


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

    relay_centroid, mcu_centroid = _compute_group_centroids(subcircuits, positions)
    relay_edge = _nearest_edge(*relay_centroid, bounds) if relay_centroid else "left"

    for fp in pcb.footprints:
        ref = fp.ref
        if ref in fixed_refs or not ref.startswith("J") or ref not in positions:
            continue

        cx, cy, rot = positions[ref]
        w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)

        func = _classify_connector_function(ref, subcircuits, adj, ref_to_nets)
        target_edge = _target_edge_for_function(
            func, relay_edge, mcu_centroid, relay_centroid, cx, cy, bounds,
        )

        group_cx, group_cy = _group_centroid_for_ref(ref, group_map, positions)

        target_x, target_y = _compute_edge_target_position(
            target_edge, cx, cy, w, h, bounds, edge_margin, group_cx, group_cy,
        )

        current_edge_dist = min(
            cx - min_x, max_x - cx, cy - min_y, max_y - cy,
        )
        if current_edge_dist <= 5.0 and _nearest_edge(cx, cy, bounds) == target_edge:
            continue

        edge_grid = _PlacementGrid(bounds)
        for other_ref, (ox, oy, _orot) in positions.items():
            if other_ref != ref:
                ow, oh = _rotation_aware_size(other_ref, positions, fp_sizes)
                edge_grid.place(ox, oy, ow, oh)

        rw, rh = _rotation_aware_size(ref, positions, fp_sizes)
        fx, fy = edge_grid.find_free_pos(target_x, target_y, rw, rh)

        new_edge_dist = min(fx - min_x, max_x - fx, fy - min_y, max_y - fy)
        if new_edge_dist < current_edge_dist or _nearest_edge(fx, fy, bounds) == target_edge:
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


_REF_PREFIX_TO_ROLE: dict[str, str] = {
    "R": "series",
    "C": "shunt",
    "D": "shunt_d",
    "Q": "switch",
    "L": "series_l",
}


def _classify_refs_by_role(
    refs: tuple[str, ...],
    anchor_ref: str,
    positions: dict[str, tuple[float, float, float]],
    fixed_refs: set[str],
) -> dict[str, list[str]]:
    """Classify subcircuit component refs by role based on ref prefix."""
    role_refs: dict[str, list[str]] = {}
    for ref in refs:
        if ref == anchor_ref or ref not in positions or ref in fixed_refs:
            continue
        r_upper = ref.upper()
        for prefix, role in _REF_PREFIX_TO_ROLE.items():
            if r_upper.startswith(prefix):
                role_refs.setdefault(role, []).append(ref)
                break
    return role_refs


def _try_template_place(
    slot: object,
    ref: str,
    ax: float,
    ay: float,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    template_fixed: set[str],
) -> bool:
    """Place ref at slot offset, skipping if it would collide."""
    min_x, min_y, max_x, max_y = bounds
    nx = ax + slot.offset_x  # type: ignore[union-attr]
    ny = ay + slot.offset_y  # type: ignore[union-attr]
    w, h = fp_sizes.get(ref, DEFAULT_FP_SIZE_MM)
    nx = max(min_x + w / 2, min(max_x - w / 2, nx))
    ny = max(min_y + h / 2, min(max_y - h / 2, ny))
    for other_ref, (ox, oy, _orot) in positions.items():
        if other_ref == ref:
            continue
        ow, oh = fp_sizes.get(other_ref, DEFAULT_FP_SIZE_MM)
        gap_x = abs(nx - ox) - (w + ow) / 2
        gap_y = abs(ny - oy) - (h + oh) / 2
        if gap_x < 0.2 and gap_y < 0.2:
            return False
    rot = slot.rotation if slot.rotation != 0.0 else positions[ref][2]  # type: ignore[union-attr]
    positions[ref] = (nx, ny, rot)
    template_fixed.add(ref)
    return True


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

    for sc in subcircuits:
        tmpl = get_subcircuit_template_by_type(sc.circuit_type)
        if tmpl is None:
            continue
        anchor_ref = sc.anchor_ref
        if anchor_ref not in positions:
            continue
        ax, ay, _arot = positions[anchor_ref]

        role_refs = _classify_refs_by_role(sc.refs, anchor_ref, positions, fixed_refs)

        # Match slots to available refs by role
        role_slot_map: list[tuple[list[object], list[str]]] = [
            (
                [s for s in tmpl.slots if s.role == ComponentRole.SERIES],
                role_refs.get("series", []) + role_refs.get("series_l", []),
            ),
            (
                [s for s in tmpl.slots if s.role == ComponentRole.SHUNT],
                role_refs.get("shunt", []) + role_refs.get("shunt_d", []),
            ),
            (
                [s for s in tmpl.slots if s.role == ComponentRole.SWITCH],
                role_refs.get("switch", []),
            ),
        ]

        placed_count = 0
        for slots, refs in role_slot_map:
            for slot, ref in zip(slots, refs, strict=False):
                if _try_template_place(
                    slot, ref, ax, ay, positions, fp_sizes, bounds, template_fixed,
                ):
                    placed_count += 1

        if placed_count > 0:
            _log.info(
                "    Template %s: placed %d/%d refs around %s",
                tmpl.name, placed_count, len(sc.refs) - 1, anchor_ref,
            )

    return positions, template_fixed
