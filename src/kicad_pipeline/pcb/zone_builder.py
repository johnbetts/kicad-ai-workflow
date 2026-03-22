"""Zone and via generation for PCB designs.

Handles GND copper pours, GND stitching vias, and RF via fences.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    LAYER_B_CU,
    LAYER_F_CU,
    ZONE_MIN_THICKNESS_MM,
)
from kicad_pipeline.models.pcb import (
    Point,
    Via,
    ZoneFill,
    ZonePolygon,
)

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import (
        BoardOutline,
        Footprint,
        Keepout,
        Track,
    )

log = logging.getLogger(__name__)


def _new_uuid() -> str:
    """Return a fresh RFC-4122 UUID string."""
    return str(uuid.uuid4())


def make_gnd_zones(
    board: BoardOutline,
    gnd_net_number: int,
    clearance_mm: float = 0.3,
    strategy: str = "both",
) -> tuple[ZonePolygon, ...]:
    """Create GND copper pours on ``F.Cu`` and/or ``B.Cu``.

    Args:
        board: The board outline; its polygon is used as the zone boundary.
        gnd_net_number: Net number of the GND net.
        clearance_mm: Zone-to-pad/track clearance in mm.
        strategy: Ground plane strategy.  ``"both"`` (default) places GND
            pours on both layers.  ``"back_only"`` places GND only on B.Cu,
            leaving F.Cu free for routing (recommended for designs with
            >20 nets or analog signals).

    Returns:
        Tuple of :class:`ZonePolygon` objects (1 or 2 zones).
    """
    back = ZonePolygon(
        net_number=gnd_net_number,
        net_name="GND",
        layer=LAYER_B_CU,
        name="GND_B",
        polygon=board.polygon,
        min_thickness=ZONE_MIN_THICKNESS_MM,
        fill=ZoneFill.SOLID,
        clearance_mm=clearance_mm,
        uuid=_new_uuid(),
    )
    if strategy == "back_only":
        log.info("build_pcb: GND plane on B.Cu only (back_only strategy)")
        return (back,)
    front = ZonePolygon(
        net_number=gnd_net_number,
        net_name="GND",
        layer=LAYER_F_CU,
        name="GND_F",
        polygon=board.polygon,
        min_thickness=ZONE_MIN_THICKNESS_MM,
        fill=ZoneFill.SOLID,
        clearance_mm=clearance_mm,
        uuid=_new_uuid(),
    )
    return (front, back)


def make_gnd_stitching_vias(
    board: BoardOutline,
    gnd_net_number: int,
    footprints: tuple[Footprint, ...],
    existing_vias: tuple[Via, ...],
    existing_tracks: tuple[Track, ...],
    spacing_mm: float = 15.0,
    keepout_zones: tuple[Keepout, ...] = (),
) -> tuple[Via, ...]:
    """Place GND stitching vias on a regular grid across the board.

    Vias are placed on a grid with *spacing_mm* pitch (default 15mm,
    midpoint of the 10-20mm spec range).  Positions are skipped if they
    fall within 2mm of any footprint bounding box or within 1mm of an
    existing via or track segment.

    Args:
        board: Board outline for dimensions.
        gnd_net_number: Net number of the GND net.
        footprints: All placed footprints.
        existing_vias: Vias already placed by the router.
        existing_tracks: Tracks already placed by the router.
        spacing_mm: Grid spacing for stitching vias.
        keepout_zones: Keepout zones to avoid.

    Returns:
        Tuple of GND stitching vias.
    """
    from kicad_pipeline.constants import (
        GND_STITCH_FP_CLEARANCE_MM,
        VIA_DIAMETER_SIGNAL_MM,
        VIA_DRILL_SIGNAL_MM,
    )

    # Compute board bounding box from outline
    xs = [p.x for p in board.polygon]
    ys = [p.y for p in board.polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Build footprint bounding boxes (with clearance)
    fp_bboxes: list[tuple[float, float, float, float]] = []  # (x1, y1, x2, y2)
    for fp in footprints:
        pad_xs = [fp.position.x]
        pad_ys = [fp.position.y]
        for pad in fp.pads:
            px, py = fp.position.x + pad.position.x, fp.position.y + pad.position.y
            pad_xs.extend([px - pad.size_x / 2, px + pad.size_x / 2])
            pad_ys.extend([py - pad.size_y / 2, py + pad.size_y / 2])
        fp_bboxes.append((
            min(pad_xs) - GND_STITCH_FP_CLEARANCE_MM,
            min(pad_ys) - GND_STITCH_FP_CLEARANCE_MM,
            max(pad_xs) + GND_STITCH_FP_CLEARANCE_MM,
            max(pad_ys) + GND_STITCH_FP_CLEARANCE_MM,
        ))

    # Collect existing via positions
    via_positions = [(v.position.x, v.position.y) for v in existing_vias]

    # Check if any GND copper exists (for proximity filtering)
    _has_gnd_copper = any(
        pad.net_number == gnd_net_number
        for fp in footprints for pad in fp.pads
    ) or any(trk.net_number == gnd_net_number for trk in existing_tracks)

    # Build grid candidates
    edge_margin = 2.0
    vias: list[Via] = []
    y = min_y + edge_margin
    while y < max_y - edge_margin:
        x = min_x + edge_margin
        while x < max_x - edge_margin:
            # Check footprint clearance
            in_fp = False
            for x1, y1, x2, y2 in fp_bboxes:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    in_fp = True
                    break
            if in_fp:
                x += spacing_mm
                continue

            # Check keepout zones
            in_keepout = False
            for ko in keepout_zones:
                ko_xs = [p.x for p in ko.polygon]
                ko_ys = [p.y for p in ko.polygon]
                if min(ko_xs) <= x <= max(ko_xs) and min(ko_ys) <= y <= max(ko_ys):
                    in_keepout = True
                    break
            if in_keepout:
                x += spacing_mm
                continue

            # Check existing via clearance (1mm)
            too_close_via = False
            for vx, vy in via_positions:
                if abs(x - vx) < 1.0 and abs(y - vy) < 1.0:
                    too_close_via = True
                    break
            if too_close_via:
                x += spacing_mm
                continue

            # Check existing track clearance (1mm)
            too_close_track = False
            for trk in existing_tracks:
                # Simple AABB check for track segment
                tx1 = min(trk.start.x, trk.end.x) - 1.0
                ty1 = min(trk.start.y, trk.end.y) - 1.0
                tx2 = max(trk.start.x, trk.end.x) + 1.0
                ty2 = max(trk.start.y, trk.end.y) + 1.0
                if tx1 <= x <= tx2 and ty1 <= y <= ty2:
                    too_close_track = True
                    break
            if too_close_track:
                x += spacing_mm
                continue

            # Only place via if there's GND copper nearby (pad or track)
            # to avoid dangling vias far from any GND connection.
            # Skip this check if there are no GND pads at all (empty board).
            if _has_gnd_copper:
                proximity_r = spacing_mm / 2.0
                has_gnd_nearby = False
                for fp in footprints:
                    for pad in fp.pads:
                        if pad.net_number == gnd_net_number:
                            px = fp.position.x + pad.position.x
                            py = fp.position.y + pad.position.y
                            if (abs(x - px) < proximity_r
                                    and abs(y - py) < proximity_r):
                                has_gnd_nearby = True
                                break
                    if has_gnd_nearby:
                        break
                if not has_gnd_nearby:
                    for trk in existing_tracks:
                        if trk.net_number == gnd_net_number:
                            mid_x = (trk.start.x + trk.end.x) / 2
                            mid_y = (trk.start.y + trk.end.y) / 2
                            if (abs(x - mid_x) < proximity_r
                                    and abs(y - mid_y) < proximity_r):
                                has_gnd_nearby = True
                                break
                if not has_gnd_nearby:
                    x += spacing_mm
                    continue

            vias.append(Via(
                position=Point(round(x, 3), round(y, 3)),
                drill=VIA_DRILL_SIGNAL_MM,
                size=VIA_DIAMETER_SIGNAL_MM,
                layers=("F.Cu", "B.Cu"),
                net_number=gnd_net_number,
            ))
            x += spacing_mm
        y += spacing_mm

    return tuple(vias)


def make_rf_via_fence(
    keepouts: tuple[Keepout, ...],
    gnd_net_num: int,
    spacing_mm: float,
    footprints: tuple[Footprint, ...] = (),
    board_width: float = 0.0,
    board_height: float = 0.0,
) -> tuple[Via, ...]:
    """Place GND stitching vias around RF keepout perimeters.

    Creates a via fence at *spacing_mm* intervals around each keepout
    that has ``no_copper=True`` and layers containing ``"F.Cu"`` -- typical
    of RF/antenna keepouts.  Vias are skipped where they overlap
    footprint bounding boxes.

    Args:
        keepouts: All board keepouts.
        gnd_net_num: GND net number.
        spacing_mm: Target via-to-via spacing along the fence.
        footprints: Footprints to avoid.
        board_width: Board width in mm (for edge clamping).
        board_height: Board height in mm (for edge clamping).

    Returns:
        Tuple of GND vias forming the fence.
    """
    import math as _m

    vias: list[Via] = []
    fence_margin = 0.5  # mm outside keepout perimeter

    for ko in keepouts:
        if not ko.no_copper or not ko.polygon:
            continue
        # Skip non-RF keepouts (mounting holes, etc.)
        if ko.tag == "mounting_hole":
            continue
        # Check if this is an RF-related keepout (on F.Cu)
        if ko.layers and "F.Cu" not in ko.layers:
            continue

        # Walk the polygon perimeter and place vias at spacing intervals
        pts = list(ko.polygon)
        if len(pts) < 3:
            continue

        # Pre-compute footprint bounding boxes for avoidance
        fp_boxes: list[tuple[float, float, float, float]] = []
        for fp in footprints:
            pad_xs = (
                [fp.position.x + p.position.x for p in fp.pads]
                if fp.pads else [fp.position.x]
            )
            pad_ys = (
                [fp.position.y + p.position.y for p in fp.pads]
                if fp.pads else [fp.position.y]
            )
            half_sx = [p.size_x / 2.0 for p in fp.pads] if fp.pads else [0.0]
            half_sy = [p.size_y / 2.0 for p in fp.pads] if fp.pads else [0.0]
            min_x = min(px - hs for px, hs in zip(pad_xs, half_sx, strict=False)) - 0.5
            max_x = max(px + hs for px, hs in zip(pad_xs, half_sx, strict=False)) + 0.5
            min_y = min(py - hs for py, hs in zip(pad_ys, half_sy, strict=False)) - 0.5
            max_y = max(py + hs for py, hs in zip(pad_ys, half_sy, strict=False)) + 0.5
            fp_boxes.append((min_x, min_y, max_x, max_y))

        # Compute centroid for outward offset direction
        cx = sum(p.x for p in pts) / len(pts)
        cy = sum(p.y for p in pts) / len(pts)

        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            edge_len = _m.hypot(p2.x - p1.x, p2.y - p1.y)
            if edge_len < 0.01:
                continue

            # Normal direction (outward from centroid)
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            nx = -dy / edge_len
            ny = dx / edge_len
            # Ensure normal points away from centroid
            mid_x = (p1.x + p2.x) / 2.0
            mid_y = (p1.y + p2.y) / 2.0
            if nx * (mid_x - cx) + ny * (mid_y - cy) < 0:
                nx, ny = -nx, -ny

            n_vias = max(1, int(edge_len / spacing_mm))
            for j in range(n_vias):
                t = (j + 0.5) / n_vias
                vx = round(p1.x + t * dx + nx * fence_margin, 3)
                vy = round(p1.y + t * dy + ny * fence_margin, 3)

                # Skip if outside board edge (0.4mm margin for edge clearance)
                edge_margin_mm = 0.4
                if (board_width > 0 and board_height > 0
                        and (vx < edge_margin_mm or vx > board_width - edge_margin_mm
                             or vy < edge_margin_mm
                             or vy > board_height - edge_margin_mm)):
                    continue

                # Skip if inside any footprint
                blocked = False
                for bx0, by0, bx1, by1 in fp_boxes:
                    if bx0 <= vx <= bx1 and by0 <= vy <= by1:
                        blocked = True
                        break
                if blocked:
                    continue

                vias.append(Via(
                    position=Point(vx, vy),
                    drill=0.6,
                    size=1.0,
                    layers=(LAYER_F_CU, LAYER_B_CU),
                    net_number=gnd_net_num,
                    uuid=_new_uuid(),
                ))

    return tuple(vias)
