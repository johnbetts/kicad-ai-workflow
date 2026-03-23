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


def _build_fp_bboxes(
    footprints: tuple[Footprint, ...],
    clearance: float,
) -> list[tuple[float, float, float, float]]:
    """Build padded bounding boxes for footprint clearance checks."""
    bboxes: list[tuple[float, float, float, float]] = []
    for fp in footprints:
        pad_xs = [fp.position.x]
        pad_ys = [fp.position.y]
        for pad in fp.pads:
            px, py = fp.position.x + pad.position.x, fp.position.y + pad.position.y
            pad_xs.extend([px - pad.size_x / 2, px + pad.size_x / 2])
            pad_ys.extend([py - pad.size_y / 2, py + pad.size_y / 2])
        bboxes.append((
            min(pad_xs) - clearance, min(pad_ys) - clearance,
            max(pad_xs) + clearance, max(pad_ys) + clearance,
        ))
    return bboxes


def _point_in_any_bbox(
    x: float,
    y: float,
    bboxes: list[tuple[float, float, float, float]],
) -> bool:
    """Return True if (x, y) is inside any bounding box."""
    for x1, y1, x2, y2 in bboxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def _point_in_any_keepout(
    x: float,
    y: float,
    keepout_zones: tuple[Keepout, ...],
) -> bool:
    """Return True if (x, y) is inside any keepout polygon AABB."""
    for ko in keepout_zones:
        ko_xs = [p.x for p in ko.polygon]
        ko_ys = [p.y for p in ko.polygon]
        if min(ko_xs) <= x <= max(ko_xs) and min(ko_ys) <= y <= max(ko_ys):
            return True
    return False


def _too_close_to_via(
    x: float,
    y: float,
    via_positions: list[tuple[float, float]],
    min_dist: float = 1.0,
) -> bool:
    """Return True if (x, y) is within *min_dist* of an existing via."""
    for vx, vy in via_positions:
        if abs(x - vx) < min_dist and abs(y - vy) < min_dist:
            return True
    return False


def _too_close_to_track(
    x: float,
    y: float,
    tracks: tuple[Track, ...],
    min_dist: float = 1.0,
) -> bool:
    """Return True if (x, y) is within *min_dist* of a track segment AABB."""
    for trk in tracks:
        tx1 = min(trk.start.x, trk.end.x) - min_dist
        ty1 = min(trk.start.y, trk.end.y) - min_dist
        tx2 = max(trk.start.x, trk.end.x) + min_dist
        ty2 = max(trk.start.y, trk.end.y) + min_dist
        if tx1 <= x <= tx2 and ty1 <= y <= ty2:
            return True
    return False


def _has_gnd_nearby(
    x: float,
    y: float,
    footprints: tuple[Footprint, ...],
    tracks: tuple[Track, ...],
    gnd_net_number: int,
    proximity_r: float,
) -> bool:
    """Return True if GND copper exists within proximity radius."""
    for fp in footprints:
        for pad in fp.pads:
            if pad.net_number == gnd_net_number:
                px = fp.position.x + pad.position.x
                py = fp.position.y + pad.position.y
                if abs(x - px) < proximity_r and abs(y - py) < proximity_r:
                    return True
    for trk in tracks:
        if trk.net_number == gnd_net_number:
            mid_x = (trk.start.x + trk.end.x) / 2
            mid_y = (trk.start.y + trk.end.y) / 2
            if abs(x - mid_x) < proximity_r and abs(y - mid_y) < proximity_r:
                return True
    return False


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

    xs = [p.x for p in board.polygon]
    ys = [p.y for p in board.polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    fp_bboxes = _build_fp_bboxes(footprints, GND_STITCH_FP_CLEARANCE_MM)
    via_positions = [(v.position.x, v.position.y) for v in existing_vias]

    has_gnd_copper = any(
        pad.net_number == gnd_net_number
        for fp in footprints for pad in fp.pads
    ) or any(trk.net_number == gnd_net_number for trk in existing_tracks)

    edge_margin = 2.0
    vias: list[Via] = []
    y = min_y + edge_margin
    while y < max_y - edge_margin:
        x = min_x + edge_margin
        while x < max_x - edge_margin:
            if (_point_in_any_bbox(x, y, fp_bboxes)
                    or _point_in_any_keepout(x, y, keepout_zones)
                    or _too_close_to_via(x, y, via_positions)
                    or _too_close_to_track(x, y, existing_tracks)):
                x += spacing_mm
                continue

            if has_gnd_copper and not _has_gnd_nearby(
                x, y, footprints, existing_tracks,
                gnd_net_number, spacing_mm / 2.0,
            ):
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


def _is_rf_keepout(ko: Keepout) -> bool:
    """Return True if keepout is an RF/antenna keepout on F.Cu."""
    if not ko.no_copper or not ko.polygon:
        return False
    if ko.tag == "mounting_hole":
        return False
    if ko.layers and "F.Cu" not in ko.layers:
        return False
    return len(ko.polygon) >= 3


def _build_rf_fp_boxes(
    footprints: tuple[Footprint, ...],
) -> list[tuple[float, float, float, float]]:
    """Build footprint bounding boxes with 0.5mm margin for RF via avoidance."""
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
    return fp_boxes


def _compute_edge_fence_vias(
    p1: Point,
    p2: Point,
    cx: float,
    cy: float,
    fence_margin: float,
    spacing_mm: float,
    fp_boxes: list[tuple[float, float, float, float]],
    board_width: float,
    board_height: float,
    gnd_net_num: int,
) -> list[Via]:
    """Place vias along one polygon edge, offset outward from centroid."""
    import math as _m

    edge_len = _m.hypot(p2.x - p1.x, p2.y - p1.y)
    if edge_len < 0.01:
        return []

    dx = p2.x - p1.x
    dy = p2.y - p1.y
    nx = -dy / edge_len
    ny = dx / edge_len
    mid_x = (p1.x + p2.x) / 2.0
    mid_y = (p1.y + p2.y) / 2.0
    if nx * (mid_x - cx) + ny * (mid_y - cy) < 0:
        nx, ny = -nx, -ny

    edge_margin_mm = 0.4
    vias: list[Via] = []
    n_vias = max(1, int(edge_len / spacing_mm))
    for j in range(n_vias):
        t = (j + 0.5) / n_vias
        vx = round(p1.x + t * dx + nx * fence_margin, 3)
        vy = round(p1.y + t * dy + ny * fence_margin, 3)

        if (board_width > 0 and board_height > 0
                and (vx < edge_margin_mm or vx > board_width - edge_margin_mm
                     or vy < edge_margin_mm or vy > board_height - edge_margin_mm)):
            continue

        if _point_in_any_bbox(vx, vy, fp_boxes):
            continue

        vias.append(Via(
            position=Point(vx, vy),
            drill=0.6,
            size=1.0,
            layers=(LAYER_F_CU, LAYER_B_CU),
            net_number=gnd_net_num,
            uuid=_new_uuid(),
        ))
    return vias


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
    vias: list[Via] = []
    fence_margin = 0.5

    for ko in keepouts:
        if not _is_rf_keepout(ko):
            continue

        pts = list(ko.polygon)
        fp_boxes = _build_rf_fp_boxes(footprints)

        cx = sum(p.x for p in pts) / len(pts)
        cy = sum(p.y for p in pts) / len(pts)

        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            vias.extend(_compute_edge_fence_vias(
                p1, p2, cx, cy, fence_margin, spacing_mm,
                fp_boxes, board_width, board_height, gnd_net_num,
            ))

    return tuple(vias)
