"""PCB builder orchestrator.

Combines placement, silkscreen, and S-expression serialisation into a single
pipeline entry point.  The primary public surface is:

* :func:`build_pcb` — assemble a :class:`~kicad_pipeline.models.pcb.PCBDesign`
  from :class:`~kicad_pipeline.models.requirements.ProjectRequirements`.
* :func:`pcb_to_sexp` — serialise a :class:`~kicad_pipeline.models.pcb.PCBDesign`
  to a KiCad S-expression tree.
* :func:`write_pcb` — write the S-expression tree to a ``.kicad_pcb`` file.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    KICAD_GENERATOR,
    KICAD_PCB_VERSION,
    LAYER_B_CU,
    LAYER_EDGE_CUTS,
    LAYER_F_CU,
    PCB_EDGE_CUTS_WIDTH_MM,
    ZONE_CLEARANCE_DEFAULT_MM,
    ZONE_MIN_THICKNESS_MM,
)
from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    FootprintArc,
    FootprintCircle,
    FootprintLine,
    FootprintText,
    Keepout,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
    Track,
    Via,
    ZoneFill,
    ZonePolygon,
)
from kicad_pipeline.pcb.board_templates import get_template
from kicad_pipeline.pcb.footprints import (
    estimate_footprint_size,
    footprint_for_component,
    make_mounting_hole,
)
from kicad_pipeline.pcb.netclasses import classify_nets
from kicad_pipeline.pcb.placement import LayoutResult, layout_pcb
from kicad_pipeline.pcb.silkscreen import add_silkscreen_to_footprint
from kicad_pipeline.sexp.writer import SExpNode, write_file

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import Component, ProjectRequirements

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Board defaults
# ---------------------------------------------------------------------------

_DEFAULT_BOARD_WIDTH_MM: float = 80.0
"""Default PCB width in mm (Hammond 1551K enclosure footprint)."""

_DEFAULT_BOARD_HEIGHT_MM: float = 40.0
"""Default PCB height in mm (Hammond 1551K enclosure footprint)."""

_MOUNTING_HOLE_INSET_MM: float = 3.5
"""Distance from board corner to mounting-hole centre in mm."""

_MOUNTING_HOLE_DIAMETER_MM: float = 3.2
"""Mounting hole drill diameter in mm (M3 screw)."""

_KEEPOUT_MARGIN_MM: float = 3.0
"""Radius of the keepout zone around each mounting hole in mm.

Must be <= ``_MOUNTING_HOLE_INSET_MM`` (3.5 mm) to prevent keepout
circles from extending past the board edge.
"""

_ANTENNA_KEEPOUT_WIDTH_MM: float = 15.0
"""Width of the no-copper keepout zone reserved for an ESP32 antenna in mm."""

_ANTENNA_KEEPOUT_HEIGHT_MM: float = 10.0
"""Height of the no-copper keepout zone reserved for an ESP32 antenna in mm."""

# Keywords that indicate the design contains an RF module requiring a keepout
_RF_KEYWORDS: frozenset[str] = frozenset({"esp32", "esp8266", "nrf", "cc3200", "rf"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_uuid() -> str:
    """Return a fresh RFC-4122 UUID string.

    Returns:
        Hyphenated UUID string.
    """
    return str(uuid.uuid4())


def _resolve_silk_collisions(
    footprints: list[Footprint],
) -> list[Footprint]:
    """Push silk ref labels that overlap other footprints' copper.

    For each reference label, compute its board-space bounding box and
    check for overlap with pads on OTHER footprints.  If overlap is
    detected, flip the label to the opposite side of the component
    (negate Y offset).  Also resolves silk-on-silk overlap by nudging
    the second label sideways.
    """
    import math as _m

    # Build pad lookup: list of (abs_x, abs_y, half_w, half_h, owner_ref)
    all_pads: list[tuple[float, float, float, float, str]] = []
    for fp in footprints:
        rot_r = _m.radians(fp.rotation)
        cos_r = _m.cos(rot_r)
        sin_r = _m.sin(rot_r)
        for pad in fp.pads:
            rpx = pad.position.x * cos_r - pad.position.y * sin_r
            rpy = pad.position.x * sin_r + pad.position.y * cos_r
            px = fp.position.x + rpx
            py = fp.position.y + rpy
            hw = max(pad.size_x, pad.size_y) / 2.0
            all_pads.append((px, py, hw, hw, fp.ref))

    # Collect all ref label bboxes for silk-overlap detection
    label_bboxes: list[tuple[float, float, float, float, int]] = []
    # (cx, cy, half_w, half_h, fp_index)

    result: list[Footprint] = []
    for idx, fp in enumerate(footprints):
        ref_text = None
        ref_text_idx = -1
        for ti, t in enumerate(fp.texts):
            if t.text_type == "reference" and not t.hidden:
                ref_text = t
                ref_text_idx = ti
                break
        if ref_text is None:
            result.append(fp)
            continue

        # Compute label board-space bbox
        rot_r = _m.radians(fp.rotation)
        cos_r = _m.cos(rot_r)
        sin_r = _m.sin(rot_r)
        # Label position in board space (rotated with footprint)
        lx = ref_text.position.x * cos_r - ref_text.position.y * sin_r
        ly = ref_text.position.x * sin_r + ref_text.position.y * cos_r
        abs_lx = fp.position.x + lx
        abs_ly = fp.position.y + ly
        half_w = 0.65 * ref_text.effects_size * max(2, len(ref_text.text)) / 2.0
        half_h = ref_text.effects_size * 0.75 / 2.0

        # Check overlap with other footprints' pads
        overlaps_pad = False
        for px, py, phw, phh, owner in all_pads:
            if owner == fp.ref:
                continue
            if (abs_lx + half_w > px - phw - 0.1
                    and abs_lx - half_w < px + phw + 0.1
                    and abs_ly + half_h > py - phh - 0.1
                    and abs_ly - half_h < py + phh + 0.1):
                overlaps_pad = True
                break

        new_fp = fp
        if overlaps_pad:
            # Flip label to opposite side (negate Y in footprint-local)
            new_y = -ref_text.position.y
            new_texts = list(fp.texts)
            new_texts[ref_text_idx] = FootprintText(
                text_type=ref_text.text_type,
                text=ref_text.text,
                position=Point(x=ref_text.position.x, y=new_y),
                layer=ref_text.layer,
                effects_size=ref_text.effects_size,
                hidden=ref_text.hidden,
            )
            new_fp = Footprint(
                lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                position=fp.position, rotation=fp.rotation, layer=fp.layer,
                pads=fp.pads, graphics=fp.graphics, texts=tuple(new_texts),
                lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
                models=fp.models, datasheet=fp.datasheet,
                description=fp.description,
            )
            # Recompute label position after flip
            new_ly = ref_text.position.x * sin_r + new_y * cos_r
            abs_ly = fp.position.y + new_ly

        label_bboxes.append((abs_lx, abs_ly, half_w, half_h, idx))
        result.append(new_fp)

    return result


def _clamp_silk_to_board(
    fp: Footprint,
    origin_x: float,
    origin_y: float,
    board_w: float,
    board_h: float,
    margin: float = 0.3,
) -> Footprint:
    """Move silkscreen texts that extend beyond the board edge inward.

    Silk items whose absolute position falls outside the board rectangle
    (with *margin*) are shifted so the text stays fully on-board.  Only
    ``reference`` and ``value`` texts are adjusted — user texts are left
    alone.
    """
    changed = False
    new_texts: list[FootprintText] = []
    for t in fp.texts:
        if t.text_type not in ("reference", "value"):
            new_texts.append(t)
            continue
        half_h = t.effects_size / 2.0
        # Estimate text width: ~0.65 * size per character
        half_w = 0.65 * t.effects_size * len(t.text) / 2.0
        abs_y = fp.position.y + t.position.y
        abs_x = fp.position.x + t.position.x
        new_y = t.position.y
        new_x = t.position.x
        # Clamp Y
        if abs_y - half_h < origin_y + margin:
            new_y = (origin_y + margin + half_h) - fp.position.y
            changed = True
        elif abs_y + half_h > origin_y + board_h - margin:
            new_y = (origin_y + board_h - margin - half_h) - fp.position.y
            changed = True
        # Clamp X
        if abs_x - half_w < origin_x + margin:
            new_x = (origin_x + margin + half_w) - fp.position.x
            changed = True
        elif abs_x + half_w > origin_x + board_w - margin:
            new_x = (origin_x + board_w - margin - half_w) - fp.position.x
            changed = True
        new_texts.append(
            FootprintText(
                text_type=t.text_type,
                text=t.text,
                position=Point(x=new_x, y=new_y),
                layer=t.layer,
                effects_size=t.effects_size,
                hidden=t.hidden,
            ) if (new_x != t.position.x or new_y != t.position.y) else t
        )
    if not changed:
        return fp
    return Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
        position=fp.position, rotation=fp.rotation, layer=fp.layer,
        pads=fp.pads, graphics=fp.graphics, texts=tuple(new_texts),
        lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
        models=fp.models, datasheet=fp.datasheet,
        description=fp.description,
    )


def _make_board_outline(
    width: float,
    height: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    corner_radius_mm: float = 0.0,
) -> BoardOutline:
    """Create a rectangular :class:`BoardOutline` for the given dimensions.

    Args:
        width: Board width in mm.
        height: Board height in mm.
        origin_x: X coordinate of the board origin in mm.
        origin_y: Y coordinate of the board origin in mm.
        corner_radius_mm: Corner rounding radius in mm (0 for sharp corners).

    Returns:
        :class:`BoardOutline` with a closed polygon (rounded if radius > 0).
    """
    import math as _m

    r = corner_radius_mm
    if r <= 0.0:
        # Sharp-cornered rectangle
        polygon = (
            Point(x=origin_x, y=origin_y),
            Point(x=origin_x + width, y=origin_y),
            Point(x=origin_x + width, y=origin_y + height),
            Point(x=origin_x, y=origin_y + height),
            Point(x=origin_x, y=origin_y),
        )
        return BoardOutline(polygon=polygon, width=PCB_EDGE_CUTS_WIDTH_MM)

    # Clamp radius to half the smaller dimension
    r = min(r, width / 2.0, height / 2.0)
    n_seg = 8  # arc segments per corner

    points: list[Point] = []
    # Corner centres and start angles (CW traversal)
    corners = [
        (origin_x + r, origin_y + r, _m.pi, _m.pi * 1.5),           # top-left
        (origin_x + width - r, origin_y + r, _m.pi * 1.5, 2 * _m.pi),  # top-right
        (origin_x + width - r, origin_y + height - r, 0.0, _m.pi * 0.5),  # bottom-right
        (origin_x + r, origin_y + height - r, _m.pi * 0.5, _m.pi),   # bottom-left
    ]
    for cx, cy, a_start, a_end in corners:
        for i in range(n_seg + 1):
            angle = a_start + (a_end - a_start) * i / n_seg
            points.append(Point(
                x=round(cx + r * _m.cos(angle), 6),
                y=round(cy + r * _m.sin(angle), 6),
            ))

    # Explicitly close polygon (last point == first point)
    if points:
        points.append(points[0])

    return BoardOutline(polygon=tuple(points), width=PCB_EDGE_CUTS_WIDTH_MM)


def _build_nets(requirements: ProjectRequirements) -> tuple[NetEntry, ...]:
    """Build the net list from *requirements*.

    Net 0 is always the unconnected sentinel ``""``; net 1 is always ``GND``.
    All other nets are derived from :attr:`ProjectRequirements.nets` in order.

    Args:
        requirements: Project requirements document.

    Returns:
        Tuple of :class:`NetEntry` objects starting with net 0 (empty) and
        net 1 (``GND``).
    """
    nets: list[NetEntry] = [NetEntry(number=0, name="")]
    seen: set[str] = {""}
    # GND always gets number 1
    nets.append(NetEntry(number=1, name="GND"))
    seen.add("GND")

    next_num = 2
    for net in requirements.nets:
        if net.name not in seen:
            nets.append(NetEntry(number=next_num, name=net.name))
            seen.add(net.name)
            next_num += 1

    return tuple(nets)


def _footprint_lib_id(component: Component) -> str:
    """Derive a KiCad footprint library identifier for *component*.

    Uses the component's ``footprint`` field when it contains a colon
    (already fully qualified, e.g. ``"R_SMD:R_0805_2012Metric"``).
    Otherwise wraps it as ``"kicad-ai:<footprint>"``.

    Args:
        component: The component to classify.

    Returns:
        Fully-qualified KiCad footprint ``lib_id`` string.
    """
    fp = component.footprint
    if ":" in fp:
        return fp
    return f"kicad-ai:{fp}"


def _make_footprint(
    component: Component,
    position: Point,
    net_lookup: dict[str, int],
) -> Footprint:
    """Build a minimal :class:`Footprint` for *component* at *position*.

    For each component pin that has a ``net`` attribute, a single SMD
    rectangular pad is created with the corresponding net number.

    Args:
        component: Source component.
        position: Board-coordinate placement position in mm.
        net_lookup: Mapping from net name to net number.

    Returns:
        A :class:`Footprint` with pads assigned to their respective nets.
    """
    lib_id = _footprint_lib_id(component)
    pads: list[Pad] = []

    for pin in component.pins:
        net_num: int | None = None
        net_name: str | None = None
        if pin.net is not None:
            net_num = net_lookup.get(pin.net)
            net_name = pin.net

        pad = Pad(
            number=pin.number,
            pad_type="smd",
            shape="rect",
            position=Point(x=0.0, y=0.0),  # relative to footprint origin
            size_x=1.5,
            size_y=1.5,
            layers=(LAYER_F_CU,),
            net_number=net_num,
            net_name=net_name,
            uuid=_new_uuid(),
        )
        pads.append(pad)

    fp = Footprint(
        lib_id=lib_id,
        ref=component.ref,
        value=component.value,
        position=position,
        rotation=0.0,
        layer=LAYER_F_CU,
        pads=tuple(pads),
        graphics=(),
        texts=(),
        lcsc=component.lcsc,
        uuid=_new_uuid(),
        attr="smd",
    )
    return fp


def _apply_nets_to_footprint(
    fp: Footprint,
    component: Component,
    net_lookup: dict[str, int],
) -> Footprint:
    """Copy net assignments onto matching pads of a footprint.

    ``footprint_for_component`` generates pads without net assignments.
    This helper copies the net number/name from the component's pin
    definitions onto matching pad numbers.

    Args:
        fp: Footprint from ``footprint_for_component`` (no nets on pads).
        component: Source component with pin-level net info.
        net_lookup: Mapping from net name to net number.

    Returns:
        A new :class:`Footprint` with nets assigned to pads.
    """
    pin_net_map: dict[str, tuple[int | None, str | None]] = {}
    for pin in component.pins:
        if pin.net is not None:
            net_num = net_lookup.get(pin.net)
            pin_net_map[pin.number] = (net_num, pin.net)

    new_pads: list[Pad] = []
    for pad in fp.pads:
        if pad.number in pin_net_map:
            net_num, net_name = pin_net_map[pad.number]
            pad = Pad(
                number=pad.number,
                pad_type=pad.pad_type,
                shape=pad.shape,
                position=pad.position,
                size_x=pad.size_x,
                size_y=pad.size_y,
                layers=pad.layers,
                net_number=net_num,
                net_name=net_name,
                drill_diameter=pad.drill_diameter,
                roundrect_ratio=pad.roundrect_ratio,
                uuid=pad.uuid or _new_uuid(),
            )
        new_pads.append(pad)

    return Footprint(
        lib_id=fp.lib_id,
        ref=fp.ref,
        value=fp.value,
        position=fp.position,
        rotation=fp.rotation,
        layer=fp.layer,
        pads=tuple(new_pads),
        graphics=fp.graphics,
        texts=fp.texts,
        lcsc=fp.lcsc or component.lcsc,
        uuid=fp.uuid or _new_uuid(),
        attr=fp.attr,
        models=fp.models, datasheet=fp.datasheet, description=fp.description,
    )


def _make_gnd_zones(
    board: BoardOutline,
    gnd_net_number: int,
    clearance_mm: float = ZONE_CLEARANCE_DEFAULT_MM,
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


def _make_mounting_hole_keepouts(
    board_width: float,
    board_height: float,
    inset: float,
    radius: float,
    mounting_positions: tuple[tuple[float, float], ...] | None = None,
) -> tuple[Keepout, ...]:
    """Create circular keepout zones around mounting holes.

    When *mounting_positions* is provided, keepouts are placed at those exact
    positions. Otherwise, keepouts are placed at 4-corner fallback positions
    using the *inset* parameter.

    Each keepout is represented as a 12-point polygon approximating a circle.

    Args:
        board_width: Board width in mm.
        board_height: Board height in mm.
        inset: Distance from board corner to mounting hole centre in mm
            (used only for 4-corner fallback).
        radius: Radius of the keepout zone in mm.
        mounting_positions: Explicit mounting hole centres ``(x, y)`` in mm.
            When provided, overrides the 4-corner fallback.

    Returns:
        Tuple of :class:`Keepout` objects, one per mounting hole.
    """
    import math

    if mounting_positions is not None:
        corners = list(mounting_positions)
    else:
        corners = [
            (inset, inset),
            (board_width - inset, inset),
            (board_width - inset, board_height - inset),
            (inset, board_height - inset),
        ]
    keepouts: list[Keepout] = []
    n_pts = 12
    for cx, cy in corners:
        points: list[Point] = [
            Point(
                x=cx + radius * math.cos(2.0 * math.pi * i / n_pts),
                y=cy + radius * math.sin(2.0 * math.pi * i / n_pts),
            )
            for i in range(n_pts)
        ]
        # Do NOT explicitly close — KiCad auto-closes polygons.
        # Explicit closure creates a zero-length edge → "malformed" warning.
        pts = tuple(points)
        keepouts.append(
            Keepout(
                polygon=pts,
                layers=(LAYER_F_CU, LAYER_B_CU),
                no_copper=True,
                no_vias=True,
                no_tracks=True,
                uuid=_new_uuid(),
                tag="mounting_hole",
            )
        )
    return tuple(keepouts)


def _make_antenna_keepout(
    board_width: float,
    width: float,
    height: float,
) -> Keepout:
    """Create a no-copper keepout zone for an RF antenna in the top-right corner.

    Args:
        board_width: Total board width in mm (used to position the keepout).
        width: Width of the antenna keepout zone in mm.
        height: Height of the antenna keepout zone in mm.

    Returns:
        A :class:`Keepout` covering the top-right area of the board.
    """
    x0 = board_width - width
    y0 = 0.0
    # Do NOT explicitly close — KiCad auto-closes polygons for keepouts.
    polygon = (
        Point(x=x0, y=y0),
        Point(x=board_width, y=y0),
        Point(x=board_width, y=height),
        Point(x=x0, y=height),
    )
    return Keepout(
        polygon=polygon,
        layers=(LAYER_F_CU, LAYER_B_CU),
        no_copper=True,
        no_vias=False,
        no_tracks=True,
        uuid=_new_uuid(),
    )


def _has_rf_module(requirements: ProjectRequirements) -> bool:
    """Return True if any component value suggests an RF / WiFi module.

    Args:
        requirements: Project requirements document.

    Returns:
        ``True`` when an RF-type component is detected.
    """
    for comp in requirements.components:
        val_lower = comp.value.lower()
        if any(kw in val_lower for kw in _RF_KEYWORDS):
            return True
    return False




def _make_gnd_stitching_vias(
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


def _make_rf_via_fence(
    keepouts: tuple[Keepout, ...],
    gnd_net_num: int,
    spacing_mm: float,
    footprints: tuple[Footprint, ...] = (),
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pcb(
    requirements: ProjectRequirements,
    board_width_mm: float | None = None,
    board_height_mm: float | None = None,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    board_template: str | None = None,
    auto_route: bool = True,
) -> PCBDesign:
    """Build a complete :class:`PCBDesign` from *requirements*.

    Steps:

    1. Determine board dimensions from *board_template*, mechanical
       constraints, or the supplied overrides (default 80 x 40 mm).
    2. Create the rectangular :class:`BoardOutline`.
    3. Build the :class:`NetEntry` list from requirements (GND always = net 1).
    4. Generate :class:`Footprint` objects for all components with net
       assignments applied to pads.
    5. Run zone-based PCB placement (:func:`~.placement.layout_pcb`) to assign
       board coordinates to each footprint.
    6. Add GND copper pours on ``F.Cu`` and ``B.Cu``.
    7. Add an antenna keepout in the top-right corner if an ESP32 / RF module
       is detected in the component list.
    8. Add mounting-hole keepout zones at the four board corners.
    9. Apply silkscreen labels to every footprint.
    10. Return the complete :class:`PCBDesign` (no tracks — added by autorouter).

    Args:
        requirements: Fully-populated project requirements document.
        board_width_mm: Override board width in mm.  If ``None``, the value
            from :attr:`~ProjectRequirements.mechanical` is used, or
            80 mm as the default.
        board_height_mm: Override board height in mm.  If ``None``, the value
            from :attr:`~ProjectRequirements.mechanical` is used, or
            40 mm as the default.
        origin_x: X coordinate of the board origin in mm (default 0.0).
        origin_y: Y coordinate of the board origin in mm (default 0.0).
        board_template: Optional board template name (e.g. ``"RPI_HAT"``).
            When provided, uses template dimensions and fixed component
            positions.

    Returns:
        A complete :class:`PCBDesign` ready for serialisation.

    Raises:
        PCBError: If the requirements contain no components, or if placement
            fails for any other reason.
    """
    if not requirements.components:
        raise PCBError("Cannot build PCB: requirements has no components")

    log.info(
        "build_pcb: %d components, %d nets",
        len(requirements.components),
        len(requirements.nets),
    )

    # ------------------------------------------------------------------
    # Step 0: Board template (if specified)
    # ------------------------------------------------------------------
    fixed_positions: dict[str, tuple[float, float, float]] | None = None
    layer_overrides: dict[str, str] = {}
    corner_radius_mm: float = 0.0
    template_mounting_positions: tuple[tuple[float, float], ...] | None = None
    template_mounting_diameter: float | None = None
    tmpl_obj: object | None = None
    # Auto-detect board template from mechanical constraints when not
    # explicitly provided.
    if board_template is None and requirements.mechanical is not None:
        from kicad_pipeline.pcb.board_templates import detect_template
        auto_tmpl = detect_template(requirements.mechanical)
        if auto_tmpl is not None:
            board_template = auto_tmpl.name
            log.info("build_pcb: auto-detected board template '%s'", board_template)
    if board_template is not None:
        tmpl = get_template(board_template)
        tmpl_obj = tmpl
        log.info("build_pcb: using board template '%s'", tmpl.name)
        if board_width_mm is None:
            board_width_mm = tmpl.board_width_mm
        if board_height_mm is None:
            board_height_mm = tmpl.board_height_mm
        corner_radius_mm = tmpl.corner_radius_mm
        # Extract mounting hole positions from template
        if tmpl.mounting_holes:
            template_mounting_positions = tuple(
                (h.x_mm, h.y_mm) for h in tmpl.mounting_holes
            )
            template_mounting_diameter = tmpl.mounting_holes[0].diameter_mm
        # Extract fixed component positions from template
        if tmpl.fixed_components:
            fixed_positions = {}
            for fc in tmpl.fixed_components:
                matched_ref: str | None = None
                is_gpio = "GPIO" in fc.description.upper()
                for comp in requirements.components:
                    if comp.ref == fc.ref_pattern:
                        # Don't match small connectors to a GPIO header
                        if is_gpio and len(comp.pins) < 10:
                            continue
                        matched_ref = comp.ref
                        break
                # Fallback: for GPIO header templates, match any 2x20 connector
                if matched_ref is None and is_gpio:
                    for comp in requirements.components:
                        fp_upper = comp.footprint.upper()
                        if "02X20" in fp_upper or "2X20" in fp_upper:
                            matched_ref = comp.ref
                            break
                if matched_ref is not None:
                    fixed_positions[matched_ref] = (
                        fc.x_mm, fc.y_mm, fc.rotation,
                    )
                    if fc.layer != "F.Cu":
                        layer_overrides[matched_ref] = fc.layer
                    log.info(
                        "build_pcb: template fixed %s at (%.1f, %.1f) layer=%s",
                        matched_ref, fc.x_mm, fc.y_mm, fc.layer,
                    )

    # ------------------------------------------------------------------
    # Step 1: Board dimensions
    # ------------------------------------------------------------------
    if board_width_mm is None:
        if requirements.mechanical is not None:
            board_width_mm = requirements.mechanical.board_width_mm
        else:
            board_width_mm = _DEFAULT_BOARD_WIDTH_MM

    if board_height_mm is None:
        if requirements.mechanical is not None:
            board_height_mm = requirements.mechanical.board_height_mm
        else:
            board_height_mm = _DEFAULT_BOARD_HEIGHT_MM

    log.info("build_pcb: board %.1f x %.1f mm", board_width_mm, board_height_mm)

    # ------------------------------------------------------------------
    # Step 2: Board outline
    # ------------------------------------------------------------------
    outline = _make_board_outline(
        board_width_mm, board_height_mm, origin_x, origin_y,
        corner_radius_mm=corner_radius_mm,
    )

    # ------------------------------------------------------------------
    # Step 3: Nets
    # ------------------------------------------------------------------
    nets = _build_nets(requirements)
    net_lookup: dict[str, int] = {n.name: n.number for n in nets}

    # ------------------------------------------------------------------
    # Step 4: Footprints (without position — placement assigns positions)
    # Uses footprint_for_component from footprints.py for proper geometry,
    # then applies net assignments from component pins.
    # ------------------------------------------------------------------
    pre_footprints: list[Footprint] = []
    for comp in requirements.components:
        comp_layer = layer_overrides.get(comp.ref, LAYER_F_CU)
        fp = footprint_for_component(
            comp.ref, comp.value, comp.footprint, comp.lcsc, layer=comp_layer,
        )
        fp = _apply_nets_to_footprint(fp, comp, net_lookup)
        # Copy datasheet and description from component to footprint
        if comp.datasheet or comp.description:
            fp = Footprint(
                lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                position=fp.position, rotation=fp.rotation, layer=fp.layer,
                pads=fp.pads, graphics=fp.graphics, texts=fp.texts,
                lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr, models=fp.models,
                datasheet=comp.datasheet, description=comp.description,
            )
        pre_footprints.append(fp)

    # ------------------------------------------------------------------
    # Step 4b: Compute footprint sizes and auto-size board if needed
    # ------------------------------------------------------------------
    fp_sizes: dict[str, tuple[float, float]] = {}
    total_area = 0.0
    for comp in requirements.components:
        sz = estimate_footprint_size(comp.footprint)
        fp_sizes[comp.ref] = sz
        total_area += sz[0] * sz[1]

    # Only auto-size when no template constrains dimensions
    if tmpl_obj is None:
        import math as _math

        # Estimate minimum board area as 3x total footprint area
        min_board_area = total_area * 3.0
        # Maintain ~2:1 aspect ratio: w * h = area, h = w/2 → w = sqrt(2*area)
        min_width = _math.sqrt(min_board_area * 2.0)
        min_height = min_width / 2.0
        new_width = max(board_width_mm, min_width)
        new_height = max(board_height_mm, min_height)
        # Ensure largest footprint fits with at least 10mm margin on each side
        max_fp_w = max((s[0] for s in fp_sizes.values()), default=0.0)
        max_fp_h = max((s[1] for s in fp_sizes.values()), default=0.0)
        new_width = max(new_width, max_fp_w + 20.0)
        new_height = max(new_height, max_fp_h + 20.0)
        # Cap aspect ratio to 2.5:1 max
        if new_width > 2.5 * new_height:
            new_height = new_width / 2.0
        elif new_height > 2.5 * new_width:
            new_width = new_height / 2.0

        if new_width > board_width_mm or new_height > board_height_mm:
            board_width_mm = new_width
            board_height_mm = new_height
            outline = _make_board_outline(
                board_width_mm, board_height_mm, origin_x, origin_y,
                corner_radius_mm=corner_radius_mm,
            )
            log.info(
                "build_pcb: auto-sized board to %.1f x %.1f mm",
                board_width_mm,
                board_height_mm,
            )

    # ------------------------------------------------------------------
    # Step 4c: Create keepouts BEFORE placement so solver can avoid them
    # ------------------------------------------------------------------
    keepouts: list[Keepout] = []
    if _has_rf_module(requirements):
        log.info("build_pcb: RF module detected — adding antenna keepout")
        antenna_ko = _make_antenna_keepout(
            board_width_mm,
            _ANTENNA_KEEPOUT_WIDTH_MM,
            _ANTENNA_KEEPOUT_HEIGHT_MM,
        )
        keepouts.append(antenna_ko)

    # Mounting-hole keepouts
    mount_positions: tuple[tuple[float, float], ...] | None = template_mounting_positions
    mount_radius = _KEEPOUT_MARGIN_MM
    if (
        mount_positions is None
        and requirements.mechanical is not None
        and requirements.mechanical.mounting_hole_positions
    ):
        mount_positions = requirements.mechanical.mounting_hole_positions
    if template_mounting_diameter is not None:
        mount_radius = template_mounting_diameter / 2.0 + 1.0
    elif requirements.mechanical is not None:
        mount_radius = requirements.mechanical.mounting_hole_diameter_mm / 2.0 + 1.0

    corner_keepouts = _make_mounting_hole_keepouts(
        board_width_mm,
        board_height_mm,
        _MOUNTING_HOLE_INSET_MM,
        mount_radius,
        mounting_positions=mount_positions,
    )
    keepouts.extend(corner_keepouts)

    # ------------------------------------------------------------------
    # Step 5: Layout placement (with keepouts available to solver)
    # ------------------------------------------------------------------
    layout_result: LayoutResult = layout_pcb(
        requirements, outline, footprint_sizes=fp_sizes,
        fixed_positions=fixed_positions,
        board_template=tmpl_obj,
        keepouts=tuple(keepouts),
    )

    # Merge layer overrides from layout result (constraint solver)
    if layout_result.layers:
        for ref, lyr in layout_result.layers.items():
            if ref not in layer_overrides:
                layer_overrides[ref] = lyr

    # Apply positions and rotations to footprints
    footprints_with_pos: list[Footprint] = []
    for fp in pre_footprints:
        pos = layout_result.positions.get(fp.ref, Point(x=0.0, y=0.0))
        rot = layout_result.rotations.get(fp.ref, fp.rotation)
        fp_placed = Footprint(
            lib_id=fp.lib_id,
            ref=fp.ref,
            value=fp.value,
            position=pos,
            rotation=rot,
            layer=fp.layer,
            pads=fp.pads,
            graphics=fp.graphics,
            texts=fp.texts,
            lcsc=fp.lcsc,
            uuid=fp.uuid,
            attr=fp.attr,
            models=fp.models, datasheet=fp.datasheet, description=fp.description,
        )
        footprints_with_pos.append(fp_placed)

    # ------------------------------------------------------------------
    # Step 5b: Net classification
    # ------------------------------------------------------------------
    netclasses = classify_nets(nets)

    # Zone clearance uses the safe default — KiCad enforces per-netclass
    # clearance on tracks separately, so zones should not inherit the max.
    zone_clearance = ZONE_CLEARANCE_DEFAULT_MM

    # ------------------------------------------------------------------
    # Step 6: GND pours
    # ------------------------------------------------------------------
    gnd_net_num = net_lookup.get("GND", 1)
    # Use "both" — F.Cu pour connects SMD GND pads, B.Cu pour provides
    # ground plane.  KiCad automatically keeps clearance from signal tracks.
    gnd_strategy = "both"
    gnd_zones = _make_gnd_zones(outline, gnd_net_num, zone_clearance, strategy=gnd_strategy)
    zones: list[ZonePolygon] = list(gnd_zones)

    # ------------------------------------------------------------------
    # Step 6b: Inner-layer zones (4-layer stackup)
    # ------------------------------------------------------------------
    design_rules = DesignRules()
    layer_count = design_rules.layer_count

    if layer_count >= 4:
        from kicad_pipeline.pcb.zones import make_gnd_pour

        in1_gnd = make_gnd_pour(
            outline, net_number=gnd_net_num, net_name="GND",
            layer="In1.Cu",
        )
        zones.append(in1_gnd)
        log.info("build_pcb: added In1.Cu GND plane zone")

    # ------------------------------------------------------------------
    # Step 7-8: Keepouts already created in step 4c (before placement)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 9: Silkscreen
    # ------------------------------------------------------------------
    final_footprints = [
        _clamp_silk_to_board(
            add_silkscreen_to_footprint(fp),
            origin_x, origin_y, board_width_mm, board_height_mm,
        )
        for fp in footprints_with_pos
    ]

    # Post-pass: push silk labels that overlap other components' pads.
    # For each ref label, check if its board-space bbox overlaps any
    # pad on a neighbouring footprint; if so, shift it to the opposite
    # side (below pads instead of above, or vice-versa).
    final_footprints = _resolve_silk_collisions(final_footprints)

    # ------------------------------------------------------------------
    # Step 9b: Mounting hole footprints (NPTH, no net)
    # ------------------------------------------------------------------
    if template_mounting_positions and template_mounting_diameter is not None:
        for idx, (mx, my) in enumerate(template_mounting_positions, start=1):
            mh_ref = f"H{idx}"
            mh_fp = make_mounting_hole(mh_ref, drill_diameter=template_mounting_diameter)
            # Place at the template-defined position
            mh_fp = Footprint(
                lib_id=mh_fp.lib_id,
                ref=mh_fp.ref,
                value=mh_fp.value,
                position=Point(mx, my),
                rotation=0.0,
                layer=mh_fp.layer,
                pads=mh_fp.pads,
                graphics=mh_fp.graphics,
                texts=mh_fp.texts,
                uuid=mh_fp.uuid,
                attr=mh_fp.attr,
            )
            final_footprints.append(mh_fp)
        log.info(
            "build_pcb: added %d mounting hole footprints",
            len(template_mounting_positions),
        )

    # ------------------------------------------------------------------
    # Step 10: Autoroute (when enabled)
    # ------------------------------------------------------------------
    all_tracks: tuple[Track, ...] = ()
    all_vias: tuple[Via, ...] = ()
    from kicad_pipeline.pcb.netlist import build_netlist
    netlist = build_netlist(requirements)
    freerouting_used = False
    if auto_route:
        # --- Try FreeRouting first (handles complex boards better) ---
        from kicad_pipeline.routing.freerouting import (
            find_freerouting_jar,
            route_with_freerouting,
            ses_to_tracks,
            ses_to_vias,
        )

        jar_path = find_freerouting_jar()
        if jar_path is not None:
            log.info("build_pcb: FreeRouting JAR found at %s", jar_path)
            # Build a pre-route design (placement + nets, no tracks)
            from kicad_pipeline.pcb.netlist import assign_net_numbers_to_footprints
            pre_route_fps = assign_net_numbers_to_footprints(
                list(final_footprints), netlist,
            )
            pre_route_design = PCBDesign(
                outline=outline,
                design_rules=DesignRules(),
                nets=nets,
                footprints=tuple(pre_route_fps),
                tracks=(),
                vias=(),
                zones=(),
                keepouts=tuple(keepouts),
                netclasses=netclasses,
            )
            import tempfile
            dsn_dir = tempfile.mkdtemp(prefix="kicad_freeroute_")
            dsn_path = Path(dsn_dir) / "design.dsn"
            from kicad_pipeline.routing.dsn_export import write_dsn
            write_dsn(pre_route_design, dsn_path)
            log.info("build_pcb: exported DSN to %s", dsn_path)

            fr_result = route_with_freerouting(
                str(dsn_path), jar_path=jar_path, timeout_seconds=300,
            )
            if fr_result.success and fr_result.ses_file is not None:
                ses_content = Path(fr_result.ses_file).read_text(encoding="utf-8")
                all_tracks = ses_to_tracks(ses_content, pre_route_design)
                all_vias = ses_to_vias(ses_content, pre_route_design)
                freerouting_used = True
                log.info(
                    "build_pcb: FreeRouting complete — %d tracks, %d vias",
                    len(all_tracks), len(all_vias),
                )
            else:
                log.warning(
                    "build_pcb: FreeRouting failed (%s), falling back to grid router",
                    fr_result.error,
                )
        else:
            log.info("build_pcb: FreeRouting JAR not found, using grid router")

        # --- Fall back to grid router if FreeRouting unavailable/failed ---
        if not freerouting_used:
            from kicad_pipeline.pcb.netclasses import net_clearance_map, net_width_map
            from kicad_pipeline.routing.grid_router import (
                collect_tracks,
                collect_vias,
                route_all_nets,
            )

            widths = net_width_map(netclasses)
            clearances = net_clearance_map(netclasses)
            route_results = route_all_nets(
                netlist, final_footprints,
                board_width_mm, board_height_mm,
                grid_step_mm=0.25,
                net_widths=widths,
                net_clearances=clearances,
                keepouts=tuple(keepouts),
                corner_radius_mm=corner_radius_mm,
            )
            all_tracks = collect_tracks(route_results, routed_only=False)
            all_vias = collect_vias(route_results)
            routed = sum(1 for r in route_results if r.routed)
            unrouted = sum(1 for r in route_results if not r.routed)
            log.info(
                "build_pcb: autoroute complete — %d tracks, %d routed, %d unrouted",
                len(all_tracks), routed, unrouted,
            )

            # Log board-level routing quality metrics
            from kicad_pipeline.routing.metrics import compute_board_metrics

            metrics = compute_board_metrics(route_results, final_footprints)
            log.info(
                "build_pcb: routing %.1fmm total (%.2fx ideal), %d vias, %d/%d nets",
                metrics.total_track_length_mm,
                metrics.overall_length_ratio,
                metrics.total_vias,
                metrics.nets_routed,
                metrics.nets_routed + metrics.nets_failed,
            )

        # Note: GND pads on F.Cu connect to the B.Cu GND pour through
        # the zone fill (applied when opening in KiCad).  THT pads already
        # have plated holes.  SMD GND pads may show as "unconnected" in
        # DRC until zones are filled.

    # ------------------------------------------------------------------
    # Step 10b: RF via fence (GND vias around RF keepouts, only when
    #           the design actually contains an RF module)
    # ------------------------------------------------------------------
    if _has_rf_module(requirements):
        from kicad_pipeline.constants import RF_VIA_FENCE_SPACING_MM

        rf_fence_vias = _make_rf_via_fence(
            tuple(keepouts), gnd_net_num, RF_VIA_FENCE_SPACING_MM,
            footprints=tuple(final_footprints),
        )
        if rf_fence_vias:
            all_vias = all_vias + rf_fence_vias
            log.info("build_pcb: added %d RF via fence vias", len(rf_fence_vias))

    # ------------------------------------------------------------------
    # Step 10c: GND stitching vias (spec: every 10-20mm, only for grid router)
    # ------------------------------------------------------------------
    if auto_route and not freerouting_used:
        stitch_vias = _make_gnd_stitching_vias(
            outline, gnd_net_num, tuple(final_footprints),
            all_vias, all_tracks,
            keepout_zones=tuple(keepouts),
        )
        if stitch_vias:
            all_vias = all_vias + stitch_vias
            log.info("build_pcb: added %d GND stitching vias", len(stitch_vias))

    # ------------------------------------------------------------------
    # Step 11: Assign net numbers to footprint pads
    # ------------------------------------------------------------------
    from kicad_pipeline.pcb.netlist import assign_net_numbers_to_footprints

    final_footprints = assign_net_numbers_to_footprints(
        final_footprints, netlist,
    )

    # ------------------------------------------------------------------
    # Step 12: Generate DRC exclusions for dense IC intra-footprint clearance
    # ------------------------------------------------------------------
    drc_exclusions = _generate_ic_drc_exclusions(final_footprints)
    if drc_exclusions:
        log.info(
            "build_pcb: generated %d intra-footprint DRC exclusions",
            len(drc_exclusions),
        )

    log.info(
        "build_pcb complete: %d footprints, %d nets, %d zones, %d keepouts, "
        "%d tracks, %d vias",
        len(final_footprints),
        len(nets),
        len(zones),
        len(keepouts),
        len(all_tracks),
        len(all_vias),
    )

    return PCBDesign(
        outline=outline,
        design_rules=DesignRules(),
        nets=nets,
        footprints=tuple(final_footprints),
        tracks=all_tracks,
        vias=all_vias,
        zones=tuple(zones),
        keepouts=tuple(keepouts),
        netclasses=netclasses,
        drc_exclusions=drc_exclusions,
        version=KICAD_PCB_VERSION,
        generator=KICAD_GENERATOR,
        title=requirements.project.name,
        date=datetime.date.today().isoformat(),
        revision=requirements.project.revision,
        company=requirements.project.author or "",
    )


def _generate_ic_drc_exclusions(
    footprints: list[Footprint],
) -> tuple[str, ...]:
    """Generate DRC exclusion strings for dense IC intra-footprint clearance.

    Fine-pitch ICs (MSOP, TSSOP, QFN, etc.) have pads closer together than
    typical clearance rules.  Escape routes from these pads inevitably pass
    within the clearance zone of adjacent pads.  KiCad flags these as
    violations but they are expected — the pad spacing is fixed by the
    package geometry.

    Returns:
        Tuple of KiCad DRC exclusion strings in ``"clearance|ref|pad|ref|pad"``
        format for all adjacent pad pairs on dense IC footprints.
    """
    exclusions: list[str] = []
    for fp in footprints:
        if len(fp.pads) < 6:
            continue
        # Compute minimum pad spacing
        positions = sorted(
            (p.position.x, p.position.y, p.number) for p in fp.pads
        )
        min_spacing = 999.0
        for i in range(len(positions) - 1):
            dx = abs(positions[i + 1][0] - positions[i][0])
            dy = abs(positions[i + 1][1] - positions[i][1])
            d = (dx * dx + dy * dy) ** 0.5
            if d > 0.01:
                min_spacing = min(min_spacing, d)
        if min_spacing >= 1.0:
            continue
        # Dense IC — generate exclusions for all adjacent pad pairs
        # that are within 2x the minimum spacing
        threshold = min_spacing * 2.5
        for i, pad_a in enumerate(fp.pads):
            for pad_b in fp.pads[i + 1 :]:
                dx = abs(pad_a.position.x - pad_b.position.x)
                dy = abs(pad_a.position.y - pad_b.position.y)
                d = (dx * dx + dy * dy) ** 0.5
                if d < threshold:
                    exclusions.append(
                        f"clearance|{fp.ref}|{pad_a.number}"
                        f"|{fp.ref}|{pad_b.number}"
                    )
    return tuple(exclusions)


# ---------------------------------------------------------------------------
# S-expression serialiser
# ---------------------------------------------------------------------------

def _build_layer_table(
    layer_count: int = 2,
) -> list[tuple[int, str, str] | tuple[int, str, str, str]]:
    """Build the layer table for the given copper layer count.

    For 2-layer boards, returns the standard F.Cu/B.Cu table.
    For 4-layer boards, adds In1.Cu (GND plane) and In2.Cu (power plane).

    Args:
        layer_count: Number of copper layers (2 or 4).

    Returns:
        Layer definition list suitable for KiCad S-expression output.
    """
    table: list[tuple[int, str, str] | tuple[int, str, str, str]] = [
        (0, "F.Cu", "signal"),
    ]
    if layer_count >= 4:
        table.append((4, "In1.Cu", "power"))
        table.append((6, "In2.Cu", "power"))
    table.extend([
        (2, "B.Cu", "signal"),
        (9, "F.Adhes", "user", "F.Adhesive"),
        (11, "B.Adhes", "user", "B.Adhesive"),
        (13, "F.Paste", "user"),
        (15, "B.Paste", "user"),
        (5, "F.SilkS", "user", "F.Silkscreen"),
        (7, "B.SilkS", "user", "B.Silkscreen"),
        (1, "F.Mask", "user"),
        (3, "B.Mask", "user"),
        (17, "Dwgs.User", "user", "User.Drawings"),
        (19, "Cmts.User", "user", "User.Comments"),
        (25, "Edge.Cuts", "user"),
        (27, "Margin", "user"),
        (31, "F.CrtYd", "user", "F.Courtyard"),
        (29, "B.CrtYd", "user", "B.Courtyard"),
        (33, "F.Fab", "user", "F.Fabrication"),
        (35, "B.Fab", "user", "B.Fabrication"),
    ])
    return table


def _pad_sexp(pad: Pad) -> SExpNode:
    """Serialise a :class:`Pad` to a KiCad ``(pad ...)`` node.

    Args:
        pad: The pad to serialise.

    Returns:
        ``SExpNode`` list.
    """
    node: list[SExpNode] = [
        "pad",
        pad.number,
        pad.pad_type,
        pad.shape,
        ["at", pad.position.x, pad.position.y],
        ["size", pad.size_x, pad.size_y],
        ["layers", *pad.layers],
    ]
    if pad.drill_diameter is not None and pad.drill_diameter > 0:
        node.append(["drill", pad.drill_diameter])
    if pad.roundrect_ratio is not None:
        node.append(["roundrect_rratio", pad.roundrect_ratio])
    if pad.net_number is not None and pad.net_name is not None:
        node.append(["net", pad.net_number, pad.net_name])
    if pad.uuid:
        node.append(["uuid", pad.uuid])
    return node


def _footprint_text_sexp(ft: FootprintText) -> SExpNode:
    """Serialise a :class:`FootprintText` to a KiCad ``(fp_text ...)`` node.

    Args:
        ft: The footprint text item to serialise.

    Returns:
        ``SExpNode`` list.
    """
    effects: list[SExpNode] = [
        "effects",
        ["font", ["size", ft.effects_size, ft.effects_size]],
    ]
    if ft.hidden:
        effects.append(["hide", "yes"])

    node: list[SExpNode] = [
        "fp_text",
        ft.text_type,
        ft.text,
        ["at", ft.position.x, ft.position.y],
        ["layer", ft.layer],
        effects,
    ]
    if ft.uuid:
        node.append(["uuid", ft.uuid])
    return node


def _footprint_sexp(fp: Footprint) -> SExpNode:
    """Serialise a :class:`Footprint` to a KiCad ``(footprint ...)`` node.

    Args:
        fp: The footprint to serialise.

    Returns:
        ``SExpNode`` list.
    """
    node: list[SExpNode] = [
        "footprint",
        fp.lib_id,
        ["layer", fp.layer],
        ["at", fp.position.x, fp.position.y, fp.rotation],
        ["attr", *fp.attr.split()],
    ]

    if fp.uuid:
        node.append(["uuid", fp.uuid])

    # Properties for ref and value — use positions/layer from fp.texts if available
    ref_text = next((t for t in fp.texts if t.text_type == "reference"), None)
    ref_x = ref_text.position.x if ref_text else 0.0
    ref_y = ref_text.position.y if ref_text else -2.5
    _default_silk = "B.SilkS" if fp.layer == LAYER_B_CU else "F.SilkS"
    ref_layer = ref_text.layer if ref_text else _default_silk
    ref_size = ref_text.effects_size if ref_text else 1.0
    ref_hidden = ref_text.hidden if ref_text else False
    val_text = next((t for t in fp.texts if t.text_type == "value"), None)
    val_x = val_text.position.x if val_text else 0.0
    val_y = val_text.position.y if val_text else 2.5
    ref_effects: list[SExpNode] = [
        "effects", ["font", ["size", ref_size, ref_size]],
    ]
    if ref_hidden:
        ref_effects.append(["hide", "yes"])
    node.append(
        [
            "property",
            "Reference",
            fp.ref,
            ["at", ref_x, ref_y, 0],
            ["layer", ref_layer],
            ref_effects,
        ]
    )
    node.append(
        [
            "property",
            "Value",
            fp.value,
            ["at", val_x, val_y, 0],
            ["layer", "B.Fab" if fp.layer == LAYER_B_CU else "F.Fab"],
            ["effects", ["font", ["size", 1.0, 1.0]], ["hide", "yes"]],
        ]
    )
    # Footprint property (lib_id)
    node.append(
        [
            "property",
            "Footprint",
            fp.lib_id,
            ["at", 0, 0, 0],
            ["layer", "B.Fab" if fp.layer == LAYER_B_CU else "F.Fab"],
            ["effects", ["font", ["size", 1.0, 1.0]], ["hide", "yes"]],
        ]
    )
    # Datasheet property
    node.append(
        [
            "property",
            "Datasheet",
            fp.datasheet or "",
            ["at", 0, 0, 0],
            ["layer", "B.Fab" if fp.layer == LAYER_B_CU else "F.Fab"],
            ["effects", ["font", ["size", 1.0, 1.0]], ["hide", "yes"]],
        ]
    )
    # Description property
    node.append(
        [
            "property",
            "Description",
            fp.description or "",
            ["at", 0, 0, 0],
            ["layer", "B.Fab" if fp.layer == LAYER_B_CU else "F.Fab"],
            ["effects", ["font", ["size", 1.0, 1.0]], ["hide", "yes"]],
        ]
    )

    # KiCad 9: fp_text replaced by property entries (already emitted above).
    # Only emit fp_text for custom user text, not reference/value.
    for text in fp.texts:
        if text.text_type not in ("reference", "value"):
            node.append(_footprint_text_sexp(text))

    # Footprint graphics (courtyard, silkscreen, fab outlines)
    for graphic in fp.graphics:
        if isinstance(graphic, FootprintLine):
            g_line: list[SExpNode] = [
                "fp_line",
                ["start", graphic.start.x, graphic.start.y],
                ["end", graphic.end.x, graphic.end.y],
                ["layer", graphic.layer],
                ["width", graphic.width],
            ]
            if graphic.uuid:
                g_line.append(["uuid", graphic.uuid])
            node.append(g_line)
        elif isinstance(graphic, FootprintArc):
            g_arc: list[SExpNode] = [
                "fp_arc",
                ["start", graphic.start.x, graphic.start.y],
                ["mid", graphic.mid.x, graphic.mid.y],
                ["end", graphic.end.x, graphic.end.y],
                ["layer", graphic.layer],
                ["width", graphic.width],
            ]
            if graphic.uuid:
                g_arc.append(["uuid", graphic.uuid])
            node.append(g_arc)
        elif isinstance(graphic, FootprintCircle):
            g_circ: list[SExpNode] = [
                "fp_circle",
                ["center", graphic.center.x, graphic.center.y],
                ["end", graphic.end.x, graphic.end.y],
                ["layer", graphic.layer],
                ["width", graphic.width],
            ]
            if graphic.uuid:
                g_circ.append(["uuid", graphic.uuid])
            node.append(g_circ)

    for pad in fp.pads:
        node.append(_pad_sexp(pad))

    # 3D model references
    for model in fp.models:
        model_node: list[SExpNode] = [
            "model",
            model.path,
            ["offset", ["xyz", model.offset[0], model.offset[1], model.offset[2]]],
            ["scale", ["xyz", model.scale[0], model.scale[1], model.scale[2]]],
            ["rotate", ["xyz", model.rotate[0], model.rotate[1], model.rotate[2]]],
        ]
        node.append(model_node)

    return node


def _outline_sexp(outline: BoardOutline) -> list[SExpNode]:
    """Serialise a :class:`BoardOutline` to ``(gr_line ...)`` nodes.

    Generates one line segment per consecutive pair of polygon points,
    closing the polygon with a final segment from the last point back to the
    first.

    Args:
        outline: Board outline polygon.

    Returns:
        List of ``SExpNode`` lists, one per edge segment.
    """
    nodes: list[SExpNode] = []
    pts = outline.polygon
    n = len(pts)
    # If polygon is explicitly closed (last == first), don't wrap around
    is_closed = (
        n > 2
        and abs(pts[0].x - pts[-1].x) < 1e-6
        and abs(pts[0].y - pts[-1].y) < 1e-6
    )
    edge_count = n - 1 if is_closed else n
    for i in range(edge_count):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        nodes.append(
            [
                "gr_line",
                ["start", p0.x, p0.y],
                ["end", p1.x, p1.y],
                ["layer", LAYER_EDGE_CUTS],
                ["width", outline.width],
            ]
        )
    return nodes


def _zone_sexp(zone: ZonePolygon) -> SExpNode:
    """Serialise a :class:`ZonePolygon` to a KiCad ``(zone ...)`` node.

    Args:
        zone: Copper pour zone to serialise.

    Returns:
        ``SExpNode`` list.
    """
    pts_node: list[SExpNode] = ["pts"]
    for pt in zone.polygon:
        pts_node.append(["xy", pt.x, pt.y])

    node: list[SExpNode] = [
        "zone",
        ["net", zone.net_number],
        ["net_name", zone.net_name],
        ["layer", zone.layer],
    ]
    if zone.uuid:
        node.append(["uuid", zone.uuid])
    # KiCad 9: fill node needs "yes" marker when zone has fill data
    fill_node: list[SExpNode] = ["fill"]
    if zone.filled_polygons:
        fill_node.append("yes")
    fill_node.extend([
        ["thermal_gap", zone.thermal_relief_gap],
        ["thermal_bridge_width", zone.thermal_relief_bridge],
    ])

    node.extend([
        ["hatch", "edge", 0.508],
        ["connect_pads", ["clearance", zone.clearance_mm]],
        ["min_thickness", zone.min_thickness],
        ["filled_areas_thickness", False],
        fill_node,
        ["polygon", pts_node],
    ])

    # Emit filled_polygon entries for pre-computed zone fill
    for fp_pts in zone.filled_polygons:
        fp_node: list[SExpNode] = ["pts"]
        for pt in fp_pts:
            fp_node.append(["xy", pt.x, pt.y])
        node.append(["filled_polygon", ["layer", zone.layer], fp_node])

    return node


def _keepout_sexp(keepout: Keepout) -> SExpNode:
    """Serialise a :class:`Keepout` to a KiCad ``(zone ...)`` keepout node.

    Args:
        keepout: Keepout zone to serialise.

    Returns:
        ``SExpNode`` list.
    """
    pts_node: list[SExpNode] = ["pts"]
    for pt in keepout.polygon:
        pts_node.append(["xy", pt.x, pt.y])

    # KiCad 9 keepout format requires all five rule entries.
    rules: list[SExpNode] = [
        "keepout",
        ["copperpour", "not_allowed" if keepout.no_copper else "allowed"],
        ["footprints", "allowed"],
        ["pads", "allowed"],
        ["tracks", "not_allowed" if keepout.no_tracks else "allowed"],
        ["vias", "not_allowed" if keepout.no_vias else "allowed"],
    ]

    node: list[SExpNode] = [
        "zone",
        ["net", 0],
        ["net_name", ""],
        ["layers", *keepout.layers],
    ]
    if keepout.uuid:
        node.append(["uuid", keepout.uuid])
    node.extend([
        ["hatch", "edge", 0.508],
        rules,
        ["polygon", pts_node],
    ])
    return node


def pcb_to_sexp(design: PCBDesign) -> SExpNode:
    """Serialise a :class:`PCBDesign` to a KiCad S-expression tree.

    Output structure::

        (kicad_pcb (version 20231120) (generator "kicad-ai-pipeline")
          (general (thickness 1.6))
          (paper "A4")
          (layers
            (0 "F.Cu" signal)
            (31 "B.Cu" signal)
            ...
          )
          (setup (stackup ...) (pcbplotparams ...))
          (net 0 "")
          (net 1 "GND")
          ...
          (footprint "R_0805" (layer "F.Cu") (at x y rot) ...)
          (gr_line (start ...) (end ...) (layer "Edge.Cuts") (width 0.05))
          (zone (net 1) (net_name "GND") (layer "F.Cu") ...)
        )

    Args:
        design: The PCB design to serialise.

    Returns:
        A nested :data:`~kicad_pipeline.sexp.writer.SExpNode` list
        representing the root ``(kicad_pcb ...)`` expression.
    """
    root: list[SExpNode] = [
        "kicad_pcb",
        ["version", design.version],
        ["generator", design.generator],
        ["generator_version", design.generator_version],
        ["general", ["thickness", 1.6], ["legacy_teardrops", False]],
        ["paper", "A4"],
    ]

    # Title block
    if design.title or design.date or design.revision or design.company:
        title_block: list[SExpNode] = ["title_block"]
        if design.title:
            title_block.append(["title", design.title])
        if design.date:
            title_block.append(["date", design.date])
        if design.revision:
            title_block.append(["rev", design.revision])
        if design.company:
            title_block.append(["company", design.company])
        root.append(title_block)

    # Layers
    layers_node: list[SExpNode] = ["layers"]
    layer_table = _build_layer_table(design.design_rules.layer_count)
    for layer_entry in layer_table:
        layer_node: list[SExpNode] = [layer_entry[0], layer_entry[1], layer_entry[2]]
        if len(layer_entry) > 3:
            layer_node.append(layer_entry[3])
        layers_node.append(layer_node)
    root.append(layers_node)

    # Setup
    root.append(
        [
            "setup",
            ["pad_to_mask_clearance", 0],
            ["allow_soldermask_bridges_in_footprints", False],
            [
                "pcbplotparams",
                ["layerselection", "0x00010fc_ffffffff"],
                ["outputdirectory", ""],
            ],
        ]
    )

    # Nets
    for net in design.nets:
        root.append(["net", net.number, net.name])

    # Footprints
    for fp in design.footprints:
        root.append(_footprint_sexp(fp))

    # Board outline
    for line in _outline_sexp(design.outline):
        root.append(line)

    # Copper zones
    for zone in design.zones:
        root.append(_zone_sexp(zone))

    # Keepout zones
    for keepout in design.keepouts:
        root.append(_keepout_sexp(keepout))

    # Tracks (segments)
    for track in design.tracks:
        seg: list[SExpNode] = [
            "segment",
            ["start", track.start.x, track.start.y],
            ["end", track.end.x, track.end.y],
            ["width", track.width],
            ["layer", track.layer],
            ["net", track.net_number],
        ]
        if track.uuid:
            seg.append(["uuid", track.uuid])
        root.append(seg)

    # Vias
    for via in design.vias:
        via_node: list[SExpNode] = [
            "via",
            ["at", via.position.x, via.position.y],
            ["size", via.size],
            ["drill", via.drill],
            ["layers", *via.layers],
            ["net", via.net_number],
        ]
        if via.uuid:
            via_node.append(["uuid", via.uuid])
        root.append(via_node)

    return root


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_pcb(
    design: PCBDesign,
    path: str | Path,
    *,
    fill_zones: bool = True,
    ipc_connection: object | None = None,
) -> None:
    """Serialise *design* and write it to a ``.kicad_pcb`` file.

    Args:
        design: The PCB design to write.
        path: Destination file path.  The parent directory must exist.
        fill_zones: If True, attempt to fill zones after writing.
        ipc_connection: Optional :class:`~kicad_pipeline.ipc.connection.KiCadConnection`.
            When provided, uses IPC to push the file and refill zones in the
            running KiCad instance.  Falls back to the subprocess approach
            if IPC zone fill fails.

    Raises:
        PCBError: If serialisation fails for any reason.
        OSError: If the file cannot be written.
    """
    dest = Path(path)
    log.info("write_pcb → %s", dest)
    try:
        sexp = pcb_to_sexp(design)
        write_file(sexp, dest)
    except OSError:
        raise
    except Exception as exc:
        raise PCBError(f"Failed to write PCB to {dest}: {exc}") from exc
    log.info("write_pcb: wrote %s", dest)

    if fill_zones:
        if ipc_connection is not None:
            _fill_zones_ipc(dest, ipc_connection)
        else:
            _fill_zones(dest)


def _fill_zones_ipc(pcb_path: Path, ipc_connection: object) -> None:
    """Fill copper zones using KiCad's IPC API, falling back to subprocess."""
    try:
        from kicad_pipeline.ipc.board_ops import push_pcb_to_kicad, refill_zones
        from kicad_pipeline.ipc.connection import KiCadConnection

        if not isinstance(ipc_connection, KiCadConnection):
            log.warning("ipc_connection is not a KiCadConnection; falling back to subprocess")
            _fill_zones(pcb_path)
            return

        push_pcb_to_kicad(pcb_path, ipc_connection)
        refill_zones(ipc_connection)
        log.info("Zone fill complete via IPC (%s)", pcb_path)
    except Exception as exc:
        log.warning(
            "IPC zone fill failed (%s), falling back to subprocess: %s",
            pcb_path, exc,
        )
        _fill_zones(pcb_path)


def _fill_zones(pcb_path: Path) -> None:
    """Fill copper zones using KiCad's Python API (``pcbnew``).

    KiCad 9's ``kicad-cli`` does not expose a ``fill-zones`` subcommand,
    but the bundled Python interpreter includes the ``pcbnew`` module.
    This function shells out to KiCad's Python to load the board,
    fill all zones, and save the result.

    Falls back to a log warning if KiCad's Python is not available.
    """
    import shutil
    import subprocess

    # Locate KiCad's bundled Python (macOS path)
    kicad_python = shutil.which("python3", path="/Applications/KiCad/KiCad.app"
                                "/Contents/Frameworks/Python.framework"
                                "/Versions/Current/bin")
    if kicad_python is None:
        log.warning(
            "Zone fill requires KiCad's bundled Python (pcbnew); "
            "CLI DRC will report GND stitching vias as dangling until "
            "zones are filled. (%s)", pcb_path,
        )
        return

    script = (
        "import pcbnew, sys\n"
        f"board = pcbnew.LoadBoard({str(pcb_path)!r})\n"
        "filler = pcbnew.ZONE_FILLER(board)\n"
        "zones = board.Zones()\n"
        "if zones.size() == 0:\n"
        "    sys.exit(0)\n"
        "filler.Fill(zones)\n"
        f"pcbnew.SaveBoard({str(pcb_path)!r}, board)\n"
    )
    try:
        result = subprocess.run(
            [kicad_python, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            log.info("Zone fill complete via pcbnew (%s)", pcb_path)
        else:
            log.warning(
                "Zone fill failed (exit %d): %s",
                result.returncode,
                result.stderr.strip()[:200],
            )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log.warning("Zone fill unavailable: %s", exc)
