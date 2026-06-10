"""Bridge from KiCad footprints to v2 cell geometry.

Cells work in CENTROID space (component position = centroid of its
pads), matching the v1 optimizer convention. All conversions go through
:func:`kicad_pipeline.pcb.pin_map.compute_centroid_offset` — the single
blessed implementation of origin/centroid math.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.pcb.pin_map import compute_centroid_offset

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint, Pad
    from kicad_pipeline.placement_v2.ir import Polygon

#: Default gap added around pad extents when a footprint has no
#: explicit courtyard graphics.
_COURTYARD_FALLBACK_MARGIN_MM = 0.25


def pad_by_number(fp: Footprint, pin: str) -> Pad | None:
    """Return the pad with the given number, or ``None``."""
    for pad in fp.pads:
        if pad.number == pin:
            return pad
    return None


def pad_offset_from_centroid(fp: Footprint, pin: str) -> tuple[float, float]:
    """Pad center relative to the footprint's pad centroid (unrotated).

    Raises ``KeyError`` if the pin does not exist — a wrong pin number
    must fail loudly, not place a component at a guessed position.
    """
    pad = pad_by_number(fp, pin)
    if pad is None:
        raise KeyError(f"{fp.ref}: no pad {pin!r} on footprint {fp.lib_id!r}")
    cx, cy = compute_centroid_offset(fp)
    return (pad.position.x - cx, pad.position.y - cy)


def pad_position_in_frame(
    fp: Footprint, pin: str, member_x: float, member_y: float,
    member_rotation_deg: float,
) -> tuple[float, float]:
    """Pad center in the cell/board frame for a member placed by centroid.

    Rotation follows the KICAD convention (positive angle negated, then
    the standard CCW matrix — see ``pin_map.pad_extent_in_board_space``)
    so solver-frame geometry is identical to the written artifact. v2
    rotations ARE KiCad rotations; there is no conversion seam.
    """
    px, py = pad_offset_from_centroid(fp, pin)
    rad = math.radians(-member_rotation_deg)
    c, s = math.cos(rad), math.sin(rad)
    return (member_x + px * c - py * s, member_y + px * s + py * c)


def courtyard_polygon(fp: Footprint) -> Polygon:
    """Courtyard rectangle relative to the pad centroid (unrotated).

    Prefers explicit ``F.CrtYd``/``B.CrtYd`` graphics; falls back to the
    pad bounding box inflated by 0.25mm. THT bodies larger than their
    pads (relays, connectors) MUST carry courtyard graphics or a
    certificate body polygon — the fallback under-approximates them.
    """
    cx, cy = compute_centroid_offset(fp)
    xs: list[float] = []
    ys: list[float] = []
    for g in fp.graphics:
        layer = getattr(g, "layer", "")
        if layer.endswith(".CrtYd"):
            for attr in ("start", "end", "center"):
                p = getattr(g, attr, None)
                if p is not None:
                    xs.append(p.x)
                    ys.append(p.y)
    if xs and ys:
        x1, x2 = min(xs) - cx, max(xs) - cx
        y1, y2 = min(ys) - cy, max(ys) - cy
    else:
        margin = _COURTYARD_FALLBACK_MARGIN_MM
        px1 = min(p.position.x - p.size_x / 2 for p in fp.pads)
        px2 = max(p.position.x + p.size_x / 2 for p in fp.pads)
        py1 = min(p.position.y - p.size_y / 2 for p in fp.pads)
        py2 = max(p.position.y + p.size_y / 2 for p in fp.pads)
        x1, x2 = px1 - margin - cx, px2 + margin - cx
        y1, y2 = py1 - margin - cy, py2 + margin - cy
    return (Point(x1, y1), Point(x2, y1), Point(x2, y2), Point(x1, y2))


def courtyard_halfdims(fp: Footprint) -> tuple[float, float]:
    """Half-width and half-height of the courtyard around the centroid."""
    poly = courtyard_polygon(fp)
    xs = [p.x for p in poly]
    ys = [p.y for p in poly]
    return (
        max(abs(min(xs)), abs(max(xs))),
        max(abs(min(ys)), abs(max(ys))),
    )


def courtyard_in_frame(
    fp: Footprint, x: float, y: float, rotation_deg: float,
) -> Polygon:
    """Courtyard polygon transformed to a member's frame position.

    KiCad rotation convention (negated angle) — critical for OFF-CENTER
    courtyards (THT relays, connectors): with the mathematical
    convention the body bulge would land mirrored about the centroid in
    the written file at 90/270 degrees.
    """
    rad = math.radians(-rotation_deg)
    c, s = math.cos(rad), math.sin(rad)
    return tuple(
        Point(x + p.x * c - p.y * s, y + p.x * s + p.y * c)
        for p in courtyard_polygon(fp)
    )
