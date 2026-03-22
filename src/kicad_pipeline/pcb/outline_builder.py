"""Board outline generation for PCB designs.

Creates :class:`~kicad_pipeline.models.pcb.BoardOutline` objects with
optional corner rounding for rectangular PCB shapes.
"""

from __future__ import annotations

from kicad_pipeline.constants import PCB_EDGE_CUTS_WIDTH_MM
from kicad_pipeline.models.pcb import BoardOutline, Point


def make_board_outline(
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
