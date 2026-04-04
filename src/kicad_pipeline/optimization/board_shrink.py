"""Board shrink-to-fit — reduce board outline to minimum content bounds.

After placement is finalized, the board outline can be shrunk from the
initial dimensions to the minimum bounding box of all placed components
plus a configurable margin. This decouples placement quality from
board sizing — start oversized, place freely, then shrink.

Usage::

    from kicad_pipeline.optimization.board_shrink import shrink_board_to_content
    new_pcb = shrink_board_to_content(pcb, margin_mm=3.0)
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign

_log = logging.getLogger(__name__)

_DEFAULT_MARGIN_MM = 3.0


def compute_content_bounds(
    pcb: PCBDesign,
    margin_mm: float = _DEFAULT_MARGIN_MM,
) -> tuple[float, float, float, float]:
    """Compute minimum bounding box of all placed components + margin."""
    if not pcb.footprints:
        return (0.0, 0.0, 100.0, 100.0)

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for fp in pcb.footprints:
        x, y = fp.position.x, fp.position.y
        pad_extent = 5.0
        if fp.pads:
            xs = [abs(p.position.x) for p in fp.pads]
            ys = [abs(p.position.y) for p in fp.pads]
            if xs and ys:
                pad_extent = max(max(xs), max(ys)) + 2.0
        min_x = min(min_x, x - pad_extent)
        min_y = min(min_y, y - pad_extent)
        max_x = max(max_x, x + pad_extent)
        max_y = max(max_y, y + pad_extent)

    min_x = round((min_x - margin_mm) * 2) / 2.0
    min_y = round((min_y - margin_mm) * 2) / 2.0
    max_x = round((max_x + margin_mm) * 2) / 2.0
    max_y = round((max_y + margin_mm) * 2) / 2.0

    return (min_x, min_y, max_x, max_y)


def shrink_board_to_content(
    pcb: PCBDesign,
    margin_mm: float = _DEFAULT_MARGIN_MM,
    min_width_mm: float | None = None,
    min_height_mm: float | None = None,
) -> PCBDesign:
    """Shrink board outline to fit placed components.

    Args:
        pcb: PCB with placed components.
        margin_mm: Margin from outermost component to board edge.
        min_width_mm: Minimum board width (None = no minimum).
        min_height_mm: Minimum board height (None = no minimum).

    Returns:
        New PCBDesign with shrunk board outline.
    """
    from kicad_pipeline.models.pcb import BoardOutline, Point

    x1, y1, x2, y2 = compute_content_bounds(pcb, margin_mm)
    width = x2 - x1
    height = y2 - y1

    if min_width_mm and width < min_width_mm:
        cx = (x1 + x2) / 2.0
        x1 = cx - min_width_mm / 2.0
        x2 = cx + min_width_mm / 2.0
        width = min_width_mm
    if min_height_mm and height < min_height_mm:
        cy = (y1 + y2) / 2.0
        y1 = cy - min_height_mm / 2.0
        y2 = cy + min_height_mm / 2.0
        height = min_height_mm

    new_outline = BoardOutline(
        polygon=(
            Point(x=x1, y=y1), Point(x=x2, y=y1),
            Point(x=x2, y=y2), Point(x=x1, y=y2),
        ),
    )

    old_w = old_h = 0.0
    if pcb.outline.polygon and len(pcb.outline.polygon) >= 2:
        oxs = [p.x for p in pcb.outline.polygon]
        oys = [p.y for p in pcb.outline.polygon]
        old_w = max(oxs) - min(oxs)
        old_h = max(oys) - min(oys)

    _log.info(
        "Board shrink: %.0fx%.0f -> %.0fx%.0f mm (margin=%.1f)",
        old_w, old_h, width, height, margin_mm,
    )

    return replace(pcb, outline=new_outline)
