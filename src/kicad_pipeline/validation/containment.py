"""Board outline containment checker.

Verifies that all footprint elements (centers, pads) are located within the
board outline bounding box, optionally with a margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign


@dataclass(frozen=True)
class ContainmentViolation:
    """A component element that extends beyond the board outline."""

    ref: str
    """Component reference designator."""
    element: str
    """Which element is outside: ``"center"``, ``"pad"``."""
    position_x: float
    """World X position of the violating element."""
    position_y: float
    """World Y position of the violating element."""
    overshoot_mm: float
    """How far outside the board bounds (mm)."""
    message: str


def _board_aabb(pcb: PCBDesign) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) from the board outline polygon."""
    pts = pcb.outline.polygon
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _rotate_point(px: float, py: float, degrees: float) -> tuple[float, float]:
    """Rotate *px, py* clockwise by *degrees* (KiCad convention)."""
    rad = math.radians(-degrees)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    return (px * cos_r - py * sin_r, px * sin_r + py * cos_r)


def _overshoot(
    x: float,
    y: float,
    half_w: float,
    half_h: float,
    bx_min: float,
    by_min: float,
    bx_max: float,
    by_max: float,
) -> float:
    """Return how far a rectangle (center x,y, half-extents) exceeds bounds.

    Returns 0.0 when fully contained; positive when outside.
    """
    left = bx_min - (x - half_w)
    right = (x + half_w) - bx_max
    top = by_min - (y - half_h)
    bottom = (y + half_h) - by_max
    return max(0.0, left, right, top, bottom)


def check_board_containment(
    pcb: PCBDesign,
    margin_mm: float = 0.0,
) -> tuple[ContainmentViolation, ...]:
    """Check all footprint elements are within the board outline.

    For each footprint, checks:

    1. Center point within board bounds + margin.
    2. Each pad (center +/- half size, after rotation) within board bounds.

    The board bounds are computed as the axis-aligned bounding box of the
    board outline polygon.

    Args:
        pcb: The PCB design to validate.
        margin_mm: Inward margin from the board edge.  A positive value
            shrinks the allowed region.

    Returns:
        Tuple of violations found.
    """
    bx_min, by_min, bx_max, by_max = _board_aabb(pcb)
    # Apply margin (shrink the allowed region)
    bx_min += margin_mm
    by_min += margin_mm
    bx_max -= margin_mm
    by_max -= margin_mm

    violations: list[ContainmentViolation] = []

    for fp in pcb.footprints:
        fp_x = fp.position.x
        fp_y = fp.position.y

        # 1. Check footprint center
        center_over = _overshoot(fp_x, fp_y, 0.0, 0.0, bx_min, by_min, bx_max, by_max)
        if center_over > 0.0:
            violations.append(
                ContainmentViolation(
                    ref=fp.ref,
                    element="center",
                    position_x=fp_x,
                    position_y=fp_y,
                    overshoot_mm=round(center_over, 4),
                    message=(
                        f"{fp.ref} center ({fp_x:.2f}, {fp_y:.2f}) is "
                        f"{center_over:.2f}mm outside board bounds"
                    ),
                )
            )

        # 2. Check each pad
        for pad in fp.pads:
            # Rotate pad position by footprint rotation
            rpx, rpy = _rotate_point(pad.position.x, pad.position.y, fp.rotation)
            world_x = fp_x + rpx
            world_y = fp_y + rpy

            # Pad extents: rotate the pad rectangle to get its AABB
            # For simplicity (axis-aligned check), compute the AABB of the
            # rotated pad rectangle.
            half_sx = pad.size_x / 2.0
            half_sy = pad.size_y / 2.0
            pad_rad = math.radians(-fp.rotation)
            cos_r = abs(math.cos(pad_rad))
            sin_r = abs(math.sin(pad_rad))
            aabb_half_w = half_sx * cos_r + half_sy * sin_r
            aabb_half_h = half_sx * sin_r + half_sy * cos_r

            pad_over = _overshoot(
                world_x, world_y, aabb_half_w, aabb_half_h,
                bx_min, by_min, bx_max, by_max,
            )
            if pad_over > 0.0:
                violations.append(
                    ContainmentViolation(
                        ref=fp.ref,
                        element="pad",
                        position_x=round(world_x, 4),
                        position_y=round(world_y, 4),
                        overshoot_mm=round(pad_over, 4),
                        message=(
                            f"{fp.ref} pad {pad.number} at ({world_x:.2f}, {world_y:.2f}) "
                            f"extends {pad_over:.2f}mm outside board bounds"
                        ),
                    )
                )

    return tuple(violations)
