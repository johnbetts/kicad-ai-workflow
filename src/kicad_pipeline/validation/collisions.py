"""Courtyard/footprint collision checker.

Detects overlapping component footprints using axis-aligned bounding boxes
derived from pad geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint, PCBDesign


@dataclass(frozen=True)
class CollisionViolation:
    """Two components whose footprints overlap or are too close."""

    ref_a: str
    ref_b: str
    gap_mm: float
    """Negative means overlap, positive means clearance."""
    message: str


def _footprint_aabb(fp: Footprint) -> tuple[float, float, float, float] | None:
    """Compute the world-space AABB of a footprint from its pads.

    Returns ``(min_x, min_y, max_x, max_y)`` or ``None`` if the footprint
    has no pads.
    """
    if not fp.pads:
        return None

    rad = math.radians(-fp.rotation)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    abs_cos = abs(cos_r)
    abs_sin = abs(sin_r)

    xs: list[float] = []
    ys: list[float] = []

    for pad in fp.pads:
        # Rotate pad center
        rpx = pad.position.x * cos_r - pad.position.y * sin_r
        rpy = pad.position.x * sin_r + pad.position.y * cos_r
        world_x = fp.position.x + rpx
        world_y = fp.position.y + rpy

        # Pad AABB half-extents after rotation
        half_sx = pad.size_x / 2.0
        half_sy = pad.size_y / 2.0
        hw = half_sx * abs_cos + half_sy * abs_sin
        hh = half_sx * abs_sin + half_sy * abs_cos

        xs.append(world_x - hw)
        xs.append(world_x + hw)
        ys.append(world_y - hh)
        ys.append(world_y + hh)

    return min(xs), min(ys), max(xs), max(ys)


def _aabb_gap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute the gap between two AABBs.

    Returns a negative value if they overlap (magnitude = overlap depth),
    positive if they are separated.
    """
    a_min_x, a_min_y, a_max_x, a_max_y = a
    b_min_x, b_min_y, b_max_x, b_max_y = b

    # Separation on each axis (positive = gap, negative = overlap)
    sep_x = max(a_min_x - b_max_x, b_min_x - a_max_x)
    sep_y = max(a_min_y - b_max_y, b_min_y - a_max_y)

    if sep_x > 0.0 or sep_y > 0.0:
        # No overlap — gap is the larger separation
        return max(sep_x, sep_y)

    # Overlap — gap is negative; return the axis with less overlap
    # (i.e., the larger of the two negative values → closest to zero)
    return max(sep_x, sep_y)


def check_collisions(
    pcb: PCBDesign,
    min_gap_mm: float = 0.0,
) -> tuple[CollisionViolation, ...]:
    """Check all footprint pairs for courtyard overlap.

    Uses axis-aligned bounding boxes for each footprint (computed from
    pads with rotation applied).  Two footprints collide if their AABBs
    overlap after accounting for *min_gap_mm*.

    Args:
        pcb: The PCB design to validate.
        min_gap_mm: Minimum required clearance between any two footprint
            AABBs.  Pairs closer than this threshold are reported.

    Returns:
        Tuple of violations found, sorted by ``gap_mm`` ascending
        (worst overlap first).
    """
    # Pre-compute AABBs, skipping footprints without pads
    fp_boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for fp in pcb.footprints:
        box = _footprint_aabb(fp)
        if box is not None:
            fp_boxes.append((fp.ref, box))

    violations: list[CollisionViolation] = []

    for i in range(len(fp_boxes)):
        ref_a, box_a = fp_boxes[i]
        for j in range(i + 1, len(fp_boxes)):
            ref_b, box_b = fp_boxes[j]
            if ref_a == ref_b:
                continue

            gap = _aabb_gap(box_a, box_b)
            if gap < min_gap_mm:
                if gap < 0.0:
                    msg = (
                        f"{ref_a} and {ref_b} overlap by {abs(gap):.2f}mm"
                    )
                else:
                    msg = (
                        f"{ref_a} and {ref_b} are {gap:.2f}mm apart "
                        f"(minimum required: {min_gap_mm:.2f}mm)"
                    )
                violations.append(
                    CollisionViolation(
                        ref_a=ref_a,
                        ref_b=ref_b,
                        gap_mm=round(gap, 4),
                        message=msg,
                    )
                )

    violations.sort(key=lambda v: v.gap_mm)
    return tuple(violations)
