"""Keepout zone generation for PCB designs.

Generates keepout zones for mounting holes, RF antennas, and RF module
bodies.  All coordinates are in millimetres.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from kicad_pipeline.constants import LAYER_B_CU, LAYER_F_CU
from kicad_pipeline.models.pcb import Keepout, Point

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOUNTING_HOLE_INSET_MM: float = 3.5
"""Distance from board corner to mounting-hole centre in mm."""

MOUNTING_HOLE_DIAMETER_MM: float = 3.2
"""Mounting hole drill diameter in mm (M3 screw)."""

KEEPOUT_MARGIN_MM: float = 3.0
"""Radius of the keepout zone around each mounting hole in mm.

Must be <= ``MOUNTING_HOLE_INSET_MM`` (3.5 mm) to prevent keepout
circles from extending past the board edge.
"""

ANTENNA_KEEPOUT_WIDTH_MM: float = 15.0
"""Width of the no-copper keepout zone reserved for an ESP32 antenna in mm."""

ANTENNA_KEEPOUT_HEIGHT_MM: float = 10.0
"""Height of the no-copper keepout zone reserved for an ESP32 antenna in mm."""

RF_MODULE_BODY_WIDTH_MM: float = 18.0
"""Width of the ESP32-S3-WROOM-1 module body in mm."""

RF_MODULE_BODY_HEIGHT_MM: float = 25.5
"""Height of the ESP32-S3-WROOM-1 module body in mm."""

# Keywords that indicate the design contains an RF module requiring a keepout
RF_KEYWORDS: frozenset[str] = frozenset({"esp32", "esp8266", "nrf", "cc3200", "rf"})


def _new_uuid() -> str:
    """Return a fresh RFC-4122 UUID string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Keepout generators
# ---------------------------------------------------------------------------


def make_mounting_hole_keepouts(
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
        # Do NOT explicitly close -- KiCad auto-closes polygons.
        # Explicit closure creates a zero-length edge -> "malformed" warning.
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


def _antenna_origin_from_rf_position(
    rf_position: tuple[float, float, float],
    width: float,
    height: float,
    board_width: float,
    board_height: float,
) -> tuple[float, float]:
    """Compute the top-left corner of the antenna keepout from RF module position.

    Uses the module half-height to locate the antenna end, then rotates and
    clamps the result to the board bounds.

    ``rf_position`` is ``(cx, cy, rotation_deg)`` of the footprint origin.
    """
    import math as _m
    cx, cy, rot = rf_position
    # ESP32-S3-WROOM-1: antenna is at the top of the module body
    # (negative Y in footprint-local coordinates). We use the module
    # half-height (12.75mm) minus half the keepout height as the offset
    # from body centre.
    module_half_h = RF_MODULE_BODY_HEIGHT_MM / 2.0  # 12.75 mm
    antenna_offset = module_half_h - height / 2.0
    angle_rad = _m.radians(rot)
    # In unrotated position, antenna points in -Y direction.
    dx = -antenna_offset * _m.sin(angle_rad)
    dy = -antenna_offset * _m.cos(angle_rad)
    ax = cx + dx
    ay = cy + dy
    x0 = ax - width / 2.0
    y0 = ay - height / 2.0
    # Clamp to board bounds
    x0 = max(0.0, min(x0, board_width - width))
    y0 = max(0.0, min(y0, board_height - height))
    return x0, y0


def make_antenna_keepout(
    board_width: float,
    width: float,
    height: float,
    rf_position: tuple[float, float, float] | None = None,
    layer_count: int = 2,
    board_height: float = 80.0,
) -> Keepout:
    """Create a no-copper keepout zone for an RF antenna.

    When *rf_position* is given ``(x, y, rotation_deg)`` the keepout is
    placed at the antenna end of the module, accounting for rotation.
    Otherwise falls back to the top-right corner of the board.

    Args:
        board_width: Total board width in mm (fallback positioning).
        width: Width of the antenna keepout zone in mm.
        height: Height of the antenna keepout zone in mm.
        rf_position: Optional ``(x, y, rotation_deg)`` of the RF module.
        layer_count: Number of copper layers (keepout spans all layers).
        board_height: Total board height in mm (for clamping).

    Returns:
        A :class:`Keepout` covering the antenna area.
    """
    if rf_position is not None:
        x0, y0 = _antenna_origin_from_rf_position(
            rf_position, width, height, board_width, board_height,
        )
    else:
        # Fallback: top-right corner
        x0 = board_width - width
        y0 = 0.0

    # Extend the keepout to the nearest board edge on the antenna side.
    # The antenna needs clear copper all the way to the board edge.
    # Determine which edge is closest to the keepout center.
    keepout_center_y = y0 + height / 2.0
    dist_to_top = keepout_center_y
    dist_to_bottom = board_height - keepout_center_y
    if dist_to_bottom <= dist_to_top:
        # Antenna faces bottom edge — extend keepout to board bottom
        y_bottom = board_height
    else:
        # Antenna faces top edge — extend keepout to board top
        y0 = 0.0
        y_bottom = y0 + height

    # Do NOT explicitly close -- KiCad auto-closes polygons for keepouts.
    polygon = (
        Point(x=x0, y=y0),
        Point(x=x0 + width, y=y0),
        Point(x=x0 + width, y=y_bottom),
        Point(x=x0, y=y_bottom),
    )
    # Keepout on all copper layers for proper isolation
    layers: list[str] = [LAYER_F_CU, LAYER_B_CU]
    if layer_count >= 4:
        layers.extend(["In1.Cu", "In2.Cu"])
    return Keepout(
        polygon=tuple(polygon),
        layers=tuple(layers),
        no_copper=True,
        no_vias=False,
        no_tracks=True,
        uuid=_new_uuid(),
    )


def make_rf_module_body_keepout(
    rf_position: tuple[float, float, float],
    layer_count: int = 2,
    board_width: float = 150.0,
    board_height: float = 80.0,
) -> Keepout | None:
    """Create an inner-layer keepout covering the RF module body.

    Prevents ground/power pours on inner copper layers from degrading
    WiFi/BT antenna performance. Only applies to In1.Cu and In2.Cu --
    F.Cu and B.Cu are left alone because the module's castellated pads
    need copper on the outer layers.

    Args:
        rf_position: ``(x, y, rotation_deg)`` of the RF module.
        layer_count: Number of copper layers (needs >= 4 for inner layers).
        board_width: Total board width in mm (for clamping).
        board_height: Total board height in mm (for clamping).

    Returns:
        A :class:`Keepout` on inner layers, or ``None`` if < 4 layers.
    """
    if layer_count < 4:
        return None

    import math as _m

    cx, cy, rot = rf_position
    angle_rad = _m.radians(rot)
    hw = RF_MODULE_BODY_WIDTH_MM / 2.0
    hh = RF_MODULE_BODY_HEIGHT_MM / 2.0

    # Module body corners in local coordinates (centered on module)
    local_corners = [
        (-hw, -hh),
        ( hw, -hh),
        ( hw,  hh),
        (-hw,  hh),
    ]

    # Rotate and translate to board coordinates
    cos_a = _m.cos(angle_rad)
    sin_a = _m.sin(angle_rad)
    polygon: list[Point] = []
    for lx, ly in local_corners:
        bx = cx + lx * cos_a - ly * sin_a
        by = cy + lx * sin_a + ly * cos_a
        # Clamp to board bounds
        bx = max(0.0, min(bx, board_width))
        by = max(0.0, min(by, board_height))
        polygon.append(Point(x=bx, y=by))

    return Keepout(
        polygon=tuple(polygon),
        layers=("In1.Cu", "In2.Cu"),
        no_copper=True,
        no_vias=False,
        no_tracks=False,
        uuid=_new_uuid(),
    )


def has_rf_module(requirements: ProjectRequirements) -> bool:
    """Return True if any component value suggests an RF / WiFi module.

    Args:
        requirements: Project requirements document.

    Returns:
        ``True`` when an RF-type component is detected.
    """
    for comp in requirements.components:
        val_lower = comp.value.lower()
        if any(kw in val_lower for kw in RF_KEYWORDS):
            return True
    return False
