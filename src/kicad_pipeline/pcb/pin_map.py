"""Pin map extraction: classify footprint pads by cardinal side.

Given a Footprint and rotation, determines which side (N/S/E/W) each pad
faces. Foundation for pin-aware placement, pad-facing scoring, and IC
layout templates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint, Pad


class CardinalSide(Enum):
    """Cardinal direction a pad faces relative to the footprint body."""

    NORTH = "north"   # top (negative Y in KiCad)
    SOUTH = "south"   # bottom (positive Y)
    EAST = "east"     # right (positive X)
    WEST = "west"     # left (negative X)
    CENTER = "center" # thermal pads near origin


@dataclass(frozen=True)
class PadSideEntry:
    """A single pad's side classification and positions."""

    pad_number: str
    side: CardinalSide
    local_position: tuple[float, float]    # unrotated (x, y)
    rotated_position: tuple[float, float]  # after rotation (x, y)
    net_name: str = ""


@dataclass(frozen=True)
class FootprintPinMap:
    """Complete pin-side mapping for a footprint at a given rotation."""

    ref: str
    rotation: float
    entries: tuple[PadSideEntry, ...]

    def pads_on_side(self, side: CardinalSide) -> tuple[PadSideEntry, ...]:
        """Return all pad entries on the given side."""
        return tuple(e for e in self.entries if e.side == side)

    def side_for_pad(self, pad_number: str) -> CardinalSide | None:
        """Return the side for a specific pad number, or None if not found."""
        for e in self.entries:
            if e.pad_number == pad_number:
                return e.side
        return None

    def nets_on_side(self, side: CardinalSide) -> frozenset[str]:
        """Return all net names connected to pads on the given side."""
        return frozenset(
            e.net_name for e in self.entries
            if e.side == side and e.net_name
        )


# ---------------------------------------------------------------------------
# Side classification
# ---------------------------------------------------------------------------

# Pads within this fraction of the footprint half-extent from the center
# are classified as CENTER (thermal pads).
_CENTER_THRESHOLD: float = 0.3


def classify_pad_side(
    dx: float,
    dy: float,
    half_w: float,
    half_h: float,
) -> CardinalSide:
    """Classify which side of the footprint body a pad is on.

    Uses normalized distance from center to determine the dominant axis.
    Pads very close to the origin (thermal pads) return CENTER.

    Args:
        dx: Pad X offset from footprint center.
        dy: Pad Y offset from footprint center.
        half_w: Half the footprint body width (X extent / 2).
        half_h: Half the footprint body height (Y extent / 2).

    Returns:
        The cardinal side the pad faces.
    """
    # Avoid division by zero for degenerate footprints
    safe_hw = max(half_w, 0.01)
    safe_hh = max(half_h, 0.01)

    norm_x = abs(dx) / safe_hw
    norm_y = abs(dy) / safe_hh

    # Thermal pad detection — close to center on both axes
    if norm_x < _CENTER_THRESHOLD and norm_y < _CENTER_THRESHOLD:
        return CardinalSide.CENTER

    if norm_x >= norm_y:
        return CardinalSide.EAST if dx >= 0 else CardinalSide.WEST
    return CardinalSide.SOUTH if dy >= 0 else CardinalSide.NORTH


def _rotate_point(
    px: float, py: float, angle_deg: float,
) -> tuple[float, float]:
    """Rotate a point around the origin by *angle_deg* degrees (KiCad CW convention)."""
    # KiCad stores rotation as CW in screen view (Y-down).
    # Negate angle for the standard CCW rotation matrix.
    rad = math.radians(-angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return px * cos_a - py * sin_a, px * sin_a + py * cos_a


_SIDE_ROTATION: dict[CardinalSide, dict[int, CardinalSide]] = {
    CardinalSide.NORTH: {0: CardinalSide.NORTH, 90: CardinalSide.EAST,
                         180: CardinalSide.SOUTH, 270: CardinalSide.WEST},
    CardinalSide.SOUTH: {0: CardinalSide.SOUTH, 90: CardinalSide.WEST,
                         180: CardinalSide.NORTH, 270: CardinalSide.EAST},
    CardinalSide.EAST:  {0: CardinalSide.EAST, 90: CardinalSide.SOUTH,
                         180: CardinalSide.WEST, 270: CardinalSide.NORTH},
    CardinalSide.WEST:  {0: CardinalSide.WEST, 90: CardinalSide.NORTH,
                         180: CardinalSide.EAST, 270: CardinalSide.SOUTH},
    CardinalSide.CENTER: {0: CardinalSide.CENTER, 90: CardinalSide.CENTER,
                          180: CardinalSide.CENTER, 270: CardinalSide.CENTER},
}


def rotate_side(side: CardinalSide, angle_deg: float) -> CardinalSide:
    """Rotate a cardinal side by the given angle.

    Supports arbitrary angles by snapping to the nearest 90-degree
    increment. CENTER is invariant under rotation.

    Args:
        side: The original side at 0 degrees rotation.
        angle_deg: Clockwise rotation in degrees.

    Returns:
        The rotated cardinal side.
    """
    if side == CardinalSide.CENTER:
        return CardinalSide.CENTER
    # Snap to nearest 90 degrees
    snapped = round(angle_deg / 90.0) % 4 * 90
    return _SIDE_ROTATION[side][snapped]


# ---------------------------------------------------------------------------
# Pin map computation
# ---------------------------------------------------------------------------


def _footprint_pad_extent(pads: tuple[Pad, ...]) -> tuple[float, float]:
    """Compute half-width and half-height from pad positions."""
    if not pads:
        return 1.0, 1.0
    xs = [p.position.x for p in pads]
    ys = [p.position.y for p in pads]
    half_w = max(max(xs) - min(xs), 1.0) / 2.0
    half_h = max(max(ys) - min(ys), 1.0) / 2.0
    return half_w, half_h


def compute_pin_map(
    footprint: Footprint,
    rotation: float = 0.0,
) -> FootprintPinMap:
    """Compute the pin map for a footprint at a given rotation.

    Classifies each pad by which side of the footprint body it's on,
    accounting for the specified rotation.

    Args:
        footprint: The footprint to analyze.
        rotation: Board-level rotation in degrees (clockwise).

    Returns:
        A FootprintPinMap with all pad entries.
    """
    half_w, half_h = _footprint_pad_extent(footprint.pads)
    entries: list[PadSideEntry] = []

    for pad in footprint.pads:
        dx, dy = pad.position.x, pad.position.y
        # Classify at 0 degrees
        base_side = classify_pad_side(dx, dy, half_w, half_h)
        # Rotate the side
        rotated_side = rotate_side(base_side, rotation)
        # Rotate the position
        rx, ry = _rotate_point(dx, dy, rotation)

        entries.append(PadSideEntry(
            pad_number=pad.number,
            side=rotated_side,
            local_position=(dx, dy),
            rotated_position=(rx, ry),
            net_name=pad.net_name or "",
        ))

    return FootprintPinMap(
        ref=footprint.ref,
        rotation=rotation,
        entries=tuple(entries),
    )


def compute_centroid_offset(footprint: Footprint) -> tuple[float, float]:
    """Compute offset from footprint origin to pad centroid in local coords.

    KiCad footprint origins are at pin 1 for connectors/headers, not
    the geometric center of the pad bounding box.  The placement
    optimizer works in centroid-of-pads coordinates, so this offset
    is needed to convert between the two coordinate systems.

    Returns:
        (cx, cy) — centroid of pads in local (unrotated) coordinates,
        relative to the footprint origin.  For symmetric footprints
        (e.g., 0603 resistors) this is (0, 0).
    """
    if not footprint.pads:
        return (0.0, 0.0)
    xs = [p.position.x for p in footprint.pads]
    ys = [p.position.y for p in footprint.pads]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return (cx, cy)


def origin_to_centroid(
    footprint: Footprint,
    ox: float,
    oy: float,
    rotation: float,
) -> tuple[float, float]:
    """Convert a KiCad origin position to centroid-of-pads position.

    Args:
        footprint: The footprint (needed for pad geometry).
        ox: KiCad origin X.
        oy: KiCad origin Y.
        rotation: Footprint rotation in degrees.

    Returns:
        (cx, cy) — centroid position in board coordinates.
    """
    loc_cx, loc_cy = compute_centroid_offset(footprint)
    # KiCad rotation convention: positive angle = CW in screen view
    # (Y-down coordinate system).  The standard rotation matrix is CCW
    # for positive angles in Y-up, so we negate to match KiCad.
    rad = math.radians(-rotation)
    rot_cx = loc_cx * math.cos(rad) - loc_cy * math.sin(rad)
    rot_cy = loc_cx * math.sin(rad) + loc_cy * math.cos(rad)
    return (ox + rot_cx, oy + rot_cy)


def centroid_to_origin(
    footprint: Footprint,
    cx: float,
    cy: float,
    rotation: float,
) -> tuple[float, float]:
    """Convert a centroid-of-pads position back to KiCad origin position.

    Inverse of :func:`origin_to_centroid`.

    Args:
        footprint: The footprint (needed for pad geometry).
        cx: Centroid X in board coordinates.
        cy: Centroid Y in board coordinates.
        rotation: Footprint rotation in degrees.

    Returns:
        (ox, oy) — KiCad origin position.
    """
    loc_cx, loc_cy = compute_centroid_offset(footprint)
    # Match KiCad CW rotation convention (see origin_to_centroid).
    rad = math.radians(-rotation)
    rot_cx = loc_cx * math.cos(rad) - loc_cy * math.sin(rad)
    rot_cy = loc_cx * math.sin(rad) + loc_cy * math.cos(rad)
    return (cx - rot_cx, cy - rot_cy)


def pad_extent_in_board_space(
    footprint: Footprint,
    ox: float,
    oy: float,
    rotation: float,
) -> tuple[float, float, float, float]:
    """Compute the bounding box of all pads in board coordinates.

    Given a footprint placed at (ox, oy) with the given rotation,
    returns (min_x, min_y, max_x, max_y) covering all pad centers.
    Useful for checking whether a component extends past the board edge.

    Args:
        footprint: The footprint (pad geometry).
        ox: KiCad origin X in board space.
        oy: KiCad origin Y in board space.
        rotation: Rotation in degrees.

    Returns:
        (min_x, min_y, max_x, max_y) bounding box of pad centers.
    """
    if not footprint.pads:
        return (ox, oy, ox, oy)
    # KiCad rotation convention: positive angle = CW in screen view.
    # Negate to match (see origin_to_centroid for rationale).
    rad = math.radians(-rotation)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    board_xs: list[float] = []
    board_ys: list[float] = []
    for pad in footprint.pads:
        px, py = pad.position.x, pad.position.y
        bx = ox + px * cos_a - py * sin_a
        by = oy + px * sin_a + py * cos_a
        board_xs.append(bx)
        board_ys.append(by)
    return (min(board_xs), min(board_ys), max(board_xs), max(board_ys))


def compute_pin_map_for_component(
    ref: str,
    value: str,
    footprint_id: str,
    rotation: float = 0.0,
    lcsc: str | None = None,
) -> FootprintPinMap | None:
    """Generate a footprint and compute its pin map.

    Convenience function that creates a footprint via
    ``footprint_for_component()`` and then computes the pin map.

    Args:
        ref: Component reference designator.
        value: Component value string.
        footprint_id: Footprint identifier (e.g., "R_0805").
        rotation: Board-level rotation in degrees.
        lcsc: Optional LCSC part number.

    Returns:
        The pin map, or None if the footprint cannot be generated.
    """
    from kicad_pipeline.pcb.footprints import footprint_for_component

    try:
        fp = footprint_for_component(ref, value, footprint_id, lcsc=lcsc)
    except Exception:
        return None

    return compute_pin_map(fp, rotation)
