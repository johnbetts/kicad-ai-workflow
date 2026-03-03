"""Copper zone and keepout generators for KiCad PCB designs.

Provides factory functions that produce :class:`~kicad_pipeline.models.pcb.ZonePolygon`
and :class:`~kicad_pipeline.models.pcb.Keepout` objects for common 2-layer PCB patterns:

* Full-board GND and power copper pours.
* Rectangular antenna keepout (e.g. ESP32 WiFi antenna area).
* Edge-clearance keepout strip around the board perimeter.
* Convenience bundles for standard 2-layer designs.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    PCB_EDGE_CUTS_WIDTH_MM,
    THERMAL_RELIEF_BRIDGE_MM,
    THERMAL_RELIEF_GAP_MM,
)
from kicad_pipeline.models.pcb import (
    BoardOutline,
    Keepout,
    Point,
    ZoneFill,
    ZonePolygon,
)

if TYPE_CHECKING:
    from kicad_pipeline.pcb.netlist import Netlist

_log = logging.getLogger(__name__)

# Number of polygon vertices used to approximate a circle
_CIRCLE_SEGMENTS: int = 16

# All-layers tuple used for antenna keepouts (no copper on any layer)
_ALL_COPPER_LAYERS: tuple[str, ...] = ("F.Cu", "B.Cu")

# Thermal relief defaults
_ZONE_MIN_THICKNESS: float = 0.127


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rect_polygon(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[Point, ...]:
    """Return a closed 5-point rectangle polygon (CW from top-left).

    Args:
        x0: Left edge x in mm.
        y0: Top edge y in mm.
        x1: Right edge x in mm.
        y1: Bottom edge y in mm.

    Returns:
        Closed polygon as a 5-element tuple of :class:`Point`.
    """
    return (
        Point(x0, y0),
        Point(x1, y0),
        Point(x1, y1),
        Point(x0, y1),
        Point(x0, y0),
    )


def _outline_bounds(
    board_outline: BoardOutline,
) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounding box of *board_outline*.

    Args:
        board_outline: The board outline whose bounding box is required.

    Returns:
        ``(min_x, min_y, max_x, max_y)`` in mm.
    """
    xs = [p.x for p in board_outline.polygon]
    ys = [p.y for p in board_outline.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _is_rectangular(board_outline: BoardOutline) -> bool:
    """Return True if *board_outline* has 4 or 5 unique-corner points.

    A closed rectangle stores 5 points (first == last).  An open rectangle
    stores 4.

    Args:
        board_outline: The outline to inspect.

    Returns:
        ``True`` when the polygon looks rectangular.
    """
    pts = board_outline.polygon
    n = len(pts)
    return n in (4, 5)


# ---------------------------------------------------------------------------
# Board outline factory (preserved for backward compatibility)
# ---------------------------------------------------------------------------


def make_board_outline(
    width_mm: float,
    height_mm: float,
    corner_radius_mm: float = 0.0,
) -> BoardOutline:
    """Create a rectangular board outline polygon on the Edge.Cuts layer.

    The outline runs counter-clockwise from the origin:
    ``(0,0) → (width,0) → (width,height) → (0,height)``.

    Rounded corners are not yet supported; *corner_radius_mm* is accepted for
    API compatibility but must be 0.0 in this release.

    Args:
        width_mm: Board width in mm.
        height_mm: Board height in mm.
        corner_radius_mm: Corner fillet radius in mm (must be 0.0).

    Returns:
        A :class:`BoardOutline` with four corner points.

    Raises:
        ValueError: When *corner_radius_mm* is non-zero (not yet supported).
    """
    if corner_radius_mm != 0.0:
        raise ValueError(
            "corner_radius_mm != 0 is not yet supported; use corner_radius_mm=0.0"
        )
    _log.debug("make_board_outline %.2f x %.2f mm", width_mm, height_mm)
    polygon = (
        Point(0.0, 0.0),
        Point(width_mm, 0.0),
        Point(width_mm, height_mm),
        Point(0.0, height_mm),
    )
    return BoardOutline(polygon=polygon, width=PCB_EDGE_CUTS_WIDTH_MM)


# ---------------------------------------------------------------------------
# Public zone factories
# ---------------------------------------------------------------------------


def make_gnd_pour(
    board_outline: BoardOutline,
    net_number: int = 1,
    net_name: str = "GND",
    layer: str = "B.Cu",
) -> ZonePolygon:
    """Create a full-board GND copper pour on *layer*.

    The zone polygon matches the board outline.  Thermal relief settings are
    taken from the project constants.

    Args:
        board_outline: Board outline used to derive the pour polygon.
        net_number: Net number for GND (default ``1``).
        net_name: Net name string (default ``"GND"``).
        layer: Copper layer for the pour (default ``"B.Cu"``).

    Returns:
        A :class:`ZonePolygon` covering the entire board outline.
    """
    _log.debug("make_gnd_pour layer=%s net=%d", layer, net_number)
    return ZonePolygon(
        net_number=net_number,
        net_name=net_name,
        layer=layer,
        name=net_name,
        polygon=board_outline.polygon,
        min_thickness=_ZONE_MIN_THICKNESS,
        fill=ZoneFill.SOLID,
        thermal_relief_gap=THERMAL_RELIEF_GAP_MM,
        thermal_relief_bridge=THERMAL_RELIEF_BRIDGE_MM,
        uuid="",
    )


def make_power_pour(
    board_outline: BoardOutline,
    net_number: int,
    net_name: str,
    layer: str = "F.Cu",
) -> ZonePolygon:
    """Create a full-board power copper pour on *layer*.

    Args:
        board_outline: Board outline used to derive the pour polygon.
        net_number: Net number for this power rail.
        net_name: Net name string (e.g. ``"+3V3"``).
        layer: Copper layer for the pour (default ``"F.Cu"``).

    Returns:
        A :class:`ZonePolygon` covering the entire board outline.
    """
    _log.debug("make_power_pour layer=%s net=%d name=%s", layer, net_number, net_name)
    return ZonePolygon(
        net_number=net_number,
        net_name=net_name,
        layer=layer,
        name=net_name,
        polygon=board_outline.polygon,
        min_thickness=_ZONE_MIN_THICKNESS,
        fill=ZoneFill.SOLID,
        thermal_relief_gap=THERMAL_RELIEF_GAP_MM,
        thermal_relief_bridge=THERMAL_RELIEF_BRIDGE_MM,
        uuid="",
    )


def gnd_pours_both_layers(
    outline: BoardOutline,
    gnd_net_number: int,
) -> tuple[ZonePolygon, ZonePolygon]:
    """Create GND copper pours on both F.Cu and B.Cu.

    Args:
        outline: :class:`BoardOutline` defining the board area.
        gnd_net_number: Net number assigned to the GND net.

    Returns:
        A two-tuple ``(front_pour, back_pour)``.
    """
    _log.debug("gnd_pours_both_layers net=%d", gnd_net_number)
    front = make_gnd_pour(outline, gnd_net_number, "GND", "F.Cu")
    back = make_gnd_pour(outline, gnd_net_number, "GND", "B.Cu")
    return front, back


# ---------------------------------------------------------------------------
# Keepout factories
# ---------------------------------------------------------------------------


def make_antenna_keepout(
    center_x: float,
    center_y: float,
    width_mm: float = 10.0,
    height_mm: float = 12.0,
) -> Keepout:
    """Create a rectangular keepout zone around an antenna area.

    Intended for ESP32 WiFi/BT antenna areas where copper must be excluded
    from both copper layers and via/track routing.  The rectangle is centred
    at (``center_x``, ``center_y``).

    Args:
        center_x: X centre of the keepout rectangle in mm.
        center_y: Y centre of the keepout rectangle in mm.
        width_mm: Width of the keepout rectangle in mm (default ``10.0``).
        height_mm: Height of the keepout rectangle in mm (default ``12.0``).

    Returns:
        A :class:`Keepout` with ``no_copper``, ``no_vias``, and
        ``no_tracks`` all set to ``True`` on both copper layers.
    """
    half_w = width_mm / 2.0
    half_h = height_mm / 2.0
    _log.debug(
        "make_antenna_keepout centre=(%.2f,%.2f) %.2f x %.2f",
        center_x,
        center_y,
        width_mm,
        height_mm,
    )
    polygon = _rect_polygon(
        center_x - half_w,
        center_y - half_h,
        center_x + half_w,
        center_y + half_h,
    )
    return Keepout(
        polygon=polygon,
        layers=("F.Cu", "B.Cu"),
        no_copper=True,
        no_vias=True,
        no_tracks=True,
        uuid="",
    )


def make_mounting_hole_keepout(
    x: float,
    y: float,
    diameter_mm: float = 3.2,
    clearance_mm: float = 1.0,
) -> Keepout:
    """Create a circular keepout zone around a mounting hole position.

    The circle is approximated with :data:`_CIRCLE_SEGMENTS` polygon vertices.
    The keepout radius equals ``diameter_mm / 2 + clearance_mm``.

    Args:
        x: X coordinate of the mounting hole centre, in mm.
        y: Y coordinate of the mounting hole centre, in mm.
        diameter_mm: Hole diameter in mm.
        clearance_mm: Additional clearance beyond the hole radius, in mm.

    Returns:
        A :class:`Keepout` with ``no_copper=True``.
    """
    radius = diameter_mm / 2.0 + clearance_mm
    _log.debug(
        "make_mounting_hole_keepout centre=(%.2f,%.2f) r=%.2f", x, y, radius
    )
    pts: list[Point] = []
    for i in range(_CIRCLE_SEGMENTS):
        angle = 2.0 * math.pi * i / _CIRCLE_SEGMENTS
        pts.append(Point(x + radius * math.cos(angle), y + radius * math.sin(angle)))

    return Keepout(
        polygon=tuple(pts),
        layers=_ALL_COPPER_LAYERS,
        no_copper=True,
        no_vias=False,
        no_tracks=False,
    )


def make_edge_keepout(
    board_outline: BoardOutline,
    margin_mm: float = 0.3,
) -> Keepout:
    """Create a keepout strip just inside the board edge.

    For rectangular boards the keepout polygon is the board bounding box
    inset by *margin_mm* on all sides.  For non-rectangular boards the
    original outline polygon is reused and only ``no_tracks`` is set.

    Args:
        board_outline: The board outline defining the perimeter.
        margin_mm: How far inside the board edge the keepout extends
            (default ``0.3`` mm — the JLCPCB recommended edge clearance).

    Returns:
        A :class:`Keepout` around the board perimeter.
    """
    if _is_rectangular(board_outline):
        min_x, min_y, max_x, max_y = _outline_bounds(board_outline)
        polygon = _rect_polygon(
            min_x + margin_mm,
            min_y + margin_mm,
            max_x - margin_mm,
            max_y - margin_mm,
        )
        return Keepout(
            polygon=polygon,
            layers=("F.Cu", "B.Cu"),
            no_copper=False,
            no_vias=False,
            no_tracks=True,
            uuid="",
        )

    # Non-rectangular: reuse the outline polygon, tracks only
    return Keepout(
        polygon=board_outline.polygon,
        layers=("F.Cu", "B.Cu"),
        no_copper=False,
        no_vias=False,
        no_tracks=True,
        uuid="",
    )


# ---------------------------------------------------------------------------
# Convenience bundles
# ---------------------------------------------------------------------------


def make_standard_zones(
    board_outline: BoardOutline,
    netlist: Netlist,
) -> tuple[ZonePolygon, ...]:
    """Build the standard copper zones for a 2-layer board.

    Always includes:

    * GND pour on ``B.Cu`` (net 1 by convention).

    Conditionally includes:

    * A ``+3V3`` / ``+3.3V`` power zone on ``F.Cu`` in the USB_POWER area
      (x 0-20 mm, y 0-12 mm) when that net exists in the netlist.

    Args:
        board_outline: The board outline.
        netlist: Populated netlist used to look up power net numbers.

    Returns:
        Tuple of :class:`ZonePolygon` objects.
    """
    zones: list[ZonePolygon] = [make_gnd_pour(board_outline)]

    # Look for +3V3 or +3.3V in the netlist
    power_net_name: str | None = None
    power_net_number: int | None = None
    for entry in netlist.entries:
        if entry.net.name in ("+3V3", "+3.3V"):
            power_net_name = entry.net.name
            power_net_number = entry.net.number
            break

    if power_net_name is not None and power_net_number is not None:
        usb_power_polygon = _rect_polygon(0.0, 0.0, 20.0, 12.0)
        zones.append(
            ZonePolygon(
                net_number=power_net_number,
                net_name=power_net_name,
                layer="F.Cu",
                name=power_net_name,
                polygon=usb_power_polygon,
                min_thickness=_ZONE_MIN_THICKNESS,
                fill=ZoneFill.SOLID,
                thermal_relief_gap=THERMAL_RELIEF_GAP_MM,
                thermal_relief_bridge=THERMAL_RELIEF_BRIDGE_MM,
                uuid="",
            )
        )

    return tuple(zones)


def make_standard_keepouts(
    board_outline: BoardOutline,
) -> tuple[Keepout, ...]:
    """Return standard keepout zones for a 2-layer board.

    Includes only the edge keepout (0.3 mm margin).  The antenna keepout
    should be added separately once the MCU placement is known by calling
    :func:`make_antenna_keepout`.

    Args:
        board_outline: The board outline.

    Returns:
        Tuple of :class:`Keepout` objects.
    """
    return (make_edge_keepout(board_outline),)
