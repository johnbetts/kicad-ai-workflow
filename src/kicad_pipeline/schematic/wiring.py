"""Wire routing between schematic symbol pins.

Provides helpers to create :class:`~kicad_pipeline.models.schematic.Wire`,
:class:`~kicad_pipeline.models.schematic.Junction`,
:class:`~kicad_pipeline.models.schematic.Label`, and
:class:`~kicad_pipeline.models.schematic.GlobalLabel` objects, and a
higher-level :func:`route_net` function that chooses between direct wires
and net labels depending on the positions of the pins involved.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.constants import SCHEMATIC_PIN_LENGTH_MM, SCHEMATIC_WIRE_GRID_MM

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import Net
from kicad_pipeline.models.schematic import (
    FontEffect,
    GlobalLabel,
    Junction,
    Label,
    Point,
    Stroke,
    Wire,
)

log = logging.getLogger(__name__)

# Distance (mm) to extend the wire stub away from a pin before placing a label.
_LABEL_STUB_MM: float = SCHEMATIC_PIN_LENGTH_MM * 3.0  # 7.62 mm — clears label overlap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_uuid() -> str:
    """Return a fresh RFC-4122 UUID string.

    Returns:
        UUID string in the form ``'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'``.
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Low-level wire / junction / label factories
# ---------------------------------------------------------------------------


def snap_to_grid(value: float, grid: float = SCHEMATIC_WIRE_GRID_MM) -> float:
    """Round *value* to the nearest multiple of *grid*.

    Args:
        value: Raw coordinate value in mm.
        grid: Grid pitch in mm (default ``SCHEMATIC_WIRE_GRID_MM`` = 1.27 mm).

    Returns:
        Grid-aligned coordinate.
    """
    return round(value / grid) * grid


def make_wire(x1: float, y1: float, x2: float, y2: float) -> Wire:
    """Create a :class:`Wire` between two grid-snapped endpoints.

    Both endpoints are snapped to the schematic grid before the object is
    constructed.

    Args:
        x1: X coordinate of the start point in mm.
        y1: Y coordinate of the start point in mm.
        x2: X coordinate of the end point in mm.
        y2: Y coordinate of the end point in mm.

    Returns:
        A new :class:`Wire` with a fresh UUID.
    """
    grid = SCHEMATIC_WIRE_GRID_MM
    return Wire(
        start=Point(x=snap_to_grid(x1, grid), y=snap_to_grid(y1, grid)),
        end=Point(x=snap_to_grid(x2, grid), y=snap_to_grid(y2, grid)),
        stroke=Stroke(),
        uuid=_new_uuid(),
    )


def make_junction(x: float, y: float) -> Junction:
    """Create a :class:`Junction` at a grid-snapped position.

    Args:
        x: X coordinate in mm.
        y: Y coordinate in mm.

    Returns:
        A new :class:`Junction` with a fresh UUID.
    """
    grid = SCHEMATIC_WIRE_GRID_MM
    return Junction(
        position=Point(x=snap_to_grid(x, grid), y=snap_to_grid(y, grid)),
        uuid=_new_uuid(),
    )


def make_global_label(
    text: str,
    x: float,
    y: float,
    shape: str = "bidirectional",
    rotation: float = 0.0,
) -> GlobalLabel:
    """Create a :class:`GlobalLabel` at a grid-snapped position.

    Args:
        text: Net name text displayed on the label.
        x: X coordinate in mm.
        y: Y coordinate in mm.
        shape: Label shape identifier (``'input'``, ``'output'``,
               ``'bidirectional'``, ``'tri_state'``, ``'passive'``).
        rotation: Label rotation in degrees (default ``0.0``).

    Returns:
        A new :class:`GlobalLabel` with a fresh UUID.
    """
    grid = SCHEMATIC_WIRE_GRID_MM
    return GlobalLabel(
        text=text,
        shape=shape,
        position=Point(x=snap_to_grid(x, grid), y=snap_to_grid(y, grid)),
        rotation=rotation,
        effects=FontEffect(),
        uuid=_new_uuid(),
    )


def make_label(text: str, x: float, y: float, rotation: float = 0.0) -> Label:
    """Create a :class:`Label` at a grid-snapped position.

    Args:
        text: Net name text displayed on the label.
        x: X coordinate in mm.
        y: Y coordinate in mm.
        rotation: Label rotation in degrees (default ``0.0``).

    Returns:
        A new :class:`Label` with a fresh UUID.
    """
    grid = SCHEMATIC_WIRE_GRID_MM
    return Label(
        text=text,
        position=Point(x=snap_to_grid(x, grid), y=snap_to_grid(y, grid)),
        rotation=rotation,
        effects=FontEffect(),
        uuid=_new_uuid(),
    )


# ---------------------------------------------------------------------------
# Pin-to-label connection helper
# ---------------------------------------------------------------------------


def connect_pin_to_label(
    pin_position: Point,
    label_text: str,
    is_global: bool = False,
    pin_side: str = "left",
) -> tuple[list[Wire], list[GlobalLabel], list[Label]]:
    """Generate a short wire stub and a net label for a pin.

    The wire extends away from the symbol body in the direction indicated by
    *pin_side*.  The label is placed at the far end of the stub.

    Args:
        pin_position: The position of the symbol pin endpoint.
        label_text: Net name to show on the label.
        is_global: If ``True``, generates a :class:`GlobalLabel`; otherwise a
            local :class:`Label`.
        pin_side: Which side of the symbol the pin is on.
            ``"left"`` → stub extends left, ``"right"`` → right,
            ``"top"`` → up, ``"bottom"`` → down.

    Returns:
        A three-element tuple ``(wires, global_labels, local_labels)`` where
        *wires* contains one :class:`Wire` and exactly one of the label lists
        is non-empty depending on *is_global*.
    """
    px, py = pin_position.x, pin_position.y
    # GlobalLabel rotation = direction the arrow POINTS (connection at arrow tip).
    # Label rotation = text reading direction (connection at start of text).
    # For "top" and "bottom" the local label rotation is opposite to global.
    if pin_side == "right":
        lx, ly = px + _LABEL_STUB_MM, py
        global_rotation = 0.0
        local_rotation = 0.0  # text reads left-to-right, connection at left end
    elif pin_side == "top":
        lx, ly = px, py - _LABEL_STUB_MM
        global_rotation = 270.0
        local_rotation = 90.0  # text reads bottom-to-top, connection at bottom
    elif pin_side == "bottom":
        lx, ly = px, py + _LABEL_STUB_MM
        global_rotation = 90.0
        local_rotation = 270.0  # text reads top-to-bottom, connection at top
    else:  # "left" (default)
        lx, ly = px - _LABEL_STUB_MM, py
        global_rotation = 180.0
        local_rotation = 180.0  # text reads right-to-left, connection at right end

    wire = make_wire(px, py, lx, ly)

    if is_global:
        return [wire], [make_global_label(label_text, lx, ly, rotation=global_rotation)], []
    return [wire], [], [make_label(label_text, lx, ly, rotation=local_rotation)]


# ---------------------------------------------------------------------------
# Net-level router
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ZoneKey:
    """Lightweight key identifying a placement zone by grid sector."""

    col: int
    row: int


def _zone_key(pt: Point, zone_w: float = 125.0, zone_h: float = 80.0) -> _ZoneKey:
    """Map a point to a grid-sector key for zone-proximity testing.

    Args:
        pt: The point to classify.
        zone_w: Approximate width of a placement zone in mm.
        zone_h: Approximate height of a placement zone in mm.

    Returns:
        A :class:`_ZoneKey` identifying the sector.
    """
    return _ZoneKey(col=int(pt.x // zone_w), row=int(pt.y // zone_h))


def route_net(
    net: Net,
    pin_positions: dict[tuple[str, str], Point],
    use_global_labels: bool = True,
    pin_sides: dict[tuple[str, str], str] | None = None,
) -> tuple[list[Wire], list[Junction], list[GlobalLabel], list[Label]]:
    """Route wires for a single net.

    Strategy:

    * **Two-pin net, same zone**: draw a direct orthogonal wire between the two
      pins (horizontal then vertical segments with a junction if needed).
    * **All other cases**: place a net label (global or local) at each pin.
      This keeps the schematic readable for multi-fan-out or long-distance
      connections.

    Args:
        net: The :class:`~kicad_pipeline.models.requirements.Net` to route.
        pin_positions: Mapping from ``(ref, pin_number)`` to the absolute
            :class:`Point` of that pin on the schematic canvas.
        use_global_labels: If ``True`` (default), global labels are used for
            multi-point or cross-zone routing; local labels otherwise.

    Returns:
        Four-element tuple ``(wires, junctions, global_labels, local_labels)``.
    """
    wires: list[Wire] = []
    junctions: list[Junction] = []
    global_labels: list[GlobalLabel] = []
    local_labels: list[Label] = []

    # Gather pin positions that we actually know about
    known_pins: list[tuple[str, str, Point]] = []
    for conn in net.connections:
        key = (conn.ref, conn.pin)
        if key in pin_positions:
            known_pins.append((conn.ref, conn.pin, pin_positions[key]))
        else:
            log.debug(
                "route_net(%s): pin %s.%s has no position; skipping",
                net.name,
                conn.ref,
                conn.pin,
            )

    if not known_pins:
        return wires, junctions, global_labels, local_labels

    # Label-per-pin: each pin gets its own wire stub and net label so KiCad
    # ERC sees every pin as connected.
    log.debug(
        "route_net(%s): label-per-pin for %d pins",
        net.name,
        len(known_pins),
    )
    for _ref, _pin, pt in known_pins:
        side = "left"
        if pin_sides is not None:
            side = pin_sides.get((_ref, _pin), "left")
        stub_wires, gls, lls = connect_pin_to_label(
            pt, net.name, is_global=use_global_labels, pin_side=side,
        )
        wires.extend(stub_wires)
        global_labels.extend(gls)
        local_labels.extend(lls)

    return wires, junctions, global_labels, local_labels
