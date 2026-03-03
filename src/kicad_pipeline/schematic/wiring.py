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
_LABEL_STUB_MM: float = SCHEMATIC_PIN_LENGTH_MM  # 2.54 mm


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
) -> tuple[list[Wire], list[GlobalLabel], list[Label]]:
    """Generate a short wire stub and a net label for a pin.

    The wire extends ``SCHEMATIC_PIN_LENGTH_MM`` (2.54 mm) to the right of the
    pin position.  The label is placed at the far end of the stub.

    Args:
        pin_position: The position of the symbol pin endpoint.
        label_text: Net name to show on the label.
        is_global: If ``True``, generates a :class:`GlobalLabel`; otherwise a
            local :class:`Label`.

    Returns:
        A three-element tuple ``(wires, global_labels, local_labels)`` where
        *wires* contains one :class:`Wire` and exactly one of the label lists
        is non-empty depending on *is_global*.
    """
    lx = pin_position.x + _LABEL_STUB_MM
    ly = pin_position.y
    wire = make_wire(pin_position.x, pin_position.y, lx, ly)

    if is_global:
        return [wire], [make_global_label(label_text, lx, ly)], []
    return [wire], [], [make_label(label_text, lx, ly)]


# ---------------------------------------------------------------------------
# Net-level router
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ZoneKey:
    """Lightweight key identifying a placement zone by grid sector."""

    col: int
    row: int


def _zone_key(pt: Point, zone_w: float = 130.0, zone_h: float = 90.0) -> _ZoneKey:
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

    # Attempt direct connection only for exactly 2 pins in the same zone
    if len(known_pins) == 2:
        _, _, pt_a = known_pins[0]
        _, _, pt_b = known_pins[1]
        if _zone_key(pt_a) == _zone_key(pt_b):
            log.debug(
                "route_net(%s): direct wire (same zone)", net.name
            )
            # Horizontal + vertical L-route
            w1 = make_wire(pt_a.x, pt_a.y, pt_b.x, pt_a.y)
            w2 = make_wire(pt_b.x, pt_a.y, pt_b.x, pt_b.y)
            wires.append(w1)
            if abs(pt_a.y - pt_b.y) > 1e-6:
                # Only add the second segment and a junction when they differ in y
                wires.append(w2)
                junctions.append(make_junction(pt_b.x, pt_a.y))
            return wires, junctions, global_labels, local_labels

    # Label-per-pin strategy
    log.debug(
        "route_net(%s): label-per-pin for %d pins",
        net.name,
        len(known_pins),
    )
    for _ref, _pin, pt in known_pins:
        stub_wires, gls, lls = connect_pin_to_label(
            pt, net.name, is_global=use_global_labels
        )
        wires.extend(stub_wires)
        global_labels.extend(gls)
        local_labels.extend(lls)

    return wires, junctions, global_labels, local_labels
