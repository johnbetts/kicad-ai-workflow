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
import math
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
    PowerSymbol,
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
    justify: str = "",
) -> GlobalLabel:
    """Create a :class:`GlobalLabel` at a grid-snapped position.

    Args:
        text: Net name text displayed on the label.
        x: X coordinate in mm.
        y: Y coordinate in mm.
        shape: Label shape identifier (``'input'``, ``'output'``,
               ``'bidirectional'``, ``'tri_state'``, ``'passive'``).
        rotation: Label rotation in degrees (default ``0.0``).
        justify: Text justification (``''``, ``'left'``, ``'right'``).

    Returns:
        A new :class:`GlobalLabel` with a fresh UUID.
    """
    grid = SCHEMATIC_WIRE_GRID_MM
    return GlobalLabel(
        text=text,
        shape=shape,
        position=Point(x=snap_to_grid(x, grid), y=snap_to_grid(y, grid)),
        rotation=rotation,
        effects=FontEffect(justify=justify),
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
    # KiCad global_label rotation: direction the label arrow points.
    #   0° = right, 90° = down, 180° = left, 270° = up
    # Convention: label arrow points AWAY from the symbol body (toward the net).
    # (justify right) is needed when rotation=180 to keep text readable.
    global_justify = ""
    if pin_side == "right":
        lx, ly = px + _LABEL_STUB_MM, py
        global_rotation = 0.0  # arrow points right (away from symbol)
        local_rotation = 0.0
    elif pin_side == "top":
        lx, ly = px, py - _LABEL_STUB_MM
        global_rotation = 270.0  # arrow points up (away from symbol)
        local_rotation = 90.0
    elif pin_side == "bottom":
        lx, ly = px, py + _LABEL_STUB_MM
        global_rotation = 90.0  # arrow points down (away from symbol)
        local_rotation = 270.0
    else:  # "left" (default)
        lx, ly = px - _LABEL_STUB_MM, py
        global_rotation = 180.0  # arrow points left (away from symbol)
        global_justify = "right"
        local_rotation = 180.0

    wire = make_wire(px, py, lx, ly)

    if is_global:
        gl = make_global_label(
            label_text, lx, ly, rotation=global_rotation, justify=global_justify,
        )
        return [wire], [gl], []
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


# ---------------------------------------------------------------------------
# Stub collision resolution (KI-024)
# ---------------------------------------------------------------------------

@dataclass
class _Stub:
    """A movable pin stub: wire index + geometry + its attachment."""

    wire: int
    start: tuple[float, float]
    end: tuple[float, float]
    net: str
    kind: str  # "gl" | "ll" | "ps"
    attach: int
    movable: bool = True


#: Candidate stub lengths tried in order when the default collides.
_STUB_CANDIDATES_MM: tuple[float, ...] = (
    _LABEL_STUB_MM, 5.08, 2.54, 10.16, 12.70,
)
_EPS = 0.01


def _seg_pt_dist(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    ln2 = dx * dx + dy * dy
    if ln2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ln2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _segments_touch(
    a1: tuple[float, float], a2: tuple[float, float],
    b1: tuple[float, float], b2: tuple[float, float],
) -> bool:
    """True when two segments intersect, touch, or overlap (incl. ends).

    KiCad merges nets on ANY contact between wires, so touching counts.
    Generated stubs are axis-parallel; endpoint/distance tests cover the
    proper-crossing case too (an axis-parallel crossing puts one
    segment's interior point at distance 0 from the other — checked via
    both segments' endpoints plus the orthogonal projection case below).
    """
    # Endpoint of one on the other (covers touch, T-joins, overlap).
    for p in (a1, a2):
        if _seg_pt_dist(p, b1, b2) < _EPS:
            return True
    for p in (b1, b2):
        if _seg_pt_dist(p, a1, a2) < _EPS:
            return True
    # Proper crossing of two axis-aligned perpendicular segments: the
    # crossing point is (vertical.x, horizontal.y).
    a_vert = abs(a1[0] - a2[0]) < _EPS
    b_vert = abs(b1[0] - b2[0]) < _EPS
    if a_vert != b_vert:
        v1, v2, h1, h2 = (a1, a2, b1, b2) if a_vert else (b1, b2, a1, a2)
        x = v1[0]
        y = h1[1]
        if (min(h1[0], h2[0]) - _EPS <= x <= max(h1[0], h2[0]) + _EPS
                and min(v1[1], v2[1]) - _EPS <= y <= max(v1[1], v2[1]) + _EPS):
            return True
    return False


def resolve_stub_collisions(
    wires: list[Wire],
    global_labels: list[GlobalLabel],
    local_labels: list[Label],
    power_symbols: list[PowerSymbol],
    pin_nets: dict[tuple[float, float], str],
) -> int:
    """Shorten label/power stubs that touch foreign nets (KI-024).

    Every connection is drawn as a pin stub with a label (or power
    symbol) at its far end. With dense symbol placement, a stub can
    touch a FOREIGN net's stub, label anchor, or pin — KiCad then
    merges the nets and the written schematic diverges from the PCB
    netlist (ethernet trainer: one AVDD decoupling label swallowed
    +3V3, GND and both LED nets).

    Strategy: identify each simple stub (wire starting on a pin with
    exactly one label/symbol at its far end), then per stub — in
    deterministic order — test its segment, endpoint, and anchor
    against all foreign geometry; on contact, retry with the candidate
    lengths in :data:`_STUB_CANDIDATES_MM` and keep the first clean
    one, moving the attachment with the endpoint. Unresolvable stubs
    are left in place and reported (the written-file sync gate fails
    the build honestly).

    Returns:
        Number of stubs that remained in conflict after resolution.
    """
    def key(p: Point) -> tuple[float, float]:
        return (round(p.x, 3), round(p.y, 3))

    # Attachments by anchor point.
    gl_at: dict[tuple[float, float], list[int]] = {}
    for i, gl in enumerate(global_labels):
        gl_at.setdefault(key(gl.position), []).append(i)
    ll_at: dict[tuple[float, float], list[int]] = {}
    for i, ll in enumerate(local_labels):
        ll_at.setdefault(key(ll.position), []).append(i)
    ps_at: dict[tuple[float, float], list[int]] = {}
    for i, ps in enumerate(power_symbols):
        ps_at.setdefault(key(ps.position), []).append(i)

    # Classify wires into simple stubs vs static segments.
    stubs: list[_Stub] = []
    static: list[tuple[tuple[float, float], tuple[float, float], str | None]] = []
    for wi, w in enumerate(wires):
        start, end = key(w.start), key(w.end)
        pin_net = pin_nets.get(start)
        if pin_net is None and pin_nets.get(end) is not None:
            start, end = end, start
            pin_net = pin_nets.get(start)
        attach_kind = attach_idx = None
        net = pin_net
        if pin_net is not None:
            if gl_at.get(end):
                attach_kind, attach_idx = "gl", gl_at[end][0]
                net = str(global_labels[attach_idx].text)
            elif ps_at.get(end):
                attach_kind, attach_idx = "ps", ps_at[end][0]
                net = str(power_symbols[attach_idx].value)
            elif ll_at.get(end):
                attach_kind, attach_idx = "ll", ll_at[end][0]
                net = str(local_labels[attach_idx].text)
        if pin_net is not None and attach_kind is not None and attach_idx is not None:
            stubs.append(_Stub(
                wire=wi, start=start, end=end, net=str(net),
                kind=attach_kind, attach=attach_idx,
            ))
        else:
            static.append((start, end, None))


    # Net EVIDENCE for static wires: a consolidated power bus is static
    # (its segments carry no anchor of their own), but it is NOT
    # foreign to its own net — treating it so once made the resolver
    # shorten a bus's middle-pin stub and drag the shared symbol off
    # the bus, orphaning every other pin (mcu_core, 2026-06-12).
    # Evidence = nets of pins touching the segment, plus the nets of
    # POWER-SYMBOL stubs whose endpoints touch it (a bus is defined by
    # its pins and its symbol; a global LABEL touching it is precisely
    # the trespasser we must stay free to move away — counting labels
    # as evidence froze the AVDD label onto the +3V3 bus), propagated
    # across touching static segments to a fixpoint.
    static_nets: list[set[str]] = [set() for _ in static]
    for si, (s1, s2, _x) in enumerate(static):
        for ppt, pnet in pin_nets.items():
            if _seg_pt_dist(ppt, s1, s2) < _EPS:
                static_nets[si].add(pnet)
        for stub in stubs:
            if stub.kind != "ps":
                continue
            if (_seg_pt_dist(stub.start, s1, s2) < _EPS
                    or _seg_pt_dist(stub.end, s1, s2) < _EPS):
                static_nets[si].add(stub.net)
    changed = True
    while changed:
        changed = False
        for i, (a1, a2, _x) in enumerate(static):
            for j, (b1, b2, _y) in enumerate(static):
                if i == j or static_nets[j] <= static_nets[i]:
                    continue
                if _segments_touch(a1, a2, b1, b2):
                    static_nets[i] |= static_nets[j]
                    changed = True

    # A stub whose END T-joins ITS OWN net's bus is STRUCTURAL: it is
    # the middle leg of a consolidated power bus carrying the shared
    # symbol; moving it would orphan every other pin (W5500 +3V3 rail,
    # 2026-06-12). A stub touching a FOREIGN bus stays movable — it is
    # the trespasser.
    for stub in stubs:
        stub.movable = not any(
            _seg_pt_dist(stub.end, s1, s2) < _EPS
            and static_nets[si] == {stub.net}
            for si, (s1, s2, _x) in enumerate(static)
        )

    def conflicts(
        seg_start: tuple[float, float],
        seg_end: tuple[float, float],
        net: str,
        skip_wire: int,
    ) -> bool:
        # Foreign pins anywhere on the stub.
        for ppt, pnet in pin_nets.items():
            if pnet != net and _seg_pt_dist(ppt, seg_start, seg_end) < _EPS:
                return True
        # Static wires: same-net evidence exempts (a bus belongs to its
        # stubs); unknown or mixed evidence stays foreign.
        for si, (s1, s2, _snet) in enumerate(static):
            if static_nets[si] == {net}:
                continue
            if _segments_touch(seg_start, seg_end, s1, s2):
                return True
        # Other stubs (current geometry).
        for other in stubs:
            if other.wire == skip_wire or other.net == net:
                continue
            if _segments_touch(seg_start, seg_end, other.start, other.end):
                return True
        # Foreign label/symbol anchors on this stub.
        for anchors in (gl_at, ll_at, ps_at):
            for apt in anchors:
                if apt in (seg_start, seg_end):
                    continue
                if _seg_pt_dist(apt, seg_start, seg_end) < _EPS:
                    return True
        return False

    unresolved = 0
    for stub in sorted(stubs, key=lambda s: (s.start, s.end)):
        start = stub.start
        end = stub.end
        net = stub.net
        wi = stub.wire
        if not stub.movable:
            continue
        if not conflicts(start, end, net, wi):
            continue
        sx, sy = start
        ex, ey = end
        length = abs(ex - sx) + abs(ey - sy)
        ux = 0.0 if abs(ex - sx) < _EPS else (1.0 if ex > sx else -1.0)
        uy = 0.0 if abs(ey - sy) < _EPS else (1.0 if ey > sy else -1.0)
        fixed = False
        # Same-direction lengths first; FLIPPED direction as a last
        # resort. A flipped stub runs across the symbol's artwork —
        # ugly but electrically inert (only pins connect), and checked
        # against the body's other pins like everything else.
        candidates = [
            (ux, uy, cand, False) for cand in _STUB_CANDIDATES_MM
            if abs(cand - length) >= _EPS
        ] + [(-ux, -uy, cand, True) for cand in _STUB_CANDIDATES_MM]
        for cux, cuy, cand, flipped in candidates:
            new_end = (round(sx + cux * cand, 3), round(sy + cuy * cand, 3))
            if conflicts(start, new_end, net, wi):
                continue
            wires[wi] = Wire(
                start=Point(sx, sy), end=Point(new_end[0], new_end[1]),
                stroke=wires[wi].stroke, uuid=wires[wi].uuid,
            )
            new_pt = Point(new_end[0], new_end[1])
            spin = 180.0 if flipped else 0.0
            idx = stub.attach
            kind = stub.kind
            if kind == "gl":
                old = global_labels[idx]
                global_labels[idx] = GlobalLabel(
                    text=old.text, position=new_pt,
                    rotation=(old.rotation + spin) % 360.0,
                    shape=old.shape, effects=old.effects, uuid=old.uuid,
                )
            elif kind == "ll":
                old_l = local_labels[idx]
                local_labels[idx] = Label(
                    text=old_l.text, position=new_pt,
                    rotation=(old_l.rotation + spin) % 360.0,
                    effects=old_l.effects, uuid=old_l.uuid,
                )
            else:
                old_ps = power_symbols[idx]
                power_symbols[idx] = PowerSymbol(
                    lib_id=old_ps.lib_id, position=new_pt,
                    ref=old_ps.ref, value=old_ps.value,
                    rotation=(old_ps.rotation + spin) % 360.0,
                    uuid=old_ps.uuid,
                )
            stub.end = new_end
            log.info(
                "resolve_stub_collisions: %s stub at (%.2f, %.2f) -> "
                "%.2fmm%s", net, sx, sy, cand,
                " flipped" if flipped else "",
            )
            fixed = True
            break
        if not fixed:
            unresolved += 1
            log.warning(
                "resolve_stub_collisions: %s stub at (%.2f, %.2f) cannot be "
                "freed from foreign contact — written-file sync will fail",
                net, sx, sy,
            )
    return unresolved
