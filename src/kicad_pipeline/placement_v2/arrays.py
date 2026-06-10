"""Array instancing — one verified channel cell stamped N times.

Relay banks, ADC channels, and buck phases are arrays of an identical
subcircuit. v2 lays out ONE channel, proves it, then instantiates it at
fixed pitch — channels are identical and ordered by construction, which
is the structural answer to "repeatable patterns" and "arrays in
sequence" requirements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.models.pcb import Point
from kicad_pipeline.optimization.geometry import convex_hull
from kicad_pipeline.placement_v2.cells import Cell, CellProof, PlacedMember, Port
from kicad_pipeline.placement_v2.ir import Axis

if TYPE_CHECKING:
    from collections.abc import Mapping


def instantiate_array(
    name: str,
    kind: str,
    channel: Cell,
    count: int,
    axis: Axis = Axis.HORIZONTAL,
    pitch_mm: float | None = None,
    ref_maps: tuple[Mapping[str, str], ...] = (),
    gap_mm: float = 1.0,
) -> Cell:
    """One verified channel cell instantiated *count* times at fixed pitch.

    Channels are identical by construction — this is what makes relay /
    ADC / buck arrays repeatable. *ref_maps[i]* renames the template's
    refs for instance *i* (e.g. ``{"K1": "K3", "Q1": "Q3"}``); when
    omitted, instance refs get an ``_chN`` suffix (tests only).
    """
    if count < 1:
        raise PCBError(f"array {name!r}: count must be >= 1")
    if ref_maps and len(ref_maps) != count:
        raise PCBError(f"array {name!r}: need {count} ref maps, got {len(ref_maps)}")

    horizontal = axis is Axis.HORIZONTAL
    span = channel.width if horizontal else channel.height
    pitch = pitch_mm if pitch_mm is not None else span + gap_mm

    members: list[PlacedMember] = []
    ports: list[Port] = []
    corners: list[Point] = []
    for i in range(count):
        dx = pitch * i if horizontal else 0.0
        dy = 0.0 if horizontal else pitch * i
        rename: Mapping[str, str] = ref_maps[i] if ref_maps else {}
        for m in channel.members:
            ref = rename.get(m.ref, f"{m.ref}_ch{i + 1}" if not ref_maps else m.ref)
            members.append(PlacedMember(ref, m.x + dx, m.y + dy, m.rotation_deg))
        for p in channel.ports:
            ports.append(Port(net=p.net, x=p.x + dx, y=p.y + dy))
        corners.extend(
            Point(pt.x + dx, pt.y + dy) for pt in channel.polygon
        )

    seen: set[str] = set()
    for m in members:
        if m.ref in seen:
            raise PCBError(f"array {name!r}: duplicate ref {m.ref!r} across channels")
        seen.add(m.ref)

    return Cell(
        name=name,
        kind=kind,
        members=tuple(members),
        polygon=convex_hull(tuple(corners)),
        ports=tuple(ports),
        keepouts=channel.keepouts,
        proof=CellProof(checks=(f"array x{count} pitch {pitch:.2f}mm",)),
    )
