"""Immutable placement cells — the bottom-up rigid bodies of v2.

A :class:`Cell` is a fully laid-out subcircuit in its own local frame:
member positions, a bounding polygon, ports (where external nets exit),
and keepouts. After its internal constraints are proven (recorded in
:class:`CellProof`), a cell is immutable — downstream stages hold only a
:class:`PlacedCell` (cell + translation + 90-degree rotation) and there
is no API to move an individual member. This makes "frozen actually
frozen" structural rather than a flag downstream phases must respect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.optimization.geometry import (
    polygon_bbox,
    transform_polygon,
)
from kicad_pipeline.placement_v2.ir import CellKeepout, Polygon, Violation

CardinalRotation = Literal[0, 90, 180, 270]


@dataclass(frozen=True)
class PlacedMember:
    """One component inside a cell: centroid position + rotation.

    Coordinates are CELL-LOCAL (anchor at origin) until the owning cell
    is placed; :meth:`PlacedCell.members_in_board` produces board-frame
    positions.
    """

    ref: str
    x: float
    y: float
    rotation_deg: float


@dataclass(frozen=True)
class Port:
    """Where an external net exits the cell hull (cell-local frame).

    The floorplanner's ratsnest objective connects ports, not member
    centroids — a far better proxy for routability.
    """

    net: str
    x: float
    y: float


@dataclass(frozen=True)
class CellProof:
    """Record of the internal verification a cell passed at generation.

    A cell with a non-empty ``violations`` tuple must never leave its
    generator; constructors raise instead. ``checks`` lists what was
    verified so the build ledger can show coverage, not just a verdict.
    """

    checks: tuple[str, ...]
    violations: tuple[Violation, ...] = ()

    @property
    def ok(self) -> bool:
        """True when every internal check passed."""
        return not self.violations


@dataclass(frozen=True)
class Cell:
    """A proven, rigid subcircuit layout in its local frame."""

    name: str  # "relay_ch2", "buck_5v"
    kind: str  # SubCircuitType value, e.g. "relay_driver"
    members: tuple[PlacedMember, ...]
    polygon: Polygon  # hull + clearance margin, local frame
    ports: tuple[Port, ...]
    keepouts: tuple[CellKeepout, ...] = ()
    proof: CellProof = field(default_factory=lambda: CellProof(checks=()))

    def __post_init__(self) -> None:
        if not self.proof.ok:
            msgs = "; ".join(v.message for v in self.proof.violations)
            raise ValueError(
                f"Cell {self.name!r} constructed with failing proof: {msgs}"
            )

    @property
    def refs(self) -> frozenset[str]:
        """Refs of all member components."""
        return frozenset(m.ref for m in self.members)

    @property
    def width(self) -> float:
        """Bounding-box width of the cell polygon."""
        x1, _, x2, _ = polygon_bbox(self.polygon)
        return x2 - x1

    @property
    def height(self) -> float:
        """Bounding-box height of the cell polygon."""
        _, y1, _, y2 = polygon_bbox(self.polygon)
        return y2 - y1

    @property
    def area(self) -> float:
        """Bounding-box area (used for packing order)."""
        return self.width * self.height


@dataclass(frozen=True)
class PlacedCell:
    """A cell with a rigid-body transform: rotate about the cell origin,
    then translate. The ONLY mutation downstream stages may express is
    replacing the whole transform — members are untouchable.
    """

    cell: Cell
    dx: float
    dy: float
    rotation: CardinalRotation = 0

    def moved_to(self, dx: float, dy: float) -> PlacedCell:
        """New placement at a different position, same rotation."""
        return PlacedCell(self.cell, dx, dy, self.rotation)

    def rotated(self, rotation: CardinalRotation) -> PlacedCell:
        """New placement with a different rotation, same position."""
        return PlacedCell(self.cell, self.dx, self.dy, rotation)

    def polygon_in_board(self) -> Polygon:
        """Cell hull in board frame."""
        return transform_polygon(
            self.cell.polygon, self.dx, self.dy, float(self.rotation)
        )

    def members_in_board(self) -> tuple[PlacedMember, ...]:
        """Member centroids and rotations in board frame."""
        pts = transform_polygon(
            tuple(Point(m.x, m.y) for m in self.cell.members),
            self.dx,
            self.dy,
            float(self.rotation),
        )
        return tuple(
            PlacedMember(
                ref=m.ref,
                x=p.x,
                y=p.y,
                rotation_deg=(m.rotation_deg + self.rotation) % 360.0,
            )
            for m, p in zip(self.cell.members, pts, strict=True)
        )

    def ports_in_board(self) -> tuple[Port, ...]:
        """Ports in board frame."""
        pts = transform_polygon(
            tuple(Point(p.x, p.y) for p in self.cell.ports),
            self.dx,
            self.dy,
            float(self.rotation),
        )
        return tuple(
            Port(net=port.net, x=p.x, y=p.y)
            for port, p in zip(self.cell.ports, pts, strict=True)
        )

    def keepouts_in_board(self) -> tuple[CellKeepout, ...]:
        """Keepout polygons in board frame (owner refs unchanged).

        Keepout polygons are stored relative to the OWNER's local
        position, which itself is cell-local; the full transform is
        owner offset + cell transform.
        """
        result: list[CellKeepout] = []
        member_by_ref = {m.ref: m for m in self.cell.members}
        for ko in self.cell.keepouts:
            owner = member_by_ref.get(ko.owner)
            if owner is None:
                continue
            local = transform_polygon(
                ko.polygon, owner.x, owner.y, owner.rotation_deg
            )
            board = transform_polygon(
                local, self.dx, self.dy, float(self.rotation)
            )
            result.append(
                CellKeepout(owner=ko.owner, polygon=board, kind=ko.kind,
                            source=ko.source)
            )
        return tuple(result)
