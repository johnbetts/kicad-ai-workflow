"""Stage 1 — cell generators: subcircuit layout by pin connectivity.

ONE generic engine lays out every subcircuit kind; specialization comes
from the constraint IR, not per-type code paths. Members are shelf-
packed along the side of the exact pad they attach to (a decoupling cap
goes beside its VDD pad, a flyback diode beside the coil pads), ordered
by pad coordinate, with deterministic rotation selection. Arrays are
one verified channel cell instantiated N times at fixed pitch.

A generator either returns a :class:`Cell` whose internal constraints
are all proven, or raises :class:`CellGenerationError` carrying the
quantified violations. There is no partially-correct output.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.optimization.geometry import (
    convex_hull,
    convex_polygons_overlap,
    inflate_convex_polygon,
)
from kicad_pipeline.placement_v2.cells import Cell, CellProof, PlacedMember, Port
from kicad_pipeline.placement_v2.footprint_geom import (
    courtyard_halfdims,
    courtyard_in_frame,
    pad_position_in_frame,
)
from kicad_pipeline.placement_v2.ir import (
    Axis,
    Severity,
    Violation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kicad_pipeline.models.pcb import Footprint, Point
    from kicad_pipeline.placement_v2.ir import (
        ConstraintSet,
        PadRef,
        PinAttach,
        SequenceAlong,
    )

_log = logging.getLogger(__name__)

_CLEARANCE_MM = 0.25
_PUSH_STEP_MM = 0.25
_MAX_PUSH_STEPS = 60
_CARDINAL_ROTATIONS = (0.0, 90.0, 180.0, 270.0)


class CellGenerationError(PCBError):
    """A subcircuit could not be laid out within its constraints."""

    def __init__(self, cell_name: str, violations: tuple[Violation, ...]) -> None:
        self.cell_name = cell_name
        self.violations = violations
        details = "; ".join(
            f"{v.refs}: {v.message} ({v.measured:.2f} > {v.limit:.2f})"
            for v in violations
        )
        super().__init__(f"cell {cell_name!r}: {details}")


def _rotated(px: float, py: float, deg: float) -> tuple[float, float]:
    """Rotate a local offset by the KiCad convention (negated angle)."""
    rad = math.radians(-deg)
    c, s = math.cos(rad), math.sin(rad)
    return (px * c - py * s, px * s + py * c)


def _side_normal(
    pad_x: float, pad_y: float, host_x: float, host_y: float,
    host_hw: float, host_hh: float,
) -> tuple[float, float]:
    """Outward unit normal of the host side the pad sits on (Y down)."""
    dx, dy = pad_x - host_x, pad_y - host_y
    nx = dx / host_hw if host_hw > 1e-9 else 0.0
    ny = dy / host_hh if host_hh > 1e-9 else 0.0
    if abs(nx) >= abs(ny):
        return (1.0, 0.0) if nx >= 0 else (-1.0, 0.0)
    return (0.0, 1.0) if ny >= 0 else (0.0, -1.0)


def _best_rotation(
    fp: Footprint, src_pin: str, normal: tuple[float, float],
) -> float:
    """Rotation putting the attaching pad on the member side facing the host.

    Minimizes the dot product of the rotated pad offset with the
    outward normal (the pad should point back toward the host pad).
    Deterministic tie-break: smallest angle.
    """
    from kicad_pipeline.placement_v2.footprint_geom import pad_offset_from_centroid

    px, py = pad_offset_from_centroid(fp, src_pin)
    best_rot = 0.0
    best_dot = math.inf
    for rot in _CARDINAL_ROTATIONS:
        rx, ry = _rotated(px, py, rot)
        dot = rx * normal[0] + ry * normal[1]
        if dot < best_dot - 1e-9:
            best_dot = dot
            best_rot = rot
    return best_rot


def _attach_position(
    fp: Footprint, src_pin: str, rotation: float,
    target_x: float, target_y: float, normal: tuple[float, float],
    ideal_mm: float,
) -> tuple[float, float]:
    """Member centroid such that its src pad sits *ideal_mm* outward of
    the target pad along *normal*."""
    from kicad_pipeline.placement_v2.footprint_geom import pad_offset_from_centroid

    px, py = pad_offset_from_centroid(fp, src_pin)
    rx, ry = _rotated(px, py, rotation)
    pad_target_x = target_x + normal[0] * ideal_mm
    pad_target_y = target_y + normal[1] * ideal_mm
    return (pad_target_x - rx, pad_target_y - ry)


def generate_cell(
    name: str,
    kind: str,
    anchor: str,
    footprints: Mapping[str, Footprint],
    constraints: ConstraintSet,
    external_nets: Mapping[str, tuple[PadRef, ...]] | None = None,
    clearance_mm: float = _CLEARANCE_MM,
) -> Cell:
    """Lay out one subcircuit and prove its internal constraints.

    *footprints* maps every member ref (anchor included) to its
    footprint. *constraints* should already be subset to this cell via
    ``ConstraintSet.for_refs``. *external_nets* maps each net leaving
    the cell to the internal pads that carry it (becomes ports).
    """
    refs = set(footprints)
    if anchor not in refs:
        raise PCBError(f"cell {name!r}: anchor {anchor!r} not in footprints")

    placed: dict[str, PlacedMember] = {
        anchor: PlacedMember(ref=anchor, x=0.0, y=0.0, rotation_deg=0.0)
    }

    seq_refs = _place_sequences(
        name, anchor, footprints, constraints.sequences, placed, clearance_mm
    )
    _place_attachments(
        footprints, constraints.pin_attach, placed, seq_refs, clearance_mm
    )
    _place_orphans(footprints, refs, placed, clearance_mm)
    _resolve_overlaps(footprints, placed, clearance_mm)

    violations = _verify_cell(footprints, constraints, placed, clearance_mm)
    checks = (
        f"pin_attach x{len(constraints.pin_attach)}",
        f"sequences x{len(constraints.sequences)}",
        "zero_courtyard_overlap",
    )
    if violations:
        raise CellGenerationError(name, violations)

    members = tuple(placed[r] for r in sorted(placed))
    corners: list[Point] = []
    for m in members:
        corners.extend(courtyard_in_frame(footprints[m.ref], m.x, m.y, m.rotation_deg))
    hull = inflate_convex_polygon(convex_hull(tuple(corners)), clearance_mm)

    ports = _build_ports(footprints, placed, external_nets or {})
    keepouts = tuple(k for k in constraints.keepouts if k.owner in refs)
    return Cell(
        name=name,
        kind=kind,
        members=members,
        polygon=hull,
        ports=ports,
        keepouts=keepouts,
        proof=CellProof(checks=checks),
    )


def _place_sequences(
    name: str,
    anchor: str,
    footprints: Mapping[str, Footprint],
    sequences: tuple[SequenceAlong, ...],
    placed: dict[str, PlacedMember],
    clearance_mm: float,
) -> frozenset[str]:
    """Place sequence chains as rows/columns in order. Returns refs placed."""
    done: set[str] = set()
    for seq in sequences:
        members = [r for r in seq.refs if r in footprints]
        if len(members) < 2:
            continue
        horizontal = seq.axis is Axis.HORIZONTAL
        # Row through the anchor when it participates, else one row below.
        if anchor in members:
            base_other = 0.0
        else:
            _, ahh = courtyard_halfdims(footprints[anchor])
            base_other = ahh + clearance_mm + max(
                courtyard_halfdims(footprints[r])[1] for r in members
            )
        cursor = 0.0
        for i, ref in enumerate(members):
            hw, hh = courtyard_halfdims(footprints[ref])
            extent = hw if horizontal else hh
            if seq.pitch_mm is not None:
                center = i * seq.pitch_mm
            else:
                center = cursor + extent
                cursor = center + extent + clearance_mm
            if ref == anchor:
                # Anchor stays at origin: shift the whole chain so this
                # member's slot lands on the origin.
                shift = -center
                for prev in members[:i]:
                    if prev in placed and prev != anchor:
                        m = placed[prev]
                        if horizontal:
                            placed[prev] = PlacedMember(m.ref, m.x + shift, m.y, m.rotation_deg)
                        else:
                            placed[prev] = PlacedMember(m.ref, m.x, m.y + shift, m.rotation_deg)
                cursor += shift
                continue
            x = center if horizontal else base_other
            y = base_other if horizontal else center
            placed[ref] = PlacedMember(ref=ref, x=x, y=y, rotation_deg=0.0)
            done.add(ref)
    if done:
        _log.debug("cell %s: %d members placed by sequence", name, len(done))
    return frozenset(done)


def _place_attachments(
    footprints: Mapping[str, Footprint],
    attachments: tuple[PinAttach, ...],
    placed: dict[str, PlacedMember],
    skip: frozenset[str],
    clearance_mm: float,
) -> None:
    """Shelf-pack attached members along the side of their target pad,
    in BFS waves out from already-placed hosts."""
    pending = [
        a for a in attachments
        if a.src.ref in footprints and a.src.ref not in skip
    ]
    for _wave in range(len(pending) + 1):
        ready = [
            a for a in pending
            if a.src.ref not in placed and a.dst.ref in placed
        ]
        if not ready:
            break
        # Group by (host, side) for shelf packing.
        groups: dict[tuple[str, tuple[float, float]], list[PinAttach]] = {}
        plan: dict[str, tuple[PinAttach, tuple[float, float], tuple[float, float]]] = {}
        for a in ready:
            if a.src.ref in plan:
                continue  # first attachment wins; rest are verified only
            host = placed[a.dst.ref]
            host_fp = footprints[a.dst.ref]
            hw, hh = courtyard_halfdims(host_fp)
            tx, ty = pad_position_in_frame(
                host_fp, a.dst.pin, host.x, host.y, host.rotation_deg
            )
            normal = _side_normal(tx, ty, host.x, host.y, hw, hh)
            plan[a.src.ref] = (a, (tx, ty), normal)
            groups.setdefault((a.dst.ref, normal), []).append(a)

        for (_host_ref, normal), group in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1])
        ):
            along_axis = (abs(normal[1]), abs(normal[0]))  # perpendicular
            group.sort(
                key=lambda a: (
                    plan[a.src.ref][1][0] * along_axis[0]
                    + plan[a.src.ref][1][1] * along_axis[1],
                    a.src.ref,
                )
            )
            cursor = -math.inf
            for a in group:
                _, (tx, ty), nrm = plan[a.src.ref]
                fp = footprints[a.src.ref]
                rot = _best_rotation(fp, a.src.pin, nrm)
                x, y = _attach_position(fp, a.src.pin, rot, tx, ty, nrm, a.ideal_mm)
                hw, hh = courtyard_halfdims(fp)
                extent = hw if along_axis[0] else hh
                desired = x * along_axis[0] + y * along_axis[1]
                pos_along = max(cursor + extent + clearance_mm, desired)
                shift = pos_along - desired
                x += shift * along_axis[0]
                y += shift * along_axis[1]
                cursor = pos_along + extent
                placed[a.src.ref] = PlacedMember(
                    ref=a.src.ref, x=x, y=y, rotation_deg=rot
                )


def _place_orphans(
    footprints: Mapping[str, Footprint],
    refs: set[str],
    placed: dict[str, PlacedMember],
    clearance_mm: float,
) -> None:
    """Members with no constraint path: deterministic shelf south of all."""
    orphans = sorted(refs - set(placed))
    if not orphans:
        return
    max_y = max(
        m.y + courtyard_halfdims(footprints[m.ref])[1] for m in placed.values()
    )
    cursor = -math.inf
    for ref in orphans:
        hw, hh = courtyard_halfdims(footprints[ref])
        x = max(cursor + hw + clearance_mm, 0.0)
        placed[ref] = PlacedMember(
            ref=ref, x=x, y=max_y + clearance_mm + hh, rotation_deg=0.0
        )
        cursor = x + hw


def _resolve_overlaps(
    footprints: Mapping[str, Footprint],
    placed: dict[str, PlacedMember],
    clearance_mm: float,
) -> None:
    """Push later members outward from the anchor until courtyards clear.

    Bounded and deterministic; residual overlaps surface as violations
    in :func:`_verify_cell` — never silently accepted.
    """
    order = sorted(placed)
    for _ in range(_MAX_PUSH_STEPS):
        moved = False
        for i, ref_a in enumerate(order):
            for ref_b in order[i + 1:]:
                a, b = placed[ref_a], placed[ref_b]
                pa = courtyard_in_frame(footprints[ref_a], a.x, a.y, a.rotation_deg)
                pb = courtyard_in_frame(footprints[ref_b], b.x, b.y, b.rotation_deg)
                if not convex_polygons_overlap(pa, pb, clearance_mm=clearance_mm):
                    continue
                dx, dy = b.x - a.x, b.y - a.y
                norm = math.hypot(dx, dy)
                if norm < 1e-9:
                    dx, dy, norm = 0.0, 1.0, 1.0
                placed[ref_b] = PlacedMember(
                    ref_b,
                    b.x + dx / norm * _PUSH_STEP_MM,
                    b.y + dy / norm * _PUSH_STEP_MM,
                    b.rotation_deg,
                )
                moved = True
        if not moved:
            return


def _verify_cell(
    footprints: Mapping[str, Footprint],
    constraints: ConstraintSet,
    placed: dict[str, PlacedMember],
    clearance_mm: float,
) -> tuple[Violation, ...]:
    """Re-derive every internal constraint from final member positions."""
    violations: list[Violation] = []

    for a in constraints.pin_attach:
        if a.src.ref not in placed or a.dst.ref not in placed:
            continue
        src, dst = placed[a.src.ref], placed[a.dst.ref]
        sx, sy = pad_position_in_frame(
            footprints[a.src.ref], a.src.pin, src.x, src.y, src.rotation_deg
        )
        dx, dy = pad_position_in_frame(
            footprints[a.dst.ref], a.dst.pin, dst.x, dst.y, dst.rotation_deg
        )
        dist = math.hypot(sx - dx, sy - dy)
        if dist > a.max_mm:
            violations.append(Violation(
                constraint=f"PinAttach({a.src}->{a.dst})",
                refs=(a.src.ref, a.dst.ref),
                severity=Severity.MAJOR,
                measured=dist,
                limit=a.max_mm,
                message=f"{a.src} is {dist:.2f}mm from {a.dst}",
            ))

    for seq in constraints.sequences:
        coords = [
            placed[r].x if seq.axis is Axis.HORIZONTAL else placed[r].y
            for r in seq.refs if r in placed
        ]
        if len(coords) >= 2 and any(
            later <= earlier + 1e-9 for earlier, later in itertools.pairwise(coords)
        ):
            violations.append(Violation(
                constraint=f"SequenceAlong({seq.refs})",
                refs=seq.refs,
                severity=Severity.MAJOR,
                measured=0.0,
                limit=0.0,
                message=f"refs not monotonic along {seq.axis.value}",
            ))

    order = sorted(placed)
    for i, ref_a in enumerate(order):
        for ref_b in order[i + 1:]:
            ma, mb = placed[ref_a], placed[ref_b]
            pa = courtyard_in_frame(footprints[ref_a], ma.x, ma.y, ma.rotation_deg)
            pb = courtyard_in_frame(footprints[ref_b], mb.x, mb.y, mb.rotation_deg)
            if convex_polygons_overlap(pa, pb):
                violations.append(Violation(
                    constraint="courtyard_overlap",
                    refs=(ref_a, ref_b),
                    severity=Severity.CRITICAL,
                    measured=1.0,
                    limit=0.0,
                    message=f"{ref_a} and {ref_b} courtyards overlap",
                ))

    return tuple(violations)


def _build_ports(
    footprints: Mapping[str, Footprint],
    placed: dict[str, PlacedMember],
    external_nets: Mapping[str, tuple[PadRef, ...]],
) -> tuple[Port, ...]:
    ports: list[Port] = []
    for net in sorted(external_nets):
        for pr in external_nets[net]:
            m = placed.get(pr.ref)
            if m is None:
                continue
            try:
                x, y = pad_position_in_frame(
                    footprints[pr.ref], pr.pin, m.x, m.y, m.rotation_deg
                )
            except KeyError:
                continue
            ports.append(Port(net=net, x=x, y=y))
            break
    return tuple(ports)
