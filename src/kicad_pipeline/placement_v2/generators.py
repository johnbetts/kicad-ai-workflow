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
from dataclasses import dataclass
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
    pad_offset_from_centroid,
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
    interface_nets: frozenset[str] = frozenset(),
) -> Cell:
    """Lay out one subcircuit and prove its internal constraints.

    *footprints* maps every member ref (anchor included) to its
    footprint. *constraints* should already be subset to this cell via
    ``ConstraintSet.for_refs``. *external_nets* maps each net leaving
    the cell to the internal pads that carry it (becomes ports).
    *interface_nets* names the connector-facing subset of those nets
    (they reach an edge-pinned ref); on THT anchors the support halo
    stacks on the opposite side of those pads.
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
    anchor_normal = _anti_interface_normal(
        footprints[anchor],
        {
            net: pads for net, pads in (external_nets or {}).items()
            if net in interface_nets
        },
    )
    normals = _place_attachments(
        footprints, constraints.pin_attach, placed, seq_refs, clearance_mm,
        anchor_normal_override={anchor: anchor_normal} if anchor_normal else None,
    )
    _place_orphans(footprints, refs, placed, clearance_mm)
    _resolve_overlaps(footprints, placed, clearance_mm, normals)

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


def _anti_interface_normal(
    anchor_fp: Footprint,
    external_nets: Mapping[str, tuple[PadRef, ...]],
) -> tuple[float, float] | None:
    """For a THT anchor: the direction AWAY from its interface pads.

    Through-hole anchors (relays, transformers, bulky connectors) talk
    to their connectors through one end — their support halo belongs on
    the opposite end, in a column along the body axis (the reference
    board's driver-band doctrine). *external_nets* here must already be
    restricted to the connector-facing interface nets: power rails span
    the whole board and carry no direction. Returns the unit direction
    opposite the mean interface pad offset, snapped to a cardinal axis;
    ``None`` for SMD anchors or when there is no clear interface side
    (supports then follow per-pad geometry as usual).
    """
    if anchor_fp.attr != "through_hole":
        return None
    offsets: list[tuple[float, float]] = []
    for pads in external_nets.values():
        for pr in pads:
            if pr.ref != anchor_fp.ref:
                continue
            try:
                offsets.append(pad_offset_from_centroid(anchor_fp, pr.pin))
            except KeyError:
                continue
    if not offsets:
        return None
    mx = sum(o[0] for o in offsets) / len(offsets)
    my = sum(o[1] for o in offsets) / len(offsets)
    if math.hypot(mx, my) < 1.0:
        return None  # interface pads surround the body: no clear side
    if abs(mx) >= abs(my):
        return (-1.0, 0.0) if mx > 0 else (1.0, 0.0)
    return (0.0, -1.0) if my > 0 else (0.0, 1.0)


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


@dataclass
class _BandMember:
    """A chain-resolved attachment awaiting band placement."""

    ref: str
    host: str
    dst_pin: str
    depth: int  # chain hops from the band root


def _resolve_chains(
    footprints: Mapping[str, Footprint],
    attachments: tuple[PinAttach, ...],
    placed: dict[str, PlacedMember],
    skip: frozenset[str],
    anchor_normal_override: dict[str, tuple[float, float]] | None,
) -> dict[tuple[str, tuple[float, float]], list[_BandMember]]:
    """Group attachments into bands keyed by (root host, side normal).

    Chain members (LED hanging off a resistor hanging off a transistor)
    inherit their root's band so a whole signal chain lays out as one
    organized block on one side of its root.
    """
    bands: dict[tuple[str, tuple[float, float]], list[_BandMember]] = {}
    member_band: dict[str, tuple[tuple[str, tuple[float, float]], int]] = {}
    pending = [
        a for a in attachments
        if a.src.ref in footprints and a.src.ref not in skip
        and a.src.ref not in placed
    ]
    for _ in range(len(pending) + 1):
        progressed = False
        for a in sorted(pending, key=lambda a: (a.dst.ref, a.src.ref)):
            if a.src.ref in member_band:
                continue  # first attachment wins; rest are verified only
            if a.dst.ref in placed:
                host = placed[a.dst.ref]
                host_fp = footprints[a.dst.ref]
                normal = (
                    (anchor_normal_override or {}).get(a.dst.ref)
                )
                if normal is None:
                    hw, hh = courtyard_halfdims(host_fp)
                    tx, ty = pad_position_in_frame(
                        host_fp, a.dst.pin, host.x, host.y, host.rotation_deg
                    )
                    normal = _side_normal(tx, ty, host.x, host.y, hw, hh)
                key = (a.dst.ref, normal)
                depth = 1
            elif a.dst.ref in member_band:
                key, host_depth = member_band[a.dst.ref]
                depth = host_depth + 1
            else:
                continue
            bands.setdefault(key, []).append(
                _BandMember(ref=a.src.ref, host=a.dst.ref,
                            dst_pin=a.dst.pin, depth=depth)
            )
            member_band[a.src.ref] = (key, depth)
            progressed = True
        if not progressed:
            break
    return bands


def _grid_rotation(
    fp: Footprint, src_pin: str,
    center: tuple[float, float], target: tuple[float, float],
) -> float:
    """Rotation minimizing the member's src-pad distance to its target."""
    best_rot, best_d = 0.0, math.inf
    for rot in _CARDINAL_ROTATIONS:
        sx, sy = pad_position_in_frame(fp, src_pin, center[0], center[1], rot)
        d = math.hypot(sx - target[0], sy - target[1])
        if d < best_d - 1e-9:
            best_d, best_rot = d, rot
    return best_rot


def _place_attachments(
    footprints: Mapping[str, Footprint],
    attachments: tuple[PinAttach, ...],
    placed: dict[str, PlacedMember],
    skip: frozenset[str],
    clearance_mm: float,
    normals: dict[str, tuple[float, float]] | None = None,
    anchor_normal_override: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Lay each host side's halo out as a GRID BAND of rows.

    Members fill rows no wider than their root host (the reference
    board's driver band: flyback+transistor row, then resistor row,
    then LED row), each row one chain-depth further out. Within a row
    members sit at their target pad's coordinate, and each picks the
    rotation (horizontal/vertical) that minimizes its actual pad-to-pad
    distance — not a fixed orientation. Returns the outward normal per
    placed ref (used by overlap resolution to push deeper, not
    sideways).
    """
    if normals is None:
        normals = {}
    bands = _resolve_chains(
        footprints, attachments, placed, skip, anchor_normal_override,
    )
    for (root, normal), members in sorted(
        bands.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        root_m = placed[root]
        rb = courtyard_in_frame(
            footprints[root], root_m.x, root_m.y, root_m.rotation_deg
        )
        rxs = [p.x for p in rb]
        rys = [p.y for p in rb]
        horizontal_band = normal[1] != 0  # rows run along x
        if horizontal_band:
            budget_lo, budget_hi = min(rxs) - clearance_mm, max(rxs) + clearance_mm
            depth_edge = max(rys) if normal[1] > 0 else min(rys)
        else:
            budget_lo, budget_hi = min(rys) - clearance_mm, max(rys) + clearance_mm
            depth_edge = max(rxs) if normal[0] > 0 else min(rxs)
        depth_sign = normal[1] if horizontal_band else normal[0]
        depth_cursor = depth_edge

        # Levels (chain depth) -> rows. Rows are laid out as WHOLE
        # units centered near their members' mean target coordinate, so
        # same-level members sit SIDE BY SIDE in a band row (the
        # reference driver band) instead of each wrapping into its own
        # row when they covet the same pad.
        by_level: dict[int, list[_BandMember]] = {}
        for m in members:
            by_level.setdefault(m.depth, []).append(m)

        for level in sorted(by_level):
            level_members = sorted(by_level[level], key=lambda m: m.ref)
            # Targets + tentative rotation/extents per member.
            infos: list[tuple[_BandMember, float, float, float, float, float, float]] = []
            for m in level_members:
                fp = footprints[m.ref]
                host_m = placed[m.host]
                tx, ty = pad_position_in_frame(
                    footprints[m.host], m.dst_pin,
                    host_m.x, host_m.y, host_m.rotation_deg,
                )
                desired = tx if horizontal_band else ty
                hw0, hh0 = courtyard_halfdims(fp)
                tentative_depth = depth_cursor + depth_sign * (
                    hh0 if horizontal_band else hw0
                )
                center = (
                    (desired, tentative_depth)
                    if horizontal_band else
                    (tentative_depth, desired)
                )
                rot = _grid_rotation(
                    fp, _src_pin_of(attachments, m.ref), center, (tx, ty),
                )
                hw_r, hh_r = (hw0, hh0) if rot % 180 == 0 else (hh0, hw0)
                half_along = hw_r if horizontal_band else hh_r
                half_depth = hh_r if horizontal_band else hw_r
                infos.append((m, desired, rot, half_along, half_depth, tx, ty))
            infos.sort(key=lambda i: (i[1], i[0].ref))

            # Split into rows that fit the budget, then center each row
            # near its members' mean desired coordinate.
            rows: list[list[tuple[_BandMember, float, float, float, float, float, float]]] = [[]]
            width = 0.0
            budget_span = budget_hi - budget_lo
            for info in infos:
                w = 2 * info[3] + (clearance_mm if rows[-1] else 0.0)
                if rows[-1] and width + w > budget_span:
                    rows.append([])
                    width = 0.0
                    w = 2 * info[3]
                rows[-1].append(info)
                width += w

            for row in rows:
                if not row:
                    continue
                total = sum(2 * i[3] for i in row) + clearance_mm * (len(row) - 1)
                mean_desired = sum(i[1] for i in row) / len(row)
                start = min(
                    max(mean_desired - total / 2, budget_lo),
                    max(budget_hi - total, budget_lo),
                )
                row_extent = max(2 * i[4] for i in row)
                cursor = start
                for m, _desired, rot, half_along, _half_depth, _tx, _ty in row:
                    along = cursor + half_along
                    depth = depth_cursor + depth_sign * (row_extent / 2)
                    x, y = (along, depth) if horizontal_band else (depth, along)
                    placed[m.ref] = PlacedMember(
                        ref=m.ref, x=x, y=y, rotation_deg=rot,
                    )
                    normals[m.ref] = normal
                    cursor = along + half_along + clearance_mm
                depth_cursor += depth_sign * (row_extent + clearance_mm)
    return normals


def _src_pin_of(attachments: tuple[PinAttach, ...], ref: str) -> str:
    """The src pin of *ref*'s first (placement-driving) attachment."""
    for a in attachments:
        if a.src.ref == ref:
            return a.src.pin
    return "1"


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
    normals: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Push later members outward until courtyards clear.

    A member with a recorded attachment normal is pushed along THAT
    direction (deeper into its chain), so same-pad attachments stack as
    a column — flyback, then transistor, then LED/resistor — instead of
    spreading sideways. Members without a normal fall back to the
    centroid-difference direction. Bounded and deterministic; residual
    overlaps surface as violations in :func:`_verify_cell`.
    """
    normals = normals or {}

    def _depth(ref: str) -> float:
        """How far out a member sits along its own attachment direction."""
        n = normals.get(ref)
        if n is None:
            return -math.inf  # anchors/sequence members are never pushed first
        m = placed[ref]
        return m.x * n[0] + m.y * n[1]

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
                # Push the DEEPER chain member further out, never the
                # host it hangs from (which would drag the host past
                # its own attachment bound).
                pushee = ref_b if _depth(ref_b) >= _depth(ref_a) else ref_a
                target = placed[pushee]
                push = normals.get(pushee)
                if push is None:
                    other = placed[ref_a if pushee == ref_b else ref_b]
                    dx, dy = target.x - other.x, target.y - other.y
                    norm = math.hypot(dx, dy)
                    if norm < 1e-9:
                        dx, dy, norm = 0.0, 1.0, 1.0
                    push = (dx / norm, dy / norm)
                placed[pushee] = PlacedMember(
                    pushee,
                    target.x + push[0] * _PUSH_STEP_MM,
                    target.y + push[1] * _PUSH_STEP_MM,
                    target.rotation_deg,
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
