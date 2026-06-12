"""Gate A check rules — each re-derives one constraint class from the artifact.

Every function here consumes only the :class:`~kicad_pipeline.models.pcb.PCBDesign`
artifact and the Constraint IR objects the solvers used; nothing is read from
solver state. All board-space math uses the KiCad rotation convention from
:mod:`kicad_pipeline.pcb.pin_map` (positive angle negated before the standard
CCW matrix — see :func:`pad_extent_in_board_space`), NOT the cell-space
``transform_polygon`` convention, because the verifier checks the artifact as
KiCad will interpret it.

Courtyards are derived via :func:`placement_v2.footprint_geom.courtyard_polygon`
(centroid-relative) and converted to board space through
:func:`pin_map.origin_to_centroid` — correctness over reuse: the cell-space
``courtyard_in_frame`` uses the opposite rotation sign and is not used here.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.optimization.geometry import point_in_polygon, polygon_bbox
from kicad_pipeline.pcb.pin_map import origin_to_centroid
from kicad_pipeline.placement_v2.footprint_geom import courtyard_polygon, pad_by_number
from kicad_pipeline.placement_v2.ir import Axis, Edge, Severity, Violation

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import BoardOutline, Footprint, Pad, PCBDesign
    from kicad_pipeline.placement_v2.ir import (
        AttachBundle,
        BoardContain,
        CellKeepout,
        ConnectorFanout,
        EdgePin,
        GroupAssoc,
        IsolationGap,
        PinAttach,
        Polygon,
        SequenceAlong,
    )

#: Tolerance when checking a SequenceAlong's fixed pitch.
PITCH_TOL_MM = 0.1
#: Tolerance when matching a derived keepout bbox to a board keepout zone.
KEEPOUT_BBOX_TOL_MM = 1.0

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Board-space geometry (KiCad rotation convention)
# ---------------------------------------------------------------------------


def _rotate_kicad(px: float, py: float, rotation_deg: float) -> tuple[float, float]:
    """Rotate a footprint-local point by the KiCad rotation convention.

    Matches :func:`kicad_pipeline.pcb.pin_map.pad_extent_in_board_space`:
    the positive KiCad angle is negated, then the standard CCW rotation
    matrix is applied (KiCad rotates CCW on screen with Y down).
    """
    rad = math.radians(-rotation_deg)
    c, s = math.cos(rad), math.sin(rad)
    return (px * c - py * s, px * s + py * c)


def _pad_center(fp: Footprint, pad: Pad) -> tuple[float, float]:
    """Pad center in board space: footprint origin + rotated local offset."""
    rx, ry = _rotate_kicad(pad.position.x, pad.position.y, fp.rotation)
    return (fp.position.x + rx, fp.position.y + ry)


def _pad_corners(fp: Footprint, pad: Pad) -> Polygon:
    """The pad rectangle's four corners in board space (rotation-aware)."""
    hx, hy = pad.size_x / 2.0, pad.size_y / 2.0
    out: list[Point] = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        rx, ry = _rotate_kicad(
            pad.position.x + sx * hx, pad.position.y + sy * hy, fp.rotation,
        )
        out.append(Point(fp.position.x + rx, fp.position.y + ry))
    return tuple(out)


def _local_to_board(fp: Footprint, polygon: Polygon) -> Polygon:
    """Centroid-local polygon -> board space via the owner's actual placement."""
    cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
    pts: list[Point] = []
    for p in polygon:
        rx, ry = _rotate_kicad(p.x, p.y, fp.rotation)
        pts.append(Point(cx + rx, cy + ry))
    return tuple(pts)


def _courtyard_in_board(fp: Footprint) -> Polygon:
    """Footprint courtyard polygon (centroid-relative, unrotated) in board space."""
    return _local_to_board(fp, courtyard_polygon(fp))


def _outline_points(outline: BoardOutline) -> Polygon:
    """Outline polygon, expanding the 2-point rect shorthand if present."""
    poly = outline.polygon
    if len(poly) == 2:
        a, b = poly
        return (Point(a.x, a.y), Point(b.x, a.y), Point(b.x, b.y), Point(a.x, b.y))
    return poly


# ---------------------------------------------------------------------------
# Polygon overlap / distance (convex polygons — courtyards are rectangles)
# ---------------------------------------------------------------------------


def _project(poly: Polygon, ax: float, ay: float) -> tuple[float, float]:
    dots = [p.x * ax + p.y * ay for p in poly]
    return (min(dots), max(dots))


def _overlap_depth(a: Polygon, b: Polygon) -> float:
    """Penetration depth of two convex polygons; 0.0 when separated (SAT)."""
    depth = math.inf
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            ax, ay = -(p2.y - p1.y), p2.x - p1.x
            length = math.hypot(ax, ay)
            if length < _EPS:
                continue
            a_min, a_max = _project(a, ax / length, ay / length)
            b_min, b_max = _project(b, ax / length, ay / length)
            overlap = min(a_max, b_max) - max(a_min, b_min)
            if overlap <= 0.0:
                return 0.0
            depth = min(depth, overlap)
    return 0.0 if math.isinf(depth) else depth


def _pt_seg_dist(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float,
) -> float:
    """Distance from a point to a line segment."""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < _EPS:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _boundary_dist(px: float, py: float, poly: Polygon) -> float:
    """Distance from a point to the polygon boundary (any winding)."""
    n = len(poly)
    dists = [
        _pt_seg_dist(px, py, poly[i].x, poly[i].y, poly[(i + 1) % n].x, poly[(i + 1) % n].y)
        for i in range(n)
    ]
    return min(dists) if dists else 0.0


def _polygon_gap(a: Polygon, b: Polygon) -> float:
    """Minimum edge-to-edge gap between two convex polygons (0 if touching)."""
    if _overlap_depth(a, b) > 0.0:
        return 0.0
    gaps = [_boundary_dist(p.x, p.y, dst) for src, dst in ((a, b), (b, a)) for p in src]
    return min(gaps) if gaps else 0.0


def _missing(constraint: object, ref: str, what: str) -> Violation:
    return Violation(
        repr(constraint), (ref,), Severity.CRITICAL, 0.0, 0.0,
        f"constraint references {what} but it is not on the board",
    )


# ---------------------------------------------------------------------------
# Checks — one function per constraint class
# ---------------------------------------------------------------------------


def check_pin_attach(
    pcb: PCBDesign, attaches: tuple[PinAttach, ...],
) -> tuple[Violation, ...]:
    """Each PinAttach: euclidean src->dst pad-center distance <= max_mm (MAJOR)."""
    out: list[Violation] = []
    for attach in attaches:
        centers: list[tuple[float, float]] = []
        for pad_ref in (attach.src, attach.dst):
            fp = pcb.get_footprint(pad_ref.ref)
            if fp is None:
                out.append(_missing(attach, pad_ref.ref, f"footprint {pad_ref.ref!r}"))
                continue
            pad = pad_by_number(fp, pad_ref.pin)
            if pad is None:
                out.append(_missing(attach, pad_ref.ref, f"pad {pad_ref}"))
                continue
            centers.append(_pad_center(fp, pad))
        if len(centers) < 2:
            continue
        (sx, sy), (dx, dy) = centers
        dist = math.hypot(dx - sx, dy - sy)
        if dist > attach.max_mm:
            out.append(Violation(
                repr(attach), (attach.src.ref, attach.dst.ref), Severity.MAJOR,
                dist, attach.max_mm,
                f"{attach.src} is {dist:.3f}mm from {attach.dst} "
                f"(max {attach.max_mm}mm, net {attach.net})",
            ))
    return tuple(out)


def check_sequences(
    pcb: PCBDesign, sequences: tuple[SequenceAlong, ...],
) -> tuple[Violation, ...]:
    """Each SequenceAlong: pad-centroid coordinate strictly monotonic (MAJOR).

    Monotonic in EITHER direction is accepted ("in order along axis" is
    direction-agnostic, per the IR docstring's "strictly monotonic").
    Also checks ``pitch_mm`` (+-0.1mm) and ``max_span_mm`` when set.
    """
    out: list[Violation] = []
    for seq in sequences:
        coords: list[float] = []
        missing = False
        for ref in seq.refs:
            fp = pcb.get_footprint(ref)
            if fp is None:
                out.append(_missing(seq, ref, f"footprint {ref!r}"))
                missing = True
                continue
            cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
            coords.append(cx if seq.axis is Axis.HORIZONTAL else cy)
        if missing or len(coords) < 2:
            continue
        deltas = [b - a for a, b in pairwise(coords)]
        increasing = all(d > _EPS for d in deltas)
        decreasing = all(d < -_EPS for d in deltas)
        if not (increasing or decreasing):
            out.append(Violation(
                repr(seq), seq.refs, Severity.MAJOR,
                min(abs(d) for d in deltas), 0.0,
                f"refs {seq.refs} are not in order along {seq.axis.value}: "
                f"coordinates {tuple(round(c, 3) for c in coords)}",
            ))
        if seq.pitch_mm is not None:
            worst = max(abs(abs(d) - seq.pitch_mm) for d in deltas)
            if worst > PITCH_TOL_MM:
                out.append(Violation(
                    repr(seq), seq.refs, Severity.MAJOR, worst, PITCH_TOL_MM,
                    f"refs {seq.refs} deviate {worst:.3f}mm from pitch {seq.pitch_mm}mm",
                ))
        if seq.max_span_mm is not None:
            span = max(coords) - min(coords)
            if span > seq.max_span_mm:
                out.append(Violation(
                    repr(seq), seq.refs, Severity.MAJOR, span, seq.max_span_mm,
                    f"refs {seq.refs} span {span:.3f}mm > {seq.max_span_mm}mm",
                ))
    return tuple(out)


def _edge_distance(
    fp_bbox: tuple[float, float, float, float],
    board_bbox: tuple[float, float, float, float],
    edge: Edge,
) -> float:
    fx1, fy1, fx2, fy2 = fp_bbox
    bx1, by1, bx2, by2 = board_bbox
    if edge is Edge.WEST:
        return fx1 - bx1
    if edge is Edge.EAST:
        return bx2 - fx2
    if edge is Edge.NORTH:
        return fy1 - by1
    return by2 - fy2


def check_edge_pins(
    pcb: PCBDesign, edge_pins: tuple[EdgePin, ...],
) -> tuple[Violation, ...]:
    """Each EdgePin: courtyard within max_edge_distance_mm of the edge (MAJOR).

    Distance is measured from the COURTYARD (body) extent, not pad
    centers: "connector at the edge" means the housing is flush so the
    cable can exit — terminal blocks carry pads several mm inside the
    body and would never satisfy a pad-center bound. Only the DISTANCE
    part of the constraint is checked here; the ``face_out``
    orientation check is not implemented yet (recorded as a skipped
    check in the Gate A report so coverage stays honest).
    """
    board_bbox = polygon_bbox(_outline_points(pcb.outline))
    out: list[Violation] = []
    for pin in edge_pins:
        fp = pcb.get_footprint(pin.ref)
        if fp is None:
            out.append(_missing(pin, pin.ref, f"footprint {pin.ref!r}"))
            continue
        fp_bbox = polygon_bbox(_courtyard_in_board(fp))
        if pin.edge is not None:
            dist, edge_name = _edge_distance(fp_bbox, board_bbox, pin.edge), pin.edge.value
        else:
            dist, edge_name = min(
                ((_edge_distance(fp_bbox, board_bbox, e), e.value) for e in Edge),
                key=lambda item: item[0],
            )
        if dist > pin.max_edge_distance_mm:
            out.append(Violation(
                repr(pin), (pin.ref,), Severity.MAJOR, dist, pin.max_edge_distance_mm,
                f"{pin.ref} is {dist:.3f}mm from the {edge_name} edge "
                f"(max {pin.max_edge_distance_mm}mm)",
            ))
        if pin.face_out:
            v = _face_out_violation(fp, pin, Edge(edge_name), board_bbox)
            if v is not None:
                out.append(v)
        out.extend(_forefield_violations(pcb, fp, pin, Edge(edge_name), board_bbox))
    return tuple(out)


def _forefield_violations(
    pcb: PCBDesign, fp: Footprint, pin: EdgePin, edge: Edge,
    board_bbox: tuple[float, float, float, float],
) -> list[Violation]:
    """Nothing may sit between an edge connector and its board edge.

    A passive parked in front of a screw terminal blocks the wire
    opening and the screwdriver — found by human review on the analog
    board and converted to this deterministic rule.
    """
    band = polygon_bbox(_courtyard_in_board(fp))
    bx1, by1, bx2, by2 = board_bbox
    if edge is Edge.SOUTH:
        zone = (band[0], band[3], band[2], by2)
    elif edge is Edge.NORTH:
        zone = (band[0], by1, band[2], band[1])
    elif edge is Edge.EAST:
        zone = (band[2], band[1], bx2, band[3])
    else:
        zone = (bx1, band[1], band[0], band[3])
    out: list[Violation] = []
    for other in pcb.footprints:
        if other.ref == fp.ref or other.ref.startswith("H"):
            continue
        ob = polygon_bbox(_courtyard_in_board(other))
        ow = min(zone[2], ob[2]) - max(zone[0], ob[0])
        oh = min(zone[3], ob[3]) - max(zone[1], ob[1])
        if ow > 0.1 and oh > 0.1:
            out.append(Violation(
                f"forefield({pin.ref})", (pin.ref, other.ref),
                Severity.MAJOR, min(ow, oh), 0.0,
                f"{other.ref} sits between {pin.ref} and the "
                f"{edge.value} board edge",
            ))
    return out


#: Body-bulge magnitude below which a connector's orientation cannot be
#: determined from geometry (symmetric parts like pin headers).
_FACE_OUT_MIN_BULGE_MM = 0.5


def _face_out_violation(
    fp: Footprint, pin: EdgePin, edge: Edge,
    board_bbox: tuple[float, float, float, float],
) -> Violation | None:
    """Connector opening must face the board edge, not the interior.

    Deterministic proxy: a connector's housing overhangs its pad field
    on the OPENING side (RJ45 jack mouth, terminal wire entries), so
    the vector from the pad centroid to the courtyard centroid must
    point toward the edge. Symmetric parts (bulge < 0.5mm) are
    indeterminate and pass. Converted from a Gate B vision finding
    (RJ45 facing the board interior) per the standing rule.
    """
    if pin.opening is not None:
        # Part-rule opening direction, rotated to board frame.
        bx, by = _rotate_kicad(pin.opening[0], pin.opening[1], fp.rotation)
    else:
        pad_cx, pad_cy = origin_to_centroid(
            fp, fp.position.x, fp.position.y, fp.rotation,
        )
        court = _courtyard_in_board(fp)
        ccx = sum(p.x for p in court) / len(court)
        ccy = sum(p.y for p in court) / len(court)
        bx, by = ccx - pad_cx, ccy - pad_cy
    if math.hypot(bx, by) < _FACE_OUT_MIN_BULGE_MM:
        return None
    normal = {
        Edge.WEST: (-1.0, 0.0), Edge.EAST: (1.0, 0.0),
        Edge.NORTH: (0.0, -1.0), Edge.SOUTH: (0.0, 1.0),
    }[edge]
    dot = bx * normal[0] + by * normal[1]
    if dot >= -_FACE_OUT_MIN_BULGE_MM:
        return None
    return Violation(
        repr(pin), (pin.ref,), Severity.MAJOR, -dot, _FACE_OUT_MIN_BULGE_MM,
        f"{pin.ref} opening faces the board interior "
        f"(body bulge {-dot:.1f}mm away from the {edge.value} edge)",
    )


def _worst_clearance(corners: Polygon, outline: Polygon) -> float | None:
    """Most negative signed clearance of any corner to the outline."""
    worst: float | None = None
    for corner in corners:
        boundary = _boundary_dist(corner.x, corner.y, outline)
        clearance = (
            boundary if point_in_polygon(corner.x, corner.y, outline) else -boundary
        )
        if worst is None or clearance < worst:
            worst = clearance
    return worst


def check_contain(pcb: PCBDesign, contain: BoardContain) -> tuple[Violation, ...]:
    """Pads AND courtyard (body extent) inside the outline (CRITICAL).

    Pads keep the configured margin. The courtyard — the body proxy —
    is checked at zero margin: flush with the edge is legal (edge
    connectors), past the edge is not. Gate C feedback 2026-06-11 item
    4a: contain checked PADS ONLY, so an ESP32 module whose castellated
    pads were in-board passed while its body hung off the outline.
    """
    outline = _outline_points(pcb.outline)
    out: list[Violation] = []
    for fp in pcb.footprints:
        if not fp.pads:
            continue
        worst_pad = _worst_clearance(
            tuple(c for pad in fp.pads for c in _pad_corners(fp, pad)), outline,
        )
        if worst_pad is not None and worst_pad < contain.margin_mm - _EPS:
            out.append(Violation(
                repr(contain), (fp.ref,), Severity.CRITICAL, worst_pad, contain.margin_mm,
                f"{fp.ref} pad clearance to board edge is {worst_pad:.3f}mm "
                f"(margin {contain.margin_mm}mm; negative = off board)",
            ))
        worst_body = _worst_clearance(_courtyard_in_board(fp), outline)
        if worst_body is not None and worst_body < -_EPS:
            out.append(Violation(
                repr(contain), (fp.ref,), Severity.CRITICAL, worst_body, 0.0,
                f"{fp.ref} courtyard/body extends {-worst_body:.3f}mm past "
                f"the board edge",
            ))
    return tuple(out)


def _segments_cross(
    a1: tuple[float, float], a2: tuple[float, float],
    b1: tuple[float, float], b2: tuple[float, float],
) -> bool:
    """True when open segments a1-a2 and b1-b2 properly intersect."""
    def orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return (
        orient(a1, a2, b1) * orient(a1, a2, b2) < -_EPS
        and orient(b1, b2, a1) * orient(b1, b2, a2) < -_EPS
    )


def check_attach_bundles(
    pcb: PCBDesign, bundles: tuple[AttachBundle, ...],
) -> tuple[Violation, ...]:
    """Each AttachBundle: zero crossings among its attach lines (MAJOR).

    Each bundle's nets are drawn as straight pad-center segments; any
    pairwise intersection means the pin order on one side does not
    mirror the other's physical pad order — an avoidable crossover
    trace (the relay NO/NC defect class, Gate C 2026-06-11 item 2).
    """
    out: list[Violation] = []
    for bundle in bundles:
        segments: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
        missing = False
        for net, (pad_a, pad_b) in zip(bundle.nets, bundle.pad_pairs, strict=True):
            ends: list[tuple[float, float]] = []
            for pad_ref in (pad_a, pad_b):
                fp = pcb.get_footprint(pad_ref.ref)
                pad = pad_by_number(fp, pad_ref.pin) if fp is not None else None
                if fp is None or pad is None:
                    out.append(_missing(bundle, pad_ref.ref, f"pad {pad_ref}"))
                    missing = True
                    break
                ends.append(_pad_center(fp, pad))
            if len(ends) == 2:
                segments.append((net, ends[0], ends[1]))
        if missing:
            continue
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                net_i, a1, a2 = segments[i]
                net_j, b1, b2 = segments[j]
                if _segments_cross(a1, a2, b1, b2):
                    out.append(Violation(
                        repr(bundle), (bundle.ref_a, bundle.ref_b), Severity.MAJOR,
                        1.0, 0.0,
                        f"attach lines {net_i} and {net_j} between "
                        f"{bundle.ref_a} and {bundle.ref_b} cross — pin order "
                        f"does not mirror the physical pad order",
                    ))
    return tuple(out)


def check_connector_fanouts(
    pcb: PCBDesign, fanouts: tuple[ConnectorFanout, ...],
) -> tuple[Violation, ...]:
    """Each ConnectorFanout: zero crossings among its ratsnest lines (MAJOR).

    Each line runs from the connector pad to the NEAREST candidate pad
    in board space — the deterministic proxy for the rendered ratsnest
    MST edge. Crossing lines at a connector mean the pin order (or the
    channel layout above it) forces avoidable crossover traces — the
    analog AIN/GND X, human finding 2026-06-11.
    """
    out: list[Violation] = []
    for fanout in fanouts:
        conn_fp = pcb.get_footprint(fanout.ref)
        if conn_fp is None:
            out.append(_missing(fanout, fanout.ref, f"connector {fanout.ref!r}"))
            continue
        segments: list[tuple[str, str, tuple[float, float], tuple[float, float]]] = []
        for line in fanout.lines:
            src_pad = pad_by_number(conn_fp, line.src.pin)
            if src_pad is None:
                out.append(_missing(fanout, fanout.ref, f"pad {line.src}"))
                continue
            sx, sy = _pad_center(conn_fp, src_pad)
            nearest: tuple[float, tuple[float, float], str] | None = None
            for cand in line.candidates:
                fp = pcb.get_footprint(cand.ref)
                pad = pad_by_number(fp, cand.pin) if fp is not None else None
                if fp is None or pad is None:
                    continue
                cx, cy = _pad_center(fp, pad)
                d = math.hypot(cx - sx, cy - sy)
                if nearest is None or d < nearest[0]:
                    nearest = (d, (cx, cy), cand.ref)
            if nearest is not None:
                segments.append((line.net, nearest[2], (sx, sy), nearest[1]))
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                net_i, ref_i, a1, a2 = segments[i]
                net_j, ref_j, b1, b2 = segments[j]
                if _segments_cross(a1, a2, b1, b2):
                    out.append(Violation(
                        repr(fanout), (fanout.ref, ref_i, ref_j), Severity.MAJOR,
                        1.0, 0.0,
                        f"ratsnest lines {net_i} (to {ref_i}) and {net_j} "
                        f"(to {ref_j}) cross at connector {fanout.ref}",
                    ))
    return tuple(out)


def check_group_assocs(
    pcb: PCBDesign, assocs: tuple[GroupAssoc, ...],
) -> tuple[Violation, ...]:
    """Each GroupAssoc: connector inside its parent's edge segment (MAJOR).

    The connector is its parent FeatureBlock's edge subgroup (board
    owner directive 2026-06-11): it must claim the edge SEGMENT
    adjacent to the parent's placement. Re-derived from the artifact:
    project the partner refs' pad centroids and the connector's
    COURTYARD interval onto the connector's edge axis; the gap between
    the two intervals must be <= tolerance. Interval gap, not centroid
    distance: a 20mm-wide RJ45 packed directly against its PHY cluster
    has a large centroid offset purely from its own half-span (ethernet
    trainer, 2026-06-11). Generalizes the K3/K4 "stranded from serving
    terminals" fab finding into a countable rule.
    """
    board_bbox = polygon_bbox(_outline_points(pcb.outline))
    out: list[Violation] = []
    for assoc in assocs:
        fp = pcb.get_footprint(assoc.ref)
        if fp is None:
            out.append(_missing(assoc, assoc.ref, f"connector {assoc.ref!r}"))
            continue
        fp_bbox = polygon_bbox(_courtyard_in_board(fp))
        _, edge_name = min(
            ((_edge_distance(fp_bbox, board_bbox, e), e.value) for e in Edge),
            key=lambda item: item[0],
        )
        horizontal = Edge(edge_name) in (Edge.NORTH, Edge.SOUTH)
        coords: list[float] = []
        for ref in assoc.partner_refs:
            partner = pcb.get_footprint(ref)
            if partner is None:
                continue
            cx, cy = origin_to_centroid(
                partner, partner.position.x, partner.position.y, partner.rotation,
            )
            coords.append(cx if horizontal else cy)
        if not coords:
            continue  # no partner on the board: nothing to measure
        hull_lo, hull_hi = min(coords), max(coords)
        conn_lo, conn_hi = (
            (fp_bbox[0], fp_bbox[2]) if horizontal else (fp_bbox[1], fp_bbox[3])
        )
        gap = max(0.0, hull_lo - conn_hi, conn_lo - hull_hi)
        if gap > assoc.tolerance_mm:
            out.append(Violation(
                repr(assoc), (assoc.ref, *assoc.partner_refs), Severity.MAJOR,
                gap, assoc.tolerance_mm,
                f"{assoc.ref} sits {gap:.1f}mm from its parent group "
                f"{assoc.group!r}'s segment on the {edge_name} edge "
                f"(connector spans [{conn_lo:.1f}, {conn_hi:.1f}], parent "
                f"hull [{hull_lo:.1f}, {hull_hi:.1f}], max {assoc.tolerance_mm}mm)",
            ))
    return tuple(out)


def check_courtyards(pcb: PCBDesign) -> tuple[Violation, ...]:
    """Zero courtyard overlaps between same-layer footprint pairs (CRITICAL)."""
    out: list[Violation] = []
    fps = pcb.footprints
    polys = [_courtyard_in_board(fp) if fp.pads else None for fp in fps]
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            poly_i, poly_j = polys[i], polys[j]
            if poly_i is None or poly_j is None or fps[i].layer != fps[j].layer:
                continue
            depth = _overlap_depth(poly_i, poly_j)
            if depth > _EPS:
                out.append(Violation(
                    "courtyard_collision", (fps[i].ref, fps[j].ref), Severity.CRITICAL,
                    depth, 0.0,
                    f"courtyards of {fps[i].ref} and {fps[j].ref} overlap by {depth:.3f}mm",
                ))
    return tuple(out)


def check_keepouts(
    pcb: PCBDesign, keepouts: tuple[CellKeepout, ...],
) -> tuple[Violation, ...]:
    """Each CellKeepout re-derived from the owner's actual placement (CRITICAL).

    Two assertions: (a) no OTHER footprint's pad center lies inside the
    derived board-frame polygon; (b) a board-level keepout zone exists
    whose bbox matches the derived polygon bbox within 1.0mm. The keepout
    polygon is owner-local relative to the owner's pad centroid (the cell
    coordinate convention), so it transforms with the owner's actual
    board position and rotation.
    """
    out: list[Violation] = []
    for keepout in keepouts:
        owner = pcb.get_footprint(keepout.owner)
        if owner is None:
            out.append(_missing(keepout, keepout.owner, f"owner {keepout.owner!r}"))
            continue
        derived = _local_to_board(owner, keepout.polygon)
        for fp in pcb.footprints:
            if fp.ref == keepout.owner:
                continue
            for pad in fp.pads:
                px, py = _pad_center(fp, pad)
                if point_in_polygon(px, py, derived):
                    out.append(Violation(
                        repr(keepout), (keepout.owner, fp.ref), Severity.CRITICAL,
                        0.0, 0.0,
                        f"{fp.ref} pad {pad.number} is inside the "
                        f"{keepout.kind.value} keepout owned by {keepout.owner}",
                    ))
                    break
        d_bbox = polygon_bbox(derived)
        matched = any(
            all(
                abs(z - d) <= KEEPOUT_BBOX_TOL_MM
                for z, d in zip(polygon_bbox(zone.polygon), d_bbox, strict=True)
            )
            for zone in pcb.keepouts
        )
        if not matched:
            out.append(Violation(
                repr(keepout), (keepout.owner,), Severity.CRITICAL,
                0.0, KEEPOUT_BBOX_TOL_MM,
                f"no board keepout zone matches the derived {keepout.kind.value} "
                f"polygon of {keepout.owner} "
                f"(expected bbox {tuple(round(v, 3) for v in d_bbox)})",
            ))
    return tuple(out)


def check_isolation(
    pcb: PCBDesign,
    isolation: tuple[IsolationGap, ...],
    domains: tuple[tuple[str, str], ...],
) -> tuple[Violation, ...]:
    """Each IsolationGap: courtyard gap between the two domains >= min_mm (MAJOR)."""
    domain_of = dict(domains)
    out: list[Violation] = []
    for gap in isolation:
        refs_a = [r for r, d in domain_of.items() if d == gap.domain_a]
        refs_b = [r for r, d in domain_of.items() if d == gap.domain_b]
        for ref_a in refs_a:
            fp_a = pcb.get_footprint(ref_a)
            if fp_a is None or not fp_a.pads:
                continue
            poly_a = _courtyard_in_board(fp_a)
            for ref_b in refs_b:
                fp_b = pcb.get_footprint(ref_b)
                if fp_b is None or not fp_b.pads:
                    continue
                measured = _polygon_gap(poly_a, _courtyard_in_board(fp_b))
                if measured < gap.min_mm - _EPS:
                    out.append(Violation(
                        repr(gap), (ref_a, ref_b), Severity.MAJOR, measured, gap.min_mm,
                        f"{ref_a} ({gap.domain_a}) is {measured:.3f}mm from "
                        f"{ref_b} ({gap.domain_b}), min {gap.min_mm}mm",
                    ))
    return tuple(out)
