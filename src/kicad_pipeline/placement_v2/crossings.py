"""Avoidable ratsnest-crossing count — the placement objective.

Council verdict 2026-06-11: crossing minimization belongs in PLACEMENT
and is the single zero-config generalizable default — it is the board
owner's stated method ("look at the lines and see what cross is
avoidable by moving components"). This module evaluates the SAME
nearest-candidate fanout semantics Gate A verifies
(:func:`verifier_rules.check_connector_fanouts`), but over solver-frame
positions so the floorplanner can optimize what the gate will measure.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from kicad_pipeline.placement_v2.footprint_geom import pad_position_in_frame

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kicad_pipeline.models.pcb import Footprint
    from kicad_pipeline.placement_v2.ir import AttachBundle, ConnectorFanout

_EPS = 1e-9

Position = tuple[float, float, float]  # centroid x, y, KiCad rotation deg


def _segments_cross(
    a1: tuple[float, float], a2: tuple[float, float],
    b1: tuple[float, float], b2: tuple[float, float],
) -> bool:
    def orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return (
        orient(a1, a2, b1) * orient(a1, a2, b2) < -_EPS
        and orient(b1, b2, a1) * orient(b1, b2, a2) < -_EPS
    )


def _pad_xy(
    positions: Mapping[str, Position],
    footprints: Mapping[str, Footprint],
    ref: str,
    pin: str,
) -> tuple[float, float] | None:
    pos = positions.get(ref)
    fp = footprints.get(ref)
    if pos is None or fp is None:
        return None
    try:
        return pad_position_in_frame(fp, pin, pos[0], pos[1], pos[2])
    except KeyError:
        return None


def count_crossings(
    positions: Mapping[str, Position],
    footprints: Mapping[str, Footprint],
    fanouts: tuple[ConnectorFanout, ...],
    bundles: tuple[AttachBundle, ...] = (),
) -> int:
    """Total avoidable crossings at the given (centroid-frame) positions.

    Fanouts: each connector pad's line runs to its NEAREST same-net pad
    (the rendered-ratsnest proxy); crossings are counted pairwise per
    connector. Bundles: pairwise crossings among a part pair's attach
    lines. Matches Gate A's violation counting exactly.
    """
    total = 0
    for fanout in fanouts:
        segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for line in fanout.lines:
            src = _pad_xy(positions, footprints, line.src.ref, line.src.pin)
            if src is None:
                continue
            best: tuple[float, tuple[float, float]] | None = None
            for cand in line.candidates:
                pt = _pad_xy(positions, footprints, cand.ref, cand.pin)
                if pt is None:
                    continue
                d = math.hypot(pt[0] - src[0], pt[1] - src[1])
                if best is None or d < best[0]:
                    best = (d, pt)
            if best is not None:
                segs.append((src, best[1]))
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                if _segments_cross(segs[i][0], segs[i][1], segs[j][0], segs[j][1]):
                    total += 1
    for bundle in bundles:
        bsegs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for pad_a, pad_b in bundle.pad_pairs:
            a = _pad_xy(positions, footprints, pad_a.ref, pad_a.pin)
            b = _pad_xy(positions, footprints, pad_b.ref, pad_b.pin)
            if a is not None and b is not None:
                bsegs.append((a, b))
        for i in range(len(bsegs)):
            for j in range(i + 1, len(bsegs)):
                if _segments_cross(bsegs[i][0], bsegs[i][1], bsegs[j][0], bsegs[j][1]):
                    total += 1
    return total


__all__ = ["Position", "count_crossings"]
