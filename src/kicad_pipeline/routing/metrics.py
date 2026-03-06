"""Board-level routing quality metrics.

Provides :class:`BoardRoutingMetrics` and :func:`compute_board_metrics`
for evaluating overall routing quality after autorouting completes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint
    from kicad_pipeline.routing.grid_router import RouteQuality, RouteResult


@dataclass(frozen=True)
class BoardRoutingMetrics:
    """Aggregate routing quality for an entire board."""

    total_track_length_mm: float
    total_vias: int
    nets_routed: int
    nets_failed: int
    overall_length_ratio: float
    max_vias_per_net: int
    per_net: tuple[RouteQuality, ...]


def compute_board_metrics(
    results: tuple[RouteResult, ...],
    footprints: list[Footprint],
) -> BoardRoutingMetrics:
    """Compute board-level routing metrics from all route results.

    Args:
        results: All routing results from :func:`route_all_nets`.
        footprints: Board footprints for pad position lookup.

    Returns:
        :class:`BoardRoutingMetrics` with per-net and aggregate data.
    """
    from kicad_pipeline.routing.grid_router import _pad_abs_pos, _score_route

    fp_by_ref: dict[str, Footprint] = {fp.ref: fp for fp in footprints}

    per_net: list[RouteQuality] = []
    total_length = 0.0
    total_vias = 0
    total_manhattan = 0.0
    nets_routed = 0
    nets_failed = 0
    max_vias = 0

    for r in results:
        if r.routed:
            nets_routed += 1
        else:
            nets_failed += 1
            continue

        # Resolve pad positions for this net
        pad_positions: list[tuple[float, float]] = []
        for ref, pad_num in getattr(r, "_pad_refs", ()):
            fp = fp_by_ref.get(ref)
            if fp is None:
                continue
            for pad in fp.pads:
                if pad.number == pad_num:
                    pad_positions.append(_pad_abs_pos(fp, pad))
                    break

        # If we can't resolve pads, estimate from track endpoints
        if len(pad_positions) < 2 and r.tracks:
            endpoints: set[tuple[float, float]] = set()
            for trk in r.tracks:
                endpoints.add((round(trk.start.x, 3), round(trk.start.y, 3)))
                endpoints.add((round(trk.end.x, 3), round(trk.end.y, 3)))
            pad_positions = list(endpoints)[:10]

        q = _score_route(r, pad_positions)
        per_net.append(q)
        total_length += q.actual_length_mm
        total_manhattan += q.manhattan_ideal_mm
        total_vias += q.via_count
        max_vias = max(max_vias, q.via_count)

    overall_ratio = (
        total_length / total_manhattan
        if total_manhattan > 0.01
        else 1.0
    )

    return BoardRoutingMetrics(
        total_track_length_mm=round(total_length, 1),
        total_vias=total_vias,
        nets_routed=nets_routed,
        nets_failed=nets_failed,
        overall_length_ratio=round(overall_ratio, 2),
        max_vias_per_net=max_vias,
        per_net=tuple(per_net),
    )


def compute_board_cost(metrics: BoardRoutingMetrics) -> float:
    """Compute aggregate board-level routing cost from metrics.

    Uses the spec cost formula:
        total_length + 14*total_vias + sum(2.5*bends) + sum(5*max(0,ratio-1.5))

    Returns:
        Board-level cost score (lower is better).
    """
    bend_penalty = sum(q.bend_count for q in metrics.per_net) * 2.5
    ratio_penalty = sum(
        5.0 * max(0.0, q.length_ratio - 1.5) for q in metrics.per_net
    )
    return (
        metrics.total_track_length_mm
        + 14.0 * metrics.total_vias
        + bend_penalty
        + ratio_penalty
    )
