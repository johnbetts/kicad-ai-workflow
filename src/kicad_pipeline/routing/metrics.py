"""Board-level routing quality metrics.

Provides :class:`BoardRoutingMetrics` and :func:`compute_board_metrics`
for evaluating overall routing quality after autorouting completes.

The cost function matches the project spec (Documentation/projectspecs.md):

.. code-block:: python

    board_cost = (
        1.00  * total_trace_length
      + 16.0  * total_vias
      + 3.00  * total_bends
      + 6.00  * sum(max(0, length_ratio - 1.55) for net in nets)
      + 70.0  * drc_violations
      + 200.0 * num_unrouted_segments_or_pins
      + 12.0  * max_congestion
      + 18.0  * avg_passive_to_dominant_pin_distance
      + 25.0  * num_via_chains_or_ping_pong
      + 10.0  * missing_gnd_pour_penalty_per_layer
      + 8.00  * detour_around_nearby_passive_penalty
    )
"""

from __future__ import annotations

import math
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
    # Extended spec-aligned metrics (default 0 for backward compat)
    total_bends: int = 0
    via_ping_pong_count: int = 0
    avg_passive_distance_mm: float = 0.0
    gnd_pour_missing_layers: int = 0
    detour_count: int = 0
    max_congestion: float = 0.0
    drc_violations: int = 0


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
    total_bends = 0
    ping_pong_count = 0

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
        total_bends += q.bend_count
        max_vias = max(max_vias, q.via_count)

        # Detect via ping-pong (top→bottom→top within one net)
        ping_pong_count += count_via_ping_pongs(r)

    overall_ratio = (
        total_length / total_manhattan
        if total_manhattan > 0.01
        else 1.0
    )

    # Compute passive proximity (avg distance to dominant connected pin)
    avg_passive_dist = compute_passive_proximity(footprints)

    return BoardRoutingMetrics(
        total_track_length_mm=round(total_length, 1),
        total_vias=total_vias,
        nets_routed=nets_routed,
        nets_failed=nets_failed,
        overall_length_ratio=round(overall_ratio, 2),
        max_vias_per_net=max_vias,
        per_net=tuple(per_net),
        total_bends=total_bends,
        via_ping_pong_count=ping_pong_count,
        avg_passive_distance_mm=round(avg_passive_dist, 2),
    )


def count_via_ping_pongs(result: RouteResult) -> int:
    """Count via ping-pong patterns (F.Cu→B.Cu→F.Cu) in a routed net.

    A ping-pong occurs when a net transitions from F.Cu to B.Cu and back
    (or vice versa), indicating unnecessary layer changes.

    Returns:
        Number of ping-pong patterns detected.
    """
    if not result.tracks or len(result.tracks) < 3:
        return 0
    # Build layer sequence (skip adjacent duplicates)
    layers: list[str] = []
    for trk in result.tracks:
        if not layers or layers[-1] != trk.layer:
            layers.append(trk.layer)
    # Each F→B→F or B→F→B is a ping-pong
    count = 0
    for i in range(len(layers) - 2):
        if layers[i] == layers[i + 2] and layers[i] != layers[i + 1]:
            count += 1
    return count


def compute_passive_proximity(
    footprints: list[Footprint],
) -> float:
    """Compute average distance from passive components to their nearest IC/connector.

    Passive components (R, C, L, D) should be placed close to their
    dominant connected pins (ICs, connectors, regulators). This measures
    the average distance to help evaluate placement quality.

    Args:
        footprints: All placed footprints.

    Returns:
        Average distance in mm from passives to nearest IC/connector,
        or 0.0 if no passives found.
    """
    passive_prefixes = ("R", "C", "L", "D")
    ic_prefixes = ("U", "J", "P", "Q", "SW")

    passives: list[tuple[float, float]] = []
    ics: list[tuple[float, float]] = []

    for fp in footprints:
        ref_alpha = ""
        for ch in fp.ref:
            if ch.isalpha():
                ref_alpha += ch
            else:
                break
        if ref_alpha in passive_prefixes:
            passives.append((fp.position.x, fp.position.y))
        elif ref_alpha in ic_prefixes:
            ics.append((fp.position.x, fp.position.y))

    if not passives or not ics:
        return 0.0

    total_dist = 0.0
    for px, py in passives:
        min_dist = float("inf")
        for ix, iy in ics:
            d = math.sqrt((px - ix) ** 2 + (py - iy) ** 2)
            min_dist = min(min_dist, d)
        total_dist += min_dist

    return total_dist / len(passives)


def count_detours(
    results: tuple[RouteResult, ...],
    threshold: float = 1.4,
) -> int:
    """Count nets with detour ratios exceeding threshold.

    A detour is a net where the actual trace length significantly exceeds
    the Manhattan ideal, suggesting the trace takes an indirect path.

    Args:
        results: Routing results.
        threshold: Length ratio above which a net counts as detouring.

    Returns:
        Number of nets with length_ratio > threshold.
    """

    count = 0
    for r in results:
        if not r.routed or not r.tracks:
            continue
        # Quick length ratio from tracks
        actual = sum(
            math.sqrt(
                (t.end.x - t.start.x) ** 2 + (t.end.y - t.start.y) ** 2,
            )
            for t in r.tracks
        )
        # Estimate manhattan from track endpoints
        if r.tracks:
            xs = [t.start.x for t in r.tracks] + [r.tracks[-1].end.x]
            ys = [t.start.y for t in r.tracks] + [r.tracks[-1].end.y]
            manhattan = abs(max(xs) - min(xs)) + abs(max(ys) - min(ys))
            if manhattan > 0.01:
                ratio = actual / manhattan
                if ratio > threshold:
                    count += 1
    return count


def compute_board_cost(
    metrics: BoardRoutingMetrics,
    *,
    drc_violations: int = 0,
    gnd_pour_missing_layers: int = 0,
) -> float:
    """Compute aggregate board-level routing cost per spec cost function.

    Full spec formula::

        board_cost = (
            1.00  * total_trace_length
          + 16.0  * total_vias
          + 3.00  * total_bends
          + 6.00  * sum(max(0, ratio - 1.55) for net)
          + 70.0  * drc_violations
          + 200.0 * nets_failed (unrouted)
          + 12.0  * max_congestion
          + 18.0  * avg_passive_to_dominant_pin_distance
          + 25.0  * via_ping_pong_count
          + 10.0  * missing_gnd_pour_penalty_per_layer
          + 8.00  * detour_count
        )

    Args:
        metrics: Board routing metrics.
        drc_violations: Number of DRC violations (from external check).
        gnd_pour_missing_layers: Number of layers missing GND pour (0-2).

    Returns:
        Board-level cost score (lower is better).
    """
    ratio_penalty = sum(
        6.0 * max(0.0, q.length_ratio - 1.55) for q in metrics.per_net
    )
    gnd_layers = gnd_pour_missing_layers or metrics.gnd_pour_missing_layers
    return (
        1.0 * metrics.total_track_length_mm
        + 16.0 * metrics.total_vias
        + 3.0 * metrics.total_bends
        + ratio_penalty
        + 70.0 * (drc_violations or metrics.drc_violations)
        + 200.0 * metrics.nets_failed
        + 12.0 * metrics.max_congestion
        + 18.0 * metrics.avg_passive_distance_mm
        + 25.0 * metrics.via_ping_pong_count
        + 10.0 * gnd_layers
        + 8.0 * metrics.detour_count
    )
