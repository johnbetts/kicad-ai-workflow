"""Post-routing validation and cleanup for the grid router.

Contains track clearance validation, pad-crossing detection, and
track/via collection utilities that run after the main routing pass.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint, Track, Via
    from kicad_pipeline.pcb.netlist import NetlistEntry
    from kicad_pipeline.routing.grid_router import RouteResult, _Grid

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def point_to_segment_dist(
    px: float, py: float,
    sx1: float, sy1: float, sx2: float, sy2: float,
) -> float:
    """Compute minimum distance from point (px, py) to line segment (sx1,sy1)-(sx2,sy2)."""
    dx = sx2 - sx1
    dy = sy2 - sy1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.sqrt((px - sx1) ** 2 + (py - sy1) ** 2)
    t = max(0.0, min(1.0, ((px - sx1) * dx + (py - sy1) * dy) / len_sq))
    cx = sx1 + t * dx
    cy = sy1 + t * dy
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def segment_min_distance(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> float:
    """Compute minimum distance between two line segments.

    Checks for intersection first, then falls back to point-to-segment
    distances for non-intersecting segments.
    """
    # Check for segment intersection using cross products
    dx_a = ax2 - ax1
    dy_a = ay2 - ay1
    dx_b = bx2 - bx1
    dy_b = by2 - by1
    denom = dx_a * dy_b - dy_a * dx_b
    if abs(denom) > 1e-12:
        t = ((bx1 - ax1) * dy_b - (by1 - ay1) * dx_b) / denom
        u = ((bx1 - ax1) * dy_a - (by1 - ay1) * dx_a) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0

    def _point_seg_dist(
        px: float, py: float,
        sx1: float, sy1: float, sx2: float, sy2: float,
    ) -> float:
        dx = sx2 - sx1
        dy = sy2 - sy1
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-12:
            return math.sqrt((px - sx1) ** 2 + (py - sy1) ** 2)
        t = max(0.0, min(1.0, ((px - sx1) * dx + (py - sy1) * dy) / len_sq))
        cx = sx1 + t * dx
        cy = sy1 + t * dy
        return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)

    return min(
        _point_seg_dist(ax1, ay1, bx1, by1, bx2, by2),
        _point_seg_dist(ax2, ay2, bx1, by1, bx2, by2),
        _point_seg_dist(bx1, by1, ax1, ay1, ax2, ay2),
        _point_seg_dist(bx2, by2, ax1, ay1, ax2, ay2),
    )


# ---------------------------------------------------------------------------
# Post-route validation
# ---------------------------------------------------------------------------


def drop_pad_crossing_tracks(
    results: list[RouteResult],
    footprints: list[Footprint],
) -> list[RouteResult]:
    """Remove individual tracks that cross pads of other nets.

    After track simplification, merged diagonal segments may cross pad areas
    that weren't in the original A* path.  This post-route filter detects
    and drops those offending segments.
    """
    import math as _math

    # Build pad info: (abs_x, abs_y, net_number, half_w, half_h, ref)
    pad_info: list[tuple[float, float, int, float, float, str]] = []
    ref_nets: dict[str, set[int]] = {}
    for fp in footprints:
        rot_rad = _math.radians(fp.rotation)
        cos_r = _math.cos(rot_rad)
        sin_r = _math.sin(rot_rad)
        fp_nets: set[int] = set()
        for pad in fp.pads:
            rpx = pad.position.x * cos_r - pad.position.y * sin_r
            rpy = pad.position.x * sin_r + pad.position.y * cos_r
            px = fp.position.x + rpx
            py = fp.position.y + rpy
            net = pad.net_number if pad.net_number is not None else 0
            pad_info.append((px, py, net, pad.size_x / 2.0, pad.size_y / 2.0, fp.ref))
            if net > 0:
                fp_nets.add(net)
        ref_nets[fp.ref] = fp_nets

    updated: list[RouteResult] = []
    for r in results:
        if not r.routed or not r.tracks:
            updated.append(r)
            continue

        good_tracks: list[Track] = []
        for track in r.tracks:
            # Only check long diagonal segments (>3mm)
            dx = abs(track.end.x - track.start.x)
            dy = abs(track.end.y - track.start.y)
            seg_len = (dx * dx + dy * dy) ** 0.5
            if seg_len < 3.0:
                good_tracks.append(track)
                continue

            thw = track.width / 2.0
            crosses = False
            for px, py, pnet, hw, hh, ref in pad_info:
                if pnet == track.net_number or pnet == 0:
                    continue
                # Skip intra-footprint crossings
                if track.net_number in ref_nets.get(ref, set()):
                    continue
                # Use point-to-segment distance for precise check
                dist = point_to_segment_dist(
                    px, py,
                    track.start.x, track.start.y,
                    track.end.x, track.end.y,
                )
                # Track crosses pad if track edge (dist - half_width)
                # is less than pad half-size
                pad_radius = max(hw, hh)
                if dist < pad_radius + thw - 0.05:
                    crosses = True
                    break

            if not crosses:
                good_tracks.append(track)

        from kicad_pipeline.routing.grid_router import RouteResult as RouteResultCls
        updated.append(RouteResultCls(
            net_name=r.net_name,
            net_number=r.net_number,
            routed=r.routed,
            tracks=tuple(good_tracks),
            vias=r.vias,
        ))

    return updated


def validate_track_clearances(
    results: list[RouteResult],
    grid: _Grid,
    bcu_grid: _Grid,
    grid_step_mm: float,
    entry_by_name: dict[str, NetlistEntry],
    route_fn: object,
    footprints: list[Footprint],
    net_clearances: dict[str, float] | None,
    net_widths: dict[str, float] | None,
    pad_positions_fn: object,
) -> list[RouteResult]:
    """Detect and fix cross-net clearance violations after routing.

    Iterates all track pairs from different nets. If edge-to-edge distance
    violates CLEARANCE_DEFAULT_MM, rips up the worse-scored net and re-routes.
    Repeats up to 3 times to resolve cascading violations.
    """
    from kicad_pipeline.constants import CLEARANCE_DEFAULT_MM
    from kicad_pipeline.routing.grid_router import _score_route

    log = _log

    previously_ripped: set[str] = set()

    for iteration in range(3):
        # Build per-net track lists
        net_tracks: dict[int, list[Track]] = {}
        net_result_idx: dict[int, int] = {}
        for idx, r in enumerate(results):
            if not r.routed:
                continue
            net_tracks[r.net_number] = list(r.tracks)
            net_result_idx[r.net_number] = idx

        # Check all cross-net pairs for clearance violations
        violating_nets: set[int] = set()
        net_nums = list(net_tracks.keys())
        for i in range(len(net_nums)):
            for j in range(i + 1, len(net_nums)):
                n1, n2 = net_nums[i], net_nums[j]
                for t1 in net_tracks[n1]:
                    for t2 in net_tracks[n2]:
                        if t1.layer != t2.layer:
                            continue
                        hw1 = t1.width / 2.0
                        hw2 = t2.width / 2.0
                        min_gap = CLEARANCE_DEFAULT_MM
                        edge_dist = segment_min_distance(
                            t1.start.x, t1.start.y, t1.end.x, t1.end.y,
                            t2.start.x, t2.start.y, t2.end.x, t2.end.y,
                        ) - hw1 - hw2
                        if edge_dist < min_gap - 0.001:
                            violating_nets.add(n1)
                            violating_nets.add(n2)

        if not violating_nets:
            return results

        log.info(
            "clearance validation (iter %d): %d nets involved in violations",
            iteration + 1, len(violating_nets),
        )

        # Score violating nets, rip up the worst half
        scored: list[tuple[float, int]] = []
        for net_num in violating_nets:
            maybe_idx = net_result_idx.get(net_num)
            if maybe_idx is None:
                continue
            idx = maybe_idx
            r = results[idx]
            entry = entry_by_name.get(r.net_name)
            if entry is None:
                continue
            q = _score_route(r, pad_positions_fn(entry))  # type: ignore[operator]
            scored.append((q.score, idx))

        if not scored:
            return results

        scored.sort(key=lambda x: x[0], reverse=True)

        # Deprioritize nets with B.Cu routes (vias) -- they are
        # harder to re-route and may lose B.Cu corridors.
        fcu_only = [(s, i) for s, i in scored
                    if not results[i].vias]
        has_vias = [(s, i) for s, i in scored
                    if results[i].vias]
        scored = fcu_only + has_vias

        # Break rip-up oscillation: on iteration 2+, prefer ripping nets
        # that haven't been ripped before so both sides get a chance.
        if previously_ripped:
            never_ripped = [(s, i) for s, i in scored
                            if results[i].net_name not in previously_ripped]
            prev_ripped = [(s, i) for s, i in scored
                           if results[i].net_name in previously_ripped]
            scored = never_ripped + prev_ripped

        n_ripup = max(1, len(scored) // 2)
        ripup_indices = [idx for _, idx in scored[:n_ripup]]

        # Unmark tracks and vias from ripped routes
        for ri in ripup_indices:
            rr = results[ri]
            for trk in rr.tracks:
                if trk.layer == "F.Cu":
                    unmark_route_tracks(grid, [trk], grid_step_mm)
                elif trk.layer == "B.Cu":
                    unmark_route_tracks(bcu_grid, [trk], grid_step_mm)
            # Unmark vias on BOTH grids (vias span both layers)
            for via in rr.vias:
                vc, vr_ = grid.to_cell(
                    via.position.x, via.position.y,
                )
                grid.unmark(vc, vr_)
                bcu_grid.unmark(vc, vr_)

        ripped_names: set[str] = set()
        for ri in ripup_indices:
            ripped_names.add(results[ri].net_name)
            previously_ripped.add(results[ri].net_name)
        for ri in sorted(ripup_indices, reverse=True):
            results.pop(ri)

        log.debug("clearance rip-up: %s", ", ".join(sorted(ripped_names)))
        for name in ripped_names:
            retry_entry = entry_by_name.get(name)
            if retry_entry is not None:
                new_result = route_fn(retry_entry)  # type: ignore[operator]
                results.append(new_result)

    return results


def unmark_route_tracks(
    grid: _Grid | None,
    tracks: list[Track],
    grid_step_mm: float,
) -> None:
    """Unmark grid cells occupied by routed tracks (for rip-up)."""
    if grid is None:
        return
    for trk in tracks:
        c1, r1 = grid.to_cell(trk.start.x, trk.start.y)
        c2, r2 = grid.to_cell(trk.end.x, trk.end.y)
        # Walk line between cells
        dc = abs(c2 - c1)
        dr = abs(r2 - r1)
        sc = 1 if c1 < c2 else -1
        sr = 1 if r1 < r2 else -1
        err = dc - dr
        cc, cr = c1, r1
        while True:
            grid.unmark(cc, cr)
            if cc == c2 and cr == r2:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                cc += sc
            if e2 < dc:
                err += dc
                cr += sr


# ---------------------------------------------------------------------------
# Track/via collection
# ---------------------------------------------------------------------------


def collect_tracks(
    results: tuple[RouteResult, ...],
    *,
    routed_only: bool = True,
    filter_dangling: bool = True,
) -> tuple[Track, ...]:
    """Flatten all Track objects from all RouteResults into a single tuple.

    When *routed_only* is ``True`` (default), tracks from partially-routed
    nets are excluded.  Partial routes create tracks that don't complete
    connections, causing both ``unconnected`` and ``clearance``/``shorting``
    DRC violations -- removing them reduces overall violation count.

    When *filter_dangling* is ``True`` (default), single-segment tracks whose
    endpoints don't connect to any other track in the same net are removed.

    Args:
        results: Routing results to collect tracks from.
        routed_only: Only include tracks from fully-routed nets (default True).
        filter_dangling: Remove orphan single-segment stubs (default True).

    Returns:
        Combined tuple of all tracks.
    """
    tracks: list[Track] = []
    for r in results:
        if routed_only and not r.routed:
            continue
        tracks.extend(r.tracks)

    if not filter_dangling or len(tracks) < 2:
        return tuple(tracks)

    # Group tracks by net, then find dangling endpoints
    by_net: dict[int, list[Track]] = {}
    for t in tracks:
        by_net.setdefault(t.net_number, []).append(t)

    keep: list[Track] = []
    for net_tracks in by_net.values():
        if len(net_tracks) <= 1:
            # A single-segment net is OK (direct pad-to-pad)
            keep.extend(net_tracks)
            continue

        # Build endpoint connectivity: count how many tracks touch each point
        eps: dict[tuple[float, float], int] = {}
        for t in net_tracks:
            sk = (round(t.start.x, 4), round(t.start.y, 4))
            ek = (round(t.end.x, 4), round(t.end.y, 4))
            eps[sk] = eps.get(sk, 0) + 1
            eps[ek] = eps.get(ek, 0) + 1

        for t in net_tracks:
            sk = (round(t.start.x, 4), round(t.start.y, 4))
            ek = (round(t.end.x, 4), round(t.end.y, 4))
            # A stub has both endpoints only appearing once (no connections)
            if eps.get(sk, 0) <= 1 and eps.get(ek, 0) <= 1:
                continue  # orphan stub -- skip
            keep.append(t)

    return tuple(keep)


def collect_vias(
    results: tuple[RouteResult, ...],
    *,
    routed_only: bool = True,
) -> tuple[Via, ...]:
    """Flatten all Via objects from RouteResults into a single tuple.

    Args:
        results: Routing results to collect vias from.
        routed_only: When ``True`` (default), skip vias from unrouted nets
            to avoid dangling via DRC violations.

    Returns:
        Combined tuple of all vias.
    """
    vias: list[Via] = []
    for r in results:
        if routed_only and not r.routed:
            continue
        vias.extend(r.vias)

    # Deduplicate: skip vias at same position (within 0.01mm)
    seen: set[tuple[float, float]] = set()
    deduped: list[Via] = []
    for v in vias:
        key = (round(v.position.x, 2), round(v.position.y, 2))
        if key not in seen:
            seen.add(key)
            deduped.append(v)

    # Distance-based dedup: skip if any previously-kept same-net via
    # is within via.size mm (pad diameter), preventing hole_to_hole violations
    final: list[Via] = []
    for v in deduped:
        too_close = False
        for kept in final:
            if kept.net_number != v.net_number:
                continue
            dist = math.hypot(
                v.position.x - kept.position.x,
                v.position.y - kept.position.y,
            )
            if dist < v.size:
                too_close = True
                break
        if not too_close:
            final.append(v)
    return tuple(final)
