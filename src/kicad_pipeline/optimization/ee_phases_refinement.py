"""EE placement optimizer — late refinement and finalization phases.

Contains the late-stage placement refinement phases: decoupling re-pull,
ADC/relay re-alignment, collision resolution, board-edge clamping,
review loop, and final PCB assembly.

Extracted from ``ee_phases.py`` to reduce module size.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.optimization.collision_resolver import (
    _count_collisions,
    _resolve_collisions,
    _rotation_aware_size,
)
from kicad_pipeline.optimization.functional_grouper import (
    SubCircuitType,
)
from kicad_pipeline.optimization.group_helpers import (
    _apply_review_fixes,
    _extract_group_bboxes,
)
from kicad_pipeline.optimization.placement_types import (
    PlacementContext,
    _apply_positions,
    _dict_to_positions,
)
from kicad_pipeline.pcb.pin_map import (
    centroid_to_origin,
    origin_to_centroid,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from kicad_pipeline.models.pcb import Footprint, PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.review_agent import PlacementReview

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared clamping / collision helpers
# ---------------------------------------------------------------------------


def _clamp_ref_pad_extent(
    ref: str,
    rx: float,
    ry: float,
    rot: float,
    fp_obj: object,
    bounds: tuple[float, float, float, float],
    edge_margin: float,
) -> tuple[float, float] | None:
    """Clamp a component by pad extent to stay within board bounds.

    Returns:
        ``(new_cx, new_cy)`` if clamped, or ``None`` if no shift needed.
    """
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space

    min_x, min_y, max_x, max_y = bounds
    ori_x, ori_y = centroid_to_origin(fp_obj, rx, ry, rot)  # type: ignore[arg-type]
    px0, py0, px1, py1 = pad_extent_in_board_space(
        fp_obj, ori_x, ori_y, rot,  # type: ignore[arg-type]
    )
    shift_x = shift_y = 0.0
    if px0 < min_x + edge_margin:
        shift_x = (min_x + edge_margin) - px0
    elif px1 > max_x - edge_margin:
        shift_x = (max_x - edge_margin) - px1
    if py0 < min_y + edge_margin:
        shift_y = (min_y + edge_margin) - py0
    elif py1 > max_y - edge_margin:
        shift_y = (max_y - edge_margin) - py1
    if shift_x == 0.0 and shift_y == 0.0:
        return None
    new_cx, new_cy = origin_to_centroid(
        fp_obj, ori_x + shift_x, ori_y + shift_y, rot,  # type: ignore[arg-type]
    )
    return new_cx, new_cy


def _clamp_ref_simple(
    rx: float,
    ry: float,
    w: float,
    h: float,
    bounds: tuple[float, float, float, float],
    edge_margin: float = 1.5,
) -> tuple[float, float]:
    """Clamp a component center to stay within board bounds using simple w/h."""
    min_x, min_y, max_x, max_y = bounds
    clamped_x = max(min_x + w / 2 + edge_margin, min(max_x - w / 2 - edge_margin, rx))
    clamped_y = max(min_y + h / 2 + edge_margin, min(max_y - h / 2 - edge_margin, ry))
    return clamped_x, clamped_y


def _clamp_all_positions(
    positions: dict[str, tuple[float, float, float]],
    fp_lookup: dict[str, object],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    edge_margin: float,
    skip_refs: set[str] | None = None,
    label: str = "Clamp",
) -> int:
    """Clamp all positions to board bounds, using pad extent when available.

    Returns:
        Number of components clamped.
    """
    count = 0
    for ref, (rx, ry, rot) in list(positions.items()):
        if skip_refs and ref in skip_refs:
            continue
        fp_obj = fp_lookup.get(ref)
        if fp_obj is not None and fp_obj.pads:  # type: ignore[union-attr]
            result = _clamp_ref_pad_extent(
                ref, rx, ry, rot, fp_obj, bounds, edge_margin,
            )
            if result is not None:
                positions[ref] = (result[0], result[1], rot)
                count += 1
        else:
            w, h = _rotation_aware_size(ref, positions, fp_sizes)
            cx, cy = _clamp_ref_simple(rx, ry, w, h, bounds)
            if cx != rx or cy != ry:
                positions[ref] = (cx, cy, rot)
                count += 1
    if count:
        _log.info("  %s: %d components repositioned", label, count)
    return count


def _build_subcircuit_fixed(ctx: PlacementContext) -> set[str]:
    """Build the set of subcircuit-fixed refs from context."""
    return (ctx.relay_support_refs | ctx.adc_channel_refs
            | ctx.mcu_peripheral_refs | ctx.power_group_fixed
            | ctx.ethernet_fixed | ctx.template_fixed
            | ctx.top_edge_connector_refs)


def _build_always_base_refs(ctx: PlacementContext,
                            positions: dict[str, tuple[float, float, float]],
                            ) -> set[str]:
    """Build the base set of always-protected refs for collision resolution."""
    return (ctx.mcu_peripheral_refs | ctx.top_edge_connector_refs
            | ctx.ethernet_fixed | ctx.adc_channel_refs | ctx.adc_ic_refs
            | ctx.relay_support_refs | ctx.power_group_fixed
            | {r for r in positions if r.startswith("K")})


def _unprotect_small_colliders(
    collisions: list[tuple[str, str]],
    always_base: set[str],
    fp_sizes: dict[str, tuple[float, float]],
    power_group_fixed: set[str],
) -> set[str]:
    """Find refs that should be unprotected because they are the smaller
    component in a collision where both sides are in always_base."""
    unprotect: set[str] = set()
    for a, b in collisions:
        if a in always_base and b in always_base:
            area_a = fp_sizes.get(a, (2, 2))[0] * fp_sizes.get(a, (2, 2))[1]
            area_b = fp_sizes.get(b, (2, 2))[0] * fp_sizes.get(b, (2, 2))[1]
            smaller = a if area_a <= area_b else b
            if (not smaller.startswith(("U", "Y", "K", "Q"))
                    and smaller not in power_group_fixed):
                unprotect.add(smaller)
    return unprotect


def _resolve_post_phase_collisions(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
    subcircuit_fixed: set[str],
    label: str,
) -> dict[str, tuple[float, float, float]]:
    """Run collision detection and resolution with standard protection logic."""
    collisions = _count_collisions(positions, fp_sizes)
    if not collisions:
        return positions

    _log.info("  %d %s collisions -- resolving", len(collisions), label)
    colliding = set()
    for a, b in collisions:
        colliding.add(a)
        colliding.add(b)

    always_base = _build_always_base_refs(ctx, positions)
    unprotect = _unprotect_small_colliders(
        collisions, always_base, fp_sizes, ctx.power_group_fixed,
    )
    always_fixed = always_base - unprotect
    targeted = (ctx.fixed_refs
                | (subcircuit_fixed - colliding)
                | always_fixed)
    return _resolve_collisions(positions, fp_sizes, bounds, targeted)


def _compute_edge_distance(
    cx: float, cy: float, cw: float, ch: float,
    ix: float, iy: float, iw: float, ih: float,
) -> float:
    """Compute edge-to-edge distance between two component bounding boxes."""
    dx_edge = abs(cx - ix) - (iw + cw) / 2.0
    dy_edge = abs(cy - iy) - (ih + ch) / 2.0
    if dx_edge <= 0 and dy_edge <= 0:
        return 0.0
    if dx_edge <= 0:
        return dy_edge
    if dy_edge <= 0:
        return dx_edge
    return math.sqrt(dx_edge ** 2 + dy_edge ** 2)


def _cap_side_position(
    placed_count: int,
    ix: float, iy: float, iw: float, ih: float,
    cw: float, ch: float,
) -> tuple[float, float]:
    """Compute decoupling cap position on one of 4 IC sides in round-robin order."""
    side = placed_count % 4
    tier = placed_count // 4
    if side == 0:
        return ix + tier * (cw + 0.5), iy - ih / 2.0 - ch / 2.0 - 0.5
    if side == 1:
        return ix + tier * (cw + 0.5), iy + ih / 2.0 + ch / 2.0 + 0.5
    if side == 2:
        return ix + iw / 2.0 + cw / 2.0 + 0.5, iy + tier * (ch + 0.5)
    return ix - iw / 2.0 - cw / 2.0 - 0.5, iy + tier * (ch + 0.5)


def _post_clamp_decoupling_repull(
    ctx: PlacementContext,
) -> None:
    """Re-pull decoupling caps that drifted too far from their IC after clamping."""
    _ref_to_group: dict[str, str] = getattr(ctx, "_ref_to_group", {})
    if not _ref_to_group:
        for feat in ctx.requirements.features:
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                _ref_to_group[r] = feat.name

    sc_list = list(ctx.subcircuits)
    _post_clamp_decoup = 0
    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.DECOUPLING:
            continue
        ic_ref = sc.anchor_ref
        if ic_ref not in ctx.positions:
            continue
        ic_group = _ref_to_group.get(ic_ref, "")
        ix, iy, _irot = ctx.positions[ic_ref]
        iw, ih = ctx.fp_sizes.get(ic_ref, (5.0, 5.0))
        if _irot % 180 in (90.0, 270.0):
            iw, ih = ih, iw
        placed_count = 0
        for cap_ref in sc.refs:
            if cap_ref == ic_ref or not cap_ref.startswith("C"):
                continue
            if cap_ref not in ctx.positions:
                continue
            # Skip caps that were deliberately positioned by a group phase
            # (e.g. ethernet phase) — those placements are intentional and
            # must not be overridden here.
            if cap_ref in ctx.power_group_fixed or cap_ref in ctx.fixed_refs:
                continue
            cap_group = _ref_to_group.get(cap_ref, "")
            if cap_group != ic_group:
                continue
            cx, cy, crot = ctx.positions[cap_ref]
            cw, ch = ctx.fp_sizes.get(cap_ref, (1.5, 1.0))
            edge_dist = _compute_edge_distance(cx, cy, cw, ch, ix, iy, iw, ih)
            if edge_dist <= 5.0:
                continue
            tx, ty = _cap_side_position(placed_count, ix, iy, iw, ih, cw, ch)
            tx = max(ctx.bounds[0] + 1.0, min(ctx.bounds[2] - 1.0, tx))
            ty = max(ctx.bounds[1] + 1.0, min(ctx.bounds[3] - 1.0, ty))
            ctx.positions[cap_ref] = (tx, ty, crot)
            placed_count += 1
            _post_clamp_decoup += 1
    if _post_clamp_decoup:
        _log.info("  Post-clamp decoupling re-pull: %d caps repositioned",
                  _post_clamp_decoup)


def _resolve_crystal_overlaps(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> None:
    """Resolve overlaps between crystal refs (Y*) and capacitors (C*)."""
    min_x, min_y, max_x, max_y = bounds
    crystal_refs = [r for r in positions if r.startswith("Y")]
    for yref in crystal_refs:
        yx, yy, yrot = positions[yref]
        yw, yh = fp_sizes.get(yref, (3.2, 1.5))
        if yrot % 180 in (90.0, 270.0):
            yw, yh = yh, yw
        for cref in list(positions):
            if cref == yref or not cref.startswith("C"):
                continue
            cx, cy, crot = positions[cref]
            cw, ch = fp_sizes.get(cref, (1.5, 1.0))
            if crot % 180 in (90.0, 270.0):
                cw, ch = ch, cw
            overlap_x = (cw + yw) / 2.0 + 0.5 - abs(cx - yx)
            overlap_y = (ch + yh) / 2.0 + 0.5 - abs(cy - yy)
            if overlap_x > 0 and overlap_y > 0:
                if overlap_x < overlap_y:
                    shift = overlap_x + 0.5
                    new_cx = cx + shift if cx > yx else cx - shift
                    new_cx = max(min_x + 2, min(max_x - 2, new_cx))
                    positions[cref] = (new_cx, cy, crot)
                else:
                    shift = overlap_y + 0.5
                    new_cy = cy + shift if cy > yy else cy - shift
                    new_cy = max(min_y + 2, min(max_y - 2, new_cy))
                    positions[cref] = (cx, new_cy, crot)
                _log.info("Final crystal overlap fix: shifted %s away from %s",
                          cref, yref)


def _phase_late_decoupling(ctx: PlacementContext) -> None:
    """3c-late: Re-pull decoupling caps close to ICs after all group phases."""
    _log.info("  3c-late: Late decoupling re-tightening")
    sc_list = list(ctx.subcircuits)
    bounds = ctx.bounds

    _ref_to_group: dict[str, str] = {}
    for feat in ctx.requirements.features:
        for comp in feat.components:
            r = comp.ref if hasattr(comp, "ref") else comp
            _ref_to_group[r] = feat.name

    _decoupling_pulled = 0
    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.DECOUPLING:
            continue
        ic_ref = sc.anchor_ref
        if ic_ref not in ctx.positions:
            continue
        ic_group = _ref_to_group.get(ic_ref, "")
        ix, iy, _irot = ctx.positions[ic_ref]
        iw, ih = ctx.fp_sizes.get(ic_ref, (5.0, 5.0))
        if _irot % 180 in (90.0, 270.0):
            iw, ih = ih, iw

        placed_count = 0
        for cap_ref in sc.refs:
            if cap_ref == ic_ref or not cap_ref.startswith("C"):
                continue
            if cap_ref not in ctx.positions:
                continue
            if cap_ref in ctx.power_group_fixed or cap_ref in ctx.fixed_refs:
                continue
            cap_group = _ref_to_group.get(cap_ref, "")
            if cap_group != ic_group:
                continue
            cx, cy, crot = ctx.positions[cap_ref]
            cw, ch = ctx.fp_sizes.get(cap_ref, (1.5, 1.0))
            edge_dist = _compute_edge_distance(cx, cy, cw, ch, ix, iy, iw, ih)
            if edge_dist <= 3.0:
                continue

            tx, ty = _cap_side_position(placed_count, ix, iy, iw, ih, cw, ch)
            tx = max(bounds[0] + 1.0, min(bounds[2] - 1.0, tx))
            ty = max(bounds[1] + 1.0, min(bounds[3] - 1.0, ty))
            # Respect placement constraints before committing move
            from kicad_pipeline.optimization.constraint_guard import respect_constraints
            tx, ty, crot = respect_constraints(cap_ref, tx, ty, crot, ctx)
            ctx.positions[cap_ref] = (tx, ty, crot)
            placed_count += 1
            _decoupling_pulled += 1

    _log.info("    3c-late: re-pulled %d decoupling caps", _decoupling_pulled)

    # Store ref_to_group on ctx for use by later phases
    ctx._ref_to_group = _ref_to_group  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 3i: Pad-facing rotation optimization
# ---------------------------------------------------------------------------

_CANDIDATE_ROTATIONS: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)


def _build_signal_ref_connections(
    ctx: PlacementContext,
) -> tuple[dict[str, object], dict[str, list[tuple[str, str, str]]]]:
    from kicad_pipeline.visualization.ratsnest import POWER_NETS

    net_conns: dict[str, list[tuple[str, str]]] = {}
    fp_map: dict[str, object] = {}
    for fp in ctx.initial_pcb.footprints:
        fp_map[fp.ref] = fp
        for pad in fp.pads:
            if pad.net_name and pad.net_name.upper() not in POWER_NETS:
                net_conns.setdefault(pad.net_name, []).append((fp.ref, pad.number))

    ref_connections: dict[str, list[tuple[str, str, str]]] = {}
    for _net_name, conns in net_conns.items():
        if len(conns) != 2:
            continue
        ref_a, pad_a = conns[0]
        ref_b, pad_b = conns[1]
        if ref_a == ref_b:
            continue
        ref_connections.setdefault(ref_a, []).append((pad_a, ref_b, pad_b))
        ref_connections.setdefault(ref_b, []).append((pad_b, ref_a, pad_a))

    return fp_map, ref_connections


def _score_candidate_rotation(
    fp: object,
    candidate_rot: float,
    cx: float,
    cy: float,
    connections: list[tuple[str, str, str]],
    positions: dict[str, tuple[float, float, float]],
    side_vectors: dict[object, tuple[float, float]],
) -> float:
    from kicad_pipeline.pcb.pin_map import CardinalSide, compute_pin_map

    pin_map = compute_pin_map(fp, candidate_rot)
    total_score = 0.0
    count = 0
    for my_pad, partner_ref, _partner_pad in connections:
        my_side = pin_map.side_for_pad(my_pad)
        if my_side is None or my_side == CardinalSide.CENTER:
            total_score += 1.0
            count += 1
            continue
        if partner_ref not in positions:
            continue
        px, py, _prot = positions[partner_ref]
        dx_dir = px - cx
        dy_dir = py - cy
        dist = math.sqrt(dx_dir * dx_dir + dy_dir * dy_dir)
        if dist < 0.1:
            total_score += 1.0
            count += 1
            continue
        dx_dir /= dist
        dy_dir /= dist
        va = side_vectors[my_side]
        dot = va[0] * dx_dir + va[1] * dy_dir
        total_score += (dot + 1.0) / 2.0
        count += 1
    return total_score / max(count, 1)


def _phase_pad_facing_optimization(ctx: PlacementContext) -> None:
    """3i: Optimize 2-pad passive rotations for pad-facing alignment.

    For each 2-pad passive (R, C, D) connected via signal nets, tries all
    four cardinal rotations and picks the one that maximizes the sum of
    pad-facing scores across all signal-net connections of that component.

    Only rotates components NOT in fixed_refs. Positions are unchanged;
    only the rotation component of ``ctx.positions[ref]`` is updated.
    """
    from kicad_pipeline.pcb.pin_map import CardinalSide

    _log.info("  3i: Pad-facing rotation optimization")

    fp_map, ref_connections = _build_signal_ref_connections(ctx)

    side_vectors: dict[CardinalSide, tuple[float, float]] = {
        CardinalSide.NORTH: (0.0, -1.0),
        CardinalSide.SOUTH: (0.0, 1.0),
        CardinalSide.EAST: (1.0, 0.0),
        CardinalSide.WEST: (-1.0, 0.0),
        CardinalSide.CENTER: (0.0, 0.0),
    }

    # Skip connectors, ICs, mounting holes — their rotation is
    # functionally significant.  Also skip power_group_fixed passives —
    # the power signal-flow phase (_phase_power_chain_flow) chose their
    # rotations to match signal flow direction and the offsets assume
    # those specific rotations for clearance.
    rotation_exempt = {
        ref for ref in ctx.positions
        if ref.startswith(("J", "U", "H", "K", "SW"))
    } | ctx.power_group_fixed

    optimized_count = 0
    for ref, connections in ref_connections.items():
        if ref in rotation_exempt or ref not in ctx.positions:
            continue
        fp = fp_map.get(ref)
        if fp is None:
            continue
        if len(fp.pads) != 2:
            continue
        if not any(ref.startswith(p) for p in ("R", "C", "D")):
            continue

        cx, cy, current_rot = ctx.positions[ref]
        best_rot = current_rot
        best_score = -1.0

        for candidate_rot in _CANDIDATE_ROTATIONS:
            avg_score = _score_candidate_rotation(
                fp, candidate_rot, cx, cy, connections, ctx.positions, side_vectors,
            )
            if avg_score > best_score:
                best_score = avg_score
                best_rot = candidate_rot

        if best_rot != current_rot:
            ctx.positions[ref] = (cx, cy, best_rot)
            optimized_count += 1

    _log.info("    3i: optimized rotation for %d passives", optimized_count)


def _rot_size(w: float, h: float, rot: float) -> tuple[float, float]:
    """Return (width, height) after applying rotation (90/270 swap)."""
    if abs(rot) % 180 in (90.0, 270.0):
        return h, w
    return w, h


def _find_tht_connector_refs(ctx: PlacementContext) -> set[str]:
    return {
        fp.ref for fp in ctx.initial_pcb.footprints
        if fp.ref.startswith("J") and any(pad.pad_type == "thru_hole" for pad in fp.pads)
    }


def _compute_edge_position(
    edge_name: str,
    cx: float,
    cy: float,
    w: float,
    h: float,
    bounds: tuple[float, float, float, float],
    margin: float,
) -> tuple[float, float, float]:
    min_x, min_y, max_x, max_y = bounds
    if edge_name in ("left", "right"):
        new_rot = 90.0 if w > h else 0.0
        new_rw, _ = _rot_size(w, h, new_rot)
        new_y = cy
        if edge_name == "left":
            new_x = min_x + margin + new_rw / 2.0
        else:
            new_x = max_x - margin - new_rw / 2.0
    else:
        new_rot = 90.0 if h > w else 0.0
        _, new_rh = _rot_size(w, h, new_rot)
        new_x = cx
        if edge_name == "top":
            new_y = min_y + margin + new_rh / 2.0
        else:
            new_y = max_y - margin - new_rh / 2.0
    return new_x, new_y, new_rot


def _has_placement_collision(
    ref: str,
    new_x: float,
    new_y: float,
    final_rw: float,
    final_rh: float,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> bool:
    gap = 0.3
    for other_ref, (ox, oy, orot) in positions.items():
        if other_ref == ref:
            continue
        ow, oh = fp_sizes.get(other_ref, (2.0, 2.0))
        orw, orh = _rot_size(ow, oh, orot)
        if (abs(new_x - ox) < (final_rw + orw) / 2.0 + gap
                and abs(new_y - oy) < (final_rh + orh) / 2.0 + gap):
            return True
    return False


def _move_connector_to_edge(
    ref: str,
    cx: float,
    cy: float,
    w: float,
    h: float,
    edges: list[tuple[float, str]],
    ctx: PlacementContext,
    margin: float,
) -> None:
    placed = False
    for _dist, edge_name in edges:
        new_x, new_y, new_rot = _compute_edge_position(
            edge_name, cx, cy, w, h, ctx.bounds, margin,
        )
        final_rw, final_rh = _rot_size(w, h, new_rot)
        if not _has_placement_collision(
            ref, new_x, new_y, final_rw, final_rh, ctx.positions, ctx.fp_sizes,
        ):
            _log.info("    %s -> %s edge (%.1f,%.1f) rot=%.0f",
                      ref, edge_name, new_x, new_y, new_rot)
            ctx.positions[ref] = (new_x, new_y, new_rot)
            placed = True
            break

    if not placed:
        _dist, edge_name = edges[0]
        new_x, new_y, new_rot = _compute_edge_position(
            edge_name, cx, cy, w, h, ctx.bounds, margin,
        )
        _log.info("    %s -> %s edge (%.1f,%.1f) FORCED (collision expected)",
                  ref, edge_name, new_x, new_y)
        ctx.positions[ref] = (new_x, new_y, new_rot)


def _enforce_tht_connectors_to_edge(ctx: PlacementContext) -> None:
    """Push THT connectors (J*) to their nearest board edge after collision resolution.

    Through-hole connectors need board-edge access for soldering and wire entry.
    The collision resolver may have pushed them inward, so this re-enforces
    proximity using the same body-edge distance metric as the review agent.

    For each THT connector exceeding ``CONNECTOR_EDGE_MAX_MM``:
    1. Compute body-edge distance to all four board edges.
    2. Try the nearest edge first -- place connector with margin.
    3. Orient the long axis parallel to the target edge.
    4. If the new position collides, try the next-nearest edge.
    """
    from kicad_pipeline.constants import CONNECTOR_EDGE_MAX_MM

    min_x, min_y, max_x, max_y = ctx.bounds
    margin = 4.0  # pad center inset from board edge

    tht_connector_refs = _find_tht_connector_refs(ctx)
    if not tht_connector_refs:
        return

    was_fixed = tht_connector_refs & ctx.fixed_refs
    ctx.fixed_refs -= was_fixed

    for ref in sorted(tht_connector_refs):
        if ref not in ctx.positions or ref in ctx.fixed_refs:
            continue

        cx, cy, rot = ctx.positions[ref]
        w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        rw, rh = _rot_size(w, h, rot)

        d_left = (cx - rw / 2.0) - min_x
        d_right = max_x - (cx + rw / 2.0)
        d_top = (cy - rh / 2.0) - min_y
        d_bottom = max_y - (cy + rh / 2.0)
        body_edge_dist = min(d_left, d_right, d_top, d_bottom)

        _log.debug(
            "  THT check: %s at (%.1f,%.1f) rot=%.0f size=(%.1f,%.1f) "
            "rsize=(%.1f,%.1f) body_edge=%.1f bounds=%s",
            ref, cx, cy, rot, w, h, rw, rh, body_edge_dist, ctx.bounds,
        )

        if body_edge_dist <= CONNECTOR_EDGE_MAX_MM:
            continue

        _log.info(
            "  Post-3g THT edge: %s body %.1fmm from edge (max %.1f)",
            ref, body_edge_dist, CONNECTOR_EDGE_MAX_MM,
        )

        edges: list[tuple[float, str]] = sorted(
            [(d_left, "left"), (d_right, "right"), (d_top, "top"), (d_bottom, "bottom")],
            key=lambda e: e[0],
        )
        _move_connector_to_edge(ref, cx, cy, w, h, edges, ctx, margin)

    ctx.fixed_refs |= was_fixed


def _phase_collision_resolution(ctx: PlacementContext) -> None:
    """3g: Collision resolution (group-constrained, then unconstrained)."""
    _log.info("  3g: Collision resolution")
    group_bboxes = _extract_group_bboxes(ctx.requirements, ctx.positions, ctx.fp_sizes)
    subcircuit_fixed = (ctx.relay_support_refs | ctx.adc_channel_refs
                        | ctx.mcu_peripheral_refs | ctx.power_group_fixed
                        | ctx.ethernet_fixed | ctx.template_fixed
                        | ctx.top_edge_connector_refs)
    relay_fixed = ctx.fixed_refs | subcircuit_fixed | {
        r for r in ctx.positions if r.startswith("K")
    }
    ctx.positions = _resolve_collisions(
        ctx.positions, ctx.fp_sizes, ctx.bounds, relay_fixed, group_bboxes=group_bboxes,
    )
    # Targeted final pass
    remaining_collisions = _count_collisions(ctx.positions, ctx.fp_sizes)
    if remaining_collisions:
        _log.info("  3g: %d remaining — targeted pass", len(remaining_collisions))
        colliding_refs = set()
        for a, b in remaining_collisions:
            colliding_refs.add(a)
            colliding_refs.add(b)
        always_fixed_base = (ctx.mcu_peripheral_refs | ctx.top_edge_connector_refs
                             | ctx.ethernet_fixed | ctx.adc_channel_refs
                             | ctx.adc_ic_refs | ctx.relay_support_refs
                             | ctx.power_group_fixed
                             | {r for r in ctx.positions if r.startswith("K")})
        intra_fixed_unprotect: set[str] = set()
        for a, b in remaining_collisions:
            if a in always_fixed_base and b in always_fixed_base:
                area_a = ctx.fp_sizes.get(a, (2, 2))[0] * ctx.fp_sizes.get(a, (2, 2))[1]
                area_b = ctx.fp_sizes.get(b, (2, 2))[0] * ctx.fp_sizes.get(b, (2, 2))[1]
                smaller = a if area_a <= area_b else b
                if (not smaller.startswith(("U", "Y", "K", "Q"))
                        and smaller not in ctx.power_group_fixed):
                    intra_fixed_unprotect.add(smaller)
        always_fixed = always_fixed_base - intra_fixed_unprotect
        targeted_fixed = ctx.fixed_refs | (subcircuit_fixed - colliding_refs) | always_fixed
        ctx.positions = _resolve_collisions(
            ctx.positions, ctx.fp_sizes, ctx.bounds, targeted_fixed,
        )

    # Post-3g: Enforce ethernet connectors on bottom edge
    max_y = ctx.bounds[3]
    for ref in ctx.ethernet_fixed:
        if ref.startswith("J") and ref in ctx.positions and ref not in ctx.fixed_refs:
            w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
            bottom_target = max_y - h / 2.0 - 1.0
            rx, ry, rot = ctx.positions[ref]
            if ry < bottom_target - 5.0:
                _log.info("  Enforcing %s to bottom edge: y %.1f -> %.1f", ref, ry, bottom_target)
                ctx.positions[ref] = (rx, bottom_target, 180.0)

    # Post-3g: Enforce THT connectors to nearest board edge
    _enforce_tht_connectors_to_edge(ctx)


def _phase_first_clamp(ctx: PlacementContext) -> None:
    """First board-edge clamp using pad extent."""
    _log.info("=== Final: Clamping and review ===")
    fp_lookup = {fp.ref: fp for fp in ctx.initial_pcb.footprints}
    _edge_m = 1.5

    _clamp_all_positions(
        ctx.positions, fp_lookup, ctx.fp_sizes, ctx.bounds, _edge_m,
        skip_refs=ctx.fixed_refs, label="First clamp",
    )

    # Post-clamp collision resolution
    subcircuit_fixed = _build_subcircuit_fixed(ctx)
    ctx.positions = _resolve_post_phase_collisions(
        ctx.positions, ctx.fp_sizes, ctx.bounds, ctx, subcircuit_fixed,
        label="post-clamp",
    )

    # Post-clamp decoupling re-pull
    _post_clamp_decoupling_repull(ctx)

    # Store fp_lookup and _edge_m on ctx for use by later phases
    ctx._fp_lookup = fp_lookup  # type: ignore[attr-defined]
    ctx._edge_m = _edge_m  # type: ignore[attr-defined]


def _phase_review_loop(ctx: PlacementContext) -> None:
    """EE Review loop — run review_placement and apply fixes."""
    from kicad_pipeline.optimization.functional_grouper import (
        classify_voltage_domains,
    )
    from kicad_pipeline.optimization.review_agent import review_placement

    _log.info("  Running EE review (max %d passes)", ctx.max_review_passes)
    subcircuits = ctx.subcircuits
    domain_map = classify_voltage_domains(ctx.requirements)

    best_positions = dict(ctx.positions)
    best_review: PlacementReview | None = None
    best_violation_count = float("inf")

    subcircuit_fixed = _build_subcircuit_fixed(ctx)

    for pass_num in range(ctx.max_review_passes):
        positions_tuple = _dict_to_positions(ctx.positions)
        current_pcb = _apply_positions(ctx.initial_pcb, positions_tuple)
        review = review_placement(
            current_pcb, ctx.requirements,
            subcircuits=subcircuits, domain_map=domain_map,
        )

        critical_major = sum(
            1 for v in review.violations
            if v.severity in ("critical", "major")
        )
        _log.info(
            "  Review pass %d: %s -- %d critical/major",
            pass_num + 1, review.summary, critical_major,
        )

        if critical_major < best_violation_count:
            best_violation_count = critical_major
            best_positions = dict(ctx.positions)
            best_review = review

        if critical_major == 0:
            break

        relay_fixed_review = ctx.fixed_refs | subcircuit_fixed | {
            r for r in ctx.positions if r.startswith("K")
        }
        ctx.positions = _apply_review_fixes(
            ctx.positions, review, relay_fixed_review, ctx.fp_sizes, ctx.bounds,
        )

    # Post-review collision resolution
    best_positions = _resolve_post_phase_collisions(
        best_positions, ctx.fp_sizes, ctx.bounds, ctx, subcircuit_fixed,
        label="post-review",
    )

    ctx.best_positions = best_positions
    ctx._best_review = best_review  # type: ignore[attr-defined]
    ctx._domain_map = domain_map  # type: ignore[attr-defined]


def _build_r_top_connector_x_map(
    requirements: ProjectRequirements,
    best_positions: dict[str, tuple[float, float, float]],
) -> dict[str, float]:
    """Map R-prefixed refs to their connected connector's X position."""
    r_top_x: dict[str, float] = {}
    for net in requirements.nets:
        j_conns = [c for c in net.connections if c.ref.startswith("J")]
        r_conns = [c for c in net.connections
                   if c.ref.startswith("R") and c.ref in best_positions]
        if not (j_conns and r_conns):
            continue
        for j_conn in j_conns:
            j_pos = best_positions.get(j_conn.ref)
            if j_pos:
                for r_conn in r_conns:
                    r_top_x[r_conn.ref] = j_pos[0]
    return r_top_x


def _build_adc_strip_order(passives: list[str]) -> list[str]:
    """Build the vertical strip order for an ADC channel's passives."""
    r_refs = sorted(r for r in passives if r.startswith("R"))
    d_refs = [r for r in passives if r.startswith("D")]
    c_refs = [r for r in passives if r.startswith("C")]
    strip_order: list[str] = []
    if len(r_refs) >= 1:
        strip_order.append(r_refs[0])
    strip_order.extend(c_refs)
    strip_order.extend(d_refs)
    if len(r_refs) >= 2:
        strip_order.append(r_refs[1])
    return strip_order


def _place_adc_strip(
    strip_order: list[str],
    ch_x: float,
    ch_y_top: float,
    strip_gap: float,
    best_positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> int:
    """Place a single ADC channel's passive strip vertically. Returns realigned count."""
    realigned = 0
    strip_y = ch_y_top
    for ref in strip_order:
        if ref not in best_positions:
            continue
        _raw_w, raw_h = fp_sizes.get(ref, (2.0, 2.0))
        target_x = max(bounds[0] + 2.0, min(bounds[2] - 2.0, ch_x))
        target_y = max(bounds[1] + 2.0, min(bounds[3] - 2.0, strip_y + raw_h / 2.0))
        old_x, old_y, _old_rot = best_positions[ref]
        if abs(old_x - target_x) > 1.0 or abs(old_y - target_y) > 1.0:
            realigned += 1
        best_positions[ref] = (target_x, target_y, 0.0)
        strip_y = target_y + raw_h / 2.0 + strip_gap
    return realigned


def _sort_adc_channels_by_connector_x(
    adc_channels: list[tuple[str, str, list[str]]],
    r_top_x: dict[str, float],
) -> list[tuple[float, str, str, list[str]]]:
    result: list[tuple[float, str, str, list[str]]] = []
    for ic_ref, ic_pin, passives in adc_channels:
        conn_x = 999.0
        for r in sorted(r for r in passives if r.startswith("R")):
            if r in r_top_x:
                conn_x = r_top_x[r]
                break
        result.append((conn_x, ic_ref, ic_pin, passives))
    result.sort(key=lambda t: t[0])
    return result


def _compute_adc_zone_layout(
    ctx: PlacementContext,
    n_channels: int,
    channel_spacing_mm: float,
) -> tuple[float, float, float]:
    bounds = ctx.bounds
    _az = next((z for z in ctx.zones if z.name == "analog"), None)
    az_x1, az_y1, az_x2, _az_y2 = _az.rect if _az else bounds

    ch_zone_width = az_x2 - az_x1 - 4.0
    ch_spacing = min(channel_spacing_mm, ch_zone_width / max(n_channels - 1, 1))
    total_ch_width = (n_channels - 1) * ch_spacing
    ch_x_start = az_x1 + 2.0 + (ch_zone_width - total_ch_width) / 2.0

    _gap_mm = 2.0
    _connector_bottoms: list[float] = []
    for _jref in ctx.adc_channel_refs:
        if not _jref.startswith("J") or _jref not in ctx.best_positions:
            continue
        _jx, _jy, _jrot = ctx.best_positions[_jref]
        _jw, _jh = ctx.fp_sizes.get(_jref, (5.0, 5.0))
        if _jrot % 180 in (90.0, 270.0):
            _jw, _jh = _jh, _jw
        _connector_bottoms.append(_jy + _jh / 2.0)
    ch_y_top = max(_connector_bottoms) + _gap_mm if _connector_bottoms else az_y1 + 2.0
    return ch_x_start, ch_spacing, ch_y_top


def _reposition_adc_ics(
    ic_ch_xs: dict[str, list[float]],
    ch_y_top: float,
    bounds: tuple[float, float, float, float],
    best_positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> None:
    for ic_ref, ch_xs in ic_ch_xs.items():
        if ic_ref not in best_positions:
            continue
        ic_new_x = sum(ch_xs) / len(ch_xs)
        ic_new_y = ch_y_top + 22.0
        iw, ih = fp_sizes.get(ic_ref, (5.0, 5.0))
        ic_new_x = max(bounds[0] + iw / 2, min(bounds[2] - iw / 2, ic_new_x))
        ic_new_y = max(bounds[1] + ih / 2, min(bounds[3] - ih / 2, ic_new_y))
        best_positions[ic_ref] = (ic_new_x, ic_new_y, 0.0)
        _log.info(
            "    3c2-late: %s -> (%.1f, %.1f) center of %d channels",
            ic_ref, ic_new_x, ic_new_y, len(ch_xs),
        )


def _phase_late_adc_realignment(ctx: PlacementContext) -> None:
    """3c2-late: Post-collision ADC channel re-alignment."""
    adc_channels: list[tuple[str, str, list[str]]] = getattr(ctx, "_adc_channels", [])
    strip_gap_mm: float = getattr(ctx, "_STRIP_GAP_MM", 1.5)
    channel_spacing_mm: float = getattr(ctx, "_CHANNEL_SPACING_MM", 8.0)

    if not (adc_channels and ctx.adc_ic_refs):
        return

    _log.info("  3c2-late: ADC channel re-alignment (connector-ordered)")

    late_r_top_x = _build_r_top_connector_x_map(ctx.requirements, ctx.best_positions)
    all_ch_with_x = _sort_adc_channels_by_connector_x(adc_channels, late_r_top_x)

    ch_x_start, ch_spacing, ch_y_top = _compute_adc_zone_layout(
        ctx, len(all_ch_with_x), channel_spacing_mm,
    )

    _realigned = 0
    _ic_ch_xs: dict[str, list[float]] = {}

    for ch_idx, (conn_x, ic_ref, ic_pin, passives) in enumerate(all_ch_with_x):
        ch_x = ch_x_start + ch_idx * ch_spacing
        _ic_ch_xs.setdefault(ic_ref, []).append(ch_x)
        strip_order = _build_adc_strip_order(passives)
        _realigned += _place_adc_strip(
            strip_order, ch_x, ch_y_top, strip_gap_mm,
            ctx.best_positions, ctx.fp_sizes, ctx.bounds,
        )
        _log.info(
            "    3c2-late: ch%d (%s.%s) -> x=%.1f (conn_x=%.1f)",
            ch_idx, ic_ref, ic_pin, ch_x, conn_x,
        )

    _reposition_adc_ics(_ic_ch_xs, ch_y_top, ctx.bounds, ctx.best_positions, ctx.fp_sizes)
    _log.info("    3c2-late: re-aligned %d ADC channel components", _realigned)

    subcircuit_fixed = _build_subcircuit_fixed(ctx)
    ctx.best_positions = _resolve_post_phase_collisions(
        ctx.best_positions, ctx.fp_sizes, ctx.bounds, ctx, subcircuit_fixed,
        label="3c2-late post-alignment",
    )


def _place_grid_below_anchor(
    refs: list[str],
    anchor_x: float,
    anchor_w: float,
    start_y: float,
    cols_per_row: int,
    bounds: tuple[float, float, float, float],
    fp_sizes: dict[str, tuple[float, float]],
    positions: dict[str, tuple[float, float, float]],
) -> tuple[int, float, int, float]:
    """Place refs in a grid below an anchor component.

    Returns:
        Tuple of (realigned_count, current_row_y, current_col, current_row_max_h).
    """
    realigned = 0
    col = 0
    row_y = start_y
    row_max_h = 0.0
    for ref in refs:
        _w, h = fp_sizes.get(ref, (2.0, 2.0))
        px = anchor_x - anchor_w / 2.0 + (col + 0.5) * (anchor_w / cols_per_row)
        py = row_y + h / 2.0
        px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
        py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
        old_x, old_y, old_rot = positions[ref]
        if abs(old_x - px) > 1.0 or abs(old_y - py) > 1.0:
            realigned += 1
        positions[ref] = (px, py, old_rot)
        row_max_h = max(row_max_h, h)
        col += 1
        if col >= cols_per_row:
            col = 0
            row_y += row_max_h + 0.5
            row_max_h = 0.0
    return realigned, row_y, col, row_max_h


def _collect_all_relay_refs(
    ctx: PlacementContext,
    sc_list: list[object],
    relay_leds: dict[str, list[str]],
) -> set[str]:
    """Build the complete set of relay-related refs across all relay driver subcircuits."""
    all_refs: set[str] = set()
    for sc in sc_list:
        if sc.circuit_type == SubCircuitType.RELAY_DRIVER:  # type: ignore[union-attr]
            all_refs.update(sc.refs)  # type: ignore[union-attr]
            all_refs.update(relay_leds.get(sc.anchor_ref, []))  # type: ignore[union-attr]
    all_refs.update(r for r in ctx.best_positions if r.startswith("K"))
    return all_refs


def _classify_relay_support_members(
    support_members: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split support members into (q_refs, d_refs, all_r_refs, other_refs)."""
    q_refs = sorted(r for r in support_members if r.startswith("Q"))
    d_refs = sorted(r for r in support_members if r.startswith("D"))
    all_r_refs = sorted(r for r in support_members if r.startswith("R"))
    other_refs = sorted(
        r for r in support_members
        if not r.startswith("Q") and not r.startswith("D") and not r.startswith("R")
    )
    return q_refs, d_refs, all_r_refs, other_refs


_RELAY_POWER_NETS = frozenset({"GND", "+5V", "+5V_RELAY", "+5V_LOGIC", "VCC"})


def _separate_gate_resistors(
    all_r_refs: list[str],
    q_refs: list[str],
    ctx: PlacementContext,
) -> tuple[list[str], list[str]]:
    """Split R refs into gate resistors (share non-power net with Q) and others."""
    r_gate: list[str] = []
    r_other: list[str] = []
    for r_ref in all_r_refs:
        shares_net = False
        for net in ctx.requirements.nets:
            if net.name.upper() in _RELAY_POWER_NETS:
                continue
            r_in = any(c.ref == r_ref for c in net.connections)
            q_in = any(c.ref in q_refs for c in net.connections)
            if r_in and q_in:
                shares_net = True
                break
        if shares_net:
            r_gate.append(r_ref)
        else:
            r_other.append(r_ref)
    return r_gate, r_other


def _find_extra_gate_resistors(
    q_refs: list[str],
    r_gate: list[str],
    r_other: list[str],
    ctx: PlacementContext,
) -> list[str]:
    """Find gate resistors outside the subcircuit that share a DRIVE net with Q."""
    extra: list[str] = []
    for net in ctx.requirements.nets:
        if net.name.upper() in _RELAY_POWER_NETS:
            continue
        if "DRIVE" not in net.name.upper():
            continue
        if not any(c.ref in q_refs for c in net.connections):
            continue
        for conn in net.connections:
            if (conn.ref.startswith("R")
                    and conn.ref not in r_gate
                    and conn.ref not in r_other
                    and conn.ref in ctx.best_positions):
                extra.append(conn.ref)
    return extra


def _place_two_column_ref(
    ref: str,
    col_x: float,
    row_y: float,
    rot: float,
    bounds: tuple[float, float, float, float],
    positions: dict[str, tuple[float, float, float]],
) -> int:
    """Place *ref* at (col_x, row_y, rot) clamped to bounds. Returns 1 if moved."""
    px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, col_x))
    py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, row_y))
    old_x, old_y, _ = positions[ref]
    positions[ref] = (px, py, rot)
    return 1 if abs(old_x - px) > 1.0 or abs(old_y - py) > 1.0 else 0


def _place_relay_driver_columns(
    sc: object,
    relay_leds: dict[str, list[str]],
    ctx: PlacementContext,
) -> int:
    """Place D, Q, R, other, and LED members for one relay driver subcircuit.

    Uses size-aware cursor placement to prevent courtyard collisions
    between D (flyback diode) and Q (transistor) in the left column.

    Returns count of repositioned components.
    """
    anchor: str = sc.anchor_ref  # type: ignore[union-attr]
    if anchor not in ctx.best_positions:
        return 0

    kx, ky, krot = ctx.best_positions[anchor]
    kw, kh = ctx.fp_sizes.get(anchor, (18.0, 16.0))
    if krot % 180 in (90.0, 270.0):
        kw, kh = kh, kw
    _log.info(
        "    3b-late: %s centroid=(%.1f,%.1f) rot=%.0f size=%.1fx%.1f",
        anchor, kx, ky, krot, kw, kh,
    )

    bounds = ctx.bounds
    support_members = [r for r in sc.refs if r != anchor and r in ctx.best_positions]  # type: ignore[union-attr]
    q_refs, d_refs, all_r_refs, other_refs = _classify_relay_support_members(support_members)
    r_gate, r_other = _separate_gate_resistors(all_r_refs, q_refs, ctx)
    r_gate.extend(_find_extra_gate_resistors(q_refs, r_gate, r_other, ctx))
    other_refs.extend(r_other)

    led_members = sorted(
        set(relay_leds.get(anchor, [])) & set(ctx.best_positions.keys())
    )

    base_left_x = kx - 4.3
    base_right_x = kx + 4.0
    moved = 0

    # Size-aware cursor placement for driver column (D then Q)
    # Gap between components must account for courtyard extents
    gap = 2.5  # mm clearance — SOD-323 + SOT-23 courtyards need >1.5mm

    # Anchor D at coil pin Y when available for minimal flyback loop area.
    # Place outside the relay body on the coil-pin side to avoid courtyard
    # collisions while minimising flyback loop area.
    from kicad_pipeline.optimization.ee_phases import _find_coil_pin_abs_pos
    coil_pos = _find_coil_pin_abs_pos(anchor, ctx, positions=ctx.best_positions)
    coil_y = coil_pos[1] if coil_pos is not None else None
    if coil_y is not None and coil_y < ky:
        # Coil pin above relay centre → place driver column above relay
        relay_top = ky - kh / 2.0
        driver_cursor_y = relay_top - gap
        direction = -1.0
    else:
        driver_cursor_y = ky + kh / 2.0 + gap
        direction = 1.0

    # Place D+Q on the coil-pin side, R on the opposite side
    if coil_pos is not None and coil_pos[0] > kx:
        dq_x = base_right_x   # D+Q on right (coil pin side)
        r_x = base_left_x     # R on left
    else:
        dq_x = base_left_x    # D+Q on left
        r_x = base_right_x    # R on right

    for d_ref in d_refs:
        _dw, dh = ctx.fp_sizes.get(d_ref, (2.0, 2.0))
        d_y = driver_cursor_y + direction * dh / 2.0
        moved += _place_two_column_ref(d_ref, dq_x, d_y, 0.0, bounds, ctx.best_positions)
        driver_cursor_y = d_y + direction * (dh / 2.0 + gap)

    for q_ref in q_refs:
        _qw, qh = ctx.fp_sizes.get(q_ref, (3.0, 3.0))
        q_y = driver_cursor_y + direction * qh / 2.0
        moved += _place_two_column_ref(q_ref, dq_x, q_y, 180.0, bounds, ctx.best_positions)
        driver_cursor_y = q_y + direction * (qh / 2.0 + gap)

    # R_gate column: same start cursor as driver column
    if coil_y is not None and coil_y < ky:
        relay_top = ky - kh / 2.0
        r_cursor_y = relay_top - gap
    else:
        r_cursor_y = ky + kh / 2.0 + gap
    for r_ref in r_gate:
        _rw, rh = ctx.fp_sizes.get(r_ref, (2.0, 2.0))
        r_y = r_cursor_y + direction * rh / 2.0
        moved += _place_two_column_ref(
            r_ref, r_x, r_y, 180.0, bounds, ctx.best_positions,
        )
        r_cursor_y = r_y + direction * (rh / 2.0 + gap)

    if other_refs:
        # Place others on the far side of the taller column (away from relay)
        if direction < 0:
            others_start_y = min(driver_cursor_y, r_cursor_y)
        else:
            others_start_y = max(driver_cursor_y, r_cursor_y)
        count_other, _, _, _ = _place_grid_below_anchor(
            other_refs, kx, kw, others_start_y, 2, bounds, ctx.fp_sizes, ctx.best_positions,
        )
        moved += count_other

    # When driver column is above relay, LEDs go below the relay body
    led_cursor_y = ky + kh / 2.0 + gap if direction < 0 else driver_cursor_y
    moved += _place_relay_led_members(
        led_members, dq_x, kx, kw, ky, bounds, ctx, led_cursor_y,
    )
    return moved


def _place_relay_led_members(
    led_members: list[str],
    left_x: float,
    kx: float,
    kw: float,
    ky: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
    cursor_y: float = 0.0,
) -> int:
    """Place LED resistors and diodes in the left column below Q. Returns moved count.

    Args:
        cursor_y: Y position of the bottom edge of the last placed component in the
            left column.  When > 0 the LED members are placed below this cursor
            with size-aware gaps; when 0 a fallback offset from *ky* is used.
    """
    led_r = sorted(r for r in led_members if r.startswith("R"))
    led_d = sorted(r for r in led_members if r.startswith("D"))
    led_other = sorted(r for r in led_members if not r.startswith("R") and not r.startswith("D"))
    gap = 2.0  # mm clearance between component edges
    moved = 0

    # Use cursor from left-column placement when available
    if cursor_y <= 0.0:
        kh = ctx.fp_sizes.get(
            next((r for r in ctx.best_positions if r.startswith("K")), ""), (18.0, 16.0),
        )[1]
        cursor_y = ky + kh / 2.0 + 14.0  # legacy fallback below driver column

    for ref in led_r:
        _rw, rh = ctx.fp_sizes.get(ref, (2.0, 2.0))
        py = cursor_y + rh / 2.0
        moved += _place_two_column_ref(ref, left_x, py, 0.0, bounds, ctx.best_positions)
        cursor_y = py + rh / 2.0 + gap

    for ref in led_d:
        _dw, dh = ctx.fp_sizes.get(ref, (2.0, 2.0))
        py = cursor_y + dh / 2.0
        moved += _place_two_column_ref(ref, left_x, py, 180.0, bounds, ctx.best_positions)
        cursor_y = py + dh / 2.0 + gap

    if led_other:
        count_led, _, _, _ = _place_grid_below_anchor(
            led_other, kx, kw, cursor_y, 2, bounds, ctx.fp_sizes, ctx.best_positions,
        )
        moved += count_led
    return moved


def _place_relay_terminal_connectors(
    ctx: PlacementContext,
    all_relay_refs: set[str],
) -> int:
    """Align relay terminal connectors (J) to their relay X-coordinate. Returns moved count."""
    from kicad_pipeline.optimization.ee_phases import _build_connector_to_relay_map

    conn_to_relay = _build_connector_to_relay_map(ctx.requirements)
    min_x, min_y, max_x, max_y = ctx.bounds
    terminal_y = min_y + 5.0
    moved = 0
    for j_ref, k_ref in conn_to_relay.items():
        if j_ref not in ctx.best_positions or k_ref not in ctx.best_positions:
            continue
        kx_late, _, _ = ctx.best_positions[k_ref]
        old_jx, old_jy, _ = ctx.best_positions[j_ref]
        px = max(min_x + 2.0, min(max_x - 2.0, kx_late))
        py = max(min_y + 2.0, min(max_y - 2.0, terminal_y))
        if abs(old_jx - px) > 1.0 or abs(old_jy - py) > 1.0:
            moved += 1
        # Relay terminals sit near top edge — wire entry faces outward (rot=0)
        ctx.best_positions[j_ref] = (px, py, 0.0)
        all_relay_refs.add(j_ref)
    return moved


def _resolve_relay_post_alignment_collisions(
    ctx: PlacementContext,
    all_relay_refs: set[str],
    n_realigned: int,
) -> None:
    """Resolve collisions after relay realignment if any components were moved."""
    if not n_realigned:
        return
    _log.info("    3b-late: re-aligned %d relay support components", n_realigned)
    post_collisions = _count_collisions(ctx.best_positions, ctx.fp_sizes)
    if not post_collisions:
        return
    _log.info("    3b-late: %d post-alignment collisions — resolving", len(post_collisions))
    relay_fixed = (
        ctx.fixed_refs
        | all_relay_refs
        | ctx.top_edge_connector_refs
        | ctx.adc_channel_refs
        | ctx.adc_ic_refs
        | ctx.power_group_fixed
    )
    ctx.best_positions = _resolve_collisions(
        ctx.best_positions, ctx.fp_sizes, ctx.bounds, relay_fixed,
    )


def _phase_late_relay_realignment(
    ctx: PlacementContext,
    _relay_leds: dict[str, list[str]],
) -> None:
    """3b-late: Post-collision relay driver re-alignment (two-column).

    Re-applies the pad-connectivity-driven two-column layout after collision
    resolution may have displaced components.

    LEFT column (dx ~ -4.3mm from K.x) — size-aware cursor placement:
      D_flyback below relay bottom edge + gap, rot=0
      Q below D + gap, rot=180
      R_LED below Q + gap, rot=0
      D_LED below R_LED + gap, rot=180

    RIGHT column (dx ~ +4.0mm from K.x):
      R_gate at relay bottom edge + gap, rot=180

    All Y offsets are computed from actual component sizes (fp_sizes)
    with 2mm edge-to-edge clearance to prevent courtyard collisions.
    """
    _log.info("  3b-late: Relay driver re-alignment (two-column)")
    sc_list = list(ctx.subcircuits)
    all_relay_refs = _collect_all_relay_refs(ctx, sc_list, _relay_leds)

    relay_realigned = 0
    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.RELAY_DRIVER:
            continue
        relay_realigned += _place_relay_driver_columns(sc, _relay_leds, ctx)

    relay_realigned += _place_relay_terminal_connectors(ctx, all_relay_refs)
    _resolve_relay_post_alignment_collisions(ctx, all_relay_refs, relay_realigned)


def _phase_mcu_decoupling_repull(ctx: PlacementContext) -> None:
    """MCU decoupling re-pull — re-pull caps tight against U3's LEFT side."""
    from kicad_pipeline.optimization.functional_grouper import _find_mcu_ref as _find_mcu
    mcu_ref_c3 = _find_mcu(ctx.requirements)
    if not (mcu_ref_c3 and mcu_ref_c3 in ctx.best_positions):
        return

    bounds = ctx.bounds
    _mcu_fx, _mcu_fy, _mcu_fr = ctx.best_positions[mcu_ref_c3]
    _mcu_fw, _mcu_fh = ctx.fp_sizes.get(mcu_ref_c3, (19.5, 25.4))
    if _mcu_fr % 180 in (90.0, 270.0):
        _mcu_fw, _mcu_fh = _mcu_fh, _mcu_fw
    _mcu_left = _mcu_fx - _mcu_fw / 2.0
    _mcu_decoup_pulled = 0
    _mcu_decoup_refs = sorted(
        r for r in ctx.best_positions
        if r.startswith("C") and r in ctx.mcu_peripheral_refs
    )
    _cap_x = _mcu_left - 3.0
    _cap_y_start = _mcu_fy - (_mcu_fh / 3.0)
    _cap_y = _cap_y_start
    for ref in _mcu_decoup_refs:
        cx, cy, crot = ctx.best_positions[ref]
        cw, ch = ctx.fp_sizes.get(ref, (2.5, 1.5))
        dx_edge = max(0.0, abs(cx - _mcu_fx) - (_mcu_fw + cw) / 2.0)
        dy_edge = max(0.0, abs(cy - _mcu_fy) - (_mcu_fh + ch) / 2.0)
        edge_dist = (dx_edge ** 2 + dy_edge ** 2) ** 0.5
        if edge_dist > 5.0:
            tx = _cap_x
            ty = _cap_y
            _cap_y += ch + 1.0
            tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
            ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
            ctx.best_positions[ref] = (tx, ty, crot)
            _mcu_decoup_pulled += 1
    if _mcu_decoup_pulled:
        _log.info("MCU decoupling re-pull: %d caps to U3 left side",
                  _mcu_decoup_pulled)


def _phase_final_clamp(ctx: PlacementContext) -> None:
    """Final board-edge clamp — ensure ALL components are inside board."""
    fp_lookup: dict[str, object] = getattr(ctx, "_fp_lookup", {})
    if not fp_lookup:
        fp_lookup = {fp.ref: fp for fp in ctx.initial_pcb.footprints}
    _edge_m: float = getattr(ctx, "_edge_m", 1.5)

    # Late-phase edge clamp
    _clamp_all_positions(
        ctx.best_positions, fp_lookup, ctx.fp_sizes, ctx.bounds, _edge_m,
        label="Late-phase clamp",
    )

    # Final crystal-cap overlap resolution
    _resolve_crystal_overlaps(ctx.best_positions, ctx.fp_sizes, ctx.bounds)

    # Final board-edge clamp
    _clamp_all_positions(
        ctx.best_positions, fp_lookup, ctx.fp_sizes, ctx.bounds, _edge_m,
        label="Final board-edge clamp",
    )


def _aabbs_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    min_gap: float,
) -> bool:
    """Return True if two AABBs overlap (or are within *min_gap* mm)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (
        ax0 - min_gap < bx1
        and ax1 + min_gap > bx0
        and ay0 - min_gap < by1
        and ay1 + min_gap > by0
    )


def _find_safe_shift(
    fp: Footprint,
    ox: float,
    oy: float,
    shift_x: float,
    shift_y: float,
    self_idx: int,
    aabbs: list[tuple[float, float, float, float] | None],
    min_gap: float,
    compute_aabb: Callable[[Footprint], tuple[float, float, float, float] | None],
) -> tuple[float, float]:
    """Binary-search for the largest fraction of (shift_x, shift_y) that is collision-free.

    Tests fractions from 1.0 down to 0.0 in ~10 steps.  Returns the
    largest safe (shift_x * f, shift_y * f) pair, or (0, 0) if even zero
    shift would collide (shouldn't happen -- zero shift is the original
    position).
    """
    from kicad_pipeline.models.pcb import Point

    best_f = 0.0
    # 10 binary-search iterations gives ~0.1% precision on the shift
    lo, hi = 0.0, 1.0
    for _ in range(10):
        mid = (lo + hi) / 2.0
        candidate_fp = replace(
            fp, position=Point(x=ox + shift_x * mid, y=oy + shift_y * mid),
        )
        candidate_box = compute_aabb(candidate_fp)
        if candidate_box is None:
            lo = mid
            best_f = mid
            continue
        collides = False
        for j, other_box in enumerate(aabbs):
            if j == self_idx or other_box is None:
                continue
            if _aabbs_overlap(candidate_box, other_box, min_gap):
                collides = True
                break
        if collides:
            hi = mid  # reduce shift
        else:
            lo = mid  # try more shift
            best_f = mid

    return shift_x * best_f, shift_y * best_f


def _post_apply_pad_extent_clamp(
    final_pcb: PCBDesign,
    bounds: tuple[float, float, float, float],
    edge_m: float,
) -> PCBDesign:
    """Clamp footprints so pad extents stay within board bounds.

    Collision-aware: before applying a clamp shift, checks whether the new
    position would overlap another footprint's AABB.  If so, reduces the
    shift to the largest amount that doesn't create a collision.  This
    prevents the Q2-on-D2 class of bugs at the source rather than relying
    on post-clamp nudging.
    """
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space
    from kicad_pipeline.validation.collisions import (
        _footprint_aabb as _compute_aabb,
    )

    min_x, min_y, max_x, max_y = bounds
    _min_clearance = 0.15  # mm — minimum gap to preserve between AABBs

    clamped = 0
    new_fps = list(final_pcb.footprints)

    # Pre-compute AABBs for all footprints (updated as we clamp)
    aabbs: list[tuple[float, float, float, float] | None] = [
        _compute_aabb(fp) for fp in new_fps
    ]

    for i, fp in enumerate(new_fps):
        if not fp.pads:
            continue
        ox, oy = fp.position.x, fp.position.y
        rot = fp.rotation
        px0, py0, px1, py1 = pad_extent_in_board_space(fp, ox, oy, rot)
        shift_x = shift_y = 0.0
        if px0 < min_x + edge_m:
            shift_x = (min_x + edge_m) - px0
        elif px1 > max_x - edge_m:
            shift_x = (max_x - edge_m) - px1
        if py0 < min_y + edge_m:
            shift_y = (min_y + edge_m) - py0
        elif py1 > max_y - edge_m:
            shift_y = (max_y - edge_m) - py1

        if shift_x == 0.0 and shift_y == 0.0:
            continue

        # --- Collision-aware shift reduction ---
        # Build a candidate AABB at the fully-shifted position and check
        # whether it would overlap any other footprint.
        candidate_fp = replace(fp, position=Point(x=ox + shift_x, y=oy + shift_y))
        candidate_box = _compute_aabb(candidate_fp)
        if candidate_box is not None:
            has_collision = False
            for j, other_box in enumerate(aabbs):
                if j == i or other_box is None:
                    continue
                if _aabbs_overlap(candidate_box, other_box, _min_clearance):
                    has_collision = True
                    break

            if has_collision:
                # Binary-search for the largest safe fraction of the shift
                safe_shift_x, safe_shift_y = _find_safe_shift(
                    fp, ox, oy, shift_x, shift_y,
                    i, aabbs, _min_clearance, _compute_aabb,
                )
                if safe_shift_x == 0.0 and safe_shift_y == 0.0:
                    _log.warning(
                        "  Pad-extent clamp %s: cannot shift (%.1f, %.1f) "
                        "without collision — skipping to avoid overlap",
                        fp.ref, shift_x, shift_y,
                    )
                    continue
                shift_x, shift_y = safe_shift_x, safe_shift_y

        new_fps[i] = replace(fp, position=Point(x=ox + shift_x, y=oy + shift_y))
        # Update the cached AABB so subsequent footprints see correct boxes
        aabbs[i] = _compute_aabb(new_fps[i])
        clamped += 1
        _log.info("  Post-apply clamp %s: shifted (%.1f, %.1f)", fp.ref, shift_x, shift_y)
    if clamped:
        final_pcb = replace(final_pcb, footprints=tuple(new_fps))
        _log.info("Post-apply board-edge clamp: %d components", clamped)

    # Post-clamp collision resolution: clamping can push components together
    from kicad_pipeline.validation.collisions import check_collisions
    violations = check_collisions(final_pcb, min_gap_mm=0.15)
    if violations:
        _log.info(
            "Post-clamp collisions detected: %d — nudging apart",
            len(violations),
        )
        new_fps2 = list(final_pcb.footprints)
        fp_by_ref = {fp.ref: i for i, fp in enumerate(new_fps2)}
        for v in violations:
            # Push the smaller component away from the larger one
            ia = fp_by_ref.get(v.ref_a)
            ib = fp_by_ref.get(v.ref_b)
            if ia is None or ib is None:
                continue
            fa, fb = new_fps2[ia], new_fps2[ib]
            area_a = sum(p.size_x * p.size_y for p in fa.pads)
            area_b = sum(p.size_x * p.size_y for p in fb.pads)
            # Move the smaller component
            if area_a <= area_b:
                mover_i, anchor = ia, fb
            else:
                mover_i, anchor = ib, fa
            mover = new_fps2[mover_i]
            dx = mover.position.x - anchor.position.x
            dy = mover.position.y - anchor.position.y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 0.01:
                dy = 1.0
                dist = 1.0
            # Push apart by the overlap amount + 0.5mm margin
            nudge = abs(v.gap_mm) + 0.5
            nx = mover.position.x + dx / dist * nudge
            ny = mover.position.y + dy / dist * nudge
            # Keep within bounds
            nx = max(min_x + edge_m, min(max_x - edge_m, nx))
            ny = max(min_y + edge_m, min(max_y - edge_m, ny))
            new_fps2[mover_i] = replace(
                mover, position=Point(x=nx, y=ny),
            )
            _log.info(
                "  Nudged %s away from %s by %.1fmm",
                mover.ref, anchor.ref, nudge,
            )
        final_pcb = replace(final_pcb, footprints=tuple(new_fps2))

    return final_pcb


def _filter_stale_violations(
    best_review: PlacementReview,
    final_pcb: PCBDesign,
    best_positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> PlacementReview:
    """Remove violations that no longer apply after final clamping."""
    from kicad_pipeline.optimization.review_agent import (
        PlacementReview as _PlacementReview,
    )
    from kicad_pipeline.optimization.review_agent import (
        PlacementRule,
        PlacementViolation,
        _check_board_edge_clearance,
        _compute_grade,
    )

    fresh_edge = _check_board_edge_clearance(final_pcb)
    fresh_edge_refs = {r for v in fresh_edge for r in v.refs}
    fresh_collision_pairs = {
        tuple(sorted((a, b))) for a, b in _count_collisions(best_positions, fp_sizes)
    }

    filtered: list[PlacementViolation] = []
    stale = 0
    for v in best_review.violations:
        if v.rule == PlacementRule.BOARD_EDGE_CLEARANCE:
            if not any(r in fresh_edge_refs for r in v.refs):
                stale += 1
                continue
            matched = next((fv for fv in fresh_edge if fv.refs == v.refs), v)
            filtered.append(matched)
        elif v.rule == PlacementRule.COLLISION:
            if tuple(sorted(v.refs)) not in fresh_collision_pairs:
                stale += 1
                continue
            filtered.append(v)
        else:
            filtered.append(v)

    if not stale:
        return best_review

    _log.info("Filtered %d stale edge clearance violations", stale)
    grade = _compute_grade(tuple(filtered))
    n_crit = sum(1 for v in filtered if v.severity == "critical")
    n_major = sum(1 for v in filtered if v.severity == "major")
    n_minor = sum(1 for v in filtered if v.severity == "minor")
    return _PlacementReview(
        violations=tuple(filtered),
        summary=f"Grade {grade}: {len(filtered)} violations "
                f"({n_crit} critical, {n_major} major, "
                f"{n_minor} minor)",
        grade=grade,
    )


def _find_rf_footprint(pcb: PCBDesign) -> object | None:
    _rf_kw = ("esp32", "wroom", "nina", "w5500", "wifi", "ble", "nrf52")
    return next(
        (fp for fp in pcb.footprints if any(kw in (fp.value or "").lower() for kw in _rf_kw)),
        None,
    )


def _filter_stale_antenna_keepouts(pcb: PCBDesign) -> list[object]:
    from kicad_pipeline.pcb.keepout_builder import (
        ANTENNA_KEEPOUT_HEIGHT_MM,
        ANTENNA_KEEPOUT_WIDTH_MM,
    )

    fresh_keepouts: list[object] = []
    for ko in pcb.keepouts:
        xs = [pt.x for pt in ko.polygon]
        ys = [pt.y for pt in ko.polygon]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        is_antenna_ko = (
            len(ko.polygon) == 4
            and w >= ANTENNA_KEEPOUT_WIDTH_MM - 1.0
            and h >= ANTENNA_KEEPOUT_HEIGHT_MM - 1.0
            and getattr(ko, "tag", "") != "mounting_hole"
        )
        if not is_antenna_ko:
            fresh_keepouts.append(ko)
    return fresh_keepouts


def _build_antenna_keepout_polygon(rf_fp: object) -> tuple[object, float, float]:
    import math as _math

    from kicad_pipeline.models.pcb import Keepout, Point

    esp32_body_w = 18.0
    esp32_body_h = 25.5
    antenna_ext = 3.5
    half_w = esp32_body_w / 2.0
    half_h = esp32_body_h / 2.0

    rot_rad = _math.radians(rf_fp.rotation)
    cos_r, sin_r = _math.cos(rot_rad), _math.sin(rot_rad)

    signal_ys = [p.position.y for p in rf_fp.pads if p.number not in ("41", "V1")]
    pad_min_local_y = min(signal_ys) if signal_ys else -half_h + 5.0
    pad_max_local_y = max(signal_ys) if signal_ys else half_h - 5.0

    fab_ys: list[float] = []
    for g in rf_fp.graphics:
        if hasattr(g, "start") and hasattr(g, "end"):
            fab_ys.extend([g.start.y, g.end.y])
    if fab_ys:
        min_fab_y, max_fab_y = min(fab_ys), max(fab_ys)
        antenna_at_min_y = (
            sum(1 for y in fab_ys if y < min_fab_y + 5.0)
            > sum(1 for y in fab_ys if y > max_fab_y - 5.0) * 1.5
        )
    else:
        antenna_at_min_y = True

    if antenna_at_min_y:
        ko_edge_local = pad_min_local_y - 0.5
        ko_far_local = -(half_h + antenna_ext)
    else:
        ko_edge_local = pad_max_local_y + 0.5
        ko_far_local = half_h + antenna_ext
    corners_local = [
        (-half_w, ko_edge_local), (half_w, ko_edge_local),
        (half_w, ko_far_local), (-half_w, ko_far_local),
    ]

    corners_board = [
        Point(
            rf_fp.position.x + lx * cos_r - ly * sin_r,
            rf_fp.position.y + lx * sin_r + ly * cos_r,
        )
        for lx, ly in corners_local
    ]
    new_ko = Keepout(
        polygon=tuple(corners_board),
        layers=("F.Cu", "B.Cu"),
        no_copper=True, no_vias=True, no_tracks=True,
    )
    return new_ko, esp32_body_w, abs(ko_far_local - ko_edge_local)


def _refresh_antenna_keepout(pcb: PCBDesign) -> PCBDesign:
    """Replace stale board-level antenna keepout with one based on final RF position.

    ``build_pcb()`` creates the board-level antenna keepout from the
    pre-optimisation footprint position.  After ``optimize_placement_ee``
    moves the RF module to its final location the stored keepout is stale.
    This function finds the RF module's actual placed position and rebuilds the
    keepout so it tracks the antenna end of the module correctly.
    """
    rf_fp = _find_rf_footprint(pcb)
    if rf_fp is None:
        return pcb

    fresh_keepouts = _filter_stale_antenna_keepouts(pcb)
    new_ko, ko_w, ko_h = _build_antenna_keepout_polygon(rf_fp)
    fresh_keepouts.append(new_ko)

    _log.info(
        "refresh_antenna_keepout: RF %s at (%.1f,%.1f,rot=%.0f), antenna keepout %.1fx%.1fmm",
        rf_fp.ref, rf_fp.position.x, rf_fp.position.y, rf_fp.rotation, ko_w, ko_h,
    )

    # Strip any stale antenna vias — keepout zone only, no via fence.
    gnd_net = 1
    fresh_vias = [
        v for v in pcb.vias
        if not (v.net_number == gnd_net and abs(v.drill - 0.6) < 0.01)
    ]

    _log.info(
        "refresh_antenna_keepout: updated keepout to final RF position, "
        "%d keepouts, %d vias retained",
        len(fresh_keepouts), len(fresh_vias),
    )

    return replace(
        pcb,
        keepouts=tuple(fresh_keepouts),
        vias=tuple(fresh_vias),
    )  # type: ignore[arg-type]


def _phase_build_final(
    ctx: PlacementContext,
) -> tuple[PCBDesign, PlacementReview]:
    """Build final PCB and filter stale violations."""
    from kicad_pipeline.optimization.placement_guard import validate_placement

    _edge_m: float = getattr(ctx, "_edge_m", 1.5)
    best_review: PlacementReview | None = getattr(ctx, "_best_review", None)
    domain_map = getattr(ctx, "_domain_map", {})

    # Build final PCB
    if best_review is None:
        from kicad_pipeline.optimization.functional_grouper import (
            classify_voltage_domains,
        )
        from kicad_pipeline.optimization.review_agent import review_placement
        positions_tuple = _dict_to_positions(ctx.best_positions)
        final_pcb = _apply_positions(ctx.initial_pcb, positions_tuple)
        domain_map = classify_voltage_domains(ctx.requirements)
        best_review = review_placement(
            final_pcb, ctx.requirements,
            subcircuits=ctx.subcircuits, domain_map=domain_map,
        )
    else:
        positions_tuple = _dict_to_positions(ctx.best_positions)
        final_pcb = _apply_positions(ctx.initial_pcb, positions_tuple)

    # Refresh the board-level antenna keepout position (no via fence).
    final_pcb = _refresh_antenna_keepout(final_pcb)

    final_pcb = _post_apply_pad_extent_clamp(final_pcb, ctx.bounds, _edge_m)

    if best_review is not None:
        best_review = _filter_stale_violations(
            best_review, final_pcb, ctx.best_positions, ctx.fp_sizes,
        )

    # Validation gate
    guard = validate_placement(final_pcb, ctx.requirements)
    if guard.issues:
        _log.warning("Placement guard issues (%d):", len(guard.issues))
        for issue in guard.issues:
            _log.warning("  %s", issue)
    else:
        _log.info("Placement guard: ALL CHECKS PASSED")

    _log.info("EE placement v5 complete: %s", best_review.summary)
    return final_pcb, best_review
