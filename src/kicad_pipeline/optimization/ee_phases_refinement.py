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
    from kicad_pipeline.models.pcb import PCBDesign
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
            if cap_ref in ctx.power_group_fixed:
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
            ctx.positions[cap_ref] = (tx, ty, crot)
            placed_count += 1
            _decoupling_pulled += 1

    _log.info("    3c-late: re-pulled %d decoupling caps", _decoupling_pulled)

    # Store ref_to_group on ctx for use by later phases
    ctx._ref_to_group = _ref_to_group  # type: ignore[attr-defined]


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


def _phase_late_adc_realignment(ctx: PlacementContext) -> None:
    """3c2-late: Post-collision ADC channel re-alignment."""
    adc_channels: list[tuple[str, str, list[str]]] = getattr(
        ctx, "_adc_channels", [],
    )
    _STRIP_GAP_MM: float = getattr(ctx, "_STRIP_GAP_MM", 1.5)
    _CHANNEL_SPACING_MM: float = getattr(ctx, "_CHANNEL_SPACING_MM", 8.0)

    if not (adc_channels and ctx.adc_ic_refs):
        return

    _log.info("  3c2-late: ADC channel re-alignment (connector-ordered)")
    bounds = ctx.bounds

    late_r_top_x = _build_r_top_connector_x_map(ctx.requirements, ctx.best_positions)

    _all_ch_with_x: list[tuple[float, str, str, list[str]]] = []
    for ic_ref, ic_pin, passives in adc_channels:
        conn_x = 999.0
        for r in sorted(r for r in passives if r.startswith("R")):
            if r in late_r_top_x:
                conn_x = late_r_top_x[r]
                break
        _all_ch_with_x.append((conn_x, ic_ref, ic_pin, passives))
    _all_ch_with_x.sort(key=lambda t: t[0])

    _az = None
    for z in ctx.zones:
        if z.name == "analog":
            _az = z
            break
    az_x1, az_y1, az_x2, _az_y2 = _az.rect if _az else bounds

    n_total_ch = len(_all_ch_with_x)
    ch_zone_width = az_x2 - az_x1 - 4.0
    ch_spacing = min(_CHANNEL_SPACING_MM, ch_zone_width / max(n_total_ch - 1, 1))
    total_ch_width = (n_total_ch - 1) * ch_spacing
    ch_x_start = az_x1 + 2.0 + (ch_zone_width - total_ch_width) / 2.0
    ch_y_top = az_y1 + 2.0

    _realigned = 0
    _ic_ch_xs: dict[str, list[float]] = {}

    for ch_idx, (conn_x, ic_ref, ic_pin, passives) in enumerate(_all_ch_with_x):
        ch_x = ch_x_start + ch_idx * ch_spacing
        _ic_ch_xs.setdefault(ic_ref, []).append(ch_x)

        strip_order = _build_adc_strip_order(passives)
        _realigned += _place_adc_strip(
            strip_order, ch_x, ch_y_top, _STRIP_GAP_MM,
            ctx.best_positions, ctx.fp_sizes, bounds,
        )
        _log.info(
            "    3c2-late: ch%d (%s.%s) -> x=%.1f (conn_x=%.1f)",
            ch_idx, ic_ref, ic_pin, ch_x, conn_x,
        )

    # Reposition ADC ICs below their channel groups
    for ic_ref, ch_xs in _ic_ch_xs.items():
        if ic_ref not in ctx.best_positions:
            continue
        ic_new_x = sum(ch_xs) / len(ch_xs)
        ic_new_y = ch_y_top + 22.0
        iw, ih = ctx.fp_sizes.get(ic_ref, (5.0, 5.0))
        ic_new_x = max(bounds[0] + iw / 2, min(bounds[2] - iw / 2, ic_new_x))
        ic_new_y = max(bounds[1] + ih / 2, min(bounds[3] - ih / 2, ic_new_y))
        ctx.best_positions[ic_ref] = (ic_new_x, ic_new_y, 0.0)
        _log.info(
            "    3c2-late: %s -> (%.1f, %.1f) center of %d channels",
            ic_ref, ic_new_x, ic_new_y, len(ch_xs),
        )

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


def _phase_late_relay_realignment(
    ctx: PlacementContext,
    _relay_leds: dict[str, list[str]],
) -> None:
    """3b-late: Post-collision relay driver re-alignment (two-column).

    Re-applies the pad-connectivity-driven two-column layout after collision
    resolution may have displaced components.

    LEFT column (dx ~ -4.3mm from K.x):
      D_flyback at dy=+10.8, rot=0
      Q at dy=+13.3, rot=180
      R_LED at dy=+15.5, rot=0
      D_LED at dy=+17.7, rot=180

    RIGHT column (dx ~ +4.0mm from K.x):
      R_gate at dy=+15.4, rot=180
    """
    _log.info("  3b-late: Relay driver re-alignment (two-column)")
    sc_list = list(ctx.subcircuits)
    bounds = ctx.bounds

    _all_relay_refs: set[str] = set()
    for sc in sc_list:
        if sc.circuit_type == SubCircuitType.RELAY_DRIVER:
            _all_relay_refs.update(sc.refs)
            _all_relay_refs.update(_relay_leds.get(sc.anchor_ref, []))
    _all_relay_refs.update(r for r in ctx.best_positions if r.startswith("K"))

    _relay_realigned = 0
    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.RELAY_DRIVER:
            continue
        anchor = sc.anchor_ref
        if anchor not in ctx.best_positions:
            continue
        kx, ky, krot = ctx.best_positions[anchor]
        kw, kh = ctx.fp_sizes.get(anchor, (18.0, 16.0))
        if krot % 180 in (90.0, 270.0):
            kw, kh = kh, kw
        _log.info("    3b-late: %s centroid=(%.1f,%.1f) rot=%.0f size=%.1fx%.1f",
                  anchor, kx, ky, krot, kw, kh)

        support_members = [
            r for r in sc.refs
            if r != anchor and r in ctx.best_positions
        ]

        # Classify by component type
        q_refs = sorted(r for r in support_members if r.startswith("Q"))
        d_refs = sorted(r for r in support_members if r.startswith("D"))
        all_r_refs = sorted(r for r in support_members if r.startswith("R"))
        other_refs = sorted(
            r for r in support_members
            if not r.startswith("Q") and not r.startswith("D") and not r.startswith("R")
        )

        # Separate gate resistors from LED resistors
        power_nets = {"GND", "+5V", "+5V_RELAY", "+5V_LOGIC", "VCC"}
        r_refs: list[str] = []
        r_other: list[str] = []
        for r_ref in all_r_refs:
            shares_net = False
            for net in ctx.requirements.nets:
                if net.name.upper() in power_nets:
                    continue
                r_in = any(c.ref == r_ref for c in net.connections)
                q_in = any(c.ref in q_refs for c in net.connections)
                if r_in and q_in:
                    shares_net = True
                    break
            if shares_net:
                r_refs.append(r_ref)
            else:
                r_other.append(r_ref)

        # Also find gate resistors NOT in subcircuit but sharing a DRIVE net with Q
        for net in ctx.requirements.nets:
            if net.name.upper() in power_nets:
                continue
            if "DRIVE" not in net.name.upper():
                continue
            q_in_net = any(c.ref in q_refs for c in net.connections)
            if not q_in_net:
                continue
            for conn in net.connections:
                if (conn.ref.startswith("R")
                        and conn.ref not in r_refs
                        and conn.ref not in r_other
                        and conn.ref in ctx.best_positions):
                    r_refs.append(conn.ref)

        other_refs.extend(r_other)

        led_members = sorted(
            set(_relay_leds.get(anchor, []))
            & set(ctx.best_positions.keys()),
        )

        # Two-column layout offsets (relative to K centroid)
        left_x = kx - 4.3
        right_x = kx + 4.0

        # LEFT column: D_flyback (anode facing down toward Q collector)
        for d_ref in d_refs:
            old_x, old_y, _old_rot = ctx.best_positions[d_ref]
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ky + 10.8))
            if abs(old_x - px) > 1.0 or abs(old_y - py) > 1.0:
                _relay_realigned += 1
            ctx.best_positions[d_ref] = (px, py, 0.0)

        # LEFT column: Q (collector up toward D, 180 deg)
        for q_ref in q_refs:
            old_x, old_y, _old_rot = ctx.best_positions[q_ref]
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ky + 13.3))
            if abs(old_x - px) > 1.0 or abs(old_y - py) > 1.0:
                _relay_realigned += 1
            ctx.best_positions[q_ref] = (px, py, 180.0)

        # RIGHT column: R_gate
        for r_ref in r_refs:
            old_x, old_y, _old_rot = ctx.best_positions[r_ref]
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, right_x))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ky + 15.4))
            if abs(old_x - px) > 1.0 or abs(old_y - py) > 1.0:
                _relay_realigned += 1
            ctx.best_positions[r_ref] = (px, py, 180.0)

        # Place other support refs in grid below
        if other_refs:
            other_y = ky + 18.0
            count_other, _, _, _ = _place_grid_below_anchor(
                other_refs, kx, kw, other_y, 2,
                bounds, ctx.fp_sizes, ctx.best_positions,
            )
            _relay_realigned += count_other

        # LED pairs in LEFT column below Q
        led_r = sorted(r for r in led_members if r.startswith("R"))
        led_d = sorted(r for r in led_members if r.startswith("D"))
        led_other = sorted(
            r for r in led_members
            if not r.startswith("R") and not r.startswith("D")
        )
        # R_LED: below Q, pad 1 facing up (rot=0)
        for ref in led_r:
            old_x, old_y, _old_rot = ctx.best_positions[ref]
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ky + 15.5))
            if abs(old_x - px) > 1.0 or abs(old_y - py) > 1.0:
                _relay_realigned += 1
            ctx.best_positions[ref] = (px, py, 0.0)
        # D_LED: below R_LED, anode facing up (rot=180)
        for ref in led_d:
            old_x, old_y, _old_rot = ctx.best_positions[ref]
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, left_x))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ky + 17.7))
            if abs(old_x - px) > 1.0 or abs(old_y - py) > 1.0:
                _relay_realigned += 1
            ctx.best_positions[ref] = (px, py, 180.0)
        if led_other:
            count_led, _, _, _ = _place_grid_below_anchor(
                led_other, kx, kw, ky + 19.5, 2,
                bounds, ctx.fp_sizes, ctx.best_positions,
            )
            _relay_realigned += count_led

    # Also re-align relay terminal connectors (J) to their relay X
    from kicad_pipeline.optimization.ee_phases import _build_connector_to_relay_map
    _conn_to_relay = _build_connector_to_relay_map(ctx.requirements)
    min_x, min_y, max_x, max_y = bounds
    terminal_y = min_y + 5.0
    for j_ref, k_ref in _conn_to_relay.items():
        if j_ref not in ctx.best_positions or k_ref not in ctx.best_positions:
            continue
        kx_late, _ky_late, _krot_late = ctx.best_positions[k_ref]
        old_jx, old_jy, _old_jrot = ctx.best_positions[j_ref]
        px = max(min_x + 2.0, min(max_x - 2.0, kx_late))
        py = max(min_y + 2.0, min(max_y - 2.0, terminal_y))
        if abs(old_jx - px) > 1.0 or abs(old_jy - py) > 1.0:
            _relay_realigned += 1
        ctx.best_positions[j_ref] = (px, py, 180.0)
        _all_relay_refs.add(j_ref)

    if _relay_realigned:
        _log.info("    3b-late: re-aligned %d relay support components", _relay_realigned)
        _post_relay_collisions = _count_collisions(ctx.best_positions, ctx.fp_sizes)
        if _post_relay_collisions:
            _log.info(
                "    3b-late: %d post-alignment collisions — resolving",
                len(_post_relay_collisions),
            )
            _relay_fixed = (ctx.fixed_refs
                            | _all_relay_refs
                            | ctx.top_edge_connector_refs
                            | ctx.adc_channel_refs | ctx.adc_ic_refs
                            | ctx.power_group_fixed)
            ctx.best_positions = _resolve_collisions(
                ctx.best_positions, ctx.fp_sizes, ctx.bounds, _relay_fixed,
            )


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


def _post_apply_pad_extent_clamp(
    final_pcb: PCBDesign,
    bounds: tuple[float, float, float, float],
    edge_m: float,
) -> PCBDesign:
    """Clamp footprints so pad extents stay within board bounds."""
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space

    min_x, min_y, max_x, max_y = bounds
    clamped = 0
    new_fps = list(final_pcb.footprints)
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
        if shift_x != 0.0 or shift_y != 0.0:
            new_fps[i] = replace(fp, position=Point(x=ox + shift_x, y=oy + shift_y))
            clamped += 1
            _log.info("  Post-apply clamp %s: shifted (%.1f, %.1f)", fp.ref, shift_x, shift_y)
    if clamped:
        final_pcb = replace(final_pcb, footprints=tuple(new_fps))
        _log.info("Post-apply board-edge clamp: %d components", clamped)
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
