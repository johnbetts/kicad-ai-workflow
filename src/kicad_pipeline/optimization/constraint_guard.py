"""Constraint guard — ensures optimizer phases respect placement constraints.

Every phase that moves a component should call ``respect_constraints()``
before committing the move.  Constraints represent explicit design intent
and must not be silently overridden by optimizer heuristics.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import PlacementContext

_log = logging.getLogger(__name__)


def respect_constraints(
    ref: str,
    new_x: float,
    new_y: float,
    new_rot: float,
    ctx: PlacementContext,
) -> tuple[float, float, float]:
    """Adjust a proposed position to satisfy placement constraints.

    Called by optimizer phases before committing ``ctx.positions[ref]``.
    If the proposed position would violate a constraint, the position is
    adjusted to satisfy it.  If no constraints apply, the proposed
    position is returned unchanged.

    Args:
        ref: Component reference designator.
        new_x: Proposed X position.
        new_y: Proposed Y position.
        new_rot: Proposed rotation.
        ctx: Placement context with constraints.

    Returns:
        Adjusted ``(x, y, rotation)`` satisfying constraints.
    """
    if ctx.constraints is None:
        return new_x, new_y, new_rot

    adjusted_x, adjusted_y = new_x, new_y

    # --- Proximity constraints ---
    for prox in ctx.constraints.proximity:
        if prox.ref != ref:
            continue
        if prox.target_ref not in ctx.positions:
            continue
        tx, ty, _ = ctx.positions[prox.target_ref]
        dist = math.sqrt((adjusted_x - tx) ** 2 + (adjusted_y - ty) ** 2)
        if dist > prox.max_distance_mm:
            # Pull back toward target to satisfy constraint
            ratio = prox.max_distance_mm / max(dist, 0.1)
            adjusted_x = tx + (adjusted_x - tx) * ratio
            adjusted_y = ty + (adjusted_y - ty) * ratio
            _log.debug(
                "  constraint_guard: %s pulled toward %s (%.1f,%.1f) "
                "to satisfy proximity %.1fmm",
                ref, prox.target_ref, adjusted_x, adjusted_y,
                prox.max_distance_mm,
            )

    # --- Ordering constraints ---
    for chain in ctx.constraints.ordering:
        if ref not in chain.refs:
            continue
        chain_refs = [r for r in chain.refs if r in ctx.positions]
        ref_idx = chain_refs.index(ref) if ref in chain_refs else -1
        if ref_idx < 0:
            continue

        # Determine dominant axis from chain spread
        xs = [ctx.positions[r][0] for r in chain_refs]
        ys = [ctx.positions[r][1] for r in chain_refs]
        use_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))

        # Check if proposed position breaks ordering
        if ref_idx > 0:
            prev_ref = chain_refs[ref_idx - 1]
            prev_val = ctx.positions[prev_ref][0 if use_x else 1]
            proposed_val = adjusted_x if use_x else adjusted_y
            if proposed_val < prev_val - 1.0:
                # Would move before previous — clamp to just after
                if use_x:
                    adjusted_x = prev_val + 2.0
                else:
                    adjusted_y = prev_val + 2.0
                _log.debug(
                    "  constraint_guard: %s kept after %s in chain '%s'",
                    ref, prev_ref, chain.group,
                )

        if ref_idx < len(chain_refs) - 1:
            next_ref = chain_refs[ref_idx + 1]
            next_val = ctx.positions[next_ref][0 if use_x else 1]
            proposed_val = adjusted_x if use_x else adjusted_y
            if proposed_val > next_val + 1.0:
                # Would move after next — clamp to just before
                if use_x:
                    adjusted_x = next_val - 2.0
                else:
                    adjusted_y = next_val - 2.0
                _log.debug(
                    "  constraint_guard: %s kept before %s in chain '%s'",
                    ref, next_ref, chain.group,
                )

    return adjusted_x, adjusted_y, new_rot
