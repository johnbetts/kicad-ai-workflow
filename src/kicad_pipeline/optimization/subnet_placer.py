"""Generic subcircuit placement using subnet topology.

Provides ``_phase_subnet_placement`` — a single placement phase that
handles ALL subcircuit types by resolving subnet connections and placing
passives facing their associated IC pins.  Series chains (dividers,
filters) are arranged in connected flow order.

This phase runs BEFORE the type-specific phases (relay drivers, power
chain, etc.) so they can refine the generic placement if needed.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import PlacementContext


@runtime_checkable
class _SubnetConnectionLike(Protocol):
    """Structural protocol matching SubnetConnection from subnet_resolver."""

    @property
    def passive_ref(self) -> str: ...

    @property
    def passive_pin(self) -> str: ...

    @property
    def ic_ref(self) -> str: ...

    @property
    def ic_pin(self) -> str: ...

    @property
    def subnet_name(self) -> str: ...

    @property
    def role(self) -> str: ...


_log = logging.getLogger(__name__)

# Default gap between a passive and its IC pin (mm).
_DEFAULT_CHAIN_GAP_MM: float = 2.0

# Default inter-component gap along a series chain (mm).
_DEFAULT_SERIES_GAP_MM: float = 1.5

# Minimum distance (mm) from IC pin before subnet placer will move a passive.
# Components already within this radius are left alone ("do no harm" guard).
_MOVE_THRESHOLD_MM: float = 8.0


def _phase_subnet_placement(ctx: PlacementContext) -> None:
    """Generic subcircuit placement using subnet topology.

    For each detected subcircuit:

    1. Resolve subnet connections to find IC pin -> passive mappings.
    2. Place passives using pad-facing engine (each passive faces its IC pin).
    3. For series chains (dividers, filters), chain components with pads facing.
    4. Mark placed components as fixed.

    This is a no-op when the subnet resolver finds no connections (e.g.
    when the dependency module is not yet available or the design has no
    subnet-resolvable subcircuits).
    """
    try:
        _mod = importlib.import_module("kicad_pipeline.optimization.subnet_resolver")
    except (ImportError, ModuleNotFoundError):
        _log.debug("subnet_resolver not available — skipping subnet placement")
        return

    _resolve_subnets = getattr(_mod, "resolve_subnets", None)
    _resolve_ic_pin = getattr(_mod, "resolve_ic_pin_position", None)
    _compute_pad_facing = getattr(_mod, "compute_pad_facing_position", None)
    if not all((_resolve_subnets, _resolve_ic_pin, _compute_pad_facing)):
        _log.debug("subnet_resolver missing required functions — skipping")
        return
    # Narrow types for mypy after the None guard above
    assert callable(_resolve_subnets)
    assert callable(_resolve_ic_pin)
    assert callable(_compute_pad_facing)

    import math

    from kicad_pipeline.optimization.signal_flow import order_subcircuit_by_flow

    raw_connections: list[_SubnetConnectionLike] = _resolve_subnets(ctx.requirements)
    if not raw_connections:
        _log.debug("No subnet connections resolved — skipping subnet placement")
        return

    _log.info("  Subnet placement: %d connections resolved", len(raw_connections))

    # Build lookup: passive_ref -> list of SubnetConnection
    passive_connections: dict[str, list[_SubnetConnectionLike]] = {}
    for sn_conn in raw_connections:
        passive_connections.setdefault(sn_conn.passive_ref, []).append(sn_conn)

    # Track which refs we've placed in this phase
    placed_refs: set[str] = set()

    # Place each passive facing its IC pin
    for passive_ref, conns in passive_connections.items():
        if passive_ref in ctx.fixed_refs:
            continue
        if passive_ref not in ctx.positions:
            continue

        # Use the first connection's IC pin for primary placement
        first_conn = conns[0]
        ic_ref: str = first_conn.ic_ref
        ic_pin: str = first_conn.ic_pin

        try:
            result: tuple[float, float, str] = _resolve_ic_pin(
                ic_ref,
                ic_pin,
                ctx.initial_pcb,
            )
            ic_x, ic_y, ic_side = result
        except (ValueError, KeyError):
            _log.debug(
                "Cannot resolve IC pin position for %s.%s — skipping %s",
                ic_ref,
                ic_pin,
                passive_ref,
            )
            continue

        # Bug 3: Skip components already close to their target IC pin.
        cur_x, cur_y, _cur_rot = ctx.positions[passive_ref]
        current_dist = math.sqrt(
            (cur_x - ic_x) ** 2 + (cur_y - ic_y) ** 2,
        )
        if current_dist < _MOVE_THRESHOLD_MM:
            _log.debug(
                "Skipping %s: already %.1fmm from %s.%s (threshold %.1f)",
                passive_ref, current_dist, ic_ref, ic_pin, _MOVE_THRESHOLD_MM,
            )
            continue

        # Get passive size
        pw, ph = ctx.fp_sizes.get(passive_ref, (1.6, 0.8))
        passive_size = (pw, ph)

        try:
            pos_result: tuple[float, float, float] = _compute_pad_facing(
                passive_size,
                ic_x,
                ic_y,
                ic_side,
                _DEFAULT_CHAIN_GAP_MM,
            )
            new_x, new_y, new_rot = pos_result
        except (ValueError, TypeError):
            _log.debug(
                "Cannot compute pad-facing position for %s -> %s.%s",
                passive_ref,
                ic_ref,
                ic_pin,
            )
            continue

        ctx.positions[passive_ref] = (new_x, new_y, new_rot)
        placed_refs.add(passive_ref)

    # For series chains within subcircuits, adjust positions to form
    # a connected flow.
    for sc in ctx.subcircuits:
        flow_order = order_subcircuit_by_flow(sc, ctx.requirements)
        if len(flow_order) >= 2:
            _chain_series_components(ctx, flow_order, placed_refs)

    # Bug 2: Do NOT mark subnet-placed refs as fixed.  Later type-specific
    # phases (relay driver, power chain, etc.) should be free to refine
    # these positions with better domain-specific rules.

    _log.info(
        "  Subnet placement complete: %d components placed",
        len(placed_refs),
    )


def _chain_series_components(
    ctx: PlacementContext,
    flow_order: list[str],
    placed_refs: set[str],
) -> None:
    """Adjust positions of series-chain components to form a connected flow.

    Given an ordered list of refs representing a signal chain, nudge
    each component so it lines up with the previous one along the chain
    direction.  Only adjusts refs that are already in ``placed_refs``
    and not in ``ctx.fixed_refs``.
    """
    # Find the first ref that has a position (anchor)
    anchor_ref: str | None = None
    for ref in flow_order:
        if ref in ctx.positions:
            anchor_ref = ref
            break

    if anchor_ref is None:
        return

    anchor_idx = flow_order.index(anchor_ref)
    ax, ay, _arot = ctx.positions[anchor_ref]

    # Chain forward from anchor
    prev_x, prev_y = ax, ay
    for i in range(anchor_idx + 1, len(flow_order)):
        ref = flow_order[i]
        if ref not in ctx.positions:
            continue
        if ref in ctx.fixed_refs and ref not in placed_refs:
            # Use this ref's position as the new "previous" without moving it
            prev_x, prev_y, _ = ctx.positions[ref]
            continue

        _cx, _cy, rot = ctx.positions[ref]
        pw, ph = ctx.fp_sizes.get(ref, (1.6, 0.8))

        # Place to the right of previous component
        new_x = prev_x + pw / 2.0 + _DEFAULT_SERIES_GAP_MM
        new_y = prev_y

        ctx.positions[ref] = (new_x, new_y, rot)
        placed_refs.add(ref)
        prev_x = new_x
        prev_y = new_y

    # Chain backward from anchor
    prev_x, prev_y = ax, ay
    for i in range(anchor_idx - 1, -1, -1):
        ref = flow_order[i]
        if ref not in ctx.positions:
            continue
        if ref in ctx.fixed_refs and ref not in placed_refs:
            prev_x, prev_y, _ = ctx.positions[ref]
            continue

        _cx, _cy, rot = ctx.positions[ref]
        pw, ph = ctx.fp_sizes.get(ref, (1.6, 0.8))

        new_x = prev_x - pw / 2.0 - _DEFAULT_SERIES_GAP_MM
        new_y = prev_y

        ctx.positions[ref] = (new_x, new_y, rot)
        placed_refs.add(ref)
        prev_x = new_x
        prev_y = new_y
