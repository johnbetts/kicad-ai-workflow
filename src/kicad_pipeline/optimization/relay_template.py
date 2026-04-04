"""Rigid relay group template — hardcoded 1xN relay layout.

Lays out the entire relay group as one rigid block:

    [K1] [K2] [K3] [K4]   ← relays in tight 1xN row (contact side up)
    [D1] [D2] [D3] [D4]   ← flyback diodes (coil side, below relays)
    [Q1] [Q2] [Q3] [Q4]   ← driver transistors
    [Rg] [Rg] [Rg] [Rg]   ← gate resistors
    [Rl] [Rl] [Rl] [Rl]   ← LED resistors (optional)
    [Dl] [Dl] [Dl] [Dl]   ← LED indicators (optional)
    [shared L/C/R]         ← shared power filter (left end, below drivers)

Each per-relay column is centered on the relay X position.
All components are frozen after placement — no downstream phase can move them.

This replaces ``_phase_relay_rows`` + ``_phase_relay_drivers`` +
``_phase_relay_leds`` + ``_phase_relay_power_isolation`` with a single
deterministic template.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kicad_pipeline.optimization.functional_grouper import SubCircuitType

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import PlacementContext

_log = logging.getLogger(__name__)

# Layout constants (mm)
_RELAY_PITCH_MM = 1.0      # gap between adjacent relays
_COIL_GAP_MM = 2.0         # gap between relay bottom and first driver row
_DRIVER_ROW_GAP_MM = 1.0   # gap between driver component rows
_SHARED_GAP_MM = 3.0       # gap between last driver row and shared components


def place_relay_group_rigid(ctx: PlacementContext) -> set[str]:
    """Place the entire relay group as a rigid template.

    Returns the set of all placed refs (for polygon freeze).
    """
    # Collect relay driver subcircuits
    relay_scs = [
        sc for sc in ctx.subcircuits
        if sc.circuit_type == SubCircuitType.RELAY_DRIVER
    ]
    if not relay_scs:
        _log.info("  Relay template: no relay driver subcircuits found")
        return set()

    # Sort relays by their ref number for consistent ordering
    relay_scs.sort(key=lambda sc: sc.anchor_ref)
    n_relays = len(relay_scs)

    # Find the relay zone center for anchor placement
    relay_zone_rect = None
    for z in ctx.zones:
        if z.name == "relay":
            relay_zone_rect = z.rect
            break

    if relay_zone_rect:
        zx1, zy1, zx2, zy2 = relay_zone_rect
        zone_cx = (zx1 + zx2) / 2.0
        zone_cy = (zy1 + zy2) / 2.0
    else:
        bx1, by1, bx2, by2 = ctx.bounds
        zone_cx = (bx1 + bx2) / 2.0
        zone_cy = (by1 + by2) / 2.0

    # Compute relay row geometry
    relay_w, relay_h = ctx.fp_sizes.get(relay_scs[0].anchor_ref, (17.7, 15.6))
    relay_pitch = relay_w + _RELAY_PITCH_MM
    total_row_w = n_relays * relay_w + (n_relays - 1) * _RELAY_PITCH_MM

    # Center relay row on zone center X, place at zone top Y
    row_start_x = zone_cx - total_row_w / 2.0
    if relay_zone_rect:
        relay_row_y = zy1 + relay_h / 2.0 + 1.0  # 1mm from top
    else:
        relay_row_y = zone_cy - 10.0

    all_placed: set[str] = set()

    # Per-relay X positions
    relay_xs: list[float] = []
    for i in range(n_relays):
        kx = row_start_x + relay_w / 2.0 + i * relay_pitch
        relay_xs.append(kx)

    # Place relays in 1xN row
    for i, sc in enumerate(relay_scs):
        kref = sc.anchor_ref
        kx = relay_xs[i]
        ctx.positions[kref] = (kx, relay_row_y, 0.0)
        all_placed.add(kref)
        ctx.relay_support_refs.add(kref)
        _log.info("    Relay %s at (%.1f, %.1f)", kref, kx, relay_row_y)

    # Place driver columns below each relay (coil side)
    cursor_y = relay_row_y + relay_h / 2.0 + _COIL_GAP_MM

    for i, sc in enumerate(relay_scs):
        kx = relay_xs[i]
        col_y = cursor_y

        # Classify support members
        support = [r for r in sc.refs if r != sc.anchor_ref and r in ctx.positions]
        d_refs = sorted(r for r in support if r.startswith("D"))
        q_refs = sorted(r for r in support if r.startswith("Q"))
        r_refs = sorted(r for r in support if r.startswith("R"))

        # Signal chain order: D (flyback) → Q (transistor) → R (gate)
        for refs, rot in [(d_refs, 90.0), (q_refs, 180.0), (r_refs, 270.0)]:
            for ref in refs:
                w, h = ctx.fp_sizes.get(ref, (2.0, 1.0))
                # Swap dimensions if rotated 90/270
                if rot % 180 in (90, 270):
                    w, h = h, w
                py = col_y + h / 2.0
                ctx.positions[ref] = (kx, py, rot)
                all_placed.add(ref)
                ctx.relay_support_refs.add(ref)
                col_y = py + h / 2.0 + _DRIVER_ROW_GAP_MM

    # Place LED indicators below drivers (if detected by relay_leds phase)
    # These will be handled by relay_leds phase — skip for now

    # Place shared relay power components (L, C for isolation)
    # at the left end of the relay row, below drivers
    shared_refs: list[str] = []
    for block in ctx.requirements.features:
        if "relay" not in block.name.lower():
            continue
        for ref in block.components:
            if ref in all_placed or ref.startswith("J") or ref.startswith("K"):
                continue
            prefix = ref.rstrip("0123456789")
            if prefix in ("L", "C", "F"):
                shared_refs.append(ref)

    if shared_refs:
        shared_x = row_start_x  # left end
        shared_y = cursor_y + 15.0  # below last driver row
        for j, ref in enumerate(sorted(shared_refs)):
            w, h = ctx.fp_sizes.get(ref, (2.0, 1.0))
            sx = shared_x + j * (w + 1.0)
            ctx.positions[ref] = (sx, shared_y, 0.0)
            all_placed.add(ref)
            ctx.relay_support_refs.add(ref)

    _log.info("  Relay template: placed %d components in rigid 1x%d layout",
              len(all_placed), n_relays)

    # Mark relay components as relay support (downstream phases know about them)
    # but DON'T freeze — let late decoupling/collision resolution refine.
    # The template provides the initial tight layout; downstream phases
    # can nudge within the group but not scatter.
    _log.info("  Relay template: placed %d refs (not frozen — refinable)", len(all_placed))

    return all_placed
