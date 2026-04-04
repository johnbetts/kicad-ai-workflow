"""Rigid relay group template — hardcoded 1xN relay layout.

Replaces ALL 5 relay phases (relay_rows, relay_connector_alignment,
relay_drivers, relay_leds, relay_power_isolation) with a single
deterministic template:

    [K1] [K2] [K3] [K4]   ← relays in tight 1xN row (contact side up)
    [D6] [D7] [D8] [D9]   ← flyback diodes (coil side)
    [Q1] [Q2] [Q3] [Q4]   ← driver transistors
    [R10][R11][R12][R13]   ← gate resistors
    [R33][R34][R35][R36]   ← LED resistors
    [D18][D19][D20][D21]   ← LED indicators
    [C24 C27 L3 L5]       ← shared power filter (left end)

Each per-relay column is centered on the relay X position.
All 28 components are frozen after placement — no downstream phase
can move them.  This produces a clean, tight relay group every time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kicad_pipeline.optimization.functional_grouper import SubCircuitType

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import PlacementContext

_log = logging.getLogger(__name__)

# Layout constants (mm)
_RELAY_GAP_MM = 1.0        # gap between adjacent relays
_COIL_GAP_MM = 2.0         # gap between relay bottom and first driver row
_ROW_GAP_MM = 1.0          # gap between driver component rows
_SHARED_OFFSET_X_MM = -3.0  # shared components offset left of relay row


def _find_relay_led_pairs(
    ctx: PlacementContext,
    relay_scs: list[object],
) -> dict[str, list[str]]:
    """Find LED indicator components (D_LED + R_LED) per relay.

    Traces coil nets to find R refs on the coil net, then traces
    those R refs to their connected D refs (the LEDs).  Falls back to
    matching by ref number suffix (R33→D18 for K1, etc.).
    """
    # Get all relay group refs
    relay_group_refs: set[str] = set()
    for block in ctx.requirements.features:
        if "relay" in block.name.lower():
            relay_group_refs.update(block.components)

    # Driver refs (from subcircuits)
    driver_refs: set[str] = set()
    for sc in relay_scs:
        driver_refs.update(sc.refs)  # type: ignore[union-attr]

    # LED candidates = relay group refs that are NOT in any driver subcircuit
    # and start with D or R
    led_candidates = {
        r for r in relay_group_refs - driver_refs
        if r.startswith(("D", "R")) and r in ctx.positions
    }

    # Map LED refs to relays by index ordering
    # Sort relays and LEDs by ref number
    k_refs = sorted(sc.anchor_ref for sc in relay_scs)  # type: ignore[union-attr]
    d_leds = sorted(r for r in led_candidates if r.startswith("D"))
    r_leds = sorted(r for r in led_candidates if r.startswith("R"))

    relay_leds: dict[str, list[str]] = {}
    n = len(k_refs)
    for i, kref in enumerate(k_refs):
        leds: list[str] = []
        if i < len(r_leds):
            leds.append(r_leds[i])
        if i < len(d_leds):
            leds.append(d_leds[i])
        if leds:
            relay_leds[kref] = leds

    return relay_leds


def place_relay_group_rigid(
    ctx: PlacementContext,
) -> tuple[dict[str, list[str]], set[str], set[str]]:
    """Place the entire relay group as a rigid template.

    Returns:
        Tuple of (relay_leds, relay_led_refs, all_placed_refs).
        The first two match the interface of ``_phase_relay_leds``.
    """
    # Collect relay driver subcircuits
    relay_scs = [
        sc for sc in ctx.subcircuits
        if sc.circuit_type == SubCircuitType.RELAY_DRIVER
    ]
    if not relay_scs:
        _log.info("  Relay template: no relay driver subcircuits found")
        return {}, set(), set()

    relay_scs.sort(key=lambda sc: sc.anchor_ref)
    n_relays = len(relay_scs)

    # Find relay zone
    relay_zone_rect = None
    for z in ctx.zones:
        if z.name == "relay":
            relay_zone_rect = z.rect
            break

    if relay_zone_rect:
        zx1, zy1, zx2, zy2 = relay_zone_rect
        zone_cx = (zx1 + zx2) / 2.0
    else:
        bx1, _by1, bx2, _by2 = ctx.bounds
        zone_cx = (bx1 + bx2) / 2.0
        zy1 = 15.0  # fallback

    # ── Row 1: Relays ──────────────────────────────────────────────
    relay_w, relay_h = ctx.fp_sizes.get(relay_scs[0].anchor_ref, (17.7, 15.6))
    relay_pitch = relay_w + _RELAY_GAP_MM
    total_row_w = n_relays * relay_w + (n_relays - 1) * _RELAY_GAP_MM

    row_start_x = zone_cx - total_row_w / 2.0
    relay_row_y = zy1 + relay_h / 2.0 + 1.0

    all_placed: set[str] = set()
    relay_xs: list[float] = []

    for i, sc in enumerate(relay_scs):
        kref = sc.anchor_ref
        kx = row_start_x + relay_w / 2.0 + i * relay_pitch
        relay_xs.append(kx)
        ctx.positions[kref] = (kx, relay_row_y, 0.0)
        all_placed.add(kref)
        ctx.relay_support_refs.add(kref)

    _log.info("  Relay template: %d relays in 1x%d row at Y=%.1f",
              n_relays, n_relays, relay_row_y)

    # ── Rows 2-4: Per-relay driver columns (D, Q, R) ──────────────
    driver_top_y = relay_row_y + relay_h / 2.0 + _COIL_GAP_MM

    for i, sc in enumerate(relay_scs):
        kx = relay_xs[i]
        col_y = driver_top_y

        support = [r for r in sc.refs if r != sc.anchor_ref and r in ctx.positions]
        d_refs = sorted(r for r in support if r.startswith("D"))
        q_refs = sorted(r for r in support if r.startswith("Q"))
        r_refs = sorted(r for r in support if r.startswith("R"))

        for refs, rot in [(d_refs, 90.0), (q_refs, 180.0), (r_refs, 270.0)]:
            for ref in refs:
                w, h = ctx.fp_sizes.get(ref, (2.0, 1.0))
                if rot % 180 in (90, 270):
                    w, h = h, w
                py = col_y + h / 2.0
                ctx.positions[ref] = (kx, py, rot)
                all_placed.add(ref)
                ctx.relay_support_refs.add(ref)
                col_y = py + h / 2.0 + _ROW_GAP_MM

    # ── Rows 5-6: LED indicators (R_LED, D_LED) ───────────────────
    relay_leds = _find_relay_led_pairs(ctx, relay_scs)
    relay_led_refs: set[str] = set()

    # LED row starts after the last driver row
    # Use the cursor from the FIRST relay column (all columns same height)
    led_top_y = driver_top_y
    # Advance past 3 driver rows (D + Q + R)
    for _row in range(3):
        led_top_y += 3.0 + _ROW_GAP_MM  # ~3mm per driver component

    for i, sc in enumerate(relay_scs):
        kref = sc.anchor_ref
        kx = relay_xs[i]
        leds = relay_leds.get(kref, [])
        col_y = led_top_y

        for ref in leds:
            if ref not in ctx.positions:
                continue
            w, h = ctx.fp_sizes.get(ref, (2.0, 1.0))
            rot = 90.0  # LED indicators rotated for vertical flow
            if rot % 180 in (90, 270):
                w, h = h, w
            py = col_y + h / 2.0
            ctx.positions[ref] = (kx, py, rot)
            all_placed.add(ref)
            ctx.relay_support_refs.add(ref)
            relay_led_refs.add(ref)
            col_y = py + h / 2.0 + _ROW_GAP_MM

    # ── Shared power filter (L, C at left end) ────────────────────
    shared_refs: list[str] = []
    for block in ctx.requirements.features:
        if "relay" not in block.name.lower():
            continue
        for ref in block.components:
            if ref in all_placed or ref.startswith(("J", "K")):
                continue
            prefix = ref.rstrip("0123456789")
            if prefix in ("L", "C", "F"):
                shared_refs.append(ref)

    if shared_refs:
        shared_x = row_start_x + _SHARED_OFFSET_X_MM
        shared_y = driver_top_y  # same height as first driver row
        for j, ref in enumerate(sorted(shared_refs)):
            w, h = ctx.fp_sizes.get(ref, (2.0, 1.0))
            sy = shared_y + j * (h + _ROW_GAP_MM)
            ctx.positions[ref] = (shared_x, sy, 0.0)
            all_placed.add(ref)
            ctx.relay_support_refs.add(ref)

    _log.info("  Relay template: placed %d/28 components (4K + %d drivers + %d LEDs + %d shared)",
              len(all_placed), len(all_placed) - n_relays - len(relay_led_refs) - len(shared_refs),
              len(relay_led_refs), len(shared_refs))

    # FREEZE all relay components — no downstream phase can scatter them
    ctx.fixed_refs.update(all_placed)
    _log.info("  Relay template: FROZEN %d refs", len(all_placed))

    return relay_leds, relay_led_refs, all_placed
