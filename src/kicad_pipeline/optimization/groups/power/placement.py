"""Power group placement helpers.

Contains power-specific placement phases that organize components
within the power functional group.

Extracted from ``ee_phases_groups.py`` to reduce module size.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kicad_pipeline.optimization.collision_resolver import (
    _PlacementGrid,
    _rotation_aware_size,
)
from kicad_pipeline.optimization.groups import (
    _build_exclusion_grid,
    _build_net_to_group_refs,
    _clamp,
    _classify_refs_by_prefix,
    _collect_feature_refs,
    _find_zone_rect,
)

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import PlacementContext

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Power chain signal-flow phase (learned from human reference boards)
# ---------------------------------------------------------------------------

# Relative offsets (dx, dy, rotation) for passives around a BUCK IC anchor.
# Learned from human-routed TPS54331 layout on a 50x40mm board.
# Coordinates are relative to the buck IC centroid.
#
# Clearance check (0805 courtyard 2.7x2.0mm, SOD-323 ~2.7x2.0mm, L1210 4.1x3.4mm):
#   input_cap vs bootstrap_cap: same x, dy=10.1 — OK (>>2.0mm)
#   inductor vs catch_diode: dy=3.2, min=(3.4+2.0)/2=2.7 -> OK
#   catch_diode vs fb_bot_r: dy=3.3 > 2.0mm minimum — OK
#   fb_bot_r vs fb_top_r: dy=2.5 > 2.0mm minimum — OK
_BUCK_PASSIVE_OFFSETS: dict[str, tuple[float, float, float]] = {
    # (role, (dx, dy, rotation))  — role matched by net name keywords
    # Offsets are ABSOLUTE distances from the buck IC centroid.
    # IC is placed at +90deg rotation so:
    #   PH/SW pin (pin 7) → RIGHT side (dx=+2.7, dy=-0.6)
    #   VSNS/FB pin (pin 5) → RIGHT side (dx=+2.7, dy=+1.9)
    #   VIN pin (pin 2) → LEFT side (dx=-2.7, dy=-0.6)
    #   BOOT pin (pin 1) → LEFT side (dx=-2.7, dy=-1.9)
    #
    # Footprint sizes (rotation-aware, from estimate_footprint_size):
    #   U1 SOIC-8 at ±90°: 8.9w x 5.9h, half=(4.45, 2.95)
    #   L1210:              3.7w x 3.0h, half=(1.85, 1.50)
    #   SOD-323 (D1):       4.8w x 2.2h, half=(2.40, 1.10)
    #   0805 cap (C1):      4.1w x 2.2h, half=(2.05, 1.10)
    #   0805 cap (C2):      2.5w x 1.8h, half=(1.25, 0.90)
    #   0402 R/C:           1.5w x 1.0h, half=(0.75, 0.50)
    #
    # Min center-to-center (with 0.5mm gap):
    #   U1-L1 dx: 4.45+1.85+0.50=6.80   U1-D1 dx: 4.45+2.40+0.50=7.35
    #   U1-0805 dx: 4.45+2.05+0.50=7.00 U1-0402 dx: 4.45+0.75+0.50=5.70
    #   L1-D1 dy: 1.50+1.10+0.50=3.10   R-R dy: 0.50+0.50+0.50=1.50
    #
    # Signal flow: input(left) -> IC -> PH/inductor/diode(right) -> output(far right)
    # Layout:
    #   C3(bootstrap)  C1(input)   U1(IC)   L1(inductor)  C2(output)
    #                                        D1(diode)
    #                                        R2(fb_bot) R1(fb_top)
    # Collision-safe offsets for SOIC-8 at 90° rotation:
    #   SOIC-8 90°: ~8.9w x 5.9h, half=(4.45, 2.95)
    #   L1210:      3.7w x 3.0h, half=(1.85, 1.50)
    #   0805 cap:   2.5w x 1.8h, half=(1.25, 0.90)
    #   SOD-323:    3.0w x 3.0h, half=(1.50, 1.50)
    #   0402:       1.5w x 1.0h, half=(0.75, 0.50)
    # Min center-to-center with 0.5mm gap:
    #   U1-L1: 4.45+1.85+0.5 = 6.8mm
    #   U1-0805: 4.45+1.25+0.5 = 6.2mm
    #   L1-0805: 1.85+1.25+0.5 = 3.6mm
    "input_cap": (-6.5, 0.5, 0.0),        # C on VIN rail, LEFT of IC
    "bootstrap_cap": (-6.5, -2.5, 0.0),   # C on BST net, LEFT of IC, above C1
    "inductor": (7.0, -0.6, 0.0),         # L on SW net, RIGHT of IC near PH pin
    "catch_diode": (7.0, 3.0, 180.0),     # D on SW net, RIGHT below inductor
    "fb_bot_r": (7.0, 5.5, 0.0),          # R on FB+GND, below catch diode
    "fb_top_r": (7.0, 7.5, 180.0),        # R on FB only, below fb_bot_r
    # C at output (L1@7.0 + L1_half 1.85 + C_half 1.25 + gap 0.9 = 11.0)
    "output_cap": (11.0, 0.0, -90.0),
}

# Relative offsets for passives around an LDO IC anchor.
# Learned from human-routed AMS1117-3.3 layout on a 50x40mm board.
#
# Clearance calculation (actual measured values from easyeda2kicad SOT-223 footprint):
#   fp_size_dict returns U2=9.36x6.70mm (includes tab pad), half-width=4.68mm
#   C_0805 fp_size at -90°: effective width=2.35mm, half=1.175mm
#   No-collision condition: cap_cx ± 1.175 must not overlap [u2_cx ± 4.68]
#   Input cap (left): cap_cx + 1.175 ≤ u2_cx - 4.68 → offset ≤ -5.855 → use -6.5mm
#   Output cap (right): cap_cx - 1.175 ≥ u2_cx + 4.68 → offset ≥ +5.855 → use +6.5mm
#
# The LDO input cap sits just outside the IC tab pad (NOT at the midpoint
# between the buck and LDO — the old -17.4mm offset placed it on top of the
# buck output cap, and -5.5mm still left 0.355mm overlap due to the wide tab).
_LDO_PASSIVE_OFFSETS: dict[str, tuple[float, float, float]] = {
    # LDO IC at 180deg: VIN (pin 3) faces LEFT, VOUT (pin 2) faces LEFT,
    # VOUT_TAB (pin 4) faces RIGHT.
    # SOT-223 (U2): 10.4w x 7.7h at 180deg (no swap), half=(5.2, 3.85)
    #   VIN pin at dx=-3.0, dy=+2.3 from center
    #   VOUT pin at dx=-3.0, dy=0.0 from center
    #   VOUT_TAB at dx=+3.0, dy=0.0 from center
    # 0805 cap at -90deg: 2.2w x 4.1h, half-w=1.1
    # Min dx from IC center = 5.2 + 1.1 + 0.50 = 6.80
    #
    # Output cap near VOUT pin on LEFT side (shortest decoupling path).
    # Input cap on LEFT side below output cap, near VIN pin.
    # 0805 at -90deg: 2.2w x 4.1h. Two caps stacked vertically need
    # dy >= (4.1+4.1)/2 + 0.5 = 4.6mm
    # Input and output caps stacked vertically on LEFT side of LDO.
    # Output cap at dy=0 (near VOUT pin), HF bypass stacks at dy+5.
    # Input cap at dy=-5 (above output, clear of duplicate stacking below).
    "input_cap": (-6.0, -4.5, -90.0),     # C on VIN side (LEFT), above VOUT cap
    "output_cap": (-6.0, 0.0, -90.0),     # C on VOUT side (LEFT), near VOUT pin
}

# Net-name keywords used to classify passive roles in power subcircuits.
_VIN_KEYWORDS = ("VIN", "+24V", "+12V", "VBUS", "V_IN")
_BST_KEYWORDS = ("BST", "BOOT", "BOOTSTRAP")
_SW_KEYWORDS = ("SW", "PH", "PHASE")
_FB_KEYWORDS = ("FB", "VSNS", "FEEDBACK", "SENSE")
_OUTPUT_KEYWORDS = ("+5V", "+3V3", "+3.3V", "+1V8", "VOUT", "V_OUT", "BUCK_5V")


def _collect_passive_nets(
    ref: str,
    ic_ref: str,
    ctx: PlacementContext,
) -> tuple[list[str], list[str]]:
    """Return (all_nets_for_ref, shared_nets_with_ic) for a passive component.

    Both lists contain upper-cased net names.
    """
    all_nets: list[str] = []
    shared: list[str] = []
    for net in ctx.requirements.nets:
        conn_refs = {c.ref for c in net.connections}
        if ref not in conn_refs:
            continue
        name_upper = net.name.upper()
        all_nets.append(name_upper)
        if ic_ref in conn_refs:
            shared.append(name_upper)
    return all_nets, shared


def _classify_cap_role(
    all_nets: list[str],
    shared_nets: list[str],
    ic_input_voltage: float,
    ic_output_voltage: float,
) -> str:
    """Classify a capacitor's role in a power subcircuit.

    Returns one of: "bootstrap_cap", "input_cap", "output_cap".
    """
    # Bootstrap cap: on BST net
    if any(any(kw in n for kw in _BST_KEYWORDS) for n in all_nets):
        return "bootstrap_cap"
    # Input cap: on VIN net shared with IC
    if any(any(kw in n for kw in _VIN_KEYWORDS) for n in shared_nets):
        return "input_cap"
    # Use voltage magnitude when regulator voltages are known
    if ic_input_voltage > 0 and ic_output_voltage > 0:
        cap_voltage = _estimate_net_voltage(all_nets)
        if cap_voltage is not None:
            in_diff = abs(cap_voltage - ic_input_voltage)
            out_diff = abs(cap_voltage - ic_output_voltage)
            return "input_cap" if in_diff < out_diff else "output_cap"
    # Output cap: on output net keyword
    if any(any(kw in n for kw in _OUTPUT_KEYWORDS) for n in all_nets):
        return "output_cap"
    # Fallback: shared net with IC → input, otherwise output
    return "input_cap" if shared_nets else "output_cap"


_GND_NET_NAMES: frozenset[str] = frozenset({"GND", "AGND", "DGND", "PGND"})
"""Canonical GND net names used to classify feedback-divider resistors."""


def _classify_resistor_role_power(all_nets: list[str]) -> str | None:
    """Classify a resistor's role in a power subcircuit (FB divider top/bottom)."""
    for n in all_nets:
        if any(kw in n for kw in _FB_KEYWORDS):
            if any(n2 in _GND_NET_NAMES for n2 in all_nets):
                return "fb_bot_r"
            return "fb_top_r"
    return None


def _classify_passive_role_power(
    ref: str,
    ctx: PlacementContext,
    ic_ref: str,
    subcircuit_refs: set[str],
    ic_input_voltage: float = 0.0,
    ic_output_voltage: float = 0.0,
) -> str | None:
    """Classify a passive's role relative to its power IC via net connectivity.

    Returns a role key matching ``_BUCK_PASSIVE_OFFSETS`` /
    ``_LDO_PASSIVE_OFFSETS``, or ``None`` if unclassified.

    Args:
        ic_input_voltage: Estimated input voltage of the regulator (for
            distinguishing input vs output caps when net names are ambiguous).
        ic_output_voltage: Estimated output voltage of the regulator.
    """
    if ref == ic_ref or ref not in ctx.positions:
        return None

    all_nets_for_ref, shared_nets = _collect_passive_nets(ref, ic_ref, ctx)
    prefix = ref[0]

    if prefix == "L":
        # Inductors in a power subcircuit are always the main switching inductor
        return "inductor"

    if prefix == "D":
        # Diodes in a power subcircuit are always the catch/freewheeling diode
        return "catch_diode"

    if prefix == "C":
        return _classify_cap_role(
            all_nets_for_ref, shared_nets, ic_input_voltage, ic_output_voltage,
        )

    if prefix == "R":
        return _classify_resistor_role_power(all_nets_for_ref)

    return None


def _estimate_net_voltage(net_names: list[str]) -> float | None:
    """Estimate voltage from net name keywords.

    Returns the voltage in volts, or None if no voltage keyword found.
    """
    for n in net_names:
        # Try common voltage patterns
        for prefix, voltage in (
            ("+24V", 24.0), ("+12V", 12.0), ("+9V", 9.0),
            ("+5V", 5.0), ("+3V3", 3.3), ("+3.3V", 3.3),
            ("+1V8", 1.8), ("+1.8V", 1.8), ("+2V5", 2.5),
        ):
            if prefix in n:
                return voltage
    return None


def _estimate_regulator_voltages(
    ic_ref: str,
    ctx: PlacementContext,
) -> tuple[float, float]:
    """Estimate input and output voltages for a regulator IC from net names.

    Returns (input_voltage, output_voltage).  Both 0.0 if unknown.
    """
    voltages: list[float] = []
    for net in ctx.requirements.nets:
        if not any(c.ref == ic_ref for c in net.connections):
            continue
        if net.name.upper() in ("GND", "AGND", "DGND", "PGND"):
            continue
        v = _estimate_net_voltage([net.name.upper()])
        if v is not None:
            voltages.append(v)
    if len(voltages) >= 2:
        return (max(voltages), min(voltages))
    if len(voltages) == 1:
        return (voltages[0], 0.0)
    return (0.0, 0.0)


def _resolve_power_zone_bounds(
    ctx: PlacementContext,
    n_regs: int,
) -> tuple[float, float, float, float]:
    """Return (zx1, zy1, zx2, zy2) for the power zone, expanding if too narrow."""
    min_power_zone_w = 25.0
    zone_rect = _find_zone_rect(ctx, "power")
    if zone_rect is None:
        return ctx.bounds
    zx1, zy1, zx2, zy2 = zone_rect
    if zx2 - zx1 < min_power_zone_w and n_regs >= 2:
        _log.info(
            "    3c1b: power zone too narrow (%.1fmm) — expanding to board bounds",
            zx2 - zx1,
        )
        return ctx.bounds
    return zone_rect


def _compute_regulator_x_fractions(n_regs: int) -> list[float]:
    """Compute horizontal zone-fraction positions for N regulators.

    - 1 regulator: 30% (room for output passives on right)
    - 2 regulators: 17%, 82% (learned from human reference board)
    - N regulators: evenly spread 15%-85%
    """
    if n_regs == 1:
        return [0.30]
    if n_regs == 2:
        return [0.17, 0.82]
    return [0.15 + 0.70 * i / (n_regs - 1) for i in range(n_regs)]


def _sort_regulators_by_flow(
    all_reg_scs: list[object],
    ctx: PlacementContext,
    topology: object,
) -> None:
    """Sort regulator subcircuits in-place by power-chain flow order.

    Earlier in the chain (higher input voltage) sorts to a lower index
    so it is placed further left in the signal-flow layout.
    """
    def _flow_order(sc: object) -> float:
        if sc.input_domain and sc.input_domain in topology.domain_order:  # type: ignore[union-attr]
            return float(list(topology.domain_order).index(sc.input_domain))  # type: ignore[union-attr]
        max_v = 0.0
        for net in ctx.requirements.nets:
            if not any(c.ref == sc.anchor_ref for c in net.connections):  # type: ignore[union-attr]
                continue
            upper = net.name.upper()
            for prefix in ("+24V", "+12V", "+5V", "+3V3", "+3.3V", "+1V8"):
                if prefix in upper:
                    try:
                        v = float(prefix.replace("+", "").replace("V", ".").rstrip("."))
                    except ValueError:
                        continue
                    max_v = max(max_v, v)
            if "VIN" in upper or "V_IN" in upper:
                max_v = max(max_v, 100.0)
        return -max_v if max_v > 0 else 999.0

    all_reg_scs.sort(key=_flow_order)


def _gather_ic_net_refs(
    ic_ref: str,
    sc_refs: set[str],
    ctx: PlacementContext,
    power_group_refs: set[str],
) -> set[str]:
    """Return all passive refs connected (directly or 1-hop) to *ic_ref*.

    Pass 1: direct connections on non-GND nets shared with *ic_ref*.
    Pass 2: one hop — non-GND nets that share any ref from pass-1 result.
    """
    _gnd_nets = frozenset({"GND", "AGND", "DGND", "PGND"})
    ic_net_refs: set[str] = set(sc_refs)

    for net in ctx.requirements.nets:
        if net.name.upper() in _gnd_nets:
            continue
        conn_refs = {c.ref for c in net.connections}
        if ic_ref not in conn_refs:
            continue
        for c in net.connections:
            if (c.ref != ic_ref
                    and c.ref in ctx.positions
                    and c.ref[0] in "RCLDF"
                    and (c.ref in power_group_refs or c.ref in sc_refs)):
                ic_net_refs.add(c.ref)

    for net in ctx.requirements.nets:
        if net.name.upper() in _gnd_nets:
            continue
        conn_refs = {c.ref for c in net.connections}
        if conn_refs & ic_net_refs:
            for c in net.connections:
                if (c.ref not in ic_net_refs
                        and c.ref != ic_ref
                        and c.ref in ctx.positions
                        and c.ref[0] in "RCLDF"
                        and c.ref in power_group_refs):
                    ic_net_refs.add(c.ref)

    return ic_net_refs


def _place_regulator_passives(
    ic_ref: str,
    ic_x: float,
    ic_y: float,
    ic_net_refs: set[str],
    offsets: dict[str, tuple[float, float, float]],
    zone_bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
    in_v: float,
    out_v: float,
) -> set[str]:
    """Place passives for one regulator IC at learned offsets.

    Returns the set of role names that were placed.
    """
    zx1, zy1, zx2, zy2 = zone_bounds
    # Offsets are absolute physical distances (learned from reference boards).
    # They must NOT scale up with zone size — larger zones should not push
    # power components further apart.  Only scale DOWN if the zone is smaller
    # than the reference to avoid off-board placement.
    zone_w = zx2 - zx1
    zone_h = zy2 - zy1
    ref_zone_w = 35.0  # reference zone width offsets were tuned for
    ref_zone_h = 30.0  # reference zone height offsets were tuned for
    sx = min(1.0, max(0.5, zone_w / ref_zone_w))
    sy = min(1.0, max(0.5, zone_h / ref_zone_h))
    placed_roles: set[str] = set()
    # Track position of first component placed in each role so duplicates
    # (e.g. two output caps: bulk + HF bypass) can be stacked nearby.
    role_positions: dict[str, tuple[float, float]] = {}
    dup_offset_mm = 5.0  # stack duplicates far enough for rotated 0805 caps (4.1mm tall)

    for ref in sorted(ic_net_refs):
        if ref == ic_ref:
            continue
        role = _classify_passive_role_power(ref, ctx, ic_ref, ic_net_refs, in_v, out_v)
        if role is None or role not in offsets:
            continue
        dx, dy, rot = offsets[role]
        if role not in placed_roles:
            # First component with this role — place at learned offset
            placed_roles.add(role)
            px = _clamp(ic_x + dx * sx, zx1 + 1.0, zx2 - 1.0)
            py = _clamp(ic_y + dy * sy, zy1 + 1.0, zy2 - 1.0)
            role_positions[role] = (px, py)
        else:
            # Duplicate role (e.g. C6 is a second output_cap alongside C5).
            # Stack it adjacent to the primary component.
            base_x, base_y = role_positions[role]
            px = _clamp(base_x, zx1 + 1.0, zx2 - 1.0)
            py = _clamp(base_y + dup_offset_mm, zy1 + 1.0, zy2 - 1.0)
        ctx.positions[ref] = (px, py, rot)
        ctx.power_group_fixed.add(ref)
    return placed_roles


def _compute_global_pin_positions(
    ic_ref: str,
    ctx: PlacementContext,
) -> dict[str, tuple[float, float]]:
    """Compute global (x, y) positions of all pads on *ic_ref*.

    Returns a dict mapping pad net_name (upper) to global (x, y).
    For pads with the same net, the first one wins.
    """
    import math as _m

    fp = None
    for f in ctx.initial_pcb.footprints:
        if f.ref == ic_ref:
            fp = f
            break
    if fp is None:
        return {}

    cx, cy, rot_deg = ctx.positions.get(ic_ref, (0.0, 0.0, 0.0))
    rad = _m.radians(rot_deg)
    cos_r = _m.cos(rad)
    sin_r = _m.sin(rad)

    result: dict[str, tuple[float, float]] = {}
    for pad in fp.pads:
        if not pad.net_name:
            continue
        net_upper = pad.net_name.upper()
        if net_upper in result or net_upper in ("GND", "AGND", "DGND", "PGND"):
            continue
        gx = cx + pad.position.x * cos_r - pad.position.y * sin_r
        gy = cy + pad.position.x * sin_r + pad.position.y * cos_r
        result[net_upper] = (gx, gy)
    return result


def _pull_passives_toward_pins(
    ic_ref: str,
    ic_net_refs: set[str],
    zone_bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Pull placed passives toward the IC pin they share a non-GND net with.

    After offset-based placement, passives may be too far from their connected
    pin. This pass computes the global pin position and pulls each passive
    closer — placing it just outside the IC edge nearest that pin, at the
    pin's Y coordinate (for left/right pins) or X coordinate (for top/bottom
    pins), while preserving the offset's side choice (sign of dx/dy).
    """
    import math as _m

    gnd_nets = frozenset({"GND", "AGND", "DGND", "PGND"})
    pin_positions = _compute_global_pin_positions(ic_ref, ctx)
    if not pin_positions:
        return

    zx1, zy1, zx2, zy2 = zone_bounds
    ic_x, ic_y, _ic_rot = ctx.positions[ic_ref]
    ic_w, ic_h = ctx.fp_sizes.get(ic_ref, (6.0, 6.0))
    if abs(_ic_rot) % 180 in (90.0, 270.0):
        ic_w, ic_h = ic_h, ic_w
    ic_half_w = ic_w / 2.0
    ic_half_h = ic_h / 2.0
    gap = 2.0  # clearance for power loop components (was 0.2 — caused collisions)

    for ref in sorted(ic_net_refs):
        if ref == ic_ref or ref not in ctx.positions:
            continue
        # Find the shared non-GND net
        shared_pin_pos: tuple[float, float] | None = None
        for net in ctx.requirements.nets:
            name_u = net.name.upper()
            if name_u in gnd_nets:
                continue
            conn_refs = {c.ref for c in net.connections}
            if ref in conn_refs and ic_ref in conn_refs and name_u in pin_positions:
                shared_pin_pos = pin_positions[name_u]
                break

        if shared_pin_pos is None:
            continue

        px, py, p_rot = ctx.positions[ref]
        pin_x, pin_y = shared_pin_pos
        pw, ph = ctx.fp_sizes.get(ref, (2.0, 2.0))
        if p_rot % 180 in (90.0, 270.0):
            pw, ph = ph, pw

        # Determine which IC edge the pin is closest to
        pin_dx = pin_x - ic_x
        pin_dy = pin_y - ic_y

        # The passive should be placed just outside the IC edge nearest
        # the pin. Place on the side of the IC where the pin is, at min clearance
        if abs(pin_dx) >= abs(pin_dy):
            # Pin is on left or right edge
            side_sign = 1.0 if pin_dx >= 0 else -1.0
            target_x = ic_x + side_sign * (ic_half_w + pw / 2.0 + gap)
            target_y = pin_y  # Align Y with pin
        else:
            # Pin is on top or bottom edge
            side_sign = 1.0 if pin_dy >= 0 else -1.0
            target_x = pin_x  # Align X with pin
            target_y = ic_y + side_sign * (ic_half_h + ph / 2.0 + gap)

        target_x = _clamp(target_x, zx1 + 1.0, zx2 - 1.0)
        target_y = _clamp(target_y, zy1 + 1.0, zy2 - 1.0)

        # Only move if it brings the passive closer to the pin
        old_pin_dist = _m.sqrt((px - pin_x) ** 2 + (py - pin_y) ** 2)
        new_pin_dist = _m.sqrt((target_x - pin_x) ** 2 + (target_y - pin_y) ** 2)

        if new_pin_dist < old_pin_dist:
            ctx.positions[ref] = (target_x, target_y, p_rot)
            _log.info(
                "      pin-pull %s toward %s pin: (%.1f,%.1f) -> (%.1f,%.1f) "
                "[pin_d=%.1f->%.1f]",
                ref, ic_ref, px, py, target_x, target_y,
                old_pin_dist, new_pin_dist,
            )


def _place_power_chain_ic(
    sc: object,
    reg_idx: int,
    x_fracs: list[float],
    zone_bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Place one regulator IC and its passives at the signal-flow position."""
    from kicad_pipeline.optimization.functional_grouper import SubCircuitType

    ic_ref: str = sc.anchor_ref  # type: ignore[union-attr]
    if ic_ref not in ctx.positions:
        return

    zx1, zy1, zx2, zy2 = zone_bounds
    zone_w = zx2 - zx1
    zone_h = zy2 - zy1

    is_buck = sc.circuit_type == SubCircuitType.BUCK_CONVERTER  # type: ignore[union-attr]
    offsets = _BUCK_PASSIVE_OFFSETS if is_buck else _LDO_PASSIVE_OFFSETS
    y_frac = 0.40 if is_buck else 0.47
    # Buck IC at +90deg so PH/SW pin (pin 7) faces RIGHT toward inductor/output.
    # At -90deg the PH pin faces LEFT, forcing inductor placement against signal flow.
    # LDO at 180deg so VIN faces LEFT (input side) and VOUT_TAB faces RIGHT (output).
    ic_rot = 90.0 if is_buck else 180.0

    ic_x = zx1 + zone_w * x_fracs[reg_idx]
    ic_y = _clamp(zy1 + zone_h * y_frac, zy1 + 3.0, zy2 - 3.0)

    ctx.positions[ic_ref] = (ic_x, ic_y, ic_rot)
    ctx.power_group_fixed.add(ic_ref)

    power_group_refs = _collect_feature_refs(ctx, "power", "supply")
    ic_net_refs = _gather_ic_net_refs(
        ic_ref, set(sc.refs), ctx, power_group_refs,  # type: ignore[union-attr]
    )
    in_v, out_v = _estimate_regulator_voltages(ic_ref, ctx)
    placed_roles = _place_regulator_passives(
        ic_ref, ic_x, ic_y, ic_net_refs, offsets, zone_bounds, ctx, in_v, out_v,
    )

    _log.info(
        "    3c1b: placed %s (%s) at (%.1f, %.1f, %.0f) with %d passives (roles: %s)",
        ic_ref,
        "buck" if is_buck else "ldo",
        ic_x, ic_y, ic_rot,
        len(placed_roles),
        ", ".join(sorted(placed_roles)),
    )


def _nudge_connector_clear(
    j_ref: str,
    jx: float,
    jy: float,
    j_rot: float,
    ctx: PlacementContext,
    zone_bounds: tuple[float, float, float, float],
    max_attempts: int = 8,
) -> tuple[float, float]:
    """Nudge a connector position until it no longer collides with any placed component.

    Tries shifting in alternating Y then X directions. Returns the final (x, y).
    """
    zx1, zy1, zx2, zy2 = zone_bounds
    jw, jh = ctx.fp_sizes.get(j_ref, (2.5, 5.0))
    if j_rot % 180 in (90.0, 270.0):
        jw, jh = jh, jw

    clearance = 0.25  # mm

    for _attempt in range(max_attempts):
        collision_found = False
        for ref, (rx, ry, r_rot) in ctx.positions.items():
            if ref == j_ref:
                continue
            rw, rh = ctx.fp_sizes.get(ref, (2.0, 2.0))
            if r_rot % 180 in (90.0, 270.0):
                rw, rh = rh, rw

            overlap_x = (jw + rw) / 2.0 + clearance - abs(jx - rx)
            overlap_y = (jh + rh) / 2.0 + clearance - abs(jy - ry)

            if overlap_x > 0 and overlap_y > 0:
                # Collision detected — nudge in the direction of least overlap
                collision_found = True
                if overlap_y <= overlap_x:
                    # Nudge in Y
                    nudge = overlap_y + 0.5
                    jy = jy + nudge if jy >= ry else jy - nudge
                else:
                    # Nudge in X
                    nudge = overlap_x + 0.5
                    jx = jx + nudge if jx >= rx else jx - nudge
                jx = _clamp(jx, zx1 + 1.0, zx2 - 1.0)
                jy = _clamp(jy, zy1 + 1.0, zy2 - 1.0)
                break  # re-check all after nudge

        if not collision_found:
            break

    return jx, jy


def _place_power_connectors(
    ctx: PlacementContext,
    all_reg_scs: list[object],
    zone_bounds: tuple[float, float, float, float],
) -> None:
    """Place power-group connectors at learned reference-board positions.

    Only applied on dedicated power boards (<=2 features, <=4 connectors).
    On larger boards, _phase_top_edge_connectors handles connector placement.
    """
    zx1, zy1, zx2, zy2 = zone_bounds
    zone_w = zx2 - zx1
    zone_h = zy2 - zy1

    power_group_refs = _collect_feature_refs(ctx, "power", "supply")
    power_connectors = sorted(
        r for r in power_group_refs
        if r.startswith("J") and r in ctx.positions
    )
    is_power_focused_board = (
        len(ctx.requirements.features) <= 2 and len(power_connectors) <= 4
    )
    if not (power_connectors and all_reg_scs and is_power_focused_board):
        return

    # On a dedicated power board the signal-flow phase has authority over
    # connector positions.  Clear any fixed status set by the earlier
    # constraint-placement phase (which only enforced ordering, not absolute
    # positions) so we can place connectors at their correct signal-flow spots.
    for j_ref in power_connectors:
        ctx.fixed_refs.discard(j_ref)

    # index -> (x_frac, y_frac, rotation)
    # All positions are zone-relative fractions for left→right signal flow.
    # Connectors must be near board edges (within 8mm) for wire access.
    #   J1 (input 24V):  left edge, top area → wire entry faces top
    #   J2 (mid test):   between buck and LDO, below center
    #   J3 (output):     right edge, lower area → wire exit
    conn_rules: dict[int, tuple[float, float, float]] = {
        0: (0.10, 0.10, 0.0),     # J1: left edge, top area, wire entry faces top
        1: (0.50, 0.88, -90.0),   # J2: center, near bottom edge
        2: (0.97, 0.75, -90.0),   # J3: right edge, clear of zone right
    }
    for idx, j_ref in enumerate(power_connectors):
        if j_ref in ctx.fixed_refs:
            continue
        rule = conn_rules.get(idx)
        if rule is None:
            continue
        x_frac, y_frac, rot = rule
        jx = zx1 + zone_w * x_frac
        jy = zy1 + zone_h * y_frac
        clamped_x = _clamp(jx, zx1 + 1.0, zx2 - 1.0)
        clamped_y = _clamp(jy, zy1 + 1.0, zy2 - 1.0)
        # Respect placement constraints (proximity, ordering)
        from kicad_pipeline.optimization.constraint_guard import respect_constraints
        clamped_x, clamped_y, rot = respect_constraints(
            j_ref, clamped_x, clamped_y, rot, ctx,
        )
        # Nudge connector away from any component it would collide with
        clamped_x, clamped_y = _nudge_connector_clear(
            j_ref, clamped_x, clamped_y, rot, ctx, zone_bounds,
        )
        ctx.positions[j_ref] = (clamped_x, clamped_y, rot)
        ctx.power_group_fixed.add(j_ref)
        ctx.fixed_refs.add(j_ref)


def _phase_power_chain_flow(ctx: PlacementContext) -> None:
    """3c1b: Power chain signal-flow ordering.

    Enforces left-to-right signal flow for power conversion chains:
    input connector -> buck IC -> inductor -> output cap -> LDO -> output.

    Learned from human reference boards:
    - Buck IC rotated -90deg, placed at ~17% board width
    - LDO at ~82% board width, rotated 0deg
    - Passives placed at fixed offsets from their parent IC
    - Caps on output side rotated -90deg (vertical, matching horizontal flow)

    This phase runs after ``_phase_power_group`` and applies signal-flow
    corrections to power subcircuit components that were column-placed.
    """
    from kicad_pipeline.optimization.functional_grouper import (
        SubCircuitType,
        compute_power_flow_topology,
    )

    _log.info("  3c1b: Power chain signal-flow ordering")

    buck_scs = [sc for sc in ctx.subcircuits if sc.circuit_type == SubCircuitType.BUCK_CONVERTER]
    ldo_scs = [sc for sc in ctx.subcircuits if sc.circuit_type == SubCircuitType.LDO_REGULATOR]

    if not (buck_scs or ldo_scs):
        _log.info("    No power regulators found — skipping signal-flow phase")
        return

    all_reg_scs = buck_scs + ldo_scs
    zone_bounds = _resolve_power_zone_bounds(ctx, len(all_reg_scs))
    topology = compute_power_flow_topology(tuple(ctx.subcircuits))
    _sort_regulators_by_flow(all_reg_scs, ctx, topology)

    if not all_reg_scs:
        return

    x_fracs = _compute_regulator_x_fractions(len(all_reg_scs))

    for reg_idx, sc in enumerate(all_reg_scs):
        _place_power_chain_ic(sc, reg_idx, x_fracs, zone_bounds, ctx)

    _place_power_connectors(ctx, all_reg_scs, zone_bounds)

    _log.info(
        "    3c1b: signal-flow ordered %d regulators across power zone",
        len(all_reg_scs),
    )


def _pwr_place_column_down(
    ctx: PlacementContext,
    refs: list[str],
    col_x: float,
    start_y: float,
    placed_in_col: set[str],
    grid: object,
    pz_bounds: tuple[float, float, float, float],
    strip_gap: float,
) -> float:
    """Place *refs* in a vertical column going DOWN. Returns bottom Y."""
    pz_x1, pz_y1, pz_x2, pz_y2 = pz_bounds
    cy = start_y
    for ref in refs:
        if ref not in ctx.positions or ref in ctx.fixed_refs or ref in placed_in_col or ref == "":
            continue
        w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        px = max(pz_x1, min(pz_x2, col_x))
        py = max(pz_y1, min(pz_y2, cy + h / 2.0))
        ctx.positions[ref] = (px, py, 0.0)
        grid.place(px, py, w, h)
        ctx.power_group_fixed.add(ref)
        placed_in_col.add(ref)
        cy = py + h / 2.0 + strip_gap
    return cy


def _pwr_place_column_up(
    ctx: PlacementContext,
    refs: list[str],
    col_x: float,
    start_y: float,
    placed_in_col: set[str],
    grid: object,
    pz_bounds: tuple[float, float, float, float],
    strip_gap: float,
) -> float:
    """Place *refs* in a vertical column going UP. Returns top Y."""
    pz_x1, pz_y1, pz_x2, pz_y2 = pz_bounds
    cy = start_y
    for ref in refs:
        if ref not in ctx.positions or ref in ctx.fixed_refs or ref in placed_in_col or ref == "":
            continue
        w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        px = max(pz_x1, min(pz_x2, col_x))
        py = max(pz_y1, min(pz_y2, cy - h / 2.0))
        ctx.positions[ref] = (px, py, 0.0)
        grid.place(px, py, w, h)
        ctx.power_group_fixed.add(ref)
        placed_in_col.add(ref)
        cy = py - h / 2.0 - strip_gap
    return cy


def _pwr_place_fork_columns(
    ctx: PlacementContext,
    columns: dict[str, list[str]],
    anchor_x: float,
    sub_col_offset: float,
    col_spacing: float,
    output_top: float,
    placed_in_col: set[str],
    grid: object,
    pz_bounds: tuple[float, float, float, float],
    strip_gap: float,
) -> None:
    """Place output, bridge, buck2, and tail columns around the anchor IC."""
    left_x = anchor_x - sub_col_offset
    right_x = anchor_x + sub_col_offset

    def _col_down(refs: list[str], col_x: float, start_y: float) -> float:
        return _pwr_place_column_down(ctx, refs, col_x, start_y,
                                      placed_in_col, grid, pz_bounds, strip_gap)

    # Output columns
    left_bottom = _col_down(columns["output_left"], left_x, output_top)
    right_bottom = _col_down(columns["output_right"], right_x, output_top)

    # Bridge
    bridge_top = max(left_bottom, right_bottom)
    bridge_col = columns["bridge"]
    mid = len(bridge_col) // 2 + 1
    fork_y_l = _col_down(bridge_col[:mid], left_x, bridge_top)
    fork_y_r = _col_down(bridge_col[mid:], right_x, bridge_top)
    fork_y = max(fork_y_l, fork_y_r)

    # Buck #2 sub-columns
    col2_x = anchor_x + col_spacing
    col2_left_bottom = _col_down(columns["buck2_left"], col2_x, output_top)
    b2l_max_w = max(
        (ctx.fp_sizes.get(r, (2.0, 2.0))[0] for r in columns["buck2_left"] if r in ctx.fp_sizes),
        default=3.0,
    )
    b2r_max_w = max(
        (ctx.fp_sizes.get(r, (2.0, 2.0))[0] for r in columns["buck2_right"] if r in ctx.fp_sizes),
        default=2.0,
    )
    col2_right_x = col2_x + (b2l_max_w + b2r_max_w) / 2.0 + 0.5
    _col_down(columns["buck2_right"], col2_right_x, output_top)

    # Tail
    tail_y = max(fork_y, col2_left_bottom)
    tail_col = columns["tail"]
    mid_t = len(tail_col) // 2 + 1
    _col_down(tail_col[:mid_t], left_x, tail_y)
    _col_down(tail_col[mid_t:], right_x, tail_y)


def _pwr_place_anchor_ic(
    ctx: PlacementContext,
    buck1_ic: str,
    anchor_x: float,
    u1_y: float,
    u1_w: float,
    u1_h: float,
    placed_in_col: set[str],
    grid: object,
    pz_bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Place buck1 IC at its anchor position. Returns updated (u1_x, u1_y)."""
    pz_x1, pz_y1, pz_x2, pz_y2 = pz_bounds
    if not buck1_ic:
        return anchor_x, u1_y
    # Temporarily unfix — power group phase MUST be able to reposition the buck IC
    # to clear connector bodies and maintain proper power chain flow.
    ctx.fixed_refs.discard(buck1_ic)
    px = max(pz_x1, min(pz_x2, anchor_x))
    py = max(pz_y1, min(pz_y2, u1_y))
    ctx.positions[buck1_ic] = (px, py, 0.0)
    grid.place(px, py, u1_w, u1_h)  # type: ignore[union-attr]
    ctx.power_group_fixed.add(buck1_ic)
    ctx.fixed_refs.add(buck1_ic)  # re-fix after placement
    placed_in_col.add(buck1_ic)
    return px, py


def _phase_power_group(ctx: PlacementContext) -> None:
    """3c1: Power group organization — IC-anchored fork/branch layout."""
    _log.info("  3c1: Power group organization")

    power_group_refs = _collect_feature_refs(ctx, "power", "supply")
    if not power_group_refs:
        return

    power_zone_rect = _find_zone_rect(ctx, "power")
    net_to_pwr_refs = _build_net_to_group_refs(ctx, power_group_refs)
    by_prefix = _classify_refs_by_prefix(power_group_refs, ctx, "U", "J")
    power_ics, power_connectors = by_prefix["U"], by_prefix["J"]

    if not (power_ics and power_zone_rect is not None):
        return

    zx1, zy1, zx2, zy2 = power_zone_rect
    # Adaptive spacing derived from actual passive footprint heights
    passive_heights = [
        ctx.fp_sizes.get(r, (2.0, 2.0))[1]
        for r in power_group_refs if r[0] in "CRDL"
    ]
    avg_h = sum(passive_heights) / len(passive_heights) if passive_heights else 2.0
    strip_gap = max(2.0, avg_h * 0.5)   # 50% of avg height, min 2.0mm (was 0.3mm)
    col_spacing = max(5.0, avg_h * 4.0)  # proportional column spacing
    sub_col_offset = max(2.0, avg_h * 1.75)  # proportional sub-column offset
    placed_in_col: set[str] = set()

    buck1_ic = power_ics[0] if power_ics else ""
    buck2_ic = power_ics[1] if len(power_ics) > 1 else ""
    columns = _classify_power_columns(
        ctx, power_group_refs, power_ics, power_connectors,
        net_to_pwr_refs, buck1_ic, buck2_ic,
    )

    u1_x, u1_y, _u1_rot = ctx.positions.get(buck1_ic, (zx1 + 5.0, zy1 + 15.0, 0.0))
    u1_w, u1_h = ctx.fp_sizes.get(buck1_ic, (5.0, 5.0))
    # Leave room for input connector (terminal block body ~13mm from left edge)
    # plus IC half-width (~4.5mm) plus gap (1mm) = 18.5mm minimum from left
    anchor_x = max(zx1 + 18.0, min(zx2 - col_spacing - 3.0, zx1 + (zx2 - zx1) * 0.35))
    pwr_grid = _build_exclusion_grid(ctx, power_group_refs)
    pz_bounds = (zx1 + 2.0, zy1 + 2.0, zx2 - 2.0, ctx.bounds[3] - 3.0)

    for ref in power_connectors:
        ctx.power_group_fixed.add(ref)

    u1_x, u1_y = _pwr_place_anchor_ic(
        ctx, buck1_ic, anchor_x, u1_y, u1_w, u1_h, placed_in_col, pwr_grid, pz_bounds,
    )
    _pwr_place_column_up(
        ctx, columns["vin_above"], anchor_x, u1_y - u1_h / 2.0 - strip_gap,
        placed_in_col, pwr_grid, pz_bounds, strip_gap,
    )
    _pwr_place_fork_columns(
        ctx, columns, anchor_x, sub_col_offset, col_spacing,
        u1_y + u1_h / 2.0 + strip_gap,
        placed_in_col, pwr_grid, pz_bounds, strip_gap,
    )

    _log.info(
        "    3c1: organized %d power components anchored at %s (%.1f, %.1f)",
        len(ctx.power_group_fixed), buck1_ic, u1_x, u1_y,
    )


def _classify_power_columns(
    ctx: PlacementContext,
    group_refs: set[str],
    power_ics: list[str],
    power_connectors: list[str],
    net_refs: dict[str, set[str]],
    buck1_ic: str,
    buck2_ic: str,
) -> dict[str, list[str]]:
    """Classify power group refs into column lists for placement."""
    # Ferrite detection
    ferrite_refs = sorted(
        [r for r in group_refs
         if r.startswith("L") and r in ctx.positions
         and "ferrite" in (
             next((fp.value for fp in ctx.initial_pcb.footprints
                   if fp.ref == r), "")
         ).lower()],
    )
    ic_conn_set = set(power_ics) | set(power_connectors)

    # Buck #1 column
    vin_passives = sorted(net_refs.get("VIN", set()) - ic_conn_set)
    bst1_passives = sorted(
        (net_refs.get("BST", set()) | net_refs.get("SW", set()))
        - {buck1_ic} - set(power_connectors),
    )
    l1_refs = sorted(
        [r for r in group_refs
         if r.startswith("L") and r in ctx.positions
         and r not in ferrite_refs
         and r in net_refs.get("SW", set())],
    )
    fb_refs = sorted(net_refs.get("FB", set()) - {buck1_ic} - set(power_connectors))
    buck5v_caps = sorted(
        net_refs.get("BUCK_5V", set()) - {buck1_ic}
        - set(power_connectors) - set(fb_refs) - set(l1_refs),
    )

    vin_above = vin_passives
    output_left = bst1_passives + l1_refs + buck5v_caps
    output_right = fb_refs

    # Bridge
    or_diode_refs = sorted(
        net_refs.get("BUCK_5V", set())
        & {r for r in group_refs if r.startswith("D")}
        - set(vin_passives),
    )
    v5_rail_refs = sorted(
        (net_refs.get("+5V", set()) & group_refs)
        - ic_conn_set - set(or_diode_refs),
    )

    # Buck #2 column
    buck2_in_caps = sorted(
        net_refs.get("+5V", set())
        & {r for r in group_refs if r.startswith("C")}
        - {"C4"},
    )
    v5_rail_refs = [r for r in v5_rail_refs if r not in buck2_in_caps]
    bridge = or_diode_refs + v5_rail_refs
    bst2_passives = sorted(
        (net_refs.get("BST2", set()) | net_refs.get("SW2", set()))
        - {buck2_ic} - set(power_connectors),
    )
    l2_refs = sorted(
        [r for r in group_refs
         if r.startswith("L") and r in ctx.positions
         and r not in ferrite_refs
         and r in net_refs.get("SW2", set())],
    )
    v33_caps = sorted(
        net_refs.get("+3V3", set()) & group_refs
        - {buck2_ic} - set(power_connectors),
    )
    buck2_left = [*buck2_in_caps, buck2_ic, *bst2_passives, *l2_refs]
    buck2_right = v33_caps

    # Tail
    led_refs = sorted(
        (net_refs.get("LED_A", set()) & group_refs) - set(power_connectors),
    )
    all_classified = (
        set(vin_above) | {buck1_ic}
        | set(output_left) | set(output_right)
        | set(bridge) | set(buck2_left) | set(buck2_right)
        | set(led_refs) | set(ferrite_refs) | set(power_connectors)
    )
    remaining = sorted(group_refs - all_classified - {""} - ctx.fixed_refs)
    tail = led_refs + ferrite_refs + remaining

    return {
        "vin_above": vin_above,
        "output_left": output_left,
        "output_right": output_right,
        "bridge": bridge,
        "buck2_left": buck2_left,
        "buck2_right": buck2_right,
        "tail": tail,
    }


# Helper functions that need to be imported from helpers