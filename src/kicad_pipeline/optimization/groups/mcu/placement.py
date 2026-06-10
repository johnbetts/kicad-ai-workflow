"""MCU group placement helpers.

Contains MCU-specific placement phases that organize components
within the microcontroller functional group.

Extracted from ``ee_phases_groups.py`` to reduce module size.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from kicad_pipeline.optimization.collision_resolver import (
    _PlacementGrid,
    _rotation_aware_size,
)

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import PlacementContext
from kicad_pipeline.pcb.pin_map import (
    origin_to_centroid,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCU sub-step helpers
# ---------------------------------------------------------------------------

def _mcu_place_u3(
    ctx: PlacementContext,
    mcu_ref: str,
    mcu_w: float,
    mcu_h: float,
) -> tuple[float, float, float, float, float]:
    """Place the MCU IC (Step 0). Returns (mcu_x, mcu_y, eff_w, eff_h, rotation)."""
    bounds = ctx.bounds
    mcu_zone_rect = _find_zone_rect(ctx, "mcu")
    _mcu_rot = 180.0
    eff_w, eff_h = mcu_w, mcu_h

    mcu_fp = None
    for fp in ctx.initial_pcb.footprints:
        if fp.ref == mcu_ref:
            mcu_fp = fp
            break

    if mcu_fp is not None:
        from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space as _pad_ext
        _trial_ox = (bounds[0] + bounds[2]) / 2.0
        _trial_oy = (bounds[1] + bounds[3]) / 2.0
        _te = _pad_ext(mcu_fp, _trial_ox, _trial_oy, _mcu_rot)
        _pad_bot = _te[3] - _trial_oy
        _pad_top = _te[1] - _trial_oy
        # Leave 10mm at top for USB-C connector + CC resistors
        mcu_origin_y = bounds[3] - _pad_bot - 2.0
        # Ensure top pads are at least 10mm from top edge (room for connectors)
        top_clearance = bounds[1] - _pad_top + 10.0
        if _trial_oy + (mcu_origin_y - _trial_oy) + _pad_top < bounds[1] + 10.0:
            mcu_origin_y = max(mcu_origin_y, top_clearance)
        if mcu_zone_rect is not None:
            _zx1, _zy1, _zx2, _zy2 = mcu_zone_rect
            mcu_group_refs = _collect_feature_refs(ctx, "mcu", "controller", "processor")
            _j14_w = (ctx.fp_sizes.get("J14", _DEFAULT_J14_SIZE_MM)[0]
                      if "J14" in mcu_group_refs else 0.0)
            mcu_origin_x = (_zx1 + _zx2 - _j14_w) / 2.0
        else:
            mcu_origin_x = bounds[2] - eff_w / 2.0 - 10.0
        _pad_left = _te[0] - _trial_ox
        _pad_right = _te[2] - _trial_ox
        mcu_origin_x = _clamp(mcu_origin_x,
                               bounds[0] - _pad_left + 2.0,
                               bounds[2] - _pad_right - 2.0)
        mcu_x, mcu_y = origin_to_centroid(mcu_fp, mcu_origin_x,
                                           mcu_origin_y, _mcu_rot)
    else:
        if mcu_zone_rect is not None:
            _zx1, _zy1, _zx2, _zy2 = mcu_zone_rect
            mcu_x = (_zx1 + _zx2) / 2.0
        else:
            mcu_x = bounds[2] - eff_w / 2.0 - 10.0
        mcu_y = bounds[3] - eff_h / 2.0 - 5.0

    mcu_y = _clamp(mcu_y, bounds[1] + eff_h / 2.0 + 2.0,
                   bounds[3] - eff_h / 2.0 - 5.0)
    mcu_x = _clamp(mcu_x, bounds[0] + eff_w / 2.0 + 2.0,
                   bounds[2] - eff_w / 2.0 - 2.0)
    ctx.positions[mcu_ref] = (mcu_x, mcu_y, _mcu_rot)
    ctx.mcu_peripheral_refs.add(mcu_ref)
    ctx.fixed_refs.add(mcu_ref)
    _log.info("    U3 centroid at (%.1f, %.1f) rot=180 [eff_w=%.1f, eff_h=%.1f]",
              mcu_x, mcu_y, eff_w, eff_h)
    return mcu_x, mcu_y, eff_w, eff_h, _mcu_rot


def _mcu_power_pin_board_pos(
    ctx: PlacementContext,
    mcu_ref: str,
    mcu_x: float,
    mcu_y: float,
    rotation: float,
) -> tuple[float, float] | None:
    """Return board-space (x, y) of the 3V3/VCC power pad on the MCU footprint.

    Scans all pads for power-net membership.  Returns the centroid of all
    matching pad positions in board space, or ``None`` if not determinable.
    The result is used to position decoupling caps on the correct side of the
    IC regardless of its rotation.
    """
    mcu_fp = next(
        (fp for fp in ctx.initial_pcb.footprints if fp.ref == mcu_ref), None,
    )
    if mcu_fp is None:
        return None

    # Build set of nets attached to the MCU from requirements
    power_pads: list[tuple[float, float]] = []
    rad = math.radians(-rotation)  # KiCad CW convention
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    for pad in mcu_fp.pads:
        pad_net = pad.net_name if hasattr(pad, "net_name") else ""
        # Look up net via requirements connections
        if not pad_net:
            for net in ctx.requirements.nets:
                for conn in net.connections:
                    if conn.ref == mcu_ref and conn.pin == pad.number:
                        pad_net = net.name
                        break
                if pad_net:
                    break
        if not pad_net:
            continue
        net_up = pad_net.upper()
        is_power = any(
            net_up.startswith(pfx) for pfx in
            ("+3V3", "+3.3V", "VCC", "VDD", "AVCC", "DVCC", "3V3")
        )
        if not is_power:
            continue
        # Rotate footprint-local pad position to board space
        bx = mcu_x + pad.position.x * cos_a - pad.position.y * sin_a
        by = mcu_y + pad.position.x * sin_a + pad.position.y * cos_a
        power_pads.append((bx, by))

    if not power_pads:
        return None
    avg_x = sum(p[0] for p in power_pads) / len(power_pads)
    avg_y = sum(p[1] for p in power_pads) / len(power_pads)
    return avg_x, avg_y


def _mcu_place_decoupling(
    ctx: PlacementContext,
    decoupling_refs: list[str],
    mcu_x: float,
    mcu_y: float,
    mcu_left: float,
    grid: _PlacementGrid,
) -> None:
    """Place decoupling caps within 3-5mm of the MCU power pin (Step 3).

    Detects which side of the IC body has the 3V3/VCC power pad and places
    caps on that side.  With rotation=180 the ESP32's left-column 3V3 pad
    (WEST at rot=0) maps to the EAST (right) side in board coordinates, so
    caps are placed to the RIGHT rather than to the left.
    """
    bounds = ctx.bounds

    # Find the MCU ref and courtyard half-width
    mcu_ref = None
    for ref_c in ctx.mcu_peripheral_refs:
        if ref_c.startswith("U"):
            mcu_ref = ref_c
            break
    if mcu_ref and mcu_ref in ctx.fp_sizes:
        mcu_cw, _mcu_ch = ctx.fp_sizes[mcu_ref]
    else:
        mcu_cw = abs(mcu_x - mcu_left) * 2.0

    courtyard_left = mcu_x - mcu_cw / 2.0
    courtyard_right = mcu_x + mcu_cw / 2.0

    # Determine which horizontal side has the 3V3/VCC power pad.
    rotation = ctx.positions.get(mcu_ref, (0.0, 0.0, 0.0))[2] if mcu_ref else 0.0
    power_pos = (
        _mcu_power_pin_board_pos(ctx, mcu_ref, mcu_x, mcu_y, rotation)
        if mcu_ref
        else None
    )

    if power_pos is not None:
        # Place caps on the same horizontal side as the power pad, 3mm away.
        pwr_x, pwr_y = power_pos
        cap_y_start = pwr_y  # align with power pin row
        if pwr_x > mcu_x:
            # Power pin is to the right → place caps right of MCU
            def _cap_x(w: float) -> float:
                return courtyard_right + w / 2.0 + 0.5
        else:
            # Power pin is to the left → place caps left of MCU
            def _cap_x(w: float) -> float:
                return courtyard_left - w / 2.0 - 0.5
        _log.info(
            "    decoupling: power pin at (%.1f,%.1f) → placing caps on %s side",
            pwr_x, pwr_y, "right" if pwr_x > mcu_x else "left",
        )
    else:
        # Fallback: place left of MCU (original behaviour)
        cap_y_start = mcu_y - 2.0
        def _cap_x(w: float) -> float:
            return courtyard_left - w / 2.0 - 0.5
        _log.info("    decoupling: no power pin found, falling back to left of MCU")

    cap_spacing = 2.5

    for i, ref in enumerate(decoupling_refs):
        w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
        cap_x = _cap_x(w)
        tx = _clamp(cap_x, bounds[0] + w / 2.0 + 0.5, bounds[2] - w / 2.0 - 0.5)
        ty = _clamp(cap_y_start + i * cap_spacing,
                    bounds[1] + h / 2.0 + 0.5, bounds[3] - h / 2.0 - 0.5)
        # Force-place without grid search — grid may push them far away
        ctx.positions[ref] = (tx, ty, 0.0)
        ctx.mcu_peripheral_refs.add(ref)
        ctx.fixed_refs.add(ref)  # protect from clamp/collision phases
        grid.place(tx, ty, w, h)
        dist_to_mcu = ((tx - mcu_x) ** 2 + (ty - mcu_y) ** 2) ** 0.5
        _log.info("    %s (decoupling): FORCE->(%.1f,%.1f) [%.1fmm from MCU]",
                  ref, tx, ty, dist_to_mcu)


def _mcu_place_named_connector(
    ref: str,
    tx: float,
    ty: float,
    ctx: PlacementContext,
    grid: _PlacementGrid,
    rot: float = 0.0,
    max_radius: float = 25.0,
) -> tuple[float, float]:
    """Place a named MCU connector and register it."""
    w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
    bounds = ctx.bounds
    ty = _clamp(ty, bounds[1] + h / 2.0 + 1.0, bounds[3] - h / 2.0 - 1.0)
    tx = _clamp(tx, bounds[0] + w / 2.0 + 1.0, bounds[2] - w / 2.0 - 1.0)
    px, py = grid.find_free_pos(tx, ty, w, h, max_radius=max_radius)
    ctx.positions[ref] = (px, py, rot)
    grid.place(px, py, w, h)
    ctx.mcu_peripheral_refs.add(ref)
    _log.info("    %s -> (%.1f, %.1f)", ref, px, py)
    return px, py


def _mcu_place_connectors(
    ctx: PlacementContext,
    connector_refs: set[str],
    grid: _PlacementGrid,
    mcu_x: float,
    mcu_y: float,
    mcu_left: float,
    mcu_top: float,
    eff_w: float,
    eff_h: float,
) -> None:
    """Place MCU connectors at board edges (Step 4)."""
    bounds = ctx.bounds
    right_edge_x = bounds[2] - 2.0

    if "J14" in connector_refs and "J14" in ctx.positions and "J14" not in ctx.fixed_refs:
        w14, h14 = ctx.fp_sizes.get("J14", _DEFAULT_J14_SIZE_MM)
        _mcu_place_named_connector(
            "J14", bounds[2] - w14 / 2.0 - 1.0, mcu_y, ctx, grid,
        )

    if "J15" in connector_refs and "J15" in ctx.positions and "J15" not in ctx.fixed_refs:
        w15, h15 = ctx.fp_sizes.get("J15", (5.2, 12.9))
        j14_pos = ctx.positions.get("J14")
        if j14_pos:
            j14_bottom = j14_pos[1] + ctx.fp_sizes.get("J14", _DEFAULT_J14_SIZE_MM)[1] / 2.0
            tx = right_edge_x - w15 / 2.0
            ty = j14_bottom + h15 / 2.0 + 2.0
        else:
            tx = right_edge_x - w15 / 2.0
            ty = mcu_y + 10.0
        ty = _clamp(ty, bounds[1] + h15 / 2.0 + 1.5, bounds[3] - h15 / 2.0 - 1.5)
        tx = min(tx, bounds[2] - w15 / 2.0 - 1.5)
        px15, py15 = grid.find_free_pos(tx, ty, w15, h15, max_radius=25.0)
        px15 = min(px15, bounds[2] - w15 / 2.0 - 1.5)
        py15 = min(py15, bounds[3] - h15 / 2.0 - 1.5)
        ctx.positions["J15"] = (px15, py15, 0.0)
        grid.place(px15, py15, w15, h15)
        ctx.mcu_peripheral_refs.add("J15")
        _log.info("    J15 -> right edge, below J14 (%.1f, %.1f)", px15, py15)

    if "J16" in connector_refs and "J16" in ctx.positions and "J16" not in ctx.fixed_refs:
        w16, h16 = ctx.fp_sizes.get("J16", (16.2, 6.9))
        px = bounds[2] - w16 / 2.0
        py = mcu_top - h16 / 2.0 - 2.0
        py = _clamp(py, bounds[1] + h16 / 2.0 + 1.0, bounds[3] - h16 / 2.0 - 1.0)
        ctx.positions["J16"] = (px, py, 0.0)
        grid.place(px, py, w16, h16)
        ctx.mcu_peripheral_refs.add("J16")
        _log.info("    J16 -> right edge, above U3 (%.1f, %.1f)", px, py)

    # J1 (USB-C): FORCE-place at TOP edge, left of MCU (away from antenna).
    # rotation 180 so pads face into the board.
    # We skip grid.find_free_pos because U1's large footprint blocks it.
    # Guard: skip if J1 is owned by the ethernet feature block — ethernet group
    # phase places the RJ45 at the top edge with the correct rotation.
    _eth_refs_j1 = _collect_feature_refs(ctx, "ethernet", "eth")
    if ("J1" in connector_refs and "J1" in ctx.positions
            and "J1" not in ctx.fixed_refs and "J1" not in _eth_refs_j1):
        w1, h1 = ctx.fp_sizes.get("J1", (9.0, 7.5))
        j1_x = mcu_left + 3.0  # left side of MCU, away from antenna
        j1_y = bounds[1] + h1 / 2.0 + 0.5  # near top edge
        j1_x = _clamp(j1_x, bounds[0] + w1 / 2.0 + 1.0, bounds[2] - w1 / 2.0 - 1.0)
        j1_y = _clamp(j1_y, bounds[1] + h1 / 2.0 + 0.5, bounds[3] - h1 / 2.0 - 1.0)
        ctx.positions["J1"] = (j1_x, j1_y, 180.0)
        ctx.mcu_peripheral_refs.add("J1")
        ctx.fixed_refs.add("J1")  # protect from clamp/collision resolution
        _log.info("    J1 (USB-C) -> top edge at (%.1f, %.1f) rot=180", j1_x, j1_y)

    if "J2" in connector_refs and "J2" in ctx.positions and "J2" not in ctx.fixed_refs:
        w2, h2 = ctx.fp_sizes.get("J2", _DEFAULT_J2_SIZE_MM)
        tx = mcu_left - w2 / 2.0 - 8.0
        ty = bounds[3] - h2 / 2.0 - 1.0
        tx = _clamp(tx, bounds[0] + w2 / 2.0 + 1.0, bounds[2] - w2 / 2.0 - 1.0)
        _mcu_place_named_connector("J2", tx, ty, ctx, grid, rot=180.0, max_radius=20.0)


def _mcu_place_led(
    ctx: PlacementContext,
    other_passive_refs: list[str],
    grid: _PlacementGrid,
    sw_base_x: float,
    sw_base_y: float,
) -> None:
    """Place status LED near switches (Step 6b)."""
    bounds = ctx.bounds
    for led_ref in ["LED1"]:
        if (led_ref in other_passive_refs and led_ref in ctx.positions
                and led_ref not in ctx.fixed_refs):
            lw, lh = ctx.fp_sizes.get(led_ref, (2.0, 1.0))
            led_tx, led_ty = _clamp_to_bounds(sw_base_x + 8.0, sw_base_y, bounds)
            lpx, lpy = grid.find_free_pos(led_tx, led_ty, lw, lh, max_radius=10.0)
            ctx.positions[led_ref] = (lpx, lpy, 0.0)
            ctx.mcu_peripheral_refs.add(led_ref)
            grid.place(lpx, lpy, lw, lh)
            if led_ref in other_passive_refs:
                other_passive_refs.remove(led_ref)
            _log.info("    %s (status LED) -> (%.1f, %.1f)", led_ref, lpx, lpy)


def _mcu_place_usb_subcircuit(
    ctx: PlacementContext,
    other_passive_refs: list[str],
    grid: _PlacementGrid,
) -> None:
    """Place USB CC resistors near J1 (USB-C) and ESD near J2 (Step 5)."""
    # --- CC resistors (R3/R4) near J1 (USB-C) ---
    j1_pos = ctx.positions.get("J1")
    bounds = ctx.bounds
    if j1_pos:
        j1x, j1y, _ = j1_pos
        j1w, j1h = ctx.fp_sizes.get("J1", (9.0, 7.5))
        # Find CC resistors: R3/R4 are typical CC1/CC2 resistors
        cc_refs = [r for r in ("R3", "R4") if r in other_passive_refs
                   and r in ctx.positions]
        for i, ref in enumerate(cc_refs):
            w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
            # Place CC resistors just below J1, side by side
            px = j1x - 2.0 + i * (w + 2.0)
            py = j1y + j1h / 2.0 + h / 2.0 + 1.0
            px = _clamp(px, bounds[0] + w / 2.0 + 0.5, bounds[2] - w / 2.0 - 0.5)
            py = _clamp(py, bounds[1] + h / 2.0 + 0.5, bounds[3] - h / 2.0 - 0.5)
            ctx.positions[ref] = (px, py, 0.0)
            ctx.mcu_peripheral_refs.add(ref)
            ctx.fixed_refs.add(ref)
            grid.place(px, py, w, h)
            if ref in other_passive_refs:
                other_passive_refs.remove(ref)
            _log.info("    %s (CC resistor) -> near J1 at (%.1f, %.1f)", ref, px, py)

    # --- USB ESD + series resistors near J2 ---
    j2_pos = ctx.positions.get("J2")
    bounds = ctx.bounds
    if "U9" in other_passive_refs and "U9" in ctx.positions and j2_pos:
        j2x, j2y, _j2r = j2_pos
        j2w, j2h = ctx.fp_sizes.get("J2", _DEFAULT_J2_SIZE_MM)
        u9w, u9h = ctx.fp_sizes.get("U9", (3.0, 3.0))
        u9_tx = j2x - j2w / 4.0
        u9_ty = j2y - j2h / 2.0 - u9h / 2.0 - 6.0
        u9_tx, u9_ty = _clamp_to_bounds(u9_tx, u9_ty, bounds)
        u9_tx = min(u9_tx, bounds[2] - u9w / 2.0 - 1.0)
        u9x, u9y = grid.find_free_pos(u9_tx, u9_ty, u9w, u9h, max_radius=8.0)
        ctx.positions["U9"] = (u9x, u9y, 0.0)
        ctx.mcu_peripheral_refs.add("U9")
        grid.place(u9x, u9y, u9w, u9h)
        other_passive_refs.remove("U9")
        _log.info("    U9 (ESD) -> near J2 at (%.1f, %.1f)", u9x, u9y)

    usb_r_refs = [r for r in ("R6", "R7") if r in other_passive_refs
                  and r in ctx.positions]
    if usb_r_refs and j2_pos:
        j2x_r, j2y_r, _ = j2_pos
        j2w_r = ctx.fp_sizes.get("J2", _DEFAULT_J2_SIZE_MM)[0]
        j2h_r = ctx.fp_sizes.get("J2", _DEFAULT_J2_SIZE_MM)[1]
        for i, ref in enumerate(usb_r_refs):
            w, h = ctx.fp_sizes.get(ref, (1.0, 0.5))
            px = j2x_r - j2w_r / 4.0 + i * (w + 2.0)
            py = j2y_r - j2h_r / 2.0 - h / 2.0 - 1.5
            px, py = _clamp_to_bounds(px, py, bounds)
            ctx.positions[ref] = (px, py, 0.0)
            grid.place(px, py, w, h)
            other_passive_refs.remove(ref)
            _log.info("    %s (USB R) -> (%.1f, %.1f)", ref, px, py)


def _mcu_place_reset_boot(
    ctx: PlacementContext,
    other_passive_refs: list[str],
    ref_nets: dict[str, set[str]],
    grid: _PlacementGrid,
    mcu_x: float,
    mcu_y: float,
    eff_w: float,
    mcu_top: float,
) -> None:
    """Place switch+resistor pairs for reset/boot (Step 6)."""
    bounds = ctx.bounds
    # Find SW+R pairs via net connectivity (not hardcoded refs)
    sw_refs = [r for r in other_passive_refs if r in ctx.positions
               and r.startswith("SW")]
    r_refs = [r for r in other_passive_refs if r in ctx.positions
              and r.startswith("R")]
    sw_pairs: list[tuple[str, str]] = []
    used_sw: set[str] = set()
    for sw in sw_refs:
        sw_nets = ref_nets.get(sw, set())
        for res in r_refs:
            if res in sw_nets and res not in used_sw:
                sw_pairs.append((sw, res))
                used_sw.add(sw)
                used_sw.add(res)
                break
    unique_pairs = sw_pairs
    unpaired_sw = [r for r in sw_refs if r not in used_sw]

    mcu_x - eff_w / 2.0
    # Place switches on the LEFT side of the board, grouped vertically
    sw_base_x = bounds[0] + 6.0  # near left edge
    sw_base_y = mcu_y - 5.0  # near MCU vertical center

    for i, (sw, res) in enumerate(unique_pairs):
        sw_w, sw_h = ctx.fp_sizes.get(sw, (3.5, 3.5))
        r_w, r_h = ctx.fp_sizes.get(res, (1.0, 0.5))
        tx = sw_base_x - i * (sw_w + 1.5)
        ty = sw_base_y
        tx = _clamp(tx, bounds[0] + sw_w / 2.0 + 2.0, bounds[2] - sw_w / 2.0 - 2.0)
        ty = _clamp(ty, bounds[1] + sw_h / 2.0 + 2.0, bounds[3] - sw_h / 2.0 - 2.0)
        px, py = grid.find_free_pos(tx, ty, sw_w, sw_h, max_radius=10.0)
        px = _clamp(px, mcu_x - 20.0, mcu_x + 20.0)
        py = _clamp(py, max(mcu_y - 20.0, bounds[1] + sw_h / 2.0 + 2.0),
                   bounds[3] - sw_h / 2.0 - 2.0)
        ctx.positions[sw] = (px, py, 0.0)
        ctx.mcu_peripheral_refs.add(sw)
        grid.place(px, py, sw_w, sw_h)
        other_passive_refs.remove(sw)
        r_tx, r_ty = _clamp_to_bounds(px, py - sw_h / 2.0 - r_h / 2.0 - 0.5, bounds)
        rpx, rpy = grid.find_free_pos(r_tx, r_ty, r_w, r_h, max_radius=8.0)
        ctx.positions[res] = (rpx, rpy, 0.0)
        ctx.mcu_peripheral_refs.add(res)
        grid.place(rpx, rpy, r_w, r_h)
        other_passive_refs.remove(res)
        _log.info("    %s+%s (reset/boot) -> (%.1f,%.1f) / (%.1f,%.1f)",
                  sw, res, px, py, rpx, rpy)

    for sw in unpaired_sw:
        sw_w, sw_h = ctx.fp_sizes.get(sw, (3.5, 3.5))
        tx = sw_base_x - len(unique_pairs) * (sw_w + 3.0)
        ty = sw_base_y
        tx = _clamp(tx, bounds[0] + sw_w / 2.0 + 1.0, bounds[2] - 2.0)
        ty = _clamp(ty, bounds[1] + sw_h / 2.0 + 1.0, bounds[3] - 2.0)
        px, py = grid.find_free_pos(tx, ty, sw_w, sw_h, max_radius=15.0)
        ctx.positions[sw] = (px, py, 0.0)
        ctx.mcu_peripheral_refs.add(sw)
        grid.place(px, py, sw_w, sw_h)
        other_passive_refs.remove(sw)
        _log.info("    %s (switch) -> (%.1f, %.1f)", sw, px, py)

    return sw_base_x, sw_base_y  # type: ignore[return-value]


def _mcu_place_remaining(
    ctx: PlacementContext,
    remaining: list[str],
    mcu_ref: str,
    mcu_x: float,
    mcu_y: float,
    eff_w: float,
    eff_h: float,
    grid: _PlacementGrid,
) -> None:
    """Place remaining MCU passives in ring slots around the MCU (Step 7)."""
    bounds = ctx.bounds
    mcu_left = mcu_x - eff_w / 2.0
    mcu_right = mcu_x + eff_w / 2.0
    mcu_top = mcu_y - eff_h / 2.0
    mcu_target_gap = 4.0
    mcu_clear = 3.0

    ring_slots: list[tuple[float, float]] = []
    if mcu_top > bounds[1] + 10.0:
        for dx_off in range(-4, 5):
            ring_slots.append((mcu_x + dx_off * mcu_target_gap,
                               mcu_top - mcu_clear - 3.0))
        for dx_off in range(-3, 4):
            ring_slots.append((mcu_x + dx_off * mcu_target_gap,
                               mcu_top - mcu_clear - 7.0))
    slot_x_left = mcu_left - mcu_clear - 2.0
    for dy_off in range(-3, 4):
        ring_slots.append((slot_x_left, mcu_y + dy_off * 3.5))
    slot_x_right = mcu_right + mcu_clear + 2.0
    for dy_off in range(-3, 4):
        ring_slots.append((slot_x_right, mcu_y + dy_off * 3.5))

    slot_idx = 0
    for ref in remaining:
        w, h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        rrot = ctx.positions[ref][2]

        placed = False
        while slot_idx < len(ring_slots):
            sx, sy = ring_slots[slot_idx]
            slot_idx += 1
            sx, sy = _clamp_to_bounds(sx, sy, bounds)
            if grid.is_free(sx, sy, w, h):
                ctx.positions[ref] = (sx, sy, rrot)
                ctx.mcu_peripheral_refs.add(ref)
                grid.place(sx, sy, w, h)
                placed = True
                break

        if not placed:
            tx = mcu_left - mcu_target_gap - w / 2.0
            ty = mcu_y
            tx, ty = _clamp_to_bounds(tx, ty, bounds)
            px, py = grid.find_free_pos(tx, ty, w, h, max_radius=25.0)
            ctx.positions[ref] = (px, py, rrot)
            ctx.mcu_peripheral_refs.add(ref)
            grid.place(px, py, w, h)


# Helper functions that need to be imported from helpers
def _clamp(val: float, lo: float, hi: float) -> float:
    """Clamp *val* to [lo, hi]."""
    return max(lo, min(hi, val))


def _clamp_to_bounds(
    x: float, y: float, bounds: tuple[float, float, float, float],
    margin: float = 2.0,
) -> tuple[float, float]:
    """Clamp (x, y) inside *bounds* with *margin*."""
    return (
        _clamp(x, bounds[0] + margin, bounds[2] - margin),
        _clamp(y, bounds[1] + margin, bounds[3] - margin),
    )


def _collect_feature_refs(
    ctx: PlacementContext, *keywords: str,
) -> set[str]:
    """Return component refs from the first FeatureBlock whose name matches any keyword."""
    for feat in ctx.requirements.features:
        feat_lower = feat.name.lower()
        if any(kw in feat_lower for kw in keywords):
            refs: set[str] = set()
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                refs.add(r)
            return refs
    return set()


def _find_zone_rect(
    ctx: PlacementContext, zone_name: str,
) -> tuple[float, float, float, float] | None:
    """Return the rect of the zone named *zone_name*, or None."""
    for z in ctx.zones:
        if z.name == zone_name:
            return z.rect
    return None


# Constants
_DEFAULT_J14_SIZE_MM: tuple[float, float] = (2.7, 35.7)
"""Default pin-header connector (J14) footprint size."""

_BOARD_EDGE_MARGIN_MM: float = 2.0
"""Margin from board edge for component placement within groups."""


def _mcu_push_courtyard_violations(
    ctx: PlacementContext,
    mcu_ref: str,
    mcu_x: float,
    mcu_y: float,
    eff_w: float,
    eff_h: float,
    mcu_grid: object,
) -> None:
    """Push peripheral refs that overlap the MCU courtyard outward."""
    court_margin = 2.0
    court = (
        mcu_x - eff_w / 2.0 - court_margin,
        mcu_y - eff_h / 2.0 - court_margin,
        mcu_x + eff_w / 2.0 + court_margin,
        mcu_y + eff_h / 2.0 + court_margin,
    )
    for ref in list(ctx.mcu_peripheral_refs):
        if ref == mcu_ref:
            continue
        # Skip refs already in fixed_refs — they were intentionally placed
        # (e.g. J1 at top edge, decoupling caps next to MCU pads).
        if ref in ctx.fixed_refs:
            continue
        if ref.startswith("J") and ctx.positions[ref][0] > mcu_x:
            continue
        _push_component_outside_courtyard(ref, ctx, court, mcu_grid)


def _mcu_place_debounce_caps(
    ctx: PlacementContext, other_passive_refs: list[str],
) -> None:
    ref_net_names: dict[str, set[str]] = {}
    for net in ctx.requirements.nets:
        for conn in net.connections:
            ref_net_names.setdefault(conn.ref, set()).add(net.name)

    for ref in list(other_passive_refs):
        if not ref.startswith("C") or ref not in ctx.positions:
            continue
        cap_nets = ref_net_names.get(ref, set())
        non_gnd = {n for n in cap_nets if "GND" not in n.upper()}
        is_debounce = any(
            kw in n.upper() for n in non_gnd for kw in ("EN", "DEB", "RESET")
        )
        if not is_debounce:
            continue
        sw_candidates = [r for r in ctx.positions
                         if r.startswith("SW") and r in ctx.mcu_peripheral_refs]
        if not sw_candidates:
            continue
        best_sw = sw_candidates[0]
        for sw in sw_candidates:
            sw_nets = ref_net_names.get(sw, set())
            sw_non_gnd = {n for n in sw_nets if "GND" not in n.upper()}
            for cn in non_gnd:
                for sn in sw_non_gnd:
                    if any(kw in cn.upper() and kw in sn.upper()
                           for kw in ("EN", "RESET", "BOOT")):
                        best_sw = sw
        tx, ty, _ = ctx.positions[best_sw]
        tw, th = ctx.fp_sizes.get(ref, (1.0, 0.5))
        sw_w, _sw_h = ctx.fp_sizes.get(best_sw, (4.0, 4.0))
        cx = tx + sw_w / 2.0 + tw / 2.0 + 0.5
        cy = ty
        cx = _clamp(cx, ctx.bounds[0] + tw / 2.0 + 0.5, ctx.bounds[2] - tw / 2.0 - 0.5)
        cy = _clamp(cy, ctx.bounds[1] + th / 2.0 + 0.5, ctx.bounds[3] - th / 2.0 - 0.5)
        ctx.positions[ref] = (cx, cy, 0.0)
        ctx.mcu_peripheral_refs.add(ref)
        ctx.fixed_refs.add(ref)
        other_passive_refs.remove(ref)
        _log.info("    %s (debounce) -> near %s at (%.1f, %.1f)", ref, best_sw, cx, cy)