"""ADC channel placement functions.

Contains placement logic for ADC channels, including radial fan placement
and connector positioning.
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

_log = logging.getLogger(__name__)

# Radial placement parameters learned from human reference ADC layout.
# (radius_mm, angle_offset_deg) per component type, keyed by connector
# quadrant relative to ADC IC.
_ADC_RADIAL_TOP: dict[str, tuple[float, float]] = {
    # Connector above ADC IC (top edge) — components fan down-left
    "C_filt": (6.0, -40.0),
    "R_top": (11.0, -20.0),
    "R_bot": (14.0, 20.0),
    "D_tvs": (17.0, 0.0),
}
# Vertical drop from connector to passive row (mm, before scaling).
_ADC_STRIP_DY_MM: float = 8.3


def _adc_place_channel_strip(
    ch_idx: int,
    ic_pin: str,
    passives: list[str],
    ch_x_start: float,
    channel_spacing: float,
    ic_y: float,
    ic_h: float,
    strip_gap: float,
    ctx: PlacementContext,
) -> None:
    """Place one ADC channel's passives along a radial fan from ADC to connector.

    Learned from human-reference layout:
    - Components are placed along the vector from the ADC IC toward each
      channel's input connector.
    - C_filt (filter cap) closest to ADC (~5-7mm), R_top at medium distance
      (~9-14mm), R_bot offset toward board center, D_tvs along the path.
    - All passives rotated 0deg (matching human convention).
    - When no connector position is known, falls back to a vertical strip.

    The radial placement produces a fan-like layout where each channel's
    signal chain is visually distinct and follows the physical signal path
    from connector to ADC pin.
    """
    from kicad_pipeline.optimization.ee_phases_groups import (
        _clamp_to_bounds,
        _find_channel_connector_pos,
        _find_adc_ic_pos_for_channel,
        _adc_place_channel_radial,
    )

    bounds = ctx.bounds
    r_refs = sorted([r for r in passives if r.startswith("R")])
    d_refs = [r for r in passives if r.startswith("D")]
    c_refs = [r for r in passives if r.startswith("C")]

    ch_x = ch_x_start + ch_idx * channel_spacing

    # --- Try radial fan placement ---
    # Find the connector for this channel via net connectivity
    connector_pos = _find_channel_connector_pos(passives, ctx)
    # Find ADC IC position (the IC this channel feeds into)
    adc_ic_pos = _find_adc_ic_pos_for_channel(passives, ctx)

    if connector_pos is not None and adc_ic_pos is not None:
        _adc_place_channel_radial(
            r_refs, d_refs, c_refs,
            adc_ic_pos, connector_pos, bounds, ctx,
        )
        return

    # --- Fallback: vertical strip (original behavior) ---
    strip_order: list[str] = []
    if len(r_refs) >= 2:
        strip_order.append(r_refs[1])
    strip_order.extend(d_refs)
    strip_order.extend(c_refs)
    if len(r_refs) >= 1:
        strip_order.append(r_refs[0])

    strip_y = ic_y - ic_h / 2.0 - 1.0
    for ref in strip_order:
        if ref not in ctx.positions or ref in ctx.fixed_refs:
            continue
        _raw_w, _raw_h = ctx.fp_sizes.get(ref, (2.0, 2.0))
        h = _raw_h
        target_x, target_y = _clamp_to_bounds(ch_x, strip_y - h / 2.0, bounds)
        ctx.positions[ref] = (target_x, target_y, 0.0)
        ctx.adc_channel_refs.add(ref)
        strip_y = target_y - (h / 2.0 + strip_gap)


def _adc_place_channel_radial(
    r_refs: list[str],
    d_refs: list[str],
    c_refs: list[str],
    adc_pos: tuple[float, float],
    connector_pos: tuple[float, float],
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Place ADC channel passives in a radial fan from ADC IC toward connector.

    Learned from human reference board (60x40mm analog input board):
    - ADC IC at bottom, connectors at top edge
    - Each channel's passives interpolated along the IC-to-connector vector
    - Component type determines radius and angular offset from that vector
    - Connector quadrant (above/below IC) selects different offset tables
    """
    from kicad_pipeline.optimization.ee_phases_groups import _clamp_to_bounds

    ic_x, ic_y = adc_pos
    jx, jy = connector_pos
    board_h = bounds[3] - bounds[1]

    # Base angle from ADC IC toward connector
    base_angle = math.atan2(jy - ic_y, jx - ic_x)

    # Select radial parameters based on connector quadrant
    params = _ADC_RADIAL_TOP if jy < (bounds[1] + board_h * 0.5) else _ADC_RADIAL_BOTTOM

    # Build ref-to-role mapping
    role_map: dict[str, str] = {}
    if len(r_refs) >= 1:
        role_map[r_refs[0]] = "R_top"    # lower-numbered R = top of divider
    if len(r_refs) >= 2:
        role_map[r_refs[1]] = "R_bot"
    for d in d_refs:
        role_map[d] = "D_tvs"
    for c in c_refs:
        role_map[c] = "C_filt"

    for ref, role in role_map.items():
        if ref not in ctx.positions or ref in ctx.fixed_refs:
            continue
        if role not in params:
            continue
        radius, angle_off_deg = params[role]
        angle = base_angle + math.radians(angle_off_deg)
        new_x = ic_x + radius * math.cos(angle)
        new_y = ic_y + radius * math.sin(angle)
        # Clamp inside board bounds
        new_x, new_y = _clamp_to_bounds(new_x, new_y, bounds)
def _adc_assign_passive_role(
    ref: str, r_refs: list[str],
) -> str | None:
    """Map a passive ref to its role in the ADC channel strip.

    R refs are ordered numerically; the lower-numbered is R_top (top of
    voltage divider), the higher-numbered is R_bot.
    """
    if ref.startswith("D"):
        return "D_tvs"
    if ref.startswith("C"):
        return "C_filt"
    if ref.startswith("R") and ref in r_refs:
        idx = r_refs.index(ref)
        if idx == 0:
            return "R_top"
        if idx == 1:
            return "R_bot"
    return None


def _adc_init_ctx_defaults(ctx: PlacementContext) -> None:
    """Set ctx ADC attributes to empty defaults when no channels are detected."""
    ctx._adc_channels = []  # type: ignore[attr-defined]
    ctx._ic_channels = {}  # type: ignore[attr-defined]
    ctx._r_top_connector_x = {}  # type: ignore[attr-defined]
    ctx._occupied_x_ranges = []  # type: ignore[attr-defined]
    ctx._CHANNEL_SPACING_MM = 8.0  # type: ignore[attr-defined]
    ctx._STRIP_GAP_MM = 2.5  # type: ignore[attr-defined]  # was 1.5 — caused collisions


def _adc_group_by_ic(
    adc_channels: list[tuple[str, str, list[str]]],
    ctx: PlacementContext,
) -> dict[str, list[tuple[str, list[str]]]]:
    """Group ADC channels by IC ref and register IC refs on ctx."""
    ic_channels: dict[str, list[tuple[str, list[str]]]] = {}
    for ic_ref, ic_pin, passives in adc_channels:
        ic_channels.setdefault(ic_ref, []).append((ic_pin, passives))
        ctx.adc_ic_refs.add(ic_ref)
    return ic_channels


def _adc_pin_sort_key(ic_pin: str) -> tuple[int, str]:
    """Extract a numeric sort key from an ADC IC pin name.

    E.g. "AIN0" -> (0, "AIN0"), "AIN7" -> (7, "AIN7"), "A3" -> (3, "A3").
    Falls back to (999, pin_name) if no trailing digits are found.
    """
    import re

    m = re.search(r"(\d+)$", ic_pin)
    if m:
        return (int(m.group(1)), ic_pin)
def _adc_pre_move_to_zone(
    adc_channels: list[tuple[str, str, list[str]]],
    ctx: PlacementContext,
) -> int:
    """Move ADC-related components that are outside the analog zone to its center.

    Returns the number of components moved.
    """
    from kicad_pipeline.optimization.ee_phases_groups import (
        _find_zone_rect,
        _find_channel_connector_ref,
    )

    analog_zone = _find_zone_rect(ctx, "analog")
    if analog_zone is None:
        return 0
    az_x1, az_y1, az_x2, az_y2 = analog_zone
    center_x = (az_x1 + az_x2) / 2.0
    center_y = (az_y1 + az_y2) / 2.0
    moved = 0

    # Collect all refs involved in ADC channels (ICs, passives, connectors)
    all_adc_refs: set[str] = set()
    for ic_ref, _ic_pin, passives in adc_channels:
        all_adc_refs.add(ic_ref)
        all_adc_refs.update(passives)
        j_ref = _find_channel_connector_ref(passives, ctx)
        if j_ref:
            all_adc_refs.add(j_ref)

    for ref in all_adc_refs:
        if ref not in ctx.positions:
            continue
        rx, ry, rrot = ctx.positions[ref]
        if rx < az_x1 or rx > az_x2 or ry < az_y1 or ry > az_y2:
            ctx.positions[ref] = (center_x, center_y, rrot)
            moved += 1
            _log.debug(
                "    3c2: pre-moved %s from (%.1f, %.1f) to analog zone center",
                ref, rx, ry,
            )

    return moved


def _adc_build_channel_connector_list(
    adc_channels: list[tuple[str, str, list[str]]],
    ctx: PlacementContext,
) -> list[tuple[str | None, str, str, list[str]]]:
    """Build sorted (connector_ref, ic_ref, ic_pin, passives) list.

    Sorted by IC pin number (AIN0 < AIN1 < ... < AIN7) for deterministic
    left-to-right ordering.  Falls back to connector ref if pin numbers
    are identical.
    """
    from kicad_pipeline.optimization.ee_phases_groups import _find_channel_connector_ref

    channel_with_connectors: list[tuple[str | None, str, str, list[str]]] = [
        (_find_channel_connector_ref(passives, ctx), ic_ref, ic_pin, passives)
        for ic_ref, ic_pin, passives in adc_channels
    ]
    channel_with_connectors.sort(
        key=lambda t: (_adc_pin_sort_key(t[2]), t[0] or "Z999"),
    )
    return channel_with_connectors


def _adc_compute_zone_scale(
    ctx: PlacementContext,
) -> tuple[float, float, float, float, float, float]:
    """Return (az_x1, az_y1, az_x2, az_y2, sx, sy) from the analog zone."""
    from kicad_pipeline.optimization.ee_phases_groups import _find_zone_rect

    analog_zone = _find_zone_rect(ctx, "analog")
    bounds = ctx.bounds
    az_x1 = analog_zone[0] if analog_zone else bounds[0]
    az_y1 = analog_zone[1] if analog_zone else bounds[1]
    az_x2 = analog_zone[2] if analog_zone else bounds[2]
    az_y2 = analog_zone[3] if analog_zone else bounds[3]
    zone_w = az_x2 - az_x1
    zone_h = az_y2 - az_y1
    sx = zone_w / 60.0  # scale relative to 60mm reference board
    sy = zone_h / 40.0  # scale relative to 40mm reference board
    return az_x1, az_y1, az_x2, az_y2, sx, sy


def _adc_place_connectors(
    channel_with_connectors: list[tuple[str | None, str, str, list[str]]],
    az_x1: float,
    az_y1: float,
    sx: float,
    sy: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> tuple[list[str], float]:
    """Place ADC channel connectors at the top edge of the analog zone.

    Returns (connector_refs, j_spacing_mm).
    """
    from kicad_pipeline.optimization.ee_phases_groups import _clamp_to_bounds

    n_channels = len(channel_with_connectors)
    if n_channels > 1:
        j_x_start = az_x1 + 12.35 * sx
        j_x_end = az_x1 + 46.35 * sx
        j_spacing = (j_x_end - j_x_start) / (n_channels - 1)
    else:
        j_x_start = az_x1 + (az_x1 + 60.0 * sx - az_x1) * 0.5
        j_spacing = 0.0

    j_y = az_y1 + 4.5 * sy
    connector_refs: list[str] = []

    for ch_idx, (j_ref, _ic_ref, _ic_pin, _passives) in enumerate(
        channel_with_connectors,
    ):
        if j_ref is None or j_ref not in ctx.positions:
            continue
        j_x = j_x_start + ch_idx * j_spacing
        j_x_clamped, j_y_clamped = _clamp_to_bounds(j_x, j_y, bounds)
        # ADC connectors sit near top edge — wire entry faces outward (rot=0)
        ctx.positions[j_ref] = (j_x_clamped, j_y_clamped, 0.0)
        ctx.fixed_refs.add(j_ref)
        ctx.adc_channel_refs.add(j_ref)
        connector_refs.append(j_ref)
        _log.info(
            "    3c2: connector %s -> (%.1f, %.1f) rot=0",
            j_ref, j_x_clamped, j_y_clamped,
        )
def _adc_place_passive_strips(
    channel_with_connectors: list[tuple[str | None, str, str, list[str]]],
    sx: float,
    sy: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Place each channel's passives in a vertical strip below its connector.

    Vertical order (top to bottom): R_top, R_bot, D_tvs, C_filt.
    Each passive is spaced *_ADC_VERTICAL_STRIP_DY* mm apart vertically,
    centered on the connector's X position.
    """
    from kicad_pipeline.optimization.ee_phases_groups import _clamp_to_bounds

    # Vertical spacing between passives within a strip (mm, before scaling)
    vertical_dy = 2.5 * sy
    # Vertical offset from connector to first passive
    strip_start_dy = _ADC_STRIP_DY_MM * sy
    # Vertical role order: signal flows from connector down through divider
    vertical_role_order: list[str] = ["R_top", "R_bot", "D_tvs", "C_filt"]
    vertical_role_rot: dict[str, float] = {
        "R_top": 90.0,
        "R_bot": 90.0,
        "D_tvs": 0.0,
        "C_filt": 90.0,
    }

    for ch_idx, (j_ref, _ic_ref, _ic_pin, passives) in enumerate(
        channel_with_connectors,
    ):
        if j_ref is None or j_ref not in ctx.positions:
            continue
        jx, jy, _jrot = ctx.positions[j_ref]
        r_refs = sorted(r for r in passives if r.startswith("R"))

        # Build role -> ref mapping for this channel
        role_to_ref: dict[str, str] = {}
        for ref in passives:
            if ref not in ctx.positions or ref in ctx.fixed_refs:
                continue
            role = _adc_assign_passive_role(ref, r_refs)
            if role is not None:
                role_to_ref[role] = ref

        # Place in vertical order below connector
        for slot_idx, role in enumerate(vertical_role_order):
            ref = role_to_ref.get(role)
            if ref is None:
                continue
            rot = vertical_role_rot.get(role, 0.0)
            new_y = jy + strip_start_dy + slot_idx * vertical_dy
            new_x, new_y_clamped = _clamp_to_bounds(jx, new_y, bounds)
            ctx.positions[ref] = (new_x, new_y_clamped, rot)
            ctx.adc_channel_refs.add(ref)
            ctx.fixed_refs.add(ref)

        _log.info("    3c2: ch%d passives placed as vertical strip below %s", ch_idx, j_ref)


def _adc_place_ics_and_decoupling(
    az_x1: float,
    az_y1: float,
    sx: float,
    sy: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Place ADC IC(s) and their decoupling caps below the channel strips."""
    from kicad_pipeline.optimization.ee_phases_groups import _clamp_to_bounds

    ic_y = az_y1 + 28.58 * sy
    for ic_ref in sorted(ctx.adc_ic_refs):
        if ic_ref not in ctx.positions:
            continue
        ic_x = az_x1 + 45.75 * sx
        ic_x_clamped, ic_y_clamped = _clamp_to_bounds(ic_x, ic_y, bounds)
        ctx.positions[ic_ref] = (ic_x_clamped, ic_y_clamped, -90.0)
        ctx.fixed_refs.add(ic_ref)
        _log.info(
            "    3c2: ADC IC %s -> (%.1f, %.1f) rot=-90",
            ic_ref, ic_x_clamped, ic_y_clamped,
        )
        _adc_place_decoupling_caps(ic_ref, ic_x_clamped, ic_y_clamped, sy, bounds, ctx)


def _adc_place_decoupling_caps(
    ic_ref: str,
    ic_x: float,
    ic_y: float,
    sy: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> None:
    """Place decoupling caps connected to *ic_ref* in a column below it.

    Each cap is offset by 2.5mm along Y to avoid stacking them all at
    the same position (which created ~36 collisions in Run 5).
    """
    from kicad_pipeline.optimization.ee_phases_groups import _clamp_to_bounds

    cap_idx = 0
    _CAP_PITCH = 2.5  # mm between cap centers in the column
    for net in ctx.requirements.nets:
        if not any(c.ref == ic_ref for c in net.connections):
            continue
        for c in net.connections:
            if (c.ref.startswith("C")
                    and c.ref in ctx.positions
                    and c.ref not in ctx.adc_channel_refs
                    and c.ref not in ctx.fixed_refs):
                y_offset = 3.3 + cap_idx * _CAP_PITCH
                cap_x, cap_y = _clamp_to_bounds(
                    ic_x, ic_y + y_offset * sy, bounds,
                )
                ctx.positions[c.ref] = (cap_x, cap_y, 180.0)
                ctx.adc_channel_refs.add(c.ref)
                ctx.fixed_refs.add(c.ref)
                cap_idx += 1
                _log.info("    3c2: ADC decoupling %s -> (%.1f, %.1f)", c.ref, cap_x, cap_y)


def _adc_place_i2c_pullups(
    sx: float,
    sy: float,
    bounds: tuple[float, float, float, float],
    ctx: PlacementContext,
) -> list[str]:
    """Place I2C pull-up resistors near the first ADC IC.

    Returns the list of placed pull-up refs.
    """
    from kicad_pipeline.optimization.ee_phases_groups import (
        _find_i2c_pullup_refs,
        _clamp_to_bounds,
    )

    i2c_pullups = _find_i2c_pullup_refs(
        ctx.adc_ic_refs, ctx, ctx.adc_channel_refs | ctx.fixed_refs,
    )
    if not i2c_pullups or not ctx.adc_ic_refs:
        return i2c_pullups

    anchor_ic = sorted(ctx.adc_ic_refs)[0]
    if anchor_ic not in ctx.positions:
        return i2c_pullups

    aix, aiy, _airot = ctx.positions[anchor_ic]
    for pidx, pr_ref in enumerate(i2c_pullups):
        pr_x, pr_y = _clamp_to_bounds(
            aix + 7.0 * sx, aiy + (-6.0 + pidx * 3.0) * sy, bounds,
        )
        ctx.positions[pr_ref] = (pr_x, pr_y, 0.0)
        ctx.adc_channel_refs.add(pr_ref)
        ctx.fixed_refs.add(pr_ref)
        _log.info("    3c2: I2C pullup %s -> (%.1f, %.1f)", pr_ref, pr_x, pr_y)
    return i2c_pullups


def _adc_build_occupied_x_ranges(
    connector_refs: list[str],
    ctx: PlacementContext,
) -> list[tuple[float, float]]:
    """Return occupied X ranges for downstream phases based on connector positions."""
    if not connector_refs:
        return []
    placed = [ctx.positions[r][0] for r in connector_refs if r in ctx.positions]
    if not placed:
        return []
    return [(min(placed) - 5.0, max(placed) + 5.0)]