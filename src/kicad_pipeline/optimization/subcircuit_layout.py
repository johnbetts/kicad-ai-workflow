"""Stage 1: Subcircuit layout engine — pin-connectivity-driven placement.

Places each detected subcircuit's components relative to the anchor IC
at (0, 0) using net connectivity to determine position, not component
type.  Each subcircuit type has a dedicated template that encodes EE
domain knowledge (decoupling near VCC pin, flyback diode on coil side,
crystal load caps symmetric, etc.).

The output ``SubCircuitLayout`` carries relative positions and a convex
hull polygon so Stage 2 can pack subcircuits into groups without
component-level collision resolution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.optimization.functional_grouper import SubCircuitType
from kicad_pipeline.optimization.geometry import convex_hull

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.functional_grouper import DetectedSubCircuit

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout constants (mm)
# ---------------------------------------------------------------------------

# Gaps must exceed COMPONENT_CLEARANCE_GAP_MM (0.5mm) + max courtyard
# dimension to guarantee zero AABB collisions.
# Largest passive courtyard: C_1206 = 5.2x3.0mm → need >3.5mm between centers.
# Largest IC courtyard: ESP32 = 19.2x19.0mm → need >10mm from IC edge to passive.
_PASSIVE_GAP: float = 4.0       # center-to-center gap between passives
_IC_PASSIVE_GAP: float = 4.0    # IC courtyard edge to passive center
_RELAY_GAP: float = 1.0         # gap between adjacent relays
_COIL_GAP: float = 12.0         # relay bottom to first driver row — must clear 17.7mm relay courtyard + Edge.Cuts isolation cutout arc (~5mm beyond courtyard edge) + 0.5mm copper-edge clearance
_ROW_GAP: float = 4.0           # between driver component rows
_CAP_OFFSET: float = 4.0        # decoupling cap offset from IC edge (courtyard-safe)
_CRYSTAL_CAP_OFFSET: float = 3.0  # crystal load cap offset
_LINEAR_GAP: float = 4.0        # gap in linear signal chains
_RADIAL_RADIUS: float = 5.0     # radial placement distance from anchor
_DEFAULT_SIZE: tuple[float, float] = (2.0, 1.0)  # fallback component size


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubCircuitLayout:
    """Layout of a single subcircuit with positions relative to anchor.

    All positions are ``(dx, dy, rotation)`` offsets from the anchor
    component at ``(0, 0, 0)``.  The polygon is a convex hull around
    all component footprint rectangles.
    """

    subcircuit: DetectedSubCircuit
    positions: dict[str, tuple[float, float, float]]  # ref → (dx, dy, rot)
    polygon: tuple[tuple[float, float], ...]           # convex hull vertices
    anchor_ref: str
    width: float
    height: float
    signal_flow: tuple[str, ...] = ()  # ordered refs from input to output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fp_wh(
    ref: str,
    fp_sizes: dict[str, tuple[float, float]],
    rotation: float = 0.0,
) -> tuple[float, float]:
    """Return (width, height) accounting for rotation."""
    w, h = fp_sizes.get(ref, _DEFAULT_SIZE)
    if abs(rotation % 180 - 90) < 10:
        w, h = h, w
    return w, h


def _component_corners(
    dx: float, dy: float,
    w: float, h: float,
) -> list[tuple[float, float]]:
    """Return 4 bounding-box corners for a component at (dx, dy)."""
    hw, hh = w / 2.0, h / 2.0
    return [
        (dx - hw, dy - hh),
        (dx + hw, dy - hh),
        (dx + hw, dy + hh),
        (dx - hw, dy + hh),
    ]


def _build_polygon(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Compute convex hull polygon from component positions and sizes."""
    corners: list[tuple[float, float]] = []
    for ref, (dx, dy, rot) in positions.items():
        w, h = _fp_wh(ref, fp_sizes, rot)
        corners.extend(_component_corners(dx, dy, w, h))
    if not corners:
        return ((0.0, 0.0),)
    return convex_hull(tuple(corners))


def _polygon_dims(
    polygon: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    """Return (width, height) of a polygon's bounding box."""
    if not polygon:
        return (0.0, 0.0)
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (max(xs) - min(xs), max(ys) - min(ys))


def _net_map(
    requirements: ProjectRequirements,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return (ref→nets, net→refs) maps."""
    ref_to_nets: dict[str, set[str]] = {}
    net_to_refs: dict[str, set[str]] = {}
    for net in requirements.nets:
        refs = {c.ref for c in net.connections}
        net_to_refs[net.name] = refs
        for c in net.connections:
            ref_to_nets.setdefault(c.ref, set()).add(net.name)
    return ref_to_nets, net_to_refs


def _is_power_or_gnd(name: str) -> bool:
    """Check if net is power or ground."""
    upper = name.upper()
    gnd_prefixes = ("GND", "AGND", "DGND", "VSS", "AVSS")
    if any(upper == p or upper.startswith(p + "_") for p in gnd_prefixes):
        return True
    if upper in ("", "VCC", "VBUS"):
        return True
    # +3V3, +5V, +24V, VIN, VOUT, etc.
    return upper.startswith(("+", "V"))


def _signal_nets_between(
    ref_a: str,
    ref_b: str,
    ref_to_nets: dict[str, set[str]],
) -> set[str]:
    """Find signal (non-power) nets shared between two refs."""
    nets_a = ref_to_nets.get(ref_a, set())
    nets_b = ref_to_nets.get(ref_b, set())
    return {n for n in nets_a & nets_b if not _is_power_or_gnd(n)}


def _sort_refs_by_prefix(
    refs: list[str],
    order: tuple[str, ...],
) -> list[str]:
    """Sort refs by prefix priority order, then by ref number."""
    prefix_rank = {p: i for i, p in enumerate(order)}

    def key(r: str) -> tuple[int, str]:
        p = r.rstrip("0123456789")
        return (prefix_rank.get(p, 99), r)

    return sorted(refs, key=key)


# ---------------------------------------------------------------------------
# Per-type layout templates
# ---------------------------------------------------------------------------


def _layout_relay_driver(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """Relay driver: relay at (0,0) rotated 90deg, support in column below.

    Layout (top to bottom):
        K* (relay, 90deg rotation, COM faces top)
        D* (flyback diode)
        Q* (driver transistor)
        R* (gate/LED resistors)
        D* (LED indicators, if any)
    """
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref

    # Anchor relay at origin, rotated 90deg
    positions[anchor] = (0.0, 0.0, 90.0)

    # Get relay size at 90deg rotation
    raw_w, raw_h = fp_sizes.get(anchor, (17.7, 15.6))
    relay_h = raw_w  # at 90deg, height becomes the original width

    # Support components go below the relay
    support = [r for r in sc.refs if r != anchor]
    d_refs = sorted(r for r in support if r.startswith("D"))
    q_refs = sorted(r for r in support if r.startswith("Q"))
    r_refs = sorted(r for r in support if r.startswith("R"))
    other = sorted(r for r in support if r not in d_refs + q_refs + r_refs)

    # Compute coil gap from actual relay courtyard + clearance
    # At 90deg rotation, relay_h = original width (the tall dimension).
    # Need enough gap that the first driver component clears the relay courtyard.
    # Get the first driver component height to compute the gap.
    first_driver_h = 2.0  # default
    for refs_group in [d_refs, q_refs, r_refs, other]:
        if refs_group:
            first_driver_h = max(_fp_wh(r, fp_sizes, 90.0)[1] for r in refs_group)
            break
    # Gap = relay_half_h + driver_half_h + clearance (0.5mm) + margin (0.5mm)
    coil_gap = max(_COIL_GAP, first_driver_h / 2.0 + 1.5)
    col_y = relay_h / 2.0 + coil_gap

    for refs, rot in [(d_refs, 90.0), (q_refs, 180.0), (r_refs, 270.0), (other, 0.0)]:
        if not refs:
            continue
        # Courtyard-aware step: max height in this row type + 1mm gap
        all_heights = [_fp_wh(r, fp_sizes, rot)[1] for r in refs]
        row_step = max(all_heights, default=1.0) + 1.0
        for ref in refs:
            w, h = _fp_wh(ref, fp_sizes, rot)
            py = col_y + h / 2.0
            positions[ref] = (0.0, py, rot)
            col_y = py + h / 2.0 + row_step

    return positions


def _layout_buck_converter(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """Buck converter: linear signal flow Cin → IC → L → Cout.

    Components placed left-to-right in signal flow order:
    input cap(s) → regulator IC → inductor → output cap(s)
    Feedback resistors below the IC.
    """
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    ref_to_nets, _ = _net_map(requirements)

    positions[anchor] = (0.0, 0.0, 0.0)
    aw, ah = _fp_wh(anchor, fp_sizes)

    support = [r for r in sc.refs if r != anchor]

    # Classify support components
    inductors = [r for r in support if r.startswith("L")]
    caps = [r for r in support if r.startswith("C")]
    resistors = [r for r in support if r.startswith("R")]
    diodes = [r for r in support if r.startswith("D")]

    # Input caps go left of IC, output caps go right (after inductor)
    # Heuristic: caps sharing a net with the inductor are output caps
    inductor_nets: set[str] = set()
    for l_ref in inductors:
        inductor_nets |= ref_to_nets.get(l_ref, set())

    output_caps = [c for c in caps
                   if ref_to_nets.get(c, set()) & inductor_nets - {"GND"}]
    input_caps = [c for c in caps if c not in output_caps]

    # Place input caps to the left
    sorted_input = sorted(input_caps)
    input_step = max((_fp_wh(r, fp_sizes)[0] for r in sorted_input), default=1.0) + 1.0 if sorted_input else 0.0
    x = -(aw / 2.0 + _IC_PASSIVE_GAP)
    for ref in sorted_input:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (x - w / 2.0, 0.0, 0.0)
        x -= input_step

    # Place inductor to the right
    right_chain = sorted(inductors) + sorted(output_caps)
    right_step = max((_fp_wh(r, fp_sizes)[0] for r in right_chain), default=1.0) + 1.0 if right_chain else 0.0
    x = aw / 2.0 + _IC_PASSIVE_GAP
    for ref in sorted(inductors):
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (x + w / 2.0, 0.0, 0.0)
        x += right_step

    # Place output caps after inductor
    for ref in sorted(output_caps):
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (x + w / 2.0, 0.0, 0.0)
        x += right_step

    # Feedback resistors below IC — courtyard-aware vertical step
    sorted_res = sorted(resistors)
    res_step = max((_fp_wh(r, fp_sizes)[1] for r in sorted_res), default=1.0) + 1.0 if sorted_res else 0.0
    y = ah / 2.0 + _IC_PASSIVE_GAP
    for ref in sorted_res:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (0.0, y + h / 2.0, 0.0)
        y += res_step

    # Diodes (bootstrap, etc.) above IC — courtyard-aware vertical step
    sorted_diodes = sorted(diodes)
    diode_step = max((_fp_wh(r, fp_sizes)[1] for r in sorted_diodes), default=1.0) + 1.0 if sorted_diodes else 0.0
    y = -(ah / 2.0 + _IC_PASSIVE_GAP)
    for ref in sorted_diodes:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (0.0, y - h / 2.0, 0.0)
        y -= diode_step

    return positions


def _layout_ldo_regulator(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """LDO: linear Cin → IC → Cout, compact layout."""
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    aw, ah = _fp_wh(anchor, fp_sizes)

    support = sorted(r for r in sc.refs if r != anchor)
    caps = [r for r in support if r.startswith("C")]
    others = [r for r in support if r not in caps]

    # Input cap left, output cap right
    if len(caps) >= 2:
        cin, cout = caps[0], caps[1]
        w1, _ = _fp_wh(cin, fp_sizes)
        positions[cin] = (-(aw / 2.0 + _CAP_OFFSET + w1 / 2.0), 0.0, 0.0)
        w2, _ = _fp_wh(cout, fp_sizes)
        positions[cout] = (aw / 2.0 + _CAP_OFFSET + w2 / 2.0, 0.0, 0.0)
        remaining_caps = caps[2:]
    elif len(caps) == 1:
        w1, _ = _fp_wh(caps[0], fp_sizes)
        positions[caps[0]] = (aw / 2.0 + _CAP_OFFSET + w1 / 2.0, 0.0, 0.0)
        remaining_caps = []
    else:
        remaining_caps = []

    # Extra caps and other components below — courtyard-aware vertical step
    below_refs = remaining_caps + others
    below_step = max((_fp_wh(r, fp_sizes)[1] for r in below_refs), default=1.0) + 1.0 if below_refs else 0.0
    y = ah / 2.0 + _IC_PASSIVE_GAP
    for ref in below_refs:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (0.0, y + h / 2.0, 0.0)
        y += below_step

    return positions


def _layout_crystal_osc(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """Crystal oscillator: Y at center, load caps symmetric left/right."""
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    aw, _ = _fp_wh(anchor, fp_sizes)

    caps = sorted(r for r in sc.refs if r != anchor and r.startswith("C"))
    others = sorted(r for r in sc.refs if r != anchor and r not in caps)

    # Place load caps symmetrically
    if len(caps) >= 2:
        w1, _ = _fp_wh(caps[0], fp_sizes)
        w2, _ = _fp_wh(caps[1], fp_sizes)
        positions[caps[0]] = (-(aw / 2.0 + _CRYSTAL_CAP_OFFSET), 0.0, 0.0)
        positions[caps[1]] = (aw / 2.0 + _CRYSTAL_CAP_OFFSET, 0.0, 0.0)
        extra = caps[2:]
    elif len(caps) == 1:
        positions[caps[0]] = (aw / 2.0 + _CRYSTAL_CAP_OFFSET, 0.0, 0.0)
        extra = []
    else:
        extra = []

    # Remaining components below — courtyard-aware vertical step
    below_refs = extra + others
    below_step = max((_fp_wh(r, fp_sizes)[1] for r in below_refs), default=1.0) + 1.0 if below_refs else 0.0
    y = 3.0
    for ref in below_refs:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (0.0, y + h / 2.0, 0.0)
        y += below_step

    return positions


def _layout_decoupling(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """Decoupling: caps in compact grid around IC edges.

    Anchor is the IC; caps placed in 2 columns alongside the IC body
    to minimize distance while staying compact.
    """
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    aw, ah = _fp_wh(anchor, fp_sizes)

    caps = sorted(r for r in sc.refs if r != anchor)
    n = len(caps)
    if n == 0:
        return positions

    # Place caps in 2 columns alongside the IC (left and right).
    # Compute courtyard-safe X offset: IC half-width + max cap half-width + gap
    left_caps = caps[: (n + 1) // 2]
    right_caps = caps[(n + 1) // 2:]
    max_cap_w = max((_fp_wh(r, fp_sizes)[0] for r in caps), default=1.0)
    max_cap_h = max((_fp_wh(r, fp_sizes)[1] for r in caps), default=1.0)
    # X offset: IC courtyard half + cap half + clearance gap (0.5mm) + margin (0.5mm)
    # The extra 0.5mm margin prevents rounding-induced AABB collisions
    x_offset = aw / 2.0 + max_cap_w / 2.0 + 1.0
    # Y step: cap courtyard height + 1mm gap (tight but real)
    cap_step = max_cap_h + 1.0

    # Left column — centered vertically
    total_h_left = (len(left_caps) - 1) * cap_step if left_caps else 0.0
    y = -total_h_left / 2.0
    for ref in left_caps:
        positions[ref] = (-x_offset, y, 0.0)
        y += cap_step

    # Right column — centered vertically
    total_h_right = (len(right_caps) - 1) * cap_step if right_caps else 0.0
    y = -total_h_right / 2.0
    for ref in right_caps:
        positions[ref] = (x_offset, y, 0.0)
        y += cap_step

    return positions


def _layout_voltage_divider(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """Voltage divider: linear chain R_top → midpoint → R_bot."""
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    _, ah = _fp_wh(anchor, fp_sizes)

    support = sorted(r for r in sc.refs if r != anchor)
    # Courtyard-aware vertical step from all component heights in the chain
    chain_step = max((_fp_wh(r, fp_sizes)[1] for r in support), default=1.0) + 1.0 if support else 0.0
    y = ah / 2.0 + _LINEAR_GAP
    for ref in support:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (0.0, y + h / 2.0, 0.0)
        y += chain_step

    return positions


def _layout_adc_channel(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """ADC channel: signal chain connector → protection → filter → divider → ADC.

    Linear left-to-right layout following signal flow.
    Anchor is the ADC IC; passives extend to the left toward the connector.
    """
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    aw, _ = _fp_wh(anchor, fp_sizes)

    support = [r for r in sc.refs if r != anchor]
    # Order: R (protection), C (filter), then others
    r_refs = sorted(r for r in support if r.startswith("R"))
    c_refs = sorted(r for r in support if r.startswith("C"))
    d_refs = sorted(r for r in support if r.startswith("D"))
    others = sorted(r for r in support
                    if r not in r_refs and r not in c_refs and r not in d_refs)

    # Place in signal chain order: protection R → filter C → diode → others
    chain = r_refs + c_refs + d_refs + others
    # Courtyard-aware horizontal step from all component widths in the chain
    chain_step = max((_fp_wh(r, fp_sizes)[0] for r in chain), default=1.0) + 1.0 if chain else 0.0
    x = -(aw / 2.0 + _LINEAR_GAP)
    for ref in chain:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (x - w / 2.0, 0.0, 0.0)
        x -= chain_step

    return positions


def _layout_rc_filter(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """RC/LC filter: linear R/L → C."""
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    aw, _ = _fp_wh(anchor, fp_sizes)

    support = sorted(r for r in sc.refs if r != anchor)
    # Courtyard-aware horizontal step from all component widths
    chain_step = max((_fp_wh(r, fp_sizes)[0] for r in support), default=1.0) + 1.0 if support else 0.0
    x = aw / 2.0 + _LINEAR_GAP
    for ref in support:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (x + w / 2.0, 0.0, 0.0)
        x += chain_step

    return positions


def _layout_esd_protection(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """ESD protection: IC near connector, series R inline."""
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    aw, ah = _fp_wh(anchor, fp_sizes)

    support = sorted(r for r in sc.refs if r != anchor)
    connectors = [r for r in support if r.startswith("J")]
    passives = [r for r in support if r not in connectors]

    # Connector to the left (toward board edge) — courtyard-aware step
    conn_step = max((_fp_wh(r, fp_sizes)[0] for r in connectors), default=1.0) + 1.0 if connectors else 0.0
    x = -(aw / 2.0 + _IC_PASSIVE_GAP)
    for ref in connectors:
        w, _ = _fp_wh(ref, fp_sizes)
        positions[ref] = (x - w / 2.0, 0.0, 0.0)
        x -= conn_step

    # Passives (R, C) to the right (toward IC they protect) — courtyard-aware step
    pass_step = max((_fp_wh(r, fp_sizes)[0] for r in passives), default=1.0) + 1.0 if passives else 0.0
    x = aw / 2.0 + _IC_PASSIVE_GAP
    for ref in passives:
        w, _ = _fp_wh(ref, fp_sizes)
        positions[ref] = (x + w / 2.0, 0.0, 0.0)
        x += pass_step

    return positions


def _layout_poe_filter(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """PoE filter: caps adjacent to connector POE pins."""
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    _, ah = _fp_wh(anchor, fp_sizes)

    caps = sorted(r for r in sc.refs if r != anchor)
    # Courtyard-aware horizontal step from all cap widths
    cap_step = max((_fp_wh(r, fp_sizes)[0] for r in caps), default=1.0) + 1.0 if caps else 0.0
    x = 0.0
    y = ah / 2.0 + _CAP_OFFSET
    for ref in caps:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (x, y + h / 2.0, 0.0)
        x += cap_step

    return positions


def _layout_optocoupler(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """Optocoupler: IC at center, passives arranged radially by type."""
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    aw, ah = _fp_wh(anchor, fp_sizes)

    support = sorted(r for r in sc.refs if r != anchor)
    r_refs = [r for r in support if r.startswith("R")]
    d_refs = [r for r in support if r.startswith("D")]
    others = [r for r in support if r not in r_refs and r not in d_refs]

    # Input side (left): R bias, LED/D — courtyard-aware step
    input_refs = r_refs[:1] + d_refs
    input_step = max((_fp_wh(r, fp_sizes)[0] for r in input_refs), default=1.0) + 1.0 if input_refs else 0.0
    x = -(aw / 2.0 + _IC_PASSIVE_GAP)
    for ref in input_refs:
        w, _ = _fp_wh(ref, fp_sizes)
        positions[ref] = (x - w / 2.0, 0.0, 0.0)
        x -= input_step

    # Output side (right): R pullup, others — courtyard-aware step
    output_refs = r_refs[1:] + others
    output_step = max((_fp_wh(r, fp_sizes)[0] for r in output_refs), default=1.0) + 1.0 if output_refs else 0.0
    x = aw / 2.0 + _IC_PASSIVE_GAP
    for ref in output_refs:
        w, _ = _fp_wh(ref, fp_sizes)
        positions[ref] = (x + w / 2.0, 0.0, 0.0)
        x += output_step

    return positions


def _layout_generic_ic_cluster(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """Generic IC cluster: radial placement of passives around IC.

    Places passives in a ring around the IC, spacing them evenly by
    angle.  Uses net connectivity to place connected components closer.
    """
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    aw, ah = _fp_wh(anchor, fp_sizes)

    support = sorted(r for r in sc.refs if r != anchor)
    n = len(support)
    if n == 0:
        return positions

    # Use tight placement: 2 columns alongside the IC.
    # Use actual courtyard heights for zero-overlap spacing.
    left = support[: (n + 1) // 2]
    right = support[(n + 1) // 2:]

    max_h_left = max((_fp_wh(r, fp_sizes)[1] for r in left), default=1.0)
    step_left = max_h_left + 1.5  # courtyard height + 1.5mm gap
    y = -(len(left) - 1) * step_left / 2.0
    for ref in left:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (-(aw / 2.0 + _IC_PASSIVE_GAP + w / 2.0), y, 0.0)
        y += step_left

    max_h_right = max((_fp_wh(r, fp_sizes)[1] for r in right), default=1.0)
    step_right = max_h_right + 1.5
    y = -(len(right) - 1) * step_right / 2.0
    for ref in right:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (aw / 2.0 + _IC_PASSIVE_GAP + w / 2.0, y, 0.0)
        y += step_right

    return positions


def _layout_rf_antenna(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """RF antenna: module at origin, matching components adjacent."""
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    aw, _ = _fp_wh(anchor, fp_sizes)

    support = sorted(r for r in sc.refs if r != anchor)
    # Courtyard-aware horizontal step from all component widths
    chain_step = max((_fp_wh(r, fp_sizes)[0] for r in support), default=1.0) + 1.0 if support else 0.0
    x = aw / 2.0 + _IC_PASSIVE_GAP
    for ref in support:
        w, _ = _fp_wh(ref, fp_sizes)
        positions[ref] = (x + w / 2.0, 0.0, 0.0)
        x += chain_step

    return positions


def _layout_mcu_peripheral_cluster(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """MCU peripheral cluster: radial placement around MCU.

    ICs and connectors in an outer ring, passives (pullups, decoupling)
    in an inner ring.
    """
    positions: dict[str, tuple[float, float, float]] = {}
    anchor = sc.anchor_ref
    positions[anchor] = (0.0, 0.0, 0.0)
    aw, ah = _fp_wh(anchor, fp_sizes)

    support = sorted(r for r in sc.refs if r != anchor)

    # Separate ICs/connectors from passives
    ics = [r for r in support if r.startswith(("U", "J"))]
    passives = [r for r in support if r not in ics]

    # Place ICs/connectors along the top edge of the MCU
    x = -(len(ics) - 1) * (_IC_PASSIVE_GAP + 3.0) / 2.0
    y_top = -(ah / 2.0 + _IC_PASSIVE_GAP)
    for ref in ics:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (x, y_top - h / 2.0, 0.0)
        x += w + _IC_PASSIVE_GAP

    # Place passives in 2 columns alongside (courtyard-aware spacing)
    left = passives[: (len(passives) + 1) // 2]
    right = passives[(len(passives) + 1) // 2:]

    max_h_left = max((_fp_wh(r, fp_sizes)[1] for r in left), default=1.0)
    step_left = max_h_left + 1.5
    y = -(len(left) - 1) * step_left / 2.0
    for ref in left:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (-(aw / 2.0 + _IC_PASSIVE_GAP + w / 2.0), y, 0.0)
        y += step_left

    max_h_right = max((_fp_wh(r, fp_sizes)[1] for r in right), default=1.0)
    step_right = max_h_right + 1.5
    for ref in right:
        w, h = _fp_wh(ref, fp_sizes)
        positions[ref] = (aw / 2.0 + _IC_PASSIVE_GAP + w / 2.0, y, 0.0)
        y += step_right

    return positions


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_LAYOUT_DISPATCH: dict[
    SubCircuitType,
    type[object],  # callable signature varies, use object for mypy
] = {}


def _get_layout_fn(
    sc_type: SubCircuitType,
) -> object:
    """Return the layout function for a subcircuit type."""
    return _LAYOUT_FUNCTIONS.get(sc_type, _layout_generic_ic_cluster)


# Type-safe dispatch mapping
_LAYOUT_FUNCTIONS: dict[SubCircuitType, object] = {
    SubCircuitType.RELAY_DRIVER: _layout_relay_driver,
    SubCircuitType.BUCK_CONVERTER: _layout_buck_converter,
    SubCircuitType.LDO_REGULATOR: _layout_ldo_regulator,
    SubCircuitType.CRYSTAL_OSC: _layout_crystal_osc,
    SubCircuitType.DECOUPLING: _layout_decoupling,
    SubCircuitType.RC_FILTER: _layout_rc_filter,
    SubCircuitType.VOLTAGE_DIVIDER: _layout_voltage_divider,
    SubCircuitType.MCU_PERIPHERAL_CLUSTER: _layout_mcu_peripheral_cluster,
    SubCircuitType.RF_ANTENNA: _layout_rf_antenna,
    SubCircuitType.ADC_CHANNEL: _layout_adc_channel,
    SubCircuitType.ESD_PROTECTION: _layout_esd_protection,
    SubCircuitType.POE_FILTER: _layout_poe_filter,
    SubCircuitType.OPTOCOUPLER_CIRCUIT: _layout_optocoupler,
    SubCircuitType.GENERIC_IC_CLUSTER: _layout_generic_ic_cluster,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def layout_subcircuit(
    sc: DetectedSubCircuit,
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> SubCircuitLayout:
    """Compute relative layout for a single subcircuit.

    Places all components relative to the anchor at (0, 0).  Returns
    a ``SubCircuitLayout`` with positions, convex hull polygon, and
    bounding dimensions.

    Args:
        sc: Detected subcircuit with refs and anchor.
        fp_sizes: Footprint sizes ``{ref: (w, h)}``.
        requirements: Project requirements for net lookups.

    Returns:
        Layout with relative positions and bounding polygon.
    """
    layout_fn = _get_layout_fn(sc.circuit_type)
    positions = layout_fn(sc, fp_sizes, requirements)  # type: ignore[operator]

    # Ensure anchor is at (0, 0) — templates should do this but enforce
    if sc.anchor_ref not in positions:
        positions[sc.anchor_ref] = (0.0, 0.0, 0.0)

    polygon = _build_polygon(positions, fp_sizes)
    width, height = _polygon_dims(polygon)

    # Derive signal flow order: sort refs by X position (left-to-right flow)
    flow_refs = sorted(
        positions.keys(),
        key=lambda r: positions[r][0],  # sort by X coordinate
    )

    return SubCircuitLayout(
        subcircuit=sc,
        positions=positions,
        polygon=polygon,
        anchor_ref=sc.anchor_ref,
        width=width,
        height=height,
        signal_flow=tuple(flow_refs),
    )


def layout_all_subcircuits(
    subcircuits: tuple[DetectedSubCircuit, ...] | list[DetectedSubCircuit],
    fp_sizes: dict[str, tuple[float, float]],
    requirements: ProjectRequirements,
) -> list[SubCircuitLayout]:
    """Compute layouts for all detected subcircuits.

    Each subcircuit is laid out independently with its anchor at (0, 0).
    This is Stage 1 of the bottom-up placement pipeline.

    Args:
        subcircuits: All detected subcircuits from functional_grouper.
        fp_sizes: Footprint sizes for all components.
        requirements: Project requirements for net lookups.

    Returns:
        List of subcircuit layouts, one per detected subcircuit.
    """
    layouts: list[SubCircuitLayout] = []

    for sc in subcircuits:
        layout = layout_subcircuit(sc, fp_sizes, requirements)
        _log.info(
            "  Stage 1: %s (%s) -- %d components, %.1fx%.1f mm",
            sc.circuit_type.value,
            sc.anchor_ref,
            len(layout.positions),
            layout.width,
            layout.height,
        )
        layouts.append(layout)

    _log.info(
        "Stage 1 complete: %d subcircuit layouts, %d total components",
        len(layouts),
        sum(len(lay.positions) for lay in layouts),
    )
    return layouts
