"""Subnet-to-pin resolver and pad-facing placement engine.

Resolves private subnet connections between IC pins and passive components,
then places passives so their connected pad faces the IC pin with minimal
trace length.

A "subnet" is a net with few connections (<=4) that links at least one IC
(U/Y prefix) to at least one passive (R/C/D/L prefix). These represent
dedicated connections like decoupling caps, bootstrap caps, feedback
resistors, and filter networks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.pcb.pin_map import (
    CardinalSide,
    compute_pin_map,
)

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.placement_types import PlacementContext

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

_IC_PREFIXES = frozenset({"U", "Y"})
_PASSIVE_PREFIXES = frozenset({"R", "C", "D", "L"})

# Maximum connections for a net to be considered a subnet
_MAX_SUBNET_CONNECTIONS = 4


@dataclass(frozen=True)
class SubnetConnection:
    """A passive component connected to a specific IC pin via a private subnet."""

    passive_ref: str  # e.g., "C1"
    passive_pin: str  # e.g., "1"
    ic_ref: str  # e.g., "U1"
    ic_pin: str  # e.g., "8"
    subnet_name: str  # e.g., "+3V3_U1_DEC"
    role: str  # e.g., "decoupling", "bootstrap", "feedback", "filter"


# ---------------------------------------------------------------------------
# Reference prefix helpers
# ---------------------------------------------------------------------------


def _ref_prefix(ref: str) -> str:
    """Extract the alphabetic prefix from a reference designator."""
    prefix = ""
    for ch in ref:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return prefix


def _is_ic_ref(ref: str) -> bool:
    """Return True if the ref designator belongs to an IC."""
    return _ref_prefix(ref) in _IC_PREFIXES


def _is_passive_ref(ref: str) -> bool:
    """Return True if the ref designator belongs to a passive component."""
    return _ref_prefix(ref) in _PASSIVE_PREFIXES


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

# Pin name substrings that map to passive roles.
# Order matters: first match wins.
_PIN_NAME_ROLE_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("VCC", "VDD", "AVCC", "AVDD", "DVCC", "DVDD", "VBUS"), "decoupling"),
    (("BST", "BOOT", "BOOTSTRAP"), "bootstrap"),
    (("FB", "FEEDBACK"), "feedback"),
    (("SW", "SWITCH", "LX"), "inductor"),
    (("OSC", "XTAL", "XIN", "XOUT", "XI", "XO"), "load_cap"),
    (("AIN", "ADC", "ANALOG", "AN"), "filter"),
)

_PIN_FUNCTION_ROLE_MAP: dict[str, str] = {
    "vcc": "decoupling",
    "boot": "bootstrap",
    "analog_in": "filter",
    "adc": "filter",
}


def _classify_role(ic_pin_name: str, ic_pin_function: str | None) -> str:
    """Classify the passive's role based on the IC pin it connects to.

    Args:
        ic_pin_name: The name of the IC pin (e.g., "VCC", "BST", "FB").
        ic_pin_function: The PinFunction value string, or None.

    Returns:
        A role string such as "decoupling", "bootstrap", "feedback", etc.
        Falls back to "associated" if no specific role is identified.
    """
    upper_name = ic_pin_name.upper()

    # Check pin name substrings
    for keywords, role in _PIN_NAME_ROLE_MAP:
        for kw in keywords:
            if kw in upper_name:
                return role

    # Check pin function enum value
    if ic_pin_function is not None:
        fn_role = _PIN_FUNCTION_ROLE_MAP.get(ic_pin_function)
        if fn_role is not None:
            return fn_role

    return "associated"


# ---------------------------------------------------------------------------
# Subnet resolution
# ---------------------------------------------------------------------------


def resolve_subnets(requirements: ProjectRequirements) -> list[SubnetConnection]:
    """Find all private subnet connections in the requirements.

    A subnet is a net with <= 4 connections that links at least one IC
    (U/Y prefix) to at least one passive (R/C/D/L prefix).

    For each such net, one SubnetConnection is created per (passive, IC)
    pair on that net.

    Args:
        requirements: The project requirements with nets and components.

    Returns:
        List of SubnetConnection instances.
    """
    # Build a lookup from ref -> Component for pin info
    comp_map = {c.ref: c for c in requirements.components}

    connections: list[SubnetConnection] = []

    for net in requirements.nets:
        # Skip nets with too many connections (shared rails like GND, +3V3)
        if len(net.connections) > _MAX_SUBNET_CONNECTIONS:
            continue
        if len(net.connections) < 2:
            continue

        # Partition connections into IC and passive groups
        ic_conns: list[tuple[str, str]] = []  # (ref, pin)
        passive_conns: list[tuple[str, str]] = []  # (ref, pin)

        for conn in net.connections:
            if _is_ic_ref(conn.ref):
                ic_conns.append((conn.ref, conn.pin))
            elif _is_passive_ref(conn.ref):
                passive_conns.append((conn.ref, conn.pin))

        # Must have at least one IC and one passive
        if not ic_conns or not passive_conns:
            continue

        # Create a SubnetConnection for each (passive, IC) pair
        for p_ref, p_pin in passive_conns:
            for ic_ref, ic_pin in ic_conns:
                # Look up the IC pin to classify the role
                ic_comp = comp_map.get(ic_ref)
                ic_pin_name = ""
                ic_pin_function: str | None = None
                if ic_comp is not None:
                    pin_obj = ic_comp.get_pin(ic_pin)
                    if pin_obj is not None:
                        ic_pin_name = pin_obj.name
                        if pin_obj.function is not None:
                            ic_pin_function = pin_obj.function.value

                role = _classify_role(ic_pin_name, ic_pin_function)

                connections.append(SubnetConnection(
                    passive_ref=p_ref,
                    passive_pin=p_pin,
                    ic_ref=ic_ref,
                    ic_pin=ic_pin,
                    subnet_name=net.name,
                    role=role,
                ))

    _log.debug("Resolved %d subnet connections", len(connections))
    return connections


# ---------------------------------------------------------------------------
# IC pin position resolution
# ---------------------------------------------------------------------------


def resolve_ic_pin_position(
    ic_ref: str,
    ic_pin: str,
    pcb: PCBDesign,
) -> tuple[float, float, str] | None:
    """Resolve the board-space position and side of an IC pin.

    Args:
        ic_ref: The IC reference designator (e.g., "U1").
        ic_pin: The pin number (e.g., "8").
        pcb: The PCB design with placed footprints.

    Returns:
        (x, y, side) where x/y are board coordinates and side is
        "N", "S", "E", or "W". Returns None if the IC or pin is not found.
    """
    fp = pcb.get_footprint(ic_ref)
    if fp is None:
        _log.warning("IC %s not found in PCB", ic_ref)
        return None

    # Find the pad matching the pin number
    target_pad = None
    for pad in fp.pads:
        if pad.number == ic_pin:
            target_pad = pad
            break

    if target_pad is None:
        _log.warning("Pin %s not found on %s", ic_pin, ic_ref)
        return None

    # Compute board-space position accounting for footprint rotation
    rot_rad = math.radians(-fp.rotation)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    local_x, local_y = target_pad.position.x, target_pad.position.y
    board_x = fp.position.x + local_x * cos_r - local_y * sin_r
    board_y = fp.position.y + local_x * sin_r + local_y * cos_r

    # Determine which side of the IC body this pad is on
    pin_map = compute_pin_map(fp, fp.rotation)
    side_enum = pin_map.side_for_pad(ic_pin)
    if side_enum is None or side_enum == CardinalSide.CENTER:
        side_str = "S"  # default fallback for center/thermal pads
    else:
        side_str = {
            CardinalSide.NORTH: "N",
            CardinalSide.SOUTH: "S",
            CardinalSide.EAST: "E",
            CardinalSide.WEST: "W",
        }[side_enum]

    return board_x, board_y, side_str


# ---------------------------------------------------------------------------
# Pad-facing position computation
# ---------------------------------------------------------------------------


def compute_pad_facing_position(
    passive_size: tuple[float, float],
    ic_pin_x: float,
    ic_pin_y: float,
    ic_pin_side: str,
    gap_mm: float = 1.0,
) -> tuple[float, float, float]:
    """Compute position and rotation to place a passive facing an IC pin.

    The passive is placed so its pad 1 faces the IC pin with a small gap.

    Args:
        passive_size: (width, height) of the passive footprint in mm.
        ic_pin_x: Board X coordinate of the IC pin.
        ic_pin_y: Board Y coordinate of the IC pin.
        ic_pin_side: Which side of the IC the pin is on ("N", "S", "E", "W").
        gap_mm: Gap between the IC pin and the passive body edge.

    Returns:
        (x, y, rotation) for the passive footprint placement.
    """
    pw, ph = passive_size

    if ic_pin_side == "W":
        # Pin on left of IC -> passive goes further left, pad 1 faces right
        return (ic_pin_x - pw / 2 - gap_mm, ic_pin_y, 0.0)
    elif ic_pin_side == "E":
        # Pin on right of IC -> passive goes further right, pad 1 faces left
        return (ic_pin_x + pw / 2 + gap_mm, ic_pin_y, 180.0)
    elif ic_pin_side == "N":
        # Pin on top of IC -> passive goes above, pad 1 faces down
        return (ic_pin_x, ic_pin_y - ph / 2 - gap_mm, 90.0)
    else:  # "S"
        # Pin on bottom of IC -> passive goes below, pad 1 faces up
        return (ic_pin_x, ic_pin_y + ph / 2 + gap_mm, 270.0)


# ---------------------------------------------------------------------------
# Placement integration
# ---------------------------------------------------------------------------


def place_subnet_components(
    ctx: PlacementContext,
    connections: list[SubnetConnection],
) -> set[str]:
    """Place passive components so their pads face the connected IC pins.

    For each SubnetConnection, resolves the IC pin's board position, computes
    the pad-facing position for the passive, and updates ``ctx.positions``.

    Args:
        ctx: The mutable placement context with positions and PCB data.
        connections: Subnet connections from :func:`resolve_subnets`.

    Returns:
        Set of passive refs that were successfully placed (and should be
        marked as fixed to prevent later phases from moving them).
    """
    placed_refs: set[str] = set()

    for conn in connections:
        # Skip if passive is already fixed
        if conn.passive_ref in ctx.fixed_refs:
            continue

        # Resolve the IC pin position on the board
        pin_info = resolve_ic_pin_position(
            conn.ic_ref, conn.ic_pin, ctx.initial_pcb,
        )
        if pin_info is None:
            _log.debug(
                "Skipping %s: IC pin %s.%s not resolved",
                conn.passive_ref, conn.ic_ref, conn.ic_pin,
            )
            continue

        ic_pin_x, ic_pin_y, ic_pin_side = pin_info

        # Get the passive footprint size
        fp_size = ctx.fp_sizes.get(conn.passive_ref)
        if fp_size is None:
            _log.debug("Skipping %s: no footprint size", conn.passive_ref)
            continue

        # Compute the pad-facing position
        x, y, rotation = compute_pad_facing_position(
            passive_size=fp_size,
            ic_pin_x=ic_pin_x,
            ic_pin_y=ic_pin_y,
            ic_pin_side=ic_pin_side,
            gap_mm=1.0,
        )

        # Clamp to board bounds
        bx_min, by_min, bx_max, by_max = ctx.bounds
        x = max(bx_min + fp_size[0] / 2, min(bx_max - fp_size[0] / 2, x))
        y = max(by_min + fp_size[1] / 2, min(by_max - fp_size[1] / 2, y))

        ctx.positions[conn.passive_ref] = (x, y, rotation)
        placed_refs.add(conn.passive_ref)
        _log.debug(
            "Placed %s at (%.1f, %.1f, %.0f) facing %s pin %s.%s [%s]",
            conn.passive_ref, x, y, rotation,
            conn.ic_ref, conn.ic_pin, ic_pin_side, conn.role,
        )

    # Mark placed refs as fixed so later phases don't move them
    ctx.fixed_refs.update(placed_refs)
    _log.info(
        "Subnet placement: placed %d/%d passives",
        len(placed_refs), len(connections),
    )
    return placed_refs
