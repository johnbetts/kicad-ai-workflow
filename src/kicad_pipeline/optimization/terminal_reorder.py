"""Screw terminal pin reordering for trace crossing minimization.

Provides utilities to reorder pin-to-net assignments on screw terminal
connectors so that external connections minimize board trace crossings.
This is a late-stage optimization — pin assignments aren't fixed until
routing begins.

Usage::

    # Auto-optimize all terminal blocks
    reorder_terminals_for_routing(pcb, requirements)

    # Manual reorder via natural language
    apply_terminal_reorder(pcb, "swap J1 pins 1 and 3")
    apply_terminal_reorder(pcb, "reorder J1 to match left-to-right signal flow")
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

_log = logging.getLogger(__name__)


def _count_crossings_for_terminal(
    terminal_ref: str,
    pin_to_net: dict[int, str],
    net_targets: dict[str, tuple[float, float]],
    terminal_pos: tuple[float, float],
    terminal_pitch: float,
    n_pins: int,
) -> int:
    """Count ratsnest crossings for a given pin-to-net assignment.

    A crossing occurs when two traces from adjacent pins cross each other.
    This happens when pin i connects to a target that's to the right of
    pin j's target, but pin i is to the left of pin j.
    """
    # Build pin positions (left to right along terminal)
    pin_positions: list[tuple[float, float]] = []
    tx, ty = terminal_pos
    for i in range(n_pins):
        px = tx - (n_pins - 1) * terminal_pitch / 2.0 + i * terminal_pitch
        pin_positions.append((px, ty))

    # Build target positions for each pin
    targets: list[tuple[float, float] | None] = []
    for pin_num in range(1, n_pins + 1):
        net = pin_to_net.get(pin_num)
        if net and net in net_targets:
            targets.append(net_targets[net])
        else:
            targets.append(None)

    # Count crossings (sweep line)
    crossings = 0
    for i in range(n_pins):
        for j in range(i + 1, n_pins):
            if targets[i] is None or targets[j] is None:
                continue
            # Pin i is left of pin j. If target i is to the right of target j,
            # the traces cross.
            if targets[i][0] > targets[j][0] + 1.0:  # 1mm tolerance
                crossings += 1

    return crossings


def optimize_terminal_pin_order(
    terminal_ref: str,
    current_pin_to_net: dict[int, str],
    net_targets: dict[str, tuple[float, float]],
    terminal_pos: tuple[float, float],
    terminal_pitch: float = 5.08,
) -> dict[int, str] | None:
    """Find the pin order that minimizes trace crossings.

    Tries all permutations of pin-to-net assignments and returns the
    one with fewest crossings. Returns None if current order is already
    optimal or if there's only one pin.

    Args:
        terminal_ref: Reference designator (e.g., "J1").
        current_pin_to_net: Current pin number → net name mapping.
        net_targets: Net name → (x, y) of the nearest connected component.
        terminal_pos: (x, y) position of the terminal block center.
        terminal_pitch: Pin-to-pin spacing in mm (default 5.08mm).

    Returns:
        Optimized pin_to_net mapping, or None if no improvement.
    """
    n_pins = len(current_pin_to_net)
    if n_pins <= 1:
        return None

    nets = list(current_pin_to_net.values())
    pin_nums = list(current_pin_to_net.keys())

    current_crossings = _count_crossings_for_terminal(
        terminal_ref, current_pin_to_net, net_targets,
        terminal_pos, terminal_pitch, n_pins,
    )

    best_crossings = current_crossings
    best_order: list[str] | None = None

    # Try all permutations (feasible for ≤8 pins)
    if n_pins <= 8:
        for perm in itertools.permutations(nets):
            candidate = dict(zip(pin_nums, perm))
            crossings = _count_crossings_for_terminal(
                terminal_ref, candidate, net_targets,
                terminal_pos, terminal_pitch, n_pins,
            )
            if crossings < best_crossings:
                best_crossings = crossings
                best_order = list(perm)

    if best_order is None:
        return None

    _log.info(
        "Terminal %s: reordered %d pins, crossings %d → %d",
        terminal_ref, n_pins, current_crossings, best_crossings,
    )
    return dict(zip(pin_nums, best_order))


def reorder_terminals_for_routing(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> dict[str, dict[int, str]]:
    """Auto-optimize all screw terminal pin orders.

    Returns a mapping of terminal_ref → optimized pin_to_net for each
    terminal that was improved. Empty dict if no improvements found.
    """
    from kicad_pipeline.pcb.pin_map import origin_to_centroid

    # Build position lookup
    pos: dict[str, tuple[float, float]] = {}
    for fp in pcb.footprints:
        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        pos[fp.ref] = (cx, cy)

    # Find terminal blocks (J* with "Terminal" in footprint or >2 pins at 5.08mm pitch)
    terminals: list[str] = []
    for fp in pcb.footprints:
        if not fp.ref.startswith("J"):
            continue
        if "terminal" in fp.footprint_id.lower() or "5.08" in fp.footprint_id:
            terminals.append(fp.ref)

    if not terminals:
        return {}

    # Build net → connected component positions
    net_component_pos: dict[str, list[tuple[float, float]]] = {}
    for net in requirements.nets:
        for conn in net.connections:
            if conn.ref in pos and not conn.ref.startswith("J"):
                net_component_pos.setdefault(net.name, []).append(pos[conn.ref])

    # For each net, compute the centroid of non-connector connected components
    net_targets: dict[str, tuple[float, float]] = {}
    for net_name, positions_list in net_component_pos.items():
        if positions_list:
            cx = sum(p[0] for p in positions_list) / len(positions_list)
            cy = sum(p[1] for p in positions_list) / len(positions_list)
            net_targets[net_name] = (cx, cy)

    # Build pin-to-net for each terminal
    results: dict[str, dict[int, str]] = {}
    for t_ref in terminals:
        pin_to_net: dict[int, str] = {}
        for net in requirements.nets:
            for conn in net.connections:
                if conn.ref == t_ref:
                    try:
                        pin_num = int(conn.pin)
                    except (ValueError, TypeError):
                        continue
                    pin_to_net[pin_num] = net.name

        if len(pin_to_net) < 2:
            continue

        t_pos = pos.get(t_ref, (0.0, 0.0))
        optimized = optimize_terminal_pin_order(
            t_ref, pin_to_net, net_targets, t_pos,
        )
        if optimized is not None:
            results[t_ref] = optimized

    if results:
        _log.info("Terminal reorder: %d terminals optimized", len(results))

    return results
