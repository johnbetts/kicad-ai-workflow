"""Signal flow chainer: trace signal paths through net topology.

Given a starting component pin, follows nets to find the next component
in the signal chain, building ordered lists of (ref, pin) pairs that
represent the physical signal path.  Used by the subnet placer to order
components within detected subcircuits for flow-based placement.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kicad_pipeline.pcb.constraints import _is_power_net

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.functional_grouper import DetectedSubCircuit

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Net topology helpers
# ---------------------------------------------------------------------------


def _build_pin_net_map(
    requirements: ProjectRequirements,
) -> dict[tuple[str, str], str]:
    """Build a mapping from (ref, pin) to net name.

    Only signal nets are included (power/ground are excluded).
    """
    pin_to_net: dict[tuple[str, str], str] = {}
    for net in requirements.nets:
        if _is_power_net(net.name):
            continue
        for conn in net.connections:
            pin_to_net[(conn.ref, conn.pin)] = net.name
    return pin_to_net


def _build_net_pin_map(
    requirements: ProjectRequirements,
) -> dict[str, list[tuple[str, str]]]:
    """Build a mapping from net name to list of (ref, pin) endpoints.

    Only signal nets are included (power/ground are excluded).
    """
    net_to_pins: dict[str, list[tuple[str, str]]] = {}
    for net in requirements.nets:
        if _is_power_net(net.name):
            continue
        pins: list[tuple[str, str]] = []
        for conn in net.connections:
            pins.append((conn.ref, conn.pin))
        if pins:
            net_to_pins[net.name] = pins
    return net_to_pins


def _is_ic_ref(ref: str) -> bool:
    """Return True if the ref designator indicates an IC (U*) or ADC."""
    prefix = ref.rstrip("0123456789")
    return prefix in {"U", "IC"}


def _is_passive_ref(ref: str) -> bool:
    """Return True if the ref designator indicates a passive component."""
    prefix = ref.rstrip("0123456789")
    return prefix in {"R", "C", "L", "FB"}


# ---------------------------------------------------------------------------
# Signal chain tracing
# ---------------------------------------------------------------------------


def trace_signal_chain(
    requirements: ProjectRequirements,
    start_ref: str,
    start_pin: str,
) -> list[tuple[str, str]]:
    """Trace a signal path through the net topology.

    Starting at ``(start_ref, start_pin)``, follows the net to find the
    next component, continuing until reaching an IC pin or a dead end
    (no further connections on the other pin of the current component).

    Args:
        requirements: Project requirements with components and nets.
        start_ref: Reference designator of the starting component.
        start_pin: Pin number of the starting component pin.

    Returns:
        Ordered list of (ref, pin) pairs representing the signal chain.
        The first entry is always ``(start_ref, start_pin)``.
    """
    pin_to_net = _build_pin_net_map(requirements)
    net_to_pins = _build_net_pin_map(requirements)

    # Build a map of component ref -> list of pin numbers (signal only)
    comp_pins: dict[str, list[str]] = {}
    for (ref, pin), _net in pin_to_net.items():
        comp_pins.setdefault(ref, []).append(pin)

    chain: list[tuple[str, str]] = [(start_ref, start_pin)]
    visited_nets: set[str] = set()
    current_ref = start_ref
    current_pin = start_pin

    for _step in range(100):  # safety limit
        # Find the net connected to current (ref, pin)
        net_name = pin_to_net.get((current_ref, current_pin))
        if net_name is None:
            break
        if net_name in visited_nets:
            break
        visited_nets.add(net_name)

        # Find the next component on this net (not ourselves)
        endpoints = net_to_pins.get(net_name, [])
        next_endpoint: tuple[str, str] | None = None
        for ep_ref, ep_pin in endpoints:
            if ep_ref == current_ref:
                continue
            next_endpoint = (ep_ref, ep_pin)
            break

        if next_endpoint is None:
            break

        next_ref, next_pin = next_endpoint
        chain.append((next_ref, next_pin))

        # If we reached an IC, stop
        if _is_ic_ref(next_ref):
            break

        # Find the "other" pin of this component to continue the chain
        other_pins = [p for p in comp_pins.get(next_ref, []) if p != next_pin]
        if not other_pins:
            break

        # Continue from the other pin of this component
        other_pin = other_pins[0]
        chain.append((next_ref, other_pin))
        current_ref = next_ref
        current_pin = other_pin

    return chain


# ---------------------------------------------------------------------------
# Subcircuit ordering by signal flow
# ---------------------------------------------------------------------------

# Known signal flow orders by subcircuit type.
# Each entry maps SubCircuitType.value -> list of (ref_prefix_role, ...)
# describing the expected order of components in the signal path.

_BUCK_CONVERTER_ROLES = ("C_in", "U_VIN", "U_SW", "L", "C_out")
_LDO_ROLES = ("C_in", "U_VIN", "U_VOUT", "C_out")
_VOLTAGE_DIVIDER_ROLES = ("R_top", "R_bot")
_ADC_CHANNEL_ROLES = ("J", "R_top", "R_bot", "C_filter", "U_ADC")
_RELAY_DRIVER_ROLES = ("R_gate", "Q", "D_flyback", "K", "J")


def _classify_role(
    ref: str,
    component_value: str,
    subcircuit_refs: frozenset[str],
) -> str:
    """Classify a component's role within a subcircuit.

    Returns a role string like "R", "C", "L", "U", "Q", "D", "K", "J".
    """
    prefix = ref.rstrip("0123456789")
    return prefix


def _sort_passives_by_net_order(
    refs: list[str],
    requirements: ProjectRequirements,
) -> list[str]:
    """Sort passive refs by their position in the signal chain.

    Uses net connectivity to determine which passive comes first
    in the signal path (closer to input vs closer to output).
    """
    if len(refs) <= 1:
        return refs

    pin_to_net = _build_pin_net_map(requirements)

    # Build adjacency: which refs are directly connected via signal nets
    ref_set = frozenset(refs)
    adj: dict[str, list[str]] = {r: [] for r in refs}
    net_to_refs: dict[str, set[str]] = {}

    for (r, _p), net_name in pin_to_net.items():
        if r in ref_set:
            net_to_refs.setdefault(net_name, set()).add(r)

    for net_refs in net_to_refs.values():
        net_refs_in_sc = net_refs & ref_set
        for r1 in net_refs_in_sc:
            for r2 in net_refs_in_sc:
                if r1 != r2 and r2 not in adj[r1]:
                    adj[r1].append(r2)

    # Find endpoint (ref with only 1 neighbour within the group) — it is
    # the start of the chain.
    endpoints = [r for r in refs if len(adj[r]) <= 1]
    if not endpoints:
        return refs  # cycle or fully connected — no clear order

    # Walk from first endpoint
    ordered: list[str] = []
    visited: set[str] = set()
    current = endpoints[0]
    while current and current not in visited:
        visited.add(current)
        ordered.append(current)
        neighbors = [n for n in adj.get(current, []) if n not in visited]
        current = neighbors[0] if neighbors else ""

    # Append any refs we missed (disconnected from chain)
    for r in refs:
        if r not in visited:
            ordered.append(r)

    return ordered


def order_subcircuit_by_flow(
    subcircuit: DetectedSubCircuit,
    requirements: ProjectRequirements,
) -> list[str]:
    """Order a subcircuit's component refs by signal flow.

    Uses the subcircuit type to determine expected component ordering:

    - **BUCK_CONVERTER**: C_in -> U (VIN) -> U (SW) -> L -> C_out,
      with FB divider as branch.
    - **LDO_REGULATOR**: C_in -> U (VIN) -> U (VOUT) -> C_out.
    - **VOLTAGE_DIVIDER**: R_top -> R_bot (series chain).
    - **ADC_CHANNEL**: connector -> R_top -> R_bot -> filter -> ADC pin.
    - **RELAY_DRIVER**: R_gate -> Q -> D_flyback -> K -> terminal.

    Falls back to net-connectivity-based ordering when type-specific
    heuristics cannot determine the order.

    Args:
        subcircuit: Detected subcircuit with component refs.
        requirements: Project requirements with nets.

    Returns:
        Ordered list of ref strings following the signal flow.
    """
    from kicad_pipeline.optimization.functional_grouper import SubCircuitType

    refs = list(subcircuit.refs)
    if len(refs) <= 1:
        return refs

    sc_type = subcircuit.circuit_type

    # Group refs by prefix
    by_prefix: dict[str, list[str]] = {}
    for ref in refs:
        prefix = ref.rstrip("0123456789")
        by_prefix.setdefault(prefix, []).append(ref)

    if sc_type == SubCircuitType.BUCK_CONVERTER:
        return _order_buck_converter(by_prefix, refs, requirements)
    if sc_type == SubCircuitType.LDO_REGULATOR:
        return _order_ldo_regulator(by_prefix, refs, requirements)
    if sc_type == SubCircuitType.VOLTAGE_DIVIDER:
        return _order_voltage_divider(by_prefix, refs, requirements)
    if sc_type == SubCircuitType.ADC_CHANNEL:
        return _order_adc_channel(by_prefix, refs, requirements)
    if sc_type == SubCircuitType.RELAY_DRIVER:
        return _order_relay_driver(by_prefix, refs, requirements)

    # Default: use net connectivity ordering
    return _sort_passives_by_net_order(refs, requirements)


def _order_buck_converter(
    by_prefix: dict[str, list[str]],
    all_refs: list[str],
    requirements: ProjectRequirements,
) -> list[str]:
    """Order buck converter: C_in -> U -> L -> C_out (+ FB branch)."""
    ordered: list[str] = []
    caps = sorted(by_prefix.get("C", []))
    ics = by_prefix.get("U", []) + by_prefix.get("IC", [])
    inductors = by_prefix.get("L", []) + by_prefix.get("FB", [])
    resistors = sorted(by_prefix.get("R", []))
    diodes = sorted(by_prefix.get("D", []))

    # C_in (first cap), then IC, then inductor, then C_out (remaining caps)
    if caps:
        ordered.append(caps[0])
    ordered.extend(sorted(ics))
    ordered.extend(sorted(inductors))
    if len(caps) > 1:
        ordered.extend(caps[1:])
    # Feedback divider resistors + bootstrap diode at end
    ordered.extend(diodes)
    ordered.extend(resistors)

    # Append any refs not yet included
    seen = set(ordered)
    for ref in all_refs:
        if ref not in seen:
            ordered.append(ref)

    return ordered


def _order_ldo_regulator(
    by_prefix: dict[str, list[str]],
    all_refs: list[str],
    requirements: ProjectRequirements,
) -> list[str]:
    """Order LDO: C_in -> U -> C_out."""
    ordered: list[str] = []
    caps = sorted(by_prefix.get("C", []))
    ics = by_prefix.get("U", []) + by_prefix.get("IC", [])

    if caps:
        ordered.append(caps[0])
    ordered.extend(sorted(ics))
    if len(caps) > 1:
        ordered.extend(caps[1:])

    seen = set(ordered)
    for ref in all_refs:
        if ref not in seen:
            ordered.append(ref)

    return ordered


def _order_voltage_divider(
    by_prefix: dict[str, list[str]],
    all_refs: list[str],
    requirements: ProjectRequirements,
) -> list[str]:
    """Order voltage divider: R_top -> R_bot (series chain)."""
    resistors = sorted(by_prefix.get("R", []))
    if len(resistors) >= 2:
        return _sort_passives_by_net_order(resistors, requirements)
    return resistors or list(all_refs)


def _order_adc_channel(
    by_prefix: dict[str, list[str]],
    all_refs: list[str],
    requirements: ProjectRequirements,
) -> list[str]:
    """Order ADC channel: J -> R_top -> R_bot -> C_filter -> U."""
    ordered: list[str] = []
    connectors = sorted(by_prefix.get("J", []))
    resistors = sorted(by_prefix.get("R", []))
    caps = sorted(by_prefix.get("C", []))
    ics = by_prefix.get("U", []) + by_prefix.get("IC", [])
    diodes = sorted(by_prefix.get("D", []))

    ordered.extend(connectors)
    # Sort resistors by net connectivity
    ordered.extend(_sort_passives_by_net_order(resistors, requirements))
    ordered.extend(caps)
    ordered.extend(diodes)
    ordered.extend(sorted(ics))

    seen = set(ordered)
    for ref in all_refs:
        if ref not in seen:
            ordered.append(ref)

    return ordered


def _order_relay_driver(
    by_prefix: dict[str, list[str]],
    all_refs: list[str],
    requirements: ProjectRequirements,
) -> list[str]:
    """Order relay driver: R_gate -> Q -> D_flyback -> K -> J."""
    ordered: list[str] = []
    resistors = sorted(by_prefix.get("R", []))
    transistors = sorted(by_prefix.get("Q", []))
    diodes = sorted(by_prefix.get("D", []))
    relays = sorted(by_prefix.get("K", []))
    connectors = sorted(by_prefix.get("J", []))

    ordered.extend(resistors)
    ordered.extend(transistors)
    ordered.extend(diodes)
    ordered.extend(relays)
    ordered.extend(connectors)

    seen = set(ordered)
    for ref in all_refs:
        if ref not in seen:
            ordered.append(ref)

    return ordered
