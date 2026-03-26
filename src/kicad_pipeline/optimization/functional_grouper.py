"""Functional grouper: sub-circuit detection and voltage domain classification.

Analyses netlist topology to detect common sub-circuits (relay drivers, buck
converters, decoupling pairs, etc.) and classifies components by voltage
domain. This provides the foundation for EE-grade deterministic placement.
"""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.pcb.constraints import (
    _is_decoupling_cap,
    _is_power_net,
    build_signal_adjacency,
)

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import Component, ProjectRequirements

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SubCircuitType(enum.Enum):
    """Types of detected sub-circuits."""

    RELAY_DRIVER = "relay_driver"
    BUCK_CONVERTER = "buck_converter"
    LDO_REGULATOR = "ldo_regulator"
    CRYSTAL_OSC = "crystal_osc"
    DECOUPLING = "decoupling"
    RC_FILTER = "rc_filter"
    VOLTAGE_DIVIDER = "voltage_divider"
    MCU_PERIPHERAL_CLUSTER = "mcu_peripheral_cluster"
    RF_ANTENNA = "rf_antenna"
    ADC_CHANNEL = "adc_channel"


class VoltageDomain(enum.Enum):
    """Voltage domains for zone assignment."""

    VIN_24V = "24v"
    POWER_5V = "5v"
    DIGITAL_3V3 = "3v3"
    ANALOG = "analog"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubCircuitNode:
    """Hierarchical node within a sub-circuit.

    Describes a functional role (e.g. relay body, driver, LED indicator)
    with its component refs and optional child nodes for hierarchical
    placement.
    """

    role: str  # "relay_body", "driver", "led_indicator", "input", "output"
    refs: tuple[str, ...]
    anchor_ref: str
    children: tuple[SubCircuitNode, ...] = ()


@dataclass(frozen=True)
class DetectedSubCircuit:
    """A detected sub-circuit with component refs and domain."""

    circuit_type: SubCircuitType
    refs: tuple[str, ...]
    anchor_ref: str
    net_connections: tuple[str, ...]
    domain: VoltageDomain
    layout_hint: str = "cluster"  # "cluster" | "row" | "linear" | "edge" | "boundary"
    hierarchy: SubCircuitNode | None = None
    input_domain: VoltageDomain | None = None   # for regulators
    output_domain: VoltageDomain | None = None  # for regulators


@dataclass(frozen=True)
class DomainAffinity:
    """Cross-domain component affinity for placement co-location.

    Identifies components in different voltage domains that should be
    placed near each other (e.g. analog monitoring circuits measuring
    relay outputs).
    """

    source_refs: tuple[str, ...]
    target_refs: tuple[str, ...]
    source_domain: VoltageDomain
    target_domain: VoltageDomain
    reason: str  # "measurement", "feedback", "control"


@dataclass(frozen=True)
class BoardZoneAssignment:
    """A voltage-domain zone with assigned sub-circuits and loose components."""

    domain: VoltageDomain
    zone_rect: tuple[float, float, float, float]  # x1, y1, x2, y2
    subcircuits: tuple[DetectedSubCircuit, ...]
    loose_refs: tuple[str, ...]


@dataclass(frozen=True)
class PowerFlowTopology:
    """Power flow topology derived from regulator input/output domains.

    Describes the ordering of voltage domains following the power
    distribution path (highest voltage input to lowest voltage output)
    and the regulator boundaries between adjacent domains.
    """

    domain_order: tuple[VoltageDomain, ...]
    regulator_boundaries: tuple[tuple[VoltageDomain, VoltageDomain, str], ...]
    """Each entry: (input_domain, output_domain, regulator_anchor_ref)."""


# ---------------------------------------------------------------------------
# Net helpers
# ---------------------------------------------------------------------------


_GND_NAMES: frozenset[str] = frozenset(
    {"GND", "AGND", "DGND", "PGND", "VGND", "VSS", "VEE"}
)

_ANALOG_KEYWORDS: frozenset[str] = frozenset(
    {"ADC", "AIN", "AOUT", "DAC", "VREF", "ANALOG"}
)

_REGULATOR_KEYWORDS: frozenset[str] = frozenset({
    "LDO", "BUCK", "BOOST", "AMS1117", "LP5907", "LP5912", "TPS54",
    "TPS56", "AP2112", "MCP1700", "NCV8114", "RT9013", "ME6211",
    "HT7333", "HT7533", "SGM2019", "XC6206", "AP7361",
    "REGULATOR", "CONVERTER", "SWITCHING",
})


def _is_gnd_net(name: str) -> bool:
    """Return True if *name* is a ground net."""
    return name.upper().strip() in _GND_NAMES


def _parse_voltage_from_net(net_name: str) -> float | None:
    """Extract voltage from a power-net name.

    Returns the voltage in volts, or None if not parseable.
    Examples: "+24V" → 24.0, "+3V3" → 3.3, "+5V" → 5.0
    """
    name = net_name.upper().strip()
    if name in _GND_NAMES:
        return 0.0
    # +24V, +5V, +12V
    m = re.search(r"(\d+)\s*V(?!\w*\d)", name)
    if m:
        return float(m.group(1))
    # +3V3 style
    m = re.search(r"(\d+)V(\d+)", name)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    return None


def _classify_voltage(voltage: float | None) -> VoltageDomain:
    """Map a voltage value to a domain."""
    if voltage is None:
        return VoltageDomain.MIXED
    if voltage >= 20.0:
        return VoltageDomain.VIN_24V
    if voltage >= 4.0:
        return VoltageDomain.POWER_5V
    if voltage > 0.0:
        return VoltageDomain.DIGITAL_3V3
    return VoltageDomain.MIXED  # GND (0V) is mixed


def _ref_nets(
    requirements: ProjectRequirements,
) -> dict[str, set[str]]:
    """Map each component ref to the set of nets it connects to."""
    ref_to_nets: dict[str, set[str]] = {}
    for net in requirements.nets:
        for conn in net.connections:
            ref_to_nets.setdefault(conn.ref, set()).add(net.name)
    return ref_to_nets


def _net_refs(
    requirements: ProjectRequirements,
) -> dict[str, set[str]]:
    """Map each net name to the set of component refs on it."""
    result: dict[str, set[str]] = {}
    for net in requirements.nets:
        refs = {conn.ref for conn in net.connections}
        result[net.name] = refs
    return result


def _ref_prefix(ref: str) -> str:
    """Get the alpha prefix of a reference designator."""
    return "".join(c for c in ref if c.isalpha())


# ---------------------------------------------------------------------------
# Sub-circuit detection
# ---------------------------------------------------------------------------


def _is_tvs_or_led_diode(comp: Component | None) -> bool:
    """Return True if *comp* looks like a TVS diode or LED (not a flyback diode)."""
    if comp is None:
        return False
    dv = (comp.value or "").upper()
    dd = (comp.description or "").upper()
    return "TVS" in dv or "TVS" in dd or "LED" in dv


def _find_flyback_diode(
    relay_nets: set[str],
    net_to_refs: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    comp_map: dict[str, Component],
    claimed: set[str],
    refs: list[str],
) -> tuple[str | None, str | None]:
    """Find the best flyback diode candidate for a relay.

    Returns (diode_ref, net_name) or (None, None).
    """
    candidates: list[tuple[int, str, str]] = []
    for net_name in sorted(relay_nets):
        if _is_gnd_net(net_name):
            continue
        for r in sorted(net_to_refs.get(net_name, set())):
            if _ref_prefix(r) != "D" or r in claimed or r in refs:
                continue
            if _is_tvs_or_led_diode(comp_map.get(r)):
                continue
            d_nets = ref_to_nets.get(r, set())
            shared = len(d_nets & relay_nets)
            candidates.append((shared, r, net_name))
    if not candidates:
        return None, None
    candidates.sort(key=lambda t: (-t[0], t[1]))
    return candidates[0][1], candidates[0][2]


def _find_collector_led(
    transistor: str,
    ref_to_nets: dict[str, set[str]],
    net_to_refs: dict[str, set[str]],
    comp_map: dict[str, Component],
    claimed: set[str],
    refs: list[str],
) -> str | None:
    """Find an LED on the transistor collector path."""
    t_nets = ref_to_nets.get(transistor, set())
    for net_name in t_nets:
        if _is_gnd_net(net_name) or _is_power_net(net_name):
            continue
        for r in net_to_refs.get(net_name, set()):
            if r in refs:
                continue
            if _ref_prefix(r) == "D" and r not in claimed:
                comp = comp_map.get(r)
                if comp and "LED" in (comp.value or "").upper():
                    return r
            elif _ref_prefix(r) == "LED" and r not in claimed:
                return r
    return None


def _classify_relay_hierarchy_refs(
    refs: list[str],
    relay_ref: str,
    transistor: str | None,
    comp_map: dict[str, Component],
    adj: dict[str, set[str]],
) -> tuple[list[str], list[str], str | None, str | None]:
    """Classify relay subcircuit refs into driver and LED subgroup lists.

    Returns (driver_refs, led_node_refs, led_ref, flyback_ref).
    """
    driver_refs: list[str] = []
    led_node_refs: list[str] = []
    flyback_ref: str | None = None
    gate_resistor_ref: str | None = None
    led_ref: str | None = None
    led_resistor_ref: str | None = None

    for r in refs:
        if r == relay_ref:
            continue
        prefix = _ref_prefix(r)
        if prefix == "Q":
            driver_refs.append(r)
        elif prefix == "D":
            comp = comp_map.get(r)
            if comp and "LED" in (comp.value or "").upper():
                led_ref = r
            else:
                flyback_ref = r
        elif prefix == "LED":
            led_ref = r
        elif prefix == "R":
            if transistor and r in adj.get(transistor, set()):
                gate_resistor_ref = r
            else:
                led_resistor_ref = r

    if flyback_ref:
        driver_refs.append(flyback_ref)
    if gate_resistor_ref:
        driver_refs.append(gate_resistor_ref)
    if led_ref:
        led_node_refs.append(led_ref)
    if led_resistor_ref:
        led_node_refs.append(led_resistor_ref)

    return driver_refs, led_node_refs, led_ref, flyback_ref


def _build_relay_hierarchy(
    relay_ref: str,
    transistor: str | None,
    driver_refs: list[str],
    led_node_refs: list[str],
    led_ref: str | None,
) -> SubCircuitNode:
    """Build hierarchical SubCircuitNode tree for a relay driver."""
    children: list[SubCircuitNode] = []
    if driver_refs:
        children.append(SubCircuitNode(
            role="driver",
            refs=tuple(sorted(driver_refs)),
            anchor_ref=transistor or driver_refs[0],
        ))
    if led_node_refs:
        children.append(SubCircuitNode(
            role="led_indicator",
            refs=tuple(sorted(led_node_refs)),
            anchor_ref=led_ref or led_node_refs[0],
        ))
    return SubCircuitNode(
        role="relay_body",
        refs=(relay_ref,),
        anchor_ref=relay_ref,
        children=tuple(children),
    )


def _find_relay_support_components(
    relay_ref: str,
    adj: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    net_to_refs: dict[str, set[str]],
    comp_map: dict[str, object],
    claimed: set[str],
) -> tuple[list[str], str | None]:
    """Find transistor, flyback diode, gate resistor, LED, and LED resistor.

    Returns (refs_list, transistor_ref).
    """
    refs: list[str] = [relay_ref]
    relay_nets = ref_to_nets.get(relay_ref, set())

    # Transistor
    transistor: str | None = None
    for nb in adj.get(relay_ref, set()):
        if _ref_prefix(nb) == "Q" and nb not in claimed:
            transistor = nb
            refs.append(nb)
            break

    # Flyback diode
    best_d, _best_net = _find_flyback_diode(
        relay_nets, net_to_refs, ref_to_nets, comp_map, claimed, refs,
    )
    if best_d:
        refs.append(best_d)

    # Gate resistor
    if transistor:
        for nb in adj.get(transistor, set()):
            if _ref_prefix(nb) == "R" and nb not in claimed and nb not in refs:
                refs.append(nb)
                break

    # LED on collector path
    if transistor:
        led = _find_collector_led(
            transistor, ref_to_nets, net_to_refs, comp_map, claimed, refs,
        )
        if led:
            refs.append(led)

    # LED current-limiting resistor
    for lr in refs[:]:
        if _ref_prefix(lr) not in ("LED", "D"):
            continue
        for nb in adj.get(lr, set()):
            if _ref_prefix(nb) == "R" and nb not in claimed and nb not in refs:
                refs.append(nb)
                break

    return refs, transistor


def _relay_domain(relay_nets: set[str]) -> VoltageDomain:
    """Determine voltage domain from relay nets."""
    for net_name in relay_nets:
        v = _parse_voltage_from_net(net_name)
        if v is not None and v > 0:
            return _classify_voltage(v)
    return VoltageDomain.MIXED


def _detect_relay_drivers(
    requirements: ProjectRequirements,
    net_to_refs: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    adj: dict[str, set[str]],
) -> list[DetectedSubCircuit]:
    """Detect relay driver sub-circuits: K + Q + D (flyback) + R (gate) + LED."""
    results: list[DetectedSubCircuit] = []
    comp_map = {c.ref: c for c in requirements.components}
    claimed: set[str] = set()

    relays = [c for c in requirements.components if _ref_prefix(c.ref) == "K"]

    for relay in relays:
        if relay.ref in claimed:
            continue

        refs, transistor = _find_relay_support_components(
            relay.ref, adj, ref_to_nets, net_to_refs, comp_map, claimed,
        )

        relay_nets = ref_to_nets.get(relay.ref, set())
        all_nets: set[str] = set(relay_nets)
        best_d, best_net = _find_flyback_diode(
            relay_nets, net_to_refs, ref_to_nets, comp_map, claimed, refs,
        )
        if best_net:
            all_nets.add(best_net)

        domain = _relay_domain(relay_nets)

        for r in refs:
            claimed.add(r)

        driver_refs, led_node_refs, led_ref, _ = _classify_relay_hierarchy_refs(
            refs, relay.ref, transistor, comp_map, adj,
        )
        hierarchy = _build_relay_hierarchy(
            relay.ref, transistor, driver_refs, led_node_refs, led_ref,
        )

        results.append(DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=tuple(sorted(refs)),
            anchor_ref=relay.ref,
            net_connections=tuple(sorted(all_nets)),
            domain=domain,
            layout_hint="row",
            hierarchy=hierarchy,
        ))

    return results


def _build_ref_group_index(
    requirements: ProjectRequirements,
) -> dict[str, str]:
    """Build ref -> FeatureBlock name index for same-group checks."""
    index: dict[str, str] = {}
    for feat in requirements.features:
        for fc in feat.components:
            r = fc.ref if hasattr(fc, "ref") else fc
            index[r] = feat.name
    return index


_VIN_KEYWORDS = ("VIN", "IN")
_VOUT_KEYWORDS = ("VOUT", "OUT")
_V_TO_V_PATTERN = re.compile(
    r"(\d+)(?:\s*[-/]\s*(\d+))?\s*V\s+TO\s+(\d+(?:\.\d+)?)\s*V",
)


def _domains_from_pin_names(
    comp: Component,
) -> tuple[VoltageDomain | None, VoltageDomain | None]:
    """Infer input/output domains from VIN/VOUT pin net voltages."""
    input_d: VoltageDomain | None = None
    output_d: VoltageDomain | None = None
    for pin in comp.pins:
        if not pin.net:
            continue
        pname = (pin.name or "").upper()
        v = _parse_voltage_from_net(pin.net)
        if v is None or v <= 0:
            continue
        vd = _classify_voltage(v)
        if any(kw in pname for kw in _VIN_KEYWORDS):
            input_d = vd
        elif any(kw in pname for kw in _VOUT_KEYWORDS):
            output_d = vd
    return input_d, output_d


def _output_domain_from_fb_pin(comp: Component) -> VoltageDomain | None:
    """Infer output domain from a feedback (FB) pin's net voltage."""
    for pin in comp.pins:
        if not pin.net:
            continue
        pname = (pin.name or "").upper()
        if "FB" in pname:
            v = _parse_voltage_from_net(pin.net)
            if v is not None and v > 0:
                return _classify_voltage(v)
    return None


def _domains_from_description(
    desc: str,
    input_domain: VoltageDomain | None,
    output_domain: VoltageDomain | None,
) -> tuple[VoltageDomain | None, VoltageDomain | None]:
    """Infer input/output domains from description text like '8-32V to 5V'."""
    m = _V_TO_V_PATTERN.search(desc.upper())
    if not m:
        return input_domain, output_domain
    vin_v = float(m.group(2) or m.group(1))
    vout_v = float(m.group(3))
    if input_domain is None and vin_v > 0:
        input_domain = _classify_voltage(vin_v)
    if output_domain is None and vout_v > 0:
        output_domain = _classify_voltage(vout_v)
    return input_domain, output_domain


def _input_domain_from_highest_net(
    comp_nets: set[str],
    exclude_net: str | None,
) -> VoltageDomain | None:
    """Fallback: infer input domain from the highest-voltage net."""
    exclude = {exclude_net} if exclude_net else set()
    best: VoltageDomain | None = None
    for net_name in comp_nets - exclude:
        v = _parse_voltage_from_net(net_name)
        if v is not None and v > 0:
            d = _classify_voltage(v)
            if best is None or d == VoltageDomain.VIN_24V:
                best = d
    return best


def _classify_regulator_domains(
    comp: Component,
    comp_nets: set[str],
    inductor_output_net: str | None,
) -> tuple[VoltageDomain | None, VoltageDomain | None, VoltageDomain]:
    """Classify input/output voltage domains for a regulator IC.

    Tries pin names, inductor output net, FB pin net, description text,
    and falls back to highest-voltage net.

    Returns (input_domain, output_domain, primary_domain).
    """
    domain = VoltageDomain.POWER_5V

    # Strategy 1: pin names
    input_domain, output_domain = _domains_from_pin_names(comp)
    if output_domain is not None:
        domain = output_domain

    # Strategy 2: inductor output net
    if output_domain is None and inductor_output_net:
        v = _parse_voltage_from_net(inductor_output_net)
        if v is not None and v > 0:
            output_domain = _classify_voltage(v)
            domain = output_domain

    # Strategy 3: FB pin net
    if output_domain is None:
        fb_d = _output_domain_from_fb_pin(comp)
        if fb_d is not None:
            output_domain = fb_d
            domain = output_domain

    # Strategy 4: description text
    if input_domain is None or output_domain is None:
        input_domain, output_domain = _domains_from_description(
            comp.description or "", input_domain, output_domain,
        )
        if output_domain is not None:
            domain = output_domain

    # Strategy 5: highest-voltage net
    if input_domain is None:
        input_domain = _input_domain_from_highest_net(comp_nets, inductor_output_net)

    return input_domain, output_domain, domain


_BUCK_IO_PIN_KEYWORDS = ("VIN", "VOUT", "IN", "OUT", "FB")
_BUCK_BST_PIN_KEYWORDS = ("BST", "BOOT")


def _collect_buck_signal_nets(comp: Component) -> set[str]:
    """Collect signal nets from a buck converter's I/O and bootstrap pins.

    Skips global power rails on VIN/VOUT/FB pins since they connect to
    many unrelated components.
    """
    nets: set[str] = set()
    for pin in comp.pins:
        if not pin.net:
            continue
        pname = (pin.name or "").upper()
        if any(kw in pname for kw in _BUCK_IO_PIN_KEYWORDS):
            if not _is_power_net(pin.net):
                nets.add(pin.net)
        elif any(kw in pname for kw in _BUCK_BST_PIN_KEYWORDS):
            nets.add(pin.net)
    return nets


def _find_buck_inductor(
    comp: Component,
    net_to_refs: dict[str, set[str]],
    comp_map: dict[str, Component],
    claimed: set[str],
) -> tuple[str | None, str | None, set[str]]:
    """Find inductor on SW pin and its output net.

    Returns (inductor_ref, inductor_output_net, collected_nets).
    """
    collected: set[str] = set()
    for pin in comp.pins:
        if not pin.net:
            continue
        if not (pin.name and "SW" in pin.name.upper()):
            continue
        for r in net_to_refs.get(pin.net, set()):
            if _ref_prefix(r) != "L" or r in claimed:
                continue
            collected.add(pin.net)
            ind_comp = comp_map.get(r)
            output_net: str | None = None
            if ind_comp:
                for ip in ind_comp.pins:
                    if ip.net and ip.net != pin.net:
                        output_net = ip.net
                        collected.add(ip.net)
            return r, output_net, collected
    return None, None, collected


_MAX_CAPS_PER_NET = 2
_MAX_FB_RESISTORS = 2


def _is_cross_group_output_cap(
    r: str, net_name: str,
    inductor_output_net: str | None,
    output_is_power: bool,
    ref_group: dict[str, str],
    buck_group: str,
) -> bool:
    """Return True if *r* is a cap on the output net that belongs to a different group."""
    if not (output_is_power and net_name == inductor_output_net):
        return False
    cap_group = ref_group.get(r, "")
    return bool(cap_group and buck_group and cap_group != buck_group)


def _collect_buck_passives(
    comp: Component,
    inductor_output_net: str | None,
    net_to_refs: dict[str, set[str]],
    comp_map: dict[str, Component],
    ref_group: dict[str, str],
    claimed: set[str],
    refs: list[str],
) -> set[str]:
    """Collect caps and feedback resistors on buck signal nets.

    Mutates *refs* in place. Returns the set of nets that contributed refs.
    """
    signal_nets = _collect_buck_signal_nets(comp)
    output_is_power = False
    if inductor_output_net:
        signal_nets.add(inductor_output_net)
        output_is_power = _is_power_net(inductor_output_net)

    collected_nets: set[str] = set()
    buck_group = ref_group.get(comp.ref, "")

    for net_name in sorted(signal_nets):
        cap_count = 0
        fb_r_count = 0
        allow_resistors = not _is_power_net(net_name)
        max_caps = 1 if (
            net_name == inductor_output_net and output_is_power
        ) else _MAX_CAPS_PER_NET

        for r in sorted(net_to_refs.get(net_name, set())):
            if r in refs or r in claimed or not comp_map.get(r):
                continue
            prefix = _ref_prefix(r)
            if prefix == "C" and cap_count < max_caps:
                if _is_cross_group_output_cap(
                    r, net_name, inductor_output_net,
                    output_is_power, ref_group, buck_group,
                ):
                    continue
                refs.append(r)
                collected_nets.add(net_name)
                cap_count += 1
            elif prefix == "R" and allow_resistors and fb_r_count < _MAX_FB_RESISTORS:
                refs.append(r)
                collected_nets.add(net_name)
                fb_r_count += 1

    return collected_nets


def _detect_buck_converters(
    requirements: ProjectRequirements,
    net_to_refs: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    claimed: set[str],
) -> list[DetectedSubCircuit]:
    """Detect buck converter sub-circuits: IC with SW/FB pins + L + C_in + C_out."""
    results: list[DetectedSubCircuit] = []
    comp_map = {c.ref: c for c in requirements.components}

    buck_keywords = {"BUCK", "TPS54", "TPS56", "MP1584", "LM2596", "AP63",
                     "SY8089", "MT3608", "XL1509"}

    ref_group: dict[str, str] = _build_ref_group_index(requirements)

    for comp in requirements.components:
        if comp.ref in claimed or not comp.ref.startswith("U"):
            continue
        val_desc = f"{comp.value} {comp.description or ''}".upper()
        is_buck = any(kw in val_desc for kw in buck_keywords)
        has_sw_pin = any(
            p.name and "SW" in p.name.upper()
            for p in comp.pins
        )
        if not is_buck and not has_sw_pin:
            continue

        refs: list[str] = [comp.ref]
        comp_nets = ref_to_nets.get(comp.ref, set())
        all_nets: set[str] = set()

        # Find inductor on SW net
        ind_ref, inductor_output_net, ind_nets = _find_buck_inductor(
            comp, net_to_refs, comp_map, claimed,
        )
        if ind_ref:
            refs.append(ind_ref)
        all_nets.update(ind_nets)

        # Collect caps/resistors on signal nets
        passive_nets = _collect_buck_passives(
            comp, inductor_output_net, net_to_refs, comp_map,
            ref_group, claimed, refs,
        )
        all_nets.update(passive_nets)

        # Determine input/output domains
        input_domain, output_domain, domain = _classify_regulator_domains(
            comp, comp_nets, inductor_output_net,
        )

        for r in refs:
            claimed.add(r)

        results.append(DetectedSubCircuit(
            circuit_type=SubCircuitType.BUCK_CONVERTER,
            refs=tuple(sorted(refs)),
            anchor_ref=comp.ref,
            net_connections=tuple(sorted(all_nets)),
            domain=domain,
            layout_hint="boundary",
            input_domain=input_domain,
            output_domain=output_domain,
        ))

    return results


def _detect_ldo_regulators(
    requirements: ProjectRequirements,
    net_to_refs: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    claimed: set[str],
) -> list[DetectedSubCircuit]:
    """Detect LDO regulator sub-circuits: IC + C_in + C_out."""
    results: list[DetectedSubCircuit] = []

    ldo_keywords = {"LDO", "AMS1117", "LP5907", "LP5912", "AP2112",
                    "MCP1700", "NCV8114", "RT9013", "ME6211", "HT7333",
                    "HT7533", "SGM2019", "XC6206", "AP7361",
                    "LINEAR REG", "VOLTAGE REG"}

    for comp in requirements.components:
        if comp.ref in claimed:
            continue
        if not comp.ref.startswith("U"):
            continue
        val_desc = f"{comp.value} {comp.description or ''}".upper()
        is_ldo = any(kw in val_desc for kw in ldo_keywords)
        if not is_ldo:
            continue

        refs: list[str] = [comp.ref]
        comp_nets = ref_to_nets.get(comp.ref, set())
        all_nets: set[str] = set()

        # Find input/output caps
        for pin in comp.pins:
            if not pin.net:
                continue
            pname = (pin.name or "").upper()
            if any(kw in pname for kw in ("VIN", "VOUT", "IN", "OUT")):
                for r in net_to_refs.get(pin.net, set()):
                    if r in refs or r in claimed:
                        continue
                    if _ref_prefix(r) == "C":
                        refs.append(r)
                        all_nets.add(pin.net)

        # Determine input/output domains
        input_domain, output_domain, domain = _classify_regulator_domains(
            comp, comp_nets, None,
        )
        # LDO default domain is 3V3 if not inferred
        if output_domain is None:
            domain = VoltageDomain.DIGITAL_3V3

        for r in refs:
            claimed.add(r)

        results.append(DetectedSubCircuit(
            circuit_type=SubCircuitType.LDO_REGULATOR,
            refs=tuple(sorted(refs)),
            anchor_ref=comp.ref,
            net_connections=tuple(sorted(all_nets)),
            domain=domain,
            layout_hint="boundary",
            input_domain=input_domain,
            output_domain=output_domain,
        ))

    return results


def _detect_crystals(
    requirements: ProjectRequirements,
    net_to_refs: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    claimed: set[str],
) -> list[DetectedSubCircuit]:
    """Detect crystal oscillator sub-circuits: Y + C_load1 + C_load2."""
    results: list[DetectedSubCircuit] = []

    crystals = [c for c in requirements.components if _ref_prefix(c.ref) == "Y"]

    for crystal in crystals:
        if crystal.ref in claimed:
            continue
        refs: list[str] = [crystal.ref]
        all_nets: set[str] = set()

        # Find load caps on crystal pins
        crystal_nets = ref_to_nets.get(crystal.ref, set())
        for net_name in crystal_nets:
            if _is_gnd_net(net_name):
                continue
            for r in net_to_refs.get(net_name, set()):
                if r in refs or r in claimed:
                    continue
                if _ref_prefix(r) == "C":
                    refs.append(r)
                    all_nets.add(net_name)

        for r in refs:
            claimed.add(r)

        results.append(DetectedSubCircuit(
            circuit_type=SubCircuitType.CRYSTAL_OSC,
            refs=tuple(sorted(refs)),
            anchor_ref=crystal.ref,
            net_connections=tuple(sorted(all_nets)),
            domain=VoltageDomain.DIGITAL_3V3,
        ))

    return results


def _find_best_decoupling_ic(
    cap_power: set[str],
    cap_group: str,
    ics: list[Component],
    ic_power_nets: dict[str, set[str]],
    ref_group: dict[str, str],
    claimed: set[str],
) -> tuple[str | None, int]:
    """Find the IC with the best power-net overlap for a decoupling cap.

    Prefers ICs in the same FeatureBlock, then highest overlap count.
    Returns (ic_ref, overlap_count).
    """
    best_ic: str | None = None
    best_overlap = 0
    best_same_group = False
    for ic in ics:
        if ic.ref in claimed:
            continue
        overlap = len(cap_power & ic_power_nets.get(ic.ref, set()))
        if overlap <= 0:
            continue
        same_group = ref_group.get(ic.ref, "") == cap_group
        if same_group and not best_same_group:
            best_ic = ic.ref
            best_overlap = overlap
            best_same_group = True
        elif same_group == best_same_group and overlap > best_overlap:
            best_ic = ic.ref
            best_overlap = overlap
    return best_ic, best_overlap


def _domain_from_power_nets(shared_nets: set[str]) -> VoltageDomain:
    """Determine voltage domain from a set of shared power nets."""
    for net_name in shared_nets:
        if _is_gnd_net(net_name):
            continue
        v = _parse_voltage_from_net(net_name)
        if v is not None:
            return _classify_voltage(v)
    return VoltageDomain.DIGITAL_3V3


def _detect_decoupling_pairs(
    requirements: ProjectRequirements,
    net_to_refs: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    claimed: set[str],
) -> list[DetectedSubCircuit]:
    """Detect decoupling capacitor groups: all caps near an IC's power pins.

    Merges multiple caps sharing the same IC anchor into a single sub-circuit
    so the placement engine treats them as one group.
    """
    ics = [c for c in requirements.components
           if c.ref.startswith("U") and c.ref not in claimed]
    caps = [c for c in requirements.components
            if _is_decoupling_cap(c.ref, c.value) and c.ref not in claimed]

    # Build IC -> power nets mapping
    ic_power_nets: dict[str, set[str]] = {}
    for ic in ics:
        pnets = {n for n in ref_to_nets.get(ic.ref, set()) if _is_power_net(n)}
        ic_power_nets[ic.ref] = pnets

    ref_group = _build_ref_group_index(requirements)

    # Collect caps per IC anchor
    ic_caps: dict[str, list[str]] = {}
    ic_shared_nets: dict[str, set[str]] = {}

    for cap in caps:
        if cap.ref in claimed:
            continue
        cap_nets = ref_to_nets.get(cap.ref, set())
        cap_power = {n for n in cap_nets if _is_power_net(n)}
        cap_group = ref_group.get(cap.ref, "")

        best_ic, best_overlap = _find_best_decoupling_ic(
            cap_power, cap_group, ics, ic_power_nets, ref_group, claimed,
        )

        if best_ic and best_overlap > 0:
            claimed.add(cap.ref)
            ic_caps.setdefault(best_ic, []).append(cap.ref)
            shared = cap_power & ic_power_nets.get(best_ic, set())
            ic_shared_nets.setdefault(best_ic, set()).update(shared)

    # Build one sub-circuit per IC with same-group decoupling caps only.
    results: list[DetectedSubCircuit] = []
    for ic_ref, cap_refs in ic_caps.items():
        ic_grp = ref_group.get(ic_ref, "")
        same_group_caps = [c for c in cap_refs if ref_group.get(c, "") == ic_grp]
        if not same_group_caps:
            continue
        shared = ic_shared_nets.get(ic_ref, set())
        domain = _domain_from_power_nets(shared)

        all_refs = tuple(sorted([ic_ref, *same_group_caps]))
        results.append(DetectedSubCircuit(
            circuit_type=SubCircuitType.DECOUPLING,
            refs=all_refs,
            anchor_ref=ic_ref,
            net_connections=tuple(sorted(shared)),
            domain=domain,
        ))

    return results


def _divider_connected_to_connector(
    all_nets: set[str],
    net_to_refs: dict[str, set[str]],
) -> bool:
    """Return True if any non-power signal net in *all_nets* leads to a connector (J*)."""
    for net_name in all_nets:
        if _is_power_net(net_name) or _is_gnd_net(net_name):
            continue
        if any(_ref_prefix(ref) == "J" for ref in net_to_refs.get(net_name, set())):
            return True
    return False


def _detect_voltage_dividers(
    requirements: ProjectRequirements,
    net_to_refs: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    adj: dict[str, set[str]],
    claimed: set[str],
) -> list[DetectedSubCircuit]:
    """Detect voltage dividers: R1 + R2 in series between power/connector and GND."""
    results: list[DetectedSubCircuit] = []

    resistors = [c for c in requirements.components
                 if _ref_prefix(c.ref) == "R" and c.ref not in claimed]

    seen_pairs: set[tuple[str, str]] = set()

    for r1 in resistors:
        if r1.ref in claimed:
            continue
        r1_nets = ref_to_nets.get(r1.ref, set())
        # A voltage divider has one R on a power net, connected to another R
        # that connects to GND, with the midpoint going to an IC
        neighbours = adj.get(r1.ref, set())
        for nb in neighbours:
            if _ref_prefix(nb) != "R" or nb in claimed:
                continue
            pair_sorted = sorted((r1.ref, nb))
            pair = (pair_sorted[0], pair_sorted[1])
            if pair in seen_pairs:
                continue
            nb_nets = ref_to_nets.get(nb, set())
            # Check: one end on power, other end on GND, midpoint shared signal
            has_power = any(
                _is_power_net(n) and not _is_gnd_net(n)
                for n in r1_nets | nb_nets
            )
            # Also accept sensor-input dividers: one R connects to a
            # connector (J*) via a non-power signal net (e.g. AIN_RAW).
            if not has_power:
                has_power = _divider_connected_to_connector(
                    r1_nets | nb_nets, net_to_refs,
                )
            has_gnd = any(_is_gnd_net(n) for n in r1_nets | nb_nets)
            shared_signal = (r1_nets & nb_nets) - {
                n for n in r1_nets & nb_nets if _is_power_net(n)
            }
            if has_power and has_gnd and shared_signal:
                seen_pairs.add(pair)
                claimed.add(r1.ref)
                claimed.add(nb)

                domain = VoltageDomain.MIXED
                for n in r1_nets | nb_nets:
                    if _is_power_net(n) and not _is_gnd_net(n):
                        v = _parse_voltage_from_net(n)
                        if v is not None:
                            domain = _classify_voltage(v)
                            break

                results.append(DetectedSubCircuit(
                    circuit_type=SubCircuitType.VOLTAGE_DIVIDER,
                    refs=tuple(sorted((r1.ref, nb))),
                    anchor_ref=r1.ref,
                    net_connections=tuple(sorted(shared_signal)),
                    domain=domain,
                ))

    return results


# ---------------------------------------------------------------------------
# MCU peripheral detection
# ---------------------------------------------------------------------------

_RF_MODULE_KEYWORDS: frozenset[str] = frozenset({
    "ESP32", "ESP8266", "NRF", "CC3200", "RF", "WROOM", "WROVER",
    "NRF52", "NRF51", "CC2640", "CC1310",
})

_MCU_KEYWORDS: frozenset[str] = frozenset({
    "STM32", "ATMEGA", "ATTINY", "PIC", "MSP430", "RP2040", "RP2350",
    "SAMD", "ESP32", "ESP8266", "NRF52", "NRF51", "EFM32", "GD32",
    "CH32", "WCH",
})


def _find_mcu_ref(
    requirements: ProjectRequirements,
) -> str | None:
    """Find the MCU reference designator.

    Prefers ICs matching MCU keywords; falls back to the U* component
    with the most pins.
    """
    u_comps = [c for c in requirements.components if c.ref.startswith("U")]
    if not u_comps:
        return None

    # First: keyword match
    for comp in u_comps:
        val_desc = f"{comp.value} {comp.description or ''}".upper()
        if any(kw in val_desc for kw in _MCU_KEYWORDS):
            return comp.ref

    # Fallback: largest pin count
    best = max(u_comps, key=lambda c: len(c.pins))
    if len(best.pins) >= 8:
        return best.ref
    return None


_I2C_SPI_KEYWORDS: frozenset[str] = frozenset({
    "SDA", "SCL", "I2C", "MOSI", "MISO", "SCK", "SPI", "SCLK",
})


_DEBUG_DISPLAY_KEYWORDS: tuple[str, ...] = (
    "DEBUG", "JTAG", "SWD", "DISPLAY", "OLED", "LCD", "UART", "SERIAL",
)


def _is_mcu_peripheral_connector(comp: Component) -> bool:
    """Return True if *comp* is a debug/display or small signal connector."""
    val_desc = f"{comp.value} {comp.description or ''}".upper()
    if any(kw in val_desc for kw in _DEBUG_DISPLAY_KEYWORDS):
        return True
    pin_count = len(comp.pins) if comp.pins else 0
    return 2 <= pin_count <= 6


def _is_bus_pullup_resistor(
    ref: str,
    ref_to_nets: dict[str, set[str]],
) -> bool:
    """Return True if resistor *ref* is on an I2C/SPI bus net."""
    r_nets = ref_to_nets.get(ref, set())
    return any(
        any(kw in n.upper() for kw in _I2C_SPI_KEYWORDS)
        for n in r_nets
    )


def _collect_resistor_led_pair(
    resistor_ref: str,
    mcu_ref: str,
    adj: dict[str, set[str]],
    comp_map: dict[str, Component],
    claimed: set[str],
    peripheral_refs: list[str],
) -> None:
    """If *resistor_ref* drives an LED, append both to *peripheral_refs*."""
    r_neighbours = adj.get(resistor_ref, set())
    for rn in r_neighbours:
        if rn in claimed or rn == mcu_ref:
            continue
        if _ref_prefix(rn) not in ("LED", "D"):
            continue
        comp = comp_map.get(rn)
        if comp and "LED" in (comp.value or "").upper():
            if resistor_ref not in peripheral_refs:
                peripheral_refs.append(resistor_ref)
            if rn not in peripheral_refs:
                peripheral_refs.append(rn)
            return


def _detect_mcu_peripherals(
    requirements: ProjectRequirements,
    adj: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    net_to_refs: dict[str, set[str]],
    claimed: set[str],
) -> list[DetectedSubCircuit]:
    """Detect MCU peripheral cluster: switches, LEDs, debug headers, test points,
    I2C/SPI pull-up resistors, and small connectors near MCU.

    Walks signal adjacency from the MCU to find directly-connected
    peripherals (SW*, LED*, TP*, debug/display connectors, I2C pullups,
    small signal connectors).
    """
    mcu_ref = _find_mcu_ref(requirements)
    if mcu_ref is None or mcu_ref in claimed:
        return []

    comp_map = {c.ref: c for c in requirements.components}

    mcu_neighbours = adj.get(mcu_ref, set())
    peripheral_refs: list[str] = []
    peripheral_prefixes = {"SW", "LED", "BTN", "TP"}

    for nb in mcu_neighbours:
        if nb in claimed:
            continue
        prefix = _ref_prefix(nb)
        if prefix in peripheral_prefixes:
            peripheral_refs.append(nb)
            continue
        if prefix == "J":
            comp = comp_map.get(nb)
            if comp and _is_mcu_peripheral_connector(comp) and nb not in peripheral_refs:
                peripheral_refs.append(nb)
                continue
        if prefix == "R":
            if _is_bus_pullup_resistor(nb, ref_to_nets) and nb not in peripheral_refs:
                peripheral_refs.append(nb)
                continue
            _collect_resistor_led_pair(
                nb, mcu_ref, adj, comp_map, claimed, peripheral_refs,
            )

    if not peripheral_refs:
        return []

    for r in peripheral_refs:
        claimed.add(r)

    return [DetectedSubCircuit(
        circuit_type=SubCircuitType.MCU_PERIPHERAL_CLUSTER,
        refs=tuple(sorted(peripheral_refs)),
        anchor_ref=mcu_ref,
        net_connections=(),
        domain=VoltageDomain.DIGITAL_3V3,
        layout_hint="cluster",
    )]


def _detect_rf_antenna(
    requirements: ProjectRequirements,
    claimed: set[str],
) -> list[DetectedSubCircuit]:
    """Detect RF/WiFi modules requiring edge placement.

    Finds ESP32/nRF/CC modules and creates an RF_ANTENNA subcircuit
    with edge layout hint.
    """
    results: list[DetectedSubCircuit] = []

    for comp in requirements.components:
        if comp.ref in claimed:
            continue
        if not comp.ref.startswith("U"):
            continue
        val_desc = f"{comp.value} {comp.description or ''}".upper()
        if any(kw in val_desc for kw in _RF_MODULE_KEYWORDS):
            claimed.add(comp.ref)
            results.append(DetectedSubCircuit(
                circuit_type=SubCircuitType.RF_ANTENNA,
                refs=(comp.ref,),
                anchor_ref=comp.ref,
                net_connections=(),
                domain=VoltageDomain.DIGITAL_3V3,
                layout_hint="edge",
            ))

    return results


def _find_divider_connector(
    divider_nets: set[str],
    net_to_refs: dict[str, set[str]],
    divider_refs: tuple[str, ...],
    claimed: set[str],
) -> str | None:
    """Find a connector (J*) on signal nets connected to a voltage divider."""
    for net_name in divider_nets:
        if _is_power_net(net_name) or _is_gnd_net(net_name):
            continue
        for ref in net_to_refs.get(net_name, set()):
            if ref in claimed or ref in divider_refs:
                continue
            if _ref_prefix(ref) == "J":
                return ref
    return None


def _divider_has_mcu_connection(
    midpoint_nets: tuple[str, ...],
    net_to_refs: dict[str, set[str]],
    mcu_ref: str | None,
) -> bool:
    """Check if divider midpoint nets reach the MCU or an ADC-keyword net."""
    for net_name in midpoint_nets:
        for ref in net_to_refs.get(net_name, set()):
            if ref == mcu_ref:
                return True
            if any(kw in net_name.upper() for kw in _ANALOG_KEYWORDS):
                return True
    return False


def _collect_protection_components(
    divider_nets: set[str],
    net_to_refs: dict[str, set[str]],
    claimed: set[str],
    adc_refs: list[str],
) -> None:
    """Append TVS/zener and filter-cap components on signal nets to *adc_refs*.

    Collects D/Z (protection diodes) and C (filter capacitors) that sit on
    the same signal nets as the voltage divider.  This ensures the full
    channel signal chain (divider + TVS + filter cap) is grouped together.
    """
    for net_name in divider_nets:
        if _is_power_net(net_name) or _is_gnd_net(net_name):
            continue
        for ref in net_to_refs.get(net_name, set()):
            if ref in claimed or ref in adc_refs:
                continue
            if _ref_prefix(ref) in ("D", "Z", "C"):
                adc_refs.append(ref)
                claimed.add(ref)


def _detect_adc_channels(
    requirements: ProjectRequirements,
    net_to_refs: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    adj: dict[str, set[str]],
    subcircuits: list[DetectedSubCircuit],
    claimed: set[str],
) -> list[DetectedSubCircuit]:
    """Detect ADC channel subcircuits: voltage divider + connector + protection.

    For each VOLTAGE_DIVIDER subcircuit, traces nets to find a connected
    connector (J*) and an ADC/MCU pin. If both found, creates an ADC_CHANNEL
    subcircuit grouping the divider, connector, and any protection components.
    The anchor is the connector (channel should be placed near its terminal).
    """
    results: list[DetectedSubCircuit] = []
    mcu_ref = _find_mcu_ref(requirements)

    dividers = [sc for sc in subcircuits
                if sc.circuit_type == SubCircuitType.VOLTAGE_DIVIDER]

    for divider in dividers:
        divider_nets: set[str] = set()
        for ref in divider.refs:
            divider_nets.update(ref_to_nets.get(ref, set()))

        connector_ref = _find_divider_connector(
            divider_nets, net_to_refs, divider.refs, claimed,
        )
        if not connector_ref:
            continue
        if not _divider_has_mcu_connection(
            divider.net_connections, net_to_refs, mcu_ref,
        ):
            continue

        adc_refs: list[str] = list(divider.refs)
        adc_refs.append(connector_ref)
        claimed.add(connector_ref)

        _collect_protection_components(divider_nets, net_to_refs, claimed, adc_refs)

        results.append(DetectedSubCircuit(
            circuit_type=SubCircuitType.ADC_CHANNEL,
            refs=tuple(sorted(adc_refs)),
            anchor_ref=connector_ref,
            net_connections=divider.net_connections,
            domain=divider.domain,
            layout_hint="cluster",
        ))

    return results


# ---------------------------------------------------------------------------
# Power flow topology
# ---------------------------------------------------------------------------


def _voltage_magnitude(domain: VoltageDomain) -> float:
    """Return a representative voltage magnitude for domain ordering."""
    return {
        VoltageDomain.VIN_24V: 24.0,
        VoltageDomain.POWER_5V: 5.0,
        VoltageDomain.DIGITAL_3V3: 3.3,
        VoltageDomain.ANALOG: 3.3,
        VoltageDomain.MIXED: 0.0,
    }.get(domain, 0.0)


def _collect_regulator_graph(
    subcircuits: tuple[DetectedSubCircuit, ...],
) -> tuple[
    list[tuple[VoltageDomain, VoltageDomain, str]],
    dict[VoltageDomain, set[VoltageDomain]],
    set[VoltageDomain],
]:
    """Collect regulator edges and all voltage domains from subcircuits.

    Returns (boundaries, edges, all_domains).
    """
    boundaries: list[tuple[VoltageDomain, VoltageDomain, str]] = []
    edges: dict[VoltageDomain, set[VoltageDomain]] = {}
    all_domains: set[VoltageDomain] = set()

    for sc in subcircuits:
        if sc.circuit_type not in (
            SubCircuitType.BUCK_CONVERTER, SubCircuitType.LDO_REGULATOR,
        ):
            if sc.domain != VoltageDomain.MIXED:
                all_domains.add(sc.domain)
            continue

        if sc.input_domain is not None and sc.output_domain is not None:
            boundaries.append((sc.input_domain, sc.output_domain, sc.anchor_ref))
            edges.setdefault(sc.input_domain, set()).add(sc.output_domain)
            all_domains.add(sc.input_domain)
            all_domains.add(sc.output_domain)
        if sc.domain != VoltageDomain.MIXED:
            all_domains.add(sc.domain)

    all_domains.discard(VoltageDomain.MIXED)
    return boundaries, edges, all_domains


def _topo_sort_domains(
    edges: dict[VoltageDomain, set[VoltageDomain]],
    all_domains: set[VoltageDomain],
) -> list[VoltageDomain]:
    """Topological sort of voltage domains using Kahn's algorithm."""
    in_degree: dict[VoltageDomain, int] = {d: 0 for d in all_domains}
    for _src, dsts in edges.items():
        for dst in dsts:
            if dst in in_degree:
                in_degree[dst] += 1

    queue = sorted(
        [d for d, deg in in_degree.items() if deg == 0],
        key=_voltage_magnitude, reverse=True,
    )
    result: list[VoltageDomain] = []
    visited: set[VoltageDomain] = set()

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        result.append(node)
        for dst in sorted(edges.get(node, set()), key=_voltage_magnitude, reverse=True):
            if dst in in_degree:
                in_degree[dst] -= 1
                if in_degree[dst] <= 0 and dst not in visited:
                    queue.append(dst)
        queue.sort(key=_voltage_magnitude, reverse=True)

    # Add disconnected domains
    for d in sorted(all_domains - visited, key=_voltage_magnitude, reverse=True):
        result.append(d)
    return result


def compute_power_flow_topology(
    subcircuits: tuple[DetectedSubCircuit, ...],
) -> PowerFlowTopology:
    """Derive power flow topology from regulator subcircuits.

    Walks BUCK_CONVERTER and LDO_REGULATOR subcircuits to build a graph
    of input_domain -> output_domain edges, then topologically sorts
    the domains from highest voltage to lowest.

    Falls back to voltage-magnitude ordering when no regulators are found.

    Args:
        subcircuits: Detected subcircuits (must include regulators).

    Returns:
        PowerFlowTopology with ordered domains and boundary info.
    """
    boundaries, edges, all_domains = _collect_regulator_graph(subcircuits)

    if not boundaries:
        ordered = sorted(all_domains, key=_voltage_magnitude, reverse=True)
        return PowerFlowTopology(
            domain_order=tuple(ordered) if ordered else (VoltageDomain.MIXED,),
            regulator_boundaries=(),
        )

    result = _topo_sort_domains(edges, all_domains)
    return PowerFlowTopology(
        domain_order=tuple(result),
        regulator_boundaries=tuple(boundaries),
    )


# ---------------------------------------------------------------------------
# Cross-domain affinity detection
# ---------------------------------------------------------------------------


_FEEDBACK_KEYWORDS: frozenset[str] = frozenset({"FB", "FEEDBACK", "SENSE"})
_CONTROL_KEYWORDS: frozenset[str] = frozenset(
    {"CTRL", "CONTROL", "EN", "ENABLE"},
)


def _classify_affinity_reason(net_name_upper: str) -> str | None:
    """Classify cross-domain affinity reason from net name keywords."""
    if any(kw in net_name_upper for kw in _ANALOG_KEYWORDS):
        return "measurement"
    if any(kw in net_name_upper for kw in _FEEDBACK_KEYWORDS):
        return "feedback"
    if any(kw in net_name_upper for kw in _CONTROL_KEYWORDS):
        return "control"
    return None


def detect_cross_domain_affinities(
    requirements: ProjectRequirements,
    domain_map: dict[str, VoltageDomain],
) -> tuple[DomainAffinity, ...]:
    """Detect cross-domain component affinities for placement co-location.

    Identifies signal nets that cross voltage domain boundaries where
    components should be placed near each other despite being in different
    domains (e.g. ADC/analog monitoring of relay outputs).

    Args:
        requirements: Project requirements with nets.
        domain_map: Component ref to voltage domain mapping.

    Returns:
        Tuple of detected cross-domain affinities.
    """
    affinities: list[DomainAffinity] = []

    for net in requirements.nets:
        net_name = net.name.upper()
        # Skip power and ground nets
        if _is_power_net(net.name) or _is_gnd_net(net.name):
            continue

        # Get domains of all components on this net
        refs = [conn.ref for conn in net.connections]
        if len(refs) < 2:
            continue

        domains_on_net: dict[VoltageDomain, list[str]] = {}
        for ref in refs:
            d = domain_map.get(ref, VoltageDomain.MIXED)
            if d != VoltageDomain.MIXED:
                domains_on_net.setdefault(d, []).append(ref)

        # If net spans 2+ different domains, check for affinity
        domain_keys = [d for d in domains_on_net if d != VoltageDomain.MIXED]
        if len(domain_keys) < 2:
            continue

        # Classify affinity reason from net name
        reason = _classify_affinity_reason(net_name)

        if reason is None:
            continue

        # Check each pair of domains on this net
        for di in range(len(domain_keys)):
            for dj in range(di + 1, len(domain_keys)):
                d1, d2 = domain_keys[di], domain_keys[dj]
                affinities.append(DomainAffinity(
                    source_refs=tuple(sorted(domains_on_net[d1])),
                    target_refs=tuple(sorted(domains_on_net[d2])),
                    source_domain=d1,
                    target_domain=d2,
                    reason=reason,
                ))

    return tuple(affinities)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_subcircuits(
    requirements: ProjectRequirements,
) -> tuple[DetectedSubCircuit, ...]:
    """Detect all sub-circuits from netlist topology.

    Analyses component references, pin names, net connectivity and signal
    adjacency to identify common EE sub-circuit patterns.

    Args:
        requirements: Full project requirements with components and nets.

    Returns:
        Tuple of detected sub-circuits, each with type, refs, anchor, and
        voltage domain.
    """
    net_to_refs = _net_refs(requirements)
    ref_to_nets = _ref_nets(requirements)
    adj = build_signal_adjacency(requirements)
    claimed: set[str] = set()

    all_subcircuits: list[DetectedSubCircuit] = []

    # Order matters: detect more specific patterns first
    relay_drivers = _detect_relay_drivers(
        requirements, net_to_refs, ref_to_nets, adj,
    )
    all_subcircuits.extend(relay_drivers)
    for sc in relay_drivers:
        claimed.update(sc.refs)

    bucks = _detect_buck_converters(
        requirements, net_to_refs, ref_to_nets, claimed,
    )
    all_subcircuits.extend(bucks)

    ldos = _detect_ldo_regulators(
        requirements, net_to_refs, ref_to_nets, claimed,
    )
    all_subcircuits.extend(ldos)

    crystals = _detect_crystals(
        requirements, net_to_refs, ref_to_nets, claimed,
    )
    all_subcircuits.extend(crystals)

    dividers = _detect_voltage_dividers(
        requirements, net_to_refs, ref_to_nets, adj, claimed,
    )
    all_subcircuits.extend(dividers)

    # ADC channel detection: runs BEFORE decoupling so filter caps on
    # protection nets (AIN*_PROT) get claimed as channel members rather
    # than being misclassified as IC decoupling caps.
    adc_channels = _detect_adc_channels(
        requirements, net_to_refs, ref_to_nets, adj,
        all_subcircuits, claimed,
    )
    all_subcircuits.extend(adc_channels)

    decoupling = _detect_decoupling_pairs(
        requirements, net_to_refs, ref_to_nets, claimed,
    )
    all_subcircuits.extend(decoupling)

    # RF antenna detection (before MCU peripherals so RF module isn't claimed)
    rf_antennas = _detect_rf_antenna(requirements, claimed)
    all_subcircuits.extend(rf_antennas)

    # MCU peripheral cluster detection (expanded: TP, I2C pullups, small connectors)
    mcu_peripherals = _detect_mcu_peripherals(
        requirements, adj, ref_to_nets, net_to_refs, claimed,
    )
    all_subcircuits.extend(mcu_peripherals)

    _log.info(
        "Detected %d sub-circuits: %s",
        len(all_subcircuits),
        {t.value: sum(1 for s in all_subcircuits if s.circuit_type == t)
         for t in SubCircuitType if any(s.circuit_type == t for s in all_subcircuits)},
    )

    return tuple(all_subcircuits)


def classify_voltage_domains(
    requirements: ProjectRequirements,
) -> dict[str, VoltageDomain]:
    """Classify every component into a voltage domain.

    Uses net names to determine the dominant voltage rail each component
    connects to. Components on multiple domains are classified as MIXED.

    Args:
        requirements: Full project requirements with components and nets.

    Returns:
        Dict mapping component ref to its voltage domain.
    """
    ref_to_nets = _ref_nets(requirements)
    result: dict[str, VoltageDomain] = {}

    for comp in requirements.components:
        nets = ref_to_nets.get(comp.ref, set())
        domains: set[VoltageDomain] = set()

        for net_name in nets:
            # Check for analog keywords in net name
            if any(kw in net_name.upper() for kw in _ANALOG_KEYWORDS):
                domains.add(VoltageDomain.ANALOG)
                continue

            if not _is_power_net(net_name):
                continue
            if _is_gnd_net(net_name):
                continue
            v = _parse_voltage_from_net(net_name)
            if v is not None:
                domains.add(_classify_voltage(v))

        if not domains:
            # No power net → classify by component type
            if _ref_prefix(comp.ref) == "K":
                result[comp.ref] = VoltageDomain.VIN_24V
            elif _ref_prefix(comp.ref) == "Y":
                result[comp.ref] = VoltageDomain.DIGITAL_3V3
            else:
                result[comp.ref] = VoltageDomain.MIXED
        elif len(domains) == 1:
            result[comp.ref] = next(iter(domains))
        else:
            # Multiple domains — pick highest voltage as primary
            priority = [VoltageDomain.VIN_24V, VoltageDomain.POWER_5V,
                        VoltageDomain.DIGITAL_3V3, VoltageDomain.ANALOG]
            for d in priority:
                if d in domains:
                    result[comp.ref] = d
                    break
            else:
                result[comp.ref] = VoltageDomain.MIXED

    return result


def _collect_active_domains(
    domain_subcircuits: dict[VoltageDomain, list[DetectedSubCircuit]],
    domain_loose: dict[VoltageDomain, list[str]],
    topology: PowerFlowTopology | None,
) -> list[VoltageDomain]:
    """Return ordered list of domains that have components."""
    active: list[VoltageDomain] = []

    def _has_components(d: VoltageDomain) -> bool:
        return d in domain_subcircuits or d in domain_loose

    if topology is not None and len(topology.domain_order) > 0:
        for d in topology.domain_order:
            if _has_components(d):
                active.append(d)
        for d in VoltageDomain:
            if d not in active and _has_components(d):
                active.append(d)
    else:
        priority = [
            VoltageDomain.VIN_24V, VoltageDomain.POWER_5V,
            VoltageDomain.ANALOG, VoltageDomain.DIGITAL_3V3,
            VoltageDomain.MIXED,
        ]
        for d in priority:
            if _has_components(d):
                active.append(d)
    return active


def _compute_zone_rects(
    active_domains: list[VoltageDomain],
    board_width: float,
    board_height: float,
    boundary_w: float,
) -> dict[VoltageDomain, tuple[float, float, float, float]]:
    """Compute rectangular zone areas for each domain.

    Landscape boards use left-to-right strips; portrait/square use top-to-bottom.
    """
    n_zones = len(active_domains)
    total_boundary = boundary_w * max(0, n_zones - 1)
    is_landscape = board_width > board_height * 1.3
    rects: dict[VoltageDomain, tuple[float, float, float, float]] = {}

    if is_landscape:
        usable = board_width - total_boundary
        strip = usable / n_zones if n_zones > 0 else board_width
        cursor = 0.0
        for domain in active_domains:
            rects[domain] = (cursor, 0.0, cursor + strip, board_height)
            cursor += strip + boundary_w
    else:
        usable = board_height - total_boundary
        strip = usable / n_zones if n_zones > 0 else board_height
        cursor = 0.0
        for domain in active_domains:
            rects[domain] = (0.0, cursor, board_width, cursor + strip)
            cursor += strip + boundary_w

    return rects


def assign_zones(
    subcircuits: tuple[DetectedSubCircuit, ...],
    domain_map: dict[str, VoltageDomain],
    board_width: float,
    board_height: float,
    all_refs: tuple[str, ...],
    topology: PowerFlowTopology | None = None,
) -> tuple[BoardZoneAssignment, ...]:
    """Assign sub-circuits and loose components to board zones by domain.

    When *topology* is provided, zones follow the power flow ordering:
    - Landscape boards (width > height x 1.3): left-to-right zones
    - Portrait/square boards: top-to-bottom zones
    - Boundary strips reserved between adjacent zones for regulators

    Falls back to quadrant layout when no topology is available.

    Args:
        subcircuits: Detected sub-circuits.
        domain_map: Component ref -> voltage domain mapping.
        board_width: Board width in mm.
        board_height: Board height in mm.
        all_refs: All component refs in the design.
        topology: Optional power flow topology for ordering.

    Returns:
        Tuple of zone assignments.
    """
    from kicad_pipeline.constants import ZONE_BOUNDARY_WIDTH_MM

    # Collect refs in subcircuits
    subcircuit_refs: set[str] = set()
    for sc in subcircuits:
        subcircuit_refs.update(sc.refs)

    # Group subcircuits and loose refs by domain
    domain_subcircuits: dict[VoltageDomain, list[DetectedSubCircuit]] = {}
    for sc in subcircuits:
        domain_subcircuits.setdefault(sc.domain, []).append(sc)

    domain_loose: dict[VoltageDomain, list[str]] = {}
    for ref in all_refs:
        if ref not in subcircuit_refs:
            domain = domain_map.get(ref, VoltageDomain.MIXED)
            domain_loose.setdefault(domain, []).append(ref)

    active_domains = _collect_active_domains(
        domain_subcircuits, domain_loose, topology,
    )
    if not active_domains:
        return ()

    boundary_w = ZONE_BOUNDARY_WIDTH_MM if len(active_domains) > 1 else 0.0
    zone_rects = _compute_zone_rects(
        active_domains, board_width, board_height, boundary_w,
    )

    assignments: list[BoardZoneAssignment] = []
    for domain in active_domains:
        scs = domain_subcircuits.get(domain, [])
        loose = domain_loose.get(domain, [])
        if not scs and not loose:
            continue
        rect = zone_rects.get(domain, (0.0, 0.0, board_width, board_height))
        assignments.append(BoardZoneAssignment(
            domain=domain,
            zone_rect=rect,
            subcircuits=tuple(scs),
            loose_refs=tuple(sorted(loose)),
        ))

    return tuple(assignments)
