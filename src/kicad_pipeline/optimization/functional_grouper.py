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
    from kicad_pipeline.models.requirements import ProjectRequirements

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
        refs: list[str] = [relay.ref]
        relay_nets = ref_to_nets.get(relay.ref, set())
        all_nets: set[str] = set(relay_nets)

        # Find transistor driving the relay coil — connected via signal net
        signal_neighbours = adj.get(relay.ref, set())
        transistor: str | None = None
        for nb in signal_neighbours:
            if _ref_prefix(nb) == "Q" and nb not in claimed:
                transistor = nb
                refs.append(nb)
                break

        # Find flyback diode — shares a non-GND power/signal net with relay
        for net_name in relay_nets:
            if _is_gnd_net(net_name):
                continue
            for r in net_to_refs.get(net_name, set()):
                if _ref_prefix(r) == "D" and r not in claimed and r not in refs:
                    refs.append(r)
                    all_nets.add(net_name)
                    break

        # Find gate resistor — connected to transistor
        if transistor:
            t_neighbours = adj.get(transistor, set())
            for nb in t_neighbours:
                if _ref_prefix(nb) == "R" and nb not in claimed and nb not in refs:
                    refs.append(nb)
                    break

        # Find LED on collector path
        if transistor:
            t_nets = ref_to_nets.get(transistor, set())
            for net_name in t_nets:
                if _is_gnd_net(net_name) or _is_power_net(net_name):
                    continue
                for r in net_to_refs.get(net_name, set()):
                    if r in refs:
                        continue
                    if _ref_prefix(r) == "D" and r not in claimed:
                        # Check if it looks like an LED
                        comp = comp_map.get(r)
                        if comp and "LED" in (comp.value or "").upper():
                            refs.append(r)
                            break
                    elif _ref_prefix(r) == "LED" and r not in claimed:
                        refs.append(r)
                        break

        # Find LED current-limiting resistor
        led_refs = [r for r in refs if _ref_prefix(r) in ("LED", "D")]
        for lr in led_refs:
            lr_neighbours = adj.get(lr, set())
            for nb in lr_neighbours:
                if _ref_prefix(nb) == "R" and nb not in claimed and nb not in refs:
                    refs.append(nb)
                    break

        # Determine domain from relay nets
        domain = VoltageDomain.MIXED
        for net_name in relay_nets:
            v = _parse_voltage_from_net(net_name)
            if v is not None and v > 0:
                domain = _classify_voltage(v)
                break

        for r in refs:
            claimed.add(r)

        # Build hierarchical structure
        driver_refs: list[str] = []
        led_node_refs: list[str] = []
        flyback_ref: str | None = None
        gate_resistor_ref: str | None = None
        led_ref: str | None = None
        led_resistor_ref: str | None = None

        for r in refs:
            if r == relay.ref:
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
                # Determine if gate resistor or LED resistor
                if transistor and r in (adj.get(transistor, set())):
                    gate_resistor_ref = r
                else:
                    led_resistor_ref = r

        # Driver subgroup: Q + D_flyback + R_gate
        if flyback_ref:
            driver_refs.append(flyback_ref)
        if gate_resistor_ref:
            driver_refs.append(gate_resistor_ref)

        # LED subgroup
        if led_ref:
            led_node_refs.append(led_ref)
        if led_resistor_ref:
            led_node_refs.append(led_resistor_ref)

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

        hierarchy = SubCircuitNode(
            role="relay_body",
            refs=(relay.ref,),
            anchor_ref=relay.ref,
            children=tuple(children),
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

    for comp in requirements.components:
        if comp.ref in claimed:
            continue
        if not comp.ref.startswith("U"):
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
        for pin in comp.pins:
            if not pin.net:
                continue
            if pin.name and "SW" in pin.name.upper():
                for r in net_to_refs.get(pin.net, set()):
                    if _ref_prefix(r) == "L" and r not in claimed:
                        refs.append(r)
                        all_nets.add(pin.net)
                        break

        # Find capacitors on VIN/VOUT nets
        for pin in comp.pins:
            if not pin.net:
                continue
            pname = (pin.name or "").upper()
            if any(kw in pname for kw in ("VIN", "VOUT", "IN", "OUT", "FB")):
                for r in net_to_refs.get(pin.net, set()):
                    if r in refs or r in claimed:
                        continue
                    rc = comp_map.get(r)
                    is_cap = rc and _ref_prefix(r) == "C"
                    is_fb_r = rc and _ref_prefix(r) == "R" and "FB" in pname
                    if is_cap or is_fb_r:
                        refs.append(r)
                        all_nets.add(pin.net)

        # Determine input/output domains from pin nets
        input_domain: VoltageDomain | None = None
        output_domain: VoltageDomain | None = None
        domain = VoltageDomain.POWER_5V

        for pin in comp.pins:
            if not pin.net:
                continue
            pname = (pin.name or "").upper()
            v = _parse_voltage_from_net(pin.net)
            if v is not None and v > 0:
                vd = _classify_voltage(v)
                if any(kw in pname for kw in ("VIN", "IN")):
                    input_domain = vd
                elif any(kw in pname for kw in ("VOUT", "OUT")):
                    output_domain = vd
                    domain = vd

        if input_domain is None:
            # Fall back to highest-voltage net
            for net_name in comp_nets:
                v = _parse_voltage_from_net(net_name)
                if v is not None and v > 0:
                    d = _classify_voltage(v)
                    if input_domain is None or (v > 0 and d == VoltageDomain.VIN_24V):
                        input_domain = d

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
        input_domain: VoltageDomain | None = None
        output_domain: VoltageDomain | None = None
        domain = VoltageDomain.DIGITAL_3V3

        for pin in comp.pins:
            if not pin.net:
                continue
            pname = (pin.name or "").upper()
            v = _parse_voltage_from_net(pin.net)
            if v is not None and v > 0:
                vd = _classify_voltage(v)
                if any(kw in pname for kw in ("VIN", "IN")):
                    input_domain = vd
                elif any(kw in pname for kw in ("VOUT", "OUT")):
                    output_domain = vd
                    domain = vd

        if input_domain is None:
            for net_name in comp_nets:
                v = _parse_voltage_from_net(net_name)
                if v is not None and v > 0:
                    input_domain = _classify_voltage(v)
                    break

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
        pnets: set[str] = set()
        for net_name in ref_to_nets.get(ic.ref, set()):
            if _is_power_net(net_name):
                pnets.add(net_name)
        ic_power_nets[ic.ref] = pnets

    # Collect caps per IC anchor
    ic_caps: dict[str, list[str]] = {}
    ic_shared_nets: dict[str, set[str]] = {}

    for cap in caps:
        if cap.ref in claimed:
            continue
        cap_nets = ref_to_nets.get(cap.ref, set())
        cap_power = {n for n in cap_nets if _is_power_net(n)}

        # Find the IC sharing the most power nets
        best_ic: str | None = None
        best_overlap = 0
        for ic in ics:
            if ic.ref in claimed:
                continue
            overlap = len(cap_power & ic_power_nets.get(ic.ref, set()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ic = ic.ref

        if best_ic and best_overlap > 0:
            claimed.add(cap.ref)
            ic_caps.setdefault(best_ic, []).append(cap.ref)
            shared = cap_power & ic_power_nets.get(best_ic, set())
            ic_shared_nets.setdefault(best_ic, set()).update(shared)

    # Build one sub-circuit per IC with all its decoupling caps
    results: list[DetectedSubCircuit] = []
    for ic_ref, cap_refs in ic_caps.items():
        shared = ic_shared_nets.get(ic_ref, set())
        domain = VoltageDomain.DIGITAL_3V3
        for net_name in shared:
            if not _is_gnd_net(net_name):
                v = _parse_voltage_from_net(net_name)
                if v is not None:
                    domain = _classify_voltage(v)
                    break

        all_refs = tuple(sorted([ic_ref, *cap_refs]))
        results.append(DetectedSubCircuit(
            circuit_type=SubCircuitType.DECOUPLING,
            refs=all_refs,
            anchor_ref=ic_ref,
            net_connections=tuple(sorted(shared)),
            domain=domain,
        ))

    return results


def _detect_voltage_dividers(
    requirements: ProjectRequirements,
    net_to_refs: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    adj: dict[str, set[str]],
    claimed: set[str],
) -> list[DetectedSubCircuit]:
    """Detect voltage dividers: R1 + R2 in series between power and GND."""
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


def _detect_mcu_peripherals(
    requirements: ProjectRequirements,
    adj: dict[str, set[str]],
    claimed: set[str],
) -> list[DetectedSubCircuit]:
    """Detect MCU peripheral cluster: switches, LEDs, debug headers near MCU.

    Walks signal adjacency from the MCU to find directly-connected
    peripherals (SW*, LED*, debug/display connectors).
    """
    mcu_ref = _find_mcu_ref(requirements)
    if mcu_ref is None or mcu_ref in claimed:
        return []

    # Walk 1-hop signal adjacency from MCU
    mcu_neighbours = adj.get(mcu_ref, set())
    peripheral_refs: list[str] = []
    peripheral_prefixes = {"SW", "LED", "BTN"}

    for nb in mcu_neighbours:
        if nb in claimed:
            continue
        prefix = _ref_prefix(nb)
        if prefix in peripheral_prefixes:
            peripheral_refs.append(nb)
            continue
        # Check for debug/display connectors
        if prefix == "J":
            comp = next((c for c in requirements.components if c.ref == nb), None)
            if comp:
                val_desc = f"{comp.value} {comp.description or ''}".upper()
                if any(kw in val_desc for kw in ("DEBUG", "JTAG", "SWD", "DISPLAY",
                                                   "OLED", "LCD", "UART", "SERIAL")):
                    peripheral_refs.append(nb)
                    continue
        # Also pick up LEDs connected via resistor (2-hop)
        if prefix == "R":
            r_neighbours = adj.get(nb, set())
            for rn in r_neighbours:
                if rn in claimed or rn == mcu_ref:
                    continue
                if _ref_prefix(rn) in ("LED", "D"):
                    comp = next(
                        (c for c in requirements.components if c.ref == rn), None,
                    )
                    if comp and "LED" in (comp.value or "").upper():
                        if nb not in peripheral_refs:
                            peripheral_refs.append(nb)
                        if rn not in peripheral_refs:
                            peripheral_refs.append(rn)

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


# ---------------------------------------------------------------------------
# Cross-domain affinity detection
# ---------------------------------------------------------------------------


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
        reason: str | None = None
        if any(kw in net_name for kw in _ANALOG_KEYWORDS):
            reason = "measurement"
        elif any(kw in net_name for kw in ("FB", "FEEDBACK", "SENSE")):
            reason = "feedback"
        elif any(kw in net_name for kw in ("CTRL", "CONTROL", "EN", "ENABLE")):
            reason = "control"

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

    decoupling = _detect_decoupling_pairs(
        requirements, net_to_refs, ref_to_nets, claimed,
    )
    all_subcircuits.extend(decoupling)

    # RF antenna detection (before MCU peripherals so RF module isn't claimed)
    rf_antennas = _detect_rf_antenna(requirements, claimed)
    all_subcircuits.extend(rf_antennas)

    # MCU peripheral cluster detection
    mcu_peripherals = _detect_mcu_peripherals(requirements, adj, claimed)
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


def assign_zones(
    subcircuits: tuple[DetectedSubCircuit, ...],
    domain_map: dict[str, VoltageDomain],
    board_width: float,
    board_height: float,
    all_refs: tuple[str, ...],
) -> tuple[BoardZoneAssignment, ...]:
    """Assign sub-circuits and loose components to board zones by domain.

    Zone layout (from docs/placement_requirements.md Section 7):
    - Top-left: Power input (24V)
    - Top-right: Relay / 5V section
    - Middle-right: Analog
    - Bottom: Digital (3V3) / MCU
    - Loose connectors assigned to nearest edge zone

    Args:
        subcircuits: Detected sub-circuits.
        domain_map: Component ref → voltage domain mapping.
        board_width: Board width in mm.
        board_height: Board height in mm.
        all_refs: All component refs in the design.

    Returns:
        Tuple of zone assignments.
    """
    # Collect refs in subcircuits
    subcircuit_refs: set[str] = set()
    for sc in subcircuits:
        subcircuit_refs.update(sc.refs)

    # Group subcircuits by domain
    domain_subcircuits: dict[VoltageDomain, list[DetectedSubCircuit]] = {}
    for sc in subcircuits:
        domain_subcircuits.setdefault(sc.domain, []).append(sc)

    # Group loose refs (not in any subcircuit) by domain
    domain_loose: dict[VoltageDomain, list[str]] = {}
    for ref in all_refs:
        if ref in subcircuit_refs:
            continue
        domain = domain_map.get(ref, VoltageDomain.MIXED)
        domain_loose.setdefault(domain, []).append(ref)

    # Define zone rectangles — proportional to board dimensions
    half_w = board_width / 2.0
    half_h = board_height / 2.0
    zone_rects: dict[VoltageDomain, tuple[float, float, float, float]] = {
        VoltageDomain.VIN_24V: (0.0, 0.0, half_w, half_h),
        VoltageDomain.POWER_5V: (half_w, 0.0, board_width, half_h),
        VoltageDomain.ANALOG: (half_w, half_h * 0.6, board_width, half_h + half_h * 0.4),
        VoltageDomain.DIGITAL_3V3: (0.0, half_h, board_width, board_height),
        VoltageDomain.MIXED: (0.0, half_h, half_w, board_height),
    }

    assignments: list[BoardZoneAssignment] = []
    for domain in VoltageDomain:
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
