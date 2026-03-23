"""Infer board configuration from ProjectRequirements.

Traces nets to determine relay polarity, ADC divider ratios, bus members,
power rails, and connector pinouts — producing a machine-readable
BoardConfig that firmware agents can consume.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from kicad_pipeline.models.board_config import (
    ADCChannelConfig,
    BoardConfig,
    BusConfig,
    BusDevice,
    BusType,
    ConnectorPin,
    ConnectorPinout,
    PowerRailConfig,
    RelayConfig,
    RelayPolarity,
)

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import (
        Component,
        ProjectRequirements,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground / power net helpers (duplicated from functional_grouper to avoid
# coupling the config package to the optimisation package)
# ---------------------------------------------------------------------------

_GND_NAMES: frozenset[str] = frozenset(
    {"GND", "AGND", "DGND", "PGND", "VGND", "VSS", "VEE", "RELAY_GND"}
)

_VIN_PATTERNS: tuple[str, ...] = ("VIN", "+12V", "+24V", "VBUS", "V_IN")


def _is_gnd_net(name: str) -> bool:
    """Return True if *name* is a ground net."""
    return name.upper().strip() in _GND_NAMES


def _is_power_net(name: str) -> bool:
    """Return True if *name* is a power or ground net."""
    upper = name.upper().strip()
    if upper in _GND_NAMES:
        return True
    if upper.startswith(("+", "V", "-")):
        return True
    return bool(re.search(r"_\d+V\d*$", upper))


def _is_vin_net(name: str) -> bool:
    """Return True if *name* is a high-voltage input rail (VIN/+12V/+24V)."""
    upper = name.upper().strip()
    return upper in _VIN_PATTERNS or upper.startswith("VIN")


# ---------------------------------------------------------------------------
# Net ↔ ref mapping helpers
# ---------------------------------------------------------------------------


def _ref_nets(requirements: ProjectRequirements) -> dict[str, set[str]]:
    """Map each component ref to the set of nets it connects to."""
    ref_to_nets: dict[str, set[str]] = {}
    for net in requirements.nets:
        for conn in net.connections:
            ref_to_nets.setdefault(conn.ref, set()).add(net.name)
    return ref_to_nets


def _net_refs(requirements: ProjectRequirements) -> dict[str, set[str]]:
    """Map each net name to the set of component refs on it."""
    result: dict[str, set[str]] = {}
    for net in requirements.nets:
        refs = {conn.ref for conn in net.connections}
        result[net.name] = refs
    return result


def _net_pin_map(
    requirements: ProjectRequirements,
) -> dict[str, list[tuple[str, str]]]:
    """Map each net name to a list of (ref, pin_number) tuples."""
    result: dict[str, list[tuple[str, str]]] = {}
    for net in requirements.nets:
        for conn in net.connections:
            result.setdefault(net.name, []).append((conn.ref, conn.pin))
    return result


_MCU_KEYWORDS: tuple[str, ...] = ("ESP32", "STM32", "ATMEGA", "RP2040", "NRF52")


def _is_mcu_component(comp: Component) -> bool:
    """Return True if *comp* is a microcontroller based on its value string."""
    upper = (comp.value or "").upper()
    return any(kw in upper for kw in _MCU_KEYWORDS)


def _ref_prefix(ref: str) -> str:
    """Get the alpha prefix of a reference designator."""
    return "".join(c for c in ref if c.isalpha())


def _comp_map(requirements: ProjectRequirements) -> dict[str, Component]:
    """Map ref → Component for fast lookup."""
    return {c.ref: c for c in requirements.components}


# ---------------------------------------------------------------------------
# ADS1115 I2C address from ADDR pin net
# ---------------------------------------------------------------------------

_ADS1115_ADDR_MAP: dict[str, int] = {
    # ADDR pin connection → 7-bit I2C address
    "GND": 0x48,
    "AGND": 0x48,
    "DGND": 0x48,
    "VDD": 0x49,
    "VCC": 0x49,
    "AVCC": 0x49,
    "+3V3": 0x49,
    "SDA": 0x4A,
    "I2C_SDA": 0x4A,
    "SCL": 0x4B,
    "I2C_SCL": 0x4B,
}


# ---------------------------------------------------------------------------
# Relay inference
# ---------------------------------------------------------------------------


def _resolve_relay_pins(
    comp: Component,
) -> tuple[str, str, str, str]:
    """Resolve relay COM/NO/COIL pins to their net names."""
    com_pin = next((p for p in comp.pins if p.name.upper() == "COM"), None)
    no_pin = next((p for p in comp.pins if p.name.upper() == "NO"), None)
    coil_minus = next(
        (p for p in comp.pins if p.name.upper() in ("COIL-", "COIL_MINUS")),
        None,
    )
    coil_plus = next(
        (p for p in comp.pins if p.name.upper() in ("COIL+", "COIL_PLUS")),
        None,
    )

    # Fall back to pin numbers (SANYOU SRD convention)
    if com_pin is None:
        com_pin = comp.get_pin("1")
    if coil_minus is None:
        coil_minus = comp.get_pin("2")
    if no_pin is None:
        no_pin = comp.get_pin("3")
    if coil_plus is None:
        coil_plus = comp.get_pin("5")

    com_net = com_pin.net if com_pin and com_pin.net else ""
    output_net = no_pin.net if no_pin and no_pin.net else ""
    coil_net = coil_minus.net if coil_minus and coil_minus.net else ""
    coil_voltage_net = coil_plus.net if coil_plus and coil_plus.net else ""
    return com_net, output_net, coil_net, coil_voltage_net


def _determine_driver_polarity(
    q_comp: Component,
    relay_ref: str,
    driver_ref: str,
) -> tuple[RelayPolarity, list[str]]:
    """Determine relay polarity from transistor driver topology."""
    warnings: list[str] = []
    # SOT-23 BJT: pin 1=B, pin 2=C, pin 3=E
    emitter_pin = q_comp.get_pin("3")
    emitter_net = emitter_pin.net if emitter_pin else None
    if emitter_net and _is_gnd_net(emitter_net):
        return RelayPolarity.ACTIVE_HIGH, warnings
    if emitter_net and _is_power_net(emitter_net) and not _is_gnd_net(emitter_net):
        return RelayPolarity.ACTIVE_LOW, warnings
    warnings.append(
        f"{relay_ref}: driver {driver_ref} emitter net "
        f"'{emitter_net}' — cannot determine polarity"
    )
    return RelayPolarity.UNKNOWN, warnings


def _trace_gpio_through_base_resistor(
    q_comp: Component,
    net_to_refs: dict[str, set[str]],
    comps: dict[str, Component],
) -> str:
    """Trace from transistor base pin through resistor to find GPIO net."""
    base_pin = q_comp.get_pin("1")
    if not base_pin or not base_pin.net:
        return ""
    base_net = base_pin.net
    base_refs = net_to_refs.get(base_net, set())
    r_refs = [r for r in base_refs if _ref_prefix(r) == "R"]
    if not r_refs:
        # Direct drive (no base R)
        return base_net
    r_comp = comps.get(r_refs[0])
    if not r_comp:
        return ""
    for p in r_comp.pins:
        if p.net and p.net != base_net:
            return p.net
    return ""


def _trace_driver_chain(
    coil_net: str,
    relay_ref: str,
    net_to_refs: dict[str, set[str]],
    comps: dict[str, Component],
) -> tuple[str, str, RelayPolarity, list[str]]:
    """Trace coil driver chain to find driver ref, GPIO net, and polarity."""
    if not coil_net:
        return "", "", RelayPolarity.UNKNOWN, []

    coil_refs = net_to_refs.get(coil_net, set())
    q_refs = [r for r in coil_refs if _ref_prefix(r) == "Q"]
    if not q_refs:
        return "", "", RelayPolarity.UNKNOWN, [
            f"{relay_ref}: no transistor driver found on coil net"
        ]

    driver_ref = q_refs[0]
    q_comp = comps.get(driver_ref)
    if not q_comp:
        return driver_ref, "", RelayPolarity.UNKNOWN, []

    polarity, warnings = _determine_driver_polarity(q_comp, relay_ref, driver_ref)
    gpio_net = _trace_gpio_through_base_resistor(q_comp, net_to_refs, comps)
    return driver_ref, gpio_net, polarity, warnings


def _trace_gpio_to_mcu_pin(
    gpio_net: str,
    net_to_refs: dict[str, set[str]],
    comps: dict[str, Component],
) -> str:
    """Find the MCU pin name connected to a GPIO net."""
    if not gpio_net:
        return ""
    gpio_refs = net_to_refs.get(gpio_net, set())
    for r in gpio_refs:
        if _ref_prefix(r) != "U":
            continue
        comp = comps.get(r)
        if comp is None:
            continue
        if not (_is_mcu_component(comp) or "MCU" in (comp.description or "").upper()):
            continue
        for p in comp.pins:
            if p.net == gpio_net:
                return p.name
    return ""


def _infer_relays(requirements: ProjectRequirements) -> tuple[tuple[RelayConfig, ...], list[str]]:
    """Detect relay polarity by tracing COM pin and coil driver chain."""
    comps = _comp_map(requirements)
    net_to_refs = _net_refs(requirements)
    warnings: list[str] = []
    relays: list[RelayConfig] = []

    for comp in requirements.components:
        if _ref_prefix(comp.ref) != "K":
            continue

        com_net, output_net, coil_net, coil_voltage_net = _resolve_relay_pins(comp)

        driver_ref, gpio_net, polarity, relay_warnings = _trace_driver_chain(
            coil_net, comp.ref, net_to_refs, comps,
        )

        gpio_mcu_pin = _trace_gpio_to_mcu_pin(gpio_net, net_to_refs, comps)

        # Check for flyback diode
        if coil_net:
            coil_refs = net_to_refs.get(coil_net, set())
            d_refs = [r for r in coil_refs if _ref_prefix(r) == "D"]
            if not d_refs:
                relay_warnings.append(f"{comp.ref}: no flyback diode on coil net '{coil_net}'")

        description = comp.description or f"Relay {comp.ref}"
        warnings.extend(relay_warnings)

        relays.append(RelayConfig(
            relay_ref=comp.ref,
            description=description,
            polarity=polarity,
            gpio_net=gpio_net,
            gpio_mcu_pin=gpio_mcu_pin,
            com_net=com_net,
            output_net=output_net,
            driver_ref=driver_ref,
            coil_voltage_net=coil_voltage_net,
            warnings=tuple(relay_warnings),
        ))

    return tuple(sorted(relays, key=lambda r: r.relay_ref)), warnings


# ---------------------------------------------------------------------------
# ADC inference
# ---------------------------------------------------------------------------


def _parse_resistance(value: str) -> float | None:
    """Parse resistor value string to ohms. '100k' → 100000, '12k' → 12000."""
    value = value.strip().upper()
    m = re.match(r"^([\d.]+)\s*([KMR]?)$", value)
    if not m:
        return None
    num = float(m.group(1))
    suffix = m.group(2)
    if suffix == "K":
        return num * 1_000
    if suffix == "M":
        return num * 1_000_000
    return num


def _trace_voltage_divider(
    r_refs: list[str],
    adc_net: str,
    comps: dict[str, Component],
) -> tuple[float | None, float | None, str]:
    """Trace voltage divider resistors on an ADC net.

    Returns:
        (divider_ratio, max_input_voltage, input_net)
    """
    top_r_val: float | None = None
    bot_r_val: float | None = None
    input_net = adc_net

    for r_ref in r_refs:
        r_comp = comps.get(r_ref)
        if not r_comp:
            continue
        other_nets = [p.net for p in r_comp.pins if p.net and p.net != adc_net]
        if any(_is_gnd_net(n) for n in other_nets):
            bot_r_val = _parse_resistance(r_comp.value)
        else:
            top_r_val = _parse_resistance(r_comp.value)
            for n in other_nets:
                input_net = n
                if not _is_power_net(n):
                    break

    divider_ratio: float | None = None
    max_vin: float | None = None
    if top_r_val and bot_r_val:
        divider_ratio = round(bot_r_val / (top_r_val + bot_r_val), 4)
        max_vin = round(3.3 / divider_ratio, 1)

    return divider_ratio, max_vin, input_net


def _derive_adc_description(
    r_refs: list[str],
    comps: dict[str, Component],
    comp_ref: str,
    channel: int,
) -> str:
    """Derive a human-readable description for an ADC channel."""
    for r_ref in r_refs:
        r_comp = comps.get(r_ref)
        if r_comp and r_comp.description:
            m = re.search(r"ADC\s+(.+?)\s+(?:top|bottom)", r_comp.description)
            if m:
                return m.group(1)
    return f"{comp_ref} channel {channel}"


def _infer_adc_channels(
    requirements: ProjectRequirements,
) -> tuple[tuple[ADCChannelConfig, ...], list[str]]:
    """Detect ADC channels from ADS1115 components and their input networks."""
    comps = _comp_map(requirements)
    net_to_refs = _net_refs(requirements)
    warnings: list[str] = []
    channels: list[ADCChannelConfig] = []

    ain_pins: dict[str, int] = {"AIN0": 0, "AIN1": 1, "AIN2": 2, "AIN3": 3}

    for comp in requirements.components:
        if "ADS1115" not in comp.value.upper():
            continue

        addr_pin = next((p for p in comp.pins if p.name.upper() == "ADDR"), None)
        addr_net = addr_pin.net if addr_pin else None
        i2c_address = _ADS1115_ADDR_MAP.get(addr_net or "", 0x48)

        for pin in comp.pins:
            if pin.name.upper() not in ain_pins or not pin.net:
                continue
            channel = ain_pins[pin.name.upper()]
            adc_net = pin.net

            adc_refs = net_to_refs.get(adc_net, set())
            r_refs = sorted(r for r in adc_refs if _ref_prefix(r) == "R")

            divider_ratio, max_vin, input_net = _trace_voltage_divider(
                r_refs, adc_net, comps,
            )

            # Find connector on the input net
            input_refs = net_to_refs.get(input_net, set())
            j_refs = [r for r in input_refs if _ref_prefix(r) == "J"]
            connector_ref = sorted(j_refs)[0] if j_refs else ""

            description = _derive_adc_description(
                r_refs, comps, comp.ref, channel,
            )

            channels.append(ADCChannelConfig(
                adc_ref=comp.ref,
                channel=channel,
                adc_pin_net=adc_net,
                description=description,
                i2c_address=i2c_address,
                divider_ratio=divider_ratio,
                max_input_voltage=max_vin,
                input_net=input_net,
                connector_ref=connector_ref,
            ))

    return tuple(sorted(channels, key=lambda c: (c.adc_ref, c.channel))), warnings


# ---------------------------------------------------------------------------
# Bus inference
# ---------------------------------------------------------------------------


def _collect_mcu_bus_nets(
    requirements: ProjectRequirements,
) -> tuple[dict[str, str], dict[str, str]]:
    """Scan MCU pins for I2C and SPI signal nets.

    Returns:
        (i2c_nets, spi_nets) dicts mapping role name to net name.
    """
    from kicad_pipeline.models.requirements import PinFunction

    _FUNCTION_TO_BUS: dict[PinFunction, tuple[str, str]] = {
        PinFunction.I2C_SDA: ("i2c", "sda"),
        PinFunction.I2C_SCL: ("i2c", "scl"),
        PinFunction.SPI_CLK: ("spi", "clk"),
        PinFunction.SPI_MOSI: ("spi", "mosi"),
        PinFunction.SPI_MISO: ("spi", "miso"),
    }

    i2c_nets: dict[str, str] = {}
    spi_nets: dict[str, str] = {}
    bus_dicts = {"i2c": i2c_nets, "spi": spi_nets}

    for comp in requirements.components:
        if _ref_prefix(comp.ref) != "U" or not _is_mcu_component(comp):
            continue
        for pin in comp.pins:
            if not pin.function or not pin.net:
                continue
            mapping = _FUNCTION_TO_BUS.get(pin.function)
            if mapping:
                bus_dicts[mapping[0]][mapping[1]] = pin.net

    return i2c_nets, spi_nets


def _detect_i2c_bus(
    i2c_nets: dict[str, str],
    net_to_refs: dict[str, set[str]],
    comps: dict[str, Component],
) -> BusConfig | None:
    """Build an I2C BusConfig from detected SDA/SCL nets."""
    if "sda" not in i2c_nets or "scl" not in i2c_nets:
        return None

    sda_net = i2c_nets["sda"]
    scl_net = i2c_nets["scl"]

    sda_refs = net_to_refs.get(sda_net, set())
    devices: list[BusDevice] = []
    pullups: list[str] = []
    for ref in sorted(sda_refs):
        prefix = _ref_prefix(ref)
        bus_comp = comps.get(ref)
        if not bus_comp:
            continue
        if prefix == "R":
            pullups.append(ref)
            continue
        if prefix == "U":
            if _is_mcu_component(bus_comp):
                continue
            addr_pin = next(
                (p for p in bus_comp.pins if p.name.upper() == "ADDR"), None
            )
            addr_net = addr_pin.net if addr_pin else None
            address = _ADS1115_ADDR_MAP.get(addr_net or "")
            devices.append(BusDevice(ref=ref, value=bus_comp.value, address=address))

    scl_refs = net_to_refs.get(scl_net, set())
    for ref in sorted(scl_refs):
        if _ref_prefix(ref) == "R" and ref not in pullups:
            pullups.append(ref)

    return BusConfig(
        bus_type=BusType.I2C,
        bus_name="I2C",
        signal_nets=(sda_net, scl_net),
        devices=tuple(devices),
        pullup_refs=tuple(sorted(pullups)),
    )


def _detect_spi_bus(
    spi_nets: dict[str, str],
    net_to_refs: dict[str, set[str]],
    comps: dict[str, Component],
) -> BusConfig | None:
    """Build a SPI BusConfig from detected CLK/MOSI/MISO nets."""
    if "clk" not in spi_nets:
        return None

    signal_nets_list = [spi_nets["clk"]]
    if "mosi" in spi_nets:
        signal_nets_list.append(spi_nets["mosi"])
    if "miso" in spi_nets:
        signal_nets_list.append(spi_nets["miso"])

    clk_refs = net_to_refs.get(spi_nets["clk"], set())
    devices_spi: list[BusDevice] = []
    for ref in sorted(clk_refs):
        spi_comp = comps.get(ref)
        if not spi_comp or _ref_prefix(ref) != "U":
            continue
        if _is_mcu_component(spi_comp):
            continue
        cs_net: str | None = None
        for pin in spi_comp.pins:
            if pin.name.upper() in ("CS", "~CS", "SS", "~SS", "SCS", "SCSN"):
                cs_net = pin.net
                break
        devices_spi.append(BusDevice(ref=ref, value=spi_comp.value, cs_net=cs_net))

    if not devices_spi:
        return None

    return BusConfig(
        bus_type=BusType.SPI,
        bus_name="SPI",
        signal_nets=tuple(signal_nets_list),
        devices=tuple(devices_spi),
    )


def _infer_buses(requirements: ProjectRequirements) -> tuple[tuple[BusConfig, ...], list[str]]:
    """Detect I2C and SPI buses from MCU pin functions and shared nets."""
    comps = _comp_map(requirements)
    net_to_refs = _net_refs(requirements)
    warnings: list[str] = []
    buses: list[BusConfig] = []

    i2c_nets, spi_nets = _collect_mcu_bus_nets(requirements)

    i2c_bus = _detect_i2c_bus(i2c_nets, net_to_refs, comps)
    if i2c_bus:
        buses.append(i2c_bus)

    spi_bus = _detect_spi_bus(spi_nets, net_to_refs, comps)
    if spi_bus:
        buses.append(spi_bus)

    return tuple(buses), warnings


# ---------------------------------------------------------------------------
# Power rail inference
# ---------------------------------------------------------------------------


def _infer_power_rails(
    requirements: ProjectRequirements,
) -> tuple[tuple[PowerRailConfig, ...], list[str]]:
    """Detect power rails from components with POWER_OUT pins."""
    from kicad_pipeline.models.requirements import PinType

    net_to_refs = _net_refs(requirements)
    warnings: list[str] = []
    rails: list[PowerRailConfig] = []
    seen_rails: set[str] = set()

    for comp in requirements.components:
        for pin in comp.pins:
            if pin.pin_type != PinType.POWER_OUT or not pin.net:
                continue
            rail_name = pin.net
            if rail_name in seen_rails:
                continue
            seen_rails.add(rail_name)

            # Parse voltage from net name
            voltage = _parse_voltage(rail_name)
            consumers = tuple(
                sorted(
                    r for r in net_to_refs.get(rail_name, set()) if r != comp.ref
                )
            )
            rails.append(PowerRailConfig(
                name=rail_name,
                voltage=voltage,
                source_ref=comp.ref,
                consumers=consumers,
            ))

    return tuple(sorted(rails, key=lambda r: r.name)), warnings


def _parse_voltage(net_name: str) -> float | None:
    """Extract voltage from a net name like '+3V3', '+5V', 'VIN'."""
    name = net_name.upper().strip()
    # +3V3 → 3.3, +5V → 5.0, +1V8 → 1.8
    m = re.match(r"^\+?(\d+)V(\d*)$", name)
    if m:
        integer = m.group(1)
        frac = m.group(2)
        if frac:
            return float(f"{integer}.{frac}")
        return float(integer)
    return None


# ---------------------------------------------------------------------------
# Connector inference
# ---------------------------------------------------------------------------


def _infer_connectors(
    requirements: ProjectRequirements,
) -> tuple[tuple[ConnectorPinout, ...], list[str]]:
    """Build pin-by-pin connector descriptions."""
    warnings: list[str] = []
    connectors: list[ConnectorPinout] = []

    for comp in requirements.components:
        if _ref_prefix(comp.ref) != "J":
            continue
        pins: list[ConnectorPin] = []
        for pin in comp.pins:
            net = pin.net or "NC"
            # Derive function from net name and pin name
            func = pin.name if pin.name != "~" else net
            pins.append(ConnectorPin(
                pin_number=pin.number,
                net=net,
                function=func,
            ))
        connectors.append(ConnectorPinout(
            ref=comp.ref,
            value=comp.value,
            pins=tuple(sorted(pins, key=lambda p: p.pin_number)),
        ))

    return tuple(sorted(connectors, key=lambda c: c.ref)), warnings


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------


def generate_board_config(requirements: ProjectRequirements) -> BoardConfig:
    """Generate a complete board configuration from project requirements.

    Traces nets to infer relay polarity, ADC divider ratios, bus members,
    power rails, and connector pinouts.

    Args:
        requirements: Fully-specified project requirements.

    Returns:
        A BoardConfig with all inferred control information.
    """
    all_warnings: list[str] = []

    relays, w = _infer_relays(requirements)
    all_warnings.extend(w)

    adc_channels, w = _infer_adc_channels(requirements)
    all_warnings.extend(w)

    buses, w = _infer_buses(requirements)
    all_warnings.extend(w)

    power_rails, w = _infer_power_rails(requirements)
    all_warnings.extend(w)

    connectors, w = _infer_connectors(requirements)
    all_warnings.extend(w)

    return BoardConfig(
        project_name=requirements.project.name,
        relays=relays,
        adc_channels=adc_channels,
        buses=buses,
        power_rails=power_rails,
        connectors=connectors,
        warnings=tuple(all_warnings),
    )
