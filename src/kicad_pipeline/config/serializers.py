"""Serialize BoardConfig to JSON and Markdown formats."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.board_config import (
        ADCChannelConfig,
        BoardConfig,
        BusConfig,
        ConnectorPinout,
        PowerRailConfig,
        RelayConfig,
    )


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _relay_action(relay: RelayConfig) -> str:
    """Build a human-readable action string for a relay."""
    from kicad_pipeline.models.board_config import RelayPolarity

    if relay.polarity == RelayPolarity.ACTIVE_HIGH:
        drive = "HIGH"
    elif relay.polarity == RelayPolarity.ACTIVE_LOW:
        drive = "LOW"
    else:
        return f"Polarity unknown — verify driver topology manually"

    return (
        f"GPIO {drive} on {relay.gpio_net} energizes relay, "
        f"closes NO contact, connects {relay.com_net} to {relay.output_net}"
    )


def _relay_to_dict(relay: RelayConfig) -> dict[str, object]:
    """Convert a RelayConfig to a JSON-serialisable dict."""
    return {
        "ref": relay.relay_ref,
        "description": relay.description,
        "polarity": relay.polarity.value,
        "gpio_net": relay.gpio_net,
        "gpio_mcu_pin": relay.gpio_mcu_pin,
        "com_net": relay.com_net,
        "output_net": relay.output_net,
        "driver_ref": relay.driver_ref,
        "coil_voltage_net": relay.coil_voltage_net,
        "action": _relay_action(relay),
        "warnings": list(relay.warnings),
    }


def _adc_to_dict(ch: ADCChannelConfig) -> dict[str, object]:
    """Convert an ADCChannelConfig to a JSON-serialisable dict."""
    return {
        "adc_ref": ch.adc_ref,
        "channel": ch.channel,
        "adc_pin_net": ch.adc_pin_net,
        "description": ch.description,
        "i2c_address": f"0x{ch.i2c_address:02X}",
        "divider_ratio": ch.divider_ratio,
        "max_input_voltage": ch.max_input_voltage,
        "input_net": ch.input_net,
        "connector_ref": ch.connector_ref,
    }


def _bus_to_dict(bus: BusConfig) -> dict[str, object]:
    """Convert a BusConfig to a JSON-serialisable dict."""
    devices = []
    for d in bus.devices:
        dev: dict[str, object] = {"ref": d.ref, "value": d.value}
        if d.address is not None:
            dev["address"] = f"0x{d.address:02X}"
        if d.cs_net is not None:
            dev["cs_net"] = d.cs_net
        devices.append(dev)
    return {
        "bus_type": bus.bus_type.value,
        "bus_name": bus.bus_name,
        "signal_nets": list(bus.signal_nets),
        "devices": devices,
        "pullup_refs": list(bus.pullup_refs),
    }


def _rail_to_dict(rail: PowerRailConfig) -> dict[str, object]:
    """Convert a PowerRailConfig to a JSON-serialisable dict."""
    return {
        "name": rail.name,
        "voltage": rail.voltage,
        "source_ref": rail.source_ref,
        "consumers": list(rail.consumers),
    }


def _connector_to_dict(conn: ConnectorPinout) -> dict[str, object]:
    """Convert a ConnectorPinout to a JSON-serialisable dict."""
    return {
        "ref": conn.ref,
        "value": conn.value,
        "pins": [
            {
                "pin": p.pin_number,
                "net": p.net,
                "function": p.function,
            }
            for p in conn.pins
        ],
    }


def board_config_to_json(config: BoardConfig, indent: int = 2) -> str:
    """Serialise a BoardConfig to a JSON string.

    Args:
        config: The board configuration to serialise.
        indent: JSON indentation level.

    Returns:
        A JSON string.
    """
    data: dict[str, object] = {
        "project_name": config.project_name,
        "relays": [_relay_to_dict(r) for r in config.relays],
        "adc_channels": [_adc_to_dict(ch) for ch in config.adc_channels],
        "buses": [_bus_to_dict(b) for b in config.buses],
        "power_rails": [_rail_to_dict(r) for r in config.power_rails],
        "connectors": [_connector_to_dict(c) for c in config.connectors],
        "warnings": list(config.warnings),
    }
    return json.dumps(data, indent=indent)


# ---------------------------------------------------------------------------
# Markdown serialisation
# ---------------------------------------------------------------------------


def _md_relay_section(config: BoardConfig) -> list[str]:
    """Build Markdown lines for relay control section."""
    if not config.relays:
        return []
    lines = [
        "## Relay Control",
        "",
        "| Ref | Description | Polarity | GPIO Net | MCU Pin "
        "| COM Net | Output Net | Driver | Coil Supply |",
        "| --- | ----------- | -------- | -------- | ------- "
        "| ------- | ---------- | ------ | ----------- |",
    ]
    for r in config.relays:
        lines.append(
            f"| {r.relay_ref} | {r.description} | {r.polarity.value} "
            f"| {r.gpio_net} | {r.gpio_mcu_pin} | {r.com_net} "
            f"| {r.output_net} | {r.driver_ref} | {r.coil_voltage_net} |"
        )
    lines.append("")
    for r in config.relays:
        lines.append(f"**{r.relay_ref}**: {_relay_action(r)}")
        lines.append("")
    return lines


def _md_adc_section(config: BoardConfig) -> list[str]:
    """Build Markdown lines for ADC channels section."""
    if not config.adc_channels:
        return []
    lines = [
        "## ADC Channels",
        "",
        "| ADC | Ch | I2C Addr | Net | Description "
        "| Divider | Max Vin | Input Net | Connector |",
        "| --- | -- | -------- | --- | ----------- "
        "| ------- | ------- | --------- | --------- |",
    ]
    for ch in config.adc_channels:
        ratio = f"{ch.divider_ratio:.4f}" if ch.divider_ratio else "N/A"
        max_v = f"{ch.max_input_voltage:.1f}V" if ch.max_input_voltage else "N/A"
        lines.append(
            f"| {ch.adc_ref} | {ch.channel} | 0x{ch.i2c_address:02X} "
            f"| {ch.adc_pin_net} | {ch.description} | {ratio} "
            f"| {max_v} | {ch.input_net} | {ch.connector_ref} |"
        )
    lines.append("")
    return lines


def _md_bus_device_addr(d: object) -> str:
    """Format bus device address or CS net for display."""
    if d.address is not None:
        return f"0x{d.address:02X}"
    if d.cs_net is not None:
        return d.cs_net
    return ""


def _md_buses_section(config: BoardConfig) -> list[str]:
    """Build Markdown lines for communication buses section."""
    if not config.buses:
        return []
    lines = ["## Communication Buses", ""]
    for bus in config.buses:
        lines.append(f"### {bus.bus_name} ({bus.bus_type.value.upper()})")
        lines.append("")
        lines.append(f"**Signals**: {', '.join(bus.signal_nets)}")
        lines.append("")
        if bus.pullup_refs:
            lines.append(f"**Pull-ups**: {', '.join(bus.pullup_refs)}")
            lines.append("")
        if bus.devices:
            lines.append("| Ref | Value | Address / CS |")
            lines.append("| --- | ----- | ------------ |")
            for d in bus.devices:
                lines.append(f"| {d.ref} | {d.value} | {_md_bus_device_addr(d)} |")
            lines.append("")
    return lines


def _md_power_section(config: BoardConfig) -> list[str]:
    """Build Markdown lines for power rails section."""
    if not config.power_rails:
        return []
    lines = [
        "## Power Rails",
        "",
        "| Rail | Voltage | Source | Consumers |",
        "| ---- | ------- | ------ | --------- |",
    ]
    for rail in config.power_rails:
        v = f"{rail.voltage}V" if rail.voltage else "?"
        consumers = ", ".join(rail.consumers[:8])
        if len(rail.consumers) > 8:
            consumers += f" (+{len(rail.consumers) - 8} more)"
        lines.append(
            f"| {rail.name} | {v} | {rail.source_ref} | {consumers} |"
        )
    lines.append("")
    return lines


def _md_connectors_section(config: BoardConfig) -> list[str]:
    """Build Markdown lines for connectors section."""
    if not config.connectors:
        return []
    lines = ["## Connectors", ""]
    for conn in config.connectors:
        lines.extend([
            f"### {conn.ref} ({conn.value})",
            "",
            "| Pin | Net | Function |",
            "| --- | --- | -------- |",
        ])
        for p in conn.pins:
            lines.append(f"| {p.pin_number} | {p.net} | {p.function} |")
        lines.append("")
    return lines


def board_config_to_markdown(config: BoardConfig) -> str:
    """Serialise a BoardConfig to a human-readable Markdown document.

    Args:
        config: The board configuration to serialise.

    Returns:
        A Markdown string with section tables.
    """
    lines: list[str] = [f"# Board Configuration: {config.project_name}", ""]
    lines.extend(_md_relay_section(config))
    lines.extend(_md_adc_section(config))
    lines.extend(_md_buses_section(config))
    lines.extend(_md_power_section(config))
    lines.extend(_md_connectors_section(config))

    if config.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in config.warnings:
            lines.append(f"- **WARNING**: {w}")
        lines.append("")

    return "\n".join(lines)
