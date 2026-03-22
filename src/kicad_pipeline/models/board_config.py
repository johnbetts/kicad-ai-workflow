"""Data models for machine-readable board configuration.

Describes relay control logic, ADC channel mappings, bus topologies,
power rails, and connector pinouts — everything a firmware agent needs
to know about how to *control* the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RelayPolarity(Enum):
    """Whether a relay is energised by driving its GPIO HIGH or LOW."""

    ACTIVE_HIGH = "active_high"
    ACTIVE_LOW = "active_low"
    UNKNOWN = "unknown"


class BusType(Enum):
    """Communication bus protocol."""

    I2C = "i2c"
    SPI = "spi"
    UART = "uart"


# ---------------------------------------------------------------------------
# Relay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelayConfig:
    """Control description for a single relay."""

    relay_ref: str
    description: str
    polarity: RelayPolarity
    gpio_net: str
    gpio_mcu_pin: str
    com_net: str
    output_net: str
    driver_ref: str
    coil_voltage_net: str
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# ADC
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ADCChannelConfig:
    """Mapping for one ADC input channel."""

    adc_ref: str
    channel: int
    adc_pin_net: str
    description: str
    i2c_address: int
    divider_ratio: float | None
    max_input_voltage: float | None
    input_net: str
    connector_ref: str


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BusDevice:
    """A device on a communication bus."""

    ref: str
    value: str
    address: int | None = None  # I2C address
    cs_net: str | None = None  # SPI chip-select net


@dataclass(frozen=True)
class BusConfig:
    """A communication bus with its devices."""

    bus_type: BusType
    bus_name: str
    signal_nets: tuple[str, ...]
    devices: tuple[BusDevice, ...]
    pullup_refs: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerRailConfig:
    """A power supply rail."""

    name: str
    voltage: float | None
    source_ref: str
    consumers: tuple[str, ...]


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectorPin:
    """One pin of a connector."""

    pin_number: str
    net: str
    function: str


@dataclass(frozen=True)
class ConnectorPinout:
    """Pin-by-pin description of a connector."""

    ref: str
    value: str
    pins: tuple[ConnectorPin, ...]


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardConfig:
    """Complete machine-readable board configuration."""

    project_name: str
    relays: tuple[RelayConfig, ...] = ()
    adc_channels: tuple[ADCChannelConfig, ...] = ()
    buses: tuple[BusConfig, ...] = ()
    power_rails: tuple[PowerRailConfig, ...] = ()
    connectors: tuple[ConnectorPinout, ...] = ()
    warnings: tuple[str, ...] = ()
