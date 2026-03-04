"""Data models for PCB project requirements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PinType(Enum):
    """Electrical type of a component pin."""

    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"
    PASSIVE = "passive"
    POWER_IN = "power_in"
    POWER_OUT = "power_out"
    OPEN_COLLECTOR = "open_collector"
    NO_CONNECT = "no_connect"


class PinFunction(Enum):
    """Logical function of a component pin."""

    GPIO = "gpio"
    ADC = "adc"
    DAC = "dac"
    SPI_CLK = "spi_clk"
    SPI_MOSI = "spi_mosi"
    SPI_MISO = "spi_miso"
    SPI_CS = "spi_cs"
    I2C_SDA = "i2c_sda"
    I2C_SCL = "i2c_scl"
    UART_TX = "uart_tx"
    UART_RX = "uart_rx"
    USB_DP = "usb_dp"
    USB_DM = "usb_dm"
    PWM = "pwm"
    RESET = "reset"
    BOOT = "boot"
    ENABLE = "enable"
    INTERRUPT = "interrupt"
    VCC = "vcc"
    GND = "gnd"
    NC = "nc"
    ANALOG_IN = "analog_in"
    ANALOG_OUT = "analog_out"


@dataclass(frozen=True)
class Pin:
    """A component pin with electrical and functional metadata."""

    number: str  # "1", "A1", "PA3"
    name: str  # "VCC", "GPIO4", "~"
    pin_type: PinType
    function: PinFunction | None = None
    net: str | None = None


@dataclass(frozen=True)
class Component:
    """A fully-specified electronic component."""

    ref: str  # "R1", "U2", "C5"
    value: str  # "10k", "ESP32-S3-WROOM-1", "100nF"
    footprint: str  # "R_0805", "ESP32-S3-WROOM-1"
    lcsc: str | None = None
    description: str | None = None
    datasheet: str | None = None
    pins: tuple[Pin, ...] = ()

    def get_pin(self, number: str) -> Pin | None:
        """Return pin by number, or None if not found."""
        for p in self.pins:
            if p.number == number:
                return p
        return None


@dataclass(frozen=True)
class NetConnection:
    """A single endpoint of a net: which component pin."""

    ref: str  # "R1"
    pin: str  # "1"


@dataclass(frozen=True)
class Net:
    """An electrical net connecting one or more component pins."""

    name: str  # "+3V3", "GND", "SPI_CLK"
    connections: tuple[NetConnection, ...]


@dataclass(frozen=True)
class PowerRail:
    """A power supply rail with current budget."""

    name: str  # "+3V3", "+5V", "GND"
    voltage: float  # volts
    current_ma: float  # milliamps total
    source_ref: str  # "U1" (LDO or regulator ref)


@dataclass(frozen=True)
class PowerBudget:
    """Complete power budget for the design."""

    rails: tuple[PowerRail, ...]
    total_current_ma: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PinAssignment:
    """A single MCU pin assignment."""

    mcu_ref: str  # "U1"
    pin_number: str  # "GPIO4", "IO4"
    pin_name: str  # "IO4"
    function: PinFunction
    net: str  # net name this pin connects to
    notes: str | None = None


@dataclass(frozen=True)
class MCUPinMap:
    """Complete pin map for the MCU."""

    mcu_ref: str
    assignments: tuple[PinAssignment, ...]
    unassigned_gpio: tuple[str, ...]


@dataclass(frozen=True)
class MechanicalConstraints:
    """Physical and mechanical constraints for the PCB."""

    board_width_mm: float
    board_height_mm: float
    enclosure: str | None = None
    mounting_hole_diameter_mm: float = 3.2
    mounting_hole_positions: tuple[tuple[float, float], ...] = ()  # (x,y) pairs
    notes: str | None = None
    board_template: str | None = None


@dataclass(frozen=True)
class FeatureBlock:
    """A functional feature block (e.g. Ethernet, Power, USB)."""

    name: str  # "Ethernet", "Power", "USB"
    description: str
    components: tuple[str, ...]  # component refs in this block
    nets: tuple[str, ...]  # net names in this block
    subcircuits: tuple[str, ...]  # subcircuit types used


@dataclass(frozen=True)
class Recommendation:
    """An AI recommendation or warning about the design."""

    severity: str  # "info", "warning", "error"
    category: str  # "power", "signal", "mechanical", "bom"
    message: str
    affected_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectInfo:
    """Top-level project metadata."""

    name: str
    author: str | None = None
    revision: str = "v0.1"
    description: str | None = None


@dataclass(frozen=True)
class ProjectRequirements:
    """Complete, validated requirements document for a PCB project."""

    project: ProjectInfo
    features: tuple[FeatureBlock, ...]
    components: tuple[Component, ...]
    nets: tuple[Net, ...]
    pin_map: MCUPinMap | None = None
    power_budget: PowerBudget | None = None
    mechanical: MechanicalConstraints | None = None
    recommendations: tuple[Recommendation, ...] = ()

    def get_component(self, ref: str) -> Component | None:
        """Return component by ref, or None if not found."""
        for c in self.components:
            if c.ref == ref:
                return c
        return None

    def get_net(self, name: str) -> Net | None:
        """Return net by name, or None if not found."""
        for n in self.nets:
            if n.name == name:
                return n
        return None
