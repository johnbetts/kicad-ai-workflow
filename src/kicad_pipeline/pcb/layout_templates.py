"""Layout templates for ICs and subcircuit patterns.

Provides data-driven recipes for component arrangement:

- **IC Templates**: Pin-group maps for specific ICs (ESP32, ADS1115, etc.)
  defining which pin functions are on which side at 0-degree rotation.
- **Subcircuit Templates**: Physical layout recipes for common subcircuit
  patterns (voltage divider, buck converter, relay driver, etc.) specifying
  component roles, relative offsets, and pad-facing relationships.

Templates are versioned — customized variants maintain a separate version
string so they don't silently overwrite the base template.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from kicad_pipeline.pcb.pin_map import CardinalSide, rotate_side

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint
    from kicad_pipeline.optimization.functional_grouper import SubCircuitType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PinFunction(Enum):
    """Functional classification of IC pins."""

    SPI = "spi"
    I2C = "i2c"
    UART = "uart"
    ADC = "adc"
    GPIO = "gpio"
    POWER = "power"
    GROUND = "ground"
    BOOT = "boot"
    RESET = "reset"
    ANTENNA = "antenna"
    ETHERNET_PHY = "ethernet_phy"
    ETHERNET_MAC = "ethernet_mac"
    USB = "usb"
    OSCILLATOR = "oscillator"
    DEBUG = "debug"
    ANALOG_IN = "analog_in"
    ANALOG_OUT = "analog_out"
    PWM = "pwm"
    ENABLE = "enable"
    FEEDBACK = "feedback"


class ComponentRole(Enum):
    """Role of a component within a subcircuit pattern."""

    INPUT = "input"
    OUTPUT = "output"
    SERIES = "series"
    SHUNT = "shunt"
    ANCHOR = "anchor"
    SWITCH = "switch"
    INDICATOR = "indicator"


# ---------------------------------------------------------------------------
# IC Templates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinGroup:
    """Named group of pins on a specific side at 0-degree rotation."""

    name: str
    side: CardinalSide
    pin_numbers: tuple[str, ...]
    functions: tuple[PinFunction, ...]


@dataclass(frozen=True)
class ICTemplate:
    """Pin-side map for a specific IC or IC family.

    Attributes:
        ic_pattern: Glob pattern matched against component value
            (e.g., ``"ESP32*WROOM*"``).
        pin_groups: Named groups of pins with side/function classification.
        antenna_side: Side with antenna/RF output at 0-degree rotation.
        thermal_pad: Pad number of thermal/exposed pad, if any.
        version: Template version string. Increment when customizing
            a base template to maintain traceability.
    """

    ic_pattern: str
    pin_groups: tuple[PinGroup, ...]
    antenna_side: CardinalSide | None = None
    thermal_pad: str | None = None
    version: str = "1.0"

    def groups_on_side(
        self,
        side: CardinalSide,
        rotation: float = 0.0,
    ) -> tuple[PinGroup, ...]:
        """Return pin groups on the given side after rotation.

        Args:
            side: The target side (in board coordinates).
            rotation: Footprint rotation in degrees.

        Returns:
            Tuple of PinGroup entries whose rotated side matches.
        """
        results: list[PinGroup] = []
        for pg in self.pin_groups:
            rotated = rotate_side(pg.side, rotation)
            if rotated == side:
                results.append(pg)
        return tuple(results)

    def preferred_side_for_function(
        self,
        func: PinFunction,
        rotation: float = 0.0,
    ) -> CardinalSide | None:
        """Return the side where a given function is located after rotation.

        Args:
            func: The pin function to look up.
            rotation: Footprint rotation in degrees.

        Returns:
            The rotated CardinalSide, or None if the function is not found.
        """
        for pg in self.pin_groups:
            if func in pg.functions:
                return rotate_side(pg.side, rotation)
        return None


# ---------------------------------------------------------------------------
# Subcircuit Layout Templates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateSlot:
    """One component position in the subcircuit layout template.

    Attributes:
        role: Functional role (INPUT, OUTPUT, SERIES, SHUNT, etc.).
        ref_pattern: Descriptive name for matching (e.g., "R_top", "C_in").
        offset_x: X offset from anchor in mm.
        offset_y: Y offset from anchor in mm.
        rotation: Preferred rotation in degrees.
        pad_face_toward: ref_pattern of the component whose pad this
            component's connected pad should face.
    """

    role: ComponentRole
    ref_pattern: str
    offset_x: float
    offset_y: float
    rotation: float = 0.0
    pad_face_toward: str = ""


@dataclass(frozen=True)
class SubcircuitTemplate:
    """Physical layout recipe for a subcircuit pattern.

    Attributes:
        circuit_type_name: String matching ``SubCircuitType.value``.
        name: Human-readable name.
        flow_direction: Signal flow orientation
            (``"top_to_bottom"``, ``"left_to_right"``, ``"radial"``).
        slots: Component positions relative to the anchor.
        notes: Design notes for the template.
        version: Template version string for customized variants.
    """

    circuit_type_name: str
    name: str
    flow_direction: str
    slots: tuple[TemplateSlot, ...]
    notes: str = ""
    version: str = "1.0"


# ---------------------------------------------------------------------------
# Template Registry
# ---------------------------------------------------------------------------

_IC_TEMPLATES: dict[str, ICTemplate] = {}
_SUBCIRCUIT_TEMPLATES: dict[str, SubcircuitTemplate] = {}


def register_ic_template(template: ICTemplate) -> None:
    """Register an IC template in the global registry.

    Overwrites any existing template with the same pattern.

    Args:
        template: The IC template to register.
    """
    _IC_TEMPLATES[template.ic_pattern] = template
    logger.debug(
        "Registered IC template: %s (v%s)", template.ic_pattern, template.version,
    )


def register_subcircuit_template(template: SubcircuitTemplate) -> None:
    """Register a subcircuit template in the global registry.

    Overwrites any existing template with the same circuit type name.

    Args:
        template: The subcircuit template to register.
    """
    _SUBCIRCUIT_TEMPLATES[template.circuit_type_name] = template
    logger.debug(
        "Registered subcircuit template: %s (v%s)",
        template.circuit_type_name, template.version,
    )


def get_ic_template(component_value: str) -> ICTemplate | None:
    """Look up an IC template by component value.

    Uses glob-style pattern matching against registered templates.

    Args:
        component_value: The component value string (e.g., "ESP32-S3-WROOM-1").

    Returns:
        The matching ICTemplate, or None if no match.
    """
    for pattern, template in _IC_TEMPLATES.items():
        if fnmatch.fnmatch(component_value, pattern):
            return template
    return None


def get_subcircuit_template(circuit_type_name: str) -> SubcircuitTemplate | None:
    """Look up a subcircuit template by circuit type name.

    Args:
        circuit_type_name: The SubCircuitType value string
            (e.g., "voltage_divider").

    Returns:
        The matching SubcircuitTemplate, or None if not registered.
    """
    return _SUBCIRCUIT_TEMPLATES.get(circuit_type_name)


def get_subcircuit_template_by_type(
    circuit_type: SubCircuitType,
) -> SubcircuitTemplate | None:
    """Look up a subcircuit template by SubCircuitType enum.

    Args:
        circuit_type: The SubCircuitType enum value.

    Returns:
        The matching SubcircuitTemplate, or None if not registered.
    """
    return _SUBCIRCUIT_TEMPLATES.get(circuit_type.value)


# ---------------------------------------------------------------------------
# Auto-generation from footprint geometry
# ---------------------------------------------------------------------------


def auto_generate_ic_template(
    footprint: Footprint,
    value_pattern: str | None = None,
) -> ICTemplate:
    """Generate a basic IC template from footprint pad geometry.

    Creates pin groups based on pad positions (left/right/top/bottom)
    without any function assignment. Useful as a baseline that can be
    manually refined.

    Args:
        footprint: The footprint to analyze.
        value_pattern: Glob pattern for matching. Defaults to the
            footprint's value with a wildcard suffix.

    Returns:
        An auto-generated ICTemplate.
    """
    from kicad_pipeline.pcb.pin_map import _footprint_pad_extent, classify_pad_side

    pattern = value_pattern or f"{footprint.value}*"
    half_w, half_h = _footprint_pad_extent(footprint.pads)

    # Group pads by side
    side_pads: dict[CardinalSide, list[str]] = {}
    for pad in footprint.pads:
        side = classify_pad_side(
            pad.position.x, pad.position.y, half_w, half_h,
        )
        side_pads.setdefault(side, []).append(pad.number)

    pin_groups: list[PinGroup] = []
    for side, pad_nums in side_pads.items():
        if side == CardinalSide.CENTER:
            continue  # Skip thermal pads for pin groups
        pin_groups.append(PinGroup(
            name=f"auto_{side.value}",
            side=side,
            pin_numbers=tuple(pad_nums),
            functions=(),  # No function assignment in auto mode
        ))

    return ICTemplate(
        ic_pattern=pattern,
        pin_groups=tuple(pin_groups),
        version="auto-1.0",
    )


# ---------------------------------------------------------------------------
# Built-in IC Templates
# ---------------------------------------------------------------------------

# ESP32-S3-WROOM-1 (castellation module, 41 pins)
# Datasheet Figure 3-1 (top view, antenna at top/north):
#   Left (1-14): GND, 3V3, EN, IO4-IO7, IO15-IO18, IO8, IO19, IO20
#   Bottom (15-26): IO3, IO46, IO9-IO14, IO21, IO47, IO48, IO45
#   Right (27-40): IO0, IO35-IO42, RXD0, TXD0, IO2, IO1, GND
#   Center (41): GND exposed pad
_ESP32_S3_WROOM = ICTemplate(
    ic_pattern="ESP32*WROOM*",
    pin_groups=(
        PinGroup(
            name="PWR_EN_GPIO",
            side=CardinalSide.WEST,
            pin_numbers=(
                "1", "2", "3", "4", "5", "6", "7",
                "8", "9", "10", "11", "12", "13", "14",
            ),
            functions=(PinFunction.POWER, PinFunction.BOOT, PinFunction.GPIO),
        ),
        PinGroup(
            name="SPI_GPIO",
            side=CardinalSide.SOUTH,
            pin_numbers=(
                "15", "16", "17", "18", "19", "20",
                "21", "22", "23", "24", "25", "26",
            ),
            functions=(PinFunction.SPI, PinFunction.GPIO),
        ),
        PinGroup(
            name="GPIO_UART",
            side=CardinalSide.EAST,
            pin_numbers=(
                "27", "28", "29", "30", "31", "32", "33",
                "34", "35", "36", "37", "38", "39", "40",
            ),
            functions=(PinFunction.GPIO, PinFunction.UART, PinFunction.ADC, PinFunction.PWM),
        ),
    ),
    antenna_side=CardinalSide.NORTH,
    thermal_pad="41",
    version="2.0",
)

# ADS1115 (MSOP-10)
_ADS1115 = ICTemplate(
    ic_pattern="ADS1115*",
    pin_groups=(
        PinGroup(
            name="ANALOG_IN",
            side=CardinalSide.WEST,
            pin_numbers=("1", "2", "3", "4", "5"),
            functions=(PinFunction.ANALOG_IN, PinFunction.POWER),
        ),
        PinGroup(
            name="DIGITAL",
            side=CardinalSide.EAST,
            pin_numbers=("6", "7", "8", "9", "10"),
            functions=(PinFunction.I2C, PinFunction.GPIO, PinFunction.POWER),
        ),
    ),
    version="1.0",
)

# W5500 (QFN-48)
_W5500 = ICTemplate(
    ic_pattern="W5500*",
    pin_groups=(
        PinGroup(
            name="SPI",
            side=CardinalSide.WEST,
            pin_numbers=("1", "2", "3", "4", "5", "6"),
            functions=(PinFunction.SPI, PinFunction.RESET),
        ),
        PinGroup(
            name="ETHERNET_PHY",
            side=CardinalSide.EAST,
            pin_numbers=("31", "32", "33", "34", "35", "36", "37", "38"),
            functions=(PinFunction.ETHERNET_PHY,),
        ),
    ),
    version="1.0",
)

# LAN8720A (QFN-24)
_LAN8720A = ICTemplate(
    ic_pattern="LAN8720*",
    pin_groups=(
        PinGroup(
            name="RMII",
            side=CardinalSide.WEST,
            pin_numbers=("1", "2", "3", "4", "5", "6"),
            functions=(PinFunction.ETHERNET_MAC,),
        ),
        PinGroup(
            name="MAGNETICS",
            side=CardinalSide.EAST,
            pin_numbers=("19", "20", "21", "22", "23", "24"),
            functions=(PinFunction.ETHERNET_PHY,),
        ),
    ),
    version="1.0",
)


# ---------------------------------------------------------------------------
# Built-in Subcircuit Templates
# ---------------------------------------------------------------------------

_VOLTAGE_DIVIDER = SubcircuitTemplate(
    circuit_type_name="voltage_divider",
    name="Voltage Divider",
    flow_direction="top_to_bottom",
    slots=(
        TemplateSlot(
            role=ComponentRole.SERIES,
            ref_pattern="R_top",
            offset_x=0.0,
            offset_y=-2.5,
            pad_face_toward="R_bot",
        ),
        TemplateSlot(
            role=ComponentRole.SERIES,
            ref_pattern="R_bot",
            offset_x=0.0,
            offset_y=2.5,
            pad_face_toward="R_top",
        ),
    ),
    notes="Connected pads face each other vertically.",
    version="1.0",
)

_ADC_CHANNEL = SubcircuitTemplate(
    circuit_type_name="adc_channel",
    name="ADC Channel",
    flow_direction="top_to_bottom",
    slots=(
        TemplateSlot(
            role=ComponentRole.INPUT,
            ref_pattern="connector",
            offset_x=0.0,
            offset_y=-10.0,
        ),
        TemplateSlot(
            role=ComponentRole.SERIES,
            ref_pattern="R_top",
            offset_x=0.0,
            offset_y=-6.0,
            pad_face_toward="R_bot",
        ),
        TemplateSlot(
            role=ComponentRole.SERIES,
            ref_pattern="R_bot",
            offset_x=0.0,
            offset_y=-2.0,
            pad_face_toward="R_top",
        ),
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="D_clamp",
            offset_x=2.5,
            offset_y=-2.0,
            rotation=90.0,
        ),
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="C_filter",
            offset_x=-2.5,
            offset_y=-2.0,
            rotation=90.0,
        ),
        TemplateSlot(
            role=ComponentRole.OUTPUT,
            ref_pattern="ADC_pin",
            offset_x=0.0,
            offset_y=2.0,
        ),
    ),
    notes="Vertical strip: connector -> R_top -> R_bot with D_clamp and C_filter as shunts.",
    version="1.0",
)

_BUCK_CONVERTER = SubcircuitTemplate(
    circuit_type_name="buck_converter",
    name="Buck Converter",
    flow_direction="left_to_right",
    slots=(
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="C_in",
            offset_x=-6.0,
            offset_y=0.0,
        ),
        TemplateSlot(
            role=ComponentRole.ANCHOR,
            ref_pattern="IC",
            offset_x=0.0,
            offset_y=0.0,
        ),
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="C_bst",
            offset_x=0.0,
            offset_y=-3.0,
        ),
        TemplateSlot(
            role=ComponentRole.SERIES,
            ref_pattern="L",
            offset_x=6.0,
            offset_y=0.0,
        ),
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="C_out",
            offset_x=10.0,
            offset_y=0.0,
        ),
        TemplateSlot(
            role=ComponentRole.SERIES,
            ref_pattern="R_fb_top",
            offset_x=10.0,
            offset_y=3.0,
            pad_face_toward="R_fb_bot",
        ),
        TemplateSlot(
            role=ComponentRole.SERIES,
            ref_pattern="R_fb_bot",
            offset_x=10.0,
            offset_y=6.0,
            pad_face_toward="R_fb_top",
        ),
    ),
    notes="C_in -> IC -> L -> C_out, BST cap near IC, FB divider beside output.",
    version="1.0",
)

_LDO_REGULATOR = SubcircuitTemplate(
    circuit_type_name="ldo_regulator",
    name="LDO Regulator",
    flow_direction="left_to_right",
    slots=(
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="C_in",
            offset_x=-4.0,
            offset_y=0.0,
        ),
        TemplateSlot(
            role=ComponentRole.ANCHOR,
            ref_pattern="IC",
            offset_x=0.0,
            offset_y=0.0,
        ),
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="C_out",
            offset_x=4.0,
            offset_y=0.0,
        ),
    ),
    notes="Compact linear: C_in -> IC -> C_out.",
    version="1.0",
)

_RELAY_DRIVER = SubcircuitTemplate(
    circuit_type_name="relay_driver",
    name="Relay Driver",
    flow_direction="top_to_bottom",
    slots=(
        TemplateSlot(
            role=ComponentRole.SERIES,
            ref_pattern="R_gate",
            offset_x=0.0,
            offset_y=-8.0,
        ),
        TemplateSlot(
            role=ComponentRole.SWITCH,
            ref_pattern="Q",
            offset_x=0.0,
            offset_y=-5.0,
        ),
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="D_flyback",
            offset_x=2.5,
            offset_y=-5.0,
        ),
        TemplateSlot(
            role=ComponentRole.ANCHOR,
            ref_pattern="K",
            offset_x=0.0,
            offset_y=0.0,
        ),
        TemplateSlot(
            role=ComponentRole.OUTPUT,
            ref_pattern="terminal",
            offset_x=0.0,
            offset_y=8.0,
        ),
    ),
    notes="R_gate -> Q -> D_flyback -> K(relay) -> terminal, vertical column.",
    version="1.0",
)

_CRYSTAL_OSC = SubcircuitTemplate(
    circuit_type_name="crystal_osc",
    name="Crystal Oscillator",
    flow_direction="radial",
    slots=(
        TemplateSlot(
            role=ComponentRole.ANCHOR,
            ref_pattern="Y",
            offset_x=0.0,
            offset_y=0.0,
        ),
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="C_load1",
            offset_x=-2.0,
            offset_y=2.0,
            rotation=90.0,
        ),
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="C_load2",
            offset_x=2.0,
            offset_y=2.0,
            rotation=90.0,
        ),
    ),
    notes="Y + C_load1 + C_load2 symmetric around MCU OSC pins.",
    version="1.0",
)

_DECOUPLING = SubcircuitTemplate(
    circuit_type_name="decoupling",
    name="Decoupling Capacitor",
    flow_direction="radial",
    slots=(
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="C_small",
            offset_x=-1.5,
            offset_y=0.0,
        ),
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="C_bulk",
            offset_x=1.5,
            offset_y=0.0,
        ),
    ),
    notes="C_small + C_bulk tight to IC VCC pin, no signal flow.",
    version="1.0",
)

_RC_FILTER = SubcircuitTemplate(
    circuit_type_name="rc_filter",
    name="RC Filter",
    flow_direction="left_to_right",
    slots=(
        TemplateSlot(
            role=ComponentRole.SERIES,
            ref_pattern="R",
            offset_x=-2.0,
            offset_y=0.0,
            pad_face_toward="C",
        ),
        TemplateSlot(
            role=ComponentRole.SHUNT,
            ref_pattern="C",
            offset_x=2.0,
            offset_y=0.0,
            rotation=90.0,
        ),
    ),
    notes="R -> C to GND, compact L-shape.",
    version="1.0",
)


# ---------------------------------------------------------------------------
# Register built-in templates
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    """Register all built-in IC and subcircuit templates."""
    for ic_tmpl in (_ESP32_S3_WROOM, _ADS1115, _W5500, _LAN8720A):
        register_ic_template(ic_tmpl)

    for sc_tmpl in (
        _VOLTAGE_DIVIDER, _ADC_CHANNEL, _BUCK_CONVERTER, _LDO_REGULATOR,
        _RELAY_DRIVER, _CRYSTAL_OSC, _DECOUPLING, _RC_FILTER,
    ):
        register_subcircuit_template(sc_tmpl)


_register_builtins()
