"""Zone strategy recommendation engine.

Analyzes board characteristics (analog/digital mixing, power net count,
current levels, RF modules) to recommend optimal GND plane strategy,
power zones, and thermal relief settings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

# Patterns for detecting RF-related components
_RF_PATTERNS: tuple[str, ...] = (
    "ANT", "RF", "WIFI", "BLE", "2G4", "ESP32", "NRF", "CC2530",
)

# Patterns for detecting power net names
_POWER_NET_RE = re.compile(
    r"^(\+|VCC|VDD|VBUS|VBAT|VMOT|VSYS)"
    r"|3V3|5V|12V|24V|1V8|2V5|3V0"
    r"|_RAIL$|_PWR$",
    re.IGNORECASE,
)

# Patterns for detecting analog net names
_ANALOG_NET_RE = re.compile(
    r"ANALOG|AIN|AOUT|ADC|DAC|VREF",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ZoneStrategy:
    """Recommended zone/copper pour strategy for a PCB.

    Attributes:
        gnd_strategy: Ground plane strategy - ``"both"``, ``"back_only"``, or ``"split"``.
        power_zones: Net names that should have dedicated power zones.
        copper_fill_ratio: Estimated copper fill ratio (0.0--1.0).
        thermal_relief_style: ``"solid"`` for high-current, ``"relief"`` for standard.
        rationale: Human-readable reasons for each decision.
    """

    gnd_strategy: str
    power_zones: tuple[str, ...]
    copper_fill_ratio: float
    thermal_relief_style: str
    rationale: tuple[str, ...]


def recommend_zone_strategy(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> ZoneStrategy:
    """Recommend optimal zone strategy based on board characteristics.

    Decision logic:
    - RF module present -> ``"back_only"`` GND (preserve antenna ground plane)
    - Analog + digital nets -> ``"split"`` GND with star-point connection
    - >4 unique power nets -> dedicated power zones
    - High-current (>1A on any rail) -> ``"solid"`` thermal relief
    - Default -> ``"both"`` (GND on F.Cu and B.Cu)

    Estimates copper fill ratio based on component density.
    """
    rationale: list[str] = []

    # Determine GND strategy
    rf_present = _has_rf_module(requirements)
    analog_present = _has_analog_nets(requirements)

    if rf_present:
        gnd_strategy = "back_only"
        rationale.append(
            "RF module detected; using back-only GND plane to preserve antenna ground."
        )
    elif analog_present:
        gnd_strategy = "split"
        rationale.append(
            "Analog and digital nets detected; using split GND with star-point connection."
        )
    else:
        gnd_strategy = "both"
        rationale.append("Standard design; GND plane on both F.Cu and B.Cu.")

    # Determine power zones
    power_net_count = _count_power_nets(pcb)
    power_zones: tuple[str, ...] = ()
    if power_net_count > 4:
        power_zones = _get_power_net_names(pcb)
        rationale.append(
            f"{power_net_count} power nets detected; dedicating zones for each."
        )
    else:
        rationale.append(
            f"{power_net_count} power net(s); no dedicated power zones needed."
        )

    # Determine thermal relief style
    max_current = _max_rail_current_ma(requirements)
    if max_current > 1000.0:
        thermal_relief_style = "solid"
        rationale.append(
            f"High-current rail ({max_current:.0f} mA) detected; using solid thermal relief."
        )
    else:
        thermal_relief_style = "relief"
        rationale.append("Standard current levels; using thermal relief pads.")

    # Estimate copper fill ratio
    fill_ratio = _estimate_fill_ratio(pcb)
    rationale.append(f"Estimated copper fill ratio: {fill_ratio:.2f}.")

    return ZoneStrategy(
        gnd_strategy=gnd_strategy,
        power_zones=power_zones,
        copper_fill_ratio=fill_ratio,
        thermal_relief_style=thermal_relief_style,
        rationale=tuple(rationale),
    )


def _has_rf_module(requirements: ProjectRequirements) -> bool:
    """Check for antenna/RF/WiFi/BLE components in the design."""
    for comp in requirements.components:
        upper_ref = comp.ref.upper()
        upper_value = comp.value.upper()
        upper_fp = comp.footprint.upper()
        combined = f"{upper_ref} {upper_value} {upper_fp}"
        for pattern in _RF_PATTERNS:
            if pattern in combined:
                return True
    return False


def _has_analog_nets(requirements: ProjectRequirements) -> bool:
    """Check for ADC/DAC/analog pins or net names in the design."""
    from kicad_pipeline.models.requirements import PinFunction

    # Check pin functions
    for comp in requirements.components:
        for pin in comp.pins:
            if pin.function in (
                PinFunction.ADC,
                PinFunction.DAC,
                PinFunction.ANALOG_IN,
                PinFunction.ANALOG_OUT,
            ):
                return True

    # Check net names
    return any(_ANALOG_NET_RE.search(net.name) for net in requirements.nets)


def _count_power_nets(pcb: PCBDesign) -> int:
    """Count unique power net names in the PCB design."""
    count = 0
    for net_entry in pcb.nets:
        if net_entry.name and _POWER_NET_RE.search(net_entry.name):
            count += 1
    return count


def _get_power_net_names(pcb: PCBDesign) -> tuple[str, ...]:
    """Return names of all power nets in the PCB design."""
    names: list[str] = []
    for net_entry in pcb.nets:
        if net_entry.name and _POWER_NET_RE.search(net_entry.name):
            names.append(net_entry.name)
    return tuple(sorted(names))


def _max_rail_current_ma(requirements: ProjectRequirements) -> float:
    """Return the maximum current (mA) across all power rails.

    Returns 0.0 if no power budget is defined.
    """
    if requirements.power_budget is None:
        return 0.0
    if not requirements.power_budget.rails:
        return 0.0
    return max(rail.current_ma for rail in requirements.power_budget.rails)


def _estimate_fill_ratio(pcb: PCBDesign) -> float:
    """Estimate copper fill as ``1.0 - (component_area / board_area)``.

    Uses the board outline polygon bounding box for board area and
    footprint pad bounding boxes for component area estimation.
    Returns a value clamped to [0.0, 1.0].
    """
    board_area = _board_area_from_outline(pcb)
    if board_area <= 0.0:
        return 0.5  # sensible default for degenerate boards

    component_area = 0.0
    for fp in pcb.footprints:
        if fp.pads:
            xs = [p.position.x for p in fp.pads]
            ys = [p.position.y for p in fp.pads]
            # Approximate footprint area from pad extents + pad sizes
            max_pad_sx = max(p.size_x for p in fp.pads)
            max_pad_sy = max(p.size_y for p in fp.pads)
            w = (max(xs) - min(xs)) + max_pad_sx
            h = (max(ys) - min(ys)) + max_pad_sy
            component_area += w * h

    ratio = 1.0 - (component_area / board_area)
    return max(0.0, min(1.0, ratio))


def _board_area_from_outline(pcb: PCBDesign) -> float:
    """Compute board area from outline polygon bounding box."""
    if not pcb.outline.polygon:
        return 0.0
    xs = [p.x for p in pcb.outline.polygon]
    ys = [p.y for p in pcb.outline.polygon]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return width * height
