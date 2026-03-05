"""Net classification engine for PCB design rules.

Assigns nets to netclasses based on name-pattern matching, providing
appropriate trace widths, clearances, and via sizes for each net category.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import NetClass

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import DesignRules, NetEntry


# ---------------------------------------------------------------------------
# Pattern definitions: (compiled regex, NetClass template)
# ---------------------------------------------------------------------------

_NETCLASS_PATTERNS: tuple[tuple[re.Pattern[str], str, float, float, float, float], ...] = (
    # pattern, class_name, trace_width, clearance, via_diameter, via_drill
    (
        re.compile(r"^(GND|\+\d*V\d*|VCC|VDD|VBUS|PWR|V_\w+|VBAT)$", re.IGNORECASE),
        "Power",
        0.5,
        0.3,
        0.8,
        0.508,
    ),
    (
        re.compile(r"^(SENS|AIN|ADC|VREF)", re.IGNORECASE),
        "HighVoltageAnalog",
        0.4,
        0.3,
        0.8,
        0.508,
    ),
    (
        re.compile(r"^(SPI[_\d]|MOSI|MISO|SCLK|SCK|CS)", re.IGNORECASE),
        "SPI",
        0.2,
        0.2,
        0.8,
        0.508,
    ),
    (
        re.compile(r"^(I2C[_\d]|SDA|SCL$|SCL[_\d])", re.IGNORECASE),
        "I2C",
        0.25,
        0.2,
        0.8,
        0.508,
    ),
)

_DEFAULT_NETCLASS_NAME: str = "Default"


def classify_nets(
    net_entries: tuple[NetEntry, ...],
    design_rules: DesignRules | None = None,
) -> tuple[NetClass, ...]:
    """Classify nets into netclasses by name pattern matching.

    Each net is matched against known patterns (Power, HighVoltageAnalog,
    I2C, SPI). Unmatched nets fall into the Default class. The empty net
    (net 0) is always skipped.

    Args:
        net_entries: All nets from the PCB design.
        design_rules: Optional design rules to override default values.

    Returns:
        Tuple of NetClass objects, one per classification group that has
        at least one matching net, plus a Default class for unmatched nets.
    """
    # Bucket nets by class name
    buckets: dict[str, list[str]] = {}
    class_params: dict[str, tuple[float, float, float, float]] = {}

    for entry in net_entries:
        if not entry.name:
            continue  # skip unconnected net 0

        matched = False
        for pattern, class_name, tw, cl, vd, vdr in _NETCLASS_PATTERNS:
            if pattern.search(entry.name):
                buckets.setdefault(class_name, []).append(entry.name)
                class_params[class_name] = (tw, cl, vd, vdr)
                matched = True
                break

        if not matched:
            buckets.setdefault(_DEFAULT_NETCLASS_NAME, []).append(entry.name)

    # Build NetClass objects
    result: list[NetClass] = []

    # Default class first (always present)
    default_tw = 0.25
    default_cl = 0.2
    if design_rules is not None:
        default_tw = design_rules.default_trace_width_mm
        default_cl = design_rules.default_clearance_mm

    default_nets = tuple(buckets.pop(_DEFAULT_NETCLASS_NAME, []))
    result.append(
        NetClass(
            name=_DEFAULT_NETCLASS_NAME,
            clearance_mm=default_cl,
            trace_width_mm=default_tw,
            nets=default_nets,
        )
    )

    # Named classes
    for class_name, nets in sorted(buckets.items()):
        tw, cl, vd, vdr = class_params[class_name]
        result.append(
            NetClass(
                name=class_name,
                trace_width_mm=tw,
                clearance_mm=cl,
                via_diameter_mm=vd,
                via_drill_mm=vdr,
                nets=tuple(nets),
            )
        )

    return tuple(result)


def net_width_map(netclasses: tuple[NetClass, ...]) -> dict[str, float]:
    """Build a net_name -> trace_width mapping for use by the router.

    Args:
        netclasses: Classified netclasses from :func:`classify_nets`.

    Returns:
        Dictionary mapping each net name to its trace width in mm.
    """
    result: dict[str, float] = {}
    for nc in netclasses:
        for net_name in nc.nets:
            result[net_name] = nc.trace_width_mm
    return result


def net_clearance_map(netclasses: tuple[NetClass, ...]) -> dict[str, float]:
    """Build a net_name -> clearance mapping for use by the router.

    Args:
        netclasses: Classified netclasses from :func:`classify_nets`.

    Returns:
        Dictionary mapping each net name to its clearance in mm.
    """
    result: dict[str, float] = {}
    for nc in netclasses:
        for net_name in nc.nets:
            result[net_name] = nc.clearance_mm
    return result
