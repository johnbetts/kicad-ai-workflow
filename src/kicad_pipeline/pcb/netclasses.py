"""Net classification engine for PCB design rules.

Assigns nets to netclasses based on name-pattern matching, providing
appropriate trace widths, clearances, and via sizes for each net category.

Voltage-aware power netclasses split power rails into per-voltage subclasses
with IPC-2221-derived clearances and trace widths.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    JLCPCB_MIN_TRACE_MM,
    VOLTAGE_CLEARANCE_THRESHOLDS,
)
from kicad_pipeline.models.pcb import NetClass

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import DesignRules, NetEntry


# ---------------------------------------------------------------------------
# Pattern definitions: (compiled regex, NetClass template)
# ---------------------------------------------------------------------------

_POWER_PATTERN: re.Pattern[str] = re.compile(
    r"^(GND|\+\d*\.?\d*V\d*|VCC|VDD|VBUS|PWR|V_\w+|VBAT)$",
    re.IGNORECASE,
)

_NETCLASS_PATTERNS: tuple[tuple[re.Pattern[str], str, float, float, float, float], ...] = (
    # pattern, class_name, trace_width, clearance, via_diameter, via_drill
    (
        re.compile(r"^(SENS|AIN|ADC|VREF)", re.IGNORECASE),
        "HighVoltageAnalog",
        0.4,
        0.2,
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
    (
        re.compile(r"^(ANT|RF|WIFI|BLE|2G4)", re.IGNORECASE),
        "RF",
        0.3,
        0.3,
        0.8,
        0.508,
    ),
)

_DEFAULT_NETCLASS_NAME: str = "Default"

# ---------------------------------------------------------------------------
# Voltage parsing & IPC-2221 helpers
# ---------------------------------------------------------------------------

# Matches +3V3, +5V, +12V, +24V, +48V style net names
_VOLTAGE_PATTERN_VXV: re.Pattern[str] = re.compile(r"^\+(\d+)V(\d+)?$")
# Matches +3.3V, +12.0V style net names
_VOLTAGE_PATTERN_DOT: re.Pattern[str] = re.compile(r"^\+(\d+\.\d+)V$")


def _parse_voltage(net_name: str) -> float | None:
    """Extract voltage from a power-net name.

    Returns the voltage in volts, 0.0 for GND, or None if the voltage
    cannot be determined (e.g. VDD, VCC, VBUS).

    Examples:
        >>> _parse_voltage("+3V3")
        3.3
        >>> _parse_voltage("+5V")
        5.0
        >>> _parse_voltage("GND")
        0.0
        >>> _parse_voltage("VDD")
    """
    if net_name.upper() == "GND":
        return 0.0

    # +3.3V, +12.0V style
    m = _VOLTAGE_PATTERN_DOT.match(net_name)
    if m:
        return float(m.group(1))

    # +3V3, +5V, +12V style
    m = _VOLTAGE_PATTERN_VXV.match(net_name)
    if m:
        integer_part = m.group(1)
        decimal_part = m.group(2)
        if decimal_part:
            return float(f"{integer_part}.{decimal_part}")
        return float(integer_part)

    return None


def _voltage_clearance(voltage_v: float) -> float:
    """Return minimum clearance in mm for the given voltage (IPC-2221).

    Uses :data:`~kicad_pipeline.constants.VOLTAGE_CLEARANCE_THRESHOLDS`
    to look up the appropriate clearance for external-layer copper.
    """
    for max_v, clearance in VOLTAGE_CLEARANCE_THRESHOLDS:
        if voltage_v <= max_v:
            return clearance
    # Above all thresholds - use the last (widest) clearance
    return VOLTAGE_CLEARANCE_THRESHOLDS[-1][1]


def _current_trace_width(current_a: float, temp_rise_c: float = 10.0) -> float:
    """Return minimum trace width in mm for the given current (IPC-2221).

    Uses the IPC-2221 outer-layer formula for 1 oz copper:
        area_mils2 = (I / (k * dT^b))^(1/c)
    where k=0.048, b=0.44, c=0.725 for outer layers.

    The result is floored at :data:`~kicad_pipeline.constants.JLCPCB_MIN_TRACE_MM`.
    """
    if current_a <= 0:
        return JLCPCB_MIN_TRACE_MM
    k = 0.048
    b = 0.44
    c = 0.725
    area_mils2 = (current_a / (k * temp_rise_c**b)) ** (1.0 / c)
    # 1 oz copper thickness = 1.378 mils
    width_mils = area_mils2 / 1.378
    width_mm: float = width_mils * 0.0254
    return max(width_mm, JLCPCB_MIN_TRACE_MM)


def _voltage_label(net_name: str) -> str:
    """Extract a clean voltage label from a net name for netclass naming.

    The label preserves the KiCad naming convention where ``V`` acts as
    both a unit suffix and a decimal separator.

    Examples:
        >>> _voltage_label("+3V3")
        '3V3'
        >>> _voltage_label("+5V")
        '5V'
        >>> _voltage_label("+12V")
        '12V'
        >>> _voltage_label("+3.3V")
        '3V3'
    """
    s = net_name.lstrip("+")
    # Case 1: dot notation like "3.3V" -> strip V, replace dot: "3V3"
    if "." in s:
        if s.upper().endswith("V"):
            s = s[:-1]
        return s.replace(".", "V")
    # Case 2: already in KiCad VxV format like "3V3" -> keep as-is
    # Case 3: integer voltage like "5V" or "12V" -> keep as-is
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_nets(
    net_entries: tuple[NetEntry, ...],
    design_rules: DesignRules | None = None,
) -> tuple[NetClass, ...]:
    """Classify nets into netclasses by name pattern matching.

    Each net is matched against known patterns (Power, HighVoltageAnalog,
    I2C, SPI, RF). Power nets with identifiable voltages get per-voltage
    subclasses (e.g. ``Power_3V3``, ``Power_5V``) with IPC-2221-derived
    clearances. Unmatched nets fall into the Default class. The empty net
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

        # Check power pattern first
        if _POWER_PATTERN.search(entry.name):
            voltage = _parse_voltage(entry.name)
            if entry.name.upper() == "GND" or voltage == 0.0:
                # GND stays in generic Power class
                class_name = "Power"
                tw, cl = 0.3, 0.2
            elif voltage is not None and voltage > 0:
                # Per-voltage subclass
                v_label = _voltage_label(entry.name)
                class_name = f"Power_{v_label}"
                cl = _voltage_clearance(voltage)
                tw = 0.3  # default power trace width
            else:
                # Unknown voltage (VDD, VCC, VBUS, etc.) - conservative
                class_name = "Power"
                tw, cl = 0.3, 0.2

            buckets.setdefault(class_name, []).append(entry.name)
            class_params[class_name] = (tw, cl, 0.8, 0.508)
            continue

        # Check non-power patterns
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
