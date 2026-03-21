"""Layout and routing guide generator.

Produces a project-specific ``layout_guide.md`` that documents placement rules,
routing constraints, decoupling requirements, and keep-away distances derived
from the project requirements.
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.requirements import (
        Component,
        ProjectRequirements,
    )

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IPC-2221 trace width calculation
# ---------------------------------------------------------------------------

# Constants for IPC-2221 internal layers (conservative — external is wider)
_IPC_K = 0.024  # constant for internal copper
_IPC_B = 0.44
_IPC_C = 0.725
_COPPER_THICKNESS_OZ = 1  # 1 oz/ft² = 35 µm
_TEMP_RISE_C = 10.0  # 10°C rise


def _trace_width_mm(current_a: float) -> float:
    """Compute trace width in mm for *current_a* amps using IPC-2221.

    Uses the external-layer formula with 1 oz copper and 10°C rise.

    Args:
        current_a: Current in amps.

    Returns:
        Trace width in mm (rounded up to 0.1mm).
    """
    if current_a <= 0:
        return 0.15  # minimum JLCPCB trace

    # Area in mils²
    area = (current_a / (_IPC_K * _TEMP_RISE_C ** _IPC_B)) ** (1.0 / _IPC_C)
    # Width in mils = area / thickness (1 oz ≈ 1.378 mils)
    thickness_mils = _COPPER_THICKNESS_OZ * 1.378
    width_mils = area / thickness_mils
    width_mm = width_mils * 0.0254
    # Round up to nearest 0.1mm, floor at JLCPCB minimum
    return max(0.15, float(math.ceil(width_mm * 10)) / 10)


# ---------------------------------------------------------------------------
# Component type detection
# ---------------------------------------------------------------------------

_REF_PREFIX_TYPES: dict[str, str] = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "D": "diode",
    "Q": "transistor",
    "U": "ic",
    "J": "connector",
    "K": "relay",
    "F": "fuse",
    "SW": "switch",
    "Y": "crystal",
    "FB": "ferrite_bead",
    "TP": "test_point",
}

# Patterns in value/description that identify component sub-types
# Order matters: more specific patterns first (wifi_ble before mcu)
_IC_PATTERNS: dict[str, tuple[str, ...]] = {
    "wifi_ble": ("ESP32", "nRF52", "CC2540", "WROOM", "WROVER"),
    "mcu": ("STM32", "ATmega", "RP2040", "PIC", "SAMD"),
    "adc": ("ADS1", "MCP3", "ADS8", "AD7"),
    "ldo": ("AMS1117", "LM1117", "AP2112", "MCP1700", "HT7333", "XC6206"),
    "buck": ("LM2596", "MP1584", "TPS54", "LM3671"),
    "relay": ("HFD4", "SRD", "G5V", "G6K"),
    "sensor": ("BME280", "BMP280", "SHT3", "DHT", "LM35", "TMP36"),
    "eeprom": ("AT24C", "24LC", "M24C"),
    "rtc": ("DS1307", "DS3231", "PCF8563"),
    "display_driver": ("SSD1306", "ST7789", "ILI9341"),
    "usb": ("CH340", "CP210", "FT232", "MCP2221"),
}

_DIFF_PAIR_PATTERNS = re.compile(
    r"(USB_D[PM]|USB[_.]?D[+-]|D[+-]|_[PN]$|ETH_TX|ETH_RX|LVDS|MIPI)", re.IGNORECASE,
)

_ISOLATION_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("relay", "adc", "15mm", "Relay switching noise couples into ADC inputs"),
    ("relay", "sensor", "10mm", "Relay EMI affects sensitive analog sensors"),
    ("wifi_ble", "adc", "10mm", "RF emissions affect ADC accuracy"),
    ("wifi_ble", "crystal", "5mm", "RF can detune crystal oscillators"),
    ("buck", "adc", "15mm", "Switching regulator noise affects ADC"),
    ("buck", "sensor", "10mm", "Switching noise affects analog sensors"),
)


def _detect_ic_subtype(comp: Component) -> str | None:
    """Detect IC sub-type from value and description strings."""
    check_str = f"{comp.value} {comp.description or ''}".upper()
    for subtype, patterns in _IC_PATTERNS.items():
        for pat in patterns:
            if pat.upper() in check_str:
                return subtype
    return None


def _ref_type(ref: str) -> str:
    """Extract component type from reference designator prefix."""
    prefix = re.match(r"[A-Z]+", ref)
    if prefix:
        return _REF_PREFIX_TYPES.get(prefix.group(), "unknown")
    return "unknown"


# ---------------------------------------------------------------------------
# Guide sections
# ---------------------------------------------------------------------------


def _board_overview(req: ProjectRequirements) -> str:
    """Generate the board overview section."""
    lines = ["## Board Overview", ""]
    mech = req.mechanical
    if mech:
        lines.append(
            f"- **Dimensions**: {mech.board_width_mm} x {mech.board_height_mm} mm"
        )
        if mech.board_template:
            lines.append(f"- **Template**: {mech.board_template}")
        if mech.enclosure:
            lines.append(f"- **Enclosure**: {mech.enclosure}")
    lines.append(f"- **Components**: {len(req.components)}")
    lines.append(f"- **Nets**: {len(req.nets)}")
    if req.power_budget:
        lines.append(
            f"- **Power rails**: {', '.join(r.name for r in req.power_budget.rails)}"
        )
    lines.append("")
    return "\n".join(lines)


_IC_PLACEMENT_GUIDANCE: dict[str, tuple[str, ...]] = {
    "wifi_ble": (
        "- Place near board edge, antenna outward",
        "- 15mm copper keepout under/around antenna area",
        "- 100nF + 10µF decoupling within 3mm of VCC pins",
    ),
    "mcu": (
        "- Central placement preferred for short trace runs",
        "- 100nF decoupling on every VCC/AVCC pin, within 3mm",
        "- Crystal/oscillator within 10mm, GND guard traces",
    ),
    "adc": (
        "- Keep away from switching regulators and relays",
        "- Analog GND pour under IC, single-point GND connection",
        "- 100nF + 10µF decoupling within 2mm of VCC",
    ),
    "ldo": (
        "- Input cap within 5mm, output cap within 3mm",
        "- Thermal pad to copper pour if available",
    ),
    "buck": (
        "- Compact layout: input cap, IC, inductor, output cap — tight loop",
        "- Minimise SW node area (high dV/dt)",
        "- Keep far from sensitive analog circuits",
    ),
    "relay": (
        "- Keep away from sensitive analog (ADC, sensors)",
        "- Flyback diode within 5mm of coil pins",
    ),
    "sensor": (
        "- Isolate from heat sources and switching noise",
        "- Short, direct connections to ADC inputs",
    ),
    "usb": (
        "- Place near USB connector, short D+/D- traces",
        "- 100nF decoupling within 3mm",
    ),
}

_DEFAULT_IC_GUIDANCE: tuple[str, ...] = (
    "- 100nF decoupling within 3mm of VCC pins",
)


def _connector_guidance(desc_lower: str) -> list[str]:
    """Return placement guidance lines for a connector based on its description."""
    if "usb" in desc_lower:
        return [
            "- Board edge placement, accessible from enclosure",
            "- ESD protection IC within 10mm of connector",
        ]
    if "screw" in desc_lower or "terminal" in desc_lower:
        return [
            "- Board edge, accessible for wire insertion",
            "- Wide traces to terminal pads (match current rating)",
        ]
    if "header" in desc_lower or "gpio" in desc_lower or "pin" in desc_lower:
        return ["- Board edge for easy ribbon cable access"]
    return ["- Board edge preferred for external connections"]


def _placement_rules(req: ProjectRequirements) -> str:
    """Generate per-component placement guidance."""
    lines = ["## Placement Rules", ""]

    ic_components: list[tuple[Component, str | None]] = []
    connector_components: list[Component] = []

    for comp in req.components:
        rtype = _ref_type(comp.ref)
        if rtype == "ic":
            subtype = _detect_ic_subtype(comp)
            ic_components.append((comp, subtype))
        elif rtype == "connector":
            connector_components.append(comp)

    # ICs with placement guidance
    for comp, subtype in ic_components:
        lines.append(f"### {comp.ref} — {comp.value}")
        guidance = _IC_PLACEMENT_GUIDANCE.get(subtype or "", _DEFAULT_IC_GUIDANCE)
        lines.extend(guidance)
        lines.append("")

    # Connectors
    for comp in connector_components:
        lines.append(f"### {comp.ref} — {comp.value}")
        desc_lower = (comp.description or comp.value).lower()
        lines.extend(_connector_guidance(desc_lower))
        lines.append("")

    if not ic_components and not connector_components:
        lines.append("No ICs or connectors requiring special placement guidance.")
        lines.append("")

    return "\n".join(lines)


_DIFF_PAIR_SPECS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("USB",), "90Ω", "±0.5mm"),
    (("ETH",), "100Ω", "±1.0mm"),
    (("LVDS", "MIPI"), "100Ω", "±0.25mm"),
)


def _diff_pair_spec(net_name: str) -> tuple[str, str]:
    """Return (impedance, length_match) for a differential pair net."""
    upper = net_name.upper()
    for keywords, impedance, length_match in _DIFF_PAIR_SPECS:
        if any(kw in upper for kw in keywords):
            return impedance, length_match
    return "—", "Check datasheet"


def _routing_constraints(req: ProjectRequirements) -> str:
    """Generate routing constraint tables."""
    lines = ["## Routing Constraints", ""]

    # --- Differential pairs ---
    diff_nets: list[str] = []
    for net in req.nets:
        if _DIFF_PAIR_PATTERNS.search(net.name):
            diff_nets.append(net.name)

    if diff_nets:
        lines.append("### Differential Pairs")
        lines.append("")
        lines.append("| Signal | Impedance | Length Match |")
        lines.append("|--------|-----------|-------------|")
        for net_name in sorted(diff_nets):
            impedance, length_match = _diff_pair_spec(net_name)
            lines.append(f"| {net_name} | {impedance} | {length_match} |")
        lines.append("")

    # --- Power traces ---
    if req.power_budget and req.power_budget.rails:
        lines.append("### Power Traces")
        lines.append("")
        lines.append("| Net | Current | Min Width |")
        lines.append("|-----|---------|-----------|")
        for rail in req.power_budget.rails:
            current_a = rail.current_ma / 1000.0
            width = _trace_width_mm(current_a)
            lines.append(
                f"| {rail.name} | {rail.current_ma:.0f}mA | {width:.1f}mm |"
            )
        lines.append("")

    # --- Keep-away rules ---
    # Detect which IC subtypes are present
    present_subtypes: dict[str, list[str]] = {}
    for comp in req.components:
        rtype = _ref_type(comp.ref)
        if rtype == "ic":
            subtype = _detect_ic_subtype(comp)
            if subtype:
                present_subtypes.setdefault(subtype, []).append(comp.ref)
        elif rtype == "relay":
            present_subtypes.setdefault("relay", []).append(comp.ref)
        elif rtype == "crystal":
            present_subtypes.setdefault("crystal", []).append(comp.ref)

    keepaway_rows: list[tuple[str, str, str, str]] = []
    for type_a, type_b, distance, reason in _ISOLATION_PAIRS:
        if type_a in present_subtypes and type_b in present_subtypes:
            refs_a = ", ".join(present_subtypes[type_a])
            refs_b = ", ".join(present_subtypes[type_b])
            keepaway_rows.append((refs_a, refs_b, distance, reason))

    if keepaway_rows:
        lines.append("### Keep-Away Rules")
        lines.append("")
        lines.append("| From | To | Distance | Reason |")
        lines.append("|------|----|----------|--------|")
        for from_refs, to_refs, dist, reason in keepaway_rows:
            lines.append(f"| {from_refs} | {to_refs} | ≥{dist} | {reason} |")
        lines.append("")

    if not diff_nets and not (req.power_budget and req.power_budget.rails) and not keepaway_rows:
        lines.append("No special routing constraints detected.")
        lines.append("")

    return "\n".join(lines)


def _decoupling_section(req: ProjectRequirements) -> str:
    """Generate decoupling recommendations for ICs."""
    lines = ["## Decoupling", ""]

    ic_decoupling: list[tuple[str, str, str]] = []
    for comp in req.components:
        if _ref_type(comp.ref) != "ic":
            continue
        subtype = _detect_ic_subtype(comp)
        if subtype == "adc":
            ic_decoupling.append(
                (comp.ref, comp.value, "100nF + 10µF within 2mm of VCC, analog GND pour")
            )
        elif subtype in ("mcu", "wifi_ble"):
            ic_decoupling.append(
                (comp.ref, comp.value, "100nF on every VCC pin within 3mm, 10µF bulk nearby")
            )
        elif subtype in ("ldo", "buck"):
            ic_decoupling.append(
                (comp.ref, comp.value,
                 "Per datasheet: input cap within 5mm, output cap within 3mm")
            )
        elif subtype is not None:
            ic_decoupling.append(
                (comp.ref, comp.value, "100nF within 3mm of VCC")
            )

    if ic_decoupling:
        lines.append("| IC | Value | Decoupling |")
        lines.append("|----|-------|------------|")
        for ref, value, decoupling in ic_decoupling:
            lines.append(f"| {ref} | {value} | {decoupling} |")
    else:
        lines.append("No ICs requiring special decoupling guidance.")
    lines.append("")
    return "\n".join(lines)


def _thermal_section(req: ProjectRequirements) -> str:
    """Generate thermal management notes."""
    lines = ["## Thermal Considerations", ""]

    thermal_components: list[tuple[str, str, str]] = []
    for comp in req.components:
        subtype = _detect_ic_subtype(comp)
        if subtype == "ldo":
            thermal_components.append(
                (comp.ref, comp.value, "Thermal pad to copper pour; check dropout voltage")
            )
        elif subtype == "buck":
            thermal_components.append(
                (comp.ref, comp.value, "Copper pour under IC for heat dissipation")
            )
        elif _ref_type(comp.ref) == "relay":
            thermal_components.append(
                (comp.ref, comp.value, "Coil dissipates heat — allow airflow clearance")
            )

    if thermal_components:
        for ref, value, note in thermal_components:
            lines.append(f"- **{ref}** ({value}): {note}")
    else:
        lines.append("No components with special thermal requirements detected.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_layout_guide(
    requirements: ProjectRequirements,
    output_path: Path,
) -> Path:
    """Generate a layout and routing guide from project requirements.

    Produces a Markdown document with placement rules, routing constraints,
    decoupling recommendations, keep-away distances, and thermal notes.

    Args:
        requirements: Project requirements to analyze.
        output_path: File path to write the guide to.

    Returns:
        The path the guide was written to.
    """
    from pathlib import Path as _Path

    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    project_name = requirements.project.name

    sections = [
        f"# Layout & Routing Guide — {project_name}",
        "",
        _board_overview(requirements),
        _placement_rules(requirements),
        _routing_constraints(requirements),
        _decoupling_section(requirements),
        _thermal_section(requirements),
    ]

    content = "\n".join(sections)
    output_path.write_text(content, encoding="utf-8")
    log.info("Layout guide written: %s", output_path)
    return output_path
