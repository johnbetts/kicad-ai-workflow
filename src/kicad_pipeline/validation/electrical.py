"""Electrical design rule check (ERC) validation for PCB designs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.validation.drc import DRCViolation, Severity

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

_GND_NET_NAMES: frozenset[str] = frozenset({"GND", "AGND", "DGND"})
_POWER_NET_NAMES: frozenset[str] = frozenset({"VCC", "+5V", "+3V3", "+3.3V", "+1V8", "VBUS"})


@dataclass(frozen=True)
class ElectricalReport:
    """Result of electrical validation checks."""

    violations: tuple[DRCViolation, ...]

    @property
    def errors(self) -> tuple[DRCViolation, ...]:
        """Return only ERROR-severity violations."""
        return tuple(v for v in self.violations if v.severity == Severity.ERROR)

    @property
    def warnings(self) -> tuple[DRCViolation, ...]:
        """Return only WARNING-severity violations."""
        return tuple(v for v in self.violations if v.severity == Severity.WARNING)

    @property
    def passed(self) -> bool:
        """True if there are no ERROR-severity violations."""
        return len(self.errors) == 0


def run_electrical_checks(
    pcb: PCBDesign,
    requirements: ProjectRequirements | None = None,
) -> ElectricalReport:
    """Run all electrical validation checks against a PCBDesign.

    Args:
        pcb: The PCB design to validate.
        requirements: Optional project requirements for cross-checking.

    Returns:
        An ElectricalReport containing all violations found.
    """
    violations: list[DRCViolation] = []

    violations.extend(_check_net_completeness(pcb, requirements))
    violations.extend(_check_power_ground_nets(pcb))
    violations.extend(_check_decoupling_caps(pcb, requirements))
    violations.extend(_check_power_rail_voltage(requirements))
    violations.extend(_check_short_circuit(pcb))
    violations.extend(_check_dip_switch_protection(requirements))
    violations.extend(_check_3d_model_alignment(pcb))
    violations.extend(_check_signal_chain_placement(pcb, requirements))
    violations.extend(_check_relay_polarity(requirements))

    return ElectricalReport(violations=tuple(violations))


def _check_net_completeness(
    pcb: PCBDesign,
    requirements: ProjectRequirements | None,
) -> list[DRCViolation]:
    """Warn if any net from requirements is missing from the PCB."""
    if requirements is None:
        return []

    violations: list[DRCViolation] = []
    pcb_net_names = {net.name for net in pcb.nets}

    for req_net in requirements.nets:
        if req_net.name not in pcb_net_names:
            violations.append(
                DRCViolation(
                    rule="net_completeness",
                    message=(
                        f"Net '{req_net.name}' from requirements not found in PCB"
                    ),
                    severity=Severity.WARNING,
                )
            )
    return violations


def _check_power_ground_nets(pcb: PCBDesign) -> list[DRCViolation]:
    """Warn if no GND / AGND / DGND net is present in the PCB."""
    pcb_net_names = {net.name for net in pcb.nets}
    if not pcb_net_names & _GND_NET_NAMES:
        return [
            DRCViolation(
                rule="power_ground_nets",
                message="No GND net found in PCB -- check power connectivity",
                severity=Severity.WARNING,
            )
        ]
    return []


def _check_decoupling_caps(
    pcb: PCBDesign,
    requirements: ProjectRequirements | None,
) -> list[DRCViolation]:
    """Emit INFO if an IC has no decoupling cap in its feature block or nearby.

    When PCB footprint positions are available, also checks physical distance
    from each cap to its IC against :data:`DECOUPLING_CAP_MAX_DISTANCE_MM`.
    """
    if requirements is None:
        return []

    import math

    from kicad_pipeline.constants import DECOUPLING_CAP_MAX_DISTANCE_MM

    violations: list[DRCViolation] = []

    # Build a set of component refs that are capacitors (ref starts with "C").
    cap_refs = {comp.ref for comp in requirements.components if comp.ref.startswith("C")}

    # Build position lookup from PCB footprints
    fp_positions: dict[str, tuple[float, float]] = {}
    for fp in pcb.footprints:
        fp_positions[fp.ref] = (fp.position.x, fp.position.y)

    # For each feature block, collect all ICs and check if there is at least one cap.
    for feature in requirements.features:
        ic_refs_in_feature = [r for r in feature.components if r.startswith("U")]
        cap_refs_in_feature = cap_refs & set(feature.components)

        for ic_ref in ic_refs_in_feature:
            if not cap_refs_in_feature:
                violations.append(
                    DRCViolation(
                        rule="decoupling_caps",
                        message=(
                            f"No decoupling capacitor found near IC {ic_ref}"
                            " -- verify power supply filtering"
                        ),
                        severity=Severity.INFO,
                        ref=ic_ref,
                    )
                )
                continue

            # Physical distance check when positions are available
            ic_pos = fp_positions.get(ic_ref)
            if ic_pos is None:
                continue
            cap_distances = [
                math.hypot(cp[0] - ic_pos[0], cp[1] - ic_pos[1])
                for cap_ref in cap_refs_in_feature
                if (cp := fp_positions.get(cap_ref)) is not None
            ]
            if not cap_distances:
                continue
            closest_dist = min(cap_distances)
            if closest_dist > DECOUPLING_CAP_MAX_DISTANCE_MM:
                violations.append(
                    DRCViolation(
                        rule="decoupling_caps",
                        message=(
                            f"Nearest decoupling cap to {ic_ref} is"
                            f" {closest_dist:.1f}mm away (max"
                            f" {DECOUPLING_CAP_MAX_DISTANCE_MM:.1f}mm)"
                        ),
                        severity=Severity.INFO,
                        ref=ic_ref,
                    )
                )
    return violations


def _check_power_rail_voltage(
    requirements: ProjectRequirements | None,
) -> list[DRCViolation]:
    """Error if any power rail has an invalid (non-positive) voltage."""
    if requirements is None or requirements.power_budget is None:
        return []

    violations: list[DRCViolation] = []
    for rail in requirements.power_budget.rails:
        if rail.voltage <= 0:
            violations.append(
                DRCViolation(
                    rule="power_rail_voltage",
                    message=(
                        f"Power rail {rail.name} has invalid voltage"
                        f" {rail.voltage:.2f}V"
                    ),
                    severity=Severity.ERROR,
                )
            )
    return violations


def _check_dip_switch_protection(
    requirements: ProjectRequirements | None,
) -> list[DRCViolation]:
    """Warn if a DIP switch has outputs connecting to power nets without series resistors."""
    if requirements is None:
        return []

    violations: list[DRCViolation] = []

    # Find DIP switch components
    dip_switches = [
        c for c in requirements.components
        if c.footprint.upper().startswith("SW_DIP")
    ]

    all_power = _GND_NET_NAMES | _POWER_NET_NAMES

    for sw in dip_switches:
        # Check if any switch pin connects directly to a power net
        power_pins: list[str] = []
        for pin in sw.pins:
            if pin.net is not None and pin.net in all_power:
                power_pins.append(pin.number)

        if len(power_pins) >= 2:
            # Multiple pins on power nets without series protection
            violations.append(
                DRCViolation(
                    rule="dip_switch_protection",
                    message=(
                        f"DIP switch {sw.ref} has {len(power_pins)} pins "
                        f"connected directly to power nets ({', '.join(power_pins)}). "
                        f"Add series resistors to prevent short circuits."
                    ),
                    severity=Severity.WARNING,
                    ref=sw.ref,
                )
            )

    return violations


def _check_short_circuit(pcb: PCBDesign) -> list[DRCViolation]:
    """Error if a pad's net_name disagrees with the PCB net list for that number."""
    violations: list[DRCViolation] = []

    # Build lookup: net_number -> net_name from the authoritative net list.
    net_number_to_name: dict[int, str] = {net.number: net.name for net in pcb.nets}

    for fp in pcb.footprints:
        for pad in fp.pads:
            if (
                pad.net_number is not None
                and pad.net_number != 0
                and pad.net_name is not None
            ):
                expected = net_number_to_name.get(pad.net_number)
                if expected is not None and pad.net_name != expected:
                    violations.append(
                        DRCViolation(
                            rule="short_circuit_check",
                            message=(
                                f"Net mismatch: pad {fp.ref}.{pad.number} has"
                                f" net_number={pad.net_number} but"
                                f" net_name={pad.net_name} conflicts with"
                                f" net list name={expected}"
                            ),
                            severity=Severity.ERROR,
                            ref=fp.ref,
                        )
                    )
    return violations


def _check_3d_model_alignment(pcb: PCBDesign) -> list[DRCViolation]:
    """INFO-severity warnings for 3D model misalignment or missing models."""
    from kicad_pipeline.pcb.footprints import validate_3d_model_orientation

    violations: list[DRCViolation] = []
    for fp in pcb.footprints:
        for warning in validate_3d_model_orientation(fp):
            violations.append(
                DRCViolation(
                    rule="3d_model_alignment",
                    message=warning,
                    severity=Severity.INFO,
                    ref=fp.ref,
                )
            )
    return violations


def _check_signal_chain_placement(
    pcb: PCBDesign,
    requirements: ProjectRequirements | None,
) -> list[DRCViolation]:
    """INFO-severity warnings for suboptimal signal chain placement."""
    if requirements is None:
        return []

    from kicad_pipeline.pcb.constraints import validate_signal_chain_placement

    # Build position map from PCB footprints
    fp_positions = {}
    for fp in pcb.footprints:
        fp_positions[fp.ref] = fp.position

    if not fp_positions:
        return []

    chain_warnings = validate_signal_chain_placement(requirements, fp_positions)
    return [
        DRCViolation(
            rule="signal_chain_placement",
            message=warning,
            severity=Severity.INFO,
        )
        for warning in chain_warnings
    ]


# ---------------------------------------------------------------------------
# Relay polarity validation
# ---------------------------------------------------------------------------

_GND_RELAY_NAMES: frozenset[str] = frozenset(
    {"GND", "AGND", "DGND", "PGND", "RELAY_GND"}
)
_VIN_RELAY_NAMES: frozenset[str] = frozenset(
    {"VIN", "+12V", "+24V", "VBUS", "V_IN"}
)


def _check_relay_polarity(
    requirements: ProjectRequirements | None,
) -> list[DRCViolation]:
    """Validate relay wiring: polarity, flyback diode, driver chain."""
    if requirements is None:
        return []

    violations: list[DRCViolation] = []

    # Build net→refs map
    net_to_refs: dict[str, set[str]] = {}
    for net in requirements.nets:
        net_to_refs[net.name] = {conn.ref for conn in net.connections}

    {c.ref: c for c in requirements.components}

    for comp in requirements.components:
        prefix = "".join(c for c in comp.ref if c.isalpha())
        if prefix != "K":
            continue

        # COM pin
        com_pin = next((p for p in comp.pins if p.name.upper() == "COM"), None)
        if com_pin is None:
            com_pin = comp.get_pin("1")
        com_net = com_pin.net if com_pin else None

        if com_net is None:
            violations.append(DRCViolation(
                rule="relay_polarity",
                message=f"{comp.ref}: COM pin has no net — cannot determine polarity",
                severity=Severity.WARNING,
                ref=comp.ref,
            ))
            continue

        upper_com = com_net.upper().strip()
        if upper_com not in _GND_RELAY_NAMES and upper_com not in _VIN_RELAY_NAMES:
            if not upper_com.startswith(("VIN", "+", "V")):
                violations.append(DRCViolation(
                    rule="relay_polarity",
                    message=(
                        f"{comp.ref}: COM net '{com_net}' is ambiguous — "
                        "expected VIN/GND pattern"
                    ),
                    severity=Severity.WARNING,
                    ref=comp.ref,
                ))

        # Check flyback diode on coil
        coil_pin = next(
            (p for p in comp.pins if p.name.upper() in ("COIL-", "COIL_MINUS")),
            None,
        )
        if coil_pin is None:
            coil_pin = comp.get_pin("2")
        coil_net = coil_pin.net if coil_pin else None

        if coil_net:
            coil_refs = net_to_refs.get(coil_net, set())
            d_refs = [r for r in coil_refs if "".join(c for c in r if c.isalpha()) == "D"]
            if not d_refs:
                violations.append(DRCViolation(
                    rule="relay_flyback",
                    message=f"{comp.ref}: no flyback diode on coil net '{coil_net}'",
                    severity=Severity.WARNING,
                    ref=comp.ref,
                ))

            # Check coil pins not shorted
            coil_plus = next(
                (p for p in comp.pins if p.name.upper() in ("COIL+", "COIL_PLUS")),
                None,
            )
            if coil_plus is None:
                coil_plus = comp.get_pin("5")
            if coil_plus and coil_plus.net and coil_plus.net == coil_net:
                violations.append(DRCViolation(
                    rule="relay_coil_short",
                    message=f"{comp.ref}: COIL+ and COIL- on same net '{coil_net}'",
                    severity=Severity.ERROR,
                    ref=comp.ref,
                ))

        # Check GPIO traces to MCU
        if coil_net:
            coil_refs = net_to_refs.get(coil_net, set())
            q_refs = [r for r in coil_refs if "".join(c for c in r if c.isalpha()) == "Q"]
            if not q_refs:
                violations.append(DRCViolation(
                    rule="relay_driver",
                    message=f"{comp.ref}: no transistor driver on coil net '{coil_net}'",
                    severity=Severity.WARNING,
                    ref=comp.ref,
                ))

    return violations
