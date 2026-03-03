"""Electrical design rule check (ERC) validation for PCB designs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.validation.drc import DRCViolation, Severity

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

_GND_NET_NAMES: frozenset[str] = frozenset({"GND", "AGND", "DGND"})


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
    """Emit INFO if an IC from requirements has no capacitor in the same feature."""
    if requirements is None:
        return []

    violations: list[DRCViolation] = []

    # Build a set of component refs that are capacitors (ref starts with "C").
    cap_refs = {comp.ref for comp in requirements.components if comp.ref.startswith("C")}

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
