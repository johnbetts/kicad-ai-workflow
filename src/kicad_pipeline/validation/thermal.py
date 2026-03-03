"""Thermal analysis for PCB component power dissipation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.constants import THERMAL_WARNING_MW
from kicad_pipeline.validation.drc import DRCViolation, Severity

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THERMAL_HIGH_POWER_MW: float = THERMAL_WARNING_MW

# Power estimates by reference designator prefix (mW)
_POWER_BY_PREFIX: dict[str, float] = {
    "U": 200.0,
    "Q": 100.0,
    "D": 50.0,
    "R": 25.0,
    "C": 5.0,
    "L": 20.0,
    "J": 10.0,
    "SW": 5.0,
}

_DEFAULT_POWER_MW: float = 50.0

# LDO value strings that indicate higher dissipation
_LDO_VALUE_KEYWORDS: tuple[str, ...] = ("AMS", "AP2112")
_LDO_POWER_MW: float = 300.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentThermal:
    """Thermal estimate for a single component."""

    ref: str
    estimated_power_mw: float
    flag_high: bool


@dataclass(frozen=True)
class ThermalReport:
    """Result of thermal analysis."""

    component_thermals: tuple[ComponentThermal, ...]
    violations: tuple[DRCViolation, ...]

    @property
    def passed(self) -> bool:
        """True if there are no ERROR-severity violations."""
        return all(v.severity != Severity.ERROR for v in self.violations)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def estimate_power_mw(ref: str, value: str) -> float:
    """Estimate component power dissipation in milliwatts.

    Args:
        ref: Component reference designator (e.g. ``"U1"``, ``"R5"``).
        value: Component value string (e.g. ``"AMS1117-3.3"``).

    Returns:
        Estimated power dissipation in milliwatts.
    """
    # LDO regulator check: U prefix + known LDO value keyword
    if ref.startswith("U"):
        for keyword in _LDO_VALUE_KEYWORDS:
            if keyword in value:
                return _LDO_POWER_MW

    # Two-character prefix check first (e.g. "SW")
    two_char = ref[:2] if len(ref) >= 2 else ""
    if two_char in _POWER_BY_PREFIX:
        return _POWER_BY_PREFIX[two_char]

    # Single-character prefix
    one_char = ref[:1]
    if one_char in _POWER_BY_PREFIX:
        return _POWER_BY_PREFIX[one_char]

    return _DEFAULT_POWER_MW


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_thermal_checks(
    pcb: PCBDesign,
    requirements: ProjectRequirements | None = None,
) -> ThermalReport:
    """Run thermal analysis on all footprints in *pcb*.

    Args:
        pcb: The PCB design to analyse.
        requirements: Optional project requirements (currently unused but
            reserved for future per-component power overrides).

    Returns:
        A :class:`ThermalReport` with per-component estimates and violations.
    """
    component_thermals: list[ComponentThermal] = []
    violations: list[DRCViolation] = []

    for fp in pcb.footprints:
        power_mw = estimate_power_mw(fp.ref, fp.value)
        flag_high = power_mw > THERMAL_HIGH_POWER_MW

        component_thermals.append(
            ComponentThermal(
                ref=fp.ref,
                estimated_power_mw=power_mw,
                flag_high=flag_high,
            )
        )

        if flag_high:
            violations.append(
                DRCViolation(
                    rule="high_power_component",
                    message=(
                        f"Component {fp.ref} estimated {power_mw:.0f}mW exceeds"
                        f" threshold {THERMAL_HIGH_POWER_MW:.0f}mW"
                        " - verify thermal management"
                    ),
                    severity=Severity.WARNING,
                    ref=fp.ref,
                )
            )

    return ThermalReport(
        component_thermals=tuple(component_thermals),
        violations=tuple(violations),
    )
