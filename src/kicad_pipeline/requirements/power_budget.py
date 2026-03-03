"""Power budget calculator and regulator analysis for PCB designs.

This module calculates and validates power budgets, estimating per-component
consumption and checking regulator thermal and margin headroom.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.constants import THERMAL_WARNING_MW
from kicad_pipeline.exceptions import RequirementsError

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import PowerRail

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known typical currents (mA) for common component categories
# ---------------------------------------------------------------------------

TYPICAL_CURRENTS_MA: dict[str, float] = {
    "mcu_module": 240.0,      # ESP32 active
    "ethernet": 132.0,        # W5500
    "usb_uart": 10.0,
    "ldo": 1.0,               # quiescent
    "led": 10.0,
    "resistor": 0.1,
    "capacitor": 0.0,
    "transistor_npn": 0.1,
    "diode_switching": 0.1,
    "diode_esd": 0.0,
    "buzzer": 30.0,
    "switch": 0.0,
    "connector_usb": 0.0,
    "connector_rj45": 0.0,
    "connector_header": 0.0,
}

_LOW_MARGIN_THRESHOLD_PCT: float = 20.0
"""Warn if regulator load margin falls below this percentage."""

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentPower:
    """Power consumption estimate for a single component."""

    ref: str
    """Component reference designator (e.g. 'U1')."""

    description: str
    """Human-readable description (category hint used for estimation)."""

    supply_voltage: float
    """Supply voltage in volts."""

    current_ma: float
    """Typical current draw in milliamps."""

    power_mw: float
    """Power in milliwatts (voltage * current_ma)."""


@dataclass(frozen=True)
class RegulatorAnalysis:
    """Analysis of a voltage regulator for thermal and margin concerns."""

    ref: str
    """Component reference designator (e.g. 'U1')."""

    vin: float
    """Input voltage in volts."""

    vout: float
    """Output (regulated) voltage in volts."""

    iout_max_ma: float
    """Rated maximum output current in milliamps."""

    load_current_ma: float
    """Actual load current in milliamps."""

    dropout_v: float
    """Dropout voltage (vin - vout) in volts."""

    power_dissipation_mw: float
    """Power dissipated in the regulator: (vin - vout) * load_current_ma."""

    margin_pct: float
    """Headroom as a percentage: (iout_max - load) / iout_max * 100."""

    thermal_warning: bool
    """True if power_dissipation_mw exceeds THERMAL_WARNING_MW."""

    warnings: tuple[str, ...]
    """Any warning messages generated during analysis."""


@dataclass(frozen=True)
class PowerBudgetReport:
    """Complete power budget analysis report for a PCB design."""

    rails: tuple[PowerRail, ...]
    """Power rails defined in the design."""

    component_power: tuple[ComponentPower, ...]
    """Per-component power consumption estimates."""

    regulator_analyses: tuple[RegulatorAnalysis, ...]
    """Analysis results for all regulators."""

    total_current_ma: float
    """Sum of all component current draws in milliamps."""

    total_power_mw: float
    """Sum of all component power draws in milliwatts."""

    warnings: tuple[str, ...]
    """Aggregated warnings from all analyses."""

    errors: tuple[str, ...]
    """Aggregated errors from all analyses."""

    @property
    def has_errors(self) -> bool:
        """True if there are any errors in the report."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """True if there are any warnings in the report."""
        return len(self.warnings) > 0


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def estimate_component_power(
    ref: str,
    category: str,
    supply_voltage: float,
    typical_current_ma: float,
) -> ComponentPower:
    """Create a ComponentPower estimate for a component.

    If *typical_current_ma* is greater than zero it is used directly; otherwise
    the value is looked up from :data:`TYPICAL_CURRENTS_MA` by *category*.  If
    the category is not in the lookup table a current of ``0.0 mA`` is assumed
    and a debug message is logged.

    Args:
        ref: Component reference designator (e.g. ``'U1'``).
        category: Component category hint (e.g. ``'mcu_module'``, ``'ethernet'``).
        supply_voltage: Supply voltage in volts.
        typical_current_ma: Typical current in milliamps.  Pass ``0.0`` to use
            the category lookup table.

    Returns:
        A :class:`ComponentPower` instance with computed ``power_mw``.
    """
    if typical_current_ma > 0.0:
        current_ma = typical_current_ma
    else:
        current_ma = TYPICAL_CURRENTS_MA.get(category, 0.0)
        if category not in TYPICAL_CURRENTS_MA:
            log.debug(
                "Component %s has unknown category %r; assuming 0 mA",
                ref,
                category,
            )

    power_mw = supply_voltage * current_ma
    log.debug(
        "Component %s (%s): %.2f V @ %.2f mA = %.2f mW",
        ref,
        category,
        supply_voltage,
        current_ma,
        power_mw,
    )
    return ComponentPower(
        ref=ref,
        description=category,
        supply_voltage=supply_voltage,
        current_ma=current_ma,
        power_mw=power_mw,
    )


def analyze_regulator(
    ref: str,
    vin: float,
    vout: float,
    iout_max_ma: float,
    load_current_ma: float,
    dropout_v: float = 0.3,
) -> RegulatorAnalysis:
    """Analyze a voltage regulator for thermal and margin concerns.

    Checks performed:

    * **Input voltage adequacy** — raises :class:`~kicad_pipeline.exceptions.RequirementsError`
      if ``vin < vout + dropout_v``.
    * **Load margin** — warns if headroom is below 20 %.
    * **Thermal dissipation** — warns if ``(vin - vout) * load_current_ma``
      exceeds :data:`~kicad_pipeline.constants.THERMAL_WARNING_MW`.

    Args:
        ref: Component reference designator (e.g. ``'U1'``).
        vin: Input voltage in volts.
        vout: Output (regulated) voltage in volts.
        iout_max_ma: Rated maximum output current in milliamps.
        load_current_ma: Actual load current in milliamps.
        dropout_v: Minimum required difference between *vin* and *vout* for the
            regulator to operate correctly (volts).  Defaults to ``0.3 V``.

    Returns:
        A :class:`RegulatorAnalysis` instance.

    Raises:
        RequirementsError: If *vin* is insufficient for the regulator to
            regulate (``vin < vout + dropout_v``).
    """
    if vin < vout + dropout_v:
        raise RequirementsError(
            f"Regulator {ref}: vin ({vin} V) is less than vout + dropout "
            f"({vout} V + {dropout_v} V = {vout + dropout_v} V)"
        )

    actual_dropout_v = vin - vout
    power_dissipation_mw = actual_dropout_v * load_current_ma
    margin_pct = (iout_max_ma - load_current_ma) / iout_max_ma * 100.0
    thermal_warning = power_dissipation_mw > THERMAL_WARNING_MW

    warnings: list[str] = []

    if margin_pct < _LOW_MARGIN_THRESHOLD_PCT:
        msg = (
            f"Regulator {ref}: load margin is {margin_pct:.1f}% "
            f"(load {load_current_ma:.1f} mA of {iout_max_ma:.1f} mA max)"
        )
        log.warning(msg)
        warnings.append(msg)

    if thermal_warning:
        msg = (
            f"Regulator {ref}: power dissipation {power_dissipation_mw:.1f} mW "
            f"exceeds thermal warning threshold {THERMAL_WARNING_MW:.0f} mW"
        )
        log.warning(msg)
        warnings.append(msg)

    log.debug(
        "Regulator %s: vin=%.2fV vout=%.2fV dropout=%.2fV "
        "load=%.1f/%.1f mA margin=%.1f%% dissipation=%.1f mW",
        ref,
        vin,
        vout,
        actual_dropout_v,
        load_current_ma,
        iout_max_ma,
        margin_pct,
        power_dissipation_mw,
    )

    return RegulatorAnalysis(
        ref=ref,
        vin=vin,
        vout=vout,
        iout_max_ma=iout_max_ma,
        load_current_ma=load_current_ma,
        dropout_v=actual_dropout_v,
        power_dissipation_mw=power_dissipation_mw,
        margin_pct=margin_pct,
        thermal_warning=thermal_warning,
        warnings=tuple(warnings),
    )


def calculate_power_budget(
    rails: list[PowerRail],
    components: list[tuple[str, str, float]],
    regulators: list[tuple[str, float, float, float, float]],
) -> PowerBudgetReport:
    """Calculate a complete power budget for a PCB design.

    Args:
        rails: Power rails defined in the design.
        components: List of ``(ref, category, supply_voltage)`` tuples.  A
            current of ``0.0 mA`` is passed to :func:`estimate_component_power`
            so that category-based lookup is used automatically.
        regulators: List of ``(ref, vin, vout, iout_max_ma, dropout_v)`` tuples
            describing each linear regulator.

    Returns:
        A :class:`PowerBudgetReport` with aggregated totals, warnings, and
        per-component / per-regulator breakdown.
    """
    component_powers: list[ComponentPower] = []
    for ref, category, supply_voltage in components:
        cp = estimate_component_power(
            ref=ref,
            category=category,
            supply_voltage=supply_voltage,
            typical_current_ma=0.0,
        )
        component_powers.append(cp)

    regulator_analyses: list[RegulatorAnalysis] = []
    errors: list[str] = []
    for ref, vin, vout, iout_max_ma, dropout_v in regulators:
        try:
            ra = analyze_regulator(
                ref=ref,
                vin=vin,
                vout=vout,
                iout_max_ma=iout_max_ma,
                load_current_ma=sum(
                    cp.current_ma
                    for cp in component_powers
                    if cp.supply_voltage == vout
                ),
                dropout_v=dropout_v,
            )
            regulator_analyses.append(ra)
        except RequirementsError as exc:
            err_msg = str(exc)
            log.error(err_msg)
            errors.append(err_msg)

    total_current_ma = sum(cp.current_ma for cp in component_powers)
    total_power_mw = sum(cp.power_mw for cp in component_powers)

    all_warnings: list[str] = []
    for ra in regulator_analyses:
        all_warnings.extend(ra.warnings)

    log.info(
        "Power budget: %.1f mA total, %.1f mW total, %d warnings, %d errors",
        total_current_ma,
        total_power_mw,
        len(all_warnings),
        len(errors),
    )

    return PowerBudgetReport(
        rails=tuple(rails),
        component_power=tuple(component_powers),
        regulator_analyses=tuple(regulator_analyses),
        total_current_ma=total_current_ma,
        total_power_mw=total_power_mw,
        warnings=tuple(all_warnings),
        errors=tuple(errors),
    )
