"""Tests for kicad_pipeline.requirements.power_budget."""

from __future__ import annotations

import pytest

from kicad_pipeline.exceptions import RequirementsError
from kicad_pipeline.models.requirements import PowerRail
from kicad_pipeline.requirements.power_budget import (
    TYPICAL_CURRENTS_MA,
    PowerBudgetReport,
    analyze_regulator,
    calculate_power_budget,
    estimate_component_power,
)

# ---------------------------------------------------------------------------
# estimate_component_power
# ---------------------------------------------------------------------------


def test_estimate_component_power_led() -> None:
    """LED at 3.3 V, 10 mA → 33 mW."""
    cp = estimate_component_power(
        ref="D1",
        category="led",
        supply_voltage=3.3,
        typical_current_ma=10.0,
    )
    assert cp.ref == "D1"
    assert cp.current_ma == pytest.approx(10.0)
    assert cp.power_mw == pytest.approx(3.3 * 10.0)


def test_estimate_component_power_mcu() -> None:
    """MCU at 3.3 V uses TYPICAL_CURRENTS_MA['mcu_module'] when typical_current_ma=0."""
    cp = estimate_component_power(
        ref="U1",
        category="mcu_module",
        supply_voltage=3.3,
        typical_current_ma=0.0,
    )
    expected_ma = TYPICAL_CURRENTS_MA["mcu_module"]
    assert cp.current_ma == pytest.approx(expected_ma)
    assert cp.power_mw == pytest.approx(3.3 * expected_ma)


def test_estimate_component_power_unknown_category() -> None:
    """Unknown category → 0 mA assumed, no exception raised."""
    cp = estimate_component_power(
        ref="X1",
        category="crystal_oscillator",
        supply_voltage=3.3,
        typical_current_ma=0.0,
    )
    assert cp.current_ma == pytest.approx(0.0)
    assert cp.power_mw == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# analyze_regulator
# ---------------------------------------------------------------------------


def test_analyze_regulator_healthy() -> None:
    """Regulator with good margin and low dissipation: no warnings."""
    ra = analyze_regulator(
        ref="U2",
        vin=5.0,
        vout=3.3,
        iout_max_ma=500.0,
        load_current_ma=100.0,  # 80% margin
    )
    assert ra.ref == "U2"
    assert ra.margin_pct == pytest.approx(80.0)
    assert not ra.thermal_warning
    assert len(ra.warnings) == 0


def test_analyze_regulator_low_margin() -> None:
    """Regulator at 95% load gets a low-margin warning."""
    ra = analyze_regulator(
        ref="U3",
        vin=5.0,
        vout=3.3,
        iout_max_ma=500.0,
        load_current_ma=475.0,  # 5% margin
    )
    assert ra.margin_pct == pytest.approx(5.0)
    assert any("margin" in w.lower() for w in ra.warnings)


def test_analyze_regulator_thermal_warning() -> None:
    """High dropout * high current triggers thermal warning."""
    # vin=12, vout=3.3 → dropout=8.7 V; load=100 mA → 870 mW > 500 mW threshold
    ra = analyze_regulator(
        ref="U4",
        vin=12.0,
        vout=3.3,
        iout_max_ma=1000.0,
        load_current_ma=100.0,
    )
    assert ra.thermal_warning is True
    assert ra.power_dissipation_mw == pytest.approx((12.0 - 3.3) * 100.0)
    assert any("thermal" in w.lower() or "dissipation" in w.lower() for w in ra.warnings)


def test_analyze_regulator_insufficient_vin() -> None:
    """vin < vout + dropout raises RequirementsError."""
    with pytest.raises(RequirementsError, match="vin"):
        analyze_regulator(
            ref="U5",
            vin=3.0,
            vout=3.3,
            iout_max_ma=500.0,
            load_current_ma=100.0,
            dropout_v=0.3,
        )


# ---------------------------------------------------------------------------
# calculate_power_budget
# ---------------------------------------------------------------------------


def test_calculate_power_budget_basic() -> None:
    """End-to-end budget with 2 rails and 3 components."""
    rails = [
        PowerRail(name="+3V3", voltage=3.3, current_ma=500.0, source_ref="U1"),
        PowerRail(name="+5V", voltage=5.0, current_ma=200.0, source_ref="J1"),
    ]
    components = [
        ("U2", "mcu_module", 3.3),
        ("D1", "led", 3.3),
        ("U3", "ethernet", 3.3),
    ]
    # Regulator: 5V → 3.3V, rated 1 A
    regulators = [
        ("U1", 5.0, 3.3, 1000.0, 0.3),
    ]

    report = calculate_power_budget(rails, components, regulators)

    assert len(report.rails) == 2
    assert len(report.component_power) == 3
    assert len(report.regulator_analyses) == 1

    expected_total_ma = (
        TYPICAL_CURRENTS_MA["mcu_module"]
        + TYPICAL_CURRENTS_MA["led"]
        + TYPICAL_CURRENTS_MA["ethernet"]
    )
    assert report.total_current_ma == pytest.approx(expected_total_ma)
    assert report.total_power_mw == pytest.approx(3.3 * expected_total_ma)


def test_power_budget_report_has_errors() -> None:
    """has_errors property is True when the errors tuple is non-empty."""
    report = PowerBudgetReport(
        rails=(),
        component_power=(),
        regulator_analyses=(),
        total_current_ma=0.0,
        total_power_mw=0.0,
        warnings=(),
        errors=("something went wrong",),
    )
    assert report.has_errors is True
    assert report.has_warnings is False


def test_power_budget_report_has_warnings() -> None:
    """has_warnings property is True when the warnings tuple is non-empty."""
    report = PowerBudgetReport(
        rails=(),
        component_power=(),
        regulator_analyses=(),
        total_current_ma=0.0,
        total_power_mw=0.0,
        warnings=("thermal concern",),
        errors=(),
    )
    assert report.has_warnings is True
    assert report.has_errors is False


def test_power_budget_total_current() -> None:
    """total_current_ma sums all component currents correctly."""
    components = [
        ("R1", "resistor", 3.3),   # 0.1 mA
        ("D1", "led", 3.3),        # 10.0 mA
        ("C1", "capacitor", 3.3),  # 0.0 mA
    ]
    report = calculate_power_budget(rails=[], components=components, regulators=[])
    expected = (
        TYPICAL_CURRENTS_MA["resistor"]
        + TYPICAL_CURRENTS_MA["led"]
        + TYPICAL_CURRENTS_MA["capacitor"]
    )
    assert report.total_current_ma == pytest.approx(expected)


def test_calculate_power_budget_bad_regulator_captured_as_error() -> None:
    """A regulator with insufficient vin results in an error entry, not a crash."""
    components = [("R1", "resistor", 3.3)]
    regulators = [("U_BAD", 2.0, 3.3, 500.0, 0.3)]  # vin too low

    report = calculate_power_budget(rails=[], components=components, regulators=regulators)
    assert report.has_errors is True
    assert len(report.regulator_analyses) == 0
