"""Tests for kicad_pipeline.requirements.pin_budget."""

from __future__ import annotations

import pytest

from kicad_pipeline.exceptions import RequirementsError
from kicad_pipeline.models.requirements import MCUPinMap, PinAssignment, PinFunction
from kicad_pipeline.requirements.pin_budget import PinBudgetTracker, validate_pin_map

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(total_pins: int = 40) -> PinBudgetTracker:
    """Return a fresh PinBudgetTracker for tests."""
    return PinBudgetTracker(mcu_ref="U1", total_pins=total_pins)


# ---------------------------------------------------------------------------
# PinBudgetTracker tests
# ---------------------------------------------------------------------------


def test_assign_pin() -> None:
    """assign() records the pin assignment correctly."""
    tracker = _make_tracker()
    tracker.assign("GPIO4", "IO4", PinFunction.GPIO, "LED_CTRL")
    assignment = tracker.get_assignment("GPIO4")
    assert assignment is not None
    assert assignment.pin_number == "GPIO4"
    assert assignment.pin_name == "IO4"
    assert assignment.function == PinFunction.GPIO
    assert assignment.net == "LED_CTRL"
    assert assignment.mcu_ref == "U1"


def test_assign_duplicate_raises() -> None:
    """assign() raises RequirementsError when the same pin is assigned twice."""
    tracker = _make_tracker()
    tracker.assign("GPIO4", "IO4", PinFunction.GPIO, "LED_CTRL")
    with pytest.raises(RequirementsError):
        tracker.assign("GPIO4", "IO4", PinFunction.GPIO, "OTHER_NET")


def test_assign_if_free_success() -> None:
    """assign_if_free() returns True when the pin is unassigned."""
    tracker = _make_tracker()
    result = tracker.assign_if_free("GPIO5", "IO5", PinFunction.SPI_CLK, "SPI_CLK")
    assert result is True
    assert tracker.is_assigned("GPIO5")


def test_assign_if_free_taken() -> None:
    """assign_if_free() returns False when the pin is already assigned."""
    tracker = _make_tracker()
    tracker.assign("GPIO5", "IO5", PinFunction.SPI_CLK, "SPI_CLK")
    result = tracker.assign_if_free("GPIO5", "IO5", PinFunction.GPIO, "OTHER_NET")
    assert result is False
    # Original assignment must be untouched
    assignment = tracker.get_assignment("GPIO5")
    assert assignment is not None
    assert assignment.net == "SPI_CLK"


def test_is_assigned() -> None:
    """is_assigned() returns True after assignment, False before."""
    tracker = _make_tracker()
    assert tracker.is_assigned("GPIO6") is False
    tracker.assign("GPIO6", "IO6", PinFunction.I2C_SDA, "I2C_SDA")
    assert tracker.is_assigned("GPIO6") is True


def test_pins_by_function() -> None:
    """pins_by_function() returns only assignments with the matching function."""
    tracker = _make_tracker()
    tracker.assign("GPIO4", "IO4", PinFunction.GPIO, "LED_CTRL")
    tracker.assign("GPIO5", "IO5", PinFunction.GPIO, "BTN_IN")
    tracker.assign("GPIO21", "IO21", PinFunction.I2C_SDA, "I2C_SDA")

    gpio_pins = tracker.pins_by_function(PinFunction.GPIO)
    assert len(gpio_pins) == 2
    nets = {a.net for a in gpio_pins}
    assert "LED_CTRL" in nets
    assert "BTN_IN" in nets

    i2c_pins = tracker.pins_by_function(PinFunction.I2C_SDA)
    assert len(i2c_pins) == 1
    assert i2c_pins[0].net == "I2C_SDA"


def test_build_returns_pin_map() -> None:
    """build() returns an MCUPinMap with the correct mcu_ref."""
    tracker = _make_tracker()
    tracker.assign("GPIO4", "IO4", PinFunction.GPIO, "LED_CTRL")
    pin_map = tracker.build()
    assert isinstance(pin_map, MCUPinMap)
    assert pin_map.mcu_ref == "U1"
    assert len(pin_map.assignments) == 1
    assert pin_map.assignments[0].pin_number == "GPIO4"


def test_free_gpio_count() -> None:
    """free_gpio_count() decreases as pins are assigned."""
    tracker = _make_tracker(total_pins=40)
    assert tracker.free_gpio_count() == 40
    tracker.assign("GPIO4", "IO4", PinFunction.GPIO, "LED_CTRL")
    assert tracker.free_gpio_count() == 39
    tracker.assign("GPIO5", "IO5", PinFunction.GPIO, "BTN_IN")
    assert tracker.free_gpio_count() == 38


def test_get_assignment_found() -> None:
    """get_assignment() returns the correct PinAssignment for an assigned pin."""
    tracker = _make_tracker()
    tracker.assign("GPIO7", "IO7", PinFunction.ADC, "SENSOR_ADC", notes="analog input")
    assignment = tracker.get_assignment("GPIO7")
    assert isinstance(assignment, PinAssignment)
    assert assignment.pin_number == "GPIO7"
    assert assignment.function == PinFunction.ADC
    assert assignment.notes == "analog input"


def test_get_assignment_not_found() -> None:
    """get_assignment() returns None for an unassigned pin."""
    tracker = _make_tracker()
    assert tracker.get_assignment("GPIO99") is None


# ---------------------------------------------------------------------------
# validate_pin_map tests
# ---------------------------------------------------------------------------


def _make_assignment(
    pin_number: str,
    function: PinFunction,
    net: str,
    mcu_ref: str = "U1",
    pin_name: str = "IO0",
) -> PinAssignment:
    """Helper to construct a PinAssignment for test fixtures."""
    return PinAssignment(
        mcu_ref=mcu_ref,
        pin_number=pin_number,
        pin_name=pin_name,
        function=function,
        net=net,
    )


def test_validate_pin_map_valid() -> None:
    """validate_pin_map() returns empty warnings for a valid pin map."""
    assignments = (
        _make_assignment("GPIO4", PinFunction.GPIO, "LED_CTRL"),
        _make_assignment("GPIO5", PinFunction.SPI_CLK, "SPI_CLK"),
        _make_assignment("GPIO21", PinFunction.I2C_SDA, "I2C_SDA"),
        _make_assignment("GPIO22", PinFunction.I2C_SCL, "I2C_SCL"),
        # USB pair — both present
        _make_assignment("GPIO19", PinFunction.USB_DM, "USB_DM"),
        _make_assignment("GPIO20", PinFunction.USB_DP, "USB_DP"),
    )
    pin_map = MCUPinMap(mcu_ref="U1", assignments=assignments, unassigned_gpio=())
    warnings = validate_pin_map(pin_map)
    assert warnings == []
