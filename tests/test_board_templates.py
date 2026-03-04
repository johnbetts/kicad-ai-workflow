"""Tests for the board template registry."""

from __future__ import annotations

import pytest

from kicad_pipeline.exceptions import ConfigurationError
from kicad_pipeline.pcb.board_templates import (
    get_template,
    list_templates,
    template_to_mechanical_constraints,
)


def test_list_templates_returns_known_names() -> None:
    """list_templates includes the built-in template names."""
    names = list_templates()
    assert "ARDUINO_UNO" in names
    assert "GENERIC_50X50" in names
    assert "RPI_HAT" in names


def test_get_template_rpi_hat() -> None:
    """get_template returns the RPi HAT template with correct dimensions."""
    tmpl = get_template("RPI_HAT")
    assert tmpl.board_width_mm == 65.0
    assert tmpl.board_height_mm == 56.5
    assert len(tmpl.mounting_holes) == 4


def test_get_template_case_insensitive() -> None:
    """get_template is case-insensitive."""
    tmpl = get_template("rpi_hat")
    assert tmpl.name == "RPI_HAT"


def test_get_template_unknown_raises() -> None:
    """get_template raises ConfigurationError for unknown names."""
    with pytest.raises(ConfigurationError):
        get_template("NONEXISTENT_BOARD")


def test_get_template_arduino_uno() -> None:
    """Arduino Uno template has correct dimensions."""
    tmpl = get_template("ARDUINO_UNO")
    assert tmpl.board_width_mm == pytest.approx(68.6)
    assert tmpl.board_height_mm == pytest.approx(53.3)
    assert len(tmpl.mounting_holes) == 4


def test_get_template_generic_50x50() -> None:
    """Generic 50x50 template has correct dimensions."""
    tmpl = get_template("GENERIC_50X50")
    assert tmpl.board_width_mm == 50.0
    assert tmpl.board_height_mm == 50.0


def test_template_to_mechanical_constraints() -> None:
    """template_to_mechanical_constraints produces valid constraints."""
    tmpl = get_template("RPI_HAT")
    mc = template_to_mechanical_constraints(tmpl)
    assert mc.board_width_mm == 65.0
    assert mc.board_height_mm == 56.5
    assert len(mc.mounting_hole_positions) == 4
    assert mc.mounting_hole_diameter_mm == pytest.approx(2.7)


def test_rpi_hat_has_fixed_components() -> None:
    """RPi HAT template has a fixed GPIO header component."""
    tmpl = get_template("RPI_HAT")
    assert len(tmpl.fixed_components) >= 1
    assert tmpl.fixed_components[0].ref_pattern == "J1"


def test_board_template_is_frozen() -> None:
    """BoardTemplate is immutable."""
    tmpl = get_template("RPI_HAT")
    with pytest.raises((AttributeError, TypeError)):
        tmpl.name = "HACKED"  # type: ignore[misc]
