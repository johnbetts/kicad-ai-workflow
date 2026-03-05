"""Tests for the board template registry."""

from __future__ import annotations

import pytest

from kicad_pipeline.exceptions import ConfigurationError
from kicad_pipeline.models.requirements import MechanicalConstraints
from kicad_pipeline.pcb.board_templates import (
    detect_template,
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
    assert tmpl.board_height_mm == 56.0
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
    assert mc.board_height_mm == 56.0
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


def test_rpi_hat_mounting_holes_correct_positions() -> None:
    """RPi HAT mounting holes match the official Raspberry Pi mech spec.

    Per the RPi spec, the right-side holes are at x = 3.5 + 58.0 = 61.5mm.
    """
    tmpl = get_template("RPI_HAT")
    positions = [(h.x_mm, h.y_mm) for h in tmpl.mounting_holes]
    # Left holes at x=3.5
    assert (3.5, 3.5) in positions
    assert (3.5, 52.5) in positions
    # Right holes at x=61.5 (not 58.0)
    assert (61.5, 3.5) in positions
    assert (61.5, 52.5) in positions


def test_rpi_hat_corner_radius() -> None:
    """RPi HAT template has a non-zero corner radius."""
    tmpl = get_template("RPI_HAT")
    assert tmpl.corner_radius_mm == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# detect_template tests
# ---------------------------------------------------------------------------


def test_detect_template_from_explicit_field() -> None:
    """detect_template uses the board_template field when present."""
    mc = MechanicalConstraints(
        board_width_mm=100.0,
        board_height_mm=100.0,
        board_template="RPI_HAT",
    )
    tmpl = detect_template(mc)
    assert tmpl is not None
    assert tmpl.name == "RPI_HAT"


def test_detect_template_from_notes() -> None:
    """detect_template finds RPi HAT from notes keywords."""
    mc = MechanicalConstraints(
        board_width_mm=100.0,
        board_height_mm=100.0,
        notes="This is a Raspberry Pi HAT board",
    )
    tmpl = detect_template(mc)
    assert tmpl is not None
    assert tmpl.name == "RPI_HAT"


def test_detect_template_from_notes_arduino() -> None:
    """detect_template finds Arduino Uno from notes keywords."""
    mc = MechanicalConstraints(
        board_width_mm=100.0,
        board_height_mm=100.0,
        notes="Arduino Uno shield",
    )
    tmpl = detect_template(mc)
    assert tmpl is not None
    assert tmpl.name == "ARDUINO_UNO"


def test_detect_template_from_dimensions() -> None:
    """detect_template matches by dimensions within tolerance."""
    mc = MechanicalConstraints(
        board_width_mm=65.2,
        board_height_mm=56.3,
    )
    tmpl = detect_template(mc)
    assert tmpl is not None
    assert tmpl.name == "RPI_HAT"


def test_detect_template_no_match() -> None:
    """detect_template returns None when nothing matches."""
    mc = MechanicalConstraints(
        board_width_mm=200.0,
        board_height_mm=200.0,
    )
    tmpl = detect_template(mc)
    assert tmpl is None


def test_detect_template_none_input() -> None:
    """detect_template returns None for None input."""
    assert detect_template(None) is None
