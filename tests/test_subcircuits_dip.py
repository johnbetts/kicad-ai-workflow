"""Tests for DIP switch subcircuit generator."""

from __future__ import annotations

from kicad_pipeline.schematic.subcircuits import dip_switch_address


def test_dip_switch_address_default() -> None:
    """dip_switch_address produces expected components and nets."""
    result = dip_switch_address("SW1")
    # 1 DIP switch + 4 resistors = 5 components
    assert len(result.components) == 5
    assert result.components[0].ref == "SW1"
    assert "DIP" in result.components[0].value


def test_dip_switch_address_bit_count() -> None:
    """Custom bit_count produces the right number of resistors."""
    result = dip_switch_address("SW2", bit_count=8)
    # 1 switch + 8 resistors
    assert len(result.components) == 9


def test_dip_switch_address_custom_nets() -> None:
    """Custom target_nets are used in the subcircuit."""
    nets = ("A0", "A1", "A2", "A3")
    result = dip_switch_address("SW1", target_nets=nets)
    net_names = {n.name for n in result.nets}
    for name in nets:
        assert name in net_names


def test_dip_switch_address_has_warning() -> None:
    """Description includes a warning about one switch at a time."""
    result = dip_switch_address("SW1")
    assert "WARNING" in result.description
    assert "ONE" in result.description.upper()


def test_dip_switch_address_series_resistance() -> None:
    """Series resistors have the specified value."""
    result = dip_switch_address("SW1", series_resistance=1000.0)
    resistors = [c for c in result.components if c.ref.startswith("R_")]
    assert len(resistors) == 4
    for r in resistors:
        assert "1k" in r.value or "1000" in r.value
