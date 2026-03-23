"""Tests for kicad_pipeline.models.requirements — ProjectRequirements, Component, etc."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.requirements import (
    BoardContext,
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinFunction,
    PinType,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    ProjectRequirements,
    Recommendation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_component(ref: str = "R1", value: str = "10k") -> Component:
    return Component(
        ref=ref,
        value=value,
        footprint="R_0805",
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, net="NET_A"),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, net="GND"),
        ),
    )


def _make_requirements(
    components: tuple[Component, ...] | None = None,
    nets: tuple[Net, ...] | None = None,
) -> ProjectRequirements:
    if components is None:
        components = (_make_component(),)
    if nets is None:
        nets = (
            Net("NET_A", (NetConnection("R1", "1"),)),
            Net("GND", (NetConnection("R1", "2"),)),
        )
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        features=(),
        components=components,
        nets=nets,
    )


# ---------------------------------------------------------------------------
# Component.get_pin
# ---------------------------------------------------------------------------


def test_component_get_pin_found() -> None:
    """get_pin returns matching Pin when number exists."""
    comp = _make_component()
    pin = comp.get_pin("1")
    assert pin is not None
    assert pin.number == "1"


def test_component_get_pin_not_found() -> None:
    """get_pin returns None for unknown pin number."""
    comp = _make_component()
    assert comp.get_pin("99") is None


def test_component_get_pin_empty_pins() -> None:
    """get_pin returns None for component with no pins."""
    comp = Component(ref="R1", value="10k", footprint="R_0805")
    assert comp.get_pin("1") is None


# ---------------------------------------------------------------------------
# ProjectRequirements.get_component
# ---------------------------------------------------------------------------


def test_get_component_found() -> None:
    """get_component returns component matching ref."""
    reqs = _make_requirements()
    comp = reqs.get_component("R1")
    assert comp is not None
    assert comp.ref == "R1"


def test_get_component_not_found() -> None:
    """get_component returns None for unknown ref."""
    reqs = _make_requirements()
    assert reqs.get_component("U99") is None


def test_get_component_multiple() -> None:
    """get_component finds correct one among several."""
    c1 = _make_component("R1", "10k")
    c2 = _make_component("R2", "4.7k")
    reqs = _make_requirements(components=(c1, c2))
    result = reqs.get_component("R2")
    assert result is not None
    assert result.value == "4.7k"


# ---------------------------------------------------------------------------
# ProjectRequirements.get_net
# ---------------------------------------------------------------------------


def test_get_net_found() -> None:
    """get_net returns matching Net."""
    reqs = _make_requirements()
    net = reqs.get_net("NET_A")
    assert net is not None
    assert net.name == "NET_A"


def test_get_net_not_found() -> None:
    """get_net returns None for unknown net name."""
    reqs = _make_requirements()
    assert reqs.get_net("NONEXISTENT") is None


# ---------------------------------------------------------------------------
# Frozen dataclass enforcement
# ---------------------------------------------------------------------------


def test_component_frozen() -> None:
    """Component is frozen."""
    comp = _make_component()
    with pytest.raises(AttributeError):
        comp.ref = "changed"  # type: ignore[misc]


def test_project_requirements_frozen() -> None:
    """ProjectRequirements is frozen."""
    reqs = _make_requirements()
    with pytest.raises(AttributeError):
        reqs.project = ProjectInfo(name="new")  # type: ignore[misc]


def test_pin_frozen() -> None:
    """Pin is frozen."""
    pin = Pin(number="1", name="VCC", pin_type=PinType.POWER_IN)
    with pytest.raises(AttributeError):
        pin.net = "changed"  # type: ignore[misc]


def test_net_frozen() -> None:
    """Net is frozen."""
    net = Net("GND", (NetConnection("R1", "1"),))
    with pytest.raises(AttributeError):
        net.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Component optional fields
# ---------------------------------------------------------------------------


def test_component_optional_fields_default_none() -> None:
    """Component optional fields default to None."""
    comp = Component(ref="R1", value="10k", footprint="R_0805")
    assert comp.lcsc is None
    assert comp.description is None
    assert comp.datasheet is None
    assert comp.mpn is None
    assert comp.manufacturer is None


def test_component_with_all_optional_fields() -> None:
    """Component with all optional fields populated."""
    comp = Component(
        ref="U1",
        value="ESP32",
        footprint="ESP32-S3-WROOM-1",
        lcsc="C123456",
        description="MCU module",
        datasheet="https://example.com",
        mpn="ESP32-S3-WROOM-1",
        manufacturer="Espressif",
    )
    assert comp.lcsc == "C123456"
    assert comp.mpn == "ESP32-S3-WROOM-1"


# ---------------------------------------------------------------------------
# Pin with function
# ---------------------------------------------------------------------------


def test_pin_function_none_default() -> None:
    """Pin function defaults to None."""
    pin = Pin(number="1", name="~", pin_type=PinType.PASSIVE)
    assert pin.function is None


def test_pin_with_function() -> None:
    """Pin can have a PinFunction."""
    pin = Pin(number="1", name="SDA", pin_type=PinType.BIDIRECTIONAL, function=PinFunction.I2C_SDA)
    assert pin.function == PinFunction.I2C_SDA


# ---------------------------------------------------------------------------
# BoardContext
# ---------------------------------------------------------------------------


def test_board_context_defaults() -> None:
    """BoardContext has sensible defaults."""
    ctx = BoardContext()
    assert ctx.target_system is None
    assert ctx.shared_grounds is False
    assert ctx.shared_terminals == ()
    assert ctx.notes == ()


# ---------------------------------------------------------------------------
# MechanicalConstraints
# ---------------------------------------------------------------------------


def test_mechanical_constraints_defaults() -> None:
    """MechanicalConstraints has sensible defaults for optional fields."""
    mech = MechanicalConstraints(board_width_mm=100.0, board_height_mm=80.0)
    assert mech.enclosure is None
    assert mech.mounting_hole_diameter_mm == pytest.approx(3.2)
    assert mech.mounting_hole_positions == ()
    assert mech.notes is None
