"""Tests for kicad_pipeline.pcb.placement."""

from __future__ import annotations

import pytest

from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.models.pcb import BoardOutline, Point
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.pcb.placement import (
    PCB_ZONES,
    PCBZone,
    assign_pcb_zones,
    layout_pcb,
    place_pcb_components,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_requirements(
    extra_components: tuple[Component, ...] = (),
) -> ProjectRequirements:
    """Return a minimal ProjectRequirements for placement tests."""
    mcu = Component(
        ref="U1",
        value="ESP32-S3",
        footprint="ESP32-S3-WROOM-1",
        pins=(
            Pin(number="1", name="GND", pin_type=PinType.POWER_IN, net="GND"),
            Pin(number="2", name="VCC", pin_type=PinType.POWER_IN, net="+3V3"),
        ),
    )
    resistor = Component(
        ref="R1",
        value="10k",
        footprint="R_0805",
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, net="+3V3"),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, net="GND"),
        ),
    )
    led = Component(
        ref="D1",
        value="RED",
        footprint="LED_0805",
        pins=(
            Pin(number="1", name="K", pin_type=PinType.PASSIVE, net="GND"),
            Pin(number="2", name="A", pin_type=PinType.PASSIVE, net="LED_NET"),
        ),
    )
    all_comps = (mcu, resistor, led, *extra_components)
    fb_mcu = FeatureBlock(
        name="MCU",
        description="Microcontroller",
        components=("U1",),
        nets=("+3V3", "GND"),
        subcircuits=(),
    )
    fb_power = FeatureBlock(
        name="Power",
        description="Power supply",
        components=("R1",),
        nets=("+3V3", "GND"),
        subcircuits=(),
    )
    fb_status = FeatureBlock(
        name="Status",
        description="Status LED",
        components=("D1",),
        nets=("LED_NET",),
        subcircuits=(),
    )
    net_3v3 = Net(name="+3V3", connections=(NetConnection(ref="U1", pin="2"),))
    net_gnd = Net(name="GND", connections=(NetConnection(ref="U1", pin="1"),))
    return ProjectRequirements(
        project=ProjectInfo(name="PlacementTest"),
        features=(fb_mcu, fb_power, fb_status),
        components=all_comps,
        nets=(net_3v3, net_gnd),
    )


def _simple_board() -> BoardOutline:
    """Return a simple rectangular board outline."""
    return BoardOutline(
        polygon=(
            Point(x=0.0, y=0.0),
            Point(x=80.0, y=0.0),
            Point(x=80.0, y=40.0),
            Point(x=0.0, y=40.0),
        )
    )


# ---------------------------------------------------------------------------
# PCB_ZONES dictionary
# ---------------------------------------------------------------------------


def test_pcb_zones_dict_has_required_zones() -> None:
    """PCB_ZONES contains all required functional zones."""
    for zone_name in ("MCU", "USB_POWER", "STATUS", "ETHERNET", "RJ45", "ANALOG", "PERIPHERALS"):
        assert zone_name in PCB_ZONES, f"Zone '{zone_name}' missing from PCB_ZONES"


def test_pcb_zone_is_frozen() -> None:
    """PCBZone instances are frozen dataclasses — attribute assignment raises."""
    zone = PCBZone("TEST_FROZEN", 0.0, 0.0, 10.0, 10.0)
    with pytest.raises((AttributeError, TypeError)):
        zone.name = "CHANGED"  # type: ignore[misc]


def test_pcb_zones_positive_dimensions() -> None:
    """All zones have positive width and height."""
    for name, zone in PCB_ZONES.items():
        assert zone.width > 0.0, f"Zone {name} has non-positive width"
        assert zone.height > 0.0, f"Zone {name} has non-positive height"


# ---------------------------------------------------------------------------
# assign_pcb_zones
# ---------------------------------------------------------------------------


def test_assign_pcb_zones_mcu() -> None:
    """MCU feature name → MCU zone."""
    result = assign_pcb_zones([("U1", "MCU")])
    assert result["U1"].name == "MCU"


def test_assign_pcb_zones_usb() -> None:
    """USB feature name → USB_POWER zone."""
    result = assign_pcb_zones([("J1", "USB")])
    assert result["J1"].name == "USB_POWER"


def test_assign_pcb_zones_led() -> None:
    """LED feature name → STATUS zone."""
    result = assign_pcb_zones([("D1", "LED")])
    assert result["D1"].name == "STATUS"


def test_assign_pcb_zones_ethernet() -> None:
    """Ethernet feature name → ETHERNET zone."""
    result = assign_pcb_zones([("U2", "Ethernet")])
    assert result["U2"].name == "ETHERNET"


def test_assign_pcb_zones_rj45() -> None:
    """rj45 feature name (lowercase) → RJ45 zone."""
    result = assign_pcb_zones([("J2", "rj45")])
    assert result["J2"].name == "RJ45"


def test_assign_pcb_zones_analog() -> None:
    """analog feature name (lowercase) → ANALOG zone."""
    result = assign_pcb_zones([("U3", "analog")])
    assert result["U3"].name == "ANALOG"


def test_assign_pcb_zones_unknown_feature() -> None:
    """Unknown feature name → PERIPHERALS zone."""
    result = assign_pcb_zones([("X1", "SomethingElse")])
    assert result["X1"].name == "PERIPHERALS"


def test_assign_pcb_zones_multiple() -> None:
    """Multiple components receive correct zone assignments."""
    result = assign_pcb_zones([("U1", "MCU"), ("J1", "RJ45"), ("R1", "Unknown")])
    assert result["U1"].name == "MCU"
    assert result["J1"].name == "RJ45"
    assert result["R1"].name == "PERIPHERALS"


# ---------------------------------------------------------------------------
# place_pcb_components
# ---------------------------------------------------------------------------


def test_place_pcb_components_single() -> None:
    """Single component in MCU zone is placed at zone origin."""
    zone = PCB_ZONES["MCU"]
    result = place_pcb_components(["U1"], zone, grid_mm=0.5)
    assert "U1" in result
    pt = result["U1"]
    assert pt.x == pytest.approx(zone.x)
    assert pt.y == pytest.approx(zone.y)


def test_place_pcb_components_multiple() -> None:
    """3 components receive 3 different positions."""
    zone = PCBZone("TEST", 0.0, 0.0, 40.0, 40.0)
    refs = ["R1", "R2", "R3"]
    result = place_pcb_components(refs, zone)
    assert set(result.keys()) == set(refs)
    # All positions should be distinct
    positions = list(result.values())
    for i, pt_a in enumerate(positions):
        for pt_b in positions[i + 1:]:
            assert (pt_a.x, pt_a.y) != (pt_b.x, pt_b.y)


def test_place_pcb_components_grid_snapped() -> None:
    """Positions are exact multiples of grid_mm."""
    zone = PCBZone("TEST", 0.0, 0.0, 40.0, 40.0)
    refs = ["R1", "R2", "C1"]
    grid = 1.0
    result = place_pcb_components(refs, zone, grid_mm=grid)
    for ref, pt in result.items():
        rem_x = round(pt.x % grid, 9)
        rem_y = round(pt.y % grid, 9)
        assert rem_x == pytest.approx(0.0) or rem_x == pytest.approx(grid), (
            f"{ref} x={pt.x} not on grid {grid}"
        )
        assert rem_y == pytest.approx(0.0) or rem_y == pytest.approx(grid), (
            f"{ref} y={pt.y} not on grid {grid}"
        )


def test_place_pcb_components_empty() -> None:
    """Empty ref list returns an empty dict."""
    zone = PCBZone("TEST", 0.0, 0.0, 40.0, 40.0)
    result = place_pcb_components([], zone)
    assert result == {}


def test_place_pcb_components_too_many_raises() -> None:
    """PCBError raised when zone is too small to accommodate all components."""
    zone = PCBZone("TINY", 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(PCBError):
        place_pcb_components(["R1", "R2", "R3", "R4", "R5"], zone, component_spacing_mm=3.0)


def test_snap_internal() -> None:
    """Verify grid snapping via place_pcb_components with grid=1.0 mm."""
    zone = PCBZone("GRID_TEST", 0.3, 0.3, 20.0, 20.0)
    refs = ["R1"]
    result = place_pcb_components(refs, zone, grid_mm=1.0)
    pt = result["R1"]
    # 0.3 snapped to nearest 1.0 mm multiple = 0.0
    assert pt.x == pytest.approx(0.0)
    assert pt.y == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# layout_pcb
# ---------------------------------------------------------------------------


def test_layout_pcb_all_refs_placed() -> None:
    """layout_pcb returns a position for every component in requirements."""
    req = _minimal_requirements()
    board = _simple_board()
    result = layout_pcb(req, board)
    all_refs = {c.ref for c in req.components}
    assert all_refs <= set(result.positions.keys())


def test_layout_pcb_returns_points() -> None:
    """All values returned by layout_pcb are Point instances."""
    req = _minimal_requirements()
    board = _simple_board()
    result = layout_pcb(req, board)
    for ref, pt in result.positions.items():
        assert isinstance(pt, Point), f"{ref} value is not a Point: {pt!r}"


def test_layout_pcb_returns_layout_result() -> None:
    """layout_pcb returns a LayoutResult with positions and rotations."""
    from kicad_pipeline.pcb.placement import LayoutResult

    req = _minimal_requirements()
    board = _simple_board()
    result = layout_pcb(req, board)
    assert isinstance(result, LayoutResult)
    assert isinstance(result.positions, dict)
    assert isinstance(result.rotations, dict)
