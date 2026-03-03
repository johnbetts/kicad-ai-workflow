"""End-to-end integration test: LED blinker project.

Tests the complete pipeline from requirements to production artifacts.
Uses a minimal ESP32 + LED + resistor design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.models.requirements import (
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
)
from kicad_pipeline.pcb.builder import build_pcb, write_pcb
from kicad_pipeline.production.packager import build_production_package
from kicad_pipeline.schematic.builder import build_schematic, write_schematic


@pytest.fixture
def led_blinker_requirements() -> ProjectRequirements:
    """Minimal LED blinker: ESP32 MCU + green LED + 330R resistor."""
    pins_u1 = (
        Pin(number="1", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
        Pin(number="2", name="3V3", pin_type=PinType.POWER_IN, function=PinFunction.VCC),
        Pin(number="3", name="GPIO2", pin_type=PinType.BIDIRECTIONAL, function=PinFunction.GPIO),
    )
    u1 = Component(
        ref="U1",
        value="ESP32-WROOM-32E",
        footprint="ESP32-WROOM-32E",
        lcsc="C165948",
        pins=pins_u1,
    )

    pins_d1 = (
        Pin(number="1", name="K", pin_type=PinType.PASSIVE, function=PinFunction.NC),
        Pin(number="2", name="A", pin_type=PinType.PASSIVE, function=PinFunction.NC),
    )
    d1 = Component(
        ref="D1",
        value="Green LED",
        footprint="LED_0805",
        lcsc="C70187",
        pins=pins_d1,
    )

    pins_r1 = (
        Pin(number="1", name="~", pin_type=PinType.PASSIVE, function=PinFunction.NC),
        Pin(number="2", name="~", pin_type=PinType.PASSIVE, function=PinFunction.NC),
    )
    r1 = Component(
        ref="R1",
        value="330R",
        footprint="R_0805",
        lcsc="C17516",
        pins=pins_r1,
    )

    net_gnd = Net(
        name="GND",
        connections=(NetConnection(ref="U1", pin="1"),),
    )
    net_3v3 = Net(
        name="+3V3",
        connections=(NetConnection(ref="U1", pin="2"),),
    )
    net_led = Net(
        name="LED_OUT",
        connections=(
            NetConnection(ref="U1", pin="3"),
            NetConnection(ref="R1", pin="1"),
        ),
    )
    net_led_a = Net(
        name="LED_A",
        connections=(
            NetConnection(ref="R1", pin="2"),
            NetConnection(ref="D1", pin="2"),
        ),
    )

    feature_mcu = FeatureBlock(
        name="MCU",
        description="ESP32 microcontroller",
        components=("U1",),
        nets=("GND", "+3V3"),
        subcircuits=(),
    )
    feature_led = FeatureBlock(
        name="LED",
        description="LED indicator circuit",
        components=("D1", "R1"),
        nets=("LED_OUT", "LED_A"),
        subcircuits=(),
    )

    power_rail = PowerRail(
        name="+3V3",
        voltage=3.3,
        current_ma=500.0,
        source_ref="U1",
    )
    power_budget = PowerBudget(
        rails=(power_rail,),
        total_current_ma=500.0,
        notes=(),
    )

    mech = MechanicalConstraints(
        board_width_mm=50.0,
        board_height_mm=40.0,
    )

    return ProjectRequirements(
        project=ProjectInfo(name="led-blinker", author="test", revision="v0.1"),
        features=(feature_mcu, feature_led),
        components=(u1, d1, r1),
        nets=(net_gnd, net_3v3, net_led, net_led_a),
        power_budget=power_budget,
        mechanical=mech,
    )


# ---------------------------------------------------------------------------
# Schematic tests
# ---------------------------------------------------------------------------


def test_schematic_builds(led_blinker_requirements: ProjectRequirements) -> None:
    """build_schematic() succeeds and returns a Schematic object."""
    from kicad_pipeline.models.schematic import Schematic

    sch = build_schematic(led_blinker_requirements)
    assert isinstance(sch, Schematic)


def test_schematic_has_components(led_blinker_requirements: ProjectRequirements) -> None:
    """Schematic contains at least one symbol instance."""
    sch = build_schematic(led_blinker_requirements)
    assert len(sch.symbols) > 0


# ---------------------------------------------------------------------------
# PCB tests
# ---------------------------------------------------------------------------


def test_pcb_builds(led_blinker_requirements: ProjectRequirements) -> None:
    """build_pcb() succeeds and returns a PCBDesign object."""
    from kicad_pipeline.models.pcb import PCBDesign

    design = build_pcb(led_blinker_requirements)
    assert isinstance(design, PCBDesign)


def test_pcb_has_footprints(led_blinker_requirements: ProjectRequirements) -> None:
    """PCBDesign contains at least one footprint."""
    design = build_pcb(led_blinker_requirements)
    assert len(design.footprints) > 0


def test_pcb_has_nets(led_blinker_requirements: ProjectRequirements) -> None:
    """PCBDesign contains at least one net entry."""
    design = build_pcb(led_blinker_requirements)
    assert len(design.nets) > 0


def test_pcb_footprint_count_matches(led_blinker_requirements: ProjectRequirements) -> None:
    """PCBDesign has exactly 3 footprints (U1, D1, R1)."""
    design = build_pcb(led_blinker_requirements)
    assert len(design.footprints) == 3


# ---------------------------------------------------------------------------
# Production package tests
# ---------------------------------------------------------------------------


def test_production_package_builds(led_blinker_requirements: ProjectRequirements) -> None:
    """build_production_package() succeeds and returns a ProductionPackage."""
    from kicad_pipeline.production.packager import ProductionPackage

    design = build_pcb(led_blinker_requirements)
    pkg = build_production_package(design, "led-blinker", led_blinker_requirements)
    assert isinstance(pkg, ProductionPackage)


def test_production_gerbers_present(led_blinker_requirements: ProjectRequirements) -> None:
    """Production package gerbers include the F.Cu layer file."""
    design = build_pcb(led_blinker_requirements)
    pkg = build_production_package(design, "led-blinker", led_blinker_requirements)
    # F.Cu gerber filename contains "F_Cu"
    f_cu_keys = [k for k in pkg.gerbers if "F_Cu" in k]
    assert len(f_cu_keys) >= 1


def test_production_bom_csv_has_components(
    led_blinker_requirements: ProjectRequirements,
) -> None:
    """BOM CSV contains at least one recognisable component (ESP32 or LED)."""
    design = build_pcb(led_blinker_requirements)
    pkg = build_production_package(design, "led-blinker", led_blinker_requirements)
    assert "ESP32" in pkg.bom_csv or "LED" in pkg.bom_csv or "330R" in pkg.bom_csv


def test_production_cpl_csv_has_refs(led_blinker_requirements: ProjectRequirements) -> None:
    """CPL CSV contains component references U1 or D1."""
    design = build_pcb(led_blinker_requirements)
    pkg = build_production_package(design, "led-blinker", led_blinker_requirements)
    assert "U1" in pkg.cpl_csv or "D1" in pkg.cpl_csv or "R1" in pkg.cpl_csv


# ---------------------------------------------------------------------------
# File write roundtrip tests
# ---------------------------------------------------------------------------


def test_schematic_write_roundtrip(
    led_blinker_requirements: ProjectRequirements,
    tmp_path: Path,
) -> None:
    """write_schematic() creates a non-empty .kicad_sch file."""
    sch = build_schematic(led_blinker_requirements)
    out_path = tmp_path / "led-blinker.kicad_sch"
    write_schematic(sch, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_pcb_write_roundtrip(
    led_blinker_requirements: ProjectRequirements,
    tmp_path: Path,
) -> None:
    """write_pcb() creates a non-empty .kicad_pcb file."""
    design = build_pcb(led_blinker_requirements)
    out_path = tmp_path / "led-blinker.kicad_pcb"
    write_pcb(design, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0
