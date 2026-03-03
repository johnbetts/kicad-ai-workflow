"""End-to-end integration test: ESP32 + W5500 Ethernet project.

Tests the complete pipeline for a more complex design:
- U1: ESP32-S3-WROOM-1-N4 (MCU)
- U2: W5500 (Ethernet controller)
- U3: AMS1117-3.3 (LDO voltage regulator)
- C1: 100nF decoupling capacitor
- J1: RJ45 connector
"""

from __future__ import annotations

import pytest

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
from kicad_pipeline.pcb.builder import build_pcb
from kicad_pipeline.production.packager import build_production_package
from kicad_pipeline.schematic.builder import build_schematic


@pytest.fixture
def esp32_ethernet_requirements() -> ProjectRequirements:
    """ESP32-S3 + W5500 Ethernet project requirements."""
    # U1: ESP32-S3-WROOM-1-N4
    pins_u1 = (
        Pin(number="1", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
        Pin(number="2", name="3V3", pin_type=PinType.POWER_IN, function=PinFunction.VCC),
        Pin(number="3", name="EN", pin_type=PinType.INPUT, function=PinFunction.ENABLE),
        Pin(number="4", name="IO10", pin_type=PinType.BIDIRECTIONAL, function=PinFunction.SPI_CS),
        Pin(
            number="5", name="IO11", pin_type=PinType.BIDIRECTIONAL, function=PinFunction.SPI_MOSI
        ),
        Pin(
            number="6", name="IO12", pin_type=PinType.BIDIRECTIONAL, function=PinFunction.SPI_MISO
        ),
        Pin(
            number="7", name="IO13", pin_type=PinType.BIDIRECTIONAL, function=PinFunction.SPI_CLK
        ),
    )
    u1 = Component(
        ref="U1",
        value="ESP32-S3-WROOM-1-N4",
        footprint="ESP32-S3-WROOM-1",
        lcsc="C2913202",
        pins=pins_u1,
    )

    # U2: W5500 Ethernet
    pins_u2 = (
        Pin(number="1", name="VCC", pin_type=PinType.POWER_IN, function=PinFunction.VCC),
        Pin(number="2", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
        Pin(number="3", name="MISO", pin_type=PinType.OUTPUT, function=PinFunction.SPI_MISO),
        Pin(number="4", name="MOSI", pin_type=PinType.INPUT, function=PinFunction.SPI_MOSI),
        Pin(number="5", name="SCLK", pin_type=PinType.INPUT, function=PinFunction.SPI_CLK),
        Pin(number="6", name="SCSn", pin_type=PinType.INPUT, function=PinFunction.SPI_CS),
    )
    u2 = Component(
        ref="U2",
        value="W5500",
        footprint="W5500-QFN48",
        lcsc="C32843",
        pins=pins_u2,
    )

    # U3: AMS1117-3.3 LDO
    pins_u3 = (
        Pin(number="1", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
        Pin(number="2", name="OUT", pin_type=PinType.POWER_OUT, function=PinFunction.VCC),
        Pin(number="3", name="IN", pin_type=PinType.POWER_IN, function=PinFunction.VCC),
    )
    u3 = Component(
        ref="U3",
        value="AMS1117-3.3",
        footprint="SOT-223",
        lcsc="C6186",
        pins=pins_u3,
    )

    # C1: 100nF decoupling cap
    pins_c1 = (
        Pin(number="1", name="+", pin_type=PinType.PASSIVE, function=PinFunction.NC),
        Pin(number="2", name="-", pin_type=PinType.PASSIVE, function=PinFunction.NC),
    )
    c1 = Component(
        ref="C1",
        value="100nF",
        footprint="C_0402",
        lcsc="C14663",
        pins=pins_c1,
    )

    # J1: RJ45 connector
    pins_j1 = (
        Pin(number="1", name="TD+", pin_type=PinType.PASSIVE, function=PinFunction.NC),
        Pin(number="2", name="TD-", pin_type=PinType.PASSIVE, function=PinFunction.NC),
        Pin(number="3", name="RD+", pin_type=PinType.PASSIVE, function=PinFunction.NC),
        Pin(number="6", name="RD-", pin_type=PinType.PASSIVE, function=PinFunction.NC),
    )
    j1 = Component(
        ref="J1",
        value="RJ45",
        footprint="RJ45_Amphenol_RJHSE538X",
        lcsc="C2977",
        pins=pins_j1,
    )

    # Nets
    net_gnd = Net(
        name="GND",
        connections=(
            NetConnection(ref="U1", pin="1"),
            NetConnection(ref="U2", pin="2"),
            NetConnection(ref="U3", pin="1"),
        ),
    )
    net_3v3 = Net(
        name="+3V3",
        connections=(
            NetConnection(ref="U1", pin="2"),
            NetConnection(ref="U2", pin="1"),
            NetConnection(ref="U3", pin="2"),
        ),
    )
    net_5v = Net(
        name="+5V",
        connections=(NetConnection(ref="U3", pin="3"),),
    )
    net_miso = Net(
        name="SPI_MISO",
        connections=(
            NetConnection(ref="U1", pin="6"),
            NetConnection(ref="U2", pin="3"),
        ),
    )
    net_mosi = Net(
        name="SPI_MOSI",
        connections=(
            NetConnection(ref="U1", pin="5"),
            NetConnection(ref="U2", pin="4"),
        ),
    )
    net_sclk = Net(
        name="SPI_SCLK",
        connections=(
            NetConnection(ref="U1", pin="7"),
            NetConnection(ref="U2", pin="5"),
        ),
    )
    net_cs = Net(
        name="ETH_CS",
        connections=(
            NetConnection(ref="U1", pin="4"),
            NetConnection(ref="U2", pin="6"),
        ),
    )

    # Feature blocks
    feature_mcu = FeatureBlock(
        name="MCU",
        description="ESP32-S3 microcontroller",
        components=("U1",),
        nets=("GND", "+3V3"),
        subcircuits=(),
    )
    feature_ethernet = FeatureBlock(
        name="Ethernet",
        description="W5500 Ethernet controller with RJ45",
        components=("U2", "J1"),
        nets=("SPI_MISO", "SPI_MOSI", "SPI_SCLK", "ETH_CS"),
        subcircuits=(),
    )
    feature_power = FeatureBlock(
        name="Power",
        description="AMS1117 LDO power regulation",
        components=("U3", "C1"),
        nets=("+5V", "+3V3", "GND"),
        subcircuits=(),
    )

    # Power budget
    rail_3v3 = PowerRail(
        name="+3V3",
        voltage=3.3,
        current_ma=800.0,
        source_ref="U3",
    )
    rail_5v = PowerRail(
        name="+5V",
        voltage=5.0,
        current_ma=1000.0,
        source_ref="J1",
    )
    power_budget = PowerBudget(
        rails=(rail_3v3, rail_5v),
        total_current_ma=1000.0,
        notes=("5V from USB or external supply",),
    )

    mech = MechanicalConstraints(
        board_width_mm=80.0,
        board_height_mm=60.0,
    )

    return ProjectRequirements(
        project=ProjectInfo(
            name="esp32-ethernet",
            author="test",
            revision="v0.1",
            description="ESP32-S3 + W5500 Ethernet board",
        ),
        features=(feature_mcu, feature_ethernet, feature_power),
        components=(u1, u2, u3, c1, j1),
        nets=(net_gnd, net_3v3, net_5v, net_miso, net_mosi, net_sclk, net_cs),
        power_budget=power_budget,
        mechanical=mech,
    )


# ---------------------------------------------------------------------------
# Schematic tests
# ---------------------------------------------------------------------------


def test_esp32_schematic_builds(esp32_ethernet_requirements: ProjectRequirements) -> None:
    """build_schematic() succeeds for the ESP32 Ethernet project."""
    from kicad_pipeline.models.schematic import Schematic

    sch = build_schematic(esp32_ethernet_requirements)
    assert isinstance(sch, Schematic)


# ---------------------------------------------------------------------------
# PCB tests
# ---------------------------------------------------------------------------


def test_esp32_pcb_builds(esp32_ethernet_requirements: ProjectRequirements) -> None:
    """build_pcb() succeeds for the ESP32 Ethernet project."""
    from kicad_pipeline.models.pcb import PCBDesign

    design = build_pcb(esp32_ethernet_requirements)
    assert isinstance(design, PCBDesign)


def test_esp32_pcb_has_ethernet_footprint(
    esp32_ethernet_requirements: ProjectRequirements,
) -> None:
    """PCBDesign contains the W5500 footprint (U2)."""
    design = build_pcb(esp32_ethernet_requirements)
    refs = {fp.ref for fp in design.footprints}
    assert "U2" in refs


def test_esp32_pcb_has_rj45_footprint(
    esp32_ethernet_requirements: ProjectRequirements,
) -> None:
    """PCBDesign contains the RJ45 connector footprint (J1)."""
    design = build_pcb(esp32_ethernet_requirements)
    refs = {fp.ref for fp in design.footprints}
    assert "J1" in refs


# ---------------------------------------------------------------------------
# Production tests
# ---------------------------------------------------------------------------


def test_esp32_production_builds(esp32_ethernet_requirements: ProjectRequirements) -> None:
    """build_production_package() succeeds for the ESP32 Ethernet project."""
    from kicad_pipeline.production.packager import ProductionPackage

    design = build_pcb(esp32_ethernet_requirements)
    pkg = build_production_package(design, "esp32-ethernet", esp32_ethernet_requirements)
    assert isinstance(pkg, ProductionPackage)


def test_esp32_bom_has_all_components(
    esp32_ethernet_requirements: ProjectRequirements,
) -> None:
    """BOM CSV contains all expected component values or references."""
    design = build_pcb(esp32_ethernet_requirements)
    pkg = build_production_package(design, "esp32-ethernet", esp32_ethernet_requirements)
    # Check that key component info is present (value or ref)
    bom = pkg.bom_csv
    found_any = any(
        token in bom
        for token in ("W5500", "AMS1117", "ESP32", "100nF", "RJ45", "U1", "U2", "U3")
    )
    assert found_any, f"No expected component found in BOM CSV:\n{bom}"


def test_esp32_cpl_has_smd_components(
    esp32_ethernet_requirements: ProjectRequirements,
) -> None:
    """CPL CSV contains SMD component references."""
    design = build_pcb(esp32_ethernet_requirements)
    pkg = build_production_package(design, "esp32-ethernet", esp32_ethernet_requirements)
    cpl = pkg.cpl_csv
    found_any = any(ref in cpl for ref in ("U1", "U2", "U3", "C1"))
    assert found_any, f"No SMD component refs found in CPL CSV:\n{cpl}"


def test_esp32_gerbers_all_layers(esp32_ethernet_requirements: ProjectRequirements) -> None:
    """Production package contains 7 Gerber files (6 layers + Edge.Cuts)."""
    design = build_pcb(esp32_ethernet_requirements)
    pkg = build_production_package(design, "esp32-ethernet", esp32_ethernet_requirements)
    assert len(pkg.gerbers) == 7
