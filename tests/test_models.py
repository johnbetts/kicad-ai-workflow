"""Comprehensive tests for kicad_pipeline data models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    PCBDesign,
    ZoneFill,
)
from kicad_pipeline.models.pcb import (
    Point as PPoint,
)
from kicad_pipeline.models.production import (
    BOM,
    CPL,
    BOMEntry,
    CostEstimate,
    CPLEntry,
    DrillFile,
    GerberLayer,
    ProductionPackage,
)
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    Net,
    NetConnection,
    Pin,
    PinType,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.models.schematic import (
    Point as SPoint,
)
from kicad_pipeline.models.schematic import (
    Schematic,
    StrokeType,
    Wire,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pin(number: str = "1", name: str = "VCC") -> Pin:
    return Pin(number=number, name=name, pin_type=PinType.PASSIVE)


def _make_component(ref: str = "R1") -> Component:
    pins = (_make_pin("1", "A"), _make_pin("2", "B"))
    return Component(ref=ref, value="10k", footprint="R_0805", pins=pins)


def _make_net(name: str = "+3V3") -> Net:
    return Net(
        name=name,
        connections=(NetConnection(ref="R1", pin="1"), NetConnection(ref="C1", pin="1")),
    )


def _make_project_requirements() -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="TestProject"),
        features=(
            FeatureBlock(
                name="Power",
                description="Power supply",
                components=("U1",),
                nets=("+3V3", "GND"),
                subcircuits=("ldo",),
            ),
        ),
        components=(_make_component("R1"), _make_component("C1")),
        nets=(_make_net("+3V3"), _make_net("GND")),
    )


def _make_pcb_design() -> PCBDesign:
    outline = BoardOutline(
        polygon=(PPoint(0, 0), PPoint(100, 0), PPoint(100, 80), PPoint(0, 80))
    )
    fp = Footprint(
        lib_id="R_0805:R_0805_2012Metric",
        ref="R1",
        value="10k",
        position=PPoint(50.0, 40.0),
    )
    nets = (NetEntry(number=0, name="GND"), NetEntry(number=1, name="+3V3"))
    return PCBDesign(
        outline=outline,
        design_rules=DesignRules(),
        nets=nets,
        footprints=(fp,),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_bom(with_prices: bool = True) -> BOM:
    entries: tuple[BOMEntry, ...]
    if with_prices:
        entries = (
            BOMEntry(
                comment="10k",
                designators=("R1", "R2"),
                footprint="R_0805",
                quantity=2,
                unit_price_usd=0.01,
            ),
            BOMEntry(
                comment="100nF",
                designators=("C1",),
                footprint="C_0402",
                quantity=1,
                unit_price_usd=0.02,
            ),
        )
    else:
        entries = (
            BOMEntry(
                comment="10k",
                designators=("R1",),
                footprint="R_0805",
                quantity=1,
                unit_price_usd=0.01,
            ),
            BOMEntry(
                comment="Mysterious IC",
                designators=("U1",),
                footprint="QFN-48",
                quantity=1,
                unit_price_usd=None,  # price unknown
            ),
        )
    return BOM(entries=entries, project_name="TestProject")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComponentGetPin:
    def test_component_get_pin_found(self) -> None:
        """get_pin returns the correct Pin when the pin number exists."""
        comp = _make_component()
        pin = comp.get_pin("1")
        assert pin is not None
        assert pin.number == "1"
        assert pin.name == "A"

    def test_component_get_pin_not_found(self) -> None:
        """get_pin returns None when the pin number does not exist."""
        comp = _make_component()
        assert comp.get_pin("99") is None


class TestProjectRequirements:
    def test_project_requirements_get_component(self) -> None:
        """get_component returns component by ref."""
        req = _make_project_requirements()
        comp = req.get_component("C1")
        assert comp is not None
        assert comp.ref == "C1"

    def test_project_requirements_get_component_missing(self) -> None:
        """get_component returns None for unknown ref."""
        req = _make_project_requirements()
        assert req.get_component("Z99") is None

    def test_project_requirements_get_net(self) -> None:
        """get_net returns net by name."""
        req = _make_project_requirements()
        net = req.get_net("+3V3")
        assert net is not None
        assert net.name == "+3V3"

    def test_project_requirements_get_net_missing(self) -> None:
        """get_net returns None for unknown net name."""
        req = _make_project_requirements()
        assert req.get_net("NONEXISTENT") is None


class TestFrozenDataclasses:
    def test_frozen_dataclass_immutable(self) -> None:
        """Assigning to a frozen dataclass attribute raises FrozenInstanceError."""
        pin = _make_pin()
        with pytest.raises(FrozenInstanceError):
            pin.number = "99"  # type: ignore[misc]

    def test_frozen_component_immutable(self) -> None:
        """Assigning to a frozen Component raises FrozenInstanceError."""
        comp = _make_component()
        with pytest.raises(FrozenInstanceError):
            comp.ref = "X99"  # type: ignore[misc]


class TestEnums:
    def test_pin_type_enum_values(self) -> None:
        """PinType enum has all expected members."""
        expected = {
            "INPUT", "OUTPUT", "BIDIRECTIONAL", "PASSIVE",
            "POWER_IN", "POWER_OUT", "OPEN_COLLECTOR", "NO_CONNECT",
        }
        assert {m.name for m in PinType} == expected

    def test_pin_type_values(self) -> None:
        """PinType enum values are lowercase strings."""
        assert PinType.INPUT.value == "input"
        assert PinType.POWER_IN.value == "power_in"

    def test_zone_fill_enum(self) -> None:
        """ZoneFill enum has SOLID and HATCHED members."""
        assert ZoneFill.SOLID.value == "solid"
        assert ZoneFill.HATCHED.value == "hatched"

    def test_stroke_type_enum(self) -> None:
        """StrokeType enum has DEFAULT and DASH members."""
        assert StrokeType.DEFAULT.value == "default"
        assert StrokeType.DASH.value == "dash"


class TestNetConnection:
    def test_net_connection_tuple(self) -> None:
        """NetConnection objects are stored as a tuple inside Net."""
        net = _make_net()
        assert isinstance(net.connections, tuple)
        assert len(net.connections) == 2
        first = net.connections[0]
        assert isinstance(first, NetConnection)
        assert first.ref == "R1"
        assert first.pin == "1"


class TestBOM:
    def test_bom_total_cost_usd(self) -> None:
        """BOM.total_cost_usd sums unit_price * quantity for all entries."""
        bom = _make_bom(with_prices=True)
        # 2 * 0.01 + 1 * 0.02 = 0.04
        result = bom.total_cost_usd
        assert result is not None
        assert abs(result - 0.04) < 1e-9

    def test_bom_total_cost_none_if_missing_price(self) -> None:
        """BOM.total_cost_usd returns None when any entry has no price."""
        bom = _make_bom(with_prices=False)
        assert bom.total_cost_usd is None


class TestPCBDesign:
    def test_pcb_get_footprint(self) -> None:
        """PCBDesign.get_footprint returns the correct Footprint by ref."""
        design = _make_pcb_design()
        fp = design.get_footprint("R1")
        assert fp is not None
        assert fp.ref == "R1"
        assert fp.value == "10k"

    def test_pcb_get_footprint_missing(self) -> None:
        """PCBDesign.get_footprint returns None for unknown ref."""
        design = _make_pcb_design()
        assert design.get_footprint("Z99") is None

    def test_pcb_get_net_number(self) -> None:
        """PCBDesign.get_net_number returns the correct integer for a net name."""
        design = _make_pcb_design()
        assert design.get_net_number("+3V3") == 1
        assert design.get_net_number("GND") == 0

    def test_pcb_get_net_number_missing(self) -> None:
        """PCBDesign.get_net_number returns None for unknown net name."""
        design = _make_pcb_design()
        assert design.get_net_number("FAKE_NET") is None


class TestPoints:
    def test_point_creation(self) -> None:
        """Point can be created and its coordinates accessed."""
        p = PPoint(x=1.5, y=2.75)
        assert p.x == 1.5
        assert p.y == 2.75

    def test_schematic_point_creation(self) -> None:
        """Schematic Point can be created independently of PCB Point."""
        p = SPoint(x=10.0, y=20.0)
        assert p.x == 10.0
        assert p.y == 20.0

    def test_point_is_frozen(self) -> None:
        """Point raises FrozenInstanceError on attribute assignment."""
        p = PPoint(0.0, 0.0)
        with pytest.raises(FrozenInstanceError):
            p.x = 1.0  # type: ignore[misc]


class TestSchematic:
    def test_schematic_empty_is_valid(self) -> None:
        """Schematic with all-empty tuples can be instantiated without error."""
        sch = Schematic(
            lib_symbols=(),
            symbols=(),
            power_symbols=(),
            wires=(),
            junctions=(),
            no_connects=(),
            labels=(),
            global_labels=(),
        )
        assert sch.version == 20231120
        assert sch.generator == "kicad-ai-pipeline"
        assert sch.paper == "A4"
        assert len(sch.symbols) == 0

    def test_schematic_with_wire(self) -> None:
        """Schematic stores Wire objects in the wires tuple."""
        wire = Wire(start=SPoint(0.0, 0.0), end=SPoint(10.0, 0.0))
        sch = Schematic(
            lib_symbols=(),
            symbols=(),
            power_symbols=(),
            wires=(wire,),
            junctions=(),
            no_connects=(),
            labels=(),
            global_labels=(),
        )
        assert len(sch.wires) == 1
        assert sch.wires[0].end.x == 10.0


class TestProductionPackage:
    def test_production_package_fields(self) -> None:
        """ProductionPackage stores all expected fields correctly."""
        gerber = GerberLayer(layer_name="F.Cu", filename="proj-F_Cu.gbr", content="G04*")
        drill = DrillFile(filename="proj-PTH.drl", content="; drill")
        bom = _make_bom()
        cpl = CPL(
            entries=(
                CPLEntry(
                    designator="R1",
                    value="10k",
                    package="0805",
                    mid_x=50.0,
                    mid_y=40.0,
                    rotation=0.0,
                    layer="top",
                ),
            ),
            project_name="TestProject",
        )
        pkg = ProductionPackage(
            project_name="TestProject",
            revision="v1.0",
            gerbers=(gerber,),
            drill_files=(drill,),
            bom=bom,
            cpl=cpl,
        )
        assert pkg.project_name == "TestProject"
        assert pkg.revision == "v1.0"
        assert len(pkg.gerbers) == 1
        assert pkg.gerbers[0].layer_name == "F.Cu"
        assert len(pkg.drill_files) == 1
        assert pkg.cost_estimate is None
        assert pkg.order_guide == ""

    def test_production_package_with_cost_estimate(self) -> None:
        """ProductionPackage accepts a CostEstimate."""
        bom = _make_bom()
        cpl = CPL(entries=(), project_name="TestProject")
        cost = CostEstimate(
            bom_cost_usd=5.00,
            pcb_cost_5_usd=8.00,
            pcb_cost_10_usd=12.00,
            pcb_cost_50_usd=40.00,
            notes=("Estimate only",),
        )
        pkg = ProductionPackage(
            project_name="TestProject",
            revision="v1.0",
            gerbers=(),
            drill_files=(),
            bom=bom,
            cpl=cpl,
            cost_estimate=cost,
        )
        assert pkg.cost_estimate is not None
        assert pkg.cost_estimate.bom_cost_usd == 5.00


class TestPowerBudget:
    def test_power_budget_fields(self) -> None:
        """PowerBudget stores rails as a tuple and exposes total_current_ma."""
        rail1 = PowerRail(name="+3V3", voltage=3.3, current_ma=500.0, source_ref="U1")
        rail2 = PowerRail(name="+5V", voltage=5.0, current_ma=200.0, source_ref="J1")
        budget = PowerBudget(
            rails=(rail1, rail2),
            total_current_ma=700.0,
            notes=("Budget is conservative",),
        )
        assert isinstance(budget.rails, tuple)
        assert len(budget.rails) == 2
        assert budget.rails[0].name == "+3V3"
        assert budget.rails[1].voltage == 5.0
        assert budget.total_current_ma == 700.0
        assert budget.notes == ("Budget is conservative",)

    def test_power_rail_is_frozen(self) -> None:
        """PowerRail raises FrozenInstanceError on attribute assignment."""
        rail = PowerRail(name="+3V3", voltage=3.3, current_ma=100.0, source_ref="U1")
        with pytest.raises(FrozenInstanceError):
            rail.voltage = 5.0  # type: ignore[misc]
