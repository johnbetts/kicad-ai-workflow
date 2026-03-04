"""Tests for DIP switch ERC check in electrical validation."""

from __future__ import annotations

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    NetEntry,
    PCBDesign,
    Point,
)
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
from kicad_pipeline.validation.electrical import run_electrical_checks


def _make_minimal_pcb() -> PCBDesign:
    """Create a minimal PCBDesign for testing."""
    return PCBDesign(
        outline=BoardOutline(polygon=(
            Point(0, 0), Point(50, 0), Point(50, 50), Point(0, 50), Point(0, 0),
        )),
        design_rules=DesignRules(),
        nets=(NetEntry(number=0, name=""), NetEntry(number=1, name="GND")),
        footprints=(),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def test_dip_switch_power_warning() -> None:
    """DIP switch with multiple power net pins emits a warning."""
    sw = Component(
        ref="SW1",
        value="DIPx04",
        footprint="SW_DIP_SPSTx04",
        pins=(
            Pin(number="1", name="IN1", pin_type=PinType.PASSIVE, net="VCC"),
            Pin(number="2", name="IN2", pin_type=PinType.PASSIVE, net="GND"),
            Pin(number="3", name="OUT1", pin_type=PinType.PASSIVE, net="ADDR0"),
            Pin(number="4", name="OUT2", pin_type=PinType.PASSIVE, net="ADDR1"),
        ),
    )
    reqs = ProjectRequirements(
        project=ProjectInfo(name="Test"),
        features=(FeatureBlock(
            name="Switches", description="", components=("SW1",),
            nets=("VCC", "GND"), subcircuits=(),
        ),),
        components=(sw,),
        nets=(
            Net(name="VCC", connections=(NetConnection(ref="SW1", pin="1"),)),
            Net(name="GND", connections=(NetConnection(ref="SW1", pin="2"),)),
        ),
    )
    pcb = _make_minimal_pcb()
    report = run_electrical_checks(pcb, reqs)
    dip_warnings = [v for v in report.warnings if v.rule == "dip_switch_protection"]
    assert len(dip_warnings) >= 1


def test_dip_switch_no_warning_without_power() -> None:
    """DIP switch without power net pins does not emit a warning."""
    sw = Component(
        ref="SW1",
        value="DIPx04",
        footprint="SW_DIP_SPSTx04",
        pins=(
            Pin(number="1", name="IN1", pin_type=PinType.PASSIVE, net="ADDR0"),
            Pin(number="2", name="IN2", pin_type=PinType.PASSIVE, net="ADDR1"),
        ),
    )
    reqs = ProjectRequirements(
        project=ProjectInfo(name="Test"),
        features=(FeatureBlock(
            name="Switches", description="", components=("SW1",),
            nets=(), subcircuits=(),
        ),),
        components=(sw,),
        nets=(),
    )
    pcb = _make_minimal_pcb()
    report = run_electrical_checks(pcb, reqs)
    dip_warnings = [v for v in report.warnings if v.rule == "dip_switch_protection"]
    assert len(dip_warnings) == 0
