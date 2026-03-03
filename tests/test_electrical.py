"""Tests for the electrical validation engine."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    Net,
    NetConnection,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.validation.drc import Severity
from kicad_pipeline.validation.electrical import ElectricalReport, run_electrical_checks

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

_DEFAULT_RULES = DesignRules()

_CLOSED_RECT: tuple[Point, ...] = (
    Point(0.0, 0.0),
    Point(50.0, 0.0),
    Point(50.0, 50.0),
    Point(0.0, 50.0),
    Point(0.0, 0.0),
)


def _make_pcb_with_gnd() -> PCBDesign:
    """Minimal PCB that has a GND net and one connected pad."""
    gnd = NetEntry(number=1, name="GND")
    pad = Pad(
        number="1",
        pad_type="smd",
        shape="rect",
        position=Point(0.0, 0.0),
        size_x=1.0,
        size_y=1.0,
        layers=("F.Cu",),
        net_number=1,
        net_name="GND",
    )
    fp = Footprint(
        lib_id="R_0805:R_0805",
        ref="R1",
        value="10k",
        position=Point(10.0, 10.0),
        pads=(pad,),
    )
    return PCBDesign(
        outline=BoardOutline(polygon=_CLOSED_RECT),
        design_rules=_DEFAULT_RULES,
        nets=(gnd,),
        footprints=(fp,),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_pcb_no_gnd() -> PCBDesign:
    """Minimal PCB with no GND net at all."""
    vcc = NetEntry(number=1, name="VCC")
    return PCBDesign(
        outline=BoardOutline(polygon=_CLOSED_RECT),
        design_rules=_DEFAULT_RULES,
        nets=(vcc,),
        footprints=(),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_minimal_requirements(extra_nets: tuple[Net, ...] = ()) -> ProjectRequirements:
    """Minimal requirements with no components/features."""
    return ProjectRequirements(
        project=ProjectInfo(name="TestProject"),
        features=(),
        components=(),
        nets=extra_nets,
    )


# ---------------------------------------------------------------------------
# Frozen dataclass test
# ---------------------------------------------------------------------------


def test_electrical_report_frozen() -> None:
    report = ElectricalReport(violations=())
    with pytest.raises(AttributeError):
        report.violations = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Clean board
# ---------------------------------------------------------------------------


def test_electrical_clean_passes() -> None:
    """PCB with GND net and no requirements -> passed."""
    report = run_electrical_checks(_make_pcb_with_gnd())
    assert report.passed is True
    assert report.errors == ()


# ---------------------------------------------------------------------------
# power_ground_nets
# ---------------------------------------------------------------------------


def test_power_ground_net_missing_warning() -> None:
    """PCB with no GND net -> WARNING."""
    report = run_electrical_checks(_make_pcb_no_gnd())
    gnd_warnings = [
        v for v in report.warnings if v.rule == "power_ground_nets"
    ]
    assert gnd_warnings, "Expected power_ground_nets WARNING when GND is absent"


def test_power_ground_net_present_ok() -> None:
    """PCB has GND -> no power_ground_nets warning."""
    report = run_electrical_checks(_make_pcb_with_gnd())
    gnd_warnings = [v for v in report.warnings if v.rule == "power_ground_nets"]
    assert gnd_warnings == []


# ---------------------------------------------------------------------------
# net_completeness
# ---------------------------------------------------------------------------


def test_net_completeness_missing_warning() -> None:
    """Requirements net 'SPI_CLK' not in PCB -> WARNING."""
    req_net = Net(name="SPI_CLK", connections=())
    reqs = _make_minimal_requirements(extra_nets=(req_net,))
    report = run_electrical_checks(_make_pcb_with_gnd(), requirements=reqs)
    missing_warnings = [
        v for v in report.warnings if v.rule == "net_completeness"
    ]
    assert missing_warnings, "Expected net_completeness WARNING for SPI_CLK"


def test_net_completeness_ok() -> None:
    """All requirements nets present in PCB -> no net_completeness warning."""
    gnd_net = Net(
        name="GND",
        connections=(NetConnection(ref="R1", pin="1"),),
    )
    reqs = _make_minimal_requirements(extra_nets=(gnd_net,))
    report = run_electrical_checks(_make_pcb_with_gnd(), requirements=reqs)
    missing_warnings = [
        v for v in report.warnings if v.rule == "net_completeness"
    ]
    assert missing_warnings == []


# ---------------------------------------------------------------------------
# power_rail_voltage
# ---------------------------------------------------------------------------


def test_power_rail_invalid_voltage_error() -> None:
    """Power rail with voltage=0 -> ERROR."""
    bad_rail = PowerRail(name="VCC", voltage=0.0, current_ma=100.0, source_ref="U1")
    budget = PowerBudget(rails=(bad_rail,), total_current_ma=100.0, notes=())
    reqs = ProjectRequirements(
        project=ProjectInfo(name="TestProject"),
        features=(),
        components=(),
        nets=(),
        power_budget=budget,
    )
    report = run_electrical_checks(_make_pcb_with_gnd(), requirements=reqs)
    rail_errors = [v for v in report.errors if v.rule == "power_rail_voltage"]
    assert rail_errors, "Expected power_rail_voltage ERROR for voltage=0"


def test_power_rail_valid_voltage_ok() -> None:
    """Power rail with voltage=3.3 -> no power_rail_voltage error."""
    good_rail = PowerRail(name="VCC", voltage=3.3, current_ma=100.0, source_ref="U1")
    budget = PowerBudget(rails=(good_rail,), total_current_ma=100.0, notes=())
    reqs = ProjectRequirements(
        project=ProjectInfo(name="TestProject"),
        features=(),
        components=(),
        nets=(),
        power_budget=budget,
    )
    report = run_electrical_checks(_make_pcb_with_gnd(), requirements=reqs)
    rail_errors = [v for v in report.errors if v.rule == "power_rail_voltage"]
    assert rail_errors == []


# ---------------------------------------------------------------------------
# short_circuit_check
# ---------------------------------------------------------------------------


def test_short_circuit_mismatch_error() -> None:
    """Pad net_number=1 with net_name='VCC' while pcb.nets says 1='GND' -> ERROR."""
    gnd = NetEntry(number=1, name="GND")
    # Pad claims net_name="VCC" but net number 1 is "GND" in the net list.
    mismatched_pad = Pad(
        number="1",
        pad_type="smd",
        shape="rect",
        position=Point(0.0, 0.0),
        size_x=1.0,
        size_y=1.0,
        layers=("F.Cu",),
        net_number=1,
        net_name="VCC",  # conflicts with net list
    )
    fp = Footprint(
        lib_id="U_SOT23:U_SOT23",
        ref="U1",
        value="IC",
        position=Point(10.0, 10.0),
        pads=(mismatched_pad,),
    )
    pcb = PCBDesign(
        outline=BoardOutline(polygon=_CLOSED_RECT),
        design_rules=_DEFAULT_RULES,
        nets=(gnd,),
        footprints=(fp,),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )
    report = run_electrical_checks(pcb)
    sc_errors = [v for v in report.errors if v.rule == "short_circuit_check"]
    assert sc_errors, "Expected short_circuit_check ERROR for net_name mismatch"


def test_short_circuit_ok() -> None:
    """Consistent net numbering -> no short_circuit_check error."""
    report = run_electrical_checks(_make_pcb_with_gnd())
    sc_errors = [v for v in report.errors if v.rule == "short_circuit_check"]
    assert sc_errors == []


# ---------------------------------------------------------------------------
# decoupling_caps (no crash / INFO check)
# ---------------------------------------------------------------------------


def test_decoupling_cap_info() -> None:
    """IC in a feature block without a capacitor -> INFO violation (no crash)."""
    ic_comp = Component(
        ref="U1",
        value="MyIC",
        footprint="SOIC-8",
    )
    feature = FeatureBlock(
        name="MainBlock",
        description="Main processing block",
        components=("U1",),  # no capacitor in this feature
        nets=(),
        subcircuits=(),
    )
    reqs = ProjectRequirements(
        project=ProjectInfo(name="TestProject"),
        features=(feature,),
        components=(ic_comp,),
        nets=(),
    )
    # Should not raise; may emit INFO violations.
    report = run_electrical_checks(_make_pcb_with_gnd(), requirements=reqs)
    # Verify that any decoupling_caps violations are INFO severity only.
    decoup_violations = [v for v in report.violations if v.rule == "decoupling_caps"]
    for v in decoup_violations:
        assert v.severity == Severity.INFO, (
            f"Expected INFO severity for decoupling_caps, got {v.severity}"
        )
