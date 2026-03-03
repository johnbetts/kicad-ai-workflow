"""Tests for the DRC validation engine."""

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
    Track,
    Via,
)
from kicad_pipeline.validation.drc import (
    DRCReport,
    DRCViolation,
    Severity,
    run_drc,
)

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

_DEFAULT_RULES = DesignRules(
    default_trace_width_mm=0.25,
    min_via_drill_mm=0.3,
    min_via_diameter_mm=0.6,
)

_CLOSED_RECT: tuple[Point, ...] = (
    Point(0.0, 0.0),
    Point(50.0, 0.0),
    Point(50.0, 50.0),
    Point(0.0, 50.0),
    Point(0.0, 0.0),  # closed
)


def _make_clean_pcb() -> PCBDesign:
    """Board with one footprint (connected pad), one net, no tracks/vias."""
    net = NetEntry(number=1, name="GND")
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
        nets=(net,),
        footprints=(fp,),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_pcb_with_track(width_mm: float = 0.25) -> PCBDesign:
    """Return a clean PCB that also contains a single track of the given width."""
    base = _make_clean_pcb()
    track = Track(
        start=Point(0.0, 0.0),
        end=Point(10.0, 0.0),
        width=width_mm,
        layer="F.Cu",
        net_number=1,
    )
    return PCBDesign(
        outline=base.outline,
        design_rules=base.design_rules,
        nets=base.nets,
        footprints=base.footprints,
        tracks=(track,),
        vias=base.vias,
        zones=base.zones,
        keepouts=base.keepouts,
    )


def _make_pcb_with_via(drill: float = 0.3, size: float = 0.6) -> PCBDesign:
    """Return a clean PCB that also contains a single via."""
    base = _make_clean_pcb()
    via = Via(
        position=Point(5.0, 5.0),
        drill=drill,
        size=size,
        layers=("F.Cu", "B.Cu"),
        net_number=1,
    )
    return PCBDesign(
        outline=base.outline,
        design_rules=base.design_rules,
        nets=base.nets,
        footprints=base.footprints,
        tracks=base.tracks,
        vias=(via,),
        zones=base.zones,
        keepouts=base.keepouts,
    )


# ---------------------------------------------------------------------------
# Frozen dataclass tests
# ---------------------------------------------------------------------------


def test_drc_report_frozen() -> None:
    report = DRCReport(violations=())
    with pytest.raises(AttributeError):
        report.violations = ()  # type: ignore[misc]


def test_drc_violation_frozen() -> None:
    v = DRCViolation(rule="test", message="msg", severity=Severity.INFO)
    with pytest.raises(AttributeError):
        v.rule = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Clean board
# ---------------------------------------------------------------------------


def test_drc_clean_passes() -> None:
    report = run_drc(_make_clean_pcb())
    assert report.passed is True
    assert report.errors == ()


# ---------------------------------------------------------------------------
# min_trace_width
# ---------------------------------------------------------------------------


def test_min_trace_width_error() -> None:
    """Track 0.1 mm wide is below JLCPCB absolute minimum -> ERROR."""
    report = run_drc(_make_pcb_with_track(width_mm=0.1))
    rule_violations = [v for v in report.violations if v.rule == "min_trace_width"]
    assert rule_violations, "Expected at least one min_trace_width violation"
    severities = {v.severity for v in rule_violations}
    assert Severity.ERROR in severities


def test_min_trace_width_ok() -> None:
    """Track 0.25 mm wide meets the design rule -> no trace width error."""
    report = run_drc(_make_pcb_with_track(width_mm=0.25))
    errors = [v for v in report.errors if v.rule == "min_trace_width"]
    assert errors == []


# ---------------------------------------------------------------------------
# min_via_size
# ---------------------------------------------------------------------------


def test_via_drill_error() -> None:
    """Via drill 0.1 mm is below design rule minimum (0.3 mm) -> ERROR."""
    report = run_drc(_make_pcb_with_via(drill=0.1, size=0.6))
    drill_errors = [
        v
        for v in report.violations
        if v.rule == "min_via_size" and "drill" in v.message
    ]
    assert drill_errors, "Expected via drill ERROR"
    assert all(v.severity == Severity.ERROR for v in drill_errors)


def test_via_drill_ok() -> None:
    """Via drill 0.3 mm meets design rule minimum -> no drill error."""
    report = run_drc(_make_pcb_with_via(drill=0.3, size=0.6))
    drill_errors = [
        v
        for v in report.errors
        if v.rule == "min_via_size" and "drill" in v.message
    ]
    assert drill_errors == []


def test_via_size_error() -> None:
    """Via diameter 0.4 mm is below design rule minimum (0.6 mm) -> ERROR."""
    report = run_drc(_make_pcb_with_via(drill=0.3, size=0.4))
    size_errors = [
        v
        for v in report.violations
        if v.rule == "min_via_size" and "diameter" in v.message
    ]
    assert size_errors, "Expected via diameter ERROR"
    assert all(v.severity == Severity.ERROR for v in size_errors)


# ---------------------------------------------------------------------------
# board_outline
# ---------------------------------------------------------------------------


def test_board_outline_few_points_error() -> None:
    """Outline with only 2 points -> ERROR."""
    base = _make_clean_pcb()
    bad_outline = BoardOutline(polygon=(Point(0.0, 0.0), Point(10.0, 0.0)))
    pcb = PCBDesign(
        outline=bad_outline,
        design_rules=base.design_rules,
        nets=base.nets,
        footprints=base.footprints,
        tracks=base.tracks,
        vias=base.vias,
        zones=base.zones,
        keepouts=base.keepouts,
    )
    report = run_drc(pcb)
    outline_errors = [
        v for v in report.errors if v.rule == "board_outline_closed"
    ]
    assert outline_errors, "Expected board_outline_closed ERROR for 2-point polygon"


def test_board_outline_open_warning() -> None:
    """Outline where first != last point -> WARNING."""
    base = _make_clean_pcb()
    open_outline = BoardOutline(
        polygon=(
            Point(0.0, 0.0),
            Point(50.0, 0.0),
            Point(50.0, 50.0),
            Point(0.0, 50.0),
            # deliberately NOT closing the polygon
        )
    )
    pcb = PCBDesign(
        outline=open_outline,
        design_rules=base.design_rules,
        nets=base.nets,
        footprints=base.footprints,
        tracks=base.tracks,
        vias=base.vias,
        zones=base.zones,
        keepouts=base.keepouts,
    )
    report = run_drc(pcb)
    outline_warnings = [
        v for v in report.warnings if v.rule == "board_outline_closed"
    ]
    assert outline_warnings, "Expected board_outline_closed WARNING for open polygon"


def test_board_outline_4pts_ok() -> None:
    """A proper 5-point closed rectangle outline -> no outline error."""
    report = run_drc(_make_clean_pcb())
    outline_errors = [v for v in report.errors if v.rule == "board_outline_closed"]
    assert outline_errors == []


# ---------------------------------------------------------------------------
# duplicate_refs
# ---------------------------------------------------------------------------


def test_duplicate_refs_error() -> None:
    """Two footprints with the same ref -> ERROR."""
    base = _make_clean_pcb()
    fp2 = Footprint(
        lib_id="R_0805:R_0805",
        ref="R1",  # duplicate
        value="10k",
        position=Point(20.0, 20.0),
    )
    pcb = PCBDesign(
        outline=base.outline,
        design_rules=base.design_rules,
        nets=base.nets,
        footprints=(*base.footprints, fp2),
        tracks=base.tracks,
        vias=base.vias,
        zones=base.zones,
        keepouts=base.keepouts,
    )
    report = run_drc(pcb)
    dup_errors = [v for v in report.errors if v.rule == "duplicate_refs"]
    assert dup_errors, "Expected duplicate_refs ERROR"


def test_duplicate_refs_ok() -> None:
    """All unique refs -> no duplicate_refs error."""
    report = run_drc(_make_clean_pcb())
    dup_errors = [v for v in report.errors if v.rule == "duplicate_refs"]
    assert dup_errors == []


# ---------------------------------------------------------------------------
# unconnected_pads
# ---------------------------------------------------------------------------


def test_unconnected_pad_info() -> None:
    """SMD pad with net_number=0 -> INFO violation."""
    base = _make_clean_pcb()
    unconnected_pad = Pad(
        number="2",
        pad_type="smd",
        shape="rect",
        position=Point(2.0, 0.0),
        size_x=1.0,
        size_y=1.0,
        layers=("F.Cu",),
        net_number=0,
    )
    fp = Footprint(
        lib_id="R_0805:R_0805",
        ref="R2",
        value="10k",
        position=Point(20.0, 20.0),
        pads=(unconnected_pad,),
    )
    pcb = PCBDesign(
        outline=base.outline,
        design_rules=base.design_rules,
        nets=base.nets,
        footprints=(*base.footprints, fp),
        tracks=base.tracks,
        vias=base.vias,
        zones=base.zones,
        keepouts=base.keepouts,
    )
    report = run_drc(pcb)
    info_violations = [
        v
        for v in report.violations
        if v.rule == "unconnected_pads" and v.severity == Severity.INFO
    ]
    assert info_violations, "Expected unconnected_pads INFO violation"


# ---------------------------------------------------------------------------
# net_consistency
# ---------------------------------------------------------------------------


def test_net_consistency_error() -> None:
    """Pad references net_number=99 which is not in pcb.nets -> ERROR."""
    base = _make_clean_pcb()
    bad_pad = Pad(
        number="1",
        pad_type="smd",
        shape="rect",
        position=Point(0.0, 0.0),
        size_x=1.0,
        size_y=1.0,
        layers=("F.Cu",),
        net_number=99,  # not in pcb.nets
        net_name="UNKNOWN",
    )
    fp = Footprint(
        lib_id="R_0805:R_0805",
        ref="R3",
        value="10k",
        position=Point(30.0, 30.0),
        pads=(bad_pad,),
    )
    pcb = PCBDesign(
        outline=base.outline,
        design_rules=base.design_rules,
        nets=base.nets,
        footprints=(*base.footprints, fp),
        tracks=base.tracks,
        vias=base.vias,
        zones=base.zones,
        keepouts=base.keepouts,
    )
    report = run_drc(pcb)
    net_errors = [v for v in report.errors if v.rule == "net_consistency"]
    assert net_errors, "Expected net_consistency ERROR for unknown net number"
