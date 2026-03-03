"""Tests for kicad_pipeline.validation.manufacturing."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    Pad,
    PCBDesign,
    Point,
    Track,
    Via,
)
from kicad_pipeline.models.production import BOMEntry
from kicad_pipeline.validation.drc import Severity
from kicad_pipeline.validation.manufacturing import (
    ManufacturingViolation,
    run_manufacturing_checks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_outline(width: float = 80.0, height: float = 40.0) -> BoardOutline:
    """Return a rectangular board outline."""
    return BoardOutline(
        polygon=(
            Point(0.0, 0.0),
            Point(width, 0.0),
            Point(width, height),
            Point(0.0, height),
        )
    )


def _make_pcb(
    *,
    outline: BoardOutline | None = None,
    tracks: tuple[Track, ...] = (),
    vias: tuple[Via, ...] = (),
    footprints: tuple[Footprint, ...] = (),
) -> PCBDesign:
    """Return a minimal :class:`PCBDesign` for testing."""
    return PCBDesign(
        outline=outline or _make_outline(),
        design_rules=DesignRules(),
        nets=(),
        footprints=footprints,
        tracks=tracks,
        vias=vias,
        zones=(),
        keepouts=(),
    )


def _make_smd_footprint(
    ref: str = "U1",
    layer: str = "F.Cu",
    pad_size_x: float = 0.5,
    pad_size_y: float = 0.5,
    has_paste: bool = True,
) -> Footprint:
    """Return an SMD footprint with one paste pad."""
    layers: tuple[str, ...] = ("F.Cu", "F.Paste", "F.Mask") if has_paste else ("F.Cu", "F.Mask")
    pad = Pad(
        number="1",
        pad_type="smd",
        shape="rect",
        position=Point(0.0, 0.0),
        size_x=pad_size_x,
        size_y=pad_size_y,
        layers=layers,
    )
    return Footprint(
        lib_id="Device:R",
        ref=ref,
        value="10k",
        position=Point(10.0, 10.0),
        layer=layer,
        pads=(pad,),
        attr="smd",
    )


def _make_bom_entries(
    *,
    with_lcsc: bool = True,
) -> tuple[BOMEntry, ...]:
    """Return a tuple with one BOM entry."""
    return (
        BOMEntry(
            comment="10k",
            designators=("R1",),
            footprint="R_0805",
            lcsc="C17414" if with_lcsc else None,
            quantity=1,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_manufacturing_clean_passes() -> None:
    """A clean PCB with no violations should pass."""
    pcb = _make_pcb()
    report = run_manufacturing_checks(pcb)
    assert report.passed
    assert report.errors == ()


def test_manufacturing_report_frozen() -> None:
    """ManufacturingReport and ManufacturingViolation should be immutable."""
    report = run_manufacturing_checks(_make_pcb())
    with pytest.raises(AttributeError):
        report.violations = ()  # type: ignore[misc]

    v = ManufacturingViolation(rule="x", message="y", severity=Severity.ERROR)
    with pytest.raises(AttributeError):
        v.rule = "z"  # type: ignore[misc]


def test_trace_width_jlcpcb_violation() -> None:
    """A track width of 0.1mm should trigger an ERROR."""
    track = Track(
        start=Point(0.0, 0.0),
        end=Point(10.0, 0.0),
        width=0.1,
        layer="F.Cu",
        net_number=1,
    )
    pcb = _make_pcb(tracks=(track,))
    report = run_manufacturing_checks(pcb)
    assert not report.passed
    errors = report.errors
    assert len(errors) == 1
    assert errors[0].rule == "trace_width_jlcpcb"
    assert "0.100mm" in errors[0].message


def test_trace_width_jlcpcb_ok() -> None:
    """A track width of exactly 0.127mm should not trigger an error."""
    track = Track(
        start=Point(0.0, 0.0),
        end=Point(10.0, 0.0),
        width=0.127,
        layer="F.Cu",
        net_number=1,
    )
    pcb = _make_pcb(tracks=(track,))
    report = run_manufacturing_checks(pcb)
    trace_errors = [v for v in report.violations if v.rule == "trace_width_jlcpcb"]
    assert trace_errors == []


def test_via_drill_jlcpcb_violation() -> None:
    """A via drill of 0.15mm should trigger an ERROR."""
    via = Via(
        position=Point(5.0, 5.0),
        drill=0.15,
        size=0.4,
        layers=("F.Cu", "B.Cu"),
        net_number=1,
    )
    pcb = _make_pcb(vias=(via,))
    report = run_manufacturing_checks(pcb)
    assert not report.passed
    errors = [v for v in report.errors if v.rule == "via_drill_jlcpcb"]
    assert len(errors) == 1
    assert "0.150mm" in errors[0].message


def test_via_drill_jlcpcb_ok() -> None:
    """A via drill of exactly 0.2mm should not trigger an error."""
    via = Via(
        position=Point(5.0, 5.0),
        drill=0.2,
        size=0.4,
        layers=("F.Cu", "B.Cu"),
        net_number=1,
    )
    pcb = _make_pcb(vias=(via,))
    report = run_manufacturing_checks(pcb)
    via_errors = [v for v in report.violations if v.rule == "via_drill_jlcpcb"]
    assert via_errors == []


def test_board_dimensions_ok() -> None:
    """An 80x40mm board should pass dimension checks."""
    pcb = _make_pcb(outline=_make_outline(80.0, 40.0))
    report = run_manufacturing_checks(pcb)
    dim_errors = [v for v in report.violations if v.rule == "board_dimensions"]
    assert dim_errors == []


def test_board_dimensions_too_large() -> None:
    """A 600x600mm board should trigger a dimension ERROR."""
    pcb = _make_pcb(outline=_make_outline(600.0, 600.0))
    report = run_manufacturing_checks(pcb)
    assert not report.passed
    dim_errors = [v for v in report.errors if v.rule == "board_dimensions"]
    assert len(dim_errors) == 1
    assert "600" in dim_errors[0].message


def test_lcsc_missing_warning() -> None:
    """A BOM entry with no LCSC part number should trigger a WARNING."""
    pcb = _make_pcb()
    bom = _make_bom_entries(with_lcsc=False)
    report = run_manufacturing_checks(pcb, bom_entries=bom)
    lcsc_warnings = [v for v in report.warnings if v.rule == "lcsc_check"]
    assert len(lcsc_warnings) == 1
    assert "R1" in lcsc_warnings[0].message


def test_lcsc_present_ok() -> None:
    """A BOM entry with an LCSC part number should not trigger a warning."""
    pcb = _make_pcb()
    bom = _make_bom_entries(with_lcsc=True)
    report = run_manufacturing_checks(pcb, bom_entries=bom)
    lcsc_warnings = [v for v in report.violations if v.rule == "lcsc_check"]
    assert lcsc_warnings == []


def test_paste_aperture_too_small() -> None:
    """An SMD pad 0.1x0.1mm with F.Paste should trigger a WARNING."""
    fp = _make_smd_footprint(pad_size_x=0.1, pad_size_y=0.1, has_paste=True)
    pcb = _make_pcb(footprints=(fp,))
    report = run_manufacturing_checks(pcb)
    paste_warnings = [v for v in report.warnings if v.rule == "paste_aperture_check"]
    assert len(paste_warnings) >= 1
    assert "U1" in paste_warnings[0].message


def test_smt_both_sides_warning() -> None:
    """SMD footprints on both F.Cu and B.Cu should trigger a WARNING."""
    front_fp = _make_smd_footprint(ref="U1", layer="F.Cu")
    back_fp = _make_smd_footprint(ref="U2", layer="B.Cu")
    pcb = _make_pcb(footprints=(front_fp, back_fp))
    report = run_manufacturing_checks(pcb)
    smt_warnings = [v for v in report.warnings if v.rule == "smt_side_check"]
    assert len(smt_warnings) == 1
    assert "both sides" in smt_warnings[0].message.lower()
