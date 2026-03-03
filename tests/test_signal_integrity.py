"""Tests for kicad_pipeline.validation.signal_integrity."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    Keepout,
    NetEntry,
    PCBDesign,
    Point,
    Track,
)
from kicad_pipeline.validation.drc import Severity
from kicad_pipeline.validation.signal_integrity import (
    SIViolation,
    run_si_checks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_outline() -> BoardOutline:
    return BoardOutline(
        polygon=(
            Point(0.0, 0.0),
            Point(80.0, 0.0),
            Point(80.0, 40.0),
            Point(0.0, 40.0),
        )
    )


def _make_track(
    *,
    start: Point,
    end: Point,
    net_number: int = 1,
    layer: str = "F.Cu",
    width: float = 0.25,
) -> Track:
    return Track(
        start=start,
        end=end,
        width=width,
        layer=layer,
        net_number=net_number,
    )


def _make_footprint(ref: str, value: str, layer: str = "F.Cu") -> Footprint:
    return Footprint(
        lib_id="Device:U",
        ref=ref,
        value=value,
        position=Point(20.0, 20.0),
        layer=layer,
    )


def _make_pcb(
    *,
    nets: tuple[NetEntry, ...] = (),
    tracks: tuple[Track, ...] = (),
    footprints: tuple[Footprint, ...] = (),
    keepouts: tuple[Keepout, ...] = (),
) -> PCBDesign:
    return PCBDesign(
        outline=_make_outline(),
        design_rules=DesignRules(),
        nets=nets,
        footprints=footprints,
        tracks=tracks,
        vias=(),
        zones=(),
        keepouts=keepouts,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_si_clean_passes() -> None:
    """A clean PCB with no SI issues should pass."""
    pcb = _make_pcb()
    report = run_si_checks(pcb)
    assert report.passed
    assert report.errors == ()


def test_si_report_frozen() -> None:
    """SIReport should be immutable."""
    report = run_si_checks(_make_pcb())
    with pytest.raises(AttributeError):
        report.violations = ()  # type: ignore[misc]


def test_si_violation_frozen() -> None:
    """SIViolation should be immutable."""
    v = SIViolation(rule="x", message="y", severity=Severity.WARNING)
    with pytest.raises(AttributeError):
        v.rule = "z"  # type: ignore[misc]


def test_usb_diff_pair_ok() -> None:
    """D+ and D- tracks of the same length should not trigger a warning."""
    nets = (
        NetEntry(number=1, name="D+"),
        NetEntry(number=2, name="D-"),
    )
    # Both tracks are 10mm long
    tracks = (
        _make_track(start=Point(0.0, 0.0), end=Point(10.0, 0.0), net_number=1),
        _make_track(start=Point(0.0, 1.0), end=Point(10.0, 1.0), net_number=2),
    )
    pcb = _make_pcb(nets=nets, tracks=tracks)
    report = run_si_checks(pcb)
    usb_warnings = [v for v in report.violations if v.rule == "usb_diff_pair_check"]
    assert usb_warnings == []


def test_usb_diff_pair_mismatch() -> None:
    """D+ at 10mm and D- at 11mm should trigger a WARNING."""
    nets = (
        NetEntry(number=1, name="D+"),
        NetEntry(number=2, name="D-"),
    )
    tracks = (
        _make_track(start=Point(0.0, 0.0), end=Point(10.0, 0.0), net_number=1),
        _make_track(start=Point(0.0, 1.0), end=Point(11.0, 1.0), net_number=2),
    )
    pcb = _make_pcb(nets=nets, tracks=tracks)
    report = run_si_checks(pcb)
    usb_warnings = [v for v in report.violations if v.rule == "usb_diff_pair_check"]
    assert len(usb_warnings) == 1
    assert usb_warnings[0].severity == Severity.WARNING
    assert "D+" in usb_warnings[0].message
    assert "D-" in usb_warnings[0].message


def test_antenna_keepout_missing_warning() -> None:
    """An ESP32 footprint with no keepout zone should trigger a WARNING."""
    fp = _make_footprint("U1", "ESP32-S3-WROOM-1")
    pcb = _make_pcb(footprints=(fp,))
    report = run_si_checks(pcb)
    keepout_warnings = [v for v in report.violations if v.rule == "antenna_keepout_check"]
    assert len(keepout_warnings) == 1
    assert keepout_warnings[0].severity == Severity.WARNING
    assert "keepout" in keepout_warnings[0].message.lower()


def test_antenna_keepout_present_ok() -> None:
    """An ESP32 footprint with a proper antenna keepout should not warn."""
    fp = _make_footprint("U1", "ESP32-S3-WROOM-1")
    keepout = Keepout(
        polygon=(
            Point(0.0, 0.0),
            Point(20.0, 0.0),
            Point(20.0, 10.0),
            Point(0.0, 10.0),
        ),
        layers=("F.Cu",),
        no_copper=True,
        no_vias=True,
    )
    pcb = _make_pcb(footprints=(fp,), keepouts=(keepout,))
    report = run_si_checks(pcb)
    keepout_warnings = [v for v in report.violations if v.rule == "antenna_keepout_check"]
    assert keepout_warnings == []


def test_long_spi_trace_warning() -> None:
    """An SCK net with a 150mm trace should trigger a WARNING."""
    nets = (NetEntry(number=1, name="SPI_SCK"),)
    # A single track from (0,0) to (150,0) = 150mm
    tracks = (
        _make_track(start=Point(0.0, 0.0), end=Point(150.0, 0.0), net_number=1),
    )
    pcb = _make_pcb(nets=nets, tracks=tracks)
    report = run_si_checks(pcb)
    spi_warnings = [v for v in report.violations if v.rule == "trace_length_check"]
    assert len(spi_warnings) >= 1
    assert spi_warnings[0].severity == Severity.WARNING
    assert "150" in spi_warnings[0].message or "SPI" in spi_warnings[0].message
