"""Tests for the Gate A sign-off verifier (placement engine v2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.exceptions import ValidationError
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    FootprintLine,
    Keepout,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.placement_v2.ir import (
    Axis,
    BoardContain,
    CellKeepout,
    ConstraintSet,
    Edge,
    EdgePin,
    IsolationGap,
    KeepoutKind,
    PadRef,
    PinAttach,
    SequenceAlong,
    Severity,
)
from kicad_pipeline.placement_v2.verifier import (
    GateAReport,
    load_pcb_subset,
    run_gate_a,
    verify_board,
    verify_board_file,
)

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.placement_v2.ir import Violation

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

_OUTLINE = BoardOutline(
    polygon=(Point(0.0, 0.0), Point(50.0, 0.0), Point(50.0, 30.0), Point(0.0, 30.0)),
)


def _fp(
    ref: str,
    x: float,
    y: float,
    rotation: float = 0.0,
    *,
    pad_dx: float = 1.0,
    pad_size: float = 1.0,
    layer: str = "F.Cu",
) -> Footprint:
    """Two-pad chip footprint with pads at local (-pad_dx, 0) and (+pad_dx, 0)."""
    pads = tuple(
        Pad(
            number=str(i + 1),
            pad_type="smd",
            shape="rect",
            position=Point(dx, 0.0),
            size_x=pad_size,
            size_y=pad_size,
            layers=("F.Cu", "F.Paste", "F.Mask"),
        )
        for i, dx in enumerate((-pad_dx, pad_dx))
    )
    return Footprint(
        lib_id="Device:R_0805",
        ref=ref,
        value="10k",
        position=Point(x, y),
        rotation=rotation,
        layer=layer,
        pads=pads,
    )


def _board(
    footprints: tuple[Footprint, ...],
    keepouts: tuple[Keepout, ...] = (),
) -> PCBDesign:
    return PCBDesign(
        outline=_OUTLINE,
        design_rules=DesignRules(),
        nets=(),
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=keepouts,
    )


_NO_MARGIN = ConstraintSet(contain=BoardContain(margin_mm=0.0))


def _only(violations: tuple[Violation, ...], fragment: str) -> tuple[Violation, ...]:
    return tuple(v for v in violations if fragment in v.constraint)


# ---------------------------------------------------------------------------
# pin_attach
# ---------------------------------------------------------------------------


def test_pin_attach_within_bound_passes() -> None:
    pcb = _board((_fp("U1", 10.0, 10.0), _fp("C1", 13.0, 10.0)))
    cs = ConstraintSet(
        pin_attach=(
            PinAttach(PadRef("C1", "1"), PadRef("U1", "2"), "VDD", max_mm=2.0, ideal_mm=1.0),
        ),
        contain=BoardContain(margin_mm=0.0),
    )
    # U1.2 at (11,10), C1.1 at (12,10) -> 1mm apart
    assert _only(verify_board(pcb, cs), "PinAttach") == ()


def test_pin_attach_too_far_is_major() -> None:
    pcb = _board((_fp("U1", 10.0, 10.0), _fp("C1", 40.0, 10.0)))
    cs = ConstraintSet(
        pin_attach=(
            PinAttach(PadRef("C1", "1"), PadRef("U1", "2"), "VDD", max_mm=2.0, ideal_mm=1.0),
        ),
        contain=BoardContain(margin_mm=0.0),
    )
    found = _only(verify_board(pcb, cs), "PinAttach")
    assert len(found) == 1
    assert found[0].severity is Severity.MAJOR
    assert found[0].measured == pytest.approx(28.0)
    assert found[0].limit == 2.0
    assert set(found[0].refs) == {"C1", "U1"}


def test_pin_attach_uses_kicad_ccw_rotation_convention() -> None:
    """Pad at local (+1, 0) rotated 90 deg lands at origin + (0, -1) (CCW, Y-down).

    This matches pin_map.pad_extent_in_board_space (angle negated before
    the standard matrix). With the opposite sign the pad would land at
    (0, +1) and this attachment would measure 2mm and fail.
    """
    u1 = _fp("U1", 10.0, 10.0, rotation=90.0)  # U1.2 local (1,0) -> board (10, 9)
    c1 = _fp("C1", 11.0, 9.0)  # C1.1 local (-1,0) -> board (10, 9)
    cs = ConstraintSet(
        pin_attach=(
            PinAttach(PadRef("C1", "1"), PadRef("U1", "2"), "VDD", max_mm=0.1, ideal_mm=0.0),
        ),
        contain=BoardContain(margin_mm=0.0),
    )
    assert _only(verify_board(_board((u1, c1)), cs), "PinAttach") == ()


def test_pin_attach_missing_ref_is_critical() -> None:
    pcb = _board((_fp("U1", 10.0, 10.0),))
    cs = ConstraintSet(
        pin_attach=(
            PinAttach(PadRef("C9", "1"), PadRef("U1", "2"), "VDD", max_mm=2.0, ideal_mm=1.0),
        ),
        contain=BoardContain(margin_mm=0.0),
    )
    found = _only(verify_board(pcb, cs), "PinAttach")
    assert len(found) == 1
    assert found[0].severity is Severity.CRITICAL
    assert "C9" in found[0].message


def test_pin_attach_missing_pad_is_critical() -> None:
    pcb = _board((_fp("U1", 10.0, 10.0), _fp("C1", 12.0, 10.0)))
    cs = ConstraintSet(
        pin_attach=(
            PinAttach(PadRef("C1", "1"), PadRef("U1", "99"), "VDD", max_mm=2.0, ideal_mm=1.0),
        ),
        contain=BoardContain(margin_mm=0.0),
    )
    found = _only(verify_board(pcb, cs), "PinAttach")
    assert len(found) == 1
    assert found[0].severity is Severity.CRITICAL


# ---------------------------------------------------------------------------
# sequences
# ---------------------------------------------------------------------------


def _seq_board(*xs: float) -> PCBDesign:
    return _board(tuple(_fp(f"K{i + 1}", x, 15.0) for i, x in enumerate(xs)))


def test_sequence_in_order_passes() -> None:
    cs = ConstraintSet(
        sequences=(SequenceAlong(Axis.HORIZONTAL, ("K1", "K2", "K3")),),
        contain=BoardContain(margin_mm=0.0),
    )
    assert _only(verify_board(_seq_board(10.0, 20.0, 30.0), cs), "SequenceAlong") == ()


def test_sequence_reversed_direction_also_passes() -> None:
    """'In order along axis' is direction-agnostic (strictly monotonic)."""
    cs = ConstraintSet(
        sequences=(SequenceAlong(Axis.HORIZONTAL, ("K1", "K2", "K3")),),
        contain=BoardContain(margin_mm=0.0),
    )
    assert _only(verify_board(_seq_board(30.0, 20.0, 10.0), cs), "SequenceAlong") == ()


def test_sequence_out_of_order_is_major() -> None:
    cs = ConstraintSet(
        sequences=(SequenceAlong(Axis.HORIZONTAL, ("K1", "K2", "K3")),),
        contain=BoardContain(margin_mm=0.0),
    )
    found = _only(verify_board(_seq_board(10.0, 30.0, 20.0), cs), "SequenceAlong")
    assert len(found) == 1
    assert found[0].severity is Severity.MAJOR


def test_sequence_vertical_axis_uses_y() -> None:
    pcb = _board((_fp("K1", 10.0, 5.0), _fp("K2", 10.0, 25.0), _fp("K3", 10.0, 15.0)))
    cs = ConstraintSet(
        sequences=(SequenceAlong(Axis.VERTICAL, ("K1", "K2", "K3")),),
        contain=BoardContain(margin_mm=0.0),
    )
    assert len(_only(verify_board(pcb, cs), "SequenceAlong")) == 1


def test_sequence_pitch_deviation_is_major() -> None:
    cs = ConstraintSet(
        sequences=(SequenceAlong(Axis.HORIZONTAL, ("K1", "K2", "K3"), pitch_mm=5.0),),
        contain=BoardContain(margin_mm=0.0),
    )
    ok = _only(verify_board(_seq_board(10.0, 15.0, 20.0), cs), "SequenceAlong")
    assert ok == ()
    bad = _only(verify_board(_seq_board(10.0, 15.0, 21.0), cs), "SequenceAlong")
    assert len(bad) == 1
    assert bad[0].measured == pytest.approx(1.0)


def test_sequence_max_span_is_major() -> None:
    cs = ConstraintSet(
        sequences=(
            SequenceAlong(Axis.HORIZONTAL, ("K1", "K2", "K3"), max_span_mm=15.0),
        ),
        contain=BoardContain(margin_mm=0.0),
    )
    found = _only(verify_board(_seq_board(10.0, 20.0, 30.0), cs), "SequenceAlong")
    assert len(found) == 1
    assert found[0].measured == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# edge_pins
# ---------------------------------------------------------------------------


def test_edge_pin_near_named_edge_passes() -> None:
    pcb = _board((_fp("J1", 2.0, 15.0),))
    cs = ConstraintSet(
        edge_pins=(EdgePin("J1", edge=Edge.WEST),),
        contain=BoardContain(margin_mm=0.0),
    )
    assert _only(verify_board(pcb, cs), "EdgePin") == ()


def test_edge_pin_far_from_edge_is_major() -> None:
    pcb = _board((_fp("J1", 25.0, 15.0),))
    cs = ConstraintSet(
        edge_pins=(EdgePin("J1", edge=Edge.WEST),),
        contain=BoardContain(margin_mm=0.0),
    )
    found = _only(verify_board(pcb, cs), "EdgePin")
    assert len(found) == 1
    assert found[0].severity is Severity.MAJOR
    # Courtyard extent: pad centers at x=24, minus pad half-width 0.5
    # and courtyard fallback margin 0.25 -> min_x = 23.25.
    assert found[0].measured == pytest.approx(23.25)


def test_edge_pin_nearest_edge_when_unnamed() -> None:
    near_north = _board((_fp("J1", 25.0, 2.0),))
    cs = ConstraintSet(
        edge_pins=(EdgePin("J1", edge=None),),
        contain=BoardContain(margin_mm=0.0),
    )
    assert _only(verify_board(near_north, cs), "EdgePin") == ()
    centred = _board((_fp("J1", 25.0, 15.0),))
    assert len(_only(verify_board(centred, cs), "EdgePin")) == 1


# ---------------------------------------------------------------------------
# contain
# ---------------------------------------------------------------------------


def test_contain_inside_passes() -> None:
    pcb = _board((_fp("R1", 25.0, 15.0),))
    assert _only(verify_board(pcb, ConstraintSet()), "BoardContain") == ()


def test_contain_pad_off_board_is_critical() -> None:
    pcb = _board((_fp("R1", 0.5, 15.0),))  # pad 1 corner at x = -1.0
    found = _only(verify_board(pcb, ConstraintSet()), "BoardContain")
    # Both the pad check and the courtyard (body) check fire.
    pad_v = [v for v in found if "pad clearance" in v.message]
    body_v = [v for v in found if "courtyard/body" in v.message]
    assert len(pad_v) == 1 and len(body_v) == 1
    assert pad_v[0].severity is Severity.CRITICAL
    assert pad_v[0].measured == pytest.approx(-1.0)
    assert pad_v[0].refs == ("R1",)


def test_contain_respects_margin() -> None:
    pcb = _board((_fp("R1", 2.0, 15.0),))  # pad corner clearance = 0.5mm
    assert _only(verify_board(pcb, _NO_MARGIN), "BoardContain") == ()
    tight = ConstraintSet(contain=BoardContain(margin_mm=1.0))
    found = _only(verify_board(pcb, tight), "BoardContain")
    assert len(found) == 1
    assert found[0].measured == pytest.approx(0.5)


def test_contain_is_rotation_aware() -> None:
    """At 90 deg the pads swing onto the Y axis and cross the north edge."""
    upright = _board((_fp("R1", 25.0, 1.0, rotation=0.0),))
    assert _only(verify_board(upright, _NO_MARGIN), "BoardContain") == ()
    rotated = _board((_fp("R1", 25.0, 1.0, rotation=90.0),))
    found = _only(verify_board(rotated, _NO_MARGIN), "BoardContain")
    assert [v for v in found if "pad clearance" in v.message]


def _fp_with_courtyard(
    ref: str, x: float, y: float, court_half_w: float, court_half_h: float,
) -> Footprint:
    """Small-pad footprint with explicit oversized courtyard graphics.

    Models a module (ESP32) whose body extends well past its pad field.
    Courtyard lines are in footprint-local frame around the pad centroid
    (pads are symmetric about the origin here).
    """
    base = _fp(ref, x, y)
    lines = tuple(
        FootprintLine(start=Point(x1, y1), end=Point(x2, y2), layer="F.CrtYd")
        for (x1, y1), (x2, y2) in (
            ((-court_half_w, -court_half_h), (court_half_w, -court_half_h)),
            ((court_half_w, -court_half_h), (court_half_w, court_half_h)),
            ((court_half_w, court_half_h), (-court_half_w, court_half_h)),
            ((-court_half_w, court_half_h), (-court_half_w, -court_half_h)),
        )
    )
    return Footprint(
        lib_id=base.lib_id, ref=base.ref, value=base.value,
        position=base.position, rotation=base.rotation, layer=base.layer,
        pads=base.pads, graphics=lines,
    )


def test_contain_body_off_board_pads_inside_is_critical() -> None:
    """Gate C 2026-06-11 item 4a: ESP32 pads in-board, module body off it.

    Pads sit 10mm inside the outline but the 8mm-half-height courtyard
    crosses the north edge (y=0) — the body check must fire even though
    every pad passes with margin to spare.
    """
    pcb = _board((_fp_with_courtyard("U1", 25.0, 5.0, 6.0, 8.0),))
    found = _only(verify_board(pcb, _NO_MARGIN), "BoardContain")
    body_v = [v for v in found if "courtyard/body" in v.message]
    assert len(body_v) == 1
    assert body_v[0].severity is Severity.CRITICAL
    assert body_v[0].measured == pytest.approx(-3.0)  # 8 - 5 past the edge
    assert not [v for v in found if "pad clearance" in v.message]


def test_contain_body_flush_with_edge_passes() -> None:
    """Flush is legal — edge connectors sit exactly on the outline."""
    pcb = _board((_fp_with_courtyard("J1", 25.0, 8.0, 6.0, 8.0),))
    assert _only(verify_board(pcb, _NO_MARGIN), "BoardContain") == ()


# ---------------------------------------------------------------------------
# courtyard collisions
# ---------------------------------------------------------------------------


def test_courtyards_separated_pass() -> None:
    pcb = _board((_fp("R1", 10.0, 10.0), _fp("R2", 20.0, 10.0)))
    assert _only(verify_board(pcb, _NO_MARGIN), "courtyard") == ()


def test_courtyard_overlap_is_critical() -> None:
    # fallback courtyard is pad bbox + 0.25mm: half-width 1.75mm each
    pcb = _board((_fp("R1", 10.0, 10.0), _fp("R2", 12.0, 10.0)))
    found = _only(verify_board(pcb, _NO_MARGIN), "courtyard")
    assert len(found) == 1
    assert found[0].severity is Severity.CRITICAL
    assert found[0].measured == pytest.approx(1.5)
    assert set(found[0].refs) == {"R1", "R2"}


def test_courtyards_on_opposite_layers_do_not_collide() -> None:
    pcb = _board((
        _fp("R1", 10.0, 10.0, layer="F.Cu"),
        _fp("R2", 10.0, 10.0, layer="B.Cu"),
    ))
    assert _only(verify_board(pcb, _NO_MARGIN), "courtyard") == ()


# ---------------------------------------------------------------------------
# keepouts
# ---------------------------------------------------------------------------

_KEEPOUT_LOCAL = (Point(2.0, -2.0), Point(6.0, -2.0), Point(6.0, 2.0), Point(2.0, 2.0))
_KEEPOUT_BOARD = (Point(22.0, 13.0), Point(26.0, 13.0), Point(26.0, 17.0), Point(22.0, 17.0))


def _keepout_constraints() -> ConstraintSet:
    return ConstraintSet(
        keepouts=(
            CellKeepout(owner="U1", polygon=_KEEPOUT_LOCAL, kind=KeepoutKind.RF_ANTENNA),
        ),
        contain=BoardContain(margin_mm=0.0),
    )


def test_keepout_clear_with_matching_zone_passes() -> None:
    pcb = _board(
        (_fp("U1", 20.0, 15.0), _fp("R2", 40.0, 15.0)),
        keepouts=(Keepout(polygon=_KEEPOUT_BOARD, layers=("F.Cu",)),),
    )
    assert _only(verify_board(pcb, _keepout_constraints()), "CellKeepout") == ()


def test_keepout_intruding_pad_is_critical() -> None:
    pcb = _board(
        (_fp("U1", 20.0, 15.0), _fp("R2", 24.0, 15.0)),
        keepouts=(Keepout(polygon=_KEEPOUT_BOARD, layers=("F.Cu",)),),
    )
    found = _only(verify_board(pcb, _keepout_constraints()), "CellKeepout")
    assert len(found) == 1
    assert found[0].severity is Severity.CRITICAL
    assert found[0].refs == ("U1", "R2")


def test_keepout_missing_board_zone_is_critical() -> None:
    pcb = _board((_fp("U1", 20.0, 15.0), _fp("R2", 40.0, 15.0)))
    found = _only(verify_board(pcb, _keepout_constraints()), "CellKeepout")
    assert len(found) == 1
    assert "no board keepout zone" in found[0].message


def test_keepout_transforms_with_owner_rotation() -> None:
    """Owner rotated 90 deg CCW: local east keepout swings to board north."""
    rotated_zone = (Point(18.0, 9.0), Point(22.0, 9.0), Point(22.0, 13.0), Point(18.0, 13.0))
    pcb = _board(
        (_fp("U1", 20.0, 15.0, rotation=90.0), _fp("R2", 20.0, 11.0)),
        keepouts=(Keepout(polygon=rotated_zone, layers=("F.Cu",)),),
    )
    found = _only(verify_board(pcb, _keepout_constraints()), "CellKeepout")
    assert len(found) == 1  # R2 pads now sit inside the rotated keepout
    assert found[0].refs == ("U1", "R2")


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------

_DOMAINS = (("K1", "MAINS"), ("R1", "LOGIC"))


def test_isolation_gap_satisfied_passes() -> None:
    pcb = _board((_fp("K1", 10.0, 15.0), _fp("R1", 20.0, 15.0)))
    cs = ConstraintSet(
        isolation=(IsolationGap("MAINS", "LOGIC", min_mm=5.0),),
        contain=BoardContain(margin_mm=0.0),
    )
    assert _only(verify_board(pcb, cs, domains=_DOMAINS), "IsolationGap") == ()


def test_isolation_gap_too_small_is_major() -> None:
    pcb = _board((_fp("K1", 10.0, 15.0), _fp("R1", 20.0, 15.0)))
    cs = ConstraintSet(
        isolation=(IsolationGap("MAINS", "LOGIC", min_mm=8.0),),
        contain=BoardContain(margin_mm=0.0),
    )
    found = _only(verify_board(pcb, cs, domains=_DOMAINS), "IsolationGap")
    assert len(found) == 1
    assert found[0].severity is Severity.MAJOR
    # courtyards are 3.5mm wide -> edge gap = 10 - 1.75 - 1.75 = 6.5mm
    assert found[0].measured == pytest.approx(6.5)


def test_isolation_without_domains_checks_nothing() -> None:
    pcb = _board((_fp("K1", 10.0, 15.0), _fp("R1", 12.0, 15.0)))
    cs = ConstraintSet(
        isolation=(IsolationGap("MAINS", "LOGIC", min_mm=8.0),),
        contain=BoardContain(margin_mm=0.0),
    )
    assert _only(verify_board(pcb, cs), "IsolationGap") == ()


# ---------------------------------------------------------------------------
# file round-trip + GateAReport
# ---------------------------------------------------------------------------


def _write(pcb: PCBDesign, path: Path) -> Path:
    from kicad_pipeline.pcb.builder import write_pcb

    dest = path / "board.kicad_pcb"
    write_pcb(pcb, dest, fill_zones=False)
    return dest


def test_round_trip_clean_board_passes(tmp_path: Path) -> None:
    # 3.6mm spacing: U1.2 -> C1.1 distance 1.6mm (within 2.0) and the
    # 3.5mm-wide fallback courtyards stay 0.1mm apart.
    pcb = _board((_fp("U1", 10.0, 10.0), _fp("C1", 13.6, 10.0)))
    cs = ConstraintSet(
        pin_attach=(
            PinAttach(PadRef("C1", "1"), PadRef("U1", "2"), "VDD", max_mm=2.0, ideal_mm=1.0),
        ),
    )
    dest = _write(pcb, tmp_path)
    assert verify_board_file(dest, cs) == ()


def test_round_trip_catches_pad_off_board(tmp_path: Path) -> None:
    pcb = _board((_fp("R1", 0.5, 15.0),))  # planted: pad corner at x = -1.0
    dest = _write(pcb, tmp_path)
    found = verify_board_file(dest, ConstraintSet())
    contain = [v for v in _only(found, "BoardContain") if "pad clearance" in v.message]
    assert len(contain) == 1
    assert contain[0].severity is Severity.CRITICAL
    assert contain[0].refs == ("R1",)


def test_load_pcb_subset_round_trips_geometry(tmp_path: Path) -> None:
    pcb = _board((_fp("U1", 10.0, 10.0, rotation=90.0),))
    loaded = load_pcb_subset(_write(pcb, tmp_path))
    assert len(loaded.footprints) == 1
    fp = loaded.footprints[0]
    assert fp.ref == "U1"
    assert (fp.position.x, fp.position.y, fp.rotation) == (10.0, 10.0, 90.0)
    assert tuple(p.number for p in fp.pads) == ("1", "2")
    assert fp.pads[0].position == Point(-1.0, 0.0)
    xs = sorted({p.x for p in loaded.outline.polygon})
    ys = sorted({p.y for p in loaded.outline.polygon})
    assert (xs, ys) == ([0.0, 50.0], [0.0, 30.0])


def test_load_pcb_subset_rejects_non_pcb(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_board.kicad_pcb"
    bad.write_text("(kicad_sch (version 1))", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_pcb_subset(bad)


def test_run_gate_a_report(tmp_path: Path) -> None:
    pcb = _board((_fp("U1", 10.0, 10.0), _fp("C1", 13.6, 10.0)))
    cs = ConstraintSet(
        pin_attach=(
            PinAttach(PadRef("C1", "1"), PadRef("U1", "2"), "VDD", max_mm=2.0, ideal_mm=1.0),
        ),
    )
    report = run_gate_a(_write(pcb, tmp_path), cs)
    assert isinstance(report, GateAReport)
    assert report.passed
    assert "pin_attach x1" in report.checks_run
    assert "contain_pad x2" in report.checks_run
    assert "contain_courtyard x2" in report.checks_run
    assert "courtyard_pair x1" in report.checks_run
    assert any("face_out" in c for c in report.checks_run)


def test_run_gate_a_fails_on_planted_violation(tmp_path: Path) -> None:
    pcb = _board((_fp("R1", 0.5, 15.0),))
    report = run_gate_a(_write(pcb, tmp_path), ConstraintSet())
    assert not report.passed
    assert any(v.severity is Severity.CRITICAL for v in report.violations)


def test_gate_a_report_minor_only_passes() -> None:
    from kicad_pipeline.placement_v2.ir import Violation as IrViolation

    minor = IrViolation("x", ("R1",), Severity.MINOR, 1.0, 2.0, "minor only")
    assert GateAReport(violations=(minor,), checks_run=()).passed
    major = IrViolation("x", ("R1",), Severity.MAJOR, 3.0, 2.0, "major")
    assert not GateAReport(violations=(minor, major), checks_run=()).passed
