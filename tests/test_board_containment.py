"""Tests for board outline containment validation."""

from __future__ import annotations

from kicad_pipeline.validation.containment import check_board_containment
from tests.helpers import make_footprint, make_pad, make_pcb_design


def test_all_inside_no_violations() -> None:
    """All footprints inside the board → no violations."""
    pcb = make_pcb_design(
        footprints=(
            make_footprint("R1", x=20.0, y=20.0),
            make_footprint("R2", x=60.0, y=20.0),
        ),
        w=80.0,
        h=40.0,
    )
    violations = check_board_containment(pcb)
    assert violations == ()


def test_center_outside_board() -> None:
    """Footprint center at x=100 on an 80mm-wide board → violation."""
    pcb = make_pcb_design(
        footprints=(make_footprint("U1", x=100.0, y=20.0),),
        w=80.0,
        h=40.0,
    )
    violations = check_board_containment(pcb)
    center_violations = [v for v in violations if v.element == "center"]
    assert len(center_violations) == 1
    assert center_violations[0].ref == "U1"
    assert center_violations[0].overshoot_mm > 0.0


def test_pad_extends_outside() -> None:
    """Footprint at the edge with a pad that extends past the board."""
    # Board is 80mm wide (0..80).  Footprint center at x=79.5.
    # Default pad at +0.5 offset, size 1.0 → pad right edge at 80.5.
    pcb = make_pcb_design(
        footprints=(make_footprint("R1", x=79.5, y=20.0),),
        w=80.0,
        h=40.0,
    )
    violations = check_board_containment(pcb)
    pad_violations = [v for v in violations if v.element == "pad"]
    assert len(pad_violations) >= 1
    assert pad_violations[0].ref == "R1"
    assert pad_violations[0].overshoot_mm > 0.0


def test_margin_enforcement() -> None:
    """Footprint inside board but within the margin → violation."""
    # Board 80x40.  Footprint center at (78, 20) — inside the board
    # but within a 5mm margin from the right edge.
    pcb = make_pcb_design(
        footprints=(
            make_footprint(
                "R1",
                x=78.0,
                y=20.0,
                pads=(make_pad("1", x=0.0, y=0.0, size_x=0.5, size_y=0.5),),
            ),
        ),
        w=80.0,
        h=40.0,
    )
    # With 0 margin → no violations
    assert check_board_containment(pcb, margin_mm=0.0) == ()
    # With 5mm margin → violation (78 + 0.25 = 78.25 > 80 - 5 = 75)
    violations = check_board_containment(pcb, margin_mm=5.0)
    assert len(violations) > 0
    assert violations[0].ref == "R1"


def test_empty_pcb_no_violations() -> None:
    """PCB with no footprints → no violations."""
    pcb = make_pcb_design(footprints=(), w=80.0, h=40.0)
    violations = check_board_containment(pcb)
    assert violations == ()


def test_rotated_footprint_pad_outside() -> None:
    """A 90-degree rotated footprint whose pad now extends outside."""
    # Wide pad at x-offset, rotation swaps it to y-axis.
    # Board 20x10.  Footprint at (10, 9).  Pad at local (0, 0), size 1x4.
    # At 0 rotation: pad extends ±0.5 in x, ±2.0 in y → y range [7, 11] → outside.
    # At 90 rotation: pad extends ±2.0 in x, ±0.5 in y → y range [8.5, 9.5] → inside.
    # So test the 0-rotation case as the violating one.
    wide_pad = make_pad("1", x=0.0, y=0.0, size_x=1.0, size_y=4.0)
    pcb_rotated = make_pcb_design(
        footprints=(make_footprint("R1", x=10.0, y=9.0, rotation=90.0, pads=(wide_pad,)),),
        w=20.0,
        h=10.0,
    )
    pcb_unrotated = make_pcb_design(
        footprints=(make_footprint("R1", x=10.0, y=9.0, rotation=0.0, pads=(wide_pad,)),),
        w=20.0,
        h=10.0,
    )
    # Rotated 90 → pad fits within y bounds
    rotated_violations = check_board_containment(pcb_rotated)
    assert len(rotated_violations) == 0

    # Unrotated → pad extends past y=10 boundary
    unrotated_violations = check_board_containment(pcb_unrotated)
    pad_violations = [v for v in unrotated_violations if v.element == "pad"]
    assert len(pad_violations) >= 1
    assert pad_violations[0].ref == "R1"
    assert pad_violations[0].overshoot_mm > 0.0
