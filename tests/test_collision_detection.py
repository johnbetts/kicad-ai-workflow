"""Tests for footprint collision detection."""

from __future__ import annotations

from kicad_pipeline.validation.collisions import check_collisions
from tests.helpers import make_footprint, make_pcb_design


def test_no_overlap_no_violations() -> None:
    """Two well-separated footprints → no violations."""
    pcb = make_pcb_design(
        footprints=(
            make_footprint("R1", x=10.0, y=20.0),
            make_footprint("R2", x=60.0, y=20.0),
        ),
    )
    violations = check_collisions(pcb)
    assert violations == ()


def test_overlapping_footprints() -> None:
    """Two footprints at the same position → collision with negative gap."""
    pcb = make_pcb_design(
        footprints=(
            make_footprint("R1", x=20.0, y=20.0),
            make_footprint("R2", x=20.0, y=20.0),
        ),
    )
    violations = check_collisions(pcb)
    assert len(violations) == 1
    assert violations[0].ref_a == "R1"
    assert violations[0].ref_b == "R2"
    assert violations[0].gap_mm < 0.0


def test_adjacent_within_gap() -> None:
    """Two footprints close together, within min_gap → violation."""
    # Default pads: pad1 at -0.5, pad2 at +0.5, size 1x1.
    # R1 at x=10: AABB x=[9, 11].  R2 at x=12: AABB x=[11, 13].
    # Gap = 0mm (touching).  With min_gap=0.5 → violation.
    pcb = make_pcb_design(
        footprints=(
            make_footprint("R1", x=10.0, y=20.0),
            make_footprint("R2", x=12.0, y=20.0),
        ),
    )
    violations = check_collisions(pcb, min_gap_mm=0.5)
    assert len(violations) == 1
    assert violations[0].gap_mm < 0.5


def test_adjacent_outside_gap() -> None:
    """Two footprints far enough apart → no violation."""
    # R1 at x=10: AABB x=[9, 11].  R2 at x=15: AABB x=[14, 16].
    # Gap = 3mm.  With min_gap=0.5 → no violation.
    pcb = make_pcb_design(
        footprints=(
            make_footprint("R1", x=10.0, y=20.0),
            make_footprint("R2", x=15.0, y=20.0),
        ),
    )
    violations = check_collisions(pcb, min_gap_mm=0.5)
    assert violations == ()


def test_single_footprint_no_collision() -> None:
    """Only one footprint → no collisions possible."""
    pcb = make_pcb_design(
        footprints=(make_footprint("R1", x=40.0, y=20.0),),
    )
    violations = check_collisions(pcb)
    assert violations == ()


def test_footprint_without_pads_skipped() -> None:
    """A footprint with no pads (mounting hole) is skipped."""
    pcb = make_pcb_design(
        footprints=(
            make_footprint("R1", x=20.0, y=20.0),
            make_footprint("MH1", x=20.0, y=20.0, pads=()),
        ),
    )
    # MH1 has no pads so it's skipped — no collision reported
    violations = check_collisions(pcb)
    assert violations == ()
