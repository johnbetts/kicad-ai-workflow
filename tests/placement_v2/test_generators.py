"""Tests for placement_v2 cell generation, arrays, and cell transforms."""

from __future__ import annotations

import math

import pytest

from kicad_pipeline.models.pcb import Footprint, Pad, Point
from kicad_pipeline.placement_v2.arrays import instantiate_array
from kicad_pipeline.placement_v2.cells import Cell, CellProof, PlacedCell, PlacedMember, Port
from kicad_pipeline.placement_v2.footprint_geom import (
    courtyard_halfdims,
    pad_offset_from_centroid,
    pad_position_in_frame,
)
from kicad_pipeline.placement_v2.generators import CellGenerationError, generate_cell
from kicad_pipeline.placement_v2.ir import (
    Axis,
    ConstraintSet,
    PadRef,
    PinAttach,
    SequenceAlong,
)


def _pad(number: str, x: float, y: float, w: float = 0.6, h: float = 0.6) -> Pad:
    return Pad(
        number=number, pad_type="smd", shape="rect",
        position=Point(x, y), size_x=w, size_y=h, layers=("F.Cu",),
    )


def passive_fp(ref: str, pitch: float = 1.0) -> Footprint:
    """Two-pad passive, pads at +-pitch/2 on the x axis."""
    return Footprint(
        lib_id="R_0402", ref=ref, value="10k", position=Point(0, 0),
        pads=(_pad("1", -pitch / 2, 0.0), _pad("2", pitch / 2, 0.0)),
    )


def ic_fp(ref: str = "U1") -> Footprint:
    """8-pad IC, pads 1-4 down the west edge, 5-8 up the east edge."""
    west = [_pad(str(i + 1), -2.5, -1.5 + i * 1.0) for i in range(4)]
    east = [_pad(str(8 - i), 2.5, -1.5 + i * 1.0) for i in range(4)]
    return Footprint(
        lib_id="SOIC-8", ref=ref, value="IC", position=Point(0, 0),
        pads=(*west, *east),
    )


class TestFootprintGeom:
    def test_pad_offset_symmetric_passive(self) -> None:
        fp = passive_fp("R1")
        assert pad_offset_from_centroid(fp, "1") == pytest.approx((-0.5, 0.0))
        assert pad_offset_from_centroid(fp, "2") == pytest.approx((0.5, 0.0))

    def test_unknown_pin_raises(self) -> None:
        with pytest.raises(KeyError, match="no pad '9'"):
            pad_offset_from_centroid(passive_fp("R1"), "9")

    def test_pad_position_with_rotation(self) -> None:
        fp = passive_fp("R1")
        # KiCad convention (negated angle): member at (10, 5) rotated 90,
        # pad 1 local (-0.5, 0) -> (0, +0.5) -> (10.0, 5.5). Matches
        # pin_map.pad_extent_in_board_space exactly.
        x, y = pad_position_in_frame(fp, "1", 10.0, 5.0, 90.0)
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(5.5)

    def test_courtyard_fallback_includes_pads(self) -> None:
        hw, hh = courtyard_halfdims(passive_fp("R1"))
        assert hw == pytest.approx(0.5 + 0.3 + 0.25)  # pitch/2 + pad half + margin
        assert hh == pytest.approx(0.3 + 0.25)


def _attach(src: str, src_pin: str, dst: str, dst_pin: str,
            max_mm: float = 5.0, ideal_mm: float = 1.0) -> PinAttach:
    return PinAttach(
        src=PadRef(src, src_pin), dst=PadRef(dst, dst_pin),
        net="N", max_mm=max_mm, ideal_mm=ideal_mm,
    )


class TestGenerateCell:
    def test_decoupling_cap_lands_beside_its_pad(self) -> None:
        fps = {"U1": ic_fp(), "C1": passive_fp("C1")}
        cs = ConstraintSet(pin_attach=(_attach("C1", "1", "U1", "1"),))
        cell = generate_cell("dec", "decoupling", "U1", fps, cs)
        c1 = next(m for m in cell.members if m.ref == "C1")
        # Pad U1.1 is on the west edge -> C1 must sit west of U1.
        assert c1.x < 0
        # Attachment distance proven within bound.
        sx, sy = pad_position_in_frame(fps["C1"], "1", c1.x, c1.y, c1.rotation_deg)
        dist = math.hypot(sx - (-2.5), sy - (-1.5))
        assert dist <= 5.0
        assert cell.proof.ok

    def test_two_caps_same_side_do_not_overlap(self) -> None:
        fps = {"U1": ic_fp(), "C1": passive_fp("C1"), "C2": passive_fp("C2")}
        cs = ConstraintSet(pin_attach=(
            _attach("C1", "1", "U1", "1"),
            _attach("C2", "1", "U1", "2"),
        ))
        cell = generate_cell("dec2", "decoupling", "U1", fps, cs)
        c1 = next(m for m in cell.members if m.ref == "C1")
        c2 = next(m for m in cell.members if m.ref == "C2")
        assert c1.x < 0 and c2.x < 0
        # Ordered along the side by their target pads (pad 1 above pad 2).
        assert c1.y < c2.y

    def test_chained_attachment_bfs_wave(self) -> None:
        # R1 attaches to U1, C1 attaches to R1 (wave 2).
        fps = {"U1": ic_fp(), "R1": passive_fp("R1"), "C1": passive_fp("C1")}
        cs = ConstraintSet(pin_attach=(
            _attach("R1", "1", "U1", "8"),
            _attach("C1", "1", "R1", "2", max_mm=6.0),
        ))
        cell = generate_cell("chain", "generic_ic_cluster", "U1", fps, cs)
        assert cell.refs == frozenset({"U1", "R1", "C1"})
        r1 = next(m for m in cell.members if m.ref == "R1")
        assert r1.x > 0  # pad 8 is on the east edge

    def test_sequence_places_monotonic(self) -> None:
        fps = {
            "U1": ic_fp(),
            "L1": passive_fp("L1", pitch=2.0),
            "C1": passive_fp("C1"),
            "C2": passive_fp("C2"),
        }
        cs = ConstraintSet(sequences=(
            SequenceAlong(axis=Axis.HORIZONTAL, refs=("L1", "C1", "C2")),
        ))
        cell = generate_cell("buck", "buck_converter", "U1", fps, cs)
        xs = {m.ref: m.x for m in cell.members}
        assert xs["L1"] < xs["C1"] < xs["C2"]

    def test_impossible_attachment_raises_with_violation(self) -> None:
        fps = {"U1": ic_fp(), "C1": passive_fp("C1")}
        cs = ConstraintSet(pin_attach=(
            _attach("C1", "1", "U1", "1", max_mm=0.01, ideal_mm=0.0),
        ))
        with pytest.raises(CellGenerationError) as exc:
            generate_cell("bad", "decoupling", "U1", fps, cs)
        assert exc.value.violations
        v = exc.value.violations[0]
        assert v.measured > v.limit
        assert "C1" in v.refs

    def test_missing_anchor_raises(self) -> None:
        with pytest.raises(Exception, match="anchor"):
            generate_cell("x", "k", "U9", {"C1": passive_fp("C1")}, ConstraintSet())

    def test_orphans_are_placed_deterministically(self) -> None:
        fps = {"U1": ic_fp(), "R9": passive_fp("R9"), "R8": passive_fp("R8")}
        cell_a = generate_cell("o", "k", "U1", fps, ConstraintSet())
        cell_b = generate_cell("o", "k", "U1", fps, ConstraintSet())
        assert cell_a.members == cell_b.members  # byte-identical reruns

    def test_hull_contains_all_members(self) -> None:
        fps = {"U1": ic_fp(), "C1": passive_fp("C1")}
        cs = ConstraintSet(pin_attach=(_attach("C1", "1", "U1", "1"),))
        cell = generate_cell("dec", "decoupling", "U1", fps, cs)
        xs = [p.x for p in cell.polygon]
        for m in cell.members:
            assert min(xs) <= m.x <= max(xs)

    def test_ports_exposed_for_external_nets(self) -> None:
        fps = {"U1": ic_fp(), "C1": passive_fp("C1")}
        cs = ConstraintSet(pin_attach=(_attach("C1", "1", "U1", "1"),))
        cell = generate_cell(
            "dec", "decoupling", "U1", fps, cs,
            external_nets={"+3V3": (PadRef("U1", "8"),)},
        )
        assert len(cell.ports) == 1
        assert cell.ports[0].net == "+3V3"


def _proven_cell() -> Cell:
    return Cell(
        name="ch", kind="relay_driver",
        members=(PlacedMember("K1", 0.0, 0.0, 0.0), PlacedMember("Q1", 0.0, 3.0, 0.0)),
        polygon=(Point(-2, -2), Point(2, -2), Point(2, 5), Point(-2, 5)),
        ports=(Port("COIL", 0.0, -2.0),),
        proof=CellProof(checks=("test",)),
    )


class TestInstantiateArray:
    def test_four_channels_at_pitch(self) -> None:
        maps = tuple(
            {"K1": f"K{i + 1}", "Q1": f"Q{i + 1}"} for i in range(4)
        )
        arr = instantiate_array("relays", "relay_array", _proven_cell(), 4,
                                pitch_mm=10.0, ref_maps=maps)
        ks = sorted(m.ref for m in arr.members if m.ref.startswith("K"))
        xs = [m.x for m in sorted(arr.members, key=lambda m: m.ref)
              if m.ref.startswith("K")]
        assert len(ks) == 4
        assert xs == pytest.approx([0.0, 10.0, 20.0, 30.0])

    def test_channels_identical_internal_geometry(self) -> None:
        maps = ({"K1": "K1", "Q1": "Q1"}, {"K1": "K2", "Q1": "Q2"})
        arr = instantiate_array("relays", "relay_array", _proven_cell(), 2,
                                pitch_mm=12.0, ref_maps=maps)
        by_ref = {m.ref: m for m in arr.members}
        # Q offset from K identical in both channels.
        d1 = (by_ref["Q1"].x - by_ref["K1"].x, by_ref["Q1"].y - by_ref["K1"].y)
        d2 = (by_ref["Q2"].x - by_ref["K2"].x, by_ref["Q2"].y - by_ref["K2"].y)
        assert d1 == pytest.approx(d2)

    def test_duplicate_refs_rejected(self) -> None:
        maps = ({"K1": "K1", "Q1": "Q1"}, {"K1": "K1", "Q1": "Q2"})
        with pytest.raises(Exception, match="duplicate ref"):
            instantiate_array("relays", "relay_array", _proven_cell(), 2,
                              ref_maps=maps)

    def test_ports_replicated_per_channel(self) -> None:
        arr = instantiate_array("relays", "relay_array", _proven_cell(), 3,
                                pitch_mm=8.0)
        assert len(arr.ports) == 3
        assert [p.x for p in arr.ports] == pytest.approx([0.0, 8.0, 16.0])


class TestPlacedCellTransforms:
    def test_members_rotate_with_cell(self) -> None:
        pc = PlacedCell(_proven_cell(), dx=50.0, dy=40.0, rotation=90)
        by_ref = {m.ref: m for m in pc.members_in_board()}
        # KiCad convention: Q1 local (0, 3) at 90 -> (3, 0) -> (53, 40).
        assert by_ref["Q1"].x == pytest.approx(53.0)
        assert by_ref["Q1"].y == pytest.approx(40.0)
        assert by_ref["Q1"].rotation_deg == pytest.approx(90.0)

    def test_ports_rotate_with_cell(self) -> None:
        pc = PlacedCell(_proven_cell(), dx=10.0, dy=10.0, rotation=180)
        (port,) = pc.ports_in_board()
        assert port.x == pytest.approx(10.0)
        assert port.y == pytest.approx(12.0)

    def test_moved_and_rotated_are_pure(self) -> None:
        pc = PlacedCell(_proven_cell(), dx=0.0, dy=0.0)
        pc2 = pc.moved_to(5.0, 5.0).rotated(270)
        assert (pc.dx, pc.dy, pc.rotation) == (0.0, 0.0, 0)
        assert (pc2.dx, pc2.dy, pc2.rotation) == (5.0, 5.0, 270)
