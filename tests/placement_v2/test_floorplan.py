"""Tests for placement_v2 floorplanning, composition, and legalization."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.optimization.geometry import (
    convex_polygons_overlap,
    polygon_bbox,
)
from kicad_pipeline.placement_v2.cells import Cell, CellProof, PlacedCell, PlacedMember, Port
from kicad_pipeline.placement_v2.cells_compose import compose_group_cell
from kicad_pipeline.placement_v2.floorplan import (
    Floorplan,
    pack_board,
    pack_group,
)
from kicad_pipeline.placement_v2.ir import ConstraintSet, Edge, EdgePin
from kicad_pipeline.placement_v2.legalize import LegalizationError, legalize


def _cell(name: str, w: float, h: float, nets: tuple[str, ...] = ()) -> Cell:
    half_w, half_h = w / 2, h / 2
    return Cell(
        name=name,
        kind="test",
        members=(PlacedMember(f"{name}_m", 0.0, 0.0, 0.0),),
        polygon=(
            Point(-half_w, -half_h), Point(half_w, -half_h),
            Point(half_w, half_h), Point(-half_w, half_h),
        ),
        ports=tuple(Port(net=n, x=0.0, y=0.0) for n in nets),
        proof=CellProof(checks=("test",)),
    )


class TestPackGroup:
    def test_no_overlaps(self) -> None:
        cells = (_cell("a", 10, 8), _cell("b", 6, 6), _cell("c", 4, 4))
        plan = pack_group("g", cells)
        polys = [pc.polygon_in_board() for pc in plan.cells]
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                assert not convex_polygons_overlap(polys[i], polys[j])

    def test_deterministic(self) -> None:
        cells = (_cell("a", 10, 8), _cell("b", 6, 6), _cell("c", 4, 4))
        p1 = pack_group("g", cells)
        p2 = pack_group("g", cells)
        assert p1 == p2

    def test_shared_net_cells_packed_adjacent(self) -> None:
        # b and c share a net; d does not. b/c should end up closer.
        cells = (
            _cell("a", 12, 10, nets=("X",)),
            _cell("b", 6, 6, nets=("SHARED",)),
            _cell("c", 6, 6, nets=("SHARED",)),
        )
        plan = pack_group("g", cells)
        pos = {pc.cell.name: (pc.dx, pc.dy) for pc in plan.cells}
        d_bc = abs(pos["b"][0] - pos["c"][0]) + abs(pos["b"][1] - pos["c"][1])
        d_ab = abs(pos["a"][0] - pos["b"][0]) + abs(pos["a"][1] - pos["b"][1])
        assert d_bc <= d_ab

    def test_empty_group(self) -> None:
        plan = pack_group("g", ())
        assert plan.cells == ()
        assert plan.width == 0.0


class TestComposeGroupCell:
    def test_members_flattened_in_group_frame(self) -> None:
        plan = pack_group("g", (_cell("a", 10, 8), _cell("b", 6, 6)))
        comp = compose_group_cell(plan)
        assert len(comp.members) == 2
        assert comp.name == "group:g"
        # Composite hull spans both cells.
        assert comp.width >= 10.0

    def test_ports_preserved(self) -> None:
        plan = pack_group("g", (_cell("a", 10, 8, nets=("N1",)),))
        comp = compose_group_cell(plan)
        assert [p.net for p in comp.ports] == ["N1"]


class TestPackBoard:
    def test_fits_explicit_board(self) -> None:
        groups = (
            pack_group("g1", (_cell("a", 20, 15),)),
            pack_group("g2", (_cell("b", 10, 10),)),
        )
        fp = pack_board(groups, ConstraintSet(), board_width=80.0, board_height=60.0)
        assert fp.board_width == 80.0
        for pc in fp.placed:
            x1, y1, x2, y2 = polygon_bbox(pc.polygon_in_board())
            assert x1 >= 0.5 and y1 >= 0.5
            assert x2 <= 79.5 and y2 <= 59.5

    def test_shrink_to_fit_when_no_size(self) -> None:
        groups = (
            pack_group("g1", (_cell("a", 20, 15),)),
            pack_group("g2", (_cell("b", 10, 10),)),
        )
        fp = pack_board(groups, ConstraintSet())
        assert fp.board_width > 20.0  # holds content + margins
        for pc in fp.placed:
            x1, y1, x2, y2 = polygon_bbox(pc.polygon_in_board())
            assert x1 >= 0.0 and y1 >= 0.0
            assert x2 <= fp.board_width and y2 <= fp.board_height

    def test_edge_pinned_group_lands_on_edge(self) -> None:
        groups = (
            pack_group("conn", (_cell("jcell", 12, 8),)),
            pack_group("logic", (_cell("ucell", 25, 20),)),
        )
        cs = ConstraintSet(edge_pins=(EdgePin(ref="jcell_m", edge=Edge.SOUTH),))
        fp = pack_board(groups, cs, board_width=80.0, board_height=60.0)
        conn = next(pc for pc in fp.placed if pc.cell.name == "group:conn")
        _, _, _, y2 = polygon_bbox(conn.polygon_in_board())
        assert y2 == pytest.approx(59.0)  # flush at south edge margin

    def test_infeasible_board_raises(self) -> None:
        groups = (
            pack_group("g1", (_cell("a", 30, 30),)),
            pack_group("g2", (_cell("b", 30, 30),)),
        )
        with pytest.raises(Exception, match="no feasible slot"):
            pack_board(groups, ConstraintSet(), board_width=35.0, board_height=35.0)


class TestLegalize:
    def test_overlapping_cells_separated(self) -> None:
        a = PlacedCell(_cell("a", 10, 10), 20.0, 20.0, 0)
        b = PlacedCell(_cell("b", 10, 10), 24.0, 20.0, 0)  # 6mm overlap in x
        fp = Floorplan(placed=(a, b), board_width=60.0, board_height=40.0)
        out = legalize(fp, clearance_mm=0.5)
        pa, pb = out.placed
        assert not convex_polygons_overlap(
            pa.polygon_in_board(), pb.polygon_in_board()
        )

    def test_offboard_cell_pulled_in(self) -> None:
        a = PlacedCell(_cell("a", 10, 10), -2.0, 20.0, 0)  # extends past west
        fp = Floorplan(placed=(a,), board_width=60.0, board_height=40.0)
        out = legalize(fp)
        x1, _, _, _ = polygon_bbox(out.placed[0].polygon_in_board())
        assert x1 >= 1.0 - 1e-9

    def test_pinned_cell_does_not_move(self) -> None:
        a = PlacedCell(_cell("a", 10, 10), 20.0, 20.0, 0)
        b = PlacedCell(_cell("b", 10, 10), 26.0, 20.0, 0)
        fp = Floorplan(placed=(a, b), board_width=80.0, board_height=40.0)
        out = legalize(fp, clearance_mm=1.0, pinned=frozenset({"a"}))
        pa = next(pc for pc in out.placed if pc.cell.name == "a")
        assert (pa.dx, pa.dy) == (20.0, 20.0)

    def test_infeasible_raises_with_violations(self) -> None:
        # Two 30mm cells pinned overlapping: nothing may move.
        a = PlacedCell(_cell("a", 30, 30), 20.0, 20.0, 0)
        b = PlacedCell(_cell("b", 30, 30), 25.0, 20.0, 0)
        fp = Floorplan(placed=(a, b), board_width=60.0, board_height=60.0)
        with pytest.raises(LegalizationError) as exc:
            legalize(fp, pinned=frozenset({"a", "b"}))
        assert exc.value.violations

    def test_clean_plan_unchanged(self) -> None:
        a = PlacedCell(_cell("a", 10, 10), 15.0, 15.0, 0)
        b = PlacedCell(_cell("b", 10, 10), 40.0, 15.0, 0)
        fp = Floorplan(placed=(a, b), board_width=60.0, board_height=40.0)
        out = legalize(fp)
        assert out.placed == fp.placed
