"""Tests for group_placer module."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.requirements import FeatureBlock
from kicad_pipeline.optimization.group_placer import (
    PlacedGroup,
    _group_dimensions,
    _GroupGrid,
    pin_connectors_to_edge,
    place_groups,
)
from kicad_pipeline.optimization.zone_partitioner import BoardZone

# ---------------------------------------------------------------------------
# _GroupGrid tests
# ---------------------------------------------------------------------------

def test_group_grid_place_and_check() -> None:
    """Placing a group marks its area as occupied."""
    grid = _GroupGrid((0.0, 0.0, 100.0, 80.0))
    assert grid.is_free(25.0, 25.0, 20.0, 15.0)
    grid.place(25.0, 25.0, 20.0, 15.0)
    assert not grid.is_free(25.0, 25.0, 20.0, 15.0)


def test_group_grid_no_overlap() -> None:
    """Two groups placed far apart don't collide."""
    grid = _GroupGrid((0.0, 0.0, 100.0, 80.0))
    grid.place(20.0, 20.0, 10.0, 10.0)
    assert grid.is_free(60.0, 60.0, 10.0, 10.0)


def test_group_grid_find_free_pos_no_conflict() -> None:
    """find_free_pos returns target when position is free."""
    grid = _GroupGrid((0.0, 0.0, 100.0, 80.0))
    x, y = grid.find_free_pos(50.0, 40.0, 10.0, 10.0)
    assert x == pytest.approx(50.0)
    assert y == pytest.approx(40.0)


def test_group_grid_find_free_pos_with_conflict() -> None:
    """find_free_pos finds alternative when target is occupied."""
    grid = _GroupGrid((0.0, 0.0, 100.0, 80.0))
    grid.place(50.0, 40.0, 20.0, 20.0)
    x, y = grid.find_free_pos(50.0, 40.0, 10.0, 10.0)
    # Should be displaced from center
    assert not (x == pytest.approx(50.0) and y == pytest.approx(40.0))


# ---------------------------------------------------------------------------
# _group_dimensions tests
# ---------------------------------------------------------------------------

def test_group_dimensions_basic() -> None:
    """Dimensions computed from internal layout bounding box."""
    layout = {
        "R1": (0.0, 0.0, 0.0),
        "R2": (10.0, 0.0, 0.0),
        "U1": (5.0, 8.0, 0.0),
    }
    fp_sizes = {"R1": (2.0, 1.0), "R2": (2.0, 1.0), "U1": (5.0, 5.0)}
    w, h = _group_dimensions(layout, fp_sizes)
    # R1 at x=0 (w=2) -> min_x = -1, R2 at x=10 (w=2) -> max_x = 11
    # width = 11 - (-1) + 1 = 13
    assert w > 10.0
    assert h > 8.0


def test_group_dimensions_empty() -> None:
    """Empty layout returns default size."""
    w, h = _group_dimensions({}, {})
    assert w == 5.0
    assert h == 5.0


# ---------------------------------------------------------------------------
# place_groups tests
# ---------------------------------------------------------------------------

def _make_feature(name: str, refs: tuple[str, ...]) -> FeatureBlock:
    return FeatureBlock(
        name=name,
        description=f"Test {name}",
        components=refs,
        nets=(),
        subcircuits=(),
    )


def test_place_groups_single() -> None:
    """Single group is placed within its zone."""
    zones = [
        BoardZone("power", (10.0, 10.0, 60.0, 40.0), "top", ("Power",)),
    ]
    groups = [_make_feature("Power", ("U1", "C1", "R1"))]
    internal_layouts = {
        "Power": {
            "U1": (0.0, 0.0, 0.0),
            "C1": (5.0, 0.0, 0.0),
            "R1": (2.5, 4.0, 0.0),
        },
    }
    fp_sizes = {"U1": (5.0, 5.0), "C1": (2.0, 1.0), "R1": (2.0, 1.0)}
    board_bounds = (0.0, 0.0, 100.0, 80.0)

    placed = place_groups(zones, groups, internal_layouts, fp_sizes, board_bounds)
    assert len(placed) == 1
    pg = placed[0]
    assert pg.name == "Power"
    assert pg.zone == "power"
    assert "U1" in pg.positions
    assert "C1" in pg.positions
    assert "R1" in pg.positions

    # All positions within board bounds
    for _ref, (x, y) in pg.positions.items():
        assert 0.0 <= x <= 100.0
        assert 0.0 <= y <= 80.0


def test_place_groups_multiple_no_overlap() -> None:
    """Two groups placed in different zones don't overlap."""
    zones = [
        BoardZone("power", (10.0, 10.0, 50.0, 40.0), "top", ("Power",)),
        BoardZone("mcu", (50.0, 40.0, 90.0, 70.0), None, ("MCU",)),
    ]
    groups = [
        _make_feature("Power", ("U1", "C1")),
        _make_feature("MCU", ("U2", "R1")),
    ]
    internal_layouts = {
        "Power": {"U1": (0.0, 0.0, 0.0), "C1": (5.0, 0.0, 0.0)},
        "MCU": {"U2": (0.0, 0.0, 0.0), "R1": (5.0, 0.0, 0.0)},
    }
    fp_sizes = {"U1": (5.0, 5.0), "C1": (2.0, 1.0), "U2": (5.0, 5.0), "R1": (2.0, 1.0)}
    board_bounds = (0.0, 0.0, 100.0, 80.0)

    placed = place_groups(zones, groups, internal_layouts, fp_sizes, board_bounds)
    assert len(placed) == 2

    # Check groups are in different zones
    assert placed[0].zone != placed[1].zone or placed[0].name != placed[1].name


def test_place_groups_no_layout_skipped() -> None:
    """Group without internal layout is skipped."""
    zones = [BoardZone("power", (10.0, 10.0, 50.0, 40.0), "top", ("Power",))]
    groups = [_make_feature("Power", ("U1",))]
    internal_layouts: dict[str, dict[str, tuple[float, float, float]]] = {}
    fp_sizes = {"U1": (5.0, 5.0)}
    board_bounds = (0.0, 0.0, 100.0, 80.0)

    placed = place_groups(zones, groups, internal_layouts, fp_sizes, board_bounds)
    assert len(placed) == 0


# ---------------------------------------------------------------------------
# pin_connectors_to_edge tests
# ---------------------------------------------------------------------------

def test_pin_connectors_to_edge() -> None:
    """Connectors far from edge get pinned to nearest edge."""
    placed_groups = [
        PlacedGroup(
            name="Test",
            zone="mcu",
            origin=(30.0, 30.0),
            refs=("J1", "U1"),
            positions={"J1": (50.0, 40.0), "U1": (40.0, 40.0)},
            bbox=(30.0, 30.0, 55.0, 50.0),
        ),
    ]
    fp_sizes = {"J1": (4.0, 10.0), "U1": (5.0, 5.0)}
    board_bounds = (0.0, 0.0, 100.0, 80.0)

    result = pin_connectors_to_edge(placed_groups, fp_sizes, board_bounds, set())
    # J1 was at (50, 40) — center of board, should be moved toward an edge
    j1_x, j1_y = result["J1"]
    # Should be closer to an edge than 40mm
    min_edge_dist = min(j1_x, 100.0 - j1_x, j1_y, 80.0 - j1_y)
    assert min_edge_dist < 10.0


def test_pin_connectors_fixed_refs_unchanged() -> None:
    """Fixed connectors are not moved."""
    placed_groups = [
        PlacedGroup(
            name="Test",
            zone="mcu",
            origin=(30.0, 30.0),
            refs=("J1",),
            positions={"J1": (50.0, 40.0)},
            bbox=(30.0, 30.0, 55.0, 50.0),
        ),
    ]
    fp_sizes = {"J1": (4.0, 10.0)}
    board_bounds = (0.0, 0.0, 100.0, 80.0)

    result = pin_connectors_to_edge(placed_groups, fp_sizes, board_bounds, {"J1"})
    assert result["J1"] == (50.0, 40.0)  # unchanged
