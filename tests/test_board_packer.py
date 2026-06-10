"""Tests for board_packer — Stage 3 of the bottom-up placement pipeline."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.models.requirements import (
    FeatureBlock,
    MechanicalConstraints,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.board_packer import pack_groups_on_board
from kicad_pipeline.optimization.group_placer import PlacedGroup
from kicad_pipeline.optimization.zone_partitioner import BoardZone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOARD_BOUNDS = (0.0, 0.0, 160.0, 80.0)
_EDGE_MARGIN = 2.0  # matches _BOARD_EDGE_MARGIN_MM in board_packer


def _make_zone(name: str, x1: float, y1: float, x2: float, y2: float) -> BoardZone:
    return BoardZone(
        name=name,
        polygon=(
            Point(x=x1, y=y1),
            Point(x=x2, y=y1),
            Point(x=x2, y=y2),
            Point(x=x1, y=y2),
        ),
        edge_affinity=None,
        groups=(name,),
    )


def _make_reqs(*feature_names: str) -> ProjectRequirements:
    features = tuple(
        FeatureBlock(name=n, description="", components=(), nets=(), subcircuits=())
        for n in feature_names
    )
    return ProjectRequirements(
        project=ProjectInfo(name="test", description=""),
        features=features,
        components=(),
        nets=(),
        mechanical=MechanicalConstraints(board_width_mm=160.0, board_height_mm=80.0),
    )


def _make_group(
    name: str,
    zone: str,
    refs: tuple[str, ...],
    positions: dict[str, tuple[float, float]],
) -> PlacedGroup:
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    bbox = (min(xs) - 1, min(ys) - 1, max(xs) + 1, max(ys) + 1)
    return PlacedGroup(
        name=name,
        zone=zone,
        origin=(min(xs), min(ys)),
        refs=refs,
        positions=positions,
        bbox=bbox,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pack_single_group() -> None:
    """One group — all refs appear in result with valid in-bounds positions."""
    zone = _make_zone("power", 0.0, 0.0, 80.0, 40.0)
    group = _make_group(
        "power", "power",
        refs=("U1", "C1"),
        positions={"U1": (40.0, 20.0), "C1": (45.0, 20.0)},
    )
    reqs = _make_reqs("power")

    result = pack_groups_on_board([group], _BOARD_BOUNDS, [zone], reqs)

    assert "U1" in result
    assert "C1" in result
    for ref, (x, y, _rot) in result.items():
        assert _EDGE_MARGIN <= x <= 160.0 - _EDGE_MARGIN, f"{ref} x={x} out of bounds"
        assert _EDGE_MARGIN <= y <= 80.0 - _EDGE_MARGIN, f"{ref} y={y} out of bounds"


def test_pack_connectors_to_edge() -> None:
    """Group containing J1 should have J1 pinned near a board edge."""
    zone = _make_zone("io", 0.0, 0.0, 160.0, 80.0)
    group = _make_group(
        "io", "io",
        refs=("J1", "R1"),
        positions={"J1": (80.0, 40.0), "R1": (85.0, 40.0)},
    )
    reqs = _make_reqs("io")

    result = pack_groups_on_board([group], _BOARD_BOUNDS, [zone], reqs)

    jx, jy, _ = result["J1"]
    dist_to_edge = min(
        jx - _BOARD_BOUNDS[0],
        _BOARD_BOUNDS[2] - jx,
        jy - _BOARD_BOUNDS[1],
        _BOARD_BOUNDS[3] - jy,
    )
    assert dist_to_edge <= 5.0, (
        f"J1 at ({jx:.1f}, {jy:.1f}) is {dist_to_edge:.1f}mm from nearest edge"
    )


def test_all_positions_in_bounds() -> None:
    """Multiple groups — every component stays within board bounds + margin."""
    zones = [
        _make_zone("power", 0.0, 0.0, 80.0, 40.0),
        _make_zone("logic", 80.0, 0.0, 160.0, 40.0),
        _make_zone("analog", 0.0, 40.0, 160.0, 80.0),
    ]
    groups = [
        _make_group("power", "power", ("U1", "C1"), {"U1": (30.0, 20.0), "C1": (35.0, 20.0)}),
        _make_group("logic", "logic", ("U2", "R1"), {"U2": (120.0, 20.0), "R1": (125.0, 20.0)}),
        _make_group("analog", "analog", ("U3",), {"U3": (80.0, 60.0)}),
    ]
    reqs = _make_reqs("power", "logic", "analog")

    result = pack_groups_on_board(groups, _BOARD_BOUNDS, zones, reqs)

    assert len(result) == 5
    for ref, (x, y, _rot) in result.items():
        assert _EDGE_MARGIN <= x <= 160.0 - _EDGE_MARGIN, f"{ref} x={x} out of bounds"
        assert _EDGE_MARGIN <= y <= 80.0 - _EDGE_MARGIN, f"{ref} y={y} out of bounds"


def test_positions_have_rotation() -> None:
    """Output tuples must be (x, y, rotation_deg)."""
    zone = _make_zone("mcu", 0.0, 0.0, 160.0, 80.0)
    group = _make_group("mcu", "mcu", ("U1",), {"U1": (80.0, 40.0)})
    reqs = _make_reqs("mcu")

    result = pack_groups_on_board([group], _BOARD_BOUNDS, [zone], reqs)

    for ref, pos in result.items():
        assert len(pos) == 3, f"{ref} position should be (x, y, rot), got {pos}"
        assert isinstance(pos[2], float), f"{ref} rotation should be float"
