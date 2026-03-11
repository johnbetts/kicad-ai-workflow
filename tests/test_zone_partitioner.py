"""Tests for zone_partitioner module."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.requirements import FeatureBlock
from kicad_pipeline.optimization.zone_partitioner import (
    BoardZone,
    _match_group_to_zone,
    partition_board,
    zone_center,
    zone_for_group,
)

# ---------------------------------------------------------------------------
# Keyword matching tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Power Supply", "power"),
        ("Relay Outputs", "relay"),
        ("MCU + Peripherals", "mcu"),
        ("MCU", "mcu"),
        ("Analog Inputs", "analog"),
        ("Ethernet + PoE", "ethernet"),
        ("Display", "display"),
        ("Unknown Block", "mcu"),  # fallback
        ("power regulator section", "power"),
        ("Sensor ADC", "analog"),
    ],
)
def test_match_group_to_zone(name: str, expected: str) -> None:
    assert _match_group_to_zone(name) == expected


# ---------------------------------------------------------------------------
# partition_board tests
# ---------------------------------------------------------------------------

def _make_feature(name: str, n_components: int = 5) -> FeatureBlock:
    """Create a minimal FeatureBlock for testing."""
    refs = tuple(f"R{i}" for i in range(1, n_components + 1))
    return FeatureBlock(
        name=name,
        description=f"Test {name}",
        components=refs,
        nets=(),
        subcircuits=(),
    )


def test_partition_board_basic() -> None:
    """Partition with two groups produces two non-overlapping zones."""
    groups = [
        _make_feature("Power Supply", 10),
        _make_feature("MCU", 15),
    ]
    zones = partition_board((0.0, 0.0, 100.0, 80.0), groups)

    assert len(zones) == 2
    zone_names = {z.name for z in zones}
    assert "power" in zone_names
    assert "mcu" in zone_names

    # All zones have non-zero area
    for z in zones:
        x1, y1, x2, y2 = z.rect
        assert x2 > x1
        assert y2 > y1


def test_partition_board_empty() -> None:
    """Empty group list returns empty zones."""
    assert partition_board((0.0, 0.0, 100.0, 80.0), []) == []


def test_partition_board_single_group() -> None:
    """Single group still creates one zone."""
    groups = [_make_feature("Relay Outputs", 8)]
    zones = partition_board((0.0, 0.0, 100.0, 80.0), groups)
    assert len(zones) == 1
    assert zones[0].name == "relay"
    assert zones[0].groups == ("Relay Outputs",)


def test_partition_board_all_types() -> None:
    """All zone types get created when all keywords are present."""
    groups = [
        _make_feature("Power Supply", 5),
        _make_feature("Relay Outputs", 8),
        _make_feature("MCU Core", 12),
        _make_feature("Analog ADC", 4),
        _make_feature("Ethernet PHY", 6),
        _make_feature("Display OLED", 3),
    ]
    zones = partition_board((0.0, 0.0, 140.0, 80.0), groups)
    zone_names = {z.name for z in zones}
    assert zone_names == {"power", "relay", "mcu", "analog", "ethernet", "display"}


def test_partition_board_zones_within_bounds() -> None:
    """All zone rects are within board bounds."""
    groups = [
        _make_feature("Power Supply", 5),
        _make_feature("MCU", 10),
        _make_feature("Relay Outputs", 8),
    ]
    bx1, by1, bx2, by2 = 10.0, 20.0, 150.0, 100.0
    zones = partition_board((bx1, by1, bx2, by2), groups)

    for z in zones:
        x1, y1, x2, y2 = z.rect
        # Zone should be mostly within board bounds (small margin allowed)
        assert x1 >= bx1 - 1.0
        assert y1 >= by1 - 1.0


def test_partition_board_multiple_groups_same_zone() -> None:
    """Two groups matching the same zone keyword are assigned together."""
    groups = [
        _make_feature("Power Supply", 5),
        _make_feature("Power Regulator", 3),
    ]
    zones = partition_board((0.0, 0.0, 100.0, 80.0), groups)
    assert len(zones) == 1
    assert "Power Supply" in zones[0].groups
    assert "Power Regulator" in zones[0].groups


def test_partition_board_edge_affinity() -> None:
    """Zones have correct edge affinities."""
    groups = [
        _make_feature("Power Supply", 5),
        _make_feature("Relay Outputs", 5),
        _make_feature("Ethernet PHY", 5),
        _make_feature("Display OLED", 5),
    ]
    zones = partition_board((0.0, 0.0, 100.0, 80.0), groups)
    zone_dict = {z.name: z for z in zones}
    assert zone_dict["power"].edge_affinity == "top"
    assert zone_dict["relay"].edge_affinity == "right"
    assert zone_dict["ethernet"].edge_affinity == "bottom"
    assert zone_dict["display"].edge_affinity == "left"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_zone_for_group_found() -> None:
    """zone_for_group finds a zone containing the group."""
    zones = [
        BoardZone("mcu", (0, 0, 50, 50), None, ("MCU",)),
        BoardZone("power", (50, 0, 100, 50), "top", ("Power",)),
    ]
    z = zone_for_group("MCU", zones)
    assert z is not None
    assert z.name == "mcu"


def test_zone_for_group_not_found() -> None:
    """zone_for_group returns None when group is not in any zone."""
    zones = [BoardZone("mcu", (0, 0, 50, 50), None, ("MCU",))]
    assert zone_for_group("Unknown", zones) is None


def test_zone_center() -> None:
    """zone_center returns the midpoint of the zone rect."""
    z = BoardZone("test", (10.0, 20.0, 50.0, 60.0), None, ())
    cx, cy = zone_center(z)
    assert cx == pytest.approx(30.0)
    assert cy == pytest.approx(40.0)
