"""Tests for pcb.pin_map — pad cardinal-side classification."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Footprint, Pad, Point
from kicad_pipeline.pcb.pin_map import (
    CardinalSide,
    FootprintPinMap,
    centroid_to_origin,
    classify_pad_side,
    compute_centroid_offset,
    compute_pin_map,
    compute_pin_map_for_component,
    origin_to_centroid,
    rotate_side,
)

# ---------------------------------------------------------------------------
# classify_pad_side
# ---------------------------------------------------------------------------


class TestClassifyPadSide:
    """Unit tests for classify_pad_side()."""

    def test_right_pad(self) -> None:
        assert classify_pad_side(5.0, 0.0, 5.0, 5.0) == CardinalSide.EAST

    def test_left_pad(self) -> None:
        assert classify_pad_side(-5.0, 0.0, 5.0, 5.0) == CardinalSide.WEST

    def test_bottom_pad(self) -> None:
        assert classify_pad_side(0.0, 5.0, 5.0, 5.0) == CardinalSide.SOUTH

    def test_top_pad(self) -> None:
        assert classify_pad_side(0.0, -5.0, 5.0, 5.0) == CardinalSide.NORTH

    def test_center_pad(self) -> None:
        assert classify_pad_side(0.1, 0.1, 5.0, 5.0) == CardinalSide.CENTER

    def test_diagonal_prefers_x_when_equal(self) -> None:
        # When norm_x == norm_y, X axis wins
        result = classify_pad_side(3.0, 3.0, 5.0, 5.0)
        assert result == CardinalSide.EAST

    def test_degenerate_footprint_does_not_crash(self) -> None:
        # half_w=0 should not cause division by zero
        result = classify_pad_side(1.0, 0.0, 0.0, 1.0)
        assert result == CardinalSide.EAST


# ---------------------------------------------------------------------------
# rotate_side
# ---------------------------------------------------------------------------


class TestRotateSide:
    """Unit tests for rotate_side()."""

    def test_no_rotation(self) -> None:
        assert rotate_side(CardinalSide.NORTH, 0.0) == CardinalSide.NORTH

    def test_90_degrees(self) -> None:
        assert rotate_side(CardinalSide.NORTH, 90.0) == CardinalSide.EAST

    def test_180_degrees(self) -> None:
        assert rotate_side(CardinalSide.NORTH, 180.0) == CardinalSide.SOUTH

    def test_270_degrees(self) -> None:
        assert rotate_side(CardinalSide.NORTH, 270.0) == CardinalSide.WEST

    def test_360_degrees_wraps(self) -> None:
        assert rotate_side(CardinalSide.EAST, 360.0) == CardinalSide.EAST

    def test_south_rotates_clockwise(self) -> None:
        assert rotate_side(CardinalSide.SOUTH, 90.0) == CardinalSide.WEST

    def test_center_is_invariant(self) -> None:
        assert rotate_side(CardinalSide.CENTER, 90.0) == CardinalSide.CENTER
        assert rotate_side(CardinalSide.CENTER, 180.0) == CardinalSide.CENTER

    def test_snaps_non_90_angles(self) -> None:
        # 85 degrees snaps to 90
        assert rotate_side(CardinalSide.NORTH, 85.0) == CardinalSide.EAST
        # 46 degrees snaps to 0
        assert rotate_side(CardinalSide.NORTH, 44.0) == CardinalSide.NORTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pad(number: str, x: float, y: float, net: str = "") -> Pad:
    return Pad(
        number=number,
        pad_type="smd",
        shape="rect",
        position=Point(x=x, y=y),
        size_x=1.0,
        size_y=0.5,
        layers=("F.Cu",),
        net_name=net if net else None,
    )


def _make_footprint(ref: str, pads: tuple[Pad, ...]) -> Footprint:
    return Footprint(
        lib_id="test:test",
        ref=ref,
        value="test",
        position=Point(x=50.0, y=50.0),
        pads=pads,
    )


# ---------------------------------------------------------------------------
# Two-pin passive
# ---------------------------------------------------------------------------


class TestTwoPinPassive:
    """Pin map for a two-pin 0805 resistor."""

    def test_at_0_degrees(self) -> None:
        pads = (
            _make_pad("1", -0.9, 0.0),
            _make_pad("2", 0.9, 0.0),
        )
        fp = _make_footprint("R1", pads)
        pm = compute_pin_map(fp, rotation=0.0)

        assert pm.ref == "R1"
        assert pm.side_for_pad("1") == CardinalSide.WEST
        assert pm.side_for_pad("2") == CardinalSide.EAST

    def test_at_90_degrees(self) -> None:
        pads = (
            _make_pad("1", -0.9, 0.0),
            _make_pad("2", 0.9, 0.0),
        )
        fp = _make_footprint("R1", pads)
        pm = compute_pin_map(fp, rotation=90.0)

        # WEST rotated 90 -> NORTH, EAST rotated 90 -> SOUTH
        assert pm.side_for_pad("1") == CardinalSide.NORTH
        assert pm.side_for_pad("2") == CardinalSide.SOUTH

    def test_at_180_degrees(self) -> None:
        pads = (
            _make_pad("1", -0.9, 0.0),
            _make_pad("2", 0.9, 0.0),
        )
        fp = _make_footprint("R1", pads)
        pm = compute_pin_map(fp, rotation=180.0)

        assert pm.side_for_pad("1") == CardinalSide.EAST
        assert pm.side_for_pad("2") == CardinalSide.WEST


# ---------------------------------------------------------------------------
# Generic IC (dual-row)
# ---------------------------------------------------------------------------


class TestGenericIC:
    """Pin map for a generic SOIC-style IC with dual rows."""

    @pytest.fixture()
    def soic_fp(self) -> Footprint:
        # Simplified SOIC-8: pins 1-4 left, 5-8 right
        pads = tuple(
            _make_pad(str(i), -3.0, -1.5 + (i - 1) * 1.0)
            for i in range(1, 5)
        ) + tuple(
            _make_pad(str(i), 3.0, -1.5 + (8 - i) * 1.0)
            for i in range(5, 9)
        )
        return _make_footprint("U1", pads)

    def test_odd_pins_west_even_pins_east(self, soic_fp: Footprint) -> None:
        pm = compute_pin_map(soic_fp, rotation=0.0)
        for i in range(1, 5):
            assert pm.side_for_pad(str(i)) == CardinalSide.WEST
        for i in range(5, 9):
            assert pm.side_for_pad(str(i)) == CardinalSide.EAST

    def test_rotated_90(self, soic_fp: Footprint) -> None:
        pm = compute_pin_map(soic_fp, rotation=90.0)
        # Left (WEST) becomes NORTH at 90 degrees
        for i in range(1, 5):
            assert pm.side_for_pad(str(i)) == CardinalSide.NORTH
        for i in range(5, 9):
            assert pm.side_for_pad(str(i)) == CardinalSide.SOUTH


# ---------------------------------------------------------------------------
# FootprintPinMap query methods
# ---------------------------------------------------------------------------


class TestPinMapQueries:
    """Test pads_on_side, side_for_pad, nets_on_side."""

    @pytest.fixture()
    def pin_map(self) -> FootprintPinMap:
        pads = (
            _make_pad("1", -3.0, 0.0, net="VCC"),
            _make_pad("2", 3.0, 0.0, net="SPI_CLK"),
            _make_pad("3", 0.0, -3.0, net="I2C_SDA"),
            _make_pad("4", 0.0, 3.0, net="GND"),
            _make_pad("5", 0.1, 0.1, net="THERMAL"),
        )
        fp = _make_footprint("U1", pads)
        return compute_pin_map(fp, rotation=0.0)

    def test_pads_on_side(self, pin_map: FootprintPinMap) -> None:
        west_pads = pin_map.pads_on_side(CardinalSide.WEST)
        assert len(west_pads) == 1
        assert west_pads[0].pad_number == "1"

    def test_nets_on_side(self, pin_map: FootprintPinMap) -> None:
        east_nets = pin_map.nets_on_side(CardinalSide.EAST)
        assert "SPI_CLK" in east_nets

    def test_side_for_nonexistent_pad(self, pin_map: FootprintPinMap) -> None:
        assert pin_map.side_for_pad("99") is None

    def test_center_pad_detected(self, pin_map: FootprintPinMap) -> None:
        assert pin_map.side_for_pad("5") == CardinalSide.CENTER


# ---------------------------------------------------------------------------
# compute_pin_map_for_component
# ---------------------------------------------------------------------------


class TestComputePinMapForComponent:
    """Test convenience function that generates footprint then computes map."""

    def test_resistor_0805(self) -> None:
        pm = compute_pin_map_for_component("R1", "10k", "R_0805")
        assert pm is not None
        assert pm.ref == "R1"
        assert len(pm.entries) >= 2

    def test_invalid_footprint_returns_none(self) -> None:
        pm = compute_pin_map_for_component("X1", "?", "NONEXISTENT_PACKAGE_XYZ")
        # Should not crash — returns None or a fallback
        # The footprint generator uses a fallback 0805 for unknown packages
        # so this may actually succeed. Either way it should not raise.
        assert pm is None or isinstance(pm, FootprintPinMap)


# ---------------------------------------------------------------------------
# compute_centroid_offset
# ---------------------------------------------------------------------------


class TestComputeCentroidOffset:
    """Tests for compute_centroid_offset() — single source of truth."""

    def test_symmetric_passive_zero_offset(self) -> None:
        """An 0805 resistor with pads at -0.9 and +0.9 has zero offset."""
        pads = (_make_pad("1", -0.9, 0.0), _make_pad("2", 0.9, 0.0))
        fp = _make_footprint("R1", pads)
        cx, cy = compute_centroid_offset(fp)
        assert cx == pytest.approx(0.0)
        assert cy == pytest.approx(0.0)

    def test_connector_nonzero_offset(self) -> None:
        """A 4-pin header with origin at pin 1 has a positive Y offset."""
        pads = tuple(
            _make_pad(str(i + 1), 0.0, i * 2.54)
            for i in range(4)
        )
        fp = _make_footprint("J1", pads)
        cx, cy = compute_centroid_offset(fp)
        assert cx == pytest.approx(0.0)
        # Centroid is at (0 + 7.62) / 2 = 3.81
        assert cy == pytest.approx(3.81)

    def test_no_pads_returns_zero(self) -> None:
        fp = _make_footprint("X1", ())
        assert compute_centroid_offset(fp) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# origin_to_centroid / centroid_to_origin roundtrip
# ---------------------------------------------------------------------------


class TestOriginCentroidConversion:
    """Tests for origin_to_centroid() and centroid_to_origin()."""

    def test_roundtrip_at_0_degrees(self) -> None:
        """origin → centroid → origin should be identity."""
        pads = tuple(
            _make_pad(str(i + 1), 0.0, i * 2.54)
            for i in range(6)
        )
        fp = _make_footprint("J1", pads)
        ox, oy = 30.0, 10.0
        cx, cy = origin_to_centroid(fp, ox, oy, 0.0)
        ox2, oy2 = centroid_to_origin(fp, cx, cy, 0.0)
        assert ox2 == pytest.approx(ox)
        assert oy2 == pytest.approx(oy)

    def test_roundtrip_at_90_degrees(self) -> None:
        """Roundtrip at 90 degrees rotation."""
        pads = tuple(
            _make_pad(str(i + 1), 0.0, i * 2.54)
            for i in range(6)
        )
        fp = _make_footprint("J1", pads)
        ox, oy = 40.0, 20.0
        cx, cy = origin_to_centroid(fp, ox, oy, 90.0)
        ox2, oy2 = centroid_to_origin(fp, cx, cy, 90.0)
        assert ox2 == pytest.approx(ox)
        assert oy2 == pytest.approx(oy)

    def test_symmetric_footprint_no_shift(self) -> None:
        """Symmetric pads → centroid equals origin."""
        pads = (_make_pad("1", -0.9, 0.0), _make_pad("2", 0.9, 0.0))
        fp = _make_footprint("R1", pads)
        cx, cy = origin_to_centroid(fp, 50.0, 50.0, 0.0)
        assert cx == pytest.approx(50.0)
        assert cy == pytest.approx(50.0)

    def test_centroid_shifts_with_rotation(self) -> None:
        """A 4-pin vertical header at 90 degrees shifts X instead of Y."""
        pads = tuple(
            _make_pad(str(i + 1), 0.0, i * 2.54)
            for i in range(4)
        )
        fp = _make_footprint("J1", pads)
        # At 0 degrees, centroid is offset in Y (3.81mm)
        cx0, cy0 = origin_to_centroid(fp, 0.0, 0.0, 0.0)
        assert cx0 == pytest.approx(0.0)
        assert cy0 == pytest.approx(3.81)
        # At 90 degrees, centroid offset rotates to X axis
        cx90, cy90 = origin_to_centroid(fp, 0.0, 0.0, 90.0)
        assert abs(cx90) == pytest.approx(3.81, abs=0.01)
        assert cy90 == pytest.approx(0.0, abs=0.01)
