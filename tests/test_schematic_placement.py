"""Tests for schematic placement — SymbolExtent and extent-based layout."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.requirements import Component, Pin, PinType
from kicad_pipeline.schematic.placement import (
    PlacementZone,
    SymbolExtent,
    compute_symbol_extent,
    layout_compact,
    layout_schematic,
    place_in_zone,
)
from kicad_pipeline.schematic.symbols import (
    make_lib_symbol,
    make_passive_symbol,
)

# ---------------------------------------------------------------------------
# SymbolExtent dataclass
# ---------------------------------------------------------------------------


def test_symbol_extent_width_height() -> None:
    """SymbolExtent.width and .height are correct sums."""
    ext = SymbolExtent(left=10.0, right=15.0, top=8.0, bottom=12.0)
    assert ext.width == pytest.approx(25.0)
    assert ext.height == pytest.approx(20.0)


def test_symbol_extent_frozen() -> None:
    """SymbolExtent is immutable."""
    ext = SymbolExtent(left=1.0, right=2.0, top=3.0, bottom=4.0)
    with pytest.raises(AttributeError):
        ext.left = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_symbol_extent
# ---------------------------------------------------------------------------


def test_compute_extent_passive() -> None:
    """A 2-pin passive should have left+right >= 40mm (pin tips + stubs + labels)."""
    sym = make_passive_symbol("Device:R")
    ext = compute_symbol_extent(sym, "R1", "10k")
    # Left-side pin + wire stub + label text: should be substantial
    assert ext.left > 10.0
    assert ext.right > 10.0
    # Total width should be at least 40mm (was only 20mm with old heuristic)
    assert ext.width > 40.0


def test_compute_extent_ic_multipin() -> None:
    """An IC with pins on left/right/top/bottom has extents in all directions."""
    comp = Component(
        ref="U1",
        value="ATmega328P",
        footprint="QFP-32",
        pins=tuple(
            [
                Pin(number="1", name="VCC", pin_type=PinType.POWER_IN),
                Pin(number="2", name="GND", pin_type=PinType.POWER_IN),
                Pin(number="3", name="PA0", pin_type=PinType.INPUT),
                Pin(number="4", name="PA1", pin_type=PinType.INPUT),
                Pin(number="5", name="PA2", pin_type=PinType.INPUT),
                Pin(number="6", name="PB0", pin_type=PinType.OUTPUT),
                Pin(number="7", name="PB1", pin_type=PinType.OUTPUT),
                Pin(number="8", name="PB2", pin_type=PinType.OUTPUT),
            ]
        ),
    )
    sym = make_lib_symbol(comp)
    ext = compute_symbol_extent(sym, "U1", "ATmega328P")
    # IC has pins on left, right, top (VCC), bottom (GND)
    assert ext.left > 5.0
    assert ext.right > 5.0
    assert ext.top > 5.0
    assert ext.bottom > 5.0


def test_compute_extent_larger_than_heuristic() -> None:
    """Extent-based width for a 2-pin passive must exceed the old 20mm heuristic."""
    sym = make_passive_symbol("Device:C")
    ext = compute_symbol_extent(sym, "C1", "100nF")
    # The old _h_spacing_for_pins(2) returned 20mm — extent must be larger
    assert ext.width > 20.0


def test_compute_extent_label_length_affects_width() -> None:
    """Longer ref/value text should produce a wider extent."""
    sym = make_passive_symbol("Device:R")
    short_ext = compute_symbol_extent(sym, "R1", "10k")
    long_ext = compute_symbol_extent(sym, "R_PULLUP_123", "100k_precision_resistor")
    # Long labels should make the extent wider
    assert long_ext.width > short_ext.width


# ---------------------------------------------------------------------------
# place_in_zone with extents
# ---------------------------------------------------------------------------


def _make_test_extents(refs: list[str], width: float = 50.0) -> dict[str, SymbolExtent]:
    """Create uniform extents for test refs."""
    half_w = width / 2.0
    return {
        ref: SymbolExtent(left=half_w, right=half_w, top=10.0, bottom=10.0)
        for ref in refs
    }


def test_place_in_zone_with_extents_no_overlap() -> None:
    """Extent-placed symbols should not have overlapping bounding boxes."""
    refs = ["R1", "R2", "R3"]
    extents = _make_test_extents(refs, width=50.0)
    zone = PlacementZone("TEST", 20.0, 20.0, 200.0, 100.0)
    positions = place_in_zone(refs, zone, symbol_extents=extents)

    assert len(positions) == 3
    # Check no horizontal overlap: for adjacent symbols, distance >= extent widths + gap
    pts = [positions[r] for r in refs]
    for i in range(len(pts) - 1):
        dx = abs(pts[i + 1].x - pts[i].x)
        # Must be at least right_extent_i + left_extent_j + gap
        min_dist = extents[refs[i]].right + extents[refs[i + 1]].left
        msg = f"Overlap: {refs[i]} and {refs[i + 1]}: dx={dx}, min={min_dist}"
        assert dx >= min_dist, msg


def test_place_in_zone_wraps_rows() -> None:
    """When extents exceed zone width, symbols should wrap to a new row."""
    refs = ["R1", "R2", "R3", "R4"]
    # Each symbol is 60mm wide → only ~2 fit in 140mm zone
    extents = _make_test_extents(refs, width=60.0)
    zone = PlacementZone("TEST", 20.0, 20.0, 140.0, 200.0)
    positions = place_in_zone(refs, zone, symbol_extents=extents)

    # Should have at least 2 distinct Y values (rows)
    y_values = {round(positions[r].y, 2) for r in refs}
    assert len(y_values) >= 2, f"Expected multiple rows, got Y values: {y_values}"


def test_place_in_zone_fallback_without_extents() -> None:
    """Without extents, place_in_zone still works (backward compat)."""
    refs = ["R1", "R2", "R3"]
    zone = PlacementZone("TEST", 20.0, 20.0, 200.0, 100.0)
    positions = place_in_zone(refs, zone)
    assert len(positions) == 3


# ---------------------------------------------------------------------------
# layout_schematic / layout_compact with extents
# ---------------------------------------------------------------------------


def test_layout_schematic_with_extents() -> None:
    """layout_schematic accepts and uses symbol_extents parameter."""
    refs = ["R1", "C1"]
    feature_map = {"R1": "Power", "C1": "Power"}
    extents = _make_test_extents(refs, width=50.0)
    positions = layout_schematic(
        refs, feature_map,
        pin_count_map={"R1": 2, "C1": 2},
        symbol_extents=extents,
    )
    assert len(positions) == 2
    # R1 and C1 should be in the same zone but not overlapping
    dx = abs(positions["R1"].x - positions["C1"].x)
    assert dx > 40.0  # extent-based spacing should give at least 50mm+ gap


def test_layout_compact_with_extents() -> None:
    """layout_compact accepts and uses symbol_extents parameter."""
    refs = ["R1", "R2", "R3"]
    extents = _make_test_extents(refs, width=40.0)
    positions = layout_compact(refs, symbol_extents=extents)
    assert len(positions) == 3
    # Check monotonically increasing X for first row
    xs = [positions[r].x for r in refs]
    for i in range(len(xs) - 1):
        # Either same row (x increases) or new row
        if abs(positions[refs[i]].y - positions[refs[i + 1]].y) < 1.0:
            assert xs[i + 1] > xs[i]


# ---------------------------------------------------------------------------
# Integration: no-overlap check with real symbols
# ---------------------------------------------------------------------------


def test_real_passives_no_overlap() -> None:
    """Real passive symbols placed with extents should not overlap."""
    sym_r = make_passive_symbol("Device:R")
    sym_c = make_passive_symbol("Device:C")

    refs = ["R1", "R2", "C1", "C2"]
    extents = {
        "R1": compute_symbol_extent(sym_r, "R1", "10k"),
        "R2": compute_symbol_extent(sym_r, "R2", "4.7k"),
        "C1": compute_symbol_extent(sym_c, "C1", "100nF"),
        "C2": compute_symbol_extent(sym_c, "C2", "10uF"),
    }

    zone = PlacementZone("TEST", 25.0, 25.0, 250.0, 150.0)
    positions = place_in_zone(refs, zone, symbol_extents=extents)

    # Verify no bounding box overlaps
    for i, ref_a in enumerate(refs):
        for ref_b in refs[i + 1:]:
            ext_a = extents[ref_a]
            ext_b = extents[ref_b]
            pa = positions[ref_a]
            pb = positions[ref_b]

            # Check if bounding boxes overlap
            a_left = pa.x - ext_a.left
            a_right = pa.x + ext_a.right
            a_top = pa.y - ext_a.top
            a_bottom = pa.y + ext_a.bottom

            b_left = pb.x - ext_b.left
            b_right = pb.x + ext_b.right
            b_top = pb.y - ext_b.top
            b_bottom = pb.y + ext_b.bottom

            h_overlap = a_left < b_right and a_right > b_left
            v_overlap = a_top < b_bottom and a_bottom > b_top
            assert not (h_overlap and v_overlap), (
                f"Overlap: {ref_a} bbox=({a_left:.1f},{a_top:.1f})-({a_right:.1f},{a_bottom:.1f}) "
                f"vs {ref_b} bbox=({b_left:.1f},{b_top:.1f})-({b_right:.1f},{b_bottom:.1f})"
            )
