"""Tests for kicad_pipeline.pcb.silkscreen."""

from __future__ import annotations

from kicad_pipeline.models.pcb import Footprint, Pad, Point
from kicad_pipeline.pcb.silkscreen import (
    add_silkscreen_to_footprint,
    make_board_title,
    make_ref_label,
    make_value_label,
)


def _make_pad(number: str, x: float = 0.0, y: float = 0.0) -> Pad:
    return Pad(
        number=number,
        pad_type="smd",
        shape="rect",
        position=Point(x=x, y=y),
        size_x=1.0,
        size_y=1.0,
        layers=("F.Cu",),
        net_number=0,
        net_name="",
    )


def test_make_ref_label_type() -> None:
    label = make_ref_label("R1", Point(0, 0))
    assert label.text_type == "reference"
    assert label.text == "R1"


def test_make_value_label_hidden_by_default() -> None:
    label = make_value_label("10k", Point(0, 0))
    assert label.hidden is True


def test_make_board_title_returns_three_items() -> None:
    items = make_board_title("MyProject", "v0.1", 10.0, 5.0)
    assert len(items) == 3


def test_add_silkscreen_preserves_existing() -> None:
    """Footprint with existing ref+value texts is returned unchanged."""
    from kicad_pipeline.models.pcb import FootprintText

    ref_text = FootprintText(
        text_type="reference", text="U1",
        position=Point(0, -1.5), layer="F.SilkS",
    )
    val_text = FootprintText(
        text_type="value", text="IC",
        position=Point(0, 1.5), layer="F.SilkS",
    )
    fp = Footprint(
        lib_id="Test:IC", ref="U1", value="IC",
        position=Point(10, 10), layer="F.Cu",
        texts=(ref_text, val_text),
    )
    result = add_silkscreen_to_footprint(fp)
    assert result is fp  # Same object, nothing added


# ---------------------------------------------------------------------------
# Fix 6: Pad-aware silkscreen label placement
# ---------------------------------------------------------------------------


def test_silkscreen_labels_clear_small_footprint_pads() -> None:
    """Labels on a small footprint (pads within 1.5mm) use default offset."""
    fp = Footprint(
        lib_id="R:R_0805", ref="R1", value="10k",
        position=Point(10, 10), layer="F.Cu",
        pads=(_make_pad("1", y=-0.5), _make_pad("2", y=0.5)),
    )
    result = add_silkscreen_to_footprint(fp)
    ref_label = next(t for t in result.texts if t.text_type == "reference")
    val_label = next(t for t in result.texts if t.text_type == "value")
    # Labels should be at least 1.5mm from origin
    assert ref_label.position.y <= -1.5
    assert val_label.position.y >= 1.5


def test_silkscreen_labels_clear_large_footprint_pads() -> None:
    """Labels on a large IC (pads extending 3mm) are placed beyond pad extents."""
    # IC with pads from y=-3 to y=+3 (each 1mm tall)
    pads = (
        _make_pad("1", y=-3.0),
        _make_pad("2", y=-1.5),
        _make_pad("3", y=0.0),
        _make_pad("4", y=1.5),
        _make_pad("5", y=3.0),
    )
    fp = Footprint(
        lib_id="IC:MSOP10", ref="U1", value="ADS1115",
        position=Point(20, 20), layer="F.Cu",
        pads=pads,
    )
    result = add_silkscreen_to_footprint(fp)
    ref_label = next(t for t in result.texts if t.text_type == "reference")
    val_label = next(t for t in result.texts if t.text_type == "value")
    # Ref should be above the topmost pad extent (y=-3 - 0.5 = -3.5)
    assert ref_label.position.y <= -3.5
    # Value should be below the bottommost pad extent (y=3 + 0.5 = 3.5)
    assert val_label.position.y >= 3.5
