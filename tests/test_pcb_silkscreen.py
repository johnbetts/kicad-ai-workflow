"""Tests for kicad_pipeline.pcb.silkscreen."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Footprint, FootprintLine, FootprintText, Point
from kicad_pipeline.pcb.silkscreen import (
    add_silkscreen_to_footprint,
    make_board_title,
    make_pin1_indicator,
    make_ref_label,
    make_value_label,
    make_zone_label,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_footprint(ref: str = "R1", value: str = "10k") -> Footprint:
    """Return a minimal Footprint with no texts."""
    return Footprint(
        lib_id="R_SMD:R_0805_2012Metric",
        ref=ref,
        value=value,
        position=Point(x=10.0, y=20.0),
        texts=(),
    )


def _footprint_with_texts(ref: str = "R1", value: str = "10k") -> Footprint:
    """Return a Footprint that already has reference and value texts."""
    ref_text = FootprintText(
        text_type="reference",
        text=ref,
        position=Point(0.0, -1.5),
        layer="F.Silkscreen",
    )
    val_text = FootprintText(
        text_type="value",
        text=value,
        position=Point(0.0, 1.5),
        layer="F.Silkscreen",
    )
    return Footprint(
        lib_id="R_SMD:R_0805_2012Metric",
        ref=ref,
        value=value,
        position=Point(x=10.0, y=20.0),
        texts=(ref_text, val_text),
    )


# ---------------------------------------------------------------------------
# make_ref_label
# ---------------------------------------------------------------------------


def test_make_ref_label() -> None:
    """make_ref_label returns FootprintText with text_type='reference'."""
    label = make_ref_label("R1", Point(x=0.0, y=-1.5))
    assert isinstance(label, FootprintText)
    assert label.text_type == "reference"
    assert label.text == "R1"


def test_make_ref_label_layer() -> None:
    """make_ref_label defaults to F.Silkscreen layer."""
    label = make_ref_label("U1", Point(x=0.0, y=0.0))
    assert label.layer == "F.Silkscreen"


def test_make_ref_label_not_hidden() -> None:
    """make_ref_label produces a visible (not hidden) label."""
    label = make_ref_label("C5", Point(x=1.0, y=2.0))
    assert label.hidden is False


def test_make_ref_label_custom_size() -> None:
    """make_ref_label respects the size_mm argument."""
    label = make_ref_label("C5", Point(x=1.0, y=2.0), size_mm=2.0)
    assert label.effects_size == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# make_value_label
# ---------------------------------------------------------------------------


def test_make_value_label() -> None:
    """make_value_label returns FootprintText with text_type='value'."""
    label = make_value_label("10k", Point(x=0.0, y=1.5))
    assert isinstance(label, FootprintText)
    assert label.text_type == "value"
    assert label.text == "10k"


def test_make_value_label_hidden_by_default() -> None:
    """Value label is hidden by default."""
    label = make_value_label("100nF", Point(x=0.0, y=1.5))
    assert label.hidden is True


def test_make_value_label_not_hidden() -> None:
    """Value label is visible when hidden=False is passed."""
    label = make_value_label("10k", Point(x=0.0, y=1.5), hidden=False)
    assert label.hidden is False


# ---------------------------------------------------------------------------
# make_zone_label
# ---------------------------------------------------------------------------


def test_make_zone_label() -> None:
    """make_zone_label returns FootprintText with text_type='user' and correct text."""
    label = make_zone_label("POWER", x=5.0, y=10.0)
    assert isinstance(label, FootprintText)
    assert label.text_type == "user"
    assert label.text == "POWER"


def test_make_zone_label_size() -> None:
    """make_zone_label uses effects_size of 1.2 mm."""
    label = make_zone_label("ETHERNET", x=0.0, y=0.0)
    assert label.effects_size == pytest.approx(1.2)


def test_make_zone_label_default_layer() -> None:
    """make_zone_label defaults to F.Silkscreen layer."""
    label = make_zone_label("ANALOG", x=0.0, y=0.0)
    assert label.layer == "F.Silkscreen"


# ---------------------------------------------------------------------------
# make_board_title
# ---------------------------------------------------------------------------


def test_make_board_title_returns_3_items() -> None:
    """make_board_title returns a list of exactly 3 items."""
    items = make_board_title("MyProject", "v1.0", x=5.0, y=5.0)
    assert len(items) == 3


def test_make_board_title_content() -> None:
    """Board title items contain project name, 'Rev:' prefix, and 'Date:' prefix."""
    items = make_board_title("MyProject", "v1.0", x=0.0, y=0.0)
    texts = [item.text for item in items]
    assert any("MyProject" in t for t in texts), "Project name missing from title block"
    assert any("Rev:" in t for t in texts), "'Rev:' prefix missing from title block"
    assert any("Date:" in t for t in texts), "'Date:' prefix missing from title block"


def test_make_board_title_positions() -> None:
    """Each title block item is at a different y coordinate."""
    items = make_board_title("Proj", "v0.1", x=0.0, y=0.0)
    ys = [item.position.y for item in items]
    # All three y positions should be distinct
    assert len(set(ys)) == 3


def test_make_board_title_types_are_user() -> None:
    """All title block items have text_type='user'."""
    items = make_board_title("Proj", "v0.1", x=0.0, y=0.0)
    for item in items:
        assert item.text_type == "user"


# ---------------------------------------------------------------------------
# make_pin1_indicator
# ---------------------------------------------------------------------------


def test_make_pin1_indicator() -> None:
    """make_pin1_indicator returns FootprintLine on F.Silkscreen."""
    line = make_pin1_indicator(x=0.0, y=0.0)
    assert isinstance(line, FootprintLine)
    assert line.layer == "F.Silkscreen"


def test_make_pin1_indicator_extends_right() -> None:
    """Pin-1 indicator line extends from (x, y) rightward."""
    line = make_pin1_indicator(x=1.0, y=2.0)
    assert line.start.x == pytest.approx(1.0)
    assert line.start.y == pytest.approx(2.0)
    assert line.end.x > line.start.x


# ---------------------------------------------------------------------------
# add_silkscreen_to_footprint
# ---------------------------------------------------------------------------


def test_add_silkscreen_to_footprint_adds_missing() -> None:
    """Footprint with no texts gets ref and value labels added."""
    fp = _bare_footprint()
    updated = add_silkscreen_to_footprint(fp)
    text_types = {t.text_type for t in updated.texts}
    assert "reference" in text_types
    assert "value" in text_types


def test_add_silkscreen_to_footprint_no_duplicate() -> None:
    """Footprint with existing ref/value → texts count stays the same after call."""
    fp = _footprint_with_texts()
    original_count = len(fp.texts)
    updated = add_silkscreen_to_footprint(fp)
    # No duplicates added
    assert len(updated.texts) == original_count
    ref_count = sum(1 for t in updated.texts if t.text_type == "reference")
    val_count = sum(1 for t in updated.texts if t.text_type == "value")
    assert ref_count == 1
    assert val_count == 1


def test_add_silkscreen_to_footprint_returns_footprint() -> None:
    """add_silkscreen_to_footprint returns a Footprint instance."""
    fp = _bare_footprint()
    result = add_silkscreen_to_footprint(fp)
    assert isinstance(result, Footprint)


def test_add_silkscreen_ref_text_correct() -> None:
    """Added reference label has the footprint's ref as its text."""
    fp = _bare_footprint(ref="Q7")
    updated = add_silkscreen_to_footprint(fp)
    ref_texts = [t for t in updated.texts if t.text_type == "reference"]
    assert ref_texts[0].text == "Q7"


def test_add_silkscreen_value_text_correct() -> None:
    """Added value label has the footprint's value as its text."""
    fp = _bare_footprint(value="4k7")
    updated = add_silkscreen_to_footprint(fp)
    val_texts = [t for t in updated.texts if t.text_type == "value"]
    assert val_texts[0].text == "4k7"


def test_add_silkscreen_idempotent() -> None:
    """Calling add_silkscreen_to_footprint twice does not duplicate labels."""
    fp = _bare_footprint()
    once = add_silkscreen_to_footprint(fp)
    twice = add_silkscreen_to_footprint(once)
    ref_count = sum(1 for t in twice.texts if t.text_type == "reference")
    val_count = sum(1 for t in twice.texts if t.text_type == "value")
    assert ref_count == 1
    assert val_count == 1
