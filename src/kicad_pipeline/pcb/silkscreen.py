"""Silkscreen label and marking generation for PCB footprints.

Provides helpers that create :class:`~kicad_pipeline.models.pcb.FootprintText`
and :class:`~kicad_pipeline.models.pcb.FootprintLine` objects representing the
human-readable markings printed on the silkscreen layer of a PCB.

All size and position values are in millimetres.
"""

from __future__ import annotations

import datetime
import logging

from kicad_pipeline.constants import (
    LAYER_F_SILKSCREEN,
    PCB_SILKSCREEN_LINE_WIDTH_MM,
)
from kicad_pipeline.models.pcb import Footprint, FootprintLine, FootprintText, Point

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default text sizes
# ---------------------------------------------------------------------------

_REF_TEXT_SIZE_MM: float = 1.0
"""Default silkscreen reference-designator text height in mm."""

_VALUE_TEXT_SIZE_MM: float = 0.8
"""Default silkscreen value text height in mm."""

_ZONE_LABEL_SIZE_MM: float = 1.2
"""Default functional-zone label text height in mm."""

_BOARD_TITLE_SIZE_MM: float = 1.5
"""Board title text height in mm."""

_BOARD_REV_SIZE_MM: float = 1.0
"""Board revision / date text height in mm."""

_PIN1_LINE_LENGTH_MM: float = 1.0
"""Length of the pin-1 indicator line in mm."""

# Vertical offset between ref and value labels relative to the pad centre
_LABEL_OFFSET_MM: float = 1.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_ref_label(
    ref: str,
    position: Point,
    layer: str = LAYER_F_SILKSCREEN,
    size_mm: float = _REF_TEXT_SIZE_MM,
) -> FootprintText:
    """Generate a reference-designator label text item.

    Args:
        ref: Reference designator string, e.g. ``"R1"``.
        position: Position relative to the footprint origin (mm).
        layer: Target silkscreen layer (default ``"F.Silkscreen"``).
        size_mm: Text height in mm (default ``1.0``).

    Returns:
        A :class:`FootprintText` of type ``"reference"``.
    """
    return FootprintText(
        text_type="reference",
        text=ref,
        position=position,
        layer=layer,
        effects_size=size_mm,
        hidden=False,
    )


def make_value_label(
    value: str,
    position: Point,
    layer: str = LAYER_F_SILKSCREEN,
    size_mm: float = _VALUE_TEXT_SIZE_MM,
    hidden: bool = True,
) -> FootprintText:
    """Generate a component value label (hidden by default for clean silkscreen).

    The value label is hidden by default so that it does not clutter the
    printed board but remains accessible in KiCad's footprint editor.

    Args:
        value: Component value string, e.g. ``"10k"``, ``"100nF"``.
        position: Position relative to the footprint origin (mm).
        layer: Target silkscreen layer (default ``"F.Silkscreen"``).
        size_mm: Text height in mm (default ``0.8``).
        hidden: Whether to hide the label (default ``True``).

    Returns:
        A :class:`FootprintText` of type ``"value"``.
    """
    return FootprintText(
        text_type="value",
        text=value,
        position=position,
        layer=layer,
        effects_size=size_mm,
        hidden=hidden,
    )


def make_zone_label(
    text: str,
    x: float,
    y: float,
    layer: str = LAYER_F_SILKSCREEN,
) -> FootprintText:
    """Generate a functional-zone label (e.g. POWER, ETHERNET, ANALOG IN).

    Zone labels are rendered at a slightly larger size than component
    references so they are readable on the final board.

    Args:
        text: Label text, e.g. ``"POWER"``, ``"ETHERNET"``, ``"ANALOG IN"``.
        x: X coordinate in board space (mm).
        y: Y coordinate in board space (mm).
        layer: Target silkscreen layer (default ``"F.Silkscreen"``).

    Returns:
        A :class:`FootprintText` of type ``"user"``.
    """
    return FootprintText(
        text_type="user",
        text=text,
        position=Point(x=x, y=y),
        layer=layer,
        effects_size=_ZONE_LABEL_SIZE_MM,
        hidden=False,
    )


def make_board_title(
    project_name: str,
    revision: str,
    x: float,
    y: float,
) -> list[FootprintText]:
    """Generate a board title block with name, revision, and date.

    The title block consists of three stacked text items:

    * Line 0 — project name (largest)
    * Line 1 — revision string prefixed with ``"Rev: "``
    * Line 2 — ISO-8601 date of generation prefixed with ``"Date: "``

    Args:
        project_name: Human-readable project name.
        revision: Revision string, e.g. ``"v0.1"``.
        x: X coordinate of the top-left anchor in board space (mm).
        y: Y coordinate of the top-left anchor in board space (mm).

    Returns:
        List of three :class:`FootprintText` objects representing the title block.
    """
    today = datetime.date.today().isoformat()
    line_spacing = _BOARD_REV_SIZE_MM + 0.5

    items: list[FootprintText] = [
        FootprintText(
            text_type="user",
            text=project_name,
            position=Point(x=x, y=y),
            layer=LAYER_F_SILKSCREEN,
            effects_size=_BOARD_TITLE_SIZE_MM,
            hidden=False,
        ),
        FootprintText(
            text_type="user",
            text=f"Rev: {revision}",
            position=Point(x=x, y=y + _BOARD_TITLE_SIZE_MM + line_spacing),
            layer=LAYER_F_SILKSCREEN,
            effects_size=_BOARD_REV_SIZE_MM,
            hidden=False,
        ),
        FootprintText(
            text_type="user",
            text=f"Date: {today}",
            position=Point(x=x, y=y + _BOARD_TITLE_SIZE_MM + line_spacing * 2.0),
            layer=LAYER_F_SILKSCREEN,
            effects_size=_BOARD_REV_SIZE_MM,
            hidden=False,
        ),
    ]
    return items


def make_dip_warning_label(
    x: float,
    y: float,
    layer: str = LAYER_F_SILKSCREEN,
) -> FootprintText:
    """Generate a silkscreen warning label for a DIP switch.

    Warns the user that only one switch should be active at a time to
    prevent short circuits.

    Args:
        x: X coordinate in board space (mm).
        y: Y coordinate in board space (mm).
        layer: Target silkscreen layer (default F.SilkS).

    Returns:
        A :class:`FootprintText` with the warning message.
    """
    return FootprintText(
        text_type="user",
        text="ONE SWITCH AT A TIME",
        position=Point(x=x, y=y),
        layer=layer,
        effects_size=0.8,
        hidden=False,
    )


def make_pin1_indicator(
    x: float,
    y: float,
    layer: str = LAYER_F_SILKSCREEN,
) -> FootprintLine:
    """Generate a pin-1 indicator as a short horizontal line marker.

    The line is drawn to the left of *(x, y)* using the standard silkscreen
    line width.

    Args:
        x: X coordinate of the indicator anchor in footprint space (mm).
        y: Y coordinate of the indicator anchor in footprint space (mm).
        layer: Target silkscreen layer (default ``"F.Silkscreen"``).

    Returns:
        A :class:`FootprintLine` representing the pin-1 indicator.
    """
    return FootprintLine(
        start=Point(x=x, y=y),
        end=Point(x=x + _PIN1_LINE_LENGTH_MM, y=y),
        layer=layer,
        width=PCB_SILKSCREEN_LINE_WIDTH_MM,
    )


def add_silkscreen_to_footprint(fp: Footprint) -> Footprint:
    """Ensure reference and value labels are present on the footprint.

    If ``fp.texts`` already contains both a ``"reference"`` and a ``"value"``
    entry, the footprint is returned unchanged.  Otherwise the missing labels
    are synthesised:

    * Reference label placed 1.5 mm above the footprint origin.
    * Value label placed 1.5 mm below the footprint origin (hidden).

    Because :class:`Footprint` is immutable, a new instance is returned with
    the ``texts`` field augmented.

    Args:
        fp: Source footprint (may have an empty or partial ``texts`` tuple).

    Returns:
        A new :class:`Footprint` with at minimum ``"reference"`` and
        ``"value"`` text items present.
    """
    existing_types = {t.text_type for t in fp.texts}
    new_texts: list[FootprintText] = list(fp.texts)

    if "reference" not in existing_types:
        ref_pos = Point(x=0.0, y=-_LABEL_OFFSET_MM)
        new_texts.append(
            make_ref_label(ref=fp.ref, position=ref_pos, layer=LAYER_F_SILKSCREEN)
        )
        log.debug("add_silkscreen_to_footprint: added ref label to %s", fp.ref)

    if "value" not in existing_types:
        val_pos = Point(x=0.0, y=_LABEL_OFFSET_MM)
        new_texts.append(
            make_value_label(
                value=fp.value,
                position=val_pos,
                layer=LAYER_F_SILKSCREEN,
                hidden=True,
            )
        )
        log.debug("add_silkscreen_to_footprint: added value label to %s", fp.ref)

    # Return same object if nothing changed (optimisation)
    if len(new_texts) == len(fp.texts):
        return fp

    return Footprint(
        lib_id=fp.lib_id,
        ref=fp.ref,
        value=fp.value,
        position=fp.position,
        rotation=fp.rotation,
        layer=fp.layer,
        pads=fp.pads,
        graphics=fp.graphics,
        texts=tuple(new_texts),
        lcsc=fp.lcsc,
        uuid=fp.uuid,
        attr=fp.attr,
    )
