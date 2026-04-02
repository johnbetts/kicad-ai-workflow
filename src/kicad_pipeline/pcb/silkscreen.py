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
    LAYER_F_FAB,
    LAYER_F_SILKSCREEN,
    PCB_SILKSCREEN_LINE_WIDTH_MM,
)
from kicad_pipeline.models.pcb import Footprint, FootprintLine, FootprintText, Point

# ---------------------------------------------------------------------------
# Silk collision resolution and board-edge clamping
# ---------------------------------------------------------------------------


def resolve_silk_collisions(
    footprints: list[Footprint],
) -> list[Footprint]:
    """Push silk ref labels that overlap other footprints' copper.

    For each reference label, compute its board-space bounding box and
    check for overlap with pads on OTHER footprints.  If overlap is
    detected, flip the label to the opposite side of the component
    (negate Y offset).  Also resolves silk-on-silk overlap by nudging
    the second label sideways.
    """
    import math as _m

    # Build pad lookup: list of (abs_x, abs_y, half_w, half_h, owner_ref)
    all_pads: list[tuple[float, float, float, float, str]] = []
    for fp in footprints:
        rot_r = _m.radians(fp.rotation)
        cos_r = _m.cos(rot_r)
        sin_r = _m.sin(rot_r)
        for pad in fp.pads:
            rpx = pad.position.x * cos_r - pad.position.y * sin_r
            rpy = pad.position.x * sin_r + pad.position.y * cos_r
            px = fp.position.x + rpx
            py = fp.position.y + rpy
            hw = max(pad.size_x, pad.size_y) / 2.0
            all_pads.append((px, py, hw, hw, fp.ref))

    # Collect all ref label bboxes for silk-overlap detection
    label_bboxes: list[tuple[float, float, float, float, int]] = []
    # (cx, cy, half_w, half_h, fp_index)

    result: list[Footprint] = []
    for idx, fp in enumerate(footprints):
        ref_text = None
        ref_text_idx = -1
        for ti, t in enumerate(fp.texts):
            if t.text_type == "reference" and not t.hidden:
                ref_text = t
                ref_text_idx = ti
                break
        if ref_text is None:
            result.append(fp)
            continue

        # Compute label board-space bbox
        rot_r = _m.radians(fp.rotation)
        cos_r = _m.cos(rot_r)
        sin_r = _m.sin(rot_r)
        # Label position in board space (rotated with footprint)
        lx = ref_text.position.x * cos_r - ref_text.position.y * sin_r
        ly = ref_text.position.x * sin_r + ref_text.position.y * cos_r
        abs_lx = fp.position.x + lx
        abs_ly = fp.position.y + ly
        half_w = 0.65 * ref_text.effects_size * max(2, len(ref_text.text)) / 2.0
        half_h = ref_text.effects_size * 0.75 / 2.0

        # Check overlap with other footprints' pads
        overlaps_pad = False
        for px, py, phw, phh, owner in all_pads:
            if owner == fp.ref:
                continue
            if (abs_lx + half_w > px - phw - 0.1
                    and abs_lx - half_w < px + phw + 0.1
                    and abs_ly + half_h > py - phh - 0.1
                    and abs_ly - half_h < py + phh + 0.1):
                overlaps_pad = True
                break

        new_fp = fp
        if overlaps_pad:
            # Flip label to opposite side (negate Y in footprint-local)
            new_y = -ref_text.position.y
            new_texts = list(fp.texts)
            new_texts[ref_text_idx] = FootprintText(
                text_type=ref_text.text_type,
                text=ref_text.text,
                position=Point(x=ref_text.position.x, y=new_y),
                layer=ref_text.layer,
                effects_size=ref_text.effects_size,
                hidden=ref_text.hidden,
            )
            new_fp = Footprint(
                lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                position=fp.position, rotation=fp.rotation, layer=fp.layer,
                pads=fp.pads, graphics=fp.graphics, texts=tuple(new_texts),
                lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
                models=fp.models, datasheet=fp.datasheet,
                description=fp.description, fp_zones=fp.fp_zones,
            )
            # Recompute label position after flip
            new_ly = ref_text.position.x * sin_r + new_y * cos_r
            abs_ly = fp.position.y + new_ly

        label_bboxes.append((abs_lx, abs_ly, half_w, half_h, idx))
        result.append(new_fp)

    return result


def clamp_silk_to_board(
    fp: Footprint,
    origin_x: float,
    origin_y: float,
    board_w: float,
    board_h: float,
    margin: float = 0.3,
) -> Footprint:
    """Move silkscreen texts that extend beyond the board edge inward.

    Silk items whose absolute position falls outside the board rectangle
    (with *margin*) are shifted so the text stays fully on-board.  Only
    ``reference`` and ``value`` texts are adjusted -- user texts are left
    alone.
    """
    changed = False
    new_texts: list[FootprintText] = []
    for t in fp.texts:
        if t.text_type not in ("reference", "value"):
            new_texts.append(t)
            continue
        half_h = t.effects_size / 2.0
        # Estimate text width: ~0.65 * size per character
        half_w = 0.65 * t.effects_size * len(t.text) / 2.0
        abs_y = fp.position.y + t.position.y
        abs_x = fp.position.x + t.position.x
        new_y = t.position.y
        new_x = t.position.x
        # Clamp Y
        if abs_y - half_h < origin_y + margin:
            new_y = (origin_y + margin + half_h) - fp.position.y
            changed = True
        elif abs_y + half_h > origin_y + board_h - margin:
            new_y = (origin_y + board_h - margin - half_h) - fp.position.y
            changed = True
        # Clamp X
        if abs_x - half_w < origin_x + margin:
            new_x = (origin_x + margin + half_w) - fp.position.x
            changed = True
        elif abs_x + half_w > origin_x + board_w - margin:
            new_x = (origin_x + board_w - margin - half_w) - fp.position.x
            changed = True
        new_texts.append(
            FootprintText(
                text_type=t.text_type,
                text=t.text,
                position=Point(x=new_x, y=new_y),
                layer=t.layer,
                effects_size=t.effects_size,
                hidden=t.hidden,
            ) if (new_x != t.position.x or new_y != t.position.y) else t
        )
    if not changed:
        return fp
    return Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
        position=fp.position, rotation=fp.rotation, layer=fp.layer,
        pads=fp.pads, graphics=fp.graphics, texts=tuple(new_texts),
        lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
        models=fp.models, datasheet=fp.datasheet,
        description=fp.description,
    )

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


def _silk_pad_extents(fp: Footprint) -> tuple[float, float]:
    if fp.pads:
        return (
            min(p.position.y - p.size_y / 2 for p in fp.pads),
            max(p.position.y + p.size_y / 2 for p in fp.pads),
        )
    return -_LABEL_OFFSET_MM, _LABEL_OFFSET_MM


def _silk_text_size(pad_span_y: float) -> float:
    if pad_span_y < 2.0:
        return 0.6
    if pad_span_y < 4.0:
        return 0.8
    return _REF_TEXT_SIZE_MM


def _silk_add_ref(
    fp: Footprint,
    new_texts: list[FootprintText],
    ref_y: float,
    text_size: float,
    has_tht: bool,
    pad_span_y: float,
) -> None:
    ref_pos = Point(x=0.0, y=ref_y)
    is_compact_smd = not has_tht and pad_span_y < 2.0
    is_mounting_hole = fp.ref.startswith("H")
    ref_layer = LAYER_F_FAB if (is_compact_smd or is_mounting_hole) else LAYER_F_SILKSCREEN
    new_texts.append(
        make_ref_label(ref=fp.ref, position=ref_pos, layer=ref_layer, size_mm=text_size)
    )
    log.debug("add_silkscreen_to_footprint: added ref label to %s", fp.ref)


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

    min_y, max_y = _silk_pad_extents(fp)
    pad_span_y = max_y - min_y
    text_size = _silk_text_size(pad_span_y)

    has_tht = any(p.pad_type == "thru_hole" for p in fp.pads)
    pad_label_gap = 1.5 if has_tht else (0.75 * text_size / 2 + 0.7)
    ref_y = min(min_y - pad_label_gap, -_LABEL_OFFSET_MM)
    val_y = max(max_y + pad_label_gap, _LABEL_OFFSET_MM)

    if "reference" not in existing_types:
        _silk_add_ref(fp, new_texts, ref_y, text_size, has_tht, pad_span_y)

    if "value" not in existing_types:
        new_texts.append(
            make_value_label(
                value=fp.value,
                position=Point(x=0.0, y=val_y),
                layer=LAYER_F_SILKSCREEN,
                size_mm=min(text_size, _VALUE_TEXT_SIZE_MM),
                hidden=True,
            )
        )
        log.debug("add_silkscreen_to_footprint: added value label to %s", fp.ref)

    if len(new_texts) == len(fp.texts):
        return fp

    return Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value, position=fp.position,
        rotation=fp.rotation, layer=fp.layer, pads=fp.pads, graphics=fp.graphics,
        texts=tuple(new_texts), lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
        models=fp.models, datasheet=fp.datasheet, description=fp.description,
        fp_zones=fp.fp_zones, custom_properties=fp.custom_properties,
    )
