"""Parametric footprint generators for the kicad-ai-pipeline.

Generates :class:`~kicad_pipeline.models.pcb.Footprint` objects for common
passive, semiconductor, and connector packages without any external library
dependency.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from kicad_pipeline.constants import (
    LAYER_B_CU,
    LAYER_B_MASK,
    LAYER_B_PASTE,
    LAYER_F_COURTYARD,
    LAYER_F_CU,
    LAYER_F_FAB,
    LAYER_F_MASK,
    LAYER_F_PASTE,
    LAYER_F_SILKSCREEN,
    PCB_COURTYARD_CLEARANCE_MM,
    PCB_SILKSCREEN_LINE_WIDTH_MM,
)
from kicad_pipeline.exceptions import ConfigurationError, PCBError
from kicad_pipeline.models.pcb import (
    Footprint,
    FootprintLine,
    FootprintText,
    Pad,
    Point,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard KiCad library IDs
# ---------------------------------------------------------------------------

_KICAD_RESISTOR_LIB_IDS: dict[str, str] = {
    "0402": "Resistor_SMD:R_0402_1005Metric",
    "0603": "Resistor_SMD:R_0603_1608Metric",
    "0805": "Resistor_SMD:R_0805_2012Metric",
    "1206": "Resistor_SMD:R_1206_3216Metric",
    "1210": "Resistor_SMD:R_1210_3225Metric",
}

_KICAD_CAPACITOR_LIB_IDS: dict[str, str] = {
    "0402": "Capacitor_SMD:C_0402_1005Metric",
    "0603": "Capacitor_SMD:C_0603_1608Metric",
    "0805": "Capacitor_SMD:C_0805_2012Metric",
    "1206": "Capacitor_SMD:C_1206_3216Metric",
    "1210": "Capacitor_SMD:C_1210_3225Metric",
}

_KICAD_LED_LIB_IDS: dict[str, str] = {
    "0402": "LED_SMD:LED_0402_1005Metric",
    "0603": "LED_SMD:LED_0603_1608Metric",
    "0805": "LED_SMD:LED_0805_2012Metric",
    "1206": "LED_SMD:LED_1206_3216Metric",
    "1210": "LED_SMD:LED_1210_3225Metric",
}

# ---------------------------------------------------------------------------
# Data-file path
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data"
ROTATION_OFFSETS_FILE: Path = _DATA_DIR / "rotation_offsets.json"

# ---------------------------------------------------------------------------
# Package dimension tables
# ---------------------------------------------------------------------------

# SMD resistor / capacitor packages: (pad_w, pad_h, pitch, body_w, body_h)
_SMD_RC_DIMS: dict[str, tuple[float, float, float, float, float]] = {
    "0402": (0.5, 0.5, 1.0, 1.0, 0.5),
    "0603": (0.8, 0.8, 1.6, 1.6, 0.8),
    "0805": (1.2, 1.4, 2.0, 2.0, 1.25),
    "1206": (1.5, 1.7, 3.2, 3.2, 1.6),
    "1210": (1.5, 2.5, 3.2, 3.2, 2.5),
}

# SOT-23 variants: (pad_w, pad_h, row_pitch, col_pitch, pin_count, pin_coords)
# pin_coords: list of (x, y) relative to footprint origin
_SOT23_VARIANTS: dict[str, tuple[float, float, list[tuple[float, float]]]] = {
    "SOT-23": (
        0.9,
        1.3,
        [(-0.95, 1.0), (0.95, 1.0), (0.0, -1.0)],
    ),
    "SOT-23-5": (
        0.6,
        1.0,
        [
            (-1.5, -0.95),
            (-1.5, 0.0),
            (-1.5, 0.95),
            (1.5, 0.475),
            (1.5, -0.475),
        ],
    ),
    "SOT-23-6": (
        0.6,
        1.0,
        [
            (-1.5, -0.95),
            (-1.5, 0.0),
            (-1.5, 0.95),
            (1.5, 0.95),
            (1.5, 0.0),
            (1.5, -0.95),
        ],
    ),
}

# USB-C power/signal pad definitions: (x, y, width, height, name)
_USBC_PADS: list[tuple[float, float, float, float, str]] = [
    (-3.5, 2.5, 1.6, 1.6, "VBUS"),   # VBUS left
    (3.5, 2.5, 1.6, 1.6, "VBUS"),    # VBUS right
    (-3.5, -2.5, 1.6, 1.6, "GND"),   # GND left
    (3.5, -2.5, 1.6, 1.6, "GND"),    # GND right
    (-2.0, 2.5, 0.6, 1.6, "CC1"),    # CC1
    (2.0, 2.5, 0.6, 1.6, "CC2"),     # CC2
    (-1.0, 2.5, 0.6, 1.6, "D-"),     # D-
    (1.0, 2.5, 0.6, 1.6, "D+"),      # D+
]

# RJ45 pin row Y positions (signal pins)
_RJ45_SIGNAL_PITCH_MM: float = 1.27
_RJ45_SIGNAL_COUNT: int = 8
_RJ45_DRILL_MM: float = 1.2
_RJ45_PAD_DIAM_MM: float = 2.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _smd_pad(
    number: str,
    x: float,
    y: float,
    size_x: float,
    size_y: float,
    layer: str,
) -> Pad:
    """Build a single SMD pad on the appropriate copper layers."""
    if layer == LAYER_F_CU:
        layers: tuple[str, ...] = (LAYER_F_CU, LAYER_F_PASTE, LAYER_F_MASK)
    else:
        layers = (LAYER_B_CU, LAYER_B_PASTE, LAYER_B_MASK)
    return Pad(
        number=number,
        pad_type="smd",
        shape="rect",
        position=Point(x, y),
        size_x=size_x,
        size_y=size_y,
        layers=layers,
    )


def _thru_pad(
    number: str,
    x: float,
    y: float,
    diameter: float,
    drill: float,
    shape: str = "circle",
) -> Pad:
    """Build a through-hole pad present on both copper layers."""
    return Pad(
        number=number,
        pad_type="thru_hole",
        shape=shape,
        position=Point(x, y),
        size_x=diameter,
        size_y=diameter,
        layers=(LAYER_F_CU, LAYER_B_CU, LAYER_F_MASK, LAYER_B_MASK),
        drill_diameter=drill,
    )


def _ref_text(ref: str, y_offset: float, layer: str) -> FootprintText:
    """Build a reference designator text item."""
    return FootprintText(
        text_type="reference",
        text=ref,
        position=Point(0.0, y_offset),
        layer=layer,
        effects_size=1.0,
    )


def _val_text(value: str, y_offset: float, layer: str) -> FootprintText:
    """Build a value text item."""
    return FootprintText(
        text_type="value",
        text=value,
        position=Point(0.0, y_offset),
        layer=layer,
        effects_size=1.0,
    )


def _courtyard_rect(
    body_w: float, body_h: float, clearance: float = PCB_COURTYARD_CLEARANCE_MM
) -> tuple[FootprintLine, ...]:
    """Build a rectangular courtyard from body dimensions + clearance."""
    hw = body_w / 2.0 + clearance
    hh = body_h / 2.0 + clearance
    layer = LAYER_F_COURTYARD
    w = PCB_SILKSCREEN_LINE_WIDTH_MM
    return (
        FootprintLine(start=Point(-hw, -hh), end=Point(hw, -hh), layer=layer, width=w),
        FootprintLine(start=Point(hw, -hh), end=Point(hw, hh), layer=layer, width=w),
        FootprintLine(start=Point(hw, hh), end=Point(-hw, hh), layer=layer, width=w),
        FootprintLine(start=Point(-hw, hh), end=Point(-hw, -hh), layer=layer, width=w),
    )


def _silk_side_marks(
    body_w: float,
    body_h: float,
    pad_edge_x: float | None = None,
) -> tuple[FootprintLine, ...]:
    """Short silkscreen lines on left/right edges of component body.

    When *pad_edge_x* is given, silkscreen lines are pushed outward
    to avoid overlapping with copper pads (silk_over_copper DRC).
    """
    hw = body_w / 2.0
    if pad_edge_x is not None:
        # Push silk marks outside pad edge + mask expansion + half silk width
        hw = max(hw, pad_edge_x + 0.35)
    hh = body_h / 2.0 * 0.45  # 45 % of half-height (avoid pad mask)
    layer = LAYER_F_SILKSCREEN
    w = PCB_SILKSCREEN_LINE_WIDTH_MM
    return (
        FootprintLine(start=Point(-hw, -hh), end=Point(-hw, hh), layer=layer, width=w),
        FootprintLine(start=Point(hw, -hh), end=Point(hw, hh), layer=layer, width=w),
    )


# ---------------------------------------------------------------------------
# Public footprint generators
# ---------------------------------------------------------------------------


def make_smd_resistor_capacitor(
    ref: str,
    value: str,
    package: str = "0805",
    layer: str = LAYER_F_CU,
) -> Footprint:
    """Generate SMD resistor/capacitor footprint.

    Package dimensions (mm):

    - 0402: pads 0.5x0.5, pitch 1.0, body 1.0x0.5
    - 0603: pads 0.8x0.8, pitch 1.6, body 1.6x0.8
    - 0805: pads 1.2x1.4, pitch 2.0, body 2.0x1.25
    - 1206: pads 1.5x1.7, pitch 3.2, body 3.2x1.6
    - 1210: pads 1.5x2.5, pitch 3.2, body 3.2x2.5

    Pads: "1" at (-pitch/2, 0), "2" at (+pitch/2, 0).
    Courtyard: body + PCB_COURTYARD_CLEARANCE_MM on all sides.
    Silkscreen: short lines on top/bottom edges of body.

    Args:
        ref: Reference designator (e.g. "R1", "C4").
        value: Component value string (e.g. "10k", "100nF").
        package: IPC package code from {"0402","0603","0805","1206","1210"}.
        layer: Primary copper layer, default F.Cu.

    Returns:
        Fully constructed :class:`Footprint`.

    Raises:
        PCBError: When *package* is not a recognised code.
    """
    if package not in _SMD_RC_DIMS:
        valid = ", ".join(sorted(_SMD_RC_DIMS))
        raise PCBError(f"Unknown SMD package '{package}'; valid options: {valid}")

    pad_w, pad_h, pitch, body_w, body_h = _SMD_RC_DIMS[package]
    _log.debug("make_smd_resistor_capacitor ref=%s pkg=%s", ref, package)

    pads = (
        _smd_pad("1", -pitch / 2.0, 0.0, pad_w, pad_h, layer),
        _smd_pad("2", pitch / 2.0, 0.0, pad_w, pad_h, layer),
    )
    # Pad edge for silkscreen clearance: pitch/2 + pad_w/2
    pad_edge_x = pitch / 2.0 + pad_w / 2.0
    # For compact packages (0603/0402) skip silk marks — they inevitably
    # overlap mask apertures after rotation in dense layouts.
    if body_h <= 1.0:
        graphics = _courtyard_rect(body_w, body_h)
    else:
        graphics = (
            *_courtyard_rect(body_w, body_h),
            *_silk_side_marks(body_w, body_h, pad_edge_x=pad_edge_x),
        )
    # Compact packages: ref on F.Fab to avoid silk-over-copper DRC
    ref_layer = LAYER_F_FAB if body_h <= 1.0 else LAYER_F_SILKSCREEN
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), ref_layer),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5, LAYER_F_FAB),
    )
    # Use standard KiCad lib_id: detect R vs C from ref prefix
    ref_prefix = "".join(ch for ch in ref if ch.isalpha()).upper()
    if ref_prefix == "C":
        lib_id = _KICAD_CAPACITOR_LIB_IDS.get(package, f"Capacitor_SMD:C_{package}")
    else:
        lib_id = _KICAD_RESISTOR_LIB_IDS.get(package, f"Resistor_SMD:R_{package}")

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=layer,
        pads=pads,
        graphics=graphics,
        texts=texts,
        attr="smd",
    )


def make_smd_led(
    ref: str,
    value: str,
    package: str = "0805",
    layer: str = LAYER_F_CU,
) -> Footprint:
    """LED footprint — same geometry as resistor, pin 1 is K (cathode), pin 2 is A (anode).

    A polarity triangle is drawn near pin 1 (cathode) on the silkscreen layer.

    Args:
        ref: Reference designator (e.g. "D1").
        value: Component value string (e.g. "RED", "WS2812B").
        package: IPC package code from {"0402","0603","0805","1206","1210"}.
        layer: Primary copper layer, default F.Cu.

    Returns:
        Fully constructed :class:`Footprint`.

    Raises:
        PCBError: When *package* is not a recognised code.
    """
    # Start with a resistor base then add polarity mark
    base = make_smd_resistor_capacitor(ref, value, package, layer)
    if package not in _SMD_RC_DIMS:  # already raised above, guard for type checker
        raise PCBError(f"Unknown SMD LED package '{package}'")

    _, pad_h, pitch, _body_w, _body_h = _SMD_RC_DIMS[package]
    _log.debug("make_smd_led ref=%s pkg=%s", ref, package)

    # Polarity triangle near cathode (pin 1, negative x)
    tri_x = -pitch / 2.0
    tri_size = pad_h * 0.4
    tri_lines = (
        FootprintLine(
            start=Point(tri_x - tri_size, -tri_size / 2.0),
            end=Point(tri_x + tri_size, 0.0),
            layer=LAYER_F_SILKSCREEN,
            width=PCB_SILKSCREEN_LINE_WIDTH_MM,
        ),
        FootprintLine(
            start=Point(tri_x + tri_size, 0.0),
            end=Point(tri_x - tri_size, tri_size / 2.0),
            layer=LAYER_F_SILKSCREEN,
            width=PCB_SILKSCREEN_LINE_WIDTH_MM,
        ),
        FootprintLine(
            start=Point(tri_x - tri_size, tri_size / 2.0),
            end=Point(tri_x - tri_size, -tri_size / 2.0),
            layer=LAYER_F_SILKSCREEN,
            width=PCB_SILKSCREEN_LINE_WIDTH_MM,
        ),
    )
    combined_graphics = base.graphics + tri_lines
    led_lib_id = _KICAD_LED_LIB_IDS.get(package, f"LED_SMD:LED_{package}")
    return Footprint(
        lib_id=led_lib_id,
        ref=base.ref,
        value=base.value,
        position=base.position,
        layer=base.layer,
        pads=base.pads,
        graphics=combined_graphics,
        texts=base.texts,
        attr=base.attr,
    )


def make_sot23(
    ref: str,
    value: str,
    variant: str = "SOT-23",
    layer: str = LAYER_F_CU,
) -> Footprint:
    """SOT-23 / SOT-23-5 / SOT-23-6 footprint.

    SOT-23 (3 pins): Pin 1 bottom-left, Pin 2 bottom-right, Pin 3 top-centre.
    SOT-23-5 (5 pins): 3 left column, 2 right column, pitch 0.95 mm.
    SOT-23-6 (6 pins): 3 left column, 3 right column, pitch 0.95 mm.

    Args:
        ref: Reference designator (e.g. "Q1", "U3").
        value: Component value string.
        variant: One of "SOT-23", "SOT-23-5", "SOT-23-6".
        layer: Primary copper layer, default F.Cu.

    Returns:
        Fully constructed :class:`Footprint`.

    Raises:
        PCBError: When *variant* is not recognised.
    """
    if variant not in _SOT23_VARIANTS:
        valid = ", ".join(sorted(_SOT23_VARIANTS))
        raise PCBError(f"Unknown SOT-23 variant '{variant}'; valid options: {valid}")

    pad_w, pad_h, coords = _SOT23_VARIANTS[variant]
    _log.debug("make_sot23 ref=%s variant=%s", ref, variant)

    pads = tuple(
        _smd_pad(str(i + 1), x, y, pad_w, pad_h, layer)
        for i, (x, y) in enumerate(coords)
    )
    # Approximate body bounding box for courtyard / silkscreen
    all_x = [c[0] for c in coords]
    all_y = [c[1] for c in coords]
    body_w = (max(all_x) - min(all_x)) + pad_w + 0.2
    body_h = (max(all_y) - min(all_y)) + pad_h + 0.2

    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5, LAYER_F_FAB),
    )
    return Footprint(
        lib_id=f"Package_TO_SOT_SMD:{variant}",
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=layer,
        pads=pads,
        graphics=graphics,
        texts=texts,
        attr="smd",
    )


def make_through_hole_2pin(
    ref: str,
    value: str,
    pitch_mm: float = 2.54,
    drill_mm: float = 0.8,
    pad_diameter_mm: float = 1.6,
) -> Footprint:
    """Generic 2-pin through-hole footprint (switches, crystals, connectors).

    Pin 1 at (-pitch/2, 0), Pin 2 at (+pitch/2, 0).
    Pads are round with copper on F.Cu, B.Cu, F.Mask, B.Mask.

    Args:
        ref: Reference designator.
        value: Component value string.
        pitch_mm: Centre-to-centre distance between pads in mm.
        drill_mm: Drill hole diameter in mm.
        pad_diameter_mm: Copper pad annular ring diameter in mm.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("make_through_hole_2pin ref=%s pitch=%.2f", ref, pitch_mm)
    pads = (
        _thru_pad("1", -pitch_mm / 2.0, 0.0, pad_diameter_mm, drill_mm),
        _thru_pad("2", pitch_mm / 2.0, 0.0, pad_diameter_mm, drill_mm),
    )
    body_w = pitch_mm + pad_diameter_mm
    body_h = pad_diameter_mm + 0.5
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(
            ref,
            -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5),
            LAYER_F_SILKSCREEN,
        ),
        _val_text(
            value,
            body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5,
            LAYER_F_FAB,
        ),
    )
    return Footprint(
        lib_id="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=LAYER_F_CU,
        pads=pads,
        graphics=graphics,
        texts=texts,
        attr="through_hole",
    )


def _parse_pin_count(footprint_id: str) -> int:
    """Extract pin count from a footprint ID string.

    Searches for patterns like ``_1x04_``, ``_2x20_``, ``x04``, ``-10_``,
    or bare trailing digits after a size separator.

    The NxM pattern must be preceded by ``_`` or start of string to avoid
    matching decimal dimensions like ``9.78x12.34mm``.

    Args:
        footprint_id: Footprint identifier string.

    Returns:
        Extracted pin count, or 2 as fallback.
    """
    import re

    # Match NxM pattern preceded by _ or start (e.g. _1x04_, _2x20_)
    # Negative lookahead rejects body dimensions like _3x3mm or _3x3.5mm
    m = re.search(r"(?:^|[_])(\d+)x(\d+)(?!\.?\d*mm)", footprint_id)
    if m:
        return int(m.group(1)) * int(m.group(2))
    # Match xNN pattern (e.g. SPSTx04 in DIP switches) → 2*N (switches have 2 pins each)
    m = re.search(r"x(\d{2,3})(?:[_]|$)", footprint_id)
    if m:
        n = int(m.group(1))
        if 2 <= n <= 100:
            return n * 2
    # Match -N or _N where N looks like a pin count (2-200), allowing end-of-string
    m = re.search(r"[-_](\d{1,3})(?:[-_]|$)", footprint_id)
    if m:
        n = int(m.group(1))
        if 2 <= n <= 200:
            return n
    return 2


def _parse_pitch(footprint_id: str) -> float:
    """Extract pitch from a footprint ID string (e.g. ``P2.54mm``).

    Args:
        footprint_id: Footprint identifier string.

    Returns:
        Pitch in mm, or 2.54 as fallback.
    """
    import re

    m = re.search(r"P(\d+\.?\d*)mm", footprint_id)
    if m:
        return float(m.group(1))
    return 2.54


def make_generic_smd_ic(
    ref: str,
    value: str,
    pin_count: int,
    pitch_mm: float = 0.5,
    lib_id: str = "",
) -> Footprint:
    """Generate a generic SMD IC footprint (MSOP, TSSOP, SOIC, QFP, QFN, etc.).

    Pins are arranged in two rows: odd pins on the left, even on the right.

    Args:
        ref: Reference designator.
        value: Component value string.
        pin_count: Total number of pins.
        pitch_mm: Pin pitch in mm.
        lib_id: KiCad library ID string (auto-generated if empty).

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("make_generic_smd_ic ref=%s pins=%d pitch=%.2f", ref, pin_count, pitch_mm)
    half = pin_count // 2
    pad_w = min(pitch_mm * 0.6, 0.5)
    pad_h = min(pitch_mm * 0.8, 1.5)
    row_span = (half - 1) * pitch_mm
    col_pitch = row_span / 2.0 + 1.5  # distance from center to pad column

    pads: list[Pad] = []
    for i in range(half):
        # Left column: pins 1..half going downward
        y = -row_span / 2.0 + i * pitch_mm
        pads.append(_smd_pad(str(i + 1), -col_pitch, y, pad_h, pad_w, LAYER_F_CU))
    for i in range(half):
        # Right column: pins half+1..pin_count going upward
        y = row_span / 2.0 - i * pitch_mm
        pads.append(_smd_pad(str(half + i + 1), col_pitch, y, pad_h, pad_w, LAYER_F_CU))

    body_w = col_pitch * 2.0 + pad_h
    body_h = row_span + pad_w + 0.5
    ic_pad_edge_x = col_pitch + pad_h / 2.0
    graphics: tuple[FootprintLine, ...] = (
        *_courtyard_rect(body_w, body_h),
        *_silk_side_marks(col_pitch * 1.6, body_h, pad_edge_x=ic_pad_edge_x),
    )
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5, LAYER_F_FAB),
    )
    if not lib_id:
        lib_id = f"Package_SO:SOIC-{pin_count}_P{pitch_mm:.2f}mm"

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=LAYER_F_CU,
        pads=tuple(pads),
        graphics=graphics,
        texts=texts,
        attr="smd",
    )


def make_pin_header_socket(
    ref: str,
    value: str,
    pin_count: int,
    pitch_mm: float = 2.54,
    rows: int = 1,
    lib_id: str = "",
    row_swap: bool = False,
) -> Footprint:
    """Generate a through-hole pin header or socket footprint.

    Args:
        ref: Reference designator.
        value: Component value string.
        pin_count: Total number of pins.
        pitch_mm: Pin pitch in mm.
        rows: Number of rows (1 or 2).
        lib_id: KiCad library ID string (auto-generated if empty).
        row_swap: If True, swap row direction (negate Y). Useful for
            RPi-style headers where pin numbering follows the opposite
            row convention.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("make_pin_header_socket ref=%s pins=%d rows=%d", ref, pin_count, rows)
    drill_mm = 1.0
    pad_diam = 1.7
    cols = pin_count // max(rows, 1)
    row_pitch = pitch_mm if rows > 1 else 0.0

    pads: list[Pad] = []
    pin_num = 1
    for col in range(cols):
        for row in range(rows):
            x = col * pitch_mm - (cols - 1) * pitch_mm / 2.0
            y = row * row_pitch - (rows - 1) * row_pitch / 2.0
            if row_swap:
                y = -y  # swap row direction
            pads.append(_thru_pad(str(pin_num), x, y, pad_diam, drill_mm))
            pin_num += 1

    body_w = (cols - 1) * pitch_mm + pad_diam + 1.5
    body_h = (rows - 1) * row_pitch + pad_diam + 1.5
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5, LAYER_F_FAB),
    )
    if not lib_id:
        lib_prefix = f"Connector_PinHeader_{pitch_mm:.2f}mm"
        pitch_str = f"P{pitch_mm:.2f}mm_Vertical"
        if rows > 1:
            lib_id = f"{lib_prefix}:PinHeader_{rows}x{cols:02d}_{pitch_str}"
        else:
            lib_id = f"{lib_prefix}:PinHeader_1x{cols:02d}_{pitch_str}"

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=LAYER_F_CU,
        pads=tuple(pads),
        graphics=graphics,
        texts=texts,
        attr="through_hole",
    )


def make_terminal_block(
    ref: str,
    value: str,
    pin_count: int = 2,
    pitch_mm: float = 5.08,
) -> Footprint:
    """Generate a through-hole terminal block footprint.

    Args:
        ref: Reference designator.
        value: Component value string.
        pin_count: Number of terminals.
        pitch_mm: Terminal pitch in mm.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("make_terminal_block ref=%s pins=%d pitch=%.2f", ref, pin_count, pitch_mm)
    drill_mm = 1.3
    pad_diam = 2.5

    pads = tuple(
        _thru_pad(
            str(i + 1),
            i * pitch_mm - (pin_count - 1) * pitch_mm / 2.0,
            0.0,
            pad_diam,
            drill_mm,
        )
        for i in range(pin_count)
    )
    body_w = (pin_count - 1) * pitch_mm + pad_diam + 4.0
    body_h = pad_diam + 6.0
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5, LAYER_F_FAB),
    )
    lib_id = f"Connector_TerminalBlock:TerminalBlock_{pin_count}P_P{pitch_mm:.2f}mm"

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=LAYER_F_CU,
        pads=tuple(pads),
        graphics=graphics,
        texts=texts,
        attr="through_hole",
    )


def make_dip_switch(
    ref: str,
    value: str,
    pin_count: int = 8,
    pitch_mm: float = 2.54,
) -> Footprint:
    """Generate a through-hole DIP switch footprint.

    Pins are arranged in two rows like a standard DIP package.

    Args:
        ref: Reference designator.
        value: Component value string.
        pin_count: Total number of pins (must be even).
        pitch_mm: Pin pitch in mm.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("make_dip_switch ref=%s pins=%d", ref, pin_count)
    drill_mm = 1.0
    pad_diam = 1.7
    half = pin_count // 2
    row_pitch = 7.62  # standard DIP row spacing

    pads: list[Pad] = []
    # Left column pins 1..half top-to-bottom
    for i in range(half):
        y = i * pitch_mm - (half - 1) * pitch_mm / 2.0
        pads.append(_thru_pad(str(i + 1), -row_pitch / 2.0, y, pad_diam, drill_mm))
    # Right column pins half+1..pin_count bottom-to-top
    for i in range(half):
        y = (half - 1 - i) * pitch_mm - (half - 1) * pitch_mm / 2.0
        pads.append(_thru_pad(str(half + i + 1), row_pitch / 2.0, y, pad_diam, drill_mm))

    body_w = row_pitch + pad_diam + 0.5
    body_h = (half - 1) * pitch_mm + pad_diam + 0.5
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5, LAYER_F_FAB),
    )
    lib_id = f"Button_Switch_DIP:SW_DIP_x{half:02d}"

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=LAYER_F_CU,
        pads=tuple(pads),
        graphics=graphics,
        texts=texts,
        attr="through_hole",
    )


def make_usbc_connector(ref: str, value: str = "USB-C") -> Footprint:
    """USB-C SMD connector footprint (generic, suitable for basic ordering).

    4 power pads (VBUS x2, GND x2) + 2 CC pads + 2 USB data pairs.
    Representationally correct; not production-validated against a specific PN.

    Args:
        ref: Reference designator (e.g. "J1").
        value: Component value string, default "USB-C".

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("make_usbc_connector ref=%s", ref)
    layer = LAYER_F_CU
    pads = tuple(
        _smd_pad(str(i + 1), x, y, w, h, layer)
        for i, (x, y, w, h, _name) in enumerate(_USBC_PADS)
    )
    body_w = 9.0
    body_h = 7.35
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5, LAYER_F_FAB),
    )
    return Footprint(
        lib_id="Connector_USB:USB_C_Receptacle_GCT_USB4105",
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=layer,
        pads=pads,
        graphics=graphics,
        texts=texts,
        attr="smd",
    )


def make_rj45(ref: str, value: str = "RJ45") -> Footprint:
    """RJ45 with integrated magnetics footprint (through-hole).

    8 signal pins in a row + 2 LED indicator pins + 4 mounting holes.
    Signal pin pitch is 1.27 mm.

    Args:
        ref: Reference designator (e.g. "J2").
        value: Component value string, default "RJ45".

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("make_rj45 ref=%s", ref)
    pads: list[Pad] = []
    start_x = -(_RJ45_SIGNAL_COUNT - 1) * _RJ45_SIGNAL_PITCH_MM / 2.0

    # 8 signal pins
    for i in range(_RJ45_SIGNAL_COUNT):
        x = start_x + i * _RJ45_SIGNAL_PITCH_MM
        pads.append(_thru_pad(str(i + 1), x, 0.0, _RJ45_PAD_DIAM_MM, _RJ45_DRILL_MM))

    # 2 LED pins offset to the right
    led_x_base = start_x + (_RJ45_SIGNAL_COUNT) * _RJ45_SIGNAL_PITCH_MM + 1.0
    pads.append(_thru_pad("9", led_x_base, 0.0, _RJ45_PAD_DIAM_MM, _RJ45_DRILL_MM))
    pads.append(
        _thru_pad("10", led_x_base + _RJ45_SIGNAL_PITCH_MM, 0.0, _RJ45_PAD_DIAM_MM, _RJ45_DRILL_MM)
    )

    # 4 mounting holes (np_thru_hole, no copper)
    mh_positions = [(-7.0, -4.5), (7.0, -4.5), (-7.0, 4.5), (7.0, 4.5)]
    for j, (mx, my) in enumerate(mh_positions):
        pads.append(
            Pad(
                number=f"MP{j + 1}",
                pad_type="np_thru_hole",
                shape="circle",
                position=Point(mx, my),
                size_x=3.2,
                size_y=3.2,
                layers=(LAYER_F_CU, LAYER_B_CU),
                drill_diameter=3.2,
            )
        )

    body_w = 16.0
    body_h = 13.5
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5, LAYER_F_FAB),
    )
    return Footprint(
        lib_id="Connector_RJ:RJ45_Amphenol_54602-x08_Horizontal",
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=LAYER_F_CU,
        pads=tuple(pads),
        graphics=graphics,
        texts=texts,
        attr="through_hole",
    )


def make_mounting_hole(
    ref: str,
    drill_diameter: float = 2.75,
) -> Footprint:
    """Generate an NPTH mounting hole footprint.

    Creates a non-plated through-hole with no copper annular ring,
    suitable for M2.5 screws (2.75mm drill) or similar mechanical
    mounting hardware.

    Args:
        ref: Reference designator (e.g. "H1").
        drill_diameter: Drill hole diameter in mm. Default 2.75mm
            (M2.5 clearance hole).

    Returns:
        Fully constructed :class:`Footprint` with ``exclude_from_pos_files``
        and ``exclude_from_bom`` attributes set.
    """
    _log.debug("make_mounting_hole ref=%s drill=%.2f", ref, drill_diameter)
    pad = Pad(
        number="",
        pad_type="np_thru_hole",
        shape="circle",
        position=Point(0.0, 0.0),
        size_x=drill_diameter,
        size_y=drill_diameter,
        layers=("*.Cu", "*.Mask"),
        drill_diameter=drill_diameter,
    )
    # Courtyard circle approximated as a rectangle
    crtyd_size = drill_diameter + 2 * PCB_COURTYARD_CLEARANCE_MM
    graphics = _courtyard_rect(crtyd_size, crtyd_size)
    texts = (
        _ref_text(ref, -(crtyd_size / 2.0 + 0.5), LAYER_F_SILKSCREEN),
        _val_text("MountingHole", crtyd_size / 2.0 + 0.5, LAYER_F_FAB),
    )
    return Footprint(
        lib_id="MountingHole:MountingHole_2.7mm_M2.5",
        ref=ref,
        value="MountingHole",
        position=Point(0.0, 0.0),
        layer=LAYER_F_CU,
        pads=(pad,),
        graphics=graphics,
        texts=texts,
        attr="exclude_from_pos_files exclude_from_bom",
    )


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------


def footprint_for_component(
    ref: str,
    value: str,
    footprint_id: str,
    lcsc: str | None = None,
) -> Footprint:
    """Route to the appropriate footprint generator based on *footprint_id*.

    Parsing rules (first match wins):

    - ``"R_<pkg>"`` or ``"C_<pkg>"``  → :func:`make_smd_resistor_capacitor`
    - ``"LED_<pkg>"``                  → :func:`make_smd_led`
    - ``"SOT-23*"``                    → :func:`make_sot23`
    - ``"SOT-223"``                    → :func:`make_sot23` (SOT-23 variant)
    - ``"USB-C*"`` / ``"USB_C*"``      → :func:`make_usbc_connector`
    - ``"RJ45*"``                      → :func:`make_rj45`
    - ``"PinHeader*"`` / ``"PinSocket*"`` → :func:`make_pin_header_socket`
    - ``"TerminalBlock*"``             → :func:`make_terminal_block`
    - ``"SW_DIP*"``                    → :func:`make_dip_switch`
    - ``"MSOP*"`` / ``"TSSOP*"`` / ``"SOIC*"`` / ``"QFP*"`` / ``"QFN*"`` / ``"DIP*"`` / ``"SOP*"``
      → :func:`make_generic_smd_ic`
    - Otherwise → :func:`make_smd_resistor_capacitor` with package ``"0805"`` (fallback)

    Args:
        ref: Reference designator.
        value: Component value string.
        footprint_id: Footprint identifier string from the requirements.
        lcsc: Optional LCSC part number (stored on the returned footprint).

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("footprint_for_component ref=%s id=%s", ref, footprint_id)
    fid = footprint_id.strip()
    upper = fid.upper()

    fp: Footprint

    # LED packages
    if upper.startswith("LED_"):
        pkg = fid[4:].upper()
        pkg_norm = pkg if pkg in _SMD_RC_DIMS else "0805"
        fp = make_smd_led(ref, value, package=pkg_norm)

    # Resistor / capacitor (R_pkg or C_pkg)
    elif upper.startswith(("R_", "C_")):
        pkg = fid[2:].upper()
        pkg_norm = pkg if pkg in _SMD_RC_DIMS else "0805"
        fp = make_smd_resistor_capacitor(ref, value, package=pkg_norm)

    # SOT-223 (special case before SOT-23 prefix check)
    elif upper == "SOT-223":
        fp = make_sot23(ref, value, variant="SOT-23")

    # SOT-23 family
    elif upper.startswith("SOT-23"):
        variant = fid.upper()
        if variant not in _SOT23_VARIANTS:
            variant = "SOT-23"
        fp = make_sot23(ref, value, variant=variant)

    # USB-C
    elif upper.startswith(("USB-C", "USB_C")):
        fp = make_usbc_connector(ref, value)

    # RJ45
    elif upper.startswith("RJ45"):
        fp = make_rj45(ref, value)

    # Pin headers and sockets
    elif upper.startswith(("PINHEADER", "PINSOCKET")):
        pin_count = _parse_pin_count(fid)
        pitch = _parse_pitch(fid)
        # Detect dual-row from "2x" in the footprint ID
        rows = 2 if "2X" in upper or "2x" in fid else 1
        # Detect RPi-related footprint IDs and pass row_swap=True
        rpi_swap = "2X20" in upper or "RPI" in upper or "RASPBERRY" in upper
        fp = make_pin_header_socket(
            ref, value, pin_count, pitch, rows, lib_id=fid, row_swap=rpi_swap,
        )

    # Terminal blocks
    elif upper.startswith("TERMINALBLOCK") or upper.startswith("TB_"):
        pin_count = _parse_pin_count(fid)
        pitch = _parse_pitch(fid)
        fp = make_terminal_block(ref, value, pin_count, pitch)

    # DIP switches
    elif upper.startswith("SW_DIP"):
        pin_count = _parse_pin_count(fid)
        if pin_count < 4:
            pin_count = 8  # sensible default
        fp = make_dip_switch(ref, value, pin_count)

    # Generic SMD IC packages (MSOP, TSSOP, SOIC, QFP, QFN, SOP, DFN, etc.)
    elif any(
        upper.startswith(prefix)
        for prefix in ("MSOP", "TSSOP", "SOIC", "QFP", "QFN", "SOP", "DFN", "SSOP", "LQFP")
    ):
        pin_count = _parse_pin_count(fid)
        if pin_count < 2:
            pin_count = 8
        pitch = _parse_pitch(fid)
        if pitch > 2.0:
            pitch = 0.5  # IC packages have fine pitch
        fp = make_generic_smd_ic(ref, value, pin_count, pitch, lib_id=fid)

    # Fallback
    else:
        _log.warning(
            "footprint_for_component: unknown footprint_id '%s' for ref %s; using 0805 fallback",
            footprint_id,
            ref,
        )
        fp = make_smd_resistor_capacitor(ref, value, package="0805")

    # Attach LCSC if provided
    if lcsc is not None:
        fp = Footprint(
            lib_id=fp.lib_id,
            ref=fp.ref,
            value=fp.value,
            position=fp.position,
            rotation=fp.rotation,
            layer=fp.layer,
            pads=fp.pads,
            graphics=fp.graphics,
            texts=fp.texts,
            lcsc=lcsc,
            uuid=fp.uuid,
            attr=fp.attr,
        )
    return fp


# ---------------------------------------------------------------------------
# Rotation offset helpers
# ---------------------------------------------------------------------------


def load_rotation_offsets(file: Path = ROTATION_OFFSETS_FILE) -> dict[str, float]:
    """Load JLCPCB rotation correction offsets from a JSON data file.

    The JSON file must contain a top-level ``"offsets"`` mapping from package
    identifier strings to numeric degree values.

    Args:
        file: Path to the JSON file.  Defaults to the bundled
              ``data/rotation_offsets.json``.

    Returns:
        Mapping from package identifier to correction offset in degrees.

    Raises:
        ConfigurationError: When the file is missing or malformed.
    """
    try:
        text = file.read_text(encoding="utf-8")
        data: object = json.loads(text)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Rotation-offsets file not found: {file}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Rotation-offsets file is not valid JSON: {file}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigurationError(f"Rotation-offsets JSON must be an object: {file}")

    offsets_raw = data.get("offsets", {})
    if not isinstance(offsets_raw, dict):
        raise ConfigurationError(
            f"'offsets' key in rotation-offsets JSON must be an object: {file}"
        )

    offsets: dict[str, float] = {}
    for key, val in offsets_raw.items():
        if not isinstance(key, str):
            raise ConfigurationError(
                f"Rotation-offsets key must be a string, got {type(key)}: {file}"
            )
        if not isinstance(val, int | float):
            raise ConfigurationError(
                f"Rotation-offsets value for '{key}' must be numeric: {file}"
            )
        offsets[key] = float(val)
    return offsets


def apply_rotation_offset(
    footprint_id: str,
    kicad_rotation: float,
    offsets: dict[str, float],
) -> float:
    """Apply a JLCPCB rotation correction offset to a KiCad rotation angle.

    Looks up *footprint_id* (and common prefix variants) in *offsets* and
    adds the correction.  If no entry is found the original *kicad_rotation*
    is returned unchanged.

    Args:
        footprint_id: Footprint identifier (e.g. ``"SOT-23"``, ``"0805"``).
        kicad_rotation: KiCad rotation angle in degrees (0-360).
        offsets: Mapping returned by :func:`load_rotation_offsets`.

    Returns:
        Corrected rotation angle (modulo 360).
    """
    # Direct lookup
    correction = offsets.get(footprint_id)

    # Try common prefix strips (e.g. "Package_TO_SOT_SMD:SOT-23" → "SOT-23")
    if correction is None:
        for key in offsets:
            if footprint_id.endswith(key) or footprint_id.upper() == key.upper():
                correction = offsets[key]
                break

    if correction is None:
        _log.debug(
            "apply_rotation_offset: no entry for '%s', returning %.1f unchanged",
            footprint_id,
            kicad_rotation,
        )
        return kicad_rotation

    result = (kicad_rotation + correction) % 360.0
    _log.debug(
        "apply_rotation_offset: '%s' %.1f + %.1f = %.1f",
        footprint_id,
        kicad_rotation,
        correction,
        result,
    )
    return result


# ---------------------------------------------------------------------------
# Footprint size estimation
# ---------------------------------------------------------------------------


def estimate_footprint_size(footprint_id: str) -> tuple[float, float]:
    """Estimate the physical dimensions (width, height) of a footprint in mm.

    Uses known package dimensions and heuristics to estimate footprint size
    without generating the full footprint geometry.

    Args:
        footprint_id: Footprint identifier string.

    Returns:
        ``(width_mm, height_mm)`` estimated bounding box.
    """
    fid = footprint_id.strip()
    upper = fid.upper()

    # SMD R/C packages
    for prefix in ("R_", "C_", "LED_"):
        if upper.startswith(prefix):
            pkg = fid[len(prefix):].upper()
            if pkg in _SMD_RC_DIMS:
                _, _, pitch, body_w, body_h = _SMD_RC_DIMS[pkg]
                return (body_w + 0.5, body_h + 0.5)
            return (2.5, 1.75)  # 0805 fallback

    # SOT-23 family
    if upper.startswith("SOT-23"):
        variant = fid.upper()
        if variant in _SOT23_VARIANTS:
            _, _, coords = _SOT23_VARIANTS[variant]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            return (max(xs) - min(xs) + 1.5, max(ys) - min(ys) + 1.5)
        return (3.0, 3.0)

    if upper == "SOT-223":
        return (7.0, 4.0)

    # USB-C
    if upper.startswith(("USB-C", "USB_C")):
        return (9.5, 8.0)

    # RJ45
    if upper.startswith("RJ45"):
        return (16.5, 14.0)

    # Pin headers/sockets
    if upper.startswith(("PINHEADER", "PINSOCKET")):
        pin_count = _parse_pin_count(fid)
        pitch = _parse_pitch(fid)
        rows = 2 if "2X" in upper or "2x" in fid else 1
        cols = pin_count // max(rows, 1)
        w = (cols - 1) * pitch + 2.5
        h = (rows - 1) * pitch + 2.5 if rows > 1 else 2.5
        return (w, h)

    # Terminal blocks
    if upper.startswith(("TERMINALBLOCK", "TB_")):
        pin_count = _parse_pin_count(fid)
        pitch = _parse_pitch(fid)
        return ((pin_count - 1) * pitch + 5.0, 7.0)

    # DIP switches — try to parse explicit dimensions from name first
    if upper.startswith("SW_DIP"):
        import re as _re
        dim_m = _re.search(r"(\d+\.?\d*)x(\d+\.?\d*)mm", fid)
        if dim_m:
            return (float(dim_m.group(1)) + 1.0, float(dim_m.group(2)) + 1.0)
        pin_count = _parse_pin_count(fid)
        half = max(pin_count // 2, 2)
        return (8.5, (half - 1) * 2.54 + 3.0)

    # Generic SMD ICs
    ic_prefixes = ("MSOP", "TSSOP", "SOIC", "QFP", "QFN", "SOP", "DFN", "SSOP", "LQFP")
    if any(upper.startswith(p) for p in ic_prefixes):
        pin_count = _parse_pin_count(fid)
        pitch = _parse_pitch(fid)
        if pitch > 2.0:
            pitch = 0.5
        half = pin_count // 2
        row_span = (half - 1) * pitch
        return (row_span / 2.0 + 3.0, row_span + 1.5)

    # Generic fallback
    return (3.0, 3.0)


# Keep math in module namespace so tests / type checker see no unused import
_PI = math.pi
