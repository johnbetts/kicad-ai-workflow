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


def _silk_side_marks(body_w: float, body_h: float) -> tuple[FootprintLine, ...]:
    """Short silkscreen lines on left/right edges of component body."""
    hw = body_w / 2.0
    hh = body_h / 2.0 * 0.6  # 60 % of half-height
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
    graphics: tuple[FootprintLine, ...] = (
        *_courtyard_rect(body_w, body_h),
        *_silk_side_marks(body_w, body_h),
    )
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5, LAYER_F_FAB),
    )
    return Footprint(
        lib_id=f"Device:{package}",
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

    _, pad_h, pitch, body_w, body_h = _SMD_RC_DIMS[package]
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
    return Footprint(
        lib_id=f"Device:LED_{package}",
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


# Keep math in module namespace so tests / type checker see no unused import
_PI = math.pi
