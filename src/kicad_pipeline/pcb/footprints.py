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
    KICAD_3DMODEL_VAR,
    LAYER_B_COURTYARD,
    LAYER_B_CU,
    LAYER_B_FAB,
    LAYER_B_MASK,
    LAYER_B_PASTE,
    LAYER_B_SILKSCREEN,
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
    Footprint3DModel,
    FootprintBBox,
    FootprintLine,
    FootprintText,
    OriginType,
    Pad,
    Point,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer flip mapping (F↔B)
# ---------------------------------------------------------------------------

_LAYER_FLIP_MAP: dict[str, str] = {
    LAYER_F_CU: LAYER_B_CU,
    LAYER_B_CU: LAYER_F_CU,
    LAYER_F_SILKSCREEN: LAYER_B_SILKSCREEN,
    LAYER_B_SILKSCREEN: LAYER_F_SILKSCREEN,
    LAYER_F_FAB: LAYER_B_FAB,
    LAYER_B_FAB: LAYER_F_FAB,
    LAYER_F_COURTYARD: LAYER_B_COURTYARD,
    LAYER_B_COURTYARD: LAYER_F_COURTYARD,
    LAYER_F_MASK: LAYER_B_MASK,
    LAYER_B_MASK: LAYER_F_MASK,
    LAYER_F_PASTE: LAYER_B_PASTE,
    LAYER_B_PASTE: LAYER_F_PASTE,
}


def _flip_layer(layer: str) -> str:
    """Flip a layer from front to back or vice versa.

    Returns the input unchanged if no flip mapping exists (e.g. Edge.Cuts).
    """
    return _LAYER_FLIP_MAP.get(layer, layer)


# ---------------------------------------------------------------------------
# 3D model path mapping
# ---------------------------------------------------------------------------

_3D_MODEL_MAP: tuple[tuple[str, str, str], ...] = (
    # (pattern_prefix, 3d_directory, name_template)
    # SMD passives
    ("R_0402", "Resistor_SMD.3dshapes", "R_0402_1005Metric.step"),
    ("R_0603", "Resistor_SMD.3dshapes", "R_0603_1608Metric.step"),
    ("R_0805", "Resistor_SMD.3dshapes", "R_0805_2012Metric.step"),
    ("R_1206", "Resistor_SMD.3dshapes", "R_1206_3216Metric.step"),
    ("R_1210", "Resistor_SMD.3dshapes", "R_1210_3225Metric.step"),
    ("C_0402", "Capacitor_SMD.3dshapes", "C_0402_1005Metric.step"),
    ("C_0603", "Capacitor_SMD.3dshapes", "C_0603_1608Metric.step"),
    ("C_0805", "Capacitor_SMD.3dshapes", "C_0805_2012Metric.step"),
    ("C_1206", "Capacitor_SMD.3dshapes", "C_1206_3216Metric.step"),
    ("C_1210", "Capacitor_SMD.3dshapes", "C_1210_3225Metric.step"),
    # LEDs
    ("LED_0402", "LED_SMD.3dshapes", "LED_0402_1005Metric.step"),
    ("LED_0603", "LED_SMD.3dshapes", "LED_0603_1608Metric.step"),
    ("LED_0805", "LED_SMD.3dshapes", "LED_0805_2012Metric.step"),
    ("LED_1206", "LED_SMD.3dshapes", "LED_1206_3216Metric.step"),
    # Transistors / small ICs
    ("SOT-23-6", "Package_TO_SOT_SMD.3dshapes", "SOT-23-6.step"),
    ("SOT-23-5", "Package_TO_SOT_SMD.3dshapes", "SOT-23-5.step"),
    ("SOT-23", "Package_TO_SOT_SMD.3dshapes", "SOT-23.step"),
    ("SOT-223", "Package_TO_SOT_SMD.3dshapes", "SOT-223-3_TabPin2.step"),
    # Diodes
    ("SOD-323", "Diode_SMD.3dshapes", "D_SOD-323.step"),
    ("SOD-123", "Diode_SMD.3dshapes", "D_SOD-123.step"),
    # Inductors
    ("L_1210", "Inductor_SMD.3dshapes", "L_1210_3225Metric.step"),
    ("L_1206", "Inductor_SMD.3dshapes", "L_1206_3216Metric.step"),
    ("L_0805", "Inductor_SMD.3dshapes", "L_0805_2012Metric.step"),
    # Crystals
    ("Crystal_SMD_3215", "Crystal.3dshapes", "Crystal_SMD_3215-2Pin_3.2x1.5mm.step"),
)


# Standard IC package dimensions for model filename resolution.
# Key = uppercase prefix + pin count (e.g. "MSOP-10"), value = dimension suffix.
_IC_DIMENSION_SUFFIXES: dict[str, str] = {
    "MSOP-8": "_3x3mm_P0.65mm",
    "MSOP-10": "_3x3mm_P0.5mm",
    "MSOP-16": "_4.9x3mm_P0.5mm",
    "TSSOP-8": "_3x3mm_P0.65mm",
    "TSSOP-14": "_5x4.4mm_P0.65mm",
    "TSSOP-16": "_4.4x5mm_P0.65mm",
    "TSSOP-20": "_6.5x4.4mm_P0.65mm",
    "TSSOP-24": "_7.8x4.4mm_P0.65mm",
    "TSSOP-28": "_9.7x4.4mm_P0.65mm",
    "SOIC-8": "_3.9x4.9mm_P1.27mm",
    "SOIC-14": "_3.9x8.7mm_P1.27mm",
    "SOIC-16": "_3.9x9.9mm_P1.27mm",
    "LQFP-32": "_7x7mm_P0.8mm",
    "LQFP-44": "_10x10mm_P0.8mm",
    "LQFP-48": "_7x7mm_P0.5mm",
    "LQFP-64": "_10x10mm_P0.5mm",
    "LQFP-100": "_14x14mm_P0.5mm",
    "QFN-16": "_3x3mm_P0.5mm",
    "QFN-20": "_4x4mm_P0.5mm",
    "QFN-24": "_4x4mm_P0.5mm",
    "QFN-32": "_5x5mm_P0.5mm",
    "QFN-48": "_7x7mm_P0.5mm",
    "SOP-4": "_3.8x4.1mm_P2.54mm",
}


def _ic_model_name(name: str, prefix: str) -> str:
    """Return the full model filename (without .step) for an IC package.

    If *name* already contains dimension info (e.g. ``MSOP-10_3x3mm_P0.5mm``)
    it is returned as-is.  Otherwise a standard dimension suffix is appended
    from :data:`_IC_DIMENSION_SUFFIXES`.
    """
    # Already has dimensions (contains 'mm')
    if "mm" in name:
        return name
    # Look up by uppercase name (e.g. "MSOP-10")
    suffix = _IC_DIMENSION_SUFFIXES.get(name.upper(), "")
    return f"{name}{suffix}"


def _model_for_package(lib_id: str, layer: str = LAYER_F_CU) -> Footprint3DModel | None:
    """Determine the 3D model path for a given KiCad lib_id.

    Args:
        lib_id: KiCad library identifier (e.g. ``"Resistor_SMD:R_0805_2012Metric"``).
        layer: Component layer — B.Cu connectors use PinSocket models.

    Returns:
        A :class:`Footprint3DModel` or ``None`` if no mapping found.
    """
    # Extract the footprint name (after ':') for pattern matching
    name = lib_id.split(":")[-1] if ":" in lib_id else lib_id
    upper = name.upper()

    # Pin headers / sockets — append _Vertical if no orientation suffix
    # Pads are along Y-axis matching KiCad convention — no model rotation needed.
    if "PINHEADER" in upper or "PINSOCKET" in upper:
        if layer == LAYER_B_CU or "PINSOCKET" in upper:
            dir_name = "Connector_PinSocket_2.54mm.3dshapes"
            model_name = name.replace("PinHeader", "PinSocket")
        else:
            dir_name = "Connector_PinHeader_2.54mm.3dshapes"
            model_name = name
        # KiCad model files require orientation suffix (e.g. _Vertical)
        if not any(s in model_name for s in ("_Vertical", "_Horizontal", "_SMD")):
            model_name += "_Vertical"
        path = f"{KICAD_3DMODEL_VAR}/{dir_name}/{model_name}.step"
        return Footprint3DModel(path=path)

    # IC packages — route to correct 3D library directory
    # LQFP/QFP → Package_QFP; MSOP/SOIC/etc. → Package_SO
    for prefix in ("LQFP", "QFP"):
        if upper.startswith(prefix):
            model_name = _ic_model_name(name, prefix)
            path = f"{KICAD_3DMODEL_VAR}/Package_QFP.3dshapes/{model_name}.step"
            return Footprint3DModel(path=path)
    for prefix in ("SOIC", "MSOP", "TSSOP", "SSOP", "QFN", "DFN", "SOP"):
        if upper.startswith(prefix):
            model_name = _ic_model_name(name, prefix)
            path = f"{KICAD_3DMODEL_VAR}/Package_SO.3dshapes/{model_name}.step"
            return Footprint3DModel(path=path)

    # Terminal blocks — KiCad uses vendor-specific dirs (Phoenix MKDS series)
    # Both our pads and the .step model use X-axis layout, pin 1 at origin.
    if "TERMINALBLOCK" in upper or "MKDS" in upper:
        import re as _re
        # Extract pin count from "1x06" or "1x02" pattern in lib_id
        pin_count = 2  # default
        nx_match = _re.search(r"1x(\d+)", lib_id)
        if nx_match:
            pin_count = int(nx_match.group(1))
        # Extract pitch from "P5.08mm" pattern
        pitch_match = _re.search(r"P([\d.]+)mm", lib_id)
        pitch = float(pitch_match.group(1)) if pitch_match else 5.08
        model_name = (
            f"TerminalBlock_Phoenix_MKDS-1,5-{pin_count}-{pitch:.2f}"
            f"_1x{pin_count:02d}_P{pitch:.2f}mm_Horizontal"
        )
        path = (
            f"{KICAD_3DMODEL_VAR}/TerminalBlock_Phoenix.3dshapes/"
            f"{model_name}.step"
        )
        # Rotate 180° so terminal openings face the board edge.
        # The model origin is at pin 1, so 180° rotation around (0,0) moves
        # the model off the pads. Offset by (N-1)*pitch in X to re-center.
        offset_x = (pin_count - 1) * pitch
        return Footprint3DModel(
            path=path,
            offset=(offset_x, 0.0, 0.0),
            rotate=(0.0, 0.0, 180.0),
        )

    # ESP32 / RF modules
    if "ESP32" in upper or "WROOM" in upper:
        # Extract the module name for the 3D model
        model_name = name.split(":")[-1] if ":" in name else name
        path = f"{KICAD_3DMODEL_VAR}/RF_Module.3dshapes/{model_name}.step"
        return Footprint3DModel(path=path)

    # Relays
    if "RELAY" in upper and "SANYOU" in upper:
        path = (
            f"{KICAD_3DMODEL_VAR}/Relay_THT.3dshapes/"
            "Relay_SPDT_SANYOU_SRD_Series_Form_C.step"
        )
        return Footprint3DModel(path=path)

    # DIP switches — must check BEFORE generic SW_ match (SW_DIP contains "SPST")
    # KiCad convention uses 90° Z rotation for DIP switch models
    if upper.startswith("SW_DIP"):
        import re as _re_dip
        sw_count_m = _re_dip.search(r"SPSTx(\d+)", name, _re_dip.IGNORECASE)
        sw_count = sw_count_m.group(1) if sw_count_m else "01"
        path = (
            f"{KICAD_3DMODEL_VAR}/Button_Switch_THT.3dshapes/"
            f"SW_DIP_SPSTx{sw_count}_Slide_9.78x4.72mm_W7.62mm_P2.54mm.step"
        )
        return Footprint3DModel(path=path, rotate=(0.0, 0.0, 90.0))

    # Tactile switches — SMD vs THT
    # Check full lib_id (not just name) for SMD indicator
    lib_upper = lib_id.upper()
    if upper.startswith("SW_PUSH") or (upper.startswith("SW_") and "SPST" in upper):
        if "SMD" in lib_upper or "SMD" in upper:
            path = (
                f"{KICAD_3DMODEL_VAR}/Button_Switch_SMD.3dshapes/"
                "SW_SPST_EVQPE1.step"
            )
        else:
            path = f"{KICAD_3DMODEL_VAR}/Button_Switch_THT.3dshapes/SW_PUSH_6mm.step"
        return Footprint3DModel(path=path)

    # RJ45 — use best-match Amphenol model (KiCad ships RJHSE538X)
    if "RJ45" in upper:
        path = (
            f"{KICAD_3DMODEL_VAR}/Connector_RJ.3dshapes/"
            "RJ45_Amphenol_RJHSE538X.step"
        )
        return Footprint3DModel(path=path)

    # USB-C
    if upper.startswith(("USB-C", "USB_C")):
        path = (
            f"{KICAD_3DMODEL_VAR}/Connector_USB.3dshapes/"
            "USB_C_Receptacle_GCT_USB4105-xx-A_16P_TopMnt_Horizontal.step"
        )
        return Footprint3DModel(path=path)

    # DIP packages (optocouplers, etc.)
    if upper.startswith("DIP-"):
        path = f"{KICAD_3DMODEL_VAR}/Package_DIP.3dshapes/{name}.step"
        return Footprint3DModel(path=path)

    # WS2812B addressable LEDs (size-aware)
    if "WS2812" in upper:
        if "2.0X2.0" in upper or "2020" in upper:
            # No 2.0x2.0 .step in KiCad library; use Mini 3.5x3.5 as closest
            step = "LED_WS2812B-Mini_PLCC4_3.5x3.5mm.step"
        elif "3.5X3.5" in upper or "3535" in upper or "P2.45" in upper:
            step = "LED_WS2812B-Mini_PLCC4_3.5x3.5mm.step"
        elif "PLCC6" in upper:
            step = "LED_WS2812_PLCC6_5.0x5.0mm_P1.6mm.step"
        else:
            step = "LED_WS2812B_PLCC4_5.0x5.0mm_P3.2mm.step"
        path = f"{KICAD_3DMODEL_VAR}/LED_SMD.3dshapes/{step}"
        return Footprint3DModel(path=path)

    # Micro SD card slot
    if upper.startswith(("TF_PUSH", "MICROSD", "MICRO_SD")):
        path = (
            f"{KICAD_3DMODEL_VAR}/Connector_Card.3dshapes/"
            "microSD_HC_Hirose_DM3AT-SF-PEJM5.step"
        )
        return Footprint3DModel(path=path)

    # Generic single-row connectors (Conn_01xNN) → PinHeader_1xNN
    if upper.startswith("CONN_01X"):
        # Extract pin count: Conn_01x14_P2.54mm → 14
        import re
        m = re.match(r"CONN_01X(\d+)", upper)
        if m:
            pin_count = m.group(1)
            model_name = f"PinHeader_1x{pin_count}_P2.54mm_Vertical"
            if layer == LAYER_B_CU:
                dir_name = "Connector_PinSocket_2.54mm.3dshapes"
                model_name = f"PinSocket_1x{pin_count}_P2.54mm_Vertical"
            else:
                dir_name = "Connector_PinHeader_2.54mm.3dshapes"
            path = f"{KICAD_3DMODEL_VAR}/{dir_name}/{model_name}.step"
            return Footprint3DModel(path=path)

    # Generic relay SPDT (non-Sanyou)
    if "RELAY" in upper and "SPDT" in upper:
        path = (
            f"{KICAD_3DMODEL_VAR}/Relay_THT.3dshapes/"
            "Relay_SPDT_Omron_G5V-1.step"
        )
        return Footprint3DModel(path=path)

    # Static pattern map for passives/transistors/diodes/inductors/crystals
    # Use `in` instead of `startswith` so patterns like "SOD-323" match
    # footprint names like "D_SOD-323" which have a prefix.
    for pattern, directory, model_file in _3D_MODEL_MAP:
        if pattern.upper() in upper:
            path = f"{KICAD_3DMODEL_VAR}/{directory}/{model_file}"
            return Footprint3DModel(path=path)

    return None


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

# SOT-23 variants: (pad_w, pad_h, pin_coords)
# Dimensions from KiCad official footprints (IPC-compliant).
# pin_coords: list of (x, y) relative to footprint origin
_SOT23_VARIANTS: dict[str, tuple[float, float, list[tuple[float, float]]]] = {
    "SOT-23": (
        1.475,
        0.6,
        [(-0.9375, -0.95), (-0.9375, 0.95), (0.9375, 0.0)],
    ),
    "SOT-23-5": (
        1.325,
        0.6,
        [
            (-1.1375, -0.95),
            (-1.1375, 0.0),
            (-1.1375, 0.95),
            (1.1375, 0.95),
            (1.1375, -0.95),
        ],
    ),
    "SOT-23-6": (
        1.325,
        0.6,
        [
            (-1.1375, -0.95),
            (-1.1375, 0.0),
            (-1.1375, 0.95),
            (1.1375, 0.95),
            (1.1375, 0.0),
            (1.1375, -0.95),
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

# RJ45 HR911105A pin geometry (from KiCad official footprint)
# Signal pins are staggered: odd pins (1,3,5,7) at y=0, even pins (2,4,6,8) at y=-2.54
_RJ45_SIGNAL_COUNT: int = 8
_RJ45_SIGNAL_DRILL_MM: float = 0.89
_RJ45_SIGNAL_PAD_MM: float = 1.5
# Positions from official KiCad RJHSE538X footprint — 1.016mm pitch zigzag
_RJ45_SIGNAL_POSITIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),        # pin 1 (front row)
    (1.016, 1.78),     # pin 2 (back row)
    (2.032, 0.0),      # pin 3 (front row)
    (3.048, 1.78),     # pin 4 (back row)
    (4.064, 0.0),      # pin 5 (front row)
    (5.08, 1.78),      # pin 6 (back row)
    (6.096, 0.0),      # pin 7 (front row)
    (7.112, 1.78),     # pin 8 (back row)
)
# LED pins 9-12 (from official KiCad RJHSE538X footprint)
_RJ45_LED_DRILL_MM: float = 0.89
_RJ45_LED_PAD_MM: float = 1.5
_RJ45_LED_POSITIONS: tuple[tuple[float, float], ...] = (
    (-3.3, 6.6),       # pin 9
    (-1.01, 6.6),      # pin 10
    (8.13, 6.6),       # pin 11
    (10.42, 6.6),      # pin 12
)
# Shield pads (from official KiCad RJHSE538X footprint)
_RJ45_SHIELD_DRILL_MM: float = 1.57
_RJ45_SHIELD_PAD_MM: float = 2.3
_RJ45_SHIELD_POSITIONS: tuple[tuple[float, float], ...] = (
    (-4.57, 0.89),
    (11.69, 0.89),
)
# NPTH mounting holes (from official KiCad RJHSE538X footprint)
_RJ45_NPTH_DIAM_MM: float = 3.25
_RJ45_NPTH_POSITIONS: tuple[tuple[float, float], ...] = (
    (-2.79, -2.54),
    (9.91, -2.54),
)

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
    body_w: float,
    body_h: float,
    clearance: float = PCB_COURTYARD_CLEARANCE_MM,
    layer: str = LAYER_F_COURTYARD,
    cx: float = 0.0,
    cy: float = 0.0,
) -> tuple[FootprintLine, ...]:
    """Build a rectangular courtyard from body dimensions + clearance.

    Args:
        cx: X center offset (default 0.0).
        cy: Y center offset (default 0.0).
    """
    hw = body_w / 2.0 + clearance
    hh = body_h / 2.0 + clearance
    w = PCB_SILKSCREEN_LINE_WIDTH_MM
    return (
        FootprintLine(start=Point(cx - hw, cy - hh), end=Point(cx + hw, cy - hh), layer=layer, width=w),
        FootprintLine(start=Point(cx + hw, cy - hh), end=Point(cx + hw, cy + hh), layer=layer, width=w),
        FootprintLine(start=Point(cx + hw, cy + hh), end=Point(cx - hw, cy + hh), layer=layer, width=w),
        FootprintLine(start=Point(cx - hw, cy + hh), end=Point(cx - hw, cy - hh), layer=layer, width=w),
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

    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()

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
        models=models,
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
    model = _model_for_package(led_lib_id)
    models = (model,) if model is not None else ()
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
        models=models,
    )


def make_sod123(
    ref: str,
    value: str,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """SOD-123 diode footprint (1.65 x 2.68 mm body, 2 pads).

    Pad 1 = cathode (K), Pad 2 = anode (A). Pitch 2.0 mm.

    Args:
        ref: Reference designator (e.g. "D1").
        value: Component value string.
        layer: Primary copper layer.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    body_w = 2.68
    body_h = 1.65
    pad_w = 0.91
    pad_h = 1.22
    pitch = 2.2
    pads = (
        _smd_pad("1", -pitch / 2.0, 0.0, pad_w, pad_h, layer),
        _smd_pad("2", pitch / 2.0, 0.0, pad_w, pad_h, layer),
    )
    pad_edge_x = pitch / 2.0 + pad_w / 2.0
    graphics = (
        *_courtyard_rect(body_w, body_h),
        *_silk_side_marks(body_w, body_h, pad_edge_x=pad_edge_x),
    )
    texts = (
        _ref_text(ref, -(body_h / 2.0 + 1.0), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + 1.0, LAYER_F_FAB),
    )
    lib_id = "Diode_SMD:D_SOD-123"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=layer, pads=pads, graphics=graphics, texts=texts,
        attr="smd", models=models,
    )


def make_inductor_smd(
    ref: str,
    value: str,
    package: str = "1210",
    layer: str = LAYER_F_CU,
) -> Footprint:
    """SMD inductor footprint — same geometry as R/C packages.

    Args:
        ref: Reference designator (e.g. "L1").
        value: Component value string (e.g. "4.7uH").
        package: IPC package code (0805, 1206, 1210).
        layer: Primary copper layer.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    pkg = package if package in _SMD_RC_DIMS else "1210"
    pad_w, pad_h, pitch, body_w, body_h = _SMD_RC_DIMS[pkg]
    pads = (
        _smd_pad("1", -pitch / 2.0, 0.0, pad_w, pad_h, layer),
        _smd_pad("2", pitch / 2.0, 0.0, pad_w, pad_h, layer),
    )
    pad_edge_x = pitch / 2.0 + pad_w / 2.0
    graphics = (
        *_courtyard_rect(body_w, body_h),
        *_silk_side_marks(body_w, body_h, pad_edge_x=pad_edge_x),
    )
    texts = (
        _ref_text(ref, -(body_h / 2.0 + 1.0), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + 1.0, LAYER_F_FAB),
    )
    lib_id = f"Inductor_SMD:L_{pkg}"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=layer, pads=pads, graphics=graphics, texts=texts,
        attr="smd", models=models,
    )


def make_tact_switch(
    ref: str,
    value: str,
    size_mm: float = 4.5,
) -> Footprint:
    """Tactile push-button switch (4-pin through-hole).

    Supports common sizes: 3.0mm, 4.5mm, 6.0mm.
    Standard 2-pin-pair wiring: pins 1+2 are one pole, pins 3+4 are the other.
    The footprint only exposes 2 logical pins.

    Args:
        ref: Reference designator (e.g. "SW1").
        value: Component value string.
        size_mm: Body size (one side of square body).

    Returns:
        Fully constructed :class:`Footprint`.
    """
    # Scale pad layout to match body size
    if size_mm <= 3.5:
        # Small tact switch (e.g. 3.0x3.0mm)
        half_x = 2.0
        half_y = 1.5
        drill = 0.8
        pad_diam = 1.4
    elif size_mm <= 5.0:
        # Medium tact switch (e.g. 4.5x4.5mm)
        half_x = 2.75
        half_y = 2.0
        drill = 0.9
        pad_diam = 1.6
    else:
        # Standard 6mm tact switch
        half_x = 3.25
        half_y = 2.25
        drill = 1.0
        pad_diam = 1.8

    pads = (
        _thru_pad("1", -half_x, -half_y, pad_diam, drill),
        _thru_pad("1", half_x, -half_y, pad_diam, drill),
        _thru_pad("2", -half_x, half_y, pad_diam, drill),
        _thru_pad("2", half_x, half_y, pad_diam, drill),
    )
    body = size_mm
    graphics = _courtyard_rect(body + 1.5, body + 1.5)
    texts = (
        _ref_text(ref, -(body / 2.0 + 1.5), LAYER_F_SILKSCREEN),
        _val_text(value, body / 2.0 + 1.5, LAYER_F_FAB),
    )
    lib_id = f"Button_Switch_THT:SW_Push_{size_mm}x{size_mm}mm"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=LAYER_F_CU, pads=pads, graphics=graphics, texts=texts,
        attr="through_hole", models=models,
    )


def make_smd_tact_switch(
    ref: str,
    value: str,
    width_mm: float = 5.1,
    height_mm: float = 5.1,
) -> Footprint:
    """SMD tactile push-button switch (XKB TS-1187A style, 4-pad).

    Matches XKB TS-1187A-B-A-B (LCSC C318884) footprint layout:
    4 pads (two per terminal), body 5.1×5.1mm.
    Pin 1 pads at left-top and right-top, pin 2 pads at left-bottom
    and right-bottom (internally shorted per side).

    Args:
        ref: Reference designator (e.g. "SW1").
        value: Component value string.
        width_mm: Body width in mm (default 5.1).
        height_mm: Body height in mm (default 5.1).

    Returns:
        Fully constructed :class:`Footprint`.
    """
    # XKB TS-1187A dimensions from datasheet:
    # Body: 5.1×5.1mm, pad size: 1.5×3.0mm
    # Pad centers: horizontal span 7.0mm (±3.5), vertical span 5.0mm (±2.5)
    pad_size_x = 1.5
    pad_size_y = 3.0
    pad_x = 3.5   # horizontal center-to-center / 2
    pad_y = 2.5   # vertical center-to-center / 2

    # 4 pads: two "1" pads (left+right, top row), two "2" pads (left+right, bottom)
    pads = (
        _smd_pad("1", -pad_x, -pad_y, pad_size_x, pad_size_y, LAYER_F_CU),
        _smd_pad("1", pad_x, -pad_y, pad_size_x, pad_size_y, LAYER_F_CU),
        _smd_pad("2", -pad_x, pad_y, pad_size_x, pad_size_y, LAYER_F_CU),
        _smd_pad("2", pad_x, pad_y, pad_size_x, pad_size_y, LAYER_F_CU),
    )
    court_w = pad_x * 2 + pad_size_x + 0.5
    court_h = pad_y * 2 + pad_size_y + 0.5
    graphics = _courtyard_rect(court_w, court_h)
    texts = (
        _ref_text(ref, -(court_h / 2.0 + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, court_h / 2.0 + 0.5, LAYER_F_FAB),
    )
    lib_id = "Button_Switch_SMD:SW_SPST_TL3305A"
    model = Footprint3DModel(
        path=f"{KICAD_3DMODEL_VAR}/Button_Switch_SMD.3dshapes/SW_SPST_TL3305A.step",
    )
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=LAYER_F_CU, pads=pads, graphics=graphics, texts=texts,
        attr="smd", models=(model,),
    )


def make_relay_spdt(
    ref: str,
    value: str,
) -> Footprint:
    """SPDT relay footprint (e.g. SRD-05VDC-SL-C, Songle SRD series).

    Pad positions match KiCad's official ``Relay_SPDT_SANYOU_SRD_Series_Form_C``
    footprint with origin at pin 1.  5 pins: COM (1), Coil- (2), NO (3),
    NC (4), Coil+ (5).  Pins 1/3/4 = 3mm pads (contacts), pins 2/5 = 2.5mm (coil).

    Args:
        ref: Reference designator (e.g. "K1").
        value: Component value string.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    # Pad positions from KiCad's official footprint (origin at pin 1)
    # Pins 1,3,4 = 3mm pad / 1.3mm drill (contacts); Pins 2,5 = 2.5mm pad / 1.0mm drill (coil)
    pads = (
        _thru_pad("1", 0.0, 0.0, 3.0, 1.3),         # COM (switching arm)
        _thru_pad("2", 1.95, 6.05, 2.5, 1.0),        # Coil-
        _thru_pad("3", 14.15, 6.05, 3.0, 1.3),       # NO (normally open)
        _thru_pad("4", 14.2, -6.0, 3.0, 1.3),        # NC (normally closed)
        _thru_pad("5", 1.95, -5.95, 2.5, 1.0),       # Coil+
    )
    # Body outline: -1.4 to 18.4 in X, -7.8 to 7.8 in Y
    body_w = 19.8  # 18.4 - (-1.4)
    body_h = 15.6  # 7.8 - (-7.8)
    # Courtyard centred on body centre (8.5, 0)
    cx = (-1.4 + 18.4) / 2.0
    cy = (-7.8 + 7.8) / 2.0
    hw = body_w / 2.0 + PCB_COURTYARD_CLEARANCE_MM
    hh = body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM
    graphics = (
        FootprintLine(
            start=Point(cx - hw, cy - hh), end=Point(cx + hw, cy - hh),
            layer=LAYER_F_COURTYARD, width=0.05,
        ),
        FootprintLine(
            start=Point(cx + hw, cy - hh), end=Point(cx + hw, cy + hh),
            layer=LAYER_F_COURTYARD, width=0.05,
        ),
        FootprintLine(
            start=Point(cx + hw, cy + hh), end=Point(cx - hw, cy + hh),
            layer=LAYER_F_COURTYARD, width=0.05,
        ),
        FootprintLine(
            start=Point(cx - hw, cy + hh), end=Point(cx - hw, cy - hh),
            layer=LAYER_F_COURTYARD, width=0.05,
        ),
    )
    texts = (
        _ref_text(ref, cy - (body_h / 2.0 + 1.5), LAYER_F_SILKSCREEN),
        _val_text(value, cy + body_h / 2.0 + 1.5, LAYER_F_FAB),
    )
    lib_id = "Relay_THT:Relay_SPDT_SANYOU_SRD_Series_Form_C"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=LAYER_F_CU, pads=pads, graphics=graphics, texts=texts,
        attr="through_hole", models=models,
    )


def make_esp32_wroom(
    ref: str,
    value: str,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """ESP32-S3-WROOM-1 module footprint (18x25.5mm body, 41 castellated pads).

    Pad layout per datasheet Figure 3-1 (top view, antenna at top/north):
      - Left side:  14 pads (pins 1-14),  top to bottom
      - Bottom edge: 12 pads (pins 15-26), left to right
      - Right side: 14 pads (pins 27-40), bottom to top
      - Center:     pad 41 (GND exposed pad)

    Args:
        ref: Reference designator (e.g. "U3").
        value: Component value string.
        layer: Primary copper layer.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    body_w = 18.0
    body_h = 25.5
    pad_w = 0.9
    pad_h = 1.2
    pitch = 1.27

    # 41 pads: left(14) + bottom(12) + right(14) + center GND(1)
    pad_list: list[Pad] = []

    # Left column: 14 pads (pins 1-14), top to bottom
    # Pin 1 (GND) at top-left near antenna, pin 14 (IO20) at bottom-left.
    # The topmost pad starts ~2.5mm below the top edge (antenna zone above).
    left_x = -(body_w / 2.0 - pad_h / 2.0)
    n_left = 14
    left_top_y = -(body_h / 2.0) + 2.5 + pad_h / 2.0  # 2.5mm from top edge
    for i in range(n_left):
        pad_list.append(_smd_pad(
            str(i + 1), left_x, left_top_y + i * pitch, pad_h, pad_w, layer,
        ))

    # Bottom row: 12 pads (pins 15-26), left to right
    # Pin 15 (IO3) at bottom-left, pin 26 (IO45) at bottom-right.
    bottom_y = body_h / 2.0 - pad_h / 2.0
    n_bottom = 12
    start_x = -((n_bottom - 1) * pitch) / 2.0
    for i in range(n_bottom):
        pad_list.append(_smd_pad(
            str(15 + i), start_x + i * pitch, bottom_y, pad_w, pad_h, layer,
        ))

    # Right column: 14 pads (pins 27-40), bottom to top
    # Pin 27 (IO0) at bottom-right, pin 40 (GND) at top-right.
    right_x = body_w / 2.0 - pad_h / 2.0
    n_right = 14
    right_bot_y = body_h / 2.0 - 2.5 - pad_h / 2.0  # 2.5mm from bottom edge
    for i in range(n_right):
        pad_list.append(_smd_pad(
            str(27 + i), right_x, right_bot_y - i * pitch, pad_h, pad_w, layer,
        ))

    # Center GND pad (large thermal pad underneath) — pin 41
    pad_list.append(Pad(
        number="41",
        pad_type="smd",
        shape="rect",
        position=Point(0.0, 0.0),
        size_x=6.7,
        size_y=6.7,
        layers=(layer, LAYER_F_PASTE if layer == LAYER_F_CU else LAYER_B_PASTE,
                LAYER_F_MASK if layer == LAYER_F_CU else LAYER_B_MASK),
    ))

    graphics = _courtyard_rect(body_w, body_h)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + 1.5), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + 1.5, LAYER_F_FAB),
    )
    lib_id = "RF_Module:ESP32-S3-WROOM-1"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=layer, pads=tuple(pad_list), graphics=graphics, texts=texts,
        attr="smd", models=models,
    )


def make_crystal_smd(
    ref: str,
    value: str,
    size_w: float = 3.2,
    size_h: float = 1.5,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """SMD crystal oscillator (2-pin, e.g. 3.2x1.5mm HC49/SD package).

    Args:
        ref: Reference designator (e.g. "Y1").
        value: Component value string (e.g. "25MHz").
        size_w: Crystal body width in mm.
        size_h: Crystal body height in mm.
        layer: Primary copper layer.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    pad_w = 1.2
    pad_h = 1.0
    pitch = size_w - pad_w + 0.4
    pads = (
        _smd_pad("1", -pitch / 2.0, 0.0, pad_w, pad_h, layer),
        _smd_pad("2", pitch / 2.0, 0.0, pad_w, pad_h, layer),
    )
    graphics = _courtyard_rect(size_w, size_h)
    texts = (
        _ref_text(ref, -(size_h / 2.0 + 1.0), LAYER_F_SILKSCREEN),
        _val_text(value, size_h / 2.0 + 1.0, LAYER_F_FAB),
    )
    lib_id = f"Crystal:Crystal_SMD_{size_w:.0f}215-2Pin_{size_w}x{size_h}mm"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=layer, pads=pads, graphics=graphics, texts=texts,
        attr="smd", models=models,
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
    sot_lib_id = f"Package_TO_SOT_SMD:{variant}"
    model = _model_for_package(sot_lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=sot_lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=layer,
        pads=pads,
        graphics=graphics,
        texts=texts,
        attr="smd",
        models=models,
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

    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()

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
        models=models,
    )


def make_pin_header_socket(
    ref: str,
    value: str,
    pin_count: int,
    pitch_mm: float = 2.54,
    rows: int = 1,
    lib_id: str = "",
    row_swap: bool = False,
    layer: str = LAYER_F_CU,
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
        layer: Copper layer for the footprint (``F.Cu`` or ``B.Cu``).

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug(
        "make_pin_header_socket ref=%s pins=%d rows=%d layer=%s",
        ref, pin_count, rows, layer,
    )
    drill_mm = 1.0
    pad_diam = 1.7
    cols = pin_count // max(rows, 1)
    row_pitch = pitch_mm if rows > 1 else 0.0

    # KiCad convention: pin 1 at origin (0,0), rows along X-axis, cols along Y-axis
    pads: list[Pad] = []
    pin_num = 1
    for col in range(cols):
        for row in range(rows):
            x = row * row_pitch
            y = col * pitch_mm
            if row_swap:
                x = -x
            pads.append(_thru_pad(str(pin_num), x, y, pad_diam, drill_mm))
            pin_num += 1

    is_back = layer == LAYER_B_CU
    silk_layer = LAYER_B_SILKSCREEN if is_back else LAYER_F_SILKSCREEN
    fab_layer = LAYER_B_FAB if is_back else LAYER_F_FAB
    crtyd_layer = LAYER_B_COURTYARD if is_back else LAYER_F_COURTYARD

    span_x = (rows - 1) * row_pitch
    span_y = (cols - 1) * pitch_mm
    cx = span_x / 2.0  # center of pad span in X
    cy = span_y / 2.0  # center of pad span in Y
    body_w = span_x + pad_diam + 1.5
    body_h = span_y + pad_diam + 1.5
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h, layer=crtyd_layer, cx=cx, cy=cy),)
    ref_y = cy - (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5)
    val_y = cy + (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5)
    texts = (
        FootprintText(text_type="reference", text=ref,
                      position=Point(cx, ref_y), layer=silk_layer, effects_size=1.0),
        FootprintText(text_type="value", text=value,
                      position=Point(cx, val_y), layer=fab_layer, effects_size=1.0),
    )
    if not lib_id:
        if is_back:
            lib_prefix = f"Connector_PinSocket_{pitch_mm:.2f}mm"
            pitch_str = f"P{pitch_mm:.2f}mm_Vertical"
            if rows > 1:
                lib_id = f"{lib_prefix}:PinSocket_{rows}x{cols:02d}_{pitch_str}"
            else:
                lib_id = f"{lib_prefix}:PinSocket_1x{cols:02d}_{pitch_str}"
        else:
            lib_prefix = f"Connector_PinHeader_{pitch_mm:.2f}mm"
            pitch_str = f"P{pitch_mm:.2f}mm_Vertical"
            if rows > 1:
                lib_id = f"{lib_prefix}:PinHeader_{rows}x{cols:02d}_{pitch_str}"
            else:
                lib_id = f"{lib_prefix}:PinHeader_1x{cols:02d}_{pitch_str}"

    # For B.Cu connectors, rewrite PinHeader→PinSocket in lib_id
    if is_back and "PinHeader" in lib_id:
        lib_id = lib_id.replace("PinHeader", "PinSocket").replace(
            "Connector_PinHeader", "Connector_PinSocket"
        )

    model = _model_for_package(lib_id, layer)
    models = (model,) if model is not None else ()

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=layer,
        pads=tuple(pads),
        graphics=graphics,
        texts=texts,
        attr="through_hole",
        models=models,
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

    # Pin 1 at origin, extending right — matches KiCad MKDS convention
    pads = tuple(
        _thru_pad(
            str(i + 1),
            i * pitch_mm,
            0.0,
            pad_diam,
            drill_mm,
        )
        for i in range(pin_count)
    )
    span = (pin_count - 1) * pitch_mm
    body_w = span + pad_diam + 4.0
    body_h = pad_diam + 6.0
    # Center courtyard on the pad span
    cx = span / 2.0
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h, cx=cx),)
    ref_y = -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5)
    val_y = body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5
    texts = (
        FootprintText(text_type="reference", text=ref,
                      position=Point(cx, ref_y), layer=LAYER_F_SILKSCREEN, effects_size=1.0),
        FootprintText(text_type="value", text=value,
                      position=Point(cx, val_y), layer=LAYER_F_FAB, effects_size=1.0),
    )
    lib_id = (
        f"TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS-1,5-{pin_count}-"
        f"{pitch_mm:.2f}_1x{pin_count:02d}_P{pitch_mm:.2f}mm_Horizontal"
    )

    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()

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
        models=models,
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

    # KiCad convention: pin 1 at origin, pin N+1 at (row_pitch, 0) for x01
    # Multi-position: left column pins 1..half going down, right column bottom-to-top
    pads: list[Pad] = []
    for i in range(half):
        y = i * pitch_mm
        pads.append(_thru_pad(str(i + 1), 0.0, y, pad_diam, drill_mm))
    for i in range(half):
        y = (half - 1 - i) * pitch_mm
        pads.append(_thru_pad(str(half + i + 1), row_pitch, y, pad_diam, drill_mm))

    span_y = (half - 1) * pitch_mm
    cx = row_pitch / 2.0
    cy = span_y / 2.0
    body_w = row_pitch + pad_diam + 0.5
    body_h = span_y + pad_diam + 0.5
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h, cx=cx, cy=cy),)
    texts = (
        FootprintText(text_type="reference", text=ref,
                      position=Point(cx, cy - (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5)),
                      layer=LAYER_F_SILKSCREEN, effects_size=1.0),
        FootprintText(text_type="value", text=value,
                      position=Point(cx, cy + body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5),
                      layer=LAYER_F_FAB, effects_size=1.0),
    )
    lib_id = (
        f"Button_Switch_THT:SW_DIP_SPSTx{half:02d}_Slide_"
        f"9.78x4.72mm_W{row_pitch:.2f}mm_P{pitch_mm:.2f}mm"
    )
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()

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
        models=models,
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
    lib_id = "Connector_USB:USB_C_Receptacle_GCT_USB4105"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
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
        models=models,
    )


def make_rj45(ref: str, value: str = "RJ45") -> Footprint:
    """RJ45 with integrated magnetics footprint (Hanrun HR911105A, through-hole).

    8 signal pins in staggered zigzag (odd at y=0, even at y=-2.54),
    4 LED pins, 2 shield pads, and 2 NPTH mounting holes.
    Pad/drill sizes match the official KiCad HR911105A footprint.

    Args:
        ref: Reference designator (e.g. "J2").
        value: Component value string, default "RJ45".

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("make_rj45 ref=%s", ref)
    pads: list[Pad] = []

    # 8 signal pins — staggered zigzag layout
    for i, (x, y) in enumerate(_RJ45_SIGNAL_POSITIONS):
        shape = "roundrect" if i == 0 else "circle"
        pads.append(
            _thru_pad(str(i + 1), x, y, _RJ45_SIGNAL_PAD_MM, _RJ45_SIGNAL_DRILL_MM, shape=shape)
        )

    # 4 LED pins
    for i, (x, y) in enumerate(_RJ45_LED_POSITIONS):
        pads.append(_thru_pad(str(9 + i), x, y, _RJ45_LED_PAD_MM, _RJ45_LED_DRILL_MM))

    # 2 shield pads
    for x, y in _RJ45_SHIELD_POSITIONS:
        pads.append(_thru_pad("SH", x, y, _RJ45_SHIELD_PAD_MM, _RJ45_SHIELD_DRILL_MM))

    # 2 NPTH mounting holes (no copper)
    for mx, my in _RJ45_NPTH_POSITIONS:
        pads.append(
            Pad(
                number="",
                pad_type="np_thru_hole",
                shape="circle",
                position=Point(mx, my),
                size_x=_RJ45_NPTH_DIAM_MM,
                size_y=_RJ45_NPTH_DIAM_MM,
                layers=(LAYER_F_CU, LAYER_B_CU),
                drill_diameter=_RJ45_NPTH_DIAM_MM,
            )
        )

    # Courtyard matches official KiCad RJHSE538X: (-6.22, -8.5) to (13.34, 8.25)
    cx = 3.56   # (13.34 + -6.22) / 2
    cy = -0.125  # (8.25 + -8.5) / 2
    body_w = 19.56  # 13.34 - -6.22
    body_h = 16.75  # 8.25 - -8.5
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h, cx=cx, cy=cy),)
    texts = (
        _ref_text(ref, cy - (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, cy + (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), LAYER_F_FAB),
    )
    lib_id = "Connector_RJ:RJ45_Amphenol_RJHSE538X"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
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
        models=models,
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
        _ref_text(ref, -(crtyd_size / 2.0 + 0.5), LAYER_F_FAB),
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


def make_sod323(
    ref: str,
    value: str,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """SOD-323 diode footprint (1.25 x 1.7 mm body, 2 pads).

    Pad 1 = cathode (K), Pad 2 = anode (A).

    Args:
        ref: Reference designator (e.g. "D1").
        value: Component value string.
        layer: Primary copper layer.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    body_w = 1.7
    body_h = 1.25
    pad_w = 0.6
    pad_h = 0.55
    pitch = 2.1
    pads = (
        _smd_pad("1", -pitch / 2.0, 0.0, pad_w, pad_h, layer),
        _smd_pad("2", pitch / 2.0, 0.0, pad_w, pad_h, layer),
    )
    pad_edge_x = pitch / 2.0 + pad_w / 2.0
    graphics = (
        *_courtyard_rect(body_w, body_h),
        *_silk_side_marks(body_w, body_h, pad_edge_x=pad_edge_x),
    )
    texts = (
        _ref_text(ref, -(body_h / 2.0 + 1.0), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + 1.0, LAYER_F_FAB),
    )
    lib_id = "Diode_SMD:D_SOD-323"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=layer, pads=pads, graphics=graphics, texts=texts,
        attr="smd", models=models,
    )


def make_dip_package(
    ref: str,
    value: str,
    pin_count: int = 4,
    pitch_mm: float = 2.54,
    row_spacing_mm: float = 7.62,
    lib_id: str = "",
) -> Footprint:
    """Generic through-hole DIP package (optocoupler, timer, etc.).

    Args:
        ref: Reference designator (e.g. "U7").
        value: Component value string.
        pin_count: Total number of pins (must be even).
        pitch_mm: Pin pitch in mm (default 2.54).
        row_spacing_mm: Row-to-row distance in mm (default 7.62).
        lib_id: KiCad library footprint ID override.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("make_dip_package ref=%s pins=%d", ref, pin_count)
    drill_mm = 0.8
    pad_diam = 1.6
    half = pin_count // 2

    pads: list[Pad] = []
    # Left column: pins 1..half top-to-bottom
    for i in range(half):
        y = i * pitch_mm - (half - 1) * pitch_mm / 2.0
        pads.append(_thru_pad(str(i + 1), -row_spacing_mm / 2.0, y, pad_diam, drill_mm))
    # Right column: pins half+1..pin_count bottom-to-top
    for i in range(half):
        y = (half - 1 - i) * pitch_mm - (half - 1) * pitch_mm / 2.0
        pads.append(_thru_pad(str(half + i + 1), row_spacing_mm / 2.0, y, pad_diam, drill_mm))

    body_w = row_spacing_mm + pad_diam + 0.5
    body_h = max((half - 1) * pitch_mm + pad_diam + 0.5, 3.0)
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + 0.5, LAYER_F_FAB),
    )
    if not lib_id:
        lib_id = f"Package_DIP:DIP-{pin_count}_W{row_spacing_mm:.2f}mm"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=LAYER_F_CU, pads=tuple(pads), graphics=graphics, texts=texts,
        attr="through_hole", models=models,
    )


def make_test_point(
    ref: str,
    value: str = "TestPoint",
    pad_size: float = 1.5,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """SMD test point footprint (single pad).

    Args:
        ref: Reference designator (e.g. "TP1").
        value: Component value string.
        pad_size: Pad width/height in mm (default 1.5).
        layer: Primary copper layer.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    pads = (_smd_pad("1", 0.0, 0.0, pad_size, pad_size, layer),)
    crtyd_size = pad_size + 2 * PCB_COURTYARD_CLEARANCE_MM
    graphics = _courtyard_rect(crtyd_size, crtyd_size)
    texts = (
        _ref_text(ref, -(crtyd_size / 2.0 + 0.5), LAYER_F_SILKSCREEN),
        _val_text(value, crtyd_size / 2.0 + 0.5, LAYER_F_FAB),
    )
    lib_id = f"TestPoint:TestPoint_Pad_{pad_size:.1f}x{pad_size:.1f}mm"
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=layer, pads=pads, graphics=graphics, texts=texts,
        attr="smd",
    )


def make_ws2812b(
    ref: str,
    value: str = "WS2812B",
    layer: str = LAYER_F_CU,
    size: str = "5050",
) -> Footprint:
    """WS2812B addressable RGB LED footprint (PLCC-4, 4 pads).

    Pin 1 = VDD, Pin 2 = DOUT, Pin 3 = GND, Pin 4 = DIN.

    Supported sizes:
        - ``"5050"``: 5.0×5.0 mm body (WS2812B standard)
        - ``"3535"``: 3.5×3.5 mm body (WS2812B-Mini)
        - ``"2020"``: 2.0×2.0 mm body (WS2812C-2020)

    Args:
        ref: Reference designator (e.g. "LED1").
        value: Component value string.
        layer: Primary copper layer.
        size: Package size code.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    if size == "2020":
        # WS2812C-2020: 2.0×2.0 mm body, 4 bottom pads
        pad_w = 0.7
        pad_h = 0.5
        x_pitch = 0.75
        y_pitch = 0.55
        body_w = 2.6
        body_h = 2.6
        lib_id = "LED_SMD:LED_WS2812B_PLCC4_2.0x2.0mm"
    elif size == "3535":
        # WS2812B-Mini: 3.5×3.5 mm body
        pad_w = 1.0
        pad_h = 0.8
        x_pitch = 1.65
        y_pitch = 1.05
        body_w = 4.0
        body_h = 4.0
        lib_id = "LED_SMD:LED_WS2812B_PLCC4_3.5x3.5mm_P2.45mm"
    else:
        # WS2812B standard 5050: 5.0×5.0 mm body
        pad_w = 1.5
        pad_h = 1.0
        x_pitch = 2.45
        y_pitch = 1.6
        body_w = 5.4
        body_h = 5.4
        lib_id = "LED_SMD:LED_WS2812B_PLCC4_5.0x5.0mm_P3.2mm"

    pads = (
        _smd_pad("1", -x_pitch, -y_pitch, pad_w, pad_h, layer),  # VDD
        _smd_pad("2", x_pitch, -y_pitch, pad_w, pad_h, layer),   # DOUT
        _smd_pad("3", x_pitch, y_pitch, pad_w, pad_h, layer),    # GND
        _smd_pad("4", -x_pitch, y_pitch, pad_w, pad_h, layer),   # DIN
    )
    graphics = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + 1.0), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + 1.0, LAYER_F_FAB),
    )
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=layer, pads=pads, graphics=graphics, texts=texts,
        attr="smd", models=models,
    )


def make_microsd_slot(
    ref: str,
    value: str = "MicroSD",
) -> Footprint:
    """Micro SD card push-push slot footprint.

    Generates a simplified footprint with 8 signal pads + 2 shield/detect pads.
    Compatible with Hirose DM3AT and similar push-push slots.

    Args:
        ref: Reference designator (e.g. "J16").
        value: Component value string.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    # 8 signal pads (1.1mm pitch) + 2 shield pads
    signal_pitch = 1.1
    pad_w = 0.7
    pad_h = 1.8
    signal_y = -5.5  # signal pads at front edge
    pads: list[Pad] = []
    for i in range(8):
        x = (i - 3.5) * signal_pitch
        pads.append(_smd_pad(str(i + 1), x, signal_y, pad_w, pad_h, LAYER_F_CU))
    # Shield / card detect pads (larger, on sides)
    shield_pad_w = 1.2
    shield_pad_h = 2.0
    pads.append(_smd_pad("9", -7.0, -1.5, shield_pad_w, shield_pad_h, LAYER_F_CU))
    pads.append(_smd_pad("10", 7.0, -1.5, shield_pad_w, shield_pad_h, LAYER_F_CU))

    body_w = 15.0
    body_h = 14.5
    graphics = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + 1.0), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + 1.0, LAYER_F_FAB),
    )
    lib_id = "Connector_Card:microSD_HC_Hirose_DM3AT-SF-PEJM5"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=LAYER_F_CU, pads=tuple(pads), graphics=graphics, texts=texts,
        attr="smd", models=models,
    )


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------


def footprint_for_component(
    ref: str,
    value: str,
    footprint_id: str,
    lcsc: str | None = None,
    layer: str = LAYER_F_CU,
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
        layer: Copper layer for the footprint (default ``F.Cu``).

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("footprint_for_component ref=%s id=%s layer=%s", ref, footprint_id, layer)
    fid = footprint_id.strip()
    upper = fid.upper()

    fp: Footprint

    # WS2812B / addressable LED (before generic LED_ check)
    if "WS2812" in upper:
        ws_size = "5050"
        if "2020" in fid:
            ws_size = "2020"
        elif "3535" in fid:
            ws_size = "3535"
        fp = make_ws2812b(ref, value, layer=layer, size=ws_size)

    # LED packages
    elif upper.startswith("LED_"):
        pkg = fid[4:].upper()
        pkg_norm = pkg if pkg in _SMD_RC_DIMS else "0805"
        fp = make_smd_led(ref, value, package=pkg_norm)

    # Resistor / capacitor (R_pkg or C_pkg)
    elif upper.startswith(("R_", "C_")):
        pkg = fid[2:].upper()
        pkg_norm = pkg if pkg in _SMD_RC_DIMS else "0805"
        fp = make_smd_resistor_capacitor(ref, value, package=pkg_norm)

    # SOD-323 diodes
    elif upper.startswith("SOD-323"):
        fp = make_sod323(ref, value, layer=layer)

    # SOD-123 diodes
    elif upper.startswith("SOD-123"):
        fp = make_sod123(ref, value)

    # Inductors (L_xxxx)
    elif upper.startswith("L_"):
        pkg = fid[2:].upper()
        fp = make_inductor_smd(ref, value, package=pkg)

    # SOT-223 (special case before SOT-23 prefix check)
    elif upper == "SOT-223":
        fp = make_sot23(ref, value, variant="SOT-23")

    # SOT-23 / TSOT-23 family
    elif upper.startswith(("SOT-23", "TSOT-23")):
        # Normalize TSOT-23-6 → SOT-23-6
        variant = fid.upper().replace("TSOT-", "SOT-")
        if variant not in _SOT23_VARIANTS:
            variant = "SOT-23"
        fp = make_sot23(ref, value, variant=variant)

    # USB-C
    elif upper.startswith(("USB-C", "USB_C")):
        fp = make_usbc_connector(ref, value)

    # RJ45
    elif upper.startswith("RJ45"):
        fp = make_rj45(ref, value)

    # Pin headers and sockets (including Conn_NxM patterns)
    elif upper.startswith(("PINHEADER", "PINSOCKET", "CONN_")):
        pin_count = _parse_pin_count(fid)
        pitch = _parse_pitch(fid)
        # Detect dual-row from "2x" in the footprint ID
        rows = 2 if "2X" in upper or "2x" in fid else 1
        # Detect RPi-related footprint IDs and pass row_swap=True
        rpi_swap = "2X20" in upper or "RPI" in upper or "RASPBERRY" in upper
        fp = make_pin_header_socket(
            ref, value, pin_count, pitch, rows, lib_id=fid, row_swap=rpi_swap,
            layer=layer,
        )

    # Terminal blocks
    elif upper.startswith("TERMINALBLOCK") or upper.startswith("TB_"):
        pin_count = _parse_pin_count(fid)
        pitch = _parse_pitch(fid)
        fp = make_terminal_block(ref, value, pin_count, pitch)

    # DIP switches (SW_DIP* or DIP_Switch*)
    elif upper.startswith(("SW_DIP", "DIP_SWITCH")):
        import re as _re
        # Extract position count from "SW_DIP_x01" style names
        pos_m = _re.search(r"x(\d+)", fid)
        if pos_m:
            pin_count = int(pos_m.group(1)) * 2  # 2 pins per position
        else:
            pin_count = _parse_pin_count(fid)
            if pin_count < 2:
                pin_count = 8
        fp = make_dip_switch(ref, value, pin_count)

    # Relay (SPDT)
    elif upper.startswith("RELAY"):
        fp = make_relay_spdt(ref, value)

    # ESP32 modules
    elif "ESP32" in upper or "WROOM" in upper:
        fp = make_esp32_wroom(ref, value)

    # SMD tactile switches (SW_SPST_SMD_*, SW_Push_SMD_*)
    elif upper.startswith("SW_") and "SMD" in upper:
        import re as _re
        size_m = _re.search(r"(\d+\.?\d*)x(\d+\.?\d*)", fid)
        w = float(size_m.group(1)) if size_m else 3.0
        h = float(size_m.group(2)) if size_m else 2.5
        fp = make_smd_tact_switch(ref, value, width_mm=w, height_mm=h)

    # Through-hole tactile switches (SW_SPST, SW_Push, etc.)
    elif upper.startswith("SW_"):
        import re as _re
        size_m = _re.search(r"(\d+\.?\d*)x(\d+\.?\d*)", fid)
        size_mm = float(size_m.group(1)) if size_m else 4.5
        fp = make_tact_switch(ref, value, size_mm=size_mm)

    # Crystal oscillators
    elif upper.startswith("CRYSTAL"):
        import re as _re
        dim_m = _re.search(r"(\d+\.?\d*)x(\d+\.?\d*)", fid)
        w = float(dim_m.group(1)) if dim_m else 3.2
        h = float(dim_m.group(2)) if dim_m else 1.5
        fp = make_crystal_smd(ref, value, size_w=w, size_h=h)

    # Test points (TP_*, TestPoint_*)
    elif upper.startswith(("TP_", "TESTPOINT")):
        import re as _re
        size_m = _re.search(r"(\d+\.?\d*)x(\d+\.?\d*)", fid)
        pad_size = float(size_m.group(1)) if size_m else 1.5
        fp = make_test_point(ref, value, pad_size=pad_size, layer=layer)

    # Micro SD card slot
    elif upper.startswith(("TF_PUSH", "MICROSD", "MICRO_SD")):
        fp = make_microsd_slot(ref, value)

    # Through-hole DIP packages (DIP-N, but not DIP_SWITCH)
    elif upper.startswith("DIP-") or (upper.startswith("DIP_") and "SWITCH" not in upper):
        pin_count = _parse_pin_count(fid)
        if pin_count < 2:
            pin_count = 4
        fp = make_dip_package(ref, value, pin_count)

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
            # Default pitch by package family: SOP/SOIC = 1.27mm, others = 0.5mm
            if upper.startswith(("SOP", "SOIC")):
                pitch = 1.27
            else:
                pitch = 0.5
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
            models=fp.models,
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
# Footprint bounding box + origin detection
# ---------------------------------------------------------------------------


def compute_footprint_bbox(fp: Footprint) -> FootprintBBox:
    """Compute the axis-aligned bounding box of a footprint from pad geometry.

    Uses the actual pad positions and sizes to determine the physical extent
    relative to the footprint origin.  Falls back to
    :func:`estimate_footprint_size` when no pad data is available.

    Args:
        fp: A :class:`Footprint` with pad and/or graphic data.

    Returns:
        A :class:`FootprintBBox` relative to the footprint origin.
    """
    if not fp.pads:
        # Fallback: estimate from footprint ID and center the box
        w, h = estimate_footprint_size(fp.lib_id)
        return FootprintBBox(
            min_x=-w / 2.0,
            min_y=-h / 2.0,
            max_x=w / 2.0,
            max_y=h / 2.0,
        )

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for pad in fp.pads:
        half_x = pad.size_x / 2.0
        half_y = pad.size_y / 2.0
        px, py = pad.position.x, pad.position.y
        min_x = min(min_x, px - half_x)
        min_y = min(min_y, py - half_y)
        max_x = max(max_x, px + half_x)
        max_y = max(max_y, py + half_y)

    # Add courtyard clearance
    min_x -= PCB_COURTYARD_CLEARANCE_MM
    min_y -= PCB_COURTYARD_CLEARANCE_MM
    max_x += PCB_COURTYARD_CLEARANCE_MM
    max_y += PCB_COURTYARD_CLEARANCE_MM

    return FootprintBBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)


# ---------------------------------------------------------------------------
# Body extension factors — how much the physical component body extends
# beyond the pad field on each side, by package type.
# ---------------------------------------------------------------------------

_BODY_EXTENSION: dict[str, tuple[float, float]] = {
    # (extra_width_per_side, extra_height_per_side)
    "module": (0.5, 2.0),       # ESP32/W5500 — body extends ~2mm beyond pad field per side
    "qfn": (0.25, 0.25),        # QFN/QFP/BGA — body ≈ pad field
    "qfp": (0.25, 0.25),
    "bga": (0.25, 0.25),
    "sot": (0.75, 0.75),        # SOT-23/223 — body wider than pads
    "passive": (0.25, 0.25),    # 0402/0603/0805 — body fits between pads
    "terminal_block": (2.0, 4.5),  # Screw terminals — body extends ~4.5mm above/below pad line
    "connector": (0.5, 0.5),    # Generic THT connectors (pin headers, etc.)
    "relay": (1.0, 1.0),        # Relay modules
    "switch": (1.0, 1.0),       # Tactile switches
    "default": (0.5, 0.5),      # Catch-all
}

_COURTYARD_CLEARANCE: float = 0.25  # KiCad standard courtyard clearance per side


def _classify_package(fp: Footprint) -> str:
    """Classify a footprint's package type for body extension lookup."""
    lib_upper = fp.lib_id.upper()
    # Module detection (must be before IC detection)
    if any(kw in lib_upper for kw in ("WROOM", "W5500", "LAN8720", "ESP32", "MODULE")):
        return "module"
    if any(kw in lib_upper for kw in ("QFN", "DFN")):
        return "qfn"
    if any(kw in lib_upper for kw in ("QFP", "LQFP", "TQFP")):
        return "qfp"
    if "BGA" in lib_upper:
        return "bga"
    if any(kw in lib_upper for kw in ("SOT-23", "SOT-223", "TSOT", "SOT-")):
        return "sot"
    if any(kw in lib_upper for kw in ("0402", "0603", "0805", "1206", "1210",
                                       "R_", "C_", "L_", "LED_")):
        return "passive"
    if any(kw in lib_upper for kw in ("TERMINALBLOCK", "TB_", "MKDS")):
        return "terminal_block"
    if any(kw in lib_upper for kw in ("PINHEADER", "PINSOCKET", "CONN_",
                                       "MOLEX", "RJ45", "USB")):
        return "connector"
    if "RELAY" in lib_upper:
        return "relay"
    if "SW_" in lib_upper:
        return "switch"
    return "default"


def estimate_courtyard_mm(fp: Footprint) -> tuple[float, float]:
    """Estimate courtyard (width, height) from footprint geometry.

    Uses pad edges (not centers) plus a body extension factor that accounts
    for the physical component body extending beyond the pad field.  The
    body extension varies by package type — modules like ESP32 have large
    antenna/shield areas, while passives are close to pad extent.

    Args:
        fp: A :class:`Footprint` with pad data.

    Returns:
        ``(width_mm, height_mm)`` courtyard estimate.
    """
    if not fp.pads:
        return estimate_footprint_size(fp.lib_id)

    # Compute pad-edge extents
    min_x = min(p.position.x - p.size_x / 2.0 for p in fp.pads)
    max_x = max(p.position.x + p.size_x / 2.0 for p in fp.pads)
    min_y = min(p.position.y - p.size_y / 2.0 for p in fp.pads)
    max_y = max(p.position.y + p.size_y / 2.0 for p in fp.pads)

    pad_w = max_x - min_x
    pad_h = max_y - min_y

    # Apply body extension per package type
    pkg = _classify_package(fp)
    ext_w, ext_h = _BODY_EXTENSION.get(pkg, _BODY_EXTENSION["default"])

    w = pad_w + 2.0 * ext_w + 2.0 * _COURTYARD_CLEARANCE
    h = pad_h + 2.0 * ext_h + 2.0 * _COURTYARD_CLEARANCE

    return (max(w, 1.0), max(h, 1.0))


def detect_origin_type(fp: Footprint) -> OriginType:
    """Detect whether a footprint uses center or pin-1 origin convention.

    THT connectors (pin headers, terminal blocks) typically have the origin
    at pad 1.  SMD parts have the origin at the geometric center.

    Args:
        fp: A :class:`Footprint` to inspect.

    Returns:
        :attr:`OriginType.PIN1` if pad "1" is near ``(0, 0)`` and the
        geometric center is significantly offset; :attr:`OriginType.CENTER`
        otherwise.
    """
    if not fp.pads:
        return OriginType.CENTER

    # Find pad "1"
    pad1 = None
    for pad in fp.pads:
        if pad.number == "1":
            pad1 = pad
            break

    if pad1 is None:
        return OriginType.CENTER

    # Check if pad 1 is near the origin
    pad1_dist = math.hypot(pad1.position.x, pad1.position.y)

    # Compute geometric center of all pads
    cx = sum(p.position.x for p in fp.pads) / len(fp.pads)
    cy = sum(p.position.y for p in fp.pads) / len(fp.pads)
    center_dist = math.hypot(cx, cy)

    # PIN1 origin: pad 1 is near (0,0) AND center is significantly offset
    if pad1_dist < 1.0 and center_dist > 2.0:
        return OriginType.PIN1

    return OriginType.CENTER


# ---------------------------------------------------------------------------
# 3D model alignment verification
# ---------------------------------------------------------------------------

# Valid Z-rotation deltas for 3D models (multiples of 90°)
_VALID_Z_ROTATIONS: frozenset[float] = frozenset({0.0, 90.0, 180.0, 270.0})


def validate_3d_model_orientation(fp: Footprint) -> tuple[str, ...]:
    """Check 3D model rotation/offset vs package conventions.

    Returns warning strings for:
    - Missing 3D model on non-trivial footprints
    - Z-rotation not a multiple of 90°
    - Known package conventions violated (terminal blocks need 180° Z,
      DIP switches need 90° Z)

    Args:
        fp: A placed :class:`Footprint` to validate.

    Returns:
        Tuple of warning description strings (empty if all OK).
    """
    warnings: list[str] = []
    upper = fp.lib_id.upper()

    # Skip trivial footprints (mounting holes, test points)
    if any(t in upper for t in ("MOUNTING", "TESTPOINT", "FIDUCIAL")):
        return ()

    if not fp.models:
        if fp.pads:
            warnings.append(
                f"{fp.ref}: missing 3D model for footprint {fp.lib_id}"
            )
        return tuple(warnings)

    for model in fp.models:
        z_rot = model.rotate[2] % 360.0
        # Round to avoid floating point issues
        z_rot_rounded = round(z_rot, 1)

        # Check Z-rotation is a valid multiple of 90°
        if z_rot_rounded not in _VALID_Z_ROTATIONS:
            warnings.append(
                f"{fp.ref}: 3D model Z-rotation {z_rot_rounded}° is not a"
                " multiple of 90°"
            )

        # Terminal blocks should have 180° Z rotation
        if ("TERMINALBLOCK" in upper or "MKDS" in upper) and z_rot_rounded != 180.0:
            warnings.append(
                f"{fp.ref}: terminal block 3D model should have"
                f" Z-rotation 180° (has {z_rot_rounded}°)"
            )

        # DIP switches should have 90° Z rotation
        if ("SW_DIP" in upper) and z_rot_rounded != 90.0:
            warnings.append(
                f"{fp.ref}: DIP switch 3D model should have"
                f" Z-rotation 90° (has {z_rot_rounded}°)"
            )

    return tuple(warnings)


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

    # SOD-123 diodes
    if upper.startswith("SOD-123"):
        return (3.5, 2.5)

    # Inductors
    if upper.startswith("L_"):
        pkg = fid[2:].upper()
        if pkg in _SMD_RC_DIMS:
            _, _, pitch, body_w, body_h = _SMD_RC_DIMS[pkg]
            return (body_w + 0.5, body_h + 0.5)
        return (4.0, 3.0)

    # SOT-23 / TSOT-23 family
    if upper.startswith(("SOT-23", "TSOT-23")):
        variant = fid.upper().replace("TSOT-", "SOT-")
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

    # RJ45 (HR911105A: ~16.5mm wide, ~22mm deep)
    if upper.startswith("RJ45"):
        return (17.0, 23.0)

    # Pin headers/sockets/connectors (Conn_01x02, Conn_02x20_Stacking, etc.)
    # Layout convention: pins run vertically (Y axis), dual-row spans X axis.
    # So "2x20" means 2 columns (X) of 20 rows (Y).
    if upper.startswith(("PINHEADER", "PINSOCKET", "CONN_")):
        pin_count = _parse_pin_count(fid)
        pitch = _parse_pitch(fid)
        num_cols = 2 if "2X" in upper or "2x" in fid else 1
        pins_per_col = pin_count // max(num_cols, 1)
        # Pad width (1.7mm typical for 2.54mm pitch THT) + courtyard margin
        pad_margin = 3.5 if pitch >= 2.0 else 2.5
        w = (num_cols - 1) * pitch + pad_margin  # across columns (X)
        h = (pins_per_col - 1) * pitch + pad_margin  # along columns (Y)
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

    # Relays
    if upper.startswith("RELAY"):
        return (20.0, 16.5)

    # ESP32 modules
    if "ESP32" in upper or "WROOM" in upper:
        return (18.5, 26.0)

    # Tactile switches
    if upper.startswith("SW_"):
        import re as _re
        size_m = _re.search(r"(\d+\.?\d*)x(\d+\.?\d*)", fid)
        if size_m:
            return (float(size_m.group(1)) + 2.0, float(size_m.group(2)) + 2.0)
        return (7.0, 7.0)

    # Crystal oscillators
    if upper.startswith("CRYSTAL"):
        import re as _re
        dim_m = _re.search(r"(\d+\.?\d*)x(\d+\.?\d*)", fid)
        if dim_m:
            return (float(dim_m.group(1)) + 0.5, float(dim_m.group(2)) + 0.5)
        return (4.0, 2.0)

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
