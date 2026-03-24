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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from kicad_pipeline.constants import (
    KICAD_3DMODEL_VAR,
    LAYER_B_COURTYARD,
    LAYER_B_CU,
    LAYER_B_FAB,
    LAYER_B_MASK,
    LAYER_B_PASTE,
    LAYER_B_SILKSCREEN,
    LAYER_EDGE_CUTS,
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
    FootprintArc,
    FootprintBBox,
    FootprintCircle,
    FootprintKeepout,
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


def _model_pin_header_socket(
    name: str, upper: str, layer: str,
) -> Footprint3DModel | None:
    """Return 3D model for pin header/socket footprints."""
    if "PINHEADER" not in upper and "PINSOCKET" not in upper:
        return None
    if layer == LAYER_B_CU or "PINSOCKET" in upper:
        dir_name = "Connector_PinSocket_2.54mm.3dshapes"
        model_name = name.replace("PinHeader", "PinSocket")
    else:
        dir_name = "Connector_PinHeader_2.54mm.3dshapes"
        model_name = name
    if not any(s in model_name for s in ("_Vertical", "_Horizontal", "_SMD")):
        model_name += "_Vertical"
    path = f"{KICAD_3DMODEL_VAR}/{dir_name}/{model_name}.step"
    return Footprint3DModel(path=path)


def _model_ic_package(
    name: str, upper: str,
) -> Footprint3DModel | None:
    """Return 3D model for IC packages (QFP, SO families)."""
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
    return None


def _model_terminal_block(
    lib_id: str, upper: str,
) -> Footprint3DModel | None:
    """Return 3D model for terminal block footprints."""
    if "TERMINALBLOCK" not in upper and "MKDS" not in upper:
        return None
    import re as _re
    pin_count = 2
    nx_match = _re.search(r"1x(\d+)", lib_id)
    if nx_match:
        pin_count = int(nx_match.group(1))
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
    offset_x = (pin_count - 1) * pitch
    return Footprint3DModel(
        path=path,
        offset=(offset_x, 0.0, 0.0),
        rotate=(0.0, 0.0, 180.0),
    )


def _model_esp32(
    name: str, upper: str,
) -> Footprint3DModel | None:
    """Return 3D model for ESP32/RF modules."""
    if "ESP32" not in upper and "WROOM" not in upper:
        return None
    model_name = name.split(":")[-1] if ":" in name else name
    path = f"{KICAD_3DMODEL_VAR}/RF_Module.3dshapes/{model_name}.step"
    return Footprint3DModel(path=path)


def _model_switch(
    name: str, upper: str, lib_id: str,
) -> Footprint3DModel | None:
    """Return 3D model for DIP switches and tactile switches."""
    # DIP switches — must check BEFORE generic SW_ match
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
    return None


def _model_connector(
    name: str, upper: str, layer: str,
) -> Footprint3DModel | None:
    """Return 3D model for RJ45, USB-C, Conn_01x, MicroSD connectors."""
    if "RJ45" in upper:
        path = (
            f"{KICAD_3DMODEL_VAR}/Connector_RJ.3dshapes/"
            "RJ45_Amphenol_RJHSE538X.step"
        )
        return Footprint3DModel(path=path)

    if upper.startswith(("USB-C", "USB_C")):
        path = (
            f"{KICAD_3DMODEL_VAR}/Connector_USB.3dshapes/"
            "USB_C_Receptacle_GCT_USB4105-xx-A_16P_TopMnt_Horizontal.step"
        )
        return Footprint3DModel(path=path)

    if upper.startswith(("TF_PUSH", "MICROSD", "MICRO_SD")):
        path = (
            f"{KICAD_3DMODEL_VAR}/Connector_Card.3dshapes/"
            "microSD_HC_Hirose_DM3AT-SF-PEJM5.step"
        )
        return Footprint3DModel(path=path)

    if upper.startswith("CONN_01X"):
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

    return None


def _model_optocoupler(
    name: str, upper: str,
) -> Footprint3DModel | None:
    """Return 3D model for optocoupler packages."""
    if "PC817" not in upper and "EL817" not in upper and "OPTO" not in upper:
        return None
    if "SOP" in upper or "SMD" in upper or "MINI" in upper:
        path = (
            f"{KICAD_3DMODEL_VAR}/Package_SO.3dshapes/"
            "SOP-4_3.8x4.1mm_P2.54mm.step"
        )
    else:
        path = (
            f"{KICAD_3DMODEL_VAR}/Package_DIP.3dshapes/"
            "DIP-4_W7.62mm.step"
        )
    return Footprint3DModel(path=path)


def _model_ws2812(upper: str) -> Footprint3DModel | None:
    """Return 3D model for WS2812B addressable LEDs."""
    if "WS2812" not in upper:
        return None
    if (
        "2.0X2.0" in upper or "2020" in upper
        or "3.5X3.5" in upper or "3535" in upper or "P2.45" in upper
    ):
        step = "LED_WS2812B-Mini_PLCC4_3.5x3.5mm.step"
    elif "PLCC6" in upper:
        step = "LED_WS2812_PLCC6_5.0x5.0mm_P1.6mm.step"
    else:
        step = "LED_WS2812B_PLCC4_5.0x5.0mm_P3.2mm.step"
    path = f"{KICAD_3DMODEL_VAR}/LED_SMD.3dshapes/{step}"
    return Footprint3DModel(path=path)


def _model_sanyou_relay(
    _name: str, upper: str, _lib_id: str, _layer: str,
) -> Footprint3DModel | None:
    """Match Sanyou-specific relay 3D model."""
    if "RELAY" in upper and "SANYOU" in upper:
        path = (
            f"{KICAD_3DMODEL_VAR}/Relay_THT.3dshapes/"
            "Relay_SPDT_SANYOU_SRD_Series_Form_C.step"
        )
        return Footprint3DModel(path=path)
    return None


def _model_dip_package(
    name: str, upper: str, _lib_id: str, _layer: str,
) -> Footprint3DModel | None:
    """Match DIP package 3D model."""
    if upper.startswith("DIP-"):
        path = f"{KICAD_3DMODEL_VAR}/Package_DIP.3dshapes/{name}.step"
        return Footprint3DModel(path=path)
    return None


def _model_generic_relay(
    _name: str, upper: str, _lib_id: str, _layer: str,
) -> Footprint3DModel | None:
    """Match generic relay SPDT 3D model (non-Sanyou)."""
    if "RELAY" in upper and "SPDT" in upper:
        path = (
            f"{KICAD_3DMODEL_VAR}/Relay_THT.3dshapes/"
            "Relay_SPDT_Omron_G5V-1.step"
        )
        return Footprint3DModel(path=path)
    return None


def _model_static_pattern(
    _name: str, upper: str, _lib_id: str, _layer: str,
) -> Footprint3DModel | None:
    """Match passives/transistors/diodes/inductors/crystals via static pattern map."""
    for pattern, directory, model_file in _3D_MODEL_MAP:
        if pattern.upper() in upper:
            path = f"{KICAD_3DMODEL_VAR}/{directory}/{model_file}"
            return Footprint3DModel(path=path)
    return None


# Ordered dispatch pipeline for 3D model resolution.
# Each handler receives (name, upper, lib_id, layer) and returns model or None.
# Sanyou relay must precede generic relay; DIP package must precede static pattern map.
_3D_MODEL_DISPATCH: list[
    Callable[[str, str, str, str], Footprint3DModel | None]
] = [
    lambda name, upper, lib_id, layer: _model_pin_header_socket(name, upper, layer),
    lambda name, upper, lib_id, layer: _model_ic_package(name, upper),
    lambda name, upper, lib_id, layer: _model_terminal_block(lib_id, upper),
    lambda name, upper, lib_id, layer: _model_esp32(name, upper),
    _model_sanyou_relay,
    lambda name, upper, lib_id, layer: _model_switch(name, upper, lib_id),
    lambda name, upper, lib_id, layer: _model_connector(name, upper, layer),
    lambda name, upper, lib_id, layer: _model_optocoupler(name, upper),
    _model_dip_package,
    lambda name, upper, lib_id, layer: _model_ws2812(upper),
    _model_generic_relay,
    _model_static_pattern,
]


def _model_for_package(lib_id: str, layer: str = LAYER_F_CU) -> Footprint3DModel | None:
    """Determine the 3D model path for a given KiCad lib_id.

    Args:
        lib_id: KiCad library identifier (e.g. ``"Resistor_SMD:R_0805_2012Metric"``).
        layer: Component layer — B.Cu connectors use PinSocket models.

    Returns:
        A :class:`Footprint3DModel` or ``None`` if no mapping found.
    """
    name = lib_id.split(":")[-1] if ":" in lib_id else lib_id
    upper = name.upper()

    for handler in _3D_MODEL_DISPATCH:
        result = handler(name, upper, lib_id, layer)
        if result is not None:
            return result

    _log.debug("No 3D model mapping for lib_id=%r (KI-017)", lib_id)
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
    (-3.5, 2.5, 1.6, 1.6, "A1"),     # GND (A1)
    (-2.0, 2.5, 0.6, 1.6, "A5"),     # CC1 (A5)
    (-1.0, 2.5, 0.6, 1.6, "A7"),     # D- (A7)
    (1.0, 2.5, 0.6, 1.6, "A6"),      # D+ (A6)
    (2.0, 2.5, 0.6, 1.6, "B5"),      # CC2 (B5)
    (3.5, 2.5, 1.6, 1.6, "A4"),      # VBUS (A4)
    (-1.0, -2.5, 0.6, 1.6, "B7"),    # D-_B (B7)
    (1.0, -2.5, 0.6, 1.6, "B6"),     # D+_B (B6)
    (0.0, -3.5, 2.0, 1.0, "S1"),     # Shield (S1)
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
# Footprint dimension constants (extracted from inline magic numbers)
# ---------------------------------------------------------------------------

# SOD-123 diode dimensions (mm)
_SOD123_BODY_W: float = 2.68
_SOD123_BODY_H: float = 1.65
_SOD123_PAD_W: float = 0.91
_SOD123_PAD_H: float = 1.22
_SOD123_PITCH: float = 2.2

# SOD-323 diode dimensions (mm)
_SOD323_BODY_W: float = 1.7
_SOD323_BODY_H: float = 1.25
_SOD323_PAD_W: float = 0.6
_SOD323_PAD_H: float = 0.55
_SOD323_PITCH: float = 2.1

# ESP32-S3-WROOM-1 module dimensions (mm)
_ESP32_BODY_W: float = 18.0
_ESP32_BODY_H: float = 25.5
_ESP32_PAD_W: float = 0.9
_ESP32_PAD_H: float = 1.2
_ESP32_PITCH: float = 1.27
_ESP32_SIDE_PINS: int = 14
_ESP32_BOTTOM_PINS: int = 12
_ESP32_TOP_MARGIN: float = 2.5
_ESP32_GND_PAD_SIZE: float = 6.7

# ESP32-S3-WROOM-1 pin names (datasheet Table 3-1, top view, 41 pads).
# Index 0 = pad 1, index 40 = pad 41.
_ESP32_PIN_NAMES: tuple[str, ...] = (
    # Left column (pins 1-14): top to bottom
    "GND", "3V3", "EN", "IO4", "IO5", "IO6", "IO7",
    "IO15", "IO16", "IO17", "IO18", "IO8", "IO19", "IO20",
    # Bottom row (pins 15-26): left to right
    "IO3", "IO46", "IO9", "IO10", "IO11", "IO12",
    "IO13", "IO14", "IO21", "IO47", "IO48", "IO45",
    # Right column (pins 27-40): bottom to top
    "IO0", "IO35", "IO36", "IO37", "IO38", "IO39",
    "IO40", "IO41", "IO42", "RXD0", "TXD0", "IO2", "IO1", "GND",
    # Center pad (pin 41)
    "GND",
)

# Antenna keepout: top portion of the ESP32 module where no copper/components
# should be placed.  The antenna extends ~5mm from the top edge of the body.
_ESP32_ANTENNA_KEEPOUT_DEPTH_MM: float = 5.0

# Vertical offset for the GND pad centre.  The antenna occupies the top ~5mm
# of the 25.5mm body so the pad field (and GND pad) is shifted south by half
# the antenna depth to centre it on the active silicon area.
_ESP32_GND_PAD_Y_OFFSET: float = 2.5

# Crystal oscillator dimensions (mm)
_CRYSTAL_PAD_W: float = 1.2
_CRYSTAL_PAD_H: float = 1.0
_CRYSTAL_SHIELD_MIN_HEIGHT: float = 2.0

# Through-hole connector dimensions (mm)
_THT_CONNECTOR_DRILL: float = 1.0
_THT_CONNECTOR_PAD_DIAM: float = 1.7

# Terminal block dimensions (mm)
_TB_DRILL: float = 1.3
_TB_PAD_DIAM: float = 2.5
_TB_BODY_W_MARGIN: float = 4.0
_TB_BODY_H_MARGIN: float = 6.0

# DIP package dimensions (mm)
_DIP_DRILL: float = 0.8
_DIP_PAD_DIAM: float = 1.6
_DIP_ROW_SPACING: float = 7.62
_DIP_SWITCH_DRILL: float = 1.0
_DIP_SWITCH_PAD_DIAM: float = 1.7

# Relay SPDT pad dimensions (mm)
_RELAY_CONTACT_PAD_DIAM: float = 3.0
_RELAY_CONTACT_DRILL: float = 1.3
_RELAY_COIL_PAD_DIAM: float = 2.5
_RELAY_COIL_DRILL: float = 1.0
_RELAY_CUTOUT_WIDTH: float = 1.0
_RELAY_CUTOUT_CLEARANCE: float = 2.0

# Tact switch dimension tiers (mm)
_TACT_SMALL_HALF_X: float = 2.0
_TACT_SMALL_HALF_Y: float = 1.5
_TACT_SMALL_DRILL: float = 0.8
_TACT_SMALL_PAD_DIAM: float = 1.4
_TACT_MEDIUM_HALF_X: float = 2.75
_TACT_MEDIUM_HALF_Y: float = 2.0
_TACT_MEDIUM_DRILL: float = 0.9
_TACT_MEDIUM_PAD_DIAM: float = 1.6
_TACT_LARGE_HALF_X: float = 3.25
_TACT_LARGE_HALF_Y: float = 2.25
_TACT_LARGE_DRILL: float = 1.0
_TACT_LARGE_PAD_DIAM: float = 1.8

# SMD tact switch (XKB TS-1187A) dimensions (mm)
_SMD_TACT_PAD_W: float = 1.5
_SMD_TACT_PAD_H: float = 3.0
_SMD_TACT_PAD_X: float = 3.5
_SMD_TACT_PAD_Y: float = 2.5

# USB-C connector body dimensions (mm)
_USBC_BODY_W: float = 9.0
_USBC_BODY_H: float = 7.35

# RJ45 courtyard dimensions (mm)
_RJ45_COURTYARD_CX: float = 3.56
_RJ45_COURTYARD_CY: float = -0.125
_RJ45_COURTYARD_W: float = 19.56
_RJ45_COURTYARD_H: float = 16.75

# WS2812B LED size variants: (pad_w, pad_h, x_pitch, y_pitch, body_w, body_h)
_WS2812B_DIMS: dict[str, tuple[float, float, float, float, float, float]] = {
    "2020": (0.7, 0.5, 0.75, 0.55, 2.6, 2.6),
    "3535": (1.0, 0.8, 1.65, 1.05, 4.0, 4.0),
    "5050": (1.5, 1.0, 2.45, 1.6, 5.4, 5.4),
}

# Micro SD card slot dimensions (mm)
_MICROSD_SIGNAL_PITCH: float = 1.1
_MICROSD_PAD_W: float = 0.7
_MICROSD_PAD_H: float = 1.8
_MICROSD_SIGNAL_Y: float = -5.5
_MICROSD_SHIELD_PAD_W: float = 1.2
_MICROSD_SHIELD_PAD_H: float = 2.0
_MICROSD_BODY_W: float = 15.0
_MICROSD_BODY_H: float = 14.5

# Relay body outline coordinates (mm)
_RELAY_BODY_X_MIN: float = -1.4
_RELAY_BODY_X_MAX: float = 18.4
_RELAY_BODY_Y_MIN: float = -7.8
_RELAY_BODY_Y_MAX: float = 7.8

# Text offset from body edge (mm)
_TEXT_OFFSET_SMALL: float = 1.0
_TEXT_OFFSET_LARGE: float = 1.5

# Reference/value text margin beyond courtyard edge (mm)
_TEXT_MARGIN_MM: float = 0.5

# Body margin added to pad extent for body estimates (mm)
_BODY_MARGIN_MM: float = 0.5

# Silkscreen clearance threshold: compact packages skip silk marks (mm)
_COMPACT_PKG_THRESHOLD: float = 1.0

# Generic SMD IC constraints
_IC_PAD_W_MAX: float = 0.5
_IC_PAD_H_MAX: float = 1.5
_IC_COL_OFFSET: float = 1.5


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
    if body_h <= _COMPACT_PKG_THRESHOLD:
        graphics = _courtyard_rect(body_w, body_h)
    else:
        graphics = (
            *_courtyard_rect(body_w, body_h),
            *_silk_side_marks(body_w, body_h, pad_edge_x=pad_edge_x),
        )
    # Compact packages: ref on F.Fab to avoid silk-over-copper DRC
    ref_layer = LAYER_F_FAB if body_h <= _COMPACT_PKG_THRESHOLD else LAYER_F_SILKSCREEN
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM), ref_layer),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM, LAYER_F_FAB),
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
    body_w = _SOD123_BODY_W
    body_h = _SOD123_BODY_H
    pad_w = _SOD123_PAD_W
    pad_h = _SOD123_PAD_H
    pitch = _SOD123_PITCH
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
        _ref_text(ref, -(body_h / 2.0 + _TEXT_OFFSET_SMALL), LAYER_F_SILKSCREEN),
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
        _ref_text(ref, -(body_h / 2.0 + _TEXT_OFFSET_SMALL), LAYER_F_SILKSCREEN),
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
        half_x = _TACT_SMALL_HALF_X
        half_y = _TACT_SMALL_HALF_Y
        drill = _TACT_SMALL_DRILL
        pad_diam = _TACT_SMALL_PAD_DIAM
    elif size_mm <= 5.0:
        # Medium tact switch (e.g. 4.5x4.5mm)
        half_x = _TACT_MEDIUM_HALF_X
        half_y = _TACT_MEDIUM_HALF_Y
        drill = _TACT_MEDIUM_DRILL
        pad_diam = _TACT_MEDIUM_PAD_DIAM
    else:
        # Standard 6mm tact switch
        half_x = _TACT_LARGE_HALF_X
        half_y = _TACT_LARGE_HALF_Y
        drill = _TACT_LARGE_DRILL
        pad_diam = _TACT_LARGE_PAD_DIAM

    pads = (
        _thru_pad("1", -half_x, -half_y, pad_diam, drill),
        _thru_pad("1", half_x, -half_y, pad_diam, drill),
        _thru_pad("2", -half_x, half_y, pad_diam, drill),
        _thru_pad("2", half_x, half_y, pad_diam, drill),
    )
    body = size_mm
    graphics = _courtyard_rect(body + _TEXT_OFFSET_LARGE, body + _TEXT_OFFSET_LARGE)
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
    pad_size_x = _SMD_TACT_PAD_W
    pad_size_y = _SMD_TACT_PAD_H
    pad_x = _SMD_TACT_PAD_X   # horizontal center-to-center / 2
    pad_y = _SMD_TACT_PAD_Y   # vertical center-to-center / 2

    # 4 pads: two "1" pads (left+right, top row), two "2" pads (left+right, bottom)
    pads = (
        _smd_pad("1", -pad_x, -pad_y, pad_size_x, pad_size_y, LAYER_F_CU),
        _smd_pad("1", pad_x, -pad_y, pad_size_x, pad_size_y, LAYER_F_CU),
        _smd_pad("2", -pad_x, pad_y, pad_size_x, pad_size_y, LAYER_F_CU),
        _smd_pad("2", pad_x, pad_y, pad_size_x, pad_size_y, LAYER_F_CU),
    )
    court_w = pad_x * 2 + pad_size_x + _BODY_MARGIN_MM
    court_h = pad_y * 2 + pad_size_y + _BODY_MARGIN_MM
    graphics = _courtyard_rect(court_w, court_h)
    texts = (
        _ref_text(ref, -(court_h / 2.0 + _TEXT_MARGIN_MM), LAYER_F_SILKSCREEN),
        _val_text(value, court_h / 2.0 + _TEXT_MARGIN_MM, LAYER_F_FAB),
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
    pads = (
        _thru_pad("1", 0.0, 0.0, _RELAY_CONTACT_PAD_DIAM, _RELAY_CONTACT_DRILL),       # COM
        _thru_pad("2", 1.95, 6.05, _RELAY_COIL_PAD_DIAM, _RELAY_COIL_DRILL),            # Coil-
        _thru_pad("3", 14.15, 6.05, _RELAY_CONTACT_PAD_DIAM, _RELAY_CONTACT_DRILL),     # NO
        _thru_pad("4", 14.2, -6.0, _RELAY_CONTACT_PAD_DIAM, _RELAY_CONTACT_DRILL),      # NC
        _thru_pad("5", 1.95, -5.95, _RELAY_COIL_PAD_DIAM, _RELAY_COIL_DRILL),           # Coil+
    )
    body_w = _RELAY_BODY_X_MAX - _RELAY_BODY_X_MIN
    body_h = _RELAY_BODY_Y_MAX - _RELAY_BODY_Y_MIN
    cx = (_RELAY_BODY_X_MIN + _RELAY_BODY_X_MAX) / 2.0
    cy = (_RELAY_BODY_Y_MIN + _RELAY_BODY_Y_MAX) / 2.0
    hw = body_w / 2.0 + PCB_COURTYARD_CLEARANCE_MM
    hh = body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM
    # U-shaped isolation cutout around COM pin (pin 1 at 0,0).
    # Lives on Edge.Cuts inside the footprint so it moves with the relay.
    # The U opens toward positive X (toward relay body).
    _cutout_w = _RELAY_CUTOUT_WIDTH
    _cutout_clr = _RELAY_CUTOUT_CLEARANCE
    _com_pad_r = _RELAY_CONTACT_PAD_DIAM / 2.0
    _u_half = _com_pad_r + _cutout_clr   # half-height of the U
    _u_closed_x = -(_com_pad_r + _cutout_clr)  # closed end (left)
    _u_open_x = _com_pad_r + _cutout_clr       # open end (right, toward body)

    graphics = (
        # Courtyard
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
        # U-shaped Edge.Cuts isolation cutout around COM pin
        # Left vertical (closed end of U)
        FootprintLine(
            start=Point(_u_closed_x, -_u_half),
            end=Point(_u_closed_x, _u_half),
            layer=LAYER_EDGE_CUTS, width=_cutout_w,
        ),
        # Bottom horizontal arm
        FootprintLine(
            start=Point(_u_closed_x, -_u_half),
            end=Point(_u_open_x, -_u_half),
            layer=LAYER_EDGE_CUTS, width=_cutout_w,
        ),
        # Top horizontal arm
        FootprintLine(
            start=Point(_u_closed_x, _u_half),
            end=Point(_u_open_x, _u_half),
            layer=LAYER_EDGE_CUTS, width=_cutout_w,
        ),
    )
    texts = (
        _ref_text(ref, cy - (body_h / 2.0 + _TEXT_OFFSET_LARGE), LAYER_F_SILKSCREEN),
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

    Each pad carries the functional pin name (e.g. ``"GND"``, ``"IO4"``) so
    that KiCad displays meaningful labels instead of bare numbers.

    The footprint includes an antenna keepout zone covering the top ~5 mm of
    the module body (no copper on any layer) and a 3D model reference.

    Args:
        ref: Reference designator (e.g. "U3").
        value: Component value string.
        layer: Primary copper layer.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    body_w = _ESP32_BODY_W
    body_h = _ESP32_BODY_H
    pad_w = _ESP32_PAD_W
    pad_h = _ESP32_PAD_H
    pitch = _ESP32_PITCH

    # 41 pads: left(14) + bottom(12) + right(14) + center GND(1)
    pad_list: list[Pad] = []

    # Left and right columns share the same vertical span so pin 1
    # (top-left) aligns with pin 40 (top-right), and pin 14 (bottom-left)
    # aligns with pin 27 (bottom-right) — per datasheet Figure 3-1.
    n_side = _ESP32_SIDE_PINS
    col_top_y = -(body_h / 2.0) + _ESP32_TOP_MARGIN + pad_h / 2.0
    col_bot_y = col_top_y + (n_side - 1) * pitch

    # Left column: 14 pads (pins 1-14), top to bottom
    # Pin 1 (GND) at top-left near antenna, pin 14 (IO20) at bottom-left.
    left_x = -(body_w / 2.0 - pad_h / 2.0)
    for i in range(n_side):
        pad_list.append(_smd_pad(
            str(i + 1), left_x, col_top_y + i * pitch, pad_h, pad_w, layer,
        ))

    # Bottom row: 12 pads (pins 15-26), left to right
    # Pin 15 (IO3) at bottom-left, pin 26 (IO45) at bottom-right.
    bottom_y = body_h / 2.0 - pad_h / 2.0
    n_bottom = _ESP32_BOTTOM_PINS
    start_x = -((n_bottom - 1) * pitch) / 2.0
    for i in range(n_bottom):
        pad_list.append(_smd_pad(
            str(15 + i), start_x + i * pitch, bottom_y, pad_w, pad_h, layer,
        ))

    # Right column: 14 pads (pins 27-40), bottom to top
    # Pin 27 (IO0) at bottom-right, pin 40 (GND) at top-right.
    right_x = body_w / 2.0 - pad_h / 2.0
    for i in range(n_side):
        pad_list.append(_smd_pad(
            str(27 + i), right_x, col_bot_y - i * pitch, pad_h, pad_w, layer,
        ))

    # Center GND pad (large thermal pad underneath) — pin 41.
    # Offset south by _ESP32_GND_PAD_Y_OFFSET to centre it on the active
    # silicon area (the antenna occupies the top ~5mm of the module body).
    pad_list.append(Pad(
        number="41",
        pad_type="smd",
        shape="rect",
        position=Point(0.0, _ESP32_GND_PAD_Y_OFFSET),
        size_x=_ESP32_GND_PAD_SIZE,
        size_y=_ESP32_GND_PAD_SIZE,
        layers=(layer, LAYER_F_PASTE if layer == LAYER_F_CU else LAYER_B_PASTE,
                LAYER_F_MASK if layer == LAYER_F_CU else LAYER_B_MASK),
    ))

    # --- Pin name labels on the fab layer (Bug 1 fix) ---
    # KiCad pad numbers must stay numeric for netlist matching, so we add
    # small text labels next to each pad showing the functional pin name.
    fab_layer = LAYER_F_FAB if layer == LAYER_F_CU else LAYER_B_FAB
    pin_labels: list[FootprintText] = []
    _label_size = 0.5
    _label_offset = 1.6  # mm offset from pad centre toward body interior
    for i, pad in enumerate(pad_list[:-1]):  # skip pad 41 (GND label not needed)
        pin_name = _ESP32_PIN_NAMES[i]
        px, py = pad.position.x, pad.position.y
        # Shift label inward: left pads -> right, right pads -> left,
        # bottom pads -> up.
        if i < n_side:  # left column
            lx, ly = px + _label_offset, py
        elif i < n_side + n_bottom:  # bottom row
            lx, ly = px, py - _label_offset
        else:  # right column
            lx, ly = px - _label_offset, py
        pin_labels.append(FootprintText(
            text_type="user", text=pin_name,
            position=Point(lx, ly), layer=fab_layer,
            effects_size=_label_size,
        ))

    # --- Antenna keepout zone (footprint-level) ---
    # Covers the top _ESP32_ANTENNA_KEEPOUT_DEPTH_MM of the module body.
    # No copper allowed on any layer beneath the antenna.
    antenna_depth = _ESP32_ANTENNA_KEEPOUT_DEPTH_MM
    half_w = body_w / 2.0
    top_y = -(body_h / 2.0)
    keepout_bot_y = top_y + antenna_depth
    keepout_poly = (
        Point(-half_w, top_y),
        Point(half_w, top_y),
        Point(half_w, keepout_bot_y),
        Point(-half_w, keepout_bot_y),
    )
    antenna_keepout = FootprintKeepout(
        polygon=keepout_poly,
        layers=(LAYER_F_CU, LAYER_B_CU),
        no_copper=True,
        no_vias=True,
        no_tracks=True,
        tag="antenna",
    )

    graphics = _courtyard_rect(body_w, body_h)
    texts: tuple[FootprintText, ...] = (
        _ref_text(ref, -(body_h / 2.0 + _TEXT_OFFSET_LARGE), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + 1.5, LAYER_F_FAB),
        *pin_labels,
    )
    lib_id = "RF_Module:ESP32-S3-WROOM-1"

    # 3D model — always include a path even if the file is not present
    # locally; KiCad will display a placeholder.
    model = _model_for_package(lib_id)
    if model is None:
        model = Footprint3DModel(
            path=f"{KICAD_3DMODEL_VAR}/RF_Module.3dshapes/"
            "ESP32-S3-WROOM-1.step",
        )
    models = (model,)

    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=layer, pads=tuple(pad_list), graphics=graphics, texts=texts,
        attr="smd", models=models,
        fp_zones=(antenna_keepout,),
    )


def _enrich_esp32_footprint(fp: Footprint) -> Footprint:
    """Add pin name labels, antenna keepout, and 3D model to an ESP32 footprint.

    Works on footprints from any source (JLCPCB cache, parametric generator,
    or parsed .kicad_mod files).  Idempotent — skips enrichment that already
    exists.

    Args:
        fp: An ESP32/WROOM footprint (any origin).

    Returns:
        Enriched copy of *fp*.
    """
    layer = fp.layer
    fab_layer = LAYER_F_FAB if layer == LAYER_F_CU else LAYER_B_FAB

    # --- Bug 1: Pin name labels on the fab layer ---
    # Only add if not already present (idempotent).
    has_pin_labels = any(
        t.text_type == "user" and t.text in _ESP32_PIN_NAMES
        for t in fp.texts
    )
    extra_texts: list[FootprintText] = []
    if not has_pin_labels and len(fp.pads) >= 40:
        # Determine pad field bounding box to decide label offsets.
        _label_size = 0.5
        _label_offset = 1.6  # mm inward from pad centre
        all_x = [p.position.x for p in fp.pads[:40]]
        min_x, max_x = min(all_x), max(all_x)
        for idx, pad in enumerate(fp.pads[:40]):
            if idx >= len(_ESP32_PIN_NAMES):
                break
            pin_name = _ESP32_PIN_NAMES[idx]
            px, py = pad.position.x, pad.position.y
            # Classify pad side by position relative to centroid.
            # Left-side pads have small x, right-side have large x,
            # bottom pads have large y.
            if abs(px - min_x) < 1.0:  # left column
                lx, ly = px + _label_offset, py
            elif abs(px - max_x) < 1.0:  # right column
                lx, ly = px - _label_offset, py
            else:  # bottom row
                lx, ly = px, py - _label_offset
            extra_texts.append(FootprintText(
                text_type="user", text=pin_name,
                position=Point(lx, ly), layer=fab_layer,
                effects_size=_label_size,
            ))

    # --- Bug 3: Antenna keepout zone ---
    has_antenna_keepout = any(
        fz.tag == "antenna" for fz in fp.fp_zones
    )
    extra_zones: list[FootprintKeepout] = []
    if not has_antenna_keepout:
        # Estimate body bounds from pad field + margins.
        body_w = _ESP32_BODY_W
        body_h = _ESP32_BODY_H
        antenna_depth = _ESP32_ANTENNA_KEEPOUT_DEPTH_MM
        half_w = body_w / 2.0
        top_y = -(body_h / 2.0)
        keepout_bot_y = top_y + antenna_depth
        keepout_poly = (
            Point(-half_w, top_y),
            Point(half_w, top_y),
            Point(half_w, keepout_bot_y),
            Point(-half_w, keepout_bot_y),
        )
        extra_zones.append(FootprintKeepout(
            polygon=keepout_poly,
            layers=(LAYER_F_CU, LAYER_B_CU),
            no_copper=True,
            no_vias=True,
            no_tracks=True,
            tag="antenna",
        ))

    # --- Bug 4: 3D model ---
    has_model = len(fp.models) > 0
    models = fp.models
    if not has_model:
        model = _model_for_package("RF_Module:ESP32-S3-WROOM-1", layer)
        if model is None:
            model = Footprint3DModel(
                path=f"{KICAD_3DMODEL_VAR}/RF_Module.3dshapes/"
                "ESP32-S3-WROOM-1.step",
            )
        models = (model,)

    # Return enriched copy only if something changed.
    if not extra_texts and not extra_zones and has_model:
        return fp

    return Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
        position=fp.position, rotation=fp.rotation, layer=fp.layer,
        pads=fp.pads, graphics=fp.graphics,
        texts=(*fp.texts, *extra_texts),
        lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
        models=models,
        datasheet=fp.datasheet, description=fp.description,
        footprint_source=fp.footprint_source,
        mpn=fp.mpn, manufacturer=fp.manufacturer,
        fp_zones=(*fp.fp_zones, *extra_zones),
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
    pad_w = _CRYSTAL_PAD_W
    pad_h = _CRYSTAL_PAD_H
    pitch = size_w - pad_w + 0.4
    # Signal pads (1, 2) + shield/ground pads (3, 4) for 4-pin crystals.
    # 4-pin variants (3.2x2.5mm etc.) have ground pads at corners.
    has_shield = size_h >= _CRYSTAL_SHIELD_MIN_HEIGHT
    pads_list = [
        _smd_pad("1", -pitch / 2.0, 0.0, pad_w, pad_h, layer),
        _smd_pad("2", pitch / 2.0, 0.0, pad_w, pad_h, layer),
    ]
    if has_shield:
        shield_x = pitch / 2.0
        shield_y = size_h / 2.0 - pad_h / 2.0
        pads_list.append(_smd_pad("3", -shield_x, -shield_y, pad_w, pad_h, layer))
        pads_list.append(_smd_pad("4", shield_x, -shield_y, pad_w, pad_h, layer))
    pads = tuple(pads_list)
    graphics = _courtyard_rect(size_w, size_h)
    texts = (
        _ref_text(ref, -(size_h / 2.0 + _TEXT_OFFSET_SMALL), LAYER_F_SILKSCREEN),
        _val_text(value, size_h / 2.0 + 1.0, LAYER_F_FAB),
    )
    pin_suffix = "4Pin" if has_shield else "2Pin"
    lib_id = f"Crystal:Crystal_SMD_{size_w:.0f}215-{pin_suffix}_{size_w}x{size_h}mm"
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
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM, LAYER_F_FAB),
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
    body_h = pad_diameter_mm + _BODY_MARGIN_MM
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(
            ref,
            -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM),
            LAYER_F_SILKSCREEN,
        ),
        _val_text(
            value,
            body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM,
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
    thermal_pad: bool | None = None,
) -> Footprint:
    """Generate a generic SMD IC footprint (MSOP, TSSOP, SOIC, QFP, QFN, etc.).

    Pins are arranged in two rows: odd pins on the left, even on the right.
    QFN/DFN packages automatically get a center thermal/exposed pad (pad N+1).

    Args:
        ref: Reference designator.
        value: Component value string.
        pin_count: Total number of pins (signal pins only, thermal pad auto-added).
        pitch_mm: Pin pitch in mm.
        lib_id: KiCad library ID string (auto-generated if empty).
        thermal_pad: Whether to add a center thermal pad.  ``None`` (default)
            auto-detects from the lib_id (QFN/DFN packages get one).

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("make_generic_smd_ic ref=%s pins=%d pitch=%.2f", ref, pin_count, pitch_mm)
    half = pin_count // 2
    pad_w = min(pitch_mm * 0.6, _IC_PAD_W_MAX)
    pad_h = min(pitch_mm * 0.8, _IC_PAD_H_MAX)
    row_span = (half - 1) * pitch_mm
    col_pitch = row_span / 2.0 + _IC_COL_OFFSET

    pads: list[Pad] = []
    for i in range(half):
        # Left column: pins 1..half going downward
        y = -row_span / 2.0 + i * pitch_mm
        pads.append(_smd_pad(str(i + 1), -col_pitch, y, pad_h, pad_w, LAYER_F_CU))
    for i in range(half):
        # Right column: pins half+1..pin_count going upward
        y = row_span / 2.0 - i * pitch_mm
        pads.append(_smd_pad(str(half + i + 1), col_pitch, y, pad_h, pad_w, LAYER_F_CU))

    # Auto-detect thermal pad for QFN/DFN packages
    _upper = lib_id.upper()
    if thermal_pad is None:
        thermal_pad = any(kw in _upper for kw in ("QFN", "DFN"))
    if thermal_pad:
        # Center exposed pad, size ~60% of body
        ep_size = max(row_span * 0.5, 2.0)
        pads.append(_smd_pad(str(pin_count + 1), 0.0, 0.0, ep_size, ep_size, LAYER_F_CU))

    body_w = col_pitch * 2.0 + pad_h
    body_h = row_span + pad_w + _BODY_MARGIN_MM
    ic_pad_edge_x = col_pitch + pad_h / 2.0
    graphics: tuple[FootprintLine, ...] = (
        *_courtyard_rect(body_w, body_h),
        *_silk_side_marks(col_pitch * 1.6, body_h, pad_edge_x=ic_pad_edge_x),
    )
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM, LAYER_F_FAB),
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
    drill_mm = _THT_CONNECTOR_DRILL
    pad_diam = _THT_CONNECTOR_PAD_DIAM
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
    body_w = span_x + pad_diam + _TEXT_OFFSET_LARGE
    body_h = span_y + pad_diam + _TEXT_OFFSET_LARGE
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h, layer=crtyd_layer, cx=cx, cy=cy),)
    ref_y = cy - (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)
    val_y = cy + (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)
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
    drill_mm = _TB_DRILL
    pad_diam = _TB_PAD_DIAM

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
    body_w = span + pad_diam + _TB_BODY_W_MARGIN
    body_h = pad_diam + _TB_BODY_H_MARGIN
    # Center courtyard on the pad span
    cx = span / 2.0
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h, cx=cx),)
    ref_y = -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)
    val_y = body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM
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
    drill_mm = _DIP_SWITCH_DRILL
    pad_diam = _DIP_SWITCH_PAD_DIAM
    half = pin_count // 2
    row_pitch = _DIP_ROW_SPACING

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
    body_w = row_pitch + pad_diam + _BODY_MARGIN_MM
    body_h = span_y + pad_diam + _BODY_MARGIN_MM
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h, cx=cx, cy=cy),)
    texts = (
        FootprintText(text_type="reference", text=ref,
                      position=Point(cx, cy - (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)),
                      layer=LAYER_F_SILKSCREEN, effects_size=1.0),
        FootprintText(text_type="value", text=value,
                      position=Point(cx, cy + body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM),
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
        _smd_pad(pad_id, x, y, w, h, layer)
        for x, y, w, h, pad_id in _USBC_PADS
    )
    body_w = _USBC_BODY_W
    body_h = _USBC_BODY_H
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM, LAYER_F_FAB),
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
    cx = _RJ45_COURTYARD_CX
    cy = _RJ45_COURTYARD_CY
    body_w = _RJ45_COURTYARD_W
    body_h = _RJ45_COURTYARD_H
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h, cx=cx, cy=cy),)
    texts = (
        _ref_text(ref, cy - (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM), LAYER_F_SILKSCREEN),
        _val_text(value, cy + (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM), LAYER_F_FAB),
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
        _ref_text(ref, -(crtyd_size / 2.0 + _TEXT_MARGIN_MM), LAYER_F_FAB),
        _val_text("MountingHole", crtyd_size / 2.0 + _TEXT_MARGIN_MM, LAYER_F_FAB),
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
    body_w = _SOD323_BODY_W
    body_h = _SOD323_BODY_H
    pad_w = _SOD323_PAD_W
    pad_h = _SOD323_PAD_H
    pitch = _SOD323_PITCH
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
        _ref_text(ref, -(body_h / 2.0 + _TEXT_OFFSET_SMALL), LAYER_F_SILKSCREEN),
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
    drill_mm = _DIP_DRILL
    pad_diam = _DIP_PAD_DIAM
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

    body_w = row_spacing_mm + pad_diam + _BODY_MARGIN_MM
    body_h = max((half - 1) * pitch_mm + pad_diam + _BODY_MARGIN_MM, 3.0)
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM, LAYER_F_FAB),
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
        _ref_text(ref, -(crtyd_size / 2.0 + _TEXT_MARGIN_MM), LAYER_F_SILKSCREEN),
        _val_text(value, crtyd_size / 2.0 + _TEXT_MARGIN_MM, LAYER_F_FAB),
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
    dims = _WS2812B_DIMS.get(size, _WS2812B_DIMS["5050"])
    pad_w, pad_h, x_pitch, y_pitch, body_w, body_h = dims
    _WS2812B_LIB_IDS = {
        "2020": "LED_SMD:LED_WS2812B_PLCC4_2.0x2.0mm",
        "3535": "LED_SMD:LED_WS2812B_PLCC4_3.5x3.5mm_P2.45mm",
        "5050": "LED_SMD:LED_WS2812B_PLCC4_5.0x5.0mm_P3.2mm",
    }
    lib_id = _WS2812B_LIB_IDS.get(size, _WS2812B_LIB_IDS["5050"])

    pads = (
        _smd_pad("1", -x_pitch, -y_pitch, pad_w, pad_h, layer),  # VDD
        _smd_pad("2", x_pitch, -y_pitch, pad_w, pad_h, layer),   # DOUT
        _smd_pad("3", x_pitch, y_pitch, pad_w, pad_h, layer),    # GND
        _smd_pad("4", -x_pitch, y_pitch, pad_w, pad_h, layer),   # DIN
    )
    graphics = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + _TEXT_OFFSET_SMALL), LAYER_F_SILKSCREEN),
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
    # 8 signal pads + 2 shield pads
    pads: list[Pad] = []
    for i in range(8):
        x = (i - 3.5) * _MICROSD_SIGNAL_PITCH
        pads.append(_smd_pad(str(i + 1), x, _MICROSD_SIGNAL_Y, _MICROSD_PAD_W, _MICROSD_PAD_H, LAYER_F_CU))
    # Shield / card detect pads (larger, on sides)
    pads.append(_smd_pad("9", -7.0, -1.5, _MICROSD_SHIELD_PAD_W, _MICROSD_SHIELD_PAD_H, LAYER_F_CU))
    pads.append(_smd_pad("10", 7.0, -1.5, _MICROSD_SHIELD_PAD_W, _MICROSD_SHIELD_PAD_H, LAYER_F_CU))

    body_w = _MICROSD_BODY_W
    body_h = _MICROSD_BODY_H
    graphics = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + _TEXT_OFFSET_SMALL), LAYER_F_SILKSCREEN),
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
# Relay footprint post-processing (BUG-R07: grey blob → Edge.Cuts slot)
# ---------------------------------------------------------------------------

# Minimum F.Fab line width to classify as the EasyEDA coil-symbol blob.
_RELAY_FAB_BLOB_MIN_WIDTH: float = 2.0

# Relay isolation slot parameters — vertical slot between coil/COM and NC/NO
# groups, spanning the relay body height on Edge.Cuts.
_RELAY_SLOT_WIDTH: float = 1.0  # routed slot milling width (mm)


def _is_relay_footprint(fp: Footprint) -> bool:
    """Return True if *fp* looks like an SPDT relay (SRD series or similar)."""
    upper_id = fp.lib_id.upper()
    return "RELAY" in upper_id and len(fp.pads) == 5


def _postprocess_relay_footprint(fp: Footprint) -> Footprint:
    """Remove the EasyEDA coil-symbol F.Fab blob and add an Edge.Cuts isolation slot.

    The EasyEDA/JLCPCB relay footprint contains thick (3mm width) F.Fab lines
    that render as a grey blob in KiCad.  This function:

    1. Strips those thick F.Fab lines.
    2. Computes the midpoint X between the coil-side pads (pins 1, 4, 5) and
       the contact-side pads (pins 2, 3) to place a vertical Edge.Cuts slot
       for creepage isolation between low-voltage coil and mains/load contacts.
    """
    # --- Step 1: remove thick F.Fab blob lines ---
    cleaned: list[FootprintLine | FootprintArc | FootprintCircle] = []
    for g in fp.graphics:
        if (
            isinstance(g, FootprintLine)
            and "Fab" in g.layer
            and g.width >= _RELAY_FAB_BLOB_MIN_WIDTH
        ):
            continue  # drop the blob line
        cleaned.append(g)

    # --- Step 2: compute isolation slot position from pad geometry ---
    # SRD pinout: 1=Coil+, 4=Coil-, 5=COM (low-voltage side)
    #             2=NC, 3=NO (high-voltage contact side)
    coil_pins = {"1", "4", "5"}
    contact_pins = {"2", "3"}

    coil_xs: list[float] = []
    contact_xs: list[float] = []
    all_ys: list[float] = []
    for pad in fp.pads:
        all_ys.append(pad.position.y)
        if pad.number in coil_pins:
            coil_xs.append(pad.position.x)
        elif pad.number in contact_pins:
            contact_xs.append(pad.position.x)

    if coil_xs and contact_xs and all_ys:
        coil_max_x = max(coil_xs)
        contact_min_x = min(contact_xs)
        slot_x = (coil_max_x + contact_min_x) / 2.0

        # Slot spans slightly beyond the outermost pad Y positions.
        pad_r = max((p.size_x for p in fp.pads), default=2.0) / 2.0
        y_min = min(all_ys) - pad_r - 1.0
        y_max = max(all_ys) + pad_r + 1.0

        slot_line = FootprintLine(
            start=Point(slot_x, y_min),
            end=Point(slot_x, y_max),
            layer=LAYER_EDGE_CUTS,
            width=_RELAY_SLOT_WIDTH,
        )
        cleaned.append(slot_line)
        _log.info(
            "Relay %s: added Edge.Cuts isolation slot at x=%.1f (y %.1f..%.1f)",
            fp.ref, slot_x, y_min, y_max,
        )
    else:
        _log.warning(
            "Relay %s: could not determine pad groups for isolation slot", fp.ref,
        )

    return Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
        position=fp.position, rotation=fp.rotation, layer=fp.layer,
        pads=fp.pads, graphics=tuple(cleaned), texts=fp.texts,
        lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr, models=fp.models,
        datasheet=fp.datasheet, description=fp.description,
        footprint_source=fp.footprint_source,
    )


# ---------------------------------------------------------------------------
# ESP32 thermal pad post-processor
# ---------------------------------------------------------------------------


def _postprocess_esp32_thermal_pad(fp: Footprint) -> Footprint:
    """Merge multiple pad-41 entries into one central thermal pad.

    JLCPCB ESP32 footprints store the exposed ground pad as a 3x3 grid of
    small pads all numbered "41".  KiCad expects a single pad at the centre.
    This function merges them into one rectangle that spans the bounding box
    of all pad-41 entries.

    Args:
        fp: An ESP32 footprint (possibly with multiple pad 41).

    Returns:
        A copy of *fp* with at most one pad "41".
    """
    p41_pads = [p for p in fp.pads if p.number == "41"]
    if len(p41_pads) <= 1:
        return fp  # Already correct — nothing to do.

    # Compute bounding box of all pad-41 entries (pad extent, not centres).
    min_x = min(p.position.x - p.size_x / 2 for p in p41_pads)
    max_x = max(p.position.x + p.size_x / 2 for p in p41_pads)
    min_y = min(p.position.y - p.size_y / 2 for p in p41_pads)
    max_y = max(p.position.y + p.size_y / 2 for p in p41_pads)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    w = max_x - min_x
    h = max_y - min_y

    # Build a single merged pad, copying attributes from the first pad-41.
    template = p41_pads[0]
    merged = Pad(
        number=template.number,
        pad_type=template.pad_type,
        shape=template.shape,
        position=Point(x=cx, y=cy),
        size_x=w,
        size_y=h,
        layers=template.layers,
        net_number=template.net_number,
        net_name=template.net_name,
        drill_diameter=template.drill_diameter,
        roundrect_ratio=template.roundrect_ratio,
        uuid=template.uuid,
    )

    # Replace all pad-41 entries with the single merged pad.
    new_pads = tuple(p for p in fp.pads if p.number != "41") + (merged,)
    _log.info(
        "%s: merged %d pad-41 grid entries into single %.1fx%.1fmm thermal pad "
        "at (%.1f, %.1f)",
        fp.ref, len(p41_pads), w, h, cx, cy,
    )
    return Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
        position=fp.position, rotation=fp.rotation, layer=fp.layer,
        pads=new_pads, graphics=fp.graphics, texts=fp.texts,
        lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr, models=fp.models,
        datasheet=fp.datasheet, description=fp.description,
        footprint_source=fp.footprint_source, mpn=fp.mpn,
        manufacturer=fp.manufacturer, fp_zones=fp.fp_zones,
    )


# ---------------------------------------------------------------------------
# JLCPCB footprint loader integration
# ---------------------------------------------------------------------------


def _try_jlcpcb_footprint(
    lcsc: str,
    ref: str,
    value: str,
    layer: str,
    footprint_id: str = "",
) -> Footprint | None:
    """Attempt to load a JLCPCB-sourced footprint by LCSC number.

    Downloads (if needed) and parses the ``.kicad_mod`` file. Returns
    None on any failure, allowing fallback to parametric generation.
    """
    try:
        from kicad_pipeline.parts.footprint_cache import get_jlcpcb_footprint
        from kicad_pipeline.pcb.footprint_loader import load_kicad_mod
    except ImportError:
        return None

    mod_path = get_jlcpcb_footprint(lcsc)
    if mod_path is None:
        return None

    try:
        fp = load_kicad_mod(mod_path, ref=ref, value=value, layer=layer, lcsc=lcsc)
        # Tag source provenance for downstream verification tracking
        fp = Footprint(
            lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
            position=fp.position, rotation=fp.rotation, layer=fp.layer,
            pads=fp.pads, graphics=fp.graphics, texts=fp.texts,
            lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr, models=fp.models,
            datasheet=fp.datasheet, description=fp.description,
            footprint_source="jlcpcb", fp_zones=fp.fp_zones,
        )
        # Relay post-processing: remove F.Fab blob, add Edge.Cuts isolation slot
        if _is_relay_footprint(fp):
            fp = _postprocess_relay_footprint(fp)
        # QFN/DFN packages: add thermal/exposed pad if the JLCPCB footprint
        # doesn't include one.  The pad number is pin_count+1.
        # Check both the JLCPCB lib_id and the original requested footprint_id.
        _check_ids = (fp.lib_id.upper(), footprint_id.upper())
        if any(kw in uid for uid in _check_ids for kw in ("QFN", "DFN")):
            pin_count = _parse_pin_count(footprint_id) or _parse_pin_count(fp.lib_id)
            if pin_count > 0 and len(fp.pads) == pin_count:
                # No thermal pad present — add one
                ep_size = max(2.0, pin_count * 0.08)
                ep = _smd_pad(str(pin_count + 1), 0.0, 0.0, ep_size, ep_size, fp.layer)
                fp = Footprint(
                    lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                    position=fp.position, rotation=fp.rotation, layer=fp.layer,
                    pads=(*fp.pads, ep), graphics=fp.graphics, texts=fp.texts,
                    lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr, models=fp.models,
                    datasheet=fp.datasheet, description=fp.description,
                    footprint_source=fp.footprint_source,
                )
                _log.info("Added thermal pad %d to QFN footprint %s", pin_count + 1, ref)

        # JLCPCB .kicad_mod files rarely include 3D model references.
        # Resolve a model from the original footprint_id or the JLCPCB lib_id.
        if not fp.models:
            model = _model_for_package(footprint_id, layer)
            if model is None:
                model = _model_for_package(fp.lib_id, layer)
            if model is not None:
                fp = Footprint(
                    lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                    position=fp.position, rotation=fp.rotation, layer=fp.layer,
                    pads=fp.pads, graphics=fp.graphics, texts=fp.texts,
                    lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
                    models=(model,),
                    datasheet=fp.datasheet, description=fp.description,
                    footprint_source=fp.footprint_source, fp_zones=fp.fp_zones,
                )

        _log.info(
            "Using JLCPCB footprint for %s (%s): %d pads from %s",
            ref, lcsc, len(fp.pads), mod_path.name,
        )
        return fp
    except Exception:
        _log.warning(
            "Failed to parse JLCPCB footprint for %s (%s), falling back to parametric",
            ref, lcsc, exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------


def _fp_ws2812(
    ref: str, value: str, fid: str, _upper: str, layer: str,
) -> Footprint:
    """Build WS2812 footprint with size auto-detection."""
    ws_size = "5050"
    if "2020" in fid:
        ws_size = "2020"
    elif "3535" in fid:
        ws_size = "3535"
    return make_ws2812b(ref, value, layer=layer, size=ws_size)


def _fp_led(
    ref: str, value: str, fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build SMD LED footprint."""
    pkg = fid[4:].upper()
    pkg_norm = pkg if pkg in _SMD_RC_DIMS else "0805"
    return make_smd_led(ref, value, package=pkg_norm)


def _fp_rc(
    ref: str, value: str, fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build SMD resistor/capacitor footprint."""
    pkg = fid[2:].upper()
    pkg_norm = pkg if pkg in _SMD_RC_DIMS else "0805"
    return make_smd_resistor_capacitor(ref, value, package=pkg_norm)


def _fp_sod323(
    ref: str, value: str, _fid: str, _upper: str, layer: str,
) -> Footprint:
    """Build SOD-323 diode footprint."""
    return make_sod323(ref, value, layer=layer)


def _fp_sod123(
    ref: str, value: str, _fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build SOD-123 diode footprint."""
    return make_sod123(ref, value)


def _fp_inductor(
    ref: str, value: str, fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build SMD inductor footprint."""
    pkg = fid[2:].upper()
    return make_inductor_smd(ref, value, package=pkg)


def _fp_sot223(
    ref: str, value: str, _fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build SOT-223 footprint."""
    return make_sot23(ref, value, variant="SOT-23")


def _fp_sot23(
    ref: str, value: str, fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build SOT-23 family footprint."""
    variant = fid.upper().replace("TSOT-", "SOT-")
    if variant not in _SOT23_VARIANTS:
        variant = "SOT-23"
    return make_sot23(ref, value, variant=variant)


# Ordered (predicate, handler) dispatch table for passive/LED footprints.
# Order matters: WS2812 contains-check first, SOT-223 exact before SOT-23 prefix.
_PASSIVE_LED_DISPATCH: list[
    tuple[
        Callable[[str, str], bool],
        Callable[[str, str, str, str, str], Footprint],
    ]
] = [
    (lambda _u, _f: "WS2812" in _u, _fp_ws2812),
    (lambda _u, _f: _u.startswith("LED_"), _fp_led),
    (lambda _u, _f: _u.startswith(("R_", "C_")), _fp_rc),
    (lambda _u, _f: _u.startswith("SOD-323"), _fp_sod323),
    (lambda _u, _f: _u.startswith("SOD-123"), _fp_sod123),
    (lambda _u, _f: _u.startswith("L_"), _fp_inductor),
    (lambda _u, _f: _u == "SOT-223", _fp_sot223),
    (lambda _u, _f: _u.startswith(("SOT-23", "TSOT-23")), _fp_sot23),
]


def _route_fp_passive_led(
    ref: str, value: str, fid: str, upper: str, layer: str,
) -> Footprint | None:
    """Match WS2812, LED, R_, C_, SOD, L_, SOT packages."""
    for predicate, handler in _PASSIVE_LED_DISPATCH:
        if predicate(upper, fid):
            return handler(ref, value, fid, upper, layer)
    return None


def _fp_usbc(
    ref: str, value: str, _fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build USB-C connector footprint."""
    return make_usbc_connector(ref, value)


def _fp_rj45(
    ref: str, value: str, _fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build RJ45 connector footprint."""
    return make_rj45(ref, value)


def _fp_pin_header_socket(
    ref: str, value: str, fid: str, upper: str, layer: str,
) -> Footprint:
    """Build pin header/socket connector footprint."""
    pin_count = _parse_pin_count(fid)
    pitch = _parse_pitch(fid)
    rows = 2 if "2X" in upper or "2x" in fid else 1
    rpi_swap = "2X20" in upper or "RPI" in upper or "RASPBERRY" in upper
    return make_pin_header_socket(
        ref, value, pin_count, pitch, rows, lib_id=fid, row_swap=rpi_swap,
        layer=layer,
    )


def _fp_terminal_block(
    ref: str, value: str, fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build terminal block footprint."""
    pin_count = _parse_pin_count(fid)
    pitch = _parse_pitch(fid)
    return make_terminal_block(ref, value, pin_count, pitch)


def _fp_microsd(
    ref: str, value: str, _fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build MicroSD slot footprint."""
    return make_microsd_slot(ref, value)


# Ordered (predicate, handler) dispatch table for connector footprints.
_CONNECTOR_DISPATCH: list[
    tuple[
        Callable[[str, str], bool],
        Callable[[str, str, str, str, str], Footprint],
    ]
] = [
    (lambda _u, _f: _u.startswith(("USB-C", "USB_C")), _fp_usbc),
    (lambda _u, _f: _u.startswith("RJ45"), _fp_rj45),
    (lambda _u, _f: _u.startswith(("PINHEADER", "PINSOCKET", "CONN_")), _fp_pin_header_socket),
    (lambda _u, _f: _u.startswith(("TERMINALBLOCK", "TB_")), _fp_terminal_block),
    (lambda _u, _f: _u.startswith(("TF_PUSH", "MICROSD", "MICRO_SD")), _fp_microsd),
]


def _route_fp_connector(
    ref: str, value: str, fid: str, upper: str, layer: str,
) -> Footprint | None:
    """Match USB-C, RJ45, PinHeader, PinSocket, Conn_, TerminalBlock."""
    for predicate, handler in _CONNECTOR_DISPATCH:
        if predicate(upper, fid):
            return handler(ref, value, fid, upper, layer)
    return None


def _parse_dip_switch_pin_count(fid: str) -> int:
    """Parse pin count for DIP switches, accounting for xN notation."""
    import re as _re
    pos_m = _re.search(r"x(\d+)", fid)
    if pos_m:
        return int(pos_m.group(1)) * 2
    pin_count = _parse_pin_count(fid)
    return pin_count if pin_count >= 2 else 8


def _parse_dimensions(fid: str) -> tuple[float, float] | None:
    """Parse WxH dimensions from a footprint ID string."""
    import re as _re
    m = _re.search(r"(\d+\.?\d*)x(\d+\.?\d*)", fid)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _fp_dip_switch(
    ref: str, value: str, fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build DIP switch footprint."""
    return make_dip_switch(ref, value, _parse_dip_switch_pin_count(fid))


def _fp_relay(
    ref: str, value: str, _fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build relay SPDT footprint."""
    return make_relay_spdt(ref, value)


def _fp_esp32(
    ref: str, value: str, _fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build ESP32/WROOM footprint."""
    return make_esp32_wroom(ref, value)


def _fp_smd_tact_switch(
    ref: str, value: str, fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build SMD tactile switch footprint."""
    dims = _parse_dimensions(fid)
    w, h = dims if dims else (3.0, 2.5)
    return make_smd_tact_switch(ref, value, width_mm=w, height_mm=h)


def _fp_tact_switch(
    ref: str, value: str, fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build THT tactile switch footprint."""
    dims = _parse_dimensions(fid)
    size_mm = dims[0] if dims else 4.5
    return make_tact_switch(ref, value, size_mm=size_mm)


def _fp_crystal(
    ref: str, value: str, fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build SMD crystal footprint."""
    dims = _parse_dimensions(fid)
    w, h = dims if dims else (3.2, 1.5)
    return make_crystal_smd(ref, value, size_w=w, size_h=h)


def _fp_test_point(
    ref: str, value: str, fid: str, _upper: str, layer: str,
) -> Footprint:
    """Build test point footprint."""
    dims = _parse_dimensions(fid)
    pad_size = dims[0] if dims else 1.5
    return make_test_point(ref, value, pad_size=pad_size, layer=layer)


def _fp_dip_package(
    ref: str, value: str, fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build DIP IC package footprint."""
    pin_count = _parse_pin_count(fid)
    return make_dip_package(ref, value, max(pin_count, 4) if pin_count < 2 else pin_count)


# Ordered (predicate, handler) dispatch table for switches and misc footprints.
# Order matters: SW_DIP before SW_SMD before generic SW_, ESP32 contains-check.
_SWITCH_MISC_DISPATCH: list[
    tuple[
        Callable[[str, str], bool],
        Callable[[str, str, str, str, str], Footprint],
    ]
] = [
    (lambda _u, _f: _u.startswith(("SW_DIP", "DIP_SWITCH")), _fp_dip_switch),
    (lambda _u, _f: _u.startswith("RELAY"), _fp_relay),
    (lambda _u, _f: "ESP32" in _u or "WROOM" in _u, _fp_esp32),
    (lambda _u, _f: _u.startswith("SW_") and "SMD" in _u, _fp_smd_tact_switch),
    (lambda _u, _f: _u.startswith("SW_"), _fp_tact_switch),
    (lambda _u, _f: _u.startswith("CRYSTAL"), _fp_crystal),
    (lambda _u, _f: _u.startswith(("TP_", "TESTPOINT")), _fp_test_point),
    (
        lambda _u, _f: _u.startswith("DIP-") or (
            _u.startswith("DIP_") and "SWITCH" not in _u
        ),
        _fp_dip_package,
    ),
]


def _route_fp_switch_misc(
    ref: str, value: str, fid: str, upper: str, layer: str,
) -> Footprint | None:
    """Match DIP switches, tactile switches, relays, ESP32, crystals, test points, DIP packages."""
    for predicate, handler in _SWITCH_MISC_DISPATCH:
        if predicate(upper, fid):
            return handler(ref, value, fid, upper, layer)
    return None


def _route_fp_smd_ic(
    ref: str, value: str, fid: str, upper: str,
) -> Footprint | None:
    """Match generic SMD IC packages (MSOP, TSSOP, SOIC, QFP, QFN, etc.)."""
    ic_prefixes = ("MSOP", "TSSOP", "SOIC", "QFP", "QFN", "SOP", "DFN", "SSOP", "LQFP")
    if not any(upper.startswith(p) for p in ic_prefixes):
        return None
    pin_count = _parse_pin_count(fid)
    if pin_count < 2:
        pin_count = 8
    pitch = _parse_pitch(fid)
    if pitch > 2.0:
        if upper.startswith(("SOP", "SOIC")):
            pitch = 1.27
        else:
            pitch = 0.5
    return make_generic_smd_ic(ref, value, pin_count, pitch, lib_id=fid)


def _route_footprint(
    ref: str,
    value: str,
    fid: str,
    upper: str,
    layer: str,
    footprint_id: str,
) -> tuple[Footprint, str]:
    """Route a footprint ID to the correct generator. Returns (fp, source)."""
    _source = "parametric"

    fp = _route_fp_passive_led(ref, value, fid, upper, layer)
    if fp is not None:
        return fp, _source

    fp = _route_fp_connector(ref, value, fid, upper, layer)
    if fp is not None:
        return fp, _source

    fp = _route_fp_switch_misc(ref, value, fid, upper, layer)
    if fp is not None:
        return fp, _source

    fp = _route_fp_smd_ic(ref, value, fid, upper)
    if fp is not None:
        return fp, _source

    _log.warning(
        "footprint_for_component: unknown footprint_id '%s' for ref %s; using 0805 fallback",
        footprint_id,
        ref,
    )
    return make_smd_resistor_capacitor(ref, value, package="0805"), "parametric-fallback"


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

    # JLCPCB-first: try loading verified footprint from JLCPCB library
    if lcsc:
        fp = _try_jlcpcb_footprint(lcsc, ref, value, layer, footprint_id=footprint_id)
        if fp is not None:
            # Enrich ESP32/WROOM footprints from JLCPCB cache with pin
            # labels, antenna keepout, and 3D model.
            fid_upper = footprint_id.strip().upper()
            if "ESP32" in fid_upper or "WROOM" in fid_upper:
                fp = _postprocess_esp32_thermal_pad(fp)
                fp = _enrich_esp32_footprint(fp)
            return fp

    fid = footprint_id.strip()
    upper = fid.upper()

    fp, _source = _route_footprint(ref, value, fid, upper, layer, footprint_id)

    # Attach LCSC and source tag
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
        lcsc=lcsc if lcsc is not None else fp.lcsc,
        uuid=fp.uuid,
        attr=fp.attr,
        models=fp.models,
        footprint_source=_source,
        fp_zones=fp.fp_zones,
    )

    # Enrich ESP32/WROOM footprints with pin labels, antenna keepout, and
    # 3D model regardless of whether they came from JLCPCB cache or
    # parametric generator.
    if "ESP32" in upper or "WROOM" in upper:
        fp = _postprocess_esp32_thermal_pad(fp)
        fp = _enrich_esp32_footprint(fp)

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
    "module": (0.5, 3.5),       # ESP32/W5500 — body extends well beyond pad field (antenna, shield)
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


def _courtyard_from_graphics(fp: Footprint) -> tuple[float, float] | None:
    """Extract courtyard dimensions from F.CrtYd graphics if present.

    Scans footprint graphics for lines on the ``F.CrtYd`` or ``B.CrtYd``
    layer and computes the bounding box.  Returns ``None`` if no courtyard
    graphics exist.
    """
    from kicad_pipeline.models.pcb import FootprintLine

    xs: list[float] = []
    ys: list[float] = []
    for g in fp.graphics:
        if isinstance(g, FootprintLine) and "CrtYd" in g.layer:
            xs.extend((g.start.x, g.end.x))
            ys.extend((g.start.y, g.end.y))
    if not xs:
        return None
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if w < 0.5 or h < 0.5:
        return None
    return (w, h)


def _body_from_fab_graphics(fp: Footprint) -> tuple[float, float] | None:
    """Extract body dimensions from F.Fab graphics if present.

    The ``F.Fab`` layer typically contains the component body outline.
    Returns ``None`` if no fab graphics exist.
    """
    from kicad_pipeline.models.pcb import FootprintLine

    xs: list[float] = []
    ys: list[float] = []
    for g in fp.graphics:
        if isinstance(g, FootprintLine) and "Fab" in g.layer:
            xs.extend((g.start.x, g.end.x))
            ys.extend((g.start.y, g.end.y))
    if not xs:
        return None
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if w < 0.5 or h < 0.5:
        return None
    return (w, h)


def estimate_courtyard_mm(fp: Footprint) -> tuple[float, float]:
    """Estimate courtyard (width, height) from footprint geometry.

    Resolution order (first valid result wins):

    1. **Courtyard graphics** — ``F.CrtYd`` / ``B.CrtYd`` layer lines
       define the exact courtyard.  Used directly when present.
    2. **Fab body + clearance** — ``F.Fab`` layer lines define the
       physical body outline.  Courtyard = body + clearance margin.
    3. **Pad extents + body extension** — heuristic: pad field bounding
       box plus a package-type-dependent body extension factor.

    The first two methods use actual geometry from the footprint and are
    accurate even for modules where the body far exceeds the pad field
    (e.g., ESP32-WROOM: pads span ~18.5mm but body is 25.5mm).

    Args:
        fp: A :class:`Footprint` with pad data.

    Returns:
        ``(width_mm, height_mm)`` courtyard estimate.
    """
    # 1. Existing courtyard graphics — most authoritative
    crtyd = _courtyard_from_graphics(fp)
    if crtyd is not None:
        return (max(crtyd[0], 1.0), max(crtyd[1], 1.0))

    # 2. Fab body outline + clearance
    body = _body_from_fab_graphics(fp)
    if body is not None:
        w = body[0] + 2.0 * _COURTYARD_CLEARANCE
        h = body[1] + 2.0 * _COURTYARD_CLEARANCE
        # Ensure courtyard is at least as large as pad extents
        if fp.pads:
            pad_w = (max(p.position.x + p.size_x / 2.0 for p in fp.pads)
                     - min(p.position.x - p.size_x / 2.0 for p in fp.pads))
            pad_h = (max(p.position.y + p.size_y / 2.0 for p in fp.pads)
                     - min(p.position.y - p.size_y / 2.0 for p in fp.pads))
            w = max(w, pad_w + 2.0 * _COURTYARD_CLEARANCE)
            h = max(h, pad_h + 2.0 * _COURTYARD_CLEARANCE)
        return (max(w, 1.0), max(h, 1.0))

    # 3. Fallback: pad extents + heuristic body extension
    if not fp.pads:
        return estimate_footprint_size(fp.lib_id)

    min_x = min(p.position.x - p.size_x / 2.0 for p in fp.pads)
    max_x = max(p.position.x + p.size_x / 2.0 for p in fp.pads)
    min_y = min(p.position.y - p.size_y / 2.0 for p in fp.pads)
    max_y = max(p.position.y + p.size_y / 2.0 for p in fp.pads)

    pad_w = max_x - min_x
    pad_h = max_y - min_y

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


def _estimate_smd_rc(fid: str, upper: str) -> tuple[float, float] | None:
    """Estimate size for SMD R/C/LED packages."""
    for prefix in ("R_", "C_", "LED_"):
        if upper.startswith(prefix):
            pkg = fid[len(prefix):].upper()
            if pkg in _SMD_RC_DIMS:
                _, _, _pitch, body_w, body_h = _SMD_RC_DIMS[pkg]
                return (body_w + _BODY_MARGIN_MM, body_h + _BODY_MARGIN_MM)
            return (2.5, 1.75)  # 0805 fallback
    return None


def _estimate_inductor(fid: str, upper: str) -> tuple[float, float] | None:
    """Estimate size for inductor packages."""
    if not upper.startswith("L_"):
        return None
    pkg = fid[2:].upper()
    if pkg in _SMD_RC_DIMS:
        _, _, _pitch, body_w, body_h = _SMD_RC_DIMS[pkg]
        return (body_w + _BODY_MARGIN_MM, body_h + _BODY_MARGIN_MM)
    return (4.0, 3.0)


def _estimate_sot23(fid: str, upper: str) -> tuple[float, float] | None:
    """Estimate size for SOT-23 / TSOT-23 family."""
    if not upper.startswith(("SOT-23", "TSOT-23")):
        return None
    variant = fid.upper().replace("TSOT-", "SOT-")
    if variant in _SOT23_VARIANTS:
        _, _, coords = _SOT23_VARIANTS[variant]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return (max(xs) - min(xs) + 1.5, max(ys) - min(ys) + 1.5)
    return (3.0, 3.0)


def _estimate_connector(fid: str, upper: str) -> tuple[float, float] | None:
    """Estimate size for pin headers, sockets, and generic connectors."""
    if not upper.startswith(("PINHEADER", "PINSOCKET", "CONN_")):
        return None
    pin_count = _parse_pin_count(fid)
    pitch = _parse_pitch(fid)
    num_cols = 2 if "2X" in upper or "2x" in fid else 1
    pins_per_col = pin_count // max(num_cols, 1)
    pad_margin = 3.5 if pitch >= 2.0 else 2.5
    w = (num_cols - 1) * pitch + pad_margin
    h = (pins_per_col - 1) * pitch + pad_margin
    return (w, h)


def _estimate_terminal_block(fid: str, upper: str) -> tuple[float, float] | None:
    """Estimate size for terminal blocks."""
    if not upper.startswith(("TERMINALBLOCK", "TB_")):
        return None
    pin_count = _parse_pin_count(fid)
    pitch = _parse_pitch(fid)
    return ((pin_count - 1) * pitch + 5.0, 7.0)


def _estimate_dip_switch(fid: str, upper: str) -> tuple[float, float] | None:
    """Estimate size for DIP switches."""
    if not upper.startswith("SW_DIP"):
        return None
    import re as _re
    dim_m = _re.search(r"(\d+\.?\d*)x(\d+\.?\d*)mm", fid)
    if dim_m:
        return (float(dim_m.group(1)) + 1.0, float(dim_m.group(2)) + 1.0)
    pin_count = _parse_pin_count(fid)
    half = max(pin_count // 2, 2)
    return (8.5, (half - 1) * 2.54 + 3.0)


def _estimate_tactile_switch(fid: str, upper: str) -> tuple[float, float] | None:
    """Estimate size for tactile switches (must be checked after SW_DIP)."""
    if not upper.startswith("SW_"):
        return None
    import re as _re
    size_m = _re.search(r"(\d+\.?\d*)x(\d+\.?\d*)", fid)
    if size_m:
        return (float(size_m.group(1)) + 2.0, float(size_m.group(2)) + 2.0)
    return (7.0, 7.0)


def _estimate_crystal(fid: str, upper: str) -> tuple[float, float] | None:
    """Estimate size for crystal oscillators."""
    if not upper.startswith("CRYSTAL"):
        return None
    import re as _re
    dim_m = _re.search(r"(\d+\.?\d*)x(\d+\.?\d*)", fid)
    if dim_m:
        return (float(dim_m.group(1)) + _BODY_MARGIN_MM, float(dim_m.group(2)) + _BODY_MARGIN_MM)
    return (4.0, 2.0)


_SMD_IC_PREFIXES = ("MSOP", "TSSOP", "SOIC", "QFP", "QFN", "SOP", "DFN", "SSOP", "LQFP")


def _estimate_smd_ic(fid: str, upper: str) -> tuple[float, float] | None:
    """Estimate size for generic SMD IC packages."""
    if not any(upper.startswith(p) for p in _SMD_IC_PREFIXES):
        return None
    pin_count = _parse_pin_count(fid)
    pitch = _parse_pitch(fid)
    if pitch > 2.0:
        pitch = 0.5
    half = pin_count // 2
    row_span = (half - 1) * pitch
    return (row_span / 2.0 + 3.0, row_span + 1.5)


# Fixed-prefix size lookups (checked via startswith)
_FIXED_SIZE_PREFIXES: tuple[tuple[tuple[str, ...], tuple[float, float]], ...] = (
    (("SOD-123",), (3.5, 2.5)),
    (("SOT-223",), (7.0, 4.0)),
    (("USB-C", "USB_C"), (9.5, 8.0)),
    (("RJ45",), (17.0, 23.0)),
    (("RELAY",), (20.0, 16.5)),
)

# ESP32 uses 'in' check, not startswith
_ESP32_SIZE: tuple[float, float] = (18.5, 26.0)


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

    # SMD R/C/LED packages
    result = _estimate_smd_rc(fid, upper)
    if result is not None:
        return result

    # Fixed-prefix lookups
    for prefixes, size in _FIXED_SIZE_PREFIXES:
        if upper.startswith(prefixes):
            return size

    # ESP32 modules (uses 'in' check)
    if "ESP32" in upper or "WROOM" in upper:
        return _ESP32_SIZE

    # Parameterized estimators (order matters: SW_DIP before SW_)
    estimators = (
        _estimate_inductor,
        _estimate_sot23,
        _estimate_connector,
        _estimate_terminal_block,
        _estimate_dip_switch,
        _estimate_crystal,
        _estimate_tactile_switch,
        _estimate_smd_ic,
    )
    for estimator in estimators:
        result = estimator(fid, upper)
        if result is not None:
            return result

    # Generic fallback
    return (3.0, 3.0)


# Keep math in module namespace so tests / type checker see no unused import
_PI = math.pi
