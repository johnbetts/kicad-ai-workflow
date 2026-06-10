"""Core footprint generation utilities for the kicad-ai-pipeline.

Provides basic 3D model path resolution, layer flipping, and core footprint
generation utilities.
"""

from __future__ import annotations

import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from kicad_pipeline.models.requirements import Pin

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

# Data directory for footprint resources
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
    # Default orientation: plug opening faces north (negative Y), A-row pads
    # at the north/plug side, B-row at south, shield tab furthest south.
    # At rotation 0 with connector at the top board edge, A1 (GND) is closest
    # to the edge and A4 (VBUS) is also at the plug side.
    (-3.5, -2.5, 1.6, 1.6, "A1"),    # GND (A1) — plug side (north)
    (-2.0, -2.5, 0.6, 1.6, "A5"),    # CC1 (A5)
    (-1.0, -2.5, 0.6, 1.6, "A7"),    # D- (A7)
    (1.0, -2.5, 0.6, 1.6, "A6"),     # D+ (A6)
    (2.0, -2.5, 0.6, 1.6, "B5"),     # CC2 (B5)
    (3.5, -2.5, 1.6, 1.6, "A4"),     # VBUS (A4) — plug side (north)
    (-1.0, 2.5, 0.6, 1.6, "B7"),     # D-_B (B7) — board-interior side
    (1.0, 2.5, 0.6, 1.6, "B6"),      # D+_B (B6) — board-interior side
    (0.0, 3.5, 2.0, 1.0, "S1"),      # Shield (S1) — furthest south (interior)
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

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3D model path resolution & validation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _resolve_3d_model_dir() -> Path | None:
    """Resolve ``${KICAD10_3DMODEL_DIR}`` to a filesystem path.

    Checks the environment variable first, then standard install locations.
    Returns ``None`` if no 3D model directory is found (validation skipped).
    """
    env = os.environ.get("KICAD10_3DMODEL_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    # macOS KiCad 10
    mac = Path("/Applications/KiCad 10/KiCad.app/Contents/SharedSupport/3dmodels")
    if mac.is_dir():
        return mac
    # Linux
    linux = Path("/usr/share/kicad/3dmodels")
    if linux.is_dir():
        return linux
    return None


def _step_file_exists(model_path: str) -> bool:
    """Check whether a 3D model file (STEP or WRL) exists on disk.

    Resolves the ``${KICAD10_3DMODEL_DIR}`` prefix to the actual directory.
    Also checks absolute paths (JLCPCB cached footprints use absolute paths
    for ``.wrl`` models).
    Returns ``True`` if the directory cannot be resolved (fail-open).
    """
    # Absolute path (JLCPCB cache: /Users/.../C318884.3dshapes/model.wrl)
    if model_path.startswith("/"):
        return Path(model_path).exists()
    base = _resolve_3d_model_dir()
    if base is None:
        return True  # Can't validate — assume OK
    rel = model_path.replace("${KICAD10_3DMODEL_DIR}/", "")
    return (base / rel).exists()


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
    ("SOT-223", "Package_TO_SOT_SMD.3dshapes", "SOT-223.step"),
    # Diodes
    ("SOD-323", "Diode_SMD.3dshapes", "D_SOD-323.step"),
    ("SOD-123", "Diode_SMD.3dshapes", "D_SOD-123.step"),
    # Inductors
    ("L_1210", "Inductor_SMD.3dshapes", "L_1210_3225Metric.step"),
    ("L_1206", "Inductor_SMD.3dshapes", "L_1206_3216Metric.step"),
    ("L_0805", "Inductor_SMD.3dshapes", "L_0805_2012Metric.step"),
    # Crystals
    ("Crystal_SMD_3215", "Crystal.3dshapes", "Crystal_SMD_3215-2Pin_3.2x1.5mm.step"),
    # Buzzers — match both "Buzzer_12mm" (parametric lib_id) and "Buzzer_12x9.5mm"
    ("BUZZER_12", "Buzzer_Beeper.3dshapes", "Buzzer_12x9.5RM7.6.step"),
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
    "TSSOP-20": "_4.4x6.5mm_P0.65mm",
    "TSSOP-24": "_7.8x4.4mm_P0.5mm",
    "TSSOP-28": "_9.7x4.4mm_P0.65mm",
    "SOIC-8": "_3.9x4.9mm_P1.27mm",
    "SOIC-14": "_3.9x8.7mm_P1.27mm",
    "SOIC-16": "_3.9x9.9mm_P1.27mm",
    "LQFP-32": "_7x7mm_P0.8mm",
    "LQFP-44": "_10x10mm_P0.8mm",
    "LQFP-48": "_7x7mm_P0.5mm",
    "LQFP-64": "_10x10mm_P0.5mm",
    "LQFP-100": "_14x14mm_P0.5mm",
    "QFN-16": "-1EP_3x3mm_P0.5mm_EP1.7x1.7mm",
    "QFN-20": "-1EP_4x4mm_P0.5mm_EP2.5x2.5mm",
    "QFN-24": "-1EP_4x4mm_P0.5mm_EP2.5x2.5mm",
    "QFN-32": "-1EP_5x5mm_P0.5mm_EP3.3x3.3mm",
    "QFN-48": "-1EP_7x7mm_P0.5mm_EP5.15x5.15mm",
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
    lookup = name.upper()
    suffix = _IC_DIMENSION_SUFFIXES.get(lookup, "")
    # QFP without L/T prefix → default to LQFP (most common variant)
    if not suffix and lookup.startswith("QFP-"):
        lqfp_key = "L" + lookup
        suffix = _IC_DIMENSION_SUFFIXES.get(lqfp_key, "")
        if suffix:
            name = "L" + name  # Prepend L to get LQFP model name
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
    # Ensure pitch is present (KiCad filenames require _P2.54mm)
    if "_P" not in model_name and "2.54" not in model_name:
        model_name += "_P2.54mm"
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
    for prefix in ("QFN", "DFN"):
        if upper.startswith(prefix):
            model_name = _ic_model_name(name, prefix)
            path = f"{KICAD_3DMODEL_VAR}/Package_DFN_QFN.3dshapes/{model_name}.step"
            return Footprint3DModel(path=path)
    for prefix in ("SOIC", "MSOP", "TSSOP", "SSOP", "SOP"):
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
    # KiCad standard terminal block footprints use zero rotation.
    # The STEP model's wire entry faces +Y by default.
    return Footprint3DModel(path=path)


def _model_esp32(
    name: str, upper: str,
) -> Footprint3DModel | None:
    """Return 3D model for ESP32/RF modules.

    The KiCad standard ESP32-S3-WROOM-1.step model has its internal origin
    aligned with the KiCad standard library footprint origin, which is
    ~3.63mm north (-Y in module coords) of the pad-field centroid.
    easyeda2kicad footprints place their origin at the pad centroid, so we
    apply a +3.63mm Y offset to re-align the 3D body with the pads.
    """
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

    # Tactile switches — all SMD tact switches use TL3305A model
    # (5.1x5.1mm body, pads at ±3.6/±1.5). Verified against KiCad 10
    # standard footprint: zero offset, zero rotation.
    if upper.startswith("SW_PUSH") or (upper.startswith("SW_") and "SPST" in upper):
        return Footprint3DModel(
            path=f"{KICAD_3DMODEL_VAR}/Button_Switch_SMD.3dshapes/SW_SPST_TL3305A.step",
        )
    return None


def _model_connector(
    name: str, upper: str, layer: str,
) -> Footprint3DModel | None:
    """Return 3D model for RJ45, USB-C, Conn_01x, MicroSD connectors."""
    if "RJ45" in upper:
        # Use the Amphenol RJHSE538X model — close enough to HR911105A
        # (same standard RJ45 jack envelope) for visual placement review.
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
        "2.0X2.0" in upper or _WS2812_SIZE_2020 in upper
        or "3.5X3.5" in upper or _WS2812_SIZE_3535 in upper or "P2.45" in upper
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
    """Match Sanyou-specific relay 3D model.

    Matches both KiCad standard names (containing "SANYOU") and
    JLCPCB/easyeda2kicad names (containing "SRD" — the Sanyou SRD series).
    """
    if "RELAY" in upper and ("SANYOU" in upper or "SRD" in upper):
        path = (
            f"{KICAD_3DMODEL_VAR}/Relay_THT.3dshapes/"
            "Relay_SPDT_SANYOU_SRD_Series_Form_C.step"
        )
        return Footprint3DModel(path=path)
    return None


# ---------------------------------------------------------------------------
# Generic 3D model offset correction (registry-driven)
# ---------------------------------------------------------------------------


def _apply_registry_model_offset(
    fp: Footprint,
    kicad_ref_pad1_x: float,
    kicad_ref_pad1_y: float,
) -> Footprint:
    """Align the 3D model with the actual JLCPCB pad positions.

    KiCad STEP models for SMD packages are body-centered — their origin sits
    at the pad centroid of the KiCad library footprint.  JLCPCB footprints are
    also typically body-centered (centroid at origin).  So for SMD packages
    the offset is near zero: ``offset = -JLCPCB_centroid``.

    For THT packages where the KiCad library footprint has pin-1 at origin
    (relays, DIP, connectors), the STEP model origin sits at pad 1, NOT at
    the centroid.  The stored ``kicad_ref_pad1_x/y`` tells us where pad 1
    is relative to the centroid in the parametric footprint.  Since the
    JLCPCB footprint also has centroid at origin, we need to shift the model
    by ``kicad_ref_pad1`` to move its origin from the centroid to where pad 1
    actually is.

    The offset formulas are::

        SMD (body-centered model): offset = -JLCPCB_centroid
        THT (pad-1-origin model):  offset = kicad_ref_pad1 - JLCPCB_centroid

    For mirrored layouts (SOT-223 where signal pads flip side), applies
    a 180-degree Z rotation.
    """
    if not fp.models or not fp.pads:
        return fp

    # JLCPCB pad centroid
    jxs = [p.position.x for p in fp.pads]
    jys = [p.position.y for p in fp.pads]
    j_cx = (min(jxs) + max(jxs)) / 2.0
    j_cy = (min(jys) + max(jys)) / 2.0

    # Determine if the STEP model is pin-1-at-origin (THT packages) or
    # body-centered (SMD packages).
    # THT connectors/relays: STEP origin at pad 1 → shift to JLCPCB pad 1.
    # DIP ICs: STEP origin at body center → treat as SMD (centroid).
    is_tht = "through_hole" in fp.attr
    is_dip = "DIP" in fp.lib_id.upper()
    is_pin1_origin_model = is_tht and not is_dip

    if is_pin1_origin_model:
        # STEP model origin is at pad 1.  Find JLCPCB pad 1 position
        # and shift the model there so pin 1 aligns with pad 1.
        pin1 = next((p for p in fp.pads if p.number == "1"), None)
        if pin1 is not None:
            off_x = pin1.position.x
            off_y = pin1.position.y
        else:
            off_x = -j_cx
            off_y = -j_cy
    else:
        # STEP model origin is at body center (centroid).
        # Shift model from JLCPCB origin to centroid if they differ.
        off_x = -j_cx
        off_y = -j_cy

    # Detect mirroring for directional IC packages (SOT-223 etc.)
    # where JLCPCB puts signal pads on the opposite side from KiCad.
    rot_z_correction = 0.0
    pin1 = next((p for p in fp.pads if p.number == "1"), None)
    if pin1 is not None and fp.ref.startswith("U"):
        j1_rel_x = pin1.position.x - j_cx
        # kicad_ref_pad1 is relative to KiCad origin (centroid for body-centered)
        k1_rel_x = kicad_ref_pad1_x
        fid_upper = fp.lib_id.upper()
        if (abs(k1_rel_x) > 1.0 and abs(j1_rel_x) > 1.0
                and k1_rel_x * j1_rel_x < 0
                and any(kw in fid_upper for kw in ("SOT-223", "SOT-89", "TO-252", "TO-263"))):
            rot_z_correction = 180.0
            _log.info("Mirrored layout for %s: 180° Z rotation", fp.ref)

    if abs(off_x) < 0.01 and abs(off_y) < 0.01 and rot_z_correction == 0.0:
        return fp

    _log.debug(
        "Model offset for %s: (%.2f, %.2f) rot_z=%+.0f (pin1_origin=%s)",
        fp.ref, off_x, off_y, rot_z_correction, is_pin1_origin_model,
    )
    new_models = tuple(
        Footprint3DModel(
            path=m.path,
            offset=(off_x, off_y, m.offset[2]),
            scale=m.scale,
            rotate=(m.rotate[0], m.rotate[1], m.rotate[2] + rot_z_correction),
        )
        for m in fp.models
    )
    return Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
        position=fp.position, rotation=fp.rotation, layer=fp.layer,
        pads=fp.pads, graphics=fp.graphics, texts=fp.texts,
        lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr, models=new_models,
        datasheet=fp.datasheet, description=fp.description,
        footprint_source=fp.footprint_source,
    )


# Cached registry singleton for model offset lookups.
_REGISTRY_CACHE: object = None  # lazy ComponentRegistry


def _get_component_registry() -> object:
    """Lazy-load the component registry (singleton)."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        from kicad_pipeline.validation.component_registry import ComponentRegistry
        _REGISTRY_CACHE = ComponentRegistry()
    return _REGISTRY_CACHE


def _apply_jlcpcb_model_offset(
    fp: Footprint, footprint_id: str, *, strict: bool = False,
) -> Footprint:
    """Look up the component in the registry and apply model offset correction.

    This is the entry point called from ``_try_jlcpcb_footprint`` after the
    3D model is attached.  It loads the registry, finds the matching spec,
    and delegates to :func:`_apply_registry_model_offset`.

    Args:
        strict: If True, raise instead of returning unchanged on lookup
            failure.  Used in the PRODUCTION stage to hard-block misaligned
            models from reaching manufacturing.
    """
    if not fp.models:
        return fp
    from kicad_pipeline.exceptions import KiCadPipelineError

    try:
        registry = _get_component_registry()
        spec = _lookup_registry_spec(registry, footprint_id, fp.lib_id, lcsc=fp.lcsc)
        if spec is None:
            msg = (
                f"3D model offset NOT applied for {fp.ref} "
                f"(footprint_id={footprint_id}, lcsc={fp.lcsc}): "
                f"no matching registry entry — model may be misaligned"
            )
            if strict:
                raise KiCadPipelineError(msg)
            _log.warning(msg)
            return fp
        return _apply_registry_model_offset(fp, spec.kicad_ref_pad1_x, spec.kicad_ref_pad1_y)  # type: ignore[union-attr]
    except KiCadPipelineError:
        raise  # re-raise strict mode errors
    except Exception:
        msg = (
            f"Registry model offset lookup FAILED for {fp.ref} "
            f"(footprint_id={footprint_id}): 3D model may be misaligned"
        )
        if strict:
            raise KiCadPipelineError(msg)  # noqa: B904
        _log.warning(msg, exc_info=True)
        return fp


def _lookup_registry_spec(
    registry: object, footprint_id: str, lib_id: str,
    *, lcsc: str | None = None,
) -> object:
    """Find a registry ComponentSpec matching the footprint identifiers.

    Tries LCSC match, exact match, bare name, lib_id, then substring matching
    against all registry footprint_ids.
    """
    # LCSC part number match — most reliable for JLCPCB footprints
    if lcsc:
        for comp in registry.all_components():  # type: ignore[attr-defined]
            if comp.lcsc == lcsc:
                return comp
    # Exact match
    spec = registry.get(footprint_id)  # type: ignore[union-attr]
    if spec is not None:
        return spec
    # Strip library prefix: "Relay_THT:Relay_SPDT_SANYOU..." → "Relay_SPDT_SANYOU..."
    bare = footprint_id.rsplit(":", 1)[-1] if ":" in footprint_id else footprint_id
    spec = registry.get(bare)  # type: ignore[union-attr]
    if spec is not None:
        return spec
    # Try lib_id bare name
    lib_bare = lib_id.rsplit(":", 1)[-1] if ":" in lib_id else lib_id
    spec = registry.get(lib_bare)  # type: ignore[union-attr]
    if spec is not None:
        return spec
    # Substring match: check if any registry component's footprint_id
    # is contained in the incoming footprint_id (e.g. "Relay_SPDT" in
    # "Relay_SPDT_SANYOU_SRD_Series_Form_C")
    upper_bare = bare.upper()
    upper_fid = footprint_id.upper()
    for comp in registry.all_components():  # type: ignore[union-attr]
        cfid = comp.footprint_id.upper()
        if cfid and (cfid in upper_bare or cfid in upper_fid):
            return comp
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


def _align_tht_model_to_pad1(fp: Footprint) -> Footprint:
    """Shift 3D model origin from pin 1 to pad centroid for THT footprints.

    KiCad STEP models for THT components (pin headers, terminal blocks,
    relays, DIP switches, RJ45, etc.) have their origin at pin 1.  But
    parametric footprint generators center pads around (0,0).  This function
    shifts the model so pin 1 of the STEP model aligns with pad 1 of the
    footprint.

    When the model has a non-zero Z rotation (e.g. terminal blocks at 180°),
    the offset is applied in the **rotated** coordinate frame.  A 180° rotation
    negates X and Y, so the offset must also be negated.

    Only applies to through-hole footprints with models that have zero XY offset.
    JLCPCB footprints are handled separately by ``_apply_jlcpcb_model_offset``.
    """
    if not fp.models or not fp.pads or "through_hole" not in fp.attr:
        return fp
    # DIP packages have body-centered STEP models — no pin-1 shift needed.
    if "DIP" in fp.lib_id.upper():
        return fp
    model = fp.models[0]
    # Only correct zero-offset models (already-corrected models have non-zero)
    if abs(model.offset[0]) > 0.01 or abs(model.offset[1]) > 0.01:
        return fp
    # Find pad 1
    pad1 = next((p for p in fp.pads if p.number == "1"), None)
    if pad1 is None:
        return fp
    ox, oy = pad1.position.x, pad1.position.y
    if abs(ox) < 0.01 and abs(oy) < 0.01:
        return fp  # pad 1 already at origin — no shift needed

    # Account for model's own Z rotation.  KiCad applies offset BEFORE
    # rotation, so if the model is rotated 180°, the offset direction
    # must be reversed to land at the correct position after rotation.
    import math

    rz_rad = math.radians(model.rotate[2]) if len(model.rotate) > 2 else 0.0
    if abs(rz_rad) > 0.01:
        cos_r = math.cos(rz_rad)
        sin_r = math.sin(rz_rad)
        # Inverse rotation: rotate offset into model's local frame
        rot_ox = ox * cos_r + oy * sin_r
        rot_oy = -ox * sin_r + oy * cos_r
        ox, oy = rot_ox, rot_oy

    new_models = tuple(
        Footprint3DModel(
            path=m.path,
            offset=(ox, oy, m.offset[2]),
            scale=m.scale,
            rotate=m.rotate,
        )
        for m in fp.models
    )
    return Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
        position=fp.position, rotation=fp.rotation, layer=fp.layer,
        pads=fp.pads, graphics=fp.graphics, texts=fp.texts,
        lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr, models=new_models,
        datasheet=fp.datasheet, description=fp.description,
        footprint_source=fp.footprint_source,
    )


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
}


# ---------------------------------------------------------------------------
# Public footprint generation functions
# ---------------------------------------------------------------------------


def make_smd_resistor_capacitor(
    ref: str,
    value: str,
    package: str,
    *,
    lcsc: str | None = None,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """Create an SMD resistor or capacitor footprint.

    Args:
        ref: Reference designator (e.g. "R1", "C1").
        package: Package size (e.g. "0402", "0603", "0805", "1206", "1210").
        value: Component value for display.
        lcsc: LCSC part number for JLCPCB ordering.
        layer: Board layer (default F.Cu).

    Returns:
        Footprint with 2 pads positioned according to the package dimensions.
    """
    if package not in _SMD_RC_DIMS:
        raise PCBError(f"Unsupported SMD package: {package}")

    pad_w, pad_h, pitch, body_w, body_h = _SMD_RC_DIMS[package]

    # Create pads: left and right of origin
    pads = [
        Pad(
            number="1",
            pad_type="smd",
            shape="rect",
            position=Point(-pitch/2, 0.0),
            size_x=pad_w,
            size_y=pad_h,
            layers=(layer,),
        ),
        Pad(
            number="2",
            pad_type="smd",
            shape="rect",
            position=Point(pitch/2, 0.0),
            size_x=pad_w,
            size_y=pad_h,
            layers=(layer,),
        ),
    ]

    # Create courtyard rectangle
    courtyard_margin = 0.25  # 0.25mm margin
    courtyard_w = body_w + 2 * courtyard_margin
    courtyard_h = body_h + 2 * courtyard_margin
    graphics = [
        FootprintLine(
            start=Point(-courtyard_w/2, -courtyard_h/2),
            end=Point(courtyard_w/2, -courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(courtyard_w/2, -courtyard_h/2),
            end=Point(courtyard_w/2, courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(courtyard_w/2, courtyard_h/2),
            end=Point(-courtyard_w/2, courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(-courtyard_w/2, courtyard_h/2),
            end=Point(-courtyard_w/2, -courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        # Silkscreen outline of the body
        FootprintLine(
            start=Point(-body_w/2, -body_h/2),
            end=Point(body_w/2, -body_h/2),
            layer=LAYER_F_SILKSCREEN,
            width=0.12,
        ),
        FootprintLine(
            start=Point(body_w/2, -body_h/2),
            end=Point(body_w/2, body_h/2),
            layer=LAYER_F_SILKSCREEN,
            width=0.12,
        ),
        FootprintLine(
            start=Point(body_w/2, body_h/2),
            end=Point(-body_w/2, body_h/2),
            layer=LAYER_F_SILKSCREEN,
            width=0.12,
        ),
        FootprintLine(
            start=Point(-body_w/2, body_h/2),
            end=Point(-body_w/2, -body_h/2),
            layer=LAYER_F_SILKSCREEN,
            width=0.12,
        ),
    ]

    # Create 3D model
    lib_id = _KICAD_RESISTOR_LIB_IDS.get(package, f"Resistor_SMD:R_{package}_1608Metric")
    models = []
    model = _model_for_package(lib_id, layer)
    if model:
        models.append(model)

    # Create reference and value texts
    texts = [
        FootprintText(
            text_type="reference",
            text=ref,
            position=Point(0.0, body_h/2 + 0.5),
            layer=LAYER_F_SILKSCREEN,
            rotation=0.0,
            effects_size=1.0,
        ),
        FootprintText(
            text_type="value",
            text=value,
            position=Point(0.0, -body_h/2 - 0.5),
            layer=LAYER_F_FAB,
            rotation=0.0,
            effects_size=0.8,
        ),
    ]

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        rotation=0.0,
        layer=layer,
        pads=pads,
        graphics=graphics,
        texts=texts,
        lcsc=lcsc,
        uuid="",
        attr=set(),
        models=tuple(models),
        datasheet="",
        description=f"SMD {package} resistor/capacitor",
        footprint_source="parametric",
    )


def make_smd_led(
    ref: str,
    value: str,
    package: str,
    *,
    lcsc: str | None = None,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """Create an SMD LED footprint with polarity markings.

    Args:
        ref: Reference designator (e.g. "D1").
        package: Package size (e.g. "0402", "0603", "0805", "1206").
        value: Component value for display.
        lcsc: LCSC part number for JLCPCB ordering.
        layer: Board layer (default F.Cu).

    Returns:
        Footprint with 2 pads and cathode marking (line on silkscreen).
    """
    if package not in _SMD_RC_DIMS:
        raise ValueError(f"Unsupported SMD LED package: {package}")

    pad_w, pad_h, pitch, body_w, body_h = _SMD_RC_DIMS[package]

    # Create pads: anode (pin 1) left, cathode (pin 2) right
    pads = [
        Pad(
            number="1",  # Anode
            pad_type="smd",
            shape="rect",
            position=Point(-pitch/2, 0.0),
            size_x=pad_w,
            size_y=pad_h,
            layers=(layer,),
        ),
        Pad(
            number="2",  # Cathode
            pad_type="smd",
            shape="rect",
            position=Point(pitch/2, 0.0),
            size_x=pad_w,
            size_y=pad_h,
            layers=(layer,),
        ),
    ]

    # Create courtyard rectangle
    courtyard_margin = 0.25
    courtyard_w = body_w + 2 * courtyard_margin
    courtyard_h = body_h + 2 * courtyard_margin
    graphics = [
        FootprintLine(
            start=Point(-courtyard_w/2, -courtyard_h/2),
            end=Point(courtyard_w/2, -courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(courtyard_w/2, -courtyard_h/2),
            end=Point(courtyard_w/2, courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(courtyard_w/2, courtyard_h/2),
            end=Point(-courtyard_w/2, courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(-courtyard_w/2, courtyard_h/2),
            end=Point(-courtyard_w/2, -courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
    ]

    # Add cathode polarity marking (line on silkscreen near cathode pad)
    cathode_mark_x = pitch/2 + pad_w/2 + 0.2  # Slightly right of cathode pad
    graphics.append(
        FootprintLine(
            start=Point(cathode_mark_x, -body_h/4),
            end=Point(cathode_mark_x, body_h/4),
            layer=LAYER_F_SILKSCREEN,
            width=0.15,
        ),
    )

    # Create 3D model
    lib_id = _KICAD_LED_LIB_IDS.get(package, f"LED_SMD:LED_{package}_1608Metric")
    models = []
    model = _model_for_package(lib_id, layer)
    if model:
        models.append(model)

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        rotation=0.0,
        layer=layer,
        pads=pads,
        graphics=graphics,
        texts=[],
        lcsc=lcsc,
        uuid="",
        attr=set(),
        models=tuple(models),
        datasheet="",
        description=f"SMD {package} LED",
        footprint_source="parametric",
    )


def make_rj45(
    ref: str,
    *,
    value: str = "",
    lcsc: str | None = None,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """Create an RJ45 connector footprint (HR911105A style).

    Args:
        ref: Reference designator (e.g. "J1").
        value: Component value for display.
        lcsc: LCSC part number for JLCPCB ordering.
        layer: Board layer (default F.Cu).

    Returns:
        Footprint with 16 pads: 8 signal, 4 LED, 2 shield, 2 NPTH mounting.
    """
    pads = []

    # Signal pads (1-8): staggered zigzag pattern
    for i, (x, y) in enumerate(_RJ45_SIGNAL_POSITIONS, 1):
        pads.append(
            Pad(
                number=str(i),
                pad_type="thru_hole",
                shape="circle",
                position=Point(x, y),
                size_x=_RJ45_SIGNAL_PAD_MM,
                size_y=_RJ45_SIGNAL_PAD_MM,
                drill_diameter=_RJ45_SIGNAL_DRILL_MM,
                layers=(LAYER_F_CU, LAYER_B_CU),
            ),
        )

    # LED pads (9-12)
    for i, (x, y) in enumerate(_RJ45_LED_POSITIONS, 9):
        pads.append(
            Pad(
                number=str(i),
                pad_type="thru_hole",
                shape="circle",
                position=Point(x, y),
                size_x=_RJ45_LED_PAD_MM,
                size_y=_RJ45_LED_PAD_MM,
                drill_diameter=_RJ45_LED_DRILL_MM,
                layers=(LAYER_F_CU, LAYER_B_CU),
            ),
        )

    # Shield pads (13-14): connect to connector shell
    shield_y = 0.0  # Centered vertically
    shield_x_left = -4.5
    shield_x_right = 9.5
    pads.extend([
        Pad(
            number="13",
            pad_type="thru_hole",
            shape="oval",
            position=Point(shield_x_left, shield_y),
            size_x=2.0,
            size_y=1.0,
            drill_diameter=1.0,
            layers=(LAYER_F_CU, LAYER_B_CU),
        ),
        Pad(
            number="14",
            pad_type="thru_hole",
            shape="oval",
            position=Point(shield_x_right, shield_y),
            size_x=2.0,
            size_y=1.0,
            drill_diameter=1.0,
            layers=(LAYER_F_CU, LAYER_B_CU),
        ),
    ])

    # NPTH mounting holes (15-16)
    npth_y = 6.6  # Same as LED pins
    npth_x_left = -1.5
    npth_x_right = 8.5
    pads.extend([
        Pad(
            number="15",
            pad_type="np_thru_hole",
            shape="circle",
            position=Point(npth_x_left, npth_y),
            size_x=2.0,
            size_y=2.0,
            drill_diameter=1.5,
            layers=(),  # NPTH pads have no copper layers
        ),
        Pad(
            number="16",
            pad_type="np_thru_hole",
            shape="circle",
            position=Point(npth_x_right, npth_y),
            size_x=2.0,
            size_y=2.0,
            drill_diameter=1.5,
            layers=(),  # NPTH pads have no copper layers
        ),
    ])

    # Create courtyard outline
    courtyard_margin = 0.5
    min_x = min(p.position.x - max(p.size_x, p.size_y)/2 for p in pads if p.layers)
    max_x = max(p.position.x + max(p.size_x, p.size_y)/2 for p in pads if p.layers)
    min_y = min(p.position.y - max(p.size_x, p.size_y)/2 for p in pads if p.layers)
    max_y = max(p.position.y + max(p.size_x, p.size_y)/2 for p in pads if p.layers)

    graphics = [
        FootprintLine(
            start=Point(min_x - courtyard_margin, min_y - courtyard_margin),
            end=Point(max_x + courtyard_margin, min_y - courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(max_x + courtyard_margin, min_y - courtyard_margin),
            end=Point(max_x + courtyard_margin, max_y + courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(max_x + courtyard_margin, max_y + courtyard_margin),
            end=Point(min_x - courtyard_margin, max_y + courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(min_x - courtyard_margin, max_y + courtyard_margin),
            end=Point(min_x - courtyard_margin, min_y - courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
    ]

    # Create 3D model
    lib_id = "Connector:RJ45_Amphenol_RJHSE538X"
    models = []
    model = _model_for_package(lib_id, layer)
    if model:
        models.append(model)

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        rotation=0.0,
        layer=layer,
        pads=pads,
        graphics=graphics,
        texts=[],
        lcsc=lcsc,
        uuid="",
        attr={"through_hole"},
        models=tuple(models),
        datasheet="",
        description="RJ45 Ethernet connector",
        footprint_source="parametric",
    )


def make_sot23(
    ref: str,
    value: str,
    variant: str = "SOT-23",
    *,
    lcsc: str | None = None,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """Create a SOT-23 transistor footprint.

    Args:
        ref: Reference designator (e.g. "Q1").
        variant: SOT-23 variant ("SOT-23", "SOT-23-5", "SOT-23-6").
        value: Component value for display.
        lcsc: LCSC part number for JLCPCB ordering.
        layer: Board layer (default F.Cu).

    Returns:
        Footprint with pads positioned according to the variant.
    """
    if variant not in _SOT23_VARIANTS:
        raise ValueError(f"Unsupported SOT-23 variant: {variant}")

    pad_w, pad_h, pin_coords = _SOT23_VARIANTS[variant]

    pads = []
    for i, (x, y) in enumerate(pin_coords, 1):
        pads.append(
            Pad(
                number=str(i),
                pad_type="smd",
                shape="rect",
                position=Point(x, y),
                size_x=pad_w,
                size_y=pad_h,
                layers=(layer,),
            ),
        )

    # Create courtyard rectangle (approximate body size)
    body_w = 3.0  # Typical SOT-23 body width
    body_h = 1.4  # Typical SOT-23 body height
    courtyard_margin = 0.25
    graphics = [
        FootprintLine(
            start=Point(-body_w/2 - courtyard_margin, -body_h/2 - courtyard_margin),
            end=Point(body_w/2 + courtyard_margin, -body_h/2 - courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(body_w/2 + courtyard_margin, -body_h/2 - courtyard_margin),
            end=Point(body_w/2 + courtyard_margin, body_h/2 + courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(body_w/2 + courtyard_margin, body_h/2 + courtyard_margin),
            end=Point(-body_w/2 - courtyard_margin, body_h/2 + courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(-body_w/2 - courtyard_margin, body_h/2 + courtyard_margin),
            end=Point(-body_w/2 - courtyard_margin, -body_h/2 - courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
    ]

    # Create 3D model
    lib_id = f"Package_TO_SOT_SMD:{variant}"
    models = []
    model = _model_for_package(lib_id, layer)
    if model:
        models.append(model)

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        rotation=0.0,
        layer=layer,
        pads=pads,
        graphics=graphics,
        texts=[],
        lcsc=lcsc,
        uuid="",
        attr=set(),
        models=tuple(models),
        datasheet="",
        description=f"{variant} transistor package",
        footprint_source="parametric",
    )


def make_through_hole_2pin(
    ref: str,
    value: str,
    *,
    pitch_mm: float = 2.54,
    drill_mm: float = 0.8,
    lcsc: str | None = None,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """Create a through-hole 2-pin component footprint.

    Args:
        ref: Reference designator (e.g. "D1", "R1").
        pitch: Distance between pin centers (mm).
        value: Component value for display.
        lcsc: LCSC part number for JLCPCB ordering.
        layer: Board layer (default F.Cu).

    Returns:
        Footprint with 2 through-hole pads.
    """
    drill_dia = 0.8  # Standard THT drill diameter
    pad_dia = 1.8    # Standard THT pad diameter

    pads = [
        Pad(
            number="1",
            pad_type="thru_hole",
            shape="circle",
            position=Point(-pitch_mm/2, 0.0),
            size_x=pad_dia,
            size_y=pad_dia,
            drill_diameter=drill_mm,
            layers=(LAYER_F_CU, LAYER_B_CU),
        ),
        Pad(
            number="2",
            pad_type="thru_hole",
            shape="circle",
            position=Point(pitch_mm/2, 0.0),
            size_x=pad_dia,
            size_y=pad_dia,
            drill_diameter=drill_mm,
            layers=(LAYER_F_CU, LAYER_B_CU),
        ),
    ]

    # Create courtyard rectangle
    courtyard_margin = 0.5
    courtyard_w = pitch_mm + pad_dia + 2 * courtyard_margin
    courtyard_h = pad_dia + 2 * courtyard_margin
    graphics = [
        FootprintLine(
            start=Point(-courtyard_w/2, -courtyard_h/2),
            end=Point(courtyard_w/2, -courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(courtyard_w/2, -courtyard_h/2),
            end=Point(courtyard_w/2, courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(courtyard_w/2, courtyard_h/2),
            end=Point(-courtyard_w/2, courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(-courtyard_w/2, courtyard_h/2),
            end=Point(-courtyard_w/2, -courtyard_h/2),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
    ]

    # Generic lib_id for THT components
    lib_id = "Connector:PinHeader_1x02_P2.54mm_Vertical"

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        rotation=0.0,
        layer=layer,
        pads=pads,
        graphics=graphics,
        texts=[],
        lcsc=lcsc,
        uuid="",
        attr={"through_hole"},
        models=(),
        datasheet="",
        description=f"Through-hole 2-pin component, {pitch_mm}mm pitch",
        footprint_source="parametric",
    )


def make_usbc_connector(
    ref: str,
    *,
    value: str = "",
    lcsc: str | None = None,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """Create a USB-C connector footprint.

    Args:
        ref: Reference designator (e.g. "J2").
        value: Component value for display.
        lcsc: LCSC part number for JLCPCB ordering.
        layer: Board layer (default F.Cu).

    Returns:
        Footprint with USB-C pads positioned according to standard.
    """
    pads = []

    # Create pads from the USB-C pad definitions
    for x, y, w, h, name in _USBC_PADS:
        pads.append(
            Pad(
                number=name,
                pad_type="smd",
                shape="rect",
                position=Point(x, y),
                size_x=w,
                size_y=h,
                layers=(layer,),
            ),
        )

    # Create courtyard outline
    courtyard_margin = 0.5
    min_x = min(p.position.x - p.size_x/2 for p in pads)
    max_x = max(p.position.x + p.size_x/2 for p in pads)
    min_y = min(p.position.y - p.size_y/2 for p in pads)
    max_y = max(p.position.y + p.size_y/2 for p in pads)

    graphics = [
        FootprintLine(
            start=Point(min_x - courtyard_margin, min_y - courtyard_margin),
            end=Point(max_x + courtyard_margin, min_y - courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(max_x + courtyard_margin, min_y - courtyard_margin),
            end=Point(max_x + courtyard_margin, max_y + courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(max_x + courtyard_margin, max_y + courtyard_margin),
            end=Point(min_x - courtyard_margin, max_y + courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
        FootprintLine(
            start=Point(min_x - courtyard_margin, max_y + courtyard_margin),
            end=Point(min_x - courtyard_margin, min_y - courtyard_margin),
            layer=LAYER_F_COURTYARD,
            width=0.05,
        ),
    ]

    # Create 3D model
    lib_id = "Connector_USB:USB_C_Receptacle_GCT_USB4105-xx-A_16P_TopMnt_Horizontal"
    models = []
    model = _model_for_package(lib_id, layer)
    if model:
        models.append(model)

    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        rotation=0.0,
        layer=layer,
        pads=pads,
        graphics=graphics,
        texts=[],
        lcsc=lcsc,
        uuid="",
        attr=set(),
        models=tuple(models),
        datasheet="",
        description="USB-C receptacle connector",
        footprint_source="parametric",
    )