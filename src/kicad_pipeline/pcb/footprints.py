"""Parametric footprint generators for the kicad-ai-pipeline.

Generates :class:`~kicad_pipeline.models.pcb.Footprint` objects for common
passive, semiconductor, and connector packages without any external library
dependency.
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
    # STEP model origin is at the center of the block; footprint pin 1 is at (0,0).
    # Offset = half the total pad span to align model center with pad centroid.
    offset_x = (pin_count - 1) * pitch / 2.0
    return Footprint3DModel(
        path=path,
        offset=(offset_x, 0.0, 0.0),
        rotate=(0.0, 0.0, 180.0),
    )


def _model_esp32(
    name: str, upper: str,
) -> Footprint3DModel | None:
    """Return 3D model for ESP32/RF modules.

    The KiCad standard ESP32-S3-WROOM-1.step model has its internal origin
    aligned with the KiCad standard library footprint origin, which is
    ~3.63mm north (−Y in module coords) of the pad-field centroid.
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

    # Tactile switches — SMD vs THT
    lib_upper = lib_id.upper()
    if upper.startswith("SW_PUSH") or (upper.startswith("SW_") and "SPST" in upper):
        if "SMD" in lib_upper or "SMD" in upper:
            path = (
                f"{KICAD_3DMODEL_VAR}/Button_Switch_SMD.3dshapes/"
                "SW_SPST_EVQPE1.step"
            )
        else:
            # SMD pad geometry (e.g. make_tact_switch) — use SMD model, no rotation.
            # The easyeda footprint pads are at ±3.0 X / ±1.85 Y which is the
            # same aspect-ratio orientation as the TL3305A model (wider in X).
            return Footprint3DModel(
                path=f"{KICAD_3DMODEL_VAR}/Button_Switch_SMD.3dshapes/SW_SPST_TL3305A.step",
                rotate=(0.0, 0.0, 0.0),
            )
        return Footprint3DModel(path=path)
    return None


def _model_connector(
    name: str, upper: str, layer: str,
) -> Footprint3DModel | None:
    """Return 3D model for RJ45, USB-C, Conn_01x, MicroSD connectors."""
    if "RJ45" in upper:
        # The KiCad library only has Amphenol RJHSE538X which doesn't match
        # the HR911105A (HanRun) connector used by JLCPCB.  Skip 3D model
        # rather than show a wrong body.  TODO: add HR911105A STEP model.
        _log.debug("RJ45: no matching 3D model for HR911105A, skipping")
        return None

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
    """Match Sanyou-specific relay 3D model."""
    if "RELAY" in upper and "SANYOU" in upper:
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

    KiCad STEP models are body-centered — their origin sits at the pad
    centroid of the KiCad library footprint.  JLCPCB footprints are also
    typically body-centered (centroid at origin).  So for most packages
    the offset is near zero.

    For packages where the KiCad library footprint is NOT body-centered
    (pin-1 at origin, like relays or DIP), the model origin sits at pad 1,
    not at the centroid.  The stored ``kicad_ref_pad1_x/y`` tells us where
    pad 1 is in the KiCad library, which lets us compute the KiCad centroid
    and derive the correct offset.

    For mirrored layouts (SOT-223 where signal pads flip side), applies
    a 180-degree Z rotation.

    The offset formula is::

        offset = KiCad_centroid - JLCPCB_centroid
    """
    if not fp.models or not fp.pads:
        return fp

    # JLCPCB pad centroid
    jxs = [p.position.x for p in fp.pads]
    jys = [p.position.y for p in fp.pads]
    j_cx = (min(jxs) + max(jxs)) / 2.0
    j_cy = (min(jys) + max(jys)) / 2.0

    # KiCad pad centroid — for body-centered packages this is ≈(0,0).
    # For pin-1-origin packages, we can estimate it: the model dispatch
    # already stripped hardcoded offsets, so the model origin is at the
    # KiCad footprint origin.  Just use -JLCPCB_centroid.
    off_x = -j_cx
    off_y = -j_cy

    # Detect mirroring for directional IC packages (SOT-223 etc.)
    # where JLCPCB puts signal pads on the opposite side from KiCad.
    rot_z_correction = 0.0
    pin1 = next((p for p in fp.pads if p.number == "1"), None)
    if pin1 is not None and fp.ref.startswith("U"):
        j1_rel_x = pin1.position.x - j_cx
        # kicad_ref_pad1 is relative to KiCad origin (≈ centroid for body-centered)
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
        "Model offset for %s: (%.2f, %.2f) rot_z=%+.0f (centroid-based)",
        fp.ref, off_x, off_y, rot_z_correction,
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
    global _REGISTRY_CACHE  # noqa: PLW0603
    if _REGISTRY_CACHE is None:
        from kicad_pipeline.validation.component_registry import ComponentRegistry
        _REGISTRY_CACHE = ComponentRegistry()
    return _REGISTRY_CACHE


def _apply_jlcpcb_model_offset(fp: Footprint, footprint_id: str) -> Footprint:
    """Look up the component in the registry and apply model offset correction.

    This is the entry point called from ``_try_jlcpcb_footprint`` after the
    3D model is attached.  It loads the registry, finds the matching spec,
    and delegates to :func:`_apply_registry_model_offset`.
    """
    if not fp.models:
        return fp
    try:
        registry = _get_component_registry()
        spec = _lookup_registry_spec(registry, footprint_id, fp.lib_id)
        if spec is None:
            return fp
        return _apply_registry_model_offset(fp, spec.kicad_ref_pad1_x, spec.kicad_ref_pad1_y)  # type: ignore[union-attr]
    except Exception:
        _log.debug("Registry model offset lookup failed for %s", fp.ref, exc_info=True)
        return fp


def _lookup_registry_spec(
    registry: object, footprint_id: str, lib_id: str,
) -> object:
    """Find a registry ComponentSpec matching the footprint identifiers.

    Tries exact match, bare name, lib_id, then substring matching
    against all registry footprint_ids.
    """
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
# Distance from top body edge to center of first pad (antenna clearance).
# KiCad official footprint: pin 1 at (-8.75, -5.26) relative to body center,
# which gives TOP_MARGIN = body_h/2 + pin1_y - pad_h/2 = 12.75 - 5.26 - 0.6 = 6.89
_ESP32_TOP_MARGIN: float = 6.89
_ESP32_GND_PAD_SIZE: float = 6.7

# ESP32-S3-WROOM-1 thermal/GND pad number — KiCad pad identifier string.
# The exposed GND pad on the module underside is designated pad "41"
# (one past the 40 perimeter castellations) per the datasheet.
_ESP32_THERMAL_PAD_NUMBER = "41"

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

# Extension past the module body edge for the antenna keepout.  The antenna
# radiates beyond the module body and copper in that zone degrades RF
# performance.  3.5mm per ESP32-S3-WROOM-1 datasheet recommended layout.
_ESP32_ANTENNA_KEEPOUT_EXTENSION_MM: float = 3.5

# Vertical offset for the GND pad centre.  The antenna occupies the top ~8mm
# of the 25.5mm body so the pad field (and GND pad) is shifted south by half
# the antenna depth to centre it on the active silicon area.
_ESP32_GND_PAD_Y_OFFSET: float = 2.5

# Crystal oscillator dimensions (mm)
_CRYSTAL_PAD_W: float = 1.2
_CRYSTAL_PAD_H: float = 1.0
_CRYSTAL_SHIELD_MIN_HEIGHT: float = 2.0
# Default body size for SMD_3215 crystal (3.2x1.5mm per IEC 61671 designator)
_CRYSTAL_SMD3215_DEFAULT_W_MM: float = 3.2
_CRYSTAL_SMD3215_DEFAULT_H_MM: float = 1.5

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

# WS2812B LED size variant codes — EIA package designators (tenths of mm).
# "5050" = 5.0x5.0mm body, "3535" = 3.5x3.5mm, "2020" = 2.0x2.0mm.
_WS2812_SIZE_5050 = "5050"
_WS2812_SIZE_3535 = "3535"
_WS2812_SIZE_2020 = "2020"

# WS2812B LED size variants: (pad_w, pad_h, x_pitch, y_pitch, body_w, body_h)
_WS2812B_DIMS: dict[str, tuple[float, float, float, float, float, float]] = {
    _WS2812_SIZE_2020: (0.7, 0.5, 0.75, 0.55, 2.6, 2.6),
    _WS2812_SIZE_3535: (1.0, 0.8, 1.65, 1.05, 4.0, 4.0),
    _WS2812_SIZE_5050: (1.5, 1.0, 2.45, 1.6, 5.4, 5.4),
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

# Courtyard line width (KiCad standard, differs from silkscreen width) (mm)
_COURTYARD_LINE_WIDTH_MM: float = 0.05

# Silkscreen side-mark geometry
_SILK_PAD_MASK_EXPANSION_MM: float = 0.35  # push silk outside pad edge + mask expansion
_SILK_MARK_HEIGHT_FRACTION: float = 0.45   # fraction of half-height for silk marks

# LED polarity triangle size relative to pad height
_LED_POLARITY_TRIANGLE_RATIO: float = 0.4

# Value text offset below/above body on FAB layer (compact packages) (mm)
_VAL_TEXT_OFFSET_MM: float = 1.0

# ESP32 pin label constants
_ESP32_PIN_LABEL_SIZE: float = 0.5    # fab-layer text height (mm)
_ESP32_PIN_LABEL_OFFSET: float = 1.6  # inward offset from pad centre (mm)

# Crystal pitch geometry adjustment (mm)
_CRYSTAL_PITCH_ADJUST_MM: float = 0.4

# SOT-23 body bounding box padding beyond pad extents (mm)
_SOT23_BODY_PADDING_MM: float = 0.2

# IC silkscreen column pitch multiplier (silk marks use 1.6x col_pitch width)
_IC_SILK_COL_PITCH_FACTOR: float = 1.6

# DIP package minimum body height (mm)
_DIP_MIN_BODY_HEIGHT_MM: float = 3.0

# MicroSD shield pad positions (x, y) in mm
_MICROSD_SHIELD_PAD_LEFT_X: float = -7.0
_MICROSD_SHIELD_PAD_RIGHT_X: float = 7.0
_MICROSD_SHIELD_PAD_Y: float = -1.5

# QFN/DFN thermal pad sizing
_QFN_THERMAL_PAD_MIN_MM: float = 2.0    # minimum exposed pad size (mm)
_QFN_THERMAL_PAD_SCALE: float = 0.08   # exposed pad size = pin_count * scale

# Exposed thermal/ground pad detection.
# Pin names (case-insensitive) that indicate an exposed thermal pad underneath
# an IC body. Common on SOIC/MSOP/TSSOP PowerPAD packages (e.g. TPS54331 DGQ).
_THERMAL_PAD_PIN_NAMES: frozenset[str] = frozenset({
    "PAD", "EP", "EPAD", "GND_PAD", "THERMAL", "POWERPAD",
})

# Default exposed-pad size as a fraction of the IC body for dual-row packages
# (SOIC/MSOP/TSSOP). QFN/DFN use _QFN_THERMAL_PAD_SCALE instead.
_DUAL_ROW_THERMAL_PAD_BODY_FRACTION: float = 0.60

# 3D model paths for thermal-pad (exposed pad) package variants.
# Maps uppercase package prefix to the KiCad 3D model relative path.
_THERMAL_3D_MODEL_VARIANTS: dict[str, str] = {
    "SOIC-8": "Package_SO.3dshapes/SOIC-8-1EP_3.9x4.9mm_P1.27mm_EP2.29x3mm.step",
    "SOIC-16": "Package_SO.3dshapes/SOIC-16-1EP_3.9x9.9mm_P1.27mm_EP2.29x3mm.step",
    "MSOP-8": "Package_SO.3dshapes/MSOP-8-1EP_3x3mm_P0.65mm_EP1.68x1.88mm.step",
    "MSOP-10": "Package_SO.3dshapes/MSOP-10-1EP_3x3mm_P0.5mm_EP1.68x1.88mm.step",
    "TSSOP-16": "Package_SO.3dshapes/TSSOP-16-1EP_4.4x5mm_P0.65mm_EP3x3mm.step",
    "TSSOP-20": "Package_SO.3dshapes/TSSOP-20-1EP_4.4x6.5mm_P0.65mm_EP3x3mm.step",
}


def _find_thermal_pad_pin(pins: tuple[Pin, ...]) -> Pin | None:
    """Return the thermal/exposed pad pin if present, else ``None``.

    Detects pins whose name matches one of :data:`_THERMAL_PAD_PIN_NAMES`
    (case-insensitive) and whose electrical type is ``POWER_IN``.
    """
    from kicad_pipeline.models.requirements import PinType

    for pin in pins:
        if pin.name.upper() in _THERMAL_PAD_PIN_NAMES and pin.pin_type == PinType.POWER_IN:
            return pin
    return None


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
        FootprintLine(
            start=Point(cx - hw, cy - hh), end=Point(cx + hw, cy - hh), layer=layer, width=w),
        FootprintLine(
            start=Point(cx + hw, cy - hh), end=Point(cx + hw, cy + hh), layer=layer, width=w),
        FootprintLine(
            start=Point(cx + hw, cy + hh), end=Point(cx - hw, cy + hh), layer=layer, width=w),
        FootprintLine(
            start=Point(cx - hw, cy + hh), end=Point(cx - hw, cy - hh), layer=layer, width=w),
    )


def _ensure_courtyard(fp: Footprint) -> Footprint:
    """Add courtyard graphics if the footprint has none.

    Generates a rectangular courtyard from the pad bounding box plus
    IPC clearance.  Returns the footprint unchanged if courtyard lines
    already exist.
    """
    crtyd_layer = LAYER_F_COURTYARD if fp.layer == LAYER_F_CU else LAYER_B_COURTYARD
    has_crtyd = any(
        getattr(g, "layer", "") in (LAYER_F_COURTYARD, LAYER_B_COURTYARD)
        for g in fp.graphics
    )
    if has_crtyd or not fp.pads:
        return fp

    # Compute pad bounding box
    xs = [p.position.x for p in fp.pads]
    ys = [p.position.y for p in fp.pads]
    # Include half-pad size in the extent
    half_pads = [max(p.size_x, p.size_y) / 2.0 for p in fp.pads]
    pad_min_x = min(x - hp for x, hp in zip(xs, half_pads))
    pad_max_x = max(x + hp for x, hp in zip(xs, half_pads))
    pad_min_y = min(y - hp for y, hp in zip(ys, half_pads))
    pad_max_y = max(y + hp for y, hp in zip(ys, half_pads))

    cx = (pad_min_x + pad_max_x) / 2.0
    cy = (pad_min_y + pad_max_y) / 2.0
    body_w = pad_max_x - pad_min_x
    body_h = pad_max_y - pad_min_y

    crtyd = _courtyard_rect(body_w, body_h, layer=crtyd_layer, cx=cx, cy=cy)
    return Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
        position=fp.position, rotation=fp.rotation, layer=fp.layer,
        pads=fp.pads, graphics=(*fp.graphics, *crtyd), texts=fp.texts,
        lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr, models=fp.models,
        datasheet=fp.datasheet, description=fp.description,
        footprint_source=getattr(fp, "footprint_source", ""),
        fp_zones=fp.fp_zones,
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
        hw = max(hw, pad_edge_x + _SILK_PAD_MASK_EXPANSION_MM)
    hh = body_h / 2.0 * _SILK_MARK_HEIGHT_FRACTION  # 45 % of half-height (avoid pad mask)
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
    tri_size = pad_h * _LED_POLARITY_TRIANGLE_RATIO
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
        _val_text(value, body_h / 2.0 + _VAL_TEXT_OFFSET_MM, LAYER_F_FAB),
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
        _val_text(value, body_h / 2.0 + _VAL_TEXT_OFFSET_MM, LAYER_F_FAB),
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
        _ref_text(ref, -(body / 2.0 + _TEXT_OFFSET_LARGE), LAYER_F_SILKSCREEN),
        _val_text(value, body / 2.0 + _TEXT_OFFSET_LARGE, LAYER_F_FAB),
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
    4 pads (two per terminal), body 5.1x5.1mm.
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
    # Body: 5.1x5.1mm, pad size: 1.5x3.0mm
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
    graphics = _relay_spdt_graphics(cx, cy, body_w, body_h)
    texts = (
        _ref_text(ref, cy - (body_h / 2.0 + _TEXT_OFFSET_LARGE), LAYER_F_SILKSCREEN),
        _val_text(value, cy + body_h / 2.0 + _TEXT_OFFSET_LARGE, LAYER_F_FAB),
    )
    lib_id = "Relay_THT:Relay_SPDT_SANYOU_SRD_Series_Form_C"
    model = _model_for_package(lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=LAYER_F_CU, pads=pads, graphics=graphics, texts=texts,
        attr="through_hole", models=models,
    )


def _relay_spdt_graphics(
    cx: float, cy: float, body_w: float, body_h: float,
) -> tuple[FootprintLine, ...]:
    """Build courtyard and U-shaped Edge.Cuts graphics for make_relay_spdt."""
    hw = body_w / 2.0 + PCB_COURTYARD_CLEARANCE_MM
    hh = body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM
    # U-shaped isolation cutout around COM pin.  The closed end (bottom
    # of the U) is an arc that curves around the COM pad.  The two arms
    # and open end are straight parallel lines.
    #
    #         B────C  top arm (straight)
    #        /      \
    #   outer arc    inner arc  ← arcs around COM
    #        \      /
    #         G────F  bottom arm (straight)
    #
    cutout_w = _RELAY_CUTOUT_WIDTH
    cutout_clr = _RELAY_CUTOUT_CLEARANCE
    com_pad_r = _RELAY_CONTACT_PAD_DIAM / 2.0
    u_half = com_pad_r + cutout_clr
    sw = cutout_w / 2.0  # half channel width
    # Closed wall on left (toward contacts), open on right (safe side)
    closed_x = -(com_pad_r + cutout_clr)
    open_x = com_pad_r + cutout_clr
    _lw = 0.05  # thin outline
    # Outer arc radius and inner arc radius from COM center (0,0)
    outer_r = u_half + sw
    inner_r = u_half - sw
    # Points for arms (straight segments)
    b = Point(open_x, -u_half - sw)   # top arm outer end
    c = Point(open_x, -u_half + sw)   # top arm inner end
    f = Point(open_x, u_half - sw)    # bottom arm inner end
    g = Point(open_x, u_half + sw)    # bottom arm outer end
    # Arc endpoints on the closed side
    outer_top = Point(0.0, -outer_r)      # outer arc start (top)
    outer_bot = Point(0.0, outer_r)       # outer arc end (bottom)
    outer_mid = Point(closed_x - sw, 0.0) # outer arc midpoint (leftmost)
    inner_top = Point(0.0, -inner_r)      # inner arc start (top)
    inner_bot = Point(0.0, inner_r)       # inner arc end (bottom)
    inner_mid = Point(closed_x + sw, 0.0) # inner arc midpoint (leftmost)
    return (
        FootprintLine(
            start=Point(cx - hw, cy - hh), end=Point(cx + hw, cy - hh),
            layer=LAYER_F_COURTYARD, width=_COURTYARD_LINE_WIDTH_MM,
        ),
        FootprintLine(
            start=Point(cx + hw, cy - hh), end=Point(cx + hw, cy + hh),
            layer=LAYER_F_COURTYARD, width=_COURTYARD_LINE_WIDTH_MM,
        ),
        FootprintLine(
            start=Point(cx + hw, cy + hh), end=Point(cx - hw, cy + hh),
            layer=LAYER_F_COURTYARD, width=_COURTYARD_LINE_WIDTH_MM,
        ),
        FootprintLine(
            start=Point(cx - hw, cy + hh), end=Point(cx - hw, cy - hh),
            layer=LAYER_F_COURTYARD, width=_COURTYARD_LINE_WIDTH_MM,
        ),
        # Top arm: outer_top → b (straight)
        FootprintLine(start=outer_top, end=b, layer=LAYER_EDGE_CUTS, width=_lw),
        # Open end top: b → c (straight, closes arm tip)
        FootprintLine(start=b, end=c, layer=LAYER_EDGE_CUTS, width=_lw),
        # Top arm: c → inner_top (straight)
        FootprintLine(start=c, end=inner_top, layer=LAYER_EDGE_CUTS, width=_lw),
        # Inner arc around COM (top → bottom, curving left)
        FootprintArc(
            start=inner_top, mid=inner_mid, end=inner_bot,
            layer=LAYER_EDGE_CUTS, width=_lw,
        ),
        # Bottom arm: inner_bot → f (straight)
        FootprintLine(start=inner_bot, end=f, layer=LAYER_EDGE_CUTS, width=_lw),
        # Open end bottom: f → g (straight, closes arm tip)
        FootprintLine(start=f, end=g, layer=LAYER_EDGE_CUTS, width=_lw),
        # Bottom arm: g → outer_bot (straight)
        FootprintLine(start=g, end=outer_bot, layer=LAYER_EDGE_CUTS, width=_lw),
        # Outer arc around COM (bottom → top, curving left)
        FootprintArc(
            start=outer_bot, mid=outer_mid, end=outer_top,
            layer=LAYER_EDGE_CUTS, width=_lw,
        ),
    )


def _esp32_make_pads(layer: str) -> list[Pad]:
    """Generate all 41 ESP32-S3-WROOM-1 pads (left col, bottom row, right col, GND)."""
    body_w = _ESP32_BODY_W
    body_h = _ESP32_BODY_H
    pad_w = _ESP32_PAD_W
    pad_h = _ESP32_PAD_H
    pitch = _ESP32_PITCH
    n_side = _ESP32_SIDE_PINS
    n_bottom = _ESP32_BOTTOM_PINS

    col_top_y = -(body_h / 2.0) + _ESP32_TOP_MARGIN + pad_h / 2.0
    col_bot_y = col_top_y + (n_side - 1) * pitch

    pad_list: list[Pad] = []

    # Left column: pins 1-14, top to bottom
    # Castellated pads: center at body edge minus half pad height, with
    # 0.35mm additional inset to match KiCad official footprint (pin 1 at -8.75)
    left_x = -(body_w / 2.0 - pad_h / 2.0 + 0.35)
    for i in range(n_side):
        pad_list.append(_smd_pad(str(i + 1), left_x, col_top_y + i * pitch, pad_h, pad_w, layer))

    # Bottom row: pins 15-26, left to right
    bottom_y = body_h / 2.0 - pad_h / 2.0
    start_x = -((n_bottom - 1) * pitch) / 2.0
    for i in range(n_bottom):
        pad_list.append(_smd_pad(str(15 + i), start_x + i * pitch, bottom_y, pad_w, pad_h, layer))

    # Right column: pins 27-40, bottom to top (mirror of left_x)
    right_x = body_w / 2.0 - pad_h / 2.0 + 0.35
    for i in range(n_side):
        pad_list.append(_smd_pad(str(27 + i), right_x, col_bot_y - i * pitch, pad_h, pad_w, layer))

    # Center GND thermal pad — pin 41
    paste = LAYER_F_PASTE if layer == LAYER_F_CU else LAYER_B_PASTE
    mask = LAYER_F_MASK if layer == LAYER_F_CU else LAYER_B_MASK
    pad_list.append(Pad(
        number=_ESP32_THERMAL_PAD_NUMBER, pad_type="smd", shape="rect",
        position=Point(0.0, _ESP32_GND_PAD_Y_OFFSET),
        size_x=_ESP32_GND_PAD_SIZE, size_y=_ESP32_GND_PAD_SIZE,
        layers=(layer, paste, mask),
    ))
    return pad_list


def _esp32_make_pin_labels(
    pad_list: list[Pad],
    n_side: int,
    n_bottom: int,
    fab_layer: str,
) -> list[FootprintText]:
    """Generate fab-layer pin name labels for all signal pads (skip pad 41)."""
    labels: list[FootprintText] = []
    for i, pad in enumerate(pad_list[:-1]):  # skip pad 41
        pin_name = f"{pad.number}:{_ESP32_PIN_NAMES[i]}"
        px, py = pad.position.x, pad.position.y
        if i < n_side:  # left column → shift right
            lx, ly = px + _ESP32_PIN_LABEL_OFFSET, py
        elif i < n_side + n_bottom:  # bottom row → shift up
            lx, ly = px, py - _ESP32_PIN_LABEL_OFFSET
        else:  # right column → shift left
            lx, ly = px - _ESP32_PIN_LABEL_OFFSET, py
        labels.append(FootprintText(
            text_type="user", text=pin_name,
            position=Point(lx, ly), layer=fab_layer,
            effects_size=_ESP32_PIN_LABEL_SIZE,
        ))
    return labels


def _esp32_detect_antenna_side(
    pad_list: list[Pad],
    half_h: float,
) -> bool:
    """Return True if antenna is at positive-Y end, False for negative-Y.

    Detection: the antenna occupies the end of the module body furthest
    from the pad field.  Falls back to pin numbering if pad field is
    symmetric.
    """
    signal_pad_ys = [
        p.position.y for p in pad_list
        if p.number not in (_ESP32_THERMAL_PAD_NUMBER,) and not p.number.startswith("V")
    ]
    if not signal_pad_ys:
        return False

    min_pad_y, max_pad_y = min(signal_pad_ys), max(signal_pad_ys)
    dist_to_neg = abs(min_pad_y - (-half_h))
    dist_to_pos = abs(max_pad_y - half_h)
    if abs(dist_to_pos - dist_to_neg) > 0.5:
        return dist_to_pos > dist_to_neg

    # Symmetric — use pin numbering as tiebreaker (pins 15-26 are antenna side).
    ant_pins = [p for p in pad_list if p.number in {str(i) for i in range(15, 27)}]
    pin1 = [p for p in pad_list if p.number == "1"]
    if ant_pins and pin1:
        return ant_pins[0].position.y > pin1[0].position.y
    return False


def _esp32_make_antenna_keepout(
    pad_list: list[Pad],
    body_w: float,
    body_h: float,
) -> FootprintKeepout:
    """Simple rectangular keepout covering ONLY the antenna area of the ESP32 module.

    The keepout starts just below the bottom pad row (+ 0.5mm clearance) and
    extends to 3.5mm past the module body edge.  Full module width.  No notches,
    no vias -- just a clean rectangle with full copper/track/via restriction.
    """
    antenna_ext = _ESP32_ANTENNA_KEEPOUT_EXTENSION_MM
    half_w = body_w / 2.0
    half_h = body_h / 2.0

    antenna_at_positive_y = _esp32_detect_antenna_side(pad_list, half_h)

    # Find the edge of the bottom pad row on the antenna side.
    # "Bottom" here means the pad row closest to the antenna end.
    signal_pads = [
        p for p in pad_list
        if p.number != _ESP32_THERMAL_PAD_NUMBER and not p.number.startswith("V")
    ]
    if not signal_pads:
        # Fallback: use body edge minus a small margin
        if antenna_at_positive_y:
            keepout_top_y = half_h - 5.0
            keepout_bot_y = half_h + antenna_ext
        else:
            keepout_top_y = -half_h - antenna_ext
            keepout_bot_y = -half_h + 5.0
    elif antenna_at_positive_y:
        # Antenna at +Y: find max Y of signal pads (bottom of pad row on that side)
        max_pad_y = max(p.position.y + p.size_y / 2.0 for p in signal_pads)
        keepout_top_y = max_pad_y + 0.5  # 0.5mm clearance from pad field
        keepout_bot_y = half_h + antenna_ext
    else:
        # Antenna at -Y: find min Y of signal pads (top of pad row on that side)
        min_pad_y = min(p.position.y - p.size_y / 2.0 for p in signal_pads)
        keepout_top_y = -half_h - antenna_ext
        keepout_bot_y = min_pad_y - 0.5  # 0.5mm clearance from pad field

    # Simple 4-point rectangle, full module width
    keepout_poly = (
        Point(-half_w, keepout_top_y),
        Point(half_w, keepout_top_y),
        Point(half_w, keepout_bot_y),
        Point(-half_w, keepout_bot_y),
    )

    return FootprintKeepout(
        polygon=keepout_poly, layers=(LAYER_F_CU, LAYER_B_CU),
        no_copper=True, no_vias=True, no_tracks=True, tag="antenna",
    )



# Via fence / perimeter via / thermal via functions removed (simplified keepout approach).
# The user only needs a simple rectangular keepout under the antenna -- no vias.


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

    The footprint includes a simple rectangular antenna keepout zone covering
    only the antenna area (below the pad field to 3.5mm past the module body
    edge) with full copper/track/via restriction, and a 3D model reference.

    Args:
        ref: Reference designator (e.g. "U3").
        value: Component value string.
        layer: Primary copper layer.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    body_w = _ESP32_BODY_W
    body_h = _ESP32_BODY_H
    lib_id = "RF_Module:ESP32-S3-WROOM-1"
    fab_layer = LAYER_F_FAB if layer == LAYER_F_CU else LAYER_B_FAB

    pad_list = _esp32_make_pads(layer)
    pin_labels = _esp32_make_pin_labels(pad_list, _ESP32_SIDE_PINS, _ESP32_BOTTOM_PINS, fab_layer)
    antenna_keepout = _esp32_make_antenna_keepout(pad_list, body_w, body_h)

    graphics = _courtyard_rect(body_w, body_h)
    texts: tuple[FootprintText, ...] = (
        _ref_text(ref, -(body_h / 2.0 + _TEXT_OFFSET_LARGE), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + _TEXT_OFFSET_LARGE, LAYER_F_FAB),
        *pin_labels,
    )

    # Build footprint without 3D model, then enrich to compute the correct
    # model offset from actual pad positions (single source of truth).
    fp = Footprint(
        lib_id=lib_id, ref=ref, value=value, position=Point(0.0, 0.0),
        layer=layer, pads=tuple(pad_list), graphics=graphics, texts=texts,
        attr="smd", models=(),
        fp_zones=(antenna_keepout,),
    )
    return _enrich_esp32_footprint(fp)


def _esp32_enrich_pin_labels(fp: Footprint, fab_layer: str) -> list[FootprintText]:
    """Return fab-layer pin labels for *fp* if not already present (idempotent)."""
    if any(
        t.text_type == "user"
        and (t.text in _ESP32_PIN_NAMES or any(t.text.endswith(f":{n}") for n in _ESP32_PIN_NAMES))
        for t in fp.texts
    ):
        return []
    if len(fp.pads) < 40:
        return []
    all_x = [p.position.x for p in fp.pads[:40]]
    min_x, max_x = min(all_x), max(all_x)
    labels: list[FootprintText] = []
    for idx, pad in enumerate(fp.pads[:40]):
        if idx >= len(_ESP32_PIN_NAMES):
            break
        pin_name = f"{pad.number}:{_ESP32_PIN_NAMES[idx]}"
        px, py = pad.position.x, pad.position.y
        if abs(px - min_x) < 1.0:  # left column
            lx, ly = px + _ESP32_PIN_LABEL_OFFSET, py
        elif abs(px - max_x) < 1.0:  # right column
            lx, ly = px - _ESP32_PIN_LABEL_OFFSET, py
        else:  # bottom row
            lx, ly = px, py - _ESP32_PIN_LABEL_OFFSET
        labels.append(FootprintText(
            text_type="user", text=pin_name,
            position=Point(lx, ly), layer=fab_layer,
            effects_size=_ESP32_PIN_LABEL_SIZE,
        ))
    return labels


def _esp32_enrich_antenna_keepout(fp: Footprint) -> tuple[list[FootprintKeepout], list[Pad]]:
    """Return a simple rectangular antenna keepout for *fp*.  No vias.

    Idempotent: returns empty lists if the footprint already has an antenna
    keepout zone.
    """
    if any(fz.tag == "antenna" for fz in fp.fp_zones):
        return [], []

    body_w = _ESP32_BODY_W
    body_h = _ESP32_BODY_H
    pad_list = list(fp.pads)
    zone = _esp32_make_antenna_keepout(pad_list, body_w, body_h)

    return [zone], []


def _esp32_enrich_3d_model(fp: Footprint) -> tuple[Footprint3DModel, ...]:
    """Return the KiCad standard ESP32-S3-WROOM-1 3D model.

    JLCPCB footprints embed ``WIRELM-SMD_ESP32-S3-WROOM-1.step`` which
    does not exist in KiCad's library.  Override unconditionally with the
    known-correct path.

    The model offset is NOT computed here — it is handled generically by
    :func:`_apply_jlcpcb_model_offset` using the registry's
    ``kicad_ref_pad1_x/y`` fields.
    """
    return (Footprint3DModel(
        path=f"{KICAD_3DMODEL_VAR}/RF_Module.3dshapes/ESP32-S3-WROOM-1.step",
        offset=(0.0, 0.0, 0.0),
    ),)


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
    fab_layer = LAYER_F_FAB if fp.layer == LAYER_F_CU else LAYER_B_FAB

    extra_texts = _esp32_enrich_pin_labels(fp, fab_layer)
    models = _esp32_enrich_3d_model(fp)

    # Antenna keepout and via fence are handled at BOARD level by
    # _refresh_antenna_keepout in ee_phases_refinement.py. This correctly
    # accounts for the final placement rotation. Footprint-level keepout/vias
    # would rotate to the wrong position (e.g., rot=180 flips +Y to -Y).
    extra_pads: list[Pad] = []

    if not extra_texts and models is fp.models:
        return fp

    return Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
        position=fp.position, rotation=fp.rotation, layer=fp.layer,
        pads=(*fp.pads, *extra_pads) if extra_pads else fp.pads,
        graphics=fp.graphics,
        texts=(*fp.texts, *extra_texts),
        lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
        models=models,
        datasheet=fp.datasheet, description=fp.description,
        footprint_source=fp.footprint_source,
        mpn=fp.mpn, manufacturer=fp.manufacturer,
        fp_zones=fp.fp_zones,  # no footprint-level keepout
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
    pitch = size_w - pad_w + _CRYSTAL_PITCH_ADJUST_MM
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
    body_w = (max(all_x) - min(all_x)) + pad_w + _SOT23_BODY_PADDING_MM
    body_h = (max(all_y) - min(all_y)) + pad_h + _SOT23_BODY_PADDING_MM

    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    ref_y = -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)
    val_y = body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM
    texts = (
        _ref_text(ref, ref_y, LAYER_F_SILKSCREEN),
        _val_text(value, val_y, LAYER_F_FAB),
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


# ---------------------------------------------------------------------------
# SOT-223 dimensions (mm) — from IPC-7351B / KiCad official footprint
# ---------------------------------------------------------------------------
_SOT223_SMALL_PAD_W: float = 1.5
_SOT223_SMALL_PAD_H: float = 0.7
_SOT223_TAB_PAD_W: float = 1.5
_SOT223_TAB_PAD_H: float = 3.15
_SOT223_PITCH: float = 2.3   # Small pad centre-to-centre pitch
_SOT223_ROW_SPAN: float = 6.2  # Distance between small-pad row and tab-pad row centres


def make_sot223(
    ref: str,
    value: str,
    layer: str = LAYER_F_CU,
) -> Footprint:
    """SOT-223 footprint with 3 small pads + 1 large tab pad.

    Standard SOT-223 pinout:
      - Pins 1-3: small gull-wing pads on the bottom row
      - Pin 4 (tab): large pad on the top row

    Dimensions from KiCad official ``SOT-223-3_TabPin2`` footprint.

    Args:
        ref: Reference designator (e.g. "U1").
        value: Component value string.
        layer: Primary copper layer, default F.Cu.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    _log.debug("make_sot223 ref=%s", ref)

    # 3 small pads on the bottom row (y = +row_span/2)
    bottom_y = _SOT223_ROW_SPAN / 2.0
    pads_list: list[Pad] = []
    for i in range(3):
        x = (i - 1) * _SOT223_PITCH  # -2.3, 0.0, +2.3
        pads_list.append(
            _smd_pad(str(i + 1), x, bottom_y, _SOT223_SMALL_PAD_W, _SOT223_SMALL_PAD_H, layer)
        )
    # Tab pad on the top row (y = -row_span/2)
    top_y = -_SOT223_ROW_SPAN / 2.0
    pads_list.append(
        _smd_pad("4", 0.0, top_y, _SOT223_TAB_PAD_W, _SOT223_TAB_PAD_H, layer)
    )

    body_w = 2 * _SOT223_PITCH + _SOT223_SMALL_PAD_W + _SOT23_BODY_PADDING_MM
    body_h = _SOT223_ROW_SPAN + max(_SOT223_SMALL_PAD_H, _SOT223_TAB_PAD_H) + _SOT23_BODY_PADDING_MM

    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    ref_y = -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)
    val_y = body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM
    texts = (
        _ref_text(ref, ref_y, LAYER_F_SILKSCREEN),
        _val_text(value, val_y, LAYER_F_FAB),
    )
    sot223_lib_id = "Package_TO_SOT_SMD:SOT-223"
    model = _model_for_package(sot223_lib_id)
    models = (model,) if model is not None else ()
    return Footprint(
        lib_id=sot223_lib_id,
        ref=ref,
        value=value,
        position=Point(0.0, 0.0),
        layer=layer,
        pads=tuple(pads_list),
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
    # Match _NP pattern (terminal blocks: _3P = 3 positions)
    m = re.search(r"[-_](\d{1,3})P(?:[-_]|$)", footprint_id)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 20:
            return n
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
    exposed_pad_size: tuple[float, float] | None = None,
    exposed_pad_number: str | None = None,
) -> Footprint:
    """Generate a generic SMD IC footprint (MSOP, TSSOP, SOIC, QFP, QFN, etc.).

    Pins are arranged in two rows: odd pins on the left, even on the right.
    QFN/DFN packages automatically get a center thermal/exposed pad (pad N+1).
    Dual-row packages (SOIC/MSOP/TSSOP) also get an exposed pad when
    *exposed_pad_size* is provided or when component pins indicate one.

    Args:
        ref: Reference designator.
        value: Component value string.
        pin_count: Total number of pins (signal pins only, thermal pad auto-added).
        pitch_mm: Pin pitch in mm.
        lib_id: KiCad library ID string (auto-generated if empty).
        thermal_pad: Whether to add a center thermal pad.  ``None`` (default)
            auto-detects from the lib_id (QFN/DFN packages get one).
        exposed_pad_size: Explicit ``(width_mm, height_mm)`` for the center
            exposed/thermal pad.  Overrides auto-sizing when provided.
        exposed_pad_number: Pad number string for the exposed pad (e.g.
            ``"9"``).  Defaults to ``str(pin_count + 1)``.

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

    # --- Exposed / thermal pad logic ---
    # Priority: explicit exposed_pad_size > QFN/DFN auto-detect > thermal_pad flag
    # When the thermal pad number matches an existing signal pad (e.g. pin 8 on
    # SOIC-8 is "PAD"), REPLACE the gull-wing pad instead of appending a duplicate.
    _upper = lib_id.upper()
    if exposed_pad_size is not None:
        # Caller explicitly requested an exposed pad with known dimensions
        ep_w, ep_h = exposed_pad_size
        ep_num = exposed_pad_number or str(pin_count + 1)
        ep = _smd_pad(ep_num, 0.0, 0.0, ep_w, ep_h, LAYER_F_CU)
        existing_nums = {p.number for p in pads}
        if ep_num in existing_nums:
            pads = [ep if p.number == ep_num else p for p in pads]
        else:
            pads.append(ep)
    else:
        # Auto-detect thermal pad for QFN/DFN packages
        if thermal_pad is None:
            thermal_pad = any(kw in _upper for kw in ("QFN", "DFN"))
        if thermal_pad:
            # Center exposed pad, size ~60% of body
            ep_size = max(row_span * 0.5, _QFN_THERMAL_PAD_MIN_MM)
            ep_num = exposed_pad_number or str(pin_count + 1)
            ep = _smd_pad(ep_num, 0.0, 0.0, ep_size, ep_size, LAYER_F_CU)
            existing_nums = {p.number for p in pads}
            if ep_num in existing_nums:
                pads = [ep if p.number == ep_num else p for p in pads]
            else:
                pads.append(ep)

    body_w = col_pitch * 2.0 + pad_h
    body_h = row_span + pad_w + _BODY_MARGIN_MM
    ic_pad_edge_x = col_pitch + pad_h / 2.0
    graphics: tuple[FootprintLine, ...] = (
        *_courtyard_rect(body_w, body_h),
        *_silk_side_marks(col_pitch * _IC_SILK_COL_PITCH_FACTOR, body_h, pad_edge_x=ic_pad_edge_x),
    )
    _ic_ref_y = -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)
    _ic_val_y = body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM
    texts = (
        _ref_text(ref, _ic_ref_y, LAYER_F_SILKSCREEN),
        _val_text(value, _ic_val_y, LAYER_F_FAB),
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
    graphics: tuple[FootprintLine, ...] = (
        *_courtyard_rect(body_w, body_h, layer=crtyd_layer, cx=cx, cy=cy),
    )
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

    Pin and wire-entry orientation convention:

    - **rotation=0** (default): wire entry faces north (top edge of the
      board), pin 1 is on the left when viewed from above.  Use this
      orientation when the terminal block sits on the **top** board edge.
    - **rotation=180**: wire entry faces south (board interior) and the
      pin order is physically mirrored left-to-right.  Use this
      orientation when the terminal block sits on the **bottom** board
      edge so that wires enter from below.

    The placement optimizer (``ee_phases_refinement.py``) is responsible
    for setting the correct rotation based on which board edge the
    terminal block is assigned to.  If the rotation is wrong, the wire
    entry will face inward (toward board components) instead of outward
    toward the enclosure opening.

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
    _dip_ref_y = cy - (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)
    _dip_val_y = cy + body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM
    texts = (
        FootprintText(text_type="reference", text=ref,
                      position=Point(cx, _dip_ref_y),
                      layer=LAYER_F_SILKSCREEN, effects_size=1.0),
        FootprintText(text_type="value", text=value,
                      position=Point(cx, _dip_val_y),
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
    _usbc_ref_y = -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)
    _usbc_val_y = body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM
    texts = (
        _ref_text(ref, _usbc_ref_y, LAYER_F_SILKSCREEN),
        _val_text(value, _usbc_val_y, LAYER_F_FAB),
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
    _rj45_ref_y = cy - (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)
    _rj45_val_y = cy + (body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)
    texts = (
        _ref_text(ref, _rj45_ref_y, LAYER_F_SILKSCREEN),
        _val_text(value, _rj45_val_y, LAYER_F_FAB),
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
        _val_text(value, body_h / 2.0 + _VAL_TEXT_OFFSET_MM, LAYER_F_FAB),
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
    body_h = max((half - 1) * pitch_mm + pad_diam + _BODY_MARGIN_MM, _DIP_MIN_BODY_HEIGHT_MM)
    graphics: tuple[FootprintLine, ...] = (*_courtyard_rect(body_w, body_h),)
    _dip_tht_ref_y = -(body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM)
    _dip_tht_val_y = body_h / 2.0 + PCB_COURTYARD_CLEARANCE_MM + _TEXT_MARGIN_MM
    texts = (
        _ref_text(ref, _dip_tht_ref_y, LAYER_F_SILKSCREEN),
        _val_text(value, _dip_tht_val_y, LAYER_F_FAB),
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
    size: str = _WS2812_SIZE_5050,
) -> Footprint:
    """WS2812B addressable RGB LED footprint (PLCC-4, 4 pads).

    Pin 1 = VDD, Pin 2 = DOUT, Pin 3 = GND, Pin 4 = DIN.

    Supported sizes:
        - ``"5050"``: 5.0x5.0 mm body (WS2812B standard)
        - ``"3535"``: 3.5x3.5 mm body (WS2812B-Mini)
        - ``"2020"``: 2.0x2.0 mm body (WS2812C-2020)

    Args:
        ref: Reference designator (e.g. "LED1").
        value: Component value string.
        layer: Primary copper layer.
        size: Package size code.

    Returns:
        Fully constructed :class:`Footprint`.
    """
    dims = _WS2812B_DIMS.get(size, _WS2812B_DIMS[_WS2812_SIZE_5050])
    pad_w, pad_h, x_pitch, y_pitch, body_w, body_h = dims
    ws2812b_lib_ids = {
        _WS2812_SIZE_2020: "LED_SMD:LED_WS2812B_PLCC4_2.0x2.0mm",
        _WS2812_SIZE_3535: "LED_SMD:LED_WS2812B_PLCC4_3.5x3.5mm_P2.45mm",
        _WS2812_SIZE_5050: "LED_SMD:LED_WS2812B_PLCC4_5.0x5.0mm_P3.2mm",
    }
    lib_id = ws2812b_lib_ids.get(size, ws2812b_lib_ids[_WS2812_SIZE_5050])

    pads = (
        _smd_pad("1", -x_pitch, -y_pitch, pad_w, pad_h, layer),  # VDD
        _smd_pad("2", x_pitch, -y_pitch, pad_w, pad_h, layer),   # DOUT
        _smd_pad("3", x_pitch, y_pitch, pad_w, pad_h, layer),    # GND
        _smd_pad("4", -x_pitch, y_pitch, pad_w, pad_h, layer),   # DIN
    )
    graphics = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + _TEXT_OFFSET_SMALL), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + _VAL_TEXT_OFFSET_MM, LAYER_F_FAB),
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
        pads.append(_smd_pad(
            str(i + 1), x, _MICROSD_SIGNAL_Y, _MICROSD_PAD_W, _MICROSD_PAD_H, LAYER_F_CU,
        ))
    # Shield / card detect pads (larger, on sides)
    pads.append(_smd_pad(
        "9", _MICROSD_SHIELD_PAD_LEFT_X, _MICROSD_SHIELD_PAD_Y,
        _MICROSD_SHIELD_PAD_W, _MICROSD_SHIELD_PAD_H, LAYER_F_CU,
    ))
    pads.append(_smd_pad(
        "10", _MICROSD_SHIELD_PAD_RIGHT_X, _MICROSD_SHIELD_PAD_Y,
        _MICROSD_SHIELD_PAD_W, _MICROSD_SHIELD_PAD_H, LAYER_F_CU,
    ))

    body_w = _MICROSD_BODY_W
    body_h = _MICROSD_BODY_H
    graphics = (*_courtyard_rect(body_w, body_h),)
    texts = (
        _ref_text(ref, -(body_h / 2.0 + _TEXT_OFFSET_SMALL), LAYER_F_SILKSCREEN),
        _val_text(value, body_h / 2.0 + _VAL_TEXT_OFFSET_MM, LAYER_F_FAB),
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

    # --- Step 2: compute U-shaped isolation slot around coil pin ---
    # The coil pins are low-voltage logic but sit physically adjacent to
    # high-voltage contact pads (COM, NO, NC).  A small U-shaped
    # Edge.Cuts routed slot around the coil pin closest to the contacts
    # provides creepage isolation between mains and logic domains.
    # The U opens toward the OTHER coil pin (same LV domain) and the
    # closed wall faces the nearest contact pad.
    #
    # KiCad footprint pad semantics (Relay_SPDT_SANYOU_SRD):
    #   Pad 1=COM, 3=NO, 4=NC  → contact/mains side (high voltage)
    #   Pad 2=Coil-, 5=Coil+   → coil/logic side (low voltage)
    contact_pads_xy: list[tuple[float, float]] = []
    coil_pads_list: list[tuple[str, float, float, float]] = []  # (num, x, y, size)
    for pad in fp.pads:
        if pad.number in ("1", "3", "4"):
            contact_pads_xy.append((pad.position.x, pad.position.y))
        elif pad.number in ("2", "5"):
            coil_pads_list.append((
                pad.number, pad.position.x, pad.position.y, pad.size_x,
            ))

    if contact_pads_xy and coil_pads_list:
        # Find the coil pin closest to any contact pad — that's the one
        # that needs isolation.
        contact_cx = sum(x for x, _ in contact_pads_xy) / len(contact_pads_xy)
        contact_cy = sum(y for _, y in contact_pads_xy) / len(contact_pads_xy)
        best_coil = min(
            coil_pads_list,
            key=lambda c: (c[1] - contact_cx) ** 2 + (c[2] - contact_cy) ** 2,
        )
        coil_num, coil_x, coil_y, coil_size = best_coil
        coil_r = coil_size / 2.0
        clearance = 1.0  # mm from pad edge to slot center
        u_half = coil_r + clearance

        # U-shaped isolation cutout — arms are straight parallel lines,
        # closed end is an arc curving around the coil pin.
        if contact_cx > coil_x:
            # Contacts to the RIGHT → closed arc on left, open on right
            closed_x = coil_x - u_half
            open_x = coil_x + u_half
        else:
            # Contacts to the LEFT → closed arc on right, open on left
            closed_x = coil_x + u_half
            open_x = coil_x - u_half

        sw = _RELAY_SLOT_WIDTH / 2.0  # half channel width
        _lw = 0.05  # thin line width

        outer_r = u_half + sw
        inner_r = u_half - sw

        # Arm endpoints (straight segments)
        b = Point(open_x, coil_y - u_half - sw)   # top arm outer end
        c = Point(open_x, coil_y - u_half + sw)   # top arm inner end
        f = Point(open_x, coil_y + u_half - sw)   # bottom arm inner end
        g = Point(open_x, coil_y + u_half + sw)   # bottom arm outer end

        # Arc endpoints centered on coil pin
        outer_top = Point(coil_x, coil_y - outer_r)
        outer_bot = Point(coil_x, coil_y + outer_r)
        outer_mid = Point(closed_x - sw if closed_x < coil_x else closed_x + sw, coil_y)
        inner_top = Point(coil_x, coil_y - inner_r)
        inner_bot = Point(coil_x, coil_y + inner_r)
        inner_mid = Point(closed_x + sw if closed_x < coil_x else closed_x - sw, coil_y)

        slot_lines: list[FootprintLine | FootprintArc] = [
            # Top arm outer
            FootprintLine(start=outer_top, end=b, layer=LAYER_EDGE_CUTS, width=_lw),
            # Open end top
            FootprintLine(start=b, end=c, layer=LAYER_EDGE_CUTS, width=_lw),
            # Top arm inner
            FootprintLine(start=c, end=inner_top, layer=LAYER_EDGE_CUTS, width=_lw),
            # Inner arc (closed end, curves around coil pin)
            FootprintArc(
                start=inner_top, mid=inner_mid, end=inner_bot,
                layer=LAYER_EDGE_CUTS, width=_lw,
            ),
            # Bottom arm inner
            FootprintLine(start=inner_bot, end=f, layer=LAYER_EDGE_CUTS, width=_lw),
            # Open end bottom
            FootprintLine(start=f, end=g, layer=LAYER_EDGE_CUTS, width=_lw),
            # Bottom arm outer
            FootprintLine(start=g, end=outer_bot, layer=LAYER_EDGE_CUTS, width=_lw),
            # Outer arc (closed end, curves around coil pin)
            FootprintArc(
                start=outer_bot, mid=outer_mid, end=outer_top,
                layer=LAYER_EDGE_CUTS, width=_lw,
            ),
        ]
        cleaned.extend(slot_lines)
        _log.info(
            "Relay %s: U-shaped isolation slot around coil pad %s at (%.1f, %.1f), "
            "closed wall toward contacts, opens %s",
            fp.ref, coil_num, coil_x, coil_y,
            "left" if contact_cx > coil_x else "right",
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
    p41_pads = [p for p in fp.pads if p.number == _ESP32_THERMAL_PAD_NUMBER]
    if len(p41_pads) <= 1:
        return fp  # Already correct — nothing to do.

    # Compute bounding box of all pad-41 entries (pad extent, not centres).
    min_x = min(p.position.x - p.size_x / 2 for p in p41_pads)
    max_x = max(p.position.x + p.size_x / 2 for p in p41_pads)
    min_y = min(p.position.y - p.size_y / 2 for p in p41_pads)
    max_y = max(p.position.y + p.size_y / 2 for p in p41_pads)

    # True centroid: average of all original pad-41 centre positions.
    # For a symmetric 3x3 grid this equals the BB centre, but using the
    # average is correct for any grid arrangement.
    cx = sum(p.position.x for p in p41_pads) / len(p41_pads)
    cy = sum(p.position.y for p in p41_pads) / len(p41_pads)
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
    new_pads = (*[p for p in fp.pads if p.number != _ESP32_THERMAL_PAD_NUMBER], merged)
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

# Expected minimum pad counts for common package types.
# Used to reject JLCPCB cached footprints that have wrong pad counts
# (e.g. easyeda2kicad returning SOD-323 for an SOT-23 transistor).
_EXPECTED_PAD_COUNTS: dict[str, int] = {
    "SOT-23": 3, "SOT-23-3": 3, "SOT-23-5": 5, "SOT-23-6": 6,
    "SOT-223": 4, "SOT-89": 3,
    "SOIC-8": 8, "SOIC-14": 14, "SOIC-16": 16,
    "MSOP-8": 8, "MSOP-10": 10, "MSOP-16": 16,
    "TSSOP-8": 8, "TSSOP-14": 14, "TSSOP-16": 16, "TSSOP-20": 20,
    "LQFP-32": 32, "LQFP-48": 48, "LQFP-64": 64, "LQFP-100": 100,
    "QFN-16": 16, "QFN-20": 20, "QFN-24": 24, "QFN-32": 32, "QFN-48": 48,
    "USB-C": 4,  # minimum: 4 power+CC pins; full USB-C has 9+
    "ESP32-WROOM": 18,  # minimum 18 castellated pads
    "ESP32-S3-WROOM": 18,
}


def _expected_pad_count(footprint_id: str) -> int:
    """Return expected minimum pad count for a footprint_id, or 0 if unknown."""
    upper = footprint_id.strip().upper()
    for pattern, count in _EXPECTED_PAD_COUNTS.items():
        if pattern.upper() in upper:
            return count
    # Fallback: try parsing pin count from footprint_id (e.g. _3P, _1x06)
    parsed = _parse_pin_count(footprint_id)
    if parsed > 2:
        return parsed
    return 0


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
        # Validate pad count: if footprint_id implies a specific pin count
        # (e.g. SOT-23 → 3 pads) but JLCPCB gives fewer, reject the bad footprint.
        _expected_pads = _expected_pad_count(footprint_id)
        if _expected_pads > 0 and len(fp.pads) < _expected_pads:
            _log.warning(
                "JLCPCB footprint for %s (%s) has %d pads but %s expects %d; "
                "rejecting JLCPCB footprint",
                ref, lcsc, len(fp.pads), footprint_id, _expected_pads,
            )
            return None
        # Validate pad TYPE: if footprint_id implies SMD but JLCPCB returns
        # THT pads, reject the bad footprint.  Covers standard passives,
        # ICs, and modules (ESP32, etc.) that should all be SMD.
        _fid_upper = footprint_id.strip().upper()
        _SMD_KEYWORDS = ("R_0", "C_0", "C_1", "L_0", "L_1", "LED_0", "SOT-", "SOD-",
                         "SOIC", "MSOP", "TSSOP", "QFN", "QFP", "LQFP",
                         "ESP32", "WROOM", "WROVER", "USB-C", "USB_C")
        if any(kw in _fid_upper for kw in _SMD_KEYWORDS):
            tht_pads = [p for p in fp.pads if p.pad_type == "thru_hole"]
            if len(tht_pads) > 0:
                _log.warning(
                    "JLCPCB footprint for %s (%s) has %d THT pads but %s "
                    "expects SMD; rejecting",
                    ref, lcsc, len(tht_pads), footprint_id,
                )
                return None
        # Validate pad SIZE: if footprint_id specifies a package size (0402/0603/0805)
        # but JLCPCB pads are a different size, reject the bad footprint.
        # This catches LCSC parts that return wrong-sized cached footprints.
        if fp.pads:
            max_pad = max(max(p.size_x, p.size_y) for p in fp.pads)
            if ("0402" in _fid_upper or "_0402" in _fid_upper) and max_pad > 0.8:
                _log.warning(
                    "JLCPCB footprint for %s (%s) has %.2fmm pads but %s "
                    "expects 0402 (<0.8mm); rejecting",
                    ref, lcsc, max_pad, footprint_id,
                )
                return None
            if ("0603" in _fid_upper or "_0603" in _fid_upper) and max_pad > 1.2:
                _log.warning(
                    "JLCPCB footprint for %s (%s) has %.2fmm pads but %s "
                    "expects 0603 (<1.2mm); rejecting",
                    ref, lcsc, max_pad, footprint_id,
                )
                return None
        # Reject JLCPCB SOIC/MSOP/TSSOP/SOP with wrong pad orientation.
        # EasyEDA exports sometimes rotate the entire footprint 90 degrees
        # from standard KiCad convention: pads form horizontal rows (top/bottom)
        # instead of vertical columns (left/right).
        #
        # Detection: for a dual-row IC the first half of numbered pads should
        # share a similar X (left column) and vary in Y.  If they instead share
        # a similar Y (top row) and vary in X, the footprint is rotated.
        # Exclude any center/thermal pads (number > pin_count or at origin).
        if _fid_upper.startswith(("SOIC", "MSOP", "TSSOP", "SOP")) and len(fp.pads) >= 6:
            # Collect only perimeter signal pads (exclude thermal/exposed pads)
            sig_pads = sorted(
                [p for p in fp.pads if p.number.isdigit()],
                key=lambda p: int(p.number),
            )
            n_sig = len(sig_pads)
            if n_sig >= 6:
                # First half = one side, second half = other side
                first_half = sig_pads[: n_sig // 2]
                fh_x_spread = max(p.position.x for p in first_half) - min(
                    p.position.x for p in first_half
                )
                fh_y_spread = max(p.position.y for p in first_half) - min(
                    p.position.y for p in first_half
                )
                # Correct orientation: first-half X-spread ~ 0 (column),
                # Y-spread ~ (n/2-1)*pitch.
                # Rotated: first-half X-spread ~ (n/2-1)*pitch (row),
                # Y-spread ~ 0.
                if fh_x_spread > fh_y_spread:
                    _log.warning(
                        "JLCPCB footprint for %s (%s) has pads in horizontal "
                        "rows (EasyEDA 90deg rotation, first-half X-spread="
                        "%.2f > Y-spread=%.2f); rejecting",
                        ref, lcsc, fh_x_spread, fh_y_spread,
                    )
                    return None
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
                ep_size = max(_QFN_THERMAL_PAD_MIN_MM, pin_count * _QFN_THERMAL_PAD_SCALE)
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

        # JLCPCB .kicad_mod files may include 3D model references with
        # pre-computed offsets — these are preserved above (fp.models non-empty).
        # However, easyeda2kicad often generates model paths that don't match
        # KiCad's actual filenames.  Validate and discard if missing.
        if fp.models and not _step_file_exists(fp.models[0].path):
            _log.warning(
                "JLCPCB 3D model does not exist: %s for %s; falling back",
                fp.models[0].path.split("/")[-1], ref,
            )
            fp = Footprint(
                lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                position=fp.position, rotation=fp.rotation, layer=fp.layer,
                pads=fp.pads, graphics=fp.graphics, texts=fp.texts,
                lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
                models=(),  # discard invalid model
                datasheet=fp.datasheet, description=fp.description,
                footprint_source=fp.footprint_source, fp_zones=fp.fp_zones,
            )

        # If no model is present, look up by package type.  JLCPCB cached
        # footprints typically have body-centered pads, so strip the
        # parametric offset that _model_terminal_block/etc. add (they assume
        # pin-1-at-origin pad layout).
        if not fp.models:
            # Try lib_id first (actual package from JLCPCB), then footprint_id
            # (requirements).  Validate each candidate exists on disk before
            # accepting — JLCPCB lib_ids often produce non-existent model names.
            model = _model_for_package(fp.lib_id, layer)
            if model is not None and not _step_file_exists(model.path):
                _log.debug("Model from lib_id does not exist: %s", model.path)
                model = None
            if model is None:
                model = _model_for_package(footprint_id, layer)
            if model is not None:
                # JLCPCB footprints have different pad layouts than the
                # parametric footprints the model dispatch was designed for.
                # Strip hardcoded offset AND rotation — the generic
                # _apply_jlcpcb_model_offset() handles alignment using
                # KiCad library pad-1 reference positions from the registry.
                if model.offset != (0.0, 0.0, 0.0) or model.rotate != (0.0, 0.0, 0.0):
                    model = Footprint3DModel(
                        path=model.path,
                        offset=(0.0, 0.0, 0.0),
                        scale=model.scale,
                        rotate=(0.0, 0.0, 0.0),
                    )
                fp = Footprint(
                    lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                    position=fp.position, rotation=fp.rotation, layer=fp.layer,
                    pads=fp.pads, graphics=fp.graphics, texts=fp.texts,
                    lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
                    models=(model,),
                    datasheet=fp.datasheet, description=fp.description,
                    footprint_source=fp.footprint_source, fp_zones=fp.fp_zones,
                )

        # Generic 3D model offset correction from component registry.
        # Looks up the KiCad library pad 1 position and shifts the model
        # so it aligns with the actual (JLCPCB) pad positions.
        fp = _apply_jlcpcb_model_offset(fp, footprint_id)

        # Ensure JLCPCB footprints have courtyard graphics.
        # easyeda2kicad exports often omit F.CrtYd — generate from pad bbox.
        fp = _ensure_courtyard(fp)

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
    ws_size = _WS2812_SIZE_5050
    if _WS2812_SIZE_2020 in fid:
        ws_size = _WS2812_SIZE_2020
    elif _WS2812_SIZE_3535 in fid:
        ws_size = _WS2812_SIZE_3535
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
    return make_sot223(ref, value)


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
    # Terminal blocks default to 5.08mm pitch (Phoenix MKDS standard),
    # not 2.54mm like pin headers.  Only override if explicitly in the fid.
    import re as _re
    pitch_m = _re.search(r"P(\d+\.?\d*)mm", fid)
    pitch = float(pitch_m.group(1)) if pitch_m else 5.08
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
    w, h = dims if dims else (_CRYSTAL_SMD3215_DEFAULT_W_MM, _CRYSTAL_SMD3215_DEFAULT_H_MM)
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


def _fp_mounting_hole(
    ref: str, _value: str, fid: str, _upper: str, _layer: str,
) -> Footprint:
    """Build mounting hole footprint with drill parsed from footprint ID."""
    dims = _parse_dimensions(fid)
    drill = dims[0] if dims else 3.2
    return make_mounting_hole(ref, drill_diameter=drill)


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
    (lambda _u, _f: _u.startswith("MOUNTINGHOLE"), _fp_mounting_hole),
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
    ref: str,
    value: str,
    fid: str,
    upper: str,
    pins: tuple[Pin, ...] = (),
) -> Footprint | None:
    """Match generic SMD IC packages (MSOP, TSSOP, SOIC, QFP, QFN, etc.).

    When *pins* contains an exposed thermal pad pin (name ``PAD``/``EP``/
    ``EPAD``/``GND_PAD``/``THERMAL``/``POWERPAD`` with type ``POWER_IN``),
    the footprint is generated with a center exposed pad sized to ~60% of the
    IC body.  This covers PowerPAD variants such as TPS54331 DGQ (SOIC-8
    with exposed GND pad).
    """
    ic_prefixes = ("MSOP", "TSSOP", "SOIC", "QFP", "QFN", "SOP", "DFN", "SSOP", "LQFP")
    if not any(upper.startswith(p) for p in ic_prefixes):
        return None
    pin_count = _parse_pin_count(fid)
    if pin_count < 2:
        pin_count = 8
    pitch = _parse_pitch(fid)
    if pitch > 2.0:
        pitch = 1.27 if upper.startswith(("SOP", "SOIC")) else 0.5

    # Detect exposed thermal pad from component pin definitions
    ep_size: tuple[float, float] | None = None
    ep_number: str | None = None
    if pins:
        thermal_pin = _find_thermal_pad_pin(pins)
        if thermal_pin is not None:
            # Compute exposed pad size: ~60% of IC body dimensions
            half = pin_count // 2
            row_span = (half - 1) * pitch
            body_w = (row_span / 2.0 + _IC_COL_OFFSET) * 2.0
            body_h = row_span + _BODY_MARGIN_MM
            ep_w = max(body_w * _DUAL_ROW_THERMAL_PAD_BODY_FRACTION,
                       _QFN_THERMAL_PAD_MIN_MM)
            ep_h = max(body_h * _DUAL_ROW_THERMAL_PAD_BODY_FRACTION,
                       _QFN_THERMAL_PAD_MIN_MM)
            ep_size = (round(ep_w, 2), round(ep_h, 2))
            ep_number = thermal_pin.number
            _log.info(
                "Detected exposed thermal pad pin '%s' (#%s) for %s; "
                "adding %.1fx%.1fmm center pad",
                thermal_pin.name, thermal_pin.number, ref, ep_size[0], ep_size[1],
            )

    fp = make_generic_smd_ic(
        ref, value, pin_count, pitch, lib_id=fid,
        exposed_pad_size=ep_size, exposed_pad_number=ep_number,
    )

    # Update 3D model to thermal-pad variant if an exposed pad was added
    if ep_size is not None:
        for pkg_key, model_path in _THERMAL_3D_MODEL_VARIANTS.items():
            if pkg_key in upper:
                thermal_model = Footprint3DModel(
                    path=f"{KICAD_3DMODEL_VAR}/{model_path}",
                )
                fp = Footprint(
                    lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                    position=fp.position, rotation=fp.rotation,
                    layer=fp.layer, pads=fp.pads,
                    graphics=fp.graphics, texts=fp.texts,
                    lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
                    models=(thermal_model,),
                    footprint_source=fp.footprint_source,
                    fp_zones=fp.fp_zones,
                )
                _log.info(
                    "Updated parametric 3D model to thermal-pad variant: %s",
                    model_path,
                )
                break

    return fp


def _route_footprint(
    ref: str,
    value: str,
    fid: str,
    upper: str,
    layer: str,
    footprint_id: str,
    pins: tuple[Pin, ...] = (),
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

    fp = _route_fp_smd_ic(ref, value, fid, upper, pins=pins)
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
    pins: tuple[Pin, ...] = (),
) -> Footprint:
    """Route to the appropriate footprint generator based on *footprint_id*.

    Parsing rules (first match wins):

    - ``"R_<pkg>"`` or ``"C_<pkg>"``  → :func:`make_smd_resistor_capacitor`
    - ``"LED_<pkg>"``                  → :func:`make_smd_led`
    - ``"SOT-23*"``                    → :func:`make_sot23`
    - ``"SOT-223"``                    → :func:`make_sot223` (4 pads: 3 small + tab)
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
        pins: Component pin definitions.  When a pin with name ``PAD``,
            ``EP``, ``EPAD``, ``GND_PAD``, ``THERMAL``, or ``POWERPAD``
            (type ``POWER_IN``) is present, the footprint includes an
            exposed center thermal pad.

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
                # Re-apply registry offset after ESP32 enrichment replaced
                # the 3D model (enrichment sets offset to 0,0,0).
                fp = _apply_jlcpcb_model_offset(fp, footprint_id)

            # Add exposed thermal pad to JLCPCB footprints for dual-row
            # packages (SOIC/MSOP/TSSOP) when component pins indicate one
            # is needed but the cached footprint doesn't include it.
            if pins:
                thermal_pin = _find_thermal_pad_pin(pins)
                if thermal_pin is not None:
                    # Estimate exposed pad size from existing pad extents
                    if fp.pads:
                        max_x = max(abs(p.position.x) for p in fp.pads)
                        max_y = max(abs(p.position.y) for p in fp.pads)
                        ep_w = max(max_x * _DUAL_ROW_THERMAL_PAD_BODY_FRACTION,
                                   _QFN_THERMAL_PAD_MIN_MM)
                        ep_h = max(max_y * _DUAL_ROW_THERMAL_PAD_BODY_FRACTION,
                                   _QFN_THERMAL_PAD_MIN_MM)
                    else:
                        ep_w = ep_h = _QFN_THERMAL_PAD_MIN_MM
                    ep = _smd_pad(
                        thermal_pin.number, 0.0, 0.0,
                        round(ep_w, 2), round(ep_h, 2), fp.layer,
                    )
                    # If the pad number already exists as a gull-wing lead
                    # (e.g. SOIC-8 PowerPAD: pin 8 is thermal, not a lead),
                    # replace it.  Otherwise append.
                    existing_nums = {p.number for p in fp.pads}
                    if thermal_pin.number in existing_nums:
                        new_pads = tuple(
                            ep if p.number == thermal_pin.number else p
                            for p in fp.pads
                        )
                        action = "Replaced gull-wing pad"
                    else:
                        new_pads = (*fp.pads, ep)
                        action = "Added exposed thermal pad"
                    fp = Footprint(
                        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                        position=fp.position, rotation=fp.rotation,
                        layer=fp.layer, pads=new_pads,
                        graphics=fp.graphics, texts=fp.texts,
                        lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
                        models=fp.models, datasheet=fp.datasheet,
                        description=fp.description,
                        footprint_source=fp.footprint_source,
                        fp_zones=fp.fp_zones,
                    )
                    _log.info(
                        "%s #%s (%.1fx%.1fmm) on JLCPCB footprint for %s",
                        action, thermal_pin.number, ep_w, ep_h, ref,
                    )

                    # Update 3D model to thermal-pad variant if available
                    upper_fid = footprint_id.strip().upper()
                    for pkg_key, model_path in _THERMAL_3D_MODEL_VARIANTS.items():
                        if pkg_key in upper_fid:
                            thermal_model = Footprint3DModel(
                                path=f"{KICAD_3DMODEL_VAR}/{model_path}",
                            )
                            fp = Footprint(
                                lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                                position=fp.position, rotation=fp.rotation,
                                layer=fp.layer, pads=fp.pads,
                                graphics=fp.graphics, texts=fp.texts,
                                lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
                                models=(thermal_model,),
                                datasheet=fp.datasheet,
                                description=fp.description,
                                footprint_source=fp.footprint_source,
                                fp_zones=fp.fp_zones,
                            )
                            _log.info(
                                "Updated 3D model to thermal-pad variant: %s",
                                model_path,
                            )
                            break

            return fp

    fid = footprint_id.strip()
    upper = fid.upper()

    fp, _source = _route_footprint(ref, value, fid, upper, layer, footprint_id, pins=pins)

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
    "module": (0.5, 3.5),       # ESP32/W5500 — body extends beyond pad field (antenna, shield)
    "qfn": (0.25, 0.25),        # QFN/QFP/BGA — body ≈ pad field
    "qfp": (0.25, 0.25),
    "bga": (0.25, 0.25),
    "sot": (0.75, 0.75),        # SOT-23/223 — body wider than pads
    "passive": (0.1, 0.2),      # 0402/0603/0805 — body mostly fits between pads
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
    # Strip library prefix (e.g. "Resistor_SMD:R_0402_1005Metric" -> "R_0402_1005Metric")
    if ":" in fid:
        fid = fid.split(":", 1)[1]
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
