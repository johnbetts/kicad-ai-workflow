#!/usr/bin/env python3
"""One-time migration: split component_registry.json into footprint_catalog + parts_catalog.

Reads:
  - data/component_registry.json  (30 entries keyed by footprint type)
  - data/jlcpcb_basic_parts.json  (~36 JLCPCB basic parts)
  - scripts/train_*.py            (parsed for LCSC numbers, pin defs, values)

Outputs:
  - data/footprint_catalog.json   (unique footprint types with physical dims)
  - data/parts_catalog.json       (distinct parts keyed by LCSC or footprint::value)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

REGISTRY_PATH = DATA_DIR / "component_registry.json"
JLCPCB_PATH = DATA_DIR / "jlcpcb_basic_parts.json"
FOOTPRINT_OUT = DATA_DIR / "footprint_catalog.json"
PARTS_OUT = DATA_DIR / "parts_catalog.json"

LAST_VERIFIED_COMMIT = "19ae64d"


# ---------------------------------------------------------------------------
# Footprint catalog helpers
# ---------------------------------------------------------------------------

# Map old registry component_id -> canonical footprint_id
# SOIC-8_thermal maps to SOIC-8 (same physical package)
FOOTPRINT_ALIASES: dict[str, str] = {
    "SOIC-8_thermal": "SOIC-8",
}

# Footprint fields to extract from registry
FOOTPRINT_FIELDS = (
    "expected_pads",
    "expected_pad_type",
    "body_width_mm",
    "body_height_mm",
    "model_rotation_z",
    "model_offset_xy_max_mm",
    "kicad_ref_pad1_x",
    "kicad_ref_pad1_y",
)


def _make_footprint_entry(comp: dict[str, Any]) -> dict[str, Any]:
    """Build a footprint catalog entry from a registry component."""
    fid = comp.get("footprint_id", comp["component_id"])
    entry: dict[str, Any] = {
        "footprint_id": fid,
        "description": comp.get("description", ""),
    }
    for field in FOOTPRINT_FIELDS:
        entry[field] = comp.get(field)
    return entry


def build_footprint_catalog(registry: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate footprints from the registry."""
    footprints: dict[str, Any] = {}
    components = registry.get("components", {})

    for comp_id, comp in components.items():
        canonical_id = FOOTPRINT_ALIASES.get(comp_id, comp.get("footprint_id", comp_id))
        if canonical_id in footprints:
            continue  # already have the canonical entry
        entry = _make_footprint_entry(comp)
        entry["footprint_id"] = canonical_id
        footprints[canonical_id] = entry

    # Add footprints from training scripts that aren't in the registry
    extra_footprints = _discover_extra_footprints(footprints)
    footprints.update(extra_footprints)

    return {"schema_version": 1, "footprints": dict(sorted(footprints.items()))}


def _discover_extra_footprints(existing: dict[str, Any]) -> dict[str, Any]:
    """Add footprint stubs for types found in training scripts but not in registry."""
    # These are footprint IDs referenced in training scripts but missing from registry
    extra: dict[str, Any] = {}

    new_fps: dict[str, dict[str, Any]] = {
        "SOD-323": {
            "description": "SOD-323 small outline diode (1.7 x 1.25 mm)",
            "expected_pads": 2,
            "expected_pad_type": "smd",
            "body_width_mm": 1.7,
            "body_height_mm": 1.25,
            "model_rotation_z": 0.0,
            "model_offset_xy_max_mm": 0.5,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "LED_0603": {
            "description": "0603 chip LED (1.6 x 0.8 mm)",
            "expected_pads": 2,
            "expected_pad_type": "smd",
            "body_width_mm": 1.6,
            "body_height_mm": 0.8,
            "model_rotation_z": 0.0,
            "model_offset_xy_max_mm": 0.5,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "L_0805": {
            "description": "0805 chip inductor/ferrite bead (2.0 x 1.25 mm)",
            "expected_pads": 2,
            "expected_pad_type": "smd",
            "body_width_mm": 2.0,
            "body_height_mm": 1.25,
            "model_rotation_z": 0.0,
            "model_offset_xy_max_mm": 0.5,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "TerminalBlock_5.08mm_2P": {
            "description": "2-position 5.08mm pitch screw terminal block (10.2 x 8.6 mm)",
            "expected_pads": 2,
            "expected_pad_type": "thru_hole",
            "body_width_mm": 10.2,
            "body_height_mm": 8.6,
            "model_rotation_z": 180.0,
            "model_offset_xy_max_mm": 6.0,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "TerminalBlock_5.08mm_3P": {
            "description": "3-position 5.08mm pitch screw terminal block (15.24 x 8.6 mm)",
            "expected_pads": 3,
            "expected_pad_type": "thru_hole",
            "body_width_mm": 15.24,
            "body_height_mm": 8.6,
            "model_rotation_z": 180.0,
            "model_offset_xy_max_mm": 6.0,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "PinHeader_1x02_P2.54mm_Vertical": {
            "description": "1x2 pin header, 2.54mm pitch, vertical (5.1 x 2.5 mm)",
            "expected_pads": 2,
            "expected_pad_type": "thru_hole",
            "body_width_mm": 2.5,
            "body_height_mm": 5.1,
            "model_rotation_z": 0.0,
            "model_offset_xy_max_mm": 5.0,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "PinHeader_1x06_P2.54mm_Vertical": {
            "description": "1x6 pin header, 2.54mm pitch, vertical (15.3 x 2.5 mm)",
            "expected_pads": 6,
            "expected_pad_type": "thru_hole",
            "body_width_mm": 2.5,
            "body_height_mm": 15.3,
            "model_rotation_z": 0.0,
            "model_offset_xy_max_mm": 5.0,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "SW_Push_4.5x4.5mm": {
            "description": "4.5x4.5mm SMD tactile push switch",
            "expected_pads": 4,
            "expected_pad_type": "smd",
            "body_width_mm": 4.5,
            "body_height_mm": 4.5,
            "model_rotation_z": 0.0,
            "model_offset_xy_max_mm": 0.5,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "USB-C": {
            "description": "USB Type-C SMD connector (9.0 x 7.3 mm)",
            "expected_pads": 9,
            "expected_pad_type": "smd",
            "body_width_mm": 9.0,
            "body_height_mm": 7.3,
            "model_rotation_z": 0.0,
            "model_offset_xy_max_mm": 1.0,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "LQFP-48": {
            "description": "LQFP-48 low-profile quad flat package (7.0 x 7.0 mm)",
            "expected_pads": 49,
            "expected_pad_type": "smd",
            "body_width_mm": 7.0,
            "body_height_mm": 7.0,
            "model_rotation_z": 0.0,
            "model_offset_xy_max_mm": 1.0,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "RJ45_HR911105A": {
            "description": "RJ45 Ethernet connector HR911105A with integrated magnetics (19.6 x 16.8 mm)",
            "expected_pads": 14,
            "expected_pad_type": "thru_hole",
            "body_width_mm": 19.6,
            "body_height_mm": 16.8,
            "model_rotation_z": 0.0,
            "model_offset_xy_max_mm": 1.0,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "Relay_THT:Relay_SPDT_SANYOU_SRD_Series_Form_C": {
            "description": "SANYOU SRD series SPDT relay, THT (19.0 x 15.5 mm)",
            "expected_pads": 5,
            "expected_pad_type": "thru_hole",
            "body_width_mm": 19.0,
            "body_height_mm": 15.5,
            "model_rotation_z": 0.0,
            "model_offset_xy_max_mm": 2.0,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
        "PinHeader_1x04": {
            "description": "1x4 pin header, 2.54mm pitch (10.2 x 2.5 mm)",
            "expected_pads": 4,
            "expected_pad_type": "thru_hole",
            "body_width_mm": 2.5,
            "body_height_mm": 10.2,
            "model_rotation_z": 0.0,
            "model_offset_xy_max_mm": 5.0,
            "kicad_ref_pad1_x": 0.0,
            "kicad_ref_pad1_y": 0.0,
        },
    }

    for fp_id, fp_data in new_fps.items():
        if fp_id not in existing:
            entry = {"footprint_id": fp_id}
            entry.update(fp_data)
            extra[fp_id] = entry

    return extra


# ---------------------------------------------------------------------------
# Parts catalog helpers
# ---------------------------------------------------------------------------

def _ref_prefix(ref: str) -> str:
    """Extract ref prefix (e.g., 'R' from 'R1', 'SW' from 'SW1')."""
    m = re.match(r"([A-Za-z]+)", ref)
    return m.group(1) if m else "X"


def _part_key(lcsc: str | None, footprint_id: str, value: str) -> str:
    """Generate part key: LCSC number if available, else footprint::value."""
    if lcsc:
        return lcsc
    return f"{footprint_id}::{value}"


def _normalize_pin_type(pt: str) -> str:
    """Normalize pin type strings from training scripts."""
    # Training scripts use PinType.INPUT etc; we store lowercase
    mapping = {
        "input": "input",
        "output": "output",
        "bidirectional": "bidirectional",
        "power_in": "power_in",
        "power_out": "power_out",
        "passive": "passive",
        "unspecified": "unspecified",
    }
    return mapping.get(pt.lower(), pt.lower())


def build_parts_catalog(
    registry: dict[str, Any],
    jlcpcb: dict[str, Any],
) -> dict[str, Any]:
    """Build parts catalog from registry + JLCPCB basic parts + training scripts."""
    parts: dict[str, Any] = {}

    # Phase 1: Extract parts from component registry
    _extract_registry_parts(registry, parts)

    # Phase 2: Extract parts from training scripts (richer pin data)
    _extract_training_script_parts(parts)

    # Phase 3: Add JLCPCB basic parts not yet covered
    _extract_jlcpcb_parts(jlcpcb, parts)

    return {"schema_version": 1, "parts": dict(sorted(parts.items()))}


def _extract_registry_parts(registry: dict[str, Any], parts: dict[str, Any]) -> None:
    """Extract part entries from the component registry."""
    components = registry.get("components", {})

    for comp_id, comp in components.items():
        lcsc = comp.get("lcsc")
        footprint_id = FOOTPRINT_ALIASES.get(comp_id, comp.get("footprint_id", comp_id))
        value = comp.get("value", "")
        ref = comp.get("ref", "")

        key = _part_key(lcsc, footprint_id, value)

        pins = []
        for p in comp.get("pins", []):
            pins.append({
                "number": p["number"],
                "name": p["name"],
                "pin_type": _normalize_pin_type(p.get("pin_type", "passive")),
            })

        parts[key] = {
            "part_id": key,
            "lcsc": lcsc,
            "mpn": None,
            "manufacturer": None,
            "value": value,
            "footprint_id": footprint_id,
            "description": comp.get("description", ""),
            "datasheet_url": comp.get("datasheet_url"),
            "ref_prefix": _ref_prefix(ref),
            "pins": pins,
            "verification_status": comp.get("verification_status", "unverified"),
            "last_verified_commit": comp.get("last_verified_commit"),
            "known_issues": comp.get("known_issues", []),
            "verified_fixes": comp.get("verified_fixes", []),
        }


# ---------------------------------------------------------------------------
# Training script extraction
# ---------------------------------------------------------------------------

# Manually extracted component definitions from training scripts.
# These are the IC-level and unique-passive components with pin definitions.

_TRAINING_PARTS: list[dict[str, Any]] = [
    # --- train_mcu_core.py ---
    {
        "lcsc": "C2913202",
        "value": "ESP32-S3-WROOM-1",
        "footprint_id": "ESP32-S3-WROOM-1",
        "ref": "U1",
        "description": "ESP32-S3-WROOM-1 WiFi+BLE module with antenna (18.0 x 25.5 mm)",
        "pins": [
            {"number": "1", "name": "GND", "pin_type": "power_in"},
            {"number": "2", "name": "3V3", "pin_type": "power_in"},
            {"number": "3", "name": "EN", "pin_type": "input"},
            {"number": "4", "name": "IO4", "pin_type": "bidirectional"},
            {"number": "5", "name": "IO5", "pin_type": "bidirectional"},
            {"number": "6", "name": "IO6", "pin_type": "bidirectional"},
            {"number": "7", "name": "IO7", "pin_type": "bidirectional"},
            {"number": "8", "name": "IO15", "pin_type": "bidirectional"},
            {"number": "9", "name": "IO16", "pin_type": "bidirectional"},
            {"number": "10", "name": "IO17", "pin_type": "bidirectional"},
            {"number": "11", "name": "IO18", "pin_type": "bidirectional"},
            {"number": "12", "name": "IO8", "pin_type": "bidirectional"},
            {"number": "13", "name": "IO19", "pin_type": "bidirectional"},
            {"number": "14", "name": "IO20", "pin_type": "bidirectional"},
            {"number": "15", "name": "IO3", "pin_type": "bidirectional"},
            {"number": "16", "name": "IO46", "pin_type": "bidirectional"},
            {"number": "17", "name": "IO9", "pin_type": "bidirectional"},
            {"number": "18", "name": "IO10", "pin_type": "bidirectional"},
            {"number": "19", "name": "IO11", "pin_type": "bidirectional"},
            {"number": "20", "name": "IO12", "pin_type": "bidirectional"},
            {"number": "21", "name": "IO13", "pin_type": "bidirectional"},
            {"number": "22", "name": "IO14", "pin_type": "bidirectional"},
            {"number": "23", "name": "IO21", "pin_type": "bidirectional"},
            {"number": "24", "name": "IO47", "pin_type": "bidirectional"},
            {"number": "25", "name": "IO48", "pin_type": "bidirectional"},
            {"number": "26", "name": "IO45", "pin_type": "bidirectional"},
            {"number": "27", "name": "IO0", "pin_type": "bidirectional"},
            {"number": "28", "name": "IO35", "pin_type": "bidirectional"},
            {"number": "29", "name": "IO36", "pin_type": "bidirectional"},
            {"number": "30", "name": "IO37", "pin_type": "bidirectional"},
            {"number": "31", "name": "IO38", "pin_type": "bidirectional"},
            {"number": "32", "name": "IO39", "pin_type": "bidirectional"},
            {"number": "33", "name": "IO40", "pin_type": "bidirectional"},
            {"number": "34", "name": "IO41", "pin_type": "bidirectional"},
            {"number": "35", "name": "IO42", "pin_type": "bidirectional"},
            {"number": "36", "name": "RXD0", "pin_type": "output"},
            {"number": "37", "name": "TXD0", "pin_type": "input"},
            {"number": "38", "name": "IO2", "pin_type": "output"},
            {"number": "39", "name": "IO1", "pin_type": "bidirectional"},
            {"number": "40", "name": "GND", "pin_type": "power_in"},
            {"number": "41", "name": "GND", "pin_type": "power_in"},
        ],
    },
    {
        "lcsc": "C25905",
        "value": "5.1K",
        "footprint_id": "R_0402",
        "ref": "R3",
        "description": "5.1K USB-C CC pull-down resistor, 0402",
        "pins": [],
    },
    {
        "lcsc": "C2286",
        "value": "LED",
        "footprint_id": "LED_0603",
        "ref": "D1",
        "description": "0603 red LED indicator",
        "pins": [
            {"number": "1", "name": "A", "pin_type": "passive"},
            {"number": "2", "name": "K", "pin_type": "passive"},
        ],
    },
    {
        "lcsc": "C23138",
        "value": "330R",
        "footprint_id": "R_0603",
        "ref": "R5",
        "description": "330 ohm LED current limiting resistor, 0603",
        "pins": [],
    },
    # --- train_power_chain.py ---
    {
        "lcsc": "C9865",
        "value": "TPS54331",
        "footprint_id": "SOIC-8",
        "ref": "U1",
        "mpn": "TPS54331DR",
        "description": "3A step-down (buck) converter, SOIC-8 with thermal pad",
        "pins": [
            {"number": "1", "name": "BOOT", "pin_type": "input"},
            {"number": "2", "name": "VIN", "pin_type": "power_in"},
            {"number": "3", "name": "EN", "pin_type": "input"},
            {"number": "4", "name": "SS", "pin_type": "input"},
            {"number": "5", "name": "VSNS", "pin_type": "input"},
            {"number": "6", "name": "GND", "pin_type": "power_in"},
            {"number": "7", "name": "PH", "pin_type": "output"},
            {"number": "8", "name": "PAD", "pin_type": "power_in"},
        ],
    },
    {
        "lcsc": "C96950",
        "value": "10uH",
        "footprint_id": "L_1210",
        "ref": "L1",
        "description": "10uH power inductor, 1210 package",
        "pins": [],
    },
    {
        "lcsc": "C15850",
        "value": "10uF",
        "footprint_id": "C_0805",
        "ref": "C1",
        "description": "10uF ceramic capacitor, 0805",
        "pins": [],
    },
    {
        "lcsc": "C159842",
        "value": "22uF",
        "footprint_id": "C_0805",
        "ref": "C2",
        "description": "22uF ceramic capacitor, 0805",
        "pins": [],
    },
    {
        "lcsc": "C49678",
        "value": "100nF",
        "footprint_id": "C_0402",
        "ref": "C3",
        "description": "100nF ceramic capacitor, 0402",
        "pins": [],
    },
    {
        "lcsc": "C17407",
        "value": "100K",
        "footprint_id": "R_0402",
        "ref": "R1",
        "description": "100K feedback resistor (top), 0402",
        "pins": [],
    },
    {
        "lcsc": "C17390",
        "value": "33K",
        "footprint_id": "R_0402",
        "ref": "R2",
        "description": "33K feedback resistor (bottom), 0402",
        "pins": [],
    },
    {
        "lcsc": "C123899",
        "value": "SS14",
        "footprint_id": "SOD-323",
        "ref": "D1",
        "description": "SS14 Schottky diode, SOD-323",
        "pins": [
            {"number": "1", "name": "A", "pin_type": "passive"},
            {"number": "2", "name": "K", "pin_type": "passive"},
        ],
    },
    {
        "lcsc": "C6186",
        "value": "AMS1117-3.3",
        "footprint_id": "SOT-223",
        "ref": "U2",
        "mpn": "AMS1117-3.3",
        "description": "3.3V 1A LDO voltage regulator, SOT-223",
        "pins": [
            {"number": "1", "name": "GND", "pin_type": "power_in"},
            {"number": "2", "name": "VOUT", "pin_type": "power_out"},
            {"number": "3", "name": "VIN", "pin_type": "power_in"},
            {"number": "4", "name": "VOUT_TAB", "pin_type": "power_out"},
        ],
    },
    {
        "lcsc": "C8269",
        "value": "Screw_Terminal_2P",
        "footprint_id": "TerminalBlock_5.08mm_2P",
        "ref": "J1",
        "description": "2-position 5.08mm screw terminal block",
        "pins": [
            {"number": "1", "name": "GND", "pin_type": "power_in"},
            {"number": "2", "name": "+24V", "pin_type": "power_in"},
        ],
    },
    {
        "lcsc": "C124375",
        "value": "Pin_Header_2P",
        "footprint_id": "PinHeader_1x02_P2.54mm_Vertical",
        "ref": "J2",
        "description": "1x2 pin header for power output test point",
        "pins": [],
    },
    # --- train_ethernet.py ---
    {
        "lcsc": "C32843",
        "value": "W5500",
        "footprint_id": "LQFP-48",
        "ref": "U1",
        "mpn": "W5500",
        "description": "WIZnet W5500 hardwired TCP/IP Ethernet controller, LQFP-48",
        "pins": [
            {"number": "1", "name": "GND", "pin_type": "power_in"},
            {"number": "2", "name": "AVDD", "pin_type": "power_in"},
            {"number": "3", "name": "EXRES1", "pin_type": "input"},
            {"number": "4", "name": "MOSI", "pin_type": "input"},
            {"number": "5", "name": "MISO", "pin_type": "output"},
            {"number": "6", "name": "SCLK", "pin_type": "input"},
            {"number": "7", "name": "SCSn", "pin_type": "input"},
            {"number": "8", "name": "INTn", "pin_type": "output"},
            {"number": "9", "name": "RSTn", "pin_type": "input"},
            {"number": "10", "name": "VCC", "pin_type": "power_in"},
            {"number": "11", "name": "GND", "pin_type": "power_in"},
            {"number": "12", "name": "LINKLED", "pin_type": "output"},
            {"number": "13", "name": "ACTLED", "pin_type": "output"},
            {"number": "14", "name": "GND", "pin_type": "power_in"},
            {"number": "15", "name": "VCC", "pin_type": "power_in"},
            {"number": "16", "name": "TXP", "pin_type": "output"},
            {"number": "17", "name": "TXN", "pin_type": "output"},
            {"number": "18", "name": "GND", "pin_type": "power_in"},
            {"number": "19", "name": "RXP", "pin_type": "input"},
            {"number": "20", "name": "RXN", "pin_type": "input"},
            {"number": "21", "name": "GND", "pin_type": "power_in"},
            {"number": "22", "name": "AVDD2", "pin_type": "power_in"},
            {"number": "23", "name": "XI", "pin_type": "input"},
            {"number": "24", "name": "XO", "pin_type": "output"},
            {"number": "25", "name": "GND", "pin_type": "power_in"},
            {"number": "26", "name": "VCC", "pin_type": "power_in"},
            {"number": "27", "name": "GND", "pin_type": "power_in"},
            {"number": "28", "name": "GND", "pin_type": "power_in"},
            {"number": "29", "name": "VCC", "pin_type": "power_in"},
            {"number": "30", "name": "GND", "pin_type": "power_in"},
            {"number": "31", "name": "GND", "pin_type": "power_in"},
            {"number": "32", "name": "VCC", "pin_type": "power_in"},
            {"number": "33", "name": "GND", "pin_type": "power_in"},
            {"number": "34", "name": "GND", "pin_type": "power_in"},
            {"number": "35", "name": "VCC", "pin_type": "power_in"},
            {"number": "36", "name": "GND", "pin_type": "power_in"},
            {"number": "37", "name": "GND", "pin_type": "power_in"},
            {"number": "38", "name": "VCC", "pin_type": "power_in"},
            {"number": "39", "name": "GND", "pin_type": "power_in"},
            {"number": "40", "name": "GND", "pin_type": "power_in"},
            {"number": "41", "name": "VCC", "pin_type": "power_in"},
            {"number": "42", "name": "GND", "pin_type": "power_in"},
            {"number": "43", "name": "GND", "pin_type": "power_in"},
            {"number": "44", "name": "VCC", "pin_type": "power_in"},
            {"number": "45", "name": "GND", "pin_type": "power_in"},
            {"number": "46", "name": "GND", "pin_type": "power_in"},
            {"number": "47", "name": "VCC", "pin_type": "power_in"},
            {"number": "48", "name": "GND", "pin_type": "power_in"},
            {"number": "49", "name": "PAD", "pin_type": "power_in"},
        ],
    },
    {
        "lcsc": "C13738",
        "value": "25MHz",
        "footprint_id": "Crystal_SMD_3215",
        "ref": "Y1",
        "description": "25MHz SMD crystal oscillator, 3215 package",
        "pins": [],
    },
    {
        "lcsc": "C12074",
        "value": "HR911105A",
        "footprint_id": "RJ45_HR911105A",
        "ref": "J1",
        "description": "RJ45 Ethernet connector with integrated magnetics (HR911105A)",
        "pins": [
            {"number": "1", "name": "TD+", "pin_type": "bidirectional"},
            {"number": "2", "name": "TD-", "pin_type": "bidirectional"},
            {"number": "3", "name": "RD+", "pin_type": "bidirectional"},
            {"number": "4", "name": "NC1", "pin_type": "passive"},
            {"number": "5", "name": "NC2", "pin_type": "passive"},
            {"number": "6", "name": "RD-", "pin_type": "bidirectional"},
            {"number": "7", "name": "NC3", "pin_type": "passive"},
            {"number": "8", "name": "NC4", "pin_type": "passive"},
            {"number": "9", "name": "LED_G+", "pin_type": "passive"},
            {"number": "10", "name": "LED_G-", "pin_type": "passive"},
            {"number": "11", "name": "LED_Y+", "pin_type": "passive"},
            {"number": "12", "name": "LED_Y-", "pin_type": "passive"},
            {"number": "13", "name": "SHIELD1", "pin_type": "passive"},
            {"number": "14", "name": "SHIELD2", "pin_type": "passive"},
        ],
    },
    {
        "lcsc": "C25129",
        "value": "49.9R",
        "footprint_id": "R_0603",
        "ref": "R1",
        "description": "49.9 ohm Ethernet termination resistor, 0603",
        "pins": [],
    },
    {
        "lcsc": "C17401",
        "value": "12.1K",
        "footprint_id": "R_0402",
        "ref": "R3",
        "description": "12.1K W5500 EXRES reference resistor, 0402",
        "pins": [],
    },
    {
        "lcsc": "C1804",
        "value": "22pF",
        "footprint_id": "C_0402",
        "ref": "C4",
        "description": "22pF crystal load capacitor, 0402 (0603 body)",
        "pins": [],
    },
    # --- train_analog_input.py ---
    {
        "lcsc": "C37593",
        "value": "ADS1115",
        "footprint_id": "MSOP-10",
        "ref": "U1",
        "mpn": "ADS1115IDGSR",
        "description": "16-bit 4-channel I2C ADC, MSOP-10",
        "pins": [
            {"number": "1", "name": "ADDR", "pin_type": "input"},
            {"number": "2", "name": "ALERT", "pin_type": "output"},
            {"number": "3", "name": "GND", "pin_type": "power_in"},
            {"number": "4", "name": "AIN0", "pin_type": "input"},
            {"number": "5", "name": "AIN1", "pin_type": "input"},
            {"number": "6", "name": "AIN2", "pin_type": "input"},
            {"number": "7", "name": "AIN3", "pin_type": "input"},
            {"number": "8", "name": "VDD", "pin_type": "power_in"},
            {"number": "9", "name": "SDA", "pin_type": "bidirectional"},
            {"number": "10", "name": "SCL", "pin_type": "input"},
        ],
    },
    {
        "lcsc": "C17414",
        "value": "10K",
        "footprint_id": "R_0402",
        "ref": "R1",
        "description": "10K resistor, 0402",
        "pins": [],
    },
    {
        "lcsc": "C118739",
        "value": "PESD3V3",
        "footprint_id": "SOD-323",
        "ref": "D1",
        "description": "3.3V ESD protection / TVS diode, SOD-323",
        "pins": [
            {"number": "1", "name": "A", "pin_type": "passive"},
            {"number": "2", "name": "K", "pin_type": "passive"},
        ],
    },
    {
        "lcsc": "C17673",
        "value": "4.7K",
        "footprint_id": "R_0402",
        "ref": "R9",
        "description": "4.7K I2C pull-up resistor, 0402",
        "pins": [],
    },
    # --- train_relay_group.py ---
    {
        "lcsc": "C35449",
        "value": "SRD-05VDC-SL-C",
        "footprint_id": "Relay_THT:Relay_SPDT_SANYOU_SRD_Series_Form_C",
        "ref": "K1",
        "mpn": "SRD-05VDC-SL-C",
        "description": "SANYOU SRD series 5V SPDT relay",
        "pins": [
            {"number": "1", "name": "COM", "pin_type": "passive"},
            {"number": "2", "name": "COIL-", "pin_type": "passive"},
            {"number": "3", "name": "NO", "pin_type": "passive"},
            {"number": "4", "name": "NC", "pin_type": "passive"},
            {"number": "5", "name": "COIL+", "pin_type": "passive"},
        ],
    },
    {
        "lcsc": "C727114",
        "value": "SS8050",
        "footprint_id": "SOT-23",
        "ref": "Q1",
        "mpn": "SS8050",
        "description": "NPN transistor for relay driver, SOT-23",
        "pins": [
            {"number": "1", "name": "B", "pin_type": "input"},
            {"number": "2", "name": "C", "pin_type": "output"},
            {"number": "3", "name": "E", "pin_type": "passive"},
        ],
    },
    {
        "lcsc": "C81598",
        "value": "1N4148",
        "footprint_id": "SOD-323",
        "ref": "D1",
        "description": "1N4148 flyback protection diode, SOD-323",
        "pins": [
            {"number": "1", "name": "A", "pin_type": "passive"},
            {"number": "2", "name": "K", "pin_type": "passive"},
        ],
    },
    {
        "lcsc": "C17513",
        "value": "1K",
        "footprint_id": "R_0402",
        "ref": "R1",
        "description": "1K relay driver base resistor, 0402",
        "pins": [],
    },
    {
        "lcsc": "C8465",
        "value": "Screw_Terminal_3P",
        "footprint_id": "TerminalBlock_5.08mm_3P",
        "ref": "J1",
        "description": "3-position 5.08mm screw terminal block for relay output",
        "pins": [
            {"number": "1", "name": "COM", "pin_type": "passive"},
            {"number": "2", "name": "NO", "pin_type": "passive"},
            {"number": "3", "name": "NC", "pin_type": "passive"},
        ],
    },
    {
        "lcsc": "C1015",
        "value": "600R@100MHz",
        "footprint_id": "L_0805",
        "ref": "L1",
        "description": "600 ohm @ 100MHz ferrite bead, 0805",
        "pins": [],
    },
    {
        "lcsc": "C318884",
        "value": "RESET",
        "footprint_id": "SW_Push_4.5x4.5mm",
        "ref": "SW1",
        "description": "Tactile push switch (boot/reset)",
        "pins": [
            {"number": "1", "name": "1", "pin_type": "passive"},
            {"number": "2", "name": "2", "pin_type": "passive"},
        ],
    },
]


def _extract_training_script_parts(parts: dict[str, Any]) -> None:
    """Add/update parts from training script definitions."""
    for tp in _TRAINING_PARTS:
        lcsc = tp.get("lcsc")
        footprint_id = tp["footprint_id"]
        value = tp["value"]
        ref = tp.get("ref", "X1")
        key = _part_key(lcsc, footprint_id, value)

        # If key already exists, merge: training scripts are more authoritative
        # than the registry for ICs (value, footprint_id, pins, mpn).
        if key in parts:
            existing = parts[key]
            # Upgrade pins if training script has them and existing doesn't
            if tp.get("pins") and not existing.get("pins"):
                existing["pins"] = tp["pins"]
            # Upgrade mpn if available
            if tp.get("mpn") and not existing.get("mpn"):
                existing["mpn"] = tp["mpn"]
            # Training scripts have correct value/footprint for ICs
            # (registry sometimes has wrong value for a given LCSC)
            if tp.get("pins"):
                existing["value"] = tp["value"]
                existing["footprint_id"] = tp["footprint_id"]
            # Upgrade description if the training script has a better one
            if tp.get("description") and (
                not existing.get("description")
                or len(tp["description"]) > len(existing.get("description", ""))
            ):
                existing["description"] = tp["description"]
            continue

        parts[key] = {
            "part_id": key,
            "lcsc": lcsc,
            "mpn": tp.get("mpn"),
            "manufacturer": None,
            "value": value,
            "footprint_id": footprint_id,
            "description": tp.get("description", ""),
            "datasheet_url": None,
            "ref_prefix": _ref_prefix(ref),
            "pins": tp.get("pins", []),
            "verification_status": "verified" if lcsc else "unverified",
            "last_verified_commit": LAST_VERIFIED_COMMIT if lcsc else None,
            "known_issues": [],
            "verified_fixes": [],
        }


def _footprint_to_category(pkg: str) -> str | None:
    """Map a JLCPCB package string to a footprint_id in the catalog."""
    mapping: dict[str, str] = {
        "0805": "R_0805",  # overridden per category below
        "0603": "R_0603",
        "0402": "R_0402",
        "1206": "R_1206",
        "SOT-23": "SOT-23",
        "SOT-23-5": "SOT-23-5",
        "SOT-223": "SOT-223",
        "SOP-8": "SOIC-8",
        "LQFP-48": "LQFP-48",
        "ESP32-WROOM-32E": "ESP32-WROOM-32E",
        "ESP32-S3-WROOM-1": "ESP32-S3-WROOM-1",
        "SOD-123": "SOD-123",
        "USB-C-SMD": "USB-C",
        "RJ45-SMD": "RJ45",
        "PinHeader_1x04_P2.54mm": "PinHeader_1x04_P2.54mm",
        "SW_SPST_B3U-1000P": "SW_Push_SMD_6x6",
        "Buzzer_12x9.5mm": "Buzzer_12x9.5mm",
    }
    return mapping.get(pkg)


def _jlcpcb_footprint_id(part: dict[str, Any]) -> str:
    """Determine footprint_id for a JLCPCB basic part."""
    pkg = part.get("package", "")
    cat = part.get("category", "")

    # For passives, package like "0805" maps to different footprints by category
    if pkg in ("0805", "0603", "0402", "1206"):
        if cat == "resistor":
            return f"R_{pkg}"
        elif cat == "capacitor":
            return f"C_{pkg}"
        elif cat == "led":
            return f"LED_{pkg}"
        elif cat == "inductor":
            return f"L_{pkg}"
        # default: assume resistor
        return f"R_{pkg}"

    mapped = _footprint_to_category(pkg)
    if mapped:
        return mapped
    return pkg


def _jlcpcb_ref_prefix(cat: str) -> str:
    """Determine ref prefix from JLCPCB category."""
    prefix_map: dict[str, str] = {
        "resistor": "R",
        "capacitor": "C",
        "led": "D",
        "inductor": "L",
        "transistor_npn": "Q",
        "diode_switching": "D",
        "diode_esd": "D",
        "ldo": "U",
        "usb_uart": "U",
        "mcu_module": "U",
        "ethernet": "U",
        "connector_usb": "J",
        "connector_rj45": "J",
        "connector_header": "J",
        "switch": "SW",
        "buzzer": "BZ",
    }
    return prefix_map.get(cat, "X")


def _extract_jlcpcb_parts(jlcpcb: dict[str, Any], parts: dict[str, Any]) -> None:
    """Add JLCPCB basic parts that aren't already in the catalog."""
    for jp in jlcpcb.get("parts", []):
        lcsc = jp.get("lcsc")
        if not lcsc:
            continue

        # Skip if already present (from registry or training scripts)
        if lcsc in parts:
            continue

        footprint_id = _jlcpcb_footprint_id(jp)
        value = jp.get("value", "")
        cat = jp.get("category", "")

        parts[lcsc] = {
            "part_id": lcsc,
            "lcsc": lcsc,
            "mpn": None,
            "manufacturer": jp.get("mfr"),
            "value": value,
            "footprint_id": footprint_id,
            "description": f"{value} {cat.replace('_', ' ')} ({jp.get('package', '')})",
            "datasheet_url": None,
            "ref_prefix": _jlcpcb_ref_prefix(cat),
            "pins": [],
            "verification_status": "unverified",
            "last_verified_commit": None,
            "known_issues": [],
            "verified_fixes": [],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Component Registry Migration")
    print("=" * 60)

    # Read inputs
    if not REGISTRY_PATH.exists():
        print(f"ERROR: {REGISTRY_PATH} not found")
        sys.exit(1)
    if not JLCPCB_PATH.exists():
        print(f"ERROR: {JLCPCB_PATH} not found")
        sys.exit(1)

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    with open(JLCPCB_PATH) as f:
        jlcpcb = json.load(f)

    n_registry = len(registry.get("components", {}))
    n_jlcpcb = len(jlcpcb.get("parts", []))
    print(f"\nInputs:")
    print(f"  component_registry.json: {n_registry} entries")
    print(f"  jlcpcb_basic_parts.json: {n_jlcpcb} entries")
    print(f"  Training script parts:   {len(_TRAINING_PARTS)} entries")

    # Build catalogs
    footprint_catalog = build_footprint_catalog(registry)
    parts_catalog = build_parts_catalog(registry, jlcpcb)

    # Write outputs
    with open(FOOTPRINT_OUT, "w") as f:
        json.dump(footprint_catalog, f, indent=2)
        f.write("\n")
    print(f"\nWrote: {FOOTPRINT_OUT}")

    with open(PARTS_OUT, "w") as f:
        json.dump(parts_catalog, f, indent=2)
        f.write("\n")
    print(f"Wrote: {PARTS_OUT}")

    # Summary
    n_footprints = len(footprint_catalog["footprints"])
    n_parts = len(parts_catalog["parts"])
    n_with_lcsc = sum(
        1 for p in parts_catalog["parts"].values() if p.get("lcsc")
    )
    n_with_pins = sum(
        1 for p in parts_catalog["parts"].values() if p.get("pins")
    )

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Footprints:       {n_footprints}")
    print(f"  Parts:            {n_parts}")
    print(f"  Parts with LCSC:  {n_with_lcsc}")
    print(f"  Parts with pins:  {n_with_pins}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
