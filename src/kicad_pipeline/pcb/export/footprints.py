"""Export utilities for footprints.

Provides functions for exporting footprints to various formats and
generating footprint statistics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Footprint export functions
# ---------------------------------------------------------------------------


def export_footprint_json(fp: Footprint, output_path: Path) -> None:
    """Export footprint to JSON format.

    Args:
        fp: Footprint to export
        output_path: Path to write JSON file
    """
    data = {
        "lib_id": fp.lib_id,
        "ref": fp.ref,
        "value": fp.value,
        "position": {"x": fp.position.x, "y": fp.position.y},
        "rotation": fp.rotation,
        "layer": fp.layer,
        "pads": [
            {
                "number": pad.number,
                "position": {"x": pad.position.x, "y": pad.position.y},
                "size": pad.size,
                "drill": getattr(pad, 'drill', None),
                "shape": getattr(pad, 'shape', 'circle'),
                "layer": pad.layer,
            }
            for pad in fp.pads
        ],
        "graphics": [
            {
                "type": type(g).__name__,
                "layer": getattr(g, 'layer', ''),
                "width": getattr(g, 'width', 0.1),
            }
            for g in fp.graphics
        ],
        "models": [
            {
                "path": model.path,
                "offset": list(model.offset),
                "scale": list(model.scale),
                "rotate": list(model.rotate),
            }
            for model in fp.models
        ],
        "lcsc": fp.lcsc,
        "datasheet": fp.datasheet,
        "description": fp.description,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_footprint_stats(fp: Footprint) -> dict:
    """Generate statistics for a footprint.

    Args:
        fp: Footprint to analyze

    Returns:
        Dictionary with footprint statistics
    """
    stats = {
        "lib_id": fp.lib_id,
        "ref": fp.ref,
        "pad_count": len(fp.pads),
        "graphic_count": len(fp.graphics),
        "text_count": len(fp.texts),
        "model_count": len(fp.models),
        "layer": fp.layer,
    }

    if fp.pads:
        # Pad statistics
        pad_sizes = [pad.size for pad in fp.pads]
        stats["pad_sizes"] = pad_sizes

        # Position statistics
        xs = [p.position.x for p in fp.pads]
        ys = [p.position.y for p in fp.pads]
        stats["pad_bbox"] = {
            "x_min": min(xs),
            "x_max": max(xs),
            "y_min": min(ys),
            "y_max": max(ys),
            "width": max(xs) - min(xs),
            "height": max(ys) - min(ys),
        }

    # Layer usage
    layers = set()
    for pad in fp.pads:
        layers.add(pad.layer)
    for graphic in fp.graphics:
        if hasattr(graphic, 'layer'):
            layers.add(graphic.layer)
    stats["layers_used"] = sorted(layers)

    # Attribute analysis
    stats["attributes"] = fp.attr

    return stats


def generate_footprint_report(footprints: list[Footprint], output_path: Path) -> None:
    """Generate a comprehensive report for multiple footprints.

    Args:
        footprints: List of footprints to analyze
        output_path: Path to write report
    """
    report = {
        "summary": {
            "total_footprints": len(footprints),
            "unique_lib_ids": len(set(fp.lib_id for fp in footprints)),
            "total_pads": sum(len(fp.pads) for fp in footprints),
            "footprints_with_models": sum(1 for fp in footprints if fp.models),
        },
        "footprints": []
    }

    for fp in footprints:
        fp_stats = export_footprint_stats(fp)
        report["footprints"].append(fp_stats)

    # Component type breakdown
    component_types = {}
    for fp in footprints:
        lib_id = fp.lib_id.upper()
        if 'RESISTOR' in lib_id or lib_id.startswith('R_'):
            comp_type = 'resistor'
        elif 'CAPACITOR' in lib_id or lib_id.startswith('C_'):
            comp_type = 'capacitor'
        elif 'INDUCTOR' in lib_id or lib_id.startswith('L_'):
            comp_type = 'inductor'
        elif 'DIODE' in lib_id or lib_id.startswith('D_'):
            comp_type = 'diode'
        elif 'LED' in lib_id:
            comp_type = 'led'
        elif any(x in lib_id for x in ('QFN', 'QFP', 'LQFP', 'TQFP', 'SOIC')):
            comp_type = 'ic'
        elif 'SOT' in lib_id:
            comp_type = 'transistor'
        elif any(x in lib_id for x in ('CONN', 'HEADER', 'USB', 'RJ45')):
            comp_type = 'connector'
        else:
            comp_type = 'other'

        component_types[comp_type] = component_types.get(comp_type, 0) + 1

    report["component_breakdown"] = component_types

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def validate_export_format(fp: Footprint, format_type: str) -> tuple[str, ...]:
    """Validate footprint compatibility with export format.

    Args:
        fp: Footprint to validate
        format_type: Export format ('kicad', 'altium', 'eagle', etc.)

    Returns:
        Tuple of compatibility issues
    """
    issues = []

    if format_type.lower() == 'kicad':
        # KiCad-specific validations
        if not fp.lib_id:
            issues.append("KiCad export requires library identifier")

        # Check for KiCad-specific attributes
        if not fp.attr:
            issues.append("KiCad export recommends component attributes")

    elif format_type.lower() == 'altium':
        # Altium-specific validations
        if len(fp.pads) > 1000:
            issues.append("Altium has limits on pad count per component")

    elif format_type.lower() == 'eagle':
        # Eagle-specific validations
        if len(fp.lib_id) > 31:
            issues.append("Eagle library names limited to 31 characters")

    return tuple(issues)