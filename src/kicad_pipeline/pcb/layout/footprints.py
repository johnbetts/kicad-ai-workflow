"""Layout and analysis utilities for footprints.

Provides functions for computing footprint bounding boxes, courtyard estimation,
origin type detection, and 3D model validation.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint

from kicad_pipeline.models.pcb import FootprintBBox, OriginType

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Footprint layout analysis
# ---------------------------------------------------------------------------


def compute_footprint_bbox(fp: Footprint) -> FootprintBBox:
    """Compute the bounding box of a footprint from its pads and graphics.

    The bounding box includes all pads, graphics elements, and text.
    Used for collision detection and layout optimization.

    Args:
        fp: Footprint to analyze

    Returns:
        FootprintBBox with min/max coordinates and dimensions
    """
    if not fp.pads and not fp.graphics and not fp.texts:
        return FootprintBBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0,
                           width=0.0, height=0.0)

    coords = []

    # Collect pad coordinates
    for pad in fp.pads:
        x, y = pad.position.x, pad.position.y
        w, h = pad.size
        coords.extend([
            (x - w/2, y - h/2), (x + w/2, y - h/2),
            (x - w/2, y + h/2), (x + w/2, y + h/2)
        ])

    # Collect graphics coordinates
    for graphic in fp.graphics:
        if hasattr(graphic, 'start') and hasattr(graphic, 'end'):
            coords.extend([
                (graphic.start.x, graphic.start.y),
                (graphic.end.x, graphic.end.y)
            ])
        elif hasattr(graphic, 'center') and hasattr(graphic, 'radius'):
            # Circle/arc approximation
            c = graphic.center
            r = graphic.radius
            coords.extend([
                (c.x - r, c.y - r), (c.x + r, c.y - r),
                (c.x - r, c.y + r), (c.x + r, c.y + r)
            ])

    # Collect text coordinates (approximate)
    for text in fp.texts:
        x, y = text.position.x, text.position.y
        # Approximate text size
        size = getattr(text, 'size', 1.0)
        coords.extend([
            (x - size, y - size), (x + size, y - size),
            (x - size, y + size), (x + size, y + size)
        ])

    if not coords:
        return FootprintBBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0,
                           width=0.0, height=0.0)

    xs, ys = zip(*coords)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return FootprintBBox(
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        width=x_max - x_min, height=y_max - y_min
    )


def _classify_package(fp: Footprint) -> str:
    """Classify a footprint by package type for courtyard calculation.

    Args:
        fp: Footprint to classify

    Returns:
        Package type string ('smd', 'tht', 'connector', etc.)
    """
    lib_id = fp.lib_id.upper()

    # Through-hole components
    if 'through_hole' in fp.attr or any(x in lib_id for x in ('THT', 'THROUGH', 'AXIAL')):
        return 'tht'

    # Connectors
    if any(x in lib_id for x in ('CONN', 'HEADER', 'SOCKET', 'USB', 'RJ45', 'TERMINAL')):
        return 'connector'

    # SMD components
    if any(x in lib_id for x in ('SMD', 'SOT', 'SOIC', 'QFN', 'QFP', 'LQFP', 'TQFP')):
        return 'smd'

    # LEDs
    if 'LED' in lib_id:
        return 'led'

    # Crystals/oscillators
    if any(x in lib_id for x in ('CRYSTAL', 'OSCILLATOR', 'RESONATOR')):
        return 'crystal'

    # Default to SMD
    return 'smd'


def _courtyard_from_graphics(fp: Footprint) -> tuple[float, float] | None:
    """Extract courtyard dimensions from footprint graphics.

    Looks for courtyard layer graphics to determine component boundaries.

    Args:
        fp: Footprint to analyze

    Returns:
        (width, height) in mm if courtyard found, None otherwise
    """
    from kicad_pipeline.constants import LAYER_F_COURTYARD, LAYER_B_COURTYARD

    courtyard_graphics = [
        g for g in fp.graphics
        if getattr(g, 'layer', '') in (LAYER_F_COURTYARD, LAYER_B_COURTYARD)
    ]

    if not courtyard_graphics:
        return None

    # Find bounding box of courtyard graphics
    coords = []
    for g in courtyard_graphics:
        if hasattr(g, 'start') and hasattr(g, 'end'):
            coords.extend([(g.start.x, g.start.y), (g.end.x, g.end.y)])
        elif hasattr(g, 'center') and hasattr(g, 'radius'):
            c, r = g.center, g.radius
            coords.extend([
                (c.x - r, c.y - r), (c.x + r, c.y - r),
                (c.x - r, c.y + r), (c.x + r, c.y + r)
            ])

    if not coords:
        return None

    xs, ys = zip(*coords)
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return (width, height)


def _body_from_fab_graphics(fp: Footprint) -> tuple[float, float] | None:
    """Extract body dimensions from fabrication layer graphics.

    Args:
        fp: Footprint to analyze

    Returns:
        (width, height) in mm if body outline found, None otherwise
    """
    from kicad_pipeline.constants import LAYER_F_FAB, LAYER_B_FAB

    fab_graphics = [
        g for g in fp.graphics
        if getattr(g, 'layer', '') in (LAYER_F_FAB, LAYER_B_FAB)
    ]

    if not fab_graphics:
        return None

    # Find bounding box of fab graphics
    coords = []
    for g in fab_graphics:
        if hasattr(g, 'start') and hasattr(g, 'end'):
            coords.extend([(g.start.x, g.start.y), (g.end.x, g.end.y)])
        elif hasattr(g, 'center') and hasattr(g, 'radius'):
            c, r = g.center, g.radius
            coords.extend([
                (c.x - r, c.y - r), (c.x + r, c.y - r),
                (c.x - r, c.y + r), (c.x + r, c.y + r)
            ])

    if not coords:
        return None

    xs, ys = zip(*coords)
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return (width, height)


def estimate_courtyard_mm(fp: Footprint) -> tuple[float, float]:
    """Estimate courtyard dimensions for a footprint.

    Uses a hierarchical approach:
    1. Extract from courtyard graphics if present
    2. Extract from fabrication layer body outline
    3. Use pad-based estimation with package-specific rules

    Args:
        fp: Footprint to analyze

    Returns:
        (width, height) in mm
    """
    from kicad_pipeline.constants import PCB_COURTYARD_CLEARANCE_MM

    # Try courtyard graphics first
    courtyard_dims = _courtyard_from_graphics(fp)
    if courtyard_dims:
        return courtyard_dims

    # Try fabrication layer body
    body_dims = _body_from_fab_graphics(fp)
    if body_dims:
        w, h = body_dims
        # Add courtyard clearance
        return (w + 2 * PCB_COURTYARD_CLEARANCE_MM,
                h + 2 * PCB_COURTYARD_CLEARANCE_MM)

    # Fall back to pad-based estimation
    if not fp.pads:
        return (2.0, 2.0)  # Minimum courtyard

    # Calculate pad bounding box
    xs = [p.position.x for p in fp.pads]
    ys = [p.position.y for p in fp.pads]
    pad_width = max(xs) - min(xs)
    pad_height = max(ys) - min(ys)

    # Apply package-specific courtyard rules
    pkg_type = _classify_package(fp)

    if pkg_type == 'tht':
        # Through-hole: larger clearance
        clearance = max(PCB_COURTYARD_CLEARANCE_MM, 0.5)
    elif pkg_type == 'connector':
        # Connectors: generous clearance
        clearance = max(PCB_COURTYARD_CLEARANCE_MM, 1.0)
    elif pkg_type in ('led', 'crystal'):
        # Special components: standard clearance
        clearance = PCB_COURTYARD_CLEARANCE_MM
    else:
        # SMD default
        clearance = PCB_COURTYARD_CLEARANCE_MM

    # Ensure minimum dimensions
    width = max(pad_width + 2 * clearance, 1.0)
    height = max(pad_height + 2 * clearance, 1.0)

    return (width, height)


def detect_origin_type(fp: Footprint) -> OriginType:
    """Detect the origin type of a footprint.

    Analyzes pad positions and graphics to determine if the footprint
    origin is at the pad centroid, pin 1, or geometric center.

    Args:
        fp: Footprint to analyze

    Returns:
        OriginType enum value
    """
    if not fp.pads:
        return OriginType.CENTER

    # Calculate pad centroid
    xs = [p.position.x for p in fp.pads]
    ys = [p.position.y for p in fp.pads]
    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)

    # Check if origin (0,0) is at centroid
    centroid_distance = math.sqrt(centroid_x**2 + centroid_y**2)
    if centroid_distance < 0.1:  # Within 0.1mm
        return OriginType.CENTROID

    # Check if origin is at pin 1
    pin1 = next((p for p in fp.pads if p.number == "1"), None)
    if pin1:
        pin1_distance = math.sqrt(pin1.position.x**2 + pin1.position.y**2)
        if pin1_distance < 0.1:  # Within 0.1mm
            return OriginType.PIN1

    # Default to center (geometric center of bounding box)
    return OriginType.CENTER


def validate_3d_model_orientation(fp: Footprint) -> tuple[str, ...]:
    """Validate 3D model orientation and positioning.

    Checks for common 3D model issues:
    - Missing models
    - Incorrect rotations
    - Misaligned offsets

    Args:
        fp: Footprint with 3D model to validate

    Returns:
        Tuple of error/warning messages
    """
    issues = []

    if not fp.models:
        issues.append("Missing 3D model")
        return tuple(issues)

    model = fp.models[0]  # Primary model

    # Check for extreme rotations (likely errors)
    rx, ry, rz = model.rotate
    if abs(rx) > 90 or abs(ry) > 90:
        issues.append(f"Suspicious rotation: ({rx:.0f}, {ry:.0f}, {rz:.0f})")

    # Check for large offsets (might indicate misalignment)
    ox, oy, oz = model.offset
    if abs(ox) > 10 or abs(oy) > 10:  # More than 10mm offset
        issues.append(f"Large offset: ({ox:.2f}, {oy:.2f}, {oz:.2f})")

    # Check model path exists (if we can validate)
    if model.path.startswith('${') or model.path.startswith('/'):
        # Can't validate variable or absolute paths easily
        pass
    else:
        # Relative path - could check if it follows KiCad conventions
        if not any(pattern in model.path for pattern in ['.3dshapes', '.step', '.wrl']):
            issues.append(f"Unusual model path format: {model.path}")

    return tuple(issues)