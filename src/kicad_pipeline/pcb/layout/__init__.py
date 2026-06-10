"""Layout analysis and bounding box calculations."""

from .footprints import (
    compute_footprint_bbox,
    _classify_package,
    _courtyard_from_graphics,
    _body_from_fab_graphics,
    estimate_courtyard_mm,
    detect_origin_type,
    validate_3d_model_orientation,
)

__all__ = [
    "compute_footprint_bbox",
    "_classify_package",
    "_courtyard_from_graphics",
    "_body_from_fab_graphics",
    "estimate_courtyard_mm",
    "detect_origin_type",
    "validate_3d_model_orientation",
]
