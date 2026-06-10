"""Footprint validation and specification checking."""

from .footprints import (
    validate_footprint_integrity,
    validate_pin_assignments,
    validate_component_spec,
)

__all__ = [
    "validate_footprint_integrity",
    "validate_pin_assignments",
    "validate_component_spec",
]
