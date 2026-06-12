"""Shared utilities for symbol generation."""

from kicad_pipeline.constants import SCHEMATIC_TEXT_SIZE_MM
from kicad_pipeline.models.schematic import FontEffect

__all__ = ["_make_font"]


def _make_font(hidden: bool = False) -> FontEffect:
    """Return a :class:`FontEffect` with standard text size."""
    if hidden:
        return FontEffect(
            size_x=SCHEMATIC_TEXT_SIZE_MM,
            size_y=SCHEMATIC_TEXT_SIZE_MM,
            hidden=True,
        )
    return FontEffect(
        size_x=SCHEMATIC_TEXT_SIZE_MM,
        size_y=SCHEMATIC_TEXT_SIZE_MM,
    )
