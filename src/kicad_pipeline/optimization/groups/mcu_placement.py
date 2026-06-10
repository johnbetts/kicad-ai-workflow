"""MCU placement optimization functions.

Contains specialized placement logic for microcontroller components,
including power decoupling, connectors, LEDs, and peripheral placement.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from kicad_pipeline.optimization.collision_resolver import (
    _PlacementGrid,
    _rotation_aware_size,
)

if TYPE_CHECKING:
    from kicad_pipeline.optimization.placement_types import PlacementContext
from kicad_pipeline.pcb.pin_map import (
    origin_to_centroid,
)

_log = logging.getLogger(__name__)

# Default fallback footprint sizes (w, h) in mm for named components
_DEFAULT_J2_SIZE_MM: tuple[float, float] = (9.6, 7.6)
"""Default USB-C connector (J2) footprint size."""

_DEFAULT_J14_SIZE_MM: tuple[float, float] = (2.7, 35.7)
"""Default pin-header connector (J14) footprint size."""

_BOARD_EDGE_MARGIN_MM: float = 2.0
"""Margin from board edge for component placement within groups."""