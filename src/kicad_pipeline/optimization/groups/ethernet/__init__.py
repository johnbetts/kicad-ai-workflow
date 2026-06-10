"""Ethernet group placement helpers.

Contains ethernet-specific placement phases that organize components
within the ethernet functional group.

Extracted from ``ee_phases_groups.py`` to reduce module size.
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