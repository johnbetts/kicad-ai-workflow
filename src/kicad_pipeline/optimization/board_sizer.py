"""Iterative board size optimizer.

Replaces the fixed area multiplier with binary search for optimal
board dimensions that pass placement and maintain quality score >= threshold.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.scoring import QualityScore

log = logging.getLogger(__name__)

# Approximate footprint areas in mm^2 keyed by package pattern fragments.
_PACKAGE_AREAS: dict[str, float] = {
    "0402": 1.5,
    "0603": 3.0,
    "0805": 5.0,
    "1206": 6.0,
    "SOT-23": 9.0,
    "SOT-223": 20.0,
    "SOD-123": 6.0,
    "SOIC-8": 20.0,
    "SOIC-16": 40.0,
    "SSOP": 30.0,
    "MSOP": 15.0,
    "TSSOP": 25.0,
    "QFP": 144.0,
    "TQFP": 144.0,
    "QFN": 25.0,
    "DFN": 9.0,
    "BGA": 100.0,
    "DIP-8": 50.0,
    "DIP-14": 70.0,
    "DIP-16": 80.0,
    "DIP": 80.0,
    "TO-220": 50.0,
    "TO-92": 15.0,
    "ESP32": 200.0,
    "WROOM": 200.0,
    "WS2812": 25.0,
    "USB": 40.0,
    "Barrel_Jack": 80.0,
    "MountingHole": 30.0,
    "SW_Push": 36.0,
    "SW_DIP": 50.0,
    "Relay": 200.0,
    "Crystal": 12.0,
}

# Default area for unrecognized footprints.
_DEFAULT_AREA_MM2: float = 25.0

# Default board aspect ratio (width:height).
_DEFAULT_ASPECT_RATIO: float = 4.0 / 3.0

# Auto-size aspect ratio (3:2 per spec).
_AUTO_ASPECT_RATIO: float = 3.0 / 2.0

# Minimum area multiplier applied to total courtyard area for auto-sizing.
_MIN_AREA_MULTIPLIER: float = 2.0

# Courtyard areas in mm^2 by component category (pattern fragments, case-insensitive).
# Used only for minimum-area enforcement, not for auto-size sweeping.
_COURTYARD_AREAS: tuple[tuple[str, float], ...] = (
    # Relays — largest category, check first
    ("Relay", 400.0),
    # THT connectors
    ("TerminalBlock", 150.0),
    ("PinHeader", 150.0),
    ("RJ45", 150.0),
    ("USB", 150.0),
    # Mounting holes
    ("MountingHole", 50.0),
    # SMD ICs
    ("SOIC", 100.0),
    ("MSOP", 100.0),
    ("TSSOP", 100.0),
    ("QFP", 100.0),
    ("TQFP", 100.0),
    ("QFN", 100.0),
    ("ESP32", 100.0),
    ("WROOM", 100.0),
    # SMD passives and small discretes
    ("R_", 4.0),
    ("C_", 4.0),
    ("L_", 4.0),
    ("LED_", 4.0),
    ("SOD-", 4.0),
    ("SOT-", 4.0),
)

# Fallback courtyard area for unrecognised footprints (mm^2).
_DEFAULT_COURTYARD_AREA_MM2: float = 25.0


def _courtyard_area_for_footprint(footprint: str) -> float:
    """Return estimated courtyard area in mm^2 for a single footprint string."""
    fp_upper = footprint.upper()
    for pattern, area in _COURTYARD_AREAS:
        if pattern.upper() in fp_upper:
            return area
    return _DEFAULT_COURTYARD_AREA_MM2


def _compute_minimum_board_area(requirements: ProjectRequirements) -> float:
    """Compute the minimum board area from total courtyard footprint areas × 2.

    This gives the hard lower bound beneath which components cannot physically
    fit on the board.

    Args:
        requirements: Project requirements with component list.

    Returns:
        Minimum board area in mm^2.
    """
    total_courtyard = sum(
        _courtyard_area_for_footprint(comp.footprint)
        for comp in requirements.components
    )
    if total_courtyard <= 0:
        total_courtyard = _DEFAULT_COURTYARD_AREA_MM2
    return total_courtyard * _MIN_AREA_MULTIPLIER


def _compute_component_area(requirements: ProjectRequirements) -> float:
    """Estimate total component footprint area from requirements.

    Looks up each component's footprint string against known package
    patterns and sums the estimated areas.

    Args:
        requirements: Project requirements with component list.

    Returns:
        Estimated total component area in mm^2.
    """
    total = 0.0
    for comp in requirements.components:
        fp_upper = comp.footprint.upper()
        matched = False
        for pattern, area in _PACKAGE_AREAS.items():
            if pattern.upper() in fp_upper:
                total += area
                matched = True
                break
        if not matched:
            total += _DEFAULT_AREA_MM2
    return total


def _dimensions_from_area(
    area_mm2: float,
    aspect_ratio: float = _DEFAULT_ASPECT_RATIO,
) -> tuple[float, float]:
    """Compute board width and height from area and aspect ratio.

    Args:
        area_mm2: Target board area in mm^2.
        aspect_ratio: Width / height ratio.

    Returns:
        Tuple of ``(width_mm, height_mm)`` rounded to nearest mm.
    """
    height = math.sqrt(area_mm2 / aspect_ratio)
    width = height * aspect_ratio
    return (round(width), round(height))


def optimize_board_size(
    requirements: ProjectRequirements,
    board_template: str | None = None,
    min_area_multiplier: float = 2.0,
    max_area_multiplier: float = 4.0,
    step_mm: float = 5.0,
    quality_threshold: float = 0.7,
) -> tuple[float, float, QualityScore]:
    """Find optimal board dimensions via iterative shrinking.

    Algorithm:
    1. Estimate component area from requirements.
    2. Start at ``max_area_multiplier`` and sweep down by ``step_mm``.
    3. At each size, build a PCB and compute its quality score.
    4. Keep the smallest board that achieves ``quality_threshold``.
    5. Return ``(width_mm, height_mm, best_quality_score)``.

    If a board template is specified (e.g. ``"rpi_hat"``), the template's
    fixed dimensions are used and no size optimization is performed.

    Args:
        requirements: Project requirements.
        board_template: Optional board template name (fixed-size boards
            skip the sweep).
        min_area_multiplier: Minimum component-area multiplier to try.
        max_area_multiplier: Starting (largest) multiplier.
        step_mm: Decrement step for the width dimension.
        quality_threshold: Minimum acceptable quality score.

    Returns:
        Tuple of ``(width_mm, height_mm, quality_score)``.
    """
    from kicad_pipeline.optimization.scoring import compute_quality_score
    from kicad_pipeline.pcb.builder import build_pcb

    # If the template has fixed dimensions, honour them directly.
    if requirements.mechanical is not None:
        mech = requirements.mechanical
        if mech.board_width_mm > 0 and mech.board_height_mm > 0:
            user_area = mech.board_width_mm * mech.board_height_mm
            min_area_needed = _compute_minimum_board_area(requirements)
            if user_area < min_area_needed:
                min_w, min_h = _dimensions_from_area(min_area_needed, _AUTO_ASPECT_RATIO)
                log.warning(
                    "Board dimensions %.0fx%.0fmm (area=%.0fmm²) are smaller than the "
                    "computed minimum for %d components (min_area=%.0fmm², suggested "
                    "%.0fx%.0fmm). Components may fall off-board.",
                    mech.board_width_mm,
                    mech.board_height_mm,
                    user_area,
                    len(requirements.components),
                    min_area_needed,
                    min_w,
                    min_h,
                )
            pcb = build_pcb(
                requirements,
                board_width_mm=mech.board_width_mm,
                board_height_mm=mech.board_height_mm,
                board_template=board_template,
                auto_route=False,
            )
            score = compute_quality_score(pcb, requirements)
            return (mech.board_width_mm, mech.board_height_mm, score)

    comp_area = _compute_component_area(requirements)
    if comp_area <= 0:
        comp_area = _DEFAULT_AREA_MM2

    # For auto-sizing, use the minimum-area floor (courtyard × 2) as a lower bound,
    # then sweep from the max multiplier downward.
    min_area_floor = _compute_minimum_board_area(requirements)
    max_area = comp_area * max_area_multiplier
    min_area = max(comp_area * min_area_multiplier, min_area_floor)

    # Auto-size uses 3:2 aspect ratio per spec.
    best_w, best_h = _dimensions_from_area(max_area, _AUTO_ASPECT_RATIO)

    pcb = build_pcb(
        requirements,
        board_width_mm=best_w,
        board_height_mm=best_h,
        board_template=board_template,
        auto_route=False,
    )
    best_score: QualityScore = compute_quality_score(pcb, requirements)

    best_w, best_h, best_score = _sweep_board_sizes(
        requirements, board_template, best_w, best_h, best_score,
        min_area, step_mm, quality_threshold,
    )

    return (best_w, best_h, best_score)


def _sweep_board_sizes(
    requirements: ProjectRequirements,
    board_template: str | None,
    start_w: float,
    start_h: float,
    baseline_score: QualityScore,
    min_area: float,
    step_mm: float,
    quality_threshold: float,
) -> tuple[float, float, QualityScore]:
    """Sweep board width downward until quality drops below threshold."""
    from kicad_pipeline.optimization.scoring import compute_quality_score
    from kicad_pipeline.pcb.builder import build_pcb

    best_w = start_w
    best_h = start_h
    best_score = baseline_score
    current_w = start_w

    while current_w - step_mm >= 1.0:
        current_w -= step_mm
        current_h = round(current_w / _DEFAULT_ASPECT_RATIO)
        if current_h < 1.0:
            break
        if current_w * current_h < min_area:
            break

        pcb = build_pcb(
            requirements,
            board_width_mm=current_w,
            board_height_mm=current_h,
            board_template=board_template,
            auto_route=False,
        )
        score = compute_quality_score(pcb, requirements)

        if score.overall_score >= quality_threshold:
            best_w = current_w
            best_h = current_h
            best_score = score

    return best_w, best_h, best_score
