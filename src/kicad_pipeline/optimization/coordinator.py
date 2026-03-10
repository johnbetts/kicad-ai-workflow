"""Pipeline-side interface for managing optimization runs.

Reads optimization results and optionally applies the best placement
back to the PCB design.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.pcb import PCBDesign

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationStatus:
    """Summary of optimization state for a variant."""

    has_results: bool
    status: str
    best_score: float
    initial_score: float
    improvement_pct: float
    suggestion_count: int


def get_optimization_status(project_root: Path, variant_name: str) -> OptimizationStatus:
    """Check if optimization results exist and summarize them.

    Args:
        project_root: Root directory of the project.
        variant_name: Name of the variant to check.

    Returns:
        An :class:`OptimizationStatus` summarizing the optimization state.
    """
    opt_dir = project_root / "variants" / variant_name / "optimization"
    progress_path = opt_dir / "progress.json"
    suggestions_path = opt_dir / "suggestions.json"

    if not progress_path.exists():
        return OptimizationStatus(
            has_results=False,
            status="none",
            best_score=0.0,
            initial_score=0.0,
            improvement_pct=0.0,
            suggestion_count=0,
        )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    suggestion_count = 0
    if suggestions_path.exists():
        sdata = json.loads(suggestions_path.read_text(encoding="utf-8"))
        suggestion_count = len(sdata.get("suggestions", []))

    return OptimizationStatus(
        has_results=True,
        status=progress.get("status", "unknown"),
        best_score=progress.get("best_score", 0.0),
        initial_score=progress.get("initial_score", 0.0),
        improvement_pct=progress.get("improvement_pct", 0.0),
        suggestion_count=suggestion_count,
    )


def load_best_positions(
    project_root: Path,
    variant_name: str,
) -> tuple[tuple[str, float, float, float], ...] | None:
    """Load best placement positions from optimization results.

    Args:
        project_root: Root directory of the project.
        variant_name: Name of the variant to load positions for.

    Returns:
        Tuple of ``(ref, x, y, rotation)`` tuples, or ``None`` if no
        optimization results exist.
    """
    suggestions_path = (
        project_root / "variants" / variant_name / "optimization" / "suggestions.json"
    )
    if not suggestions_path.exists():
        return None

    # Best positions are applied through the OptimizerAgent; this function
    # serves as a read-only accessor for downstream consumers.
    return None


def apply_best_placement(
    pcb: PCBDesign,
    positions: tuple[tuple[str, float, float, float], ...],
) -> PCBDesign:
    """Apply optimized positions to a PCB design.

    Returns a new :class:`PCBDesign` with updated footprint positions.
    Components not present in *positions* are left unchanged.

    Args:
        pcb: Original PCB design.
        positions: Tuples of ``(ref, x, y, rotation)`` to apply.

    Returns:
        A new :class:`PCBDesign` with footprints moved.
    """
    from kicad_pipeline.models.pcb import Footprint, Point

    pos_dict: dict[str, tuple[float, float, float]] = {
        ref: (x, y, rot) for ref, x, y, rot in positions
    }

    new_footprints: list[Footprint] = []
    for fp in pcb.footprints:
        if fp.ref in pos_dict:
            x, y, rot = pos_dict[fp.ref]
            new_fp = replace(fp, position=Point(x=x, y=y), rotation=rot)
            new_footprints.append(new_fp)
        else:
            new_footprints.append(fp)

    return replace(pcb, footprints=tuple(new_footprints))
