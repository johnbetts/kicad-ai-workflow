"""Compare current placement against a reference board.

Provides per-group and overall similarity scoring between the current
optimizer output and a human-routed reference board.  Used by the
reference-driven placement mode to measure drift from known-good positions.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.requirements import ProjectRequirements

_log = logging.getLogger(__name__)

# Positions beyond this distance (mm) from reference are considered "failed"
_MAX_ERROR_MM = 20.0


def load_reference_positions(
    pcb_path: str | Path,
) -> dict[str, tuple[float, float, float]]:
    """Load reference positions from a .kicad_pcb file.

    Returns ref -> (x_mm, y_mm, rotation_deg) in KiCad origin space.
    """
    from kicad_pipeline.pcb.position_extractor import positions_from_pcb_file

    return positions_from_pcb_file(pcb_path)


def compare_to_reference(
    current: dict[str, tuple[float, float, float]],
    reference: dict[str, tuple[float, float, float]],
    group_map: dict[str, str],
) -> dict[str, float]:
    """Compute normalized position similarity per group.

    For each group: mean Euclidean distance between current and reference
    positions for components that exist in both.

    Returns:
        Dict mapping group names to similarity percentage (0-100).
        Special key ``"overall"`` contains the weighted average.
        Special key ``"_errors"`` is absent (errors logged instead).
    """
    # Accumulate per-group errors
    group_errors: dict[str, list[float]] = {}
    matched = 0
    unmatched: list[str] = []

    for ref, (cx, cy, _crot) in current.items():
        if ref not in reference:
            unmatched.append(ref)
            continue
        rx, ry, _rrot = reference[ref]
        dist = math.hypot(cx - rx, cy - ry)
        group = group_map.get(ref, "_ungrouped")
        group_errors.setdefault(group, []).append(dist)
        matched += 1

    if unmatched:
        _log.debug("  %d refs in current but not reference: %s",
                    len(unmatched), unmatched[:10])

    # Compute similarity per group
    result: dict[str, float] = {}
    total_error = 0.0
    total_count = 0

    for group, errors in sorted(group_errors.items()):
        mean_error = sum(errors) / len(errors)
        similarity = max(0.0, 100.0 - (mean_error / _MAX_ERROR_MM) * 100.0)
        result[group] = similarity
        total_error += sum(errors)
        total_count += len(errors)
        _log.info("  Group %-25s: %5.1f%% similarity (mean %.1fmm, %d refs)",
                   group, similarity, mean_error, len(errors))

    # Overall weighted by component count
    if total_count > 0:
        overall_mean = total_error / total_count
        result["overall"] = max(0.0, 100.0 - (overall_mean / _MAX_ERROR_MM) * 100.0)
    else:
        result["overall"] = 0.0

    _log.info("  Overall: %.1f%% similarity (%d matched, %d unmatched)",
              result["overall"], matched, len(unmatched))

    return result


def worst_group(similarity: dict[str, float]) -> str | None:
    """Return the group name with lowest similarity (excluding 'overall')."""
    candidates = {k: v for k, v in similarity.items()
                  if k not in ("overall", "_ungrouped")}
    if not candidates:
        return None
    return min(candidates, key=candidates.get)  # type: ignore[arg-type]


def build_group_map_from_requirements(
    requirements: ProjectRequirements,
) -> dict[str, str]:
    """Build ref -> group_name mapping from FeatureBlocks.

    Thin wrapper around ``_build_group_map`` for public use.
    """
    from kicad_pipeline.optimization.group_helpers import _build_group_map

    return _build_group_map(requirements)
