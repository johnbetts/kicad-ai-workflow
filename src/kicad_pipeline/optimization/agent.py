"""Background optimizer agent.

Monitors project directories for PCB stage completion, runs optimization
passes, and writes suggestions.  Never overwrites the user's PCB directly.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.scoring import QualityScore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationProgress:
    """Progress of an optimization run."""

    status: str  # "idle", "running", "completed", "failed"
    iterations_completed: int
    best_score: float
    initial_score: float
    improvement_pct: float
    history: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class OptimizationSuggestion:
    """A single optimization suggestion."""

    category: str  # "placement", "zone", "thermal", "signal", "electrical", "manufacturing"
    priority: str  # "critical", "high", "medium", "low"
    title: str
    description: str


@dataclass(frozen=True)
class OptimizationResult:
    """Complete result of an optimization run."""

    quality_grade: str
    initial_score: float
    best_score: float
    suggestions: tuple[OptimizationSuggestion, ...]
    best_positions: tuple[tuple[str, float, float, float], ...] | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_placement_pipeline(
    req: ProjectRequirements,
) -> tuple[PCBDesign, QualityScore, PCBDesign, QualityScore, object]:
    """Steps 1-3: build PCB, score it, run EE placement optimization.

    Uses lazy imports to avoid circular dependencies at module load time.
    """
    from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee
    from kicad_pipeline.optimization.scoring import compute_fast_placement_score
    from kicad_pipeline.pcb.board_templates import detect_template
    from kicad_pipeline.pcb.builder import build_pcb

    tmpl = detect_template(req.mechanical)
    board_template = tmpl.name if tmpl is not None else None
    pcb = build_pcb(req, board_template=board_template, auto_route=False)
    initial_quality = compute_fast_placement_score(pcb, req)
    best_pcb, ee_review = optimize_placement_ee(req, pcb, max_review_passes=5)
    best_quality = compute_fast_placement_score(best_pcb, req)

    return pcb, initial_quality, best_pcb, best_quality, ee_review


def _collect_ee_suggestions(violations: object) -> tuple[OptimizationSuggestion, ...]:
    """Convert EE review violations to OptimizationSuggestion objects.

    Minor violations are skipped; all others become high or critical priority.
    """
    from typing import Any
    result: list[OptimizationSuggestion] = []
    for violation in violations:  # type: ignore[union-attr]
        v: Any = violation
        if v.severity == "minor":
            continue
        result.append(OptimizationSuggestion(
            category="placement",
            priority="critical" if v.severity == "critical" else "high",
            title=f"{v.rule.value}: {', '.join(v.refs)}",
            description=v.message,
        ))
    return tuple(result)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def _score_dimension_suggestions(best: QualityScore) -> list[OptimizationSuggestion]:
    """Return suggestions for any score dimension that falls below threshold."""
    result: list[OptimizationSuggestion] = []
    checks = [
        (best.electrical_score, 0.8, "electrical", "high",
         "Electrical issues detected", "review net connectivity and power rails"),
        (best.manufacturing_score, 0.8, "manufacturing", "high",
         "Manufacturing constraints violated", "check JLCPCB trace/via limits"),
        (best.thermal_score, 0.8, "thermal", "medium",
         "Thermal concerns", "review high-power component placement"),
        (best.signal_integrity_score, 0.8, "signal", "medium",
         "Signal integrity issues", "check differential pairs and analog routing"),
        (best.placement_score, 0.7, "placement", "high",
         "Placement quality low", "decoupling caps may be too far from ICs"),
    ]
    for score_val, threshold, category, priority, title, hint in checks:
        if score_val < threshold:
            result.append(OptimizationSuggestion(
                category=category,
                priority=priority,
                title=title,
                description=f"{title.split()[0]} score {score_val:.2f}/1.0 — {hint}",
            ))
    return result


def _append_zone_suggestions(
    suggestions: list[OptimizationSuggestion],
    zone_strategy: object,
) -> None:
    """Append zone-strategy suggestions to *suggestions* in place."""
    from kicad_pipeline.optimization.zone_optimizer import ZoneStrategy

    if isinstance(zone_strategy, ZoneStrategy):
        for reason in zone_strategy.rationale:
            suggestions.append(OptimizationSuggestion(
                category="zone",
                priority="medium",
                title=f"Zone strategy: {zone_strategy.gnd_strategy}",
                description=reason,
            ))


class OptimizerAgent:
    """Background agent that runs optimization passes on PCB designs.

    Non-destructive: writes results to ``{variant}/optimization/`` directory
    but never overwrites the user's PCB file.
    """

    def __init__(self, project_root: Path) -> None:
        self._root = project_root
        self._progress: OptimizationProgress | None = None

    def optimize_variant(
        self,
        variant_name: str,
        max_iterations: int = 50,
    ) -> OptimizationResult:
        """Run full optimization pipeline on a variant.

        Steps:
            1. Load requirements and build PCB.
            2. Compute initial quality score.
            3. Run placement optimization (simulated annealing).
            4. Run zone strategy analysis.
            5. Generate suggestions.
            6. Write results to ``variant/optimization/``.
            7. Return result (never overwrite PCB).
        """
        from kicad_pipeline.optimization.placement_optimizer import _extract_positions
        from kicad_pipeline.optimization.zone_optimizer import recommend_zone_strategy
        from kicad_pipeline.requirements.decomposer import load_requirements

        vdir = self._root / "variants" / variant_name
        req = load_requirements(vdir / "requirements.json")

        pcb, initial_quality, best_pcb, best_quality, ee_review = _run_placement_pipeline(req)
        initial_score = initial_quality.overall_score
        best_score = best_quality.overall_score

        zone_strategy = recommend_zone_strategy(best_pcb, req)
        suggestions = self._generate_suggestions(
            initial_quality, best_quality, zone_strategy, pcb, req,
        )
        suggestions = suggestions + _collect_ee_suggestions(ee_review.violations)

        improvement = (
            ((best_score - initial_score) / initial_score * 100) if initial_score > 0 else 0.0
        )
        best_positions = _extract_positions(best_pcb) if best_score > initial_score else None

        result = OptimizationResult(
            quality_grade=best_quality.grade,
            initial_score=round(initial_score, 4),
            best_score=round(best_score, 4),
            suggestions=suggestions,
            best_positions=best_positions,
        )

        self._write_optimization_results(vdir, result, initial_score, best_score, improvement)

        log.info(
            "Optimization complete: %.2f -> %.2f (%+.1f%%)",
            initial_score,
            best_score,
            improvement,
        )

        return result


    def _write_optimization_results(
        self,
        vdir: Path,
        result: OptimizationResult,
        initial_score: float,
        best_score: float,
        improvement: float,
    ) -> None:
        """Step 6: Write progress and suggestions JSON to variant/optimization/."""
        opt_dir = vdir / "optimization"
        opt_dir.mkdir(parents=True, exist_ok=True)

        progress = OptimizationProgress(
            status="completed",
            iterations_completed=1,  # EE optimizer is single-pass with review loop
            best_score=round(best_score, 4),
            initial_score=round(initial_score, 4),
            improvement_pct=round(improvement, 1),
            history=(
                {"iteration": 0, "score": round(initial_score, 4)},
                {"iteration": 1, "score": round(best_score, 4)},
            ),
        )
        self._atomic_write_json(opt_dir / "progress.json", asdict(progress))
        self._atomic_write_json(
            opt_dir / "suggestions.json",
            {
                "quality_grade": result.quality_grade,
                "initial_score": result.initial_score,
                "best_score": result.best_score,
                "suggestions": [asdict(s) for s in result.suggestions],
            },
        )

    def _generate_suggestions(
        self,
        initial: QualityScore,
        best: QualityScore,
        zone_strategy: object,
        pcb: PCBDesign,
        requirements: ProjectRequirements,
    ) -> tuple[OptimizationSuggestion, ...]:
        """Generate human-readable optimization suggestions."""
        suggestions = list(_score_dimension_suggestions(best))
        _append_zone_suggestions(suggestions, zone_strategy)
        return tuple(suggestions)

    @staticmethod
    def _atomic_write_json(path: Path, data: dict[str, object]) -> None:
        """Write JSON atomically via tempfile + rename."""
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, str(path))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
