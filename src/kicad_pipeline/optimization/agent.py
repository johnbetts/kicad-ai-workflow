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
# Agent
# ---------------------------------------------------------------------------


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
        from kicad_pipeline.optimization.placement_optimizer import (
            _extract_positions,
            optimize_placement_ee,
        )
        from kicad_pipeline.optimization.scoring import compute_fast_placement_score
        from kicad_pipeline.optimization.zone_optimizer import recommend_zone_strategy
        from kicad_pipeline.pcb.board_templates import detect_template
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.requirements.decomposer import load_requirements

        vdir = self._root / "variants" / variant_name
        req = load_requirements(vdir / "requirements.json")

        tmpl = detect_template(req.mechanical)
        board_template = tmpl.name if tmpl is not None else None
        pcb = build_pcb(req, board_template=board_template, auto_route=False)

        # Initial score — use fast-path scoring for placement optimization
        initial_quality = compute_fast_placement_score(pcb, req)
        initial_score = initial_quality.overall_score

        # Placement optimization — use EE-grade deterministic placement
        best_pcb, ee_review = optimize_placement_ee(
            req, pcb, max_review_passes=5,
        )
        best_quality = compute_fast_placement_score(best_pcb, req)
        best_score = best_quality.overall_score

        # Zone analysis
        zone_strategy = recommend_zone_strategy(best_pcb, req)

        # Generate suggestions — combine agent suggestions + EE review violations
        suggestions = self._generate_suggestions(
            initial_quality,
            best_quality,
            zone_strategy,
            pcb,
            req,
        )

        # Add EE review violations as suggestions
        ee_suggestions: list[OptimizationSuggestion] = []
        for violation in ee_review.violations:
            if violation.severity == "minor":
                continue
            ee_suggestions.append(
                OptimizationSuggestion(
                    category="placement",
                    priority="critical" if violation.severity == "critical" else "high",
                    title=f"{violation.rule.value}: {', '.join(violation.refs)}",
                    description=violation.message,
                )
            )
        suggestions = suggestions + tuple(ee_suggestions)

        improvement = (
            ((best_score - initial_score) / initial_score * 100)
            if initial_score > 0
            else 0.0
        )

        # Build result
        best_positions = _extract_positions(best_pcb) if best_score > initial_score else None

        result = OptimizationResult(
            quality_grade=best_quality.grade,
            initial_score=round(initial_score, 4),
            best_score=round(best_score, 4),
            suggestions=suggestions,
            best_positions=best_positions,
        )

        # Write results atomically
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

        self._atomic_write_json(
            opt_dir / "progress.json",
            asdict(progress),
        )
        self._atomic_write_json(
            opt_dir / "suggestions.json",
            {
                "quality_grade": result.quality_grade,
                "initial_score": result.initial_score,
                "best_score": result.best_score,
                "suggestions": [asdict(s) for s in result.suggestions],
            },
        )

        log.info(
            "Optimization complete: %.2f -> %.2f (%+.1f%%)",
            initial_score,
            best_score,
            improvement,
        )

        return result

    def _generate_suggestions(
        self,
        initial: QualityScore,
        best: QualityScore,
        zone_strategy: object,
        pcb: PCBDesign,
        requirements: ProjectRequirements,
    ) -> tuple[OptimizationSuggestion, ...]:
        """Generate human-readable optimization suggestions."""
        suggestions: list[OptimizationSuggestion] = []

        # Check each score dimension
        if best.electrical_score < 0.8:
            suggestions.append(
                OptimizationSuggestion(
                    category="electrical",
                    priority="high",
                    title="Electrical issues detected",
                    description=(
                        f"Electrical score {best.electrical_score:.2f}/1.0"
                        " — review net connectivity and power rails"
                    ),
                )
            )

        if best.manufacturing_score < 0.8:
            suggestions.append(
                OptimizationSuggestion(
                    category="manufacturing",
                    priority="high",
                    title="Manufacturing constraints violated",
                    description=(
                        f"Manufacturing score {best.manufacturing_score:.2f}/1.0"
                        " — check JLCPCB trace/via limits"
                    ),
                )
            )

        if best.thermal_score < 0.8:
            suggestions.append(
                OptimizationSuggestion(
                    category="thermal",
                    priority="medium",
                    title="Thermal concerns",
                    description=(
                        f"Thermal score {best.thermal_score:.2f}/1.0"
                        " — review high-power component placement"
                    ),
                )
            )

        if best.signal_integrity_score < 0.8:
            suggestions.append(
                OptimizationSuggestion(
                    category="signal",
                    priority="medium",
                    title="Signal integrity issues",
                    description=(
                        f"SI score {best.signal_integrity_score:.2f}/1.0"
                        " — check differential pairs and analog routing"
                    ),
                )
            )

        if best.placement_score < 0.7:
            suggestions.append(
                OptimizationSuggestion(
                    category="placement",
                    priority="high",
                    title="Placement quality low",
                    description=(
                        f"Placement score {best.placement_score:.2f}/1.0"
                        " — decoupling caps may be too far from ICs"
                    ),
                )
            )

        # Zone strategy suggestions
        from kicad_pipeline.optimization.zone_optimizer import ZoneStrategy

        if isinstance(zone_strategy, ZoneStrategy):
            for reason in zone_strategy.rationale:
                suggestions.append(
                    OptimizationSuggestion(
                        category="zone",
                        priority="medium",
                        title=f"Zone strategy: {zone_strategy.gnd_strategy}",
                        description=reason,
                    )
                )

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
