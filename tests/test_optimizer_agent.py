"""Tests for the background optimizer agent and coordinator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.agents.models import CommandType
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.optimization.agent import (
    OptimizationProgress,
    OptimizationResult,
    OptimizationSuggestion,
    OptimizerAgent,
)
from kicad_pipeline.optimization.coordinator import (
    OptimizationStatus,
    apply_best_placement,
    get_optimization_status,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_pcb(*footprints: Footprint) -> PCBDesign:
    """Build a minimal PCBDesign with the given footprints."""
    return PCBDesign(
        outline=BoardOutline(
            polygon=(
                Point(x=0.0, y=0.0),
                Point(x=50.0, y=0.0),
                Point(x=50.0, y=50.0),
                Point(x=0.0, y=50.0),
                Point(x=0.0, y=0.0),
            ),
        ),
        design_rules=DesignRules(),
        nets=(NetEntry(number=0, name=""),),
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_footprint(ref: str, x: float, y: float, rot: float = 0.0) -> Footprint:
    """Create a minimal Footprint at the given position."""
    return Footprint(
        lib_id=f"Test:{ref}",
        ref=ref,
        value="100nF",
        position=Point(x=x, y=y),
        rotation=rot,
        pads=(
            Pad(
                number="1",
                pad_type="smd",
                shape="rect",
                position=Point(x=-0.5, y=0.0),
                size_x=0.6,
                size_y=0.5,
                layers=("F.Cu", "F.Paste", "F.Mask"),
            ),
            Pad(
                number="2",
                pad_type="smd",
                shape="rect",
                position=Point(x=0.5, y=0.0),
                size_x=0.6,
                size_y=0.5,
                layers=("F.Cu", "F.Paste", "F.Mask"),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Frozen dataclass tests
# ---------------------------------------------------------------------------


class TestDataclassesFrozen:
    """Verify all optimization dataclasses are immutable."""

    def test_optimization_progress_frozen(self) -> None:
        progress = OptimizationProgress(
            status="completed",
            iterations_completed=10,
            best_score=0.85,
            initial_score=0.65,
            improvement_pct=30.8,
            history=(),
        )
        with pytest.raises(AttributeError):
            progress.status = "running"  # type: ignore[misc]

    def test_optimization_suggestion_frozen(self) -> None:
        suggestion = OptimizationSuggestion(
            category="placement",
            priority="high",
            title="Test",
            description="Test description",
        )
        with pytest.raises(AttributeError):
            suggestion.category = "zone"  # type: ignore[misc]

    def test_optimization_result_frozen(self) -> None:
        result = OptimizationResult(
            quality_grade="B",
            initial_score=0.65,
            best_score=0.85,
            suggestions=(),
            best_positions=None,
        )
        with pytest.raises(AttributeError):
            result.quality_grade = "A"  # type: ignore[misc]

    def test_optimization_status_frozen(self) -> None:
        status = OptimizationStatus(
            has_results=True,
            status="completed",
            best_score=0.85,
            initial_score=0.65,
            improvement_pct=30.8,
            suggestion_count=3,
        )
        with pytest.raises(AttributeError):
            status.has_results = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Coordinator: get_optimization_status
# ---------------------------------------------------------------------------


class TestGetOptimizationStatus:
    """Tests for coordinator.get_optimization_status."""

    def test_get_optimization_status_no_results(self, tmp_path: Path) -> None:
        status = get_optimization_status(tmp_path, "nonexistent")
        assert not status.has_results
        assert status.status == "none"
        assert status.best_score == 0.0
        assert status.initial_score == 0.0
        assert status.improvement_pct == 0.0
        assert status.suggestion_count == 0

    def test_get_optimization_status_with_results(self, tmp_path: Path) -> None:
        opt_dir = tmp_path / "variants" / "test" / "optimization"
        opt_dir.mkdir(parents=True)
        (opt_dir / "progress.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "best_score": 0.85,
                    "initial_score": 0.65,
                    "improvement_pct": 30.8,
                }
            )
        )
        (opt_dir / "suggestions.json").write_text(
            json.dumps(
                {
                    "suggestions": [{"title": "test"}],
                }
            )
        )
        status = get_optimization_status(tmp_path, "test")
        assert status.has_results
        assert status.status == "completed"
        assert status.best_score == 0.85
        assert status.initial_score == 0.65
        assert status.improvement_pct == 30.8
        assert status.suggestion_count == 1

    def test_get_optimization_status_progress_only(self, tmp_path: Path) -> None:
        """Progress file exists but no suggestions file."""
        opt_dir = tmp_path / "variants" / "v1" / "optimization"
        opt_dir.mkdir(parents=True)
        (opt_dir / "progress.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "best_score": 0.70,
                    "initial_score": 0.70,
                    "improvement_pct": 0.0,
                }
            )
        )
        status = get_optimization_status(tmp_path, "v1")
        assert status.has_results
        assert status.status == "running"
        assert status.suggestion_count == 0


# ---------------------------------------------------------------------------
# Coordinator: apply_best_placement
# ---------------------------------------------------------------------------


class TestApplyBestPlacement:
    """Tests for coordinator.apply_best_placement."""

    def test_apply_best_placement_moves_footprints(self) -> None:
        fp1 = _make_footprint("R1", 10.0, 10.0, 0.0)
        fp2 = _make_footprint("R2", 20.0, 20.0, 0.0)
        pcb = _minimal_pcb(fp1, fp2)

        positions: tuple[tuple[str, float, float, float], ...] = (
            ("R1", 15.0, 15.0, 90.0),
            ("R2", 25.0, 25.0, 180.0),
        )
        result = apply_best_placement(pcb, positions)

        r1 = result.get_footprint("R1")
        assert r1 is not None
        assert r1.position.x == 15.0
        assert r1.position.y == 15.0
        assert r1.rotation == 90.0

        r2 = result.get_footprint("R2")
        assert r2 is not None
        assert r2.position.x == 25.0
        assert r2.position.y == 25.0
        assert r2.rotation == 180.0

    def test_apply_best_placement_preserves_unmoved(self) -> None:
        fp1 = _make_footprint("R1", 10.0, 10.0, 0.0)
        fp2 = _make_footprint("C1", 30.0, 30.0, 45.0)
        pcb = _minimal_pcb(fp1, fp2)

        # Only move R1, leave C1 alone
        positions: tuple[tuple[str, float, float, float], ...] = (
            ("R1", 5.0, 5.0, 270.0),
        )
        result = apply_best_placement(pcb, positions)

        c1 = result.get_footprint("C1")
        assert c1 is not None
        assert c1.position.x == 30.0
        assert c1.position.y == 30.0
        assert c1.rotation == 45.0

    def test_apply_best_placement_empty_positions(self) -> None:
        fp1 = _make_footprint("R1", 10.0, 10.0, 0.0)
        pcb = _minimal_pcb(fp1)

        result = apply_best_placement(pcb, ())
        r1 = result.get_footprint("R1")
        assert r1 is not None
        assert r1.position.x == 10.0


# ---------------------------------------------------------------------------
# Agent: atomic write
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    """Tests for OptimizerAgent._atomic_write_json."""

    def test_atomic_write_json_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        OptimizerAgent._atomic_write_json(path, {"key": "value"})
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"key": "value"}

    def test_atomic_write_json_overwrites(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        OptimizerAgent._atomic_write_json(path, {"version": 1})
        OptimizerAgent._atomic_write_json(path, {"version": 2})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"version": 2}


# ---------------------------------------------------------------------------
# Agent: init and generate_suggestions
# ---------------------------------------------------------------------------


class TestOptimizerAgent:
    """Tests for OptimizerAgent methods."""

    def test_optimizer_agent_init(self, tmp_path: Path) -> None:
        agent = OptimizerAgent(tmp_path)
        assert agent._root == tmp_path
        assert agent._progress is None

    def test_generate_suggestions_low_electrical(self, tmp_path: Path) -> None:
        from kicad_pipeline.optimization.scoring import QualityScore

        agent = OptimizerAgent(tmp_path)
        low_elec = QualityScore(
            board_cost=0.0,
            electrical_score=0.5,
            manufacturing_score=1.0,
            thermal_score=1.0,
            signal_integrity_score=1.0,
            placement_score=1.0,
            overall_score=0.8,
            grade="B",
            breakdown=(),
        )
        pcb = _minimal_pcb()
        # zone_strategy=None will skip zone suggestions
        suggestions = agent._generate_suggestions(low_elec, low_elec, None, pcb, None)  # type: ignore[arg-type]
        assert len(suggestions) >= 1
        cats = [s.category for s in suggestions]
        assert "electrical" in cats

    def test_generate_suggestions_low_placement(self, tmp_path: Path) -> None:
        from kicad_pipeline.optimization.scoring import QualityScore

        agent = OptimizerAgent(tmp_path)
        low_place = QualityScore(
            board_cost=0.0,
            electrical_score=1.0,
            manufacturing_score=1.0,
            thermal_score=1.0,
            signal_integrity_score=1.0,
            placement_score=0.5,
            overall_score=0.9,
            grade="A",
            breakdown=(),
        )
        suggestions = agent._generate_suggestions(low_place, low_place, None, _minimal_pcb(), None)  # type: ignore[arg-type]
        cats = [s.category for s in suggestions]
        assert "placement" in cats

    def test_generate_suggestions_all_good(self, tmp_path: Path) -> None:
        from kicad_pipeline.optimization.scoring import QualityScore

        agent = OptimizerAgent(tmp_path)
        good = QualityScore(
            board_cost=0.0,
            electrical_score=1.0,
            manufacturing_score=1.0,
            thermal_score=1.0,
            signal_integrity_score=1.0,
            placement_score=1.0,
            overall_score=1.0,
            grade="A",
            breakdown=(),
        )
        suggestions = agent._generate_suggestions(good, good, None, _minimal_pcb(), None)  # type: ignore[arg-type]
        # No dimensional suggestions when all scores are perfect
        # (zone_strategy=None means no zone suggestions either)
        assert len(suggestions) == 0

    def test_generate_suggestions_with_zone_strategy(self, tmp_path: Path) -> None:
        from kicad_pipeline.optimization.scoring import QualityScore
        from kicad_pipeline.optimization.zone_optimizer import ZoneStrategy

        agent = OptimizerAgent(tmp_path)
        good = QualityScore(
            board_cost=0.0,
            electrical_score=1.0,
            manufacturing_score=1.0,
            thermal_score=1.0,
            signal_integrity_score=1.0,
            placement_score=1.0,
            overall_score=1.0,
            grade="A",
            breakdown=(),
        )
        zone_strat = ZoneStrategy(
            gnd_strategy="both",
            power_zones=(),
            copper_fill_ratio=0.7,
            thermal_relief_style="relief",
            rationale=("Standard design; GND plane on both F.Cu and B.Cu.",),
        )
        suggestions = agent._generate_suggestions(good, good, zone_strat, _minimal_pcb(), None)  # type: ignore[arg-type]
        cats = [s.category for s in suggestions]
        assert "zone" in cats


# ---------------------------------------------------------------------------
# CommandType enum extensions
# ---------------------------------------------------------------------------


class TestCommandType:
    """Verify new CommandType enum members exist."""

    def test_command_type_includes_optimize(self) -> None:
        assert CommandType.OPTIMIZE.value == "optimize"
        assert CommandType.APPLY_OPTIMIZATION.value == "apply_optimization"

    def test_command_type_original_values_intact(self) -> None:
        assert CommandType.RERUN.value == "rerun"
        assert CommandType.BUG_UPDATE.value == "bug_update"
        assert CommandType.RELOAD.value == "reload"
