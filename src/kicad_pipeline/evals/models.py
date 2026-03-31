"""Eval framework data models — frozen dataclasses for golden-case evaluation.

Inspired by LangChain AgentEvals patterns, adapted for PCB design pipelines.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from kicad_pipeline.models.requirements import ProjectRequirements


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardGate:
    """Binary pass/fail gate. Failing any hard gate = eval failure."""

    name: str
    description: str


@dataclass(frozen=True)
class SoftTarget:
    """Continuous score target. Checked against baseline for regression."""

    dimension: str
    min_value: float
    regression_threshold: float = 0.05


@dataclass(frozen=True)
class EvalCase:
    """A single evaluation case — one board config to build and score."""

    case_id: str
    board_name: str
    description: str
    build_fn: Callable[[], ProjectRequirements]
    board_width_mm: float
    board_height_mm: float
    hard_gates: tuple[HardGate, ...]
    soft_targets: tuple[SoftTarget, ...]
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    """Result of a single hard gate evaluation."""

    gate_name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ScoreResult:
    """Result of a single soft target evaluation."""

    dimension: str
    value: float
    target_min: float
    baseline_value: float | None = None
    regression_pct: float | None = None
    passed: bool = True


@dataclass(frozen=True)
class EvalResult:
    """Complete result of evaluating one case."""

    case_id: str
    git_commit: str
    git_dirty: bool
    timestamp: str
    passed: bool
    duration_secs: float
    gate_results: tuple[GateResult, ...]
    score_results: tuple[ScoreResult, ...]
    overall_score: float
    grade: str
    breakdown: tuple[tuple[str, float], ...]
    error: str | None = None

    def to_json(self) -> str:
        """Serialize to JSON string for JSONL storage."""
        d = asdict(self)
        return json.dumps(d, default=str)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalBaseline:
    """Stored baseline for regression comparison."""

    case_id: str
    git_commit: str
    timestamp: str
    overall_score: float
    dimension_scores: tuple[tuple[str, float], ...]

    def to_json(self) -> str:
        """Serialize to JSON string for JSONL storage."""
        d = asdict(self)
        return json.dumps(d, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> EvalBaseline:
        """Deserialize from parsed JSON dict."""
        dims = d.get("dimension_scores", ())
        return cls(
            case_id=str(d["case_id"]),
            git_commit=str(d["git_commit"]),
            timestamp=str(d["timestamp"]),
            overall_score=float(d["overall_score"]),  # type: ignore[arg-type]
            dimension_scores=tuple(
                (str(k), float(v)) for k, v in dims
            ),
        )

    @classmethod
    def from_result(cls, result: EvalResult) -> EvalBaseline:
        """Create a baseline from a passing eval result."""
        return cls(
            case_id=result.case_id,
            git_commit=result.git_commit,
            timestamp=result.timestamp,
            overall_score=result.overall_score,
            dimension_scores=result.breakdown,
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalReport:
    """Summary of a full eval suite run."""

    run_id: str
    git_commit: str
    timestamp: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    results: tuple[EvalResult, ...]
    regressions: tuple[str, ...]
    duration_secs: float
