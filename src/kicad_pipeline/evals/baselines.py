"""Baseline persistence — JSONL load/save/compare.

Follows the same append-only JSONL pattern as ``evidence/ledger.py``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from kicad_pipeline.evals.models import EvalBaseline, EvalResult, ScoreResult, SoftTarget

_log = logging.getLogger(__name__)

DEFAULT_BASELINES_PATH = Path(__file__).parents[3] / "data" / "evals" / "baselines.jsonl"


def load_baselines(baselines_path: Path = DEFAULT_BASELINES_PATH) -> dict[str, EvalBaseline]:
    """Load the latest baseline per case_id from JSONL.

    Returns empty dict if file doesn't exist.
    """
    if not baselines_path.exists():
        return {}

    baselines: dict[str, EvalBaseline] = {}
    for line_no, line in enumerate(baselines_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            bl = EvalBaseline.from_dict(d)
            baselines[bl.case_id] = bl  # latest wins
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            _log.warning("baselines.jsonl line %d: %s", line_no, exc)
    return baselines


def save_baseline(baselines_path: Path, result: EvalResult) -> None:
    """Append a new baseline entry from a passing eval result."""
    baselines_path.parent.mkdir(parents=True, exist_ok=True)
    bl = EvalBaseline.from_result(result)
    with baselines_path.open("a") as f:
        f.write(bl.to_json() + "\n")
    _log.info("Baseline saved: %s (%.3f %s)", bl.case_id, bl.overall_score, result.grade)


def rewrite_baselines(baselines_path: Path = DEFAULT_BASELINES_PATH) -> None:
    """Rewrite JSONL keeping only the latest baseline per case_id.

    Deduplicates entries accumulated from repeated ``--update-baselines`` runs.
    """
    baselines = load_baselines(baselines_path)
    if not baselines:
        return
    with baselines_path.open("w") as f:
        for bl in baselines.values():
            f.write(bl.to_json() + "\n")
    _log.info("Baselines rewritten: %d entries", len(baselines))


def compare_to_baseline(
    targets: tuple[SoftTarget, ...],
    breakdown: tuple[tuple[str, float], ...],
    overall_score: float,
    baseline: EvalBaseline | None,
) -> tuple[ScoreResult, ...]:
    """Compare result dimensions against baseline, computing regression_pct.

    If baseline is None (first run), all scores pass if above min_value.
    """
    dim_map: dict[str, float] = dict(breakdown)
    dim_map["overall_score"] = overall_score

    baseline_map: dict[str, float] = {}
    if baseline is not None:
        baseline_map = dict(baseline.dimension_scores)
        baseline_map["overall_score"] = baseline.overall_score

    results: list[ScoreResult] = []
    for target in targets:
        value = dim_map.get(target.dimension, 0.0)
        bl_value = baseline_map.get(target.dimension)

        # Check absolute floor
        above_min = value >= target.min_value

        # Check regression
        regression_pct: float | None = None
        no_regression = True
        if bl_value is not None and bl_value > 0:
            regression_pct = (bl_value - value) / bl_value
            if regression_pct > target.regression_threshold:
                no_regression = False

        results.append(ScoreResult(
            dimension=target.dimension,
            value=value,
            target_min=target.min_value,
            baseline_value=bl_value,
            regression_pct=regression_pct,
            passed=above_min and no_regression,
        ))

    return tuple(results)
