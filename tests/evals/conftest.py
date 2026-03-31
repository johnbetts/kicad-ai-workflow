"""Shared fixtures for eval tests."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.evals.golden_cases import all_golden_cases
from kicad_pipeline.evals.runner import EvalRunner

if TYPE_CHECKING:
    from kicad_pipeline.evals.models import EvalCase, EvalResult

# All golden cases include renders_generated as a hard gate — the test suite
# enforces that 4 standard render PNGs (2D top, 3D top, 3D iso, 3D iso-back)
# exist on disk and are non-empty after each eval run.
GOLDEN_CASES = all_golden_cases()


class CachedEvalRunner(EvalRunner):
    """EvalRunner that memoizes results per case_id within a test session.

    Each golden case is built once; all test functions see the same result.
    Eliminates non-determinism from the optimizer producing different scores
    across test functions for the same case.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._cache: dict[str, EvalResult] = {}

    def run_case(self, case: EvalCase) -> EvalResult:
        if case.case_id not in self._cache:
            self._cache[case.case_id] = super().run_case(case)
        return self._cache[case.case_id]


@pytest.fixture(scope="session")
def eval_runner(tmp_path_factory: pytest.TempPathFactory) -> CachedEvalRunner:
    """Session-scoped eval runner with result caching and real baselines."""
    baselines_path = Path(__file__).parents[2] / "data" / "evals" / "baselines.jsonl"
    return CachedEvalRunner(baselines_path=baselines_path)
