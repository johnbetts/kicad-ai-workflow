"""Regression tests — no dimension should regress beyond threshold from baseline."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .conftest import GOLDEN_CASES

if TYPE_CHECKING:
    from kicad_pipeline.evals.models import EvalCase
    from kicad_pipeline.evals.runner import EvalRunner


@pytest.mark.slow
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c.case_id)
def test_no_regression(case: EvalCase, eval_runner: EvalRunner) -> None:
    """No dimension should regress more than threshold from baseline."""
    result = eval_runner.run_case(case)
    regressed = [
        s for s in result.score_results
        if s.regression_pct is not None
        and s.regression_pct > 0
        and not s.passed
    ]
    assert not regressed, (
        f"Regressions in {case.case_id}: "
        + ", ".join(
            f"{s.dimension} dropped {s.regression_pct:.1%} "
            f"({s.baseline_value:.3f} -> {s.value:.3f})"
            for s in regressed
        )
    )
