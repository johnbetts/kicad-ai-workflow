"""PCB pipeline evaluation framework.

Usage::

    python -m kicad_pipeline.evals                    # all golden cases
    python -m kicad_pipeline.evals --case mcu_core    # single case
    python -m kicad_pipeline.evals --update-baselines  # save baselines on pass
    python -m kicad_pipeline.evals --with-variants     # include package variants
"""
from __future__ import annotations

from kicad_pipeline.evals.build_utils import archive_and_clean
from kicad_pipeline.evals.golden_cases import all_golden_cases
from kicad_pipeline.evals.models import EvalCase, EvalReport, EvalResult
from kicad_pipeline.evals.runner import EvalRunner
from kicad_pipeline.evals.variants import all_cases_with_variants

__all__ = [
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
    "all_cases_with_variants",
    "all_golden_cases",
    "archive_and_clean",
    "run_eval_check",
]


def run_eval_check(case_id: str | None = None) -> tuple[bool, str]:
    """Quick eval check for use in iterative loops and build scripts.

    Args:
        case_id: Run a specific case, or None for all golden cases.

    Returns:
        (passed, summary_text) tuple.

    Example::

        from kicad_pipeline.evals import run_eval_check
        passed, summary = run_eval_check("mcu_core")
        if not passed:
            print(f"Eval failed: {summary}")
    """
    cases = all_golden_cases()
    if case_id is not None:
        cases = tuple(c for c in cases if c.case_id == case_id)
        if not cases:
            return False, f"Unknown case: {case_id}"

    runner = EvalRunner()
    report = runner.run_all(cases)

    lines: list[str] = []
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"{r.case_id}: {status} ({r.overall_score:.2f} {r.grade})")
        if not r.passed:
            for g in r.gate_results:
                if not g.passed:
                    lines.append(f"  gate {g.gate_name}: {g.detail}")
            for s in r.score_results:
                if not s.passed:
                    lines.append(
                        f"  {s.dimension}: {s.value:.3f} < {s.target_min}"
                    )

    summary = "\n".join(lines)
    return report.failed_cases == 0, summary
