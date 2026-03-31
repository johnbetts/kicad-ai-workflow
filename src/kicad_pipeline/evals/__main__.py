"""CLI entry point for the eval framework.

Usage::

    python -m kicad_pipeline.evals                    # all golden cases
    python -m kicad_pipeline.evals --tags mcu          # filter by tag
    python -m kicad_pipeline.evals --with-variants     # include package variants
    python -m kicad_pipeline.evals --update-baselines  # save baselines on pass
    python -m kicad_pipeline.evals --case mcu_core     # single case
"""
from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from kicad_pipeline.evals.golden_cases import all_golden_cases
from kicad_pipeline.evals.runner import EvalRunner
from kicad_pipeline.evals.variants import all_cases_with_variants

if TYPE_CHECKING:
    from kicad_pipeline.evals.models import EvalReport


def _print_report(report: EvalReport) -> None:
    """Print human-readable eval report."""
    print(f"\n{'=' * 60}")
    print(f"PCB Eval Suite — git {report.git_commit}")
    print(f"{'=' * 60}")

    for i, result in enumerate(report.results, 1):
        status = "PASS" if result.passed else "FAIL"
        tag = f"({result.overall_score:.2f} {result.grade})"
        err = f" ERROR: {result.error}" if result.error else ""
        dots = "." * max(1, 35 - len(result.case_id))
        print(f"\n[{i}/{report.total_cases}] {result.case_id} {dots} "
              f"{status} {tag} {result.duration_secs:.1f}s{err}")

        # Hard gates
        failed_gates = [g for g in result.gate_results if not g.passed]
        if failed_gates:
            for g in failed_gates:
                print(f"  GATE FAIL: {g.gate_name} — {g.detail}")
        else:
            print(f"  Hard gates: {len(result.gate_results)}/{len(result.gate_results)} passed")

        # Soft targets
        for s in result.score_results:
            status_mark = "ok" if s.passed else "FAIL"
            bl_str = ""
            if s.baseline_value is not None and s.regression_pct is not None:
                delta = -s.regression_pct * 100
                bl_str = f", baseline={s.baseline_value:.3f}, delta={delta:+.1f}%"
            print(f"  {s.dimension}: {s.value:.3f} (min={s.target_min}{bl_str}) [{status_mark}]")

    print(f"\n{'=' * 60}")
    print(f"Passed: {report.passed_cases}/{report.total_cases} | "
          f"Regressions: {len(report.regressions)} | "
          f"Duration: {report.duration_secs:.1f}s")
    if report.regressions:
        print(f"Regressed cases: {', '.join(report.regressions)}")
    print(f"{'=' * 60}")


def main() -> None:
    """Run the eval suite."""
    parser = argparse.ArgumentParser(description="PCB pipeline eval suite")
    parser.add_argument("--case", help="Run single case by case_id")
    parser.add_argument("--tags", help="Filter cases by tag (comma-separated)")
    parser.add_argument("--with-variants", action="store_true",
                        help="Include package-size variants")
    parser.add_argument("--update-baselines", action="store_true",
                        help="Update baselines for passing cases")
    parser.add_argument("--force-update-baselines", action="store_true",
                        help="Reset baselines for any case that passes hard gates "
                        "(ignores soft target regressions)")
    args = parser.parse_args()

    # Build case list
    golden = all_golden_cases()
    cases = all_cases_with_variants(golden) if args.with_variants else golden

    # Filter by case_id
    if args.case:
        cases = tuple(c for c in cases if c.case_id == args.case)
        if not cases:
            print(f"Unknown case: {args.case}")
            print(f"Available: {', '.join(c.case_id for c in golden)}")
            sys.exit(1)

    # Filter by tags
    if args.tags:
        tag_set = frozenset(args.tags.split(","))
        cases = tuple(c for c in cases if tag_set & frozenset(c.tags))

    if not cases:
        print("No cases to run.")
        sys.exit(1)

    print(f"Running {len(cases)} eval case(s)...")

    runner = EvalRunner(
        update_baselines=args.update_baselines,
        force_update_baselines=args.force_update_baselines,
    )
    report = runner.run_all(cases)
    _print_report(report)

    if args.update_baselines or args.force_update_baselines:
        from kicad_pipeline.evals.baselines import rewrite_baselines

        rewrite_baselines()

    # Exit 1 only on regressions (score dropped from baseline), not on
    # pre-existing failures.  This allows committing fixes that don't
    # worsen existing issues.
    sys.exit(1 if report.regressions else 0)


if __name__ == "__main__":
    main()
