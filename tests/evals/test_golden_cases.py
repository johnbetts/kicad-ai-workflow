"""Golden case eval tests — each training board must pass hard gates and soft targets."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from .conftest import GOLDEN_CASES

if TYPE_CHECKING:
    from kicad_pipeline.evals.models import EvalCase
    from kicad_pipeline.evals.runner import EvalRunner


@pytest.mark.slow
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c.case_id)
def test_golden_case_hard_gates(case: EvalCase, eval_runner: EvalRunner) -> None:
    """Every golden case must pass all hard gates."""
    result = eval_runner.run_case(case)
    failed = [g for g in result.gate_results if not g.passed]
    assert not failed, (
        f"Hard gates failed for {case.case_id}: "
        + ", ".join(f"{g.gate_name}: {g.detail}" for g in failed)
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c.case_id)
def test_golden_case_soft_targets(case: EvalCase, eval_runner: EvalRunner) -> None:
    """Every golden case must meet soft score targets."""
    result = eval_runner.run_case(case)
    failed = [s for s in result.score_results if not s.passed]
    assert not failed, (
        f"Soft targets failed for {case.case_id}: "
        + ", ".join(f"{s.dimension}={s.value:.3f} (min={s.target_min})" for s in failed)
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c.case_id)
def test_golden_case_no_build_error(case: EvalCase, eval_runner: EvalRunner) -> None:
    """Every golden case must build without error."""
    result = eval_runner.run_case(case)
    assert result.error is None, f"Build error for {case.case_id}: {result.error}"


@pytest.mark.slow
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c.case_id)
def test_golden_case_renders_exist(case: EvalCase, eval_runner: EvalRunner) -> None:
    """Every golden case must produce 4 standard render images."""
    result = eval_runner.run_case(case)
    # The renders_generated hard gate should have passed
    render_gate = next(
        (g for g in result.gate_results if g.gate_name == "renders_generated"),
        None,
    )
    assert render_gate is not None, "renders_generated gate not found"
    assert render_gate.passed, f"Renders not generated: {render_gate.detail}"


@pytest.mark.slow
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c.case_id)
def test_golden_case_render_files_valid(case: EvalCase, eval_runner: EvalRunner) -> None:
    """Render files must exist on disk and be non-empty PNGs."""
    result = eval_runner.run_case(case)
    # Skip file checks if the build itself failed
    assert result.error is None, f"Build error for {case.case_id}: {result.error}"

    render_dir = Path(__file__).parents[2] / "output" / f"eval_{case.case_id}"

    expected_suffixes = ["_2d_top.png", "_3d_top.png", "_3d_iso.png", "_3d_isoback.png"]
    for suffix in expected_suffixes:
        img = render_dir / f"{case.case_id}{suffix}"
        assert img.exists(), f"Missing render: {img}"
        size = img.stat().st_size
        assert size > 1000, (
            f"Render too small (corrupt?): {img} ({size} bytes)"
        )
