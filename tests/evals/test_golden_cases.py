"""Golden case eval tests — each training board must pass hard gates and soft targets."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from .conftest import GOLDEN_CASES

if TYPE_CHECKING:
    from kicad_pipeline.evals.models import EvalCase
    from kicad_pipeline.evals.runner import EvalRunner

# Placement DFM gates — enforced as hard failures (blocking).
# These check placement correctness, NOT routing (routing is manual in KiCad).
# Gates listed here MUST pass or the test fails.
_ENFORCED_DFM_GATES = frozenset({
    "footprint_registry_match",
    "no_collisions",  # collision = unmanufacturable (council: must enforce)
    "all_pads_within_board",
    "mounting_hole_clearance",
    "schematic_pcb_sync",
    "board_sizing",
    "package_match",  # enforced after KI-022 fix
})

# Placement DFM gates — tracked (non-blocking) with promotion deadlines.
# Each gate has a target date to move to enforced.
# Promotion path: fix optimizer → threshold passes → promote to enforced.
_TRACKED_DFM_GATES = frozenset({
    "decoupling_proximity",  # target: enforce at 10mm by 2026-04-15
    "subcircuit_spread",  # target: enforce at 15mm by 2026-04-15
    "zone_membership",  # target: enforce by 2026-04-30
    "subcircuit_completeness",  # target: enforce by 2026-04-15
    "component_isolation_zones",  # target: enforce by 2026-04-30
    # package_match — ENFORCED after KI-022 fix (moved to _ENFORCED_DFM_GATES)
})


@pytest.mark.slow
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c.case_id)
def test_golden_case_hard_gates(case: EvalCase, eval_runner: EvalRunner) -> None:
    """Every golden case must pass enforced hard gates.

    DFM gates in _TRACKED_DFM_GATES are reported but non-blocking.
    Move gates from tracked → enforced as the optimizer improves.
    """
    result = eval_runner.run_case(case)
    failed = [
        g for g in result.gate_results
        if not g.passed and g.gate_name not in _TRACKED_DFM_GATES
    ]
    assert not failed, (
        f"Hard gates failed for {case.case_id}: "
        + ", ".join(f"{g.gate_name}: {g.detail}" for g in failed)
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c.case_id)
def test_dfm_gates_report(case: EvalCase, eval_runner: EvalRunner) -> None:
    """Report DFM gate results for all gates (tracked + enforced).

    This test always passes — it logs findings for tracked gates.
    Use this to monitor progress toward promoting tracked → enforced.
    """
    result = eval_runner.run_case(case)
    tracked_failures = [
        g for g in result.gate_results
        if not g.passed and g.gate_name in _TRACKED_DFM_GATES
    ]
    if tracked_failures:
        import logging
        log = logging.getLogger(__name__)
        log.warning(
            "DFM tracked issues for %s (%d): %s",
            case.case_id,
            len(tracked_failures),
            "; ".join(f"{g.gate_name}: {g.detail}" for g in tracked_failures),
        )
    # Always passes — this is a diagnostic test
    assert True


# Promotion deadlines — tracked gates MUST be promoted to enforced by these dates.
# If a deadline passes and the gate is still tracked, test_tracked_gate_deadlines fails.
_GATE_DEADLINES: dict[str, str] = {
    "decoupling_proximity": "2026-04-15",
    "subcircuit_spread": "2026-04-15",
    "zone_membership": "2026-04-30",
    "subcircuit_completeness": "2026-04-15",
    "component_isolation_zones": "2026-04-30",
}


def test_tracked_gate_deadlines() -> None:
    """Tracked gates must be promoted to enforced by their deadline date.

    If a deadline has passed and the gate is still in _TRACKED_DFM_GATES,
    either promote it or explicitly extend the deadline with justification.
    """
    today = date.today()
    overdue: list[str] = []
    for gate_name, deadline_str in _GATE_DEADLINES.items():
        deadline = date.fromisoformat(deadline_str)
        if today > deadline and gate_name in _TRACKED_DFM_GATES:
            overdue.append(
                f"{gate_name} deadline was {deadline_str}"
            )
    assert not overdue, (
        "Tracked gates past promotion deadline — promote to enforced "
        "or update deadline with justification:\n  "
        + "\n  ".join(overdue)
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
