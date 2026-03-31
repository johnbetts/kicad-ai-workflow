"""Eval runner — orchestrates build -> score -> gate -> compare."""
from __future__ import annotations

import logging
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from kicad_pipeline.evals.baselines import (
    DEFAULT_BASELINES_PATH,
    compare_to_baseline,
    load_baselines,
    save_baseline,
)
from kicad_pipeline.evals.models import (
    EvalBaseline,
    EvalCase,
    EvalReport,
    EvalResult,
    GateResult,
)

_log = logging.getLogger(__name__)


def _git_info() -> tuple[str, bool]:
    """Return (short_sha, is_dirty)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
        dirty = bool(subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, timeout=5,
        ).returncode)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        sha, dirty = "unknown", False
    return sha, dirty


class EvalRunner:
    """Executes eval cases and produces results."""

    def __init__(
        self,
        baselines_path: Path = DEFAULT_BASELINES_PATH,
        update_baselines: bool = False,
        force_update_baselines: bool = False,
    ) -> None:
        self._baselines_path = baselines_path
        self._baselines = load_baselines(baselines_path)
        self._update_baselines = update_baselines
        self._force_update_baselines = force_update_baselines

    def _archive_render_dir(self, case: EvalCase) -> Path:
        render_dir = Path(__file__).parents[3] / "output" / f"eval_{case.case_id}"
        from kicad_pipeline.evals.build_utils import archive_and_clean
        archive_path = archive_and_clean(render_dir)
        if archive_path:
            _log.info("Eval %s: archived previous to %s", case.case_id, archive_path)
        return render_dir

    def _build_and_score(
        self, case: EvalCase, render_dir: Path,
    ) -> tuple[
        object | None, object | None, list[object],
        object | None, str | None, object | None,
    ]:
        pcb: object | None = None
        score: object | None = None
        build_error: str | None = None
        integrity_issues: list[object] = []
        review = None
        requirements = None
        try:
            requirements = case.build_fn()
            from kicad_pipeline.pcb.builder import build_pcb
            pcb = build_pcb(requirements, auto_route=False, placement_mode="grouped")
            from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee
            pcb, _opt_review = optimize_placement_ee(requirements, pcb)
            from kicad_pipeline.optimization.scoring import compute_fast_placement_score
            score = compute_fast_placement_score(pcb, requirements)
            from kicad_pipeline.validation.pcb_integrity import validate_pcb_integrity
            integrity_issues = validate_pcb_integrity(pcb, requirements)
            from kicad_pipeline.optimization.review_agent import review_placement
            review = review_placement(
                pcb, requirements, render_dir=render_dir, board_name=case.case_id,
            )
            if review.render_paths:
                _log.info(
                    "Eval %s: %d renders at %s",
                    case.case_id, len(review.render_paths), render_dir,
                )
        except Exception as exc:
            build_error = f"{type(exc).__name__}: {exc}"
            _log.error("Eval %s build failed: %s", case.case_id, build_error)
        return pcb, score, integrity_issues, review, build_error, requirements

    def _extract_score_fields(
        self, score: object | None,
    ) -> tuple[float, str, tuple[tuple[str, float], ...]]:
        if score is None:
            return 0.0, "F", ()
        overall = score.overall_score  # type: ignore[union-attr]
        grade = score.grade  # type: ignore[union-attr]
        breakdown = tuple((d.category, d.score) for d in score.breakdown)  # type: ignore[union-attr]
        return overall, grade, breakdown

    def _maybe_update_baseline(
        self, result: EvalResult, gate_results: tuple[GateResult, ...], build_error: str | None,
    ) -> None:
        gates_pass = all(g.passed for g in gate_results) and build_error is None
        should_save = (
            (self._update_baselines and result.passed)
            or (self._force_update_baselines and gates_pass)
        )
        if should_save:
            save_baseline(self._baselines_path, result)
            self._baselines[result.case_id] = EvalBaseline.from_result(result)

    def run_case(self, case: EvalCase) -> EvalResult:
        """Execute a single eval case end-to-end."""
        git_commit, git_dirty = _git_info()
        t0 = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()

        render_dir = self._archive_render_dir(case)
        pcb, score, integrity_issues, review, build_error, requirements = (
            self._build_and_score(case, render_dir)
        )

        if pcb is not None:
            from kicad_pipeline.evals.grok_review import is_enabled as grok_enabled
            if grok_enabled():
                self._run_grok_review(case, pcb)

        gate_results = self._evaluate_hard_gates(
            case, pcb, integrity_issues, build_error, review,
            requirements=requirements,
        )

        overall, grade, breakdown = self._extract_score_fields(score)
        baseline = self._baselines.get(case.case_id)
        score_results = compare_to_baseline(
            case.soft_targets, breakdown, overall, baseline,
        )

        all_gates_pass = all(g.passed for g in gate_results)
        all_scores_pass = all(s.passed for s in score_results)
        passed = all_gates_pass and all_scores_pass and build_error is None

        result = EvalResult(
            case_id=case.case_id,
            git_commit=git_commit,
            git_dirty=git_dirty,
            timestamp=timestamp,
            passed=passed,
            duration_secs=round(time.monotonic() - t0, 1),
            gate_results=gate_results,
            score_results=score_results,
            overall_score=overall,
            grade=grade,
            breakdown=breakdown,
            error=build_error,
        )

        self._maybe_update_baseline(result, gate_results, build_error)
        return result

    def run_all(self, cases: tuple[EvalCase, ...]) -> EvalReport:
        """Run all cases sequentially, produce report."""
        git_commit, _dirty = _git_info()
        timestamp = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()

        results: list[EvalResult] = []
        for case in cases:
            result = self.run_case(case)
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        regressions = tuple(
            r.case_id for r in results
            if any(
                s.regression_pct is not None and s.regression_pct > 0
                and not s.passed
                for s in r.score_results
            )
        )

        return EvalReport(
            run_id=uuid.uuid4().hex[:12],
            git_commit=git_commit,
            timestamp=timestamp,
            total_cases=len(results),
            passed_cases=passed,
            failed_cases=failed,
            results=tuple(results),
            regressions=regressions,
            duration_secs=round(time.monotonic() - t0, 1),
        )

    def _evaluate_hard_gates(
        self,
        case: EvalCase,
        pcb: object | None,
        integrity_issues: list[object],
        build_error: str | None,
        review: object | None = None,
        requirements: object | None = None,
    ) -> tuple[GateResult, ...]:
        """Check each hard gate against the build artifacts."""
        from kicad_pipeline.evals.dfm_gates import ALL_DFM_GATES, evaluate_dfm_gate

        results: list[GateResult] = []

        for gate in case.hard_gates:
            if gate.name == "build_succeeds":
                passed = build_error is None
                detail = build_error or "OK"
                results.append(GateResult(gate.name, passed, detail))

            elif gate.name == "zero_critical_integrity":
                critical = [
                    i for i in integrity_issues
                    if getattr(i, "severity", "") == "critical"
                ]
                passed = len(critical) == 0
                detail = f"{len(critical)} critical issues" if critical else "OK"
                results.append(GateResult(gate.name, passed, detail))

            elif gate.name == "all_components_placed":
                if pcb is None:
                    results.append(GateResult(gate.name, False, "no PCB built"))
                else:
                    fps = getattr(pcb, "footprints", ())
                    active = [f for f in fps if getattr(f, "pads", ())]
                    results.append(GateResult(
                        gate.name, len(active) > 0,
                        f"{len(active)} footprints placed",
                    ))

            elif gate.name == "all_on_board":
                if pcb is None:
                    results.append(GateResult(gate.name, False, "no PCB built"))
                else:
                    margin = 2.0
                    off_board: list[str] = []
                    for fp in getattr(pcb, "footprints", ()):
                        pos = getattr(fp, "position", None)
                        if pos is None:
                            continue
                        if (pos.x < -margin or pos.y < -margin
                                or pos.x > case.board_width_mm + margin
                                or pos.y > case.board_height_mm + margin):
                            off_board.append(getattr(fp, "ref", "?"))
                    passed = len(off_board) == 0
                    detail = (
                        f"{len(off_board)} off-board: {', '.join(off_board[:5])}"
                        if off_board else "OK"
                    )
                    results.append(GateResult(gate.name, passed, detail))

            elif gate.name == "renders_generated":
                render_paths = getattr(review, "render_paths", ()) if review else ()
                count = len(render_paths)
                passed = count >= 4
                detail = (
                    f"{count}/4 renders" if not passed
                    else f"OK ({count} renders)"
                )
                results.append(GateResult(gate.name, passed, detail))

            elif gate.name in ALL_DFM_GATES:
                # DFM gates require both PCB and requirements
                if pcb is None or requirements is None:
                    results.append(GateResult(
                        gate.name, False, "no PCB or requirements available",
                    ))
                else:
                    try:
                        result = evaluate_dfm_gate(gate.name, pcb, requirements)
                        results.append(result)
                    except Exception as exc:
                        _log.error("DFM gate %s failed: %s", gate.name, exc)
                        results.append(GateResult(
                            gate.name, False, f"gate error: {exc}",
                        ))

            else:
                results.append(GateResult(gate.name, False, f"unknown gate: {gate.name}"))

        return tuple(results)

    def _run_grok_review(
        self,
        case: EvalCase,
        pcb: object,
    ) -> object | None:
        """Run optional Grok PCB design review with rendered images.

        Renders 2D + 3D images to a temp directory, then sends to Grok
        vision for second-opinion review.  Returns GrokReviewResult or None.
        """
        import tempfile

        from kicad_pipeline.evals.grok_review import review_pcb_design

        fp_count = len(getattr(pcb, "footprints", ()))

        # Try to find existing renders, or skip
        # (rendering requires kicad-image-gen which may not be available)
        out_dir = Path(tempfile.mkdtemp(prefix="grok_review_"))
        img_2d = out_dir / "board_2d.png"
        img_3d = out_dir / "board_3d.png"

        try:
            import subprocess

            # Write PCB to temp file for rendering
            from kicad_pipeline.pcb.builder import write_pcb

            pcb_path = out_dir / "board.kicad_pcb"
            write_pcb(pcb, pcb_path)  # type: ignore[arg-type]

            subprocess.run(
                ["kicad-image-gen", "2d", str(pcb_path), "-o", str(img_2d)],
                capture_output=True, timeout=30,
            )
            subprocess.run(
                ["kicad-image-gen", "3d", str(pcb_path), "--view", "iso",
                 "-o", str(img_3d)],
                capture_output=True, timeout=30,
            )
        except Exception as exc:
            _log.debug("Grok review render failed: %s", exc)
            return None

        if not img_2d.exists() or not img_3d.exists():
            return None

        result = review_pcb_design(
            image_2d=img_2d,
            image_3d=img_3d,
            board_name=case.board_name,
            component_count=fp_count,
        )

        if result.findings:
            _log.info(
                "Grok review %s: %s", case.case_id, result.summary,
            )

        return result
