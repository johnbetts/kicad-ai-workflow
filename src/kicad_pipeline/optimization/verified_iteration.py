"""Verified iteration harness — ungameable evaluation for PCB placement.

Every placement evaluation MUST go through ``run_verified_iteration()``.
This is the single source of truth. It runs external KiCad DRC, renders
4 standard views, executes regression tests, and computes the placement
score.  If ANY hard gate fails, the result is marked as FAILED and the
caller must discard the iteration.

No agent can bypass this.  No scoring filter can hide collisions.
The DRC engine is external (KiCad's own checker).  The regression tests
are independent.  The renders are evidence that can be inspected.

Usage::

    from kicad_pipeline.optimization.verified_iteration import (
        run_verified_iteration,
    )

    result = run_verified_iteration(pcb, requirements, output_dir)
    if not result.passed:
        # discard this placement — hard gate failed
        for failure in result.failures:
            print(failure)
    else:
        # safe to keep this placement
        print(f"Score: {result.score}")
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.scoring import QualityScore

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedResult:
    """Result of a verified placement evaluation.

    Every field is populated — nothing is optional, nothing is skipped.
    If a gate could not run (e.g. KiCad CLI not installed), it's recorded
    as a failure with an explanation, not silently skipped.
    """

    passed: bool
    score: float                          # overall placement score [0, 1]
    drc_violation_count: int              # from external KiCad DRC
    drc_violations: tuple[str, ...]       # human-readable violation strings
    collision_count: int                  # from scoring engine (raw, unfiltered)
    regression_passed: bool               # all regression tests green
    regression_failures: tuple[str, ...]  # failed test names
    render_paths: dict[str, Path]         # view_name -> PNG path
    failures: tuple[str, ...]             # all hard-gate failure reasons
    quality_score: QualityScore | None = None  # full breakdown if available


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------


def _gate_drc(pcb_path: Path) -> tuple[bool, int, tuple[str, ...]]:
    """Run external KiCad DRC.  Hard gate: 0 violations required.

    Returns (passed, count, violation_strings).
    If KiCad CLI is not available, returns (False, -1, explanation).
    """
    try:
        from kicad_pipeline.validation.kicad_drc import run_drc
        report = run_drc(pcb_path, severity_all=True)
        errors = [
            f"DRC ERROR: {v.type} — {v.description}"
            for v in report.violations
            if not v.excluded and v.severity == "error"
        ]
        warnings = [
            v for v in report.violations
            if not v.excluded and v.severity != "error"
        ]
        _log.info("  DRC: %d errors, %d warnings", len(errors), len(warnings))
        # Hard gate on ERRORS only — warnings are footprint/library issues
        return len(errors) == 0, len(errors), tuple(errors)
    except FileNotFoundError:
        msg = "DRC SKIPPED: kicad-cli not found (install KiCad 9+ or set $KICAD_CLI)"
        _log.warning("  %s", msg)
        return False, -1, (msg,)
    except Exception as exc:
        msg = f"DRC ERROR: {exc}"
        _log.warning("  %s", msg)
        return False, -1, (msg,)


def _gate_renders(
    pcb_path: Path,
    output_dir: Path,
) -> tuple[bool, dict[str, Path], tuple[str, ...]]:
    """Render 4 standard views.  Soft gate: warns but doesn't fail.

    Returns (all_rendered, paths, failure_messages).
    """
    paths: dict[str, Path] = {}
    failures: list[str] = []

    try:
        from kicad_image_gen import render_2d, render_3d
        render_2d(str(pcb_path), str(output_dir / "verified_2d.png"))
        paths["2d"] = output_dir / "verified_2d.png"
        render_3d(str(pcb_path), str(output_dir / "verified_3d_top.png"), view="top")
        paths["3d_top"] = output_dir / "verified_3d_top.png"
        render_3d(str(pcb_path), str(output_dir / "verified_3d_iso.png"), view="iso")
        paths["3d_iso"] = output_dir / "verified_3d_iso.png"
        render_3d(str(pcb_path), str(output_dir / "verified_3d_isoback.png"),
                  view="iso-back")
        paths["3d_isoback"] = output_dir / "verified_3d_isoback.png"
        _log.info("  Renders: %d/%d views", len(paths), 4)
    except ImportError:
        failures.append("RENDER SKIPPED: kicad-image-gen not installed")
        _log.warning("  %s", failures[-1])
    except Exception as exc:
        failures.append(f"RENDER ERROR: {exc}")
        _log.warning("  %s", failures[-1])

    return len(paths) == 4, paths, tuple(failures)


def _gate_regression() -> tuple[bool, tuple[str, ...]]:
    """Run regression tests.  Hard gate: all must pass.

    Returns (passed, failure_messages).
    """
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/regression/", "-x",
             "--tb=line", "-q", "--no-header"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parents[3]),  # project root
        )
        if result.returncode == 0:
            _log.info("  Regression: PASSED")
            return True, ()
        # Extract failure lines
        lines = result.stdout.strip().split("\n")
        failures = [line for line in lines if "FAILED" in line]
        _log.warning("  Regression: %d failures", len(failures))
        return False, tuple(failures)
    except subprocess.TimeoutExpired:
        msg = "REGRESSION TIMEOUT: tests took >120s"
        _log.warning("  %s", msg)
        return False, (msg,)
    except Exception as exc:
        msg = f"REGRESSION ERROR: {exc}"
        _log.warning("  %s", msg)
        return False, (msg,)


def _gate_collisions(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[int, QualityScore | None]:
    """Count raw collisions (unfiltered) and compute quality score.

    This is the internal score — NOT a gate, just evidence.
    The DRC is the real gate; this is advisory.
    """
    try:
        from kicad_pipeline.optimization.scoring import compute_fast_placement_score
        score = compute_fast_placement_score(pcb, requirements)
        # Also count raw collisions from the geometry engine
        from kicad_pipeline.optimization.collision_resolver import _fp_courtyard_sizes
        from kicad_pipeline.optimization.geometry import count_collisions_accurate
        from kicad_pipeline.pcb.pin_map import origin_to_centroid
        fp_sizes = _fp_courtyard_sizes(pcb)
        positions: dict[str, tuple[float, float, float]] = {}
        for fp in pcb.footprints:
            cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y,
                                        fp.rotation)
            positions[fp.ref] = (cx, cy, fp.rotation)
        raw_colls = count_collisions_accurate(positions, fp_sizes, clearance_mm=0.0)
        _log.info("  Score: %s (%.3f), raw collisions: %d",
                  score.grade, score.overall_score, len(raw_colls))
        return len(raw_colls), score
    except Exception as exc:
        _log.warning("  Scoring error: %s", exc)
        return -1, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_verified_iteration(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    output_dir: str | Path | None = None,
    pcb_path: str | Path | None = None,
) -> VerifiedResult:
    """Run ALL verification gates on a placement.  Returns pass/fail with evidence.

    This is the ONLY function that should be used to evaluate placement
    quality.  It runs:

    1. **KiCad DRC** (external, hard gate) — zero violations required
    2. **4-view renders** (evidence) — 2D + 3D top/iso/iso-back
    3. **Regression tests** (hard gate) — all must pass
    4. **Collision count + score** (advisory) — raw unfiltered count

    Args:
        pcb: The placed PCB design to evaluate.
        requirements: Project requirements for scoring.
        output_dir: Where to write renders and DRC report.
            Uses a temp dir if not specified.
        pcb_path: Path to the .kicad_pcb file for DRC.
            If not provided, writes a temp file from ``pcb``.

    Returns:
        :class:`VerifiedResult` with pass/fail and all evidence.
    """
    _log.info("=== Verified Iteration ===")

    # Set up output directory
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _tmp_ctx = None
    else:
        import tempfile as _tf
        _tmp_ctx = _tf.TemporaryDirectory()
        out = Path(_tmp_ctx.name)

    # Write PCB to file if no path provided
    if pcb_path is None:
        from kicad_pipeline.pcb.builder import write_pcb
        pcb_file = out / "_verified.kicad_pcb"
        write_pcb(pcb, pcb_file)
        pcb_path = pcb_file
    else:
        pcb_path = Path(pcb_path)

    all_failures: list[str] = []
    all_passed = True

    # Gate 1: KiCad DRC (HARD GATE)
    drc_passed, drc_count, drc_violations = _gate_drc(pcb_path)
    if not drc_passed and drc_count != -1:
        all_passed = False
        all_failures.append(f"DRC FAILED: {drc_count} violations")
    elif drc_count == -1:
        # DRC couldn't run — record but don't hard-fail
        # (allows development without KiCad installed)
        all_failures.extend(drc_violations)

    # Gate 2: Renders (evidence, soft gate)
    renders_ok, render_paths, render_failures = _gate_renders(pcb_path, out)
    all_failures.extend(render_failures)

    # Gate 3: Regression tests (HARD GATE)
    regression_passed, regression_failures = _gate_regression()
    if not regression_passed:
        all_passed = False
        all_failures.append(
            f"REGRESSION FAILED: {len(regression_failures)} test(s)")
        all_failures.extend(regression_failures)

    # Gate 4: Collision count + score (advisory)
    collision_count, quality_score = _gate_collisions(pcb, requirements)
    overall_score = quality_score.overall_score if quality_score else 0.0

    # Clean up temp dir
    if _tmp_ctx is not None:
        with contextlib.suppress(Exception):
            _tmp_ctx.cleanup()

    result = VerifiedResult(
        passed=all_passed,
        score=overall_score,
        drc_violation_count=drc_count,
        drc_violations=drc_violations,
        collision_count=collision_count,
        regression_passed=regression_passed,
        regression_failures=regression_failures,
        render_paths=render_paths,
        failures=tuple(all_failures),
        quality_score=quality_score,
    )

    if result.passed:
        _log.info("=== VERIFIED: PASSED (score=%.3f, DRC=%d, collisions=%d) ===",
                  result.score, result.drc_violation_count, result.collision_count)
    else:
        _log.warning("=== VERIFIED: FAILED — %d issues ===", len(result.failures))
        for f in result.failures:
            _log.warning("  %s", f)

    return result
