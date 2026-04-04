"""PipelineService — stateless facade over all pipeline builders.

Wraps raw builders with structured error handling (:class:`PipelineError`
codes), evidence-record generation, wall-clock timing, and artifact
tracking.  Every consumer (CLI, API, dashboard, tests) should call
through this facade instead of importing builders directly.

All heavy imports are deferred into each method body so that importing
this module does not pull the entire pipeline into memory.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import RequirementsError
from kicad_pipeline.service.errors import classify_error
from kicad_pipeline.service.models import (
    PipelineError,
    PipelineRequest,
    PipelineResult,
    StageOutcome,
)

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements

logger = logging.getLogger(__name__)


class PipelineService:
    """Stateless facade over pipeline builders.

    Wraps raw builders with:

    - Structured error handling (:class:`PipelineError` with codes)
    - Evidence record generation
    - Timing
    - Artifact tracking
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve_requirements(self, request: PipelineRequest) -> ProjectRequirements:
        """Load requirements from *request* (file or direct object).

        Returns the :class:`ProjectRequirements` instance to use for
        downstream stages.

        Raises:
            RequirementsError: If neither ``requirements`` nor
                ``requirements_path`` is set on the request.
        """
        if request.requirements is not None:
            return request.requirements

        if request.requirements_path is not None:
            from kicad_pipeline.requirements.decomposer import load_requirements

            return load_requirements(request.requirements_path)

        raise RequirementsError(
            "PipelineRequest must provide either 'requirements' or 'requirements_path'."
        )

    # ------------------------------------------------------------------
    # Stage: requirements
    # ------------------------------------------------------------------

    def run_requirements(self, request: PipelineRequest) -> StageOutcome:
        """Validate and optionally save requirements.

        Loads (or re-uses) the :class:`ProjectRequirements`, optionally
        writes them to ``<output_dir>/<board_name>/requirements.json``,
        and records an evidence entry.
        """
        t0 = time.monotonic()
        artifacts: list[str] = []
        errors: list[PipelineError] = []
        warnings: list[str] = []

        try:
            req = self.resolve_requirements(request)

            # Persist a normalised copy so later stages can reload.
            from kicad_pipeline.requirements.decomposer import save_requirements

            out_dir = request.output_dir / request.board_name
            out_dir.mkdir(parents=True, exist_ok=True)
            req_path = out_dir / "requirements.json"
            save_requirements(req, req_path)
            artifacts.append(str(req_path))

            if not req.components:
                warnings.append("Requirements contain no components.")

            logger.info(
                "Requirements resolved: %d components, %d nets",
                len(req.components),
                len(req.nets),
            )

            self._record_evidence(
                request,
                stage="requirements",
                step="resolve",
                passed=True,
                summary=f"Loaded {len(req.components)} components, {len(req.nets)} nets",
                artifacts=artifacts,
            )

        except Exception as exc:
            errors.append(classify_error(exc, stage="requirements"))
            logger.error("Requirements resolution failed: %s", exc)

        elapsed = time.monotonic() - t0
        return StageOutcome(
            stage="requirements",
            success=len(errors) == 0,
            artifacts=tuple(artifacts),
            errors=tuple(errors),
            warnings=tuple(warnings),
            duration_secs=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Stage: schematic
    # ------------------------------------------------------------------

    def run_schematic(self, request: PipelineRequest) -> StageOutcome:
        """Generate schematic files from requirements."""
        t0 = time.monotonic()
        artifacts: list[str] = []
        errors: list[PipelineError] = []
        warnings: list[str] = []

        try:
            from kicad_pipeline.schematic.builder import (
                build_project_schematics,
                write_hierarchical_schematic,
                write_schematic,
            )

            req = self.resolve_requirements(request)
            schematics = build_project_schematics(req, project_name=request.board_name)

            out_dir = request.output_dir / request.board_name
            out_dir.mkdir(parents=True, exist_ok=True)

            if len(schematics) == 1:
                sch_path = out_dir / f"{request.board_name}.kicad_sch"
                write_schematic(next(iter(schematics.values())), str(sch_path))
                artifacts.append(str(sch_path))
            else:
                written = write_hierarchical_schematic(
                    schematics, out_dir, request.board_name
                )
                artifacts.extend(str(p) for p in written)

            logger.info("Schematic generated: %d file(s)", len(artifacts))

            self._record_evidence(
                request,
                stage="schematic",
                step="generate",
                passed=True,
                summary=f"Generated {len(artifacts)} schematic file(s)",
                artifacts=artifacts,
            )

        except Exception as exc:
            errors.append(classify_error(exc, stage="schematic"))
            logger.error("Schematic generation failed: %s", exc)

        elapsed = time.monotonic() - t0
        return StageOutcome(
            stage="schematic",
            success=len(errors) == 0,
            artifacts=tuple(artifacts),
            errors=tuple(errors),
            warnings=tuple(warnings),
            duration_secs=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Stage: pcb
    # ------------------------------------------------------------------

    def _pcb_build_write_score(
        self, request: PipelineRequest, artifacts: list[str], warnings: list[str],
    ) -> tuple[float, str]:
        from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee
        from kicad_pipeline.optimization.scoring import compute_fast_placement_score
        from kicad_pipeline.pcb.builder import build_pcb, write_pcb
        from kicad_pipeline.validation.collisions import check_collisions
        from kicad_pipeline.validation.containment import check_board_containment

        req = self.resolve_requirements(request)
        pcb = build_pcb(
            req,
            board_width_mm=request.board_width_mm,
            board_height_mm=request.board_height_mm,
            placement_mode=request.placement_mode,
            auto_route=request.auto_route,
            project_name=request.board_name,
        )
        # Load reference positions if a reference board was provided
        ref_positions = None
        if request.reference_pcb_path is not None:
            from kicad_pipeline.optimization.reference_comparator import (
                load_reference_positions,
            )
            ref_positions = load_reference_positions(request.reference_pcb_path)
            logger.info("Loaded %d reference positions from %s",
                        len(ref_positions), request.reference_pcb_path)

        pcb, _review = optimize_placement_ee(
            req, pcb, reference_positions=ref_positions,
        )

        out_dir = request.output_dir / request.board_name
        out_dir.mkdir(parents=True, exist_ok=True)
        pcb_path = out_dir / f"{request.board_name}.kicad_pcb"
        write_pcb(pcb, pcb_path)
        artifacts.append(str(pcb_path))

        quality = compute_fast_placement_score(pcb, req)
        score_overall = quality.overall_score
        score_grade = quality.grade
        logger.info("PCB scored %.3f (%s)", score_overall, score_grade)

        # Log reference similarity if in reference-seeded mode
        if ref_positions is not None:
            from kicad_pipeline.optimization.reference_comparator import (
                build_group_map_from_requirements,
                compare_to_reference,
            )
            from kicad_pipeline.pcb.position_extractor import positions_from_pcb_file

            current_positions = positions_from_pcb_file(pcb_path)
            group_map = build_group_map_from_requirements(req)
            similarity = compare_to_reference(current_positions, ref_positions, group_map)
            logger.info("Reference similarity: %.1f%%", similarity.get("overall", 0.0))

        containment_violations = check_board_containment(pcb)
        for cv in containment_violations:
            warnings.append(f"Containment: {cv}")
        collision_violations = check_collisions(pcb)
        for col in collision_violations:
            warnings.append(f"Collision: {col}")

        self._record_evidence(
            request, stage="pcb", step="build_and_score", passed=True,
            summary=(
                f"PCB generated — score {score_overall:.3f} ({score_grade}), "
                f"{len(containment_violations)} containment, "
                f"{len(collision_violations)} collision warnings"
            ),
            artifacts=artifacts,
            details={
                "score_overall": score_overall, "score_grade": score_grade,
                "containment_violations": len(containment_violations),
                "collision_violations": len(collision_violations),
            },
        )
        return score_overall, score_grade

    def run_pcb(self, request: PipelineRequest) -> StageOutcome:
        """Generate PCB, optimise placement, score, and validate.

        Pipeline:

        1. ``build_pcb()`` with the placement mode from *request*
        2. ``optimize_placement_ee()``
        3. ``write_pcb()`` to ``<output_dir>/<board_name>/``
        4. ``compute_fast_placement_score()`` — populates
           ``score_overall`` / ``score_grade`` on the outcome
        5. Run ``check_board_containment()`` and ``check_collisions()``
           — violations become warnings
        """
        t0 = time.monotonic()
        artifacts: list[str] = []
        errors: list[PipelineError] = []
        warnings: list[str] = []
        score_overall: float | None = None
        score_grade: str | None = None

        try:
            score_overall, score_grade = self._pcb_build_write_score(
                request, artifacts, warnings,
            )
        except Exception as exc:
            errors.append(classify_error(exc, stage="pcb"))
            logger.error("PCB generation failed: %s", exc)

        elapsed = time.monotonic() - t0
        return StageOutcome(
            stage="pcb",
            success=len(errors) == 0,
            artifacts=tuple(artifacts),
            errors=tuple(errors),
            warnings=tuple(warnings),
            duration_secs=round(elapsed, 2),
            score_overall=score_overall,
            score_grade=score_grade,
        )

    # ------------------------------------------------------------------
    # Stage: validation
    # ------------------------------------------------------------------

    def _run_all_checks(
        self,
        request: PipelineRequest,
        errors: list[PipelineError],
        warnings: list[str],
    ) -> None:
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.validation.collisions import check_collisions
        from kicad_pipeline.validation.containment import check_board_containment
        from kicad_pipeline.validation.drc import Severity as DRCSeverity
        from kicad_pipeline.validation.drc import run_drc
        from kicad_pipeline.validation.electrical import run_electrical_checks
        from kicad_pipeline.validation.manufacturing import run_manufacturing_checks
        from kicad_pipeline.validation.pcb_integrity import validate_pcb_integrity

        req = self.resolve_requirements(request)
        pcb = build_pcb(
            req,
            board_width_mm=request.board_width_mm,
            board_height_mm=request.board_height_mm,
            placement_mode=request.placement_mode,
            auto_route=False,
            project_name=request.board_name,
        )

        for violation in run_drc(pcb).violations:
            if violation.severity == DRCSeverity.ERROR:
                errors.append(PipelineError(
                    code="KAP-021", message=str(violation),
                    cause="DRC violation found.",
                    fix="Review DRC report and fix clearance / drill / width issues.",
                    severity="fatal", stage="validation",
                ))
            else:
                warnings.append(f"DRC: {violation}")

        for mfg_v in run_manufacturing_checks(pcb).violations:
            warnings.append(f"Manufacturing: {mfg_v}")

        for elec_v in run_electrical_checks(pcb, req).violations:
            warnings.append(f"Electrical: {elec_v}")

        for issue in validate_pcb_integrity(pcb, req):
            if issue.severity in ("critical", "major"):
                errors.append(PipelineError(
                    code="KAP-020", message=str(issue),
                    cause="PCB integrity check failed.",
                    fix="Review integrity report and fix component / net mismatches.",
                    severity="fatal", stage="validation",
                ))
            else:
                warnings.append(f"Integrity: {issue}")

        for cv in check_board_containment(pcb):
            warnings.append(f"Containment: {cv}")
        for col in check_collisions(pcb):
            warnings.append(f"Collision: {col}")

    def run_validation(self, request: PipelineRequest) -> StageOutcome:
        """Run all validation checks on an existing PCB.

        Expects a ``.kicad_pcb`` file to already exist under
        ``<output_dir>/<board_name>/``.  Runs:

        - ``run_drc()``
        - ``run_manufacturing_checks()``
        - ``run_electrical_checks()``
        - ``validate_pcb_integrity()``
        - ``check_board_containment()``
        - ``check_collisions()``

        Critical DRC / integrity issues are surfaced as
        :class:`PipelineError` instances.  Minor issues become warnings.
        """
        t0 = time.monotonic()
        artifacts: list[str] = []
        errors: list[PipelineError] = []
        warnings: list[str] = []

        try:
            self._run_all_checks(request, errors, warnings)
            logger.info(
                "Validation complete: %d error(s), %d warning(s)", len(errors), len(warnings),
            )
            self._record_evidence(
                request, stage="validation", step="full_check",
                passed=len(errors) == 0,
                summary=f"Validation: {len(errors)} error(s), {len(warnings)} warning(s)",
                details={"error_count": len(errors), "warning_count": len(warnings)},
            )
        except Exception as exc:
            errors.append(classify_error(exc, stage="validation"))
            logger.error("Validation failed: %s", exc)

        elapsed = time.monotonic() - t0
        return StageOutcome(
            stage="validation",
            success=len(errors) == 0,
            artifacts=tuple(artifacts),
            errors=tuple(errors),
            warnings=tuple(warnings),
            duration_secs=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Stage: production
    # ------------------------------------------------------------------

    def run_production(self, request: PipelineRequest) -> StageOutcome:
        """Generate production artifacts (Gerbers, BOM, CPL, drill, ZIP)."""
        t0 = time.monotonic()
        artifacts: list[str] = []
        errors: list[PipelineError] = []
        warnings: list[str] = []

        try:
            from kicad_pipeline.pcb.builder import build_pcb
            from kicad_pipeline.production.packager import (
                build_production_package,
            )

            req = self.resolve_requirements(request)

            # Rebuild PCB (same caveat as run_validation).
            pcb = build_pcb(
                req,
                board_width_mm=request.board_width_mm,
                board_height_mm=request.board_height_mm,
                placement_mode=request.placement_mode,
                auto_route=request.auto_route,
                project_name=request.board_name,
            )

            package = build_production_package(pcb, request.board_name, req)

            # Collect artifact paths from the package.
            if hasattr(package, "zip_path") and package.zip_path:
                artifacts.append(str(package.zip_path))
            if hasattr(package, "gerber_dir") and package.gerber_dir:
                artifacts.append(str(package.gerber_dir))
            if hasattr(package, "bom_path") and package.bom_path:
                artifacts.append(str(package.bom_path))
            if hasattr(package, "cpl_path") and package.cpl_path:
                artifacts.append(str(package.cpl_path))

            logger.info(
                "Production package generated: %d artifact(s)", len(artifacts)
            )

            self._record_evidence(
                request,
                stage="production",
                step="package",
                passed=True,
                summary=f"Generated {len(artifacts)} production artifact(s)",
                artifacts=artifacts,
            )

        except Exception as exc:
            errors.append(classify_error(exc, stage="production"))
            logger.error("Production generation failed: %s", exc)

        elapsed = time.monotonic() - t0
        return StageOutcome(
            stage="production",
            success=len(errors) == 0,
            artifacts=tuple(artifacts),
            errors=tuple(errors),
            warnings=tuple(warnings),
            duration_secs=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full(self, request: PipelineRequest) -> PipelineResult:
        """Run all stages sequentially. Stops on first fatal error.

        Executes: requirements → schematic → pcb → validation → production.
        If any stage produces a fatal error, subsequent stages are skipped
        and a partial :class:`PipelineResult` is returned.
        """
        outcomes: list[StageOutcome] = []
        board_path: str | None = None
        production_path: str | None = None

        stages: tuple[tuple[str, object], ...] = (
            ("requirements", self.run_requirements),
            ("schematic", self.run_schematic),
            ("pcb", self.run_pcb),
            ("validation", self.run_validation),
            ("production", self.run_production),
        )

        for stage_name, runner in stages:
            logger.info("Running stage: %s", stage_name)
            outcome = runner(request)  # type: ignore[operator]
            outcomes.append(outcome)

            # Track key artifact paths.
            for art in outcome.artifacts:
                if art.endswith(".kicad_pcb"):
                    board_path = art
                if art.endswith(".zip"):
                    production_path = art

            # Stop on fatal errors.
            has_fatal = any(
                e.severity == "fatal" for e in outcome.errors
            )
            if has_fatal:
                logger.warning(
                    "Stage '%s' had fatal errors — stopping pipeline.",
                    stage_name,
                )
                break

        overall_success = all(o.success for o in outcomes)

        return PipelineResult(
            outcomes=tuple(outcomes),
            overall_success=overall_success,
            board_path=board_path,
            production_path=production_path,
        )

    # ------------------------------------------------------------------
    # Evidence helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _record_evidence(
        request: PipelineRequest,
        *,
        stage: str,
        step: str,
        passed: bool,
        summary: str,
        artifacts: list[str] | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Append an evidence record for the given stage.

        Failures to write evidence are logged but never propagate — the
        pipeline should not abort because evidence recording broke.
        """
        try:
            from kicad_pipeline.evidence.ledger import append_record
            from kicad_pipeline.evidence.models import (
                EvidenceKind,
                EvidenceRecord,
            )

            board_dir = request.output_dir / request.board_name
            pcb_path = board_dir / f"{request.board_name}.kicad_pcb"

            record = EvidenceRecord(
                kind=EvidenceKind.VERIFICATION,
                stage=stage,
                step=step,
                board=request.board_name,
                producer="PipelineService",
                passed=passed,
                summary=summary,
                artifacts=list(artifacts or []),
                details=dict(details or {}),
            )

            append_record(pcb_path, record)

        except Exception:
            logger.debug(
                "Evidence recording failed for stage=%s step=%s (non-fatal)",
                stage,
                step,
                exc_info=True,
            )
