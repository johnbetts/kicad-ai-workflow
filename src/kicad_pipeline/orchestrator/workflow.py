"""Workflow engine: stage-gated lifecycle management for design variants.

The :class:`WorkflowEngine` operates on a project directory containing a
``kicad-project.json`` manifest.  It enforces stage ordering, transition
rules, and coordinates artifact generation across the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import OrchestrationError
from kicad_pipeline.orchestrator.manifest import load_manifest, save_manifest
from kicad_pipeline.orchestrator.models import (
    STAGE_ORDER,
    StageId,
    StageRecord,
    StageState,
)

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.orchestrator.models import ProjectManifest, VariantRecord

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageResult:
    """Outcome of a workflow operation (generate, approve, rollback)."""

    success: bool
    stage: StageId
    message: str
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _stage_index(stage_id: StageId) -> int:
    """Return the zero-based index of *stage_id* in STAGE_ORDER."""
    for i, sid in enumerate(STAGE_ORDER):
        if sid == stage_id:
            return i
    raise OrchestrationError(f"Unknown stage: {stage_id}")  # pragma: no cover


def _render_placement_pngs(
    pcb: object,
    req: object,
    pcb_path: Path,
    vdir: Path,
    variant_name: str,
) -> None:
    """Render placement PNGs to project output directory (non-blocking)."""
    try:
        from kicad_pipeline.optimization.functional_grouper import classify_voltage_domains
        from kicad_pipeline.optimization.placement_optimizer import _build_group_map
        from kicad_pipeline.visualization.placement_render import render_placement

        output_dir = vdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        group_map = _build_group_map(req)
        render_placement(
            pcb, req, output_dir / "placement_groups.png",
            title=f"{variant_name} (Groups)", group_map=group_map,
        )
        domain_map = classify_voltage_domains(req)
        render_placement(
            pcb, req, output_dir / "placement_domains.png",
            title=f"{variant_name} (Domains)", domain_map=domain_map,
        )
        log.info("Placement renders written: %s", output_dir)

        try:
            from kicad_pipeline.visualization.kicad_export import export_pcb_image
            export_pcb_image(pcb_path, output_dir / "placement_hifi.png", pcb=pcb)
        except Exception:
            log.debug("kicad-cli hi-fi export unavailable (non-blocking)")
    except Exception:
        log.exception("Placement render failed (non-blocking)")


def _report_footprint_provenance(pcb: object, vdir: Path) -> None:
    """Log and write a provenance report for footprint sources.

    Parametric footprints need manual verification against datasheets.
    Writes ``output/footprint_provenance.json`` for downstream tools.
    """
    from kicad_pipeline.models.pcb import PCBDesign

    if not isinstance(pcb, PCBDesign):
        return

    jlcpcb: list[str] = []
    parametric: list[str] = []
    parametric_fallback: list[str] = []
    other: list[str] = []

    for fp in pcb.footprints:
        src = fp.footprint_source
        entry = f"{fp.ref} ({fp.lib_id})"
        if src == "jlcpcb":
            jlcpcb.append(entry)
        elif src == "parametric-fallback":
            parametric_fallback.append(entry)
        elif src == "parametric":
            parametric.append(entry)
        else:
            other.append(entry)

    needs_verify = parametric + parametric_fallback
    if needs_verify:
        log.warning(
            "Footprint verification needed: %d parametric footprints "
            "(pad geometry not verified against datasheets): %s",
            len(needs_verify),
            ", ".join(needs_verify),
        )
    if parametric_fallback:
        log.warning(
            "UNKNOWN footprint IDs fell back to 0805: %s",
            ", ".join(parametric_fallback),
        )
    log.info(
        "Footprint provenance: %d JLCPCB, %d parametric, %d fallback",
        len(jlcpcb), len(parametric), len(parametric_fallback),
    )

    # Write machine-readable report for downstream verification tools
    import json
    output_dir = vdir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "jlcpcb": sorted(jlcpcb),
        "parametric": sorted(parametric),
        "parametric_fallback": sorted(parametric_fallback),
        "needs_verification": sorted(needs_verify),
        "summary": {
            "total": len(pcb.footprints),
            "jlcpcb": len(jlcpcb),
            "parametric": len(parametric),
            "parametric_fallback": len(parametric_fallback),
        },
    }
    report_path = output_dir / "footprint_provenance.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Footprint provenance report: %s", report_path)


# ---------------------------------------------------------------------------
# Workflow engine
# ---------------------------------------------------------------------------


class WorkflowEngine:
    """Manage the stage lifecycle for variants in a project.

    Args:
        project_root: Path to the project directory containing
            ``kicad-project.json``.
    """

    def __init__(self, project_root: Path) -> None:
        self._root = project_root
        self._manifest: ProjectManifest = load_manifest(project_root)

    # -- persistence --------------------------------------------------------

    def _save(self) -> None:
        """Persist manifest changes to disk."""
        save_manifest(self._manifest, self._root)

    # -- variant helpers ----------------------------------------------------

    def _get_variant(self, variant_name: str) -> VariantRecord:
        """Look up a variant by name.

        Raises:
            OrchestrationError: If the variant does not exist.
        """
        for v in self._manifest.variants:
            if v.name == variant_name:
                return v
        raise OrchestrationError(f"Variant not found: {variant_name!r}")

    def _update_variant(self, updated: VariantRecord) -> None:
        """Replace a variant in the manifest by name."""
        new_variants = tuple(
            updated if v.name == updated.name else v
            for v in self._manifest.variants
        )
        self._manifest = replace(self._manifest, variants=new_variants)

    # -- stage helpers ------------------------------------------------------

    def _get_stage(self, variant: VariantRecord, stage_id: StageId) -> StageRecord:
        """Look up a stage record within a variant.

        Raises:
            OrchestrationError: If the stage is not found.
        """
        for sr in variant.stages:
            if sr.stage == stage_id:
                return sr
        raise OrchestrationError(
            f"Stage {stage_id.value!r} not found in variant {variant.name!r}"
        )

    def _update_stage(
        self, variant: VariantRecord, updated_stage: StageRecord
    ) -> VariantRecord:
        """Return a new VariantRecord with *updated_stage* replacing its match."""
        new_stages = tuple(
            updated_stage if sr.stage == updated_stage.stage else sr
            for sr in variant.stages
        )
        return replace(variant, stages=new_stages)

    def _variant_dir(self, variant_name: str) -> Path:
        """Return the filesystem path for a variant's working directory."""
        return self._root / "variants" / variant_name

    # -- public API ---------------------------------------------------------

    def get_current_stage(self, variant_name: str) -> StageId:
        """Return the earliest non-approved stage for a variant.

        If all stages are approved, returns the last stage.
        """
        variant = self._get_variant(variant_name)
        for sid in STAGE_ORDER:
            sr = self._get_stage(variant, sid)
            if sr.state != StageState.APPROVED:
                return sid
        # All approved — return last stage
        return STAGE_ORDER[-1]

    def generate_stage(
        self,
        variant_name: str,
        stage_id: StageId | None = None,
    ) -> StageResult:
        """Generate artifacts for a stage.

        If *stage_id* is ``None``, generates the current (earliest
        non-approved) stage.  The previous stage must be APPROVED before
        generation can proceed (REQUIREMENTS has no predecessor).

        Returns:
            A :class:`StageResult` describing the outcome.
        """
        variant = self._get_variant(variant_name)
        if stage_id is None:
            stage_id = self.get_current_stage(variant_name)

        idx = _stage_index(stage_id)

        # Gate: previous stage must be approved (unless this is the first)
        if idx > 0:
            prev_stage = self._get_stage(variant, STAGE_ORDER[idx - 1])
            if prev_stage.state != StageState.APPROVED:
                return StageResult(
                    success=False,
                    stage=stage_id,
                    message=(
                        f"Cannot generate {stage_id.value}: previous stage "
                        f"{STAGE_ORDER[idx - 1].value} is not approved "
                        f"(state={prev_stage.state.value})"
                    ),
                )

        vdir = self._variant_dir(variant_name)
        warnings: list[str] = []

        try:
            self._run_stage_generation(stage_id, variant_name, vdir, warnings)
        except Exception as exc:
            sr = self._get_stage(variant, stage_id)
            updated_sr = replace(sr, state=StageState.FAILED)
            variant = self._update_stage(variant, updated_sr)
            self._update_variant(variant)
            self._save()
            return StageResult(
                success=False,
                stage=stage_id,
                message=f"Generation failed: {exc}",
            )

        # Update stage record
        variant = self._get_variant(variant_name)
        sr = self._get_stage(variant, stage_id)
        updated_sr = replace(
            sr,
            state=StageState.GENERATED,
            generated_at=_now_iso(),
            generation_count=sr.generation_count + 1,
        )
        variant = self._update_stage(variant, updated_sr)
        self._update_variant(variant)
        self._save()

        return StageResult(
            success=True,
            stage=stage_id,
            message=f"Stage {stage_id.value} generated successfully",
            warnings=tuple(warnings),
        )

    def _run_stage_generation(
        self,
        stage_id: StageId,
        variant_name: str,
        vdir: Path,
        warnings: list[str],
    ) -> None:
        """Dispatch generation logic for each stage type."""
        dispatch: dict[StageId, object] = {
            StageId.REQUIREMENTS: lambda: self._generate_requirements(vdir),
            StageId.SCHEMATIC: lambda: self._generate_schematic(variant_name, vdir),
            StageId.PCB: lambda: self._generate_pcb(variant_name, vdir),
            StageId.VALIDATION: lambda: self._generate_validation(variant_name, vdir, warnings),
            StageId.PRODUCTION: lambda: self._generate_production(variant_name, vdir),
        }
        handler = dispatch.get(stage_id)
        if handler is not None:
            handler()

    def _generate_requirements(self, vdir: Path) -> None:
        """Validate that requirements.json exists and download datasheets."""
        req_path = vdir / "requirements.json"
        if not req_path.exists():
            raise OrchestrationError(
                f"requirements.json not found at {req_path}"
            )
        log.info("Requirements file validated: %s", req_path)

        # Download datasheets for components (best-effort)
        try:
            from kicad_pipeline.requirements.decomposer import load_requirements
            from kicad_pipeline.research.datasheets import download_datasheets

            req = load_requirements(req_path)
            download_datasheets(req, vdir / "docs" / "datasheets")
        except Exception:
            log.info("Datasheet download skipped or failed (non-blocking)")

    @staticmethod
    def _enrich_requirements(req: object) -> object:
        """Try to enrich requirements with JLCPCB parts. Returns req as-is on failure."""
        try:
            from kicad_pipeline.parts.jlcpcb_db import JLCPCBPartsDB
            from kicad_pipeline.parts.selector import enrich_requirements_with_parts

            with JLCPCBPartsDB() as db:
                req, _ = enrich_requirements_with_parts(req, db=db)  # type: ignore[arg-type]
        except Exception:
            log.info("JLCPCB parts DB unavailable, using requirements as-is")
        return req

    def _generate_schematic(self, variant_name: str, vdir: Path) -> None:
        """Build and write a schematic from requirements."""
        from kicad_pipeline.pcb.footprint_library import (
            build_footprint_library,
            write_fp_lib_table,
        )
        from kicad_pipeline.project_file import write_project_file
        from kicad_pipeline.requirements.decomposer import load_requirements
        from kicad_pipeline.schematic.builder import (
            build_project_schematics,
            write_hierarchical_schematic,
            write_schematic,
        )
        from kicad_pipeline.validation.consistency import compute_requirements_hash

        req = load_requirements(vdir / "requirements.json")
        req = self._enrich_requirements(req)  # type: ignore[assignment]
        schematics = build_project_schematics(req, project_name=variant_name)

        if len(schematics) == 1:
            sch = next(iter(schematics.values()))
            sch_path = vdir / f"{variant_name}.kicad_sch"
            write_schematic(sch, sch_path)
            log.info("Schematic written: %s", sch_path)
        else:
            written = write_hierarchical_schematic(schematics, vdir, variant_name)
            log.info("Hierarchical schematic written: %d files", len(written))

        # Generate project-local footprint library
        build_footprint_library(req, vdir, variant_name)
        write_fp_lib_table(vdir, variant_name)

        # Generate .kicad_pro so KiCad can open the project
        pro_path = vdir / f"{variant_name}.kicad_pro"
        if not pro_path.exists():
            write_project_file(variant_name, vdir)
            log.info("Project file written: %s", pro_path)

        # Store requirements hash for later drift detection
        req_path = vdir / "requirements.json"
        if req_path.exists():
            req_hash = compute_requirements_hash(req_path)
            variant = self._get_variant(variant_name)
            sr = self._get_stage(variant, StageId.SCHEMATIC)
            updated_sr = replace(sr, requirements_hash=req_hash)
            variant = self._update_stage(variant, updated_sr)
            self._update_variant(variant)
            log.info("Requirements hash stored: %s", req_hash[:12])

    def _generate_pcb(self, variant_name: str, vdir: Path) -> None:
        """Build and write a PCB from requirements."""
        from kicad_pipeline.pcb.board_templates import detect_template
        from kicad_pipeline.pcb.builder import build_pcb, write_pcb
        from kicad_pipeline.project_file import write_project_file
        from kicad_pipeline.requirements.decomposer import load_requirements
        from kicad_pipeline.validation.consistency import (
            check_consistency,
            check_requirements_hash,
            consistency_report_to_text,
        )

        req = load_requirements(vdir / "requirements.json")
        req = self._enrich_requirements(req)  # type: ignore[assignment]

        # Generate layout guide if not already present
        guide_path = vdir / "docs" / "layout_guide.md"
        if not guide_path.exists():
            try:
                from kicad_pipeline.research.layout_guide import generate_layout_guide

                generate_layout_guide(req, guide_path)
            except Exception:
                log.info("Layout guide generation skipped (non-blocking)")

        # Auto-detect board template from mechanical constraints
        tmpl = detect_template(req.mechanical)
        board_template = tmpl.name if tmpl is not None else None
        pcb = build_pcb(req, board_template=board_template, project_name=variant_name)
        pcb_path = vdir / f"{variant_name}.kicad_pcb"
        write_pcb(pcb, pcb_path)
        log.info("PCB written: %s", pcb_path)

        # Report footprint provenance — parametric footprints need verification
        _report_footprint_provenance(pcb, vdir)

        # Regenerate project-local footprint library (ensures PCB lib_ids match)
        from kicad_pipeline.pcb.footprint_library import (
            build_footprint_library,
            write_fp_lib_table,
        )
        build_footprint_library(req, vdir, variant_name)
        write_fp_lib_table(vdir, variant_name)

        # Regenerate project file with netclass definitions so KiCad
        # applies correct clearances and track widths when routing.
        write_project_file(
            variant_name, vdir, netclasses=pcb.netclasses,
            drc_exclusions=pcb.drc_exclusions or None,
        )
        log.info("Project file updated with netclasses: %s", vdir)

        _render_placement_pngs(pcb, req, pcb_path, vdir, variant_name)
        self._check_drift_and_consistency(
            variant_name, vdir, pcb_path,
            check_requirements_hash, check_consistency, consistency_report_to_text,
        )

    def _check_drift_and_consistency(
        self,
        variant_name: str,
        vdir: Path,
        pcb_path: Path,
        check_requirements_hash: object,
        check_consistency: object,
        consistency_report_to_text: object,
    ) -> None:
        """Check for requirements drift and schematic-PCB consistency."""
        req_path = vdir / "requirements.json"
        variant = self._get_variant(variant_name)
        sch_stage = self._get_stage(variant, StageId.SCHEMATIC)
        if sch_stage.requirements_hash and req_path.exists():
            drift = check_requirements_hash(sch_stage.requirements_hash, req_path)  # type: ignore[operator]
            if drift is not None:
                log.warning("Requirements drift: %s", drift.message)

        sch_path = vdir / f"{variant_name}.kicad_sch"
        if sch_path.exists() and pcb_path.exists():
            report = check_consistency(sch_path, pcb_path)  # type: ignore[operator]
            if not report.passed:
                log.warning(
                    "Schematic-PCB consistency: %d errors, %d warnings",
                    len(report.errors),
                    len(report.warnings),
                )
                log.warning(consistency_report_to_text(report))  # type: ignore[operator]

    def _generate_validation(
        self, variant_name: str, vdir: Path, warnings: list[str]
    ) -> None:
        """Run pre-production validation: consistency check + parts validation."""
        val_dir = vdir / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)

        self._check_consistency(variant_name, vdir, val_dir, warnings)
        pcb, req, report, db = self._validate_parts(
            variant_name, vdir, val_dir, warnings,
        )
        self._run_extended_validation(pcb, req, val_dir, warnings)
        self._enforce_parts_gate(report, val_dir)

    def _check_consistency(
        self,
        variant_name: str,
        vdir: Path,
        val_dir: Path,
        warnings: list[str],
    ) -> None:
        """Hard gate: schematic-PCB consistency check."""
        from kicad_pipeline.validation.consistency import (
            check_consistency,
            consistency_report_to_text,
        )

        sch_path = vdir / f"{variant_name}.kicad_sch"
        pcb_path = vdir / f"{variant_name}.kicad_pcb"
        if not (sch_path.exists() and pcb_path.exists()):
            return

        consistency = check_consistency(sch_path, pcb_path)
        (val_dir / "consistency_report.txt").write_text(
            consistency_report_to_text(consistency), encoding="utf-8"
        )
        log.info("Consistency report written to %s", val_dir)

        if not consistency.passed:
            raise OrchestrationError(
                f"Schematic-PCB consistency check failed: "
                f"{len(consistency.errors)} error(s). "
                f"See {val_dir / 'consistency_report.txt'}"
            )

        for w in consistency.warnings:
            warnings.append(f"Consistency: {w.message}")

    def _validate_parts(
        self,
        variant_name: str,
        vdir: Path,
        val_dir: Path,
        warnings: list[str],
    ) -> tuple[object, object, object, object]:
        """Run parts validation and return (pcb, req, report, db)."""
        from kicad_pipeline.pcb.board_templates import detect_template
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.production.bom import generate_bom
        from kicad_pipeline.production.parts_validator import (
            report_to_json,
            report_to_text,
            validate_bom_parts,
        )
        from kicad_pipeline.requirements.component_db import ComponentDB
        from kicad_pipeline.requirements.decomposer import load_requirements

        req = load_requirements(vdir / "requirements.json")
        req = self._enrich_requirements(req)  # type: ignore[assignment]
        tmpl = detect_template(req.mechanical)
        board_template = tmpl.name if tmpl is not None else None
        pcb = build_pcb(req, board_template=board_template, project_name=variant_name)
        bom_rows = generate_bom(pcb, req)

        db = ComponentDB()
        report = validate_bom_parts(
            bom_rows, db=db, check_web_stock=False, project_name=variant_name,
        )

        (val_dir / "parts_validation_report.txt").write_text(
            report_to_text(report), encoding="utf-8"
        )
        (val_dir / "parts_validation_report.json").write_text(
            report_to_json(report), encoding="utf-8"
        )
        log.info("Validation reports written to %s", val_dir)

        for ps in report.parts:
            if ps.status == "ok" and ps.tier == 1:
                part = db.find_by_lcsc(ps.lcsc)
                if part is not None and not getattr(part, "basic", True):
                    warnings.append(
                        f"{ps.lcsc} ({', '.join(ps.ref_designators)}) is extended "
                        f"-- adds $3 JLCPCB setup fee"
                    )

        if report.low_stock_count > 0:
            for ps in report.parts:
                if ps.status == "low_stock":
                    warnings.append(
                        f"{ps.lcsc} ({', '.join(ps.ref_designators)}) "
                        f"low stock: {ps.stock_qty} units -- "
                        f"approval required"
                    )

        return pcb, req, report, db

    def _run_extended_validation(
        self,
        pcb: object,
        req: object,
        val_dir: Path,
        warnings: list[str],
    ) -> None:
        """Run full validation suite (non-blocking)."""
        try:
            from kicad_pipeline.optimization.scoring import compute_quality_score
            from kicad_pipeline.validation.drc import run_drc
            from kicad_pipeline.validation.electrical import run_electrical_checks
            from kicad_pipeline.validation.manufacturing import run_manufacturing_checks
            from kicad_pipeline.validation.report import (
                build_validation_report,
                format_report_markdown,
            )
            from kicad_pipeline.validation.signal_integrity import run_si_checks
            from kicad_pipeline.validation.thermal import run_thermal_checks

            drc_report = run_drc(pcb)  # type: ignore[arg-type]
            electrical_report = run_electrical_checks(pcb, req)  # type: ignore[arg-type]
            manufacturing_report = run_manufacturing_checks(pcb)  # type: ignore[arg-type]
            thermal_report = run_thermal_checks(pcb, req)  # type: ignore[arg-type]
            si_report = run_si_checks(pcb, req)  # type: ignore[arg-type]

            full_report = build_validation_report(
                drc=drc_report, electrical=electrical_report,
                manufacturing=manufacturing_report,
                thermal=thermal_report, si=si_report,
            )

            (val_dir / "full_validation_report.md").write_text(
                format_report_markdown(full_report), encoding="utf-8"
            )

            quality = compute_quality_score(
                pcb, req, validation_report=full_report,  # type: ignore[arg-type]
            )
            self._write_quality_score(quality, val_dir)
            log.info("Quality score: %.2f (%s)", quality.overall_score, quality.grade)

            for d in quality.breakdown:
                if d.issues:
                    for issue in d.issues[:3]:
                        warnings.append(f"{d.category}: {issue}")
        except Exception as exc:
            log.warning("Extended validation failed (non-blocking): %s", exc)
            warnings.append(f"Extended validation skipped: {exc}")

    @staticmethod
    def _write_quality_score(quality: object, val_dir: Path) -> None:
        """Serialise quality score to JSON."""
        import json as _json

        (val_dir / "quality_score.json").write_text(
            _json.dumps(
                {
                    "overall_score": quality.overall_score,  # type: ignore[union-attr]
                    "grade": quality.grade,  # type: ignore[union-attr]
                    "board_cost": quality.board_cost,  # type: ignore[union-attr]
                    "electrical_score": quality.electrical_score,  # type: ignore[union-attr]
                    "manufacturing_score": quality.manufacturing_score,  # type: ignore[union-attr]
                    "thermal_score": quality.thermal_score,  # type: ignore[union-attr]
                    "signal_integrity_score": quality.signal_integrity_score,  # type: ignore[union-attr]
                    "placement_score": quality.placement_score,  # type: ignore[union-attr]
                    "breakdown": [
                        {
                            "category": d.category,  # type: ignore[union-attr]
                            "score": d.score,  # type: ignore[union-attr]
                            "weight": d.weight,  # type: ignore[union-attr]
                            "issues": list(d.issues),  # type: ignore[union-attr]
                        }
                        for d in quality.breakdown  # type: ignore[union-attr]
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _enforce_parts_gate(report: object, val_dir: Path) -> None:
        """Hard gate: block if parts are unavailable or low stock."""
        if not report.all_parts_available:  # type: ignore[union-attr]
            details: list[str] = []
            unavailable = report.unresolved_count - report.low_stock_count  # type: ignore[union-attr]
            if unavailable > 0:
                details.append(f"{unavailable} unavailable")
            if report.low_stock_count > 0:  # type: ignore[union-attr]
                details.append(
                    f"{report.low_stock_count} low stock (<1000 qty)"  # type: ignore[union-attr]
                )
            raise OrchestrationError(
                f"Parts validation failed: {', '.join(details)}. "
                f"See {val_dir / 'parts_validation_report.txt'}"
            )

    def _generate_production(self, variant_name: str, vdir: Path) -> None:
        """Build and write production artifacts."""
        from kicad_pipeline.pcb.board_templates import detect_template
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.production.packager import (
            build_production_package,
            write_production_package,
        )
        from kicad_pipeline.requirements.decomposer import load_requirements

        req = load_requirements(vdir / "requirements.json")
        tmpl = detect_template(req.mechanical)
        board_template = tmpl.name if tmpl is not None else None
        pcb = build_pcb(req, board_template=board_template, project_name=variant_name)
        pkg = build_production_package(pcb, variant_name, req)
        prod_dir = vdir / "production"
        prod_dir.mkdir(parents=True, exist_ok=True)
        write_production_package(pkg, prod_dir)
        log.info("Production artifacts written: %s", prod_dir)

    def review_stage(
        self,
        variant_name: str,
        stage_id: StageId | None = None,
    ) -> dict[str, object]:
        """Return a summary dict for the current or specified stage.

        The summary contents vary by stage type.
        """
        self._get_variant(variant_name)  # validates variant exists
        if stage_id is None:
            stage_id = self.get_current_stage(variant_name)

        vdir = self._variant_dir(variant_name)

        if stage_id == StageId.REQUIREMENTS:
            return self._review_requirements(vdir)
        if stage_id == StageId.SCHEMATIC:
            return self._review_schematic(variant_name, vdir)
        if stage_id == StageId.PCB:
            return self._review_pcb(variant_name, vdir)
        if stage_id == StageId.VALIDATION:
            return self._review_validation(vdir)
        if stage_id == StageId.PRODUCTION:
            return self._review_production(variant_name, vdir)
        return {}  # pragma: no cover

    def _review_requirements(self, vdir: Path) -> dict[str, object]:
        """Summarize requirements.json."""
        from kicad_pipeline.requirements.decomposer import load_requirements

        req_path = vdir / "requirements.json"
        if not req_path.exists():
            return {"stage": "requirements", "error": "requirements.json not found"}
        req = load_requirements(req_path)
        power_rails: list[str] = []
        if req.power_budget is not None:
            power_rails = [r.name for r in req.power_budget.rails]
        return {
            "stage": "requirements",
            "component_count": len(req.components),
            "net_count": len(req.nets),
            "power_rails": power_rails,
        }

    def _review_schematic(
        self, variant_name: str, vdir: Path
    ) -> dict[str, object]:
        """Summarize the schematic file."""
        sch_path = vdir / f"{variant_name}.kicad_sch"
        if not sch_path.exists():
            return {"stage": "schematic", "error": "schematic file not found"}
        stat = sch_path.stat()
        # Count component instances by looking for (symbol (lib_id lines
        content = sch_path.read_text(encoding="utf-8")
        component_count = content.count("(symbol (lib_id")
        return {
            "stage": "schematic",
            "file_size_bytes": stat.st_size,
            "component_count": component_count,
        }

    def _review_pcb(
        self, variant_name: str, vdir: Path
    ) -> dict[str, object]:
        """Summarize the PCB file."""
        pcb_path = vdir / f"{variant_name}.kicad_pcb"
        if not pcb_path.exists():
            return {"stage": "pcb", "error": "PCB file not found"}
        stat = pcb_path.stat()
        content = pcb_path.read_text(encoding="utf-8")
        footprint_count = content.count("(footprint ")
        return {
            "stage": "pcb",
            "file_size_bytes": stat.st_size,
            "footprint_count": footprint_count,
        }

    def _review_validation(self, vdir: Path) -> dict[str, object]:
        """Summarize parts validation and consistency reports."""
        import json as _json

        report_path = vdir / "validation" / "parts_validation_report.json"
        if not report_path.exists():
            return {"stage": "validation", "error": "validation report not found"}
        data = _json.loads(report_path.read_text(encoding="utf-8"))
        result: dict[str, object] = {
            "stage": "validation",
            "all_parts_available": data.get("all_parts_available", False),
            "unresolved_count": data.get("unresolved_count", 0),
            "total_bom_cost_usd": data.get("total_bom_cost_usd"),
            "summary": data.get("summary_text", ""),
        }

        # Include consistency report if available
        consistency_path = vdir / "validation" / "consistency_report.txt"
        if consistency_path.exists():
            result["consistency_report"] = consistency_path.read_text(
                encoding="utf-8"
            )

        return result

    def _review_production(
        self, variant_name: str, vdir: Path
    ) -> dict[str, object]:
        """Summarize production output directory."""
        prod_dir = vdir / "production"
        if not prod_dir.exists():
            return {"stage": "production", "error": "production directory not found"}
        files = sorted(
            str(p.relative_to(prod_dir))
            for p in prod_dir.rglob("*")
            if p.is_file()
        )
        bom_path = prod_dir / f"{variant_name}_bom.csv"
        bom_summary = ""
        if bom_path.exists():
            bom_lines = bom_path.read_text(encoding="utf-8").strip().splitlines()
            bom_summary = f"{len(bom_lines) - 1} rows (excl. header)"
        return {
            "stage": "production",
            "file_list": files,
            "bom_summary": bom_summary,
        }

    def approve_stage(
        self,
        variant_name: str,
        stage_id: StageId | None = None,
    ) -> StageResult:
        """Mark a stage as APPROVED.

        The stage must be in GENERATED or REVIEWING state.

        Returns:
            A :class:`StageResult` describing the outcome.
        """
        variant = self._get_variant(variant_name)
        if stage_id is None:
            stage_id = self.get_current_stage(variant_name)

        sr = self._get_stage(variant, stage_id)
        if sr.state not in (StageState.GENERATED, StageState.REVIEWING):
            return StageResult(
                success=False,
                stage=stage_id,
                message=(
                    f"Cannot approve stage {stage_id.value}: "
                    f"state is {sr.state.value}, expected generated or reviewing"
                ),
            )

        updated_sr = replace(
            sr,
            state=StageState.APPROVED,
            approved_at=_now_iso(),
        )
        variant = self._update_stage(variant, updated_sr)
        self._update_variant(variant)
        self._save()

        return StageResult(
            success=True,
            stage=stage_id,
            message=f"Stage {stage_id.value} approved",
        )

    def rollback_stage(
        self,
        variant_name: str,
        to_stage: StageId,
    ) -> StageResult:
        """Roll back a variant to a previous stage.

        Sets the target stage and all later stages to PENDING with
        reset generation counts.

        Returns:
            A :class:`StageResult` describing the outcome.
        """
        variant = self._get_variant(variant_name)
        target_idx = _stage_index(to_stage)

        new_stages: list[StageRecord] = []
        for sr in variant.stages:
            idx = _stage_index(sr.stage)
            if idx >= target_idx:
                new_stages.append(
                    StageRecord(stage=sr.stage)  # fresh PENDING record
                )
            else:
                new_stages.append(sr)

        variant = replace(variant, stages=tuple(new_stages))
        self._update_variant(variant)
        self._save()

        return StageResult(
            success=True,
            stage=to_stage,
            message=f"Rolled back to {to_stage.value}; later stages reset to pending",
        )
