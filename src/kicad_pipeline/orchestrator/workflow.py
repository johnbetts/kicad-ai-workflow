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
        if stage_id == StageId.REQUIREMENTS:
            self._generate_requirements(vdir)
        elif stage_id == StageId.SCHEMATIC:
            self._generate_schematic(variant_name, vdir)
        elif stage_id == StageId.PCB:
            self._generate_pcb(variant_name, vdir)
        elif stage_id == StageId.VALIDATION:
            warnings.append("Validation not yet wired")
        elif stage_id == StageId.PRODUCTION:
            self._generate_production(variant_name, vdir)

    def _generate_requirements(self, vdir: Path) -> None:
        """Validate that requirements.json exists."""
        req_path = vdir / "requirements.json"
        if not req_path.exists():
            raise OrchestrationError(
                f"requirements.json not found at {req_path}"
            )
        log.info("Requirements file validated: %s", req_path)

    def _generate_schematic(self, variant_name: str, vdir: Path) -> None:
        """Build and write a schematic from requirements."""
        from kicad_pipeline.project_file import write_project_file
        from kicad_pipeline.requirements.decomposer import load_requirements
        from kicad_pipeline.schematic.builder import build_schematic, write_schematic

        req = load_requirements(vdir / "requirements.json")
        sch = build_schematic(req)
        sch_path = vdir / f"{variant_name}.kicad_sch"
        write_schematic(sch, sch_path)
        log.info("Schematic written: %s", sch_path)

        # Generate .kicad_pro so KiCad can open the project
        pro_path = vdir / f"{variant_name}.kicad_pro"
        if not pro_path.exists():
            write_project_file(variant_name, vdir)
            log.info("Project file written: %s", pro_path)

    def _generate_pcb(self, variant_name: str, vdir: Path) -> None:
        """Build and write a PCB from requirements."""
        from kicad_pipeline.pcb.builder import build_pcb, write_pcb
        from kicad_pipeline.requirements.decomposer import load_requirements

        req = load_requirements(vdir / "requirements.json")
        pcb = build_pcb(req)
        pcb_path = vdir / f"{variant_name}.kicad_pcb"
        write_pcb(pcb, pcb_path)
        log.info("PCB written: %s", pcb_path)

    def _generate_production(self, variant_name: str, vdir: Path) -> None:
        """Build and write production artifacts."""
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.production.packager import (
            build_production_package,
            write_production_package,
        )
        from kicad_pipeline.requirements.decomposer import load_requirements

        req = load_requirements(vdir / "requirements.json")
        pcb = build_pcb(req)
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
            return {"stage": "validation", "status": "not yet wired"}
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
