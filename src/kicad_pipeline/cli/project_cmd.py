"""CLI commands for the ``project`` subcommand group.

Provides interactive project management: initialization, variant creation,
stage lifecycle (generate/review/approve/rollback), revision management,
and release creation.
"""

from __future__ import annotations

import argparse  # noqa: TC003 — used at runtime for parser construction
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from kicad_pipeline.exceptions import OrchestrationError
from kicad_pipeline.orchestrator.manifest import (
    MANIFEST_FILENAME,
    load_manifest,
    save_manifest,
)
from kicad_pipeline.orchestrator.models import (
    PackageStrategy,
    ProjectManifest,
    StageId,
    VariantRecord,
    VariantStatus,
    default_stages,
    get_strategy_by_name,
)


def add_project_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``project`` command group with all sub-subcommands."""
    project_p = subparsers.add_parser("project", help="Manage orchestrated projects")
    project_sub = project_p.add_subparsers(dest="project_command")

    # -- init --------------------------------------------------------------
    init_p = project_sub.add_parser("init", help="Initialize a new project")
    init_p.add_argument("--name", required=True, help="Project name")
    init_p.add_argument("--description", "-d", default="", help="Project description")
    init_p.add_argument("--spec", help="Path to natural-language spec file")

    # -- status ------------------------------------------------------------
    status_p = project_sub.add_parser("status", help="Show project status")
    status_p.add_argument("--variant", help="Show status for a specific variant")

    # -- variant -----------------------------------------------------------
    variant_p = project_sub.add_parser("variant", help="Manage variants")
    variant_sub = variant_p.add_subparsers(dest="variant_command")

    vc = variant_sub.add_parser("create", help="Create a new variant")
    vc.add_argument("--name", required=True, help="Variant slug (e.g. compact-0603)")
    vc.add_argument("--display-name", help="Human-readable name")
    vc.add_argument(
        "--strategy", default="0805",
        help="Package strategy (0805, 0603, 0402, through-hole)",
    )
    vc.add_argument("--description", "-d", default="", help="Variant description")

    variant_sub.add_parser("list", help="List all variants")

    va = variant_sub.add_parser("activate", help="Set the active variant")
    va.add_argument("name", help="Variant name to activate")

    # -- stage -------------------------------------------------------------
    stage_p = project_sub.add_parser("stage", help="Manage pipeline stages")
    stage_sub = stage_p.add_subparsers(dest="stage_command")

    sg = stage_sub.add_parser("generate", help="Generate current/specified stage")
    sg.add_argument("--variant", help="Variant name (default: active)")
    sg.add_argument("--stage", help="Stage to generate (default: current)")

    sr = stage_sub.add_parser("review", help="Review current/specified stage")
    sr.add_argument("--variant", help="Variant name (default: active)")
    sr.add_argument("--stage", help="Stage to review")

    sa = stage_sub.add_parser("approve", help="Approve current/specified stage")
    sa.add_argument("--variant", help="Variant name (default: active)")
    sa.add_argument("--stage", help="Stage to approve")

    sb = stage_sub.add_parser("rollback", help="Roll back to a previous stage")
    sb.add_argument("--variant", help="Variant name (default: active)")
    sb.add_argument("--to", required=True, dest="to_stage", help="Stage to roll back to")

    # -- revision ----------------------------------------------------------
    rev_p = project_sub.add_parser("revision", help="Manage production revisions")
    rev_sub = rev_p.add_subparsers(dest="revision_command")

    rc = rev_sub.add_parser("create", help="Create a production revision snapshot")
    rc.add_argument("--variant", help="Variant name (default: active)")
    rc.add_argument("--notes", default="", help="Revision notes")

    rev_sub.add_parser("list", help="List revisions").add_argument(
        "--variant", help="Variant name (default: active)"
    )

    rf = rev_sub.add_parser("fab", help="Mark revision as sent to fabrication")
    rf.add_argument("--variant", help="Variant name (default: active)")
    rf.add_argument("--rev", type=int, required=True, help="Revision number")
    rf.add_argument("--order-id", help="Fabrication order ID")

    # -- release -----------------------------------------------------------
    rel_p = project_sub.add_parser("release", help="Create a release")
    rel_p.add_argument("--variant", help="Variant name (default: active)")
    rel_p.add_argument("--version", required=True, help="Version tag (e.g. v1.0)")


def dispatch_project(args: argparse.Namespace) -> int:
    """Dispatch ``project`` subcommands. Returns exit code."""
    cmd = getattr(args, "project_command", None)
    if cmd is None:
        print("Usage: kicad-pipeline project {init,status,variant,stage,revision,release}")
        return 0

    from collections.abc import Callable  # noqa: TC003 — used at runtime

    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "init": _cmd_init,
        "status": _cmd_status,
        "variant": _cmd_variant,
        "stage": _cmd_stage,
        "revision": _cmd_revision,
        "release": _cmd_release,
    }
    handler = handlers.get(cmd)
    if handler is None:
        print(f"Unknown project command: {cmd}", file=sys.stderr)
        return 1

    try:
        return handler(args)
    except OrchestrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _resolve_variant(args: argparse.Namespace, manifest: ProjectManifest) -> str:
    """Resolve variant name from args or active variant."""
    name = getattr(args, "variant", None)
    if name:
        return str(name)
    if manifest.active_variant:
        return manifest.active_variant
    raise OrchestrationError(
        "No variant specified and no active variant set. "
        "Use --variant or 'project variant activate <name>'"
    )


def _parse_stage_id(stage_str: str) -> StageId:
    """Parse a stage string to StageId enum."""
    try:
        return StageId(stage_str)
    except ValueError as exc:
        valid = ", ".join(s.value for s in StageId)
        raise OrchestrationError(
            f"Unknown stage {stage_str!r}. Valid stages: {valid}"
        ) from exc


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> int:
    """Handle ``project init``."""
    root = Path.cwd()
    manifest_path = root / MANIFEST_FILENAME

    if manifest_path.exists():
        print(f"Project already initialized at {root}")
        return 1

    now = _now_iso()
    spec_path = ""
    if args.spec:
        spec_path = args.spec

    manifest = ProjectManifest(
        project_name=args.name,
        description=args.description,
        original_spec=spec_path,
        created_at=now,
        updated_at=now,
    )

    # Create base directory
    (root / "base").mkdir(exist_ok=True)
    (root / "variants").mkdir(exist_ok=True)

    save_manifest(manifest, root)
    print(f"Initialized project '{args.name}' at {root}")
    print(f"  Manifest: {manifest_path}")
    print("  Next: create a variant with 'project variant create --name <name>'")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Handle ``project status``."""
    root = Path.cwd()
    manifest = load_manifest(root)

    print(f"Project: {manifest.project_name}")
    print(f"Description: {manifest.description}")
    if manifest.active_variant:
        print(f"Active variant: {manifest.active_variant}")
    print(f"Variants: {len(manifest.variants)}")
    print()

    variant_filter = getattr(args, "variant", None)

    for v in manifest.variants:
        if variant_filter and v.name != variant_filter:
            continue
        print(f"  {v.display_name} [{v.name}]")
        print(f"    Status: {v.status.value}")
        print(f"    Strategy: {v.package_strategy.name}")
        for sr in v.stages:
            marker = "x" if sr.state.value == "approved" else " "
            gen = f" (gen {sr.generation_count})" if sr.generation_count > 0 else ""
            print(f"    [{marker}] {sr.stage.value}: {sr.state.value}{gen}")
        if v.revisions:
            print(f"    Revisions: {len(v.revisions)}")
            for rev in v.revisions:
                fab = " [SENT TO FAB]" if rev.sent_to_fab else ""
                print(f"      rev{rev.number}: {rev.git_tag}{fab}")
        print()

    return 0


def _cmd_variant(args: argparse.Namespace) -> int:
    """Handle ``project variant {create,list,activate}``."""
    vcmd = getattr(args, "variant_command", None)
    if vcmd is None:
        print("Usage: kicad-pipeline project variant {create,list,activate}")
        return 0

    root = Path.cwd()
    manifest = load_manifest(root)

    if vcmd == "list":
        if not manifest.variants:
            print("No variants defined.")
            return 0
        for v in manifest.variants:
            active = " (active)" if v.name == manifest.active_variant else ""
            print(f"  {v.name}: {v.display_name} [{v.status.value}]{active}")
        return 0

    if vcmd == "activate":
        name = args.name
        found = any(v.name == name for v in manifest.variants)
        if not found:
            print(f"ERROR: Variant {name!r} not found", file=sys.stderr)
            return 1
        from dataclasses import replace
        manifest = replace(manifest, active_variant=name, updated_at=_now_iso())
        save_manifest(manifest, root)
        print(f"Active variant set to '{name}'")
        return 0

    if vcmd == "create":
        return _create_variant(args, root, manifest)

    print(f"Unknown variant command: {vcmd}", file=sys.stderr)
    return 1


def _create_variant(
    args: argparse.Namespace, root: Path, manifest: ProjectManifest
) -> int:
    """Create a new variant."""
    from dataclasses import replace as dc_replace

    name: str = args.name
    # Check for duplicate
    if any(v.name == name for v in manifest.variants):
        print(f"ERROR: Variant {name!r} already exists", file=sys.stderr)
        return 1

    strategy = get_strategy_by_name(args.strategy)
    if strategy is None:
        strategy = PackageStrategy(name=args.strategy)

    display = getattr(args, "display_name", None) or name.replace("-", " ").title()
    now = _now_iso()

    variant = VariantRecord(
        name=name,
        display_name=display,
        description=args.description,
        status=VariantStatus.DRAFT,
        package_strategy=strategy,
        stages=default_stages(),
        created_at=now,
        updated_at=now,
    )

    new_variants = (*manifest.variants, variant)
    active = manifest.active_variant or name
    manifest = dc_replace(
        manifest, variants=new_variants, active_variant=active, updated_at=now,
    )

    # Create variant directory
    vdir = root / "variants" / name
    vdir.mkdir(parents=True, exist_ok=True)

    save_manifest(manifest, root)
    print(f"Created variant '{name}' with strategy '{strategy.name}'")
    if active == name:
        print("  Set as active variant")
    return 0


def _cmd_stage(args: argparse.Namespace) -> int:
    """Handle ``project stage {generate,review,approve,rollback}``."""
    scmd = getattr(args, "stage_command", None)
    if scmd is None:
        print("Usage: kicad-pipeline project stage {generate,review,approve,rollback}")
        return 0

    root = Path.cwd()
    manifest = load_manifest(root)
    variant_name = _resolve_variant(args, manifest)

    stage_str = getattr(args, "stage", None) or getattr(args, "to_stage", None)
    stage_id: StageId | None = _parse_stage_id(stage_str) if stage_str else None

    from kicad_pipeline.orchestrator.workflow import WorkflowEngine

    engine = WorkflowEngine(root)

    if scmd == "generate":
        result = engine.generate_stage(variant_name, stage_id)
        if result.success:
            print(f"Generated: {result.message}")
            for w in result.warnings:
                print(f"  WARNING: {w}")
        else:
            print(f"FAILED: {result.message}", file=sys.stderr)
            return 1
        return 0

    if scmd == "review":
        summary = engine.review_stage(variant_name, stage_id)
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if scmd == "approve":
        result = engine.approve_stage(variant_name, stage_id)
        if result.success:
            print(f"Approved: {result.message}")
        else:
            print(f"FAILED: {result.message}", file=sys.stderr)
            return 1
        return 0

    if scmd == "rollback":
        if stage_id is None:
            print("ERROR: --to is required for rollback", file=sys.stderr)
            return 1
        result = engine.rollback_stage(variant_name, stage_id)
        print(f"Rollback: {result.message}")
        return 0

    print(f"Unknown stage command: {scmd}", file=sys.stderr)
    return 1


def _cmd_revision(args: argparse.Namespace) -> int:
    """Handle ``project revision {create,list,fab}``."""
    rcmd = getattr(args, "revision_command", None)
    if rcmd is None:
        print("Usage: kicad-pipeline project revision {create,list,fab}")
        return 0

    root = Path.cwd()
    manifest = load_manifest(root)
    variant_name = _resolve_variant(args, manifest)

    from dataclasses import replace as dc_replace

    from kicad_pipeline.orchestrator.revisions import RevisionManager

    mgr = RevisionManager(root)

    if rcmd == "create":
        rev = mgr.create_revision(variant_name, manifest, notes=args.notes)
        # Update manifest with new revision
        for v in manifest.variants:
            if v.name == variant_name:
                new_revisions = (*v.revisions, rev)
                updated_v = dc_replace(v, revisions=new_revisions, updated_at=_now_iso())
                new_variants = tuple(
                    updated_v if vr.name == variant_name else vr
                    for vr in manifest.variants
                )
                manifest = dc_replace(manifest, variants=new_variants, updated_at=_now_iso())
                save_manifest(manifest, root)
                break
        print(f"Created revision {rev.number} for '{variant_name}' (tag: {rev.git_tag})")
        return 0

    if rcmd == "list":
        revisions = mgr.list_revisions(variant_name, manifest)
        if not revisions:
            print(f"No revisions for variant '{variant_name}'")
            return 0
        for rev in revisions:
            fab = " [SENT TO FAB]" if rev.sent_to_fab else ""
            order = f" (order: {rev.fab_order_id})" if rev.fab_order_id else ""
            print(f"  rev{rev.number}: {rev.created_at}{fab}{order}")
            if rev.notes:
                print(f"    Notes: {rev.notes}")
        return 0

    if rcmd == "fab":
        manifest, rev = mgr.mark_sent_to_fab(
            variant_name, args.rev, manifest, order_id=getattr(args, "order_id", None)
        )
        save_manifest(manifest, root)
        order_info = f" (order: {rev.fab_order_id})" if rev.fab_order_id else ""
        print(f"Marked rev{rev.number} as sent to fabrication{order_info}")
        return 0

    print(f"Unknown revision command: {rcmd}", file=sys.stderr)
    return 1


def _cmd_release(args: argparse.Namespace) -> int:
    """Handle ``project release``."""
    root = Path.cwd()
    manifest = load_manifest(root)
    variant_name = _resolve_variant(args, manifest)
    version: str = args.version

    from dataclasses import replace as dc_replace

    # Update variant status and released_tag
    for v in manifest.variants:
        if v.name == variant_name:
            updated_v = dc_replace(
                v,
                status=VariantStatus.RELEASED,
                released_tag=f"{variant_name}/{version}",
                updated_at=_now_iso(),
            )
            new_variants = tuple(
                updated_v if vr.name == variant_name else vr
                for vr in manifest.variants
            )
            manifest = dc_replace(manifest, variants=new_variants, updated_at=_now_iso())
            save_manifest(manifest, root)
            print(f"Released variant '{variant_name}' as {version}")
            print(f"  Tag: {variant_name}/{version}")
            return 0

    print(f"ERROR: Variant {variant_name!r} not found", file=sys.stderr)
    return 1
