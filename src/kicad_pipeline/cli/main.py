#!/usr/bin/env python3
"""kicad-ai-pipeline CLI entry point."""

from __future__ import annotations

import argparse
import sys


def _add_requirements_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'requirements' subcommand."""
    req_p = sub.add_parser("requirements", help="Manage project requirements")
    req_p.add_argument("--input", "-i", required=True, help="Input requirements JSON")
    req_p.add_argument("--output", "-o", help="Output requirements JSON")
    req_p.add_argument("--validate", action="store_true", help="Validate requirements only")


def _add_schematic_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'schematic' subcommand."""
    sch_p = sub.add_parser("schematic", help="Generate KiCad schematic")
    sch_p.add_argument("--requirements", "-r", required=True, help="Requirements JSON")
    sch_p.add_argument("--output", "-o", required=True, help="Output .kicad_sch file or directory")
    sch_p.add_argument(
        "--flat", action="store_true", default=False,
        help="Force flat (single-sheet) schematic output",
    )


def _add_pcb_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'pcb' subcommand."""
    pcb_p = sub.add_parser("pcb", help="Generate KiCad PCB")
    pcb_p.add_argument("--requirements", "-r", required=True, help="Requirements JSON")
    pcb_p.add_argument("--output", "-o", required=True, help="Output .kicad_pcb file")
    pcb_p.add_argument(
        "--live", action="store_true", default=False,
        help="Connect to running KiCad via IPC for zone fill and board sync",
    )


def _add_route_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'route' subcommand."""
    route_p = sub.add_parser("route", help="Autoroute PCB")
    route_p.add_argument("--pcb", "-p", required=True, help="Input .kicad_pcb file")
    route_p.add_argument("--output", "-o", required=True, help="Output .kicad_pcb file")
    route_p.add_argument("--freerouting", action="store_true", help="Use FreeRouting")


def _add_validate_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'validate' subcommand."""
    val_p = sub.add_parser("validate", help="Validate PCB design")
    val_p.add_argument("--pcb", "-p", required=True, help="PCB JSON or kicad_pcb file")
    val_p.add_argument("--report", "-r", help="Output report JSON")


def _add_produce_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'produce' subcommand."""
    prod_p = sub.add_parser("produce", help="Generate production artifacts")
    prod_p.add_argument("--pcb", "-p", required=True, help="PCB JSON or kicad_pcb file")
    prod_p.add_argument("--output", "-o", required=True, help="Output directory")
    prod_p.add_argument("--name", "-n", default="project", help="Project name")
    prod_p.add_argument(
        "--requirements", "-r", default=None, help="Requirements JSON for BOM enrichment",
    )
    prod_p.add_argument(
        "--validate-parts", action="store_true", default=False,
        help="Validate JLCPCB part availability",
    )
    prod_p.add_argument(
        "--no-validate-parts", action="store_true", default=False,
        help="Skip parts validation",
    )
    prod_p.add_argument(
        "--web-check", action="store_true", default=False,
        help="Check live LCSC stock (requires internet)",
    )
    prod_p.add_argument(
        "--no-web-check", action="store_true", default=False,
        help="Skip live LCSC stock check",
    )
    prod_p.add_argument(
        "--auto-replace", action="store_true", default=False,
        help="Auto-apply replacement parts from ComponentDB",
    )
    prod_p.add_argument(
        "--commit", action="store_true", default=False,
        help="Git commit production artifacts after generation",
    )


def _add_pipeline_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'pipeline' subcommand."""
    pipe_p = sub.add_parser("pipeline", help="Run full pipeline end-to-end")
    pipe_p.add_argument("--requirements", "-r", required=True, help="Requirements JSON")
    pipe_p.add_argument("--output", "-o", required=True, help="Output directory")
    pipe_p.add_argument("--name", "-n", default="project", help="Project name")
    pipe_p.add_argument(
        "--live", action="store_true", default=False,
        help="Connect to running KiCad via IPC for zone fill and board sync",
    )


def _add_enrich_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'enrich' subcommand."""
    enrich_p = sub.add_parser(
        "enrich", help="Enrich existing .kicad_pcb with 3D models and layer flips",
    )
    enrich_p.add_argument("--pcb", "-p", required=True, help="Input .kicad_pcb file")
    enrich_p.add_argument("--output", "-o", default=None, help="Output path (default: overwrite)")
    enrich_p.add_argument(
        "--flip-to-bcu", action="append", default=[], metavar="REF",
        help="Ref(s) to move to B.Cu (repeatable)",
    )
    enrich_p.add_argument(
        "--no-3d-models", action="store_true", default=False,
        help="Skip 3D model injection",
    )
    enrich_p.add_argument(
        "--model-var", default="${KICAD10_3DMODEL_DIR}",
        help="3D model env var (default: ${KICAD10_3DMODEL_DIR})",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="kicad-pipeline",
        description="AI-assisted KiCad EDA pipeline: requirements to production files.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = False

    _add_requirements_subparser(subparsers)
    _add_schematic_subparser(subparsers)
    _add_pcb_subparser(subparsers)
    _add_route_subparser(subparsers)
    _add_validate_subparser(subparsers)
    _add_produce_subparser(subparsers)
    _add_pipeline_subparser(subparsers)
    _add_enrich_subparser(subparsers)

    from kicad_pipeline.cli.project_cmd import add_project_subparser
    add_project_subparser(subparsers)

    from kicad_pipeline.cli.agents_cmd import add_agents_subparser
    add_agents_subparser(subparsers)

    from kicad_pipeline.cli.suggestions_cmd import register_subcommand as _reg_suggestions
    _reg_suggestions(subparsers)

    return parser


def _try_enrich_parts(req: object) -> object:
    """Try to enrich requirements with JLCPCB parts. Returns req as-is on failure."""
    try:
        from kicad_pipeline.parts.jlcpcb_db import JLCPCBPartsDB
        from kicad_pipeline.parts.selector import enrich_requirements_with_parts

        with JLCPCBPartsDB() as db:
            req, suggestions = enrich_requirements_with_parts(req, db=db)  # type: ignore[arg-type]
            enriched = sum(1 for s in suggestions if s.preferred is not None)
            if enriched > 0:
                print(f"  Parts enrichment: {enriched} component(s) matched to JLCPCB parts")
    except Exception:
        pass  # Silently fall back — DB may not be installed
    return req


def _try_ipc_connect(args: argparse.Namespace) -> object | None:
    """Try to connect to KiCad via IPC.  Warns and returns None on failure."""
    try:
        from kicad_pipeline.ipc.connection import connect

        conn = connect()
        print(f"Connected to KiCad IPC ({conn.info.kicad_version})")
        return conn
    except Exception as exc:
        print(f"WARNING: KiCad IPC unavailable ({exc}), using file-based workflow",
              file=sys.stderr)
        return None


def _dispatch_project(args: argparse.Namespace) -> int:
    """Dispatch to the 'project' subcommand handler."""
    from kicad_pipeline.cli.project_cmd import dispatch_project
    return dispatch_project(args)


def _dispatch_agents(args: argparse.Namespace) -> int:
    """Dispatch to the 'agents' subcommand handler."""
    from kicad_pipeline.cli.agents_cmd import dispatch_agents
    return dispatch_agents(args)


def _dispatch_suggestions(args: argparse.Namespace) -> int:
    """Dispatch to the 'suggestions' subcommand handler."""
    from kicad_pipeline.cli.suggestions_cmd import _run_suggestions
    return _run_suggestions(args)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    dispatch: dict[str, object] = {
        "requirements": _cmd_requirements,
        "schematic": _cmd_schematic,
        "pcb": _cmd_pcb,
        "route": _cmd_route,
        "validate": _cmd_validate,
        "produce": _cmd_produce,
        "pipeline": _cmd_pipeline,
        "enrich": _cmd_enrich,
        "project": _dispatch_project,
        "agents": _dispatch_agents,
        "suggestions": _dispatch_suggestions,
    }

    handler = dispatch.get(args.command)
    if handler is not None:
        return handler(args)  # type: ignore[operator]

    parser.print_help()
    return 0


def _cmd_requirements(args: argparse.Namespace) -> int:
    """Handle 'requirements' subcommand."""
    from pathlib import Path

    from kicad_pipeline.requirements.decomposer import load_requirements, save_requirements

    try:
        req = load_requirements(Path(args.input))
        print(f"Loaded requirements: {req.project.name}")
        if args.validate:
            print("Requirements are valid.")
            return 0
        if args.output:
            save_requirements(req, Path(args.output))
            print(f"Saved to {args.output}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _cmd_schematic(args: argparse.Namespace) -> int:
    """Handle 'schematic' subcommand."""
    from pathlib import Path

    from kicad_pipeline.requirements.decomposer import load_requirements
    from kicad_pipeline.schematic.builder import (
        build_project_schematics,
        write_hierarchical_schematic,
        write_schematic,
    )

    try:
        req = load_requirements(Path(args.requirements))
        req = _try_enrich_parts(req)  # type: ignore[assignment]
        hierarchical = False if getattr(args, "flat", False) else None
        schematics = build_project_schematics(req, hierarchical=hierarchical)

        if len(schematics) == 1:
            # Single flat schematic
            sch = next(iter(schematics.values()))
            write_schematic(sch, args.output)
            print(f"Schematic written to {args.output}")
        else:
            # Hierarchical: write to directory
            out_dir = Path(args.output)
            if out_dir.suffix == ".kicad_sch":
                out_dir = out_dir.parent
            written = write_hierarchical_schematic(
                schematics, out_dir, req.project.name,
            )
            print(f"Hierarchical schematic written ({len(written)} files) to {out_dir}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _cmd_pcb(args: argparse.Namespace) -> int:
    """Handle 'pcb' subcommand."""
    from pathlib import Path

    from kicad_pipeline.pcb.builder import build_pcb, write_pcb
    from kicad_pipeline.requirements.decomposer import load_requirements

    try:
        req = load_requirements(Path(args.requirements))
        req = _try_enrich_parts(req)  # type: ignore[assignment]
        design = build_pcb(req)

        ipc_conn = _try_ipc_connect(args) if getattr(args, "live", False) else None
        try:
            write_pcb(design, args.output, ipc_connection=ipc_conn)
        finally:
            if ipc_conn is not None:
                ipc_conn.close()

        print(f"PCB written to {args.output}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _cmd_route(args: argparse.Namespace) -> int:
    """Handle 'route' subcommand. Stub -- returns success."""
    print(f"Routing {args.pcb} -> {args.output} (stub)")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handle 'validate' subcommand. Stub."""
    print(f"Validating {args.pcb} (stub)")
    return 0


def _produce_validate_parts(
    args: argparse.Namespace,
    bom_rows: object,
) -> tuple[str, str, object, object]:
    """Run parts validation and optional auto-replace.

    Returns:
        (validation_text, validation_json, updated_pcb_or_None, updated_bom_or_None).
    """
    from kicad_pipeline.production.parts_validator import (
        report_to_json,
        report_to_text,
        validate_bom_parts,
    )
    from kicad_pipeline.requirements.component_db import ComponentDB

    web_check = args.web_check and not args.no_web_check
    print("[2/4] Validating parts availability...")
    db = ComponentDB()
    report = validate_bom_parts(
        bom_rows, db=db, check_web_stock=web_check, project_name=args.name,
    )
    validation_text = report_to_text(report)
    validation_json = report_to_json(report)
    print(report.summary_text)

    updated_pcb = None
    updated_bom = None
    if args.auto_replace and not report.all_parts_available:
        from kicad_pipeline.production.part_replacer import (
            apply_replacements,
            replacement_map_from_report,
        )

        repl_map = replacement_map_from_report(report)
        if repl_map:
            print(f"  Applying {len(repl_map)} replacement(s)...")
            updated_pcb = apply_replacements  # Return the callable
            updated_bom = repl_map

    return validation_text, validation_json, updated_pcb, updated_bom


def _produce_git_commit(out_path: object, project_name: str) -> None:
    """Git-add and commit production artifacts."""
    import subprocess

    print("Committing production artifacts...")
    subprocess.run(
        ["git", "add", str(out_path)], check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m",
         f"release(production): generate {project_name} manufacturing artifacts"],
        check=True, capture_output=True,
    )
    print("Git commit created.")


def _cmd_produce(args: argparse.Namespace) -> int:
    """Handle 'produce' subcommand — generate production artifacts."""
    from pathlib import Path

    from kicad_pipeline.production.bom import generate_bom
    from kicad_pipeline.production.packager import (
        build_production_package,
        write_production_package,
    )

    try:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)

        if not args.requirements:
            print("ERROR: --requirements is required for produce command", file=sys.stderr)
            return 1

        from kicad_pipeline.requirements.decomposer import load_requirements

        requirements = load_requirements(Path(args.requirements))
        requirements = _try_enrich_parts(requirements)  # type: ignore[assignment]

        from kicad_pipeline.pcb.builder import build_pcb

        pcb = build_pcb(requirements)

        print(f"[1/4] Generating BOM for {args.name}...")
        bom_rows = generate_bom(pcb, requirements)

        validate = args.validate_parts and not args.no_validate_parts
        validation_text = ""
        validation_json = ""

        if validate:
            validation_text, validation_json, apply_fn, repl_map = (
                _produce_validate_parts(args, bom_rows)
            )
            if apply_fn is not None and repl_map is not None:
                pcb = apply_fn(pcb, repl_map)
                bom_rows = generate_bom(pcb, requirements)
        else:
            print("[2/4] Skipping parts validation")

        print("[3/4] Building production package...")
        pkg = build_production_package(pcb, args.name, requirements)

        if validation_text:
            from dataclasses import replace as _replace

            pkg = _replace(
                pkg,
                validation_report_text=validation_text,
                validation_report_json=validation_json,
            )

        print("[4/4] Writing output files...")
        write_production_package(pkg, str(out))

        if args.commit:
            _produce_git_commit(out, args.name)

        print(f"Production artifacts written to {out}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _cmd_enrich(args: argparse.Namespace) -> int:
    """Handle 'enrich' subcommand — post-process existing .kicad_pcb."""
    from kicad_pipeline.pcb.enrich import enrich_pcb_file

    try:
        flip_refs = tuple(args.flip_to_bcu) if args.flip_to_bcu else ()
        add_models = not args.no_3d_models
        enrich_pcb_file(
            pcb_path=args.pcb,
            output_path=args.output,
            flip_refs=flip_refs,
            add_3d_models=add_models,
            model_var=args.model_var,
        )
        out = args.output or args.pcb
        print(f"Enriched PCB written to {out}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """Handle full 'pipeline' subcommand."""
    from pathlib import Path

    from kicad_pipeline.pcb.builder import build_pcb, write_pcb
    from kicad_pipeline.production.packager import (
        build_production_package,
        write_production_package,
    )
    from kicad_pipeline.requirements.decomposer import load_requirements
    from kicad_pipeline.schematic.builder import (
        build_project_schematics,
        write_hierarchical_schematic,
        write_schematic,
    )

    try:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)

        print(f"[1/4] Loading requirements from {args.requirements}...")
        req = load_requirements(Path(args.requirements))
        req = _try_enrich_parts(req)  # type: ignore[assignment]

        print("[2/4] Generating schematic...")
        schematics = build_project_schematics(req)
        if len(schematics) == 1:
            sch = next(iter(schematics.values()))
            sch_path = out / f"{args.name}.kicad_sch"
            write_schematic(sch, str(sch_path))
        else:
            write_hierarchical_schematic(schematics, out, args.name)
            print(f"  Hierarchical: {len(schematics)} sheets")

        print("[3/4] Generating PCB...")
        design = build_pcb(req)
        pcb_path = out / f"{args.name}.kicad_pcb"

        ipc_conn = _try_ipc_connect(args) if getattr(args, "live", False) else None
        try:
            write_pcb(design, str(pcb_path), ipc_connection=ipc_conn)
        finally:
            if ipc_conn is not None:
                ipc_conn.close()

        print("[3.5/4] Generating project file...")
        from kicad_pipeline.project_file import write_project_file

        write_project_file(args.name, out)

        print("[4/4] Generating production artifacts...")
        pkg = build_production_package(design, args.name, req)
        write_production_package(pkg, str(out / "production"))

        print(f"Pipeline complete. Output in {out}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
