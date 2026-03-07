#!/usr/bin/env python3
"""kicad-ai-pipeline CLI entry point."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="kicad-pipeline",
        description="AI-assisted KiCad EDA pipeline: requirements to production files.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = False

    # requirements subcommand
    req_p = subparsers.add_parser("requirements", help="Manage project requirements")
    req_p.add_argument("--input", "-i", required=True, help="Input requirements JSON")
    req_p.add_argument("--output", "-o", help="Output requirements JSON")
    req_p.add_argument("--validate", action="store_true", help="Validate requirements only")

    # schematic subcommand
    sch_p = subparsers.add_parser("schematic", help="Generate KiCad schematic")
    sch_p.add_argument("--requirements", "-r", required=True, help="Requirements JSON")
    sch_p.add_argument("--output", "-o", required=True, help="Output .kicad_sch file")

    # pcb subcommand
    pcb_p = subparsers.add_parser("pcb", help="Generate KiCad PCB")
    pcb_p.add_argument("--requirements", "-r", required=True, help="Requirements JSON")
    pcb_p.add_argument("--output", "-o", required=True, help="Output .kicad_pcb file")
    pcb_p.add_argument(
        "--live", action="store_true", default=False,
        help="Connect to running KiCad via IPC for zone fill and board sync",
    )

    # route subcommand
    route_p = subparsers.add_parser("route", help="Autoroute PCB")
    route_p.add_argument("--pcb", "-p", required=True, help="Input .kicad_pcb file")
    route_p.add_argument("--output", "-o", required=True, help="Output .kicad_pcb file")
    route_p.add_argument("--freerouting", action="store_true", help="Use FreeRouting")

    # validate subcommand
    val_p = subparsers.add_parser("validate", help="Validate PCB design")
    val_p.add_argument("--pcb", "-p", required=True, help="PCB JSON or kicad_pcb file")
    val_p.add_argument("--report", "-r", help="Output report JSON")

    # produce subcommand
    prod_p = subparsers.add_parser("produce", help="Generate production artifacts")
    prod_p.add_argument("--pcb", "-p", required=True, help="PCB JSON or kicad_pcb file")
    prod_p.add_argument("--output", "-o", required=True, help="Output directory")
    prod_p.add_argument("--name", "-n", default="project", help="Project name")
    prod_p.add_argument(
        "--requirements", "-r", default=None, help="Requirements JSON for BOM enrichment"
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

    # pipeline subcommand (full end-to-end)
    pipe_p = subparsers.add_parser("pipeline", help="Run full pipeline end-to-end")
    pipe_p.add_argument("--requirements", "-r", required=True, help="Requirements JSON")
    pipe_p.add_argument("--output", "-o", required=True, help="Output directory")
    pipe_p.add_argument("--name", "-n", default="project", help="Project name")
    pipe_p.add_argument(
        "--live", action="store_true", default=False,
        help="Connect to running KiCad via IPC for zone fill and board sync",
    )

    # enrich subcommand (post-process existing PCB)
    enrich_p = subparsers.add_parser(
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
        "--model-var", default="${KICAD9_3DMODEL_DIR}",
        help="3D model env var (default: ${KICAD9_3DMODEL_DIR})",
    )

    # project subcommand (orchestrated workflow)
    from kicad_pipeline.cli.project_cmd import add_project_subparser

    add_project_subparser(subparsers)

    return parser


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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handlers
    if args.command == "requirements":
        return _cmd_requirements(args)
    if args.command == "schematic":
        return _cmd_schematic(args)
    if args.command == "pcb":
        return _cmd_pcb(args)
    if args.command == "route":
        return _cmd_route(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "produce":
        return _cmd_produce(args)
    if args.command == "pipeline":
        return _cmd_pipeline(args)
    if args.command == "enrich":
        return _cmd_enrich(args)
    if args.command == "project":
        from kicad_pipeline.cli.project_cmd import dispatch_project

        return dispatch_project(args)

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
    from kicad_pipeline.schematic.builder import build_schematic, write_schematic

    try:
        req = load_requirements(Path(args.requirements))
        sch = build_schematic(req)
        write_schematic(sch, args.output)
        print(f"Schematic written to {args.output}")
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

        # Build PCB from requirements
        requirements = None
        if args.requirements:
            from kicad_pipeline.requirements.decomposer import load_requirements

            requirements = load_requirements(Path(args.requirements))

        # Build PCB from requirements (produce command needs the PCBDesign object)
        from kicad_pipeline.pcb.builder import build_pcb

        if requirements is not None:
            pcb = build_pcb(requirements)
        else:
            print("ERROR: --requirements is required for produce command", file=sys.stderr)
            return 1

        print(f"[1/4] Generating BOM for {args.name}...")
        bom_rows = generate_bom(pcb, requirements)

        # Parts validation
        validate = args.validate_parts and not args.no_validate_parts
        web_check = args.web_check and not args.no_web_check
        validation_text = ""
        validation_json = ""

        if validate:
            from kicad_pipeline.production.parts_validator import (
                report_to_json,
                report_to_text,
                validate_bom_parts,
            )
            from kicad_pipeline.requirements.component_db import ComponentDB

            print("[2/4] Validating parts availability...")
            db = ComponentDB()
            report = validate_bom_parts(
                bom_rows, db=db, check_web_stock=web_check, project_name=args.name,
            )
            validation_text = report_to_text(report)
            validation_json = report_to_json(report)
            print(report.summary_text)

            # Auto-replace if requested
            if args.auto_replace and not report.all_parts_available:
                from kicad_pipeline.production.part_replacer import (
                    apply_replacements,
                    replacement_map_from_report,
                )

                repl_map = replacement_map_from_report(report)
                if repl_map:
                    print(f"  Applying {len(repl_map)} replacement(s)...")
                    pcb = apply_replacements(pcb, repl_map)
                    bom_rows = generate_bom(pcb, requirements)
        else:
            print("[2/4] Skipping parts validation")

        print("[3/4] Building production package...")
        pkg = build_production_package(pcb, args.name, requirements)

        # Inject validation reports if available
        if validation_text:
            from dataclasses import replace as _replace

            pkg = _replace(
                pkg,
                validation_report_text=validation_text,
                validation_report_json=validation_json,
            )

        print("[4/4] Writing output files...")
        write_production_package(pkg, str(out))

        # Git commit if requested
        if args.commit:
            import subprocess

            print("Committing production artifacts...")
            subprocess.run(
                ["git", "add", str(out)], check=True, capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m",
                 f"release(production): generate {args.name} manufacturing artifacts"],
                check=True, capture_output=True,
            )
            print("Git commit created.")

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
    from kicad_pipeline.schematic.builder import build_schematic, write_schematic

    try:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)

        print(f"[1/4] Loading requirements from {args.requirements}...")
        req = load_requirements(Path(args.requirements))

        print("[2/4] Generating schematic...")
        sch = build_schematic(req)
        sch_path = out / f"{args.name}.kicad_sch"
        write_schematic(sch, str(sch_path))

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
