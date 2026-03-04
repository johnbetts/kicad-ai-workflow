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

    # pipeline subcommand (full end-to-end)
    pipe_p = subparsers.add_parser("pipeline", help="Run full pipeline end-to-end")
    pipe_p.add_argument("--requirements", "-r", required=True, help="Requirements JSON")
    pipe_p.add_argument("--output", "-o", required=True, help="Output directory")
    pipe_p.add_argument("--name", "-n", default="project", help="Project name")

    # project subcommand (orchestrated workflow)
    from kicad_pipeline.cli.project_cmd import add_project_subparser

    add_project_subparser(subparsers)

    return parser


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
        write_pcb(design, args.output)
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
    """Handle 'produce' subcommand."""
    print(f"Producing artifacts to {args.output} (stub)")
    return 0


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
        write_pcb(design, str(pcb_path))

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
