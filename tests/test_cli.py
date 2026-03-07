"""Tests for the kicad-pipeline CLI argument parser and command dispatch."""

from __future__ import annotations

import argparse

import pytest

from kicad_pipeline.cli.main import build_parser, main


def test_build_parser_returns_parser() -> None:
    """build_parser() returns an ArgumentParser instance."""
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_main_no_args_returns_0() -> None:
    """main([]) prints help and returns 0."""
    result = main([])
    assert result == 0


def test_main_version() -> None:
    """main(['--version']) raises SystemExit (argparse version action)."""
    with pytest.raises(SystemExit):
        main(["--version"])


def test_main_help_exits() -> None:
    """main(['--help']) raises SystemExit (argparse help action)."""
    with pytest.raises(SystemExit):
        main(["--help"])


def test_requirements_subcommand_exists() -> None:
    """'requirements --input x.json' parses without error."""
    parser = build_parser()
    args = parser.parse_args(["requirements", "--input", "x.json"])
    assert args.command == "requirements"
    assert args.input == "x.json"


def test_schematic_subcommand_exists() -> None:
    """'schematic -r x.json -o x.kicad_sch' parses without error."""
    parser = build_parser()
    args = parser.parse_args(["schematic", "-r", "x.json", "-o", "x.kicad_sch"])
    assert args.command == "schematic"
    assert args.requirements == "x.json"
    assert args.output == "x.kicad_sch"


def test_pcb_subcommand_exists() -> None:
    """'pcb -r x.json -o x.kicad_pcb' parses without error."""
    parser = build_parser()
    args = parser.parse_args(["pcb", "-r", "x.json", "-o", "x.kicad_pcb"])
    assert args.command == "pcb"
    assert args.requirements == "x.json"
    assert args.output == "x.kicad_pcb"


def test_route_subcommand_exists() -> None:
    """'route -p x.kicad_pcb -o y.kicad_pcb' parses without error."""
    parser = build_parser()
    args = parser.parse_args(["route", "-p", "x.kicad_pcb", "-o", "y.kicad_pcb"])
    assert args.command == "route"
    assert args.pcb == "x.kicad_pcb"
    assert args.output == "y.kicad_pcb"


def test_validate_subcommand_exists() -> None:
    """'validate -p x.kicad_pcb' parses without error."""
    parser = build_parser()
    args = parser.parse_args(["validate", "-p", "x.kicad_pcb"])
    assert args.command == "validate"
    assert args.pcb == "x.kicad_pcb"


def test_produce_subcommand_exists() -> None:
    """'produce -p x.kicad_pcb -o out/' parses without error."""
    parser = build_parser()
    args = parser.parse_args(["produce", "-p", "x.kicad_pcb", "-o", "out/"])
    assert args.command == "produce"
    assert args.pcb == "x.kicad_pcb"
    assert args.output == "out/"


def test_pipeline_subcommand_exists() -> None:
    """'pipeline -r x.json -o out/' parses without error."""
    parser = build_parser()
    args = parser.parse_args(["pipeline", "-r", "x.json", "-o", "out/"])
    assert args.command == "pipeline"
    assert args.requirements == "x.json"
    assert args.output == "out/"


def test_main_route_stub_returns_0() -> None:
    """main(['route', '-p', 'x.kicad_pcb', '-o', 'y.kicad_pcb']) returns 0 (stub)."""
    result = main(["route", "-p", "x.kicad_pcb", "-o", "y.kicad_pcb"])
    assert result == 0


def test_main_validate_stub_returns_0() -> None:
    """main(['validate', '-p', 'x.kicad_pcb']) returns 0 (stub)."""
    result = main(["validate", "-p", "x.kicad_pcb"])
    assert result == 0


def test_main_produce_requires_requirements() -> None:
    """produce without --requirements returns 1 (needs requirements to build PCB)."""
    result = main(["produce", "-p", "x.kicad_pcb", "-o", "out/"])
    assert result == 1


def test_requirements_validate_flag() -> None:
    """'requirements --input x.json --validate' sets validate=True."""
    parser = build_parser()
    args = parser.parse_args(["requirements", "--input", "x.json", "--validate"])
    assert args.validate is True


def test_requirements_output_flag() -> None:
    """'requirements --input x.json --output y.json' sets output."""
    parser = build_parser()
    args = parser.parse_args(["requirements", "--input", "x.json", "--output", "y.json"])
    assert args.output == "y.json"


def test_pipeline_name_default() -> None:
    """'pipeline -r x.json -o out/' uses default name 'project'."""
    parser = build_parser()
    args = parser.parse_args(["pipeline", "-r", "x.json", "-o", "out/"])
    assert args.name == "project"


def test_pipeline_name_custom() -> None:
    """'pipeline -r x.json -o out/ -n my-board' sets custom name."""
    parser = build_parser()
    args = parser.parse_args(["pipeline", "-r", "x.json", "-o", "out/", "-n", "my-board"])
    assert args.name == "my-board"
