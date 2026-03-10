"""CLI subcommand for viewing and managing improvement suggestions."""

from __future__ import annotations

import argparse  # noqa: TC003 — used at runtime for parser construction
import sys
from pathlib import Path


def register_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the 'suggestions' subcommand."""
    parser = subparsers.add_parser(
        "suggestions",
        help="View and manage pipeline improvement suggestions",
    )
    parser.add_argument(
        "--status",
        choices=["proposed", "accepted", "implemented", "verified", "all"],
        default="proposed",
        help="Filter by status (default: proposed)",
    )
    parser.add_argument(
        "--category",
        choices=[
            "performance",
            "quality",
            "ux",
            "architecture",
            "testing",
            "documentation",
            "all",
        ],
        default="all",
        help="Filter by category",
    )
    parser.add_argument(
        "--accept",
        metavar="SUGGESTION_ID",
        help="Accept a suggestion by ID",
    )
    parser.add_argument(
        "--implement",
        metavar="SUGGESTION_ID",
        help="Mark a suggestion as implemented",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to suggestions.jsonl (default: ~/.claude/kicad-agents/suggestions.jsonl)",
    )
    parser.set_defaults(func=_run_suggestions)


def _run_suggestions(args: argparse.Namespace) -> int:
    """Execute the suggestions subcommand."""
    from kicad_pipeline.agents.suggestions import SuggestionReporter

    reporter = SuggestionReporter("cli", suggestions_path=args.path)

    if args.accept:
        if reporter.update_status(args.accept, "accepted"):
            print(f"Accepted: {args.accept}")
        else:
            print(f"Not found: {args.accept}", file=sys.stderr)
            return 1
        return 0

    if args.implement:
        if reporter.update_status(args.implement, "implemented"):
            print(f"Implemented: {args.implement}")
        else:
            print(f"Not found: {args.implement}", file=sys.stderr)
            return 1
        return 0

    # List suggestions
    suggestions = reporter.load_all()
    if args.status != "all":
        suggestions = tuple(s for s in suggestions if s.status == args.status)
    if args.category != "all":
        suggestions = tuple(s for s in suggestions if s.category == args.category)

    if not suggestions:
        print("No suggestions found.")
        return 0

    for s in suggestions:
        priority_icon = {"critical": "!!", "high": "!", "medium": "-", "low": "."}
        icon = priority_icon.get(s.priority, "-")
        print(f"[{icon}] {s.suggestion_id} ({s.category}/{s.priority})")
        print(f"    {s.title}")
        if s.affected_module:
            print(f"    Module: {s.affected_module}")
        print(f"    Status: {s.status} | Effort: {s.effort}")
        print()

    return 0
