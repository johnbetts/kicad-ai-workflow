#!/usr/bin/env python3
"""Verification pipeline — Python-driven, per-component, dual-persona.

The script IS the process. It decides what runs, validates output, tracks
known bugs, and enforces per-component granularity.

Usage::

    # Full pipeline (programmatic + AI visual + known bugs)
    python scripts/verify_pipeline.py output/train_relay/train_relay.kicad_pcb

    # Programmatic only (fast, no AI — suitable for CI)
    python scripts/verify_pipeline.py output/train_relay/train_relay.kicad_pcb --programmatic-only

    # Single persona
    python scripts/verify_pipeline.py output/train_relay/train_relay.kicad_pcb --persona fab

    # Known bugs regression check only
    python scripts/verify_pipeline.py output/train_relay/train_relay.kicad_pcb --known-bugs-only

    # Component subset
    python scripts/verify_pipeline.py output/train_relay/train_relay.kicad_pcb \
        --components R_0805,C_0402
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from kicad_pipeline.verification.checklist import VerificationChecklist
from kicad_pipeline.verification.orchestrator import VerificationOrchestrator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PCB verification pipeline — orchestrates checks with evidence",
    )
    parser.add_argument(
        "board",
        type=Path,
        help="Path to .kicad_pcb file",
    )
    parser.add_argument(
        "--programmatic-only",
        action="store_true",
        help="Run only programmatic checks (fast, no AI)",
    )
    parser.add_argument(
        "--persona",
        choices=("fab", "ee"),
        default=None,
        help="Run only one persona's checks",
    )
    parser.add_argument(
        "--known-bugs-only",
        action="store_true",
        help="Only check items linked to known bugs",
    )
    parser.add_argument(
        "--components",
        type=str,
        default=None,
        help="Comma-separated component IDs to check (default: all)",
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=None,
        help="Path to verification_checklist.json (default: data/verification_checklist.json)",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="Directory for evidence artifacts",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write JSON report to file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Validate board path
    board_path = args.board.resolve()
    if not board_path.exists():
        print(f"ERROR: Board file not found: {board_path}", file=sys.stderr)
        return 1

    # Load checklist
    checklist = VerificationChecklist(path=args.checklist)
    bug_count = len(checklist.open_bugs)
    pat_count = len(checklist.patterns)
    print(f"Loaded checklist: {bug_count} open bugs, {pat_count} patterns")

    # Parse component list
    component_refs: list[str] | None = None
    if args.components:
        component_refs = [c.strip() for c in args.components.split(",")]
        print(f"Checking components: {component_refs}")

    # Create orchestrator
    orchestrator = VerificationOrchestrator(
        board_path=board_path,
        checklist=checklist,
        evidence_dir=args.evidence_dir,
    )

    # Run verification
    report = orchestrator.run_all(
        programmatic_only=args.programmatic_only,
        persona=args.persona,
        component_refs=component_refs,
        known_bugs_only=args.known_bugs_only,
    )

    # Print summary
    print()
    print("=" * 60)
    print(report.summary())
    print("=" * 60)

    # Write JSON report if requested
    if args.json_output:
        json_data = {
            "board": report.board_path,
            "passed": report.passed,
            "total_checks": report.total_checks,
            "passed_checks": report.passed_checks,
            "failed_checks": report.failed_checks,
            "started": report.started,
            "finished": report.finished,
            "steps": [
                {
                    "name": step.step_name,
                    "passed": step.passed,
                    "duration_secs": step.duration_secs,
                    "checks": [
                        {
                            "item_id": cr.item_id,
                            "passed": cr.passed,
                            "detail": cr.detail,
                            "severity": cr.severity.value,
                            "confidence": cr.confidence,
                        }
                        for cr in step.check_results
                    ],
                }
                for step in report.steps
            ],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(json_data, indent=2), encoding="utf-8",
        )
        print(f"\nJSON report written to: {args.json_output}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
