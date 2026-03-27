#!/usr/bin/env python3
"""Process Runner — mandatory-step orchestration framework.

Drives Claude agents through a sequence of verified steps. Each step:
1. Launches an agent with a specific, narrow prompt
2. Verifies the agent's output (files exist, content correct, etc.)
3. If verification fails: re-launches with error feedback
4. Only proceeds to next step when current step passes

Reusable for any mandatory process — PCB review, production, testing, etc.

Usage:
  # Define a process as a list of Steps, then run it:
  python scripts/process_runner.py pcb-review mcu
  python scripts/process_runner.py pcb-review all
  python scripts/process_runner.py pcb-review relay --max-retries 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")

TRAINING_BOARDS: dict[str, str] = {
    "mcu": "output/train_mcu_core/train_mcu_core.kicad_pcb",
    "relay": "output/train_relay/train_relay.kicad_pcb",
    "power": "output/train_power/train_power.kicad_pcb",
    "analog": "output/train_analog_input/train_analog_input.kicad_pcb",
    "ethernet": "output/train_ethernet/train_ethernet.kicad_pcb",
}


# ---------------------------------------------------------------------------
# Core framework
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    """Result of a verification check."""
    passed: bool
    message: str
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class Step:
    """One mandatory step in a process."""
    name: str
    description: str
    agent_prompt: str  # Template with {board}, {board_dir}, {board_name}, {errors}
    verify: Callable[[str], VerifyResult]  # Takes board_path, returns VerifyResult
    max_retries: int = 3
    timeout: int = 300  # seconds for agent
    retry_prompt: str = ""  # Additional prompt on retry, template with {errors}


@dataclass
class StepResult:
    """Result of running one step."""
    step_name: str
    passed: bool
    attempts: int
    duration_seconds: float
    final_message: str
    errors: list[str] = field(default_factory=list)


@dataclass
class ProcessResult:
    """Result of running an entire process."""
    process_name: str
    board: str
    passed: bool
    steps: list[StepResult] = field(default_factory=list)
    started: str = ""
    finished: str = ""

    def summary(self) -> str:
        lines = [f"Process: {self.process_name} on {Path(self.board).stem}"]
        lines.append(f"Result: {'PASS' if self.passed else 'FAIL'}")
        for sr in self.steps:
            status = "PASS" if sr.passed else "FAIL"
            lines.append(f"  [{status}] {sr.step_name} ({sr.attempts} attempts, {sr.duration_seconds:.0f}s)")
            if sr.errors:
                for e in sr.errors[:3]:
                    lines.append(f"         {e}")
        return "\n".join(lines)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def run_claude(prompt: str, timeout: int = 300) -> str:
    """Run Claude in non-interactive mode."""
    try:
        result = subprocess.run(
            [CLAUDE_BIN, "-p", "--output-format", "text", prompt],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.getcwd(),
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except FileNotFoundError:
        log(f"ERROR: Claude CLI not found at {CLAUDE_BIN}")
        sys.exit(1)


def run_step(step: Step, board_path: str) -> StepResult:
    """Run a single step with retries and verification."""
    board_dir = str(Path(board_path).parent)
    board_name = Path(board_path).stem
    start = time.time()
    errors: list[str] = []

    for attempt in range(1, step.max_retries + 1):
        log(f"  Step '{step.name}' — attempt {attempt}/{step.max_retries}")

        # Build prompt
        error_context = "\n".join(errors) if errors else "First attempt"
        prompt = step.agent_prompt.format(
            board=board_path,
            board_dir=board_dir,
            board_name=board_name,
            errors=error_context,
        )

        # On retry, prepend retry prompt with error feedback
        if attempt > 1 and step.retry_prompt:
            retry_msg = step.retry_prompt.format(errors=error_context)
            prompt = retry_msg + "\n\n" + prompt

        # Run agent
        log(f"    Running agent...")
        response = run_claude(prompt, timeout=step.timeout)

        if response == "[TIMEOUT]":
            errors.append(f"Attempt {attempt}: agent timed out after {step.timeout}s")
            log(f"    TIMEOUT after {step.timeout}s")
            continue

        # Verify
        log(f"    Verifying...")
        result = step.verify(board_path)

        if result.passed:
            log(f"    VERIFIED: {result.message}")
            return StepResult(
                step_name=step.name, passed=True, attempts=attempt,
                duration_seconds=time.time() - start,
                final_message=result.message,
            )

        # Verification failed — collect errors for next retry
        errors = result.errors + result.missing
        log(f"    FAILED: {result.message}")
        for e in errors[:5]:
            log(f"      - {e}")

    # All retries exhausted
    return StepResult(
        step_name=step.name, passed=False, attempts=step.max_retries,
        duration_seconds=time.time() - start,
        final_message=f"Failed after {step.max_retries} attempts",
        errors=errors,
    )


def run_process(name: str, steps: list[Step], board_path: str) -> ProcessResult:
    """Run a complete process — all steps in sequence, stop on first failure."""
    log(f"Process '{name}' starting on {Path(board_path).stem}")
    result = ProcessResult(
        process_name=name, board=board_path,
        passed=False, started=datetime.now().isoformat(),
    )

    for step in steps:
        log(f"\n{'─' * 50}")
        log(f"STEP: {step.name} — {step.description}")
        log(f"{'─' * 50}")

        sr = run_step(step, board_path)
        result.steps.append(sr)

        if not sr.passed:
            log(f"\nSTEP '{step.name}' FAILED — process halted.")
            result.finished = datetime.now().isoformat()
            return result

    result.passed = True
    result.finished = datetime.now().isoformat()
    log(f"\nProcess '{name}' COMPLETE — all steps passed.")
    return result


# ---------------------------------------------------------------------------
# PCB Review process definition
# ---------------------------------------------------------------------------

import re


def verify_renders(board_path: str) -> VerifyResult:
    """Verify all required renders exist."""
    board = Path(board_path)
    out_dir = board.parent
    name = board.stem
    missing: list[str] = []

    # Board-level renders
    for view in ["2d_top", "3d_top", "3d_iso", "3d_isoback", "3d_hires_top"]:
        f = out_dir / f"{name}_{view}.png"
        if not f.exists():
            missing.append(f"{name}_{view}.png")

    # Per-component crops
    pcb_text = board.read_text()
    refs: list[str] = []
    for m in re.finditer(
        r'\(footprint\s+"[^"]*".*?\(property\s+"Reference"\s+"([^"]+)"',
        pcb_text, re.DOTALL,
    ):
        refs.append(m.group(1))

    comp_missing = 0
    for ref in refs:
        f = out_dir / f"{name}_3d_comp_{ref}.png"
        if not f.exists():
            comp_missing += 1
            if comp_missing <= 5:
                missing.append(f"{name}_3d_comp_{ref}.png")

    if comp_missing > 5:
        missing.append(f"...and {comp_missing - 5} more component crops")

    if not missing:
        total = 5 + len(refs)
        return VerifyResult(passed=True, message=f"All {total} renders present")

    return VerifyResult(
        passed=False,
        message=f"{len(missing)} renders missing",
        missing=missing,
        errors=[f"Missing renders. Run: python scripts/render-board.py {board_path}"],
    )


def verify_review_json(review_type: str, board_path: str) -> VerifyResult:
    """Verify a review JSON was written with actual issues analyzed."""
    board = Path(board_path)
    out_dir = board.parent
    name = board.stem
    review_file = out_dir / f"{name}_{review_type}_review.json"

    if not review_file.exists():
        return VerifyResult(
            passed=False,
            message=f"{review_type} review file not found",
            missing=[str(review_file)],
            errors=[f"Agent must write review to {review_file}"],
        )

    try:
        data = json.loads(review_file.read_text())
    except json.JSONDecodeError:
        return VerifyResult(
            passed=False, message="Review file is not valid JSON",
            errors=["Rewrite as valid JSON"],
        )

    if "issues" not in data and "components" not in data:
        return VerifyResult(
            passed=False, message="Review JSON missing 'issues' or 'components' key",
            errors=["Review must contain structured findings"],
        )

    return VerifyResult(passed=True, message=f"{review_type} review written ({len(data.get('issues', data.get('components', [])))} items)")


def verify_component_review(board_path: str) -> VerifyResult:
    """Verify per-component 3D review was done with findings for each component."""
    board = Path(board_path)
    out_dir = board.parent
    name = board.stem
    review_file = out_dir / f"{name}_3d_component_review.json"

    if not review_file.exists():
        return VerifyResult(
            passed=False,
            message="Component review file not found",
            missing=[str(review_file)],
            errors=[f"Agent must write component-by-component review to {review_file}"],
        )

    try:
        data = json.loads(review_file.read_text())
    except json.JSONDecodeError:
        return VerifyResult(passed=False, message="Not valid JSON", errors=["Rewrite as valid JSON"])

    components = data.get("components", [])
    if not components:
        return VerifyResult(
            passed=False, message="No components reviewed",
            errors=["Review must list EVERY component with PASS/FAIL status"],
        )

    # Check that the number of reviewed components is close to actual component count
    pcb_text = board.read_text()
    actual_refs = set()
    for m in re.finditer(
        r'\(property\s+"Reference"\s+"([^"]+)"', pcb_text,
    ):
        actual_refs.add(m.group(1))

    reviewed_refs = {c.get("ref", "") for c in components}
    missing_refs = actual_refs - reviewed_refs
    coverage = len(reviewed_refs) / max(len(actual_refs), 1)

    if coverage < 0.8:
        return VerifyResult(
            passed=False,
            message=f"Only {len(reviewed_refs)}/{len(actual_refs)} components reviewed ({coverage:.0%})",
            errors=[
                f"Must review ALL components. Missing: {', '.join(sorted(missing_refs)[:10])}",
                "Read EVERY 3d_comp_<REF>.png and report PASS/FAIL for each",
            ],
        )

    return VerifyResult(
        passed=True,
        message=f"{len(reviewed_refs)}/{len(actual_refs)} components reviewed ({coverage:.0%})",
    )


# Process steps
PCB_REVIEW_STEPS: list[Step] = [
    Step(
        name="render",
        description="Generate all board-level + per-component renders",
        agent_prompt=(
            "Run the render script to generate ALL images for this board:\n"
            "  python scripts/render-board.py {board}\n\n"
            "This generates board-level 2D/3D views AND per-component 3D crops.\n"
            "After running, verify with: python scripts/verify-renders.py {board}\n"
            "Do NOT proceed until verify passes."
        ),
        verify=verify_renders,
        max_retries=3,
        timeout=600,
        retry_prompt=(
            "PREVIOUS ATTEMPT FAILED. Missing renders:\n{errors}\n\n"
            "Re-run: python scripts/render-board.py {board}\n"
            "Then verify: python scripts/verify-renders.py {board}"
        ),
    ),
    Step(
        name="fab-review",
        description="Fabricator persona reviews all renders",
        agent_prompt=(
            "You are a PCB fabricator. Review the board at {board}.\n\n"
            "READ these images with the Read tool (every single one):\n"
            "  {board_dir}/{board_name}_2d_top.png\n"
            "  {board_dir}/{board_name}_3d_top.png\n"
            "  {board_dir}/{board_name}_3d_iso.png\n"
            "  {board_dir}/{board_name}_3d_isoback.png\n\n"
            "Check: board bounds, courtyard overlaps, connector edge distance,\n"
            "component spacing, silkscreen clarity, 3D body overlaps, assembly feasibility.\n\n"
            "Write your review as JSON to: {board_dir}/{board_name}_fab_review.json\n"
            'Format: {{"issues": [{{"ref": "R1", "type": "...", "severity": "CRITICAL|MAJOR|MINOR", '
            '"description": "...", "fix": "..."}}], "pass": true/false}}'
        ),
        verify=lambda bp: verify_review_json("fab", bp),
        max_retries=2,
        timeout=300,
        retry_prompt=(
            "PREVIOUS ATTEMPT FAILED:\n{errors}\n\n"
            "You MUST write the review JSON file. Read the images, analyze them, "
            "and write the structured JSON output."
        ),
    ),
    Step(
        name="ee-review",
        description="Electrical engineer persona reviews all renders",
        agent_prompt=(
            "You are an electrical engineer. Review the board at {board}.\n\n"
            "READ these images with the Read tool (every single one):\n"
            "  {board_dir}/{board_name}_2d_top.png\n"
            "  {board_dir}/{board_name}_3d_top.png\n"
            "  {board_dir}/{board_name}_3d_iso.png\n"
            "  {board_dir}/{board_name}_3d_isoback.png\n\n"
            "Check: decoupling placement, signal flow, analog/digital separation,\n"
            "power flow direction, ground return paths, crystal loop area, voltage isolation.\n\n"
            "Write your review as JSON to: {board_dir}/{board_name}_ee_review.json\n"
            'Format: {{"issues": [{{"ref": "U1", "type": "...", "severity": "CRITICAL|MAJOR|MINOR", '
            '"description": "...", "fix": "..."}}], "pass": true/false}}'
        ),
        verify=lambda bp: verify_review_json("ee", bp),
        max_retries=2,
        timeout=300,
        retry_prompt=(
            "PREVIOUS ATTEMPT FAILED:\n{errors}\n\n"
            "You MUST write the review JSON file."
        ),
    ),
    Step(
        name="3d-component-verify",
        description="Per-component 3D body-to-pad alignment check",
        agent_prompt=(
            "Verify 3D model alignment for EVERY component on {board}.\n\n"
            "For EVERY component (ICs, connectors, resistors, capacitors, ALL of them):\n"
            "1. Read the per-component crop: {board_dir}/{board_name}_3d_comp_<REF>.png\n"
            "2. Also cross-reference with {board_dir}/{board_name}_2d_top.png for pad positions\n"
            "3. Check: body centered on pads? Correct rotation? Flat on board? Model present?\n\n"
            "List EVERY component with PASS or FAIL status.\n\n"
            "Write results to: {board_dir}/{board_name}_3d_component_review.json\n"
            'Format: {{"components": [{{"ref": "U1", "status": "PASS"}}, '
            '{{"ref": "C1", "status": "FAIL", "issue": "offset 2mm left"}}], "issues_count": N}}'
        ),
        verify=verify_component_review,
        max_retries=3,
        timeout=600,
        retry_prompt=(
            "PREVIOUS ATTEMPT FAILED:\n{errors}\n\n"
            "You MUST review EVERY component. Read each {board_name}_3d_comp_<REF>.png file.\n"
            "Do NOT skip any component. Write the full JSON with all components listed."
        ),
    ),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_boards(selector: str) -> list[str]:
    if selector == "all":
        return [p for p in TRAINING_BOARDS.values() if Path(p).exists()]
    if "," in selector:
        result: list[str] = []
        for part in selector.split(","):
            result.extend(resolve_boards(part.strip()))
        return result
    if selector in TRAINING_BOARDS:
        return [TRAINING_BOARDS[selector]]
    if Path(selector).exists():
        return [selector]
    matches = [k for k in TRAINING_BOARDS if selector in k]
    if len(matches) == 1:
        return [TRAINING_BOARDS[matches[0]]]
    print(f"ERROR: Unknown '{selector}'. Known: {', '.join(TRAINING_BOARDS)}, all")
    sys.exit(1)


PROCESSES: dict[str, list[Step]] = {
    "pcb-review": PCB_REVIEW_STEPS,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process Runner — mandatory-step orchestration with verification gates"
    )
    parser.add_argument("process", choices=list(PROCESSES.keys()),
                        help="Process to run")
    parser.add_argument("board", nargs="?", default=None,
                        help="Board selector (mcu/relay/power/analog/ethernet/all)")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Override max retries for all steps")
    parser.add_argument("--start-at", type=str, default=None,
                        help="Skip to this step (e.g., 'ee-review')")
    parser.add_argument("--list-steps", action="store_true",
                        help="List steps in the process and exit")
    args = parser.parse_args()

    steps = PROCESSES[args.process]

    if args.list_steps:
        print(f"Process: {args.process}")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s.name} — {s.description} (max {s.max_retries} retries)")
        return

    if not args.board and not args.list_steps:
        parser.error("board selector required")

    if args.max_retries is not None:
        for s in steps:
            s.max_retries = args.max_retries

    if args.start_at:
        idx = next((i for i, s in enumerate(steps) if s.name == args.start_at), None)
        if idx is None:
            print(f"Unknown step '{args.start_at}'. Steps: {[s.name for s in steps]}")
            sys.exit(1)
        steps = steps[idx:]
        log(f"Starting at step '{args.start_at}' (skipping {idx} steps)")

    boards = resolve_boards(args.board)

    results: list[ProcessResult] = []
    for board_path in boards:
        result = run_process(args.process, steps, board_path)
        results.append(result)
        print(f"\n{result.summary()}\n")

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"\n{'=' * 50}")
    print(f"TOTAL: {passed} passed, {failed} failed out of {len(results)} boards")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
