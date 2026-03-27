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
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.evidence.gates import check_gate
from kicad_pipeline.evidence.known_issues import load_known_issues
from kicad_pipeline.evidence.ledger import append_record, load_ledger
from kicad_pipeline.evidence.models import EvidenceKind, EvidenceRecord, Issue, Severity

if TYPE_CHECKING:
    from collections.abc import Callable

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")

# ---------------------------------------------------------------------------
# Dashboard notification (fire-and-forget)
# ---------------------------------------------------------------------------

_dashboard_url: str | None = None


def notify_dashboard(record: EvidenceRecord) -> None:
    """POST an EvidenceRecord to the dashboard. Fire-and-forget."""
    if _dashboard_url is None:
        return
    try:
        data = record.model_dump_json().encode("utf-8")
        req = urllib.request.Request(
            f"{_dashboard_url}/api/evidence",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # Dashboard down must never block the runner


def notify_dashboard_log(message: str, level: str = "info") -> None:
    """POST a log message to the dashboard. Fire-and-forget."""
    if _dashboard_url is None:
        return
    try:
        payload = json.dumps({"message": message, "level": level}).encode("utf-8")
        req = urllib.request.Request(
            f"{_dashboard_url}/api/log",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass

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
            lines.append(
                f"  [{status}] {sr.step_name}"
                f" ({sr.attempts} attempts, {sr.duration_seconds:.0f}s)"
            )
            if sr.errors:
                for e in sr.errors[:3]:
                    lines.append(f"         {e}")
        return "\n".join(lines)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    notify_dashboard_log(msg)


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
        log("    Running agent...")
        response = run_claude(prompt, timeout=step.timeout)

        if response == "[TIMEOUT]":
            errors.append(f"Attempt {attempt}: agent timed out after {step.timeout}s")
            log(f"    TIMEOUT after {step.timeout}s")
            continue

        # Verify
        log("    Verifying...")
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

    item_count = len(data.get("issues", data.get("components", [])))
    return VerifyResult(
        passed=True,
        message=f"{review_type} review written ({item_count} items)",
    )


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
        return VerifyResult(
            passed=False, message="Not valid JSON",
            errors=["Rewrite as valid JSON"],
        )

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
            message=(
                f"Only {len(reviewed_refs)}/{len(actual_refs)}"
                f" components reviewed ({coverage:.0%})"
            ),
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
            'Format: {{"issues": [{{"ref": "R1", "type": "...",'
            ' "severity": "CRITICAL|MAJOR|MINOR", '
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
            'Format: {{"issues": [{{"ref": "U1", "type": "...",'
            ' "severity": "CRITICAL|MAJOR|MINOR", '
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
# Evidence-hardened execution
# ---------------------------------------------------------------------------

_REVIEW_STEP_NAMES = frozenset({"fab-review", "ee-review"})

_VERIFICATION_PROMPT_TEMPLATE = (
    "You are a verification agent. You did NOT produce this work. "
    "The {original_step} agent claimed:\n\n{original_output}\n\n"
    "Verify this is accurate by reading the actual files. "
    "Report any discrepancies you find."
)

_HUMAN_GATE_POLL_INTERVAL = 5  # seconds


@dataclass
class EvidenceStep:
    """A process step that records evidence to the ledger."""

    name: str
    description: str
    agent_prompt: str
    verify: Callable[[str], VerifyResult]
    max_retries: int = 3
    timeout: int = 300
    retry_prompt: str = ""
    stage: str = "pcb"
    evidence_kind: EvidenceKind = EvidenceKind.REVIEW
    producer: str = "harness"
    requires_gate: bool = False
    inject_known_issues: bool = True


def _step_from_evidence_step(estep: EvidenceStep) -> Step:
    """Downcast an EvidenceStep to a plain Step for run_step()."""
    return Step(
        name=estep.name,
        description=estep.description,
        agent_prompt=estep.agent_prompt,
        verify=estep.verify,
        max_retries=estep.max_retries,
        timeout=estep.timeout,
        retry_prompt=estep.retry_prompt,
    )


def _issues_from_errors(errors: list[str]) -> list[Issue]:
    """Convert plain error strings to Issue models."""
    return [
        Issue(description=e, severity=Severity.MAJOR)
        for e in errors
    ]


def _write_evidence(
    board_path: str,
    step: EvidenceStep,
    result: StepResult,
) -> None:
    """Write an EvidenceRecord to the ledger after a step completes."""
    record = EvidenceRecord(
        kind=step.evidence_kind,
        stage=step.stage,
        step=step.name,
        board=Path(board_path).stem,
        producer=step.producer,
        passed=result.passed,
        summary=result.final_message,
        issues=_issues_from_errors(result.errors),
    )
    append_record(Path(board_path), record)
    notify_dashboard(record)


def run_evidence_step(step: EvidenceStep, board_path: str) -> StepResult:
    """Run a single step with evidence recording.

    If ``requires_gate`` is set, checks the stage gate first and fails
    immediately when the gate is not satisfied. If ``inject_known_issues``
    is set, known issues are prepended to the agent prompt.
    """
    # Gate check
    if step.requires_gate:
        gate = check_gate(Path(board_path), step.stage)
        if not gate.passed:
            log(f"  GATE BLOCKED: {gate.feedback}")
            sr = StepResult(
                step_name=step.name,
                passed=False,
                attempts=0,
                duration_seconds=0.0,
                final_message=f"Blocked by gate: {gate.feedback}",
                errors=gate.missing,
            )
            _write_evidence(board_path, step, sr)
            return sr

    # Known-issues injection
    working_step = EvidenceStep(
        name=step.name,
        description=step.description,
        agent_prompt=step.agent_prompt,
        verify=step.verify,
        max_retries=step.max_retries,
        timeout=step.timeout,
        retry_prompt=step.retry_prompt,
        stage=step.stage,
        evidence_kind=step.evidence_kind,
        producer=step.producer,
        requires_gate=step.requires_gate,
        inject_known_issues=step.inject_known_issues,
    )

    if step.inject_known_issues:
        board_dir = Path(board_path).parent
        context = load_known_issues(
            project_root=Path.cwd(),
            board_dir=board_dir,
            stage=step.stage,
            board=Path(board_path).stem,
        )
        if context:
            working_step.agent_prompt = context + "\n\n" + working_step.agent_prompt

    # Run the underlying step
    plain_step = _step_from_evidence_step(working_step)
    sr = run_step(plain_step, board_path)

    # Record evidence
    _write_evidence(board_path, step, sr)

    return sr


def run_verification_step(
    board_path: str,
    original_step: str,
    original_output: str,
) -> StepResult:
    """Run an independent verification agent against a prior step's output.

    Creates a VERIFICATION evidence record in the ledger.
    """
    start = time.time()

    prompt = _VERIFICATION_PROMPT_TEMPLATE.format(
        original_step=original_step,
        original_output=original_output,
    )
    response = run_claude(prompt, timeout=300)

    passed = response != "[TIMEOUT]"

    record = EvidenceRecord(
        kind=EvidenceKind.VERIFICATION,
        stage="pcb",
        step=f"verify-{original_step}",
        board=Path(board_path).stem,
        producer="verify-agent",
        passed=passed,
        summary=response[:500] if passed else "Verification timed out",
    )
    append_record(Path(board_path), record)
    notify_dashboard(record)

    return StepResult(
        step_name=f"verify-{original_step}",
        passed=passed,
        attempts=1,
        duration_seconds=time.time() - start,
        final_message=response[:200] if passed else "Verification timed out",
    )


def _poll_for_human_approval(board_path: str, stage: str) -> bool:
    """Poll the ledger every few seconds until HUMAN_APPROVAL appears.

    Returns True when approval is found.
    """
    log(f"  Waiting for human approval (stage={stage})...")
    log(f"  Add approval via: python -m kicad_pipeline.evidence.approve {board_path}")
    notify_dashboard_log("Waiting for human approval via dashboard", "warning")
    while True:
        ledger = load_ledger(Path(board_path))
        if ledger.has_passing(stage, EvidenceKind.HUMAN_APPROVAL):
            log("  Human approval received.")
            return True
        time.sleep(_HUMAN_GATE_POLL_INTERVAL)


def run_hardened_process(
    name: str,
    steps: list[EvidenceStep],
    board_path: str,
) -> ProcessResult:
    """Run a complete evidence-hardened process.

    Like ``run_process()`` but with:
    - Gate checks before gated steps
    - Known-issues injection
    - Evidence recording after every step
    - Verification steps after review steps
    - Stage-transition gate results
    - Human approval polling at human gate steps
    """
    log(f"Hardened process '{name}' starting on {Path(board_path).stem}")
    result = ProcessResult(
        process_name=name,
        board=board_path,
        passed=False,
        started=datetime.now().isoformat(),
    )

    prev_stage: str | None = None

    for step in steps:
        log(f"\n{'─' * 50}")
        log(f"STEP: {step.name} — {step.description}")
        log(f"{'─' * 50}")

        # Stage transition — write gate result
        if prev_stage is not None and step.stage != prev_stage:
            gate = check_gate(Path(board_path), prev_stage)
            gate_record = EvidenceRecord(
                kind=EvidenceKind.GATE_RESULT,
                stage=prev_stage,
                step=f"{prev_stage}-gate",
                board=Path(board_path).stem,
                producer="harness",
                passed=gate.passed,
                summary=gate.feedback or f"{prev_stage} gate passed",
            )
            append_record(Path(board_path), gate_record)
            notify_dashboard(gate_record)
            if not gate.passed:
                log(f"Stage transition gate failed: {gate.feedback}")
                result.finished = datetime.now().isoformat()
                return result

        # Run the evidence step
        sr = run_evidence_step(step, board_path)
        result.steps.append(sr)

        if not sr.passed:
            log(f"\nSTEP '{step.name}' FAILED — process halted.")
            result.finished = datetime.now().isoformat()
            return result

        # Run verification after review steps
        if step.name in _REVIEW_STEP_NAMES:
            log(f"  Running verification for {step.name}...")
            vr = run_verification_step(
                board_path,
                original_step=step.name,
                original_output=sr.final_message,
            )
            result.steps.append(vr)
            if not vr.passed:
                log(f"\nVERIFICATION of '{step.name}' FAILED — process halted.")
                result.finished = datetime.now().isoformat()
                return result

        prev_stage = step.stage

    result.passed = True
    result.finished = datetime.now().isoformat()
    log(f"\nHardened process '{name}' COMPLETE — all steps passed.")
    return result


# Evidence-aware step definitions
PCB_REVIEW_EVIDENCE_STEPS: list[EvidenceStep] = [
    EvidenceStep(
        name="render",
        description="Generate all board-level + per-component renders",
        agent_prompt=PCB_REVIEW_STEPS[0].agent_prompt,
        verify=verify_renders,
        max_retries=3,
        timeout=600,
        retry_prompt=PCB_REVIEW_STEPS[0].retry_prompt,
        stage="pcb",
        evidence_kind=EvidenceKind.RENDER,
        producer="harness",
    ),
    EvidenceStep(
        name="fab-review",
        description="Fabricator persona reviews all renders",
        agent_prompt=PCB_REVIEW_STEPS[1].agent_prompt,
        verify=lambda bp: verify_review_json("fab", bp),
        max_retries=2,
        timeout=300,
        retry_prompt=PCB_REVIEW_STEPS[1].retry_prompt,
        stage="pcb",
        evidence_kind=EvidenceKind.REVIEW,
        producer="fab-agent",
    ),
    EvidenceStep(
        name="ee-review",
        description="Electrical engineer persona reviews all renders",
        agent_prompt=PCB_REVIEW_STEPS[2].agent_prompt,
        verify=lambda bp: verify_review_json("ee", bp),
        max_retries=2,
        timeout=300,
        retry_prompt=PCB_REVIEW_STEPS[2].retry_prompt,
        stage="pcb",
        evidence_kind=EvidenceKind.REVIEW,
        producer="ee-agent",
    ),
    EvidenceStep(
        name="3d-component-verify",
        description="Per-component 3D body-to-pad alignment check",
        agent_prompt=PCB_REVIEW_STEPS[3].agent_prompt,
        verify=verify_component_review,
        max_retries=3,
        timeout=600,
        retry_prompt=PCB_REVIEW_STEPS[3].retry_prompt,
        stage="pcb",
        evidence_kind=EvidenceKind.VERIFICATION,
        producer="verify-agent",
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

EVIDENCE_PROCESSES: dict[str, list[EvidenceStep]] = {
    "pcb-review-hardened": PCB_REVIEW_EVIDENCE_STEPS,
}


def main() -> None:
    all_process_names = list(PROCESSES.keys()) + list(EVIDENCE_PROCESSES.keys())
    parser = argparse.ArgumentParser(
        description="Process Runner — mandatory-step orchestration with verification gates"
    )
    parser.add_argument("process", choices=all_process_names,
                        help="Process to run")
    parser.add_argument("board", nargs="?", default=None,
                        help="Board selector (mcu/relay/power/analog/ethernet/all)")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Override max retries for all steps")
    parser.add_argument("--start-at", type=str, default=None,
                        help="Skip to this step (e.g., 'ee-review')")
    parser.add_argument("--list-steps", action="store_true",
                        help="List steps in the process and exit")
    parser.add_argument("--hardened", action="store_true",
                        help="Use evidence-hardened execution (auto for *-hardened processes)")
    parser.add_argument("--dashboard", type=str, default="http://localhost:8080",
                        help="Dashboard URL for live notifications (default: http://localhost:8080)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable dashboard notifications")
    args = parser.parse_args()

    # Configure dashboard URL
    global _dashboard_url
    _dashboard_url = None if args.no_dashboard else args.dashboard

    # Determine if this is an evidence-hardened process
    use_hardened = args.hardened or args.process in EVIDENCE_PROCESSES

    if use_hardened:
        # Resolve to evidence process
        evidence_key = args.process
        if evidence_key not in EVIDENCE_PROCESSES:
            evidence_key = f"{evidence_key}-hardened"
        if evidence_key not in EVIDENCE_PROCESSES:
            print(f"No hardened variant for '{args.process}'. "
                  f"Available: {', '.join(EVIDENCE_PROCESSES.keys())}")
            sys.exit(1)
        evidence_steps: list[EvidenceStep] = list(EVIDENCE_PROCESSES[evidence_key])
        step_list_for_display: list[Step | EvidenceStep] = list(evidence_steps)
    else:
        steps = list(PROCESSES[args.process])
        step_list_for_display = list(steps)

    if args.list_steps:
        label = f"{args.process} (hardened)" if use_hardened else args.process
        print(f"Process: {label}")
        for i, s in enumerate(step_list_for_display, 1):
            print(f"  {i}. {s.name} — {s.description} (max {s.max_retries} retries)")
        return

    if not args.board and not args.list_steps:
        parser.error("board selector required")

    if args.max_retries is not None:
        for s in step_list_for_display:
            s.max_retries = args.max_retries

    if args.start_at:
        idx = next(
            (i for i, s in enumerate(step_list_for_display) if s.name == args.start_at),
            None,
        )
        if idx is None:
            print(
                f"Unknown step '{args.start_at}'. "
                f"Steps: {[s.name for s in step_list_for_display]}"
            )
            sys.exit(1)
        if use_hardened:
            evidence_steps = evidence_steps[idx:]
        else:
            steps = steps[idx:]
        log(f"Starting at step '{args.start_at}' (skipping {idx} steps)")

    boards = resolve_boards(args.board)

    results: list[ProcessResult] = []
    for board_path in boards:
        if use_hardened:
            result = run_hardened_process(
                args.process, evidence_steps, board_path,
            )
        else:
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
