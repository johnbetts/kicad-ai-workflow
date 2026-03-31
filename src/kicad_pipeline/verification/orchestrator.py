"""Verification orchestrator — runs checks sequentially, each via fresh AI agent.

The orchestrator is the single entry point for all PCB verification. It:
1. Runs programmatic structural checks (fast, deterministic, no AI)
2. Renders images via kicad-image-gen
3. Runs per-component AI visual checks (batched 5-8 per agent call)
4. Runs board-level AI checks (one call per persona)
5. Cross-references programmatic vs visual results
6. Writes evidence records to the ledger

Each AI agent gets ONLY the data it needs — focused prompt, specific images,
relevant known bugs. The script validates agent JSON output before proceeding.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.verification.checklist import VerificationChecklist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Persona(str, enum.Enum):
    """Who performs the check."""

    PROGRAMMATIC = "programmatic"
    FAB = "fab"
    EE = "ee"


class CheckSeverity(str, enum.Enum):
    """How serious a failure is."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class StepType(str, enum.Enum):
    """How the step is executed."""

    PROGRAMMATIC = "programmatic"
    AI_VISUAL = "ai_visual"
    AI_PINOUT = "ai_pinout"
    AI_BOARD_LEVEL = "ai_board_level"
    CROSS_REFERENCE = "cross_reference"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckItem:
    """Single item within a verification step.

    Each check item has a unique ``item_id`` that follows the pattern
    ``{PERSONA}-{CATEGORY}-{REF}`` (e.g. ``FAB-BODY-ALIGN-R1``).
    Board-level checks omit the ref: ``FAB-BOARD-PATTERN``.
    """

    item_id: str
    description: str
    component_ref: str | None  # None for board-level checks
    persona: Persona
    category: str
    known_bug_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckResult:
    """Result of a single check item — always pass/fail, never ambiguous."""

    item_id: str
    passed: bool
    detail: str
    severity: CheckSeverity
    confidence: float = 1.0  # 1.0 for programmatic, 0.0-1.0 for AI
    evidence_path: str | None = None


@dataclass(frozen=True)
class StepResult:
    """Result of one verification step (may contain many check items)."""

    step_name: str
    passed: bool
    check_results: tuple[CheckResult, ...]
    duration_secs: float
    step_type: StepType = StepType.PROGRAMMATIC
    agent_model: str | None = None
    raw_response: str | None = None


@dataclass
class VerificationReport:
    """Aggregate result of a full verification run."""

    board_path: str
    started: str = ""
    finished: str = ""
    steps: list[StepResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Report passes only if no step has critical/major failures."""
        for step in self.steps:
            for cr in step.check_results:
                if not cr.passed and cr.severity in (
                    CheckSeverity.CRITICAL,
                    CheckSeverity.MAJOR,
                ):
                    return False
        return True

    @property
    def total_checks(self) -> int:
        return sum(len(s.check_results) for s in self.steps)

    @property
    def passed_checks(self) -> int:
        return sum(
            1
            for s in self.steps
            for cr in s.check_results
            if cr.passed
        )

    @property
    def failed_checks(self) -> int:
        return self.total_checks - self.passed_checks

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [
            cr
            for s in self.steps
            for cr in s.check_results
            if not cr.passed and cr.severity == CheckSeverity.CRITICAL
        ]

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Verification: {status}",
            f"Board: {self.board_path}",
            f"Checks: {self.passed_checks}/{self.total_checks} passed",
        ]
        crits = self.critical_failures
        if crits:
            lines.append(f"CRITICAL failures ({len(crits)}):")
            for cr in crits[:10]:
                lines.append(f"  [{cr.item_id}] {cr.detail}")
        for step in self.steps:
            fails = [cr for cr in step.check_results if not cr.passed]
            status_icon = "PASS" if step.passed else "FAIL"
            lines.append(
                f"  [{status_icon}] {step.step_name}"
                f" ({len(step.check_results) - len(fails)}/{len(step.check_results)}"
                f" passed, {step.duration_secs:.1f}s)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class VerificationOrchestrator:
    """Runs verification steps sequentially, validates output at each step."""

    def __init__(
        self,
        board_path: Path,
        checklist: VerificationChecklist | None = None,
        evidence_dir: Path | None = None,
    ) -> None:
        self.board_path = board_path
        self.board_name = board_path.stem
        self.board_dir = board_path.parent
        self.evidence_dir = evidence_dir or self.board_dir / "verification_evidence"
        self.checklist = checklist
        self._report = VerificationReport(board_path=str(board_path))

    def run_programmatic_checks(
        self,
        component_refs: list[str] | None = None,
    ) -> list[StepResult]:
        """Run all programmatic structural checks. Fast, no AI."""
        from kicad_pipeline.verification.component_checks import (
            run_programmatic_checks,
        )

        start = time.time()
        results = run_programmatic_checks(
            self.board_path,
            component_refs=component_refs,
            checklist=self.checklist,
        )
        duration = time.time() - start

        step = StepResult(
            step_name="programmatic_structural",
            passed=all(cr.passed for cr in results if cr.severity in (
                CheckSeverity.CRITICAL, CheckSeverity.MAJOR,
            )),
            check_results=tuple(results),
            duration_secs=duration,
            step_type=StepType.PROGRAMMATIC,
        )
        self._report.steps.append(step)
        return [step]

    def run_all(
        self,
        programmatic_only: bool = False,
        persona: str | None = None,
        component_refs: list[str] | None = None,
        known_bugs_only: bool = False,
    ) -> VerificationReport:
        """Full verification pipeline.

        Args:
            programmatic_only: Skip AI checks (fast mode for CI).
            persona: Run only one persona ("fab" or "ee").
            component_refs: Subset of components to check.
            known_bugs_only: Only check items linked to known bugs.

        Returns:
            VerificationReport with all results.
        """
        from datetime import datetime, timezone

        self._report.started = datetime.now(tz=timezone.utc).isoformat()

        # Step 1: Programmatic checks (always run)
        logger.info("Running programmatic structural checks...")
        prog_steps = self.run_programmatic_checks(component_refs)

        # Hard gate: stop on critical programmatic failures
        critical_fails = [
            cr
            for s in prog_steps
            for cr in s.check_results
            if not cr.passed and cr.severity == CheckSeverity.CRITICAL
        ]
        if critical_fails:
            logger.error(
                "HARD GATE: %d critical programmatic failures — stopping",
                len(critical_fails),
            )
            self._report.finished = datetime.now(tz=timezone.utc).isoformat()
            return self._report

        if programmatic_only:
            self._report.finished = datetime.now(tz=timezone.utc).isoformat()
            return self._report

        # Steps 2-8: AI visual checks (Phase 2 — not yet implemented)
        # Planned: image rendering, fabricator/EE visual checks,
        # board-level reviews, and cross-reference validation.
        logger.info(
            "AI visual checks not yet implemented (Phase 2). "
            "Use --programmatic-only for now."
        )

        self._report.finished = datetime.now(tz=timezone.utc).isoformat()
        return self._report

    @property
    def report(self) -> VerificationReport:
        """Access the current report."""
        return self._report
