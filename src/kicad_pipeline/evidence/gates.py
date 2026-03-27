"""Stage gate definitions — what evidence is required to proceed.

Each pipeline stage has a gate that checks the evidence ledger for
required proof before allowing transition to the next stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.evidence.ledger import load_ledger

if TYPE_CHECKING:
    from pathlib import Path
from kicad_pipeline.evidence.models import EvidenceKind, GateResult

# What evidence kinds each stage requires before the NEXT stage can start.
STAGE_GATES: dict[str, list[EvidenceKind]] = {
    "requirements": [
        EvidenceKind.REVIEW,
    ],
    "schematic": [
        EvidenceKind.DRC_REPORT,
        EvidenceKind.REVIEW,
    ],
    "pcb": [
        EvidenceKind.RENDER,
        EvidenceKind.DRC_REPORT,
        EvidenceKind.REVIEW,
        EvidenceKind.VERIFICATION,
        EvidenceKind.SCORE,
    ],
    "validation": [
        EvidenceKind.DRC_REPORT,
        EvidenceKind.SCORE,
        EvidenceKind.HUMAN_APPROVAL,
    ],
    "production": [
        EvidenceKind.HUMAN_APPROVAL,
    ],
}

ALL_STAGES = ("requirements", "schematic", "pcb", "validation", "production")


def check_gate(board_path: Path, stage: str) -> GateResult:
    """Check if all required evidence is present and passing for a stage.

    Returns a GateResult with pass/fail and details about what's missing.
    """
    required = STAGE_GATES.get(stage, [])
    if not required:
        return GateResult(
            gate_name=f"{stage}-gate",
            stage=stage,
            passed=True,
            required_evidence=[],
            feedback="No evidence required for this stage.",
        )

    ledger = load_ledger(board_path)
    present: list[str] = []
    missing: list[str] = []

    for kind in required:
        if ledger.has_passing(stage, kind):
            present.append(kind.value)
        else:
            missing.append(kind.value)

    passed = len(missing) == 0
    feedback = ""
    if not passed:
        feedback = f"Missing evidence for {stage} gate: {', '.join(missing)}"

    return GateResult(
        gate_name=f"{stage}-gate",
        stage=stage,
        passed=passed,
        required_evidence=[k.value for k in required],
        present_evidence=present,
        missing=missing,
        feedback=feedback,
    )
