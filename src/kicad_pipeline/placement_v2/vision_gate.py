"""Gate B — adversarial visual review with machine-readable verdicts.

Vision is the scout; geometry is the garrison. A vision reviewer (a
subagent reading the 4-view renders — NEVER the main orchestrator
context) receives :data:`VISION_CHECKLIST`, and must answer in the
strict JSON schema parsed by :func:`parse_verdicts`. Prose findings do
not count; only parsed verdicts enter the build ledger.

Standing rule (architecture §7): when a vision check fails on something
Gate A missed, fixing the board is NOT sufficient — a deterministic IR
constraint or verifier rule covering that defect class must be added in
the same change, so vision findings monotonically convert into
geometry checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import ValidationError
from kicad_pipeline.placement_v2.ir import Severity, Violation
from kicad_pipeline.placement_v2.ledger import StageRecord, sha256_file

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

#: The 4-view render standard. A Gate B run without all four is invalid.
REQUIRED_VIEWS: tuple[str, ...] = ("2d", "3d_top", "3d_iso", "3d_iso_back")


class VisionProtocolError(ValidationError):
    """The vision reviewer's output violated the verdict schema."""


@dataclass(frozen=True)
class VisionCheck:
    """One question the vision reviewer must answer for given views."""

    check_id: str
    views: tuple[str, ...]
    prompt: str
    severity: Severity


@dataclass(frozen=True)
class VisionVerdict:
    """The reviewer's answer for one check — machine-readable, blocking."""

    check_id: str
    passed: bool
    refs: tuple[str, ...]  # offending components, empty when passed
    evidence: str  # what was seen, in one sentence


VISION_CHECKLIST: tuple[VisionCheck, ...] = (
    VisionCheck(
        check_id="bodies_no_overlap",
        views=("3d_top", "3d_iso", "3d_iso_back"),
        prompt=(
            "Do any two component bodies intersect or stack on each other? "
            "Any overlap of 3D bodies is a FAIL regardless of pad geometry."
        ),
        severity=Severity.CRITICAL,
    ),
    VisionCheck(
        check_id="bodies_flat_on_board",
        views=("3d_iso", "3d_iso_back"),
        prompt="Is every component body sitting flat on the PCB surface "
               "(not floating, not sunken, not tilted)?",
        severity=Severity.CRITICAL,
    ),
    VisionCheck(
        check_id="bodies_on_pads",
        views=("3d_top",),
        prompt="Does every component body sit centered over its own pads "
               "(no body displaced from its footprint)?",
        severity=Severity.CRITICAL,
    ),
    VisionCheck(
        check_id="orientation_sane",
        views=("2d", "3d_top"),
        prompt="Are connectors opening toward board edges, relays aligned "
               "as a uniform row, and polarized parts consistently oriented?",
        severity=Severity.MAJOR,
    ),
    VisionCheck(
        check_id="antenna_zone_clear",
        views=("2d", "3d_top"),
        prompt="Is the RF antenna keepout region (if any) at a board edge "
               "and completely free of components and copper?",
        severity=Severity.CRITICAL,
    ),
    VisionCheck(
        check_id="groups_visually_coherent",
        views=("2d",),
        prompt="Do functional groups read as tight, organized, grid-aligned "
               "clusters rather than scattered parts?",
        severity=Severity.MAJOR,
    ),
)


def reviewer_instructions(renders: Mapping[str, Path]) -> str:
    """The exact prompt block handed to the vision review subagent."""
    checks = "\n".join(
        f"- {c.check_id} (views: {', '.join(c.views)}): {c.prompt}"
        for c in VISION_CHECKLIST
    )
    views = "\n".join(f"- {view}: {path}" for view, path in sorted(renders.items()))
    return (
        "Review these PCB renders as a skeptical fabricator. Try to FAIL "
        "each check; pass only when the render clearly satisfies it.\n\n"
        f"Renders:\n{views}\n\nChecks:\n{checks}\n\n"
        "Answer with ONLY a JSON array, one object per check:\n"
        '[{"check_id": "...", "passed": true|false, '
        '"refs": ["R1", ...], "evidence": "one sentence"}]'
    )


def parse_verdicts(raw: str) -> tuple[VisionVerdict, ...]:
    """Parse reviewer output; strict — every check answered exactly once."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisionProtocolError(f"verdicts are not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise VisionProtocolError("verdicts must be a JSON array")

    verdicts: list[VisionVerdict] = []
    for item in data:
        if not isinstance(item, dict):
            raise VisionProtocolError(f"verdict entry is not an object: {item!r}")
        try:
            verdicts.append(VisionVerdict(
                check_id=str(item["check_id"]),
                passed=bool(item["passed"]),
                refs=tuple(str(r) for r in item.get("refs", ())),
                evidence=str(item.get("evidence", "")),
            ))
        except KeyError as exc:
            raise VisionProtocolError(f"verdict missing field {exc}") from exc

    expected = {c.check_id for c in VISION_CHECKLIST}
    got = [v.check_id for v in verdicts]
    if sorted(got) != sorted(expected):
        missing = expected - set(got)
        extra = set(got) - expected
        raise VisionProtocolError(
            f"checklist mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    return tuple(verdicts)


def verdicts_to_violations(
    verdicts: tuple[VisionVerdict, ...],
) -> tuple[Violation, ...]:
    """Failed verdicts become ledger violations at the check's severity."""
    by_id = {c.check_id: c for c in VISION_CHECKLIST}
    return tuple(
        Violation(
            constraint=f"vision:{v.check_id}",
            refs=v.refs,
            severity=by_id[v.check_id].severity,
            measured=1.0,
            limit=0.0,
            message=v.evidence or f"vision check {v.check_id} failed",
        )
        for v in verdicts
        if not v.passed
    )


def gate_b_record(
    verdicts: tuple[VisionVerdict, ...],
    renders: Mapping[str, Path],
    board_sha256: str,
    timestamp: str,
) -> StageRecord:
    """Build the Gate B ledger record from parsed verdicts + renders.

    Raises :class:`VisionProtocolError` when any required view's render
    is missing — a review of fewer than 4 views is not a review.
    """
    missing = [v for v in REQUIRED_VIEWS if v not in renders]
    if missing:
        raise VisionProtocolError(f"missing required render views: {missing}")
    render_hash = ",".join(
        f"{view}:{sha256_file(renders[view])[:12]}" for view in REQUIRED_VIEWS
    )
    violations = verdicts_to_violations(verdicts)
    return StageRecord(
        stage="gate_b",
        input_sha256=board_sha256,
        output_sha256=render_hash,
        checks_run=tuple(c.check_id for c in VISION_CHECKLIST),
        violations=violations,
        passed=not violations,
        detail=f"{len(verdicts)} verdicts over {len(REQUIRED_VIEWS)} views",
        timestamp=timestamp,
    )


def golden_diff(render: Path, baseline: Path) -> bool:
    """True when a render matches its golden baseline byte-for-byte.

    Renders are deterministic for identical boards, so a clean diff
    proves placement regression-freedom without vision in the loop.
    """
    if not baseline.exists():
        return False
    return sha256_file(render) == sha256_file(baseline)
