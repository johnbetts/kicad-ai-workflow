"""Dual-persona review gates — fabricator + EE, machine-readable verdicts.

Gate C feedback 2026-06-11 item 6: the per-board pipeline gains a
MANDATORY dual-persona review stage between the renders and the Gate B
vision checklist. Two independent personas review the same 4-view
renders:

* ``fab`` — a fabricator/assembler: DFM, clearances, accessibility,
  visual organization (clean / aligned / grid-ordered).
* ``ee`` — an electrical engineer: decoupling, signal flow, crossing
  attach lines, isolation, RF, power topology.

Like Gate B (:mod:`kicad_pipeline.placement_v2.vision_gate`), prose
findings do not count: each persona answers a fixed checklist in a
strict JSON schema, parsed by :func:`parse_persona_verdicts`, and only
parsed verdicts enter the build ledger (stages ``review_fab`` and
``review_ee``). The same standing rule applies — a persona finding that
Gate A missed must be converted into a deterministic check in the same
change that fixes the board.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.placement_v2.ir import Severity, Violation
from kicad_pipeline.placement_v2.ledger import StageRecord, sha256_file
from kicad_pipeline.placement_v2.vision_gate import (
    REQUIRED_VIEWS,
    VisionProtocolError,
    VisionVerdict,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

#: The two mandatory review personas, in ledger-stage order.
PERSONAS: tuple[str, ...] = ("fab", "ee")


@dataclass(frozen=True)
class PersonaCheck:
    """One question a persona must answer over the given views."""

    check_id: str
    views: tuple[str, ...]
    prompt: str
    severity: Severity


FAB_CHECKLIST: tuple[PersonaCheck, ...] = (
    PersonaCheck(
        check_id="fab_clearances",
        views=("2d", "3d_top"),
        prompt=(
            "Do all component bodies/courtyards keep visible assembly "
            "clearance (no touching or near-touching parts, including "
            "between THT bodies and neighboring passives)?"
        ),
        severity=Severity.CRITICAL,
    ),
    PersonaCheck(
        check_id="fab_connector_access",
        views=("2d", "3d_top", "3d_iso"),
        prompt=(
            "Is every connector/terminal at a board edge with its "
            "wire/plug opening facing OUTWARD, and is the area between "
            "each connector and its edge completely clear (screwdriver "
            "and cable access)?"
        ),
        severity=Severity.CRITICAL,
    ),
    PersonaCheck(
        check_id="fab_grid_alignment",
        views=("2d",),
        prompt=(
            "Do repeated subcircuits have IDENTICAL layouts, and do "
            "components form aligned rows/columns rather than scattered "
            "or ragged placement?"
        ),
        severity=Severity.MAJOR,
    ),
    PersonaCheck(
        check_id="fab_silkscreen",
        views=("2d",),
        prompt=(
            "Are reference designators legible, near their parts, and "
            "not overlapping pads or other text?"
        ),
        severity=Severity.MINOR,
    ),
    PersonaCheck(
        check_id="fab_board_utilization",
        views=("2d",),
        prompt=(
            "Is the board area used proportionately — no large dead "
            "regions while another area is congested, and no component "
            "stranded far from its functional group?"
        ),
        severity=Severity.MAJOR,
    ),
)

EE_CHECKLIST: tuple[PersonaCheck, ...] = (
    PersonaCheck(
        check_id="ee_domain_zoning",
        views=("2d",),
        prompt=(
            "Do voltage domains form contiguous zones with field "
            "wiring at its own board edge? The 24V/relay-contact side "
            "must hug the edge where its harness terminal sits — high "
            "voltage must never have to cross the board interior, and "
            "low-voltage digital lines must not run through the "
            "relay/24V zone to reach a terminal. Ferrite-isolated "
            "analog must be one contiguous region apart from the "
            "digital section."
        ),
        severity=Severity.MAJOR,
    ),
    PersonaCheck(
        check_id="ee_decoupling",
        views=("2d",),
        prompt=(
            "Is every decoupling capacitor immediately adjacent to the "
            "IC power pin it serves (a few mm, no other part between)?"
        ),
        severity=Severity.MAJOR,
    ),
    PersonaCheck(
        check_id="ee_signal_flow",
        views=("2d",),
        prompt=(
            "Do subcircuit chains read in signal order (input -> "
            "processing -> output) without doubling back across the "
            "group?"
        ),
        severity=Severity.MAJOR,
    ),
    PersonaCheck(
        check_id="ee_crossing_attach_lines",
        views=("2d",),
        prompt=(
            "For each paired connector/part pin row (relay contacts to "
            "screw terminal, ADC channel to its terminal), would the "
            "straight pad-to-pad connections cross each other? Crossing "
            "pairs force avoidable crossover traces."
        ),
        severity=Severity.MAJOR,
    ),
    PersonaCheck(
        check_id="ee_isolation",
        views=("2d",),
        prompt=(
            "Are mains/high-voltage parts (relays, their terminals) "
            "separated from logic by a visible gap or isolation "
            "slot, with no logic part inside the isolation region?"
        ),
        severity=Severity.CRITICAL,
    ),
    PersonaCheck(
        check_id="ee_rf_antenna",
        views=("2d", "3d_top"),
        prompt=(
            "If an RF module is present, is its antenna section at a "
            "board edge with the keepout free of components and copper?"
        ),
        severity=Severity.CRITICAL,
    ),
    PersonaCheck(
        check_id="ee_power_topology",
        views=("2d",),
        prompt=(
            "Does power flow input -> protection/filter -> regulation "
            "-> loads in placement order, with bulk capacitors at the "
            "input side of their regulator?"
        ),
        severity=Severity.MAJOR,
    ),
)

_CHECKLISTS: dict[str, tuple[PersonaCheck, ...]] = {
    "fab": FAB_CHECKLIST,
    "ee": EE_CHECKLIST,
}

_PERSONA_FRAMING: dict[str, str] = {
    "fab": (
        "Review these PCB renders as a skeptical FABRICATOR/ASSEMBLER. "
        "You care about DFM: clearances, access, alignment, silkscreen, "
        "board utilization. Try to FAIL each check; pass only when the "
        "render clearly satisfies it."
    ),
    "ee": (
        "Review these PCB renders as a skeptical ELECTRICAL ENGINEER. "
        "You care about signal integrity, decoupling, isolation, RF, "
        "and routability. Try to FAIL each check; pass only when the "
        "render clearly satisfies it."
    ),
}


def checklist_for(persona: str) -> tuple[PersonaCheck, ...]:
    """The checklist for a persona; raises on unknown persona."""
    if persona not in _CHECKLISTS:
        raise VisionProtocolError(f"unknown persona {persona!r}; expected {PERSONAS}")
    return _CHECKLISTS[persona]


def persona_instructions(persona: str, renders: Mapping[str, Path]) -> str:
    """The exact prompt block handed to one persona's review subagent."""
    checks = "\n".join(
        f"- {c.check_id} (views: {', '.join(c.views)}): {c.prompt}"
        for c in checklist_for(persona)
    )
    views = "\n".join(f"- {view}: {path}" for view, path in sorted(renders.items()))
    return (
        f"{_PERSONA_FRAMING[persona]}\n\n"
        f"Renders:\n{views}\n\nChecks:\n{checks}\n\n"
        "Answer with ONLY a JSON array, one object per check:\n"
        '[{"check_id": "...", "passed": true|false, '
        '"refs": ["R1", ...], "evidence": "one sentence"}]\n'
        "A check that does not apply to this board (e.g. no RF module) "
        "passes with evidence \"not applicable\"."
    )


def parse_persona_verdicts(persona: str, raw: str) -> tuple[VisionVerdict, ...]:
    """Parse one persona's verdicts; strict — every check answered once."""
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
    expected = {c.check_id for c in checklist_for(persona)}
    got = [v.check_id for v in verdicts]
    if sorted(got) != sorted(expected):
        missing = expected - set(got)
        extra = set(got) - expected
        raise VisionProtocolError(
            f"{persona} checklist mismatch: missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    return tuple(verdicts)


def persona_violations(
    persona: str, verdicts: tuple[VisionVerdict, ...],
) -> tuple[Violation, ...]:
    """Failed verdicts become ledger violations at the check's severity."""
    by_id = {c.check_id: c for c in checklist_for(persona)}
    return tuple(
        Violation(
            constraint=f"review_{persona}:{v.check_id}",
            refs=v.refs,
            severity=by_id[v.check_id].severity,
            measured=1.0,
            limit=0.0,
            message=v.evidence or f"{persona} check {v.check_id} failed",
        )
        for v in verdicts
        if not v.passed
    )


def persona_record(
    persona: str,
    verdicts: tuple[VisionVerdict, ...],
    renders: Mapping[str, Path],
    board_sha256: str,
    timestamp: str,
) -> StageRecord:
    """Build the ``review_<persona>`` ledger record from parsed verdicts.

    Raises :class:`VisionProtocolError` when any required view's render
    is missing — a persona review of fewer than 4 views is not a review.
    Blocking severities are CRITICAL and MAJOR; a MINOR-only review
    still passes (matching :class:`GateAReport` semantics).
    """
    missing = [v for v in REQUIRED_VIEWS if v not in renders]
    if missing:
        raise VisionProtocolError(f"missing required render views: {missing}")
    render_hash = ",".join(
        f"{view}:{sha256_file(renders[view])[:12]}" for view in REQUIRED_VIEWS
    )
    violations = persona_violations(persona, verdicts)
    blocking = (Severity.CRITICAL, Severity.MAJOR)
    return StageRecord(
        stage=f"review_{persona}",
        input_sha256=board_sha256,
        output_sha256=render_hash,
        checks_run=tuple(c.check_id for c in checklist_for(persona)),
        violations=violations,
        passed=all(v.severity not in blocking for v in violations),
        detail=f"{len(verdicts)} verdicts over {len(REQUIRED_VIEWS)} views",
        timestamp=timestamp,
    )


#: Every stage that must be green before a board may be PRESENTED to
#: the human — the machine definition of "reviewed" (Gate C item 6).
PRESENTATION_STAGES: tuple[str, ...] = (
    "certify",
    "cells",
    "floorplan",
    "sync",
    "gate_a",
    "review_fab",
    "review_ee",
    "gate_b",
)
