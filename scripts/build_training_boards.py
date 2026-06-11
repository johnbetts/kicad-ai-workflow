#!/usr/bin/env python3
"""Gated batch driver for the five v2 training boards.

Gate C feedback 2026-06-11 item 6 (MANDATORY process): per-board
pipeline is build -> Gate A -> render -> dual-persona review (EE +
fabricator, structured verdicts) -> Gate B vision checklist -> ledger
``all_green()`` including BOTH review stages -> only then present to
the human. This script encodes that loop so no stage can be manually
skipped: ``present`` refuses any board whose ledger is not fully green.

Subcommands::

    build [board ...]      Build, write PCB+schematic+project, run the
                           sync gate and Gate A, render 4 views, and
                           write review_requests.json with the exact
                           prompts for the review subagents.
    verdict <board> <stage> <verdicts.json>
                           Ingest a subagent's verdicts (stage one of
                           review_fab / review_ee / gate_b) into the
                           board's build ledger.
    status [board ...]     Ledger summary + presentability per board.
    present [board ...]    List final artifacts; REFUSES (exit 1) any
                           board not all-green across every stage.

Boards: relay_group, analog_input, power_chain, mcu_core, ethernet.
Output: output/<board>_v2/ (board, schematic, project, renders,
build_ledger.jsonl, review_requests.json).

Review verdicts are produced by vision subagents reading the renders —
NEVER by the orchestrator's own context. The prompts handed to those
subagents come verbatim from this script's review_requests.json.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: board name -> training-script module providing _build_requirements().
BOARDS: dict[str, str] = {
    "relay_group": "train_relay_group",
    "analog_input": "train_analog_input",
    "power_chain": "train_power_chain",
    "mcu_core": "train_mcu_core",
    "ethernet": "train_ethernet",
}

PART_RULES_PATH = _repo / "data" / "part_rules.json"

#: render view name (ledger/REQUIRED_VIEWS key) -> kicad-image-gen args.
_VIEWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("2d", ("2d", "-w", "1024")),
    ("3d_top", ("3d", "--view", "top", "-w", "1024")),
    ("3d_iso", ("3d", "--view", "iso", "-w", "1024")),
    ("3d_iso_back", ("3d", "--view", "iso-back", "-w", "1024")),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _out_dir(board: str) -> Path:
    return _repo / "output" / f"{board}_v2"


def _ledger(board: str):
    from kicad_pipeline.placement_v2.ledger import BuildLedger

    return BuildLedger(_out_dir(board) / "build_ledger.jsonl")


def _render_paths(board: str) -> dict[str, Path]:
    out = _out_dir(board)
    return {view: out / f"{board}_v2_{view}.png" for view, _ in _VIEWS}


def _render_views(board: str, pcb_path: Path) -> dict[str, Path]:
    import subprocess

    renders: dict[str, Path] = {}
    for view, args in _VIEWS:
        out_png = _render_paths(board)[view]
        cmd = ["kicad-image-gen", args[0], str(pcb_path), *args[1:], "-o", str(out_png)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"  WARNING: render {view} failed: {exc}")
            continue
        if out_png.exists() and out_png.stat().st_size > 1000:
            renders[view] = out_png
    return renders


def _sync_record(pcb, requirements, board_sha: str):
    """Run the schematic_pcb_sync hard gate as a ledger stage."""
    from kicad_pipeline.evals.dfm_gates import check_schematic_pcb_sync
    from kicad_pipeline.placement_v2.ir import Severity, Violation
    from kicad_pipeline.placement_v2.ledger import StageRecord

    result = check_schematic_pcb_sync(pcb, requirements)
    violations = ()
    if not result.passed:
        violations = (Violation(
            constraint="schematic_pcb_sync", refs=(),
            severity=Severity.CRITICAL, measured=1.0, limit=0.0,
            message=result.detail,
        ),)
    return StageRecord(
        stage="sync", input_sha256=board_sha, output_sha256=board_sha,
        checks_run=("schematic_pcb_sync",), violations=violations,
        passed=result.passed, detail=result.detail, timestamp=_now(),
    )


def build_board(board: str) -> bool:
    """Run the deterministic pipeline for one board. True when green so far."""
    from kicad_pipeline.pcb.builder import build_pcb, write_pcb
    from kicad_pipeline.placement_v2.compile import compile_constraints
    from kicad_pipeline.placement_v2.ledger import sha256_file
    from kicad_pipeline.placement_v2.part_rules import (
        apply_part_rules,
        load_part_rules,
    )
    from kicad_pipeline.placement_v2.persona_review import (
        PERSONAS,
        persona_instructions,
    )
    from kicad_pipeline.placement_v2.verifier import run_gate_a
    from kicad_pipeline.placement_v2.vision_gate import reviewer_instructions
    from kicad_pipeline.project_file import write_project_file
    from kicad_pipeline.requirements.decomposer import save_requirements
    from kicad_pipeline.schematic.builder import build_schematic, write_schematic

    module = importlib.import_module(BOARDS[board])
    requirements = module._build_requirements()
    out = _out_dir(board)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{board}_v2"
    pcb_path = out / f"{stem}.kicad_pcb"
    ledger_path = out / "build_ledger.jsonl"

    print(f"=== {board}: build ===")
    save_requirements(requirements, out / "requirements.json")

    # ---- build (v2 placement; certify/cells/floorplan -> ledger) ----------
    try:
        pcb = build_pcb(
            requirements, auto_route=False, placement_mode="v2",
            project_name=stem, v2_ledger_path=ledger_path,
        )
    except Exception as exc:
        print(f"  HALTED: {exc}")
        return False

    write_pcb(pcb, pcb_path, fill_zones=False)
    board_sha = sha256_file(pcb_path)

    # ---- schematic from the SAME requirements + sync hard gate ------------
    schematic = build_schematic(requirements, project_name=stem)
    write_schematic(schematic, out / f"{stem}.kicad_sch", project_name=stem)
    write_project_file(stem, out)
    ledger = _ledger(board)
    sync_rec = _sync_record(pcb, requirements, board_sha)
    ledger.append(sync_rec)
    print(f"  sync: {'PASS' if sync_rec.passed else 'FAIL — ' + sync_rec.detail}")

    # ---- Gate A re-derived from the written file ---------------------------
    constraints = compile_constraints(requirements, part_rules_path=PART_RULES_PATH)
    domains = apply_part_rules(
        load_part_rules(PART_RULES_PATH), requirements.components,
    ).domains
    report = run_gate_a(pcb_path, constraints, domains=domains)
    from kicad_pipeline.placement_v2.ledger import StageRecord

    ledger.append(StageRecord(
        stage="gate_a", input_sha256=board_sha, output_sha256=board_sha,
        checks_run=report.checks_run, violations=report.violations,
        passed=report.passed,
        detail=f"{len(report.violations)} violations",
        timestamp=_now(),
    ))
    print(f"  gate_a: {'PASS' if report.passed else 'FAIL'} "
          f"({len(report.violations)} violations)")
    for v in report.violations[:8]:
        print(f"    - [{v.severity.value}] {v.message}")

    # ---- renders + review requests -----------------------------------------
    renders = _render_views(board, pcb_path)
    print(f"  rendered {len(renders)}/4 views")
    requests = {
        "board": board,
        "board_sha256": board_sha,
        "renders": {k: str(v) for k, v in renders.items()},
        "prompts": {
            **{f"review_{p}": persona_instructions(p, renders) for p in PERSONAS},
            "gate_b": reviewer_instructions(renders),
        },
    }
    (out / "review_requests.json").write_text(
        json.dumps(requests, indent=2) + "\n", encoding="utf-8",
    )
    print(f"  review requests -> {out / 'review_requests.json'}")
    return sync_rec.passed and report.passed and len(renders) == 4


def ingest_verdict(board: str, stage: str, verdicts_path: Path) -> bool:
    """Parse and ledger one review stage's verdicts. True when passed."""
    from kicad_pipeline.placement_v2.ledger import sha256_file
    from kicad_pipeline.placement_v2.persona_review import (
        parse_persona_verdicts,
        persona_record,
    )
    from kicad_pipeline.placement_v2.vision_gate import gate_b_record, parse_verdicts

    pcb_path = _out_dir(board) / f"{board}_v2.kicad_pcb"
    board_sha = sha256_file(pcb_path)
    renders = {k: v for k, v in _render_paths(board).items() if v.exists()}
    raw = verdicts_path.read_text(encoding="utf-8")
    if stage == "gate_b":
        record = gate_b_record(parse_verdicts(raw), renders, board_sha, _now())
    elif stage in ("review_fab", "review_ee"):
        persona = stage.removeprefix("review_")
        verdicts = parse_persona_verdicts(persona, raw)
        record = persona_record(persona, verdicts, renders, board_sha, _now())
    else:
        raise SystemExit(f"unknown verdict stage {stage!r}")
    _ledger(board).append(record)
    mark = "PASS" if record.passed else "FAIL"
    print(f"  {board} {stage}: {mark} ({len(record.violations)} violations)")
    for v in record.violations:
        print(f"    - [{v.severity.value}] {v.message}")
    return record.passed


def board_status(board: str) -> bool:
    """Print ledger summary; True when presentable."""
    from kicad_pipeline.placement_v2.persona_review import PRESENTATION_STAGES

    ledger = _ledger(board)
    print(f"=== {board} ===")
    if not ledger.path.exists():
        print("  (no ledger — not built)")
        return False
    print("  " + ledger.summary().replace("\n", "\n  "))
    green = ledger.all_green(PRESENTATION_STAGES)
    missing = [s for s in PRESENTATION_STAGES if ledger.latest(s) is None]
    if missing:
        print(f"  missing stages: {', '.join(missing)}")
    print(f"  presentable: {'YES' if green else 'NO'}")
    return green


def present(board: str) -> bool:
    """List final artifacts; refuse when any stage is missing or red."""
    if not board_status(board):
        print(f"  REFUSED: {board} has un-reviewed or failing stages — "
              "not presenting renders")
        return False
    out = _out_dir(board)
    print("  artifacts:")
    for view, path in sorted(_render_paths(board).items()):
        if path.exists():
            print(f"    {view}: {path}")
    print(f"    pcb: {out / f'{board}_v2.kicad_pcb'}")
    print(f"    sch: {out / f'{board}_v2.kicad_sch'}")
    print(f"    ledger: {out / 'build_ledger.jsonl'}")
    return True


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "verdict", "status", "present"):
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "verdict":
        if len(sys.argv) != 5:
            raise SystemExit("usage: build_training_boards.py verdict "
                             "<board> <stage> <verdicts.json>")
        board = sys.argv[2]
        if board not in BOARDS:
            raise SystemExit(f"unknown board {board!r}; expected {sorted(BOARDS)}")
        return 0 if ingest_verdict(board, sys.argv[3], Path(sys.argv[4])) else 1

    boards = sys.argv[2:] or sorted(BOARDS)
    unknown = [b for b in boards if b not in BOARDS]
    if unknown:
        raise SystemExit(f"unknown board(s) {unknown}; expected {sorted(BOARDS)}")
    action = {"build": build_board, "status": board_status, "present": present}[cmd]
    results = {b: action(b) for b in boards}
    print()
    for b, ok in results.items():
        print(f"{b}: {'OK' if ok else 'NOT GREEN'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
