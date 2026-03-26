#!/usr/bin/env python3
"""PCB Review-Iterate Loop — External orchestrator that drives Claude Code agents.

Mechanically enforces: generate → render → review → fix → repeat.
No single agent controls the loop — this script does.

Usage:
  # Single board by path
  python scripts/pcb-review-loop.py output/train_mcu_core/train_mcu_core.kicad_pcb

  # Single board by shorthand
  python scripts/pcb-review-loop.py mcu
  python scripts/pcb-review-loop.py relay --max-iterations 50 --human-every 10

  # Multiple boards (comma-separated)
  python scripts/pcb-review-loop.py relay,power,mcu

  # All 5 training boards
  python scripts/pcb-review-loop.py all

  # Standalone project board
  python scripts/pcb-review-loop.py output/nl-s-3c-placement.kicad_pcb --human-every 10

Board selectors: mcu, relay, power, analog, ethernet, all

Human sign-off is required:
  - Every N iterations (--human-every, default 5)
  - Before commit (always)
  - When the review finds zero issues (potential completion)
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
RENDERS_DIR = Path(".claude/review-renders")
EVIDENCE_FILE = Path(".claude/last_review.json")
LOG_FILE = Path(".claude/review-loop.log")
STATE_FILE = Path(".claude/review-loop-state.json")

# Training board shorthand → path (relative to kicad-ai-workflow root)
TRAINING_BOARDS: dict[str, str] = {
    "mcu": "output/train_mcu_core/train_mcu_core.kicad_pcb",
    "relay": "output/train_relay/train_relay.kicad_pcb",
    "power": "output/train_power/train_power.kicad_pcb",
    "analog": "output/train_analog_input/train_analog_input.kicad_pcb",
    "ethernet": "output/train_ethernet/train_ethernet.kicad_pcb",
}


def resolve_boards(selector: str) -> list[str]:
    """Resolve a board selector to a list of .kicad_pcb paths.

    Accepts:
      - 'all' → all 5 training boards
      - 'mcu' → single training board
      - 'relay,power,mcu' → multiple training boards
      - 'output/foo/bar.kicad_pcb' → literal path
    """
    if selector == "all":
        boards = list(TRAINING_BOARDS.values())
        missing = [b for b in boards if not Path(b).exists()]
        if missing:
            print(f"WARNING: Missing boards: {missing}")
            boards = [b for b in boards if Path(b).exists()]
        return boards

    # Comma-separated list of selectors
    if "," in selector:
        result: list[str] = []
        for part in selector.split(","):
            result.extend(resolve_boards(part.strip()))
        return result

    # Known shorthand
    if selector in TRAINING_BOARDS:
        path = TRAINING_BOARDS[selector]
        if not Path(path).exists():
            print(f"ERROR: Board not found: {path}")
            sys.exit(1)
        return [path]

    # Literal path
    if Path(selector).exists():
        return [selector]

    # Fuzzy match — check if selector is a substring of any known board
    matches = [k for k in TRAINING_BOARDS if selector in k]
    if len(matches) == 1:
        return [TRAINING_BOARDS[matches[0]]]

    print(f"ERROR: Unknown board selector '{selector}'")
    print(f"Known boards: {', '.join(TRAINING_BOARDS.keys())}, all")
    sys.exit(1)

# Personas with specific review instructions
FABRICATOR_PROMPT = """\
You are a PCB fabricator reviewing a board for manufacturability.
You have been given 2D and 3D renders of a KiCad PCB.

READ each image file listed below, then evaluate against this checklist:

## Fabricator Checklist
1. **Board bounds**: Are ALL components (including pad extents) within the board outline?
2. **Courtyard overlaps**: Do any component courtyards intersect?
3. **Decoupling proximity**: Are bypass caps within 3-5mm of their IC power pins?
4. **Crystal proximity**: Is the crystal within 5mm of the MCU oscillator pins?
5. **Connector edge distance**: Are all connectors within 5mm of the board edge?
6. **Component spacing**: Minimum 0.2mm between component bodies?
7. **Silkscreen clarity**: Are reference designators readable and not overlapping pads?
8. **3D body overlap**: Do any 3D component bodies physically overlap?
9. **3D body flatness**: Are all components flat on the PCB surface (no floating)?
10. **Orientation consistency**: Are similar components (e.g., all 0805 caps) oriented consistently?
11. **Thermal relief**: Are large copper pours properly relieved?
12. **Assembly feasibility**: Could a pick-and-place machine place all SMD components?
13. **Through-hole access**: Are all THT pads accessible for soldering?

For EACH issue found, report:
- Component reference (e.g., R1, U3)
- Issue type (from checklist above)
- Severity: CRITICAL (board won't work), MAJOR (assembly problem), MINOR (cosmetic)
- Suggested fix with specific coordinates if applicable

Output as JSON:
{{"issues": [{{"ref": "R1", "type": "courtyard_overlap", "severity": "MAJOR", "description": "...", "fix": "..."}}], "pass": false}}

If no issues: {{"issues": [], "pass": true}}

Image files to review:
"""

EE_PROMPT = """\
You are an electrical engineer reviewing a PCB layout for signal integrity and electrical correctness.
You have been given 2D and 3D renders of a KiCad PCB.

READ each image file listed below, then evaluate against this checklist:

## Electrical Engineer Checklist
1. **Decoupling placement**: Are bypass caps on the SAME SIDE as the IC and close to power pins?
2. **Signal flow**: Do signal paths follow logical flow (input → processing → output)?
3. **Analog/digital separation**: Are analog and digital sections physically separated?
4. **Power flow direction**: Does power flow from regulators toward loads?
5. **Ground return paths**: Are high-speed signal returns short and direct?
6. **EMI considerations**: Are clock sources away from board edges and connectors?
7. **Crystal loop area**: Is the crystal oscillator loop area minimized?
8. **Voltage isolation**: Are different voltage domains physically separated?
9. **Hot loop minimization**: Are switching regulator hot loops tight?
10. **ESD protection**: Are TVS/ESD components near their protected connectors?
11. **Antenna clearance**: Is the RF antenna area free of copper/components?
12. **Pull-up/pull-down proximity**: Are pull resistors near their driven ICs?

For EACH issue found, report:
- Component reference or net name
- Issue type (from checklist above)
- Severity: CRITICAL (won't function), MAJOR (performance issue), MINOR (suboptimal)
- Suggested fix

Output as JSON:
{{"issues": [{{"ref": "U1", "type": "decoupling_placement", "severity": "CRITICAL", "description": "...", "fix": "..."}}], "pass": false}}

If no issues: {{"issues": [], "pass": true}}

Image files to review:
"""

COMPONENT_3D_PROMPT = """\
You are verifying 3D model correctness for every component on a PCB.
READ each 3D render image below, then for EVERY visible component check:

1. **Body-to-pad alignment**: Is the 3D body centered on its pads?
2. **Component type match**: Does the 3D model look like the right component type?
   (e.g., a capacitor should look like a capacitor, not a resistor)
3. **Rotation correctness**: Is pin 1 / polarity mark on the correct side?
4. **Body dimensions**: Is the 3D body proportional to the footprint?
5. **Z-position**: Is the component sitting flat on the board (not floating)?
6. **Missing models**: Are any footprints missing their 3D model?

List EVERY component you can see and its status. Output as JSON:
{{"components": [{{"ref": "U1", "status": "PASS"}}, {{"ref": "C1", "status": "FAIL", "issue": "floating 1mm above board"}}], "issues_count": 1}}

Image files to review:
"""

FIX_PROMPT_TEMPLATE = """\
You are fixing PCB placement issues in a KiCad pipeline project.

The following issues were found by fabricator and EE reviewers:

{issues_json}

Fix these issues by editing the relevant source files. The PCB is generated by code in:
- src/kicad_pipeline/optimization/ (placement optimizer, review agent, scoring)
- src/kicad_pipeline/pcb/ (builder, footprints, placement, zones)

Rules:
- Fix the ROOT CAUSE in the generator code, not the .kicad_pcb file directly
- Run the relevant tests after your fix: pytest tests/ -x --tb=short -q
- Do NOT commit — the orchestrator handles commits
- Do NOT declare "done" — just make the fix and stop
- Focus on CRITICAL and MAJOR issues first

After making fixes, regenerate the board:
  pytest tests/integration/test_placement_visual.py -x --tb=short -s 2>&1 | tail -20

Board file: {board_path}
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    iteration: int
    timestamp: str
    fab_issues: list[dict] = field(default_factory=list)
    ee_issues: list[dict] = field(default_factory=list)
    component_issues: int = 0
    fab_pass: bool = False
    ee_pass: bool = False
    component_pass: bool = False
    fix_applied: bool = False
    human_verdict: str | None = None  # "approved", "rejected", "feedback:..."

    @property
    def all_pass(self) -> bool:
        return self.fab_pass and self.ee_pass and self.component_pass

    @property
    def critical_count(self) -> int:
        return sum(
            1 for i in self.fab_issues + self.ee_issues
            if i.get("severity") == "CRITICAL"
        )

    @property
    def major_count(self) -> int:
        return sum(
            1 for i in self.fab_issues + self.ee_issues
            if i.get("severity") == "MAJOR"
        )


@dataclass
class LoopState:
    board_path: str
    iteration: int = 0
    max_iterations: int = 20
    human_every: int = 5
    results: list[dict] = field(default_factory=list)
    started: str = ""
    last_human_iteration: int = 0

    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.__dict__, indent=2) + "\n")

    @classmethod
    def load(cls) -> LoopState | None:
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                state = cls(**{k: v for k, v in data.items()
                              if k in cls.__dataclass_fields__})
                return state
            except (json.JSONDecodeError, TypeError):
                return None
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Print and log to file."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_claude(prompt: str, timeout: int = 300) -> str:
    """Run Claude Code in non-interactive print mode and return output."""
    try:
        result = subprocess.run(
            [CLAUDE_BIN, "-p", "--output-format", "text", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return '{"error": "Claude timed out"}'
    except FileNotFoundError:
        log(f"ERROR: Claude CLI not found at {CLAUDE_BIN}")
        log("Set CLAUDE_BIN env var or ensure 'claude' is in PATH")
        sys.exit(1)


def run_shell(cmd: str, timeout: int = 120) -> tuple[int, str]:
    """Run a shell command directly (no Claude needed for deterministic steps)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
            cwd=os.getcwd(),
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 1, "Command timed out"


def render_board(board_path: str) -> dict[str, Path]:
    """Render 2D and 3D images directly via kicad-image-gen CLI.

    Returns dict of render type → file path.
    Renders are placed in the board's own directory (next to the .kicad_pcb file).
    No Claude needed — deterministic CLI tool.
    """
    board_dir = Path(board_path).parent
    render_dir = board_dir  # renders go alongside the .kicad_pcb
    render_dir.mkdir(parents=True, exist_ok=True)
    renders: dict[str, Path] = {}

    views = [
        ("2d", [], "2d_top.png"),
        ("3d", ["--view", "top"], "3d_top.png"),
        ("3d", ["--view", "iso"], "3d_iso.png"),
        ("3d", ["--view", "iso-back"], "3d_iso_back.png"),
        ("3d", ["--view", "front"], "3d_front.png"),
    ]

    for mode, extra_args, filename in views:
        out_path = render_dir / filename
        cmd = ["kicad-image-gen", mode, board_path, "-o", str(out_path)]
        cmd.extend(extra_args)
        log(f"  Rendering {filename}...")
        rc, output = run_shell(" ".join(cmd), timeout=60)
        if rc == 0 and out_path.exists():
            renders[filename] = out_path
        else:
            log(f"  WARNING: Failed to render {filename}: {output[:200]}")

    return renders


def parse_json_from_response(response: str) -> dict:
    """Extract JSON from Claude response (may have markdown fences)."""
    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    import re
    match = re.search(r"```json\s*\n(.*?)\n\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"error": "Could not parse JSON from response", "raw": response[:500]}


def ask_human(iteration: int, result: IterationResult, board_path: str = "") -> str:
    """Present results and get human verdict."""
    board_name = Path(board_path).stem if board_path else "unknown"
    board_dir = Path(board_path).parent if board_path else RENDERS_DIR
    print("\n" + "=" * 60)
    print(f"  HUMAN REVIEW REQUIRED — {board_name} — Iteration {iteration}")
    print("=" * 60)
    print(f"\n  Fabricator: {'PASS' if result.fab_pass else 'FAIL'}"
          f" ({len(result.fab_issues)} issues)")
    print(f"  EE Review:  {'PASS' if result.ee_pass else 'FAIL'}"
          f" ({len(result.ee_issues)} issues)")
    print(f"  3D Verify:  {'PASS' if result.component_pass else 'FAIL'}"
          f" ({result.component_issues} issues)")
    print(f"  Critical: {result.critical_count}  Major: {result.major_count}")

    if result.fab_issues or result.ee_issues:
        print("\n  Top issues:")
        for issue in (result.fab_issues + result.ee_issues)[:5]:
            sev = issue.get("severity", "?")
            ref = issue.get("ref", "?")
            desc = issue.get("description", issue.get("type", "?"))
            print(f"    [{sev}] {ref}: {desc}")

    print(f"\n  Renders saved to: {board_dir}/")
    print("  Open them to inspect the board visually.\n")

    while True:
        choice = input("  [a]pprove / [r]eject / [f]eedback / [c]ontinue N more / [q]uit: ").strip().lower()
        if choice == "a":
            return "approved"
        elif choice == "r":
            return "rejected"
        elif choice == "q":
            return "quit"
        elif choice.startswith("c"):
            parts = choice.split()
            n = int(parts[1]) if len(parts) > 1 else 5
            return f"continue:{n}"
        elif choice.startswith("f"):
            feedback = input("  Your feedback: ").strip()
            return f"feedback:{feedback}"
        else:
            print("  Invalid choice. Use a/r/f/c/q.")


def record_evidence(rendered_2d: bool, rendered_3d: bool, dual_persona: bool,
                    human_signoff: str | None) -> None:
    """Write review evidence file for the commit gate hook."""
    EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "timestamp": time.time(),
        "rendered_2d": rendered_2d,
        "rendered_3d": rendered_3d,
        "dual_persona_review": dual_persona,
        "human_signoff": human_signoff,
        "source": "pcb-review-loop.py",
    }
    EVIDENCE_FILE.write_text(json.dumps(evidence, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Main loop phases
# ---------------------------------------------------------------------------

def phase_render(board_path: str) -> dict[str, Path]:
    """Phase 1: Render all views."""
    log("PHASE 1: Rendering board images")
    renders = render_board(board_path)
    if not renders:
        log("ERROR: No renders produced. Is kicad-image-gen installed?")
        sys.exit(1)
    log(f"  Produced {len(renders)} renders")
    return renders


def phase_fab_review(renders: dict[str, Path]) -> tuple[bool, list[dict]]:
    """Phase 2: Fabricator persona review."""
    log("PHASE 2: Fabricator review")
    file_list = "\n".join(f"- {path}" for path in renders.values())
    prompt = FABRICATOR_PROMPT + file_list
    response = run_claude(prompt, timeout=180)
    result = parse_json_from_response(response)
    issues = result.get("issues", [])
    passed = result.get("pass", len(issues) == 0)
    log(f"  Fabricator: {'PASS' if passed else 'FAIL'} ({len(issues)} issues)")
    for issue in issues[:3]:
        log(f"    [{issue.get('severity', '?')}] {issue.get('ref', '?')}: "
            f"{issue.get('description', issue.get('type', '?'))}")
    return passed, issues


def phase_ee_review(renders: dict[str, Path]) -> tuple[bool, list[dict]]:
    """Phase 3: EE persona review."""
    log("PHASE 3: EE review")
    file_list = "\n".join(f"- {path}" for path in renders.values())
    prompt = EE_PROMPT + file_list
    response = run_claude(prompt, timeout=180)
    result = parse_json_from_response(response)
    issues = result.get("issues", [])
    passed = result.get("pass", len(issues) == 0)
    log(f"  EE: {'PASS' if passed else 'FAIL'} ({len(issues)} issues)")
    for issue in issues[:3]:
        log(f"    [{issue.get('severity', '?')}] {issue.get('ref', '?')}: "
            f"{issue.get('description', issue.get('type', '?'))}")
    return passed, issues


def phase_3d_verify(renders: dict[str, Path]) -> tuple[bool, int]:
    """Phase 4: Component-by-component 3D verification."""
    log("PHASE 4: Component 3D verification")
    # Only send 3D renders for this check
    three_d = {k: v for k, v in renders.items() if "3d" in k}
    if not three_d:
        log("  SKIP: No 3D renders available")
        return True, 0
    file_list = "\n".join(f"- {path}" for path in three_d.values())
    prompt = COMPONENT_3D_PROMPT + file_list
    response = run_claude(prompt, timeout=180)
    result = parse_json_from_response(response)
    issues_count = result.get("issues_count", 0)
    passed = issues_count == 0
    log(f"  3D Verify: {'PASS' if passed else 'FAIL'} ({issues_count} component issues)")
    return passed, issues_count


def phase_fix(board_path: str, fab_issues: list[dict], ee_issues: list[dict],
              human_feedback: str | None = None) -> bool:
    """Phase 5: Apply fixes via Claude agent."""
    all_issues = fab_issues + ee_issues
    if not all_issues and not human_feedback:
        return False

    log("PHASE 5: Applying fixes")

    issues_json = json.dumps(all_issues, indent=2)
    if human_feedback:
        issues_json += f"\n\nHuman feedback: {human_feedback}"

    prompt = FIX_PROMPT_TEMPLATE.format(
        issues_json=issues_json,
        board_path=board_path,
    )
    response = run_claude(prompt, timeout=600)
    log(f"  Fix agent response length: {len(response)} chars")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_board(board_path: str, max_iterations: int, human_every: int,
              resume: bool, skip_3d: bool) -> None:
    """Run the review-iterate loop for a single board."""
    if not Path(board_path).exists():
        log(f"ERROR: Board file not found: {board_path}")
        return

    # Load or create state
    state: LoopState | None = None
    if resume:
        state = LoopState.load()
        if state:
            log(f"Resuming from iteration {state.iteration}")

    if state is None:
        state = LoopState(
            board_path=board_path,
            max_iterations=max_iterations,
            human_every=human_every,
            started=datetime.now().isoformat(),
        )

    log(f"PCB Review Loop starting: {board_path}")
    log(f"Max iterations: {state.max_iterations}, Human every: {state.human_every}")

    human_feedback: str | None = None

    while state.iteration < state.max_iterations:
        state.iteration += 1
        log(f"\n{'='*60}")
        log(f"ITERATION {state.iteration}/{state.max_iterations}")
        log(f"{'='*60}")

        # Phase 1: Render
        renders = phase_render(board_path)

        # Phase 2: Fabricator review
        fab_pass, fab_issues = phase_fab_review(renders)

        # Phase 3: EE review
        ee_pass, ee_issues = phase_ee_review(renders)

        # Phase 4: 3D verification (optional)
        if skip_3d:
            comp_pass, comp_issues = True, 0
        else:
            comp_pass, comp_issues = phase_3d_verify(renders)

        # Build result
        result = IterationResult(
            iteration=state.iteration,
            timestamp=datetime.now().isoformat(),
            fab_issues=fab_issues,
            ee_issues=ee_issues,
            component_issues=comp_issues,
            fab_pass=fab_pass,
            ee_pass=ee_pass,
            component_pass=comp_pass,
        )

        # Record partial evidence (renders done, reviews done)
        record_evidence(
            rendered_2d=bool(any("2d" in k for k in renders)),
            rendered_3d=bool(any("3d" in k for k in renders)),
            dual_persona=True,
            human_signoff=None,  # Not yet
        )

        # Decide if human review needed
        needs_human = False
        since_human = state.iteration - state.last_human_iteration

        if result.all_pass:
            log("ALL REVIEWS PASSED — requesting human sign-off")
            needs_human = True
        elif since_human >= state.human_every:
            log(f"Human check-in due (every {state.human_every} iterations)")
            needs_human = True

        if needs_human:
            verdict = ask_human(state.iteration, result, board_path)
            result.human_verdict = verdict
            state.last_human_iteration = state.iteration

            if verdict == "approved":
                log("HUMAN APPROVED — recording evidence and stopping")
                record_evidence(
                    rendered_2d=True, rendered_3d=True,
                    dual_persona=True, human_signoff="approved",
                )
                state.results.append(result.__dict__)
                state.save()
                print(f"\nReview evidence written to {EVIDENCE_FILE}")
                print("You can now `git commit` — the gate hook will allow it.")
                return

            elif verdict == "rejected":
                log("HUMAN REJECTED — reverting last fix and stopping")
                run_shell("git checkout -- src/")
                state.results.append(result.__dict__)
                state.save()
                return

            elif verdict == "quit":
                log("HUMAN QUIT — saving state for resume")
                state.results.append(result.__dict__)
                state.save()
                return

            elif verdict.startswith("continue:"):
                extra = int(verdict.split(":")[1])
                state.max_iterations = state.iteration + extra
                log(f"Continuing for {extra} more iterations (new max: {state.max_iterations})")
                human_feedback = None

            elif verdict.startswith("feedback:"):
                human_feedback = verdict.split(":", 1)[1]
                log(f"Human feedback: {human_feedback}")

        # Phase 5: Fix (if there are issues or human feedback)
        if not result.all_pass or human_feedback:
            fix_applied = phase_fix(
                board_path, fab_issues, ee_issues, human_feedback,
            )
            result.fix_applied = fix_applied
            human_feedback = None  # Consumed

        state.results.append(result.__dict__)
        state.save()

        # If all passed but human hasn't been asked yet (shouldn't happen, but safety)
        if result.all_pass and not needs_human:
            log("All reviews passed — will ask human next iteration")

    # Max iterations reached
    log(f"\nMax iterations ({state.max_iterations}) reached for {board_path}.")
    log("Run with --resume to continue, or review renders manually.")
    state.save()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCB Review-Iterate Loop — mechanically enforced review workflow"
    )
    parser.add_argument(
        "board", nargs="?", default=None,
        help="Board selector: path to .kicad_pcb, shorthand (mcu/relay/power/analog/ethernet), "
             "comma list (relay,power), or 'all'",
    )
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="Max iterations per board (default: 20)")
    parser.add_argument("--human-every", type=int, default=5,
                        help="Ask human every N iterations (default: 5)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved state")
    parser.add_argument("--skip-3d", action="store_true",
                        help="Skip component-by-component 3D verification")
    parser.add_argument("--list", action="store_true", dest="list_boards",
                        help="List available training boards and exit")
    args = parser.parse_args()

    if args.list_boards:
        print("Available training boards:")
        for name, path in TRAINING_BOARDS.items():
            exists = "OK" if Path(path).exists() else "MISSING"
            print(f"  {name:10s} → {path}  [{exists}]")
        return

    if args.board is None:
        parser.error("board selector is required (e.g., 'mcu', 'all', or a .kicad_pcb path)")

    boards = resolve_boards(args.board)
    log(f"Boards to process: {len(boards)}")
    for b in boards:
        log(f"  - {b}")

    for i, board_path in enumerate(boards):
        if len(boards) > 1:
            log(f"\n{'#' * 60}")
            log(f"# BOARD {i + 1}/{len(boards)}: {Path(board_path).stem}")
            log(f"{'#' * 60}")
        run_board(
            board_path=board_path,
            max_iterations=args.max_iterations,
            human_every=args.human_every,
            resume=args.resume,
            skip_3d=args.skip_3d,
        )

    log(f"\nAll {len(boards)} board(s) processed.")


if __name__ == "__main__":
    main()
