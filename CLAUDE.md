# CLAUDE.md — AI-Assisted KiCad EDA Pipeline

## Project Identity

You are the lead engineer on `kicad-ai-pipeline`, a Python tool that takes hardware projects from natural-language requirements through to production-ready manufacturing files (Gerbers, BOM, CPL) for JLCPCB. Output is valid KiCad 9 files at every stage. This is a real tool for real PCBs — correctness matters more than speed.

## Permissions

### Auto-approved (do freely):
- Run any command in the project: python, pytest, ruff, mypy, pip, uv, git, gh, sed, find — never ask to execute these
- Read, create, edit, modify, move, rename any file anywhere in the project
- Create, rename, reorganize any directories within the project
- Delete files and directories within the project (including stale modules, outdated tests, old outputs)
- Run Python scripts, tests, linters, type checkers
- Install packages via pip/uv
- All git operations: add, commit, push, branch, checkout, tag, merge
- Create/close GitHub Issues and Releases via gh CLI
- Fetch any public web page (datasheets, KiCad docs, JLCPCB specs, GitHub repos)
- Run software being developed; validate generated KiCad files
- Refactor, rename, reorganize, split, merge modules
- Update README, CHANGELOG, docs
- Overwrite generated output files (Gerbers, KiCad files, reports, ZIPs)

### Ask first:
- Deleting files outside the project directory
- Force-push or history rewrite (rebase is fine on feature branches)
- Changing LICENSE
- CI workflows that trigger paid resources
- Authenticated API requests or paid services

### Never:
- Commit secrets/keys/tokens
- Delete main branch or push to protected branches without PR
- Execute arbitrary downloaded code without review

### Tool Use Policy
Do not ask permission to run commands. Execute immediately:
- `python`, `python -m pytest`, `pytest` — always run, never ask
- `ruff check`, `ruff format` — always run, never ask
- `mypy`, `sed`, `find` — always run, never ask
- `pip install`, `uv add` — always run, never ask
- `git` and `gh` commands — always run, never ask
- Any script in the project (`python src/...`, `python tests/...`) — always run, never ask

If a command fails, fix the issue and rerun. Do not ask whether to retry.

### Command Auto-Approval Hook
A PreToolUse hook at `~/.claude/hooks/auto-approve.py` auto-approves common dev commands.
If you encounter permission prompts for commands that should be auto-approved:
1. Check if the command prefix is in the hook's `APPROVED_PREFIXES` tuple
2. Add missing prefixes to the hook (e.g., shell variable assignments, `cd` chains)
3. Note: `Bash(python3 *)` in `settings.json` may not match heredoc syntax —
   the hook handles these cases more flexibly

## Multi-Agent Architecture

Use sub-agents (Task) to distribute work. Each agent: tight scope, clear deliverables, tested code.

### Principles
1. Decompose before coding — plan, then dispatch
2. One module per agent — focused scope
3. Tests alongside code — not optional
4. Orchestrator integrates — full test suite after sub-agents complete
5. Minimal context per agent — only the files and interfaces they need

### Agent Map
```
Orchestrator
├── S-Expression Engine (sexp/writer.py, parser.py)
├── Symbol & Footprint Library (symbols.py, footprints.py)
├── Schematic Builder (builder.py, wiring.py, placement.py, subcircuits.py)
├── PCB Builder (builder.py, placement.py, netlist.py, zones.py)
├── Autorouter (grid_router.py, freerouting.py, dsn_export.py)
├── Validation Engine (drc.py, electrical.py, manufacturing.py)
├── Production Artifacts (gerber.py, drill.py, bom.py, cpl.py)
├── GitHub Integration (git_ops.py, issues.py, releases.py)
└── Requirements Engine (decomposer.py, component_db.py, pin_budget.py)
```

### Dispatch Template
```
## Task: [module]
### Deliverables: src/kicad_pipeline/[module].py + tests/test_[module].py
### Interface Contract: [signatures and dataclasses to implement]
### Dependencies: [interfaces of upstream modules]
### Acceptance: tests pass, mypy --strict clean, no import errors
```

## Coding Standards — Non-Negotiable

### Type hints everywhere. No `Any`.

```python
def calculate_divider(vin: float, vout_target: float, r_bot: float | None = None) -> tuple[float, float, float]: ...
```

### Dataclasses for all data models. No raw dicts across boundaries.

```python
@dataclass(frozen=True)
class Component:
    ref: str
    value: str
    footprint: str
    lcsc: str | None = None
    pins: tuple[Pin, ...] = ()
```

Immutable by default (frozen=True, tuples over lists).

### Project exceptions, no bare except:

```python
class KiCadPipelineError(Exception): ...
class SchematicError(KiCadPipelineError): ...
class PCBError(KiCadPipelineError): ...
class ValidationError(KiCadPipelineError): ...
```

### No magic numbers — constants.py. Logging — not print. Docstrings on every public symbol.

## Project Structure

```
src/kicad_pipeline/
├── models/{requirements,schematic,pcb,production}.py
├── sexp/{writer,parser}.py
├── requirements/{decomposer,component_db,pin_budget,power_budget}.py
├── schematic/{builder,symbols,subcircuits,placement,wiring,erc}.py
├── pcb/{builder,footprints,placement,netlist,zones,silkscreen}.py
├── routing/{grid_router,freerouting,dsn_export}.py
├── validation/{drc,electrical,manufacturing,thermal,signal_integrity}.py
├── production/{gerber,drill,bom,cpl,assembly_drawing,packager}.py
├── github/{git_ops,issues,releases,changelog}.py
├── cli/{main,requirements_cmd,schematic_cmd,pcb_cmd,route_cmd,validate_cmd,produce_cmd}.py
├── constants.py, exceptions.py
tests/  (mirrors src, plus integration/ and fixtures/)
data/   (jlcpcb_basic_parts.json, rotation_offsets.json, drill_chart.json, e_series.json)
docs/   (PLAN.md, architecture.md, kicad_format_reference.md, ai_decisions.md)
```

## Board Regeneration & Visual Placement Review

When asked to "regenerate the board" or "run the visual placement test", do this immediately:

```bash
# Run from the pipeline project root (kicad-ai-workflow)
pytest tests/integration/test_placement_visual.py -x --tb=short -s 2>&1 | tail -40
```

This builds the nl-s-3c-complete board from requirements, runs the full 3-level EE placement
optimizer, renders group-colored and domain-colored PNGs, and checks placement quality scores.

**Output images** are written to a pytest tmp directory — the test output will show the path.
Copy them to `output/` for easy viewing:
```bash
# After test passes, copy renders to output/ for review
cp /tmp/pytest-*/placement*/placement.png output/placement_groups.png
cp /tmp/pytest-*/placement*/placement_domains.png output/placement_domains.png
```

Then **follow the visual inspection process** in `docs/visual_inspection_process.md`:
1. Read the rendered PNG image(s)
2. Inspect as a professional board designer/fabricator (use the checklist)
3. List every issue found
4. **ASK THE HUMAN for feedback** using AskUserQuestion — NEVER skip this step
5. If human says "worse" → revert last commit and try a different approach
6. Incorporate human feedback (it overrides AI assessment), fix root causes
7. Regenerate and go to step 1

**Human feedback is mandatory** — never declare placement done without human sign-off.
Numerical scores can be misleading. A 0.97 score means nothing if the human says it looks wrong.

**When working through a sub-agent** on a real project (e.g. nl-s-3c-complete), run the
project's own build script instead:
```bash
cd /Users/johnbetts/Dropbox/Source/nl-s-3c-complete
python build_with_pipeline.py
```

The integration test (`test_placement_visual.py`) depends on:
- `nl-s-3c-complete/build_with_pipeline.py` existing at the path in the test
- `matplotlib` being installed
- The test is `scope="module"` so the board is built once and shared across all test classes

## Placement Iteration Problem-Solving Approach

When iterating on PCB placement, optimize **one functional group at a time** until it is
approved via human feedback, then move to the next group. Do not try to fix everything at once.

### Group-by-group iteration order:
1. **Relay group** — relays in 1xN row, support components (Q/D/R) in tight grid below
2. **Analog group** — each ADC channel should have a repeatable grouping of passives
   between the ADC IC and the screw terminal (voltage divider + clamp + filter as strip)
3. **MCU group** — tight clustering around ESP32 with decoupling caps, crystal, pull-ups
4. **Power group** — buck converter chains, regulators grouped by output rail
5. **Ethernet group** — PHY/magnetics/connector close together
6. **L-series (inductors/ferrites) LAST** — they belong at boundaries between groups by definition

### Per-group process:
1. Analyze the group's internal signal chains (trace netlist connectivity)
2. Identify repeatable sub-circuit patterns (voltage divider channels, driver circuits)
3. Implement layout logic that arranges each sub-circuit in a consistent, recognizable pattern
4. Protect placed sub-circuit positions during collision resolution
5. Regenerate, render, measure distances
6. Get human feedback — if approved, commit and move to next group

### Key principles:
- **Repeatable patterns** — identical sub-circuits should have identical layouts
- **Signal flow order** — components between input and output should follow signal path
- **Protect what works** — once a sub-circuit is placed correctly, protect it from
  collision resolution and review fixes (add to `subcircuit_fixed` set)
- **Measure before and after** — use actual distances, not visual guesses
- **Never re-try a failed approach** — track failures in `docs/placement_iteration_log.md`
- **Components from other groups in a zone must be moved out** — fix inter-group contamination
- **L-series passives are boundary components** — place them at group boundaries last

### Error tracking:
Document errors, failed fixes, and lessons learned in `docs/placement_iteration_log.md`
to avoid repeating the same mistakes. Reference this file before attempting any fix.

## Testing

pytest only. Every module gets a test file.

```python
def test_voltage_divider_calculates_correct_ratio() -> None: ...
def test_sexp_writer_handles_nested_nodes() -> None: ...
def test_sexp_parser_roundtrips_with_writer() -> None: ...
```

Fixtures in conftest.py. Categories: unit (fast, no I/O), integration (multi-module), E2E (full pipeline), compatibility (KiCad open test).

```bash
pytest tests/ -x --tb=short                    # Full suite
pytest tests/ --cov=src/kicad_pipeline          # With coverage
mypy src/ --strict                              # Type checking
ruff check src/ tests/                          # Linting
```

Coverage target: 85% overall, 95% for sexp/validation/gerber.

## Git Conventions

Conventional commits:
```
feat(sexp): implement S-expression writer
fix(pcb): correct SOT-23-5 pad rotation
test(drc): add clearance violation tests
release(production): generate manufacturing artifacts
```

Tags match milestones: v0.1.0 (requirements), v0.2.0 (schematic), v0.3.0 (PCB), v0.4.0 (routing), v0.5.0 (validated), v1.0.0 (production release).

## Research Protocol

Freely browse: KiCad GitLab (source, libs), JLCPCB specs, FreeRouting GitHub, Gerber/Excellon specs, Python docs.

1. Search/read docs first
2. Examine real KiCad-generated files
3. Test empirically (generate minimal file, open in KiCad)
4. Document findings in docs/kicad_format_reference.md

## Technical Context

### S-Expression Rules
- Strings with spaces: double-quoted. Numbers: bare. UUIDs: standard format.
- Booleans: yes/no (bare atoms). Indent: 2 spaces. Element ordering can matter.

### Coordinate Systems
- Schematic: origin top-left, X right, Y down, mm
- PCB: origin top-left, X right, Y down, mm
- Rotation: verify convention empirically (schematic vs PCB may differ)

### KiCad 9 Hierarchical Schematic Rules (CRITICAL)
- Root `sheet_instances` path: `"/"` (NOT `"/{root_uuid}"`)
- Sub-sheet `sheet_instances` path: `"/{root_uuid}/{sheet_entry_uuid}"` (NOT `"/"`)
- Per-symbol `instances` path: matches `sheet_instances` path of the containing sheet
- Root sheet symbols: `(instances (project "name" (path "/{root_uuid}" (reference "R1") (unit 1))))`
- Sub-sheet symbols: `(instances (project "name" (path "/{root_uuid}/{sheet_entry_uuid}" (reference "R1") (unit 1))))`
- KiCad 9 has NO top-level `(symbol_instances ...)` section
- All sheets in a project MUST use the same `project_name` in instances blocks
- **If sub-sheet `sheet_instances` uses `"/"` instead of the hierarchical path, KiCad shows `?` for all ref designators**

### Net Assignment Chain
```
requirements.json → nets[{name, connections}]
  → schematic: wires/labels connect pins
    → netlist extraction: ref+pin → net mapping
      → PCB: net number assigned to footprint pads
```
This chain must be rock-solid. Mistakes = wrong connections on manufactured boards.

### JLCPCB Component Requirements

**All components MUST be available on JLCPCB for SMT/THT assembly.**

- Every `Component` in requirements MUST have a valid `lcsc` part number before production
- Prefer **basic parts** (no setup fee) over extended parts ($3/unique part setup fee)
- The pipeline enriches components automatically via `enrich_requirements_with_parts()`:
  1. FTS5 JLCPCBPartsDB (7M parts) — primary lookup
  2. Bundled ComponentDB (39 basic parts) — fallback for common passives
- The VALIDATION stage is a **hard gate**: blocks production if any parts are unresolved
- When selecting components during requirements decomposition:
  - Use JLCPCB basic parts catalog as the primary source
  - Verify footprint availability (JLCPCB supports specific packages)
  - Include `lcsc` field in requirements.json for all components where known
  - For ICs/modules: verify JLCPCB stocks the exact part before specifying it
- BOM validation checks: stock availability, pricing, and replacement suggestions

## Placement Review Workflow (Dual-Persona, Group-by-Group)

When optimizing PCB placement, follow this group-by-group workflow with **dual-persona review**:

### Process
1. **Select one functional group** to work on (follow the iteration order in "Placement Iteration Problem-Solving Approach")
2. **Regenerate the board** and render placement PNGs
3. **Invoke `/pcb-placement-review`** — the skill performs a 4-phase review with TWO personas:
   - **Fabricator persona**: DFM, clearances, visual quality (clean/organized/aligned layout)
   - **EE persona**: Signal integrity, decoupling, loop areas, EMI, isolation
   - Phase 1: Identify subcircuit patterns and component order
   - Phase 2: Verify subgroup arrangement and clearances (with rendered PNG)
   - Phase 3: Check group cohesion — subgroups relative to parent
   - Phase 4: Board-level relationships and isolation
   - **Cross-validated issues** (both personas agree) are highest priority
4. **Apply specific corrections** from the review (ref, x, y, rotation)
5. **Re-render and dual-persona re-review** — verify fixes visually from both perspectives
6. **Ask the human for sign-off** on this group before moving to next

### Rules
- Complete one group at a time — do not scatter fixes across groups
- Human verification between groups is **mandatory**
- **Both personas must review at each iteration** — never skip the EE or Fab perspective
- After all groups are individually approved, do a final Phase 4 inter-group review
- The rendered PNG is the source of truth — scores can be misleading
- The layout must look **clean, organized, and grid-aligned** — not scattered and random
- Reference `docs/board_guidelines.md` Section 10 for subcircuit pattern layouts

## No Shortcuts — Ever

**NEVER work around a problem — fix the root cause.**

- Do NOT use post-placement overrides to mask optimizer bugs. Fix the optimizer.
- Do NOT claim a bug is "fixed" without verifying the fix in the actual output.
- Do NOT acknowledge a bug and defer it — fix it or explicitly say "I cannot fix this yet because X".
- Do NOT let the same bug be reported twice. Track bugs in `docs/design_rules/*_bugs.md` and check them BEFORE showing output to the human.
- Do NOT bypass footprint generation issues with script workarounds. Fix `footprints.py` or `builder.py`.
- Do NOT skip the dual-persona review. Run it and address findings before human review.
- If the optimizer produces wrong output, fix the optimizer code — not the test script.
- If a footprint has wrong labels, fix the footprint generator — not the render.
- If a component is missing, add it to the requirements AND the pipeline — not just the script.

**The training board scripts should use the SAME pipeline code as production boards.
Any fix that only works in the training script is not a fix.**

- NEVER overwrite human-edited PCB files without backing up first
- Training scripts must auto-backup to output/reference/ before writing
- Always compare generated output against the latest reference board

## Quality Gates

Before marking any module complete:
```bash
pytest tests/test_{module}.py -v      # Tests pass
mypy src/kicad_pipeline/{module}.py   # Types clean
ruff check src/kicad_pipeline/        # Lint clean
```

Before any commit to main:
```bash
pytest tests/ -x && mypy src/ --strict && ruff check src/ tests/
```

## Issue Triage Protocol

When a bug surfaces:

1. **Classify**: CORE (pipeline code bug) or DEPLOYMENT (project config/user error)?

2. **If CORE**:
   - Add entry to `docs/known_issues.md`
   - Write regression test in `tests/regression/test_known_issues.py`
   - Fix the pipeline code
   - Issue RERUN to affected board agents

3. **If DEPLOYMENT**:
   - Update that project's design docs or requirements
   - Do NOT change pipeline code
   - Document in the project's `design/design_decisions.md`

4. **Rules**:
   - Every core fix MUST have a regression test
   - Never let the same bug surface twice without a test
   - Never commit a fix without a test that would have caught it

5. **Sub-Agent Bug Reporting Protocol** (MANDATORY for downstream project agents):
   - Sub-agents (e.g. esp32-test, nl-s-3c-complete) MUST NOT write bug reports to their own `output/` directory
   - **CORE bugs** MUST be filed as `docs/bug_report_<topic>.md` in the **pipeline project** (`kicad-ai-workflow/`)
   - Each bug report MUST classify every issue as CORE or DEPLOYMENT
   - Each CORE bug MUST reference an existing KI-NNN ID or request a new one
   - Sub-agents MUST capture ALL user-reported issues — never drop or summarize away feedback
   - **DEPLOYMENT issues** stay in the downstream project's `design/` directory
   - The upstream pipeline agent is responsible for: registering KI-NNN entries, writing regression tests, fixing code
   - The downstream project agent is responsible for: requirements/config changes, re-running builds after pipeline fixes

## Decision Log

Record significant decisions in docs/ai_decisions.md:
```
## Decision: [title]
Date: YYYY-MM-DD
Context: [what prompted this]
Options: [A vs B vs C with pros/cons]
Decision: [chosen]
Rationale: [why]
```

## pyproject.toml Essentials

```toml
[project]
name = "kicad-ai-pipeline"
requires-python = ">=3.10"
dependencies = []  # Core is dependency-free

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-x --tb=short"

[tool.mypy]
strict = true
python_version = "3.10"

[tool.ruff]
target-version = "py310"
line-length = 99
[tool.ruff.lint]
select = ["E","W","F","I","N","UP","B","SIM","TCH","RUF"]
```

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **kicad-ai-workflow** (5116 symbols, 17856 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/kicad-ai-workflow/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/kicad-ai-workflow/context` | Codebase overview, check index freshness |
| `gitnexus://repo/kicad-ai-workflow/clusters` | All functional areas |
| `gitnexus://repo/kicad-ai-workflow/processes` | All execution flows |
| `gitnexus://repo/kicad-ai-workflow/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## CLI

- Re-index: `npx gitnexus analyze`
- Check freshness: `npx gitnexus status`
- Generate docs: `npx gitnexus wiki`

<!-- gitnexus:end -->
