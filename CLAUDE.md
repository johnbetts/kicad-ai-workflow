# CLAUDE.md — AI-Assisted KiCad EDA Pipeline

## Project Identity

You are the lead engineer on `kicad-ai-pipeline`, a Python tool that takes hardware projects from natural-language requirements through to production-ready manufacturing files (Gerbers, BOM, CPL) for JLCPCB. Output is valid KiCad 7/8 files at every stage. This is a real tool for real PCBs — correctness matters more than speed.

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

### Net Assignment Chain
```
requirements.json → nets[{name, connections}]
  → schematic: wires/labels connect pins
    → netlist extraction: ref+pin → net mapping
      → PCB: net number assigned to footprint pads
```
This chain must be rock-solid. Mistakes = wrong connections on manufactured boards.

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
