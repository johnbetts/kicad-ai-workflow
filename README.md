# kicad-ai-pipeline

AI-assisted KiCad EDA pipeline that takes hardware projects from natural-language requirements through to production-ready manufacturing files (Gerbers, BOM, CPL) for JLCPCB.

## What it does

1. **Requirements** — Define your board in plain English: components, connections, constraints
2. **Schematic** — Generates KiCad 10 hierarchical schematics with correct net assignment
3. **PCB Layout** — Places components using a 3-level EE-grade placement optimizer (zone partitioning, group placement, subcircuit refinement)
4. **Validation** — DRC, ERC, electrical checks, manufacturing checks, signal integrity
5. **Production** — Gerber, drill, BOM, CPL files ready for JLCPCB SMT assembly

Every stage outputs valid KiCad 10 files that open directly in KiCad.

## Install

```bash
pip install -e .
# or with dev dependencies:
pip install -e ".[dev]"
```

Requires Python 3.10+. No runtime dependencies for core pipeline.

## Quick start

```bash
# From a requirements JSON file:
kicad-pipeline build requirements.json --output-dir output/

# Or use the Python API:
python -c "
from kicad_pipeline.pcb.builder import build_pcb
from kicad_pipeline.models.requirements import ProjectRequirements
import json

req = ProjectRequirements.from_dict(json.load(open('requirements.json')))
pcb = build_pcb(req)
"
```

## Pipeline stages

| Stage | Module | Output |
|-------|--------|--------|
| Requirements | `requirements/` | `ProjectRequirements` dataclass |
| Parts enrichment | `parts/` | LCSC part numbers from JLCPCB 7M-part database |
| Schematic | `schematic/` | `.kicad_sch` files (hierarchical) |
| PCB layout | `pcb/` + `optimization/` | `.kicad_pcb` with placed components |
| Routing | `routing/` | Tracks and vias (grid router + FreeRouting) |
| Validation | `validation/` | DRC, ERC, electrical, manufacturing, thermal, SI reports |
| Production | `production/` | Gerbers, drill, BOM, CPL, assembly drawings |

## Project structure

```
src/kicad_pipeline/
  models/          # Frozen dataclasses: PCBDesign, ProjectRequirements, etc.
  sexp/            # KiCad S-expression parser and writer
  requirements/    # NL decomposition, component DB, pin/power budgets
  schematic/       # Schematic builder, symbols, subcircuits, wiring, ERC
  pcb/             # PCB builder, footprints, placement, netlist, zones
  optimization/    # EE placement optimizer (24 phases), scoring, review agent
  routing/         # Grid router, FreeRouting integration, DSN export
  validation/      # DRC, electrical, manufacturing, thermal, signal integrity
  production/      # Gerber, drill, BOM, CPL, assembly drawing, packager
  orchestrator/    # Multi-stage workflow engine with variant support
  github/          # Git ops, issues, releases, changelog
  cli/             # Command-line interface
tests/             # 2850+ tests (unit, integration, regression)
```

## Testing

```bash
pytest tests/ -x --tb=short          # Full suite
pytest tests/ -m "not slow"          # Fast tests only
pytest tests/ --cov=src/kicad_pipeline  # With coverage
```

## Code quality

```bash
mypy src/ --strict    # Type checking
ruff check src/ tests/  # Linting
ruff format src/ tests/ # Formatting
```

## JLCPCB compatibility

All components are validated against JLCPCB's parts catalog. The pipeline prefers basic parts (no setup fee) and verifies stock availability before generating production files.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
