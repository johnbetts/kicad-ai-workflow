---
module: src/kicad_pipeline/visualization
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [tooling/visualization, hardware/pcb, hardware/pcb/placement, ml, reliability/logging]
---

# src/kicad_pipeline/visualization

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/visualization/`.
1536 lines of code, 12 public functions.

## Domains
- `tooling/visualization`
- `hardware/pcb`
- `hardware/pcb/placement`
- `ml`
- `reliability/logging`

## Health: RED
- LOC: 1536/500
- Public functions: 12/8
- Dependencies: 13/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- cairosvg
- contextlib
- kicad_pipeline
- logging
- math
- matplotlib
- pathlib
- shutil
- subprocess
- tempfile
- typing
- xml

## Interface Contract
See interfaces.md

## Test Requirements
- Unit: every public function has >= 3 test cases
- Integration: module works correctly with its dependencies
- Regression: tests for every bug in bugs.md

## Version History
| Version | Date | Change | Agent |
|---------|------|--------|-------|
| 1 | 2026-03-17 | Initial bootstrap | project-architect |
