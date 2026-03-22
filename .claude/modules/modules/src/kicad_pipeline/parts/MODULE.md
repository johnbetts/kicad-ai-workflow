---
module: src/kicad_pipeline/parts
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [data, data/cache, hardware/pcb, hardware/pcb/fabrication, hardware/pcb/footprints]
---

# src/kicad_pipeline/parts

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/parts/`.
1930 lines of code, 46 public functions.

## Domains
- `data`
- `data/cache`
- `hardware/pcb`
- `hardware/pcb/fabrication`
- `hardware/pcb/footprints`

## Health: RED
- LOC: 1930/500
- Public functions: 46/8
- Dependencies: 10/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- dataclasses
- kicad_pipeline
- logging
- os
- pathlib
- re
- sqlite3
- subprocess
- typing

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
