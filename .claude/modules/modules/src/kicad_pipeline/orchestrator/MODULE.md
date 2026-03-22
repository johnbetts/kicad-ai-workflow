---
module: src/kicad_pipeline/orchestrator
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [api-design, data, hardware/pcb/footprints, ml, reliability/logging]
---

# src/kicad_pipeline/orchestrator

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/orchestrator/`.
2996 lines of code, 66 public functions.

## Domains
- `api-design`
- `data`
- `hardware/pcb/footprints`
- `ml`
- `reliability/logging`

## Health: RED
- LOC: 2996/500
- Public functions: 66/8
- Dependencies: 11/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- dataclasses
- datetime
- enum
- json
- kicad_pipeline
- logging
- pathlib
- re
- shutil
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
