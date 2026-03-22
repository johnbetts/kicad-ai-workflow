---
module: src/kicad_pipeline
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [data, reliability/validation, requirements, tooling/kicad, hardware/pcb]
---

# src/kicad_pipeline

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/`.
2284 lines of code, 78 public functions.

## Domains
- `data`
- `reliability/validation`
- `requirements`
- `tooling/kicad`
- `hardware/pcb`

## Health: RED
- LOC: 2284/500
- Public functions: 78/8
- Dependencies: 8/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- datetime
- json
- kicad_pipeline
- logging
- pathlib
- re
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
