---
module: src/kicad_pipeline/validation
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [data, reliability/validation, api-design, hardware/pcb, hardware/pcb/footprints]
---

# src/kicad_pipeline/validation

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/validation/`.
6218 lines of code, 162 public functions.

## Domains
- `data`
- `reliability/validation`
- `api-design`
- `hardware/pcb`
- `hardware/pcb/footprints`

## Health: RED
- LOC: 6218/500
- Public functions: 162/8
- Dependencies: 12/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- dataclasses
- enum
- hashlib
- json
- kicad_pipeline
- logging
- math
- pathlib
- subprocess
- tempfile
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
