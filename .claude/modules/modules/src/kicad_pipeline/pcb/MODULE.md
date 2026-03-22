---
module: src/kicad_pipeline/pcb
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [reliability/validation, tooling/kicad, api-design, data, hardware/pcb]
---

# src/kicad_pipeline/pcb

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/pcb/`.
21316 lines of code, 260 public functions.

## Domains
- `reliability/validation`
- `tooling/kicad`
- `api-design`
- `data`
- `hardware/pcb`

## Health: RED
- LOC: 21316/500
- Public functions: 260/8
- Dependencies: 17/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- copy
- dataclasses
- datetime
- enum
- fnmatch
- json
- kicad_pipeline
- logging
- math
- pathlib
- re
- shutil
- subprocess
- tempfile
- typing
- uuid

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
