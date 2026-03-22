---
module: src/kicad_pipeline/ipc
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [hardware/pcb, hardware/pcb/footprints, reliability/logging, tooling/cli, tooling/kicad]
---

# src/kicad_pipeline/ipc

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/ipc/`.
874 lines of code, 36 public functions.

## Domains
- `hardware/pcb`
- `hardware/pcb/footprints`
- `reliability/logging`
- `tooling/cli`
- `tooling/kicad`

## Health: RED
- LOC: 874/500
- Public functions: 36/8
- Dependencies: 9/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- contextlib
- dataclasses
- kicad_pipeline
- kipy
- logging
- pathlib
- types
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
