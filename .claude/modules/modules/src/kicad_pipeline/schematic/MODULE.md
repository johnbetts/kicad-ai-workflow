---
module: src/kicad_pipeline/schematic
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [requirements, tooling/kicad, api-design, data, hardware/pcb/placement]
---

# src/kicad_pipeline/schematic

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/schematic/`.
9152 lines of code, 108 public functions.

## Domains
- `requirements`
- `tooling/kicad`
- `api-design`
- `data`
- `hardware/pcb/placement`

## Health: RED
- LOC: 9152/500
- Public functions: 108/8
- Dependencies: 13/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- collections
- dataclasses
- datetime
- enum
- inspect
- kicad_pipeline
- logging
- math
- pathlib
- re
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
