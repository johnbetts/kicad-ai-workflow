---
module: src/kicad_pipeline/optimization
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [api-design, data/messaging, hardware/pcb/placement, infrastructure, infrastructure/config]
---

# src/kicad_pipeline/optimization

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/optimization/`.
17730 lines of code, 144 public functions.

## Domains
- `api-design`
- `data/messaging`
- `hardware/pcb/placement`
- `infrastructure`
- `infrastructure/config`

## Health: RED
- LOC: 17730/500
- Public functions: 144/8
- Dependencies: 17/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- collections
- contextlib
- dataclasses
- datetime
- enum
- json
- kicad_pipeline
- logging
- math
- os
- pathlib
- random
- re
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
