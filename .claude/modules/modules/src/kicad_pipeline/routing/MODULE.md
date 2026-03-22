---
module: src/kicad_pipeline/routing
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [api-design, hardware/pcb, hardware/pcb/routing, reliability/logging, reliability/monitoring]
---

# src/kicad_pipeline/routing

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/routing/`.
6862 lines of code, 66 public functions.

## Domains
- `api-design`
- `hardware/pcb`
- `hardware/pcb/routing`
- `reliability/logging`
- `reliability/monitoring`

## Health: RED
- LOC: 6862/500
- Public functions: 66/8
- Dependencies: 12/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- collections
- dataclasses
- heapq
- kicad_pipeline
- logging
- math
- os
- pathlib
- re
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
