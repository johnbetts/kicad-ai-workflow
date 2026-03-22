---
module: src/kicad_pipeline/github
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [reliability/validation, api-design, data, infrastructure/production, reliability/logging]
---

# src/kicad_pipeline/github

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/github/`.
1236 lines of code, 52 public functions.

## Domains
- `reliability/validation`
- `api-design`
- `data`
- `infrastructure/production`
- `reliability/logging`

## Health: RED
- LOC: 1236/500
- Public functions: 52/8
- Dependencies: 7/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- dataclasses
- datetime
- kicad_pipeline
- pathlib
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
