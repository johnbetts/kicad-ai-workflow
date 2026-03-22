---
module: src/kicad_pipeline/agents
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [data, ml, reliability/logging, reliability/monitoring, reliability/validation]
---

# src/kicad_pipeline/agents

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/agents/`.
2142 lines of code, 100 public functions.

## Domains
- `data`
- `ml`
- `reliability/logging`
- `reliability/monitoring`
- `reliability/validation`

## Health: RED
- LOC: 2142/500
- Public functions: 100/8
- Dependencies: 14/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- contextlib
- dataclasses
- datetime
- enum
- json
- kicad_pipeline
- logging
- os
- pathlib
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
