---
module: src/kicad_pipeline/cli
owner: specialist
status: active
version: 1
created: 2026-03-17
last_modified: 2026-03-17
domains: [tooling/cli, reliability/logging, tooling, tooling/kicad]
---

# src/kicad_pipeline/cli

## Purpose
<!-- Describe the purpose of this module -->
Auto-discovered module at `src/kicad_pipeline/cli/`.
2234 lines of code, 16 public functions.

## Domains
- `tooling/cli`
- `reliability/logging`
- `tooling`
- `tooling/kicad`

## Health: RED
- LOC: 2234/500
- Public functions: 16/8
- Dependencies: 14/10
- Test coverage: 0%/85%

## Invariants
<!-- List invariants that must always hold -->

## Dependencies
- __future__
- argparse
- collections
- dataclasses
- datetime
- json
- kicad_pipeline
- logging
- os
- pathlib
- shutil
- subprocess
- sys
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
