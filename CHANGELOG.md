# Changelog

All notable changes to kicad-ai-pipeline are documented here.

## [Unreleased]

### Added
- 477 new tests with edge cases and error paths (2853 total)
- Quality scanner with EDA-tuned thresholds
- Module-scoped test fixtures for 16x faster builder tests
- `PlacementContext` dataclass for phase-based optimizer architecture
- `_BuildContext`, `_RouteContext`, `_PlacementState`, `_BcuEndpoint` dataclasses
- Named constants for magic numbers across optimization and PCB modules
- Apache 2.0 license, README, CHANGELOG

### Changed
- Split `placement_optimizer.py` (5192 LOC) into 10 focused modules
- Decomposed `optimize_placement_ee` (CC=465) into 24 named phase functions
- Reduced `build_pcb` CC from 120 to 8, `route_net` from 166 to 30
- Eliminated all critical-CC functions (was 13, now 0)
- Improved quality scanner: EDA thresholds, false positive suppression, prefix test matching
- Quality grade: C (58.8) to B (78.2)

### Fixed
- Scanner double-counting from stale git worktrees
- `_BcuEndpoint` signature mismatch in grid_router tests
- 8 dead imports removed

## [0.1.0] - 2026-03-01

### Added
- Complete pipeline: requirements through production-ready manufacturing files
- KiCad 10 compatible output (schematic version 20250114, PCB version 20241229)
- 3-level EE-grade placement optimizer with 10 subcircuit types
- Grid router with A* pathfinding and B.Cu fallback
- JLCPCB parts enrichment from 7M-part FTS5 database
- Hierarchical schematic support with correct instance paths
- DRC, ERC, electrical, manufacturing, thermal, signal integrity validation
- Gerber, drill, BOM, CPL production artifact generation
- CLI interface (`kicad-pipeline`)
- 2376 tests passing
