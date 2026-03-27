# KiCad AI Workflow — Project Management Framework Roadmap

## Status: active
Started: 2026-03-26
Last updated: 2026-03-26

## User Personas (architectural constraint)
| Persona | Role | Scope | Manages |
|---------|------|-------|---------|
| **Framework Developer** | Builds/maintains pipeline code | The codebase | Code bugs, regressions, architecture, releases |
| **Framework User (Admin)** | Deploys/configures the framework | An instance with shared config | Shared parts DB, review policies, user access, multi-board orchestration |
| **Project User** | Designs board(s) within a deployment | Their board project(s) | Requirements, placement, review progress, manufacturing readiness |

All features must be designed with this hierarchy: Framework → Deployment → Project(s).

## Bugs (fix these first)
- [ ] **P0** BUG-001: Agents skip verification steps and claim success — no enforcement mechanism
- [ ] **P0** BUG-002: Agents repeat known mistakes (don't consult docs/known_issues.md)
- [ ] **P1** BUG-003: Agents lose context mid-task, forget earlier findings
- [ ] **P1** BUG-004: Self-verification — agent that made changes also verifies them

## Phases

### Phase 1a: Evidence Model — COMPLETE
- [x] TASK-001: Create `src/kicad_pipeline/evidence/models.py` (EvidenceRecord, EvidenceLedger, GateResult, Issue, ScoreSnapshot)
- [x] TASK-002: Create `src/kicad_pipeline/evidence/ledger.py` (JSONL persistence)
- [x] TASK-003: Create `src/kicad_pipeline/evidence/gates.py` (STAGE_GATES definitions)
- [x] TASK-004: Write `tests/test_evidence_models.py` (26 tests)
- [x] TASK-005: Write `tests/test_evidence_ledger.py` (10 tests)
- [x] TASK-006: Write `tests/test_evidence_gates.py` (8 tests)
- [x] TASK-006b: Create `src/kicad_pipeline/evidence/known_issues.py` + tests (6 tests)
Milestone: **ACHIEVED** — 54 tests passing, ruff clean, JSONL round-trip works, gates for all 5 stages

### Phase 1b: Runner Hardening — COMPLETE
- [x] TASK-007: Add `EvidenceStep` dataclass extending `Step`
- [x] TASK-008: Add `run_evidence_step()` with evidence writing
- [x] TASK-009: Add `check_gate()` blocking gate function (via evidence/gates.py)
- [x] TASK-010: Add `run_hardened_process()` with gates between stages
- [x] TASK-011: Create `evidence/known_issues.py` — injector for known issues + lessons
- [x] TASK-012: Add verification agent pattern (separate agent call for verification)
- [x] TASK-013: Define `PCB_REVIEW_EVIDENCE_STEPS` + `EVIDENCE_PROCESSES` registry
- [x] TASK-014: Write `tests/test_known_issues.py` (6 tests)
- [x] TASK-015: Write `tests/test_process_runner_hardened.py` (21 tests)
- [x] TASK-015b: Add `--hardened` CLI flag
Milestone: **ACHIEVED** — 21 runner tests + 6 known issues tests passing, ruff clean

### Phase 1c: NiceGUI Dashboard MVP — COMPLETE
- [x] TASK-016: Add `dashboard = ["nicegui>=2.0"]` to pyproject.toml
- [x] TASK-017: Create `dashboard/app.py` — three-panel NiceGUI layout
- [x] TASK-018: Create `dashboard/panels.py` — image gallery, log stream, context panels
- [x] TASK-019: Create `dashboard/api.py` — Starlette endpoints (POST evidence, GET ledger, approve/reject)
- [x] TASK-020: Create `scripts/dashboard.py` — CLI entry point
- [x] TASK-021: Write `tests/test_dashboard_api.py` (8 tests, TestClient)
Milestone: **ACHIEVED** — 8 dashboard tests passing, ruff clean, NiceGUI installed

### Phase 2: Integration — COMPLETE
- [x] TASK-022: Runner `--dashboard` / `--no-dashboard` CLI flags with fire-and-forget HTTP POST
- [x] TASK-023: Human gate polls ledger for HUMAN_APPROVAL (via `_poll_for_human_approval()`)
- [x] TASK-024: Dashboard approve/reject buttons write to evidence ledger (already in Phase 1c)
- [x] TASK-025: `notify_dashboard()` and `notify_dashboard_log()` with stdlib urllib (no deps)
- [x] TASK-025b: Integration tests (11 tests) — HTTP posting, error resilience, CLI flags
Milestone: **ACHIEVED** — 94 total tests passing. Runner POSTs to dashboard, dashboard writes approvals back, human gate detects them.

### Phase 3: Polish — not started
- [ ] TASK-026: Requirements tracking panel in dashboard
- [ ] TASK-027: Scoring trend charts (ECharts via ui.chart)
- [ ] TASK-028: Before/after diff view for iterations
- [ ] TASK-029: Board selector dropdown for multi-board projects
Milestone: Dashboard is the primary review interface, faster than CLI scrolling

## Wishlist
- WISH-001: Real-time render preview via WebSocket — live image updates during kicad-image-gen
- WISH-002: Annotation overlay on PCB images — click to add notes on specific components
- WISH-003: Export review report as PDF — for stakeholders/documentation
- WISH-004: KiCad MCP server integration — direct LLM→KiCad interaction

## Moonshots
- MOON-001: Hosted multi-tenant service — auth, cloud rendering, per-user projects (flux.ai competitor)
- MOON-002: Natural language → manufactured PCB end-to-end — consumer-grade "idea to board"
- MOON-003: Community subcircuit template library — proven patterns shared across users
- MOON-004: Mobile review app — approve/reject from phone while boards iterate in background

## Decision Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-26 | Custom lightweight runner over LangGraph/Dagster | Existing process_runner.py is 80% there; external frameworks add dependency weight. Borrow patterns instead. |
| 2026-03-26 | NiceGUI over Streamlit/FastAPI+HTMX | Pure Python, FastAPI underneath (API-first), WebSocket auto-sync, three-panel native. Path to hosted service. |
| 2026-03-26 | JSONL over SQLite for evidence ledger | Append-only, crash-safe, human-readable, no dependency. Adequate for local single-user. |
| 2026-03-26 | Pydantic for evidence, keep frozen dataclasses for existing models | Evidence needs JSON serialization + validation. Don't change existing patterns. |
| 2026-03-26 | Verification by separate agent call | Prevents self-verification. Different prompt, different context. |
| 2026-03-26 | Enforcement first, visibility second | Prevent the lie before building the dashboard to catch it. |
| 2026-03-26 | Three-tier user hierarchy | Framework Developer → Framework User (Admin) → Project User. All features designed with this in mind. |

## Session Log
| Date | Session | Summary | Outcome |
|------|---------|---------|---------|
| 2026-03-26 | #1 | Problem-solving: interrogate → brainstorm → research → recommend → discuss → plan | Plan approved, roadmap initialized |
