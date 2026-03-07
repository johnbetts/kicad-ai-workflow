# AI-Assisted KiCad EDA Pipeline — Plan Part 2

## PHASE 5: VALIDATION & TESTING

### 5.1 DRC (Design Rule Check)

Geometric checks against the PCB:
- Trace/pad/via clearances per net class
- Annular ring and drill minimums
- Trace width minimums per net class
- Hole-to-hole spacing, component-to-edge clearance
- Silk-to-pad clearance and minimum silk width
- No overlapping pads on different nets
- No copper in keepout zones
- Board outline is closed polygon

### 5.2 Electrical Validation

Cross-check schematic vs PCB:
- All schematic nets present in PCB, no phantom nets
- Every schematic component has a PCB footprint
- Pin-to-pad mapping correct
- No shorted nets from copper pour
- Power rail and ground continuity
- Decoupling caps within 3mm of IC power pins

### 5.3 Manufacturing Validation (JLCPCB)

- All features meet JLCPCB minimums
- Drill sizes in standard chart
- Board dimensions within limits
- No acid traps
- Paste apertures appropriate
- Thermal relief on pour pads
- LCSC part numbers valid and in stock
- Footprint pads match expectations
- SMT on one side (or flagged)
- Component orientations consistent

### 5.4 Thermal & Signal Integrity

Thermal: power dissipation per component, flag >500mW, verify thermal
vias and copper area, check for hot spots.

Signal integrity: USB diff pair length matching, SPI length limits,
no analog/digital parallel runs, antenna keepout clear.

### 5.5 Validation Report

JSON report with overall status (PASS/PASS_WITH_WARNINGS/FAIL),
per-category breakdown, and recommended actions. Each error creates
a GitHub Issue automatically.

### 5.6 GitHub: commit report, auto-create issues, tag v0.5.0-validated if 0 errors.

---

## PHASE 6: PRODUCTION ARTIFACTS

### 6.1 Gerbers (RS-274X)

One file per layer: F/B Cu, Paste, Silk, Mask, Edge.Cuts, inner layers.
Format: RS-274X, 4.6 coordinate (mm), trailing zero suppression, absolute.

### 6.2 Drill Files (Excellon)

Separate PTH and NPTH. Excellon 2 format, mm, tool table,
rounded to JLCPCB standard increments.

### 6.3 BOM (JLCPCB CSV)

```
Comment,Designator,Footprint,LCSC Part Number
10k,"R1 R2 R5",R_0805,C17414
```

Extended BOM with pricing, stock, alternatives.
Cost estimate for qty 5/10/50/100.

### 6.4 CPL / Pick-and-Place (JLCPCB)

```
Designator,Val,Package,Mid X,Mid Y,Rotation,Layer
R1,10k,0805,25.4,12.7,0,top
```

Includes JLCPCB rotation correction offsets.

### 6.5 Assembly Drawings (PDF)

Top/bottom views with refs, polarity marks, dimensions,
stackup diagram, special notes.

### 6.6 Production Package

```
production/
├── gerbers/project_gerbers.zip
├── assembly/{bom, cpl, assembly.zip}
├── docs/{assembly pdfs, fab notes, schematic print}
├── source/{.kicad_sch, .kicad_pcb, .kicad_pro}
└── ORDER_GUIDE.md
```

### 6.7 Auto-Generated Order Guide

Step-by-step JLCPCB instructions: exact settings, which files where,
rotation issues to watch, cost estimates.

### 6.8 GitHub: commit production/, create Release (tag v1.0.0) with zip assets.

---

## PHASE 7: GITHUB INTEGRATION (Continuous)

### 7.1 Repository Structure

```
{project}-hardware/
├── .github/{workflows/, ISSUE_TEMPLATE/, PR_TEMPLATE}
├── requirements/
├── schematic/
├── pcb/
├── reports/
├── production/
├── docs/{architecture.md, ai_decisions.md, review_checklist.md}
├── CHANGELOG.md, README.md, LICENSE, .gitignore
```

### 7.2 Conventions

Semantic commits: `feat(phase): description`, `fix(category): description`,
`test(validation): description`, `release(production): description`.

### 7.3 Project Board (Kanban)

Columns: Backlog | Requirements | Schematic | PCB/Route | Validate | Released

### 7.4 Labels

Phase (requirements..production), Type (bug, enhancement, feature-request),
Severity (critical..cosmetic), Category (erc, drc, bom, routing, thermal, SI),
Status (needs-triage, in-progress, blocked), Auto (ai-generated).

### 7.5 GitHub Actions

validate-on-push: ERC + DRC + BOM validation on every push.
generate-production: Gerbers + BOM + CPL on version tag → GitHub Release.

### 7.6 Iteration Loop

Issue → branch → AI modifies design → validation → commit → merge →
CHANGELOG → close issue → move card.

---

## BUILD ORDER (14 weeks)

**Sprint 1 (W1-2) Foundation**: S-exp writer/parser, UUID, JSON schemas,
repo init, CLI framework.

**Sprint 2 (W2-3) Requirements**: Feature decomposition, component DB,
pin/power budgets, constraint resolver, JSON generator.

**Sprint 3 (W3-5) Schematic**: Symbol library, auto-symbol gen, subcircuit
generators, placement, wire routing, ERC, .kicad_sch output.

**Sprint 4 (W5-7) PCB Layout**: Footprint library, netlist extractor,
design rule engine, board outline, placement engine, copper zones,
keepouts, silkscreen, .kicad_pcb output.

**Sprint 5 (W7-9) Autorouting**: Grid router, A* pathfinder, multi-pass,
via optimization, rip-up/retry, FreeRouting integration, cleanup.

**Sprint 6 (W9-10) Validation**: DRC engine, electrical validation,
JLCPCB mfg validation, thermal, SI checks, report + auto-issues.

**Sprint 7 (W10-11) Production**: Gerber gen, drill gen, BOM gen,
CPL gen (with rotation correction), assembly PDF, packager, order guide, cost estimator.

**Sprint 8 (W11-12) GitHub**: Git ops, GitHub API, commit/changelog gen,
Actions workflows, issue templates, release creation, board automation.

**Sprint 9 (W12-14) Polish**: E2E tests (LED blinker + ESP32 project),
KiCad verification, JLCPCB upload test, docs, error handling, edge cases.

---

## TECHNICAL DECISIONS

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.10+ | Stdlib core, rich ecosystem |
| File format | KiCad 7/8 S-expression | Current, documented |
| Dependencies | Minimal core | Standalone, no KiCad needed |
| Symbols | Built-in + auto-generate | Coverage + simplicity |
| Footprints | Parametric generators | Covers 90% common packages |
| Routing | Built-in + FreeRouting | Default + escape hatch |
| Gerbers | Custom RS-274X writer | No KiCad dependency |
| GitHub | PyGithub + subprocess git | Full automation |
| Validation | Custom DRC/ERC | Runs without KiCad |
| BOM/CPL | JLCPCB CSV primary | Target manufacturer |

---

## DATA MODELS

### Requirements Domain
```
ProjectRequirements
├── project: {name, author, revision}
├── features: List[FeatureBlock] → {name, components[], nets[], subcircuits[]}
├── components: List[Component] → {ref, value, footprint, lcsc, pins[]}
├── pin_assignment: MCUPinMap
├── nets: List[Net] → {name, connections[{ref, pin}]}
├── power_budget: PowerBudget
├── mechanical: {board_size, enclosure, mounting[]}
└── ai_recommendations: List[Recommendation]
```

### Schematic Domain
```
Schematic
├── lib_symbols, components (placed), power_symbols
├── wires, labels, global_labels
└── subcircuits
```

### PCB Domain
```
PCBDesign
├── board (outline, layers, stackup), design_rules
├── nets (numbered), footprints (placed, net-assigned)
├── zones, keepouts, tracks, vias
├── mounting_holes, silkscreen
```

### Production Domain
```
ProductionPackage
├── gerbers, drills, bom, cpl
├── assembly_drawings, order_guide, cost_estimate
```

---

## RISKS

| Risk | Mitigation |
|------|------------|
| KiCad format v9 changes | Version detection, multiple writers |
| Autorouter poor results | FreeRouting fallback, quality metrics |
| JLCPCB rotation offsets | Correction database, visual verify |
| Footprint geometry wrong | Validate vs KiCad lib, test with real orders |
| S-expression rejected | Round-trip testing vs KiCad output |

---

## OPEN QUESTIONS

1. KiCad coordinate/rotation conventions — verify empirically
2. Minimum viable lib_symbol S-expression
3. Does KiCad validate PCB nets against external netlist?
4. KiCad 8 compatibility with v7 format version
5. FreeRouting headless operation + Java compatibility
6. JLCPCB API availability for stock/price
7. CI validation without KiCad installation
