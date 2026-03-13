# Framework v2.0 -- Preventive Knowledge Base for PCB Placement

Version: 2.0
Date: 2026-03-12
Status: Active

This document captures recurring PCB placement bugs, embeds preventive knowledge as
engineering and fabrication checklists, and defines the self-improvement protocol that
governs every placement optimization iteration.

---

## 1. Self-Reflection: 10 Recurring Issues

These issues have surfaced repeatedly during placement engine development (v1 through v6).
Each is mapped to a known-issues entry (KI-001 through KI-010) in `docs/known_issues.md`.

| # | Issue | Root Cause | Times Seen | Prevention |
|---|-------|-----------|-----------|------------|
| 1 | Components off-board (KI-005) | Pad extent not checked -- only center-of-footprint checked against board bounds | 3x | `pin_map.pad_extent_in_board_space()` gate on every placement move; regression test `TestKI005ConnectorPadExtent` |
| 2 | Zone overlap (KI-008, KI-010) | Zone fractions not tiled properly -- zones allocated as percentages without accounting for component physical sizes | 4x | Non-overlapping zone rects via `partition_board()` with min-height constraints + integration tests |
| 3 | Cross-group contamination | Subcircuit detection (`detect_subcircuits()`) claims components across FeatureBlock boundaries via shared power nets | 3x | Same-group preference: only match components within the same FeatureBlock; `_is_power_net()` filter |
| 4 | Centroid math duplication (KI-004) | Origin-to-centroid conversion reimplemented inline in placement_optimizer, review_agent, scoring, placement_render | 2x | Single source of truth: `pin_map.compute_centroid_offset()`, `origin_to_centroid()`, `centroid_to_origin()` |
| 5 | Power rail false positives | Buck converter and decoupling detection follows shared GND/VCC nets, pulling unrelated components into subcircuits | 3x | `_is_power_net()` filter excludes GND/VCC/+3V3/+5V from adjacency search; per-pin connection limits |
| 6 | TVS diode misclassification | Relay flyback diode detection matches any diode near a relay, including TVS clamp diodes and indicator LEDs | 2x | Value/description filtering: only match 1N4148, 1N4007, SS14, etc.; exclude TVS/LED by value pattern |
| 7 | Phase ordering breaks patterns (KI-009) | Phase 3c places decoupling caps near ICs, then later group-level phases scatter them during collision resolution | 2x | Late re-pull (phase 3c-late) runs AFTER group phases; decoupling caps added to `subcircuit_fixed` set |
| 8 | Unrealistic scoring thresholds | Scoring thresholds assume 0402 passives but board uses relays (22mm), modules (25mm), large connectors | 2x | Size-aware thresholds: relay subgroup spread 5-8mm, power group 40-60mm, small passives 15-25mm |
| 9 | Collision resolution breaks subcircuits (KI-006, KI-007) | Phase 3g pushes components out of carefully placed subcircuit patterns to resolve overlaps | 3x | `subcircuit_fixed` and `template_protected` sets: collision resolution must not move protected components; instead move the OTHER component |
| 10 | Connector placement conflicts | Re-pinning connectors in subcircuit phases (3a-3f) undoes Level 2 edge placement from `place_groups()` | 2x | Level 3 phases must NOT re-position connectors that are already edge-pinned; check `connector_edge_pinned` set |

### Lessons Learned

- **Tests passing does NOT mean the board is correct.** Scores measure statistical
  properties; they cannot detect a single component 1mm off the board edge.
- **Visual verification is mandatory.** Read the rendered PNG after every code change.
- **Protection sets are essential.** Any component placed by a specialized phase must be
  added to a protection set so later phases do not undo the work.
- **Single source of truth prevents divergence.** When the same calculation appears in
  multiple files, consolidate to one function and import everywhere.

---

## 2. EE Engineer Review Checklist (Schematic)

These checks should run on every schematic before it is accepted. They correspond to
common errors that cause board respins.

### Connectivity & Completeness
- [ ] Net connectivity completeness -- no floating pins on active ICs
- [ ] All power pins on every IC connected to the correct rail
- [ ] No duplicate net names with different intended signals

### Protection & Reliability
- [ ] Diode polarity verification -- anode/cathode net connectivity matches intended current flow
- [ ] Flyback diode on every relay coil (1N4148 or equivalent)
- [ ] TVS diode voltage rating vs bus voltage (clamping voltage < max IC input)
- [ ] Protection diode rating vs signal voltage range

### Signal Integrity
- [ ] Pull-up/pull-down on all open-drain/open-collector outputs
- [ ] Crystal load cap calculation matches MCU datasheet requirements
- [ ] Filter cap value vs cutoff frequency target (RC time constant check)
- [ ] Voltage divider ratio check vs ADC input range (full scale < VREF)

### Power
- [ ] Decoupling cap presence per IC power pin (100nF minimum per VCC pin)
- [ ] Bulk capacitor on each power rail (10uF+ within 50mm of regulator output)

### Implementation

```python
# Automated checks in validation/electrical.py
def run_ee_schematic_review(requirements: ProjectRequirements) -> list[str]:
    """Run EE engineer checklist against requirements model."""
    issues: list[str] = []
    # 1. Check every IC has decoupling cap on each VCC pin
    # 2. Check relay coils have flyback diodes
    # 3. Check voltage divider ratios vs ADC VREF
    # 4. Check crystal load caps vs MCU requirements
    # ... (each check returns descriptive issue strings)
    return issues
```

---

## 3. Fabricator Review Checklist (Layout)

Pre-fabrication checks on every placement. These catch physical errors that cause
manufacturing rejects or non-functional boards.

### Board Boundary
- [ ] All components within board bounds (pad extents via `pad_extent_in_board_space()`, not just centers)
- [ ] No courtyard overlaps between any components (courtyard-to-courtyard gap >= 0)

### Proximity Constraints
- [ ] Decoupling caps within 3mm edge-to-edge of IC power pins
- [ ] Crystal and load caps within 5mm of MCU oscillator pins
- [ ] Connectors within 5mm of nearest board edge

### Isolation
- [ ] Relay-to-analog zone separation >= 10mm
- [ ] RF/antenna keepout zone clear (15mm radius, no copper pour, no components)
- [ ] No component from Group A placed inside Group B's zone (cross-group contamination)

### Signal & Power Flow
- [ ] Power flow direction consistent (input connectors at top/left, output at bottom/right)
- [ ] All identical subcircuits have identical physical layouts (relay drivers, ADC channels)
- [ ] Signal flow order matches component physical order (input->filter->IC->output)

### Manufacturing
- [ ] Via-to-pad clearance >= 0.2mm
- [ ] Thermal relief on all power pads
- [ ] Minimum trace width >= 0.15mm (JLCPCB 4-layer minimum)
- [ ] Minimum annular ring >= 0.13mm

### Implementation

```python
# Automated checks in optimization/review_agent.py
def run_fabricator_review(pcb: PCBDesign, requirements: ProjectRequirements) -> list[PlacementViolation]:
    """Run fabricator review checklist against placed PCB."""
    violations: list[PlacementViolation] = []
    # 1. Board boundary check (pad extents)
    # 2. Courtyard overlap check
    # 3. Decoupling distance check (edge-to-edge)
    # 4. Connector edge distance check
    # 5. Zone isolation check
    # 6. Subcircuit pattern consistency check
    return violations
```

---

## 4. Self-Improvement Protocol

This protocol governs every placement optimization iteration. It ensures regressions
are caught immediately and the system improves monotonically.

### After every placement optimization iteration:

1. **Run full test suite** (regression + integration)
   ```bash
   pytest tests/regression/ tests/integration/ -x --tb=short
   ```

2. **Render placement PNGs** (groups + domains + hi-fi)
   ```bash
   pytest tests/integration/test_placement_visual.py -x --tb=short -s
   ```

3. **Run EE engineer checklist** (automated via `validation/electrical.py`)

4. **Run fabricator checklist** (automated via `optimization/review_agent.py`)

5. **Compare scores against baseline**
   - No dimension may regress more than 5% from the previous iteration
   - If any dimension regresses, the iteration FAILS regardless of overall score
   - Baseline scores stored in `tests/regression/baseline_scores.json`

6. **If any check fails**: diagnose root cause, add regression test, fix, re-run

7. **If human feedback says "worse"**: `git revert HEAD`, try a different approach

8. **Log iteration results** in `docs/placement_iteration_log.md`:
   ```markdown
   ## Iteration N (YYYY-MM-DD)
   - Changes: [description]
   - Score: [overall] (delta: [+/-X.XXX])
   - Dimensions regressed: [list or "none"]
   - Human feedback: [approved / rejected / pending]
   ```

9. **After 50 iterations without progress**: STOP and change strategy fundamentally

### Version tracking

- Every framework update increments version (currently v2.0)
- Breaking changes documented with migration notes
- All past bugs have corresponding regression tests in `tests/regression/test_known_issues.py`
- Framework version recorded in this file header

---

## 5. Test-Optimize-Retry Loop

The canonical loop for iterative placement improvement. This pseudocode captures the
exact sequence of operations, including the regression guard.

```python
def iterative_placement_improvement(
    requirements: ProjectRequirements,
    max_iterations: int = 100,
    target_score: float = 0.95,
    regression_threshold: float = 0.05,
) -> tuple[PCBDesign, PlacementScore]:
    """Iteratively improve placement with regression protection."""

    previous_score: float = 0.0
    best_pcb: PCBDesign | None = None
    best_score: float = 0.0
    stall_counter: int = 0

    for iteration in range(max_iterations):
        # 1. Build and optimize
        pcb = build_pcb(requirements, placement_mode="grouped")
        pcb, review = optimize_placement_ee(requirements, pcb)

        # 2. Score the result
        score = compute_fast_placement_score(pcb, requirements)
        violations = review_placement(pcb, requirements)

        # 3. Check for success
        critical_violations = [v for v in violations if v.severity == "critical"]
        if score.overall >= target_score and not critical_violations:
            return pcb, score  # Success

        # 4. Check for regression
        if iteration > 0 and score.overall < previous_score - regression_threshold:
            # Regression detected -- revert to best known state
            log_iteration(iteration, score, "REVERTED -- regression detected")
            pcb = best_pcb
            score_value = best_score
        else:
            score_value = score.overall

        # 5. Track best result
        if score_value > best_score:
            best_score = score_value
            best_pcb = pcb
            stall_counter = 0
        else:
            stall_counter += 1

        # 6. Stall detection
        if stall_counter >= 50:
            log_iteration(iteration, score, "STALLED -- changing strategy")
            break  # Change strategy fundamentally

        # 7. Apply fixes from review
        for violation in violations:
            if violation.suggested_position is not None:
                apply_suggested_fix(pcb, violation)

        # 8. Log iteration
        previous_score = score_value
        log_iteration(iteration, score, "OK")

    assert best_pcb is not None
    return best_pcb, compute_fast_placement_score(best_pcb, requirements)
```

### Key invariants in the loop

- **Regression guard**: Never accept a result that scores 5%+ worse than the previous best.
- **Stall detection**: After 50 iterations without improvement, stop and change approach.
- **Best-state tracking**: Always keep a reference to the best PCB seen so far.
- **Violation-driven fixes**: Only apply fixes that come from the review agent's
  `suggested_position` field -- never make ad hoc adjustments.
- **Logging**: Every iteration is logged with score, delta, and outcome for post-mortem analysis.

---

## 6. Version 2.0 Confirmation -- Bug Status Audit

Status of each known issue as of v2.0 (2026-03-12):

### KI-001: Ref designator shows '?'
- **Regression test**: Yes -- `TestKI001RefDesignatorQuestionMark` (flat + hierarchical)
- **Fix in codebase**: Yes -- `_validate_no_question_mark_refs()` in schematic builder
- **Can it recur**: No. The regression test checks both flat and hierarchical schematics.
  The fix validates at build time, not just at test time.

### KI-002: Schematic-PCB component desync
- **Regression test**: Yes -- `TestKI002SchematicPCBSync`
- **Fix in codebase**: Yes -- `check_consistency()` hard gate in VALIDATION stage
- **Can it recur**: No. The consistency check runs as a hard gate before production
  artifacts are generated. Any desync fails the pipeline.

### KI-003: Footprint mismatch SCH-to-PCB
- **Regression test**: Yes -- `TestKI003FootprintConsistency`
- **Fix in codebase**: Yes -- `check_consistency()` footprint match check
- **Can it recur**: No. Same hard gate as KI-002 catches footprint mismatches.

### KI-004: Centroid offset duplicated 4x
- **Regression test**: Indirect -- all tests that use placement scoring exercise the
  consolidated `pin_map.py` functions. No dedicated regression test for the duplication
  itself, but the single-source-of-truth pattern prevents re-introduction.
- **Fix in codebase**: Yes -- `compute_centroid_offset()`, `origin_to_centroid()`,
  `centroid_to_origin()` consolidated in `pcb/pin_map.py`
- **Can it recur**: Unlikely. The MEMORY.md and this document both flag inline centroid
  math as prohibited. Code review should catch any new duplication.

### KI-005: Connectors off-board
- **Regression test**: Yes -- `TestKI005ConnectorPadExtent`
- **Fix in codebase**: Yes -- `pad_extent_in_board_space()` used in connector clamping
- **Can it recur**: Unlikely for connectors. Could recur for other large components
  (relays, modules) if pad extent checking is not applied universally. The regression
  test only checks J-prefix refs. Consider extending to all refs.

### KI-006: MCU group overlapping components
- **Regression test**: Not yet -- marked OPEN in known_issues.md
- **Fix in codebase**: Partial -- `subcircuit_fixed` and `template_protected` sets exist
  but protection-set audit is incomplete
- **Can it recur**: YES. This needs a regression test that verifies zero courtyard
  overlaps after optimization. Priority: HIGH.

### KI-007: Relay driver components scattered
- **Regression test**: Not yet -- marked OPEN in known_issues.md
- **Fix in codebase**: Partial -- phase 3b relay driver tightening exists but target
  distances may be too large
- **Can it recur**: YES. Needs regression test measuring Q/D/R distance from relay
  centroid (must be < 8mm). Priority: HIGH.

### KI-008: Power tail overflow to y=70
- **Regression test**: Not yet -- marked OPEN in known_issues.md
- **Fix in codebase**: Partial -- zone partitioning improved but column overflow not
  fully guarded
- **Can it recur**: YES. Needs regression test that checks all components are within
  their assigned zone boundaries. Priority: MEDIUM.

### KI-009: ADC channel order vs screw terminals
- **Regression test**: Not yet -- marked OPEN in known_issues.md
- **Fix in codebase**: Not yet -- channel ordering logic not implemented
- **Can it recur**: YES. Needs both fix and regression test. Priority: MEDIUM.

### KI-010: Analog/power blocks too high
- **Regression test**: Not yet -- marked OPEN in known_issues.md
- **Fix in codebase**: Partial -- zone top margin adjusted but not fully validated
- **Can it recur**: YES. Needs regression test that checks zone top margins >= 16mm
  (20% of 80mm board). Priority: LOW.

### Summary

| ID | Has Regression Test | Fix in Codebase | Can Recur |
|----|-------------------|----------------|-----------|
| KI-001 | Yes | Yes | No |
| KI-002 | Yes | Yes | No |
| KI-003 | Yes | Yes | No |
| KI-004 | Indirect | Yes | Unlikely |
| KI-005 | Yes | Yes | Unlikely |
| KI-006 | **No** | Partial | **YES** |
| KI-007 | **No** | Partial | **YES** |
| KI-008 | **No** | Partial | **YES** |
| KI-009 | **No** | No | **YES** |
| KI-010 | **No** | Partial | **YES** |

**Action items for v2.1**: Write regression tests for KI-006 through KI-010 and
complete the fixes. These five open issues represent the highest-risk failure modes
in the current placement engine.

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `src/kicad_pipeline/pcb/pin_map.py` | Single source of truth for centroid math |
| `src/kicad_pipeline/pcb/layout_templates.py` | IC and subcircuit layout templates |
| `src/kicad_pipeline/optimization/placement_optimizer.py` | 3-level hierarchical placement |
| `src/kicad_pipeline/optimization/review_agent.py` | Placement review with 12 rules |
| `src/kicad_pipeline/optimization/scoring.py` | 13-dimension placement scoring |
| `src/kicad_pipeline/optimization/functional_grouper.py` | Subcircuit detection and grouping |
| `src/kicad_pipeline/validation/electrical.py` | Electrical rule checks |
| `tests/regression/test_known_issues.py` | Regression tests for KI-001 through KI-005 |
| `docs/known_issues.md` | Known issues registry |
| `docs/placement_iteration_log.md` | Iteration history and results |
