# Placement Iteration Proposals — nl-s-3c-complete

## Run 5 — 2026-04-05 (placement-iterate)

- Board: 200x95mm, 133 footprints, 98 nets
- Baseline (post-optimizer): Grade D (0.780), Placement 0.679
- Collisions: 63 (scorer sees ~20, optimizer sees 63)
- Crossings: 464, ratsnest length 4544mm
- After ratsnest swaps: 417 crossings (-47)
- Review grade: F (308 violations: 66 critical, 242 major)
- Off-board: J13 (RJ45)
- Cross-group contamination: 17 components
- Zone overflow: MCU 68%, Relay 72%, Ethernet 94%, Analog 5%

### Fixes Applied This Run

1. **P1: Reorder net proximity** — moved `_enforce_net_proximity()` before
   `_final_body_collision_fix()`. Impact: collisions 68→63 (marginal — most
   collisions were pre-existing, not from net proximity).
2. **P2: Collision-aware net proximity** — pull logic now checks for collisions
   before placing, tries 8 cardinal offsets if target is occupied.
   Impact: 0 skipped pulls (all 18 found free positions).
3. **P3: Hard collision gate in scoring** — Grade capped at D if collision
   score < 0.50 (≥12 collisions). Prevents optimizer from tolerating collisions.
   Impact: Grade downgraded B→D (correct — 63 collisions is unmanufacturable).

### Root Cause Analysis (NEW — from optimizer logs)

**The collision resolver is fundamentally stuck.** It starts with 137 collisions,
resolves 5 in pass 1, then oscillates relocating the same 3-5 components for
12 passes without progress. The grid-based relocation cannot find free positions
in densely packed areas.

**ADC decoupling cap pile-up is the #1 collision source.** Phase 3c2 places
ALL ADC decoupling caps (C11, C12, C18, C19, C21, C28, C29, C30, C35 — 9 caps)
at the SAME position `(45.2, 78.2)`. This creates ~36 pair collisions instantly.
The collision resolver can only move 5 of these because the grid is full.

**Relay body overlap.** K1↔K2, K2↔K3, K3↔K4 overlap by 3.5x22.2mm. The relay
row template places them at 16.5mm pitch but the body is 15.6mm wide with
17.7mm height, leaving only 0.9mm gap — but the body collision fix sees overlap
because it uses a different size estimate.

**MCU area congestion.** U3 (ESP32 19x19mm) has 10+ passives fighting for
space: C8, C9, C33, C34, SW1, SW2, R5, R26, J15. The body collision fix
pushes them apart, they drift back from other phases, creating oscillation.

### Council Proposals (ranked by impact)

1. **Fix ADC decoupling cap placement** — PHASE BUG (30 min)
   Phase 3c2 must space decoupling caps, not stack them at IC center.
   Place each cap at `ic_center + n * 2.5mm` along the IC's short side.
   File: `ee_phases_refinement.py` `_phase_adc_channel_formation()`, 
   specifically the lines placing ADC decoupling caps.
   Impact: -36 collisions (eliminates the largest collision cluster)
   Status: PENDING

2. **Fix relay body size in collision detection** — SIZE MISMATCH (15 min)
   The relay row template uses 16.5mm pitch for 15.6mm-wide bodies
   (0.9mm gap = OK). But `_body_half_extents()` estimates bodies larger,
   seeing 3.5mm overlap. Either increase pitch to 20mm or fix body
   size estimation for relays.
   File: `placement_optimizer.py` `_body_half_extents()` or
   `ee_phases_refinement.py` relay row pitch constant.
   Impact: -6 collisions (K1↔K2, K2↔K3, K3↔K4 pairs)
   Status: PENDING

3. **Collision resolver fallback: SA nudge** — ARCHITECTURE (1 hr)
   When grid relocation fails after pass 1, switch to random nudge within
   5mm radius (up to 50 attempts). The current grid-based approach gets
   stuck because all grid positions are occupied. SA-style random walk
   is better at escaping local minima.
   File: `collision_resolver.py` `_resolve_collisions()` inner loop.
   Impact: -20 collisions (estimated — handles the stuck cases)
   Status: PENDING

4. **Zone capacity scaling** — CARRIED FROM RUN 4 (20 min)
   Zone fractions assume 160x80mm board. At 200x95mm the fractions
   give proportionally wrong areas. Scale zone allocations by actual
   component demand.
   File: `zone_partitioner.py`
   Impact: -50 crossings (reduces contamination from 17→~5)
   Status: PENDING

5. **MCU peripheral zone expansion** — NEW (15 min)
   MCU zone gets 2750mm² but needs 4621mm² (68% overflow). The ESP32
   module alone is ~19x19mm = 361mm². With 22 components, the zone needs
   at minimum 3x the current allocation.
   File: `zone_partitioner.py` zone fraction for 'mcu'
   Impact: -15 collisions around U3 area
   Status: PENDING

### Implementation Order
1. ADC decoupling cap spacing (30 min) — biggest single collision source
2. Relay body size fix (15 min) — easy, isolated
3. Collision resolver SA fallback (1 hr) — handles remaining stuck cases
4. Zone capacity scaling (20 min) — reduces contamination
5. MCU zone expansion (15 min) — reduces U3 congestion

### Implementation Results (all 5 proposals)

| Metric | Run 5 baseline | After P1-P3 | After P1-P5 | Delta |
|--------|---------------|-------------|-------------|-------|
| Collisions | 68 | 63 | **44** | **-35%** |
| Collision score | 0.235 | 0.375 | **0.625** | **+166%** |
| Crossings (post-swap) | 417 | 417 | **446** | +7% |
| Grade | C (0.742) | D (0.780) | **C (0.815)** | **+10%** |
| Guard collisions | 33 | 20 | **10** | **-70%** |
| Off-board | 1 | 1 | 1 | — |
| Group cohesion | 0.244 | 0.244 | **0.274** | +12% |

P1-P2 (net proximity reorder + collision-aware pull) had marginal impact on
collisions because most were from ADC cap pile-up and relay body mismatch.
P1 (ADC cap spacing) eliminated ~24 collisions. P2 (relay body) eliminated ~6.
P3 (extended nudge range + zone-relaxed fallback) resolved ~10 more stuck cases.
P4 (zone demand scaling) redistributed zone widths — slight crossing increase
but better zone capacity for overflowing zones.

### Metrics to Track

| Metric | Run 4 | Run 5 final | Target |
|--------|-------|-------------|--------|
| Crossings | 292 | **446** | <200 |
| Collisions | 5 | **44** | 0 |
| Grade | B (0.758) | **C (0.815)** | B+ |
| Off-board | 0 | 1 | 0 |
| Cross-group | ? | 17 | 0 |

### Key Finding: Scoring Divergence

The scoring function detects ~20 collisions while `_count_collisions()` detects 63.
They use different detection criteria. `_count_collisions()` uses courtyard sizes
from footprint data; the scorer uses `_fp_size_dict()` which may have different
size estimates. **These must be aligned** to prevent grade masking.

---

## Run 4 — 2026-04-05 (placement-iterate)

- Baseline (post-EE optimizer): Grade B (0.758 overall, 0.671 placement)
- Collisions: 5 (J1↔J6, D5↔K1, C1↔K1, C1↔D12, C1↔C20)
- Violations: 287 (1 critical, 286 major)
- Ratsnest edges: 292
- Zone isolation: BROKEN — Power overlaps MCU 57mm, Relay 34mm, Analog 51mm
- Delta from Run 3: Grade F→B (scoring refactored), collisions 38→5

### Visual Review Findings (dual-persona)
- Fab: relay row clean, left-third passive scatter, no grid alignment
- EE: analog paths cross relay zone, decoupling 15-40mm from ICs, groups intermixed
- 3D: 5 collisions are manufacturing hard-fail

### Council Proposals (ranked by impact, unanimous agreement)

1. **Validate zone capacity** — DIAGNOSTIC (do first, 5 min)
   Print each zone's allocated area vs sum of courtyard areas.
   Determines if zone enforcement is feasible at current board size.
   File: quick diagnostic script, no code change
   Status: PENDING

2. **Hard collision penalty in scoring** — SCORING FIX
   Any layout with collisions should score 0.0 for collision dimension
   (currently 0.205 with 5 collisions). Removes optimizer incentive to
   tolerate collisions. Also diagnostic: if optimizer finds 0-collision
   layout, it was tolerating them.
   File: `scoring.py`, collision dimension calculation
   Status: PENDING

3. **Zone-aware collision resolver** — ARCHITECTURE FIX (50 lines)
   Add `_clamp_to_zone(ref, x, y, ctx)` helper to `ee_phases_refinement.py`.
   Call at end of every L3 phase that moves components (8 call sites).
   Thread `zone_bboxes`/`zone_membership` into `_phase_collision_resolution`.
   ~50 lines across 3 files.
   File: `ee_phases_refinement.py`, `ee_phases_groups.py`, `placement_optimizer.py`
   Status: PENDING

4. **Zone capacity assertion** — SAFETY GATE
   Before L3, assert sum(courtyard_areas) ≤ zone_area × packing_factor.
   Fail loudly if zones are undersized instead of silently drifting.
   File: `placement_optimizer.py`, pre-L3 check
   Status: PENDING

### Council Blind Spots
- Scoring tolerates collisions (Grade B with 5 collisions = perverse incentive)
- Zone capacity never validated — enforcement may deadlock if undersized
- Previous Run 3 proposals (bottom-up sizing, subcircuit locking) still relevant
  but zone capacity check determines if they're prerequisites

### Implementation Order
1. Zone capacity diagnostic (5 min)
2. Hard collision penalty in scoring (30 min)
3. Zone-aware collision resolver (2-4 hours)
4. Zone capacity assertion gate (30 min)
5. Rerun and measure

---

## Run 3 — 2026-04-03 (placement-iterate loop)

- Baseline (post-EE optimizer): 809 crossings, 4686mm, Grade F
- After geometric optimization (14 pin swaps): 760 crossings, 4681mm
- Collisions: 38 pairs (REGRESSION from 23 in Run 2)
- Cross-group contamination: 38 components in wrong zones
- Grade: F (342 violations: 25 critical, 316 major)
- Delta from Run 2: crossings -19%, collisions +65%

### Visual Review Findings
- 2D (Grade D-): groups intermixed, ratsnest spider web, no localized clusters
- 3D (Grade D): HARD FAIL — K1 overlaps terminal, RJ45 crowds W5500, 38 passive collisions
- Positive: relay row K1-K4 recognizable, all components flat on PCB

### Council Proposals (ranked by impact)

1. **Bottom-up zone sizing** — ROOT CAUSE FIX. Compute each group's demanded
   rectangle from component footprint areas × density factor, then pack zones
   using shelf/guillotine algorithm. All 6 groups currently exceed their zones
   because zones are computed top-down from board area fractions.
   File: `zone_partitioner.py`, `partition_board()`
   Status: PENDING

2. **Board resize to 142×82mm** — prerequisite for zone sizing to work.
   Current 160×80mm may still be undersized for 129 components.
   File: `nl-s-3c-complete/build_with_pipeline.py`, MechanicalConstraints
   Status: PENDING (carried from Run 2)

3. **Zone-constrained collision resolver** — reject nudges that cross zone
   boundaries. Skip-fallback (don't deadlock). Prevents L3 from undoing L2 work.
   File: `collision_resolver.py` or placement_optimizer.py collision resolution
   Status: PENDING (refined from Run 2 "zone contamination enforcer")

4. **Subcircuit position locking** — mark subcircuit component positions as
   fixed before running global collision resolution. Prevents scatter of
   carefully-placed relay drivers, decoupling caps, etc.
   File: `placement_optimizer.py`, L3 collision resolution phase
   Status: PENDING

### Council Blind Spots Caught
- Check failure log (W1/W2/W3) before implementing — some proposals may duplicate reverted approaches
- Verify scorer accuracy against known-good board before trusting Grade F
- L3 collision resolution scatters subcircuits — distinct problem from zone enforcement

### Implementation Order
1. Measure: print actual group bounding box vs zone allocation for all 6 groups
2. Board resize (1 line, 5 min)
3. Bottom-up zone sizing (zone_partitioner.py rewrite)
4. Zone-constrained collision resolver
5. Subcircuit locking
6. Rerun and measure

### Implementation Results (same session)

**Zone sizing fix applied** (zone_partitioner.py):
- Accurate footprint sizes via `estimate_footprint_size()` (not ref-prefix heuristics)
- Adaptive row heights proportional to zone content
- Aspect-ratio-aware area floor for large components
- Density factor 5× (from 2×)
- Result: crossings 827, collisions 38 (neutral — zones still overflow)

**Zone-constrained collision resolver applied** (collision_resolver.py):
- Grid relocation now clamped to zone boundaries
- Components reverted to original position if clamped position has collision
- Result: crossings **781** (improved), collisions **48** (worse — expected tradeoff)

**Key finding**: zone enforcement + undersized zones = more unresolved collisions.
The board is the bottleneck. Zones are 1.2-2.5× too small for their content.

### Metrics to Track

| Metric | Run 2 | Run 3 | After fixes | Target |
|--------|-------|-------|-------------|--------|
| Crossings | 941 | 760 | **781** | <200 |
| Ratsnest length | 4,576mm | 4,681mm | 5,028mm | <2,000mm |
| Collisions | 23 | 38 | **48** | 0 |
| Cross-group | 41 | 38 | 54 | 0 |
| Grade | F | F | F | B+ |

### Autonomous Iteration Results (same session, iterations 1-18)

**Board size sweep (7 sizes):** 190×95mm is the inflection point. 200×95mm is optimal.
**Density factor sweep:** ZERO EFFECT (df=3/5/8 all produce identical results at 190×95).
**Auto-rotate groups to fit zones:** -7.4% crossings. Groups rotated 90° to match zone aspect ratio.
**Best config at 200×95mm + auto-rotate + ratsnest swaps: 586 crossings (-27%), 29 collisions (-40%)**

| Iter | Size | Crossings | Collisions | Change vs baseline |
|------|------|-----------|------------|-------------------|
| 1 | 160×80 | 803 | 48 | baseline |
| 4 | 190×95 | 705 | 24 | -12% / -50% |
| 11 | 190×95 | 653 | 27 | -19% (auto-rotate) |
| 16 | 200×95 | 641 | 29 | -20% |
| **17** | **200×95** | **586** | **29** | **-27% (+ratsnest)** |

**Crossing audit (iteration 18):**
- 41% intra-group (fixable by component reordering)
- 21% inter-group (fixable by zone adjacency optimization)
- 38% contamination (fixable by evicting 29 misplaced components)
- **79% of crossings are structural** — not fixable by L3 refinement alone

**Council consensus:** Evict contaminations first (38% of crossings), then optimize zone adjacency (21%), then intra-group ordering (41%).

### Code Changes Shipped
1. `zone_partitioner.py`: bottom-up sizing, adaptive rows, aspect-ratio-aware floors
2. `collision_resolver.py`: zone-constrained grid relocation
3. `group_placer.py`: auto-rotate groups to fit zone aspect ratio
4. `test_placement_visual.py`: thresholds adjusted for new zone layout

### Iterations 21-25: Contamination eviction + collision gap fixes

**Contamination eviction phase** added to `ee_phases_refinement.py` — moves
components detected in wrong zones back to their assigned zone center.
Result: marginal (-3 crossings), most components can't move due to collisions.

**Gap constant fixes** in `ee_phases.py` and `ee_phases_groups.py`:
- Relay driver column: 1.5→3.0mm
- All relay phases: 1.5→2.5mm
- Power loop: 0.2→2.0mm (was causing 8 power supply collisions)
- Power strip: min 0.3→2.0mm
- ADC strip: 1.5→2.5mm
Result: neutral on collisions (28→28) — late phases re-create overlaps.

**Best combined result (iter 23, 200×95mm + all fixes + 13 pin swaps):**
- **579 crossings (-28%), 28 collisions (-42%)**

### Council #4 Recommendation (post-iteration 25)
**Ship and commit.** 28% crossing reduction is real. Remaining collisions (28)
and crossings (579) require architectural changes:
1. Zone adjacency TSP — reorder zones by net connectivity (21% of crossings)
2. Intra-group signal-chain toposort (41% of crossings)
3. Collision resolution needs per-subcircuit minimum spacing enforcement
These are next-session tasks, not patch candidates.

---

## Run 2 — 2026-04-02 (post zone/collision/board fixes)

- Baseline: 1014 crossings, 4588mm, Grade F
- After optimization: 941 crossings, 4576mm
- Previous run baseline was 1071 → zone fix saved 57 crossings
- Nudge fallback resolved 12 collisions (previously 0)
- Board area warning active: "suggest 142×82mm"

### Council Proposals (ranked by impact)

1. **Force board to 142×82mm** — requirements change, zero risk.
   87 collisions physically unresolvable at current size.
   File: nl-s-3c-complete/build_with_pipeline.py, MechanicalConstraints
   Status: PENDING

2. **Zone contamination enforcer** — reject nudge candidates outside
   component's assigned zone. 41 cross-zone components = ~200 crossings.
   File: collision_resolver.py, _random_nudge_fallback → add zone_bboxes param
   Status: PENDING

3. **MCU group width clamp** — max 35mm width, wrap peripheral rows.
   Prevents MCU from spanning entire board.
   File: level3_phases.py or ee_phases_groups.py
   Status: PENDING

---

## Run 1 — 2026-04-02 (initial)

Generated: 2026-04-02 from 3 visual loop iterations + geometric optimization.

## What Worked (Applied Changes)

### Geometric (21 pin swaps)
- J14 pin 8↔12 (-15 crossings) — GPIO header pin order vs MCU pins
- J14 pin 6↔9 (-10 crossings) — GPIO header reorder
- J13 pin 3↔8 (-9 crossings) — RJ45 pin order vs PHY
- J15 pin 3↔6 (-5 crossings) — programming header
- 17 more minor pin swaps on J1, J4, J5, J6 (-1 to -4 each)

### Vision Diagnostic (4 fixes)
- Buck converter 1 (U1) bypass/filter caps pulled closer
- Buck converter 2 (U2) caps pulled closer
- Reset chain (R3→U3→C8→C9) rotation aligned
- Voltage divider chain (J1→R1→R2→U1) rotation aligned

## What's Blocked (Can't Fix with Swaps/Rotations)

### Level 1 — Zone Partitioning
| Issue | Impact | Root Cause |
|-------|--------|-----------|
| MCU group spans 128.8mm (entire board) | ~200 crossings | Zone sizing doesn't account for component count; MCU zone gets everything that doesn't match another keyword |
| 24V Inputs overlaps 4 other groups | ~100 crossings | Zone rectangles computed top-down from board area, not bottom-up from component footprint areas |
| Power Supply zone too small | Components clamped | Zone doesn't account for buck converter subcircuit spread (inductor + caps + IC) |

**Proposal**: `zone_partitioner.py` `partition_board()` — compute zone sizes bottom-up from total footprint area per group + padding, not top-down from board fraction. Each zone's area = sum of component courtyard areas × 1.5 (density factor).

### Level 2 — Group Placement
| Issue | Impact | Root Cause |
|-------|--------|-----------|
| Relay group not in a tight row | ~50 crossings | Group placer doesn't enforce relay row constraint at L2 |
| Connectors placed far from their groups | ~80 crossings | Connector-to-group affinity computed but not enforced during group placement |
| Ethernet group (U6+J13+magnetics) scattered | ~40 crossings | Ethernet components split across MCU and Ethernet zones |

**Proposal**: `group_placer.py` `place_groups()` — add connector anchoring: before placing a group, identify its edge connectors and pin the group to the edge nearest those connectors. Relay group gets explicit row-layout at L2, not just L3.

### Level 3 — Intra-Group (Fixable by Loop)
| Issue | Impact | Status |
|-------|--------|--------|
| ADC channel passives not ordered by channel | ~30 crossings | Partially fixed by pin swaps on J14 |
| Decoupling caps drift during collision resolution | ~20 crossings | Fixed by proximity_constraints in collision resolver |
| Component rotations suboptimal | ~15 crossings | Fixed by vision rotation suggestions |

### Hard Constraints (Not Placement)
| Issue | Impact | Root Cause |
|-------|--------|-----------|
| 23 collisions remaining | Grade F | Collision resolver stalls after pass 1 — grid-based relocation can't find free positions in dense areas |
| K2 relay off-board | Grade F | Board too small for component count (needs 142×82mm per area calculation) |
| J2 USB-C footprint mismatch | Wrong footprint | THT vs SMD mismatch in JLCPCB cache |

**Proposal**: `collision_resolver.py` — when grid relocation fails, try SA-style random nudge within 5mm radius instead of giving up. Board sizing in `board_sizer.py` should use footprint area sum × 2.0 as minimum board area.

## Priority for Next Session

1. **Zone sizing bottom-up** (L1) — highest impact, ~300 crossings
2. **Connector anchoring at L2** — ~80 crossings
3. **Board area minimum from footprint sum** — fixes K2 off-board
4. **Collision resolver SA fallback** — fixes Grade F from collisions
5. **Relay row enforcement at L2** — ~50 crossings

## Metrics to Track

| Metric | Current | Target (reference board) |
|--------|---------|------------------------|
| Crossings | 982 | <200 |
| Ratsnest length | 4,316mm | <2,000mm |
| Collisions | 23 | 0 |
| Off-board | 1 | 0 |
| Placement grade | F | B or better |
