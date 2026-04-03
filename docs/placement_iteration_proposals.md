# Placement Iteration Proposals — nl-s-3c-complete

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
