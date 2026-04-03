# Placement Iteration Proposals — nl-s-3c-complete

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
