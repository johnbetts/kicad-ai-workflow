# Bottom-Up Placement Rewrite Plan

## Why

The current 25-phase L3 engine places by component TYPE, not by NET CONNECTIVITY. This produces boards where electrically-connected components are 30-80mm apart (65+ violations on a 160x80mm board). Post-hoc enforcement can't fix this because pulling components toward one IC creates violations on other shared nets. The architecture is fundamentally wrong.

## The New Architecture

Replace the 25-phase pipeline with 4 stages:

```
Stage 1: Subcircuit Layout (parallel, per-subcircuit)
Stage 2: Group Packing (subcircuits → groups, polygon collision)
Stage 3: Board Packing (groups → board, rotation/translation)
Stage 4: Enforcement (DRC rules, edge pinning, keepout zones)
```

### Stage 1: Subcircuit Layout

**Input**: Detected subcircuits + netlist + component sizes
**Output**: Per-subcircuit positioned layout + bounding polygon

For each subcircuit (relay_driver, buck_converter, voltage_divider, decoupling, crystal_osc, adc_channel, esd_protection, etc.):

1. Find anchor IC and all support components connected by nets
2. Place anchor at (0,0)
3. Place each support component at its IC pin position + offset along the net direction
4. Generate tight convex hull polygon around the placed subcircuit
5. Validate: all net-connected pairs within subcircuit are <10mm

**Key principle**: Layout is driven by PIN CONNECTIVITY, not component type. A decoupling cap goes next to the IC pin it decouples. An ESD protector goes next to the connector pin it protects.

**Parallelizable**: Each subcircuit is independent. 10 subcircuits = 10 parallel agents.

**New subcircuit types to detect**:
- `ESD_PROTECTION`: U9 protecting J2 (connected via USB_DP/USB_DM)
- `POE_FILTER`: C31/C32 connected to J13 via POE nets
- `OPTOCOUPLER_CIRCUIT`: U7 + R25/R32/D17/LED2/SW3

### Stage 2: Group Packing

**Input**: Subcircuit polygons + FeatureBlock group assignments
**Output**: Per-group positioned layout + group polygon

For each FeatureBlock group (Relay Outputs, Power Supply, MCU, Analog Inputs, Ethernet):

1. Collect all subcircuit polygons belonging to this group
2. Pack subcircuit polygons using polygon collision avoidance
3. Place inter-group bridge components (shared nets) at polygon edges nearest the connected group
4. Generate group polygon (convex hull of all subcircuit polygons)
5. Export as KiCad group for user manipulation

**Packing algorithm**: 
- Start with largest subcircuit at center
- Place next-largest adjacent, minimizing shared-net wire length
- Repeat until all subcircuits placed
- No component-level collision resolution needed — subcircuits are pre-validated rigid units

### Stage 3: Board Packing

**Input**: Group polygons + board dimensions (or oversized)
**Output**: Positioned board layout

1. Start with oversized board (2x target)
2. Place groups using polygon collision avoidance with rotation
3. Optimize group positions to minimize total ratsnest wire length
4. Shrink board outline to minimum bounding rectangle + margin
5. If target size specified, iterate compression until fit

**Rotation**: Try 0°, 90°, 180°, 270° for each group, pick the one with shortest ratsnest to adjacent groups.

**Connector edge pinning**: THT connectors are placed at the nearest board edge BEFORE group packing, then the group is packed around them.

### Stage 4: Enforcement

**Input**: Positioned board
**Output**: Final validated board

Run all 7 DRC rules (from docs/drc_rules_spec.md):
1. RULE-001: No pads off board edge (auto-fix: pull inward)
2. RULE-002: Zero courtyard collisions (auto-fix: push apart)
3. RULE-003: Subcircuit components adjacent to anchor (VERIFY, not fix — Stage 1 guarantees this)
4. RULE-004: Connector groups contiguous (auto-fix: slide along edge)
5. RULE-005: RF antenna at board edge with keepout (auto-fix: rotate/shift)
6. RULE-006: Components in correct group zone (log only)
7. RULE-007: THT connectors at edge facing outward (auto-fix: push to edge)

Plus net proximity audit: flag any non-power net pair >30mm.

## What to Keep from Current Framework

| Module | Keep? | Why |
|--------|-------|-----|
| `geometry.py` | YES | Polygon operations |
| `BoardZone` | YES | Polygon zones with contains/clamp |
| `reference_comparator.py` | YES | Diagnostic comparison |
| `relay_template.py` | REPLACE | Becomes Stage 1 relay_driver subcircuit |
| `scoring.py` | REFACTOR | Add net proximity as primary metric |
| `placement_guard.py` | YES | Stage 4 enforcement |
| `functional_grouper.py` | EXTEND | Add new subcircuit types (ESD, POE, optocoupler) |
| `zone_partitioner.py` | SIMPLIFY | Only needed for initial group anchor hints |
| `group_placer.py` | REPLACE | Becomes Stage 2 polygon packer |
| `ee_phases.py` | DELETE | Replaced by Stages 1-3 |
| `ee_phases_groups.py` | DELETE | Replaced by Stage 1 subcircuit layouts |
| `ee_phases_refinement.py` | KEEP PARTIALLY | Stage 4 enforcement functions |
| `collision_resolver.py` | SIMPLIFY | Only Stage 4, not mid-pipeline |
| `subnet_placer.py` | DELETE | Replaced by Stage 1 pin-connectivity placement |
| `placement_optimizer.py` | REWRITE | New 4-stage pipeline entry point |

## Implementation Order

### Phase A: New Subcircuit Detector (2 hours)
Extend `functional_grouper.py` to detect:
- ESD_PROTECTION (U* near J* via signal nets)
- POE_FILTER (C* on POE nets)
- OPTOCOUPLER_CIRCUIT (U7 + connected passives)
- GENERIC_IC_CLUSTER (any U* + its directly-connected R/C/D within 2 net hops)

### Phase B: Stage 1 — Subcircuit Layout Engine (4 hours)
New `subcircuit_layout.py`:
- `layout_subcircuit(subcircuit, pcb) → SubcircuitLayout`
- Pin-connectivity-driven placement (IC pin → passive position)
- Polygon generation
- Per-subcircuit validation (all pairs <10mm)

### Phase C: Stage 2 — Group Polygon Packer (3 hours)
Extend `group_placer.py`:
- `pack_subcircuits_into_group(subcircuit_layouts, group) → GroupLayout`
- Polygon collision avoidance
- Bridge component placement at polygon edges

### Phase D: Stage 3 — Board Packer (2 hours)
New `board_packer.py`:
- `pack_groups_on_board(group_layouts, board_bounds) → BoardLayout`
- Rotation optimization
- Board shrink-to-fit

### Phase E: Stage 4 — Enforcement (1 hour)
Wire DRC rules from `placement_guard.py` as auto-fix pass.

### Phase F: Integration + Testing (2 hours)
- New `optimize_placement_ee()` that calls Stages 1-4
- Run on nl-s-3c-complete board
- Compare violations count: target <10 (from current 65+)
- Visual inspection + human review

## Total Estimate: 14 hours (2 sessions)

## Success Criteria

1. Net proximity: <10 pairs >30mm apart (from 65+)
2. Courtyard collisions: 0 (from 17)
3. Guard passed: True with 0 RECURRING
4. All protection components within 5mm of what they protect
5. All decoupling caps within 3mm of their IC
6. Visual grade: B+ or higher from human review
7. Relay group: clean 1x4 row with driver columns (already works)

## Risk

The 25-phase engine encodes hard-won EE domain knowledge (relay coil direction, crystal proximity, buck signal flow ordering). The subcircuit layout engine in Phase B needs to replicate this knowledge for each subcircuit type. If the layout templates are wrong, the board will be electrically incorrect even if proximities pass.

Mitigation: Keep the relay_template.py approach — each subcircuit type has a hardcoded layout template validated against datasheets. The templates are the EE knowledge, extracted from phases into declarative form.
