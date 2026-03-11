# Placement Engine Iteration Log

Track errors found, fixes attempted, and results to avoid repeating mistakes.

## Iteration Rules
- **Never re-try a fix that already failed** — try a different approach
- **Measure before and after** — use actual distances, not visual guesses
- **Fix root cause, not symptoms** — if distances are wrong, fix the source, not downstream

---

## Errors Found (v5 baseline)

### E1: Relay drivers 10-19mm from relays (goal <8mm)
- **Root cause**: `_layout_group()` Step 7.5 (relay post-processing) runs AFTER `_resolve_overlaps` (Step 7). Overlaps push Q/D/R components away from K, then Step 7.5 stacks them in a column with 1.5mm gaps that accumulate to 20mm+.
- **Fix applied**: Swap Steps 7 and 7.5 — do relay post-processing FIRST, then resolve overlaps. Use tight 2-column grid (0.5mm gap) instead of single-column (1.5mm gap).
- **Status**: In progress

### E2: K3 relay placed 56mm from other relays (should be in 1x4 row)
- **Root cause**: K3 ends up in a different zone area. The group placer places the relay group as a rigid unit, but collision resolution in Level 3 displaces K3. The relay row formation (Level 3a `_place_row_layout`) should fix this but it may not be aggressive enough.
- **Fix needed**: Ensure Level 3a places ALL K refs in a single row regardless of initial positions. K refs should be treated as a fixed row unit after Level 3a.
- **Status**: Pending (Task #12)

### E3: Decoupling cap C10 is 16.7mm from U3 (goal <3mm)
- **Root cause**: U3 and C10 are in the same group (MCU), so `_layout_group()` should place them close. The distance is from Level 3 collision resolution scattering them. Level 3c tries to tighten but can't overcome the large initial displacement.
- **Fix needed**: Investigate why C10 ends up so far from U3 after Level 2 group placement. May need to ensure collision resolution doesn't push decoupling caps away from their ICs.
- **Status**: Pending (Task #12)

### E4: Zone fractions don't match reference layout
- **Root cause**: `_DEFAULT_ZONE_FRACTIONS` had overlapping zone rectangles, causing groups to intermix. In iteration attempts, zones were rearranged but then groups were larger than zones and spilled out.
- **Fix attempted (reverted)**: Rearranged zone fractions + asymmetric compression. Made things worse because compression distorted internal group layouts.
- **Lesson learned**: Don't compress groups to fit zones — fix the group sizes instead. Or use dynamic zone sizing based on actual group dimensions.
- **Status**: Will address after sub-circuit clustering is fixed

### E5: Connectors not on board edges
- **Root cause**: `pin_connectors_to_edge()` has an 8mm threshold — connectors already within 8mm of edge are skipped. Many connectors are placed mid-board by the group placer and never reach 8mm proximity to trigger pinning.
- **Fix attempted (reverted)**: Reduced threshold to 3mm. Made some connectors edge-pinned but didn't fix the fundamental issue.
- **Status**: Will address after sub-circuit clustering is fixed

---

## Fixes That Made Things Worse (DO NOT REPEAT)

### W1: Asymmetric group compression to fit zones
- **What**: Scaled group internal layouts to fit within zone rectangles. Applied scale_x and scale_y independently, capped at 0.6 (40% max compression).
- **Why it failed**: Compression distorted carefully-constructed internal layouts. Relay row got squished from 79mm to 46mm, making relays stack 2x2 instead of 1x4. Power group got squished destroying signal flow ordering.
- **Lesson**: Never compress/scale internal group layouts. Instead, make zones big enough for the groups, or make groups produce smaller internal layouts.

### W2: Zone-order placement sorting (top-to-bottom, left-to-right)
- **What**: Changed group placement order from largest-first to zone-position-order.
- **Why it failed**: Earlier groups took up space that later groups needed, causing the later groups to be displaced to wrong zones. MCU and Ethernet swapped positions.
- **Lesson**: Largest-first placement is better for collision avoidance. Fix zone sizing to accommodate groups rather than changing placement order.

### W3: Zone-constrained grid search
- **What**: Created a zone-constrained `_GroupGrid` for each group, falling back to global grid if zone was full.
- **Why it failed**: Zone grids found positions at zone edges that the global grid then rejected, causing fallback to random global positions.
- **Lesson**: Use a single global grid with zone-center targeting. The spiral search naturally stays close to the target.

---

## Key Constants & Their Impact

| Constant | Current | Effect |
|----------|---------|--------|
| `max_row_width` (relay) | 999.0 | Relays always in single row (79mm for 4 relays) |
| `max_row_width` (general) | 120.0 | Very wide rows, large groups |
| `_ANCHOR_GAP_MM` | 2.0 | Gap between anchors in a row |
| `_PASSIVE_STANDOFF_MM` | 1.0 | Gap from anchor pin to passive |
| `_PASSIVE_STACK_GAP_MM` | 0.5 | Gap between stacked passives |
| `_GROUP_MARGIN_MM` | 2.0 | Margin around group boundary |
| `_ZONE_GAP_MM` | 5.0 | Gap between zones |
| `_CONNECTOR_EDGE_MARGIN_MM` | 3.0 | Connector distance from edge |

## Measurement Targets

| Metric | Target | Current |
|--------|--------|---------|
| K-to-Q distance | <5mm | 21mm |
| K-to-D distance | <8mm | 10-17mm |
| Relay row spread (Y) | <2mm | 56mm |
| Decoupling cap distance | <3mm | 3-17mm |
| Crystal-MCU distance | <10mm | unknown |
| Connector edge distance | <3mm | many >10mm |
