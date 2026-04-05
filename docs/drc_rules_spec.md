# DRC Rules Specification

## 7 Parametric Rules to Prevent Recurring Placement Bugs

Each rule has: ID, pass/fail condition, auto-fix action, guard check.

---

### RULE-001: No Pads Off Board Edge

**Issue**: J13 (RJ45) pads extend past board outline.

**Pass condition**: For every footprint, ALL pad centers (in board space) must be inside the board outline polygon with ≥0.5mm margin.

**Check**: `for pad in fp.pads: board_contains(pad_board_position, margin=0.5)`

**Auto-fix**: If a THT connector's pads extend past the edge, pull the footprint inward by the overhang amount. Connectors are allowed to have their BODY overhang (plastic housing extends past edge for cable access), but pads must stay inside for copper/drill.

**Severity**: CRITICAL (fabrication blocker)

---

### RULE-002: Zero Courtyard Collisions

**Issue**: 19 courtyard overlaps across 10 component pairs.

**Pass condition**: No two footprint courtyard bounding boxes overlap (0.15mm minimum gap).

**Check**: AABB overlap test for all `(n*(n-1)/2)` pairs, using rotation-aware courtyard sizes.

**Auto-fix**: For each colliding pair, push the SMALLER component away from the larger by `overlap_distance + 0.5mm` in the direction away from the larger component's center. Respect board bounds. Run iteratively until 0 collisions or 10 passes.

**Severity**: CRITICAL (DRC violation, fabrication may reject)

---

### RULE-003: Subcircuit Components Adjacent to Anchor

**Issue**: Relay driver Q/D/R scattered instead of in columns below each relay.

**Pass condition**: For each detected subcircuit, ALL support components must be within `max_distance_mm` of their anchor IC:
- relay_driver: 15mm
- decoupling: 5mm
- buck_converter: 10mm
- voltage_divider: 8mm
- crystal_osc: 5mm

**Check**: `for ref in sc.refs: dist(ref, anchor) <= threshold[sc.type]`

**Auto-fix**: Components exceeding the threshold are pulled toward their anchor along the vector from current position to anchor, stopping at the threshold distance.

**Severity**: MAJOR (placement quality, signal integrity)

---

### RULE-004: Connector Groups Contiguous (No Gaps)

**Issue**: Screw terminals J1,J3-J6 split into two groups with gap.

**Pass condition**: All connectors assigned to the same edge (by the connector edge-pinning phase) must be contiguous — the gap between any two adjacent connectors on the same edge must be ≤ `max_gap_mm` (default: 5mm body-to-body).

**Check**: Sort connectors by position along their shared edge. For each consecutive pair, `gap = pos[i+1] - pos[i] - width[i]/2 - width[i+1]/2`. Fail if `gap > max_gap_mm`.

**Auto-fix**: Slide connectors along the edge to close gaps, centering the group on the edge midpoint.

**Severity**: MAJOR (assembly, routing)

---

### RULE-005: RF Antenna Faces Board Edge with Keepout

**Issue**: ESP32 antenna facing mid-board, components on antenna side.

**Pass condition**: 
1. The antenna end of the RF module must be within 5mm of a board edge.
2. No non-GND components within the keepout zone polygon.
3. A keepout zone polygon exists within 15mm of the antenna end.

**Check**: Compute antenna end from module centroid + rotation + module_half_height. Check distance to nearest board edge. Check no component centroids inside keepout polygon.

**Auto-fix**: Rotate or shift the RF module so the antenna end faces the nearest board edge. Move any violating components outside the keepout zone.

**Severity**: CRITICAL (RF performance, certification)

---

### RULE-006: Components Stay in Assigned Group Zone

**Issue**: 11 components placed in wrong group's zone.

**Pass condition**: For each component, if it's assigned to group G (from FeatureBlock), and group G maps to zone Z, then the component's centroid should be within Z's polygon OR within 5mm of Z's boundary (tolerance for edge components).

**Check**: `zone.contains(cx, cy) or dist_to_zone_boundary < 5mm`

**Auto-fix**: Log only (soft rule). Cross-zone placement is sometimes intentional for signal routing. Escalate to RECURRING only if the component is in a DIFFERENT group's zone (not just outside all zones).

**Severity**: MAJOR (organization, routing complexity)

---

### RULE-007: THT Connectors Face Off-Board with Body at Edge

**Issue**: USB-C position unclear, connectors not obviously at edges.

**Pass condition**: Every THT connector (attr=through_hole, ref starts with J) must have:
1. Its nearest edge distance ≤ 5mm (body at board edge)
2. The connector opening/cable entry direction faces AWAY from board center

**Check**: For each J ref with attr=through_hole:
- `min(dist_to_edge for all 4 edges) <= 5mm`
- Connector's cable-entry direction (determined by footprint type) points toward the nearest edge

**Auto-fix**: Push THT connectors to the nearest edge. Rotate so cable entry faces outward. SMD connectors (USB-C, SD card) use similar logic but with different distance thresholds.

**Severity**: MAJOR (assembly, enclosure compatibility)

---

## Implementation Priority

1. **RULE-001** (pads off board) — immediate, prevents fab rejection
2. **RULE-002** (courtyard collisions) — collision resolver already exists, needs final-pass enforcement
3. **RULE-007** (THT connector edges) — connector enforcement exists, needs pad-extent check
4. **RULE-005** (antenna keepout) — guard exists, needs auto-fix for antenna-facing-edge
5. **RULE-003** (subcircuit adjacency) — scoring dimension exists, needs enforcement pass
6. **RULE-004** (connector contiguity) — new rule, straightforward
7. **RULE-006** (zone membership) — soft rule, logging + monitoring

## Guard Integration

All rules should:
- Run in `validate_placement()` after all phases complete
- Fire as RECURRING if violated (blocks build)
- Each rule has a `_guard_RULE_NNN()` function
- Auto-fix runs in `_enforce_connector_rules()` or a new `_enforce_drc_rules()` pass
