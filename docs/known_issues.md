# Known Issues Registry

Tracks pipeline bugs that have been fixed, with regression tests to prevent recurrence.
Every CORE pipeline fix MUST have an entry here and a corresponding test in
`tests/regression/test_known_issues.py`.

| ID | Title | Root Cause | Guard | Fixed In |
|----|-------|-----------|-------|----------|
| KI-001 | Ref designator shows `?` | Sub-sheet `sheet_instances` path used `"/"` instead of hierarchical `"/{root_uuid}/{sheet_entry_uuid}"` | `_validate_no_question_mark_refs()` in builder + regression test | Sprint 16 |
| KI-002 | Schematic-PCB component desync | Independent `_enrich_requirements()` calls in schematic and PCB builders could diverge | `check_consistency()` hard gate in VALIDATION stage | Sprint 17 |
| KI-003 | Footprint mismatch SCH↔PCB | Variant remapping applied inconsistently between schematic and PCB generation | `check_consistency()` footprint match check | Sprint 17 |
| KI-004 | Centroid offset duplicated 4x | `_centroid_offset()` reimplemented in placement_optimizer, review_agent, scoring, placement_render instead of using pin_map.py | `compute_centroid_offset()` in pin_map.py is single source of truth; alias in optimizer | v6 centroid consolidation |
| KI-005 | Connectors off-board (J1,J3-J6,J13) | `_orient_connectors()` rotation/sizing logic + phase 3f2 ordering doesn't properly space screw terminals; J13 not clamped to board | **OPEN — needs regression test for all-components-within-board-bounds** | Not yet fixed |
| KI-006 | MCU group overlapping components | Collision resolution (phase 3g) blocked by protection sets (`subcircuit_fixed`, `template_protected`) leaving overlaps in MCU zone | **OPEN — needs overlap audit of protection sets** | Not yet fixed |
| KI-007 | Relay driver components scattered | Q/D/R driver components placed 15-25mm below relays instead of tight 5-8mm subgroups. Phase 3b tightening not pulling close enough. | **OPEN — need to verify 3b phase actually runs, check target distances** | Not yet fixed |
| KI-008 | Power tail overflow to y=70 | Power column layout overflows zone boundary — tail components (L3-L6, C6, C7, R3, C35) placed at y=70mm despite power zone ending at y=40mm | **OPEN — zone height too small or columns too long** | Not yet fixed |
| KI-009 | ADC channel order vs screw terminals | ADC channel strips don't follow screw terminal connector order — causes crossing traces instead of direct routing | **OPEN — reorder channels by connector proximity** | Not yet fixed |
| KI-010 | Analog/power blocks too high | Power and analog zones start at y=15% (12mm) instead of y=20% (16mm) — components interfere with screw terminal area | **OPEN — adjust zone top margin** | Not yet fixed |
| KI-011 | Buck converter false positives | `_detect_buck_converters()` follows shared power rails (GND, +3V3, +5V) claiming 19+ components for one buck | `_is_power_net()` filter + `_MAX_CAPS_PER_NET=2` + `_MAX_FB_RESISTORS=2` limits | Framework v2.0 |
| KI-012 | TVS diode misclassified as flyback | Relay driver detection claims TVS/LED diodes as flyback protection diodes | Value/description filtering: skip `"TVS"` and `"LED"` in `_detect_relay_drivers()` | Framework v2.0 |
| KI-013 | Cross-group decoupling contamination | Decoupling detection assigns caps to ICs in different FeatureBlocks, scattering groups | Same-group preference in `_detect_decoupling_pairs()` + cross-group cap filtering | Framework v2.0 |
| KI-014 | Phase ordering scatters decoupling caps | Phase 3c places caps near ICs, then group phases (3c1-3c4) move them away | Phase 3c-late: re-pull after all group phases, respecting FeatureBlock boundaries | Framework v2.0 |
| KI-015 | Unrealistic subgroup cohesion thresholds | Scoring thresholds (8mm relay, 5mm decoupling) don't account for component physical sizes | Size-aware thresholds: relay=22mm, decoupling=10mm, buck=18mm | Framework v2.0 |
| KI-016 | Post-optimization validation missing | No gate to catch off-board, collisions, cross-group contamination after optimization | `validate_placement()` gate runs after EE placement, logs all issues | Framework v2.0 |

## Adding a New Entry

1. Assign the next `KI-NNN` ID
2. Fill in root cause and guard (test function name)
3. Add a regression test to `tests/regression/test_known_issues.py`
4. Reference the commit or sprint that fixed the issue
