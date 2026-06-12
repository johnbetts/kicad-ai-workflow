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
| KI-017 | Missing 3D models for some components | `footprint_3d_model()` coverage gaps — some package types (e.g. opto-couplers, DIP switches, certain connectors) not in `_3D_MODEL_MAP` | Extend `_3D_MODEL_MAP` and add fallback by package keyword | **OPEN** |
| KI-018 | ESP32 antenna keepout doesn't follow module | `_pin_rf_to_edge()` correctly pins ESP32 to edge, but antenna keepout was created BEFORE placement with hardcoded fallback (same root cause as KI-019) | Fixed together with KI-019: post-placement keepout creation | Fixed with KI-019 |
| KI-019 | Orphan isolation zone at fixed board corner | `_make_antenna_keepout()` falls back to hardcoded top-right corner (board_width-15, 0) when ESP32 position not in `fixed_positions` — creates orphan zone regardless of actual placement | Post-placement keepout creation; regression test `TestKI019AntennaKeepoutPosition` | 2026-03-15 |
| KI-020 | Mounting hole keepouts without NPTH holes | `_make_mounting_hole_keepouts()` creates keepout zones but `make_mounting_hole()` footprints only added when `template_mounting_positions` is set; `requirements.mechanical.mounting_hole_positions` path skips footprint creation | Unified footprint creation for all mounting position sources; regression test `TestKI020MountingHoleFootprints` | 2026-03-15 |
| KI-021 | Connector mating face not flush with board edge | `_orient_connectors()` uses 3mm edge margin keeping ALL pads inside board; USB-C, RJ45, etc. need mating face flush/protruding past edge for cable access | Connector type detection via `edge_mount_keywords`; 0mm margin for edge-mount types, clamping exemption on mating side | 2026-03-15 |
| KI-022 | JLCPCB footprint fallback selects wrong package size | When JLCPCB footprint lookup rejects a part (e.g. pad size mismatch), parametric fallback may select a different package size (R_0603→R_0402). DFM gate `package_match` detects this. | **OPEN — fix footprint fallback to respect requirements package spec** | Not yet fixed |

| KI-023 | Generated schematics fail to LOAD in KiCad 10 | Symbol factory packages (`schematic/symbols/{active,passive,power}`) passed NUMERIC pin electrical types (`pin_type=1`) emitted verbatim as `(pin 1 line`; top-level `(embedded_fonts no)` (mandatory since format 20260101) was never written. KiCad refused every generated `.kicad_sch`; the in-memory sync gate compared ref sets only and was blind to it | Fixed 2026-06-12: keyword pin types + embedded_fonts in builder; guarded by `tests/integration/test_sch_pcb_sync.py` (kicad-cli round-trip) | 2026-06-12 |
| KI-024 | Schematic wiring diverges from netlist | The written schematic's KiCad-derived netlist disagrees with the written PCB pad nets on some topologies (ethernet: caps on invented `AVDD_U1_DEC`; analog: `AINx_DIV` vs `AINx_PROT`; nl-s-3c: pins merged onto `K1_COIL`) — suspected overlapping generated wires creating unintended junctions. 'Update PCB from Schematic' would silently rewire the board. PCB matches requirements; the SCHEMATIC is wrong | Fixed 2026-06-12 in three layers: netlist lint rejects shared-pin net splits (`check_duplicate_pin_nets` — trainers' EN_DEB/XTAL_C/AIN_PROT 'private subnets' merged into their true nodes); power-symbol consolidation disabled on multi-net symbol sides (collinear bus overlap) with per-net staggered depths; `wiring.resolve_stub_collisions` shortens/flips stubs touching foreign nets, with bus net-evidence + structural-stub rules. All 5 trainers pass the written-file sync gate end to end | 2026-06-12 |

## Adding a New Entry

1. Assign the next `KI-NNN` ID
2. Fill in root cause and guard (test function name)
3. Add a regression test to `tests/regression/test_known_issues.py`
4. Reference the commit or sprint that fixed the issue
