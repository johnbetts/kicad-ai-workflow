# Placement Issue Tracker — nl-s-3c-complete vs Reference Design

Reference: `rereference/nl-s-3c-complete-2026-03-09_234333 - reference design/`
Generated: Current pipeline output with legacy optimizer + bug fixes

## CRITICAL (blocks manufacturing/safety)

| ID | Issue | Root Cause | Fix Location |
|----|-------|------------|--------------|
| PLI-001 | Analog screw terminals on TOP edge — should be LEFT edge | Connector edge-pinning uses type-based heuristic, not functional association. Reference has analog terminals on LEFT, relay terminals on TOP | `ee_phases.py` _phase_all_connectors_to_edges + _phase_top_edge_connectors |
| PLI-002 | No board slots between relay contacts and logic | Pipeline has no slot/cutout generation. Reference has U-shaped isolation cutouts | `pcb/builder.py` — new feature: board slots |
| PLI-003 | Relay drivers (Q/D/R) scattered, not in organized columns near relays | Phase 3b places drivers but collision resolution scatters them | `ee_phases.py` _phase_relay_drivers + collision protection |
| PLI-004 | Display header extends off board edge | Pin header placement doesn't respect board bounds for large THT components | `ee_phases_refinement.py` _enforce_tht_connectors_to_edge |

## MAJOR (placement quality)

| ID | Issue | Root Cause | Fix Location |
|----|-------|------------|--------------|
| PLI-005 | Voltage divider chains scattered — should be horizontal rows between terminals and ADC ICs | ADS1115 ADC channels not detected (1/8). Only MCU-direct ADC patterns recognized | `functional_grouper.py` _detect_adc_channels |
| PLI-006 | Buck converter loops not tight (IC+L+Cin+Cout not adjacent) | Power chain flow ordering exists but doesn't enforce tight loop area | `ee_phases.py` _phase_power_chain_flow |
| PLI-007 | ESP32-S3 crammed into right corner — should be bottom-center | MCU zone placed at right side of board by zone partitioner | `zone_partitioner.py` _DEFAULT_ZONE_FRACTIONS |
| PLI-008 | RJ45 missing 3D model | Footprint generator doesn't assign 3D model path for RJ45_Amphenol | `pcb/footprints.py` model mapping |
| PLI-009 | PoE module footprint incorrect/broken | JLCPCB footprint may have wrong geometry | `pcb/footprints.py` PoE module handling |
| PLI-010 | 40% board area empty (upper-right) while left side cramped | Zone sizing doesn't match reference layout proportions | `zone_partitioner.py` zone fractions |
| PLI-011 | USB-C not at bottom-left (reference position) | Connector type→edge mapping puts USB at bottom but not specifically bottom-left | `ee_phases.py` _phase_all_connectors_to_edges |
| PLI-012 | W5500 + crystal + caps not clustered tight | Ethernet group phase exists but doesn't enforce tight clustering | `ee_phases.py` _phase_ethernet_group |
| PLI-013 | Relay U-cutout wrong shape — large U in middle instead of isolating common from coil | Keepout/slot geometry doesn't match relay pin layout | `pcb/builder.py` or `zones.py` relay keepout |

## MINOR (cosmetic/refinement)

| ID | Issue | Root Cause | Fix Location |
|----|-------|------------|--------------|
| PLI-014 | Component density imbalance — right side cramped, left sparse | Zone proportions don't match reference | `zone_partitioner.py` |
| PLI-015 | Screw terminal spacing uneven along top edge | Terminal width calculation doesn't account for mixed sizes | `ee_phases.py` _phase_top_edge_connectors |
| PLI-016 | No "Generator AutoStart board v1.0" silkscreen text | Pipeline doesn't add project title to silkscreen | `pcb/silkscreen.py` |
| PLI-017 | Anchor gap too large (8mm vs reference 4-5mm) | _ANCHOR_GAP_MM constant too conservative | `pcb/placement.py` |

## Reference Layout Zone Map

```
+------------------------------------------+
|  J1(6p)  J3(3p)   [RELAY TERMINALS]     |
|  ANALOG SECTION    |  RELAY SECTION       |
|  Voltage dividers  |  K1 K2 K3 K4       |
|  ADC ICs           |  [Board slots]      |
|  J4 J5 J6 (LEFT)   |  Q/D/R columns     |
|--------------------+---------------------|
|  POWER SECTION     |  TRANSITION ZONE    |
|  Buck converters   |  Ferrite beads      |
|  U1→L1→C_out       |  W5500 + crystal    |
|  U2→L2→C_out       |                     |
|--------------------+---------------------|
|  USB-C  PoE        |  ESP32-S3  Display  |
|  (bottom-left)     |  header   RJ45      |
|                    |  SW1 SW2  SD LED    |
+------------------------------------------+
```
