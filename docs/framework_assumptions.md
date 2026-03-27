# Framework Assumptions Registry

> **Purpose**: Single source of truth for every hardcoded default, threshold, mapping,
> and implicit assumption in the kicad-ai-pipeline framework. Each entry is trackable,
> auditable, and linked to source code. Review this before any board generation run
> to catch spec drift.
>
> **Last full audit**: 2026-03-26
> **Audited by**: Claude + human review of 3D verification failures across 5 training boards

## How to Use This Document

- **Before adding a new default**: check if one already exists here
- **Before changing a value**: update the entry here FIRST, then change code
- **After a bug surfaces**: check if a wrong assumption caused it — update status
- **Periodic audit**: run `grep -n` for each FA-ID's value in the source to verify sync

## Status Key

| Status | Meaning |
|--------|---------|
| `OK` | Verified correct, matches spec |
| `BUG` | Known incorrect — tracked in docs/known_issues.md |
| `STALE` | May be outdated — needs verification |
| `REVIEW` | Needs human decision — spec unclear |
| `CONFLICT` | Two locations disagree — needs resolution |

---

## 1. Package Size Defaults

### FA-001: Default passive package fallback
- **Value**: `0805` when footprint_id is unrecognized
- **Location**: `src/kicad_pipeline/pcb/footprints.py:3058,3067,3368-3372`
- **Rationale**: Conservative fallback — large enough to hand-solder
- **Spec**: "0402 where practical, otherwise best fit"
- **Status**: `REVIEW` — fallback is reasonable but should log a WARNING when triggered
- **Note**: The fallback itself is fine. The bug is in training scripts hardcoding 0805
  for components that should be 0402/0603. See FA-050.

### FA-002: SMD passive dimension table
- **Value**: `_SMD_RC_DIMS` dict with 0402, 0603, 0805, 1206, 1210
- **Location**: `footprints.py:489-497`
- **Status**: `OK` — dimensions are IPC-compliant
- **Pad sizes**: 0402=(0.5,0.5), 0603=(0.8,0.8), 0805=(1.2,1.4), 1206=(1.0,1.8), 1210=(1.0,2.25)

### FA-003: LED packages use same geometry as R/C
- **Value**: LED_0402, LED_0603, LED_0805 share `_SMD_RC_DIMS`
- **Location**: `footprints.py:100-102, _fp_led()`
- **Status**: `OK` — standard practice for chip LEDs

---

## 2. 3D Model Mapping

### FA-010: Static 3D model pattern map
- **Value**: `_3D_MODEL_MAP` tuple of (pattern, dir, model_file)
- **Location**: `footprints.py:86-120`
- **Status**: `OK` — map has correct entries for all sizes (R_0402→R_0402, etc.)
- **Note**: Map is fine. Bug is in what patterns reach the lookup. See FA-011.

### FA-011: 3D model lookup uses requirements footprint_id, not actual footprint
- **Value**: `_model_for_package(footprint_id, layer)` at line 2995
- **Location**: `footprints.py:2994-2997`
- **Status**: `FIXED 2026-03-26` — swapped lookup order so `fp.lib_id` (actual footprint)
  is tried first, `footprint_id` (requirements) is fallback only.
- **Affected**: Any board where JLCPCB footprint differs from requirements footprint_id
- **Fix applied**: `footprints.py:2995` now calls `_model_for_package(fp.lib_id)` first

### FA-012: Inductor 3D model coverage
- **Value**: L_0805, L_1206, L_1210 in map
- **Location**: `footprints.py:113-115`
- **Status**: `OK` — map entries exist. Bug was in training scripts using wrong footprint_id.

### FA-013: Diode 3D model coverage
- **Value**: SOD-323, SOD-123 in map; NO SOT-23-3 TVS entry
- **Location**: `footprints.py:110-111`
- **Status**: `REVIEW` — SOT-23-3 TVS diodes are common. The `SOT-23` entry at line 107
  would match if the footprint_id contained "SOT-23", but training scripts specify
  "SOD-323" as the footprint for TVS diodes (wrong package entirely).

### FA-014: Crystal vs oscillator 3D model
- **Value**: Only `Crystal_SMD_3215` (2-pin) in map
- **Location**: `footprints.py:117`
- **Status**: `FIXED 2026-03-26` — added `OSC-SMD_4P` → `Oscillator_SMD_EuroQuartz_XO32-4Pin`
  mapping. Verified on ethernet board (Y1 now uses correct 4-pin oscillator model).

### FA-015: KiCad 3D model path variable
- **Value**: `${KICAD10_3DMODEL_DIR}`
- **Location**: `constants.py:KICAD_3DMODEL_VAR`
- **Status**: `OK` — standard KiCad 10 env var

---

## 3. Component Type Classification

### FA-020: Reference prefix → component type
- **Value**: R=resistor, C=cap, L=inductor, D=diode, Q=transistor, U=IC, J=connector,
  K=relay, SW=switch, Y=crystal, H=mounting hole, LED=LED
- **Location**: Used throughout `functional_grouper.py`, `placement.py`, `bom.py`
- **Status**: `OK` — standard IEC 60617 / IEEE conventions

### FA-021: Flyback diode detection heuristic
- **Value**: Prefix "D" + NOT TVS/LED in value/description
- **Location**: `functional_grouper.py:224-230`
- **Status**: `REVIEW` — TVS detection depends on "TVS" string in value field.
  Components without explicit "TVS" in value may be misclassified.

### FA-022: MCU identification keywords
- **Value**: STM32, ATMEGA, ATTINY, PIC, MSP430, RP2040, RP2350, ESP32
- **Location**: `functional_grouper.py:1125-1136`
- **Status**: `OK` — covers common MCU families. Falls back to any "U" prefix.

### FA-023: WiFi/RF module keywords for edge placement
- **Value**: esp32, wroom, wrover, wifi, ble, nrf52, cc2640, sx1276/1262, rfm95/96
- **Location**: `placement.py:473-487`
- **Status**: `OK` — comprehensive list

---

## 4. Board Generation Defaults

### FA-030: Default board dimensions
- **Value**: 80.0 x 40.0 mm
- **Location**: `builder.py:127,130`
- **Status**: `OK` — Hammond 1551K reference. Overridden by requirements in practice.

### FA-031: Board edge margin (components)
- **Value**: 2.0 mm
- **Location**: `constants.py:199`
- **Status**: `OK` — standard JLCPCB recommendation

### FA-032: Courtyard-to-courtyard clearance
- **Value**: 0.5 mm
- **Location**: `constants.py:202`
- **Status**: `OK` — standard for SMD assembly

### FA-033: Connector edge margin
- **Value**: 3.0 mm from centroid
- **Location**: `constants.py:214`
- **Status**: `REVIEW` — review_agent uses 8.0mm threshold (FA-044). These should align.

---

## 5. JLCPCB Manufacturing Constraints

### FA-040: Minimum trace width
- **Value**: 0.127 mm (5 mil) absolute, 0.2 mm recommended
- **Location**: `constants.py:42-51`
- **Status**: `OK` — matches JLCPCB spec sheet

### FA-041: Minimum via drill / annular ring
- **Value**: 0.3 mm drill, 0.13 mm ring (constants.py) vs 0.2 mm drill, 0.1 mm ring (manufacturing.py)
- **Location**: `constants.py:54-55` vs `manufacturing.py:23-24`
- **Status**: `CONFLICT` — constants.py has standard-process values, manufacturing.py has
  advanced-process values. Need to decide which process to target and make consistent.

### FA-042: Minimum silk width
- **Value**: 0.153 mm
- **Location**: `constants.py:69`
- **Status**: `OK` — JLCPCB minimum

### FA-043: Minimum stock filter
- **Value**: 1000 units
- **Location**: `constants.py:94-100`
- **Status**: `REVIEW` — may reject legitimate low-stock parts. Consider 100 for extended.

---

## 6. Placement Thresholds

### FA-044: Connector edge max distance
- **Value**: 8.0 mm (review_agent threshold)
- **Location**: `constants.py:243`
- **Status**: `OK` — measured from centroid. Edge-mount connectors exempt (KI-021).

### FA-045: Decoupling cap max distance
- **Value**: 5.0 mm edge-to-edge (not center-to-center)
- **Location**: `constants.py:223`
- **Status**: `OK` — industry standard 3-5mm

### FA-046: RF edge max distance
- **Value**: 6.0 mm
- **Location**: `constants.py:257`
- **Status**: `OK` — antenna must be near board edge

### FA-047: MCU peripheral max distance
- **Value**: 20.0 mm
- **Location**: `constants.py:254`
- **Status**: `OK` — generous for switches, LEDs, debug headers

### FA-048: Relay row max Y spread
- **Value**: 5.0 mm
- **Location**: `constants.py:265`
- **Status**: `OK` — keeps relays in a visually aligned row

### FA-049: Voltage domain min gap
- **Value**: 2.0 mm
- **Location**: `constants.py:240`
- **Status**: `OK` — minimum for isolation

---

## 7. Scoring Weights

### FA-060: Placement quality scoring weights
- **Values**: Collision=13.5%, Voltage isolation=13.5%, Constraint compliance=10%,
  Connector edge=9%, Decoupling=9%, MCU peripheral=9%, all others=4.5%
- **Location**: `scoring.py:34-47`
- **Status**: `OK` — tuned through iterative testing. Collision and voltage isolation
  are intentionally weighted highest.

### FA-061: Grade thresholds
- **Value**: A≥0.90, B≥0.75, C≥0.60, D≥0.40, F<0.40
- **Location**: `scoring.py:56-59`
- **Status**: `OK`

### FA-062: Collision penalty per overlap
- **Value**: 0.05 (5% score reduction per collision)
- **Location**: `scoring.py:71`
- **Status**: `OK`

---

## 8. Trace Width / Net Class Defaults

### FA-070: Signal trace width
- **Value**: 0.25 mm
- **Location**: `constants.py:106` and `netclasses.py:~50`
- **Status**: `OK` — consistent across both files

### FA-071: Power trace width
- **Value**: 0.5 mm (constants.py) vs 0.3 mm (netclasses.py)
- **Location**: `constants.py:109` vs `netclasses.py:~60`
- **Status**: `CONFLICT` — needs resolution. 0.5mm is more appropriate for power.

### FA-072: USB differential trace width
- **Value**: 0.3 mm
- **Location**: `constants.py:112`
- **Status**: `REVIEW` — should be impedance-controlled (90ohm diff). Width depends on
  stackup. 0.3mm is a reasonable starting point for standard 1.6mm 2-layer.

### FA-073: Signal via (small)
- **Value**: 0.3 mm drill, 0.6 mm diameter
- **Location**: `constants.py:140-143`
- **Status**: `OK` — JLCPCB minimum standard via

---

## 9. Coordinate System

### FA-080: Origin convention
- **Value**: Top-left (0,0), X right, Y down
- **Location**: `constants.py:6-10`, used everywhere
- **Status**: `OK` — matches KiCad convention

### FA-081: KiCad origin vs centroid for footprints
- **Value**: KiCad stores origin at pin 1, optimizer uses pad centroid
- **Location**: `pcb/pin_map.py` (compute_centroid_offset, origin_to_centroid, centroid_to_origin)
- **Status**: `OK` — consolidated into single source of truth after KI-004 bug fix

---

## 10. KiCad File Format

### FA-090: KiCad schematic version
- **Value**: 20260101
- **Location**: `constants.py:16`
- **Status**: `OK` — KiCad 10

### FA-091: KiCad PCB version
- **Value**: 20260206
- **Location**: `constants.py:19`
- **Status**: `OK` — KiCad 10

### FA-092: Hierarchical schematic path rules
- **Value**: Root `sheet_instances` path = `"/"`, sub-sheet = `"/{root}/{sheet}"`
- **Location**: Documented in CLAUDE.md, implemented in schematic builder
- **Status**: `OK` — verified against KiCad 9/10 behavior

---

## 50. Training Script Assumptions (DEPLOYMENT — not framework)

### FA-050: Training scripts hardcode 0805 for all passives
- **Value**: Was `_R0805_FP` for all R/C/LED; now uses R_0402, R_0603, C_0402, LED_0603
- **Location**: All 5 `scripts/train_*.py` files
- **Spec**: "0402 where practical, otherwise best fit"
- **Status**: `FIXED 2026-03-26` — R≤100K→0402, R=330R/49.9R→0603, C≤100nF→0402,
  C≥10uF→0805, LED→0603. Verified via 3D model spot-check on all 5 boards.

### FA-051: Training scripts hardcode SOD-323 for all diodes
- **Value**: `_SOD323_FP = "SOD-323"` used for flyback AND TVS AND Schottky
- **Location**: `scripts/train_relay_group.py:66`
- **Status**: `BUG` — SOD-323 is wrong for:
  - TVS diodes (PESD3V3 = SOT-23-3 package)
  - Schottky diodes (SS14 = SOD-123F package)
  - Flyback diodes (1N4148 = SOD-323 is actually correct)

### FA-052: Training scripts use R_0805 for ferrite beads
- **Value**: Was `_FERRITE_FP = "R_0805"`; now `_FERRITE_FP = "L_0805"`
- **Location**: `scripts/train_relay_group.py:73`
- **Status**: `FIXED 2026-03-26` — ferrites now get L_0805_2012Metric.step 3D model

### FA-053: Transistor footprint naming
- **Value**: Some training scripts specify "SOD-323" for transistor Q components
- **Location**: `scripts/train_relay_group.py:134`
- **Status**: `BUG` — transistors should be "SOT-23". The footprint predicate at
  `footprints.py:3121` matches "SOD-323" and builds a SOD-323 diode footprint,
  but the JLCPCB cache overrides with the real SOT-23 footprint. Naming is misleading.

---

## Changelog

| Date | FA-ID | Change | By |
|------|-------|--------|----|
| 2026-03-26 | ALL | Initial audit from 5-board 3D verification | Claude |
| 2026-03-26 | FA-011 | Identified as root cause of wrong 3D models on JLCPCB footprints | Claude |
| 2026-03-26 | FA-050-053 | Identified training script hardcoding as source of wrong packages | Claude |
| 2026-03-26 | FA-041 | Flagged via drill conflict between constants.py and manufacturing.py | Claude |
| 2026-03-26 | FA-071 | Flagged power trace width conflict between constants.py and netclasses.py | Claude |
