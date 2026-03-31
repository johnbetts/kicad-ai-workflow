# Bug Report: Visual Review Gaps (2026-03-28)

Discovered during dual-persona visual review of all 5 training boards.
These are CORE issues — gaps in `review_placement()` and the scoring system.

## BUG-REVIEW-001: No compactness/density metric
**Severity:** MAJOR
**Boards:** MCU Core, Power, Analog
**Description:** Boards waste 50-75% of PCB area. Components float in empty space
with huge gaps. No rule penalizes low density or wasted board area.
**Impact:** Unnecessarily large boards = higher JLCPCB cost, longer traces, worse EMI.
**Fix location:** `src/kicad_pipeline/optimization/scoring.py` — add compactness dimension.

## BUG-REVIEW-002: No buck converter topology check
**Severity:** CRITICAL
**Boards:** Power
**Description:** The IC-inductor-diode hot loop triangle isn't validated. Only individual
cap-to-IC distance is checked. D1 is 10+mm from U1's switch node — the hot loop is
enormous and would radiate badly.
**Fix location:** `src/kicad_pipeline/optimization/review_agent.py` — new PlacementRule
for POWER_LOOP_AREA that validates the IC/inductor/catch-diode triangle.

## BUG-REVIEW-003: Connector-edge should be CRITICAL for THT
**Severity:** MAJOR
**Boards:** MCU Core, Power, Analog
**Description:** Through-hole pin headers (J2, J3) placed mid-board get only a minor
violation. THT headers mid-board are physically unusable for cable access — should be
CRITICAL, not minor.
**Fix location:** `src/kicad_pipeline/optimization/review_agent.py` — `_check_connector_edge`
should escalate severity to CRITICAL for THT connectors.

## BUG-REVIEW-004: Missing 3D models not reflected in placement grade
**Severity:** MAJOR
**Boards:** MCU Core, Power
**Description:** `pcb_integrity.py` catches missing 3D models but `review_placement()`
doesn't factor integrity results into its grade. A board with invisible passives
still gets Grade A.
**Fix location:** `src/kicad_pipeline/optimization/review_agent.py` — integrate
integrity issue count into grade computation.

## BUG-REVIEW-005: Decoupling uses center-to-center only
**Severity:** MAJOR
**Boards:** Ethernet, Analog
**Description:** Cap column stacking makes the Nth cap exceed the 5mm threshold purely
from geometry. The check doesn't account for pin-side proximity or pad-facing direction.
2+ caps should use a 2-column layout.
**Fix location:** `src/kicad_pipeline/optimization/review_agent.py` — `_check_decoupling_distance`
should use pin-level distance, not centroid.

## BUG-REVIEW-006: No flyback diode loop area check
**Severity:** MAJOR
**Boards:** Relay
**Description:** Flyback diodes D1-D4 are 5-8mm from relay coil terminals. No rule
validates that the diode is across the coil pins with minimal loop area. Critical for EMI.
**Fix location:** `src/kicad_pipeline/optimization/review_agent.py` — new PlacementRule
for FLYBACK_DIODE_PROXIMITY.

## BUG-REVIEW-007: Pull-up resistor proximity not checked
**Severity:** MINOR
**Boards:** Analog
**Description:** I2C/SPI pull-up resistors (R9/R10) 15+mm from bus master IC. Not
classified as MCU peripherals so proximity isn't checked.
**Fix location:** `src/kicad_pipeline/optimization/review_agent.py` — expand
MCU peripheral detection to include pull-up resistors on bus nets.

## BUG-REVIEW-009: Exposed thermal pad missing on TPS54331
**Severity:** CRITICAL
**Boards:** Power
**Description:** TPS54331 (SOIC-8 with PowerPAD) requires 9 pads — 8 gull-wing leads
plus a large exposed thermal/GND pad underneath. The footprint matcher selects standard
SOIC-8 (8 pads only, no thermal pad). Pin 8 in requirements is "PAD" type=POWER_IN but
gets mapped to a regular gull-wing pad instead of the exposed pad.
Without the thermal pad: no heat dissipation (IC overheats), inadequate GND return current
path for the buck converter's switching currents.
**Root cause:** `footprints.py` matches on package name "SOIC-8" without checking whether
the component's pin list includes an exposed/thermal pad. The JLCPCB EasyEDA footprint
`SOIC-8_L5.0-W4.0-P1.27-LS6.0-BL` is plain SOIC-8 without PowerPAD.
**Fix location:** `src/kicad_pipeline/pcb/footprints.py` — when a component has a pin
named "PAD"/"EP"/"EPAD" with type POWER_IN, select the thermal-pad variant of the package
(e.g. SOIC-8-EP instead of SOIC-8). Add parametric generator for SOIC-8 + exposed pad.

## BUG-REVIEW-008: Screw terminal orientation not validated
**Severity:** MAJOR
**Boards:** Analog (confirmed), possibly Relay
**Description:** Screw terminal wire entry faces board interior instead of board edge.
No rule checks connector orientation against nearest edge direction.
**Fix location:** `src/kicad_pipeline/optimization/review_agent.py` —
`_check_connector_orientation` or new rule for terminal block facing.
