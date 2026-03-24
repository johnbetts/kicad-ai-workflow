# Relay Training Board — Open Bugs

Bugs reported by human reviewer. Each must be verified fixed (not just claimed fixed)
before closing. The dual-persona review and audit process must check these FIRST
before asking the human to verify.

## Open Bugs

### BUG-R08: Power isolation group too close to relay area
- **Status**: OPEN
- **Issue**: L1/L2/C1/C2 placed in relay zone, L1-C1 distance exceeds 8mm
- **Expected**: Power isolation as separate zone, location TBD by board context
- **Deferred**: Location depends on full board layout (other groups determine where power entry is)

### BUG-R01: Screw terminals not aligned with their relays
- **Status**: OPEN (reported 3x)
- **Issue**: J1-J4 are not positioned directly above K1-K4. J ordering doesn't match relay ordering.
- **Expected**: J1 above K1, J2 above K2, etc. Same X (±2mm), all J at same Y, all rotated 180°
- **Root cause**: Optimizer places connectors by edge proximity, not by net connectivity to their relay
- **Verification**: Check J-K X alignment in compliance output. ALL channels must show "OK"

### BUG-R02: Relays K1 and K2 overlap (collision)
- **Status**: OPEN (reported 2x)
- **Issue**: K1 and K2 courtyard collision. Relay spacing too tight.
- **Expected**: All relays evenly spaced with no overlaps. Min gap = courtyard clearance (0.5mm)
- **Root cause**: Board too narrow for 4 relays, or relay X positions not accounting for full width
- **Verification**: Zero collisions in review output

### BUG-R03: LED subgroups (D5-D8, R5-R8) scattered, not near their relays
- **Status**: OPEN (reported 3x)
- **Issue**: LED indicators placed far from their relay channel. D5-D8 and R5-R8 at random locations.
- **Expected**: D5+R5 in CH1 column, D6+R6 in CH2 column, etc. Within ±3mm of K X position.
- **Root cause**: Optimizer doesn't know LED-to-relay channel association
- **Verification**: All "LED in column" checks pass in compliance output

### BUG-R04: C1 not a standard SMD capacitor footprint
- **Status**: OPEN (reported 2x)
- **Issue**: 100µF bulk cap has wrong/oversized footprint. Should be standard SMD (0805 or 1206).
- **Expected**: Standard 0805 or 1206 SMD capacitor footprint for 100µF MLCC
- **Root cause**: Script specifies non-standard package or footprint generator doesn't match
- **Verification**: Open in KiCad, verify C1 footprint matches standard SMD cap

### BUG-R05: No L2 ferrite bead on GND
- **Status**: OPEN (reported 1x)
- **Issue**: Only L1 on +5V_RELAY, no ferrite on GND_RELAY
- **Expected**: L2 ferrite bead between GND_LOGIC and GND_RELAY
- **Root cause**: Script only creates L1, not L2
- **Verification**: Check component list includes L2 with GND_LOGIC/GND_RELAY nets

### BUG-R06: Footprint labels (ref text) positioned away from footprint body
- **Status**: OPEN (reported 3x)
- **Issue**: Reference designator text (R1, K1, etc.) not centered on footprint in KiCad PCB
- **Expected**: Ref text centered on footprint body, readable orientation
- **Root cause**: Silkscreen text placement in `_footprint_sexp` or footprint builder uses offset instead of center
- **Verification**: Open in KiCad, verify ref text is centered on each footprint

### BUG-R07: Grey blob in relay footprint — should be Edge.Cuts isolation slot
- **Status**: OPEN (reported 2x)
- **Issue**: Relay footprint has a grey filled area in the middle. This should be an Edge.Cuts semicircle cutout for creepage isolation between COM (mains pins 2/3/5) and coil (pins 1/4).
- **Expected**: Edge.Cuts arc/slot between mains and low-voltage sides of relay
- **Root cause**: Relay footprint generator doesn't create isolation cutouts. The grey area may be a courtyard or fab layer artifact.
- **Verification**: Open in KiCad, verify Edge.Cuts slot exists between pin groups

## Closed Bugs

### BUG-R01: Screw terminals not aligned — CLOSED
- Fixed in optimizer: `_phase_relay_connector_alignment()` in `ee_phases.py`
- Maps J→K via net connectivity, aligns X positions

### BUG-R02: Relay overlap — CLOSED
- Fixed in optimizer: `_place_row_layout()` in `level3_phases.py`
- Uses actual relay width + courtyard gap for spacing, falls back to full board width if zone too narrow
- Board widened to 90mm to fit 4 relays physically

### BUG-R03: LED scattered — CLOSED
- Fixed in optimizer: `_phase_relay_leds()` in `ee_phases.py`
- Places D_LED and R_LED at K.x ±1.5mm, below drivers

### BUG-R04: C1 footprint — CLOSED
- Changed to C_0805 in training script

### BUG-R05: No L2 for GND — CLOSED
- Added L2 ferrite bead + GND_LOGIC net in training script

### BUG-R06: Labels off-center — CLOSED
- Fixed in `_fp_standard_properties()` in `builder.py` — ref/value text now at (0,0)

### BUG-R07: Grey blob in relay — CLOSED
- Fixed in `footprints.py` — removes thick F.Fab lines from JLCPCB cached footprint, adds Edge.Cuts isolation slot

## Process Rule
- Before ANY iteration is shown to the human, ALL open bugs must be checked
- A bug is only CLOSED when the human confirms it's fixed
- The dual-persona review must reference this file and verify each bug
