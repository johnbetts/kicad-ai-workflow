# Bug Report: Placement v7 — Comprehensive Issues (2026-03-11)

Filed by: human review of KiCad board render

## Board-Level Issues

### BUG-V7-01: Board too narrow — components off right edge
- **Severity**: Critical
- **Description**: Board width insufficient. J14, J15, R13 and other components extend past right board edge.
- **Root cause**: Board dimensions not properly accounting for all component placements, or auto-sizer not expanding enough.
- **Fix**: Ensure 140x80mm board is wide enough, or components are placed within bounds.

### BUG-V7-02: Vias off-board
- **Severity**: Critical
- **Description**: Isolation zone vias extend past board boundaries.
- **Root cause**: Via stitching code doesn't check board bounds.

### BUG-V7-03: Random isolation zone with vias top-right
- **Severity**: High
- **Description**: Spurious isolation zone with vias in top-right corner, no functional purpose.
- **Root cause**: Zone generation creates zones for areas without components.

## Component Placement Issues

### BUG-V7-04: J14 and J15 overlap
- **Severity**: Critical
- **Description**: J14 and J15 connectors are placed overlapping each other on right edge.
- **Root cause**: Collision resolution not handling edge-pinned connectors.

### BUG-V7-05: Components inside U3 (ESP32) footprint
- **Severity**: Critical
- **Description**: Multiple small components placed inside ESP32-S3-WROOM courtyard.
- **Root cause**: Collision resolution not protecting U3 courtyard adequately.

### BUG-V7-06: J2 and J16 in wrong position
- **Severity**: High
- **Description**: J2 and J16 placed in wrong functional area.
- **Root cause**: Connector placement not following functional association.

### BUG-V7-07: Y1 & C2 misplaced in other footprints
- **Severity**: High
- **Description**: Crystal Y1 and cap C2 placed inside other component courtyards.
- **Root cause**: Crystal oscillator subcircuit placement not protecting positions.

### BUG-V7-08: C31/C32 should be closer to U8
- **Severity**: Medium
- **Description**: Decoupling caps C31/C32 too far from U8 (Ethernet PHY).
- **Root cause**: Decoupling placement not prioritizing U8.

### BUG-V7-09: SW1+SW2 should be next to each other and next to LED
- **Severity**: Medium
- **Description**: UI cluster (switches + LED) scattered instead of grouped.
- **Root cause**: No UI cluster subcircuit detection/placement.

### BUG-V7-10: U8 should be rotated 90deg left, placed top of J13
- **Severity**: Medium
- **Description**: Ethernet PHY U8 should be rotated 90deg CCW and placed directly above J13 (centered).
- **Root cause**: Ethernet signal-chain placement not optimizing U8 orientation.

## Group/Zone Issues

### BUG-V7-11: Analog and power groups too high
- **Severity**: High
- **Description**: Analog and power groups placed too high, interfering with screw terminal connectors.
- **Root cause**: Zone fractions place power in top half, overlapping with screw terminals.

### BUG-V7-12: L3,4,5,6 in power group — should be at domain boundaries
- **Severity**: High
- **Description**: Inductors L3-L6 placed inside power group. They should be boundary components between power→3V3, power→analog, power→5V, power→5V_relay.
- **Root cause**: Ferrite/inductor boundary placement not implemented for these specific rails.

### BUG-V7-13: J15 sitting in 5V_relay isolation zone
- **Severity**: High
- **Description**: J15 connector placed inside what should be the 5V relay isolation zone.
- **Root cause**: Connector zone assignment incorrect.

### BUG-V7-14: Missing isolation zones (24V, 5V_relay, analog, digital)
- **Severity**: High
- **Description**: Board needs proper isolation zones with nofill zones between them for:
  - 24V side
  - 5V_relay
  - Analog
  - Digital
- **Root cause**: Zone generation doesn't create voltage-domain-based copper pour zones with nofill gaps.

## Relay Group Issues

### BUG-V7-15: Relays not rotated 90deg left
- **Severity**: High
- **Description**: Relays K1-K4 still at wrong rotation. Need 90deg CCW rotation.
- **Root cause**: Phase 3 fix may not have taken effect, or rotation direction wrong.

### BUG-V7-16: Relay subgroups have wrong components
- **Severity**: Critical
- **Description**: Support components (Q, D, R) assigned to wrong relay subgroups.
- **Root cause**: Subcircuit detection matching components to wrong relay anchors.

### BUG-V7-17: Diodes in 5V relay need swapping (D18↔D8) and 180deg rotation
- **Severity**: Medium
- **Description**: D18 and D8 should be swapped for shorter traces, with 180deg rotation.
- **Root cause**: Diode placement doesn't optimize for trace length within relay subcircuit.

## Routing/Signal Issues

### BUG-V7-18: ADC ladder traces cross — need left/right sorting
- **Severity**: Medium
- **Description**: ADC voltage divider ladders for J6,5,4,3,1 create crossing ratsnest lines.
  Components connected to right-side MCU pins should use right ladder, left-side to left ladder.
- **Root cause**: ADC channel placement doesn't consider MCU pin side for ordering.

## Layer Stack Issues

### BUG-V7-19: 4-layer board, In2.Cu should be mixed signal layer
- **Severity**: Medium
- **Description**: Board should be 4 layers. In2.Cu designated as mixed signal layer.
- **Root cause**: Layer stack configuration not set for this board.

### BUG-V7-20: Isolation vias for U3 incomplete
- **Severity**: Medium
- **Description**: ESP32 isolation via fence has gaps — many vias missing.
- **Root cause**: Via stitching density/coverage insufficient around U3.
