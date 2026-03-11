# Reference Placement Analysis — Human-Routed KiCad Board

Source: KiCad screenshot of NL-S-3C board after human routing (saved as
`docs/reference_placement_kicad.png` — place screenshot there).

This document captures the placement patterns from the reference board to drive
automated placement optimization. The reference board represents EE-quality
placement that the pipeline should match.

---

## Board Layout (152 x 80mm, 4-layer)

```
+------------------------------------------------------------------+
|  J6  J3        |  Power (U1,C1,C2,D5,R1,R2)  |  J1 (28mm screw) |
|  Analog inputs |  Buck+LDO chain              |  Power input      |
|  (top-left)    |  (center-top)                |  (top-right edge) |
|                |                               |                   |
|  U4 U5 U7      |  U2 regulator section         |  K1  K2  K3  K4  |
|  ADC+filtering |  C4,C5,C17,L2                 |  Relay bank       |
|  R14-R31       |                               |  (right side)     |
|  C11-C22       |                               |  Q1-Q4, D6-D9    |
|                |  L3,L5,C27,C24                |  R10-R13,R33-R36 |
|  J4,J5         |                               |  D18-D21          |
|  (left edge)   |                               |                   |
|----------------+-------------------------------+-------------------|
|                |  U6 (Ethernet PHY)            |  U3 (MCU/ESP32)  |
|                |  W5500 — center               |  (right-center)  |
|                |  C7,C31,C32,D3,D16            |  C8-C10,R4-R7   |
|                |                               |  SW1,SW2,LED1    |
|                |  U8 (magnetics)               |  J15 (USB)       |
|                |  Y1 (crystal)                 |  J10 (debug)      |
|  J16 (SD)      |  U9 (WiFi) — board edge       |                   |
|  (bottom-left) |  J2 (aux) — bottom edge       |  SW2 (bottom-rt) |
|                |  J13 (RJ45) — bottom edge      |                   |
+------------------------------------------------------------------+
```

---

## Key Placement Patterns to Match

### 1. Group Compactness (CRITICAL)

Each functional group forms a TIGHT cluster, NOT a spread-out blob:

- **Relay bank**: K1-K4 in a horizontal row across the right side, ~60mm span.
  Each relay's driver (Q), flyback diode (D), and gate resistor (R) are
  directly adjacent to their relay. Subgroups are VERY tight (5-8mm spread
  per relay+driver unit).

- **Power section**: U1 (buck) with C1,C2,R1,R2 in a tight 20mm cluster at
  top-center. U2 (LDO) with C4,C5,C17,L2 in another tight cluster below.
  The two power subcircuits are adjacent but distinct.

- **Analog section**: U4,U5,U7 with all their filtering passives (R14-R31,
  C11-C22) packed tightly in the top-left quadrant. Connectors J4,J5,J6 at
  the LEFT board edge, close to the analog ICs.

- **MCU section**: U3 (ESP32) with decoupling (C8-C10), reset circuit (R4,SW1),
  boot circuit (R5,SW2), and indicators (LED1) all within 15mm.

- **Ethernet section**: U6 (W5500) centered, with U8 (magnetics) and Y1
  (crystal) adjacent. J13 (RJ45) at bottom edge directly below. All
  Ethernet components within 25mm of U6.

### 2. Isolation Zones

- **Analog zone** (top-left) is physically separated from **relay zone**
  (right side) by ~50mm. No relay components intrude into analog area.

- **Power zone** (top-center) acts as a buffer between analog and relay zones.

- **Digital zone** (bottom half) is below the power/relay zones, with the MCU
  and Ethernet subsystems having their own areas.

- **WiFi module** (U9) is at the bottom edge of the board with antenna
  clearance.

### 3. Connector Placement

ALL connectors are at board edges:
- J1 (power screw terminal): **top-right edge**
- J3 (power aux): **top edge**, near J1
- J4, J5 (analog inputs): **left edge**
- J6 (analog input): **top-left edge**
- J13 (RJ45): **bottom edge**
- J2 (aux): **bottom edge**
- J15 (USB): **right side**, near MCU
- J16 (SD card): **bottom-left edge**
- J10 (debug): **right side**, near MCU

No connector is more than 5mm from a board edge.

### 4. Signal Flow

Left-to-right and top-to-bottom flow:
- Power enters top-right (J1) → regulators (center-top) → distribution
- Analog signals enter left (J4-J6) → ADC ICs (U4,U5) → MCU (U3, right)
- Relay control: MCU (U3) → transistors (Q1-Q4) → relays (K1-K4)
- Ethernet: MCU (U3) → W5500 (U6) → magnetics (U8) → RJ45 (J13, bottom)

### 5. Subgroup Spacing

Within a group, subgroups have MINIMAL gaps:
- Relay driver subgroup (K+Q+D+R): all components within **8mm** of relay
- Buck converter (U1+C1+C2+L1): all within **10mm**
- Decoupling caps: within **3-5mm** of IC power pin

Between groups, there is **10-20mm** of clear space.

---

## Metrics to Target

| Metric | Reference Value | Current Pipeline |
|--------|----------------|-----------------|
| Group max spread (small, ≤10 refs) | 15-25mm | 25-40mm |
| Group max spread (medium, ≤25 refs) | 25-40mm | 60-80mm |
| Group max spread (large, >25 refs) | 40-60mm | 80-133mm |
| Subgroup spread (relay driver) | 5-8mm | 15-25mm |
| Connector edge distance | <5mm | 5-20mm |
| Inter-group gap | 10-20mm | Often overlapping |
| Decoupling cap distance | 3-5mm | 5-10mm |
| Relay-to-analog separation | ≥50mm | Variable |

---

## Action Items for Pipeline

1. **Halve group internal layout spacing** — `_ANCHOR_GAP_MM` (currently 8mm)
   should be 4-5mm. Passive placement gaps should be tighter.

2. **Subgroup-aware placement** — Each relay driver subcircuit should be placed
   as a micro-unit (K+Q+D+R+LED within 8mm), then the 4 micro-units arranged
   in a row. Currently the group layout doesn't know about subgroups.

3. **Strict connector edge pinning** — Every J ref must be within 5mm of a
   board edge. This is a hard constraint, not a soft score.

4. **Inter-group clearance** — After placing groups, ensure 10-15mm clear
   space between group bounding boxes. Currently group boundaries overlap.

5. **Reduce `max_row_width`** — The 80mm threshold still creates wide groups.
   Consider 50-60mm with better passive wrapping logic.

6. **Power section topology** — Buck → LDO → distribution should follow a
   spatial chain, not a single long row.
