# PCB Placement — Hierarchical Group Requirements

## Core Principle

The schematic defines hierarchical **groups** and **subgroups** of components.
PCB placement must preserve this hierarchy: components within a subgroup stay
together, subgroups within a group stay together, and groups are arranged in
zones that respect power flow topology and isolation requirements.

---

## 1. Group Hierarchy

Placement operates on three levels:

| Level | Description | Visual Indicator |
|-------|-------------|-----------------|
| **Zone** | Voltage domain or functional region (24V, 5V, 3.3V, Analog, RF) | Color-coded background |
| **Group** | Major functional block (Power Supply, Relays, MCU, Analog, Ethernet) | Dotted box outline, same color for all members |
| **Subgroup** | Individual circuit within a group (one relay driver, one ADC channel) | Tight cluster, components touching |

### Zone → Group → Subgroup Example

```
24V Zone (orange)
├── Power Input Group
│   ├── 24V Buck Subgroup: U1, C1, C2, C3, L1, R1, R2
│   └── Input Protection: J1, D5
5V Zone (blue)
├── 3.3V Buck Group: U2, C5, C6, C7, C17, L2
├── Power OR'ing Group: D1, D2, D3, C4
├── Ferrite Distribution Group: L3, L4, L5, L6, C27, C35
└── Relay Zone Group
    ├── Relay 1 Subgroup: K1, Q1, R10, D6, R33, D18
    ├── Relay 2 Subgroup: K2, Q2, R11, D7, R34, D19
    ├── Relay 3 Subgroup: K3, Q3, R12, D8, R35, D20
    ├── Relay 4 Subgroup: K4, Q4, R13, D9, R36, D21
    └── Relay Bulk: C24
3.3V Zone (green)
├── MCU Group
│   ├── MCU Core: U3, C8, C9
│   ├── Reset/Boot: R4, C10, SW1, R5, SW2
│   ├── USB: J2, U9, R6, R7
│   ├── I2C: R8, R9
│   ├── 1-Wire: R26, D16
│   ├── SD Card: J16, C33
│   ├── Status LED: LED1, C34
│   └── GPIO Header: J15
Analog Zone (yellow)
├── ADC Group
│   ├── ADC1 Subgroup: U4, C11
│   │   ├── Ch0 Ladder: R14, R15, D10, C18
│   │   ├── Ch1 Ladder: R16, R17, D11, C19
│   │   ├── Ch2 Ladder: R18, R19, D12, C20
│   │   └── Ch3 Ladder: R20, R21, D13, C21
│   ├── ADC2 Subgroup: U5, C12
│   │   ├── Ch0 Ladder Switch: R22, R23, R24, C28, J4
│   │   ├── Ch1 Opto: U7, R25, R27, R32, SW3, J5, D17, LED2
│   │   ├── Ch2 Spare: R28, R29, D14, C29, J6
│   │   └── Ch3 Switched: R30, R31, D15, C30
│   └── ADC Connectors: J3 (sensor inputs)
Ethernet Zone (at board edge)
├── Ethernet Group
│   ├── W5500 Core: U6, C13, C14, C15, C16
│   ├── Crystal: Y1, C25, C26
│   ├── RJ45: J13
│   └── PoE: U8, C31, C32
Display (connector only)
└── TFT Header: J14
```

---

## 2. Placement Rules by Level

### 2.1 Zone-Level Rules

- Zones follow **power flow topology**: highest voltage leftmost (landscape)
  or topmost (portrait), lowest voltage rightmost/bottommost
- **Boundary strips** between adjacent zones for regulators/ferrites
- Zone width proportional to component count
- Minimum 8mm separation between zones of different voltage domains

### 2.2 Group-Level Rules

- All components in a group share the **same color** in visualization
- Groups bounded by **dotted box outline** in visualization
- Groups placed as a unit — the group centroid defines zone membership
- Inter-group spacing: 3-5mm between adjacent groups within the same zone
- Groups with mutual connections placed adjacent (e.g., MCU near I2C ADCs)

### 2.3 Subgroup-Level Rules

- Components within a subgroup placed as **tight clusters**
- Maximum spread from subgroup anchor: 10mm (passives), 15mm (with ICs)
- Specific proximity rules:
  - Decoupling caps: ≤3mm from IC power pin
  - Crystal + load caps: ≤5mm from IC clock pins
  - Flyback diode: ≤3mm from relay coil
  - Driver transistor: ≤5mm from relay coil
  - Feedback resistors: ≤5mm from regulator FB pin
  - Voltage divider: ≤8mm from ADC input pin

---

## 3. Connector Placement Rules

Connectors are placed by **functional association**, not nearest edge:

| Connector | Associated Group | Edge Rule |
|-----------|-----------------|-----------|
| J1 (power harness) | Power Input / Relay Outputs | Same edge as relay bank |
| J2 (USB-C) | MCU Group | Board edge, near MCU |
| J3 (sensor inputs) | ADC Group | Same edge as analog zone |
| J4 (ladder switch) | ADC2 Ch0 | Same edge as analog connectors |
| J5 (opto input) | ADC2 Ch1 | Same edge as analog connectors |
| J6 (spare ADC) | ADC2 Ch2 | Same edge as analog connectors |
| J13 (RJ45) | Ethernet Group | Board edge, with antenna clearance |
| J14 (TFT header) | Display Group | Board edge, near MCU SPI |
| J15 (GPIO header) | MCU Group | Board edge, near MCU |
| J16 (SD card) | MCU Group | Board edge |

**Rule**: All analog connectors (J3, J4, J5, J6) on the **same edge**.
All power/relay connectors (J1) on the **same edge as relay bank**.

---

## 4. Power Flow & Isolation

### 4.1 Power Flow Path

```
J1 (24V input)
  → D5 (TVS protection)
  → U1 (24V→5V buck): C1, L1, C3
      → Power OR (D1, D2, D3) → +5V
          ├── L3 ferrite → RELAY_5V (relay domain)
          ├── L6 ferrite → AVCC (analog domain)
          ├── U2 (5V→3.3V buck): C5, L2, C6, C7 → +3V3
          │   └── MCU, W5500, digital ICs
          └── Direct: LEDs, headers
```

### 4.2 Isolation Boundaries

Ferrites L3/L5 form the **relay isolation boundary** — place at zone border.
Ferrites L4/L6 form the **analog isolation boundary** — place at zone border.

Components on the isolated side of a ferrite must NOT be in the same
tight cluster as components on the other side.

### 4.3 Cross-Domain Measurement

ADC voltage dividers measure signals from other domains (24V relay outputs,
switched power). The dividers and their protection components belong to the
**analog group** even though they measure 24V signals. They should be placed
in the analog zone, near the ADC they feed.

---

## 5. Visualization Requirements

### 5.1 Color Coding

Each **major group** gets a unique color. All components in that group,
including passives, share the same color fill:

| Group | Color |
|-------|-------|
| Power Supply (bucks, OR, ferrites) | Orange |
| Relay Outputs (K1-K4 + drivers) | Red |
| MCU + Peripherals | Green |
| Analog (ADCs + dividers + connectors) | Yellow |
| Ethernet (W5500 + RJ45 + PoE) | Blue |
| Display (TFT header) | Purple |

### 5.2 Group Boundaries

- **Dotted box** around each major group boundary
- Optional: lighter dotted box around subgroups within a group
- Box should be tight-fit around the group's component bounding box + 2mm margin

### 5.3 Annotations

- Group name label inside or above each dotted box
- Score overlay (existing)
- Domain coloring as background (existing)

---

## 6. Iteration Strategy

### 6.1 Macro-First, Micro-Second

When running the placement optimization loop:

1. **First pass — Zone layout**: Place groups as units in their correct zones.
   Get the major spatial relationships right (power left, analog isolated,
   relays together, MCU central, Ethernet at edge).

2. **Second pass — Group arrangement**: Within each zone, arrange groups
   relative to each other. Relay bank in a row. ADCs near their connectors.
   Regulators at zone boundaries.

3. **Third pass — Subgroup tightening**: Within each group, arrange
   subgroups. Each relay driver cluster tight. Each ADC channel ladder
   tight. Crystal near MCU.

4. **Fourth pass — Component-level**: Fine-tune individual component
   positions. Decoupling cap proximity. Connector orientation.
   Collision resolution.

### 6.2 Review Loop

Each iteration of the visual inspection loop should:

1. Render the current placement with group colors and boundaries
2. Run the review agent for rule violations
3. Identify the **highest-level** issue first:
   - Wrong zone? → Move the whole group
   - Group in wrong position within zone? → Rearrange groups
   - Subgroup spread? → Tighten subgroup
   - Component too far? → Pull individual component
4. Fix the highest-level issue, re-render, repeat
5. Do NOT fix micro issues (cap distance) while macro issues (wrong zone) exist

### 6.3 Exit Conditions

- All groups in correct zones
- All subgroups within 15mm of group centroid
- All critical proximity rules met (crystal, decoupling, flyback)
- All connectors at board edges with correct functional association
- Zero courtyard collisions
- Score ≥ 0.90 AND grade A or B
- Human visual verification approved

---

## 7. NL-S-3C Specific Group Definitions

For the nl-s-3c-complete board, these are the concrete groups derived
from the schematic's FeatureBlock structure:

### Power Supply (FeatureBlock: "Power Supply")
- **24V Buck**: U1, C1, C2, C3, L1, R1, R2
- **3.3V Buck**: U2, C5, C6, C7, C17, L2
- **Power OR**: D1, D2, D3, C4
- **Power LED**: R3, D4
- **Ferrites**: L3, L4, L5, L6, C27, C35

### Relay Outputs (FeatureBlock: "Relay Outputs")
- **Relay 1**: K1, Q1, R10, D6, R33, D18
- **Relay 2**: K2, Q2, R11, D7, R34, D19
- **Relay 3**: K3, Q3, R12, D8, R35, D20
- **Relay 4**: K4, Q4, R13, D9, R36, D21
- **Relay Bulk**: C24

### MCU (FeatureBlock: "MCU + Peripherals")
- **ESP32 Core**: U3, C8, C9
- **Reset/Boot**: R4, C10, SW1, R5, SW2
- **USB**: J2, U9, R6, R7
- **I2C Bus**: R8, R9
- **1-Wire**: R26, D16
- **SD Card**: J16, C33
- **Status LED**: LED1, C34
- **GPIO Breakout**: J15

### Analog Inputs (FeatureBlock: "Analog Inputs")
- **ADC1**: U4, C11, R14, R15, D10, C18, R16, R17, D11, C19,
  R18, R19, D12, C20, R20, R21, D13, C21
- **ADC2**: U5, C12, R22, R23, R24, C28, U7, R25, R27, R32,
  SW3, D17, LED2, R28, R29, D14, C29, R30, R31, D15, C30
- **Analog Connectors**: J3, J4, J5, J6

### Ethernet (FeatureBlock: "Ethernet + PoE")
- **W5500**: U6, C13, C14, C15, C16
- **Crystal**: Y1, C25, C26
- **RJ45**: J13
- **PoE**: U8, C31, C32

### Display (FeatureBlock: "Display")
- **TFT Header**: J14
