# Relay Driver Channel — Layout Design Rules

## Relay Pinout Reference (Songle SRD-05VDC-SL-C)

Standard SPDT relay pinout:
```
Pin 1: Coil+  -> +5V_RELAY
Pin 2: NC     -> Normally Closed contact
Pin 3: NO     -> Normally Open contact
Pin 4: Coil-  -> Q collector / flyback diode anode (RELAY_COIL net)
Pin 5: COM    -> Common contact
```

Mains-side pins: 2 (NC), 3 (NO), 5 (COM)
Coil-side pins: 1 (Coil+), 4 (Coil-)

## SOT-23 BJT Pinout (SS8050)

Codebase convention for SOT-23 NPN:
```
Pin 1: Base      -> RELAY_DRIVE (via R_gate)
Pin 2: Collector -> RELAY_COIL (K pin 4, D_flyback anode, R_LED pad 1)
Pin 3: Emitter   -> GND
```

## Relative Positioning Rules (from human feedback)

### Channel Column Structure
Each relay channel is a vertical column. All components share the relay's X center (+/-5mm).

### Vertical Order (top to bottom, relative to board edge)
1. J (screw terminal) -- closest to board edge, rotated 180 deg (pads toward edge)
2. K (relay) -- below terminal, 90 deg rotation
3. Q (transistor) + D_flyback (diode) -- beside and below K, near coil pin side
   - Q and D_flyback at same Y level
   - Q on one side of K center, D on the other side
4. R_gate -- directly below Q (dx~0, dy~+3mm)
5. D_LED + R_LED -- below R_gate, offset from K center
   - R_LED above D_LED (dy~-2mm)

### Terminal Row Rules
- All terminals at the SAME Y position
- All terminals rotated 180 deg (pads face board edge)
- Terminal X aligns with its relay X (+/-2mm)

### Relay Row Rules
- All relays at the SAME Y position
- All relays rotated 90 deg
- Evenly spaced (equal center-to-center)

### Mirroring Rule
- Q is on the coil-pin side of K
- D_flyback is on the opposite side from Q (mirrors across K center line)
- This creates a symmetric driver arrangement flanking the relay

### Power Isolation
- L (ferrite) + C (bulk caps) form a separate zone
- NOT mixed with relay channel area
- Location determined by board-level context (left edge, bottom, or power entry point)

### Repeating Pattern Rule
- All 4 channels MUST be visually identical vertical columns
- Spacing between channels is uniform
- Components within a channel maintain the same relative offsets

## Net Connectivity

### Per-Channel Nets
| Net | Components Connected |
|-----|---------------------|
| RELAY_COIL{N} | Q{N} pin 2 (collector), K{N} pin 4 (coil-), D{N} pin 1 (anode), R{N+4} pin 1 |
| RELAY_DRIVE{N} | R{N} pin 2, Q{N} pin 1 (base) |
| GPIO{N} | R{N} pin 1 |
| LED{N}_A | R{N+4} pin 2, D{N+4} pin 1 (anode) |
| RELAY_COM{N} | K{N} pin 5, J{N} pin 1 |
| RELAY_NO{N} | K{N} pin 3, J{N} pin 2 |
| RELAY_NC{N} | K{N} pin 2, J{N} pin 3 |

### Power Nets
| Net | Components Connected |
|-----|---------------------|
| +5V_LOGIC | L1 pin 1 |
| +5V_RELAY | L1 pin 2, C1 pin 1, C2 pin 1, K1-K4 pin 1, D1-D4 pin 2 (cathode) |
| GND | C1 pin 2, C2 pin 2, Q1-Q4 pin 3 (emitter), D5-D8 pin 2 (cathode) |

## LED Indicator Placement

D5-D8 are relay state indicator LEDs. R5-R8 are their current-limiting resistors.
Each LED pair belongs to a specific channel:
- D5 + R5 -> Channel 1 (R5 pad 1 on RELAY_COIL1 net)
- D6 + R6 -> Channel 2 (R6 pad 1 on RELAY_COIL2 net)
- D7 + R7 -> Channel 3 (R7 pad 1 on RELAY_COIL3 net)
- D8 + R8 -> Channel 4 (R8 pad 1 on RELAY_COIL4 net)

Each LED pair MUST be placed within the same channel column as its relay (+/-3mm X).
The LED pair is typically placed below the driver transistor and base resistor.

## Pad Facing Rules
| Component | Pad 1 faces | Pad 2 faces |
|-----------|-------------|-------------|
| R_gate | toward MCU (down) | toward Q base (up) |
| D_flyback | toward Q collector | toward +5V/K coil |
| R_LED | toward Q collector (up) | toward D_LED (down) |

## Orientation
| Component | Rotation | Reason |
|-----------|----------|--------|
| K (relay) | 90 deg | Coil pins face down toward driver, contacts face up toward terminal |
| J (terminal) | 180 deg | Pads flush with top board edge (facing outward) |
| Q (SOT-23) | 0 deg | Base faces down (toward MCU), collector faces up (toward relay) |
| D_flyback (SOD-323) | 0 deg | Across coil pins horizontally |
| R_gate (0805) | 90 deg | In signal flow (vertical) |
| D_LED (0805) | 0 deg | Horizontal, beside R_LED |
| R_LED (0805) | 0 deg | Horizontal, beside D_LED |

## Creepage Isolation Slot (DFM Requirement)

The relay footprint SHOULD include an Edge.Cuts semicircle slot between the
mains-side pins (COM=5, NO=3, NC=2) and the coil-side pins (1, 4). This provides
creepage isolation between mains voltage and logic-level domains.

Requirements:
- Minimum slot width: 1.5mm (IPC-2221 for 250V working voltage)
- Slot shape: semicircle or rectangular cutout in the PCB
- Location: between relay contact pins and coil pins within the relay footprint
- This is implemented at the footprint level, not by the placement engine

For boards that carry mains voltage through the relay contacts, this is a
safety-critical feature. For low-voltage applications (<50V), the slot is
recommended but not strictly required.

## Power Isolation

Relay coil switching generates high-frequency noise that must not propagate to
the logic supply. The relay power domain (+5V_RELAY) is isolated from the logic
domain (+5V_LOGIC) with a ferrite bead and bulk capacitors.

### Components
| Ref | Value | Package | Purpose |
|-----|-------|---------|---------|
| L1 | 600R@100MHz ferrite | 0805 | Series isolation on +5V rail |
| C1 | 100uF | 1210 SMD | Bulk energy storage on relay side |
| C2 | 10uF ceramic | 0805 | High-frequency decoupling on relay side |

### Topology
```
+5V_LOGIC --[L1 ferrite]--+-- +5V_RELAY --> relay coils (K1-K4 pin 1)
                           |                  flyback cathodes (D1-D4)
                           +-- C1 (100uF) --> GND
                           +-- C2 (10uF)  --> GND
```

### Placement Rules
- L1 is placed at the boundary between the logic and relay power zones
- C1 and C2 are placed within 8mm of L1 on the relay side
- C2 (ceramic) should be closer to L1 than C1 for better HF filtering
- GND is typically shared (single-point star ground recommended for high-noise boards)

## Compliance Check Rules (enforced by _check_design_rules)

### Row Alignment (relative)
| Rule | Tolerance | Description |
|------|-----------|-------------|
| All J at same Y | +/-1mm | Terminal row must be horizontally aligned |
| All K at same Y | +/-1mm | Relay row must be horizontally aligned |
| J-K X alignment | +/-2mm per channel | Each terminal directly above its relay |
| Equal relay spacing | max 2mm deviation from average | Uniform channel spacing |
| Q-D_flyback same Y | +/-2mm per channel | Driver pair at same vertical level |
| R_gate below Q | dx < 2mm | Gate resistor directly below transistor |

### Power Isolation
| Rule | Tolerance | Description |
|------|-----------|-------------|
| L1-C1 distance | <= 8mm | Bulk cap near ferrite |
| L1-C2 distance | <= 8mm | Ceramic cap near ferrite |

## Quality Criteria (for review)
- [ ] All 4 channels visually identical vertical columns
- [ ] No component overlaps
- [ ] All terminals at same Y (+/-1mm)
- [ ] All relays at same Y (+/-1mm)
- [ ] J{N} directly above K{N} (+/-2mm X alignment)
- [ ] Equal spacing between relay columns
- [ ] Q and D_flyback at same Y within each channel
- [ ] R_gate directly below Q (dx < 2mm)
- [ ] Terminals rotated 180 deg (pads face board edge)
- [ ] Signal flow is top-to-bottom
- [ ] Connected pads face each other
- [ ] LED pairs within their channel column (+/-3mm from relay X)
- [ ] Power isolation components (L1, C1, C2) grouped separately from channels
- [ ] Correct relay pinout used (pin 4 = coil-, not pin 2)
- [ ] Correct SOT-23 pinout (pin 2 = collector, not pin 3)
