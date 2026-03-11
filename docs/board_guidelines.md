# PCB Design Guidelines — Component Placement & Board Layout

Reference guide for PCB placement decisions, isolation strategies, and design
rules. Applies to any mixed-signal board — adapt thresholds to your specific
design.

---

## 1. Layer Stack-Up Strategy

### 4-Layer (Recommended for Mixed-Signal)

| Layer | Purpose | Notes |
|-------|---------|-------|
| L1 (Top) | Components, high-priority signals | Keep analog and digital separated |
| L2 (GND) | Solid ground plane | Never route signals — keep continuous |
| L3 (Power) | Power distribution, some signals | Ground pour under RF/crystal areas |
| L4 (Bottom) | Secondary components, low-speed signals | Minimize component count |

### 2-Layer (Budget-Constrained)

| Layer | Purpose | Notes |
|-------|---------|-------|
| Top | All components, signals | Group by function |
| Bottom | Ground pour, power traces | Prioritize complete ground plane |

**Rule of thumb**: Use 4 layers when any of these apply:
- Board has WiFi/BLE module + sensitive analog (ADC)
- More than 80 nets
- High-current switching (relays, motor drivers) + precision analog
- Controlled impedance required (USB 2.0+, Ethernet, LVDS)

---

## 2. Component Placement by Function

### 2.1 Microcontroller / SoC

| Guideline | Detail |
|-----------|--------|
| Position | Center of board or dedicated digital zone |
| Crystal | Within 10-20mm, short symmetric traces |
| Decoupling | 100nF within 3mm of each VCC pin; 10uF within 10mm |
| Orientation | Align for shortest traces to high-speed peripherals |

### 2.2 WiFi / BLE / RF Modules

| Guideline | Detail |
|-----------|--------|
| Position | Board edge, antenna facing outward |
| Keepout | 15-20mm clear zone around antenna — no copper, traces, or components |
| Ground plane | Continuous under module body, terminated at antenna edge |
| Orientation | Antenna overhang past board edge when possible |
| Isolation | No digital traces routed under antenna area |
| Matching | Follow module datasheet for antenna feed impedance |

### 2.3 Ethernet Interface

| Guideline | Detail |
|-----------|--------|
| RJ45 | Board edge for connector access |
| Magnetics | Between PHY and connector, short traces |
| PHY IC | Within 25mm of RJ45 connector |
| Differential pairs | 100-ohm impedance, length-matched, no vias if possible |
| Isolation | Magnetics provide 1500V galvanic isolation |

### 2.4 USB Interface

| Guideline | Detail |
|-----------|--------|
| Connector | Board edge |
| ESD diodes | Within 5mm of connector pins |
| D+/D- | 90-ohm differential, no vias, length-matched |
| VBUS | Wide trace (≥0.5mm) with bulk cap near connector |
| CC pins (USB-C) | 5.1k pull-downs within 10mm of connector |

### 2.5 Relays and Power Switching

| Guideline | Detail |
|-----------|--------|
| Position | Isolated zone, away from analog and RF (≥10-15mm) |
| Orientation | Coils parallel to board edges |
| Flyback diodes | Directly across coil pins (≤3mm) |
| Contact traces | ≥1mm width (adjust for current — use trace width calculator) |
| Isolation | Board slots between contacts and logic if voltage >50V |
| Grouping | Cluster relays with their driver transistors and diodes |
| Snubbers | RC across contacts for inductive loads |

### 2.6 ADC / Precision Analog

| Guideline | Detail |
|-----------|--------|
| Position | Dedicated analog zone, far from digital noise sources |
| Proximity | Near analog input connectors to minimize trace length |
| Decoupling | 100nF ceramic + 10uF on VDD, ferrite bead on power supply |
| Input filtering | RC low-pass on each analog input |
| Traces | Short, wide (15-20 mil), away from digital/relay traces |
| Ground | Analog ground pour connected to main GND at single point |
| Guard rings | Around high-impedance inputs if required |

### 2.7 Voltage Regulators (LDO / Buck / Boost)

| Guideline | Detail |
|-----------|--------|
| Position | Near power entry point |
| Input cap | Within 5mm of VIN pin |
| Output cap | Within 5mm of VOUT pin |
| Inductor (buck) | Short, wide traces to SW pin |
| Thermal pad | Thermal vias (0.3mm drill, 4-6 vias) to ground plane |
| Star topology | Power distribution from regulator output, not daisy-chain |
| Grouping | Regulator + input cap + output cap + inductor as a unit |

### 2.8 Connectors (General)

| Guideline | Detail |
|-----------|--------|
| Position | Board edges for external access |
| Screw terminals | LEFT or BOTTOM edge, away from sensitive circuits |
| Pin headers | Grouped by function (I2C, SPI, GPIO) |
| Mounting | Secure mechanical connection — strain relief where possible |
| ESD | TVS diodes near connector pins for external interfaces |

### 2.9 Passive Components (R, C, L)

| Guideline | Detail |
|-----------|--------|
| Decoupling caps | Within 3-5mm of IC power pins they serve |
| Pull-up/down resistors | Near the IC pin they're pulling |
| Series resistors | In-line with the signal trace |
| Filter components | Near the circuit they're filtering |
| Orientation | Align for shortest traces — rotation optimization |

### 2.10 Crystals and Oscillators

| Guideline | Detail |
|-----------|--------|
| Position | Within 10-20mm of IC |
| Load caps | Adjacent to crystal pins |
| Ground plane | Solid underneath, no signal routing |
| Guard traces | GND guard on both sides if space permits |
| Isolation | Away from high-speed digital and switching noise |

---

## 3. Board Isolation Techniques

### 3.1 Grounding Strategy

**Single continuous ground plane** — no splits unless absolutely necessary.

- Multiple vias for low-impedance paths (≥4 vias per square cm)
- If analog/digital separation needed: single-point connection near power supply
- Never route signals over ground plane gaps
- Via fencing around noisy sections (relays, switching regulators)

### 3.2 Zone Partitioning

Divide the board into logical zones:

| Zone | Contents | Isolation |
|------|----------|-----------|
| Digital | MCU, Ethernet PHY, USB, logic | Standard ground plane |
| Analog | ADC, op-amps, voltage references | Ferrite bead on power, single-point GND |
| Power | Regulators, bulk caps | Wide traces, thermal management |
| Switching | Relays, FETs, motor drivers | Via fence, board slots for >50V |
| RF | WiFi/BLE modules, antennas | Keepout zones, no digital routing |

**Physical separation**: 10-20mm between zones where possible.

**Signal crossings**: Short bridges over ground plane on inner layer, GND vias
at both ends.

### 3.3 Noise Reduction

| Technique | Application |
|-----------|-------------|
| Decoupling caps | 100nF near every IC power pin |
| Ferrite beads | Analog power supply isolation |
| RC filters | Analog inputs, sensitive signals |
| Via fencing | Around RF keepout zones, relay zones |
| Ground pours | Fill unused copper with ground |
| Shield cans | Over sensitive RF sections (optional) |

---

## 4. Trace Routing Rules

### 4.1 Trace Width by Current

| Current (A) | Width (mm) @ 1oz Cu | Width (mm) @ 2oz Cu |
|-------------|---------------------|---------------------|
| 0.5 | 0.25 | 0.15 |
| 1.0 | 0.50 | 0.30 |
| 2.0 | 1.00 | 0.60 |
| 3.0 | 1.50 | 0.90 |
| 5.0 | 2.50 | 1.50 |

*Based on IPC-2221 for 10°C rise. Use a trace width calculator for exact values.*

### 4.2 Clearance Rules

| Voltage | Min Clearance (mm) | Standard |
|---------|-------------------|----------|
| <30V | 0.15-0.25 | IPC-2221 |
| 30-60V | 0.6 | IPC-2221 |
| 60-150V | 1.0 | IPC-2221 |
| 150-300V | 1.5 | IPC-2221 |
| Mains (240V) | 2.5+ | IEC 60950 |

### 4.3 Impedance-Controlled Traces

| Signal Type | Impedance | Trace Width (typical) |
|-------------|-----------|----------------------|
| USB 2.0 D+/D- | 90Ω diff | 0.3mm, 0.15mm gap |
| Ethernet 10/100 | 100Ω diff | 0.25mm, 0.15mm gap |
| Single-ended 50Ω | 50Ω | ~0.45mm on standard FR4 |

*Exact width depends on stackup — use impedance calculator.*

---

## 5. Thermal Management

### 5.1 Heat Sources

| Component | Thermal Strategy |
|-----------|-----------------|
| Voltage regulators | Thermal vias (4-6x 0.3mm drill), copper pour |
| Power FETs | Exposed pad vias, wide traces |
| Relays | Copper fill for coil dissipation |
| LEDs (high-power) | Thermal pad connected to ground plane |
| MCU (high-clock) | Ground pour under package |

### 5.2 Thermal Via Array

For components with thermal/exposed pads:
- Via drill: 0.3mm
- Via spacing: 1.0-1.2mm pitch
- Count: 4-6 minimum for regulators, 9-16 for power ICs
- Connect to inner ground plane

---

## 6. Manufacturing Considerations (JLCPCB)

### 6.1 Standard Capabilities

| Parameter | Min Value |
|-----------|-----------|
| Trace width | 0.127mm (5 mil) |
| Trace spacing | 0.127mm (5 mil) |
| Via drill | 0.3mm (12 mil) |
| Via annular ring | 0.15mm |
| Hole-to-hole | 0.5mm |
| Board edge clearance | 0.3mm |
| Solder mask bridge | 0.1mm |

### 6.2 Assembly (SMT)

| Parameter | Value |
|-----------|-------|
| Min component size | 0402 (1005 metric) |
| Component spacing | ≥0.5mm between bodies |
| Fiducials | 3x global, ≥1mm diameter |
| Orientation | Match rotation offset database |

---

## 7. Design Review Checklist

### Required (Before Fab)
- [ ] All power nets have correct trace width for current
- [ ] Decoupling caps within 5mm of every IC power pin
- [ ] WiFi antenna keepout zone created (if applicable)
- [ ] Relay isolation verified (slots/clearance if >50V)
- [ ] Zone fill complete (press B in KiCad)
- [ ] DRC passes with zero errors
- [ ] All nets connected (zero unrouted)

### Recommended
- [ ] Thermal vias under all regulators with thermal pads
- [ ] ESD protection on all external connectors
- [ ] Test points on critical signals
- [ ] Silkscreen legible and not overlapping pads
- [ ] Board outline clearance for enclosure

### Optional
- [ ] Via-in-pad with plugged vias for BGA/QFN
- [ ] Controlled impedance verified with stackup calculator
- [ ] Panelization rails designed for V-cut or tab routing
- [ ] Assembly drawing with polarity markers

---

## 8. Module-Specific Guidelines

### 8.1 ESP32-S3-WROOM-1

- 39 GPIO pins, WiFi+BLE antenna integrated
- Place at board edge, antenna past edge or with 15mm keepout
- 3.3V supply only — do not connect 5V to any pin
- USB D+/D- directly to GPIO19/GPIO20 (no external PHY)
- Decoupling: 10uF + 100nF on 3V3, as close as possible
- EN pin: 10k pull-up + 1uF to GND for clean power-on reset
- IO0: 10k pull-up, boot button to GND for programming mode
- Crystal is internal — no external crystal needed
- Keep ground plane solid under module (except antenna area)
- Datasheet: https://www.espressif.com/sites/default/files/documentation/esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf

### 8.2 ADS1115 (16-bit ADC)

- I2C interface (SDA/SCL), 4 analog inputs
- Operating voltage: 2.0V to 5.5V
- Place in analog zone, away from digital noise
- 100nF + 10uF on VDD pin
- Ferrite bead on VDD supply from digital power rail
- Input voltage dividers: precision resistors (1% or better)
- ADDR pin sets I2C address (GND=0x48, VDD=0x49, SDA=0x4A, SCL=0x4B)
- Route I2C traces with ground guard traces if running near digital signals
- Datasheet: https://www.ti.com/lit/ds/symlink/ads1115.pdf

### 8.3 W5500 (Ethernet Controller)

- SPI interface, hardwired TCP/IP stack
- 3.3V supply, 25MHz crystal required
- Crystal: load caps per datasheet, within 10mm
- Decoupling: 100nF on each VCC pin + 10uF on AVDD
- Route SPI traces short — MISO/MOSI/SCK/CS within 50mm
- RJ45 connector with integrated magnetics recommended
- Keep analog section (AVDD, crystal, magnetics) isolated from digital
- Datasheet: https://www.wiznet.io/product-item/w5500/

### 8.4 Relay Driver Circuit (SRD-05VDC-SL-C)

- 5V coil, SPDT contacts rated 10A/250VAC
- Driver: NPN transistor (BC817) with 1k base resistor from GPIO
- Flyback diode (1N4148/1N4007) across coil — cathode to +5V
- Contact traces: ≥1mm for resistive loads, wider for inductive
- Isolation: minimum 10mm from analog section
- Board slots recommended between contacts and logic for >50V
- Group each relay with its transistor, base resistor, and flyback diode
- Screw terminals for relay outputs at board edge

### 8.5 USB-C Connector (USB 2.0)

- CC1/CC2: 5.1k pull-down to GND each (device mode)
- VBUS: TVS diode, ferrite bead, bulk cap (47uF+)
- D+/D-: ESD diodes near connector, 90-ohm differential routing
- Shield: connect to GND through 1M + 4.7nF to chassis
- Place at board edge with mechanical reinforcement

### 8.6 Buck Converter (AP63205WU)

- Input: 3.8-32V, Output: adjustable via feedback divider
- Input cap: 22uF ceramic, within 5mm of VIN
- Output cap: 22uF ceramic, within 5mm of output
- Bootstrap cap: 100nF between BST and SW
- Inductor: 4.7uH, short wide traces to SW pin
- Feedback divider: place near FB pin, away from inductor
- Ground plane solid under entire converter section
- Thermal pad: vias to ground plane for heat dissipation
- Keep high-current loop (VIN-SW-inductor-output) tight and short

### 8.7 LDO Regulator (AMS1117-3.3)

- Input: up to 15V, Output: 3.3V fixed
- Input cap: 10uF within 5mm of VIN
- Output cap: 22uF within 5mm of VOUT
- Thermal: SOT-223 tab connects to output — use copper pour for heatsinking
- Dropout: 1.1V typical — need ≥4.4V input for clean 3.3V
- Current: 800mA max — verify power budget

---

## 9. Common Anti-Patterns

| Anti-Pattern | Why It's Bad | Fix |
|--------------|-------------|-----|
| Split ground plane | Creates EMI loops, increases noise | Use continuous plane, single-point connection if needed |
| Decoupling cap far from IC | Ineffective at high frequencies | Place within 3-5mm of power pin |
| Signal routed under antenna | Degrades WiFi/BLE performance | Keepout zone — no copper under antenna |
| Relay contacts near ADC | Switching noise corrupts measurements | Separate by ≥15mm, use isolation slots |
| Thin power traces | Excessive voltage drop, heating | Calculate required width from current |
| Via in pad (no plugging) | Solder wicking during reflow | Use plugged/filled vias or move via away |
| Daisy-chain power | Voltage drop accumulates | Star topology from regulator |
| No ESD on external connectors | Susceptible to discharge damage | TVS diodes within 5mm of connector |

---

## 10. Subcircuit Pattern Library

Physical layout patterns for common subcircuits. Each pattern shows signal flow order,
pad connections, clearance targets, and rotation guidance. Use these as the reference
when reviewing or optimizing component placement.

**Notation**: `[pad N]` indicates which pad of a component connects to which pad of
the next. Arrow `-->` shows signal flow direction. Components in `( )` are optional.

### 10.1 Voltage Divider / ADC Input Channel

**Signal flow**: Connector pin --> R_top --> midpoint --> R_bot --> GND

```
                    Signal In (from connector)
                         |
                    +-----------+
                    |  R_top    |  pad 1 = signal in
                    |  (10k)    |  pad 2 = midpoint
                    +-----------+
                         |
                    midpoint net ----+---- to ADC input pin
                         |          |
                    +-----------+   |
                    |  R_bot    |   +-----------+
                    |  (10k)    |   | D_clamp   |  anode = midpoint
                    +-----------+   | (TVS/Zener)|  cathode = GND or 3V3
                         |         +-----------+
                        GND              |
                                   +-----------+
                                   | C_filter  |  pad 1 = midpoint
                                   | (100nF)   |  pad 2 = GND
                                   +-----------+
                                         |
                                        GND
```

**Physical layout** (top view, 0603 pads):
```
  [Connector]
      |
  [ R_top ]     ← vertical, pad1=top (signal in), pad2=bottom (midpoint)
      |
  [ R_bot ]     ← vertical, pad1=top (midpoint), pad2=bottom (GND)
      |
     GND
            [ D_clamp ]  ← horizontal, beside midpoint, anode facing R_top/R_bot junction
            [ C_filter ] ← horizontal, below D_clamp, pad1 facing midpoint net
```

**Clearance targets**:
- R_top pad2 to R_bot pad1: ≤1.5mm (series, same net = midpoint)
- D_clamp/C_filter to midpoint junction: ≤2.5mm
- Entire channel width: ≤8mm
- Channel-to-channel spacing: 3-5mm (repeating pattern)
- Channel group to ADC IC: ≤15mm

**Rotation rules**:
- R_top, R_bot: 0deg (vertical) — pads aligned along signal flow
- D_clamp: 90deg — anode pad facing the midpoint net trace
- C_filter: 90deg — pad1 facing midpoint net

**Repeatable pattern**: All ADC channels should use identical layout, offset horizontally.
Channel order should match ADC pin order (AIN0 closest to pin, AIN3 furthest).

### 10.2 Relay Driver Circuit

**Signal flow**: GPIO --> R_gate --> Q base --> Q collector --> K coil --> +5V
                                                                  D_flyback across K coil

```
  GPIO (from MCU)
      |
  +-----------+
  |  R_gate   |  pad 1 = GPIO net
  |  (1k)     |  pad 2 = Q base
  +-----------+
      |
  +-----------+
  |  Q (NPN)  |  base = R_gate pad2
  | (BC817)   |  collector = K coil pin
  +-----------+  emitter = GND
      |
  collector net
      |
  +-----------+       +-----------+
  |  K relay  |       | D_flyback |  cathode = +5V (coil supply)
  |  coil     |       | (1N4148)  |  anode = collector net
  +-----------+       +-----------+
      |                     |
    +5V coil              +5V
      |
  [Terminal Block] (relay contacts, on board edge)
```

**Physical layout** (top view):
```
  MCU side                                    Board edge side
  ─────────                                   ──────────────
  [R_gate] ── [Q] ── [D_flyback]  ──  [K relay]  ──  [Terminal]
                         |
                       +5V rail
```

**Clearance targets**:
- R_gate to Q: ≤3mm (gate resistor close to transistor base)
- Q collector to K coil pin: ≤5mm
- D_flyback across K coil pins: ≤3mm from coil
- K relay to terminal block: ≤10mm
- Relay driver subgroup (R+Q+D): spread ≤8mm

**Rotation rules**:
- R_gate: horizontal, pad2 facing Q base pad
- Q (SOT-23): orient so base faces R_gate, collector faces relay
- D_flyback: orient cathode toward +5V rail, anode toward collector
- K relay: coils parallel to board edge, contacts toward terminal

**Repeatable pattern**: All relay drivers in a 1xN row. Support components (R, Q, D)
on MCU side of each relay. Terminal blocks on board edge side.

### 10.3 Buck Converter

**Signal flow**: VIN --> C_in --> U (VIN pin) --> SW --> L --> VOUT --> C_out
                                    U (FB pin) <-- voltage divider

```
  Power In
      |
  +-----------+
  |  C_in     |  pad 1 = VIN
  | (22uF)    |  pad 2 = GND
  +-----------+
      |
  +-----------+
  |  U (buck) |  VIN = C_in, SW = inductor, FB = divider, GND = ground
  | AP63205WU |  BST = C_bst
  +-----------+
      |  SW pin
  +-----------+
  |  L (ind)  |  pad 1 = SW, pad 2 = VOUT
  | (4.7uH)  |
  +-----------+
      |
  +-----------+
  |  C_out    |  pad 1 = VOUT
  | (22uF)    |  pad 2 = GND
  +-----------+
      |
   VOUT rail

  Near FB pin:
  +-----------+
  |  R_fb_top |  pad 1 = VOUT, pad 2 = FB
  +-----------+
      |
  +-----------+
  |  R_fb_bot |  pad 1 = FB, pad 2 = GND
  +-----------+
```

**Physical layout** (top view):
```
  [C_in] ── [U buck] ── [L] ── [C_out]
                |
             [C_bst]
                |
          [R_fb_top]
          [R_fb_bot]
```

**Clearance targets**:
- C_in to U VIN pin: ≤3mm
- L to U SW pin: ≤5mm (short, wide trace)
- C_out to L output: ≤5mm
- Feedback divider to FB pin: ≤5mm
- C_bst to BST/SW pins: ≤3mm
- Total converter footprint: ≤20x15mm

**Key rule**: Minimize the high-current loop area (VIN → SW → L → C_out → GND → C_in).
Keep this loop tight and on the same layer.

### 10.4 LDO Regulator

**Signal flow**: VIN --> C_in --> U (VIN) --> U (VOUT) --> C_out --> VOUT rail

```
  [C_in] ── [U LDO] ── [C_out]
```

**Physical layout**: Linear, all three components in a row.

**Clearance targets**:
- C_in to U VIN pin: ≤5mm
- C_out to U VOUT pin: ≤5mm
- Total footprint: ≤15x8mm

### 10.5 Decoupling Pair

**Signal flow**: VCC pin --> C_small (100nF) --> C_bulk (10uF) --> GND

```
  IC VCC pin
      |
  [C_small 100nF]  ← closest to pin (≤3mm edge-to-edge)
      |
  [C_bulk 10uF]    ← slightly further (≤8mm from pin)
      |
     GND via
```

**Clearance targets**:
- C_small to IC VCC pin: ≤3mm edge-to-edge (NOT center-to-center)
- C_bulk to IC: ≤8mm
- Both caps connected to same GND via or nearby GND vias

**Rotation rules**: Orient so one pad connects directly to VCC trace, other pad
has short path to GND via. Minimize loop area (VCC → cap → GND → back to IC GND).

### 10.6 RC Low-Pass Filter

**Signal flow**: Signal In --> R --> junction --> C --> GND
                                     junction --> Signal Out (filtered)

```
  Signal In ── [R] ── junction ── Signal Out
                         |
                        [C]
                         |
                        GND
```

**Physical layout**: R and C in an L-shape. R inline with signal, C perpendicular
dropping to ground.

**Clearance targets**:
- R pad2 to C pad1: ≤1.5mm (same net = junction)
- Total footprint: ≤5x5mm

### 10.7 Crystal Oscillator

**Signal flow**: MCU OSC_IN --> Y pin1 --> Y pin2 --> MCU OSC_OUT
                               Y pin1 --> C_load1 --> GND
                               Y pin2 --> C_load2 --> GND

```
       MCU
    [OSC pins]
        |
   +---------+
   | C_load1 |    [Y crystal]    | C_load2 |
   +---------+                   +---------+
       |                              |
      GND                           GND
```

**Physical layout**: Crystal between the two MCU oscillator pins. Load caps
flanking the crystal, each connecting one crystal pin to GND.

**Clearance targets**:
- Crystal to MCU OSC pins: ≤5mm
- Load caps to crystal pins: ≤3mm
- All components within 10mm of MCU
- Solid ground plane underneath — no signal routing

**Rotation rules**: Orient crystal so pin1/pin2 align with MCU OSC_IN/OSC_OUT pins
for shortest symmetric traces.

### 10.8 I2C Pull-Up Network

**Signal flow**: VCC --> R_sda --> SDA bus
                 VCC --> R_scl --> SCL bus

```
    VCC
     |
  [R_sda]  [R_scl]
     |        |
    SDA      SCL
```

**Physical layout**: Two resistors side by side, near the MCU I2C pins (not near
the peripheral). Both connect to the same VCC rail.

**Clearance targets**:
- Pull-ups to MCU I2C pins: ≤10mm
- R_sda to R_scl: ≤3mm (same orientation, parallel)
