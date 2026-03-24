# MCU Core (ESP32-S3-WROOM-1) — Layout Design Rules

## ESP32-S3-WROOM-1 Pinout Reference

Datasheet Figure 3-1 (top view, antenna at top/north):
```
Left (1-14):   GND, 3V3, EN, IO4-IO7, IO15-IO18, IO8, IO19, IO20
Bottom (15-26): IO3, IO46, IO9-IO14, IO21, IO47, IO48, IO45
Right (27-40):  IO0, IO35-IO42, RXD0, TXD0, IO2, IO1, GND
Center (41):    GND exposed pad
```

Key pin assignments for this board:
| Function | GPIO | Pin # | Side |
|----------|------|-------|------|
| 3V3 | - | 2 | Left |
| EN/RESET | - | 3 | Left |
| USB D+ | IO19 | 13 | Left |
| USB D- | IO20 | 14 | Left |
| BOOT | IO0 | 27 | Right |
| TXD0 | - | 37 | Right |
| RXD0 | - | 36 | Right |
| LED | IO2 | 38 | Right |
| GND | - | 1, 40, 41 | Left, Right, Center |

## Antenna Placement (CRITICAL)

- The ESP32-S3-WROOM-1 has an onboard PCB antenna at the **top (north)** edge of the module
- The antenna end of the module MUST face a **board edge** (top or right)
- **No copper pour, traces, or components** within the antenna keepout zone
- Antenna keepout: 10mm clear zone beyond the module antenna edge
- Ground plane under the module body is acceptable (and recommended), but NOT under the antenna

## Decoupling Capacitors

- C1 (100nF) and C2 (10uF) MUST be within **3mm edge-to-edge** of U1 pin 2 (3V3)
- C1 (100nF, high-frequency) should be **closer** to the 3V3 pin than C2 (10uF, bulk)
- Both caps connect to 3V3 and GND — short, wide traces to GND via
- Place on the same side of the module as pin 2 (left side)

## Crystal Oscillator

- Y1 (40MHz) + C3/C4 (22pF load caps) within **5mm** of U1 OSC pins
- Note: ESP32-S3-WROOM-1 has an internal crystal; this is included for training purposes
- Crystal traces should be short, symmetric, and guarded by GND
- C3 and C4 should be equidistant from Y1 pads

## USB-C Connector

- J1 (USB-C) MUST be at a **board edge** (bottom or left preferred)
- USB D+/D- traces should be length-matched (90-ohm differential pair)
- CC1/CC2 pull-down resistors (R3, R4: 5.1K) within 5mm of J1
- VBUS trace should be wide (minimum 0.5mm for 500mA)

## Boot/Reset Buttons

- SW1 (BOOT) and SW2 (RESET) should be **accessible** — near a board edge
- R1 (10K pull-up on EN) within 5mm of U1 pin 3 (EN)
- R2 (10K pull-up on BOOT/IO0) within 5mm of U1 pin 27 (IO0)
- C5 (100nF debounce on EN) adjacent to R1 and SW2

## UART Debug Header

- J2 (4-pin header: TX, RX, 3V3, GND) at a **board edge**
- Pin order: TX, RX, 3V3, GND (standard FTDI pinout)
- Place on opposite side from USB-C if possible

## Status LED

- D1 (LED) + R5 (330R) connected to IO2 (pin 38, right side)
- Place on same side as IO2 pin for short trace

## No-Go Zones

- No traces under antenna area (extends beyond module body at antenna end)
- No high-speed signals near crystal
- Keep USB D+/D- away from noisy digital signals

## Compliance Check Rules

### Distance Checks
| Rule | Max Distance | Description |
|------|-------------|-------------|
| C1 to U1 3V3 pin | 3mm edge-to-edge | HF decoupling proximity |
| C2 to U1 3V3 pin | 3mm edge-to-edge | Bulk decoupling proximity |
| Y1 to U1 | 5mm center-to-center | Crystal proximity |
| R3/R4 to J1 | 5mm | CC resistors near USB-C |

### Edge Placement
| Rule | Tolerance | Description |
|------|-----------|-------------|
| J1 at board edge | <2mm from edge | USB-C connector flush with edge |
| J2 at board edge | <3mm from edge | UART header accessible |
| U1 antenna at edge | <5mm from edge | Antenna faces board edge |

### Pull-up Proximity
| Rule | Max Distance | Description |
|------|-------------|-------------|
| R1 to U1 EN (pin 3) | 5mm | Reset pull-up near EN pin |
| R2 to U1 IO0 (pin 27) | 5mm | Boot pull-up near IO0 pin |

## Quality Criteria (for review)
- [ ] Antenna end of U1 faces board edge with clear keepout
- [ ] Decoupling caps within 3mm of 3V3 pin
- [ ] Crystal within 5mm of U1
- [ ] USB-C at board edge
- [ ] CC resistors near USB-C connector
- [ ] Boot/Reset buttons accessible
- [ ] UART header at board edge
- [ ] No component overlaps
- [ ] Clean, organized layout with logical signal flow
- [ ] Status LED visible and accessible
