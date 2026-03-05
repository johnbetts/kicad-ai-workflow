# Hardware & Manufacturing Specifications Reference

All hardcoded spec values used by `kicad-ai-pipeline` with their authoritative sources.
Values here are the single source of truth; code should reference this document in comments.

---

## Raspberry Pi HAT Mechanical Specification

Source: [Raspberry Pi HAT Design Guide](https://github.com/raspberrypi/hats/blob/master/designguide.md)
and [HAT Board Mechanical Specification](https://github.com/raspberrypi/hats/blob/master/hat-board-mechanical.pdf)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Board width | 65.0 mm | |
| Board height | 56.0 mm | |
| Corner radius | 3.0 mm | All four corners |
| Mounting hole diameter | 2.7 mm (M2.5 clearance) | 4 holes |
| Mounting hole 1 (bottom-left) | x=3.5, y=3.5 mm | From bottom-left origin |
| Mounting hole 2 (top-left) | x=3.5, y=52.5 mm | |
| Mounting hole 3 (bottom-right) | x=61.5, y=3.5 mm | |
| Mounting hole 4 (top-right) | x=61.5, y=52.5 mm | |
| GPIO header center position | x=32.504, y=3.502 mm | 2x20 pin, 2.54mm pitch (pin 1 at 8.374, 4.772) |
| GPIO header orientation | Pins on bottom, socket on top | Mates with Pi GPIO |
| Keepout: under GPIO header | 3mm below PCB surface | For Pi components |
| Keepout: camera/display connectors | See HAT spec Fig 3 | Avoid routing copper here |
| PCB thickness | 1.0-1.6 mm | Standard HAT uses 1.6mm |

### GPIO Pin Assignments (active pins used by pipeline)

| Physical Pin | GPIO | Function | Net Name |
|-------------|------|----------|----------|
| 1 | - | 3.3V | +3V3 |
| 2 | - | 5V | +5V |
| 3 | GPIO2 | I2C1 SDA | I2C_SDA |
| 4 | - | 5V | +5V |
| 5 | GPIO3 | I2C1 SCL | I2C_SCL |
| 6 | - | GND | GND |
| 9 | - | GND | GND |
| 11 | GPIO17 | Alert/IRQ | ALRT |
| 14 | - | GND | GND |
| 17 | - | 3.3V | +3V3 |
| 20 | - | GND | GND |
| 25 | - | GND | GND |
| 30 | - | GND | GND |
| 34 | - | GND | GND |
| 39 | - | GND | GND |

---

## Arduino Uno R3 Shield Mechanical Specification

Source: [Arduino Uno R3 Reference Design](https://docs.arduino.cc/hardware/uno-rev3/)

| Parameter | Value |
|-----------|-------|
| Board width | 68.6 mm |
| Board height | 53.3 mm |
| Mounting hole 1 | x=14.0, y=2.54 mm |
| Mounting hole 2 | x=15.24, y=50.8 mm |
| Mounting hole 3 | x=66.04, y=7.62 mm |
| Mounting hole 4 | x=66.04, y=35.56 mm |
| Mounting hole diameter | 3.2 mm (M3 clearance) |

---

## JLCPCB Manufacturing Capabilities

Source: [JLCPCB Capabilities](https://jlcpcb.com/capabilities/pcb-capabilities)
(Standard 2-layer process; tighter values available with higher cost)

### PCB Fabrication

| Parameter | Minimum | Recommended | Used In Code |
|-----------|---------|-------------|-------------|
| Trace width | 0.127 mm (5 mil) | 0.2 mm (8 mil) | `JLCPCB_MIN_TRACE_MM`, `JLCPCB_RECOMMENDED_TRACE_MM` |
| Clearance (trace-to-trace) | 0.127 mm | 0.2 mm | `JLCPCB_MIN_CLEARANCE_MM`, `JLCPCB_RECOMMENDED_CLEARANCE_MM` |
| Via drill | 0.3 mm | 0.3 mm | `JLCPCB_MIN_VIA_DRILL_MM` |
| Via annular ring | 0.13 mm | 0.15 mm | `JLCPCB_MIN_VIA_ANNULAR_RING_MM` |
| Through-hole drill min | 0.2 mm | - | `JLCPCB_MIN_DRILL_MM` |
| Through-hole drill max | 6.3 mm | - | `JLCPCB_MAX_DRILL_MM` |
| Through-hole annular ring | 0.13 mm | 0.15 mm | `JLCPCB_MIN_ANNULAR_RING_MM` |
| Silkscreen line width | 0.153 mm (6 mil) | 0.2 mm | `JLCPCB_MIN_SILK_WIDTH_MM` |
| Copper-to-edge clearance | 0.3 mm | 0.5 mm | `JLCPCB_BOARD_EDGE_CLEARANCE_MM` |
| Board size min | 10 x 10 mm | - | `JLCPCB_MIN_BOARD_SIZE_MM` |
| Board size max | 500 x 500 mm | - | `JLCPCB_MAX_BOARD_SIZE_MM` |
| Solder mask clearance | 0.05 mm | 0.05 mm | `solder_mask_clearance` in project_file.py |
| Solder mask min web width | 0.1 mm | 0.1 mm | `solder_mask_min_width` |
| Hole-to-hole minimum | 0.25 mm | 0.5 mm | `min_hole_to_hole` |

### SMT Assembly (JLCPCB)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Min component size | 0402 (1005 metric) | Standard process |
| Placement accuracy | +/- 0.05 mm | |
| Component rotation offsets | See `data/rotation_offsets.json` | Per-package correction for pick-and-place |

---

## IPC Footprint Standards

Source: IPC-7351B (Generic Requirements for Surface Mount Design and Land Pattern Standard)

### Passive SMD Packages (R/C/L)

| Package | Pad W x H (mm) | Pitch (mm) | Body W x H (mm) | Used In |
|---------|----------------|------------|-----------------|---------|
| 0402 | 0.5 x 0.5 | 1.0 | 1.0 x 0.5 | `_SMD_RC_DIMS` |
| 0603 | 0.8 x 0.8 | 1.6 | 1.6 x 0.8 | `_SMD_RC_DIMS` |
| 0805 | 1.2 x 1.4 | 2.0 | 2.0 x 1.25 | `_SMD_RC_DIMS` |
| 1206 | 1.5 x 1.7 | 3.2 | 3.2 x 1.6 | `_SMD_RC_DIMS` |
| 1210 | 1.5 x 2.5 | 3.2 | 3.2 x 2.5 | `_SMD_RC_DIMS` |

### SOT-23 Family (JEDEC)

Source: JEDEC MO-178 (SOT-23), JEDEC MO-178D (SOT-23-5/6)

| Package | Pads | Pad W x H (mm) | Pin Pitch (mm) |
|---------|------|----------------|----------------|
| SOT-23 | 3 | 0.9 x 1.3 | ~1.9 (row), 2.0 (col) |
| SOT-23-5 | 5 | 0.6 x 1.0 | 0.95 (row), 1.8 (col) |
| SOT-23-6 | 6 | 0.6 x 1.0 | 0.95 (row), 1.8 (col) |

---

## KiCad 9 File Format

Source: [KiCad File Formats](https://dev-docs.kicad.org/en/file-formats/)

| Parameter | Value | Constant |
|-----------|-------|----------|
| Schematic format version | 20250114 | `KICAD_SCH_VERSION` |
| PCB format version | 20241229 | `KICAD_PCB_VERSION` |
| Generator name | "kicad-ai-pipeline" | `KICAD_GENERATOR` |
| Generator version | "9.0" | `KICAD_GENERATOR_VERSION` |

### Layer Numbers (PCB)

| Layer | Number | S-expr Name |
|-------|--------|-------------|
| Front Copper | 0 | F.Cu |
| Front Mask | 1 | F.Mask |
| Back Copper | 2 | B.Cu |
| Back Mask | 3 | B.Mask |
| Front Silkscreen | 5 | F.SilkS |
| Back Silkscreen | 4 | B.SilkS |
| Front Courtyard | 6 | F.CrtYd |
| Back Courtyard | 7 | B.CrtYd |
| Front Fab | 8 | F.Fab |
| Back Fab | 9 | B.Fab |
| Edge Cuts | 44 | Edge.Cuts |

### Coordinate System

- Origin: top-left corner of the board
- X: increases rightward (mm)
- Y: increases downward (mm)
- Rotation: degrees, clockwise positive (KiCad PCB convention)
- Schematic and PCB share the same Y-down convention

---

## Net Class Design Rules

These are the pipeline's default net class assignments. Values are chosen for
mid-range PCB complexity (2-layer board, standard JLCPCB process).

| Net Class | Trace Width | Clearance | Via Dia | Via Drill | Matched Nets |
|-----------|-------------|-----------|---------|-----------|--------------|
| Default | 0.25 mm | 0.2 mm | 0.8 mm | 0.508 mm | All unmatched |
| Power | 0.3 mm | 0.2 mm | 0.8 mm | 0.508 mm | GND, +nVn, VCC, VDD, VBUS, PWR, V_*, VBAT |
| HighVoltageAnalog | 0.4 mm | 0.2 mm | 0.8 mm | 0.508 mm | SENS*, AIN*, ADC*, VREF* |
| SPI | 0.2 mm | 0.2 mm | 0.8 mm | 0.508 mm | SPI*, MOSI, MISO, SCLK, SCK, CS |
| I2C | 0.25 mm | 0.2 mm | 0.2 mm | 0.508 mm | I2C*, SDA, SCL |

### Design Rationale

- **All clearances unified at 0.2mm**: Matches KiCad default netclass.
  Prevents inherent DRC violations on fine-pitch ICs (e.g., MSOP-10
  with 0.5mm pin pitch has only 0.2mm pad-to-pad gap).
- **Power trace 0.3mm**: ~1.5x default for current handling.
  0.3mm at 1oz Cu handles ~0.5A with 10C rise (IPC-2221).
- **HVA trace 0.4mm**: Wider for noise immunity on high-impedance analog inputs.
- **Via 0.8mm / 0.508mm drill**: Standard JLCPCB via (20 mil drill).

---

## Placement Engine Constants

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `PLACEMENT_GAP_MM` | 0.5 mm | Prevents solder mask bridging between SMD pads |
| `_THT_GAP_MM` | 1.0 mm | THT pads/solder rings are larger |
| `_THT_SIZE_THRESHOLD_MM` | 5.0 mm | Footprint dimension above which THT gap applies |
| `_DEFAULT_GRID_MM` | 0.5 mm | Occupancy grid resolution for placement solver |
| Board edge margin | 3.0 mm | Connector placement margin from edge |
| Mounting hole keepout | 2.5 mm radius | Copper/component clearance around M2.5/M3 holes |

---

## Routing Grid Constants

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Grid step | 0.25 mm | 4x finer than placement grid; balances resolution vs performance |
| Pad clearance | max(netclass clearances) + max(track widths)/2 | Conservative: any track approaching any pad |
| Track exclusion | ceil((clearance + width) / grid_step) - 1 cells | Ensures edge-to-edge gap meets netclass clearance |
| Board edge margin | ceil(JLCPCB_BOARD_EDGE_CLEARANCE_MM / grid_step) + 1 cells | JLCPCB copper-to-edge requirement |

---

## Zone / Copper Pour

| Parameter | Value | Source |
|-----------|-------|--------|
| Zone clearance | 0.2 mm | KiCad default (`ZONE_CLEARANCE_DEFAULT_MM`) |
| Zone min thickness | 0.25 mm | IPC standard (`ZONE_MIN_THICKNESS_MM`) |
| Thermal relief gap | 0.3 mm | KiCad default |
| Thermal relief bridge | 0.5 mm | KiCad default |
| GND pour strategy | "both" (F.Cu + B.Cu) | Dual-layer improves GND return path |

---

## Standard Component Values

Source: IEC 60063 (Preferred number series for resistors and capacitors)

File: `data/e_series.json`

| Series | Values/Decade | Tolerance | Use Case |
|--------|--------------|-----------|----------|
| E6 | 6 | 20% | Rough filtering |
| E12 | 12 | 10% | General purpose |
| E24 | 24 | 5% | Standard precision |
| E96 | 96 | 1% | Precision circuits |
