# Analog Input / ADC Channel — Layout Design Rules

## ADS1115 Pinout Reference (MSOP-10)

```
Pin 1:  ADDR    -> I2C address select (tied to GND for 0x48)
Pin 2:  ALERT   -> Alert/DRDY output (no connect or pull-up)
Pin 3:  GND     -> Ground
Pin 4:  AIN0    -> Analog input channel 0 (AIN1_DIV)
Pin 5:  AIN1    -> Analog input channel 1 (AIN2_DIV)
Pin 6:  AIN2    -> Analog input channel 2 (AIN3_DIV)
Pin 7:  AIN3    -> Analog input channel 3 (AIN4_DIV)
Pin 8:  VDD     -> +3V3 power supply
Pin 9:  SDA     -> I2C data
Pin 10: SCL     -> I2C clock
```

## Per-Channel Signal Chain

Each analog input channel is a vertical strip from connector to ADC pin:

```
J{N} pin 1  (AIN{N}_RAW)
   |
R{2N-1}     (voltage divider top, 0805)
   |
   +-------- AIN{N}_DIV node
   |    |    |
R{2N}  D{N} C{N+1}     (divider bot, TVS, filter cap — all 0805/SOD-323)
   |    |    |
  GND  GND  GND
```

The AIN{N}_DIV node connects to U1 AIN{N-1} pin.

## Relative Positioning Rules

### Channel Strip Structure
Each channel is a vertical strip. All components in a channel share the same X center (+/-2mm).

### Vertical Order (top to bottom)
1. J{N} (2-pin screw terminal) — closest to top board edge, pads facing outward
2. R_top (voltage divider top resistor) — below connector
3. R_bot + D + C_filter (protection cluster) — at divider midpoint node
   - R_bot directly below R_top (vertical signal flow)
   - D (TVS) beside R_bot (same Y, offset X by ~3mm)
   - C_filter beside R_bot on opposite side from D (same Y, offset X by ~3mm)
4. U1 AIN pin — at bottom, receives filtered/divided signal

### ADC IC Placement
- U1 (ADS1115) centered horizontally on the board
- Positioned in the lower half of the board
- Analog input pins face upward toward the channel strips
- Digital pins (SDA, SCL) face downward toward J5 header

### Connector Row Rules
- All J1-J4 at the SAME Y position (top edge, within 3mm of board edge)
- Evenly spaced horizontally
- J5 (I2C header) at bottom edge of board, centered under U1

### Channel Spacing
- Channels spaced evenly, repeating identical pattern
- Minimum 10mm center-to-center between channels
- All 4 channels must be visually identical vertical strips

## Component Roles

### Per Channel (x4)
| Ref | Value | Package | Purpose |
|-----|-------|---------|---------|
| J{N} | Screw_Terminal_2P | 5.08mm pitch | Sensor input connector |
| R{2N-1} | 10K (top) | 0805 | Voltage divider top resistor |
| R{2N} | 10K (bottom) | 0805 | Voltage divider bottom resistor |
| D{N} | PESD3V3 | SOD-323 | TVS/Zener input protection (3.3V clamp) |
| C{N+1} | 100nF | 0805 | Anti-aliasing / noise filter capacitor |

### ADC
| Ref | Value | Package | Purpose |
|-----|-------|---------|---------|
| U1 | ADS1115 | MSOP-10 | 16-bit 4-channel I2C ADC |
| C1 | 100nF | 0805 | ADC VDD decoupling capacitor |

### I2C Pull-ups
| Ref | Value | Package | Purpose |
|-----|-------|---------|---------|
| R9 | 4.7K | 0805 | SDA pull-up to +3V3 |
| R10 | 4.7K | 0805 | SCL pull-up to +3V3 |

### MCU Header
| Ref | Value | Package | Purpose |
|-----|-------|---------|---------|
| J5 | PinHeader_1x04 | 2.54mm pitch | SDA, SCL, +3V3, GND to MCU |

## Net Connectivity

### Per-Channel Nets
| Net | Components Connected |
|-----|---------------------|
| AIN{N}_RAW | J{N} pin 1, R{2N-1} pin 1 |
| AIN{N}_DIV | R{2N-1} pin 2, R{2N} pin 1, D{N} anode, C{N+1} pin 1, U1 AIN{N-1} |
| GND | J{N} pin 2, R{2N} pin 2, D{N} cathode, C{N+1} pin 2 |

### Power / I2C Nets
| Net | Components Connected |
|-----|---------------------|
| +3V3 | U1 pin 8 (VDD), C1 pin 1, R9 pin 1, R10 pin 1, J5 pin 3 |
| GND | U1 pin 3, C1 pin 2, J5 pin 4 |
| SDA | U1 pin 9, R9 pin 2, J5 pin 1 |
| SCL | U1 pin 10, R10 pin 2, J5 pin 2 |

## Pad Facing Rules
| Component | Pad 1 faces | Pad 2 faces |
|-----------|-------------|-------------|
| R_top | toward connector (up) | toward R_bot / ADC (down) |
| R_bot | toward R_top (up) | toward GND (down) |
| D (TVS) | toward divider node (AIN_DIV) | toward GND |
| C_filter | toward divider node (AIN_DIV) | toward GND |

## Orientation
| Component | Rotation | Reason |
|-----------|----------|--------|
| J1-J4 (terminal) | 0 deg | Pads face top board edge |
| R_top, R_bot | 90 deg | Vertical signal flow (top to bottom) |
| D (TVS) | 0 deg | Horizontal, beside R_bot |
| C_filter | 0 deg | Horizontal, beside R_bot |
| U1 (ADC) | 0 deg | AIN pins face upward, I2C pins face downward |
| R9, R10 | 90 deg | Vertical, near ADC I2C pins |
| J5 (header) | 0 deg | At bottom edge |

## Analog Layout Best Practices

- **Short analog traces**: Keep AIN_DIV traces as short as possible (divider output to ADC pin).
- **GND pour under ADC**: Solid ground plane under U1 for stable voltage reference.
- **Separate analog and digital**: I2C traces (digital) should not cross analog input traces.
- **Decoupling proximity**: C1 must be within 3mm of U1 VDD pin.
- **Filter cap proximity**: Each C_filter should be within 5mm of its corresponding U1 AIN pin.
- **Guard ring consideration**: For high-impedance inputs, consider guard traces around AIN traces.
- **Star ground**: All analog GND connections should converge at a single point near U1 GND pin.

## Compliance Check Rules

### Channel Strip Alignment
| Rule | Tolerance | Description |
|------|-----------|-------------|
| All J at same Y | +/-1mm | Connector row horizontally aligned |
| R_top below J | dy 3-8mm, dx < 2mm | Top resistor directly below its connector |
| R_bot below R_top | dy 3-8mm, dx < 2mm | Bottom resistor directly below top resistor |
| Channel spacing uniform | max 2mm deviation | Equal spacing between channel strips |

### ADC Proximity
| Rule | Tolerance | Description |
|------|-----------|-------------|
| C1 to U1 | <= 5mm | Decoupling cap near ADC VDD |
| R9/R10 to U1 | <= 8mm | I2C pull-ups near ADC |
| J at board edge | Y < 5mm from top | Connectors at top edge |

### Repeating Pattern
| Rule | Tolerance | Description |
|------|-----------|-------------|
| All channels identical layout | visual | Same relative component positions per channel |
| Protection cluster at same Y | +/-2mm | D and C_filter aligned across channels |

## Quality Criteria (for review)
- [ ] All 4 channels are visually identical vertical strips
- [ ] No component overlaps
- [ ] All connectors J1-J4 at same Y (+/-1mm), at top edge
- [ ] R_top directly below its connector (dx < 2mm)
- [ ] R_bot directly below R_top (dx < 2mm)
- [ ] D and C_filter at same Y as R_bot junction
- [ ] Even spacing between channels
- [ ] U1 centered, in lower half of board
- [ ] C1 within 5mm of U1 VDD
- [ ] R9/R10 within 8mm of U1 SDA/SCL
- [ ] J5 at bottom edge, below U1
- [ ] Analog traces short, digital traces separated
- [ ] GND pour area under U1
