# Ethernet Subsystem Design Rules

## Overview

W5500 SPI-to-Ethernet controller with 25MHz crystal, RJ45+magnetics connector,
differential pair termination, and SPI breakout header.

## Signal Flow

RJ45 (board edge) -> W5500 (center) -> SPI header (opposite edge).
Differential pairs TX+/TX- and RX+/RX- run between W5500 and magnetics/RJ45.

## RJ45 Connector Placement

1. **J1 (RJ45) must be flush with the board edge** (within 2mm of top edge).
   Edge-mount connectors overhang the PCB; the jack face must be accessible.
2. The RJ45 is the mechanical anchor — place it first, then arrange everything
   relative to it.

## W5500 Placement

1. **W5500 (U1) must be within 15mm of J1** — short differential pairs reduce
   EMI and improve signal integrity.
2. Orient so that TX+/TX-/RX+/RX- pins face toward J1 for direct routing.
3. AVDD and VCC decoupling caps must be on the same side as the power pins.

## Crystal Layout

1. **Y1 (25MHz crystal) must be within 5mm of U1 XTAL pins** — long traces add
   parasitic capacitance and can cause startup failure.
2. **C4, C5 (load caps) must be within 3mm of Y1** — they form the resonant
   circuit with the crystal; extra trace inductance shifts the frequency.
3. Keep the crystal area away from high-speed digital and power switching nodes.
4. GND pour directly under the crystal for a low-impedance return path.

## Decoupling Capacitors

1. **C1 (100nF VCC decoupling) must be within 3mm of U1 VCC pin.**
2. **C2 (10uF bulk) must be within 5mm of U1** — provides bulk charge reservoir.
3. **C3 (100nF AVDD decoupling) must be within 3mm of U1 AVDD pin.**
4. Place decoupling caps on the same layer as U1, with short vias to GND plane.

## Differential Pair Routing (TX+/TX-, RX+/RX-)

1. TX+ and TX- traces must be **matched length** (within 0.5mm).
2. RX+ and RX- traces must be **matched length** (within 0.5mm).
3. Differential pairs should run parallel with controlled spacing (100 ohm differential impedance).
4. **R1, R2 (49.9R termination resistors) must be within 3mm of U1 TX pins** —
   series termination at the source end.
5. Keep TX and RX pairs separated by at least 2mm to minimize crosstalk.

## RSVD Resistor

1. **R3 (12.1K RSVD) must be within 5mm of U1** — connected to the RSVD pin
   for internal bias.

## SPI Header Placement

1. **J2 (SPI 6-pin header) should be on the opposite edge from J1** — the SPI
   bus runs to an external MCU board; placing it far from the RJ45 reduces
   coupling between Ethernet and SPI.
2. **J3 (power 2-pin header) should be adjacent to J2** — keeps the MCU
   interconnect compact.
3. Both headers should be within 5mm of the bottom board edge.

## Spacing and Isolation

- GND pour under the entire Ethernet area (W5500, crystal, magnetics, RJ45).
- Keep the analog section (AVDD, crystal) separated from the digital SPI section.
- Minimum 6mm between RJ45 shield pins and any signal trace (EMI coupling).
- The 25MHz crystal should not be between the W5500 and the RJ45 — place it
  to the side to avoid interfering with differential pair routing.

## Board Dimensions

50mm x 40mm. RJ45 at top edge, SPI headers at bottom edge.
