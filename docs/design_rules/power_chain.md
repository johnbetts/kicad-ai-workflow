# Power Chain Design Rules

## Signal Flow

Left-to-right: 24V input (left edge) -> 5V buck (center-left) -> 3.3V LDO (center-right) -> outputs (right edge).

## Buck Converter (TPS54331) Layout Rules

1. **Input capacitor C1** must be within 5mm of U1 VIN pin — minimizes high-current input loop.
2. **Output inductor L1** must be within 5mm of U1 SW pin — minimize SW node copper area (high dV/dt).
3. **Output capacitor C2** must be within 5mm of L1 output side — closes the power loop tightly.
4. **Bootstrap capacitor C3** must be within 3mm of U1 BST pin — high-frequency path, keep short.
5. **Catch diode D1** must be within 3mm of U1 SW pin — carries inductor current during off-time.
6. **Feedback divider (R1, R2)** must be within 5mm of U1 FB pin — sensitive analog signal, keep away from SW node.
7. **High-current loop** (C1 -> U1 VIN -> U1 SW -> L1 -> C2 -> GND -> C1 GND) must be as small as possible.
8. Component order along signal flow: C1 -> U1 -> L1 -> C2 (left to right).

## LDO (AMS1117-3.3) Layout Rules

1. **Input capacitor C4** must be within 3mm of U2 VIN pin.
2. **Output capacitor C5** must be within 3mm of U2 VOUT pin.
3. Compact linear arrangement: C4 -> U2 -> C5 (left to right).

## Connector Placement

1. **J1** (24V input): left edge of board, within 3mm of edge.
2. **J2** (5V test point): board edge, near buck output section.
3. **J3** (3.3V test point): right edge of board, within 3mm of edge.

## Spacing and Isolation

- Buck converter high-current loop area should be minimized.
- Feedback divider R1/R2 should be on the quiet side of L1 (output side), away from the SW node.
- GND plane should be continuous under the power chain for low-impedance return path.
