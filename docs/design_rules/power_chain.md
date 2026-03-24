# Power Chain Design Rules

## Signal Flow

Left-to-right: 24V input (left edge) -> 5V buck (center-left) -> 3.3V LDO (center-right) -> outputs (right edge).

## Subnet Architecture

Private subnets encode proximity-critical connections into the netlist itself.
When two components share a private subnet, the placement optimizer knows they
must be physically adjacent — the subnet IS the proximity constraint.

| Subnet | Connections | Purpose |
|--------|-------------|---------|
| `+5V_C2_DEC` | L1 pin 2, C2 pin 1 | C2 (buck output cap) must sit right at L1 output / U1 output node |
| `+5V_C4_DEC` | U2 VIN, C4 pin 1 | C4 (LDO input cap) must sit right at U2 VIN pin |
| `+3V3_C5_DEC` | U2 VOUT, U2 VOUT_TAB, C5 pin 1 | C5 (LDO output cap) must sit right at U2 VOUT pin |
| `BST_U1` | U1 BOOT, C3 pin 1 | C3 (bootstrap cap) must sit right at U1 BST pin |
| `+24V` | J1 pin 1, C1 pin 1, U1 VIN, U1 EN | Shared input rail — C1 serves the whole rail |
| `+5V` | R1 pin 1, J2 pin 1 | Shared 5V rail — feedback divider top and test point |
| `+3V3` | J3 pin 1 | Shared 3.3V rail — test point only |

### Why subnets, not just proximity rules?

1. **Self-documenting**: the netlist itself says "C2 belongs at the buck output"
2. **Tool-agnostic**: any placement optimizer that minimises net wirelength will
   automatically pull C2 toward L1, without needing hardcoded rules
3. **Scalable**: adding a second buck channel just means adding another `+12V_Cx_DEC` subnet
4. **Auditable**: DRC can flag if a decoupling cap is far from its IC by checking
   subnet wirelength, not component naming heuristics

### Capacitor assignment summary

| Ref | Value | Subnet | Physical location |
|-----|-------|--------|-------------------|
| C1 | 10uF | `+24V` (shared) | At U1 VIN — input decoupling for buck |
| C2 | 22uF | `+5V_C2_DEC` | At L1 output / U1 output — output decoupling for buck |
| C3 | 100nF | `BST_U1` | At U1 BST pin — bootstrap cap |
| C4 | 10uF | `+5V_C4_DEC` | At U2 VIN — input decoupling for LDO |
| C5 | 22uF | `+3V3_C5_DEC` | At U2 VOUT — output decoupling for LDO |

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
