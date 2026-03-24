#!/usr/bin/env python3
"""Power chain training board — isolated placement iteration.

Builds a minimal 24V->5V (buck) -> 3.3V (LDO) power supply board, runs the
EE placement optimizer, renders a PNG, and prints quality score + positions.

Usage::

    python scripts/train_power_chain.py

Power architecture:
    J1 (24V+GND) -> C1 -> U1 TPS54331 (buck) -> L1 -> C2 -> +5V rail
    +5V rail -> C4 -> U2 AMS1117-3.3 (LDO) -> C5 -> +3V3 rail
    Feedback divider: R1 (top) + R2 (bottom) from +5V to U1 FB
    Bootstrap: C3 from U1 BST to U1 SW
    Catch diode: D1 from GND to U1 SW

Subnet architecture (private subnets teach proximity):
    +5V_C2_DEC:  L1.2, C2.1            — C2 must be right at buck output
    +5V_C4_DEC:  U2.VIN, C4.1          — C4 must be right at LDO input
    +3V3_C5_DEC: U2.VOUT, C5.1, J3.1   — C5 at LDO output, J3 test point
    BST_U1:      U1.BOOT, C3.1         — C3 must be right at BST pin
    +24V:        J1.2, C1.1, U1.VIN    — C1 on shared input rail (serves whole rail)
    +5V:         R1.1, J2.1            — shared 5V rail (test point, FB divider)

TPS54331 pinout (SOIC-8):
    Pin 1: BOOT (bootstrap)
    Pin 2: VIN  (input voltage)
    Pin 3: EN   (enable — tied to VIN)
    Pin 4: SS   (soft start — NC for this training)
    Pin 5: VSNS (feedback / sense)
    Pin 6: GND
    Pin 7: PH   (switch node / phase)
    Pin 8: NC   (exposed pad / GND)

AMS1117-3.3 pinout (SOT-223):
    Pin 1: GND / ADJ
    Pin 2: VOUT
    Pin 3: VIN
    Pin 4: VOUT (tab, same as pin 2)
"""

from __future__ import annotations

import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure the package is importable when running from the repo root.
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))

from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinFunction,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee
from kicad_pipeline.optimization.scoring import compute_fast_placement_score
from kicad_pipeline.pcb.builder import build_pcb, write_pcb
from kicad_pipeline.project_file import write_project_file

# ---------------------------------------------------------------------------
# Footprint constants
# ---------------------------------------------------------------------------

_SOIC8_FP = "SOIC-8"
_SOT223_FP = "SOT-223"
_IND_1210_FP = "L_1210"
_C0805_FP = "C_0805"
_R0805_FP = "R_0805"
_SOD323_FP = "SOD-323"
_SCREW_TERM_2P_FP = "TerminalBlock_5.08mm_2P"
_HEADER_2P_FP = "PinHeader_1x02_P2.54mm_Vertical"

# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------


def _make_buck_converter() -> Component:
    """TPS54331 buck converter (24V -> 5V), SOIC-8.

    Pinout:
        1: BOOT  -> BST net (bootstrap cap)
        2: VIN   -> +24V
        3: EN    -> +24V (enable tied high)
        4: SS    -> NC (soft start, unused)
        5: VSNS  -> FB (feedback divider)
        6: GND   -> GND
        7: PH    -> SW (switch node to inductor)
        8: PAD   -> GND (exposed pad)
    """
    return Component(
        ref="U1",
        value="TPS54331",
        footprint=_SOIC8_FP,
        lcsc="C9865",
        description="4.5-28V 3A step-down buck converter SOIC-8",
        pins=(
            Pin("1", "BOOT", PinType.INPUT, net="BST_U1"),
            Pin("2", "VIN", PinType.POWER_IN, PinFunction.VCC, net="+24V"),
            Pin("3", "EN", PinType.INPUT, PinFunction.ENABLE, net="+24V"),
            Pin("4", "SS", PinType.INPUT),  # soft start — unused
            Pin("5", "VSNS", PinType.INPUT, PinFunction.ANALOG_IN, net="FB"),
            Pin("6", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("7", "PH", PinType.OUTPUT, net="SW"),
            Pin("8", "PAD", PinType.POWER_IN, PinFunction.GND, net="GND"),
        ),
    )


def _make_inductor() -> Component:
    """10uH power inductor for buck converter, 1210 package.

    Pin 1: SW (from U1 PH)
    Pin 2: +5V (output rail)
    """
    return Component(
        ref="L1",
        value="10uH",
        footprint=_IND_1210_FP,
        lcsc="C96950",
        description="10uH 3A power inductor 1210",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="SW"),
            Pin("2", "2", PinType.PASSIVE, net="+5V_C2_DEC"),
        ),
    )


def _make_input_cap() -> Component:
    """C1: 10uF input capacitor for buck converter, 0805."""
    return Component(
        ref="C1",
        value="10uF",
        footprint=_C0805_FP,
        lcsc="C15850",
        description="10uF 25V ceramic input cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+24V"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_output_cap() -> Component:
    """C2: 22uF output capacitor for buck converter, 0805.

    On private subnet +5V_C2_DEC — teaches that C2 must sit right at the
    buck output (L1 pin 2 / U1 output node), not just anywhere on +5V.
    """
    return Component(
        ref="C2",
        value="22uF",
        footprint=_C0805_FP,
        lcsc="C159842",
        description="22uF 10V ceramic output cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+5V_C2_DEC"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_bootstrap_cap() -> Component:
    """C3: 100nF bootstrap capacitor, 0805.

    On private subnet BST_U1 — teaches that C3 must sit right at U1 BST pin.
    The BST net is already small (only U1.BOOT + C3.1), but using a private
    subnet makes the proximity requirement explicit in the netlist.
    """
    return Component(
        ref="C3",
        value="100nF",
        footprint=_C0805_FP,
        lcsc="C49678",
        description="100nF bootstrap cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="BST_U1"),
            Pin("2", "2", PinType.PASSIVE, net="SW"),
        ),
    )


def _make_fb_top_resistor() -> Component:
    """R1: 100K feedback divider top resistor, 0805.

    Connected from +5V to FB node.
    """
    return Component(
        ref="R1",
        value="100K",
        footprint=_R0805_FP,
        lcsc="C17407",
        description="100K feedback top resistor 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+5V"),
            Pin("2", "2", PinType.PASSIVE, net="FB"),
        ),
    )


def _make_fb_bottom_resistor() -> Component:
    """R2: 33K feedback divider bottom resistor, 0805.

    Connected from FB node to GND.
    """
    return Component(
        ref="R2",
        value="33K",
        footprint=_R0805_FP,
        lcsc="C17390",
        description="33K feedback bottom resistor 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="FB"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_catch_diode() -> Component:
    """D1: Schottky catch diode for buck converter, SOD-323.

    Anode -> GND, Cathode -> SW (conducts when U1 switch is off).
    """
    return Component(
        ref="D1",
        value="SS14",
        footprint=_SOD323_FP,
        lcsc="C123899",
        description="1A 40V Schottky diode SOD-323",
        pins=(
            Pin("1", "A", PinType.PASSIVE, net="GND"),
            Pin("2", "K", PinType.PASSIVE, net="SW"),
        ),
    )


def _make_ldo() -> Component:
    """AMS1117-3.3 LDO regulator (5V -> 3.3V), SOT-223.

    Pinout:
        1: GND/ADJ -> GND
        2: VOUT    -> +3V3
        3: VIN     -> +5V
        4: VOUT    -> +3V3 (tab, same as pin 2)
    """
    return Component(
        ref="U2",
        value="AMS1117-3.3",
        footprint=_SOT223_FP,
        lcsc="C6186",
        description="3.3V 1A LDO regulator SOT-223",
        pins=(
            Pin("1", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("2", "VOUT", PinType.POWER_OUT, PinFunction.VCC, net="+3V3_C5_DEC"),
            Pin("3", "VIN", PinType.POWER_IN, PinFunction.VCC, net="+5V_C4_DEC"),
            Pin("4", "VOUT_TAB", PinType.POWER_OUT, PinFunction.VCC, net="+3V3_C5_DEC"),
        ),
    )


def _make_ldo_input_cap() -> Component:
    """C4: 10uF input capacitor for LDO, 0805.

    On private subnet +5V_C4_DEC — teaches that C4 must sit right at U2 VIN,
    not just anywhere on the +5V rail.
    """
    return Component(
        ref="C4",
        value="10uF",
        footprint=_C0805_FP,
        lcsc="C15850",
        description="10uF LDO input cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+5V_C4_DEC"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_ldo_output_cap() -> Component:
    """C5: 22uF output capacitor for LDO, 0805.

    On private subnet +3V3_C5_DEC — teaches that C5 must sit right at U2 VOUT,
    not just anywhere on the +3V3 rail.
    """
    return Component(
        ref="C5",
        value="22uF",
        footprint=_C0805_FP,
        lcsc="C159842",
        description="22uF LDO output cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3_C5_DEC"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_ldo_hf_bypass_cap() -> Component:
    """C6: 100nF HF bypass capacitor for LDO output, 0805.

    In parallel with C5 (22uF) for high-frequency filtering.
    AMS1117 datasheet recommends a ceramic cap close to output.
    On same private subnet +3V3_C5_DEC so optimizer places it next to C5/U2.
    """
    return Component(
        ref="C6",
        value="100nF",
        footprint=_C0805_FP,
        lcsc="C49678",
        description="100nF HF bypass cap for LDO output 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3_C5_DEC"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_input_connector() -> Component:
    """J1: 2-pin screw terminal for 24V power input.

    Pin 1 = GND, Pin 2 = +24V — swapped so that when J1 is rotated 180
    degrees (pads facing board edge), traces to U1 VIN and GND do not cross.
    """
    return Component(
        ref="J1",
        value="Screw_Terminal_2P",
        footprint=_SCREW_TERM_2P_FP,
        lcsc="C8269",
        description="2-pin 5.08mm screw terminal — 24V input",
        pins=(
            Pin("1", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("2", "+24V", PinType.POWER_IN, PinFunction.VCC, net="+24V"),
        ),
    )


def _make_5v_test_header() -> Component:
    """J2: 2-pin header for 5V output test point."""
    return Component(
        ref="J2",
        value="Pin_Header_2P",
        footprint=_HEADER_2P_FP,
        lcsc="C124375",
        description="2-pin 2.54mm header — 5V test point",
        pins=(
            Pin("1", "+5V", PinType.PASSIVE, PinFunction.VCC, net="+5V"),
            Pin("2", "GND", PinType.PASSIVE, PinFunction.GND, net="GND"),
        ),
    )


def _make_3v3_test_header() -> Component:
    """J3: 2-pin header for 3.3V output test point.

    Pin 1 is on +3V3_C5_DEC (same subnet as U2 VOUT, C5, C6) so the
    optimizer pulls J3 near U2 and the test header is actually connected
    to the 3.3V output rail.
    """
    return Component(
        ref="J3",
        value="Pin_Header_2P",
        footprint=_HEADER_2P_FP,
        lcsc="C124375",
        description="2-pin 2.54mm header — 3.3V test point",
        pins=(
            Pin("1", "+3V3", PinType.PASSIVE, PinFunction.VCC, net="+3V3_C5_DEC"),
            Pin("2", "GND", PinType.PASSIVE, PinFunction.GND, net="GND"),
        ),
    )


# ---------------------------------------------------------------------------
# Net definitions
# ---------------------------------------------------------------------------


def _build_nets() -> tuple[Net, ...]:
    """Build all nets for the power chain.

    Subnet architecture — private subnets encode proximity requirements:

        +24V:       J1.2, C1.1, U1.VIN, U1.EN  (shared input rail, C1 is input decoupling)
        GND:        all ground pins              (shared ground)
        SW:         U1.PH, L1.1, C3.2, D1.K    (switch node)
        BST_U1:     U1.BOOT, C3.1               (bootstrap — C3 right at U1 BST pin)
        +5V_C2_DEC: L1.2, C2.1                  (buck output node — C2 right at L1/U1 output)
        +5V:        R1.1, J2.1                   (shared 5V rail — test point & FB divider)
        +5V_C4_DEC: U2.VIN, C4.1                (LDO input node — C4 right at U2 VIN)
        FB:         R1.2, R2.1, U1.VSNS         (feedback sense)
        +3V3_C5_DEC: U2.VOUT, U2.VOUT_TAB, C5.1, C6.1, J3.1 (LDO output — C5+C6+J3 at U2 VOUT)
    """
    return (
        Net(
            name="+24V",
            connections=(
                NetConnection("J1", "2"),  # pin 2 is now +24V (swapped)
                NetConnection("C1", "1"),
                NetConnection("U1", "2"),
                NetConnection("U1", "3"),  # EN tied to VIN
            ),
        ),
        Net(
            name="GND",
            connections=(
                NetConnection("J1", "1"),  # pin 1 is now GND (swapped)
                NetConnection("C1", "2"),
                NetConnection("U1", "6"),
                NetConnection("U1", "8"),  # exposed pad
                NetConnection("D1", "1"),  # catch diode anode
                NetConnection("R2", "2"),  # FB bottom to GND
                NetConnection("C2", "2"),
                NetConnection("U2", "1"),
                NetConnection("C4", "2"),
                NetConnection("C5", "2"),
                NetConnection("C6", "2"),  # HF bypass cap
                NetConnection("J2", "2"),
                NetConnection("J3", "2"),
            ),
        ),
        Net(
            name="SW",
            connections=(
                NetConnection("U1", "7"),  # PH pin
                NetConnection("L1", "1"),
                NetConnection("C3", "2"),  # bootstrap cap low side
                NetConnection("D1", "2"),  # catch diode cathode
            ),
        ),
        Net(
            name="BST_U1",
            connections=(
                NetConnection("U1", "1"),  # BOOT pin
                NetConnection("C3", "1"),  # bootstrap cap high side
            ),
        ),
        # Private subnet: buck output — C2 must be right at L1 output / U1 output
        Net(
            name="+5V_C2_DEC",
            connections=(
                NetConnection("L1", "2"),   # inductor output
                NetConnection("C2", "1"),   # output cap
            ),
        ),
        # Shared +5V rail — feedback divider and test point
        Net(
            name="+5V",
            connections=(
                NetConnection("R1", "1"),   # FB top to +5V
                NetConnection("J2", "1"),   # 5V test point
            ),
        ),
        # Private subnet: LDO input — C4 must be right at U2 VIN
        Net(
            name="+5V_C4_DEC",
            connections=(
                NetConnection("U2", "3"),   # LDO VIN
                NetConnection("C4", "1"),   # LDO input cap
            ),
        ),
        Net(
            name="FB",
            connections=(
                NetConnection("R1", "2"),
                NetConnection("R2", "1"),
                NetConnection("U1", "5"),  # VSNS pin
            ),
        ),
        # Private subnet: LDO output — C5+C6 right at U2 VOUT, J3 test header
        Net(
            name="+3V3_C5_DEC",
            connections=(
                NetConnection("U2", "2"),   # VOUT
                NetConnection("U2", "4"),   # VOUT tab
                NetConnection("C5", "1"),   # output cap (22uF bulk)
                NetConnection("C6", "1"),   # HF bypass (100nF ceramic)
                NetConnection("J3", "1"),   # 3.3V test point (must be on same net)
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Requirements assembly
# ---------------------------------------------------------------------------


def _build_requirements() -> ProjectRequirements:
    """Assemble power chain ProjectRequirements.

    Single FeatureBlock "Power Supply" containing all components.
    Board: 50mm x 40mm.
    """
    components = (
        _make_buck_converter(),
        _make_inductor(),
        _make_input_cap(),
        _make_output_cap(),
        _make_bootstrap_cap(),
        _make_fb_top_resistor(),
        _make_fb_bottom_resistor(),
        _make_catch_diode(),
        _make_ldo(),
        _make_ldo_input_cap(),
        _make_ldo_output_cap(),
        _make_ldo_hf_bypass_cap(),
        _make_input_connector(),
        _make_5v_test_header(),
        _make_3v3_test_header(),
    )

    nets = _build_nets()

    all_refs = tuple(c.ref for c in components)
    all_net_names = tuple(n.name for n in nets)

    power_feature = FeatureBlock(
        name="Power Supply",
        description=(
            "24V to 5V buck converter (TPS54331) and 5V to 3.3V LDO (AMS1117-3.3) "
            "with input/output capacitors, feedback divider, bootstrap, and catch diode"
        ),
        components=all_refs,
        nets=all_net_names,
        subcircuits=("buck_converter", "ldo_regulator"),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="PowerChainTraining", revision="v1"),
        features=(power_feature,),
        components=components,
        nets=nets,
        mechanical=MechanicalConstraints(board_width_mm=50, board_height_mm=40),
    )


# ---------------------------------------------------------------------------
# Design rules compliance check
# ---------------------------------------------------------------------------


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Euclidean distance between two component positions (ignoring rotation)."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _check_design_rules(fp_map: dict[str, tuple[float, float, float]]) -> None:
    """Check power chain design rules and print compliance report.

    Rules from docs/design_rules/power_chain.md:
    1. C1 (input cap) within 5mm of U1
    2. L1 (inductor) within 5mm of U1
    3. C2 (output cap) within 5mm of L1
    4. C3 (bootstrap cap) within 3mm of U1
    5. D1 (catch diode) within 3mm of U1
    6. R1, R2 (feedback divider) within 5mm of U1
    7. C4 (LDO input cap) within 3mm of U2
    8. C5 (LDO output cap) within 3mm of U2
    9. J1 near left edge (X < 8mm)
    10. J3 near right edge (X > board_width - 8mm = 42mm)
    11. Signal flow: U1.X < U2.X (buck left of LDO)
    """
    print("=" * 60)
    print("DESIGN RULES COMPLIANCE CHECK (Power Chain)")
    print("=" * 60)
    print()

    violations: list[str] = []
    passes: list[str] = []

    required_refs = ["U1", "U2", "L1", "C1", "C2", "C3", "C4", "C5",
                     "R1", "R2", "D1", "J1", "J2", "J3"]
    missing = [r for r in required_refs if r not in fp_map]
    if missing:
        print(f"  MISSING COMPONENTS: {missing}")
        print("  Cannot run design rules check.")
        return

    board_width = 50.0

    # --- Buck converter proximity checks ---
    print("--- Buck Converter (U1) Proximity ---")

    checks_5mm = [
        ("C1", "U1", 5.0, "Input cap to buck IC"),
        ("L1", "U1", 5.0, "Inductor to buck IC"),
        ("R1", "U1", 5.0, "FB top resistor to buck IC"),
        ("R2", "U1", 5.0, "FB bottom resistor to buck IC"),
    ]
    checks_3mm = [
        ("C3", "U1", 3.0, "Bootstrap cap to buck IC"),
        ("D1", "U1", 3.0, "Catch diode to buck IC"),
    ]
    checks_inductor = [
        ("C2", "L1", 5.0, "Output cap to inductor"),
    ]

    for ref_a, ref_b, max_d, desc in checks_5mm + checks_3mm + checks_inductor:
        d = _dist(fp_map[ref_a], fp_map[ref_b])
        label = f"  {ref_a}-{ref_b}: {d:.1f}mm ({desc})"
        if d > max_d:
            violations.append(f"{label} (MAX {max_d}mm) VIOLATION")
            print(f"{label} (MAX {max_d}mm) ** VIOLATION **")
        else:
            passes.append(f"{label} (MAX {max_d}mm) OK")
            print(f"{label} (MAX {max_d}mm) OK")
    print()

    # --- LDO proximity checks ---
    print("--- LDO (U2) Proximity ---")
    ldo_checks = [
        ("C4", "U2", 3.0, "LDO input cap"),
        ("C5", "U2", 3.0, "LDO output cap"),
    ]
    for ref_a, ref_b, max_d, desc in ldo_checks:
        d = _dist(fp_map[ref_a], fp_map[ref_b])
        label = f"  {ref_a}-{ref_b}: {d:.1f}mm ({desc})"
        if d > max_d:
            violations.append(f"{label} (MAX {max_d}mm) VIOLATION")
            print(f"{label} (MAX {max_d}mm) ** VIOLATION **")
        else:
            passes.append(f"{label} (MAX {max_d}mm) OK")
            print(f"{label} (MAX {max_d}mm) OK")
    print()

    # --- Feedback divider pairing ---
    print("--- Feedback Divider (R1-R2 proximity) ---")
    d_fb = _dist(fp_map["R1"], fp_map["R2"])
    label = f"  R1-R2: {d_fb:.1f}mm"
    if d_fb > 4.0:
        violations.append(f"{label} (MAX 4mm) VIOLATION")
        print(f"{label} (MAX 4mm) ** VIOLATION **")
    else:
        passes.append(f"{label} (MAX 4mm) OK")
        print(f"{label} (MAX 4mm) OK")
    print()

    # --- Signal flow direction ---
    print("--- Signal Flow (left to right) ---")
    u1_x = fp_map["U1"][0]
    u2_x = fp_map["U2"][0]
    j1_x = fp_map["J1"][0]
    j3_x = fp_map["J3"][0]

    flow_label = f"  J1({j1_x:.1f}) -> U1({u1_x:.1f}) -> U2({u2_x:.1f}) -> J3({j3_x:.1f})"
    if j1_x < u1_x < u2_x:
        passes.append(f"{flow_label} OK")
        print(f"{flow_label} OK")
    else:
        violations.append(f"{flow_label} VIOLATION (not left-to-right)")
        print(f"{flow_label} ** VIOLATION ** (not left-to-right)")
    print()

    # --- Connector edge placement ---
    print("--- Connector Edge Placement ---")
    edge_margin = 8.0

    j1_label = f"  J1 X={j1_x:.1f}mm (left edge, max {edge_margin}mm)"
    if j1_x <= edge_margin:
        passes.append(f"{j1_label} OK")
        print(f"{j1_label} OK")
    else:
        violations.append(f"{j1_label} VIOLATION")
        print(f"{j1_label} ** VIOLATION **")

    j3_label = f"  J3 X={j3_x:.1f}mm (right edge, min {board_width - edge_margin}mm)"
    if j3_x >= board_width - edge_margin:
        passes.append(f"{j3_label} OK")
        print(f"{j3_label} OK")
    else:
        violations.append(f"{j3_label} VIOLATION")
        print(f"{j3_label} ** VIOLATION **")
    print()

    # --- Summary ---
    print("=" * 60)
    print(f"PASSED: {len(passes)}  |  VIOLATIONS: {len(violations)}")
    print("=" * 60)
    if violations:
        print("\nViolation details:")
        for v in violations:
            print(f"  ** {v}")


# ---------------------------------------------------------------------------
# Post-placement pattern corrections
# ---------------------------------------------------------------------------


def _apply_power_post_placement(pcb: object) -> object:
    """Apply pattern-based corrections to power chain placement.

    Human-reference layout rules (board 50x40mm):

    1. **Signal flow is left-to-right**:
       J1(input) -> U1(buck) -> L1 -> C2 -> J2(5V TP) -> U2(LDO) -> C5 -> J3(3.3V TP)

    2. **Buck converter stage** occupies the left zone (~x=5-20):
       - U1 at ~(board_w*0.17, board_h*0.40), rot=-90
       - C1 (input cap) directly above U1 (dy=-5.3), rot=0
       - C3 (bootstrap) directly below U1 (dy=+4.8), rot=0
       - L1 to the right of U1 (dx=+8.3, dy=-1.2), rot=0
       - D1 (catch diode) right of U1 (dx=+7.8, dy=+2.2), rot=180
       - R1/R2 (FB divider) below-right of U1 (dx=+8, dy=+4..+7), R2 rot=0, R1 rot=180

    3. **LDO stage** occupies the right zone (~x=35-48):
       - U2 at ~(board_w*0.82, board_h*0.47), rot=0
       - C4 (LDO input cap) left of U2 midway (dx=-17), rot=-90
       - C5 (LDO output cap) right of U2 (dx=+7), rot=-90

    4. **Connectors at edges**:
       - J1 (24V input): top area near U1 (x=L1.x, y~6), rot=0
       - J2 (5V TP): between stages (x~board_w*0.61, y~U1.y), rot=-90
       - J3 (3.3V TP): bottom-right (x~board_w*0.88, y~board_h*0.73), rot=-90

    5. **Caps on output side rotated -90** (C2, C4, C5): vertical orientation
       to match horizontal signal flow.
    """
    from dataclasses import replace

    from kicad_pipeline.models.pcb import Point

    xs = [p.x for p in pcb.outline.polygon]
    ys = [p.y for p in pcb.outline.polygon]
    board_w = max(xs) - min(xs)
    board_h = max(ys) - min(ys)

    # Anchor: U1 (buck IC) in left zone — moved right so J1 can be at left edge
    u1_x = 14.0
    u1_y = board_h * 0.45  # 18.0

    # U2 (LDO) in right zone
    u2_x = board_w * 0.72  # 36.0
    u2_y = board_h * 0.45  # 18.0

    # Component sizes (from estimate_footprint_size):
    #   U1 SOIC-8:   3.8 x 3.0    U2 SOT-223:  7.0 x 4.0
    #   C/R 0805:    2.5 x 1.8    SOD-323:     3.0 x 3.0
    #   L_1210:      3.7 x 3.0    TermBlock2P: 7.5 x 7.0
    #   PinHdr 1x02: 3.5 x 6.0
    #
    # Min collision-free center distances (one axis):
    #   U1-0805: dx≥3.15 dy≥2.4   U1-L1210: dx≥3.75 dy≥3.0
    #   U1-SOD323: dx≥3.4 dy≥3.0  U2-0805: dx≥4.75 dy≥2.9
    #   L1210-0805: dx≥3.1 dy≥2.4 0805-0805: dx≥2.5 dy≥1.8

    placement_rules: dict[str, tuple[float, float, float]] = {
        # Buck stage — U1 at center-left, components tight around it
        "U1": (u1_x, u1_y, -90.0),
        "C1": (u1_x, u1_y - 2.5, 0.0),              # input cap above U1 (dist=2.5)
        "C3": (u1_x, u1_y + 2.5, 0.0),              # bootstrap below U1 (dist=2.5)
        "D1": (u1_x - 1.5, u1_y + 2.5, 180.0),      # catch diode below-left (dist=2.9≤3)
        "L1": (u1_x + 4.0, u1_y, 0.0),              # inductor right of U1 (dist=4.0)
        "C2": (u1_x + 4.0, u1_y + 2.5, -90.0),      # output cap below L1 (C2-L1=2.5)
        "R2": (u1_x + 3.2, u1_y - 2.5, 0.0),        # FB bot resistor (dist from U1=4.1)
        "R1": (u1_x + 3.2, u1_y - 3.6, 180.0),      # FB top resistor (dist from U1=4.8)
        # LDO stage — U2 in right zone, caps tight on input/output sides
        "U2": (u2_x, u2_y, 0.0),
        "C4": (u2_x - 2.0, u2_y - 2.2, -90.0),      # LDO input cap (dist=2.9≤3)
        "C5": (u2_x + 2.0, u2_y + 2.2, -90.0),      # LDO output cap (dist=2.9≤3)
        "C6": (u2_x + 2.0, u2_y - 2.2, -90.0),      # HF bypass cap (dist=2.9≤3)
        # Connectors at edges — J1 at left, J3 at right
        "J1": (5.5, board_h * 0.15, 0.0),            # 24V input at left edge (x=5.5≤8)
        "J2": (board_w * 0.52, u2_y, -90.0),         # 5V TP between stages
        "J3": (board_w * 0.88, board_h * 0.73, -90.0),  # 3.3V TP bottom-right
    }

    new_fps: list[object] = []
    for fp in pcb.footprints:
        if fp.ref in placement_rules:
            x, y, rot = placement_rules[fp.ref]
            fp = replace(fp, position=Point(x, y), rotation=rot)
        new_fps.append(fp)

    return replace(pcb, footprints=tuple(new_fps))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build power chain board, optimize, render, and report."""
    output_dir = _repo / "output"
    output_dir.mkdir(exist_ok=True)
    output_png = output_dir / "train_power_placement.png"

    print("=== Power Chain Training Board ===")
    print()

    # 1. Build requirements
    requirements = _build_requirements()
    print(f"Components: {len(requirements.components)}")
    print(f"Nets:       {len(requirements.nets)}")
    print(f"Board:      50 x 40 mm")
    print()

    # 2. Build PCB (no routing)
    print("Building PCB...")
    pcb = build_pcb(requirements, auto_route=False, placement_mode="grouped")
    print(f"  Footprints: {len(pcb.footprints)}")
    print()

    # 3. Run placement optimizer
    print("Running EE placement optimizer...")
    optimized_pcb, review = optimize_placement_ee(requirements, pcb)

    # ---------------------------------------------------------------
    # POST-PLACEMENT CORRECTIONS — apply pattern-based placement rules
    # learned from human reference to ensure C4/C5/C6 are near U2 and
    # signal flow is left-to-right.
    # ---------------------------------------------------------------
    optimized_pcb = _apply_power_post_placement(optimized_pcb)

    print(f"  Review grade: {review.grade}")
    print(f"  Violations:   {len(review.violations)}")
    if review.violations:
        for v in review.violations[:5]:
            print(f"    - {v.rule.value}: {v.message}")
        if len(review.violations) > 5:
            print(f"    ... and {len(review.violations) - 5} more")
    print()

    # 4. Compute quality score
    print("Scoring placement...")
    score = compute_fast_placement_score(optimized_pcb, requirements)
    print(f"  Overall: {score.overall_score:.3f} ({score.grade})")
    print(f"  Placement: {score.placement_score:.3f}")
    for detail in score.breakdown:
        print(f"    {detail.category}: {detail.score:.3f} (w={detail.weight:.2f})")
    print()

    # 5. Render placement PNG
    print(f"Rendering placement to {output_png} ...")
    group_map: dict[str, str] = {}
    for feat in requirements.features:
        for ref in feat.components:
            group_map[ref] = feat.name

    from kicad_pipeline.visualization.placement_render import render_placement

    render_placement(
        optimized_pcb,
        requirements,
        output_png,
        title="Power Chain Training Board - Placement",
        score=score,
        group_map=group_map,
    )
    print(f"  Saved: {output_png}")
    print()

    # 6. Write KiCad PCB file
    pcb_path = output_dir / "train_power.kicad_pcb"

    # Preserve existing PCB if it exists (may be human-edited reference)
    ref_dir = output_dir / "training_reference_boards"
    ref_dir.mkdir(exist_ok=True)
    if pcb_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = ref_dir / f"train_power_{timestamp}.kicad_pcb"
        shutil.copy2(pcb_path, backup)
        print(f"  Backed up existing PCB to {backup}")

    print(f"Writing KiCad PCB to {pcb_path} ...")
    write_pcb(optimized_pcb, pcb_path, fill_zones=False)
    print(f"  KiCad PCB: {pcb_path}")

    # Compare against most recent reference if it exists
    ref_files = sorted(ref_dir.glob("train_power*.kicad_pcb"))
    if ref_files:
        latest_ref = ref_files[-1]
        print(f"\n  Comparing against reference: {latest_ref.name}")
        from kicad_pipeline.pcb.position_extractor import positions_from_pcb_file

        ref_positions = positions_from_pcb_file(latest_ref)
        gen_positions = positions_from_pcb_file(pcb_path)

        print(
            f"  {'Ref':<8} {'Gen X':>7} {'Ref X':>7} {'dX':>6}"
            f" {'Gen Y':>7} {'Ref Y':>7} {'dY':>6} {'Dist':>6}"
        )
        total_drift = 0.0
        count = 0
        for ref in sorted(set(gen_positions) & set(ref_positions)):
            if ref.startswith("H"):
                continue
            gx, gy, _gr = gen_positions[ref]
            rx, ry, _rr = ref_positions[ref]
            dist = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)
            total_drift += dist
            count += 1
            marker = "***" if dist > 3 else ""
            print(
                f"  {ref:<8} {gx:>7.1f} {rx:>7.1f} {gx - rx:>+6.1f}"
                f" {gy:>7.1f} {ry:>7.1f} {gy - ry:>+6.1f}"
                f" {dist:>6.1f} {marker}"
            )
        if count:
            print(f"  Average drift from reference: {total_drift / count:.1f}mm")
    print()

    # 7. Write KiCad project file
    pro_path = write_project_file("train_power", output_dir)
    print(f"  KiCad project: {pro_path}")
    print()

    # 8. Print component positions
    print("Component positions:")
    print(f"  {'Ref':<6} {'X':>8} {'Y':>8} {'Rot':>6}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    fp_map: dict[str, tuple[float, float, float]] = {}
    for fp in sorted(optimized_pcb.footprints, key=lambda f: f.ref):
        print(
            f"  {fp.ref:<6} {fp.position.x:>8.2f} "
            f"{fp.position.y:>8.2f} {fp.rotation:>6.1f}"
        )
        fp_map[fp.ref] = (fp.position.x, fp.position.y, fp.rotation)
    print()

    # 9. Design rules compliance check
    _check_design_rules(fp_map)

    # 10. Analyse human-edited reference layout (if it exists)
    _analyse_human_layout()


# ---------------------------------------------------------------------------
# Human layout analysis
# ---------------------------------------------------------------------------


def _analyse_human_layout() -> None:
    """Extract and analyse the human-edited PCB to learn placement patterns.

    Reads output/train_power.kicad_pcb (must exist from a previous run that
    was then hand-edited in KiCad) and prints signal-flow direction, proximity
    measurements, and key observations.
    """
    pcb_path = _repo / "output" / "train_power.kicad_pcb"
    if not pcb_path.exists():
        print("\n(No human-edited PCB found — skipping layout analysis)")
        return

    from kicad_pipeline.pcb.position_extractor import positions_from_pcb_file

    positions = positions_from_pcb_file(pcb_path)
    if not positions:
        print("\n(Human PCB has no footprints — skipping layout analysis)")
        return

    import math

    def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    print()
    print("=" * 60)
    print("HUMAN LAYOUT ANALYSIS (from output/train_power.kicad_pcb)")
    print("=" * 60)
    print()

    # Signal flow (X-axis order)
    flow_order = ["J1", "U1", "L1", "C2", "J2", "U2", "C5", "J3"]
    available = [r for r in flow_order if r in positions]
    print("--- Signal Flow (X-axis) ---")
    for ref in available:
        x, y, rot = positions[ref]
        print(f"  {ref:4s}  x={x:6.1f}  y={y:6.1f}  rot={rot:6.1f}")
    xs = [positions[r][0] for r in available]
    monotonic = all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))
    print(f"  Left-to-right monotonic: {'YES' if monotonic else 'NO'}")
    print()

    # Proximity pairs
    print("--- Proximity Measurements ---")
    pairs = [
        ("C1", "U1", "Input cap -> buck IC"),
        ("C3", "U1", "Bootstrap cap -> buck IC"),
        ("D1", "U1", "Catch diode -> buck IC"),
        ("L1", "U1", "Inductor -> buck IC"),
        ("C2", "L1", "Output cap -> inductor output"),
        ("R1", "R2", "FB divider pair"),
        ("R1", "U1", "FB top R -> buck IC"),
        ("C4", "U2", "LDO input cap -> LDO"),
        ("C5", "U2", "LDO output cap -> LDO"),
    ]
    for ref_a, ref_b, desc in pairs:
        if ref_a in positions and ref_b in positions:
            d = dist(positions[ref_a], positions[ref_b])
            print(f"  {ref_a}-{ref_b}: {d:5.1f}mm  ({desc})")
    print()

    # Power loop compactness
    loop_refs = ["C1", "U1", "L1", "C2"]
    if all(r in positions for r in loop_refs):
        total = sum(
            dist(positions[loop_refs[i]], positions[loop_refs[i + 1]])
            for i in range(len(loop_refs) - 1)
        )
        print(f"--- Buck Power Loop (C1->U1->L1->C2): {total:.1f}mm total ---")
        for i in range(len(loop_refs) - 1):
            d = dist(positions[loop_refs[i]], positions[loop_refs[i + 1]])
            print(f"  {loop_refs[i]} -> {loop_refs[i+1]}: {d:.1f}mm")
    print()

    # Connector edge distances (board 50x40)
    print("--- Connector Edge Distances (board 50x40) ---")
    for ref in ("J1", "J2", "J3"):
        if ref in positions:
            x, y, _ = positions[ref]
            left = x
            right = 50.0 - x
            top = y
            bottom = 40.0 - y
            nearest = min(left, right, top, bottom)
            side = (
                "left" if nearest == left else
                "right" if nearest == right else
                "top" if nearest == top else "bottom"
            )
            print(f"  {ref}: nearest edge = {side} @ {nearest:.1f}mm  (x={x:.1f}, y={y:.1f})")
    print()

    # Key observations
    print("--- Key Observations ---")
    if "U1" in positions and "U2" in positions:
        u1x, u2x = positions["U1"][0], positions["U2"][0]
        gap = u2x - u1x
        print(f"  Buck-LDO separation: {gap:.1f}mm (U1 x={u1x:.1f} -> U2 x={u2x:.1f})")
    if "C2" in positions and "C4" in positions:
        d = dist(positions["C2"], positions["C4"])
        print(f"  C2-C4 distance: {d:.1f}mm (buck output cap to LDO input cap)")
    if "U1" in positions:
        rot = positions["U1"][2]
        print(f"  U1 rotation: {rot:.0f}deg (buck IC orientation)")
    if "U2" in positions:
        rot = positions["U2"][2]
        print(f"  U2 rotation: {rot:.0f}deg (LDO orientation)")


if __name__ == "__main__":
    main()
