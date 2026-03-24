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

import sys
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
            Pin("1", "BOOT", PinType.INPUT, net="BST"),
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
            Pin("2", "2", PinType.PASSIVE, net="+5V"),
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
    """C2: 22uF output capacitor for buck converter, 0805."""
    return Component(
        ref="C2",
        value="22uF",
        footprint=_C0805_FP,
        lcsc="C159842",
        description="22uF 10V ceramic output cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+5V"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_bootstrap_cap() -> Component:
    """C3: 100nF bootstrap capacitor, 0805."""
    return Component(
        ref="C3",
        value="100nF",
        footprint=_C0805_FP,
        lcsc="C49678",
        description="100nF bootstrap cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="BST"),
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
            Pin("2", "VOUT", PinType.POWER_OUT, PinFunction.VCC, net="+3V3"),
            Pin("3", "VIN", PinType.POWER_IN, PinFunction.VCC, net="+5V"),
            Pin("4", "VOUT_TAB", PinType.POWER_OUT, PinFunction.VCC, net="+3V3"),
        ),
    )


def _make_ldo_input_cap() -> Component:
    """C4: 10uF input capacitor for LDO, 0805."""
    return Component(
        ref="C4",
        value="10uF",
        footprint=_C0805_FP,
        lcsc="C15850",
        description="10uF LDO input cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+5V"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_ldo_output_cap() -> Component:
    """C5: 22uF output capacitor for LDO, 0805."""
    return Component(
        ref="C5",
        value="22uF",
        footprint=_C0805_FP,
        lcsc="C159842",
        description="22uF LDO output cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_input_connector() -> Component:
    """J1: 2-pin screw terminal for 24V power input."""
    return Component(
        ref="J1",
        value="Screw_Terminal_2P",
        footprint=_SCREW_TERM_2P_FP,
        lcsc="C8269",
        description="2-pin 5.08mm screw terminal — 24V input",
        pins=(
            Pin("1", "+24V", PinType.POWER_IN, PinFunction.VCC, net="+24V"),
            Pin("2", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
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
    """J3: 2-pin header for 3.3V output test point."""
    return Component(
        ref="J3",
        value="Pin_Header_2P",
        footprint=_HEADER_2P_FP,
        lcsc="C124375",
        description="2-pin 2.54mm header — 3.3V test point",
        pins=(
            Pin("1", "+3V3", PinType.PASSIVE, PinFunction.VCC, net="+3V3"),
            Pin("2", "GND", PinType.PASSIVE, PinFunction.GND, net="GND"),
        ),
    )


# ---------------------------------------------------------------------------
# Net definitions
# ---------------------------------------------------------------------------


def _build_nets() -> tuple[Net, ...]:
    """Build all nets for the power chain.

    Net topology:
        +24V: J1.1 -> C1.1 -> U1.VIN -> U1.EN
        GND:  J1.2, C1.2, U1.GND, U1.PAD, D1.A, R2.2, C2.2, C3... (throughout)
        SW:   U1.PH -> L1.1 -> C3.2 -> D1.K
        BST:  U1.BOOT -> C3.1
        +5V:  L1.2 -> C2.1 -> R1.1 -> U2.VIN -> C4.1 -> J2.1
        FB:   R1.2 -> R2.1 -> U1.VSNS
        +3V3: U2.VOUT -> U2.VOUT_TAB -> C5.1 -> J3.1
    """
    return (
        Net(
            name="+24V",
            connections=(
                NetConnection("J1", "1"),
                NetConnection("C1", "1"),
                NetConnection("U1", "2"),
                NetConnection("U1", "3"),  # EN tied to VIN
            ),
        ),
        Net(
            name="GND",
            connections=(
                NetConnection("J1", "2"),
                NetConnection("C1", "2"),
                NetConnection("U1", "6"),
                NetConnection("U1", "8"),  # exposed pad
                NetConnection("D1", "1"),  # catch diode anode
                NetConnection("R2", "2"),  # FB bottom to GND
                NetConnection("C2", "2"),
                NetConnection("U2", "1"),
                NetConnection("C4", "2"),
                NetConnection("C5", "2"),
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
            name="BST",
            connections=(
                NetConnection("U1", "1"),  # BOOT pin
                NetConnection("C3", "1"),  # bootstrap cap high side
            ),
        ),
        Net(
            name="+5V",
            connections=(
                NetConnection("L1", "2"),
                NetConnection("C2", "1"),
                NetConnection("R1", "1"),  # FB top to +5V
                NetConnection("U2", "3"),  # LDO VIN
                NetConnection("C4", "1"),
                NetConnection("J2", "1"),
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
        Net(
            name="+3V3",
            connections=(
                NetConnection("U2", "2"),  # VOUT
                NetConnection("U2", "4"),  # VOUT tab
                NetConnection("C5", "1"),
                NetConnection("J3", "1"),
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
    print(f"Writing KiCad PCB to {pcb_path} ...")
    write_pcb(optimized_pcb, pcb_path, fill_zones=False)
    print(f"  KiCad PCB: {pcb_path}")
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


if __name__ == "__main__":
    main()
