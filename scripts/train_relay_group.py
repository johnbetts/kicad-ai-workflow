#!/usr/bin/env python3
"""Relay group training board — isolated placement iteration.

Builds a minimal 4-channel relay board, runs the EE placement optimizer,
renders a PNG, and prints the quality score + component positions.

Usage::

    python scripts/train_relay_group.py

Relay pinout reference (Songle SRD-05VDC-SL-C, standard SPDT):
    Pin 1: Coil+  (connects to +5V_RELAY)
    Pin 2: NC     (Normally Closed contact)
    Pin 3: NO     (Normally Open contact)
    Pin 4: Coil-  (connects to Q collector / flyback diode anode)
    Pin 5: COM    (Common contact)

Power architecture:
    +5V_LOGIC --[L1 ferrite]--> +5V_RELAY --[relay coils]
    GND is shared (single-point star ground recommended).
    C1_bulk (100uF electrolytic) + C2_bulk (10uF ceramic) on relay side.

DFM note — creepage isolation:
    The relay footprint should include an Edge.Cuts semicircle slot between
    the mains-side pins (COM=5, NO=3, NC=2) and the coil-side pins (1, 4).
    This provides creepage isolation between mains and logic domains.
    Implementation is at the footprint level (not handled by this script).
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
# Component definitions
# ---------------------------------------------------------------------------

_RELAY_FP = "Relay_THT:Relay_SPDT_SANYOU_SRD_Series_Form_C"
_SOT23_FP = "SOT-23"
_SOD323_FP = "SOD-323"
_R0805_FP = "R_0805"
_LED0805_FP = "LED_0805"
_SCREW_TERM_FP = "TerminalBlock_5.08mm_3P"
_FERRITE_FP = "R_0805"  # Ferrite bead in 0805 package
_CAP_ELEC_FP = "C_0805"  # 100uF MLCC in 0805 (BUG-R04 fix)
_CAP_0805_FP = "C_0805"  # 10uF ceramic


def _make_relay(ch: int) -> Component:
    """SRD-05VDC-SL-C SPDT relay.

    Songle SRD-05VDC-SL-C pinout:
        Pin 1: Coil+  → +5V_RELAY
        Pin 2: NC     → RELAY_NC{ch}
        Pin 3: NO     → RELAY_NO{ch}
        Pin 4: Coil-  → RELAY_COIL{ch} (Q collector / D anode)
        Pin 5: COM    → RELAY_COM{ch}
    """
    return Component(
        ref=f"K{ch}",
        value="SRD-05VDC-SL-C",
        footprint=_RELAY_FP,
        lcsc="C35449",
        description="5V SPDT relay",
        pins=(
            Pin("1", "COIL+", PinType.PASSIVE, net="+5V_RELAY"),
            Pin("2", "NC", PinType.PASSIVE, net=f"RELAY_NC{ch}"),
            Pin("3", "NO", PinType.PASSIVE, net=f"RELAY_NO{ch}"),
            Pin("4", "COIL-", PinType.PASSIVE, net=f"RELAY_COIL{ch}"),
            Pin("5", "COM", PinType.PASSIVE, net=f"RELAY_COM{ch}"),
        ),
    )


def _make_transistor(ch: int) -> Component:
    """SS8050 NPN transistor (relay driver).

    SOT-23 BJT pinout (codebase convention):
        Pin 1: Base
        Pin 2: Collector
        Pin 3: Emitter
    """
    return Component(
        ref=f"Q{ch}",
        value="SS8050",
        footprint=_SOT23_FP,
        lcsc="C727114",
        description="NPN transistor SOT-23",
        pins=(
            Pin("1", "B", PinType.INPUT, net=f"RELAY_DRIVE{ch}"),
            Pin("2", "C", PinType.OUTPUT, net=f"RELAY_COIL{ch}"),
            Pin("3", "E", PinType.PASSIVE, PinFunction.GND, net="GND"),
        ),
    )


def _make_flyback_diode(ch: int) -> Component:
    """1N4148 flyback diode across relay coil.

    Anode → RELAY_COIL (K pin 4 / Q collector)
    Cathode → +5V_RELAY (K pin 1)
    """
    return Component(
        ref=f"D{ch}",
        value="1N4148",
        footprint=_SOD323_FP,
        lcsc="C81598",
        description="Flyback diode SOD-323",
        pins=(
            Pin("1", "A", PinType.PASSIVE, net=f"RELAY_COIL{ch}"),
            Pin("2", "K", PinType.PASSIVE, net="+5V_RELAY"),
        ),
    )


def _make_base_resistor(ch: int) -> Component:
    """1K base resistor for transistor Q{ch}."""
    return Component(
        ref=f"R{ch}",
        value="1K",
        footprint=_R0805_FP,
        lcsc="C17513",
        description="1K base resistor 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net=f"GPIO{ch}"),
            Pin("2", "2", PinType.PASSIVE, net=f"RELAY_DRIVE{ch}"),
        ),
    )


def _make_led(ch: int) -> Component:
    """LED indicator for relay channel {ch} state.

    D5-D8 are relay state indicator LEDs:
        D5 + R5 → Channel 1 (connected to Q1 collector / RELAY_COIL1 net)
        D6 + R6 → Channel 2 (connected to Q2 collector / RELAY_COIL2 net)
        D7 + R7 → Channel 3 (connected to Q3 collector / RELAY_COIL3 net)
        D8 + R8 → Channel 4 (connected to Q4 collector / RELAY_COIL4 net)

    Each LED pair should be placed in the same channel column as its relay.
    """
    idx = ch + 4  # D5-D8
    return Component(
        ref=f"D{idx}",
        value="LED",
        footprint=_LED0805_FP,
        lcsc="C2286",
        description=f"Red LED 0805 — Channel {ch} indicator",
        pins=(
            Pin("1", "A", PinType.PASSIVE, net=f"LED{ch}_A"),
            Pin("2", "K", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_led_resistor(ch: int) -> Component:
    """330R current-limit resistor for LED D{ch+4} (Channel {ch} indicator).

    Pad 1 connects to RELAY_COIL{ch} (same net as Q collector),
    pad 2 connects to D_LED anode.  The LED lights when Q is driving.
    """
    idx = ch + 4  # R5-R8
    return Component(
        ref=f"R{idx}",
        value="330R",
        footprint=_R0805_FP,
        lcsc="C23138",
        description=f"330R LED resistor 0805 — Channel {ch} indicator",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net=f"RELAY_COIL{ch}"),
            Pin("2", "2", PinType.PASSIVE, net=f"LED{ch}_A"),
        ),
    )


def _make_screw_terminal(ch: int) -> Component:
    """3-pin screw terminal block for relay channel {ch} output.

    Terminal-to-relay alignment: J{ch} must be placed directly above K{ch}
    (same X position +/-2mm) because their COM/NO/NC nets are connected.
        J{ch} pin 1 (COM) ↔ K{ch} pin 5 (COM)
        J{ch} pin 2 (NO)  ↔ K{ch} pin 3 (NO)
        J{ch} pin 3 (NC)  ↔ K{ch} pin 2 (NC)
    """
    return Component(
        ref=f"J{ch}",
        value="Screw_Terminal_3P",
        footprint=_SCREW_TERM_FP,
        lcsc="C8465",
        description=f"3-pin 5.08mm screw terminal — Channel {ch}",
        pins=(
            Pin("1", "COM", PinType.PASSIVE, net=f"RELAY_COM{ch}"),
            Pin("2", "NO", PinType.PASSIVE, net=f"RELAY_NO{ch}"),
            Pin("3", "NC", PinType.PASSIVE, net=f"RELAY_NC{ch}"),
        ),
    )


# ---------------------------------------------------------------------------
# Power isolation components
# ---------------------------------------------------------------------------


def _make_ferrite_bead() -> Component:
    """Ferrite bead on +5V rail — isolates relay switching noise from logic.

    Placed at the boundary between +5V_LOGIC and +5V_RELAY domains.
    """
    return Component(
        ref="L1",
        value="600R@100MHz",
        footprint=_FERRITE_FP,
        lcsc="C1015",
        description="Ferrite bead 0805 — relay power isolation",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+5V_LOGIC"),
            Pin("2", "2", PinType.PASSIVE, net="+5V_RELAY"),
        ),
    )


def _make_gnd_ferrite_bead() -> Component:
    """Ferrite bead on GND rail — isolates relay return current from logic GND.

    BUG-R05: L2 for GND isolation was missing.
    """
    return Component(
        ref="L2",
        value="600R@100MHz",
        footprint=_FERRITE_FP,
        lcsc="C1015",
        description="Ferrite bead 0805 — relay GND isolation",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="GND_LOGIC"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_bulk_cap_elec() -> Component:
    """100uF electrolytic bulk decoupling on relay +5V_RELAY rail."""
    return Component(
        ref="C1",
        value="100uF",
        footprint=_CAP_ELEC_FP,
        lcsc="C65221",
        description="100uF electrolytic — relay bulk decoupling",
        pins=(
            Pin("1", "+", PinType.PASSIVE, net="+5V_RELAY"),
            Pin("2", "-", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_bulk_cap_ceramic() -> Component:
    """10uF ceramic decoupling on relay +5V_RELAY rail."""
    return Component(
        ref="C2",
        value="10uF",
        footprint=_CAP_0805_FP,
        lcsc="C15850",
        description="10uF ceramic — relay high-freq decoupling",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+5V_RELAY"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


# ---------------------------------------------------------------------------
# Net definitions
# ---------------------------------------------------------------------------


def _channel_nets(ch: int) -> tuple[Net, ...]:
    """Build all nets for a single relay channel.

    SOT-23 BJT pin convention: pin 1=B, pin 2=C, pin 3=E.

    Net connectivity for Songle SRD-05VDC-SL-C:
        RELAY_COIL: Q collector (pin 2) → K pin 4 (coil-) → D_flyback anode (pin 1)
                    → R_LED pad 1 (LED indicator taps off collector node)
        +5V_RELAY:  K pin 1 (coil+), D_flyback cathode (pin 2)
        RELAY_COM:  K pin 5 → J pin 1
        RELAY_NO:   K pin 3 → J pin 2
        RELAY_NC:   K pin 2 → J pin 3
        LED_A:      R_LED pad 2 → D_LED anode (pin 1)
    """
    return (
        Net(
            name=f"GPIO{ch}",
            connections=(NetConnection(f"R{ch}", "1"),),
        ),
        Net(
            name=f"RELAY_DRIVE{ch}",
            connections=(
                NetConnection(f"R{ch}", "2"),
                NetConnection(f"Q{ch}", "1"),
            ),
        ),
        Net(
            name=f"RELAY_COIL{ch}",
            connections=(
                NetConnection(f"Q{ch}", "2"),
                NetConnection(f"K{ch}", "4"),
                NetConnection(f"D{ch}", "1"),
                NetConnection(f"R{ch + 4}", "1"),
            ),
        ),
        Net(
            name=f"RELAY_COM{ch}",
            connections=(
                NetConnection(f"K{ch}", "5"),
                NetConnection(f"J{ch}", "1"),
            ),
        ),
        Net(
            name=f"RELAY_NO{ch}",
            connections=(
                NetConnection(f"K{ch}", "3"),
                NetConnection(f"J{ch}", "2"),
            ),
        ),
        Net(
            name=f"RELAY_NC{ch}",
            connections=(
                NetConnection(f"K{ch}", "2"),
                NetConnection(f"J{ch}", "3"),
            ),
        ),
        Net(
            name=f"LED{ch}_A",
            connections=(
                NetConnection(f"R{ch + 4}", "2"),
                NetConnection(f"D{ch + 4}", "1"),
            ),
        ),
    )


def _build_requirements() -> ProjectRequirements:
    """Assemble full relay-only ProjectRequirements.

    Power architecture:
        +5V_LOGIC → L1 (ferrite) → +5V_RELAY → relay coils
        C1 (100uF) + C2 (10uF) on +5V_RELAY for bulk decoupling.

    Channel hierarchy:
        Channel N: K{N} relay, Q{N} transistor, D{N} flyback, R{N} base resistor,
                   D{N+4} indicator LED, R{N+4} LED resistor, J{N} screw terminal.
        J{N} must be placed directly above K{N} (same X, connected by COM/NO/NC nets).
    """
    components: list[Component] = []
    nets: list[Net] = []
    all_refs: list[str] = []
    all_net_names: list[str] = []

    for ch in range(1, 5):
        relay = _make_relay(ch)
        transistor = _make_transistor(ch)
        flyback = _make_flyback_diode(ch)
        base_r = _make_base_resistor(ch)
        led = _make_led(ch)
        led_r = _make_led_resistor(ch)
        terminal = _make_screw_terminal(ch)

        components.extend([relay, transistor, flyback, base_r, led, led_r, terminal])

        ch_nets = _channel_nets(ch)
        nets.extend(ch_nets)

        all_refs.extend([relay.ref, transistor.ref, flyback.ref, base_r.ref,
                         led.ref, led_r.ref, terminal.ref])
        all_net_names.extend(n.name for n in ch_nets)

    # Power isolation components (BUG-R05: added L2 for GND)
    ferrite_5v = _make_ferrite_bead()
    ferrite_gnd = _make_gnd_ferrite_bead()
    bulk_elec = _make_bulk_cap_elec()
    bulk_ceramic = _make_bulk_cap_ceramic()
    components.extend([ferrite_5v, ferrite_gnd, bulk_elec, bulk_ceramic])
    all_refs.extend([ferrite_5v.ref, ferrite_gnd.ref, bulk_elec.ref, bulk_ceramic.ref])

    # Power nets
    # +5V_LOGIC: clean side — connects to L1 input
    logic_5v_conns: list[NetConnection] = [
        NetConnection("L1", "1"),
    ]

    # +5V_RELAY: dirty side — L1 output, relay coils, flyback cathodes, bulk caps
    relay_5v_conns: list[NetConnection] = [
        NetConnection("L1", "2"),
        NetConnection("C1", "1"),
        NetConnection("C2", "1"),
    ]
    for ch in range(1, 5):
        relay_5v_conns.append(NetConnection(f"K{ch}", "1"))
        relay_5v_conns.append(NetConnection(f"D{ch}", "2"))

    # GND_LOGIC: clean side — connects to L2 input
    gnd_logic_conns: list[NetConnection] = [
        NetConnection("L2", "1"),
    ]

    # GND: relay side — L2 output, transistor emitters, LED cathodes, bulk caps
    gnd_conns: list[NetConnection] = [
        NetConnection("L2", "2"),
        NetConnection("C1", "2"),
        NetConnection("C2", "2"),
    ]
    for ch in range(1, 5):
        gnd_conns.append(NetConnection(f"Q{ch}", "3"))
        gnd_conns.append(NetConnection(f"D{ch + 4}", "2"))

    nets.append(Net(name="+5V_LOGIC", connections=tuple(logic_5v_conns)))
    nets.append(Net(name="+5V_RELAY", connections=tuple(relay_5v_conns)))
    nets.append(Net(name="GND_LOGIC", connections=tuple(gnd_logic_conns)))
    nets.append(Net(name="GND", connections=tuple(gnd_conns)))
    all_net_names.extend(["+5V_LOGIC", "+5V_RELAY", "GND_LOGIC", "GND"])

    relay_feature = FeatureBlock(
        name="Relay Outputs",
        description=(
            "4-channel relay output with flyback protection, LED indicators, "
            "and power isolation (ferrite + bulk caps)"
        ),
        components=tuple(all_refs),
        nets=tuple(all_net_names),
        subcircuits=("relay_driver",),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="RelayTraining", revision="v2"),
        features=(relay_feature,),
        components=tuple(components),
        nets=tuple(nets),
        mechanical=MechanicalConstraints(board_width_mm=90, board_height_mm=55),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build relay board, optimize, render, and report."""
    output_dir = _repo / "output"
    output_dir.mkdir(exist_ok=True)
    output_png = output_dir / "train_relay_placement.png"

    print("=== Relay Group Training Board ===")
    print()

    # 1. Build requirements
    requirements = _build_requirements()
    print(f"Components: {len(requirements.components)}")
    print(f"Nets:       {len(requirements.nets)}")
    print(f"Board:      80 x 50 mm")
    print()

    # 2. Build PCB (no routing)
    print("Building PCB...")
    pcb = build_pcb(requirements, auto_route=False, placement_mode="grouped")
    print(f"  Footprints: {len(pcb.footprints)}")
    print()

    # 3. Run placement optimizer
    print("Running EE placement optimizer...")
    optimized_pcb, review = optimize_placement_ee(requirements, pcb)

    # Relay design rules are now encoded in the optimizer itself
    # (phases 3a, 3a2, 3b, 3b2 and late realignment).
    # No post-placement overrides needed.
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
    # NOTE: render_placement() draws labels at the footprint centroid (center of
    # pad-extent bounding box).  See _draw_footprints() in placement_render.py —
    # cx/cy are computed from pad_extent_in_board_space(), and ax.text() is placed
    # at (cx, cy) with ha="center", va="center".
    print(f"Rendering placement to {output_png} ...")
    # Build group_map: ref -> feature block name
    group_map: dict[str, str] = {}
    for feat in requirements.features:
        for ref in feat.components:
            group_map[ref] = feat.name

    from kicad_pipeline.visualization.placement_render import render_placement

    render_placement(
        optimized_pcb,
        requirements,
        output_png,
        title="Relay Training Board — Group Placement",
        score=score,
        group_map=group_map,
    )
    print(f"  Saved: {output_png}")
    print()

    # 6. Write KiCad PCB file
    pcb_path = output_dir / "train_relay.kicad_pcb"
    print(f"Writing KiCad PCB to {pcb_path} ...")
    write_pcb(optimized_pcb, pcb_path, fill_zones=False)
    print(f"  KiCad PCB: {pcb_path}")
    print()

    # 7. Write KiCad project file
    pro_path = write_project_file("train_relay", output_dir)
    print(f"  KiCad project: {pro_path}")
    print()

    # 8. Print component positions
    print("Component positions:")
    print(f"  {'Ref':<6} {'X':>8} {'Y':>8} {'Rot':>6}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    fp_map: dict[str, tuple[float, float, float]] = {}
    for fp in sorted(optimized_pcb.footprints, key=lambda f: f.ref):
        print(f"  {fp.ref:<6} {fp.position.x:>8.2f} {fp.position.y:>8.2f} {fp.rotation:>6.1f}")
        fp_map[fp.ref] = (fp.position.x, fp.position.y, fp.rotation)
    print()

    # 9. Design rules compliance check
    _check_design_rules(fp_map)


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Euclidean distance between two component positions (ignoring rotation)."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _check_design_rules(fp_map: dict[str, tuple[float, float, float]]) -> None:
    """Check relay driver design rules and print compliance report.

    Rules are RELATIVE positioning checks (from human feedback):
    - All J at same Y (+/-1mm)
    - All K at same Y (+/-1mm)
    - J-K X alignment (+/-2mm per channel)
    - Equal relay spacing (max 2mm deviation from average)
    - Q and D_flyback at same Y (+/-2mm per channel)
    - R_gate directly below Q (dx < 2mm)
    - Power isolation: L1 near C1/C2 (<8mm)
    """
    print("=" * 60)
    print("DESIGN RULES COMPLIANCE CHECK (Relative Positioning)")
    print("=" * 60)
    print()

    violations: list[str] = []
    passes: list[str] = []

    # ---------------------------------------------------------------
    # Collect per-channel positions
    # ---------------------------------------------------------------
    j_positions: dict[int, tuple[float, float, float]] = {}
    k_positions: dict[int, tuple[float, float, float]] = {}
    q_positions: dict[int, tuple[float, float, float]] = {}
    d_positions: dict[int, tuple[float, float, float]] = {}  # flyback

    for ch in range(1, 5):
        refs = {
            "J": f"J{ch}", "K": f"K{ch}", "Q": f"Q{ch}", "R": f"R{ch}",
            "D": f"D{ch}", "D_LED": f"D{ch+4}", "R_LED": f"R{ch+4}",
        }
        missing = [v for v in refs.values() if v not in fp_map]
        if missing:
            violations.append(f"  CH{ch}: Missing components: {missing}")
            continue
        j_positions[ch] = fp_map[refs["J"]]
        k_positions[ch] = fp_map[refs["K"]]
        q_positions[ch] = fp_map[refs["Q"]]
        d_positions[ch] = fp_map[refs["D"]]

    # ---------------------------------------------------------------
    # 1. All J at same Y (+/-1mm)
    # ---------------------------------------------------------------
    print("--- Terminal Row Alignment (all J same Y, +/-1mm) ---")
    if len(j_positions) >= 2:
        j_ys = [pos[1] for pos in j_positions.values()]
        j_y_avg = sum(j_ys) / len(j_ys)
        for ch, pos in sorted(j_positions.items()):
            dev = abs(pos[1] - j_y_avg)
            label = f"  J{ch} Y={pos[1]:.1f}mm (avg={j_y_avg:.1f}, dev={dev:.1f}mm)"
            if dev > 1.0:
                violations.append(f"{label} VIOLATION")
                print(f"{label} ** VIOLATION **")
            else:
                passes.append(f"{label} OK")
                print(f"{label} OK")
    print()

    # ---------------------------------------------------------------
    # 2. All K at same Y (+/-1mm)
    # ---------------------------------------------------------------
    print("--- Relay Row Alignment (all K same Y, +/-1mm) ---")
    if len(k_positions) >= 2:
        k_ys = [pos[1] for pos in k_positions.values()]
        k_y_avg = sum(k_ys) / len(k_ys)
        for ch, pos in sorted(k_positions.items()):
            dev = abs(pos[1] - k_y_avg)
            label = f"  K{ch} Y={pos[1]:.1f}mm (avg={k_y_avg:.1f}, dev={dev:.1f}mm)"
            if dev > 1.0:
                violations.append(f"{label} VIOLATION")
                print(f"{label} ** VIOLATION **")
            else:
                passes.append(f"{label} OK")
                print(f"{label} OK")
    print()

    # ---------------------------------------------------------------
    # 3. J-K X alignment (+/-2mm per channel)
    # ---------------------------------------------------------------
    print("--- J-K X Alignment (per channel, +/-2mm) ---")
    for ch in range(1, 5):
        if ch not in j_positions or ch not in k_positions:
            continue
        dx = abs(j_positions[ch][0] - k_positions[ch][0])
        label = f"  CH{ch} J{ch}-K{ch} dx={dx:.1f}mm"
        if dx > 2.0:
            violations.append(f"{label} (MAX +/-2mm) VIOLATION")
            print(f"{label} (MAX +/-2mm) ** VIOLATION **")
        else:
            passes.append(f"{label} (MAX +/-2mm) OK")
            print(f"{label} (MAX +/-2mm) OK")
    print()

    # ---------------------------------------------------------------
    # 4. Equal relay spacing (max 2mm deviation from average)
    # ---------------------------------------------------------------
    print("--- Relay Spacing Uniformity (max 2mm deviation) ---")
    if len(k_positions) >= 2:
        k_xs = [k_positions[ch][0] for ch in sorted(k_positions)]
        spacings = [k_xs[i + 1] - k_xs[i] for i in range(len(k_xs) - 1)]
        if spacings:
            avg_spacing = sum(spacings) / len(spacings)
            max_dev = max(abs(s - avg_spacing) for s in spacings)
            label = (
                f"  Spacings: {[f'{s:.1f}' for s in spacings]}, "
                f"avg={avg_spacing:.1f}mm, max_dev={max_dev:.1f}mm"
            )
            if max_dev > 2.0:
                violations.append(f"{label} VIOLATION")
                print(f"{label} ** VIOLATION **")
            else:
                passes.append(f"{label} OK")
                print(f"{label} OK")
    print()

    # ---------------------------------------------------------------
    # 5. Q and D_flyback at same Y (+/-2mm per channel)
    # ---------------------------------------------------------------
    print("--- Q-D_flyback Y Alignment (per channel, +/-2mm) ---")
    for ch in range(1, 5):
        if ch not in q_positions or ch not in d_positions:
            continue
        dy = abs(q_positions[ch][1] - d_positions[ch][1])
        label = f"  CH{ch} Q{ch}-D{ch} dy={dy:.1f}mm"
        if dy > 2.0:
            violations.append(f"{label} (MAX +/-2mm) VIOLATION")
            print(f"{label} (MAX +/-2mm) ** VIOLATION **")
        else:
            passes.append(f"{label} (MAX +/-2mm) OK")
            print(f"{label} (MAX +/-2mm) OK")
    print()

    # ---------------------------------------------------------------
    # 6. R_gate directly below Q (dx < 2mm)
    # ---------------------------------------------------------------
    print("--- R_gate Below Q (dx < 2mm) ---")
    for ch in range(1, 5):
        r_ref = f"R{ch}"
        q_ref = f"Q{ch}"
        if r_ref not in fp_map or q_ref not in fp_map:
            continue
        r_pos = fp_map[r_ref]
        q_pos = fp_map[q_ref]
        dx = abs(r_pos[0] - q_pos[0])
        label = f"  CH{ch} R{ch}-Q{ch} dx={dx:.1f}mm"
        if dx > 2.0:
            violations.append(f"{label} (MAX 2mm) VIOLATION")
            print(f"{label} (MAX 2mm) ** VIOLATION **")
        else:
            passes.append(f"{label} (MAX 2mm) OK")
            print(f"{label} (MAX 2mm) OK")
    print()

    # ---------------------------------------------------------------
    # 7. LED pair in channel column (+/-3mm from relay X)
    # ---------------------------------------------------------------
    print("--- LED Pair in Channel Column (+/-3mm from K) ---")
    for ch in range(1, 5):
        d_led_ref = f"D{ch+4}"
        r_led_ref = f"R{ch+4}"
        k_ref = f"K{ch}"
        if d_led_ref not in fp_map or k_ref not in fp_map:
            continue
        d_led_dx = abs(fp_map[d_led_ref][0] - fp_map[k_ref][0])
        label = f"  CH{ch} D{ch+4} dx from K{ch}={d_led_dx:.1f}mm"
        if d_led_dx > 3.0:
            violations.append(f"{label} (MAX +/-3mm) VIOLATION")
            print(f"{label} (MAX +/-3mm) ** VIOLATION **")
        else:
            passes.append(f"{label} (MAX +/-3mm) OK")
            print(f"{label} (MAX +/-3mm) OK")
        if r_led_ref in fp_map:
            r_led_dx = abs(fp_map[r_led_ref][0] - fp_map[k_ref][0])
            label = f"  CH{ch} R{ch+4} dx from K{ch}={r_led_dx:.1f}mm"
            if r_led_dx > 3.0:
                violations.append(f"{label} (MAX +/-3mm) VIOLATION")
                print(f"{label} (MAX +/-3mm) ** VIOLATION **")
            else:
                passes.append(f"{label} (MAX +/-3mm) OK")
                print(f"{label} (MAX +/-3mm) OK")
    print()

    # ---------------------------------------------------------------
    # 8. Power isolation checks
    # ---------------------------------------------------------------
    print("--- Power Isolation (L1/L2/C1/C2 proximity < 8mm) ---")
    power_refs = ["L1", "L2", "C1", "C2"]
    power_missing = [r for r in power_refs if r not in fp_map]
    if power_missing:
        violations.append(f"  Power isolation: Missing components: {power_missing}")
        print(f"  Power isolation: Missing components: {power_missing}")
    else:
        l1 = fp_map["L1"]
        c1 = fp_map["C1"]
        c2 = fp_map["C2"]
        for cap_ref, cap_pos in [("C1", c1), ("C2", c2)]:
            d = _dist(l1, cap_pos)
            label = f"  L1-{cap_ref}: {d:.1f}mm"
            if d > 8.0:
                violations.append(f"{label} (MAX 8mm) VIOLATION")
                print(f"{label} (MAX 8mm) ** VIOLATION **")
            else:
                passes.append(f"{label} (MAX 8mm) OK")
                print(f"{label} (MAX 8mm) OK")
    print()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("=" * 60)
    print(f"PASSED: {len(passes)}  |  VIOLATIONS: {len(violations)}")
    print("=" * 60)
    if violations:
        print("\nViolation details:")
        for v in violations:
            print(f"  ** {v}")


if __name__ == "__main__":
    main()
