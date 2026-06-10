#!/usr/bin/env python3
"""Relay group training board — isolated placement iteration.

Builds a minimal 4-channel relay board, runs the EE placement optimizer,
renders a PNG, and prints the quality score + component positions.

Usage::

    python scripts/train_relay_group.py

Relay pinout reference (KiCad Relay_SPDT_SANYOU_SRD_Series_Form_C footprint):
    Pad 1: COM    (Common contact)
    Pad 2: Coil-  (connects to Q collector / flyback diode anode)
    Pad 3: NO     (Normally Open contact)
    Pad 4: NC     (Normally Closed contact)
    Pad 5: Coil+  (connects to +5V_RELAY)

Power architecture:
    +5V_LOGIC --[L1 ferrite]--> +5V_RELAY --[relay coils]
    GND       --[L2 ferrite]--> GND_RELAY  --[Q emitters, LED cathodes, bulk caps]
    C1_bulk (100uF MLCC 0805) + C2_bulk (10uF ceramic) on relay side.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _train_common import (  # noqa: E402
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
    build_group_map,
    build_pcb,
    compute_fast_placement_score,
    optimize_placement_ee,
    print_component_positions,
    write_and_compare_pcb,
    write_project_file,
)

# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------

_RELAY_FP = "Relay_THT:Relay_SPDT_SANYOU_SRD_Series_Form_C"
_SOT23_FP = "SOT-23"
_SOD323_FP = "SOD-323"
_R0805_FP = "R_0805"
_R0603_FP = "R_0603"
_R0402_FP = "R_0402"
_LED0805_FP = "LED_0805"
_LED0603_FP = "LED_0603"
_SCREW_TERM_FP = "TerminalBlock_5.08mm_3P"
_FERRITE_FP = "L_0805"  # Ferrite bead in 0805 package
_CAP_ELEC_FP = "C_0805"  # 100uF MLCC in 0805 (BUG-R04 fix)
_CAP_0805_FP = "C_0805"  # 10uF ceramic

# ---------------------------------------------------------------------------
# Board and design rule constants
# ---------------------------------------------------------------------------

_BOARD_WIDTH_MM = 100.0
_BOARD_HEIGHT_MM = 60.0

# Post-placement geometry constants (mm)
# Driver column offsets from relay center (dx negative = left column)
_RELAY_DRIVER_LEFT_DX_MM = 4.3     # left column X offset from relay center
_RELAY_D_FLYBACK_DY_MM = 12.0      # flyback diode (D1-D4) below relay
_RELAY_Q_DY_MM = 15.0              # transistor (Q1-Q4) below relay — was 14.1, widened for SOT-23
_RELAY_R_LED_DY_MM = 17.5          # LED resistor (R5-R8) below relay — was 16.8
_RELAY_D_LED_DY_MM = 20.0          # LED indicator (D5-D8) below relay — was 19.1
_RELAY_R_GATE_DX_MM = 4.0          # right column X offset from relay center
_RELAY_R_GATE_DY_MM = 15.4         # gate resistor (R1-R4) below relay
_RELAY_LED_LEFT_DX_MM = _RELAY_DRIVER_LEFT_DX_MM  # backward compat
_RELAY_LEFT_MARGIN_MM = 3.0

# Power isolation cluster — placed clear of CH1 driver column (x≥5.85) and H4 keepout (x≤5.1)
# All offsets are from board_x_min (left edge).
# Layout: L1(10,52) L2(13,52) in a row, C2(16.5,52) beside them, C1(15.5,47.5) above C2.
# L0805 at rot=90: world bbox +/-1.02 (X) x -1.87..+1.94 (Y)
# C_0805 courtyard: +/-1.25 (X) x +/-0.875 (Y)
# C0805 silk: +/-2.06 (X) x +/-1.0 (Y)
# Verified no courtyard/silk overlaps between any pair.
_RELAY_PWR_L1_X_MM = 10.0        # L1 absolute X from board left
_RELAY_PWR_L2_X_MM = 13.0        # L2 absolute X from board left
_RELAY_PWR_C1_X_MM = 15.5        # C1 absolute X from board left
_RELAY_PWR_C2_X_MM = 16.5        # C2 absolute X from board left
_RELAY_PWR_BOTTOM_OFFSET_MM = 3.0  # L1/L2/C2 Y = board_h - offset = 52mm
_RELAY_PWR_C1_DY_MM = 7.5        # C1 Y = board_h - 7.5 = 47.5mm (above the row)


def _make_relay(ch: int) -> Component:
    """SRD-05VDC-SL-C SPDT relay.

    KiCad footprint ``Relay_SPDT_SANYOU_SRD_Series_Form_C`` pad semantics
    (matches physical pad positions on the PCB footprint):
        Pad 1: COM    → RELAY_COM{ch}   (contact, 3mm pad at origin)
        Pad 2: Coil-  → RELAY_COIL{ch}  (coil, 2.5mm pad — Q collector / D anode)
        Pad 3: NO     → RELAY_NO{ch}    (contact, 3mm pad)
        Pad 4: NC     → RELAY_NC{ch}    (contact, 3mm pad)
        Pad 5: Coil+  → +5V_RELAY       (coil, 2.5mm pad)
    """
    return Component(
        ref=f"K{ch}",
        value="SRD-05VDC-SL-C",
        footprint=_RELAY_FP,
        lcsc="C35449",
        description="5V SPDT relay",
        pins=(
            Pin("1", "COM", PinType.PASSIVE, net=f"RELAY_COM{ch}"),
            Pin("2", "COIL-", PinType.PASSIVE, net=f"RELAY_COIL{ch}"),
            Pin("3", "NO", PinType.PASSIVE, net=f"RELAY_NO{ch}"),
            Pin("4", "NC", PinType.PASSIVE, net=f"RELAY_NC{ch}"),
            Pin("5", "COIL+", PinType.PASSIVE, net="+5V_RELAY"),
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
            Pin("3", "E", PinType.PASSIVE, PinFunction.GND, net="GND_RELAY"),
        ),
    )


def _make_flyback_diode(ch: int) -> Component:
    """1N4148 flyback diode across relay coil.

    Anode → RELAY_COIL (K pad 2 / Q collector)
    Cathode → +5V_RELAY (K pad 5)
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
        footprint=_R0402_FP,
        lcsc="C17513",
        description="1K base resistor 0402",
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
        footprint=_LED0603_FP,
        lcsc="C2286",
        description=f"Red LED 0603 — Channel {ch} indicator",
        pins=(
            Pin("1", "A", PinType.PASSIVE, net=f"LED{ch}_A"),
            Pin("2", "K", PinType.PASSIVE, net="GND_RELAY"),
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
        footprint=_R0603_FP,
        lcsc="C23138",
        description=f"330R LED resistor 0603 — Channel {ch} indicator",
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
            Pin("1", "NO", PinType.PASSIVE, net=f"RELAY_NO{ch}"),
            Pin("2", "COM", PinType.PASSIVE, net=f"RELAY_COM{ch}"),
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

    Pin 1 = GND (source/logic side), Pin 2 = GND_RELAY (relay side).
    BUG-R05: L2 for GND isolation was missing.
    """
    return Component(
        ref="L2",
        value="600R@100MHz",
        footprint=_FERRITE_FP,
        lcsc="C1015",
        description="Ferrite bead 0805 — relay GND isolation",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="GND"),
            Pin("2", "2", PinType.PASSIVE, net="GND_RELAY"),
        ),
    )


def _make_bulk_cap_elec() -> Component:
    """100uF electrolytic bulk decoupling on relay +5V_RELAY rail."""
    return Component(
        ref="C1",
        value="100uF",
        footprint=_CAP_ELEC_FP,
        lcsc=None,  # Force parametric 0805; C65221 is electrolytic (wrong package)
        description="100uF MLCC 0805 — relay bulk decoupling",
        pins=(
            Pin("1", "+", PinType.PASSIVE, net="+5V_RELAY"),
            Pin("2", "-", PinType.PASSIVE, net="GND_RELAY"),
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
            Pin("2", "2", PinType.PASSIVE, net="GND_RELAY"),
        ),
    )


# ---------------------------------------------------------------------------
# Net definitions
# ---------------------------------------------------------------------------


def _channel_nets(ch: int) -> tuple[Net, ...]:
    """Build all nets for a single relay channel.

    SOT-23 BJT pin convention: pin 1=B, pin 2=C, pin 3=E.

    Net connectivity (KiCad footprint pad numbering):
        RELAY_COIL: Q collector (pin 2) → K pad 2 (coil-) → D_flyback anode (pin 1)
                    → R_LED pad 1 (LED indicator taps off collector node)
        +5V_RELAY:  K pad 5 (coil+), D_flyback cathode (pin 2)
        RELAY_COM:  K pad 1 → J pin 1
        RELAY_NO:   K pad 3 → J pin 2
        RELAY_NC:   K pad 4 → J pin 3
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
                NetConnection(f"K{ch}", "2"),
                NetConnection(f"D{ch}", "1"),
                NetConnection(f"R{ch + 4}", "1"),
            ),
        ),
        Net(
            name=f"RELAY_COM{ch}",
            connections=(
                NetConnection(f"K{ch}", "1"),
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
                NetConnection(f"K{ch}", "4"),
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
        GND       → L2 (ferrite) → GND_RELAY → Q emitters, LED cathodes, bulk caps
        C1 (100uF MLCC) + C2 (10uF ceramic) on +5V_RELAY/GND_RELAY for bulk decoupling.

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
        relay_5v_conns.append(NetConnection(f"K{ch}", "5"))
        relay_5v_conns.append(NetConnection(f"D{ch}", "2"))

    # GND: source/logic side — connects to L2 input
    gnd_conns: list[NetConnection] = [
        NetConnection("L2", "1"),
    ]

    # GND_RELAY: relay side — L2 output, transistor emitters, LED cathodes, bulk caps
    gnd_relay_conns: list[NetConnection] = [
        NetConnection("L2", "2"),
        NetConnection("C1", "2"),
        NetConnection("C2", "2"),
    ]
    for ch in range(1, 5):
        gnd_relay_conns.append(NetConnection(f"Q{ch}", "3"))
        gnd_relay_conns.append(NetConnection(f"D{ch + 4}", "2"))

    nets.append(Net(name="+5V_LOGIC", connections=tuple(logic_5v_conns)))
    nets.append(Net(name="+5V_RELAY", connections=tuple(relay_5v_conns)))
    nets.append(Net(name="GND", connections=tuple(gnd_conns)))
    nets.append(Net(name="GND_RELAY", connections=tuple(gnd_relay_conns)))
    all_net_names.extend(["+5V_LOGIC", "+5V_RELAY", "GND", "GND_RELAY"])

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
        mechanical=MechanicalConstraints(
            board_width_mm=_BOARD_WIDTH_MM, board_height_mm=_BOARD_HEIGHT_MM
        ),
    )


# ---------------------------------------------------------------------------
# Post-placement pattern corrections
# ---------------------------------------------------------------------------


def _apply_relay_post_placement(pcb: object) -> object:
    """Apply pattern-based corrections to relay placement.

    Rules (relative to each relay K[ch]):

    LEFT column (dx = -4.3mm from relay centre):
        D_flyback (D1-D4):  dy=+11.5, rot=0    — already placed by optimizer
        Q (Q1-Q4):          dy=+14.1, rot=180   — already placed by optimizer
        R_LED (R5-R8):      dy=+16.8, rot=0     — FIXED here
        D_LED (D5-D8):      dy=+19.1, rot=180   — FIXED here

    RIGHT column (dx = +4.0mm):
        R_gate (R1-R4):     dy=+15.4, rot=180   — already placed by optimizer

    Power isolation cluster (clear of H4 keepout and CH1 driver column):
        L1: (board_left+10.0, board_h-3.0), rot=90
        L2: (board_left+13.0, board_h-3.0), rot=90
        C1: (board_left+15.5, board_h-7.5), rot=180   [above the L/C2 row]
        C2: (board_left+16.5, board_h-3.0), rot=0
    """
    from dataclasses import replace

    from kicad_pipeline.models.pcb import Footprint, Point

    fp_map: dict[str, Footprint] = {fp.ref: fp for fp in pcb.footprints}

    # Compute board dimensions from outline polygon
    xs = [p.x for p in pcb.outline.polygon]
    ys = [p.y for p in pcb.outline.polygon]
    board_h = max(ys) - min(ys)
    board_y_min = min(ys)
    board_x_min = min(xs)
    # Use a small left margin
    board_x_min + _RELAY_LEFT_MARGIN_MM

    new_fps: list[Footprint] = []
    for fp in pcb.footprints:
        ref = fp.ref
        updated = fp

        # --- Pattern 1: ALL driver components per relay channel ---
        # Each channel has: D_flyback, Q_transistor, R_gate, R_LED, D_LED
        # All positioned relative to relay K[ch] center with collision-free spacing

        # D1-D4 flyback diodes — left column
        if ref.startswith("D") and ref[1:].isdigit():
            idx = int(ref[1:])
            if 1 <= idx <= 4:
                k_ref = f"K{idx}"
                if k_ref in fp_map:
                    k_fp = fp_map[k_ref]
                    new_x = k_fp.position.x - _RELAY_DRIVER_LEFT_DX_MM
                    new_y = k_fp.position.y + _RELAY_D_FLYBACK_DY_MM
                    updated = replace(fp, position=Point(new_x, new_y), rotation=0.0)

        # D5-D8 LED indicators — left column, below R_LED
        if ref.startswith("D") and ref[1:].isdigit():
            idx = int(ref[1:])
            if 5 <= idx <= 8:
                ch = idx - 4
                k_ref = f"K{ch}"
                if k_ref in fp_map:
                    k_fp = fp_map[k_ref]
                    new_x = k_fp.position.x - _RELAY_DRIVER_LEFT_DX_MM
                    new_y = k_fp.position.y + _RELAY_D_LED_DY_MM
                    updated = replace(fp, position=Point(new_x, new_y), rotation=0.0)

        # Q1-Q4 transistors — left column, below flyback diode
        if ref.startswith("Q") and ref[1:].isdigit():
            idx = int(ref[1:])
            if 1 <= idx <= 4:
                k_ref = f"K{idx}"
                if k_ref in fp_map:
                    k_fp = fp_map[k_ref]
                    new_x = k_fp.position.x - _RELAY_DRIVER_LEFT_DX_MM
                    new_y = k_fp.position.y + _RELAY_Q_DY_MM
                    updated = replace(fp, position=Point(new_x, new_y), rotation=180.0)

        # R1-R4 gate resistors — right column
        if ref.startswith("R") and ref[1:].isdigit():
            idx = int(ref[1:])
            if 1 <= idx <= 4:
                k_ref = f"K{idx}"
                if k_ref in fp_map:
                    k_fp = fp_map[k_ref]
                    new_x = k_fp.position.x + _RELAY_R_GATE_DX_MM
                    new_y = k_fp.position.y + _RELAY_R_GATE_DY_MM
                    updated = replace(fp, position=Point(new_x, new_y), rotation=180.0)

        # R5-R8 LED resistors — left column, below Q
        if ref.startswith("R") and ref[1:].isdigit():
            idx = int(ref[1:])
            if 5 <= idx <= 8:
                ch = idx - 4
                k_ref = f"K{ch}"
                if k_ref in fp_map:
                    k_fp = fp_map[k_ref]
                    new_x = k_fp.position.x - _RELAY_DRIVER_LEFT_DX_MM
                    new_y = k_fp.position.y + _RELAY_R_LED_DY_MM
                    updated = replace(fp, position=Point(new_x, new_y), rotation=0.0)

        # --- Pattern 2: Power isolation cluster ---
        # Placed clear of H4 mounting hole (keepout to x≈5.1) and CH1 driver column (left
        # edge ≈5.85mm).  L1/L2 rot=90 in a horizontal row at y=52; C1 above the row at
        # y=47.5 to stay clear of C2's wide silk footprint; C2 beside L2 in the row.
        # All X offsets are absolute from board_x_min (not from bl_x).
        row_y = board_y_min + board_h - _RELAY_PWR_BOTTOM_OFFSET_MM
        if ref == "L1":
            updated = replace(
                fp,
                position=Point(board_x_min + _RELAY_PWR_L1_X_MM, row_y),
                rotation=90.0,
            )
        elif ref == "L2":
            updated = replace(
                fp,
                position=Point(board_x_min + _RELAY_PWR_L2_X_MM, row_y),
                rotation=90.0,
            )
        elif ref == "C1":
            updated = replace(
                fp,
                position=Point(
                    board_x_min + _RELAY_PWR_C1_X_MM,
                    board_y_min + board_h - _RELAY_PWR_C1_DY_MM,
                ),
                rotation=180.0,
            )
        elif ref == "C2":
            updated = replace(
                fp,
                position=Point(board_x_min + _RELAY_PWR_C2_X_MM, row_y),
                rotation=0.0,
            )

        new_fps.append(updated)

    return replace(pcb, footprints=tuple(new_fps))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build relay board, optimize, render, and report."""
    output_dir = _repo / "output" / "train_relay"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "train_relay_placement.png"

    print("=== Relay Group Training Board ===")
    print()

    # 1. Build requirements
    requirements = _build_requirements()
    print(f"Components: {len(requirements.components)}")
    print(f"Nets:       {len(requirements.nets)}")
    print("Board:      80 x 50 mm")
    print()

    # 1b. Save requirements.json for review agents and sync checking
    from kicad_pipeline.requirements.decomposer import save_requirements
    save_requirements(requirements, output_dir / "requirements.json")

    # 2. Build PCB (no routing)
    print("Building PCB...")
    pcb = build_pcb(requirements, auto_route=False, placement_mode="grouped")
    print(f"  Footprints: {len(pcb.footprints)}")
    print()

    # 3. Run placement optimizer
    print("Running EE placement optimizer...")
    optimized_pcb, review = optimize_placement_ee(requirements, pcb)

    # NOTE: Post-placement overrides removed — the framework optimizer handles
    # all placement. Training scripts must NOT override the optimizer
    # (see feedback: "post-placement scripts are framework bugs").
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
    group_map = build_group_map(requirements)

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

    # 6. Write KiCad PCB file and compare against reference
    pcb_path = output_dir / "train_relay.kicad_pcb"
    write_and_compare_pcb(optimized_pcb, pcb_path, requirements=requirements)

    # 7. Write KiCad project file
    pro_path = write_project_file("train_relay", output_dir)
    print(f"  KiCad project: {pro_path}")
    print()

    # 8. Print component positions
    fp_map = print_component_positions(optimized_pcb)

    # 9. Design rules compliance check
    _check_design_rules(fp_map)


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Euclidean distance between two component positions (ignoring rotation)."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


_Pos = tuple[float, float, float]
_PosMap = dict[str, _Pos]
_PerChannel = dict[int, _Pos]


def _relay_collect_positions(
    fp_map: _PosMap,
    violations: list[str],
) -> tuple[
    _PerChannel, _PerChannel, _PerChannel, _PerChannel,
    _PerChannel, _PerChannel, _PerChannel,
]:
    """Collect per-channel component positions; record missing-component violations."""
    j_positions: _PerChannel = {}
    k_positions: _PerChannel = {}
    q_positions: _PerChannel = {}
    d_positions: _PerChannel = {}
    r_gate_positions: _PerChannel = {}
    r_led_positions: _PerChannel = {}
    d_led_positions: _PerChannel = {}

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
        r_gate_positions[ch] = fp_map[refs["R"]]
        r_led_positions[ch] = fp_map[refs["R_LED"]]
        d_led_positions[ch] = fp_map[refs["D_LED"]]

    return (
        j_positions, k_positions, q_positions, d_positions,
        r_gate_positions, r_led_positions, d_led_positions,
    )


def _relay_check_terminal_row_alignment(
    j_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 1: All J at same Y (+/-1mm)."""
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


def _relay_check_relay_row_alignment(
    k_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 2: All K at same Y (+/-1mm)."""
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


def _relay_check_jk_x_alignment(
    j_positions: _PerChannel,
    k_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 3: J-K X alignment (+/-2mm per channel)."""
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


def _relay_check_spacing_uniformity(
    k_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 4: Equal relay spacing (max 2mm deviation from average)."""
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


def _relay_check_flyback_x_alignment(
    q_positions: _PerChannel,
    d_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 5: LEFT column — D_flyback and Q same X (dx < 1mm)."""
    print("--- LEFT Column: D_flyback-Q X Alignment (dx < 1mm) ---")
    for ch in range(1, 5):
        if ch not in q_positions or ch not in d_positions:
            continue
        dx = abs(q_positions[ch][0] - d_positions[ch][0])
        label = f"  CH{ch} Q{ch}-D{ch} dx={dx:.1f}mm"
        if dx > 1.0:
            violations.append(f"{label} (MAX 1mm) VIOLATION")
            print(f"{label} (MAX 1mm) ** VIOLATION **")
        else:
            passes.append(f"{label} (MAX 1mm) OK")
            print(f"{label} (MAX 1mm) OK")
    print()


def _relay_check_flyback_above_q(
    q_positions: _PerChannel,
    d_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 6: D_flyback above Q (D.y < Q.y in KiCad coords)."""
    print("--- LEFT Column: D_flyback Above Q (D.y < Q.y) ---")
    for ch in range(1, 5):
        if ch not in q_positions or ch not in d_positions:
            continue
        d_y = d_positions[ch][1]
        q_y = q_positions[ch][1]
        label = f"  CH{ch} D{ch}.y={d_y:.1f} Q{ch}.y={q_y:.1f}"
        if d_y >= q_y:
            violations.append(f"{label} (D must be above Q) VIOLATION")
            print(f"{label} (D must be above Q) ** VIOLATION **")
        else:
            passes.append(f"{label} OK")
            print(f"{label} OK")
    print()


def _relay_check_rgate_opposite_side(
    q_positions: _PerChannel,
    k_positions: _PerChannel,
    r_gate_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 7: R_gate on opposite side from Q (RIGHT column)."""
    print("--- R_gate Opposite Side from Q (R.x > K.x, Q.x < K.x) ---")
    for ch in range(1, 5):
        if ch not in q_positions or ch not in k_positions or ch not in r_gate_positions:
            continue
        k_x = k_positions[ch][0]
        q_x = q_positions[ch][0]
        r_x = r_gate_positions[ch][0]
        q_side = "left" if q_x < k_x else "right"
        r_side = "right" if r_x > k_x else "left"
        label = (
            f"  CH{ch} Q{ch} {q_side} (x={q_x:.1f}), "
            f"R{ch} {r_side} (x={r_x:.1f}), K{ch} x={k_x:.1f}"
        )
        if q_side == r_side:
            violations.append(f"{label} (must be opposite sides) VIOLATION")
            print(f"{label} (must be opposite sides) ** VIOLATION **")
        else:
            passes.append(f"{label} OK")
            print(f"{label} OK")
    print()


def _relay_check_led_pair_x_alignment(
    q_positions: _PerChannel,
    r_led_positions: _PerChannel,
    d_led_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 8: LED pair same X as Q (LEFT column, dx < 1mm)."""
    print("--- LEFT Column: LED Pair X Alignment with Q (dx < 1mm) ---")
    for ch in range(1, 5):
        if ch not in q_positions or ch not in r_led_positions or ch not in d_led_positions:
            continue
        q_x = q_positions[ch][0]
        r_led_dx = abs(r_led_positions[ch][0] - q_x)
        d_led_dx = abs(d_led_positions[ch][0] - q_x)
        label_r = f"  CH{ch} R{ch+4}-Q{ch} dx={r_led_dx:.1f}mm"
        label_d = f"  CH{ch} D{ch+4}-Q{ch} dx={d_led_dx:.1f}mm"
        if r_led_dx > 1.0:
            violations.append(f"{label_r} (MAX 1mm) VIOLATION")
            print(f"{label_r} (MAX 1mm) ** VIOLATION **")
        else:
            passes.append(f"{label_r} (MAX 1mm) OK")
            print(f"{label_r} (MAX 1mm) OK")
        if d_led_dx > 1.0:
            violations.append(f"{label_d} (MAX 1mm) VIOLATION")
            print(f"{label_d} (MAX 1mm) ** VIOLATION **")
        else:
            passes.append(f"{label_d} (MAX 1mm) OK")
            print(f"{label_d} (MAX 1mm) OK")
    print()


def _relay_check_vertical_chain(
    q_positions: _PerChannel,
    r_led_positions: _PerChannel,
    d_led_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 9: Vertical signal chain — R_LED below Q, D_LED below R_LED."""
    print("--- LEFT Column: Vertical Chain (Q -> R_LED -> D_LED, Y increasing) ---")
    for ch in range(1, 5):
        if ch not in q_positions or ch not in r_led_positions or ch not in d_led_positions:
            continue
        q_y = q_positions[ch][1]
        r_led_y = r_led_positions[ch][1]
        d_led_y = d_led_positions[ch][1]
        label = f"  CH{ch} Q{ch}.y={q_y:.1f} R{ch+4}.y={r_led_y:.1f} D{ch+4}.y={d_led_y:.1f}"
        if r_led_y <= q_y:
            violations.append(f"{label} (R_LED must be below Q) VIOLATION")
            print(f"{label} (R_LED must be below Q) ** VIOLATION **")
        elif d_led_y <= r_led_y:
            violations.append(f"{label} (D_LED must be below R_LED) VIOLATION")
            print(f"{label} (D_LED must be below R_LED) ** VIOLATION **")
        else:
            passes.append(f"{label} OK")
            print(f"{label} OK")
    print()


def _relay_check_power_isolation(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 10: Power isolation — L1 near C1/C2 (<8mm)."""
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


def _check_design_rules(fp_map: dict[str, tuple[float, float, float]]) -> None:
    """Check relay driver design rules and print compliance report.

    Two-column pad-connectivity-driven layout rules:
    - All J at same Y (+/-1mm)
    - All K at same Y (+/-1mm)
    - J-K X alignment (+/-2mm per channel)
    - Equal relay spacing (max 2mm deviation from average)
    - D_flyback and Q same X (LEFT column, dx < 1mm)
    - D_flyback above Q (D.y < Q.y)
    - R_gate on opposite side from Q (RIGHT column)
    - R_LED and D_LED same X as Q (LEFT column, dx < 1mm)
    - R_LED below Q, D_LED below R_LED (vertical signal chain)
    - Power isolation: L1 near C1/C2 (<8mm)
    """
    print("=" * 60)
    print("DESIGN RULES COMPLIANCE CHECK (Two-Column Pad-Connectivity)")
    print("=" * 60)
    print()

    violations: list[str] = []
    passes: list[str] = []

    (
        j_positions, k_positions, q_positions, d_positions,
        r_gate_positions, r_led_positions, d_led_positions,
    ) = _relay_collect_positions(fp_map, violations)

    _relay_check_terminal_row_alignment(j_positions, violations, passes)
    _relay_check_relay_row_alignment(k_positions, violations, passes)
    _relay_check_jk_x_alignment(j_positions, k_positions, violations, passes)
    _relay_check_spacing_uniformity(k_positions, violations, passes)
    _relay_check_flyback_x_alignment(q_positions, d_positions, violations, passes)
    _relay_check_flyback_above_q(q_positions, d_positions, violations, passes)
    _relay_check_rgate_opposite_side(
        q_positions, k_positions, r_gate_positions, violations, passes
    )
    _relay_check_led_pair_x_alignment(
        q_positions, r_led_positions, d_led_positions, violations, passes
    )
    _relay_check_vertical_chain(q_positions, r_led_positions, d_led_positions, violations, passes)
    _relay_check_power_isolation(fp_map, violations, passes)

    print("=" * 60)
    print(f"PASSED: {len(passes)}  |  VIOLATIONS: {len(violations)}")
    print("=" * 60)
    if violations:
        print("\nViolation details:")
        for v in violations:
            print(f"  ** {v}")


if __name__ == "__main__":
    main()
