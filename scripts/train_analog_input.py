#!/usr/bin/env python3
"""Analog input / ADC channel training board — isolated placement iteration.

Builds a minimal 4-channel analog input board with voltage dividers, TVS
protection, filter capacitors, and an ADS1115 16-bit ADC.  Runs the EE
placement optimizer, renders a PNG, and prints the quality score + component
positions.

Usage::

    python scripts/train_analog_input.py

ADS1115 pinout reference (MSOP-10):
    Pin 1:  ADDR   -> GND (I2C address 0x48)
    Pin 2:  ALERT  -> NC
    Pin 3:  GND    -> Ground
    Pin 4:  AIN0   -> Channel 1 divided/filtered signal
    Pin 5:  AIN1   -> Channel 2 divided/filtered signal
    Pin 6:  AIN2   -> Channel 3 divided/filtered signal
    Pin 7:  AIN3   -> Channel 4 divided/filtered signal
    Pin 8:  VDD    -> +3V3
    Pin 9:  SDA    -> I2C data
    Pin 10: SCL    -> I2C clock

Per-channel signal chain:
    J{N} pin 1 (AIN{N}_RAW) -> R_top -> [AIN{N}_DIV node] -> R_bot/D/C -> U1 AIN
    J{N} pin 2 -> GND
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
# Footprint identifiers
# ---------------------------------------------------------------------------

_MSOP10_FP = "MSOP-10"
_R0805_FP = "R_0805"
_C0805_FP = "C_0805"
_R0402_FP = "R_0402"
_C0402_FP = "C_0402"
_SOD323_FP = "SOD-323"
_SCREW_TERM_2P_FP = "TerminalBlock_5.08mm_2P"
_PIN_HEADER_4P_FP = "PinHeader_1x04"

# ---------------------------------------------------------------------------
# Board and design rule constants
# ---------------------------------------------------------------------------

_BOARD_WIDTH_MM = 65.0
_BOARD_HEIGHT_MM = 40.0

# Design rule thresholds (mm)
_CONNECTOR_ROW_ALIGN_TOL_MM = 1.0
_CONNECTOR_EDGE_MAX_MM = 5.0
_CHANNEL_STRIP_SPREAD_MAX_MM = 12.0
_R_PAIR_ROW_DY_MAX_MM = 2.0
_ADC_DECOUP_MAX_MM = 5.0
_I2C_PULLUP_MAX_MM = 12.0
_CHANNEL_SPACING_DEV_MAX_MM = 2.0
_PROTECTION_CLUSTER_DY_MAX_MM = 2.0
_MCU_HEADER_MIN_X_MM = 50.0


# Post-placement geometry constants (mm)
_ANALOG_REF_J1_X_MM = 10.0
_ANALOG_REF_J4_X_MM = 53.0
_ANALOG_J_Y_MM = 3.5
_ANALOG_STRIP_DY_MM = 8.3
_ANALOG_U1_X_MM = 45.75
_ANALOG_U1_Y_MM = 28.58
_ANALOG_C1_X_MM = 45.61
_ANALOG_C1_Y_MM = 32.0
_ANALOG_R9_X_MM = 49.0
_ANALOG_R9_Y_MM = 22.16
_ANALOG_R10_X_MM = 49.0
_ANALOG_R10_Y_MM = 25.16
_ANALOG_J5_X_MM = 62.00
_ANALOG_J5_Y_MM = 22.50

# ---------------------------------------------------------------------------
# ADS1115 ADC
# ---------------------------------------------------------------------------


def _make_ads1115() -> Component:
    """ADS1115 16-bit 4-channel I2C ADC (MSOP-10).

    Pin mapping:
        1=ADDR (tied GND), 2=ALERT (NC), 3=GND, 4=AIN0, 5=AIN1,
        6=AIN2, 7=AIN3, 8=VDD, 9=SDA, 10=SCL.

    Channel-to-pin mapping (avoids trace crossing when J1-J4 left-to-right):
        MSOP-10 left column (top-to-bottom): pin 5 (AIN1), pin 4 (AIN0)
        MSOP-10 right column (top-to-bottom): pin 6 (AIN2), pin 7 (AIN3)
        J1 (leftmost)  -> pin 5 (AIN1) — top-left, shortest path
        J2 (second)     -> pin 4 (AIN0) — second-from-top-left
        J3 (third)      -> pin 6 (AIN2) — top-right
        J4 (rightmost)  -> pin 7 (AIN3) — second-from-top-right
    """
    return Component(
        ref="U1",
        value="ADS1115",
        footprint=_MSOP10_FP,
        lcsc="C37593",
        description="16-bit 4-channel I2C ADC MSOP-10",
        pins=(
            Pin("1", "ADDR", PinType.INPUT, net="GND"),
            Pin("2", "ALERT", PinType.OUTPUT),  # no connect
            Pin("3", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("4", "AIN0", PinType.INPUT, PinFunction.ADC, net="AIN2_DIV"),
            Pin("5", "AIN1", PinType.INPUT, PinFunction.ADC, net="AIN1_DIV"),
            Pin("6", "AIN2", PinType.INPUT, PinFunction.ADC, net="AIN3_DIV"),
            Pin("7", "AIN3", PinType.INPUT, PinFunction.ADC, net="AIN4_DIV"),
            Pin("8", "VDD", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("9", "SDA", PinType.BIDIRECTIONAL, PinFunction.I2C_SDA, net="SDA"),
            Pin("10", "SCL", PinType.INPUT, PinFunction.I2C_SCL, net="SCL"),
        ),
    )


def _make_adc_decoupling() -> Component:
    """100nF decoupling capacitor for ADS1115 VDD (C1).

    Uses private subnet +3V3_U1_DEC so C1 is placed right next to U1 VDD.
    """
    return Component(
        ref="C1",
        value="100nF",
        footprint=_C0402_FP,
        lcsc="C49678",
        description="100nF ADC decoupling cap 0402",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3_U1_DEC"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


# ---------------------------------------------------------------------------
# Per-channel components
# ---------------------------------------------------------------------------


def _make_divider_top(ch: int) -> Component:
    """Voltage divider top resistor (R1, R3, R5, R7).

    Pad 1 connects to raw input from connector.
    Pad 2 connects to divider midpoint (AIN{ch}_DIV).
    """
    ref_num = 2 * ch - 1  # 1, 3, 5, 7
    return Component(
        ref=f"R{ref_num}",
        value="10K",
        footprint=_R0402_FP,
        lcsc="C17414",
        description=f"10K voltage divider top 0402 — CH{ch}",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net=f"AIN{ch}_RAW"),
            Pin("2", "2", PinType.PASSIVE, net=f"AIN{ch}_DIV"),
        ),
    )


def _make_divider_bot(ch: int) -> Component:
    """Voltage divider bottom resistor (R2, R4, R6, R8).

    Pad 1 connects to divider midpoint (AIN{ch}_DIV).
    Pad 2 connects to GND.
    """
    ref_num = 2 * ch  # 2, 4, 6, 8
    return Component(
        ref=f"R{ref_num}",
        value="10K",
        footprint=_R0402_FP,
        lcsc="C17414",
        description=f"10K voltage divider bottom 0402 — CH{ch}",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net=f"AIN{ch}_DIV"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_tvs_diode(ch: int) -> Component:
    """TVS/Zener protection diode (D1-D4, SOD-323).

    Uses private subnet AIN{ch}_PROT so TVS is placed right next to the
    divider midpoint and ADC input, not pulled by shared AIN{ch}_DIV.
    """
    return Component(
        ref=f"D{ch}",
        value="PESD3V3",
        footprint=_SOD323_FP,
        lcsc="C118739",
        description=f"3.3V TVS protection SOD-323 — CH{ch}",
        pins=(
            Pin("1", "A", PinType.PASSIVE, net=f"AIN{ch}_PROT"),
            Pin("2", "K", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_filter_cap(ch: int) -> Component:
    """Anti-aliasing / noise filter capacitor (C2-C5, 0805).

    Uses private subnet AIN{ch}_PROT so filter cap is placed right next to
    the divider midpoint and ADC input.
    """
    ref_num = ch + 1  # C2, C3, C4, C5
    return Component(
        ref=f"C{ref_num}",
        value="100nF",
        footprint=_C0402_FP,
        lcsc="C49678",
        description=f"100nF anti-aliasing filter 0402 — CH{ch}",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net=f"AIN{ch}_PROT"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_input_connector(ch: int) -> Component:
    """2-pin screw terminal for sensor input (J1-J4).

    Pin 1: signal input (AIN{ch}_RAW).
    Pin 2: GND reference.
    """
    return Component(
        ref=f"J{ch}",
        value="Screw_Terminal_2P",
        footprint=_SCREW_TERM_2P_FP,
        lcsc="C8269",
        description=f"2-pin 5.08mm screw terminal — CH{ch} input",
        pins=(
            Pin("1", "SIG", PinType.PASSIVE, net=f"AIN{ch}_RAW"),
            Pin("2", "GND", PinType.PASSIVE, PinFunction.GND, net="GND"),
        ),
    )


# ---------------------------------------------------------------------------
# I2C pull-ups and MCU header
# ---------------------------------------------------------------------------


def _make_i2c_pullup_sda() -> Component:
    """4.7K SDA pull-up resistor (R9)."""
    return Component(
        ref="R9",
        value="4.7K",
        footprint=_R0402_FP,
        lcsc="C17673",
        description="4.7K I2C SDA pull-up 0402",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3"),
            Pin("2", "2", PinType.PASSIVE, net="SDA"),
        ),
    )


def _make_i2c_pullup_scl() -> Component:
    """4.7K SCL pull-up resistor (R10)."""
    return Component(
        ref="R10",
        value="4.7K",
        footprint=_R0402_FP,
        lcsc="C17673",
        description="4.7K I2C SCL pull-up 0402",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3"),
            Pin("2", "2", PinType.PASSIVE, net="SCL"),
        ),
    )


def _make_mcu_header() -> Component:
    """4-pin header to MCU (J5): SCL, SDA, +3V3, GND.

    Pin order mirrors the physical order of each net's nearest target
    (R10 SCL pullup sits above R9 SDA pullup): zero ratsnest crossings
    at the header, enforced by Gate A's connector-fanout check (human
    finding 2026-06-11).
    """
    return Component(
        ref="J5",
        value="PinHeader_1x04",
        footprint=_PIN_HEADER_4P_FP,
        lcsc=None,  # C2337 is 40-pin! Use parametric 4-pin header
        description="4-pin 2.54mm header — MCU I2C interface",
        pins=(
            Pin("1", "SCL", PinType.INPUT, PinFunction.I2C_SCL, net="SCL"),
            Pin("2", "SDA", PinType.BIDIRECTIONAL, PinFunction.I2C_SDA, net="SDA"),
            Pin("3", "VCC", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("4", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
        ),
    )


# ---------------------------------------------------------------------------
# Net definitions
# ---------------------------------------------------------------------------


def _channel_nets(ch: int) -> tuple[Net, ...]:
    """Build all nets for a single analog input channel.

    Signal chain with subnet architecture:
        J{ch} -> R_top -> [AIN{ch}_DIV] -> R_bot (to GND)
        [AIN{ch}_DIV] feeds into private subnet [AIN{ch}_PROT] containing
        D{ch} (TVS) and C_filt, which then connects to U1 AIN pin.

    This forces the protection/filter components to be placed between
    the divider midpoint and the ADC input pin.
    """
    r_top = 2 * ch - 1  # R1, R3, R5, R7
    r_bot = 2 * ch      # R2, R4, R6, R8
    c_filt = ch + 1     # C2, C3, C4, C5
    # Channel-to-pin mapping avoids trace crossing with L-R connector order:
    # CH1->pin5(AIN1), CH2->pin4(AIN0), CH3->pin6(AIN2), CH4->pin7(AIN3)
    ch_to_pin = {1: "5", 2: "4", 3: "6", 4: "7"}
    ain_pin = ch_to_pin[ch]

    return (
        # Raw input from connector to divider top
        Net(
            name=f"AIN{ch}_RAW",
            connections=(
                NetConnection(f"J{ch}", "1"),
                NetConnection(f"R{r_top}", "1"),
            ),
        ),
        # Divider midpoint: R_top out -> R_bot in (shared divider node)
        Net(
            name=f"AIN{ch}_DIV",
            connections=(
                NetConnection(f"R{r_top}", "2"),
                NetConnection(f"R{r_bot}", "1"),
            ),
        ),
        # Private protection subnet: D{ch} + C_filt between divider and ADC
        Net(
            name=f"AIN{ch}_PROT",
            connections=(
                NetConnection(f"R{r_bot}", "1"),
                NetConnection(f"D{ch}", "1"),
                NetConnection(f"C{c_filt}", "1"),
                NetConnection("U1", ain_pin),
            ),
        ),
    )


def _build_requirements() -> ProjectRequirements:
    """Assemble full analog input board ProjectRequirements.

    Architecture:
        4 identical analog channels: connector -> voltage divider -> TVS + filter -> ADC
        ADS1115 ADC with I2C interface to MCU via pull-ups and 4-pin header.
        Single +3V3 power rail with 100nF decoupling on ADC VDD.
    """
    components: list[Component] = []
    nets: list[Net] = []
    all_refs: list[str] = []
    all_net_names: list[str] = []

    # ADC and decoupling
    adc = _make_ads1115()
    decap = _make_adc_decoupling()
    components.extend([adc, decap])
    all_refs.extend([adc.ref, decap.ref])

    # Per-channel components and nets
    for ch in range(1, 5):
        r_top = _make_divider_top(ch)
        r_bot = _make_divider_bot(ch)
        tvs = _make_tvs_diode(ch)
        filt = _make_filter_cap(ch)
        conn = _make_input_connector(ch)

        components.extend([r_top, r_bot, tvs, filt, conn])
        all_refs.extend([r_top.ref, r_bot.ref, tvs.ref, filt.ref, conn.ref])

        ch_nets = _channel_nets(ch)
        nets.extend(ch_nets)
        all_net_names.extend(n.name for n in ch_nets)

    # I2C pull-ups and MCU header
    r_sda = _make_i2c_pullup_sda()
    r_scl = _make_i2c_pullup_scl()
    mcu_hdr = _make_mcu_header()
    components.extend([r_sda, r_scl, mcu_hdr])
    all_refs.extend([r_sda.ref, r_scl.ref, mcu_hdr.ref])

    # ---------------------------------------------------------------
    # Power net: +3V3 (shared rail — pull-ups, header)
    # ---------------------------------------------------------------
    vcc_conns: list[NetConnection] = [
        NetConnection("R9", "1"),      # SDA pull-up
        NetConnection("R10", "1"),     # SCL pull-up
        NetConnection("J5", "3"),      # MCU header VCC
    ]
    nets.append(Net(name="+3V3", connections=tuple(vcc_conns)))
    all_net_names.append("+3V3")

    # Private ADC decoupling subnet: C1 <-> U1 VDD
    nets.append(Net(
        name="+3V3_U1_DEC",
        connections=(
            NetConnection("U1", "8"),
            NetConnection("C1", "1"),
        ),
    ))
    all_net_names.append("+3V3_U1_DEC")

    # ---------------------------------------------------------------
    # GND net
    # ---------------------------------------------------------------
    gnd_conns: list[NetConnection] = [
        NetConnection("U1", "1"),      # ADDR pin tied to GND
        NetConnection("U1", "3"),      # ADC GND
        NetConnection("C1", "2"),      # decoupling
        NetConnection("J5", "4"),      # MCU header GND
    ]
    for ch in range(1, 5):
        r_bot_num = 2 * ch
        c_filt_num = ch + 1
        gnd_conns.extend([
            NetConnection(f"J{ch}", "2"),       # connector GND
            NetConnection(f"R{r_bot_num}", "2"),  # divider bottom to GND
            NetConnection(f"D{ch}", "2"),        # TVS cathode to GND
            NetConnection(f"C{c_filt_num}", "2"),  # filter cap to GND
        ])
    nets.append(Net(name="GND", connections=tuple(gnd_conns)))
    all_net_names.append("GND")

    # ---------------------------------------------------------------
    # I2C nets: SDA, SCL
    # ---------------------------------------------------------------
    nets.append(Net(
        name="SDA",
        connections=(
            NetConnection("U1", "9"),
            NetConnection("R9", "2"),
            NetConnection("J5", "2"),
        ),
    ))
    nets.append(Net(
        name="SCL",
        connections=(
            NetConnection("U1", "10"),
            NetConnection("R10", "2"),
            NetConnection("J5", "1"),
        ),
    ))
    all_net_names.extend(["SDA", "SCL"])

    # ---------------------------------------------------------------
    # Feature block
    # ---------------------------------------------------------------
    analog_feature = FeatureBlock(
        name="Analog Inputs",
        description=(
            "4-channel analog input with voltage dividers, TVS protection, "
            "anti-aliasing filters, and ADS1115 16-bit ADC with I2C interface"
        ),
        components=tuple(all_refs),
        nets=tuple(all_net_names),
        subcircuits=("adc_channel", "voltage_divider"),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="AnalogInputTraining", revision="v1"),
        features=(analog_feature,),
        components=tuple(components),
        nets=tuple(nets),
        mechanical=MechanicalConstraints(
            board_width_mm=_BOARD_WIDTH_MM, board_height_mm=_BOARD_HEIGHT_MM
        ),
    )


# ---------------------------------------------------------------------------
# Post-placement pattern corrections
# ---------------------------------------------------------------------------


def _place_channel_passive(
    fp: object,
    j_positions: dict[int, tuple[float, float]],
    comp_offsets: dict[str, tuple[float, float, float]],
    sx: float,
    strip_dy: float,
) -> object:
    """Return *fp* repositioned per-channel passive strip, or unchanged if not applicable."""
    from dataclasses import replace

    from kicad_pipeline.models.pcb import Point

    ref = fp.ref  # type: ignore[union-attr]
    ch = _get_analog_channel(ref)
    if ch is None:
        return fp
    comp_type = _get_component_type(ref, ch)
    if not comp_type or comp_type not in comp_offsets:
        return fp
    jx, jy = j_positions[ch]
    avg_dx, slope_dx, rot = comp_offsets[comp_type]
    dx = (avg_dx + slope_dx * (ch - 1)) * sx
    return replace(fp, position=Point(jx + dx, jy + strip_dy), rotation=rot)


def _apply_analog_post_placement(pcb: object) -> object:
    """Apply reference-derived pattern corrections to analog input placement.

    Layout pattern learned from human-routed reference board (60x40mm):

    1. **J1-J4 connectors** at top edge, left-to-right, evenly spaced ~11mm,
       y~6.4mm, rotation=180 (wire entry faces top edge).

    2. **Per-channel passive strip** ~8.3mm below connector, horizontal row:
       Left-to-right order: C_filt, D_tvs, R_bot, R_top.
       Average dx offsets from connector center (interpolated across channels):
         C_filt: dx ~ -4.6mm  (leftmost)
         D_tvs:  dx ~ -1.8mm
         R_bot:  dx ~ +1.6mm
         R_top:  dx ~ +4.1mm  (rightmost)

    3. **U1 (ADS1115)** at bottom-right (~x=45.8, y=28.6), rot=-90.
       C1 decoupling just below U1 (dx~0, dy~+3.3).

    4. **I2C pull-ups** R9/R10 to the right of U1 (dx~+7, dy~-6/-3).
       J5 MCU header at right edge (x~57, y~22.5).

    The approach: place connectors at evenly-spaced top positions, then
    compute each channel's passive positions as offsets from the connector.
    U1/C1/R9/R10/J5 get fixed positions derived from the reference.
    """
    from dataclasses import replace

    from kicad_pipeline.models.pcb import Point

    # Board dimensions
    xs = [p.x for p in pcb.outline.polygon]
    ys = [p.y for p in pcb.outline.polygon]
    x_min = min(xs)
    y_min = min(ys)
    board_w = max(xs) - x_min
    board_h = max(ys) - y_min

    # Scale factors for non-60x40 boards
    sx = board_w / _BOARD_WIDTH_MM
    sy = board_h / _BOARD_HEIGHT_MM

    # --- Connector positions (top edge, evenly spaced) ---
    # Reference: J1=12.35, J2=23.22, J3=34.10, J4=46.35 → avg spacing ~11.3
    # Use y=4.5 to keep connectors within 5mm of top edge
    j_y = y_min + _ANALOG_J_Y_MM * sy
    j_x_start = x_min + _ANALOG_REF_J1_X_MM * sx
    j_x_end = x_min + _ANALOG_REF_J4_X_MM * sx
    j_spacing = (j_x_end - j_x_start) / 3.0
    j_positions = {
        ch: (j_x_start + (ch - 1) * j_spacing, j_y)
        for ch in range(1, 5)
    }

    # --- Per-channel passive offsets from connector (averaged from reference) ---
    # Reference pattern: horizontal row ~8.3mm below connector
    # Left-to-right: C_filt, D_tvs, R_bot, R_top
    # Average offsets (dx, dy) from connector across all 4 channels:
    strip_dy = _ANALOG_STRIP_DY_MM * sy  # vertical drop from connector to passive row

    # Per-component dx offsets from connector center (averaged across channels)
    # and per-channel linear interpolation slopes (components shift right
    # for channels further right, toward U1)
    #
    # Courtyard sizes (from estimate_courtyard_mm, rotation-aware):
    #   R (0805) at 90deg:  2.4mm wide x 4.1mm tall
    #   D (SOD-323) at 0deg: 5.5mm wide x 4.5mm tall
    #   C (0805) at 90deg:  2.4mm wide x 4.4mm tall
    # Minimum center-to-center gaps: C-D = 3.95mm, D-R = 3.95mm, R-R = 2.4mm
    # Added 0.3mm margin beyond minimum to each gap.
    comp_offsets: dict[str, tuple[float, float, float]] = {
        # (avg_dx, per_ch_slope_dx, rotation)
        # avg_dx: base offset from connector center
        # per_ch_slope_dx: additional dx per channel index (0-based)
        "C_filt": (-6.40, 0.75, 90.0),    # leftmost in strip
        "D_tvs":  (-2.15, 0.75, 0.0),     # second from left (4.25mm from C_filt)
        "R_bot":  (+2.10, 0.76, 90.0),    # second from right (4.25mm from D_tvs)
        "R_top":  (+4.80, 0.76, -90.0),   # rightmost in strip (2.7mm from R_bot)
    }

    # --- Fixed component positions (reference-derived, scaled) ---
    u1_x = x_min + _ANALOG_U1_X_MM * sx
    u1_y = y_min + _ANALOG_U1_Y_MM * sy
    c1_x = x_min + _ANALOG_C1_X_MM * sx
    c1_y = y_min + _ANALOG_C1_Y_MM * sy
    r9_x = x_min + _ANALOG_R9_X_MM * sx
    r9_y = y_min + _ANALOG_R9_Y_MM * sy
    r10_x = x_min + _ANALOG_R10_X_MM * sx
    r10_y = y_min + _ANALOG_R10_Y_MM * sy
    j5_x = x_min + _ANALOG_J5_X_MM * sx
    j5_y = y_min + _ANALOG_J5_Y_MM * sy

    new_fps: list[object] = []
    for fp in pcb.footprints:
        ref = fp.ref
        updated = fp

        # --- Connectors J1-J4 ---
        if ref in ("J1", "J2", "J3", "J4"):
            ch = int(ref[1])
            jx, jy = j_positions[ch]
            updated = replace(fp, position=Point(jx, jy), rotation=180.0)

        # --- U1 (ADC) ---
        elif ref == "U1":
            updated = replace(fp, position=Point(u1_x, u1_y), rotation=-90.0)

        # --- C1 (ADC decoupling) ---
        elif ref == "C1":
            updated = replace(fp, position=Point(c1_x, c1_y), rotation=180.0)

        # --- I2C pull-ups ---
        elif ref == "R9":
            updated = replace(fp, position=Point(r9_x, r9_y), rotation=0.0)
        elif ref == "R10":
            updated = replace(fp, position=Point(r10_x, r10_y), rotation=0.0)

        # --- MCU header ---
        elif ref == "J5":
            updated = replace(fp, position=Point(j5_x, j5_y), rotation=0.0)

        # --- Channel passives ---
        else:
            updated = _place_channel_passive(fp, j_positions, comp_offsets, sx, strip_dy)

        new_fps.append(updated)

    return replace(pcb, footprints=tuple(new_fps))


def _get_analog_channel(ref: str) -> int | None:
    """Map a component ref to its analog channel (1-4), or None."""
    # R1,R2 -> CH1; R3,R4 -> CH2; R5,R6 -> CH3; R7,R8 -> CH4
    if ref.startswith("R") and ref[1:].isdigit():
        idx = int(ref[1:])
        if 1 <= idx <= 8:
            return (idx + 1) // 2
    # D1-D4 -> CH1-CH4
    if ref.startswith("D") and ref[1:].isdigit():
        idx = int(ref[1:])
        if 1 <= idx <= 4:
            return idx
    # C2-C5 -> CH1-CH4 (C1 is ADC decoupling)
    if ref.startswith("C") and ref[1:].isdigit():
        idx = int(ref[1:])
        if 2 <= idx <= 5:
            return idx - 1
    # J1-J4 are connectors — don't move them
    return None


def _get_component_type(ref: str, ch: int) -> str | None:
    """Map a component ref to its type in the channel signal chain."""
    if ref.startswith("R") and ref[1:].isdigit():
        idx = int(ref[1:])
        r_top = 2 * ch - 1  # odd: 1, 3, 5, 7
        r_bot = 2 * ch       # even: 2, 4, 6, 8
        if idx == r_top:
            return "R_top"
        elif idx == r_bot:
            return "R_bot"
    elif ref.startswith("D"):
        return "D_tvs"
    elif ref.startswith("C"):
        return "C_filt"
    return None


# ---------------------------------------------------------------------------
# Design rules compliance checker
# ---------------------------------------------------------------------------


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Euclidean distance between two component positions (ignoring rotation)."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


_ANALOG_COMPONENTS: tuple[Component, ...] = ()  # set in main()


_Pos = tuple[float, float, float]
_PosMap = dict[str, _Pos]
_PerChannel = dict[int, _Pos]


def _analog_collect_positions(
    fp_map: _PosMap,
    violations: list[str],
) -> tuple[_PerChannel, _PerChannel, _PerChannel, _PerChannel, _PerChannel]:
    """Collect per-channel component positions; record missing-component violations."""
    j_positions: _PerChannel = {}
    r_top_positions: _PerChannel = {}
    r_bot_positions: _PerChannel = {}
    d_positions: _PerChannel = {}
    c_positions: _PerChannel = {}

    for ch in range(1, 5):
        r_top_ref = f"R{2 * ch - 1}"
        r_bot_ref = f"R{2 * ch}"
        d_ref = f"D{ch}"
        c_ref = f"C{ch + 1}"
        j_ref = f"J{ch}"

        refs = {
            "J": j_ref, "R_top": r_top_ref, "R_bot": r_bot_ref,
            "D": d_ref, "C_filt": c_ref,
        }
        missing = [v for v in refs.values() if v not in fp_map]
        if missing:
            violations.append(f"  CH{ch}: Missing components: {missing}")
            continue

        j_positions[ch] = fp_map[j_ref]
        r_top_positions[ch] = fp_map[r_top_ref]
        r_bot_positions[ch] = fp_map[r_bot_ref]
        d_positions[ch] = fp_map[d_ref]
        c_positions[ch] = fp_map[c_ref]

    return j_positions, r_top_positions, r_bot_positions, d_positions, c_positions


def _analog_check_connector_row_alignment(
    j_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 1: All connectors J1-J4 at same Y (+/-1mm)."""
    print("--- Connector Row Alignment (all J same Y, +/-1mm) ---")
    if len(j_positions) >= 2:
        j_ys = [pos[1] for pos in j_positions.values()]
        j_y_avg = sum(j_ys) / len(j_ys)
        for ch, pos in sorted(j_positions.items()):
            dev = abs(pos[1] - j_y_avg)
            label = f"  J{ch} Y={pos[1]:.1f}mm (avg={j_y_avg:.1f}, dev={dev:.1f}mm)"
            if dev > _CONNECTOR_ROW_ALIGN_TOL_MM:
                violations.append(f"{label} VIOLATION")
                print(f"{label} ** VIOLATION **")
            else:
                passes.append(f"{label} OK")
                print(f"{label} OK")
    print()


def _analog_check_connector_edge_proximity(
    j_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 2: Connectors at board edge (Y < 5mm from top)."""
    print("--- Connector Edge Proximity (Y < 5mm from top edge) ---")
    for ch, pos in sorted(j_positions.items()):
        label = f"  J{ch} Y={pos[1]:.1f}mm"
        if pos[1] > _CONNECTOR_EDGE_MAX_MM:
            violations.append(f"{label} (MAX {_CONNECTOR_EDGE_MAX_MM}mm from top) VIOLATION")
            print(f"{label} (MAX {_CONNECTOR_EDGE_MAX_MM}mm from top) ** VIOLATION **")
        else:
            passes.append(f"{label} (MAX {_CONNECTOR_EDGE_MAX_MM}mm from top) OK")
            print(f"{label} (MAX {_CONNECTOR_EDGE_MAX_MM}mm from top) OK")
    print()


def _analog_check_channel_strip_spread(
    j_positions: _PerChannel,
    r_top_positions: _PerChannel,
    r_bot_positions: _PerChannel,
    d_positions: _PerChannel,
    c_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 3: Channel strip horizontal spread (all passives within 12mm of J)."""
    print("--- Channel Strip Spread (all passives within 12mm dx of J) ---")
    for ch in range(1, 5):
        if ch not in j_positions:
            continue
        j_x = j_positions[ch][0]

        for comp_name, pos_dict in [
            (f"R{2*ch-1}", r_top_positions),
            (f"R{2*ch}", r_bot_positions),
            (f"D{ch}", d_positions),
            (f"C{ch+1}", c_positions),
        ]:
            if ch in pos_dict:
                dx = abs(pos_dict[ch][0] - j_x)
                label = f"  CH{ch} {comp_name}-J{ch} dx={dx:.1f}mm"
                if dx > _CHANNEL_STRIP_SPREAD_MAX_MM:
                    violations.append(f"{label} (MAX {_CHANNEL_STRIP_SPREAD_MAX_MM}mm) VIOLATION")
                    print(f"{label} (MAX {_CHANNEL_STRIP_SPREAD_MAX_MM}mm) ** VIOLATION **")
                else:
                    passes.append(f"{label} (MAX {_CHANNEL_STRIP_SPREAD_MAX_MM}mm) OK")
                    print(f"{label} (MAX {_CHANNEL_STRIP_SPREAD_MAX_MM}mm) OK")
    print()


def _analog_check_r_pair_row_alignment(
    r_top_positions: _PerChannel,
    r_bot_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 4: R_top to R_bot same-row alignment (dy < 2mm — horizontal strip)."""
    print("--- R_top/R_bot Same Row (dy < 2mm, horizontal strip) ---")
    for ch in range(1, 5):
        if ch not in r_top_positions:
            continue
        r_top_ref = f"R{2 * ch - 1}"
        r_bot_ref = f"R{2 * ch}"
        dy = abs(r_bot_positions[ch][1] - r_top_positions[ch][1])
        label = f"  CH{ch} {r_bot_ref}-{r_top_ref} dy={dy:.1f}mm"
        if dy > _R_PAIR_ROW_DY_MAX_MM:
            violations.append(f"{label} (MAX {_R_PAIR_ROW_DY_MAX_MM}mm) VIOLATION")
            print(f"{label} (MAX {_R_PAIR_ROW_DY_MAX_MM}mm) ** VIOLATION **")
        else:
            passes.append(f"{label} (MAX {_R_PAIR_ROW_DY_MAX_MM}mm) OK")
            print(f"{label} (MAX {_R_PAIR_ROW_DY_MAX_MM}mm) OK")
    print()


def _analog_check_adc_decoupling(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 5: ADC decoupling distance (C1 within 5mm of U1)."""
    print("--- ADC Decoupling (C1 within 5mm of U1) ---")
    if "C1" in fp_map and "U1" in fp_map:
        d_c1_u1 = _dist(fp_map["C1"], fp_map["U1"])
        label = f"  C1-U1: {d_c1_u1:.1f}mm"
        if d_c1_u1 > _ADC_DECOUP_MAX_MM:
            violations.append(f"{label} (MAX {_ADC_DECOUP_MAX_MM}mm) VIOLATION")
            print(f"{label} (MAX {_ADC_DECOUP_MAX_MM}mm) ** VIOLATION **")
        else:
            passes.append(f"{label} (MAX {_ADC_DECOUP_MAX_MM}mm) OK")
            print(f"{label} (MAX {_ADC_DECOUP_MAX_MM}mm) OK")
    else:
        missing = [r for r in ("C1", "U1") if r not in fp_map]
        violations.append(f"  ADC decoupling: Missing components: {missing}")
        print(f"  Missing: {missing}")
    print()


def _analog_check_i2c_pullups(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 6: I2C pull-ups near ADC (R9/R10 within 12mm of U1)."""
    print("--- I2C Pull-ups Near ADC (R9/R10 within 12mm of U1) ---")
    for r_ref in ("R9", "R10"):
        if r_ref in fp_map and "U1" in fp_map:
            d_r_u1 = _dist(fp_map[r_ref], fp_map["U1"])
            label = f"  {r_ref}-U1: {d_r_u1:.1f}mm"
            if d_r_u1 > _I2C_PULLUP_MAX_MM:
                violations.append(f"{label} (MAX {_I2C_PULLUP_MAX_MM}mm) VIOLATION")
                print(f"{label} (MAX {_I2C_PULLUP_MAX_MM}mm) ** VIOLATION **")
            else:
                passes.append(f"{label} (MAX {_I2C_PULLUP_MAX_MM}mm) OK")
                print(f"{label} (MAX {_I2C_PULLUP_MAX_MM}mm) OK")
        elif r_ref not in fp_map:
            violations.append(f"  {r_ref}: Missing")
            print(f"  {r_ref}: Missing")
    print()


def _analog_check_channel_spacing(
    j_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 7: Channel spacing uniformity (max 2mm deviation from average)."""
    print("--- Channel Spacing Uniformity (max 2mm deviation) ---")
    if len(j_positions) >= 2:
        j_xs = [j_positions[ch][0] for ch in sorted(j_positions)]
        spacings = [j_xs[i + 1] - j_xs[i] for i in range(len(j_xs) - 1)]
        if spacings:
            avg_spacing = sum(spacings) / len(spacings)
            max_dev = max(abs(s - avg_spacing) for s in spacings)
            label = (
                f"  Spacings: {[f'{s:.1f}' for s in spacings]}, "
                f"avg={avg_spacing:.1f}mm, max_dev={max_dev:.1f}mm"
            )
            if max_dev > _CHANNEL_SPACING_DEV_MAX_MM:
                violations.append(f"{label} VIOLATION")
                print(f"{label} ** VIOLATION **")
            else:
                passes.append(f"{label} OK")
                print(f"{label} OK")
    print()


def _analog_check_protection_cluster_alignment(
    d_positions: _PerChannel,
    c_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 8: Protection cluster alignment (D and C_filt at same Y +/-2mm)."""
    print("--- Protection Cluster Alignment (D/C same Y +/-2mm) ---")
    for ch in range(1, 5):
        if ch not in d_positions or ch not in c_positions:
            continue
        dy = abs(d_positions[ch][1] - c_positions[ch][1])
        label = f"  CH{ch} D{ch}-C{ch+1} dy={dy:.1f}mm"
        if dy > _PROTECTION_CLUSTER_DY_MAX_MM:
            violations.append(f"{label} (MAX +/-{_PROTECTION_CLUSTER_DY_MAX_MM}mm) VIOLATION")
            print(f"{label} (MAX +/-{_PROTECTION_CLUSTER_DY_MAX_MM}mm) ** VIOLATION **")
        else:
            passes.append(f"{label} (MAX +/-{_PROTECTION_CLUSTER_DY_MAX_MM}mm) OK")
            print(f"{label} (MAX +/-{_PROTECTION_CLUSTER_DY_MAX_MM}mm) OK")
    print()


def _analog_check_mcu_header_position(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 9: J5 at right edge (X > 50mm for 60mm board)."""
    print(f"--- MCU Header Position (J5 near right edge, X > {_MCU_HEADER_MIN_X_MM}mm) ---")
    if "J5" in fp_map:
        j5_x = fp_map["J5"][0]
        j5_y = fp_map["J5"][1]
        label = f"  J5 X={j5_x:.1f}mm Y={j5_y:.1f}mm"
        if j5_x < _MCU_HEADER_MIN_X_MM:
            violations.append(
                f"{label} (should be X>{_MCU_HEADER_MIN_X_MM}mm for right edge) VIOLATION"
            )
            print(f"{label} (should be X>{_MCU_HEADER_MIN_X_MM}mm) ** VIOLATION **")
        else:
            passes.append(f"{label} (X>{_MCU_HEADER_MIN_X_MM}mm) OK")
            print(f"{label} (X>{_MCU_HEADER_MIN_X_MM}mm) OK")
    else:
        violations.append("  J5: Missing")
        print("  J5: Missing")
    print()


def _analog_check_screw_terminal_orientation(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 10: Screw terminal orientation (J1-J4 wire entry faces edge, rot=180)."""
    print("--- Screw Terminal Orientation (J1-J4 rot=180 for top edge) ---")
    for ch in range(1, 5):
        j_ref = f"J{ch}"
        if j_ref in fp_map:
            rot = fp_map[j_ref][2]
            label = f"  {j_ref} rot={rot:.0f}"
            # 180 deg = wire entry faces top edge (away from board center)
            if rot != 180.0:
                violations.append(f"{label} (expected 180) VIOLATION")
                print(f"{label} (expected 180) ** VIOLATION **")
            else:
                passes.append(f"{label} OK")
                print(f"{label} OK")
    print()


def _analog_check_channel_ordering(
    j_positions: _PerChannel,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 11: Channel ordering (J1.x < J2.x < J3.x < J4.x)."""
    print("--- Channel Ordering (J1.x < J2.x < J3.x < J4.x) ---")
    if len(j_positions) == 4:
        j_xs = [j_positions[ch][0] for ch in range(1, 5)]
        ordered = all(j_xs[i] < j_xs[i + 1] for i in range(3))
        label = (
            f"  X positions: J1={j_xs[0]:.1f} J2={j_xs[1]:.1f} "
            f"J3={j_xs[2]:.1f} J4={j_xs[3]:.1f}"
        )
        if ordered:
            passes.append(f"{label} OK")
            print(f"{label} OK")
        else:
            violations.append(f"{label} (not left-to-right) VIOLATION")
            print(f"{label} (not left-to-right) ** VIOLATION **")
    else:
        print("  Cannot check — not all J1-J4 present")
    print()


def _analog_check_lcsc_footprints(
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 12: LCSC footprint verification."""
    print("--- LCSC Footprint Verification ---")
    # Known-bad LCSC numbers that pull wrong footprints
    known_bad_lcsc = {
        "C2337": "pulls 40-pin header (not 4-pin)",
    }
    for comp in _ANALOG_COMPONENTS:
        if comp.lcsc and comp.lcsc in known_bad_lcsc:
            msg = f"  {comp.ref} LCSC={comp.lcsc}: {known_bad_lcsc[comp.lcsc]}"
            violations.append(f"{msg} VIOLATION")
            print(f"{msg} ** VIOLATION **")
        elif comp.lcsc:
            passes.append(f"  {comp.ref} LCSC={comp.lcsc} OK")
            print(f"  {comp.ref} LCSC={comp.lcsc} OK")
        else:
            print(f"  {comp.ref} LCSC=None (parametric footprint)")
    print()


def _check_design_rules(fp_map: dict[str, tuple[float, float, float]]) -> None:
    """Check analog input design rules and print compliance report.

    Rules are RELATIVE positioning checks:
    - Channel strip alignment (all components in vertical line per channel)
    - R_top to R_bot proximity (vertical pair)
    - Connector at edge (low Y value)
    - ADC decoupling distance (C1 near U1)
    - Channel spacing uniformity
    - Screw terminal orientation (wire entry faces board edge)
    - Channel ordering (J1.x < J2.x < J3.x < J4.x)
    - LCSC footprint verification
    """
    print("=" * 60)
    print("DESIGN RULES COMPLIANCE CHECK (Analog Input Board)")
    print("=" * 60)
    print()

    violations: list[str] = []
    passes: list[str] = []

    j_positions, r_top_positions, r_bot_positions, d_positions, c_positions = (
        _analog_collect_positions(fp_map, violations)
    )

    _analog_check_connector_row_alignment(j_positions, violations, passes)
    _analog_check_connector_edge_proximity(j_positions, violations, passes)
    _analog_check_channel_strip_spread(
        j_positions, r_top_positions, r_bot_positions, d_positions, c_positions,
        violations, passes,
    )
    _analog_check_r_pair_row_alignment(r_top_positions, r_bot_positions, violations, passes)
    _analog_check_adc_decoupling(fp_map, violations, passes)
    _analog_check_i2c_pullups(fp_map, violations, passes)
    _analog_check_channel_spacing(j_positions, violations, passes)
    _analog_check_protection_cluster_alignment(d_positions, c_positions, violations, passes)
    _analog_check_mcu_header_position(fp_map, violations, passes)
    _analog_check_screw_terminal_orientation(fp_map, violations, passes)
    _analog_check_channel_ordering(j_positions, violations, passes)
    _analog_check_lcsc_footprints(violations, passes)

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
    """Build analog input board, optimize, render, and report."""
    output_dir = _repo / "output" / "train_analog_input"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "train_analog_input_placement.png"

    print("=== Analog Input / ADC Channel Training Board ===")
    print()

    # 1. Build requirements
    requirements = _build_requirements()
    print(f"Components: {len(requirements.components)}")
    print(f"Nets:       {len(requirements.nets)}")
    print("Board:      60 x 40 mm")
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
    print(f"Rendering placement to {output_png} ...")
    group_map = build_group_map(requirements)

    from kicad_pipeline.visualization.placement_render import render_placement

    render_placement(
        optimized_pcb,
        requirements,
        output_png,
        title="Analog Input Training Board — Group Placement",
        score=score,
        group_map=group_map,
    )
    print(f"  Saved: {output_png}")
    print()

    # 6. Write KiCad PCB file and compare against reference
    pcb_path = output_dir / "train_analog_input.kicad_pcb"
    write_and_compare_pcb(optimized_pcb, pcb_path, requirements=requirements)

    # 7. Write KiCad project file
    pro_path = write_project_file("train_analog_input", output_dir)
    print(f"  KiCad project: {pro_path}")
    print()

    # 8. Print component positions
    fp_map = print_component_positions(optimized_pcb)

    # 9. Design rules compliance check
    global _ANALOG_COMPONENTS
    _ANALOG_COMPONENTS = requirements.components
    _check_design_rules(fp_map)


if __name__ == "__main__":
    main()
