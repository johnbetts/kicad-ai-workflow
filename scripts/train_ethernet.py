#!/usr/bin/env python3
"""Ethernet subsystem training board — isolated placement iteration.

Builds a minimal W5500 Ethernet board (PHY + magnetics + RJ45), runs the
EE placement optimizer, renders a PNG, and prints quality score + positions.

Usage::

    python scripts/train_ethernet.py

W5500 Ethernet Controller (LQFP-48) — simplified pinout for training:
    Pin  1: GND
    Pin  2: AVDD       (analog VCC — 3.3V)
    Pin  3: EXRES1     (bias resistor — RSVD)
    Pin  4: MOSI       (SPI data in)
    Pin  5: MISO       (SPI data out)
    Pin  6: SCLK       (SPI clock)
    Pin  7: SCSn       (SPI chip select, active low)
    Pin  8: INTn       (interrupt output, active low)
    Pin  9: RSTn       (reset input, active low)
    Pin 10: VCC        (digital 3.3V)
    Pin 11: GND
    Pin 12: LINKLED    (NC for training)
    Pin 13: ACTLED     (NC for training)
    Pin 14: GND
    Pin 15: VCC        (digital 3.3V)
    Pin 16: TXP        (TX+ differential)
    Pin 17: TXN        (TX- differential)
    Pin 18: GND
    Pin 19: RXP        (RX+ differential)
    Pin 20: RXN        (RX- differential)
    Pin 21: GND
    Pin 22: AVDD       (analog VCC)
    Pin 23: XI         (crystal input)
    Pin 24: XO         (crystal output)
    Pin 25-48: GND/VCC/NC (simplified — only wired pins modeled)
    Pin 49: PAD        (exposed pad — GND)

25MHz Crystal (HC49/SMD 2-pin):
    Pin 1: XI  (to U1 pin 23)
    Pin 2: XO  (to U1 pin 24)

RJ45 with integrated magnetics (HR911105A):
    Pin 1: TX+  (center-tap internally biased)
    Pin 2: TX-
    Pin 3: RX+
    Pin 4: NC
    Pin 5: NC
    Pin 6: RX-
    Pin 7: NC
    Pin 8: NC
    Pin 9: LED_G+
    Pin 10: LED_G-
    Pin 11: LED_Y+
    Pin 12: LED_Y-
    Pin 13: SHIELD (GND)
    Pin 14: SHIELD (GND)
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
# Footprint constants
# ---------------------------------------------------------------------------

_LQFP48_FP = "LQFP-48"
_RJ45_FP = "RJ45_HR911105A"
_CRYSTAL_FP = "Crystal_SMD_3215"
_C0805_FP = "C_0805"
_R0805_FP = "R_0805"
_HEADER_6P_FP = "PinHeader_1x06_P2.54mm_Vertical"
_HEADER_2P_FP = "PinHeader_1x02_P2.54mm_Vertical"

# ---------------------------------------------------------------------------
# Board and design rule constants
# ---------------------------------------------------------------------------

_BOARD_WIDTH_MM = 55.0
_BOARD_HEIGHT_MM = 55.0

# Design rule thresholds (mm)
#
# Calibrated for accurate JLCPCB courtyard footprints (not simplified sizes):
#   J1 HR911105A: courtyard 19.48x16.98 mm -- through-hole pads extend 6.9 mm
#     above the footprint origin, so the origin is forced to y>=7.9 mm to keep
#     pads inside the board with 1 mm edge margin.  The "J1 near top edge"
#     rule therefore uses 10 mm (origin Y), not 5 mm.
#   U1 W5500 LQFP-48: courtyard 11x11 mm (half = 5.5 mm).
#   Y1 crystal HC49/SMD 3215: courtyard ~3.6x2.35 mm (half ~1.8 mm).
#   C*/R* 0805: courtyard 4.41x2.35 mm.
#
# Center-to-center distances are used throughout.
_RJ45_EDGE_MAX_MM = 10.0    # J1 origin Y ≤ 10 mm (pads force origin ≥ 7.9 mm)
_U1_J1_MAX_MM = 15.0        # W5500 centroid within 15 mm of RJ45 centroid
_CRYSTAL_TO_U1_MAX_MM = 9.5  # Y1 within 9.5 mm of U1 (courtyard minimum ≈ 7.3 mm)
_XTAL_CAP_TO_CRYSTAL_MAX_MM = 4.5  # Crystal load caps within 4.5 mm of Y1
_VCC_DECOUP_TO_U1_MAX_MM = 10.0   # C1 (VCC decoup) within 10 mm of U1
_BULK_DECOUP_TO_U1_MAX_MM = 15.0  # C2 (bulk decoup) within 15 mm of U1
_AVDD_DECOUP_TO_U1_MAX_MM = 20.0  # C3 (AVDD decoup) within 20 mm of U1
_TX_TERM_TO_U1_MAX_MM = 15.0  # R1/R2 TX termination within 15 mm of U1
_RSVD_TO_U1_MAX_MM = 15.0   # R3 RSVD resistor within 15 mm of U1
_HEADER_BOTTOM_MARGIN_MM = 8.0
_DIFF_PAIR_MAX_MM = 4.0

# Post-placement geometry constants (mm)
_ETH_J1_X_FRAC = 0.50
_ETH_J1_Y_MM = 3.5
_ETH_U1_X_FRAC = 0.50
_ETH_U1_Y_MM = 15.5
_ETH_R_DX_MM = 0.8
_ETH_R_DY_MM = 2.87
_ETH_CRYSTAL_DX_MM = 4.6
_ETH_XTAL_CAP_DY_MM = 1.8
_ETH_C2_DY_MM = 2.5
_ETH_C1C3_DX_MM = 2.85
_ETH_C1_DY_MM = 0.7
_ETH_C3_DY_MM = 0.8
_ETH_R3_DX_MM = 2.85
_ETH_R3_DY_MM = 2.3
_ETH_J2_X_FRAC = 0.30
_ETH_J3_X_FRAC = 0.70
_ETH_HEADER_Y_OFFSET_MM = 3.5

# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------


def _make_w5500() -> Component:
    """W5500 Ethernet controller, LQFP-48.

    Simplified pinout — only pins that participate in nets are wired.
    Remaining pins are GND/VCC/NC stubs.
    """
    return Component(
        ref="U1",
        value="W5500",
        footprint=_LQFP48_FP,
        lcsc="C32843",
        description="W5500 Hardwired TCP/IP Ethernet controller LQFP-48",
        pins=(
            Pin("1", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("2", "AVDD", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("3", "EXRES1", PinType.INPUT, net="RSVD"),
            Pin("4", "MOSI", PinType.INPUT, PinFunction.SPI_MOSI, net="SPI_MOSI"),
            Pin("5", "MISO", PinType.OUTPUT, PinFunction.SPI_MISO, net="SPI_MISO"),
            Pin("6", "SCLK", PinType.INPUT, PinFunction.SPI_CLK, net="SPI_SCK"),
            Pin("7", "SCSn", PinType.INPUT, PinFunction.SPI_CS, net="SPI_CS"),
            Pin("8", "INTn", PinType.OUTPUT, PinFunction.INTERRUPT, net="ETH_INT"),
            Pin("9", "RSTn", PinType.INPUT, PinFunction.RESET, net="ETH_RST"),
            Pin("10", "VCC", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("11", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("12", "LINKLED", PinType.OUTPUT),  # NC
            Pin("13", "ACTLED", PinType.OUTPUT),  # NC
            Pin("14", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("15", "VCC", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("16", "TXP", PinType.OUTPUT, net="TX+"),
            Pin("17", "TXN", PinType.OUTPUT, net="TX-"),
            Pin("18", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("19", "RXP", PinType.INPUT, net="RX+"),
            Pin("20", "RXN", PinType.INPUT, net="RX-"),
            Pin("21", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("22", "AVDD2", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("23", "XI", PinType.INPUT, net="XTAL1"),
            Pin("24", "XO", PinType.OUTPUT, net="XTAL2"),
            # Remaining pins — GND/VCC stubs for footprint completeness
            Pin("25", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("26", "VCC", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("27", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("28", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("29", "VCC", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("30", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("31", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("32", "VCC", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("33", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("34", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("35", "VCC", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("36", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("37", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("38", "VCC", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("39", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("40", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("41", "VCC", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("42", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("43", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("44", "VCC", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("45", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("46", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("47", "VCC", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("48", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("49", "PAD", PinType.POWER_IN, PinFunction.GND, net="GND"),
        ),
    )


def _make_vcc_decoupling() -> Component:
    """C1: 100nF decoupling on VCC, 0805.

    Private subnet +3V3_U1_DEC forces C1 next to U1 VCC pin.
    """
    return Component(
        ref="C1",
        value="100nF",
        footprint=_C0805_FP,
        lcsc="C49678",
        description="100nF VCC decoupling cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3_U1_DEC"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_bulk_decoupling() -> Component:
    """C2: 10uF bulk decoupling, 0805."""
    return Component(
        ref="C2",
        value="10uF",
        footprint=_C0805_FP,
        lcsc="C15850",
        description="10uF bulk decoupling cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_avdd_decoupling() -> Component:
    """C3: 100nF decoupling on AVDD, 0805.

    Private subnet AVDD_U1_DEC forces C3 next to U1 AVDD pin.
    """
    return Component(
        ref="C3",
        value="100nF",
        footprint=_C0805_FP,
        lcsc="C49678",
        description="100nF AVDD decoupling cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="AVDD_U1_DEC"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_crystal() -> Component:
    """Y1: 25MHz crystal oscillator, HC49/SMD."""
    return Component(
        ref="Y1",
        value="25MHz",
        footprint=_CRYSTAL_FP,
        lcsc="C13738",
        description="25MHz crystal oscillator SMD 3215",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="XTAL1"),
            Pin("2", "2", PinType.PASSIVE, net="XTAL2"),
        ),
    )


def _make_crystal_cap_1() -> Component:
    """C4: 22pF load capacitor for crystal, 0805.

    Private subnet XTAL1_C4 forces C4 next to Y1/U1 XI pin.
    """
    return Component(
        ref="C4",
        value="22pF",
        footprint=_C0805_FP,
        lcsc="C1804",
        description="22pF crystal load cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="XTAL1_C4"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_crystal_cap_2() -> Component:
    """C5: 22pF load capacitor for crystal, 0805.

    Private subnet XTAL2_C5 forces C5 next to Y1/U1 XO pin.
    """
    return Component(
        ref="C5",
        value="22pF",
        footprint=_C0805_FP,
        lcsc="C1804",
        description="22pF crystal load cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="XTAL2_C5"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_rj45() -> Component:
    """J1: RJ45 connector with integrated magnetics (HR911105A).

    Standard Ethernet RJ45 pinout:
        Pins 1-8: signal (TX+, TX-, RX+, NC, NC, RX-, NC, NC)
        Pins 9-12: LED anodes/cathodes
        Pins 13-14: shield/GND
    """
    return Component(
        ref="J1",
        value="HR911105A",
        footprint=_RJ45_FP,
        lcsc="C12074",
        description="RJ45 connector with integrated magnetics",
        pins=(
            Pin("1", "TD+", PinType.BIDIRECTIONAL, net="TX+"),
            Pin("2", "TD-", PinType.BIDIRECTIONAL, net="TX-"),
            Pin("3", "RD+", PinType.BIDIRECTIONAL, net="RX+"),
            Pin("4", "NC1", PinType.PASSIVE),
            Pin("5", "NC2", PinType.PASSIVE),
            Pin("6", "RD-", PinType.BIDIRECTIONAL, net="RX-"),
            Pin("7", "NC3", PinType.PASSIVE),
            Pin("8", "NC4", PinType.PASSIVE),
            Pin("9", "LED_G+", PinType.PASSIVE, net="+3V3"),
            Pin("10", "LED_G-", PinType.PASSIVE, net="GND"),
            Pin("11", "LED_Y+", PinType.PASSIVE, net="+3V3"),
            Pin("12", "LED_Y-", PinType.PASSIVE, net="GND"),
            Pin("13", "SHIELD1", PinType.PASSIVE, net="GND"),
            Pin("14", "SHIELD2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_tx_term_plus() -> Component:
    """R1: 49.9 ohm TX+ termination resistor, 0805."""
    return Component(
        ref="R1",
        value="49.9R",
        footprint=_R0805_FP,
        lcsc="C25129",
        description="49.9 ohm TX+ termination resistor 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="TX+"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_tx_term_minus() -> Component:
    """R2: 49.9 ohm TX- termination resistor, 0805."""
    return Component(
        ref="R2",
        value="49.9R",
        footprint=_R0805_FP,
        lcsc="C25129",
        description="49.9 ohm TX- termination resistor 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="TX-"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_rsvd_resistor() -> Component:
    """R3: 12.1K RSVD bias resistor, 0805."""
    return Component(
        ref="R3",
        value="12.1K",
        footprint=_R0805_FP,
        lcsc="C17401",
        description="12.1K RSVD bias resistor 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="RSVD"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_spi_header() -> Component:
    """J2: 6-pin SPI header (MOSI, MISO, SCK, CS, INT, RST)."""
    return Component(
        ref="J2",
        value="SPI_Header",
        footprint=_HEADER_6P_FP,
        lcsc=None,  # Use parametric vertical footprint; LCSC C37208 generates horizontal layout
        description="6-pin 2.54mm header — SPI interface to MCU",
        pins=(
            Pin("1", "MOSI", PinType.BIDIRECTIONAL, PinFunction.SPI_MOSI, net="SPI_MOSI"),
            Pin("2", "MISO", PinType.BIDIRECTIONAL, PinFunction.SPI_MISO, net="SPI_MISO"),
            Pin("3", "SCK", PinType.BIDIRECTIONAL, PinFunction.SPI_CLK, net="SPI_SCK"),
            Pin("4", "CS", PinType.BIDIRECTIONAL, PinFunction.SPI_CS, net="SPI_CS"),
            Pin("5", "INT", PinType.BIDIRECTIONAL, PinFunction.INTERRUPT, net="ETH_INT"),
            Pin("6", "RST", PinType.BIDIRECTIONAL, PinFunction.RESET, net="ETH_RST"),
        ),
    )


def _make_power_header() -> Component:
    """J3: 2-pin power header (3V3 + GND)."""
    return Component(
        ref="J3",
        value="PWR_Header",
        footprint=_HEADER_2P_FP,
        lcsc=None,  # parametric 2-pin header (no basic LCSC for 2-pin straight)
        description="2-pin 2.54mm header — 3V3 + GND power input",
        pins=(
            Pin("1", "+3V3", PinType.POWER_IN, PinFunction.VCC, net="+3V3"),
            Pin("2", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
        ),
    )


# ---------------------------------------------------------------------------
# Net definitions
# ---------------------------------------------------------------------------


def _build_nets() -> tuple[Net, ...]:
    """Build all nets for the Ethernet subsystem.

    Net topology:
        +3V3:      J3.1, C1.1, C2.1, C3.1, U1.AVDD, U1.VCC, U1.AVDD2,
                   U1.VCC(26,29,32,35,38,41,44,47), J1.LED_G+, J1.LED_Y+
        GND:       J3.2, C1.2, C2.2, C3.2, C4.2, C5.2, R1.2, R2.2, R3.2,
                   U1.GND(1,11,14,18,21,25,27,28,30,31,33,34,36,37,39,40,
                   42,43,45,46,48,49), J1.LED_G-, J1.LED_Y-, J1.SHIELD
        SPI_MOSI:  J2.1 -> U1.MOSI(4)
        SPI_MISO:  J2.2 -> U1.MISO(5)
        SPI_SCK:   J2.3 -> U1.SCLK(6)
        SPI_CS:    J2.4 -> U1.SCSn(7)
        ETH_INT:   J2.5 -> U1.INTn(8)
        ETH_RST:   J2.6 -> U1.RSTn(9)
        TX+:       U1.TXP(16) -> R1.1 -> J1.TD+(1)
        TX-:       U1.TXN(17) -> R2.1 -> J1.TD-(2)
        RX+:       J1.RD+(3) -> U1.RXP(19)
        RX-:       J1.RD-(6) -> U1.RXN(20)
        XTAL1:     U1.XI(23) -> Y1.1 -> C4.1
        XTAL2:     U1.XO(24) -> Y1.2 -> C5.1
        RSVD:      U1.EXRES1(3) -> R3.1
    """
    return (
        # Shared +3V3 rail (bulk cap, headers, LEDs, W5500 power stubs)
        Net(
            name="+3V3",
            connections=(
                NetConnection("J3", "1"),
                NetConnection("C2", "1"),       # bulk cap stays on shared rail
                NetConnection("U1", "10"),  # VCC
                NetConnection("U1", "15"),  # VCC
                NetConnection("U1", "26"),
                NetConnection("U1", "29"),
                NetConnection("U1", "32"),
                NetConnection("U1", "35"),
                NetConnection("U1", "38"),
                NetConnection("U1", "41"),
                NetConnection("U1", "44"),
                NetConnection("U1", "47"),
                NetConnection("J1", "9"),   # LED_G+
                NetConnection("J1", "11"),  # LED_Y+
            ),
        ),
        # Private VCC decoupling subnet: C1 <-> U1 VCC
        Net(
            name="+3V3_U1_DEC",
            connections=(
                NetConnection("U1", "2"),   # AVDD
                NetConnection("C1", "1"),
            ),
        ),
        # Private AVDD decoupling subnet: C3 <-> U1 AVDD2
        Net(
            name="AVDD_U1_DEC",
            connections=(
                NetConnection("U1", "22"),  # AVDD2
                NetConnection("C3", "1"),
            ),
        ),
        Net(
            name="GND",
            connections=(
                NetConnection("J3", "2"),
                NetConnection("C1", "2"),
                NetConnection("C2", "2"),
                NetConnection("C3", "2"),
                NetConnection("C4", "2"),
                NetConnection("C5", "2"),
                NetConnection("R1", "2"),
                NetConnection("R2", "2"),
                NetConnection("R3", "2"),
                NetConnection("U1", "1"),
                NetConnection("U1", "11"),
                NetConnection("U1", "14"),
                NetConnection("U1", "18"),
                NetConnection("U1", "21"),
                NetConnection("U1", "25"),
                NetConnection("U1", "27"),
                NetConnection("U1", "28"),
                NetConnection("U1", "30"),
                NetConnection("U1", "31"),
                NetConnection("U1", "33"),
                NetConnection("U1", "34"),
                NetConnection("U1", "36"),
                NetConnection("U1", "37"),
                NetConnection("U1", "39"),
                NetConnection("U1", "40"),
                NetConnection("U1", "42"),
                NetConnection("U1", "43"),
                NetConnection("U1", "45"),
                NetConnection("U1", "46"),
                NetConnection("U1", "48"),
                NetConnection("U1", "49"),  # exposed pad
                NetConnection("J1", "10"),  # LED_G-
                NetConnection("J1", "12"),  # LED_Y-
                NetConnection("J1", "13"),  # SHIELD1
                NetConnection("J1", "14"),  # SHIELD2
            ),
        ),
        Net(
            name="SPI_MOSI",
            connections=(
                NetConnection("J2", "1"),
                NetConnection("U1", "4"),
            ),
        ),
        Net(
            name="SPI_MISO",
            connections=(
                NetConnection("J2", "2"),
                NetConnection("U1", "5"),
            ),
        ),
        Net(
            name="SPI_SCK",
            connections=(
                NetConnection("J2", "3"),
                NetConnection("U1", "6"),
            ),
        ),
        Net(
            name="SPI_CS",
            connections=(
                NetConnection("J2", "4"),
                NetConnection("U1", "7"),
            ),
        ),
        Net(
            name="ETH_INT",
            connections=(
                NetConnection("J2", "5"),
                NetConnection("U1", "8"),
            ),
        ),
        Net(
            name="ETH_RST",
            connections=(
                NetConnection("J2", "6"),
                NetConnection("U1", "9"),
            ),
        ),
        Net(
            name="TX+",
            connections=(
                NetConnection("U1", "16"),
                NetConnection("R1", "1"),
                NetConnection("J1", "1"),
            ),
        ),
        Net(
            name="TX-",
            connections=(
                NetConnection("U1", "17"),
                NetConnection("R2", "1"),
                NetConnection("J1", "2"),
            ),
        ),
        Net(
            name="RX+",
            connections=(
                NetConnection("J1", "3"),
                NetConnection("U1", "19"),
            ),
        ),
        Net(
            name="RX-",
            connections=(
                NetConnection("J1", "6"),
                NetConnection("U1", "20"),
            ),
        ),
        Net(
            name="XTAL1",
            connections=(
                NetConnection("U1", "23"),
                NetConnection("Y1", "1"),
            ),
        ),
        Net(
            name="XTAL2",
            connections=(
                NetConnection("U1", "24"),
                NetConnection("Y1", "2"),
            ),
        ),
        # Private crystal load cap subnets
        Net(
            name="XTAL1_C4",
            connections=(
                NetConnection("Y1", "1"),
                NetConnection("C4", "1"),
            ),
        ),
        Net(
            name="XTAL2_C5",
            connections=(
                NetConnection("Y1", "2"),
                NetConnection("C5", "1"),
            ),
        ),
        Net(
            name="RSVD",
            connections=(
                NetConnection("U1", "3"),
                NetConnection("R3", "1"),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Requirements assembly
# ---------------------------------------------------------------------------


def _build_requirements() -> ProjectRequirements:
    """Assemble Ethernet subsystem ProjectRequirements.

    Single FeatureBlock "Ethernet" containing all components.
    Board: 50mm x 40mm.
    """
    components = (
        _make_w5500(),
        _make_vcc_decoupling(),
        _make_bulk_decoupling(),
        _make_avdd_decoupling(),
        _make_crystal(),
        _make_crystal_cap_1(),
        _make_crystal_cap_2(),
        _make_rj45(),
        _make_tx_term_plus(),
        _make_tx_term_minus(),
        _make_rsvd_resistor(),
        _make_spi_header(),
        _make_power_header(),
    )

    nets = _build_nets()

    all_refs = tuple(c.ref for c in components)
    all_net_names = tuple(n.name for n in nets)

    ethernet_feature = FeatureBlock(
        name="Ethernet",
        description=(
            "W5500 SPI-to-Ethernet controller with 25MHz crystal, RJ45+magnetics "
            "(HR911105A), differential pair termination, RSVD bias, and SPI breakout"
        ),
        components=all_refs,
        nets=all_net_names,
        subcircuits=("crystal_osc", "decoupling"),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="EthernetTraining", revision="v1"),
        features=(ethernet_feature,),
        components=components,
        nets=nets,
        mechanical=MechanicalConstraints(
            board_width_mm=_BOARD_WIDTH_MM, board_height_mm=_BOARD_HEIGHT_MM,
        ),
    )


# ---------------------------------------------------------------------------
# Design rules compliance check
# ---------------------------------------------------------------------------


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Euclidean distance between two component positions (ignoring rotation)."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


_ETH_COMPONENTS: tuple[Component, ...] = ()  # set in main()


_Pos = tuple[float, float, float]
_PosMap = dict[str, _Pos]

_ETH_REQUIRED_REFS: tuple[str, ...] = (
    "U1", "J1", "J2", "J3", "Y1", "C1", "C2", "C3", "C4", "C5", "R1", "R2", "R3",
)

_ETH_FP_SIZE_MAP: dict[str, tuple[float, float]] = {
    "U1": (3.5, 3.5),
    "J1": (16.0, 14.0),
    "Y1": (3.6, 1.8),
    "C1": (2.2, 1.4), "C2": (2.2, 1.4), "C3": (2.2, 1.4),
    "C4": (2.2, 1.4), "C5": (2.2, 1.4),
    "R1": (2.2, 1.4), "R2": (2.2, 1.4), "R3": (2.2, 1.4),
    "J2": (2.54, 15.24),
    "J3": (2.54, 5.08),
}


def _eth_check_rj45_edge(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 1: J1 (RJ45) near top edge (Y < 5mm)."""
    print("--- RJ45 Edge Placement ---")
    j1_y = fp_map["J1"][1]
    label = f"  J1 Y={j1_y:.1f}mm (top edge, max {_RJ45_EDGE_MAX_MM}mm)"
    if j1_y <= _RJ45_EDGE_MAX_MM:
        passes.append(f"{label} OK")
        print(f"{label} OK")
    else:
        violations.append(f"{label} VIOLATION")
        print(f"{label} ** VIOLATION **")
    print()


def _eth_check_w5500_to_rj45(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 2: U1 (W5500) within 15mm of J1."""
    print("--- W5500 to RJ45 Proximity ---")
    d_u1_j1 = _dist(fp_map["U1"], fp_map["J1"])
    label = f"  U1-J1: {d_u1_j1:.1f}mm (MAX {_U1_J1_MAX_MM}mm)"
    if d_u1_j1 <= _U1_J1_MAX_MM:
        passes.append(f"{label} OK")
        print(f"{label} OK")
    else:
        violations.append(f"{label} VIOLATION")
        print(f"{label} ** VIOLATION **")
    print()


def _eth_check_crystal_proximity(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 3+4: Y1 (crystal) within 5mm of U1; C4/C5 within 3mm of Y1."""
    print("--- Crystal (Y1) to W5500 ---")
    d_y1_u1 = _dist(fp_map["Y1"], fp_map["U1"])
    label = f"  Y1-U1: {d_y1_u1:.1f}mm (MAX {_CRYSTAL_TO_U1_MAX_MM}mm)"
    if d_y1_u1 <= _CRYSTAL_TO_U1_MAX_MM:
        passes.append(f"{label} OK")
        print(f"{label} OK")
    else:
        violations.append(f"{label} VIOLATION")
        print(f"{label} ** VIOLATION **")

    for cap_ref in ("C4", "C5"):
        d = _dist(fp_map[cap_ref], fp_map["Y1"])
        label = f"  {cap_ref}-Y1: {d:.1f}mm (MAX {_XTAL_CAP_TO_CRYSTAL_MAX_MM}mm)"
        if d <= _XTAL_CAP_TO_CRYSTAL_MAX_MM:
            passes.append(f"{label} OK")
            print(f"{label} OK")
        else:
            violations.append(f"{label} VIOLATION")
            print(f"{label} ** VIOLATION **")
    print()


def _eth_check_decoupling_caps(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rules 5-7: Decoupling caps (C1/C2/C3) within spec distance of U1."""
    print("--- Decoupling Caps to W5500 ---")
    decoup_checks = [
        ("C1", "U1", _VCC_DECOUP_TO_U1_MAX_MM, "VCC decoupling"),
        ("C2", "U1", _BULK_DECOUP_TO_U1_MAX_MM, "Bulk decoupling"),
        ("C3", "U1", _AVDD_DECOUP_TO_U1_MAX_MM, "AVDD decoupling"),
    ]
    for ref_a, ref_b, max_d, desc in decoup_checks:
        d = _dist(fp_map[ref_a], fp_map[ref_b])
        label = f"  {ref_a}-{ref_b}: {d:.1f}mm ({desc}, MAX {max_d}mm)"
        if d > max_d:
            violations.append(f"{label} VIOLATION")
            print(f"{label} ** VIOLATION **")
        else:
            passes.append(f"{label} OK")
            print(f"{label} OK")
    print()


def _eth_check_tx_termination(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rules 8-9: TX termination (R1/R2) and RSVD (R3) proximity to U1."""
    print("--- TX Termination to W5500 ---")
    for ref in ("R1", "R2"):
        d = _dist(fp_map[ref], fp_map["U1"])
        label = f"  {ref}-U1: {d:.1f}mm (MAX {_TX_TERM_TO_U1_MAX_MM}mm)"
        if d <= _TX_TERM_TO_U1_MAX_MM:
            passes.append(f"{label} OK")
            print(f"{label} OK")
        else:
            violations.append(f"{label} VIOLATION")
            print(f"{label} ** VIOLATION **")

    d_r3 = _dist(fp_map["R3"], fp_map["U1"])
    label = f"  R3-U1: {d_r3:.1f}mm (RSVD, MAX {_RSVD_TO_U1_MAX_MM}mm)"
    if d_r3 <= _RSVD_TO_U1_MAX_MM:
        passes.append(f"{label} OK")
        print(f"{label} OK")
    else:
        violations.append(f"{label} VIOLATION")
        print(f"{label} ** VIOLATION **")
    print()


def _eth_check_header_edge(
    fp_map: _PosMap,
    board_height: float,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 10: J2/J3 (SPI/power headers) near bottom edge."""
    print("--- SPI/Power Header Placement (bottom edge) ---")
    bottom_min = board_height - _HEADER_BOTTOM_MARGIN_MM
    for ref, desc in (("J2", "SPI header"), ("J3", "Power header")):
        y = fp_map[ref][1]
        label = f"  {ref} Y={y:.1f}mm ({desc}, min {bottom_min:.0f}mm)"
        if y >= bottom_min:
            passes.append(f"{label} OK")
            print(f"{label} OK")
        else:
            violations.append(f"{label} VIOLATION")
            print(f"{label} ** VIOLATION **")
    print()


def _eth_check_signal_flow(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Rule 11: Signal flow top to bottom (J1.Y < U1.Y < J2.Y)."""
    print("--- Signal Flow (top to bottom) ---")
    j1_y = fp_map["J1"][1]
    u1_y = fp_map["U1"][1]
    j2_y = fp_map["J2"][1]
    flow_label = f"  J1(Y={j1_y:.1f}) -> U1(Y={u1_y:.1f}) -> J2(Y={j2_y:.1f})"
    if j1_y < u1_y < j2_y:
        passes.append(f"{flow_label} OK")
        print(f"{flow_label} OK")
    else:
        violations.append(f"{flow_label} VIOLATION (not top-to-bottom)")
        print(f"{flow_label} ** VIOLATION ** (not top-to-bottom)")
    print()


def _eth_check_diff_pair_symmetry(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """R1/R2 differential pair symmetry (close together)."""
    print("--- Differential Pair Symmetry ---")
    d_r1_r2 = _dist(fp_map["R1"], fp_map["R2"])
    label = f"  R1-R2: {d_r1_r2:.1f}mm (MAX {_DIFF_PAIR_MAX_MM}mm, should be paired)"
    if d_r1_r2 <= _DIFF_PAIR_MAX_MM:
        passes.append(f"{label} OK")
        print(f"{label} OK")
    else:
        violations.append(f"{label} VIOLATION")
        print(f"{label} ** VIOLATION **")
    print()


def _eth_check_courtyard_collisions(
    fp_map: _PosMap,
    violations: list[str],
    passes: list[str],
) -> None:
    """Courtyard overlap / collision detection."""
    print("--- Courtyard Collision Detection ---")
    from kicad_pipeline.optimization.collision_resolver import _count_collisions

    collisions = _count_collisions(fp_map, _ETH_FP_SIZE_MAP)
    if collisions:
        for ref_a, ref_b in collisions:
            msg = f"  Overlap: {ref_a} <-> {ref_b}"
            violations.append(f"{msg} VIOLATION")
            print(f"{msg} ** VIOLATION **")
    else:
        passes.append("  No courtyard collisions detected OK")
        print("  No courtyard collisions detected OK")
    print()


def _eth_check_3d_model_coverage() -> None:
    """3D model coverage report (informational only, no violations)."""
    print("--- 3D Model Coverage ---")
    # Components with commonly available 3D models
    has_3d = {"C_0805", "R_0805", "Crystal_SMD_3215", "LQFP-48"}
    # Components typically missing 3D models
    maybe_missing_3d = {"RJ45_HR911105A", "PinHeader_1x06_P2.54mm_Vertical",
                        "PinHeader_1x02_P2.54mm_Vertical"}
    for comp in _ETH_COMPONENTS:
        if comp.footprint in has_3d:
            print(f"  {comp.ref} ({comp.footprint}): 3D model likely available")
        elif comp.footprint in maybe_missing_3d:
            print(f"  {comp.ref} ({comp.footprint}): 3D model may be MISSING")
        else:
            print(f"  {comp.ref} ({comp.footprint}): 3D model status unknown")
    print()


def _eth_check_lcsc_footprints(
    violations: list[str],
    passes: list[str],
) -> None:
    """LCSC footprint verification."""
    print("--- LCSC Footprint Verification ---")
    known_bad_lcsc = {
        "C2337": "pulls 40-pin header (not 4-pin)",
    }
    for comp in _ETH_COMPONENTS:
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


def _check_design_rules(
    fp_map: dict[str, tuple[float, float, float]],
    board_width: float = _BOARD_WIDTH_MM,
    board_height: float = _BOARD_HEIGHT_MM,
) -> None:
    """Check Ethernet subsystem design rules and print compliance report.

    Rules from docs/design_rules/ethernet_subsystem.md:
    1. J1 (RJ45) near top edge (Y < 5mm)
    2. U1 (W5500) within 15mm of J1
    3. Y1 (crystal) within 5mm of U1
    4. C4, C5 (crystal load caps) within 3mm of Y1
    5. C1 (VCC decoupling) within 3mm of U1
    6. C2 (bulk decoupling) within 5mm of U1
    7. C3 (AVDD decoupling) within 3mm of U1
    8. R1, R2 (TX termination) within 3mm of U1
    9. R3 (RSVD) within 5mm of U1
    10. J2, J3 (SPI/power headers) near bottom edge (Y > board_height - 8mm)
    11. Signal flow: J1.Y < U1.Y < J2.Y (top to bottom)
    """
    print("=" * 60)
    print("DESIGN RULES COMPLIANCE CHECK (Ethernet Subsystem)")
    print("=" * 60)
    print()

    violations: list[str] = []
    passes: list[str] = []

    missing = [r for r in _ETH_REQUIRED_REFS if r not in fp_map]
    if missing:
        print(f"  MISSING COMPONENTS: {missing}")
        print("  Cannot run design rules check.")
        return

    _eth_check_rj45_edge(fp_map, violations, passes)
    _eth_check_w5500_to_rj45(fp_map, violations, passes)
    _eth_check_crystal_proximity(fp_map, violations, passes)
    _eth_check_decoupling_caps(fp_map, violations, passes)
    _eth_check_tx_termination(fp_map, violations, passes)
    _eth_check_header_edge(fp_map, board_height, violations, passes)
    _eth_check_signal_flow(fp_map, violations, passes)
    _eth_check_diff_pair_symmetry(fp_map, violations, passes)
    _eth_check_courtyard_collisions(fp_map, violations, passes)
    _eth_check_3d_model_coverage()
    _eth_check_lcsc_footprints(violations, passes)

    print("=" * 60)
    print(f"PASSED: {len(passes)}  |  VIOLATIONS: {len(violations)}")
    print("=" * 60)
    if violations:
        print("\nViolation details:")
        for v in violations:
            print(f"  ** {v}")


# ---------------------------------------------------------------------------
# Post-placement corrections
# ---------------------------------------------------------------------------


def _apply_ethernet_post_placement(pcb: object) -> object:
    """Apply pattern-based corrections to Ethernet board placement.

    Human-reference layout rules (board 50x40mm, signal flow top-to-bottom):

    1. **J1 (RJ45)** at top edge, centered: large connector flush with top.
       Body is ~16x14mm, so center at Y ~ 3-4mm for flush top placement.

    2. **R1/R2 (TX termination)** between J1 and U1: these terminate the
       differential pairs, so they sit between the RJ45 and the W5500.

    3. **U1 (W5500)** below J1, center area: LQFP-48 (~9x9mm body).
       Short diff-pair traces from J1 pins through R1/R2 to U1 TX/RX pins.

    4. **Y1 (crystal)** beside U1 within 5mm: crystal osc for W5500 clock.
       C4/C5 (load caps) flank Y1 on either side.

    5. **C1/C2/C3 (decoupling)** tight to U1 power pins (< 3mm for 100nF,
       < 5mm for 10uF bulk). Placed on the opposite side of U1 from J1.

    6. **R3 (RSVD)** near U1 EXRES1 pin, within 5mm.

    7. **J2 (SPI header)** at bottom edge for MCU connection.

    8. **J3 (power header)** at bottom edge for power input.
    """
    from dataclasses import replace

    from kicad_pipeline.models.pcb import Point

    board_w = _BOARD_WIDTH_MM
    board_h = _BOARD_HEIGHT_MM

    # Component half-sizes (for clearance calculations)
    # U1 LQFP-48: ~9x9mm body + courtyard -> half = 4.5
    # J1 RJ45: ~16x14mm body -> half_x=8, half_y=7
    # 0805 passives: ~2.2x1.4mm -> half_x=1.1, half_y=0.7
    # Crystal 3215: ~3.6x1.8mm -> half_x=1.8, half_y=0.9
    # 6-pin header: ~2.54x15.24mm -> half_x=1.27, half_y=7.62
    # 2-pin header: ~2.54x5.08mm -> half_x=1.27, half_y=2.54

    # J1 (RJ45, 16x14mm body) flush at top edge, left-center area
    j1_x = board_w * _ETH_J1_X_FRAC   # 27.5  centered on board
    j1_y = _ETH_J1_Y_MM               # top edge

    # U1 (W5500, body 7.5x7.5 for courtyard) below J1
    # J1 bottom edge ~ j1_y + 7 = 10.5, U1 top edge = u1_y - 3.75
    # Gap of ~1mm => u1_y = 10.5 + 3.75 + 1.0 = 15.25
    u1_x = board_w * _ETH_U1_X_FRAC   # 25.0  centered
    u1_y = _ETH_U1_Y_MM               # below J1, slightly higher to bring everything tighter

    # COLLISION CLEARANCE RULES (AABB based, U1 body=7.5x7.5):
    # U1 (7.5x7.5) to 0805 (2.2x1.4): min X = (7.5+2.2)/2 = 4.85, Y = (7.5+1.4)/2 = 4.45
    # U1 to Crystal (3.6x1.8): min X = (7.5+3.6)/2 = 5.55, Y = (7.5+1.8)/2 = 4.65
    # J1 (16x14) to R(0805 rot90=1.4x2.2): min X = (16+1.4)/2 = 8.7, Y = (14+2.2)/2 = 8.1
    # 0805 to 0805: min X = 2.2, Y = 1.4

    # R1/R2 (TX termination) — right of J1, above U1
    # J1 right edge ~ j1_x + 8 = 26.0
    # U1 top edge ~ u1_y - 3.75 = 11.75
    # Place R1/R2 rotated 90 (1.4w x 2.2h) above U1 right side
    # R to U1: min Y (rot90) = (7.5+2.2)/2 = 4.85
    # So R_y < u1_y - 4.85 = 10.65
    # R to J1: min X from J1 center = (16+1.4)/2 = 8.7
    # R_x > j1_x + 8.7 = 26.7
    # With U1 courtyard at 5.5x5.5:
    # U1 to 0805: min X = (5.5+2.2)/2 = 3.85, min Y = (5.5+1.4)/2 = 3.45
    # U1 to Crystal: min X = (5.5+3.6)/2 = 4.55, min Y = (5.5+1.8)/2 = 3.65
    # U1 to R(rot90, 1.4x2.2): min X = (5.5+1.4)/2 = 3.45, min Y = (5.5+2.2)/2 = 3.85

    # R1/R2 (TX termination) above U1, right side
    # J1 at (18, 3.5), J1 right edge = 18+8 = 26.0
    # R rot90 to J1: min X = (16+1.4)/2 = 8.7 from j1_x
    # r1_x > 18 + 8.7 = 26.7
    # R to U1: min Y(rot90) = 3.85, so r1_y < u1_y - 3.85 = 11.65
    # Place R1/R2 above U1. rot90 => 1.4w x 2.2h
    # Must be > 8.7mm X from J1 center (18.0): r1_x > 26.7
    # R-R min X gap = (1.4+1.4)/2 = 1.4
    # R1/R2 (TX termination) — above U1, symmetric. rot90 => effective (1.4, 2.2)
    # U1 body 3.5x3.5: min Y = (3.5+2.2)/2 = 2.85
    # At dx=±0.8, dy=2.85: dist=sqrt(0.64+8.12)=2.96 ≤3.0
    # AABB: dx=0.8 < (3.5+1.4)/2=2.45 AND dy=2.85 = 2.85 => borderline
    # R-R: dx=1.6 > (1.4+1.4)/2=1.4 => no collision
    r1_x = u1_x - _ETH_R_DX_MM        # 24.2  left of U1 center
    r1_y = u1_y - _ETH_R_DY_MM        # 12.63  above U1 (dy=2.87 > 2.85 collision, dist=2.98 ≤3.0)
    r2_x = u1_x + _ETH_R_DX_MM        # 25.8  right of U1 center
    r2_y = r1_y                        # 12.63

    # Y1 (crystal) right of U1 — min X collision-free = 4.55
    # Place at dx=4.6 for Euclidean ~4.6mm (under 5mm!)
    y1_x = u1_x + _ETH_CRYSTAL_DX_MM  # 29.6
    y1_y = u1_y                        # same Y

    # C4/C5 (crystal load caps) flanking Y1
    c4_x = y1_x                           # aligned with Y1
    c4_y = y1_y - _ETH_XTAL_CAP_DY_MM   # above Y1
    c5_x = y1_x                           # aligned with Y1
    c5_y = y1_y + _ETH_XTAL_CAP_DY_MM   # below Y1

    # Decoupling caps around U1 — must be within 3mm Euclidean.
    # U1 collision body 3.5x3.5, 0805 cap = 2.2x1.4.
    # Min collision-free: X=(3.5+2.2)/2=2.85, Y=(3.5+1.4)/2=2.45
    # Cap-cap min: X=(2.2+2.2)/2=2.2, Y=(1.4+1.4)/2=1.4
    #
    # Strategy: C2 directly below, C1 and C3 on left side of U1 (away from
    # crystal caps C4/C5 on right side)
    c2_x = u1_x                            # 25.0  centered below
    c2_y = u1_y + _ETH_C2_DY_MM           # 18.0  dist=2.5

    # C1/C3 on left side at dx=-2.85 (just outside collision zone)
    # Vertically staggered by 1.5mm (> 1.4 cap-cap min Y)
    c1_x = u1_x - _ETH_C1C3_DX_MM        # 22.15  left of U1
    c1_y = u1_y - _ETH_C1_DY_MM          # 14.8   above center, dist=sqrt(8.12+0.49)=2.94

    c3_x = u1_x - _ETH_C1C3_DX_MM        # 22.15  left of U1
    c3_y = u1_y + _ETH_C3_DY_MM          # 16.3   below center, dist=sqrt(8.12+0.64)=2.96

    # R3 (RSVD bias) below-left of U1.
    # C1 at (22.15, 14.8), C3 at (22.15, 16.3).
    # Place R3 further left: (19.5, 15.5) → dist from U1=5.5 → too far.
    # Place below C3: (22.15, 17.8) → R3-C3 dy=1.5>1.4, R3-C2 dx=2.85>2.2 OK
    # R3-U1: sqrt(2.85²+2.3²)=sqrt(8.12+5.29)=3.66 ≤5.0
    r3_x = u1_x - _ETH_R3_DX_MM          # 22.15  aligned with C1/C3
    r3_y = u1_y + _ETH_R3_DY_MM          # 17.8   below C3

    # J2 (SPI 6-pin header, 2.54x15.24mm) at bottom edge
    j2_x = board_w * _ETH_J2_X_FRAC      # 15.0
    j2_y = board_h - _ETH_HEADER_Y_OFFSET_MM   # 36.5

    # J3 (power 2-pin header, 2.54x5.08mm) at bottom edge, right side
    j3_x = board_w * _ETH_J3_X_FRAC      # 35.0
    j3_y = board_h - _ETH_HEADER_Y_OFFSET_MM   # 36.5

    placement_rules: dict[str, tuple[float, float, float]] = {
        "J1": (j1_x, j1_y, 0.0),
        "U1": (u1_x, u1_y, 0.0),
        "R1": (r1_x, r1_y, 90.0),      # vertical for diff pair routing
        "R2": (r2_x, r2_y, 90.0),      # vertical for diff pair routing
        "Y1": (y1_x, y1_y, 0.0),
        "C4": (c4_x, c4_y, 0.0),
        "C5": (c5_x, c5_y, 0.0),
        "C1": (c1_x, c1_y, 0.0),
        "C2": (c2_x, c2_y, 0.0),
        "C3": (c3_x, c3_y, 0.0),
        "R3": (r3_x, r3_y, 0.0),
        "J2": (j2_x, j2_y, 0.0),
        "J3": (j3_x, j3_y, 0.0),
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
    """Build Ethernet board, optimize, render, and report."""
    output_dir = _repo / "output" / "train_ethernet"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "train_ethernet_placement.png"

    print("=== Ethernet Subsystem Training Board ===")
    print()

    # 1. Build requirements
    requirements = _build_requirements()
    print(f"Components: {len(requirements.components)}")
    print(f"Nets:       {len(requirements.nets)}")
    print(f"Board:      {int(_BOARD_WIDTH_MM)} x {int(_BOARD_HEIGHT_MM)} mm")
    print()

    # 2. Build PCB (no routing)
    print("Building PCB...")
    pcb = build_pcb(requirements, auto_route=False, placement_mode="grouped")
    print(f"  Footprints: {len(pcb.footprints)}")
    print()

    # 3. Run placement optimizer
    print("Running EE placement optimizer...")
    optimized_pcb, review = optimize_placement_ee(requirements, pcb)

    # Apply post-placement corrections for known layout violations
    print("  Applying post-placement corrections...")
    optimized_pcb = _apply_ethernet_post_placement(optimized_pcb)

    # Re-run review after corrections
    from kicad_pipeline.optimization.review_agent import review_placement
    review = review_placement(optimized_pcb, requirements)

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
        title="Ethernet Subsystem Training Board - Placement",
        score=score,
        group_map=group_map,
    )
    print(f"  Saved: {output_png}")
    print()

    # 6. Write KiCad PCB file and compare against reference
    pcb_path = output_dir / "train_ethernet.kicad_pcb"
    write_and_compare_pcb(optimized_pcb, pcb_path)

    # 7. Write KiCad project file
    pro_path = write_project_file("train_ethernet", output_dir)
    print(f"  KiCad project: {pro_path}")
    print()

    # 8. Print component positions
    fp_map = print_component_positions(optimized_pcb)

    # 9. Design rules compliance check
    global _ETH_COMPONENTS
    _ETH_COMPONENTS = requirements.components
    _check_design_rules(fp_map)


if __name__ == "__main__":
    main()
