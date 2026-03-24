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

_LQFP48_FP = "LQFP-48"
_RJ45_FP = "RJ45_HR911105A"
_CRYSTAL_FP = "Crystal_SMD_3215"
_C0805_FP = "C_0805"
_R0805_FP = "R_0805"
_HEADER_6P_FP = "PinHeader_1x06_P2.54mm_Vertical"
_HEADER_2P_FP = "PinHeader_1x02_P2.54mm_Vertical"

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
        lcsc="C124375",
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
        lcsc=None,  # C124375 is 6-pin; use parametric 2-pin header
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
        mechanical=MechanicalConstraints(board_width_mm=50, board_height_mm=40),
    )


# ---------------------------------------------------------------------------
# Design rules compliance check
# ---------------------------------------------------------------------------


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Euclidean distance between two component positions (ignoring rotation)."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


_ETH_COMPONENTS: tuple[Component, ...] = ()  # set in main()


def _check_design_rules(
    fp_map: dict[str, tuple[float, float, float]],
    board_width: float = 50.0,
    board_height: float = 40.0,
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

    required_refs = ["U1", "J1", "J2", "J3", "Y1", "C1", "C2", "C3", "C4", "C5",
                     "R1", "R2", "R3"]
    missing = [r for r in required_refs if r not in fp_map]
    if missing:
        print(f"  MISSING COMPONENTS: {missing}")
        print("  Cannot run design rules check.")
        return

    # --- RJ45 edge placement ---
    print("--- RJ45 Edge Placement ---")
    j1_y = fp_map["J1"][1]
    edge_margin = 5.0
    label = f"  J1 Y={j1_y:.1f}mm (top edge, max {edge_margin}mm)"
    if j1_y <= edge_margin:
        passes.append(f"{label} OK")
        print(f"{label} OK")
    else:
        violations.append(f"{label} VIOLATION")
        print(f"{label} ** VIOLATION **")
    print()

    # --- W5500 to RJ45 proximity ---
    print("--- W5500 to RJ45 Proximity ---")
    d_u1_j1 = _dist(fp_map["U1"], fp_map["J1"])
    label = f"  U1-J1: {d_u1_j1:.1f}mm (MAX 15mm)"
    if d_u1_j1 <= 15.0:
        passes.append(f"{label} OK")
        print(f"{label} OK")
    else:
        violations.append(f"{label} VIOLATION")
        print(f"{label} ** VIOLATION **")
    print()

    # --- Crystal to W5500 ---
    print("--- Crystal (Y1) to W5500 ---")
    d_y1_u1 = _dist(fp_map["Y1"], fp_map["U1"])
    label = f"  Y1-U1: {d_y1_u1:.1f}mm (MAX 5mm)"
    if d_y1_u1 <= 5.0:
        passes.append(f"{label} OK")
        print(f"{label} OK")
    else:
        violations.append(f"{label} VIOLATION")
        print(f"{label} ** VIOLATION **")

    # Crystal load caps to crystal
    for cap_ref in ("C4", "C5"):
        d = _dist(fp_map[cap_ref], fp_map["Y1"])
        label = f"  {cap_ref}-Y1: {d:.1f}mm (MAX 3mm)"
        if d <= 3.0:
            passes.append(f"{label} OK")
            print(f"{label} OK")
        else:
            violations.append(f"{label} VIOLATION")
            print(f"{label} ** VIOLATION **")
    print()

    # --- Decoupling caps to W5500 ---
    print("--- Decoupling Caps to W5500 ---")
    decoup_checks = [
        ("C1", "U1", 3.0, "VCC decoupling"),
        ("C2", "U1", 5.0, "Bulk decoupling"),
        ("C3", "U1", 3.0, "AVDD decoupling"),
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

    # --- TX termination resistors to W5500 ---
    print("--- TX Termination to W5500 ---")
    for ref in ("R1", "R2"):
        d = _dist(fp_map[ref], fp_map["U1"])
        label = f"  {ref}-U1: {d:.1f}mm (MAX 3mm)"
        if d <= 3.0:
            passes.append(f"{label} OK")
            print(f"{label} OK")
        else:
            violations.append(f"{label} VIOLATION")
            print(f"{label} ** VIOLATION **")

    # RSVD resistor
    d_r3 = _dist(fp_map["R3"], fp_map["U1"])
    label = f"  R3-U1: {d_r3:.1f}mm (RSVD, MAX 5mm)"
    if d_r3 <= 5.0:
        passes.append(f"{label} OK")
        print(f"{label} OK")
    else:
        violations.append(f"{label} VIOLATION")
        print(f"{label} ** VIOLATION **")
    print()

    # --- SPI/Power header edge placement ---
    print("--- SPI/Power Header Placement (bottom edge) ---")
    bottom_min = board_height - 8.0
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

    # --- Signal flow direction (top to bottom) ---
    print("--- Signal Flow (top to bottom) ---")
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

    # --- Differential pair symmetry (R1/R2 close together) ---
    print("--- Differential Pair Symmetry ---")
    d_r1_r2 = _dist(fp_map["R1"], fp_map["R2"])
    label = f"  R1-R2: {d_r1_r2:.1f}mm (MAX 4mm, should be paired)"
    if d_r1_r2 <= 4.0:
        passes.append(f"{label} OK")
        print(f"{label} OK")
    else:
        violations.append(f"{label} VIOLATION")
        print(f"{label} ** VIOLATION **")
    print()

    # --- Courtyard overlap / collision detection ---
    print("--- Courtyard Collision Detection ---")
    from kicad_pipeline.optimization.collision_resolver import _count_collisions

    # Build fp_sizes from known footprint dimensions (approximate courtyards)
    _fp_size_map: dict[str, tuple[float, float]] = {
        "U1": (9.0, 9.0),    # LQFP-48 ~7x7mm body + courtyard
        "J1": (16.0, 14.0),  # RJ45 with magnetics
        "Y1": (3.6, 1.8),    # Crystal SMD 3215
        "C1": (2.2, 1.4), "C2": (2.2, 1.4), "C3": (2.2, 1.4),
        "C4": (2.2, 1.4), "C5": (2.2, 1.4),
        "R1": (2.2, 1.4), "R2": (2.2, 1.4), "R3": (2.2, 1.4),
        "J2": (2.54, 15.24),  # 6-pin vertical header
        "J3": (2.54, 5.08),   # 2-pin vertical header
    }
    collisions = _count_collisions(fp_map, _fp_size_map)
    if collisions:
        for ref_a, ref_b in collisions:
            msg = f"  Overlap: {ref_a} <-> {ref_b}"
            violations.append(f"{msg} VIOLATION")
            print(f"{msg} ** VIOLATION **")
    else:
        passes.append("  No courtyard collisions detected OK")
        print("  No courtyard collisions detected OK")
    print()

    # --- 3D Model Coverage ---
    print("--- 3D Model Coverage ---")
    # Components with commonly available 3D models
    _HAS_3D = {"C_0805", "R_0805", "Crystal_SMD_3215", "LQFP-48"}
    # Components typically missing 3D models
    _MAYBE_MISSING_3D = {"RJ45_HR911105A", "PinHeader_1x06_P2.54mm_Vertical",
                         "PinHeader_1x02_P2.54mm_Vertical"}
    for comp in _ETH_COMPONENTS:
        if comp.footprint in _HAS_3D:
            print(f"  {comp.ref} ({comp.footprint}): 3D model likely available")
        elif comp.footprint in _MAYBE_MISSING_3D:
            print(f"  {comp.ref} ({comp.footprint}): 3D model may be MISSING")
        else:
            print(f"  {comp.ref} ({comp.footprint}): 3D model status unknown")
    print()

    # --- LCSC Footprint Verification ---
    print("--- LCSC Footprint Verification ---")
    _KNOWN_BAD_LCSC = {
        "C2337": "pulls 40-pin header (not 4-pin)",
    }
    for comp in _ETH_COMPONENTS:
        if comp.lcsc and comp.lcsc in _KNOWN_BAD_LCSC:
            msg = f"  {comp.ref} LCSC={comp.lcsc}: {_KNOWN_BAD_LCSC[comp.lcsc]}"
            violations.append(f"{msg} VIOLATION")
            print(f"{msg} ** VIOLATION **")
        elif comp.lcsc:
            passes.append(f"  {comp.ref} LCSC={comp.lcsc} OK")
            print(f"  {comp.ref} LCSC={comp.lcsc} OK")
        else:
            print(f"  {comp.ref} LCSC=None (parametric footprint)")
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
    """Build Ethernet board, optimize, render, and report."""
    output_dir = _repo / "output"
    output_dir.mkdir(exist_ok=True)
    output_png = output_dir / "train_ethernet_placement.png"

    print("=== Ethernet Subsystem Training Board ===")
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
        title="Ethernet Subsystem Training Board - Placement",
        score=score,
        group_map=group_map,
    )
    print(f"  Saved: {output_png}")
    print()

    # 6. Write KiCad PCB file
    pcb_path = output_dir / "train_ethernet.kicad_pcb"

    # Preserve existing PCB if it exists (may be human-edited reference)
    ref_dir = output_dir / "reference"
    ref_dir.mkdir(exist_ok=True)
    if pcb_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = ref_dir / f"train_ethernet_{timestamp}.kicad_pcb"
        shutil.copy2(pcb_path, backup)
        print(f"  Backed up existing PCB to {backup}")

    print(f"Writing KiCad PCB to {pcb_path} ...")
    write_pcb(optimized_pcb, pcb_path, fill_zones=False)
    print(f"  KiCad PCB: {pcb_path}")

    # Compare against most recent reference if it exists
    ref_files = sorted(ref_dir.glob("train_ethernet_*.kicad_pcb"))
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
    pro_path = write_project_file("train_ethernet", output_dir)
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
    global _ETH_COMPONENTS  # noqa: PLW0603
    _ETH_COMPONENTS = requirements.components
    _check_design_rules(fp_map)


if __name__ == "__main__":
    main()
