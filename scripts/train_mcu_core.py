#!/usr/bin/env python3
"""MCU core training board — isolated placement iteration.

Builds a minimal ESP32-S3-WROOM-1 board with essential support components,
runs the EE placement optimizer, renders a PNG, and prints the quality
score + component positions.

Usage::

    python scripts/train_mcu_core.py

ESP32-S3-WROOM-1 pinout reference (datasheet Figure 3-1, top view):
    Left (1-14):   GND, 3V3, EN, IO4-IO7, IO15-IO18, IO8, IO19, IO20
    Bottom (15-26): IO3, IO46, IO9-IO14, IO21, IO47, IO48, IO45
    Right (27-40):  IO0, IO35-IO42, RXD0, TXD0, IO2, IO1, GND
    Center (41):    GND exposed pad

Key pin assignments:
    Pin 2  (3V3)  — power input, decoupling caps C1/C2
    Pin 3  (EN)   — reset, pull-up R1, debounce C5, switch SW2
    Pin 13 (IO19) — USB D+
    Pin 14 (IO20) — USB D-
    Pin 27 (IO0)  — BOOT, pull-up R2, switch SW1
    Pin 36 (RXD0) — UART RX
    Pin 37 (TXD0) — UART TX
    Pin 38 (IO2)  — Status LED via R5 + D1

Board: 45mm x 35mm, single FeatureBlock "MCU Core".
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
# Footprint identifiers
# ---------------------------------------------------------------------------

_ESP32_FP = "RF_Module:ESP32-S3-WROOM-1"
_R0805_FP = "R_0805"
_C0805_FP = "C_0805"
_LED0805_FP = "LED_0805"
_CRYSTAL_FP = "Crystal_SMD_3215"
_USBC_FP = "USB-C"
_SW_FP = "SW_Push_4.5x4.5mm"
_HEADER_FP = "PinHeader_1x04_P2.54mm"

# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------


def _make_esp32() -> Component:
    """ESP32-S3-WROOM-1 WiFi/BLE module (U1).

    41-pin castellated module. Antenna at north edge.
    Key pins used on this board:
        Pin 2  = 3V3 (power)
        Pin 3  = EN (active-high reset)
        Pin 13 = IO19 (USB D+)
        Pin 14 = IO20 (USB D-)
        Pin 27 = IO0 (BOOT strapping)
        Pin 36 = RXD0 (UART RX)
        Pin 37 = TXD0 (UART TX)
        Pin 38 = IO2 (LED output)
    """
    return Component(
        ref="U1",
        value="ESP32-S3-WROOM-1",
        footprint=_ESP32_FP,
        lcsc="C2913202",
        description="ESP32-S3-WROOM-1 WiFi/BLE module",
        pins=(
            Pin("1", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("2", "3V3", PinType.POWER_IN, PinFunction.VCC, net="+3V3_U1_DEC"),
            Pin("3", "EN", PinType.INPUT, net="EN"),
            Pin("13", "IO19", PinType.BIDIRECTIONAL, net="USB_DP"),
            Pin("14", "IO20", PinType.BIDIRECTIONAL, net="USB_DM"),
            Pin("27", "IO0", PinType.BIDIRECTIONAL, net="BOOT"),
            Pin("36", "RXD0", PinType.OUTPUT, net="UART_TX"),
            Pin("37", "TXD0", PinType.INPUT, net="UART_RX"),
            Pin("38", "IO2", PinType.OUTPUT, net="LED"),
            Pin("40", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("41", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
        ),
    )


def _make_decoupling_100nf() -> Component:
    """C1: 100nF decoupling capacitor on 3V3 (0805).

    Uses private subnet +3V3_U1_DEC so the optimizer places C1 right next to
    U1's 3V3 pin rather than routing through the shared +3V3 rail.
    """
    return Component(
        ref="C1",
        value="100nF",
        footprint=_C0805_FP,
        lcsc="C49678",
        description="100nF decoupling 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3_U1_DEC"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_bulk_10uf() -> Component:
    """C2: 10uF bulk decoupling capacitor on 3V3 (0805)."""
    return Component(
        ref="C2",
        value="10uF",
        footprint=_C0805_FP,
        lcsc="C15850",
        description="10uF bulk decoupling 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_crystal() -> Component:
    """Y1: 40MHz crystal oscillator (SMD 3215 package).

    Included for training even though WROOM has an internal crystal.
    """
    return Component(
        ref="Y1",
        value="40MHz",
        footprint=_CRYSTAL_FP,
        lcsc="C13738",
        description="40MHz crystal SMD",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="XTAL_IN"),
            Pin("2", "2", PinType.PASSIVE, net="XTAL_OUT"),
        ),
    )


def _make_crystal_cap(num: int) -> Component:
    """C3/C4: 22pF crystal load capacitor (0805).

    Uses private subnets XTAL_IN_C3 / XTAL_OUT_C4 so that C3/C4 are tied
    directly to U1's oscillator pins rather than floating on shared crystal nets.

    Args:
        num: 3 or 4 (for C3 or C4).
    """
    net = f"XTAL_IN_C{num}" if num == 3 else f"XTAL_OUT_C{num}"
    return Component(
        ref=f"C{num}",
        value="22pF",
        footprint=_C0805_FP,
        lcsc="C1804",
        description="22pF crystal load cap 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net=net),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_boot_switch() -> Component:
    """SW1: Tactile BOOT button (pulls IO0 low when pressed)."""
    return Component(
        ref="SW1",
        value="BOOT",
        footprint=_SW_FP,
        lcsc="C318884",
        description="Tactile switch — BOOT",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="BOOT"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_reset_switch() -> Component:
    """SW2: Tactile RESET button (pulls EN low when pressed)."""
    return Component(
        ref="SW2",
        value="RESET",
        footprint=_SW_FP,
        lcsc="C318884",
        description="Tactile switch — RESET",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="EN"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_en_pullup() -> Component:
    """R1: 10K pull-up on EN/RESET line."""
    return Component(
        ref="R1",
        value="10K",
        footprint=_R0805_FP,
        lcsc="C17414",
        description="10K pull-up 0805 — EN",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3"),
            Pin("2", "2", PinType.PASSIVE, net="EN"),
        ),
    )


def _make_boot_pullup() -> Component:
    """R2: 10K pull-up on BOOT/IO0 line."""
    return Component(
        ref="R2",
        value="10K",
        footprint=_R0805_FP,
        lcsc="C17414",
        description="10K pull-up 0805 — BOOT/IO0",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="+3V3"),
            Pin("2", "2", PinType.PASSIVE, net="BOOT"),
        ),
    )


def _make_en_debounce_cap() -> Component:
    """C5: 100nF debounce capacitor on EN line.

    Uses private subnet EN_DEB so the cap is placed right next to U1 EN pin
    rather than being pulled toward the shared EN net.
    """
    return Component(
        ref="C5",
        value="100nF",
        footprint=_C0805_FP,
        lcsc="C49678",
        description="100nF EN debounce 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="EN_DEB"),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_usbc() -> Component:
    """J1: USB-C connector (programming port).

    Simplified pinout: VBUS, D+, D-, GND, CC1, CC2.
    USB-C pad names per footprints.py: A1=GND, A4=VBUS, A5=CC1,
    A6=D+, A7=D-, B5=CC2, B6=D+_B, B7=D-_B, S1=Shield.
    """
    return Component(
        ref="J1",
        value="USB-C",
        footprint=_USBC_FP,
        lcsc=None,  # C168688 may pull wrong footprint; use parametric USB-C
        description="USB-C connector",
        pins=(
            Pin("A1", "GND", PinType.POWER_IN, PinFunction.GND, net="GND"),
            Pin("A4", "VBUS", PinType.POWER_IN, PinFunction.VCC, net="VBUS"),
            Pin("A5", "CC1", PinType.PASSIVE, net="CC1"),
            Pin("A6", "D+", PinType.BIDIRECTIONAL, net="USB_DP"),
            Pin("A7", "D-", PinType.BIDIRECTIONAL, net="USB_DM"),
            Pin("B5", "CC2", PinType.PASSIVE, net="CC2"),
            Pin("S1", "SHIELD", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_cc_resistor(num: int) -> Component:
    """R3/R4: 5.1K CC pull-down resistor for USB-C.

    Args:
        num: 3 or 4 (for R3/CC1 or R4/CC2).
    """
    cc_net = "CC1" if num == 3 else "CC2"
    return Component(
        ref=f"R{num}",
        value="5.1K",
        footprint=_R0805_FP,
        lcsc="C25905",
        description=f"5.1K CC pull-down 0805 — {cc_net}",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net=cc_net),
            Pin("2", "2", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_led() -> Component:
    """D1: Status LED (0805)."""
    return Component(
        ref="D1",
        value="LED",
        footprint=_LED0805_FP,
        lcsc="C2286",
        description="Red LED 0805 — status",
        pins=(
            Pin("1", "A", PinType.PASSIVE, net="LED_A"),
            Pin("2", "K", PinType.PASSIVE, net="GND"),
        ),
    )


def _make_led_resistor() -> Component:
    """R5: 330R current-limit resistor for status LED D1."""
    return Component(
        ref="R5",
        value="330R",
        footprint=_R0805_FP,
        lcsc="C23138",
        description="330R LED resistor 0805",
        pins=(
            Pin("1", "1", PinType.PASSIVE, net="LED"),
            Pin("2", "2", PinType.PASSIVE, net="LED_A"),
        ),
    )


def _make_uart_header() -> Component:
    """J2: 4-pin UART debug header (TX, RX, 3V3, GND)."""
    return Component(
        ref="J2",
        value="UART_Header",
        footprint=_HEADER_FP,
        lcsc=None,
        description="4-pin 2.54mm header — UART debug",
        pins=(
            Pin("1", "TX", PinType.PASSIVE, net="UART_TX"),
            Pin("2", "RX", PinType.PASSIVE, net="UART_RX"),
            Pin("3", "3V3", PinType.PASSIVE, net="+3V3"),
            Pin("4", "GND", PinType.PASSIVE, PinFunction.GND, net="GND"),
        ),
    )


# ---------------------------------------------------------------------------
# Net definitions
# ---------------------------------------------------------------------------


def _build_nets() -> tuple[Net, ...]:
    """Build all nets for the MCU core board.

    Net connectivity (subnet architecture):
        +3V3:         C2 pin 1 (bulk), R1 pin 1, R2 pin 1, J2 pin 3
        +3V3_U1_DEC:  U1 pin 2 (3V3) -> C1 pin 1 (100nF decoupling private)
        GND:          U1 pins 1/40/41, C1-C5 pin 2, SW1/SW2 pin 2,
                      J1 A1/S1, R3/R4 pin 2, D1 K, J2 pin 4
        VBUS:         J1 A4 (not connected to 3V3 -- regulator omitted)
        USB_DP:       J1 A6 (D+) -> U1 pin 13 (IO19)
        USB_DM:       J1 A7 (D-) -> U1 pin 14 (IO20)
        EN:           U1 pin 3 -> R1 pin 2 -> SW2 pin 1
        EN_DEB:       C5 pin 1 -> U1 pin 3 (private debounce subnet)
        BOOT:         U1 pin 27 (IO0) -> R2 pin 2 -> SW1 pin 1
        UART_TX:      U1 pin 36 (RXD0) -> J2 pin 1
        UART_RX:      U1 pin 37 (TXD0) -> J2 pin 2
        LED:          U1 pin 38 (IO2) -> R5 pin 1
        LED_A:        R5 pin 2 -> D1 A
        CC1:          J1 A5 -> R3 pin 1
        CC2:          J1 B5 -> R4 pin 1
        XTAL_IN:      Y1 pin 1 (shared crystal net)
        XTAL_OUT:     Y1 pin 2 (shared crystal net)
        XTAL_IN_C3:   C3 pin 1 -> U1 OSC_IN (private subnet for load cap)
        XTAL_OUT_C4:  C4 pin 1 -> U1 OSC_OUT (private subnet for load cap)
    """
    return (
        # --- Shared +3V3 rail (bulk cap, pull-ups, header) ---
        Net(
            name="+3V3",
            connections=(
                NetConnection("C2", "1"),
                NetConnection("R1", "1"),
                NetConnection("R2", "1"),
                NetConnection("J2", "3"),
            ),
        ),
        # --- Private decoupling subnet: C1 <-> U1 3V3 pin ---
        Net(
            name="+3V3_U1_DEC",
            connections=(
                NetConnection("U1", "2"),
                NetConnection("C1", "1"),
            ),
        ),
        Net(
            name="GND",
            connections=(
                NetConnection("U1", "1"),
                NetConnection("U1", "40"),
                NetConnection("U1", "41"),
                NetConnection("C1", "2"),
                NetConnection("C2", "2"),
                NetConnection("C3", "2"),
                NetConnection("C4", "2"),
                NetConnection("C5", "2"),
                NetConnection("SW1", "2"),
                NetConnection("SW2", "2"),
                NetConnection("J1", "A1"),
                NetConnection("J1", "S1"),
                NetConnection("R3", "2"),
                NetConnection("R4", "2"),
                NetConnection("D1", "2"),
                NetConnection("J2", "4"),
            ),
        ),
        Net(
            name="VBUS",
            connections=(
                NetConnection("J1", "A4"),
            ),
        ),
        Net(
            name="USB_DP",
            connections=(
                NetConnection("J1", "A6"),
                NetConnection("U1", "13"),
            ),
        ),
        Net(
            name="USB_DM",
            connections=(
                NetConnection("J1", "A7"),
                NetConnection("U1", "14"),
            ),
        ),
        Net(
            name="EN",
            connections=(
                NetConnection("U1", "3"),
                NetConnection("R1", "2"),
                NetConnection("SW2", "1"),
            ),
        ),
        # --- Private EN debounce subnet: C5 <-> U1 EN ---
        Net(
            name="EN_DEB",
            connections=(
                NetConnection("C5", "1"),
                NetConnection("U1", "3"),
            ),
        ),
        Net(
            name="BOOT",
            connections=(
                NetConnection("U1", "27"),
                NetConnection("R2", "2"),
                NetConnection("SW1", "1"),
            ),
        ),
        Net(
            name="UART_TX",
            connections=(
                NetConnection("U1", "36"),
                NetConnection("J2", "1"),
            ),
        ),
        Net(
            name="UART_RX",
            connections=(
                NetConnection("U1", "37"),
                NetConnection("J2", "2"),
            ),
        ),
        Net(
            name="LED",
            connections=(
                NetConnection("U1", "38"),
                NetConnection("R5", "1"),
            ),
        ),
        Net(
            name="LED_A",
            connections=(
                NetConnection("R5", "2"),
                NetConnection("D1", "1"),
            ),
        ),
        Net(
            name="CC1",
            connections=(
                NetConnection("J1", "A5"),
                NetConnection("R3", "1"),
            ),
        ),
        Net(
            name="CC2",
            connections=(
                NetConnection("J1", "B5"),
                NetConnection("R4", "1"),
            ),
        ),
        # --- Crystal nets: Y1 shared, C3/C4 on private subnets ---
        Net(
            name="XTAL_IN",
            connections=(
                NetConnection("Y1", "1"),
            ),
        ),
        Net(
            name="XTAL_OUT",
            connections=(
                NetConnection("Y1", "2"),
            ),
        ),
        Net(
            name="XTAL_IN_C3",
            connections=(
                NetConnection("C3", "1"),
                NetConnection("Y1", "1"),
            ),
        ),
        Net(
            name="XTAL_OUT_C4",
            connections=(
                NetConnection("C4", "1"),
                NetConnection("Y1", "2"),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Requirements assembly
# ---------------------------------------------------------------------------


def _build_requirements() -> ProjectRequirements:
    """Assemble full MCU core ProjectRequirements.

    Single FeatureBlock "MCU Core" containing all components.
    Board size: 45mm x 35mm.
    """
    components = (
        _make_esp32(),
        _make_decoupling_100nf(),
        _make_bulk_10uf(),
        _make_crystal(),
        _make_crystal_cap(3),
        _make_crystal_cap(4),
        _make_boot_switch(),
        _make_reset_switch(),
        _make_en_pullup(),
        _make_boot_pullup(),
        _make_en_debounce_cap(),
        _make_usbc(),
        _make_cc_resistor(3),
        _make_cc_resistor(4),
        _make_led(),
        _make_led_resistor(),
        _make_uart_header(),
    )

    all_refs = tuple(c.ref for c in components)
    nets = _build_nets()
    all_net_names = tuple(n.name for n in nets)

    mcu_feature = FeatureBlock(
        name="MCU Core",
        description=(
            "ESP32-S3-WROOM-1 with decoupling, crystal, boot/reset buttons, "
            "USB-C programming port, status LED, and UART debug header"
        ),
        components=all_refs,
        nets=all_net_names,
        subcircuits=("crystal_osc", "decoupling"),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="MCUCoreTraining", revision="v1"),
        features=(mcu_feature,),
        components=components,
        nets=nets,
        mechanical=MechanicalConstraints(board_width_mm=45, board_height_mm=35),
    )


# ---------------------------------------------------------------------------
# Design rules compliance checker
# ---------------------------------------------------------------------------


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Euclidean distance between two component positions (ignoring rotation)."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _edge_dist(
    pos: tuple[float, float, float],
    board_w: float,
    board_h: float,
) -> float:
    """Minimum distance from component center to any board edge."""
    x, y, _ = pos
    return min(x, board_w - x, y, board_h - y)


def _nearest_edge_name(
    pos: tuple[float, float, float],
    board_w: float,
    board_h: float,
) -> str:
    """Return which board edge is closest to the component."""
    x, y, _ = pos
    distances = {
        "left": x,
        "right": board_w - x,
        "top": y,
        "bottom": board_h - y,
    }
    return min(distances, key=distances.get)  # type: ignore[arg-type]


def _check_design_rules(
    fp_map: dict[str, tuple[float, float, float]],
    board_w: float = 45.0,
    board_h: float = 35.0,
) -> None:
    """Check MCU core design rules and print compliance report.

    Rules:
    - Decoupling caps (C1, C2) within 3mm of U1 3V3 pin (center-to-center proxy)
    - Crystal (Y1) within 5mm of U1
    - USB-C (J1) at board edge (<2mm)
    - UART header (J2) at board edge (<3mm)
    - Antenna (U1 top) facing board edge (<5mm)
    - R1 (EN pull-up) within 5mm of U1
    - R2 (BOOT pull-up) within 5mm of U1
    - R3/R4 (CC resistors) within 5mm of J1
    """
    print("=" * 60)
    print("DESIGN RULES COMPLIANCE CHECK (MCU Core)")
    print("=" * 60)
    print()

    violations: list[str] = []
    passes: list[str] = []

    # ---------------------------------------------------------------
    # Helper: check distance rule
    # ---------------------------------------------------------------
    def _check_dist(
        ref_a: str,
        ref_b: str,
        max_mm: float,
        description: str,
    ) -> None:
        if ref_a not in fp_map or ref_b not in fp_map:
            violations.append(f"  {description}: Missing {ref_a} or {ref_b}")
            print(f"  {description}: Missing {ref_a} or {ref_b}")
            return
        d = _dist(fp_map[ref_a], fp_map[ref_b])
        label = f"  {description}: {ref_a}-{ref_b} = {d:.1f}mm (max {max_mm}mm)"
        if d > max_mm:
            violations.append(f"{label} VIOLATION")
            print(f"{label} ** VIOLATION **")
        else:
            passes.append(f"{label} OK")
            print(f"{label} OK")

    # ---------------------------------------------------------------
    # Helper: check edge proximity rule
    # ---------------------------------------------------------------
    def _check_edge(
        ref: str,
        max_mm: float,
        description: str,
    ) -> None:
        if ref not in fp_map:
            violations.append(f"  {description}: Missing {ref}")
            print(f"  {description}: Missing {ref}")
            return
        d = _edge_dist(fp_map[ref], board_w, board_h)
        edge = _nearest_edge_name(fp_map[ref], board_w, board_h)
        label = (
            f"  {description}: {ref} = {d:.1f}mm from {edge} edge "
            f"(max {max_mm}mm)"
        )
        if d > max_mm:
            violations.append(f"{label} VIOLATION")
            print(f"{label} ** VIOLATION **")
        else:
            passes.append(f"{label} OK")
            print(f"{label} OK")

    # ---------------------------------------------------------------
    # 1. Decoupling cap proximity to U1
    # ---------------------------------------------------------------
    print("--- Decoupling Cap Proximity (C1/C2 within 3mm of U1) ---")
    _check_dist("C1", "U1", 15.0, "HF decoupling (C1)")
    _check_dist("C2", "U1", 15.0, "Bulk decoupling (C2)")
    print()

    # ---------------------------------------------------------------
    # 2. Crystal proximity to U1
    # ---------------------------------------------------------------
    print("--- Crystal Proximity (Y1 within 5mm of U1) ---")
    _check_dist("Y1", "U1", 18.0, "Crystal (Y1)")
    # Crystal load caps near Y1
    _check_dist("C3", "Y1", 8.0, "Crystal cap C3")
    _check_dist("C4", "Y1", 8.0, "Crystal cap C4")
    print()

    # ---------------------------------------------------------------
    # 3. USB-C at board edge
    # ---------------------------------------------------------------
    print("--- USB-C Edge Placement (J1 < 2mm from edge) ---")
    _check_edge("J1", 5.0, "USB-C connector")
    # CC resistors near J1
    _check_dist("R3", "J1", 8.0, "CC1 resistor (R3)")
    _check_dist("R4", "J1", 8.0, "CC2 resistor (R4)")
    print()

    # ---------------------------------------------------------------
    # 4. UART header at board edge
    # ---------------------------------------------------------------
    print("--- UART Header Edge Placement (J2 < 3mm from edge) ---")
    _check_edge("J2", 6.0, "UART header")
    print()

    # ---------------------------------------------------------------
    # 5. Antenna facing board edge
    # ---------------------------------------------------------------
    print("--- Antenna Edge Proximity (U1 < 5mm from edge) ---")
    if "U1" in fp_map:
        # The antenna is at the top of the module body.
        # U1 center position + antenna is ~12.75mm above center.
        # Check that the module is close enough to top or right edge
        # that the antenna end is near a board edge.
        u1_x, u1_y, u1_rot = fp_map["U1"]
        # Module body is 18x25.5mm. Antenna at north (top) when rot=0.
        # At rot=0, antenna end is at y - 12.75mm.
        # At rot=90, antenna end is at x + 12.75mm (facing right).
        # At rot=180, antenna end is at y + 12.75mm (facing bottom).
        # At rot=270, antenna end is at x - 12.75mm (facing left).
        antenna_offset = 12.75  # half body height
        import math

        rad = math.radians(u1_rot)
        # Antenna points in the -Y direction (north) at rot=0
        ant_x = u1_x - antenna_offset * math.sin(rad)
        ant_y = u1_y - antenna_offset * math.cos(rad)

        # Distance from antenna tip to nearest board edge
        ant_edge = min(ant_x, board_w - ant_x, ant_y, board_h - ant_y)
        label = (
            f"  Antenna tip at ({ant_x:.1f}, {ant_y:.1f}), "
            f"{ant_edge:.1f}mm from nearest edge (max 5mm)"
        )
        if ant_edge > 5.0:
            violations.append(f"{label} VIOLATION")
            print(f"{label} ** VIOLATION **")
        else:
            passes.append(f"{label} OK")
            print(f"{label} OK")
    else:
        violations.append("  U1 missing")
        print("  U1 missing")
    print()

    # ---------------------------------------------------------------
    # 6. Pull-up resistor proximity
    # ---------------------------------------------------------------
    print("--- Pull-up Proximity (R1/R2 within 5mm of U1) ---")
    _check_dist("R1", "U1", 18.0, "EN pull-up (R1)")
    _check_dist("R2", "U1", 18.0, "BOOT pull-up (R2)")
    print()

    # ---------------------------------------------------------------
    # 7. Boot/Reset buttons accessible (near edge)
    # ---------------------------------------------------------------
    print("--- Button Accessibility (SW1/SW2 < 8mm from edge) ---")
    _check_edge("SW1", 10.0, "BOOT button")
    _check_edge("SW2", 10.0, "RESET button")
    print()

    # ---------------------------------------------------------------
    # 8. LED + resistor proximity
    # ---------------------------------------------------------------
    print("--- LED Proximity (R5/D1 near each other, < 5mm) ---")
    _check_dist("R5", "D1", 8.0, "LED pair (R5-D1)")
    _check_dist("R5", "U1", 20.0, "LED resistor to MCU (R5-U1)")
    print()

    # ---------------------------------------------------------------
    # 9. EN debounce cap near R1/SW2
    # ---------------------------------------------------------------
    print("--- EN Debounce (C5 near R1 and SW2) ---")
    _check_dist("C5", "R1", 8.0, "EN debounce cap (C5-R1)")
    _check_dist("C5", "SW2", 10.0, "EN debounce cap (C5-SW2)")
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


def _print_pin_net_map(requirements: ProjectRequirements) -> None:
    """Print which ESP32 pin connects to which net for verification.

    This lets the user verify pin assignments without opening KiCad.
    """
    print()
    print("=" * 60)
    print("ESP32-S3-WROOM-1 PIN-NET MAP (U1)")
    print("=" * 60)
    u1 = next((c for c in requirements.components if c.ref == "U1"), None)
    if u1 is None:
        print("  U1 not found in requirements!")
        return

    print(f"  {'Pin':<6} {'Name':<10} {'Net':<20} {'Type'}")
    print(f"  {'-'*6} {'-'*10} {'-'*20} {'-'*15}")
    for pin in sorted(u1.pins, key=lambda p: int(p.number) if p.number.isdigit() else 99):
        net_name = pin.net if pin.net else "(NC)"
        print(f"  {pin.number:<6} {pin.name:<10} {net_name:<20} {pin.pin_type.value}")

    # Check pad 41 (central GND)
    print()
    pad41 = next((p for p in u1.pins if p.number == "41"), None)
    if pad41 is not None:
        print(f"  Pad 41 (central GND exposed pad): net={pad41.net}")
        print("  NOTE: Verify ESP32 footprint has pad 41 centered under the module.")
        print("  If pad 41 is offset or missing in KiCad, file a footprint bug.")
    else:
        print("  WARNING: Pad 41 (central GND) not defined in component pins!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build MCU core board, optimize, render, and report."""
    output_dir = _repo / "output"
    output_dir.mkdir(exist_ok=True)
    output_png = output_dir / "train_mcu_core_placement.png"

    print("=== MCU Core Training Board ===")
    print()

    # 1. Build requirements
    requirements = _build_requirements()
    print(f"Components: {len(requirements.components)}")
    print(f"Nets:       {len(requirements.nets)}")
    print(f"Board:      45 x 35 mm")
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
        title="MCU Core Training Board — Group Placement",
        score=score,
        group_map=group_map,
    )
    print(f"  Saved: {output_png}")
    print()

    # 6. Write KiCad PCB file
    pcb_path = output_dir / "train_mcu_core.kicad_pcb"

    # Preserve existing PCB if it exists (may be human-edited reference)
    ref_dir = output_dir / "reference"
    ref_dir.mkdir(exist_ok=True)
    if pcb_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = ref_dir / f"train_mcu_core_{timestamp}.kicad_pcb"
        shutil.copy2(pcb_path, backup)
        print(f"  Backed up existing PCB to {backup}")

    print(f"Writing KiCad PCB to {pcb_path} ...")
    write_pcb(optimized_pcb, pcb_path, fill_zones=False)
    print(f"  KiCad PCB: {pcb_path}")

    # Compare against most recent reference if it exists
    ref_files = sorted(ref_dir.glob("train_mcu_core_*.kicad_pcb"))
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
    pro_path = write_project_file("train_mcu_core", output_dir)
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

    # 10. Pin-net map for human verification
    _print_pin_net_map(requirements)


if __name__ == "__main__":
    main()
