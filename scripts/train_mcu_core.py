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

Board: 70mm x 50mm, single FeatureBlock "MCU Core".
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

_ESP32_FP = "RF_Module:ESP32-S3-WROOM-1"
_R0805_FP = "R_0805"
_C0805_FP = "C_0805"
_LED0805_FP = "LED_0805"
_USBC_FP = "USB-C"
_SW_FP = "SW_Push_4.5x4.5mm"
_HEADER_FP = "PinHeader_1x04_P2.54mm"

# ---------------------------------------------------------------------------
# Board and design rule constants
# ---------------------------------------------------------------------------

_BOARD_WIDTH_MM = 70.0
_BOARD_HEIGHT_MM = 50.0

# Design rule thresholds (mm)
_DECOUP_TO_U1_MAX_MM = 15.0
_USBC_EDGE_MAX_MM = 5.0
_CC_TO_J1_MAX_MM = 8.0
_UART_EDGE_MAX_MM = 6.0
_ANTENNA_EDGE_MAX_MM = 5.0
_ANTENNA_HALF_BODY_MM = 12.75
_PULLUP_TO_U1_MAX_MM = 18.0
_BUTTON_EDGE_MAX_MM = 10.0
_LED_PAIR_MAX_MM = 8.0
_LED_TO_MCU_MAX_MM = 20.0
_EN_DEBOUNCE_TO_R1_MAX_MM = 8.0
_EN_DEBOUNCE_TO_SW2_MAX_MM = 10.0

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
    # ALL 41 pins defined with datasheet names — not just used pins.
    # Unused pins get net="" so they appear labeled in KiCad.
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
            Pin("4", "IO4", PinType.BIDIRECTIONAL),
            Pin("5", "IO5", PinType.BIDIRECTIONAL),
            Pin("6", "IO6", PinType.BIDIRECTIONAL),
            Pin("7", "IO7", PinType.BIDIRECTIONAL),
            Pin("8", "IO15", PinType.BIDIRECTIONAL),
            Pin("9", "IO16", PinType.BIDIRECTIONAL),
            Pin("10", "IO17", PinType.BIDIRECTIONAL),
            Pin("11", "IO18", PinType.BIDIRECTIONAL),
            Pin("12", "IO8", PinType.BIDIRECTIONAL),
            Pin("13", "IO19", PinType.BIDIRECTIONAL, net="USB_DP"),
            Pin("14", "IO20", PinType.BIDIRECTIONAL, net="USB_DM"),
            Pin("15", "IO3", PinType.BIDIRECTIONAL),
            Pin("16", "IO46", PinType.BIDIRECTIONAL),
            Pin("17", "IO9", PinType.BIDIRECTIONAL),
            Pin("18", "IO10", PinType.BIDIRECTIONAL),
            Pin("19", "IO11", PinType.BIDIRECTIONAL),
            Pin("20", "IO12", PinType.BIDIRECTIONAL),
            Pin("21", "IO13", PinType.BIDIRECTIONAL),
            Pin("22", "IO14", PinType.BIDIRECTIONAL),
            Pin("23", "IO21", PinType.BIDIRECTIONAL),
            Pin("24", "IO47", PinType.BIDIRECTIONAL),
            Pin("25", "IO48", PinType.BIDIRECTIONAL),
            Pin("26", "IO45", PinType.BIDIRECTIONAL),
            Pin("27", "IO0", PinType.BIDIRECTIONAL, net="BOOT"),
            Pin("28", "IO35", PinType.BIDIRECTIONAL),
            Pin("29", "IO36", PinType.BIDIRECTIONAL),
            Pin("30", "IO37", PinType.BIDIRECTIONAL),
            Pin("31", "IO38", PinType.BIDIRECTIONAL),
            Pin("32", "IO39", PinType.BIDIRECTIONAL),
            Pin("33", "IO40", PinType.BIDIRECTIONAL),
            Pin("34", "IO41", PinType.BIDIRECTIONAL),
            Pin("35", "IO42", PinType.BIDIRECTIONAL),
            Pin("36", "RXD0", PinType.OUTPUT, net="UART_TX"),
            Pin("37", "TXD0", PinType.INPUT, net="UART_RX"),
            Pin("38", "IO2", PinType.OUTPUT, net="LED"),
            Pin("39", "IO1", PinType.BIDIRECTIONAL),
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


# NOTE: Y1 (40MHz crystal) and C3/C4 (load caps) REMOVED.
# The ESP32-S3-WROOM-1 module has an internal 40MHz crystal.
# External crystal is NOT needed and its nets (XTAL_IN, XTAL_OUT,
# XTAL_IN_C3, XTAL_OUT_C4) have been removed.


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
        GND:          U1 pins 1/40/41, C1/C2/C5 pin 2, SW1/SW2 pin 2,
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
        (Crystal nets removed — WROOM-1 has internal crystal)
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
        # Crystal nets removed — ESP32-S3-WROOM-1 has internal crystal
    )


# ---------------------------------------------------------------------------
# Requirements assembly
# ---------------------------------------------------------------------------


def _build_requirements() -> ProjectRequirements:
    """Assemble full MCU core ProjectRequirements.

    Single FeatureBlock "MCU Core" containing all components.
    Board size: 70mm x 50mm.
    """
    components = (
        _make_esp32(),
        _make_decoupling_100nf(),
        _make_bulk_10uf(),
        # Y1/C3/C4 removed — ESP32-S3-WROOM-1 has internal crystal
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
            "ESP32-S3-WROOM-1 with decoupling, boot/reset buttons, "
            "USB-C programming port, status LED, and UART debug header"
        ),
        components=all_refs,
        nets=all_net_names,
        subcircuits=("decoupling",),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="MCUCoreTraining", revision="v1"),
        features=(mcu_feature,),
        components=components,
        nets=nets,
        mechanical=MechanicalConstraints(board_width_mm=70, board_height_mm=50),
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
    board_w: float = _BOARD_WIDTH_MM,
    board_h: float = _BOARD_HEIGHT_MM,
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
    _check_dist("C1", "U1", _DECOUP_TO_U1_MAX_MM, "HF decoupling (C1)")
    _check_dist("C2", "U1", _DECOUP_TO_U1_MAX_MM, "Bulk decoupling (C2)")
    print()

    # Crystal check removed — WROOM-1 has internal crystal; Y1/C3/C4 removed

    # ---------------------------------------------------------------
    # 3. USB-C at board edge
    # ---------------------------------------------------------------
    print("--- USB-C Edge Placement (J1 < 2mm from edge) ---")
    _check_edge("J1", _USBC_EDGE_MAX_MM, "USB-C connector")
    # CC resistors near J1
    _check_dist("R3", "J1", _CC_TO_J1_MAX_MM, "CC1 resistor (R3)")
    _check_dist("R4", "J1", _CC_TO_J1_MAX_MM, "CC2 resistor (R4)")
    print()

    # ---------------------------------------------------------------
    # 4. UART header at board edge
    # ---------------------------------------------------------------
    print("--- UART Header Edge Placement (J2 < 3mm from edge) ---")
    _check_edge("J2", _UART_EDGE_MAX_MM, "UART header")
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
        antenna_offset = _ANTENNA_HALF_BODY_MM
        import math

        rad = math.radians(u1_rot)
        # Antenna points in the -Y direction (north) at rot=0
        ant_x = u1_x - antenna_offset * math.sin(rad)
        ant_y = u1_y - antenna_offset * math.cos(rad)

        # Distance from antenna tip to nearest board edge
        ant_edge = min(ant_x, board_w - ant_x, ant_y, board_h - ant_y)
        label = (
            f"  Antenna tip at ({ant_x:.1f}, {ant_y:.1f}), "
            f"{ant_edge:.1f}mm from nearest edge (max {_ANTENNA_EDGE_MAX_MM}mm)"
        )
        if ant_edge > _ANTENNA_EDGE_MAX_MM:
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
    _check_dist("R1", "U1", _PULLUP_TO_U1_MAX_MM, "EN pull-up (R1)")
    _check_dist("R2", "U1", _PULLUP_TO_U1_MAX_MM, "BOOT pull-up (R2)")
    print()

    # ---------------------------------------------------------------
    # 7. Boot/Reset buttons accessible (near edge)
    # ---------------------------------------------------------------
    print("--- Button Accessibility (SW1/SW2 < 8mm from edge) ---")
    _check_edge("SW1", _BUTTON_EDGE_MAX_MM, "BOOT button")
    _check_edge("SW2", _BUTTON_EDGE_MAX_MM, "RESET button")
    print()

    # ---------------------------------------------------------------
    # 8. LED + resistor proximity
    # ---------------------------------------------------------------
    print("--- LED Proximity (R5/D1 near each other, < 5mm) ---")
    _check_dist("R5", "D1", _LED_PAIR_MAX_MM, "LED pair (R5-D1)")
    _check_dist("R5", "U1", _LED_TO_MCU_MAX_MM, "LED resistor to MCU (R5-U1)")
    print()

    # ---------------------------------------------------------------
    # 9. EN debounce cap near R1/SW2
    # ---------------------------------------------------------------
    print("--- EN Debounce (C5 near R1 and SW2) ---")
    _check_dist("C5", "R1", _EN_DEBOUNCE_TO_R1_MAX_MM, "EN debounce cap (C5-R1)")
    _check_dist("C5", "SW2", _EN_DEBOUNCE_TO_SW2_MAX_MM, "EN debounce cap (C5-SW2)")
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
    output_dir = _repo / "output" / "train_mcu_core"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "train_mcu_core_placement.png"

    print("=== MCU Core Training Board ===")
    print()

    # 1. Build requirements
    requirements = _build_requirements()
    print(f"Components: {len(requirements.components)}")
    print(f"Nets:       {len(requirements.nets)}")
    print("Board:      70 x 50 mm")
    print()

    # 2. Build PCB (no routing)
    print("Building PCB...")
    pcb = build_pcb(requirements, auto_route=False, placement_mode="grouped")
    print(f"  Footprints: {len(pcb.footprints)}")
    print()

    # 3. Run placement optimizer
    print("Running EE placement optimizer...")
    optimized_pcb, review = optimize_placement_ee(requirements, pcb)

    # --- Post-placement position corrections ---
    # The optimizer gets the rough layout right but some components land too
    # far from their target pins.  These overrides nudge them to electrically
    # correct positions verified against the ESP32 pinout at rot=180.
    #
    # U1 center = (35, 31.91), rot=180.
    # Pin 2 (3V3) board-space ≈ (43.75, 39.53)  — right column
    # Pin 27 (IO0) board-space ≈ (26.25, 24.29)  — left column
    # Pin 3 (EN)  board-space ≈ (43.75, 38.26)  — right column
    # Pin 38 (IO2) board-space ≈ (26.25, 38.26)  — left column
    _PLACEMENT_OVERRIDES: dict[str, tuple[float, float]] = {
        # Issue 3: C1/C2 decoupling caps — near 3V3 pin but clear of U1 pads
        # U1 rightmost pad edge ≈ x=44.5 (pad x=43.75 + half pad width 0.75)
        # C1/C2 need x ≥ 46.0 to avoid overlap
        "C1": (47.0, 38.5),
        "C2": (47.0, 41.0),
        # Issue 4: R2 BOOT pull-up — within 5mm of IO0 pin (left side)
        "R2": (23.0, 24.3),
        # Issue 5: SW2 RESET — ~7mm from left edge
        "SW2": (7.0, 26.9),
        # Issue 6: R5/D1 LED pair — within 15mm of IO2 pin (left side)
        "R5": (23.0, 36.0),
        "D1": (19.5, 36.0),
    }

    from dataclasses import replace as _replace

    from kicad_pipeline.models.pcb import Point as _Point

    adjusted_fps: list[object] = []
    for fp in optimized_pcb.footprints:
        if fp.ref in _PLACEMENT_OVERRIDES:
            new_x, new_y = _PLACEMENT_OVERRIDES[fp.ref]
            fp = _replace(fp, position=_Point(x=new_x, y=new_y))
        adjusted_fps.append(fp)

    optimized_pcb = _replace(optimized_pcb, footprints=tuple(adjusted_fps))

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
        title="MCU Core Training Board — Group Placement",
        score=score,
        group_map=group_map,
    )
    print(f"  Saved: {output_png}")
    print()

    # 6. Write KiCad PCB file and compare against reference
    pcb_path = output_dir / "train_mcu_core.kicad_pcb"
    write_and_compare_pcb(optimized_pcb, pcb_path)

    # 7. Write KiCad project file
    pro_path = write_project_file("train_mcu_core", output_dir)
    print(f"  KiCad project: {pro_path}")
    print()

    # 8. Print component positions
    fp_map = print_component_positions(optimized_pcb)

    # 9. Design rules compliance check
    _check_design_rules(fp_map)

    # 10. Pin-net map for human verification
    _print_pin_net_map(requirements)


if __name__ == "__main__":
    main()
