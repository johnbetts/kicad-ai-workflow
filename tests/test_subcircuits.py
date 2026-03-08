"""Tests for kicad_pipeline.schematic.subcircuits."""

from __future__ import annotations

import pytest

from kicad_pipeline.schematic.subcircuits import (
    SubcircuitResult,
    decoupling_cap,
    instantiate_subcircuit,
    ldo_regulator,
    led_drive,
    led_limit_resistor,
    merge_subcircuit_results,
    npn_buzzer_drive,
    relay_driver,
    usb_c_input,
    voltage_divider,
    voltage_divider_vout,
)

# ---------------------------------------------------------------------------
# voltage_divider
# ---------------------------------------------------------------------------


def test_voltage_divider_has_two_resistors() -> None:
    """voltage_divider() returns exactly 2 components."""
    result = voltage_divider("R1", "R2", "VIN", "VOUT")
    assert len(result.components) == 2


def test_voltage_divider_nets() -> None:
    """Net names VIN, VOUT, GND are present in voltage_divider output."""
    result = voltage_divider("R1", "R2", "VIN", "VOUT", gnd_net="GND")
    net_names = {n.name for n in result.nets}
    assert "VIN" in net_names
    assert "VOUT" in net_names
    assert "GND" in net_names


def test_voltage_divider_vout_calculation() -> None:
    """50/50 voltage divider produces Vout = Vin / 2."""
    vout = voltage_divider_vout(vin=5.0, r_top=10_000.0, r_bot=10_000.0)
    assert abs(vout - 2.5) < 1e-9


def test_voltage_divider_vout_calculation_asymmetric() -> None:
    """Asymmetric divider: 2k top / 1k bot → Vout = Vin / 3."""
    vout = voltage_divider_vout(vin=3.0, r_top=2_000.0, r_bot=1_000.0)
    assert abs(vout - 1.0) < 1e-9


def test_voltage_divider_refs_correct() -> None:
    """Component refs match the arguments supplied."""
    result = voltage_divider("RTOP", "RBOT", "VIN", "VMID")
    refs = {c.ref for c in result.components}
    assert "RTOP" in refs
    assert "RBOT" in refs


def test_voltage_divider_pin_nets() -> None:
    """Top resistor pin 1 connects to VIN, pin 2 connects to VOUT."""
    result = voltage_divider("R1", "R2", "VIN", "VOUT")
    r_top = next(c for c in result.components if c.ref == "R1")
    pin1 = r_top.get_pin("1")
    pin2 = r_top.get_pin("2")
    assert pin1 is not None and pin1.net == "VIN"
    assert pin2 is not None and pin2.net == "VOUT"


def test_voltage_divider_custom_gnd() -> None:
    """Custom gnd_net name propagates to net list."""
    result = voltage_divider("R1", "R2", "VIN", "VOUT", gnd_net="AGND")
    net_names = {n.name for n in result.nets}
    assert "AGND" in net_names


# ---------------------------------------------------------------------------
# decoupling_cap
# ---------------------------------------------------------------------------


def test_decoupling_cap_one_component() -> None:
    """decoupling_cap() returns exactly 1 component."""
    result = decoupling_cap("C1", "+3V3")
    assert len(result.components) == 1


def test_decoupling_cap_nets() -> None:
    """VCC and GND nets are present in decoupling_cap output."""
    result = decoupling_cap("C1", "+3V3", gnd_net="GND")
    net_names = {n.name for n in result.nets}
    assert "+3V3" in net_names
    assert "GND" in net_names


def test_decoupling_cap_default_value() -> None:
    """Default capacitor value is 100 nF."""
    result = decoupling_cap("C1", "VCC")
    cap = result.components[0]
    assert "100nF" in cap.value or "100" in cap.value


def test_decoupling_cap_custom_value() -> None:
    """Custom value_uf parameter is reflected in component value."""
    result = decoupling_cap("C1", "VCC", value_uf=10.0)
    cap = result.components[0]
    assert "10uF" in cap.value or "10" in cap.value


def test_decoupling_cap_ref() -> None:
    """Capacitor ref matches the argument supplied."""
    result = decoupling_cap("C99", "VCC")
    assert result.components[0].ref == "C99"


# ---------------------------------------------------------------------------
# led_drive
# ---------------------------------------------------------------------------


def test_led_drive_two_components() -> None:
    """led_drive() returns exactly 2 components (LED + resistor)."""
    result = led_drive("D1", "R3", "+3V3", "GPIO1")
    assert len(result.components) == 2


def test_led_drive_resistor_value() -> None:
    """Nominal R = (3.3 - 2.1) / 0.01 = 120 Ω for default parameters."""
    r_nominal = led_limit_resistor(vcc_v=3.3, vf_v=2.1, target_ma=10.0)
    assert abs(r_nominal - 120.0) < 1e-6


def test_led_drive_resistor_e24() -> None:
    """led_limit_resistor raises ValueError for non-positive current."""
    with pytest.raises(ValueError):
        led_limit_resistor(vcc_v=3.3, vf_v=2.1, target_ma=0.0)


def test_led_drive_nets() -> None:
    """Anode and GPIO nets appear in led_drive output."""
    result = led_drive("D1", "R3", "VCC", "GPIO4")
    net_names = {n.name for n in result.nets}
    assert "VCC" in net_names
    assert "GPIO4" in net_names


def test_led_drive_led_ref() -> None:
    """LED component ref matches ref_led argument."""
    result = led_drive("D_STATUS", "R_LED", "+3V3", "GPIO0")
    refs = {c.ref for c in result.components}
    assert "D_STATUS" in refs
    assert "R_LED" in refs


# ---------------------------------------------------------------------------
# ldo_regulator
# ---------------------------------------------------------------------------


def test_ldo_three_components() -> None:
    """ldo_regulator() returns LDO + 2 caps = 3 components total."""
    result = ldo_regulator("U2", "C10", "C11", "+5V", "+3V3")
    assert len(result.components) == 3


def test_ldo_nets() -> None:
    """Input, output, and GND nets are present in ldo_regulator output."""
    result = ldo_regulator("U2", "C10", "C11", "VIN", "VOUT")
    net_names = {n.name for n in result.nets}
    assert "VIN" in net_names
    assert "VOUT" in net_names
    assert "GND" in net_names


def test_ldo_refs() -> None:
    """All three refs are present in ldo_regulator output."""
    result = ldo_regulator("U2", "C10", "C11", "VIN", "VOUT")
    refs = {c.ref for c in result.components}
    assert "U2" in refs
    assert "C10" in refs
    assert "C11" in refs


# ---------------------------------------------------------------------------
# usb_c_input
# ---------------------------------------------------------------------------


def test_usb_c_three_components() -> None:
    """usb_c_input() returns connector + 2 CC resistors = 3 components."""
    result = usb_c_input("J1", "R_CC1", "R_CC2")
    assert len(result.components) == 3


def test_usb_c_nets() -> None:
    """VBUS, GND, USB_DP, USB_DM, CC1, CC2 nets are present."""
    result = usb_c_input("J1", "R_CC1", "R_CC2")
    net_names = {n.name for n in result.nets}
    assert "VBUS" in net_names
    assert "GND" in net_names
    assert "USB_DP" in net_names
    assert "USB_DM" in net_names
    assert "USB_CC1" in net_names
    assert "USB_CC2" in net_names


def test_usb_c_cc_resistor_refs() -> None:
    """CC resistor refs match the arguments supplied."""
    result = usb_c_input("J_USB", "R5", "R6")
    refs = {c.ref for c in result.components}
    assert "J_USB" in refs
    assert "R5" in refs
    assert "R6" in refs


# ---------------------------------------------------------------------------
# npn_buzzer_drive
# ---------------------------------------------------------------------------


def test_npn_buzzer_three_components() -> None:
    """npn_buzzer_drive() returns transistor + base resistor + diode = 3 components."""
    result = npn_buzzer_drive("Q1", "R_BASE", "D_FLY", "BUZZER", "GPIO2", "VCC")
    assert len(result.components) == 3


def test_npn_buzzer_nets() -> None:
    """GPIO, buzzer, VCC, and GND nets are present in npn_buzzer_drive output."""
    result = npn_buzzer_drive("Q1", "R_BASE", "D_FLY", "BUZZER", "GPIO2", "VCC")
    net_names = {n.name for n in result.nets}
    assert "GPIO2" in net_names
    assert "BUZZER" in net_names
    assert "VCC" in net_names
    assert "GND" in net_names


def test_npn_buzzer_refs() -> None:
    """All three component refs are present."""
    result = npn_buzzer_drive("Q1", "R_BASE", "D_FLY", "BUZZER", "GPIO2", "VCC")
    refs = {c.ref for c in result.components}
    assert "Q1" in refs
    assert "R_BASE" in refs
    assert "D_FLY" in refs


# ---------------------------------------------------------------------------
# relay_driver
# ---------------------------------------------------------------------------


def test_relay_driver_five_components_with_terminal() -> None:
    """relay_driver() with terminal returns R + Q + D + K + J = 5 components."""
    result = relay_driver("Q1", "R10", "D6", "K1", "J3", "RELAY_1")
    assert len(result.components) == 5


def test_relay_driver_four_components_without_terminal() -> None:
    """relay_driver() without terminal returns R + Q + D + K = 4 components."""
    result = relay_driver("Q1", "R10", "D6", "K1", None, "RELAY_1")
    assert len(result.components) == 4


def test_relay_driver_nets() -> None:
    """GPIO, VCC, GND, base, coil, and contact nets are present."""
    result = relay_driver("Q1", "R10", "D6", "K1", "J3", "RELAY_1")
    net_names = {n.name for n in result.nets}
    assert "RELAY_1" in net_names
    assert "Q1_BASE" in net_names
    assert "K1_COIL" in net_names
    assert "+5V" in net_names
    assert "GND" in net_names
    assert "K1_COM" in net_names
    assert "K1_NO" in net_names
    assert "K1_NC" in net_names


def test_relay_driver_refs() -> None:
    """All component refs are present in relay_driver output."""
    result = relay_driver("Q1", "R10", "D6", "K1", "J3", "RELAY_1")
    refs = {c.ref for c in result.components}
    assert refs == {"Q1", "R10", "D6", "K1", "J3"}


def test_relay_driver_spst() -> None:
    """SPST relay has no NC net and 4 relay pins."""
    result = relay_driver("Q2", "R11", "D7", "K2", "J4", "RELAY_2", relay_type="SPST")
    net_names = {n.name for n in result.nets}
    assert "K2_COM" in net_names
    assert "K2_NO" in net_names
    assert "K2_NC" not in net_names
    relay = next(c for c in result.components if c.ref == "K2")
    assert len(relay.pins) == 4


def test_relay_driver_custom_vcc() -> None:
    """Custom vcc_net propagates to diode and relay coil connections."""
    result = relay_driver("Q1", "R10", "D6", "K1", None, "GPIO1", vcc_net="+12V")
    net_names = {n.name for n in result.nets}
    assert "+12V" in net_names
    diode = next(c for c in result.components if c.ref == "D6")
    assert diode.get_pin("2") is not None
    assert diode.get_pin("2").net == "+12V"


def test_relay_driver_terminal_connects_contacts() -> None:
    """Screw terminal pins share nets with relay contact pins."""
    result = relay_driver("Q1", "R10", "D6", "K1", "J3", "RELAY_1")
    # J3 pin 1 should connect to K1_NO, pin 2 to K1_COM, pin 3 to K1_NC
    terminal = next(c for c in result.components if c.ref == "J3")
    assert terminal.get_pin("1").net == "K1_NO"
    assert terminal.get_pin("2").net == "K1_COM"
    assert terminal.get_pin("3").net == "K1_NC"


def test_relay_driver_description() -> None:
    """Description contains key component info."""
    result = relay_driver("Q1", "R10", "D6", "K1", "J3", "RELAY_1")
    assert "BC817" in result.description
    assert "1N4148" in result.description
    assert "SPDT" in result.description


# ---------------------------------------------------------------------------
# instantiate_subcircuit
# ---------------------------------------------------------------------------


def test_instantiate_relay_driver_x4() -> None:
    """Instantiate 4 relay drivers with correct ref numbering."""
    results = instantiate_subcircuit(
        relay_driver,
        count=4,
        ref_start={"Q": 1, "R": 10, "D": 6, "K": 1, "J": 3},
        net_prefix="RELAY",
        vcc_net="+5V",
    )
    assert len(results) == 4

    # Check refs for each instance
    for i, r in enumerate(results, 1):
        refs = {c.ref for c in r.components}
        assert f"Q{i}" in refs
        assert f"R{9 + i}" in refs
        assert f"D{5 + i}" in refs
        assert f"K{i}" in refs
        assert f"J{2 + i}" in refs

    # Check net naming
    net_names = {n.name for n in results[0].nets}
    assert "RELAY_1" in net_names
    net_names_4 = {n.name for n in results[3].nets}
    assert "RELAY_4" in net_names_4


def test_instantiate_without_terminal() -> None:
    """Instantiate relay drivers without screw terminals."""
    results = instantiate_subcircuit(
        relay_driver,
        count=2,
        ref_start={"Q": 1, "R": 10, "D": 6, "K": 1},
        net_prefix="RLY",
        vcc_net="+5V",
    )
    # J prefix not in ref_start → ref_j = None → no terminal
    for r in results:
        assert len(r.components) == 4
        assert not any(c.ref.startswith("J") for c in r.components)


# ---------------------------------------------------------------------------
# merge_subcircuit_results
# ---------------------------------------------------------------------------


def test_merge_combines_components() -> None:
    """merge_subcircuit_results combines components from all inputs."""
    r1 = relay_driver("Q1", "R10", "D6", "K1", "J3", "RELAY_1")
    r2 = relay_driver("Q2", "R11", "D7", "K2", "J4", "RELAY_2")
    merged = merge_subcircuit_results((r1, r2))
    assert len(merged.components) == len(r1.components) + len(r2.components)


def test_merge_deduplicates_shared_nets() -> None:
    """Shared nets (GND, +5V) are merged into single entries."""
    r1 = relay_driver("Q1", "R10", "D6", "K1", None, "RELAY_1")
    r2 = relay_driver("Q2", "R11", "D7", "K2", None, "RELAY_2")
    merged = merge_subcircuit_results((r1, r2))

    # GND should appear once, with connections from both instances
    gnd_nets = [n for n in merged.nets if n.name == "GND"]
    assert len(gnd_nets) == 1
    assert len(gnd_nets[0].connections) == 2  # Q1.E + Q2.E


def test_merge_preserves_descriptions() -> None:
    """Merged description contains all individual descriptions."""
    r1 = decoupling_cap("C1", "VCC")
    r2 = decoupling_cap("C2", "+3V3")
    merged = merge_subcircuit_results((r1, r2))
    assert r1.description in merged.description
    assert r2.description in merged.description


# ---------------------------------------------------------------------------
# SubcircuitResult immutability
# ---------------------------------------------------------------------------


def test_subcircuit_result_is_frozen() -> None:
    """SubcircuitResult is immutable (frozen dataclass)."""
    result = decoupling_cap("C1", "VCC")
    with pytest.raises((AttributeError, TypeError)):
        result.description = "mutated"  # type: ignore[misc]


def test_subcircuit_result_has_description() -> None:
    """SubcircuitResult.description is a non-empty string."""
    result = voltage_divider("R1", "R2", "VIN", "VOUT")
    assert isinstance(result.description, str)
    assert len(result.description) > 0


# ---------------------------------------------------------------------------
# voltage_divider_vout edge cases
# ---------------------------------------------------------------------------


def test_voltage_divider_vout_zero_top() -> None:
    """Zero top resistor → output equals input voltage."""
    vout = voltage_divider_vout(vin=5.0, r_top=0.0, r_bot=10_000.0)
    assert abs(vout - 5.0) < 1e-9


def test_voltage_divider_vout_degenerate() -> None:
    """Both resistors zero raises ValueError."""
    with pytest.raises(ValueError):
        voltage_divider_vout(vin=5.0, r_top=0.0, r_bot=0.0)
