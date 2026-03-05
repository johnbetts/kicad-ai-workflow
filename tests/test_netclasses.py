"""Tests for kicad_pipeline.pcb.netclasses."""

from __future__ import annotations

from kicad_pipeline.constants import JLCPCB_MIN_TRACE_MM
from kicad_pipeline.models.pcb import DesignRules, NetClass, NetEntry
from kicad_pipeline.pcb.netclasses import (
    _current_trace_width,
    _parse_voltage,
    _voltage_clearance,
    _voltage_label,
    classify_nets,
    net_width_map,
)

# ---------------------------------------------------------------------------
# _parse_voltage
# ---------------------------------------------------------------------------


def test_parse_voltage_3v3() -> None:
    """+3V3 should parse to 3.3 volts."""
    assert _parse_voltage("+3V3") == 3.3


def test_parse_voltage_5v() -> None:
    """+5V should parse to 5.0 volts."""
    assert _parse_voltage("+5V") == 5.0


def test_parse_voltage_12v() -> None:
    """+12V should parse to 12.0 volts."""
    assert _parse_voltage("+12V") == 12.0


def test_parse_voltage_dot_notation() -> None:
    """+3.3V should parse to 3.3 volts."""
    assert _parse_voltage("+3.3V") == 3.3


def test_parse_voltage_gnd() -> None:
    """GND should parse to 0.0 volts."""
    assert _parse_voltage("GND") == 0.0


def test_parse_voltage_vdd() -> None:
    """VDD should return None (unknown voltage)."""
    assert _parse_voltage("VDD") is None


def test_parse_voltage_vcc() -> None:
    """VCC should return None (unknown voltage)."""
    assert _parse_voltage("VCC") is None


def test_parse_voltage_vbus() -> None:
    """VBUS should return None (unknown voltage)."""
    assert _parse_voltage("VBUS") is None


# ---------------------------------------------------------------------------
# _voltage_clearance
# ---------------------------------------------------------------------------


def test_voltage_clearance_low() -> None:
    """5V is well under 50V threshold -> 0.2mm clearance."""
    assert _voltage_clearance(5.0) == 0.2


def test_voltage_clearance_medium() -> None:
    """100V -> 0.5mm clearance."""
    assert _voltage_clearance(100.0) == 0.5


def test_voltage_clearance_high() -> None:
    """250V -> 1.0mm clearance."""
    assert _voltage_clearance(250.0) == 1.0


def test_voltage_clearance_above_all() -> None:
    """Above all thresholds -> last clearance value (2.5mm)."""
    assert _voltage_clearance(1000.0) == 2.5


# ---------------------------------------------------------------------------
# _current_trace_width
# ---------------------------------------------------------------------------


def test_current_trace_width_1a() -> None:
    """1A should require a trace wider than JLCPCB minimum."""
    width = _current_trace_width(1.0)
    assert width > JLCPCB_MIN_TRACE_MM


def test_current_trace_width_zero() -> None:
    """Zero current should return JLCPCB minimum trace width."""
    assert _current_trace_width(0.0) == JLCPCB_MIN_TRACE_MM


def test_current_trace_width_negative() -> None:
    """Negative current should return JLCPCB minimum trace width."""
    assert _current_trace_width(-1.0) == JLCPCB_MIN_TRACE_MM


def test_current_trace_width_increases_with_current() -> None:
    """Higher current should require wider trace."""
    w1 = _current_trace_width(1.0)
    w2 = _current_trace_width(3.0)
    assert w2 > w1


# ---------------------------------------------------------------------------
# _voltage_label
# ---------------------------------------------------------------------------


def test_voltage_label_3v3() -> None:
    """+3V3 -> '3V3'."""
    assert _voltage_label("+3V3") == "3V3"


def test_voltage_label_5v() -> None:
    """+5V -> '5V'."""
    assert _voltage_label("+5V") == "5V"


def test_voltage_label_dot_notation() -> None:
    """+3.3V -> '3V3'."""
    assert _voltage_label("+3.3V") == "3V3"


# ---------------------------------------------------------------------------
# classify_nets
# ---------------------------------------------------------------------------


def test_classify_nets_empty() -> None:
    """No nets produces a single Default class with no nets."""
    result = classify_nets(())
    assert len(result) == 1
    assert result[0].name == "Default"
    assert result[0].nets == ()


def test_classify_nets_skips_empty_net() -> None:
    """Net 0 (empty name) should be skipped."""
    entries = (NetEntry(number=0, name=""),)
    result = classify_nets(entries)
    assert len(result) == 1
    assert result[0].name == "Default"
    assert result[0].nets == ()


def test_classify_nets_gnd_is_power() -> None:
    """GND net should be classified as Power."""
    entries = (
        NetEntry(number=0, name=""),
        NetEntry(number=1, name="GND"),
    )
    result = classify_nets(entries)
    names = {nc.name for nc in result}
    assert "Power" in names
    power = next(nc for nc in result if nc.name == "Power")
    assert "GND" in power.nets
    assert power.trace_width_mm == 0.3
    assert power.clearance_mm == 0.2


def test_classify_nets_splits_power_voltages() -> None:
    """Voltage rails +3V3 and +5V should get different netclass names."""
    entries = (
        NetEntry(number=0, name=""),
        NetEntry(number=1, name="+3V3"),
        NetEntry(number=2, name="+5V"),
    )
    result = classify_nets(entries)
    names = {nc.name for nc in result}
    assert "Power_3V3" in names
    assert "Power_5V" in names

    p3v3 = next(nc for nc in result if nc.name == "Power_3V3")
    assert "+3V3" in p3v3.nets
    assert p3v3.clearance_mm == 0.2  # 3.3V < 50V threshold

    p5v = next(nc for nc in result if nc.name == "Power_5V")
    assert "+5V" in p5v.nets
    assert p5v.clearance_mm == 0.2  # 5V < 50V threshold


def test_classify_nets_voltage_rails_and_unknown() -> None:
    """Known voltage rails get subclasses, VCC/VDD stay in generic Power."""
    entries = (
        NetEntry(number=0, name=""),
        NetEntry(number=1, name="+3V3"),
        NetEntry(number=2, name="+5V"),
        NetEntry(number=3, name="VCC"),
        NetEntry(number=4, name="VDD"),
    )
    result = classify_nets(entries)
    names = {nc.name for nc in result}
    assert "Power_3V3" in names
    assert "Power_5V" in names
    assert "Power" in names  # VCC + VDD

    power = next(nc for nc in result if nc.name == "Power")
    assert "VCC" in power.nets
    assert "VDD" in power.nets
    assert "+3V3" not in power.nets
    assert "+5V" not in power.nets


def test_classify_nets_high_voltage_analog() -> None:
    """SENS*, AIN*, ADC*, VREF nets should be HighVoltageAnalog."""
    entries = (
        NetEntry(number=0, name=""),
        NetEntry(number=1, name="SENS_IN"),
        NetEntry(number=2, name="AIN0"),
        NetEntry(number=3, name="ADC_CH1"),
        NetEntry(number=4, name="VREF"),
    )
    result = classify_nets(entries)
    hva = next(nc for nc in result if nc.name == "HighVoltageAnalog")
    assert "SENS_IN" in hva.nets
    assert "AIN0" in hva.nets
    assert "ADC_CH1" in hva.nets
    assert "VREF" in hva.nets
    assert hva.clearance_mm == 0.2
    assert hva.trace_width_mm == 0.4


def test_classify_nets_i2c() -> None:
    """I2C nets should be classified as I2C."""
    entries = (
        NetEntry(number=0, name=""),
        NetEntry(number=1, name="SDA"),
        NetEntry(number=2, name="SCL"),
        NetEntry(number=3, name="I2C_SDA"),
    )
    result = classify_nets(entries)
    i2c = next(nc for nc in result if nc.name == "I2C")
    assert "SDA" in i2c.nets
    assert "SCL" in i2c.nets
    assert "I2C_SDA" in i2c.nets


def test_classify_nets_spi() -> None:
    """SPI nets should be classified as SPI."""
    entries = (
        NetEntry(number=0, name=""),
        NetEntry(number=1, name="MOSI"),
        NetEntry(number=2, name="MISO"),
        NetEntry(number=3, name="SCLK"),
        NetEntry(number=4, name="CS"),
    )
    result = classify_nets(entries)
    spi = next(nc for nc in result if nc.name == "SPI")
    assert "MOSI" in spi.nets
    assert "MISO" in spi.nets
    assert "SCLK" in spi.nets
    assert "CS" in spi.nets
    assert spi.trace_width_mm == 0.2


def test_classify_nets_rf_class() -> None:
    """RF/antenna nets should be classified as RF."""
    entries = (
        NetEntry(number=0, name=""),
        NetEntry(number=1, name="ANT_2G4"),
        NetEntry(number=2, name="RF_OUT"),
        NetEntry(number=3, name="WIFI_TX"),
        NetEntry(number=4, name="BLE_DATA"),
    )
    result = classify_nets(entries)
    names = {nc.name for nc in result}
    assert "RF" in names
    rf = next(nc for nc in result if nc.name == "RF")
    assert "ANT_2G4" in rf.nets
    assert "RF_OUT" in rf.nets
    assert "WIFI_TX" in rf.nets
    assert "BLE_DATA" in rf.nets
    assert rf.trace_width_mm == 0.3
    assert rf.clearance_mm == 0.3


def test_classify_nets_unmatched_goes_to_default() -> None:
    """Nets that match no pattern go into Default."""
    entries = (
        NetEntry(number=0, name=""),
        NetEntry(number=1, name="LED_EN"),
        NetEntry(number=2, name="RESET"),
    )
    result = classify_nets(entries)
    default = next(nc for nc in result if nc.name == "Default")
    assert "LED_EN" in default.nets
    assert "RESET" in default.nets


def test_classify_nets_respects_design_rules() -> None:
    """Default class should use design_rules values when provided."""
    entries = (
        NetEntry(number=0, name=""),
        NetEntry(number=1, name="SIG"),
    )
    rules = DesignRules(default_trace_width_mm=0.3, default_clearance_mm=0.25)
    result = classify_nets(entries, design_rules=rules)
    default = next(nc for nc in result if nc.name == "Default")
    assert default.trace_width_mm == 0.3
    assert default.clearance_mm == 0.25


def test_classify_nets_mixed() -> None:
    """Mixed net types should produce multiple classes."""
    entries = (
        NetEntry(number=0, name=""),
        NetEntry(number=1, name="GND"),
        NetEntry(number=2, name="+3V3"),
        NetEntry(number=3, name="SDA"),
        NetEntry(number=4, name="SENS_0_30V"),
        NetEntry(number=5, name="LED_EN"),
    )
    result = classify_nets(entries)
    names = {nc.name for nc in result}
    assert "Default" in names
    assert "Power" in names  # GND
    assert "Power_3V3" in names  # +3V3
    assert "I2C" in names
    assert "HighVoltageAnalog" in names


# ---------------------------------------------------------------------------
# net_width_map
# ---------------------------------------------------------------------------


def test_net_width_map_basic() -> None:
    """net_width_map should map each net to its class trace width."""
    classes = (
        NetClass(name="Default", trace_width_mm=0.25, nets=("SIG",)),
        NetClass(name="Power", trace_width_mm=0.5, nets=("GND", "+3V3")),
    )
    widths = net_width_map(classes)
    assert widths["SIG"] == 0.25
    assert widths["GND"] == 0.5
    assert widths["+3V3"] == 0.5


def test_net_width_map_empty() -> None:
    """Empty netclasses should produce empty map."""
    assert net_width_map(()) == {}


def test_netclass_frozen() -> None:
    """NetClass should be immutable."""
    nc = NetClass(name="Test", nets=("A",))
    import pytest

    with pytest.raises(AttributeError):
        nc.name = "Changed"  # type: ignore[misc]
