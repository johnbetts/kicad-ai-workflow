"""Tests for kicad_pipeline.pcb.netclasses."""

from __future__ import annotations

from kicad_pipeline.models.pcb import DesignRules, NetClass, NetEntry
from kicad_pipeline.pcb.netclasses import classify_nets, net_width_map

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


def test_classify_nets_voltage_rails() -> None:
    """Voltage rails like +3V3, +5V should be Power."""
    entries = (
        NetEntry(number=0, name=""),
        NetEntry(number=1, name="+3V3"),
        NetEntry(number=2, name="+5V"),
        NetEntry(number=3, name="VCC"),
        NetEntry(number=4, name="VDD"),
    )
    result = classify_nets(entries)
    power = next(nc for nc in result if nc.name == "Power")
    assert "+3V3" in power.nets
    assert "+5V" in power.nets
    assert "VCC" in power.nets
    assert "VDD" in power.nets


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
    assert names == {"Default", "Power", "I2C", "HighVoltageAnalog"}


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
