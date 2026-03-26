"""Tests for board configuration inference and serialisation."""

from __future__ import annotations

import json

from kicad_pipeline.config.board_config_generator import (
    _parse_resistance,
    generate_board_config,
)
from kicad_pipeline.config.serializers import (
    board_config_to_json,
    board_config_to_markdown,
)
from kicad_pipeline.models.board_config import (
    BusType,
    RelayPolarity,
)
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    Net,
    NetConnection,
    Pin,
    PinFunction,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_requirements(
    components: tuple[Component, ...] = (),
    nets: tuple[Net, ...] = (),
) -> ProjectRequirements:
    """Build minimal ProjectRequirements for testing."""
    return ProjectRequirements(
        project=ProjectInfo(name="TestBoard"),
        features=(
            FeatureBlock("Test", "test block", tuple(c.ref for c in components), (), ()),
        ),
        components=components,
        nets=nets,
    )


def _relay_circuit(
    num: int,
    com_net: str,
    output_net: str,
    gpio_net: str,
) -> tuple[tuple[Component, ...], tuple[Net, ...]]:
    """Build a complete relay driver circuit: R → Q → K with flyback D."""
    q_base = f"Q{num}_BASE"
    coil_net = f"K{num}_COIL"

    r = Component(
        ref=f"R{num}", value="1k", footprint="R_0402",
        pins=(
            Pin("1", "~", PinType.PASSIVE, net=gpio_net),
            Pin("2", "~", PinType.PASSIVE, net=q_base),
        ),
    )
    q = Component(
        ref=f"Q{num}", value="BC817", footprint="SOT-23",
        pins=(
            Pin("1", "B", PinType.INPUT, net=q_base),
            Pin("2", "C", PinType.PASSIVE, net=coil_net),
            Pin("3", "E", PinType.PASSIVE, net="GND"),
        ),
    )
    d = Component(
        ref=f"D{num}", value="1N4148", footprint="SOD-123",
        pins=(
            Pin("1", "A", PinType.PASSIVE, net=coil_net),
            Pin("2", "K", PinType.PASSIVE, net="+5V"),
        ),
    )
    k = Component(
        ref=f"K{num}", value="SRD-05VDC-SL-C", footprint="Relay_SPDT",
        description=f"Relay {num}",
        pins=(
            Pin("1", "COM", PinType.PASSIVE, net=com_net),
            Pin("2", "COIL-", PinType.PASSIVE, net=coil_net),
            Pin("3", "NO", PinType.PASSIVE, net=output_net),
            Pin("4", "NC", PinType.PASSIVE),
            Pin("5", "COIL+", PinType.PASSIVE, net="+5V"),
        ),
    )

    nets = (
        Net(gpio_net, (NetConnection(r.ref, "1"),)),
        Net(q_base, (NetConnection(r.ref, "2"), NetConnection(q.ref, "1"))),
        Net(coil_net, (
            NetConnection(q.ref, "2"),
            NetConnection(d.ref, "1"),
            NetConnection(k.ref, "2"),
        )),
        Net(com_net, (NetConnection(k.ref, "1"),)),
        Net(output_net, (NetConnection(k.ref, "3"),)),
        Net("+5V", (
            NetConnection(d.ref, "2"),
            NetConnection(k.ref, "5"),
        )),
        Net("GND", (NetConnection(q.ref, "3"),)),
    )
    return (r, q, d, k), nets


def _mcu_component(gpio_nets: dict[str, str]) -> Component:
    """Build a minimal MCU with GPIO pins on given nets."""
    pins: list[Pin] = [
        Pin("1", "VCC", PinType.POWER_IN, net="+3V3"),
        Pin("2", "GND", PinType.POWER_IN, net="GND"),
    ]
    for i, (name, net) in enumerate(gpio_nets.items(), start=3):
        pins.append(Pin(str(i), name, PinType.BIDIRECTIONAL, PinFunction.GPIO, net))
    return Component(
        ref="U1", value="ESP32-S3-WROOM-1", footprint="ESP32-S3-WROOM-1",
        description="MCU",
        pins=tuple(pins),
    )


# ---------------------------------------------------------------------------
# Relay inference tests
# ---------------------------------------------------------------------------


class TestRelayInference:
    """Test relay polarity detection from net tracing."""

    def test_npn_driver_is_active_high(self) -> None:
        """NPN driver (emitter→GND) should be ACTIVE_HIGH regardless of COM net."""
        comps, nets = _relay_circuit(1, "VIN", "HARNESS_1", "RELAY_1")
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        assert len(config.relays) == 1
        assert config.relays[0].polarity == RelayPolarity.ACTIVE_HIGH

    def test_npn_driver_gnd_com_still_active_high(self) -> None:
        """NPN driver with COM→GND is still ACTIVE_HIGH — COM net is not polarity."""
        comps, nets = _relay_circuit(1, "GND", "HARNESS_1", "RELAY_1")
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        assert len(config.relays) == 1
        assert config.relays[0].polarity == RelayPolarity.ACTIVE_HIGH
        assert config.relays[0].com_net == "GND"

    def test_relay_driver_chain(self) -> None:
        """Relay should trace: K → Q driver → R → GPIO net."""
        comps, nets = _relay_circuit(1, "VIN", "OUT", "GPIO_1")
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        relay = config.relays[0]
        assert relay.driver_ref == "Q1"
        assert relay.gpio_net == "GPIO_1"
        assert relay.coil_voltage_net == "+5V"

    def test_relay_output_net(self) -> None:
        """NO pin net should be captured as output_net."""
        comps, nets = _relay_circuit(1, "VIN", "HARNESS_PREHEAT", "R1_GPIO")
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        assert config.relays[0].output_net == "HARNESS_PREHEAT"

    def test_no_driver_warns_unknown_polarity(self) -> None:
        """Relay with no Q transistor should warn and have UNKNOWN polarity."""
        # Build relay without transistor driver
        k = Component(
            ref="K1", value="SRD-05VDC-SL-C", footprint="Relay_SPDT",
            description="Relay 1",
            pins=(
                Pin("1", "COM", PinType.PASSIVE, net="VIN"),
                Pin("2", "COIL-", PinType.PASSIVE, net="K1_COIL"),
                Pin("3", "NO", PinType.PASSIVE, net="OUT"),
                Pin("4", "NC", PinType.PASSIVE),
                Pin("5", "COIL+", PinType.PASSIVE, net="+5V"),
            ),
        )
        nets = (
            Net("VIN", (NetConnection("K1", "1"),)),
            Net("K1_COIL", (NetConnection("K1", "2"),)),
            Net("OUT", (NetConnection("K1", "3"),)),
            Net("+5V", (NetConnection("K1", "5"),)),
        )
        reqs = _make_requirements((k,), nets)
        config = generate_board_config(reqs)
        assert config.relays[0].polarity == RelayPolarity.UNKNOWN
        assert any("no transistor" in w.lower() for w in config.relays[0].warnings)

    def test_multiple_relays_sorted(self) -> None:
        """Multiple relays should be sorted by ref."""
        c1, n1 = _relay_circuit(2, "GND", "OUT2", "GPIO_2")
        c2, n2 = _relay_circuit(1, "VIN", "OUT1", "GPIO_1")
        all_comps = c1 + c2
        # Merge nets by name
        net_map: dict[str, set[tuple[str, str]]] = {}
        for net in n1 + n2:
            conns = net_map.setdefault(net.name, set())
            for conn in net.connections:
                conns.add((conn.ref, conn.pin))
        all_nets = tuple(
            Net(name, tuple(NetConnection(r, p) for r, p in conns))
            for name, conns in net_map.items()
        )
        reqs = _make_requirements(all_comps, all_nets)
        config = generate_board_config(reqs)
        assert len(config.relays) == 2
        assert config.relays[0].relay_ref == "K1"
        assert config.relays[1].relay_ref == "K2"

    def test_gpio_traces_to_mcu_pin(self) -> None:
        """GPIO net should resolve to MCU pin name when MCU is present."""
        comps, nets = _relay_circuit(1, "VIN", "OUT", "RELAY_1")
        mcu = _mcu_component({"IO15": "RELAY_1"})
        all_comps = (*comps, mcu)
        # Add MCU to RELAY_1 net
        net_map: dict[str, set[tuple[str, str]]] = {}
        for net in nets:
            conns = net_map.setdefault(net.name, set())
            for conn in net.connections:
                conns.add((conn.ref, conn.pin))
        net_map.setdefault("RELAY_1", set()).add(("U1", "3"))
        net_map.setdefault("+3V3", set()).add(("U1", "1"))
        net_map.setdefault("GND", set()).add(("U1", "2"))
        all_nets = tuple(
            Net(name, tuple(NetConnection(r, p) for r, p in conns))
            for name, conns in net_map.items()
        )
        reqs = _make_requirements(all_comps, all_nets)
        config = generate_board_config(reqs)
        assert config.relays[0].gpio_mcu_pin == "IO15"


# ---------------------------------------------------------------------------
# ADC inference tests
# ---------------------------------------------------------------------------


class TestADCInference:
    """Test ADC channel detection from ADS1115 components."""

    def _adc_circuit(self) -> tuple[tuple[Component, ...], tuple[Net, ...]]:
        """Build ADS1115 + voltage divider on AIN0."""
        adc = Component(
            ref="U4", value="ADS1115", footprint="MSOP-10",
            description="16-bit ADC (0x48)",
            pins=(
                Pin("1", "ADDR", PinType.INPUT, net="AGND"),
                Pin("2", "ALERT", PinType.OUTPUT, net="ADC1_ALERT"),
                Pin("3", "GND", PinType.POWER_IN, net="AGND"),
                Pin("4", "AIN0", PinType.INPUT, net="ADC_COOLANT"),
                Pin("5", "AIN1", PinType.INPUT, net="ADC_OIL"),
                Pin("8", "SDA", PinType.BIDIRECTIONAL, net="I2C_SDA"),
                Pin("9", "SCL", PinType.BIDIRECTIONAL, net="I2C_SCL"),
                Pin("10", "VDD", PinType.POWER_IN, net="AVCC"),
            ),
        )
        r_top = Component(
            ref="R14", value="100k", footprint="R_0402",
            description="ADC Coolant temp top divider",
            pins=(
                Pin("1", "~", PinType.PASSIVE, net="VIN_ADC1"),
                Pin("2", "~", PinType.PASSIVE, net="ADC_COOLANT"),
            ),
        )
        r_bot = Component(
            ref="R15", value="12k", footprint="R_0402",
            description="ADC Coolant temp bottom divider",
            pins=(
                Pin("1", "~", PinType.PASSIVE, net="ADC_COOLANT"),
                Pin("2", "~", PinType.PASSIVE, net="AGND"),
            ),
        )
        j1 = Component(
            ref="J1", value="Screw_Terminal", footprint="TerminalBlock_01x02",
            pins=(
                Pin("1", "VIN", PinType.PASSIVE, net="VIN_ADC1"),
                Pin("2", "GND", PinType.PASSIVE, net="AGND"),
            ),
        )
        nets = (
            Net("AGND", (
                NetConnection("U4", "1"), NetConnection("U4", "3"),
                NetConnection("R15", "2"), NetConnection("J1", "2"),
            )),
            Net("ADC_COOLANT", (
                NetConnection("U4", "4"),
                NetConnection("R14", "2"), NetConnection("R15", "1"),
            )),
            Net("ADC_OIL", (NetConnection("U4", "5"),)),
            Net("VIN_ADC1", (
                NetConnection("R14", "1"), NetConnection("J1", "1"),
            )),
            Net("AVCC", (NetConnection("U4", "10"),)),
            Net("I2C_SDA", (NetConnection("U4", "8"),)),
            Net("I2C_SCL", (NetConnection("U4", "9"),)),
            Net("ADC1_ALERT", (NetConnection("U4", "2"),)),
        )
        return (adc, r_top, r_bot, j1), nets

    def test_adc_channel_detected(self) -> None:
        """ADS1115 AIN0 should be detected as channel 0."""
        comps, nets = self._adc_circuit()
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        ch0 = [c for c in config.adc_channels if c.channel == 0]
        assert len(ch0) == 1
        assert ch0[0].adc_ref == "U4"
        assert ch0[0].adc_pin_net == "ADC_COOLANT"

    def test_i2c_address_from_addr_pin(self) -> None:
        """ADDR→AGND should give I2C address 0x48."""
        comps, nets = self._adc_circuit()
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        ch0 = [c for c in config.adc_channels if c.channel == 0]
        assert ch0[0].i2c_address == 0x48

    def test_divider_ratio_calculated(self) -> None:
        """100k/12k divider should give ratio ~0.1071."""
        comps, nets = self._adc_circuit()
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        ch0 = [c for c in config.adc_channels if c.channel == 0]
        assert ch0[0].divider_ratio is not None
        assert abs(ch0[0].divider_ratio - 12000 / 112000) < 0.001

    def test_max_input_voltage(self) -> None:
        """Max Vin = 3.3V / divider_ratio."""
        comps, nets = self._adc_circuit()
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        ch0 = [c for c in config.adc_channels if c.channel == 0]
        assert ch0[0].max_input_voltage is not None
        assert ch0[0].max_input_voltage > 30.0  # ~30.8V for 100k/12k

    def test_connector_ref_found(self) -> None:
        """Connector on the input net should be detected."""
        comps, nets = self._adc_circuit()
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        ch0 = [c for c in config.adc_channels if c.channel == 0]
        assert ch0[0].connector_ref == "J1"

    def test_description_from_divider_label(self) -> None:
        """Description should be extracted from divider resistor description."""
        comps, nets = self._adc_circuit()
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        ch0 = [c for c in config.adc_channels if c.channel == 0]
        assert "Coolant temp" in ch0[0].description


# ---------------------------------------------------------------------------
# Bus inference tests
# ---------------------------------------------------------------------------


class TestBusInference:
    """Test I2C/SPI bus detection."""

    def test_i2c_bus_detected(self) -> None:
        """MCU with I2C pins should produce an I2C bus config."""
        mcu = Component(
            ref="U1", value="ESP32-S3-WROOM-1", footprint="ESP32",
            pins=(
                Pin("1", "SDA", PinType.BIDIRECTIONAL, PinFunction.I2C_SDA, "I2C_SDA"),
                Pin("2", "SCL", PinType.BIDIRECTIONAL, PinFunction.I2C_SCL, "I2C_SCL"),
            ),
        )
        adc = Component(
            ref="U4", value="ADS1115", footprint="MSOP-10",
            pins=(
                Pin("1", "ADDR", PinType.INPUT, net="AGND"),
                Pin("8", "SDA", PinType.BIDIRECTIONAL, net="I2C_SDA"),
                Pin("9", "SCL", PinType.BIDIRECTIONAL, net="I2C_SCL"),
            ),
        )
        r_sda = Component(
            ref="R1", value="4.7k", footprint="R_0402",
            pins=(
                Pin("1", "~", PinType.PASSIVE, net="+3V3"),
                Pin("2", "~", PinType.PASSIVE, net="I2C_SDA"),
            ),
        )
        nets = (
            Net("I2C_SDA", (
                NetConnection("U1", "1"), NetConnection("U4", "8"),
                NetConnection("R1", "2"),
            )),
            Net("I2C_SCL", (
                NetConnection("U1", "2"), NetConnection("U4", "9"),
            )),
            Net("AGND", (NetConnection("U4", "1"),)),
            Net("+3V3", (NetConnection("R1", "1"),)),
        )
        reqs = _make_requirements((mcu, adc, r_sda), nets)
        config = generate_board_config(reqs)
        assert len(config.buses) == 1
        bus = config.buses[0]
        assert bus.bus_type == BusType.I2C
        assert "I2C_SDA" in bus.signal_nets
        assert any(d.ref == "U4" for d in bus.devices)
        assert "R1" in bus.pullup_refs

    def test_spi_bus_with_cs(self) -> None:
        """SPI bus should detect devices and their CS nets."""
        mcu = Component(
            ref="U1", value="ESP32-S3-WROOM-1", footprint="ESP32",
            pins=(
                Pin("1", "SPI_CLK", PinType.OUTPUT, PinFunction.SPI_CLK, "SPI_CLK"),
                Pin("2", "SPI_MOSI", PinType.OUTPUT, PinFunction.SPI_MOSI, "SPI_MOSI"),
                Pin("3", "SPI_MISO", PinType.INPUT, PinFunction.SPI_MISO, "SPI_MISO"),
                Pin("4", "SPI_CS", PinType.OUTPUT, PinFunction.SPI_CS, "W5500_CS"),
            ),
        )
        eth = Component(
            ref="U3", value="W5500", footprint="QFP-48",
            pins=(
                Pin("1", "SCLK", PinType.INPUT, net="SPI_CLK"),
                Pin("2", "MOSI", PinType.INPUT, net="SPI_MOSI"),
                Pin("3", "MISO", PinType.OUTPUT, net="SPI_MISO"),
                Pin("4", "CS", PinType.INPUT, net="W5500_CS"),
            ),
        )
        nets = (
            Net("SPI_CLK", (NetConnection("U1", "1"), NetConnection("U3", "1"))),
            Net("SPI_MOSI", (NetConnection("U1", "2"), NetConnection("U3", "2"))),
            Net("SPI_MISO", (NetConnection("U1", "3"), NetConnection("U3", "3"))),
            Net("W5500_CS", (NetConnection("U1", "4"), NetConnection("U3", "4"))),
        )
        reqs = _make_requirements((mcu, eth), nets)
        config = generate_board_config(reqs)
        spi_buses = [b for b in config.buses if b.bus_type == BusType.SPI]
        assert len(spi_buses) == 1
        assert any(d.ref == "U3" and d.cs_net == "W5500_CS" for d in spi_buses[0].devices)


# ---------------------------------------------------------------------------
# Connector inference tests
# ---------------------------------------------------------------------------


class TestConnectorInference:
    """Test connector pinout generation."""

    def test_connector_pinout(self) -> None:
        """J-prefix components should produce connector pinouts."""
        j1 = Component(
            ref="J1", value="Screw_Terminal_01x03", footprint="TerminalBlock",
            pins=(
                Pin("1", "VIN", PinType.PASSIVE, net="VIN"),
                Pin("2", "GND", PinType.PASSIVE, net="GND"),
                Pin("3", "SIG", PinType.PASSIVE, net="SIGNAL_1"),
            ),
        )
        nets = (
            Net("VIN", (NetConnection("J1", "1"),)),
            Net("GND", (NetConnection("J1", "2"),)),
            Net("SIGNAL_1", (NetConnection("J1", "3"),)),
        )
        reqs = _make_requirements((j1,), nets)
        config = generate_board_config(reqs)
        assert len(config.connectors) == 1
        conn = config.connectors[0]
        assert conn.ref == "J1"
        assert len(conn.pins) == 3
        assert conn.pins[0].net == "VIN"


# ---------------------------------------------------------------------------
# Resistance parser tests
# ---------------------------------------------------------------------------


class TestParseResistance:
    """Test resistor value parsing."""

    def test_k_suffix(self) -> None:
        assert _parse_resistance("100k") == 100_000

    def test_m_suffix(self) -> None:
        assert _parse_resistance("1M") == 1_000_000

    def test_bare_value(self) -> None:
        assert _parse_resistance("470") == 470

    def test_decimal(self) -> None:
        assert _parse_resistance("4.7k") == 4_700

    def test_invalid(self) -> None:
        assert _parse_resistance("abc") is None


# ---------------------------------------------------------------------------
# Serialisation tests
# ---------------------------------------------------------------------------


class TestJSONSerialisation:
    """Test JSON output format."""

    def test_json_roundtrip(self) -> None:
        """JSON output should be valid and contain all sections."""
        comps, nets = _relay_circuit(1, "VIN", "OUT", "GPIO_1")
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        json_str = board_config_to_json(config)
        data = json.loads(json_str)
        assert data["project_name"] == "TestBoard"
        assert len(data["relays"]) == 1
        assert data["relays"][0]["polarity"] == "active_high"
        assert "action" in data["relays"][0]

    def test_adc_json_has_address(self) -> None:
        """ADC JSON should include formatted I2C address."""
        adc = Component(
            ref="U4", value="ADS1115", footprint="MSOP-10",
            pins=(
                Pin("1", "ADDR", PinType.INPUT, net="AGND"),
                Pin("4", "AIN0", PinType.INPUT, net="ADC_CH0"),
            ),
        )
        nets = (
            Net("AGND", (NetConnection("U4", "1"),)),
            Net("ADC_CH0", (NetConnection("U4", "4"),)),
        )
        reqs = _make_requirements((adc,), nets)
        config = generate_board_config(reqs)
        json_str = board_config_to_json(config)
        data = json.loads(json_str)
        assert data["adc_channels"][0]["i2c_address"] == "0x48"


class TestMarkdownSerialisation:
    """Test Markdown output format."""

    def test_markdown_has_relay_table(self) -> None:
        """Markdown should contain a relay control table."""
        comps, nets = _relay_circuit(1, "VIN", "OUT", "GPIO_1")
        reqs = _make_requirements(comps, nets)
        config = generate_board_config(reqs)
        md = board_config_to_markdown(config)
        assert "## Relay Control" in md
        assert "| K1 |" in md
        assert "active_high" in md

    def test_markdown_has_adc_section(self) -> None:
        """Markdown should contain ADC channel info when present."""
        adc = Component(
            ref="U4", value="ADS1115", footprint="MSOP-10",
            pins=(
                Pin("1", "ADDR", PinType.INPUT, net="AGND"),
                Pin("4", "AIN0", PinType.INPUT, net="ADC_CH0"),
            ),
        )
        nets = (
            Net("AGND", (NetConnection("U4", "1"),)),
            Net("ADC_CH0", (NetConnection("U4", "4"),)),
        )
        reqs = _make_requirements((adc,), nets)
        config = generate_board_config(reqs)
        md = board_config_to_markdown(config)
        assert "## ADC Channels" in md

    def test_markdown_warnings(self) -> None:
        """Warnings should appear in markdown output."""
        # Relay with no driver → produces a warning
        k = Component(
            ref="K1", value="SRD", footprint="Relay_SPDT",
            pins=(
                Pin("1", "COM", PinType.PASSIVE, net="VIN"),
                Pin("2", "COIL-", PinType.PASSIVE, net="K1_COIL"),
                Pin("3", "NO", PinType.PASSIVE, net="OUT"),
                Pin("5", "COIL+", PinType.PASSIVE, net="+5V"),
            ),
        )
        nets = (
            Net("VIN", (NetConnection("K1", "1"),)),
            Net("K1_COIL", (NetConnection("K1", "2"),)),
            Net("OUT", (NetConnection("K1", "3"),)),
            Net("+5V", (NetConnection("K1", "5"),)),
        )
        reqs = _make_requirements((k,), nets)
        config = generate_board_config(reqs)
        md = board_config_to_markdown(config)
        assert "## Warnings" in md
        assert "no transistor" in md.lower()

    def test_empty_config_no_crash(self) -> None:
        """Empty config should produce valid markdown."""
        reqs = _make_requirements()
        config = generate_board_config(reqs)
        md = board_config_to_markdown(config)
        assert "# Board Configuration: TestBoard" in md


# ---------------------------------------------------------------------------
# Relay COM cutout tests (footprint-embedded Edge.Cuts)
# ---------------------------------------------------------------------------


class TestRelayCOMCutout:
    """Test U-shaped cutout is embedded in relay footprint graphics."""

    def test_relay_footprint_has_edge_cuts(self) -> None:
        """make_relay_spdt() should include 3 Edge.Cuts lines for the U cutout."""
        from kicad_pipeline.pcb.footprints import make_relay_spdt

        fp = make_relay_spdt("K1", "SRD-05VDC-SL-C")
        edge_cuts = [
            g for g in fp.graphics
            if hasattr(g, "layer") and g.layer == "Edge.Cuts"
        ]
        # U shape = 3 segments (left vertical + top horizontal + bottom horizontal)
        assert len(edge_cuts) == 3

    def test_cutout_width_is_1mm(self) -> None:
        """Edge.Cuts lines should have 1mm slot width."""
        from kicad_pipeline.pcb.footprints import make_relay_spdt

        fp = make_relay_spdt("K1", "SRD-05VDC-SL-C")
        edge_cuts = [
            g for g in fp.graphics
            if hasattr(g, "layer") and g.layer == "Edge.Cuts"
        ]
        for line in edge_cuts:
            assert line.width == 1.0

    def test_cutout_surrounds_com_pin(self) -> None:
        """U cutout should be centred around COM pad (pin 1 at 0,0)."""
        from kicad_pipeline.pcb.footprints import make_relay_spdt

        fp = make_relay_spdt("K1", "SRD-05VDC-SL-C")
        edge_cuts = [
            g for g in fp.graphics
            if hasattr(g, "layer") and g.layer == "Edge.Cuts"
        ]
        # The closed end (vertical line) should be at negative X
        all_x = []
        for line in edge_cuts:
            all_x.extend([line.start.x, line.end.x])
        # Closed end is to the left of COM (x=0)
        assert min(all_x) < 0.0
        # Open end is to the right
        assert max(all_x) > 0.0
        # Symmetric about Y=0 (COM pin)
        all_y = []
        for line in edge_cuts:
            all_y.extend([line.start.y, line.end.y])
        assert min(all_y) < 0.0
        assert max(all_y) > 0.0
        assert abs(abs(min(all_y)) - abs(max(all_y))) < 0.01

    def test_cutout_dimensions_reasonable(self) -> None:
        """Cutout slot width and height should be reasonable for relay COM wire."""
        from kicad_pipeline.pcb.footprints import make_relay_spdt

        fp = make_relay_spdt("K1", "SRD-05VDC-SL-C")
        edge_cuts = [
            g for g in fp.graphics
            if hasattr(g, "layer") and g.layer == "Edge.Cuts"
        ]
        all_y = []
        for line in edge_cuts:
            all_y.extend([line.start.y, line.end.y])
        span = max(all_y) - min(all_y)
        # Cutout should be between 2mm and 10mm wide
        assert 2.0 < span < 10.0

    def test_cutout_moves_with_footprint(self) -> None:
        """Edge.Cuts graphics are in local coords — they move with the footprint.

        This is the key property: unlike standalone gr_line segments,
        fp_line entries on Edge.Cuts are relative to the footprint origin
        and automatically track the footprint position and rotation.
        """
        from kicad_pipeline.pcb.footprints import make_relay_spdt

        fp = make_relay_spdt("K1", "SRD-05VDC-SL-C")
        edge_cuts = [
            g for g in fp.graphics
            if hasattr(g, "layer") and g.layer == "Edge.Cuts"
        ]
        # Verify all coordinates are in local space (small values near origin)
        for line in edge_cuts:
            assert abs(line.start.x) < 10.0
            assert abs(line.start.y) < 10.0
            assert abs(line.end.x) < 10.0
            assert abs(line.end.y) < 10.0


# ---------------------------------------------------------------------------
# Config helper edge cases
# ---------------------------------------------------------------------------


class TestConfigHelpers:
    """Test internal helper functions in board_config_generator."""

    def test_is_gnd_net_matches(self) -> None:
        from kicad_pipeline.config.board_config_generator import _is_gnd_net

        assert _is_gnd_net("GND") is True
        assert _is_gnd_net("AGND") is True
        assert _is_gnd_net("DGND") is True

    def test_is_gnd_net_rejects(self) -> None:
        from kicad_pipeline.config.board_config_generator import _is_gnd_net

        assert _is_gnd_net("+3V3") is False
        assert _is_gnd_net("SPI_CLK") is False

    def test_is_power_net_matches(self) -> None:
        from kicad_pipeline.config.board_config_generator import _is_power_net

        assert _is_power_net("GND") is True
        assert _is_power_net("+3V3") is True
        assert _is_power_net("VIN") is True
        assert _is_power_net("+5V") is True

    def test_is_power_net_rejects(self) -> None:
        from kicad_pipeline.config.board_config_generator import _is_power_net

        assert _is_power_net("SPI_CLK") is False
        assert _is_power_net("RELAY_1") is False

    def test_parse_voltage_valid(self) -> None:
        from kicad_pipeline.config.board_config_generator import _parse_voltage

        assert _parse_voltage("+3V3") == 3.3
        assert _parse_voltage("+5V") == 5.0
        assert _parse_voltage("+1V8") == 1.8

    def test_parse_voltage_invalid(self) -> None:
        from kicad_pipeline.config.board_config_generator import _parse_voltage

        assert _parse_voltage("VIN") is None
        assert _parse_voltage("GND") is None
        assert _parse_voltage("SPI_CLK") is None

    def test_is_mcu_component(self) -> None:
        from kicad_pipeline.config.board_config_generator import _is_mcu_component

        mcu = Component(ref="U1", value="ESP32-S3-WROOM-1", footprint="ESP32")
        assert _is_mcu_component(mcu) is True
        resistor = Component(ref="R1", value="10k", footprint="R_0805")
        assert _is_mcu_component(resistor) is False

    def test_ref_prefix(self) -> None:
        from kicad_pipeline.config.board_config_generator import _ref_prefix

        assert _ref_prefix("R1") == "R"
        assert _ref_prefix("U42") == "U"
        assert _ref_prefix("K1") == "K"
        assert _ref_prefix("123") == ""

    def test_parse_resistance_r_suffix(self) -> None:
        """_parse_resistance handles R suffix (e.g. '100R')."""
        assert _parse_resistance("100R") == 100.0

    def test_empty_requirements_no_crash(self) -> None:
        """generate_board_config on empty requirements produces valid config."""
        reqs = _make_requirements()
        config = generate_board_config(reqs)
        assert config.project_name == "TestBoard"
        assert config.relays == ()
        assert config.adc_channels == ()
        assert config.buses == ()


# ---------------------------------------------------------------------------
# Ratsnest edge cases (additional to test_ratsnest.py)
# ---------------------------------------------------------------------------


class TestRatsnestEdgeCases:
    """Additional edge cases for ratsnest utilities."""

    def test_rotate_point_270_degrees(self) -> None:
        from kicad_pipeline.visualization.ratsnest import rotate_point

        x, y = rotate_point(1.0, 0.0, 270.0)
        assert abs(x - 0.0) < 1e-9
        assert abs(y - (-1.0)) < 1e-9

    def test_rotate_point_45_degrees(self) -> None:
        import math

        from kicad_pipeline.visualization.ratsnest import rotate_point

        x, y = rotate_point(1.0, 0.0, 45.0)
        expected_x = math.cos(math.radians(45))
        expected_y = math.sin(math.radians(45))
        assert abs(x - expected_x) < 1e-9
        assert abs(y - expected_y) < 1e-9

    def test_mst_four_points_has_three_edges(self) -> None:
        from kicad_pipeline.visualization.ratsnest import minimum_spanning_tree

        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        edges = minimum_spanning_tree(points)
        assert len(edges) == 3

    def test_mst_duplicate_points(self) -> None:
        """MST with duplicate points still produces valid tree."""
        from kicad_pipeline.visualization.ratsnest import minimum_spanning_tree

        points = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
        edges = minimum_spanning_tree(points)
        assert len(edges) == 2

    def test_power_nets_are_frozenset(self) -> None:
        from kicad_pipeline.visualization.ratsnest import POWER_NETS

        assert isinstance(POWER_NETS, frozenset)
        assert "GND" in POWER_NETS
        assert "+3V3" in POWER_NETS
        assert "+5V" in POWER_NETS
