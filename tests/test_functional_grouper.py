"""Tests for functional_grouper: sub-circuit detection and voltage domains."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.functional_grouper import (
    BoardZoneAssignment,
    DetectedSubCircuit,
    DomainAffinity,
    SubCircuitNode,
    SubCircuitType,
    VoltageDomain,
    _classify_voltage,
    _find_mcu_ref,
    _parse_voltage_from_net,
    assign_zones,
    classify_voltage_domains,
    detect_cross_domain_affinities,
    detect_subcircuits,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pin(num: str, name: str, net: str | None = None) -> Pin:
    return Pin(
        number=num,
        name=name,
        pin_type=PinType.PASSIVE,
        function=None,
        net=net,
    )


def _comp(ref: str, value: str, fp: str = "R_0402", pins: tuple[Pin, ...] = ()) -> Component:
    return Component(
        ref=ref,
        value=value,
        footprint=fp,
        lcsc=None,
        description=None,
        datasheet=None,
        pins=pins,
    )


def _make_requirements(
    components: list[Component],
    nets: list[Net],
    features: list[FeatureBlock] | None = None,
) -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="test", author="test", revision="1", description="test"),
        features=tuple(features or []),
        components=tuple(components),
        nets=tuple(nets),
        pin_map=None,
        power_budget=None,
        mechanical=MechanicalConstraints(
            board_width_mm=100.0,
            board_height_mm=80.0,
        ),
        recommendations=(),
        board_context=None,
    )


# ---------------------------------------------------------------------------
# Voltage parsing tests
# ---------------------------------------------------------------------------


class TestParseVoltage:
    def test_24v(self) -> None:
        assert _parse_voltage_from_net("+24V") == 24.0

    def test_5v(self) -> None:
        assert _parse_voltage_from_net("+5V") == 5.0

    def test_3v3(self) -> None:
        assert _parse_voltage_from_net("+3V3") == 3.3

    def test_gnd(self) -> None:
        assert _parse_voltage_from_net("GND") == 0.0

    def test_unknown(self) -> None:
        assert _parse_voltage_from_net("SPI_CLK") is None

    def test_12v(self) -> None:
        assert _parse_voltage_from_net("VIN_12V") == 12.0


class TestClassifyVoltage:
    def test_high_voltage(self) -> None:
        assert _classify_voltage(24.0) == VoltageDomain.VIN_24V

    def test_5v(self) -> None:
        assert _classify_voltage(5.0) == VoltageDomain.POWER_5V

    def test_3v3(self) -> None:
        assert _classify_voltage(3.3) == VoltageDomain.DIGITAL_3V3

    def test_none(self) -> None:
        assert _classify_voltage(None) == VoltageDomain.MIXED


# ---------------------------------------------------------------------------
# Relay driver detection
# ---------------------------------------------------------------------------


class TestDetectRelayDriver:
    def test_basic_relay_driver(self) -> None:
        """K1 + Q1 + D1 (flyback) detected as relay_driver."""
        components = [
            _comp("K1", "RELAY", "Relay_SPDT", pins=(
                _pin("1", "COIL+", net="+24V"),
                _pin("2", "COIL-", net="RELAY1_DRIVE"),
                _pin("3", "COM", net="COM1"),
                _pin("4", "NO", net="NO1"),
            )),
            _comp("Q1", "2N7002", "SOT-23", pins=(
                _pin("1", "G", net="MCU_RELAY1"),
                _pin("2", "D", net="RELAY1_DRIVE"),
                _pin("3", "S", net="GND"),
            )),
            _comp("D1", "1N4148", "SOD-123", pins=(
                _pin("1", "A", net="RELAY1_DRIVE"),
                _pin("2", "K", net="+24V"),
            )),
            _comp("R1", "10k", "R_0402", pins=(
                _pin("1", "1", net="MCU_RELAY1"),
                _pin("2", "2", net="MCU_GPIO1"),
            )),
        ]
        nets = [
            Net(name="+24V", connections=(
                NetConnection(ref="K1", pin="1"),
                NetConnection(ref="D1", pin="2"),
            )),
            Net(name="RELAY1_DRIVE", connections=(
                NetConnection(ref="K1", pin="2"),
                NetConnection(ref="Q1", pin="2"),
                NetConnection(ref="D1", pin="1"),
            )),
            Net(name="MCU_RELAY1", connections=(
                NetConnection(ref="Q1", pin="1"),
                NetConnection(ref="R1", pin="1"),
            )),
            Net(name="GND", connections=(
                NetConnection(ref="Q1", pin="3"),
            )),
            Net(name="MCU_GPIO1", connections=(
                NetConnection(ref="R1", pin="2"),
            )),
        ]
        reqs = _make_requirements(components, nets)
        subcircuits = detect_subcircuits(reqs)

        relay_scs = [s for s in subcircuits if s.circuit_type == SubCircuitType.RELAY_DRIVER]
        assert len(relay_scs) == 1
        sc = relay_scs[0]
        assert sc.anchor_ref == "K1"
        assert "K1" in sc.refs
        assert "Q1" in sc.refs
        assert "D1" in sc.refs

    def test_no_relay_no_detection(self) -> None:
        """No K* refs → no relay drivers detected."""
        components = [_comp("R1", "10k"), _comp("R2", "10k")]
        nets = [Net(name="SIG", connections=(
            NetConnection(ref="R1", pin="1"),
            NetConnection(ref="R2", pin="1"),
        ))]
        reqs = _make_requirements(components, nets)
        subcircuits = detect_subcircuits(reqs)
        assert not any(s.circuit_type == SubCircuitType.RELAY_DRIVER for s in subcircuits)


# ---------------------------------------------------------------------------
# Buck converter detection
# ---------------------------------------------------------------------------


class TestDetectBuckConverter:
    def test_basic_buck(self) -> None:
        """IC with SW pin + inductor detected as buck converter."""
        components = [
            _comp("U1", "TPS54331", "SOIC-8", pins=(
                _pin("1", "VIN", net="+24V"),
                _pin("2", "SW", net="SW_NODE"),
                _pin("3", "GND", net="GND"),
                _pin("4", "FB", net="FB_5V"),
                _pin("5", "VOUT", net="+5V"),
            )),
            _comp("L1", "10uH", "IND_1210", pins=(
                _pin("1", "1", net="SW_NODE"),
                _pin("2", "2", net="+5V"),
            )),
            _comp("C1", "10uF", "C_0805", pins=(
                _pin("1", "1", net="+24V"),
                _pin("2", "2", net="GND"),
            )),
            _comp("C2", "22uF", "C_0805", pins=(
                _pin("1", "1", net="+5V"),
                _pin("2", "2", net="GND"),
            )),
        ]
        nets = [
            Net(name="+24V", connections=(
                NetConnection(ref="U1", pin="1"),
                NetConnection(ref="C1", pin="1"),
            )),
            Net(name="SW_NODE", connections=(
                NetConnection(ref="U1", pin="2"),
                NetConnection(ref="L1", pin="1"),
            )),
            Net(name="GND", connections=(
                NetConnection(ref="U1", pin="3"),
                NetConnection(ref="C1", pin="2"),
                NetConnection(ref="C2", pin="2"),
            )),
            Net(name="FB_5V", connections=(
                NetConnection(ref="U1", pin="4"),
            )),
            Net(name="+5V", connections=(
                NetConnection(ref="U1", pin="5"),
                NetConnection(ref="L1", pin="2"),
                NetConnection(ref="C2", pin="1"),
            )),
        ]
        reqs = _make_requirements(components, nets)
        subcircuits = detect_subcircuits(reqs)

        bucks = [s for s in subcircuits if s.circuit_type == SubCircuitType.BUCK_CONVERTER]
        assert len(bucks) == 1
        assert bucks[0].anchor_ref == "U1"
        assert "L1" in bucks[0].refs


# ---------------------------------------------------------------------------
# Crystal detection
# ---------------------------------------------------------------------------


class TestDetectCrystal:
    def test_crystal_with_load_caps(self) -> None:
        """Y1 + 2 load caps detected."""
        components = [
            _comp("Y1", "8MHz", "Crystal_SMD", pins=(
                _pin("1", "1", net="XTAL_IN"),
                _pin("2", "2", net="XTAL_OUT"),
            )),
            _comp("C10", "22pF", "C_0402", pins=(
                _pin("1", "1", net="XTAL_IN"),
                _pin("2", "2", net="GND"),
            )),
            _comp("C11", "22pF", "C_0402", pins=(
                _pin("1", "1", net="XTAL_OUT"),
                _pin("2", "2", net="GND"),
            )),
        ]
        nets = [
            Net(name="XTAL_IN", connections=(
                NetConnection(ref="Y1", pin="1"),
                NetConnection(ref="C10", pin="1"),
            )),
            Net(name="XTAL_OUT", connections=(
                NetConnection(ref="Y1", pin="2"),
                NetConnection(ref="C11", pin="1"),
            )),
            Net(name="GND", connections=(
                NetConnection(ref="C10", pin="2"),
                NetConnection(ref="C11", pin="2"),
            )),
        ]
        reqs = _make_requirements(components, nets)
        subcircuits = detect_subcircuits(reqs)

        crystals = [s for s in subcircuits if s.circuit_type == SubCircuitType.CRYSTAL_OSC]
        assert len(crystals) == 1
        assert crystals[0].anchor_ref == "Y1"
        assert "C10" in crystals[0].refs
        assert "C11" in crystals[0].refs
        assert crystals[0].domain == VoltageDomain.DIGITAL_3V3


# ---------------------------------------------------------------------------
# Decoupling pair detection
# ---------------------------------------------------------------------------


class TestDetectDecoupling:
    def test_decoupling_cap_paired_with_ic(self) -> None:
        """100nF cap sharing power net with IC detected as decoupling pair."""
        components = [
            _comp("U1", "STM32F4", "LQFP-48", pins=(
                _pin("1", "VDD", net="+3V3"),
                _pin("2", "GND", net="GND"),
                _pin("3", "PA0", net="SIG1"),
            )),
            _comp("C1", "100nF", "C_0402", pins=(
                _pin("1", "1", net="+3V3"),
                _pin("2", "2", net="GND"),
            )),
        ]
        nets = [
            Net(name="+3V3", connections=(
                NetConnection(ref="U1", pin="1"),
                NetConnection(ref="C1", pin="1"),
            )),
            Net(name="GND", connections=(
                NetConnection(ref="U1", pin="2"),
                NetConnection(ref="C1", pin="2"),
            )),
            Net(name="SIG1", connections=(
                NetConnection(ref="U1", pin="3"),
            )),
        ]
        reqs = _make_requirements(components, nets)
        subcircuits = detect_subcircuits(reqs)

        decoup = [s for s in subcircuits if s.circuit_type == SubCircuitType.DECOUPLING]
        assert len(decoup) == 1
        assert decoup[0].anchor_ref == "U1"
        assert "C1" in decoup[0].refs


# ---------------------------------------------------------------------------
# Voltage divider detection
# ---------------------------------------------------------------------------


class TestDetectVoltageDivider:
    def test_basic_divider(self) -> None:
        """Two resistors in series between power and GND with midpoint signal."""
        components = [
            _comp("R1", "10k", pins=(
                _pin("1", "1", net="+5V"),
                _pin("2", "2", net="FB_MID"),
            )),
            _comp("R2", "4.7k", pins=(
                _pin("1", "1", net="FB_MID"),
                _pin("2", "2", net="GND"),
            )),
        ]
        nets = [
            Net(name="+5V", connections=(NetConnection(ref="R1", pin="1"),)),
            Net(name="FB_MID", connections=(
                NetConnection(ref="R1", pin="2"),
                NetConnection(ref="R2", pin="1"),
            )),
            Net(name="GND", connections=(NetConnection(ref="R2", pin="2"),)),
        ]
        reqs = _make_requirements(components, nets)
        subcircuits = detect_subcircuits(reqs)

        dividers = [s for s in subcircuits if s.circuit_type == SubCircuitType.VOLTAGE_DIVIDER]
        assert len(dividers) == 1
        assert set(dividers[0].refs) == {"R1", "R2"}


# ---------------------------------------------------------------------------
# Voltage domain classification
# ---------------------------------------------------------------------------


class TestClassifyDomains:
    def test_component_on_24v_rail(self) -> None:
        components = [
            _comp("K1", "RELAY", pins=(_pin("1", "1", net="+24V"),)),
        ]
        nets = [Net(name="+24V", connections=(NetConnection(ref="K1", pin="1"),))]
        reqs = _make_requirements(components, nets)
        domains = classify_voltage_domains(reqs)
        assert domains["K1"] == VoltageDomain.VIN_24V

    def test_component_on_3v3_rail(self) -> None:
        components = [
            _comp("U1", "STM32", pins=(_pin("1", "VDD", net="+3V3"),)),
        ]
        nets = [Net(name="+3V3", connections=(NetConnection(ref="U1", pin="1"),))]
        reqs = _make_requirements(components, nets)
        domains = classify_voltage_domains(reqs)
        assert domains["U1"] == VoltageDomain.DIGITAL_3V3

    def test_analog_net(self) -> None:
        components = [
            _comp("R1", "10k", pins=(_pin("1", "1", net="ADC_IN"),)),
        ]
        nets = [Net(name="ADC_IN", connections=(NetConnection(ref="R1", pin="1"),))]
        reqs = _make_requirements(components, nets)
        domains = classify_voltage_domains(reqs)
        assert domains["R1"] == VoltageDomain.ANALOG

    def test_relay_default_domain(self) -> None:
        """K* with no power nets defaults to VIN_24V."""
        components = [_comp("K1", "RELAY", pins=(_pin("1", "1", net="SIG"),))]
        nets = [Net(name="SIG", connections=(NetConnection(ref="K1", pin="1"),))]
        reqs = _make_requirements(components, nets)
        domains = classify_voltage_domains(reqs)
        assert domains["K1"] == VoltageDomain.VIN_24V


# ---------------------------------------------------------------------------
# Zone assignment
# ---------------------------------------------------------------------------


class TestAssignZones:
    def test_zones_created_per_domain(self) -> None:
        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=("K1", "Q1"),
            anchor_ref="K1",
            net_connections=("+24V",),
            domain=VoltageDomain.VIN_24V,
        )
        domain_map = {"K1": VoltageDomain.VIN_24V, "Q1": VoltageDomain.VIN_24V,
                      "U1": VoltageDomain.DIGITAL_3V3}
        zones = assign_zones(
            subcircuits=(sc,),
            domain_map=domain_map,
            board_width=100.0,
            board_height=80.0,
            all_refs=("K1", "Q1", "U1"),
        )
        assert len(zones) >= 2
        vin_zone = [z for z in zones if z.domain == VoltageDomain.VIN_24V]
        assert len(vin_zone) == 1
        assert sc in vin_zone[0].subcircuits

    def test_loose_refs_assigned(self) -> None:
        domain_map = {"R1": VoltageDomain.DIGITAL_3V3}
        zones = assign_zones(
            subcircuits=(),
            domain_map=domain_map,
            board_width=100.0,
            board_height=80.0,
            all_refs=("R1",),
        )
        assert any("R1" in z.loose_refs for z in zones)

    def test_empty_no_zones(self) -> None:
        zones = assign_zones(
            subcircuits=(),
            domain_map={},
            board_width=100.0,
            board_height=80.0,
            all_refs=(),
        )
        assert len(zones) == 0


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


class TestDataclassImmutability:
    def test_detected_subcircuit_frozen(self) -> None:
        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=("K1",),
            anchor_ref="K1",
            net_connections=(),
            domain=VoltageDomain.VIN_24V,
        )
        with pytest.raises(AttributeError):
            sc.anchor_ref = "K2"  # type: ignore[misc]

    def test_zone_assignment_frozen(self) -> None:
        za = BoardZoneAssignment(
            domain=VoltageDomain.VIN_24V,
            zone_rect=(0, 0, 50, 40),
            subcircuits=(),
            loose_refs=(),
        )
        with pytest.raises(AttributeError):
            za.domain = VoltageDomain.POWER_5V  # type: ignore[misc]

    def test_subcircuit_node_frozen(self) -> None:
        node = SubCircuitNode(
            role="relay_body",
            refs=("K1",),
            anchor_ref="K1",
        )
        with pytest.raises(AttributeError):
            node.role = "driver"  # type: ignore[misc]

    def test_domain_affinity_frozen(self) -> None:
        da = DomainAffinity(
            source_refs=("R1",),
            target_refs=("R2",),
            source_domain=VoltageDomain.VIN_24V,
            target_domain=VoltageDomain.DIGITAL_3V3,
            reason="measurement",
        )
        with pytest.raises(AttributeError):
            da.reason = "control"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# New data model fields
# ---------------------------------------------------------------------------


class TestDetectedSubCircuitExtensions:
    def test_default_layout_hint(self) -> None:
        """DetectedSubCircuit defaults to 'cluster' layout hint."""
        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=("K1", "Q1"),
            anchor_ref="K1",
            net_connections=(),
            domain=VoltageDomain.VIN_24V,
        )
        assert sc.layout_hint == "cluster"
        assert sc.hierarchy is None
        assert sc.input_domain is None
        assert sc.output_domain is None

    def test_layout_hint_row(self) -> None:
        """Relay drivers can use 'row' layout hint."""
        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=("K1", "Q1"),
            anchor_ref="K1",
            net_connections=(),
            domain=VoltageDomain.VIN_24V,
            layout_hint="row",
        )
        assert sc.layout_hint == "row"

    def test_hierarchy_field(self) -> None:
        """DetectedSubCircuit can carry hierarchical structure."""
        driver_node = SubCircuitNode(
            role="driver",
            refs=("Q1", "D1", "R1"),
            anchor_ref="Q1",
        )
        root_node = SubCircuitNode(
            role="relay_body",
            refs=("K1",),
            anchor_ref="K1",
            children=(driver_node,),
        )
        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=("D1", "K1", "Q1", "R1"),
            anchor_ref="K1",
            net_connections=(),
            domain=VoltageDomain.VIN_24V,
            hierarchy=root_node,
        )
        assert sc.hierarchy is not None
        assert sc.hierarchy.role == "relay_body"
        assert len(sc.hierarchy.children) == 1
        assert sc.hierarchy.children[0].role == "driver"

    def test_regulator_domains(self) -> None:
        """Regulators can specify input/output domain."""
        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.BUCK_CONVERTER,
            refs=("U1", "L1", "C1"),
            anchor_ref="U1",
            net_connections=(),
            domain=VoltageDomain.POWER_5V,
            layout_hint="boundary",
            input_domain=VoltageDomain.VIN_24V,
            output_domain=VoltageDomain.POWER_5V,
        )
        assert sc.input_domain == VoltageDomain.VIN_24V
        assert sc.output_domain == VoltageDomain.POWER_5V


class TestNewSubCircuitTypes:
    def test_mcu_peripheral_cluster(self) -> None:
        assert SubCircuitType.MCU_PERIPHERAL_CLUSTER.value == "mcu_peripheral_cluster"

    def test_rf_antenna(self) -> None:
        assert SubCircuitType.RF_ANTENNA.value == "rf_antenna"


class TestSubCircuitNode:
    def test_basic_node(self) -> None:
        node = SubCircuitNode(
            role="relay_body",
            refs=("K1",),
            anchor_ref="K1",
        )
        assert node.role == "relay_body"
        assert node.refs == ("K1",)
        assert node.children == ()

    def test_nested_hierarchy(self) -> None:
        led = SubCircuitNode(role="led_indicator", refs=("D2", "R2"), anchor_ref="D2")
        driver = SubCircuitNode(role="driver", refs=("Q1", "D1", "R1"), anchor_ref="Q1")
        root = SubCircuitNode(
            role="relay_body",
            refs=("K1",),
            anchor_ref="K1",
            children=(driver, led),
        )
        assert len(root.children) == 2
        assert root.children[0].role == "driver"
        assert root.children[1].role == "led_indicator"


class TestDomainAffinity:
    def test_measurement_affinity(self) -> None:
        da = DomainAffinity(
            source_refs=("R1", "R2"),
            target_refs=("U1",),
            source_domain=VoltageDomain.VIN_24V,
            target_domain=VoltageDomain.DIGITAL_3V3,
            reason="measurement",
        )
        assert da.reason == "measurement"
        assert da.source_domain != da.target_domain


# ---------------------------------------------------------------------------
# Phase 2: Detection improvements
# ---------------------------------------------------------------------------


class TestRelayHierarchy:
    def test_relay_has_hierarchy(self) -> None:
        """Relay driver detection produces hierarchical structure."""
        components = [
            _comp("K1", "RELAY", "Relay_SPDT", pins=(
                _pin("1", "COIL+", net="+24V"),
                _pin("2", "COIL-", net="RELAY1_DRIVE"),
            )),
            _comp("Q1", "2N7002", "SOT-23", pins=(
                _pin("1", "G", net="MCU_RELAY1"),
                _pin("2", "D", net="RELAY1_DRIVE"),
                _pin("3", "S", net="GND"),
            )),
            _comp("D1", "1N4148", "SOD-123", pins=(
                _pin("1", "A", net="RELAY1_DRIVE"),
                _pin("2", "K", net="+24V"),
            )),
            _comp("R1", "10k", "R_0402", pins=(
                _pin("1", "1", net="MCU_RELAY1"),
                _pin("2", "2", net="MCU_GPIO1"),
            )),
        ]
        nets = [
            Net(name="+24V", connections=(
                NetConnection(ref="K1", pin="1"),
                NetConnection(ref="D1", pin="2"),
            )),
            Net(name="RELAY1_DRIVE", connections=(
                NetConnection(ref="K1", pin="2"),
                NetConnection(ref="Q1", pin="2"),
                NetConnection(ref="D1", pin="1"),
            )),
            Net(name="MCU_RELAY1", connections=(
                NetConnection(ref="Q1", pin="1"),
                NetConnection(ref="R1", pin="1"),
            )),
            Net(name="GND", connections=(NetConnection(ref="Q1", pin="3"),)),
            Net(name="MCU_GPIO1", connections=(NetConnection(ref="R1", pin="2"),)),
        ]
        reqs = _make_requirements(components, nets)
        subcircuits = detect_subcircuits(reqs)
        relay_scs = [s for s in subcircuits if s.circuit_type == SubCircuitType.RELAY_DRIVER]
        assert len(relay_scs) == 1
        sc = relay_scs[0]
        assert sc.layout_hint == "row"
        assert sc.hierarchy is not None
        assert sc.hierarchy.role == "relay_body"
        assert sc.hierarchy.anchor_ref == "K1"
        # Should have driver child with Q1
        driver_children = [c for c in sc.hierarchy.children if c.role == "driver"]
        assert len(driver_children) == 1
        assert "Q1" in driver_children[0].refs


class TestBuckBoundary:
    def test_buck_has_boundary_hint(self) -> None:
        """Buck converter detected with boundary layout hint and domains."""
        components = [
            _comp("U1", "TPS54331", "SOIC-8", pins=(
                _pin("1", "VIN", net="+24V"),
                _pin("2", "SW", net="SW_NODE"),
                _pin("3", "GND", net="GND"),
                _pin("5", "VOUT", net="+5V"),
            )),
            _comp("L1", "10uH", "IND_1210", pins=(
                _pin("1", "1", net="SW_NODE"),
                _pin("2", "2", net="+5V"),
            )),
        ]
        nets = [
            Net(name="+24V", connections=(NetConnection(ref="U1", pin="1"),)),
            Net(name="SW_NODE", connections=(
                NetConnection(ref="U1", pin="2"),
                NetConnection(ref="L1", pin="1"),
            )),
            Net(name="GND", connections=(NetConnection(ref="U1", pin="3"),)),
            Net(name="+5V", connections=(
                NetConnection(ref="U1", pin="5"),
                NetConnection(ref="L1", pin="2"),
            )),
        ]
        reqs = _make_requirements(components, nets)
        subcircuits = detect_subcircuits(reqs)
        bucks = [s for s in subcircuits if s.circuit_type == SubCircuitType.BUCK_CONVERTER]
        assert len(bucks) == 1
        assert bucks[0].layout_hint == "boundary"
        assert bucks[0].input_domain == VoltageDomain.VIN_24V
        assert bucks[0].output_domain == VoltageDomain.POWER_5V


class TestFindMCU:
    def test_finds_esp32(self) -> None:
        components = [
            _comp("U1", "ESP32-S3-WROOM-1", "QFN-48", pins=(
                _pin("1", "GND", net="GND"),
            )),
            _comp("U2", "AMS1117-3.3", "SOT-223", pins=(
                _pin("1", "VIN", net="+5V"),
            )),
        ]
        reqs = _make_requirements(components, [])
        assert _find_mcu_ref(reqs) == "U1"

    def test_finds_stm32(self) -> None:
        components = [
            _comp("U1", "STM32F411CEU6", "LQFP-48", pins=tuple(
                _pin(str(i), f"P{i}") for i in range(48)
            )),
        ]
        reqs = _make_requirements(components, [])
        assert _find_mcu_ref(reqs) == "U1"

    def test_fallback_largest_pin_count(self) -> None:
        components = [
            _comp("U1", "SomeIC", "QFP-44", pins=tuple(
                _pin(str(i), f"P{i}") for i in range(44)
            )),
            _comp("U2", "SmallIC", "SOT-23", pins=(
                _pin("1", "1"), _pin("2", "2"), _pin("3", "3"),
            )),
        ]
        reqs = _make_requirements(components, [])
        assert _find_mcu_ref(reqs) == "U1"

    def test_no_mcu(self) -> None:
        components = [_comp("R1", "10k")]
        reqs = _make_requirements(components, [])
        assert _find_mcu_ref(reqs) is None


class TestMCUPeripheralDetection:
    def test_detects_switches_and_leds(self) -> None:
        """MCU with directly-connected switches and LEDs → peripheral cluster."""
        components = [
            _comp("U1", "STM32F4", "LQFP-48", pins=(
                _pin("1", "PA0", net="SW1_SIG"),
                _pin("2", "PA1", net="LED1_SIG"),
            )),
            _comp("SW1", "BUTTON", "SW_Push", pins=(
                _pin("1", "1", net="SW1_SIG"),
                _pin("2", "2", net="GND"),
            )),
            _comp("LED1", "RED", "LED_0603", pins=(
                _pin("1", "A", net="LED1_SIG"),
                _pin("2", "K", net="GND"),
            )),
        ]
        nets = [
            Net(name="SW1_SIG", connections=(
                NetConnection(ref="U1", pin="1"),
                NetConnection(ref="SW1", pin="1"),
            )),
            Net(name="LED1_SIG", connections=(
                NetConnection(ref="U1", pin="2"),
                NetConnection(ref="LED1", pin="1"),
            )),
            Net(name="GND", connections=(
                NetConnection(ref="SW1", pin="2"),
                NetConnection(ref="LED1", pin="2"),
            )),
        ]
        reqs = _make_requirements(components, nets)
        subcircuits = detect_subcircuits(reqs)
        mcu_periph = [s for s in subcircuits
                      if s.circuit_type == SubCircuitType.MCU_PERIPHERAL_CLUSTER]
        assert len(mcu_periph) == 1
        assert mcu_periph[0].anchor_ref == "U1"
        assert "SW1" in mcu_periph[0].refs
        assert "LED1" in mcu_periph[0].refs
        assert mcu_periph[0].layout_hint == "cluster"

    def test_no_peripherals_no_cluster(self) -> None:
        """MCU with no switch/LED connections → no peripheral cluster."""
        components = [
            _comp("U1", "STM32F4", "LQFP-48", pins=(
                _pin("1", "PA0", net="SPI_CLK"),
            )),
            _comp("U2", "SPI_FLASH", "SOIC-8", pins=(
                _pin("1", "CLK", net="SPI_CLK"),
            )),
        ]
        nets = [
            Net(name="SPI_CLK", connections=(
                NetConnection(ref="U1", pin="1"),
                NetConnection(ref="U2", pin="1"),
            )),
        ]
        reqs = _make_requirements(components, nets)
        subcircuits = detect_subcircuits(reqs)
        mcu_periph = [s for s in subcircuits
                      if s.circuit_type == SubCircuitType.MCU_PERIPHERAL_CLUSTER]
        assert len(mcu_periph) == 0


class TestRFAntennaDetection:
    def test_esp32_detected(self) -> None:
        """ESP32 module detected as RF_ANTENNA."""
        components = [
            _comp("U1", "ESP32-S3-WROOM-1", "QFN-48", pins=(
                _pin("1", "GND", net="GND"),
            )),
        ]
        reqs = _make_requirements(components, [
            Net(name="GND", connections=(NetConnection(ref="U1", pin="1"),)),
        ])
        subcircuits = detect_subcircuits(reqs)
        rf = [s for s in subcircuits if s.circuit_type == SubCircuitType.RF_ANTENNA]
        assert len(rf) == 1
        assert rf[0].anchor_ref == "U1"
        assert rf[0].layout_hint == "edge"

    def test_non_rf_not_detected(self) -> None:
        """Non-RF IC not detected as RF_ANTENNA."""
        components = [
            _comp("U1", "STM32F411", "LQFP-48", pins=(
                _pin("1", "GND", net="GND"),
            )),
        ]
        reqs = _make_requirements(components, [
            Net(name="GND", connections=(NetConnection(ref="U1", pin="1"),)),
        ])
        subcircuits = detect_subcircuits(reqs)
        rf = [s for s in subcircuits if s.circuit_type == SubCircuitType.RF_ANTENNA]
        assert len(rf) == 0


class TestCrossDomainAffinity:
    def test_adc_measurement_affinity(self) -> None:
        """ADC net crossing voltage domains → measurement affinity."""
        components = [
            _comp("R1", "10k", pins=(
                _pin("1", "1", net="+24V"),
                _pin("2", "2", net="ADC_RELAY1"),
            )),
            _comp("R2", "4.7k", pins=(
                _pin("1", "1", net="ADC_RELAY1"),
                _pin("2", "2", net="GND"),
            )),
            _comp("U1", "STM32", "LQFP-48", pins=(
                _pin("1", "VDD", net="+3V3"),
                _pin("2", "PA0", net="ADC_RELAY1"),
            )),
        ]
        nets = [
            Net(name="+24V", connections=(NetConnection(ref="R1", pin="1"),)),
            Net(name="ADC_RELAY1", connections=(
                NetConnection(ref="R1", pin="2"),
                NetConnection(ref="R2", pin="1"),
                NetConnection(ref="U1", pin="2"),
            )),
            Net(name="+3V3", connections=(NetConnection(ref="U1", pin="1"),)),
            Net(name="GND", connections=(NetConnection(ref="R2", pin="2"),)),
        ]
        reqs = _make_requirements(components, nets)
        domain_map = classify_voltage_domains(reqs)
        affinities = detect_cross_domain_affinities(reqs, domain_map)
        assert len(affinities) >= 1
        adc_aff = [a for a in affinities if a.reason == "measurement"]
        assert len(adc_aff) >= 1
        # Should include VIN_24V <-> 3V3 or VIN_24V <-> ANALOG pair
        domains_seen = set()
        for a in adc_aff:
            domains_seen.add(a.source_domain)
            domains_seen.add(a.target_domain)
        assert VoltageDomain.VIN_24V in domains_seen

    def test_no_cross_domain_no_affinity(self) -> None:
        """Net within single domain → no affinity."""
        components = [
            _comp("R1", "10k", pins=(
                _pin("1", "1", net="+3V3"),
                _pin("2", "2", net="SIG"),
            )),
            _comp("R2", "10k", pins=(
                _pin("1", "1", net="SIG"),
                _pin("2", "2", net="+3V3"),
            )),
        ]
        nets = [
            Net(name="+3V3", connections=(
                NetConnection(ref="R1", pin="1"),
                NetConnection(ref="R2", pin="2"),
            )),
            Net(name="SIG", connections=(
                NetConnection(ref="R1", pin="2"),
                NetConnection(ref="R2", pin="1"),
            )),
        ]
        reqs = _make_requirements(components, nets)
        domain_map = classify_voltage_domains(reqs)
        affinities = detect_cross_domain_affinities(reqs, domain_map)
        assert len(affinities) == 0

    def test_feedback_affinity(self) -> None:
        """FB net crossing domains → feedback affinity."""
        components = [
            _comp("R1", "10k", pins=(
                _pin("1", "1", net="+5V"),
                _pin("2", "2", net="FB_OUT"),
            )),
            _comp("U1", "STM32", pins=(
                _pin("1", "VDD", net="+3V3"),
                _pin("2", "PA0", net="FB_OUT"),
            )),
        ]
        nets = [
            Net(name="+5V", connections=(NetConnection(ref="R1", pin="1"),)),
            Net(name="FB_OUT", connections=(
                NetConnection(ref="R1", pin="2"),
                NetConnection(ref="U1", pin="2"),
            )),
            Net(name="+3V3", connections=(NetConnection(ref="U1", pin="1"),)),
        ]
        reqs = _make_requirements(components, nets)
        domain_map = classify_voltage_domains(reqs)
        affinities = detect_cross_domain_affinities(reqs, domain_map)
        fb_aff = [a for a in affinities if a.reason == "feedback"]
        assert len(fb_aff) == 1
