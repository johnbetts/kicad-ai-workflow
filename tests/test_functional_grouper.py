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
    SubCircuitType,
    VoltageDomain,
    _classify_voltage,
    _parse_voltage_from_net,
    assign_zones,
    classify_voltage_domains,
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
