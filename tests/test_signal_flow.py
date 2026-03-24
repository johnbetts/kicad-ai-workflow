"""Tests for kicad_pipeline.optimization.signal_flow."""

from __future__ import annotations

from kicad_pipeline.models.requirements import (
    Component,
    Net,
    NetConnection,
    Pin,
    PinType,
)
from kicad_pipeline.optimization.functional_grouper import (
    DetectedSubCircuit,
    SubCircuitType,
    VoltageDomain,
)
from kicad_pipeline.optimization.signal_flow import (
    order_subcircuit_by_flow,
    trace_signal_chain,
)
from tests.helpers import make_component, make_requirements

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reqs_with_nets(
    components: tuple[Component, ...],
    nets: tuple[Net, ...],
) -> object:
    """Build ProjectRequirements with given components and nets."""
    return make_requirements(components=components, nets=nets)


def _make_subcircuit(
    sc_type: SubCircuitType,
    refs: tuple[str, ...],
    anchor: str = "",
) -> DetectedSubCircuit:
    """Build a minimal DetectedSubCircuit."""
    return DetectedSubCircuit(
        circuit_type=sc_type,
        refs=refs,
        anchor_ref=anchor or refs[0],
        net_connections=(),
        domain=VoltageDomain.DIGITAL_3V3,
    )


# ---------------------------------------------------------------------------
# trace_signal_chain tests
# ---------------------------------------------------------------------------


class TestTraceSignalChainBuckConverter:
    """Test tracing a signal chain through a buck converter path."""

    def test_vin_to_sw_to_inductor_to_vout(self) -> None:
        """VIN pin -> SW pin -> inductor -> output cap."""
        comps = (
            make_component(
                "C1",
                "10uF",
                "C_0805",
                pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE),
                ),
            ),
            make_component(
                "U1",
                "TPS54302",
                "SOT-23-6",
                pins=(
                    Pin(number="1", name="VIN", pin_type=PinType.POWER_IN),
                    Pin(number="3", name="SW", pin_type=PinType.OUTPUT),
                ),
            ),
            make_component(
                "L1",
                "4.7uH",
                "L_0805",
                pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE),
                ),
            ),
            make_component(
                "C2",
                "22uF",
                "C_0805",
                pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE),
                ),
            ),
        )
        nets = (
            # SW net connects U1.3 (SW) and L1.1
            Net(
                name="SW_NET",
                connections=(
                    NetConnection(ref="U1", pin="3"),
                    NetConnection(ref="L1", pin="1"),
                ),
            ),
            # L1.2 to C2.1 — output (use signal-style name, not "VOUT"
            # which _is_power_net() filters)
            Net(
                name="OUT_RAIL",
                connections=(
                    NetConnection(ref="L1", pin="2"),
                    NetConnection(ref="C2", pin="1"),
                ),
            ),
        )
        reqs = _make_reqs_with_nets(comps, nets)

        chain = trace_signal_chain(reqs, "U1", "3")  # Start at SW pin

        # Should trace: U1.3 -> L1.1, L1.2 -> C2.1
        assert len(chain) >= 3
        assert chain[0] == ("U1", "3")
        # Next hop: L1 on pin 1
        assert ("L1", "1") in chain
        # Should reach C2
        refs_in_chain = [r for r, _p in chain]
        assert "L1" in refs_in_chain


class TestTraceSignalChainVoltageDivider:
    """Test tracing through a voltage divider."""

    def test_r_top_to_r_bot(self) -> None:
        """R_top and R_bot connected in series."""
        comps = (
            make_component(
                "R1",
                "10k",
                "R_0805",
                pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE),
                ),
            ),
            make_component(
                "R2",
                "4.7k",
                "R_0805",
                pins=(
                    Pin(number="1", name="1", pin_type=PinType.PASSIVE),
                    Pin(number="2", name="2", pin_type=PinType.PASSIVE),
                ),
            ),
        )
        nets = (
            Net(
                name="DIV_MID",
                connections=(
                    NetConnection(ref="R1", pin="2"),
                    NetConnection(ref="R2", pin="1"),
                ),
            ),
        )
        reqs = _make_reqs_with_nets(comps, nets)

        chain = trace_signal_chain(reqs, "R1", "2")

        assert chain[0] == ("R1", "2")
        assert ("R2", "1") in chain


# ---------------------------------------------------------------------------
# order_subcircuit_by_flow tests
# ---------------------------------------------------------------------------


class TestOrderSubcircuitRelayDriver:
    """Test relay driver ordering: R -> Q -> D -> K -> J."""

    def test_relay_driver_order(self) -> None:
        reqs = make_requirements(
            components=(
                make_component("R1", "1k", "R_0805"),
                make_component("Q1", "2N7002", "SOT-23"),
                make_component("D1", "1N4148", "SOD-323"),
                make_component("K1", "G5V-1", "Relay_SPDT"),
                make_component("J1", "TermBlock", "TermBlock_2P"),
            ),
        )
        sc = _make_subcircuit(
            SubCircuitType.RELAY_DRIVER,
            ("R1", "Q1", "D1", "K1", "J1"),
            anchor="K1",
        )

        result = order_subcircuit_by_flow(sc, reqs)

        # R before Q before D before K before J
        assert result.index("R1") < result.index("Q1")
        assert result.index("Q1") < result.index("D1")
        assert result.index("D1") < result.index("K1")
        assert result.index("K1") < result.index("J1")


class TestOrderSubcircuitLDO:
    """Test LDO ordering: C_in -> U -> C_out."""

    def test_ldo_order(self) -> None:
        reqs = make_requirements(
            components=(
                make_component("C1", "1uF", "C_0805"),
                make_component("U1", "AMS1117-3.3", "SOT-223"),
                make_component("C2", "10uF", "C_0805"),
            ),
        )
        sc = _make_subcircuit(
            SubCircuitType.LDO_REGULATOR,
            ("C1", "U1", "C2"),
            anchor="U1",
        )

        result = order_subcircuit_by_flow(sc, reqs)

        assert result.index("C1") < result.index("U1")
        assert result.index("U1") < result.index("C2")


class TestOrderEmptySubcircuit:
    """Test edge case: empty or single-ref subcircuit."""

    def test_empty_refs(self) -> None:
        reqs = make_requirements()
        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.DECOUPLING,
            refs=(),
            anchor_ref="",
            net_connections=(),
            domain=VoltageDomain.DIGITAL_3V3,
        )
        # Should not crash, return empty list
        result = order_subcircuit_by_flow(sc, reqs)
        assert result == []

    def test_single_ref(self) -> None:
        reqs = make_requirements(
            components=(make_component("C1", "100nF", "C_0402"),),
        )
        sc = _make_subcircuit(
            SubCircuitType.DECOUPLING,
            ("C1",),
            anchor="C1",
        )
        result = order_subcircuit_by_flow(sc, reqs)
        assert result == ["C1"]


class TestChainWithBranch:
    """Test buck converter with feedback divider branch."""

    def test_buck_with_fb_divider(self) -> None:
        """Feedback resistors should appear after the main power chain."""
        reqs = make_requirements(
            components=(
                make_component("C1", "10uF", "C_0805"),
                make_component("U1", "TPS54302", "SOT-23-6"),
                make_component("L1", "4.7uH", "L_0805"),
                make_component("C2", "22uF", "C_0805"),
                make_component("R1", "100k", "R_0805"),  # FB top
                make_component("R2", "33k", "R_0805"),  # FB bottom
                make_component("D1", "SS34", "SMA"),  # Bootstrap
            ),
        )
        sc = _make_subcircuit(
            SubCircuitType.BUCK_CONVERTER,
            ("C1", "U1", "L1", "C2", "R1", "R2", "D1"),
            anchor="U1",
        )

        result = order_subcircuit_by_flow(sc, reqs)

        # Main chain: C1 -> U1 -> L1 -> C2, then D1, then R1, R2
        assert result.index("C1") < result.index("U1")
        assert result.index("U1") < result.index("L1")
        assert result.index("L1") < result.index("C2")
        # FB resistors after main chain
        assert result.index("C2") < result.index("R1")
