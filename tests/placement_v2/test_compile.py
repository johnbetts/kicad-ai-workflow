"""Tests for the placement v2 constraint compiler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicad_pipeline.exceptions import ConfigurationError
from kicad_pipeline.models.requirements import (
    Component,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.placement_v2.compile import compile_constraints
from kicad_pipeline.placement_v2.ir import (
    Axis,
    ConstraintSet,
    ConstraintSource,
    Edge,
    KeepoutKind,
    PadRef,
    PinAttach,
)

REPO_PART_RULES = Path(__file__).resolve().parents[2] / "data" / "part_rules.json"

# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------


def _comp(
    ref: str,
    *,
    value: str = "val",
    footprint: str = "fp",
    pins: tuple[Pin, ...] = (),
    group: str | None = None,
    near: str | None = None,
    order: int | None = None,
    near_max_mm: float | None = None,
) -> Component:
    return Component(
        ref=ref,
        value=value,
        footprint=footprint,
        pins=pins,
        placement_group=group,
        placement_near=near,
        placement_order=order,
        placement_near_max_mm=near_max_mm,
    )


def _net(name: str, *conns: tuple[str, str]) -> Net:
    return Net(name=name, connections=tuple(NetConnection(ref=r, pin=p) for r, p in conns))


def _req(components: tuple[Component, ...], nets: tuple[Net, ...] = ()) -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        features=(),
        components=components,
        nets=nets,
    )


def _attach_map(cs: ConstraintSet) -> dict[tuple[str, str], PinAttach]:
    return {(str(a.src), str(a.dst)): a for a in cs.pin_attach}


# ---------------------------------------------------------------------------
# netlist: decoupling
# ---------------------------------------------------------------------------


class TestDecoupling:
    def test_decoupling_cap_attaches_to_ic_power_pad(self) -> None:
        req = _req(
            components=(_comp("U1", footprint="SOIC-8"), _comp("C1", value="100nF")),
            nets=(
                _net("+3V3", ("U1", "8"), ("C1", "1")),
                _net("GND", ("U1", "4"), ("C1", "2")),
            ),
        )
        cs = compile_constraints(req)
        attaches = _attach_map(cs)
        assert ("C1.1", "U1.8") in attaches
        pa = cs.pin_attach[0]
        assert pa.net == "+3V3"
        assert pa.max_mm == 5.0
        assert pa.ideal_mm == 2.0
        assert pa.source is ConstraintSource.NETLIST

    def test_decoupling_prefers_ic_sharing_placement_group(self) -> None:
        req = _req(
            components=(
                _comp("U1", group="power"),
                _comp("U2", group="mcu"),
                _comp("C1", group="mcu"),
            ),
            nets=(
                _net("+3V3", ("U1", "1"), ("U2", "3"), ("C1", "1")),
                _net("GND", ("U1", "2"), ("U2", "4"), ("C1", "2")),
            ),
        )
        cs = compile_constraints(req)
        decoupling = [a for a in cs.pin_attach if a.src.ref == "C1"]
        assert len(decoupling) == 1
        assert decoupling[0].dst == PadRef("U2", "3")

    def test_power_in_pin_type_marks_net_as_power(self) -> None:
        # Net name "RAIL_A" matches no power pattern; only the POWER_IN pin
        # on U1 qualifies it for decoupling.
        u1 = _comp(
            "U1",
            pins=(Pin(number="7", name="VDDA", pin_type=PinType.POWER_IN),),
        )
        req = _req(
            components=(u1, _comp("C1")),
            nets=(
                _net("RAIL_A", ("U1", "7"), ("C1", "1")),
                _net("GND", ("U1", "2"), ("C1", "2")),
            ),
        )
        cs = compile_constraints(req)
        assert ("C1.1", "U1.7") in _attach_map(cs)

    def test_cap_without_gnd_pin_is_not_decoupling(self) -> None:
        req = _req(
            components=(_comp("U1"), _comp("C1")),
            nets=(_net("+3V3", ("U1", "1"), ("C1", "1")), _net("SIG", ("C1", "2"))),
        )
        cs = compile_constraints(req)
        assert not cs.pin_attach


# ---------------------------------------------------------------------------
# netlist: placement_near
# ---------------------------------------------------------------------------


class TestPlacementNear:
    def test_placement_near_compiles_with_shared_net_pin(self) -> None:
        u1 = _comp("U1", pins=(Pin(number="3", name="VIN", pin_type=PinType.POWER_IN),))
        req = _req(
            components=(u1, _comp("L1", near="U1:VIN", near_max_mm=4.0)),
            nets=(_net("VIN_RAW", ("U1", "3"), ("L1", "2")),),
        )
        cs = compile_constraints(req)
        near = [a for a in cs.pin_attach if a.src.ref == "L1"]
        assert len(near) == 1
        pa = near[0]
        assert pa.src == PadRef("L1", "2")
        assert pa.dst == PadRef("U1", "3")  # resolved VIN -> pin number 3
        assert pa.net == "VIN_RAW"
        assert pa.max_mm == 4.0
        assert pa.ideal_mm == 2.0

    def test_placement_near_falls_back_to_pin_1(self) -> None:
        req = _req(
            components=(_comp("U1"), _comp("R5", near="U1:2", near_max_mm=6.0)),
            nets=(),
        )
        cs = compile_constraints(req)
        near = [a for a in cs.pin_attach if a.src.ref == "R5"]
        assert near[0].src == PadRef("R5", "1")
        assert near[0].dst == PadRef("U1", "2")


# ---------------------------------------------------------------------------
# netlist: relay drivers
# ---------------------------------------------------------------------------


class TestRelayDriver:
    def _relay_req(self) -> ProjectRequirements:
        return _req(
            components=(
                _comp("K1", value="G5LE", footprint="Relay_THT"),
                _comp("Q1", value="MMBT2222"),
                _comp("D1", value="1N4148"),
                _comp("R1", value="1k"),
            ),
            nets=(
                _net("RELAY1_COIL", ("Q1", "3"), ("K1", "A2"), ("D1", "1")),
                _net("RELAY1_DRV", ("R1", "2"), ("Q1", "1")),
                _net(
                    "GND",
                    ("Q1", "2"),
                ),
            ),
        )

    def test_transistor_attaches_to_relay_coil(self) -> None:
        attaches = _attach_map(compile_constraints(self._relay_req()))
        pa = attaches[("Q1.3", "K1.A2")]
        assert pa.max_mm == 10.0

    def test_flyback_diode_attaches_to_relay(self) -> None:
        attaches = _attach_map(compile_constraints(self._relay_req()))
        pa = attaches[("D1.1", "K1.A2")]
        assert pa.max_mm == 8.0

    def test_base_resistor_attaches_to_transistor(self) -> None:
        attaches = _attach_map(compile_constraints(self._relay_req()))
        pa = attaches[("R1.2", "Q1.1")]
        assert pa.max_mm == 6.0


# ---------------------------------------------------------------------------
# netlist: ESD / connector support
# ---------------------------------------------------------------------------


class TestConnectorSupport:
    def test_esd_diode_on_short_connector_net_attaches_to_j_pad(self) -> None:
        req = _req(
            components=(_comp("J1", footprint="USB_C"), _comp("D2", value="TVS")),
            nets=(_net("USB_DP", ("J1", "3"), ("D2", "1"), ("U1", "10")),),
        )
        attaches = _attach_map(compile_constraints(req))
        assert ("D2.1", "J1.3") in attaches

    def test_busy_net_does_not_attach(self) -> None:
        req = _req(
            components=(_comp("J1", footprint="Conn_01x04"), _comp("D2")),
            nets=(
                _net(
                    "BUS",
                    ("J1", "1"),
                    ("D2", "1"),
                    ("U1", "1"),
                    ("U2", "2"),
                ),
            ),
        )
        cs = compile_constraints(req)
        assert not cs.pin_attach  # 4 connections > 3


# ---------------------------------------------------------------------------
# netlist: sequences
# ---------------------------------------------------------------------------


class TestSequences:
    def test_sequence_from_placement_order(self) -> None:
        req = _req(
            components=(
                _comp("C7", group="buck", order=3),
                _comp("U3", group="buck", order=1),
                _comp("L2", group="buck", order=2),
                _comp("R9", group="buck"),  # no order -> excluded
            ),
        )
        cs = compile_constraints(req)
        assert len(cs.sequences) == 1
        seq = cs.sequences[0]
        assert seq.axis is Axis.HORIZONTAL
        assert seq.refs == ("U3", "L2", "C7")

    def test_relay_array_sequence(self) -> None:
        relays = tuple(_comp(f"K{i}", value="G5LE", footprint="Relay_THT") for i in (1, 2, 3, 10))
        cs = compile_constraints(_req(relays))
        assert any(s.refs == ("K1", "K2", "K3", "K10") for s in cs.sequences)

    def test_connector_array_needs_three(self) -> None:
        two = tuple(_comp(f"J{i}", value="5.08", footprint="TerminalBlock") for i in (1, 2))
        cs = compile_constraints(_req(two))
        assert not cs.sequences
        three = tuple(_comp(f"J{i}", value="5.08", footprint="TerminalBlock") for i in (1, 2, 3))
        cs = compile_constraints(_req(three))
        assert any(s.refs == ("J1", "J2", "J3") for s in cs.sequences)


# ---------------------------------------------------------------------------
# netlist: edge pins + containment
# ---------------------------------------------------------------------------


class TestEdgePins:
    def test_connector_edge_pin_by_ref_and_footprint(self) -> None:
        req = _req(
            components=(
                _comp("J1", footprint="ScrewTerminal"),
                _comp("X1", footprint="USB_C_Receptacle"),
                _comp("R1", footprint="R_0402"),
            ),
        )
        cs = compile_constraints(req)
        pinned = {e.ref for e in cs.edge_pins}
        assert pinned == {"J1", "X1"}
        assert all(e.edge is None and e.face_out for e in cs.edge_pins)

    def test_board_contain_always_present(self) -> None:
        cs = compile_constraints(_req(()))
        assert cs.contain.margin_mm == 0.5


# ---------------------------------------------------------------------------
# part rules
# ---------------------------------------------------------------------------


class TestPartRules:
    def test_esp32_keepout_owned_by_matching_component(self) -> None:
        req = _req(
            components=(
                _comp("U1", footprint="ESP32-S3-WROOM-1"),
                _comp("U2", footprint="SOIC-8"),
            ),
        )
        cs = compile_constraints(req, part_rules_path=REPO_PART_RULES)
        assert len(cs.keepouts) == 1
        ko = cs.keepouts[0]
        assert ko.owner == "U1"
        assert ko.kind is KeepoutKind.RF_ANTENNA
        assert ko.source is ConstraintSource.PART_RULE
        assert len(ko.polygon) == 4
        assert all(p.y < 0 for p in ko.polygon)  # antenna end is y-negative
        # ESP32 rule also pins the module to a board edge
        esp_edges = [e for e in cs.edge_pins if e.ref == "U1"]
        assert esp_edges and esp_edges[0].source is ConstraintSource.PART_RULE

    def test_isolation_gap_loaded(self) -> None:
        cs = compile_constraints(_req(()), part_rules_path=REPO_PART_RULES)
        assert len(cs.isolation) == 1
        gap = cs.isolation[0]
        assert (gap.domain_a, gap.domain_b, gap.min_mm) == ("MAINS", "LOGIC", 6.0)

    def test_rj45_part_rule_overrides_netlist_edge_pin(self) -> None:
        req = _req(components=(_comp("J1", footprint="RJ45_Amphenol"),))
        cs = compile_constraints(req, part_rules_path=REPO_PART_RULES)
        edges = [e for e in cs.edge_pins if e.ref == "J1"]
        assert len(edges) == 1
        assert edges[0].source is ConstraintSource.PART_RULE

    def test_malformed_part_rules_unknown_key_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "rules.json"
        bad.write_text(json.dumps({"rules": [{"match": {"ref_prefix": "K"}, "keepuot": {}}]}))
        with pytest.raises(ConfigurationError, match="keepuot"):
            compile_constraints(_req(()), part_rules_path=bad)

    def test_invalid_json_raises_configuration_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "rules.json"
        bad.write_text("{not json")
        with pytest.raises(ConfigurationError, match=str(bad)):
            compile_constraints(_req(()), part_rules_path=bad)

    def test_ref_prefix_match_is_exact(self, tmp_path: Path) -> None:
        rules = tmp_path / "rules.json"
        rules.write_text(json.dumps({"rules": [{"match": {"ref_prefix": "K"}, "edge_pin": True}]}))
        req = _req(components=(_comp("KA1", footprint="x"), _comp("K1", footprint="x")))
        cs = compile_constraints(req, part_rules_path=rules)
        assert {e.ref for e in cs.edge_pins} == {"K1"}


# ---------------------------------------------------------------------------
# feedback locks
# ---------------------------------------------------------------------------


class TestFeedbackLocks:
    def test_missing_feedback_file_is_fine(self, tmp_path: Path) -> None:
        cs = compile_constraints(_req(()), feedback_locks_path=tmp_path / "does_not_exist.json")
        assert not cs.pin_attach

    def test_feedback_lock_overrides_netlist_attach(self, tmp_path: Path) -> None:
        req = _req(
            components=(_comp("U1"), _comp("U2"), _comp("C1")),
            nets=(
                _net("+3V3", ("U1", "1"), ("U2", "3"), ("C1", "1")),
                _net("GND", ("U1", "2"), ("U2", "4"), ("C1", "2")),
            ),
        )
        locks = tmp_path / "locks.json"
        locks.write_text(
            json.dumps(
                {
                    "locks": [
                        {
                            "type": "pin_attach",
                            "src": "C1.1",
                            "dst": "U2.3",
                            "net": "+3V3",
                            "max_mm": 3.0,
                        }
                    ]
                }
            )
        )
        cs = compile_constraints(req, feedback_locks_path=locks)
        c1 = [a for a in cs.pin_attach if a.src.ref == "C1"]
        assert len(c1) == 1  # netlist C1->U1 replaced, not duplicated
        assert c1[0].dst == PadRef("U2", "3")
        assert c1[0].source is ConstraintSource.HUMAN_FEEDBACK
        assert c1[0].ideal_mm == 1.5  # defaults to max/2

    def test_feedback_lock_adds_sequence_and_edge_pin(self, tmp_path: Path) -> None:
        locks = tmp_path / "locks.json"
        locks.write_text(
            json.dumps(
                {
                    "locks": [
                        {"type": "sequence", "axis": "vertical", "refs": ["K1", "K2"]},
                        {"type": "edge_pin", "ref": "J9", "edge": "south"},
                    ]
                }
            )
        )
        cs = compile_constraints(_req(()), feedback_locks_path=locks)
        assert cs.sequences[0].axis is Axis.VERTICAL
        assert cs.sequences[0].source is ConstraintSource.HUMAN_FEEDBACK
        assert cs.edge_pins[0].edge is Edge.SOUTH
        assert cs.edge_pins[0].source is ConstraintSource.HUMAN_FEEDBACK

    def test_malformed_feedback_locks_raise_value_error_naming_path(self, tmp_path: Path) -> None:
        locks = tmp_path / "locks.json"
        locks.write_text(json.dumps({"locks": [{"type": "teleport", "ref": "J1"}]}))
        with pytest.raises(ValueError, match=str(locks)):
            compile_constraints(_req(()), feedback_locks_path=locks)

    def test_feedback_bad_json_raises_value_error(self, tmp_path: Path) -> None:
        locks = tmp_path / "locks.json"
        locks.write_text("[1,")
        with pytest.raises(ValueError, match=str(locks)):
            compile_constraints(_req(()), feedback_locks_path=locks)
