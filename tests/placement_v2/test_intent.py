"""Tests for the board-intent draft generator (inference proposes)."""

from __future__ import annotations

from kicad_pipeline.models.requirements import (
    BoardIntent,
    Component,
    ConnectorIntent,
    Net,
    NetConnection,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.placement_v2.intent import (
    FIELD_WIRING_EDGE,
    IO_EDGE,
    draft_board_intent,
)
from kicad_pipeline.requirements.decomposer import (
    requirements_from_dict,
    requirements_to_dict,
)


def _comp(ref: str, footprint: str) -> Component:
    return Component(ref=ref, value="v", footprint=footprint)


def _req(components: tuple[Component, ...]) -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        features=(),
        components=components,
        nets=(),
    )


class TestDraftBoardIntent:
    def test_field_wiring_and_io_claim_opposite_edges(self) -> None:
        draft = draft_board_intent(_req((
            _comp("J1", "TerminalBlock_5.08mm_6P"),
            _comp("J2", "USB_C_Receptacle"),
            _comp("R1", "R_0402"),
        )))
        by_ref = {c.ref: c for c in draft.connectors}
        assert by_ref["J1"].edge == FIELD_WIRING_EDGE
        assert by_ref["J2"].edge == IO_EDGE
        assert "R1" not in by_ref

    def test_extra_io_refs_join_io_cohort(self) -> None:
        draft = draft_board_intent(
            _req((_comp("U3", "ESP32-S3-WROOM"),)), extra_io_refs=("U3",),
        )
        assert draft.get_connector("U3") is not None
        assert draft.get_connector("U3").edge == IO_EDGE

    def test_draft_never_declares_pin_freedom(self) -> None:
        draft = draft_board_intent(_req((
            _comp("J1", "TerminalBlock_2P"),
            _comp("J2", "PinHeader_1x06"),
        )))
        assert all(c.pins_interchangeable is None for c in draft.connectors)

    def test_nl_s_3c_locks_are_the_regression_target(self) -> None:
        """Council 2026-06-11: every hand-authored lock is a compiler
        regression test. The retired placement_locks.json pinned
        J1/J3-J6 north, J2/J13/J15/J16/U3 south, J14 east. The draft
        must derive the two cohorts; J14's EAST is the one entry that
        stays a human declaration (the draft proposes south — the
        display ribbon is genuine intent the netlist cannot know).
        """
        draft = draft_board_intent(
            _req((
                _comp("J1", "TerminalBlock_5.08mm_6P"),
                _comp("J3", "TerminalBlock_5.08mm_4P"),
                _comp("J4", "TerminalBlock_3.5mm_2P"),
                _comp("J5", "TerminalBlock_3.5mm_2P"),
                _comp("J6", "TerminalBlock_3.5mm_2P"),
                _comp("J2", "USB_C_Receptacle_HRO"),
                _comp("J13", "RJ45_RJHSE538X"),
                _comp("J15", "PinHeader_1x10"),
                _comp("J16", "microSD_Card_Socket"),
                _comp("J14", "PinHeader_1x14"),
            )),
            extra_io_refs=("U3",),
        )
        by_ref = {c.ref: c.edge for c in draft.connectors}
        assert {r: by_ref[r] for r in ("J1", "J3", "J4", "J5", "J6")} == {
            r: "north" for r in ("J1", "J3", "J4", "J5", "J6")
        }
        assert {r: by_ref[r] for r in ("J2", "J13", "J15", "J16", "U3")} == {
            r: "south" for r in ("J2", "J13", "J15", "J16", "U3")
        }
        # Inference proposes south for J14; the human-confirmed intent
        # declares east. Declaration governs — the draft is a proposal.
        assert by_ref["J14"] == "south"


class TestBoardIntentSerialization:
    def test_round_trip(self) -> None:
        req = ProjectRequirements(
            project=ProjectInfo(name="t"),
            features=(),
            components=(
                _comp("J14", "PinHeader_1x14"),
                _comp("U3", "ESP32"),
            ),
            nets=(Net(
                name="SPI",
                connections=(NetConnection("J14", "1"), NetConnection("U3", "2")),
            ),),
            board_intent=BoardIntent(connectors=(
                ConnectorIntent("J14", "east", pins_interchangeable=False),
                ConnectorIntent("J9"),
                ConnectorIntent("J4", "north", pins_interchangeable=True),
            )),
        )
        again = requirements_from_dict(requirements_to_dict(req))
        assert again.board_intent == req.board_intent

    def test_absent_intent_round_trips_as_none(self) -> None:
        req = ProjectRequirements(
            project=ProjectInfo(name="t"),
            features=(),
            components=(_comp("R1", "R_0402"),),
            nets=(),
        )
        again = requirements_from_dict(requirements_to_dict(req))
        assert again.board_intent is None


class TestDraftIsolationRegions:
    """Ferrite-subtree region drafts (oracle: the human reference board
    passes the drafted nl-s-3c regions with zero violations)."""

    @staticmethod
    def _ferrite_board() -> ProjectRequirements:
        from kicad_pipeline.models.requirements import Pin, PinType

        comps = (
            Component(ref="U1", value="reg", footprint="SOT-223", pins=(
                Pin(number="1", name="OUT", pin_type=PinType.POWER_OUT),
            )),
            _comp("L3", "L_0805"),
            Component(ref="L7", value="Ferrite_600R", footprint="L_0805"),
            Component(ref="L8", value="4.7uH", footprint="L_1210"),
            _comp("K1", "Relay"), _comp("Q1", "SOT-23"), _comp("R1", "R_0402"),
            _comp("C1", "C_0402"), _comp("D1", "LED_0603"),
            _comp("J9", "PinHeader_1x02"),
        )
        nets = (
            # +5V is the main rail (regulator POWER_OUT + canonical name).
            Net(name="+5V", connections=(
                NetConnection("U1", "1"), NetConnection("L7", "1"),
                NetConnection("L8", "1"),
            )),
            # RELAY_5V: private rail behind ferrite L7, 3 members.
            Net(name="RELAY_5V", connections=(
                NetConnection("L7", "2"), NetConnection("K1", "1"),
                NetConnection("Q1", "1"), NetConnection("C1", "1"),
            )),
            # Buck inductor L8 bridges to a 2-member node: no region.
            Net(name="SW_NODE", connections=(
                NetConnection("L8", "2"), NetConnection("R1", "1"),
            )),
            # Passive chain: D1 hangs off region member Q1 via a 2-pin
            # signal net -> closure 2 pulls it in.
            Net(name="LED_SIG", connections=(
                NetConnection("Q1", "2"), NetConnection("D1", "1"),
            )),
            # Connector whose only signal partners are region members ->
            # closure 3 pulls it in.
            Net(name="COIL_OUT", connections=(
                NetConnection("J9", "1"), NetConnection("K1", "3"),
            )),
        )
        return ProjectRequirements(
            project=ProjectInfo(name="t"), features=(),
            components=comps, nets=nets,
        )

    def test_ferrite_subtree_drafts_region(self) -> None:
        from kicad_pipeline.placement_v2.intent import draft_isolation_regions

        regions = draft_isolation_regions(self._ferrite_board())
        assert len(regions) == 1
        region = regions[0]
        assert region.name == "RELAY_5V"
        assert region.boundary_refs == ("L7",)
        assert "K1" in region.refs and "Q1" in region.refs and "C1" in region.refs

    def test_passive_chain_and_connector_closures(self) -> None:
        from kicad_pipeline.placement_v2.intent import draft_isolation_regions

        region = draft_isolation_regions(self._ferrite_board())[0]
        assert "D1" in region.refs  # closure 2: 2-pin signal chain
        assert "J9" in region.refs  # closure 3: all partners inside

    def test_power_inductor_never_drafts_region(self) -> None:
        from kicad_pipeline.placement_v2.intent import draft_isolation_regions

        regions = draft_isolation_regions(self._ferrite_board())
        assert all("SW_NODE" not in r.name for r in regions)
