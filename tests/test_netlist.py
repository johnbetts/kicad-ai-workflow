"""Tests for kicad_pipeline.pcb.netlist."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Footprint, NetEntry, Pad, Point
from kicad_pipeline.models.requirements import (
    Component,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.pcb.netlist import (
    Netlist,
    NetlistEntry,
    assign_net_numbers_to_footprints,
    build_netlist,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_requirements(
    components: list[Component],
    nets: list[Net],
) -> ProjectRequirements:
    """Build a minimal ProjectRequirements for testing."""
    return ProjectRequirements(
        project=ProjectInfo(name="TestProject"),
        features=(),
        components=tuple(components),
        nets=tuple(nets),
    )


def _simple_component(ref: str, pin_numbers: list[str]) -> Component:
    """Build a minimal Component with the given pin numbers."""
    pins = tuple(
        Pin(number=p, name=p, pin_type=PinType.PASSIVE) for p in pin_numbers
    )
    return Component(ref=ref, value="dummy", footprint="R_0805", pins=pins)


def _smd_footprint(ref: str, pad_numbers: list[str]) -> Footprint:
    """Build a minimal SMD Footprint with the given pad numbers."""
    pads = tuple(
        Pad(
            number=pn,
            pad_type="smd",
            shape="rect",
            position=Point(0.0, 0.0),
            size_x=1.0,
            size_y=1.0,
            layers=("F.Cu",),
        )
        for pn in pad_numbers
    )
    return Footprint(
        lib_id="Device:R_0805",
        ref=ref,
        value="10k",
        position=Point(0.0, 0.0),
        pads=pads,
    )


# ---------------------------------------------------------------------------
# build_netlist — net numbering
# ---------------------------------------------------------------------------


def test_build_netlist_gnd_gets_number_1() -> None:
    """GND net must always receive net number 1."""
    comps = [_simple_component("R1", ["1", "2"])]
    nets = [
        Net(name="+3V3", connections=(NetConnection(ref="R1", pin="2"),)),
        Net(name="GND", connections=(NetConnection(ref="R1", pin="1"),)),
    ]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)
    gnd_entry = next(e for e in nl.entries if e.net.name == "GND")
    assert gnd_entry.net.number == 1


def test_build_netlist_non_gnd_starts_at_2() -> None:
    """Non-GND nets get numbers starting from 2."""
    comps = [_simple_component("R1", ["1", "2"])]
    nets = [
        Net(name="GND", connections=(NetConnection(ref="R1", pin="1"),)),
        Net(name="VCC", connections=(NetConnection(ref="R1", pin="2"),)),
    ]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)
    vcc_entry = next(e for e in nl.entries if e.net.name == "VCC")
    assert vcc_entry.net.number == 2


def test_build_netlist_multiple_nets() -> None:
    """Multiple non-GND nets are numbered consecutively from 2."""
    comps = [_simple_component("R1", ["1", "2", "3", "4"])]
    nets = [
        Net(name="GND", connections=(NetConnection(ref="R1", pin="1"),)),
        Net(name="NET_A", connections=(NetConnection(ref="R1", pin="2"),)),
        Net(name="NET_B", connections=(NetConnection(ref="R1", pin="3"),)),
        Net(name="NET_C", connections=(NetConnection(ref="R1", pin="4"),)),
    ]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)
    numbers = {e.net.name: e.net.number for e in nl.entries}
    assert numbers["GND"] == 1
    assert numbers["NET_A"] == 2
    assert numbers["NET_B"] == 3
    assert numbers["NET_C"] == 4


def test_build_netlist_unknown_component_raises() -> None:
    """build_netlist raises PCBError when a net references an unknown component."""
    from kicad_pipeline.exceptions import PCBError

    nets = [Net(name="GND", connections=(NetConnection(ref="R99", pin="1"),))]
    req = _make_requirements([], nets)
    with pytest.raises(PCBError):
        build_netlist(req)


def test_build_netlist_empty_nets() -> None:
    """build_netlist with no nets produces an empty Netlist."""
    req = _make_requirements([], [])
    nl = build_netlist(req)
    assert nl.entries == ()


def test_build_netlist_returns_netlist_type() -> None:
    """build_netlist returns a Netlist instance."""
    comps = [_simple_component("R1", ["1"])]
    nets = [Net(name="GND", connections=(NetConnection(ref="R1", pin="1"),))]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)
    assert isinstance(nl, Netlist)


# ---------------------------------------------------------------------------
# Netlist lookup helpers
# ---------------------------------------------------------------------------


def test_netlist_net_for_pad_found() -> None:
    """net_for_pad returns the correct NetEntry for a known ref+pad."""
    comps = [_simple_component("R1", ["1", "2"])]
    nets = [Net(name="GND", connections=(NetConnection(ref="R1", pin="1"),))]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)
    entry = nl.net_for_pad("R1", "1")
    assert entry is not None
    assert entry.name == "GND"
    assert entry.number == 1


def test_netlist_net_for_pad_not_found() -> None:
    """net_for_pad returns None for an unknown ref/pad combination."""
    comps = [_simple_component("R1", ["1", "2"])]
    nets = [Net(name="GND", connections=(NetConnection(ref="R1", pin="1"),))]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)
    assert nl.net_for_pad("R99", "1") is None
    assert nl.net_for_pad("R1", "99") is None


def test_netlist_pads_for_net_found() -> None:
    """pads_for_net returns all (ref, pad_number) tuples for the named net."""
    comps = [
        _simple_component("R1", ["1", "2"]),
        _simple_component("C1", ["1", "2"]),
    ]
    nets = [
        Net(
            name="GND",
            connections=(
                NetConnection(ref="R1", pin="1"),
                NetConnection(ref="C1", pin="1"),
            ),
        )
    ]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)
    pads = nl.pads_for_net("GND")
    assert ("R1", "1") in pads
    assert ("C1", "1") in pads


def test_netlist_pads_for_net_not_found() -> None:
    """pads_for_net returns an empty tuple for an unknown net name."""
    comps = [_simple_component("R1", ["1"])]
    nets = [Net(name="GND", connections=(NetConnection(ref="R1", pin="1"),))]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)
    assert nl.pads_for_net("DOES_NOT_EXIST") == ()


# ---------------------------------------------------------------------------
# assign_net_numbers_to_footprints
# ---------------------------------------------------------------------------


def test_assign_net_numbers_to_footprints_basic() -> None:
    """assign_net_numbers_to_footprints sets pad net_number for known pads."""
    comps = [_simple_component("R1", ["1", "2"])]
    nets = [Net(name="GND", connections=(NetConnection(ref="R1", pin="1"),))]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)

    footprints = [_smd_footprint("R1", ["1", "2"])]
    updated = assign_net_numbers_to_footprints(footprints, nl)
    pad1 = next(p for p in updated[0].pads if p.number == "1")
    assert pad1.net_number == 1
    assert pad1.net_name == "GND"


def test_assign_net_numbers_to_footprints_unconnected() -> None:
    """Pads with no net in the netlist get net_number=0 and net_name=None."""
    comps = [_simple_component("R1", ["1", "2"])]
    nets = [Net(name="GND", connections=(NetConnection(ref="R1", pin="1"),))]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)

    footprints = [_smd_footprint("R1", ["1", "2"])]
    updated = assign_net_numbers_to_footprints(footprints, nl)
    pad2 = next(p for p in updated[0].pads if p.number == "2")
    assert pad2.net_number == 0
    assert pad2.net_name is None


def test_assign_net_numbers_preserves_footprint_count() -> None:
    """Same number of footprints is returned after assignment."""
    comps = [
        _simple_component("R1", ["1", "2"]),
        _simple_component("C1", ["1", "2"]),
    ]
    nets = [Net(name="GND", connections=(NetConnection(ref="R1", pin="1"),))]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)

    footprints = [_smd_footprint("R1", ["1", "2"]), _smd_footprint("C1", ["1", "2"])]
    updated = assign_net_numbers_to_footprints(footprints, nl)
    assert len(updated) == len(footprints)


def test_gnd_aliases_get_distinct_numbers() -> None:
    """AGND, DGND, PGND are separate ground domains with distinct net numbers.

    Only canonical ``"GND"`` receives net number 1.  Separate ground domains
    must keep their own numbers for correct PCB pad assignments.
    """
    comps = [_simple_component("R1", ["1", "2", "3"])]
    nets = [
        Net(name="AGND", connections=(NetConnection(ref="R1", pin="1"),)),
        Net(name="DGND", connections=(NetConnection(ref="R1", pin="2"),)),
        Net(name="PGND", connections=(NetConnection(ref="R1", pin="3"),)),
    ]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)
    numbers = {entry.net.name: entry.net.number for entry in nl.entries}
    # Each alias gets a unique number (none should be 1 since "GND" isn't present)
    assert len(set(numbers.values())) == 3, f"Expected 3 distinct net numbers, got {numbers}"
    assert all(n >= 2 for n in numbers.values()), f"Only canonical GND gets net 1: {numbers}"


def test_assign_net_numbers_immutable() -> None:
    """Original Footprint objects are unchanged after assignment (frozen dataclass)."""
    comps = [_simple_component("R1", ["1"])]
    nets = [Net(name="GND", connections=(NetConnection(ref="R1", pin="1"),))]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)

    original = [_smd_footprint("R1", ["1"])]
    updated = assign_net_numbers_to_footprints(original, nl)
    # Returned list contains new objects
    assert updated[0] is not original[0]
    # Original pads still have net_number=None (frozen, unchanged)
    assert original[0].pads[0].net_number is None


def test_netlist_entry_is_frozen() -> None:
    """NetlistEntry is a frozen dataclass — attribute assignment raises."""
    entry = NetlistEntry(
        net=NetEntry(number=1, name="GND"),
        pad_refs=(("R1", "1"),),
    )
    with pytest.raises(AttributeError):
        entry.net = NetEntry(number=2, name="VCC")  # type: ignore[misc]


def test_netlist_is_frozen() -> None:
    """Netlist is a frozen dataclass — attribute assignment raises."""
    nl = Netlist(entries=())
    with pytest.raises(AttributeError):
        nl.entries = ()  # type: ignore[misc]


def test_build_netlist_basic_entry_count() -> None:
    """Netlist from 2-net requirements has 2 entries."""
    comps = [
        _simple_component("R1", ["1", "2"]),
        _simple_component("C1", ["1", "2"]),
    ]
    nets = [
        Net(
            name="GND",
            connections=(NetConnection(ref="R1", pin="1"), NetConnection(ref="C1", pin="1")),
        ),
        Net(
            name="+3V3",
            connections=(NetConnection(ref="R1", pin="2"), NetConnection(ref="C1", pin="2")),
        ),
    ]
    req = _make_requirements(comps, nets)
    nl = build_netlist(req)
    assert len(nl.entries) == 2
