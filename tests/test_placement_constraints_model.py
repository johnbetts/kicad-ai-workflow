"""Tests for Phase 1 placement constraint model fields."""

from __future__ import annotations

import dataclasses

from kicad_pipeline.models.pcb import (
    Footprint,
    OrderingChain,
    PlacementConstraintSet,
    Point,
    ProximityConstraint,
    TraceLengthPair,
)
from kicad_pipeline.models.requirements import Component

# ---------------------------------------------------------------------------
# Component placement fields
# ---------------------------------------------------------------------------


def test_component_placement_fields_default_none() -> None:
    """Component() with no placement args has all placement fields as None."""
    c = Component(ref="R1", value="10k", footprint="R_0402")
    assert c.placement_group is None
    assert c.placement_near is None
    assert c.placement_order is None
    assert c.placement_near_max_mm is None


def test_component_placement_fields_set() -> None:
    """Component() accepts explicit placement field values."""
    c = Component(
        ref="C1",
        value="100nF",
        footprint="C_0402",
        placement_group="buck_input_stage",
        placement_near="U1:VIN",
        placement_order=2,
        placement_near_max_mm=3.5,
    )
    assert c.placement_group == "buck_input_stage"
    assert c.placement_near == "U1:VIN"
    assert c.placement_order == 2
    assert c.placement_near_max_mm == 3.5


def test_component_replace_adds_placement() -> None:
    """dataclasses.replace() can add placement fields to an existing Component."""
    base = Component(ref="R2", value="4.7k", footprint="R_0402")
    updated = dataclasses.replace(
        base,
        placement_group="relay_driver",
        placement_order=1,
    )
    assert updated.ref == "R2"
    assert updated.placement_group == "relay_driver"
    assert updated.placement_order == 1
    assert updated.placement_near is None
    assert updated.placement_near_max_mm is None


# ---------------------------------------------------------------------------
# Footprint custom_properties field
# ---------------------------------------------------------------------------


def test_footprint_custom_properties_default_empty() -> None:
    """Footprint() with no custom_properties has empty tuple."""
    fp = Footprint(lib_id="R:R_0402", ref="R1", value="10k", position=Point(0.0, 0.0))
    assert fp.custom_properties == ()


def test_footprint_custom_properties_set() -> None:
    """Footprint() accepts custom_properties as tuple of (name, value) pairs."""
    fp = Footprint(
        lib_id="R:R_0402",
        ref="R1",
        value="10k",
        position=Point(0.0, 0.0),
        custom_properties=(("PlacementGroup", "buck"), ("PlacementOrder", "1")),
    )
    assert fp.custom_properties == (("PlacementGroup", "buck"), ("PlacementOrder", "1"))


# ---------------------------------------------------------------------------
# ProximityConstraint
# ---------------------------------------------------------------------------


def test_proximity_constraint() -> None:
    """ProximityConstraint stores ref, target, optional pin, and distance."""
    pc = ProximityConstraint(ref="C1", target_ref="U1", target_pin="VIN", max_distance_mm=4.0)
    assert pc.ref == "C1"
    assert pc.target_ref == "U1"
    assert pc.target_pin == "VIN"
    assert pc.max_distance_mm == 4.0


def test_proximity_constraint_defaults() -> None:
    """ProximityConstraint default max_distance_mm is 5.0 and target_pin is None."""
    pc = ProximityConstraint(ref="C2", target_ref="U2")
    assert pc.target_pin is None
    assert pc.max_distance_mm == 5.0


# ---------------------------------------------------------------------------
# OrderingChain
# ---------------------------------------------------------------------------


def test_ordering_chain() -> None:
    """OrderingChain stores group name and ordered refs tuple."""
    chain = OrderingChain(group="voltage_divider", refs=("R1", "R2", "C1"))
    assert chain.group == "voltage_divider"
    assert chain.refs == ("R1", "R2", "C1")


def test_ordering_chain_defaults() -> None:
    """OrderingChain defaults to empty refs tuple."""
    chain = OrderingChain(group="my_group")
    assert chain.refs == ()


# ---------------------------------------------------------------------------
# TraceLengthPair
# ---------------------------------------------------------------------------


def test_trace_length_pair() -> None:
    """TraceLengthPair stores net names and max skew."""
    pair = TraceLengthPair(net_a="CLK+", net_b="CLK-", max_skew_mm=0.25)
    assert pair.net_a == "CLK+"
    assert pair.net_b == "CLK-"
    assert pair.max_skew_mm == 0.25


def test_trace_length_pair_default_skew() -> None:
    """TraceLengthPair default max_skew_mm is 0.5."""
    pair = TraceLengthPair(net_a="D+", net_b="D-")
    assert pair.max_skew_mm == 0.5


# ---------------------------------------------------------------------------
# PlacementConstraintSet
# ---------------------------------------------------------------------------


def test_placement_constraint_set_empty() -> None:
    """PlacementConstraintSet() with no args has all-empty tuple fields."""
    pcs = PlacementConstraintSet()
    assert pcs.groups == ()
    assert pcs.proximity == ()
    assert pcs.ordering == ()
    assert pcs.trace_length_match == ()


def test_placement_constraint_set_with_data() -> None:
    """PlacementConstraintSet accepts and stores all constraint types."""
    pcs = PlacementConstraintSet(
        groups=(
            ("buck_stage", ("U1", "C1", "C2", "L1")),
            ("relay_driver", ("Q1", "D1", "R1")),
        ),
        proximity=(
            ProximityConstraint(ref="C1", target_ref="U1", target_pin="VIN"),
            ProximityConstraint(ref="C2", target_ref="U1", target_pin="GND", max_distance_mm=2.0),
        ),
        ordering=(OrderingChain(group="buck_stage", refs=("C1", "L1", "C2")),),
        trace_length_match=(TraceLengthPair(net_a="DIFF+", net_b="DIFF-"),),
    )
    assert len(pcs.groups) == 2
    assert pcs.groups[0][0] == "buck_stage"
    assert pcs.groups[0][1] == ("U1", "C1", "C2", "L1")
    assert len(pcs.proximity) == 2
    assert pcs.proximity[1].max_distance_mm == 2.0
    assert len(pcs.ordering) == 1
    assert pcs.ordering[0].refs == ("C1", "L1", "C2")
    assert len(pcs.trace_length_match) == 1
    assert pcs.trace_length_match[0].net_b == "DIFF-"
