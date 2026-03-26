"""Tests for optimization.constraint_resolver."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import OrderingChain, PlacementConstraintSet, ProximityConstraint
from kicad_pipeline.models.requirements import Component, ProjectInfo, ProjectRequirements
from kicad_pipeline.optimization.constraint_resolver import resolve_constraints

if TYPE_CHECKING:
    import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_req(*components: Component) -> ProjectRequirements:
    """Build a minimal ProjectRequirements from the given component list."""
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        features=(),
        components=components,
        nets=(),
    )


def _comp(
    ref: str,
    *,
    group: str | None = None,
    near: str | None = None,
    order: int | None = None,
    near_max_mm: float | None = None,
) -> Component:
    return Component(
        ref=ref,
        value="10k",
        footprint="R_0402",
        placement_group=group,
        placement_near=near,
        placement_order=order,
        placement_near_max_mm=near_max_mm,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resolve_empty_requirements() -> None:
    """Empty requirements produce an empty constraint set."""
    req = _make_req()
    result = resolve_constraints(req)
    assert isinstance(result, PlacementConstraintSet)
    assert result.groups == ()
    assert result.proximity == ()
    assert result.ordering == ()
    assert result.trace_length_match == ()


def test_resolve_explicit_group() -> None:
    """Components with placement_group end up in the groups tuple."""
    r1 = _comp("R1", group="power_stage")
    r2 = _comp("R2", group="power_stage")
    c1 = _comp("C1", group="power_stage")
    req = _make_req(r1, r2, c1)
    result = resolve_constraints(req)

    assert len(result.groups) == 1
    group_name, refs = result.groups[0]
    assert group_name == "power_stage"
    assert set(refs) == {"R1", "R2", "C1"}


def test_resolve_explicit_group_multiple_groups() -> None:
    """Components in different groups appear in distinct group entries."""
    r1 = _comp("R1", group="stage_a")
    r2 = _comp("R2", group="stage_b")
    req = _make_req(r1, r2)
    result = resolve_constraints(req)

    assert len(result.groups) == 2
    names = {g[0] for g in result.groups}
    assert names == {"stage_a", "stage_b"}


def test_resolve_explicit_proximity() -> None:
    """Component with placement_near='U1:VIN' → ProximityConstraint with target_pin."""
    c1 = _comp("C1", near="U1:VIN")
    req = _make_req(Component(ref="U1", value="IC", footprint="SOIC-8"), c1)
    result = resolve_constraints(req)

    assert len(result.proximity) == 1
    pc = result.proximity[0]
    assert isinstance(pc, ProximityConstraint)
    assert pc.ref == "C1"
    assert pc.target_ref == "U1"
    assert pc.target_pin == "VIN"
    assert pc.max_distance_mm == 5.0


def test_resolve_proximity_custom_max_mm() -> None:
    """placement_near_max_mm overrides the default distance."""
    c1 = _comp("C1", near="U1:VIN", near_max_mm=2.5)
    req = _make_req(Component(ref="U1", value="IC", footprint="SOIC-8"), c1)
    result = resolve_constraints(req)

    assert result.proximity[0].max_distance_mm == 2.5


def test_resolve_near_without_pin() -> None:
    """placement_near='U1' (no colon) → target_pin is None."""
    c1 = _comp("C1", near="U1")
    req = _make_req(Component(ref="U1", value="IC", footprint="SOIC-8"), c1)
    result = resolve_constraints(req)

    pc = result.proximity[0]
    assert pc.target_ref == "U1"
    assert pc.target_pin is None


def test_resolve_explicit_ordering() -> None:
    """Components with placement_order in the same group form an OrderingChain."""
    r1 = _comp("R1", group="divider", order=1)
    r2 = _comp("R2", group="divider", order=2)
    req = _make_req(r1, r2)
    result = resolve_constraints(req)

    assert len(result.ordering) == 1
    chain = result.ordering[0]
    assert isinstance(chain, OrderingChain)
    assert chain.group == "divider"
    assert chain.refs == ("R1", "R2")


def test_resolve_ordering_sorted_by_order_value() -> None:
    """Ordering chain is sorted by placement_order, not declaration order."""
    r2 = _comp("R2", group="divider", order=2)
    r1 = _comp("R1", group="divider", order=1)
    r3 = _comp("R3", group="divider", order=3)
    req = _make_req(r2, r1, r3)
    result = resolve_constraints(req)

    chain = result.ordering[0]
    assert chain.refs == ("R1", "R2", "R3")


def test_resolve_no_subcircuits() -> None:
    """resolve_constraints works fine when subcircuits=None."""
    r1 = _comp("R1", group="foo")
    req = _make_req(r1)
    result = resolve_constraints(req, subcircuits=None)
    assert result.groups[0][0] == "foo"


def test_resolve_subcircuits_linear_inferred_ordering() -> None:
    """Linear subcircuit infers an OrderingChain for non-explicitly-grouped refs."""
    from kicad_pipeline.optimization.functional_grouper import (
        DetectedSubCircuit,
        SubCircuitType,
        VoltageDomain,
    )

    sc = DetectedSubCircuit(
        circuit_type=SubCircuitType.RC_FILTER,
        refs=("R1", "C1"),
        anchor_ref="R1",
        net_connections=("NET1", "NET2"),
        domain=VoltageDomain.DIGITAL_3V3,
        layout_hint="linear",
    )
    req = _make_req(
        Component(ref="R1", value="1k", footprint="R_0402"),
        Component(ref="C1", value="100nF", footprint="C_0402"),
    )
    result = resolve_constraints(req, subcircuits=[sc])

    assert len(result.ordering) == 1
    chain = result.ordering[0]
    assert set(chain.refs) == {"R1", "C1"}


def test_resolve_merge_explicit_wins() -> None:
    """Explicit placement_order overrides subcircuit-inferred ordering."""
    from kicad_pipeline.optimization.functional_grouper import (
        DetectedSubCircuit,
        SubCircuitType,
        VoltageDomain,
    )

    # Subcircuit says R1, C1 (linear order)
    sc = DetectedSubCircuit(
        circuit_type=SubCircuitType.RC_FILTER,
        refs=("R1", "C1"),
        anchor_ref="R1",
        net_connections=("NET1",),
        domain=VoltageDomain.DIGITAL_3V3,
        layout_hint="linear",
    )
    # Explicit says C1 first, then R1 (reversed) in group "myfilter"
    r1 = _comp("R1", group="myfilter", order=2)
    c1 = _comp("C1", group="myfilter", order=1)
    req = _make_req(r1, c1)

    result = resolve_constraints(req, subcircuits=[sc])

    # Should have the explicit ordering chain (C1 before R1)
    explicit_chains = [ch for ch in result.ordering if ch.group == "myfilter"]
    assert len(explicit_chains) == 1
    assert explicit_chains[0].refs == ("C1", "R1")


_RESOLVER_LOGGER = "kicad_pipeline.optimization.constraint_resolver"


def test_resolve_missing_near_target_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """placement_near referencing nonexistent ref logs a warning, doesn't raise."""
    c1 = _comp("C1", near="UNONEXISTENT:VCC")
    req = _make_req(c1)

    with caplog.at_level(logging.WARNING, logger=_RESOLVER_LOGGER):
        result = resolve_constraints(req)

    assert any("UNONEXISTENT" in r.message for r in caplog.records)
    # Still produces a ProximityConstraint (best-effort)
    assert len(result.proximity) == 1


def test_resolve_group_ordering_not_created_for_single_member() -> None:
    """A group with only one ordered component does NOT produce an OrderingChain."""
    r1 = _comp("R1", group="solo", order=1)
    req = _make_req(r1)
    result = resolve_constraints(req)

    # Group should exist but no ordering chain (need ≥2 for a chain)
    assert len(result.groups) == 1
    assert result.ordering == ()


def test_resolve_multiple_proximity_constraints() -> None:
    """Multiple components can each have their own proximity constraint."""
    u1 = Component(ref="U1", value="IC", footprint="SOIC-8")
    c1 = _comp("C1", near="U1:VIN")
    c2 = _comp("C2", near="U1:GND", near_max_mm=3.0)
    req = _make_req(u1, c1, c2)
    result = resolve_constraints(req)

    assert len(result.proximity) == 2
    refs = {pc.ref for pc in result.proximity}
    assert refs == {"C1", "C2"}
