"""Tests for kicad_pipeline.optimization.subnet_placer."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import (
        Component,
        Net,
    )

from kicad_pipeline.optimization.functional_grouper import (
    DetectedSubCircuit,
    SubCircuitType,
    VoltageDomain,
)
from kicad_pipeline.optimization.placement_types import PlacementContext
from kicad_pipeline.optimization.subnet_placer import (
    _chain_series_components,
    _phase_subnet_placement,
)
from tests.helpers import (
    make_component,
    make_footprint,
    make_pcb_design,
    make_requirements,
)

# ---------------------------------------------------------------------------
# Fake subnet_resolver dataclass & helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeSubnetConnection:
    """Stand-in for SubnetConnection from the not-yet-created subnet_resolver."""

    passive_ref: str
    passive_pin: str
    ic_ref: str
    ic_pin: str
    subnet_name: str
    role: str


def _make_fake_resolver_module(
    connections: list[_FakeSubnetConnection] | None = None,
) -> types.ModuleType:
    """Build a fake subnet_resolver module with configurable behaviour."""
    mod = types.ModuleType("kicad_pipeline.optimization.subnet_resolver")

    def resolve_subnets(requirements: object) -> list[_FakeSubnetConnection]:
        return connections if connections is not None else []

    def resolve_ic_pin_position(
        ic_ref: str,
        ic_pin: str,
        pcb: object,
    ) -> tuple[float, float, str]:
        # Default: IC pin is at (20, 20) on the south side
        return (20.0, 20.0, "S")

    def compute_pad_facing_position(
        passive_size: tuple[float, float],
        ic_pin_x: float,
        ic_pin_y: float,
        ic_pin_side: str,
        gap_mm: float,
    ) -> tuple[float, float, float]:
        # Place passive below the IC pin (south side)
        return (ic_pin_x, ic_pin_y + gap_mm + passive_size[1] / 2.0, 0.0)

    mod.resolve_subnets = resolve_subnets  # type: ignore[attr-defined]
    mod.resolve_ic_pin_position = resolve_ic_pin_position  # type: ignore[attr-defined]
    mod.compute_pad_facing_position = compute_pad_facing_position  # type: ignore[attr-defined]
    mod.SubnetConnection = _FakeSubnetConnection  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_context(
    components: tuple[Component, ...] | None = None,
    nets: tuple[Net, ...] = (),
    positions: dict[str, tuple[float, float, float]] | None = None,
    fp_sizes: dict[str, tuple[float, float]] | None = None,
    fixed_refs: set[str] | None = None,
    subcircuits: list[DetectedSubCircuit] | None = None,
) -> PlacementContext:
    """Build a minimal PlacementContext for testing."""
    if components is None:
        components = (
            make_component("U1", "ESP32", "QFN-48"),
            make_component("C1", "100nF", "C_0402"),
        )

    reqs = make_requirements(components=components, nets=nets)

    pcb_fps = tuple(make_footprint(c.ref, x=10.0, y=10.0) for c in components)
    pcb = make_pcb_design(footprints=pcb_fps)

    if positions is None:
        positions = {c.ref: (10.0, 10.0, 0.0) for c in components}
    if fp_sizes is None:
        fp_sizes = {c.ref: (1.6, 0.8) for c in components}
        # Make ICs bigger
        for c in components:
            if c.ref.startswith("U"):
                fp_sizes[c.ref] = (8.0, 8.0)
    if fixed_refs is None:
        fixed_refs = set()
    if subcircuits is None:
        subcircuits = []

    return PlacementContext(
        positions=positions,
        fp_sizes=fp_sizes,
        bounds=(0.0, 0.0, 80.0, 40.0),
        fixed_refs=fixed_refs,
        requirements=reqs,
        initial_pcb=pcb,
        zones=[],
        subcircuits=subcircuits,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhaseSubnetPlacementDecoupling:
    """Cap placed facing IC VCC pin."""

    def test_cap_moved_to_face_ic_pin(self) -> None:
        """Decoupling cap C1 should be placed near U1's VCC pin."""
        comps = (
            make_component("U1", "ESP32", "QFN-48"),
            make_component("C1", "100nF", "C_0402"),
        )
        ctx = _make_context(
            components=comps,
            positions={"U1": (20.0, 20.0, 0.0), "C1": (50.0, 50.0, 0.0)},
        )

        connections = [
            _FakeSubnetConnection(
                passive_ref="C1",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="VCC",
                subnet_name="VCC_subnet",
                role="decoupling",
            ),
        ]
        fake_mod = _make_fake_resolver_module(connections)

        with patch.dict(
            sys.modules,
            {
                "kicad_pipeline.optimization.subnet_resolver": fake_mod,
            },
        ):
            _phase_subnet_placement(ctx)

        # C1 should have moved closer to U1 (was at 50,50)
        c1_x, c1_y, _rot = ctx.positions["C1"]
        assert c1_x == pytest.approx(20.0, abs=5.0)
        # Should be below IC pin (south side in our fake)
        assert c1_y > 20.0


class TestPhaseSubnetPlacementNoSubnets:
    """No-op when no subnet connections are resolved."""

    def test_no_op_when_no_connections(self) -> None:
        ctx = _make_context()
        original_positions = dict(ctx.positions)

        fake_mod = _make_fake_resolver_module(connections=[])

        with patch.dict(
            sys.modules,
            {
                "kicad_pipeline.optimization.subnet_resolver": fake_mod,
            },
        ):
            _phase_subnet_placement(ctx)

        # Positions should not change
        assert ctx.positions == original_positions

    def test_no_op_when_resolver_missing(self) -> None:
        """If subnet_resolver is not importable, phase is a no-op."""
        ctx = _make_context()
        original_positions = dict(ctx.positions)

        # Ensure the module is NOT available
        with patch.dict(
            sys.modules,
            {
                "kicad_pipeline.optimization.subnet_resolver": None,
            },
        ):
            # The import inside _phase_subnet_placement will raise ImportError
            # when the module value is None in sys.modules
            _phase_subnet_placement(ctx)

        assert ctx.positions == original_positions


class TestPhaseSubnetPlacementPreservesFixedRefs:
    """Fixed refs should not be moved by subnet placement."""

    def test_fixed_ref_not_moved(self) -> None:
        comps = (
            make_component("U1", "ESP32", "QFN-48"),
            make_component("C1", "100nF", "C_0402"),
        )
        ctx = _make_context(
            components=comps,
            positions={"U1": (20.0, 20.0, 0.0), "C1": (50.0, 50.0, 0.0)},
            fixed_refs={"C1"},
        )

        connections = [
            _FakeSubnetConnection(
                passive_ref="C1",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="VCC",
                subnet_name="VCC_subnet",
                role="decoupling",
            ),
        ]
        fake_mod = _make_fake_resolver_module(connections)

        with patch.dict(
            sys.modules,
            {
                "kicad_pipeline.optimization.subnet_resolver": fake_mod,
            },
        ):
            _phase_subnet_placement(ctx)

        # C1 is fixed — should NOT have moved
        assert ctx.positions["C1"] == (50.0, 50.0, 0.0)


class TestGenericPhaseWorksForRelay:
    """Relay driver subcircuit gets ordered and placed."""

    def test_relay_driver_components_placed(self) -> None:
        comps = (
            make_component("U1", "ESP32", "QFN-48"),
            make_component("R1", "1k", "R_0805"),
            make_component("Q1", "2N7002", "SOT-23"),
            make_component("K1", "G5V-1", "Relay"),
        )
        ctx = _make_context(
            components=comps,
            positions={
                "U1": (20.0, 20.0, 0.0),
                "R1": (5.0, 5.0, 0.0),
                "Q1": (60.0, 5.0, 0.0),
                "K1": (60.0, 35.0, 0.0),
            },
            subcircuits=[
                DetectedSubCircuit(
                    circuit_type=SubCircuitType.RELAY_DRIVER,
                    refs=("R1", "Q1", "K1"),
                    anchor_ref="K1",
                    net_connections=(),
                    domain=VoltageDomain.POWER_5V,
                ),
            ],
        )

        connections = [
            _FakeSubnetConnection(
                passive_ref="R1",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="GPIO4",
                subnet_name="relay_gate",
                role="gate_resistor",
            ),
        ]
        fake_mod = _make_fake_resolver_module(connections)

        with patch.dict(
            sys.modules,
            {
                "kicad_pipeline.optimization.subnet_resolver": fake_mod,
            },
        ):
            _phase_subnet_placement(ctx)

        # R1 should have been placed near U1 (was at 5,5)
        r1_x, r1_y, _ = ctx.positions["R1"]
        assert r1_x == pytest.approx(20.0, abs=5.0)
        # R1 should now be in fixed_refs
        assert "R1" in ctx.fixed_refs


class TestGenericPhaseWorksForBuck:
    """Buck converter subcircuit gets ordered and chained."""

    def test_buck_converter_chain(self) -> None:
        comps = (
            make_component("U1", "TPS54302", "SOT-23-6"),
            make_component("C1", "10uF", "C_0805"),
            make_component("L1", "4.7uH", "L_0805"),
            make_component("C2", "22uF", "C_0805"),
        )
        ctx = _make_context(
            components=comps,
            positions={
                "U1": (20.0, 20.0, 0.0),
                "C1": (5.0, 5.0, 0.0),
                "L1": (60.0, 30.0, 0.0),
                "C2": (70.0, 5.0, 0.0),
            },
            subcircuits=[
                DetectedSubCircuit(
                    circuit_type=SubCircuitType.BUCK_CONVERTER,
                    refs=("C1", "U1", "L1", "C2"),
                    anchor_ref="U1",
                    net_connections=(),
                    domain=VoltageDomain.POWER_5V,
                ),
            ],
        )

        connections = [
            _FakeSubnetConnection(
                passive_ref="C1",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="VIN",
                subnet_name="vin",
                role="input_cap",
            ),
            _FakeSubnetConnection(
                passive_ref="L1",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="SW",
                subnet_name="sw",
                role="inductor",
            ),
            _FakeSubnetConnection(
                passive_ref="C2",
                passive_pin="1",
                ic_ref="U1",
                ic_pin="VOUT",
                subnet_name="vout",
                role="output_cap",
            ),
        ]
        fake_mod = _make_fake_resolver_module(connections)

        with patch.dict(
            sys.modules,
            {
                "kicad_pipeline.optimization.subnet_resolver": fake_mod,
            },
        ):
            _phase_subnet_placement(ctx)

        # All passives should have been placed and marked fixed
        assert "C1" in ctx.fixed_refs
        assert "L1" in ctx.fixed_refs
        assert "C2" in ctx.fixed_refs

        # Passives should be near IC (within reasonable distance)
        for ref in ("C1", "L1", "C2"):
            rx, ry, _ = ctx.positions[ref]
            # Should be within ~25mm of IC at (20, 20)
            dist = ((rx - 20.0) ** 2 + (ry - 20.0) ** 2) ** 0.5
            assert dist < 25.0, f"{ref} too far from IC: {dist:.1f}mm"


# ---------------------------------------------------------------------------
# _chain_series_components unit tests
# ---------------------------------------------------------------------------


class TestChainSeriesComponents:
    """Unit tests for the series-chain layout helper."""

    def test_chains_forward_from_anchor(self) -> None:
        """Components after anchor should be placed sequentially."""
        comps = (
            make_component("R1", "10k", "R_0805"),
            make_component("R2", "4.7k", "R_0805"),
            make_component("R3", "1k", "R_0805"),
        )
        ctx = _make_context(
            components=comps,
            positions={
                "R1": (10.0, 20.0, 0.0),
                "R2": (50.0, 50.0, 0.0),
                "R3": (60.0, 60.0, 0.0),
            },
            fp_sizes={"R1": (1.6, 0.8), "R2": (1.6, 0.8), "R3": (1.6, 0.8)},
        )

        placed: set[str] = set()
        _chain_series_components(ctx, ["R1", "R2", "R3"], placed)

        # R2 and R3 should be to the right of R1
        r1_x = ctx.positions["R1"][0]
        r2_x = ctx.positions["R2"][0]
        r3_x = ctx.positions["R3"][0]
        assert r2_x > r1_x
        assert r3_x > r2_x
        # All at same Y as anchor
        assert ctx.positions["R2"][1] == pytest.approx(20.0)
        assert ctx.positions["R3"][1] == pytest.approx(20.0)

    def test_empty_flow_order_no_crash(self) -> None:
        """Empty flow order should not crash."""
        ctx = _make_context()
        placed: set[str] = set()
        _chain_series_components(ctx, [], placed)
        # No assertion needed — just verify no exception

    def test_single_ref_no_crash(self) -> None:
        """Single ref in flow order — nothing to chain."""
        ctx = _make_context(
            components=(make_component("R1", "10k", "R_0805"),),
            positions={"R1": (10.0, 20.0, 0.0)},
        )
        placed: set[str] = set()
        _chain_series_components(ctx, ["R1"], placed)
        assert ctx.positions["R1"] == (10.0, 20.0, 0.0)
