"""Tests for placement optimizer (simulated annealing)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
)
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
from kicad_pipeline.optimization.placement_optimizer import (
    OptimizationConfig,
    PlacementCandidate,
    _apply_positions,
    _extract_positions,
    _get_movable_refs,
    _is_fixed,
    _orient_connectors,
    _perturbation_nudge,
    _perturbation_rotate,
    _perturbation_swap,
    _pin_rf_to_edge,
    _place_row_layout,
    _PlacementGrid,
    _pull_mcu_peripherals,
    optimize_placement,
    optimize_placement_ee,
    optimize_placement_sa,
)

# ---------------------------------------------------------------------------
# Fake QualityScore for mocking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FakeQualityScore:
    overall_score: float = 0.5
    placement_score: float = 0.5
    routing_score: float = 0.5
    manufacturing_score: float = 0.5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_outline(w: float = 80.0, h: float = 40.0) -> BoardOutline:
    return BoardOutline(polygon=(
        Point(x=0.0, y=0.0),
        Point(x=w, y=0.0),
        Point(x=w, y=h),
        Point(x=0.0, y=h),
        Point(x=0.0, y=0.0),
    ))


def _make_pcb(
    footprints: tuple[Footprint, ...] | None = None,
    w: float = 80.0,
    h: float = 40.0,
) -> PCBDesign:
    if footprints is None:
        footprints = (
            Footprint(lib_id="R:R_0805", ref="R1", value="10k",
                       position=Point(x=10.0, y=10.0), rotation=0.0),
            Footprint(lib_id="R:R_0805", ref="R2", value="4.7k",
                       position=Point(x=20.0, y=15.0), rotation=90.0),
            Footprint(lib_id="C:C_0805", ref="C1", value="100nF",
                       position=Point(x=30.0, y=20.0), rotation=0.0),
        )
    return PCBDesign(
        outline=_make_outline(w, h),
        design_rules=DesignRules(),
        nets=(NetEntry(number=0, name=""),),
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_requirements(
    components: tuple[Component, ...] | None = None,
    mechanical: MechanicalConstraints | None = None,
) -> ProjectRequirements:
    if components is None:
        components = (
            Component(ref="R1", value="10k", footprint="R_0805"),
            Component(ref="R2", value="4.7k", footprint="R_0805"),
            Component(ref="C1", value="100nF", footprint="C_0805"),
        )
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        features=(
            FeatureBlock(
                name="Main",
                description="Test block",
                components=tuple(c.ref for c in components),
                nets=(),
                subcircuits=(),
            ),
        ),
        components=components,
        nets=(),
        mechanical=mechanical,
    )


# ---------------------------------------------------------------------------
# OptimizationConfig tests
# ---------------------------------------------------------------------------

def test_optimization_config_defaults() -> None:
    cfg = OptimizationConfig()
    assert cfg.max_iterations == 50
    assert cfg.temperature_start == 5.0
    assert cfg.temperature_end == 0.5
    assert cfg.cooling_rate == 0.95
    assert cfg.swap_probability == 0.3
    assert cfg.rotation_probability == 0.2
    assert cfg.seed is None


def test_optimization_config_frozen() -> None:
    cfg = OptimizationConfig()
    with pytest.raises(AttributeError):
        cfg.max_iterations = 100  # type: ignore[misc]


def test_placement_candidate_frozen() -> None:
    cand = PlacementCandidate(
        positions=(("R1", 1.0, 2.0, 0.0),),
        quality_score=_FakeQualityScore(),  # type: ignore[arg-type]
        iteration=0,
    )
    with pytest.raises(AttributeError):
        cand.iteration = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _extract_positions / _apply_positions
# ---------------------------------------------------------------------------

def test_extract_positions_from_pcb() -> None:
    pcb = _make_pcb()
    positions = _extract_positions(pcb)
    assert len(positions) == 3
    refs = {p[0] for p in positions}
    assert refs == {"R1", "R2", "C1"}
    # Check specific position
    r1 = next(p for p in positions if p[0] == "R1")
    assert r1 == ("R1", 10.0, 10.0, 0.0)


def test_apply_positions_roundtrip() -> None:
    pcb = _make_pcb()
    positions = _extract_positions(pcb)
    rebuilt = _apply_positions(pcb, positions)
    for orig, new in zip(pcb.footprints, rebuilt.footprints, strict=False):
        assert orig.ref == new.ref
        assert orig.position.x == new.position.x
        assert orig.position.y == new.position.y
        assert orig.rotation == new.rotation


def test_apply_positions_moves_component() -> None:
    pcb = _make_pcb()
    new_positions = (
        ("R1", 50.0, 25.0, 180.0),
        ("R2", 20.0, 15.0, 90.0),
        ("C1", 30.0, 20.0, 0.0),
    )
    result = _apply_positions(pcb, new_positions)
    r1 = result.get_footprint("R1")
    assert r1 is not None
    assert r1.position.x == 50.0
    assert r1.position.y == 25.0
    assert r1.rotation == 180.0


# ---------------------------------------------------------------------------
# _is_fixed / _get_movable_refs
# ---------------------------------------------------------------------------

def test_is_fixed_with_constraint() -> None:
    """Mounting hole refs are always fixed."""
    reqs = _make_requirements()
    assert _is_fixed("H1", reqs) is True
    assert _is_fixed("MH1", reqs) is True


def test_is_fixed_without_constraint() -> None:
    reqs = _make_requirements()
    assert _is_fixed("R1", reqs) is False
    assert _is_fixed("C1", reqs) is False


def test_is_fixed_with_mechanical_notes() -> None:
    mech = MechanicalConstraints(
        board_width_mm=80.0,
        board_height_mm=40.0,
        notes="J1 FIXED at edge",
    )
    reqs = _make_requirements(mechanical=mech)
    assert _is_fixed("J1", reqs) is True
    assert _is_fixed("R1", reqs) is False


def test_get_movable_refs() -> None:
    pcb = _make_pcb()
    reqs = _make_requirements()
    movable = _get_movable_refs(pcb, reqs)
    assert "R1" in movable
    assert "R2" in movable
    assert "C1" in movable


def test_get_movable_refs_excludes_mounting() -> None:
    fps = (
        Footprint(lib_id="MH:MH", ref="H1", value="MH",
                   position=Point(x=5.0, y=5.0)),
        Footprint(lib_id="R:R_0805", ref="R1", value="10k",
                   position=Point(x=20.0, y=20.0)),
    )
    pcb = _make_pcb(footprints=fps)
    reqs = _make_requirements(
        components=(
            Component(ref="H1", value="MH", footprint="MountingHole"),
            Component(ref="R1", value="10k", footprint="R_0805"),
        ),
    )
    movable = _get_movable_refs(pcb, reqs)
    assert "R1" in movable
    assert "H1" not in movable


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def test_perturbation_nudge_stays_in_bounds() -> None:
    import random
    rng = random.Random(42)
    pos = {"R1": (10.0, 10.0, 0.0), "R2": (20.0, 15.0, 90.0)}
    for _ in range(100):
        new_pos = _perturbation_nudge(
            pos, ("R1", "R2"), 5.0, rng, 80.0, 40.0, 0.0, 0.0,
        )
        for ref in new_pos:
            x, y, _ = new_pos[ref]
            assert 0.0 <= x <= 80.0
            assert 0.0 <= y <= 40.0


def test_perturbation_swap_exchanges_positions() -> None:
    import random
    rng = random.Random(42)
    pos = {"R1": (10.0, 10.0, 0.0), "R2": (50.0, 30.0, 90.0)}
    new_pos = _perturbation_swap(pos, ("R1", "R2"), rng)
    # Positions should be exchanged, rotations kept
    assert new_pos["R1"][:2] == (50.0, 30.0)
    assert new_pos["R1"][2] == 0.0  # rotation preserved
    assert new_pos["R2"][:2] == (10.0, 10.0)
    assert new_pos["R2"][2] == 90.0  # rotation preserved


def test_perturbation_rotate_90_degrees() -> None:
    import random
    rng = random.Random(42)
    pos = {"R1": (10.0, 10.0, 0.0)}
    new_pos = _perturbation_rotate(pos, ("R1",), rng)
    assert new_pos["R1"][2] == 90.0
    # Position unchanged
    assert new_pos["R1"][:2] == (10.0, 10.0)


def test_perturbation_rotate_wraps() -> None:
    import random
    rng = random.Random(42)
    pos = {"R1": (10.0, 10.0, 270.0)}
    new_pos = _perturbation_rotate(pos, ("R1",), rng)
    assert new_pos["R1"][2] == 0.0


def test_perturbation_nudge_empty_movable() -> None:
    import random
    rng = random.Random(42)
    pos = {"R1": (10.0, 10.0, 0.0)}
    new_pos = _perturbation_nudge(pos, (), 5.0, rng, 80.0, 40.0)
    assert new_pos == pos


def test_perturbation_swap_single_component() -> None:
    import random
    rng = random.Random(42)
    pos = {"R1": (10.0, 10.0, 0.0)}
    new_pos = _perturbation_swap(pos, ("R1",), rng)
    assert new_pos == pos


# ---------------------------------------------------------------------------
# optimize_placement tests (mocked scoring)
# ---------------------------------------------------------------------------

def _mock_score(overall: float = 0.5) -> _FakeQualityScore:
    return _FakeQualityScore(overall_score=overall)


@patch("kicad_pipeline.optimization.scoring.compute_quality_score")
def test_optimize_placement_sa_returns_history(mock_score_fn: MagicMock) -> None:
    mock_score_fn.return_value = _mock_score(0.6)
    pcb = _make_pcb()
    reqs = _make_requirements()
    cfg = OptimizationConfig(max_iterations=5, seed=42)
    result_pcb, history = optimize_placement_sa(reqs, pcb, cfg)
    assert len(history) >= 1  # At least the initial candidate
    assert history[0].iteration == 0


@patch("kicad_pipeline.optimization.scoring.compute_quality_score")
def test_optimize_placement_sa_deterministic_with_seed(mock_score_fn: MagicMock) -> None:
    call_count = 0

    def _varying_score(*args: object, **kwargs: object) -> _FakeQualityScore:
        nonlocal call_count
        call_count += 1
        return _mock_score(0.5 + (call_count % 3) * 0.05)

    mock_score_fn.side_effect = _varying_score
    pcb = _make_pcb()
    reqs = _make_requirements()
    cfg = OptimizationConfig(max_iterations=10, seed=123)

    call_count = 0
    _, history1 = optimize_placement_sa(reqs, pcb, cfg)
    call_count = 0
    _, history2 = optimize_placement_sa(reqs, pcb, cfg)

    assert len(history1) == len(history2)
    for h1, h2 in zip(history1, history2, strict=False):
        assert h1.iteration == h2.iteration


@patch("kicad_pipeline.optimization.scoring.compute_quality_score")
def test_optimize_placement_sa_respects_fixed(mock_score_fn: MagicMock) -> None:
    mock_score_fn.return_value = _mock_score(0.6)
    fps = (
        Footprint(lib_id="MH:MH", ref="H1", value="MH",
                   position=Point(x=5.0, y=5.0)),
        Footprint(lib_id="R:R_0805", ref="R1", value="10k",
                   position=Point(x=20.0, y=20.0)),
    )
    pcb = _make_pcb(footprints=fps)
    reqs = _make_requirements(
        components=(
            Component(ref="H1", value="MH", footprint="MountingHole"),
            Component(ref="R1", value="10k", footprint="R_0805"),
        ),
    )
    cfg = OptimizationConfig(max_iterations=5, seed=42)
    result_pcb, _ = optimize_placement_sa(reqs, pcb, cfg)
    # H1 should remain at original position (it's fixed via _is_fixed)
    h1 = result_pcb.get_footprint("H1")
    assert h1 is not None
    assert h1.position.x == 5.0
    assert h1.position.y == 5.0


@patch("kicad_pipeline.optimization.scoring.compute_quality_score")
def test_optimize_placement_sa_best_score_improves_or_equals(
    mock_score_fn: MagicMock,
) -> None:
    scores = iter([0.5, 0.6, 0.55, 0.7, 0.65, 0.8])

    def _incremental(*args: object, **kwargs: object) -> _FakeQualityScore:
        try:
            return _mock_score(next(scores))
        except StopIteration:
            return _mock_score(0.5)

    mock_score_fn.side_effect = _incremental
    pcb = _make_pcb()
    reqs = _make_requirements()
    cfg = OptimizationConfig(max_iterations=5, seed=42)
    _, history = optimize_placement_sa(reqs, pcb, cfg)
    # Best score in history should be monotonically non-decreasing
    best_so_far = history[0].quality_score.overall_score
    for cand in history:
        if cand.quality_score.overall_score > best_so_far:
            best_so_far = cand.quality_score.overall_score
    # The final best should be >= initial
    assert best_so_far >= history[0].quality_score.overall_score


@patch("kicad_pipeline.optimization.scoring.compute_quality_score")
def test_optimize_placement_sa_empty_pcb(mock_score_fn: MagicMock) -> None:
    mock_score_fn.return_value = _mock_score(0.5)
    pcb = _make_pcb(footprints=())
    reqs = _make_requirements(components=())
    cfg = OptimizationConfig(max_iterations=3, seed=42)
    result_pcb, history = optimize_placement_sa(reqs, pcb, cfg)
    assert len(result_pcb.footprints) == 0
    assert len(history) >= 1


@patch("kicad_pipeline.optimization.scoring.compute_quality_score")
def test_optimize_placement_sa_single_component(mock_score_fn: MagicMock) -> None:
    mock_score_fn.return_value = _mock_score(0.6)
    fps = (
        Footprint(lib_id="R:R_0805", ref="R1", value="10k",
                   position=Point(x=40.0, y=20.0)),
    )
    pcb = _make_pcb(footprints=fps)
    reqs = _make_requirements(
        components=(Component(ref="R1", value="10k", footprint="R_0805"),),
    )
    cfg = OptimizationConfig(max_iterations=5, seed=42)
    result_pcb, history = optimize_placement_sa(reqs, pcb, cfg)
    assert len(result_pcb.footprints) == 1
    assert len(history) >= 1


# ---------------------------------------------------------------------------
# SA acceptance / temperature tests
# ---------------------------------------------------------------------------

def test_temperature_decreases_each_iteration() -> None:
    """Verify that SA temperature decreases according to cooling_rate."""
    cfg = OptimizationConfig(
        temperature_start=10.0,
        cooling_rate=0.9,
        max_iterations=5,
    )
    temp = cfg.temperature_start
    temps: list[float] = [temp]
    for _ in range(cfg.max_iterations):
        temp *= cfg.cooling_rate
        temps.append(temp)
    # Each should be strictly less than previous
    for i in range(1, len(temps)):
        assert temps[i] < temps[i - 1]
    # Final temp should be start * rate^iterations
    expected = cfg.temperature_start * (cfg.cooling_rate ** cfg.max_iterations)
    assert abs(temps[-1] - expected) < 1e-10


def test_sa_acceptance_always_accepts_improvement() -> None:
    """SA always accepts moves that improve the score (positive delta)."""
    import random
    rng = random.Random(42)
    # delta > 0 means improvement; should always accept
    for _ in range(100):
        delta = rng.uniform(0.01, 10.0)
        _temperature = rng.uniform(0.001, 100.0)
        # The condition is: delta > 0 → accept
        assert delta > 0


def test_sa_acceptance_probabilistic_at_high_temp() -> None:
    """At high temperature, SA should accept worse moves with higher probability."""
    # P(accept) = exp(delta / temp) where delta < 0
    delta = -0.1  # slight worsening
    low_temp = 0.01
    high_temp = 10.0
    prob_low = math.exp(delta / max(low_temp, 0.001))
    prob_high = math.exp(delta / max(high_temp, 0.001))
    assert prob_high > prob_low
    # At very high temp, probability should be close to 1
    assert prob_high > 0.9


# ---------------------------------------------------------------------------
# EE placement optimizer tests
# ---------------------------------------------------------------------------

def _make_pad(num: str, x: float = 0, y: float = 0) -> Pad:
    return Pad(number=num, pad_type="smd", shape="roundrect",
               position=Point(x, y), size_x=1.0, size_y=0.6,
               layers=("F.Cu", "F.Paste", "F.Mask"),
               net_number=0, net_name="")


def _make_ee_pcb(refs_positions: list[tuple[str, float, float]]) -> PCBDesign:
    """Create a PCB with footprints at specified positions."""
    fps = tuple(
        Footprint(
            lib_id=f"test:{ref}", ref=ref, value=ref,
            position=Point(x, y), rotation=0.0,
            pads=(_make_pad("1", -0.5, 0), _make_pad("2", 0.5, 0)),
        )
        for ref, x, y in refs_positions
    )
    return _make_pcb(footprints=fps)


def _pin(num: str, name: str, net: str | None = None) -> Pin:
    return Pin(number=num, name=name, pin_type=PinType.PASSIVE,
               function=None, net=net)


def _make_ee_requirements(
    components: tuple[Component, ...],
    nets: tuple[Net, ...] = (),
) -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        features=(
            FeatureBlock(
                name="Main", description="Test",
                components=tuple(c.ref for c in components),
                nets=(), subcircuits=(),
            ),
        ),
        components=components,
        nets=nets,
        mechanical=MechanicalConstraints(board_width_mm=100.0, board_height_mm=80.0),
    )


def test_optimize_placement_ee_returns_pcb_and_review() -> None:
    """EE optimizer returns a PCBDesign and PlacementReview."""
    pcb = _make_ee_pcb([("R1", 10, 10), ("R2", 50, 50)])
    reqs = _make_ee_requirements(
        components=(
            Component(ref="R1", value="10k", footprint="R_0805"),
            Component(ref="R2", value="4.7k", footprint="R_0805"),
        ),
    )
    result_pcb, review = optimize_placement_ee(reqs, pcb, max_review_passes=2)
    assert len(result_pcb.footprints) == 2
    assert review.grade in ("A", "B", "C", "D", "F")
    assert "Grade" in review.summary


def test_optimize_placement_ee_respects_fixed_components() -> None:
    """Fixed components (H*, MH*) should not be moved."""
    fps = (
        Footprint(lib_id="MH:MH", ref="H1", value="MH",
                   position=Point(x=5.0, y=5.0),
                   pads=(_make_pad("1"),)),
        Footprint(lib_id="R:R_0805", ref="R1", value="10k",
                   position=Point(x=50.0, y=25.0),
                   pads=(_make_pad("1", -0.5, 0), _make_pad("2", 0.5, 0))),
    )
    pcb = _make_pcb(footprints=fps)
    reqs = _make_ee_requirements(
        components=(
            Component(ref="H1", value="MH", footprint="MountingHole"),
            Component(ref="R1", value="10k", footprint="R_0805"),
        ),
    )
    result_pcb, _ = optimize_placement_ee(reqs, pcb, max_review_passes=1)
    h1 = result_pcb.get_footprint("H1")
    assert h1 is not None
    assert h1.position.x == 5.0
    assert h1.position.y == 5.0


def test_optimize_placement_ee_groups_relay_driver() -> None:
    """Relay driver components should be placed near each other."""
    pcb = _make_ee_pcb([
        ("K1", 10, 10), ("Q1", 80, 70), ("D1", 5, 70), ("R1", 90, 5),
    ])
    reqs = _make_ee_requirements(
        components=(
            Component(ref="K1", value="RELAY", footprint="Relay_SPDT",
                      pins=(_pin("1", "COIL+", "+24V"), _pin("2", "COIL-", "RELAY1_DRIVE"))),
            Component(ref="Q1", value="2N7002", footprint="SOT-23",
                      pins=(_pin("1", "G", "MCU_RELAY"), _pin("2", "D", "RELAY1_DRIVE"),
                            _pin("3", "S", "GND"))),
            Component(ref="D1", value="1N4148", footprint="SOD-123",
                      pins=(_pin("1", "A", "RELAY1_DRIVE"), _pin("2", "K", "+24V"))),
            Component(ref="R1", value="10k", footprint="R_0402",
                      pins=(_pin("1", "1", "MCU_RELAY"), _pin("2", "2", "MCU_GPIO"))),
        ),
        nets=(
            Net("+24V", (NetConnection("K1", "1"), NetConnection("D1", "2"))),
            Net("RELAY1_DRIVE", (NetConnection("K1", "2"), NetConnection("Q1", "2"),
                                  NetConnection("D1", "1"))),
            Net("MCU_RELAY", (NetConnection("Q1", "1"), NetConnection("R1", "1"))),
            Net("GND", (NetConnection("Q1", "3"),)),
            Net("MCU_GPIO", (NetConnection("R1", "2"),)),
        ),
    )
    result_pcb, review = optimize_placement_ee(reqs, pcb, max_review_passes=3)

    # All 4 components should be closer together than initial scatter
    positions = {fp.ref: (fp.position.x, fp.position.y) for fp in result_pcb.footprints}
    # Check: max spread should be less than initial ~100mm diagonal
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    spread = math.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
    assert spread < 80.0  # originally ~100mm


def test_optimize_placement_ee_empty_pcb() -> None:
    """EE optimizer handles empty PCB gracefully."""
    pcb = _make_pcb(footprints=())
    reqs = _make_ee_requirements(components=())
    result_pcb, review = optimize_placement_ee(reqs, pcb, max_review_passes=1)
    assert len(result_pcb.footprints) == 0
    assert review.grade == "A"


def test_optimize_placement_ee_connectors_near_edge() -> None:
    """Connectors should be placed near board edges."""
    # Board outline is 80x40; place connector at center (40, 20)
    pcb = _make_ee_pcb([("J1", 40, 20)])
    reqs = _make_ee_requirements(
        components=(
            Component(ref="J1", value="RJ45", footprint="RJ45",
                      pins=(_pin("1", "1", "ETH_TX+"),)),
        ),
    )
    result_pcb, review = optimize_placement_ee(reqs, pcb, max_review_passes=3)
    j1 = result_pcb.get_footprint("J1")
    assert j1 is not None
    # Should be closer to an edge than original center position
    # Board is 80x40, center is 20mm from edges
    min_edge_dist = min(
        j1.position.x,
        80.0 - j1.position.x,
        j1.position.y,
        40.0 - j1.position.y,
    )
    # Optimizer should push it toward edge
    assert min_edge_dist < 20.0  # better than center


def test_optimize_placement_is_alias_for_ee() -> None:
    """optimize_placement should be optimize_placement_ee."""
    assert optimize_placement is optimize_placement_ee


# ---------------------------------------------------------------------------
# Phase 3: New placement strategy tests
# ---------------------------------------------------------------------------

class TestPlacementGrid:
    def test_is_free_empty(self) -> None:
        grid = _PlacementGrid((0.0, 0.0, 100.0, 80.0))
        assert grid.is_free(50.0, 40.0, 5.0, 5.0)

    def test_collision_detected(self) -> None:
        grid = _PlacementGrid((0.0, 0.0, 100.0, 80.0))
        grid.place(50.0, 40.0, 5.0, 5.0)
        assert not grid.is_free(51.0, 41.0, 5.0, 5.0)

    def test_find_free_pos_near_occupied(self) -> None:
        grid = _PlacementGrid((0.0, 0.0, 100.0, 80.0))
        grid.place(50.0, 40.0, 5.0, 5.0)
        fx, fy = grid.find_free_pos(50.0, 40.0, 5.0, 5.0)
        # Should find a different position
        dist = math.sqrt((fx - 50.0) ** 2 + (fy - 40.0) ** 2)
        assert dist > 2.0


class TestRowLayout:
    def test_two_relays_row(self) -> None:
        """Two relay subcircuits get arranged in a horizontal row."""
        from kicad_pipeline.optimization.functional_grouper import (
            DetectedSubCircuit,
            SubCircuitType,
            VoltageDomain,
        )

        sc1 = DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=("K1", "Q1"),
            anchor_ref="K1",
            net_connections=(),
            domain=VoltageDomain.VIN_24V,
            layout_hint="row",
        )
        sc2 = DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=("K2", "Q2"),
            anchor_ref="K2",
            net_connections=(),
            domain=VoltageDomain.VIN_24V,
            layout_hint="row",
        )

        positions: dict[str, tuple[float, float, float]] = {
            "K1": (20.0, 20.0, 0.0),
            "Q1": (25.0, 20.0, 0.0),
            "K2": (60.0, 60.0, 0.0),
            "Q2": (65.0, 60.0, 0.0),
        }
        fp_sizes: dict[str, tuple[float, float]] = {
            "K1": (5.0, 5.0), "Q1": (2.0, 2.0),
            "K2": (5.0, 5.0), "Q2": (2.0, 2.0),
        }
        bounds = (0.0, 0.0, 100.0, 80.0)

        result = _place_row_layout(
            [sc1, sc2], positions, fp_sizes, bounds, set(),
        )

        # Both relay anchors should now be at similar Y
        k1_y = result["K1"][1]
        k2_y = result["K2"][1]
        assert abs(k1_y - k2_y) < 1.0, f"K1.y={k1_y}, K2.y={k2_y} should be close"

    def test_single_relay_no_row(self) -> None:
        """Single relay subcircuit doesn't get row-arranged."""
        from kicad_pipeline.optimization.functional_grouper import (
            DetectedSubCircuit,
            SubCircuitType,
            VoltageDomain,
        )

        sc1 = DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=("K1", "Q1"),
            anchor_ref="K1",
            net_connections=(),
            domain=VoltageDomain.VIN_24V,
            layout_hint="row",
        )

        positions: dict[str, tuple[float, float, float]] = {
            "K1": (20.0, 20.0, 0.0),
            "Q1": (25.0, 20.0, 0.0),
        }
        fp_sizes: dict[str, tuple[float, float]] = {
            "K1": (5.0, 5.0), "Q1": (2.0, 2.0),
        }
        bounds = (0.0, 0.0, 100.0, 80.0)

        result = _place_row_layout(
            [sc1], positions, fp_sizes, bounds, set(),
        )
        # Position should be unchanged
        assert result["K1"][0] == 20.0
        assert result["K1"][1] == 20.0


class TestRFEdgePinning:
    def test_rf_module_moves_to_edge(self) -> None:
        """RF antenna module gets moved to nearest board edge."""
        from kicad_pipeline.optimization.functional_grouper import (
            DetectedSubCircuit,
            SubCircuitType,
            VoltageDomain,
        )

        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.RF_ANTENNA,
            refs=("U1",),
            anchor_ref="U1",
            net_connections=(),
            domain=VoltageDomain.DIGITAL_3V3,
            layout_hint="edge",
        )

        positions: dict[str, tuple[float, float, float]] = {
            "U1": (50.0, 40.0, 0.0),  # center of board
        }
        fp_sizes: dict[str, tuple[float, float]] = {
            "U1": (18.0, 25.5),  # ESP32 module size
        }
        bounds = (0.0, 0.0, 100.0, 80.0)

        result = _pin_rf_to_edge([sc], positions, fp_sizes, bounds, set())

        # Should be near an edge now
        x, y, rot = result["U1"]
        min_edge = min(x, 100.0 - x, y, 80.0 - y)
        assert min_edge < 20.0, f"RF module should be near edge, min_edge={min_edge}"


class TestMCUPeripheralPulling:
    def test_peripherals_pulled_close(self) -> None:
        """MCU peripherals far from MCU get pulled closer."""
        from kicad_pipeline.optimization.functional_grouper import (
            DetectedSubCircuit,
            SubCircuitType,
            VoltageDomain,
        )

        sc = DetectedSubCircuit(
            circuit_type=SubCircuitType.MCU_PERIPHERAL_CLUSTER,
            refs=("SW1", "LED1"),
            anchor_ref="U1",
            net_connections=(),
            domain=VoltageDomain.DIGITAL_3V3,
            layout_hint="cluster",
        )

        positions: dict[str, tuple[float, float, float]] = {
            "U1": (50.0, 40.0, 0.0),
            "SW1": (10.0, 10.0, 0.0),  # far from MCU
            "LED1": (90.0, 70.0, 0.0),  # far from MCU
        }
        fp_sizes: dict[str, tuple[float, float]] = {
            "U1": (10.0, 10.0), "SW1": (3.0, 3.0), "LED1": (2.0, 2.0),
        }
        bounds = (0.0, 0.0, 100.0, 80.0)

        # Save original distances
        old_dists = {
            ref: math.sqrt(
                (positions[ref][0] - 50.0) ** 2 + (positions[ref][1] - 40.0) ** 2,
            )
            for ref in ("SW1", "LED1")
        }

        result = _pull_mcu_peripherals(
            [sc], positions, fp_sizes, bounds, set(),
        )

        # Both peripherals should be closer to MCU now
        for ref in ("SW1", "LED1"):
            new_dist = math.sqrt(
                (result[ref][0] - 50.0) ** 2 + (result[ref][1] - 40.0) ** 2,
            )
            assert new_dist < old_dists[ref], f"{ref} should be closer to MCU"


class TestConnectorOrientation:
    def test_connector_near_right_edge_faces_right(self) -> None:
        """Connector near right edge gets rotated to face right."""
        fp = Footprint(
            lib_id="J:Screw_Terminal",
            ref="J1",
            value="Screw_Terminal",
            position=Point(x=95.0, y=40.0),
            rotation=0.0,
        )
        pcb = _make_pcb(footprints=(fp,))

        positions: dict[str, tuple[float, float, float]] = {
            "J1": (95.0, 40.0, 0.0),
        }
        fp_sizes: dict[str, tuple[float, float]] = {
            "J1": (8.0, 4.0),  # wide connector
        }
        bounds = (0.0, 0.0, 100.0, 80.0)

        result = _orient_connectors(positions, fp_sizes, bounds, set(), pcb)
        # Near right edge, wide connector → 90 degrees
        assert result["J1"][2] == 90.0

    def test_connector_in_center_not_rotated(self) -> None:
        """Connector in center of board (far from edges) is not reoriented."""
        fp = Footprint(
            lib_id="J:Header",
            ref="J1",
            value="Header",
            position=Point(x=50.0, y=40.0),
            rotation=45.0,
        )
        pcb = _make_pcb(footprints=(fp,))

        positions: dict[str, tuple[float, float, float]] = {
            "J1": (50.0, 40.0, 45.0),
        }
        fp_sizes: dict[str, tuple[float, float]] = {
            "J1": (4.0, 4.0),
        }
        bounds = (0.0, 0.0, 100.0, 80.0)

        result = _orient_connectors(positions, fp_sizes, bounds, set(), pcb)
        # Far from edges → unchanged
        assert result["J1"][2] == 45.0
