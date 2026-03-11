"""Tests for fast-path placement scoring (compute_fast_placement_score)."""

from __future__ import annotations

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
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.scoring import (
    _score_block_cohesion,
    _score_boundary,
    _score_collisions,
    _score_net_proximity,
    compute_fast_placement_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outline(w: float = 80.0, h: float = 60.0) -> BoardOutline:
    return BoardOutline(
        polygon=(
            Point(0.0, 0.0),
            Point(w, 0.0),
            Point(w, h),
            Point(0.0, h),
            Point(0.0, 0.0),
        ),
    )


def _pad(x: float = 0.0, y: float = 0.0) -> Pad:
    return Pad(
        number="1", pad_type="smd", shape="rect",
        position=Point(x, y), size_x=1.0, size_y=1.0,
        layers=("F.Cu",),
    )


def _fp(
    ref: str,
    x: float,
    y: float,
    prefix: str = "R_0805",
    pads: tuple[Pad, ...] | None = None,
) -> Footprint:
    if pads is None:
        pads = (_pad(-0.5, 0.0), _pad(0.5, 0.0))
    return Footprint(
        lib_id=f"{prefix}:{prefix}",
        ref=ref,
        value="10k",
        position=Point(x, y),
        pads=pads,
    )


def _pcb(
    footprints: tuple[Footprint, ...] = (),
    w: float = 80.0,
    h: float = 60.0,
) -> PCBDesign:
    return PCBDesign(
        outline=_outline(w, h),
        design_rules=DesignRules(),
        nets=(NetEntry(number=0, name=""),),
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _req(
    components: tuple[Component, ...] = (),
    nets: tuple[Net, ...] = (),
    features: tuple[FeatureBlock, ...] = (),
) -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        components=components,
        nets=nets,
        features=features,
        mechanical=MechanicalConstraints(board_width_mm=80.0, board_height_mm=60.0),
    )


# ---------------------------------------------------------------------------
# Collision scoring
# ---------------------------------------------------------------------------


class TestScoreCollisions:
    def test_no_footprints_gives_perfect(self) -> None:
        pcb = _pcb()
        score, issues = _score_collisions(pcb)
        assert score == 1.0
        assert issues == []

    def test_non_overlapping_gives_perfect(self) -> None:
        pcb = _pcb(footprints=(
            _fp("R1", 10.0, 10.0),
            _fp("R2", 30.0, 10.0),
        ))
        score, issues = _score_collisions(pcb)
        assert score == 1.0
        assert issues == []

    def test_overlapping_reduces_score(self) -> None:
        # Two components at same position → collision
        pcb = _pcb(footprints=(
            _fp("R1", 10.0, 10.0),
            _fp("R2", 10.0, 10.0),
        ))
        score, issues = _score_collisions(pcb)
        assert score < 1.0
        assert len(issues) == 1
        assert "R1" in issues[0] and "R2" in issues[0]

    def test_many_collisions_floor_at_zero(self) -> None:
        # 10 components at same spot → many collisions → score floors at 0
        fps = tuple(_fp(f"R{i}", 10.0, 10.0) for i in range(10))
        pcb = _pcb(footprints=fps)
        score, _ = _score_collisions(pcb)
        # 10 components → C(10,2) = 45 collisions * 0.05 = 2.25 → clamped to 0.0
        assert score == 0.0

    def test_adjacent_no_collision(self) -> None:
        # Components placed just far enough apart (pad span + margin = 2.0)
        pcb = _pcb(footprints=(
            _fp("R1", 10.0, 10.0),
            _fp("R2", 12.5, 10.0),  # 2.5mm apart, each ~2mm wide
        ))
        score, issues = _score_collisions(pcb)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Net proximity scoring
# ---------------------------------------------------------------------------


class TestScoreNetProximity:
    def test_no_nets_gives_perfect(self) -> None:
        pcb = _pcb(footprints=(_fp("R1", 10.0, 10.0),))
        req = _req(
            components=(Component(ref="R1", value="10k", footprint="R_0805"),),
            nets=(),
        )
        score, issues = _score_net_proximity(pcb, req)
        assert score == 1.0

    def test_close_connected_components_score_high(self) -> None:
        pcb = _pcb(footprints=(
            _fp("U1", 20.0, 20.0, prefix="SOIC-8"),
            _fp("R1", 22.0, 20.0),
        ))
        req = _req(
            components=(
                Component(ref="U1", value="IC", footprint="SOIC-8"),
                Component(ref="R1", value="10k", footprint="R_0805"),
            ),
            nets=(
                Net(name="SIG1", connections=(
                    NetConnection(ref="U1", pin="1"),
                    NetConnection(ref="R1", pin="1"),
                )),
            ),
        )
        score, _ = _score_net_proximity(pcb, req)
        assert score > 0.9  # 2mm apart, well under 50mm max

    def test_far_connected_components_score_low(self) -> None:
        pcb = _pcb(footprints=(
            _fp("U1", 5.0, 5.0, prefix="SOIC-8"),
            _fp("R1", 75.0, 55.0),
        ))
        req = _req(
            components=(
                Component(ref="U1", value="IC", footprint="SOIC-8"),
                Component(ref="R1", value="10k", footprint="R_0805"),
            ),
            nets=(
                Net(name="SIG1", connections=(
                    NetConnection(ref="U1", pin="1"),
                    NetConnection(ref="R1", pin="1"),
                )),
            ),
        )
        score, _ = _score_net_proximity(pcb, req)
        assert score < 0.5  # ~95mm apart

    def test_power_nets_excluded(self) -> None:
        """GND/VCC nets should not count toward proximity scoring."""
        pcb = _pcb(footprints=(
            _fp("U1", 5.0, 5.0, prefix="SOIC-8"),
            _fp("R1", 75.0, 55.0),
        ))
        req = _req(
            components=(
                Component(ref="U1", value="IC", footprint="SOIC-8"),
                Component(ref="R1", value="10k", footprint="R_0805"),
            ),
            nets=(
                Net(name="GND", connections=(
                    NetConnection(ref="U1", pin="4"),
                    NetConnection(ref="R1", pin="2"),
                )),
            ),
        )
        score, _ = _score_net_proximity(pcb, req)
        # GND is a power net → excluded → no signal pairs → 1.0
        assert score == 1.0


# ---------------------------------------------------------------------------
# Block cohesion scoring
# ---------------------------------------------------------------------------


class TestScoreBlockCohesion:
    def test_no_features_gives_perfect(self) -> None:
        pcb = _pcb(footprints=(_fp("R1", 10.0, 10.0),))
        req = _req(features=())
        score, _ = _score_block_cohesion(pcb, req)
        assert score == 1.0

    def test_tight_cluster_scores_high(self) -> None:
        pcb = _pcb(footprints=(
            _fp("U1", 20.0, 20.0, prefix="SOIC-8"),
            _fp("R1", 22.0, 20.0),
            _fp("C1", 20.0, 22.0),
        ))
        req = _req(
            components=(
                Component(ref="U1", value="IC", footprint="SOIC-8"),
                Component(ref="R1", value="10k", footprint="R_0805"),
                Component(ref="C1", value="100n", footprint="C_0805"),
            ),
            features=(
                FeatureBlock(
                    name="Power",
                    description="Power block",
                    components=("U1", "R1", "C1"),
                    nets=(),
                    subcircuits=(),
                ),
            ),
        )
        score, _ = _score_block_cohesion(pcb, req)
        assert score > 0.9

    def test_scattered_block_scores_low(self) -> None:
        pcb = _pcb(footprints=(
            _fp("U1", 5.0, 5.0, prefix="SOIC-8"),
            _fp("R1", 75.0, 5.0),
            _fp("C1", 5.0, 55.0),
        ))
        req = _req(
            features=(
                FeatureBlock(
                    name="Power",
                    description="Power block",
                    components=("U1", "R1", "C1"),
                    nets=(),
                    subcircuits=(),
                ),
            ),
        )
        score, issues = _score_block_cohesion(pcb, req)
        assert score < 0.8
        assert len(issues) >= 1


# ---------------------------------------------------------------------------
# Boundary scoring
# ---------------------------------------------------------------------------


class TestScoreBoundary:
    def test_all_inside_gives_perfect(self) -> None:
        pcb = _pcb(footprints=(
            _fp("R1", 10.0, 10.0),
            _fp("R2", 70.0, 50.0),
        ))
        score, _ = _score_boundary(pcb)
        assert score == 1.0

    def test_outside_reduces_score(self) -> None:
        pcb = _pcb(footprints=(
            _fp("R1", -10.0, 10.0),  # off left edge
            _fp("R2", 40.0, 30.0),
        ))
        score, issues = _score_boundary(pcb)
        assert score < 1.0
        assert any("R1" in i for i in issues)

    def test_multiple_outside(self) -> None:
        pcb = _pcb(footprints=(
            _fp("R1", -10.0, -10.0),
            _fp("R2", 100.0, 100.0),
        ))
        score, issues = _score_boundary(pcb)
        assert score < 0.8
        assert len(issues) == 2


# ---------------------------------------------------------------------------
# Full fast-path scoring
# ---------------------------------------------------------------------------


class TestComputeFastPlacementScore:
    def test_returns_quality_score(self) -> None:
        pcb = _pcb(footprints=(_fp("R1", 10.0, 10.0),))
        req = _req(
            components=(Component(ref="R1", value="10k", footprint="R_0805"),),
        )
        result = compute_fast_placement_score(pcb, req)
        assert 0.0 <= result.overall_score <= 1.0
        assert result.grade in ("A", "B", "C", "D", "F")
        assert len(result.breakdown) == 7

    def test_good_placement_scores_higher(self) -> None:
        """Components placed together near center should score higher."""
        good_pcb = _pcb(footprints=(
            _fp("U1", 40.0, 30.0, prefix="SOIC-8"),
            _fp("R1", 42.0, 30.0),
            _fp("C1", 40.0, 32.0),
        ))
        bad_pcb = _pcb(footprints=(
            _fp("U1", 5.0, 5.0, prefix="SOIC-8"),
            _fp("R1", 75.0, 55.0),
            _fp("C1", 5.0, 55.0),
        ))
        req = _req(
            components=(
                Component(ref="U1", value="IC", footprint="SOIC-8"),
                Component(ref="R1", value="10k", footprint="R_0805"),
                Component(ref="C1", value="100n", footprint="C_0805"),
            ),
            nets=(
                Net(name="SIG1", connections=(
                    NetConnection(ref="U1", pin="1"),
                    NetConnection(ref="R1", pin="1"),
                )),
                Net(name="VCC_BYPASS", connections=(
                    NetConnection(ref="U1", pin="8"),
                    NetConnection(ref="C1", pin="1"),
                )),
            ),
            features=(
                FeatureBlock(
                    name="MCU",
                    description="MCU block",
                    components=("U1", "R1", "C1"),
                    nets=("SIG1",),
                    subcircuits=(),
                ),
            ),
        )

        good_score = compute_fast_placement_score(good_pcb, req)
        bad_score = compute_fast_placement_score(bad_pcb, req)

        assert good_score.overall_score > bad_score.overall_score

    def test_collisions_reduce_score(self) -> None:
        """Overlapping components should significantly reduce score."""
        no_collision = _pcb(footprints=(
            _fp("R1", 10.0, 10.0),
            _fp("R2", 30.0, 10.0),
        ))
        collision = _pcb(footprints=(
            _fp("R1", 10.0, 10.0),
            _fp("R2", 10.0, 10.0),
        ))
        req = _req(
            components=(
                Component(ref="R1", value="10k", footprint="R_0805"),
                Component(ref="R2", value="4k7", footprint="R_0805"),
            ),
        )

        ok_result = compute_fast_placement_score(no_collision, req)
        bad_result = compute_fast_placement_score(collision, req)

        assert ok_result.overall_score > bad_result.overall_score

    def test_deterministic(self) -> None:
        """Same input should always produce same score."""
        pcb = _pcb(footprints=(
            _fp("U1", 20.0, 20.0, prefix="SOIC-8"),
            _fp("R1", 22.0, 20.0),
        ))
        req = _req(
            components=(
                Component(ref="U1", value="IC", footprint="SOIC-8"),
                Component(ref="R1", value="10k", footprint="R_0805"),
            ),
        )
        s1 = compute_fast_placement_score(pcb, req)
        s2 = compute_fast_placement_score(pcb, req)
        assert s1.overall_score == s2.overall_score

    def test_off_board_penalized(self) -> None:
        inside = _pcb(footprints=(_fp("R1", 40.0, 30.0),))
        outside = _pcb(footprints=(_fp("R1", -20.0, -20.0),))
        req = _req(
            components=(Component(ref="R1", value="10k", footprint="R_0805"),),
        )

        in_score = compute_fast_placement_score(inside, req)
        out_score = compute_fast_placement_score(outside, req)
        assert in_score.overall_score > out_score.overall_score
