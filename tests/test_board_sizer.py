"""Tests for board size optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.board_sizer import (
    _DEFAULT_ASPECT_RATIO,
    _compute_component_area,
    _dimensions_from_area,
    optimize_board_size,
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

def _make_requirements(
    components: tuple[Component, ...] | None = None,
    mechanical: MechanicalConstraints | None = None,
) -> ProjectRequirements:
    if components is None:
        components = (
            Component(ref="R1", value="10k", footprint="R_0805"),
            Component(ref="R2", value="4.7k", footprint="R_0805"),
            Component(ref="C1", value="100nF", footprint="C_0603"),
            Component(ref="U1", value="ATmega328", footprint="TQFP-32"),
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
# _compute_component_area
# ---------------------------------------------------------------------------

def test_compute_component_area() -> None:
    reqs = _make_requirements()
    area = _compute_component_area(reqs)
    # R1 (0805) = 5.0, R2 (0805) = 5.0, C1 (0603) = 3.0, U1 (TQFP) = 144.0
    assert area == pytest.approx(5.0 + 5.0 + 3.0 + 144.0)


def test_compute_component_area_unknown_footprint() -> None:
    reqs = _make_requirements(
        components=(
            Component(ref="X1", value="Custom", footprint="UnknownPackage"),
        ),
    )
    area = _compute_component_area(reqs)
    assert area == pytest.approx(25.0)  # default area


def test_compute_component_area_empty() -> None:
    reqs = _make_requirements(components=())
    area = _compute_component_area(reqs)
    assert area == 0.0


# ---------------------------------------------------------------------------
# optimize_board_size
# ---------------------------------------------------------------------------

@patch("kicad_pipeline.optimization.scoring.compute_quality_score")
@patch("kicad_pipeline.pcb.builder.build_pcb")
def test_optimize_board_size_returns_valid_dimensions(
    mock_build: MagicMock,
    mock_score: MagicMock,
) -> None:
    mock_build.return_value = MagicMock()
    mock_score.return_value = _FakeQualityScore(overall_score=0.8)
    reqs = _make_requirements()
    w, h, score = optimize_board_size(reqs)
    assert w > 0
    assert h > 0
    assert score.overall_score >= 0.7


@patch("kicad_pipeline.optimization.scoring.compute_quality_score")
@patch("kicad_pipeline.pcb.builder.build_pcb")
def test_optimize_board_size_respects_min_size(
    mock_build: MagicMock,
    mock_score: MagicMock,
) -> None:
    mock_build.return_value = MagicMock()
    # All scores pass threshold — optimizer should shrink to minimum
    mock_score.return_value = _FakeQualityScore(overall_score=0.9)
    reqs = _make_requirements()
    comp_area = _compute_component_area(reqs)
    w, h, _ = optimize_board_size(reqs, min_area_multiplier=2.0)
    actual_area = w * h
    # Should not go below min_area_multiplier * component_area
    assert actual_area >= comp_area * 2.0 - 50  # some tolerance for rounding


@patch("kicad_pipeline.optimization.scoring.compute_quality_score")
@patch("kicad_pipeline.pcb.builder.build_pcb")
def test_optimize_board_size_quality_threshold(
    mock_build: MagicMock,
    mock_score: MagicMock,
) -> None:
    mock_build.return_value = MagicMock()
    # Only large boards pass the threshold
    call_count = 0

    def _size_dependent_score(*args: object, **kwargs: object) -> _FakeQualityScore:
        nonlocal call_count
        call_count += 1
        # First call (largest) passes, subsequent fail
        if call_count <= 1:
            return _FakeQualityScore(overall_score=0.8)
        return _FakeQualityScore(overall_score=0.3)

    mock_score.side_effect = _size_dependent_score
    reqs = _make_requirements()
    w, h, score = optimize_board_size(reqs, quality_threshold=0.7)
    # Should return the largest board (first that passed)
    assert score.overall_score >= 0.7


def test_optimize_board_size_default_aspect_ratio() -> None:
    """Verify dimensions_from_area uses 4:3 aspect ratio."""
    w, h = _dimensions_from_area(1200.0)
    ratio = w / h if h > 0 else 0
    assert abs(ratio - _DEFAULT_ASPECT_RATIO) < 0.15  # rounding tolerance


@patch("kicad_pipeline.optimization.scoring.compute_quality_score")
@patch("kicad_pipeline.pcb.builder.build_pcb")
def test_optimize_board_size_fixed_mechanical(
    mock_build: MagicMock,
    mock_score: MagicMock,
) -> None:
    """When mechanical constraints have fixed dimensions, use them directly."""
    mock_build.return_value = MagicMock()
    mock_score.return_value = _FakeQualityScore(overall_score=0.75)
    mech = MechanicalConstraints(board_width_mm=65.0, board_height_mm=56.0)
    reqs = _make_requirements(mechanical=mech)
    w, h, score = optimize_board_size(reqs)
    assert w == 65.0
    assert h == 56.0
