"""Tests for placement rendering module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

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
    MechanicalConstraints,
    Net,
    NetConnection,
    ProjectInfo,
    ProjectRequirements,
)

matplotlib = pytest.importorskip("matplotlib")


def _pad(x: float = 0.0, y: float = 0.0) -> Pad:
    return Pad(
        number="1", pad_type="smd", shape="rect",
        position=Point(x, y), size_x=1.0, size_y=1.0,
        layers=("F.Cu",),
    )


def _fp(ref: str, x: float, y: float) -> Footprint:
    return Footprint(
        lib_id="R_0805:R_0805", ref=ref, value="10k",
        position=Point(x, y),
        pads=(_pad(-0.5, 0.0), _pad(0.5, 0.0)),
    )


def _pcb(footprints: tuple[Footprint, ...] = ()) -> PCBDesign:
    return PCBDesign(
        outline=BoardOutline(polygon=(
            Point(0.0, 0.0), Point(80.0, 0.0),
            Point(80.0, 60.0), Point(0.0, 60.0), Point(0.0, 0.0),
        )),
        design_rules=DesignRules(),
        nets=(NetEntry(number=0, name=""),),
        footprints=footprints,
        tracks=(), vias=(), zones=(), keepouts=(),
    )


def _req(
    components: tuple[Component, ...] = (),
    nets: tuple[Net, ...] = (),
) -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        components=components, nets=nets, features=(),
        mechanical=MechanicalConstraints(board_width_mm=80.0, board_height_mm=60.0),
    )


class TestRenderPlacement:
    def test_renders_to_png(self, tmp_path: Path) -> None:
        from kicad_pipeline.visualization.placement_render import render_placement

        pcb = _pcb(footprints=(
            _fp("R1", 20.0, 20.0),
            _fp("R2", 40.0, 30.0),
        ))
        req = _req(components=(
            Component(ref="R1", value="10k", footprint="R_0805"),
            Component(ref="R2", value="4k7", footprint="R_0805"),
        ))

        out = tmp_path / "test_placement.png"
        result = render_placement(pcb, req, out)
        assert result.exists()
        assert result.stat().st_size > 1000  # non-trivial PNG

    def test_custom_title(self, tmp_path: Path) -> None:
        from kicad_pipeline.visualization.placement_render import render_placement

        pcb = _pcb(footprints=(_fp("R1", 20.0, 20.0),))
        req = _req(components=(
            Component(ref="R1", value="10k", footprint="R_0805"),
        ))

        out = tmp_path / "titled.png"
        render_placement(pcb, req, out, title="Custom Title")
        assert out.exists()

    def test_with_score(self, tmp_path: Path) -> None:
        from kicad_pipeline.optimization.scoring import compute_fast_placement_score
        from kicad_pipeline.visualization.placement_render import render_placement

        pcb = _pcb(footprints=(
            _fp("R1", 20.0, 20.0),
            _fp("R2", 40.0, 30.0),
        ))
        req = _req(components=(
            Component(ref="R1", value="10k", footprint="R_0805"),
            Component(ref="R2", value="4k7", footprint="R_0805"),
        ))

        score = compute_fast_placement_score(pcb, req)
        out = tmp_path / "scored.png"
        render_placement(pcb, req, out, score=score)
        assert out.exists()

    def test_with_ratsnest(self, tmp_path: Path) -> None:
        from kicad_pipeline.visualization.placement_render import render_placement

        pcb = _pcb(footprints=(
            _fp("U1", 20.0, 20.0),
            _fp("R1", 40.0, 20.0),
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

        out = tmp_path / "ratsnest.png"
        render_placement(pcb, req, out, show_ratsnest=True)
        assert out.exists()

    def test_empty_board(self, tmp_path: Path) -> None:
        from kicad_pipeline.visualization.placement_render import render_placement

        pcb = _pcb()
        req = _req()
        out = tmp_path / "empty.png"
        render_placement(pcb, req, out)
        assert out.exists()
