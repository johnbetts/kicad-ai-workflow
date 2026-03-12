"""Regression test: board auto-sizer must respect MechanicalConstraints.

When ``requirements.mechanical`` specifies board dimensions, ``build_pcb()``
must NOT expand the board via auto-sizing, even if the caller does not
pass ``board_width_mm`` / ``board_height_mm`` explicitly.
"""

from __future__ import annotations

from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinType,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.pcb.builder import build_pcb


def _passive_pins() -> tuple[Pin, ...]:
    return (
        Pin(number="1", name="~", pin_type=PinType.PASSIVE),
        Pin(number="2", name="~", pin_type=PinType.PASSIVE),
    )


def _make_requirements_with_mechanical(
    width: float = 140.0,
    height: float = 80.0,
    n_components: int = 25,
) -> ProjectRequirements:
    """Build requirements with MechanicalConstraints and many components."""
    components: list[Component] = []
    for i in range(1, n_components + 1):
        components.append(Component(
            ref=f"R{i}", value="10k", footprint="R_0805",
            lcsc="C17414", pins=_passive_pins(),
        ))

    nets: list[Net] = []
    for i in range(1, n_components):
        nets.append(Net(
            name=f"NET{i}",
            connections=(
                NetConnection(ref=f"R{i}", pin="1"),
                NetConnection(ref=f"R{i+1}", pin="2"),
            ),
        ))

    return ProjectRequirements(
        project=ProjectInfo(name="sizing-test"),
        features=(
            FeatureBlock(
                name="MCU",
                description="Test passives",
                components=tuple(f"R{i}" for i in range(1, n_components + 1)),
                nets=tuple(f"NET{i}" for i in range(1, n_components)),
                subcircuits=(),
            ),
        ),
        components=tuple(components),
        nets=tuple(nets),
        power_budget=PowerBudget(
            rails=(PowerRail(name="+3V3", voltage=3.3, current_ma=100.0, source_ref="R1"),),
            total_current_ma=100.0,
            notes=(),
        ),
        mechanical=MechanicalConstraints(board_width_mm=width, board_height_mm=height),
    )


class TestBoardSizingRespectsMechanical:
    """Auto-sizer must not expand a board when MechanicalConstraints are set."""

    def test_board_stays_at_mechanical_dimensions(self) -> None:
        """Board outline must match MechanicalConstraints exactly."""
        req = _make_requirements_with_mechanical(140.0, 80.0, n_components=25)
        pcb = build_pcb(req, auto_route=False)

        pts = pcb.outline.polygon
        board_w = max(p.x for p in pts) - min(p.x for p in pts)
        board_h = max(p.y for p in pts) - min(p.y for p in pts)

        assert board_w <= 140.0 + 0.1, f"Board width {board_w:.1f} > 140mm"
        assert board_h <= 80.0 + 0.1, f"Board height {board_h:.1f} > 80mm"

    def test_small_board_not_expanded(self) -> None:
        """Even a small board with many components must not expand."""
        req = _make_requirements_with_mechanical(50.0, 30.0, n_components=20)
        pcb = build_pcb(req, auto_route=False)

        pts = pcb.outline.polygon
        board_w = max(p.x for p in pts) - min(p.x for p in pts)
        board_h = max(p.y for p in pts) - min(p.y for p in pts)

        assert board_w <= 50.0 + 0.1, f"Board width {board_w:.1f} > 50mm"
        assert board_h <= 30.0 + 0.1, f"Board height {board_h:.1f} > 30mm"
