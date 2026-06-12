"""Stage 3.5 — minimal-motion legalization of a floorplan.

Cleans sub-millimeter residuals the greedy packer can leave (overlap
within clearance, bbox slightly past the outline) by projected
constraint iteration: each violated pairwise-separation or containment
constraint moves the involved cells by the *minimum* distance that
satisfies it, split evenly between movable cells. Deterministic, runs a
bounded number of sweeps, and — critically — NEVER claims success: the
caller must re-verify with the Stage 4 verifier. If constraints remain
violated after the sweep budget, :class:`LegalizationError` reports
exactly which, so an infeasible board halts the build instead of
shipping degraded.

This replaces v1's unbounded push-apart loops (and their oscillation
cascades) with a single bounded pass over whole rigid cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.optimization.geometry import (
    convex_polygons_overlap,
    polygon_bbox,
)
from kicad_pipeline.placement_v2.floorplan import _EDGE_MARGIN_MM
from kicad_pipeline.placement_v2.ir import Severity, Violation

if TYPE_CHECKING:
    from kicad_pipeline.placement_v2.cells import PlacedCell
    from kicad_pipeline.placement_v2.floorplan import Floorplan

_MAX_SWEEPS = 50


class LegalizationError(PCBError):
    """The floorplan could not be made feasible by minimal motion."""

    def __init__(self, violations: tuple[Violation, ...]) -> None:
        self.violations = violations
        super().__init__(
            "legalization failed: "
            + "; ".join(v.message for v in violations)
        )


def _separation_vector(
    a: PlacedCell, b: PlacedCell, clearance: float,
) -> tuple[float, float] | None:
    """Minimal axis-aligned vector to separate two cell bboxes, or None.

    Uses bounding boxes (conservative for convex hulls): the overlap is
    resolved along the axis of least penetration, the classic minimal-
    motion choice.
    """
    if not convex_polygons_overlap(
        a.polygon_in_board(), b.polygon_in_board(), clearance_mm=clearance
    ):
        return None
    ax1, ay1, ax2, ay2 = polygon_bbox(a.polygon_in_board())
    bx1, by1, bx2, by2 = polygon_bbox(b.polygon_in_board())
    pen_x = min(ax2, bx2) - max(ax1, bx1) + clearance
    pen_y = min(ay2, by2) - max(ay1, by1) + clearance
    if pen_x <= 0 or pen_y <= 0:
        # Hull overlap without bbox overlap is impossible; clearance-only
        # contact resolves along the smaller residual axis.
        pen_x = max(pen_x, 0.01)
        pen_y = max(pen_y, 0.01)
    a_cx, a_cy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    b_cx, b_cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    if pen_x <= pen_y:
        return (pen_x if b_cx >= a_cx else -pen_x, 0.0)
    return (0.0, pen_y if b_cy >= a_cy else -pen_y)


def _containment_shift(
    pc: PlacedCell, bw: float, bh: float,
) -> tuple[float, float]:
    """Minimal translation pulling a cell fully inside the outline.

    Sub-micron residuals are clamped to zero: a connector snapped to
    EXACTLY the edge margin would otherwise register as a violation
    through float round-off.
    """
    x1, y1, x2, y2 = polygon_bbox(pc.polygon_in_board())
    dx = 0.0
    dy = 0.0
    if x1 < _EDGE_MARGIN_MM:
        dx = _EDGE_MARGIN_MM - x1
    elif x2 > bw - _EDGE_MARGIN_MM:
        dx = (bw - _EDGE_MARGIN_MM) - x2
    if y1 < _EDGE_MARGIN_MM:
        dy = _EDGE_MARGIN_MM - y1
    elif y2 > bh - _EDGE_MARGIN_MM:
        dy = (bh - _EDGE_MARGIN_MM) - y2
    if abs(dx) < 1e-6:
        dx = 0.0
    if abs(dy) < 1e-6:
        dy = 0.0
    return (dx, dy)


def legalize(
    plan: Floorplan,
    clearance_mm: float = 0.5,
    pinned: frozenset[str] = frozenset(),
) -> Floorplan:
    """Resolve residual overlaps/containment by minimal whole-cell moves.

    *pinned* names cells (by ``cell.name``) that must not move — e.g.
    edge-snapped connector groups; their share of any separation is
    transferred to the other cell.
    """
    from kicad_pipeline.placement_v2.floorplan import Floorplan as FloorplanResult

    cells = list(plan.placed)
    bw, bh = plan.board_width, plan.board_height

    for _sweep in range(_MAX_SWEEPS):
        dirty = False
        # Containment first: a cell pushed off-board must come back
        # before pairwise separation distributes that error around.
        for i, pc in enumerate(cells):
            if pc.cell.name in pinned:
                continue
            dx, dy = _containment_shift(pc, bw, bh)
            if dx or dy:
                cells[i] = pc.moved_to(pc.dx + dx, pc.dy + dy)
                dirty = True
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                vec = _separation_vector(cells[i], cells[j], clearance_mm)
                if vec is None:
                    continue
                dirty = True
                vx, vy = vec
                i_pin = cells[i].cell.name in pinned
                j_pin = cells[j].cell.name in pinned
                if i_pin and j_pin:
                    continue  # both pinned: report at the end
                if i_pin:
                    cells[j] = cells[j].moved_to(
                        cells[j].dx + vx, cells[j].dy + vy
                    )
                elif j_pin:
                    cells[i] = cells[i].moved_to(
                        cells[i].dx - vx, cells[i].dy - vy
                    )
                else:
                    cells[i] = cells[i].moved_to(
                        cells[i].dx - vx / 2, cells[i].dy - vy / 2
                    )
                    cells[j] = cells[j].moved_to(
                        cells[j].dx + vx / 2, cells[j].dy + vy / 2
                    )
        if not dirty:
            break

    residual = _residual_violations(cells, bw, bh, clearance_mm)
    if residual:
        raise LegalizationError(residual)
    return FloorplanResult(
        placed=tuple(cells), board_width=bw, board_height=bh,
        pinned=plan.pinned,
    )


def _residual_violations(
    cells: list[PlacedCell], bw: float, bh: float, clearance: float,
) -> tuple[Violation, ...]:
    out: list[Violation] = []
    for i, pc in enumerate(cells):
        dx, dy = _containment_shift(pc, bw, bh)
        if dx or dy:
            out.append(Violation(
                constraint="BoardContain",
                refs=(pc.cell.name,),
                severity=Severity.CRITICAL,
                measured=max(abs(dx), abs(dy)),
                limit=0.0,
                message=f"{pc.cell.name} extends past board outline",
            ))
        for other in cells[i + 1:]:
            if convex_polygons_overlap(
                pc.polygon_in_board(), other.polygon_in_board()
            ):
                out.append(Violation(
                    constraint="cell_overlap",
                    refs=(pc.cell.name, other.cell.name),
                    severity=Severity.CRITICAL,
                    measured=1.0,
                    limit=0.0,
                    message=(
                        f"{pc.cell.name} overlaps {other.cell.name}"
                        " after legalization"
                    ),
                ))
    return tuple(out)
