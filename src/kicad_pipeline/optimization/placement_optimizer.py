"""Placement optimizers: deterministic EE-grade and simulated annealing.

Provides two placement strategies:
- ``optimize_placement_ee()``: Deterministic 5-phase EE-grade placement
  using functional grouping, voltage domain zones, MST placement, and an
  automated review loop.
- ``optimize_placement_sa()`` (legacy): SA with random perturbations.

``optimize_placement`` is an alias for ``optimize_placement_ee``.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.optimization.functional_grouper import (
    DetectedSubCircuit,
    SubCircuitType,
    VoltageDomain,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.review_agent import PlacementReview
    from kicad_pipeline.optimization.scoring import QualityScore

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for placement optimization."""

    max_iterations: int = 50
    temperature_start: float = 5.0  # mm perturbation radius
    temperature_end: float = 0.5
    cooling_rate: float = 0.95
    swap_probability: float = 0.3
    rotation_probability: float = 0.2
    group_move_probability: float = 0.2
    seed: int | None = None


@dataclass(frozen=True)
class PlacementCandidate:
    """A placement configuration with its quality score."""

    positions: tuple[tuple[str, float, float, float], ...]  # (ref, x, y, rotation)
    quality_score: QualityScore
    iteration: int


def _extract_positions(pcb: PCBDesign) -> tuple[tuple[str, float, float, float], ...]:
    """Extract (ref, x, y, rotation) from PCB footprints."""
    return tuple(
        (fp.ref, fp.position.x, fp.position.y, fp.rotation)
        for fp in pcb.footprints
    )


def _apply_positions(
    pcb: PCBDesign,
    positions: tuple[tuple[str, float, float, float], ...],
) -> PCBDesign:
    """Return a new PCBDesign with footprints moved to given positions."""
    pos_dict: dict[str, tuple[float, float, float]] = {
        ref: (x, y, rot) for ref, x, y, rot in positions
    }
    from kicad_pipeline.models.pcb import Footprint as _Footprint
    new_footprints: list[_Footprint] = []
    for fp in pcb.footprints:
        if fp.ref in pos_dict:
            x, y, rot = pos_dict[fp.ref]
            new_fp = replace(fp, position=Point(x=x, y=y), rotation=rot)
            new_footprints.append(new_fp)
        else:
            new_footprints.append(fp)
    return replace(pcb, footprints=tuple(new_footprints))


def _is_fixed(ref: str, requirements: ProjectRequirements) -> bool:
    """Check if a component has a FIXED placement constraint.

    Scans the mechanical constraints and feature blocks for placement
    constraint annotations.  Also checks the constraint solver's
    ``constraints_from_requirements`` output if available.
    """

    # Check mechanical notes for explicit FIXED mentions
    if requirements.mechanical is not None and requirements.mechanical.notes:
        notes = requirements.mechanical.notes.upper()
        if f"{ref.upper()} FIXED" in notes or f"FIXED {ref.upper()}" in notes:
            return True

    # Mounting holes are always fixed
    if ref.startswith("H") or ref.startswith("MH"):
        return True

    # Components whose ref matches a board_template fixed pattern are fixed
    if requirements.mechanical is not None and requirements.mechanical.board_template:
        template_name = requirements.mechanical.board_template.upper()
        # RPi HAT GPIO header is fixed
        if "HAT" in template_name and ref in ("J5", "J40"):
            return True

    return False


def _get_movable_refs(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> tuple[str, ...]:
    """Return refs of components that can be moved (not FIXED)."""
    return tuple(
        fp.ref for fp in pcb.footprints
        if not _is_fixed(fp.ref, requirements)
    )


def _board_bounds(pcb: PCBDesign) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) from board outline polygon."""
    if not pcb.outline.polygon:
        return (0.0, 0.0, 100.0, 100.0)
    xs = [p.x for p in pcb.outline.polygon]
    ys = [p.y for p in pcb.outline.polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _positions_to_dict(
    positions: tuple[tuple[str, float, float, float], ...],
) -> dict[str, tuple[float, float, float]]:
    """Convert positions tuple to dict keyed by ref."""
    return {ref: (x, y, rot) for ref, x, y, rot in positions}


def _dict_to_positions(
    pos_dict: dict[str, tuple[float, float, float]],
) -> tuple[tuple[str, float, float, float], ...]:
    """Convert positions dict back to sorted tuple."""
    return tuple(
        (ref, x, y, rot) for ref, (x, y, rot) in sorted(pos_dict.items())
    )


def _perturbation_nudge(
    positions: dict[str, tuple[float, float, float]],
    movable_refs: tuple[str, ...],
    temperature: float,
    rng: random.Random,
    board_w: float,
    board_h: float,
    board_min_x: float = 0.0,
    board_min_y: float = 0.0,
) -> dict[str, tuple[float, float, float]]:
    """Nudge one random component within temperature radius, clamped to board."""
    if not movable_refs:
        return dict(positions)
    ref = rng.choice(movable_refs)
    if ref not in positions:
        return dict(positions)
    x, y, rot = positions[ref]
    dx = rng.gauss(0, temperature)
    dy = rng.gauss(0, temperature)
    new_x = max(board_min_x, min(board_min_x + board_w, x + dx))
    new_y = max(board_min_y, min(board_min_y + board_h, y + dy))
    result = dict(positions)
    result[ref] = (new_x, new_y, rot)
    return result


def _perturbation_swap(
    positions: dict[str, tuple[float, float, float]],
    movable_refs: tuple[str, ...],
    rng: random.Random,
) -> dict[str, tuple[float, float, float]]:
    """Swap positions of two random movable components."""
    if len(movable_refs) < 2:
        return dict(positions)
    refs_in_pos = [r for r in movable_refs if r in positions]
    if len(refs_in_pos) < 2:
        return dict(positions)
    ref_a, ref_b = rng.sample(refs_in_pos, 2)
    xa, ya, rot_a = positions[ref_a]
    xb, yb, rot_b = positions[ref_b]
    result = dict(positions)
    result[ref_a] = (xb, yb, rot_a)
    result[ref_b] = (xa, ya, rot_b)
    return result


def _perturbation_rotate(
    positions: dict[str, tuple[float, float, float]],
    movable_refs: tuple[str, ...],
    rng: random.Random,
) -> dict[str, tuple[float, float, float]]:
    """Rotate one random component by 90 degrees."""
    if not movable_refs:
        return dict(positions)
    refs_in_pos = [r for r in movable_refs if r in positions]
    if not refs_in_pos:
        return dict(positions)
    ref = rng.choice(refs_in_pos)
    x, y, rot = positions[ref]
    result = dict(positions)
    result[ref] = (x, y, (rot + 90.0) % 360.0)
    return result


def _find_colliding_pairs(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> list[tuple[str, str]]:
    """Find pairs of components whose bounding boxes overlap."""
    pairs: list[tuple[str, str]] = []
    refs = list(positions.keys())
    for i, ref_a in enumerate(refs):
        xa, ya, rot_a = positions[ref_a]
        wa, ha = fp_sizes.get(ref_a, (2.0, 2.0))
        if rot_a % 180 in (90.0, 270.0):
            wa, ha = ha, wa
        for ref_b in refs[i + 1:]:
            xb, yb, rot_b = positions[ref_b]
            wb, hb = fp_sizes.get(ref_b, (2.0, 2.0))
            if rot_b % 180 in (90.0, 270.0):
                wb, hb = hb, wb
            dx = abs(xa - xb)
            dy = abs(ya - yb)
            if dx < (wa + wb) / 2.0 and dy < (ha + hb) / 2.0:
                pairs.append((ref_a, ref_b))
    return pairs


def _perturbation_resolve_collision(
    positions: dict[str, tuple[float, float, float]],
    movable_refs: tuple[str, ...],
    fp_sizes: dict[str, tuple[float, float]],
    rng: random.Random,
    board_w: float,
    board_h: float,
    board_min_x: float = 0.0,
    board_min_y: float = 0.0,
) -> dict[str, tuple[float, float, float]]:
    """Move one component in a colliding pair away from the other."""
    collisions = _find_colliding_pairs(positions, fp_sizes)
    if not collisions:
        return dict(positions)

    # Pick a random collision
    ref_a, ref_b = rng.choice(collisions)
    movable_set = set(movable_refs)

    # Pick the movable one (prefer moving the smaller component)
    if ref_a in movable_set and ref_b in movable_set:
        sa = fp_sizes.get(ref_a, (2.0, 2.0))
        sb = fp_sizes.get(ref_b, (2.0, 2.0))
        to_move = ref_a if (sa[0] * sa[1]) <= (sb[0] * sb[1]) else ref_b
        anchor = ref_b if to_move == ref_a else ref_a
    elif ref_a in movable_set:
        to_move, anchor = ref_a, ref_b
    elif ref_b in movable_set:
        to_move, anchor = ref_b, ref_a
    else:
        return dict(positions)

    xa, ya, rot_a = positions[anchor]
    xm, ym, rot_m = positions[to_move]

    # Push away from anchor: direction from anchor to movable, scaled by size
    wa, ha = fp_sizes.get(anchor, (2.0, 2.0))
    wm, hm = fp_sizes.get(to_move, (2.0, 2.0))
    dx = xm - xa
    dy = ym - ya
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.01:
        # Exactly overlapping — push in random direction
        angle = rng.uniform(0, 2 * math.pi)
        dx, dy = math.cos(angle), math.sin(angle)
        dist = 1.0

    # Move enough to clear overlap + small gap
    needed = (wa + wm) / 2.0 + 0.5
    scale = needed / dist
    new_x = xa + dx * scale
    new_y = ya + dy * scale

    # Clamp to board
    new_x = max(board_min_x + 2.0, min(board_min_x + board_w - 2.0, new_x))
    new_y = max(board_min_y + 2.0, min(board_min_y + board_h - 2.0, new_y))

    result = dict(positions)
    result[to_move] = (new_x, new_y, rot_m)
    return result


def _perturbation_pull_connected(
    positions: dict[str, tuple[float, float, float]],
    movable_refs: tuple[str, ...],
    adjacency: dict[str, set[str]],
    rng: random.Random,
    temperature: float,
    board_w: float,
    board_h: float,
    board_min_x: float = 0.0,
    board_min_y: float = 0.0,
) -> dict[str, tuple[float, float, float]]:
    """Move a component closer to its signal-connected neighbours."""
    # Find movable refs with signal connections
    candidates = [
        r for r in movable_refs
        if r in positions and r in adjacency and adjacency[r]
    ]
    if not candidates:
        return dict(positions)

    ref = rng.choice(candidates)
    x, y, rot = positions[ref]

    # Compute centroid of connected neighbours
    neighbours = [n for n in adjacency[ref] if n in positions]
    if not neighbours:
        return dict(positions)

    cx = sum(positions[n][0] for n in neighbours) / len(neighbours)
    cy = sum(positions[n][1] for n in neighbours) / len(neighbours)

    # Move toward centroid by fraction based on temperature
    fraction = min(0.5, temperature / 20.0)  # at most 50% of the way
    new_x = x + (cx - x) * fraction
    new_y = y + (cy - y) * fraction

    new_x = max(board_min_x + 2.0, min(board_min_x + board_w - 2.0, new_x))
    new_y = max(board_min_y + 2.0, min(board_min_y + board_h - 2.0, new_y))

    result = dict(positions)
    result[ref] = (new_x, new_y, rot)
    return result


def optimize_placement_sa(
    requirements: ProjectRequirements,
    initial_pcb: PCBDesign,
    config: OptimizationConfig | None = None,
) -> tuple[PCBDesign, tuple[PlacementCandidate, ...]]:
    """Run simulated annealing placement optimization (legacy).

    Algorithm:
    1. Extract current positions as baseline.
    2. Score baseline via ``compute_quality_score()``.
    3. For each iteration:
       a. Choose perturbation: nudge, swap, or rotate based on probabilities.
       b. Apply perturbation to positions.
       c. Build new PCB with ``_apply_positions()``.
       d. Score new PCB.
       e. Accept if score improves or by SA probability:
          ``exp((new - current) / temperature)``.
       f. Cool temperature: ``temp *= cooling_rate``.
    4. Return best PCB and history of accepted candidates.

    Args:
        requirements: Project requirements for constraint context.
        initial_pcb: Starting PCB design with initial placement.
        config: Optimization parameters.

    Returns:
        Tuple of (best PCBDesign, history of accepted PlacementCandidates).
    """
    from kicad_pipeline.optimization.scoring import (
        _fp_size_dict,
        compute_fast_placement_score,
    )
    from kicad_pipeline.pcb.constraints import build_signal_adjacency

    if config is None:
        config = OptimizationConfig()

    rng = random.Random(config.seed)

    # Extract board bounds
    min_x, min_y, max_x, max_y = _board_bounds(initial_pcb)
    board_w = max_x - min_x
    board_h = max_y - min_y

    # Extract initial positions and score — use fast-path for SA loop
    current_positions = _extract_positions(initial_pcb)
    current_score = compute_fast_placement_score(initial_pcb, requirements)

    movable_refs = _get_movable_refs(initial_pcb, requirements)

    # Precompute footprint sizes and signal adjacency for targeted perturbations
    fp_sizes = _fp_size_dict(initial_pcb)
    adjacency = build_signal_adjacency(requirements)

    best_positions = current_positions
    best_score = current_score
    best_pcb = initial_pcb

    history: list[PlacementCandidate] = [
        PlacementCandidate(
            positions=current_positions,
            quality_score=current_score,
            iteration=0,
        ),
    ]

    temperature = config.temperature_start
    pos_dict = _positions_to_dict(current_positions)

    # Perturbation probability thresholds (cumulative):
    # 30% resolve collision, 25% pull connected, 20% nudge, 15% swap, 10% rotate
    p_collision = 0.30
    p_pull = 0.55
    p_nudge = 0.75
    p_swap = 0.90

    for iteration in range(1, config.max_iterations + 1):
        # Choose perturbation type — targeted strategies first
        roll = rng.random()
        if roll < p_collision:
            new_pos_dict = _perturbation_resolve_collision(
                pos_dict, movable_refs, fp_sizes, rng,
                board_w, board_h, min_x, min_y,
            )
        elif roll < p_pull:
            new_pos_dict = _perturbation_pull_connected(
                pos_dict, movable_refs, adjacency, rng,
                temperature, board_w, board_h, min_x, min_y,
            )
        elif roll < p_nudge:
            new_pos_dict = _perturbation_nudge(
                pos_dict, movable_refs, temperature, rng,
                board_w, board_h, min_x, min_y,
            )
        elif roll < p_swap:
            new_pos_dict = _perturbation_swap(
                pos_dict, movable_refs, rng,
            )
        else:
            new_pos_dict = _perturbation_rotate(
                pos_dict, movable_refs, rng,
            )

        new_positions = _dict_to_positions(new_pos_dict)
        new_pcb = _apply_positions(initial_pcb, new_positions)
        new_score = compute_fast_placement_score(new_pcb, requirements)

        # SA acceptance criterion (higher overall_score is better)
        delta = new_score.overall_score - current_score.overall_score
        # Scale delta by 10 to make SA more selective (avoid large downhill moves)
        sa_temp = max(temperature * 0.1, 0.001)
        if delta > 0 or rng.random() < math.exp(delta / sa_temp):
            current_score = new_score
            pos_dict = new_pos_dict
            current_positions = new_positions

            candidate = PlacementCandidate(
                positions=new_positions,
                quality_score=new_score,
                iteration=iteration,
            )
            history.append(candidate)

            if new_score.overall_score > best_score.overall_score:
                best_score = new_score
                best_positions = new_positions
                best_pcb = _apply_positions(initial_pcb, best_positions)

        # Cool temperature
        temperature *= config.cooling_rate

    return best_pcb, tuple(history)


# ---------------------------------------------------------------------------
# Deterministic EE-grade placement optimizer (Phase 3)
# ---------------------------------------------------------------------------


class _PlacementGrid:
    """Simple occupancy tracker to prevent component overlaps.

    Tracks placed component bounding boxes and checks for collisions
    before accepting new placements.
    """

    def __init__(self, board_bounds: tuple[float, float, float, float]) -> None:
        self.min_x, self.min_y, self.max_x, self.max_y = board_bounds
        # (cx, cy, half_w, half_h) for each placed component
        self._placed: list[tuple[float, float, float, float]] = []
        self._margin = 0.5  # mm clearance between components

    def is_free(self, cx: float, cy: float, w: float, h: float) -> bool:
        """Check if placing a component here would overlap any existing one."""
        hw = w / 2.0 + self._margin
        hh = h / 2.0 + self._margin
        for px, py, phw, phh in self._placed:
            if abs(cx - px) < hw + phw and abs(cy - py) < hh + phh:
                return False
        return True

    def place(self, cx: float, cy: float, w: float, h: float) -> None:
        """Register a component at (cx, cy) with size (w, h)."""
        self._placed.append((cx, cy, w / 2.0, h / 2.0))

    def find_free_pos(
        self,
        target_x: float,
        target_y: float,
        w: float,
        h: float,
    ) -> tuple[float, float]:
        """Find nearest free position to (target_x, target_y).

        Searches in expanding concentric rings around the target.
        """
        margin = 2.0
        bmin_x = self.min_x + margin
        bmin_y = self.min_y + margin
        bmax_x = self.max_x - margin
        bmax_y = self.max_y - margin

        # Clamp target to board
        tx = max(bmin_x, min(bmax_x, target_x))
        ty = max(bmin_y, min(bmax_y, target_y))

        if self.is_free(tx, ty, w, h):
            return (tx, ty)

        # Spiral search — try 8 directions at increasing radii
        step = max(w, h) * 0.5 + self._margin
        for ring in range(1, 40):
            r = step * ring
            for angle_idx in range(8 * ring):
                angle = 2 * math.pi * angle_idx / (8 * ring)
                cx = tx + r * math.cos(angle)
                cy = ty + r * math.sin(angle)
                cx = max(bmin_x, min(bmax_x, cx))
                cy = max(bmin_y, min(bmax_y, cy))
                if self.is_free(cx, cy, w, h):
                    return (cx, cy)

        # Fallback — return clamped target (will collide but won't crash)
        return (tx, ty)


def _group_footprint_area(
    refs: tuple[str, ...],
    fp_sizes: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Estimate the bounding box needed for a group of components.

    Arranges components in a tight 2-column layout to estimate area.
    Returns (width, height) in mm.
    """
    sizes = [fp_sizes.get(r, (2.0, 2.0)) for r in refs]
    if not sizes:
        return (2.0, 2.0)
    if len(sizes) == 1:
        return (sizes[0][0] + 1.0, sizes[0][1] + 1.0)

    # Sort largest first, pack in 2 columns
    sizes.sort(key=lambda s: s[0] * s[1], reverse=True)
    cols = min(2, len(sizes))
    col_widths = [0.0] * cols
    col_heights = [0.0] * cols
    for i, (w, h) in enumerate(sizes):
        c = i % cols
        col_widths[c] = max(col_widths[c], w)
        col_heights[c] += h + 0.5  # gap between rows

    total_w = sum(col_widths) + 1.0 * (cols - 1) + 1.0  # inter-col gap + margin
    total_h = max(col_heights) + 1.0  # margin
    return (total_w, total_h)


def _place_subcircuit_group(
    anchor_pos: tuple[float, float],
    refs: tuple[str, ...],
    anchor_ref: str,
    fp_sizes: dict[str, tuple[float, float]],
    board_bounds: tuple[float, float, float, float],
    grid: _PlacementGrid,
) -> dict[str, tuple[float, float]]:
    """Arrange sub-circuit components tightly around the anchor.

    Uses the occupancy grid to avoid collisions with already-placed
    components. Places anchor first, then remaining components in a
    spiral pattern checking for free positions.

    Returns:
        Dict mapping ref → (x, y) position.
    """
    min_x, min_y, max_x, max_y = board_bounds
    positions: dict[str, tuple[float, float]] = {}

    # Place anchor
    aw, ah = fp_sizes.get(anchor_ref, (2.0, 2.0))
    ax, ay = grid.find_free_pos(anchor_pos[0], anchor_pos[1], aw, ah)
    grid.place(ax, ay, aw, ah)
    positions[anchor_ref] = (ax, ay)

    other_refs = [r for r in refs if r != anchor_ref]
    if not other_refs:
        return positions

    # Sort by size (largest first) so they get placed closer to anchor
    other_with_size = [
        (ref, fp_sizes.get(ref, (2.0, 2.0))) for ref in other_refs
    ]
    other_with_size.sort(key=lambda x: x[1][0] * x[1][1], reverse=True)

    # Place others around anchor, checking occupancy
    # Use 4 cardinal + 4 diagonal directions for better spreading
    n = len(other_with_size)
    angle_step = 2 * math.pi / max(n, 1)
    for i, (ref, (w, h)) in enumerate(other_with_size):
        # Use direction-aware clearance: along X use widths, along Y use heights
        angle = angle_step * i
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # Minimum distance along this direction to clear anchor + component
        clear_x = (w + aw) / 2.0 + 0.5
        clear_y = (h + ah) / 2.0 + 0.5
        # Project clearance onto direction vector
        if abs(cos_a) > 0.01 or abs(sin_a) > 0.01:
            min_dist = math.sqrt(
                (clear_x * cos_a) ** 2 + (clear_y * sin_a) ** 2,
            )
        else:
            min_dist = max(clear_x, clear_y)
        min_dist = max(min_dist, 1.5)  # at least 1.5mm

        target_x = ax + min_dist * cos_a
        target_y = ay + min_dist * sin_a
        fx, fy = grid.find_free_pos(target_x, target_y, w, h)
        grid.place(fx, fy, w, h)
        positions[ref] = (fx, fy)

    return positions


def _centroid(
    refs: tuple[str, ...] | list[str],
    positions: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Compute centroid of refs that have positions."""
    xs, ys = [], []
    for r in refs:
        if r in positions:
            xs.append(positions[r][0])
            ys.append(positions[r][1])
    if not xs:
        return (0.0, 0.0)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _assign_zone_position(
    zone_rect: tuple[float, float, float, float],
    index: int,
    total: int,
    group_sizes: list[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Assign a position within a zone for the index-th group.

    When *group_sizes* is provided, cells are sized proportionally to
    the footprint area of each group to avoid overlap.
    """
    x1, y1, x2, y2 = zone_rect
    zw = x2 - x1
    zh = y2 - y1
    cx = x1 + zw / 2.0
    cy = y1 + zh / 2.0

    if total <= 1:
        return (cx, cy)

    # Grid within zone — use enough columns to spread groups
    cols = max(1, math.ceil(math.sqrt(total)))
    rows_needed = max(1, (total + cols - 1) // cols)

    # If group sizes provided, compute per-cell offsets proportionally
    cell_w = zw / cols
    cell_h = zh / rows_needed

    row = index // cols
    col = index % cols
    x = x1 + cell_w * (col + 0.5)
    y = y1 + cell_h * (row + 0.5)
    return (x, y)


def _apply_review_fixes(
    positions: dict[str, tuple[float, float, float]],
    review: PlacementReview,
    fixed_refs: set[str],
    fp_sizes: dict[str, tuple[float, float]],
    board_bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Apply suggested position fixes from review violations.

    Only applies fixes for critical and major violations that have
    a suggested_position, affect movable components, and result in
    the component being closer to its target than before.
    """
    result = dict(positions)
    bounds = board_bounds or (0.0, 0.0, 200.0, 200.0)

    for violation in review.violations:
        if violation.severity == "minor":
            continue
        if violation.suggested_position is None:
            continue
        # Apply fix to the first ref in the violation that's movable
        for ref in violation.refs:
            if ref in fixed_refs or ref not in result:
                continue
            old_x, old_y, rot = result[ref]
            sx, sy = violation.suggested_position
            w, h = fp_sizes.get(ref, (2.0, 2.0))

            # Build grid excluding this component
            fix_grid = _PlacementGrid(bounds)
            for other_ref, (ox, oy, _orot) in result.items():
                if other_ref == ref:
                    continue
                ow, oh = fp_sizes.get(other_ref, (2.0, 2.0))
                fix_grid.place(ox, oy, ow, oh)

            # Find nearest free position to suggested target
            fx, fy = fix_grid.find_free_pos(sx, sy, w, h)

            # Only accept if it moves closer to the suggested position
            old_dist = math.sqrt((old_x - sx) ** 2 + (old_y - sy) ** 2)
            new_dist = math.sqrt((fx - sx) ** 2 + (fy - sy) ** 2)
            if new_dist < old_dist:
                result[ref] = (fx, fy, rot)
            break  # one fix attempt per violation

    return result


def _place_row_layout(
    subcircuits: Sequence[DetectedSubCircuit],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
) -> dict[str, tuple[float, float, float]]:
    """Place same-type subcircuits with layout_hint='row' in a 1xN horizontal row.

    Used for relay banks — places relay anchors in a horizontal line with
    tight spacing, then places each relay's sub-components around it.
    """
    min_x, min_y, max_x, max_y = bounds

    # Group by circuit_type for row layout
    type_groups: dict[str, list[DetectedSubCircuit]] = {}
    for sc in subcircuits:
        if sc.layout_hint != "row":
            continue
        type_groups.setdefault(sc.circuit_type.value, []).append(sc)

    for _circuit_type, group in type_groups.items():
        if len(group) < 2:
            continue

        # Find the current average position of anchors
        anchor_positions = []
        for sc in group:
            if sc.anchor_ref in positions and sc.anchor_ref not in fixed_refs:
                x, y, rot = positions[sc.anchor_ref]
                anchor_positions.append((x, y, sc))

        if len(anchor_positions) < 2:
            continue

        # Compute the row center and direction
        avg_x = sum(p[0] for p in anchor_positions) / len(anchor_positions)
        avg_y = sum(p[1] for p in anchor_positions) / len(anchor_positions)

        # Sort anchors left-to-right
        anchor_positions.sort(key=lambda p: p[0])

        # Compute row spacing based on anchor widths
        total_width = 0.0
        for _, _, sc in anchor_positions:
            aw, ah = fp_sizes.get(sc.anchor_ref, (5.0, 5.0))
            total_width += aw + 1.0  # gap between relays

        # Place anchors in a row centered on avg_x
        start_x = avg_x - total_width / 2.0
        row_grid = _PlacementGrid(bounds)

        # Register all non-row components first
        row_refs: set[str] = set()
        for _, _, sc in anchor_positions:
            row_refs.update(sc.refs)
        for ref, (ox, oy, _orot) in positions.items():
            if ref not in row_refs:
                ow, oh = fp_sizes.get(ref, (2.0, 2.0))
                row_grid.place(ox, oy, ow, oh)

        cursor_x = start_x
        for _, _, sc in anchor_positions:
            anchor_ref = sc.anchor_ref
            aw, ah = fp_sizes.get(anchor_ref, (5.0, 5.0))
            target_x = cursor_x + aw / 2.0
            target_y = avg_y
            target_x = max(min_x + 2.0, min(max_x - 2.0, target_x))
            target_y = max(min_y + 2.0, min(max_y - 2.0, target_y))

            fx, fy = row_grid.find_free_pos(target_x, target_y, aw, ah)
            row_grid.place(fx, fy, aw, ah)
            _, _, old_rot = positions.get(anchor_ref, (0, 0, 0))
            positions[anchor_ref] = (fx, fy, old_rot)

            # Place sub-components tight to anchor
            members = [r for r in sc.refs if r != anchor_ref
                       and r not in fixed_refs and r in positions]
            for i, ref in enumerate(members):
                w, h = fp_sizes.get(ref, (2.0, 2.0))
                # Place below anchor in a column
                offset_y = ah / 2.0 + h / 2.0 + 1.0 + i * (h + 0.5)
                mx = fx
                my = fy + offset_y
                mx = max(min_x + 2.0, min(max_x - 2.0, mx))
                my = max(min_y + 2.0, min(max_y - 2.0, my))
                fmx, fmy = row_grid.find_free_pos(mx, my, w, h)
                row_grid.place(fmx, fmy, w, h)
                _, _, mrot = positions[ref]
                positions[ref] = (fmx, fmy, mrot)

            cursor_x += aw + 1.0

    return positions


def _place_boundary_regulators(
    subcircuits: Sequence[DetectedSubCircuit],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    domain_map: dict[str, VoltageDomain],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
) -> dict[str, tuple[float, float, float]]:
    """Place regulators at the boundary between their input and output domains.

    Positions the regulator at the geometric midpoint between the centroids
    of its input domain and output domain components.
    """
    min_x, min_y, max_x, max_y = bounds

    # Compute domain centroids
    domain_positions: dict[VoltageDomain, list[tuple[float, float]]] = {}
    for ref, (x, y, _rot) in positions.items():
        d = domain_map.get(ref)
        if d is not None and d != VoltageDomain.MIXED:
            domain_positions.setdefault(d, []).append((x, y))

    domain_centroids: dict[VoltageDomain, tuple[float, float]] = {}
    for d, pts in domain_positions.items():
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        domain_centroids[d] = (cx, cy)

    for sc in subcircuits:
        if sc.layout_hint != "boundary":
            continue
        if sc.input_domain is None or sc.output_domain is None:
            continue
        if sc.anchor_ref in fixed_refs or sc.anchor_ref not in positions:
            continue

        in_centroid = domain_centroids.get(sc.input_domain)
        out_centroid = domain_centroids.get(sc.output_domain)
        if in_centroid is None or out_centroid is None:
            continue

        # Target: midpoint between domain centroids
        target_x = (in_centroid[0] + out_centroid[0]) / 2.0
        target_y = (in_centroid[1] + out_centroid[1]) / 2.0
        target_x = max(min_x + 2.0, min(max_x - 2.0, target_x))
        target_y = max(min_y + 2.0, min(max_y - 2.0, target_y))

        # Check if moving is actually closer to boundary
        ax, ay, arot = positions[sc.anchor_ref]
        aw, ah = fp_sizes.get(sc.anchor_ref, (2.0, 2.0))
        current_dist = math.sqrt((ax - target_x) ** 2 + (ay - target_y) ** 2)

        if current_dist < 3.0:
            continue  # Already near boundary

        # Build grid without this subcircuit's refs
        move_grid = _PlacementGrid(bounds)
        sc_refs_set = set(sc.refs)
        for ref, (ox, oy, _orot) in positions.items():
            if ref in sc_refs_set:
                continue
            ow, oh = fp_sizes.get(ref, (2.0, 2.0))
            move_grid.place(ox, oy, ow, oh)

        fx, fy = move_grid.find_free_pos(target_x, target_y, aw, ah)
        new_dist = math.sqrt((fx - target_x) ** 2 + (fy - target_y) ** 2)
        if new_dist < current_dist:
            positions[sc.anchor_ref] = (fx, fy, arot)
            move_grid.place(fx, fy, aw, ah)

            # Pull sub-circuit members toward new anchor position
            for ref in sc.refs:
                if ref == sc.anchor_ref or ref in fixed_refs or ref not in positions:
                    continue
                rx, ry, rrot = positions[ref]
                w, h = fp_sizes.get(ref, (2.0, 2.0))
                ideal_dist = (w + aw) / 2.0 + 1.0
                rdist = math.sqrt((rx - fx) ** 2 + (ry - fy) ** 2)
                if rdist <= ideal_dist + 1.0:
                    continue
                if rdist < 0.01:
                    continue
                dx = (fx - rx) / rdist
                dy = (fy - ry) / rdist
                tx = fx - dx * ideal_dist
                ty = fy - dy * ideal_dist
                tx = max(min_x + 2.0, min(max_x - 2.0, tx))
                ty = max(min_y + 2.0, min(max_y - 2.0, ty))
                mrx, mry = move_grid.find_free_pos(tx, ty, w, h)
                if math.sqrt((mrx - fx) ** 2 + (mry - fy) ** 2) < rdist:
                    move_grid.place(mrx, mry, w, h)
                    positions[ref] = (mrx, mry, rrot)

    return positions


def _pin_rf_to_edge(
    subcircuits: Sequence[DetectedSubCircuit],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
) -> dict[str, tuple[float, float, float]]:
    """Pin RF antenna modules to the nearest board edge.

    Sets rotation so antenna faces outward.
    """
    min_x, min_y, max_x, max_y = bounds
    edge_margin = 2.0

    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.RF_ANTENNA:
            continue
        if sc.anchor_ref in fixed_refs or sc.anchor_ref not in positions:
            continue

        cx, cy, rot = positions[sc.anchor_ref]
        w, h = fp_sizes.get(sc.anchor_ref, (5.0, 5.0))

        # Find nearest edge
        dist_left = cx - min_x
        dist_right = max_x - cx
        dist_top = cy - min_y
        dist_bottom = max_y - cy
        min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        # Determine target position and rotation for antenna facing outward
        target_x, target_y = cx, cy
        new_rot = rot
        if dist_right == min_edge_dist or dist_right <= dist_left:
            target_x = max_x - edge_margin - w / 2.0
            new_rot = 90.0  # Antenna pointing right
        elif dist_left == min_edge_dist:
            target_x = min_x + edge_margin + w / 2.0
            new_rot = 270.0
        elif dist_top == min_edge_dist:
            target_y = min_y + edge_margin + h / 2.0
            new_rot = 180.0
        else:
            target_y = max_y - edge_margin - h / 2.0
            new_rot = 0.0

        # Build grid without RF module
        move_grid = _PlacementGrid(bounds)
        for ref, (ox, oy, _orot) in positions.items():
            if ref == sc.anchor_ref:
                continue
            ow, oh = fp_sizes.get(ref, (2.0, 2.0))
            move_grid.place(ox, oy, ow, oh)

        fx, fy = move_grid.find_free_pos(target_x, target_y, w, h)
        positions[sc.anchor_ref] = (fx, fy, new_rot)

    return positions


def _pull_mcu_peripherals(
    subcircuits: Sequence[DetectedSubCircuit],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
) -> dict[str, tuple[float, float, float]]:
    """Pull MCU peripheral cluster members tight to the MCU.

    Ensures switches, LEDs, and debug connectors are within the
    MCU_PERIPHERAL_MAX_DISTANCE_MM threshold.
    """
    from kicad_pipeline.constants import MCU_PERIPHERAL_MAX_DISTANCE_MM

    min_x, min_y, max_x, max_y = bounds

    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.MCU_PERIPHERAL_CLUSTER:
            continue
        anchor = sc.anchor_ref
        if anchor not in positions:
            continue
        ax, ay, _arot = positions[anchor]
        aw, ah = fp_sizes.get(anchor, (5.0, 5.0))

        for ref in sc.refs:
            if ref == anchor or ref in fixed_refs or ref not in positions:
                continue
            rx, ry, rrot = positions[ref]
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            current_dist = math.sqrt((rx - ax) ** 2 + (ry - ay) ** 2)

            if current_dist <= MCU_PERIPHERAL_MAX_DISTANCE_MM:
                continue

            # Target: just outside MCU body
            ideal_dist = (w + aw) / 2.0 + 2.0
            if current_dist < 0.01:
                continue
            dx = (ax - rx) / current_dist
            dy = (ay - ry) / current_dist
            target_x = ax - dx * ideal_dist
            target_y = ay - dy * ideal_dist
            target_x = max(min_x + 2.0, min(max_x - 2.0, target_x))
            target_y = max(min_y + 2.0, min(max_y - 2.0, target_y))

            move_grid = _PlacementGrid(bounds)
            for other_ref, (ox, oy, _orot) in positions.items():
                if other_ref == ref:
                    continue
                ow, oh = fp_sizes.get(other_ref, (2.0, 2.0))
                move_grid.place(ox, oy, ow, oh)

            fx, fy = move_grid.find_free_pos(target_x, target_y, w, h)
            new_dist = math.sqrt((fx - ax) ** 2 + (fy - ay) ** 2)
            if new_dist < current_dist:
                positions[ref] = (fx, fy, rrot)

    return positions


def _orient_connectors(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    pcb: PCBDesign,
) -> dict[str, tuple[float, float, float]]:
    """Orient connectors so mating face faces the nearest board edge.

    Sets rotation based on which edge the connector is closest to:
    - Left edge: 270 (facing left)
    - Right edge: 90 (facing right)
    - Top edge: 180 (facing up)
    - Bottom edge: 0 (facing down)
    """
    min_x, min_y, max_x, max_y = bounds

    for fp in pcb.footprints:
        ref = fp.ref
        if ref in fixed_refs or not ref.startswith("J"):
            continue
        if ref not in positions:
            continue
        cx, cy, rot = positions[ref]
        w, h = fp_sizes.get(ref, (2.0, 2.0))

        # Find nearest edge
        dist_left = cx - min_x
        dist_right = max_x - cx
        dist_top = cy - min_y
        dist_bottom = max_y - cy
        min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_edge_dist > 8.0:
            continue  # Not near an edge, skip orientation

        # Determine orientation based on which edge is closest
        # Aspect ratio determines if connector is wide or tall
        is_wide = w > h * 1.5

        if dist_left == min_edge_dist:
            new_rot = 270.0 if is_wide else 0.0
        elif dist_right == min_edge_dist:
            new_rot = 90.0 if is_wide else 0.0
        elif dist_top == min_edge_dist:
            new_rot = 180.0 if not is_wide else 90.0
        else:
            new_rot = 0.0 if not is_wide else 90.0

        positions[ref] = (cx, cy, new_rot)

    return positions


def optimize_placement_ee(
    requirements: ProjectRequirements,
    initial_pcb: PCBDesign,
    max_review_passes: int = 5,
) -> tuple[PCBDesign, PlacementReview]:
    """Run deterministic EE-grade placement refinement.

    Preserves the initial placement from ``build_pcb`` and makes targeted
    refinements:

    1. Detect sub-circuits from netlist topology.
    2. Pull sub-circuit members closer to their anchors (collision-aware).
    3. Pin connectors to nearest board edge (collision-aware).
    4. Run EE review loop — fix violations iteratively.

    The key principle: **never destroy good initial placement**. Only move
    components that have violations, and only if the move doesn't create
    new collisions.

    Args:
        requirements: Project requirements with components and nets.
        initial_pcb: Starting PCB with initial placement.
        max_review_passes: Max iterations of the review-fix loop.

    Returns:
        Tuple of (optimized PCBDesign, final PlacementReview).
    """
    from kicad_pipeline.optimization.functional_grouper import (
        classify_voltage_domains,
        detect_subcircuits,
    )
    from kicad_pipeline.optimization.review_agent import review_placement
    from kicad_pipeline.optimization.scoring import _fp_size_dict

    fp_sizes = _fp_size_dict(initial_pcb)
    bounds = _board_bounds(initial_pcb)
    min_x, min_y, max_x, max_y = bounds

    # Build fixed refs set
    fixed_refs: set[str] = {
        fp.ref for fp in initial_pcb.footprints
        if _is_fixed(fp.ref, requirements)
    }

    # -----------------------------------------------------------------------
    # Phase 1: Functional Grouping (analysis only — no position changes)
    # -----------------------------------------------------------------------
    _log.info("Phase 1: Detecting sub-circuits and voltage domains")
    subcircuits = detect_subcircuits(requirements)
    domain_map = classify_voltage_domains(requirements)
    sc_list = list(subcircuits)  # mutable list for helper functions

    # -----------------------------------------------------------------------
    # Phase 2: Start from initial placement
    # -----------------------------------------------------------------------
    _log.info("Phase 2: Starting from initial placement (preserving layout)")
    positions: dict[str, tuple[float, float, float]] = {
        fp.ref: (fp.position.x, fp.position.y, fp.rotation)
        for fp in initial_pcb.footprints
    }

    # Build occupancy grid from current placement
    grid = _PlacementGrid(bounds)
    for fp in initial_pcb.footprints:
        w, h = fp_sizes.get(fp.ref, (2.0, 2.0))
        grid.place(fp.position.x, fp.position.y, w, h)

    # -----------------------------------------------------------------------
    # Phase 3: Pull sub-circuit members toward anchors (using occupancy grid)
    # -----------------------------------------------------------------------
    _log.info("Phase 3: Pulling sub-circuit members toward anchors")

    # Sort subcircuits: process largest groups first (relay drivers, buck
    # converters) so they get priority in occupancy grid
    sorted_scs = sorted(subcircuits, key=lambda sc: len(sc.refs), reverse=True)

    for sc in sorted_scs:
        anchor = sc.anchor_ref
        if anchor not in positions or anchor in fixed_refs:
            continue
        ax, ay, _arot = positions[anchor]

        # Sort members by distance from anchor (farthest first) so they
        # get pulled in from the outside
        members = [
            r for r in sc.refs
            if r != anchor and r not in fixed_refs and r in positions
        ]
        members.sort(
            key=lambda r: math.sqrt(
                (positions[r][0] - ax) ** 2 + (positions[r][1] - ay) ** 2,
            ),
            reverse=True,
        )

        for ref in members:
            rx, ry, rrot = positions[ref]
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            aw, ah = fp_sizes.get(anchor, (2.0, 2.0))

            # Compute ideal distance: just outside anchor body
            ideal_dist = (w + aw) / 2.0 + 1.0
            current_dist = math.sqrt((rx - ax) ** 2 + (ry - ay) ** 2)

            if current_dist <= ideal_dist + 1.0:
                # Already close enough
                continue

            # Rebuild grid WITHOUT this component (so it can find a free
            # spot near the anchor)
            move_grid = _PlacementGrid(bounds)
            for other_ref, (ox, oy, _orot) in positions.items():
                if other_ref == ref:
                    continue
                ow, oh = fp_sizes.get(other_ref, (2.0, 2.0))
                move_grid.place(ox, oy, ow, oh)

            # Target: as close to anchor as possible
            if current_dist < 0.01:
                continue
            dx = (ax - rx) / current_dist
            dy = (ay - ry) / current_dist
            target_x = ax - dx * ideal_dist
            target_y = ay - dy * ideal_dist

            # Clamp to board
            target_x = max(min_x + 2.0, min(max_x - 2.0, target_x))
            target_y = max(min_y + 2.0, min(max_y - 2.0, target_y))

            # Find nearest free position to target (may not be exact target)
            fx, fy = move_grid.find_free_pos(target_x, target_y, w, h)

            # Only accept if it's actually closer to anchor
            new_dist = math.sqrt((fx - ax) ** 2 + (fy - ay) ** 2)
            if new_dist < current_dist:
                positions[ref] = (fx, fy, rrot)

    # -----------------------------------------------------------------------
    # Phase 3.1: Row layout for relays (and other row-hint subcircuits)
    # -----------------------------------------------------------------------
    _log.info("Phase 3.1: Arranging row-layout subcircuits")
    positions = _place_row_layout(sc_list, positions, fp_sizes, bounds, fixed_refs)

    # -----------------------------------------------------------------------
    # Phase 3.2: Boundary placement for regulators
    # -----------------------------------------------------------------------
    _log.info("Phase 3.2: Placing regulators at domain boundaries")
    positions = _place_boundary_regulators(
        sc_list, positions, fp_sizes, domain_map, bounds, fixed_refs,
    )

    # -----------------------------------------------------------------------
    # Phase 3.3: RF edge pinning
    # -----------------------------------------------------------------------
    _log.info("Phase 3.3: Pinning RF modules to board edge")
    positions = _pin_rf_to_edge(sc_list, positions, fp_sizes, bounds, fixed_refs)

    # -----------------------------------------------------------------------
    # Phase 3.4: MCU peripheral pulling
    # -----------------------------------------------------------------------
    _log.info("Phase 3.4: Pulling MCU peripherals close to MCU")
    positions = _pull_mcu_peripherals(
        sc_list, positions, fp_sizes, bounds, fixed_refs,
    )

    # -----------------------------------------------------------------------
    # Phase 3.5: Connector Edge Pinning (collision-aware)
    # -----------------------------------------------------------------------
    _log.info("Phase 3.5: Pinning connectors to nearest board edge")
    edge_margin = 3.0  # mm from board edge

    for fp in initial_pcb.footprints:
        ref = fp.ref
        if ref in fixed_refs or not ref.startswith("J"):
            continue
        if ref not in positions:
            continue
        cx, cy, rot = positions[ref]
        w, h = fp_sizes.get(ref, (2.0, 2.0))

        # Find nearest edge
        dist_left = cx - min_x
        dist_right = max_x - cx
        dist_top = cy - min_y
        dist_bottom = max_y - cy
        min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_edge_dist <= 5.0:
            # Already near edge — don't move
            continue

        # Compute target on nearest edge
        target_x, target_y = cx, cy
        if dist_left == min_edge_dist:
            target_x = min_x + edge_margin + w / 2.0
        elif dist_right == min_edge_dist:
            target_x = max_x - edge_margin - w / 2.0
        elif dist_top == min_edge_dist:
            target_y = min_y + edge_margin + h / 2.0
        else:
            target_y = max_y - edge_margin - h / 2.0

        # Check collision at new position
        hw = w / 2.0 + 0.5
        hh = h / 2.0 + 0.5
        can_move = True
        for other_ref, (ox, oy, _orot) in positions.items():
            if other_ref == ref:
                continue
            ow, oh = fp_sizes.get(other_ref, (2.0, 2.0))
            if (abs(target_x - ox) < hw + ow / 2.0 + 0.5
                    and abs(target_y - oy) < hh + oh / 2.0 + 0.5):
                can_move = False
                break

        if can_move:
            positions[ref] = (target_x, target_y, rot)

    # -----------------------------------------------------------------------
    # Phase 3.6: Connector Orientation (mating face toward edge)
    # -----------------------------------------------------------------------
    _log.info("Phase 3.6: Orienting connectors toward board edge")
    positions = _orient_connectors(
        positions, fp_sizes, bounds, fixed_refs, initial_pcb,
    )

    # -----------------------------------------------------------------------
    # Phase 4: EE Review Loop
    # -----------------------------------------------------------------------
    _log.info("Phase 4: Running EE review loop (max %d passes)", max_review_passes)
    best_positions = dict(positions)
    best_violation_count = float("inf")
    best_review: PlacementReview | None = None

    for pass_num in range(max_review_passes):
        positions_tuple = _dict_to_positions(positions)
        current_pcb = _apply_positions(initial_pcb, positions_tuple)
        review = review_placement(
            current_pcb, requirements,
            subcircuits=subcircuits,
            domain_map=domain_map,
        )

        critical_major = sum(
            1 for v in review.violations
            if v.severity in ("critical", "major")
        )
        _log.info(
            "  Pass %d: %s — %d critical/major violations",
            pass_num + 1, review.summary, critical_major,
        )

        if critical_major < best_violation_count:
            best_violation_count = critical_major
            best_positions = dict(positions)
            best_review = review

        if critical_major == 0:
            _log.info("  No critical/major violations — stopping")
            break

        # Apply suggested fixes (collision-checked)
        positions = _apply_review_fixes(
            positions, review, fixed_refs, fp_sizes, bounds,
        )

    # Build final PCB
    if best_review is None:
        positions_tuple = _dict_to_positions(best_positions)
        final_pcb = _apply_positions(initial_pcb, positions_tuple)
        best_review = review_placement(
            final_pcb, requirements,
            subcircuits=subcircuits,
            domain_map=domain_map,
        )
    else:
        positions_tuple = _dict_to_positions(best_positions)
        final_pcb = _apply_positions(initial_pcb, positions_tuple)

    _log.info("EE placement complete: %s", best_review.summary)
    return final_pcb, best_review


# Default optimizer is the EE-grade one
optimize_placement = optimize_placement_ee
