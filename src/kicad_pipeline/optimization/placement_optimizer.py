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
    BoardZoneAssignment,
    DetectedSubCircuit,
    DomainAffinity,
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


@dataclass(frozen=True)
class GroupBoundingBox:
    """Bounding box for a FeatureBlock group with internal component offsets.

    Captures the internal layout of a group so it can be placed as a rigid unit.
    """

    name: str
    refs: tuple[str, ...]
    internal_offsets: dict[str, tuple[float, float]]  # ref → (dx, dy) from origin
    width: float
    height: float


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
        self._margin = 0.25  # mm clearance between components

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


def _fp_courtyard_sizes(pcb: PCBDesign) -> dict[str, tuple[float, float]]:
    """Build ref → (width, height) from pad extents (matching review agent).

    Includes physical pad sizes, not just pad centers. This produces larger
    (more accurate) bounding boxes that match the review agent's collision
    detection.
    """
    result: dict[str, tuple[float, float]] = {}
    for fp in pcb.footprints:
        if not fp.pads:
            result[fp.ref] = (3.0, 3.0)
            continue
        xs = ([p.position.x - p.size_x / 2 for p in fp.pads]
              + [p.position.x + p.size_x / 2 for p in fp.pads])
        ys = ([p.position.y - p.size_y / 2 for p in fp.pads]
              + [p.position.y + p.size_y / 2 for p in fp.pads])
        w = max(xs) - min(xs) + 1.0
        h = max(ys) - min(ys) + 1.0
        result[fp.ref] = (w, h)
    return result


def _rotation_aware_size(
    ref: str,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Get rotation-aware bounding box for a component."""
    w, h = fp_sizes.get(ref, (2.0, 2.0))
    if ref in positions:
        rot = positions[ref][2]
        if rot % 180 in (90.0, 270.0):
            w, h = h, w
    return w, h


def _count_collisions(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> list[tuple[str, str]]:
    """Detect all AABB collisions (rotation-aware, matching scoring)."""
    collisions: list[tuple[str, str]] = []
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

            if (abs(xa - xb) < (wa + wb) / 2.0
                    and abs(ya - yb) < (ha + hb) / 2.0):
                collisions.append((ref_a, ref_b))
    return collisions


def _resolve_collisions(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    group_bboxes: list[GroupBoundingBox] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Resolve courtyard collisions using grid-based relocation.

    For each colliding component (smaller one in the pair), relocate it
    to the nearest free position using the occupancy grid. This guarantees
    the relocated component won't collide with anything already placed.

    When *group_bboxes* is provided, relocated positions are clamped to stay
    within their group's current bounding box + a small margin so that
    collision resolution doesn't scatter group members.
    """
    result = dict(positions)

    collisions = _count_collisions(result, fp_sizes)
    if not collisions:
        _log.info("  Collision resolution: no collisions found")
        return result

    _log.info("  Collision resolution: %d initial collisions", len(collisions))

    # Build current group rect lookup (re-computed from positions each pass)
    def _group_rect(
        grp: GroupBoundingBox,
        pos: dict[str, tuple[float, float, float]],
    ) -> tuple[float, float, float, float]:
        """Current bounding rect of group members in absolute coords."""
        gmin_x = float("inf")
        gmin_y = float("inf")
        gmax_x = float("-inf")
        gmax_y = float("-inf")
        for r in grp.refs:
            if r not in pos:
                continue
            rx, ry, _rot = pos[r]
            w, h = fp_sizes.get(r, (2.0, 2.0))
            gmin_x = min(gmin_x, rx - w / 2)
            gmin_y = min(gmin_y, ry - h / 2)
            gmax_x = max(gmax_x, rx + w / 2)
            gmax_y = max(gmax_y, ry + h / 2)
        margin = 5.0  # allow 5mm expansion for collision resolution
        return (gmin_x - margin, gmin_y - margin,
                gmax_x + margin, gmax_y + margin)

    # Iteratively relocate colliding components
    for _pass in range(5):
        # Recompute colliding refs each pass (relocations may create new collisions)
        current_collisions = _count_collisions(result, fp_sizes)
        if not current_collisions:
            break
        colliding_refs: set[str] = set()
        for ref_a, ref_b in current_collisions:
            if ref_a not in fixed_refs:
                colliding_refs.add(ref_a)
            if ref_b not in fixed_refs:
                colliding_refs.add(ref_b)

        # Sort: move smaller components first (less disruptive)
        sorted_refs = sorted(
            colliding_refs,
            key=lambda r: (
                fp_sizes.get(r, (2.0, 2.0))[0] * fp_sizes.get(r, (2.0, 2.0))[1]
            ),
        )

        moved = 0
        for ref in sorted_refs:
            if ref in fixed_refs:
                continue
            rx, ry, rot = result[ref]
            w, h = _rotation_aware_size(ref, result, fp_sizes)

            # Check if this ref still collides
            has_collision = False
            for other_ref, (ox, oy, _orot) in result.items():
                if other_ref == ref:
                    continue
                ow, oh = _rotation_aware_size(other_ref, result, fp_sizes)
                if (abs(rx - ox) < (w + ow) / 2.0
                        and abs(ry - oy) < (h + oh) / 2.0):
                    has_collision = True
                    break

            if not has_collision:
                continue

            # Build grid WITHOUT this component
            grid = _PlacementGrid(bounds)
            for other_ref, (ox, oy, _orot) in result.items():
                if other_ref == ref:
                    continue
                ow, oh = _rotation_aware_size(other_ref, result, fp_sizes)
                grid.place(ox, oy, ow, oh)

            # Find nearest free position to current location
            fx, fy = grid.find_free_pos(rx, ry, w, h)

            # Clamp to group bounding box if group constraints active
            if group_bboxes is not None:
                grp = _group_of_ref(ref, group_bboxes)
                if grp is not None:
                    grx1, gry1, grx2, gry2 = _group_rect(grp, result)
                    fx = max(grx1 + w / 2, min(grx2 - w / 2, fx))
                    fy = max(gry1 + h / 2, min(gry2 - h / 2, fy))

            result[ref] = (fx, fy, rot)
            moved += 1

        remaining = len(_count_collisions(result, fp_sizes))
        _log.info(
            "  Collision resolution pass %d: relocated %d, %d remaining",
            _pass + 1, moved, remaining,
        )
        if remaining == 0:
            break

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
    zone_assignments: tuple[BoardZoneAssignment, ...] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Place regulators at the boundary between their input and output domains.

    When *zone_assignments* are available, uses the shared edge between
    input and output zone rects. Falls back to midpoint between domain
    centroids.
    """
    min_x, min_y, max_x, max_y = bounds

    # Build zone rect lookup
    zone_rects: dict[VoltageDomain, tuple[float, float, float, float]] = {}
    if zone_assignments:
        for za in zone_assignments:
            zone_rects[za.domain] = za.zone_rect

    # Compute domain centroids (fallback)
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

        # Try zone rects first: find shared edge between input/output zones
        in_rect = zone_rects.get(sc.input_domain)
        out_rect = zone_rects.get(sc.output_domain)

        if in_rect and out_rect:
            # Find the shared boundary between the two zone rects
            # Check if they share a vertical boundary (left-right layout)
            ix1, iy1, ix2, iy2 = in_rect
            ox1, oy1, ox2, oy2 = out_rect
            # Input right edge meets output left edge
            if abs(ix2 - ox1) < 12.0:
                target_x = (ix2 + ox1) / 2.0 + min_x
                target_y = (max(iy1, oy1) + min(iy2, oy2)) / 2.0 + min_y
            # Input bottom edge meets output top edge
            elif abs(iy2 - oy1) < 12.0:
                target_x = (max(ix1, ox1) + min(ix2, ox2)) / 2.0 + min_x
                target_y = (iy2 + oy1) / 2.0 + min_y
            else:
                # No clear shared edge — use centroid midpoint
                in_c = domain_centroids.get(sc.input_domain)
                out_c = domain_centroids.get(sc.output_domain)
                if in_c is None or out_c is None:
                    continue
                target_x = (in_c[0] + out_c[0]) / 2.0
                target_y = (in_c[1] + out_c[1]) / 2.0
        else:
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


def _classify_connector_function(
    ref: str,
    subcircuits: tuple[DetectedSubCircuit, ...],
    adj: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
) -> str:
    """Classify a connector by its functional role based on signal adjacency.

    Returns one of: "relay_terminal", "mcu_peripheral", "power_input",
    "analog_input", "general".
    """
    from kicad_pipeline.pcb.constraints import _is_power_net

    # Check if connector is in a subcircuit
    for sc in subcircuits:
        if ref in sc.refs:
            if sc.circuit_type == SubCircuitType.RELAY_DRIVER:
                return "relay_terminal"
            if sc.circuit_type == SubCircuitType.MCU_PERIPHERAL_CLUSTER:
                return "mcu_peripheral"
            if sc.circuit_type == SubCircuitType.ADC_CHANNEL:
                return "analog_input"

    # Check signal adjacency
    neighbours = adj.get(ref, set())
    relay_refs = {sc.anchor_ref for sc in subcircuits
                  if sc.circuit_type == SubCircuitType.RELAY_DRIVER}
    mcu_refs = {sc.anchor_ref for sc in subcircuits
                if sc.circuit_type == SubCircuitType.MCU_PERIPHERAL_CLUSTER}

    # Check if connector shares nets with relays
    for nb in neighbours:
        if nb in relay_refs or any(nb in sc.refs for sc in subcircuits
                                    if sc.circuit_type == SubCircuitType.RELAY_DRIVER):
            return "relay_terminal"

    # Check if connected to MCU
    for nb in neighbours:
        if nb in mcu_refs:
            return "mcu_peripheral"

    # Check for analog nets
    conn_nets = ref_to_nets.get(ref, set())
    from kicad_pipeline.optimization.functional_grouper import _ANALOG_KEYWORDS
    if any(any(kw in n.upper() for kw in _ANALOG_KEYWORDS) for n in conn_nets):
        return "analog_input"

    # Check for power-only connector
    if all(_is_power_net(n) or n.upper() in {"GND", "VCC"}
           for n in conn_nets if n.strip()):
        return "power_input"

    return "general"


def _pin_connectors_by_function(
    subcircuits: tuple[DetectedSubCircuit, ...],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    adj: dict[str, set[str]],
    ref_to_nets: dict[str, set[str]],
    pcb: PCBDesign,
    group_map: dict[str, str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Pin connectors to board edges based on their functional classification.

    Instead of pushing every connector to its nearest edge, classifies
    each connector by function and pins it to the edge that makes sense:
    - relay_terminal: same edge as relay bank
    - mcu_peripheral: edge nearest MCU
    - power_input: edge nearest power zone
    - analog_input: same edge as relay terminals (measuring those circuits)
    - general: nearest edge (fallback)

    When *group_map* is provided, connector moves are constrained to stay
    near their group's centroid (edge nearest to group, not global).
    """
    min_x, min_y, max_x, max_y = bounds
    edge_margin = 3.0

    # Compute functional group centroids for edge targeting
    relay_centroid: tuple[float, float] | None = None
    relay_positions = []
    for sc in subcircuits:
        if sc.circuit_type == SubCircuitType.RELAY_DRIVER and sc.anchor_ref in positions:
            rx, ry, _ = positions[sc.anchor_ref]
            relay_positions.append((rx, ry))
    if relay_positions:
        relay_centroid = (
            sum(p[0] for p in relay_positions) / len(relay_positions),
            sum(p[1] for p in relay_positions) / len(relay_positions),
        )

    mcu_centroid: tuple[float, float] | None = None
    for sc in subcircuits:
        if (sc.circuit_type == SubCircuitType.MCU_PERIPHERAL_CLUSTER
                and sc.anchor_ref in positions):
            mx, my, _ = positions[sc.anchor_ref]
            mcu_centroid = (mx, my)
            break

    # Find dominant edge for relay group (edge closest to relay centroid)
    def _nearest_edge(
        cx: float, cy: float,
    ) -> str:
        dists = {
            "left": cx - min_x,
            "right": max_x - cx,
            "top": cy - min_y,
            "bottom": max_y - cy,
        }
        return min(dists, key=lambda k: dists[k])

    relay_edge = _nearest_edge(*relay_centroid) if relay_centroid else "left"

    for fp in pcb.footprints:
        ref = fp.ref
        if ref in fixed_refs or not ref.startswith("J"):
            continue
        if ref not in positions:
            continue

        cx, cy, rot = positions[ref]
        w, h = fp_sizes.get(ref, (2.0, 2.0))

        # Classify connector function
        func = _classify_connector_function(ref, subcircuits, adj, ref_to_nets)

        # Determine target edge based on function
        if func == "relay_terminal":
            target_edge = relay_edge
        elif func == "analog_input":
            target_edge = relay_edge  # analog inputs near relay terminals
        elif func == "mcu_peripheral" and mcu_centroid:
            target_edge = _nearest_edge(*mcu_centroid)
        elif func == "power_input":
            # Power connectors toward the high-voltage zone
            is_left = relay_centroid and relay_centroid[0] < (max_x + min_x) / 2
            target_edge = "left" if is_left else "right"
        else:
            # General: nearest edge to current position
            target_edge = _nearest_edge(cx, cy)

        # When group_map is provided, use group centroid Y for the
        # connector's secondary axis so it stays near its group
        group_cy: float | None = None
        group_cx: float | None = None
        if group_map and ref in group_map:
            gname = group_map[ref]
            gpositions = [
                (positions[r][0], positions[r][1])
                for r, g in group_map.items()
                if g == gname and r in positions
            ]
            if gpositions:
                group_cx = sum(p[0] for p in gpositions) / len(gpositions)
                group_cy = sum(p[1] for p in gpositions) / len(gpositions)

        # Compute target position on target edge
        target_x, target_y = cx, cy
        if target_edge == "left":
            target_x = min_x + edge_margin + w / 2.0
            if group_cy is not None:
                target_y = group_cy
        elif target_edge == "right":
            target_x = max_x - edge_margin - w / 2.0
            if group_cy is not None:
                target_y = group_cy
        elif target_edge == "top":
            target_y = min_y + edge_margin + h / 2.0
            if group_cx is not None:
                target_x = group_cx
        else:
            target_y = max_y - edge_margin - h / 2.0
            if group_cx is not None:
                target_x = group_cx

        # Only move if not already on the target edge
        current_edge_dist = min(
            cx - min_x, max_x - cx, cy - min_y, max_y - cy,
        )
        if current_edge_dist <= 5.0:
            # Already on an edge — check if it's the right edge
            current_edge = _nearest_edge(cx, cy)
            if current_edge == target_edge:
                continue  # Already on correct edge

        # Use grid-based collision-aware placement
        edge_grid = _PlacementGrid(bounds)
        for other_ref, (ox, oy, _orot) in positions.items():
            if other_ref == ref:
                continue
            ow, oh = _rotation_aware_size(other_ref, positions, fp_sizes)
            edge_grid.place(ox, oy, ow, oh)

        rw, rh = _rotation_aware_size(ref, positions, fp_sizes)
        fx, fy = edge_grid.find_free_pos(target_x, target_y, rw, rh)

        # Only accept if closer to target edge than before
        new_edge_dist = min(
            fx - min_x, max_x - fx, fy - min_y, max_y - fy,
        )
        if new_edge_dist < current_edge_dist or _nearest_edge(fx, fy) == target_edge:
            positions[ref] = (fx, fy, rot)

    return positions


def _place_adc_channels(
    subcircuits: tuple[DetectedSubCircuit, ...],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
) -> dict[str, tuple[float, float, float]]:
    """Pull ADC channel subcircuit members near their input terminal connector.

    ADC_CHANNEL subcircuits have the connector as anchor — pull divider
    and protection components close to it.
    """
    for sc in subcircuits:
        if sc.circuit_type != SubCircuitType.ADC_CHANNEL:
            continue
        anchor = sc.anchor_ref
        if anchor not in positions or anchor in fixed_refs:
            continue
        ax, ay, _ = positions[anchor]
        aw, ah = fp_sizes.get(anchor, (2.0, 2.0))

        for ref in sc.refs:
            if ref == anchor or ref in fixed_refs or ref not in positions:
                continue
            rx, ry, rrot = positions[ref]
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            current_dist = math.sqrt((rx - ax) ** 2 + (ry - ay) ** 2)

            from kicad_pipeline.constants import ADC_CHANNEL_MAX_SPREAD_MM
            if current_dist <= ADC_CHANNEL_MAX_SPREAD_MM:
                continue

            # Target: tight to connector
            ideal_dist = (w + aw) / 2.0 + 1.0
            if current_dist < 0.01:
                continue
            dx = (ax - rx) / current_dist
            dy = (ay - ry) / current_dist
            tx = ax - dx * ideal_dist
            ty = ay - dy * ideal_dist
            tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
            ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))

            move_grid = _PlacementGrid(bounds)
            for oref, (ox, oy, _orot) in positions.items():
                if oref == ref:
                    continue
                ow, oh = fp_sizes.get(oref, (2.0, 2.0))
                move_grid.place(ox, oy, ow, oh)

            fx, fy = move_grid.find_free_pos(tx, ty, w, h)
            new_dist = math.sqrt((fx - ax) ** 2 + (fy - ay) ** 2)
            if new_dist < current_dist:
                positions[ref] = (fx, fy, rrot)

    return positions


def _apply_cross_domain_affinity_overrides(
    affinities: tuple[DomainAffinity, ...],
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    fixed_refs: set[str],
    domain_map: dict[str, VoltageDomain],
) -> dict[str, tuple[float, float, float]]:
    """Move cross-domain monitoring components closer to their target domain.

    For "measurement" affinities (e.g. ADC monitoring 24V relay outputs),
    moves source_refs toward the centroid of target_refs so they're placed
    near the domain boundary they're monitoring.
    """
    result = dict(positions)

    for aff in affinities:
        if aff.reason != "measurement":
            continue

        # Compute centroid of target refs (the domain being measured)
        target_positions = [
            (result[r][0], result[r][1])
            for r in aff.target_refs if r in result
        ]
        if not target_positions:
            continue
        target_cx = sum(p[0] for p in target_positions) / len(target_positions)
        target_cy = sum(p[1] for p in target_positions) / len(target_positions)

        # Move source refs toward target domain boundary
        for ref in aff.source_refs:
            if ref in fixed_refs or ref not in result:
                continue
            rx, ry, rot = result[ref]
            w, h = fp_sizes.get(ref, (2.0, 2.0))

            current_dist = math.sqrt((rx - target_cx) ** 2 + (ry - target_cy) ** 2)
            if current_dist < 10.0:
                continue  # Already reasonably close

            # Target: partway toward the target domain (60% of the way)
            tx = rx + (target_cx - rx) * 0.6
            ty = ry + (target_cy - ry) * 0.6

            # Build grid without this component
            move_grid = _PlacementGrid(bounds)
            for oref, (ox, oy, _orot) in result.items():
                if oref == ref:
                    continue
                ow, oh = fp_sizes.get(oref, (2.0, 2.0))
                move_grid.place(ox, oy, ow, oh)

            fx, fy = move_grid.find_free_pos(tx, ty, w, h)
            new_dist = math.sqrt((fx - target_cx) ** 2 + (fy - target_cy) ** 2)
            if new_dist < current_dist:
                result[ref] = (fx, fy, rot)
                _log.info(
                    "  Affinity override: moved %s %.1fmm closer to %s domain",
                    ref, current_dist - new_dist, aff.target_domain.value,
                )

    return result


def _extract_group_bboxes(
    requirements: ProjectRequirements,
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
) -> list[GroupBoundingBox]:
    """Extract bounding boxes for each FeatureBlock from current positions.

    For each FeatureBlock, compute the bounding box from member positions
    and store internal offsets relative to the group origin (min_x, min_y).
    """
    if not requirements.features:
        return []

    groups: list[GroupBoundingBox] = []
    for block in requirements.features:
        refs_in_pos = [r for r in block.components if r in positions]
        if not refs_in_pos:
            continue

        # Compute bounding box including component sizes
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        for ref in refs_in_pos:
            x, y, _rot = positions[ref]
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            min_x = min(min_x, x - w / 2)
            min_y = min(min_y, y - h / 2)
            max_x = max(max_x, x + w / 2)
            max_y = max(max_y, y + h / 2)

        # Internal offsets relative to group origin
        origin_x = min_x
        origin_y = min_y
        offsets: dict[str, tuple[float, float]] = {}
        for ref in refs_in_pos:
            x, y, _rot = positions[ref]
            offsets[ref] = (x - origin_x, y - origin_y)

        groups.append(GroupBoundingBox(
            name=block.name,
            refs=tuple(refs_in_pos),
            internal_offsets=offsets,
            width=max_x - min_x,
            height=max_y - min_y,
        ))

    return groups


def _group_of_ref(
    ref: str,
    group_bboxes: list[GroupBoundingBox],
) -> GroupBoundingBox | None:
    """Return the GroupBoundingBox containing a given ref, or None."""
    for g in group_bboxes:
        if ref in g.refs:
            return g
    return None


def _build_group_map(requirements: ProjectRequirements) -> dict[str, str]:
    """Build ref → group_name mapping from FeatureBlocks."""
    group_map: dict[str, str] = {}
    for block in requirements.features:
        for ref in block.components:
            group_map[ref] = block.name
    return group_map


def _optimize_placement_ee_v4(
    requirements: ProjectRequirements,
    initial_pcb: PCBDesign,
    max_review_passes: int = 5,
) -> tuple[PCBDesign, PlacementReview]:
    """Legacy v4 optimizer (15-phase). Preserved for fallback.

    Use ``optimize_placement_ee()`` (v5, 3-level) instead.
    """
    from kicad_pipeline.optimization.functional_grouper import (
        classify_voltage_domains,
        detect_subcircuits,
    )
    from kicad_pipeline.optimization.review_agent import review_placement

    subcircuits = detect_subcircuits(requirements)
    domain_map = classify_voltage_domains(requirements)

    # Simplified: just run review on current placement and return
    review = review_placement(initial_pcb, requirements, subcircuits=subcircuits,
                              domain_map=domain_map)
    return initial_pcb, review


def optimize_placement_ee(
    requirements: ProjectRequirements,
    initial_pcb: PCBDesign,
    max_review_passes: int = 5,
) -> tuple[PCBDesign, PlacementReview]:
    """3-level hierarchical placement optimizer (v5).

    Replaces the 15-phase v4 optimizer with a clean top-down pipeline:

    **Level 1 — Zone Partitioning**: Partition the board into non-overlapping
    rectangular zones based on FeatureBlock groups.

    **Level 2 — Group Placement**: Place each FeatureBlock as a rigid unit
    within its assigned zone. Internal component offsets from ``_layout_group()``
    are preserved. Groups never mix between zones.

    **Level 3 — Intra-Group Refinement**: Fine-tune within groups:
    relay row formation, decoupling cap tightening, connector edge pinning,
    RF edge placement, connector orientation, collision resolution.

    Single-pass, deterministic. Each level is complete before the next starts.
    No level undoes work from a previous level.

    Args:
        requirements: Project requirements with components and nets.
        initial_pcb: Starting PCB with initial placement.
        max_review_passes: Max iterations of the review-fix loop.

    Returns:
        Tuple of (optimized PCBDesign, final PlacementReview).
    """
    from kicad_pipeline.optimization.functional_grouper import (
        classify_voltage_domains,
        compute_power_flow_topology,
        detect_subcircuits,
    )
    from kicad_pipeline.optimization.review_agent import review_placement

    fp_sizes = _fp_courtyard_sizes(initial_pcb)
    bounds = _board_bounds(initial_pcb)
    min_x, min_y, max_x, max_y = bounds

    fixed_refs: set[str] = {
        fp.ref for fp in initial_pcb.footprints
        if _is_fixed(fp.ref, requirements)
    }

    has_groups = bool(requirements.features)

    # ===================================================================
    # Level 1: Zone Partitioning
    # ===================================================================
    _log.info("=== Level 1: Zone Partitioning ===")
    subcircuits = detect_subcircuits(requirements)
    domain_map = classify_voltage_domains(requirements)
    topology = compute_power_flow_topology(subcircuits)
    sc_list = list(subcircuits)

    from kicad_pipeline.optimization.zone_partitioner import partition_board
    zones = partition_board(bounds, list(requirements.features), topology)
    _log.info("  %d zones created", len(zones))

    # ===================================================================
    # Level 2: Group Placement (groups as rigid units)
    # ===================================================================
    _log.info("=== Level 2: Group Placement ===")

    # Extract current positions
    positions: dict[str, tuple[float, float, float]] = {
        fp.ref: (fp.position.x, fp.position.y, fp.rotation)
        for fp in initial_pcb.footprints
    }

    if has_groups and zones:
        from kicad_pipeline.optimization.group_placer import (
            pin_connectors_to_edge,
            place_groups,
        )

        # Extract internal layouts per group from current positions
        # (these come from _layout_group() via build_pcb)
        internal_layouts: dict[str, dict[str, tuple[float, float, float]]] = {}
        for block in requirements.features:
            layout: dict[str, tuple[float, float, float]] = {}
            refs_in_pos = [r for r in block.components if r in positions]
            if not refs_in_pos:
                continue
            # Use current positions as the internal layout
            for ref in refs_in_pos:
                x, y, rot = positions[ref]
                layout[ref] = (x, y, rot)
            internal_layouts[block.name] = layout

        # Place groups as rigid units within zones
        placed_groups = place_groups(
            zones, list(requirements.features),
            internal_layouts, fp_sizes, bounds,
        )

        # Merge placed group positions back into main positions dict
        # Skip fixed refs — they must not be moved
        for pg in placed_groups:
            for ref, (px, py) in pg.positions.items():
                if ref in fixed_refs:
                    continue
                if ref in positions:
                    _, _, rot = positions[ref]  # preserve rotation
                    positions[ref] = (px, py, rot)

        # Pin connectors to board edges
        _log.info("  Pinning connectors to board edges")
        edge_positions = pin_connectors_to_edge(
            placed_groups, fp_sizes, bounds, fixed_refs,
        )
        for ref, (px, py) in edge_positions.items():
            if ref.startswith("J") and ref not in fixed_refs and ref in positions:
                _, _, rot = positions[ref]
                positions[ref] = (px, py, rot)

        _log.info("  %d groups placed", len(placed_groups))
    else:
        _log.info("  No groups — using initial placement")

    # ===================================================================
    # Level 3: Intra-Group Refinement
    # ===================================================================
    _log.info("=== Level 3: Intra-Group Refinement ===")

    # 3a. Relay row formation — arrange relays in 1xN horizontal row
    _log.info("  3a: Relay row formation")
    positions = _place_row_layout(sc_list, positions, fp_sizes, bounds, fixed_refs)

    # 3b. Relay driver subgroup tightening — Q+D+R within 8mm of K
    _log.info("  3b: Relay driver subgroup tightening")
    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.RELAY_DRIVER:
            continue
        anchor = sc.anchor_ref
        if anchor not in positions:
            continue
        kx, ky, _krot = positions[anchor]
        kw, kh = fp_sizes.get(anchor, (18.0, 16.0))

        support_members = [
            r for r in sc.refs
            if r != anchor and r in positions and r not in fixed_refs
        ]
        support_members.sort(key=lambda r: (
            0 if r.startswith("Q") else 1 if r.startswith("D") else 2, r,
        ))

        target_y_base = ky + kh / 2.0 + 1.5
        col = 0
        row_y = target_y_base
        row_max_h = 0.0
        cols_per_row = 3

        tight_grid = _PlacementGrid(bounds)
        for oref, (ox, oy, _or) in positions.items():
            if oref in support_members:
                continue
            ow, oh = fp_sizes.get(oref, (2.0, 2.0))
            tight_grid.place(ox, oy, ow, oh)

        for ref in support_members:
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            tx = kx - kw / 2.0 + (col + 0.5) * (kw / cols_per_row)
            ty = row_y + h / 2.0
            fx, fy = tight_grid.find_free_pos(tx, ty, w, h)
            _, _, rot = positions[ref]
            positions[ref] = (fx, fy, rot)
            tight_grid.place(fx, fy, w, h)
            row_max_h = max(row_max_h, h)
            col += 1
            if col >= cols_per_row:
                col = 0
                row_y += row_max_h + 1.0
                row_max_h = 0.0

    # 3c. Decoupling cap tightening — within 3-5mm of IC
    _log.info("  3c: Decoupling cap tightening")
    for sc in sc_list:
        if sc.circuit_type != SubCircuitType.DECOUPLING:
            continue
        ic_ref = sc.anchor_ref
        if ic_ref not in positions:
            continue
        ix, iy, _irot = positions[ic_ref]
        iw, ih = fp_sizes.get(ic_ref, (5.0, 5.0))

        for cap_ref in sc.refs:
            if cap_ref == ic_ref or not cap_ref.startswith("C"):
                continue
            if cap_ref not in positions or cap_ref in fixed_refs:
                continue
            cx, cy, crot = positions[cap_ref]
            cw, ch = fp_sizes.get(cap_ref, (1.5, 1.0))

            dist = math.sqrt((cx - ix) ** 2 + (cy - iy) ** 2)
            edge_dist = max(0.0, dist - (iw + cw) / 2.0)
            if edge_dist <= 4.0:
                continue

            dx = ix - cx
            dy = iy - cy
            d = math.sqrt(dx * dx + dy * dy) or 1.0
            target_dist = (iw + cw) / 2.0 + 2.0
            tx = ix - dx / d * target_dist
            ty = iy - dy / d * target_dist

            cap_grid = _PlacementGrid(bounds)
            for oref, (ox, oy, _or) in positions.items():
                if oref == cap_ref:
                    continue
                ow, oh = fp_sizes.get(oref, (2.0, 2.0))
                cap_grid.place(ox, oy, ow, oh)

            fx, fy = cap_grid.find_free_pos(tx, ty, cw, ch)
            new_dist = math.sqrt((fx - ix) ** 2 + (fy - iy) ** 2)
            if new_dist < dist:
                positions[cap_ref] = (fx, fy, crot)

    # 3d. Crystal-MCU proximity — within 10mm
    _log.info("  3d: Crystal-MCU proximity")
    from kicad_pipeline.optimization.functional_grouper import _find_mcu_ref
    mcu_ref = _find_mcu_ref(requirements)
    if mcu_ref and mcu_ref in positions:
        mcu_x, mcu_y, _mcu_rot = positions[mcu_ref]
        mcu_w, mcu_h = fp_sizes.get(mcu_ref, (5.0, 5.0))
        for sc in sc_list:
            if sc.circuit_type != SubCircuitType.CRYSTAL_OSC:
                continue
            for ref in sc.refs:
                if ref in fixed_refs or ref not in positions:
                    continue
                rx, ry, rrot = positions[ref]
                dist = math.sqrt((rx - mcu_x) ** 2 + (ry - mcu_y) ** 2)
                if dist <= 10.0:
                    continue
                w, h = fp_sizes.get(ref, (2.0, 2.0))
                gap = 1.0
                candidates = [
                    (mcu_x + (mcu_w + w) / 2.0 + gap, mcu_y),
                    (mcu_x - (mcu_w + w) / 2.0 - gap, mcu_y),
                    (mcu_x, mcu_y + (mcu_h + h) / 2.0 + gap),
                    (mcu_x, mcu_y - (mcu_h + h) / 2.0 - gap),
                ]
                pull_grid = _PlacementGrid(bounds)
                for oref, (ox, oy, _or) in positions.items():
                    if oref != ref:
                        ow, oh = _rotation_aware_size(oref, positions, fp_sizes)
                        pull_grid.place(ox, oy, ow, oh)
                best_pos: tuple[float, float] | None = None
                best_dist = dist
                for txx, tyy in candidates:
                    fx, fy = pull_grid.find_free_pos(txx, tyy, w, h)
                    new_d = math.sqrt((fx - mcu_x) ** 2 + (fy - mcu_y) ** 2)
                    if new_d < best_dist:
                        best_dist = new_d
                        best_pos = (fx, fy)
                if best_pos is not None:
                    positions[ref] = (best_pos[0], best_pos[1], rrot)

    # 3e. RF edge pinning — pin RF modules to board edge
    _log.info("  3e: RF edge pinning")
    positions = _pin_rf_to_edge(sc_list, positions, fp_sizes, bounds, fixed_refs)

    # 3f. Connector orientation — face outward from board edge
    _log.info("  3f: Connector orientation")
    positions = _orient_connectors(
        positions, fp_sizes, bounds, fixed_refs, initial_pcb,
    )

    # 3g. Collision resolution (group-constrained, then unconstrained final pass)
    _log.info("  3g: Collision resolution")
    group_bboxes = _extract_group_bboxes(requirements, positions, fp_sizes)
    positions = _resolve_collisions(
        positions, fp_sizes, bounds, fixed_refs, group_bboxes=group_bboxes,
    )
    # Unconstrained final pass — no group constraints, correctness trumps cohesion
    remaining_collisions = _count_collisions(positions, fp_sizes)
    if remaining_collisions:
        _log.info("  3g: %d remaining — unconstrained pass", len(remaining_collisions))
        positions = _resolve_collisions(
            positions, fp_sizes, bounds, fixed_refs,
        )

    # ===================================================================
    # Final: Clamp, score, review
    # ===================================================================
    _log.info("=== Final: Clamping and review ===")

    # Clamp off-board components
    for ref, (rx, ry, rot) in positions.items():
        if ref in fixed_refs:
            continue
        w, h = fp_sizes.get(ref, (2.0, 2.0))
        clamped_x = max(min_x + w / 2 + 1, min(max_x - w / 2 - 1, rx))
        clamped_y = max(min_y + h / 2 + 1, min(max_y - h / 2 - 1, ry))
        if clamped_x != rx or clamped_y != ry:
            positions[ref] = (clamped_x, clamped_y, rot)

    # Post-clamp collision resolution
    post_clamp_collisions = len(_count_collisions(positions, fp_sizes))
    if post_clamp_collisions > 0:
        _log.info("  %d post-clamp collisions — resolving", post_clamp_collisions)
        positions = _resolve_collisions(positions, fp_sizes, bounds, fixed_refs)

    # EE Review loop (limited passes)
    _log.info("  Running EE review (max %d passes)", max_review_passes)
    best_positions = dict(positions)
    best_review: PlacementReview | None = None
    best_violation_count = float("inf")

    for pass_num in range(max_review_passes):
        positions_tuple = _dict_to_positions(positions)
        current_pcb = _apply_positions(initial_pcb, positions_tuple)
        review = review_placement(
            current_pcb, requirements,
            subcircuits=subcircuits, domain_map=domain_map,
        )

        critical_major = sum(
            1 for v in review.violations
            if v.severity in ("critical", "major")
        )
        _log.info(
            "  Review pass %d: %s — %d critical/major",
            pass_num + 1, review.summary, critical_major,
        )

        if critical_major < best_violation_count:
            best_violation_count = critical_major
            best_positions = dict(positions)
            best_review = review

        if critical_major == 0:
            break

        # Apply suggested fixes
        relay_fixed_review = fixed_refs | {
            r for r in positions if r.startswith("K")
        }
        positions = _apply_review_fixes(
            positions, review, relay_fixed_review, fp_sizes, bounds,
        )

    # Post-review collision resolution — review fixes can introduce overlaps
    post_review_collisions = _count_collisions(best_positions, fp_sizes)
    if post_review_collisions:
        _log.info(
            "  %d post-review collisions — resolving", len(post_review_collisions),
        )
        best_positions = _resolve_collisions(
            best_positions, fp_sizes, bounds, fixed_refs,
        )

    # Build final PCB
    if best_review is None:
        positions_tuple = _dict_to_positions(best_positions)
        final_pcb = _apply_positions(initial_pcb, positions_tuple)
        best_review = review_placement(
            final_pcb, requirements,
            subcircuits=subcircuits, domain_map=domain_map,
        )
    else:
        positions_tuple = _dict_to_positions(best_positions)
        final_pcb = _apply_positions(initial_pcb, positions_tuple)

    _log.info("EE placement v5 complete: %s", best_review.summary)
    return final_pcb, best_review


# Default optimizer is the EE-grade one
optimize_placement = optimize_placement_ee
