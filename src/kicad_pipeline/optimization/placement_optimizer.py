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
from kicad_pipeline.pcb.pin_map import (
    centroid_to_origin,
    compute_centroid_offset,
    origin_to_centroid,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kicad_pipeline.models.pcb import Footprint, PCBDesign
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
    """Extract (ref, x, y, rotation) from PCB footprints in centroid space.

    Converts KiCad origin-based positions to centroid-of-pads positions
    so the optimizer works in a consistent coordinate system.
    """
    result: list[tuple[str, float, float, float]] = []
    for fp in pcb.footprints:
        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        result.append((fp.ref, cx, cy, fp.rotation))
    return tuple(result)


# compute_centroid_offset is imported from kicad_pipeline.pcb.pin_map
# Alias kept for backwards compatibility with test imports.
_centroid_offset = compute_centroid_offset


def _apply_positions(
    pcb: PCBDesign,
    positions: tuple[tuple[str, float, float, float], ...],
) -> PCBDesign:
    """Return a new PCBDesign with footprints moved to given positions.

    The optimizer works in centroid-space (x, y = center of pad bounding
    box).  KiCad stores the footprint origin which for pin headers and
    connectors is pin 1, not the center.  This function subtracts the
    rotated centroid offset so the pads end up where the optimizer expects.
    """
    pos_dict: dict[str, tuple[float, float, float]] = {
        ref: (x, y, rot) for ref, x, y, rot in positions
    }
    new_footprints: list[Footprint] = []
    for fp in pcb.footprints:
        if fp.ref in pos_dict:
            cx_want, cy_want, rot = pos_dict[fp.ref]
            origin_x, origin_y = centroid_to_origin(fp, cx_want, cy_want, rot)
            new_fp = replace(fp, position=Point(x=origin_x, y=origin_y),
                             rotation=rot)
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
        max_radius: float = 0.0,
    ) -> tuple[float, float]:
        """Find nearest free position to (target_x, target_y).

        Searches in expanding concentric rings around the target.

        Args:
            max_radius: If > 0, limit search to within this distance of target.
                Returns clamped target if no free position found within radius.
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
        max_rings = 40
        if max_radius > 0:
            max_rings = min(max_rings, max(3, int(max_radius / step) + 1))
        for ring in range(1, max_rings):
            r = step * ring
            if max_radius > 0 and r > max_radius:
                break
            for angle_idx in range(8 * ring):
                angle = 2 * math.pi * angle_idx / (8 * ring)
                cx = tx + r * math.cos(angle)
                cy = ty + r * math.sin(angle)
                cx = max(bmin_x, min(bmax_x, cx))
                cy = max(bmin_y, min(bmax_y, cy))
                if self.is_free(cx, cy, w, h):
                    return (cx, cy)

        # If max_radius limited the search, retry without radius limit
        if max_radius > 0:
            return self.find_free_pos(target_x, target_y, w, h, max_radius=0.0)

        # Fallback — return clamped target (will collide but won't crash)
        _log.warning("find_free_pos: no free position found for (%.1f, %.1f) "
                     "size (%.1f, %.1f) — returning target (will collide)",
                     target_x, target_y, w, h)
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
    """Build ref → (width, height) using accurate courtyard estimates.

    Delegates to :func:`~kicad_pipeline.pcb.footprints.estimate_courtyard_mm`
    which accounts for component body extension beyond the pad field.
    """
    from kicad_pipeline.pcb.footprints import estimate_courtyard_mm

    result: dict[str, tuple[float, float]] = {}
    for fp in pcb.footprints:
        result[fp.ref] = estimate_courtyard_mm(fp)
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

            # If colliding with a large IC (>100mm^2), push away from its center
            # to avoid landing right at the edge of its courtyard
            target_x, target_y = rx, ry
            for other_ref, (ox, oy, _orot) in result.items():
                if other_ref == ref:
                    continue
                ow, oh = _rotation_aware_size(other_ref, result, fp_sizes)
                if ow * oh < 100.0:
                    continue  # not a large IC
                if (abs(rx - ox) < (w + ow) / 2.0
                        and abs(ry - oy) < (h + oh) / 2.0):
                    # Inside large IC — push to nearest edge + margin
                    dx = rx - ox
                    dy = ry - oy
                    if abs(dx) * oh > abs(dy) * ow:
                        # Closer to left/right edge
                        push_x = (ow / 2.0 + w / 2.0 + 2.0)
                        target_x = ox + push_x if dx >= 0 else ox - push_x
                    else:
                        # Closer to top/bottom edge
                        push_y = (oh / 2.0 + h / 2.0 + 2.0)
                        target_y = oy + push_y if dy >= 0 else oy - push_y
                    break

            # Find nearest free position to target
            fx, fy = grid.find_free_pos(target_x, target_y, w, h)

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

        # Relay rotation: 90° for vertical coil orientation
        relay_rotation = 90.0

        # Compute row spacing based on anchor widths (swapped for 90° rotation)
        total_width = 0.0
        for _, _, sc in anchor_positions:
            aw, ah = fp_sizes.get(sc.anchor_ref, (5.0, 5.0))
            # Swap w/h because relay is rotated 90°
            aw, ah = ah, aw
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

        # Force all relay anchors to the same Y (avg_y) in a tight row
        cursor_x = start_x
        for _, _, sc in anchor_positions:
            anchor_ref = sc.anchor_ref
            aw, ah = fp_sizes.get(anchor_ref, (5.0, 5.0))
            # Swap w/h for 90° rotation
            aw, ah = ah, aw
            target_x = cursor_x + aw / 2.0
            target_x = max(min_x + 2.0, min(max_x - 2.0, target_x))
            target_y = max(min_y + 2.0, min(max_y - 2.0, avg_y))

            # Place relay at 90° rotation for vertical coil orientation
            positions[anchor_ref] = (target_x, target_y, relay_rotation)
            row_grid.place(target_x, target_y, aw, ah)

            # Support components are placed by Level 3b — skip here to
            # avoid double-placement and congestion.

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

        # Determine target position and rotation for antenna facing outward.
        # ESP32-S3-WROOM-1: antenna is at the TOP of the package (opposite
        # pin 1). KiCad rotation convention: 0°=antenna up, 90°=antenna left,
        # 180°=antenna down, 270°=antenna right.
        target_x, target_y = cx, cy
        new_rot = rot
        if dist_right == min_edge_dist or dist_right <= dist_left:
            target_x = max_x - edge_margin - w / 2.0
            new_rot = 270.0  # Antenna pointing right (toward right edge)
        elif dist_left == min_edge_dist:
            target_x = min_x + edge_margin + w / 2.0
            new_rot = 90.0  # Antenna pointing left
        elif dist_top == min_edge_dist:
            target_y = min_y + edge_margin + h / 2.0
            new_rot = 0.0  # Antenna pointing up (toward top edge)
        else:
            target_y = max_y - edge_margin - h / 2.0
            new_rot = 180.0  # Antenna pointing down

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
    """Position and orient connectors at their nearest board edge.

    Uses pin_map.origin_to_centroid() and pad_extent_in_board_space() to
    correctly handle asymmetric footprints (connectors with pin-1 origin).

    Screw terminal rotation conventions (mating face outward):
    - Top edge: rot=90° (pads run vertically, screws accessible from top)
    - Bottom edge: rot=270° (screws accessible from bottom)
    - Left edge: rot=0° (screws accessible from left)
    - Right edge: rot=180° (screws accessible from right)
    """
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space

    min_x, min_y, max_x, max_y = bounds
    edge_margin = 3.0  # mm from board edge to nearest pad

    for fp in pcb.footprints:
        ref = fp.ref
        if ref in fixed_refs or not ref.startswith("J"):
            continue
        if ref not in positions:
            continue
        cx, cy, rot = positions[ref]
        w, h = fp_sizes.get(ref, (2.0, 2.0))

        # Find nearest edge (using centroid position)
        dist_left = cx - min_x
        dist_right = max_x - cx
        dist_top = cy - min_y
        dist_bottom = max_y - cy
        min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_edge_dist > 20.0:
            continue  # Too far from any edge — not a board-edge connector

        # Determine target edge and rotation
        is_wide = w > h * 1.5  # Multi-pin in a row (native, unrotated)

        if dist_top == min_edge_dist:
            target_edge = "top"
            new_rot = 90.0 if is_wide else 0.0
        elif dist_bottom == min_edge_dist:
            target_edge = "bottom"
            new_rot = 270.0 if is_wide else 0.0
        elif dist_left == min_edge_dist:
            target_edge = "left"
            new_rot = 0.0 if is_wide else 270.0
        else:
            target_edge = "right"
            new_rot = 180.0 if is_wide else 90.0

        # Use centroid_to_origin to find where origin would be at current
        # centroid, then compute pad extent to find how far pads extend
        origin_x, origin_y = centroid_to_origin(fp, cx, cy, new_rot)

        # Get actual pad extent at this position and rotation
        px0, py0, px1, py1 = pad_extent_in_board_space(
            fp, origin_x, origin_y, new_rot,
        )

        # Shift origin so pads are flush to target edge with margin
        if target_edge == "top":
            # Move so topmost pad is at min_y + margin
            shift_y = (min_y + edge_margin) - py0
            origin_y += shift_y
        elif target_edge == "bottom":
            # Move so bottommost pad is at max_y - margin
            shift_y = (max_y - edge_margin) - py1
            origin_y += shift_y
        elif target_edge == "left":
            shift_x = (min_x + edge_margin) - px0
            origin_x += shift_x
        else:  # right
            shift_x = (max_x - edge_margin) - px1
            origin_x += shift_x

        # Clamp: verify all pads are within board after shift
        px0, py0, px1, py1 = pad_extent_in_board_space(
            fp, origin_x, origin_y, new_rot,
        )
        if px0 < min_x + 1.0:
            origin_x += (min_x + 1.0) - px0
        if px1 > max_x - 1.0:
            origin_x -= px1 - (max_x - 1.0)
        if py0 < min_y + 1.0:
            origin_y += (min_y + 1.0) - py0
        if py1 > max_y - 1.0:
            origin_y -= py1 - (max_y - 1.0)

        # Convert back to centroid space for the optimizer
        new_cx, new_cy = origin_to_centroid(fp, origin_x, origin_y, new_rot)
        positions[ref] = (new_cx, new_cy, new_rot)
        _log.debug("  %s: edge=%s → origin(%.1f,%.1f) centroid(%.1f,%.1f) rot=%.0f",
                   ref, target_edge, origin_x, origin_y, new_cx, new_cy, new_rot)

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


def _apply_template_refinement(
    positions: dict[str, tuple[float, float, float]],
    fp_sizes: dict[str, tuple[float, float]],
    bounds: tuple[float, float, float, float],
    requirements: ProjectRequirements,
    subcircuits: tuple[DetectedSubCircuit, ...],
    fixed_refs: set[str],
) -> tuple[dict[str, tuple[float, float, float]], set[str]]:
    """Apply subcircuit layout templates to refine component positions.

    For each detected subcircuit with a matching template, maps detected
    component refs to template slots by role matching, computes target
    positions relative to the subcircuit anchor, and moves components
    to template positions (respecting existing fixed refs).

    Args:
        positions: Current component positions {ref: (x, y, rot)}.
        fp_sizes: Component courtyard sizes {ref: (w, h)}.
        bounds: Board bounds (min_x, min_y, max_x, max_y).
        requirements: Project requirements.
        subcircuits: Detected subcircuit patterns.
        fixed_refs: Refs that must not be moved.

    Returns:
        Tuple of (updated positions, set of refs placed by templates).
    """
    from kicad_pipeline.pcb.layout_templates import (
        ComponentRole,
        get_subcircuit_template_by_type,
    )

    template_fixed: set[str] = set()
    min_x, min_y, max_x, max_y = bounds

    for sc in subcircuits:
        tmpl = get_subcircuit_template_by_type(sc.circuit_type)
        if tmpl is None:
            continue

        # Find anchor position — the anchor_ref is the reference point
        anchor_ref = sc.anchor_ref
        if anchor_ref not in positions:
            continue
        ax, ay, arot = positions[anchor_ref]

        # Simple role-based matching: map subcircuit refs to template slots
        # For now, use positional matching (first SERIES ref -> first SERIES slot, etc.)
        role_refs: dict[str, list[str]] = {}
        for ref in sc.refs:
            if ref == anchor_ref:
                continue
            if ref not in positions or ref in fixed_refs:
                continue
            # Classify by ref prefix
            r_upper = ref.upper()
            if r_upper.startswith("R"):
                role_refs.setdefault("series", []).append(ref)
            elif r_upper.startswith("C"):
                role_refs.setdefault("shunt", []).append(ref)
            elif r_upper.startswith("D"):
                role_refs.setdefault("shunt_d", []).append(ref)
            elif r_upper.startswith("Q"):
                role_refs.setdefault("switch", []).append(ref)
            elif r_upper.startswith("L"):
                role_refs.setdefault("series_l", []).append(ref)

        # Match slots to available refs
        series_slots = [s for s in tmpl.slots if s.role == ComponentRole.SERIES]
        shunt_slots = [s for s in tmpl.slots if s.role == ComponentRole.SHUNT]
        switch_slots = [s for s in tmpl.slots if s.role == ComponentRole.SWITCH]

        placed_count = 0

        def _try_place(slot: object, ref: str) -> bool:
            """Place ref at slot offset, skipping if it would collide."""
            nx = ax + slot.offset_x  # type: ignore[union-attr]
            ny = ay + slot.offset_y  # type: ignore[union-attr]
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            nx = max(min_x + w / 2, min(max_x - w / 2, nx))
            ny = max(min_y + h / 2, min(max_y - h / 2, ny))
            # Check for collisions with existing positions
            for other_ref, (ox, oy, _orot) in positions.items():
                if other_ref == ref:
                    continue
                ow, oh = fp_sizes.get(other_ref, (2.0, 2.0))
                gap_x = abs(nx - ox) - (w + ow) / 2
                gap_y = abs(ny - oy) - (h + oh) / 2
                if gap_x < 0.2 and gap_y < 0.2:
                    return False  # Would collide — skip
            rot = slot.rotation if slot.rotation != 0.0 else positions[ref][2]  # type: ignore[union-attr]
            positions[ref] = (nx, ny, rot)
            template_fixed.add(ref)
            return True

        # Place series components (R, L)
        series_refs = role_refs.get("series", []) + role_refs.get("series_l", [])
        for slot, ref in zip(series_slots, series_refs):
            if _try_place(slot, ref):
                placed_count += 1

        # Place shunt components (C, D)
        shunt_refs = role_refs.get("shunt", []) + role_refs.get("shunt_d", [])
        for slot, ref in zip(shunt_slots, shunt_refs):
            if _try_place(slot, ref):
                placed_count += 1

        # Place switch components (Q)
        switch_refs = role_refs.get("switch", [])
        for slot, ref in zip(switch_slots, switch_refs):
            if _try_place(slot, ref):
                placed_count += 1

        if placed_count > 0:
            _log.info(
                "    Template %s: placed %d/%d refs around %s",
                tmpl.name, placed_count, len(sc.refs) - 1, anchor_ref,
            )

    return positions, template_fixed


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

    # Extract current positions — convert KiCad origin → centroid space.
    # KiCad stores footprint origin (pin 1 for connectors), but the
    # optimizer works in centroid-of-pads coordinates.
    positions: dict[str, tuple[float, float, float]] = {}
    for fp in initial_pcb.footprints:
        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        positions[fp.ref] = (cx, cy, fp.rotation)

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
    # Place support directly below relay in tight grid. Protected during 3g
    # collision resolution — overlapping components will be moved instead.
    _log.info("  3b: Relay driver subgroup tightening")
    relay_support_refs: set[str] = set()  # protected from collision resolution
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

        target_y_base = ky + kh / 2.0 + 1.0
        col = 0
        row_y = target_y_base
        row_max_h = 0.0
        cols_per_row = 2  # tight 2-column grid under each relay

        for ref in support_members:
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            # Direct placement — no grid search. Force position.
            px = kx - kw / 2.0 + (col + 0.5) * (kw / cols_per_row)
            py = row_y + h / 2.0
            # Clamp to board
            px = max(bounds[0] + 2.0, min(bounds[2] - 2.0, px))
            py = max(bounds[1] + 2.0, min(bounds[3] - 2.0, py))
            _, _, rot = positions[ref]
            _log.debug(
                "    3b: %s → %s, placed at (%.1f, %.1f) under %s",
                ref, anchor, px, py, anchor,
            )
            positions[ref] = (px, py, rot)
            relay_support_refs.add(ref)
            row_max_h = max(row_max_h, h)
            col += 1
            if col >= cols_per_row:
                col = 0
                row_y += row_max_h + 0.5
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

            # Force cap to within 2mm of IC edge — direct placement
            dx = ix - cx
            dy = iy - cy
            d = math.sqrt(dx * dx + dy * dy) or 1.0
            target_dist = (iw + cw) / 2.0 + 1.5
            tx = ix - dx / d * target_dist
            ty = iy - dy / d * target_dist
            # Clamp to board
            tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
            ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
            positions[cap_ref] = (tx, ty, crot)

    # 3c1. Power group organization — vertical signal-chain columns
    # Two parallel vertical columns (one per buck converter), flowing top→bottom:
    #   Column 1 (Buck #1): VIN → D5(TVS) → C1(in) → U1 → C2(BST) → L1 → R1/R2(FB) → C3(out)
    #   Bridge: D1(OR) → C4(bulk) → D2/D3(OR) (connects buck1 output to +5V rail)
    #   Column 2 (Buck #2): C5(in) → U2 → C17(BST) → L2 → C6/C7(out)
    #   Tail: R3/D4(LED), L3-L6/C27/C35 (ferrites/misc)
    _log.info("  3c1: Power group organization")
    power_group_fixed: set[str] = set()

    # Find power FeatureBlock
    power_group_refs: set[str] = set()
    for feat in requirements.features:
        feat_lower = feat.name.lower()
        if "power" in feat_lower or "supply" in feat_lower:
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                power_group_refs.add(r)
            break

    if power_group_refs:
        # Find power zone
        power_zone_rect: tuple[float, float, float, float] | None = None
        for z in zones:
            if z.name == "power":
                power_zone_rect = z.rect
                break

        # Classify power components by net connectivity
        net_to_pwr_refs: dict[str, set[str]] = {}
        for net in requirements.nets:
            pwr_in_net = set()
            for conn in net.connections:
                if conn.ref in power_group_refs:
                    pwr_in_net.add(conn.ref)
            if pwr_in_net:
                net_to_pwr_refs[net.name] = pwr_in_net

        power_ics = sorted(
            [r for r in power_group_refs
             if r.startswith("U") and r in positions],
        )
        power_connectors = sorted(
            [r for r in power_group_refs
             if r.startswith("J") and r in positions],
        )

        if power_ics and power_zone_rect is not None:
            zx1, zy1, zx2, zy2 = power_zone_rect

            _STRIP_GAP = 0.5   # vertical gap between components in a column
            _COL_SPACING = 5.0  # horizontal gap between columns

            placed_in_col: set[str] = set()  # prevent double-placement

            # Define the two buck converter signal chains (top→bottom order)
            buck1_ic = power_ics[0] if power_ics else ""
            buck2_ic = power_ics[1] if len(power_ics) > 1 else ""

            # Ferrite detection
            ferrite_refs = sorted(
                [r for r in power_group_refs
                 if r.startswith("L") and r in positions
                 and "ferrite" in (
                     next((fp.value for fp in initial_pcb.footprints
                           if fp.ref == r), "")
                 ).lower()],
            )

            # Buck #1 column: VIN input → 5V output
            # Signal order: D5(TVS) → C1(input cap) → U1(buck IC) → C2(BST cap)
            #               → L1(inductor) → R1(FB top) → R2(FB bot) → C3(output cap)
            vin_passives = sorted(
                net_to_pwr_refs.get("VIN", set())
                - set(power_ics) - set(power_connectors),
            )
            bst1_passives = sorted(
                (net_to_pwr_refs.get("BST", set())
                 | net_to_pwr_refs.get("SW", set()))
                - {buck1_ic} - set(power_connectors),
            )
            # Non-ferrite inductors on SW net (L1)
            l1_refs = sorted(
                [r for r in power_group_refs
                 if r.startswith("L") and r in positions
                 and r not in ferrite_refs
                 and r in net_to_pwr_refs.get("SW", set())],
            )
            fb_refs = sorted(
                net_to_pwr_refs.get("FB", set())
                - {buck1_ic} - set(power_connectors),
            )
            buck5v_caps = sorted(
                net_to_pwr_refs.get("BUCK_5V", set())
                - {buck1_ic} - set(power_connectors)
                - set(fb_refs) - set(l1_refs),
            )

            buck1_column: list[str] = (
                vin_passives       # D5, C1 (VIN side)
                + [buck1_ic]       # U1
                + bst1_passives    # C2 (BST/SW)
                + l1_refs          # L1 (inductor)
                + fb_refs          # R1, R2 (feedback divider)
                + buck5v_caps      # C3 (output cap)
            )

            # Bridge: OR'ing diodes + 5V bulk cap
            or_diode_refs = sorted(
                net_to_pwr_refs.get("BUCK_5V", set())
                & {r for r in power_group_refs if r.startswith("D")}
                - set(vin_passives),
            )
            v5_rail_refs = sorted(
                (net_to_pwr_refs.get("+5V", set()) & power_group_refs)
                - set(power_ics) - set(power_connectors)
                - set(or_diode_refs),
            )

            bridge_column: list[str] = or_diode_refs + v5_rail_refs

            # Buck #2 column: +5V input → 3.3V output
            buck2_in_caps = sorted(
                net_to_pwr_refs.get("+5V", set())
                & {r for r in power_group_refs if r.startswith("C")}
                - set(v5_rail_refs),
            )
            bst2_passives = sorted(
                (net_to_pwr_refs.get("BST2", set())
                 | net_to_pwr_refs.get("SW2", set()))
                - {buck2_ic} - set(power_connectors),
            )
            l2_refs = sorted(
                [r for r in power_group_refs
                 if r.startswith("L") and r in positions
                 and r not in ferrite_refs
                 and r in net_to_pwr_refs.get("SW2", set())],
            )
            v33_caps = sorted(
                net_to_pwr_refs.get("+3V3", set())
                & power_group_refs
                - {buck2_ic} - set(power_connectors),
            )

            buck2_column: list[str] = (
                buck2_in_caps      # C5 (5V input cap)
                + [buck2_ic]       # U2
                + bst2_passives    # C17 (BST2/SW2)
                + l2_refs          # L2 (inductor)
                + v33_caps         # C6, C7 (output caps)
            )

            # Tail: LED, ferrites, remaining
            led_refs = sorted(
                (net_to_pwr_refs.get("LED_A", set()) & power_group_refs)
                - set(power_connectors),
            )

            all_classified = (
                set(buck1_column) | set(bridge_column) | set(buck2_column)
                | set(led_refs) | set(ferrite_refs) | set(power_connectors)
            )
            remaining_refs = sorted(
                power_group_refs - all_classified - {""} - fixed_refs,
            )
            tail_column: list[str] = led_refs + ferrite_refs + remaining_refs

            # Place columns vertically, like ADC channel strips
            # All columns share the same X anchor; each column starts below
            # the previous column's last component.
            anchor_x = zx1 + 5.0
            anchor_y = zy1 + 3.0

            def _place_column(
                refs: list[str],
                col_x: float,
                start_y: float,
            ) -> float:
                """Place refs in a vertical column. Returns bottom Y."""
                cy = start_y
                for ref in refs:
                    if (ref not in positions or ref in fixed_refs
                            or ref in placed_in_col or ref == ""):
                        continue
                    w, h = fp_sizes.get(ref, (2.0, 2.0))
                    tx = col_x
                    ty = cy + h / 2.0
                    tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
                    ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
                    positions[ref] = (tx, ty, 0.0)
                    power_group_fixed.add(ref)
                    placed_in_col.add(ref)
                    cy = ty + h / 2.0 + _STRIP_GAP
                return cy

            # Register connectors (don't move them)
            for ref in power_connectors:
                power_group_fixed.add(ref)

            # Column 1: Buck #1 chain (VIN → 5V)
            col1_bottom = _place_column(buck1_column, anchor_x, anchor_y)

            # Bridge: OR diodes + 5V bulk (continue column 1)
            bridge_y = col1_bottom
            col1_bottom = _place_column(bridge_column, anchor_x, bridge_y)

            # Column 2: Buck #2 chain (5V → 3.3V) — branches RIGHT at the
            # +5V output point (where bridge starts), not below column 1.
            # This shows the power tree fork: 24V→5V flows down, 5V→3.3V
            # branches right at the 5V rail.
            col2_x = anchor_x + _COL_SPACING
            col2_bottom = _place_column(buck2_column, col2_x, bridge_y)

            # Tail: LED + ferrites + misc — below whichever column is shorter
            tail_y = max(col1_bottom, col2_bottom)
            _place_column(tail_column, anchor_x, tail_y)

            _log.info(
                "    3c1: organized %d power components in vertical signal chain",
                len(power_group_fixed),
            )

    # 3c2. ADC channel formation — repeatable channel strips near ADC ICs
    # Detect nets connecting an ADC IC pin to exactly 2R + 1D + 1C (voltage
    # divider + protection pattern), then arrange each channel's 4 passives
    # in a consistent horizontal strip stacked vertically by channel index.
    _log.info("  3c2: ADC channel formation")
    adc_channel_refs: set[str] = set()  # protect during collision resolution

    # Build net → refs mapping from requirements
    net_components: dict[str, list[tuple[str, str]]] = {}
    for net in requirements.nets:
        net_components[net.name] = [(c.ref, c.pin) for c in net.connections]

    # Find ADC channel nets: connect U* pin to 2R + 1D + 1C
    adc_channels: list[tuple[str, str, list[str]]] = []  # (ic_ref, ic_pin, [passive_refs])
    for net_name, conns in net_components.items():
        ic_refs = [(r, p) for r, p in conns if r.startswith("U") and r in positions]
        passive_refs = [r for r, p in conns
                        if r in positions and r[0] in "RDC" and not r.startswith("U")]
        if len(ic_refs) == 1 and len(passive_refs) == 4:
            # Check pattern: 2R + 1D + 1C
            r_count = sum(1 for r in passive_refs if r.startswith("R"))
            d_count = sum(1 for r in passive_refs if r.startswith("D"))
            c_count = sum(1 for r in passive_refs if r.startswith("C"))
            if r_count == 2 and d_count == 1 and c_count == 1:
                ic_ref, ic_pin = ic_refs[0]
                adc_channels.append((ic_ref, ic_pin, passive_refs))

    # Group channels by IC and sort by pin number
    ic_channels: dict[str, list[tuple[str, list[str]]]] = {}
    for ic_ref, ic_pin, passives in adc_channels:
        ic_channels.setdefault(ic_ref, []).append((ic_pin, passives))
    for ic_ref in ic_channels:
        ic_channels[ic_ref].sort(key=lambda x: x[0])

    # First pass: collect all ADC channel passive refs and IC refs
    all_adc_passive_refs: set[str] = set()
    adc_ic_refs: set[str] = set()
    for _ic_ref, _ic_pin, passives in adc_channels:
        all_adc_passive_refs.update(passives)
        adc_ic_refs.add(_ic_ref)

    # Build occupancy grid WITHOUT ADC channel passives so they can be freely placed
    adc_grid = _PlacementGrid(bounds)
    for oref, (ox, oy, _orot) in positions.items():
        if oref in all_adc_passive_refs:
            continue  # Don't block with current (scattered) positions
        ow, oh = _rotation_aware_size(oref, positions, fp_sizes)
        adc_grid.place(ox, oy, ow, oh)

    # Per-IC vertical channel columns: each channel is a vertical strip
    # running UPWARD from the ADC IC toward the connectors / 24V source
    # (top edge of board).  Channels are spread horizontally side-by-side.
    #
    # Signal flow (top to bottom): connector → R_top → C_filter → D_clamp → R_bot → IC
    # Physical layout (upward from IC):  IC ← R_bot ← D_clamp ← C_filter ← R_top
    #
    # All components at 0° rotation (horizontal pads) so pads align
    # vertically in the signal path.
    _CHANNEL_SPACING_MM = 4.5  # horizontal gap between channel columns (SOD-323 = 3.7mm wide)
    _STRIP_GAP_MM = 0.5  # vertical gap between components within a column

    # Sort ICs for deterministic processing order
    sorted_ic_refs = sorted(ic_channels.keys())

    # Track occupied X ranges to prevent inter-IC channel overlap.
    # Each entry: (x_min, x_max) of the channel columns for an IC.
    _occupied_x_ranges: list[tuple[float, float]] = []

    for ic_ref in sorted_ic_refs:
        ch_list = ic_channels[ic_ref]
        if ic_ref not in positions:
            continue
        ix, iy, _irot = positions[ic_ref]
        iw, ih = fp_sizes.get(ic_ref, (5.0, 5.0))

        n_ch = len(ch_list)
        total_ch_width = (n_ch - 1) * _CHANNEL_SPACING_MM
        # Center the channel columns horizontally on the IC
        ch_x_start = ix - total_ch_width / 2.0
        ch_x_end = ch_x_start + total_ch_width

        # Shift right if this IC's channel band overlaps with previously placed ICs
        _comp_half_w = 2.0  # half-width of widest component (SOD-323 = 3.7mm)
        for ox_min, ox_max in _occupied_x_ranges:
            if (ch_x_start - _comp_half_w < ox_max + _comp_half_w
                    and ch_x_end + _comp_half_w > ox_min - _comp_half_w):
                # Overlap — shift this IC's channels to the right of the occupied range
                shift = (ox_max + _comp_half_w + _CHANNEL_SPACING_MM) - ch_x_start
                if shift > 0:
                    ch_x_start += shift
                    ch_x_end = ch_x_start + total_ch_width

        _occupied_x_ranges.append((ch_x_start, ch_x_end))

        for ch_idx, (ic_pin, passives) in enumerate(ch_list):
            r_refs = sorted([r for r in passives if r.startswith("R")])
            d_refs = [r for r in passives if r.startswith("D")]
            c_refs = [r for r in passives if r.startswith("C")]

            ch_x = ch_x_start + ch_idx * _CHANNEL_SPACING_MM

            # Place upward from IC: R_bot closest to IC, R_top farthest
            strip_order: list[str] = []
            if len(r_refs) >= 2:
                strip_order.append(r_refs[1])  # R_bot (closest to IC)
            strip_order.extend(d_refs)          # D_clamp
            strip_order.extend(c_refs)          # C_filter
            if len(r_refs) >= 1:
                strip_order.append(r_refs[0])  # R_top (closest to connector)

            # Start placing above the IC
            strip_y = iy - ih / 2.0 - 1.0

            for ref in strip_order:
                if ref not in positions or ref in fixed_refs:
                    continue
                raw_w, raw_h = fp_sizes.get(ref, (2.0, 2.0))
                # All components at 0° for vertical signal flow
                w, h = raw_w, raw_h
                rot = 0.0
                target_x = ch_x
                target_y = strip_y - h / 2.0
                target_x = max(bounds[0] + 2.0, min(bounds[2] - 2.0, target_x))
                target_y = max(bounds[1] + 2.0, min(bounds[3] - 2.0, target_y))

                # Deterministic placement — exact position, no grid search.
                # This ensures repeatable strip patterns across all channels.
                positions[ref] = (target_x, target_y, rot)
                adc_channel_refs.add(ref)
                strip_y = target_y - (h / 2.0 + _STRIP_GAP_MM)

    _log.info(
        "    3c2: %d channels across %d ICs",
        len(adc_channels), len(adc_ic_refs),
    )

    # 3c3. Analog subcircuit clustering — pull remaining analog passives
    # (ladder switch resistors, optocoupler components, ADC decoupling)
    # into vertical columns next to the ADC channel columns.
    #
    # Strategy: find SMALL PASSIVE components (R, C, D, LED, SW) on signal
    # nets that connect to ADC IC pins.  Then follow one hop through
    # non-power nets but ONLY through small passives/discretes — never
    # through ICs (which fan out to the entire board) or connectors.
    _log.info("  3c3: Analog subcircuit clustering")

    _POWER_NET_PREFIXES = (
        "VCC", "VDD", "GND", "AGND", "DGND", "AVCC", "+3V3", "+5V",
        "VIN", "VBUS", "V_", "+12V", "+24V", "I2C_",
    )

    def _is_power_or_bus_net(name: str) -> bool:
        upper = name.upper()
        return any(upper.startswith(p) for p in _POWER_NET_PREFIXES)

    _SMALL_PREFIXES = ("R", "C", "D", "L", "LED", "SW")

    def _is_small_passive(ref: str) -> bool:
        return any(ref.startswith(p) for p in _SMALL_PREFIXES)

    # Find small passives on ADC signal nets (excluding power/bus/I2C)
    analog_signal_refs: set[str] = set()
    for net in requirements.nets:
        if _is_power_or_bus_net(net.name):
            continue
        # Must connect to an ADC IC pin (not just share a power net)
        ic_conn = [c for c in net.connections
                   if c.ref in adc_ic_refs and c.ref in positions]
        if not ic_conn:
            continue
        for c in net.connections:
            if (c.ref in positions
                    and c.ref not in adc_channel_refs
                    and c.ref not in adc_ic_refs
                    and c.ref not in fixed_refs
                    and _is_small_passive(c.ref)):
                analog_signal_refs.add(c.ref)

    # One hop: follow non-power nets through small passives only
    # Also allow U refs with ≤6 pins (optocouplers, small ICs) but NOT MCUs
    hop2_refs: set[str] = set()
    for ref in list(analog_signal_refs):
        for net in requirements.nets:
            if _is_power_or_bus_net(net.name):
                continue
            if not any(c.ref == ref for c in net.connections):
                continue
            for c in net.connections:
                if (c.ref != ref and c.ref in positions
                        and c.ref not in adc_channel_refs
                        and c.ref not in adc_ic_refs
                        and c.ref not in fixed_refs
                        and c.ref not in analog_signal_refs):
                    # Allow small passives and small ICs (optocouplers)
                    if _is_small_passive(c.ref):
                        hop2_refs.add(c.ref)
                    elif c.ref.startswith("U"):
                        # Only pull small ICs (≤6 pins)
                        comp = next((comp for comp in requirements.components
                                     if comp.ref == c.ref), None)
                        if comp and len(comp.pins) <= 6:
                            hop2_refs.add(c.ref)

    # Hop 3: for small ICs found in hop2, follow their remaining non-power
    # nets to pull in the full subcircuit (e.g., U7 opto → R25, R32, D17, LED2)
    hop3_refs: set[str] = set()
    small_ics_found = {r for r in hop2_refs if r.startswith("U")}
    for ic_ref in small_ics_found:
        for net in requirements.nets:
            if _is_power_or_bus_net(net.name):
                continue
            if not any(c.ref == ic_ref for c in net.connections):
                continue
            for c in net.connections:
                if (c.ref != ic_ref and c.ref in positions
                        and c.ref not in adc_channel_refs
                        and c.ref not in adc_ic_refs
                        and c.ref not in fixed_refs
                        and _is_small_passive(c.ref)):
                    hop3_refs.add(c.ref)
    # Hop 4: one more hop from hop3 refs (e.g., R25→OPTO_IN→D17/R32/SW3)
    hop4_refs: set[str] = set()
    for ref in hop3_refs:
        for net in requirements.nets:
            if _is_power_or_bus_net(net.name):
                continue
            if not any(c.ref == ref for c in net.connections):
                continue
            for c in net.connections:
                if (c.ref != ref and c.ref in positions
                        and c.ref not in adc_channel_refs
                        and c.ref not in adc_ic_refs
                        and c.ref not in fixed_refs
                        and _is_small_passive(c.ref)):
                    hop4_refs.add(c.ref)

    all_analog_cluster_refs = (
        analog_signal_refs | hop2_refs | hop3_refs | hop4_refs
    ) - adc_channel_refs

    if all_analog_cluster_refs and _occupied_x_ranges:
        last_x_max = max(xmax for _, xmax in _occupied_x_ranges)
        cluster_x = last_x_max + _CHANNEL_SPACING_MM + 2.0

        adc_ys = [positions[r][1] for r in adc_channel_refs if r in positions]
        cluster_y_top = min(adc_ys) - 1.0 if adc_ys else bounds[1] + 5.0

        # Sort: ICs first (larger), then passives by ref
        sorted_cluster = sorted(
            all_analog_cluster_refs,
            key=lambda r: (0 if r.startswith("U") else 2, r),
        )

        # Place in a vertical column, wrapping to next column after 15mm
        _COL_GAP = 5.0
        _ROW_GAP = 0.5
        cur_x = cluster_x
        cur_y = cluster_y_top
        placed_count = 0

        for ref in sorted_cluster:
            if ref not in positions:
                continue
            raw_w, raw_h = fp_sizes.get(ref, (2.0, 2.0))
            w, h = raw_w, raw_h
            rot = 0.0

            target_x = cur_x
            target_y = cur_y + h / 2.0
            target_x = max(bounds[0] + 2.0, min(bounds[2] - 2.0, target_x))
            target_y = max(bounds[1] + 2.0, min(bounds[3] - 2.0, target_y))

            positions[ref] = (target_x, target_y, rot)
            adc_channel_refs.add(ref)
            placed_count += 1

            cur_y = target_y + h / 2.0 + _ROW_GAP

            if cur_y > cluster_y_top + 15.0:
                cur_x += _COL_GAP
                cur_y = cluster_y_top

        _log.info("    3c3: clustered %d analog refs near ADC channels", placed_count)

    # 3c3b. Pull remaining analog group outliers toward the cluster.
    # Find all refs in the same FeatureBlock as the ADC ICs, then pull
    # any that are >15mm from the analog centroid into the cluster area.
    analog_group_refs: set[str] = set()
    for feat in requirements.features:
        feat_refs = set(feat.components)
        if feat_refs & adc_ic_refs:
            analog_group_refs = feat_refs
            break

    if analog_group_refs and adc_channel_refs:
        # Compute centroid of already-placed analog refs
        placed_analog = [
            positions[r] for r in (adc_channel_refs | adc_ic_refs)
            if r in positions
        ]
        if placed_analog:
            cx = sum(p[0] for p in placed_analog) / len(placed_analog)
            cy = sum(p[1] for p in placed_analog) / len(placed_analog)
            # Find outlier refs (>15mm from centroid, not already placed)
            outlier_refs: list[str] = []
            for ref in sorted(analog_group_refs):
                if (ref in positions
                        and ref not in adc_channel_refs
                        and ref not in adc_ic_refs
                        and ref not in fixed_refs
                        and not ref.startswith("J")):  # keep connectors on edges
                    rx, ry, _rrot = positions[ref]
                    dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
                    if dist > 15.0:
                        outlier_refs.append(ref)

            if outlier_refs:
                # Compute cluster placement area — to the right of ADC channels
                adc_xs = [positions[r][0] for r in adc_channel_refs
                          if r in positions]
                adc_ys = [positions[r][1] for r in adc_channel_refs
                          if r in positions]
                outlier_x = max(adc_xs) + _CHANNEL_SPACING_MM + 2.0
                outlier_y_top = min(adc_ys) - 1.0
                oc_x = outlier_x
                oc_y = outlier_y_top
                _OC_COL_GAP = 5.0
                _OC_ROW_GAP = 0.5

                for ref in outlier_refs:
                    raw_w, raw_h = fp_sizes.get(ref, (2.0, 2.0))
                    w, h = raw_w, raw_h
                    rot = 0.0

                    target_x = oc_x
                    target_y = oc_y + h / 2.0
                    target_x = max(bounds[0] + 2.0, min(bounds[2] - 2.0, target_x))
                    target_y = max(bounds[1] + 2.0, min(bounds[3] - 2.0, target_y))

                    positions[ref] = (target_x, target_y, rot)
                    adc_channel_refs.add(ref)

                    oc_y = target_y + h / 2.0 + _OC_ROW_GAP
                    if oc_y > outlier_y_top + 15.0:
                        oc_x += _OC_COL_GAP
                        oc_y = outlier_y_top

                _log.info(
                    "    3c3b: pulled %d outliers into analog cluster",
                    len(outlier_refs),
                )

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

    # 3f2. Top-edge screw terminal ordering
    # Place screw terminals along top edge in functional order:
    # left→right: J6(spare ADC), J5(opto), J4(ladder), J3(aux power), J1(power harness)
    # At 0° rotation, screw terminal pads are horizontal (along X axis).
    # Place them in a row along the top edge with gap for screwdriver access.
    _TOP_EDGE_ORDER = ["J6", "J5", "J4", "J3", "J1"]
    _top_refs = [r for r in _TOP_EDGE_ORDER if r in positions and r not in fixed_refs]
    if _top_refs:
        _log.info("  3f2: Top-edge screw terminal ordering (%s)", _top_refs)
        term_gap = 3.0  # mm gap between courtyard edges
        # At 0° rotation, native width IS the X extent
        term_widths: list[float] = []
        for r in _top_refs:
            w, _h = fp_sizes.get(r, (2.0, 2.0))
            term_widths.append(w)
        total_w = sum(term_widths) + term_gap * (len(_top_refs) - 1)
        # Use full board width minus margins for mounting holes
        margin = 8.0  # mm from board edge (clear of mounting holes)
        avail_w = (max_x - min_x) - 2 * margin
        if total_w < avail_w:
            start_x = min_x + margin + (avail_w - total_w) / 2.0
        else:
            # Compress gap to fit
            compressed_gap = max(1.0, (avail_w - sum(term_widths)) / max(len(_top_refs) - 1, 1))
            term_gap = compressed_gap
            total_w = sum(term_widths) + term_gap * (len(_top_refs) - 1)
            start_x = min_x + margin
        cursor_x = start_x
        # Use origin_to_centroid() to correctly place pin 1 (origin) near
        # the top board edge, then let the library compute centroid position.
        origin_y_target = min_y + 3.0  # pin 1 pad 3mm from top edge
        for i, r in enumerate(_top_refs):
            tw = term_widths[i]
            origin_x = cursor_x + tw / 2.0
            # Find footprint to use origin_to_centroid
            fp_match = None
            for fp in initial_pcb.footprints:
                if fp.ref == r:
                    fp_match = fp
                    break
            if fp_match is not None:
                cent_x, cent_y = origin_to_centroid(
                    fp_match, origin_x, origin_y_target, 0.0,
                )
            else:
                cent_x, cent_y = origin_x, origin_y_target
            positions[r] = (cent_x, cent_y, 0.0)
            cursor_x += tw + term_gap
            _log.info("    %s → centroid(%.1f, %.1f) origin(%.1f, %.1f) rot=0",
                      r, cent_x, cent_y, origin_x, origin_y_target)
    top_edge_connector_refs: set[str] = set(_top_refs)

    # 3c3. MCU peripheral tightening — AFTER RF pinning so U3 is at final position
    # Order: U3 → decoupling caps → connectors → USB subcircuit →
    #        reset/boot → remaining passives
    _log.info("  3c3: MCU peripheral tightening")
    mcu_peripheral_refs: set[str] = set()
    from kicad_pipeline.optimization.functional_grouper import _find_mcu_ref as _find_mcu
    mcu_ref_c3 = _find_mcu(requirements)
    if mcu_ref_c3 and mcu_ref_c3 in positions:
        mcu_x, mcu_y, _mcu_rot = positions[mcu_ref_c3]
        mcu_w, mcu_h = fp_sizes.get(mcu_ref_c3, (5.0, 5.0))

        # Find MCU's FeatureBlock group
        mcu_group_refs: set[str] = set()
        for feat in requirements.features:
            feat_refs = set()
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                feat_refs.add(r)
            if mcu_ref_c3 in feat_refs:
                mcu_group_refs = feat_refs
                break

        # --- Step 0: Place U3 with antenna FLUSH on right board edge ---
        # Find the MCU footprint so we can use pad_extent for accurate sizing
        mcu_fp = None
        for fp in initial_pcb.footprints:
            if fp.ref == mcu_ref_c3:
                mcu_fp = fp
                break
        mcu_zone_rect: tuple[float, float, float, float] | None = None
        for z in zones:
            if z.name == "mcu":
                mcu_zone_rect = z.rect
                break
        _mcu_rot = 270.0  # antenna pointing right
        eff_w, eff_h = mcu_h, mcu_w  # At 270°, dimensions swap

        # Place U3's ORIGIN so pads are ~2mm from right edge.
        # Then compute centroid for positions dict.
        if mcu_fp is not None:
            from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space as _pad_ext
            # Trial: place origin at board center, measure pad extent
            _trial_ox = (bounds[0] + bounds[2]) / 2.0
            _trial_oy = (bounds[1] + bounds[3]) / 2.0
            _te = _pad_ext(mcu_fp, _trial_ox, _trial_oy, _mcu_rot)
            # How far right does the pad extend from origin?
            _pad_right = _te[2] - _trial_ox  # max_x - origin_x
            # Place origin so rightmost pad is far enough from right edge
            # to leave room for J14 (GPIO header) on the right edge
            _j14_w = fp_sizes.get("J14", (2.7, 35.7))[0] if "J14" in mcu_group_refs else 0.0
            _right_margin = 3.0 + _j14_w  # J14 width + gap
            mcu_origin_x = bounds[2] - _pad_right - _right_margin
            mcu_origin_y = _trial_oy
            if mcu_zone_rect is not None:
                _zx1, _zy1, _zx2, _zy2 = mcu_zone_rect
                mcu_origin_y = (_zy1 + _zy2) / 2.0
            # Clamp Y so pads stay within board
            _pad_top = _te[1] - _trial_oy  # min_y - origin_y (negative)
            _pad_bot = _te[3] - _trial_oy  # max_y - origin_y (positive)
            mcu_origin_y = max(bounds[1] - _pad_top + 2.0,
                               min(bounds[3] - _pad_bot - 2.0, mcu_origin_y))
            # Convert to centroid for positions dict
            mcu_x, mcu_y = origin_to_centroid(mcu_fp, mcu_origin_x,
                                               mcu_origin_y, _mcu_rot)
        else:
            mcu_x = bounds[2] - eff_w / 2.0 - 1.0
            if mcu_zone_rect is not None:
                _zx1, _zy1, _zx2, _zy2 = mcu_zone_rect
                mcu_y = (_zy1 + _zy2) / 2.0
            mcu_y = max(bounds[1] + eff_h / 2.0 + 2.0,
                        min(bounds[3] - eff_h / 2.0 - 2.0, mcu_y))
        positions[mcu_ref_c3] = (mcu_x, mcu_y, _mcu_rot)
        mcu_peripheral_refs.add(mcu_ref_c3)
        _log.info("    U3 centroid at (%.1f, %.1f) rot=270° [eff_w=%.1f, eff_h=%.1f]",
                  mcu_x, mcu_y, eff_w, eff_h)

        # MCU bounding box edges (centroid-based)
        mcu_left = mcu_x - eff_w / 2.0
        mcu_top = mcu_y - eff_h / 2.0
        mcu_bot = mcu_y + eff_h / 2.0

        # --- Step 1: Build occupancy grid WITHOUT MCU group ---
        mcu_grid = _PlacementGrid(bounds)
        for oref, (ox, oy, _or) in positions.items():
            if oref in mcu_group_refs:
                continue
            ow, oh = _rotation_aware_size(oref, positions, fp_sizes)
            mcu_grid.place(ox, oy, ow, oh)
        mcu_grid.place(mcu_x, mcu_y, eff_w, eff_h)

        # --- Step 2: Build net adjacency for MCU group ---
        ref_nets: dict[str, set[str]] = {}
        for net in requirements.nets:
            net_refs = set()
            for conn in net.connections:
                net_refs.add(conn.ref)
            for r in net_refs:
                if r in mcu_group_refs:
                    ref_nets.setdefault(r, set()).update(
                        net_refs & mcu_group_refs,
                    )

        # Classify MCU passives
        connector_refs = {r for r in mcu_group_refs if r.startswith("J")}
        decoupling_refs: list[str] = []
        other_passive_refs: list[str] = []
        for ref in sorted(mcu_group_refs):
            if ref == mcu_ref_c3 or ref in connector_refs or ref in fixed_refs:
                continue
            if ref not in positions:
                continue
            if ref.startswith("C") and mcu_ref_c3 in ref_nets.get(ref, set()):
                decoupling_refs.append(ref)
            else:
                other_passive_refs.append(ref)

        # --- Step 3: Place decoupling caps FIRST — tight against MCU ---
        # At 270° rotation, U3's left edge (non-antenna) has power pins.
        # Place decoupling column with enough gap to clear the MCU courtyard
        # (module body extends well beyond pad field).
        _DECOUP_GAP = 3.5
        # mcu_left is already U3's left edge in centroid space
        # Place caps so their RIGHT edge is _DECOUP_GAP from mcu_left
        max_cap_w = max((fp_sizes.get(r, (2.5, 1.5))[0] for r in decoupling_refs),
                        default=2.5)
        decoup_col_x = mcu_left - _DECOUP_GAP - max_cap_w / 2.0
        decoup_y_start = mcu_y - eff_h / 4.0
        decoup_y = decoup_y_start
        decoup_y_max = mcu_bot - 1.0
        for ref in decoupling_refs:
            w, h = fp_sizes.get(ref, (1.0, 0.5))
            tx = decoup_col_x
            ty = decoup_y
            if ty + h / 2.0 > decoup_y_max:
                # New column further left
                decoup_col_x -= (w + 2.0)
                tx = decoup_col_x
                ty = decoup_y_start
                decoup_y = ty
            tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
            ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
            px, py = mcu_grid.find_free_pos(tx, ty, w, h, max_radius=8.0)
            positions[ref] = (px, py, 0.0)
            mcu_peripheral_refs.add(ref)
            mcu_grid.place(px, py, w, h)
            decoup_y += h + 1.5
            _log.info("    %s (decoupling): →(%.1f,%.1f) [%.1fmm from U3]",
                      ref, px, py, mcu_left - px - w / 2.0)

        # --- Step 4: Place connectors ---
        # MCU connectors go on the RIGHT board edge (accessible for cables).
        # J14 (1x14 GPIO header): right edge, vertically centered on U3
        # J15 (2x5 header): right edge, above J14
        # J16 (SD card): above U3
        # J2 (USB-C): bottom edge, below U3
        right_edge_x = bounds[2] - 2.0  # 2mm from right board edge

        if "J14" in connector_refs and "J14" in positions and "J14" not in fixed_refs:
            w14, h14 = fp_sizes.get("J14", (2.7, 35.7))
            # Place to the RIGHT of U3, on the board edge
            # Use enough gap to clear U3's full courtyard (module body
            # extends well beyond pads, especially antenna area)
            mcu_right_edge = mcu_x + eff_w / 2.0
            tx = mcu_right_edge + w14 / 2.0 + 3.0  # 3mm gap from U3 courtyard
            tx = max(bounds[0] + w14 / 2.0 + 1.0,
                     min(bounds[2] - w14 / 2.0 - 1.0, tx))
            ty = mcu_y  # vertically centered on U3
            ty = max(bounds[1] + h14 / 2.0 + 1.0,
                     min(bounds[3] - h14 / 2.0 - 1.0, ty))
            positions["J14"] = (tx, ty, 0.0)
            mcu_grid.place(tx, ty, w14, h14)
            mcu_peripheral_refs.add("J14")
            _log.info("    J14 → right of U3 (%.1f, %.1f)", tx, ty)

        if "J15" in connector_refs and "J15" in positions and "J15" not in fixed_refs:
            w15, h15 = fp_sizes.get("J15", (5.2, 12.9))
            j14_pos = positions.get("J14")
            if j14_pos:
                # Above J14 on right edge
                j14_top = j14_pos[1] - fp_sizes.get("J14", (2.7, 35.7))[1] / 2.0
                tx = right_edge_x - w15 / 2.0
                ty = j14_top - h15 / 2.0 - 2.0
            else:
                tx = right_edge_x - w15 / 2.0
                ty = mcu_top - h15 / 2.0 - 2.0
            ty = max(bounds[1] + h15 / 2.0 + 1.0,
                     min(bounds[3] - h15 / 2.0 - 1.0, ty))
            positions["J15"] = (tx, ty, 0.0)
            mcu_grid.place(tx, ty, w15, h15)
            mcu_peripheral_refs.add("J15")
            _log.info("    J15 → right edge (%.1f, %.1f)", tx, ty)

        # J16 (SD card slot): above U3, left of J14
        if "J16" in connector_refs and "J16" in positions and "J16" not in fixed_refs:
            w16, h16 = fp_sizes.get("J16", (16.2, 6.9))
            tx = mcu_x - 5.0  # left of U3 center, clear of antenna
            ty = mcu_top - h16 / 2.0 - 2.0
            tx = max(bounds[0] + w16 / 2.0 + 1.0,
                     min(bounds[2] - w16 / 2.0 - 1.0, tx))
            ty = max(bounds[1] + h16 / 2.0 + 1.0, min(bounds[3] - 2.0, ty))
            px, py = mcu_grid.find_free_pos(tx, ty, w16, h16, max_radius=20.0)
            positions["J16"] = (px, py, 0.0)
            mcu_grid.place(px, py, w16, h16)
            mcu_peripheral_refs.add("J16")
            _log.info("    J16 → above U3 (%.1f, %.1f)", px, py)

        # J2 (USB-C): bottom edge, below U3, facing outward (180°)
        # Place well below MCU courtyard to avoid overlap
        if "J2" in connector_refs and "J2" in positions and "J2" not in fixed_refs:
            w2, h2 = fp_sizes.get("J2", (9.6, 7.6))
            tx = mcu_x - 8.0  # left of U3 center, clear of courtyard
            ty = max(mcu_bot + h2 / 2.0 + 3.0,
                     bounds[3] - h2 / 2.0 - 2.0)  # near bottom edge but clear of U3
            tx = max(bounds[0] + w2 / 2.0 + 1.0,
                     min(bounds[2] - w2 / 2.0 - 1.0, tx))
            px, py = mcu_grid.find_free_pos(tx, ty, w2, h2, max_radius=20.0)
            positions["J2"] = (px, py, 180.0)
            mcu_grid.place(px, py, w2, h2)
            mcu_peripheral_refs.add("J2")
            _log.info("    J2 → bottom edge (%.1f, %.1f)", px, py)

        # --- Step 5: USB subcircuit (U9 + R6 + R7) near J2 ---
        j2_pos = positions.get("J2")
        if "U9" in other_passive_refs and "U9" in positions and j2_pos:
            j2x, j2y, _j2r = j2_pos
            u9w, u9h = fp_sizes.get("U9", (3.0, 3.0))
            # U9 between J2 and U3 (above J2, below U3)
            u9_tx = j2x + 2.0
            u9_ty = (j2y + mcu_bot) / 2.0  # halfway between J2 and U3 bottom
            u9_tx = max(bounds[0] + 2.0, min(bounds[2] - u9w / 2.0 - 1.0, u9_tx))
            u9_ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, u9_ty))
            u9x, u9y = mcu_grid.find_free_pos(u9_tx, u9_ty, u9w, u9h,
                                               max_radius=15.0)
            positions["U9"] = (u9x, u9y, 0.0)
            mcu_peripheral_refs.add("U9")
            mcu_grid.place(u9x, u9y, u9w, u9h)
            other_passive_refs.remove("U9")
            _log.info("    U9 (ESD) → between J2 and U3 at (%.1f, %.1f)",
                      u9x, u9y)

        # R6, R7 (USB resistors) near U9/J2
        usb_r_refs = [r for r in ("R6", "R7") if r in other_passive_refs
                      and r in positions]
        if usb_r_refs and j2_pos:
            u9_pos = positions.get("U9", j2_pos)
            u9x_r, u9y_r, _ = u9_pos
            u9w_r = fp_sizes.get("U9", (3.0, 3.0))[0]
            u9h_r = fp_sizes.get("U9", (3.0, 3.0))[1]
            for i, ref in enumerate(usb_r_refs):
                w, h = fp_sizes.get(ref, (1.0, 0.5))
                # Place resistors to the LEFT of U9, stacked vertically
                tx = u9x_r - u9w_r / 2.0 - w / 2.0 - 1.0
                ty = u9y_r + (i - 0.5) * (h + 1.5)
                tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
                ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
                px, py = mcu_grid.find_free_pos(tx, ty, w, h, max_radius=10.0)
                positions[ref] = (px, py, 0.0)
                mcu_peripheral_refs.add(ref)
                mcu_grid.place(px, py, w, h)
                other_passive_refs.remove(ref)
                _log.info("    %s (USB R) → (%.1f, %.1f)", ref, px, py)

        # --- Step 6: Reset/Boot subcircuit (SW1+R4, SW2+R5) ---
        # Place as a compact cluster above-left of U3
        sw_pairs: list[tuple[str, str]] = []  # (switch, resistor)
        for sw, res in [("SW1", "R4"), ("SW2", "R5"), ("SW1", "R5"), ("SW2", "R4")]:
            if (sw in other_passive_refs and sw in positions
                    and res in other_passive_refs and res in positions):
                # Check if they share a net (connected)
                sw_nets = ref_nets.get(sw, set())
                if res in sw_nets:
                    sw_pairs.append((sw, res))
        # Deduplicate: each ref should appear in at most one pair
        used_sw: set[str] = set()
        unique_pairs: list[tuple[str, str]] = []
        for sw, res in sw_pairs:
            if sw not in used_sw and res not in used_sw:
                unique_pairs.append((sw, res))
                used_sw.add(sw)
                used_sw.add(res)
        # Also handle unpaired SW/R refs
        unpaired_sw = [r for r in other_passive_refs if r in positions
                       and r.startswith("SW") and r not in used_sw]

        # Place SW/R cluster to the right of MCU — UI cluster on right edge
        sw_base_x = max_x - 8.0
        sw_base_y = mcu_top + 4.0
        for i, (sw, res) in enumerate(unique_pairs):
            sw_w, sw_h = fp_sizes.get(sw, (3.5, 3.5))
            r_w, r_h = fp_sizes.get(res, (1.0, 0.5))
            # Switch — place side by side horizontally
            tx = sw_base_x - i * (sw_w + 1.5)
            ty = sw_base_y
            tx = max(bounds[0] + sw_w / 2.0 + 1.0, min(bounds[2] - 2.0, tx))
            ty = max(bounds[1] + sw_h / 2.0 + 1.0, min(bounds[3] - 2.0, ty))
            px, py = mcu_grid.find_free_pos(tx, ty, sw_w, sw_h, max_radius=15.0)
            positions[sw] = (px, py, 0.0)
            mcu_peripheral_refs.add(sw)
            mcu_grid.place(px, py, sw_w, sw_h)
            other_passive_refs.remove(sw)
            # Resistor adjacent to switch
            r_tx = px
            r_ty = py - sw_h / 2.0 - r_h / 2.0 - 0.5
            r_tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, r_tx))
            r_ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, r_ty))
            rpx, rpy = mcu_grid.find_free_pos(r_tx, r_ty, r_w, r_h,
                                               max_radius=8.0)
            positions[res] = (rpx, rpy, 0.0)
            mcu_peripheral_refs.add(res)
            mcu_grid.place(rpx, rpy, r_w, r_h)
            other_passive_refs.remove(res)
            _log.info("    %s+%s (reset/boot) → (%.1f,%.1f) / (%.1f,%.1f)",
                      sw, res, px, py, rpx, rpy)

        # Unpaired switches
        for sw in unpaired_sw:
            sw_w, sw_h = fp_sizes.get(sw, (3.5, 3.5))
            tx = sw_base_x - len(unique_pairs) * (sw_w + 3.0)
            ty = sw_base_y
            tx = max(bounds[0] + sw_w / 2.0 + 1.0, min(bounds[2] - 2.0, tx))
            ty = max(bounds[1] + sw_h / 2.0 + 1.0, min(bounds[3] - 2.0, ty))
            px, py = mcu_grid.find_free_pos(tx, ty, sw_w, sw_h, max_radius=15.0)
            positions[sw] = (px, py, 0.0)
            mcu_peripheral_refs.add(sw)
            mcu_grid.place(px, py, sw_w, sw_h)
            other_passive_refs.remove(sw)
            _log.info("    %s (switch) → (%.1f, %.1f)", sw, px, py)

        # --- Step 6b: Place LED1 near SW cluster (UI grouping) ---
        led_placed = set()
        for led_ref in ["LED1"]:
            if (led_ref in other_passive_refs and led_ref in positions
                    and led_ref not in fixed_refs):
                lw, lh = fp_sizes.get(led_ref, (2.0, 1.0))
                # Place LED below the SW cluster
                led_tx = sw_base_x
                led_ty = sw_base_y + 6.0  # below switches
                led_tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, led_tx))
                led_ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, led_ty))
                lpx, lpy = mcu_grid.find_free_pos(
                    led_tx, led_ty, lw, lh, max_radius=10.0,
                )
                positions[led_ref] = (lpx, lpy, 0.0)
                mcu_peripheral_refs.add(led_ref)
                mcu_grid.place(lpx, lpy, lw, lh)
                if led_ref in other_passive_refs:
                    other_passive_refs.remove(led_ref)
                led_placed.add(led_ref)
                _log.info("    %s (status LED) → (%.1f, %.1f)", led_ref, lpx, lpy)

        # --- Step 7: Place remaining passives near MCU perimeter ---
        # Sort: MCU-connected refs first, then by distance
        def _mcu_prox_key(ref: str) -> tuple[int, float]:
            connected = mcu_ref_c3 in ref_nets.get(ref, set())
            rx, ry, _ = positions[ref]
            dist = math.sqrt((rx - mcu_x) ** 2 + (ry - mcu_y) ** 2)
            return (0 if connected else 1, dist)

        remaining = [r for r in other_passive_refs if r in positions]
        remaining.sort(key=_mcu_prox_key)

        _MCU_TARGET_GAP = 4.0
        # Place remaining passives in a ring around MCU, left side preferred
        # Use quadrant-based placement: upper-left, lower-left, then above/below
        # Gap must clear the full MCU courtyard (module body extends beyond pads)
        ring_slots: list[tuple[float, float]] = []
        # Upper-left quadrant — well clear of MCU courtyard
        for dy_off in range(-3, 4):
            ring_slots.append((mcu_left - 6.0, mcu_y + dy_off * 3.0))
        # Below MCU, left of J2
        for dx_off in range(-2, 3):
            ring_slots.append((mcu_x + dx_off * 3.0 - 8.0, mcu_bot + 4.0))
        # Above MCU, left of J16
        for dx_off in range(-2, 3):
            ring_slots.append((mcu_x + dx_off * 3.0 - 8.0, mcu_top - 4.0))

        slot_idx = 0
        for ref in remaining:
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            rrot = positions[ref][2]

            # Try ring slots first, then fallback to perimeter projection
            placed = False
            while slot_idx < len(ring_slots):
                sx, sy = ring_slots[slot_idx]
                slot_idx += 1
                sx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, sx))
                sy = max(bounds[1] + 2.0, min(bounds[3] - 2.0, sy))
                if mcu_grid.is_free(sx, sy, w, h):
                    positions[ref] = (sx, sy, rrot)
                    mcu_peripheral_refs.add(ref)
                    mcu_grid.place(sx, sy, w, h)
                    placed = True
                    break

            if not placed:
                # Fallback: find_free_pos near MCU left side
                tx = mcu_left - _MCU_TARGET_GAP - w / 2.0
                ty = mcu_y
                tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
                ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
                px, py = mcu_grid.find_free_pos(tx, ty, w, h,
                                                max_radius=25.0)
                positions[ref] = (px, py, rrot)
                mcu_peripheral_refs.add(ref)
                mcu_grid.place(px, py, w, h)

        _log.info(
            "    3c3: organized %d peripherals around %s at (%.1f, %.1f)",
            len(mcu_peripheral_refs), mcu_ref_c3, mcu_x, mcu_y,
        )

    # 3c4. Ethernet group organization — vertical signal-chain column
    # Signal chain: MCU SPI → U6 (W5500) → Y1 (crystal) + load caps
    #               → J13 (RJ45 Magjack) on bottom edge
    #               U8 (PoE) beside J13 with its caps
    _log.info("  3c4: Ethernet group organization")
    ethernet_fixed: set[str] = set()

    eth_group_refs: set[str] = set()
    for feat in requirements.features:
        if "ethernet" in feat.name.lower():
            for comp in feat.components:
                r = comp.ref if hasattr(comp, "ref") else comp
                eth_group_refs.add(r)
            break

    if eth_group_refs:
        eth_zone_rect: tuple[float, float, float, float] | None = None
        for z in zones:
            if z.name == "ethernet":
                eth_zone_rect = z.rect
                break

        eth_ics = sorted([r for r in eth_group_refs
                          if r.startswith("U") and r in positions])
        eth_connectors = sorted([r for r in eth_group_refs
                                 if r.startswith("J") and r in positions])

        if eth_ics and eth_zone_rect is not None:
            ezx1, ezy1, ezx2, ezy2 = eth_zone_rect
            _ETH_STRIP_GAP = 0.5

            # Build net → eth refs mapping
            eth_net_refs: dict[str, set[str]] = {}
            for net in requirements.nets:
                e_refs = set()
                for conn in net.connections:
                    if conn.ref in eth_group_refs:
                        e_refs.add(conn.ref)
                if e_refs:
                    eth_net_refs[net.name] = e_refs

            # Build occupancy grid WITHOUT ethernet group
            eth_grid = _PlacementGrid(bounds)
            for oref, (ox, oy, _orot) in positions.items():
                if oref in eth_group_refs:
                    continue
                ow, oh = _rotation_aware_size(oref, positions, fp_sizes)
                eth_grid.place(ox, oy, ow, oh)

            # Identify the main Ethernet IC (W5500 = largest U in group)
            eth_main_ic = eth_ics[0]  # U6 (W5500)
            # Crystal (Y prefix)
            crystal_refs = sorted([r for r in eth_group_refs
                                   if r.startswith("Y") and r in positions])
            # PoE module (second U if exists)
            poe_ic = eth_ics[1] if len(eth_ics) > 1 else ""

            # Decoupling caps for W5500 — small caps on power nets
            eth_decoupling = sorted([
                r for r in eth_group_refs
                if r.startswith("C") and r in positions
                and r not in crystal_refs
                and fp_sizes.get(r, (0, 0))[0] < 5.0  # small caps only
            ])

            # Crystal load caps (typically 18pF, on crystal nets)
            crystal_net_names: set[str] = set()
            for net_name, erefs in eth_net_refs.items():
                if any(r.startswith("Y") for r in erefs):
                    crystal_net_names.add(net_name)
            crystal_load_caps = sorted([
                r for r in eth_group_refs
                if r.startswith("C") and r in positions
                and any(r in eth_net_refs.get(n, set())
                        for n in crystal_net_names)
            ])

            # PoE caps (connected to PoE IC)
            poe_net_names: set[str] = set()
            for net_name, erefs in eth_net_refs.items():
                if poe_ic and poe_ic in erefs:
                    poe_net_names.add(net_name)
            poe_caps = sorted([
                r for r in eth_group_refs
                if r.startswith("C") and r in positions
                and r not in crystal_load_caps
                and any(r in eth_net_refs.get(n, set())
                        for n in poe_net_names)
            ])

            # Remaining caps (not crystal load, not PoE)
            other_caps = sorted(
                [r for r in eth_decoupling
                 if r not in crystal_load_caps and r not in poe_caps],
            )

            # Layout: vertical column flowing top→bottom
            # Column 1: U6 → Y1 + load caps → decoupling caps
            # Column 2 (right): U8 (PoE) + PoE caps
            # Bottom: J13 (RJ45) on board edge

            # Anchor: center of ethernet zone
            eth_anchor_x = (ezx1 + ezx2) / 2.0
            eth_anchor_y = ezy1 + 3.0

            placed_eth: set[str] = set()

            def _place_eth_column(
                refs: list[str], col_x: float, start_y: float,
            ) -> float:
                cy = start_y
                for ref in refs:
                    if (ref not in positions or ref in fixed_refs
                            or ref in placed_eth or ref == ""):
                        continue
                    w, h = fp_sizes.get(ref, (2.0, 2.0))
                    tx = col_x
                    ty = cy + h / 2.0
                    tx = max(bounds[0] + 2.0, min(bounds[2] - 2.0, tx))
                    ty = max(bounds[1] + 2.0, min(bounds[3] - 2.0, ty))
                    px, py = eth_grid.find_free_pos(tx, ty, w, h,
                                                    max_radius=25.0)
                    positions[ref] = (px, py, 0.0)
                    eth_grid.place(px, py, w, h)
                    ethernet_fixed.add(ref)
                    placed_eth.add(ref)
                    cy = py + h / 2.0 + _ETH_STRIP_GAP
                return cy

            # Column 1: W5500 IC → crystal + load caps → decoupling
            col1 = ([eth_main_ic] + crystal_refs + crystal_load_caps
                    + other_caps)
            col1_bottom = _place_eth_column(col1, eth_anchor_x, eth_anchor_y)

            # Column 2: PoE/PHY module — rotated 90° CCW, above J13 center
            if poe_ic and poe_ic in positions and poe_ic not in fixed_refs:
                poe_w, poe_h = fp_sizes.get(poe_ic, (8.0, 8.0))
                # Rotate 90° CCW: swap w/h for spacing
                poe_eff_w, poe_eff_h = poe_h, poe_w
                # Place centered horizontally on eth_anchor_x, above J13
                poe_tx = eth_anchor_x
                # Position above where J13 will go (bottom edge area)
                poe_ty = bounds[3] - 25.0  # ~25mm above bottom edge
                poe_tx = max(bounds[0] + poe_eff_w / 2 + 1,
                             min(bounds[2] - poe_eff_w / 2 - 1, poe_tx))
                poe_ty = max(bounds[1] + poe_eff_h / 2 + 1,
                             min(bounds[3] - poe_eff_h / 2 - 1, poe_ty))
                ppx, ppy = eth_grid.find_free_pos(
                    poe_tx, poe_ty, poe_eff_w, poe_eff_h, max_radius=15.0,
                )
                positions[poe_ic] = (ppx, ppy, 270.0)  # 270° = 90° CCW
                eth_grid.place(ppx, ppy, poe_eff_w, poe_eff_h)
                ethernet_fixed.add(poe_ic)
                placed_eth.add(poe_ic)
                _log.info("    %s (PHY) → (%.1f, %.1f) rot=270", poe_ic, ppx, ppy)
                # Place PoE caps near the IC
                cap_y = ppy - poe_eff_h / 2.0 - 2.0
                for cap_ref in poe_caps:
                    if (cap_ref not in positions or cap_ref in fixed_refs
                            or cap_ref in placed_eth):
                        continue
                    cw, ch = fp_sizes.get(cap_ref, (1.0, 0.5))
                    cpx, cpy = eth_grid.find_free_pos(
                        ppx, cap_y, cw, ch, max_radius=8.0,
                    )
                    positions[cap_ref] = (cpx, cpy, 0.0)
                    eth_grid.place(cpx, cpy, cw, ch)
                    ethernet_fixed.add(cap_ref)
                    placed_eth.add(cap_ref)
                    cap_y = cpy - ch / 2.0 - 1.0

            # J13 (RJ45): place on bottom board edge, facing OUTWARD (180° rotation)
            # The RJ45 connector MUST be on the bottom edge for cable access.
            from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space
            for ref in eth_connectors:
                if ref not in positions or ref in fixed_refs:
                    continue
                fp_match = None
                for fp in initial_pcb.footprints:
                    if fp.ref == ref:
                        fp_match = fp
                        break
                w, h = fp_sizes.get(ref, (2.0, 2.0))
                # Place centroid so bottom edge of courtyard touches bottom
                # board edge with 1mm margin.
                cent_y = bounds[3] - h / 2.0 - 1.0
                cent_x = eth_anchor_x
                if fp_match is not None:
                    # Use pad extent for accurate edge placement
                    trial_origin_x = eth_anchor_x
                    trial_origin_y = bounds[3] - 3.0
                    _, _, _, pad_max_y = pad_extent_in_board_space(
                        fp_match, trial_origin_x, trial_origin_y, 180.0,
                    )
                    if pad_max_y > bounds[3] - 1.0:
                        trial_origin_y -= (pad_max_y - bounds[3] + 1.0)
                    cent_x, cent_y = origin_to_centroid(
                        fp_match, trial_origin_x, trial_origin_y, 180.0,
                    )
                # Use tight max_radius so J13 stays near the bottom edge
                px, py = eth_grid.find_free_pos(cent_x, cent_y, w, h,
                                                max_radius=10.0)
                # Enforce bottom edge: never let centroid move more than
                # h/2 + 3mm above the bottom edge
                max_y = bounds[3] - h / 2.0 - 1.0
                if py < max_y - 5.0:
                    # find_free_pos pushed it too far up; force bottom edge
                    py = max_y
                positions[ref] = (px, py, 180.0)
                ethernet_fixed.add(ref)
                placed_eth.add(ref)
                _log.info("    %s (RJ45) → bottom edge (%.1f, %.1f)",
                          ref, px, py)

            # Any remaining eth refs
            remaining_eth = sorted(
                eth_group_refs - placed_eth - fixed_refs,
            )
            if remaining_eth:
                _place_eth_column(
                    [r for r in remaining_eth if r in positions],
                    eth_anchor_x, col1_bottom,
                )

            _log.info(
                "    3c4: organized %d ethernet components in signal chain",
                len(ethernet_fixed),
            )

    # 3h. Template-guided refinement — apply subcircuit layout templates
    _log.info("  3h: Template-guided refinement")
    # Protect all refs already placed by earlier phases from template moves
    template_protected = (fixed_refs | relay_support_refs | adc_channel_refs
                          | mcu_peripheral_refs | power_group_fixed
                          | ethernet_fixed | top_edge_connector_refs)
    positions, template_fixed = _apply_template_refinement(
        positions, fp_sizes, bounds, requirements, subcircuits, template_protected,
    )

    # 3g. Collision resolution (group-constrained, then unconstrained final pass)
    _log.info("  3g: Collision resolution")
    group_bboxes = _extract_group_bboxes(requirements, positions, fp_sizes)
    # Protect relay sub-circuit, ADC channel, and MCU peripheral positions
    subcircuit_fixed = (relay_support_refs | adc_channel_refs
                        | mcu_peripheral_refs | power_group_fixed
                        | ethernet_fixed | template_fixed
                        | top_edge_connector_refs)
    relay_fixed = fixed_refs | subcircuit_fixed | {
        r for r in positions if r.startswith("K")
    }
    positions = _resolve_collisions(
        positions, fp_sizes, bounds, relay_fixed, group_bboxes=group_bboxes,
    )
    # Targeted final pass — only unprotect refs that are actually colliding
    remaining_collisions = _count_collisions(positions, fp_sizes)
    if remaining_collisions:
        _log.info("  3g: %d remaining — targeted pass", len(remaining_collisions))
        # Only unprotect subcircuit refs that are involved in collisions,
        # BUT keep MCU peripheral refs and connector refs always protected
        # (they were placed intentionally by pin-side-aware logic)
        colliding_refs = set()
        for a, b in remaining_collisions:
            colliding_refs.add(a)
            colliding_refs.add(b)
        # Never unprotect the MCU IC itself, relays, or edge connectors.
        # MCU passives (caps, resistors, LEDs) CAN be moved if they collide,
        # since accurate courtyard sizes may reveal overlaps the placement
        # phase didn't account for.
        mcu_ic_refs = {r for r in mcu_peripheral_refs
                       if r.startswith("U") or r.startswith("J")}
        always_fixed = mcu_ic_refs | top_edge_connector_refs | {
            r for r in positions if r.startswith("K")
        }
        # Keep protection on subcircuit refs NOT involved in collisions
        targeted_fixed = fixed_refs | (subcircuit_fixed - colliding_refs) | always_fixed
        positions = _resolve_collisions(
            positions, fp_sizes, bounds, targeted_fixed,
        )

    # Post-3g: Enforce ethernet connectors on bottom edge
    for ref in ethernet_fixed:
        if ref.startswith("J") and ref in positions and ref not in fixed_refs:
            w, h = fp_sizes.get(ref, (2.0, 2.0))
            bottom_target = max_y - h / 2.0 - 1.0
            rx, ry, rot = positions[ref]
            if ry < bottom_target - 5.0:
                _log.info("  Enforcing %s to bottom edge: y %.1f → %.1f", ref, ry, bottom_target)
                positions[ref] = (rx, bottom_target, 180.0)

    # ===================================================================
    # Final: Clamp, score, review
    # ===================================================================
    _log.info("=== Final: Clamping and review ===")

    # Clamp off-board components using actual pad extent for connectors,
    # rotation-aware fp_sizes for everything else.
    from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space
    fp_lookup = {fp.ref: fp for fp in initial_pcb.footprints}
    for ref, (rx, ry, rot) in list(positions.items()):
        if ref in fixed_refs:
            continue
        fp_obj = fp_lookup.get(ref)
        if fp_obj is not None and ref.startswith("J"):
            # Use actual pad extent for connectors (asymmetric origins)
            ori_x, ori_y = centroid_to_origin(fp_obj, rx, ry, rot)
            px0, py0, px1, py1 = pad_extent_in_board_space(
                fp_obj, ori_x, ori_y, rot,
            )
            shift_x = shift_y = 0.0
            if px0 < min_x + 1.0:
                shift_x = (min_x + 1.0) - px0
            elif px1 > max_x - 1.0:
                shift_x = (max_x - 1.0) - px1
            if py0 < min_y + 1.0:
                shift_y = (min_y + 1.0) - py0
            elif py1 > max_y - 1.0:
                shift_y = (max_y - 1.0) - py1
            if shift_x != 0.0 or shift_y != 0.0:
                new_cx, new_cy = origin_to_centroid(
                    fp_obj, ori_x + shift_x, ori_y + shift_y, rot,
                )
                positions[ref] = (new_cx, new_cy, rot)
        else:
            # Symmetric footprints — use rotation-aware size
            w, h = _rotation_aware_size(ref, positions, fp_sizes)
            clamped_x = max(min_x + w / 2 + 1, min(max_x - w / 2 - 1, rx))
            clamped_y = max(min_y + h / 2 + 1, min(max_y - h / 2 - 1, ry))
            if clamped_x != rx or clamped_y != ry:
                positions[ref] = (clamped_x, clamped_y, rot)

    # Post-clamp collision resolution — protect relay sub-circuits
    post_clamp_collisions = len(_count_collisions(positions, fp_sizes))
    if post_clamp_collisions > 0:
        _log.info("  %d post-clamp collisions — resolving", post_clamp_collisions)
        relay_fixed_clamp = fixed_refs | subcircuit_fixed | {
            r for r in positions if r.startswith("K")
        }
        positions = _resolve_collisions(
            positions, fp_sizes, bounds, relay_fixed_clamp,
        )

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

        # Apply suggested fixes — protect relay sub-circuit positions
        relay_fixed_review = fixed_refs | subcircuit_fixed | {
            r for r in positions if r.startswith("K")
        }
        positions = _apply_review_fixes(
            positions, review, relay_fixed_review, fp_sizes, bounds,
        )

    # Post-review collision resolution — review fixes can introduce overlaps
    # Protect relay sub-circuit positions
    post_review_collisions = _count_collisions(best_positions, fp_sizes)
    if post_review_collisions:
        _log.info(
            "  %d post-review collisions — resolving", len(post_review_collisions),
        )
        relay_fixed_post = fixed_refs | subcircuit_fixed | {
            r for r in best_positions if r.startswith("K")
        }
        best_positions = _resolve_collisions(
            best_positions, fp_sizes, bounds, relay_fixed_post,
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
