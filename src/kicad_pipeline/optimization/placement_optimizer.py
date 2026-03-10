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

if TYPE_CHECKING:
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


def _place_subcircuit_group(
    anchor_pos: tuple[float, float],
    refs: tuple[str, ...],
    anchor_ref: str,
    fp_sizes: dict[str, tuple[float, float]],
    board_bounds: tuple[float, float, float, float],
) -> dict[str, tuple[float, float]]:
    """Arrange sub-circuit components tightly around the anchor.

    Places components in a spiral pattern around the anchor position,
    respecting footprint sizes for clearance.

    Returns:
        Dict mapping ref → (x, y) position.
    """
    positions: dict[str, tuple[float, float]] = {anchor_ref: anchor_pos}
    min_x, min_y, max_x, max_y = board_bounds
    other_refs = [r for r in refs if r != anchor_ref]

    # Place in expanding ring around anchor
    angle_step = 2 * math.pi / max(len(other_refs), 1)
    radius = 3.0  # start 3mm from anchor

    for i, ref in enumerate(other_refs):
        w, h = fp_sizes.get(ref, (2.0, 2.0))
        aw, ah = fp_sizes.get(anchor_ref, (2.0, 2.0))
        min_dist = (max(w, h) + max(aw, ah)) / 2.0 + 0.5
        r = max(radius, min_dist)
        angle = angle_step * i
        x = anchor_pos[0] + r * math.cos(angle)
        y = anchor_pos[1] + r * math.sin(angle)
        # Clamp to board bounds with margin
        x = max(min_x + 2.0, min(max_x - 2.0, x))
        y = max(min_y + 2.0, min(max_y - 2.0, y))
        positions[ref] = (x, y)

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
) -> tuple[float, float]:
    """Assign a position within a zone for the index-th group."""
    x1, y1, x2, y2 = zone_rect
    zw = x2 - x1
    zh = y2 - y1
    cx = x1 + zw / 2.0
    cy = y1 + zh / 2.0

    if total <= 1:
        return (cx, cy)

    # Grid within zone
    cols = max(1, math.ceil(math.isqrt(total)))
    row = index // cols
    col = index % cols
    cell_w = zw / cols
    rows_needed = max(1, (total + cols - 1) // cols)
    cell_h = zh / rows_needed
    x = x1 + cell_w * (col + 0.5)
    y = y1 + cell_h * (row + 0.5)
    return (x, y)


def _apply_review_fixes(
    positions: dict[str, tuple[float, float, float]],
    review: PlacementReview,
    fixed_refs: set[str],
) -> dict[str, tuple[float, float, float]]:
    """Apply suggested position fixes from review violations.

    Only applies fixes for critical and major violations that have
    a suggested_position and affect movable components.
    """
    result = dict(positions)
    for violation in review.violations:
        if violation.severity == "minor":
            continue
        if violation.suggested_position is None:
            continue
        # Apply fix to the first ref in the violation that's movable
        for ref in violation.refs:
            if ref in fixed_refs or ref not in result:
                continue
            _x, _y, rot = result[ref]
            sx, sy = violation.suggested_position
            result[ref] = (sx, sy, rot)
            break  # one fix per violation

    return result


def optimize_placement_ee(
    requirements: ProjectRequirements,
    initial_pcb: PCBDesign,
    max_review_passes: int = 5,
) -> tuple[PCBDesign, PlacementReview]:
    """Run deterministic EE-grade placement optimization.

    5-phase algorithm:
    1. Functional grouping — detect sub-circuits and voltage domains.
    2. Zone assignment — divide board by voltage domain.
    3. Inter-group MST placement — minimize wire length between groups.
    4. Intra-group refinement — tight internal arrangement per sub-circuit.
    5. EE review loop — fix violations iteratively.

    Args:
        requirements: Project requirements with components and nets.
        initial_pcb: Starting PCB with initial placement.
        max_review_passes: Max iterations of the review-fix loop.

    Returns:
        Tuple of (optimized PCBDesign, final PlacementReview).
    """
    from kicad_pipeline.optimization.functional_grouper import (
        assign_zones,
        classify_voltage_domains,
        detect_subcircuits,
    )
    from kicad_pipeline.optimization.review_agent import review_placement
    from kicad_pipeline.optimization.scoring import _fp_size_dict

    fp_sizes = _fp_size_dict(initial_pcb)
    bounds = _board_bounds(initial_pcb)
    min_x, min_y, max_x, max_y = bounds
    board_w = max_x - min_x
    board_h = max_y - min_y

    # Build fixed refs set
    fixed_refs: set[str] = {
        fp.ref for fp in initial_pcb.footprints
        if _is_fixed(fp.ref, requirements)
    }

    # -----------------------------------------------------------------------
    # Phase 1: Functional Grouping
    # -----------------------------------------------------------------------
    _log.info("Phase 1: Detecting sub-circuits and voltage domains")
    subcircuits = detect_subcircuits(requirements)
    domain_map = classify_voltage_domains(requirements)

    # -----------------------------------------------------------------------
    # Phase 2: Zone Assignment
    # -----------------------------------------------------------------------
    _log.info("Phase 2: Assigning zones by voltage domain")
    all_refs = tuple(fp.ref for fp in initial_pcb.footprints)
    zone_assignments = assign_zones(
        subcircuits=subcircuits,
        domain_map=domain_map,
        board_width=board_w,
        board_height=board_h,
        all_refs=all_refs,
    )

    # -----------------------------------------------------------------------
    # Phase 3: Inter-Group MST Placement
    # -----------------------------------------------------------------------
    _log.info("Phase 3: Inter-group placement via MST")

    # Start with existing positions for fixed refs
    new_positions: dict[str, tuple[float, float, float]] = {}
    for fp in initial_pcb.footprints:
        if fp.ref in fixed_refs:
            new_positions[fp.ref] = (fp.position.x, fp.position.y, fp.rotation)

    # Place sub-circuit groups within their assigned zones
    placed_refs: set[str] = set(fixed_refs)
    subcircuit_positions: dict[str, tuple[float, float]] = {}

    for zone_assignment in zone_assignments:
        zone_rect = zone_assignment.zone_rect
        # Offset zone_rect by board origin
        adjusted_rect = (
            zone_rect[0] + min_x,
            zone_rect[1] + min_y,
            zone_rect[2] + min_x,
            zone_rect[3] + min_y,
        )

        # Place each sub-circuit group
        sc_list = list(zone_assignment.subcircuits)
        for sc_idx, sc in enumerate(sc_list):
            anchor_pos = _assign_zone_position(
                adjusted_rect, sc_idx, len(sc_list) + len(zone_assignment.loose_refs),
            )
            group_positions = _place_subcircuit_group(
                anchor_pos=anchor_pos,
                refs=sc.refs,
                anchor_ref=sc.anchor_ref,
                fp_sizes=fp_sizes,
                board_bounds=bounds,
            )
            for ref, (x, y) in group_positions.items():
                if ref in fixed_refs:
                    continue
                # Preserve existing rotation or default 0
                existing_rot = 0.0
                for fp in initial_pcb.footprints:
                    if fp.ref == ref:
                        existing_rot = fp.rotation
                        break
                new_positions[ref] = (x, y, existing_rot)
                placed_refs.add(ref)
                subcircuit_positions[ref] = (x, y)

        # Place loose refs
        loose_offset = len(sc_list)
        total_in_zone = len(sc_list) + len(zone_assignment.loose_refs)
        for lr_idx, ref in enumerate(zone_assignment.loose_refs):
            if ref in placed_refs:
                continue
            x, y = _assign_zone_position(
                adjusted_rect, loose_offset + lr_idx, total_in_zone,
            )
            existing_rot = 0.0
            for fp in initial_pcb.footprints:
                if fp.ref == ref:
                    existing_rot = fp.rotation
                    break
            new_positions[ref] = (x, y, existing_rot)
            placed_refs.add(ref)

    # Handle any refs not yet placed (shouldn't happen, but safety net)
    for fp in initial_pcb.footprints:
        if fp.ref not in placed_refs:
            new_positions[fp.ref] = (fp.position.x, fp.position.y, fp.rotation)

    # -----------------------------------------------------------------------
    # Phase 4: Intra-Group Refinement
    # -----------------------------------------------------------------------
    _log.info("Phase 4: Intra-group refinement")
    # For each subcircuit, pull connected components closer to anchor
    for sc in subcircuits:
        anchor = sc.anchor_ref
        if anchor not in new_positions:
            continue
        ax, ay, arot = new_positions[anchor]
        for ref in sc.refs:
            if ref == anchor or ref in fixed_refs or ref not in new_positions:
                continue
            rx, ry, rrot = new_positions[ref]
            # Pull toward anchor — move 50% closer
            nx = rx + (ax - rx) * 0.3
            ny = ry + (ay - ry) * 0.3
            # Clamp
            nx = max(min_x + 2.0, min(max_x - 2.0, nx))
            ny = max(min_y + 2.0, min(max_y - 2.0, ny))
            new_positions[ref] = (nx, ny, rrot)

    # -----------------------------------------------------------------------
    # Phase 5: EE Review Loop
    # -----------------------------------------------------------------------
    _log.info("Phase 5: Running EE review loop (max %d passes)", max_review_passes)
    best_positions = dict(new_positions)
    best_violation_count = float("inf")
    best_review: PlacementReview | None = None

    for pass_num in range(max_review_passes):
        positions_tuple = _dict_to_positions(new_positions)
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
            best_positions = dict(new_positions)
            best_review = review

        if critical_major == 0:
            _log.info("  No critical/major violations — stopping")
            break

        # Apply suggested fixes
        new_positions = _apply_review_fixes(new_positions, review, fixed_refs)

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
