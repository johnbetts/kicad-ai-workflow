"""Ratsnest-guided placement refinement — second-order optimization.

After initial derivative placement, analyzes ratsnest crossings and
identifies component/pin swaps that reduce crossing count without
changing the schematic's electrical function.

Swap types:
- **Component swap**: Two same-value, same-footprint components exchange
  positions (e.g. R1↔R2 if both are 10k 0402).
- **Terminal swap**: Functionally interchangeable connector pins exchange
  net assignments (e.g. ADC_CH1↔ADC_CH2 on identical screw terminals).

The algorithm:
1. Build ratsnest from pad positions + net assignments
2. Count crossings (line segment intersection test)
3. Enumerate swap candidates (same value+footprint, or same connector)
4. For each candidate, compute crossing count if swapped
5. Apply greedy best-improvement swaps until no improvement found

This is a post-placement pass — it does NOT move components, it
reassigns nets or swaps component positions to reduce routing complexity.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint, PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ratsnest crossing computation
# ---------------------------------------------------------------------------


def _segments_intersect(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> bool:
    """Test if two line segments intersect (proper crossing, not touching)."""
    def _cross(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    d1 = _cross(bx1, by1, bx2, by2, ax1, ay1)
    d2 = _cross(bx1, by1, bx2, by2, ax2, ay2)
    d3 = _cross(ax1, ay1, ax2, ay2, bx1, by1)
    d4 = _cross(ax1, ay1, ax2, ay2, bx2, by2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _build_ratsnest_segments(
    pcb: PCBDesign,
) -> list[tuple[float, float, float, float, str]]:
    """Build MST ratsnest segments for all signal nets.

    Returns list of (x1, y1, x2, y2, net_name) segments.
    """
    from kicad_pipeline.visualization.ratsnest import (
        build_net_pad_map,
        minimum_spanning_tree,
    )

    net_pads = build_net_pad_map(pcb)
    segments: list[tuple[float, float, float, float, str]] = []

    for net_name, pads in net_pads.items():
        if len(pads) < 2:
            continue
        edges = minimum_spanning_tree(pads)
        for i, j in edges:
            x1, y1 = pads[i]
            x2, y2 = pads[j]
            segments.append((x1, y1, x2, y2, net_name))

    return segments


def count_crossings(pcb: PCBDesign) -> int:
    """Count the number of ratsnest segment crossings in a PCB."""
    segments = _build_ratsnest_segments(pcb)
    crossings = 0
    n = len(segments)
    for i in range(n):
        x1, y1, x2, y2, net_a = segments[i]
        for j in range(i + 1, n):
            x3, y3, x4, y4, net_b = segments[j]
            # Don't count crossings within the same net
            if net_a == net_b:
                continue
            if _segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
                crossings += 1
    return crossings


def total_ratsnest_length(pcb: PCBDesign) -> float:
    """Sum of all ratsnest segment lengths (mm). Lower = better placement."""
    segments = _build_ratsnest_segments(pcb)
    return sum(
        math.dist((x1, y1), (x2, y2))
        for x1, y1, x2, y2, _ in segments
    )


# ---------------------------------------------------------------------------
# Swap candidate discovery
# ---------------------------------------------------------------------------


def _find_component_swap_candidates(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[tuple[str, str]]:
    """Find pairs of components that can be position-swapped.

    Two components are swappable if they have the same value AND same
    footprint AND are in the same functional group (placement_group).
    """
    from collections import defaultdict

    # Group components by (value, footprint, placement_group)
    # Components WITHOUT a placement_group are NOT swappable — swapping
    # across groups destroys zone layout even if it improves ratsnest.
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for comp in requirements.components:
        if not comp.placement_group:
            continue  # ungrouped components cannot be swapped
        key = (comp.value, comp.footprint, comp.placement_group)
        groups[key].append(comp.ref)

    candidates: list[tuple[str, str]] = []
    for key, refs in groups.items():
        if len(refs) < 2:
            continue
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                candidates.append((refs[i], refs[j]))

    _log.info("Found %d component swap candidates", len(candidates))
    return candidates


def _find_terminal_swap_candidates(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
) -> list[tuple[str, str, str, str]]:
    """Find connector pin pairs that can exchange net assignments.

    Returns list of (ref, pin_a, pin_b, reason) tuples.
    Targets: screw terminals, pin headers where pins serve identical
    electrical functions (e.g., multiple ADC input channels).
    """
    candidates: list[tuple[str, str, str, str]] = []

    # Find multi-pin connectors where pins have the same electrical type
    for comp in requirements.components:
        if not comp.ref.startswith(("J", "P")):
            continue
        if len(comp.pins) < 2:
            continue

        # Group pins by type — pins of same type are swap candidates
        from collections import defaultdict
        type_groups: dict[str, list[str]] = defaultdict(list)
        for pin in comp.pins:
            if pin.net and pin.pin_type:
                type_groups[pin.pin_type.value].append(pin.number)

        for pin_type, pin_nums in type_groups.items():
            if len(pin_nums) < 2:
                continue
            # Don't swap power/ground pins
            if pin_type in ("power_in", "power_out"):
                continue
            for i in range(len(pin_nums)):
                for j in range(i + 1, len(pin_nums)):
                    candidates.append((
                        comp.ref, pin_nums[i], pin_nums[j],
                        f"same type ({pin_type})",
                    ))

    _log.info("Found %d terminal swap candidates", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Swap application
# ---------------------------------------------------------------------------


def _apply_component_swap(
    pcb: PCBDesign,
    ref_a: str,
    ref_b: str,
) -> PCBDesign:
    """Swap the positions of two components on the PCB."""
    from kicad_pipeline.models.pcb import Point

    fp_a = pcb.get_footprint(ref_a)
    fp_b = pcb.get_footprint(ref_b)
    if fp_a is None or fp_b is None:
        return pcb

    new_footprints: list[Footprint] = []
    for fp in pcb.footprints:
        if fp.ref == ref_a:
            new_footprints.append(replace(
                fp, position=fp_b.position, rotation=fp_b.rotation,
            ))
        elif fp.ref == ref_b:
            new_footprints.append(replace(
                fp, position=fp_a.position, rotation=fp_a.rotation,
            ))
        else:
            new_footprints.append(fp)

    return replace(pcb, footprints=tuple(new_footprints))


def _apply_terminal_swap(
    pcb: PCBDesign,
    ref: str,
    pin_a: str,
    pin_b: str,
) -> PCBDesign:
    """Swap the net assignments of two pins on a connector."""
    fp = pcb.get_footprint(ref)
    if fp is None:
        return pcb

    # Find the two pads
    pad_a = next((p for p in fp.pads if p.number == pin_a), None)
    pad_b = next((p for p in fp.pads if p.number == pin_b), None)
    if pad_a is None or pad_b is None:
        return pcb

    # Swap their net assignments
    new_pads = []
    for p in fp.pads:
        if p.number == pin_a:
            new_pads.append(replace(
                p, net_number=pad_b.net_number, net_name=pad_b.net_name,
            ))
        elif p.number == pin_b:
            new_pads.append(replace(
                p, net_number=pad_a.net_number, net_name=pad_a.net_name,
            ))
        else:
            new_pads.append(p)

    new_fp = replace(fp, pads=tuple(new_pads))
    new_footprints = tuple(
        new_fp if f.ref == ref else f for f in pcb.footprints
    )
    return replace(pcb, footprints=new_footprints)


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------


def optimize_ratsnest(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    max_iterations: int = 50,
) -> tuple[PCBDesign, list[str]]:
    """Greedy ratsnest crossing minimization via component/pin swaps.

    Iteratively applies the single best swap that reduces crossing count
    the most, until no improvement is found or max_iterations reached.

    Args:
        pcb: Current PCB with placed components.
        requirements: Project requirements for swap candidate discovery.
        max_iterations: Maximum number of swaps to apply.

    Returns:
        Tuple of (optimized PCBDesign, list of swap descriptions applied).
    """
    current_pcb = pcb
    baseline_crossings = count_crossings(current_pcb)
    baseline_length = total_ratsnest_length(current_pcb)
    _log.info(
        "Ratsnest optimizer: baseline %d crossings, %.1fmm total length",
        baseline_crossings, baseline_length,
    )

    if baseline_crossings == 0:
        _log.info("No ratsnest crossings — nothing to optimize")
        return current_pcb, []

    # Discover swap candidates once
    comp_candidates = _find_component_swap_candidates(current_pcb, requirements)
    term_candidates = _find_terminal_swap_candidates(current_pcb, requirements)

    applied_swaps: list[str] = []

    for iteration in range(max_iterations):
        current_crossings = count_crossings(current_pcb)
        best_improvement = 0
        best_pcb: PCBDesign | None = None
        best_desc = ""

        # Try all component swaps
        for ref_a, ref_b in comp_candidates:
            trial = _apply_component_swap(current_pcb, ref_a, ref_b)
            trial_crossings = count_crossings(trial)
            improvement = current_crossings - trial_crossings
            if improvement > best_improvement:
                best_improvement = improvement
                best_pcb = trial
                best_desc = f"swap {ref_a}↔{ref_b} (-{improvement} crossings)"

        # Try all terminal swaps
        for ref, pin_a, pin_b, reason in term_candidates:
            trial = _apply_terminal_swap(current_pcb, ref, pin_a, pin_b)
            trial_crossings = count_crossings(trial)
            improvement = current_crossings - trial_crossings
            if improvement > best_improvement:
                best_improvement = improvement
                best_pcb = trial
                best_desc = (
                    f"swap {ref} pin {pin_a}↔{pin_b} "
                    f"(-{improvement} crossings, {reason})"
                )

        if best_improvement <= 0 or best_pcb is None:
            _log.info(
                "Ratsnest optimizer: no improvement found after %d swaps",
                len(applied_swaps),
            )
            break

        current_pcb = best_pcb
        applied_swaps.append(best_desc)
        _log.info("Ratsnest optimizer [%d]: %s", iteration + 1, best_desc)

    final_crossings = count_crossings(current_pcb)
    final_length = total_ratsnest_length(current_pcb)
    _log.info(
        "Ratsnest optimizer complete: %d→%d crossings (-%d), "
        "%.1f→%.1fmm length (%.1f%%), %d swaps applied",
        baseline_crossings, final_crossings,
        baseline_crossings - final_crossings,
        baseline_length, final_length,
        (final_length - baseline_length) / baseline_length * 100
        if baseline_length > 0 else 0,
        len(applied_swaps),
    )

    return current_pcb, applied_swaps


# ---------------------------------------------------------------------------
# Chain rotation optimization
# ---------------------------------------------------------------------------


def _component_ratsnest_cost(
    pcb: PCBDesign,
    ref: str,
) -> float:
    """Sum of ratsnest segment lengths touching this component's pads."""
    from kicad_pipeline.visualization.ratsnest import (
        build_net_pad_map,
        rotate_point,
    )

    fp = pcb.get_footprint(ref)
    if fp is None:
        return 0.0

    # Get this component's pad positions in board space
    my_pads: set[str] = set()
    for pad in fp.pads:
        if pad.net_name:
            my_pads.add(pad.net_name)

    # Sum ratsnest lengths for nets this component participates in
    segments = _build_ratsnest_segments(pcb)
    cost = 0.0
    for x1, y1, x2, y2, net in segments:
        if net in my_pads:
            cost += math.dist((x1, y1), (x2, y2))
    return cost


def _try_rotation(
    pcb: PCBDesign,
    ref: str,
    new_rotation: float,
) -> PCBDesign:
    """Return a copy of PCB with one component rotated."""
    new_footprints = []
    for fp in pcb.footprints:
        if fp.ref == ref:
            new_footprints.append(replace(fp, rotation=new_rotation))
        else:
            new_footprints.append(fp)
    return replace(pcb, footprints=tuple(new_footprints))


def optimize_rotations(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    rotation_steps: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
) -> tuple[PCBDesign, list[str]]:
    """Optimize individual component rotations to minimize ratsnest length.

    For each non-fixed component, tries all rotation_steps and keeps the
    rotation that minimizes total ratsnest length through that component's
    nets. This is fast because it only recomputes ratsnest for the
    component being rotated, not the entire board.

    Args:
        pcb: Current PCB with placed components.
        requirements: Project requirements.
        rotation_steps: Rotations to try (degrees).

    Returns:
        Tuple of (optimized PCB, list of rotation changes applied).
    """
    current_pcb = pcb
    applied: list[str] = []

    # Skip fixed components (mounting holes, connectors already edge-pinned)
    skip_prefixes = ("H",)

    for fp in pcb.footprints:
        if fp.ref.startswith(skip_prefixes):
            continue
        if len(fp.pads) < 2:
            continue

        current_rot = fp.rotation
        best_rot = current_rot
        best_length = total_ratsnest_length(current_pcb)

        for rot in rotation_steps:
            if abs(rot - current_rot) < 1.0:
                continue  # skip current rotation
            trial = _try_rotation(current_pcb, fp.ref, rot)
            trial_length = total_ratsnest_length(trial)
            if trial_length < best_length - 0.5:  # 0.5mm min improvement
                best_length = trial_length
                best_rot = rot

        if abs(best_rot - current_rot) > 1.0:
            current_pcb = _try_rotation(current_pcb, fp.ref, best_rot)
            improvement = total_ratsnest_length(pcb) - best_length
            desc = f"rotate {fp.ref}: {current_rot:.0f}°→{best_rot:.0f}° ({improvement:.1f}mm shorter)"
            applied.append(desc)
            _log.info("Rotation optimizer: %s", desc)

    if applied:
        _log.info("Rotation optimizer: %d components rotated", len(applied))
    else:
        _log.info("Rotation optimizer: no beneficial rotations found")

    return current_pcb, applied


# ---------------------------------------------------------------------------
# Vision-based ratsnest review
# ---------------------------------------------------------------------------


def _render_ratsnest_2d(pcb: PCBDesign, output_path: str | None = None) -> str | None:
    """Render 2D view with ratsnest overlay. Returns path to PNG or None."""
    import tempfile
    from pathlib import Path as _Path

    try:
        from kicad_image_gen import render_2d
        from kicad_pipeline.pcb.builder import write_pcb
    except ImportError:
        _log.debug("kicad-image-gen not available for ratsnest render")
        return None

    if output_path is None:
        output_path = str(_Path(tempfile.mkdtemp()) / "ratsnest_2d.png")

    tmp_pcb = str(_Path(output_path).parent / "_ratsnest_verify.kicad_pcb")
    try:
        write_pcb(pcb, _Path(tmp_pcb))
        render_2d(tmp_pcb, output_path, ratsnest=True, pad_labels=True)
        return output_path
    except Exception:
        _log.warning("Failed to render ratsnest 2D", exc_info=True)
        return None


def vision_review_ratsnest(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    render_path: str | None = None,
) -> list[str]:
    """Use Claude vision to review a 2D ratsnest render for placement improvements.

    A subagent reads the rendered 2D board with ratsnest lines and pad labels,
    then suggests qualitative improvements a professional fabricator would make:
    - Groups that should be mirrored or rotated
    - Connectors facing the wrong direction
    - Obvious component clusters that should be repositioned
    - Signal flow improvements not captured by crossing count

    Args:
        pcb: Current PCB design.
        requirements: Project requirements for context.
        render_path: Optional path to pre-rendered 2D PNG. If None, renders one.

    Returns:
        List of suggestion strings from the vision review.
    """
    from pathlib import Path as _Path

    # Render if not provided
    if render_path is None:
        render_path = _render_ratsnest_2d(pcb)
    if render_path is None or not _Path(render_path).exists():
        _log.info("No ratsnest render available for vision review")
        return []

    # Check if vision is available
    try:
        from kicad_pipeline.validation.visual_inspector import (
            _call_claude_vision,
            is_enabled,
        )
        if not is_enabled():
            _log.debug("Visual inspector disabled — skipping vision ratsnest review")
            return []
    except ImportError:
        return []

    # Build component summary for context
    comp_summary = []
    for comp in requirements.components:
        group = comp.placement_group or "ungrouped"
        comp_summary.append(f"{comp.ref}({comp.value}, {group})")
    comp_text = ", ".join(comp_summary[:40])  # limit context size

    prompt = (
        "You are a senior PCB layout engineer reviewing a 2D board render "
        "with ratsnest lines (thin lines showing unrouted connections).\n\n"
        "Components on this board: " + comp_text + "\n\n"
        "The ratsnest lines show which pads need to be connected by copper traces. "
        "Long ratsnest lines or crossed lines indicate suboptimal placement.\n\n"
        "Look at the ratsnest pattern and suggest specific, actionable improvements "
        "a professional fabricator would make. Focus on:\n\n"
        "1. **Component swaps**: Same-value components that should swap positions "
        "to shorten or uncross ratsnest lines. Name both refs.\n"
        "2. **Group moves**: Functional groups that should shift, mirror, or rotate "
        "to improve signal flow.\n"
        "3. **Connector orientation**: Connectors that face the wrong direction "
        "relative to their ratsnest connections.\n"
        "4. **Channel reordering**: Multi-channel designs (ADC, relay drivers) "
        "where reordering channels would straighten ratsnest lines.\n\n"
        "Respond with ONLY a JSON array of suggestion objects:\n"
        '[{"type": "swap|move|orient|reorder", '
        '"refs": ["R1", "R2"], '
        '"description": "what to do and why", '
        '"estimated_improvement": "high|medium|low"}]\n\n'
        "If the placement looks good with minimal crossings, return an empty array: []"
    )

    images = [("2d_ratsnest", _Path(render_path))]
    response = _call_claude_vision(images, prompt, max_tokens=2048)

    if response is None:
        _log.warning("Vision ratsnest review: API call failed")
        return []

    # Parse suggestions
    suggestions: list[str] = []
    try:
        import json
        # Extract JSON from response (may have markdown wrapping)
        json_text = response
        if "```" in json_text:
            json_text = json_text.split("```")[1]
            if json_text.startswith("json"):
                json_text = json_text[4:]
        items = json.loads(json_text.strip())
        if isinstance(items, list):
            for item in items:
                desc = item.get("description", "")
                refs = item.get("refs", [])
                stype = item.get("type", "")
                est = item.get("estimated_improvement", "")
                suggestion = f"[{stype}] {', '.join(refs)}: {desc} ({est})"
                suggestions.append(suggestion)
                _log.info("Vision suggestion: %s", suggestion)
    except (json.JSONDecodeError, ValueError, KeyError):
        # Fall back to raw text
        if response.strip():
            suggestions.append(response.strip()[:500])
            _log.info("Vision review (raw): %s", response.strip()[:200])

    _log.info("Vision ratsnest review: %d suggestions", len(suggestions))
    return suggestions


def optimize_ratsnest_with_vision(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    max_iterations: int = 50,
) -> tuple[PCBDesign, list[str], list[str]]:
    """Full ratsnest optimization: swaps + rotations + vision review.

    1. Run geometric crossing minimization (component/pin swaps)
    2. Run per-component rotation optimization
    3. Render the result as 2D with ratsnest
    4. Run vision review for qualitative suggestions

    Args:
        pcb: Current PCB with placed components.
        requirements: Project requirements.
        max_iterations: Max geometric swap iterations.

    Returns:
        Tuple of (optimized PCB, applied changes, vision suggestions).
    """
    # Phase 1: Geometric swap optimization
    optimized_pcb, swaps = optimize_ratsnest(pcb, requirements, max_iterations)

    # Phase 2: Per-component rotation optimization
    optimized_pcb, rotations = optimize_rotations(optimized_pcb, requirements)
    all_changes = swaps + rotations

    # Phase 3: Render the optimized board
    render_path = _render_ratsnest_2d(optimized_pcb)

    # Phase 4: Vision review
    vision_suggestions = vision_review_ratsnest(
        optimized_pcb, requirements, render_path,
    )

    return optimized_pcb, all_changes, vision_suggestions
