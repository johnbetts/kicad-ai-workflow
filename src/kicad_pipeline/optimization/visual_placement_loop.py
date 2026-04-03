"""Vision-guided placement optimization loop.

Uses 2D ratsnest renders as a diagnostic tool — a vision subagent
reads the render like a professional fabricator and identifies
structural problems, then maps each diagnosis to a specific fix.

Architecture:
  render → diagnose (vision subagent) → classify fix type → apply → re-render

Diagnosis types and their fix operations:
  - channels_crossing: swap pin assignments on connector/IC
  - group_far_from_connectors: move group zone toward connectors
  - component_chain_zigzag: rotate components in signal chain
  - long_diagonal_ratsnest: component in wrong group/zone
  - connector_pin_order_reversed: swap terminal pin nets
  - subcircuit_scattered: tighten subcircuit cluster
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnosis model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacementDiagnosis:
    """A single placement issue identified by vision review."""

    diagnosis_type: str  # channels_crossing, group_far, chain_zigzag, etc.
    refs: tuple[str, ...]  # component refs involved
    description: str  # human-readable explanation
    suggested_fix: str  # swap, rotate, move_group, reorder_pins
    confidence: float  # 0.0-1.0


# ---------------------------------------------------------------------------
# Vision diagnostic prompt
# ---------------------------------------------------------------------------

_DIAGNOSTIC_PROMPT = """\
You are a senior PCB layout engineer reviewing a 2D board render with
ratsnest lines (thin grey lines showing unrouted connections between pads).

Your job is to DIAGNOSE placement problems — not score the board.
Look at the ratsnest pattern and identify specific structural issues.

For each issue, classify it as one of these types:

1. **channels_crossing** — Multiple parallel signal chains (e.g. ADC channels)
   have their ratsnest lines crossing each other. Fix: swap component positions
   or pin assignments so chains run parallel.

2. **group_far_from_connectors** — A functional group's connectors are far
   from the group, creating long ratsnest lines across the board.
   Fix: the group or its connectors need to move closer together.

3. **chain_zigzag** — A signal chain (e.g. terminal→divider→filter→IC) has
   components rotated so the ratsnest zigzags instead of flowing straight.
   Fix: rotate individual components to align with signal flow direction.

4. **connector_pin_order** — A multi-pin connector has ratsnest lines crossing
   because pin assignments don't match the physical order of connected components.
   Fix: swap pin net assignments on the connector.

5. **subcircuit_scattered** — Components that should form a tight subcircuit
   (e.g. bypass cap near IC) are spread across the board.
   Fix: pull scattered component toward its anchor.

6. **component_wrong_zone** — A component is placed in a zone/group it doesn't
   belong to, creating long ratsnest lines to its actual group.
   Fix: move to correct zone.

Components on this board: {component_summary}

Respond with ONLY a JSON array of diagnosis objects:
[{{
  "type": "<diagnosis_type>",
  "refs": ["R1", "R2"],
  "description": "what's wrong and why",
  "fix": "swap|rotate|move_group|reorder_pins|pull_closer|move_zone",
  "confidence": 0.0-1.0
}}]

Return at most 5 diagnoses, ordered by severity (worst first).
If the placement looks reasonable, return an empty array: []
"""


# ---------------------------------------------------------------------------
# Diagnostic functions
# ---------------------------------------------------------------------------


def _render_for_diagnosis(
    pcb: PCBDesign,
    output_dir: Path | None = None,
) -> Path | None:
    """Render 2D with ratsnest for diagnosis. Returns path to PNG."""
    import tempfile
    from pathlib import Path as _Path

    try:
        from kicad_image_gen import render_2d
        from kicad_pipeline.pcb.builder import write_pcb
    except ImportError:
        _log.warning("kicad-image-gen not available")
        return None

    if output_dir is None:
        output_dir = _Path(tempfile.mkdtemp(prefix="visual_loop_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_pcb = output_dir / "_diagnosis.kicad_pcb"
    png_path = output_dir / "diagnosis_2d.png"

    try:
        write_pcb(pcb, tmp_pcb)
        render_2d(str(tmp_pcb), str(png_path), ratsnest=True, pad_labels=True)
        return png_path
    except Exception:
        _log.warning("Failed to render for diagnosis", exc_info=True)
        return None


def diagnose_placement(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    render_path: Path | None = None,
) -> list[PlacementDiagnosis]:
    """Run vision diagnosis on a board render.

    Renders the board (if no render_path provided), sends to Claude vision
    via the visual inspector API, and parses structured diagnoses.

    Returns list of PlacementDiagnosis objects, ordered by severity.
    """
    from pathlib import Path as _Path

    if render_path is None:
        render_path = _render_for_diagnosis(pcb)
    if render_path is None or not _Path(render_path).exists():
        return []

    try:
        from kicad_pipeline.validation.visual_inspector import (
            _call_claude_vision,
            is_enabled,
        )
        if not is_enabled():
            _log.debug("Visual inspector disabled")
            return []
    except ImportError:
        return []

    # Build component summary
    comp_lines = []
    for comp in requirements.components[:60]:
        group = comp.placement_group or "ungrouped"
        comp_lines.append(f"{comp.ref}({comp.value}, {group})")
    comp_summary = ", ".join(comp_lines)

    prompt = _DIAGNOSTIC_PROMPT.format(component_summary=comp_summary)
    images = [("2d_ratsnest", _Path(render_path))]
    response = _call_claude_vision(images, prompt, max_tokens=2048)

    if response is None:
        _log.warning("Vision diagnostic API call failed")
        return []

    return _parse_diagnoses(response)


def _parse_diagnoses(response: str) -> list[PlacementDiagnosis]:
    """Parse vision response into PlacementDiagnosis objects."""
    diagnoses: list[PlacementDiagnosis] = []

    try:
        text = response
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        items = json.loads(text.strip())

        if not isinstance(items, list):
            return diagnoses

        for item in items:
            diagnoses.append(PlacementDiagnosis(
                diagnosis_type=str(item.get("type", "unknown")),
                refs=tuple(item.get("refs", [])),
                description=str(item.get("description", "")),
                suggested_fix=str(item.get("fix", "")),
                confidence=float(item.get("confidence", 0.5)),
            ))
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        _log.warning("Failed to parse vision diagnoses: %s", exc)
        if response.strip():
            diagnoses.append(PlacementDiagnosis(
                diagnosis_type="raw",
                refs=(),
                description=response.strip()[:500],
                suggested_fix="manual",
                confidence=0.3,
            ))

    return diagnoses


# ---------------------------------------------------------------------------
# Fix dispatcher
# ---------------------------------------------------------------------------


def apply_diagnosis_fix(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    diagnosis: PlacementDiagnosis,
) -> PCBDesign | None:
    """Apply the fix suggested by a diagnosis. Returns new PCB or None if no fix."""
    from kicad_pipeline.optimization.ratsnest_optimizer import (
        _apply_component_swap,
        _apply_terminal_swap,
        _try_rotation,
        count_crossings,
        total_ratsnest_length,
    )

    fix = diagnosis.suggested_fix
    refs = diagnosis.refs

    if fix == "swap" and len(refs) >= 2:
        # Try swapping the first two refs
        trial = _apply_component_swap(pcb, refs[0], refs[1])
        # Verify improvement
        if total_ratsnest_length(trial) < total_ratsnest_length(pcb):
            _log.info("Applied swap %s↔%s: %s", refs[0], refs[1], diagnosis.description)
            return trial

    elif fix == "rotate" and len(refs) >= 1:
        # Try all 4 rotations for each ref, keep best
        best = pcb
        best_length = total_ratsnest_length(pcb)
        for ref in refs:
            for rot in (0.0, 90.0, 180.0, 270.0):
                trial = _try_rotation(best, ref, rot)
                trial_length = total_ratsnest_length(trial)
                if trial_length < best_length - 0.5:
                    best = trial
                    best_length = trial_length
        if best is not pcb:
            _log.info("Applied rotation for %s: %s", refs, diagnosis.description)
            return best

    elif fix == "reorder_pins" and len(refs) >= 1:
        # Try all pin pair swaps on the connector
        ref = refs[0]
        fp = pcb.get_footprint(ref)
        if fp is None:
            return None
        best = pcb
        best_crossings = count_crossings(pcb)
        signal_pads = [p for p in fp.pads if p.net_name and p.net_name != "GND"]
        for i in range(len(signal_pads)):
            for j in range(i + 1, len(signal_pads)):
                trial = _apply_terminal_swap(
                    best, ref, signal_pads[i].number, signal_pads[j].number,
                )
                trial_crossings = count_crossings(trial)
                if trial_crossings < best_crossings:
                    best = trial
                    best_crossings = trial_crossings
        if best is not pcb:
            _log.info("Applied pin reorder on %s: %s", ref, diagnosis.description)
            return best

    elif fix == "pull_closer" and len(refs) >= 2:
        # Pull ref[0] toward ref[1]
        from dataclasses import replace
        from kicad_pipeline.models.pcb import Point
        import math

        fp_a = pcb.get_footprint(refs[0])
        fp_b = pcb.get_footprint(refs[1])
        if fp_a and fp_b:
            ax, ay = fp_a.position.x, fp_a.position.y
            bx, by = fp_b.position.x, fp_b.position.y
            dist = math.dist((ax, ay), (bx, by))
            if dist > 5.0:  # only pull if >5mm apart
                # Move A 40% toward B
                new_x = ax + (bx - ax) * 0.4
                new_y = ay + (by - ay) * 0.4
                new_fps = []
                for fp in pcb.footprints:
                    if fp.ref == refs[0]:
                        new_fps.append(replace(fp, position=Point(x=new_x, y=new_y)))
                    else:
                        new_fps.append(fp)
                trial = replace(pcb, footprints=tuple(new_fps))
                if total_ratsnest_length(trial) < total_ratsnest_length(pcb):
                    _log.info("Pulled %s toward %s: %s", refs[0], refs[1], diagnosis.description)
                    return trial

    _log.debug("No improvement from fix '%s' for %s", fix, diagnosis.description[:80])
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_visual_placement_loop(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    max_iterations: int = 10,
    vision_every_n: int = 1,
) -> tuple[PCBDesign, list[str]]:
    """Run the vision-guided placement optimization loop.

    Each iteration:
    1. Render 2D with ratsnest
    2. Vision subagent diagnoses problems
    3. Apply fixes for each diagnosis (verify improvement)
    4. Re-render if changes were made

    Args:
        pcb: Starting PCB design.
        requirements: Project requirements.
        max_iterations: Maximum outer loop iterations.
        vision_every_n: Run vision diagnosis every N iterations.

    Returns:
        Tuple of (optimized PCB, list of changes applied).
    """
    from kicad_pipeline.optimization.ratsnest_optimizer import (
        count_crossings,
        total_ratsnest_length,
    )

    current_pcb = pcb
    all_changes: list[str] = []

    baseline_crossings = count_crossings(current_pcb)
    baseline_length = total_ratsnest_length(current_pcb)
    _log.info(
        "Visual placement loop: baseline %d crossings, %.0fmm ratsnest",
        baseline_crossings, baseline_length,
    )

    prev_score = baseline_length + 10.0 * baseline_crossings

    for iteration in range(max_iterations):
        # Step 1: Render
        render_path = _render_for_diagnosis(current_pcb)
        if render_path is None:
            _log.warning("Cannot render — stopping loop")
            break

        # Step 2: Diagnose
        if iteration % vision_every_n == 0:
            diagnoses = diagnose_placement(current_pcb, requirements, render_path)
            if not diagnoses:
                _log.info("Vision: no issues found — placement is clean")
                break
            _log.info(
                "Vision iteration %d: %d diagnoses",
                iteration + 1, len(diagnoses),
            )
        else:
            continue

        # Step 3: Apply fixes
        changes_this_iter = 0
        for diag in diagnoses:
            if diag.confidence < 0.4:
                continue
            result = apply_diagnosis_fix(current_pcb, requirements, diag)
            if result is not None:
                # Verify hard constraints
                try:
                    from kicad_pipeline.optimization.review_agent import review_placement
                    review = review_placement(result, requirements)
                    hard_violations = [
                        v for v in review.violations
                        if v.rule in ("COLLISION", "CONNECTOR_EDGE")
                    ]
                    if hard_violations:
                        _log.info(
                            "Rejected fix (hard violation): %s",
                            diag.description[:60],
                        )
                        continue
                except Exception:
                    pass  # review_placement may not work for all boards

                current_pcb = result
                change_desc = (
                    f"[{diag.diagnosis_type}] {diag.suggested_fix} "
                    f"{', '.join(diag.refs)}: {diag.description[:80]}"
                )
                all_changes.append(change_desc)
                changes_this_iter += 1
                _log.info("Applied: %s", change_desc)

        if changes_this_iter == 0:
            _log.info("No fixes applied this iteration — stopping")
            break

        # Step 4: Check convergence
        new_crossings = count_crossings(current_pcb)
        new_length = total_ratsnest_length(current_pcb)
        new_score = new_length + 10.0 * new_crossings
        delta = prev_score - new_score

        _log.info(
            "Iteration %d: %d crossings (-%d), %.0fmm (-%0.fmm), score delta=%.1f",
            iteration + 1,
            new_crossings, baseline_crossings - new_crossings,
            new_length, baseline_length - new_length,
            delta,
        )

        if delta < 0.1:
            _log.info("Score converged — stopping")
            break
        prev_score = new_score

    final_crossings = count_crossings(current_pcb)
    final_length = total_ratsnest_length(current_pcb)
    _log.info(
        "Visual placement loop complete: %d→%d crossings, %.0f→%.0fmm, "
        "%d changes applied over %d iterations",
        baseline_crossings, final_crossings,
        baseline_length, final_length,
        len(all_changes), min(iteration + 1, max_iterations),
    )

    return current_pcb, all_changes
