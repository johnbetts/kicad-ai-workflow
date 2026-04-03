"""Board size and parameter sweep for placement optimization.

Runs the full placement pipeline at various board sizes and parameters,
measuring crossings, collisions, and contamination. Outputs a comparison table.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.disable(logging.INFO)  # Suppress verbose logs

# ---------------------------------------------------------------------------
# Load nl-s-3c build module once
# ---------------------------------------------------------------------------
_NL_S3C_PATH = Path("/Users/johnbetts/Dropbox/Source/nl-s-3c-complete/build_with_pipeline.py")


def _load_module() -> types.ModuleType:
    mod = types.ModuleType("build_nl_s3c")
    mod.__file__ = str(_NL_S3C_PATH)
    source = _NL_S3C_PATH.read_text()
    cut = source.find("if __name__")
    if cut > 0:
        source = source[:cut]
    exec(compile(source, str(_NL_S3C_PATH), "exec"), mod.__dict__)
    return mod


_MOD = _load_module()


def _make_requirements(
    board_w: float = 160.0, board_h: float = 80.0,
) -> object:
    from kicad_pipeline.models.requirements import (
        BoardContext,
        MechanicalConstraints,
        ProjectInfo,
        ProjectRequirements,
    )
    components = _MOD._make_components()
    nets = _MOD._make_nets(components)
    features = _MOD._make_features(components)
    return ProjectRequirements(
        project=ProjectInfo(name="NL-S-3C", author="T", revision="v0.1", description="t"),
        features=tuple(features),
        components=tuple(components),
        nets=tuple(nets),
        mechanical=MechanicalConstraints(board_width_mm=board_w, board_height_mm=board_h),
        board_context=BoardContext(target_system="T", shared_grounds=True, notes=()),
    )


@dataclass
class IterationResult:
    iteration: int
    board_w: float
    board_h: float
    crossings: int
    length_mm: float
    collisions: int
    cross_group: int
    zone_overflows: int
    elapsed_s: float
    notes: str = ""


def run_iteration(
    iteration: int,
    board_w: float = 160.0,
    board_h: float = 80.0,
    notes: str = "",
) -> IterationResult:
    """Run one full placement iteration and return metrics."""
    from kicad_pipeline.optimization.placement_optimizer import (
        _count_collisions,
        _fp_courtyard_sizes,
        optimize_placement_ee,
    )
    from kicad_pipeline.optimization.ratsnest_optimizer import (
        count_crossings,
        total_ratsnest_length,
    )
    from kicad_pipeline.pcb.builder import build_pcb
    from kicad_pipeline.pcb.pin_map import origin_to_centroid

    t0 = time.time()

    reqs = _make_requirements(board_w, board_h)
    pcb = build_pcb(
        reqs, auto_route=False, placement_mode="grouped",
        layer_count=4, preserve_routing=False, skip_inner_zones=True,
    )
    pcb_opt, review = optimize_placement_ee(reqs, pcb)

    crossings = count_crossings(pcb_opt)
    length = total_ratsnest_length(pcb_opt)

    fp_sizes = _fp_courtyard_sizes(pcb_opt)
    positions: dict[str, tuple[float, float, float]] = {}
    for fp in pcb_opt.footprints:
        cx, cy = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)
        positions[fp.ref] = (cx, cy, fp.rotation)
    collisions = _count_collisions(positions, fp_sizes)

    # Count cross-group contamination from review
    cross_group = 0
    for v in review.violations:
        if "cross" in str(v.rule).lower() or "zone" in str(v.rule).lower():
            cross_group += 1

    # Count zone overflow warnings (re-run partitioner to check)
    zone_overflows = 0
    try:
        from kicad_pipeline.optimization.zone_partitioner import partition_board
        from kicad_pipeline.optimization.functional_grouper import (
            compute_power_flow_topology,
            detect_subcircuits,
        )
        sc = detect_subcircuits(reqs)
        topo = compute_power_flow_topology(sc)
        margin = 5.0
        zones = partition_board(
            (margin, margin, board_w - margin, board_h - margin),
            list(reqs.features), topo, requirements=reqs,
        )
        for z in zones:
            x0, y0, x1, y1 = z.rect
            allocated = (x1 - x0) * (y1 - y0)
            # Check if any zone still overflows (simplified check)
            if allocated < 500:  # very small zone
                zone_overflows += 1
    except Exception:
        pass

    elapsed = time.time() - t0

    return IterationResult(
        iteration=iteration,
        board_w=board_w,
        board_h=board_h,
        crossings=crossings,
        length_mm=round(length),
        collisions=len(collisions),
        cross_group=cross_group,
        zone_overflows=zone_overflows,
        elapsed_s=round(elapsed, 1),
        notes=notes,
    )


def print_result(r: IterationResult) -> None:
    print(
        f"  #{r.iteration:>2d} | {r.board_w:.0f}x{r.board_h:.0f}mm | "
        f"crossings={r.crossings:>4d} | length={r.length_mm:>5.0f}mm | "
        f"collisions={r.collisions:>3d} | {r.elapsed_s:.1f}s | {r.notes}"
    )


if __name__ == "__main__":
    print("=== Board Size Sweep ===")
    print(f"{'#':>4} | {'Size':>10} | {'Cross':>7} | {'Length':>8} | {'Coll':>5} | {'Time':>5} | Notes")
    print("-" * 80)

    results = []
    sizes = [
        (160, 80, "baseline"),
        (170, 85, "+6%"),
        (180, 90, "+12%"),
        (190, 95, "+19%"),
        (200, 100, "+25%"),
        (180, 100, "wide"),
        (200, 90, "tall"),
    ]

    for i, (w, h, note) in enumerate(sizes, 1):
        r = run_iteration(i, w, h, note)
        print_result(r)
        results.append(r)

    # Save results
    out = Path("output/sweep_results.json")
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nResults saved to {out}")
