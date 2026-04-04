# Reference-Driven Placement Optimization Plan

## Status: READY TO IMPLEMENT (next session)

## Problem Statement

50 iterations of closed-loop metric optimization produced a Grade D board.
The optimizer achieved "0 collisions" while components scattered into an
unroutable mess. The metrics said "improving" while the board got worse.
Root cause: no visual gates, no reference comparison, proxy metrics diverged
from physical quality.

## Strategy: Reference-Seeded Placement

Instead of searching for good positions from scratch, extract positions from
the human-routed reference board and initialize the placer from them. The
optimization loop becomes drift-correction, not discovery.

### Reference Board

- Path: `/Users/johnbetts/Dropbox/Source/nl-s-3c-complete/nl-s-3c-complete.kicad_pcb`
- Size: 152x80mm, 129 components, 4-layer
- Human-routed, verified produceable
- Sub-boards also available in `output/training_reference_boards/`

### Reference Metrics (from docs/reference_placement_analysis.md)

| Metric | Reference Target |
|--------|-----------------|
| Group spread (small, <=10 refs) | 15-25mm |
| Group spread (medium, <=25 refs) | 25-40mm |
| Group spread (large, >25 refs) | 40-60mm |
| Subgroup spread (relay driver) | 5-8mm |
| Connector edge distance | <5mm (HARD) |
| Decoupling cap distance | 3-5mm from IC |
| Inter-group gap | 10-20mm |
| Board layout zones | Power top-center, Analog left, Relays right, MCU center-right, Ethernet bottom-center |

## Implementation Steps

### Step 1: Reference Position Extractor

```python
def extract_reference_positions(kicad_pcb_path: str) -> dict[str, tuple[float, float, float]]:
    """Extract ref -> (x, y, rotation) from a .kicad_pcb file."""
```

- Parse the reference .kicad_pcb S-expression
- Return dict mapping ref designator to (x, y, rotation)
- Handle footprint origin vs centroid conversion

### Step 2: Reference-Seeded Initialization

In `optimize_placement_ee()`:
- Add `reference_positions: dict[str, tuple[float, float, float]] | None = None`
- When provided, initialize every component at its reference position
- Skip L1 zone partitioning and L2 group placement (reference IS the placement)
- Run only: collision resolution, connector edge snapping, final cleanup

### Step 3: Per-Group Similarity Score

```python
def compare_to_reference(
    current: dict[str, tuple[float, float, float]],
    reference: dict[str, tuple[float, float, float]],
    group_map: dict[str, str],
) -> dict[str, float]:
    """Compute normalized mean position error per group (mm)."""
```

- For each group: mean Euclidean distance between current and reference positions
- Normalize: 0mm = perfect match, >20mm = failed
- Similarity % = max(0, 100 - (mean_error / 20) * 100)
- Overall similarity = weighted average across groups

### Step 4: Visual-Gated Iteration Loop

```
for iteration in range(max_iterations):
    # 1. Generate placement (reference-seeded or from previous iteration)
    pcb = build_and_optimize(requirements, reference_positions)
    
    # 2. Render 4 views
    render_2d(pcb, f"iter_{iteration}_2d.png")
    render_3d(pcb, f"iter_{iteration}_3d.png", view="iso")
    
    # 3. Compare to reference
    similarity = compare_to_reference(pcb.positions, reference, group_map)
    print(f"Iteration {iteration}: {similarity['overall']:.0f}% match")
    print(f"  Worst group: {worst_group} at {similarity[worst_group]:.0f}%")
    
    # 4. Visual inspection (subagent — MANDATORY)
    visual_report = inspect_renders_via_subagent(renders)
    
    # 5. Exit conditions
    if similarity['overall'] >= 80 and visual_report.grade >= 'C':
        if similarity['overall'] >= 90 or not improving:
            break  # good enough or plateaued
    
    # 6. Identify what to fix
    worst_group = min(similarity, key=similarity.get)
    
    # 7. Make framework code change for worst group
    # (This is where the actual pipeline improvement happens)
    fix_group_placement(worst_group, similarity, visual_report)
    
    # 8. Verify improvement visually before continuing
```

### Step 5: What "Fix Framework Code" Means

Each iteration identifies the worst-matching group and diagnoses WHY
the placer moved components away from reference positions. Fixes go in
the framework code (not post-hoc overrides):

- Zone sizing wrong? Fix `zone_partitioner.py`
- Group placed in wrong zone? Fix `_match_group_to_zone()`
- Subcircuit spacing wrong? Fix gap constants in `ee_phases.py`
- Connector not at edge? Fix edge pinning in `group_placer.py`
- Collision resolver scattered a group? Fix collision resolver boundaries

Each fix is a permanent pipeline improvement that helps ALL future boards.

## Anti-Patterns to Avoid

1. **NO blind metric loops** — every iteration gets a visual gate
2. **NO post-hoc position overrides** — fix the framework, not the output
3. **NO collision push-apart that scatters groups** — this is what broke the board
4. **NO declaring "done" without visual review** — the board owner decides, not metrics
5. **NO optimizing proxies** — reference similarity + visual quality are the only measures

## Revert Plan

Before starting, revert the aggressive push-apart code that scattered
components (commits 555b050 through bf2187b). Keep the structural
improvements from d13987d (zone sizing, auto-rotate, gap constants).
The collision guard and push-apart made metrics better but the board worse.

## Success Criteria

- Overall reference similarity >80% (weighted mean position error <4mm per group)
- Visual inspection grade >= C (no scattered groups, no unroutable ratsnest)
- 0 physical collisions (verified by independent checker, not self-reported)
- Board owner says "this is a reasonable starting point for routing"

## Session Estimate

- Step 1 (extractor): 30 minutes
- Step 2 (seeded init): 1 hour
- Step 3 (comparison): 30 minutes
- Step 4-8 (iteration loop): 2-4 hours depending on number of framework fixes needed
- Total: 1 session (4-6 hours)
