# Visual Inspection & Iteration Process

## When to Use

Every time a board is regenerated with visual placement, follow this process.
Do NOT declare placement "done" until the loop exits cleanly.

## The Loop

```
1. Regenerate board → render PNGs to output/
2. Read the rendered image(s)
3. Inspect as a professional board designer and fabricator (see checklist below)
4. List every issue found
5. Fix the root cause in the placement engine code
6. Go to step 1
```

Exit condition: The visual inspection finds no major issues, OR the user says stop.

## Step 3: Visual Inspection Checklist

Look at the image and tell me what you think is wrong, if you were a professional
board designer and fabricator. Specifically check:

### Board Density & Size
- [ ] Is the board appropriately sized for the component count? (no massive empty areas)
- [ ] Could the board be significantly smaller without violating clearances?
- [ ] Are components packed with reasonable density (not too sparse, not too tight)?

### Group Cohesion
- [ ] Does each functional group form a tight, recognizable cluster?
- [ ] Are group members scattered or intermixed with other groups?
- [ ] Is each group's spread reasonable? (small groups <20mm, medium <35mm, large <50mm)

### Relay Layout (if applicable)
- [ ] Are relays in a tight 1xN row?
- [ ] Are driver components (Q, D, R) within 8mm of their relay?
- [ ] Are relay terminal blocks on the board edge nearest the relay bank?

### Signal Flow
- [ ] Is there a logical left-to-right or top-to-bottom flow? (power in → regulation → loads)
- [ ] Does the layout tell a story an EE can follow?

### Connectors
- [ ] Are ALL connectors within 3mm of a board edge?
- [ ] Are connectors oriented so cables route away from the board?
- [ ] Are functionally related connectors near their associated circuits?

### Voltage Domain Separation
- [ ] Are high-voltage (24V) and low-voltage (3.3V) components in separate regions?
- [ ] Is there visible clearance (>5mm) between voltage domains?
- [ ] Are domain colors (in domain render) cleanly separated, not intermixed?

### Decoupling & Critical Proximity
- [ ] Are decoupling caps within 3-5mm of their IC power pins?
- [ ] Is the crystal within 5-10mm of the MCU?
- [ ] Are pull-up resistors near their associated IC?

### Ratsnest Quality
- [ ] Are ratsnest lines (grey) short and mostly local?
- [ ] Are there long ratsnest lines crossing the entire board? (bad — means poor placement)
- [ ] Do ratsnest lines cross between groups excessively?

### Component Overlap
- [ ] Are any components visually overlapping?
- [ ] Are components from different groups intermixed in the same area?

### Professional Impression
- [ ] Would you send this board to fab as-is? Why or why not?
- [ ] What would a senior EE reviewer say about this layout?
- [ ] Does it look like a human designed it, or like an algorithm dumped parts on a board?

## Fixing Issues

When issues are found, fix the **root cause** in the placement engine, not symptoms:

| Symptom | Likely Root Cause | Fix Location |
|---------|------------------|--------------|
| Groups too spread out | `_layout_group()` row width too wide | `pcb/placement.py` |
| Groups intermixed | Zone rects overlap or group placer ignoring zones | `zone_partitioner.py`, `group_placer.py` |
| Connectors not on edge | `pin_connectors_to_edge()` threshold too high | `group_placer.py` |
| Empty space / board too big | Zone fractions don't match component density | `zone_partitioner.py` |
| Relays scattered | Relay row formation not working | `placement_optimizer.py` Level 3a |
| Long ratsnest | Groups placed far from connected groups | `zone_partitioner.py` zone layout |
| Decoupling caps far from IC | Level 3c tightening radius too large | `placement_optimizer.py` |
| No signal flow | Zone order doesn't follow power flow | `zone_partitioner.py` |

## Regeneration Command

```bash
# From kicad-ai-workflow project root:
pytest tests/integration/test_placement_visual.py -x --tb=short -s

# Or for direct render to output/:
python -c "
import sys, types
from pathlib import Path

NL_S3C_PATH = '/Users/johnbetts/Dropbox/Source/nl-s-3c-complete/build_with_pipeline.py'
build_path = Path(NL_S3C_PATH)
mod = types.ModuleType('build_nl_s3c')
mod.__file__ = str(build_path)
sys.modules['build_nl_s3c'] = mod
source = build_path.read_text()
cut = source.find('if __name__')
if cut > 0:
    source = source[:cut]
exec(compile(source, str(build_path), 'exec'), mod.__dict__)

from kicad_pipeline.models.requirements import (
    BoardContext, MechanicalConstraints, ProjectInfo, ProjectRequirements,
)
components = mod._make_components()
nets = mod._make_nets(components)
features = mod._make_features(components)
requirements = ProjectRequirements(
    project=ProjectInfo(name='NL-S-3C-Complete', author='Test', revision='v0.1', description='Integration test board'),
    features=tuple(features), components=tuple(components), nets=tuple(nets),
    mechanical=MechanicalConstraints(board_width_mm=140.0, board_height_mm=80.0),
    board_context=BoardContext(target_system='Test', shared_grounds=True, notes=()),
)

from kicad_pipeline.pcb.builder import build_pcb
from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee, _build_group_map, _fp_courtyard_sizes, _count_collisions
from kicad_pipeline.optimization.scoring import compute_fast_placement_score
from kicad_pipeline.visualization.placement_render import render_placement
from kicad_pipeline.optimization.functional_grouper import classify_voltage_domains

pcb = build_pcb(requirements, auto_route=False, placement_mode='grouped', layer_count=4, preserve_routing=False, skip_inner_zones=True)
pcb_opt, review = optimize_placement_ee(requirements, pcb)
score = compute_fast_placement_score(pcb_opt, requirements)
group_map = _build_group_map(requirements)
domain_map = classify_voltage_domains(requirements)

out = Path('output')
out.mkdir(exist_ok=True)
render_placement(pcb_opt, requirements, out / 'placement_groups.png', title='NL-S-3C v5 (Groups)', score=score, group_map=group_map)
render_placement(pcb_opt, requirements, out / 'placement_domains.png', title='NL-S-3C v5 (Domains)', score=score, domain_map=domain_map)

fp_sizes = _fp_courtyard_sizes(pcb_opt)
positions = {fp.ref: (fp.position.x, fp.position.y, fp.rotation) for fp in pcb_opt.footprints}
collisions = _count_collisions(positions, fp_sizes)
print(f'Score: {score.overall_score:.3f} ({score.grade})')
print(f'Collisions: {len(collisions)}')
print(f'Violations: {len(review.violations)} ({len([v for v in review.violations if v.severity == \"critical\"])} critical)')
"
```
