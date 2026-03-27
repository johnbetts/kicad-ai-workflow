# Session Learnings — 2026-03-26/27

## Context
Attempted to fix 3D model mismatches across 5 training boards. Discovered and fixed
multiple framework bugs, but also broke board layouts by changing package sizes without
the optimizer being robust enough to handle the change.

## Framework Improvements Made (KEEP)
1. **3D model STEP validation** (`footprints.py`): `_step_file_exists()` checks model
   paths against KiCad's actual library. Catches JLCPCB easyeda2kicad naming mismatches.
2. **JLCPCB pad count validation**: `_expected_pad_count()` rejects cached footprints
   with wrong pad count (e.g., SOD-323 2-pad for SOT-23 3-pad transistor).
3. **3D model fallback priority**: lib_id tried before footprint_id, with existence check.
4. **ESP32 model override**: Always uses correct `ESP32-S3-WROOM-1.step`.
5. **SOT-223 model name fix**: `SOT-223.step` (was `SOT-223-3_TabPin2.step`).
6. **PinHeader pitch in filename**: Adds `_P2.54mm` when missing.
7. **Human feedback scoring**: 15th scoring dimension with position locks, collision
   targets, and proximity constraints from `.pcb-review/human_constraints.json`.
8. **Framework assumptions registry**: `docs/framework_assumptions.md` with ~50 tracked entries.

## Key Bugs Found (FRAMEWORK — need fixing)

### BUG-1: Optimizer collision resolution ignores rotation-aware sizes
- **Impact**: SOIC-8 at -90° has 5.5x6.5mm physical extent (swapped from 6.5x5.5),
  but collision resolution uses unrotated dimensions. Components placed "clear" by
  unrotated math actually overlap when rotation is applied.
- **Where**: `src/kicad_pipeline/optimization/placement_optimizer.py` phase 3g
- **Test**: Generate power board, check U1(SOIC-8, -90°) vs L1 collision.

### BUG-2: Optimizer doesn't enforce minimum courtyard gap
- **Impact**: Components get placed with center-to-center distances that clear
  center-based checks but fail courtyard-based checks because courtyard extends
  beyond the center point by half the body size.
- **Where**: `_resolve_collisions()` in placement_optimizer.py
- **Fix**: All collision checks must use `(body_w/2 + body_w/2 + gap)` not just
  center distance.

### BUG-3: Package size change causes full layout regeneration with worse results
- **Impact**: Changing 0805→0402 in requirements should produce similar layout with
  tighter spacing. Instead the optimizer produces completely different (worse) layout
  because the initial placement seed changes with different footprint sizes.
- **Where**: Zone partitioning and group placement use footprint sizes to compute
  zone fractions and spacing. Smaller packages → different zone math → different layout.
- **Fix**: The optimizer should produce SIMILAR layouts for different package sizes of
  the same schematic. Package size is a physical detail, not a topology change.

### BUG-4: Subcircuit layout patterns are in training scripts, not framework
- **Impact**: Relay driver column layout, buck converter chain layout, ADC channel
  strip layout — all hand-coded in per-project training scripts. New projects don't
  get these patterns.
- **Where**: Should be in `placement_optimizer.py` subcircuit layout phases (3a-3f)
- **Fix**: Each `SubCircuitType` should have a layout strategy that produces the
  correct pattern regardless of which project uses it.

### BUG-5: Connector edge placement is a soft suggestion, not enforced
- **Impact**: Connectors end up 15-25mm from edges. The scorer penalizes this but
  the optimizer doesn't actively push connectors to edges.
- **Where**: `placement_optimizer.py` connector placement logic
- **Fix**: Connectors should be placed AT edges as a hard constraint, not nudged
  toward edges as a soft optimization.

### BUG-6: review_placement() Grade A doesn't correlate with visual quality
- **Impact**: Boards with scattered, ugly layouts get Grade A because no individual
  rule is violated. But the overall layout is clearly not production-ready.
- **Where**: `scoring.py` — missing dimensions for board utilization, signal flow
  coherence, layout compactness, visual organization.
- **Fix**: Add scoring dimensions: board_utilization (% of board area used by
  components), signal_flow_direction (are components ordered along signal path),
  layout_compactness (how tight is the clustering).

## Anti-Patterns Discovered

### 1. Agents editing .kicad_pcb directly
- **Problem**: Review-fix agents moved component coordinates in the PCB file,
  destroying production-ready layouts the optimizer had produced.
- **Rule**: NEVER edit .kicad_pcb coordinates. Fixes go in the optimizer or
  requirements. The pipeline regenerates from requirements → schematic → PCB.

### 2. Declaring "Grade A" without verifying renders
- **Problem**: Multiple rounds of agents reported Grade A but the boards were
  visually terrible. The scoring doesn't catch layout quality issues.
- **Rule**: Grade A from review_placement() is necessary but NOT sufficient.
  Fresh 3D renders must be generated and visually inspected every time.

### 3. Bulk changes to all boards at once
- **Problem**: Changed 5 training scripts simultaneously, couldn't isolate which
  change caused which regression.
- **Rule**: Change one board at a time. Verify before moving to next.

### 4. Training script post-placement is a framework bug
- **Problem**: Each training script has `_apply_*_post_placement()` that overrides
  the optimizer's output. This means the optimizer is insufficient.
- **Rule**: If post-placement is needed, the pattern should be in the optimizer.
  Training scripts should ONLY define requirements.

## What the Training Boards Are For
The 5 training boards exist to **test and improve the framework**. They are NOT
deliverables. The goal is: a new board with similar requirements should come out
correctly from `build_pcb()` without any per-project post-placement hacking.
Spending time on training board layouts is only valuable if it improves the framework.
