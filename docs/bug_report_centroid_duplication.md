# Bug Report: Centroid Offset Logic Duplicated Instead of Using pin_map.py

## Date: 2026-03-11

## Summary

The `_centroid_offset()` function was reimplemented 3 times across the codebase instead
of using the `pin_map.py` module that was purpose-built for footprint pad analysis. This
led to the origin-vs-centroid bug where connectors (J14, J1, etc.) appeared off-board in
KiCad but correct in the matplotlib render.

## Root Cause

During Phase 1 of the "Pin-Aware Placement & Layout Templates" plan, `pin_map.py` was
created with `_footprint_pad_extent()` and pad position utilities. However, when the
centroid offset correction was later needed in placement_optimizer.py, review_agent.py,
and placement_render.py, the fix was implemented by **inlining the centroid computation**
in each file rather than adding a `compute_centroid_offset()` function to pin_map.py and
importing it.

## Affected Files (duplicate centroid logic)

| File | Function | Line | Should Use |
|------|----------|------|-----------|
| `placement_optimizer.py` | `_centroid_offset(fp)` | ~85 | `pin_map.compute_centroid_offset(fp)` |
| `review_agent.py` | `_fp_positions(pcb)` (inline) | ~135 | `pin_map.compute_centroid_offset(fp)` |
| `placement_render.py` | `render_placement()` (inline) | ~237 | `pin_map.compute_centroid_offset(fp)` |
| `scoring.py` | `_fp_position_dict(pcb)` (inline) | ~207 | `pin_map.compute_centroid_offset(fp)` |

## Why pin_map.py Was Not Used

1. `pin_map.py` focuses on **pad side classification** (CardinalSide), not centroid
   computation — it has `_footprint_pad_extent()` but no public `compute_centroid_offset()`
2. The centroid fix was made reactively (user reported off-board connectors) rather than
   planned, so the developer inlined the fix in each file independently
3. `scoring.py` already uses `pin_map` for `_score_pad_facing()` — proving the module
   IS accessible, the integration was just skipped for the centroid logic

## Fix Required

1. Add `compute_centroid_offset(footprint) -> tuple[float, float]` to `pin_map.py`
2. Replace all 4 inline centroid computations with the pin_map import
3. Single source of truth for origin ↔ centroid conversion

## Impact

- Connectors appeared off-board in KiCad PCB file but correct in matplotlib render
- Screw terminals had wrong positions (up to 16.5mm offset for 14-pin headers)
- Distance measurements in scoring and review were inconsistent between modules
- Any future fix to centroid logic must be applied in 4 places instead of 1

## Severity: High

This is a coordinate system correctness issue that affects every footprint with
non-centered origins (pin headers, screw terminals, THT connectors).

## Resolution (2026-03-11)

**Fixed.** Added three public functions to `pin_map.py`:
- `compute_centroid_offset(footprint) -> tuple[float, float]` — local coords offset
- `origin_to_centroid(footprint, ox, oy, rotation) -> tuple[float, float]` — converts KiCad origin to centroid
- `centroid_to_origin(footprint, cx, cy, rotation) -> tuple[float, float]` — inverse

All 4 inline implementations replaced with imports from `pin_map`:
- `placement_optimizer.py` — `_centroid_offset` aliased to `compute_centroid_offset`, `_extract_positions` and `_apply_positions` use `origin_to_centroid`/`centroid_to_origin`
- `review_agent.py` — `_fp_positions()` uses `origin_to_centroid`
- `scoring.py` — `_fp_position_dict()` uses `origin_to_centroid`
- `placement_render.py` — both inline centroid computations use `origin_to_centroid`

7 new tests added in `test_pin_map.py` covering offset computation, roundtrip conversion, and rotation handling.
