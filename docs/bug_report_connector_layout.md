# Bug Report: Screw Terminals (J1,J3-J6) Jammed Off-Board + J13 Off-Board

## Date: 2026-03-11

## Summary

Screw terminals J1, J3, J4, J5, J6 are crammed together in the top-left corner of the
board, partially extending off the board edge. They should be spaced evenly along the
top edge. J13 (RJ45) extends past the bottom board edge. MCU group has overlapping
components.

## Visual Evidence

From `output/placement_hifi.png`:
- J6, J5, J4, J3 are stacked/overlapping in top-left corner
- J1 (6-pin power harness) is also in the same cluster
- All five are partially off the left and top board edges
- J13 (RJ45 connector) extends below the bottom board edge
- MCU zone (U3 area) has component overlaps

## Root Cause

### 1. _orient_connectors() rotation logic
The `_orient_connectors()` function (placement_optimizer.py ~line 1337) determines which
edge a connector is nearest to and rotates it to face outward. However:
- It does not properly account for screw terminal widths at different rotations
- The "nearest edge" calculation pulls all screw terminals to the same corner
- Rotation-aware sizing (`eff_hw`, `eff_hh`) may be computing wrong dimensions

### 2. Phase 3f2 top-edge ordering
Phase 3f2 orders screw terminals left-to-right as [J6, J5, J4, J3, J1], but:
- Starting X position computation may not account for board left boundary
- Terminal gap (_TERM_GAP = 2.0mm) may be too small for 5mm+ pitch screw terminals
- The phase may not be executing at all if refs are in `fixed_refs` prematurely

### 3. J13 not clamped to board
J13 (RJ45) is placed by the ethernet phase (3c5) at the bottom edge but extends past
the board boundary. The final clamping step should prevent this but may not account for
the full footprint extent.

### 4. MCU overlaps
Collision resolution (phase 3g) is not resolving all overlaps in the MCU zone, possibly
because some components are in the `subcircuit_fixed` or `template_protected` sets.

## Fix Required

1. **Debug _orient_connectors()**: Print actual positions and rotations assigned to each
   connector. Verify edge detection and rotation logic with real footprint dimensions.
2. **Debug phase 3f2**: Print the starting X, gap, and resulting positions for each
   screw terminal. Verify they fit within board boundaries.
3. **J13 clamping**: Ensure rotation-aware bounding box is used for final edge clamping.
4. **MCU overlaps**: Audit which refs are in protection sets and whether collision
   resolution is being blocked.

## Process Failure

This bug persisted across multiple sessions because:
1. **Tests passed but board was not visually verified** — score thresholds don't catch
   layout issues like off-board components
2. **Code refactoring (centroid consolidation) was done without re-rendering** — the
   refactor was assumed to be behavior-preserving but was never visually confirmed
3. **CLAUDE.md mandates render→review→fix loop but it was skipped** — the mandatory
   human-in-the-loop verification was bypassed

## Severity: Critical

Components off-board = non-manufacturable board.
