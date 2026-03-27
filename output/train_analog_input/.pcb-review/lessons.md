# PCB Review Lessons Learned

## Board-Specific Notes

## What Works

## What Doesn't Work

## What Works
- Spreading components to eliminate courtyard collisions — increased offsets by 2-4mm
- Using review_placement() collision detection as hard gate before accepting placement
- Reducing board size to bring connectors closer (ethernet 55→40mm height)

## What Doesn't Work
- Tight post-placement packing (<3mm from IC courtyard) always causes collisions
- Relying only on script-level design rules without review_placement() collision check

## Key Findings
- KiCad DRC treats pad size (width, height) as GLOBAL coordinates, NOT local to the footprint rotation
- When a footprint is rotated -90, pad size (w, h) stays (w, h) in global frame — the position rotates but pad extents do NOT
- MSOP-10 original pad size (0.28 x 1.62) caused DRC shorts at -90 rotation because 1.62mm Y-extent at 0.5mm Y-pitch = massive overlap
- Fix: swap pad dimensions to (0.95, 0.25) so small dimension (0.25mm) aligns with the 0.5mm pitch direction in global Y after -90 rotation
- DRC went from 45 violations (17 errors including shorts) to 0 errors (11 silk warnings only)
