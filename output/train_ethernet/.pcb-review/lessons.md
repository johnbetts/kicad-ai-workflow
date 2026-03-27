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
- Moving connectors too close to board edge with rotation — pad extent must be checked against board boundary
- J3 at (53.0, 38.0) collides with H3 mounting hole and extends off board; (52.0, 25.5) is safe
