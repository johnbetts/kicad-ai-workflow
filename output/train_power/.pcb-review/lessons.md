# PCB Review Lessons Learned

## Board-Specific Notes
- Board is 60x40mm with 15 components (buck converter + LDO + connectors + mounting holes)
- build_pcb() with preserve_from does NOT preserve mounting hole positions — it reassigns them to corners. Must match .kicad_pcb mounting hole positions to what build_pcb expects.
- SOIC-8 easyeda2kicad footprint (U1/TPS54331) has pre-existing pad clearance DRC errors — pads too close together. This is a footprint issue, not placement.
- Screw terminal (WJ128V) has large body (~10mm) — keep 10mm+ clearance from corner mounting holes

## What Works

## What Doesn't Work

## What Works
- Spreading components to eliminate courtyard collisions — increased offsets by 2-4mm
- Using review_placement() collision detection as hard gate before accepting placement
- Reducing board size to bring connectors closer (ethernet 55→40mm height)

## What Doesn't Work  
- Tight post-placement packing (<3mm from IC courtyard) always causes collisions
- Relying only on script-level design rules without review_placement() collision check
