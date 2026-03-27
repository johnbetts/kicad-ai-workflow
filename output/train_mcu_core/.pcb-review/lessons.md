# PCB Review Lessons Learned

## Board-Specific Notes
- Board is 70x50mm, 14 active components + 4 mounting holes
- ESP32-S3-WROOM-1 (U1) pad extent is ~19x18.7mm — courtyard extends further
- Moving U1 to Y=37 (rotated 180) puts antenna within 6mm of south edge (Y=50)
- Decoupling caps need at least 5mm clearance from U1 pad extent edge
- R1/R2 pull-ups need ~3mm horizontal separation to avoid courtyard collision
- Single iteration D->A: spread C1/C2 right, R1/R2 left, U1 south toward edge

## What Works

## What Doesn't Work

## What Works
- Spreading components to eliminate courtyard collisions — increased offsets by 2-4mm
- Using review_placement() collision detection as hard gate before accepting placement
- Reducing board size to bring connectors closer (ethernet 55→40mm height)

## What Doesn't Work  
- Tight post-placement packing (<3mm from IC courtyard) always causes collisions
- Relying only on script-level design rules without review_placement() collision check
