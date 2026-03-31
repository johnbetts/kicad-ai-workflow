# Power Board 12 Issues — Visual Review Failures

These issues were visible in the 3D render but the scoring system gave Grade A.
Every one of these must be caught by the framework.

## Issues

1. **J1 terminal block hanging off board** — 40%+ of body extends past Edge.Cuts
2. **Mounting holes too large** for 35x22mm board — consume ~30% of usable area
3. **U2 SOT-223 body/rotation wrong** — massive black body, tab direction unclear
4. **U1 3D body looks small** relative to its pads — body-to-pad size mismatch
5. **C1 overlapping J1 area** — courtyard collision with terminal block
6. **D1 overlapping/touching U1** — no clearance between components
7. **R1/R2 floating orphans** — tiny 0402s with no visual grouping
8. **C6 isolated** — tiny component between stages, orphaned
9. **J2/J3 bodies may overlap mounting holes** at bottom corners
10. **Passives at inconsistent rotations** — no grid alignment
11. **Board too small** — 35mm can't fit terminal block + mounting holes + 2 IC stages
12. **3D bodies physically overlapping** — tight spacing = actual collisions scorer doesn't see
