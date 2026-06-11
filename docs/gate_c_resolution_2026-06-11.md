# Gate C Feedback Resolution — Training Board Batch (2026-06-11)

Status of every item in `gate_c_feedback_2026-06-11.md`, the first full
run of the mandatory gated loop, and the proposed next framework work.

## Item resolutions

**1. Terminal openings (WRONG direction)** — RESOLVED.
`scripts/calibrate_part_openings.py` renders isolated single-part
boards and a vision subagent measures the wire-entry side; measured
vectors live in `data/part_rules.json` with `calibrated: true`
(uncalibrated openings are a build warning; tests pin the values).
Measured: TerminalBlock **south [0,1]** — the asserted guess was
inverted, exactly as diagnosed; USB-C south; RJ45 north; ESP32 antenna
north.

**2. Relay NO/NC swap** — RESOLVED, but NOT by swapping.
The crossover was observed on a board whose terminals were rotated
180° by the wrong opening (item 1). On the rebuilt board the current
order (J1=NO, J2=COM, J3=NC) is crossing-free and the literal swap
would RE-introduce the crossing. The invariant is geometric, so the
feedback's own generalization shipped instead: `AttachBundle` IR +
`check_attach_bundles` Gate A rule — parts joined by ≥2 two-pin signal
nets (free-pin-order connectors only) must have crossing-free attach
lines, counted per build. It immediately caught a real instance on the
MCU board (J2 UART header order — fixed in requirements).
Schematic sync: every board now generates its schematic from the same
requirements and runs `schematic_pcb_sync` as ledger stage `sync`.

**3. Power chain direction + utilization** — direction RESOLVED via
item 1; utilization is OPEN (see "Next work").

**4. ESP32 body off-board / no 3D body; USB orientation** — RESOLVED.
The JLCPCB-cache ESP32 footprint carried a pad-extent courtyard that
omitted the antenna end; the module body hung 6mm off-board while pads
and courtyard passed. Fixes: full-body courtyard in
`_enrich_esp32_footprint`; Gate A `check_contain` extended to
courtyard/body (flush legal, past edge CRITICAL); lifted-connector
edge snap uses the calibrated opening (the USB-C courtyard-bulge proxy
pointed the wrong way).

**5. RJ45 backwards + displaced model** — RESOLVED.
Three defects: model anchored at (0,0,0) instead of the centered
pin-1; **KiCad model offset Y is in the 3D viewer frame (+Y = board
−Y)** — the board-frame Y must be negated (verified against the
official KiCad pairing render); courtyard was centroid-symmetric while
the official is mouth-asymmetric. Post-fix residual <0.5mm. Also
settled by ground-truth render: the RJHSE538X mouth faces NORTH and
its LED pads are REAR pins feeding front light pipes — the model is
NOT 180° rotated; opening [0,-1] stands.

**6. Mandatory dual-persona pipeline** — RESOLVED.
`placement_v2/persona_review.py` (review_fab / review_ee ledger stages,
strict JSON verdicts, CRITICAL+MAJOR block) +
`scripts/build_training_boards.py` driver: build → sync gate → Gate A →
4-view render → review prompts → verdict ingestion → `present` REFUSES
any board not all-green across certify/cells/floorplan/sync/gate_a/
review_fab/review_ee/gate_b.

**Standing reminder (hole-vs-terminal courtyards)** — RESOLVED:
mounting-hole collision check is courtyard-based (was pad-based
fp_sizes) and treats placed holes as obstacles.

## First full gated run (all five boards)

Deterministic stages: **all green** (certify, cells, floorplan, sync,
Gate A 0 violations on every board). Gate B vision: green on relay,
analog, mcu; power and ethernet each had findings. Dual-persona review:
**fab persona blocked all five boards** — the gates are doing their
job; nothing was presented as done.

Blocking findings by theme (cross-board):
- **Board utilization / dead space** (all 5, fab): top/half of each
  board empty while groups congest. Root cause named in the feedback
  itself: the floorplanner is greedy-feasible (HPWL only) with no
  optimality pressure; the board-level score exists but is unused.
- **Silkscreen overlaps** (all 5, fab MINOR + part of clearances):
  ref designators collide with pads/each other; no silkscreen
  placement pass runs for v2 boards.
- **Decoupling distance** (ee, power/ethernet/mcu): bulk/decoupling
  caps 8–10mm from their IC — cells satisfy the per-pin attach bound
  but group/floorplan placement separates rail caps from their loads.
- **Ethernet group coherence** (gate_b + ee): J3 stranded mid-board,
  SPI header far from PHY; ratsnest doubles back across U1.
- **Analog channel crossing** (ee): terminal AIN/GND pins cross to the
  divider strip — a 2-net crossing between DIFFERENT part pairs (R vs
  C/D), outside the current AttachBundle pair scope.
- **power_chain R1/R2 render with no 3D bodies** (gate_b CRITICAL):
  two R_0402 footprints show bare pads — model attachment gap to
  verify via the isolation pipeline.
- Reviewer noise to discount with evidence: the mcu "UART crossing"
  claim is contradicted by geometry (Gate A bundle check measures the
  segments as parallel; the reviewer paired pin NAMES, not nets).

## Proposed next framework work (in order)

1. **Floorplan utilization objective**: add dead-space/board-
   utilization pressure to `pack_board` (shrink-to-fit already exists
   for auto-sized boards; explicit-dim boards need group spreading or
   board shrink proposal). This clears the dominant fab blocker on all
   five boards.
2. **Silkscreen placement pass for v2**: reuse the existing silkscreen
   module post-placement; add a deterministic overlap check (text bbox
   vs pad bbox) so the finding converts per the standing rule.
3. **Rail-cap pulling**: extend group placement so power-rail caps
   (C2-class) attach to their load cluster, not the rail topology
   midpoint.
4. **Cross-pair crossing check**: generalize AttachBundle to count
   crossings between ALL attach lines sharing a connector, not only
   same-pair bundles (covers the analog AIN/GND channel crossing).
5. **R_0402 3D model**: verify via `verify_components.py` isolation
   pipeline; fix model attachment if absent.
