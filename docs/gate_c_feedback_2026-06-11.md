# Gate C Feedback — Training Board Batch (2026-06-11)

Human review findings on the five v2 training boards, with root-cause
diagnosis of WHY each escaped the gates. Every item must become a
deterministic rule or a calibrated datum — per the standing rule.

## 1. Analog: screw terminals face the WRONG direction
**Why not caught**: `EdgePin.opening` for TerminalBlock was ASSERTED as
`[0,-1]` in part_rules.json (my guess from the reference board) — never
verified against ground truth. The face_out check then *confirmed the
guess*, not reality. A wrong calibration datum makes the check
actively misleading.
**Fix**: calibrate every part-rule `opening` the same way the relay 3D
model was calibrated — isolated single-part render, vision measurement
of which side the wire entry is on, write the measured vector back.
Add a `calibrated: true` flag; uncalibrated openings are a build
warning.

## 2. Relay: terminal pins 1 & 3 (NO/NC) nets should swap — avoidable
crossover trace
**Why not caught**: no rule relates terminal pin ORDER to the relay's
physical pad positions (NO is on one side, NC the other; the terminal
pin assignment must mirror them or traces cross).
**Fix**: swap NO/NC on J pins 1/3 in `train_relay_group.py` (Component
pins AND Net list; extend the pin/net-agreement regression test).
Generalize later: a ratsnest-crossing check between paired
connector/part pin rows (Gate A countable: segment intersections
between attach lines).
**Schematic sync**: requirements are the single source for both
schematic and PCB; training boards must ALSO generate the schematic
(they currently skip it) and the existing `schematic_pcb_sync` hard
gate must run per board so the schema can never drift from the PCB.

## 3. Power chain: terminals wrong direction + poor space utilization
**Why not caught**: same opening-calibration gap as #1, plus there is
NO optimality pressure — the floorplanner is greedy-feasible (HPWL
only) and no EE/fabricator persona review ran on the batch.
**Fix**: dual-persona review step (see #5); add board-utilization and
dead-space metrics to the floorplan objective (score exists but is
unused at board level).

## 4. MCU: J1 (USB) wrong orientation; U1 (ESP32) body HANGS OFF the
board while its antenna keepout is correctly placed; render shows pads
only (no module body)
**Why not caught — three separate gaps**:
a) Gate A `contain` checks PADS ONLY. The ESP32's castellated pads are
   in-board while the module BODY extends past the outline. Extend
   contain to courtyard/body extent (the IR already says "pad,
   courtyard, and 3D body").
b) The ESP32 parametric footprint's 3D model did not render (no body
   visible) — model path/offset broken => nothing for vision to flag
   as "hanging off". Needs the isolated-render calibration treatment
   (same as relay/RJ45).
c) USB J1 orientation: face_out relied on courtyard bulge; needs a
   calibrated `opening` for USB-C too.

## 5. Ethernet (from prior review): RJ45 backwards + model displaced
~8mm — needs `opening` rule + isolated-render model calibration.

## 6. Process change (MANDATORY)
Per-board pipeline becomes: build → Gate A → render → **dual-persona
review (EE + fabricator personas, structured verdicts)** → Gate B
vision checklist → ledger `all_green()` including BOTH review stages →
only then present to human. The batch loop must execute this for every
board with no manual skipping — encode as a driver script
(`scripts/build_training_boards.py`) that refuses to emit renders for
un-reviewed boards.

## Standing reminders
- Hole-vs-terminal courtyard collisions (power/analog corners) still
  open — hole placer uses fp_sizes, needs courtyard-based check.
- All five boards currently PASS Gate A — meaning Gate A is still
  blind to items 1–4. Close the gaps before re-presenting.
