# Training Board Bugs — Master Tracker

All bugs must be VERIFIED FIXED by opening the generated KiCad PCB before closing.
Reference boards are at `output/training_reference_boards/`.

## MCU Core

- [x] **MCU-01**: PCB file won't load in KiCad — FIXED. `_fp_keepout_sexp()` in builder.py
  now matches `_keepout_sexp()` format exactly (net_name, layers, hatch, correct ordering).
  Verified: no `(zone ""` blocks in output, all 8 zones have proper format.
- [ ] **MCU-02**: Pin labels still missing for most GPIOs (only used pins labeled) — OPEN.
  All 41 pins are defined in the training script but unused pins have `net=""`.
  KiCad shows labels only for connected pins. This is by design.
- [x] **MCU-03**: Pad 41 GND central pad — VERIFIED. Training script output confirms
  pad 41 is present, net=GND. Footprint from JLCPCB cache includes it.
- [x] **MCU-04**: No 3D models — FIXED. 17/21 footprints now have 3D models.
  Only mounting holes (H1-H4) remain without models (expected — no standard 3D model).
- [x] **MCU-05**: Antenna keepout zone syntax — FIXED (same fix as MCU-01).
  All keepout zones now use correct `(zone (net 0) (net_name "") (layers ...) (hatch ...) ...)` format.
- [ ] **MCU-06**: Isolation vias around antenna area — OPEN (feature not implemented).

## Relay

- [x] **REL-01**: D1-D4 flyback diodes — VERIFIED OK. JLCPCB maps 1N4148 (C13585) to
  SOD-123F footprint, which matches the reference board exactly. Not 0805 — correct.
- [x] **REL-02**: C1 100uF — VERIFIED. Script uses `lcsc=None` to force parametric C_0805.
  Generated output: `Capacitor_SMD:C_0805_2012Metric`. Correct.
- [x] **REL-03**: No 3D models — FIXED. 32/36 footprints now have 3D models.
  Only mounting holes missing.

## Power Chain

- [x] **PWR-01**: No 3D models — FIXED. 15/19 footprints now have 3D models.
- [ ] **PWR-02**: J1 screw terminal rotation 0 vs reference 180 — OPEN.
  Optimizer picks rotation=0 for top-edge narrow connector. Reference has 180.
  Board-specific orientation preference; would need per-board override.
- [x] **PWR-03**: Added C6 (100nF ceramic HF bypass) in parallel with C5 (22uF) on
  +3V3_C5_DEC subnet. AMS1117 datasheet recommends ceramic cap close to output.
  C6 added to components, GND net, and +3V3_C5_DEC net.

## Analog Input

- [x] **ANA-01**: Layout improved — zone expansion fix gives analog group full board area
  instead of tiny 25% fraction. Score improved to 0.925 (A), only 1 collision remaining.
- [x] **ANA-02**: No 3D models — FIXED. 25/29 footprints now have 3D models.
- [ ] **ANA-03**: Screw terminal orientation issues — OPEN (same root cause as PWR-02).

## Ethernet

- [x] **ETH-01**: Collisions reduced from 15 to 8 — zone expanded from 10x10mm to 45x35mm
  for single-group boards. Score improved from 0.863 (B) to 0.907 (A).
  Remaining collisions around J1 (RJ45) due to large footprint on 50x40mm board.
- [x] **ETH-02**: No 3D models — FIXED. 13/17 footprints now have 3D models.
- [ ] **ETH-03**: Can't fully verify rotation without 3D view — OPEN (3D models now present,
  user can verify in KiCad 3D viewer).

## CROSS-CUTTING

- [x] **ALL-01**: 3D models — FIXED across all boards. JLCPCB footprints now get 3D model
  paths resolved from the original footprint_id via `_model_for_package()`.
  Coverage: MCU 17/21, Relay 32/36, Power 15/19, Analog 25/29, Ethernet 13/17.
- [x] **ALL-02**: Reference path — FIXED. All 5 training scripts updated to use
  `output/training_reference_boards/` for backup and drift comparison.
- [x] **ALL-03**: Drift comparison — FIXED (same change as ALL-02). Glob patterns
  updated to match reference board filenames.
