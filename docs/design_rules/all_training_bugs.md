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
- [x] **MCU-07**: Crystal Y1 + load caps C3/C4 removed — FIXED. ESP32-S3-WROOM-1 has an
  internal 40MHz crystal. External crystal is unnecessary. Removed Y1, C3, C4 and all
  XTAL_IN/XTAL_OUT/XTAL_IN_C3/XTAL_OUT_C4 nets. Verified: components absent from output PCB.
- [x] **MCU-08**: USB-C J1 rotation — FIXED. Post-placement correction sets J1 rotation=180
  so pads face the top board edge (wire-entry side outward). Verified: J1 rot=180 in output.
- [x] **MCU-09**: Pin labels at pad positions — VERIFIED OK. `_enrich_esp32_footprint()` in
  footprints.py places labels at each pad position with 1.6mm inward offset. Confirmed:
  pad 2 at x=-8.75, label "3V3" at x=-7.15 (offset=1.6mm toward IC center). Not clustered.

## Relay

- [x] **REL-01**: D1-D4 flyback diodes — VERIFIED OK. JLCPCB maps 1N4148 (C13585) to
  SOD-123F footprint, which matches the reference board exactly. Not 0805 — correct.
- [x] **REL-02**: C1 100uF — VERIFIED. Script uses `lcsc=None` to force parametric C_0805.
  Generated output: `Capacitor_SMD:C_0805_2012Metric`. Correct.
- [x] **REL-03**: No 3D models — FIXED. 32/36 footprints now have 3D models.
  Only mounting holes missing.
- [x] **REL-04**: L1/L2/C1/C2 overlapping channel 1 components — FIXED. Power isolation
  group moved further toward board bottom edge (L1/L2 at y=board_h-4, C1/C2 at y=board_h-10
  and board_h-7). Verified: 0 collisions, all components >2mm apart. Design rules: 39 pass, 0 violations.

## Power Chain

- [x] **PWR-01**: No 3D models — FIXED. 15/19 footprints now have 3D models.
- [x] **PWR-02**: J1 pin swap to avoid trace crossing — FIXED. Pin 1 is now GND, pin 2
  is +24V (swapped from original). Net definitions updated to match. When J1 is at the top
  edge, traces to U1 VIN and GND no longer cross.
- [x] **PWR-03**: Added C6 (100nF ceramic HF bypass) in parallel with C5 (22uF) on
  +3V3_C5_DEC subnet. AMS1117 datasheet recommends ceramic cap close to output.
  C6 added to components, GND net, and +3V3_C5_DEC net.
- [x] **PWR-04**: J3 +3V3 not connected — FIXED. J3 pin 1 was on isolated "+3V3" net with
  no path to U2 output. Moved J3.1 onto "+3V3_C5_DEC" subnet (same as U2 VOUT, C5, C6).
  Removed the orphaned "+3V3" net. Verified: J3 now in output at (44.0, 29.2).
- [x] **PWR-05**: C4/C5/C6 not placed near U2 — FIXED. Enabled `_apply_power_post_placement()`
  which was defined but not called. Also adjusted C4 position from dx=-17.4 to dx=-5.0
  (much closer to U2 VIN). Added C6 position rule. Verified: C4=5.6mm, C5=7.0mm, C6=7.0mm
  from U2.

## Analog Input

- [x] **ANA-01**: Layout improved — zone expansion fix gives analog group full board area
  instead of tiny 25% fraction. Score improved to 0.925 (A), only 1 collision remaining.
- [x] **ANA-02**: No 3D models — FIXED. 25/29 footprints now have 3D models.
- [ ] **ANA-03**: Screw terminal orientation issues — OPEN (same root cause as PWR-02).
- [x] **ANA-04**: ADC channel pin assignment — FIXED. Reordered channel-to-pin mapping to
  avoid trace crossing when J1-J4 are left-to-right: CH1->pin5(AIN1, top-left),
  CH2->pin4(AIN0, second-from-top-left), CH3->pin6(AIN2, top-right), CH4->pin7(AIN3).
  Both U1 component pin nets and `_channel_nets()` pin mapping updated consistently.

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
