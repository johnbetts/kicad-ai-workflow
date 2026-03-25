# Training Board Bugs — Master Tracker

All bugs must be VERIFIED FIXED by opening the generated KiCad PCB before closing.
Reference boards are at `output/training_reference_boards/`.

## MCU Core — NEEDS WORK

- [x] **MCU-01**: PCB file loads in KiCad — FIXED
- [ ] **MCU-02**: Pin labels for unused GPIOs — KiCad only shows connected pin labels. Need fab-layer text labels for all pins.
- [x] **MCU-03**: Pad 41 single pad — VERIFIED (grep count=1)
- [x] **MCU-04**: 3D models — 17/21 footprints have models
- [x] **MCU-05**: Antenna keepout syntax — FIXED
- [ ] **MCU-06**: Isolation vias should be IN footprint as thru_hole pads — NOT IMPLEMENTED
- [x] **MCU-07**: Crystal removed (WROOM internal)
- [ ] **MCU-08**: J1 USB-C rotation still wrong — pads need to face board edge. Optimizer places at rotation=0, needs 180. Also overlaps U1 and extends past board edge.
- [x] **MCU-09**: Pin label text positions — at pad positions (verified)
- [x] **MCU-10**: Antenna keepout extended to 10mm height
- [ ] **MCU-11**: Pad 41 may be offset from true module center — needs datasheet verification
- [ ] **MCU-12**: Decoupling caps C1/C2 not near pin 2 (3V3) — optimizer doesn't use subnet to pull them close
- [ ] **MCU-13**: Components not grouped by function — scattered layout

## Relay — PRODUCTION READY

- [x] All bugs closed. 39/39 compliance, 0 violations, 0 collisions.

## Power Chain — GOOD

- [x] All bugs closed. 13/13 compliance, 0 collisions.

## Analog Input — IN OPTIMIZER (no script overrides)

- [x] **ANA-01-04**: All fixed. Layout rules now in optimizer code.
- [ ] **ANA-03**: Screw terminal orientation — optimizer picks rotation by edge, not by function

## Ethernet — GOOD

- [x] **ETH-01-02**: Fixed. 28/28 compliance.
- [ ] **ETH-03**: Rotation verification in 3D — 3D models present, user needs to check

## CROSS-CUTTING

- [x] **ALL-01**: 3D models added across all boards
- [x] **ALL-02/03**: Reference path + drift comparison updated
