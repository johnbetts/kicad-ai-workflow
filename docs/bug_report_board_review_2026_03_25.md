# Bug Report: Training Board Issues — 2026-03-25

Filed after detailed human + KiCad PCB editor review of all 5 training boards.
ALL are CORE pipeline bugs in kicad-ai-workflow (not kicad-image-gen).

## P0 — Electrical Correctness

### RELAY: Pin assignment regression
- **Bug**: Relay coil vs NO/NC pins are swapped. Common pin ratsnest ends at wrong pad. Board would not function.
- **PCB**: `/Users/johnbetts/Dropbox/Source/kicad-ai-workflow/output/train_relay/train_relay.kicad_pcb`
- **Root cause**: `src/kicad_pipeline/pcb/footprints.py` — `make_relay_spdt()` pin-to-pad mapping
- **Fix**: Verify against relay datasheet. Pin 1=COIL+, Pin 2=COM, Pin 3/4=NO/NC.

## P1 — Footprint/3D Generation (framework bugs)

### ALL BOARDS: 3D model positions offset from footprint pads
- **Bug**: SW1/SW2 on MCU, headers on power, screw terminals on analog — 3D bodies not aligned with pads
- **Root cause**: `src/kicad_pipeline/pcb/footprints.py` and `src/kicad_pipeline/pcb/builder.py` — 3D model offset/rotation in `(model ...)` sexp entries are wrong for JLCPCB cached footprints
- **Fix**: Audit all `_model_path_for_package()` assignments. Verify 3D model offset matches footprint origin for each package type.

### POWER: Missing 3D models on some components
- **Root cause**: `footprints.py` — model path lookup fails for some package types, or the path doesn't resolve on the user's system
- **Fix**: Add fallback 3D model paths. Log warnings for unresolved models.

### ANALOG: J1-J4 screw terminals rotated 180 degrees wrong
- **Bug**: Wire entry faces inward instead of toward board edge
- **Root cause**: `src/kicad_pipeline/optimization/ee_phases.py` — connector orientation phase doesn't know screw terminal wire-entry direction
- **Fix**: Orient screw terminal wire-entry toward nearest board edge

### MCU: Antenna keepout/via fence on wrong side of ESP32
- **Bug**: Isolation zone and vias appear on wrong end of module. Legacy keepout at bottom-left still present.
- **Root cause**: `src/kicad_pipeline/pcb/footprints.py` — `_enrich_esp32_footprint()` antenna side detection after board size change
- **Fix**: Verify keepout Y coordinates against pin numbering after rotation

### MCU: Decoupling caps on wrong side of IC
- **Bug**: C1/C2 left of U1 but 3V3 pin is on right side. Traces would route around entire module.
- **Root cause**: `src/kicad_pipeline/optimization/ee_phases_groups.py` — `_mcu_place_decoupling()` uses left pad edge, should use power pin side
- **Fix**: Detect which side has VCC/3V3 pin, place caps there

### ETHERNET: J1/U1 overlap, C4 inside J1 footprint
- **Root cause**: Ethernet group phase doesn't account for RJ45 through-hole footprint extent
- **Fix**: Use actual courtyard extents for gap calculation

### POWER: C6 (HF bypass) 20mm from U2 output pin
- **Root cause**: Power chain phase doesn't enforce bypass cap proximity
- **Fix**: Add C6 to LDO proximity constraints (placement_near="U2:VOUT", max 2mm)

### POWER: J1 at top-center instead of left edge
- **Root cause**: Connector edge assignment doesn't match signal flow direction
- **Fix**: Place input connector at left edge for left-to-right flow

## Filed to kicad-image-gen (rendering issue only)

### BUG-10: 2D render doesn't match KiCad editor for relay board
- **Filed at**: `/Users/johnbetts/Dropbox/Source/kicad-image-gen/BUGS.md` (BUG-10)
- This is the ONLY issue that belongs to kicad-image-gen. All 3D/footprint/rotation issues are pipeline bugs.
