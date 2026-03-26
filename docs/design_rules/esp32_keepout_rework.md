# ESP32 Antenna Keepout Rework Plan

## Current State (KI-022)

The antenna keepout for ESP32-S3-WROOM-1 is a simple rectangle that:
- Doesn't match the official WROOM-1 keepout shape (missing notch for castellated pads)
- Only extends 1.5mm beyond the module body (should be 3-4mm per datasheet)
- Crosses pad boundaries on pins 4-6 area
- Via stitching is only at the antenna boundary (needs full perimeter)
- `no_vias=True` blocks all vias in the keepout (via fence should be around it, not blocked)

## Required Changes

### 1. Keepout polygon shape
Replace the simple 4-corner rectangle with the official WROOM-1 keepout:
- Full module width at the antenna end
- Notched corners to avoid castellated edge pads
- Tapered at the pad-field boundary to not overlap signal pads

Reference: ESP32-S3-WROOM-1 datasheet Figure 3-4 "Recommended PCB Layout"

### 2. Extension beyond module
Change `_ESP32_ANTENNA_KEEPOUT_EXTENSION_MM` from 1.5 to 3.5mm.
The antenna radiates beyond the module body and copper in that zone
degrades RF performance.

### 3. Keepout rules
Change `no_vias=False` (allow GND stitching vias inside the keepout for
the ground plane, but no signal vias). Keep `no_copper=True` to block
copper pour and `no_tracks=True` to block signal traces.

### 4. Perimeter via stitching
Add GND via stitching along the ENTIRE module shield perimeter (all 4 sides),
not just the antenna boundary. This improves:
- Ground plane continuity under the module
- Thermal dissipation for the shield
- EMI containment

Spacing: 2mm between vias, 0.8mm from shield edge.

### 5. Pad 41 thermal via array
Add a 3x3 or 4x4 grid of GND vias under pad 41 (center thermal pad)
for thermal relief. These go through to the back copper pour.
Via specs: 0.3mm drill, 0.6mm pad, spaced 1.2mm apart.
Must be inside the pad 41 outline but respect annular ring rules.

## Files to Modify
- `src/kicad_pipeline/pcb/footprints.py`: `_esp32_enrich_antenna_keepout()`, constants
- `src/kicad_pipeline/pcb/zone_builder.py`: `make_rf_via_fence()` → add perimeter mode
- Tests: add regression test for keepout shape

## Priority
HIGH — affects RF performance on any ESP32 board.
