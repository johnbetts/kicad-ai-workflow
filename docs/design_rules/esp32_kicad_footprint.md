# TODO: Use KiCad Official ESP32-S3-WROOM-1 Footprint

## Rationale

The KiCad standard library footprint `RF_Module:ESP32-S3-WROOM-1` has correct:
- Keepout zone shape (matches datasheet recommended layout)
- Courtyard dimensions
- Pad positions and sizes
- 3D model with proper offset/rotation
- Pin numbering

Currently the pipeline uses JLCPCB cached footprints (easyeda2kicad) for ESP32,
which requires manual offset corrections, custom keepout generation, and enrichment
steps that are error-prone.

## Proposed Change

Add an allowlist of components where the KiCad standard library footprint is
preferred over the JLCPCB cache. The ESP32-S3-WROOM-1 would be the first entry.

Implementation: in `footprint_for_component()`, check if the component's footprint
ID matches a KiCad standard library entry. If so, load the `.kicad_mod` from the
KiCad installation path instead of the JLCPCB cache.

## Blockers

- Need to detect the KiCad installation path reliably across platforms
- Need fallback if KiCad is not installed (CI environments)
- Pin numbering must match between KiCad and JLCPCB conventions
